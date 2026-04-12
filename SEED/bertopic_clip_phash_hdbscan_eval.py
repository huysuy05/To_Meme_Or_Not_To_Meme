#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from scipy.fftpack import dct
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, CLIPVisionModel

from seed_unsupervised_eval import (
    align_dataframe,
    build_reducer,
    collect_labeled_rows,
    compute_centroids,
    compute_metrics,
    drop_small_classes,
    filter_valid_image_rows,
    load_rgb_image,
    majority_vote_mapping,
    parse_int_list,
    pick_device,
    set_seed,
)
from meme_research_eval_utils import finalize_run_timing, load_split_rows, now_iso, summarize_batch_timings


@dataclass
class RunConfig:
    imgflip_root: Path | None
    train_parquet: Path | None = None
    test_parquet: Path | None = None
    output_dir: Path
    train_size: float = 0.80
    min_images_per_class: int = 7
    random_seed: int = 42
    batch_size: int = 64
    embedding_method: str = "clip"
    clip_model_id: str = "openai/clip-vit-base-patch32"
    reducer_dim: int = 128
    hdbscan_min_cluster_size: int = 10
    hdbscan_min_samples: int | None = None
    max_templates: int | None = None
    max_images_per_template: int | None = None
    seeds: tuple[int, ...] | None = None


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(
        description=(
            "BERTopic-style unsupervised template discovery baseline adapted to the local dataset: "
            "choose either CLIP or pHash embeddings, reduce them with UMAP, cluster with HDBSCAN, "
            "and assign template names by majority vote on the discovery split."
        )
    )
    parser.add_argument("--imgflip-root", default=None)
    parser.add_argument("--train-parquet", default=None)
    parser.add_argument("--test-parquet", default=None)
    parser.add_argument("--output-dir", default="SEED/uns_runs/bertopic_clip_phash_hdbscan_eval")
    parser.add_argument("--train-size", type=float, default=0.80)
    parser.add_argument("--min-images-per-class", type=int, default=7)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embedding-method", choices=["clip", "phash"], default="clip")
    parser.add_argument("--clip-model-id", default="openai/clip-vit-base-patch32")
    parser.add_argument("--reducer-dim", type=int, default=128)
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=10)
    parser.add_argument("--hdbscan-min-samples", type=int, default=5)
    parser.add_argument("--max-templates", type=int, default=None)
    parser.add_argument("--max-images-per-template", type=int, default=None)
    parser.add_argument("--seeds", type=parse_int_list, default=None)
    args = parser.parse_args()
    return RunConfig(
        imgflip_root=None if args.imgflip_root is None else Path(args.imgflip_root).expanduser().resolve(),
        train_parquet=None if args.train_parquet is None else Path(args.train_parquet).expanduser().resolve(),
        test_parquet=None if args.test_parquet is None else Path(args.test_parquet).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        train_size=args.train_size,
        min_images_per_class=args.min_images_per_class,
        random_seed=args.random_seed,
        batch_size=args.batch_size,
        embedding_method=args.embedding_method,
        clip_model_id=args.clip_model_id,
        reducer_dim=args.reducer_dim,
        hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
        hdbscan_min_samples=args.hdbscan_min_samples,
        max_templates=args.max_templates,
        max_images_per_template=args.max_images_per_template,
        seeds=args.seeds,
    )


def l2_normalize(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.clip(norms, 1e-12, None)


class ClipPhashEmbedder:
    def __init__(self, embedding_method: str, model_id: str, device: torch.device, batch_size: int):
        self.embedding_method = embedding_method
        self.device = device
        self.batch_size = batch_size
        self.processor = None
        self.model = None
        if self.embedding_method == "clip":
            self.processor = AutoImageProcessor.from_pretrained(model_id)
            self.model = CLIPVisionModel.from_pretrained(model_id).to(device).eval()

    @torch.inference_mode()
    def encode_paths(self, paths: list[str], desc: str) -> tuple[np.ndarray, list[str]]:
        clip_vectors: list[torch.Tensor] = []
        phash_vectors: list[np.ndarray] = []
        kept_paths: list[str] = []

        for start in tqdm(range(0, len(paths), self.batch_size), desc=desc):
            batch_paths = paths[start:start + self.batch_size]
            images: list[Image.Image] = []
            batch_phashes: list[np.ndarray] = []
            valid_paths: list[str] = []

            for path in batch_paths:
                image = load_rgb_image(path)
                if image is None:
                    continue
                images.append(image)
                valid_paths.append(path)
                if self.embedding_method == "phash":
                    batch_phashes.append(self.compute_phash_bits(image))

            if not valid_paths:
                continue

            if self.embedding_method == "clip":
                assert self.processor is not None and self.model is not None
                inputs = self.processor(images=images, return_tensors="pt")
                inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
                outputs = self.model(**inputs)
                pooled = outputs.pooler_output if outputs.pooler_output is not None else outputs.last_hidden_state[:, 0]
                features = F.normalize(pooled.float(), dim=-1)
                clip_vectors.append(features.cpu())
            else:
                phash_vectors.extend(batch_phashes)
            kept_paths.extend(valid_paths)

        if self.embedding_method == "clip":
            if not clip_vectors:
                raise ValueError(f"No valid images encoded for {desc}")
            return torch.cat(clip_vectors, dim=0).numpy(), kept_paths
        if not phash_vectors:
            raise ValueError(f"No valid images encoded for {desc}")
        return np.vstack(phash_vectors).astype(np.float32), kept_paths

    @staticmethod
    def compute_phash_bits(image: Image.Image) -> np.ndarray:
        return compute_phash_bits(image).astype(np.float32)


def compute_phash_bits(image: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> np.ndarray:
    img_size = hash_size * highfreq_factor
    gray = image.convert("L").resize((img_size, img_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.float32)
    dct_rows = dct(pixels, axis=0, norm="ortho")
    dct_2d = dct(dct_rows, axis=1, norm="ortho")
    low_freq = dct_2d[:hash_size, :hash_size]
    flat = low_freq.reshape(-1)
    median = np.median(flat[1:]) if flat.size > 1 else np.median(flat)
    return (low_freq > median).astype(np.uint8).reshape(-1)

def select_features(embedding_method: str, features: np.ndarray) -> np.ndarray:
    if embedding_method == "clip":
        return l2_normalize(features.astype(np.float32))
    if embedding_method == "phash":
        phash_centered = features.astype(np.float32) * 2.0 - 1.0
        return l2_normalize(phash_centered)
    raise ValueError(f"Unsupported embedding_method: {embedding_method}")


def fit_hdbscan(x: np.ndarray, min_cluster_size: int, min_samples: int | None):
    try:
        from sklearn.cluster import HDBSCAN as SklearnHDBSCAN

        model = SklearnHDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
        labels = model.fit_predict(x)
        return model, labels
    except Exception:
        try:
            import hdbscan  # type: ignore

            model = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                prediction_data=True,
            )
            labels = model.fit_predict(x)
            return model, labels
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "HDBSCAN is not installed. Install scikit-learn>=1.3 or hdbscan."
            ) from exc


def assign_to_centroids(x: np.ndarray, centroids: np.ndarray, centroid_ids: np.ndarray) -> np.ndarray:
    scores = x @ centroids.T
    best = scores.argmax(axis=1)
    return centroid_ids[best]


def run_once(cfg: RunConfig, run_dir: Path) -> dict[str, float | int | str]:
    set_seed(cfg.random_seed)
    run_started_at = now_iso()
    start_perf = time.perf_counter()
    device = pick_device()
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"device={device}")
    print(f"run_dir={run_dir}")

    if cfg.train_parquet is not None and cfg.test_parquet is not None:
        train_df, test_df = load_split_rows(str(cfg.train_parquet), str(cfg.test_parquet))
        imgflip_df = pd.concat([train_df, test_df], ignore_index=True)
        filtering_summary = {"precomputed_split": True}
    else:
        if cfg.imgflip_root is None:
            raise ValueError("Either --imgflip-root or both --train-parquet and --test-parquet must be provided.")
        imgflip_df = collect_labeled_rows(
            cfg.imgflip_root,
            max_templates=cfg.max_templates,
            max_images_per_template=cfg.max_images_per_template,
        )
        imgflip_df = filter_valid_image_rows(imgflip_df)
        imgflip_df, filtering_summary = drop_small_classes(imgflip_df, cfg.min_images_per_class)

        train_df, test_df = train_test_split(
            imgflip_df,
            train_size=cfg.train_size,
            stratify=imgflip_df["template"],
            random_state=cfg.random_seed,
        )
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        train_df["source"] = "imgflip_train"
        test_df["source"] = "imgflip_test"

    embedder = ClipPhashEmbedder(
        embedding_method=cfg.embedding_method,
        model_id=cfg.clip_model_id,
        device=device,
        batch_size=cfg.batch_size,
    )
    train_paths = train_df["image_path"].tolist()
    test_paths = test_df["image_path"].tolist()

    train_encoded, kept_train_paths = embedder.encode_paths(
        train_paths,
        desc=f"encode discovery {cfg.embedding_method}",
    )
    test_encoded, kept_test_paths = embedder.encode_paths(
        test_paths,
        desc=f"encode test {cfg.embedding_method}",
    )
    train_df = align_dataframe(train_df, kept_train_paths)
    test_df = align_dataframe(test_df, kept_test_paths)

    train_features = select_features(cfg.embedding_method, train_encoded)
    test_features = select_features(cfg.embedding_method, test_encoded)

    reducer = build_reducer("umap", cfg.reducer_dim, cfg.random_seed)
    train_reduced = reducer.fit_transform(train_features)
    test_reduced = reducer.transform(test_features)
    train_reduced = l2_normalize(np.asarray(train_reduced, dtype=np.float32))
    test_reduced = l2_normalize(np.asarray(test_reduced, dtype=np.float32))

    _, cluster_labels = fit_hdbscan(
        train_reduced,
        min_cluster_size=cfg.hdbscan_min_cluster_size,
        min_samples=cfg.hdbscan_min_samples,
    )
    centroids, centroid_ids = compute_centroids(train_reduced, cluster_labels)
    cluster_to_template = majority_vote_mapping(train_df, cluster_labels)
    test_cluster_ids = assign_to_centroids(test_reduced, centroids, centroid_ids)
    test_pred_templates = np.array([cluster_to_template.get(int(cid), "UNKNOWN_CLUSTER") for cid in test_cluster_ids])
    y_true = test_df["template"].to_numpy()
    metrics = compute_metrics(y_true, test_pred_templates)

    predictions_df = test_df.copy()
    predictions_df["pred_cluster"] = test_cluster_ids
    predictions_df["pred_template"] = test_pred_templates
    predictions_df["correct"] = predictions_df["pred_template"].to_numpy() == predictions_df["template"].to_numpy()
    predictions_df.to_csv(run_dir / "test_predictions.csv", index=False)

    metadata = {
        "config": {
            **asdict(cfg),
            "imgflip_root": None if cfg.imgflip_root is None else str(cfg.imgflip_root),
            "train_parquet": None if cfg.train_parquet is None else str(cfg.train_parquet),
            "test_parquet": None if cfg.test_parquet is None else str(cfg.test_parquet),
            "output_dir": str(cfg.output_dir),
            "device": str(device),
        },
        "dataset": {
            "imgflip_images_total_after_filtering": int(len(imgflip_df)),
            "imgflip_classes_after_filtering": int(imgflip_df["template"].nunique()),
            "imgflip_train_images": int(len(train_df)),
            "imgflip_test_images": int(len(test_df)),
            "small_class_filtering": filtering_summary,
        },
        "clustering": {
            "cluster_method": "hdbscan",
            "discovered_cluster_count": int(len(centroid_ids)),
            "mapped_cluster_count": int(len(cluster_to_template)),
            "unknown_cluster_predictions": int((test_pred_templates == "UNKNOWN_CLUSTER").sum()),
        },
        "test_metrics": metrics,
    }
    summary_row = {
        "run_dir": str(run_dir),
        "seed": int(cfg.random_seed),
        "embedding_method": cfg.embedding_method,
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "mcc": float(metrics["mcc"]),
        "cohen_kappa": float(metrics["cohen_kappa"]),
        "discovered_cluster_count": int(len(centroid_ids)),
        "mapped_cluster_count": int(len(cluster_to_template)),
    }
    timing = finalize_run_timing(metadata, summary_row, run_started_at, start_perf)
    (run_dir / "run_config.json").write_text(json.dumps(metadata, indent=2))
    (run_dir / "test_metrics.json").write_text(json.dumps({"test_metrics": metrics, "timing": timing}, indent=2))
    pd.DataFrame([summary_row]).to_csv(run_dir / "metrics_summary.csv", index=False)

    print(f"imgflip_train_images={len(train_df)} imgflip_test_images={len(test_df)}")
    print(f"discovered_cluster_count={len(centroid_ids)} mapped_cluster_count={len(cluster_to_template)}")
    print(f"test_accuracy={metrics['accuracy']:.4f}")
    print(f"test_f1={metrics['f1']:.4f}")
    print(f"test_precision={metrics['precision']:.4f}")
    print(f"test_recall={metrics['recall']:.4f}")
    print(f"test_mcc={metrics['mcc']:.4f}")
    print(f"test_cohen_kappa={metrics['cohen_kappa']:.4f}")
    print(f"duration_seconds={timing['duration_seconds']:.3f}")
    print(f"predictions_saved={run_dir / 'test_predictions.csv'}")
    return summary_row


def main() -> None:
    cfg = parse_args()
    seed_values = cfg.seeds if cfg.seeds is not None else (cfg.random_seed,)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if len(seed_values) == 1:
        single_cfg = RunConfig(**{**asdict(cfg), "random_seed": int(seed_values[0]), "seeds": None})
        run_once(single_cfg, cfg.output_dir / timestamp)
        return

    batch_dir = cfg.output_dir / f"{timestamp}_batch"
    batch_dir.mkdir(parents=True, exist_ok=True)
    batch_rows: list[dict[str, float | int | str]] = []
    for seed in seed_values:
        print(f"\n=== Running seed {seed} ===")
        seed_cfg = RunConfig(**{**asdict(cfg), "random_seed": int(seed), "seeds": None})
        batch_rows.append(run_once(seed_cfg, batch_dir / f"seed_{int(seed)}"))
    batch_timing = summarize_batch_timings(batch_rows)
    pd.DataFrame(batch_rows).to_csv(batch_dir / "batch_metrics_summary.csv", index=False)
    (batch_dir / "batch_config.json").write_text(
        json.dumps(
            {
                "output_dir": str(cfg.output_dir),
                "batch_dir": str(batch_dir),
                "seeds": [int(seed) for seed in seed_values],
                "timing": batch_timing,
            },
            indent=2,
        )
    )
    print(f"average_duration_seconds={batch_timing['average_duration_seconds']:.3f}")
    print(f"\nbatch_summary_saved={batch_dir / 'batch_metrics_summary.csv'}")


if __name__ == "__main__":
    main()
