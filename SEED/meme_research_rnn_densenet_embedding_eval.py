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
import torch.nn as nn
import torchvision.models as models
from sklearn.model_selection import train_test_split
from sklearn.neighbors import RadiusNeighborsClassifier
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from meme_research_eval_utils import (
    compute_metrics_with_rejection,
    drop_small_classes,
    finalize_run_timing,
    filter_valid_image_rows,
    load_dataset_rows,
    load_rgb_image,
    load_split_rows,
    now_iso,
    parse_int_list,
    pick_device,
    set_seed,
    stratified_split,
    summarize_batch_timings,
)


REJECT_TOKEN = "__OUTLIER__"
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


@dataclass
class RunConfig:
    dataset_root: Path | None
    parquet_path: Path | None
    train_parquet: Path | None
    test_parquet: Path | None
    image_root: Path | None
    output_dir: Path
    train_size: float = 0.80
    val_size: float = 0.10
    min_images_per_class: int = 7
    random_seed: int = 42
    batch_size: int = 64
    num_workers: int = 4
    radius_values: tuple[float, ...] = (20.0, 25.0, 30.0, 35.0, 40.0, 50.0)
    weights: tuple[str, ...] = ("uniform", "distance")
    max_templates: int | None = None
    max_images_per_template: int | None = None
    path_prefix_from: str | None = None
    path_prefix_to: str | None = None
    seeds: tuple[int, ...] | None = None


def parse_float_list(text: str) -> tuple[float, ...]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one float radius.")
    return tuple(float(value) for value in values)


def parse_str_list(text: str) -> tuple[str, ...]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one weight string.")
    return tuple(values)


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Closed-set ImgFlip evaluation for the meme-research rNN + DenseNet embedding baseline. "
            "Images are embedded with a frozen pretrained DenseNet-121 and classified with RadiusNeighborsClassifier."
        )
    )
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--parquet-path", default=None)
    parser.add_argument("--train-parquet", default=None)
    parser.add_argument("--test-parquet", default=None)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--output-dir", default="SEED/uns_runs/meme_research_rnn_densenet_embedding_eval")
    parser.add_argument("--train-size", type=float, default=0.80)
    parser.add_argument("--val-size", type=float, default=0.10)
    parser.add_argument("--min-images-per-class", type=int, default=7)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--radius-values", type=parse_float_list, default=(20.0, 25.0, 30.0, 35.0, 40.0, 50.0))
    parser.add_argument("--weights", type=parse_str_list, default=("uniform", "distance"))
    parser.add_argument("--max-templates", type=int, default=None)
    parser.add_argument("--max-images-per-template", type=int, default=None)
    parser.add_argument("--path-prefix-from", default=None)
    parser.add_argument("--path-prefix-to", default=None)
    parser.add_argument("--seeds", type=parse_int_list, default=None)
    args = parser.parse_args()
    return RunConfig(
        dataset_root=None if args.dataset_root is None else Path(args.dataset_root).expanduser().resolve(),
        parquet_path=None if args.parquet_path is None else Path(args.parquet_path).expanduser().resolve(),
        train_parquet=None if args.train_parquet is None else Path(args.train_parquet).expanduser().resolve(),
        test_parquet=None if args.test_parquet is None else Path(args.test_parquet).expanduser().resolve(),
        image_root=None if args.image_root is None else Path(args.image_root).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        train_size=args.train_size,
        val_size=args.val_size,
        min_images_per_class=args.min_images_per_class,
        random_seed=args.random_seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        radius_values=args.radius_values,
        weights=args.weights,
        max_templates=args.max_templates,
        max_images_per_template=args.max_images_per_template,
        path_prefix_from=args.path_prefix_from,
        path_prefix_to=args.path_prefix_to,
        seeds=args.seeds,
    )


class PathDataset(Dataset):
    def __init__(self, paths: list[str], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.paths[index]
        image = load_rgb_image(path)
        if image is None:
            raise ValueError(f"Unreadable image slipped through filtering: {path}")
        return self.transform(image), path


class DenseNetEmbedder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        base = models.densenet121(weights="DEFAULT")
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.relu(x, inplace=False)
        x = self.pool(x)
        return torch.flatten(x, 1)


def build_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ])


@torch.inference_mode()
def encode_paths(
    model: nn.Module,
    device: torch.device,
    paths: list[str],
    batch_size: int,
    num_workers: int,
    desc: str,
) -> np.ndarray:
    loader = DataLoader(
        PathDataset(paths, build_transform()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    vectors: list[np.ndarray] = []
    model.eval()
    for images, _ in tqdm(loader, desc=desc):
        images = images.to(device)
        features = model(images).cpu().numpy().astype(np.float32)
        vectors.append(features)
    return np.vstack(vectors)


def fit_and_predict(
    train_vectors: np.ndarray,
    train_labels: np.ndarray,
    test_vectors: np.ndarray,
    radius: float,
    weights: str,
) -> np.ndarray:
    clf = RadiusNeighborsClassifier(
        radius=radius,
        weights=weights,
        metric="euclidean",
        outlier_label=-1,
    )
    clf.fit(train_vectors, train_labels)
    return clf.predict(test_vectors)


def decode_predictions(preds: np.ndarray, idx_to_label: dict[int, str]) -> np.ndarray:
    return np.array(
        [idx_to_label[int(pred)] if int(pred) >= 0 else REJECT_TOKEN for pred in preds],
        dtype=object,
    )


def run_once(cfg: RunConfig, run_dir: Path) -> dict[str, float | int | str]:
    set_seed(cfg.random_seed)
    run_started_at = now_iso()
    start_perf = time.perf_counter()
    device = pick_device()
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}")
    print(f"run_dir={run_dir}")

    if cfg.train_parquet is not None and cfg.test_parquet is not None:
        train_df, test_df = load_split_rows(
            str(cfg.train_parquet),
            str(cfg.test_parquet),
            image_root=None if cfg.image_root is None else str(cfg.image_root),
            path_prefix_from=cfg.path_prefix_from,
            path_prefix_to=cfg.path_prefix_to,
        )
        df = pd.concat([train_df, test_df], ignore_index=True)
        filtering_summary = {"precomputed_split": True}
    else:
        df = load_dataset_rows(
            dataset_root=None if cfg.dataset_root is None else str(cfg.dataset_root),
            parquet_path=None if cfg.parquet_path is None else str(cfg.parquet_path),
            image_root=None if cfg.image_root is None else str(cfg.image_root),
            path_prefix_from=cfg.path_prefix_from,
            path_prefix_to=cfg.path_prefix_to,
            max_templates=cfg.max_templates,
            max_images_per_template=cfg.max_images_per_template,
        )
        df = filter_valid_image_rows(df)
        df, filtering_summary = drop_small_classes(df, cfg.min_images_per_class)
        train_df, test_df = stratified_split(df, train_size=cfg.train_size, random_seed=cfg.random_seed)
    tune_train_df, tune_val_df = train_test_split(
        train_df,
        test_size=cfg.val_size,
        stratify=train_df["template"],
        random_state=cfg.random_seed,
    )
    tune_train_df = tune_train_df.reset_index(drop=True)
    tune_val_df = tune_val_df.reset_index(drop=True)

    labels = sorted(train_df["template"].unique().tolist())
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    model = DenseNetEmbedder().to(device)

    tune_train_vectors = encode_paths(
        model, device, tune_train_df["image_path"].tolist(), cfg.batch_size, cfg.num_workers, "encode tune train"
    )
    tune_val_vectors = encode_paths(
        model, device, tune_val_df["image_path"].tolist(), cfg.batch_size, cfg.num_workers, "encode tune val"
    )
    train_vectors = encode_paths(
        model, device, train_df["image_path"].tolist(), cfg.batch_size, cfg.num_workers, "encode train"
    )
    test_vectors = encode_paths(
        model, device, test_df["image_path"].tolist(), cfg.batch_size, cfg.num_workers, "encode test"
    )

    tune_train_y = tune_train_df["template"].map(label_to_idx).to_numpy(dtype=np.int64)
    train_y = train_df["template"].map(label_to_idx).to_numpy(dtype=np.int64)

    search_rows: list[dict[str, float | str]] = []
    best_score = float("-inf")
    best_params: tuple[float, str] | None = None

    for radius in cfg.radius_values:
        for weights in cfg.weights:
            val_preds = fit_and_predict(tune_train_vectors, tune_train_y, tune_val_vectors, radius, weights)
            val_pred_labels = decode_predictions(val_preds, idx_to_label)
            metrics = compute_metrics_with_rejection(
                tune_val_df["template"].to_numpy(),
                val_pred_labels,
                REJECT_TOKEN,
            )
            row = {
                "radius": float(radius),
                "weights": str(weights),
                "val_f1": float(metrics["f1"]),
                "val_accuracy": float(metrics["accuracy"]),
                "val_coverage": float(metrics["coverage"]),
                "val_covered_accuracy": float(metrics["covered_accuracy"]),
            }
            search_rows.append(row)
            print(row)
            if float(metrics["f1"]) > best_score:
                best_score = float(metrics["f1"])
                best_params = (float(radius), str(weights))

    if best_params is None:
        raise RuntimeError("Hyperparameter search failed.")

    best_radius, best_weights = best_params
    final_preds = fit_and_predict(train_vectors, train_y, test_vectors, best_radius, best_weights)
    test_pred_labels = decode_predictions(final_preds, idx_to_label)
    metrics = compute_metrics_with_rejection(
        test_df["template"].to_numpy(),
        test_pred_labels,
        REJECT_TOKEN,
    )

    predictions_df = test_df.copy()
    predictions_df["pred_template"] = test_pred_labels
    predictions_df["is_rejected"] = predictions_df["pred_template"].eq(REJECT_TOKEN)
    predictions_df["correct"] = predictions_df["pred_template"].to_numpy() == predictions_df["template"].to_numpy()
    predictions_df = predictions_df.loc[:, ["image_path", "template", "pred_template", "is_rejected", "correct", "source"]]
    predictions_df.to_csv(run_dir / "test_predictions.csv", index=False)
    pd.DataFrame(search_rows).to_csv(run_dir / "val_search.csv", index=False)

    metadata = {
        "config": {
            **asdict(cfg),
            "dataset_root": None if cfg.dataset_root is None else str(cfg.dataset_root),
            "parquet_path": None if cfg.parquet_path is None else str(cfg.parquet_path),
            "train_parquet": None if cfg.train_parquet is None else str(cfg.train_parquet),
            "test_parquet": None if cfg.test_parquet is None else str(cfg.test_parquet),
            "image_root": None if cfg.image_root is None else str(cfg.image_root),
            "output_dir": str(cfg.output_dir),
        },
        "dataset": {
            "images_total_after_filtering": int(len(df)),
            "classes_after_filtering": int(df["template"].nunique()),
            "train_images": int(len(train_df)),
            "val_images": int(len(tune_val_df)),
            "test_images": int(len(test_df)),
            "small_class_filtering": filtering_summary,
        },
        "selection": {
            "best_radius": float(best_radius),
            "best_weights": best_weights,
            "best_val_f1": float(best_score),
        },
        "test_metrics": metrics,
    }
    summary_row = {
        "run_dir": str(run_dir),
        "seed": int(cfg.random_seed),
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "mcc": float(metrics["mcc"]),
        "cohen_kappa": float(metrics["cohen_kappa"]),
        "coverage": float(metrics["coverage"]),
        "covered_accuracy": float(metrics["covered_accuracy"]),
    }
    timing = finalize_run_timing(metadata, summary_row, run_started_at, start_perf)
    (run_dir / "run_config.json").write_text(json.dumps(metadata, indent=2))
    (run_dir / "test_metrics.json").write_text(json.dumps({"test_metrics": metrics, "timing": timing}, indent=2))
    pd.DataFrame([summary_row]).to_csv(run_dir / "metrics_summary.csv", index=False)

    print(f"best_radius={best_radius} best_weights={best_weights}")
    print(f"test_accuracy={metrics['accuracy']:.4f}")
    print(f"test_f1={metrics['f1']:.4f}")
    print(f"test_coverage={metrics['coverage']:.4f}")
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
