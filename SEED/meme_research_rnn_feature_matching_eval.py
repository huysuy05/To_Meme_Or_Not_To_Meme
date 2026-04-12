#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import ORB, match_descriptors
from sklearn.model_selection import train_test_split
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
    set_seed,
    stratified_split,
    summarize_batch_timings,
)


REJECT_TOKEN = "__OUTLIER__"


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
    max_refs_per_template: int = 20
    max_test_images: int | None = None
    target_size: int = 256
    n_keypoints: int = 512
    fast_threshold: float = 0.08
    radius_values: tuple[float, ...] = (0.90, 0.93, 0.95, 0.97, 0.99)
    weights: tuple[str, ...] = ("uniform", "distance")
    num_workers: int = 4
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
            "Closed-set ImgFlip evaluation for a standalone feature-matching rNN baseline. "
            "Images are represented by ORB descriptors and classified by radius-neighbor voting over feature-match distance."
        )
    )
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--parquet-path", default=None)
    parser.add_argument("--train-parquet", default=None)
    parser.add_argument("--test-parquet", default=None)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--output-dir", default="SEED/uns_runs/meme_research_rnn_feature_matching_eval")
    parser.add_argument("--train-size", type=float, default=0.80)
    parser.add_argument("--val-size", type=float, default=0.10)
    parser.add_argument("--min-images-per-class", type=int, default=7)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-refs-per-template", type=int, default=20)
    parser.add_argument("--max-test-images", type=int, default=None)
    parser.add_argument("--target-size", type=int, default=256)
    parser.add_argument("--n-keypoints", type=int, default=512)
    parser.add_argument("--fast-threshold", type=float, default=0.08)
    parser.add_argument("--radius-values", type=parse_float_list, default=(0.90, 0.93, 0.95, 0.97, 0.99))
    parser.add_argument("--weights", type=parse_str_list, default=("uniform", "distance"))
    parser.add_argument("--num-workers", type=int, default=4)
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
        max_refs_per_template=args.max_refs_per_template,
        max_test_images=args.max_test_images,
        target_size=args.target_size,
        n_keypoints=args.n_keypoints,
        fast_threshold=args.fast_threshold,
        radius_values=args.radius_values,
        weights=args.weights,
        num_workers=args.num_workers,
        max_templates=args.max_templates,
        max_images_per_template=args.max_images_per_template,
        path_prefix_from=args.path_prefix_from,
        path_prefix_to=args.path_prefix_to,
        seeds=args.seeds,
    )


def extract_descriptors(path: str, target_size: int, n_keypoints: int, fast_threshold: float) -> np.ndarray | None:
    image = load_rgb_image(path)
    if image is None:
        raise ValueError(f"Unreadable image slipped through filtering: {path}")
    gray = image.convert("L").resize((target_size, target_size), resample=Image.Resampling.LANCZOS)
    arr = np.asarray(gray, dtype=np.float32) / 255.0
    orb = ORB(n_keypoints=n_keypoints, fast_threshold=fast_threshold)
    try:
        orb.detect_and_extract(arr)
        descriptors = orb.descriptors
    except RuntimeError:
        descriptors = None
    if descriptors is None or descriptors.size == 0:
        return None
    return descriptors


def feature_match_distance(desc_a: np.ndarray | None, desc_b: np.ndarray | None) -> float:
    if desc_a is None or desc_b is None or len(desc_a) == 0 or len(desc_b) == 0:
        return 1.0
    matches = match_descriptors(desc_a, desc_b, metric="hamming", cross_check=True)
    if matches.size == 0:
        return 1.0
    norm = float(max(len(desc_a), len(desc_b)))
    match_ratio = min(float(len(matches)) / max(norm, 1.0), 1.0)
    return 1.0 - match_ratio


def extract_many_descriptors(paths: list[str], cfg: RunConfig, desc: str) -> list[np.ndarray | None]:
    descriptors: list[np.ndarray | None] = []
    for path in tqdm(paths, desc=desc):
        descriptors.append(extract_descriptors(path, cfg.target_size, cfg.n_keypoints, cfg.fast_threshold))
    return descriptors


def compute_distance_matrix(
    query_descs: list[np.ndarray | None],
    ref_descs: list[np.ndarray | None],
    num_workers: int,
    desc: str,
) -> np.ndarray:
    matrix = np.ones((len(query_descs), len(ref_descs)), dtype=np.float32)

    def process_one(query_idx: int) -> tuple[int, np.ndarray]:
        row = np.empty(len(ref_descs), dtype=np.float32)
        query_desc = query_descs[query_idx]
        for ref_idx, ref_desc in enumerate(ref_descs):
            row[ref_idx] = feature_match_distance(query_desc, ref_desc)
        return query_idx, row

    if num_workers > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for query_idx, row in tqdm(
                executor.map(process_one, range(len(query_descs))),
                total=len(query_descs),
                desc=desc,
            ):
                matrix[query_idx] = row
    else:
        for query_idx in tqdm(range(len(query_descs)), desc=desc):
            _, row = process_one(query_idx)
            matrix[query_idx] = row
    return matrix


def predict_from_distance_matrix(
    distance_matrix: np.ndarray,
    train_labels: np.ndarray,
    radius: float,
    weights: str,
) -> np.ndarray:
    preds = np.full(distance_matrix.shape[0], -1, dtype=np.int64)
    num_classes = int(train_labels.max()) + 1
    for row_idx, row in enumerate(distance_matrix):
        neighbor_idx = np.where(row <= radius)[0]
        if neighbor_idx.size == 0:
            continue
        labels = train_labels[neighbor_idx]
        if weights == "uniform":
            scores = np.bincount(labels, minlength=num_classes).astype(np.float32)
        elif weights == "distance":
            inv = 1.0 / np.clip(row[neighbor_idx], 1e-6, None)
            scores = np.bincount(labels, weights=inv, minlength=num_classes).astype(np.float32)
        else:
            raise ValueError(f"Unsupported weights={weights}")
        preds[row_idx] = int(scores.argmax())
    return preds


def decode_predictions(preds: np.ndarray, idx_to_label: dict[int, str]) -> np.ndarray:
    return np.array(
        [idx_to_label[int(pred)] if int(pred) >= 0 else REJECT_TOKEN for pred in preds],
        dtype=object,
    )


def run_once(cfg: RunConfig, run_dir: Path) -> dict[str, float | int | str]:
    set_seed(cfg.random_seed)
    run_started_at = now_iso()
    start_perf = time.perf_counter()
    run_dir.mkdir(parents=True, exist_ok=True)
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
    tune_train_df = (
        tune_train_df.groupby("template", group_keys=False)
        .head(cfg.max_refs_per_template)
        .reset_index(drop=True)
    )
    train_df = (
        train_df.groupby("template", group_keys=False)
        .head(cfg.max_refs_per_template)
        .reset_index(drop=True)
    )
    tune_val_df = tune_val_df.reset_index(drop=True)
    if cfg.max_test_images is not None:
        test_df = test_df.iloc[:cfg.max_test_images].reset_index(drop=True)

    labels = sorted(train_df["template"].unique().tolist())
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    tune_train_descs = extract_many_descriptors(tune_train_df["image_path"].tolist(), cfg, "extract tune-train descriptors")
    tune_val_descs = extract_many_descriptors(tune_val_df["image_path"].tolist(), cfg, "extract tune-val descriptors")
    train_descs = extract_many_descriptors(train_df["image_path"].tolist(), cfg, "extract train descriptors")
    test_descs = extract_many_descriptors(test_df["image_path"].tolist(), cfg, "extract test descriptors")

    tune_val_dist = compute_distance_matrix(tune_val_descs, tune_train_descs, cfg.num_workers, "match tune-val to tune-train")
    test_dist = compute_distance_matrix(test_descs, train_descs, cfg.num_workers, "match test to train")

    tune_train_y = tune_train_df["template"].map(label_to_idx).to_numpy(dtype=np.int64)
    train_y = train_df["template"].map(label_to_idx).to_numpy(dtype=np.int64)

    search_rows: list[dict[str, float | str]] = []
    best_score = float("-inf")
    best_params: tuple[float, str] | None = None

    for radius in cfg.radius_values:
        for weights in cfg.weights:
            val_preds = predict_from_distance_matrix(tune_val_dist, tune_train_y, radius, weights)
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
    final_preds = predict_from_distance_matrix(test_dist, train_y, best_radius, best_weights)
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
