#!/usr/bin/env python3
from __future__ import annotations

import io
import random
import time
import zipfile
import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_int_list(text: str) -> tuple[int, ...]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer seed.")
    return tuple(int(value) for value in values)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_rgb_image(path: str | Path) -> Image.Image | None:
    try:
        with Image.open(path) as img:
            return ImageOps.exif_transpose(img).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def collect_labeled_rows(
    dataset_root: Path,
    max_templates: int | None = None,
    max_images_per_template: int | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    template_dirs = sorted([path for path in dataset_root.iterdir() if path.is_dir()])
    if max_templates is not None:
        template_dirs = template_dirs[:max_templates]

    for template_dir in template_dirs:
        image_paths = sorted(
            [path for path in template_dir.iterdir() if path.is_file() and path.suffix.lower() in VALID_EXTS]
        )
        if max_images_per_template is not None:
            image_paths = image_paths[:max_images_per_template]
        for image_path in image_paths:
            rows.append(
                {
                    "image_path": str(image_path),
                    "template": template_dir.name,
                    "source": "imgflip",
                }
            )

    if not rows:
        raise ValueError(f"No labeled images found under {dataset_root}")
    return pd.DataFrame(rows)


def _read_parquet_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            parquet_names = [name for name in zf.namelist() if name.lower().endswith(".parquet")]
            if not parquet_names:
                raise ValueError(f"No parquet file found inside {path}")
            if len(parquet_names) > 1:
                preferred = [name for name in parquet_names if "meme_entries" in Path(name).name]
                if preferred:
                    parquet_name = preferred[0]
                else:
                    raise ValueError(
                        f"Multiple parquet files found inside {path}. "
                        "Please unzip it first or keep only the target parquet."
                    )
            else:
                parquet_name = parquet_names[0]
            with zf.open(parquet_name) as infile:
                return pd.read_parquet(io.BytesIO(infile.read()))
    return pd.read_parquet(path)


def _normalize_split_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if {"image_path", "template"}.issubset(df.columns):
        rows = df.copy()
    elif {"path", "template_name"}.issubset(df.columns):
        rows = df.loc[:, ["path", "template_name"]].rename(
            columns={"path": "image_path", "template_name": "template"}
        ).copy()
    else:
        raise ValueError(
            "Parquet must contain either ['image_path', 'template'] or ['path', 'template_name'] columns."
        )

    rows["image_path"] = rows["image_path"].astype(str)
    rows["template"] = rows["template"].astype(str)
    if "source" not in rows.columns:
        rows["source"] = "imgflip"
    else:
        rows["source"] = rows["source"].astype(str)
    return rows.reset_index(drop=True)


def collect_rows_from_parquet(
    parquet_path: Path,
    image_root: Path | None = None,
    path_prefix_from: str | None = None,
    path_prefix_to: str | None = None,
    max_templates: int | None = None,
    max_images_per_template: int | None = None,
) -> pd.DataFrame:
    df = _read_parquet_file(parquet_path)
    rows = _normalize_split_dataframe(df)

    if path_prefix_from is not None and path_prefix_to is not None:
        rows["image_path"] = rows["image_path"].str.replace(path_prefix_from, path_prefix_to, regex=False)

    if image_root is not None:
        image_root = image_root.expanduser().resolve()
        rows["image_path"] = rows["image_path"].map(
            lambda value: str(image_root / Path(value).parent.name / Path(value).name)
            if not Path(value).exists()
            else value
        )

    if max_templates is not None:
        keep_templates = sorted(rows["template"].unique())[:max_templates]
        rows = rows[rows["template"].isin(keep_templates)].copy()

    if max_images_per_template is not None:
        rows = (
            rows.groupby("template", group_keys=False)
            .head(max_images_per_template)
            .reset_index(drop=True)
        )
    else:
        rows = rows.reset_index(drop=True)

    if rows.empty:
        raise ValueError(f"No rows collected from {parquet_path}")
    return rows


def load_dataset_rows(
    dataset_root: str | None,
    parquet_path: str | None,
    image_root: str | None = None,
    path_prefix_from: str | None = None,
    path_prefix_to: str | None = None,
    max_templates: int | None = None,
    max_images_per_template: int | None = None,
) -> pd.DataFrame:
    if dataset_root:
        return collect_labeled_rows(
            Path(dataset_root).expanduser().resolve(),
            max_templates=max_templates,
            max_images_per_template=max_images_per_template,
        )
    if parquet_path:
        return collect_rows_from_parquet(
            Path(parquet_path).expanduser().resolve(),
            image_root=None if image_root is None else Path(image_root),
            path_prefix_from=path_prefix_from,
            path_prefix_to=path_prefix_to,
            max_templates=max_templates,
            max_images_per_template=max_images_per_template,
        )
    raise ValueError("Either dataset_root or parquet_path must be provided.")


def load_split_rows(
    train_parquet: str,
    test_parquet: str,
    image_root: str | None = None,
    path_prefix_from: str | None = None,
    path_prefix_to: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = collect_rows_from_parquet(
        Path(train_parquet).expanduser().resolve(),
        image_root=None if image_root is None else Path(image_root),
        path_prefix_from=path_prefix_from,
        path_prefix_to=path_prefix_to,
    )
    test_df = collect_rows_from_parquet(
        Path(test_parquet).expanduser().resolve(),
        image_root=None if image_root is None else Path(image_root),
        path_prefix_from=path_prefix_from,
        path_prefix_to=path_prefix_to,
    )
    train_df["source"] = "imgflip_train"
    test_df["source"] = "imgflip_test"
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def filter_valid_image_rows(df: pd.DataFrame) -> pd.DataFrame:
    keep_rows: list[bool] = []
    for image_path in tqdm(df["image_path"].tolist(), desc="verify images"):
        keep_rows.append(load_rgb_image(image_path) is not None)
    filtered = df[keep_rows].reset_index(drop=True)
    dropped = len(df) - len(filtered)
    if dropped:
        print(f"Dropped {dropped} unreadable images.")
    return filtered


def drop_small_classes(
    df: pd.DataFrame,
    min_images_per_class: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    counts = df["template"].value_counts()
    keep_templates = counts[counts >= min_images_per_class].index
    filtered = df[df["template"].isin(keep_templates)].copy().reset_index(drop=True)

    removed_templates = counts[counts < min_images_per_class]
    summary = {
        "min_images_required_per_class": int(min_images_per_class),
        "templates_before": int(counts.shape[0]),
        "templates_after": int(filtered["template"].nunique()),
        "templates_removed": int(removed_templates.shape[0]),
        "images_before": int(len(df)),
        "images_after": int(len(filtered)),
        "images_removed": int(len(df) - len(filtered)),
        "removed_template_examples": removed_templates.head(20).to_dict(),
    }
    return filtered, summary


def stratified_split(
    df: pd.DataFrame,
    train_size: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df,
        train_size=train_size,
        stratify=df["template"],
        random_state=random_seed,
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    train_df["source"] = "imgflip_train"
    test_df["source"] = "imgflip_test"
    return train_df, test_df


def align_dataframe(df: pd.DataFrame, kept_paths: list[str]) -> pd.DataFrame:
    path_to_index = {path: idx for idx, path in enumerate(df["image_path"].tolist())}
    return df.iloc[[path_to_index[path] for path in kept_paths]].reset_index(drop=True)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def compute_metrics_with_rejection(y_true: np.ndarray, y_pred: np.ndarray, reject_token: str) -> dict[str, float]:
    metrics = compute_metrics(y_true, y_pred)
    covered_mask = y_pred != reject_token
    coverage = float(covered_mask.mean()) if y_pred.size else 0.0
    metrics["coverage"] = coverage
    if covered_mask.any():
        covered_true = y_true[covered_mask]
        covered_pred = y_pred[covered_mask]
        metrics["covered_accuracy"] = float(accuracy_score(covered_true, covered_pred))
        metrics["covered_count"] = int(covered_mask.sum())
    else:
        metrics["covered_accuracy"] = 0.0
        metrics["covered_count"] = 0
    metrics["rejected_count"] = int((~covered_mask).sum())
    return metrics


def finalize_run_timing(
    metadata: dict[str, Any],
    summary_row: dict[str, Any],
    run_started_at: str,
    start_perf: float,
) -> dict[str, Any]:
    duration_seconds = round(time.perf_counter() - start_perf, 3)
    duration_minutes = round(duration_seconds / 60.0, 3)
    timing = {
        "started_at": run_started_at,
        "finished_at": now_iso(),
        "duration_seconds": duration_seconds,
        "duration_minutes": duration_minutes,
    }
    metadata["timing"] = timing
    summary_row["duration_seconds"] = duration_seconds
    summary_row["duration_minutes"] = duration_minutes
    return timing


def summarize_batch_timings(batch_rows: list[dict[str, Any]]) -> dict[str, float | int]:
    durations = [float(row["duration_seconds"]) for row in batch_rows if "duration_seconds" in row]
    if not durations:
        return {
            "run_count": int(len(batch_rows)),
            "average_duration_seconds": 0.0,
            "average_duration_minutes": 0.0,
            "total_duration_seconds": 0.0,
            "total_duration_minutes": 0.0,
        }
    total_duration_seconds = round(float(sum(durations)), 3)
    average_duration_seconds = round(total_duration_seconds / len(durations), 3)
    return {
        "run_count": int(len(durations)),
        "average_duration_seconds": average_duration_seconds,
        "average_duration_minutes": round(average_duration_seconds / 60.0, 3),
        "total_duration_seconds": total_duration_seconds,
        "total_duration_minutes": round(total_duration_seconds / 60.0, 3),
    }
