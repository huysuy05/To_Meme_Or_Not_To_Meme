#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from seed_unsupervised_eval import (
    VALID_EXTS,
    FrozenImageEmbedder,
    align_dataframe,
    collect_labeled_rows,
    drop_small_classes,
    filter_valid_image_rows,
    l2_normalize,
    load_existing_ssft_checkpoint,
    pick_device,
    set_seed,
)


NO_TEMPLATE_LABEL = "NO_TEMPLATE"


@dataclass
class RunConfig:
    reference_root: Path
    target_root: Path
    ssft_checkpoint_dir: Path
    output_dir: Path
    alpha: float = 0.60
    tau: float = 0.85
    delta: float = 0.10
    min_images_per_class: int = 7
    random_seed: int = 42
    batch_size: int = 16
    num_workers: int = 0
    max_templates: int | None = None
    max_targets: int | None = None
    max_images_per_template: int | None = None
    target_chunk_size: int = 10000
    siglip_model_id: str = "google/siglip2-base-patch16-224"
    dino_model_id: str = "facebook/dinov2-base"


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Assign known ImgFlip meme templates to unlabeled Reddit images by prototype matching "
            "with SSFT-adapted SigLIP and DINO backbones."
        )
    )
    parser.add_argument("--reference-root", required=True, help="Labeled ImgFlip folder-of-folders root.")
    parser.add_argument("--target-root", required=True, help="Root directory containing Reddit images.")
    parser.add_argument(
        "--ssft-checkpoint-dir",
        required=True,
        help="Directory containing `ssft_siglip.pt` and `ssft_dino.pt`.",
    )
    parser.add_argument("--output-dir", default="SEED/runs/reddit_template_assignment")
    parser.add_argument("--alpha", type=float, default=0.60)
    parser.add_argument("--tau", type=float, default=0.85)
    parser.add_argument("--delta", type=float, default=0.10)
    parser.add_argument("--min-images-per-class", type=int, default=7)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-templates", type=int, default=None)
    parser.add_argument("--max-targets", type=int, default=None)
    parser.add_argument("--max-images-per-template", type=int, default=None)
    parser.add_argument("--target-chunk-size", type=int, default=10000)
    parser.add_argument("--siglip-model-id", default="google/siglip2-base-patch16-224")
    parser.add_argument("--dino-model-id", default="facebook/dinov2-base")
    args = parser.parse_args()
    return RunConfig(
        reference_root=Path(args.reference_root).expanduser().resolve(),
        target_root=Path(args.target_root).expanduser().resolve(),
        ssft_checkpoint_dir=Path(args.ssft_checkpoint_dir).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        alpha=args.alpha,
        tau=args.tau,
        delta=args.delta,
        min_images_per_class=args.min_images_per_class,
        random_seed=args.random_seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_templates=args.max_templates,
        max_targets=args.max_targets,
        max_images_per_template=args.max_images_per_template,
        target_chunk_size=args.target_chunk_size,
        siglip_model_id=args.siglip_model_id,
        dino_model_id=args.dino_model_id,
    )


def collect_target_rows(root: Path, max_targets: int | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    count = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VALID_EXTS:
            continue
        rows.append(
            {
                "image_path": str(path),
                "image_id": path.stem,
            }
        )
        count += 1
        if max_targets is not None and count >= max_targets:
            break
    if not rows:
        raise ValueError(f"No target images found under {root}")
    return pd.DataFrame(rows)


def build_prototypes(reference_df: pd.DataFrame, embeddings: np.ndarray, template_names: list[str]) -> np.ndarray:
    label_array = reference_df["template"].to_numpy()
    prototypes: list[np.ndarray] = []
    for template_name in template_names:
        mask = label_array == template_name
        if not mask.any():
            raise ValueError(f"No reference images found for template={template_name}")
        prototype = embeddings[mask].mean(axis=0, keepdims=True)
        prototypes.append(l2_normalize(prototype)[0])
    return np.vstack(prototypes)


def fused_scores(
    target_siglip: np.ndarray,
    target_dino: np.ndarray,
    proto_siglip: np.ndarray,
    proto_dino: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    siglip_scores = target_siglip @ proto_siglip.T
    dino_scores = target_dino @ proto_dino.T
    scores = alpha * siglip_scores + (1.0 - alpha) * dino_scores
    return scores, siglip_scores, dino_scores


def summarize_scores(
    scores: np.ndarray,
    siglip_scores: np.ndarray,
    dino_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    best_idx = scores.argmax(axis=1)
    row_idx = np.arange(scores.shape[0])
    best_scores = scores[row_idx, best_idx]

    if scores.shape[1] == 1:
        second_scores = np.full(scores.shape[0], -np.inf, dtype=scores.dtype)
    else:
        masked = scores.copy()
        masked[row_idx, best_idx] = -np.inf
        second_scores = masked.max(axis=1)

    margins = best_scores - second_scores
    best_siglip_scores = siglip_scores[row_idx, best_idx]
    best_dino_scores = dino_scores[row_idx, best_idx]
    return best_idx, best_scores, second_scores, margins, np.vstack([best_siglip_scores, best_dino_scores]).T


def iter_chunks(df: pd.DataFrame, chunk_size: int) -> Iterator[pd.DataFrame]:
    if chunk_size <= 0:
        raise ValueError(f"target_chunk_size must be positive, got {chunk_size}")
    for start in range(0, len(df), chunk_size):
        yield df.iloc[start:start + chunk_size].reset_index(drop=True)


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.random_seed)
    device = pick_device()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = cfg.output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"device={device}")
    print(f"run_dir={run_dir}")

    reference_df = collect_labeled_rows(
        cfg.reference_root,
        max_templates=cfg.max_templates,
        max_images_per_template=cfg.max_images_per_template,
    )
    reference_df = filter_valid_image_rows(reference_df)
    reference_df, filtering_summary = drop_small_classes(reference_df, cfg.min_images_per_class)
    template_names = sorted(reference_df["template"].unique().tolist())

    target_df = collect_target_rows(cfg.target_root, max_targets=cfg.max_targets)

    siglip = FrozenImageEmbedder(cfg.siglip_model_id, "siglip", device, cfg.batch_size)
    dino = FrozenImageEmbedder(cfg.dino_model_id, "dino", device, cfg.batch_size)

    print(f"Loading existing SSFT checkpoints from {cfg.ssft_checkpoint_dir}")
    siglip_summary = load_existing_ssft_checkpoint(siglip, cfg.ssft_checkpoint_dir, device)
    dino_summary = load_existing_ssft_checkpoint(dino, cfg.ssft_checkpoint_dir, device)

    reference_paths = reference_df["image_path"].tolist()
    reference_siglip, kept_reference_siglip = siglip.encode_paths(reference_paths, desc="encode reference siglip")
    reference_dino, kept_reference_dino = dino.encode_paths(reference_paths, desc="encode reference dino")

    if kept_reference_siglip != kept_reference_dino:
        raise ValueError("Reference embedding path mismatch between SigLIP and DINO.")

    reference_df = align_dataframe(reference_df, kept_reference_siglip)
    proto_siglip = build_prototypes(reference_df, reference_siglip, template_names)
    proto_dino = build_prototypes(reference_df, reference_dino, template_names)

    predictions_path = run_dir / "reddit_template_predictions.csv"
    target_images_processed = 0
    assigned_known = 0
    assigned_no_template = 0

    for chunk_idx, target_chunk_df in enumerate(iter_chunks(target_df, cfg.target_chunk_size), start=1):
        print(f"Scoring target chunk {chunk_idx} with {len(target_chunk_df)} images")
        target_paths = target_chunk_df["image_path"].tolist()
        target_siglip, kept_target_siglip = siglip.encode_paths(target_paths, desc=f"encode target siglip chunk {chunk_idx}")
        target_dino, kept_target_dino = dino.encode_paths(target_paths, desc=f"encode target dino chunk {chunk_idx}")

        if kept_target_siglip != kept_target_dino:
            raise ValueError(f"Target embedding path mismatch in chunk {chunk_idx} between SigLIP and DINO.")

        target_chunk_df = align_dataframe(target_chunk_df, kept_target_siglip)
        scores, siglip_scores, dino_scores = fused_scores(
            target_siglip,
            target_dino,
            proto_siglip,
            proto_dino,
            alpha=cfg.alpha,
        )
        best_idx, best_scores, second_scores, margins, encoder_best_scores = summarize_scores(
            scores,
            siglip_scores,
            dino_scores,
        )
        matched_known = (best_scores >= cfg.tau) & (margins >= cfg.delta)
        pred_templates = np.where(
            matched_known,
            np.array([template_names[idx] for idx in best_idx], dtype=object),
            NO_TEMPLATE_LABEL,
        )

        chunk_results = target_chunk_df.copy()
        chunk_results["best_template_idx"] = best_idx.astype(int)
        chunk_results["best_template_name"] = np.array([template_names[idx] for idx in best_idx], dtype=object)
        chunk_results["pred_template"] = pred_templates
        chunk_results["matched_known_template"] = matched_known.astype(bool)
        chunk_results["best_score"] = best_scores.astype(float)
        chunk_results["second_score"] = second_scores.astype(float)
        chunk_results["margin"] = margins.astype(float)
        chunk_results["siglip_best_score"] = encoder_best_scores[:, 0].astype(float)
        chunk_results["dino_best_score"] = encoder_best_scores[:, 1].astype(float)
        chunk_results.to_csv(predictions_path, mode="a", index=False, header=(chunk_idx == 1))
        target_images_processed += len(chunk_results)
        assigned_known += int(chunk_results["matched_known_template"].sum())
        assigned_no_template += int((chunk_results["pred_template"] == NO_TEMPLATE_LABEL).sum())

    config_dict = asdict(cfg)
    config_dict["reference_root"] = str(cfg.reference_root)
    config_dict["target_root"] = str(cfg.target_root)
    config_dict["ssft_checkpoint_dir"] = str(cfg.ssft_checkpoint_dir)
    config_dict["output_dir"] = str(cfg.output_dir)

    run_metadata = {
        "config": {
            **config_dict,
            "device": str(device),
        },
        "reference_bank": {
            "num_templates": int(len(template_names)),
            "num_reference_images": int(len(reference_df)),
            "small_class_filtering": filtering_summary,
        },
        "targets": {
            "num_target_images_requested": int(len(target_df)),
            "num_target_images_processed": int(target_images_processed),
            "assigned_known_templates": assigned_known,
            "assigned_no_template": assigned_no_template,
        },
        "checkpoints": {
            "siglip": siglip_summary,
            "dino": dino_summary,
        },
        "parameters": {
            "alpha": float(cfg.alpha),
            "tau": float(cfg.tau),
            "delta": float(cfg.delta),
        },
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_metadata, indent=2))

    print(f"reference_templates={len(template_names)} reference_images={len(reference_df)}")
    print(f"target_images_processed={target_images_processed}")
    print(f"assigned_known_templates={assigned_known} assigned_no_template={assigned_no_template}")
    print(f"predictions_saved={predictions_path}")


if __name__ == "__main__":
    main()
