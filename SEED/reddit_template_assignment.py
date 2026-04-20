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
    assign_to_centroids,
    build_reducer,
    collect_labeled_rows,
    drop_small_classes,
    filter_valid_image_rows,
    fit_clusterer,
    fused_embedding,
    l2_normalize,
    load_existing_ssft_checkpoint,
    load_split_rows,
    majority_vote_mapping,
    pick_device,
    refine_clusters,
    set_seed,
)


NO_TEMPLATE_LABEL = "NO_TEMPLATE"
DEFAULT_RUN_ROOT = Path("SEED/runs/reddit_template_assignment")
DEFAULT_TRAIN_PARQUET = Path("SEED/splits/imgflip_80_20/train.parquet")
DEFAULT_TEST_PARQUET = Path("SEED/splits/imgflip_80_20/test.parquet")
DEFAULT_UPDATE_JSONL = Path("data/merged_parsed_results_with_template_predictions.jsonl")


@dataclass
class RunConfig:
    target_root: Path
    output_dir: Path
    train_parquet: Path | None = DEFAULT_TRAIN_PARQUET.resolve()
    test_parquet: Path | None = DEFAULT_TEST_PARQUET.resolve()
    reference_root: Path | None = None
    ssft_checkpoint_dir: Path | None = None
    update_jsonl_path: Path | None = DEFAULT_UPDATE_JSONL.resolve()
    alpha: float = 0.65
    tau: float = 0.60
    delta: float = 0.05
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
    filter_jsonl_path: Path | None = None
    reducer: str = "pca"
    reducer_dim: int = 128
    cluster_method: str = "hdbscan"
    kmeans_clusters: int | None = None
    refinement_steps: int = 1


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Assign ImgFlip template labels to unlabeled Reddit images by clustering ImgFlip train embeddings, "
            "then mapping Reddit images to the discovered cluster centroids at test time."
        )
    )
    parser.add_argument("--target-root", required=True, help="Root directory containing Reddit images.")
    parser.add_argument(
        "--train-parquet",
        default=str(DEFAULT_TRAIN_PARQUET),
        help="ImgFlip train split parquet used to fit clusters.",
    )
    parser.add_argument(
        "--test-parquet",
        default=str(DEFAULT_TEST_PARQUET),
        help="Optional ImgFlip test split parquet. Stored in metadata for provenance only.",
    )
    parser.add_argument(
        "--reference-root",
        default=None,
        help="Fallback labeled ImgFlip folder-of-folders root when train/test split parquets are unavailable.",
    )
    parser.add_argument(
        "--ssft-checkpoint-dir",
        default=None,
        help="Optional directory containing `ssft_siglip.pt` and `ssft_dino.pt`. Omit for baseline mode.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument(
        "--update-jsonl-path",
        default=str(DEFAULT_UPDATE_JSONL),
        help="JSONL file to update in place with the reassigned cluster-based template predictions.",
    )
    parser.add_argument(
        "--filter-jsonl-path",
        default=None,
        help="Optional JSONL file with `key` values used only to filter which Reddit images are assigned.",
    )
    parser.add_argument("--alpha", type=float, default=0.65)
    parser.add_argument("--tau", type=float, default=0.60)
    parser.add_argument("--delta", type=float, default=0.05)
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
    parser.add_argument("--reducer", choices=["none", "pca", "umap"], default="pca")
    parser.add_argument("--reducer-dim", type=int, default=128)
    parser.add_argument("--cluster-method", choices=["kmeans", "hdbscan"], default="hdbscan")
    parser.add_argument("--kmeans-clusters", type=int, default=None)
    parser.add_argument("--refinement-steps", type=int, default=1)
    args = parser.parse_args()
    return RunConfig(
        target_root=Path(args.target_root).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        train_parquet=None if args.train_parquet is None else Path(args.train_parquet).expanduser().resolve(),
        test_parquet=None if args.test_parquet is None else Path(args.test_parquet).expanduser().resolve(),
        reference_root=None if args.reference_root is None else Path(args.reference_root).expanduser().resolve(),
        ssft_checkpoint_dir=(
            None if args.ssft_checkpoint_dir is None else Path(args.ssft_checkpoint_dir).expanduser().resolve()
        ),
        update_jsonl_path=(
            None if args.update_jsonl_path is None else Path(args.update_jsonl_path).expanduser().resolve()
        ),
        filter_jsonl_path=(
            None if args.filter_jsonl_path is None else Path(args.filter_jsonl_path).expanduser().resolve()
        ),
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
        reducer=args.reducer,
        reducer_dim=args.reducer_dim,
        cluster_method=args.cluster_method,
        kmeans_clusters=args.kmeans_clusters,
        refinement_steps=args.refinement_steps,
    )


def load_jsonl_keys(jsonl_path: Path) -> set[str]:
    keys: set[str] = set()
    with jsonl_path.open("r", encoding="utf-8") as infile:
        for line_number, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            key = payload.get("key")
            if key is None:
                raise KeyError(f"Missing `key` in {jsonl_path} at line {line_number}")
            keys.add(str(key))
    if not keys:
        raise ValueError(f"No keys were loaded from {jsonl_path}")
    return keys


def collect_target_rows(
    root: Path,
    max_targets: int | None = None,
    allowed_image_ids: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total_images_seen = 0
    skipped_by_key_filter = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VALID_EXTS:
            continue
        total_images_seen += 1
        image_id = path.stem
        if allowed_image_ids is not None and image_id not in allowed_image_ids:
            skipped_by_key_filter += 1
            continue
        rows.append({"image_path": str(path), "image_id": image_id})
        if max_targets is not None and len(rows) >= max_targets:
            break
    if not rows:
        if allowed_image_ids is not None:
            raise ValueError(f"No target images found under {root} after applying JSONL key filter.")
        raise ValueError(f"No target images found under {root}")
    summary = {
        "total_images_seen": int(total_images_seen),
        "skipped_by_key_filter": int(skipped_by_key_filter),
        "allowed_image_ids_count": None if allowed_image_ids is None else int(len(allowed_image_ids)),
        "target_images_collected": int(len(rows)),
    }
    return pd.DataFrame(rows), summary


def collect_train_reference_rows(cfg: RunConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    if cfg.train_parquet is not None and cfg.train_parquet.exists():
        if cfg.test_parquet is None or not cfg.test_parquet.exists():
            raise FileNotFoundError(
                "When using --train-parquet, --test-parquet must also exist because the loader expects the fixed split pair."
            )
        train_df, test_df = load_split_rows(str(cfg.train_parquet), str(cfg.test_parquet))
        if cfg.max_templates is not None:
            keep_templates = sorted(train_df["template"].unique().tolist())[: cfg.max_templates]
            train_df = train_df[train_df["template"].isin(keep_templates)].reset_index(drop=True)
        if cfg.max_images_per_template is not None:
            train_df = (
                train_df.groupby("template", group_keys=False)
                .head(cfg.max_images_per_template)
                .reset_index(drop=True)
            )
        return train_df, {
            "source_mode": "precomputed_train_split",
            "test_split_path": str(cfg.test_parquet),
            "train_rows_before_filtering": int(len(train_df)),
            "test_rows_available_for_reference_only": int(len(test_df)),
        }

    if cfg.reference_root is None:
        raise ValueError("Provide either --train-parquet/--test-parquet or --reference-root.")
    reference_df = collect_labeled_rows(
        cfg.reference_root,
        max_templates=cfg.max_templates,
        max_images_per_template=cfg.max_images_per_template,
    )
    return reference_df, {"source_mode": "reference_root_full_dataset"}


def build_cluster_centroids(embeddings: np.ndarray, labels: np.ndarray, centroid_ids: np.ndarray) -> np.ndarray:
    centroids: list[np.ndarray] = []
    for cluster_id in centroid_ids:
        mask = labels == cluster_id
        if not mask.any():
            raise ValueError(f"No discovery embeddings found for cluster_id={cluster_id}")
        centroid = embeddings[mask].mean(axis=0, keepdims=True)
        centroids.append(l2_normalize(centroid)[0])
    return np.vstack(centroids)


def pick_effective_reducer_dim(x: np.ndarray, requested_dim: int) -> int:
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, got shape={x.shape}")
    max_dim = min(int(x.shape[0]), int(x.shape[1]))
    return max(1, min(int(requested_dim), max_dim))


def summarize_scores(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    return best_idx, best_scores, second_scores, margins


def iter_chunks(df: pd.DataFrame, chunk_size: int) -> Iterator[pd.DataFrame]:
    if chunk_size <= 0:
        raise ValueError(f"target_chunk_size must be positive, got {chunk_size}")
    for start in range(0, len(df), chunk_size):
        yield df.iloc[start:start + chunk_size].reset_index(drop=True)


def compute_template_final(template_original: Any, pred_template: Any) -> str:
    if isinstance(pred_template, str) and pred_template and pred_template != NO_TEMPLATE_LABEL:
        return pred_template
    if isinstance(template_original, str) and template_original:
        return template_original
    if isinstance(pred_template, str) and pred_template:
        return pred_template
    return NO_TEMPLATE_LABEL


def compute_template_source(template_original: Any, pred_template: Any) -> str:
    if isinstance(pred_template, str) and pred_template and pred_template != NO_TEMPLATE_LABEL:
        return "cluster_test_assignment"
    if isinstance(template_original, str) and template_original and template_original != NO_TEMPLATE_LABEL:
        return "annotation"
    return "cluster_test_no_template"


def update_assignment_jsonl(jsonl_path: Path, assignments: dict[str, dict[str, Any]]) -> dict[str, int]:
    temp_path = jsonl_path.with_suffix(jsonl_path.suffix + ".tmp")
    updated_rows = 0
    missing_rows = 0
    with jsonl_path.open("r", encoding="utf-8") as infile, temp_path.open("w", encoding="utf-8") as outfile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            key = str(payload.get("key"))
            assignment = assignments.get(key)
            if assignment is None:
                outfile.write(json.dumps(payload, ensure_ascii=True) + "\n")
                missing_rows += 1
                continue

            data = payload.get("data")
            if not isinstance(data, dict):
                data = {}
                payload["data"] = data

            template_original = data.get("template_original", data.get("template"))
            pred_template = assignment["pred_template"]
            data["template_original"] = template_original
            data["pred_template"] = pred_template
            data["template_final"] = compute_template_final(template_original, pred_template)
            data["template_source"] = compute_template_source(template_original, pred_template)
            data["template_prediction"] = assignment["template_prediction"]
            outfile.write(json.dumps(payload, ensure_ascii=True) + "\n")
            updated_rows += 1

    temp_path.replace(jsonl_path)
    return {"updated_rows": int(updated_rows), "untouched_rows": int(missing_rows)}


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.random_seed)
    device = pick_device()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = cfg.output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"device={device}")
    print(f"run_dir={run_dir}")

    reference_df, reference_source_summary = collect_train_reference_rows(cfg)
    reference_df = filter_valid_image_rows(reference_df)
    reference_df, filtering_summary = drop_small_classes(reference_df, cfg.min_images_per_class)
    reference_df["source"] = "imgflip_train"

    allowed_image_ids: set[str] | None = None
    key_source_path = cfg.filter_jsonl_path or cfg.update_jsonl_path
    if key_source_path is not None:
        print(f"Loading allowed image IDs from {key_source_path}")
        allowed_image_ids = load_jsonl_keys(key_source_path)
        print(f"loaded_jsonl_keys={len(allowed_image_ids)}")

    target_df, target_collection_summary = collect_target_rows(
        cfg.target_root,
        max_targets=cfg.max_targets,
        allowed_image_ids=allowed_image_ids,
    )

    siglip = FrozenImageEmbedder(cfg.siglip_model_id, "siglip", device, cfg.batch_size)
    dino = FrozenImageEmbedder(cfg.dino_model_id, "dino", device, cfg.batch_size)

    if cfg.ssft_checkpoint_dir is not None:
        print(f"Loading existing SSFT checkpoints from {cfg.ssft_checkpoint_dir}")
        assignment_mode = "cluster_ssft"
        siglip_summary = load_existing_ssft_checkpoint(siglip, cfg.ssft_checkpoint_dir, device)
        dino_summary = load_existing_ssft_checkpoint(dino, cfg.ssft_checkpoint_dir, device)
    else:
        print("No SSFT checkpoints provided; using baseline frozen backbones.")
        assignment_mode = "cluster_baseline"
        siglip_summary = {"loaded_existing": False, "mode": assignment_mode}
        dino_summary = {"loaded_existing": False, "mode": assignment_mode}

    reference_paths = reference_df["image_path"].tolist()
    reference_siglip, kept_reference_siglip = siglip.encode_paths(reference_paths, desc="encode train siglip")
    reference_dino, kept_reference_dino = dino.encode_paths(reference_paths, desc="encode train dino")
    if kept_reference_siglip != kept_reference_dino:
        raise ValueError("Reference embedding path mismatch between SigLIP and DINO.")
    reference_df = align_dataframe(reference_df, kept_reference_siglip)

    reference_fused = fused_embedding(reference_siglip, reference_dino, cfg.alpha)
    effective_reducer_dim = pick_effective_reducer_dim(reference_fused, cfg.reducer_dim)
    reducer = build_reducer(cfg.reducer, effective_reducer_dim, cfg.random_seed)
    reference_reduced = reducer.fit_transform(reference_fused)
    reference_reduced = l2_normalize(np.asarray(reference_reduced, dtype=np.float32))

    num_clusters = cfg.kmeans_clusters or reference_df["template"].nunique()
    _, initial_labels = fit_clusterer(
        method=cfg.cluster_method,
        x=reference_reduced,
        num_clusters=num_clusters,
        random_seed=cfg.random_seed,
    )
    refined_labels, reduced_centroids, centroid_ids = refine_clusters(
        reference_reduced,
        initial_labels,
        steps=cfg.refinement_steps,
    )
    cluster_to_template = majority_vote_mapping(reference_df, refined_labels)

    cluster_siglip_centroids = build_cluster_centroids(reference_siglip, refined_labels, centroid_ids)
    cluster_dino_centroids = build_cluster_centroids(reference_dino, refined_labels, centroid_ids)

    predictions_path = run_dir / "reddit_template_predictions.csv"
    assignments_for_jsonl: dict[str, dict[str, Any]] = {}
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
        target_fused = fused_embedding(target_siglip, target_dino, cfg.alpha)
        target_reduced = reducer.transform(target_fused)
        target_reduced = l2_normalize(np.asarray(target_reduced, dtype=np.float32))

        cluster_ids = assign_to_centroids(target_reduced, reduced_centroids, centroid_ids)
        reduced_scores = target_reduced @ reduced_centroids.T
        best_idx, best_scores, second_scores, margins = summarize_scores(reduced_scores)

        siglip_scores = target_siglip @ cluster_siglip_centroids.T
        dino_scores = target_dino @ cluster_dino_centroids.T
        row_idx = np.arange(len(target_chunk_df))
        best_siglip_scores = siglip_scores[row_idx, best_idx]
        best_dino_scores = dino_scores[row_idx, best_idx]

        best_cluster_template = np.array(
            [cluster_to_template.get(int(cluster_id), NO_TEMPLATE_LABEL) for cluster_id in cluster_ids],
            dtype=object,
        )
        matched_known = (
            (best_scores >= cfg.tau)
            & (margins >= cfg.delta)
            & np.array([value != NO_TEMPLATE_LABEL for value in best_cluster_template], dtype=bool)
        )
        pred_templates = np.where(matched_known, best_cluster_template, NO_TEMPLATE_LABEL)

        chunk_results = target_chunk_df.copy()
        chunk_results["cluster_id"] = cluster_ids.astype(int)
        chunk_results["best_template_name"] = best_cluster_template
        chunk_results["pred_template"] = pred_templates
        chunk_results["matched_known_template"] = matched_known.astype(bool)
        chunk_results["best_score"] = best_scores.astype(float)
        chunk_results["second_score"] = second_scores.astype(float)
        chunk_results["margin"] = margins.astype(float)
        chunk_results["siglip_best_score"] = best_siglip_scores.astype(float)
        chunk_results["dino_best_score"] = best_dino_scores.astype(float)
        chunk_results["assignment_mode"] = assignment_mode
        chunk_results["cluster_method"] = cfg.cluster_method
        chunk_results["reducer"] = cfg.reducer
        chunk_results.to_csv(predictions_path, mode="a", index=False, header=(chunk_idx == 1))

        for row in chunk_results.itertuples(index=False):
            assignments_for_jsonl[str(row.image_id)] = {
                "pred_template": str(row.pred_template),
                "template_prediction": {
                    "image_path": str(row.image_path),
                    "best_template_idx": int(row.cluster_id),
                    "best_template_name": str(row.best_template_name),
                    "pred_template": str(row.pred_template),
                    "matched_known_template": bool(row.matched_known_template),
                    "best_score": float(row.best_score),
                    "second_score": float(row.second_score),
                    "margin": float(row.margin),
                    "siglip_best_score": float(row.siglip_best_score),
                    "dino_best_score": float(row.dino_best_score),
                    "assignment_method": assignment_mode,
                    "cluster_method": cfg.cluster_method,
                    "reducer": cfg.reducer,
                },
            }

        target_images_processed += len(chunk_results)
        assigned_known += int(chunk_results["matched_known_template"].sum())
        assigned_no_template += int((chunk_results["pred_template"] == NO_TEMPLATE_LABEL).sum())

    jsonl_update_summary: dict[str, int] | None = None
    if cfg.update_jsonl_path is not None:
        print(f"Updating JSONL assignments in {cfg.update_jsonl_path}")
        jsonl_update_summary = update_assignment_jsonl(cfg.update_jsonl_path, assignments_for_jsonl)

    config_dict = asdict(cfg)
    config_dict["target_root"] = str(cfg.target_root)
    config_dict["output_dir"] = str(cfg.output_dir)
    config_dict["train_parquet"] = None if cfg.train_parquet is None else str(cfg.train_parquet)
    config_dict["test_parquet"] = None if cfg.test_parquet is None else str(cfg.test_parquet)
    config_dict["reference_root"] = None if cfg.reference_root is None else str(cfg.reference_root)
    config_dict["ssft_checkpoint_dir"] = None if cfg.ssft_checkpoint_dir is None else str(cfg.ssft_checkpoint_dir)
    config_dict["update_jsonl_path"] = None if cfg.update_jsonl_path is None else str(cfg.update_jsonl_path)
    config_dict["filter_jsonl_path"] = None if cfg.filter_jsonl_path is None else str(cfg.filter_jsonl_path)

    run_metadata = {
        "config": {**config_dict, "device": str(device), "assignment_mode": assignment_mode},
        "reference_bank": {
            "num_templates": int(reference_df["template"].nunique()),
            "num_reference_images": int(len(reference_df)),
            "reference_source": reference_source_summary,
            "small_class_filtering": filtering_summary,
        },
        "targets": {
            **target_collection_summary,
            "num_target_images_requested": int(len(target_df)),
            "num_target_images_processed": int(target_images_processed),
            "assigned_known_templates": assigned_known,
            "assigned_no_template": assigned_no_template,
        },
        "clustering": {
            "cluster_method": cfg.cluster_method,
            "reducer": cfg.reducer,
            "reducer_dim_requested": int(cfg.reducer_dim),
            "reducer_dim_effective": int(effective_reducer_dim),
            "requested_kmeans_clusters": None if cfg.kmeans_clusters is None else int(cfg.kmeans_clusters),
            "requested_cluster_count_default": int(num_clusters),
            "discovered_cluster_count": int(len(centroid_ids)),
            "mapped_cluster_count": int(len(cluster_to_template)),
            "refinement_steps": int(cfg.refinement_steps),
        },
        "checkpoints": {"siglip": siglip_summary, "dino": dino_summary},
        "parameters": {"alpha": float(cfg.alpha), "tau": float(cfg.tau), "delta": float(cfg.delta)},
        "jsonl_update": jsonl_update_summary,
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")

    print(f"reference_templates={reference_df['template'].nunique()} reference_images={len(reference_df)}")
    print(f"discovered_cluster_count={len(centroid_ids)} mapped_cluster_count={len(cluster_to_template)}")
    print(f"target_images_processed={target_images_processed}")
    print(f"assigned_known_templates={assigned_known} assigned_no_template={assigned_no_template}")
    if jsonl_update_summary is not None:
        print(f"jsonl_updated_rows={jsonl_update_summary['updated_rows']}")
    print(f"predictions_saved={predictions_path}")


if __name__ == "__main__":
    main()
