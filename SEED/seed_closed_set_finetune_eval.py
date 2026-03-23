#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps, UnidentifiedImageError
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, Dinov2Model, SiglipVisionModel


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}


@dataclass(frozen=True)
class FineTuneHyperparams:
    lr: float
    epochs: int
    weight_decay: float


@dataclass(frozen=True)
class SeedHyperparams:
    alpha: float
    beta: float
    tau: float
    delta: float


@dataclass(frozen=True)
class CandidateConfig:
    siglip: FineTuneHyperparams
    dino: FineTuneHyperparams
    seed: SeedHyperparams


@dataclass
class RunConfig:
    dataset_root: Path
    output_dir: Path
    train_size: float = 0.80
    cv_folds: int = 1
    tuning_val_size: float = 0.10
    random_seed: int = 42
    batch_size: int = 16
    eval_batch_size: int = 32
    num_workers: int = 4
    train_backbone: bool = True
    closed_set: bool = True
    save_fold_checkpoints: bool = False
    max_templates: int | None = None
    max_images_per_template: int | None = None
    siglip_model_id: str = "google/siglip2-base-patch16-224"
    dino_model_id: str = "facebook/dinov2-base"
    siglip_epochs: int = 1
    dino_epochs: int = 1
    weight_decay: float = 1e-4
    siglip_lrs: tuple[float, ...] = (2e-5,)
    dino_lrs: tuple[float, ...] = (2e-5,)
    alphas: tuple[float, ...] = (0.60,)
    betas: tuple[float, ...] = (0.15,)
    taus: tuple[float, ...] = (0.85,)
    deltas: tuple[float, ...] = (0.10,)


def parse_float_list(text: str) -> tuple[float, ...]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one float value.")
    return tuple(float(value) for value in values)


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


def l2_normalize(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.clip(norms, 1e-12, None)


def load_rgb_image(path: str | Path) -> Image.Image | None:
    try:
        with Image.open(path) as img:
            return ImageOps.exif_transpose(img).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def collect_dataset_rows(
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
            [
                path
                for path in template_dir.iterdir()
                if path.is_file() and path.suffix.lower() in VALID_EXTS
            ]
        )
        if max_images_per_template is not None:
            image_paths = image_paths[:max_images_per_template]

        for image_path in image_paths:
            rows.append({"image_path": str(image_path), "template": template_dir.name})

    if not rows:
        raise ValueError(f"No images found under {dataset_root}")

    df = pd.DataFrame(rows)
    return df


def filter_valid_image_rows(df: pd.DataFrame) -> pd.DataFrame:
    keep_rows: list[bool] = []
    for image_path in tqdm(df["image_path"].tolist(), desc="verify images"):
        keep_rows.append(load_rgb_image(image_path) is not None)
    filtered = df[keep_rows].reset_index(drop=True)
    dropped = len(df) - len(filtered)
    if dropped:
        print(f"Dropped {dropped} unreadable images.")
    return filtered


def check_class_counts(
    df: pd.DataFrame,
    cv_folds: int,
    train_size: float,
    tuning_val_size: float,
) -> None:
    counts = df["template"].value_counts()
    too_small = counts[counts < 2]
    if not too_small.empty:
        sample = too_small.head(10).to_dict()
        raise ValueError(
            "At least one class has fewer than 2 images, which breaks the train/test split. "
            f"Examples: {sample}"
        )

    train_counts = np.floor(counts.to_numpy() * train_size).astype(int)
    if train_counts.min() < 2:
        low_classes = counts.iloc[np.where(train_counts < 2)[0]].head(10).to_dict()
        raise ValueError(
            "At least one class would have fewer than 2 training images after the outer split. "
            f"Examples: {low_classes}"
        )

    if cv_folds > 1:
        if train_counts.min() < cv_folds:
            low_classes = counts.iloc[np.where(train_counts < cv_folds)[0]].head(10).to_dict()
            raise ValueError(
                f"At least one class would have fewer than {cv_folds} training images after the outer split, "
                "so stratified CV cannot be run with that many folds. "
                f"Examples: {low_classes}"
            )
    else:
        inner_train_counts = np.floor(train_counts * (1.0 - tuning_val_size)).astype(int)
        inner_val_counts = train_counts - inner_train_counts
        bad_mask = (inner_train_counts < 1) | (inner_val_counts < 1)
        if bad_mask.any():
            low_classes = counts.iloc[np.where(bad_mask)[0]].head(10).to_dict()
            raise ValueError(
                "At least one class would fail the single validation split inside the training set. "
                f"Examples: {low_classes}"
            )


def minimum_images_required(train_size: float, cv_folds: int, tuning_val_size: float) -> int:
    required = 2
    while True:
        train_count = int(np.floor(required * train_size))
        if train_count < 2:
            required += 1
            continue
        if cv_folds > 1:
            if train_count >= cv_folds:
                return required
        else:
            inner_train_count = int(np.floor(train_count * (1.0 - tuning_val_size)))
            inner_val_count = train_count - inner_train_count
            if inner_train_count >= 1 and inner_val_count >= 1:
                return required
        required += 1


def drop_small_classes(
    df: pd.DataFrame,
    cv_folds: int,
    train_size: float,
    tuning_val_size: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    min_required = minimum_images_required(
        train_size=train_size,
        cv_folds=cv_folds,
        tuning_val_size=tuning_val_size,
    )
    counts = df["template"].value_counts()
    keep_templates = counts[counts >= min_required].index
    filtered = df[df["template"].isin(keep_templates)].copy().reset_index(drop=True)

    removed_templates = counts[counts < min_required]
    summary = {
        "min_images_required_per_class": int(min_required),
        "templates_before": int(counts.shape[0]),
        "templates_after": int(filtered["template"].nunique()),
        "templates_removed": int(removed_templates.shape[0]),
        "images_before": int(len(df)),
        "images_after": int(len(filtered)),
        "images_removed": int(len(df) - len(filtered)),
        "removed_template_examples": removed_templates.head(20).to_dict(),
    }
    return filtered, summary


class ImagePathDataset(Dataset):
    def __init__(self, paths: list[str], labels: list[int] | None = None):
        self.paths = paths
        self.labels = labels

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[str, int | None]:
        label = None if self.labels is None else self.labels[index]
        return self.paths[index], label


class ProcessorCollator:
    def __init__(self, processor, include_labels: bool):
        self.processor = processor
        self.include_labels = include_labels

    def __call__(self, batch: list[tuple[str, int | None]]) -> dict[str, Any] | None:
        images = []
        labels: list[int] = []
        paths: list[str] = []

        for image_path, label in batch:
            image = load_rgb_image(image_path)
            if image is None:
                continue
            images.append(image)
            paths.append(image_path)
            if self.include_labels and label is not None:
                labels.append(int(label))

        if not images:
            return None

        inputs = self.processor(images=images, return_tensors="pt")
        batch_dict: dict[str, Any] = dict(inputs)
        batch_dict["paths"] = paths
        if self.include_labels:
            batch_dict["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch_dict


def pooled_output(outputs: Any) -> torch.Tensor:
    pooled = getattr(outputs, "pooler_output", None)
    if pooled is not None:
        return pooled
    return outputs.last_hidden_state[:, 0]


class VisionClassifier(nn.Module):
    def __init__(self, backbone_kind: str, model_id: str, num_classes: int):
        super().__init__()
        self.backbone_kind = backbone_kind
        self.model_id = model_id

        if backbone_kind == "siglip":
            self.backbone = SiglipVisionModel.from_pretrained(model_id)
        elif backbone_kind == "dino":
            self.backbone = Dinov2Model.from_pretrained(model_id)
        else:
            raise ValueError(f"Unsupported backbone_kind={backbone_kind}")

        hidden_size = int(self.backbone.config.hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, labels: torch.Tensor | None = None, **inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.backbone(**inputs)
        features = pooled_output(outputs)
        logits = self.classifier(features)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits, "features": features}

    @torch.inference_mode()
    def encode_batch(self, **inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(**inputs)
        features = pooled_output(outputs)
        return F.normalize(features.float(), dim=-1)


def batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            output[key] = value.to(device)
        else:
            output[key] = value
    return output


def prepare_model_for_training(model: VisionClassifier, train_backbone: bool) -> None:
    for param in model.backbone.parameters():
        param.requires_grad = train_backbone
    for param in model.classifier.parameters():
        param.requires_grad = True


def build_loader(
    paths: list[str],
    labels: list[int] | None,
    processor,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    include_labels: bool,
) -> DataLoader:
    dataset = ImagePathDataset(paths, labels)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=ProcessorCollator(processor, include_labels=include_labels),
    )


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


def train_single_model(
    backbone_kind: str,
    model_id: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    label_names: list[str],
    hyperparams: FineTuneHyperparams,
    run_cfg: RunConfig,
    checkpoint_dir: Path,
    device: torch.device,
) -> tuple[VisionClassifier, dict[str, float]]:
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = VisionClassifier(backbone_kind=backbone_kind, model_id=model_id, num_classes=len(label_names))
    prepare_model_for_training(model, train_backbone=run_cfg.train_backbone)
    model.to(device)

    optimizer = AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=hyperparams.lr,
        weight_decay=hyperparams.weight_decay,
    )
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    train_loader = build_loader(
        paths=train_df["image_path"].tolist(),
        labels=train_df["label_idx"].tolist(),
        processor=processor,
        batch_size=run_cfg.batch_size,
        num_workers=run_cfg.num_workers,
        shuffle=True,
        include_labels=True,
    )
    val_loader = None
    if val_df is not None:
        val_loader = build_loader(
            paths=val_df["image_path"].tolist(),
            labels=val_df["label_idx"].tolist(),
            processor=processor,
            batch_size=run_cfg.eval_batch_size,
            num_workers=run_cfg.num_workers,
            shuffle=False,
            include_labels=True,
        )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{backbone_kind}_best.pt"

    best_val_acc = -1.0
    best_train_loss = float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(1, hyperparams.epochs + 1):
        model.train()
        train_losses = []
        train_accs = []

        for batch in tqdm(train_loader, desc=f"train {backbone_kind} epoch {epoch}", leave=False):
            if batch is None:
                continue

            batch = batch_to_device(batch, device)
            labels = batch.pop("labels")
            batch.pop("paths", None)

            optimizer.zero_grad(set_to_none=True)
            amp_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
                if use_amp
                else nullcontext()
            )
            with amp_context:
                outputs = model(labels=labels, **batch)
                loss = outputs["loss"]
            assert loss is not None
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(float(loss.item()))
            train_accs.append(accuracy_from_logits(outputs["logits"].detach(), labels.detach()))

        val_metrics = {"loss": float("nan"), "accuracy": float("nan")}
        if val_loader is not None:
            val_metrics = evaluate_classifier(model, val_loader, device)
        epoch_summary = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
            "train_acc": float(np.mean(train_accs)) if train_accs else float("nan"),
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
        }
        history.append(epoch_summary)
        print(
            f"[{backbone_kind}] epoch={epoch} "
            f"train_loss={epoch_summary['train_loss']:.4f} "
            f"train_acc={epoch_summary['train_acc']:.4f} "
            f"val_loss={epoch_summary['val_loss']:.4f} "
            f"val_acc={epoch_summary['val_acc']:.4f}"
        )

        should_save = False
        if val_loader is not None:
            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                should_save = True
        else:
            current_train_loss = epoch_summary["train_loss"]
            if current_train_loss < best_train_loss:
                best_train_loss = current_train_loss
                should_save = True

        if should_save:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "backbone_kind": backbone_kind,
                    "model_id": model_id,
                    "label_names": label_names,
                    "hyperparams": asdict(hyperparams),
                    "history": history,
                },
                checkpoint_path,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if not run_cfg.save_fold_checkpoints:
        pass

    return model, {
        "best_val_acc": float(best_val_acc) if val_loader is not None else float("nan"),
        "epochs": float(hyperparams.epochs),
        "lr": float(hyperparams.lr),
        "weight_decay": float(hyperparams.weight_decay),
    }


@torch.inference_mode()
def evaluate_classifier(model: VisionClassifier, data_loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    losses = []
    correct = 0
    total = 0

    for batch in data_loader:
        if batch is None:
            continue

        batch = batch_to_device(batch, device)
        labels = batch.pop("labels")
        batch.pop("paths", None)

        outputs = model(labels=labels, **batch)
        loss = outputs["loss"]
        logits = outputs["logits"]
        preds = logits.argmax(dim=-1)

        if loss is not None:
            losses.append(float(loss.item()))
        correct += int((preds == labels).sum().item())
        total += int(labels.numel())

    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "accuracy": float(correct / total) if total else 0.0,
    }


@torch.inference_mode()
def encode_paths(
    model: VisionClassifier,
    model_id: str,
    paths: list[str],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    desc: str,
) -> tuple[np.ndarray, list[str]]:
    processor = AutoImageProcessor.from_pretrained(model_id)
    data_loader = build_loader(
        paths=paths,
        labels=None,
        processor=processor,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        include_labels=False,
    )

    vectors: list[torch.Tensor] = []
    kept_paths: list[str] = []
    model.eval()

    for batch in tqdm(data_loader, desc=desc, leave=False):
        if batch is None:
            continue

        paths_batch = batch.pop("paths")
        batch = batch_to_device(batch, device)
        features = model.encode_batch(**batch)
        vectors.append(features.cpu())
        kept_paths.extend(paths_batch)

    if not vectors:
        raise ValueError(f"No valid images were encoded for {desc}")

    return torch.cat(vectors, dim=0).numpy(), kept_paths


def align_dataframe_to_kept_paths(df: pd.DataFrame, kept_paths: list[str]) -> pd.DataFrame:
    index_map = {path: idx for idx, path in enumerate(df["image_path"].tolist())}
    missing = [path for path in kept_paths if path not in index_map]
    if missing:
        raise ValueError(f"Encoded paths not found in dataframe: {missing[:5]}")
    aligned = df.iloc[[index_map[path] for path in kept_paths]].reset_index(drop=True)
    return aligned


def build_prototypes(labels: np.ndarray, embeddings: np.ndarray, num_classes: int) -> np.ndarray:
    prototypes = []
    for class_idx in range(num_classes):
        mask = labels == class_idx
        if not mask.any():
            raise ValueError(f"No training examples found for class_idx={class_idx}")
        centroid = embeddings[mask].mean(axis=0, keepdims=True)
        prototypes.append(l2_normalize(centroid)[0])
    return np.vstack(prototypes)


def fused_scores(
    target_siglip: np.ndarray,
    target_dino: np.ndarray,
    proto_siglip: np.ndarray,
    proto_dino: np.ndarray,
    alpha: float,
) -> np.ndarray:
    siglip_scores = target_siglip @ proto_siglip.T
    dino_scores = target_dino @ proto_dino.T
    return alpha * siglip_scores + (1.0 - alpha) * dino_scores


def assign_from_scores(
    scores: np.ndarray,
    tau: float,
    delta: float,
    closed_set: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    if closed_set:
        assigned_idx = best_idx.copy()
    else:
        accepted = (best_scores >= tau) & (margins >= delta)
        assigned_idx = np.where(accepted, best_idx, -1)
    return assigned_idx, best_scores, second_scores, margins


def update_prototypes(
    seed_labels: np.ndarray,
    seed_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    assigned_idx: np.ndarray,
    num_classes: int,
    beta: float,
) -> np.ndarray:
    updated = []
    for class_idx in range(num_classes):
        seed_mask = seed_labels == class_idx
        weighted_sum = seed_embeddings[seed_mask].sum(axis=0)
        total_weight = float(seed_mask.sum())

        pseudo_mask = assigned_idx == class_idx
        if pseudo_mask.any():
            weighted_sum = weighted_sum + beta * target_embeddings[pseudo_mask].sum(axis=0)
            total_weight += beta * float(pseudo_mask.sum())

        prototype = (weighted_sum / total_weight)[None, :]
        updated.append(l2_normalize(prototype)[0])
    return np.vstack(updated)


def run_seed_two_stage(
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
    train_siglip: np.ndarray,
    train_dino: np.ndarray,
    target_siglip: np.ndarray,
    target_dino: np.ndarray,
    seed_hparams: SeedHyperparams,
    num_classes: int,
    closed_set: bool,
) -> pd.DataFrame:
    train_labels = train_df["label_idx"].to_numpy()
    target_labels = target_df["label_idx"].to_numpy()

    proto_siglip = build_prototypes(train_labels, train_siglip, num_classes)
    proto_dino = build_prototypes(train_labels, train_dino, num_classes)

    phase1_scores = fused_scores(target_siglip, target_dino, proto_siglip, proto_dino, alpha=seed_hparams.alpha)
    phase1_idx, phase1_best, phase1_second, phase1_margin = assign_from_scores(
        phase1_scores,
        tau=seed_hparams.tau,
        delta=seed_hparams.delta,
        closed_set=closed_set,
    )

    updated_siglip = update_prototypes(
        seed_labels=train_labels,
        seed_embeddings=train_siglip,
        target_embeddings=target_siglip,
        assigned_idx=phase1_idx,
        num_classes=num_classes,
        beta=seed_hparams.beta,
    )
    updated_dino = update_prototypes(
        seed_labels=train_labels,
        seed_embeddings=train_dino,
        target_embeddings=target_dino,
        assigned_idx=phase1_idx,
        num_classes=num_classes,
        beta=seed_hparams.beta,
    )

    final_scores = fused_scores(target_siglip, target_dino, updated_siglip, updated_dino, alpha=seed_hparams.alpha)
    final_idx, final_best, final_second, final_margin = assign_from_scores(
        final_scores,
        tau=seed_hparams.tau,
        delta=seed_hparams.delta,
        closed_set=closed_set,
    )

    results = target_df.copy()
    results["phase1_pred_idx"] = phase1_idx
    results["phase1_best_score"] = phase1_best
    results["phase1_second_score"] = phase1_second
    results["phase1_margin"] = phase1_margin
    results["final_pred_idx"] = final_idx
    results["final_best_score"] = final_best
    results["final_second_score"] = final_second
    results["final_margin"] = final_margin
    results["phase1_correct"] = results["phase1_pred_idx"].to_numpy() == target_labels
    results["final_correct"] = results["final_pred_idx"].to_numpy() == target_labels
    return results


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_macro),
        "recall": float(recall_macro),
        "f1": float(f1_macro),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def candidate_id(candidate: CandidateConfig) -> str:
    return (
        f"siglr={candidate.siglip.lr:g}_"
        f"dinolr={candidate.dino.lr:g}_"
        f"alpha={candidate.seed.alpha:g}_"
        f"beta={candidate.seed.beta:g}_"
        f"tau={candidate.seed.tau:g}_"
        f"delta={candidate.seed.delta:g}"
    )


def build_candidates(run_cfg: RunConfig) -> list[CandidateConfig]:
    candidates = []
    for siglip_lr, dino_lr, alpha, beta, tau, delta in product(
        run_cfg.siglip_lrs,
        run_cfg.dino_lrs,
        run_cfg.alphas,
        run_cfg.betas,
        run_cfg.taus,
        run_cfg.deltas,
    ):
        candidates.append(
            CandidateConfig(
                siglip=FineTuneHyperparams(
                    lr=siglip_lr,
                    epochs=run_cfg.siglip_epochs,
                    weight_decay=run_cfg.weight_decay,
                ),
                dino=FineTuneHyperparams(
                    lr=dino_lr,
                    epochs=run_cfg.dino_epochs,
                    weight_decay=run_cfg.weight_decay,
                ),
                seed=SeedHyperparams(alpha=alpha, beta=beta, tau=tau, delta=delta),
            )
        )
    return candidates


def evaluate_candidate_on_fold(
    candidate: CandidateConfig,
    train_fold_df: pd.DataFrame,
    val_fold_df: pd.DataFrame,
    label_names: list[str],
    run_cfg: RunConfig,
    fold_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    siglip_model, siglip_train_metrics = train_single_model(
        backbone_kind="siglip",
        model_id=run_cfg.siglip_model_id,
        train_df=train_fold_df,
        val_df=val_fold_df,
        label_names=label_names,
        hyperparams=candidate.siglip,
        run_cfg=run_cfg,
        checkpoint_dir=fold_dir / "checkpoints",
        device=device,
    )
    dino_model, dino_train_metrics = train_single_model(
        backbone_kind="dino",
        model_id=run_cfg.dino_model_id,
        train_df=train_fold_df,
        val_df=val_fold_df,
        label_names=label_names,
        hyperparams=candidate.dino,
        run_cfg=run_cfg,
        checkpoint_dir=fold_dir / "checkpoints",
        device=device,
    )

    train_siglip, kept_train_siglip = encode_paths(
        siglip_model,
        run_cfg.siglip_model_id,
        train_fold_df["image_path"].tolist(),
        batch_size=run_cfg.eval_batch_size,
        num_workers=run_cfg.num_workers,
        device=device,
        desc="encode train siglip",
    )
    train_dino, kept_train_dino = encode_paths(
        dino_model,
        run_cfg.dino_model_id,
        train_fold_df["image_path"].tolist(),
        batch_size=run_cfg.eval_batch_size,
        num_workers=run_cfg.num_workers,
        device=device,
        desc="encode train dino",
    )
    val_siglip, kept_val_siglip = encode_paths(
        siglip_model,
        run_cfg.siglip_model_id,
        val_fold_df["image_path"].tolist(),
        batch_size=run_cfg.eval_batch_size,
        num_workers=run_cfg.num_workers,
        device=device,
        desc="encode val siglip",
    )
    val_dino, kept_val_dino = encode_paths(
        dino_model,
        run_cfg.dino_model_id,
        val_fold_df["image_path"].tolist(),
        batch_size=run_cfg.eval_batch_size,
        num_workers=run_cfg.num_workers,
        device=device,
        desc="encode val dino",
    )

    if kept_train_siglip != kept_train_dino:
        raise ValueError("Train embedding path mismatch between SigLIP and DINO.")
    if kept_val_siglip != kept_val_dino:
        raise ValueError("Validation embedding path mismatch between SigLIP and DINO.")

    train_fold_df = align_dataframe_to_kept_paths(train_fold_df, kept_train_siglip)
    val_fold_df = align_dataframe_to_kept_paths(val_fold_df, kept_val_siglip)

    seed_results = run_seed_two_stage(
        train_df=train_fold_df,
        target_df=val_fold_df,
        train_siglip=train_siglip,
        train_dino=train_dino,
        target_siglip=val_siglip,
        target_dino=val_dino,
        seed_hparams=candidate.seed,
        num_classes=len(label_names),
        closed_set=run_cfg.closed_set,
    )

    phase1_acc = float(seed_results["phase1_correct"].mean())
    final_acc = float(seed_results["final_correct"].mean())

    metrics = {
        "phase1_acc": phase1_acc,
        "final_acc": final_acc,
        "siglip_best_val_acc": siglip_train_metrics["best_val_acc"],
        "dino_best_val_acc": dino_train_metrics["best_val_acc"],
        "n_train": int(len(train_fold_df)),
        "n_val": int(len(val_fold_df)),
    }

    if not run_cfg.save_fold_checkpoints:
        for checkpoint_path in (fold_dir / "checkpoints").glob("*.pt"):
            checkpoint_path.unlink()

    return metrics


def select_best_candidate(
    train_df: pd.DataFrame,
    label_names: list[str],
    run_cfg: RunConfig,
    run_dir: Path,
    device: torch.device,
) -> tuple[CandidateConfig, pd.DataFrame]:
    candidates = build_candidates(run_cfg)
    results: list[dict[str, Any]] = []
    if run_cfg.cv_folds > 1:
        cv = StratifiedKFold(n_splits=run_cfg.cv_folds, shuffle=True, random_state=run_cfg.random_seed)

        for candidate in candidates:
            cand_name = candidate_id(candidate)
            print(f"Evaluating candidate: {cand_name}")
            fold_metrics: list[dict[str, Any]] = []

            for fold_idx, (train_idx, val_idx) in enumerate(
                cv.split(train_df["image_path"], train_df["label_idx"]),
                start=1,
            ):
                print(f"  Fold {fold_idx}/{run_cfg.cv_folds}")
                fold_dir = run_dir / "cv" / cand_name / f"fold_{fold_idx}"
                train_fold_df = train_df.iloc[train_idx].reset_index(drop=True)
                val_fold_df = train_df.iloc[val_idx].reset_index(drop=True)
                fold_result = evaluate_candidate_on_fold(
                    candidate=candidate,
                    train_fold_df=train_fold_df,
                    val_fold_df=val_fold_df,
                    label_names=label_names,
                    run_cfg=run_cfg,
                    fold_dir=fold_dir,
                    device=device,
                )
                fold_result["fold"] = fold_idx
                fold_result["candidate_id"] = cand_name
                results.append(
                    {
                        "candidate_id": cand_name,
                        "fold": fold_idx,
                        "phase1_acc": fold_result["phase1_acc"],
                        "final_acc": fold_result["final_acc"],
                        "siglip_best_val_acc": fold_result["siglip_best_val_acc"],
                        "dino_best_val_acc": fold_result["dino_best_val_acc"],
                        "siglip_lr": candidate.siglip.lr,
                        "dino_lr": candidate.dino.lr,
                        "alpha": candidate.seed.alpha,
                        "beta": candidate.seed.beta,
                        "tau": candidate.seed.tau,
                        "delta": candidate.seed.delta,
                    }
                )
                fold_metrics.append(fold_result)

            mean_final = float(np.mean([item["final_acc"] for item in fold_metrics]))
            mean_phase1 = float(np.mean([item["phase1_acc"] for item in fold_metrics]))
            print(f"  mean phase1 acc={mean_phase1:.4f} | mean final acc={mean_final:.4f}")
    else:
        tune_train_df, tune_val_df = train_test_split(
            train_df,
            test_size=run_cfg.tuning_val_size,
            stratify=train_df["label_idx"],
            random_state=run_cfg.random_seed,
        )
        tune_train_df = tune_train_df.reset_index(drop=True)
        tune_val_df = tune_val_df.reset_index(drop=True)
        print(
            "Using a single stratified validation split for tuning: "
            f"train={len(tune_train_df)} val={len(tune_val_df)}"
        )

        for candidate in candidates:
            cand_name = candidate_id(candidate)
            print(f"Evaluating candidate: {cand_name}")
            split_dir = run_dir / "single_split" / cand_name
            split_result = evaluate_candidate_on_fold(
                candidate=candidate,
                train_fold_df=tune_train_df,
                val_fold_df=tune_val_df,
                label_names=label_names,
                run_cfg=run_cfg,
                fold_dir=split_dir,
                device=device,
            )
            results.append(
                {
                    "candidate_id": cand_name,
                    "fold": 1,
                    "phase1_acc": split_result["phase1_acc"],
                    "final_acc": split_result["final_acc"],
                    "siglip_best_val_acc": split_result["siglip_best_val_acc"],
                    "dino_best_val_acc": split_result["dino_best_val_acc"],
                    "siglip_lr": candidate.siglip.lr,
                    "dino_lr": candidate.dino.lr,
                    "alpha": candidate.seed.alpha,
                    "beta": candidate.seed.beta,
                    "tau": candidate.seed.tau,
                    "delta": candidate.seed.delta,
                }
            )
            print(
                f"  split phase1 acc={split_result['phase1_acc']:.4f} | "
                f"split final acc={split_result['final_acc']:.4f}"
            )

    results_df = pd.DataFrame(results)
    if results_df.empty:
        raise ValueError("No CV results were produced.")

    summary = (
        results_df.groupby("candidate_id", as_index=False)
        .agg(
            mean_phase1_acc=("phase1_acc", "mean"),
            mean_final_acc=("final_acc", "mean"),
            siglip_lr=("siglip_lr", "first"),
            dino_lr=("dino_lr", "first"),
            alpha=("alpha", "first"),
            beta=("beta", "first"),
            tau=("tau", "first"),
            delta=("delta", "first"),
        )
        .sort_values(["mean_final_acc", "mean_phase1_acc"], ascending=False)
        .reset_index(drop=True)
    )
    summary.to_csv(run_dir / "cv_summary.csv", index=False)
    results_df.to_csv(run_dir / "cv_fold_results.csv", index=False)

    best_row = summary.iloc[0]
    best_candidate = CandidateConfig(
        siglip=FineTuneHyperparams(
            lr=float(best_row["siglip_lr"]),
            epochs=run_cfg.siglip_epochs,
            weight_decay=run_cfg.weight_decay,
        ),
        dino=FineTuneHyperparams(
            lr=float(best_row["dino_lr"]),
            epochs=run_cfg.dino_epochs,
            weight_decay=run_cfg.weight_decay,
        ),
        seed=SeedHyperparams(
            alpha=float(best_row["alpha"]),
            beta=float(best_row["beta"]),
            tau=float(best_row["tau"]),
            delta=float(best_row["delta"]),
        ),
    )
    return best_candidate, summary


def retrain_and_evaluate(
    best_candidate: CandidateConfig,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_names: list[str],
    run_cfg: RunConfig,
    run_dir: Path,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    full_dir = run_dir / "final_train"
    siglip_model, siglip_metrics = train_single_model(
        backbone_kind="siglip",
        model_id=run_cfg.siglip_model_id,
        train_df=train_df,
        val_df=None,
        label_names=label_names,
        hyperparams=best_candidate.siglip,
        run_cfg=run_cfg,
        checkpoint_dir=full_dir / "checkpoints",
        device=device,
    )
    dino_model, dino_metrics = train_single_model(
        backbone_kind="dino",
        model_id=run_cfg.dino_model_id,
        train_df=train_df,
        val_df=None,
        label_names=label_names,
        hyperparams=best_candidate.dino,
        run_cfg=run_cfg,
        checkpoint_dir=full_dir / "checkpoints",
        device=device,
    )

    train_siglip, kept_train_siglip = encode_paths(
        siglip_model,
        run_cfg.siglip_model_id,
        train_df["image_path"].tolist(),
        batch_size=run_cfg.eval_batch_size,
        num_workers=run_cfg.num_workers,
        device=device,
        desc="encode final train siglip",
    )
    train_dino, kept_train_dino = encode_paths(
        dino_model,
        run_cfg.dino_model_id,
        train_df["image_path"].tolist(),
        batch_size=run_cfg.eval_batch_size,
        num_workers=run_cfg.num_workers,
        device=device,
        desc="encode final train dino",
    )
    test_siglip, kept_test_siglip = encode_paths(
        siglip_model,
        run_cfg.siglip_model_id,
        test_df["image_path"].tolist(),
        batch_size=run_cfg.eval_batch_size,
        num_workers=run_cfg.num_workers,
        device=device,
        desc="encode final test siglip",
    )
    test_dino, kept_test_dino = encode_paths(
        dino_model,
        run_cfg.dino_model_id,
        test_df["image_path"].tolist(),
        batch_size=run_cfg.eval_batch_size,
        num_workers=run_cfg.num_workers,
        device=device,
        desc="encode final test dino",
    )

    if kept_train_siglip != kept_train_dino:
        raise ValueError("Final-train embedding path mismatch between SigLIP and DINO.")
    if kept_test_siglip != kept_test_dino:
        raise ValueError("Final-test embedding path mismatch between SigLIP and DINO.")

    train_df = align_dataframe_to_kept_paths(train_df, kept_train_siglip)
    test_df = align_dataframe_to_kept_paths(test_df, kept_test_siglip)

    results_df = run_seed_two_stage(
        train_df=train_df,
        target_df=test_df,
        train_siglip=train_siglip,
        train_dino=train_dino,
        target_siglip=test_siglip,
        target_dino=test_dino,
        seed_hparams=best_candidate.seed,
        num_classes=len(label_names),
        closed_set=run_cfg.closed_set,
    )

    idx_to_label = {idx: label for idx, label in enumerate(label_names)}
    results_df["true_label"] = results_df["label_idx"].map(idx_to_label)
    results_df["phase1_pred_label"] = results_df["phase1_pred_idx"].map(idx_to_label)
    results_df["final_pred_label"] = results_df["final_pred_idx"].map(idx_to_label)
    results_df.to_csv(run_dir / "test_predictions.csv", index=False)

    y_true = results_df["label_idx"].to_numpy()
    y_phase1 = results_df["phase1_pred_idx"].to_numpy()
    y_final = results_df["final_pred_idx"].to_numpy()

    metrics = {
        "phase1_metrics": compute_classification_metrics(y_true, y_phase1),
        "final_metrics": compute_classification_metrics(y_true, y_final),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "num_classes": int(len(label_names)),
        "siglip_best_val_acc": siglip_metrics["best_val_acc"],
        "dino_best_val_acc": dino_metrics["best_val_acc"],
        "best_candidate": {
            "siglip": asdict(best_candidate.siglip),
            "dino": asdict(best_candidate.dino),
            "seed": asdict(best_candidate.seed),
        },
    }
    return results_df, metrics


def build_run_config(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        dataset_root=Path(args.dataset_root).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        train_size=args.train_size,
        cv_folds=args.cv_folds,
        tuning_val_size=args.tuning_val_size,
        random_seed=args.random_seed,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        train_backbone=not args.head_only,
        closed_set=not args.allow_rejection,
        save_fold_checkpoints=args.save_fold_checkpoints,
        max_templates=args.max_templates,
        max_images_per_template=args.max_images_per_template,
        siglip_model_id=args.siglip_model_id,
        dino_model_id=args.dino_model_id,
        siglip_epochs=args.siglip_epochs,
        dino_epochs=args.dino_epochs,
        weight_decay=args.weight_decay,
        siglip_lrs=args.siglip_lrs,
        dino_lrs=args.dino_lrs,
        alphas=args.alphas,
        betas=args.betas,
        taus=args.taus,
        deltas=args.deltas,
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune SigLIP2 and DINOv2, then evaluate the two-stage SEED method "
            "in a closed-set ImgFlip-style template classification setting."
        )
    )
    parser.add_argument("--dataset-root", required=True, help="Folder-of-folders dataset root.")
    parser.add_argument(
        "--output-dir",
        default="SEED/runs/seed_closed_set_finetune_eval",
        help="Directory for checkpoints, metrics, and predictions.",
    )
    parser.add_argument("--train-size", type=float, default=0.80)
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=1,
        help="Number of stratified tuning folds. Use 1 for a single validation split.",
    )
    parser.add_argument(
        "--tuning-val-size",
        type=float,
        default=0.20,
        help="Validation fraction used inside the training split when --cv-folds=1.",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--siglip-model-id", default="google/siglip2-base-patch16-224")
    parser.add_argument("--dino-model-id", default="facebook/dinov2-base")
    parser.add_argument("--siglip-epochs", type=int, default=1)
    parser.add_argument("--dino-epochs", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--siglip-lrs", type=parse_float_list, default=(2e-5,))
    parser.add_argument("--dino-lrs", type=parse_float_list, default=(2e-5,))
    parser.add_argument("--alphas", type=parse_float_list, default=(0.60,))
    parser.add_argument("--betas", type=parse_float_list, default=(0.15,))
    parser.add_argument("--taus", type=parse_float_list, default=(0.85,))
    parser.add_argument("--deltas", type=parse_float_list, default=(0.10,))
    parser.add_argument("--max-templates", type=int, default=None)
    parser.add_argument("--max-images-per-template", type=int, default=None)
    parser.add_argument(
        "--head-only",
        action="store_true",
        help="Freeze the backbones and only train the classifier heads.",
    )
    parser.add_argument(
        "--allow-rejection",
        action="store_true",
        help="Enable open-set rejection with tau/delta instead of closed-set top-1 assignment.",
    )
    parser.add_argument(
        "--save-fold-checkpoints",
        action="store_true",
        help="Keep fold-level checkpoints instead of deleting them after each CV fold.",
    )
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    run_cfg = build_run_config(args)

    set_seed(run_cfg.random_seed)
    device = pick_device()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = run_cfg.output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"device={device}")
    print(f"run_dir={run_dir}")

    dataset_df = collect_dataset_rows(
        dataset_root=run_cfg.dataset_root,
        max_templates=run_cfg.max_templates,
        max_images_per_template=run_cfg.max_images_per_template,
    )
    dataset_df = filter_valid_image_rows(dataset_df)
    dataset_df, filtering_summary = drop_small_classes(
        dataset_df,
        cv_folds=run_cfg.cv_folds,
        train_size=run_cfg.train_size,
        tuning_val_size=run_cfg.tuning_val_size,
    )
    if filtering_summary["templates_removed"] > 0:
        print(
            "Dropped classes that are too small for the requested split/CV: "
            f"{filtering_summary['templates_removed']} templates, "
            f"{filtering_summary['images_removed']} images removed. "
            f"Minimum images per class required={filtering_summary['min_images_required_per_class']}"
        )
    else:
        print(
            "No classes were dropped for size. "
            f"Minimum images per class required={filtering_summary['min_images_required_per_class']}"
        )

    label_names = sorted(dataset_df["template"].unique().tolist())
    label_to_idx = {label: idx for idx, label in enumerate(label_names)}
    dataset_df["label_idx"] = dataset_df["template"].map(label_to_idx)
    check_class_counts(
        dataset_df,
        cv_folds=run_cfg.cv_folds,
        train_size=run_cfg.train_size,
        tuning_val_size=run_cfg.tuning_val_size,
    )

    train_df, test_df = train_test_split(
        dataset_df,
        train_size=run_cfg.train_size,
        stratify=dataset_df["label_idx"],
        random_state=run_cfg.random_seed,
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    metadata = {
        "run_config": {
            **asdict(run_cfg),
            "dataset_root": str(run_cfg.dataset_root),
            "output_dir": str(run_cfg.output_dir),
            "device": str(device),
        },
        "dataset": {
            "num_images_total": int(len(dataset_df)),
            "num_classes": int(len(label_names)),
            "num_images_train": int(len(train_df)),
            "num_images_test": int(len(test_df)),
            "small_class_filtering": filtering_summary,
        },
    }
    (run_dir / "run_config.json").write_text(json.dumps(metadata, indent=2))

    best_candidate, cv_summary = select_best_candidate(
        train_df=train_df,
        label_names=label_names,
        run_cfg=run_cfg,
        run_dir=run_dir,
        device=device,
    )

    results_df, test_metrics = retrain_and_evaluate(
        best_candidate=best_candidate,
        train_df=train_df,
        test_df=test_df,
        label_names=label_names,
        run_cfg=run_cfg,
        run_dir=run_dir,
        device=device,
    )

    metrics_payload = {
        "dataset": metadata["dataset"],
        "closed_set": run_cfg.closed_set,
        "cv_best_candidate": {
            "siglip": asdict(best_candidate.siglip),
            "dino": asdict(best_candidate.dino),
            "seed": asdict(best_candidate.seed),
        },
        "cv_summary_top5": cv_summary.head(5).to_dict(orient="records"),
        "test_metrics": test_metrics,
    }
    (run_dir / "test_metrics.json").write_text(json.dumps(metrics_payload, indent=2))

    print(f"num_classes={len(label_names)}")
    print(f"train_images={len(train_df)} test_images={len(test_df)}")
    print(f"best_candidate={metrics_payload['cv_best_candidate']}")
    print(f"phase1_test_accuracy={test_metrics['phase1_metrics']['accuracy']:.4f}")
    print(f"final_test_accuracy={test_metrics['final_metrics']['accuracy']:.4f}")
    print(f"final_test_f1={test_metrics['final_metrics']['f1']:.4f}")
    print(f"final_test_precision={test_metrics['final_metrics']['precision']:.4f}")
    print(f"final_test_recall={test_metrics['final_metrics']['recall']:.4f}")
    print(f"final_test_mcc={test_metrics['final_metrics']['mcc']:.4f}")
    print(f"final_test_cohen_kappa={test_metrics['final_metrics']['cohen_kappa']:.4f}")
    print(f"predictions_saved={run_dir / 'test_predictions.csv'}")


if __name__ == "__main__":
    main()
