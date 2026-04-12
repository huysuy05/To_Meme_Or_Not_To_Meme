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
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from meme_research_eval_utils import (
    compute_metrics,
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


NORMALIZE_MEAN = [0.5325, 0.4980, 0.4715]
NORMALIZE_STD = [0.3409, 0.3384, 0.3465]


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
    batch_size: int = 32
    eval_batch_size: int = 64
    num_workers: int = 4
    model_name: str = "densenet121"
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    epochs: int = 10
    patience: int = 2
    freeze_backbone: bool = True
    max_templates: int | None = None
    max_images_per_template: int | None = None
    path_prefix_from: str | None = None
    path_prefix_to: str | None = None
    seeds: tuple[int, ...] | None = None


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Remote-friendly closed-set ImgFlip evaluation for the meme-research multiclass CNN. "
            "This mirrors the old transfer-learning template classifier without notebook or Lightning dependencies."
        )
    )
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--parquet-path", default=None)
    parser.add_argument("--train-parquet", default=None)
    parser.add_argument("--test-parquet", default=None)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--output-dir", default="SEED/runs/meme_research_cnn_eval")
    parser.add_argument("--train-size", type=float, default=0.80)
    parser.add_argument("--val-size", type=float, default=0.10)
    parser.add_argument("--min-images-per-class", type=int, default=7)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--model-name", choices=["resnet18", "densenet121", "efficientnet_v2_s"], default="densenet121")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--freeze-backbone", dest="freeze_backbone", action="store_true")
    parser.add_argument("--train-backbone", dest="freeze_backbone", action="store_false")
    parser.set_defaults(freeze_backbone=True)
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
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        freeze_backbone=args.freeze_backbone,
        max_templates=args.max_templates,
        max_images_per_template=args.max_images_per_template,
        path_prefix_from=args.path_prefix_from,
        path_prefix_to=args.path_prefix_to,
        seeds=args.seeds,
    )


class ImgFlipDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_to_idx: dict[str, int], transform):
        self.df = df.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        row = self.df.iloc[index]
        image = load_rgb_image(row["image_path"])
        if image is None:
            raise ValueError(f"Unreadable image slipped through filtering: {row['image_path']}")
        return self.transform(image), self.label_to_idx[row["template"]], row["image_path"]


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=(0, 10)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0)),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ])
    return train_transform, eval_transform


class TransferLearningClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int, freeze_backbone: bool) -> None:
        super().__init__()
        if model_name == "resnet18":
            backbone = models.resnet18(weights="DEFAULT")
            input_shape = (224, 224)
        elif model_name == "densenet121":
            backbone = models.densenet121(weights="DEFAULT")
            input_shape = (224, 224)
        elif model_name == "efficientnet_v2_s":
            backbone = models.efficientnet_v2_s(weights="DEFAULT")
            input_shape = (224, 224)
        else:
            raise ValueError(f"Unsupported model_name={model_name}")

        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        with torch.no_grad():
            dummy = torch.rand(1, 3, *input_shape)
            feature_dim = int(self.feature_extractor(dummy).reshape(1, -1).shape[1])
        self.classifier = nn.Linear(feature_dim, num_classes)

        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        return self.classifier(features.reshape(features.shape[0], -1))


def make_loader(dataset: Dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    losses: list[float] = []
    all_probs: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_paths: list[str] = []

    model.eval()
    with torch.inference_mode():
        for images, targets, paths in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            loss = loss_fn(logits, targets)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            losses.append(float(loss.item()) * images.shape[0])
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_paths.extend(paths)

    total = max(len(all_paths), 1)
    return (
        float(sum(losses) / total),
        np.concatenate(all_probs, axis=0),
        np.concatenate(all_preds, axis=0),
        np.concatenate(all_targets, axis=0),
        all_paths,
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
    train_df, val_df = train_test_split(
        train_df,
        test_size=cfg.val_size,
        stratify=train_df["template"],
        random_state=cfg.random_seed,
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    templates = sorted(train_df["template"].unique().tolist())
    label_to_idx = {label: idx for idx, label in enumerate(templates)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    train_transform, eval_transform = build_transforms()
    train_loader = make_loader(
        ImgFlipDataset(train_df, label_to_idx, train_transform),
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=True,
    )
    val_loader = make_loader(
        ImgFlipDataset(val_df, label_to_idx, eval_transform),
        batch_size=cfg.eval_batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
    )
    test_loader = make_loader(
        ImgFlipDataset(test_df, label_to_idx, eval_transform),
        batch_size=cfg.eval_batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
    )

    model = TransferLearningClassifier(
        model_name=cfg.model_name,
        num_classes=len(label_to_idx),
        freeze_backbone=cfg.freeze_backbone,
    ).to(device)
    loss_fn = nn.CrossEntropyLoss()
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    best_state = None
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        seen = 0
        progress = tqdm(train_loader, desc=f"train epoch {epoch + 1}/{cfg.epochs}")
        for images, targets, _ in progress:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * images.shape[0]
            seen += int(images.shape[0])
            progress.set_postfix(train_loss=f"{running_loss / max(seen, 1):.4f}")

        train_loss = running_loss / max(seen, 1)
        val_loss, _, val_preds, val_targets, _ = evaluate_model(model, val_loader, device, loss_fn)
        val_metrics = compute_metrics(
            np.array([idx_to_label[int(i)] for i in val_targets]),
            np.array([idx_to_label[int(i)] for i in val_preds]),
        )
        epoch_summary = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_f1": float(val_metrics["f1"]),
        }
        history.append(epoch_summary)
        print(epoch_summary)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.patience:
                print(f"Early stopping at epoch {epoch + 1}.")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    torch.save(best_state, run_dir / "best_model.pt")

    test_loss, test_probs, test_preds, test_targets, test_paths = evaluate_model(model, test_loader, device, loss_fn)
    test_true_labels = np.array([idx_to_label[int(i)] for i in test_targets])
    test_pred_labels = np.array([idx_to_label[int(i)] for i in test_preds])
    test_confidences = test_probs.max(axis=1)
    metrics = compute_metrics(test_true_labels, test_pred_labels)

    predictions_df = test_df.copy()
    predictions_df["pred_template"] = test_pred_labels
    predictions_df["confidence"] = test_confidences
    predictions_df["correct"] = predictions_df["template"].to_numpy() == predictions_df["pred_template"].to_numpy()
    predictions_df = predictions_df.loc[:, ["image_path", "template", "pred_template", "confidence", "correct", "source"]]
    predictions_df.to_csv(run_dir / "test_predictions.csv", index=False)

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
            "val_images": int(len(val_df)),
            "test_images": int(len(test_df)),
            "small_class_filtering": filtering_summary,
        },
        "training": {
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss),
            "history": history,
        },
        "test_metrics": {
            **metrics,
            "test_loss": float(test_loss),
        },
    }
    summary_row = {
        "run_dir": str(run_dir),
        "seed": int(cfg.random_seed),
        "model_name": cfg.model_name,
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "mcc": float(metrics["mcc"]),
        "cohen_kappa": float(metrics["cohen_kappa"]),
        "test_loss": float(test_loss),
    }
    timing = finalize_run_timing(metadata, summary_row, run_started_at, start_perf)
    (run_dir / "run_config.json").write_text(json.dumps(metadata, indent=2))
    (run_dir / "test_metrics.json").write_text(
        json.dumps({"test_metrics": metadata["test_metrics"], "timing": timing}, indent=2)
    )
    pd.DataFrame([summary_row]).to_csv(run_dir / "metrics_summary.csv", index=False)

    print(f"classes={len(label_to_idx)}")
    print(f"test_accuracy={metrics['accuracy']:.4f}")
    print(f"test_f1={metrics['f1']:.4f}")
    print(f"test_mcc={metrics['mcc']:.4f}")
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
