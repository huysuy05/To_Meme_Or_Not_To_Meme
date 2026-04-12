#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from meme_research_eval_utils import (
    drop_small_classes,
    filter_valid_image_rows,
    load_dataset_rows,
    set_seed,
    stratified_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one canonical fixed ImgFlip train/test split so every baseline can reuse the exact same examples."
        )
    )
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--parquet-path", default=None)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--output-dir", default="SEED/splits/imgflip_fixed_split")
    parser.add_argument("--train-size", type=float, default=0.80)
    parser.add_argument("--min-images-per-class", type=int, default=7)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-templates", type=int, default=None)
    parser.add_argument("--max-images-per-template", type=int, default=None)
    parser.add_argument("--path-prefix-from", default=None)
    parser.add_argument("--path-prefix-to", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.random_seed)

    split_dir = Path(args.output_dir).expanduser().resolve()
    split_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset_rows(
        dataset_root=args.dataset_root,
        parquet_path=args.parquet_path,
        image_root=args.image_root,
        path_prefix_from=args.path_prefix_from,
        path_prefix_to=args.path_prefix_to,
        max_templates=args.max_templates,
        max_images_per_template=args.max_images_per_template,
    )
    df = filter_valid_image_rows(df)
    df, filtering_summary = drop_small_classes(df, args.min_images_per_class)
    train_df, test_df = stratified_split(df, train_size=args.train_size, random_seed=args.random_seed)

    train_path = split_dir / "train.parquet"
    test_path = split_dir / "test.parquet"
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    metadata = {
        "output_dir": str(split_dir),
        "train_parquet": str(train_path),
        "test_parquet": str(test_path),
        "random_seed": int(args.random_seed),
        "train_size": float(args.train_size),
        "dataset": {
            "images_total_after_filtering": int(len(df)),
            "classes_after_filtering": int(df["template"].nunique()),
            "train_images": int(len(train_df)),
            "test_images": int(len(test_df)),
            "small_class_filtering": filtering_summary,
        },
    }
    (split_dir / "split_config.json").write_text(json.dumps(metadata, indent=2))

    print(f"split_dir={split_dir}")
    print(f"train_parquet={train_path}")
    print(f"test_parquet={test_path}")


if __name__ == "__main__":
    main()
