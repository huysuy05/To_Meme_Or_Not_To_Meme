#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


NO_TEMPLATE = "NO_TEMPLATE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge the enriched meme JSONL with Reddit metadata parquet into the analysis table."
    )
    parser.add_argument(
        "--jsonl-path",
        default="data/merged_parsed_results_with_template_predictions.jsonl",
        help="Annotated JSONL path. Prefer the enriched JSONL with cluster/template prediction fields already merged in.",
    )
    parser.add_argument(
        "--metadata-parquet",
        default="Reddit2024_nolabel/subreddits23/meme_submissions.zst.parquet",
        help="Reddit metadata parquet path.",
    )
    parser.add_argument(
        "--output-parquet",
        default="data/template_first_analysis_table.parquet",
        help="Merged analysis table output path.",
    )
    parser.add_argument(
        "--output-summary-json",
        default="data/template_first_analysis_table_summary.json",
        help="Summary metadata output path.",
    )
    parser.add_argument(
        "--label-mode",
        choices=["strict_imgflip", "annotation_fallback"],
        default="strict_imgflip",
        help=(
            "How to derive template_final. "
            "`strict_imgflip` uses only pred_template/NO_TEMPLATE. "
            "`annotation_fallback` falls back to template_original when prediction is NO_TEMPLATE."
        ),
    )
    return parser.parse_args()


def _json_list_to_text(value: Any) -> str:
    if isinstance(value, list):
        return " | ".join(str(item) for item in value if item is not None and str(item).strip())
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _json_list_to_json(value: Any) -> str:
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=True)
    if value is None:
        return "[]"
    if isinstance(value, str):
        return json.dumps([value], ensure_ascii=True)
    return json.dumps([str(value)], ensure_ascii=True)


def load_annotations(jsonl_path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as infile:
        for line_number, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            data = payload.get("data")
            if not isinstance(data, dict):
                continue
            local_context = data.get("local_context", {})
            if not isinstance(local_context, dict):
                local_context = {}
            prediction = data.get("template_prediction", {})
            if not isinstance(prediction, dict):
                prediction = {}
            rows.append(
                {
                    "key": str(payload.get("key")),
                    "template_original": data.get("template_original", data.get("template")),
                    "pred_template": data.get("pred_template"),
                    "template_final_existing": data.get("template_final"),
                    "template_source_existing": data.get("template_source"),
                    "global_context_description": data.get("global_context_description"),
                    "local_context_user_texts_json": _json_list_to_json(local_context.get("user_texts")),
                    "local_context_user_texts_text": _json_list_to_text(local_context.get("user_texts")),
                    "local_context_text_meaning": local_context.get("text_meaning"),
                    "local_context_instance_specific_image_description": local_context.get(
                        "instance_specific_image_description"
                    ),
                    "global_context_keywords_json": _json_list_to_json(data.get("global_context_keywords")),
                    "global_context_keywords_text": _json_list_to_text(data.get("global_context_keywords")),
                    "local_context_keywords_json": _json_list_to_json(data.get("local_context_keywords")),
                    "local_context_keywords_text": _json_list_to_text(data.get("local_context_keywords")),
                    "image_path": prediction.get("image_path"),
                    "best_template_name": prediction.get("best_template_name"),
                    "matched_known_template": prediction.get("matched_known_template"),
                    "best_score": prediction.get("best_score"),
                    "second_score": prediction.get("second_score"),
                    "margin": prediction.get("margin"),
                    "siglip_best_score": prediction.get("siglip_best_score"),
                    "dino_best_score": prediction.get("dino_best_score"),
                    "assignment_method": prediction.get("assignment_method"),
                    "cluster_method": prediction.get("cluster_method"),
                    "reducer": prediction.get("reducer"),
                }
            )
    if not rows:
        raise ValueError(f"No annotation rows loaded from {jsonl_path}")
    return pd.DataFrame(rows)


def load_metadata(metadata_parquet: Path) -> pd.DataFrame:
    columns = ["id", "score", "title", "body", "url", "image_url", "post_link", "created_utc"]
    available = pd.read_parquet(metadata_parquet, columns=columns)
    return available.rename(columns={"id": "key"})


def compute_template_fields(df: pd.DataFrame, label_mode: str) -> pd.DataFrame:
    merged = df.copy()
    merged["template_original"] = merged["template_original"].fillna(NO_TEMPLATE).astype(str)
    merged["pred_template"] = merged["pred_template"].fillna(NO_TEMPLATE).astype(str)

    if label_mode == "strict_imgflip":
        merged["template_final"] = merged["pred_template"]
        merged["template_source"] = merged["pred_template"].map(
            lambda value: "imgflip_closed_set_assignment" if value != NO_TEMPLATE else "imgflip_closed_set_no_template"
        )
    else:
        merged["template_final"] = merged.apply(
            lambda row: row["pred_template"] if row["pred_template"] != NO_TEMPLATE else row["template_original"],
            axis=1,
        )
        merged["template_source"] = merged.apply(
            lambda row: "imgflip_assignment"
            if row["pred_template"] != NO_TEMPLATE
            else ("annotation" if row["template_original"] != NO_TEMPLATE else "unresolved"),
            axis=1,
        )
    return merged


def main() -> None:
    args = parse_args()
    jsonl_path = Path(args.jsonl_path).expanduser().resolve()
    metadata_parquet = Path(args.metadata_parquet).expanduser().resolve()
    output_parquet = Path(args.output_parquet).expanduser().resolve()
    output_summary_json = Path(args.output_summary_json).expanduser().resolve()

    annotations = load_annotations(jsonl_path)
    metadata = load_metadata(metadata_parquet)
    merged = annotations.merge(metadata, on="key", how="left", validate="one_to_one")
    merged = compute_template_fields(merged, label_mode=args.label_mode)
    merged["created_utc"] = pd.to_datetime(merged["created_utc"], errors="coerce", utc=False)

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_parquet, index=False)

    summary = {
        "rows": int(len(merged)),
        "unique_keys": int(merged["key"].nunique()),
        "rows_with_metadata": int(merged["score"].notna().sum()),
        "rows_with_predictions": int(merged["pred_template"].ne(NO_TEMPLATE).sum()),
        "label_mode": str(args.label_mode),
        "unique_templates_original": int(merged["template_original"].nunique()),
        "unique_templates_final": int(merged["template_final"].nunique()),
        "final_no_template_rows": int((merged["template_final"] == NO_TEMPLATE).sum()),
        "time_min": None if merged["created_utc"].dropna().empty else str(merged["created_utc"].min()),
        "time_max": None if merged["created_utc"].dropna().empty else str(merged["created_utc"].max()),
        "source_paths": {
            "jsonl_path": str(jsonl_path),
            "metadata_parquet": str(metadata_parquet),
            "output_parquet": str(output_parquet),
        },
    }
    output_summary_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
