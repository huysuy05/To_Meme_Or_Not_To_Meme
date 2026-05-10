#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NO_TEMPLATE = "NO_TEMPLATE"
EXCLUDED_TEMPLATE_LABELS = {NO_TEMPLATE, "NON_MEME", "NO_MEME"}
LOCAL_DOWNLOADS_DIRNAME = "LOCAL_DOWNLOADS"
AFFECT_EMOTION_LABELS = [
    "joy",
    "anticipation",
    "disgust",
    "sadness",
    "anger",
    "optimism",
    "surprise",
    "pessimism",
    "fear",
    "trust",
    "love",
]
AFFECT_SENTIMENT_LABELS = ["neutral", "positive", "negative"]


def configure_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.figsize": (10.5, 5.5),
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def try_import_sentence_transformers() -> Any | None:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        return SentenceTransformer
    except Exception:
        return None


def try_import_vader() -> Any | None:
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore

        return SentimentIntensityAnalyzer
    except Exception:
        return None


def parse_json_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [part.strip() for part in text.split("|") if part.strip()]
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    if parsed is None:
        return []
    return [str(parsed).strip()]


def build_text(*parts: Any) -> str:
    values: list[str] = []
    for part in parts:
        if part is None:
            continue
        if isinstance(part, float) and np.isnan(part):
            continue
        text = str(part).strip()
        if text:
            values.append(text)
    return " ".join(values)


def safe_log1p(values: pd.Series) -> pd.Series:
    return np.log1p(pd.to_numeric(values, errors="coerce").fillna(0).clip(lower=0))


def normalize_template_name(name: str) -> str:
    lowered = str(name).lower()
    return re.sub(r"[^a-z0-9]+", "", lowered)


def prepare_analysis_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "created_utc" not in df.columns:
        raise ValueError("Missing required column: created_utc")
    if "template_final" not in df.columns:
        raise ValueError("Missing required column: template_final")

    df["created_utc"] = pd.to_datetime(df["created_utc"], errors="coerce", utc=True).dt.tz_convert(None)
    df["score"] = pd.to_numeric(df.get("score"), errors="coerce").fillna(0.0)
    if "num_comments" in df.columns:
        df["num_comments"] = pd.to_numeric(df["num_comments"], errors="coerce")
    else:
        df["num_comments"] = np.nan

    for column, out_column in [
        ("global_context_keywords_json", "global_keywords_list"),
        ("local_context_keywords_json", "local_keywords_list"),
        ("local_context_user_texts_json", "user_text_list"),
    ]:
        if column in df.columns:
            df[out_column] = df[column].map(parse_json_list)
        else:
            df[out_column] = [[] for _ in range(len(df))]

    df["global_text"] = df.apply(
        lambda row: build_text(
            row.get("global_context_description", ""),
            row.get("global_context_keywords_text", ""),
        ),
        axis=1,
    )
    df["local_text"] = df.apply(
        lambda row: build_text(
            row.get("local_context_user_texts_text", ""),
            row.get("local_context_text_meaning", ""),
            row.get("local_context_instance_specific_image_description", ""),
            row.get("local_context_keywords_text", ""),
            row.get("title", ""),
            row.get("body", ""),
        ),
        axis=1,
    )
    df["template_final"] = df["template_final"].fillna(NO_TEMPLATE).astype(str)
    df = df[df["template_final"].notna()].copy()
    df = df[~df["template_final"].isin(EXCLUDED_TEMPLATE_LABELS)].copy()
    df = df[df["created_utc"].notna()].copy()

    df["year_month"] = df["created_utc"].dt.to_period("M")
    df["week_start"] = df["created_utc"].dt.to_period("W").dt.start_time
    df["year"] = df["created_utc"].dt.year.astype("Int64")
    df["weekday"] = df["created_utc"].dt.weekday.astype("Int64")
    df["hour"] = df["created_utc"].dt.hour.astype("Int64")
    df["log_score_pos"] = safe_log1p(df["score"])
    df["month_score_pct"] = df.groupby(["year_month"])["score"].rank(method="average", pct=True)
    return df.reset_index(drop=True)


def load_analysis_dataframe(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(Path(path).expanduser().resolve())
    return prepare_analysis_dataframe(df)


def image_basename_from_row(row: pd.Series) -> str:
    image_path = row.get("image_path")
    if isinstance(image_path, str) and image_path.strip():
        return Path(image_path).name
    key = row.get("key")
    if isinstance(key, str) and key.strip():
        return f"{key}.jpg"
    return ""


def _read_two_column_csv(path: Path, col1: str, col2: str) -> pd.DataFrame:
    frame = pd.read_csv(path, header=None, names=[col1, col2])
    frame[col1] = frame[col1].astype(str)
    frame[col2] = frame[col2].fillna("").astype(str)
    frame = frame[frame[col1].str.strip().ne("")].copy()
    frame = frame.drop_duplicates(subset=[col1], keep="first").reset_index(drop=True)
    return frame


def _resolve_local_downloads_dir(data_dir: str | Path) -> Path:
    base_dir = Path(data_dir).expanduser().resolve()
    if base_dir.name == LOCAL_DOWNLOADS_DIRNAME:
        return base_dir
    return base_dir / LOCAL_DOWNLOADS_DIRNAME


def _bundle_key_from_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    return Path(text).stem.strip()


def _key_to_bundle_filename(key: str) -> str:
    normalized = str(key).strip()
    if not normalized:
        return ""
    return f"{normalized}.jpg"


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


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


def _resolve_parsed_context_jsonl(data_dir: str | Path) -> Path:
    base_dir = Path(data_dir).expanduser().resolve()
    if base_dir.name == LOCAL_DOWNLOADS_DIRNAME:
        return base_dir.parent / "merged_parsed_results_with_template_predictions.jsonl"
    return base_dir / "merged_parsed_results_with_template_predictions.jsonl"


def _load_parsed_context_frame(data_dir: str | Path) -> pd.DataFrame:
    jsonl_path = _resolve_parsed_context_jsonl(data_dir)
    if not jsonl_path.exists():
        columns = [
            "bundle_key",
            "global_context_description",
            "local_context_user_texts_json",
            "local_context_user_texts_text",
            "local_context_text_meaning",
            "local_context_instance_specific_image_description",
            "global_context_keywords_json",
            "global_context_keywords_text",
            "local_context_keywords_json",
            "local_context_keywords_text",
        ]
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as infile:
        for line in infile:
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
            bundle_key = str(payload.get("key", "")).strip()
            if not bundle_key:
                continue
            rows.append(
                {
                    "bundle_key": bundle_key,
                    "global_context_description": data.get("global_context_description", ""),
                    "local_context_user_texts_json": _json_list_to_json(local_context.get("user_texts")),
                    "local_context_user_texts_text": _json_list_to_text(local_context.get("user_texts")),
                    "local_context_text_meaning": local_context.get("text_meaning", ""),
                    "local_context_instance_specific_image_description": local_context.get(
                        "instance_specific_image_description",
                        "",
                    ),
                    "global_context_keywords_json": _json_list_to_json(data.get("global_context_keywords")),
                    "global_context_keywords_text": _json_list_to_text(data.get("global_context_keywords")),
                    "local_context_keywords_json": _json_list_to_json(data.get("local_context_keywords")),
                    "local_context_keywords_text": _json_list_to_text(data.get("local_context_keywords")),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "bundle_key",
                "global_context_description",
                "local_context_user_texts_json",
                "local_context_user_texts_text",
                "local_context_text_meaning",
                "local_context_instance_specific_image_description",
                "global_context_keywords_json",
                "global_context_keywords_text",
                "local_context_keywords_json",
                "local_context_keywords_text",
            ]
        )
    return pd.DataFrame(rows).drop_duplicates(subset=["bundle_key"], keep="last").reset_index(drop=True)


def _load_local_downloads_sentiment_frame(base_dir: Path, suffix: str) -> pd.DataFrame:
    payload = _load_json_object(base_dir / f"{suffix}_sentiments.json")
    rows: list[dict[str, Any]] = []
    for key, values in payload.items():
        bundle_key = str(key).strip()
        if not bundle_key or not isinstance(values, dict):
            continue
        row: dict[str, Any] = {
            "bundle_key": bundle_key,
            "bundle_filename": _key_to_bundle_filename(bundle_key),
        }
        for label in AFFECT_SENTIMENT_LABELS:
            row[f"{label}_{suffix}"] = values.get(label)
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["bundle_key", "bundle_filename", *[f"{label}_{suffix}" for label in AFFECT_SENTIMENT_LABELS]])
    frame = pd.DataFrame(rows).drop_duplicates(subset=["bundle_key"], keep="last").reset_index(drop=True)
    for label in AFFECT_SENTIMENT_LABELS:
        column = f"{label}_{suffix}"
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _load_local_downloads_emotion_frame(base_dir: Path, suffix: str) -> pd.DataFrame:
    payload = _load_json_object(base_dir / f"{suffix}_emotions.json")
    rows: list[dict[str, Any]] = []
    for key, values in payload.items():
        bundle_key = str(key).strip()
        if not bundle_key or not isinstance(values, list):
            continue
        row: dict[str, Any] = {
            "bundle_key": bundle_key,
            "bundle_filename": _key_to_bundle_filename(bundle_key),
        }
        for item in values:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", "")).strip()
            if label in AFFECT_EMOTION_LABELS:
                row[f"{label}_{suffix}"] = item.get("score")
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["bundle_key", "bundle_filename", *[f"{label}_{suffix}" for label in AFFECT_EMOTION_LABELS]])
    frame = pd.DataFrame(rows).drop_duplicates(subset=["bundle_key"], keep="last").reset_index(drop=True)
    for label in AFFECT_EMOTION_LABELS:
        column = f"{label}_{suffix}"
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _load_local_downloads_affect_frame(base_dir: Path, suffix: str) -> pd.DataFrame:
    sentiment = _load_local_downloads_sentiment_frame(base_dir, suffix)
    emotions = _load_local_downloads_emotion_frame(base_dir, suffix)
    if sentiment.empty and emotions.empty:
        columns = [
            "bundle_key",
            "bundle_filename",
            *[f"{label}_{suffix}" for label in AFFECT_SENTIMENT_LABELS],
            *[f"{label}_{suffix}" for label in AFFECT_EMOTION_LABELS],
        ]
        return pd.DataFrame(columns=columns)
    if sentiment.empty:
        frame = emotions.copy()
    elif emotions.empty:
        frame = sentiment.copy()
    else:
        frame = sentiment.merge(
            emotions.drop(columns=["bundle_filename"], errors="ignore"),
            on="bundle_key",
            how="outer",
        )
    if "bundle_filename" not in frame.columns:
        frame["bundle_filename"] = frame["bundle_key"].map(_key_to_bundle_filename)
    return frame.drop_duplicates(subset=["bundle_key"], keep="last").reset_index(drop=True)


def enrich_with_current_data_bundle(
    df: pd.DataFrame,
    data_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    base_dir = _resolve_local_downloads_dir(data_dir)
    metadata: dict[str, Any] = {"current_data_bundle_dir": str(base_dir)}

    out["bundle_filename"] = out.apply(image_basename_from_row, axis=1)
    out["bundle_key"] = out["bundle_filename"].map(_bundle_key_from_value)
    if "key" in out.columns:
        out["bundle_key"] = out["bundle_key"].where(
            out["bundle_key"].astype(str).str.strip().ne(""),
            out["key"].fillna("").astype(str).map(_bundle_key_from_value),
        )

    metadata["bundle_metadata_overlap_rows"] = 0
    metadata["bundle_metadata_base_coverage"] = 0.0
    metadata["bundle_global_context_overlap_rows"] = 0
    metadata["bundle_local_context_overlap_rows"] = 0
    metadata["bundle_local_text_length_overlap_rows"] = 0

    parsed_context_df = _load_parsed_context_frame(data_dir)
    metadata["bundle_parsed_context_rows"] = int(len(parsed_context_df))
    if not parsed_context_df.empty:
        out = out.merge(parsed_context_df, on="bundle_key", how="left")
        metadata["bundle_global_context_overlap_rows"] = int(
            out["global_context_description"].fillna("").astype(str).str.strip().ne("").sum()
        )
        metadata["bundle_local_context_overlap_rows"] = int(
            (
                out["local_context_user_texts_text"].fillna("").astype(str).str.strip().ne("")
                | out["local_context_text_meaning"].fillna("").astype(str).str.strip().ne("")
                | out["local_context_instance_specific_image_description"].fillna("").astype(str).str.strip().ne("")
            ).sum()
        )

    for suffix in ["global", "local"]:
        affect_df = _load_local_downloads_affect_frame(base_dir, suffix)
        metadata[f"bundle_sentiment_{suffix}_rows"] = int(
            affect_df[f"positive_{suffix}"].notna().sum()
        ) if f"positive_{suffix}" in affect_df.columns else 0
        metadata[f"bundle_emotion_{suffix}_rows"] = int(
            affect_df[[f"{label}_{suffix}" for label in AFFECT_EMOTION_LABELS if f"{label}_{suffix}" in affect_df.columns]]
            .notna()
            .any(axis=1)
            .sum()
        ) if any(f"{label}_{suffix}" in affect_df.columns for label in AFFECT_EMOTION_LABELS) else 0
        if affect_df.empty:
            continue
        out = out.merge(
            affect_df.drop(columns=["bundle_filename"], errors="ignore"),
            on="bundle_key",
            how="left",
        )
        metadata[f"bundle_sentiment_{suffix}_overlap_rows"] = int(
            out[f"positive_{suffix}"].notna().sum()
        ) if f"positive_{suffix}" in out.columns else 0

    if {"positive_global", "negative_global"}.issubset(out.columns):
        out["global_sentiment_score_precomputed"] = (
            pd.to_numeric(out["positive_global"], errors="coerce").fillna(0.0)
            - pd.to_numeric(out["negative_global"], errors="coerce").fillna(0.0)
        )
    if {"positive_local", "negative_local"}.issubset(out.columns):
        out["local_sentiment_score_precomputed"] = (
            pd.to_numeric(out["positive_local"], errors="coerce").fillna(0.0)
            - pd.to_numeric(out["negative_local"], errors="coerce").fillna(0.0)
        )

    metadata["bundle_filename_nonempty_rows"] = int(out["bundle_filename"].astype(str).str.strip().ne("").sum())
    metadata["bundle_filename_unique"] = int(out["bundle_filename"].astype(str).nunique())
    metadata["bundle_filename_base_matchable_rows"] = int(
        out["bundle_key"].astype(str).str.strip().ne("").sum()
    )
    metadata["bundle_base_unique_keys"] = int(
        out["bundle_key"].astype(str).str.strip().nunique()
    )
    return out.drop(columns=["bundle_key"], errors="ignore"), metadata


def build_current_data_bundle_raw(data_dir: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    base_dir = _resolve_local_downloads_dir(data_dir)
    metadata: dict[str, Any] = {"current_data_bundle_dir": str(base_dir)}

    filename_parts: list[pd.Series] = []
    metadata["bundle_source"] = "LOCAL_DOWNLOADS"
    metadata["bundle_metadata_rows"] = 0
    metadata["bundle_local_text_length_rows"] = 0
    parsed_context_df = _load_parsed_context_frame(data_dir)
    metadata["bundle_parsed_context_rows"] = int(len(parsed_context_df))
    metadata["bundle_global_context_rows"] = int(
        parsed_context_df["global_context_description"].fillna("").astype(str).str.strip().ne("").sum()
    ) if "global_context_description" in parsed_context_df.columns else 0
    metadata["bundle_local_context_rows"] = int(
        (
            parsed_context_df.get("local_context_user_texts_text", pd.Series(dtype="object")).fillna("").astype(str).str.strip().ne("")
            | parsed_context_df.get("local_context_text_meaning", pd.Series(dtype="object")).fillna("").astype(str).str.strip().ne("")
            | parsed_context_df.get("local_context_instance_specific_image_description", pd.Series(dtype="object")).fillna("").astype(str).str.strip().ne("")
        ).sum()
    ) if not parsed_context_df.empty else 0

    metadata_df = pd.DataFrame(columns=["bundle_key", "bundle_filename", "score", "created_utc"])
    global_context_df = pd.DataFrame(columns=["bundle_key", "bundle_filename", "global_context_description"])
    local_context_df = pd.DataFrame(columns=["bundle_key", "bundle_filename", "local_context_user_texts_text"])
    local_text_length_df = pd.DataFrame(columns=["bundle_key", "bundle_filename", "bundle_local_text_length"])
    context_frame = parsed_context_df.copy()
    if not context_frame.empty:
        context_frame["bundle_filename"] = context_frame["bundle_key"].map(_key_to_bundle_filename)
        filename_parts.append(context_frame["bundle_key"])

    affect_frames: dict[str, pd.DataFrame] = {}
    for suffix in ["global", "local"]:
        affect_df = _load_local_downloads_affect_frame(base_dir, suffix)
        if not affect_df.empty:
            filename_parts.append(affect_df["bundle_key"])
        affect_frames[suffix] = affect_df
        metadata[f"bundle_sentiment_{suffix}_rows"] = int(
            affect_df[f"positive_{suffix}"].notna().sum()
        ) if f"positive_{suffix}" in affect_df.columns else 0
        metadata[f"bundle_emotion_{suffix}_rows"] = int(
            affect_df[[f"{label}_{suffix}" for label in AFFECT_EMOTION_LABELS if f"{label}_{suffix}" in affect_df.columns]]
            .notna()
            .any(axis=1)
            .sum()
        ) if any(f"{label}_{suffix}" in affect_df.columns for label in AFFECT_EMOTION_LABELS) else 0

    if filename_parts:
        keys = pd.concat(filename_parts, ignore_index=True).fillna("").astype(str)
        keys = keys[keys.str.strip().ne("")]
        base = pd.DataFrame({"bundle_key": keys.drop_duplicates().tolist()})
        base["bundle_filename"] = base["bundle_key"].map(_key_to_bundle_filename)
    else:
        base = pd.DataFrame(columns=["bundle_key", "bundle_filename"])

    for frame in [metadata_df, global_context_df, local_context_df, local_text_length_df, context_frame, *affect_frames.values()]:
        if frame.empty:
            continue
        base = base.merge(
            frame.drop(columns=["bundle_filename"], errors="ignore"),
            on="bundle_key",
            how="left",
        )

    base["key"] = base["bundle_key"].astype(str)
    base["image_path"] = base["bundle_filename"].astype(str)
    base["score"] = pd.to_numeric(base.get("score"), errors="coerce")
    base["created_utc"] = pd.to_datetime(base.get("created_utc"), errors="coerce", utc=True)
    base["template_original"] = NO_TEMPLATE
    base["pred_template"] = NO_TEMPLATE
    base["template_final_existing"] = NO_TEMPLATE
    base["template_source_existing"] = ""
    base["template_source"] = ""
    base["best_template_name"] = ""
    base["matched_known_template"] = False
    base["best_score"] = np.nan
    base["second_score"] = np.nan
    base["margin"] = np.nan
    base["siglip_best_score"] = np.nan
    base["dino_best_score"] = np.nan
    base["assignment_method"] = ""
    base["cluster_method"] = ""
    base["reducer"] = ""
    for column, default in [
        ("title", ""),
        ("body", ""),
        ("url", ""),
        ("image_url", ""),
        ("post_link", ""),
        ("global_context_description", ""),
        ("global_context_keywords_json", "[]"),
        ("global_context_keywords_text", ""),
        ("local_context_keywords_json", "[]"),
        ("local_context_keywords_text", ""),
        ("local_context_user_texts_json", "[]"),
        ("local_context_user_texts_text", ""),
        ("local_context_text_meaning", ""),
        ("local_context_instance_specific_image_description", ""),
    ]:
        if column not in base.columns:
            base[column] = default
        else:
            base[column] = base[column].fillna(default)

    if {"positive_global", "negative_global"}.issubset(base.columns):
        base["global_sentiment_score_precomputed"] = (
            pd.to_numeric(base["positive_global"], errors="coerce").fillna(0.0)
            - pd.to_numeric(base["negative_global"], errors="coerce").fillna(0.0)
        )
    if {"positive_local", "negative_local"}.issubset(base.columns):
        base["local_sentiment_score_precomputed"] = (
            pd.to_numeric(base["positive_local"], errors="coerce").fillna(0.0)
            - pd.to_numeric(base["negative_local"], errors="coerce").fillna(0.0)
        )

    metadata["bundle_base_rows"] = int(len(base))
    metadata["bundle_base_unique_keys"] = int(base["key"].astype(str).nunique())
    return base.drop(columns=["bundle_key"], errors="ignore"), metadata


def load_current_data_bundle_analysis_dataframe(
    data_dir: str | Path,
    default_template_label: str = "CURRENT_BUNDLE",
    require_metadata: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw, metadata = build_current_data_bundle_raw(data_dir)
    raw = raw.copy()
    raw["template_final"] = str(default_template_label)
    if require_metadata:
        raw = raw[pd.to_datetime(raw["created_utc"], errors="coerce").notna()].copy()
        metadata["bundle_rows_after_metadata_filter"] = int(len(raw))
    prepared = prepare_analysis_dataframe(raw)
    metadata["bundle_rows_prepared"] = int(len(prepared))
    return prepared, metadata


def attach_template_assignments_from_parquet(
    df: pd.DataFrame,
    analysis_parquet: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    template_df = pd.read_parquet(
        Path(analysis_parquet).expanduser().resolve(),
        columns=[
            "key",
            "template_original",
            "pred_template",
            "template_final_existing",
            "template_source_existing",
            "template_final",
            "template_source",
            "image_path",
            "title",
            "body",
            "url",
            "image_url",
            "post_link",
            "best_template_name",
            "matched_known_template",
            "best_score",
            "second_score",
            "margin",
            "siglip_best_score",
            "dino_best_score",
            "assignment_method",
            "cluster_method",
            "reducer",
        ],
    ).drop_duplicates(subset=["key"], keep="last")
    merged = out.merge(template_df, on="key", how="left", suffixes=("", "_tpl"))
    overlap = int(merged["template_final"].notna().sum()) if "template_final" in merged.columns else 0
    metadata = {
        "template_assignment_parquet": str(Path(analysis_parquet).expanduser().resolve()),
        "template_assignment_overlap_rows": overlap,
        "template_assignment_base_coverage": float(overlap / max(len(merged), 1)),
    }
    return merged, metadata


def add_sentiment_layers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    analyzer_cls = try_import_vader()
    if analyzer_cls is None:
        df["global_sentiment_label"] = "unavailable"
        df["local_sentiment_label"] = "unavailable"
        df["global_sentiment_score"] = np.nan
        df["local_sentiment_score"] = np.nan
        df["sentiment_mismatch"] = np.nan
        df["sentiment_backend"] = "none"
        return df

    analyzer = analyzer_cls()

    def score_text(text: str) -> tuple[str, float]:
        if not isinstance(text, str) or not text.strip():
            return "neutral", 0.0
        compound = float(analyzer.polarity_scores(text).get("compound", 0.0))
        if compound >= 0.05:
            return "positive", compound
        if compound <= -0.05:
            return "negative", compound
        return "neutral", compound

    global_pairs = df["global_text"].map(score_text)
    local_pairs = df["local_text"].map(score_text)
    df["global_sentiment_label"] = [pair[0] for pair in global_pairs]
    df["global_sentiment_score"] = [pair[1] for pair in global_pairs]
    df["local_sentiment_label"] = [pair[0] for pair in local_pairs]
    df["local_sentiment_score"] = [pair[1] for pair in local_pairs]
    df["sentiment_mismatch"] = (
        df["global_sentiment_label"].astype(str) != df["local_sentiment_label"].astype(str)
    ).astype(float)
    df["sentiment_backend"] = "vader"
    return df


def write_run_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2))


def write_summary_markdown(path: Path, title: str, bullets: Sequence[str]) -> None:
    lines = [f"# {title}", ""]
    lines.extend([f"- {bullet}" for bullet in bullets])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory
