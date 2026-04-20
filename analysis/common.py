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


def _parse_topic_id(name: Any) -> pd.Series:
    text = pd.Series(name, dtype="object").fillna("").astype(str)
    return pd.to_numeric(text.str.extract(r"^\s*(-?\d+)")[0], errors="coerce").astype("Int64")


def enrich_with_current_data_bundle(
    df: pd.DataFrame,
    data_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    base_dir = Path(data_dir).expanduser().resolve()
    metadata: dict[str, Any] = {"current_data_bundle_dir": str(base_dir)}

    out["bundle_filename"] = out.apply(image_basename_from_row, axis=1)
    bundle_filename_set = set(out["bundle_filename"].astype(str))

    metadata_path = base_dir / "metadata_Reddit.csv"
    if metadata_path.exists():
        meta = pd.read_csv(metadata_path)
        if {"Filename", "score", "created_utc"}.issubset(meta.columns):
            meta["Filename"] = meta["Filename"].astype(str)
            meta = meta.drop_duplicates(subset=["Filename"], keep="last")
            out = out.merge(
                meta.rename(columns={"score": "bundle_score", "created_utc": "bundle_created_utc"}),
                left_on="bundle_filename",
                right_on="Filename",
                how="left",
            ).drop(columns=["Filename"])
            overlap = out["bundle_score"].notna().sum()
            metadata["bundle_metadata_overlap_rows"] = int(overlap)
            metadata["bundle_metadata_base_coverage"] = float(overlap / max(len(out), 1))
        else:
            metadata["bundle_metadata_overlap_rows"] = 0
            metadata["bundle_metadata_base_coverage"] = 0.0

    global_context_path = base_dir / "global_context.csv"
    if global_context_path.exists():
        global_context = _read_two_column_csv(global_context_path, "bundle_filename", "bundle_global_context_text")
        out = out.merge(global_context, on="bundle_filename", how="left")
        metadata["bundle_global_context_overlap_rows"] = int(out["bundle_global_context_text"].notna().sum())
        if "global_text" in out.columns:
            out["global_text"] = out["bundle_global_context_text"].fillna(out["global_text"])

    local_context_path = base_dir / "local_context.csv"
    if local_context_path.exists():
        local_context = _read_two_column_csv(local_context_path, "bundle_filename", "bundle_local_context_text")
        out = out.merge(local_context, on="bundle_filename", how="left")
        metadata["bundle_local_context_overlap_rows"] = int(out["bundle_local_context_text"].notna().sum())
        if "local_text" in out.columns:
            out["local_text"] = out["bundle_local_context_text"].fillna(out["local_text"])

    local_text_length_path = base_dir / "local_text_length.csv"
    if local_text_length_path.exists():
        local_text_length = pd.read_csv(local_text_length_path)
        if {"Filename", "text_length"}.issubset(local_text_length.columns):
            local_text_length = (
                local_text_length.loc[:, ["Filename", "text_length"]]
                .rename(columns={"Filename": "bundle_filename", "text_length": "bundle_local_text_length"})
                .drop_duplicates(subset=["bundle_filename"], keep="last")
            )
            out = out.merge(local_text_length, on="bundle_filename", how="left")

    affect_labels = [
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
        "neutral",
        "positive",
        "negative",
    ]
    for path, suffix, text_col_name in [
        (base_dir / "sentiment_global.csv", "global", "bundle_global_affect_text"),
        (base_dir / "sentiment_local.csv", "local", "bundle_local_affect_text"),
    ]:
        if not path.exists():
            continue
        affect_df = pd.read_csv(path)
        if "Filename" not in affect_df.columns:
            continue
        rename_map = {"Filename": "bundle_filename"}
        if suffix == "global" and "Caption" in affect_df.columns:
            rename_map["Caption"] = text_col_name
        if suffix == "local" and "Extracted Text + Title" in affect_df.columns:
            rename_map["Extracted Text + Title"] = text_col_name
        for label in affect_labels:
            if label in affect_df.columns:
                rename_map[label] = f"{label}_{suffix}"
        affect_df = affect_df.rename(columns=rename_map)
        keep_cols = ["bundle_filename"] + [col for col in [text_col_name, *[f"{label}_{suffix}" for label in affect_labels]] if col in affect_df.columns]
        affect_df = affect_df.loc[:, keep_cols].drop_duplicates(subset=["bundle_filename"], keep="last")
        out = out.merge(affect_df, on="bundle_filename", how="left")
        metadata[f"bundle_sentiment_{suffix}_overlap_rows"] = int(out[f"positive_{suffix}"].notna().sum()) if f"positive_{suffix}" in out.columns else 0

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

    if global_context_path.exists() and (base_dir / "topic_global_context.csv").exists():
        global_context = _read_two_column_csv(global_context_path, "bundle_filename", "Document")
        global_topics = pd.read_csv(base_dir / "topic_global_context.csv")
        if {"Document", "Name", "Probability"}.issubset(global_topics.columns):
            global_topics = global_topics.drop_duplicates(subset=["Document"], keep="last")
            global_topic_lookup = global_context.merge(global_topics, on="Document", how="left")
            global_topic_lookup = global_topic_lookup.rename(
                columns={
                    "Name": "global_topic_name",
                    "Probability": "global_topic_probability",
                }
            )
            global_topic_lookup["global_topic"] = _parse_topic_id(global_topic_lookup["global_topic_name"])
            global_topic_lookup = global_topic_lookup.loc[
                :, ["bundle_filename", "global_topic", "global_topic_name", "global_topic_probability"]
            ].drop_duplicates(subset=["bundle_filename"], keep="last")
            out = out.merge(global_topic_lookup, on="bundle_filename", how="left")

    if local_context_path.exists() and (base_dir / "topic_local_context.csv").exists():
        local_context = _read_two_column_csv(local_context_path, "bundle_filename", "Document")
        local_topics = pd.read_csv(base_dir / "topic_local_context.csv")
        if {"Document", "Name", "Probability"}.issubset(local_topics.columns):
            local_topics = local_topics.drop_duplicates(subset=["Document"], keep="last")
            local_topic_lookup = local_context.merge(local_topics, on="Document", how="left")
            local_topic_lookup = local_topic_lookup.rename(
                columns={
                    "Name": "local_topic_name",
                    "Probability": "local_topic_probability",
                }
            )
            local_topic_lookup["local_topic"] = _parse_topic_id(local_topic_lookup["local_topic_name"])
            local_topic_lookup = local_topic_lookup.loc[
                :, ["bundle_filename", "local_topic", "local_topic_name", "local_topic_probability"]
            ].drop_duplicates(subset=["bundle_filename"], keep="last")
            out = out.merge(local_topic_lookup, on="bundle_filename", how="left")

    usage_result_path = base_dir / "usage_result.csv"
    if usage_result_path.exists():
        usage_df = _read_two_column_csv(usage_result_path, "usage_asset_name", "usage_labels")
        usage_df["usage_template_stub"] = usage_df["usage_asset_name"].astype(str).str.replace(r"_[0-9]+\.[A-Za-z0-9]+$", "", regex=True)
        usage_df["usage_template_stub"] = usage_df["usage_template_stub"].str.replace(r"\.[A-Za-z0-9]+$", "", regex=True)
        usage_df = usage_df.drop_duplicates(subset=["usage_asset_name"], keep="first")
        metadata["bundle_usage_rows"] = int(len(usage_df))

    metadata["bundle_filename_nonempty_rows"] = int(out["bundle_filename"].astype(str).str.strip().ne("").sum())
    metadata["bundle_filename_unique"] = int(out["bundle_filename"].astype(str).nunique())
    metadata["bundle_filename_base_matchable_rows"] = int(
        sum(name in bundle_filename_set for name in out["bundle_filename"].astype(str).unique())
    )
    return out, metadata


def build_current_data_bundle_raw(data_dir: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    base_dir = Path(data_dir).expanduser().resolve()
    metadata: dict[str, Any] = {"current_data_bundle_dir": str(base_dir)}

    filename_parts: list[pd.Series] = []

    metadata_path = base_dir / "metadata_Reddit.csv"
    metadata_df = pd.DataFrame(columns=["bundle_filename", "score", "created_utc"])
    if metadata_path.exists():
        meta = pd.read_csv(metadata_path)
        if {"Filename", "score", "created_utc"}.issubset(meta.columns):
            metadata_df = (
                meta.loc[:, ["Filename", "score", "created_utc"]]
                .rename(columns={"Filename": "bundle_filename"})
                .drop_duplicates(subset=["bundle_filename"], keep="last")
            )
            filename_parts.append(metadata_df["bundle_filename"])
    metadata["bundle_metadata_rows"] = int(len(metadata_df))

    global_context_path = base_dir / "global_context.csv"
    global_context_df = pd.DataFrame(columns=["bundle_filename", "global_context_description"])
    if global_context_path.exists():
        global_context_df = _read_two_column_csv(global_context_path, "bundle_filename", "global_context_description")
        filename_parts.append(global_context_df["bundle_filename"])
    metadata["bundle_global_context_rows"] = int(len(global_context_df))

    local_context_path = base_dir / "local_context.csv"
    local_context_df = pd.DataFrame(columns=["bundle_filename", "local_context_user_texts_text"])
    if local_context_path.exists():
        local_context_df = _read_two_column_csv(local_context_path, "bundle_filename", "local_context_user_texts_text")
        filename_parts.append(local_context_df["bundle_filename"])
    metadata["bundle_local_context_rows"] = int(len(local_context_df))

    local_text_length_path = base_dir / "local_text_length.csv"
    local_text_length_df = pd.DataFrame(columns=["bundle_filename", "bundle_local_text_length"])
    if local_text_length_path.exists():
        local_text_length = pd.read_csv(local_text_length_path)
        if {"Filename", "text_length"}.issubset(local_text_length.columns):
            local_text_length_df = (
                local_text_length.loc[:, ["Filename", "text_length"]]
                .rename(columns={"Filename": "bundle_filename", "text_length": "bundle_local_text_length"})
                .drop_duplicates(subset=["bundle_filename"], keep="last")
            )
            filename_parts.append(local_text_length_df["bundle_filename"])
    metadata["bundle_local_text_length_rows"] = int(len(local_text_length_df))

    affect_labels = [
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
        "neutral",
        "positive",
        "negative",
    ]
    affect_frames: dict[str, pd.DataFrame] = {}
    for path, suffix, text_col_name in [
        (base_dir / "sentiment_global.csv", "global", "bundle_global_affect_text"),
        (base_dir / "sentiment_local.csv", "local", "bundle_local_affect_text"),
    ]:
        affect_df = pd.DataFrame(columns=["bundle_filename"])
        if path.exists():
            source = pd.read_csv(path)
            if "Filename" in source.columns:
                rename_map = {"Filename": "bundle_filename"}
                if suffix == "global" and "Caption" in source.columns:
                    rename_map["Caption"] = text_col_name
                if suffix == "local" and "Extracted Text + Title" in source.columns:
                    rename_map["Extracted Text + Title"] = text_col_name
                for label in affect_labels:
                    if label in source.columns:
                        rename_map[label] = f"{label}_{suffix}"
                source = source.rename(columns=rename_map)
                keep_cols = ["bundle_filename"] + [
                    col
                    for col in [text_col_name, *[f"{label}_{suffix}" for label in affect_labels]]
                    if col in source.columns
                ]
                affect_df = source.loc[:, keep_cols].drop_duplicates(subset=["bundle_filename"], keep="last")
                filename_parts.append(affect_df["bundle_filename"])
        affect_frames[suffix] = affect_df
        metadata[f"bundle_sentiment_{suffix}_rows"] = int(len(affect_df))

    if filename_parts:
        filenames = pd.concat(filename_parts, ignore_index=True).fillna("").astype(str)
        filenames = filenames[filenames.str.strip().ne("")]
        base = pd.DataFrame({"bundle_filename": filenames.drop_duplicates().tolist()})
    else:
        base = pd.DataFrame(columns=["bundle_filename"])

    for frame in [metadata_df, global_context_df, local_context_df, local_text_length_df, *affect_frames.values()]:
        if frame.empty:
            continue
        base = base.merge(frame, on="bundle_filename", how="left")

    base["key"] = base["bundle_filename"].astype(str).str.replace(r"\.[A-Za-z0-9]+$", "", regex=True)
    base["image_path"] = base["bundle_filename"].astype(str)
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
    base["title"] = ""
    base["body"] = ""
    base["url"] = ""
    base["image_url"] = ""
    base["post_link"] = ""
    base["global_context_keywords_json"] = "[]"
    base["global_context_keywords_text"] = ""
    base["local_context_keywords_json"] = "[]"
    base["local_context_keywords_text"] = ""
    base["local_context_user_texts_json"] = "[]"
    base["local_context_text_meaning"] = ""
    base["local_context_instance_specific_image_description"] = ""

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

    if not global_context_df.empty and (base_dir / "topic_global_context.csv").exists():
        global_topics = pd.read_csv(base_dir / "topic_global_context.csv")
        if {"Document", "Name", "Probability"}.issubset(global_topics.columns):
            global_topics = global_topics.drop_duplicates(subset=["Document"], keep="last")
            global_lookup = global_context_df.rename(columns={"global_context_description": "Document"}).merge(
                global_topics,
                on="Document",
                how="left",
            )
            global_lookup = global_lookup.rename(
                columns={"Name": "global_topic_name", "Probability": "global_topic_probability"}
            )
            global_lookup["global_topic"] = _parse_topic_id(global_lookup["global_topic_name"])
            global_lookup = global_lookup.loc[
                :, ["bundle_filename", "global_topic", "global_topic_name", "global_topic_probability"]
            ].drop_duplicates(subset=["bundle_filename"], keep="last")
            base = base.merge(global_lookup, on="bundle_filename", how="left")
            metadata["bundle_global_topic_rows"] = int(global_lookup["global_topic_name"].notna().sum())

    if not local_context_df.empty and (base_dir / "topic_local_context.csv").exists():
        local_topics = pd.read_csv(base_dir / "topic_local_context.csv")
        if {"Document", "Name", "Probability"}.issubset(local_topics.columns):
            local_topics = local_topics.drop_duplicates(subset=["Document"], keep="last")
            local_lookup = local_context_df.rename(columns={"local_context_user_texts_text": "Document"}).merge(
                local_topics,
                on="Document",
                how="left",
            )
            local_lookup = local_lookup.rename(
                columns={"Name": "local_topic_name", "Probability": "local_topic_probability"}
            )
            local_lookup["local_topic"] = _parse_topic_id(local_lookup["local_topic_name"])
            local_lookup = local_lookup.loc[
                :, ["bundle_filename", "local_topic", "local_topic_name", "local_topic_probability"]
            ].drop_duplicates(subset=["bundle_filename"], keep="last")
            base = base.merge(local_lookup, on="bundle_filename", how="left")
            metadata["bundle_local_topic_rows"] = int(local_lookup["local_topic_name"].notna().sum())

    usage_result_path = base_dir / "usage_result.csv"
    if usage_result_path.exists():
        usage_df = _read_two_column_csv(usage_result_path, "usage_asset_name", "usage_labels")
        usage_df["usage_template_stub"] = usage_df["usage_asset_name"].astype(str).str.replace(
            r"_[0-9]+\.[A-Za-z0-9]+$",
            "",
            regex=True,
        )
        usage_df["usage_template_stub"] = usage_df["usage_template_stub"].str.replace(r"\.[A-Za-z0-9]+$", "", regex=True)
        usage_df = usage_df.drop_duplicates(subset=["usage_asset_name"], keep="first")
        metadata["bundle_usage_rows"] = int(len(usage_df))

    metadata["bundle_base_rows"] = int(len(base))
    metadata["bundle_base_unique_keys"] = int(base["key"].astype(str).nunique())
    return base, metadata


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
