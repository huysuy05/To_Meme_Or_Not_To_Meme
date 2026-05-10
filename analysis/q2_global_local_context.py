#!/usr/bin/env python3
from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import spearmanr

try:
    from PIL import Image, ImageOps, UnidentifiedImageError
except Exception:
    Image = None
    ImageOps = None
    UnidentifiedImageError = OSError

from common import (
    add_sentiment_layers,
    build_current_data_bundle_raw,
    build_text,
    configure_plot_style,
    ensure_dir,
    try_import_sentence_transformers,
)

try:
    from wordcloud import WordCloud
except Exception:
    WordCloud = None

try:
    from wordcloud import STOPWORDS as WORDCLOUD_STOPWORDS
except Exception:
    WORDCLOUD_STOPWORDS = set()


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")
NOTEBOOK_EMOTION_LABELS = [
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
NOTEBOOK_SENTIMENT_LABELS = ["neutral", "positive", "negative"]
ALL_AFFECT_LABELS = NOTEBOOK_EMOTION_LABELS + NOTEBOOK_SENTIMENT_LABELS
LABEL_COLORS = {
    "joy": "#ffb703",
    "anticipation": "#8ecae6",
    "disgust": "#2a9d8f",
    "sadness": "#4361ee",
    "anger": "#e63946",
    "optimism": "#f4a261",
    "surprise": "#9b5de5",
    "pessimism": "#577590",
    "fear": "#6d597a",
    "trust": "#43aa8b",
    "love": "#f15bb5",
    "neutral": "#8d99ae",
    "positive": "#06d6a0",
    "negative": "#d62828",
}
CONTEXT_COLORS = {
    "global": "#3a86ff",
    "local": "#ff006e",
}
Q2_FONT_SIZES = {
    "title": 24,
    "suptitle": 30,
    "axis_label": 19,
    "tick_label": 24,
    "xtick_label": 24,
    "legend": 15,
    "annotation": 15,
    "heatmap_annotation": 13,
    "subplot_title": 18,
    "wordcloud_title": 20,
}
DEFAULT_STOPWORDS = {
    *{str(value).lower() for value in WORDCLOUD_STOPWORDS},
    "the",
    "and",
    "for",
    "that",
    "this",
    "with",
    "from",
    "have",
    "your",
    "they",
    "them",
    "then",
    "than",
    "into",
    "about",
    "when",
    "what",
    "would",
    "could",
    "should",
    "there",
    "their",
    "it's",
    "dont",
    "doesnt",
    "didnt",
    "cant",
    "wont",
    "youre",
    "thats",
    "image",
    "images",
    "meme",
    "memes",
    "text",
    "caption",
    "captions",
    "context",
    "global",
    "local",
    "description",
    "describes",
    "described",
    "depicts",
    "depicted",
    "depicts",
    "shows",
    "showing",
    "shown",
    "panel",
    "panels",
    "top",
    "bottom",
    "left",
    "right",
    "middle",
    "character",
    "characters",
    "person",
    "people",
    "someone",
    "thing",
    "things",
    "one",
    "two",
    "three",
    "humor",
    "humorous",
    "humorously",
    "joke",
    "jokes",
    "suggest",
    "suggests",
    "suggesting",
    "suggested",
    "imply",
    "implies",
    "implying",
    "implied",
    "using",
    "used",
    "uses",
    "title",
    "body",
    "post",
    "posts",
    "user",
    "users",
    "reddit",
    "scene",
    "format",
    "template",
    "reaction",
    "common",
    "perceived",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Research Question 2: generate paired global/local affect charts for score distributions, "
            "dominant-label proportions, dominant-score summaries, and per-label word clouds."
        )
    )
    parser.add_argument("--analysis-parquet", default="data/template_first_analysis_table.parquet")
    parser.add_argument("--input-jsonl", default="data/combined_memes_sentiments_emotions.jsonl")
    parser.add_argument("--results-dir", default="analysis/results/q2_global_local_context")
    parser.add_argument("--global-topics-json", default="Meme_gemini/bertopic_models/meme_global_contexts_topics.json")
    parser.add_argument("--local-topics-json", default="Meme_gemini/bertopic_models/meme_local_contexts_topics.json")
    parser.add_argument("--topic-top-n", type=int, default=12)
    parser.add_argument("--topic-min-count", type=int, default=20)
    parser.add_argument("--topic-label-words", type=int, default=4)
    parser.add_argument("--topic-examples-per-topic", type=int, default=3)
    parser.add_argument("--topics-only", action="store_true")
    parser.add_argument("--emotion-backend", choices=["cardiff", "vader"], default="cardiff")
    parser.add_argument("--emotion-cache-dir", default="analysis/cache/q2_cardiff_affect")
    parser.add_argument("--emotion-batch-size", type=int, default=32)
    parser.add_argument("--emotion-max-length", type=int, default=512)
    parser.add_argument("--emotion-threshold", type=float, default=0.7)
    parser.add_argument("--hist-bins", type=int, default=28)
    parser.add_argument("--dominant-top-n", type=int, default=3)
    parser.add_argument("--wordcloud-max-words", type=int, default=150)
    parser.add_argument("--wordcloud-min-token-len", type=int, default=3)
    parser.add_argument("--keyword-embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--keyword-max-docs-per-class", type=int, default=5000)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def _text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def _normalize_context_name(context: str) -> str:
    return str(context).strip().lower()


def _tokenize(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    return TOKEN_PATTERN.findall(text.lower())


def _list_to_text(value: Any) -> str:
    if isinstance(value, list):
        return " | ".join(str(item) for item in value if item is not None and str(item).strip())
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value)


def _list_to_json(value: Any) -> str:
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=True)
    if value is None:
        return "[]"
    if isinstance(value, float) and np.isnan(value):
        return "[]"
    if isinstance(value, str):
        return json.dumps([value], ensure_ascii=True)
    return json.dumps([str(value)], ensure_ascii=True)


def _emotion_scores_to_columns(value: Any, suffix: str) -> dict[str, float]:
    scores = {f"{label}_{suffix}": np.nan for label in NOTEBOOK_EMOTION_LABELS}
    if not isinstance(value, list):
        return scores
    for item in value:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        if label in NOTEBOOK_EMOTION_LABELS:
            scores[f"{label}_{suffix}"] = item.get("score", np.nan)
    return scores


def _sentiment_scores_to_columns(value: Any, suffix: str) -> dict[str, float]:
    scores = {f"{label}_{suffix}": np.nan for label in NOTEBOOK_SENTIMENT_LABELS}
    if not isinstance(value, dict):
        return scores
    for label in NOTEBOOK_SENTIMENT_LABELS:
        scores[f"{label}_{suffix}"] = value.get(label, np.nan)
    return scores


def load_combined_sentiment_emotion_jsonl(path: str | Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    jsonl_path = Path(path).expanduser().resolve()
    with jsonl_path.open("r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            data = payload.get("data", {})
            if not isinstance(data, dict):
                data = {}
            local_context = data.get("local_context", {})
            if not isinstance(local_context, dict):
                local_context = {}
            template_prediction = data.get("template_prediction", {})
            if not isinstance(template_prediction, dict):
                template_prediction = {}

            row: dict[str, Any] = {
                "key": str(payload.get("key", "")).strip(),
                "global_context_description": data.get("global_context_description", ""),
                "global_context_keywords_json": _list_to_json(data.get("global_context_keywords")),
                "global_context_keywords_text": _list_to_text(data.get("global_context_keywords")),
                "local_context_user_texts_json": _list_to_json(local_context.get("user_texts")),
                "local_context_user_texts_text": _list_to_text(local_context.get("user_texts")),
                "local_context_text_meaning": local_context.get("text_meaning", ""),
                "local_context_instance_specific_image_description": local_context.get(
                    "instance_specific_image_description",
                    "",
                ),
                "local_context_keywords_json": _list_to_json(data.get("local_context_keywords")),
                "local_context_keywords_text": _list_to_text(data.get("local_context_keywords")),
                "template_original": data.get("template_original", data.get("template", "")),
                "pred_template": data.get("pred_template", ""),
                "template_final": data.get("template_final", ""),
                "template_source": data.get("template_source", ""),
                "image_path": template_prediction.get("image_path", ""),
                "best_template_name": template_prediction.get("best_template_name", ""),
                "matched_known_template": template_prediction.get("matched_known_template", np.nan),
                "best_score": template_prediction.get("best_score", np.nan),
                "second_score": template_prediction.get("second_score", np.nan),
                "margin": template_prediction.get("margin", np.nan),
                "siglip_best_score": template_prediction.get("siglip_best_score", np.nan),
                "dino_best_score": template_prediction.get("dino_best_score", np.nan),
                "assignment_method": template_prediction.get("assignment_method", ""),
                "cluster_method": template_prediction.get("cluster_method", ""),
                "reducer": template_prediction.get("reducer", ""),
                "title": "",
                "body": "",
                "score": 0.0,
            }
            row.update(_sentiment_scores_to_columns(payload.get("global_sentiments"), "global"))
            row.update(_sentiment_scores_to_columns(payload.get("local_sentiments"), "local"))
            row.update(_emotion_scores_to_columns(payload.get("global_emotions"), "global"))
            row.update(_emotion_scores_to_columns(payload.get("local_emotions"), "local"))
            rows.append(row)

    frame = pd.DataFrame(rows)
    for label in ALL_AFFECT_LABELS:
        for suffix in ("global", "local"):
            column = f"{label}_{suffix}"
            if column not in frame.columns:
                frame[column] = np.nan
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _safe_dominant_from_columns(frame: pd.DataFrame, suffix_to_strip: str) -> tuple[pd.Series, pd.Series]:
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    all_na = numeric.isna().all(axis=1)
    labels = numeric.fillna(-np.inf).idxmax(axis=1).str.replace(suffix_to_strip, "", regex=False)
    scores = numeric.max(axis=1, skipna=True)
    labels = labels.mask(all_na, "unavailable")
    scores = scores.mask(all_na, np.nan)
    return labels, scores


def _lighten_color(color: str, amount: float) -> tuple[float, float, float]:
    r, g, b = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = 1 - (1 - l) * (1 - amount)
    return colorsys.hls_to_rgb(h, l, s)


def _build_wordcloud_color_func(base_color: str):
    base_rgb = np.array(mcolors.to_rgb(base_color))

    def color_func(
        word: str,
        font_size: int,
        position: tuple[int, int],
        orientation: int | None,
        random_state: Any | None = None,
        **kwargs: Any,
    ) -> str:
        rng = random_state if random_state is not None else np.random
        mix = 0.2 + 0.55 * float(rng.random())
        rgb = base_rgb * (1 - mix) + np.ones(3) * mix
        return mcolors.to_hex(np.clip(rgb, 0, 1))

    return color_func


def _configure_q2_plot_style() -> None:
    configure_plot_style()
    plt.rcParams.update(
        {
            "axes.titlesize": Q2_FONT_SIZES["title"],
            "axes.labelsize": Q2_FONT_SIZES["axis_label"],
            "xtick.labelsize": Q2_FONT_SIZES["tick_label"],
            "ytick.labelsize": Q2_FONT_SIZES["tick_label"],
            "legend.fontsize": Q2_FONT_SIZES["legend"],
            "axes.facecolor": "#f7f9fc",
            "figure.facecolor": "#ffffff",
            "grid.color": "#d5deed",
            "grid.alpha": 0.45,
        }
    )


def _theme_plot(figsize: tuple[float, float]) -> tuple[plt.Figure, plt.Axes]:
    _configure_q2_plot_style()
    fig, ax = plt.subplots(figsize=figsize)
    ax.tick_params(axis="x", labelsize=Q2_FONT_SIZES["xtick_label"])
    ax.tick_params(axis="y", labelsize=Q2_FONT_SIZES["tick_label"])
    return fig, ax


def _theme_affective_transition_plot(figsize: tuple[float, float]) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = _theme_plot(figsize)
    return fig, ax


def load_topic_assignments(path: str | Path, prefix: str) -> pd.DataFrame:
    topic_path = Path(path).expanduser()
    if not topic_path.exists():
        return pd.DataFrame(columns=["key", f"{prefix}_topic", f"{prefix}_topic_probability"])

    text = topic_path.read_text(errors="ignore").strip()
    data: dict[str, Any] = {}
    try:
        loaded = json.loads(text)
        if isinstance(loaded, dict):
            data = loaded
    except json.JSONDecodeError:
        # Some saved topic files are truncated at exactly 512 KiB. Extract complete records.
        pattern = re.compile(
            r'"([^"]+)":\s*\{\s*"topic":\s*(-?\d+),\s*"probability":\s*([0-9.eE+-]+)\s*\}'
        )
        data = {
            key: {"topic": int(topic), "probability": float(probability)}
            for key, topic, probability in pattern.findall(text)
        }

    rows = []
    for key, value in data.items():
        if not isinstance(value, dict):
            continue
        rows.append(
            {
                "key": str(key),
                f"{prefix}_topic": int(value.get("topic", -1)),
                f"{prefix}_topic_probability": float(value.get("probability", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def attach_reddit_metadata(df: pd.DataFrame, analysis_parquet: str | Path) -> pd.DataFrame:
    path = Path(analysis_parquet).expanduser()
    if not path.exists() or "key" not in df.columns:
        return df
    try:
        metadata = pd.read_parquet(path)
    except Exception:
        return df
    keep_cols = [
        column
        for column in ["key", "score", "title", "body", "created_utc", "template_final"]
        if column in metadata.columns
    ]
    if "key" not in keep_cols:
        return df
    metadata = metadata.loc[:, keep_cols].drop_duplicates(subset=["key"], keep="first")
    merged = df.merge(metadata, on="key", how="left", suffixes=("", "_metadata"))
    for column in ["score", "title", "body", "created_utc", "template_final"]:
        meta_col = f"{column}_metadata"
        if meta_col in merged.columns:
            if column == "score":
                if column in merged.columns:
                    merged[column] = merged[meta_col].where(merged[meta_col].notna(), merged[column])
                else:
                    merged[column] = merged[meta_col]
            elif column in merged.columns:
                merged[column] = merged[column].where(merged[column].notna() & merged[column].astype(str).ne(""), merged[meta_col])
            else:
                merged[column] = merged[meta_col]
            merged = merged.drop(columns=[meta_col])
    return merged


def attach_topic_assignments(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    global_topics = load_topic_assignments(args.global_topics_json, "global")
    local_topics = load_topic_assignments(args.local_topics_json, "local")
    out = df.copy()
    if not global_topics.empty:
        out = out.merge(global_topics, on="key", how="left")
    if not local_topics.empty:
        out = out.merge(local_topics, on="key", how="left")
    return out


def _zscore_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        values = pd.to_numeric(out[column], errors="coerce")
        std = float(values.std(ddof=0))
        mean = float(values.mean())
        out[column] = 0.0 if std == 0 or np.isnan(std) else (values - mean) / std
    return out


def _load_rgb_image(path: str | Path, max_side: int = 900):
    if Image is None or ImageOps is None:
        return None
    try:
        with Image.open(Path(path).expanduser()) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((max_side, max_side))
            return image.copy()
    except (FileNotFoundError, OSError, UnidentifiedImageError, ValueError):
        return None


def summarize_topic_affect(df: pd.DataFrame, context: str, top_n: int, min_count: int) -> pd.DataFrame:
    topic_col = f"{context}_topic"
    prob_col = f"{context}_topic_probability"
    summary_columns = [
        "context",
        "topic",
        "frequency",
        "mean_upvotes",
        "median_upvotes",
        "mean_topic_probability",
        "sentiment_balance",
        "topic_share",
    ]
    if topic_col not in df.columns:
        return pd.DataFrame(columns=summary_columns)
    affect_cols = [
        *[f"{label}_{context}" for label in NOTEBOOK_SENTIMENT_LABELS if f"{label}_{context}" in df.columns],
        *[f"{label}_{context}" for label in NOTEBOOK_EMOTION_LABELS if f"{label}_{context}" in df.columns],
    ]
    working = df[df[topic_col].notna()].copy()
    working[topic_col] = pd.to_numeric(working[topic_col], errors="coerce")
    working = working[working[topic_col].ne(-1)]
    if working.empty:
        return pd.DataFrame(columns=summary_columns)
    positive = pd.to_numeric(
        working[f"positive_{context}"] if f"positive_{context}" in working.columns else pd.Series(0.0, index=working.index),
        errors="coerce",
    ).fillna(0.0)
    negative = pd.to_numeric(
        working[f"negative_{context}"] if f"negative_{context}" in working.columns else pd.Series(0.0, index=working.index),
        errors="coerce",
    ).fillna(0.0)
    working["sentiment_balance"] = positive - negative
    agg_spec = {
        "frequency": ("key", "size"),
        "mean_upvotes": ("score", "mean"),
        "median_upvotes": ("score", "median"),
        "mean_topic_probability": (prob_col, "mean") if prob_col in working.columns else ("key", "size"),
        "sentiment_balance": ("sentiment_balance", "mean"),
    }
    for column in affect_cols:
        agg_spec[column] = (column, "mean")
    summary = working.groupby(topic_col, observed=True).agg(**agg_spec).reset_index().rename(columns={topic_col: "topic"})
    summary["topic"] = summary["topic"].astype(int)
    summary["topic_share"] = summary["frequency"] / max(float(len(working)), 1.0)
    summary = summary[summary["frequency"].ge(int(min_count))].sort_values("frequency", ascending=False).head(int(top_n)).copy()
    summary.insert(0, "context", context)
    return summary


def compute_topic_metric_correlations(summary: pd.DataFrame, context: str) -> pd.DataFrame:
    columns = ["context", "metric", "affect", "spearman_rho", "p_value", "n_topics"]
    if summary.empty:
        return pd.DataFrame(columns=columns)
    metric_cols = ["frequency", "topic_share", "mean_upvotes", "median_upvotes"]
    affect_cols = ["sentiment_balance", *[f"{label}_{context}" for label in NOTEBOOK_SENTIMENT_LABELS + NOTEBOOK_EMOTION_LABELS if f"{label}_{context}" in summary.columns]]
    rows = []
    for metric in metric_cols:
        for affect in affect_cols:
            if metric not in summary.columns or affect not in summary.columns:
                continue
            valid = summary[[metric, affect]].dropna()
            if len(valid) < 3:
                rho, pvalue = np.nan, np.nan
            elif valid[metric].nunique(dropna=True) < 2 or valid[affect].nunique(dropna=True) < 2:
                rho, pvalue = np.nan, np.nan
            else:
                rho, pvalue = spearmanr(valid[metric], valid[affect])
            rows.append(
                {
                    "context": context,
                    "metric": metric,
                    "affect": affect,
                    "spearman_rho": float(rho) if pd.notna(rho) else np.nan,
                    "p_value": float(pvalue) if pd.notna(pvalue) else np.nan,
                    "n_topics": int(len(valid)),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def derive_topic_labels(df: pd.DataFrame, summary: pd.DataFrame, context: str, n_words: int) -> pd.DataFrame:
    if summary.empty:
        out = summary.copy()
        out["topic_keywords"] = ""
        out["topic_label"] = ""
        return out
    topic_col = f"{context}_topic"
    text_col = f"{context}_text"
    if topic_col not in df.columns or text_col not in df.columns:
        out = summary.copy()
        out["topic_keywords"] = out["topic"].map(lambda value: f"topic {int(value)}")
        out["topic_label"] = out["topic_keywords"]
        return out

    working = df[[topic_col, text_col]].dropna().copy()
    working[topic_col] = pd.to_numeric(working[topic_col], errors="coerce")
    selected_topics = set(pd.to_numeric(summary["topic"], errors="coerce").dropna().astype(int))
    working = working[working[topic_col].isin(selected_topics)]
    rows: list[dict[str, Any]] = []
    stop_words = sorted(DEFAULT_STOPWORDS)
    for topic in summary["topic"].astype(int).tolist():
        texts = working.loc[working[topic_col].eq(topic), text_col].fillna("").astype(str)
        texts = texts[texts.str.strip().ne("")].head(400).tolist()
        keywords: list[str] = []
        if texts:
            try:
                vectorizer = TfidfVectorizer(
                    stop_words=stop_words,
                    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_'-]{2,}\b",
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=1500,
                )
                matrix = vectorizer.fit_transform(texts)
                scores = np.asarray(matrix.sum(axis=0)).ravel()
                terms = np.asarray(vectorizer.get_feature_names_out())
                order = scores.argsort()[::-1]
                for idx in order:
                    term = str(terms[idx]).replace("_", " ").strip()
                    if term and term not in keywords:
                        keywords.append(term)
                    if len(keywords) >= int(n_words):
                        break
            except ValueError:
                keywords = []
        label = ", ".join(keywords) if keywords else f"topic {topic}"
        rows.append({"topic": topic, "topic_keywords": "; ".join(keywords), "topic_label": label})
    label_df = pd.DataFrame(rows)
    return summary.merge(label_df, on="topic", how="left")


def select_topic_examples(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    context: str,
    examples_per_topic: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if summary.empty:
        return pd.DataFrame(rows)
    topic_col = f"{context}_topic"
    prob_col = f"{context}_topic_probability"
    if topic_col not in df.columns or "image_path" not in df.columns:
        return pd.DataFrame(rows)

    working = df.copy()
    working[topic_col] = pd.to_numeric(working[topic_col], errors="coerce")
    working["score"] = pd.to_numeric(working.get("score"), errors="coerce").fillna(0.0)
    if prob_col in working.columns:
        working[prob_col] = pd.to_numeric(working[prob_col], errors="coerce").fillna(0.0)
    else:
        working[prob_col] = 0.0

    label_lookup = summary.set_index("topic")["topic_label"].to_dict() if "topic_label" in summary.columns else {}
    for topic in summary["topic"].astype(int).tolist():
        subset = working[working[topic_col].eq(topic)].copy()
        subset = subset.sort_values([prob_col, "score"], ascending=[False, False])
        rank = 0
        for candidate in subset.itertuples(index=False):
            image_path = str(getattr(candidate, "image_path", "") or "")
            if not image_path or _load_rgb_image(image_path) is None:
                continue
            rank += 1
            rows.append(
                {
                    "context": context,
                    "topic": int(topic),
                    "topic_label": label_lookup.get(topic, f"topic {topic}"),
                    "example_rank": rank,
                    "key": getattr(candidate, "key", ""),
                    "template_final": getattr(candidate, "template_final", ""),
                    "title": getattr(candidate, "title", ""),
                    "score": float(getattr(candidate, "score", 0.0) or 0.0),
                    "topic_probability": float(getattr(candidate, prob_col, 0.0) or 0.0),
                    "image_path": image_path,
                }
            )
            if rank >= int(examples_per_topic):
                break
    return pd.DataFrame(rows)


def save_topic_examples_grid(examples: pd.DataFrame, context: str, outpath: Path, examples_per_topic: int) -> None:
    if examples.empty:
        return
    _configure_q2_plot_style()
    topic_order = examples[["topic", "topic_label"]].drop_duplicates().reset_index(drop=True)
    nrows = len(topic_order)
    ncols = int(examples_per_topic)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.8 * nrows), squeeze=False)
    for row_idx, topic_row in enumerate(topic_order.itertuples(index=False)):
        topic_examples = examples[examples["topic"].eq(topic_row.topic)].sort_values("example_rank")
        for col_idx in range(ncols):
            ax = axes[row_idx, col_idx]
            ax.axis("off")
            if col_idx >= len(topic_examples):
                continue
            example = topic_examples.iloc[col_idx]
            image = _load_rgb_image(example["image_path"])
            if image is not None:
                ax.imshow(image)
            title_lines = []
            if col_idx == 0:
                title_lines.append(f"T{int(topic_row.topic)}: {str(topic_row.topic_label).title()}")
            title_lines.append(f"{str(example.get('template_final', '')).replace('_', ' ')[:42]}")
            title_lines.append(f"score={float(example.get('score', 0.0)):.0f}, p={float(example.get('topic_probability', 0.0)):.2f}")
            ax.set_title(
                "\n".join(line for line in title_lines if line),
                fontsize=Q2_FONT_SIZES["wordcloud_title"],
                pad=16,
            )
    fig.suptitle(
        f"Representative Memes Aligned with {context.title()} Topics",
        fontsize=Q2_FONT_SIZES["suptitle"],
        y=0.995,
    )
    fig.tight_layout(h_pad=2.2, w_pad=1.0)
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _short_topic_label(value: Any, max_words: int = 4) -> str:
    label = str(value or "").replace(";", ",").strip()
    if not label:
        return "unlabeled"
    parts = [part.strip() for part in re.split(r"[,|]", label) if part.strip()]
    if parts:
        return ", ".join(parts[:max_words])
    words = label.split()
    return " ".join(words[:max_words])


def save_topic_engagement_affect_scatter(summary: pd.DataFrame, context: str, outpath: Path) -> None:
    _configure_q2_plot_style()
    fig, ax = plt.subplots(figsize=(14, 10), constrained_layout=True)
    title = f"{context.title()} Topic Engagement and Sentiment Orientation"
    if summary.empty:
        ax.axis("off")
        ax.set_title(f"{title}: no topic data", fontsize=Q2_FONT_SIZES["title"])
        fig.savefig(outpath, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return
    x = pd.to_numeric(summary["mean_upvotes"], errors="coerce").fillna(0.0)
    y = pd.to_numeric(summary["sentiment_balance"], errors="coerce").fillna(0.0)
    size = pd.to_numeric(summary["frequency"], errors="coerce").fillna(1.0)
    sizes = 220 + 1300 * (size / max(float(size.max()), 1.0))
    ax.scatter(
        x,
        y,
        s=sizes,
        color=CONTEXT_COLORS.get(context, "#3a86ff"),
        alpha=0.72,
        edgecolor="#17202a",
        linewidth=1.0,
    )
    for row in summary.itertuples(index=False):
        ax.annotate(
            f"T{int(row.topic)}",
            (float(getattr(row, "mean_upvotes", 0.0)), float(getattr(row, "sentiment_balance", 0.0))),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=Q2_FONT_SIZES["annotation"],
        )
    legend_lines = [
        f"T{int(row.topic)}: {_short_topic_label(getattr(row, 'topic_label', ''), max_words=4).title()}"
        for row in summary.itertuples(index=False)
    ]
    ax.text(
        1.02,
        0.98,
        "\n".join(legend_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=Q2_FONT_SIZES["annotation"],
        linespacing=1.35,
    )
    ax.axhline(0, color="#17202a", linewidth=1.0, alpha=0.45)
    ax.set_xscale("symlog", linthresh=25)
    ax.set_xlabel("Mean Upvotes", fontsize=Q2_FONT_SIZES["axis_label"])
    ax.set_ylabel("Sentiment Balance (Positive - Negative)", fontsize=Q2_FONT_SIZES["axis_label"])
    ax.set_title(title, fontsize=Q2_FONT_SIZES["title"])
    ax.grid(True, alpha=0.3)
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_topic_affect_profile_figure(summary: pd.DataFrame, context: str, outpath: Path) -> None:
    heatmap_columns = ["frequency", "mean_upvotes", "sentiment_balance", "joy", "anger", "sadness", "fear", "disgust", "optimism"]
    fig, (ax, legend_ax) = plt.subplots(
        1,
        2,
        figsize=(21, 10),
        gridspec_kw={"width_ratios": [3.2, 1.25]},
        constrained_layout=True,
    )
    if summary.empty:
        ax.axis("off")
        legend_ax.axis("off")
        ax.set_title(f"{context.title()} Topics: no topic data", fontsize=Q2_FONT_SIZES["title"])
        fig.savefig(outpath, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return
    cols = []
    col_labels = []
    for col in heatmap_columns:
        source_col = f"{col}_{context}" if col in NOTEBOOK_EMOTION_LABELS + NOTEBOOK_SENTIMENT_LABELS else col
        if source_col in summary.columns:
            cols.append(source_col)
            col_labels.append(col.replace("_", " ").title())
    plot_df = summary.sort_values("frequency", ascending=False).copy()
    z = _zscore_columns(plot_df, cols)
    matrix = z[cols].to_numpy(dtype=float)
    image = ax.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-2.0, vmax=2.0)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(col_labels, rotation=40, ha="right", fontsize=Q2_FONT_SIZES["tick_label"])
    ax.set_yticks(np.arange(len(plot_df)))
    ax.set_yticklabels([f"T{int(row.topic)}" for row in plot_df.itertuples(index=False)], fontsize=Q2_FONT_SIZES["tick_label"])
    ax.set_title(f"{context.title()} Topic Frequency, Upvotes, and Affective Profile", fontsize=Q2_FONT_SIZES["title"])
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            ax.text(
                col_idx,
                row_idx,
                f"{matrix[row_idx, col_idx]:.1f}",
                ha="center",
                va="center",
                fontsize=Q2_FONT_SIZES["heatmap_annotation"],
                color="#17202a",
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    colorbar.set_label("Topic-Level Z-Score", fontsize=Q2_FONT_SIZES["axis_label"])
    colorbar.ax.tick_params(labelsize=Q2_FONT_SIZES["tick_label"])

    legend_ax.axis("off")
    legend_ax.set_title("Topic Legend", fontsize=Q2_FONT_SIZES["title"], loc="left")
    legend_lines = [
        f"T{int(row.topic)}  {_short_topic_label(getattr(row, 'topic_label', ''), max_words=4).title()}"
        for row in plot_df.itertuples(index=False)
    ]
    legend_ax.text(
        0.0,
        0.96,
        "\n".join(legend_lines),
        transform=legend_ax.transAxes,
        ha="left",
        va="top",
        fontsize=Q2_FONT_SIZES["annotation"],
        linespacing=1.55,
    )
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_topic_affect_analysis(df: pd.DataFrame, outdir: Path, args: argparse.Namespace) -> None:
    topic_dir = ensure_dir(outdir / "topic_affect")
    global_summary = summarize_topic_affect(df, "global", args.topic_top_n, args.topic_min_count)
    local_summary = summarize_topic_affect(df, "local", args.topic_top_n, args.topic_min_count)
    global_summary = derive_topic_labels(df, global_summary, "global", args.topic_label_words)
    local_summary = derive_topic_labels(df, local_summary, "local", args.topic_label_words)
    global_summary.to_csv(topic_dir / "global_topic_affect_summary.csv", index=False)
    local_summary.to_csv(topic_dir / "local_topic_affect_summary.csv", index=False)
    global_examples = select_topic_examples(df, global_summary, "global", args.topic_examples_per_topic)
    local_examples = select_topic_examples(df, local_summary, "local", args.topic_examples_per_topic)
    global_examples.to_csv(topic_dir / "global_topic_examples.csv", index=False)
    local_examples.to_csv(topic_dir / "local_topic_examples.csv", index=False)
    correlations = pd.concat(
        [
            compute_topic_metric_correlations(global_summary, "global"),
            compute_topic_metric_correlations(local_summary, "local"),
        ],
        ignore_index=True,
    )
    correlations.to_csv(topic_dir / "topic_metric_affect_correlations.csv", index=False)
    save_topic_affect_profile_figure(
        global_summary,
        "global",
        topic_dir / "fig_global_topic_affect_profile.png",
    )
    save_topic_affect_profile_figure(
        local_summary,
        "local",
        topic_dir / "fig_local_topic_affect_profile.png",
    )
    save_topic_engagement_affect_scatter(
        global_summary,
        "global",
        topic_dir / "fig_global_topic_engagement_sentiment_scatter.png",
    )
    save_topic_engagement_affect_scatter(
        local_summary,
        "local",
        topic_dir / "fig_local_topic_engagement_sentiment_scatter.png",
    )
    save_topic_examples_grid(
        global_examples,
        "global",
        topic_dir / "fig_global_topic_example_memes.png",
        args.topic_examples_per_topic,
    )
    save_topic_examples_grid(
        local_examples,
        "local",
        topic_dir / "fig_local_topic_example_memes.png",
        args.topic_examples_per_topic,
    )


def add_cardiff_affect_layers(
    df: pd.DataFrame,
    cache_dir: str | Path,
    batch_size: int,
    max_length: int,
) -> pd.DataFrame:
    try:
        import torch
        from transformers import AutoTokenizer, pipeline  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Cardiff emotion scoring requires `transformers` and `torch`."
        ) from exc

    cache_path = ensure_dir(cache_dir)
    emotion_cache_path = cache_path / "emotion_scores.parquet"
    sentiment_cache_path = cache_path / "sentiment_scores.parquet"
    texts = pd.concat([df["global_text"], df["local_text"]], ignore_index=True).fillna("").astype(str)
    texts = texts[texts.str.strip().ne("")].drop_duplicates().reset_index(drop=True)
    if texts.empty:
        enriched = df.copy()
        for label in ALL_AFFECT_LABELS:
            enriched[f"{label}_global"] = np.nan
            enriched[f"{label}_local"] = np.nan
        enriched["global_dominant_emotion"] = "unavailable"
        enriched["local_dominant_emotion"] = "unavailable"
        enriched["global_sentiment_label"] = "unavailable"
        enriched["local_sentiment_label"] = "unavailable"
        enriched["global_sentiment_score"] = np.nan
        enriched["local_sentiment_score"] = np.nan
        enriched["emotion_mismatch"] = np.nan
        enriched["sentiment_mismatch"] = np.nan
        enriched["sentiment_backend"] = "none"
        return enriched

    lookup = pd.DataFrame({"text": texts})
    lookup["text_hash"] = lookup["text"].map(_text_hash)

    if emotion_cache_path.exists():
        emotion_cache = pd.read_parquet(emotion_cache_path)
    else:
        emotion_cache = pd.DataFrame(columns=["text_hash", "text", *NOTEBOOK_EMOTION_LABELS])
    if sentiment_cache_path.exists():
        sentiment_cache = pd.read_parquet(sentiment_cache_path)
    else:
        sentiment_cache = pd.DataFrame(columns=["text_hash", "text", *NOTEBOOK_SENTIMENT_LABELS])

    cached_hashes = set(emotion_cache.get("text_hash", [])) & set(sentiment_cache.get("text_hash", []))
    missing_hashes = set(lookup["text_hash"]) - cached_hashes
    missing_lookup = lookup[lookup["text_hash"].isin(missing_hashes)].copy()

    if not missing_lookup.empty:
        device: int | str = -1
        if torch.cuda.is_available():
            device = 0
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"

        emotion_model_name = "cardiffnlp/twitter-roberta-base-emotion-multilabel-latest"
        sentiment_model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        tokenizer = AutoTokenizer.from_pretrained(emotion_model_name)
        effective_max_length = min(int(max_length), int(getattr(tokenizer, "model_max_length", max_length)))
        emotion_pipe = pipeline(
            "text-classification",
            model=emotion_model_name,
            tokenizer=tokenizer,
            top_k=None,
            device=device,
        )
        sentiment_pipe = pipeline(
            "text-classification",
            model=sentiment_model_name,
            top_k=None,
            device=device,
        )

        emotion_rows: list[dict[str, Any]] = []
        sentiment_rows: list[dict[str, Any]] = []
        texts_to_score = missing_lookup["text"].tolist()
        hashes_to_score = missing_lookup["text_hash"].tolist()
        for start in range(0, len(texts_to_score), batch_size):
            batch_texts = texts_to_score[start : start + batch_size]
            batch_hashes = hashes_to_score[start : start + batch_size]
            emotion_results = emotion_pipe(
                batch_texts,
                batch_size=batch_size,
                truncation=True,
                max_length=effective_max_length,
            )
            sentiment_results = sentiment_pipe(
                batch_texts,
                batch_size=batch_size,
                truncation=True,
                max_length=effective_max_length,
            )
            for text_value, text_hash, emotion_output, sentiment_output in zip(
                batch_texts, batch_hashes, emotion_results, sentiment_results
            ):
                emotion_scores = {label: 0.0 for label in NOTEBOOK_EMOTION_LABELS}
                emotion_scores.update({item["label"]: float(item["score"]) for item in emotion_output})
                sentiment_scores = {label: 0.0 for label in NOTEBOOK_SENTIMENT_LABELS}
                sentiment_scores.update({item["label"]: float(item["score"]) for item in sentiment_output})
                emotion_rows.append({"text_hash": text_hash, "text": text_value, **emotion_scores})
                sentiment_rows.append({"text_hash": text_hash, "text": text_value, **sentiment_scores})

        emotion_new = pd.DataFrame(emotion_rows)
        sentiment_new = pd.DataFrame(sentiment_rows)
        emotion_cache = (
            pd.concat([emotion_cache, emotion_new], ignore_index=True)
            .drop_duplicates(subset=["text_hash"], keep="last")
            .reset_index(drop=True)
        )
        sentiment_cache = (
            pd.concat([sentiment_cache, sentiment_new], ignore_index=True)
            .drop_duplicates(subset=["text_hash"], keep="last")
            .reset_index(drop=True)
        )

    score_lookup = emotion_cache.merge(
        sentiment_cache.drop(columns=["text"], errors="ignore"),
        on="text_hash",
        how="inner",
        validate="one_to_one",
    )
    global_lookup = pd.DataFrame({"global_text": df["global_text"].fillna("").astype(str)})
    global_lookup["text_hash"] = global_lookup["global_text"].map(_text_hash)
    local_lookup = pd.DataFrame({"local_text": df["local_text"].fillna("").astype(str)})
    local_lookup["text_hash"] = local_lookup["local_text"].map(_text_hash)

    enriched = df.reset_index(drop=True).copy()
    global_scored = global_lookup.merge(score_lookup, on="text_hash", how="left")
    local_scored = local_lookup.merge(score_lookup, on="text_hash", how="left")

    for label in ALL_AFFECT_LABELS:
        enriched[f"{label}_global"] = pd.to_numeric(global_scored[label], errors="coerce")
        enriched[f"{label}_local"] = pd.to_numeric(local_scored[label], errors="coerce")

    global_emotion_cols = [f"{label}_global" for label in ALL_AFFECT_LABELS]
    local_emotion_cols = [f"{label}_local" for label in ALL_AFFECT_LABELS]
    global_sentiment_cols = [f"{label}_global" for label in NOTEBOOK_SENTIMENT_LABELS]
    local_sentiment_cols = [f"{label}_local" for label in NOTEBOOK_SENTIMENT_LABELS]

    enriched["global_dominant_emotion"], _ = _safe_dominant_from_columns(enriched[global_emotion_cols], "_global")
    enriched["local_dominant_emotion"], _ = _safe_dominant_from_columns(enriched[local_emotion_cols], "_local")
    enriched["global_sentiment_label"], _ = _safe_dominant_from_columns(enriched[global_sentiment_cols], "_global")
    enriched["local_sentiment_label"], _ = _safe_dominant_from_columns(enriched[local_sentiment_cols], "_local")
    enriched["global_sentiment_score"] = enriched["positive_global"].fillna(0.0) - enriched["negative_global"].fillna(0.0)
    enriched["local_sentiment_score"] = enriched["positive_local"].fillna(0.0) - enriched["negative_local"].fillna(0.0)
    enriched["emotion_mismatch"] = (
        enriched["global_dominant_emotion"].astype(str) != enriched["local_dominant_emotion"].astype(str)
    ).astype(float)
    enriched["sentiment_mismatch"] = (
        enriched["global_sentiment_label"].astype(str) != enriched["local_sentiment_label"].astype(str)
    ).astype(float)
    enriched["sentiment_backend"] = "cardiff"
    return enriched


def add_affect_layers(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if args.emotion_backend == "cardiff":
        if _has_any_precomputed_cardiff(df):
            enriched = _finalize_precomputed_cardiff(df)
            if getattr(args, "topics_only", False):
                return enriched
            missing_mask = enriched["positive_global"].isna() | enriched["positive_local"].isna()
            if bool(missing_mask.any()):
                computed_missing = add_cardiff_affect_layers(
                    df.loc[missing_mask].copy(),
                    cache_dir=args.emotion_cache_dir,
                    batch_size=args.emotion_batch_size,
                    max_length=args.emotion_max_length,
                )
                for column in computed_missing.columns:
                    enriched.loc[missing_mask, column] = computed_missing[column].to_numpy()
            return enriched
        return add_cardiff_affect_layers(
            df,
            cache_dir=args.emotion_cache_dir,
            batch_size=args.emotion_batch_size,
            max_length=args.emotion_max_length,
        )
    enriched = add_sentiment_layers(df)
    for label in NOTEBOOK_SENTIMENT_LABELS:
        enriched[f"{label}_global"] = np.where(enriched["global_sentiment_label"].eq(label), 1.0, 0.0)
        enriched[f"{label}_local"] = np.where(enriched["local_sentiment_label"].eq(label), 1.0, 0.0)
    for label in NOTEBOOK_EMOTION_LABELS:
        enriched[f"{label}_global"] = np.nan
        enriched[f"{label}_local"] = np.nan
    enriched["global_dominant_emotion"] = enriched["global_sentiment_label"]
    enriched["local_dominant_emotion"] = enriched["local_sentiment_label"]
    enriched["emotion_mismatch"] = enriched["sentiment_mismatch"]
    return enriched


def _has_any_precomputed_cardiff(df: pd.DataFrame) -> bool:
    required = [
        "positive_global",
        "negative_global",
        "neutral_global",
        "positive_local",
        "negative_local",
        "neutral_local",
    ]
    return all(column in df.columns for column in required)


def _finalize_precomputed_cardiff(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    for label in ALL_AFFECT_LABELS:
        for suffix in ["global", "local"]:
            column = f"{label}_{suffix}"
            if column not in enriched.columns:
                enriched[column] = np.nan
            enriched[column] = pd.to_numeric(enriched[column], errors="coerce")

    global_emotion_cols = [f"{label}_global" for label in ALL_AFFECT_LABELS]
    local_emotion_cols = [f"{label}_local" for label in ALL_AFFECT_LABELS]
    global_sentiment_cols = [f"{label}_global" for label in NOTEBOOK_SENTIMENT_LABELS]
    local_sentiment_cols = [f"{label}_local" for label in NOTEBOOK_SENTIMENT_LABELS]

    enriched["global_dominant_emotion"], _ = _safe_dominant_from_columns(enriched[global_emotion_cols], "_global")
    enriched["local_dominant_emotion"], _ = _safe_dominant_from_columns(enriched[local_emotion_cols], "_local")
    enriched["global_sentiment_label"], _ = _safe_dominant_from_columns(enriched[global_sentiment_cols], "_global")
    enriched["local_sentiment_label"], _ = _safe_dominant_from_columns(enriched[local_sentiment_cols], "_local")
    enriched["global_sentiment_score"] = enriched["positive_global"].fillna(0.0) - enriched["negative_global"].fillna(0.0)
    enriched["local_sentiment_score"] = enriched["positive_local"].fillna(0.0) - enriched["negative_local"].fillna(0.0)
    enriched["emotion_mismatch"] = (
        enriched["global_dominant_emotion"].astype(str) != enriched["local_dominant_emotion"].astype(str)
    ).astype(float)
    enriched["sentiment_mismatch"] = (
        enriched["global_sentiment_label"].astype(str) != enriched["local_sentiment_label"].astype(str)
    ).astype(float)
    enriched["sentiment_backend"] = "cardiff_precomputed"
    return enriched


def compute_dominant_affect(df: pd.DataFrame, context: str) -> pd.DataFrame:
    suffix = _normalize_context_name(context)
    score_columns = [f"{label}_{suffix}" for label in ALL_AFFECT_LABELS if f"{label}_{suffix}" in df.columns]
    if not score_columns:
        return pd.DataFrame(columns=["dominant_label", "dominant_score", "text"])

    working = df.copy()
    working["dominant_label"], working["dominant_score"] = _safe_dominant_from_columns(
        working[score_columns],
        f"_{suffix}",
    )
    working["context"] = suffix
    working["text"] = working[f"{suffix}_text"].fillna("").astype(str)
    return working.loc[:, ["dominant_label", "dominant_score", "text", f"{suffix}_text"]].rename(
        columns={f"{suffix}_text": "source_text"}
    )


def compute_dominant_summary(df: pd.DataFrame, context: str, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    dominant = compute_dominant_affect(df, context)
    dominant["dominant_score"] = pd.to_numeric(dominant["dominant_score"], errors="coerce")
    filtered = dominant[
        dominant["dominant_label"].isin(ALL_AFFECT_LABELS) & dominant["dominant_score"].ge(float(threshold))
    ].copy()
    if filtered.empty:
        summary = pd.DataFrame(columns=["label", "count", "proportion", "mean_score", "median_score"])
        return filtered, summary

    summary = (
        filtered.groupby("dominant_label", observed=True)["dominant_score"]
        .agg(count="size", mean_score="mean", median_score="median")
        .reset_index()
        .rename(columns={"dominant_label": "label"})
        .sort_values(["count", "mean_score"], ascending=[False, False])
        .reset_index(drop=True)
    )
    summary["proportion"] = summary["count"] / summary["count"].sum()
    return filtered, summary


def _row_normalize_counts(counts: pd.DataFrame) -> pd.DataFrame:
    row_sums = counts.sum(axis=1).replace(0, np.nan)
    probabilities = counts.div(row_sums, axis=0).fillna(0.0)
    return probabilities


def _entropy(probabilities: np.ndarray) -> float:
    positive = probabilities[probabilities > 0]
    if len(positive) == 0:
        return 0.0
    return float(-(positive * np.log2(positive)).sum())


def _kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-12) -> float:
    p_safe = np.asarray(p, dtype=float) + epsilon
    q_safe = np.asarray(q, dtype=float) + epsilon
    p_safe = p_safe / p_safe.sum()
    q_safe = q_safe / q_safe.sum()
    return float((p_safe * np.log2(p_safe / q_safe)).sum())


def compute_threshold_cooccurrence(df: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = [label for label in ALL_AFFECT_LABELS if f"{label}_global" in df.columns and f"{label}_local" in df.columns]
    counts = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
    for global_label in labels:
        global_mask = pd.to_numeric(df[f"{global_label}_global"], errors="coerce").ge(float(threshold))
        if not bool(global_mask.any()):
            continue
        for local_label in labels:
            local_mask = pd.to_numeric(df[f"{local_label}_local"], errors="coerce").ge(float(threshold))
            counts.loc[global_label, local_label] = int((global_mask & local_mask).sum())
    return counts, _row_normalize_counts(counts)


def compute_dominant_transition_counts(df: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    global_dominant = compute_dominant_affect(df, "global").rename(
        columns={"dominant_label": "global_label", "dominant_score": "global_score"}
    )
    local_dominant = compute_dominant_affect(df, "local").rename(
        columns={"dominant_label": "local_label", "dominant_score": "local_score"}
    )
    paired = pd.DataFrame(
        {
            "global_label": global_dominant["global_label"],
            "global_score": pd.to_numeric(global_dominant["global_score"], errors="coerce"),
            "local_label": local_dominant["local_label"],
            "local_score": pd.to_numeric(local_dominant["local_score"], errors="coerce"),
        }
    )
    paired = paired[
        paired["global_label"].isin(ALL_AFFECT_LABELS)
        & paired["local_label"].isin(ALL_AFFECT_LABELS)
        & paired["global_score"].ge(float(threshold))
        & paired["local_score"].ge(float(threshold))
    ].copy()

    if paired.empty:
        counts = pd.DataFrame(0, index=ALL_AFFECT_LABELS, columns=ALL_AFFECT_LABELS, dtype=int)
        return paired, counts, _row_normalize_counts(counts)

    counts = pd.crosstab(
        pd.Categorical(paired["global_label"], categories=ALL_AFFECT_LABELS, ordered=True),
        pd.Categorical(paired["local_label"], categories=ALL_AFFECT_LABELS, ordered=True),
        dropna=False,
    )
    counts.index = ALL_AFFECT_LABELS
    counts.columns = ALL_AFFECT_LABELS
    counts = counts.astype(int)
    return paired, counts, _row_normalize_counts(counts)


def summarize_alignment(counts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = int(counts.to_numpy().sum())
    aligned_total = 0
    for label in counts.index:
        row_total = int(counts.loc[label].sum())
        aligned = int(counts.loc[label, label]) if label in counts.columns else 0
        aligned_total += aligned
        rows.append(
            {
                "global_label": label,
                "count": row_total,
                "aligned_count": aligned,
                "alignment_rate": float(aligned / row_total) if row_total else np.nan,
            }
        )
    rows.append(
        {
            "global_label": "OVERALL",
            "count": total,
            "aligned_count": int(aligned_total),
            "alignment_rate": float(aligned_total / total) if total else np.nan,
        }
    )
    return pd.DataFrame(rows)


def summarize_neutral_injection(counts: pd.DataFrame) -> pd.DataFrame:
    if "neutral" not in counts.index:
        return pd.DataFrame(columns=["global_label", "local_group", "count", "rate"])
    row = counts.loc["neutral"]
    total = int(row.sum())
    neutral_count = int(row.get("neutral", 0))
    injected_count = int(total - neutral_count)
    return pd.DataFrame(
        [
            {
                "global_label": "neutral",
                "local_group": "neutral",
                "count": neutral_count,
                "rate": float(neutral_count / total) if total else np.nan,
            },
            {
                "global_label": "neutral",
                "local_group": "non_neutral",
                "count": injected_count,
                "rate": float(injected_count / total) if total else np.nan,
            },
        ]
    )


def summarize_incongruity(counts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for global_label in counts.index:
        row_total = int(counts.loc[global_label].sum())
        if row_total == 0:
            continue
        for local_label in counts.columns:
            if global_label == local_label:
                continue
            count = int(counts.loc[global_label, local_label])
            if count == 0:
                continue
            rows.append(
                {
                    "global_label": global_label,
                    "local_label": local_label,
                    "count": count,
                    "row_probability": float(count / row_total),
                    "is_neutral_transition": bool(global_label == "neutral" or local_label == "neutral"),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=["global_label", "local_label", "count", "row_probability", "is_neutral_transition"]
        )
    return pd.DataFrame(rows).sort_values(["count", "row_probability"], ascending=[False, False]).reset_index(drop=True)


def summarize_entropy(probabilities: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    max_entropy = math.log2(len(probabilities.columns)) if len(probabilities.columns) else np.nan
    for global_label in probabilities.index:
        entropy = _entropy(probabilities.loc[global_label].to_numpy(dtype=float))
        rows.append(
            {
                "global_label": global_label,
                "count": int(counts.loc[global_label].sum()),
                "conditional_entropy_bits": entropy,
                "normalized_entropy": float(entropy / max_entropy) if max_entropy and not np.isnan(max_entropy) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["conditional_entropy_bits", "count"], ascending=[False, False])


def summarize_kl_divergence(probabilities: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    local_marginal = counts.sum(axis=0).to_numpy(dtype=float)
    if local_marginal.sum() == 0:
        local_marginal = np.ones(len(counts.columns), dtype=float)
    rows: list[dict[str, Any]] = []
    for global_label in probabilities.index:
        rows.append(
            {
                "global_label": global_label,
                "count": int(counts.loc[global_label].sum()),
                "kl_to_local_marginal_bits": _kl_divergence(
                    probabilities.loc[global_label].to_numpy(dtype=float),
                    local_marginal,
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["kl_to_local_marginal_bits", "count"], ascending=[False, False])


def summarize_mutual_information(counts: pd.DataFrame) -> pd.DataFrame:
    values = counts.to_numpy(dtype=float)
    total = float(values.sum())
    if total == 0:
        return pd.DataFrame(
            [
                {
                    "n": 0,
                    "mutual_information_bits": np.nan,
                    "normalized_mutual_information": np.nan,
                    "global_entropy_bits": np.nan,
                    "local_entropy_bits": np.nan,
                    "conditional_entropy_bits": np.nan,
                }
            ]
        )

    joint = values / total
    global_marginal = joint.sum(axis=1)
    local_marginal = joint.sum(axis=0)
    expected = np.outer(global_marginal, local_marginal)
    mask = joint > 0
    mutual_information = float((joint[mask] * np.log2(joint[mask] / expected[mask])).sum())
    global_entropy = _entropy(global_marginal)
    local_entropy = _entropy(local_marginal)
    conditional_entropy = global_entropy + local_entropy - mutual_information - global_entropy
    denominator = min(global_entropy, local_entropy)
    return pd.DataFrame(
        [
            {
                "n": int(total),
                "mutual_information_bits": mutual_information,
                "normalized_mutual_information": float(mutual_information / denominator) if denominator > 0 else np.nan,
                "global_entropy_bits": global_entropy,
                "local_entropy_bits": local_entropy,
                "conditional_entropy_bits": conditional_entropy,
            }
        ]
    )


def save_transition_heatmap(matrix: pd.DataFrame, outpath: Path, title: str, colorbar_label: str) -> None:
    if matrix.empty:
        return
    fig, ax = _theme_affective_transition_plot((13.5, 10))
    values = matrix.to_numpy(dtype=float)
    image = ax.imshow(values, cmap="YlGnBu", aspect="auto", vmin=0, vmax=max(0.01, float(np.nanmax(values))))
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("Local Label")
    ax.set_ylabel("Global Label")
    ax.set_title(title, fontsize=Q2_FONT_SIZES["title"])
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            if value <= 0:
                continue
            label = f"{value:.2f}" if value < 1 else f"{int(value)}"
            ax.text(
                col_idx,
                row_idx,
                label,
                ha="center",
                va="center",
                fontsize=Q2_FONT_SIZES["heatmap_annotation"],
                color="#17202a",
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label)
    colorbar.ax.tick_params(labelsize=Q2_FONT_SIZES["tick_label"])
    colorbar.ax.yaxis.label.set_fontsize(Q2_FONT_SIZES["axis_label"])
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def save_alignment_rate_chart(alignment_df: pd.DataFrame, outpath: Path) -> None:
    plot_df = alignment_df[alignment_df["global_label"].ne("OVERALL")].dropna(subset=["alignment_rate"]).copy()
    plot_df = plot_df[plot_df["count"].gt(0)].sort_values("alignment_rate", ascending=True)
    if plot_df.empty:
        return
    fig, ax = _theme_affective_transition_plot((11.5, 7))
    colors = [LABEL_COLORS.get(label, "#8d99ae") for label in plot_df["global_label"]]
    ax.barh(plot_df["global_label"], plot_df["alignment_rate"], color=colors, edgecolor="#233142", linewidth=0.9)
    for _, row in plot_df.iterrows():
        ax.text(
            float(row["alignment_rate"]) + 0.012,
            row["global_label"],
            f"{row['alignment_rate']:.2f} (n={int(row['count'])})",
            va="center",
            fontsize=Q2_FONT_SIZES["annotation"],
        )
    ax.set_xlim(0, min(1.08, max(0.1, float(plot_df["alignment_rate"].max()) + 0.16)))
    ax.set_xlabel("Alignment Rate")
    ax.set_ylabel("Global Label")
    ax.set_title("Affective Alignment by Global Label")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def save_neutral_injection_chart(neutral_df: pd.DataFrame, outpath: Path) -> None:
    if neutral_df.empty or neutral_df["count"].sum() == 0:
        return
    plot_df = neutral_df.copy()
    color_map = {"neutral": LABEL_COLORS["neutral"], "non_neutral": "#ef476f"}
    fig, ax = _theme_affective_transition_plot((7.5, 6))
    bars = ax.bar(
        plot_df["local_group"].str.replace("_", " ", regex=False),
        plot_df["rate"],
        color=[color_map.get(group, "#8d99ae") for group in plot_df["local_group"]],
        edgecolor="#233142",
        linewidth=1.0,
    )
    for bar, (_, row) in zip(bars, plot_df.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{row['rate']:.2f}\nn={int(row['count'])}",
            ha="center",
            va="bottom",
            fontsize=Q2_FONT_SIZES["annotation"],
        )
    ax.set_ylim(0, min(1.1, max(0.1, float(plot_df["rate"].max()) + 0.16)))
    ax.set_xlabel("Local Affect Group")
    ax.set_ylabel("Rate within Neutral Global Templates")
    ax.set_title("Neutral Template Affect Injection")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def save_top_incongruity_chart(incongruity_df: pd.DataFrame, outpath: Path, top_n: int = 20) -> None:
    plot_df = incongruity_df[~incongruity_df["is_neutral_transition"].astype(bool)].head(top_n).copy()
    if plot_df.empty:
        plot_df = incongruity_df.head(top_n).copy()
    if plot_df.empty:
        return
    plot_df["transition"] = plot_df["global_label"].astype(str) + " -> " + plot_df["local_label"].astype(str)
    plot_df = plot_df.sort_values("count", ascending=True)
    fig, ax = _theme_affective_transition_plot((12.5, 8))
    colors = [LABEL_COLORS.get(label, "#8d99ae") for label in plot_df["global_label"]]
    ax.barh(plot_df["transition"], plot_df["count"], color=colors, edgecolor="#233142", linewidth=0.9)
    for _, row in plot_df.iterrows():
        ax.text(
            int(row["count"]) + max(1.0, float(plot_df["count"].max()) * 0.012),
            row["transition"],
            f"p={row['row_probability']:.2f}",
            va="center",
            fontsize=Q2_FONT_SIZES["annotation"],
        )
    ax.set_xlabel("Count")
    ax.set_ylabel("Off-Diagonal Transition")
    ax.set_title(f"Top {len(plot_df)} Affective Incongruity Transitions")
    ax.set_xlim(0, float(plot_df["count"].max()) * 1.18)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def save_entropy_chart(entropy_df: pd.DataFrame, outpath: Path) -> None:
    plot_df = entropy_df[entropy_df["count"].gt(0)].copy()
    if plot_df.empty:
        return
    plot_df = plot_df.sort_values("conditional_entropy_bits", ascending=True)
    fig, ax = _theme_affective_transition_plot((11.5, 7))
    colors = [LABEL_COLORS.get(label, "#8d99ae") for label in plot_df["global_label"]]
    ax.barh(plot_df["global_label"], plot_df["conditional_entropy_bits"], color=colors, edgecolor="#233142", linewidth=0.9)
    for _, row in plot_df.iterrows():
        ax.text(
            float(row["conditional_entropy_bits"]) + 0.03,
            row["global_label"],
            f"{row['conditional_entropy_bits']:.2f}",
            va="center",
            fontsize=Q2_FONT_SIZES["annotation"],
        )
    ax.set_xlabel("Conditional Entropy H(Local | Global Label), bits")
    ax.set_ylabel("Global Label")
    ax.set_title("Caption Affect Flexibility by Template Affect")
    ax.set_xlim(0, float(plot_df["conditional_entropy_bits"].max()) * 1.15)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def save_kl_divergence_chart(kl_df: pd.DataFrame, outpath: Path) -> None:
    plot_df = kl_df[kl_df["count"].gt(0)].copy()
    if plot_df.empty:
        return
    plot_df = plot_df.sort_values("kl_to_local_marginal_bits", ascending=True)
    fig, ax = _theme_affective_transition_plot((11.5, 7))
    colors = [LABEL_COLORS.get(label, "#8d99ae") for label in plot_df["global_label"]]
    ax.barh(plot_df["global_label"], plot_df["kl_to_local_marginal_bits"], color=colors, edgecolor="#233142", linewidth=0.9)
    for _, row in plot_df.iterrows():
        ax.text(
            float(row["kl_to_local_marginal_bits"]) + 0.03,
            row["global_label"],
            f"{row['kl_to_local_marginal_bits']:.2f}",
            va="center",
            fontsize=Q2_FONT_SIZES["annotation"],
        )
    ax.set_xlabel("KL Divergence from Overall Local Affect, bits")
    ax.set_ylabel("Global Label")
    ax.set_title("How Strongly Each Template Affect Shifts Caption Affect")
    ax.set_xlim(0, float(plot_df["kl_to_local_marginal_bits"].max()) * 1.15)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def save_transition_share_chart(transition_probabilities: pd.DataFrame, transition_counts: pd.DataFrame, outpath: Path) -> None:
    rows: list[dict[str, Any]] = []
    for global_label in transition_probabilities.index:
        row_total = int(transition_counts.loc[global_label].sum())
        if row_total == 0:
            continue
        aligned = float(transition_probabilities.loc[global_label, global_label])
        neutral_share = float(transition_probabilities.loc[global_label, "neutral"]) if "neutral" in transition_probabilities.columns else 0.0
        transformed = max(0.0, 1.0 - aligned - (neutral_share if global_label != "neutral" else 0.0))
        rows.append(
            {
                "global_label": global_label,
                "aligned": aligned,
                "muted_to_neutral": 0.0 if global_label == "neutral" else neutral_share,
                "transformed": transformed,
                "count": row_total,
            }
        )
    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        return
    plot_df = plot_df.sort_values("count", ascending=False)
    fig, ax = _theme_affective_transition_plot((12.5, 7))
    bottom = np.zeros(len(plot_df))
    segments = [
        ("aligned", "#2a9d8f", "Aligned"),
        ("muted_to_neutral", LABEL_COLORS["neutral"], "Muted to Neutral"),
        ("transformed", "#ef476f", "Transformed"),
    ]
    x = np.arange(len(plot_df))
    for column, color, label in segments:
        values = plot_df[column].to_numpy(dtype=float)
        ax.bar(x, values, bottom=bottom, color=color, edgecolor="#233142", linewidth=0.7, label=label)
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["global_label"], rotation=40, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Global Label")
    ax.set_ylabel("Share of Dominant Transitions")
    ax.set_title("Aligned vs Muted vs Transformed Caption Affect")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def save_affective_transition_analysis(df: pd.DataFrame, outdir: Path, threshold: float) -> None:
    outdir = Path(outdir)
    transition_dir = ensure_dir(outdir / "affective_transitions")
    threshold_counts, threshold_probabilities = compute_threshold_cooccurrence(df, threshold)
    dominant_pairs, transition_counts, transition_probabilities = compute_dominant_transition_counts(df, threshold)

    threshold_counts.to_csv(transition_dir / "threshold_cooccurrence_counts.csv")
    threshold_probabilities.to_csv(transition_dir / "threshold_cooccurrence_probabilities.csv")
    dominant_pairs.to_csv(transition_dir / "dominant_transition_pairs.csv", index=False)
    transition_counts.to_csv(transition_dir / "dominant_transition_counts.csv")
    transition_probabilities.to_csv(transition_dir / "dominant_transition_probabilities.csv")

    alignment_df = summarize_alignment(transition_counts)
    neutral_injection_df = summarize_neutral_injection(transition_counts)
    incongruity_df = summarize_incongruity(transition_counts)
    entropy_df = summarize_entropy(transition_probabilities, transition_counts)
    kl_df = summarize_kl_divergence(transition_probabilities, transition_counts)

    alignment_df.to_csv(transition_dir / "alignment_rates.csv", index=False)
    neutral_injection_df.to_csv(transition_dir / "neutral_injection_rates.csv", index=False)
    incongruity_df.to_csv(transition_dir / "incongruity_transitions.csv", index=False)
    entropy_df.to_csv(transition_dir / "conditional_entropy_by_global_label.csv", index=False)
    summarize_mutual_information(transition_counts).to_csv(transition_dir / "mutual_information.csv", index=False)
    kl_df.to_csv(transition_dir / "kl_divergence_by_global_label.csv", index=False)

    save_transition_heatmap(
        threshold_counts,
        transition_dir / "fig_threshold_cooccurrence_counts.png",
        f"Global vs Local Affect Co-occurrence Counts (Score > {threshold:.1f})",
        "Count",
    )
    save_transition_heatmap(
        threshold_probabilities,
        transition_dir / "fig_threshold_cooccurrence_probabilities.png",
        f"Row-Normalized Co-occurrence: P(Local Affect | Global Affect), Score > {threshold:.1f}",
        "Probability",
    )
    save_transition_heatmap(
        transition_counts,
        transition_dir / "fig_dominant_transition_counts.png",
        f"Dominant Global-to-Local Affect Transitions (Score > {threshold:.1f})",
        "Count",
    )
    save_transition_heatmap(
        transition_probabilities,
        transition_dir / "fig_dominant_transition_probabilities.png",
        f"Dominant Transition Matrix: P(Local Affect | Global Affect), Score > {threshold:.1f}",
        "Probability",
    )
    save_alignment_rate_chart(alignment_df, transition_dir / "fig_alignment_rates.png")
    save_neutral_injection_chart(neutral_injection_df, transition_dir / "fig_neutral_injection.png")
    save_top_incongruity_chart(incongruity_df, transition_dir / "fig_top_incongruity_transitions.png")
    save_entropy_chart(entropy_df, transition_dir / "fig_conditional_entropy.png")
    save_kl_divergence_chart(kl_df, transition_dir / "fig_kl_divergence.png")
    save_transition_share_chart(
        transition_probabilities,
        transition_counts,
        transition_dir / "fig_transition_type_shares.png",
    )


def build_word_frequencies(
    texts: pd.Series,
    min_token_len: int,
) -> dict[str, float]:
    frequencies: dict[str, float] = {}
    for text in texts.fillna("").astype(str):
        for token in _tokenize(text):
            normalized = token.strip("'").lower()
            if not normalized:
                continue
            if len(token) < min_token_len:
                continue
            if normalized.isdigit():
                continue
            if normalized in DEFAULT_STOPWORDS:
                continue
            frequencies[normalized] = frequencies.get(normalized, 0.0) + 1.0
    return frequencies


def build_filtered_text(text: str, min_token_len: int) -> str:
    if not isinstance(text, str):
        return ""
    tokens: list[str] = []
    for token in _tokenize(text):
        normalized = token.strip("'").lower()
        if not normalized:
            continue
        if len(normalized) < min_token_len:
            continue
        if normalized.isdigit():
            continue
        if normalized in DEFAULT_STOPWORDS:
            continue
        tokens.append(normalized)
    return " ".join(tokens)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def _rank_words_tfidf(
    texts_by_label: dict[str, list[str]],
    freqs_by_label: dict[str, dict[str, float]],
) -> tuple[dict[str, dict[str, float]], str]:
    corpus: list[str] = []
    owners: list[str] = []
    for label, texts in texts_by_label.items():
        for text in texts:
            if text:
                corpus.append(text)
                owners.append(label)
    if not corpus:
        return {}, "none"

    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", lowercase=False)
    matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    label_order = sorted(texts_by_label)

    centroids: list[np.ndarray] = []
    for label in label_order:
        mask = np.array([owner == label for owner in owners], dtype=bool)
        if not mask.any():
            centroids.append(np.zeros(matrix.shape[1], dtype=np.float32))
            continue
        centroid = np.asarray(matrix[mask].mean(axis=0)).ravel().astype(np.float32)
        centroids.append(centroid)
    centroid_matrix = _normalize_rows(np.vstack(centroids))
    token_to_idx = {token: idx for idx, token in enumerate(feature_names.tolist())}

    ranked: dict[str, dict[str, float]] = {}
    for label_idx, label in enumerate(label_order):
        scores: dict[str, float] = {}
        for token, freq in freqs_by_label.get(label, {}).items():
            token_idx = token_to_idx.get(token)
            if token_idx is None:
                continue
            word_vec = np.zeros(matrix.shape[1], dtype=np.float32)
            word_vec[token_idx] = 1.0
            sims = centroid_matrix @ word_vec
            own = float(sims[label_idx])
            others = np.delete(sims, label_idx)
            margin = own - float(others.max()) if len(others) else own
            weight = max(margin, 0.0) * math.log1p(float(freq))
            if weight > 0:
                scores[token] = weight
        ranked[label] = scores
    return ranked, "tfidf"


def _rank_words_sbert(
    texts_by_label: dict[str, list[str]],
    freqs_by_label: dict[str, dict[str, float]],
    model_name: str,
    max_docs_per_class: int,
    random_seed: int,
) -> tuple[dict[str, dict[str, float]], str]:
    sentence_transformer_cls = try_import_sentence_transformers()
    if sentence_transformer_cls is None:
        return {}, "none"

    label_order = sorted(texts_by_label)
    centroid_texts: list[str] = []
    centroid_owners: list[str] = []
    for label in label_order:
        label_texts = [text for text in texts_by_label.get(label, []) if text]
        if not label_texts:
            continue
        if len(label_texts) > max_docs_per_class:
            rng = np.random.default_rng(random_seed + abs(hash(label)) % 10007)
            take_idx = rng.choice(len(label_texts), size=max_docs_per_class, replace=False)
            label_texts = [label_texts[int(idx)] for idx in take_idx]
        centroid_texts.extend(label_texts)
        centroid_owners.extend([label] * len(label_texts))
    if not centroid_texts:
        return {}, "none"

    model = sentence_transformer_cls(model_name)
    doc_embeddings = np.asarray(
        model.encode(centroid_texts, show_progress_bar=False, normalize_embeddings=True),
        dtype=np.float32,
    )

    centroid_rows: list[np.ndarray] = []
    for label in label_order:
        mask = np.array([owner == label for owner in centroid_owners], dtype=bool)
        if not mask.any():
            centroid_rows.append(np.zeros(doc_embeddings.shape[1], dtype=np.float32))
            continue
        centroid = doc_embeddings[mask].mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm > 0:
            centroid = centroid / norm
        centroid_rows.append(centroid.astype(np.float32))
    centroid_matrix = np.vstack(centroid_rows)

    vocab = sorted({token for scores in freqs_by_label.values() for token in scores})
    if not vocab:
        return {}, "none"
    word_embeddings = np.asarray(
        model.encode(vocab, show_progress_bar=False, normalize_embeddings=True),
        dtype=np.float32,
    )
    similarities = word_embeddings @ centroid_matrix.T
    token_to_idx = {token: idx for idx, token in enumerate(vocab)}

    ranked: dict[str, dict[str, float]] = {}
    for label_idx, label in enumerate(label_order):
        scores: dict[str, float] = {}
        for token, freq in freqs_by_label.get(label, {}).items():
            token_idx = token_to_idx.get(token)
            if token_idx is None:
                continue
            sims = similarities[token_idx]
            own = float(sims[label_idx])
            others = np.delete(sims, label_idx)
            margin = own - float(others.max()) if len(others) else own
            weight = max(margin, 0.0) * math.log1p(float(freq))
            if weight > 0:
                scores[token] = weight
        ranked[label] = scores
    return ranked, "sbert"


def rank_words_for_wordclouds(
    dominant_df: pd.DataFrame,
    min_token_len: int,
    embedding_model: str,
    max_docs_per_class: int,
    random_seed: int,
) -> tuple[dict[str, dict[str, float]], pd.DataFrame, str]:
    working = dominant_df.copy()
    working["filtered_text"] = working["source_text"].fillna("").astype(str).map(
        lambda text: build_filtered_text(text, min_token_len=min_token_len)
    )
    working = working[working["filtered_text"].str.strip().ne("")].copy()
    if working.empty:
        return {}, pd.DataFrame(columns=["label", "word", "weight", "frequency"]), "none"

    texts_by_label = (
        working.groupby("dominant_label", observed=True)["filtered_text"].apply(list).to_dict()
    )
    freqs_by_label = {
        label: build_word_frequencies(pd.Series(texts), min_token_len=min_token_len)
        for label, texts in texts_by_label.items()
    }

    ranked, used_backend = _rank_words_sbert(
        texts_by_label=texts_by_label,
        freqs_by_label=freqs_by_label,
        model_name=embedding_model,
        max_docs_per_class=max_docs_per_class,
        random_seed=random_seed,
    )
    if used_backend == "none":
        raise RuntimeError("sentence-transformers is unavailable for keyword ranking.")

    rows: list[dict[str, Any]] = []
    for label, scores in ranked.items():
        for word, weight in sorted(scores.items(), key=lambda item: item[1], reverse=True):
            rows.append(
                {
                    "label": label,
                    "word": word,
                    "weight": float(weight),
                    "frequency": float(freqs_by_label.get(label, {}).get(word, 0.0)),
                }
            )
    return ranked, pd.DataFrame(rows), used_backend


def save_score_histogram_grid(
    df: pd.DataFrame,
    context: str,
    outpath: Path,
    bins: int,
) -> None:
    suffix = _normalize_context_name(context)
    columns = [f"{label}_{suffix}" for label in ALL_AFFECT_LABELS if f"{label}_{suffix}" in df.columns]
    if not columns:
        return

    _configure_q2_plot_style()
    ncols = 4
    nrows = math.ceil(len(columns) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(17, 4.25 * nrows))
    axes_list = np.atleast_1d(axes).ravel().tolist()

    for ax, label in zip(axes_list, ALL_AFFECT_LABELS):
        column = f"{label}_{suffix}"
        if column not in df.columns:
            ax.axis("off")
            continue
        scores = pd.to_numeric(df[column], errors="coerce").dropna()
        color = LABEL_COLORS.get(label, CONTEXT_COLORS[suffix])
        ax.hist(scores, bins=bins, color=color, alpha=0.85, edgecolor="#223047", linewidth=0.9)
        if not scores.empty:
            ax.axvline(float(scores.mean()), color="#111111", linestyle="--", linewidth=1.2, alpha=0.85)
        ax.set_title(f"{label}_{suffix}", fontsize=Q2_FONT_SIZES["subplot_title"])
        ax.set_xlim(0, 1)
        ax.grid(True, axis="y")

    for ax in axes_list[len(ALL_AFFECT_LABELS) :]:
        ax.axis("off")

    fig.suptitle(
        f"Histogram of Affect Scores ({suffix.title()} Context)",
        fontsize=Q2_FONT_SIZES["suptitle"],
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def save_dominant_proportion_chart(
    summary_df: pd.DataFrame,
    context: str,
    threshold: float,
    outpath: Path,
) -> None:
    if summary_df.empty:
        return
    plot_df = summary_df.copy()
    fig, ax = _theme_plot((12.5, 6.5))
    colors = [LABEL_COLORS.get(label, CONTEXT_COLORS[_normalize_context_name(context)]) for label in plot_df["label"]]
    ax.bar(plot_df["label"], plot_df["proportion"], color=colors, edgecolor="#233142", linewidth=1.0, alpha=0.95)
    ax.set_title(
        f"Proportion of Dominant Affect Classes in {context.title()} Context (Score > {threshold:.1f})"
    )
    ax.set_xlabel("Affect Class")
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, max(0.05, float(plot_df["proportion"].max()) * 1.18))
    ax.tick_params(axis="x", rotation=40, labelsize=Q2_FONT_SIZES["xtick_label"])
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def _smooth_histogram(counts: np.ndarray) -> np.ndarray:
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=float)
    kernel /= kernel.sum()
    return np.convolve(counts, kernel, mode="same")


def save_dominant_score_overlay(
    dominant_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    context: str,
    threshold: float,
    outpath: Path,
    bins: int,
    top_n: int,
) -> None:
    if dominant_df.empty or summary_df.empty:
        return
    top_labels = summary_df.head(top_n)["label"].tolist()
    if not top_labels:
        return

    fig, ax = _theme_plot((13, 6.5))
    edges = np.linspace(threshold, 1.0, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    legend_labels: list[str] = []
    for label in top_labels:
        values = dominant_df.loc[dominant_df["dominant_label"] == label, "dominant_score"].dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        color = LABEL_COLORS.get(label, CONTEXT_COLORS[_normalize_context_name(context)])
        ax.hist(
            values,
            bins=edges,
            color=color,
            alpha=0.42,
            edgecolor="#111111",
            linewidth=0.9,
            label=f"Dominant-{label.title()} Memes",
        )
        counts, _ = np.histogram(values, bins=edges)
        ax.plot(centers, _smooth_histogram(counts), color=color, linewidth=2.2)
        legend_labels.append(label)

    if not legend_labels:
        plt.close(fig)
        return

    ax.set_title(
        f"Distribution of Dominant Affect Scores in {context.title()} Context (Score > {threshold:.1f})"
    )
    ax.set_xlabel("Dominant Score")
    ax.set_ylabel("Count")
    ax.set_xlim(threshold, 1.0)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def save_mean_dominant_score_chart(
    summary_df: pd.DataFrame,
    context: str,
    threshold: float,
    outpath: Path,
) -> None:
    if summary_df.empty:
        return
    plot_df = summary_df.sort_values(["mean_score", "count"], ascending=[False, False]).copy()
    fig, ax = _theme_plot((12.5, 6.5))
    colors = [LABEL_COLORS.get(label, CONTEXT_COLORS[_normalize_context_name(context)]) for label in plot_df["label"]]
    bars = ax.bar(
        plot_df["label"],
        plot_df["mean_score"],
        color=colors,
        edgecolor="#233142",
        linewidth=1.0,
        alpha=0.95,
    )
    for bar, count in zip(bars, plot_df["count"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"n={int(count)}",
            ha="center",
            va="bottom",
            fontsize=Q2_FONT_SIZES["annotation"],
        )
    ax.set_title(
        f"Mean Dominant Affect Score by Class in {context.title()} Context (Score > {threshold:.1f})"
    )
    ax.set_xlabel("Affect Class")
    ax.set_ylabel("Mean Dominant Score")
    ax.set_ylim(threshold, 1.03)
    ax.tick_params(axis="x", rotation=40, labelsize=Q2_FONT_SIZES["xtick_label"])
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def save_wordclouds(
    dominant_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    context: str,
    threshold: float,
    outdir: Path,
    max_words: int,
    min_token_len: int,
    keyword_embedding_model: str,
    keyword_max_docs_per_class: int,
    random_seed: int,
) -> tuple[int, int, str]:
    if dominant_df.empty or summary_df.empty:
        return 0, 0, "none"

    context_name = _normalize_context_name(context)
    context_dir = ensure_dir(outdir / f"wordclouds_{context_name}")
    ranked_words, keyword_df, used_backend = rank_words_for_wordclouds(
        dominant_df=dominant_df,
        min_token_len=min_token_len,
        embedding_model=keyword_embedding_model,
        max_docs_per_class=keyword_max_docs_per_class,
        random_seed=random_seed,
    )
    if WordCloud is None:
        return 0, 0, used_backend

    cloud_arrays: list[tuple[str, np.ndarray]] = []
    saved = 0
    for label in ALL_AFFECT_LABELS:
        frequencies = ranked_words.get(label, {})
        if not frequencies:
            continue
        base_color = LABEL_COLORS.get(label, CONTEXT_COLORS[context_name])
        wordcloud = WordCloud(
            width=1400,
            height=820,
            background_color="white",
            prefer_horizontal=0.92,
            collocations=False,
            max_words=max_words,
            random_state=random_seed,
            color_func=_build_wordcloud_color_func(base_color),
        ).generate_from_frequencies(frequencies)
        image = wordcloud.to_array()
        cloud_arrays.append((label, image))

        _configure_q2_plot_style()
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.imshow(image, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(
            f"Word Cloud for {label.title()} Memes ({context.title()} Context, Score > {threshold:.1f})",
            fontsize=Q2_FONT_SIZES["wordcloud_title"],
        )
        fig.tight_layout()
        fig.savefig(context_dir / f"{label}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved += 1

    if not cloud_arrays:
        return 0, 0, used_backend

    ncols = 3
    nrows = math.ceil(len(cloud_arrays) / ncols)
    _configure_q2_plot_style()
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5.2 * nrows))
    axes_list = np.atleast_1d(axes).ravel().tolist()
    for ax, (label, image) in zip(axes_list, cloud_arrays):
        ax.imshow(image, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(label, fontsize=Q2_FONT_SIZES["subplot_title"])
    for ax in axes_list[len(cloud_arrays) :]:
        ax.axis("off")
    fig.suptitle(
        f"Word Clouds by Dominant Affect Class ({context.title()} Context)",
        fontsize=Q2_FONT_SIZES["suptitle"],
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(outdir / f"fig_wordcloud_grid_{context_name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return saved, len(cloud_arrays), used_backend


def run_analysis(args: argparse.Namespace) -> Path:
    outdir = ensure_dir(args.results_dir)
    input_jsonl = Path(args.input_jsonl).expanduser().resolve()
    if input_jsonl.exists():
        df = load_combined_sentiment_emotion_jsonl(input_jsonl)
        bundle_metadata = {
            "data_source": "combined_sentiments_emotions_jsonl",
            "input_jsonl": str(input_jsonl),
            "input_jsonl_rows": int(len(df)),
        }
    else:
        data_dir = Path(args.analysis_parquet).expanduser().resolve().parent
        df, bundle_metadata = build_current_data_bundle_raw(data_dir)
        bundle_metadata["data_source"] = "current_data_bundle_raw"
    df = attach_reddit_metadata(df, args.analysis_parquet)
    df = attach_topic_assignments(df, args)
    df["score"] = pd.to_numeric(df.get("score"), errors="coerce").fillna(0.0)
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
    df = df[
        df["global_text"].fillna("").astype(str).str.strip().ne("")
        | df["local_text"].fillna("").astype(str).str.strip().ne("")
    ].copy()
    bundle_metadata["bundle_rows_with_any_text"] = int(len(df))
    if args.max_rows > 0:
        df = df.head(args.max_rows).copy()

    df = add_affect_layers(df, args)
    save_topic_affect_analysis(df, outdir=outdir, args=args)
    if args.topics_only:
        return outdir
    save_affective_transition_analysis(df, outdir=outdir, threshold=args.emotion_threshold)
    for context in ("global", "local"):
        save_score_histogram_grid(
            df,
            context=context,
            outpath=outdir / f"fig_histogram_scores_{context}.png",
            bins=args.hist_bins,
        )

        dominant_df, summary_df = compute_dominant_summary(df, context=context, threshold=args.emotion_threshold)

        save_dominant_proportion_chart(
            summary_df,
            context=context,
            threshold=args.emotion_threshold,
            outpath=outdir / f"fig_dominant_proportions_{context}.png",
        )
        save_dominant_score_overlay(
            dominant_df,
            summary_df,
            context=context,
            threshold=args.emotion_threshold,
            outpath=outdir / f"fig_dominant_score_distribution_{context}.png",
            bins=args.hist_bins,
            top_n=args.dominant_top_n,
        )
        save_mean_dominant_score_chart(
            summary_df,
            context=context,
            threshold=args.emotion_threshold,
            outpath=outdir / f"fig_mean_dominant_scores_{context}.png",
        )
        saved_clouds, grid_clouds, used_keyword_backend = save_wordclouds(
            dominant_df=dominant_df,
            summary_df=summary_df,
            context=context,
            threshold=args.emotion_threshold,
            outdir=outdir,
            max_words=args.wordcloud_max_words,
            min_token_len=args.wordcloud_min_token_len,
            keyword_embedding_model=args.keyword_embedding_model,
            keyword_max_docs_per_class=args.keyword_max_docs_per_class,
            random_seed=args.random_seed,
        )
    return outdir


def main() -> None:
    args = parse_args()
    outdir = run_analysis(args)
    print(outdir)


if __name__ == "__main__":
    main()
