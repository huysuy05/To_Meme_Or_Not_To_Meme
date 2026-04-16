#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from common import (
    add_sentiment_layers,
    configure_plot_style,
    ensure_dir,
    load_analysis_dataframe,
    try_import_sentence_transformers,
    write_run_metadata,
    write_summary_markdown,
)


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")
PHASE_ORDER = ["early", "peak", "mature", "decline", "tail"]
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Research Question 2: paired global/local context analysis with sentiment mismatch, "
            "semantic alignment, topic transitions, lifecycle phases, keyword summaries, local complexity, "
            "and normalized temporal analyses."
        )
    )
    parser.add_argument("--analysis-parquet", required=True)
    parser.add_argument("--results-dir", default="analysis/results/q2_global_local_context")
    parser.add_argument("--template-ranking", choices=["count", "score"], default="count")
    parser.add_argument("--top-k-templates", type=int, default=10)
    parser.add_argument("--min-template-posts", type=int, default=50)
    parser.add_argument("--monthly-min-posts", type=int, default=30)
    parser.add_argument("--rolling-months", type=int, default=3)
    parser.add_argument("--lifecycle-freq", default="M")
    parser.add_argument("--low-frac", type=float, default=0.25)
    parser.add_argument("--sustain-periods", type=int, default=3)
    parser.add_argument("--zero-run-periods", type=int, default=4)
    parser.add_argument("--alignment-backend", choices=["auto", "tfidf", "sbert"], default="auto")
    parser.add_argument("--sbert-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--alignment-max-features", type=int, default=10000)
    parser.add_argument("--emotion-backend", choices=["cardiff", "vader"], default="cardiff")
    parser.add_argument("--emotion-cache-dir", default="analysis/cache/q2_cardiff_affect")
    parser.add_argument("--emotion-batch-size", type=int, default=32)
    parser.add_argument("--emotion-max-length", type=int, default=512)
    parser.add_argument("--emotion-threshold", type=float, default=0.7)
    parser.add_argument("--emotion-top-n", type=int, default=7)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--topic-model", choices=["bertopic", "nmf"], default="bertopic")
    parser.add_argument("--topic-embedding-model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--bertopic-batch-size", type=int, default=128)
    parser.add_argument("--n-topics", type=int, default=12)
    parser.add_argument("--topic-max-features", type=int, default=6000)
    parser.add_argument("--topic-min-df", type=int, default=10)
    parser.add_argument("--max-topic-docs-fit", type=int, default=60000)
    parser.add_argument("--keyword-top-n", type=int, default=12)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def _normalize_freq(freq: str) -> str:
    if str(freq).upper() == "M":
        return "MS"
    return freq


def _tokenize(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    return TOKEN_PATTERN.findall(text.lower())


def _text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


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
        emotion_cache.to_parquet(emotion_cache_path, index=False)
        sentiment_cache.to_parquet(sentiment_cache_path, index=False)

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

    enriched["global_dominant_emotion"] = (
        enriched[global_emotion_cols].idxmax(axis=1).str.replace("_global", "", regex=False)
    )
    enriched["local_dominant_emotion"] = (
        enriched[local_emotion_cols].idxmax(axis=1).str.replace("_local", "", regex=False)
    )
    enriched["global_sentiment_label"] = (
        enriched[global_sentiment_cols].idxmax(axis=1).str.replace("_global", "", regex=False)
    )
    enriched["local_sentiment_label"] = (
        enriched[local_sentiment_cols].idxmax(axis=1).str.replace("_local", "", regex=False)
    )
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
        return add_cardiff_affect_layers(
            df,
            cache_dir=args.emotion_cache_dir,
            batch_size=args.emotion_batch_size,
            max_length=args.emotion_max_length,
        )
    enriched = add_sentiment_layers(df)
    enriched["global_dominant_emotion"] = enriched["global_sentiment_label"]
    enriched["local_dominant_emotion"] = enriched["local_sentiment_label"]
    enriched["emotion_mismatch"] = enriched["sentiment_mismatch"]
    return enriched


def add_local_complexity_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    local_tokens = df["local_text"].map(_tokenize)
    global_tokens = df["global_text"].map(_tokenize)
    df["local_char_len"] = df["local_text"].astype(str).str.len()
    df["local_token_len"] = local_tokens.map(len)
    df["global_token_len"] = global_tokens.map(len)
    df["user_text_segments"] = df["user_text_list"].map(len)
    df["local_keyword_count"] = df["local_keywords_list"].map(len)
    df["global_keyword_count"] = df["global_keywords_list"].map(len)
    df["local_unique_tokens"] = local_tokens.map(lambda values: len(set(values)))
    df["global_unique_tokens"] = global_tokens.map(lambda values: len(set(values)))
    df["local_lexical_diversity"] = df["local_unique_tokens"] / df["local_token_len"].clip(lower=1)
    df["global_lexical_diversity"] = df["global_unique_tokens"] / df["global_token_len"].clip(lower=1)
    complexity_cols = [
        "local_char_len",
        "local_token_len",
        "user_text_segments",
        "local_keyword_count",
        "local_lexical_diversity",
    ]
    for column in complexity_cols:
        values = pd.to_numeric(df[column], errors="coerce")
        std = float(values.std(ddof=0))
        if std == 0 or np.isnan(std):
            df[f"{column}_z"] = 0.0
        else:
            df[f"{column}_z"] = (values - float(values.mean())) / std
    z_cols = [f"{column}_z" for column in complexity_cols]
    df["local_complexity_index"] = df[z_cols].mean(axis=1)
    return df


def compute_alignment(df: pd.DataFrame, backend: str, sbert_model: str, max_features: int) -> pd.DataFrame:
    df = df.copy()
    mask = df["global_text"].astype(str).str.strip().ne("") & df["local_text"].astype(str).str.strip().ne("")
    df["global_local_similarity"] = np.nan
    df["alignment_backend"] = "none"
    if not mask.any():
        return df

    subset = df.loc[mask, ["global_text", "local_text"]].copy()
    if backend in {"auto", "sbert"}:
        sentence_transformer_cls = try_import_sentence_transformers()
        if sentence_transformer_cls is not None:
            model = sentence_transformer_cls(sbert_model)
            global_emb = np.asarray(
                model.encode(subset["global_text"].tolist(), show_progress_bar=False, normalize_embeddings=True),
                dtype=np.float32,
            )
            local_emb = np.asarray(
                model.encode(subset["local_text"].tolist(), show_progress_bar=False, normalize_embeddings=True),
                dtype=np.float32,
            )
            similarity = (global_emb * local_emb).sum(axis=1)
            df.loc[mask, "global_local_similarity"] = similarity
            df["alignment_backend"] = "sbert"
            return df
        if backend == "sbert":
            raise RuntimeError("sentence-transformers is unavailable for --alignment-backend=sbert")

    vectorizer = TfidfVectorizer(stop_words="english", min_df=5, max_features=max_features)
    combined_corpus = pd.concat([subset["global_text"], subset["local_text"]], ignore_index=True).tolist()
    matrix = vectorizer.fit_transform(combined_corpus)
    midpoint = len(subset)
    global_matrix = matrix[:midpoint]
    local_matrix = matrix[midpoint:]
    similarity = global_matrix.multiply(local_matrix).sum(axis=1).A1
    df.loc[mask, "global_local_similarity"] = similarity
    df["alignment_backend"] = "tfidf"
    return df


def select_top_templates(df: pd.DataFrame, ranking: str, top_k: int, min_posts: int) -> pd.DataFrame:
    group = (
        df.groupby("template_final", observed=True)
        .agg(
            total_posts=("template_final", "size"),
            mean_score=("score", "mean"),
            mean_alignment=("global_local_similarity", "mean"),
        )
        .reset_index()
    )
    group = group[group["total_posts"] >= min_posts].copy()
    if ranking == "score":
        group = group.sort_values(["mean_score", "total_posts"], ascending=[False, False]).reset_index(drop=True)
    else:
        group = group.sort_values(["total_posts", "mean_score"], ascending=[False, False]).reset_index(drop=True)
    selected = group.head(top_k).copy()
    selected["template_rank"] = range(1, len(selected) + 1)
    return selected


def compute_paired_sentiment_outputs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_counts = (
        df.groupby(["global_sentiment_label", "local_sentiment_label"], observed=True)
        .size()
        .rename("count")
        .reset_index()
    )
    pair_matrix = (
        pair_counts.pivot(index="global_sentiment_label", columns="local_sentiment_label", values="count")
        .fillna(0)
        .astype(int)
    )
    row_totals = pair_matrix.sum(axis=1).replace(0, np.nan)
    normalized = pair_matrix.div(row_totals, axis=0).fillna(0.0)
    return pair_matrix, normalized


def compute_paired_emotion_outputs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_counts = (
        df.groupby(["global_dominant_emotion", "local_dominant_emotion"], observed=True)
        .size()
        .rename("count")
        .reset_index()
    )
    if pair_counts.empty:
        return pd.DataFrame(), pd.DataFrame()
    pair_matrix = (
        pair_counts.pivot(index="global_dominant_emotion", columns="local_dominant_emotion", values="count")
        .fillna(0)
        .astype(int)
    )
    row_totals = pair_matrix.sum(axis=1).replace(0, np.nan)
    normalized = pair_matrix.div(row_totals, axis=0).fillna(0.0)
    return pair_matrix, normalized


def compute_dominant_affect_counts(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in df.columns:
        return pd.DataFrame()
    counts = df[column].fillna("unavailable").astype(str).value_counts().rename_axis("label").reset_index(name="count")
    counts["share"] = counts["count"] / counts["count"].sum()
    return counts


def compute_threshold_proportions(df: pd.DataFrame, suffix: str, threshold: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = max(len(df), 1)
    for label in ALL_AFFECT_LABELS:
        column = f"{label}_{suffix}"
        if column not in df.columns:
            continue
        share = float(pd.to_numeric(df[column], errors="coerce").gt(threshold).mean())
        rows.append({"label": label, "share": share, "count": int(round(share * total))})
    return pd.DataFrame(rows).sort_values("share", ascending=False)


def compute_monthly_affect_trends(
    df: pd.DataFrame,
    suffix: str,
    threshold: float,
    top_n: int,
) -> pd.DataFrame:
    available_cols = [f"{label}_{suffix}" for label in ALL_AFFECT_LABELS if f"{label}_{suffix}" in df.columns]
    if not available_cols:
        return pd.DataFrame()
    monthly_total = df.groupby("year_month", observed=True).size().rename("posts").reset_index()
    threshold_counts = (
        df.groupby("year_month", observed=True)[available_cols]
        .agg(lambda series: series.gt(threshold).sum())
        .reset_index()
    )
    monthly = threshold_counts.merge(monthly_total, on="year_month", how="left")
    for column in available_cols:
        monthly[f"{column}_normalized"] = pd.to_numeric(monthly[column], errors="coerce") / monthly["posts"].replace(0, np.nan)
    monthly["month_start"] = pd.PeriodIndex(monthly["year_month"], freq="M").to_timestamp()
    label_strength = [
        (
            label,
            float(monthly.get(f"{label}_{suffix}_normalized", pd.Series(dtype=float)).mean())
            if f"{label}_{suffix}_normalized" in monthly.columns
            else float("nan"),
        )
        for label in ALL_AFFECT_LABELS
    ]
    selected_labels = [label for label, _ in sorted(label_strength, key=lambda item: item[1], reverse=True)[:top_n]]
    keep_columns = ["year_month", "month_start", "posts"]
    for label in selected_labels:
        keep_columns.extend([f"{label}_{suffix}", f"{label}_{suffix}_normalized"])
    return monthly.loc[:, [column for column in keep_columns if column in monthly.columns]]


def save_count_bar_chart(
    summary_df: pd.DataFrame,
    outpath: Path,
    title: str,
    ylabel: str,
    color: str,
) -> None:
    if summary_df.empty:
        return
    plot_df = summary_df.copy()
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(plot_df["label"], plot_df["count"], color=color, alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("Label")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def save_share_bar_chart(
    summary_df: pd.DataFrame,
    outpath: Path,
    title: str,
    ylabel: str,
    color: str,
) -> None:
    if summary_df.empty:
        return
    plot_df = summary_df.copy()
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(plot_df["label"], plot_df["share"], color=color, alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("Label")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def save_monthly_affect_plot(monthly_df: pd.DataFrame, suffix: str, outpath: Path, title: str) -> None:
    normalized_cols = [column for column in monthly_df.columns if column.endswith(f"_{suffix}_normalized")]
    if monthly_df.empty or not normalized_cols:
        return
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(13, 6))
    for column in normalized_cols:
        label = column[: -(len(f"_{suffix}_normalized"))]
        ax.plot(monthly_df["month_start"], monthly_df[column], marker="o", linewidth=1.6, label=label)
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel("Normalized Share of Posts")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def save_heatmap(
    matrix_df: pd.DataFrame,
    outpath: Path,
    title: str,
    cmap: str = "Blues",
    annotate: bool = True,
) -> None:
    if matrix_df.empty:
        return
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    values = matrix_df.to_numpy(dtype=float)
    im = ax.imshow(values, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(matrix_df.columns)))
    ax.set_xticklabels(matrix_df.columns.tolist(), rotation=45, ha="right")
    ax.set_yticks(range(len(matrix_df.index)))
    ax.set_yticklabels(matrix_df.index.tolist())
    ax.set_title(title)
    if annotate:
        for row_idx in range(values.shape[0]):
            for col_idx in range(values.shape[1]):
                value = values[row_idx, col_idx]
                text = f"{value:.2f}" if np.issubdtype(values.dtype, np.floating) else str(int(value))
                ax.text(col_idx, row_idx, text, ha="center", va="center", fontsize=9, color="#111111")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def save_alignment_distribution(df: pd.DataFrame, outpath: Path) -> None:
    plot_df = df["global_local_similarity"].dropna()
    if plot_df.empty:
        return
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(plot_df.to_numpy(), bins=40, color="#457b9d", alpha=0.85, edgecolor="white")
    ax.set_title("Distribution of Global-Local Semantic Similarity")
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Posts")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def save_sentiment_score_hexbin(df: pd.DataFrame, outpath: Path) -> None:
    plot_df = df.dropna(subset=["global_sentiment_score", "local_sentiment_score"])
    if plot_df.empty:
        return
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    hb = ax.hexbin(
        plot_df["global_sentiment_score"],
        plot_df["local_sentiment_score"],
        gridsize=30,
        cmap="viridis",
        mincnt=1,
    )
    ax.set_title("Global vs Local Sentiment Scores")
    ax.set_xlabel("Global Sentiment Score")
    ax.set_ylabel("Local Sentiment Score")
    fig.colorbar(hb, ax=ax, label="Posts")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def compute_monthly_metrics(df: pd.DataFrame, min_posts: int, rolling_months: int) -> pd.DataFrame:
    mismatch_column = "emotion_mismatch" if "emotion_mismatch" in df.columns else "sentiment_mismatch"
    monthly = (
        df.groupby("year_month", observed=True)
        .agg(
            posts=("template_final", "size"),
            mean_alignment=("global_local_similarity", "mean"),
            mismatch_rate=(mismatch_column, "mean"),
            mean_local_complexity=("local_complexity_index", "mean"),
            mean_score=("score", "mean"),
        )
        .reset_index()
        .sort_values("year_month")
    )
    monthly["month_start"] = pd.PeriodIndex(monthly["year_month"], freq="M").to_timestamp()
    monthly["is_reliable_month"] = monthly["posts"] >= min_posts
    for column in ["mean_alignment", "mismatch_rate", "mean_local_complexity", "mean_score"]:
        monthly[f"{column}_rolling"] = monthly[column].rolling(rolling_months, min_periods=1).mean()
    return monthly


def save_monthly_line_plot(
    monthly_df: pd.DataFrame,
    value_col: str,
    outpath: Path,
    title: str,
    ylabel: str,
) -> None:
    if monthly_df.empty:
        return
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(monthly_df["month_start"], monthly_df[value_col], color="#1d3557", linewidth=2)
    ax.scatter(
        monthly_df.loc[monthly_df["is_reliable_month"], "month_start"],
        monthly_df.loc[monthly_df["is_reliable_month"], value_col],
        color="#e63946",
        s=16,
    )
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def compute_template_conditioned_summary(df: pd.DataFrame, top_templates: pd.DataFrame) -> pd.DataFrame:
    selected = df[df["template_final"].isin(set(top_templates["template_final"]))].copy()
    if selected.empty:
        return pd.DataFrame()
    mismatch_column = "emotion_mismatch" if "emotion_mismatch" in selected.columns else "sentiment_mismatch"
    summary = (
        selected.groupby("template_final", observed=True)
        .agg(
            total_posts=("template_final", "size"),
            mismatch_rate=(mismatch_column, "mean"),
            mean_alignment=("global_local_similarity", "mean"),
            mean_local_complexity=("local_complexity_index", "mean"),
            mean_score=("score", "mean"),
            local_topic_diversity=("local_text", lambda s: float(s.astype(str).str.len().gt(0).mean())),
        )
        .reset_index()
    )
    return summary.merge(top_templates.loc[:, ["template_final", "template_rank"]], on="template_final", how="left")


def save_template_bar(
    summary_df: pd.DataFrame,
    value_col: str,
    outpath: Path,
    title: str,
    ylabel: str,
    color: str,
) -> None:
    if summary_df.empty:
        return
    plot_df = summary_df.sort_values("template_rank").copy()
    labels = [f"{int(rank)}. {name}" for rank, name in zip(plot_df["template_rank"], plot_df["template_final"])]
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(labels, plot_df[value_col], color=color, alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("Template")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=50)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def compute_lifecycle_phase_boundaries(
    df: pd.DataFrame,
    template_name: str,
    freq: str,
    low_frac: float,
    sustain_periods: int,
    zero_run_periods: int,
) -> dict[str, Any] | None:
    pandas_freq = _normalize_freq(freq)
    subset = df[df["template_final"] == template_name].copy().sort_values("created_utc")
    if subset.empty:
        return None
    period_counts = (
        subset.set_index("created_utc")
        .resample(pandas_freq)
        .size()
        .rename("count")
        .reset_index()
        .rename(columns={"created_utc": "period_start"})
    )
    if period_counts.empty:
        return None
    full_range = pd.date_range(period_counts["period_start"].min(), period_counts["period_start"].max(), freq=pandas_freq)
    period_counts = (
        period_counts.set_index("period_start")
        .reindex(full_range, fill_value=0)
        .rename_axis("period_start")
        .reset_index()
    )
    period_counts["smooth"] = period_counts["count"].rolling(3, min_periods=1).mean()
    peak_idx = int(period_counts["smooth"].idxmax())
    peak_period = pd.Timestamp(period_counts.loc[peak_idx, "period_start"])
    peak_value = float(period_counts.loc[peak_idx, "smooth"])
    threshold = peak_value * low_frac

    post_peak = period_counts.loc[peak_idx + 1 :].copy()
    post_peak["low"] = post_peak["smooth"] <= threshold
    post_peak["low_run"] = post_peak["low"].rolling(sustain_periods, min_periods=sustain_periods).sum()
    decline_candidates = post_peak.loc[post_peak["low_run"] == sustain_periods, "period_start"]
    decline_start = pd.NaT if decline_candidates.empty else pd.Timestamp(decline_candidates.iloc[0])

    post_peak["zero"] = post_peak["count"] == 0
    post_peak["zero_run"] = post_peak["zero"].rolling(zero_run_periods, min_periods=zero_run_periods).sum()
    expired_candidates = post_peak.loc[post_peak["zero_run"] == zero_run_periods, "period_start"]
    expired_at = pd.NaT if expired_candidates.empty else pd.Timestamp(expired_candidates.iloc[0])

    offset = pd.tseries.frequencies.to_offset(pandas_freq)
    peak_window_start = peak_period - offset
    peak_window_end = peak_period + offset
    mature_start = peak_window_end
    if pd.isna(decline_start):
        decline_start = pd.NaT
    return {
        "template_final": template_name,
        "first_seen": pd.Timestamp(subset["created_utc"].min()),
        "peak_period": peak_period,
        "peak_window_start": peak_window_start,
        "peak_window_end": peak_window_end,
        "decline_start": decline_start,
        "expired_at": expired_at,
    }


def assign_lifecycle_phases(
    df: pd.DataFrame,
    top_templates: pd.DataFrame,
    freq: str,
    low_frac: float,
    sustain_periods: int,
    zero_run_periods: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    phase_rows: list[dict[str, Any]] = []
    annotated_parts: list[pd.DataFrame] = []
    for row in top_templates.itertuples(index=False):
        boundaries = compute_lifecycle_phase_boundaries(
            df,
            template_name=row.template_final,
            freq=freq,
            low_frac=low_frac,
            sustain_periods=sustain_periods,
            zero_run_periods=zero_run_periods,
        )
        if boundaries is None:
            continue
        phase_rows.append({**boundaries, "template_rank": int(row.template_rank)})
        subset = df[df["template_final"] == row.template_final].copy()
        if subset.empty:
            continue
        subset["template_rank"] = int(row.template_rank)

        def phase_for_time(ts: pd.Timestamp) -> str:
            if ts <= boundaries["peak_window_start"]:
                return "early"
            if ts <= boundaries["peak_window_end"]:
                return "peak"
            if pd.isna(boundaries["decline_start"]):
                return "mature"
            if ts < boundaries["decline_start"]:
                return "mature"
            if pd.notna(boundaries["expired_at"]) and ts >= boundaries["expired_at"]:
                return "tail"
            return "decline"

        subset["lifecycle_phase"] = subset["created_utc"].map(phase_for_time)
        annotated_parts.append(subset)
    phase_df = pd.DataFrame(phase_rows)
    annotated = pd.concat(annotated_parts, ignore_index=True) if annotated_parts else pd.DataFrame()
    return phase_df, annotated


def save_phase_metric_plot(
    phase_summary: pd.DataFrame,
    value_col: str,
    outpath: Path,
    title: str,
    ylabel: str,
) -> None:
    if phase_summary.empty:
        return
    plot_df = phase_summary.copy()
    plot_df["lifecycle_phase"] = pd.Categorical(plot_df["lifecycle_phase"], categories=PHASE_ORDER, ordered=True)
    plot_df = plot_df.sort_values(["template_rank", "lifecycle_phase"])
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    for template_name, group in plot_df.groupby("template_final", observed=True):
        ax.plot(
            group["lifecycle_phase"].astype(str),
            group[value_col],
            marker="o",
            linewidth=1.8,
            label=f"{int(group['template_rank'].iloc[0])}. {template_name}",
        )
    ax.set_title(title)
    ax.set_xlabel("Lifecycle Phase")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def compute_topic_outputs_bertopic(
    df: pd.DataFrame,
    n_topics: int,
    topic_max_features: int,
    topic_min_df: int,
    max_docs_fit: int,
    random_seed: int,
    topic_embedding_model: str,
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        from bertopic import BERTopic  # type: ignore
        from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance  # type: ignore
        from hdbscan import HDBSCAN  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore
        from umap import UMAP  # type: ignore
    except Exception as exc:
        raise RuntimeError("BERTopic stack is unavailable. Install bertopic, umap-learn, and hdbscan.") from exc

    working = df.copy()
    working["global_text"] = working["global_text"].fillna("").astype(str)
    working["local_text"] = working["local_text"].fillna("").astype(str)
    fit_docs = pd.concat([working["global_text"], working["local_text"]], ignore_index=True)
    fit_docs = fit_docs[fit_docs.str.strip().astype(bool)]
    if fit_docs.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if len(fit_docs) > max_docs_fit:
        fit_docs = fit_docs.sample(max_docs_fit, random_state=random_seed).reset_index(drop=True)

    embedding_model = SentenceTransformer(topic_embedding_model)
    umap_model = UMAP(n_neighbors=20, n_components=5, min_dist=0.0, metric="cosine", random_state=random_seed)
    hdbscan_model = HDBSCAN(
        min_cluster_size=max(15, min(30, max(10, len(fit_docs) // 400))),
        min_samples=10,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    vectorizer_model = CountVectorizer(stop_words="english", max_features=topic_max_features, min_df=topic_min_df)
    representation_model = [KeyBERTInspired(), MaximalMarginalRelevance(diversity=0.5)]
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        nr_topics=max(2, n_topics),
        representation_model=representation_model,
        verbose=False,
    )

    fit_embeddings = embedding_model.encode(
        fit_docs.tolist(),
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    topic_model.fit(fit_docs.tolist(), fit_embeddings)
    working["global_topic_id"] = -1
    working["local_topic_id"] = -1
    working["global_topic_weight"] = np.nan
    working["local_topic_weight"] = np.nan

    for text_column, topic_column, weight_column in [
        ("global_text", "global_topic_id", "global_topic_weight"),
        ("local_text", "local_topic_id", "local_topic_weight"),
    ]:
        valid_mask = working[text_column].str.strip().ne("")
        if not valid_mask.any():
            continue
        docs = working.loc[valid_mask, text_column].tolist()
        embeddings = embedding_model.encode(
            docs,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        topics, probs = topic_model.transform(docs, embeddings=embeddings)
        working.loc[valid_mask, topic_column] = np.asarray(topics, dtype=int)
        working.loc[valid_mask, weight_column] = [
            float(np.nanmax(prob)) if hasattr(prob, "__len__") and len(prob) else float("nan") for prob in probs
        ]

    topic_info = topic_model.get_topic_info()
    topic_terms = topic_info.loc[:, ["Topic", "Name"]].rename(columns={"Topic": "topic_id", "Name": "top_terms"})
    topic_terms["topic_id"] = topic_terms["topic_id"].astype(int)
    global_summary = (
        working["global_topic_id"].value_counts(normalize=True)
        .rename_axis("topic_id")
        .reset_index(name="share_global")
        .merge(topic_terms, on="topic_id", how="left")
        .sort_values("share_global", ascending=False)
    )
    local_summary = (
        working["local_topic_id"].value_counts(normalize=True)
        .rename_axis("topic_id")
        .reset_index(name="share_local")
        .merge(topic_terms, on="topic_id", how="left")
        .sort_values("share_local", ascending=False)
    )
    transition = (
        working.groupby(["global_topic_id", "local_topic_id"], observed=True)
        .size()
        .rename("count")
        .reset_index()
    )
    return working, topic_terms, global_summary, local_summary.merge(
        transition, how="outer", left_on="topic_id", right_on="local_topic_id"
    )


def compute_topic_outputs(
    df: pd.DataFrame,
    n_topics: int,
    topic_max_features: int,
    topic_min_df: int,
    max_docs_fit: int,
    random_seed: int,
    topic_model: str,
    topic_embedding_model: str,
    bertopic_batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if topic_model == "bertopic":
        try:
            return compute_topic_outputs_bertopic(
                df,
                n_topics=n_topics,
                topic_max_features=topic_max_features,
                topic_min_df=topic_min_df,
                max_docs_fit=max_docs_fit,
                random_seed=random_seed,
                topic_embedding_model=topic_embedding_model,
                batch_size=bertopic_batch_size,
            )
        except Exception:
            pass
    working = df.copy()
    working["global_text"] = working["global_text"].fillna("").astype(str)
    working["local_text"] = working["local_text"].fillna("").astype(str)
    fit_docs = pd.concat([working["global_text"], working["local_text"]], ignore_index=True)
    fit_docs = fit_docs[fit_docs.str.strip().astype(bool)]
    if fit_docs.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if len(fit_docs) > max_docs_fit:
        fit_docs = fit_docs.sample(max_docs_fit, random_state=random_seed).reset_index(drop=True)

    vectorizer = TfidfVectorizer(stop_words="english", max_features=topic_max_features, min_df=topic_min_df)
    fit_matrix = vectorizer.fit_transform(fit_docs.tolist())
    effective_topics = max(2, min(n_topics, fit_matrix.shape[0] - 1, fit_matrix.shape[1] - 1))
    nmf = NMF(n_components=effective_topics, random_state=random_seed, init="nndsvda", max_iter=400)
    nmf.fit(fit_matrix)

    global_matrix = vectorizer.transform(working["global_text"].tolist())
    local_matrix = vectorizer.transform(working["local_text"].tolist())
    global_topics = nmf.transform(global_matrix)
    local_topics = nmf.transform(local_matrix)
    working["global_topic_id"] = np.asarray(global_topics.argmax(axis=1), dtype=int)
    working["local_topic_id"] = np.asarray(local_topics.argmax(axis=1), dtype=int)
    working["global_topic_weight"] = global_topics.max(axis=1)
    working["local_topic_weight"] = local_topics.max(axis=1)

    feature_names = vectorizer.get_feature_names_out()
    topic_rows: list[dict[str, Any]] = []
    for topic_idx, weights in enumerate(nmf.components_):
        top_idx = np.argsort(weights)[::-1][:12]
        topic_rows.append(
            {
                "topic_id": int(topic_idx),
                "top_terms": " | ".join(feature_names[top_idx].tolist()),
            }
        )
    topic_terms = pd.DataFrame(topic_rows)

    global_summary = (
        working["global_topic_id"].value_counts(normalize=True)
        .rename_axis("topic_id")
        .reset_index(name="share_global")
        .merge(topic_terms, on="topic_id", how="left")
        .sort_values("share_global", ascending=False)
    )
    local_summary = (
        working["local_topic_id"].value_counts(normalize=True)
        .rename_axis("topic_id")
        .reset_index(name="share_local")
        .merge(topic_terms, on="topic_id", how="left")
        .sort_values("share_local", ascending=False)
    )
    transition = (
        working.groupby(["global_topic_id", "local_topic_id"], observed=True)
        .size()
        .rename("count")
        .reset_index()
    )
    return working, topic_terms, global_summary, local_summary.merge(transition, how="outer", left_on="topic_id", right_on="local_topic_id")


def compute_topic_transition_matrix(df: pd.DataFrame) -> pd.DataFrame:
    if "global_topic_id" not in df.columns or "local_topic_id" not in df.columns:
        return pd.DataFrame()
    matrix = (
        df.groupby(["global_topic_id", "local_topic_id"], observed=True)
        .size()
        .rename("count")
        .reset_index()
        .pivot(index="global_topic_id", columns="local_topic_id", values="count")
        .fillna(0)
    )
    row_totals = matrix.sum(axis=1).replace(0, np.nan)
    return matrix.div(row_totals, axis=0).fillna(0.0)


def save_topic_bar(summary_df: pd.DataFrame, share_col: str, outpath: Path, title: str) -> None:
    if summary_df.empty:
        return
    plot_df = summary_df.head(10).copy()
    labels = [f"T{int(topic_id)}" for topic_id in plot_df["topic_id"]]
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(labels, plot_df[share_col], color="#6c8ebf")
    ax.set_title(title)
    ax.set_xlabel("Topic")
    ax.set_ylabel("Share")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def top_terms_by_group(
    texts: pd.Series,
    groups: pd.Series,
    top_n: int,
    max_features: int,
    min_df: int,
) -> pd.DataFrame:
    valid = texts.astype(str).str.strip().ne("") & groups.astype(str).str.strip().ne("")
    if not valid.any():
        return pd.DataFrame()
    texts = texts.loc[valid].astype(str)
    groups = groups.loc[valid].astype(str)
    vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features, min_df=min_df)
    try:
        matrix = vectorizer.fit_transform(texts.tolist())
    except ValueError:
        fallback_vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features, min_df=1)
        try:
            matrix = fallback_vectorizer.fit_transform(texts.tolist())
            vectorizer = fallback_vectorizer
        except ValueError:
            return pd.DataFrame()
    feature_names = vectorizer.get_feature_names_out()
    rows: list[dict[str, Any]] = []
    for group_name in sorted(groups.unique().tolist()):
        mask = (groups == group_name).to_numpy(dtype=bool)
        group_mean = np.asarray(matrix[mask].mean(axis=0)).ravel()
        top_idx = np.argsort(group_mean)[::-1][:top_n]
        for rank, idx in enumerate(top_idx, start=1):
            rows.append(
                {
                    "group": group_name,
                    "rank": int(rank),
                    "term": str(feature_names[idx]),
                    "score": float(group_mean[idx]),
                }
            )
    return pd.DataFrame(rows)


def save_group_term_plot(term_df: pd.DataFrame, outpath: Path, title: str) -> None:
    if term_df.empty:
        return
    groups = term_df["group"].unique().tolist()[:4]
    configure_plot_style()
    fig, axes = plt.subplots(len(groups), 1, figsize=(11, 3.2 * len(groups)))
    if len(groups) == 1:
        axes = [axes]
    for ax, group_name in zip(axes, groups):
        group = term_df[term_df["group"] == group_name].sort_values("score", ascending=True)
        ax.barh(group["term"], group["score"], color="#f4a261")
        ax.set_title(f"{group_name}")
        ax.set_xlabel("Mean TF-IDF")
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def save_complexity_vs_score(df: pd.DataFrame, outpath: Path) -> None:
    plot_df = df.dropna(subset=["local_complexity_index", "score"]).copy()
    if plot_df.empty:
        return
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(plot_df["local_complexity_index"], np.log1p(plot_df["score"]), s=8, alpha=0.18, color="#2a9d8f")
    ax.set_title("Local Complexity vs Log Upvotes")
    ax.set_xlabel("Local Complexity Index")
    ax.set_ylabel("log(1 + score)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def save_alignment_vs_score(df: pd.DataFrame, outpath: Path) -> None:
    plot_df = df.dropna(subset=["global_local_similarity", "score"]).copy()
    if plot_df.empty:
        return
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(plot_df["global_local_similarity"], np.log1p(plot_df["score"]), s=8, alpha=0.18, color="#8d99ae")
    ax.set_title("Global-Local Alignment vs Log Upvotes")
    ax.set_xlabel("Global-Local Similarity")
    ax.set_ylabel("log(1 + score)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def run_analysis(args: argparse.Namespace) -> Path:
    outdir = ensure_dir(args.results_dir)
    df = load_analysis_dataframe(args.analysis_parquet)
    if args.max_rows > 0:
        df = df.head(args.max_rows).copy()
    df = add_affect_layers(df, args)
    df = add_local_complexity_features(df)
    df = compute_alignment(
        df,
        backend=args.alignment_backend,
        sbert_model=args.sbert_model,
        max_features=args.alignment_max_features,
    )
    top_templates = select_top_templates(
        df,
        ranking=args.template_ranking,
        top_k=args.top_k_templates,
        min_posts=args.min_template_posts,
    )

    pair_matrix, pair_norm = compute_paired_sentiment_outputs(df)
    emotion_pair_matrix, emotion_pair_norm = compute_paired_emotion_outputs(df)
    pair_matrix.to_csv(outdir / "paired_sentiment_counts.csv")
    pair_norm.to_csv(outdir / "paired_sentiment_row_normalized.csv")
    emotion_pair_matrix.to_csv(outdir / "paired_emotion_counts.csv")
    emotion_pair_norm.to_csv(outdir / "paired_emotion_row_normalized.csv")
    save_heatmap(pair_matrix, outdir / "fig_paired_sentiment_counts.png", "Global vs Local Sentiment Counts", cmap="Purples")
    save_heatmap(pair_norm, outdir / "fig_paired_sentiment_normalized.png", "Global vs Local Sentiment Transition Rates", cmap="Blues")
    save_heatmap(
        emotion_pair_matrix,
        outdir / "fig_paired_emotion_counts.png",
        "Global vs Local Dominant Emotion Counts",
        cmap="Oranges",
        annotate=False,
    )
    save_heatmap(
        emotion_pair_norm,
        outdir / "fig_paired_emotion_normalized.png",
        "Global vs Local Dominant Emotion Transition Rates",
        cmap="YlOrBr",
        annotate=False,
    )
    save_alignment_distribution(df, outdir / "fig_alignment_distribution.png")
    save_sentiment_score_hexbin(df, outdir / "fig_sentiment_score_hexbin.png")

    global_dominant = compute_dominant_affect_counts(df, "global_dominant_emotion")
    local_dominant = compute_dominant_affect_counts(df, "local_dominant_emotion")
    global_threshold = compute_threshold_proportions(df, "global", args.emotion_threshold)
    local_threshold = compute_threshold_proportions(df, "local", args.emotion_threshold)
    global_monthly_affect = compute_monthly_affect_trends(df, "global", args.emotion_threshold, args.emotion_top_n)
    local_monthly_affect = compute_monthly_affect_trends(df, "local", args.emotion_threshold, args.emotion_top_n)
    global_dominant.to_csv(outdir / "global_dominant_emotion_counts.csv", index=False)
    local_dominant.to_csv(outdir / "local_dominant_emotion_counts.csv", index=False)
    global_threshold.to_csv(outdir / "global_threshold_emotion_proportions.csv", index=False)
    local_threshold.to_csv(outdir / "local_threshold_emotion_proportions.csv", index=False)
    global_monthly_affect.to_csv(outdir / "global_monthly_affect_trends.csv", index=False)
    local_monthly_affect.to_csv(outdir / "local_monthly_affect_trends.csv", index=False)
    save_count_bar_chart(
        global_dominant,
        outdir / "fig_global_dominant_emotions.png",
        "Global Context: Dominant Emotion Distribution",
        "Posts",
        "#4c78a8",
    )
    save_count_bar_chart(
        local_dominant,
        outdir / "fig_local_dominant_emotions.png",
        "Local Context: Dominant Emotion Distribution",
        "Posts",
        "#f58518",
    )
    save_share_bar_chart(
        global_threshold,
        outdir / "fig_global_threshold_emotions.png",
        f"Global Context: Share Above Emotion Threshold ({args.emotion_threshold:.2f})",
        "Share of Posts",
        "#54a24b",
    )
    save_share_bar_chart(
        local_threshold,
        outdir / "fig_local_threshold_emotions.png",
        f"Local Context: Share Above Emotion Threshold ({args.emotion_threshold:.2f})",
        "Share of Posts",
        "#e45756",
    )
    save_monthly_affect_plot(
        global_monthly_affect,
        "global",
        outdir / "fig_global_monthly_affect_trends.png",
        "Global Context: Monthly Normalized Emotion Trends",
    )
    save_monthly_affect_plot(
        local_monthly_affect,
        "local",
        outdir / "fig_local_monthly_affect_trends.png",
        "Local Context: Monthly Normalized Emotion Trends",
    )

    monthly_metrics = compute_monthly_metrics(df, args.monthly_min_posts, args.rolling_months)
    monthly_metrics.to_csv(outdir / "monthly_semantic_metrics.csv", index=False)
    save_monthly_line_plot(
        monthly_metrics,
        "mean_alignment_rolling",
        outdir / "fig_monthly_alignment.png",
        "Monthly Global-Local Alignment",
        "Rolling Mean Similarity",
    )
    save_monthly_line_plot(
        monthly_metrics,
        "mismatch_rate_rolling",
        outdir / "fig_monthly_sentiment_mismatch.png",
        "Monthly Global-Local Emotion Mismatch",
        "Rolling Dominant-Emotion Mismatch Rate",
    )
    save_monthly_line_plot(
        monthly_metrics,
        "mean_local_complexity_rolling",
        outdir / "fig_monthly_local_complexity.png",
        "Monthly Local Complexity",
        "Rolling Mean Complexity",
    )
    save_monthly_line_plot(
        monthly_metrics,
        "mean_score_rolling",
        outdir / "fig_monthly_mean_score.png",
        "Monthly Mean Meme Score",
        "Rolling Mean Score",
    )

    template_summary = compute_template_conditioned_summary(df, top_templates)
    template_summary.to_csv(outdir / "top_template_semantic_summary.csv", index=False)
    top_templates.to_csv(outdir / "top_templates_selected.csv", index=False)
    save_template_bar(
        template_summary,
        "mean_alignment",
        outdir / "fig_top_templates_alignment.png",
        "Top Templates: Mean Global-Local Alignment",
        "Mean Similarity",
        "#457b9d",
    )
    save_template_bar(
        template_summary,
        "mismatch_rate",
        outdir / "fig_top_templates_sentiment_mismatch.png",
        "Top Templates: Dominant Emotion Mismatch Rate",
        "Dominant Emotion Mismatch Rate",
        "#e76f51",
    )
    save_template_bar(
        template_summary,
        "mean_local_complexity",
        outdir / "fig_top_templates_local_complexity.png",
        "Top Templates: Local Complexity",
        "Mean Complexity Index",
        "#2a9d8f",
    )

    phase_boundaries, phase_annotated = assign_lifecycle_phases(
        df,
        top_templates=top_templates,
        freq=args.lifecycle_freq,
        low_frac=args.low_frac,
        sustain_periods=args.sustain_periods,
        zero_run_periods=args.zero_run_periods,
    )
    phase_boundaries.to_csv(outdir / "top_template_lifecycle_boundaries.csv", index=False)
    if not phase_annotated.empty:
        mismatch_column = "emotion_mismatch" if "emotion_mismatch" in phase_annotated.columns else "sentiment_mismatch"
        phase_summary = (
            phase_annotated.groupby(["template_final", "template_rank", "lifecycle_phase"], observed=True)
            .agg(
                posts=("template_final", "size"),
                mean_alignment=("global_local_similarity", "mean"),
                mismatch_rate=(mismatch_column, "mean"),
                mean_local_complexity=("local_complexity_index", "mean"),
                mean_score=("score", "mean"),
            )
            .reset_index()
        )
    else:
        phase_summary = pd.DataFrame()
    phase_summary.to_csv(outdir / "top_template_phase_summary.csv", index=False)
    save_phase_metric_plot(
        phase_summary,
        "mean_alignment",
        outdir / "fig_phase_alignment.png",
        "Alignment Across Lifecycle Phases for Top Templates",
        "Mean Similarity",
    )
    save_phase_metric_plot(
        phase_summary,
        "mismatch_rate",
        outdir / "fig_phase_sentiment_mismatch.png",
        "Dominant Emotion Mismatch Across Lifecycle Phases for Top Templates",
        "Dominant Emotion Mismatch Rate",
    )
    save_phase_metric_plot(
        phase_summary,
        "mean_local_complexity",
        outdir / "fig_phase_local_complexity.png",
        "Local Complexity Across Lifecycle Phases for Top Templates",
        "Mean Complexity Index",
    )

    topic_df, topic_terms, global_topic_summary, local_topic_plus = compute_topic_outputs(
        df,
        n_topics=args.n_topics,
        topic_max_features=args.topic_max_features,
        topic_min_df=args.topic_min_df,
        max_docs_fit=args.max_topic_docs_fit,
        random_seed=args.random_seed,
        topic_model=args.topic_model,
        topic_embedding_model=args.topic_embedding_model,
        bertopic_batch_size=args.bertopic_batch_size,
    )
    if not topic_df.empty:
        df = topic_df
    topic_terms.to_csv(outdir / "shared_topics_terms.csv", index=False)
    global_topic_summary.to_csv(outdir / "global_topic_summary.csv", index=False)
    local_topic_plus.to_csv(outdir / "local_topic_summary_and_transitions_long.csv", index=False)
    transition_matrix = compute_topic_transition_matrix(df)
    transition_matrix.to_csv(outdir / "global_to_local_topic_transition.csv")
    save_heatmap(
        transition_matrix,
        outdir / "fig_global_to_local_topic_transition.png",
        "Global Topic to Local Topic Transition Rates",
        cmap="YlGnBu",
        annotate=False,
    )
    save_topic_bar(
        global_topic_summary,
        "share_global",
        outdir / "fig_global_topic_prevalence.png",
        "Top Shared-Space Topics in Global Context",
    )
    if not local_topic_plus.empty:
        local_topic_summary = (
            local_topic_plus.loc[:, ["topic_id", "share_local", "top_terms"]]
            .drop_duplicates(subset=["topic_id"])
            .sort_values("share_local", ascending=False)
        )
    else:
        local_topic_summary = pd.DataFrame()
    save_topic_bar(
        local_topic_summary,
        "share_local",
        outdir / "fig_local_topic_prevalence.png",
        "Top Shared-Space Topics in Local Context",
    )

    if not phase_annotated.empty:
        global_phase_terms = top_terms_by_group(
            phase_annotated["global_text"],
            phase_annotated["lifecycle_phase"],
            top_n=args.keyword_top_n,
            max_features=args.topic_max_features,
            min_df=max(3, min(args.topic_min_df, 5)),
        )
        local_phase_terms = top_terms_by_group(
            phase_annotated["local_text"],
            phase_annotated["lifecycle_phase"],
            top_n=args.keyword_top_n,
            max_features=args.topic_max_features,
            min_df=max(3, min(args.topic_min_df, 5)),
        )
    else:
        global_phase_terms = pd.DataFrame()
        local_phase_terms = pd.DataFrame()
    global_phase_terms.to_csv(outdir / "global_phase_keywords.csv", index=False)
    local_phase_terms.to_csv(outdir / "local_phase_keywords.csv", index=False)
    save_group_term_plot(
        global_phase_terms,
        outdir / "fig_global_phase_keywords.png",
        "Top Global Terms by Lifecycle Phase",
    )
    save_group_term_plot(
        local_phase_terms,
        outdir / "fig_local_phase_keywords.png",
        "Top Local Terms by Lifecycle Phase",
    )

    save_complexity_vs_score(df, outdir / "fig_local_complexity_vs_score.png")
    save_alignment_vs_score(df, outdir / "fig_alignment_vs_score.png")

    if not top_templates.empty:
        top_set = set(top_templates["template_final"])
        top_monthly = (
            df[df["template_final"].isin(top_set)]
            .groupby(["year_month", "template_final"], observed=True)
            .agg(
                posts=("template_final", "size"),
                mean_alignment=("global_local_similarity", "mean"),
                mean_local_complexity=("local_complexity_index", "mean"),
            )
            .reset_index()
        )
        top_monthly["month_start"] = pd.PeriodIndex(top_monthly["year_month"], freq="M").to_timestamp()
        top_monthly = top_monthly.merge(
            top_templates.loc[:, ["template_final", "template_rank"]],
            on="template_final",
            how="left",
        )
        top_monthly.to_csv(outdir / "top_template_monthly_metrics.csv", index=False)

        configure_plot_style()
        fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
        for template_name, group in top_monthly.groupby("template_final", observed=True):
            rank_value = int(group["template_rank"].iloc[0])
            label = f"{rank_value}. {template_name}"
            axes[0].plot(group["month_start"], group["mean_alignment"], linewidth=1.5, label=label)
            axes[1].plot(group["month_start"], group["mean_local_complexity"], linewidth=1.5, label=label)
        axes[0].set_title("Top Templates: Monthly Global-Local Alignment")
        axes[0].set_ylabel("Mean Similarity")
        axes[1].set_title("Top Templates: Monthly Local Complexity")
        axes[1].set_ylabel("Mean Complexity")
        axes[1].set_xlabel("Month")
        axes[0].legend(loc="upper left", fontsize=8, ncol=2)
        axes[0].grid(alpha=0.25)
        axes[1].grid(alpha=0.25)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(outdir / "fig_top_template_monthly_semantic_metrics.png", dpi=180)
        plt.close(fig)

    pair_melt = pd.DataFrame()
    if not pair_matrix.empty:
        pair_melt = pair_matrix.stack().reset_index().rename(columns={0: "count"})
        pair_melt = pair_melt.sort_values("count", ascending=False).reset_index(drop=True)

    bullets = [
        f"Rows analyzed: {len(df)}.",
        f"Top templates analyzed in detail: {len(top_templates)} with ranking='{args.template_ranking}' and min_posts={args.min_template_posts}.",
        (
            f"Emotion mismatch rate: {df['emotion_mismatch'].mean():.3f}; "
            f"sentiment mismatch rate: {df['sentiment_mismatch'].mean():.3f}; "
            f"mean global-local similarity: {df['global_local_similarity'].dropna().mean():.3f}."
            if df["global_local_similarity"].notna().any()
            else "No valid global-local similarity scores were computed."
        ),
        (
            f"Most common paired sentiment transition: "
            f"{pair_melt.iloc[0, 0]} -> {pair_melt.iloc[0, 1]} ({int(pair_melt.iloc[0, 2])} posts)."
            if not pair_melt.empty
            else "No paired sentiment matrix available."
        ),
        (
            f"Most common dominant emotion in global context: {global_dominant.iloc[0]['label']} "
            f"({int(global_dominant.iloc[0]['count'])} posts); "
            f"in local context: {local_dominant.iloc[0]['label']} ({int(local_dominant.iloc[0]['count'])} posts)."
            if not global_dominant.empty and not local_dominant.empty
            else "Dominant emotion distributions were unavailable."
        ),
        (
            f"Lifecycle-phase rows available for top templates: {len(phase_annotated)}."
            if not phase_annotated.empty
            else "No lifecycle-phase annotations were generated."
        ),
        (
            f"Shared topic model generated {len(topic_terms)} topics; "
            f"largest global topic share={global_topic_summary['share_global'].max():.3f}."
            if not topic_terms.empty and not global_topic_summary.empty
            else "Topic modeling outputs were unavailable."
        ),
    ]

    write_summary_markdown(outdir / "summary.md", "Q2 Global vs Local Context", bullets)
    write_run_metadata(
        outdir / "run_metadata.json",
        {
            "analysis_parquet": str(Path(args.analysis_parquet).expanduser().resolve()),
            "results_dir": str(outdir),
            "rows": int(len(df)),
            "top_k_templates": int(args.top_k_templates),
            "min_template_posts": int(args.min_template_posts),
            "template_ranking": str(args.template_ranking),
            "alignment_backend": str(df["alignment_backend"].iloc[0]) if len(df) else "none",
            "emotion_backend": str(args.emotion_backend),
            "n_topics": int(args.n_topics),
            "topic_model": str(args.topic_model),
            "random_seed": int(args.random_seed),
        },
    )
    print(f"results_dir={outdir}")
    return outdir


def main() -> None:
    run_analysis(parse_args())


if __name__ == "__main__":
    main()
