#!/usr/bin/env python3
from __future__ import annotations

import argparse
import colorsys
import hashlib
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from common import (
    add_sentiment_layers,
    build_current_data_bundle_raw,
    build_text,
    configure_plot_style,
    ensure_dir,
    try_import_sentence_transformers,
    write_run_metadata,
    write_summary_markdown,
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
    parser.add_argument("--analysis-parquet", required=True)
    parser.add_argument("--results-dir", default="analysis/results/q2_global_local_context")
    parser.add_argument("--emotion-backend", choices=["cardiff", "vader"], default="cardiff")
    parser.add_argument("--emotion-cache-dir", default="analysis/cache/q2_cardiff_affect")
    parser.add_argument("--emotion-batch-size", type=int, default=32)
    parser.add_argument("--emotion-max-length", type=int, default=512)
    parser.add_argument("--emotion-threshold", type=float, default=0.7)
    parser.add_argument("--hist-bins", type=int, default=28)
    parser.add_argument("--dominant-top-n", type=int, default=3)
    parser.add_argument("--wordcloud-max-words", type=int, default=150)
    parser.add_argument("--wordcloud-min-token-len", type=int, default=3)
    parser.add_argument("--keyword-backend", choices=["auto", "sbert", "tfidf"], default="auto")
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


def _theme_plot(figsize: tuple[float, float]) -> tuple[plt.Figure, plt.Axes]:
    configure_plot_style()
    plt.rcParams.update(
        {
            "axes.facecolor": "#f7f9fc",
            "figure.facecolor": "#ffffff",
            "grid.color": "#d5deed",
            "grid.alpha": 0.45,
        }
    )
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


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
    backend: str,
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

    ranked: dict[str, dict[str, float]] = {}
    used_backend = "none"
    if backend in {"auto", "sbert"}:
        ranked, used_backend = _rank_words_sbert(
            texts_by_label=texts_by_label,
            freqs_by_label=freqs_by_label,
            model_name=embedding_model,
            max_docs_per_class=max_docs_per_class,
            random_seed=random_seed,
        )
        if backend == "sbert" and used_backend == "none":
            raise RuntimeError("sentence-transformers is unavailable for --keyword-backend=sbert")

    if used_backend == "none":
        ranked, used_backend = _rank_words_tfidf(
            texts_by_label=texts_by_label,
            freqs_by_label=freqs_by_label,
        )

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

    configure_plot_style()
    plt.rcParams.update(
        {
            "axes.facecolor": "#f7f9fc",
            "figure.facecolor": "#ffffff",
            "grid.color": "#d5deed",
            "grid.alpha": 0.45,
        }
    )
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
        ax.set_title(f"{label}_{suffix}", fontsize=16)
        ax.set_xlim(0, 1)
        ax.grid(True, axis="y")

    for ax in axes_list[len(ALL_AFFECT_LABELS) :]:
        ax.axis("off")

    fig.suptitle(
        f"Histogram of Affect Scores ({suffix.title()} Context)",
        fontsize=28,
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
    ax.tick_params(axis="x", rotation=40)
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
            fontsize=8,
        )
    ax.set_title(
        f"Mean Dominant Affect Score by Class in {context.title()} Context (Score > {threshold:.1f})"
    )
    ax.set_xlabel("Affect Class")
    ax.set_ylabel("Mean Dominant Score")
    ax.set_ylim(threshold, 1.03)
    ax.tick_params(axis="x", rotation=40)
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
    keyword_backend: str,
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
        backend=keyword_backend,
        embedding_model=keyword_embedding_model,
        max_docs_per_class=keyword_max_docs_per_class,
        random_seed=random_seed,
    )
    keyword_df.to_csv(outdir / f"{context_name}_wordcloud_keyword_weights.csv", index=False)
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

        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.imshow(image, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(
            f"Word Cloud for {label.title()} Memes ({context.title()} Context, Score > {threshold:.1f})",
            fontsize=16,
        )
        fig.tight_layout()
        fig.savefig(context_dir / f"{label}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved += 1

    if not cloud_arrays:
        return 0, 0

    ncols = 3
    nrows = math.ceil(len(cloud_arrays) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5.2 * nrows))
    axes_list = np.atleast_1d(axes).ravel().tolist()
    for ax, (label, image) in zip(axes_list, cloud_arrays):
        ax.imshow(image, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(label, fontsize=14)
    for ax in axes_list[len(cloud_arrays) :]:
        ax.axis("off")
    fig.suptitle(
        f"Word Clouds by Dominant Affect Class ({context.title()} Context)",
        fontsize=24,
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(outdir / f"fig_wordcloud_grid_{context_name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return saved, len(cloud_arrays), used_backend


def run_analysis(args: argparse.Namespace) -> Path:
    outdir = ensure_dir(args.results_dir)
    data_dir = Path(args.analysis_parquet).expanduser().resolve().parent
    df, bundle_metadata = build_current_data_bundle_raw(data_dir)
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
    metadata: dict[str, Any] = {
        "analysis_parquet": str(Path(args.analysis_parquet).expanduser().resolve()),
        "data_source": "current_data_bundle",
        "results_dir": str(outdir),
        "rows_analyzed": int(len(df)),
        "emotion_backend": args.emotion_backend,
        "emotion_threshold": float(args.emotion_threshold),
        "hist_bins": int(args.hist_bins),
        "dominant_top_n": int(args.dominant_top_n),
        "wordcloud_available": bool(WordCloud is not None),
        "wordcloud_max_words": int(args.wordcloud_max_words),
        "keyword_backend_requested": args.keyword_backend,
        "keyword_embedding_model": args.keyword_embedding_model,
        "keyword_max_docs_per_class": int(args.keyword_max_docs_per_class),
        **bundle_metadata,
    }

    for context in ("global", "local"):
        save_score_histogram_grid(
            df,
            context=context,
            outpath=outdir / f"fig_histogram_scores_{context}.png",
            bins=args.hist_bins,
        )

        dominant_df, summary_df = compute_dominant_summary(df, context=context, threshold=args.emotion_threshold)
        summary_df.to_csv(outdir / f"{context}_dominant_affect_summary.csv", index=False)
        dominant_df.to_csv(outdir / f"{context}_dominant_affect_rows.csv", index=False)

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
            keyword_backend=args.keyword_backend,
            keyword_embedding_model=args.keyword_embedding_model,
            keyword_max_docs_per_class=args.keyword_max_docs_per_class,
            random_seed=args.random_seed,
        )
        metadata[f"{context}_dominant_rows"] = int(len(dominant_df))
        metadata[f"{context}_dominant_labels"] = summary_df["label"].tolist()
        metadata[f"{context}_wordcloud_count"] = int(saved_clouds)
        metadata[f"{context}_wordcloud_grid_items"] = int(grid_clouds)
        metadata[f"{context}_keyword_backend_used"] = used_keyword_backend

    if "emotion_mismatch" in df.columns:
        metadata["emotion_mismatch_rate"] = float(pd.to_numeric(df["emotion_mismatch"], errors="coerce").mean())
    if "sentiment_mismatch" in df.columns:
        metadata["sentiment_mismatch_rate"] = float(pd.to_numeric(df["sentiment_mismatch"], errors="coerce").mean())

    summary_lines = [
        f"Rows analyzed: {len(df)}.",
        f"Affect backend: {args.emotion_backend}. Dominant-score threshold: {args.emotion_threshold:.2f}.",
        (
            f"Global dominant classes above threshold: {', '.join(metadata['global_dominant_labels'][:8])}."
            if metadata.get("global_dominant_labels")
            else "Global dominant classes above threshold: none."
        ),
        (
            f"Local dominant classes above threshold: {', '.join(metadata['local_dominant_labels'][:8])}."
            if metadata.get("local_dominant_labels")
            else "Local dominant classes above threshold: none."
        ),
        (
            f"Word clouds generated: global={metadata.get('global_wordcloud_count', 0)}, "
            f"local={metadata.get('local_wordcloud_count', 0)}."
            if WordCloud is not None
            else "Word cloud dependency was unavailable, so word-cloud charts were skipped."
        ),
        (
            f"Keyword ranking backend used: global={metadata.get('global_keyword_backend_used', 'none')}, "
            f"local={metadata.get('local_keyword_backend_used', 'none')}."
        ),
    ]
    if "emotion_mismatch_rate" in metadata:
        summary_lines.append(f"Global/local dominant-emotion mismatch rate: {metadata['emotion_mismatch_rate']:.3f}.")
    if "sentiment_mismatch_rate" in metadata:
        summary_lines.append(f"Global/local sentiment-label mismatch rate: {metadata['sentiment_mismatch_rate']:.3f}.")

    write_run_metadata(outdir / "run_metadata.json", metadata)
    write_summary_markdown(
        outdir / "summary.md",
        "Research Question 2: Global vs Local Affect Charts",
        summary_lines,
    )
    return outdir


def main() -> None:
    args = parse_args()
    outdir = run_analysis(args)
    print(outdir)


if __name__ == "__main__":
    main()
