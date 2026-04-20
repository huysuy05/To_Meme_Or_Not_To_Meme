#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from common import (
    attach_template_assignments_from_parquet,
    build_current_data_bundle_raw,
    configure_plot_style,
    ensure_dir,
    prepare_analysis_dataframe,
    write_run_metadata,
    write_summary_markdown,
)
from q1_template_popularity import prepare_template_dataframe
from q2_global_local_context import (
    LABEL_COLORS,
    NOTEBOOK_EMOTION_LABELS,
    NOTEBOOK_SENTIMENT_LABELS,
    add_affect_layers,
    rank_words_for_wordclouds,
)


DEFAULT_CLASS_ORDER = ["evergreen", "recurring", "faded", "emerging", "bursty"]
DEFAULT_CLUSTER_FEATURES = [
    "active_ratio",
    "entropy_norm",
    "peak_share",
    "top3_share",
    "pre_peak_share",
    "post_peak_share",
    "post_peak_active_ratio",
    "half_ratio",
    "peak_to_median_active_log",
    "active_run_density",
    "reactivation_ratio",
    "longest_zero_ratio",
    "peak_age_ratio",
    "peak_recency_ratio",
    "recent_share_3",
    "recent_growth_log",
]
CLASS_COLORS = {
    "evergreen": "#2a9d8f",
    "recurring": "#457b9d",
    "faded": "#8d99ae",
    "emerging": "#f4a261",
    "bursty": "#e76f51",
}
LIFECYCLE_SCORE_SPECS = {
    "evergreen": {
        "active_ratio": 1.0,
        "entropy_norm": 1.0,
        "post_peak_active_ratio": 0.95,
        "active_run_density": 0.45,
        "reactivation_ratio": 0.25,
        "peak_share": -0.85,
        "top3_share": -0.75,
        "peak_to_median_active_log": -0.55,
        "longest_zero_ratio": -0.7,
        "peak_age_ratio": -0.15,
    },
    "recurring": {
        "active_run_density": 1.0,
        "reactivation_ratio": 0.95,
        "entropy_norm": 0.6,
        "longest_zero_ratio": 0.45,
        "active_ratio": -0.2,
        "peak_share": -0.35,
        "top3_share": -0.3,
        "peak_to_median_active_log": -0.2,
    },
    "faded": {
        "peak_age_ratio": 1.0,
        "pre_peak_share": 0.6,
        "longest_zero_ratio": 0.8,
        "recent_share_3": -0.95,
        "recent_growth_log": -0.8,
        "post_peak_active_ratio": -0.65,
        "peak_recency_ratio": -0.7,
    },
    "emerging": {
        "recent_share_3": 1.0,
        "recent_growth_log": 1.0,
        "peak_recency_ratio": 0.95,
        "post_peak_share": 0.45,
        "pre_peak_share": -0.45,
        "peak_age_ratio": -0.85,
        "half_ratio": -0.25,
    },
    "bursty": {
        "peak_share": 1.0,
        "top3_share": 0.95,
        "peak_to_median_active_log": 0.95,
        "entropy_norm": -0.9,
        "active_ratio": -0.8,
        "post_peak_active_ratio": -0.65,
        "active_run_density": -0.55,
        "reactivation_ratio": -0.3,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "H1 analysis: classify meme templates by lifecycle shape into data-driven lifecycle clusters "
            "(for example evergreen, recurring, faded, emerging, and bursty), then compare class-level "
            "virality and content composition."
        )
    )
    parser.add_argument("--analysis-parquet", required=True)
    parser.add_argument("--results-dir", default="analysis/results/h1_analysis")
    parser.add_argument("--min-posts", type=int, default=20)
    parser.add_argument("--min-observed-days", type=int, default=180)
    parser.add_argument("--lifecycle-freq", default="M")
    parser.add_argument("--n-clusters", type=int, default=5)
    parser.add_argument(
        "--cluster-method",
        choices=["kmeans", "hierarchical", "gmm"],
        default="kmeans",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--top-k-representatives", type=int, default=6)
    parser.add_argument(
        "--global-topic-json",
        default="Meme_gemini/bertopic_models/meme_global_contexts_topics.json",
        help="Optional BERTopic JSON mapping meme key -> {topic, probability} for global context topics.",
    )
    parser.add_argument(
        "--local-topic-json",
        default="Meme_gemini/bertopic_models/meme_local_contexts_topics.json",
        help="Optional BERTopic JSON mapping meme key -> {topic, probability} for local context topics.",
    )
    parser.add_argument("--emotion-backend", choices=["cardiff", "vader", "none"], default="cardiff")
    parser.add_argument("--emotion-cache-dir", default="analysis/cache/h1_cardiff_affect")
    parser.add_argument("--emotion-batch-size", type=int, default=32)
    parser.add_argument("--emotion-max-length", type=int, default=512)
    parser.add_argument("--keyword-backend", choices=["auto", "sbert", "tfidf"], default="auto")
    parser.add_argument("--keyword-embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--keyword-max-docs-per-class", type=int, default=5000)
    parser.add_argument("--keyword-min-token-len", type=int, default=3)
    parser.add_argument("--keyword-max-per-class", type=int, default=25)
    parser.add_argument(
        "--topic-columns",
        nargs="*",
        default=[],
        help="Optional topic columns already present in the parquet to summarize by lifecycle class.",
    )
    return parser.parse_args()


def _normalize_pandas_freq(freq: str) -> str:
    upper = str(freq).upper()
    if upper == "M":
        return "MS"
    return freq


def _aligned_resample_endpoint(timestamp: pd.Timestamp, freq: str) -> pd.Timestamp:
    anchor = pd.Series([1], index=pd.DatetimeIndex([pd.Timestamp(timestamp)])).resample(freq).sum()
    return pd.Timestamp(anchor.index.max())


def _ordered_classes(values: pd.Series) -> list[str]:
    present = [name for name in DEFAULT_CLASS_ORDER if name in set(values.astype(str))]
    extras = sorted(set(values.astype(str)) - set(present))
    return present + extras


def _set_category_order(df: pd.DataFrame, column: str) -> pd.DataFrame:
    ordered = _ordered_classes(df[column])
    out = df.copy()
    out[column] = pd.Categorical(out[column], categories=ordered, ordered=True)
    return out.sort_values(column).reset_index(drop=True)


def _safe_divide(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return float(a) / float(b)


def build_template_lifecycle_features(
    df: pd.DataFrame,
    freq: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pandas_freq = _normalize_pandas_freq(freq)
    dataset_end = _aligned_resample_endpoint(pd.Timestamp(df["created_utc"].max()), pandas_freq)
    metric_rows: list[dict[str, Any]] = []
    curve_rows: list[pd.DataFrame] = []

    for template_key, group in df.groupby("template_key", observed=True):
        template_name = str(group["template_display"].iloc[0])
        first_seen = pd.Timestamp(group["created_utc"].min())
        last_seen = pd.Timestamp(group["created_utc"].max())
        total_posts = int(len(group))

        period_counts = (
            group.set_index("created_utc")
            .resample(pandas_freq)
            .size()
            .rename("count")
            .reset_index()
            .rename(columns={"created_utc": "period_start"})
        )
        full_range = pd.date_range(period_counts["period_start"].min(), dataset_end, freq=pandas_freq)
        period_counts = (
            period_counts.set_index("period_start")
            .reindex(full_range, fill_value=0)
            .rename_axis("period_start")
            .reset_index()
        )
        counts = period_counts["count"].to_numpy(dtype=float)
        active_mask = counts > 0
        active_counts = counts[active_mask]
        observed_periods = int(len(counts))
        active_periods = int(active_mask.sum())
        peak_idx = int(np.argmax(counts))
        peak_count = float(counts[peak_idx]) if len(counts) else 0.0
        peak_period = pd.Timestamp(period_counts.loc[peak_idx, "period_start"]) if len(period_counts) else pd.NaT
        sorted_counts = np.sort(counts)
        top3_share = _safe_divide(sorted_counts[-3:].sum(), total_posts)
        peak_share = _safe_divide(peak_count, total_posts)
        median_active = float(np.median(active_counts)) if len(active_counts) else 0.0
        peak_to_median_active = _safe_divide(peak_count, max(median_active, 1.0))
        peak_to_mean_active = _safe_divide(peak_count, max(float(active_counts.mean()) if len(active_counts) else 0.0, 1.0))

        probs = counts / max(total_posts, 1)
        nonzero_probs = probs[probs > 0]
        entropy = float(-(nonzero_probs * np.log(nonzero_probs)).sum()) if len(nonzero_probs) else 0.0
        entropy_norm = _safe_divide(entropy, math.log(observed_periods)) if observed_periods > 1 else 0.0
        hhi = float((probs**2).sum()) if len(probs) else 0.0

        longest_zero_run = 0
        current_zero_run = 0
        active_run_count = 0
        reactivation_count = 0
        in_active_run = False
        has_seen_active_run = False
        for value in counts:
            if value <= 0:
                current_zero_run += 1
                longest_zero_run = max(longest_zero_run, current_zero_run)
                in_active_run = False
            else:
                current_zero_run = 0
                if not in_active_run:
                    active_run_count += 1
                    if has_seen_active_run:
                        reactivation_count += 1
                    has_seen_active_run = True
                    in_active_run = True

        after_peak = counts[peak_idx + 1 :]
        post_peak_share = _safe_divide(after_peak.sum(), total_posts)
        post_peak_active_ratio = _safe_divide((after_peak > 0).sum(), len(after_peak))
        pre_peak_share = _safe_divide(counts[:peak_idx].sum(), total_posts)
        half_idx = int(np.argmax(np.cumsum(counts) >= total_posts * 0.5)) if total_posts > 0 else 0
        periods_to_half = half_idx + 1 if observed_periods > 0 else 0
        peak_age_ratio = _safe_divide(observed_periods - peak_idx - 1, max(observed_periods - 1, 1))
        peak_recency_ratio = 1.0 - peak_age_ratio

        recent_window = min(3, observed_periods)
        recent_share_3 = _safe_divide(counts[-recent_window:].sum(), total_posts) if recent_window > 0 else 0.0
        previous_share_3 = (
            _safe_divide(counts[-(2 * recent_window) : -recent_window].sum(), total_posts)
            if recent_window > 0 and observed_periods > recent_window
            else 0.0
        )
        recent_growth_log = math.log(
            _safe_divide((recent_share_3 * total_posts) + 1.0, (previous_share_3 * total_posts) + 1.0)
        )

        slope_window = min(6, observed_periods)
        recent_slope = 0.0
        if slope_window >= 2:
            slope_x = np.arange(slope_window, dtype=float)
            slope_y = counts[-slope_window:] / max(total_posts, 1.0)
            recent_slope = float(np.polyfit(slope_x, slope_y, deg=1)[0])

        period_counts["template_key"] = template_key
        period_counts["template_final"] = template_name
        period_counts["smooth"] = period_counts["count"].rolling(3, min_periods=1).mean()
        max_smooth = float(period_counts["smooth"].max()) if not period_counts.empty else 0.0
        period_counts["smooth_norm"] = period_counts["smooth"] / max(max_smooth, 1.0)
        curve_rows.append(period_counts)

        metric_rows.append(
            {
                "template_key": template_key,
                "template_final": template_name,
                "total_posts": total_posts,
                "first_seen": first_seen,
                "last_seen": last_seen,
                "observed_days_online": ((dataset_end - first_seen).total_seconds() / 86400.0) + 1.0,
                "active_days_in_dataset": ((last_seen - first_seen).total_seconds() / 86400.0) + 1.0,
                "observed_periods": observed_periods,
                "active_periods": active_periods,
                "active_ratio": _safe_divide(active_periods, observed_periods),
                "peak_period": peak_period,
                "peak_count": peak_count,
                "peak_share": peak_share,
                "top3_share": top3_share,
                "peak_to_median_active": peak_to_median_active,
                "peak_to_mean_active": peak_to_mean_active,
                "entropy_norm": entropy_norm,
                "hhi": hhi,
                "active_run_count": active_run_count,
                "active_run_density": _safe_divide(active_run_count, observed_periods),
                "reactivation_count": reactivation_count,
                "reactivation_ratio": _safe_divide(reactivation_count, max(active_run_count, 1)),
                "longest_zero_run": longest_zero_run,
                "longest_zero_ratio": _safe_divide(longest_zero_run, observed_periods),
                "post_peak_share": post_peak_share,
                "post_peak_active_ratio": post_peak_active_ratio,
                "pre_peak_share": pre_peak_share,
                "periods_to_half": periods_to_half,
                "half_ratio": _safe_divide(periods_to_half, observed_periods),
                "peak_age_ratio": peak_age_ratio,
                "peak_recency_ratio": peak_recency_ratio,
                "recent_share_3": recent_share_3,
                "previous_share_3": previous_share_3,
                "recent_growth_log": recent_growth_log,
                "recent_slope": recent_slope,
                "temporal_virality_index": peak_share + top3_share + math.log1p(peak_to_median_active),
            }
        )

    metrics = pd.DataFrame(metric_rows)
    curves = pd.concat(curve_rows, ignore_index=True) if curve_rows else pd.DataFrame()
    return metrics, curves


def _resolve_cluster_features(metrics: pd.DataFrame) -> list[str]:
    feature_cols = [col for col in DEFAULT_CLUSTER_FEATURES if col in metrics.columns]
    if "peak_to_median_active" in metrics.columns and "peak_to_median_active_log" not in metrics.columns:
        metrics["peak_to_median_active_log"] = np.log1p(
            pd.to_numeric(metrics["peak_to_median_active"], errors="coerce").fillna(0.0).clip(lower=0.0)
        )
        feature_cols = [col for col in DEFAULT_CLUSTER_FEATURES if col in metrics.columns]
    return feature_cols


def _fit_lifecycle_clusterer(
    X: np.ndarray,
    method: str,
    n_clusters: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    if method == "kmeans":
        model = KMeans(n_clusters=int(n_clusters), random_state=int(random_seed), n_init=20)
        labels = model.fit_predict(X)
        centers_z = np.asarray(model.cluster_centers_, dtype=float)
        return labels, centers_z
    if method == "hierarchical":
        model = AgglomerativeClustering(n_clusters=int(n_clusters))
        labels = model.fit_predict(X)
        return labels, None
    if method == "gmm":
        model = GaussianMixture(
            n_components=int(n_clusters),
            covariance_type="full",
            random_state=int(random_seed),
            n_init=5,
        )
        labels = model.fit_predict(X)
        centers_z = np.asarray(model.means_, dtype=float)
        return labels, centers_z
    raise ValueError(f"Unsupported cluster method: {method}")


def _cluster_centers_from_assignments(X: np.ndarray, labels: np.ndarray, n_clusters: int) -> np.ndarray:
    centers: list[np.ndarray] = []
    for cluster_id in range(int(n_clusters)):
        cluster_points = X[np.asarray(labels) == cluster_id]
        if len(cluster_points) == 0:
            centers.append(np.zeros(X.shape[1], dtype=float))
            continue
        centers.append(cluster_points.mean(axis=0))
    return np.vstack(centers)


def _build_cluster_profile(
    working: pd.DataFrame,
    feature_cols: list[str],
    scaler: StandardScaler,
    centers_z: np.ndarray,
) -> pd.DataFrame:
    center_z_df = pd.DataFrame(centers_z, columns=feature_cols)
    center_raw_df = pd.DataFrame(scaler.inverse_transform(centers_z), columns=feature_cols)
    profile = center_raw_df.copy()
    profile["lifecycle_cluster_id"] = range(len(profile))
    profile["templates_in_cluster"] = (
        working["lifecycle_cluster_id"].value_counts().sort_index().reindex(range(len(profile)), fill_value=0).to_numpy()
    )
    for label_name, weight_map in LIFECYCLE_SCORE_SPECS.items():
        profile[f"{label_name}_score"] = 0.0
        for column, weight in weight_map.items():
            if column in center_z_df.columns:
                profile[f"{label_name}_score"] += float(weight) * center_z_df[column]
    return profile


def _assign_cluster_names_from_scores(profile: pd.DataFrame) -> dict[int, str]:
    label_names = list(LIFECYCLE_SCORE_SPECS)
    cluster_ids = profile["lifecycle_cluster_id"].astype(int).tolist()
    used_labels = label_names[: min(len(label_names), len(cluster_ids))]
    score_matrix = np.asarray(
        [[float(profile.loc[profile["lifecycle_cluster_id"] == cluster_id, f"{label}_score"].iloc[0]) for label in used_labels] for cluster_id in cluster_ids],
        dtype=float,
    )
    row_idx, col_idx = linear_sum_assignment(-score_matrix)
    cluster_to_name: dict[int, str] = {}
    for row, col in zip(row_idx, col_idx):
        cluster_to_name[int(cluster_ids[row])] = str(used_labels[col])

    if len(cluster_ids) > len(used_labels):
        extras = [cluster_id for cluster_id in cluster_ids if cluster_id not in cluster_to_name]
        for cluster_id in extras:
            cluster_to_name[int(cluster_id)] = f"cluster_{int(cluster_id)}"
    return cluster_to_name


def classify_template_lifecycles(
    metrics: pd.DataFrame,
    n_clusters: int,
    random_seed: int,
    cluster_method: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    working = metrics.copy()
    n_clusters = max(1, min(int(n_clusters), len(working)))
    working["peak_to_median_active_log"] = np.log1p(
        pd.to_numeric(working["peak_to_median_active"], errors="coerce").fillna(0.0).clip(lower=0.0)
    )
    feature_cols = _resolve_cluster_features(working)
    if not feature_cols:
        raise RuntimeError("No lifecycle feature columns were available for clustering.")

    scaler = StandardScaler()
    X = scaler.fit_transform(working[feature_cols])
    labels, centers_z = _fit_lifecycle_clusterer(
        X,
        method=cluster_method,
        n_clusters=n_clusters,
        random_seed=random_seed,
    )
    working["lifecycle_cluster_id"] = np.asarray(labels, dtype=int)

    if centers_z is None:
        centers_z = _cluster_centers_from_assignments(X, labels=working["lifecycle_cluster_id"].to_numpy(), n_clusters=n_clusters)
    profile = _build_cluster_profile(working, feature_cols=feature_cols, scaler=scaler, centers_z=centers_z)
    cluster_to_name = _assign_cluster_names_from_scores(profile)

    working["lifecycle_class"] = working["lifecycle_cluster_id"].map(cluster_to_name).fillna("unassigned")
    profile["lifecycle_class"] = profile["lifecycle_cluster_id"].map(cluster_to_name).fillna("unassigned")
    working = _set_category_order(working, "lifecycle_class")
    profile = _set_category_order(profile, "lifecycle_class")
    return working, profile, feature_cols


def select_representative_templates(
    metrics: pd.DataFrame,
    feature_cols: list[str],
    top_k: int,
) -> pd.DataFrame:
    working = metrics.copy()
    scaler = StandardScaler()
    X = scaler.fit_transform(working[feature_cols])
    x_df = pd.DataFrame(X, columns=feature_cols, index=working.index)
    x_df["lifecycle_cluster_id"] = working["lifecycle_cluster_id"].to_numpy()
    centers = x_df.groupby("lifecycle_cluster_id", observed=True)[feature_cols].mean()

    distances: list[float] = []
    for idx, row in working.iterrows():
        center_values = centers.loc[int(row["lifecycle_cluster_id"]), feature_cols].to_numpy(dtype=float)
        point_values = x_df.loc[idx, feature_cols].to_numpy(dtype=float)
        distances.append(float(np.linalg.norm(point_values - center_values)))
    working["centroid_distance"] = distances
    working["distance_rank"] = (
        working.groupby("lifecycle_class", observed=True)["centroid_distance"]
        .rank(method="dense", ascending=True)
        .astype(float)
    )
    working["posts_rank"] = (
        working.groupby("lifecycle_class", observed=True)["total_posts"]
        .rank(method="dense", ascending=False)
        .astype(float)
    )
    working["representative_score"] = working["distance_rank"] + (0.65 * working["posts_rank"])

    representatives = (
        working.sort_values(
            ["lifecycle_class", "representative_score", "centroid_distance", "total_posts", "active_ratio"],
            ascending=[True, True, True, False, False],
        )
        .groupby("lifecycle_class", observed=True)
        .head(max(int(top_k), 1))
        .reset_index(drop=True)
    )
    return _set_category_order(representatives, "lifecycle_class")


def save_representative_lifecycle_plot(
    curve_df: pd.DataFrame,
    representatives: pd.DataFrame,
    lifecycle_class: str,
    outpath: Path,
    value_col: str,
    ylabel: str,
) -> None:
    subset = representatives[representatives["lifecycle_class"].astype(str) == lifecycle_class].copy()
    if subset.empty:
        return
    selected_keys = subset["template_key"].tolist()
    plot_df = curve_df[curve_df["template_key"].isin(selected_keys)].copy()
    if plot_df.empty:
        return

    rank_lookup = {key: idx + 1 for idx, key in enumerate(selected_keys)}
    name_lookup = subset.set_index("template_key")["template_final"].to_dict()
    posts_lookup = subset.set_index("template_key")["total_posts"].to_dict()

    configure_plot_style()
    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(selected_keys), 1)))
    for color, template_key in zip(colors, selected_keys):
        group = plot_df[plot_df["template_key"] == template_key].copy()
        ax.plot(
            group["period_start"],
            group[value_col],
            linewidth=2.0,
            color=color,
            label=(
                f"{rank_lookup[template_key]}. {name_lookup.get(template_key, template_key)} "
                f"(n={int(posts_lookup.get(template_key, 0))})"
            ),
        )
    ax.set_title(f"Representative {lifecycle_class.title()} Template Lifecycles")
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def save_simple_bar_chart(
    df: pd.DataFrame,
    category_col: str,
    value_col: str,
    title: str,
    ylabel: str,
    outpath: Path,
) -> None:
    if df.empty:
        return
    plot_df = _set_category_order(df.copy(), category_col)
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    colors = [CLASS_COLORS.get(str(value), "#577590") for value in plot_df[category_col].astype(str)]
    bars = ax.bar(plot_df[category_col].astype(str), plot_df[value_col], color=colors, alpha=0.92)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ymax = float(plot_df[value_col].max()) if not plot_df.empty else 0.0
    for bar, value in zip(bars, plot_df[value_col]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(ymax * 0.02, 0.5),
            f"{float(value):.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def save_stacked_sentiment_chart(
    df: pd.DataFrame,
    outpath: Path,
    title: str,
) -> pd.DataFrame:
    if df.empty or "local_sentiment_label" not in df.columns:
        return pd.DataFrame()
    working = df.copy()
    working = working[working["local_sentiment_label"].notna()].copy()
    if working.empty:
        return pd.DataFrame()

    summary = (
        working.groupby(["lifecycle_class", "local_sentiment_label"], observed=True)
        .size()
        .reset_index(name="count")
    )
    totals = summary.groupby("lifecycle_class", observed=True)["count"].sum().rename("class_total").reset_index()
    summary = summary.merge(totals, on="lifecycle_class", how="left")
    summary["proportion"] = summary["count"] / summary["class_total"]
    summary = _set_category_order(summary, "lifecycle_class")

    pivot = (
        summary.pivot(index="lifecycle_class", columns="local_sentiment_label", values="count")
        .fillna(0)
        .reindex(_ordered_classes(summary["lifecycle_class"]), fill_value=0)
    )
    if pivot.empty:
        return summary

    configure_plot_style()
    fig, ax = plt.subplots(figsize=(10.5, 6))
    bottoms = np.zeros(len(pivot), dtype=float)
    color_map = {
        "positive": LABEL_COLORS.get("positive", "#2a9d8f"),
        "neutral": LABEL_COLORS.get("neutral", "#8d99ae"),
        "negative": LABEL_COLORS.get("negative", "#d62828"),
        "unavailable": "#6c757d",
    }
    for sentiment in [col for col in ["positive", "neutral", "negative", "unavailable"] if col in pivot.columns]:
        values = pivot[sentiment].to_numpy(dtype=float)
        ax.bar(
            pivot.index.astype(str),
            values,
            bottom=bottoms,
            label=sentiment,
            color=color_map.get(sentiment, "#577590"),
            alpha=0.92,
        )
        bottoms += values
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Number of Memes")
    ax.legend(title="Local Sentiment", fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)
    return summary


def add_derived_affect_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "local_sentiment_score" in out.columns:
        out["polarity_strength"] = pd.to_numeric(out["local_sentiment_score"], errors="coerce").abs()
    else:
        out["polarity_strength"] = np.nan

    if {"positive_local", "negative_local"}.issubset(out.columns):
        pos = pd.to_numeric(out["positive_local"], errors="coerce").fillna(0.0).clip(lower=0.0)
        neg = pd.to_numeric(out["negative_local"], errors="coerce").fillna(0.0).clip(lower=0.0)
        denom = (pos + neg).replace(0, np.nan)
        out["ambivalence_index"] = (2 * np.minimum(pos, neg)) / denom
    else:
        out["ambivalence_index"] = np.nan

    local_emotion_cols = [f"{label}_local" for label in NOTEBOOK_EMOTION_LABELS if f"{label}_local" in out.columns]
    if local_emotion_cols:
        local_emotions = out[local_emotion_cols].apply(pd.to_numeric, errors="coerce")
        out["emotion_intensity"] = local_emotions.max(axis=1, skipna=True)
        out["emotion_entropy"] = np.nan
        probs = local_emotions.to_numpy(dtype=float)
        row_sums = probs.sum(axis=1, keepdims=True)
        valid = np.squeeze(row_sums > 0)
        if probs.size:
            normalized = np.divide(
                probs,
                np.where(row_sums == 0, np.nan, row_sums),
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                entropy = -(normalized * np.log(normalized))
            entropy = np.nansum(entropy, axis=1)
            denom = math.log(len(local_emotion_cols)) if len(local_emotion_cols) > 1 else np.nan
            out["emotion_entropy"] = entropy / denom if denom and not np.isnan(denom) else np.nan
            if "emotion_entropy" in out.columns:
                out.loc[~valid, "emotion_entropy"] = np.nan
    else:
        out["emotion_intensity"] = out["polarity_strength"]
        out["emotion_entropy"] = np.nan

    if "local_dominant_emotion" not in out.columns:
        out["local_dominant_emotion"] = "unavailable"
    if "local_sentiment_label" not in out.columns:
        out["local_sentiment_label"] = "unavailable"
    return out


def compute_class_level_affect_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    numeric_candidates = [
        "local_sentiment_score",
        "global_sentiment_score",
        "polarity_strength",
        "ambivalence_index",
        "emotion_intensity",
        "emotion_entropy",
        "positive_local",
        "neutral_local",
        "negative_local",
        "joy_local",
        "anger_local",
        "fear_local",
        "sadness_local",
        "surprise_local",
    ]
    available_numeric = [col for col in numeric_candidates if col in df.columns]
    aggregations: dict[str, Any] = {}
    for col in available_numeric:
        aggregations[f"{col}_mean"] = (col, "mean")
        aggregations[f"{col}_median"] = (col, "median")

    label_counts = (
        df.groupby(["lifecycle_class", "local_sentiment_label"], observed=True)
        .size()
        .reset_index(name="count")
    )
    if aggregations:
        summary = df.groupby("lifecycle_class", observed=True).agg(**aggregations).reset_index()
    else:
        summary = pd.DataFrame({"lifecycle_class": sorted(df["lifecycle_class"].astype(str).unique())})
    for sentiment in ["positive", "neutral", "negative", "unavailable"]:
        subset = label_counts[label_counts["local_sentiment_label"].astype(str) == sentiment].copy()
        subset = subset.rename(columns={"count": f"local_{sentiment}_count"}).drop(columns=["local_sentiment_label"])
        summary = summary.merge(subset, on="lifecycle_class", how="left")
    for column in [col for col in summary.columns if col.endswith("_count")]:
        summary[column] = pd.to_numeric(summary[column], errors="coerce").fillna(0).astype(int)
    count_cols = [col for col in summary.columns if col.endswith("_count")]
    if count_cols:
        summary["memes_in_class"] = summary[count_cols].sum(axis=1)
        for column in count_cols:
            share_col = column.replace("_count", "_share")
            summary[share_col] = summary[column] / summary["memes_in_class"].replace(0, np.nan)
    return _set_category_order(summary, "lifecycle_class")


def save_affect_metric_chart(summary_df: pd.DataFrame, outpath: Path) -> None:
    if summary_df.empty:
        return
    metric_cols = [
        col
        for col in [
            "polarity_strength_mean",
            "ambivalence_index_mean",
            "emotion_intensity_mean",
            "emotion_entropy_mean",
        ]
        if col in summary_df.columns
    ]
    if not metric_cols:
        return

    plot_df = _set_category_order(summary_df.copy(), "lifecycle_class")
    x = np.arange(len(plot_df))
    width = 0.18 if len(metric_cols) >= 4 else 0.22

    configure_plot_style()
    fig, ax = plt.subplots(figsize=(11.5, 6))
    colors = ["#264653", "#e9c46a", "#e76f51", "#2a9d8f"]
    for idx, column in enumerate(metric_cols):
        ax.bar(
            x + (idx - (len(metric_cols) - 1) / 2) * width,
            plot_df[column].to_numpy(dtype=float),
            width=width,
            label=column.replace("_mean", ""),
            color=colors[idx % len(colors)],
            alpha=0.92,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["lifecycle_class"].astype(str).tolist())
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean Value")
    ax.set_title("Affect Intensity and Ambivalence by Lifecycle Class")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def save_stacked_emotion_chart(
    df: pd.DataFrame,
    outpath: Path,
    title: str,
) -> pd.DataFrame:
    if df.empty or "local_dominant_emotion" not in df.columns:
        return pd.DataFrame()
    working = df.copy()
    working = working[working["local_dominant_emotion"].notna()].copy()
    if working.empty:
        return pd.DataFrame()

    summary = (
        working.groupby(["lifecycle_class", "local_dominant_emotion"], observed=True)
        .size()
        .reset_index(name="count")
    )
    summary = summary[summary["local_dominant_emotion"].astype(str) != "unavailable"].copy()
    if summary.empty:
        return pd.DataFrame()
    totals = summary.groupby("lifecycle_class", observed=True)["count"].sum().rename("class_total").reset_index()
    summary = summary.merge(totals, on="lifecycle_class", how="left")
    summary["proportion"] = summary["count"] / summary["class_total"]
    summary = _set_category_order(summary, "lifecycle_class")

    pivot = (
        summary.pivot(index="lifecycle_class", columns="local_dominant_emotion", values="count")
        .fillna(0)
        .reindex(_ordered_classes(summary["lifecycle_class"]), fill_value=0)
    )
    if pivot.empty:
        return summary

    emotion_order = [label for label in NOTEBOOK_EMOTION_LABELS if label in pivot.columns]
    if not emotion_order:
        return summary

    configure_plot_style()
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    bottoms = np.zeros(len(pivot), dtype=float)
    for emotion in emotion_order:
        values = pivot[emotion].to_numpy(dtype=float)
        ax.bar(
            pivot.index.astype(str),
            values,
            bottom=bottoms,
            label=emotion,
            color=LABEL_COLORS.get(emotion, "#577590"),
            alpha=0.92,
        )
        bottoms += values
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Number of Memes")
    ax.legend(title="Dominant Local Emotion", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)
    return summary


def build_template_affect_profiles(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    numeric_candidates = [
        "score",
        "local_sentiment_score",
        "polarity_strength",
        "ambivalence_index",
        "emotion_intensity",
        "emotion_entropy",
    ]
    numeric_candidates.extend([f"{label}_local" for label in NOTEBOOK_SENTIMENT_LABELS if f"{label}_local" in df.columns])
    numeric_candidates.extend([f"{label}_local" for label in NOTEBOOK_EMOTION_LABELS if f"{label}_local" in df.columns])
    numeric_candidates = [col for col in numeric_candidates if col in df.columns]

    aggregations: dict[str, Any] = {
        "template_final": ("template_final", "first"),
        "lifecycle_class": ("lifecycle_class", "first"),
        "total_memes": ("key", "size"),
    }
    for col in numeric_candidates:
        aggregations[f"{col}_mean"] = (col, "mean")
    profiles = df.groupby("template_key", observed=True).agg(**aggregations).reset_index()
    return _set_category_order(profiles, "lifecycle_class")


def save_template_affect_space_chart(template_df: pd.DataFrame, outpath: Path) -> None:
    if template_df.empty:
        return
    x_col = "polarity_strength_mean"
    y_col = "emotion_intensity_mean"
    color_col = "ambivalence_index_mean"
    if any(col not in template_df.columns for col in [x_col, y_col, color_col]):
        return

    classes = _ordered_classes(template_df["lifecycle_class"])
    if not classes:
        return

    configure_plot_style()
    fig, axes = plt.subplots(
        1,
        len(classes),
        figsize=(6.2 * len(classes), 5.5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes_list = np.atleast_1d(axes).ravel().tolist()
    scatter_obj = None
    for ax, lifecycle_class in zip(axes_list, classes):
        subset = template_df[template_df["lifecycle_class"].astype(str) == lifecycle_class].copy()
        if subset.empty:
            ax.axis("off")
            continue
        sizes = np.sqrt(pd.to_numeric(subset["total_memes"], errors="coerce").fillna(0.0).clip(lower=0.0)) * 8.0
        scatter_obj = ax.scatter(
            pd.to_numeric(subset[x_col], errors="coerce"),
            pd.to_numeric(subset[y_col], errors="coerce"),
            s=sizes,
            c=pd.to_numeric(subset[color_col], errors="coerce"),
            cmap="viridis",
            alpha=0.72,
            edgecolors="#1f2937",
            linewidths=0.35,
        )
        ax.set_title(f"{lifecycle_class.title()} Templates")
        ax.set_xlabel("Mean Polarity Strength")
        ax.set_ylabel("Mean Emotion Intensity")
        ax.grid(alpha=0.25)

        top_labels = subset.sort_values("total_memes", ascending=False).head(4)
        for row in top_labels.itertuples(index=False):
            x_val = getattr(row, x_col)
            y_val = getattr(row, y_col)
            if pd.notna(x_val) and pd.notna(y_val):
                ax.text(float(x_val), float(y_val), str(row.template_final), fontsize=7.5, alpha=0.9)

    for ax in axes_list[len(classes) :]:
        ax.axis("off")
    if scatter_obj is not None:
        cbar = fig.colorbar(scatter_obj, ax=axes_list[: len(classes)], fraction=0.03, pad=0.03)
        cbar.set_label("Mean Ambivalence Index")
    fig.suptitle("Template Affect Space by Lifecycle Class", fontsize=14, y=1.02)
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_per_class_affect_profile_charts(summary_df: pd.DataFrame, outdir: Path) -> None:
    if summary_df.empty:
        return
    profile_specs = [
        ("positive_local_mean", "Positive", LABEL_COLORS.get("positive", "#06d6a0")),
        ("neutral_local_mean", "Neutral", LABEL_COLORS.get("neutral", "#8d99ae")),
        ("negative_local_mean", "Negative", LABEL_COLORS.get("negative", "#d62828")),
        ("joy_local_mean", "Joy", LABEL_COLORS.get("joy", "#ffb703")),
        ("anger_local_mean", "Anger", LABEL_COLORS.get("anger", "#e63946")),
        ("fear_local_mean", "Fear", LABEL_COLORS.get("fear", "#6d597a")),
        ("sadness_local_mean", "Sadness", LABEL_COLORS.get("sadness", "#4361ee")),
        ("surprise_local_mean", "Surprise", LABEL_COLORS.get("surprise", "#9b5de5")),
        ("polarity_strength_mean", "Polarity Strength", "#264653"),
        ("ambivalence_index_mean", "Ambivalence", "#f4a261"),
        ("emotion_intensity_mean", "Emotion Intensity", "#2a9d8f"),
    ]

    for row in summary_df.itertuples(index=False):
        lifecycle_class = str(row.lifecycle_class)
        rows: list[tuple[str, float, str]] = []
        for column, label, color in profile_specs:
            value = getattr(row, column, np.nan)
            if pd.notna(value):
                rows.append((label, float(value), color))
        if not rows:
            continue
        plot_df = pd.DataFrame(rows, columns=["label", "value", "color"]).sort_values("value", ascending=True)
        configure_plot_style()
        fig, ax = plt.subplots(figsize=(9.5, max(5.0, 0.42 * len(plot_df) + 1.5)))
        ax.barh(plot_df["label"], plot_df["value"], color=plot_df["color"], alpha=0.92)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Mean Value")
        ax.set_ylabel("")
        ax.set_title(f"{lifecycle_class.title()} Affect Profile")
        for idx, value in enumerate(plot_df["value"]):
            ax.text(min(float(value) + 0.015, 1.01), idx, f"{float(value):.3f}", va="center", fontsize=8.5)
        fig.tight_layout()
        fig.savefig(outdir / f"fig_affect_profile_{lifecycle_class}.png", dpi=180)
        plt.close(fig)


def save_temporal_metric_chart(summary_df: pd.DataFrame, outpath: Path) -> None:
    if summary_df.empty:
        return
    metric_cols = [col for col in ["peak_share", "top3_share", "post_peak_active_ratio", "entropy_norm"] if col in summary_df.columns]
    if not metric_cols:
        return

    plot_df = _set_category_order(summary_df.copy(), "lifecycle_class")
    x = np.arange(len(plot_df))
    width = 0.18 if len(metric_cols) >= 4 else 0.22

    configure_plot_style()
    fig, ax = plt.subplots(figsize=(11.5, 6))
    colors = ["#e76f51", "#f4a261", "#2a9d8f", "#577590"]
    for idx, column in enumerate(metric_cols):
        ax.bar(
            x + (idx - (len(metric_cols) - 1) / 2) * width,
            plot_df[column].to_numpy(dtype=float),
            width=width,
            label=column,
            color=colors[idx % len(colors)],
            alpha=0.92,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["lifecycle_class"].astype(str).tolist())
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean Value")
    ax.set_title("Temporal Virality and Reuse Metrics by Lifecycle Class")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def summarize_topic_columns(df: pd.DataFrame, topic_columns: list[str]) -> pd.DataFrame:
    if df.empty or not topic_columns:
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    for column in topic_columns:
        if column not in df.columns:
            continue
        working = df[df[column].notna()].copy()
        if working.empty:
            continue
        summary = (
            working.groupby(["lifecycle_class", column], observed=True)
            .size()
            .reset_index(name="count")
            .rename(columns={column: "topic_value"})
        )
        totals = summary.groupby("lifecycle_class", observed=True)["count"].sum().rename("class_total").reset_index()
        summary = summary.merge(totals, on="lifecycle_class", how="left")
        summary["proportion"] = summary["count"] / summary["class_total"]
        summary["topic_column"] = column
        rows.append(summary)
    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows, ignore_index=True)
    return _set_category_order(combined, "lifecycle_class")


def load_topic_json_frame(path: str | Path, prefix: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    json_path = Path(path).expanduser().resolve()
    if not json_path.exists():
        return (
            pd.DataFrame(columns=["key", f"{prefix}_topic", f"{prefix}_topic_probability"]),
            {"status": "missing", "error": None, "path": str(json_path)},
        )

    try:
        payload = json.loads(json_path.read_text())
    except json.JSONDecodeError as exc:
        return (
            pd.DataFrame(columns=["key", f"{prefix}_topic", f"{prefix}_topic_probability"]),
            {"status": "invalid_json", "error": str(exc), "path": str(json_path)},
        )
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a dict in topic JSON: {json_path}")

    rows: list[dict[str, Any]] = []
    for key, value in payload.items():
        if not isinstance(value, dict):
            continue
        rows.append(
            {
                "key": str(key),
                f"{prefix}_topic": value.get("topic"),
                f"{prefix}_topic_probability": value.get("probability"),
            }
        )
    topic_df = pd.DataFrame(rows)
    if topic_df.empty:
        return (
            pd.DataFrame(columns=["key", f"{prefix}_topic", f"{prefix}_topic_probability"]),
            {"status": "empty", "error": None, "path": str(json_path)},
        )
    topic_df[f"{prefix}_topic"] = pd.to_numeric(topic_df[f"{prefix}_topic"], errors="coerce").astype("Int64")
    topic_df[f"{prefix}_topic_probability"] = pd.to_numeric(
        topic_df[f"{prefix}_topic_probability"], errors="coerce"
    )
    return topic_df, {"status": "loaded", "error": None, "path": str(json_path)}


def attach_topic_columns(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    out = df.copy()
    loaded_columns: list[str] = []
    metadata: dict[str, Any] = {}

    for prefix, source in [("global", args.global_topic_json), ("local", args.local_topic_json)]:
        if not source:
            metadata[f"{prefix}_topic_json"] = None
            metadata[f"{prefix}_topic_rows_loaded"] = 0
            metadata[f"{prefix}_topic_json_status"] = "disabled"
            metadata[f"{prefix}_topic_json_error"] = None
            continue
        topic_df, load_meta = load_topic_json_frame(source, prefix=prefix)
        metadata[f"{prefix}_topic_json"] = load_meta["path"]
        metadata[f"{prefix}_topic_rows_loaded"] = int(len(topic_df))
        metadata[f"{prefix}_topic_json_status"] = load_meta["status"]
        metadata[f"{prefix}_topic_json_error"] = load_meta["error"]
        if topic_df.empty:
            continue
        out = out.merge(topic_df, on="key", how="left", validate="one_to_one")
        loaded_columns.append(f"{prefix}_topic")

    return out, loaded_columns, metadata


def add_h1_affect_layers(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if args.emotion_backend == "none":
        out = df.copy()
        out["local_sentiment_label"] = "unavailable"
        out["global_sentiment_label"] = "unavailable"
        out["local_sentiment_score"] = np.nan
        out["global_sentiment_score"] = np.nan
        return add_derived_affect_metrics(out)

    affect_args = argparse.Namespace(
        emotion_backend=args.emotion_backend,
        emotion_cache_dir=args.emotion_cache_dir,
        emotion_batch_size=args.emotion_batch_size,
        emotion_max_length=args.emotion_max_length,
    )
    return add_derived_affect_metrics(add_affect_layers(df, affect_args))


def run_analysis(args: argparse.Namespace) -> Path:
    outdir = ensure_dir(args.results_dir)
    data_dir = Path(args.analysis_parquet).expanduser().resolve().parent
    raw_df, bundle_metadata = build_current_data_bundle_raw(data_dir)
    raw_df, template_merge_metadata = attach_template_assignments_from_parquet(raw_df, args.analysis_parquet)
    df = prepare_template_dataframe(prepare_analysis_dataframe(raw_df))
    df, auto_topic_columns, topic_load_metadata = attach_topic_columns(df, args)
    lifecycle_metrics, lifecycle_curves = build_template_lifecycle_features(df, args.lifecycle_freq)

    eligible = (
        lifecycle_metrics[
            (lifecycle_metrics["total_posts"] >= int(args.min_posts))
            & (lifecycle_metrics["observed_days_online"] >= float(args.min_observed_days))
        ]
        .copy()
        .reset_index(drop=True)
    )
    if eligible.empty:
        raise RuntimeError("No templates passed the lifecycle eligibility filters.")

    classified, cluster_profiles, feature_cols = classify_template_lifecycles(
        eligible,
        n_clusters=args.n_clusters,
        random_seed=args.random_seed,
        cluster_method=args.cluster_method,
    )
    representatives = select_representative_templates(
        classified,
        feature_cols=feature_cols,
        top_k=args.top_k_representatives,
    )

    class_lookup = classified.set_index("template_key")["lifecycle_class"].to_dict()
    meme_level = df.copy()
    meme_level["lifecycle_class"] = meme_level["template_key"].map(class_lookup)
    meme_level = meme_level[meme_level["lifecycle_class"].notna()].copy()
    meme_level = _set_category_order(meme_level, "lifecycle_class")
    meme_level = add_h1_affect_layers(meme_level, args)

    template_counts = (
        classified.groupby("lifecycle_class", observed=True)
        .size()
        .reset_index(name="template_count")
    )
    meme_counts = (
        meme_level.groupby("lifecycle_class", observed=True)
        .size()
        .reset_index(name="meme_count")
    )
    temporal_summary = (
        classified.groupby("lifecycle_class", observed=True)[
            [
                "active_ratio",
                "peak_share",
                "top3_share",
                "post_peak_active_ratio",
                "entropy_norm",
                "active_run_density",
                "reactivation_ratio",
                "longest_zero_ratio",
                "peak_age_ratio",
                "recent_share_3",
                "recent_growth_log",
                "temporal_virality_index",
            ]
        ]
        .mean()
        .reset_index()
    )
    affect_summary = compute_class_level_affect_summary(meme_level)
    template_affect_profiles = build_template_affect_profiles(meme_level)

    keyword_input = meme_level.loc[:, ["lifecycle_class", "local_text"]].rename(
        columns={"lifecycle_class": "dominant_label", "local_text": "source_text"}
    )
    keyword_weights, keyword_rows, keyword_backend = rank_words_for_wordclouds(
        dominant_df=keyword_input,
        min_token_len=args.keyword_min_token_len,
        backend=args.keyword_backend,
        embedding_model=args.keyword_embedding_model,
        max_docs_per_class=args.keyword_max_docs_per_class,
        random_seed=args.random_seed,
    )
    if not keyword_rows.empty:
        keyword_rows = (
            keyword_rows.sort_values(["label", "weight"], ascending=[True, False])
            .groupby("label", observed=True)
            .head(int(args.keyword_max_per_class))
            .reset_index(drop=True)
        )

    requested_topic_columns = [str(col) for col in args.topic_columns]
    effective_topic_columns: list[str] = []
    bundle_topic_candidates = [
        column
        for column in ["local_topic_name", "global_topic_name", "local_topic", "global_topic"]
        if column in meme_level.columns
    ]
    for column in requested_topic_columns + bundle_topic_candidates + auto_topic_columns:
        if column not in effective_topic_columns and column in meme_level.columns:
            effective_topic_columns.append(column)
    topic_summary = summarize_topic_columns(meme_level, topic_columns=effective_topic_columns)

    lifecycle_metrics.to_csv(outdir / "template_lifecycle_metrics_all.csv", index=False)
    classified.to_csv(outdir / "template_lifecycle_metrics_classified.csv", index=False)
    cluster_profiles.to_csv(outdir / "lifecycle_cluster_profiles.csv", index=False)
    representatives.to_csv(outdir / "representative_templates.csv", index=False)
    meme_level.loc[:, ["key", "template_final", "template_key", "lifecycle_class"]].to_parquet(
        outdir / "meme_level_lifecycle_assignments.parquet",
        index=False,
    )
    template_counts.to_csv(outdir / "lifecycle_class_template_counts.csv", index=False)
    meme_counts.to_csv(outdir / "lifecycle_class_meme_counts.csv", index=False)
    temporal_summary.to_csv(outdir / "lifecycle_class_temporal_summary.csv", index=False)
    affect_summary.to_csv(outdir / "lifecycle_class_affect_summary.csv", index=False)
    template_affect_profiles.to_csv(outdir / "template_affect_profiles.csv", index=False)
    keyword_rows.to_csv(outdir / "lifecycle_class_keyword_weights.csv", index=False)
    if not topic_summary.empty:
        topic_summary.to_csv(outdir / "lifecycle_class_topic_summary.csv", index=False)

    representative_curve_keys = representatives["template_key"].tolist()
    representative_curves = lifecycle_curves[lifecycle_curves["template_key"].isin(representative_curve_keys)].copy()
    representative_curves.to_csv(outdir / "representative_template_lifecycle_curves.csv", index=False)

    for lifecycle_class in _ordered_classes(classified["lifecycle_class"]):
        safe_name = str(lifecycle_class).replace(" ", "_")
        save_representative_lifecycle_plot(
            representative_curves,
            representatives,
            lifecycle_class=str(lifecycle_class),
            outpath=outdir / f"fig_representative_{safe_name}_lifecycles_normalized.png",
            value_col="smooth_norm",
            ylabel="Relative Lifecycle Intensity",
        )
    save_simple_bar_chart(
        template_counts,
        category_col="lifecycle_class",
        value_col="template_count",
        title="Templates per Lifecycle Class",
        ylabel="Number of Templates",
        outpath=outdir / "fig_lifecycle_class_template_counts.png",
    )
    save_simple_bar_chart(
        meme_counts,
        category_col="lifecycle_class",
        value_col="meme_count",
        title="Memes per Lifecycle Class",
        ylabel="Number of Memes",
        outpath=outdir / "fig_lifecycle_class_meme_counts.png",
    )
    sentiment_counts = save_stacked_sentiment_chart(
        meme_level,
        outpath=outdir / "fig_lifecycle_class_meme_counts_by_sentiment.png",
        title="Memes per Lifecycle Class, Split by Local Sentiment",
    )
    if not sentiment_counts.empty:
        sentiment_counts.to_csv(outdir / "lifecycle_class_sentiment_counts.csv", index=False)
    emotion_counts = save_stacked_emotion_chart(
        meme_level,
        outpath=outdir / "fig_lifecycle_class_meme_counts_by_emotion.png",
        title="Memes per Lifecycle Class, Split by Dominant Local Emotion",
    )
    if not emotion_counts.empty:
        emotion_counts.to_csv(outdir / "lifecycle_class_emotion_counts.csv", index=False)
    save_temporal_metric_chart(
        temporal_summary,
        outpath=outdir / "fig_temporal_virality_by_lifecycle_class.png",
    )
    save_affect_metric_chart(
        affect_summary,
        outpath=outdir / "fig_affect_metrics_by_lifecycle_class.png",
    )
    save_template_affect_space_chart(
        template_affect_profiles,
        outpath=outdir / "fig_template_affect_space_by_lifecycle_class.png",
    )
    save_per_class_affect_profile_charts(
        affect_summary,
        outdir=outdir,
    )

    class_overview = template_counts.merge(meme_counts, on="lifecycle_class", how="outer")
    for column in ["template_count", "meme_count"]:
        if column in class_overview.columns:
            class_overview[column] = pd.to_numeric(class_overview[column], errors="coerce").fillna(0).astype(int)
    class_overview = _set_category_order(class_overview, "lifecycle_class")
    actual_cluster_count = int(classified["lifecycle_cluster_id"].nunique())
    class_summary_bits: list[str] = []
    for row in class_overview.itertuples(index=False):
        lifecycle_class = str(row.lifecycle_class)
        examples = (
            representatives[representatives["lifecycle_class"].astype(str) == lifecycle_class]["template_final"]
            .head(3)
            .tolist()
        )
        class_summary_bits.append(
            f"{lifecycle_class}: {int(row.template_count)} templates / {int(row.meme_count)} memes "
            f"(examples: {', '.join(examples) if examples else 'none'})"
        )
    evergreen_metrics = classified[classified["lifecycle_class"].astype(str) == "evergreen"]
    bursty_metrics = classified[classified["lifecycle_class"].astype(str) == "bursty"]

    bullets = [
        (
            f"Rows analyzed: {len(df)} memes, {df['template_key'].nunique()} template keys total; "
            f"{len(classified)} templates passed the H1 lifecycle filters."
        ),
        (
            f"Lifecycle eligibility: at least {int(args.min_posts)} posts and "
            f"{int(args.min_observed_days)} observed days online."
        ),
        (
            f"Clustering: {args.cluster_method} with k={actual_cluster_count}, fit on {', '.join(feature_cols)}."
        ),
        (
            "Lifecycle score formulas: evergreen rewards persistence and low concentration; recurring rewards "
            "reactivation and multi-peak reuse; faded rewards old peaks and weak recent activity; emerging "
            "rewards recent share and positive recent growth; bursty rewards concentrated peaks and low persistence."
        ),
        (
            f"Cluster overview: {'; '.join(class_summary_bits)}."
        ),
        (
            f"Mean temporal virality contrast: evergreen peak_share={evergreen_metrics['peak_share'].mean():.3f}, "
            f"bursty peak_share={bursty_metrics['peak_share'].mean():.3f}; evergreen active_ratio="
            f"{evergreen_metrics['active_ratio'].mean():.3f}, bursty active_ratio={bursty_metrics['active_ratio'].mean():.3f}."
            if not evergreen_metrics.empty and not bursty_metrics.empty
            else "Evergreen/bursty temporal contrast unavailable because one class was empty."
        ),
        (
            f"Affect contrast: evergreen polarity_strength={affect_summary.loc[affect_summary['lifecycle_class'].astype(str) == 'evergreen', 'polarity_strength_mean'].iloc[0]:.3f}, "
            f"bursty polarity_strength={affect_summary.loc[affect_summary['lifecycle_class'].astype(str) == 'bursty', 'polarity_strength_mean'].iloc[0]:.3f}; "
            f"evergreen emotion_intensity={affect_summary.loc[affect_summary['lifecycle_class'].astype(str) == 'evergreen', 'emotion_intensity_mean'].iloc[0]:.3f}, "
            f"bursty emotion_intensity={affect_summary.loc[affect_summary['lifecycle_class'].astype(str) == 'bursty', 'emotion_intensity_mean'].iloc[0]:.3f}."
            if {
                "polarity_strength_mean",
                "emotion_intensity_mean",
            }.issubset(affect_summary.columns)
            and (affect_summary["lifecycle_class"].astype(str) == "evergreen").any()
            and (affect_summary["lifecycle_class"].astype(str) == "bursty").any()
            else "Affect contrast unavailable because the affect summary did not include the needed metrics."
        ),
        (
            f"Keyword backend used for class-level summaries: {keyword_backend}."
        ),
        (
            f"Topic summary columns included: {', '.join(effective_topic_columns)}."
            if effective_topic_columns
            else "Topic summaries skipped because no topic columns were available."
        ),
    ]
    write_summary_markdown(outdir / "summary.md", "H1 Lifecycle Cluster Analysis", bullets)
    write_run_metadata(
        outdir / "run_metadata.json",
        {
            "analysis_parquet": str(Path(args.analysis_parquet).expanduser().resolve()),
            "data_source": "current_data_bundle_with_template_merge",
            "results_dir": str(outdir),
            "rows_total": int(len(df)),
            "template_keys_total": int(df["template_key"].nunique()),
            "eligible_templates": int(len(classified)),
            "eligible_memes": int(len(meme_level)),
            "min_posts": int(args.min_posts),
            "min_observed_days": int(args.min_observed_days),
            "lifecycle_freq": str(args.lifecycle_freq),
            "n_clusters": int(args.n_clusters),
            "clusters_fit": actual_cluster_count,
            "cluster_method": str(args.cluster_method),
            "cluster_feature_columns": feature_cols,
            "lifecycle_score_specs": LIFECYCLE_SCORE_SPECS,
            "random_seed": int(args.random_seed),
            "top_k_representatives": int(args.top_k_representatives),
            "emotion_backend": str(args.emotion_backend),
            "emotion_cache_dir": str(Path(args.emotion_cache_dir).expanduser().resolve()),
            "keyword_backend_requested": str(args.keyword_backend),
            "keyword_backend_used": keyword_backend,
            "keyword_embedding_model": str(args.keyword_embedding_model),
            "keyword_min_token_len": int(args.keyword_min_token_len),
            "topic_columns_requested": requested_topic_columns,
            "topic_columns_loaded": effective_topic_columns,
            **bundle_metadata,
            **template_merge_metadata,
            **topic_load_metadata,
        },
    )
    print(f"results_dir={outdir}")
    return outdir


def main() -> None:
    run_analysis(parse_args())


if __name__ == "__main__":
    main()
