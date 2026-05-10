#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import shutil
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

try:
    from PIL import Image, ImageOps, UnidentifiedImageError
except Exception:
    Image = None
    ImageOps = None
    UnidentifiedImageError = OSError

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from common import build_text, configure_plot_style, ensure_dir  # noqa: E402
from q2_global_local_context import (  # noqa: E402
    NOTEBOOK_EMOTION_LABELS,
    NOTEBOOK_SENTIMENT_LABELS,
    add_affect_layers,
    attach_reddit_metadata,
    attach_topic_assignments,
    load_combined_sentiment_emotion_jsonl,
)


FONT = {
    "title": 24,
    "suptitle": 30,
    "axis_label": 19,
    "tick": 20,
    "annotation": 14,
}
GROUP_ORDER = ["popular", "unpopular"]
GROUP_LABELS = {"popular": "Popular Templates", "unpopular": "Unpopular Templates"}
NO_TEMPLATE_LABELS = {"NO_TEMPLATE", "NON_MEME", "NO_MEME", ""}


def load_dotenv_file(path: str | Path) -> None:
    dotenv_path = Path(path)
    if not dotenv_path.exists():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compact current-progress analysis combining popularity, affect, topics, and temporal clusters."
    )
    parser.add_argument("--analysis-parquet", default="data/template_first_analysis_table.parquet")
    parser.add_argument("--input-jsonl", default="data/combined_memes_sentiments_emotions.jsonl")
    parser.add_argument("--global-topics-json", default="Meme_gemini/bertopic_models/meme_global_contexts_topics.json")
    parser.add_argument("--local-topics-json", default="Meme_gemini/bertopic_models/meme_local_contexts_topics.json")
    parser.add_argument("--results-dir", default="current_progress/results")
    parser.add_argument("--popular-top-n", type=int, default=20)
    parser.add_argument("--unpopular-bottom-n", type=int, default=20)
    parser.add_argument("--template-grid-top-n", type=int, default=10)
    parser.add_argument("--min-template-posts", type=int, default=20)
    parser.add_argument("--transition-threshold", type=float, default=0.0)
    parser.add_argument("--temporal-freq", default="MS")
    parser.add_argument("--temporal-clusters", type=int, default=3)
    parser.add_argument("--trajectory-clusters", type=int, default=5)
    parser.add_argument("--trajectory-normalization", choices=["max", "total"], default="max")
    parser.add_argument("--trajectory-smooth-window", type=int, default=1)
    parser.add_argument(
        "--estimate-template-years",
        action="store_true",
        help="Call a multimodal LLM to estimate each clustered template's first appearance year.",
    )
    parser.add_argument("--template-year-provider", choices=["openai"], default="openai")
    parser.add_argument("--template-year-model", default="gpt-4.1-nano")
    parser.add_argument("--template-year-cache", default="current_progress/results/rq2_template_trajectory_clusters/template_first_year_llm_cache.jsonl")
    parser.add_argument("--top-topics-per-cluster", type=int, default=8)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def configure_current_plot_style() -> None:
    configure_plot_style()
    plt.rcParams.update(
        {
            "axes.titlesize": FONT["title"],
            "axes.labelsize": FONT["axis_label"],
            "xtick.labelsize": FONT["tick"],
            "ytick.labelsize": FONT["tick"],
            "legend.fontsize": FONT["annotation"],
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#f8fafc",
            "grid.alpha": 0.35,
        }
    )


def report_figures_dir(results_dir: str | Path) -> Path:
    return ensure_dir(ROOT / "curr_figures")


def copy_report_figure(source: str | Path, figures_dir: Path) -> None:
    source_path = Path(source)
    if source_path.exists():
        shutil.copy2(source_path, figures_dir / source_path.name)


def load_analysis_frame(args: argparse.Namespace) -> pd.DataFrame:
    df = load_combined_sentiment_emotion_jsonl(args.input_jsonl)
    df = attach_reddit_metadata(df, args.analysis_parquet)
    topic_args = SimpleNamespace(
        global_topics_json=args.global_topics_json,
        local_topics_json=args.local_topics_json,
    )
    df = attach_topic_assignments(df, topic_args)
    df["score"] = pd.to_numeric(df.get("score"), errors="coerce").fillna(0.0)
    df["created_utc"] = pd.to_datetime(df.get("created_utc"), errors="coerce")
    df["template_final"] = df.get("template_final", "").fillna("").astype(str)
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
    affect_args = SimpleNamespace(
        emotion_backend="cardiff",
        topics_only=True,
        emotion_cache_dir="analysis/cache/q2_cardiff_affect",
        emotion_batch_size=32,
        emotion_max_length=512,
    )
    return add_affect_layers(df, affect_args)


def dominant_from_columns(df: pd.DataFrame, labels: list[str], suffix: str) -> pd.Series:
    cols = [f"{label}_{suffix}" for label in labels if f"{label}_{suffix}" in df.columns]
    if not cols:
        return pd.Series("unavailable", index=df.index)
    numeric = df[cols].apply(pd.to_numeric, errors="coerce")
    all_na = numeric.isna().all(axis=1)
    dominant = numeric.fillna(-np.inf).idxmax(axis=1).str.replace(f"_{suffix}", "", regex=False)
    return dominant.mask(all_na, "unavailable")


def assign_popularity_groups(df: pd.DataFrame, args: argparse.Namespace, outdir: Path) -> pd.DataFrame:
    template_counts = (
        df.loc[~df["template_final"].isin(NO_TEMPLATE_LABELS), "template_final"]
        .value_counts()
        .rename_axis("template_final")
        .reset_index(name="post_count")
    )
    eligible = template_counts[template_counts["post_count"].ge(args.min_template_posts)].copy()
    popular = eligible.head(args.popular_top_n).copy()
    popular["popularity_group"] = "popular"
    unpopular = eligible.sort_values(["post_count", "template_final"], ascending=[True, True]).head(args.unpopular_bottom_n).copy()
    unpopular["popularity_group"] = "unpopular"
    template_groups = pd.concat([popular, unpopular], ignore_index=True)
    template_groups.to_csv(outdir / "popularity_template_groups.csv", index=False)

    group_lookup = template_groups.set_index("template_final")["popularity_group"].to_dict()
    out = df.copy()
    out["popularity_group"] = out["template_final"].map(group_lookup)
    out = out[out["popularity_group"].isin(GROUP_ORDER)].copy()
    out.to_csv(outdir / "popularity_group_posts.csv", index=False)
    return out


def load_rgb_image(path: str | Path, max_side: int = 900):
    if Image is None or ImageOps is None:
        return None
    try:
        with Image.open(Path(path).expanduser()) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((max_side, max_side))
            return image.copy()
    except (FileNotFoundError, OSError, UnidentifiedImageError, ValueError):
        return None


def select_template_representatives(grouped: pd.DataFrame, group: str, top_n: int) -> pd.DataFrame:
    subset = grouped[grouped["popularity_group"].eq(group)].copy()
    if subset.empty or "image_path" not in subset.columns:
        return pd.DataFrame()
    counts = (
        subset["template_final"]
        .value_counts()
        .rename_axis("template_final")
        .reset_index(name="post_count")
    )
    if group == "popular":
        counts = counts.sort_values(["post_count", "template_final"], ascending=[False, True]).head(top_n)
    else:
        counts = counts.sort_values(["post_count", "template_final"], ascending=[True, True]).head(top_n)

    rows: list[dict[str, Any]] = []
    for rank, template_row in enumerate(counts.itertuples(index=False), start=1):
        template_posts = subset[subset["template_final"].eq(template_row.template_final)].copy()
        template_posts["score"] = pd.to_numeric(template_posts.get("score"), errors="coerce").fillna(0.0)
        template_posts = template_posts.sort_values(["score", "created_utc"], ascending=[False, True])
        image_path = ""
        for candidate in template_posts["image_path"].dropna().astype(str):
            if load_rgb_image(candidate) is not None:
                image_path = candidate
                break
        rows.append(
            {
                "rank": rank,
                "popularity_group": group,
                "template_final": template_row.template_final,
                "post_count": int(template_row.post_count),
                "image_path": image_path,
            }
        )
    return pd.DataFrame(rows)


def save_template_image_grid(representatives: pd.DataFrame, outpath: Path, title: str) -> None:
    if representatives.empty:
        return
    configure_current_plot_style()
    ncols = 5
    nrows = int(np.ceil(len(representatives) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(19, 5.4 * nrows), squeeze=False)
    axes_list = axes.ravel().tolist()

    for ax, row in zip(axes_list, representatives.itertuples(index=False)):
        ax.axis("off")
        image = load_rgb_image(row.image_path) if row.image_path else None
        if image is not None:
            ax.imshow(image)
        wrapped_name = "\n".join(textwrap.wrap(str(row.template_final).replace("_", " "), width=28))
        ax.set_title(
            f"{int(row.rank)}. {wrapped_name}\nn={int(row.post_count)}",
            fontsize=FONT["tick"],
            pad=16,
        )

    for ax in axes_list[len(representatives) :]:
        ax.axis("off")

    fig.suptitle(title, fontsize=FONT["suptitle"], y=0.995)
    fig.tight_layout(h_pad=2.4, w_pad=1.2)
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def compute_group_affect_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group in GROUP_ORDER:
        subset = df[df["popularity_group"].eq(group)].copy()
        row: dict[str, Any] = {
            "popularity_group": group,
            "posts": int(len(subset)),
            "templates": int(subset["template_final"].nunique()),
            "mean_upvotes": float(subset["score"].mean()) if len(subset) else np.nan,
            "median_upvotes": float(subset["score"].median()) if len(subset) else np.nan,
        }
        for context in ["global", "local"]:
            positive = pd.to_numeric(subset.get(f"positive_{context}"), errors="coerce")
            negative = pd.to_numeric(subset.get(f"negative_{context}"), errors="coerce")
            row[f"{context}_sentiment_balance"] = float((positive - negative).mean())
            for label in NOTEBOOK_SENTIMENT_LABELS + NOTEBOOK_EMOTION_LABELS:
                col = f"{label}_{context}"
                if col in subset.columns:
                    row[col] = float(pd.to_numeric(subset[col], errors="coerce").mean())
        rows.append(row)
    return pd.DataFrame(rows)


def zscore_matrix(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in columns:
        values = pd.to_numeric(out[col], errors="coerce")
        std = values.std(ddof=0)
        out[col] = 0.0 if pd.isna(std) or std == 0 else (values - values.mean()) / std
    return out


def save_group_affect_heatmap(summary: pd.DataFrame, outpath: Path) -> None:
    configure_current_plot_style()
    columns = [
        "global_sentiment_balance",
        "local_sentiment_balance",
        "joy_global",
        "joy_local",
        "anger_global",
        "anger_local",
        "sadness_global",
        "sadness_local",
        "disgust_global",
        "disgust_local",
        "optimism_global",
        "optimism_local",
    ]
    columns = [col for col in columns if col in summary.columns]
    plot_df = summary.set_index("popularity_group").loc[GROUP_ORDER].reset_index()
    z = zscore_matrix(plot_df, columns)
    matrix = z[columns].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(18, 6.5), constrained_layout=True)
    image = ax.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-1.5, vmax=1.5)
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels([col.replace("_", " ").title() for col in columns], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(GROUP_ORDER)))
    ax.set_yticklabels([GROUP_LABELS[group] for group in GROUP_ORDER])
    ax.set_title("Affective Profiles by Template Popularity Group")
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            ax.text(col_idx, row_idx, f"{matrix[row_idx, col_idx]:.1f}", ha="center", va="center", fontsize=13)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label("Group-Level Z-Score")
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def transition_matrix(df: pd.DataFrame, labels: list[str], group: str, kind: str) -> pd.DataFrame:
    subset = df[df["popularity_group"].eq(group)].copy()
    subset["global_label"] = dominant_from_columns(subset, labels, "global")
    subset["local_label"] = dominant_from_columns(subset, labels, "local")
    subset = subset[subset["global_label"].isin(labels) & subset["local_label"].isin(labels)].copy()
    counts = pd.crosstab(
        pd.Categorical(subset["global_label"], categories=labels, ordered=True),
        pd.Categorical(subset["local_label"], categories=labels, ordered=True),
        dropna=False,
    )
    counts.index = labels
    counts.columns = labels
    probs = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    probs.index.name = f"global_{kind}"
    probs.columns.name = f"local_{kind}"
    return probs


def save_transition_figure(df: pd.DataFrame, labels: list[str], kind: str, outpath: Path) -> None:
    configure_current_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), constrained_layout=True)
    image = None
    for ax, group in zip(axes, GROUP_ORDER):
        probs = transition_matrix(df, labels, group, kind)
        image = ax.imshow(probs.to_numpy(dtype=float), cmap="YlGnBu", vmin=0, vmax=max(0.01, float(probs.max().max())))
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels([label.title() for label in labels], rotation=35, ha="right")
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels([label.title() for label in labels])
        ax.set_xlabel(f"Local {kind.title()}")
        ax.set_ylabel(f"Global {kind.title()}")
        ax.set_title(GROUP_LABELS[group])
        for row_idx in range(len(labels)):
            for col_idx in range(len(labels)):
                value = probs.iloc[row_idx, col_idx]
                if value >= 0.05:
                    ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", fontsize=11)
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
        colorbar.set_label("Row-Normalized Probability")
    fig.suptitle(f"Global-to-Local {kind.title()} Transitions by Popularity Group", fontsize=FONT["suptitle"])
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_topic_share_comparison(df: pd.DataFrame, outpath: Path, top_n: int = 8) -> None:
    configure_current_plot_style()
    rows = []
    for context in ["global", "local"]:
        topic_col = f"{context}_topic"
        if topic_col not in df.columns:
            continue
        for group in GROUP_ORDER:
            subset = df[df["popularity_group"].eq(group)].copy()
            subset[topic_col] = pd.to_numeric(subset[topic_col], errors="coerce")
            subset = subset[subset[topic_col].notna() & subset[topic_col].ne(-1)]
            total = max(len(subset), 1)
            shares = subset[topic_col].astype(int).value_counts(normalize=True).head(top_n)
            for topic, share in shares.items():
                rows.append({"context": context, "popularity_group": group, "topic": int(topic), "share": float(share), "total": total})
    topic_df = pd.DataFrame(rows)
    if topic_df.empty:
        return
    topic_df.to_csv(outpath.with_suffix(".csv"), index=False)
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), constrained_layout=True)
    for ax, context in zip(axes, ["global", "local"]):
        subset = topic_df[topic_df["context"].eq(context)].copy()
        labels = sorted(subset["topic"].unique())
        x = np.arange(len(labels))
        width = 0.38
        for offset, group in [(-width / 2, "popular"), (width / 2, "unpopular")]:
            values = [
                subset.loc[subset["popularity_group"].eq(group) & subset["topic"].eq(topic), "share"].sum()
                for topic in labels
            ]
            ax.bar(x + offset, values, width=width, label=GROUP_LABELS[group])
        ax.set_xticks(x)
        ax.set_xticklabels([f"T{topic}" for topic in labels], rotation=35, ha="right")
        ax.set_ylabel("Topic Share")
        ax.set_title(f"{context.title()} Topic Shares")
        ax.legend()
    fig.suptitle("Topic Concentration by Template Popularity Group", fontsize=FONT["suptitle"])
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_popularity_analysis(df: pd.DataFrame, args: argparse.Namespace, outdir: Path) -> None:
    pop_dir = ensure_dir(outdir / "rq1_rq2_popularity_groups")
    figures_dir = report_figures_dir(outdir)
    grouped = assign_popularity_groups(df, args, pop_dir)
    popular_representatives = select_template_representatives(grouped, "popular", args.template_grid_top_n)
    unpopular_representatives = select_template_representatives(grouped, "unpopular", args.template_grid_top_n)
    popular_representatives.to_csv(pop_dir / "top_popular_template_examples.csv", index=False)
    unpopular_representatives.to_csv(pop_dir / "top_unpopular_template_examples.csv", index=False)
    popular_grid_path = pop_dir / "fig_top_10_popular_template_examples.png"
    unpopular_grid_path = pop_dir / "fig_top_10_unpopular_template_examples.png"
    save_template_image_grid(popular_representatives, popular_grid_path, "Top 10 Popular Template Examples")
    save_template_image_grid(unpopular_representatives, unpopular_grid_path, "Top 10 Unpopular Template Examples")
    summary = compute_group_affect_summary(grouped)
    summary.to_csv(pop_dir / "popularity_group_affect_summary.csv", index=False)
    affect_path = pop_dir / "fig_popularity_group_affect_heatmap.png"
    sentiment_path = pop_dir / "fig_popularity_group_sentiment_transitions.png"
    emotion_path = pop_dir / "fig_popularity_group_emotion_transitions.png"
    topic_path = pop_dir / "fig_popularity_group_topic_shares.png"
    save_group_affect_heatmap(summary, affect_path)
    save_transition_figure(grouped, NOTEBOOK_SENTIMENT_LABELS, "sentiment", sentiment_path)
    emotion_labels = ["joy", "anticipation", "disgust", "sadness", "anger", "optimism", "surprise", "fear"]
    save_transition_figure(grouped, emotion_labels, "emotion", emotion_path)
    save_topic_share_comparison(grouped, topic_path, top_n=8)
    for path in [popular_grid_path, unpopular_grid_path, affect_path, sentiment_path, emotion_path, topic_path]:
        copy_report_figure(path, figures_dir)


def build_template_trajectory_matrix(
    df: pd.DataFrame,
    freq: str,
    min_posts: int,
    normalization: str,
    smooth_window: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = df[df["created_utc"].notna()].copy()
    working = working[~working["template_final"].isin(NO_TEMPLATE_LABELS)].copy()
    if working.empty:
        return pd.DataFrame(), pd.DataFrame()

    template_counts = working["template_final"].value_counts()
    eligible_templates = template_counts[template_counts.ge(int(min_posts))].index.sort_values().tolist()
    working = working[working["template_final"].isin(eligible_templates)].copy()
    if working.empty:
        return pd.DataFrame(), pd.DataFrame()

    period_freq = "M" if str(freq).upper() == "MS" else str(freq)
    working["period_start"] = working["created_utc"].dt.to_period(period_freq).dt.to_timestamp()
    full_periods = pd.date_range(working["period_start"].min(), working["period_start"].max(), freq=freq)
    counts = (
        working.groupby(["template_final", "period_start"], observed=True)
        .size()
        .rename("count")
        .reset_index()
        .pivot(index="template_final", columns="period_start", values="count")
        .reindex(index=eligible_templates, columns=full_periods, fill_value=0)
        .fillna(0.0)
        .astype(float)
    )
    counts.index.name = "template_final"
    counts.columns.name = "period_start"

    window = max(1, int(smooth_window))
    smoothed = counts.T.rolling(window=window, min_periods=1).mean().T
    smoothed.index.name = "template_final"
    smoothed.columns.name = "period_start"
    if normalization == "max":
        denom = smoothed.max(axis=1).replace(0.0, np.nan)
    elif normalization == "total":
        denom = smoothed.sum(axis=1).replace(0.0, np.nan)
    else:
        raise ValueError(f"Unsupported trajectory normalization: {normalization}")
    normalized = smoothed.div(denom, axis=0).fillna(0.0)
    normalized.index.name = "template_final"
    normalized.columns.name = "period_start"

    curve_long = counts.stack().rename("count").reset_index()
    smooth_long = smoothed.stack().rename("smoothed_count").reset_index()
    norm_long = normalized.stack().rename("normalized_count").reset_index()
    curve_long = curve_long.merge(smooth_long, on=["template_final", "period_start"], how="left")
    curve_long = curve_long.merge(norm_long, on=["template_final", "period_start"], how="left")
    return normalized, curve_long


def assign_template_trajectory_clusters(
    trajectories: pd.DataFrame,
    n_clusters: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if trajectories.empty:
        return pd.DataFrame(), pd.DataFrame()
    k = max(1, min(int(n_clusters), len(trajectories)))
    model = KMeans(n_clusters=k, random_state=int(random_seed), n_init=20)
    raw_labels = model.fit_predict(trajectories.to_numpy(dtype=float))
    centroids = pd.DataFrame(model.cluster_centers_, columns=trajectories.columns)
    peak_order = centroids.idxmax(axis=1).sort_values().index.tolist()
    remap = {raw_cluster: idx + 1 for idx, raw_cluster in enumerate(peak_order)}

    assignments = pd.DataFrame(
        {
            "template_final": trajectories.index.astype(str),
            "trajectory_cluster_raw": raw_labels.astype(int),
        }
    )
    assignments["trajectory_cluster"] = assignments["trajectory_cluster_raw"].map(remap).astype(int)
    assignments["total_posts"] = trajectories.sum(axis=1).reindex(assignments["template_final"]).to_numpy(dtype=float)

    centroid_rows: list[dict[str, Any]] = []
    for raw_cluster, cluster_id in remap.items():
        centroid = centroids.loc[raw_cluster]
        for period_start, value in centroid.items():
            centroid_rows.append(
                {
                    "trajectory_cluster": int(cluster_id),
                    "period_start": pd.Timestamp(period_start),
                    "centroid_value": float(value),
                }
            )
    return assignments.sort_values(["trajectory_cluster", "template_final"]).reset_index(drop=True), pd.DataFrame(centroid_rows)


def save_template_trajectory_centroids(centroids_long: pd.DataFrame, outpath: Path) -> None:
    if centroids_long.empty:
        return
    configure_current_plot_style()
    fig, ax = plt.subplots(figsize=(18, 7), constrained_layout=True)
    clusters = sorted(centroids_long["trajectory_cluster"].dropna().astype(int).unique())
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(clusters), 1)))
    for color, cluster in zip(colors, clusters):
        subset = centroids_long[centroids_long["trajectory_cluster"].eq(cluster)].sort_values("period_start")
        ax.plot(
            subset["period_start"],
            subset["centroid_value"],
            color=color,
            linewidth=2.8,
            label=f"Cluster {int(cluster)}",
        )
    ax.set_xlabel("Time")
    ax.set_ylabel("Normalized Template Trajectory")
    ax.set_title("Centroids of Normalized Meme-Template Trajectory Clusters")
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def compute_trajectory_cluster_affect_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    clusters = sorted(df["trajectory_cluster"].dropna().astype(int).unique())
    for cluster in clusters:
        subset = df[df["trajectory_cluster"].eq(cluster)].copy()
        row: dict[str, Any] = {
            "trajectory_cluster": int(cluster),
            "posts": int(len(subset)),
            "templates": int(subset["template_final"].nunique()),
            "mean_upvotes": float(subset["score"].mean()) if len(subset) else np.nan,
            "median_upvotes": float(subset["score"].median()) if len(subset) else np.nan,
        }
        for context in ["global", "local"]:
            positive = pd.to_numeric(subset.get(f"positive_{context}"), errors="coerce")
            negative = pd.to_numeric(subset.get(f"negative_{context}"), errors="coerce")
            row[f"{context}_sentiment_balance"] = float((positive - negative).mean())
            for label in NOTEBOOK_SENTIMENT_LABELS + NOTEBOOK_EMOTION_LABELS:
                col = f"{label}_{context}"
                if col in subset.columns:
                    row[col] = float(pd.to_numeric(subset[col], errors="coerce").mean())
        rows.append(row)
    return pd.DataFrame(rows)


def save_trajectory_cluster_affect_heatmap(summary: pd.DataFrame, outpath: Path) -> None:
    if summary.empty:
        return
    configure_current_plot_style()
    columns = [
        "global_sentiment_balance",
        "local_sentiment_balance",
        "joy_global",
        "joy_local",
        "anger_global",
        "anger_local",
        "sadness_global",
        "sadness_local",
        "disgust_global",
        "disgust_local",
        "optimism_global",
        "optimism_local",
    ]
    columns = [col for col in columns if col in summary.columns]
    z = zscore_matrix(summary, columns)
    matrix = z[columns].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(18, 8), constrained_layout=True)
    image = ax.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-1.8, vmax=1.8)
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels([col.replace("_", " ").title() for col in columns], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(summary)))
    ax.set_yticklabels([f"Cluster {int(value)}" for value in summary["trajectory_cluster"]])
    ax.set_title("Sentiment and Emotion Profiles by Template-Trajectory Cluster")
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            ax.text(col_idx, row_idx, f"{matrix[row_idx, col_idx]:.1f}", ha="center", va="center", fontsize=12)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label("Cluster-Level Z-Score")
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_trajectory_cluster_topic_heatmap(df: pd.DataFrame, outpath: Path, top_n: int) -> None:
    rows = []
    for context in ["global", "local"]:
        topic_col = f"{context}_topic"
        if topic_col not in df.columns:
            continue
        temp = df.copy()
        temp[topic_col] = pd.to_numeric(temp[topic_col], errors="coerce")
        temp = temp[temp[topic_col].notna() & temp[topic_col].ne(-1)]
        top_topics = temp[topic_col].astype(int).value_counts().head(top_n).index.tolist()
        for cluster, cluster_df in temp.groupby("trajectory_cluster", observed=True):
            total = max(len(cluster_df), 1)
            for topic in top_topics:
                share = float(cluster_df[topic_col].astype(int).eq(int(topic)).sum() / total)
                rows.append({"context": context, "trajectory_cluster": int(cluster), "topic": int(topic), "share": share})
    topic_df = pd.DataFrame(rows)
    topic_df.to_csv(outpath.with_suffix(".csv"), index=False)
    if topic_df.empty:
        return

    configure_current_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), constrained_layout=True)
    image = None
    for ax, context in zip(axes, ["global", "local"]):
        subset = topic_df[topic_df["context"].eq(context)]
        pivot = subset.pivot_table(index="trajectory_cluster", columns="topic", values="share", aggfunc="sum").fillna(0.0)
        matrix = pivot.to_numpy(dtype=float)
        image = ax.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=0, vmax=max(float(matrix.max()), 0.01))
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([f"T{int(topic)}" for topic in pivot.columns], rotation=35, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([f"Cluster {int(cluster)}" for cluster in pivot.index])
        ax.set_title(f"{context.title()} Topic Shares by Template-Trajectory Cluster")
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
        colorbar.set_label("Topic Share")
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _distribution_distance(left: pd.Series, right: pd.Series, labels: list[Any]) -> float:
    left_probs = left.reindex(labels, fill_value=0.0).to_numpy(dtype=float)
    right_probs = right.reindex(labels, fill_value=0.0).to_numpy(dtype=float)
    left_probs = left_probs / max(left_probs.sum(), 1e-12)
    right_probs = right_probs / max(right_probs.sum(), 1e-12)
    midpoint = 0.5 * (left_probs + right_probs)
    left_kl = np.zeros_like(left_probs)
    right_kl = np.zeros_like(right_probs)
    left_mask = left_probs > 0
    right_mask = right_probs > 0
    left_kl[left_mask] = left_probs[left_mask] * np.log(left_probs[left_mask] / np.maximum(midpoint[left_mask], 1e-12))
    right_kl[right_mask] = right_probs[right_mask] * np.log(right_probs[right_mask] / np.maximum(midpoint[right_mask], 1e-12))
    return float(np.sqrt(0.5 * left_kl.sum() + 0.5 * right_kl.sum()))


def save_trajectory_cluster_pairwise_comparisons(df: pd.DataFrame, outpath: Path) -> None:
    rows: list[dict[str, Any]] = []
    clusters = sorted(df["trajectory_cluster"].dropna().astype(int).unique())
    for context in ["global", "local"]:
        for kind, labels in [("sentiment", NOTEBOOK_SENTIMENT_LABELS), ("emotion", NOTEBOOK_EMOTION_LABELS)]:
            label_col = f"{context}_{kind}_label"
            working = df.copy()
            working[label_col] = dominant_from_columns(working, labels, context)
            distributions = {
                cluster: working.loc[working["trajectory_cluster"].eq(cluster), label_col].value_counts(normalize=True)
                for cluster in clusters
            }
            for i, left_cluster in enumerate(clusters):
                for right_cluster in clusters[i + 1 :]:
                    rows.append(
                        {
                            "context": context,
                            "feature_type": kind,
                            "left_cluster": int(left_cluster),
                            "right_cluster": int(right_cluster),
                            "jensen_shannon_distance": _distribution_distance(
                                distributions[left_cluster],
                                distributions[right_cluster],
                                labels,
                            ),
                        }
                    )

        topic_col = f"{context}_topic"
        if topic_col in df.columns:
            working = df.copy()
            working[topic_col] = pd.to_numeric(working[topic_col], errors="coerce")
            working = working[working[topic_col].notna() & working[topic_col].ne(-1)]
            labels = sorted(working[topic_col].astype(int).unique().tolist())
            distributions = {
                cluster: working.loc[working["trajectory_cluster"].eq(cluster), topic_col].astype(int).value_counts(normalize=True)
                for cluster in clusters
            }
            for i, left_cluster in enumerate(clusters):
                for right_cluster in clusters[i + 1 :]:
                    rows.append(
                        {
                            "context": context,
                            "feature_type": "topic",
                            "left_cluster": int(left_cluster),
                            "right_cluster": int(right_cluster),
                            "jensen_shannon_distance": _distribution_distance(
                                distributions[left_cluster],
                                distributions[right_cluster],
                                labels,
                            ),
                        }
                    )
    pd.DataFrame(rows).to_csv(outpath, index=False)


def _distribution_vector(series: pd.Series, labels: list[Any]) -> np.ndarray:
    counts = series.value_counts(normalize=True)
    values = counts.reindex(labels, fill_value=0.0).to_numpy(dtype=float)
    return values / max(values.sum(), 1e-12)


def _distance_from_vectors(left_probs: np.ndarray, right_probs: np.ndarray) -> float:
    left_probs = np.asarray(left_probs, dtype=float)
    right_probs = np.asarray(right_probs, dtype=float)
    left_probs = left_probs / max(left_probs.sum(), 1e-12)
    right_probs = right_probs / max(right_probs.sum(), 1e-12)
    midpoint = 0.5 * (left_probs + right_probs)
    left_kl = np.zeros_like(left_probs)
    right_kl = np.zeros_like(right_probs)
    left_mask = left_probs > 0
    right_mask = right_probs > 0
    left_kl[left_mask] = left_probs[left_mask] * np.log(left_probs[left_mask] / np.maximum(midpoint[left_mask], 1e-12))
    right_kl[right_mask] = right_probs[right_mask] * np.log(right_probs[right_mask] / np.maximum(midpoint[right_mask], 1e-12))
    return float(np.sqrt(0.5 * left_kl.sum() + 0.5 * right_kl.sum()))


def _template_distribution_profiles(
    df: pd.DataFrame,
    context: str,
    feature_type: str,
    labels: list[Any],
) -> pd.DataFrame:
    if feature_type == "topic":
        value_col = f"{context}_topic"
        if value_col not in df.columns:
            return pd.DataFrame()
        working = df.copy()
        working[value_col] = pd.to_numeric(working[value_col], errors="coerce")
        working = working[working[value_col].notna() & working[value_col].ne(-1)].copy()
        working[value_col] = working[value_col].astype(int)
    else:
        value_col = f"{context}_{feature_type}_label"
        working = df.copy()
        working[value_col] = dominant_from_columns(working, labels, context)
        working = working[working[value_col].isin(labels)].copy()

    rows: list[dict[str, Any]] = []
    if working.empty:
        return pd.DataFrame()
    for (cluster, template), template_df in working.groupby(["trajectory_cluster", "template_final"], observed=True):
        vector = _distribution_vector(template_df[value_col], labels)
        row: dict[str, Any] = {
            "context": context,
            "feature_type": feature_type,
            "trajectory_cluster": int(cluster),
            "template_final": str(template),
            "posts": int(len(template_df)),
        }
        for label, value in zip(labels, vector):
            prefix = "topic" if feature_type == "topic" else feature_type
            row[f"{prefix}_{label}_share"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def save_template_level_intra_cluster_profiles(df: pd.DataFrame, outdir: Path) -> None:
    profile_frames: list[pd.DataFrame] = []
    for context in ["global", "local"]:
        profile_frames.append(_template_distribution_profiles(df, context, "sentiment", NOTEBOOK_SENTIMENT_LABELS))
        profile_frames.append(_template_distribution_profiles(df, context, "emotion", NOTEBOOK_EMOTION_LABELS))
        topic_col = f"{context}_topic"
        if topic_col in df.columns:
            working = df.copy()
            working[topic_col] = pd.to_numeric(working[topic_col], errors="coerce")
            topic_labels = sorted(working.loc[working[topic_col].notna() & working[topic_col].ne(-1), topic_col].astype(int).unique())
            profile_frames.append(_template_distribution_profiles(df, context, "topic", topic_labels))

    profiles = pd.concat([frame for frame in profile_frames if not frame.empty], ignore_index=True) if profile_frames else pd.DataFrame()
    profiles.to_csv(outdir / "template_level_intra_cluster_distribution_profiles.csv", index=False)


def compute_intra_cluster_template_distances(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for context in ["global", "local"]:
        specs: list[tuple[str, list[Any]]] = [
            ("sentiment", NOTEBOOK_SENTIMENT_LABELS),
            ("emotion", NOTEBOOK_EMOTION_LABELS),
        ]
        topic_col = f"{context}_topic"
        if topic_col in df.columns:
            topic_values = pd.to_numeric(df[topic_col], errors="coerce")
            topic_labels = sorted(topic_values[topic_values.notna() & topic_values.ne(-1)].astype(int).unique().tolist())
            specs.append(("topic", topic_labels))

        for feature_type, labels in specs:
            profiles: dict[tuple[int, str], np.ndarray] = {}
            posts_lookup: dict[tuple[int, str], int] = {}
            profile_df = _template_distribution_profiles(df, context, feature_type, labels)
            if profile_df.empty:
                continue
            value_cols = [
                col
                for col in profile_df.columns
                if col not in {"context", "feature_type", "trajectory_cluster", "template_final", "posts"}
            ]
            for row in profile_df.itertuples(index=False):
                key = (int(row.trajectory_cluster), str(row.template_final))
                profiles[key] = np.asarray([getattr(row, col) for col in value_cols], dtype=float)
                posts_lookup[key] = int(row.posts)

            for cluster, cluster_profiles in profile_df.groupby("trajectory_cluster", observed=True):
                templates = cluster_profiles["template_final"].astype(str).tolist()
                distances: list[float] = []
                weighted_distance_sum = 0.0
                weight_sum = 0.0
                for left_idx, left_template in enumerate(templates):
                    left_key = (int(cluster), left_template)
                    for right_template in templates[left_idx + 1 :]:
                        right_key = (int(cluster), right_template)
                        distance = _distance_from_vectors(profiles[left_key], profiles[right_key])
                        weight = min(posts_lookup[left_key], posts_lookup[right_key])
                        distances.append(distance)
                        weighted_distance_sum += distance * max(weight, 1)
                        weight_sum += max(weight, 1)
                        detail_rows.append(
                            {
                                "context": context,
                                "feature_type": feature_type,
                                "trajectory_cluster": int(cluster),
                                "left_template": left_template,
                                "right_template": right_template,
                                "left_posts": posts_lookup[left_key],
                                "right_posts": posts_lookup[right_key],
                                "jensen_shannon_distance": distance,
                            }
                        )

                summary_rows.append(
                    {
                        "context": context,
                        "feature_type": feature_type,
                        "trajectory_cluster": int(cluster),
                        "templates": int(len(templates)),
                        "template_pairs": int(len(distances)),
                        "mean_intra_cluster_distance": float(np.mean(distances)) if distances else np.nan,
                        "median_intra_cluster_distance": float(np.median(distances)) if distances else np.nan,
                        "weighted_mean_intra_cluster_distance": float(weighted_distance_sum / weight_sum) if weight_sum else np.nan,
                        "max_intra_cluster_distance": float(np.max(distances)) if distances else np.nan,
                    }
                )
    return pd.DataFrame(detail_rows), pd.DataFrame(summary_rows)


def save_intra_cluster_distance_heatmap(summary: pd.DataFrame, outpath: Path) -> None:
    if summary.empty:
        return
    configure_current_plot_style()
    summary = summary.copy()
    summary["metric"] = summary["context"] + " " + summary["feature_type"]
    pivot = (
        summary.pivot_table(
            index="trajectory_cluster",
            columns="metric",
            values="mean_intra_cluster_distance",
            aggfunc="mean",
        )
        .sort_index()
        .fillna(0.0)
    )
    fig, ax = plt.subplots(figsize=(18, 8), constrained_layout=True)
    matrix = pivot.to_numpy(dtype=float)
    image = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=max(float(matrix.max()), 0.01))
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(col).title() for col in pivot.columns], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"Cluster {int(cluster)}" for cluster in pivot.index])
    ax.set_title("Intra-Cluster Template-to-Template Distribution Diversity")
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            ax.text(col_idx, row_idx, f"{matrix[row_idx, col_idx]:.2f}", ha="center", va="center", fontsize=12)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label("Mean Jensen-Shannon Distance")
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


TEMPLATE_YEAR_SYSTEM_PROMPT = """You are a meme-history research assistant. Estimate when a meme template first appeared or became recognizable as an internet meme template.

Return JSON only with these fields:
{
  "template_name": string,
  "first_year": integer or null,
  "confidence": "high" | "medium" | "low",
  "reason": string,
  "source_hint": string
}

Use the template name and image together. If the image is an instance of a known template, estimate the first public appearance year of the template, not the year this specific post was made. If unsure, use null and low confidence."""


def select_template_year_representatives(clustered_posts: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    if "image_path" not in clustered_posts.columns:
        return pd.DataFrame(columns=["template_final", "trajectory_cluster", "total_posts", "image_path", "example_key", "example_score"])

    rows: list[dict[str, Any]] = []
    for assignment in assignments.itertuples(index=False):
        template = str(assignment.template_final)
        subset = clustered_posts[clustered_posts["template_final"].eq(template)].copy()
        if subset.empty:
            continue
        subset["score"] = pd.to_numeric(subset.get("score"), errors="coerce").fillna(0.0)
        subset = subset.sort_values(["score", "created_utc"], ascending=[False, True])
        image_path = ""
        example_key = ""
        example_score = 0.0
        for candidate in subset.itertuples(index=False):
            candidate_path = str(getattr(candidate, "image_path", "") or "")
            if candidate_path and load_rgb_image(candidate_path) is not None:
                image_path = candidate_path
                example_key = str(getattr(candidate, "key", "") or "")
                example_score = float(getattr(candidate, "score", 0.0) or 0.0)
                break
        rows.append(
            {
                "template_final": template,
                "trajectory_cluster": int(assignment.trajectory_cluster),
                "total_posts": float(getattr(assignment, "total_posts", np.nan)),
                "image_path": image_path,
                "example_key": example_key,
                "example_score": example_score,
            }
        )
    return pd.DataFrame(rows)


def write_template_year_llm_manifest(representatives: pd.DataFrame, outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with outpath.open("w", encoding="utf-8") as handle:
        for row in representatives.itertuples(index=False):
            image_path = str(row.image_path or "")
            mime_type, _ = mimetypes.guess_type(image_path)
            prompt = (
                f"Template name: {row.template_final}\n"
                f"Question: When was this meme template first created or first recognizable as an internet meme template?"
            )
            handle.write(
                json.dumps(
                    {
                        "template_final": row.template_final,
                        "trajectory_cluster": int(row.trajectory_cluster),
                        "image_path": image_path,
                        "mime_type": mime_type or "image/jpeg",
                        "system_prompt": TEMPLATE_YEAR_SYSTEM_PROMPT,
                        "prompt": prompt,
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )


def load_template_year_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    if not cache_path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with cache_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            template = str(payload.get("template_final", "") or "")
            if template:
                rows[template] = payload
    return rows


def _parse_template_year_response(text: str) -> dict[str, Any]:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        payload = json.loads(match.group(0)) if match else {}
    year = payload.get("first_year")
    try:
        year = int(year) if year is not None else None
    except (TypeError, ValueError):
        year = None
    if year is not None and (year < 1990 or year > 2026):
        year = None
    return {
        "first_year": year,
        "confidence": str(payload.get("confidence", "low") or "low").lower(),
        "reason": str(payload.get("reason", "") or ""),
        "source_hint": str(payload.get("source_hint", "") or ""),
    }


def estimate_template_years_with_openai(
    representatives: pd.DataFrame,
    cache_path: Path,
    model_name: str,
) -> pd.DataFrame:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required when --estimate-template-years is set.")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached = load_template_year_cache(cache_path)
    rows: list[dict[str, Any]] = []
    endpoint = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    with cache_path.open("a", encoding="utf-8") as cache_file:
        for row in representatives.itertuples(index=False):
            template = str(row.template_final)
            if template in cached:
                rows.append(cached[template])
                continue
            image_path = Path(str(row.image_path or ""))
            if not image_path.exists():
                result = {
                    "template_final": template,
                    "trajectory_cluster": int(row.trajectory_cluster),
                    "total_posts": float(row.total_posts),
                    "image_path": str(row.image_path or ""),
                    "first_year": None,
                    "confidence": "low",
                    "reason": "No representative image was available.",
                    "source_hint": "",
                }
            else:
                mime_type, _ = mimetypes.guess_type(str(image_path))
                mime_type = mime_type or "image/jpeg"
                encoded_image = base64.b64encode(image_path.read_bytes()).decode("utf-8")
                data_url = f"data:{mime_type};base64,{encoded_image}"
                payload = {
                    "model": model_name,
                    "temperature": 0,
                    "max_output_tokens": 512,
                    "input": [
                        {
                            "role": "system",
                            "content": [{"type": "input_text", "text": TEMPLATE_YEAR_SYSTEM_PROMPT}],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": (
                                        f"Template name: {template}\n"
                                        "Question: When was this meme template first created or first recognizable as an internet meme template?"
                                    ),
                                },
                                {"type": "input_image", "image_url": data_url, "detail": "low"},
                            ],
                        },
                    ],
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": "template_first_year",
                            "strict": True,
                            "schema": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "template_name": {"type": "string"},
                                    "first_year": {"type": ["integer", "null"]},
                                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                                    "reason": {"type": "string"},
                                    "source_hint": {"type": "string"},
                                },
                                "required": ["template_name", "first_year", "confidence", "reason", "source_hint"],
                            },
                        }
                    },
                }
                response = requests.post(endpoint, headers=headers, json=payload, timeout=90)
                response.raise_for_status()
                response_json = response.json()
                parsed = _parse_template_year_response(str(response_json.get("output_text", "")))
                if not parsed.get("reason"):
                    output_items = response_json.get("output", [])
                    text_chunks = []
                    for item in output_items:
                        for content in item.get("content", []):
                            if content.get("type") in {"output_text", "text"}:
                                text_chunks.append(str(content.get("text", "")))
                    parsed = _parse_template_year_response("\n".join(text_chunks))
                result = {
                    "template_final": template,
                    "trajectory_cluster": int(row.trajectory_cluster),
                    "total_posts": float(row.total_posts),
                    "image_path": str(row.image_path or ""),
                    **parsed,
                }
            cache_file.write(json.dumps(result, ensure_ascii=True) + "\n")
            cache_file.flush()
            cached[template] = result
            rows.append(result)
    return pd.DataFrame(rows)


def summarize_template_years_by_cluster(years: pd.DataFrame, outpath: Path) -> pd.DataFrame:
    columns = [
        "trajectory_cluster",
        "templates",
        "estimated_templates",
        "mean_first_year",
        "median_first_year",
        "min_first_year",
        "max_first_year",
        "mean_template_age_years_as_of_2026",
        "high_or_medium_confidence_share",
    ]
    if years.empty:
        summary = pd.DataFrame(columns=columns)
        summary.to_csv(outpath, index=False)
        return summary
    working = years.copy()
    working["first_year"] = pd.to_numeric(working.get("first_year"), errors="coerce")
    working["has_year"] = working["first_year"].notna()
    working["high_or_medium_confidence"] = working.get("confidence", "").astype(str).str.lower().isin(["high", "medium"])
    summary = (
        working.groupby("trajectory_cluster", observed=True)
        .agg(
            templates=("template_final", "nunique"),
            estimated_templates=("has_year", "sum"),
            mean_first_year=("first_year", "mean"),
            median_first_year=("first_year", "median"),
            min_first_year=("first_year", "min"),
            max_first_year=("first_year", "max"),
            high_or_medium_confidence_share=("high_or_medium_confidence", "mean"),
        )
        .reset_index()
    )
    summary["mean_template_age_years_as_of_2026"] = 2026 - summary["mean_first_year"]
    summary = summary.loc[:, columns]
    summary.to_csv(outpath, index=False)
    return summary


def run_template_trajectory_rq2(df: pd.DataFrame, args: argparse.Namespace, outdir: Path) -> None:
    rq2_dir = ensure_dir(outdir / "rq2_template_trajectory_clusters")
    figures_dir = report_figures_dir(outdir)
    trajectories, curve_long = build_template_trajectory_matrix(
        df,
        freq=args.temporal_freq,
        min_posts=args.min_template_posts,
        normalization=args.trajectory_normalization,
        smooth_window=args.trajectory_smooth_window,
    )
    curve_long.to_csv(rq2_dir / "template_trajectory_curves.csv", index=False)
    if trajectories.empty:
        return
    trajectories.to_csv(rq2_dir / "template_trajectory_matrix_normalized.csv")

    assignments, centroids_long = assign_template_trajectory_clusters(
        trajectories,
        n_clusters=args.trajectory_clusters,
        random_seed=args.random_seed,
    )
    raw_counts = curve_long.groupby("template_final", observed=True)["count"].sum().rename("total_posts").reset_index()
    assignments = assignments.drop(columns=["total_posts"], errors="ignore").merge(raw_counts, on="template_final", how="left")
    assignments.to_csv(rq2_dir / "template_trajectory_cluster_assignments.csv", index=False)
    centroids_long.to_csv(rq2_dir / "template_trajectory_cluster_centroids.csv", index=False)

    clustered_posts = df.merge(assignments[["template_final", "trajectory_cluster"]], on="template_final", how="inner")
    clustered_posts.to_csv(rq2_dir / "template_trajectory_cluster_posts.csv", index=False)

    year_representatives = select_template_year_representatives(clustered_posts, assignments)
    year_representatives_path = rq2_dir / "template_first_year_llm_representatives.csv"
    year_manifest_path = rq2_dir / "template_first_year_llm_manifest.jsonl"
    year_estimates_path = rq2_dir / "template_first_year_llm_estimates.csv"
    year_summary_path = rq2_dir / "template_first_year_cluster_summary.csv"
    year_representatives.to_csv(year_representatives_path, index=False)
    write_template_year_llm_manifest(year_representatives, year_manifest_path)

    year_cache_path = Path(args.template_year_cache)
    if not year_cache_path.is_absolute():
        year_cache_path = ROOT / year_cache_path
    if args.estimate_template_years:
        year_estimates = estimate_template_years_with_openai(
            year_representatives,
            cache_path=year_cache_path,
            model_name=args.template_year_model,
        )
    else:
        year_estimates = pd.DataFrame(load_template_year_cache(year_cache_path).values())
    year_estimates.to_csv(year_estimates_path, index=False)
    summarize_template_years_by_cluster(year_estimates, year_summary_path)

    centroid_path = rq2_dir / "fig_template_trajectory_cluster_centroids.png"
    affect_path = rq2_dir / "fig_template_trajectory_cluster_affect_heatmap.png"
    topic_path = rq2_dir / "fig_template_trajectory_cluster_topic_heatmap.png"
    save_template_trajectory_centroids(centroids_long, centroid_path)
    affect_summary = compute_trajectory_cluster_affect_summary(clustered_posts)
    affect_summary.to_csv(rq2_dir / "template_trajectory_cluster_affect_summary.csv", index=False)
    save_trajectory_cluster_affect_heatmap(affect_summary, affect_path)
    save_trajectory_cluster_topic_heatmap(clustered_posts, topic_path, args.top_topics_per_cluster)
    save_trajectory_cluster_pairwise_comparisons(clustered_posts, rq2_dir / "template_trajectory_cluster_pairwise_distribution_distances.csv")
    save_template_level_intra_cluster_profiles(clustered_posts, rq2_dir)
    intra_details, intra_summary = compute_intra_cluster_template_distances(clustered_posts)
    intra_details.to_csv(rq2_dir / "template_intra_cluster_pairwise_distribution_distances.csv", index=False)
    intra_summary.to_csv(rq2_dir / "template_intra_cluster_distribution_distance_summary.csv", index=False)
    intra_heatmap_path = rq2_dir / "fig_template_intra_cluster_distribution_diversity.png"
    save_intra_cluster_distance_heatmap(intra_summary, intra_heatmap_path)

    for path in [centroid_path, affect_path, topic_path, intra_heatmap_path]:
        copy_report_figure(path, figures_dir)


def build_monthly_features(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    working = df[df["created_utc"].notna()].copy()
    period_freq = "M" if str(freq).upper() == "MS" else str(freq)
    working["month"] = working["created_utc"].dt.to_period(period_freq).dt.to_timestamp()
    working["global_sentiment_balance"] = pd.to_numeric(working["positive_global"], errors="coerce") - pd.to_numeric(
        working["negative_global"], errors="coerce"
    )
    working["local_sentiment_balance"] = pd.to_numeric(working["positive_local"], errors="coerce") - pd.to_numeric(
        working["negative_local"], errors="coerce"
    )
    agg = {
        "posts": ("key", "size"),
        "mean_upvotes": ("score", "mean"),
        "global_sentiment_balance": ("global_sentiment_balance", "mean"),
        "local_sentiment_balance": ("local_sentiment_balance", "mean"),
    }
    for context in ["global", "local"]:
        for label in ["joy", "anger", "sadness", "disgust", "optimism", "fear"]:
            col = f"{label}_{context}"
            if col in working.columns:
                agg[col] = (col, "mean")
    monthly = working.groupby("month", observed=True).agg(**agg).reset_index()
    return monthly.sort_values("month").reset_index(drop=True)


def assign_temporal_clusters(monthly: pd.DataFrame, n_clusters: int, random_seed: int) -> pd.DataFrame:
    feature_cols = [col for col in monthly.columns if col != "month"]
    usable = monthly.dropna(subset=feature_cols).copy()
    if usable.empty:
        monthly["temporal_cluster"] = -1
        return monthly
    k = min(int(n_clusters), len(usable))
    scaler = StandardScaler()
    features = scaler.fit_transform(usable[feature_cols])
    labels = KMeans(n_clusters=k, random_state=random_seed, n_init=20).fit_predict(features)
    usable["temporal_cluster_raw"] = labels
    order = usable.groupby("temporal_cluster_raw")["month"].min().sort_values().index.tolist()
    remap = {raw: idx + 1 for idx, raw in enumerate(order)}
    usable["temporal_cluster"] = usable["temporal_cluster_raw"].map(remap)
    out = monthly.merge(usable[["month", "temporal_cluster"]], on="month", how="left")
    return out


def save_temporal_timeline(monthly: pd.DataFrame, outpath: Path) -> None:
    configure_current_plot_style()
    fig, ax = plt.subplots(figsize=(18, 7), constrained_layout=True)
    clusters = sorted(monthly["temporal_cluster"].dropna().unique())
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(clusters), 1)))
    for color, cluster in zip(colors, clusters):
        subset = monthly[monthly["temporal_cluster"].eq(cluster)]
        ax.scatter(subset["month"], subset["posts"], s=170, color=color, label=f"Cluster {int(cluster)}")
    ax.plot(monthly["month"], monthly["posts"], color="#17202a", alpha=0.35, linewidth=1.8)
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Meme Posts")
    ax.set_title("Temporal Clusters over Meme Posting Volume")
    ax.legend()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_temporal_affect_heatmap(monthly: pd.DataFrame, outpath: Path) -> pd.DataFrame:
    configure_current_plot_style()
    feature_cols = [
        "posts",
        "mean_upvotes",
        "global_sentiment_balance",
        "local_sentiment_balance",
        "joy_global",
        "joy_local",
        "anger_global",
        "anger_local",
        "sadness_global",
        "sadness_local",
        "optimism_global",
        "optimism_local",
    ]
    feature_cols = [col for col in feature_cols if col in monthly.columns]
    summary = monthly.groupby("temporal_cluster", observed=True)[feature_cols].mean().reset_index()
    summary.to_csv(outpath.with_suffix(".csv"), index=False)
    z = zscore_matrix(summary, feature_cols)
    matrix = z[feature_cols].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(18, 8), constrained_layout=True)
    image = ax.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-1.8, vmax=1.8)
    ax.set_xticks(np.arange(len(feature_cols)))
    ax.set_xticklabels([col.replace("_", " ").title() for col in feature_cols], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(summary)))
    ax.set_yticklabels([f"Cluster {int(value)}" for value in summary["temporal_cluster"]])
    ax.set_title("Affective and Engagement Profiles of Temporal Clusters")
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            ax.text(col_idx, row_idx, f"{matrix[row_idx, col_idx]:.1f}", ha="center", va="center", fontsize=12)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label("Cluster-Level Z-Score")
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return summary


def save_temporal_topic_heatmap(df: pd.DataFrame, monthly: pd.DataFrame, outpath: Path, top_n: int) -> None:
    working = df[df["created_utc"].notna()].copy()
    working["month"] = working["created_utc"].dt.to_period("M").dt.to_timestamp()
    working = working.merge(monthly[["month", "temporal_cluster"]], on="month", how="inner")
    rows = []
    for context in ["global", "local"]:
        topic_col = f"{context}_topic"
        if topic_col not in working.columns:
            continue
        temp = working.copy()
        temp[topic_col] = pd.to_numeric(temp[topic_col], errors="coerce")
        temp = temp[temp[topic_col].notna() & temp[topic_col].ne(-1)]
        top_topics = temp[topic_col].astype(int).value_counts().head(top_n).index.tolist()
        for cluster, cluster_df in temp.groupby("temporal_cluster", observed=True):
            total = max(len(cluster_df), 1)
            for topic in top_topics:
                share = float(cluster_df[topic_col].astype(int).eq(int(topic)).sum() / total)
                rows.append({"context": context, "temporal_cluster": int(cluster), "topic": int(topic), "share": share})
    topic_df = pd.DataFrame(rows)
    topic_df.to_csv(outpath.with_suffix(".csv"), index=False)
    if topic_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), constrained_layout=True)
    for ax, context in zip(axes, ["global", "local"]):
        subset = topic_df[topic_df["context"].eq(context)]
        pivot = subset.pivot_table(index="temporal_cluster", columns="topic", values="share", aggfunc="sum").fillna(0.0)
        matrix = pivot.to_numpy(dtype=float)
        image = ax.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=0, vmax=max(float(matrix.max()), 0.01))
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([f"T{int(topic)}" for topic in pivot.columns], rotation=35, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([f"Cluster {int(cluster)}" for cluster in pivot.index])
        ax.set_title(f"{context.title()} Topic Shares by Temporal Cluster")
    colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    colorbar.set_label("Topic Share")
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_temporal_analysis(df: pd.DataFrame, args: argparse.Namespace, outdir: Path) -> None:
    temporal_dir = ensure_dir(outdir / "rq3_rq4_temporal_clusters")
    figures_dir = report_figures_dir(outdir)
    monthly = build_monthly_features(df, args.temporal_freq)
    monthly = assign_temporal_clusters(monthly, args.temporal_clusters, args.random_seed)
    monthly.to_csv(temporal_dir / "monthly_temporal_clusters.csv", index=False)
    timeline_path = temporal_dir / "fig_temporal_cluster_timeline.png"
    affect_path = temporal_dir / "fig_temporal_cluster_affect_heatmap.png"
    topic_path = temporal_dir / "fig_temporal_cluster_topic_heatmap.png"
    save_temporal_timeline(monthly, timeline_path)
    save_temporal_affect_heatmap(monthly, affect_path)
    save_temporal_topic_heatmap(df, monthly, topic_path, args.top_topics_per_cluster)
    for path in [timeline_path, affect_path, topic_path]:
        copy_report_figure(path, figures_dir)


def main() -> None:
    load_dotenv_file(ROOT / ".env")
    args = parse_args()
    outdir = ensure_dir(args.results_dir)
    df = load_analysis_frame(args)
    run_popularity_analysis(df, args, outdir)
    run_template_trajectory_rq2(df, args, outdir)
    run_temporal_analysis(df, args, outdir)
    print(outdir)


if __name__ == "__main__":
    main()
