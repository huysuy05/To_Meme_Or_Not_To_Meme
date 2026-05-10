#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from common import (
    configure_plot_style,
    ensure_dir,
    load_analysis_dataframe,
    normalize_template_name,
)

Q1_FONT_SIZES = {
    "title": 24,
    "suptitle": 30,
    "axis_label": 19,
    "tick_label": 24,
    "legend": 15,
    "annotation": 15,
    "image_title": 20,
}


def configure_q1_plot_style() -> None:
    configure_plot_style()
    plt.rcParams.update(
        {
            "axes.titlesize": Q1_FONT_SIZES["title"],
            "axes.labelsize": Q1_FONT_SIZES["axis_label"],
            "xtick.labelsize": Q1_FONT_SIZES["tick_label"],
            "ytick.labelsize": Q1_FONT_SIZES["tick_label"],
            "legend.fontsize": Q1_FONT_SIZES["legend"],
        }
    )


def apply_q1_axis_fonts(ax: plt.Axes) -> None:
    ax.title.set_fontsize(Q1_FONT_SIZES["title"])
    ax.xaxis.label.set_fontsize(Q1_FONT_SIZES["axis_label"])
    ax.yaxis.label.set_fontsize(Q1_FONT_SIZES["axis_label"])
    ax.tick_params(axis="both", labelsize=Q1_FONT_SIZES["tick_label"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Identify the most popular meme templates overall using a popularity metric "
            "normalized by how long each template has been observed online in the dataset."
        )
    )
    parser.add_argument("--analysis-parquet", required=True)
    parser.add_argument("--results-dir", default="analysis/results/q1_template_popularity")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-posts", type=int, default=20)
    parser.add_argument("--min-observed-days", type=int, default=90)
    parser.add_argument("--overall-freq", default="M")
    parser.add_argument("--lifecycle-freq", default="W")
    parser.add_argument("--low-frac", type=float, default=0.2)
    parser.add_argument("--sustain-periods", type=int, default=4)
    parser.add_argument("--zero-run-periods", type=int, default=6)
    parser.add_argument("--kym-match-csv", default="data/top_20_reddit_kym_match.csv")
    return parser.parse_args()


def prepare_template_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["template_key"] = df["template_final"].map(normalize_template_name)
    label_lookup = (
        df.groupby(["template_key", "template_final"], observed=True)
        .size()
        .reset_index(name="label_count")
        .sort_values(["template_key", "label_count", "template_final"], ascending=[True, False, True])
        .drop_duplicates(subset=["template_key"], keep="first")
        .rename(columns={"template_final": "display_template"})
    )
    variant_counts = (
        df.groupby("template_key", observed=True)["template_final"]
        .nunique()
        .reset_index(name="label_variants")
    )
    popularity = (
        df.groupby("template_key", observed=True)
        .agg(
            total_posts=("template_final", "size"),
            first_seen=("created_utc", "min"),
            last_seen=("created_utc", "max"),
            avg_score=("score", "mean"),
            median_score=("score", "median"),
        )
        .reset_index()
    )
    popularity = popularity.merge(label_lookup.loc[:, ["template_key", "display_template"]], on="template_key", how="left")
    popularity = popularity.merge(variant_counts, on="template_key", how="left")
    popularity["template_final"] = popularity["display_template"].fillna(popularity["template_key"])
    popularity = popularity.drop(columns=["display_template"])
    template_name_map = popularity.set_index("template_key")["template_final"].to_dict()
    df["template_display"] = df["template_key"].map(template_name_map).fillna(df["template_final"])
    df["label_variants"] = df["template_key"].map(popularity.set_index("template_key")["label_variants"].to_dict()).fillna(1).astype(int)
    return df


def compute_template_popularity(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    dataset_end = pd.Timestamp(df["created_utc"].max())
    popularity = (
        df.groupby("template_key", observed=True)
        .agg(
            template_final=("template_display", "first"),
            total_posts=("template_display", "size"),
            first_seen=("created_utc", "min"),
            last_seen=("created_utc", "max"),
            avg_score=("score", "mean"),
            median_score=("score", "median"),
            label_variants=("label_variants", "max"),
        )
        .reset_index()
    )
    popularity["observed_days_online"] = (
        (dataset_end - popularity["first_seen"]).dt.total_seconds() / 86400.0
    ).clip(lower=1.0) + 1.0
    popularity["observed_years_online"] = popularity["observed_days_online"] / 365.25
    popularity["active_days_in_dataset"] = (
        (popularity["last_seen"] - popularity["first_seen"]).dt.total_seconds() / 86400.0
    ).clip(lower=0.0) + 1.0
    popularity["active_years_in_dataset"] = popularity["active_days_in_dataset"] / 365.25
    popularity["posts_per_observed_year"] = popularity["total_posts"] / popularity["observed_years_online"]
    popularity["posts_per_active_year"] = popularity["total_posts"] / popularity["active_years_in_dataset"]
    popularity = popularity.sort_values(
        ["posts_per_observed_year", "total_posts", "first_seen"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    popularity["overall_normalized_rank"] = range(1, len(popularity) + 1)
    popularity["raw_rank"] = popularity["total_posts"].rank(method="dense", ascending=False).astype(int)
    return popularity, dataset_end


def _format_template_label(template_name: str, rank_value: int) -> str:
    return f"{int(rank_value)}. {template_name.replace('_', ' ')}"


def _load_rgb_image(path: str | Path) -> Image.Image | None:
    try:
        with Image.open(Path(path).expanduser()) as image:
            return ImageOps.exif_transpose(image).convert("RGB")
    except (FileNotFoundError, OSError, UnidentifiedImageError, ValueError):
        return None


def select_representative_images(df: pd.DataFrame, templates_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if "image_path" not in df.columns:
        return pd.DataFrame(rows)

    for template in templates_df.itertuples(index=False):
        subset = df[df["template_key"] == template.template_key].copy()
        subset["score_numeric"] = pd.to_numeric(subset.get("score"), errors="coerce").fillna(0)
        subset = subset.sort_values(["score_numeric", "created_utc"], ascending=[False, True])
        image_path = ""
        for candidate in subset["image_path"].dropna().astype(str):
            if _load_rgb_image(candidate) is not None:
                image_path = candidate
                break
        rows.append(
            {
                "template_key": template.template_key,
                "template_final": template.template_final,
                "normalized_rank": int(template.normalized_rank),
                "total_posts": int(template.total_posts),
                "observed_years_online": float(template.observed_years_online),
                "image_path": image_path,
            }
        )
    return pd.DataFrame(rows)


def save_representative_image_grid(representatives: pd.DataFrame, outpath: Path, title: str) -> None:
    if representatives.empty:
        return
    configure_q1_plot_style()
    plot_df = representatives.sort_values("normalized_rank").copy()
    ncols = 5
    nrows = int(np.ceil(len(plot_df) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(19, 5.1 * nrows))
    axes_list = np.atleast_1d(axes).ravel().tolist()

    for ax, row in zip(axes_list, plot_df.itertuples(index=False)):
        ax.axis("off")
        image = _load_rgb_image(row.image_path) if row.image_path else None
        if image is not None:
            ax.imshow(image)
        ax.set_title(
            (
                f"{_format_template_label(str(row.template_final), int(row.normalized_rank))}\n"
                f"n={int(row.total_posts)}, years={float(row.observed_years_online):.2f}"
            ),
            fontsize=Q1_FONT_SIZES["image_title"],
            ha="center",
            pad=18,
        )

    for ax in axes_list[len(plot_df) :]:
        ax.axis("off")

    fig.suptitle(title, fontsize=Q1_FONT_SIZES["suptitle"], y=0.995)
    fig.tight_layout(h_pad=2.4, w_pad=1.2)
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _normalize_pandas_freq(freq: str) -> str:
    upper = str(freq).upper()
    if upper == "M":
        return "MS"
    return freq


def compute_overall_volume(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    pandas_freq = _normalize_pandas_freq(freq)
    counts = (
        df.set_index("created_utc")
        .resample(pandas_freq)
        .size()
        .rename("post_count")
        .reset_index()
        .rename(columns={"created_utc": "period_start"})
    )
    if counts.empty:
        return counts
    full_range = pd.date_range(
        counts["period_start"].min(),
        counts["period_start"].max(),
        freq=pandas_freq,
    )
    counts = (
        counts.set_index("period_start")
        .reindex(full_range, fill_value=0)
        .rename_axis("period_start")
        .reset_index()
    )
    counts["post_count"] = counts["post_count"].astype(int)
    return counts


def save_overall_volume_plot(volume_df: pd.DataFrame, outpath: Path, freq_label: str) -> None:
    if volume_df.empty:
        return
    configure_q1_plot_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        volume_df["period_start"],
        volume_df["post_count"],
        color="#264653",
        linewidth=2.0,
        marker="o",
        markersize=4.5,
    )
    ax.set_title(f"Total Number of Memes Posted Over Time ({freq_label})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Number of Memes")
    apply_q1_axis_fonts(ax)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_ranked_bar_chart(
    df: pd.DataFrame,
    outpath: Path,
    title: str,
    color: str,
) -> None:
    if df.empty:
        return
    configure_q1_plot_style()
    plot_df = df.copy().sort_values("posts_per_observed_year", ascending=True).reset_index(drop=True)
    labels = [_format_template_label(name, rank) for name, rank in zip(plot_df["template_final"], plot_df["normalized_rank"])]

    fig_height = max(5.5, 0.58 * len(plot_df) + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    bars = ax.barh(labels, plot_df["posts_per_observed_year"], color=color, alpha=0.9)

    xmax = float(plot_df["posts_per_observed_year"].max()) if not plot_df.empty else 0.0
    ax.set_xlim(0, xmax * 1.22 if xmax > 0 else 1.0)
    ax.set_title(title)
    ax.set_xlabel("Posts per Observed Year")
    ax.set_ylabel("Template")
    apply_q1_axis_fonts(ax)

    for bar, (_, row) in zip(bars, plot_df.iterrows()):
        ax.text(
            bar.get_width() + max(xmax * 0.015, 0.5),
            bar.get_y() + bar.get_height() / 2,
            f"n={int(row['total_posts'])}, years={row['observed_years_online']:.2f}",
            va="center",
            ha="left",
            fontsize=Q1_FONT_SIZES["annotation"],
            color="#333333",
        )

    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def compute_template_lifecycle(
    df: pd.DataFrame,
    template_key: str,
    template_name: str,
    freq: str,
    low_frac: float,
    sustain: int,
    zero_run: int,
) -> tuple[pd.DataFrame, dict[str, object]] | None:
    pandas_freq = _normalize_pandas_freq(freq)
    subset = df[df["template_key"] == template_key].copy().sort_values("created_utc")
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

    full_range = pd.date_range(period_counts["period_start"].min(), df["created_utc"].max(), freq=pandas_freq)
    period_counts = (
        period_counts.set_index("period_start")
        .reindex(full_range, fill_value=0)
        .rename_axis("period_start")
        .reset_index()
    )
    period_counts["template_key"] = template_key
    period_counts["template_final"] = template_name
    period_counts["smooth"] = period_counts["count"].rolling(3, min_periods=1).mean()
    max_smooth = float(period_counts["smooth"].max())
    period_counts["smooth_norm"] = period_counts["smooth"] / max(max_smooth, 1.0)

    first_seen = pd.Timestamp(subset["created_utc"].min())
    peak_idx = int(period_counts["smooth"].idxmax())
    peak_period = pd.Timestamp(period_counts.loc[peak_idx, "period_start"])
    peak_value = float(period_counts.loc[peak_idx, "smooth"])
    threshold = peak_value * float(low_frac)

    post_peak = period_counts.loc[peak_idx + 1 :].copy()
    post_peak["low"] = post_peak["smooth"] <= threshold
    post_peak["low_run"] = post_peak["low"].rolling(int(sustain), min_periods=int(sustain)).sum()
    decline_candidates = post_peak.loc[post_peak["low_run"] == int(sustain), "period_start"]
    decline_start = pd.NaT if decline_candidates.empty else pd.Timestamp(decline_candidates.iloc[0])

    post_peak["zero"] = post_peak["count"] == 0
    post_peak["zero_run"] = post_peak["zero"].rolling(int(zero_run), min_periods=int(zero_run)).sum()
    expired_candidates = post_peak.loc[post_peak["zero_run"] == int(zero_run), "period_start"]
    expired_at = pd.NaT if expired_candidates.empty else pd.Timestamp(expired_candidates.iloc[0])

    lifecycle_summary = {
        "template_key": template_key,
        "template_final": template_name,
        "first_seen": first_seen,
        "peak_period": peak_period,
        "peak_value": peak_value,
        "decline_start": decline_start,
        "expired_at": expired_at,
        "total_posts": int(len(subset)),
        "active_periods": int((period_counts["count"] > 0).sum()),
        "observed_periods": int(len(period_counts)),
    }
    return period_counts, lifecycle_summary


def build_lifecycle_outputs(
    df: pd.DataFrame,
    templates_df: pd.DataFrame,
    freq: str,
    low_frac: float,
    sustain: int,
    zero_run: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    curve_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for row in templates_df.itertuples(index=False):
        output = compute_template_lifecycle(
            df=df,
            template_key=row.template_key,
            template_name=row.template_final,
            freq=freq,
            low_frac=low_frac,
            sustain=sustain,
            zero_run=zero_run,
        )
        if output is None:
            continue
        curve_df, summary = output
        curve_df["normalized_rank"] = int(row.normalized_rank)
        curve_rows.append(curve_df)
        summary["normalized_rank"] = int(row.normalized_rank)
        summary["posts_per_observed_year"] = float(row.posts_per_observed_year)
        summary_rows.append(summary)
    curve_out = pd.concat(curve_rows, ignore_index=True) if curve_rows else pd.DataFrame()
    summary_out = pd.DataFrame(summary_rows)
    return curve_out, summary_out


def save_lifecycle_plot(
    curve_df: pd.DataFrame,
    outpath: Path,
    title: str,
    value_col: str,
    ylabel: str,
) -> None:
    if curve_df.empty:
        return
    configure_q1_plot_style()
    fig, ax = plt.subplots(figsize=(13, 7))
    colors = plt.cm.tab20(np.linspace(0, 1, max(curve_df["template_final"].nunique(), 1)))
    for color, (template_name, group) in zip(colors, curve_df.groupby("template_final", observed=True)):
        rank_value = int(group["normalized_rank"].iloc[0])
        ax.plot(
            group["period_start"],
            group[value_col],
            linewidth=1.8,
            label=_format_template_label(template_name, rank_value),
            color=color,
        )
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    apply_q1_axis_fonts(ax)
    ax.legend(loc="upper left", fontsize=Q1_FONT_SIZES["legend"], ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def _split_kym_values(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    parts = re.split(r"[,;/]", str(value))
    return [part.strip() for part in parts if part.strip() and part.strip().lower() != "nan"]


def _safe_kym_year(value: object) -> float:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return np.nan
    match = re.search(r"\d{4}", str(value))
    return float(match.group(0)) if match else np.nan


def _compact_join(values: pd.Series, max_items: int = 80) -> str:
    seen: set[str] = set()
    parts: list[str] = []
    for value in values.dropna().astype(str):
        text = value.strip()
        if not text or text.lower() == "nan" or text in seen:
            continue
        seen.add(text)
        parts.append(text)
        if len(parts) >= max_items:
            break
    return " ".join(parts)


def _build_template_context_summary(df: pd.DataFrame, template_key: str) -> dict[str, object]:
    subset = df[df["template_key"] == template_key].copy()
    if subset.empty:
        return {
            "global_template_text": "",
            "local_reuse_text": "",
            "local_unique_keyword_count": 0,
            "local_unique_keyword_rate": 0.0,
        }

    subset["score_numeric"] = pd.to_numeric(subset.get("score"), errors="coerce").fillna(0)
    ranked = subset.sort_values(["score_numeric", "created_utc"], ascending=[False, True])
    global_text = _compact_join(
        ranked.get("global_context_description", pd.Series(dtype=str)).astype(str)
        + " "
        + ranked.get("global_context_keywords_text", pd.Series(dtype=str)).astype(str),
        max_items=15,
    )
    local_parts = []
    for column in [
        "local_context_user_texts_text",
        "local_context_text_meaning",
        "local_context_instance_specific_image_description",
        "local_context_keywords_text",
        "title",
        "body",
    ]:
        if column in ranked.columns:
            local_parts.append(ranked[column])
    local_text = _compact_join(pd.concat(local_parts, ignore_index=True), max_items=120) if local_parts else ""

    local_keywords: set[str] = set()
    if "local_context_keywords_text" in subset.columns:
        for value in subset["local_context_keywords_text"].dropna().astype(str):
            for token in re.split(r"[|,;/]", value):
                cleaned = token.strip().lower()
                if cleaned:
                    local_keywords.add(cleaned)
    return {
        "global_template_text": global_text,
        "local_reuse_text": local_text,
        "local_unique_keyword_count": int(len(local_keywords)),
        "local_unique_keyword_rate": float(len(local_keywords) / max(len(subset), 1)),
    }


def _cosine_for_pairs(left_texts: list[str], right_texts: list[str]) -> list[float]:
    texts = [*left_texts, *right_texts]
    if not any(str(text).strip() for text in texts):
        return [np.nan for _ in left_texts]
    vectorizer = TfidfVectorizer(stop_words="english", min_df=1, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts)
    left_matrix = matrix[: len(left_texts)]
    right_matrix = matrix[len(left_texts) :]
    sims = cosine_similarity(left_matrix, right_matrix)
    return [float(sims[i, i]) for i in range(len(left_texts))]


def _classify_abstraction(row: pd.Series) -> str:
    kym_local = float(row.get("kym_local_similarity", 0.0))
    global_local = float(row.get("global_local_similarity", 0.0))
    kym_global = float(row.get("kym_global_similarity", 0.0))
    if kym_local >= 0.18:
        return "origin-retaining reuse"
    if global_local >= kym_local and global_local >= 0.12:
        return "structural abstraction"
    if kym_global >= 0.12 and kym_local < 0.12:
        return "origin-to-affordance shift"
    return "high local detachment"


def infer_social_function(row: pd.Series) -> str:
    text = " ".join(
        str(row.get(column, ""))
        for column in ["kym_name", "kym_type", "kym_tags", "kym_about"]
    ).lower()
    if any(term in text for term in ["alignment", "chart", "categor"]):
        return "classification / social sorting"
    if any(term in text for term in ["who would win", "versus", "battle", "compare", "comparison"]):
        return "comparison / hypothetical judgment"
    if any(term in text for term in ["same picture", "difference", "draw 25", "preference", "hotline bling"]):
        return "choice / contrast"
    if any(term in text for term in ["reaction", "image macro", "uncanny", "panik", "kalm"]):
        return "reaction / affect display"
    if any(term in text for term in ["catchphrase", "there is no war", "quote"]):
        return "catchphrase / shared reference"
    if any(term in text for term in ["exploitable", "photoshop", "template"]):
        return "remixable visual format"
    return "pop-culture reference"


CULTURAL_BIOGRAPHY_OVERRIDES = {
    "whowouldwin": {
        "image_origin_type": "internet forum comparison format",
        "recognizable_anchor": "the contrastive setup rather than a single celebrity or character",
        "visual_affordance": "stages a simple contest between two entities",
        "cultural_meaning": "invites playful judgment, absurd comparison, and collective ranking",
        "why_it_travels": "the format is easy to reuse because any two people, objects, groups, or ideas can be inserted into the comparison.",
    },
    "alignmentchart": {
        "image_origin_type": "role-playing game classification grid",
        "recognizable_anchor": "the Dungeons & Dragons moral-alignment matrix",
        "visual_affordance": "sorts examples into a stable two-axis classification system",
        "cultural_meaning": "turns cultural knowledge into social taxonomy and identity signaling",
        "why_it_travels": "the grid can classify almost any fandom, behavior, ideology, or everyday object while preserving a familiar interpretive structure.",
    },
    "drakehotlinebling": {
        "image_origin_type": "music video / celebrity performance",
        "recognizable_anchor": "Drake as a globally recognizable pop artist",
        "visual_affordance": "contrasts rejection in the upper panel with approval in the lower panel",
        "cultural_meaning": "expresses preference, taste, and everyday judgment through a celebrity gesture",
        "why_it_travels": "Drake's recognizability and the two-panel yes/no structure make the template legible even when the inserted topics are unrelated to music.",
    },
    "thereisnowarinbasingse": {
        "image_origin_type": "animated television quote",
        "recognizable_anchor": "Avatar: The Last Airbender and the city of Ba Sing Se",
        "visual_affordance": "uses a familiar phrase as a compact frame for denial or enforced consensus",
        "cultural_meaning": "signals propaganda, institutional denial, or collective refusal to acknowledge an obvious problem",
        "why_it_travels": "the phrase is specific enough to carry fandom meaning but general enough to describe many social situations involving denial.",
    },
    "panikkalmpanik": {
        "image_origin_type": "Meme Man multi-panel reaction format",
        "recognizable_anchor": "Meme Man and the intentionally misspelled affect labels",
        "visual_affordance": "organizes emotional escalation and reversal across repeated panels",
        "cultural_meaning": "turns anxiety, temporary relief, and renewed panic into a shared comic rhythm",
        "why_it_travels": "many everyday situations follow the same panic-calm-panic sequence, making the template broadly reusable.",
    },
    "theyarethesamepicture": {
        "image_origin_type": "television scene from The Office",
        "recognizable_anchor": "Pam's deadpan comparison from a widely known sitcom",
        "visual_affordance": "collapses two supposedly different objects into one judgment",
        "cultural_meaning": "communicates sameness, hypocrisy, or false distinction",
        "why_it_travels": "the template provides a concise way to challenge distinctions that users see as artificial or absurd.",
    },
    "signaturelookofsuperiority": {
        "image_origin_type": "Star Wars reference image",
        "recognizable_anchor": "Count Dooku and Star Wars prequel culture",
        "visual_affordance": "uses facial expression and posture to index smug superiority",
        "cultural_meaning": "performs elitism, condescension, or ironic self-importance",
        "why_it_travels": "the expression can be attached to many situations where someone claims higher status, taste, or knowledge.",
    },
    "grusplan": {
        "image_origin_type": "animated film presentation scene",
        "recognizable_anchor": "Gru from Despicable Me",
        "visual_affordance": "uses a four-panel plan-and-reaction sequence",
        "cultural_meaning": "frames failed planning, unintended consequences, and self-defeating logic",
        "why_it_travels": "the panel sequence maps cleanly onto many narratives where a plan collapses at the final step.",
    },
    "unodraw25cards": {
        "image_origin_type": "card-game image macro",
        "recognizable_anchor": "Uno cards and the familiar penalty of drawing cards",
        "visual_affordance": "poses an ultimatum between doing something undesirable and accepting a penalty",
        "cultural_meaning": "dramatizes avoidance, stubbornness, and refusal",
        "why_it_travels": "the template converts many social obligations into a simple choice-and-consequence joke.",
    },
}


def build_cultural_biography(row: pd.Series) -> dict[str, object]:
    key = str(row.get("reddit_template_key", ""))
    override = CULTURAL_BIOGRAPHY_OVERRIDES.get(key, {})
    about = str(row.get("kym_about", "")).strip()
    origin = str(row.get("kym_origin", "")).strip()
    kym_type = str(row.get("kym_type", "")).strip()
    return {
        "reddit_template_final": row.get("reddit_template_final", ""),
        "kym_name": row.get("kym_name", ""),
        "reddit_rank": row.get("reddit_normalized_rank", np.nan),
        "reddit_total_posts": row.get("reddit_total_posts", np.nan),
        "reddit_posts_per_observed_year": row.get("reddit_posts_per_observed_year", np.nan),
        "kym_year": row.get("kym_year", ""),
        "kym_origin": origin,
        "kym_type": kym_type,
        "social_function": row.get("social_function", ""),
        "image_origin_type": override.get("image_origin_type", origin or kym_type),
        "recognizable_anchor": override.get("recognizable_anchor", "KYM-documented meme image or phrase"),
        "visual_affordance": override.get("visual_affordance", "reusable image-text structure"),
        "cultural_meaning": override.get("cultural_meaning", about[:280]),
        "why_it_travels": override.get(
            "why_it_travels",
            "The template can be detached from its original context and reused as a recognizable frame for new situations.",
        ),
        "kym_about": about,
    }


def save_cultural_biography_outputs(matched: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    biographies = pd.DataFrame([build_cultural_biography(row) for _, row in matched.iterrows()])
    if biographies.empty:
        return biographies

    biographies.to_csv(outdir / "q1_kym_cultural_biographies.csv", index=False)
    lines = ["# Q1 Cultural Biographies of Popular Meme Templates", ""]
    for row in biographies.sort_values("reddit_rank").itertuples(index=False):
        lines.extend(
            [
                f"## {int(row.reddit_rank)}. {row.reddit_template_final} / {row.kym_name}",
                "",
                f"- **Origin:** {row.image_origin_type}",
                f"- **Recognizable anchor:** {row.recognizable_anchor}",
                f"- **Visual affordance:** {row.visual_affordance}",
                f"- **Cultural meaning:** {row.cultural_meaning}",
                f"- **Why it travels:** {row.why_it_travels}",
                f"- **Reddit popularity:** {int(row.reddit_total_posts)} posts; {float(row.reddit_posts_per_observed_year):.2f} posts per observed year.",
                "",
            ]
        )
    (outdir / "q1_kym_cultural_biographies.md").write_text("\n".join(lines))
    return biographies


def save_simple_bar_chart(
    series: pd.Series,
    outpath: Path,
    title: str,
    xlabel: str,
    ylabel: str = "Count",
    color: str = "#2a6f97",
) -> None:
    if series.empty:
        return
    configure_q1_plot_style()
    plot_series = series.sort_values(ascending=True)
    fig_height = max(4.8, 0.5 * len(plot_series) + 1.6)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.barh(plot_series.index.astype(str), plot_series.values, color=color, alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    apply_q1_axis_fonts(ax)
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def save_template_profile_matrix(df: pd.DataFrame, outpath: Path) -> None:
    if df.empty:
        return
    configure_q1_plot_style()
    plot_df = df.sort_values("reddit_normalized_rank").copy()
    rows = plot_df["reddit_template_final"].astype(str).tolist()
    columns = ["Origin", "Format", "Function"]
    cell_text = []
    for row in plot_df.itertuples(index=False):
        cell_text.append(
            [
                str(getattr(row, "kym_origin", "Unknown")) if pd.notna(getattr(row, "kym_origin", "")) else "Unknown",
                str(getattr(row, "kym_type", "Unknown")) if pd.notna(getattr(row, "kym_type", "")) else "Unknown",
                str(getattr(row, "social_function", "Unknown")),
            ]
        )

    fig_height = max(6.0, 0.72 * len(rows) + 2.3)
    fig, ax = plt.subplots(figsize=(18, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        rowLabels=rows,
        colLabels=columns,
        cellLoc="left",
        rowLoc="right",
        loc="center",
        colWidths=[0.28, 0.30, 0.32],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(15)
    table.scale(1, 2.0)
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#d5deed")
        if row_idx == 0:
            cell.set_facecolor("#264653")
            cell.set_text_props(color="white", weight="bold")
        elif col_idx == -1:
            cell.set_facecolor("#edf2f4")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#ffffff" if row_idx % 2 else "#f7f9fc")
    ax.set_title("Cultural Profiles of Popular Meme Templates", fontsize=Q1_FONT_SIZES["title"], pad=24)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_kym_age_chart(df: pd.DataFrame, outpath: Path) -> None:
    plot_df = df.dropna(subset=["kym_year_numeric", "reddit_first_year"]).copy()
    if plot_df.empty:
        return
    configure_q1_plot_style()
    plot_df = plot_df.sort_values("kym_year_numeric", ascending=True)
    y = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(14, max(6.0, 0.72 * len(plot_df) + 2.0)))
    for y_pos, row in enumerate(plot_df.itertuples(index=False)):
        ax.plot(
            [row.kym_year_numeric, row.reddit_first_year],
            [y_pos, y_pos],
            color="#8d99ae",
            linewidth=3.0,
            alpha=0.85,
        )
    ax.scatter(plot_df["kym_year_numeric"], y, color="#2a6f97", s=130, label="Template origin year", zorder=3)
    ax.scatter(plot_df["reddit_first_year"], y, color="#e76f51", s=130, label="First seen in Reddit data", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["reddit_template_final"])
    ax.set_xlabel("Year")
    ax.set_ylabel("Template")
    ax.set_title("Template Origin Year and First Appearance in the Reddit Dataset")
    apply_q1_axis_fonts(ax)
    ax.legend(fontsize=Q1_FONT_SIZES["legend"])
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def save_kym_performance_chart(df: pd.DataFrame, outpath: Path) -> None:
    if df.empty:
        return
    configure_q1_plot_style()
    plot_df = df.sort_values("reddit_posts_per_observed_year", ascending=True)
    fig, ax = plt.subplots(figsize=(18, max(4.8, 0.5 * len(plot_df) + 1.6)))
    ax.barh(
        plot_df["reddit_template_final"],
        plot_df["reddit_posts_per_observed_year"],
        color="#457b9d",
        alpha=0.9,
    )
    ax.set_title("Popularity of Culturally Documented Templates")
    ax.set_xlabel("Posts per Observed Year")
    ax.set_ylabel("Template")
    apply_q1_axis_fonts(ax)
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_cultural_abstraction_analysis(matched: pd.DataFrame, analysis_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in matched.iterrows():
        template_key = str(row.get("reddit_template_key", ""))
        context = _build_template_context_summary(analysis_df, template_key)
        kym_origin_text = " ".join(
            str(row.get(column, ""))
            for column in ["kym_name", "kym_tags", "kym_about", "kym_origin_article", "kym_spread"]
            if pd.notna(row.get(column, ""))
        )
        rows.append(
            {
                **row.to_dict(),
                **context,
                "kym_origin_text": kym_origin_text,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["kym_global_similarity"] = _cosine_for_pairs(
        out["kym_origin_text"].fillna("").astype(str).tolist(),
        out["global_template_text"].fillna("").astype(str).tolist(),
    )
    out["global_local_similarity"] = _cosine_for_pairs(
        out["global_template_text"].fillna("").astype(str).tolist(),
        out["local_reuse_text"].fillna("").astype(str).tolist(),
    )
    out["kym_local_similarity"] = _cosine_for_pairs(
        out["kym_origin_text"].fillna("").astype(str).tolist(),
        out["local_reuse_text"].fillna("").astype(str).tolist(),
    )
    out["origin_detachment"] = 1.0 - out["kym_local_similarity"]
    out["affordance_retention"] = out["global_local_similarity"]
    out["abstraction_gap"] = out["global_local_similarity"] - out["kym_local_similarity"]
    out["abstraction_type"] = out.apply(_classify_abstraction, axis=1)
    return out


def save_cultural_abstraction_scatter(df: pd.DataFrame, outpath: Path) -> None:
    if df.empty:
        return
    configure_q1_plot_style()
    plot_df = df.copy()
    sizes = np.clip(pd.to_numeric(plot_df["reddit_total_posts"], errors="coerce").fillna(0), 50, None)
    sizes = 110 + 520 * (sizes / max(float(sizes.max()), 1.0))
    fig, ax = plt.subplots(figsize=(14.5, 9))
    colors = {
        "origin-retaining reuse": "#2a9d8f",
        "structural abstraction": "#457b9d",
        "origin-to-affordance shift": "#f4a261",
        "high local detachment": "#e76f51",
    }
    for label, group in plot_df.groupby("abstraction_type", observed=True):
        idx = group.index
        ax.scatter(
            group["kym_local_similarity"],
            group["global_local_similarity"],
            s=sizes[plot_df.index.get_indexer(idx)],
            color=colors.get(label, "#8d99ae"),
            alpha=0.82,
            edgecolor="#233142",
            linewidth=1.0,
            label=label,
        )
    for row in plot_df.itertuples(index=False):
        ax.annotate(
            str(row.reddit_template_final),
            (float(row.kym_local_similarity), float(row.global_local_similarity)),
            textcoords="offset points",
            xytext=(7, 5),
            fontsize=12,
            alpha=0.95,
        )
    ax.set_xlabel("Similarity Between Cultural Origin and Local Reddit Reuse")
    ax.set_ylabel("Similarity Between Template Structure and Local Reddit Reuse")
    ax.set_title("Cultural Abstraction in Popular Meme Templates")
    ax.grid(alpha=0.25)
    apply_q1_axis_fonts(ax)
    ax.legend(loc="best", fontsize=Q1_FONT_SIZES["legend"])
    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_abstraction_profile_table(df: pd.DataFrame, outpath: Path) -> None:
    if df.empty:
        return
    configure_q1_plot_style()
    plot_df = df.sort_values("reddit_normalized_rank").copy()
    rows = plot_df["reddit_template_final"].astype(str).tolist()
    columns = ["Abstraction Type", "Origin-Local", "Global-Local", "Keyword Diversity"]
    cell_text = []
    for row in plot_df.itertuples(index=False):
        cell_text.append(
            [
                str(row.abstraction_type),
                f"{float(row.kym_local_similarity):.2f}",
                f"{float(row.global_local_similarity):.2f}",
                f"{float(row.local_unique_keyword_rate):.2f}",
            ]
        )
    fig, ax = plt.subplots(figsize=(16, max(6.0, 0.7 * len(rows) + 2.2)))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        rowLabels=rows,
        colLabels=columns,
        cellLoc="center",
        rowLoc="right",
        loc="center",
        colWidths=[0.38, 0.18, 0.18, 0.22],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(15)
    table.scale(1, 2.0)
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#d5deed")
        if row_idx == 0:
            cell.set_facecolor("#264653")
            cell.set_text_props(color="white", weight="bold")
        elif col_idx == -1:
            cell.set_facecolor("#edf2f4")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#ffffff" if row_idx % 2 else "#f7f9fc")
    ax.set_title("From Cultural Origin to Local Reddit Reuse", fontsize=Q1_FONT_SIZES["title"], pad=24)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_kym_social_science_outputs(match_csv: str | Path, outdir: Path, analysis_df: pd.DataFrame) -> None:
    match_path = Path(match_csv).expanduser()
    if not match_path.exists():
        print(f"Skipping KYM outputs; match file not found: {match_path}")
        return

    raw = pd.read_csv(match_path)
    if raw.empty or "review_decision" not in raw.columns:
        print(f"Skipping KYM outputs; match file has no review_decision column: {match_path}")
        return

    matched = raw[raw["review_decision"].astype(str).str.lower().eq("match")].copy()
    if matched.empty:
        print(f"Skipping KYM outputs; no confirmed matches in: {match_path}")
        return

    matched["kym_year_numeric"] = matched["kym_year"].map(_safe_kym_year)
    matched["reddit_first_seen_dt"] = pd.to_datetime(matched["reddit_first_seen"], errors="coerce")
    matched["reddit_first_year"] = matched["reddit_first_seen_dt"].dt.year.astype(float)
    matched["kym_to_reddit_lag_years"] = matched["reddit_first_year"] - matched["kym_year_numeric"]
    matched["social_function"] = matched.apply(infer_social_function, axis=1)
    biographies = save_cultural_biography_outputs(matched, outdir)
    abstraction_df = build_cultural_abstraction_analysis(matched, analysis_df)

    origin_counts = matched["kym_origin"].fillna("Unknown").replace("", "Unknown").value_counts()
    type_rows = []
    for row in matched.itertuples(index=False):
        for kym_type in _split_kym_values(getattr(row, "kym_type", "")):
            type_rows.append(
                {
                    "reddit_template_final": row.reddit_template_final,
                    "kym_name": row.kym_name,
                    "kym_type_component": kym_type,
                }
            )
    type_df = pd.DataFrame(type_rows)
    type_counts = type_df["kym_type_component"].value_counts() if not type_df.empty else pd.Series(dtype=int)
    function_counts = matched["social_function"].value_counts()

    coverage = pd.DataFrame(
        [
            {
                "top_templates_reviewed": int(len(raw)),
                "confirmed_kym_matches": int(len(matched)),
                "confirmed_match_share": float(len(matched) / max(len(raw), 1)),
                "excluded_non_matches": int((~raw["review_decision"].astype(str).str.lower().eq("match")).sum()),
            }
        ]
    )
    performance = matched[
        [
            "reddit_template_final",
            "reddit_total_posts",
            "reddit_posts_per_observed_year",
            "reddit_avg_score",
            "reddit_median_score",
            "kym_name",
            "kym_year",
            "kym_origin",
            "kym_type",
            "social_function",
        ]
    ].sort_values("reddit_posts_per_observed_year", ascending=False)

    matched.to_csv(outdir / "q1_confirmed_kym_matches.csv", index=False)
    coverage.to_csv(outdir / "q1_kym_match_coverage.csv", index=False)
    origin_counts.rename_axis("kym_origin").reset_index(name="count").to_csv(outdir / "q1_kym_origin_counts.csv", index=False)
    type_df.to_csv(outdir / "q1_kym_type_components.csv", index=False)
    type_counts.rename_axis("kym_type").reset_index(name="count").to_csv(outdir / "q1_kym_type_counts.csv", index=False)
    function_counts.rename_axis("social_function").reset_index(name="count").to_csv(outdir / "q1_kym_social_function_counts.csv", index=False)
    performance.to_csv(outdir / "q1_kym_matched_template_performance.csv", index=False)
    abstraction_df.to_csv(outdir / "q1_cultural_abstraction_metrics.csv", index=False)
    matched[
        [
            "reddit_template_final",
            "kym_name",
            "kym_year",
            "reddit_first_year",
            "kym_to_reddit_lag_years",
            "reddit_observed_years_online",
        ]
    ].to_csv(outdir / "q1_kym_template_age_lag.csv", index=False)

    save_template_profile_matrix(matched, outdir / "fig_q1_template_cultural_profiles.png")
    save_kym_age_chart(matched, outdir / "fig_q1_kym_age_lag.png")
    save_kym_performance_chart(matched, outdir / "fig_q1_kym_matched_popularity.png")
    save_cultural_abstraction_scatter(abstraction_df, outdir / "fig_q1_cultural_abstraction_scatter.png")
    save_abstraction_profile_table(abstraction_df, outdir / "fig_q1_cultural_abstraction_profiles.png")

    summary_lines = [
        "# Q1 KYM Social-Science Summary",
        "",
        f"- Reviewed top templates: {len(raw)}",
        f"- Confirmed KYM matches analyzed: {len(matched)}",
        f"- Confirmed match share: {len(matched) / max(len(raw), 1):.2%}",
        "",
        "## Dominant KYM Origins",
        *[f"- {idx}: {value}" for idx, value in origin_counts.items()],
        "",
        "## Dominant KYM Types",
        *[f"- {idx}: {value}" for idx, value in type_counts.items()],
        "",
        "## Social Functions",
        *[f"- {idx}: {value}" for idx, value in function_counts.items()],
        "",
        "## Cultural Biography Lens",
        (
            "For each confirmed match, `q1_kym_cultural_biographies.md` summarizes the image origin, "
            "recognizable cultural anchor, visual affordance, cultural meaning, and why the template travels."
            if not biographies.empty
            else "No cultural biographies were generated."
        ),
        "",
        "## Cultural Abstraction Lens",
        (
            "The file `q1_cultural_abstraction_metrics.csv` compares KYM origin text, Gemini global template context, "
            "and aggregated local Reddit reuse. The scatter plot visualizes whether templates preserve origin meaning, "
            "retain reusable structure, or detach into locally varied uses."
            if not abstraction_df.empty
            else "No cultural abstraction metrics were generated."
        ),
    ]
    (outdir / "q1_kym_social_science_summary.md").write_text("\n".join(summary_lines))


def run_analysis(args: argparse.Namespace) -> Path:
    outdir = ensure_dir(args.results_dir)
    df = prepare_template_dataframe(load_analysis_dataframe(args.analysis_parquet))
    popularity, dataset_end = compute_template_popularity(df)
    eligible = (
        popularity[
            (popularity["total_posts"] >= int(args.min_posts))
            & (popularity["observed_days_online"] >= float(args.min_observed_days))
        ]
        .copy()
        .reset_index(drop=True)
    )
    eligible["normalized_rank"] = range(1, len(eligible) + 1)

    top_k = max(int(args.top_k), 1)
    top_10 = eligible.head(top_k).copy()
    next_10 = eligible.iloc[top_k : top_k * 2].copy()
    top_10_representatives = select_representative_images(df, top_10)
    next_10_representatives = select_representative_images(df, next_10)
    overall_volume = compute_overall_volume(df, args.overall_freq)
    top_10_curves, top_10_lifecycle = build_lifecycle_outputs(
        df=df,
        templates_df=top_10,
        freq=args.lifecycle_freq,
        low_frac=args.low_frac,
        sustain=args.sustain_periods,
        zero_run=args.zero_run_periods,
    )
    next_10_curves, next_10_lifecycle = build_lifecycle_outputs(
        df=df,
        templates_df=next_10,
        freq=args.lifecycle_freq,
        low_frac=args.low_frac,
        sustain=args.sustain_periods,
        zero_run=args.zero_run_periods,
    )

    save_ranked_bar_chart(
        top_10,
        outdir / "fig_top_10_templates_normalized.png",
        "Top 10 Meme Templates by Popularity per Observed Year",
        "#2a6f97",
    )
    save_ranked_bar_chart(
        next_10,
        outdir / "fig_rank_11_20_templates_normalized.png",
        "Templates Ranked 11-20 by Popularity per Observed Year",
        "#8fb9d4",
    )
    save_representative_image_grid(
        top_10_representatives,
        outdir / "fig_top_10_template_examples.png",
        "Representative Meme Images for Top 10 Templates",
    )
    save_representative_image_grid(
        next_10_representatives,
        outdir / "fig_rank_11_20_template_examples.png",
        "Representative Meme Images for Templates Ranked 11-20",
    )
    save_overall_volume_plot(
        overall_volume,
        outdir / "fig_overall_meme_volume.png",
        "Monthly" if str(args.overall_freq).upper() == "M" else str(args.overall_freq),
    )
    save_lifecycle_plot(
        top_10_curves,
        outdir / "fig_top_10_template_lifecycles.png",
        "Lifecycle Curves for Top 10 Templates",
        "smooth",
        f"Smoothed Posts per {args.lifecycle_freq}",
    )
    save_lifecycle_plot(
        next_10_curves,
        outdir / "fig_rank_11_20_template_lifecycles.png",
        "Lifecycle Curves for Templates Ranked 11-20",
        "smooth",
        f"Smoothed Posts per {args.lifecycle_freq}",
    )
    save_lifecycle_plot(
        top_10_curves,
        outdir / "fig_top_10_template_lifecycles_normalized.png",
        "Normalized Lifecycle Curves for Top 10 Templates",
        "smooth_norm",
        "Relative Lifecycle Intensity",
    )
    save_lifecycle_plot(
        next_10_curves,
        outdir / "fig_rank_11_20_template_lifecycles_normalized.png",
        "Normalized Lifecycle Curves for Templates Ranked 11-20",
        "smooth_norm",
        "Relative Lifecycle Intensity",
    )
    save_kym_social_science_outputs(args.kym_match_csv, outdir, df)

    print(f"results_dir={outdir}")
    return outdir


def main() -> None:
    run_analysis(parse_args())


if __name__ == "__main__":
    main()
