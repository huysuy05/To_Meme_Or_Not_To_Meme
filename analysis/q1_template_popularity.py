#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (
    configure_plot_style,
    ensure_dir,
    load_analysis_dataframe,
    normalize_template_name,
    write_run_metadata,
    write_summary_markdown,
)


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
    configure_plot_style()
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
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def save_ranked_bar_chart(
    df: pd.DataFrame,
    outpath: Path,
    title: str,
    color: str,
) -> None:
    if df.empty:
        return
    configure_plot_style()
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

    for bar, (_, row) in zip(bars, plot_df.iterrows()):
        ax.text(
            bar.get_width() + max(xmax * 0.015, 0.5),
            bar.get_y() + bar.get_height() / 2,
            f"n={int(row['total_posts'])}, years={row['observed_years_online']:.2f}",
            va="center",
            ha="left",
            fontsize=8.5,
            color="#333333",
        )

    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
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
    configure_plot_style()
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
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


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

    popularity_out = popularity.copy()
    for column in ["first_seen", "last_seen"]:
        popularity_out[column] = popularity_out[column].dt.strftime("%Y-%m-%d %H:%M:%S")
    popularity_out.to_csv(outdir / "template_popularity_all.csv", index=False)
    eligible_out = eligible.copy()
    for column in ["first_seen", "last_seen"]:
        eligible_out[column] = eligible_out[column].dt.strftime("%Y-%m-%d %H:%M:%S")
    eligible_out.to_csv(outdir / "template_popularity_normalized.csv", index=False)
    top_10.to_csv(outdir / "top_10_templates_normalized.csv", index=False)
    next_10.to_csv(outdir / "rank_11_20_templates_normalized.csv", index=False)
    overall_volume.to_csv(outdir / "overall_meme_volume.csv", index=False)
    top_10_curves.to_csv(outdir / "top_10_template_lifecycle_curves.csv", index=False)
    next_10_curves.to_csv(outdir / "rank_11_20_template_lifecycle_curves.csv", index=False)
    top_10_lifecycle.to_csv(outdir / "top_10_template_lifecycle_summary.csv", index=False)
    next_10_lifecycle.to_csv(outdir / "rank_11_20_template_lifecycle_summary.csv", index=False)

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

    top_template = eligible.iloc[0] if not eligible.empty else None
    top_raw_template = (
        popularity.sort_values(["total_posts", "posts_per_observed_year"], ascending=[False, False]).iloc[0]
        if not popularity.empty
        else None
    )
    overall_peak = overall_volume.loc[overall_volume["post_count"].idxmax()] if not overall_volume.empty else None
    tenth_value = float(top_10.iloc[-1]["posts_per_observed_year"]) if len(top_10) == top_k else float("nan")
    eleventh_value = float(next_10.iloc[0]["posts_per_observed_year"]) if not next_10.empty else float("nan")
    top_10_peak_periods = top_10_lifecycle["peak_period"].dropna() if not top_10_lifecycle.empty else pd.Series(dtype="datetime64[ns]")
    top_10_peak_median = (
        top_10_peak_periods.sort_values().iloc[len(top_10_peak_periods) // 2]
        if not top_10_peak_periods.empty
        else None
    )
    top_10_unexpired = (
        int(top_10_lifecycle["expired_at"].isna().sum())
        if not top_10_lifecycle.empty and "expired_at" in top_10_lifecycle.columns
        else 0
    )
    bullets = [
        f"Rows analyzed: {len(df)} across {popularity['template_final'].nunique()} templates.",
        (
            f"Normalization: total posts divided by years observed online in the dataset, "
            f"using each template's first observed post through the dataset end date {dataset_end:%Y-%m-%d}."
        ),
        (
            f"Ranking filter: templates need at least {int(args.min_posts)} posts and "
            f"{int(args.min_observed_days)} observed days; {len(eligible)} templates qualify."
        ),
        (
            f"Top normalized template: {top_template['template_final']} "
            f"({top_template['posts_per_observed_year']:.1f} posts/year, {int(top_template['total_posts'])} posts)."
            if top_template is not None
            else "No templates available after filtering."
        ),
        (
            f"Top raw-count template: {top_raw_template['template_final']} "
            f"({int(top_raw_template['total_posts'])} posts)."
            if top_raw_template is not None
            else "No raw-count leader available."
        ),
        (
            f"Overall posting peak: {int(overall_peak['post_count'])} memes in "
            f"{pd.Timestamp(overall_peak['period_start']):%Y-%m}."
            if overall_peak is not None
            else "Overall posting peak unavailable."
        ),
        (
            f"Shared lifecycle point among the top 10: median peak timing is "
            f"{pd.Timestamp(top_10_peak_median):%Y-%m}, and {top_10_unexpired} of the top 10 do not show "
            "a sustained zero-activity expiration inside the observed window."
            if top_10_peak_median is not None
            else "Top-10 lifecycle summary unavailable."
        ),
        (
            f"Cutoff contrast between ranks {top_k} and {top_k + 1}: "
            f"{tenth_value:.1f} vs {eleventh_value:.1f} posts/year."
            if pd.notna(tenth_value) and pd.notna(eleventh_value)
            else f"Fewer than {top_k * 2} templates available for the two-chart comparison."
        ),
    ]
    write_summary_markdown(outdir / "summary.md", "Q1 Most Popular Meme Templates", bullets)
    write_run_metadata(
        outdir / "run_metadata.json",
        {
            "analysis_parquet": str(Path(args.analysis_parquet).expanduser().resolve()),
            "results_dir": str(outdir),
            "rows": int(len(df)),
            "templates": int(popularity["template_final"].nunique()),
            "eligible_templates": int(len(eligible)),
            "dataset_end": str(dataset_end),
            "top_k": top_k,
            "min_posts": int(args.min_posts),
            "min_observed_days": int(args.min_observed_days),
            "normalization": "posts_per_observed_year = total_posts / years_since_first_observed_post_until_dataset_end",
            "overall_freq": str(args.overall_freq),
            "lifecycle_freq": str(args.lifecycle_freq),
            "low_frac": float(args.low_frac),
            "sustain_periods": int(args.sustain_periods),
            "zero_run_periods": int(args.zero_run_periods),
        },
    )
    print(f"results_dir={outdir}")
    return outdir


def main() -> None:
    run_analysis(parse_args())


if __name__ == "__main__":
    main()
