#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from common import load_analysis_dataframe  # noqa: E402
from q1_template_popularity import compute_template_popularity, prepare_template_dataframe  # noqa: E402


DEFAULT_ANALYSIS_PARQUET = PROJECT_ROOT / "data/template_first_analysis_table.parquet"
DEFAULT_KYM_CSV = Path("/Volumes/huysuy05/ssd_data/KYM/kym_memes.csv")
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "data/top_20_reddit_kym_match.csv"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
KYM_COLUMNS = ["name", "year", "type", "origin", "tags", "about", "origin_article", "spread"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review top Reddit template to KYM metadata matches.")
    parser.add_argument("--analysis-parquet", default=str(DEFAULT_ANALYSIS_PARQUET))
    parser.add_argument("--kym-csv", default=str(DEFAULT_KYM_CSV))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--candidate-count", type=int, default=6)
    parser.add_argument("--min-posts", type=int, default=20)
    parser.add_argument("--min-observed-days", type=int, default=90)
    parser.add_argument(
        "--kym-image-root",
        default=None,
        help="Optional folder containing KYM images named similarly to KYM entry names.",
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    text = str(value).lower()
    text = re.sub(r"[_-]+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_file_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def load_rgb_image(path: str | Path | None) -> Image.Image | None:
    if path is None or not str(path).strip():
        return None
    try:
        with Image.open(Path(path).expanduser()) as image:
            return ImageOps.exif_transpose(image).convert("RGB")
    except (FileNotFoundError, OSError, UnidentifiedImageError, ValueError):
        return None


@st.cache_data(show_spinner="Loading Q1 top templates...")
def load_top_templates(analysis_parquet: str, top_n: int, min_posts: int, min_observed_days: int) -> pd.DataFrame:
    df = prepare_template_dataframe(load_analysis_dataframe(analysis_parquet))
    popularity, _ = compute_template_popularity(df)
    eligible = (
        popularity[
            (popularity["total_posts"] >= int(min_posts))
            & (popularity["observed_days_online"] >= float(min_observed_days))
        ]
        .copy()
        .reset_index(drop=True)
    )
    eligible["normalized_rank"] = range(1, len(eligible) + 1)
    top_templates = eligible.head(int(top_n)).copy()

    representative_rows: list[dict[str, Any]] = []
    for template in top_templates.itertuples(index=False):
        subset = df[df["template_key"] == template.template_key].copy()
        subset["score_numeric"] = pd.to_numeric(subset.get("score"), errors="coerce").fillna(0)
        subset = subset.sort_values(["score_numeric", "created_utc"], ascending=[False, True])
        image_path = ""
        for candidate in subset.get("image_path", pd.Series(dtype=str)).dropna().astype(str):
            if load_rgb_image(candidate) is not None:
                image_path = candidate
                break
        representative_rows.append(
            {
                "reddit_template_key": template.template_key,
                "reddit_template_final": template.template_final,
                "reddit_normalized_rank": int(template.normalized_rank),
                "reddit_total_posts": int(template.total_posts),
                "reddit_first_seen": template.first_seen,
                "reddit_last_seen": template.last_seen,
                "reddit_avg_score": float(template.avg_score),
                "reddit_median_score": float(template.median_score),
                "reddit_observed_days_online": float(template.observed_days_online),
                "reddit_observed_years_online": float(template.observed_years_online),
                "reddit_posts_per_observed_year": float(template.posts_per_observed_year),
                "reddit_representative_image_path": image_path,
            }
        )
    return pd.DataFrame(representative_rows)


@st.cache_data(show_spinner="Loading KYM metadata...")
def load_kym_metadata(kym_csv: str) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(
                kym_csv,
                encoding=encoding,
                usecols=KYM_COLUMNS,
                dtype=str,
                low_memory=False,
            )
            break
        except UnicodeDecodeError:
            continue
    else:
        raise UnicodeError(f"Could not read KYM CSV: {kym_csv}")

    df = df.dropna(subset=["name"]).drop_duplicates(subset=["name"]).reset_index(drop=True)
    df["kym_row_id"] = df.index
    df["kym_name_norm"] = df["name"].map(normalize_text)
    df["kym_search_text"] = (
        df["name"].fillna("") + " " + df.get("tags", "").fillna("")
    ).map(normalize_text)
    return df


def score_match(query: str, candidate: str) -> float:
    query_norm = normalize_text(query)
    candidate_norm = normalize_text(candidate)
    if not query_norm or not candidate_norm:
        return 0.0

    query_tokens = set(query_norm.split())
    candidate_tokens = set(candidate_norm.split())
    token_score = 100.0 * len(query_tokens & candidate_tokens) / max(len(query_tokens | candidate_tokens), 1)
    contains_score = 0.0
    if query_norm in candidate_norm or candidate_norm in query_norm:
        contains_score = 100.0 * min(len(query_norm), len(candidate_norm)) / max(len(query_norm), len(candidate_norm))
    ratio_score = 100.0 * difflib.SequenceMatcher(None, query_norm, candidate_norm).ratio()
    return max(token_score, contains_score, ratio_score)


@st.cache_data(show_spinner="Building KYM candidates...")
def build_candidates(top_templates: pd.DataFrame, kym: pd.DataFrame, candidate_count: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for template in top_templates.itertuples(index=False):
        scored = []
        query = str(template.reddit_template_final)
        for kym_row in kym.itertuples(index=False):
            score = max(
                score_match(query, kym_row.name),
                score_match(query, kym_row.kym_search_text),
            )
            scored.append((score, int(kym_row.kym_row_id)))
        scored = sorted(scored, reverse=True)[: int(candidate_count)]
        for rank, (score, kym_row_id) in enumerate(scored, start=1):
            rows.append(
                {
                    "reddit_template_key": template.reddit_template_key,
                    "candidate_rank": rank,
                    "match_score": float(score),
                    "kym_row_id": int(kym_row_id),
                }
            )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Indexing optional KYM images...")
def build_kym_image_index(kym_image_root: str | None) -> dict[str, str]:
    if not kym_image_root:
        return {}
    root = Path(kym_image_root).expanduser()
    if not root.exists():
        return {}
    index: dict[str, str] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            index.setdefault(normalize_file_key(path.stem), str(path))
    return index


def find_kym_image_path(kym_name: str, image_index: dict[str, str]) -> str:
    if not image_index:
        return ""
    key = normalize_file_key(kym_name)
    if key in image_index:
        return image_index[key]
    for image_key, path in image_index.items():
        if key and (key in image_key or image_key in key):
            return path
    return ""


def load_existing_output(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def write_decision(output_path: Path, row: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing_output(output_path)
    if not existing.empty and "reddit_template_key" in existing.columns:
        existing = existing[existing["reddit_template_key"] != row["reddit_template_key"]].copy()
    out = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    out = out.sort_values("reddit_normalized_rank").reset_index(drop=True)
    out.to_csv(output_path, index=False)


def make_output_row(
    reddit_row: pd.Series,
    kym_row: pd.Series | None,
    decision: str,
    match_score: float | None,
    notes: str,
    kym_image_path: str,
) -> dict[str, Any]:
    row = reddit_row.to_dict()
    row["review_decision"] = decision
    row["review_match_score"] = match_score
    row["review_notes"] = notes
    row["kym_image_path"] = kym_image_path
    if kym_row is not None:
        for column, value in kym_row.to_dict().items():
            row[f"kym_{column}"] = value
    else:
        row["kym_row_id"] = None
    return row


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_csv).expanduser()

    st.set_page_config(page_title="Reddit-KYM Template Match Review", layout="wide")
    st.title("Reddit-KYM Template Match Review")

    top_templates = load_top_templates(
        args.analysis_parquet,
        args.top_n,
        args.min_posts,
        args.min_observed_days,
    )
    kym = load_kym_metadata(args.kym_csv)
    candidates = build_candidates(top_templates, kym, args.candidate_count)
    kym_image_index = build_kym_image_index(args.kym_image_root)
    existing = load_existing_output(output_path)

    if "idx" not in st.session_state:
        st.session_state.idx = 0

    reviewed_keys = set(existing["reddit_template_key"].astype(str)) if not existing.empty and "reddit_template_key" in existing else set()
    st.caption(f"Output: `{output_path}`")
    st.progress(len(reviewed_keys) / max(len(top_templates), 1), text=f"{len(reviewed_keys)} / {len(top_templates)} reviewed")

    idx = min(max(int(st.session_state.idx), 0), len(top_templates) - 1)
    reddit_row = top_templates.iloc[idx]
    template_key = reddit_row["reddit_template_key"]
    candidate_rows = candidates[candidates["reddit_template_key"] == template_key].copy()
    candidate_rows = candidate_rows.merge(kym, on="kym_row_id", how="left")

    st.subheader(
        f"{int(reddit_row['reddit_normalized_rank'])}. {reddit_row['reddit_template_final']}"
    )

    left, right = st.columns([1.1, 1.4], gap="large")
    with left:
        st.markdown("#### Reddit template")
        image = load_rgb_image(reddit_row["reddit_representative_image_path"])
        if image is not None:
            st.image(image, use_container_width=True)
        else:
            st.warning("No readable Reddit representative image found.")
        st.write(
            {
                "posts": int(reddit_row["reddit_total_posts"]),
                "years": round(float(reddit_row["reddit_observed_years_online"]), 2),
                "posts_per_observed_year": round(float(reddit_row["reddit_posts_per_observed_year"]), 2),
            }
        )

    candidate_labels = [
        f"{int(row.candidate_rank)}. {row.name} | score={float(row.match_score):.1f}"
        for row in candidate_rows.itertuples(index=False)
    ]
    with right:
        st.markdown("#### KYM candidates")
        selected_label = st.radio("Select candidate", candidate_labels, index=0)
        selected_pos = candidate_labels.index(selected_label)
        candidate = candidate_rows.iloc[selected_pos]
        kym_image_path = find_kym_image_path(str(candidate["name"]), kym_image_index)
        kym_image = load_rgb_image(kym_image_path)
        if kym_image is not None:
            st.image(kym_image, caption=str(candidate["name"]), use_container_width=True)
        else:
            st.info("No local KYM image found. Pass `--kym-image-root` if you have KYM images.")
        st.markdown(f"**KYM name:** {candidate['name']}")
        st.markdown(f"**Year:** {candidate.get('year', '')}")
        st.markdown(f"**Type:** {candidate.get('type', '')}")
        st.markdown(f"**Origin:** {candidate.get('origin', '')}")
        st.markdown(f"**Tags:** {candidate.get('tags', '')}")
        st.text_area("About", str(candidate.get("about", "")), height=180, disabled=True)

    notes = st.text_area("Reviewer notes", key=f"notes_{template_key}")
    controls = st.columns([1, 1, 1, 1, 2])
    with controls[0]:
        if st.button("Previous", disabled=idx == 0):
            st.session_state.idx = max(0, idx - 1)
            st.rerun()
    with controls[1]:
        if st.button("Match"):
            output_row = make_output_row(
                reddit_row=reddit_row,
                kym_row=candidate,
                decision="match",
                match_score=float(candidate["match_score"]),
                notes=notes,
                kym_image_path=kym_image_path,
            )
            write_decision(output_path, output_row)
            st.session_state.idx = min(len(top_templates) - 1, idx + 1)
            st.rerun()
    with controls[2]:
        if st.button("Not match"):
            output_row = make_output_row(
                reddit_row=reddit_row,
                kym_row=candidate,
                decision="not_match",
                match_score=float(candidate["match_score"]),
                notes=notes,
                kym_image_path=kym_image_path,
            )
            write_decision(output_path, output_row)
            st.session_state.idx = min(len(top_templates) - 1, idx + 1)
            st.rerun()
    with controls[3]:
        if st.button("No KYM entry"):
            output_row = make_output_row(
                reddit_row=reddit_row,
                kym_row=None,
                decision="no_kym_entry",
                match_score=None,
                notes=notes,
                kym_image_path="",
            )
            write_decision(output_path, output_row)
            st.session_state.idx = min(len(top_templates) - 1, idx + 1)
            st.rerun()
    with controls[4]:
        if st.button("Next", disabled=idx >= len(top_templates) - 1):
            st.session_state.idx = min(len(top_templates) - 1, idx + 1)
            st.rerun()

    st.markdown("#### Saved decisions")
    existing = load_existing_output(output_path)
    if existing.empty:
        st.info("No decisions saved yet.")
    else:
        st.dataframe(
            existing[
                [
                    "reddit_normalized_rank",
                    "reddit_template_final",
                    "review_decision",
                    "kym_name",
                    "review_match_score",
                ]
                if "kym_name" in existing.columns
                else existing.columns
            ],
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
