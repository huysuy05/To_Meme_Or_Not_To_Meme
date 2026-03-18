# annotate_memes.py
import os
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Meme Template Annotation", layout="wide")

CSV_PATH = "annotations/template_label_sheet.csv"           
OUT_PATH = "data/template_label_sheet_annotated.csv"  
IMG_DIR  = "Reddit2024_nolabel/images"

NO = "NO_TEMPLATE"
IMG_EXT = ".jpg"


def find_image_path(img_dir: str, key: str):
    p = Path(img_dir) / f"{key}{IMG_EXT}"
    return str(p) if p.exists() else None


def load_df():
    path = OUT_PATH if os.path.exists(OUT_PATH) else CSV_PATH
    df = pd.read_csv(path, low_memory=False)

    # ensure required columns exist
    if "is_correct_text" not in df.columns:
        df["is_correct_text"] = pd.NA
    if "is_correct_img" not in df.columns:
        df["is_correct_img"] = pd.NA

    # cleanup legacy columns from previous skip implementation
    df = df.drop(columns=["is_skipped", "skip_reason", "is_correct", "true_template"], errors="ignore")

    # make sure key is string
    if "key" in df.columns:
        df["key"] = df["key"].astype(str)

    return df


def save_df(df):
    df.to_csv(OUT_PATH, index=False)


def first_unlabeled_idx(df):
    # choose the first row where BOTH correctness fields are missing
    mask = df["is_correct_text"].isna() & df["is_correct_img"].isna()
    if mask.any():
        return int(mask.idxmax())
    return 0


def find_next_with_image(df, start_idx: int):
    for j in range(start_idx, len(df)):
        # only consider rows that are still fully unlabeled
        row = df.iloc[j]
        if not (pd.isna(row.get("is_correct_text")) and pd.isna(row.get("is_correct_img"))):
            continue
        key = str(row["key"])
        if find_image_path(IMG_DIR, key):
            return j
    return None


def find_next_existing_image_idx(df, start_idx: int):
    for j in range(start_idx, len(df)):
        key = str(df.iloc[j]["key"])
        if find_image_path(IMG_DIR, key):
            return j
    return None


def find_prev_existing_image_idx(df, start_idx: int):
    for j in range(start_idx, -1, -1):
        key = str(df.iloc[j]["key"])
        if find_image_path(IMG_DIR, key):
            return j
    return None


def first_unlabeled_with_image_idx(df):
    idx = find_next_with_image(df, 0)
    return idx if idx is not None else 0


# -----------------------
# App state
# -----------------------
if "df" not in st.session_state:
    st.session_state.df = load_df()
if "i" not in st.session_state:
    st.session_state.i = first_unlabeled_with_image_idx(st.session_state.df)

df = st.session_state.df
i = st.session_state.i

# clamp
i = max(0, min(i, len(df) - 1))
st.session_state.i = i

row = df.iloc[i]
key = str(row["key"])

img_path = find_image_path(IMG_DIR, key)

# if current row has no image, automatically move forward to the next row with image
if img_path is None:
    next_i = find_next_existing_image_idx(df, i + 1)
    if next_i is not None:
        st.session_state.i = next_i
        st.info("Current image not found. Moved to next available image.")
        st.rerun()

# -----------------------
# UI
# -----------------------
st.title("Meme Template Annotation")

c1, c2, c3 = st.columns([2, 1, 1])

with c1:
    st.subheader(f"Item {i+1}/{len(df)} — key: {key}")
    if img_path:
        try:
            st.image(img_path, use_container_width=True)
        except TypeError:
            st.image(img_path, use_column_width=True)
    else:
        st.warning(f"Image not found for key={key}. Check IMG_DIR or filenames.")

with c2:
    st.markdown("### Predicted templates")
    st.write("**Text template:**", row.get("text_template", ""))
    st.write("**Image template:**", row.get("img_template", ""))

    st.markdown("### Text confidence")
    if "text_sim_top1" in df.columns:
        st.write("top1_sim:", row.get("text_sim_top1", ""))
    if "text_margin" in df.columns:
        st.write("margin:", row.get("text_margin", ""))

with c3:
    st.markdown("### Image confidence")
    if "match_method" in df.columns:
        st.write("method:", row.get("match_method", ""))
    if "img_confidence" in df.columns:
        st.write("confidence:", row.get("img_confidence", ""))
    if "phash_dist" in df.columns:
        st.write("phash_dist:", row.get("phash_dist", ""))
    if "clip_top1" in df.columns:
        st.write("clip_top1:", row.get("clip_top1", ""))
    if "clip_margin" in df.columns:
        st.write("clip_margin:", row.get("clip_margin", ""))

st.divider()

# current values
cur_text = row.get("is_correct_text", pd.NA)
cur_img  = row.get("is_correct_img", pd.NA)

b1, b2 = st.columns([1, 1])

with b1:
    st.markdown("### Text correct?")
    colA, colB = st.columns(2)
    if colA.button("👍 Text", use_container_width=True):
        df.at[i, "is_correct_text"] = 1
        save_df(df)
        st.rerun()
    if colB.button("👎 Text", use_container_width=True):
        df.at[i, "is_correct_text"] = 0
        save_df(df)
        st.rerun()
    st.caption(f"Current: {cur_text}")

with b2:
    st.markdown("### Image correct?")
    colA, colB = st.columns(2)
    if colA.button("👍 Image", use_container_width=True):
        df.at[i, "is_correct_img"] = 1
        save_df(df)
        st.rerun()
    if colB.button("👎 Image", use_container_width=True):
        df.at[i, "is_correct_img"] = 0
        save_df(df)
        st.rerun()
    st.caption(f"Current: {cur_img}")

st.divider()

nav1, nav2, nav3, nav4 = st.columns([1, 1, 1, 2])

with nav1:
    if st.button("⬅ Prev", use_container_width=True):
        target = find_prev_existing_image_idx(df, i - 1)
        st.session_state.i = target if target is not None else i
        st.rerun()

with nav2:
    if st.button("Next ➡", use_container_width=True):
        target = find_next_existing_image_idx(df, i + 1)
        st.session_state.i = target if target is not None else i
        st.rerun()

with nav3:
    if st.button("Next Unlabeled", use_container_width=True):
        # jump to next BOTH-NaN row with an available image, searching forward then wrapping
        target = find_next_with_image(df, i + 1)
        if target is None:
            target = find_next_with_image(df, 0)
        if target is None:
            target = i
        st.session_state.i = target
        st.rerun()

with nav4:
    jump = st.number_input("Jump to item #", min_value=1, max_value=len(df), value=i+1)
    if st.button("Jump", use_container_width=True):
        st.session_state.i = int(jump) - 1
        st.rerun()

# progress
done = (~(df["is_correct_text"].isna() | df["is_correct_img"].isna())).sum()
st.caption(f"Progress: {done}/{len(df)} fully labeled. Autosaving to {OUT_PATH}.")
