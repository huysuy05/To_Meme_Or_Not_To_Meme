
from pathlib import Path
import sys
import re
import shutil
from difflib import SequenceMatcher

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from python_scripts.data_loader import DataLoader
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import pandas as pd
loader = DataLoader("/Volumes/huysuy05/ssd_data/meme_or_not/IMGFlip2024_haslabel")


MATCH_THRESHOLD = 0.7
APPEND_THRESHOLD = 1.0
LEXICAL_WEIGHT = 0.3
TOP_K_RERANK = 50
IMAGE_ROOT = REPO_ROOT / "Reddit2024_nolabel" / "images"
VALID_EXTS = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"]


def coerce_template_value(value):
    if value is None:
        return pd.NA
    if isinstance(value, float) and pd.isna(value):
        return pd.NA
    if isinstance(value, (list, tuple, set, np.ndarray)):
        parts = []
        for item in value:
            coerced = coerce_template_value(item)
            if not pd.isna(coerced):
                parts.append(str(coerced))
        if not parts:
            return pd.NA
        return " | ".join(parts)

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return pd.NA
    return text


def normalize_template_name(value: str) -> str:
    value = str(value).lower().replace("-", " ").replace("_", " ")
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def lexical_similarity(left: str, right: str) -> float:
    left_norm = normalize_template_name(left)
    right_norm = normalize_template_name(right)

    if not left_norm or not right_norm:
        return 0.0

    left_tokens = set(left_norm.split())
    right_tokens = set(right_norm.split())
    shared_tokens = left_tokens & right_tokens

    seq_ratio = SequenceMatcher(None, left_norm, right_norm).ratio()
    jaccard = len(shared_tokens) / len(left_tokens | right_tokens) if (left_tokens or right_tokens) else 0.0
    token_overlap = len(shared_tokens) / min(len(left_tokens), len(right_tokens)) if (left_tokens and right_tokens) else 0.0

    return 0.45 * seq_ratio + 0.20 * jaccard + 0.35 * token_overlap


def resolve_reddit_image_path(key: str, image_root: Path) -> Path | None:
    for ext in VALID_EXTS:
        candidate = image_root / f"{key}{ext}"
        if candidate.exists():
            return candidate
    return None


def append_high_confidence_images(rows_df: pd.DataFrame, image_root: Path, ground_truth_root: Path) -> pd.DataFrame:
    copied_rows = []
    unique_rows = rows_df[["key", "matched_template", "match_confidence"]].dropna().drop_duplicates()

    for row in unique_rows.itertuples(index=False):
        source_path = resolve_reddit_image_path(str(row.key), image_root)
        if source_path is None:
            copied_rows.append(
                {
                    "key": row.key,
                    "matched_template": row.matched_template,
                    "match_confidence": row.match_confidence,
                    "status": "missing_source_image",
                    "source_path": pd.NA,
                    "destination_path": pd.NA,
                }
            )
            continue

        destination_dir = ground_truth_root / str(row.matched_template)
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_path = destination_dir / source_path.name

        if destination_path.exists():
            status = "already_exists"
        else:
            shutil.copy2(source_path, destination_path)
            status = "copied"

        copied_rows.append(
            {
                "key": row.key,
                "matched_template": row.matched_template,
                "match_confidence": row.match_confidence,
                "status": status,
                "source_path": str(source_path),
                "destination_path": str(destination_path),
            }
        )

    return pd.DataFrame(copied_rows)


# ============================ IMGFLIP GROUND TRUTH =======================

# Create an array of ImgFlip Templates, which each template is a subfolder
template_arr = np.array(loader.get_template_names())
print(template_arr[:5])

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
ground_truth_embeddings = model.encode(
    template_arr.tolist(),
    convert_to_numpy=True,
    normalize_embeddings=True,
)

print("Shape of ImgFlip templates embeddings: ", ground_truth_embeddings.shape) 


# ==================== LLM ANNOTATED PROCESSING ================================

# Load the LLM-annotated templates
jsonl_path = REPO_ROOT / "data" / "merged_parsed_results.jsonl"

with open(jsonl_path, "r") as f:
    meme_data = [json.loads(line) for line in f]


# print('Number of memes instancesloaded:', len(meme_data))

# Number of non-memes
i = 0
for meme in meme_data:
    try:
        if meme['data']['template'] not in ["NON_MEME", "NO_MEME"]:
            i += 1
    except:
        print(meme['key'])
# print('Number of normal memes:', i)

# Filtering out non-memes
meme_data = [meme for meme in meme_data if isinstance(meme['data'], dict)]
meme_data = [meme for meme in meme_data if meme['data'].get('template') not in ["NON_MEME", "NO_MEME"]]
# print('Number of memes after filtering NON_MEME:', len(meme_data))


# === Turn meme_data from an array of jsons into a Dataframe for better visuals ===
from pandas import json_normalize
meme_data_df = pd.DataFrame(meme_data)
meme_data_df = json_normalize(
    meme_data,
    sep="_",
    meta=["key"],
    record_path=None
)

# Remove the leading "_data"
meme_data_df.columns = meme_data_df.columns.str.replace("data_", "", regex=False)
meme_data_df["template"] = meme_data_df["template"].map(coerce_template_value)


# Get the LLM annotated templates
has_template_mask = meme_data_df["template"].notna() & (meme_data_df["template"] != "NO_TEMPLATE")
has_template = meme_data_df[has_template_mask].copy()
candidate_tpls_np = has_template["template"].to_numpy()
unique_candidate_tpls = pd.Index(pd.Series(candidate_tpls_np).map(str).unique())

candidates_embeddings = model.encode(
    unique_candidate_tpls.tolist(),
    convert_to_numpy=True,
    normalize_embeddings=True,
)

print("Shape of unique candidate template embeddings: ", candidates_embeddings.shape)



# ======================= COSINE SIM BETWEEN 2 EMBEDDINGS ======================

ground_truth_embeddings = ground_truth_embeddings / np.linalg.norm(
    ground_truth_embeddings, axis=1, keepdims=True
)
candidates_embeddings = candidates_embeddings / np.linalg.norm(
    candidates_embeddings, axis=1, keepdims=True
)

# cosine similarity: each candidate vs each ground-truth template
sims = candidates_embeddings @ ground_truth_embeddings.T
print(sims.shape)   # (unique_C, G)

# best ground-truth template for each unique candidate after lexical reranking
top_k = min(TOP_K_RERANK, sims.shape[1])
best_templates = []
best_embedding_scores = []
best_lexical_scores = []
best_hybrid_scores = []

for row_idx, candidate_template in enumerate(unique_candidate_tpls):
    sim_row = sims[row_idx]
    kth = max(0, sim_row.shape[0] - top_k)
    top_idx = np.argpartition(sim_row, kth)[-top_k:]
    top_idx = top_idx[np.argsort(sim_row[top_idx])[::-1]]

    lexical_scores = np.array(
        [lexical_similarity(candidate_template, template_arr[idx]) for idx in top_idx],
        dtype="float32",
    )
    hybrid_scores = sim_row[top_idx] + (LEXICAL_WEIGHT * lexical_scores)

    local_best_idx = int(np.argmax(hybrid_scores))
    matched_idx = int(top_idx[local_best_idx])

    best_templates.append(template_arr[matched_idx])
    best_embedding_scores.append(float(sim_row[matched_idx]))
    best_lexical_scores.append(float(lexical_scores[local_best_idx]))
    best_hybrid_scores.append(float(hybrid_scores[local_best_idx]))

template_match_df = pd.DataFrame(
    {
        "original_template": unique_candidate_tpls,
        "matched_template": best_templates,
        "embedding_score": best_embedding_scores,
        "lexical_score": best_lexical_scores,
        "match_confidence": best_hybrid_scores,
    }
)

has_template["original_template"] = has_template["template"].map(str)
has_template = has_template.merge(template_match_df, on="original_template", how="left")
new_templates = np.where(
    has_template["match_confidence"].to_numpy() >= MATCH_THRESHOLD,
    has_template["matched_template"].to_numpy(),
    has_template["original_template"].to_numpy(),
)
has_template["final_template"] = new_templates
has_template["append_to_ground_truth"] = has_template["match_confidence"] >= APPEND_THRESHOLD

append_candidates_df = has_template[has_template["append_to_ground_truth"]].copy()
append_report_df = append_high_confidence_images(
    rows_df=append_candidates_df,
    image_root=IMAGE_ROOT,
    ground_truth_root=loader.root_dir,
)
append_report_path = REPO_ROOT / "SEED" / "ground_truth_appended_images.csv"
append_report_df.to_csv(append_report_path, index=False)
print(f"Saved append report to {append_report_path}")
if not append_report_df.empty:
    print(append_report_df["status"].value_counts().to_dict())

meme_data_df["original_template"] = meme_data_df["template"]
meme_data_df["matched_template"] = pd.NA
meme_data_df["embedding_score"] = np.nan
meme_data_df["lexical_score"] = np.nan
meme_data_df["match_confidence"] = np.nan
meme_data_df["append_to_ground_truth"] = False

meme_data_df.loc[has_template_mask, "matched_template"] = has_template["matched_template"].to_numpy()
meme_data_df.loc[has_template_mask, "embedding_score"] = has_template["embedding_score"].to_numpy()
meme_data_df.loc[has_template_mask, "lexical_score"] = has_template["lexical_score"].to_numpy()
meme_data_df.loc[has_template_mask, "match_confidence"] = has_template["match_confidence"].to_numpy()
meme_data_df.loc[has_template_mask, "append_to_ground_truth"] = has_template["append_to_ground_truth"].to_numpy()
meme_data_df.loc[has_template_mask, "template"] = new_templates
output_path = REPO_ROOT / "SEED" / "meme_data_reassigned_templates.csv"
meme_data_df.to_csv(output_path, index=False)
print(f"Saved reassigned meme data to {output_path}")

for row in has_template[["original_template", "matched_template", "embedding_score", "lexical_score", "match_confidence"]].drop_duplicates().head(10).itertuples(index=False):
    print(
        f"{row.original_template} -> {row.matched_template} "
        f"(emb={row.embedding_score:.4f}, lex={row.lexical_score:.4f}, hybrid={row.match_confidence:.4f})"
    )
