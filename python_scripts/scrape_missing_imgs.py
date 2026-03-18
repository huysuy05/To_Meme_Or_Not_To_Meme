from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from PIL import Image

M_SUB_PATH = "data/meme_submissions.zst.parquet"
PARSED_PATH = "data/merged_parsed_results.jsonl"
IMG_PATH = Path("Reddit2024_nolabel/images")
MISSING_OUT_PATH = "data/missing_images_from_parsed_results.csv"
FAILED_OUT_PATH = "data/failed_image_downloads.csv"
NO = "NO_TEMPLATE"

# Toggle this if you only want the missing-image report.
DOWNLOAD_MISSING = True
MAX_WORKERS = 16
REQUEST_TIMEOUT = 20

# Optional: if True, skip rows where parsed template is NO_TEMPLATE.
EXCLUDE_NO_TEMPLATE = False


def is_http_url(value) -> bool:
    return isinstance(value, str) and value.startswith(("http://", "https://"))


def is_direct_image_url(value: str) -> bool:
    if not is_http_url(value):
        return False
    path = urlparse(value).path.lower()
    return path.endswith((".jpg", ".jpeg", ".png", ".webp", ".gif"))


def imgur_fallback(value: str) -> str | None:
    """Convert common imgur page links to a direct-image guess."""
    if not is_http_url(value):
        return None
    p = urlparse(value)
    host = p.netloc.lower()
    if "imgur.com" not in host:
        return None
    if host.startswith("i.imgur.com"):
        return value

    parts = [x for x in p.path.split("/") if x]
    if not parts:
        return None

    # Handles urls like /abc123 or /gallery/abc123.
    token = parts[-1]
    token = token.split(".")[0]
    if token:
        return f"https://i.imgur.com/{token}.jpg"
    return None


def pick_source_url(row) -> str | None:
    image_url = row.get("image_url")
    post_url = row.get("url")

    if is_http_url(image_url):
        return image_url
    if is_direct_image_url(post_url):
        return post_url

    guessed = imgur_fallback(post_url) if isinstance(post_url, str) else None
    if guessed:
        return guessed
    return None


def extract_parsed_template(value) -> str | None:
    if isinstance(value, dict):
        tpl = value.get("template")
        return str(tpl).strip() if tpl is not None else None
    return None


def fetch_to_jpg(url: str, out_path: Path) -> tuple[bool, str]:
    try:
        resp = requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0 (compatible; meme-downloader/1.0)"},
        )
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        img.save(out_path, format="JPEG", quality=90)
        return True, "ok"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


m_sub = pd.read_parquet(M_SUB_PATH)
parsed_df = pd.read_json(PARSED_PATH, lines=True)

if "key" not in parsed_df.columns:
    raise KeyError("Expected a 'key' column in merged_parsed_results.jsonl")

base_df = parsed_df[["key"]].copy()
if "data" in parsed_df.columns:
    base_df["parsed_template"] = parsed_df["data"].map(extract_parsed_template)

# Normalize keys to avoid mismatches during merge.
base_df["key"] = base_df["key"].astype(str).str.strip()
m_sub["id"] = m_sub["id"].astype(str).str.strip()

merged = base_df.merge(
    m_sub,
    left_on="key",
    right_on="id",
    how="left",
)

if EXCLUDE_NO_TEMPLATE and "parsed_template" in merged.columns:
    merged = merged[merged["parsed_template"].fillna(NO).ne(NO)].copy()

print("=== MERGE SUMMARY ===")
print(f"Rows: {len(merged)}")
print(f"Columns: {len(merged.columns)}")

IMG_PATH.mkdir(parents=True, exist_ok=True)

# Check if expected file <key>.jpg exists in IMG_PATH.
merged["img_file"] = merged["key"].astype(str) + ".jpg"
merged["img_exists"] = merged["img_file"].map(lambda f: (IMG_PATH / f).exists())
merged["source_url"] = merged.apply(pick_source_url, axis=1)

total = len(merged)
found = int(merged["img_exists"].sum())
missing = int((~merged["img_exists"]).sum())
missing_cols = ["key", "img_file", "source_url"]
if "parsed_template" in merged.columns:
    missing_cols.append("parsed_template")
missing_df = merged.loc[~merged["img_exists"], missing_cols].copy()

print("\n=== IMAGE CHECK (<key>.jpg) ===")
print(f"Total rows: {total}")
print(f"Found: {found} ({found / total:.2%})" if total else "Found: 0")
print(f"Missing: {missing} ({missing / total:.2%})" if total else "Missing: 0")

missing_df.to_csv(MISSING_OUT_PATH, index=False)
print(f"Saved missing list: {MISSING_OUT_PATH}")

if DOWNLOAD_MISSING and not missing_df.empty:
    missing_with_url = missing_df[missing_df["source_url"].notna()].copy()
    print(f"\nAttempting download for missing rows with URL: {len(missing_with_url)}")

    failed_rows = []
    downloaded = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {}
        for row in missing_with_url.itertuples(index=False):
            out_file = IMG_PATH / row.img_file
            futures[ex.submit(fetch_to_jpg, row.source_url, out_file)] = row

        for fut in as_completed(futures):
            row = futures[fut]
            ok, msg = fut.result()
            if ok:
                downloaded += 1
            else:
                failed_rows.append({"key": row.key, "source_url": row.source_url, "error": msg})

    print(f"Downloaded: {downloaded}")
    print(f"Failed: {len(failed_rows)}")

    if failed_rows:
        failed_df = pd.DataFrame(failed_rows)
        failed_df.to_csv(FAILED_OUT_PATH, index=False)
        print(f"Saved failed list: {FAILED_OUT_PATH}")
