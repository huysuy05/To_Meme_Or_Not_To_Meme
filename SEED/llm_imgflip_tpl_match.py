#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError as exc:
    genai = None
    genai_types = None
    GENAI_IMPORT_ERROR = exc
else:
    GENAI_IMPORT_ERROR = None

from meme_research_eval_utils import (
    compute_metrics_with_rejection,
    finalize_run_timing,
    load_rgb_image,
    load_split_rows,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
NO_TEMPLATE_LABEL = "NO_TEMPLATE"
NON_MEME_LABEL = "NON_MEME"
DEFAULT_MODEL_ID = "gemini-2.5-flash-lite"
DEFAULT_MAX_OUTPUT_TOKENS = 2048
DEFAULT_RUN_ROOT = SCRIPT_DIR / "runs" / "llm_imgflip_tpl_match"
DEFAULT_TRAIN_PARQUET = SCRIPT_DIR / "splits" / "imgflip_80_20" / "train.parquet"
DEFAULT_TEST_PARQUET = SCRIPT_DIR / "splits" / "imgflip_80_20" / "test.parquet"
API_KEY_PATTERN = re.compile(r'api_key\s*=\s*["\']([^"\']+)["\']')


SYSTEM_TEXT = """You are an AI Meme Analyzer for closed-set ImgFlip template classification.

Your primary task is to assign the image to exactly one known meme template from the provided template list.
One or more labeled in-context examples may be provided before the query image. Those examples are only for the
template classification decision. Do not copy their wording or descriptions into the final answer.

Return ONLY valid JSON with this schema:
{
  "template": "string",
  "global_context_description": "string",
  "local_context": {
    "user_texts": ["string"],
    "text_meaning": "string",
    "instance_specific_image_description": "string"
  },
  "global_context_keywords": ["string"],
  "local_context_keywords": ["string"]
}

Rules:
1. `template` must be exactly one label from the supplied template list.
2. If the query image does not match any provided template with sufficient confidence, return `NO_TEMPLATE`.
3. If the query image is clearly not a meme, return `NON_MEME`.
4. Even when the answer is `NO_TEMPLATE`, still return the full JSON schema with best-effort descriptions.
5. The dataset does not provide an external post title. Infer meaning only from the image and any overlaid meme text.
6. `global_context_description` should describe the reusable template itself, not the instance-specific joke.
7. `user_texts` should contain the visible meme text added to this specific image.
8. `text_meaning` should explain the joke or claim made by the instance using only the image content and visible text.
9. `instance_specific_image_description` should be empty when the main visual is already the template itself.
10. `global_context_keywords` and `local_context_keywords` should each contain 3-7 short keywords.
"""


@dataclass(frozen=True)
class InContextExample:
    image_path: str
    template: str


@dataclass
class RunConfig:
    train_parquet: Path
    test_parquet: Path
    output_dir: Path
    model_id: str = DEFAULT_MODEL_ID
    icl_shots: int = 1
    random_seed: int = 42
    max_test_samples: int | None = None
    max_templates: int | None = None
    request_timeout_retries: int = 5
    retry_sleep_seconds: float = 5.0
    temperature: float = 0.0
    top_p: float = 0.95
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Gemini as a closed-set ImgFlip meme-template matcher, optionally with "
            "few-shot in-context examples, while still returning template/local/global descriptions."
        )
    )
    parser.add_argument("--train-parquet", default=str(DEFAULT_TRAIN_PARQUET))
    parser.add_argument("--test-parquet", default=str(DEFAULT_TEST_PARQUET))
    parser.add_argument("--output-dir", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=(
            "Gemini model to use. Default is the free-tier `gemini-2.0-flash-lite`; "
            "override if you want another compatible model."
        ),
    )
    parser.add_argument("--icl-shots", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--max-templates", type=int, default=None)
    parser.add_argument("--request-timeout-retries", type=int, default=5)
    parser.add_argument("--retry-sleep-seconds", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    args = parser.parse_args()
    return RunConfig(
        train_parquet=Path(args.train_parquet).expanduser().resolve(),
        test_parquet=Path(args.test_parquet).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        model_id=args.model_id,
        icl_shots=args.icl_shots,
        random_seed=args.random_seed,
        max_test_samples=args.max_test_samples,
        max_templates=args.max_templates,
        request_timeout_retries=args.request_timeout_retries,
        retry_sleep_seconds=args.retry_sleep_seconds,
        temperature=args.temperature,
        top_p=args.top_p,
        max_output_tokens=args.max_output_tokens,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def discover_api_key() -> str:
    for env_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        value = os.getenv(env_name)
        if value:
            return value

    candidate_files = [
        PROJECT_ROOT / "Meme_gemini" / "gemini_memes_batched.py",
        PROJECT_ROOT / "Meme_gemini" / "gemini_meme_only_template.py",
        PROJECT_ROOT / "Meme_gemini" / "gemini_memes.py",
    ]
    for path in candidate_files:
        if not path.exists():
            continue
        match = API_KEY_PATTERN.search(path.read_text(encoding="utf-8"))
        if match:
            return match.group(1)

    raise RuntimeError(
        "Could not find a Gemini API key. Set GEMINI_API_KEY/GOOGLE_API_KEY or keep the "
        "existing Gemini scripts available for fallback discovery."
    )


def maybe_limit_templates(df: pd.DataFrame, max_templates: int | None) -> pd.DataFrame:
    if max_templates is None:
        return df.reset_index(drop=True)
    keep_templates = sorted(df["template"].unique().tolist())[:max_templates]
    return df[df["template"].isin(keep_templates)].reset_index(drop=True)


def filter_readable_rows(df: pd.DataFrame, limit: int | None = None) -> pd.DataFrame:
    keep_rows: list[int] = []
    for idx, image_path in enumerate(df["image_path"].tolist()):
        if load_rgb_image(image_path) is not None:
            keep_rows.append(idx)
            if limit is not None and len(keep_rows) >= limit:
                break
    return df.iloc[keep_rows].reset_index(drop=True)


def sample_icl_examples(train_df: pd.DataFrame, shots: int, seed: int) -> list[InContextExample]:
    if shots <= 0:
        return []

    rng = random.Random(seed)
    examples: list[InContextExample] = []
    by_template = {template: group.reset_index(drop=True) for template, group in train_df.groupby("template")}
    template_names = sorted(by_template)
    rng.shuffle(template_names)

    for template in template_names:
        if len(examples) >= shots:
            break
        group = by_template[template]
        indices = list(range(len(group)))
        rng.shuffle(indices)
        for idx in indices:
            image_path = group.iloc[idx]["image_path"]
            if load_rgb_image(image_path) is None:
                continue
            examples.append(InContextExample(image_path=image_path, template=template))
            break

    if len(examples) < shots:
        raise ValueError(f"Requested {shots} ICL shots, but only found {len(examples)} readable examples.")
    return examples


def build_template_list_text(template_names: list[str]) -> str:
    return "; ".join(template_names)


def image_part_from_path(image_path: str | Path) -> Any:
    if genai_types is None:
        raise RuntimeError(
            "Missing Gemini SDK dependency. Install it with `python3 -m pip install google-genai`. "
            "The placeholder `google` package is not the correct dependency."
        ) from GENAI_IMPORT_ERROR
    image_path = Path(image_path)
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if mime_type is None:
        mime_type = "image/jpeg"
    data = image_path.read_bytes()
    return genai_types.Part.from_bytes(data=data, mime_type=mime_type)


def build_contents(
    query_image_path: str,
    template_list_text: str,
    icl_examples: list[InContextExample],
) -> list[Any]:
    contents: list[Any] = [
        (
            "Task: classify the query image into one known meme template from the supplied template list. "
            "Use any in-context examples only for the template decision. Then return the full JSON schema "
            "for the query image."
        )
    ]

    for shot_idx, example in enumerate(icl_examples, start=1):
        contents.extend(
            [
                f"In-context example {shot_idx}: labeled reference image for template classification only.",
                image_part_from_path(example.image_path),
                json.dumps({"template": example.template}),
            ]
        )

    contents.extend(
        [
            "Query image:",
            image_part_from_path(query_image_path),
            "Known template list:",
            template_list_text,
            (
                "Reminder: return valid JSON only. For the query image, set `template` to an exact label from "
                "the known template list, or `NO_TEMPLATE`, or `NON_MEME`."
            ),
        ]
    )
    return contents


def parse_response_json(response_text: str) -> dict[str, Any]:
    cleaned = response_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json"):].strip()
    if cleaned.startswith("```"):
        cleaned = cleaned[len("```"):].strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()
    return json.loads(cleaned)


def coerce_template_value(template_value: Any, known_templates: set[str]) -> tuple[str, bool]:
    if isinstance(template_value, list):
        if len(template_value) == 1:
            template_value = template_value[0]
        else:
            return NO_TEMPLATE_LABEL, False

    if not isinstance(template_value, str):
        return NO_TEMPLATE_LABEL, False

    template_value = template_value.strip()
    if template_value in known_templates:
        return template_value, True
    if template_value in {NO_TEMPLATE_LABEL, NON_MEME_LABEL}:
        return NO_TEMPLATE_LABEL, True
    return NO_TEMPLATE_LABEL, False


def generate_prediction(
    client: Any,
    cfg: RunConfig,
    query_image_path: str,
    template_list_text: str,
    known_templates: set[str],
    icl_examples: list[InContextExample],
) -> dict[str, Any]:
    contents = build_contents(query_image_path, template_list_text, icl_examples)
    last_error: Exception | None = None
    last_response_text = ""
    for attempt in range(1, cfg.request_timeout_retries + 1):
        try:
            response = client.models.generate_content(
                model=cfg.model_id,
                contents=contents,
                config=genai_types.GenerateContentConfig(
                    system_instruction=SYSTEM_TEXT,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    max_output_tokens=cfg.max_output_tokens,
                    response_mime_type="application/json",
                ),
            )
            response_text = response.text or ""
            last_response_text = response_text
            payload = parse_response_json(response_text)
            predicted_template_raw = payload.get("template")
            predicted_template_eval, valid_exact = coerce_template_value(predicted_template_raw, known_templates)

            local_context = payload.get("local_context", {})
            if not isinstance(local_context, dict):
                local_context = {}

            return {
                "raw_response_text": response_text,
                "raw_template": predicted_template_raw,
                "predicted_template": predicted_template_eval,
                "template_valid_exact": bool(valid_exact),
                "global_context_description": payload.get("global_context_description", ""),
                "local_user_texts_json": json.dumps(local_context.get("user_texts", []), ensure_ascii=False),
                "local_text_meaning": local_context.get("text_meaning", ""),
                "local_instance_specific_image_description": local_context.get(
                    "instance_specific_image_description", ""
                ),
                "global_context_keywords_json": json.dumps(
                    payload.get("global_context_keywords", []), ensure_ascii=False
                ),
                "local_context_keywords_json": json.dumps(
                    payload.get("local_context_keywords", []), ensure_ascii=False
                ),
                "error": "",
            }
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == cfg.request_timeout_retries:
                break
            time.sleep(cfg.retry_sleep_seconds * attempt)

    return {
        "raw_response_text": last_response_text,
        "raw_template": "",
        "predicted_template": NO_TEMPLATE_LABEL,
        "template_valid_exact": False,
        "global_context_description": "",
        "local_user_texts_json": "[]",
        "local_text_meaning": "",
        "local_instance_specific_image_description": "",
        "global_context_keywords_json": "[]",
        "local_context_keywords_json": "[]",
        "error": "" if last_error is None else repr(last_error),
    }


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.random_seed)

    if genai is None or genai_types is None:
        raise RuntimeError(
            "Gemini SDK import failed. Install the current SDK with `python3 -m pip install google-genai` "
            "and rerun this script."
        ) from GENAI_IMPORT_ERROR

    api_key = discover_api_key()
    client = genai.Client(api_key=api_key)

    run_started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    start_perf = time.perf_counter()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = cfg.output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"run_dir={run_dir}")
    print(f"model_id={cfg.model_id}")

    train_df, test_df = load_split_rows(str(cfg.train_parquet), str(cfg.test_parquet))
    train_df = maybe_limit_templates(train_df, cfg.max_templates)
    test_df = test_df[test_df["template"].isin(train_df["template"].unique())].reset_index(drop=True)

    test_df = filter_readable_rows(test_df, limit=cfg.max_test_samples)
    if test_df.empty:
        raise ValueError("No readable test images remain after filtering.")

    template_names = sorted(train_df["template"].unique().tolist())
    known_templates = set(template_names)
    template_list_text = build_template_list_text(template_names)
    icl_examples = sample_icl_examples(train_df, cfg.icl_shots, cfg.random_seed)

    predictions_rows: list[dict[str, Any]] = []
    for row in test_df.itertuples(index=False):
        row_start = time.perf_counter()
        prediction = generate_prediction(
            client=client,
            cfg=cfg,
            query_image_path=row.image_path,
            template_list_text=template_list_text,
            known_templates=known_templates,
            icl_examples=icl_examples,
        )
        latency_seconds = round(time.perf_counter() - row_start, 3)
        predictions_rows.append(
            {
                "image_path": row.image_path,
                "template_true": row.template,
                "template_pred": prediction["predicted_template"],
                "raw_template": prediction["raw_template"],
                "template_valid_exact": prediction["template_valid_exact"],
                "global_context_description": prediction["global_context_description"],
                "local_user_texts_json": prediction["local_user_texts_json"],
                "local_text_meaning": prediction["local_text_meaning"],
                "local_instance_specific_image_description": prediction[
                    "local_instance_specific_image_description"
                ],
                "global_context_keywords_json": prediction["global_context_keywords_json"],
                "local_context_keywords_json": prediction["local_context_keywords_json"],
                "raw_response_text": prediction["raw_response_text"],
                "error": prediction["error"],
                "latency_seconds": latency_seconds,
            }
        )

    predictions_df = pd.DataFrame(predictions_rows)
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(run_dir / "test_predictions.csv", index=False)

    y_true = predictions_df["template_true"].to_numpy()
    y_pred = predictions_df["template_pred"].to_numpy()
    metrics = compute_metrics_with_rejection(y_true, y_pred, reject_token=NO_TEMPLATE_LABEL)
    metrics["evaluated_samples"] = int(len(predictions_df))
    metrics["template_count"] = int(len(template_names))
    metrics["icl_shots"] = int(cfg.icl_shots)
    metrics["template_valid_exact_rate"] = float(predictions_df["template_valid_exact"].mean())
    metrics["error_count"] = int((predictions_df["error"] != "").sum())
    metrics["mean_latency_seconds"] = float(predictions_df["latency_seconds"].mean()) if len(predictions_df) else 0.0

    summary_row = dict(metrics)
    run_metadata = {
        "config": json_ready(asdict(cfg)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "template_names": template_names,
        "icl_examples": [asdict(example) for example in icl_examples],
        "gemini_api_key_source": "env_or_existing_gemini_script",
    }
    finalize_run_timing(run_metadata, summary_row, run_started_at, start_perf)

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
    pd.DataFrame([summary_row]).to_csv(run_dir / "metrics_summary.csv", index=False)

    print(f"predictions_saved={run_dir / 'test_predictions.csv'}")
    print(f"metrics_saved={run_dir / 'metrics_summary.csv'}")
    print(f"run_config_saved={run_dir / 'run_config.json'}")


if __name__ == "__main__":
    main()
