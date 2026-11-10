"""
LLM 기반 SSH 합성 트래픽 생성기
================================
meta_feature_extractor.py로 뽑은 meta_feature_profile.json을 로컬 LLM 서버
(LM Studio 등 OpenAI 호환 엔드포인트)에 전달해서, 그 통계 프로파일에 맞는
합성 트래픽 샘플을 생성합니다.

Usage:
    python generate_synthetic_traffic.py --profile meta_feature_profile.json \
        --label ssh --num-samples 500 --batch-size 50 \
        --temperature 0.8 --top-p 0.95 \
        --server http://172.21.0.1:1234 --output synthetic_ssh.csv

Notes:
    --num-samples가 전체 생성 목표치이고, --batch-size는 LLM 한 번 호출에
    요청하는 샘플 수입니다. num-samples가 batch-size보다 크면 자동으로
    여러 번 나눠서 호출합니다 (한 번에 너무 많이 요청하면 상관관계 유지가
    더 어려워지는 것으로 관찰되어, 배치를 나누는 걸 기본으로 합니다).

    USE_JSON_SCHEMA를 켜두면 (기본값) LM Studio의 구조화 출력(response_format
    JSON Schema)을 사용해서, 모델이 8개 피처 중 하나를 통째로 빠뜨리는 문제를
    디코딩 단계에서 원천 차단합니다. 서버/모델이 이 기능을 지원하지 않으면
    --use-json-schema false로 끄고 기존 프롬프트 지시문 방식만 쓰시면 됩니다.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import requests
import numpy as np
import pandas as pd


class TruncatedOutputError(ValueError):
    """Raised when the model's completion appears to have been cut off
    mid-generation (context/output token limit), as opposed to genuinely
    malformed JSON. Callers can react by shrinking the batch size instead
    of just retrying at the same size."""


# --------------------------------------------------------------------------
# Mahalanobis-distance quality filter (mimics a discriminator's role: reject
# samples that are individually well-formed but sit outside where real SSH
# traffic actually lives in feature space). See filter_synthetic_quality.py
# for the standalone, post-hoc version of the same idea; here it runs inline
# per batch so rejected samples are automatically compensated for by the
# existing top-up loop in main().
# --------------------------------------------------------------------------
def load_real_sample(path: str, label_col: str, label: str,
                      sample_size: Optional[int], seed: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    if label_col and label_col in df.columns:
        available = sorted(df[label_col].astype(str).unique())
        before = len(df)
        df = df[df[label_col].astype(str) == str(label)]
        print(f"[info] filtered --real to label '{label}' via column '{label_col}': "
              f"{before} -> {len(df)} rows (available labels: {available})")
        if len(df) == 0:
            raise ValueError(f"No rows with label '{label}' found in column '{label_col}'")
    elif label_col:
        print(f"[warn] label column '{label_col}' not found in --real, using all rows unfiltered")

    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
        print(f"[info] sampled {sample_size} rows from --real (seed={seed})")
    else:
        print(f"[info] using all {len(df)} available rows from --real")

    return df


def fit_mean_cov(X: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    cov = cov + ridge * np.eye(cov.shape[0])
    inv_cov = np.linalg.pinv(cov)
    return mean, inv_cov


def mahalanobis_distances(X: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray) -> np.ndarray:
    diff = X - mean
    left = diff @ inv_cov
    d2 = np.einsum("ij,ij->i", left, diff)
    return np.sqrt(np.clip(d2, 0, None))


def build_quality_filter(real_csv: str, label_col: str, label: str, sample_size: int,
                          seed: int, feature_names: list[str], ridge: float,
                          percentile: float) -> Optional[dict]:
    """Fit the real-sample mean/covariance once and return a small dict with
    everything needed to score a batch. Returns None (filter disabled) if
    the real CSV can't be found."""
    try:
        real_df = load_real_sample(real_csv, label_col, label, sample_size, seed)
    except FileNotFoundError:
        print(f"[warn] --real file '{real_csv}' not found; quality filter disabled")
        return None
    missing = [f for f in feature_names if f not in real_df.columns]
    if missing:
        print(f"[warn] --real is missing feature(s) {missing}; quality filter disabled")
        return None
    X_real = real_df[feature_names].to_numpy(dtype=float)
    mean, inv_cov = fit_mean_cov(X_real, ridge)
    d_real = mahalanobis_distances(X_real, mean, inv_cov)
    threshold = float(np.percentile(d_real, percentile))
    std = X_real.std(axis=0)
    std[std == 0] = 1e-9  # avoid div-by-zero for a constant feature
    print(f"[info] quality filter ready: real n={len(real_df)}, "
          f"p{percentile:.0f} distance threshold={threshold:.3f}")
    return {"mean": mean, "inv_cov": inv_cov, "threshold": threshold, "std": std}


def apply_quality_filter(rows: list[dict], feature_names: list[str],
                          qf: dict) -> tuple[list[dict], list[dict], int]:
    if not rows:
        return rows, [], 0
    X = np.array([[r[f] for f in feature_names] for r in rows], dtype=float)
    d = mahalanobis_distances(X, qf["mean"], qf["inv_cov"])
    keep = d <= qf["threshold"]
    kept_rows = [r for r, k in zip(rows, keep) if k]
    rejected_rows = [r for r, k in zip(rows, keep) if not k]
    return kept_rows, rejected_rows, int((~keep).sum())


def summarize_rejection_feedback(rejected_rows: list[dict], feature_names: list[str],
                                  qf: dict, top_k: int = 3) -> Optional[str]:
    """Turn a batch's rejected samples into a short, human-readable note
    pointing at which features drove the rejections (largest average
    |z-score| against the real sample), so the NEXT prompt can ask the
    model to be more careful specifically there. This is in-context
    feedback, not gradient-based training: the model itself never updates,
    only what it's told next time changes."""
    if not rejected_rows:
        return None
    X = np.array([[r[f] for f in feature_names] for r in rejected_rows], dtype=float)
    z = (X - qf["mean"]) / qf["std"]
    mean_abs_z = np.abs(z).mean(axis=0)
    order = np.argsort(-mean_abs_z)[:top_k]
    parts = []
    for i in order:
        direction = "above" if (X[:, i] - qf["mean"][i]).mean() > 0 else "below"
        parts.append(f"{feature_names[i]} (avg |z|={mean_abs_z[i]:.2f}, typically too far {direction} the real mean)")
    return (
        f"{len(rejected_rows)} sample(s) from the previous batch were rejected as statistical "
        f"outliers (Mahalanobis distance too large). The features that drove those rejections "
        f"the most were: {'; '.join(parts)}. Stay closer to the reported quantile ranges for "
        f"these features in particular, while still following the diversity rules above."
    )

# ============================================================================
# 여기서 직접 값을 수정하시면 됩니다 (터미널 인자 없이 그냥 실행 가능)
# ============================================================================
PROFILE_JSON = "meta_feature_profile_darpa.json"   # <- meta_feature_extractor.py 출력 파일
TARGET_LABEL = "ssh"                          # <- 프로파일 안에서 쓸 클래스 키 (예: "ssh", "non-ssh")
NUM_SAMPLES = 1000                             # <- 최종적으로 생성할 총 샘플 수
BATCH_SIZE = 25                               # <- LLM 한 번 호출당 요청할 샘플 수
TEMPERATURE = 1.2
TOP_P = 0.95
SERVER_URL = "http://127.0.0.1:1234"         # <- LM Studio 서버 주소 (포트 포함, 끝에 / 없이)
MODEL_NAME = ""                                # <- 보통 비워둬도 됨 (LM Studio가 로드된 모델을 자동 사용)
OUTPUT_CSV = "fake dataset-DARPA99/2_4_llm_delta_with qg and feedback/synthetic_ssh_filtered.csv"
MAX_RETRIES = 3                                # <- 배치 하나가 실패했을 때 재시도 횟수
USE_JSON_SCHEMA = True                         # <- LM Studio의 구조화 출력(JSON Schema 강제)을 쓸지 여부
MAX_CONSECUTIVE_BATCH_FAILURES = 5             # <- 배치가 연속으로 이만큼 완전히 실패하면 (서버 다운 등) 중단
DISABLE_THINKING = True                        # <- Qwen3 계열 등 reasoning 모델의 thinking 모드를 끌지 여부
MAX_TOKENS_BASE = 500                          # <- max_tokens = BASE + batch_n * PER_SAMPLE
MAX_TOKENS_PER_SAMPLE = 250                    # <- reasoning 모델은 순수 JSON보다 토큰을 더 쓰므로 여유있게
ENABLE_QUALITY_FILTER = True                   # <- Mahalanobis distance 기반 이상치 필터 사용 여부
REAL_CSV = "dataset-DARPA99/training dataset_week4_normalized.csv"               # <- 메타 피처 뽑을 때 쓴 소량 real 샘플 (라벨 컬럼 있어도 됨)
REAL_LABEL_COL = "LABEL"                       # <- real CSV의 라벨 컬럼명 (없으면 "" 로 비워두기)
REAL_LABEL = "ssh"                             # <- 사용할 라벨 값
REAL_SAMPLE_SIZE = 500                         # <- 라벨 필터링 후 여기서 무작위로 뽑을 샘플 수 (0이면 전체)
QUALITY_PERCENTILE = 99.0                      # <- 이 percentile을 넘는 합성 샘플은 배치에서 바로 제외
QUALITY_RIDGE = 1e-6                           # <- 공분산 역행렬 안정화용 작은 값
# ============================================================================


SYSTEM_INSTRUCTION = (
    "You are a conditional synthetic network traffic generator. "
    "You will be given a statistical profile (univariate distributions, "
    "correlation structure with residual scatter ratios, and PCA structure) "
    "describing one class of network traffic. Generate NEW synthetic samples "
    "that are statistically consistent with this profile but NOT identical "
    "to each other or to a straight-line reproduction of the correlations. "
    "Each feature value must be a float in [0,1] rounded to 4 decimal places.\n\n"
    "Follow these rules when generating a batch:\n"
    "1. For every critical pair, its residual_scatter_ratio tells you exactly "
    "how tightly coupled the two features should be. A LOW ratio (e.g. "
    "0.2-0.4) means the relationship is nearly deterministic in the real "
    "data, so add only minimal noise. A HIGH ratio (e.g. 0.7-0.8) means the "
    "relationship is loose, so allow substantial independent variation. "
    "Treat each critical pair according to its own ratio; do not apply the "
    "same amount of scatter to all of them.\n"
    "2. Do not produce two or more samples in the batch with identical "
    "values (to 4 decimal places) for any feature.\n"
    "3. Do not build one 'typical' row and then create the rest of the "
    "batch by making small edits to it. Generate each sample as if it came "
    "from an independent session.\n"
    "4. For each feature, spread the generated values roughly evenly across "
    "the reported quantile bands (min-q1, q1-median, median-q3, q3-max) "
    "rather than clustering near the mean or median.\n\n"
    "Output ONLY a JSON array of objects, one object per sample, with "
    "exactly the given feature names as keys and nothing else: no "
    "explanation, no markdown, no code fences."
)


# --------------------------------------------------------------------------
# JSON Schema for LM Studio structured output
# --------------------------------------------------------------------------
def build_response_schema(feature_names: list[str]) -> dict:
    """LM Studio structured-output schema: forces every sample object to
    contain exactly these keys. Range ([0,1]) is intentionally NOT encoded
    as a JSON-Schema minimum/maximum here: some local backends' strict mode
    silently return an empty completion when numeric bounds are combined
    with a large nested array-of-objects schema. Range is enforced
    downstream in parse_samples() instead, which is safer across backends."""
    sample_schema = {
        "type": "object",
        "properties": {f: {"type": "number"} for f in feature_names},
        "required": feature_names,
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "synthetic_traffic_batch",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"samples": {"type": "array", "items": sample_schema}},
                "required": ["samples"],
                "additionalProperties": False,
            },
        },
    }


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------
def build_user_prompt(profile: dict, feature_names: list[str], n_samples: int,
                       feedback: Optional[str] = None) -> str:
    payload = {
        "feature_names": feature_names,
        "univariate_profile": profile["univariate_profile"],
        "correlation_profile": {
            "global_stats": profile["correlation_profile"]["global_stats"],
            "critical_pairs": profile["correlation_profile"]["critical_pairs"],
        },
        "pca_profile": profile.get("pca_profile", {}),
        "generation_request": {
            "num_samples": n_samples,
            "diversity_instruction": (
                "Do not repeat identical values across samples and do not "
                "clone-and-tweak one template row. For each critical pair, "
                "use its own residual_scatter_ratio to decide how much "
                "independent scatter to add: pairs with a low ratio stay "
                "close to a fixed relationship, pairs with a high ratio "
                "should show real spread. Distribute samples across each "
                "feature's quantile bands rather than bunching near the "
                "mean."
            ),
        },
    }
    if feedback:
        payload["feedback_from_previous_batch"] = feedback
    return json.dumps(payload, ensure_ascii=False)


# --------------------------------------------------------------------------
# LLM call (OpenAI-compatible /v1/chat/completions, e.g. LM Studio)
# --------------------------------------------------------------------------
def call_llm(server_url: str, model: str, system_msg: str, user_msg: str,
             temperature: float, top_p: float, response_format: Optional[dict] = None,
             max_tokens: Optional[int] = None, disable_thinking: bool = True,
             timeout: int = 600) -> str:
    url = server_url.rstrip("/") + "/v1/chat/completions"
    if disable_thinking:
        # Qwen3 계열의 표준 관례: 프롬프트에 /no_think를 넣으면 thinking(reasoning) 단계를
        # 건너뛰고 바로 최종 답만 생성한다. 다른 계열 모델에는 영향이 없거나 무시된다.
        system_msg = system_msg + "\n\n/no_think"
    body = {
        "model": model or "local-model",
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "temperature": temperature,
        "top_p": top_p,
    }
    if response_format is not None:
        body["response_format"] = response_format
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    resp = requests.post(url, json=body, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    message = data["choices"][0]["message"]
    content = message.get("content") or ""

    if not content.strip():
        # reasoning 모델(Qwen3 계열 등)이 thinking 모드에서 실제 출력을
        # content가 아니라 reasoning_content에 써버리는 경우가 있다.
        # 그 안에 쓸만한 JSON이 있으면 그걸 대신 쓴다.
        reasoning = message.get("reasoning_content") or ""
        if reasoning.strip():
            print("[info] content was empty but reasoning_content had text; using that instead "
                  "(the model likely ignored /no_think or the server doesn't support it)")
            return reasoning
        finish_reason = data["choices"][0].get("finish_reason", "unknown")
        usage = data.get("usage", {})
        raise ValueError(
            f"Model returned an EMPTY completion (finish_reason={finish_reason}, "
            f"usage={usage}). This usually means the server rejected or silently "
            f"failed on the response_format/JSON-schema request, or a reasoning "
            f"model ran out of tokens while thinking. Try --use-json-schema false, "
            f"or increase --max-tokens-base. Full response for debugging: "
            f"{json.dumps(data)[:800]}"
        )
    return content


# --------------------------------------------------------------------------
# Parse + validate model output
# --------------------------------------------------------------------------
def parse_samples(raw_text: str, feature_names: list[str]) -> list[dict]:
    text = raw_text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:]
    text = text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        # If the failure point is right near the end of what we received,
        # this is almost certainly a cut-off completion (hit the model's
        # context/output limit mid-generation), not a formatting mistake
        # somewhere in the middle. Distinguish the two so the caller can
        # react differently (shrink the batch) instead of just retrying
        # the same size and hitting the same wall again.
        if e.pos >= len(text) - 30:
            raise TruncatedOutputError(
                f"Model output was CUT OFF (likely hit the context/output token limit): "
                f"{e}\n--- raw output (first 500 chars) ---\n{raw_text[:500]}"
            )
        raise ValueError(
            f"Model output was not valid JSON ({e}).\n--- raw output (first 500 chars) ---\n{raw_text[:500]}"
        )

    if isinstance(parsed, dict) and "samples" in parsed:
        parsed = parsed["samples"]  # structured-output mode wraps the array

    if not isinstance(parsed, list):
        raise ValueError(f"Expected a JSON array of sample objects, got {type(parsed).__name__}")

    rows = []
    for i, item in enumerate(parsed):
        if not isinstance(item, dict):
            print(f"[warn] sample {i} is not a JSON object, skipping")
            continue
        missing = [f for f in feature_names if f not in item]
        if missing:
            print(f"[warn] sample {i} missing feature(s) {missing}, skipping")
            continue
        row, out_of_range = {}, []
        for f in feature_names:
            try:
                v = float(item[f])
            except (TypeError, ValueError):
                print(f"[warn] sample {i} has non-numeric value for '{f}', skipping sample")
                row = None
                break
            if v < 0.0 or v > 1.0:
                out_of_range.append((f, round(v, 4)))
            row[f] = v
        if row is None:
            continue
        if out_of_range:
            print(f"[warn] sample {i} has out-of-range value(s), DROPPING sample: {out_of_range}")
            continue
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# One batch = one LLM call, with retries, and auto-splitting on truncation
# --------------------------------------------------------------------------
MIN_SPLIT_SIZE = 4  # below this, stop trying to split further; just fail normally


def generate_batch(server_url: str, model: str, profile: dict, feature_names: list[str],
                    batch_n: int, temperature: float, top_p: float, max_retries: int,
                    max_tokens_base: int, max_tokens_per_sample: int,
                    response_format: Optional[dict] = None,
                    disable_thinking: bool = True, feedback: Optional[str] = None) -> list[dict]:
    user_prompt = build_user_prompt(profile, feature_names, batch_n, feedback=feedback)
    max_tokens = max_tokens_base + batch_n * max_tokens_per_sample
    last_err: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            raw = call_llm(server_url, model, SYSTEM_INSTRUCTION, user_prompt,
                            temperature, top_p, response_format=response_format,
                            max_tokens=max_tokens, disable_thinking=disable_thinking)
            rows = parse_samples(raw, feature_names)
            if len(rows) == 0:
                raise ValueError("Parsed 0 valid rows from model output")
            return rows
        except TruncatedOutputError as e:
            last_err = e
            print(f"[warn] batch attempt {attempt}/{max_retries} was truncated (n={batch_n}): {e}")
            if batch_n > MIN_SPLIT_SIZE:
                half_a = batch_n // 2
                half_b = batch_n - half_a
                print(f"[info] output looks cut off at n={batch_n}; splitting into {half_a}+{half_b} "
                      f"and retrying each half separately")
                rows_a = generate_batch(server_url, model, profile, feature_names, half_a,
                                         temperature, top_p, max_retries, max_tokens_base,
                                         max_tokens_per_sample, response_format, disable_thinking,
                                         feedback)
                rows_b = generate_batch(server_url, model, profile, feature_names, half_b,
                                         temperature, top_p, max_retries, max_tokens_base,
                                         max_tokens_per_sample, response_format, disable_thinking,
                                         feedback)
                return rows_a + rows_b
            time.sleep(1.5)
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"[warn] batch attempt {attempt}/{max_retries} failed: {e}")
            time.sleep(1.5)
    raise RuntimeError(f"Batch generation failed after {max_retries} attempts: {last_err}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic traffic via a local LLM server")
    parser.add_argument("--profile", default=PROFILE_JSON, help="Path to meta_feature_profile.json")
    parser.add_argument("--label", default=TARGET_LABEL, help="Which class key inside the profile JSON to use")
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES, help="Total synthetic samples to generate")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Samples requested per single LLM call")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=TOP_P)
    parser.add_argument("--server", default=SERVER_URL, help="Base URL of the local LLM server, e.g. http://172.21.0.1:1234")
    parser.add_argument("--model", default=MODEL_NAME, help="Model name if the server requires it")
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES)
    parser.add_argument("--use-json-schema", type=lambda v: v.lower() != "false",
                         default=USE_JSON_SCHEMA,
                         help="Use LM Studio JSON-schema structured output (True/False, default True)")
    parser.add_argument("--max-consecutive-failures", type=int, default=MAX_CONSECUTIVE_BATCH_FAILURES,
                         help="Stop if this many batches in a row fail completely (e.g. server down)")
    parser.add_argument("--disable-thinking", type=lambda v: v.lower() != "false",
                         default=DISABLE_THINKING,
                         help="Append /no_think for Qwen3-style reasoning models (True/False, default True)")
    parser.add_argument("--max-tokens-base", type=int, default=MAX_TOKENS_BASE)
    parser.add_argument("--max-tokens-per-sample", type=int, default=MAX_TOKENS_PER_SAMPLE)
    parser.add_argument("--enable-quality-filter", type=lambda v: v.lower() != "false",
                         default=ENABLE_QUALITY_FILTER,
                         help="Reject samples too far (Mahalanobis distance) from the real sample (True/False)")
    parser.add_argument("--real", default=REAL_CSV, help="CSV of the small real sample used for the meta-feature profile")
    parser.add_argument("--real-label-col", default=REAL_LABEL_COL)
    parser.add_argument("--real-label", default=REAL_LABEL)
    parser.add_argument("--real-sample-size", type=int, default=REAL_SAMPLE_SIZE)
    parser.add_argument("--quality-percentile", type=float, default=QUALITY_PERCENTILE)
    parser.add_argument("--quality-ridge", type=float, default=QUALITY_RIDGE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=OUTPUT_CSV)
    args = parser.parse_known_args()[0]

    with open(args.profile, encoding="utf-8") as f:
        full_profile = json.load(f)

    if args.label not in full_profile:
        raise KeyError(f"Label '{args.label}' not found in profile. Available labels: {list(full_profile.keys())}")
    profile = full_profile[args.label]
    feature_names = list(profile["univariate_profile"].keys())
    response_format = build_response_schema(feature_names) if args.use_json_schema else None
    if response_format:
        print("[info] using JSON-schema structured output (all feature keys will be enforced)")
    if args.disable_thinking:
        print("[info] appending /no_think to the system prompt (Qwen3-style reasoning models)")

    quality_filter = None
    if args.enable_quality_filter:
        quality_filter = build_quality_filter(
            args.real, args.real_label_col, args.real_label, args.real_sample_size,
            args.seed, feature_names, args.quality_ridge, args.quality_percentile,
        )

    target = args.num_samples
    all_rows: list[dict] = []
    seen: set = set()  # for exact-duplicate detection
    batch_idx = 0
    consecutive_failures = 0
    feedback: Optional[str] = None  # in-context note built from the previous batch's rejections

    # 목표는 "요청한 배치 수"가 아니라 "실제로 쌓인 유효 샘플 수" 기준으로 판단한다.
    # 배치 하나가 완전히 실패해도(재시도 다 소진) 스크립트를 죽이지 않고 다음 배치로 넘어간다.
    while len(all_rows) < target:
        batch_idx += 1
        remaining = target - len(all_rows)
        this_batch = min(args.batch_size, remaining)
        est_max_tokens = args.max_tokens_base + this_batch * args.max_tokens_per_sample
        print(f"[info] requesting batch {batch_idx}: {this_batch} samples "
              f"(temperature={args.temperature}, top_p={args.top_p}, max_tokens~={est_max_tokens}, "
              f"progress={len(all_rows)}/{target}) ...")
        if feedback:
            print(f"[info] including feedback from previous batch: {feedback}")
        try:
            rows = generate_batch(args.server, args.model, profile, feature_names,
                                   this_batch, args.temperature, args.top_p, args.max_retries,
                                   args.max_tokens_base, args.max_tokens_per_sample,
                                   response_format=response_format,
                                   disable_thinking=args.disable_thinking, feedback=feedback)
        except RuntimeError as e:
            consecutive_failures += 1
            print(f"[warn] batch {batch_idx} failed completely after {args.max_retries} retries, "
                  f"skipping it and moving on ({consecutive_failures}/{args.max_consecutive_failures} "
                  f"consecutive failures): {e}")
            if consecutive_failures >= args.max_consecutive_failures:
                print("[error] too many consecutive batch failures in a row, stopping "
                      "(server may be unreachable or misconfigured)")
                break
            continue

        consecutive_failures = 0
        feedback = None  # reset; only re-set below if this batch had rejections

        if quality_filter is not None:
            n_batch_before = len(rows)
            rows, rejected_rows, n_rejected = apply_quality_filter(rows, feature_names, quality_filter)
            if n_rejected:
                print(f"[warn] batch {batch_idx}: quality filter rejected {n_rejected}/{n_batch_before} "
                      f"sample(s) as too far from the real distribution")
                feedback = summarize_rejection_feedback(rejected_rows, feature_names, quality_filter)

        n_before = len(all_rows)
        n_dupes = 0
        for row in rows:
            key = tuple(row[f] for f in feature_names)
            if key in seen:
                n_dupes += 1
                continue
            seen.add(key)
            all_rows.append(row)
        if n_dupes:
            print(f"[warn] batch {batch_idx}: dropped {n_dupes} exact-duplicate sample(s)")
        print(f"[info] batch {batch_idx}: received {len(rows)} valid samples, "
              f"{len(all_rows) - n_before} were new (total now {len(all_rows)}/{target})")

    if len(all_rows) < target:
        print(f"[warn] stopped short of target: {len(all_rows)}/{target} samples collected")

    df = pd.DataFrame(all_rows, columns=feature_names)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[done] wrote {len(df)} synthetic samples to {out_path.resolve()}")


if __name__ == "__main__":
    main()