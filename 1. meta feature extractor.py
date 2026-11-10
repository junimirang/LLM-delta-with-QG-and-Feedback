"""
Meta-Feature Extractor (Univariate + Correlation + PCA)
========================================================
Loads the training CSV (already log-transformed + Min-Max normalized, [0,1])
and computes:

    Meta-Feature Extractor
     |- 1. Univariate Profile        (Mean, Var, 5-Quantiles)
     |- 2. Inter-Feature Correlation (Global Stats + Critical Pairs |r| >= tau)
     `- 3. Structural PCA Profile    (Explained Variance + Loading Vectors)

Matches the dataset schema:
    srcip, sport, dstip, dsport,
    log_mean_of_IPA_time, log_ratio_trans_receive, log_speed_transmit_connect,
    log_dur, log_sbytes, log_ct_src_dport_ltm, log_ct_dst_ltm, log_ct_src_ltm,
    new_label

Usage:
    python meta_feature_extractor.py --input train.csv --sample-size 500 \
        --label-col new_label --tau 0.6 --n-pca 3 --output profile.json

Notes:
    --sample-size caps how many rows are drawn (with a fixed seed) before
    computing the profile: this is the "small sample" step, not the full
    training set, per the privacy-preserving design discussed earlier.
    If --label-col is present in the data, one profile is built per class
    (e.g. 'ssh' vs 'non-ssh'); otherwise a single overall profile is built.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# --------------------------------------------------------------------------
# Dataset schema
# --------------------------------------------------------------------------
#FEATURE_COLUMNS = [
#    "log_mean_of_IPA_time",
#    "log_ratio_trans_receive",
#    "log_speed_transmit_connect",
#    "log_dur",
#    "log_sbytes",
#    "log_ct_src_dport_ltm",
#    "log_ct_dst_ltm",
#    "log_ct_src_ltm",
#]

FEATURE_COLUMNS = [
"log_time_interval_mean",
"log_time_interval_var",
"log_time_taken",
"log_cs_byte",
"log_ratio_trans_receive",
"log_count_connect_IP",
"log_count_total_connect",
"log_avg_count_connect",
"log_transmit_speed_BPS"
]

# ============================================================================
# 여기서 직접 값을 수정하시면 됩니다 (터미널 인자 없이 그냥 실행 가능)
# EDIT THESE DIRECTLY — the script runs fine with zero command-line args
# ============================================================================
#INPUT_CSV = "dataset-NB15/UNSW_NB15_new_training_log_normalized.csv"              # <- 학습 CSV 경로 (예: "C:/Users/John/Documents/train.csv")
INPUT_CSV = "dataset-DARPA99/training dataset_week4_normalized.csv"
SAMPLE_SIZE = 10000                    # <- 프로파일 계산에 쓸 샘플 크기
#LABEL_COL = "new_label"              # <- 클래스 컬럼명 (없으면 "" 로 비워두기)
LABEL_COL = "LABEL"              # <- 클래스 컬럼명 (없으면 "" 로 비워두기)
TARGET_LABELS = ["ssh"]                   # <- 특정 라벨만 쓰고 싶으면 예: ["ssh"] 또는 ["ssh", "non-ssh"]
                                      #    비워두면([]) 데이터에 있는 모든 라벨을 사용
TAU = 0.6                            # <- critical correlation pair 임계값
N_PCA = 3                            # <- PCA 컴포넌트 개수
#OUTPUT_JSON = "meta_feature_profile.json"  # <- 출력 JSON 경로
OUTPUT_JSON = "meta_feature_profile_darpa.json"  # <- 출력 JSON 경로
# ============================================================================


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_data(path: str, feature_cols: list[str], label_col: Optional[str]) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing expected feature columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    cols = feature_cols + ([label_col] if label_col and label_col in df.columns else [])
    out = df[cols].copy()
    for c in feature_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    n_before = len(out)
    out = out.dropna(subset=feature_cols)
    if len(out) < n_before:
        print(f"[warn] dropped {n_before - len(out)} rows with non-numeric/NaN feature values")
    return out


def check_normalized_range(df: pd.DataFrame, feature_cols: list[str], tol: float = 1e-6) -> None:
    """Sanity check: features are expected to already be log-transformed +
    Min-Max normalized to [0,1]. Warn (don't fail) if that assumption breaks."""
    offenders = []
    for c in feature_cols:
        lo, hi = df[c].min(), df[c].max()
        if lo < -tol or hi > 1 + tol:
            offenders.append((c, round(float(lo), 4), round(float(hi), 4)))
    if offenders:
        print("[warn] feature(s) outside expected [0,1] range (check upstream normalization):")
        for c, lo, hi in offenders:
            print(f"        - {c}: min={lo}, max={hi}")


# --------------------------------------------------------------------------
# 1. Univariate Profile — Mean, Var, 5-Quantiles
# --------------------------------------------------------------------------
def univariate_profile(sample: pd.DataFrame, feature_cols: list[str]) -> dict:
    profile = {}
    for col in feature_cols:
        x = sample[col].to_numpy(dtype=float)
        q = np.quantile(x, [0.0, 0.25, 0.5, 0.75, 1.0])
        profile[col] = {
            "mean": round(float(np.mean(x)), 6),
            "var": round(float(np.var(x, ddof=1)) if len(x) > 1 else 0.0, 6),
            "quantiles": {
                "min": round(float(q[0]), 6),
                "q1": round(float(q[1]), 6),
                "median": round(float(q[2]), 6),
                "q3": round(float(q[3]), 6),
                "max": round(float(q[4]), 6),
            },
        }
    return profile


# --------------------------------------------------------------------------
# 2. Inter-Feature Correlation — Global Stats + Critical Pairs (|r| >= tau)
# --------------------------------------------------------------------------
def correlation_profile(sample: pd.DataFrame, feature_cols: list[str], tau: float = 0.6) -> dict:
    corr = sample[feature_cols].corr(method="pearson").fillna(0.0)
    n = len(feature_cols)
    iu, ju = np.triu_indices(n, k=1)
    off_diag = corr.to_numpy()[iu, ju]

    critical_pairs = []
    for i, j in zip(iu, ju):
        r = float(corr.iloc[i, j])
        if abs(r) >= tau:
            if abs(r) >= 0.8:
                rel = "strongly positive" if r > 0 else "strongly negative"
            else:
                rel = "moderately positive" if r > 0 else "moderately negative"
            critical_pairs.append({
                "pair": [feature_cols[i], feature_cols[j]],
                "pearson_r": round(r, 4),
                "relationship": rel,
                "residual_scatter_ratio": round(float(np.sqrt(max(0.0, 1 - r ** 2))), 4),
            })

    return {
        "tau": tau,
        "global_stats": {
            "mean_abs_r": round(float(np.mean(np.abs(off_diag))), 4) if len(off_diag) else 0.0,
            "max_abs_r": round(float(np.max(np.abs(off_diag))), 4) if len(off_diag) else 0.0,
        },
        "critical_pairs": critical_pairs,
        "full_matrix": corr.round(4).to_dict(),
    }


# --------------------------------------------------------------------------
# 3. Structural PCA Profile — Explained Variance + Loading Vectors
# --------------------------------------------------------------------------
def pca_profile(sample: pd.DataFrame, feature_cols: list[str], n_components: int = 3) -> dict:
    X = sample[feature_cols].to_numpy(dtype=float)
    n_components = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(X)
    return {
        "explained_variance_ratio": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        "cumulative_explained_variance": round(float(np.sum(pca.explained_variance_ratio_)), 4),
        "loading_vectors": {
            f"pc{i + 1}": {
                feature_cols[j]: round(float(pca.components_[i, j]), 4)
                for j in range(len(feature_cols))
            }
            for i in range(n_components)
        },
    }


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def build_profile(df: pd.DataFrame, feature_cols: list[str], sample_size: int,
                   tau: float, n_pca: int, random_state: int = 42) -> dict:
    n = min(sample_size, len(df))
    sample = df.sample(n=n, random_state=random_state) if n < len(df) else df
    return {
        "dataset_metadata": {
            "num_features": len(feature_cols),
            "feature_names": feature_cols,
            "sample_size_reflected": int(len(sample)),
        },
        "univariate_profile": univariate_profile(sample, feature_cols),
        "correlation_profile": correlation_profile(sample, feature_cols, tau=tau),
        "pca_profile": pca_profile(sample, feature_cols, n_components=n_pca),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Meta-feature extractor: Univariate + Correlation + PCA")
    parser.add_argument("--input", default=INPUT_CSV, help="Path to preprocessed (log + Min-Max normalized) training CSV")
    parser.add_argument("--features", nargs="+", default=FEATURE_COLUMNS,
                         help="Feature columns to profile (defaults to the 8 log_* columns)")
    parser.add_argument("--label-col", default=LABEL_COL,
                         help="Class column; if present, one profile is built per class")
    parser.add_argument("--labels", nargs="+", default=TARGET_LABELS,
                         help="Only include these label value(s), e.g. --labels ssh, or --labels ssh non-ssh. "
                              "Leave empty to use every label found in the data.")
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE,
                         help="Rows to sample before computing the profile (the 'small sample', not the full set)")
    parser.add_argument("--tau", type=float, default=TAU, help="Threshold for 'critical' correlation pairs")
    parser.add_argument("--n-pca", type=int, default=N_PCA)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=OUTPUT_JSON)
    args = parser.parse_known_args()[0]

    df = load_data(args.input, args.features, args.label_col)
    check_normalized_range(df, args.features)

    if args.label_col and args.label_col in df.columns and args.labels:
        available = set(df[args.label_col].astype(str).unique())
        requested = set(args.labels)
        unknown = requested - available
        if unknown:
            print(f"[warn] requested label(s) not found in data: {sorted(unknown)}. "
                  f"Available labels: {sorted(available)}")
        before = len(df)
        df = df[df[args.label_col].astype(str).isin(args.labels)]
        print(f"[info] filtered to labels {sorted(requested & available)}: {before} -> {len(df)} rows")

    if args.label_col and args.label_col in df.columns:
        result = {}
        for label, group in df.groupby(args.label_col):
            print(f"Building profile for class '{label}' (n={len(group)}) ...")
            result[str(label)] = build_profile(group, args.features, args.sample_size, args.tau, args.n_pca, args.seed)
    else:
        result = build_profile(df, args.features, args.sample_size, args.tau, args.n_pca, args.seed)

    out_path = Path(args.output)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote meta-feature profile to {out_path.resolve()}")


if __name__ == "__main__":
    main()