# -*- coding: utf-8 -*-
"""
evaluate_quality.py
====================
t-SNE / JSD / Shannon 엔트로피 비교 평가 스크립트

[비교 대상 - 각 1000샘플]
  1. Original raccess
  2. Original non-raccess
  3. WGAN-GP
  4. LLM_delta
  5. LLM_delta_qg_feedback

[지표]
  - t-SNE  : 2D 시각화 (5개 그룹 동시 플롯)
  - JSD    : Jensen-Shannon Divergence (Original raccess 기준, 피처별 평균)
  - SH_Entropy: 합성 데이터의 피처별 Shannon 엔트로피 평균 (log2, bits)

기존 계산 기준 유지: JSD는 원본/합성 쌍의 공통 범위와 자연로그,
Shannon 엔트로피는 합성 데이터 자체 범위와 log2를 사용합니다.
각 피처의 주변분포를 평가하며, 결합분포나 프라이버시를 보장하지 않습니다.

[실행]
  python evaluate_quality.py
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import entropy
from sklearn.manifold import TSNE
from common_config import DATASET_CONFIG, SSH_LABEL, NON_SSH_LABEL

# ══════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════
DATASET    = "darpa99"       # "darpa99" 또는 "nb15"
N_SAMPLE   = 1000            # 각 그룹 샘플 수
N_BINS     = 10              # JSD / SH_Entropy 히스토그램 bin 수
RANDOM_SEED = 42

# 상대 경로는 실행 작업 폴더가 아니라 이 스크립트 위치를 기준으로 해석
BASE_DIR = Path(__file__).resolve().parent

def resolve_path(value):
    candidate = Path(value)
    return candidate if candidate.is_absolute() else BASE_DIR / candidate

# 데이터셋에 따라 출력 파일명이 자동으로 구분되도록 DATASET 값을 반영
OUTPUT_TSNE = str(resolve_path(f"figures/tsne_comparison_{DATASET}.png"))
OUTPUT_METRICS = str(resolve_path(f"figures/metrics_comparison_{DATASET}.png"))
OUTPUT_CSV     = str(resolve_path(f"figures/metrics_result_{DATASET}.csv"))

# ══════════════════════════════════════════════════════
# 경로 설정
# ══════════════════════════════════════════════════════
cfg = DATASET_CONFIG[DATASET]

PATHS = {
    "Original raccess":         ("real",  cfg["real_dir"],  cfg["train_file"],     SSH_LABEL),
    "Original non-raccess":     ("real",  cfg["real_dir"],  cfg["train_file"],     NON_SSH_LABEL),
    "WGAN-GP": ("fake", os.path.join(cfg["fake_dir"], "wgan-gp"), "synthetic_ssh.csv", None),
    "LLM_delta": ("fake", os.path.join(cfg["fake_dir"], "2_2_llm_delta"), "synthetic_ssh.csv", None),
    "LLM_delta_qg_feedback": ("fake", os.path.join(cfg["fake_dir"], "2_4_llm_delta_with qg and feedback"), "synthetic_ssh_filtered.csv", None),
}

COLORS = {
    "Original raccess":        "#1f77b4",   # 파랑
    "Original non-raccess":    "black",   # 검정
    "WGAN-GP":         "green",   # 녹색
    "LLM_delta":             "#FFB74D",   # 주황
    "LLM_delta_qg_feedback": "#9467bd",   # 보라
}

MARKERS = {
    "Original raccess":        "o",
    "Original non-raccess":    "s",
    "WGAN-GP":         "^",
    "LLM_delta":             "D",
    "LLM_delta_qg_feedback": "P",
}


# ══════════════════════════════════════════════════════
# 데이터 로드
# ══════════════════════════════════════════════════════
def load_samples(name, kind, dirpath, filename, label_filter):
    fpath = str(resolve_path(os.path.join(dirpath, filename)))
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"[{name}] 파일 없음: {fpath}")

    df = pd.read_csv(fpath).replace([float("inf"), float("-inf")], float("nan"))
    missing = [c for c in cfg["feature_cols"] if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] 피처 컬럼 없음: {missing}")
    df[cfg["feature_cols"]] = df[cfg["feature_cols"]].apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=cfg["feature_cols"]).reset_index(drop=True)

    if label_filter is not None:
        df = df[df[cfg["label_col"]] == label_filter].reset_index(drop=True)

    if len(df) < N_SAMPLE:
        raise ValueError(f"[{name}] 샘플 부족: {len(df)} < {N_SAMPLE}")

    sampled = df.sample(n=N_SAMPLE, random_state=RANDOM_SEED)
    X = sampled[cfg["feature_cols"]].values.astype(np.float32)
    print(f"  Loaded  {name:<25}: {X.shape}")
    return X


SYNTHETIC_MODELS = ("WGAN-GP", "LLM_delta", "LLM_delta_qg_feedback")

# ══════════════════════════════════════════════════════
# 공통 유틸
# ══════════════════════════════════════════════════════
def hist_prob(arr, bins):
    counts, _ = np.histogram(arr, bins=bins)
    prob = counts + 1e-10
    return prob / prob.sum()

def jsd_1d(p_arr, q_arr):
    min_v = min(p_arr.min(), q_arr.min())
    max_v = max(p_arr.max(), q_arr.max())
    if max_v == min_v:
        return 0.0
    bins = np.linspace(min_v, max_v, N_BINS + 1)
    p = hist_prob(p_arr, bins)
    q = hist_prob(q_arr, bins)
    m = 0.5 * (p + q)
    return float(0.5 * (entropy(p, m) + entropy(q, m)))

def entropy_1d(arr, bins):
    counts, _ = np.histogram(arr, bins=bins)
    prob = counts + 1e-10
    prob = prob / prob.sum()
    return float(entropy(prob))


def main():
    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  Dataset   : {DATASET.upper()}")
    print(f"  N_sample  : {N_SAMPLE}")
    print(f"  Features  : {cfg['feature_cols']}")
    print(f"{'='*60}\n")

    print("[1/4] 데이터 로드...")
    data = {}
    for name, (kind, dirpath, filename, label_filter) in PATHS.items():
        data[name] = load_samples(name, kind, dirpath, filename, label_filter)


    # ══════════════════════════════════════════════════════
    # JSD / SH_Entropy 계산 (Original raccess 기준)
    # ══════════════════════════════════════════════════════
    print("[2/4] JSD / SH_Entropy 계산...")

    ref = data["Original raccess"]
    n_feat = ref.shape[1]
    feat_names = cfg["feature_cols"]

    results = []
    for name in SYNTHETIC_MODELS:
        syn = data[name]
        jsd_list, sh_list = [], []
        for fi in range(n_feat):
            p = ref[:, fi]
            q = syn[:, fi]
            min_v = min(p.min(), q.min())
            max_v = max(p.max(), q.max())
            bins = np.linspace(min_v, max_v, N_BINS + 1) if max_v > min_v else np.array([min_v - 1e-6, max_v + 1e-6])
            jsd_list.append(jsd_1d(p, q))

            # Shannon Entropy of synthetic data (log₂)
            # H(X_Synthetic) = -∑ p(x_i) log₂ p(x_i)
            q_bins = np.linspace(q.min(), q.max(), N_BINS + 1) if q.max() > q.min() else np.array(
                [q.min() - 1e-6, q.max() + 1e-6])
            counts, _ = np.histogram(q, bins=q_bins)
            prob = counts + 1e-10
            prob = prob / prob.sum()
            h_syn = -np.sum(prob * np.log2(prob))  # log₂ 사용
            sh_list.append(h_syn)
        mean_jsd = np.mean(jsd_list)
        mean_sh = np.mean(sh_list)
        results.append({"model": name, "JSD": mean_jsd, "SH_Entropy": mean_sh})
        print(f"  {name:<25}: JSD={mean_jsd:.6f}  SH_Entropy={mean_sh:.6f}")

    df_result = pd.DataFrame(results)
    df_result.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"  -> 결과 저장: {OUTPUT_CSV}")


    # ══════════════════════════════════════════════════════
    # t-SNE
    # ══════════════════════════════════════════════════════
    print("[3/4] t-SNE 실행...")

    all_X = np.vstack([data[n] for n in PATHS.keys()])
    group_labels = []
    for name in PATHS.keys():
        group_labels.extend([name] * N_SAMPLE)
    group_labels = np.array(group_labels)

    tsne = TSNE(n_components=2, random_state=RANDOM_SEED,
                perplexity=40, max_iter=1000, verbose=0)
    X_2d = tsne.fit_transform(all_X)
    print(f"  t-SNE 완료: {X_2d.shape}")

    # t-SNE 플롯
    fig, ax = plt.subplots(figsize=(10, 8))
    for name in PATHS.keys():
        mask = group_labels == name
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=COLORS[name], marker=MARKERS[name],
                   s=18, alpha=0.6, label=name, linewidths=0)
    ax.set_title(f"t-SNE Visualization  (n={N_SAMPLE} per group)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("t-SNE Dim 1", fontsize=11)
    ax.set_ylabel("t-SNE Dim 2", fontsize=11)
    ax.legend(fontsize=9, markerscale=1.8, frameon=True, edgecolor="gray")
    ax.grid(linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_TSNE, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> 저장: {OUTPUT_TSNE}")


    # ══════════════════════════════════════════════════════
    # JSD / SH_Entropy 막대 그래프
    # ══════════════════════════════════════════════════════
    print("[4/4] 지표 시각화...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    model_names = [r["model"] for r in results]
    short_names = ["WGAN-GP", "LLM δ", "LLM δ + QG + Feedback"]
    bar_colors  = [COLORS[n] for n in model_names]

    # JSD
    jsd_vals = [r["JSD"] for r in results]
    axes[0].bar(short_names, jsd_vals, color=bar_colors, edgecolor="white", width=0.5)
    for i, v in enumerate(jsd_vals):
        axes[0].text(i, v + max(jsd_vals)*0.01, f"{v:.5f}", ha="center", fontsize=10)
    axes[0].set_title("(a) JSD  (vs Original raccess)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Jensen-Shannon Divergence", fontsize=11)
    axes[0].set_ylim(0, max(max(jsd_vals) * 1.18, 0.01))
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    # Shannon entropy
    vee_vals = [r["SH_Entropy"] for r in results]
    axes[1].bar(short_names, vee_vals, color=bar_colors, edgecolor="white", width=0.5)
    for i, v in enumerate(vee_vals):
        axes[1].text(i, v + max(vee_vals)*0.01, f"{v:.5f}", ha="center", fontsize=10)
    axes[1].set_title("(b)Shannon Entropy", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Shannon Entropy (bits, log₂)", fontsize=11)
    axes[1].set_ylim(0, max(max(vee_vals) * 1.18, 0.01))
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    plt.suptitle(f"Synthetic Data Quality Metrics  (n={N_SAMPLE})",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_METRICS, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> 저장: {OUTPUT_METRICS}")

    print(f"\n{'='*60}")
    print("  완료")
    print(f"  t-SNE  : {OUTPUT_TSNE}")
    print(f"  지표   : {OUTPUT_METRICS}")
    print(f"  CSV    : {OUTPUT_CSV}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
