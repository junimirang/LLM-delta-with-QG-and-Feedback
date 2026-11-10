# -*- coding: utf-8 -*-
"""
generate_traffic_batch.py  v1.0
================================
model_generator_batch.py로 학습된 WGAN-GP / VAE-WGAN-GP 체크포인트를 이용하여
DARPA99 / NB15 전체 조합에 대해 합성 트래픽을 한 번의 실행으로 생성

[대상 조합 — 4개]
  (darpa99, wgan_gp)
  (darpa99, vae_wgan_gp)
  (nb15,    wgan_gp)
  (nb15,    vae_wgan_gp)

  ※ vae_wgan_gp_lcl(α 필요)은 다루지 않음
    → traffic_generator_for_each_alpha_*.py 에서 별도 처리

[샘플 선택 방식]
  wgan_gp     : z ~ N(0,1)            → softmax(label) > softmax_thresh
  vae_wgan_gp : z_vae = μ+ε·σ·scale   → softmax(label) > softmax_thresh
                (LCL은 정보 기록용으로만 계산, 선택 기준으로 사용하지 않음
                 — model_generator_batch.py에서 LCL을 학습 손실에 포함하지
                   않았으므로 선택 기준 우선순위는 softmax로 통일)

[실행]
  python generate_traffic_batch.py
"""

import os
import glob
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from common_config import (DATASET_CONFIG, MODEL_DIR, SSH_LABEL, NON_SSH_LABEL,
                           LATENT_DIM, NOISE_SCALE)
from input_helper import ask, section

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 배치 대상 조합 (model_generator_batch.py와 동일)
DATASETS = ["darpa99"]
MODES    = ["wgan_gp"]


# ══════════════════════════════════════════════════════
# 대화형 입력 — 공통 파라미터만 한 번 입력
# ══════════════════════════════════════════════════════

def get_params():
    section("generate_traffic_batch.py  v1.0  —  공통 파라미터 입력")
    print(f"  대상 조합: {[(d, m) for d in DATASETS for m in MODES]}\n")

    target_class   = ask("생성 클래스",         choices=["ssh","non-ssh"], default="ssh")
    n_samples      = ask("데이터셋×모델 조합당 생성할 샘플 수", cast=int,   default=100000)
    softmax_thresh = ask("softmax 필터 임계값",  cast=float,                default=0.7)

    class P: pass
    p = P()
    p.target_class   = target_class
    p.n_samples      = n_samples
    p.softmax_thresh = softmax_thresh
    return p


# ══════════════════════════════════════════════════════
# 모델 정의 (model_generator_batch.py와 동일 구조)
# ══════════════════════════════════════════════════════

class VAE(nn.Module):
    def __init__(self, n_total):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_total, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256),     nn.LeakyReLU(0.2),
        )
        self.fc_mu     = nn.Linear(256, LATENT_DIM)
        self.fc_logvar = nn.Linear(256, LATENT_DIM)
        self.decoder   = nn.Sequential(
            nn.Linear(LATENT_DIM, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512),        nn.LeakyReLU(0.2),
            nn.Linear(512, n_total),    nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        sigma = (0.5 * logvar).exp()
        return mu + torch.randn_like(sigma) * sigma * NOISE_SCALE

    def forward(self, x):
        h      = self.encoder(x)
        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z_vae  = self.reparameterize(mu, logvar)
        x_hat  = self.decoder(z_vae)
        return x_hat, mu, logvar, z_vae


class Generator(nn.Module):
    def __init__(self, n_total):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(LATENT_DIM, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
            nn.Linear(128, 256),        nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512),        nn.BatchNorm1d(512), nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),       nn.BatchNorm1d(1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, n_total),   nn.Sigmoid(),
        )
    def forward(self, z):
        return self.model(z)


def load_real_loader(cfg, target_class, batch_size=512):
    fpath = os.path.join(cfg["real_dir"], cfg["train_file"])
    df    = pd.read_csv(fpath)
    if target_class == "ssh":
        df = df[df[cfg["label_col"]] == SSH_LABEL]
    else:
        df = df[df[cfg["label_col"]] != SSH_LABEL]
    x = np.concatenate([
        df[cfg["feature_cols"]].astype(float).values,
        (df[cfg["label_col"]] == SSH_LABEL).astype(float).values.reshape(-1, 1),
    ], axis=1)
    return DataLoader(TensorDataset(torch.tensor(x, dtype=torch.float32)),
                      batch_size=batch_size, shuffle=True)


# ══════════════════════════════════════════════════════
# 단일 (dataset, mode) 조합 생성
# ══════════════════════════════════════════════════════

def generate_for_combo(cfg, mode, opt):
    n_total  = cfg["N_TOTAL"]
    n_feat   = cfg["N_FEATURES"]
    mode_dir = MODEL_DIR[mode]

    ckpt_list = sorted(glob.glob(os.path.join(cfg["ckpt_dir"], mode_dir, "generator_*.pt")))
    if not ckpt_list:
        print(f"  [경고] 체크포인트 없음: {mode_dir} (건너뜀) — model_generator_batch.py 먼저 실행 필요")
        return

    # vae_wgan_gp 모드만 VAE 필요
    vae = None
    if mode == "vae_wgan_gp":
        vae_ckpt_path = os.path.join(cfg["ckpt_dir"], "vae_checkpoint.pt")
        vae = VAE(n_total).to(DEVICE).eval()
        if os.path.exists(vae_ckpt_path):
            vae.load_state_dict(torch.load(vae_ckpt_path, map_location=DEVICE))
            print(f"  VAE 가중치 로드: {vae_ckpt_path}")
        else:
            print(f"  ※ VAE 가중치 없음({vae_ckpt_path}) → 랜덤 초기화 상태로 진행")

    real_loader = load_real_loader(cfg, opt.target_class)
    real_iter   = iter(real_loader)

    pool_feats, pool_labels, pool_lcl, pool_ckpts = [], [], [], []
    batch_cnt = 0

    print(f"\n  >>> 생성 시작 (mode={mode_dir})  목표: {opt.n_samples:,}개")

    while sum(len(f) for f in pool_feats) < opt.n_samples * 2:
        ckpt_path = random.choice(ckpt_list)
        G = Generator(n_total).to(DEVICE).eval()
        G.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))

        try:
            (x_real,) = next(real_iter)
        except StopIteration:
            real_iter = iter(real_loader)
            (x_real,) = next(real_iter)
        x_real = x_real.to(DEVICE)

        with torch.no_grad():
            if mode == "vae_wgan_gp":
                x_hat, mu, logvar, z_vae = vae(x_real)
                x_tilde  = G(z_vae)
                lcl_vals = ((x_tilde - x_hat) ** 2).mean(dim=1).cpu().numpy()
            else:   # wgan_gp
                z_in     = torch.randn(x_real.size(0), LATENT_DIM, device=DEVICE)
                x_tilde  = G(z_in)
                lcl_vals = np.zeros(x_tilde.size(0))   # VAE 없음 → LCL 정의 안 됨 (0으로 채움)

        x_np = x_tilde.cpu().numpy()

        # ── softmax(label) 기준 필터 (두 모드 공통) ─────────────
        label_prob = x_np[:, n_feat]
        if opt.target_class == "ssh":
            mask = label_prob >= opt.softmax_thresh
        else:
            mask = (1 - label_prob) >= opt.softmax_thresh

        x_sel   = x_np[mask]
        lcl_sel = lcl_vals[mask]
        if len(x_sel) > 0:
            pool_feats.append(x_sel[:, :n_feat])
            pool_labels.append(x_sel[:, n_feat])
            pool_lcl.append(lcl_sel)
            pool_ckpts.extend([os.path.basename(ckpt_path)] * len(x_sel))

        batch_cnt += 1
        total_now = sum(len(f) for f in pool_feats)
        if batch_cnt % 20 == 0:
            print(f"    배치 {batch_cnt:>4}  누적 {total_now:>8,}개")
        if total_now >= opt.n_samples * 2:
            break

    if not pool_feats:
        print(f"  [경고] 선택된 샘플이 없습니다 (softmax_thresh={opt.softmax_thresh} 너무 높을 수 있음)")
        return

    all_feats  = np.concatenate(pool_feats,  axis=0)
    all_lcl    = np.concatenate(pool_lcl,    axis=0)
    idx = np.random.permutation(len(all_feats))[:opt.n_samples]

    df_out = pd.DataFrame(all_feats[idx], columns=cfg["feature_cols"])
    df_out[cfg["label_col"]] = SSH_LABEL if opt.target_class == "ssh" else NON_SSH_LABEL
    df_out["lcl_loss"]       = all_lcl[idx]
    df_out["generator_ckpt"] = [pool_ckpts[i] for i in idx]

    out_dir  = os.path.join(cfg["fake_dir"], mode_dir)
    os.makedirs(out_dir, exist_ok=True)
    fname    = f"synthetic_{opt.target_class.replace('-', '_')}.csv"
    out_path = os.path.join(out_dir, fname)
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  → 저장 완료: {out_path}  ({len(df_out):,}행)")


# ══════════════════════════════════════════════════════
# 배치 실행 — 4개 조합 순차 생성
# ══════════════════════════════════════════════════════

def run_batch(opt):
    total = len(DATASETS) * len(MODES)
    done  = 0

    print(f"\n{'='*70}")
    print(f"  generate_traffic_batch.py  v1.0")
    print(f"  device     : {DEVICE}")
    print(f"  대상 조합  : {total}개  (datasets={DATASETS}, modes={MODES})")
    print(f"  생성 클래스: {opt.target_class}")
    print(f"{'='*70}")

    for dataset in DATASETS:
        cfg = DATASET_CONFIG[dataset]
        for mode in MODES:
            done += 1
            print(f"\n\n{'#'*70}")
            print(f"  [{done}/{total}]  dataset={dataset.upper()}  mode={mode}")
            print(f"{'#'*70}")
            generate_for_combo(cfg, mode, opt)

    print(f"\n{'='*70}")
    print(f"  전체 배치 완료  ({done}/{total})")
    print(f"{'='*70}")
    for dataset in DATASETS:
        for mode in MODES:
            mode_dir = MODEL_DIR[mode]
            fname    = f"synthetic_{opt.target_class.replace('-', '_')}.csv"
            out_path = os.path.join(DATASET_CONFIG[dataset]["fake_dir"], mode_dir, fname)
            print(f"  {dataset:<8} {mode:<14} → {out_path}")


if __name__ == "__main__":
    opt = get_params()
    run_batch(opt)