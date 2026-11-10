# -*- coding: utf-8 -*-
"""DCR evaluation for normalized DARPA99 feature vectors.
DCR(s) = min_r ||s-r||_2; uses all valid original training SSH rows.
No new scaling is fitted: input CSVs must share the same normalization.
Zero DCR means identical feature vectors, not necessarily identical raw traffic.
DCR is a proximity diagnostic, not a privacy guarantee.
Run: python 5_Evaluate_DCR.py
Optional held-out baseline: python 5_Evaluate_DCR.py --with-holdout
Dependencies: numpy pandas scipy matplotlib
"""
from pathlib import Path
from datetime import datetime
import argparse
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

BASE_DIR = Path(__file__).resolve().parent
REAL_CSV = 'dataset-DARPA99/training dataset_week4_normalized.csv'
SYNTHETIC_CSV = 'fake dataset-DARPA99/2_4_llm_delta_with qg and feedback/synthetic_ssh_filtered.csv'
HOLDOUT_CSV = 'dataset-DARPA99/test dataset_week1-3_normalized.csv'
LABEL_COL = 'LABEL'
POSITIVE_LABEL = 'ssh'
FEATURE_COLS = [
    'log_time_interval_mean', 'log_time_interval_var', 'log_time_taken',
    'log_cs_byte', 'log_ratio_trans_receive', 'log_count_connect_IP',
    'log_count_total_connect', 'log_avg_count_connect', 'log_transmit_speed_BPS',
]
BATCH_SIZE = 1000

def path(value):
    p = Path(value)
    return p if p.is_absolute() else BASE_DIR / p

def load_features(filename, positive_only):
    p = path(filename)
    if not p.is_file():
        raise FileNotFoundError(f'입력 파일 없음: {p}')
    df = pd.read_csv(p)
    missing = [c for c in FEATURE_COLS if c not in df]
    if positive_only and LABEL_COL not in df:
        missing.append(LABEL_COL)
    if missing:
        raise ValueError(f'{p.name}: 필수 컬럼 없음 {missing}')
    n_total = len(df)
    if positive_only:
        df = df[df[LABEL_COL].astype(str).str.strip().str.lower() == POSITIVE_LABEL.lower()]
    selected = len(df)
    numeric = df[FEATURE_COLS].apply(pd.to_numeric, errors='coerce')
    valid = np.isfinite(numeric.to_numpy(dtype=np.float64)).all(axis=1)
    # Preserve input row position (header excluded, zero based) for nearest-row tracing.
    indices = df.index.to_numpy()[valid]
    x = numeric.loc[valid].to_numpy(dtype=np.float64)
    if not len(x):
        raise ValueError(f'{p.name}: 유효한 대상 샘플 없음')
    print(f'[입력] {p.name}: 전체 {n_total:,}, 대상 {selected:,}, 유효 {len(x):,}, 제외 {selected-len(x):,}', flush=True)
    return x, indices, {'path': str(p), 'total_rows': n_total, 'selected_rows': selected,
                        'valid_rows': len(x), 'invalid_rows': selected-len(x)}

def calculate_dcr(query, reference, query_rows, reference_rows, condition):
    if BATCH_SIZE < 1:
        raise ValueError('BATCH_SIZE는 1 이상이어야 합니다')
    tree = cKDTree(reference)
    reference_keys = {tuple(row) for row in reference}
    frames = []
    for start in range(0, len(query), BATCH_SIZE):
        batch = query[start:start+BATCH_SIZE]
        distances, neighbors = tree.query(batch, k=1)
        frames.append(pd.DataFrame({
            'condition': condition,
            'query_row_index': query_rows[start:start+len(batch)],
            'nearest_training_row_index': reference_rows[neighbors],
            'DCR': distances,
            'exact_feature_match': [tuple(row) in reference_keys for row in batch],
        }))
        print(f'[DCR] {condition}: {min(start+len(batch),len(query)):,}/{len(query):,}', flush=True)
    return pd.concat(frames, ignore_index=True)

def summarize(frame, n_reference):
    d = frame.DCR.to_numpy()
    return {'condition': frame.condition.iloc[0], 'n_query': len(d),
            'n_reference': n_reference, 'min': float(d.min()),
            'p01': float(np.quantile(d,.01)), 'p05': float(np.quantile(d,.05)),
            'median': float(np.median(d)), 'mean': float(d.mean()),
            'sd': float(d.std(ddof=1)) if len(d)>1 else None,
            'p95': float(np.quantile(d,.95)), 'max': float(d.max()),
            'exact_feature_match_count': int(frame.exact_feature_match.sum()),
            'exact_feature_match_ratio': float(frame.exact_feature_match.mean())}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real', default=REAL_CSV)
    parser.add_argument('--synthetic', default=SYNTHETIC_CSV)
    parser.add_argument('--holdout', default=HOLDOUT_CSV)
    parser.add_argument('--with-holdout', action='store_true')
    args = parser.parse_args()
    print('[1/3] 입력 파일·컬럼·유효 값 확인', flush=True)
    real, ri, rm = load_features(args.real, True)
    syn, si, sm = load_features(args.synthetic, False)
    held = load_features(args.holdout, True) if args.with_holdout else None
    if path(args.real).resolve() == path(args.synthetic).resolve():
        raise ValueError('원본과 합성 입력 파일이 같습니다')
    if held and path(args.real).resolve() == path(args.holdout).resolve():
        raise ValueError('독립 평가용 원본은 학습 원본과 다른 파일이어야 합니다')
    out = BASE_DIR / 'result' / 'dcr' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    print('[2/3] 최근접 원본 거리 계산', flush=True)
    frames = [calculate_dcr(syn, real, si, ri, 'LLM_delta_qg_feedback')]
    if held:
        hx, hi, hm = held
        frames.append(calculate_dcr(hx, real, hi, ri, 'Held-out real SSH'))
    pd.concat(frames, ignore_index=True).to_csv(out/'dcr_per_sample.csv',index=False,encoding='utf-8-sig')
    summary = pd.DataFrame([summarize(f,len(real)) for f in frames])
    summary.to_csv(out/'dcr_summary.csv',index=False,encoding='utf-8-sig')
    metadata = {'metric': 'minimum Euclidean distance', 'features': FEATURE_COLS,
                'scaling': 'existing input normalization; no additional scaling',
                'reference': rm, 'synthetic': sm, 'holdout': held[2] if held else None,
                'row_index': 'zero-based data row position, excluding header',
                'exact_match': 'all selected feature values equal after float64 conversion',
                'interpretation': 'proximity diagnostic, not a privacy guarantee'}
    (out/'dcr_settings.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print('[3/3] 분포 그래프 저장', flush=True)
    fig, axes = plt.subplots(1,2,figsize=(12,5))
    names = [f.condition.iloc[0] for f in frames]
    axes[0].boxplot([f.DCR.to_numpy() for f in frames],showfliers=True)
    axes[0].set_xticks(range(1,len(names)+1),names)
    axes[0].set_ylabel('DCR (Euclidean distance)')
    axes[0].set_title('(a) DCR distribution')
    for f, name in zip(frames,names):
        d = np.sort(f.DCR.to_numpy())
        axes[1].step(d,np.arange(1,len(d)+1)/len(d),where='post',label=name)
    axes[1].set_xlabel('DCR (Euclidean distance)')
    axes[1].set_ylabel('Empirical cumulative probability')
    axes[1].set_title('(b) DCR ECDF')
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(axis='y',alpha=.3)
        ax.tick_params(axis='x',labelsize=8)
    fig.tight_layout()
    fig.savefig(out/'dcr_distribution.png',dpi=300,bbox_inches='tight')
    plt.close(fig)
    print(summary.to_string(index=False))
    print(f'[완료] {out}')

if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError) as exc:
        print(f'[오류] {exc}',file=sys.stderr)
        sys.exit(1)
