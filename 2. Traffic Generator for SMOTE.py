# -*- coding: utf-8 -*-
"""
Week 4 학습 데이터로 SMOTE 합성 세션 피처 50,000개 생성

- 전체 학습 데이터와 라벨을 SMOTE에 입력
- 양성 표본 사이에서 보간
- 원본 행을 제외한 합성 행만 저장
- QG 및 중복 제거 미적용
"""

import json
import platform
from pathlib import Path

import imblearn
import numpy as np
import pandas as pd
import sklearn
from imblearn.over_sampling import SMOTE


# ============================================================
# 설정 — 실행 환경에 맞게 수정
# ============================================================

INPUT_CSV = "dataset-DARPA99/training dataset_week4_normalized.csv"

OUTPUT_CSV = "fake dataset-DARPA99/smote/synthetic_ssh.csv"
OUTPUT_JSON = "fake dataset-DARPA99/smote/generation_info.json"

LABEL_COL = "LABEL"

# 기존 학습·시험·탐지 코드와 동일한 라벨 사용
POSITIVE_LABEL = "ssh"
NEGATIVE_LABEL = "non-ssh"

N_SYNTHETIC = 50_000
K_NEIGHBORS = 5
RANDOM_SEED = 42

FEATURE_COLUMNS = [
    "log_time_interval_mean",
    "log_time_interval_var",
    "log_time_taken",
    "log_cs_byte",
    "log_ratio_trans_receive",
    "log_count_connect_IP",
    "log_count_total_connect",
    "log_avg_count_connect",
    "log_transmit_speed_BPS",
]


def main():
    input_path = Path(INPUT_CSV)
    output_path = Path(OUTPUT_CSV)
    metadata_path = Path(OUTPUT_JSON)

    if N_SYNTHETIC <= 0:
        raise ValueError("N_SYNTHETIC은 양수여야 합니다")

    if K_NEIGHBORS < 1:
        raise ValueError("K_NEIGHBORS는 1 이상이어야 합니다")

    # 1. 입력 데이터 로드
    df = pd.read_csv(input_path, low_memory=False)

    required = FEATURE_COLUMNS + [LABEL_COL]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"필수 열 누락: {missing}")

    # 2. 라벨 검증
    if df[LABEL_COL].isna().any():
        raise ValueError("라벨에 결측값이 있습니다")

    y = df[LABEL_COL].astype(str)

    expected_labels = {POSITIVE_LABEL, NEGATIVE_LABEL}
    actual_labels = set(y.unique())

    if actual_labels != expected_labels:
        raise ValueError(
            f"라벨 불일치: 실제={actual_labels}, 예상={expected_labels}"
        )

    # 3. 피처 검증 — 결측 행을 자동으로 삭제하지 않음
    features = df[FEATURE_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )
    X = features.to_numpy(dtype=np.float64)

    invalid = ~np.isfinite(X)
    if invalid.any():
        bad_columns = [
            name
            for index, name in enumerate(FEATURE_COLUMNS)
            if invalid[:, index].any()
        ]
        raise ValueError(
            f"결측값·무한값·숫자가 아닌 값이 포함된 피처: {bad_columns}"
        )

    n_original = len(df)
    n_positive = int((y == POSITIVE_LABEL).sum())
    n_negative = int((y == NEGATIVE_LABEL).sum())

    if n_positive <= K_NEIGHBORS:
        raise ValueError(
            f"양성 표본 {n_positive}개로는 "
            f"k_neighbors={K_NEIGHBORS}를 사용할 수 없습니다"
        )

    print(f"학습 데이터 전체: {n_original:,}")
    print(f"원본 양성: {n_positive:,}")
    print(f"원본 음성: {n_negative:,}")
    print(f"추가 생성 목표: {N_SYNTHETIC:,}")

    # 4. 양성 클래스만 지정된 개수만큼 증강
    smote = SMOTE(
        sampling_strategy={
            POSITIVE_LABEL: n_positive + N_SYNTHETIC,
        },
        k_neighbors=K_NEIGHBORS,
        random_state=RANDOM_SEED,
    )

    X_resampled, y_resampled = smote.fit_resample(
        X,
        y.to_numpy(),
    )

    # 5. 기본 SMOTE가 보존한 원본 접두부 확인 후 추가 행 분리
    if not np.array_equal(X_resampled[:n_original], X):
        raise RuntimeError("증강 결과의 원본 행 순서가 예상과 다릅니다")

    if not np.array_equal(
        y_resampled[:n_original],
        y.to_numpy(),
    ):
        raise RuntimeError("증강 결과의 원본 라벨 순서가 예상과 다릅니다")

    X_synthetic = X_resampled[n_original:]
    y_synthetic = y_resampled[n_original:]

    if len(X_synthetic) != N_SYNTHETIC:
        raise RuntimeError(
            f"합성 개수 불일치: {len(X_synthetic)} != {N_SYNTHETIC}"
        )

    if not np.all(y_synthetic == POSITIVE_LABEL):
        raise RuntimeError("합성 결과에 목표 클래스 외 라벨이 포함됐습니다")

    if not np.isfinite(X_synthetic).all():
        raise RuntimeError("합성 결과에 유효하지 않은 값이 있습니다")

    # SMOTE 보간 결과가 원본 양성 피처 범위 안에 있는지 확인
    positive_X = X[y.to_numpy() == POSITIVE_LABEL]
    tolerance = 1e-10

    if (
        (X_synthetic < positive_X.min(axis=0) - tolerance).any()
        or (X_synthetic > positive_X.max(axis=0) + tolerance).any()
    ):
        raise RuntimeError("합성 결과가 원본 양성 피처 범위를 벗어났습니다")

    synthetic_df = pd.DataFrame(
        X_synthetic,
        columns=FEATURE_COLUMNS,
    )
    synthetic_df[LABEL_COL] = y_synthetic

    # 6. 합성 CSV 저장
    output_path.parent.mkdir(parents=True, exist_ok=True)
    synthetic_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    # 가능한 경우 원본 프로토콜 구성 기록
    positive_protocol_counts = None
    if "LABEL_original" in df.columns:
        positive_protocol_counts = {
            str(label): int(count)
            for label, count in df.loc[
                y == POSITIVE_LABEL, "LABEL_original"
            ].value_counts(dropna=False).items()
        }

    metadata = {
        "method": "SMOTE",
        "input_csv": str(input_path.resolve()),
        "output_csv": str(output_path.resolve()),
        "feature_columns": FEATURE_COLUMNS,
        "label_column": LABEL_COL,
        "positive_label": POSITIVE_LABEL,
        "negative_label": NEGATIVE_LABEL,
        "original_total": n_original,
        "original_positive": n_positive,
        "original_negative": n_negative,
        "positive_protocol_counts": positive_protocol_counts,
        "synthetic_count": len(synthetic_df),
        "synthetic_duplicate_rows": int(
            synthetic_df.duplicated(subset=FEATURE_COLUMNS).sum()
        ),
        "k_neighbors": K_NEIGHBORS,
        "random_seed": RANDOM_SEED,
        "quality_gate_applied": False,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "imbalanced_learn": imblearn.__version__,
        },
    }

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"합성 CSV 저장: {output_path}")
    print(f"생성 정보 저장: {metadata_path}")
    print(f"합성 행 수: {len(synthetic_df):,}")


if __name__ == "__main__":
    main()