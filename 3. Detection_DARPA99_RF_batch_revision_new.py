# -*- coding: utf-8 -*-
"""DARPA99 RF 일괄 평가 — 기존 common_config.py와 같은 폴더에 저장 후 실행
python 3_Detection_DARPA99_batch_revision.py
python 3_Detection_DARPA99_batch_revision.py --iterations 5

LLM delta, QG, feedback 조건을 각각 구분하여 평가
original은 기본 포함, --skip-original 옵션으로 제외
출력 지표는 ssh 양성 클래스 기준, 0~1 단위, 표본 SD(ddof=1)
고정 합성 풀의 샘플링 및 RF 학습 변동성 평가
"""
from pathlib import Path
from datetime import datetime
import argparse
import json
import hashlib
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from common_config import DATASET_CONFIG, SSH_LABEL, NON_SSH_LABEL

BASE_DIR = Path(__file__).resolve().parent
N_ESTIMATORS = 100
N_ITER = 50
BASE_SEED = 42
DUMMY_IP_PORT = "255.255.255.255:1"
SMOTE_WGAN_COUNTS = [10, 500, 1000]
LLM_COUNTS = [10, 50, 100, 200, 300, 500, 700, 1000]
# fake_dir 기준 상대 경로 — 실제 QG 파일명에 맞게 아래 두 경로 수정
SYNTHETIC_FILES = {
    #'smote':'smote/synthetic_ssh.csv',
    #'smote-qg': 'smote-qg/synthetic_ssh_filtered.csv',
    #'wgan-gp':'wgan-gp/synthetic_ssh.csv',
    #'wgan-gp-qg':'wgan-gp-qg/synthetic_ssh_filtered.csv',
    #'llm_nodelta':'2_1_llm_without delta/synthetic_ssh.csv',
    #'llm_delta':'2_2_llm_delta/synthetic_ssh.csv',
    #'llm_deltal_qg': '2_3_llm_delta_with qg/synthetic_ssh_filtered_500.csv',
    'llm_delta_qg_feedback': '2_4_llm_delta_with qg and feedback/synthetic_ssh_filtered.csv',

}
CONDITIONS = [
    ("original", "original", "N/A", [0]),
    ("smote", "SMOTE", "off", SMOTE_WGAN_COUNTS),
    ("smote-qg", "SMOTE", "on", SMOTE_WGAN_COUNTS),
    ("wgan-gp", "WGAN-GP", "off", SMOTE_WGAN_COUNTS),
    ("wgan-gp-qg", "WGAN-GP", "on", SMOTE_WGAN_COUNTS),
    ("llm_nodelta", "LLM without delta", "off", LLM_COUNTS),
    ("llm_delta", "LLM delta", "off", LLM_COUNTS),
    ("llm_deltal_qg", "LLM delta", "on", LLM_COUNTS),
    ("llm_delta_qg_feedback", "LLM delta + feedback", "on", LLM_COUNTS),
]
METRICS = ["precision", "recall", "f1", "fpr"]


def evaluate_quality(real, pools, cfg, output, seeds, n_sample=1000, bins_count=10, plot=True):
    """기존 JSD(natural log), VEE(합성 Shannon entropy, log2) 정의 유지
    모든 비교에 공통 feature별 bin 사용, 반복별 동일 원본 표본 사용
    VEE는 기존 구현 호환명이며 엔트로피 차이나 분산을 뜻하지 않음
    """
    from scipy.stats import entropy
    features = cfg["feature_cols"]
    positives = real[real[cfg["label_col"]] == str(SSH_LABEL)]
    eligible = {key: df for key, df in pools.items() if len(df) >= n_sample}
    status = [{"condition": key, "available": len(df), "requested": n_sample,
               "status": "completed" if len(df) >= n_sample else "insufficient"}
              for key, df in pools.items()]
    pd.DataFrame(status).to_csv(output / "quality_status.csv", index=False, encoding="utf-8-sig")
    if len(positives) < n_sample or not eligible:
        print("품질 평가 건너뜀: 원본 양성 또는 합성 데이터 부족")
        return
    arrays = [positives[features].to_numpy()] + [df[features].to_numpy() for df in eligible.values()]
    edges = []
    for fi in range(len(features)):
        lo = min(x[:, fi].min() for x in arrays)
        hi = max(x[:, fi].max() for x in arrays)
        edges.append(np.linspace(lo, hi, bins_count+1) if hi > lo else np.array([lo-1e-6, hi+1e-6]))
    pd.DataFrame({features[i]: pd.Series(e) for i, e in enumerate(edges)}).to_csv(output / "quality_bins.csv", index=False)
    def probability(values, edge):
        counts = np.histogram(values, bins=edge)[0].astype(float) + 1e-10
        return counts/counts.sum()
    rows = []
    for iteration, seed in enumerate(seeds, 1):
        ref = positives.sample(n_sample, random_state=seed)[features].to_numpy()
        for key, df in eligible.items():
            syn = df.sample(n_sample, random_state=seed)[features].to_numpy()
            for fi, feature in enumerate(features):
                p, q = probability(ref[:, fi], edges[fi]), probability(syn[:, fi], edges[fi])
                midpoint = (p+q)/2
                rows.append(dict(condition=key, iteration=iteration, seed=seed, feature=feature,
                                 JSD=float((entropy(p, midpoint)+entropy(q, midpoint))/2),
                                 VEE=float(entropy(q, base=2)),
                                 reference_entropy=float(entropy(p, base=2))))
    detail = pd.DataFrame(rows)
    detail.to_csv(output / "quality_feature_runs.csv", index=False, encoding="utf-8-sig")
    runs = detail.groupby(["condition", "iteration", "seed"], sort=False)[["JSD", "VEE", "reference_entropy"]].mean().reset_index()
    runs.to_csv(output / "quality_runs.csv", index=False, encoding="utf-8-sig")
    summary = runs.groupby("condition", sort=False)[["JSD", "VEE"]].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(output / "quality_summary.csv", index=False, encoding="utf-8-sig")
    if not plot:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    groups = {"Original SSH": positives.sample(n_sample, random_state=seeds[0])[features].to_numpy()}
    negatives = real[real[cfg["label_col"]] == str(NON_SSH_LABEL)]
    if len(negatives) >= n_sample:
        groups["Original Non-SSH"] = negatives.sample(n_sample, random_state=seeds[0])[features].to_numpy()
    groups.update({key: df.sample(n_sample, random_state=seeds[0])[features].to_numpy() for key, df in eligible.items()})
    all_x = np.vstack(list(groups.values()))
    xy = TSNE(n_components=2, random_state=seeds[0], perplexity=min(40, len(all_x)-1),
              max_iter=1000).fit_transform(all_x)
    labels = np.concatenate([np.repeat(key, len(x)) for key, x in groups.items()])
    pd.DataFrame({"group": labels, "x": xy[:, 0], "y": xy[:, 1]}).to_csv(output / "tsne_coordinates.csv", index=False)
    fig, ax = plt.subplots(figsize=(11, 8))
    for key in groups:
        mask = labels == key
        ax.scatter(xy[mask, 0], xy[mask, 1], s=12, alpha=.5, label=key)
    ax.legend(); ax.set_title(f"Joint t-SNE (n={n_sample}/group, seed={seeds[0]})")
    fig.tight_layout(); fig.savefig(output / "tsne_comparison.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric in zip(axes, ["JSD", "VEE"]):
        ax.bar(summary.index, summary[metric+"_mean"], yerr=summary[metric+"_std"], capsize=4)
        ax.set_title("JSD (natural log)" if metric == "JSD" else "Synthetic Shannon entropy (bits)")
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout(); fig.savefig(output / "quality_metrics.png", dpi=150); plt.close(fig)


def export_excel(output):
    from openpyxl.styles import Font, PatternFill, Alignment
    csvs = {"Summary": output / "summary_numeric.csv", "Raw_Data": output / "all_runs.csv",
            "Status": output / "condition_status.csv", "Quality": output / "quality_summary.csv"}
    for path in output.glob("comparison_*.csv"):
        csvs[f"Compare{len(csvs)-3}"] = path
    with pd.ExcelWriter(output / "pivot_detection_summary.xlsx", engine="openpyxl") as writer:
        for sheet, path in csvs.items():
            if path.exists():
                pd.read_csv(path, keep_default_na=False).to_excel(writer, sheet_name=sheet, index=False)
        for sheet in writer.sheets.values():
            sheet.freeze_panes = "C2"
            sheet.auto_filter.ref = sheet.dimensions
            for cell in sheet[1]:
                cell.font = Font(bold=True)
                cell.fill = PatternFill("solid", fgColor="EBF1DE")
                cell.alignment = Alignment(horizontal="center")
            for column in sheet.columns:
                width = max(len(str(cell.value or "")) for cell in column)
                sheet.column_dimensions[column[0].column_letter].width = min(65, width+3)


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else BASE_DIR / path


def fingerprint(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_data(path, cfg, synthetic=False):
    df = pd.read_csv(path)
    features, label = cfg["feature_cols"], cfg["label_col"]
    missing = set(features) - set(df.columns)
    if missing:
        raise ValueError(f"필수 feature 누락: {sorted(missing)}")
    for feature in features:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")
    valid = np.isfinite(df[features].to_numpy(dtype=float)).all(axis=1)
    print(f"  {path.name}: 전체 {len(df):,}, 숫자 검증 제외 {(~valid).sum():,}")
    df = df.loc[valid].copy()
    if synthetic:
        # 라벨이 있으면 양성만 사용, 없는 경우 양성 생성 파일로 취급
        if label in df:
            positive = df[label].astype(str) == str(SSH_LABEL)
            print(f"  합성 음성/기타 라벨 제외: {(~positive).sum():,}")
            df = df.loc[positive].copy()
        df[label] = str(SSH_LABEL)
    else:
        if label not in df:
            raise ValueError(f"원본 라벨 컬럼 누락: {label}")
        df[label] = df[label].astype(str)
        if not df[label].isin([str(SSH_LABEL), str(NON_SSH_LABEL)]).all():
            raise ValueError("원본 데이터에 미지정 또는 결측 라벨 존재")
    if df.empty:
        raise ValueError("검증 후 유효 데이터 없음")
    return df.reset_index(drop=True)


def majority_groups(df):
    if not {"dstip", "dsport"}.issubset(df.columns):
        print("  목적지 컬럼 없음: 다수결 결과는 raw와 동일")
        return []
    keys = df["dstip"].astype(str) + ":" + df["dsport"].astype(str)
    return [np.asarray(indices) for key, indices in keys.groupby(keys).groups.items()
            if key != DUMMY_IP_PORT]


def propagate_majority(pred, groups):
    pred = pred.copy()
    for indices in groups:
        positives = np.count_nonzero(pred[indices] == str(SSH_LABEL))
        pred[indices] = str(SSH_LABEL) if positives > len(indices) - positives else str(NON_SSH_LABEL)
    return pred


def metrics(y_true, y_pred):
    actual = y_true == str(SSH_LABEL)
    predicted = y_pred == str(SSH_LABEL)
    tp = int((actual & predicted).sum())
    fp = int((~actual & predicted).sum())
    fn = int((actual & ~predicted).sum())
    tn = int((~actual & ~predicted).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return dict(precision=p, recall=r, f1=2*p*r/(p+r) if p+r else 0.0,
                fpr=fp/(fp+tn) if fp+tn else 0.0, tp=tp, fp=fp, fn=fn, tn=tn)


def save_tables(records, statuses, output):
    pd.DataFrame(statuses).to_csv(output / "condition_status.csv", index=False, encoding="utf-8-sig")
    if not records:
        return
    frame = pd.DataFrame(records)
    frame.to_csv(output / "all_runs.csv", index=False, encoding="utf-8-sig")
    keys = ["test", "condition", "generator", "qg", "n_fake", "prediction"]
    summary = frame.groupby(keys, sort=False)[METRICS].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(output / "summary_numeric.csv", index=False, encoding="utf-8-sig")
    for test in dict.fromkeys(s["test"] for s in statuses):
        for prediction in ["rf_raw", "rf_majority"]:
            rows = []
            test_status = [s for s in statuses if s["test"] == test]
            for generator, n in dict.fromkeys((s["generator"], s["n_fake"]) for s in test_status):
                row = {"generator": generator, "n_fake": n}
                for gate, suffix in [("off", "without_qg"), ("on", "with_qg")]:
                    candidates = [s for s in test_status if s["generator"] == generator
                                  and s["n_fake"] == n and s["qg"] == gate]
                    if generator == "original" and gate == "off":
                        candidates = [s for s in test_status if s["generator"] == "original"]
                    for metric in ["precision", "recall", "f1"]:
                        value = "N/A"
                        if candidates:
                            status = candidates[0]
                            if status["status"] != "completed":
                                value = f"SKIPPED: {status['reason']}"
                            else:
                                match = summary[(summary.test == test) &
                                                (summary.condition == status["condition"]) &
                                                (summary.n_fake == n) & (summary.prediction == prediction)]
                                if len(match):
                                    item = match.iloc[0]
                                    value = f"{item[metric+'_mean']:.4f} ± {item[metric+'_std']:.4f}"
                        row[f"{metric}_{suffix}"] = value
                rows.append(row)
            safe_test = str(test).replace("/", "_").replace("\\", "_")
            pd.DataFrame(rows).to_csv(output / f"comparison_{safe_test}_{prediction}.csv",
                                      index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    original_group = parser.add_mutually_exclusive_group()
    original_group.add_argument("--skip-original", dest="include_original", action="store_false",
                                help="원본 데이터만 학습하는 original 조건 제외")
    original_group.add_argument("--include-original", dest="include_original", action="store_true",
                                help="original 조건 포함 (기본값)")
    parser.set_defaults(include_original=True)
    parser.add_argument("--iterations", type=int, default=N_ITER)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--paths-json", help='JSON 객체로 SYNTHETIC_FILES 경로 덮어쓰기')
    parser.add_argument("--skip-detection", action="store_true", help="품질 평가만 실행")
    parser.add_argument("--skip-quality", action="store_true", help="탐지 평가만 실행")
    parser.add_argument("--no-plots", action="store_true", help="품질 수치만 계산")
    parser.add_argument("--quality-samples", type=int, default=1000)
    args = parser.parse_args()
    if args.iterations < 2:
        parser.error("표본 SD 계산을 위해 iterations >= 2 필요")
    if args.quality_samples < 2:
        parser.error("quality-samples >= 2 필요")
    conditions = [c for c in CONDITIONS if args.include_original or c[0] != "original"]
    cfg = DATASET_CONFIG["darpa99"]
    paths = SYNTHETIC_FILES.copy()
    if args.paths_json:
        with open(resolve(args.paths_json), encoding="utf-8") as stream:
            overrides = json.load(stream)
        if not isinstance(overrides, dict) or set(overrides) - set(paths):
            raise ValueError("paths JSON 키는 SYNTHETIC_FILES 키만 허용")
        paths.update(overrides)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = resolve(cfg["result_dir"]) / "batch_revision" / run_id
    output.mkdir(parents=True, exist_ok=False)
    seeds = [args.base_seed + i for i in range(args.iterations)]
    real_path = resolve(cfg["real_dir"]) / cfg["train_file"]
    real = load_data(real_path, cfg)
    manifest = dict(run_id=run_id, config=cfg, seeds=seeds, n_estimators=N_ESTIMATORS,
                    conditions=conditions, synthetic_paths=paths,
                    train_sha256=fingerprint(real_path), input_files={},
                    metric_scale="0..1", std_ddof=1,
                    note="Separate delta/QG/feedback conditions; fixed-pool sampling and RF variability")
    pools, failures = {}, {}
    for condition, filename in paths.items():
        path = resolve(cfg["fake_dir"]) / filename
        try:
            pools[condition] = load_data(path, cfg, synthetic=True)
            manifest["input_files"][condition] = dict(path=str(path), sha256=fingerprint(path),
                                                     valid_rows=len(pools[condition]))
        except (OSError, ValueError) as exc:
            failures[condition] = str(exc)
            print(f"[{condition}] 사용 불가: {exc}")
    records, statuses = [], []
    start = time.perf_counter()
    features = cfg["feature_cols"]
    x_real = real[features].to_numpy(dtype=np.float32)
    y_real = real[cfg["label_col"]].to_numpy(dtype=str)
    with open(output / "run_config.json", "w", encoding="utf-8") as stream:
        json.dump({**manifest, "arguments": vars(args)}, stream, ensure_ascii=False, indent=2, default=str)
    for test, filename in ([] if args.skip_detection else cfg["test_files"].items()):
        test_path = resolve(cfg["real_dir"]) / filename
        test_df = load_data(test_path, cfg)
        manifest["input_files"][f"test:{test}"] = dict(path=str(test_path), sha256=fingerprint(test_path))
        x_test = test_df[features].to_numpy(dtype=np.float32)
        y_test = test_df[cfg["label_col"]].to_numpy(dtype=str)
        groups = majority_groups(test_df)
        for condition, generator, gate, counts in conditions:
            for n in counts:
                status = dict(test=test, condition=condition, generator=generator, qg=gate,
                              n_fake=n, status="completed", reason="")
                if n and condition in failures:
                    status.update(status="skipped", reason=failures[condition])
                elif n and condition not in pools:
                    status.update(status="skipped", reason="synthetic path not configured or pool unavailable")
                elif n and n > len(pools[condition]):
                    status.update(status="skipped", reason=f"insufficient rows {len(pools[condition])} < {n}")
                if status["status"] == "skipped":
                    statuses.append(status)
                    print(f"[{test}] {condition} n={n}: {status['reason']}")
                    save_tables(records, statuses, output)
                    continue
                for iteration, seed in enumerate(seeds, 1):
                    if n:
                        sample = pools[condition].sample(n=n, replace=False, random_state=seed)
                        x_train = np.concatenate([x_real, sample[features].to_numpy(dtype=np.float32)])
                        y_train = np.concatenate([y_real, np.full(n, str(SSH_LABEL))])
                    else:
                        x_train, y_train = x_real, y_real
                    model = RandomForestClassifier(n_estimators=N_ESTIMATORS, n_jobs=-1, random_state=seed)
                    model.fit(x_train, y_train)
                    raw = model.predict(x_test).astype(str)
                    for prediction, pred in [("rf_raw", raw), ("rf_majority", propagate_majority(raw, groups))]:
                        records.append(dict(test=test, condition=condition, generator=generator,
                                            qg=gate, n_fake=n, prediction=prediction,
                                            iteration=iteration, sample_seed=seed, model_seed=seed,
                                            **metrics(y_test, pred)))
                    print(f"[{test}] {condition} n={n} {iteration}/{len(seeds)} F1={records[-1]['f1']:.4f}", flush=True)
                    # 각 반복 저장 — 중단되더라도 완료된 반복 보존
                    pd.DataFrame(records).to_csv(output / "all_runs.csv", index=False, encoding="utf-8-sig")
                statuses.append(status)
                save_tables(records, statuses, output)
        with open(output / "run_config.json", "w", encoding="utf-8") as stream:
            json.dump({**manifest, "arguments": vars(args)}, stream, ensure_ascii=False, indent=2, default=str)
    if not args.skip_quality:
        evaluate_quality(real, pools, cfg, output, seeds, args.quality_samples, plot=not args.no_plots)
    export_excel(output)
    print(f"완료: {output}\n소요 시간: {(time.perf_counter()-start)/60:.1f}분")


if __name__ == "__main__":
    main()
