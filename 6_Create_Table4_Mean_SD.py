# -*- coding: utf-8 -*-
"""Table 4: Original → SMOTE → WGAN-GP → LLM, mean ± sample SD; Precision/Recall/F1/FPR.
설치: pip install pandas openpyxl
실행: python 6_Create_Table4_Mean_SD.py
추가 결과: python 6_Create_Table4_Mean_SD.py --append smote_qg_runs.csv
QG 단독 비교: python 6_Create_Table4_Mean_SD.py --llm-after llm_delta_qg
입력은 피봇이 아닌 반복별 원시 결과. csv/xlsx 추가 파일 모두 지원.
원본을 덮어쓰지 않고 result/table4/Table4_mean_sd.xlsx 생성.
"""
from pathlib import Path
import argparse
import warnings
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_XLSX = BASE_DIR / 'result/dataset-DARPA99-revision/total_detection.xlsx'
OUTPUT_XLSX = BASE_DIR / 'result/table4/Table4_mean_sd.xlsx'
LLM_AFTER = 'llm_delta_qg_feedback'  # 기존 Table 4 최종 구성; QG 단독 효과와 구분
EXPECTED_REPEATS = 50
DIGITS = 4
ALIASES = {'smote_qg': 'smote-qg', 'smote+qg': 'smote-qg',
           'wgan_gp': 'wgan-gp', 'wgan_gp_qg': 'wgan-gp-qg'}
REQUIRED = {'test', 'condition', 'n_fake', 'prediction', 'model_seed',
            'precision', 'recall', 'f1'}
METRICS = ['precision', 'recall', 'f1', 'fpr']


def read_runs(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'입력 파일 없음: {path}')
    if path.suffix.lower() == '.csv':
        frames = [pd.read_csv(path, encoding='utf-8-sig')]
    else:
        # 이름이 all_runs/rf_all_runs1 등이어도 원시 컬럼으로 자동 탐색
        frames = [d for d in pd.read_excel(path, sheet_name=None).values()
                  if REQUIRED.issubset(d.columns)]
    valid = [d for d in frames if REQUIRED.issubset(d.columns)]
    if not valid:
        raise ValueError(f'{path.name}: 반복별 원시 시트/컬럼 필요: {sorted(REQUIRED)}')
    return pd.concat(valid, ignore_index=True)


def prepare(paths):
    df = pd.concat([read_runs(p) for p in paths], ignore_index=True)
    # FPR은 실행별 값을 사용하고, 없는 경우 실행별 FP/TN으로 계산
    if 'fpr' not in df.columns:
        if not {'fp', 'tn'}.issubset(df.columns):
            raise ValueError('FPR 집계에는 실행별 fpr 또는 fp/tn 컬럼이 필요함')
        fp = pd.to_numeric(df['fp'], errors='raise')
        tn = pd.to_numeric(df['tn'], errors='raise')
        if fp.isna().any() or tn.isna().any() or (fp < 0).any() or (tn < 0).any() or ((fp + tn) <= 0).any():
            raise ValueError('FP/TN은 결측 없는 비음수이고 음성 샘플 수는 양수여야 함')
        df['fpr'] = fp / (fp + tn)
    elif df['fpr'].isna().any() and {'fp', 'tn'}.issubset(df.columns):
        missing = df['fpr'].isna()
        fp = pd.to_numeric(df.loc[missing, 'fp'], errors='raise')
        tn = pd.to_numeric(df.loc[missing, 'tn'], errors='raise')
        if fp.isna().any() or tn.isna().any() or (fp < 0).any() or (tn < 0).any() or ((fp + tn) <= 0).any():
            raise ValueError('FPR 누락 행의 유효한 FP/TN이 필요함')
        df.loc[missing, 'fpr'] = fp / (fp + tn)
    df['condition'] = df.condition.astype(str).str.strip().str.lower().replace(ALIASES)
    df['prediction'] = df.prediction.astype(str).str.strip().str.lower()
    df['test'] = df.test.astype(str).str.strip()
    # raw·majority는 중복 반복으로 합산하지 않음
    df = df[df.prediction.isin(['rf_raw', 'mlp_raw'])].copy()
    if df.empty:
        raise ValueError('rf_raw 또는 mlp_raw 결과가 없음')
    for col in ['n_fake', 'model_seed', *METRICS]:
        df[col] = pd.to_numeric(df[col], errors='raise')
        if df[col].isna().any():
            raise ValueError(f'{col}: 결측값 존재')
    if not (df.n_fake.ge(0) & df.n_fake.mod(1).eq(0)).all():
        raise ValueError('n_fake는 0 이상의 정수여야 함')
    df['n_fake'] = df.n_fake.astype(int)
    if not df[METRICS].apply(lambda x: x.between(0, 1)).all().all():
        raise ValueError('precision/recall/f1/fpr은 0~1 범위여야 함')
    keys = ['test', 'condition', 'n_fake', 'prediction', 'model_seed']
    # 같은 파일 재첨부의 동일값은 제거, 같은 시드의 다른값은 오류
    df = df.drop_duplicates(subset=keys + METRICS)
    dup = df[df.duplicated(keys, keep=False)]
    if not dup.empty:
        raise ValueError('동일 조건·수량·시드에 다른 점수 존재. 실행을 구분하거나 중복 제거 필요:\n'
                         + dup[keys + METRICS].head(8).to_string(index=False))
    return df


def summarize(df):
    rows = []
    for key, group in df.groupby(['test', 'prediction', 'condition', 'n_fake'], sort=False):
        count = len(group)
        if count != EXPECTED_REPEATS:
            warnings.warn(f'{key}: 반복 {count}회 (권장 {EXPECTED_REPEATS}회)')
        row = dict(zip(['test', 'prediction', 'condition', 'n_fake'], key))
        row['n_repeats'] = count
        for metric in METRICS:
            row[metric + '_mean'] = group[metric].mean()
            row[metric + '_sd'] = group[metric].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def make_table(summary, test, prediction, llm_after):
    part = summary[(summary.test == test) & (summary.prediction == prediction)]
    index = part.set_index(['condition', 'n_fake'])
    order = [('Original', 'original', None), ('SMOTE', 'smote', 'smote-qg'),
             ('WGAN-GP', 'wgan-gp', 'wgan-gp-qg'), ('LLM', 'llm_delta', llm_after)]
    rows = []
    for label, before, after in order:
        if before == 'original':
            sizes = [0]
        elif before in set(part.condition) or after in set(part.condition):
            sizes = sorted(set(part[part.condition.isin([before, after])].n_fake))
        else:
            continue
        for n in sizes:
            row = [label, 'N/A' if before == 'original' else n]
            for metric in METRICS:
                for cond in [before, after]:
                    if cond is None or (cond, n) not in index.index:
                        row.append('N/A')
                        continue
                    r = index.loc[(cond, n)]
                    mean, sd = r[metric + '_mean'], r[metric + '_sd']
                    row.append(f'{mean:.{DIGITS}f} ± {sd:.{DIGITS}f}'
                               if pd.notna(sd) else f'{mean:.{DIGITS}f} ± N/A')
            rows.append(row)
    return rows


def make_combined_table(summary, test, llm_after):
    rows = []
    for pred, model in [('rf_raw', 'RF'), ('mlp_raw', 'MLP')]:
        if pred not in set(summary[summary.test == test].prediction):
            continue
        for row in make_table(summary, test, pred, llm_after):
            rows.append(row[:2] + [model] + row[2:])
    order = {'Original': 0, 'SMOTE': 1, 'WGAN-GP': 2, 'LLM': 3}
    return sorted(rows, key=lambda r: (order[r[0]], 0 if r[1] == 'N/A' else int(r[1]),
                                       0 if r[2] == 'RF' else 1))


def export_xlsx(summary, output, llm_after):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Side, Font, PatternFill
    wb = Workbook()
    wb.remove(wb.active)
    gray = Side(style='thin', color='D9D9D9')
    for test, _ in summary.groupby('test', sort=False):
        name = ''.join('_' if ch in '\\/*?:[]' else ch for ch in f'Table4_{test}')[:31]
        if name in wb.sheetnames:
            raise ValueError('시트명 충돌: ' + name)
        ws = wb.create_sheet(name)
        ws.append(['Generator', 'Augmented samples', 'Model', 'Precision', '', 'Recall', '', 'F1-score', '', 'FPR', ''])
        ws.append(['', '', '', 'Without', 'With', 'Without', 'With', 'Without', 'With', 'Without', 'With'])
        for region in ['A1:A2', 'B1:B2', 'C1:C2', 'D1:E1', 'F1:G1', 'H1:I1', 'J1:K1']:
            ws.merge_cells(region)
        rows = make_combined_table(summary, test, llm_after)
        for row in rows:
            ws.append(row)
        end = ws.max_row
        for row in ws.iter_rows(min_row=1, max_row=end, max_col=11):
            for c in row:
                c.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
                c.border = Border(left=gray, right=gray, top=gray, bottom=gray)
                c.font = Font(name='Arial', size=10, bold=c.row <= 2)
                if c.row <= 2:
                    c.fill = PatternFill('solid', fgColor='E7EEF5')
        ws.column_dimensions['A'].width = 17
        ws.column_dimensions['B'].width = 20
        ws.column_dimensions['C'].width = 10
        for col in 'DEFGHIJK':
            ws.column_dimensions[col].width = 22
        for n in range(1, end + 1):
            ws.row_dimensions[n].height = 29
        notes = [
            'Values: mean ± sample SD (ddof=1); computed from individual raw runs.',
            'FPR = FP / (FP + TN), calculated per run; all metrics use the 0-1 scale.',
            'Augmented samples = synthetic records added to training, not QG reference size.',
            'Original baseline uses real data only. N/A = unavailable or not applicable.',
            'SMOTE/WGAN-GP With = QG on; WGAN-GP feedback = N/A.',
            ('LLM Without = delta, no QG/feedback; With = delta + QG + feedback.'
             if llm_after.endswith('feedback') else
             'LLM Without = delta, no QG/feedback; With = delta + QG, no feedback.'),
            'The feedback configuration measures combined QG/feedback processing, not QG alone.',
            'Raw and majority are not pooled; RF and MLP are adjacent rows; each test has its own table.',
        ]
        for i, text in enumerate(notes, end + 2):
            ws.cell(i, 1, text)
            ws.merge_cells(start_row=i, start_column=1, end_row=i, end_column=11)
            ws.cell(i, 1).alignment = Alignment(wrap_text=True, vertical='center')
            ws.cell(i, 1).font = Font(name='Arial', size=10)
            ws.row_dimensions[i].height = 28
        ws.freeze_panes = 'D3'
        ws.print_title_rows = '1:2'
        ws.sheet_properties.pageSetUpPr.fitToPage = True
        ws.page_setup.orientation = 'landscape'
        ws.page_setup.paperSize = ws.PAPERSIZE_A4
        ws.page_setup.fitToWidth = 1
        ws.page_setup.fitToHeight = 0
        ws.print_area = f'A1:K{ws.max_row}'
    numeric = wb.create_sheet('Mean_SD_details')
    numeric.append(list(summary.columns))
    for row in summary.itertuples(index=False, name=None):
        numeric.append([None if pd.isna(v) else v for v in row])
    numeric.freeze_panes = 'A2'
    numeric.auto_filter.ref = numeric.dimensions
    for col in numeric.columns:
        numeric.column_dimensions[col[0].column_letter].width = 23
    output.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=INPUT_XLSX)
    parser.add_argument('--append', nargs='*', type=Path, default=[])
    parser.add_argument('--output', type=Path, default=OUTPUT_XLSX)
    parser.add_argument('--llm-after', choices=['llm_delta_qg', 'llm_delta_qg_feedback'], default=LLM_AFTER)
    parser.add_argument('--check-only', action='store_true', help='집계·검증만 수행, 파일 저장 안 함')
    args = parser.parse_args()
    df = prepare([args.input, *args.append])
    summary = summarize(df)
    print(f'[확인] raw 실행 {len(df):,}개, 조건 집계 {len(summary)}개')
    for test, _ in summary.groupby('test'):
        rows = make_combined_table(summary, test, args.llm_after)
        print(f'[표] {test} / RF+MLP: {len(rows)}행, Original → SMOTE → WGAN-GP → LLM')
    if not args.check_only:
        if args.output.resolve() in {p.resolve() for p in [args.input, *args.append]}:
            raise ValueError('출력 경로가 입력과 같음: 원본 덮어쓰기 금지')
        export_xlsx(summary, args.output, args.llm_after)
        print(f'[완료] {args.output.resolve()}')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError, ImportError) as exc:
        raise SystemExit(f'[오류] {exc}')
