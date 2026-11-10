# -*- coding: utf-8 -*-
"""최종 Table4_mean_sd.xlsx 기준 RF 평균±표본 SD 그래프
기본 실행: 내장된 최종 결과 사용 (별도 입력 파일 불필요)
최신 표 읽기: --input Table4_mean_sd.xlsx
"""
from pathlib import Path
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
CONDITION = 'llm_delta'
DEFAULT_DATA = [{'test': 'week1-3', 'prediction': 'rf_raw', 'condition': 'original', 'n_fake': 0, 'n_repeats': 50, 'precision_mean': 0.93108258604, 'precision_sd': 0.03608936601204527, 'recall_mean': 0.86884276724, 'recall_sd': 0.006552335922783913, 'f1_mean': 0.89854825974, 'f1_sd': 0.01786525709284786}, {'test': 'week1-3', 'prediction': 'rf_raw', 'condition': 'llm_delta', 'n_fake': 10, 'n_repeats': 50, 'precision_mean': 0.9250654526000001, 'precision_sd': 0.04064195243511487, 'recall_mean': 0.86872955968, 'recall_sd': 0.007759225158343531, 'f1_mean': 0.89553526206, 'f1_sd': 0.01944875268265282}, {'test': 'week1-3', 'prediction': 'rf_raw', 'condition': 'llm_delta', 'n_fake': 50, 'n_repeats': 50, 'precision_mean': 0.91727212092, 'precision_sd': 0.04046078527044576, 'recall_mean': 0.8689182389400001, 'recall_sd': 0.01003163062695333, 'f1_mean': 0.89193728514, 'f1_sd': 0.01938376799570458}, {'test': 'week1-3', 'prediction': 'rf_raw', 'condition': 'llm_delta', 'n_fake': 100, 'n_repeats': 50, 'precision_mean': 0.9157050481600001, 'precision_sd': 0.03140738247634139, 'recall_mean': 0.8675471698799998, 'recall_sd': 0.008353379577791633, 'f1_mean': 0.8906872124799999, 'f1_sd': 0.01536978485146322}, {'test': 'week1-3', 'prediction': 'rf_raw', 'condition': 'llm_delta', 'n_fake': 200, 'n_repeats': 50, 'precision_mean': 0.9023931592400001, 'precision_sd': 0.04330804562530984, 'recall_mean': 0.86633962258, 'recall_sd': 0.009913569294889851, 'f1_mean': 0.8834578254800001, 'f1_sd': 0.02164274043921333}, {'test': 'week1-3', 'prediction': 'rf_raw', 'condition': 'llm_delta', 'n_fake': 300, 'n_repeats': 50, 'precision_mean': 0.90681796816, 'precision_sd': 0.03981967618543317, 'recall_mean': 0.8674339622799999, 'recall_sd': 0.007150950092042961, 'f1_mean': 0.8862315683399999, 'f1_sd': 0.0193381516263115}, {'test': 'week1-3', 'prediction': 'rf_raw', 'condition': 'llm_delta', 'n_fake': 500, 'n_repeats': 50, 'precision_mean': 0.8945168017, 'precision_sd': 0.04153985342051451, 'recall_mean': 0.86732075464, 'recall_sd': 0.008181489668970776, 'f1_mean': 0.8802147027999999, 'f1_sd': 0.02062826727497353}, {'test': 'week1-3', 'prediction': 'rf_raw', 'condition': 'llm_delta', 'n_fake': 700, 'n_repeats': 50, 'precision_mean': 0.8980843118999999, 'precision_sd': 0.04370441253980462, 'recall_mean': 0.8672830188199999, 'recall_sd': 0.009251641779416427, 'f1_mean': 0.8818911401, 'f1_sd': 0.02249473021785096}, {'test': 'week1-3', 'prediction': 'rf_raw', 'condition': 'llm_delta', 'n_fake': 1000, 'n_repeats': 50, 'precision_mean': 0.8869326651599999, 'precision_sd': 0.04085517722655246, 'recall_mean': 0.8661383647199998, 'recall_sd': 0.01062050851838599, 'f1_mean': 0.87593137298, 'f1_sd': 0.02113265303504669}]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='Mean_SD_details 시트를 가진 최종 결과 XLSX')
    parser.add_argument('--test', default='week1-3')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'figures' / 'detection_vs_nfake_nofilter.png')
    args = parser.parse_args()
    df = pd.read_excel(args.input, sheet_name='Mean_SD_details') if args.input else pd.DataFrame(DEFAULT_DATA)
    required = {'test','prediction','condition','n_fake','n_repeats'} | {m+'_'+s for m in ['precision','recall','f1'] for s in ['mean','sd']}
    if required-set(df.columns):
        raise ValueError(f'필수 컬럼 누락: {sorted(required-set(df.columns))}')
    df = df[(df.test==args.test)&(df.prediction=='rf_raw')]
    baseline = df[df.condition=='original']
    syn = df[df.condition==CONDITION].sort_values('n_fake')
    if len(baseline)!=1 or syn.empty or syn.n_fake.duplicated().any():
        raise ValueError('original 1행과 중복 없는 합성 결과가 필요합니다')
    for m in ['precision','recall','f1']:
        for stat in ['mean','sd']:
            df.loc[:,m+'_'+stat] = pd.to_numeric(df[m+'_'+stat],errors='raise')
    original = baseline.iloc[0]
    fig,axes = plt.subplots(1,3,figsize=(16,5),sharey=True)
    for ax,(metric,title) in zip(axes,[('precision','Precision'),('recall','Recall'),('f1','F1-score')]):
        mean,sd=float(original[metric+'_mean']),float(original[metric+'_sd'])
        ax.axhline(mean,color='#424242',ls='--',label=f'Original: {mean:.4f} +/- {sd:.4f}')
        ax.axhspan(max(0,mean-sd),min(1,mean+sd),color='#424242',alpha=.08)
        ax.errorbar(syn.n_fake,syn[metric+'_mean'],yerr=syn[metric+'_sd'],color='#EF6C00',marker='o',capsize=3,linewidth=2,label=CONDITION)
        ax.set(title=title,xlabel='Synthetic records added to training',ylabel=title,ylim=(.80,1.01))
        ax.grid(ls='--',alpha=.3)
        ax.legend(fontsize=8,loc='lower left')
    fig.suptitle(f'DARPA99 / RF / {CONDITION}\nMean +/- sample SD; original band = +/- 1 SD',fontsize=12)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=300,bbox_inches='tight')
    fig.savefig(args.output.with_suffix('.pdf'),bbox_inches='tight')
    plt.close(fig)
    print(f'저장 완료: {args.output}')

if __name__=='__main__':
    main()
