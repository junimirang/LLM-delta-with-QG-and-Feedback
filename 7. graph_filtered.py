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
CONDITION = 'llm_delta_qg_feedback'
DEFAULT_DATA = [{'test': 'week1-3', 'prediction': 'rf_raw', 'condition': 'original', 'n_fake': 0, 'n_repeats': 50, 'precision_mean': 0.93108258604, 'precision_sd': 0.03608936601204527, 'recall_mean': 0.86884276724, 'recall_sd': 0.006552335922783913, 'f1_mean': 0.89854825974, 'f1_sd': 0.01786525709284786}, {'test': 'week1-3', 'prediction': 'rf_raw', 'condition': 'llm_delta_qg_feedback', 'n_fake': 10, 'n_repeats': 50, 'precision_mean': 0.9347107952964899, 'precision_sd': 0.03313660567242337, 'recall_mean': 0.8678616352201253, 'recall_sd': 0.007331503355985013, 'f1_mean': 0.8997844137259069, 'f1_sd': 0.01713057235129032}, {'test': 'week1-3', 'prediction': 'rf_raw', 'condition': 'llm_delta_qg_feedback', 'n_fake': 50, 'n_repeats': 50, 'precision_mean': 0.9439098496481904, 'precision_sd': 0.0259693411498949, 'recall_mean': 0.8686666666666664, 'recall_sd': 0.008133723100153306, 'f1_mean': 0.9045404286881452, 'f1_sd': 0.01319190419018197}, {'test': 'week1-3', 'prediction': 'rf_raw', 'condition': 'llm_delta_qg_feedback', 'n_fake': 100, 'n_repeats': 50, 'precision_mean': 0.944978250865449, 'precision_sd': 0.02453834878103642, 'recall_mean': 0.8693962264150941, 'recall_sd': 0.008726393991193246, 'f1_mean': 0.9054427974956579, 'f1_sd': 0.01263231320824923}, {'test': 'week1-3', 'prediction': 'rf_raw', 'condition': 'llm_delta_qg_feedback', 'n_fake': 200, 'n_repeats': 50, 'precision_mean': 0.9443719411134721, 'precision_sd': 0.02068625978532132, 'recall_mean': 0.8726540880503141, 'recall_sd': 0.00803341024592142, 'f1_mean': 0.9069763800078353, 'f1_sd': 0.01076982518699372}, {'test': 'week1-3', 'prediction': 'rf_raw', 'condition': 'llm_delta_qg_feedback', 'n_fake': 300, 'n_repeats': 50, 'precision_mean': 0.9330550809659499, 'precision_sd': 0.01884959481300432, 'recall_mean': 0.8742389937106916, 'recall_sd': 0.008016027291627674, 'f1_mean': 0.9025864546442335, 'f1_sd': 0.01015244937447736}, {'test': 'week1-3', 'prediction': 'rf_raw', 'condition': 'llm_delta_qg_feedback', 'n_fake': 500, 'n_repeats': 50, 'precision_mean': 0.9315210387146134, 'precision_sd': 0.01643652475607238, 'recall_mean': 0.8751320754716977, 'recall_sd': 0.00811138154188929, 'f1_mean': 0.9023501161796821, 'f1_sd': 0.008463296550185562}, {'test': 'week1-3', 'prediction': 'rf_raw', 'condition': 'llm_delta_qg_feedback', 'n_fake': 700, 'n_repeats': 50, 'precision_mean': 0.9284237807135425, 'precision_sd': 0.01634851547647439, 'recall_mean': 0.8764528301886788, 'recall_sd': 0.007088442025849339, 'f1_mean': 0.901610290698866, 'f1_sd': 0.008707475375766096}, {'test': 'week1-3', 'prediction': 'rf_raw', 'condition': 'llm_delta_qg_feedback', 'n_fake': 1000, 'n_repeats': 50, 'precision_mean': 0.9225649407035463, 'precision_sd': 0.01649281715958299, 'recall_mean': 0.877270440251572, 'recall_sd': 0.007762314466660197, 'f1_mean': 0.899240666214493, 'f1_sd': 0.007823521033858022}]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='Mean_SD_details 시트를 가진 최종 결과 XLSX')
    parser.add_argument('--test', default='week1-3')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'figures' / 'detection_vs_nfake_filter.png')
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
