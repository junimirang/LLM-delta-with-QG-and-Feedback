# -*- coding: utf-8 -*-
"""기존 생성기·탐지기가 있는 프로젝트 폴더에 저장 후 실행
python 4_Revision_Sequential_Orchestrator.py
--batches 50 --batch-size 25 --repeats 3 (기본값)
50 outer batches/run, no splitting; attempts=1 means exactly 50 requests.
"""
from pathlib import Path
from datetime import datetime
import argparse, json, time, hashlib, random, re
try:
    import numpy as np
    import pandas as pd
    import requests
except ImportError as exc:
    raise SystemExit(f"필수 패키지 누락: {exc}\n설치: python -m pip install numpy pandas requests scikit-learn scipy openpyxl")
import importlib.util, sys
backend = None

BASE = Path(__file__).resolve().parent
CONDITIONS = [('no_delta_feedback', False, True), ('delta_no_feedback', True, False), ('delta_feedback', True, True)]

def path(value):
    p = Path(value)
    return p if p.is_absolute() else BASE / p

def sha256(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def strip_delta(value):
    if isinstance(value, dict):
        return {k: strip_delta(v) for k,v in value.items() if k not in {'residual_scatter_ratio','delta','delta_ij','δ'}}
    if isinstance(value, list):
        return [strip_delta(v) for v in value]
    return value

def prompts(profile, features, n, delta, feedback):
    system = backend.SYSTEM_INSTRUCTION
    payload = json.loads(backend.build_user_prompt(profile, features, n, feedback=feedback))
    if not delta:
        system = system.replace('correlation structure with residual scatter ratios', 'correlation structure')
        system = re.sub(r'1\. For every critical pair,.*?(?=2\. Do not produce)',
                        '1. Use the reported correlation coefficients to guide the direction and strength of relationships between feature pairs.\n', system, flags=re.S)
        payload = strip_delta(payload)
        payload['generation_request']['diversity_instruction'] = (
            'Do not repeat identical values across samples and do not clone-and-tweak one template row. '
            "Distribute samples across each feature's quantile bands rather than bunching near the mean.")
        if 'residual_scatter_ratio' in system:
            raise ValueError('δ 지침 제거 실패: 원본 SYSTEM_INSTRUCTION 확인 필요')
    # Match structured-output schema in every condition
    system = system.replace('Output ONLY a JSON array of objects, one object per sample, with exactly the given feature names as keys and nothing else:',
                            'Output ONLY a JSON object with a samples array. Each sample must have exactly the given feature names as keys:')
    return system+'\n/no_think', json.dumps(payload, ensure_ascii=False)

def save_summary(out, totals):
    frame=pd.DataFrame(totals)
    frame.to_csv(out/'run_summary.csv',index=False,encoding='utf-8-sig')
    if frame.empty: return
    columns=['qg_evaluated','qg_passed','qg_pass_rate','unique_saved','duplicates','total_seconds','llm_seconds','qg_seconds','feedback_seconds','valid_per_minute','actual_calls','failed_batches']
    summary=frame.groupby('condition',sort=False)[columns].agg(['mean','std','count'])
    summary.columns=[f'{a}_{b}' for a,b in summary.columns]
    summary.to_csv(out/'condition_summary_mean_sd.csv',encoding='utf-8-sig')

def run(args):
    if min(args.batches,args.batch_size,args.repeats,args.attempts)<1: raise ValueError('모든 횟수 >= 1 필요')
    out=path(args.output)/datetime.now().strftime('%Y%m%d_%H%M%S_%f');out.mkdir(parents=True)
    profile_path,real_path=path(args.profile),path(args.real)
    profile=json.loads(profile_path.read_text(encoding='utf-8'))[args.label]
    features=list(profile['univariate_profile'])
    real=pd.read_csv(real_path)
    if args.label_col not in real: raise ValueError('QG 원본 라벨 컬럼 없음')
    real=real.loc[real[args.label_col].astype(str)==args.label].copy()
    values=real[features].apply(pd.to_numeric,errors='coerce')
    real=values.loc[np.isfinite(values.to_numpy()).all(axis=1)]
    if len(real)<args.qg_samples or args.qg_samples<2: raise ValueError('QG 유효 원본 표본 수 부족')
    ref=real.sample(args.qg_samples,random_state=args.reference_seed)
    ref.to_csv(out/'qg_reference.csv',index=False,encoding='utf-8-sig')
    mean,inv_cov=backend.fit_mean_cov(ref.to_numpy(),args.ridge)
    threshold=float(np.percentile(backend.mahalanobis_distances(ref.to_numpy(),mean,inv_cov),args.percentile))
    std=ref.to_numpy().std(axis=0);std[std==0]=1e-9
    qf=dict(mean=mean,inv_cov=inv_cov,threshold=threshold,std=std)
    config=dict(vars(args),profile_sha256=sha256(profile_path),real_sha256=sha256(real_path),backend_sha256=sha256(Path(backend.__file__)),threshold=threshold,
                conditions=CONDITIONS,split_batches=False,model_seed_sent=False,model_selection='LM Studio user-selected model; request placeholder local-model',note='repeats independent runs; LLM RNG not fixed; sample std ddof=1; QG reference fixed')
    (out/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    totals=[]
    experiment_start=time.perf_counter()
    total_batches=args.repeats*args.batches*(len(CONDITIONS)+int(args.include_no_qg))
    completed_batches=0
    print(f"[계획] 총 {total_batches}배치, 최대 {total_batches*args.attempts}회 호출 | 순차 실행",flush=True)
    for repeat in range(1,args.repeats+1):
        order=CONDITIONS.copy()
        if args.include_no_qg: order.append(('delta_no_qg', True, False))
        for condition,delta,use_feedback in order:
            folder=out/condition/f'repeat_{repeat}';folder.mkdir(parents=True)
            print(f'\n[조건 시작] {condition} | 반복 {repeat}/{args.repeats}',flush=True)
            accepted=[];seen=set();feedback=None;batch_records=[];call_records=[]
            start=time.perf_counter();started=datetime.now().isoformat()
            try:
                for batch in range(1,args.batches+1):
                    print(f"[요청 시작] {condition} | 배치 {batch}/{args.batches} | {args.batch_size}개 요청",flush=True)
                    bt=time.perf_counter();rows=[];error='';llm_seconds=0;calls_before=len(call_records)
                    system,user=prompts(profile,features,args.batch_size,delta,feedback if use_feedback else None)
                    (folder/f'prompt_{batch:03d}.json').write_text(json.dumps(dict(system=system,user=json.loads(user)),ensure_ascii=False,indent=2),encoding='utf-8')
                    for attempt in range(1,args.attempts+1):
                        call_start=time.perf_counter();finish=None;usage={};http_ok=False;attempt_error=''
                        try:
                            body=dict(model="local-model",messages=[dict(role='system',content=system),dict(role='user',content=user)],temperature=args.temperature,top_p=args.top_p,max_tokens=500+args.batch_size*250,stream=False,response_format=backend.build_response_schema(features))
                            response=requests.post(args.server.rstrip('/')+'/v1/chat/completions',json=body,timeout=args.timeout)
                            response.raise_for_status();data=response.json();http_ok=True
                            usage=data.get('usage') or {};choice=data['choices'][0];finish=choice.get('finish_reason')
                            raw=choice['message'].get('content') or ''
                            (folder/f'raw_{batch:03d}_{attempt}.txt').write_text(raw,encoding='utf-8')
                            if finish=='length': raise ValueError('finish_reason=length: 응답 잘림')
                            rows=backend.parse_samples(raw,features)
                            rows=[row for row in rows if all(np.isfinite(float(row[f])) for f in features)]
                            if len(rows)>args.batch_size: raise ValueError('요청보다 많은 행 반환: 해당 응답 제외')
                            if not rows: raise ValueError('유효 파싱 행 0개')
                            error=''
                        except (requests.RequestException, ValueError, KeyError, TypeError, IndexError) as exc:
                            attempt_error=error=str(exc);rows=[]
                        elapsed=time.perf_counter()-call_start;llm_seconds+=elapsed
                        call_records.append(dict(batch=batch,attempt=attempt,seconds=elapsed,http_ok=http_ok,finish_reason=finish,error=attempt_error,**{k:usage.get(k) for k in ['prompt_tokens','completion_tokens','total_tokens']}))
                        pd.DataFrame(call_records).to_csv(folder/'calls.csv',index=False,encoding='utf-8-sig')
                        if rows: break
                    evaluated=len(rows);qstart=time.perf_counter()
                    kept,rejected,n_rejected = (rows, [], 0) if condition == 'delta_no_qg' else backend.apply_quality_filter(rows,features,qf)
                    qseconds=time.perf_counter()-qstart
                    fstart=time.perf_counter()
                    feedback=backend.summarize_rejection_feedback(rejected,features,qf) if use_feedback and rejected else None
                    fseconds=time.perf_counter()-fstart
                    duplicates=0
                    for row in kept:
                        key=tuple(row[f] for f in features)
                        if key in seen: duplicates+=1;continue
                        seen.add(key);accepted.append(row)
                    df=pd.DataFrame(accepted,columns=features);df[args.label_col]=args.label
                    df.to_csv(folder/'synthetic_ssh_filtered.csv',index=False,encoding='utf-8-sig')
                    batch_records.append(dict(batch=batch,requested=args.batch_size,qg_evaluated=evaluated,qg_passed=len(kept),qg_rejected=n_rejected,duplicates=duplicates,new_valid=len(kept)-duplicates,total_valid=len(accepted),actual_calls=len(call_records)-calls_before,llm_seconds=llm_seconds,qg_seconds=qseconds,feedback_seconds=fseconds,batch_seconds=time.perf_counter()-bt,error=error))
                    pd.DataFrame(batch_records).to_csv(folder/'batches.csv',index=False,encoding='utf-8-sig')
                    completed_batches+=1
                    wall=time.perf_counter()-experiment_start
                    eta=wall/completed_batches*(total_batches-completed_batches)
                    print(f'[진척] {completed_batches}/{total_batches} ({completed_batches/total_batches:.1%}) | '
                          f'{condition} 반복 {repeat}/{args.repeats} 배치 {batch}/{args.batches} | '
                          f'QG {len(kept)}/{evaluated}, 누적 {len(accepted)}개 | '
                          f'경과 {wall/60:.1f}분, 예상잔여 {eta/60:.1f}분 | '
                          f"상태: {error or '성공'}",flush=True)
            finally:
                elapsed=time.perf_counter()-start; br=pd.DataFrame(batch_records)
                evaluated=sum(r['qg_evaluated'] for r in batch_records);passed=sum(r['qg_passed'] for r in batch_records)
                totals.append(dict(condition=condition,repeat=repeat,started=started,ended=datetime.now().isoformat(),status='completed' if len(batch_records)==args.batches else 'interrupted',completed_batches=len(batch_records),qg_evaluated=evaluated,qg_passed=passed,qg_pass_rate=passed/evaluated if evaluated else np.nan,unique_saved=len(accepted),duplicates=sum(r['duplicates'] for r in batch_records),total_seconds=elapsed,llm_seconds=sum(r['seconds'] for r in call_records),qg_seconds=sum(r['qg_seconds'] for r in batch_records),feedback_seconds=sum(r['feedback_seconds'] for r in batch_records),valid_per_minute=len(accepted)*60/elapsed,actual_calls=len(call_records),failed_batches=sum(bool(r['error']) for r in batch_records)))
                save_summary(out,totals)
    print(f'생성 완료: {out}')
    return out

def load_module(filename, name):
    filename=path(filename)
    sys.path.insert(0,str(filename.parent))
    spec=importlib.util.spec_from_file_location(name,filename)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def selected_pools(args):
    pools={
        #'smote':str(path('fake dataset-DARPA99/smote/synthetic_ssh.csv')),
        'smote-qg':str(path('fake dataset-DARPA99/smote-qg/synthetic_ssh_filtered.csv')),
        #'wgan-gp':str(path('fake dataset-DARPA99/wgan-gp/synthetic_ssh.csv')),
        #'wgan-gp-qg':str(path('fake dataset-DARPA99/wgan-gp-qg/synthetic_ssh_filtered.csv')),
        #'llm_nodelta':str(path('fake dataset-DARPA99/2_1_llm_without delta/synthetic_ssh.csv')),
        #'llm_delta':str(path('fake dataset-DARPA99/2_2_llm_delta/synthetic_ssh.csv')),
        #'llm_deltal_qg': str(path('fake dataset-DARPA99/2_3_llm_delta_with qg/synthetic_ssh_filtered_500.csv')),
        # 'llm_delta_qg_feedback':str(path('fake dataset-DARPA99/2_4_llm_delta_with qg and feedback/synthetic_ssh_filtered.csv')),
    }
    if args.data_paths:
        overrides=json.loads(path(args.data_paths).read_text(encoding='utf-8'))
        if not isinstance(overrides,dict) or set(overrides)-set(pools):
            raise ValueError('data-paths에는 활성 pools 키만 사용 가능')
        pools.update({k:str(path(v)) for k,v in overrides.items()})
    return pools


# 표시 이름은 유지하고 기존 탐지 코드의 조건 정의와 연결
POOL_ALIASES = {
    'llm_nodelta': 'llm', 'llm_delta': 'llm',
    'llm_deltal_qg': 'llm-qg', 'llm_delta_qg': 'llm-qg',
    'llm_delta_qg_feedback': 'llm-qg',
}


def configure_detector(module, pools):
    templates={row[0]: row for row in module.CONDITIONS}
    conditions=[templates['original']]
    for key in pools:
        template=templates[POOL_ALIASES.get(key,key)]
        conditions.append((key,key,template[2],template[3]))
    module.CONDITIONS=conditions
    module.SYNTHETIC_FILES=pools.copy()


def run_detection(args, generation_out=None):
    # 기존 main을 메모리상의 설정만 바꿔 순차 실행; 원본 파일은 변경하지 않음
    pools=selected_pools(args)
    detection_root=(generation_out if generation_out else path(args.output)/datetime.now().strftime('%Y%m%d_%H%M%S_%f'))/'detection'
    detection_root.mkdir(parents=True,exist_ok=True)
    repeats=[1]  # 지정한 기존 풀을 한 번 평가; 탐지 내부에서 다중 시드 반복
    for repeat in repeats:
        current=pools.copy()
        paths_file=detection_root/f'paths_repeat_{repeat}.json'
        paths_file.write_text(json.dumps(current,ensure_ascii=False,indent=2),encoding='utf-8')
        for kind,script in [('RF',args.rf_script),('MLP',args.mlp_script)]:
            module=load_module(script,'revision_detector_'+kind)
            configure_detector(module,current)
            cfg=module.DATASET_CONFIG['darpa99'].copy()
            for key in ['real_dir','fake_dir']: cfg[key]=str(path(cfg[key]))
            cfg['result_dir']=str(detection_root/f'repeat_{repeat}'/kind)
            module.DATASET_CONFIG={'darpa99':cfg}
            previous=sys.argv
            sys.argv=[script,'--iterations',str(args.detection_iterations),'--base-seed',str(args.detection_seed),'--paths-json',str(paths_file),'--skip-quality']
            try:
                print(f'순차 탐지 시작: {kind}, 생성 반복 {repeat}',flush=True)
                module.main()
            finally: sys.argv=previous
    print(f'탐지 완료: {detection_root}')


def preflight(args):
    print('[사전 확인] 실행 조건 검사 시작',flush=True)
    errors=[]
    def check(name, operation):
        try:
            result=operation()
            print(f'  [OK] {name}',flush=True)
            return result
        except Exception as exc:
            errors.append(f'{name}: {exc}')
            print(f'  [FAIL] {name}: {exc}',flush=True)
            return None
    def settings():
        if min(args.batches,args.batch_size,args.repeats,args.attempts,args.timeout)<1:
            raise ValueError('배치·반복·시도·timeout은 양수 필요')
        if args.qg_samples<2 or not 0<args.percentile<100 or args.ridge<=0:
            raise ValueError('QG 표본>=2, 0<percentile<100, ridge>0 필요')
        if args.temperature<0 or not 0<args.top_p<=1:
            raise ValueError('temperature>=0, 0<top-p<=1 필요')
        if args.detection_iterations<2: raise ValueError('탐지 반복>=2 필요')
    check('수치 설정',settings)
    def writable():
        import tempfile
        folder=path(args.output);folder.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryFile(dir=folder) as stream: stream.write(b'check')
    check('출력 폴더 저장 권한',writable)
    if args.phase in ['generation','all']:
        global backend
        backend=check('기존 생성기 불러오기',lambda:load_module(args.backend,'revision_generator_backend'))
        def generation_inputs():
            if backend is None: raise ValueError('생성기 로드 실패')
            for name in ['fit_mean_cov','mahalanobis_distances','apply_quality_filter','summarize_rejection_feedback','build_response_schema','build_user_prompt','parse_samples']:
                if not callable(getattr(backend,name,None)): raise ValueError(f'필수 함수 없음: {name}')
            profile=json.loads(path(args.profile).read_text(encoding='utf-8'))[args.label]
            features=list(profile['univariate_profile'])
            if len(features)<2: raise ValueError('공분산 계산에 feature 2개 이상 필요')
            real=pd.read_csv(path(args.real))
            if set(features+[args.label_col])-set(real.columns): raise ValueError('원본 필수 컬럼 누락')
            x=real.loc[real[args.label_col].astype(str)==args.label,features].apply(pd.to_numeric,errors='coerce')
            valid=np.isfinite(x.to_numpy()).all(axis=1)
            if valid.sum()<args.qg_samples: raise ValueError(f'QG 유효 표본 {valid.sum()} < {args.qg_samples}')
            for _,delta,feedback in CONDITIONS:
                system,user=prompts(profile,features,args.batch_size,delta,None)
                if not delta and 'residual_scatter_ratio' in user: raise ValueError('δ JSON 제거 실패')
            print(f'    원본 유효 양성 {valid.sum()}개, QG {args.qg_samples}개, 출력한도 {500+args.batch_size*250}토큰')
        check('프로파일·원본·δ 제거 지시문',generation_inputs)
        def server():
            response=requests.get(args.server.rstrip('/')+'/v1/models',timeout=10)
            response.raise_for_status();models=response.json().get('data',[])
            if not models: raise ValueError('모델 목록 비어 있음: LM Studio 모델/서버 확인')
            print('    서버 모델 목록: '+', '.join(str(item.get('id')) for item in models))
            print('    실제 GPU·컨텍스트 설정 및 local-model 라우팅은 LM Studio에서 확인 필요')
        check('LM Studio 서버 응답',server)
    if args.phase in ['detection','all']:
        for kind,filename in [('RF',args.rf_script),('MLP',args.mlp_script)]:
            def detector(filename=filename,kind=kind):
                import openpyxl
                module=load_module(filename,'preflight_detector_'+kind)
                if not callable(getattr(module,'main',None)): raise ValueError('main 함수 없음')
                cfg=module.DATASET_CONFIG['darpa99']
                for key in ['train_file']:
                    module.load_data(path(cfg['real_dir'])/cfg[key],cfg)
                for file in cfg['test_files'].values(): module.load_data(path(cfg['real_dir'])/file,cfg)
            check(kind+' 탐지 코드·패키지·원본 데이터',detector)
        def synthetic():
            files=selected_pools(args)
            module=load_module(args.rf_script,'preflight_pool_checker')
            cfg=module.DATASET_CONFIG['darpa99']
            for key,file in files.items():
                df=module.load_data(path(file),cfg,synthetic=True)
                print(f'    {key}: 유효 {len(df)}개 (부족한 증강 수는 탐지 코드에서 건너뜀)')
        check('기존 합성 데이터 경로·컬럼·유효 행',synthetic)
    if errors: raise ValueError('사전 확인 실패 — 실행하지 않음\n'+'\n'.join(errors))
    print('[사전 확인 완료] 파일·설정·연결 확인 통과',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    defaults=dict(profile='meta_feature_profile_darpa.json',real='dataset-DARPA99/training dataset_week4_normalized.csv',label='ssh',label_col='LABEL',server='http://127.0.0.1:1234',output='result/ablation_generation',batches=50,batch_size=25,repeats=3,attempts=1,timeout=600,qg_samples=500,reference_seed=42,order_seed=42,temperature=1.2,top_p=.95,ridge=1e-6,percentile=99.)
    for name,value in defaults.items(): p.add_argument('--'+name.replace('_','-'),type=type(value),default=value)
    p.add_argument('--phase',choices=['generation','detection','all'],default='all')
    p.add_argument('--backend',default='2. Traffic Generator for LLM with QG and Feedback.py')
    p.add_argument('--include-no-qg',action='store_true',help='δ 적용/QG 없는 조건도 고정배치 생성')
    p.add_argument('--rf-script',default='3. Detection_DARPA99_RF_batch_revision.py')
    p.add_argument('--mlp-script',default='3. Detection_DARPA99_MLP_batch_revision.py')
    p.add_argument('--data-paths',help='기존 합성 CSV의 경로 JSON; README 참고')
    p.add_argument('--detection-iterations',type=int,default=50)
    p.add_argument('--detection-seed',type=int,default=42)
    p.add_argument('--check-only',action='store_true',help='사전 확인만 실행')
    args=p.parse_args()
    preflight(args)
    if args.check_only: return
    global backend
    out=None
    if args.phase in ['generation','all']:
        out=run(args)
    if args.phase in ['detection','all']: run_detection(args,out)
if __name__=='__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n[중단] 마지막 완료 배치까지 저장됨',flush=True)
        sys.exit(130)
    except Exception as exc:
        import traceback
        error_path=BASE/'orchestration_error.log'
        with error_path.open('a',encoding='utf-8') as stream:
            stream.write(datetime.now().isoformat()+'\n'+traceback.format_exc()+'\n')
        print(f'[오류] {exc}\n상세 로그: {error_path}',flush=True)
        sys.exit(1)
