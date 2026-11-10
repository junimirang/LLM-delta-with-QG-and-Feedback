"""Run two existing LLM generators sequentially without editing them."""
import argparse
import ast
import csv
import importlib.util
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime
from urllib.request import urlopen

BASE = Path(__file__).resolve().parent
SCRIPTS = [
    '2. Traffic Generator for LLM with QG.py',
    '2. Traffic Generator for LLM with QG and Feedback(without delta).py',
]

def defaults(script):
    tree = ast.parse(script.read_text(encoding='utf-8-sig'), filename=str(script))
    result = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    result[target.id] = value
    return result


def resolve(root, value):
    p = Path(value).expanduser()
    return p if p.is_absolute() else root / p


def run_child(command, root, log_path, timeout):
    start = time.monotonic()
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
    proc = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                            errors='replace', env=env)
    messages = queue.Queue()
    def reader():
        for line in proc.stdout:
            messages.put(line)
        messages.put(None)
    threading.Thread(target=reader, daemon=True).start()
    heartbeat = start
    timed_out = False
    try:
        with log_path.open('w', encoding='utf-8') as log:
            while True:
                if timeout and time.monotonic() - start > timeout and proc.poll() is None:
                    timed_out = True
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    log.write('\n[manager] execution time limit reached\n')
                try:
                    line = messages.get(timeout=1)
                except queue.Empty:
                    if time.monotonic() - heartbeat >= 60:
                        print(f'[진행] 현재 코드 실행 중 · 경과 {(time.monotonic()-start)/60:.1f}분', flush=True)
                        heartbeat = time.monotonic()
                    continue
                if line is None:
                    break
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
        return proc.wait(), time.monotonic()-start, timed_out
    except BaseException:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-dir', type=Path, default=BASE)
    parser.add_argument('--first-script', default=SCRIPTS[0])
    parser.add_argument('--second-script', default=SCRIPTS[1])
    parser.add_argument('--num-samples', type=int, help='Override both scripts target count')
    parser.add_argument('--batch-size', type=int, help='Override both scripts batch size')
    parser.add_argument('--server', help='Override both scripts LM Studio URL')
    parser.add_argument('--timeout-seconds', type=float, default=0, help='Per-script time limit; 0=unlimited')
    parser.add_argument('--continue-on-error', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    for value in (args.num_samples, args.batch_size):
        if value is not None and value <= 0:
            parser.error('num-samples and batch-size must be positive')
    if args.timeout_seconds < 0:
        parser.error('timeout-seconds must not be negative')
    root = args.project_dir.expanduser().resolve()
    run_dir = root / 'result' / 'sequential_generators' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run_dir.mkdir(parents=True, exist_ok=False)
    tasks = []
    errors = []
    for name in [args.first_script, args.second_script]:
        try:
            script = resolve(root, name)
            values = defaults(script)
            for key in ['PROFILE_JSON', 'REAL_CSV', 'OUTPUT_CSV', 'SERVER_URL']:
                if key not in values:
                    raise ValueError(f'constant missing: {key}')
            for key in ['PROFILE_JSON', 'REAL_CSV']:
                file = resolve(root, values[key])
                if not file.is_file():
                    raise FileNotFoundError(file)
            json.loads(resolve(root, values['PROFILE_JSON']).read_text(encoding='utf-8-sig'))
            output = resolve(root, values['OUTPUT_CSV'])
            output.parent.mkdir(parents=True, exist_ok=True)
            probe = output.parent / f'.write_probe_{os.getpid()}'
            with probe.open('x') as f:
                f.write('ok')
            probe.unlink()
            command = [sys.executable, '-u', str(script)]
            for option, value in [('num-samples', args.num_samples), ('batch-size', args.batch_size), ('server', args.server)]:
                if value is not None:
                    command.extend(['--'+option, str(value)])
            tasks.append(dict(script=str(script), command=command, output=str(output),
                              server=args.server or values['SERVER_URL']))
            print(f'[확인] {script.name}\n  출력: {output}', flush=True)
            if output.exists():
                print('  기존 출력 파일 있음 — 생성 코드의 저장 시 덮어쓰기', flush=True)
        except Exception as exc:
            errors.append(f'{name}: {exc}')
    for package in ['numpy', 'pandas', 'requests']:
        if importlib.util.find_spec(package) is None:
            errors.append(f'필수 패키지 없음: {package}')
    for server in dict.fromkeys(t['server'] for t in tasks):
        try:
            with urlopen(server.rstrip('/')+'/v1/models', timeout=10) as response:
                data = json.load(response)
            if not data.get('data'):
                raise ValueError('모델 목록이 비어 있음')
            print(f'[확인] LM Studio 연결: {server}', flush=True)
        except Exception as exc:
            errors.append(f'LM Studio {server}: {exc}')
    (run_dir/'manifest.json').write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding='utf-8')
    if errors:
        (run_dir/'preflight_errors.txt').write_text('\n'.join(errors), encoding='utf-8')
        for error in errors:
            print(f'[FAIL] {error}')
        print(f'[중단] 사전 확인 실패 · 로그: {run_dir}')
        return 1
    print('[확인 완료] 파일·문법·JSON·출력 폴더·패키지·서버 확인 완료', flush=True)
    if args.check_only:
        return 0
    total_start = time.monotonic()
    summary = run_dir/'summary.csv'
    failed = False
    with summary.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['order','script','started','seconds','returncode','status','output','rows','log'])
        writer.writeheader()
        f.flush()
        for index, task in enumerate(tasks, 1):
            print(f'\n[시작 {index}/2] {Path(task["script"]).name}', flush=True)
            log = run_dir/f'{index}_execution.log'
            started = datetime.now().isoformat()
            code, seconds, timed_out = run_child(task['command'], root, log, args.timeout_seconds)
            rows = ''
            # Count only output from a successful run, never a stale file after failure
            if code == 0 and not timed_out and Path(task['output']).is_file():
                with Path(task['output']).open(encoding='utf-8-sig', newline='') as csv_file:
                    rows = max(0, sum(1 for _ in csv.reader(csv_file))-1)
            status = 'timeout' if timed_out else ('completed' if code == 0 else 'failed')
            writer.writerow(dict(order=index, script=task['script'], started=started, seconds=round(seconds,3),
                                 returncode=code, status=status, output=task['output'], rows=rows, log=str(log)))
            f.flush()
            print(f'[종료 {index}/2] {status} · {seconds/60:.1f}분 · 저장 행 수 {rows}', flush=True)
            if code != 0 or timed_out:
                failed = True
                if not args.continue_on_error:
                    print('[중단] 다음 코드는 실행하지 않음', flush=True)
                    break
    print(f'\n[관리 종료] 총 {(time.monotonic()-total_start)/60:.1f}분 · 결과: {run_dir}', flush=True)
    return 1 if failed else 0

if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\n[중단] 사용자 중단 — 현재 생성 프로세스 종료')
        raise SystemExit(130)
