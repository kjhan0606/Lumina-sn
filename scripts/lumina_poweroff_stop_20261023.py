#!/usr/bin/env python3
"""2026-10-23 정전 전 Lumina-sn 한정 종료·보존. 기본은 무변경 확인이다."""
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path('/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn')
RECORD = ROOT / 'docs/LUMINA_POWEROFF_2026-10-23.json'
CUTOFF = dt.datetime.fromisoformat('2026-10-23T10:50:00+09:00')
BRANCH = 'thenmc-macroatom-fluorescence'
PATHS = ['AGENTS.md', 'CLAUDE.md', str(RECORD.relative_to(ROOT)),
         'scripts/lumina_poweroff_stop_20261023.py',
         'docs/HANDOVER_2026-10-20.md',
         'docs/HANDOVER_2026-10-22.md',
         'docs/LUMINA_STATE_SPECTRUM_BRIDGE_2026-10-20.json',
         'docs/LUMINA_SIII_OPERATOR_COMMON_GRID_2026-10-23.md',
         'scripts/lumina_gpu_linear.py',
         'tests/test_lumina_gpu_linear_host.py',
         'scripts/cmfgen_model_continuum.py', 'scripts/cmfgen_model_sources.py',
         'scripts/run_lumina_bridge_field0.py',
         'scripts/run_lumina_bridge_field0.slurm',
         'scripts/run_lumina_cell_candidate.py',
         'src/cmf_exact_sliding.c', 'src/cmf_exact_sliding.h',
         'src/lumina_cell_candidate.c',
         'tests/test_cmfgen_model_continuum.py',
         'tests/test_cmfgen_model_sources.py',
         'tests/test_lumina_bridge_field0.py',
         'tests/test_lumina_cell_candidate.py']


def command(*args, timeout=60):
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True,
                            timeout=timeout, check=True)
    if result.stdout:
        print(result.stdout, flush=True)
    return result.stdout.strip()


def identity(pid):
    """PID 재사용을 구분하며 계산 실행 파일과 실제 cwd를 함께 제한한다."""
    proc = Path('/proc') / str(pid)
    try:
        if proc.stat().st_uid != os.getuid() or pid == os.getpid():
            return None
        cwd = (proc / 'cwd').resolve(strict=True)
        args = (proc / 'cmdline').read_bytes().split(b'\0')
        args = [x.decode(errors='replace') for x in args if x]
        if not args:
            return None
        # shell과 공용 에이전트는 제외한다. timeout 부모가 아닌 계산만 종료한다.
        executable = Path(args[0]).name
        script = next((Path(x).name for x in args[1:]
                       if x.endswith('.py')), '')
        python_job = executable.startswith('python') and script != Path(__file__).name and (
            'lumina' in script or 'cmfgen' in script or
            (len(args) > 2 and args[1:3] == ['-m', 'pytest']))
        native_job = executable.startswith(('lumina_', 'cmfgen', 'cmf_'))
        in_project = cwd == ROOT or ROOT in cwd.parents
        in_run = str(cwd).startswith('/gpfs/kjhan/cmfgen_runs/lumina_')
        if not (in_project or in_run) or not (python_job or native_job):
            return None
        # 프로세스 이름에 공백이 있어도 시작 시각 필드를 정확히 읽는다.
        birth = (proc / 'stat').read_text().rsplit(')', 1)[1].split()[19]
        return dict(pid=pid, birth=birth, cwd=str(cwd), argv=args)
    except (OSError, ValueError, IndexError):
        return None


def targets():
    return [item for p in Path('/proc').iterdir() if p.name.isdigit()
            if (item := identity(int(p.name))) is not None]


def save(record):
    temporary = RECORD.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(RECORD)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT or command('git', 'rev-parse', '--show-toplevel') != str(ROOT):
        raise RuntimeError('프로젝트 경로 불일치')
    if command('git', 'branch', '--show-current') != BRANCH:
        raise RuntimeError('승인된 브랜치 불일치')
    if command('git', 'remote', 'get-url', 'origin') != 'git@github.com:kjhan0606/Lumina-sn.git':
        raise RuntimeError('원격 저장소 불일치')
    selected = targets()
    print(json.dumps(dict(execute=args.execute, targets=selected,
                          commit_paths=PATHS), ensure_ascii=False), flush=True)
    if not args.execute:
        return
    if dt.datetime.now(dt.timezone.utc) < CUTOFF:
        raise RuntimeError('예약 종료 시각 이전의 실행 거부')
    with open('/tmp/lumina-poweroff-stop-20261023.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = json.loads(RECORD.read_text())
        record.update(status='STOPPING', started_at=dt.datetime.now().astimezone().isoformat(),
                      selected_processes=selected)
        save(record)
        for item in selected:
            if identity(item['pid']) == item:
                try:
                    os.kill(item['pid'], signal.SIGTERM)
                except ProcessLookupError:
                    pass
        until = time.monotonic() + 40
        while time.monotonic() < until and any(identity(x['pid']) == x for x in selected):
            time.sleep(1)
        for item in selected:
            if identity(item['pid']) == item:
                try:
                    os.kill(item['pid'], signal.SIGKILL)
                except ProcessLookupError:
                    pass
        record.update(status='STOPPED_COMMIT_PENDING', remaining_processes=targets(),
                      tests='정전 보존용 WIP 커밋. 종료 시 전체 시험을 새로 실행하지 않음.',
                      git_head_before=command('git', 'rev-parse', 'HEAD'))
        save(record)
        if record['remaining_processes']:
            raise RuntimeError('계산 잔존: 커밋 전에 운전자 확인 필요')
        command('git', 'add', '--', *PATHS)
        command('git', 'commit', '--only', '-m',
                'WIP: preserve Lumina state before scheduled power outage', '--', *PATHS,
                timeout=120)
        # 완료 기록도 푸시에 포함한다. 푸시 후의 실제 원격 해시는 journal로 확인한다.
        record.update(status='STOPPED_COMMITTED_PUSH_PENDING',
                      snapshot_commit=command('git', 'rev-parse', 'HEAD'))
        save(record)
        command('git', 'add', '--', str(RECORD.relative_to(ROOT)))
        command('git', 'commit', '--only', '-m', 'Record pre-outage snapshot identity',
                '--', str(RECORD.relative_to(ROOT)), timeout=60)
        command('git', 'push', 'origin', BRANCH, timeout=180)
        head = command('git', 'rev-parse', 'HEAD')
        remote = command('git', 'ls-remote', 'origin', 'refs/heads/' + BRANCH, timeout=45)
        if remote.split()[0] != head:
            raise RuntimeError('푸시 후 원격 커밋 불일치')
        print(json.dumps(dict(status='STOPPED_PUSH_VERIFIED', commit=head,
                              goal_pause='운전자가 goal API에서 paused 전환 필요'),
                         ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
