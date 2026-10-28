#!/usr/bin/env python3
"""같은 수용 물질에서 완성 J를 다음 F에 전달한다. 새 물질 해법은 아니다.

GPU1/CPU1/host16GiB의 같은 Slurm 할당·총시간 안에서 최대2회만 호출한다.
유한2회 응답과 F(J1)-J1 간격을 산란/물질 수렴으로 표시하지 않는다.
기본은0장이며, 명시하면 같은 물질의 검증된 완성 J에서 새 반복을 시작한다.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from lumina_accepted_field_input import open_prepared
from lumina_fullsr_sc_actual import completed_transport, publish, verify_stage
from lumina_native_columns import sha
from lumina_material_trial_field import open_transport_trial, validate_initial_arguments


def field_gap(previous, current, shape):
    """큰 J를 복사하지 않는 간격 진단. 반환 J의 잔차 인증은 아니다."""
    a = np.load(previous, mmap_mode='r', allow_pickle=False)
    try:
        b = np.load(current, mmap_mode='r', allow_pickle=False)
        try:
            if a.shape != shape or b.shape != shape or a.dtype != np.dtype('f8') or b.dtype != a.dtype:
                raise ValueError('완성 J 비교의 격자/dtype 불일치')
            difference = scale = 0.
            for start in range(0, a.size, 131072):
                x, y = a.reshape(-1)[start:start+131072], b.reshape(-1)[start:start+131072]
                if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(x < 0) or np.any(y < 0):
                    raise ValueError('완성 J의 비유한/음수')
                difference = max(difference, float(abs(y-x).max()))
                scale = max(scale, float(x.max()), float(y.max()))
            return dict(absolute_max=difference, max_norm_relative=difference/scale if scale else 0.,
                meaning='F(J_previous)-J_previous; last returned J is not independently residual-tested')
        finally:
            b._mmap.close()
    finally:
        a._mmap.close()


def run(staged_root, out, *, cache, accepted_input, accepted_input_sha256, seconds, max_sweeps=2,
        input_j=None, input_j_sha256=None, initial_preparation=None, initial_preparation_sha256=None):
    validate_initial_arguments(input_j, input_j_sha256, initial_preparation, initial_preparation_sha256)
    if (type(seconds) is not int or not 240 <= seconds <= 43200 or
            type(max_sweeps) is not int or not 1 <= max_sweeps <= 2 or
            (input_j is None) != (input_j_sha256 is None)):
        raise ValueError('명시한 총시간240~43200초·최대2회 F 필요; 실행 승인은 별도')
    if (not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_CPUS_PER_TASK') != '1' or
            os.environ.get('SLURM_MEM_PER_NODE') != '16384'):
        raise ValueError('같은 Slurm GPU1/CPU1/host16GiB 실행 전용')
    staged_root, out, cache, accepted_input = map(lambda p: Path(p).resolve(),
        (staged_root, out, cache, accepted_input))
    if out.exists():
        raise FileExistsError('기존 되먹임 결과 보존')
    start = time.monotonic(); deadline = start+seconds
    stage_sha = verify_stage(staged_root)
    p, source, _, _ = open_prepared(accepted_input, accepted_input_sha256)
    initial = None
    with source:
        shape = (source.ns, source.nf); identity = source.identity
        if input_j is not None:
            input_j = Path(input_j).resolve()
            previous_field = open_transport_trial(source, input_j, input_j_sha256,
                initial_preparation=initial_preparation, initial_preparation_sha256=initial_preparation_sha256)
            initial_role = previous_field.identity
            previous_field.close()
            initial = dict(J=dict(path=str(input_j), sha256=input_j_sha256))
    links = {n: staged_root/n for n in ('code', 'stage.json', 'chain_cuda.so', 'geometry.so')}
    if any(not path.exists() for path in links.values()):
        raise FileNotFoundError('봉인된 실제 실행 코드/바이너리 필요')
    kernel_sha = {n: sha(links[n]) for n in ('chain_cuda.so', 'geometry.so')}
    out.mkdir(); output = out/'feedback.json'
    record = dict(schema='lumina-resident-fixed-material-feedback-v1', status='STARTED',
        source_identity=identity, staged_source_sha256=stage_sha, kernel_sha256=kernel_sha,
        accepted_preparation_sha256=accepted_input_sha256, sweeps=[], last_complete=None,
        initial_complete_J=None if initial is None else initial['J'],
        limits=dict(total_seconds=seconds, maximum_F_invocations=max_sweeps, GPUs=1, CPUs=1,
                    host_memory_GiB=16, automatic_retries=0),
        material_updates=0, new_BF_BB_integrals=0, observer_calls=0,
        scattering_convergence_certified=False, final_goal='NOT_MET')
    if initial is not None:
        record['initial_field_role'] = initial_role
    def save():
        record['elapsed_seconds'] = time.monotonic()-start
        publish(output, record)
    save(); first_cost = None
    try:
        for index in range(1, max_sweeps+1):
            remaining = deadline-time.monotonic()
            # 첫 F의 실측은 입장 추정에만 쓴다. 같은 양의 시간이 보장된다는 뜻은 아니다.
            required = 240. if first_cost is None else 1.2*first_cost+180.
            if remaining < required:
                record['status'] = 'SHARED_TIME_RESERVE_STOP'; break
            child = out/('F-'+str(index)); child.mkdir()
            for name, target in links.items():
                (child/name).symlink_to(target, target_is_directory=target.is_dir())
            prior = record['last_complete'] if record['last_complete'] is not None else initial
            child_seconds = math.floor(remaining-60.)
            command = [sys.executable, str(child/'code/scripts/lumina_resident_run.py'), str(child),
                '--cache', str(cache), '--accepted-input', str(accepted_input),
                '--accepted-input-sha256', accepted_input_sha256, '--seconds', str(child_seconds)]
            if prior is not None:
                command += ['--input-j', prior['J']['path'], '--input-j-sha256', prior['J']['sha256']]
            if index == 1 and initial_preparation is not None:
                command += ['--initial-preparation', str(Path(initial_preparation).resolve()),
                            '--initial-preparation-sha256', initial_preparation_sha256]
            entry = dict(directory=str(child), input_J='ZERO_OWN_START' if prior is None else prior['J'],
                seconds_limit=child_seconds, command=command, state='RUNNING')
            record['sweeps'].append(entry); save(); child_start = time.monotonic()
            # 매 F를 같은 실제 run으로 호출하므로 bind→외부대기/ES*J 준비를 새로 수행한다.
            # 이전 광선 checkpoint는 전달하지 않는다. stdout/종료코드도 단회별로 보존한다.
            with (child/'stdout.log').open('x') as stdout, (child/'stderr.log').open('x') as stderr:
                completed = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=remaining-30.)
            code = completed.returncode
            entry.update(exit_code=code, elapsed_seconds=time.monotonic()-child_start)
            if first_cost is None: first_cost = entry['elapsed_seconds']
            report_path = child/'resident-field.json'; report = json.loads(report_path.read_text())
            entry['result_sha256'] = sha(report_path)
            if (code != 0 or not completed_transport(report, shape[1]) or
                    report.get('source_identity') != identity or
                    report.get('identity', {}).get('input_plan_sha256') != accepted_input_sha256):
                entry['state'] = 'INCOMPLETE_PRESERVED'; record['status'] = 'PARTIAL_SWEEP_STOP'; break
            for name in ('J', 'moments'):
                target = Path(report[name]['path'])
                if target != (child/('J.npy' if name == 'J' else 'moments.npy')) or sha(target) != report[name]['sha256']:
                    raise ValueError('완성 F 출력 주소/본문 변경')
            expected_input = 'ZERO_OWN_START' if prior is None else prior['J']
            if report['identity']['input_J'] != expected_input:
                raise ValueError('이전 완성 J가 다음 실제 F에 전달되지 않음')
            if prior is not None:
                entry['J_gap'] = field_gap(prior['J']['path'], report['J']['path'], shape)
            else:
                # 첫 완성 J도 전체 값의 유한성/부호를 확인하되 자기 대조를 잔차로 부르지 않는다.
                field_gap(report['J']['path'], report['J']['path'], shape)
            entry['state'] = 'COMPLETE_F_NOT_CONVERGENCE'
            record['last_complete'] = {n: report[n] for n in ('J', 'moments')}
            record['last_complete']['sweep'] = index
            record['status'] = 'FINITE_FEEDBACK_RESPONSE_NOT_CONVERGENCE'; save()
        if record['status'] == 'STARTED': record['status'] = 'NO_COMPLETE_SWEEP'
    except Exception as error:
        if record['sweeps'] and record['sweeps'][-1]['state'] == 'RUNNING':
            record['sweeps'][-1]['state'] = 'FAILED_PRESERVED'
        record.update(status='ERROR_PRESERVED', error=type(error).__name__+': '+str(error))
    finally:
        save()
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('staged-root', 'out', 'cache', 'accepted-input'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--accepted-input-sha256', required=True)
    parser.add_argument('--seconds', type=int, required=True)
    parser.add_argument('--max-sweeps', type=int, choices=(1, 2), default=2)
    parser.add_argument('--input-j', type=Path)
    parser.add_argument('--input-j-sha256')
    parser.add_argument('--initial-preparation', type=Path)
    parser.add_argument('--initial-preparation-sha256')
    result = run(**vars(parser.parse_args()))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    raise SystemExit(0 if result['status'] == 'FINITE_FEEDBACK_RESPONSE_NOT_CONVERGENCE' else 1)
