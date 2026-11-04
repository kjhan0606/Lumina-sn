#!/usr/bin/env python3
"""이미 승인된 묶음의 단회 제출 코드와 인수를 저장한다. 작업 제출은 하지 않는다."""
import argparse
import json
from pathlib import Path
import re
import subprocess

from lumina_bundle_dispatch import sha, require_callback_route


def prepare(project, out, commit, gpu_root, coefficient_result, response_parent, coefficient_commit,
            max_sweeps=1, input_j=None, input_j_sha256=None,
            initial_preparation=None, initial_preparation_sha256=None, thermal_response=False,
            submission_host_addresses=None,response_thermal_hours=30):
    project, out, gpu_root = map(lambda p: Path(p).resolve(), (project, out, gpu_root))
    routing = dict(submission_host_addresses=submission_host_addresses or {})
    require_callback_route(routing)
    if (not all(re.fullmatch(r'[0-9a-f]{40}', value) for value in (commit, coefficient_commit)) or
            type(max_sweeps) is not int or max_sweeps not in (1, 2)):
        raise ValueError('실제 계수/후속 소스 commit과 왕복당 F 1회 또는 2회 한도 필요')
    if type(thermal_response) is not bool:
        raise ValueError('반환장 열보정 선택은 불리언이어야 함')
    if (type(response_thermal_hours) is not int or response_thermal_hours not in (30,48) or
            (not thermal_response and response_thermal_hours!=30)):
        raise ValueError('명시한 열보정 시간은30 또는48시간이어야 함')
    initial = (input_j, input_j_sha256, initial_preparation, initial_preparation_sha256)
    if any(v is not None for v in initial) and (not all(v is not None for v in initial) or
            not all(re.fullmatch(r'[0-9a-f]{64}', value) for value in (input_j_sha256, initial_preparation_sha256))):
        raise ValueError('이전 물질 J와 원 준비 기록 및 각 SHA가 함께 필요')
    staged_commit = json.loads((gpu_root/'prepared.json').read_text()).get('source_commit')
    if staged_commit != commit:
        if not isinstance(staged_commit, str) or not re.fullmatch(r'[0-9a-f]{7,40}', staged_commit):
            raise ValueError('봉인 GPU 소스 commit 식별자 형식 오류')
        resolved = subprocess.run(['git', '-C', str(project), 'rev-parse', '--verify',
                                   staged_commit+'^{commit}'], check=True, capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        if not re.fullmatch(r'[0-9a-f]{40}', resolved) or resolved != commit:
            raise ValueError('봉인 GPU 소스와 후속 commit 불일치')
    if Path(response_parent).exists(): raise FileExistsError('기존 후속 물질응답 보존')
    out.mkdir(exist_ok=False)
    names = ('lumina_bundle_dispatch.py', 'submit_lumina_accepted_feedback.slurm', 'submit_lumina_new_field_response.slurm')
    for name in names:
        content = subprocess.run(['git', '-C', str(project), 'show', commit+':scripts/'+name],
            check=True, capture_output=True, timeout=10).stdout
        (out/name).write_bytes(content)
    c = dict(schema='lumina-accepted-bundle-dispatch-v1', response_commit=commit,
        coefficient_commit=coefficient_commit,
        coefficient_result=str(Path(coefficient_result).resolve()), gpu_root=str(gpu_root),
        gpu_prepared_sha256=sha(gpu_root/'prepared.json'), response_parent=str(Path(response_parent).resolve()),
        control_sha256={name: sha(out/name) for name in names},
        submission_host_addresses=dict(routing['submission_host_addresses']),
        limits=dict(GPU_allocations=1, GPU_CPU=1, GPU_GiB=16,
                    GPU_hours=12 if thermal_response or max_sweeps == 2 else 10,
                    maximum_F=max_sweeps,
                    response_allocations=1, response_CPU=16, response_GiB=32,
                    response_hours=response_thermal_hours if thermal_response else 3,
                    response_temperature_evaluations=12 if thermal_response else 0,
                    response_thermal_policy='native_instantaneous_scenario' if thermal_response else None,
                    automatic_retries=0))
    if input_j is not None:
        c['initial_field'] = dict(J=dict(path=str(Path(input_j).resolve()), sha256=input_j_sha256),
            preparation=dict(path=str(Path(initial_preparation).resolve()), sha256=initial_preparation_sha256),
            role='INITIAL_TRIAL_NOT_ACCEPTED_FOR_NEW_MATERIAL')
    config = out/'config.json'; config.write_text(json.dumps(c, indent=2)+'\n')
    return dict(config=str(config), sha256=sha(config), jobs_submitted=0)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('project', 'out', 'gpu-root', 'coefficient-result', 'response-parent'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--commit', required=True)
    p.add_argument('--coefficient-commit', required=True)
    p.add_argument('--max-sweeps', type=int, choices=(1, 2), default=1)
    p.add_argument('--input-j', type=Path)
    p.add_argument('--input-j-sha256')
    p.add_argument('--initial-preparation', type=Path)
    p.add_argument('--initial-preparation-sha256')
    p.add_argument('--thermal-response', action='store_true',
        help='반환 J에 대한 T/ne 열보정을 최대12평가 단회로 이어 실행')
    p.add_argument('--response-thermal-hours',type=int,choices=(30,48),default=30,
        help='별도 승인된 열응답 시간; 기본30시간은 유지')
    p.add_argument('--grammar-address', required=True,
        help='계산 노드에서 도달 검증한 grammar 수치 IP; 원 SSH alias/호스트키 유지')
    p.add_argument('--syntax-address', help='필요한 경우 검증한 syntax 수치 IP')
    options = vars(p.parse_args())
    addresses = dict(grammar=options.pop('grammar_address'))
    syntax = options.pop('syntax_address')
    if syntax is not None:
        addresses['syntax'] = syntax
    print(json.dumps(prepare(**options, submission_host_addresses=addresses), indent=2))
