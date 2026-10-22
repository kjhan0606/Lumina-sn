#!/usr/bin/env python3
"""같은 물리 입력의 warp/native 대체 선택으로 원 전주파수 수송을 완결한다."""
import argparse
import json
from pathlib import Path
import re

import lumina_gpu_prefix_actual as prior
from lumina_gpu_hybrid import HybridDriver
from lumina_gpu_failure_code import decode

local=prior.local
local.SCHEMA='lumina-warp-native-limit-fallback-full-interval-v1'
local.GpuDriver=HybridDriver
local.LIMITS=dict(local.LIMITS,engine=HybridDriver.engine,prep_window_seconds=3600,
    already_allocated_gpu_seconds=4341,cumulative_preparation_limit_seconds=7941,
    local_internal_seconds=3480,interval_seconds=3420,external_wall_seconds=3500,slurm_allocation_seconds=3600,
    carried_parallel_work=8*1115020)


def evidence(root):
    prior.evidence(root)
    if local.M.N.sha(root/'diagnostic.so')!=local.M.N.sha(root/'history_cuda.so'):
        raise ValueError('같은 장치 대조/실제수송 바이너리 필요')


def prepare(root):
    root=Path(root).resolve();evidence(root)
    local.prepare(root,extra_files=prior.PROOFS+('diagnostic.so','cpu-cost.json'),interval_indices=(13003,),
        comparison_context=dict(method='WARP_SOURCE_MEASURE_WITH_SEPARATE_NATIVE_LIMIT_FALLBACK_KERNEL',
            workspace_expected_bytes=1070005192,GPU_library_sha256=local.M.N.sha(root/'history_cuda.so'),
            criteria='원천1e-8/공간1e-5/완성출력별추정오차비1/독립미세3열각1e-4 유지. 10초비용은보고로분리.',
            fallback='명시적work/precision 3..7만대체. BAD_INPUT/NUMERIC_RANGE는실패. 두패널상한합<=원maxpanels.',
            time_policy='기존4341초이월+새단회최대3600=누적7941초. 장치시험/제한대조/수송은동일할당에포함.',
            work_policy='장치시험·제한867질의의보수8NF를원512NF에서먼저차감. scalar FLOP/패널수와혼용하지않음.',
            own_field='자체J0=I0=0/fullSR. 새재구성후보미채택. CMFGEN J주입없음.',
            required_before_actual='동일할당직접장치시험과원867+실패569899세조각3질의/CPU대조 exit0',
            incomplete_scope='원13003단회이며자체J1/J2/생산율/CMFGEN최종목표완료가아님',
            field_calls=0,SE_RE_updates=0,no_automatic_retry=True))


def run(path,digest):
    root=Path(path).resolve().parent;evidence(root)
    for name in ('device-test.exit','checks.exit'):
        if (root/name).read_text().strip()!='0':raise ValueError('동일할당직접대조미충족: '+name)
    diagnostic=json.loads((root/'gpu-cost.json').read_text())
    if diagnostic['status']!='LIMITED_GPU_COST_DIAGNOSIS_FINISHED_NOT_FIELD' or \
       diagnostic['query_columns']!=870 or not diagnostic['comparisons'] or \
       not all(row['met'] for row in diagnostic['comparisons']):raise ValueError('원 제한 CPU/GPU 대조 미충족')
    code=local.run(path,digest);record=root/'gpu-local.json'
    match=re.search(r'GPU 호출/장치 상태 실패: (\d+)',json.loads(record.read_text()).get('error',''))
    with (root/'failure-location.json').open('x') as stream:
        json.dump(dict(failure=decode(int(match.group(1))) if match else None,
            result_sha256=local.M.N.sha(record),final_goal='NOT_MET'),stream,ensure_ascii=False,indent=2,allow_nan=False)
    return code


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','run'));p.add_argument('path')
    p.add_argument('--sha256');args=p.parse_args()
    if args.mode=='prepare':prepare(args.path)
    else:raise SystemExit(run(args.path,args.sha256))
