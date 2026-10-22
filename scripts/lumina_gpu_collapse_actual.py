#!/usr/bin/env python3
"""실제 좌표 붕괴를 정밀도한도로 분류한 같은 물리 수송의 단회."""
import argparse
import json
from pathlib import Path
import re

import lumina_gpu_hybrid_actual as base
import lumina_native_cost_diag as diag
from lumina_gpu_failure_code import decode

local=base.local
local.SCHEMA='lumina-collapsed-frequency-position-precision-limit-v1'
local.LIMITS=dict(local.LIMITS,engine='WARP_NATIVE_COLLAPSED_POSITION_CLASSIFIED_SM89',
    prep_window_seconds=3415,already_allocated_gpu_seconds=4586,cumulative_preparation_limit_seconds=8001,
    interval_seconds=3200,local_internal_seconds=3300,external_wall_seconds=3350,slurm_allocation_seconds=3415,
    execution_host='lageunha',manual_GPU=True,carried_parallel_work=32*1115020)
CPU=Path('/gpfs/kjhan/cmfgen_runs/lumina_collapse_fix_20261022-r7el0K')
CPU_SHA='7e1736a1d4c2a9bd7c1a4facf6a6d176aa38f4cc1aa96fa4770edbb429cd1729'


def extend_execution_limits(previous_gpu_seconds, *, long_run=False):
    """새 봉인 실행에만 시간·누적작업 한도를 적용한다."""
    if type(previous_gpu_seconds) is not int or previous_gpu_seconds<4586:
        raise ValueError('이전 준비 계산의 실제 누적 GPU초가 필요하다')
    if type(long_run) is not bool or (long_run and previous_gpu_seconds<22264):
        raise ValueError('24시간 선택에는 종료된 4시간 실행의 정산이 필요하다')
    interval_seconds,internal_seconds,external_seconds,parent_seconds,full_vectors=(
        (86000,86100,86200,86400,2000000) if long_run else
        (14400,14500,14600,14700,4096))
    class ExtendedDriver(base.HybridDriver):
        interval_limit_seconds=float(interval_seconds)
    local.GpuDriver=ExtendedDriver
    local.SCHEMA=('lumina-collapsed-frequency-position-user-extended-24h-v1' if long_run else
                  'lumina-collapsed-frequency-position-user-extended-4h-v1')
    local.LIMITS=dict(local.LIMITS,prep_window_seconds=parent_seconds,
        already_allocated_gpu_seconds=previous_gpu_seconds,
        cumulative_preparation_limit_seconds=previous_gpu_seconds+parent_seconds,
        interval_seconds=interval_seconds,local_internal_seconds=internal_seconds,
        external_wall_seconds=external_seconds,
        slurm_allocation_seconds=parent_seconds,local_full_vectors_total=full_vectors,
        user_extended_full_frequency=True)


def evidence(root):
    base.evidence(root)
    path=CPU/'cpu-three-pieces.json'
    if local.M.N.sha(path)!=CPU_SHA:raise ValueError('원541362 CPU 재현 변경')
    proof=json.loads(path.read_text())
    if [r['original']['code'] for r in proof['rows']]!=[104,0,0] or \
       any(r['column']!=541362 or r['native_bin']['code'] for r in proof['rows']) or \
       any(not r['comparison_met'] for r in proof['rows'][1:]):raise ValueError('정밀도4 분류/세 조각 재현 미충족')
    if local.M.N.sha(root/'src/cmf_characteristic_linear.hpp')!=local.M.N.sha(CPU/'src/cmf_characteristic_linear.hpp'):
        raise ValueError('CPU 검증과 GPU 정밀도 분류 소스 불일치')


def prepare(root):
    evidence(root)
    local.prepare(root,extra_files=base.prior.PROOFS+tuple(str(CPU/p) for p in (
        'cpu-three-pieces.json','trace.so','host-test.log','src/cmf_characteristic_linear.hpp'))+
        ('diagnostic.so','cpu-cost.json'),interval_indices=(13003,),comparison_context=dict(
            method='POSITIVE_SPACE_PANEL_COLLAPSED_FREQUENCY_POSITION_IS_PRECISION4_NOT_NUMERIC2',
            workspace_expected_bytes=1070005192,GPU_library_sha256=local.M.N.sha(root/'history_cuda.so'),
            correction='u0<u1/span>0인데두native좌표가완전히같을때만기존PRECISION_LIMIT4. 일반INVALID는2유지.',
            direct='원35열/867질의+실제541362세조각/CPU-native3질의. 후자는구현대조이지독립물리참조가아님.',
            accuracy='원천1e-8/공간1e-5/완성출력별추정오차비1/독립미세3열각1e-4유지.10초는비용보고.',
            resources=(f"이전{local.LIMITS['already_allocated_gpu_seconds']}GPU초+"
                f"이번최대{local.LIMITS['prep_window_seconds']}="
                f"누적{local.LIMITS['cumulative_preparation_limit_seconds']}초. "
                f"장치시험/대조/수송은같은부모안.32NF준비를"
                f"{local.LIMITS['local_full_vectors_total']}NF에서차감."),
            field_calls=0,SE_RE_updates=0,no_automatic_retry=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','check','run'));parser.add_argument('path')
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--extended-after-gpu-seconds',type=int,
        help='이전 부모 종료 후 정산한 누적 GPU초. 다음 봉인 단회에 4시간/4096NF 적용')
    group.add_argument('--long-after-gpu-seconds',type=int,
        help='4시간 실행 종료 후 정산한 누적 GPU초. 새 봉인 단회에 최대24시간/2000000NF 적용')
    parser.add_argument('--sha256');a=parser.parse_args();root=Path(a.path).resolve()
    if a.extended_after_gpu_seconds is not None:extend_execution_limits(a.extended_after_gpu_seconds)
    if a.long_after_gpu_seconds is not None:extend_execution_limits(a.long_after_gpu_seconds,long_run=True)
    if a.mode=='prepare':prepare(root)
    elif a.mode=='check':
        evidence(root)
        diag.CompactDriver=local.GpuDriver
        raise SystemExit(diag.gpu(root,three_piece_reference=CPU/'cpu-three-pieces.json',
            three_piece_column=541362,reference_integrator='native_bin'))
    else:
        evidence(root.parent)
        for name in ('device-test.exit','checks.exit'):
            if (root.parent/name).read_text().strip()!='0':raise ValueError('동일 부모 직접시험/대조 미충족')
        diagnostic=json.loads((root.parent/'gpu-cost.json').read_text())
        if diagnostic['status']!='LIMITED_GPU_COST_DIAGNOSIS_FINISHED_NOT_FIELD' or \
           diagnostic['query_columns']!=870 or len(diagnostic['comparisons'])!=44 or \
           not all(r['met'] for r in diagnostic['comparisons']) or \
           diagnostic['integration']['fallback_attempts']<1 or diagnostic['integration']['fallback_unresolved']:
            raise ValueError('실제541362의대체처리/원대조미충족')
        code=local.run(root,a.sha256);record=root.parent/'gpu-local.json'
        match=re.search(r'GPU 호출/장치 상태 실패: (\d+)',json.loads(record.read_text()).get('error',''))
        with (root.parent/'failure-location.json').open('x') as stream:
            json.dump(dict(failure=decode(int(match.group(1))) if match else None,
                result_sha256=local.M.N.sha(record),final_goal='NOT_MET'),stream,ensure_ascii=False,indent=2,allow_nan=False)
        raise SystemExit(code)
