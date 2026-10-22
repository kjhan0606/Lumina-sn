#!/usr/bin/env python3
"""원 물리 입력·출력별 오차를 보존한 native-bin GPU 전주파수 단회."""
import argparse
import json
from pathlib import Path

import lumina_gpu_local as local
from lumina_gpu_compact import Runtime
from lumina_gpu_native_bin import NativeBinDriver

CPU_PROOF=Path('/gpfs/kjhan/cmfgen_runs/lumina_native_bin_budget_20261022-KdVw2W')
CPU_SHA='6f82306bff6170037a40f09303f33e1cb00d930ddbdeb0cbe38d7b8b61855be4'
PROOFS=tuple(str(CPU_PROOF/p) for p in ('host-test.log','cpu-cost.json','cpu-comparison.json',
    'cpu-probe.so','src/cmf_native_bin_stream.hpp','tests/native_cost_probe.cpp'))+(
    'host-test.log','compile.log','original-compile.log','binary.sha256')

local.SCHEMA='lumina-native-bin-full-interval-user-extended-v1'
local.LIMITS=dict(local.LIMITS,cpus=1,engine='COMPACT_NATIVE_BIN_STREAM_A100',local_calls=1,
    prep_window_seconds=3600,original_preparation_budget_seconds=1800,already_allocated_gpu_seconds=1051,
    cumulative_preparation_limit_seconds=4651,user_extended_time_budget=True,
    local_internal_seconds=3500,interval_seconds=3400,external_wall_seconds=3550,
    slurm_allocation_seconds=3600,require_cost_target=False,source_relative=1e-8,spatial_relative=1e-5,
    fine_each_output=True,shared_accuracy_only=False,cache_integrator_workspace_GiB=1,
    normalization='PER_FREQUENCY_RELATIVE_NO_GLOBAL_BRIGHT_NORM',automatic_retries=0)
local.Runtime=Runtime
local.GpuDriver=NativeBinDriver


def evidence(root):
    if local.M.N.sha(root/'driver.so')!=CPU_SHA:raise ValueError('원 기하 바이너리 변경')
    sample=json.loads((CPU_PROOF/'cpu-cost.json').read_text())
    comparison=json.loads((CPU_PROOF/'cpu-comparison.json').read_text())
    if sample['status']!='CPU_SAMPLE_FINISHED_NO_MAP_FAILURES' or len(sample['rows'])!=35 or \
       sample['library_sha256']!=local.M.N.sha(CPU_PROOF/'cpu-probe.so') or \
       comparison['input_sha256']['cpu-cost.json']!=local.M.N.sha(CPU_PROOF/'cpu-cost.json') or \
       comparison['status']!='CPU_COARSE_COMPARISON_MET_NOT_FIELD' or len(comparison['rows'])!=35 or \
       not all(row['met'] for row in comparison['rows']):raise ValueError('원 실제35열 CPU 수용 미충족')
    if local.M.N.sha(root/'src/cmf_native_bin_stream.hpp')!=local.M.N.sha(CPU_PROOF/'src/cmf_native_bin_stream.hpp'):
        raise ValueError('실제 CPU 시험과 선택 GPU 적분 소스 불일치')
    identities=dict((name,digest) for digest,name in
                    (line.split() for line in (root/'binary.sha256').read_text().splitlines()))
    if set(identities)!={'history_cuda.so','driver.so'} or any(
        local.M.N.sha(root/name)!=digest for name,digest in identities.items()):
        raise ValueError('strict 빌드 산출 바이너리 변경')


def prepare(root):
    root=Path(root).resolve();evidence(root)
    local.prepare(root,extra_files=PROOFS,interval_indices=(13003,),comparison_context=dict(
        method='EXACT_NATIVE_BIN_POLYNOMIAL_ORDERED_MAPS_OUTPUT_ERROR_BUDGET',
        GPU_library_sha256=local.M.N.sha(root/'history_cuda.so'),workspace_expected_bytes=1048771496,
        previous_failure='433944: 시간초과가 아닌 SUBDIVISION_PRECISION_LIMIT. 새 후보가 해결했는지는 실제 실행으로 판단한다.',
        same_physics='원 native logν-index 보간·signedχ·반경/ES×J·fullSR·자체 J0=I0=0 보존',
        error_policy='같은 출력의 원천하한·누적원천·하류T상계로 내부 절대예산 배분. 원천1e-8·공간1e-5·출력별 최종오차비1·독립3열각1e-4 유지.',
        cost_policy='10초 실측은 보고하되 국소 정확도 차단에서 분리. 생산 성능 통과로 표시하지 않는다.',
        time_policy='원1800초 상한은 사용자 지시로 확대. 기존1051GPU초+새 최대3600=누적4651초. CPU 빌드는 GPU 할당 밖에서 수행하고 장치시험/수송은3600초 안에 포함한다.',
        independent_reference_scope='원 고정3열이며 전주파수 독립 참조가 아니다.',
        required_before_actual='같은 할당의 작은 GPU 직접시험 exit0',
        field_calls=0,SE_RE_updates=0,no_automatic_retry=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','run'))
    parser.add_argument('path');parser.add_argument('--sha256');args=parser.parse_args()
    if args.mode=='prepare':prepare(args.path)
    else:
        root=Path(args.path).resolve().parent;evidence(root)
        if (root/'device-test.exit').read_text().strip()!='0':raise ValueError('동일 할당 장치시험 미충족')
        raise SystemExit(local.run(args.path,args.sha256))
