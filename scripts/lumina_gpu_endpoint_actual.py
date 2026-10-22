#!/usr/bin/env python3
"""끝점 재구성 재사용 후보의 원 전주파수 구간 단회. 전체장은 아니다."""
import argparse
import json
from pathlib import Path

import lumina_gpu_local as local
from lumina_gpu_compact import Runtime,EndpointCompactDriver

GPU_SHA='ae49788865afeee46ba36dfb82dc7279f9a70989df583e4066b4b02438e39335'
CPU_SHA='6f82306bff6170037a40f09303f33e1cb00d930ddbdeb0cbe38d7b8b61855be4'
PROOFS=('host-test-final.log','adapter-test.log','cpu-cost.json','cpu-comparison.json',
    '/gpfs/kjhan/cmfgen_runs/lumina_endpoint_cache_fixed_20261022-ClLEqp/device-test.log')


class EndpointIntervalDriver(EndpointCompactDriver):
    interval_limit_seconds=120.


local.SCHEMA='lumina-exact-endpoint-native-full-interval-once-v1'
local.LIMITS=dict(local.LIMITS,cpus=1,engine='COMPACT_EXACT_ENDPOINT_CACHE_A100',local_calls=1,
    prep_window_seconds=1236,original_preparation_budget_seconds=1800,already_allocated_gpu_seconds=564,
    local_internal_seconds=140,interval_seconds=120,external_wall_seconds=150,slurm_allocation_seconds=180,
    source_relative=1e-8,spatial_relative=1e-5,fine_each_output=True,shared_accuracy_only=False,
    cache_integrator_workspace_GiB=1,normalization='PER_FREQUENCY_RELATIVE_NO_GLOBAL_BRIGHT_NORM',
    automatic_retries=0)
local.Runtime=Runtime
local.GpuDriver=EndpointIntervalDriver


def evidence(root):
    for name,digest in (('history_cuda.so',GPU_SHA),('driver.so',CPU_SHA)):
        if local.M.N.sha(root/name)!=digest:raise ValueError('봉인 끝점/기하 바이너리 불일치')
    comparison=json.loads((root/'cpu-comparison.json').read_text())
    if comparison['status']!='CPU_COARSE_COMPARISON_MET_NOT_FIELD' or len(comparison['rows'])!=35 or \
       not all(row['met'] and row['same_source_bits'] and row['panels_before']==row['panels_after']
               for row in comparison['rows']):raise ValueError('같은 실제35열의 원 적응 수치 대조 미충족')


def prepare(root):
    root=Path(root).resolve();evidence(root)
    local.prepare(root,extra_files=PROOFS,interval_indices=(13003,),comparison_context=dict(
        method='EXACT_ENDPOINT_REUSE_NOT_NEW_PHYSICAL_APPROXIMATION',GPU_library_sha256=GPU_SHA,
        workspace_expected_bytes=1048771496,endpoint_cache_bytes=35680640,
        criteria='원 미세참조3열 각각/최대정규화차1e-4·원천1e-8·공간1e-5·최종출력별오차비1 이하·D2H포함10초',
        previous_combined_GPU_seconds=564,remaining_preparation_seconds=1236,
        earlier_device_tests='같은 바이너리433928의9통과 재사용. fixture tuple 대입 실패는 보존하며 물리 실패가 아니다.',
        required_before_actual='같은 할당/같은 봉인 바이너리의 작은 장치 직접시험 exit0',
        independent_reference_scope='고정3열이며 전주파수 독립 참조가 아니다.',
        field_calls=0,SE_RE_updates=0,no_automatic_retry=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','run'))
    parser.add_argument('path');parser.add_argument('--sha256');args=parser.parse_args()
    if args.mode=='prepare':prepare(args.path)
    else:
        root=Path(args.path).resolve().parent;evidence(root)
        if (root/'device-test.exit').read_text().strip()!='0':raise ValueError('동일 할당 장치시험 미충족')
        raise SystemExit(local.run(args.path,args.sha256))
