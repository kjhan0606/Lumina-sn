#!/usr/bin/env python3
"""검증한 A100 바이너리의 실제 전ν13003 단회. 전체장/재시도는 없다."""
import argparse
from pathlib import Path

import lumina_gpu_local as local
from lumina_gpu_compact import Runtime,CompactDriver

GPU_SHA='b288dd5c4d4d31f494448dd9de3cda939b8551338f29d58f862f09d2e9da18ee'
CPU_SHA='6f82306bff6170037a40f09303f33e1cb00d930ddbdeb0cbe38d7b8b61855be4'
PROOFS={
    '/gpfs/kjhan/cmfgen_runs/lumina_a100_total_cap_20261022-K57YkA/gpu-cost.json':
        'e3042313c16352e814a37fd9b5100754925739b708fea58a0552e9d335dacc83',
    '/gpfs/kjhan/cmfgen_runs/lumina_a100_total_cap_20261022-K57YkA/device-test.log':
        '9f97bb226299f07535cd455323489a2330604d563a8773b883eb9a0f9a8b1bbd',
    '/gpfs/kjhan/cmfgen_runs/lumina_a100_total_cap_20261022-K57YkA/cpu-regression.log':
        'cd35a5f9814505c502f26f5f8f2ac4658905157404b03458cb11094206e45f69',
    '/gpfs/kjhan/cmfgen_runs/lumina_adaptive_exp_20261022-dqpNi8/fixed737/case.json':
        '347b00d1379f7d8de2e7eceda45a146467b9401f42dd23194d729655102eca47',
    '/gpfs/kjhan/cmfgen_runs/lumina_adaptive_exp_20261022-dqpNi8/host-test.log':
        '6969118987d9bb5301c2fde5ca6d5f83fe913f961bc9ccb5818f7f2ced2129e2',
}


class A100IntervalDriver(CompactDriver):
    # 기존 승인120초 후속 경로만 선택한다. 기본 CompactDriver60초는 유지한다.
    interval_limit_seconds=120.


local.SCHEMA='lumina-a100-native-full-interval-once-v1'
local.LIMITS=dict(local.LIMITS,cpus=1,engine='COMPACT_ADAPTIVE_A100_TOTAL_CAP',local_calls=1,
    prep_window_seconds=1378,original_preparation_budget_seconds=1800,already_allocated_gpu_seconds=422,
    Ada_already_allocated_gpu_seconds=394,local_internal_seconds=140,interval_seconds=120,
    external_wall_seconds=150,slurm_allocation_seconds=180,source_relative=1e-8,spatial_relative=1e-5,
    fine_each_output=True,shared_accuracy_only=False,cache_integrator_workspace_GiB=1,
    normalization='PER_FREQUENCY_RELATIVE_NO_GLOBAL_BRIGHT_NORM',automatic_retries=0)
local.Runtime=Runtime
local.GpuDriver=A100IntervalDriver


def evidence(root):
    for name,digest in (('history_cuda.so',GPU_SHA),('driver.so',CPU_SHA)):
        if local.M.N.sha(Path(root)/name)!=digest:raise ValueError('검증한 A100/기하 바이너리 신원 불일치')
    if any(local.M.N.sha(Path(path))!=digest for path,digest in PROOFS.items()):
        raise ValueError('원 CPU/GPU 직접시험 증거 변경')


def prepare(root):
    root=Path(root).resolve();evidence(root)
    local.prepare(root,extra_files=tuple(PROOFS),interval_indices=(13003,),comparison_context=dict(
        method='A100_ADAPTIVE_NATIVE_BIN_EXP_TOTAL_CAP',GPU_library_sha256=GPU_SHA,
        reference='같은 원 입력/fullSR13003의 기존8192분할 미세참조3열',
        criteria='각 열/최대정규화차1e-4·원천1e-8·공간1e-5·주파수별오차비1 이하·D2H포함10초',
        workspace_expected_bytes=1013090856,strict_performance_precheck='free24GiB/util5% 이하',
        previous_A100_seconds=28,remaining_preparation_seconds=1378,
        previous_Ada_seconds=394,small_tests_reused=True,field_calls=0,SE_RE_updates=0,
        no_automatic_retry=True,independent_reference_scope='고정3열이며전ν독립참조가아니다.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','run'))
    parser.add_argument('path');parser.add_argument('--sha256');args=parser.parse_args()
    if args.mode=='prepare':prepare(args.path)
    else:
        evidence(Path(args.path).resolve().parent)
        raise SystemExit(local.run(args.path,args.sha256))
