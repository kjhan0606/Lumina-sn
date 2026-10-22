#!/usr/bin/env python3
"""사용자 시간확대 승인에 따른 검증된 끝점 경로의 전주파수 구간 단회."""
import argparse
from pathlib import Path

import lumina_gpu_local as local
from lumina_gpu_compact import Runtime,EndpointCompactDriver
from lumina_gpu_endpoint_actual import evidence,GPU_SHA,CPU_SHA

PROOFS=('host-test-final.log','adapter-test.log','long-adapter-test.log','cpu-cost.json','cpu-comparison.json',
    '/gpfs/kjhan/cmfgen_runs/lumina_endpoint_cache_fixed_20261022-ClLEqp/device-test.log',
    '/gpfs/kjhan/cmfgen_runs/lumina_endpoint_followup_20261022-0YoWCi/device-test.log',
    '/gpfs/kjhan/cmfgen_runs/lumina_endpoint_followup_20261022-0YoWCi/device-test.exit')


class LongEndpointDriver(EndpointCompactDriver):
    interval_limit_seconds=3400.


local.SCHEMA='lumina-exact-endpoint-full-interval-user-extended-v1'
local.LIMITS=dict(local.LIMITS,cpus=1,engine='COMPACT_EXACT_ENDPOINT_CACHE_A100_LONG',local_calls=1,
    prep_window_seconds=3600,original_preparation_budget_seconds=1800,already_allocated_gpu_seconds=847,
    user_extended_time_budget=True,local_internal_seconds=3500,interval_seconds=3400,
    external_wall_seconds=3550,slurm_allocation_seconds=3600,require_cost_target=False,
    source_relative=1e-8,spatial_relative=1e-5,fine_each_output=True,shared_accuracy_only=False,
    cache_integrator_workspace_GiB=1,normalization='PER_FREQUENCY_RELATIVE_NO_GLOBAL_BRIGHT_NORM',
    automatic_retries=0)
local.Runtime=Runtime
local.GpuDriver=LongEndpointDriver


def prepare(root):
    root=Path(root).resolve();evidence(root)
    if Path(PROOFS[-1]).read_text().strip()!='0':raise ValueError('재사용 장치 직접시험 미충족')
    local.prepare(root,extra_files=PROOFS,interval_indices=(13003,),comparison_context=dict(
        method='VERIFIED_EXACT_ENDPOINT_REUSE_NOT_EXPERIMENTAL_INTEGRATOR',
        GPU_library_sha256=GPU_SHA,CPU_library_sha256=CPU_SHA,
        workspace_expected_bytes=1048771496,endpoint_cache_bytes=35680640,
        authority='사용자2026-10-22: 전주파수 수송 제한이 의미 없으면 충분히 확대하라',
        time_policy='원1800초 준비 상한을 명시 확대한다. 이전847GPU초를 보존하며 새 할당 최대3600초만 별도 예약한다.',
        cost_policy='10초 실측 결과는 보고하되 이번 국소 정확도 판정을 차단하지 않는다. 생산 성능 충족으로 표시하지 않는다.',
        criteria='원 미세참조3열 각각/최대정규화차1e-4·원천1e-8·공간1e-5·최종출력별오차비1 이하',
        device_proof='동일 ae497 바이너리의 기존9+후속4장치시험을 재사용하며 반복하지 않는다.',
        previous_combined_GPU_seconds=847,independent_reference_scope='고정3열이며 전주파수 독립 참조가 아니다.',
        field_calls=0,SE_RE_updates=0,no_automatic_retry=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','run'))
    parser.add_argument('path');parser.add_argument('--sha256');args=parser.parse_args()
    if args.mode=='prepare':prepare(args.path)
    else:
        root=Path(args.path).resolve().parent;evidence(root)
        if Path(PROOFS[-1]).read_text().strip()!='0':raise ValueError('재사용 장치 직접시험 미충족')
        raise SystemExit(local.run(args.path,args.sha256))
