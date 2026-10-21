#!/usr/bin/env python3
"""검증된 원천 측도 GPU 후보의 실제13003 전ν 한 번. 전체장은 실행하지 않는다."""
import argparse
from pathlib import Path

import lumina_gpu_local as local
from lumina_gpu_measure import Runtime,MeasureDriver

GPU_LIBRARY_SHA='4c716b172dd7d8d599bcb19741ff776d7d07b6a3a703fe7c161ffb107442a590'
CPU_GEOMETRY_SHA='6f82306bff6170037a40f09303f33e1cb00d930ddbdeb0cbe38d7b8b61855be4'
PROOFS={
    '/gpfs/kjhan/cmfgen_runs/lumina_source_projection_native_20261022-rMBJ2T/selected.json':
        '3f0b967d0afb2fc6717a869c0a523d29738575fe47764157d88bd7212af8da29',
    '/gpfs/kjhan/cmfgen_runs/lumina_source_projection_GPU_weak_20261022-5FuUfA/host-test.log':
        '723222e451acb78959223ea400a020aca43d42543c272cdacaf45bc4ad6999f1',
    '/gpfs/kjhan/cmfgen_runs/lumina_source_projection_GPU_20261022-xQNYu9/device-test.log':
        '74e674960299d592f3e3f059ef88240ae21ef08d62ca7f2a8d3733d78b5540fd',
    '/gpfs/kjhan/cmfgen_runs/lumina_source_projection_GPU_weak_20261022-5FuUfA/device-test.log':
        '073b1bbed3fedd224b758a734fc16517c5bf3d17bd10f69d131ecd1b1c9690ca',
}
local.SCHEMA='lumina-gpu-source-measure-native-once-v1'
local.LIMITS=dict(local.LIMITS,engine='SPACE_LINEAR_NATIVE_SOURCE_MEASURE_GPU',local_calls=1,
    prep_window_seconds=1800-6,original_preparation_budget_seconds=1800,already_allocated_gpu_seconds=6,
    local_internal_seconds=70,source_relative=1e-8,spatial_relative=1e-5,fine_each_output=True,
    normalization='PER_FREQUENCY_RELATIVE_NO_GLOBAL_BRIGHT_NORM',cache_integrator_workspace_GiB=1,
    external_wall_seconds=75,automatic_retries=0)
local.Runtime=Runtime
local.GpuDriver=MeasureDriver


def evidence(root):
    for name,digest in (('history_cuda.so',GPU_LIBRARY_SHA),('driver.so',CPU_GEOMETRY_SHA)):
        if local.M.N.sha(Path(root)/name)!=digest:raise ValueError('직접 검증한 수치/기하 바이너리와 다르다')
    if any(local.M.N.sha(Path(p))!=h for p,h in PROOFS.items()):raise ValueError('원 통과/실패 증거 변경')


def prepare(root):
    root=Path(root).resolve();evidence(root)
    local.prepare(root,extra_files=tuple(PROOFS),interval_indices=(13003,),comparison_context=dict(
        method='NATIVE_SOURCE_MEASURE_N25',same_GPU_binary_as_small_tests=GPU_LIBRARY_SHA,
        interval0_or_CPU_C_comparison='이 실행에서는 선택하지 않는다. 다른 연산자의3e-12를 상속하지 않는다.',
        reference='원 native 입력·J0=I0=0·같은 fullSR 구간의 기존8192분할 독립 세 점',
        criteria='세 점 각각과 최대정규화차1e-4·주파수별 공간오차비1 이하·원천1e-8·D2H포함10초',
        reused_tests='167 CPU·일반 장치6·ES>0/J0=0 weak1. 원 ES=0 fixture 실패도 보존.',
        work_units='χ/η build/query/replay의 구조 작업 단위이며 scalar FLOP가 아니다. 총512×NF는 유지한다.',
        field_calls=0,SE_RE_updates=0,large_source_SHA='기존 봉인 원자료 신원 상속; 본문 재해시 없음'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','run'));p.add_argument('path');p.add_argument('--sha256')
    a=p.parse_args()
    if a.mode=='prepare':prepare(a.path)
    else:
        evidence(Path(a.path).resolve().parent)
        raise SystemExit(local.run(a.path,a.sha256))
