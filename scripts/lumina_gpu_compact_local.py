#!/usr/bin/env python3
"""장치 직접시험을 통과한 축약 원천의 실제13003 판정. 자동 재시도/전체장 없음."""
import argparse
from pathlib import Path

import lumina_gpu_local as local
from lumina_gpu_compact import Runtime,CompactDriver

GPU_LIBRARY_SHA='0843dd44e311a8d40588d5312c48a01884c16358b0141af4a598f9112e02ff29'
CPU_GEOMETRY_SHA='6f82306bff6170037a40f09303f33e1cb00d930ddbdeb0cbe38d7b8b61855be4'
PROOFS={
    '/gpfs/kjhan/cmfgen_runs/lumina_compact_shared_direct_20261022-r6DPXa/adapter-test.log':
        '31968af04ef6cc90c6c2a0ada9f46ec649acb0f83f50f9de695acd5d8223c7d1',
    '/gpfs/kjhan/cmfgen_runs/lumina_compact_shared_direct_20261022-r6DPXa/device-test.log':
        '1558b4d8f56605cb75539366aacdb2b0fb3a49bc2fc2d93cef48055fc5408786',
    '/gpfs/kjhan/cmfgen_runs/lumina_compact_shared_direct_20261022-r6DPXa/host-test.log':
        '6d74752ee92b8c9059fe59cec236989a78474f5205284e02dcc25aa2af87bfe3',
    '/gpfs/kjhan/cmfgen_runs/lumina_source_projection_native_20261022-rMBJ2T/selected.json':
        '3f0b967d0afb2fc6717a869c0a523d29738575fe47764157d88bd7212af8da29',
}
local.SCHEMA='lumina-gpu-compact-native-once-v1'
local.LIMITS=dict(local.LIMITS,engine='SPACE_LINEAR_COMPACT_SOURCE_MEASURE_GPU',local_calls=1,
    prep_window_seconds=1800-78,original_preparation_budget_seconds=1800,already_allocated_gpu_seconds=78,
    local_internal_seconds=70,source_relative=1e-8,spatial_relative=1e-5,fine_each_output=True,
    normalization='PER_FREQUENCY_RELATIVE_NO_GLOBAL_BRIGHT_NORM',cache_integrator_workspace_GiB=1,
    external_wall_seconds=75,automatic_retries=0)
local.Runtime=Runtime
local.GpuDriver=CompactDriver


def configure_shared_accuracy():
    local.SCHEMA='lumina-gpu-compact-native-shared-accuracy-once-v1'
    local.LIMITS=dict(local.LIMITS,shared_accuracy_only=True,performance_qualification='UNASSESSED_SHARED')


def configure_double_time():
    if not local.LIMITS.get('shared_accuracy_only'):raise ValueError('현재 승인된 공유 정확도 후속 실행만 허용')
    local.SCHEMA='lumina-gpu-compact-native-shared-double-time-once-v1'
    local.LIMITS=dict(local.LIMITS,local_internal_seconds=140,interval_seconds=120,external_wall_seconds=150,
                      already_allocated_gpu_seconds=148,prep_window_seconds=1800-148,user_approved_double_time=True)
    class DoubleTimeCompactDriver(CompactDriver):
        interval_limit_seconds=120.
    local.GpuDriver=DoubleTimeCompactDriver


def evidence(root):
    for name,digest in [('history_cuda.so',GPU_LIBRARY_SHA),('driver.so',CPU_GEOMETRY_SHA)]:
        if local.M.N.sha(Path(root)/name)!=digest:raise ValueError('직접 검증한 축약/기하 바이너리와 다르다')
    if any(local.M.N.sha(Path(p))!=h for p,h in PROOFS.items()):raise ValueError('원 정확도 증거 변경')


def prepare(root):
    root=Path(root).resolve();evidence(root)
    local.prepare(root,extra_files=tuple(PROOFS),interval_indices=(13003,),comparison_context=dict(
        method='NATIVE_COMPACT_SOURCE_MEASURE_CHI2_ETA32',GPU_library_sha256=GPU_LIBRARY_SHA,
        reference='원 native 입력·J0=I0=0·같은 fullSR 구간의 기존8192분할 독립 세 점',
        criteria='세 점 각각과 최대정규화차1e-4·주파수별 공간오차비1 이하·원천1e-8·D2H포함10초',
        workspace_expected_bytes=1013090856,source_measure_already_GPU_seconds=local.LIMITS['already_allocated_gpu_seconds'],
        small_tests='공유 장치 합성8통과는 정확도 증거다. 성능 수용/실제 구간 완료 증거가 아니다.',
        performance_precheck='공유 정확도 선택 시 사용률만 기록하고 성능은 미판정이다. 기본 모드는 원util5%를 유지한다. free24GiB와 원 실행/오차/10초 목표는 그대로다.',
        no_automatic_retry=True,user_approved_double_time=local.LIMITS.get('user_approved_double_time',False),
        field_calls=0,SE_RE_updates=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','run'));p.add_argument('path');p.add_argument('--sha256')
    p.add_argument('--shared-accuracy',action='store_true',help='실제 크기 정확도만 확인하며 성능 수용은 금지')
    p.add_argument('--double-time',action='store_true',help='사용자 승인 외부150/내부140/구간120초 후속 단회')
    a=p.parse_args()
    if a.shared_accuracy:configure_shared_accuracy()
    if a.double_time:configure_double_time()
    if a.mode=='prepare':prepare(a.path)
    else:
        evidence(Path(a.path).resolve().parent)
        raise SystemExit(local.run(a.path,a.sha256))
