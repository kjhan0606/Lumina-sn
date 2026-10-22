#!/usr/bin/env python3
"""첫 실패 열을 기존 sticky 오류값으로만 식별하는 제한 진단."""
import argparse
import json
from pathlib import Path
import re

import lumina_gpu_native_actual as base
from lumina_gpu_native_bin import NativeBinDriver
local=base.local


class LocationDriver(NativeBinDriver):
    interval_limit_seconds=140.


local.SCHEMA='lumina-native-bin-failure-location-only-v1'
local.LIMITS=dict(local.LIMITS,engine='NATIVE_BIN_FAILURE_LOCATION_ONLY',prep_window_seconds=180,
    already_allocated_gpu_seconds=1064,local_internal_seconds=150,interval_seconds=140,
    external_wall_seconds=160,slurm_allocation_seconds=180)
local.GpuDriver=LocationDriver


def decode(code):
    if type(code) is not int:return None
    payload=code-100-100000000
    if payload<0:return None
    column,status=divmod(payload,8)
    if not 0<=column<2000000 or not 1<=status<=7:return None
    return dict(output_column=column,characteristic_status=status,
        scope='첫 관측 실패이며 최소 실패열은 아니다. coarse/두half의 어느 조각인지는 CPU에서 확인한다.')


def prepare(root):
    root=Path(root).resolve();base.evidence(root)
    local.prepare(root,extra_files=base.PROOFS+(
        '/gpfs/kjhan/cmfgen_runs/lumina_native_bin_GPU_20261022-4EK39T/device-test.log',
        '/gpfs/kjhan/cmfgen_runs/lumina_native_bin_GPU_20261022-4EK39T/src/cmf_linear_cuda.cu'),
        interval_indices=(13003,),comparison_context=dict(
            purpose='첫 실패 열 식별만 하는 제한 진단. 전주파수/물리수용 재시도가 아니다.',
            method='동일 계산의 실패 상태값만100000000+8*(p.begin+i)+code로 표시. runtime은100을 더한다.',
            workspace_expected_bytes=1048771496,new_device_buffers=0,
            previous_combined_GPU_seconds=1064,original_preparation_budget_seconds=1800,
            authorized_candidate_cumulative_limit_seconds=4651,
            allocation_seconds=180,interval_seconds=140,internal_seconds=150,external_seconds=160,
            candidate_seconds_already_used=13,candidate_seconds_limit=3600,
            criteria='위치 관측이면 CPU의 같은 열/세 조각에서 재현한다. 미관측을 수치 통과로 표시하지 않는다.',
            earlier_direct_tests='이전 동일 수치13장치시험을 재사용하고 새 바이너리의 범위질의1개를 같은 할당에서 확인한다.',
            field_calls=0,SE_RE_updates=0,no_automatic_retry=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','run'));p.add_argument('path')
    p.add_argument('--sha256');args=p.parse_args();root=Path(args.path).resolve()
    if args.mode=='prepare':prepare(root)
    else:
        root=root.parent;base.evidence(root)
        if (root/'device-test.exit').read_text().strip()!='0':raise ValueError('동일 할당 장치시험 미충족')
        code=local.run(args.path,args.sha256)
        path=root/'gpu-local.json';record=json.loads(path.read_text())
        match=re.search(r'GPU 호출/장치 상태 실패: (\d+)',record.get('error',''))
        record['failure_location']=decode(int(match.group(1))) if match else None
        with (root/'failure-location.json').open('x') as f:
            json.dump(dict(status='FAILURE_LOCATION_OBSERVED' if record['failure_location'] else 'LOCATION_UNASSESSED',
                failure=record['failure_location'],result_sha256=local.M.N.sha(path),final_goal='NOT_MET'),
                f,ensure_ascii=False,indent=2,allow_nan=False)
        raise SystemExit(code)
