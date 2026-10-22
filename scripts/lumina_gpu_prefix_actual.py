#!/usr/bin/env python3
"""실패569899의 접두 τ 오차 수정 후 원13003 완성값을 확보한다."""
import argparse
import json
from pathlib import Path
import re

import lumina_gpu_native_actual as base
from lumina_gpu_native_bin import NativeBinDriver
from lumina_gpu_failure_code import decode
local=base.local
base.CPU_PROOF=Path('/gpfs/kjhan/cmfgen_runs/lumina_native_bin_prefix_20261022-UU9dmL')
LOCAL_PROOF=base.CPU_PROOF/'cpu-three-pieces.json'
GLOBAL_PROOF=Path('/gpfs/kjhan/cmfgen_runs/lumina_prefix569899_global_20261022-IZWHDt/cpu-three-pieces.json')
PROOFS=tuple(str(base.CPU_PROOF/p) for p in ('host-test.log','host-prefix-before.log',
    'cpu-cost.json','cpu-comparison.json','cpu-probe.so','src/cmf_native_bin_stream.hpp'))+(
    str(LOCAL_PROOF),str(GLOBAL_PROOF),'host-test.log','compile.log','original-compile.log','binary.sha256',
    '/gpfs/kjhan/cmfgen_runs/lumina_prefix_GPU_20261022-kbb8me/device-test.log')


class PrefixDriver(NativeBinDriver):
    interval_limit_seconds=3240.


local.SCHEMA='lumina-prefix-tau-uncertainty-full-interval-v1'
local.LIMITS=dict(local.LIMITS,engine='NATIVE_BIN_PREFIX_UNCERTAINTY_A100',prep_window_seconds=3480,
    already_allocated_gpu_seconds=1078,cumulative_preparation_limit_seconds=4651,
    local_internal_seconds=3340,interval_seconds=3240,external_wall_seconds=3420,slurm_allocation_seconds=3480)
local.GpuDriver=PrefixDriver


def evidence(root):
    base.evidence(root)
    for path,mode in ((LOCAL_PROOF,'ANCHORED_LOCAL_TREE'),(GLOBAL_PROOF,'ORIGINAL_GLOBAL_TREE')):
        proof=json.loads(path.read_text())
        if proof['status']!='CPU_THREE_PIECE_SAMPLE_FINISHED' or proof['coordinate_mode']!=mode or \
           len(proof['rows'])!=3 or {r['piece'] for r in proof['rows']}!={0,1,2} or \
           any(r['column']!=569899 or not r['comparison_met'] for r in proof['rows']) or \
           proof['libraries'][str(base.CPU_PROOF/'cpu-probe.so')]!=local.M.N.sha(base.CPU_PROOF/'cpu-probe.so'):
            raise ValueError('실패569899의 세 조각/양 좌표 원 T/S/오차 대조 미충족')


def prepare(root):
    root=Path(root).resolve();evidence(root)
    local.prepare(root,extra_files=PROOFS,interval_indices=(13003,),comparison_context=dict(
        method='EXACT_NATIVE_BIN_SOURCE_WITH_PREFIX_TAU_UNCERTAINTY_BOUND',
        workspace_expected_bytes=1048771496,GPU_library_sha256=local.M.N.sha(root/'history_cuda.so'),
        fix='구간[0,b]의 quadratic Bernstein 누적 기저는 각각min(b,1/3)이하. 감쇠에는 전체bin오차×min(1,3b)를 쓰고 전체T/상류합성은원전체오차를유지한다.',
        evidence='수정전독립τ1e8/1e10시험은분할한도실패. 수정후27시험·원실제35열·실패569899의세조각/국소-전역대조모두통과.',
        criteria='원천1e-8·공간1e-5·최종출력별오차비1·원독립3열각1e-4유지. 10초는실측보고이며이번국소정확도차단에서분리.',
        time_budget='기존1078초+별도warp최대60초+새3480초=4618초<=누적4651초. 시험과수송은같은3480초할당안. 내부3340/구간3240/외부3420은중첩한도다.',
        required_before_actual='동일바이너리434008의13통과재사용. 자체직전부하를외부부하로오인한모듈별문맥fixture오류는보존하고같은할당에서미실행τ1e8/1e10해석해2사례exit0확인.',
        own_field='자체J0=I0=0/fullSR이며CMFGEN J를자체입력으로쓰지않는다.',
        independent_reference_scope='고정3열이며전주파수독립참조가아니다.',
        field_calls=0,SE_RE_updates=0,no_automatic_retry=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','run'));p.add_argument('path')
    p.add_argument('--sha256');args=p.parse_args();root=Path(args.path).resolve()
    if args.mode=='prepare':prepare(root)
    else:
        root=root.parent;evidence(root)
        if (root/'device-test.exit').read_text().strip()!='0':raise ValueError('동일할당장치직접시험미충족')
        code=local.run(args.path,args.sha256);path=root/'gpu-local.json';record=json.loads(path.read_text())
        match=re.search(r'GPU 호출/장치 상태 실패: (\d+)',record.get('error',''))
        with (root/'failure-location.json').open('x') as f:
            json.dump(dict(failure=decode(int(match.group(1))) if match else None,
                result_sha256=local.M.N.sha(path),final_goal='NOT_MET'),f,ensure_ascii=False,indent=2,allow_nan=False)
        raise SystemExit(code)
