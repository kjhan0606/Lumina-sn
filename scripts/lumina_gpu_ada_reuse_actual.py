#!/usr/bin/env python3
"""검증한 같은 Ada 바이너리·원입력 대조를 재사용해 전주파수를 단회 실행한다."""
import argparse
import json
from pathlib import Path
import re

import lumina_gpu_hybrid_actual as inherited
from lumina_gpu_failure_code import decode

local=inherited.local
local.SCHEMA='lumina-ada-hybrid-verified-reuse-full-interval-v1'
local.LIMITS=dict(local.LIMITS,engine='WARP_NATIVE_LIMIT_FALLBACK_VERIFIED_ADA_SM89',
    prep_window_seconds=3598,already_allocated_gpu_seconds=4369,cumulative_preparation_limit_seconds=8001,
    slurm_allocation_seconds=3598,manual_GPU=True,execution_host='lageunha',
    carried_parallel_work=32*1115020)
LIBRARY_SHA='6eae8c4c69b09fa2e77262dd8a96549fad59bdb5b563b5516a6f1b25e6fcae6e'
PROOFS={
    '/gpfs/kjhan/cmfgen_runs/lumina_hybrid_ada_verify_20261022-Lj8pE3/device-test.log':
        'fd584518e7c7f2706a91307473120dfacf1d6732fae3b449191a717e83bcee83',
    '/gpfs/kjhan/cmfgen_runs/lumina_hybrid_ada_remaining_20261022-ZJzFIo/device-test.log':
        '5c67063d1e0338383cc01cb473b29bdbe1d791b35a3cba68a4723d101f18d887',
    '/gpfs/kjhan/cmfgen_runs/lumina_hybrid_ada_dispatch_20261022-dMEGpn/device-test.log':
        '7a2943ebf5c093db893246001f0cf114ce84658cb9f2e6985fdf3692bdecd515',
    '/gpfs/kjhan/cmfgen_runs/lumina_hybrid_ada_dispatch_20261022-dMEGpn/gpu-cost.json':
        '131fe039ffc29aac15bcdf5e639ce5f211a80fbb0642a6d5ea7cfd7843bc3456',
}


def evidence(root):
    inherited.evidence(root)
    if local.M.N.sha(root/'history_cuda.so')!=LIBRARY_SHA:raise ValueError('같은 검증 Ada 바이너리 필요')
    if any(local.M.N.sha(path)!=digest for path,digest in PROOFS.items()):raise ValueError('재사용 장치/원입력 증거 변경')
    diagnostic=json.loads(Path(next(p for p in PROOFS if p.endswith('gpu-cost.json'))).read_text())
    if diagnostic['status']!='LIMITED_GPU_COST_DIAGNOSIS_FINISHED_NOT_FIELD' or \
       diagnostic['library_sha256']!=LIBRARY_SHA or diagnostic['query_columns']!=870 or \
       len(diagnostic['comparisons'])!=44 or not all(row['met'] for row in diagnostic['comparisons']):
        raise ValueError('원870질의/44CPU대조 미충족')


def prepare(root):
    root=Path(root).resolve();evidence(root)
    local.prepare(root,extra_files=inherited.prior.PROOFS+tuple(PROOFS)+('diagnostic.so','cpu-cost.json'),
        interval_indices=(13003,),comparison_context=dict(
            method='SAME_VERIFIED_ADA_WARP_NATIVE_BINARY_REUSE_NO_REPEATED_SMALL_TESTS',
            GPU_library_sha256=LIBRARY_SHA,workspace_expected_bytes=1070005192,device_proofs=PROOFS,
            criteria='원천1e-8/공간1e-5/완성출력별추정오차비1/독립미세3열각1e-4 유지.10초성능은보고.',
            direct_scope='기본13+warp14+기본성공1+실제work-limit대체1=고유29통과. 두꺼운bin값은정확했으나대체가정시험실패를보존.',
            actual_sample='같은원870질의/44대조·실패569899세조각을완료했고이번부모에서는반복하지않음.',
            cost='이전4369GPU초이월+이번최대3598<=원확대누적8001초.사용0초취소한434129는중복정산하지않음.',
            work='원512NF에서준비32NF를차감. 세조각추가로총4트리를만드는진단을기존8NF한번으로과소계상하지않음.',
            own_field='자체J0=I0=0/fullSR. 새재구성후보미채택. 전체장/SE_RE/자동재시도0.',
            no_automatic_retry=True,final_goal='NOT_MET'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','run'));parser.add_argument('path')
    parser.add_argument('--sha256');args=parser.parse_args();root=Path(args.path).resolve()
    if args.mode=='prepare':prepare(root)
    else:
        evidence(root.parent);code=local.run(root,args.sha256);result=root.parent/'gpu-local.json'
        match=re.search(r'GPU 호출/장치 상태 실패: (\d+)',json.loads(result.read_text()).get('error',''))
        with (root.parent/'failure-location.json').open('x') as stream:
            json.dump(dict(failure=decode(int(match.group(1))) if match else None,
                result_sha256=local.M.N.sha(result),final_goal='NOT_MET'),stream,ensure_ascii=False,indent=2,allow_nan=False)
        raise SystemExit(code)
