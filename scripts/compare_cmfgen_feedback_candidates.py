#!/usr/bin/env python3
"""같은 J64의63/64A/64B 응답으로 다음 장의 출발점만 선택한다.

II–VI 전90점 지표를 모두 사용하며 최종 동질성·SE·온도해제를 판정하지 않는다.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from run_cmfgen_lambda_probe import sha


def element_metrics(report, Z):
    if report['depths'] != 90 or report['rows'] != 900:
        raise ValueError('전90점 비교가 아님')
    element=report['per_element'][str(Z)]
    if element['certified_and_committed'] != 90:
        raise ValueError('실제 C 채택 누락')
    quantiles=[element['stages'][str(s)]['absolute_difference_mass_median_p95_max'] for s in range(2,7)]
    values=np.asarray(quantiles)
    if values.shape != (5,3) or not np.isfinite(values).all() or np.any(values<0) or np.any(np.diff(values,axis=1)<0):
        raise ValueError('분포 차이 지표 오류')
    return dict(max_stage_mass_p95=float(values[:,1].max()),all_stages_depth_max=float(values[:,2].max()),
        support_mismatch_count=sum(element['stages'][str(s)]['support_mismatch_count'] for s in range(2,7)))


def compare(pre, A, B):
    for r,record in ((pre,63),(A,64),(B,64)):
        if r['record'] != record or r['radiation_record'] != 64 or r['radiation_material_record'] != 63:
            raise ValueError('서로 다른 물질/장 세대')
    selected=True;elements={}
    for Z in (14,16):
        values={case:element_metrics(r,Z) for case,r in (('63',pre),('64A',A),('64B',B))}
        improves=all(values['64B'][metric] < min(values['63'][metric],values['64A'][metric])
                     for metric in ('max_stage_mass_p95','all_stages_depth_max'))
        support=all(x['support_mismatch_count']==0 for x in values.values())
        elements[str(Z)]=dict(cases=values,both_metrics_improve=improves,support_consistent=support)
        selected=selected and improves and support
    return dict(per_element=elements,decision='NEXT_OUTER_SUPPORTED' if selected else 'REVIEW_REQUIRED',
        selected_case='64B' if selected else None,physical_equivalence='NOT_JUDGED',
        joint_convergence=False,released_T=False,final_goal='NOT_JUDGED')


def load_report(root):
    path=root/'comparison/fixed-field-owner.json'
    r=json.loads(path.read_text())
    for name,expected in r['input_sha256'].items():
        p=Path(name)
        if p.name=='summarize_cmfgen_fixed_field_owner.py':
            p=path.parent/p.name
        if sha(p)!=expected:
            raise ValueError('고정 비교 증거 변경: '+str(p))
    if sha(path.parent/'fractions.csv') != r['output_sha256']:
        raise ValueError('분포표 변경')
    binaries=[v for k,v in r['input_sha256'].items() if Path(k).name=='bench_cmfgen_owner']
    if len(binaries)!=1:
        raise ValueError('실제 C 바이너리 신원 없음')
    data=np.load(root/'completed-state.npz')
    return r,binaries[0],data['profiles'][:,[0,3,4]],path


def run(pre_root,A_root,B_root,out):
    if out.exists():raise ValueError('기존 후보 비교 보존')
    loaded=[load_report(r) for r in (pre_root,A_root,B_root)]
    if len({x[1] for x in loaded})!=1 or any(not np.array_equal(x[2],loaded[0][2]) for x in loaded[1:]):
        raise ValueError('다른 C 생산자 또는 질량/geometry 가중치')
    manifest=json.loads((B_root/'fixed-field-owner-inputs.json').read_text())
    feedback=Path(manifest['feedback_root'])
    runtime_path=feedback/'seq_logs/feedback-runtime.json'
    runtime=json.loads(runtime_path.read_text())
    if (runtime.get('ab_case')!='64B' or not runtime.get('material_complete') or
        not runtime.get('source_unchanged') or not runtime.get('field_unchanged') or
        runtime.get('printed_minimum_scale')!=1.0 or
        not runtime['full_step_check']['raw_candidate_same_within_print_precision'] or
        not runtime['full_step_check']['actual_full_step_within_print_precision'] or
        runtime['max_charge_residual']>1e-10 or
        max(runtime['element_total_change_over63'].values())>1e-10):
        raise ValueError('완료64B의 전체 step/보존 실증 불충분')
    result=compare(*(x[0] for x in loaded))
    result.update(scope=__doc__,pre_root=str(pre_root),A_root=str(A_root),B_root=str(B_root),
        native_full_SE_residual='NOT_CERTIFIED; C comparison is staging selection only',
        input_sha256={str(p):sha(p) for p in ([x[3] for x in loaded]+[runtime_path,Path(__file__).resolve()])})
    out.mkdir();shutil.copy2(Path(__file__),out/Path(__file__).name)
    (out/'candidate-comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('pre','A','B','out'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();run(a.pre.resolve(),a.A.resolve(),a.B.resolve(),a.out.resolve())
