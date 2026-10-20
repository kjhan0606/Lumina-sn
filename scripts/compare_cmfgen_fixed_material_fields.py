#!/usr/bin/env python3
"""같은 완료 물질의 두 실제 복사장 C 응답을 비교한다. 최종 수렴 판정은 아니다."""
import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np

from compare_cmfgen_feedback_candidates import load_report
from summarize_cmfgen_fixed_field_owner import quantiles,read_response
from run_cmfgen_lambda_probe import sha
from analyze_cmfgen_ray_energy import field_difference


def compare(old,new,weights,nd):
    expected={(d,Z,s) for d in range(1,nd+1) for Z in (14,16) for s in range(2,7)}
    if set(old)!=expected or set(new)!=expected:raise ValueError('전단계/깊이 coverage 불일치')
    for key in expected:
        for name in ('Te_K','ne_cm3','n_element','pre_fraction'):
            if old[key][name]!=new[key][name]:raise ValueError('복사장 이외의 물질 변화: '+str((key,name)))
    stages={}
    for Z in (14,16):
        stages[str(Z)]={}
        for s in range(2,7):
            delta=np.array([float(new[d,Z,s]['candidate_fraction'])-float(old[d,Z,s]['candidate_fraction']) for d in range(1,nd+1)])
            stages[str(Z)][str(s)]=dict(absolute_change_mass_median_p95_max=quantiles(abs(delta),weights),
                worst_depth_1based=int(np.argmax(abs(delta)))+1)
    return dict(same_T_ne_element_density_initial_fractions_decimal_byte_equal=True,
        rows=len(expected),per_element_stages=stages)


def run(old_root,new_root,out):
    if out.exists():raise ValueError('기존 장만 비교 출력 보존')
    loaded=[load_report(p) for p in (old_root,new_root)]
    if (loaded[0][0]['record']!=loaded[1][0]['record'] or
        loaded[0][1]!=loaded[1][1] or not np.array_equal(loaded[0][2],loaded[1][2])):
        raise ValueError('다른 물질 세대/C 생산자/질량 격자')
    data=[np.load(p/'completed-state.npz') for p in (old_root,new_root)]
    for name in ('state','profiles','columns','owner_variable_indexes'):
        if not np.array_equal(data[0][name],data[1][name]):raise ValueError('다른 물질 입력: '+name)
    if any(r[0]['depths']!=90 or r[0]['rows']!=900 for r in loaded):raise ValueError('전90점 비교 아님')
    responses=[];field_roots=[];evidence=[]
    for root in (old_root,new_root):
        reference={(int(r['depth_1based']),int(r['Z']),int(r['stage'])):r for r in csv.DictReader((root/'reference.csv').open())}
        responses.append(read_response(root/'C_response',reference,90))
        path=root/'fixed-field-owner-inputs.json';m=json.loads(path.read_text());evidence.append(path)
        fields=[Path(k) for k in m['input_sha256'] if Path(k).name=='FORMAL_PRIMARY_FIELDS.bin']
        if len(fields)!=1 or sha(fields[0])!=m['input_sha256'][str(fields[0])]:raise ValueError('실제 장 신원 불일치')
        field_roots.append(fields[0].parent);evidence.append(fields[0])
    p=data[0]['profiles'];weights=p[:,0]*(p[:,4]**3-p[:,3]**3)
    report=compare(*responses,weights,90)
    report.update(scope=__doc__,material_record=loaded[0][0]['record'],
        old_radiation_record=loaded[0][0]['radiation_record'],new_radiation_record=loaded[1][0]['radiation_record'],
        old_radiation_material_record=loaded[0][0]['radiation_material_record'],
        new_radiation_material_record=loaded[1][0]['radiation_material_record'],
        actual_field_difference=field_difference(field_roots[1],field_roots[0]),
        old_comparison=str(loaded[0][3]),new_comparison=str(loaded[1][3]),
        joint_convergence=False,physical_equivalence='NOT_JUDGED',final_goal='NOT_JUDGED')
    points=[]
    for key in sorted(responses[0]):
        d,Z,s=key;a,b=(r[key] for r in responses)
        points.append(dict(depth_1based=d,Z=Z,stage=s,native_fraction=float(a['pre_fraction']),
            C_old_J=float(a['candidate_fraction']),C_new_J=float(b['candidate_fraction']),
            signed_field_only_change=float(b['candidate_fraction'])-float(a['candidate_fraction'])))
    out.mkdir();producer=Path(__file__).resolve();shutil.copy2(producer,out/producer.name)
    with (out/'fractions.csv').open('x',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(points[0]));w.writeheader();w.writerows(points)
    report.update(input_sha256={str(p):sha(p) for p in [*evidence,*[x[3] for x in loaded],producer]},
        output_sha256=sha(out/'fractions.csv'))
    (out/'field-only.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('old','new','out'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();run(a.old.resolve(),a.new.resolve(),a.out.resolve())
