#!/usr/bin/env python3
"""같은 완료 물질/T/ne/외부장에서 native Si/S와 실제 C owner 분포를 비교한다."""
import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np

from run_cmfgen_lambda_probe import sha


def quantiles(values,weights):
    values=np.asarray(values);weights=np.asarray(weights)
    if values.ndim!=1 or weights.shape!=values.shape or len(values)==0 or np.any(weights<=0):
        raise ValueError('양의 질량 가중치 필요')
    order=np.argsort(values);cumulative=np.cumsum(weights[order])/weights.sum()
    return [float(values[order[min(np.searchsorted(cumulative,q),len(values)-1)]]) for q in (.5,.95,1.)]


def read_response(root,reference,nd):
    if f'CMF_OWNER_RESPONSE_COMPLETE depths={nd}' not in (root/'owner.log').read_text():
        raise ValueError('전깊이 C owner 미완료')
    rows=list(csv.DictReader((root/'response.csv').open()))
    keys=[(int(r['depth_1based']),int(r['Z']),int(r['stage'])) for r in rows]
    expected={(d,Z,s) for d in range(1,nd+1) for Z in (14,16) for s in range(2,7)}
    if len(set(keys))!=len(keys) or set(keys)!=expected:raise ValueError('행 누락/중복')
    result=dict(zip(keys,rows))
    totals={(d,Z):sum(float(reference[d,Z,s]['population_cm3']) for s in range(2,7))
            for d in range(1,nd+1) for Z in (14,16)}
    for key,row in result.items():
        for name in ('Te_K','ne_cm3','n_element','pre_fraction','candidate_fraction','committed_fraction',
                     'scaled_se_residual','conservation_residual','certificate_probability_error_display',
                     'certificate_relative_error_display'):
            if not row[name] or not np.isfinite(float(row[name])):raise ValueError('비유한 C 결과: '+str((key,name)))
        for name in ('completed','candidate_valid','certificate_pass','verdict_pass','commit_performed'):
            if row[name]!='1':raise ValueError('유효하지 않거나 실제 채택되지 않은 C 결과: '+str((key,name)))
        ref=reference[key]
        if (not np.isclose(float(row['pre_fraction']),float(ref['fraction']),rtol=2e-13,atol=1e-300) or
            float(row['Te_K'])!=float(ref['Te_K']) or float(row['ne_cm3'])!=float(ref['ne_cm3'])):
            raise ValueError('다른 물질/T/ne의 C 응답')
        if not np.isclose(float(row['n_element']),totals[key[:2]],rtol=2e-13,atol=0):
            raise ValueError('원소총밀도가 다른 C 응답')
        if abs(float(row['candidate_fraction'])-float(row['committed_fraction']))>1e-12:
            raise ValueError('실제 채택 인구가 반환 후보와 다름')
    for d in range(1,nd+1):
        for Z in (14,16):
            for name in ('pre_fraction','candidate_fraction','committed_fraction'):
                if abs(sum(float(result[d,Z,s][name]) for s in range(2,7))-1)>2e-12:
                    raise ValueError('원소합 불일치')
    return result


def summarize(root,response,binary,out):
    if out.exists():raise ValueError('기존 완료 상태 비교 보존')
    manifest=json.loads((root/'fixed-field-owner-inputs.json').read_text())
    for name,h in manifest['output_sha256'].items():
        if sha(root/name)!=h:raise ValueError('완료 상태/입력 증거 변경: '+name)
    data=np.load(root/'completed-state.npz');state=data['state'];old=data['previous59']
    parent=data['previous_parent'] if 'previous_parent' in data else (data['previous63'] if 'previous63' in data else old)
    profiles=data['profiles'];indexes=data['owner_variable_indexes'];columns=data['columns'];nd=len(profiles)
    reference={(int(r['depth_1based']),int(r['Z']),int(r['stage'])):r for r in csv.DictReader((root/'reference.csv').open())}
    rows=read_response(response,reference,nd)
    weights=profiles[:,0]*(profiles[:,4]**3-profiles[:,3]**3)
    if not np.isfinite(weights).all() or np.any(weights<=0):raise ValueError('고정 shell 질량 오류')
    points=[];per_element={}
    for Z in (14,16):
        stages={};owned=np.array([i for i,c in enumerate(columns) if c[0]==Z])
        old_total=old[indexes[owned]].sum(axis=0)
        parent_total=parent[indexes[owned]].sum(axis=0)
        for stage in range(2,7):
            native=np.array([float(reference[d,Z,stage]['fraction']) for d in range(1,nd+1)])
            candidate=np.array([float(rows[d,Z,stage]['candidate_fraction']) for d in range(1,nd+1)])
            selected=[indexes[i] for i in owned if columns[i,1]==stage-1]
            previous=old[selected].sum(axis=0)/old_total;absolute=abs(candidate-native)
            parent_fraction=parent[selected].sum(axis=0)/parent_total
            active=np.maximum(candidate,native)>1e-8
            positive=(candidate>0)&(native>0)
            ratios=abs(np.log10(candidate[active&positive]/native[active&positive]))
            stages[str(stage)]=dict(
                absolute_difference_mass_median_p95_max=quantiles(absolute,weights),
                native_change_from59_mass_median_p95_max=quantiles(abs(native-previous),weights),
                native_change_from_parent_mass_median_p95_max=quantiles(abs(native-parent_fraction),weights),
                support_mismatch_count=int(np.sum((candidate>0)!=(native>0))),
                fraction_threshold_subset_count=int(active.sum()),
                fraction_threshold_subset_max_abs_dex=float(ratios.max()) if len(ratios) else None,
                fraction_threshold_subset_is_not_full_heat_opacity_active_set=True,
                worst_depth_1based=int(np.argmax(absolute))+1)
            for d in range(nd):
                points.append(dict(depth_1based=d+1,Z=Z,stage=stage,native59=previous[d],native_record=native[d],
                    native_parent=parent_fraction[d],
                    C_candidate=candidate[d],C_committed=float(rows[d+1,Z,stage]['committed_fraction']),
                    C_minus_native=candidate[d]-native[d],Te_K=profiles[d,1],ne_cm3=profiles[d,2],
                    ne_parent_cm3=parent[-2,d],ne59_cm3=old[-2,d]))
        group=[rows[d,Z,3] for d in range(1,nd+1)]
        per_element[str(Z)]=dict(stages=stages,certified_and_committed=len(group),
            C_max_scaled_SE=max(float(r['scaled_se_residual']) for r in group),
            C_max_conservation=max(float(r['conservation_residual']) for r in group),
            legacy_rank_kappa_failures_preserved=sum(r['legacy_topology_pass']=='0' or r['legacy_numerical_pass']=='0' for r in group))
    out.mkdir()
    with (out/'fractions.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(points[0]));writer.writeheader();writer.writerows(points)
    source=Path(__file__).resolve();shutil.copy2(source,out/source.name)
    report=dict(scope=__doc__,record=manifest['material_snapshot']['record'],depths=nd,rows=len(points),
        material_parent_record=manifest.get('material_parent_record',59),
        radiation_record=manifest.get('radiation_record',60),
        radiation_material_record=manifest.get('radiation_material_record',59),
        field_phase_warning=manifest.get('field_phase_warning','J60는 물질59의 장; 조건부 비교이며 공동수렴 미입증'),
        per_element=per_element,max_ne_change_over59=manifest['max_ne_change_over59'],
        max_native_charge_residual=manifest['max_native_charge_residual'],
        element_total_max_relative_change=manifest['element_total_max_relative_change'],
        weighting='rho*(v_outer^3-v_inner^3), sorted weighted empirical-CDF quantiles, no active-level floor',
        native_full_SE_residual='NOT_EXTRACTED; correction is not substituted',
        fixed_J_conditional_not_joint_convergence=True,CMFGEN_converged=False,
        input_sha256={str(p):sha(p) for p in (root/'fixed-field-owner-inputs.json',root/'completed-state.npz',
            root/'reference.csv',response/'response.csv',response/'owner.log',binary,source)},
        output_sha256=sha(out/'fractions.csv'),physical_equivalence='NOT_JUDGED',final_goal='NOT_JUDGED')
    (out/'fixed-field-owner.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(record=report['record'],per_element=per_element),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','response','binary','out'):parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args();summarize(*(getattr(a,k).resolve() for k in ('root','response','binary','out')))
