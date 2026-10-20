#!/usr/bin/env python3
"""기존65–67 또는 명시 등록된 유한 계약의 증거를 판독한다. 제출은 하지 않는다.

같은 장의 조건부 일치와 공동수렴을 구분한다. 모든 결과의 최종 물리 목표는
미판정이며, 이 도구는 승인된 최대2outer를 자동으로 연장하지 않는다.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from compare_cmfgen_feedback_candidates import load_report,element_metrics
from run_cmfgen_lambda_probe import sha


def maxima(stages,key):
    a=np.array([stages[str(s)][key] for s in range(2,7)],dtype=float)
    if a.shape!=(5,3) or not np.isfinite(a).all() or np.any(a<0) or np.any(np.diff(a,axis=1)<0):
        raise ValueError('전단계 질량 quantile 오류')
    return [float(a[:,1].max()),float(a[:,2].max())]


def field_material_vector(frame):
    values = np.array([value for Z in ('14', '16') for key in ('field', 'material')
        for value in frame['elements'][Z][key]], dtype=float)
    if values.shape != (8,) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('양원소 F/P 총8성분 오류')
    return values


def decide(frames, *, coupling=False, coupling_initial_record=67):
    records=[f['record'] for f in frames]
    if coupling_initial_record not in (67,69) or (not coupling and coupling_initial_record != 67):
        raise ValueError('등록되지 않은 명시 계약 세대')
    initial = coupling_initial_record if coupling else 65
    allowed=list(range(initial,initial+3))
    if not frames or len(frames)>3 or records!=allowed[:len(frames)]:
        raise ValueError('이번 최대2outer 계약의 연속 증거만 허용')
    result=dict(records=records,physical_equivalence='NOT_JUDGED',joint_convergence=False,
        released_T=False,final_goal='NOT_JUDGED',submits_calculations=False)
    followup = coupling and initial == 69
    if followup:
        for frame in frames:
            field_material_vector(frame)
            if not np.isfinite(frame['ne_step']) or frame['ne_step'] < 0:
                raise ValueError('실제 전자밀도 step 오류')
    if len(frames)==1:
        return dict(result,decision=f'WAIT_FOR_COMPLETED{initial+1}')
    if len(frames)==3:
        if followup:
            ne_below = all(frame['ne_step'] < .005 for frame in frames[1:])
            fp_below = bool(np.all(field_material_vector(frames[-1]) < field_material_vector(frames[0])))
            result['attenuation_observation'] = dict(reference_record=69,
                each_new_ne_step_below_half_percent=ne_below,
                final_F_P_all_components_below_reference=fp_below,
                objective_observed=ne_below and fp_below,
                physical_equivalence_acceptance=False)
        return dict(result,decision='LIMIT_REACHED_EVALUATE',automatic_extension=False,
            latest_full_step_premise=frames[-1]['full_step_premise'])
    a,b=frames
    if coupling and b.get('coupling_step_permitted') is not True:
        return dict(result,decision=f'STOP_BEFORE{initial+2}_STEP_CAUSE_UNPROVEN',
            **{f'completed{initial+1}_preserved':True})
    if not coupling and b['full_step_premise'] is not True:
        return dict(result,decision='STOP_BEFORE67_FULL_STEP_UNPROVEN',completed66_preserved=True)
    per_element={}
    all_deteriorate=True
    for Z in ('14','16'):
        before=np.array(a['elements'][Z]['field']+a['elements'][Z]['material'])
        after=np.array(b['elements'][Z]['field']+b['elements'][Z]['material'])
        cond_before=np.array(a['elements'][Z]['conditional'])
        cond_after=np.array(b['elements'][Z]['conditional'])
        worsening=bool(np.all(after>=before) and np.any(after>before))
        conditional_not_better=bool(np.all(cond_after>=cond_before))
        per_element[Z]=dict(field_and_material_all_non_decreasing_with_increase=worsening,
                            conditional_no_component_improves=conditional_not_better)
        all_deteriorate=all_deteriorate and worsening and conditional_not_better
    ne_not_better=b['ne_step']>=a['ne_step']
    result.update(per_element=per_element,ne_not_improved=ne_not_better)
    if followup:
        before,after=field_material_vector(a),field_material_vector(b)
        all_deteriorate=bool(np.all(after>=before) and np.any(after>before))
        result.update(conditional_used_for_early_stop=False,
            all_field_material_non_decreasing_with_increase=all_deteriorate)
    target=initial+2
    return dict(result,decision=f'SKIP{target}_JOINT_DETERIORATION' if all_deteriorate and ne_not_better else f'CONTINUE{target}_WITHIN_LIMIT')


def cycle_candidates(first,middle,last):
    if not (first.shape==middle.shape==last.shape):raise ValueError('상태 경로 차원 오류')
    a=middle-first;b=last-middle;net=last-first
    flip=(np.sign(a)!=np.sign(b))&(a!=0)&(b!=0)
    candidates=flip&(abs(net)<abs(a))&(abs(net)<abs(b))
    amplitude=np.minimum(abs(a),abs(b))
    location=list(map(int,np.unravel_index(int(np.argmax(np.where(candidates,amplitude,-np.inf))),amplitude.shape))) if candidates.any() else None
    return dict(sign_reversals=int(flip.sum()),two_step_closer_than_both_one_steps=int(candidates.sum()),
        maximum_reversing_amplitude=float(amplitude[flip].max()) if flip.any() else 0.,
        maximum_cycle_candidate_amplitude=float(amplitude[candidates].max()) if candidates.any() else 0.,
        maximum_candidate_index_0based=location,
        interpretation='같은 shell/stage의2주기 후보다. 작은 진폭도 포함하며 안정/불안정의 증명이나 최종합격이 아니다.')


def sealed(path,cache):
    r=json.loads(path.read_text())
    for name,expected in r.get('input_sha256',{}).items():
        p=Path(name)
        if p.name=='compare_cmfgen_fixed_material_fields.py':p=path.parent/p.name
        key=str(p.resolve())
        if key not in cache:cache[key]=sha(p)
        if cache[key]!=expected:raise ValueError('기록된 증거 변경: '+str(p))
    if sha(path.parent/'fractions.csv')!=r['output_sha256']:raise ValueError('전90점 분포표 변경')
    return r


def load_frame(owner,field_only,cache):
    r,binary,geometry,comparison=load_report(owner)
    field=sealed(field_only,cache)
    record=r['record']
    if (r['radiation_record']!=record or r['radiation_material_record']!=record-1 or
        r['material_parent_record']!=record-1 or field['material_record']!=record-1 or
        field['old_radiation_record']!=record-1 or field['new_radiation_record']!=record or
        field['new_radiation_material_record']!=record-1 or field['rows']!=900 or
        not field['same_T_ne_element_density_initial_fractions_decimal_byte_equal']):
        raise ValueError('장/물질 세대 또는 동일물질 대조 오류')
    m=json.loads((owner/'fixed-field-owner-inputs.json').read_text())
    runtime_path=Path(m['feedback_root'])/'seq_logs/feedback-runtime.json'
    runtime=json.loads(runtime_path.read_text())
    if (not runtime.get('material_complete') or not runtime.get('field_complete') or
        not runtime.get('source_unchanged') or runtime.get('failure') or
        runtime['field']['returncode']!=97 or runtime['field'].get('stop_reason') or
        runtime['material']['returncode']!=0 or runtime['material'].get('stop_reason') or
        runtime['material_record']!=record or runtime['radiation_produced_from_record']!=record-1):
        raise ValueError('실제 완료 교대 상태가 아님')
    for name,expected in runtime['material_evidence_sha256'].items():
        p=Path(m['run'])/name;key=str(p.resolve())
        if key not in cache:cache[key]=sha(p)
        if cache[key]!=expected:raise ValueError('현재 native 완료 증거 변경')
    data=np.load(owner/'completed-state.npz');state=data['state'];parent=data['previous_parent']
    pre_owner=Path(field['new_comparison']).parent.parent
    if not np.array_equal(np.load(pre_owner/'completed-state.npz')['state'],parent):
        raise ValueError('장단독 비교의 물질과 실제 부모가 다름')
    ne_step=float(np.max(abs(state[-2]/parent[-2]-1)))
    if ne_step!=runtime['max_ne_relative_step']:raise ValueError('실제 ne step 불일치')
    fractions=[];elements={}
    for Z in (14,16):
        q=element_metrics(r,Z)
        if q['support_mismatch_count']:raise ValueError('조건부 분포 support 불일치')
        elements[str(Z)]=dict(field=maxima(field['per_element_stages'][str(Z)],'absolute_change_mass_median_p95_max'),
            material=maxima(r['per_element'][str(Z)]['stages'],'native_change_from_parent_mass_median_p95_max'),
            conditional=[q['max_stage_mass_p95'],q['all_stages_depth_max']])
        columns=data['columns'];indexes=data['owner_variable_indexes'];owned=indexes[columns[:,0]==Z]
        total=state[owned].sum(axis=0)
        for stage in range(2,7):
            chosen=indexes[(columns[:,0]==Z)&(columns[:,1]==stage-1)]
            fractions.append(state[chosen].sum(axis=0)/total)
    return dict(record=record,elements=elements,ne_step=ne_step,
        full_step_premise=(runtime.get('next_outer_eligible') is True and
            runtime.get('printed_minimum_scale')==1. and runtime.get('actual_full_step_within_print_precision') is True) if record>=66 else None,
        comparison=str(comparison),field_only=str(field_only),runtime=str(runtime_path),
        _state=state,_parent=parent,_fraction=np.array(fractions),_ne=state[-2],_geometry=geometry,_binary=binary)


def run(pairs,out,*,coupling_contract=None,coupling_contract_sha256=None):
    if out.exists():raise ValueError('기존 묶음 판독 보존')
    coupling=None
    if coupling_contract is not None:
        from cmfgen_coupling_contract import load_contract
        coupling=load_contract(coupling_contract,coupling_contract_sha256)
    elif coupling_contract_sha256 is not None:
        raise ValueError('경로 없는 부분-step 계약')
    cache={};frames=[load_frame(o,f,cache) for o,f in pairs]
    if coupling is not None:
        for frame in frames:
            if frame['_binary']!=coupling['C_binary_sha256']:
                raise ValueError('수정 C1과 다른 분포/장 민감도 혼용 금지')
            if frame['record']==coupling['initial_record']:
                if sha(Path(frame['runtime'])) != coupling['initial_runtime_sha256']:
                    raise ValueError('별도 계약의 봉인 초기 물질과 다른 기준 frame')
            if frame['record']>coupling['initial_record']:
                runtime=json.loads(Path(frame['runtime']).read_text())
                if runtime.get('coupling_contract_sha256')!=coupling['_sha256']:
                    raise ValueError('다른 장 결합 계약의 실행')
                observed=runtime.get('native_SE_assessment',{})
                if (observed.get('phase',{}).get('parent')!=frame['record']-1 or
                    observed.get('radiation_record')!=frame['record'] or
                    observed.get('radiation_produced_from_record')!=frame['record']-1):
                    raise ValueError('물질/장/원시 SE 관찰 phase 불일치')
                frame['coupling_step_permitted']=runtime.get('coupling_next_input_permitted') is True
    for a,b in zip(frames,frames[1:]):
        if (a['_binary']!=b['_binary'] or not np.array_equal(a['_geometry'],b['_geometry']) or
            not np.array_equal(a['_state'],b['_parent'])):
            raise ValueError('다른 생산자/질량격자 또는 끊긴 실제 상태 경로')
    result=decide(frames,coupling=coupling is not None,
        coupling_initial_record=coupling['initial_record'] if coupling else 67)
    if len(frames)==3:
        result['cycle_candidates']=dict(populations={str(Z):cycle_candidates(*[f['_fraction'][j*5:(j+1)*5] for f in frames])
            for j,Z in enumerate((14,16))},ne=cycle_candidates(*[f['_ne'] for f in frames]),
            population_axes='stage II–VI index0–4, depth index0–89; 진폭은 분율')
    result.update(scope=__doc__,frames=[{k:v for k,v in f.items() if not k.startswith('_')} for f in frames],
        input_sha256={str(p):sha(p) for p in [*[Path(f[k]) for f in frames for k in ('comparison','field_only','runtime')],Path(__file__).resolve()]},
        rule='양원소의F/P 네성분모두비감소·일부증가 및 N/Q도개선없을때만67생략; 혼합/감소는67까지. 67뒤자동연장없음.')
    if coupling is not None:
        result.update(coupling_contract_sha256=coupling['_sha256'],coupling_purpose=coupling['purpose_id'],
            rule=coupling['second_outer_decision'])
        result['input_sha256'][str(coupling_contract)]=coupling['_sha256']
        if coupling['initial_record']==69 and len(frames)==3:
            result['end_rule']=coupling['end_decision']
        if len(frames)==2:
            result['selected_material']=json.loads((pairs[1][0]/'fixed-field-owner-inputs.json').read_text())['run']
    out.mkdir();shutil.copy2(Path(__file__),out/Path(__file__).name)
    if coupling is not None:
        # 이후 작업트리 편집이 이미 만든 선택 증거의 재현성을 깨지 않게 한다.
        result['input_sha256'].pop(str(Path(__file__).resolve()))
        result['input_sha256'][str(out/Path(__file__).name)]=sha(out/Path(__file__).name)
    (out/'feedback-progress.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frame',nargs=2,type=Path,action='append',required=True,metavar=('OWNER','FIELD_ONLY_JSON'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--coupling-contract',type=Path)
    p.add_argument('--coupling-contract-sha256')
    a=p.parse_args();run([(o.resolve(),f.resolve()) for o,f in a.frame],a.out.resolve(),
        coupling_contract=a.coupling_contract.resolve() if a.coupling_contract else None,
        coupling_contract_sha256=a.coupling_contract_sha256)
