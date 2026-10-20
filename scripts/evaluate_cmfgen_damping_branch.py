#!/usr/bin/env python3
"""공통 반-step 뒤 원래 step을 복원한 응답을 사전 대조군71과 비교한다.

기존 완료 frame 판독 함수를 재사용하되 별도 분기 계획과 출처를 확인한다.
반-step 적용량 감소나 수치 조건 충족을 물리 동질성·실행 연장으로 바꾸지 않는다.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from cmfgen_step_precision import applied_full_step
from compare_cmfgen_feedback_candidates import load_report
from evaluate_cmfgen_feedback_66_67 import load_frame
from ptf_gated_metrics import parse_solutions
from run_cmfgen_damping70_branch import load_plan
from run_cmfgen_lambda_probe import sha

# 분율/상대변화 출력의 binary64 반올림 여유다. 물리 허용치를 바꾸지 않는다.
OUTPUT_ROUNDOFF = 64 * np.finfo(float).eps


def vector(frame, key):
    values = np.asarray([v for Z in ('14','16') for v in frame['elements'][Z][key]])
    if values.shape != (4,) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('양원소 전단계 p95/최대 지표 오류')
    return values


def decide(control, candidate, motion, threshold):
    if not np.isfinite([motion,threshold]).all() or motion < 0 or threshold <= 0:
        raise ValueError('같은 셀의 실제 이동 진폭 오류')
    old = np.r_[vector(control,'material'), control['ne_step']]
    new = np.r_[vector(candidate,'material'), candidate['ne_step']]
    if not np.isfinite(np.r_[old,new]).all() or np.any(np.r_[old,new] < 0):
        raise ValueError('전자밀도/물질 이동 오류')
    before,after = vector(control,'conditional'),vector(candidate,'conditional')
    # F도 실제로 있어야 한다. 물질 이동 P의 감소로 F를 대체하지 않는다.
    old_field,new_field = vector(control,'field'),vector(candidate,'field')
    nonworse = bool(np.all(new <= old))
    improved = bool(np.any(old-new > OUTPUT_ROUNDOFF*(1+abs(old)+abs(new))))
    generally_worse_Q = bool(np.all(after >= before) and np.any(after > before))
    local_improved = bool(motion < threshold)
    observed = nonworse and improved and local_improved and not generally_worse_Q
    mixed = bool(np.any(new < old) and np.any(new > old))
    Q_mixed = bool(np.any(after < before) and np.any(after > before))
    F_mixed = bool(np.any(new_field < old_field) and np.any(new_field > old_field))
    decision = ('MIXED_REQUIRES_EVALUATION' if Q_mixed or F_mixed else
                'EVALUATE_STABILIZATION_CANDIDATE')
    return dict(decision=decision if observed else
        ('MIXED_NOT_ADOPTED' if mixed else 'CRITERIA_NOT_MET'),
        numerical_adoption_conditions_observed=observed,
        local_SiV_depth10_step_below_preregistered=local_improved,
        final_P_and_ne_all_no_worse=nonworse,
        at_least_one_improvement_beyond_output_roundoff=improved,
        generally_worse_Q=generally_worse_Q,
        Q_mixed=Q_mixed,F_mixed=F_mixed,
        control_P_ne=old.tolist(),candidate_P_ne=new.tolist(),
        P_ne_component_order=['Si_P_p95','Si_P_max','S_P_p95','S_P_max','ne_step'],
        control_F=old_field.tolist(),candidate_F=new_field.tolist(),
        control_Q=before.tolist(),candidate_Q=after.tolist(),
        SiV_depth10_absolute_step=motion,preregistered_motion_threshold=threshold,
        output_roundoff_rule='64*binary64_eps*(1+abs(control)+abs(candidate)); 출력 비교 전용',
        F_and_raw_correction_still_require_physical_review=True,
        automatically_adopted=False,automatic_extension=False,
        joint_convergence=False,released_T=False,
        physical_equivalence='NOT_JUDGED',final_goal='NOT_JUDGED')


def motion_at_depth10(owner, frame):
    data = np.load(owner/'completed-state.npz')
    columns,indexes = data['columns'],data['owner_variable_indexes']
    element = indexes[columns[:,0] == 14]
    stage = indexes[(columns[:,0] == 14) & (columns[:,1] == 4)]
    parent = frame['_parent']
    previous = parent[stage].sum(axis=0)/parent[element].sum(axis=0)
    return float(abs(frame['_fraction'][3,9]-previous[9]))


def raw_step(owner, frame):
    manifest = json.loads((owner/'fixed-field-owner-inputs.json').read_text())
    run = Path(manifest['run'])
    path = run/'STEQ_VALS'
    digest = sha(path)
    nt,nd = frame['_state'].shape
    solutions = parse_solutions(path,nt,nd)
    if not solutions or not np.isfinite(solutions[-1]).all() or sha(path) != digest:
        raise ValueError('실제 raw 보정 누락/변경')
    correction = solutions[-1]
    if not applied_full_step(frame['_parent'],frame['_state'],correction):
        raise ValueError('원래 full-step 적용을 실제 상태에서 재현할 수 없음')
    return dict(all_population_rows_abs_p95_max=np.quantile(abs(correction[:-2]),[.95,1]).tolist(),
        ne_abs_max=float(abs(correction[-2]).max()),T_abs_max=float(abs(correction[-1]).max()),
        actual_full_step_reproduced=True,precision='기존 STEQ_VALS ES12.4',
        scope='전체 인구행 진단이며 full heat/opacity 활성 집합이나 SE 잔차가 아님'),path,digest


def run(control_owner,control_field,branch_owner,branch_field,plan_path,plan_sha,out):
    if out.exists():
        raise ValueError('기존 분기 결과 판독 보존')
    helpers = [Path(fn.__code__.co_filename).resolve() for fn in
        (load_frame,load_report,applied_full_step,parse_solutions,load_plan,sha)]
    producer = Path(__file__).resolve()
    before = {str(p):sha(p) for p in [producer,*helpers]}
    plan = load_plan(plan_path,plan_sha)
    cache = {}
    control = load_frame(control_owner,control_field,cache)
    branch = load_frame(branch_owner,branch_field,cache)
    manifests = [json.loads((p/'fixed-field-owner-inputs.json').read_text())
                 for p in (control_owner,branch_owner)]
    cm,bm = manifests
    if (Path(cm['run']) != Path(plan['reference71']) or control['record'] != 71 or
        branch['record'] != 71 or control['_binary'] != branch['_binary'] or
        branch['_binary'] != plan['sealed_inputs'][plan['C_binary']] or
        not np.array_equal(control['_geometry'],branch['_geometry'])):
        raise ValueError('다른 대조군71/C1 생산자/질량 격자')
    control_runtime = Path(control['runtime'])
    if sha(control_runtime) != plan['sealed_inputs'][str(control_runtime)]:
        raise ValueError('봉인 대조군71 runtime 변경')
    branch_root = Path(bm['feedback_root'])
    branch_runtime = json.loads(Path(branch['runtime']).read_text())
    inputs_path = branch_root/'seq_logs/feedback-inputs.json'
    inputs = json.loads(inputs_path.read_text())
    claim_root = Path(plan['claims_root'])
    first_claim = json.loads((claim_root/'70B.json').read_text())
    last_claim = json.loads((claim_root/'71B.json').read_text())
    first_root = Path(first_claim['root'])
    first_runtime_path = first_root/'seq_logs/feedback-runtime.json'
    first_runtime = json.loads(first_runtime_path.read_text())
    if (first_claim['plan_sha256'] != plan_sha or last_claim['plan_sha256'] != plan_sha or
        Path(last_claim['root']) != branch_root or Path(last_claim['source_root']) != first_root or
        branch_runtime.get('branch') != '71B' or branch_runtime.get('failure') or
        branch_runtime.get('branch_plan_sha256') != plan_sha or
        branch_runtime.get('RT_executions') != 1 or branch_runtime.get('native_executions') != 1 or
        branch_runtime.get('printed_minimum_scale') != 1.0 or
        branch_runtime.get('actual_full_step_within_print_precision') is not True or
        branch_runtime.get('next_outer_eligible') is not False or
        first_runtime.get('branch_next_permitted') is not True or first_runtime.get('failure') or
        first_runtime.get('branch_plan_sha256') != plan_sha or
        not first_runtime.get('damping_evidence',{}).get('verified') or
        inputs.get('branch_plan_sha256') != plan_sha or inputs.get('omega') != 1.0 or
        Path(inputs['source']) != first_root/'material'):
        raise ValueError('사전 유한 분기/원래 step 복원 계보 불일치')
    if sha(first_runtime_path) != inputs['source_sha256'][str(first_runtime_path)]:
        raise ValueError('첫 분기 runtime 변경')
    first_owner = Path(inputs['source_C1'])
    first_report,first_binary,first_geometry,_ = load_report(first_owner)
    if (first_binary != branch['_binary'] or first_report['record'] != 70 or
        not np.array_equal(np.load(first_owner/'completed-state.npz')['state'],branch['_parent'])):
        raise ValueError('복원 갱신의 부모가 실제70B와 다름')
    witness_manifest = Path(plan['restored_witness_build'])/'se-witness-build.json'
    witness = json.loads(witness_manifest.read_text())
    original_executable = str(Path(plan['restored_witness_build'])/'cmfgen_se_witness.exe')
    actual_executable = inputs['actual_executable']
    if (actual_executable['path'] != original_executable or
        actual_executable['sha256'] != witness['output_sha256'][original_executable]):
        raise ValueError('원래 검증 witness가 아닌 복원 실행파일')
    raw_control,control_path,control_sha = raw_step(control_owner,control)
    raw_branch,branch_path,branch_sha = raw_step(branch_owner,branch)
    threshold = plan['adoption_criteria']['restored_SiV_depth10_absolute_step_below_fraction']
    control_motion = motion_at_depth10(control_owner,control)
    if abs(control_motion-threshold) > OUTPUT_ROUNDOFF:
        raise ValueError('사전 같은 셀 기준과 실제 대조군 불일치')
    result = decide(control,branch,motion_at_depth10(branch_owner,branch),threshold)
    result.update(scope=__doc__,purpose=plan['purpose_id'],plan_sha256=plan_sha,
        control_motion=control_motion,raw_control=raw_control,raw_candidate=raw_branch,
        control={k:v for k,v in control.items() if not k.startswith('_')},
        candidate={k:v for k,v in branch.items() if not k.startswith('_')},
        default_next_outer_eligible=False,default_eligibility_policy_changed=False,
        restored_full_step_reproduced=True)
    evidence = [plan_path,control_owner/'comparison/fixed-field-owner.json',control_field,
        branch_owner/'comparison/fixed-field-owner.json',branch_field,control_runtime,
        Path(branch['runtime']),first_runtime_path,inputs_path,claim_root/'70B.json',claim_root/'71B.json',
        first_owner/'comparison/fixed-field-owner.json',witness_manifest]
    result['input_sha256'] = {str(p):sha(p) for p in evidence}
    result['input_sha256'].update({str(control_path):control_sha,str(branch_path):branch_sha})
    if any(sha(Path(p)) != digest for p,digest in before.items()):
        raise ValueError('판독 중 생산자/지원 코드 변경')
    result['input_sha256'].update({str(p):before[str(p)] for p in helpers})
    out.mkdir()
    shutil.copy2(producer,out/producer.name)
    result['input_sha256'][str(out/producer.name)] = sha(out/producer.name)
    (out/'damping-comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('control-owner','control-field','branch-owner','branch-field','plan','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--plan-sha256',required=True)
    args = parser.parse_args()
    run(args.control_owner.resolve(),args.control_field.resolve(),args.branch_owner.resolve(),
        args.branch_field.resolve(),args.plan.resolve(),args.plan_sha256,args.out.resolve())
