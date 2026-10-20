#!/usr/bin/env python3
"""증명된70B의 새 장과 원래 native step으로 마지막71B를 만든다.

공통 반-step의 수치 증명을 실제 상태에서 재현하고 전90점 C1 입력과 묶는다.
기존 기본 full-step 적격성이나 원래70/71의 이력은 변경하지 않는다.
"""
import argparse
import json
import os
from pathlib import Path
import re
import resource
import shutil
import time

import numpy as np

from cmfgen_step_precision import applied_full_step
from compare_cmfgen_feedback_candidates import load_report, element_metrics
from prepare_cmfgen_fixed_field_owner import point_records, snapshot
from ptf_gated_metrics import ScrtempReader, parse_solutions, read_dimensions, read_variables, validate_rvtj
from run_cmfgen_damping70_branch import load_plan, midpoint_evidence, preflight
from run_cmfgen_field_feedback import RESTART_OVERLAY, child, clone_material, total_bytes
from run_cmfgen_formal_primary import coverage, fields, validate_edd
from run_cmfgen_lambda_probe import sha
from run_cmfgen_native_se_observation import assess, check_hashes, write_json


def verify_source(plan, root, owner, runtime_sha):
    runtime_path = root/'seq_logs/feedback-runtime.json'
    if sha(runtime_path) != runtime_sha:
        raise ValueError('완료70B runtime 신원 불일치')
    runtime = json.loads(runtime_path.read_text())
    inputs = json.loads((root/'seq_logs/feedback-inputs.json').read_text())
    claim = json.loads((Path(plan['claims_root'])/'70B.json').read_text())
    if (Path(claim['root']) != root or claim['branch'] != '70B' or
        claim['plan_sha256'] != runtime['branch_plan_sha256'] or
        inputs['branch_plan_sha256'] != runtime['branch_plan_sha256'] or
        not runtime.get('material_complete') or not runtime.get('field_complete') or
        runtime.get('failure') or not runtime.get('source_unchanged') or
        not runtime.get('branch_next_permitted') or runtime.get('next_outer_eligible') is not False or
        runtime['branch'] != '70B' or runtime['RT_executions'] != 0 or
        runtime['native_executions'] != 1 or runtime['material_record'] != 70 or
        runtime['material']['returncode'] != 0 or runtime['material'].get('stop_reason') or
        inputs['omega'] != 0.5 or inputs['source'] != plan['source69']):
        raise ValueError('정상 완료된70B 분기가 아님')
    state, _, parent, _, _, full, raw, field = preflight(plan)
    source = root/'material'
    check_hashes({str(source/n):h for n,h in runtime['material_evidence_sha256'].items()})
    check_hashes({str(root/'field'/n):h for n,h in runtime['field_evidence_sha256'].items()})
    if (root/'field').resolve() != field.resolve():
        raise ValueError('70B의 원래 J70 참조가 다름')
    actual, _, identity = snapshot(source,Path(plan['original']),70)
    nd, nt = read_dimensions(source)
    solutions = parse_solutions(source/'STEQ_VALS',nt,nd)
    rows = re.findall(r'LUMINA_LOCAL_DAMPING_V1\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)',
                      (source/'OUTGEN').read_text())
    if (not solutions or not midpoint_evidence(state,actual,full,solutions[-1],raw,rows)['verified'] or
        sha(source/'NATIVE_SE_BOUND_000070.bin') !=
        sha(Path(plan['reference70'])/'NATIVE_SE_BOUND_000070.bin')):
        raise ValueError('70B의 반-step 증거를 재현할 수 없음')
    report,binary,geometry,comparison = load_report(owner)
    manifest = json.loads((owner/'fixed-field-owner-inputs.json').read_text())
    if (Path(manifest['run']) != source or report['record'] != 70 or
        manifest['material_snapshot']['record_storage_sha256'] != identity['record_storage_sha256'] or
        binary != plan['sealed_inputs'][plan['C_binary']] or
        not np.array_equal(np.load(owner/'completed-state.npz')['state'],actual) or
        any(report['per_element'][str(Z)]['certified_and_committed'] != 90 or
            element_metrics(report,Z)['support_mismatch_count'] != 0 for Z in (14,16))):
        raise ValueError('70B의 실제 C1 비교와 입력이 다름')
    return source, actual, identity, parent, comparison


def run(plan_path, expected_sha, source_root, source_owner, source_runtime_sha, root):
    if os.environ.get('OMP_NUM_THREADS') != '16' or not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Slurm CPU16/OMP16 내부에서만 실행')
    if root.exists():
        raise ValueError('기존71B 출력 보존')
    if (os.environ.get('SLURM_CPUS_PER_TASK') != '16' or
        os.environ.get('SLURM_MEM_PER_NODE') != '32768' or
        root.parent != Path('/gpfs/kjhan/cmfgen_runs') or
        root.name != 'toy06_19p48d_damping71B_' + os.environ['SLURM_JOB_ID']):
        raise ValueError('71B 자원/출력 경로 불일치')
    started = time.monotonic()
    deadline = started+2400
    plan = load_plan(plan_path,expected_sha)
    source,state,identity,parent,comparison = verify_source(
        plan,source_root,source_owner,source_runtime_sha)
    first_inputs = json.loads((source_root/'seq_logs/feedback-inputs.json').read_text())
    if first_inputs['branch_plan_sha256'] != expected_sha:
        raise ValueError('현재 계획과 다른70B')
    original,native,build,witness = (Path(plan[k]) for k in
        ('original','native','new_field_build','restored_witness_build'))
    field_model = json.loads((build/'feedback-build.json').read_text())
    witness_model = json.loads((witness/'se-witness-build.json').read_text())
    if (field_model['expected_iteration'] != 71 or field_model['pre_population_stop'] != 97 or
        Path(field_model['native']) != native or Path(witness_model['native']) != native):
        raise ValueError('71B 수송/복원 native 정체성 불일치')
    check_hashes({**field_model['input_sha256'],**field_model['output_sha256'],
                  **witness_model['input_sha256'],**witness_model['output_sha256']})
    executable = witness/'cmfgen_se_witness.exe'
    hashes = {str(source/n):sha(source/n) for n in RESTART_OVERLAY}
    hashes.update(plan['sealed_inputs'])
    hashes.update({str(plan_path):expected_sha,str(comparison):sha(comparison),
        str(source_root/'seq_logs/feedback-runtime.json'):source_runtime_sha,
        str(executable):sha(executable)})
    write_json(Path(plan['claims_root'])/'71B.json',dict(branch='71B',root=str(root),
        source_root=str(source_root),plan_sha256=expected_sha,native_executions=1,
        RT_executions=1,retry=0))
    root.mkdir();(root/'seq_logs').mkdir()
    write_json(root/'seq_logs/feedback-inputs.json',dict(scope=__doc__,source=str(source),
        original=str(original),source_state=identity,source_sha256=hashes,
        branch='71B',branch_plan_sha256=expected_sha,source_C1=str(source_owner),
        omega=1.0,actual_executable=dict(path=str(executable),sha256=sha(executable)),
        final_goal='NOT_JUDGED'))
    resource.setrlimit(resource.RLIMIT_FSIZE,(8*1024**3,8*1024**3))
    report = dict(branch='71B',branch_plan_sha256=expected_sha,field_complete=False,
        material_complete=False,RT_executions=0,native_executions=0,
        next_outer_eligible=False,branch_next_permitted=False,automatic_extension=False,
        joint_convergence=False,released_T=False,final_goal='NOT_JUDGED')
    field,material = root/'field',root/'material'
    status = 1
    try:
        clone_material(original,source,field,native,False,70,'1.0D6')
        env = dict(os.environ,LUMINA_FORMAL_PRIMARY='1',LUMINA_FORMAL_PRIMARY_WHOLE='1',
            LUMINA_FORMAL_CELL_BUDGET='',LUMINA_WHOLE_MODE='',LUMINA_MOM_CAPTURE_MODE='',
            LUMINA_FORMAL_CAPTURE_MODE='',LUMINA_NATIVE_SE_WITNESS='',LUMINA_LOCAL_STEP_OMEGA='1')
        report['RT_executions'] = 1
        report['field'] = child(root,field,build/'cmfgen_feedback_field.exe',env,deadline)
        if report['field']['returncode'] != 97 or report['field']['stop_reason']:
            raise ValueError('71B 새 장 실행 실패')
        if (field/'FORMAL_PRIMARY_COMPLETE.txt').read_text().splitlines()[0].split()[:3] != ['71','197291','197291']:
            raise ValueError('71B 새 장 완료 세대/coverage 불일치')
        report['field_coverage'] = coverage(field/'FORMAL_PRIMARY_SWEEP.csv')
        _,array = fields(field/'FORMAL_PRIMARY_FIELDS.bin')
        if report['field_coverage']['count'] != 197291 or array['field'].shape != (197291,8,90):
            raise ValueError('71B 새 장 불완전')
        validate_edd(field,array)
        if sha(field/'SCRTEMP') != hashes[str(source/'SCRTEMP')]:
            raise ValueError('71B 장 계산에서 부모 물질 변경')
        report['field_complete'] = True
        report['field_evidence_sha256'] = {n:sha(field/n) for n in (
            'FORMAL_PRIMARY_FIELDS.bin','FORMAL_PRIMARY_SWEEP.csv','FORMAL_PRIMARY_COMPLETE.txt',
            'EDDFACTOR','EDDFACTOR_INFO')}
        if deadline-time.monotonic() < 540:
            raise ValueError('71B 물질 갱신의 남은 예산9분 미만')
        clone_material(original,source,material,native,True,70,'1.0D6')
        for name in ('EDDFACTOR','EDDFACTOR_INFO'):
            shutil.copy2(field/name,material/name)
        fixed = {n:sha(material/n) for n in ('EDDFACTOR','EDDFACTOR_INFO')}
        report['native_executions'] = 1
        material_env = dict(env,LUMINA_FORMAL_PRIMARY='',LUMINA_FORMAL_PRIMARY_WHOLE='',
                            LUMINA_NATIVE_SE_WITNESS='1')
        write_json(material/'seq_logs/material-execution.json',dict(branch='71B',omega=1.0,
            branch_plan_sha256=expected_sha,input_record=70,radiation_record=71,
            radiation_material_record=70,source_branch='70B',actual_executable=str(executable),
            actual_executable_sha256=sha(executable),observer_enabled=True,
            extra_observation_native=False))
        report['material'] = child(root,material,executable,material_env,deadline)
        if report['material']['returncode'] != 0 or report['material']['stop_reason']:
            raise ValueError('71B 원래 step 물질 실행 실패')
        nd,nt = read_dimensions(material)
        reader = ScrtempReader(material/'SCRTEMP',nd,nt)
        if (reader.last_it != 71 or point_records(material) != dict(POINT1=71,POINT2=71) or
            not validate_rvtj(reader,material/'RVTJ')['complete']):
            raise ValueError('71B 완료 record/RVTJ 불일치')
        updated = reader.state(71)
        if (not np.isfinite(updated).all() or np.any(updated[:-2] < 0) or
            np.any(updated[-2:] <= 0) or np.any(updated[state > 0] <= 0) or
            not np.array_equal(updated[-1],state[-1])):
            raise ValueError('71B 양수성/고정T 위반')
        if any(sha(material/n) != h for n,h in fixed.items()):
            raise ValueError('71B 물질 갱신에서 장 변경')
        variables = read_variables(material,nt-2)
        charge = sum(v.charge*updated[v.index-1] for v in variables)
        report['max_charge_residual'] = float(np.max(abs(charge-updated[-2])/updated[-2]))
        totals = {}
        for species in sorted({v.species for v in variables}):
            ix = [v.index-1 for v in variables if v.species == species]
            totals[species] = float(np.max(abs(updated[ix].sum(axis=0)/state[ix].sum(axis=0)-1)))
        report['element_total_change_over_parent'] = totals
        if report['max_charge_residual'] > 1e-10 or max(totals.values()) > 1e-10:
            raise ValueError('71B 전하/원소합 보존 위반')
        solutions = parse_solutions(material/'STEQ_VALS',nt,nd)
        scale = re.findall(r'The local minimum value of scale is:\s*(\S+)',(material/'OUTGEN').read_text())
        report['printed_minimum_scale'] = float(scale[-1]) if scale else None
        report['actual_full_step_within_print_precision'] = bool(solutions and
            applied_full_step(state,updated,solutions[-1]))
        report.update(material_complete=True,material_record=71,radiation_produced_from_record=70,
            max_ne_relative_step=float(np.max(abs(updated[-2]/state[-2]-1))),
            material_evidence_sha256={n:sha(material/n) for n in ('SCRTEMP','POINT1','POINT2','RVTJ')},
            invalid_material_outputs=['dummy H/K 기반 flux/heat/OBSFLUX'])
        report['native_SE_assessment'] = assess(material,state,70,radiation_produced_from_record=70)
        if report['printed_minimum_scale'] != 1.0 or not report['actual_full_step_within_print_precision']:
            raise ValueError('71B 원래 full-step 복원 미입증; 추가 실행 금지')
        status = 0
    except (ValueError,OSError) as error:
        report['failure'] = str(error)
    finally:
        report.update(elapsed_seconds=time.monotonic()-started,root_bytes=total_bytes(root),
            source_unchanged=all(sha(Path(p)) == h for p,h in hashes.items()))
        if not report['source_unchanged'] or report['root_bytes'] > 16*1024**3:
            report['failure'] = '71B 원본 보존 또는 root16GiB 한도 위반'
            status = 1
        write_json(root/'seq_logs/feedback-runtime.json',report)
    print(json.dumps(report,ensure_ascii=False,allow_nan=False))
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan','source-root','source-owner','run'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--source-runtime-sha256',required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.plan.resolve(),args.plan_sha256,args.source_root.resolve(),
        args.source_owner.resolve(),args.source_runtime_sha256,args.run.resolve()))
