#!/usr/bin/env python3
"""원래69/J70에서 공통 반-step 분기70B를 한 번만 만든다.

기존 J70을 읽기 전용으로 재사용하고 RT는 실행하지 않는다. 기본 후속
적격성을 완화하지 않으며 별도 분기 증명 없이는 새 장을 허용하지 않는다.
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
from prepare_cmfgen_fixed_field_owner import point_records, snapshot
from ptf_gated_metrics import (ScrtempReader, parse_solutions, read_dimensions,
                              read_variables, validate_rvtj)
from run_cmfgen_field_feedback import (RESTART_OVERLAY, child, clone_material,
    total_bytes, verify_material)
from run_cmfgen_lambda_probe import sha
from run_cmfgen_native_se_observation import assess, check_hashes, write_json

PURPOSE = 'CMFGEN_DAMPING_BRANCH_2026-10-20'


def load_plan(path, expected):
    if sha(path) != expected:
        raise ValueError('분기 계획 신원 불일치')
    plan = json.loads(path.read_text())
    if (plan['purpose_id'] != PURPOSE or plan['maximum_native_executions'] != 2 or
        plan['maximum_RT_executions'] != 1 or plan['automatic_extension'] is not False or
        plan['omega_sequence'] != [0.5, 1.0] or plan['T_release'] is not False or
        plan['first_parent_seconds'] != 1200 or plan['first_child_seconds'] != 900 or
        plan['second_parent_seconds'] != 2700 or plan['second_child_seconds'] != 2400):
        raise ValueError('등록된 단일 감쇠/원래 step 복원 계획이 아님')
    check_hashes(plan['sealed_inputs'])
    return plan


def build_identity(build, native):
    manifest = build / 'local-damping-build.json'
    model = json.loads(manifest.read_text())
    if Path(model['native']) != native or model['supported_omega'] != [1.0, 0.5]:
        raise ValueError('다른 감쇠 build')
    check_hashes({**model['input_sha256'], **model['output_sha256']})
    executable = build / 'cmfgen_local_damping.exe'
    return executable, dict(path=str(executable), sha256=sha(executable),
                            manifest_sha256=sha(manifest))


def midpoint_evidence(before, actual, reference, raw, reference_raw, scale_rows):
    if not (before.shape == actual.shape == reference.shape == raw.shape == reference_raw.shape):
        raise ValueError('반-step 증거 차원 불일치')
    expected = (before + reference) * 0.5
    bound = 16 * np.finfo(float).eps * (abs(before) + abs(reference))
    scales = np.asarray(scale_rows, dtype=float)
    correct_scales = (scales.shape == (before.shape[1], 4) and
        np.array_equal(scales[:, 0], np.arange(1, before.shape[1]+1)) and
        np.all(scales[:, 1] == 1.0) and np.all(scales[:, 2:] == 0.5))
    report = dict(raw_correction_array_equal=bool(np.array_equal(raw, reference_raw)),
        reference_full_step_reproduced=applied_full_step(before, reference, reference_raw),
        expected_convex_midpoint_matches=bool(np.isfinite(actual).all() and
            np.all(abs(actual - expected) <= bound)),
        raw_local_scale_omega_applied_scale_verified=bool(correct_scales),
        midpoint_rounding_bound='16*eps*(abs(parent)+abs(original_full_step))',
        raw_correction_precision='기존 STEQ_VALS ES12.4; 허용 폭은 기존 full-step 규약 그대로',
        default_next_outer_eligibility_overridden=False,
        physical_SE_convergence='NOT_JUDGED')
    report['verified'] = all(report[k] for k in ('raw_correction_array_equal',
        'reference_full_step_reproduced', 'expected_convex_midpoint_matches',
        'raw_local_scale_omega_applied_scale_verified'))
    return report


def preflight(plan):
    original, source, reference = (Path(plan[k]) for k in
        ('original', 'source69', 'reference70'))
    native, build = (Path(plan[k]) for k in ('native', 'damping_build'))
    state, identity, parent = verify_material(source, original, 69)
    if identity['record_storage_sha256'] != plan['source69_record_sha256']:
        raise ValueError('봉인69와 다른 입력')
    executable, executable_identity = build_identity(build, native)
    old_root = reference.parent
    old = json.loads((old_root/'seq_logs/feedback-inputs.json').read_text())
    runtime = json.loads((old_root/'seq_logs/feedback-runtime.json').read_text())
    if (Path(old['source']) != source or not runtime.get('field_complete') or
        not runtime.get('material_complete') or runtime.get('failure') or
        not runtime.get('source_unchanged') or runtime['material_record'] != 70 or
        runtime['radiation_produced_from_record'] != 69 or
        runtime.get('next_outer_eligible') is not True):
        raise ValueError('원래69/J70→70 대조군이 아님')
    field = old_root/'field'
    check_hashes({str(field/n):h for n,h in runtime['field_evidence_sha256'].items()})
    check_hashes({str(reference/n):h for n,h in runtime['material_evidence_sha256'].items()})
    if sha(field/'SCRTEMP') != sha(source/'SCRTEMP'):
        raise ValueError('J70의 생산 물질이 원래69와 다름')
    full, _, _ = snapshot(reference, original, 70)
    nd, nt = read_dimensions(reference)
    raw = parse_solutions(reference/'STEQ_VALS', nt, nd)
    if not raw or not applied_full_step(state, full, raw[-1]):
        raise ValueError('원래70의 full-step 실증 불충분')
    if not np.array_equal(full[-1], state[-1]):
        raise ValueError('고정T 입력이 아님')
    return state, identity, parent, executable, executable_identity, full, raw[-1], field


def run(plan_path, expected_sha, root):
    if os.environ.get('OMP_NUM_THREADS') != '16' or not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Slurm CPU16/OMP16 내부에서만 실행')
    if root.exists():
        raise ValueError('기존 분기 출력 보존')
    if (os.environ.get('SLURM_CPUS_PER_TASK') != '16' or
        os.environ.get('SLURM_MEM_PER_NODE') != '32768' or
        root.parent != Path('/gpfs/kjhan/cmfgen_runs') or
        root.name != 'toy06_19p48d_damping70B_' + os.environ['SLURM_JOB_ID']):
        raise ValueError('분기 실행 자원/출력 경로 불일치')
    started = time.monotonic()
    deadline = started + 900
    plan = load_plan(plan_path, expected_sha)
    state, identity, parent, executable, exe_id, full, raw, field = preflight(plan)
    original, source, native = (Path(plan[k]) for k in ('original', 'source69', 'native'))
    hashes = {str(source/n): sha(source/n) for n in RESTART_OVERLAY}
    hashes.update(plan['sealed_inputs'])
    hashes[str(plan_path)] = expected_sha
    hashes[str(executable)] = exe_id['sha256']
    for n in ('FORMAL_PRIMARY_FIELDS.bin', 'EDDFACTOR', 'EDDFACTOR_INFO'):
        hashes[str(field/n)] = sha(field/n)
    claim_root = Path(plan['claims_root'])
    claim_root.mkdir(exist_ok=True)
    write_json(claim_root/'70B.json', dict(purpose_id=PURPOSE, branch='70B', root=str(root),
        plan_sha256=expected_sha, native_executions=1, RT_executions=0, retry=0))
    root.mkdir()
    (root/'seq_logs').mkdir()
    # 이 링크는 완주 J70의 읽기 전용 참조다. 수송을 새로 실행했다고 기록하지 않는다.
    (root/'field').symlink_to(field, target_is_directory=True)
    inputs = dict(scope=__doc__, source=str(source), original=str(original), source_state=identity,
        source_sha256=hashes, branch='70B', branch_plan_sha256=expected_sha,
        field_reused_from=str(field), actual_new_RT_executions=0,
        material_executable=exe_id, omega=0.5, final_goal='NOT_JUDGED')
    write_json(root/'seq_logs/feedback-inputs.json', inputs)
    resource.setrlimit(resource.RLIMIT_FSIZE, (8*1024**3, 8*1024**3))
    report = dict(branch='70B', branch_plan_sha256=expected_sha,
        field_complete=True, field_reused=True, material_complete=False,
        RT_executions=0, native_executions=0, next_outer_eligible=False,
        branch_next_permitted=False, joint_convergence=False, released_T=False,
        field_evidence_sha256={n:sha(field/n) for n in (
            'FORMAL_PRIMARY_FIELDS.bin', 'FORMAL_PRIMARY_SWEEP.csv',
            'FORMAL_PRIMARY_COMPLETE.txt', 'EDDFACTOR', 'EDDFACTOR_INFO')},
        final_goal='NOT_JUDGED')
    material = root/'material'
    status = 1
    try:
        clone_material(original, source, material, native, True, 69, '1.0D6')
        for name in ('EDDFACTOR', 'EDDFACTOR_INFO'):
            shutil.copy2(field/name, material/name)
        fixed = {n:sha(material/n) for n in ('EDDFACTOR', 'EDDFACTOR_INFO')}
        write_json(material/'seq_logs/material-execution.json', dict(
            branch='70B', branch_plan_sha256=expected_sha, input_record=69,
            radiation_record=70, radiation_material_record=69, omega=0.5,
            actual_executable=exe_id, original_native_binary=parent['binary'],
            observer_enabled=True, extra_observation_native=False))
        env = dict(os.environ, LUMINA_FORMAL_PRIMARY='', LUMINA_FORMAL_PRIMARY_WHOLE='',
            LUMINA_FORMAL_CELL_BUDGET='', LUMINA_WHOLE_MODE='', LUMINA_MOM_CAPTURE_MODE='',
            LUMINA_FORMAL_CAPTURE_MODE='', LUMINA_NATIVE_SE_WITNESS='1',
            LUMINA_LOCAL_STEP_OMEGA='0.5')
        report['native_executions'] = 1
        report['material'] = child(root, material, executable, env, deadline)
        if report['material']['returncode'] != 0 or report['material']['stop_reason']:
            raise ValueError('분기70B native 실행 실패')
        nd, nt = read_dimensions(material)
        reader = ScrtempReader(material/'SCRTEMP', nd, nt)
        if (reader.last_it != 70 or point_records(material) != dict(POINT1=70,POINT2=70) or
            not validate_rvtj(reader, material/'RVTJ')['complete']):
            raise ValueError('분기70B 완료 record/RVTJ 불일치')
        actual = reader.state(70)
        if (not np.isfinite(actual).all() or np.any(actual[:-2] < 0) or
            np.any(actual[-2:] <= 0) or np.any(actual[state > 0] <= 0) or
            not np.array_equal(actual[-1], state[-1])):
            raise ValueError('분기70B 양수성/고정T 위반')
        if any(sha(material/n) != h for n,h in fixed.items()):
            raise ValueError('분기70B에서 고정장 변경')
        variables = read_variables(material, nt-2)
        charge = sum(v.charge*actual[v.index-1] for v in variables)
        report['max_charge_residual'] = float(np.max(abs(charge-actual[-2])/actual[-2]))
        totals = {}
        for species in sorted({v.species for v in variables}):
            ix = [v.index-1 for v in variables if v.species == species]
            totals[species] = float(np.max(abs(actual[ix].sum(axis=0)/state[ix].sum(axis=0)-1)))
        report['element_total_change_over_parent'] = totals
        if report['max_charge_residual'] > 1e-10 or max(totals.values()) > 1e-10:
            raise ValueError('분기70B 전하/원소합 보존 위반')
        report.update(material_complete=True, material_record=70,
            radiation_produced_from_record=69,
            max_ne_relative_step=float(np.max(abs(actual[-2]/state[-2]-1))),
            material_evidence_sha256={n:sha(material/n) for n in
                ('SCRTEMP','POINT1','POINT2','RVTJ')},
            invalid_material_outputs=['dummy H/K 기반 flux/heat/OBSFLUX'])
        solutions = parse_solutions(material/'STEQ_VALS', nt, nd)
        if not solutions:
            raise ValueError('분기70B raw 보정 없음')
        rows = re.findall(r'LUMINA_LOCAL_DAMPING_V1\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)',
                          (material/'OUTGEN').read_text())
        proof = midpoint_evidence(state, actual, full, solutions[-1], raw, rows)
        reference = Path(plan['reference70'])
        proof['pre_update_witness_byte_equal'] = (
            sha(material/'NATIVE_SE_BOUND_000070.bin') == sha(reference/'NATIVE_SE_BOUND_000070.bin'))
        proof['verified'] = proof['verified'] and proof['pre_update_witness_byte_equal']
        report['damping_evidence'] = proof
        report['actual_full_step_within_print_precision'] = applied_full_step(state,actual,solutions[-1])
        report['native_SE_assessment'] = assess(material,state,69,radiation_produced_from_record=69)
        if not proof['verified']:
            raise ValueError('반-step 예상 상태/raw 보정/갱신 전 SE 재현 실패')
        report['branch_next_permitted'] = True
        status = 0
    except (ValueError, OSError) as error:
        report['failure'] = str(error)
    finally:
        report.update(elapsed_seconds=time.monotonic()-started, root_bytes=total_bytes(root),
            source_unchanged=all(sha(Path(p)) == h for p,h in hashes.items()))
        if not report['source_unchanged'] or report['root_bytes'] > 16*1024**3:
            report['branch_next_permitted'] = False
            report['failure'] = '원본 보존 또는 root16GiB 한도 위반'
            status = 1
        write_json(root/'seq_logs/feedback-runtime.json', report)
    print(json.dumps(report,ensure_ascii=False,allow_nan=False))
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--run',type=Path,required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.plan.resolve(),args.plan_sha256,args.run.resolve()))
