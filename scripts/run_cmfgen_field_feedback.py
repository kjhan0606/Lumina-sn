#!/usr/bin/env python3
"""명시한 완료 물질 → 그 물질의 새 전주파수 장 → native 갱신 한 번.

두 자식에 하나의 2400초 예산을 적용한다. 고정T staging이며, 수렴한 권위
상태나 released-T 앵커/스펙트럼으로 표시하지 않는다. 원본은 변경하지 않는다.
"""
import argparse
import json
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import time

import numpy as np

from configure_cmfgen_stageA import set_control
from prepare_cmfgen_fixed_field_owner import point_records, snapshot
from ptf_gated_metrics import ScrtempReader, read_dimensions, read_variables, validate_rvtj, parse_solutions
from cmfgen_step_precision import applied_full_step
from run_cmfgen_fixed_field import output_bytes, scan
from run_cmfgen_formal_primary import coverage, fields, validate_edd
from run_cmfgen_lambda_probe import sha, stage, control, stop_child

RESTART_OVERLAY = ('SCRTEMP', 'POINT1', 'POINT2', 'RVTJ', 'STEQ_VALS',
                   'EDDFACTOR', 'EDDFACTOR_INFO', 'CUR_MODEL_DATA')


def controls(vadat, in_its, fixed_j, max_lam='2.0D0'):
    if max_lam not in ('2.0D0', '1.0D6'):
        raise ValueError('등록되지 않은 증가 상한')
    expected = dict(FIX_T='T', FIX_T_AUTO='F', FIX_NE='F', DO_DDT='F',
                    DO_NG='F', TRT_NON_TE='F', USE_FIXED_J='F', INC_GRID='F',
                    WRITE_RATES='T', WRITE_JH='T')
    for key, value in expected.items():
        if control(vadat, key) != value:
            raise ValueError('원래 입력 제어 불일치: '+key)
    for key in ('DO_LAM_AUTO', 'DO_T_AUTO'):
        if control(in_its, key) != 'F':
            raise ValueError('자동 모드 전환 금지')
    changes = dict(USE_FIXED_J='T' if fixed_j else 'F', LAM_SCALE_OPT='NONE',
                   SCALE_OPT='LOCAL', MAX_LAM=max_lam, WRITE_RATES='F')
    for key, value in changes.items():
        vadat = set_control(vadat, key, value, append=key == 'LAM_SCALE_OPT')
    in_its = set_control(set_control(in_its, 'NUM_ITS', '1'), 'DO_LAM_IT', 'T')
    return vadat, in_its


def verify_feedback_material(source, original, record, selection=None, coupling=None):
    """실제로 종료된 교대 실행의 물질만 다음 새 장의 입력으로 사용한다."""
    root = source.parent
    if source.name != 'material' or record < 64:
        raise ValueError('완료 교대 물질 경로/세대가 아님')
    inputs = json.loads((root/'seq_logs/feedback-inputs.json').read_text())
    runtime = json.loads((root/'seq_logs/feedback-runtime.json').read_text())
    m = json.loads((source/'seq_logs/feedback-stage.json').read_text())
    if (Path(inputs['original']) != original or not runtime.get('material_complete') or
        not runtime.get('field_complete') or not runtime.get('source_unchanged') or
        runtime.get('failure') or runtime['material']['returncode'] != 0 or
        runtime['material'].get('stop_reason') or runtime['material_record'] != record or
        runtime['radiation_produced_from_record'] != record-1 or
        m['input_state_record'] != record-1 or m['next_native_iteration'] != record or
        not m['USE_FIXED_J'] or point_records(source) != dict(POINT1=record, POINT2=record)):
        raise ValueError('정상 완료 교대 물질이 아님')
    if coupling is not None:
        from cmfgen_coupling_contract import authorize_source
        authorize_source(source, record, inputs, runtime, coupling, selection)
    elif record>=66 and runtime.get('next_outer_eligible') is not True:
        raise ValueError('전체 갱신 적용 전제가 확인되지 않은 후속 물질')
    if record == 64:
        if selection is None:
            raise ValueError('64A/B의 기존 출발점 선택 근거 필요')
        chosen = json.loads(selection.read_text())
        if chosen['decision'] != 'NEXT_OUTER_SUPPORTED' or chosen['selected_case'] != '64B':
            raise ValueError('64B 후속 새 장이 선택되지 않음')
        for name, expected in chosen['input_sha256'].items():
            if sha(Path(name)) != expected:
                raise ValueError('출발점 선택 근거 변경')
        selected_manifest = json.loads((Path(chosen['B_root'])/'fixed-field-owner-inputs.json').read_text())
        if Path(selected_manifest['run']) != source or runtime.get('ab_case') != '64B':
            raise ValueError('선택한64B와 다른 물질')
    for name, expected in runtime['material_evidence_sha256'].items():
        if sha(source/name) != expected:
            raise ValueError('완료 교대 물질 변경: '+name)
    for name, expected in runtime['field_evidence_sha256'].items():
        if sha(root/'field'/name) != expected:
            raise ValueError('완료 교대 장 변경: '+name)
    for name, expected in m['controls_sha256'].items():
        if sha(source/name) != expected:
            raise ValueError('교대 물질 제어 변경')
    expected_v, expected_i = controls((original/'VADAT').read_text(), (original/'IN_ITS').read_text(), True, '1.0D6')
    if (source/'VADAT').read_text() != expected_v or (source/'IN_ITS').read_text() != expected_i:
        raise ValueError('선택된 고정T 교대 제어와 다름')
    for name, item in m['linked_inputs'].items():
        if sha(source/name) != item['sha256'] or sha(Path(item['target'])) != item['sha256']:
            raise ValueError('완료 교대 원자료 변경: '+name)
    state, _, identity = snapshot(source, original, record)
    if (coupling is not None and record == coupling['initial_record'] and
        identity['record_storage_sha256'] != coupling['initial_record_storage_sha256']):
        raise ValueError(f'계약의 원래{record} 완료 record 변경')
    nd, nt = read_dimensions(source)
    reader = ScrtempReader(source/'SCRTEMP', nd, nt)
    if reader.last_it != record or not validate_rvtj(reader, source/'RVTJ')['complete']:
        raise ValueError('교대 물질/RVTJ 연결 불완전')
    for name in ('BAMAT', 'BAMATPNT', 'CONT_FREQ', 'JEW',
                 'MODEL_SPEC', 'SN_HYDRO_DATA', 'LEVEL_SL_STEQ_LINKS'):
        if sha(source/name) != sha(original/name):
            raise ValueError('예상 밖 restart 변화: '+name)
    return state, identity, dict(binary=m['native_binary'], linked_inputs=m['linked_inputs'])


def verify_material(source, original, record=63, selection=None, coupling=None):
    if record != 63:
        return verify_feedback_material(source, original, record, selection, coupling)
    if coupling is not None:
        raise ValueError('부분-step 계약의 등록된 초기/후속 세대만 허용')
    m = json.loads((source/'seq_logs/fixed-field-inputs.json').read_text())
    r = json.loads((source/'seq_logs/fixed-field-runtime.json').read_text())
    if (Path(m['seed']) != original or r['returncode'] != 0 or
        not all(r[k] for k in ('four_native_updates_complete', 'fixed_J_unchanged',
                              'seed_unchanged', 'T_unchanged')) or
        r['last_completed_record'] != 63 or point_records(source) != dict(POINT1=63, POINT2=63)):
        raise ValueError('정상 완료 물질63이 아님')
    if set_control((source/'VADAT').read_text(), 'USE_FIXED_J', 'F') != (original/'VADAT').read_text():
        raise ValueError('원래 물리 제어와 다른 물질63')
    for name, expected in m['input_controls_sha256'].items():
        if sha(source/name) != expected:
            raise ValueError('완료 run 제어 변경')
    for name, expected in m['fixed_field_inputs'].items():
        if sha(source/name) != expected:
            raise ValueError('물질63의 고정장 변경')
    state, previous, identity = snapshot(source, original, 63)
    nd, nt = read_dimensions(source)
    if not validate_rvtj(ScrtempReader(source/'SCRTEMP', nd, nt), source/'RVTJ')['complete']:
        raise ValueError('물질63/RVTJ 불일치')
    # overlay하지 않는 restart/물리 입력은 원래59와 같아야 한다.
    for name in ('BAMAT', 'BAMATPNT', 'CONT_FREQ', 'JEW',
                 'MODEL_SPEC', 'SN_HYDRO_DATA', 'LEVEL_SL_STEQ_LINKS'):
        if sha(source/name) != sha(original/name):
            raise ValueError('예상 밖 restart 변화: '+name)
    return state, identity, m


def clone_material(original, source, target, native, fixed_j, source_record=63, max_lam='2.0D0'):
    # 59 전용 기존 helper의 검사를 약화하지 않는다. 검증한 원래 입력을 복제한
    # 뒤 실제 native 완료 상태의 restart만 덮어 새 세대의 계보를 기록한다.
    inherited = stage(original, target, native/'exe/cmfgen_dev.exe', native, write_record=False)
    for name in RESTART_OVERLAY:
        shutil.copy2(source/name, target/name)
    v, i = controls((original/'VADAT').read_text(), (original/'IN_ITS').read_text(), fixed_j, max_lam)
    (target/'VADAT').write_text(v)
    (target/'IN_ITS').write_text(i)
    if point_records(target) != dict(POINT1=source_record, POINT2=source_record):
        raise ValueError('복제본이 명시한 완료 물질을 가리키지 않음')
    record = dict(material_source=str(source), original_physical_seed=str(original),
        input_state_record=source_record, next_native_iteration=source_record+1,
        restart_sha256={n: sha(target/n) for n in RESTART_OVERLAY},
        controls_sha256={n: sha(target/n) for n in ('VADAT', 'IN_ITS')},
        native_binary=inherited['binary'], linked_inputs=inherited['linked_inputs'],
        numerical_controls=dict(LAM_SCALE_OPT='NONE', SCALE_OPT='LOCAL', MAX_LAM=float(max_lam.replace('D','E'))),
        WRITE_RATES='F: 중복 per-line/EW 대량 출력만 생략; 원래 SE 조립 유지',
        USE_FIXED_J=fixed_j, FIX_T=True, full_coupled_Jacobian=False, final_goal='NOT_JUDGED')
    (target/'seq_logs/feedback-stage.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    return record


def total_bytes(root):
    total = sum(p.stat().st_size for p in (root/'seq_logs').iterdir() if p.is_file())
    for name in ('field', 'material'):
        if (root/name).is_dir():
            total += output_bytes(root/name)
    return total


def child(root, step, executable, env, deadline):
    cursors = {}
    reason = None
    started = time.monotonic()
    if deadline <= started:
        return dict(returncode=None, stop_reason='shared_child_budget_2400s', elapsed_seconds=0)
    with (step/'seq_logs/cmfgen.stdout').open('xb') as out, (step/'seq_logs/cmfgen.stderr').open('xb') as err:
        process = subprocess.Popen(['srun', '--cpu-bind=cores', str(executable)], cwd=step,
            env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True)
        while process.poll() is None:
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                reason = 'shared_child_budget_2400s'
                stop_child(process)
                break
            try:
                process.wait(timeout=min(60, remaining))
            except subprocess.TimeoutExpired:
                reason = scan(step, cursors)
                if total_bytes(root) > 16*1024**3:
                    reason = 'bundle_root_16GiB'
                if reason:
                    stop_child(process)
                    break
    return dict(returncode=process.returncode, stop_reason=reason or scan(step, cursors),
                elapsed_seconds=time.monotonic()-started)


def run(original, source, root, build, native, source_record=63, max_lam='2.0D0', selection=None,
        coupling_contract=None, coupling_contract_sha256=None):
    if os.environ.get('OMP_NUM_THREADS') != '16' or not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Slurm CPU16/OMP16 내부에서만 실행')
    if root.exists():
        raise ValueError('기존 feedback 실행 보존')
    started = time.monotonic()
    deadline = started+2400
    if source_record < 63 or (source_record > 63 and max_lam != '1.0D6'):
        raise ValueError('등록된 완료 세대/증가 상한만 허용')
    coupling = None
    material_executable = native/'exe/cmfgen_dev.exe'
    if coupling_contract is not None:
        from cmfgen_coupling_contract import load_contract, witness_executable
        coupling = load_contract(coupling_contract, coupling_contract_sha256)
        material_executable = witness_executable(coupling)
    elif coupling_contract_sha256 is not None:
        raise ValueError('계약 경로 없는 신원값')
    next_record = source_record+1
    state, identity, parent = verify_material(source, original, source_record, selection, coupling)
    bm = json.loads((build/'feedback-build.json').read_text())
    if bm['expected_iteration'] != next_record or bm['pre_population_stop'] != 97 or Path(bm['native']) != native:
        raise ValueError('다른 수송 세대/종료/원래 코드')
    for name, expected in {**bm['input_sha256'], **bm['output_sha256']}.items():
        if sha(Path(name)) != expected:
            raise ValueError('봉인 build 입력 변경: '+name)
    if sha(native/'exe/cmfgen_dev.exe') != parent['binary']['sha256']:
        raise ValueError('원래 native 실행파일 변경')
    source_hashes = {str(source/n): sha(source/n) for n in RESTART_OVERLAY}
    if coupling is not None:
        source_hashes.update({str(coupling_contract): coupling['_sha256'],
            coupling['noninterference_evidence']: coupling['noninterference_evidence_sha256'],
            str(material_executable): coupling['native_witness_binary_sha256']})
    root.mkdir()
    (root/'seq_logs').mkdir()
    field, material = root/'field', root/'material'
    inputs = dict(scope=__doc__, source=str(source), original=str(original),
        source_state=identity, source_sha256=source_hashes, build=str(build),
        build_manifest_sha256=sha(build/'feedback-build.json'),
        resources='CPU16/OMP16/32GiB/GPU0 parent2700s shared2400s retry0 no node pin',
        acceptance=f'완료 물질{source_record}의 새 완주장과 native{next_record} 연결; 공동수렴/앵커 미인증',
        selection_path=str(selection) if selection else None,
        selection_sha256=sha(selection) if selection else None,
        numerical_controls=dict(MAX_LAM=max_lam, SCALE_OPT='LOCAL', LAM_SCALE_OPT='NONE'),
        native_SE_scaling='UNAVAILABLE; correction/BA norm으로 대체하지 않음',
        final_goal='NOT_JUDGED')
    if coupling is not None:
        inputs.update(coupling_contract=str(coupling_contract), coupling_contract_sha256=coupling['_sha256'],
            coupling_purpose=coupling['purpose_id'], material_executable=dict(path=str(material_executable),
                sha256=coupling['native_witness_binary_sha256']),
            native_SE_scaling='전 population 원시 평가합산 구간과 충돌 subset 충분상한; 원래 fullD/RE 미인증')
    (root/'seq_logs/feedback-inputs.json').write_text(json.dumps(inputs, ensure_ascii=False, indent=2)+'\n')
    resource.setrlimit(resource.RLIMIT_FSIZE, (8*1024**3, 8*1024**3))
    report = dict(field_complete=False, material_complete=False, joint_convergence=False,
                  released_T=False, final_goal='NOT_JUDGED')
    status = 1
    try:
        clone_material(original, source, field, native, False, source_record, max_lam)
        env = dict(os.environ, LUMINA_FORMAL_PRIMARY='1', LUMINA_FORMAL_PRIMARY_WHOLE='1',
            LUMINA_FORMAL_CELL_BUDGET='', LUMINA_WHOLE_MODE='', LUMINA_MOM_CAPTURE_MODE='',
            LUMINA_FORMAL_CAPTURE_MODE='', LUMINA_NATIVE_SE_WITNESS='')
        if coupling is not None:
            from cmfgen_coupling_contract import claim_generation
            report['coupling_generation_claim'] = claim_generation(coupling, source_record, root)
        report['field'] = child(root, field, build/'cmfgen_feedback_field.exe', env, deadline)
        if report['field']['returncode'] != 97 or report['field']['stop_reason']:
            raise ValueError('새 장 계산 실패')
        done = (field/'FORMAL_PRIMARY_COMPLETE.txt').read_text().splitlines()[0].split()
        if done[:3] != [str(next_record), '197291', '197291']:
            raise ValueError('새 장 완료 세대/coverage 오류')
        report['field_coverage'] = coverage(field/'FORMAL_PRIMARY_SWEEP.csv')
        g, a = fields(field/'FORMAL_PRIMARY_FIELDS.bin')
        if report['field_coverage']['count'] != 197291 or a['field'].shape != (197291, 8, 90):
            raise ValueError('새 장 불완전')
        validate_edd(field, a)
        if sha(field/'SCRTEMP') != source_hashes[str(source/'SCRTEMP')]:
            raise ValueError('pre-pop 수송에서 물질 변경')
        report['field_complete'] = True
        report['field_evidence_sha256'] = {n: sha(field/n) for n in (
            'FORMAL_PRIMARY_FIELDS.bin', 'FORMAL_PRIMARY_SWEEP.csv', 'FORMAL_PRIMARY_COMPLETE.txt',
            'EDDFACTOR', 'EDDFACTOR_INFO')}
        # 남은 시간 부족 시 완주장만 보존한다. 자식별40분이나 자동 재시도 금지.
        if deadline-time.monotonic() < 540:
            raise ValueError('물질 갱신의 계획 예산9분 미만; 완주장만 보존')
        clone_material(original, source, material, native, True, source_record, max_lam)
        for name in ('EDDFACTOR', 'EDDFACTOR_INFO'):
            shutil.copy2(field/name, material/name)
        fixed = {n: sha(material/n) for n in ('EDDFACTOR', 'EDDFACTOR_INFO')}
        material_env = dict(env, LUMINA_FORMAL_PRIMARY='', LUMINA_FORMAL_PRIMARY_WHOLE='')
        if coupling is not None:
            material_env['LUMINA_NATIVE_SE_WITNESS'] = '1'
            execution = dict(coupling_contract_sha256=coupling['_sha256'], input_record=source_record,
                radiation_record=next_record, radiation_material_record=source_record,
                original_native_binary=parent['binary'], actual_executable=inputs['material_executable'],
                observer_enabled=True, extra_observation_native=False)
            (material/'seq_logs/material-execution.json').write_text(json.dumps(execution, ensure_ascii=False, indent=2)+'\n')
        report['material'] = child(root, material, material_executable, material_env, deadline)
        if report['material']['returncode'] != 0 or report['material']['stop_reason']:
            raise ValueError('native 물질 갱신 실패')
        nd, nt = read_dimensions(material)
        reader = ScrtempReader(material/'SCRTEMP', nd, nt)
        if reader.last_it != next_record or point_records(material) != dict(POINT1=next_record, POINT2=next_record):
            raise ValueError('실제 다음 세대 완료 물질 없음')
        updated = reader.state(next_record)
        if (not np.isfinite(updated).all() or np.any(updated[:-2] < 0) or
            np.any(updated[-2:] <= 0) or np.any(updated[state > 0] <= 0) or
            not np.array_equal(updated[-1], state[-1])):
            raise ValueError('비물리 물질 또는 고정T 변경')
        if any(sha(material/n) != h for n, h in fixed.items()):
            raise ValueError('물질 갱신에서 고정J 변경')
        variables = read_variables(material, nt-2)
        charge = sum(v.charge*updated[v.index-1] for v in variables)
        report['max_charge_residual'] = float(np.max(abs(charge-updated[-2])/updated[-2]))
        report['element_total_change_over_parent'] = {}
        for species in sorted({v.species for v in variables}):
            ix = [v.index-1 for v in variables if v.species == species]
            report['element_total_change_over_parent'][species] = float(np.max(abs(updated[ix].sum(axis=0)/state[ix].sum(axis=0)-1)))
        report['original_element_equation_residual'] = 'NOT_REPLACED_BY_CHANGE_OVER_PARENT'
        if report['max_charge_residual'] > 1e-10 or max(report['element_total_change_over_parent'].values()) > 1e-10:
            raise ValueError('교대 물질 전하/봉인 부모 원소총량 보존 위반')
        report.update(material_complete=True, max_ne_relative_step=float(np.max(abs(updated[-2]/state[-2]-1))),
            material_record=next_record, radiation_produced_from_record=source_record,
            invalid_material_outputs=['dummy H/K에 의존한 flux/heat/OBSFLUX'],
            material_evidence_sha256={n: sha(material/n) for n in ('SCRTEMP', 'POINT1', 'POINT2', 'RVTJ')})
        # 완료 상태는 보존한다. 전체 step 전제 미입증은 후속 outer를 막을 뿐,
        # 이미 끝난 물질 계산을 없었던 것으로 표시하지 않는다.
        report['next_outer_eligible'] = False
        try:
            solutions = parse_solutions(material/'STEQ_VALS', nt, nd)
            if not solutions:raise ValueError('실제 native 보정 배열 없음')
            scale = re.findall(r'The local minimum value of scale is:\s*(\S+)', (material/'OUTGEN').read_text())
            report['printed_minimum_scale'] = float(scale[-1]) if scale else None
            report['actual_full_step_within_print_precision'] = applied_full_step(state, updated, solutions[-1])
            report['next_outer_eligible'] = (report['printed_minimum_scale']==1.0 and
                report['actual_full_step_within_print_precision'])
        except (ValueError, OSError, MemoryError) as error:
            report['step_evidence_error'] = str(error)
        if coupling is not None:
            from cmfgen_coupling_contract import step_evidence
            from run_cmfgen_native_se_observation import assess
            report['coupling_contract_sha256'] = coupling['_sha256']
            report['coupling_step'] = step_evidence(material, report)
            report['coupling_next_input_permitted'] = report['coupling_step']['allowed']
            report['native_SE_assessment'] = assess(material, state, source_record,
                radiation_produced_from_record=source_record)
            report['native_SE_evidence_sha256'] = {n: sha(material/n) for n in (
                f'NATIVE_SE_BOUND_{next_record:06d}.bin', 'seq_logs/native-se-assessment.json',
                'seq_logs/population-se-rows.csv')}
        status = 0
    except (ValueError, OSError) as error:
        report['failure'] = str(error)
    finally:
        report.update(elapsed_seconds=time.monotonic()-started, root_bytes=total_bytes(root),
            source_unchanged=all(sha(Path(p)) == h for p, h in source_hashes.items()))
        if not report['source_unchanged'] or report['root_bytes'] > 16*1024**3:
            status = 1
            report['material_complete'] = False
            report['failure'] = 'source 보존 또는 기존 root16GiB 한도 위반'
        (root/'seq_logs/feedback-runtime.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False))
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('original', 'source', 'run', 'build', 'native'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--source-record', type=int, default=63)
    parser.add_argument('--max-lam', choices=('2.0D0', '1.0D6'), default='2.0D0')
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--coupling-contract', type=Path)
    parser.add_argument('--coupling-contract-sha256')
    args = parser.parse_args()
    raise SystemExit(run(*(getattr(args, k).resolve() for k in ('original', 'source', 'run', 'build', 'native')),
        source_record=args.source_record, max_lam=args.max_lam,
        selection=args.selection.resolve() if args.selection else None,
        coupling_contract=args.coupling_contract.resolve() if args.coupling_contract else None,
        coupling_contract_sha256=args.coupling_contract_sha256))
