#!/usr/bin/env python3
"""수용 T/ne/SL 후보를 같은 반경점의 새 계수 입력으로 내보낸다.

저장 자료의 직렬화와 분율 재현뿐이다. C full-level export, BF/BB 적분,
SE/RE, 수송은 실행하지 않으며 기존 물질70 manifest를 가장하지 않는다.
"""
import argparse
import csv
import json
from pathlib import Path
import struct

import numpy as np

import cmfgen_model_continuum as B
from prepare_lumina_material_cells import read_nodes
from prepare_lumina_material_seed import atomic_asset
from lumina_thermal_replay import ENERGY_POLICIES, energy_policy


def relative_error(actual, expected):
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('물질/분율의 형상 또는 유한성 오류')
    scale = np.maximum(abs(a), abs(b))
    return float(np.max(np.divide(abs(a-b), scale, out=np.zeros_like(scale), where=scale > 0)))


def validate(candidate, columns, physical, population, rows, models):
    """수용 물질을 정규화·재평형하지 않고 실제 보존량과 위상을 확인한다."""
    n, nc, nl = len(physical), len(columns), len(rows)
    p = candidate['population']; te = candidate['temperature']; ne = candidate['ne']
    expected = {(z, stage-1, sl) for (z, stage), m in models.items() for sl in range(m['n_super'])}
    expected |= {(z, stage, -1) for z, stage in models if (z, stage+1) not in models}
    if (columns.dtype.kind not in 'iu' or len(set(map(tuple, columns))) != nc or
            set(map(tuple, columns)) != expected or
            not np.array_equal(candidate['columns'], columns) or
            p.shape != (nc, n) or te.shape != (n,) or ne.shape != (n,) or
            not all(np.isfinite(a).all() and np.all(a > 0) for a in (p, te, ne)) or
            candidate['fraction'].shape != (nl, n) or candidate['occupation'].shape != (nl, n)):
        raise ValueError('수용 물질의 SL/terminal 주소 또는 T/ne/인구 오류')
    species = np.unique(columns[:, 0]); ledger = candidate['ledger']
    if (ledger.shape != (len(species), n, 7) or not np.isfinite(ledger).all() or
            np.any(ledger[:, :, 6] < 0) or np.max(ledger[:, :, 6]) > 1.e-8):
        raise ValueError('수용 후보의 정상 SE 증거 오류')
    q = np.sum(np.asarray(p, dtype=np.longdouble)*columns[:, 1, None], axis=0)-ne
    if (np.max(abs(q/ne)) > 1.e-10 or
            candidate['charge_residual'].shape != ne.shape or
            not np.isfinite(candidate['charge_residual']).all() or
            np.max(abs((q-candidate['charge_residual'])/ne)) > 1.e-12):
        raise ValueError('수용 후보의 실제 전하 잔차 오류')
    nuclei = 0.
    for z in species:
        select = columns[:, 0] == z
        nuclei = max(nuclei, relative_error(p[select].sum(axis=0), population[:, select].sum(axis=1)))
    if nuclei > 1.e-10:
        raise ValueError('수용 후보의 원소총수 변경')
    return dict(charge_relative_max=float(np.max(abs(q/ne))), nuclei_relative_max=nuclei,
                stationary_SE_max=float(np.max(ledger[:, :, 6])))


def thermal_selection(report):
    """전역 물질 보정의 실제 마지막 수용 후보를 찾는다. 거부 후보는 전달하지 않는다."""
    block = report['temperature_block']; order = report.get('thermal_evaluation_order', [])
    history = block['history']
    if (len(history) != len(order) or block['evaluations_used'] != len(history) or
            any(entry['evaluation'] != i for i, entry in enumerate(history, 1)) or
            any(entry.get('fixed_field_thermal_callback') is not True for entry in order)):
        raise ValueError('온도 응답의 호출 순서/완료 평가 불일치')
    accepted = [e for e in history if e.get('accepted') is True]
    # Standalone projection and the coupled solver publish distinct mode keys.
    # Both accept only the final charge-closed projected step, not its probes.
    projected = block.get('charge_projection') or block.get('charge_projected_trials')
    accepted_phase = 'charge_projected_line' if projected else 'line_search'
    if (block['accepted_iterations'] != len(accepted) or
            any(e['phase'] != accepted_phase for e in accepted)):
        raise ValueError('온도 응답의 실제 수용 반복 불일치')
    if accepted:
        index = accepted[-1]['evaluation']; expected = order[index-1]
        kind = 'FINITE_NEW_FIELD_TNE_ACCEPTED_STEP'
    elif block['fixed_field_numerical_converged'] is True:
        index = 0; expected = report['thermal_initial_response']
        kind = 'FINITE_NEW_FIELD_TNE_CONVERGED_BASE'
    else:
        raise ValueError('수용된 온도 보정이 없음; 실패 후 기존 물질을 새 후보로 전달하지 않음')
    if report['selected_response'] != expected:
        raise ValueError('온도 응답의 마지막 수용 후보와 선택 결과 불일치')
    return index, expected, kind


def thermal_checks(report, candidate, directory):
    """고정-J 응답 개선 또는 사전 RE/전하 문턱의 실제 수렴을 확인한다."""
    first = Path(report['thermal_initial_response']['root'])
    policy = energy_policy(report, default='physical_stored')
    if policy not in ENERGY_POLICIES: raise ValueError('온도 후보의 열방정식 정책 오류')
    ns = len(candidate['ne'])
    with np.load(first/'balance.npz', allow_pickle=False) as initial, np.load(directory/'balance.npz', allow_pickle=False) as final:
        old = initial[policy+'_net']; gross = initial[policy+'_gross']
        new = final[policy+'_net']; new_gross = final[policy+'_gross']
        if (any(a.shape != (ns,) or not np.isfinite(a).all()
                for a in (old, gross, new, new_gross)) or
                np.any(gross < 0) or np.any(new_gross < 0)):
            raise ValueError('온도 응답의 실제 열장부 형상/값 오류')
        scale = .5*(gross+abs(old))
        selected_scale = .5*(new_gross+abs(new))
        if np.any(scale <= 0): raise ValueError('초기 물질 열잔차 척도 오류')
        if np.any(selected_scale <= 0): raise ValueError('선택 물질 열잔차 척도 오류')
        before = float(np.max(abs(old)/scale))
        after_on_base_scale = float(np.max(abs(new)/scale))
        # v2 acceptance normalizes each physical state by its own heating/cooling
        # scale; retain the fixed initial scale separately to reproduce line search.
        after = float(np.max(abs(new)/selected_scale))
    with np.load(first/'response.npz', allow_pickle=False) as initial:
        ne = initial['ne']
        if ne.shape != (ns,) or not np.isfinite(ne).all() or np.any(ne <= 0):
            raise ValueError('초기 물질 전하잔차 척도 오류')
        charge = float(np.max(abs(candidate['charge_residual'])/ne))
        selected_ne = candidate['ne']
        if selected_ne.shape != (ns,) or not np.isfinite(selected_ne).all() or np.any(selected_ne <= 0):
            raise ValueError('선택 물질 전하잔차 척도 오류')
        selected_charge = float(np.max(abs(candidate['charge_residual'])/selected_ne))
    block = report['temperature_block']
    converged = block['fixed_field_numerical_converged']
    charge_tolerance = block.get('charge_tolerance', 1.e-10)
    # Older sealed response runs predate an explicit tolerance field. In that
    # case use the v2 physical RE acceptance, not the solver's stricter default.
    thermal_tolerance = block.get('thermal_tolerance', 1.e-3)
    if (not np.isfinite([charge_tolerance, thermal_tolerance]).all() or
            charge_tolerance <= 0 or thermal_tolerance <= 0 or
            charge_tolerance > 1.e-10 or thermal_tolerance > 1.e-3):
        raise ValueError('고정장 전하/RE 정지기준이 사전 허용치보다 느슨함')
    allowed_thermal = min(thermal_tolerance, 1.e-3)
    if after > allowed_thermal:
        raise ValueError('고정장 물질 전달의 실제 열잔차가 사전 허용치 초과')
    if converged:
        if selected_charge > min(charge_tolerance, 1.e-10) or after > min(thermal_tolerance, 1.e-3):
            raise ValueError('고정장 수렴 표기와 실제 전하/RE 잔차 불일치')
    elif not (after_on_base_scale < before or
              (after_on_base_scale <= 1.e-10 and before <= 1.e-10)):
        raise ValueError('온도 보정의 실제 열잔차 개선 없음')
    return dict(thermal_initial_scaled_max=before, thermal_selected_scaled_max=after,
                thermal_selected_base_scaled_max=after_on_base_scale,
                charge_initial_scaled_max=charge, fixed_field_numerical_converged=converged,
                charge_physical_scaled_max=selected_charge,
                charge_tolerance=charge_tolerance, thermal_tolerance=thermal_tolerance)


def prepare(result, result_sha, seed, run, out):
    result, seed, run, out = [Path(p).resolve() for p in (result, seed, run, out)]
    if out.exists():
        raise FileExistsError('기존 수용 물질 입력 보존')
    if B.sha(result) != result_sha:
        raise ValueError('봉인 수용 보고서 신원 불일치')
    report = json.loads(result.read_text())
    policy = energy_policy(report, default='physical_stored' if report['state'] ==
        'COMPLETE_FINITE_NEW_FIELD_TNE_RESPONSE_NOT_JOINT_CONVERGENCE' else None)
    if report['state'] == 'COMPLETE_DEPTH_COUPLED_MATERIAL_STEP' and report['charge_solved'] is True:
        accepted = [r for r in report['history'] if r.get('accepted') is True]
        if not accepted or not 1 <= accepted[-1]['evaluation'] <= len(report['evaluations']):
            raise ValueError('수용 evaluation 주소 없음')
        index = accepted[-1]['evaluation']; directory = result.parent/('evaluation-'+str(index))
        if Path(report['accepted_source_directory']).resolve() != directory:
            raise ValueError('수용 evaluation 주소 불일치')
        entry = report['evaluations'][index-1]['output_sha256']
        response = result.parent/'accepted-response.npz'; response_sha = report['accepted_output_sha256']
        if entry['response.npz'] != response_sha:
            raise ValueError('수용 후보와 실제 evaluation의 신원 불일치')
        source_kind = 'DEPTH_COUPLED_ACCEPTED_STEP'
    elif (report['state'] == 'COMPLETE_FINITE_NEW_FIELD_RESPONSE_NOT_JOINT_CONVERGENCE' and
          report['charge_converged'] is True and report['fixed_accepted_temperature'] is True):
        # 첫 평가가 이미 전하 수렴했을 수 있다. 그 외에는 실제 수용된 마지막
        # secant 단계만 전달한다. 이것은 새 장·물질 공동수렴이나 RE 해가 아니다.
        selected = report['selected_response']; directory = Path(selected['root']).resolve()
        matches = [i for i, item in enumerate(report['evaluations'], 1) if item == selected]
        if len(matches) != 1: raise ValueError('새 장 응답의 선택 평가 누락/중복')
        index = matches[0]
        if directory != result.parent/('evaluation-'+str(index)):
            raise ValueError('새 장 응답의 선택 평가 주소 불일치')
        accepted = [item['evaluation'] for item in report['charge_history'] if item.get('accepted') is True]
        if (index == 1 and accepted) or (index > 1 and (not accepted or accepted[-1] != index-1)):
            raise ValueError('새 장 응답의 마지막 전하 수용 단계 불일치')
        response = directory/'response.npz'; response_sha = selected['response_sha256']
        entry = {'response.npz': response_sha, 'balance.npz': selected['balance_sha256']}
        source_kind = 'FINITE_NEW_FIELD_FIXED_T_CHARGE_RESPONSE'
    elif (report['state'] == 'COMPLETE_FINITE_NEW_FIELD_TNE_RESPONSE_NOT_JOINT_CONVERGENCE' and
          report['charge_converged'] is True and report['fixed_accepted_temperature'] is False):
        index, selected, source_kind = thermal_selection(report)
        directory = Path(selected['root']).resolve()
        response = directory/'response.npz'; response_sha = selected['response_sha256']
        entry = {'response.npz': response_sha, 'balance.npz': selected['balance_sha256']}
    else:
        raise ValueError('전하 수렴이 확인된 지원 물질 후보가 아님')
    identities = {str(result): result_sha, str(response): response_sha,
                  **{str(directory/name): entry[name] for name in ('response.npz', 'balance.npz')}}
    thermal = source_kind.startswith('FINITE_NEW_FIELD_TNE_')
    if thermal:
        first = report['thermal_initial_response']
        for name, key in (('response.npz', 'response_sha256'), ('balance.npz', 'balance_sha256')):
            path = str(Path(first['root'])/name)
            if path in identities and identities[path] != first[key]:
                raise ValueError('초기/선택 물질 파일의 신원 충돌')
            identities[path] = first[key]
    # 전체 큰 장/계수를 다시 읽지 않는다. 이번 전달에서 실제 소비하는 봉인 입력만 확인한다.
    nodes = seed/'native_material_nodes.bin'; levels = seed/'atomic/levels.csv'
    for path in (nodes, levels, *[run/name for name in ('MODEL', 'MODEL_SPEC', 'atomic_links.txt')]):
        identities[str(path)] = report['input_sha256'][str(path)]
    manifest = seed/'material-seed.json'; original = json.loads(manifest.read_text())
    identities[str(manifest)] = B.sha(manifest)
    for name in ('native_material_nodes.bin', 'atomic/abundances.csv',
                 'atomic/element_number_densities.csv', 'atomic/electron_temperatures.csv'):
        path = str(seed/name); h = original['output_sha256'][name]
        if path in identities and identities[path] != h:
            raise ValueError('원 점값 manifest와 봉인 물질 불일치')
        identities[path] = h
    if any(B.sha(Path(p)) != h for p, h in identities.items()):
        raise ValueError('수용 물질 전달 입력 변경')
    with np.load(response, allow_pickle=False) as f:
        candidate = {k: f[k] for k in f.files}
    columns, physical, old_population, nl, epoch = read_nodes(nodes)
    rows = list(csv.DictReader(levels.open()))
    models = B.atomic.load_cmfgen_model_spec(run/'MODEL_SPEC', run/'atomic_links.txt')
    dissolved = B.dissolution_flag(run/'MODEL')
    if dissolved:
        # MODEL_SPEC 자체에는 occupation에 필요한 이온화 문턱/차폐전하가 없다.
        # 계수 생산자 load_model과 같은 OSC parser를 쓰되 PHOT/율은 적분하지 않는다.
        links = B.atomic.load_cmfgen_links(run/'atomic_links.txt')
        for key, model in models.items():
            path = Path(links[key]['osc']); expected = report['input_sha256'][str(path)]
            if B.sha(path) != expected:
                raise ValueError('occupation 원 OSC 입력 변경')
            identities[str(path)] = expected
            osc = B.parse_osc(path, include_transitions=False)
            model['ionization_eV'] = float(osc.ionization_eV); model['z_screen'] = float(osc.z_screen)
    if (nl != len(rows) or report['points'] != len(physical) or
            not np.isfinite(physical).all() or np.any(physical <= 0) or
            not np.all(np.diff(physical[:, 0]) < 0)):
        raise ValueError('원래점 기하/원자 모형 불일치')
    checks = validate(candidate, columns, physical, old_population, rows, models)
    if thermal: checks.update(thermal_checks(report, candidate, directory))
    out.mkdir(); atomic = out/'atomic'; atomic.mkdir()
    for path in (seed/'atomic').iterdir():
        if atomic_asset(path.name) and path.is_file():
            (atomic/path.name).symlink_to(path.resolve())
    # 조성과 반경/rho/epoch는 그대로, T/ne/SL은 실제 수용 배열 그대로 둔다.
    (atomic/'abundances.csv').write_bytes((seed/'atomic/abundances.csv').read_bytes())
    with (atomic/'electron_temperatures.csv').open('x', newline='') as f:
        writer = csv.writer(f); writer.writerow(['shell_id', 'T_e'])
        writer.writerows(enumerate(candidate['temperature']))
    (atomic/'element_number_densities.csv').write_bytes((seed/'atomic/element_number_densities.csv').read_bytes())
    physical = physical.copy(); physical[:, 1] = candidate['temperature']; physical[:, 2] = candidate['ne']
    material = out/'native_material_nodes.bin'
    with material.open('xb') as f:
        f.write(struct.pack('<8sIIId', b'LUMNOD01', len(physical), len(columns), nl, epoch))
        for a, dtype in ((columns, '<i4'), (physical, '<f8'), (candidate['population'].T, '<f8')):
            f.write(np.asarray(a, dtype=dtype).tobytes())
    # 새 상태에서 기존 계수 생산자와 같은 occupation/FL 복원을 실행한다. 적분은 없다.
    restored = B.read_material(material, rows, models)
    occupation = (B.material_occupation(rows, models, restored) if dissolved
                  else np.ones((nl, len(physical))))
    restored = B.read_material(material, rows, models, occupation)
    checks['occupation_relative_max'] = relative_error(occupation, candidate['occupation'])
    lookup = {tuple(k): i for i, k in enumerate(columns)}
    mapping = np.array([lookup[int(r['atomic_number']), int(r['ion_number']), int(r['super_level'])] for r in rows])
    expected_full = candidate['population'][mapping]*candidate['fraction']
    checks['restored_full_relative_max'] = relative_error(restored['full'], expected_full)
    if checks['occupation_relative_max'] > 5.e-12 or checks['restored_full_relative_max'] > 5.e-12:
        raise ValueError('새 물질 occupation/full-level이 저장 수용 상태와 다름; 재평형으로 숨기지 않음')
    with (out/'occupation.bin').open('xb') as f:
        f.write(struct.pack('<8sII', b'LMATOCC1', nl, len(physical)))
        f.write(np.asarray(occupation, dtype='<f8').tobytes())
    if any(B.sha(Path(p)) != h for p, h in identities.items()):
        raise ValueError('수용 물질 전달 중 원 입력 변경')
    names = ['native_material_nodes.bin', 'occupation.bin', 'atomic/abundances.csv',
             'atomic/element_number_densities.csv', 'atomic/electron_temperatures.csv']
    record = dict(schema='lumina-accepted-nodal-material-v1', status='EXPORTED_NOT_TRANSPORT_READY',
        accepted_result_sha256=result_sha, accepted_response_sha256=response_sha, source_kind=source_kind,
        accepted_evaluation=index, points=len(physical), full_levels=nl, columns=len(columns),
        source_script_sha256=B.sha(Path(__file__)), input_sha256=identities,
        coordinate_role='original outer-to-inner point samples; no averaging, interpolation or remap',
        geometry_density_epoch_unchanged=True, temperature_and_ne_from_accepted_response=True,
        boundary='inner T updated; new diffusion gradient must use new chi/eta, not the old70 array',
        C_full_population_exported=False, checks=checks,
        next='actual C import/full export, new whole-grid coefficients, new diffusion boundary, new self field/flux and material response',
        output_sha256={name: B.sha(out/name) for name in names},
        new_BF_BB_integrals=0, new_SE_solves=0, new_transport=0, final_goal='NOT_MET')
    if policy is not None:
        record['thermal_energy_policy'] = policy
    if thermal:
        record.update(accepted_evaluation_role='thermal callback order; 0 is initial response',
                      joint_convergence_certified=False)
    with (out/'accepted-material.json').open('x') as f:
        json.dump(record, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('result', 'seed', 'run', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--result-sha256', required=True)
    a = parser.parse_args()
    print(json.dumps(prepare(a.result, a.result_sha256, a.seed, a.run, a.out),
                     ensure_ascii=False, indent=2, allow_nan=False))
