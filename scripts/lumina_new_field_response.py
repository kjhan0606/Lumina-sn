#!/usr/bin/env python3
"""완성된 새 자체 J의 전원소 물질응답과 같은 반경 CMFGEN70 비교.

수용 T는 고정하고 ne/SE만 갱신한다. 유한 F와 전하 수렴을 RE/스펙트럼
동등성으로 표시하지 않는다. old J70 율이나 CONT_FREQ를 재사용하지 않는다.
"""
import argparse
from contextlib import contextmanager
import csv
import json
import os
from pathlib import Path
import resource
import shutil
import struct
import time

import numpy as np

import cmfgen_model_continuum as B
import cmfgen_model_sources as S
from lumina_accepted_field_input import open_prepared
from lumina_fullsr_sc_actual import InputField, publish
from lumina_common_charge import solve
from lumina_model_collision_source import ModelCollisionSource
from lumina_model_freefree import ModelFreeFree
from lumina_model_linear_lines import LinearModelLines
from lumina_model_material_bridge import MaterialBridge
from lumina_model_material_rates import ModelMaterialRates
from lumina_model_response import ModelResponse
from lumina_model_response_balance import ResponseBalance
from run_lumina_common_material import absent_partial_auto_inputs
from run_lumina_full_material_response import PARAMETERS


@contextmanager
def open_field(preparation, preparation_sha, feedback, feedback_sha):
    """완성된 마지막 F만 소비한다. F2 실패 시 보존된 F1도 명시적으로 허용한다."""
    preparation, feedback = Path(preparation).resolve(), Path(feedback).resolve()
    if B.sha(feedback) != feedback_sha: raise ValueError('새 장 되먹임 기록 변경')
    report = json.loads(feedback.read_text())
    if report.get('schema') != 'lumina-resident-fixed-material-feedback-v1':
        raise ValueError('새 물질 되먹임 기록 필요')
    last = report.get('last_complete')
    if not last: raise ValueError('완성 자체장 없음; 부분 J 사용 금지')
    p, source, _, _ = open_prepared(preparation, preparation_sha)
    with source:
        if (report['source_identity'] != source.identity or
                report['accepted_preparation_sha256'] != preparation_sha):
            raise ValueError('새 장과 물질의 계보 불일치')
        child = feedback.parent/('F-'+str(last['sweep']))
        entry = report['sweeps'][last['sweep']-1]
        if (entry['state'] != 'COMPLETE_F_NOT_CONVERGENCE' or
                Path(last['J']['path']) != child/'J.npy' or
                B.sha(child/'resident-field.json') != entry['result_sha256']):
            raise ValueError('완성 장 기록/주소 변경')
        field = InputField(source, last['J']['path'], last['J']['sha256'])
        try:
            record = json.loads((child/'resident-field.json').read_text())
            if record['moments'] != last['moments'] or B.sha(Path(last['moments']['path'])) != last['moments']['sha256']:
                raise ValueError('완성 모멘트 변경')
            yield p, source, field, last
        finally:
            field.close()


def trapezoid_weights(nu):
    nu = np.asarray(nu)
    if nu.ndim != 1 or len(nu) < 2 or not np.isfinite(nu).all() or np.any(nu <= 0) or np.any(np.diff(nu) <= 0):
        raise ValueError('원 주파수 격자 오류')
    width = np.zeros_like(nu, dtype='f8')
    width[:-1] += .5*np.diff(nu); width[1:] += .5*np.diff(nu)
    return width


def comparison(columns, reference, accepted, candidate, radius):
    """동일 원자주소/반경의 이온분율. 작은 종의 log cutoff를 합격 기준으로 쓰지 않는다."""
    arrays = [np.asarray(v) for v in (reference, accepted, candidate)]
    if any(v.shape != (len(columns), len(radius)) or not np.isfinite(v).all() or np.any(v < 0) for v in arrays):
        raise ValueError('비교 인구 형상/값 오류')
    rows = []; summary = []
    for z in sorted(set(map(int, columns[:, 0]))):
        element = columns[:, 0] == z
        totals = [v[element].sum(axis=0) for v in arrays]
        if any(np.any(v <= 0) for v in totals): raise ValueError('원소 총수0 비교 불가')
        for charge in sorted(set(map(int, columns[element, 1]))):
            select = element & (columns[:, 1] == charge)
            fractions = [v[select].sum(axis=0)/total for v, total in zip(arrays, totals)]
            delta = fractions[2]-fractions[0]; index = int(np.argmax(abs(delta)))
            summary.append(dict(Z=z, charge=charge, max_absolute_fraction_difference=float(abs(delta[index])),
                node=index, radius_cm=float(radius[index]), reference_fraction=float(fractions[0][index]),
                new_response_fraction=float(fractions[2][index]),
                max_change_from_accepted=float(np.max(abs(fractions[2]-fractions[1])))))
            if z in (14, 16):
                for i, r in enumerate(radius):
                    rows.append(dict(node=i, radius_cm=float(r), Z=z, charge=charge,
                        CMFGEN70_fraction=float(fractions[0][i]), accepted_fraction=float(fractions[1][i]),
                        new_J_response_fraction=float(fractions[2][i]), difference=float(delta[i])))
    return summary, rows


def memory_inventory(ns, nf, nr, nl, nfull):
    # 복사본과 해시의 bytes 임시값, BF longdouble 누적 및 재인수 배열을 포함한다.
    parts = dict(field_copies_and_hash_temporaries=6*ns*nf*8, frequency_vectors=16*nf*8,
        rate_arrays=12*nr*ns*6*8, line_arrays=3*nl*ns*8,
        material_arrays=40*nfull*ns*8, runtime_collision_and_SE_reserve=8*1024**3)
    return dict(parts=parts, known_allocation_and_reserve_bytes=sum(parts.values()),
                whole_process_peak_RSS_verified=False)


def run(args):
    if (os.environ.get('SLURM_CPUS_PER_TASK') != '16' or os.environ.get('SLURM_MEM_PER_NODE') != '32768'
            or os.environ.get('OMP_NUM_THREADS') != '16' or not 600 <= args.seconds <= 13800):
        raise ValueError('CPU16/32GiB/OpenMP16·명시한 단회 시간 필요')
    out = args.out.resolve()
    if out.exists(): raise FileExistsError('기존 물질응답 보존')
    if shutil.disk_usage(out.parent).free < 8*1024**3: raise OSError('새 율/후보 보존 여유8GiB 필요')
    out.mkdir(); started = time.monotonic(); deadline = started+args.seconds
    result = dict(state='STARTED', final_goal='NOT_MET', source_commit=args.source_commit,
        fixed_accepted_temperature=True, new_transport=0, new_native=0, evaluations=[],
        scattering_convergence_certified=False, RE_solved=False, full_BF_integrations=0,
        comparison_baseline='CMFGEN70 fixed-T nonconverged reference, not released-T anchor',
        difference_interpretation='joint change of J, frequency grid and continuum evaluation; not isolated field effect')
    def save():
        result.update(wall_seconds=time.monotonic()-started,
                      peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        publish(out/'result.json', result)
    def check():
        if time.monotonic() >= deadline: raise TimeoutError('새 J 물질응답 시간 한도')
    save()
    try:
        with open_field(args.preparation, args.preparation_sha, args.feedback, args.feedback_sha) as (p, source, field, last):
            ns, nf = source.ns, source.nf
            if (ns, nf) != (90, 1115020): raise ValueError('원90점·전주파수 범위 변경')
            manifest = json.loads(Path(p['accepted_material']).read_text())
            original_nodes = [Path(n) for n in manifest['input_sha256'] if n.endswith('/native_material_nodes.bin')]
            model_paths = [Path(n) for n in manifest['input_sha256'] if n.endswith('/MODEL')]
            if len(original_nodes) != 1 or len(model_paths) != 1: raise ValueError('원 물질/모형 주소 모호')
            material = Path(p['material']); runroot = model_paths[0].parent; atomic = material.parent/'atomic'
            source_report = json.loads(Path(p['sources']).read_text())
            phot = Path('/gpfs/kjhan/cmfgen_runs/lumina_model_spec_428222/cmfgen_phot_smoothing.so')
            kernel = Path('/gpfs/kjhan/cmfgen_runs/lumina_full_source_tests_20261020-0sY1jL/kernels/model_sources.so')
            inputs = {str(Path(v).resolve()): B.sha(v) for v in (args.preparation, args.feedback,
                args.library, args.line_library, PARAMETERS, phot, kernel)}
            for path, digest in manifest['input_sha256'].items():
                if B.sha(Path(path)) != digest: raise ValueError('수용 물질의 원 입력 변경')
                inputs[path] = digest
            for path in (phot, kernel):
                if B.sha(path) != source_report['input_sha256'][str(path)]: raise ValueError('계수/율 원자 라이브러리 불일치')
            rows, models, routes, policy, _, _ = B.load_model(runroot, atomic, phot)
            if policy != source_report['process_selection']: raise ValueError('원자 과정 계약 변경')
            absent_partial_auto_inputs(runroot, models)
            state = B.read_material(material, rows, models)
            reference = B.read_material(original_nodes[0], rows, models)
            if not np.array_equal(state['columns'], reference['columns']): raise ValueError('원자주소 변경')
            if (not np.array_equal(state['physical'][:, 0], source.radius) or
                    not np.array_equal(reference['physical'][:, 0], source.radius)):
                raise ValueError('장/수용 물질/참조 물질 반경 변경')
            lines, _, line_inputs = S.load_lines(runroot, rows, models)
            inputs.update(line_inputs)
            if (len(models), len(rows), len(routes), len(lines['lower'])) != (27, 24542, 24667, 790493):
                raise ValueError('전모형 율 roster 누락')
            result['memory_inventory'] = memory_inventory(ns, nf, len(routes), len(lines['lower']), len(rows))
            if result['memory_inventory']['known_allocation_and_reserve_bytes'] > 32*1024**3:
                raise MemoryError('새 장 소비의 알려진 할당/예비분32GiB 초과')
            result.update(field=last, source_identity=source.identity, input_sha256=inputs,
                phase='NEW_LINEAR_BB', frequencies=nf, points=ns); save()
            nu = source.frequency; width = trapezoid_weights(nu)
            linear = LinearModelLines(lines, S.fixed_profile(runroot/'MODEL'), args.line_library,
                                      nu, field.values.T, radiation_generation=1)
            jbar = np.lib.format.open_memmap(out/'jbar.npy', mode='w+', dtype='f8', shape=(len(lines['lower']), ns))
            status = np.zeros(len(jbar), dtype='u1')
            for first in range(0, len(jbar), 4096):
                check(); end = min(first+4096, len(jbar)); values = linear.contract(first, end, radiation_generation=1)
                jbar[first:end] = values['jbar']; status[first:end] = values['status']
            jbar.flush(); np.save(out/'line-status.npy', status)
            result['BB_signature'] = linear.signature; del linear
            if not np.isin(status, [1, 2]).all(): raise ValueError('새 장 선 프로파일 범위 결손; 0 채움 금지')
            provider = ModelMaterialRates(routes, rows, models, args.library, nu, width, field.values,
                radiation_generation=1, dissolution_enabled=B.dissolution_flag(runroot/'MODEL'))
            trial = provider.prepare(state, material_generation=1)
            result.update(phase='NEW_FULL_BF', BF_field_signature=provider.field_signature); save()
            tick = time.monotonic()
            bound, total = provider.integrate(trial, material_generation=1, radiation_generation=1, check=check)
            result.update(full_BF_integrations=1, full_BF_seconds=time.monotonic()-tick); del total
            np.save(out/'bound-rates.npy', bound); save(); check()
            roster = dict(lower=np.array([r.lower for r in provider.routes]), target=np.array([r.target for r in provider.routes]),
                Z=np.array([r.Z for r in provider.routes]), stage=np.array([r.stage for r in provider.routes]),
                threshold=provider.edge, sigma_edge=provider.sigma_edge,
                phot_grid=np.array([r.phot_grid for r in provider.routes]),
                probability=np.array([r.probability for r in provider.routes]))
            np.savez(out/'route-roster.npz', **roster)
            ff = ModelFreeFree(nu, width, field.values, S.SourceKernels(kernel), models, state['columns'], radiation_generation=1)
            ffc = ff.prepare(state['temperature'], radiation_generation=1); check()
            ffcoef = dict(signature=ff.signature, temperature=ffc.temperature, heating=ffc.heating, cooling=ffc.cooling)
            np.savez(out/'FF-coefficients.npz', **ffcoef)
            reference_table = Path(__file__).resolve().parents[1]/'data/atomic/ionization_reference.csv'
            balance = ResponseBalance(runroot, rows, models, state['columns'], reference_table,
                source.radius, state['epoch_seconds'], args.library, roster, lines, jbar, ff, ffcoef, state['superlevel'])
            inputs.update(balance.owner.input_sha256); inputs.update(balance.decay_metadata['input_sha256'])
            occupation = out/'occupation.bin'
            with occupation.open('xb') as stream:
                stream.write(struct.pack('<8sII', b'LMATOCC1', len(rows), ns)); stream.write(trial.state['occupation'].astype('<f8').tobytes())
            candidates = {}; evaluation_roots = {}
            def record(data, metrics, current_trial, rates):
                root = out/('evaluation-'+str(len(result['evaluations'])+1)); root.mkdir()
                np.savez(root/'response.npz', **data)
                energy, summary = balance.evaluate(data, current_trial, rates)
                np.savez(root/'balance.npz', **energy)
                nuclei = np.max(abs(data['population'].sum(axis=0)/state['superlevel'].sum(axis=0)-1))
                if nuclei > 1.e-10: raise ValueError('새 물질 핵수 보존 실패')
                entry = dict(**metrics, **summary, nuclei_relative=float(nuclei), root=str(root),
                    response_sha256=B.sha(root/'response.npz'), balance_sha256=B.sha(root/'balance.npz'))
                publish(root/'result.json', entry); result['evaluations'].append(entry)
                candidates[id(data)] = entry; evaluation_roots[id(data)] = root; save()
            with MaterialBridge(args.library, atomic, material, occupation, PARAMETERS) as bridge:
                response = ModelResponse(provider, bridge, ModelCollisionSource(runroot, rows), lines, jbar,
                    status, trial, bound, roster, deadline=deadline)
                del bound
                def evaluate(ne): return response.evaluate(state['temperature'], ne, save=record)[0]
                result['phase'] = 'NEW_FIELD_COMMON_CHARGE'; save()
                first = evaluate(state['ne']); lower = np.zeros(ns); upper = np.zeros(ns)
                for z in response.species:
                    select = state['columns'][:, 0] == z; n = state['superlevel'][select].sum(axis=0)
                    lower += state['columns'][select, 1].min()*n; upper += state['columns'][select, 1].max()*n
                def decision(data, entry): publish(evaluation_roots[id(data)]/'decision.json', entry)
                solved = solve(evaluate, state['ne'], first['charge_residual'], lower, upper,
                               max_evaluations=7, on_evaluation=decision)
                final = first if solved['payload'] is None else solved['payload']
                for ion in response.ions.values(): inputs.update(ion.identity)
                result.update(charge_converged=solved['converged'], charge_history=solved['history'],
                    selected_response=candidates[id(final)])
            summary, table = comparison(state['columns'], reference['superlevel'], state['superlevel'], final['population'], source.radius)
            with (out/'Si-S-comparison.csv').open('x') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
            result['ion_comparison'] = summary
            moments = np.load(last['moments']['path'], mmap_mode='r', allow_pickle=False)
            try:
                if moments.shape != (3, ns, nf) or moments.dtype != np.dtype('f8'): raise ValueError('모멘트 형상 오류')
                for node in (0, ns-1):
                    if not np.isfinite(moments[:, node]).all(): raise ValueError('끝점 모멘트 비유한')
                result['native_endpoint_luminosity_erg_s'] = {
                    name: float(16*np.pi**2*source.radius[node]**2*np.sum(moments[1, node]*width, dtype=np.longdouble))
                    for name, node in (('outer_native_point_not_escape_observer', 0), ('inner_native_point', ns-1))}
            finally: moments._mmap.close()
            if any(B.sha(Path(p)) != h for p, h in inputs.items()): raise ValueError('새 장 응답 중 원 입력 변경')
            result['state'] = 'COMPLETE_FINITE_NEW_FIELD_RESPONSE_NOT_JOINT_CONVERGENCE'
    except Exception as error:
        result.update(state='INCOMPLETE_PRESERVED', error=type(error).__name__+': '+str(error))
    finally: save()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('preparation', 'feedback', 'library', 'line-library', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('preparation-sha', 'feedback-sha', 'source-commit'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--seconds', type=int, default=6600)
    result = run(parser.parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result['state'] == 'COMPLETE_FINITE_NEW_FIELD_RESPONSE_NOT_JOINT_CONVERGENCE' else 1)
