#!/usr/bin/env python3
"""완성된 새 자체 J의 전원소 물질응답과 같은 반경 CMFGEN70 비교.

기본은 수용 T 고정 ne/SE 응답이며 명시적으로 선택하면 기존 감쇠 T/ne
보정을 연결한다. 유한 F와 내부 물질 수렴을 공동 RE/스펙트럼 동등성으로
표시하지 않는다. old J70 율이나 CONT_FREQ를 재사용하지 않는다.
"""
import argparse
from contextlib import contextmanager
import csv
import json
import os
from pathlib import Path
import resource
import shutil
import socket
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
import lumina_fixed_field_thermal as thermal_block
from run_lumina_fixed_state_balance import adiabatic
import lumina_completed_field_seed as completed_seed
import lumina_thermal_replay as thermal_replay
from lumina_temperature_response_cache import TemperatureResponseCache, extra_memory_reserve


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


def memory_inventory(ns, nf, nr, nl, nfull, *, temperature_cache=False):
    # 복사본과 해시의 bytes 임시값, BF longdouble 누적 및 재인수 배열을 포함한다.
    parts = dict(field_copies_and_hash_temporaries=6*ns*nf*8, frequency_vectors=16*nf*8,
        rate_arrays=12*nr*ns*6*8, line_arrays=3*nl*ns*8,
        material_arrays=40*nfull*ns*8, runtime_collision_and_SE_reserve=8*1024**3)
    if temperature_cache:
        parts['temperature_bound_cache_and_workspace'] = extra_memory_reserve(nr, ns)
    return dict(parts=parts, known_allocation_and_reserve_bytes=sum(parts.values()),
                whole_process_peak_RSS_verified=False)


def execution_resources(seconds):
    """Slurm 할당 또는 실제 수동 제한을 확인한다. 가짜 Slurm 변수는 쓰지 않는다."""
    mode = os.environ.get('LUMINA_RESPONSE_EXECUTOR', 'slurm')
    max_seconds = 42600 if mode == 'manual-lageunha' else 13800
    if os.environ.get('OMP_NUM_THREADS') != '16' or not 600 <= seconds <= max_seconds:
        raise ValueError('CPU16/32GiB/OpenMP16·명시한 단회 시간 필요')
    record = dict(executor=mode, hostname=socket.gethostname(), pid=os.getpid(),
                  CPUs=16, memory_limit_bytes=32*1024**3, seconds=seconds)
    if mode == 'slurm':
        if (os.environ.get('SLURM_CPUS_PER_TASK') != '16' or
                os.environ.get('SLURM_MEM_PER_NODE') != '32768'):
            raise ValueError('CPU16/32GiB Slurm 할당 필요')
        record['job_id'] = os.environ.get('SLURM_JOB_ID')
    elif mode == 'manual-lageunha':
        soft, _ = resource.getrlimit(resource.RLIMIT_AS)
        affinity = sorted(os.sched_getaffinity(0))
        if (socket.gethostname().split('.')[0].lower() != 'lageunha' or
                os.environ.get('SLURM_JOB_ID') or len(affinity) != 16 or
                soft <= 0 or soft > 32*1024**3):
            raise ValueError('lageunha 수동 CPU affinity16/주소공간32GiB 제한 필요')
        record.update(cpu_affinity=affinity, address_space_limit_bytes=soft)
    else:
        raise ValueError('알 수 없는 물질응답 실행 방식')
    return record


def preserve_thermal_direction(out, index, value, matrix, residual, delta,
                               charge_scale, thermal_scale, *, field,
                               source_identity, probes, policy='physical_stored'):
    """비싼 섭동을 다시 계산하지 않도록 방향 후보 실행 전에 선형계를 보존한다.

    저장은 수용/수렴 판정이나 새 후보 실행 권한이 아니다. 후보와 그 열장부는
    기존 실제 물질 callback으로 별도 평가해야 한다.
    """
    root = Path(out)/('thermal-direction-'+str(index)); root.mkdir()
    data, energy, metadata = value
    path = root/'linearization.npz'
    np.savez(path, temperature=data['temperature'], ne=data['ne'],
        jacobian=matrix, scaled_residual=residual, log_direction=delta,
        charge_scale=charge_scale, thermal_scale=thermal_scale)
    record = dict(state='SAVED_DIRECTION_NOT_ACCEPTED_CANDIDATE',
        field=field, source_identity=source_identity, base_response=metadata,
        probes=probes, rows=['charge', 'RE'], columns=['logT', 'logne'],
        energy_policy=policy, centered_log_step=1.e-4,
        numerical_tolerance=1.e-10, max_log_step=.1,
        root=str(root), linearization_sha256=B.sha(path),
        new_material_evaluations=0, new_transport=0, joint_convergence=False)
    publish(root/'result.json', record)
    return record


def run(args):
    execution = execution_resources(args.seconds)
    temperature_evaluations = getattr(args, 'temperature_evaluations', 0)
    replay_path = getattr(args, 'resume_thermal_result', None)
    replay_sha = getattr(args, 'resume_thermal_result_sha', None)
    minimum = 1 if replay_path else 5
    if type(temperature_evaluations) is not int or (temperature_evaluations != 0 and not minimum <= temperature_evaluations <= 12):
        raise ValueError('온도 보정은 비활성0, 신규5~12회 또는 보존평가 재개1~12회 필요')
    resume = getattr(args, 'resume_result', None)
    resume_sha = getattr(args, 'resume_result_sha', None)
    if bool(resume) != bool(resume_sha) or (resume and not temperature_evaluations):
        raise ValueError('완료 계수 재개는 보고서/해시 쌍과 명시한 온도 보정 필요')
    if bool(replay_path) != bool(replay_sha) or (replay_path and (not resume or not temperature_evaluations)):
        raise ValueError('열평가 재개는 보고서/해시 쌍과 완료 기준 계수 및 명시한 신규 평가 한도 필요')
    thermal_policy = getattr(args, 'thermal_policy', None)
    if (temperature_evaluations and thermal_policy is None) or (
            thermal_policy is not None and thermal_policy not in thermal_replay.ENERGY_POLICIES):
        raise ValueError('명시한 지원 열방정식 정책 필요')
    replay_prefix = getattr(args, 'thermal_replay_prefix', None)
    if replay_prefix is not None and (not replay_path or type(replay_prefix) is not int or not 1 <= replay_prefix <= 4):
        raise ValueError('초기 열섭동 재사용은 봉인 재개 보고서와1~4개 명시가 필요')
    out = args.out.resolve()
    if out.exists(): raise FileExistsError('기존 물질응답 보존')
    if shutil.disk_usage(out.parent).free < 8*1024**3: raise OSError('새 율/후보 보존 여유8GiB 필요')
    out.mkdir(); started = time.monotonic(); deadline = started+args.seconds
    result = dict(state='STARTED', final_goal='NOT_MET', source_commit=args.source_commit,
        execution=execution, fixed_accepted_temperature=temperature_evaluations == 0,
        temperature_evaluations_max=temperature_evaluations, thermal_energy_policy=thermal_policy,
        new_transport=0, new_native=0, evaluations=[],
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
    cached_response = None
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
            result['memory_inventory'] = memory_inventory(ns, nf, len(routes), len(lines['lower']), len(rows),
                                                        temperature_cache=bool(temperature_evaluations))
            if result['memory_inventory']['known_allocation_and_reserve_bytes'] > 32*1024**3:
                raise MemoryError('새 장 소비의 알려진 할당/예비분32GiB 초과')
            seed = None
            if resume:
                seed = completed_seed.load(resume, resume_sha, field=last,
                    source_identity=source.identity, state=state, inputs=inputs,
                    library=args.library, line_library=args.line_library,
                    routes=len(routes), lines=len(lines['lower']))
                result['baseline_reuse'] = seed['provenance']
                for name in completed_seed.COEFFICIENTS:
                    check(); shutil.copyfile(seed['paths'][name], out/name)
            if temperature_evaluations:
                thermal_code = ('lumina_coupled_material.py', 'lumina_model_response.py',
                    'lumina_model_response_balance.py', 'lumina_model_collision_source.py',
                    'lumina_model_material_bridge.py', 'lumina_model_energy.py',
                    'lumina_model_balance.py', 'lumina_native_heat_sources.py',
                    'run_lumina_fixed_state_balance.py')
                result['thermal_implementation_sha256'] = {
                    name: B.sha(Path(__file__).with_name(name)) for name in thermal_code}
                for name in (*thermal_code, 'lumina_fixed_field_thermal.py',
                             'lumina_completed_field_seed.py', 'lumina_thermal_replay.py',
                             'lumina_temperature_response_cache.py'):
                    path = Path(__file__).with_name(name)
                    inputs[str(path)] = B.sha(path)
            result.update(field=last, source_identity=source.identity, input_sha256=inputs,
                phase='NEW_LINEAR_BB', frequencies=nf, points=ns); save()
            nu = source.frequency; width = trapezoid_weights(nu)
            linear = LinearModelLines(lines, S.fixed_profile(runroot/'MODEL'), args.line_library,
                                      nu, field.values.T, radiation_generation=1)
            if seed:
                if linear.signature != seed['report']['BB_signature']:
                    raise ValueError('완료 BB의 실제 장/격자/프로파일/선주소 불일치')
                jbar = np.load(out/'jbar.npy', mmap_mode='r', allow_pickle=False)
                status = np.load(out/'line-status.npy', allow_pickle=False)
            else:
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
            roster = dict(lower=np.array([r.lower for r in provider.routes]), target=np.array([r.target for r in provider.routes]),
                Z=np.array([r.Z for r in provider.routes]), stage=np.array([r.stage for r in provider.routes]),
                threshold=provider.edge, sigma_edge=provider.sigma_edge,
                phot_grid=np.array([r.phot_grid for r in provider.routes]),
                probability=np.array([r.probability for r in provider.routes]))
            if seed:
                if provider.field_signature != seed['report']['BF_field_signature']:
                    raise ValueError('완료 BF의 실제 장/격자/적분 가중치 불일치')
                with np.load(out/'route-roster.npz', allow_pickle=False) as saved_roster:
                    if set(saved_roster.files) != set(roster) or any(
                            not np.array_equal(saved_roster[k], v) for k, v in roster.items()):
                        raise ValueError('완료 BF의 원자 경로 불일치')
                bound = np.load(out/'bound-rates.npy', mmap_mode='r', allow_pickle=False)
                result['full_BF_seconds'] = 0.
            else:
                tick = time.monotonic()
                bound, total = provider.integrate(trial, material_generation=1, radiation_generation=1, check=check)
                result.update(full_BF_integrations=1, full_BF_seconds=time.monotonic()-tick); del total
                np.save(out/'bound-rates.npy', bound)
                np.savez(out/'route-roster.npz', **roster)
            save(); check()
            ff = ModelFreeFree(nu, width, field.values, S.SourceKernels(kernel), models, state['columns'], radiation_generation=1)
            if seed:
                with np.load(out/'FF-coefficients.npz', allow_pickle=False) as saved_ff:
                    ffcoef = {k: saved_ff[k] for k in saved_ff.files}
                if str(ffcoef['signature']) != ff.signature:
                    raise ValueError('완료 FF의 실제 장/격자/원자주소 불일치')
            else:
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
            candidates = {}; evaluation_roots = {}; evaluation_energy = {}; replay_ids = set()
            def record(data, metrics, current_trial, rates):
                root = out/('evaluation-'+str(len(result['evaluations'])+1)); root.mkdir()
                np.savez(root/'response.npz', **data)
                energy, summary = balance.evaluate(data, current_trial, rates)
                np.savez(root/'balance.npz', **energy)
                for ion in response.ions.values(): inputs.update(ion.identity)
                nuclei = np.max(abs(data['population'].sum(axis=0)/state['superlevel'].sum(axis=0)-1))
                if nuclei > 1.e-10: raise ValueError('새 물질 핵수 보존 실패')
                entry = dict(**metrics, **summary, nuclei_relative=float(nuclei), root=str(root),
                    response_sha256=B.sha(root/'response.npz'), balance_sha256=B.sha(root/'balance.npz'))
                if metrics.get('full_BF_reintegration', not np.array_equal(data['temperature'], state['temperature'])):
                    result['full_BF_integrations'] += 1
                    entry['full_BF_reintegration'] = True
                if result.get('phase') == 'FIXED_NEW_FIELD_T_NE_CORRECTION':
                    entry['fixed_field_thermal_callback'] = True
                    result.setdefault('thermal_evaluation_order', []).append(entry)
                publish(root/'result.json', entry); result['evaluations'].append(entry)
                candidates[id(data)] = entry; evaluation_roots[id(data)] = root; save()
                evaluation_energy[id(data)] = energy
            with MaterialBridge(args.library, atomic, material, occupation, PARAMETERS) as bridge:
                response = ModelResponse(provider, bridge, ModelCollisionSource(runroot, rows), lines, jbar,
                    status, trial, bound, roster, deadline=deadline)
                del bound
                cached_response = TemperatureResponseCache(response, out=out/'temperature-bound-cache') if temperature_evaluations else None
                def material_evaluate(t, ne):
                    owner = response if cached_response is None else cached_response
                    return owner.evaluate(t, ne, save=record)
                def evaluate(ne): return material_evaluate(state['temperature'], ne)[0]
                result['phase'] = 'NEW_FIELD_COMMON_CHARGE'; save()
                replay_source = None
                if replay_path:
                    replay_source = thermal_replay.load(replay_path, replay_sha,
                        field=last, source_identity=source.identity,
                        implementation=result['thermal_implementation_sha256'], state=state,
                        library=args.library, line_library=args.line_library, inputs=inputs,
                        policy=thermal_policy, prefix=replay_prefix)
                    result['thermal_replay'] = replay_source['provenance']
                    for data, energy, metadata in [replay_source['first'], *replay_source['replay']]:
                        candidates[id(data)] = metadata; evaluation_energy[id(data)] = energy
                        replay_ids.add(id(data))
                    final = replay_source['first'][0]
                    solved = dict(history=replay_source['charge_history'])
                else:
                    initial_ne = state['ne'] if seed is None else seed['ne']
                    first = evaluate(initial_ne); lower = np.zeros(ns); upper = np.zeros(ns)
                    for z in response.species:
                        select = state['columns'][:, 0] == z; n = state['superlevel'][select].sum(axis=0)
                        lower += state['columns'][select, 1].min()*n; upper += state['columns'][select, 1].max()*n
                    def decision(data, entry): publish(evaluation_roots[id(data)]/'decision.json', entry)
                    solved = solve(evaluate, initial_ne, first['charge_residual'], lower, upper,
                                   max_evaluations=7, on_evaluation=decision)
                    final = first if solved['payload'] is None else solved['payload']
                if temperature_evaluations:
                    result['thermal_initial_response'] = candidates[id(final)]
                    result['charge_history'] = solved['history']
                    result['phase'] = 'FIXED_NEW_FIELD_T_NE_CORRECTION'; save()
                    def evaluate_temperature(t, ne):
                        data = material_evaluate(t, ne)[0]
                        return data, evaluation_energy[id(data)], candidates[id(data)]
                    def full_ad(t, ne, ni, u):
                        return adiabatic(args.library, source.radius, state['epoch_seconds'], t, ne, ni, u)['signed_total']
                    def thermal_decision(value, entry):
                        entry['charge_relative'] = float(np.max(abs(value[0]['charge_residual']/value[0]['ne'])))
                        if id(value[0]) not in replay_ids:
                            publish(evaluation_roots[id(value[0])]/'thermal-decision.json', entry)
                    def replayed(value):
                        result.setdefault('thermal_evaluation_order', []).append(value[2]); save()
                    def thermal_direction(value, matrix, residual, delta, charge_scale, thermal_scale):
                        records = result.setdefault('thermal_directions', [])
                        records.append(preserve_thermal_direction(out, len(records)+1,
                            value, matrix, residual, delta, charge_scale, thermal_scale,
                            field=last, source_identity=source.identity,
                            probes=result['thermal_evaluation_order'][-4:].copy(), policy=thermal_policy))
                        save()
                    block = thermal_block.solve(evaluate_temperature,
                        (final, evaluation_energy[id(final)], candidates[id(final)]), full_ad,
                        policy=thermal_policy, max_evaluations=temperature_evaluations,
                        on_evaluation=thermal_decision, on_direction=thermal_direction,
                        replay=() if replay_source is None else replay_source['replay'], on_replay=replayed)
                    final = block.pop('selected')[0]
                    result['temperature_block'] = block
                    result['temperature_updated'] = not np.array_equal(final['temperature'], state['temperature'])
                    result['field_moments_scope'] = 'input accepted material; not recomputed for the temperature candidate'
                for ion in response.ions.values(): inputs.update(ion.identity)
                result.update(charge_converged=bool(np.max(abs(final['charge_residual']/final['ne'])) <= 1.e-10), charge_history=solved['history'],
                    selected_response=candidates[id(final)])
            summary, table = comparison(state['columns'], reference['superlevel'], state['superlevel'], final['population'], source.radius)
            with (out/'Si-S-comparison.csv').open('x') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
            result['ion_comparison'] = summary
            np.savez(out/'physical-profiles.npz', radius_cm=source.radius,
                CMFGEN70_temperature=reference['temperature'], CMFGEN70_ne=reference['ne'],
                accepted_temperature=state['temperature'], accepted_ne=state['ne'],
                new_response_temperature=final['temperature'], new_response_ne=final['ne'])
            result['physical_comparison'] = {
                name: dict(max_relative_difference=float(np.max(abs(final[key]/reference[key]-1))),
                           median_relative_difference=float(np.median(abs(final[key]/reference[key]-1))))
                for name, key in (('temperature_vs_CMFGEN70', 'temperature'), ('ne_vs_CMFGEN70', 'ne'))}
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
            result['output_sha256'] = {name: B.sha(out/name) for name in
                ('Si-S-comparison.csv', 'physical-profiles.npz', 'bound-rates.npy',
                 'route-roster.npz', 'jbar.npy', 'line-status.npy', 'FF-coefficients.npz')}
            result['state'] = ('COMPLETE_FINITE_NEW_FIELD_TNE_RESPONSE_NOT_JOINT_CONVERGENCE' if temperature_evaluations else
                               'COMPLETE_FINITE_NEW_FIELD_RESPONSE_NOT_JOINT_CONVERGENCE')
    except Exception as error:
        result.update(state='INCOMPLETE_PRESERVED', error=type(error).__name__+': '+str(error))
    finally:
        if cached_response is not None:
            result['temperature_bound_cache'] = cached_response.report()
        save()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('preparation', 'feedback', 'library', 'line-library', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('preparation-sha', 'feedback-sha', 'source-commit'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--seconds', type=int, default=6600)
    parser.add_argument('--temperature-evaluations', type=int, default=0,
        help='신규 실제평가 한도: 새 보정5~12, 보존평가 재개1~12. 실행 승인을 대신하지 않음')
    parser.add_argument('--resume-result', type=Path,
        help='같은 장·수용 물질의 봉인 완료 고정온도 보고서. 계수만 재사용하고 SE/열은 재평가')
    parser.add_argument('--resume-result-sha')
    parser.add_argument('--resume-thermal-result', type=Path,
        help='종료된 같은 장 열평가를 정확한 호출 순서로 재생. 추가 실행 승인을 대신하지 않음')
    parser.add_argument('--resume-thermal-result-sha')
    parser.add_argument('--thermal-policy', choices=thermal_replay.ENERGY_POLICIES,
        help='온도 보정 때 반드시 명시하는 열정책. 기존 matched-native 계약은 native_instantaneous_scenario')
    parser.add_argument('--thermal-replay-prefix', type=int,
        help='정책 변경 시 기존 방향 후보가 아닌 초기1~4개 물질 섭동만 재사용. 새 정책의 잔차/척도/방향을 재구성')
    result = run(parser.parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result['state'] in ('COMPLETE_FINITE_NEW_FIELD_RESPONSE_NOT_JOINT_CONVERGENCE',
        'COMPLETE_FINITE_NEW_FIELD_TNE_RESPONSE_NOT_JOINT_CONVERGENCE') else 1)
