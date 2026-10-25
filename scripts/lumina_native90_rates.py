"""native90 자체장을 같은 원래점 물질의 생산 BF 소비율에 연결한다.

89셀/보조점 변환은 없다. 원자물리·용해·사다리꼴 함수는 기존 생산 함수를
재사용한다. 고정 물질의 율 비교이지 SE/RE·이온분포·스펙트럼 합격이 아니다.
"""
import csv
import json
from pathlib import Path
import resource
import time

import numpy as np

import cmfgen_model_continuum as B
from compare_lumina_checkpoint_rates import TARGETS, ion_rates_stream, production_frequency, _ratio
from run_cmfgen_formal_primary import fields as reference_fields


def material_state(c, plan, source, report):
    """수송 전에 실제90점 FL 소비가 가능한지 같은 생산 함수로 확인한다."""
    run, atomic = Path(plan['initial_material']), Path(c['cells'])/'atomic'
    material, population = Path(c['seed'])/'native_material_nodes.bin', Path(c['native_full_population'])
    for path in (material, population, atomic/'levels.csv', Path(c['phot_library'])):
        if B.sha(path) != report['input_sha256'].get(str(path)):
            raise ValueError('native90 물질/인구 사전 신원 오류')
    rows, models, routes, policy, smoothing, counts = B.load_model(run, atomic, Path(c['phot_library']))
    if policy != report['process_selection'] or counts != report['routes_by_ion']:
        raise ValueError('기존 생산 PHOT 과정/route 변경')
    state = B.read_material(material, rows, models)
    do_dis = B.dissolution_flag(run/'MODEL')
    if do_dis: state = B.read_material(material, rows, models, B.material_occupation(rows, models, state))
    if state['epoch_seconds'] != source.identity['epoch_seconds'] or not np.array_equal(state['physical'][:, 0], source.radius):
        raise ValueError('율 물질의 원래점/epoch 불일치')
    full_check = B.consume_c_full_population(population, state)
    return rows, models, routes, state, do_dis, full_check, smoothing


def radiation_block(frequency, reference_nu, reference_j, own_nu, own_fields, *, reference_log=None, own_log=None):
    """같은 원래점에서만 logν 보간한다. 정확한 생산 점은 그대로 읽는다."""
    frequency = np.asarray(frequency)
    if frequency[0] < reference_nu[0] or frequency[-1] > reference_nu[-1]:
        raise ValueError('전 참조 주파수 범위 밖 율 대조')
    if frequency[0] < own_nu[0] or frequency[-1] > own_nu[-1]:
        raise ValueError('자체 주파수 범위 밖 율 대조')
    logs = np.log(frequency)
    if reference_log is None: reference_log=np.log(reference_nu)
    result = {'CMFGEN': np.stack([np.interp(logs, reference_log, row) for row in reference_j])}
    indices = np.searchsorted(own_nu, frequency)
    exact = bool(np.all(indices < len(own_nu)) and np.array_equal(own_nu[indices], frequency))
    if not exact and own_log is None: own_log=np.log(own_nu)
    for name, values in own_fields.items():
        if values.shape[0] != reference_j.shape[0] or values.shape[1] != len(own_nu):
            raise ValueError('자체장/참조의 원래점 차원 불일치')
        result[name] = np.asarray(values[:, indices]) if exact else np.stack([
            np.interp(logs, own_log, row) for row in values])
    return result


def compare(config, source, contract, completed, out, deadline):
    out, config = Path(out), Path(config)
    if out.exists(): raise ValueError('기존 native90 율 결과 보존')
    if not completed or len(completed) > 2: raise ValueError('완료된 native90 장1개 또는2개 필요')
    if (source.ns, source.nf) != (90, 1115020) or contract['source_identity'] != source.identity:
        raise ValueError('원래점90 수송/율 물질 신원 불일치')
    def check():
        if time.monotonic() >= deadline: raise TimeoutError('전체 native90 묶음 시간 한도')
    check(); started = time.monotonic()
    c = json.loads(config.read_text()); plan = json.loads(Path(c['plan']).read_text())
    seed = json.loads((Path(c['seed'])/'material-seed.json').read_text())
    report_path = Path(c['reuse_sources']['boundary_sources']['directory'])/'sources.json'
    if B.sha(report_path) != c['reuse_sources']['boundary_sources']['report_sha256']:
        raise ValueError('생산 native 계수 보고서 변경')
    report = json.loads(report_path.read_text()); expected = report['input_sha256']
    material = Path(c['seed'])/'native_material_nodes.bin'; full_population = Path(c['native_full_population'])
    run, atomic = Path(plan['initial_material']), Path(c['cells'])/'atomic'
    fixed = [config, Path(c['plan']), Path(c['seed'])/'material-seed.json', report_path,
             material, full_population, atomic/'levels.csv', Path(c['phot_library']),
             *[run/name for name in ('MODEL', 'MODEL_SPEC', 'LEVEL_SL_STEQ_LINKS', 'atomic_links.txt')]]
    identities = {str(p): B.sha(p) for p in fixed}
    for path in fixed[4:]:
        if identities[str(path)] != expected.get(str(path)):
            raise ValueError('생산 native 물질/인구/원자 입력 변경: '+str(path))
    if (identities[str(config)] != contract['config_sha256'] or
            identities[str(Path(c['plan']))] != c['plan_sha256'] or
            source.identity['material_record'] != seed['source_record'] or
            seed['source_record'] != plan['reference_radiation_material_record']):
        raise ValueError('native 수송/율/참조의 세대 혼합')
    for name in ('cmfgen_model_continuum.py', 'cmfgen_parser.py', 'expand_atomic_data_cmfgen.py', 'cmfgen_phot_smoothing.py'):
        candidates = [v for p, v in expected.items() if Path(p).name == name]
        if len(candidates) != 1 or B.sha(Path(__file__).with_name(name)) != candidates[0]:
            raise ValueError('생산 BF 코드 신원 변경: '+name)
    geometry, records = reference_fields(Path(plan['same_material_reference_field'])/'FORMAL_PRIMARY_FIELDS.bin')
    mapped = {}
    try:
        if not np.array_equal(geometry[0]*1e10, source.radius):
            raise ValueError('CMFGEN/native 자체장의 원래점 반경 불일치')
        reference_nu = np.asarray(records['freq'][::-1])*1e15
        reference_j = np.ascontiguousarray(records['field'][::-1, 0, :].T)
        for index, entry in enumerate(completed, 1):
            if (entry['iteration'] != index or entry['input_iteration'] != index-1 or
                    entry['report']['completed_actual_F_calls'] != 1 or
                    entry['report']['physical_intervals'] != contract['plan_layout']['physical_intervals']):
                raise ValueError('미완료 장 또는 연속되지 않은 native 장 계보')
            path = Path(entry['J']['path'])
            if B.sha(path) != entry['J']['sha256']: raise ValueError('완성 native J 변경')
            values = np.load(path, mmap_mode='r', allow_pickle=False)
            if values.dtype != np.dtype('f8') or values.shape != (90, len(source.frequency)):
                values._mmap.close(); raise ValueError('완성 native J 형상 오류')
            mapped[f'J{index}'] = values
        rows, models, routes, state, do_dis, full_check, smoothing = material_state(c, plan, source, report)
        frequency = production_frequency(source.frequency, reference_nu)
        reference_log, own_log=np.log(reference_nu),np.log(source.frequency)
        def block(sl):
            check()
            return radiation_block(frequency[sl], reference_nu, reference_j, source.frequency, mapped,
                                   reference_log=reference_log,own_log=own_log)
        output_rows = []
        for (Z, stage), result in ion_rates_stream(frequency, block, routes, rows, models, state, do_dis):
            members = [i for i, row in enumerate(rows) if (int(row['atomic_number']), int(row['ion_number'])+1) == (Z, stage)]
            population = state['full'][members].sum(axis=0)
            for node in range(90):
                row = dict(Z=Z, stage=stage, native_node=node, radius_cm=float(source.radius[node]),
                           lower_ion_population_cm3=float(population[node]), routes=result['routes'])
                for name, rates in result['total'].items():
                    for process, value in rates.items(): row[f'{name}_{process}_cm3_s'] = float(value[node])
                    row[f'{name}_effective_photo_s'] = _ratio(rates['photo'][node], population[node])
                    for process in ('photo', 'reverse'):
                        row[f'{name}_{process}_over_CMFGEN'] = _ratio(rates[process][node], result['total']['CMFGEN'][process][node])
                    for process in ('photo', 'stimulated', 'spontaneous'):
                        row[f'{name}_bound_{process}_cm3_s'] = float(result['bound'][name][process][node])
                if len(completed) == 2:
                    row['photo_J2_over_J1'] = _ratio(result['total']['J2']['photo'][node], result['total']['J1']['photo'][node])
                output_rows.append(row)
            print(json.dumps(dict(status='NATIVE_ION_RATES_COMPLETED', Z=Z, stage=stage,
                                  elapsed_seconds=time.monotonic()-started)), flush=True)
        check()
        if any(B.sha(Path(p)) != value for p, value in identities.items()): raise ValueError('율 소비 도중 고정 입력 변경')
        for entry in completed:
            if B.sha(Path(entry['J']['path'])) != entry['J']['sha256']: raise ValueError('율 소비 도중 자체장 변경')
        out.mkdir()
        with (out/'rates.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(output_rows[0])); writer.writeheader(); writer.writerows(output_rows)
        result = dict(schema='lumina-native90-bf-rates-v1', status='UNVERIFIED_NATIVE_FIELD_RATE_COMPARISON',
            material_record=70, reference_radiation_record=plan['reference_radiation_record'], native_nodes=90,
            frequencies=len(frequency), fields=[f'J{i}' for i in range(1, len(completed)+1)], targets=list(TARGETS),
            spatial_rule='same native outer-to-inner point values; no cells, no auxiliary nodes, no radial averaging',
            integration='production points inside full reference band plus exact endpoints; unchanged composite trapezoid',
            field_frequency_rule='linear J in log(nu); exact own-grid values read without projection',
            physical_scope='fixed native90 full C populations; per-ion BF rate densities, not SE/RE or spectrum acceptance',
            rate_density_unit='cm^-3 s^-1', effective_rate_unit='s^-1',
            full_population=full_check, do_lev_dissolution=do_dis, phot_preprocessing=smoothing,
            input_sha256=identities, reference_sha256=plan['reference_field_sha256'],
            reference_hash_verification='INHERITED_SEALED_PLAN_NOT_REHASHED',
            rates_csv_sha256=B.sha(out/'rates.csv'), elapsed_seconds=time.monotonic()-started,
            peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            physical_RT_executions=0, SE_RE_solved=False, accepted_physical_field=False, final_goal='NOT_MET')
        with (out/'rates.json').open('x') as stream: json.dump(result, stream, indent=2, allow_nan=False)
        return result
    finally:
        for value in mapped.values(): value._mmap.close()
        records._mmap.close()
