#!/usr/bin/env python3
"""미수렴 저장 J 두 개의 Si/S BF 소비율을 같은 물질 참조와 비교한다.

생산 PHOT/Milne/준위 용해 함수를 재사용한다. 전 native 참조 주파수에서의
공통 사다리꼴 적분이며, 연속 적분의 오차 인증이나 SE/RE·이온분포 판정은 아니다.
수송·물질 갱신을 실행하지 않으며 기존 장과 입력을 변경하지 않는다.
"""
import argparse
import csv
import json
from pathlib import Path
import resource
import time

import numpy as np

import cmfgen_model_continuum as B
import cmfgen_model_sources as S
from compare_lumina_bridge_field0 import fields, matched_shells
from lumina_field_checkpoint import digest
from run_lumina_cell_candidate import read_material

TARGETS = ((14, 2), (14, 3), (16, 2), (16, 3))


def photon_weights(nu):
    """주파수 점값의 사다리꼴 가중치. 누락 구간이나 문턱을 보정하지 않는다."""
    nu = np.asarray(nu, dtype='f8')
    if (nu.ndim != 1 or len(nu) < 2 or not np.isfinite(nu).all() or
            np.any(nu <= 0) or np.any(np.diff(nu) <= 0)):
        raise ValueError('양의 증가하는 전 주파수 격자 필요')
    widths = np.diff(nu)
    measure = np.zeros_like(nu)
    measure[:-1] += .5*widths
    measure[1:] += .5*widths
    return measure*(4*np.pi/(B.H*nu))


def contract_rates(coefficients, radiation, weights):
    """흡수·유도·자발을 분리한다. 순χ로 흡수를 대체하지 않는다."""
    a, s, e = (coefficients[key] for key in ('chi_abs', 'chi_stim', 'eta_spont'))
    if (a.ndim != 2 or any(x.shape != a.shape for x in (s, e)) or
            weights.shape != (a.shape[1],) or not radiation):
        raise ValueError('BF 계수/장/가중치 차원 오류')
    for value in (a, s, e, weights, *radiation.values()):
        if not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError('비유한/음수 BF 소비 입력')
    if any(j.shape != a.shape for j in radiation.values()):
        raise ValueError('장과 같은 물질·주파수 차원 필요')
    spontaneous = e@weights
    result = {}
    for name, j in radiation.items():
        photo = np.einsum('ij,ij,j->i', a, j, weights)
        stimulated = np.einsum('ij,ij,j->i', s, j, weights)
        result[name] = dict(photo=photo, stimulated=stimulated,
                            spontaneous=spontaneous, reverse=stimulated+spontaneous)
        if not all(np.isfinite(v).all() for v in result[name].values()):
            raise ValueError('BF 적분 수치 범위 초과')
    return result


def mapped_rate_coefficients(coefficients,nu,evaluation,index,temperature):
    """CONT_FREQ의 흡수는 유지하고 유도항/진η만 실제ν로 옮긴다.

    순χ로 광흡수를 대체하지 않는다. 입력은 재사용 전의 원 계수이며
    이미 보정된 source binary를 다시 보정하는 경로가 아니다.
    """
    if index is None:return coefficients
    # identity에서 fancy indexing의 배열 stride까지 바꾸면 BLAS 합산이
    # 마지막 bit에서 달라질 수 있다. 원 계수를 그대로 소비한다.
    if np.array_equal(evaluation,nu) and np.array_equal(index,np.arange(len(nu))):
        return coefficients
    mapped={key:value[:,index] for key,value in coefficients.items()}
    evaluation=evaluation[index]
    _,eta=S.rescale_continuum(nu,evaluation,temperature,mapped['chi_net'],mapped['eta_spont'])
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        stimulated=mapped['chi_stim']*np.exp(
            -B.H*(nu-evaluation)[None,:]/(B.KB*np.asarray(temperature)[:,None]))
    if not np.isfinite(stimulated).all():raise ValueError('CONT_FREQ 유도항 비유한')
    return dict(chi_abs=mapped['chi_abs'],chi_stim=stimulated,
                chi_net=mapped['chi_abs']-stimulated,eta_spont=eta)


def rate_sampling(nu,evaluation,support_policy):
    if support_policy not in ('opacity','native-SE'):
        raise ValueError('알 수 없는 준위 용해 소비 규약')
    return S.continuum_frequencies(nu,evaluation)


def ion_rates(nu, radiation, routes, rows, models, state, do_dis, targets=TARGETS,
              block_bins=512,continuum_evaluation_Hz=None,dissolution_policy='opacity'):
    """각 실제 이온의 모든 route를 합한다. 작은 인구·단면을 자르지 않는다."""
    weights = photon_weights(nu)
    evaluation,index=rate_sampling(nu,continuum_evaluation_Hz,dissolution_policy)
    for key in targets:
        selected = [r for r in routes if (r.Z, r.stage) == key]
        if not selected:
            raise ValueError(f'요청 이온의 실제 PHOT route 누락: {key}')
        coefficients = B.continuum_grid(selected, rows, state, evaluation, block_bins)
        sampled=mapped_rate_coefficients(coefficients,nu,evaluation,index,state['temperature'])
        bound = contract_rates(sampled, radiation, weights)
        del sampled
        if do_dis:
            B.add_dissolved_continuum(coefficients, selected, rows, models, state, evaluation,
                                     support_policy=dissolution_policy)
            sampled=mapped_rate_coefficients(coefficients,nu,evaluation,index,state['temperature'])
            total = contract_rates(sampled, radiation, weights)
            del sampled
        else:
            total = bound
        yield key, dict(routes=len(selected), bound=bound, total=total)
        # 소비자가 다음 이온을 요청할 때 이전의 대형 계수를 먼저 해제한다.
        del coefficients, bound, total


def ion_rates_stream(nu, field_block, routes, rows, models, state, do_dis,
                     targets=TARGETS, integration_block_bins=8192, block_report=None,
                     continuum_evaluation_Hz=None,dissolution_policy='opacity'):
    """생산 격자 점을 버리지 않고 같은 점값 적분을 제한된 메모리로 합한다.

    선택 block_report는 이미 합산한 블록의 과정별 기여만 관측한다.
    전역 사다리꼴 가중치나 계수 계산은 이 선택으로 바꾸지 않는다.
    """
    weights = photon_weights(nu)
    # 전체 대응을 먼저 검증해 뒤 블록의 손상이 부분 율을 만들어 내지 않게 한다.
    rate_sampling(nu,continuum_evaluation_Hz,dissolution_policy)
    if type(integration_block_bins) is not int or integration_block_bins < 1:
        raise ValueError('양의 적분 블록 크기 필요')
    for key in targets:
        selected = [r for r in routes if (r.Z, r.stage) == key]
        if not selected:
            raise ValueError(f'요청 이온의 실제 PHOT route 누락: {key}')
        accumulated = {}
        for begin in range(0, len(nu), integration_block_bins):
            sl = slice(begin, min(len(nu), begin+integration_block_bins))
            radiation = field_block(sl)
            evaluation,index=rate_sampling(nu[sl],
                None if continuum_evaluation_Hz is None else continuum_evaluation_Hz[sl],dissolution_policy)
            coefficients = B.continuum_grid(selected, rows, state, evaluation, 512)
            sampled=mapped_rate_coefficients(coefficients,nu[sl],evaluation,index,state['temperature'])
            bound = contract_rates(sampled, radiation, weights[sl])
            del sampled
            if do_dis:
                B.add_dissolved_continuum(coefficients, selected, rows, models, state, evaluation,
                                         support_policy=dissolution_policy)
                sampled=mapped_rate_coefficients(coefficients,nu[sl],evaluation,index,state['temperature'])
                total = contract_rates(sampled, radiation, weights[sl])
                del sampled
            else:
                total = bound
            for scope, rates in (('bound', bound), ('total', total)):
                if scope not in accumulated:
                    accumulated[scope] = {name: {p: v.copy() for p, v in values.items()}
                                          for name, values in rates.items()}
                else:
                    if set(rates) != set(accumulated[scope]):
                        raise ValueError('블록 사이 복사장 목록 변경')
                    for name, values in rates.items():
                        for process, value in values.items():
                            accumulated[scope][name][process] += value
            if block_report is not None:
                block_report(key,sl,bound,total)
            del coefficients, radiation, bound, total
        if not all(np.isfinite(v).all() for rates in accumulated.values()
                   for values in rates.values() for v in values.values()):
            raise ValueError('누적 BF 적분 수치 범위 초과')
        yield key, dict(routes=len(selected), **accumulated)


def production_frequency(own_nu, reference_nu):
    """전 참조 구간 안의 모든 생산 점과 양 끝점을 유지한다. 외삽은 없다."""
    photon_weights(own_nu)
    photon_weights(reference_nu)
    if reference_nu[0] < own_nu[0] or reference_nu[-1] > own_nu[-1]:
        raise ValueError('참조 전 구간을 덮는 생산 격자 필요')
    inside = own_nu[(own_nu >= reference_nu[0]) & (own_nu <= reference_nu[-1])]
    return np.unique(np.r_[reference_nu[0], inside, reference_nu[-1]])


def production_rate_iterator(nu, reference_nu, reference_j, own_nu, material,
                             checkpoint, entries, routes, rows, models, state, do_dis):
    """자체 미세장을 내려 뭉개지 않고 참조만 같은 logν 규칙으로 올린다."""
    mapped = {}
    try:
        for name, entry in entries.items():
            mapped[name] = _entry(entry, (material['ns']+2, len(own_nu)), checkpoint)
        log_own, log_reference = np.log(own_nu), np.log(reference_nu)

        def block(sl):
            log_frequency = np.log(nu[sl])
            fields = {'CMFGEN': np.stack([np.interp(log_frequency, log_reference, row)
                                          for row in reference_j])}
            for name, j in mapped.items():
                fields[name] = np.stack([np.interp(log_frequency, log_own, row)
                                          for row in j[:material['ns']]])
            return fields

        yield from ion_rates_stream(nu, block, routes, rows, models, state, do_dis)
    finally:
        for j in mapped.values():
            j._mmap.close()


def _entry(entry, shape, folder):
    n = entry['iterations']
    if type(n) is not int or n < 1 or entry['filename'] != f'J-{n:06d}.npy':
        raise ValueError('초기장/저장 장 파일 신원 오류')
    path = folder/entry['filename']
    if B.sha(path) != entry['sha256']:
        raise ValueError('저장 장 변경')
    value = np.load(path, mmap_mode='r', allow_pickle=False)
    if value.dtype != np.dtype('f8') or value.shape != tuple(shape):
        value._mmap.close()
        raise ValueError('저장 장 차원/형식 오류')
    return value


def projected_fields(material, own_nu, checkpoint, entries, reference):
    """기존 비교기의 전 nativeν/반경 중심 규칙을 그대로 재사용한다."""
    geometry, records = fields(reference)
    projected = {}
    nu = None
    try:
        shape = (material['ns']+2, len(own_nu))
        for name, entry in entries.items():
            j = _entry(entry, shape, checkpoint)
            try:
                candidate = np.empty((material['ns'], len(records)))
                if 'CMFGEN' not in projected:
                    projected['CMFGEN'] = np.empty_like(candidate)
                for shell, target, own, native in matched_shells(
                        geometry, records, material, own_nu, j[:material['ns']]):
                    candidate[shell] = own
                    projected['CMFGEN'][shell] = native
                    if nu is None:
                        nu = target.copy()
                projected[name] = candidate
            finally:
                j._mmap.close()
    finally:
        records._mmap.close()
    if nu is None:
        raise ValueError('비교할 저장 장 없음')
    return nu, projected


def _ratio(a, b):
    # 0/0은 정합 비율을 정의하지 않는다. 바닥값으로 정상 비율을 만들지 않는다.
    return float(a/b) if b > 0 else ('UNDEFINED_0_OVER_0' if a == 0 else 'Infinity')


def compare(config, checkpoint, previous_diagnostic, current_diagnostic, out, integration_grid='reference'):
    config, checkpoint, out = Path(config).resolve(), Path(checkpoint).resolve(), Path(out)
    if out.exists():
        raise ValueError('기존 율 비교 산출물 보존')
    if integration_grid not in ('reference', 'production'):
        raise ValueError('명시적인 율 적분 격자 필요')
    started = time.monotonic()
    captured = {}

    def capture(path):
        path = Path(path).resolve()
        raw = path.read_bytes()
        captured[path] = (raw, B.sha(path))
        return json.loads(raw)

    c = capture(config)
    plan_path = Path(c['plan'])
    plan = capture(plan_path)
    seed_path = Path(c['seed'])/'material-seed.json'
    seed = capture(seed_path)
    state_path = checkpoint/'checkpoint.json'
    saved = capture(state_path)
    diagnostics = [capture(previous_diagnostic), capture(current_diagnostic)]
    unsigned = {k: v for k, v in saved.items() if k != 'metadata_sha256'}
    if (saved['schema'] != 'lumina-outer-nonlinear-checkpoint-v1' or
            digest(unsigned) != saved['metadata_sha256'] or saved['phase'] == 'APPLY_PENDING' or
            saved['accepted_physical_field'] is not False or saved['componentwise_verified'] is not False):
        raise ValueError('정지한 미검증 외부대기 저장 장 필요')
    if (captured[plan_path][1] != c['plan_sha256'] or seed['plan_sha256'] != c['plan_sha256'] or
            seed['source_record'] != plan['initial_native_record'] or
            seed['source_record'] != plan['reference_radiation_material_record'] or
            seed['record_storage_sha256'] != plan['initial_record_storage_sha256']):
        raise ValueError('봉인 물질/복사장 세대 불일치')
    prior, current = saved['previous_residual']['input_J'], saved['J']
    if (saved['previous_residual']['output_J'] != current or
            prior['iterations']+1 != current['iterations'] or
            current['iterations'] != saved['completed_iterations']):
        raise ValueError('연속된 저장 장 쌍 필요')
    for d, selection, entry in zip(diagnostics, ('previous', 'current'), (prior, current), strict=True):
        if (d['schema'] != 'lumina-outer-checkpoint-diagnostic-v1' or
                d['status'] != 'UNVERIFIED_CHECKPOINT_DIAGNOSTIC' or
                d['accepted_physical_field'] is not False or d['componentwise_verified'] is not False or
                d['selection'] != selection or d['comparison_J'] != entry or d['current_J'] != current or
                d['material_record'] != seed['source_record'] or
                d['reference_record'] != plan['reference_radiation_record'] or
                d['radiation_sha256']['cmfgen'] != plan['reference_field_sha256']):
            raise ValueError('기존 장 비교 증거와 입력 불일치')
        for path in (config, plan_path, seed_path, state_path):
            if d['captured_sha256'][str(path)] != captured[path][1]:
                raise ValueError('비교 이후 입력/저장 상태 변경')
    source_path = Path(c['reuse_sources']['cell_sources']['directory'])/'sources.json'
    source = capture(source_path)
    if captured[source_path][1] != c['reuse_sources']['cell_sources']['report_sha256']:
        raise ValueError('생산 계수 기록 신원 불일치')
    expected = source['input_sha256']
    material_path = Path(c['cells'])/'material_cells.bin'
    run = Path(plan['initial_material'])
    atomic = Path(c['cells'])/'atomic'
    fixed = [material_path, atomic/'levels.csv', Path(c['full_population']), Path(c['phot_library']),
             Path(c['grid']), *[run/n for n in ('MODEL', 'MODEL_SPEC', 'LEVEL_SL_STEQ_LINKS', 'atomic_links.txt')]]
    dependencies = ('cmfgen_model_continuum.py', 'cmfgen_parser.py',
                    'expand_atomic_data_cmfgen.py', 'cmfgen_phot_smoothing.py')
    for name in dependencies:
        identities = [v for p, v in expected.items() if Path(p).name == name]
        if len(identities) != 1 or B.sha(Path(__file__).with_name(name)) != identities[0]:
            raise ValueError(f'생산 BF 소비 코드 신원 불일치: {name}')
    identities = {str(p): B.sha(p) for p in fixed}
    if any(identities[str(p)] != expected[str(p)] for p in fixed):
        raise ValueError('생산 물질/원자/격자 입력 변경')
    reference = Path(plan['same_material_reference_field'])/'FORMAL_PRIMARY_FIELDS.bin'
    if B.sha(reference) != plan['reference_field_sha256']:
        raise ValueError('봉인 참조장 변경')
    own_nu = np.load(c['grid'], allow_pickle=False)
    material = read_material(material_path, b'LUMCEL01')
    if saved['contract']['shape'] != [material['ns']+2, len(own_nu)]:
        raise ValueError('원래 물질/보조점/주파수 차원 불일치')
    names = {f'J{prior["iterations"]}': prior, f'J{current["iterations"]}': current}
    nu, radiation = projected_fields(material, own_nu, checkpoint, names, reference)
    rows, models, routes, policy, smoothing, counts = B.load_model(run, atomic, Path(c['phot_library']))
    if policy != source['process_selection'] or counts != source['routes_by_ion']:
        raise ValueError('생산 PHOT 과정/route 구성 변경')
    matter = B.read_material(material_path, rows, models)
    do_dis = B.dissolution_flag(run/'MODEL')
    if do_dis:
        occupation = B.material_occupation(rows, models, matter)
        matter = B.read_material(material_path, rows, models, occupation)
    full_check = B.consume_c_full_population(Path(c['full_population']), matter)
    if integration_grid == 'production':
        reference_nu, reference_j = nu, radiation['CMFGEN']
        del radiation
        nu = production_frequency(own_nu, reference_nu)
        iterator = production_rate_iterator(nu, reference_nu, reference_j, own_nu, material,
            checkpoint, names, routes, rows, models, matter, do_dis)
    else:
        iterator = ion_rates(nu, radiation, routes, rows, models, matter, do_dis)
    out.mkdir()
    print(json.dumps(dict(status='RATE_INPUTS_READY', cells=material['ns'], frequencies=len(nu),
                         elapsed_seconds=time.monotonic()-started)), flush=True)
    output_rows = []
    for (Z, stage), result in iterator:
        members = [l for l, row in enumerate(rows) if
                   (int(row['atomic_number']), int(row['ion_number'])+1) == (Z, stage)]
        population = matter['full'][members].sum(axis=0)
        for shell in range(material['ns']):
            row = dict(Z=Z, stage=stage, shell=shell, routes=result['routes'],
                       radius_cm=float(.5*sum(material['physical'][shell, :2])),
                       lower_ion_population_cm3=float(population[shell]))
            for name, rates in result['total'].items():
                for process, values in rates.items():
                    row[f'{name}_{process}_cm3_s'] = float(values[shell])
                row[f'{name}_effective_photo_s'] = _ratio(rates['photo'][shell], population[shell])
                for process in ('photo', 'stimulated', 'spontaneous'):
                    row[f'{name}_bound_{process}_cm3_s'] = float(result['bound'][name][process][shell])
                for process in ('photo', 'reverse'):
                    row[f'{name}_{process}_over_CMFGEN'] = _ratio(
                        rates[process][shell], result['total']['CMFGEN'][process][shell])
            output_rows.append(row)
        print(json.dumps(dict(status='ION_RATES_COMPLETED', Z=Z, stage=stage,
                              elapsed_seconds=time.monotonic()-started)), flush=True)
    if any(path.read_bytes() != raw for path, (raw, _) in captured.items()):
        raise ValueError('율 소비 도중 저장 상태/구성 변경')
    if any(B.sha(Path(p)) != value for p, value in identities.items()):
        raise ValueError('율 소비 도중 고정 물질/원자 입력 변경')
    for entry in names.values():
        if B.sha(checkpoint/entry['filename']) != entry['sha256']:
            raise ValueError('율 소비 도중 저장 장 변경')
    with (out/'rates.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader(); writer.writerows(output_rows)
    result = dict(schema='lumina-checkpoint-bf-rates-v1', status='UNVERIFIED_FIELD_RATE_COMPARISON',
        material_record=seed['source_record'], reference_record=plan['reference_radiation_record'],
        cells=material['ns'], frequencies=len(nu), targets=list(TARGETS), fields=names,
        frequency_domain_Hz=[float(nu[0]), float(nu[-1])],
        integration_grid=integration_grid,
        integration='all selected-grid point samples; composite trapezoid of coefficient*J/nu; no mask/floor/normalization',
        grid_definition=('all production points inside full native reference band plus exact endpoints'
                         if integration_grid == 'production' else 'all native reference points'),
        field_frequency_rule='linear J in log(nu); own fine field retained on production integration grid',
        rate_density_unit='cm^-3 s^-1',
        domain_limitation='only full native reference band; no extrapolation and no assertion about outside-band integral',
        quadrature_error='NOT_CERTIFIED; point samples do not resolve every PHOT edge/resonance',
        spatial_rule='same arithmetic radial midpoints and linear-in-r reference as field comparison',
        rate_meaning='fixed C full populations; per-ion summed rate density; effective photo per lower-ion population is not a single-level Gamma',
        physics='production raw PHOT + target occupation + Milne induced/spontaneous + MODEL dissolution; induced counted once',
        do_lev_dissolution=do_dis, C_full_population_comparison=full_check,
        captured_sha256={str(p): identity for p, (_, identity) in captured.items()},
        input_sha256=identities, reference_sha256=plan['reference_field_sha256'],
        source_sha256={str(Path(__file__)): B.sha(Path(__file__)),
                       **{name: B.sha(Path(__file__).with_name(name)) for name in dependencies}},
        rates_csv_sha256=B.sha(out/'rates.csv'), phot_preprocessing=smoothing,
        elapsed_seconds=time.monotonic()-started, peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        physical_RT_executions=0, native_executions=0, matter_updates=0, SE_RE_solved=False,
        accepted_physical_field=False, componentwise_verified=False,
        physical_equivalence='NOT_JUDGED', final_goal='NOT_JUDGED')
    with (out/'rates.json').open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'checkpoint', 'previous-diagnostic', 'current-diagnostic', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--integration-grid', choices=('reference', 'production'), default='reference')
    args = parser.parse_args()
    print(json.dumps(compare(args.config, args.checkpoint, args.previous_diagnostic,
                             args.current_diagnostic, args.out, args.integration_grid), ensure_ascii=False))
