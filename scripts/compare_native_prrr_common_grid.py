#!/usr/bin/env python3
"""원70/J70/원 주파수에서 Si/S II·III PRRR와 C bound/pseudo를 SL별 대조한다.

새 수송·물질 갱신은 없다. PRRR의 5유효숫자 인쇄 한계를 별도로 보존한다.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import resource
import time

import numpy as np

import cmfgen_model_continuum as B
import cmfgen_model_sources as S
import compare_lumina_checkpoint_rates as C
import lumina_native90_rates as R
import lumina_native_columns as N
from compare_cmfgen_bf_superlevel_rates import (parse_prrr_superlevel_rates,
                                                parse_prrr_temperature)
from run_cmfgen_formal_primary import fields


def continuum_mapping(path, frequency):
    """원 CONT_FREQ의 인쇄 정밀도를 보존한 실제ν→계수 평가ν 대응."""
    frequency = np.asarray(frequency, dtype='f8')
    if frequency.ndim != 1 or len(frequency) < 2 or np.any(np.diff(frequency) <= 0):
        raise ValueError('양의 증가하는 원 주파수 필요')
    rows = []
    with Path(path).open() as stream:
        # CONT_FREQ 뒤에는 별도의 continuum 요약표가 이어진다. 첫 원 격자 표만 읽는다.
        header = [next(stream).strip(), next(stream).strip(), next(stream).strip()]
        if (header[0] or not header[1].startswith('!NCF,') or
                int(header[1].split(',')[1]) != len(frequency) or
                not header[2].startswith('!')):
            raise ValueError('CONT_FREQ 원 격자 header 불일치')
        for _ in range(len(frequency)):
            line = next(stream)
            if not line.strip() or line.lstrip().startswith('!'):
                raise ValueError('CONT_FREQ 원 격자 중간 공백')
            fields = line.split()
            if len(fields) < 4 or int(fields[0]) != len(rows)+1:
                raise ValueError('CONT_FREQ 주소 누락/중복')
            rows.append((float(fields[1]), float(fields[2])))
    if len(rows) != len(frequency):
        raise ValueError('CONT_FREQ와 원 주파수 개수 불일치')
    actual, evaluation = np.asarray(rows, dtype='f8').T
    actual = actual[::-1]*1e15
    evaluation = evaluation[::-1]*1e15
    if (np.any(evaluation <= 0) or not np.isfinite(evaluation).all() or
            not np.allclose(actual, frequency, rtol=5.1e-8, atol=0)):
        raise ValueError('CONT_FREQ 실제 주파수와 원 장 주파수 불일치')
    # 인쇄값을 물리 주파수로 사용하지 않는다. 원 FREQ 배열의 정확한 주소로 복원한다.
    right = np.searchsorted(frequency, evaluation)
    left = np.maximum(right-1, 0)
    right = np.minimum(right, len(frequency)-1)
    mapped_index = np.where(abs(frequency[left]-evaluation) <=
                            abs(frequency[right]-evaluation), left, right)
    restored = frequency[mapped_index]
    if np.max(abs(restored/evaluation-1)) >= 5e-8:
        raise ValueError('CONT_FREQ 원 격자 주소 복원 실패')
    return restored


def printed_half_step(value):
    """원 PRRR ES 유효숫자5개의 최대 반올림 폭(0에는 상한을 주장하지 않음)."""
    if value <= 0:
        return 0.
    return .5 * 10.**(math.floor(math.log10(value))-4)


def cached_exponent_frequency(frequency, evaluation):
    """native의 내림차순 ν 순회에서 각 CONT_FREQ 묶음의 첫 실제 ν.

    USE_FIXED_J 경로의 COMPUTE_NEW_CROSS 조건부 EMHNUKT 갱신을
    진단용으로만 복제한다. 실제 물리 계산의 지수 규약을 바꾸지 않는다.
    """
    frequency, evaluation = map(lambda v: np.asarray(v, dtype='f8'),
                                (frequency, evaluation))
    if (frequency.ndim != 1 or not len(frequency) or
            evaluation.shape != frequency.shape or
            not np.isfinite(frequency).all() or not np.isfinite(evaluation).all() or
            np.any(frequency <= 0) or np.any(evaluation <= 0) or
            np.any(np.diff(frequency) <= 0)):
        raise ValueError('지수 cache의 원 주파수/대응 오류')
    ends = np.r_[np.flatnonzero(evaluation[:-1] != evaluation[1:]), len(frequency)-1]
    sizes = np.diff(np.r_[-1, ends])
    return np.repeat(frequency[ends], sizes)


def cached_exponent_rates(sampled, frequency, cached_frequency, temperature,
                          radiation, weights):
    """검사 전용: 올바른 실제ν 계수에 구형 고정장 지수의 영향만 분리한다."""
    factor = np.exp(-B.H*(cached_frequency-frequency)[None, :]/
                    (B.KB*np.asarray(temperature)[:, None]))
    old = dict(chi_abs=sampled['chi_abs'],
               chi_stim=sampled['chi_stim']*factor,
               eta_spont=sampled['eta_spont']*factor)
    return compact(C.contract_rates(old, radiation, weights))


def one_node(state, node):
    one = dict(state)
    for key in ('full', 'occupation', 'superlevel'):
        one[key] = state[key][:, node:node+1]
    for key in ('temperature', 'ne'):
        one[key] = state[key][node:node+1]
    return one


def compact(rate):
    return {key: float(rate['CMFGEN'][key][0]) for key in
            ('photo', 'spontaneous', 'stimulated', 'reverse')}


def check_prior(prior, Z, stage, node, bound, total):
    selected = [row for row in prior if
                row['mode'] == 'mapped-native-SE' and int(row['Z']) == Z and
                int(row['stage']) == stage and int(row['native_node']) == node]
    if len(selected) != 1:
        raise ValueError('기존 동일 조건 전체율 행 누락/중복')
    row = selected[0]
    for scope, values in (('bound', bound), ('total', total)):
        for process in ('photo', 'spontaneous', 'stimulated', 'reverse'):
            expected = float(row[f'{scope}_Jref_{process}'])
            actual = values[process]
            if not np.isclose(actual, expected, rtol=1e-8, atol=1e-7):
                raise ValueError(f'기존 전체율 재현 실패: {Z}:{stage} {scope}:{process} '
                                 f'{actual} vs {expected}')


def run(base_path, prrr_dir, continuum_path, prior_path, output, node):
    started = time.monotonic()
    base_path, prrr_dir, continuum_path, prior_path, output = map(
        Path, (base_path, prrr_dir, continuum_path, prior_path, output))
    if output.exists():
        raise FileExistsError('기존 결과 보존')
    base = json.loads(base_path.read_text())
    c = json.loads(Path(base['config']).read_text())
    plan = json.loads(Path(c['plan']).read_text())
    prior_report = json.loads((prior_path.parent/'rates.json').read_text())
    if (base['source_identity']['material_record'] != 70 or
            plan['reference_radiation_material_record'] != 70 or
            prior_report['material_record'] != 70 or
            prior_report['frequencies'] != 197291 or
            B.sha(continuum_path) != prior_report['CONT_FREQ_sha256'] or
            B.sha(prior_path) != prior_report['output_sha256']['rates.csv']):
        raise ValueError('원70/기존 전체율/CONT_FREQ 봉인 불일치')
    field_path = Path(plan['same_material_reference_field'])/'FORMAL_PRIMARY_FIELDS.bin'
    if B.sha(field_path) != plan['reference_field_sha256']:
        raise ValueError('원70 복사장 신원 변경')
    code = [Path(__file__), Path(B.__file__), Path(S.__file__), Path(C.__file__),
            Path(R.__file__), Path(N.__file__)]
    source_hashes = {str(path): B.sha(path) for path in code}
    native_inputs_path = prrr_dir.parent/'seq_logs/verbose-se-inputs.json'
    native_inputs = json.loads(native_inputs_path.read_text())
    native_executable = Path(native_inputs['executable']['path'])
    native_build = json.loads((native_executable.parent/'se-witness-build.json').read_text())
    native_source = native_executable.parent/'cmfgen_sub.f'
    for path in (native_source, native_executable):
        if B.sha(path) != native_build['output_sha256'].get(str(path)):
            raise ValueError('원 실행의 build 소스/바이너리 변경')
    if B.sha(native_executable) != native_inputs['executable']['sha256']:
        raise ValueError('PRRR 원 실행 바이너리 계보 불일치')
    legacy_branch = '''\t    IF(USE_FIXED_J)THEN
\t      IF(COMPUTE_NEW_CROSS)THEN
                T1=-HDKT*CONT_FREQ
                EMHNUKT_CONT(1:ND)=EXP(T1/T(1:ND))
                T1=-HDKT*FL
                EMHNUKT(1:ND)=EXP(T1/T(1:ND))
\t      END IF'''
    if native_source.read_text().count(legacy_branch) != 1:
        raise ValueError('진단 대상의 원 고정장 지수 cache 분기 불일치')
    source_hashes[str(native_source)] = B.sha(native_source)
    native_identity = dict(inputs=str(native_inputs_path), inputs_sha256=B.sha(native_inputs_path),
                           executable=str(native_executable), executable_sha256=B.sha(native_executable),
                           source=str(native_source), source_sha256=B.sha(native_source))
    with N.open_sealed_native(base['config'], base['preparation'], base['preparation_sha256']) as source:
        if source.identity != base['source_identity'] or not 0 <= node < source.ns:
            raise ValueError('원래점/원 물질 신원 불일치')
        geometry, field = fields(field_path)
        try:
            nu = np.ascontiguousarray(field['freq'][::-1]*1e15)
            j = np.ascontiguousarray(field['field'][::-1, 0, node][None, :])
            if (not np.array_equal(geometry[0]*1e10, source.radius) or
                    nu.shape != (197291,) or j.shape != (1, len(nu)) or
                    not np.isfinite(j).all() or np.any(j < 0)):
                raise ValueError('원 복사장 좌표/차원 오류')
            evaluation = continuum_mapping(continuum_path, nu)
            cached_frequency = cached_exponent_frequency(nu, evaluation)
            saved_frequency_path = prior_path.parent/'frequency.npy'
            saved_evaluation_path = prior_path.parent/'continuum-frequency.npy'
            if (B.sha(saved_frequency_path) != prior_report['output_sha256']['frequency.npy'] or
                    B.sha(saved_evaluation_path) !=
                    prior_report['output_sha256']['continuum-frequency.npy'] or
                    not np.array_equal(nu, np.load(saved_frequency_path, allow_pickle=False)) or
                    not np.array_equal(evaluation,
                                       np.load(saved_evaluation_path, allow_pickle=False))):
                raise ValueError('기존 전체율의 원 주파수/CONT_FREQ 정확한 주소 불일치')
            eval_unique, index = S.continuum_frequencies(nu, evaluation)
            report_path = Path(c['reuse_sources']['boundary_sources']['directory'])/'sources.json'
            report = json.loads(report_path.read_text())
            rows, models, routes, state, do_dis, full_check, smoothing = R.material_state(
                c, plan, source, report)
            if not do_dis:
                raise ValueError('원 준위용해가 켜진 물질만 비교')
            state = one_node(state, node)
            weights = C.photon_weights(nu)
            radiation = {'CMFGEN': j}
            with prior_path.open(newline='') as stream:
                prior = list(csv.DictReader(stream))
            result = {}
            for Z, stage, label in ((14, 2, 'Sk2'), (14, 3, 'SkIII'),
                                    (16, 2, 'S2'), (16, 3, 'SIII')):
                selected = [route for route in routes if (route.Z, route.stage) == (Z, stage)]
                nsl = int(models[Z, stage]['n_super'])
                grouped = [[] for _ in range(nsl)]
                for route in selected:
                    sl = int(rows[route.lower]['super_level'])
                    if not 0 <= sl < nsl:
                        raise ValueError('FL→SL 주소 범위 밖')
                    grouped[sl].append(route)
                if any(not group for group in grouped):
                    raise ValueError('빈 초준위 BF route')
                path = prrr_dir/f'{label}PRRR'
                native_photo = parse_prrr_superlevel_rates(path, label, 90, nsl)[:, node]
                native_reverse = parse_prrr_superlevel_rates(path, label, 90, nsl,
                                                              process='recombination')[:, node]
                native_t = parse_prrr_temperature(path, 90)[node]
                if not np.isclose(native_t, state['temperature'][0], rtol=5.1e-5, atol=0):
                    raise ValueError('PRRR와 원70 T가 다름')
                records = []
                for sl, group in enumerate(grouped):
                    by_grid = []
                    for grid in sorted({route.phot_grid for route in group}):
                        subset = [route for route in group if route.phot_grid == grid]
                        coefficients = B.continuum_grid(subset, rows, state, eval_unique, 512)
                        grid_bound = compact(C.contract_rates(C.mapped_rate_coefficients(
                            coefficients, nu, eval_unique, index, state['temperature']),
                            radiation, weights))
                        B.add_dissolved_continuum(coefficients, subset, rows, models, state,
                                                 eval_unique, support_policy='native-SE')
                        sampled = C.mapped_rate_coefficients(
                            coefficients, nu, eval_unique, index, state['temperature'])
                        grid_total = compact(C.contract_rates(sampled, radiation, weights))
                        cached = cached_exponent_rates(sampled, nu, cached_frequency,
                                                        state['temperature'], radiation, weights)
                        by_grid.append(dict(grid=grid, routes=len(subset), bound=grid_bound,
                                            total=grid_total, cached_exponent=cached))
                    bound = {name:float(sum(entry['bound'][name] for entry in by_grid))
                             for name in by_grid[0]['bound']}
                    total = {name:float(sum(entry['total'][name] for entry in by_grid))
                             for name in by_grid[0]['total']}
                    cached = {name:float(sum(entry['cached_exponent'][name] for entry in by_grid))
                              for name in by_grid[0]['total']}
                    records.append(dict(sl=sl, routes=len(group), bound=bound, total=total,
                        per_phot_grid=by_grid,
                        cached_exponent=cached,
                        pseudo={name:total[name]-bound[name] for name in bound},
                        native=dict(photo=float(native_photo[sl]),
                                    reverse=float(native_reverse[sl])),
                        native_print_half_step=dict(photo=printed_half_step(native_photo[sl]),
                                                    reverse=printed_half_step(native_reverse[sl]))))
                sums = {scope:{name:float(sum(row[scope][name] for row in records))
                               for name in ('photo','spontaneous','stimulated','reverse')}
                        for scope in ('bound','total','pseudo','cached_exponent')}
                check_prior(prior, Z, stage, node, sums['bound'], sums['total'])
                native = dict(photo=float(native_photo.sum()), reverse=float(native_reverse.sum()))
                precision = {name:float(sum(row['native_print_half_step'][name]
                                            for row in records)) for name in ('photo','reverse')}
                # QUAD_MULTI_V10의 1/5.27296e-3과 LTEPOP_WLD_V2의
                # 2.07078e-22를 현 물리상수로 바꿔치기하지 않고 분리 보존한다.
                photo_scale = (1e25*B.H/(4*np.pi))/5.27296e-3
                saha_t4 = .5*(B.H*B.H/(2*np.pi*B.ME*B.KB*1e4))**1.5
                reverse_scale = photo_scale*(2.07078e-22/saha_t4)
                old = dict(photo=sums['cached_exponent']['photo']*photo_scale,
                           reverse=sums['cached_exponent']['reverse']*reverse_scale)
                result[label] = dict(Z=Z, stage=stage, native_node=node, routes=len(selected),
                    frequencies=len(nu), continuum_unique=len(eval_unique), native=native,
                    native_print_half_step=precision, C=sums,
                    cached_exponent_with_native_constants=old,
                    native_constant_scales=dict(photo=photo_scale, reverse=reverse_scale),
                    native_minus_cached_native_constants={name:native[name]-old[name]
                                                          for name in ('photo','reverse')},
                    native_minus_C={name:native[name]-sums['total'][name]
                                    for name in ('photo','reverse')}, per_SL=records)
                print(json.dumps(dict(status='ION_COMPLETE', ion=label,
                                      elapsed_seconds=time.monotonic()-started)), flush=True)
            if any(B.sha(Path(path)) != digest for path, digest in source_hashes.items()):
                raise ValueError('계산 도중 비교 코드 변경')
            output.mkdir()
            report = dict(schema='lumina-native-prrr-common-grid-sl-v1', status='READ_ONLY_RATE_DIAGNOSIS',
                material_record=70, node=node, source_identity=source.identity,
                source_hashes=source_hashes, field_sha256=plan['reference_field_sha256'],
                continuum_sha256=prior_report['CONT_FREQ_sha256'],
                prior_rate_sha256=B.sha(prior_path), full_population=full_check,
                cached_exponent_diagnostic=dict(
                    physical_rule='actual frequency at every point',
                    native_fixed_J_rule='first descending frequency of each contiguous CONT_FREQ group',
                    affected_frequencies=int(np.count_nonzero(cached_frequency != nu)),
                    maximum_anchor_over_actual=float(np.max(cached_frequency/nu)),
                    production_rule_changed=False),
                legacy_native_identity=native_identity,
                results=result, elapsed_seconds=time.monotonic()-started,
                peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                physical_transport_runs=0, native_runs=0, material_updates=0,
                final_goal='NOT_MET')
            (output/'comparison.json').write_text(json.dumps(report, ensure_ascii=False,
                                                       indent=2, allow_nan=False)+'\n')
            return report
        finally:
            field._mmap.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('base','prrr-dir','continuum','prior','output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--node', type=int, default=49)
    a = p.parse_args()
    report = run(a.base, a.prrr_dir, a.continuum, a.prior, a.output, a.node)
    print(json.dumps(dict(status=report['status'], node=report['node'],
                          elapsed_seconds=report['elapsed_seconds']), ensure_ascii=False))
