#!/usr/bin/env python3
"""완료된 자체장0과 같은 물질의 CMFGEN J를 비교한다.

물질의 부피평균과 복사장의 평가 위치는 다르다. 자체 J는 산술 반경 중심값이며
참조 J는 인접 원래 점을 반경에 대해 선형 보간한다. 주파수는 참조 전 격자를
유지하고 자체 J를 log(nu)에 대해 선형 보간한다. 마스크/바닥값/정규화는 없다.
이 비교는 고정물질 개발 비교이며 최종 공동수렴·스펙트럼 판정이 아니다.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from run_cmfgen_formal_primary import fields
from run_lumina_cell_candidate import read_material, sha, identity, REL_FRAME


def finite_number(value):
    return float(value) if np.isfinite(value) else 'Infinity'


def norm_ratio(error_max, j_max):
    return finite_number(error_max/j_max) if j_max > 0 else (0. if error_max == 0 else 'Infinity')


def iteration_error_maxima(error, shape):
    """기존 전성분 상한의 크기만 읽는다. 보간·물리 오차 상한으로 확대하지 않는다."""
    if error.shape != shape:
        raise ValueError('자체장 오차상한 차원 불일치')
    maxima = np.zeros(shape[0])
    for shell, row in enumerate(error):
        for begin in range(0, len(row), 131072):
            block = row[begin:begin+131072]
            if not np.isfinite(block).all() or np.any(block < 0):
                raise ValueError('자체장 오차상한 비유한/음수')
            maxima[shell] = max(maxima[shell], float(block.max()))
    return maxima


def statistics(candidate, reference):
    """0/0은 차이0, 양수/0은 무한 상대차로 전 표본에 포함한다."""
    if candidate.shape != reference.shape or not candidate.size:
        raise ValueError('비교 배열 크기 불일치')
    for a in (candidate, reference):
        if not np.isfinite(a).all() or np.any(a < 0):
            raise ValueError('비유한/음수 J는 비교 불가')
    absolute = abs(candidate-reference)
    relative = np.zeros_like(absolute)
    np.divide(absolute, reference, out=relative, where=reference > 0)
    relative[(reference == 0) & (candidate > 0)] = np.inf
    quantiles = np.quantile(relative, [.5, .95, 1], method='nearest')
    result = dict(samples=int(reference.size), zero_zero=int(np.count_nonzero(
        (reference == 0) & (candidate == 0))), positive_over_zero=int(np.count_nonzero(
        (reference == 0) & (candidate > 0))),
        absolute_median=float(np.median(absolute)), absolute_max=float(absolute.max()),
        relative_median=finite_number(quantiles[0]), relative_p95=finite_number(quantiles[1]),
        relative_max=finite_number(quantiles[2]))
    return result, relative


def matched_shells(geometry, records, material, nu, j):
    """전 native 주파수의 중심 J 두 배열을 셀 하나씩 산출한다."""
    radius = geometry[0]*1e10
    physical = material['physical']
    if (material['ns'] != len(radius)-1 or
        not np.array_equal(physical[:, 0], radius[1:][::-1]) or
        not np.array_equal(physical[:, 1], radius[:-1][::-1])):
        raise ValueError('참조 점과 실제 셀 경계 불일치')
    target = np.asarray(records['freq'][::-1])*1e15
    if (nu.ndim != 1 or len(nu) < 2 or not np.isfinite(nu).all() or
        np.any(nu <= 0) or not np.all(np.diff(nu) > 0) or
        j.shape != (material['ns'], len(nu))):
        raise ValueError('자체장 주파수/셀 차원 오류')
    if target[0] < nu[0] or target[-1] > nu[-1]:
        raise ValueError('전체 참조 범위를 덮지 못함; 끝값 연장 금지')
    lognu, logtarget = np.log(nu), np.log(target)
    for shell in range(material['ns']):
        if not np.isfinite(j[shell]).all() or np.any(j[shell] < 0):
            raise ValueError('자체 전 주파수 J에 비유한/음수 값')
        native = len(radius)-2-shell
        reference = .5*(records['field'][::-1, 0, native] +
                        records['field'][::-1, 0, native+1])
        candidate = np.interp(logtarget, lognu, j[shell])
        yield shell, target, candidate, reference


def compare(config, execution, out):
    config, execution, out = Path(config).resolve(), Path(execution).resolve(), Path(out)
    if out.exists():
        raise ValueError('기존 비교 산출물 보존')
    c = json.loads(config.read_text())
    run = json.loads(execution.read_text())
    if run['status'] != 'FIELD0_COMPLETED' or run['physical_field_executions'] != 1:
        raise ValueError('완료된 실제 자체장0만 비교 가능')
    if sha(config) != run['config_sha256']:
        raise ValueError('실행 구성 신원 불일치')
    plan_path = Path(c['plan'])
    if sha(plan_path) != c['plan_sha256'] or c['plan_sha256'] != run['plan_sha256']:
        raise ValueError('계획 신원 불일치')
    plan = json.loads(plan_path.read_text())
    seed_report = json.loads((Path(c['seed'])/'material-seed.json').read_text())
    if (seed_report['source_record'] != plan['reference_radiation_material_record'] or
        seed_report['source_record'] != plan['initial_native_record'] or
        seed_report['record_storage_sha256'] != plan['initial_record_storage_sha256'] or
        seed_report['plan_sha256'] != c['plan_sha256']):
        raise ValueError('복사장과 물질 세대 불일치')
    report_path = execution.parent/'field0/preflight.json'
    report = json.loads(report_path.read_text())
    if (report != run['field0'] or report['status'] != 'CANDIDATE_COMPLETED' or
        report['implemented_frame'] != REL_FRAME or not report['candidate']['componentwise_verified']):
        raise ValueError('완료 수송/오차 확인 증거 불일치')
    material_path = Path(c['cells'])/'material_cells.bin'
    identity(report, material_path)
    material = read_material(material_path, b'LUMCEL01')
    own = execution.parent/'field0/candidate_field.npz'
    if sha(own) != report['output_sha256']['candidate_field.npz']:
        raise ValueError('완료 자체장 신원 불일치')
    reference_path = Path(plan['same_material_reference_field'])/'FORMAL_PRIMARY_FIELDS.bin'
    if sha(reference_path) != plan['reference_field_sha256']:
        raise ValueError('봉인 CMFGEN 참조장 신원 불일치')
    geometry, records = fields(reference_path)
    if sha(Path(c['grid'])) != c['grid_sha256']:
        raise ValueError('봉인 전체 주파수 격자 불일치')
    with np.load(own, allow_pickle=False) as archive:
        nu = archive['nu_Hz']
        if 'error_upper' not in archive.files:
            raise ValueError('완료 자체장의 기존 오차상한 배열 누락')
        error = archive['error_upper']
        error_maxima = iteration_error_maxima(error, (material['ns'], len(nu)))
        del error  # 큰 J와 오차 배열을 동시에 보유하지 않는다.
        j = archive['J']  # 큰 J는 한 번만 읽는다.
    if not np.array_equal(nu, np.load(c['grid'], allow_pickle=False)):
        raise ValueError('자체장과 실행 주파수 격자 불일치')
    out.mkdir()
    shape = (material['ns'], len(records))
    arrays = {name: np.lib.format.open_memmap(out/(name+'.npy'), mode='w+',
              dtype='f8', shape=shape) for name in ('lumina_J', 'cmfgen_J', 'relative_error')}
    rows = []
    for shell, target, candidate, reference in matched_shells(geometry, records, material, nu, j):
        metrics, relative = statistics(candidate, reference)
        arrays['lumina_J'][shell] = candidate
        arrays['cmfgen_J'][shell] = reference
        arrays['relative_error'][shell] = relative
        radius = .5*(material['physical'][shell, 0]+material['physical'][shell, 1])
        own_max = float(j[shell].max())
        # 절대값을 보존한다. 비율을 맞추기 위한 적분 재정규화는 하지 않는다.
        rows.append(dict(shell=shell, radius_cm=float(radius), **metrics,
            own_grid_J_max=own_max,
            own_grid_iteration_error_upper_max=float(error_maxima[shell]),
            own_grid_iteration_error_max_norm_ratio=norm_ratio(float(error_maxima[shell]), own_max),
            integral_lumina_J_dnu=float(np.sum(.5*(candidate[:-1]+candidate[1:])*np.diff(target))),
            integral_cmfgen_J_dnu=float(np.sum(.5*(reference[:-1]+reference[1:])*np.diff(target)))))
    for a in arrays.values():
        a.flush()
    np.save(out/'frequency_Hz.npy', target, allow_pickle=False)
    with (out/'shells.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    q = np.quantile(arrays['relative_error'], [.5, .95, 1], method='nearest')
    result = dict(schema='lumina-bridge-field0-comparison-v1',
        material_record=plan['initial_native_record'], reference_record=plan['reference_radiation_record'],
        input_sha256={str(p): sha(p) for p in (config, execution, report_path, material_path)},
        radiation_sha256=dict(lumina=report['output_sha256']['candidate_field.npz'],
                             cmfgen=plan['reference_field_sha256']),
        cells=shape[0], reference_frequencies=shape[1], own_frequencies=len(nu),
        spatial_rule='arithmetic radial midpoint; linear-in-r native J; not volume-averaged J',
        frequency_rule='linear J in log(nu) onto every native frequency; no extrapolation',
        units='nu: Hz; r: cm; J_nu: erg s^-1 cm^-2 Hz^-1 sr^-1',
        quantile_rule='all samples, NumPy method=nearest; 0/0 discrepancy=0; positive/0=Infinity',
        relative_median=finite_number(q[0]), relative_p95=finite_number(q[1]),
        relative_max=finite_number(q[2]),
        positive_over_zero=sum(r['positive_over_zero'] for r in rows),
        zero_zero=sum(r['zero_zero'] for r in rows),
        own_grid_J_max=max(r['own_grid_J_max'] for r in rows),
        own_grid_iteration_error_upper_max=float(error_maxima.max()),
        own_grid_iteration_error_max_norm_ratio=norm_ratio(float(error_maxima.max()),
            max(r['own_grid_J_max'] for r in rows)),
        iteration_error_scope='existing discrete fixed-point bound on every own-grid sample; '
            'max(error_upper)/max(J); not grid/interpolation/physical error certification',
        iteration_error_adequacy='NOT_JUDGED; a valid bound is not automatically a sufficiently small one',
        comparison_source_sha256=sha(__file__),
        physical_equivalence='NOT_JUDGED', final_goal='NOT_JUDGED',
        next='미설명 차이가 있으면 물질 갱신 중단; 원인을 수송/계수/공간·주파수 이산화별 조사')
    (out/'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                                allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('config', 'execution', 'out'):
        parser.add_argument('--'+key, type=Path, required=True)
    a = parser.parse_args()
    print(json.dumps(compare(a.config, a.execution, a.out), ensure_ascii=False))
