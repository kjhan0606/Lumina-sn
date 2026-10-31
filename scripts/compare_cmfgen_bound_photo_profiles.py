#!/usr/bin/env python3
"""고정 인구의 Si/S bound photo율을 전 반경에서 같은 격자로 비교한다.

logν 보간의 전치 작용으로 가중치를 원 CMFGEN 격자에 합친다. 원래 fine
격자의 σ·구적·보간식을 바꾸지 않는다. SE/열/수송이나 수용 판정은 하지 않는다.
"""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np


class LogInterpolationAdjoint:
    """sum(q * interp(J)) == sum(project(q) * J); 외삽/율 floor 없음."""
    def __init__(self, native, fine):
        native, fine = (np.asarray(x, dtype='f8') for x in (native, fine))
        for x in (native, fine):
            if (x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all() or
                    np.any(x <= 0) or np.any(np.diff(x) <= 0)):
                raise ValueError('양의 오름차순 주파수 격자 필요')
        if fine[0] < native[0] or fine[-1] > native[-1]:
            raise ValueError('CMFGEN 장 범위 밖 외삽 금지')
        x, y = np.log(native), np.log(fine)
        if np.any(np.diff(x) <= 0):
            raise ValueError('log 주파수에서 구분되지 않는 원격자')
        self.left = np.minimum(np.searchsorted(x, y, side='right')-1, len(x)-2)
        self.alpha = (y-x[self.left])/(x[self.left+1]-x[self.left])
        self.size = len(native)
        if np.any(self.alpha < 0) or np.any(self.alpha > 1):
            raise ValueError('보간 지지 범위 오류')

    def project(self, weights):
        weights = np.asarray(weights, dtype='f8')
        if (weights.shape != self.alpha.shape or not np.isfinite(weights).all() or
                np.any(weights < 0)):
            raise ValueError('bound photo의 유한한 양의 구적 가중치 필요')
        return (np.bincount(self.left, weights=weights*(1-self.alpha), minlength=self.size)+
                np.bincount(self.left+1, weights=weights*self.alpha, minlength=self.size))


def run(config_path, config_sha, out):
    import cmfgen_model_continuum as B
    from lumina_new_field_response import trapezoid_weights
    from prepare_lumina_outer_input import read_native
    from run_cmfgen_formal_primary import fields

    started = time.monotonic()
    config_path, out = Path(config_path).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError('기존 비교 출력 보존')
    identities = {}

    def checked(path, sha256):
        path = Path(path).resolve()
        if B.sha(path) != sha256:
            raise ValueError('비교 입력 변경: '+path.name)
        identities[str(path)] = sha256
        return path

    checked(config_path, config_sha)
    config = json.loads(config_path.read_text())
    if config.get('schema') != 'lumina-bound-photo-profile-config-v1':
        raise ValueError('bound photo 비교 구성 필요')
    paths = {k: checked(**config[k]) for k in
             ('old_result', 'new_result', 'reference_field', 'population_response')}
    old, new = (json.loads(paths[k].read_text()) for k in ('old_result', 'new_result'))
    population_digest = config['population_response']['sha256']
    if population_digest == new['source_identity']['accepted_response_sha256']:
        population_role = 'current_transport_input_material'
    elif population_digest == new['selected_response']['response_sha256']:
        population_role = 'current_returned_material'
    else:
        raise ValueError('현재 수송의 입력/반환 인구만 고정 비교에 사용한다')
    shared = set(old['input_sha256']) & set(new['input_sha256'])
    if any(old['input_sha256'][p] != new['input_sha256'][p] for p in shared):
        raise ValueError('두 생산자의 공유 입력이 달라 저장율을 직접 비교할 수 없음')
    digest = new['source_identity']['preparation_sha256']
    matches = [p for p, h in new['input_sha256'].items() if h == digest]
    if len(matches) != 1:
        raise ValueError('현재 계수 준비 기록 모호')
    preparation = json.loads(checked(matches[0], digest).read_text())
    # 실제 원자자료/율 구현의 봉인 신원을 계승하되 현재 읽는 입력도 대조한다.
    for path, digest in new['input_sha256'].items():
        checked(path, digest)
    nu = np.load(checked(preparation['grid'], new['source_identity']['grid_sha256']), allow_pickle=False)
    geometry, reference = fields(paths['reference_field'])
    mapped = []
    try:
        physical, _ = read_native(checked(preparation['material'],
                                         preparation['input_sha256'][preparation['material']]))
        old_digest = old['source_identity']['preparation_sha256']
        old_matches = [p for p, h in old['input_sha256'].items() if h == old_digest]
        if len(old_matches) != 1:
            raise ValueError('이전 계수 준비 기록 모호')
        old_preparation = json.loads(checked(old_matches[0], old_digest).read_text())
        old_material = checked(old_preparation['material'],
                               old_preparation['input_sha256'][old_preparation['material']])
        old_physical, _ = read_native(old_material)
        if not np.array_equal(old_physical[:, 0], physical[:, 0]):
            raise ValueError('두 저장율의 실제 반경 불일치')
        if not np.array_equal(geometry[0]*1e10, physical[:, 0]):
            raise ValueError('CMFGEN 기준 장과 실제 반경 불일치')
        native = np.asarray(reference['freq'][::-1])*1e15
        operator = LogInterpolationAdjoint(native, nu)
        J = np.ascontiguousarray(reference['field'][::-1, 0, :].T)
        roster_values = []
        for report, key in ((old, 'old_result'), (new, 'new_result')):
            folder = paths[key].parent
            for name in ('route-roster.npz', 'bound-rates.npy'):
                checked(folder/name, report['output_sha256'][name])
            with np.load(folder/'route-roster.npz', allow_pickle=False) as data:
                roster_values.append({k: data[k] for k in data.files})
            mapped.append(np.load(folder/'bound-rates.npy', mmap_mode='r', allow_pickle=False))
        roster = roster_values[0]
        if (set(roster) != set(roster_values[1]) or
                any(not np.array_equal(v, roster_values[1][k]) for k, v in roster.items())):
            raise ValueError('두 저장 bound율의 원자 경로 불일치')
        if old['source_identity']['grid_sha256'] != new['source_identity']['grid_sha256']:
            raise ValueError('두 저장 bound율의 주파수 격자 불일치')
        model_paths = [Path(p) for p in new['input_sha256'] if Path(p).name == 'MODEL']
        phot_paths = [Path(p) for p in new['input_sha256'] if Path(p).name == 'cmfgen_phot_smoothing.so']
        if len(model_paths) != 1 or len(phot_paths) != 1:
            raise ValueError('실제 MODEL/PHOT 생산자 모호')
        rows, models, routes, _, _, _ = B.load_model(
            model_paths[0].parent, Path(preparation['material']).parent/'atomic', phot_paths[0])
        routes = sorted(routes, key=lambda r: (r.lower, r.phot_grid, r.target_superlevel))
        expected = {'lower': 'lower', 'target': 'target', 'Z': 'Z', 'stage': 'stage',
                    'threshold': 'threshold_Hz', 'phot_grid': 'phot_grid', 'probability': 'probability'}
        if any(not np.array_equal(roster[k], [getattr(r, attr) for r in routes])
               for k, attr in expected.items()):
            raise ValueError('현재 PHOT와 저장 경로 불일치')
        with np.load(paths['population_response'], allow_pickle=False) as data:
            columns, pop, fraction = (data[k] for k in ('columns', 'population', 'fraction'))
        n = len(physical)
        if (pop.shape != (len(columns), n) or fraction.shape != (len(rows), n) or
                not np.isfinite(pop).all() or not np.isfinite(fraction).all() or
                np.any(pop < 0) or np.any(fraction < 0) or
                any(x.shape != (len(routes), n, 6) for x in mapped)):
            raise ValueError('고정 인구/저장율의 깊이·형상·유한성 오류')
        index = {tuple(map(int, c)): i for i, c in enumerate(columns)}
        expected_columns = {(int(r['atomic_number']), int(r['ion_number']), int(r['super_level']))
                            for r in rows}
        expected_columns |= {(z, stage, -1) for z, stage in models if (z, stage+1) not in models}
        if len(index) != len(columns) or set(index) != expected_columns:
            raise ValueError('고정 인구의 SL/terminal 주소 불일치')
        group = np.array([index[int(r['atomic_number']), int(r['ion_number']), int(r['super_level'])]
                          for r in rows])
        width = 4*np.pi*trapezoid_weights(nu)/(B.H*nu)
        stages = ((14, 2), (14, 3), (16, 2), (16, 3))
        records = []
        out.mkdir()
        report = dict(state='INCOMPLETE_BOUND_PHOTO_COMPARISON', config_sha256=config_sha,
                      scope='fixed population; bound photo only; no SE/RE/RT or equivalence verdict',
                      population_role=population_role,
                      implementation_sha256={name: B.sha(Path(__file__).with_name(name)) for name in
                          (Path(__file__).name, 'cmfgen_model_continuum.py',
                           'run_cmfgen_formal_primary.py', 'lumina_new_field_response.py')},
                      input_sha256=identities, stages=[], new_SE=0, new_transport=0, final_goal='NOT_MET')

        def save():
            report['wall_seconds'] = time.monotonic()-started
            temporary = out/'result.partial.json'
            temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
            temporary.replace(out/'result.json')

        save()
        for z, stage in stages:
            ids = np.flatnonzero((roster['Z'] == z) & (roster['stage'] == stage))
            density = pop[(columns[:, 0] == z) & (columns[:, 1] == stage-1)].sum(axis=0)
            if not len(ids) or np.any(density <= 0):
                raise ValueError('선택 이온의 인구/경로 누락')
            totals = np.zeros((3, n), dtype=np.longdouble)
            for rid in ids:
                if time.monotonic()-started >= 285:
                    raise TimeoutError('소형 CPU 비교의 wall300초 여유 보존')
                route = routes[int(rid)]
                projected = operator.project(width*route.sigma(nu))
                gamma = J@projected
                weight = np.asarray(pop[group[route.lower]]*fraction[route.lower]*route.probability/density,
                                    dtype=np.longdouble)
                for k, values in enumerate((mapped[0][rid, :, 0], mapped[1][rid, :, 0], gamma)):
                    if not np.isfinite(values).all() or np.any(values < 0):
                        raise ValueError('bound photo율의 비유한/음수')
                    totals[k] += weight*values
            for node in range(n):
                a, b, c = map(float, totals[:, node])
                records.append(dict(point_1based=node+1, radius_cm=float(physical[node, 0]), Z=z,
                                    stage=stage, old_gamma=a, new_gamma=b, CMFGEN_J_gamma=c,
                                    old_over_CMFGEN_J=None if c == 0 else a/c,
                                    new_over_CMFGEN_J=None if c == 0 else b/c))
            report['stages'].append(dict(Z=z, stage=stage, routes=len(ids), points=n))
            with (out/'profiles.csv').open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader(); writer.writerows(records)
            save()
            print(json.dumps(report['stages'][-1]), flush=True)
        for path, digest in identities.items():
            checked(path, digest)
        report.update(state='COMPLETE_BOUND_PHOTO_COMPARISON_NOT_EQUIVALENCE',
                      profiles_sha256=B.sha(out/'profiles.csv'))
        save()
        return report
    finally:
        reference._mmap.close()
        for data in mapped:
            data._mmap.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config, args.config_sha256, args.out)
    print(json.dumps({k: result[k] for k in ('state', 'wall_seconds', 'profiles_sha256')}))
