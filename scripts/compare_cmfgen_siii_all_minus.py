#!/usr/bin/env python3
"""봉인 S III inward 전광선의 목표 한 구간 공간세분을 J·PHOT율로 합산한다.

원 외부 입사·이웃 source와 나머지 구간/방향/물질은 고정한다.
이는 국소 조건부 민감도이며 전체 formal/SE/RE 또는 새 스펙트럼이 아니다.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze_cmfgen_formal_capture import geometry
from analyze_cmfgen_siii_compact_capture import records
from compare_cmfgen_siii_all_plus import production_weight
from compare_cmfgen_siii_ray_substeps import sc_sweep
from prepare_cmfgen_mom_capture import sha
from prepare_cmfgen_siii_compact_capture import FIRST, LAST


def collect_minus(capture):
    field = capture / 'field'
    candidates = list(field.glob(f'FORMAL_{FIRST}_*_geometry.bin'))
    if len(candidates) != 1:
        raise ValueError('첫 주파수 봉인 기하가 유일하지 않음')
    g = geometry(candidates[0])
    nrays = g['NP'] - 49  # native49, Fortran ID50: inward 방향 56광선
    if nrays != 56 or g['RAY_POINTS_INSERTED']:
        raise ValueError('봉인 native49 inward 광선 계약 불일치')
    links = np.array([[int(ray['LNK'][j])-1 for j in (48, 49)]
                      for ray in g['rays'][:nrays]])
    upstream_link, target_link = links.T
    if np.any(upstream_link < 0) or np.any(target_link <= upstream_link):
        raise ValueError('원 inward native 링크 순서 오류')
    far_link = np.array([int(ray['LNK'][50])-1 for ray in g['rays'][:nrays]])
    if np.any(far_link[:-1] <= target_link[:-1]) or far_link[-1] != -1:
        raise ValueError('원 inward 이웃/끝점 링크 오류')
    frame, frequencies, intervals, native_j, indices = [], [], [], [], []

    def keep(row):
        upstream = np.asarray(row['points'][0][:nrays], dtype='f8')
        target = np.asarray(row['points'][1][:nrays], dtype='f8')
        far = np.asarray(row['points'][2][:nrays-1], dtype='f8')
        arrays = np.stack((upstream[:, 1], target[:, 1],
                           upstream[:, 3], target[:, 3],
                           row['chi_ext'][upstream_link],
                           row['chi_ext'][target_link],
                           row['eta_ext'][upstream_link],
                           row['eta_ext'][target_link]), axis=1)
        far_arrays = np.stack((far[:, 3], row['chi_ext'][far_link[:-1]],
                               row['eta_ext'][far_link[:-1]]), axis=1)
        if not np.isfinite(arrays).all() or not np.isfinite(far_arrays).all():
            raise ValueError('원 inward 광선 장·계수 비유한')
        frame.append((arrays, far_arrays))
        frequencies.append(row['freq'])
        intervals.append(row['dlognu'])
        native_j.append(row['native'][1])
        indices.append(row['index'])

    prior = None
    for row in records(field / 'siii_compact.bin', g, include_rays=True):
        if prior is not None and row['index'] != prior['index']:
            keep(prior)
        prior = row
    if prior is not None:
        keep(prior)
    if len(indices) != LAST-FIRST+1 or indices != list(range(FIRST, LAST+1)):
        raise ValueError('원 목표 대역 최종 formal 호출 누락')
    data = np.stack([item[0] for item in frame[::-1]])
    far_data = np.stack([item[1] for item in frame[::-1]])
    freq = np.asarray(frequencies[::-1], dtype='f8')
    dlognu = np.asarray(intervals[::-1], dtype='f8')
    j = np.asarray(native_j[::-1], dtype='f8')
    if (np.any(data[:, :, [0, 1, 2, 3, 6, 7]] < 0) or
            np.any(far_data[:, :, [0, 2]] < 0) or
            not np.allclose(np.diff(np.log(freq)), dlognu[:-1],
                            rtol=1e-8, atol=1e-12) or
            np.max(np.abs(data[:-1, :, 2]-data[1:, :, 0])) != 0 or
            np.max(np.abs(data[:-1, :, 3]-data[1:, :, 1])) != 0):
        raise ValueError('원 inward 청색 연쇄 또는 주파수 계약 불일치')
    weights = np.asarray(g['weights']['JQW_M'][49, :nrays], dtype='f8')
    if np.any(weights < 0) or not np.isfinite(weights).all():
        raise ValueError('원 inward 각도 가중치 오류')
    rays = []
    names = ('incoming', 'observed', 'previous_inner', 'previous_outer',
             'chi_inner', 'chi_outer', 'eta_inner', 'eta_outer')
    for r in range(nrays):
        ray = g['rays'][r]
        u, t = links[r]
        length = float(ray['S_M'][t]-ray['S_M'][u])
        if length <= 0:
            raise ValueError('원 inward 공간 길이 오류')
        values = {name: data[:, r, k] for k, name in enumerate(names)}
        values.update(length=length, dlognu=dlognu,
                      b_inner=float(ray['B_M'][u]),
                      b_outer=float(ray['B_M'][t]))
        if r < nrays-1:
            f = far_link[r]
            far_length = float(ray['S_M'][f]-ray['S_M'][t])
            if far_length <= 0:
                raise ValueError('원 inward 이웃 길이 오류')
            values['far_neighbor'] = dict(previous=far_data[:, r, 0],
                                          chi=far_data[:, r, 1],
                                          eta=far_data[:, r, 2],
                                          b=float(ray['B_M'][f]),
                                          length=far_length)
        else:
            values['far_neighbor'] = None
        rays.append(values)
    return dict(freq=freq, native_j=j, rays=rays, angle_weights=weights,
                geometry=candidates[0], compact=field/'siii_compact.bin')


def analyze(capture, transport_root, baseline_rates, out_root, subdivisions=8):
    if out_root.exists():
        raise FileExistsError('기존 inward 각도/율 결과 보존')
    if type(subdivisions) is not int or not 2 <= subdivisions <= 64:
        raise ValueError('한정된 조건부 공간 세분 수 필요')
    captured = collect_minus(capture)
    photo_weight, f_rate, j_rate, original, f_path = production_weight(
        captured['freq']*1e15, transport_root, baseline_rates,
        captured['native_j'])
    delta = np.zeros(len(captured['freq']))
    baseline_max_abs = 0.
    for r, ray in enumerate(captured['rays']):
        terminal = ray['far_neighbor'] is None
        baseline = sc_sweep(ray, 1, mode='bezier', terminal_linear=terminal)
        error = np.max(np.abs(baseline-ray['observed']))
        baseline_max_abs = max(baseline_max_abs, float(error))
        if error > 1e-10*max(float(np.max(ray['observed'])), 1e-300):
            raise ValueError(f'원 inward 광선 {r+1} 전대역 재현 실패')
        refined = sc_sweep(ray, subdivisions, mode='bezier',
                           terminal_linear=terminal)
        delta += captured['angle_weights'][r]*(refined-ray['observed'])
    j_adjusted = captured['native_j']+delta
    if not np.isfinite(j_adjusted).all() or np.any(j_adjusted < 0):
        raise ValueError('조건부 각도 적분 J 오류')
    delta_rate = float(np.sum(photo_weight*delta))
    samples = {str(k): dict(wavelength_A=float(2.99792458e18/
                                           (captured['freq'][k]*1e15)),
                            original_J=float(captured['native_j'][k]),
                            inward_delta_J=float(delta[k]),
                            conditional_J=float(j_adjusted[k]))
               for k in (2543, 2542, 839)}
    out_root.mkdir(parents=False)
    delta_path = out_root/'inward_delta_J.npy'
    np.save(delta_path, delta, allow_pickle=False)
    report = dict(scope=__doc__, status='LOCAL_CONDITIONAL_ONLY',
                  native_zero_index=49, inward_crossing_ray_count=len(captured['rays']),
                  frequencies=len(captured['freq']), subdivisions=subdivisions,
                  original_bezier_baseline_max_abs=baseline_max_abs,
                  minus_angle_weight_sum=float(np.sum(captured['angle_weights'])),
                  original_F_photo_cm3_s=f_rate, original_J_photo_cm3_s=j_rate,
                  original_F_minus_J_photo_cm3_s=f_rate-j_rate,
                  inward_delta_J_photo_cm3_s=delta_rate,
                  conditional_F_minus_J_photo_cm3_s=f_rate-j_rate-delta_rate,
                  samples=samples, delta_j_sha256=sha(delta_path),
                  file_sha256={str(p): sha(p) for p in
                               (captured['geometry'], captured['compact'],
                                original, f_path, baseline_rates, Path(__file__))},
                  actual_new_transport_calls=0, SE_RE_updates=0,
                  final_goal='NOT_MET')
    report_path = out_root/'report.json'
    with report_path.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2,
                  allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(result=str(report_path),
                          baseline_max_abs=baseline_max_abs,
                          inward_delta_J_photo_cm3_s=delta_rate,
                          conditional_F_minus_J_photo_cm3_s=
                          report['conditional_F_minus_J_photo_cm3_s'],
                          samples=samples), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--transport-root', type=Path, required=True)
    parser.add_argument('--baseline-rates', type=Path, required=True)
    parser.add_argument('--out-root', type=Path, required=True)
    parser.add_argument('--subdivisions', type=int, default=8)
    args = parser.parse_args()
    analyze(args.capture.resolve(), args.transport_root.resolve(),
            args.baseline_rates.resolve(), args.out_root.resolve(),
            args.subdivisions)
