#!/usr/bin/env python3
"""봉인 한 광선·한 native 구간의 청색→적색 SC 연쇄와 공간 세분을 대조한다.

안쪽 입사 스펙트럼은 원 포획을 강제한다. 내부 새 노드의 첫 청색
이전 강도는 두 원 끝점 사이 선형 초기화이므로, 세분 계산은
원 formal 전체 광선 또는 새 자기일관 장이 아니다.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from check_cmfgen_local_sc_packet import bezier_source, e_coeff
from compare_cmfgen_siii_ray_segment import actual_ray
from prepare_cmfgen_mom_capture import sha


def sc_sweep(data, subdivisions, minimum_index=0, mode='linear',
             initial_profile='linear'):
    """원 ν간격의 상풍 source와 선형 또는 Bézier 공간 SC를 연쇄한다."""
    if (type(subdivisions) is not int or not 1 <= subdivisions <= 64 or
            type(minimum_index) is not int or
            not 0 <= minimum_index < len(data['incoming']) or
            mode not in ('linear', 'bezier') or
            initial_profile not in ('linear', 'inner', 'outer') or
            (mode == 'bezier' and data.get('far_neighbor') is None)):
        raise ValueError('원 광선 세분·주파수 범위 오류')
    n = len(data['incoming'])
    fraction = np.linspace(0., 1., subdivisions + 1)
    b = ((1. - fraction) * data['b_inner'] +
         fraction * data['b_outer'])
    previous = ((1. - fraction) * data['previous_inner'][-1] +
                fraction * data['previous_outer'][-1])
    if initial_profile == 'inner':
        previous[1:-1] = data['previous_inner'][-1]
    elif initial_profile == 'outer':
        previous[1:-1] = data['previous_outer'][-1]
    current = np.empty_like(previous)
    result = np.full(n, np.nan)
    step = data['length'] / subdivisions
    for k in range(n - 1, minimum_index - 1, -1):
        previous[0] = data['previous_inner'][k]
        alpha = 1. / data['dlognu'][k]
        chi = ((1. - fraction) * data['chi_inner'][k] +
               fraction * data['chi_outer'][k])
        eta = ((1. - fraction) * data['eta_inner'][k] +
               fraction * data['eta_outer'][k])
        effective = chi + (alpha + 3.) * b
        if np.any(effective <= 0):
            raise ValueError('원 상풍 유효 χ 비양수')
        source = (eta + alpha * b * previous) / effective
        if mode == 'bezier':
            far = data['far_neighbor']
            effective_far = far['chi'][k] + (alpha + 3.) * far['b']
            if effective_far <= 0:
                raise ValueError('원 Bézier 바깥 이웃 유효 χ 비양수')
            source_far = ((far['eta'][k] + alpha * far['b'] *
                           far['previous'][k]) / effective_far)
            outer_tau = .5 * far['length'] * (effective[-1] + effective_far)
        current[0] = data['incoming'][k]
        tau_nodes = .5 * step * (effective[:-1] + effective[1:])
        for j in range(1, subdivisions + 1):
            tau = tau_nodes[j-1]
            ee, e0, e1, _ = e_coeff(tau)
            if mode == 'bezier':
                next_source = source[j+1] if j < subdivisions else source_far
                next_tau = tau_nodes[j] if j < subdivisions else outer_tau
                current[j] = (current[j-1] * ee +
                              bezier_source(source[j-1], source[j],
                                            next_source, tau, next_tau))
            else:
                current[j] = (current[j-1] * ee +
                              (e1 / tau) * source[j] +
                              (e0 - e1 / tau) * source[j-1])
        if not np.isfinite(current).all() or np.any(current < 0):
            raise ValueError('원 상풍 세분 광선 강도 오류')
        result[k] = current[-1]
        previous, current = current, previous
    return result


def linear_sc_sweep(data, subdivisions, minimum_index=0):
    return sc_sweep(data, subdivisions, minimum_index, 'linear')


def forced_bezier_one_step(data, k):
    """원 이전 주파수 세 점을 강제한 실제 native 한 구간 SC 직접 재현."""
    far = data.get('far_neighbor')
    if far is None or not 0 <= k < len(data['incoming']):
        raise ValueError('원 Bézier 강제 표본 입력 오류')
    alpha = 1. / data['dlognu'][k]
    effective = np.array([
        data['chi_inner'][k] + (alpha + 3.) * data['b_inner'],
        data['chi_outer'][k] + (alpha + 3.) * data['b_outer'],
        far['chi'][k] + (alpha + 3.) * far['b']])
    if np.any(effective <= 0):
        raise ValueError('원 Bézier 강제 표본 유효 χ 비양수')
    source = np.array([
        (data['eta_inner'][k] + alpha * data['b_inner'] *
         data['previous_inner'][k]) / effective[0],
        (data['eta_outer'][k] + alpha * data['b_outer'] *
         data['previous_outer'][k]) / effective[1],
        (far['eta'][k] + alpha * far['b'] * far['previous'][k]) /
        effective[2]])
    tau = .5 * data['length'] * (effective[0] + effective[1])
    neighbor = .5 * far['length'] * (effective[1] + effective[2])
    return float(data['incoming'][k] * e_coeff(tau)[0] +
                 bezier_source(source[0], source[1], source[2], tau, neighbor))


def analyze(capture, out, ray_number=1, outer_native_index=49,
            sample_indices=(2543, 2542, 839),
            subdivisions=(1, 2, 4, 8, 16, 32, 64)):
    if out.exists():
        raise FileExistsError('기존 광선 SC 세분 결과 보존')
    data = actual_ray(capture, ray_number, outer_native_index)
    n = len(data['incoming'])
    if (not sample_indices or len(set(sample_indices)) != len(sample_indices) or
            any(type(k) is not int or not 0 <= k < n for k in sample_indices) or
            tuple(sorted(set(subdivisions))) != subdivisions):
        raise ValueError('표본 인덱스·세분 목록 오류')
    previous_outer_match = np.max(np.abs(data['previous_outer'][:-1] -
                                         data['observed'][1:]))
    previous_inner_match = np.max(np.abs(data['previous_inner'][:-1] -
                                         data['incoming'][1:]))
    forced = {k: forced_bezier_one_step(data, k) for k in sample_indices}
    if any(abs(forced[k]-data['observed'][k]) >
           1e-10 * max(abs(data['observed'][k]), 1e-300)
           for k in sample_indices):
        raise ValueError('원 native Bézier 강제 표본 재현 실패')
    results = {}
    baseline_max_abs = None
    baseline_max_scaled = None
    for mode in ('linear', 'bezier'):
        results[mode] = {}
        for pieces in subdivisions:
            output = sc_sweep(data, pieces, 0, mode)
            results[mode][str(pieces)] = [float(output[k]) for k in sample_indices]
            if mode == 'bezier' and pieces == 1:
                error = np.abs(output - data['observed'])
                baseline_max_abs = float(np.max(error))
                baseline_max_scaled = float(baseline_max_abs /
                                            max(np.max(data['observed']), 1e-300))
                if baseline_max_scaled > 1e-10:
                    raise ValueError('원 native Bézier 전대역 청색 연쇄 재현 실패')
    samples = [dict(reverse_band_index=k,
                    wavelength_A=float(2.99792458e18 /
                                       (np.exp(data['log_frequency'][k]) * 1e15)),
                    original_exit_intensity=float(data['observed'][k]),
                    forced_bezier_exit_intensity=forced[k],
                    previous_outer_intensity=float(data['previous_outer'][k]),
                    simulated_sc={mode: {key: value[j] for key, value in
                                         cases.items()}
                                  for mode, cases in results.items()})
               for j, k in enumerate(sample_indices)]
    report = dict(scope=__doc__, status='LOCAL_CONDITIONAL_ONLY',
                  ray_number=ray_number,
                  outer_native_zero_index=outer_native_index,
                  blue_to_red_frequencies_simulated=n,
                  baseline_bezier_one_segment_max_abs=baseline_max_abs,
                  baseline_bezier_one_segment_max_scaled=baseline_max_scaled,
                  first_blue_previous_profile='linear interpolation of captured inner/outer previous intensity',
                  boundary_inner_current='captured at every frequency',
                  previous_inner='captured at every frequency',
                  max_previous_outer_chain_abs=float(previous_outer_match),
                  max_previous_inner_chain_abs=float(previous_inner_match),
                  samples=samples,
                  file_sha256={str(p): sha(p) for p in
                               (data['geometry'], data['compact'], Path(__file__))},
                  actual_new_transport_calls=0, SE_RE_updates=0,
                  final_goal='NOT_MET')
    with out.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2,
                  allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(result=str(out), samples=samples), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--ray', type=int, default=1,
                        help='목표 두 native 점을 통과하는 원 광선 번호')
    args = parser.parse_args()
    analyze(args.capture.resolve(), args.out.resolve(), ray_number=args.ray)
