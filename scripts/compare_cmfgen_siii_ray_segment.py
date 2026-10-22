#!/usr/bin/env python3
"""실제 CMFGEN 한 광선의 지정 native 구간에서 주파수 연산자를 조건부 대조한다.

두 후보는 같은 입사 강도와 구간 평균 χ/η/b를 사용한다. 이 구간 평균은
원 CMFGEN Bézier 공간 SC나 전체 Lumina 수송의 계수 복원이 아니다.
포획 대역 밖 청색 입력이 필요한 주파수는 연속해 비교에서 제외한다.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from analyze_cmfgen_formal_capture import geometry
from analyze_cmfgen_siii_compact_capture import records
from check_cmfgen_local_sc_packet import e_coeff
from lumina_upwind_p0_bdf import bdf_p0_segment_nonuniform
from prepare_cmfgen_mom_capture import sha
from prepare_cmfgen_siii_compact_capture import FIRST, LAST


def native_segment_rows(outer_native_zero_index, np_rays):
    """포획 ID49–52와 Python native 배열의 0-based 인덱스를 연결한다."""
    if (type(outer_native_zero_index) is not int or
            outer_native_zero_index not in (48, 49, 50) or
            type(np_rays) is not int or np_rays <= outer_native_zero_index + 1):
        raise ValueError('포획 범위의 인접 native 구간 필요')
    outer_row = outer_native_zero_index - 48
    return outer_row, outer_row + 1, np_rays - outer_native_zero_index - 1


def actual_ray(capture, ray_number, outer_native_zero_index=49):
    field = capture / 'field'
    geometry_files = list(field.glob(f'FORMAL_{FIRST}_*_geometry.bin'))
    if len(geometry_files) != 1:
        raise ValueError('봉인 첫 주파수 기하가 유일하지 않음')
    g = geometry(geometry_files[0])
    outer_row, inner_row, crossing_rays = native_segment_rows(
        outer_native_zero_index, g['NP'])
    if not 1 <= ray_number <= crossing_rays:
        raise ValueError('선택한 native 두 점을 지나는 광선 번호 필요')
    ray = g['rays'][ray_number - 1]
    outer = int(ray['LNK'][outer_native_zero_index]) - 1
    inner = int(ray['LNK'][outer_native_zero_index + 1]) - 1
    if not (0 <= outer < inner < len(ray['S_P'])):
        raise ValueError('외향 광선 native49/50 링크 순서 오류')
    length = float(ray['S_P'][outer] - ray['S_P'][inner])
    drift = .5 * float(ray['B_P'][outer] + ray['B_P'][inner]) * length
    if not np.isfinite([length, drift]).all() or length <= 0 or drift <= 0:
        raise ValueError('실제 광선의 양의 길이·청색 드리프트 필요')
    values = []
    indices = []

    def keep(row):
        p_outer = row['points'][outer_row][ray_number - 1]
        p_inner = row['points'][inner_row][ray_number - 1]
        indices.append(row['index'])
        values.append((row['freq'], p_inner[0], p_outer[0],
                       .5 * (row['chi_ext'][outer] + row['chi_ext'][inner]),
                       .5 * (row['eta_ext'][outer] + row['eta_ext'][inner]),
                       row['dlognu'], p_outer[2], p_inner[2],
                       row['chi_ext'][outer], row['chi_ext'][inner],
                       row['eta_ext'][outer], row['eta_ext'][inner]))

    previous = None
    for row in records(field / 'siii_compact.bin', g, include_rays=True):
        if previous is not None and row['index'] != previous['index']:
            keep(previous)
        previous = row
    if previous is not None:
        keep(previous)
    if (len(indices) != LAST - FIRST + 1 or
            [indices[0], indices[-1]] != [FIRST, LAST] or
            any(b != a + 1 for a, b in zip(indices[:-1], indices[1:]))):
        raise ValueError('마지막 formal 호출의 전체 대역 누락')
    data = np.asarray(values[::-1], dtype='f8')
    log_frequency = np.log(data[:, 0])
    if (not np.isfinite(data).all() or np.any(np.diff(log_frequency) <= 0) or
            np.any(data[:, 1:3] < 0) or np.any(data[:, 4] < 0) or
            np.any(data[:, 6:8] < 0) or np.any(data[:, 10:12] < 0) or
            not np.allclose(np.diff(log_frequency), data[:-1, 5], rtol=1e-8, atol=1e-12)):
        raise ValueError('원격자·광선·계수 신원 불일치')
    return dict(log_frequency=log_frequency, incoming=data[:, 1],
                observed=data[:, 2], chi=data[:, 3], eta=data[:, 4],
                dlognu=data[:, 5], previous_outer=data[:, 6],
                previous_inner=data[:, 7], chi_outer=data[:, 8],
                chi_inner=data[:, 9], eta_outer=data[:, 10],
                eta_inner=data[:, 11], b_outer=float(ray['B_P'][outer]),
                b_inner=float(ray['B_P'][inner]),
                outer_native_zero_index=outer_native_zero_index,
                length=length, drift=drift, geometry=geometry_files[0],
                compact=field / 'siii_compact.bin')


def cmfgen_linear_spatial_exit(length, dlognu, chi_outer, chi_inner,
                               eta_outer, eta_inner, previous_outer,
                               previous_inner, b_outer, b_inner, current_inner):
    """원 CMFGEN 상풍 source를 고정한 두 끝점 선형 공간 SC 대조.

    원 내부 구간의 Bézier 제한자는 포함하지 않는다. 따라서 이 값과
    포획 출구의 차이만 공간 source 적분식의 조건부 차이다.
    """
    terms = np.asarray((length, dlognu, chi_outer, chi_inner, eta_outer,
                        eta_inner, previous_outer, previous_inner,
                        b_outer, b_inner, current_inner), dtype='f8')
    if (not np.isfinite(terms).all() or length <= 0 or dlognu <= 0 or
            min(eta_outer, eta_inner, previous_outer, previous_inner,
                current_inner) < 0):
        raise ValueError('원 공간 source 대조 입력 오류')
    alpha = 1. / dlognu
    effective_outer = chi_outer + (alpha + 3.) * b_outer
    effective_inner = chi_inner + (alpha + 3.) * b_inner
    if min(effective_outer, effective_inner) <= 0:
        raise ValueError('원 공간 source 유효 χ 비양수')
    source_outer = (eta_outer + alpha * b_outer * previous_outer) / effective_outer
    source_inner = (eta_inner + alpha * b_inner * previous_inner) / effective_inner
    tau = .5 * length * (effective_outer + effective_inner)
    ee, e0, e1, _ = e_coeff(tau)
    linear = (current_inner * ee + (e1 / tau) * source_outer +
              (e0 - e1 / tau) * source_inner)
    if not np.isfinite((source_outer, source_inner, tau, linear)).all():
        raise ValueError('원 공간 source 대조 결과 비유한')
    return dict(tau=float(tau), source_outer=float(source_outer),
                source_inner=float(source_inner), linear_exit=float(linear))


def continuous_scalar_reference(log_frequency, length, drift, chi, eta,
                                incident, target_index, rtol=1e-8):
    """한 실제 ν에서 모든 원격자 접점을 분할한 강성 연속해 기준값.

    전주파수 생산 계산기가 아니라 좁은 선의 공간 적분을 판별하는
    소수 표본 oracle이다. 청색 범위 밖 외삽은 허용하지 않는다.
    """
    x = np.asarray(log_frequency, dtype='f8')
    chi = np.asarray(chi, dtype='f8')
    eta = np.asarray(eta, dtype='f8')
    incident = np.asarray(incident, dtype='f8')
    if (x.ndim != 1 or len(x) < 2 or any(a.shape != x.shape for a in
            (chi, eta, incident)) or not all(np.isfinite(a).all() for a in
            (x, chi, eta, incident)) or np.any(np.diff(x) <= 0) or
            np.any(eta < 0) or np.any(incident < 0) or
            not np.isfinite([length, drift, rtol]).all() or length <= 0 or
            drift <= 0 or not 0 < rtol <= 1e-3 or
            type(target_index) is not int or not 0 <= target_index < len(x)):
        raise ValueError('단일 주파수 강성 연속해 입력 오류')
    target = float(x[target_index])
    blue = target + drift
    if blue > x[-1]:
        raise ValueError('포획 대역 밖 청색 경계 필요')
    knots = x[(x > target) & (x < blue)][::-1]
    boundaries = np.r_[blue, knots, target]
    intensity = float(np.interp(blue, x, incident))
    for high, low in zip(boundaries[:-1], boundaries[1:]):
        duration = length * (high - low) / drift
        q_high = float(np.interp(high, x, chi)) + 3 * drift / length
        q_low = float(np.interp(low, x, chi)) + 3 * drift / length
        e_high = float(np.interp(high, x, eta))
        e_low = float(np.interp(low, x, eta))

        def derivative(t, value):
            q = q_high + t * (q_low - q_high)
            e = e_high + t * (e_low - e_high)
            return duration * (e - q * value)

        result = solve_ivp(derivative, (0., 1.), [intensity], method='BDF',
                           rtol=rtol, atol=1e-18)
        if not result.success or not np.isfinite(result.y).all():
            raise ValueError('원격자 접점별 강성 연속해 실패')
        intensity = float(result.y[0, -1])
    if not np.isfinite(intensity) or intensity < 0:
        raise ValueError('원격자 접점별 연속해 음수/비유한')
    return intensity, len(boundaries) - 1


def compare_spot(capture, out, ray_number, target_index,
                 outer_native_zero_index=49):
    """원격자 접점별 강성 연속해와 P0를 실제 한 광선·한 ν에서 대조한다."""
    script_sha256 = sha(Path(__file__))
    if out.exists():
        raise FileExistsError('기존 실제 광선 표본 결과 보존')
    data = actual_ray(capture, ray_number, outer_native_zero_index)
    x = data['log_frequency']
    if (type(target_index) is not int or not 0 <= target_index < len(x) or
            x[target_index] + 2 * data['drift'] > x[-1]):
        raise ValueError('포획 청색 경계에서 두 구간 이동량 이상 떨어진 표본 필요')
    args = (x, data['length'], data['drift'], data['chi'], data['eta'], data['incoming'])
    p0_loose, info_loose = bdf_p0_segment_nonuniform(*args, rtol=1e-8, atol=1e-15)
    p0_tight, info_tight = bdf_p0_segment_nonuniform(*args, rtol=1e-10, atol=1e-18)
    continuous_loose, pieces = continuous_scalar_reference(*args, target_index,
                                                             rtol=1e-9)
    continuous_tight, pieces_tight = continuous_scalar_reference(*args, target_index,
                                                                   rtol=1e-10)
    if pieces != pieces_tight:
        raise ValueError('강성 연속해 원격자 접점 수 불일치')
    ref = float(data['observed'][target_index])
    k = target_index
    spatial = cmfgen_linear_spatial_exit(
        data['length'], data['dlognu'][k], data['chi_outer'][k],
        data['chi_inner'][k], data['eta_outer'][k], data['eta_inner'][k],
        data['previous_outer'][k], data['previous_inner'][k],
        data['b_outer'], data['b_inner'], data['incoming'][k])
    p0 = float(p0_tight[target_index])
    norm = max(abs(continuous_tight), abs(p0), 1e-300)
    report = dict(scope=__doc__, status='LOCAL_CONDITIONAL_ONLY',
                  ray_number=ray_number, reverse_band_index=target_index,
                  outer_native_zero_index=outer_native_zero_index,
                  fortran_outer_id=outer_native_zero_index+1,
                  fortran_inner_id=outer_native_zero_index+2,
                  frequency_Hz=float(np.exp(x[target_index]) * 1e15),
                  wavelength_A=float(2.99792458e18 / (np.exp(x[target_index]) * 1e15)),
                  length=data['length'], drift_log_frequency=data['drift'],
                  blue_margin_log_frequency=float(x[-1] - x[target_index]),
                  original_cmfgen_exit_intensity=ref,
                  cmfgen_linear_spatial=spatial,
                  original_minus_linear_spatial=ref-spatial['linear_exit'],
                  p0_intensity=p0, continuous_intensity=continuous_tight,
                  p0_minus_continuous=p0 - continuous_tight,
                  p0_minus_continuous_relative_to_pair_scale=(p0-continuous_tight)/norm,
                  original_minus_continuous=ref - continuous_tight,
                  p0_tolerance_change=float(abs(p0_loose[target_index] - p0)),
                  continuous_tolerance_change=abs(continuous_loose-continuous_tight),
                  continuous_frequency_pieces=pieces,
                  p0_bdf_nfev=[info_loose['nfev'], info_tight['nfev']],
                  coefficient_rule='same captured incident spectrum, endpoint-mean chi/eta, one specified native ray segment',
                  interpretation='original CMFGEN Bezier SC uses a different spatial stencil; original-minus-model is not a pure frequency-operator effect',
                  file_sha256={str(data['geometry']): sha(data['geometry']),
                               str(data['compact']): sha(data['compact']),
                               str(Path(__file__)): script_sha256},
                  final_goal='NOT_MET', operator_contribution='NOT_DETERMINED')
    with out.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(result=str(out), wavelength_A=report['wavelength_A'],
                          p0_minus_continuous=report['p0_minus_continuous'],
                          continuous_tolerance_change=report['continuous_tolerance_change']),
                     ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--ray', type=int, default=1)
    parser.add_argument('--outer-native-index', type=int, default=49,
                        help='Python 0-based 바깥 native 인덱스; 48,49,50 중 선택')
    parser.add_argument('--spot-index', type=int, required=True,
                        help='증가하는 주파수 순서의 0-based 표본 인덱스')
    args = parser.parse_args()
    compare_spot(args.capture.resolve(), args.out.resolve(), args.ray,
                 args.spot_index, args.outer_native_index)
