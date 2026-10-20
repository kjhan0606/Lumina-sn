#!/usr/bin/env python3
"""실제 내부경계 입력으로 얇은 흡수층의 독립 정적 근사를 계산한다.

실제 광선/blue 이력을 재현하거나 수송 에너지 폐합을 판정하는 도구는 아니다.
flux에서 source를 역산하지 않고 고정 local source와 diffusion I+를 사용한다.
"""
import argparse
import json
import math
from pathlib import Path
import re
import shutil

import numpy as np

from run_cmfgen_formal_primary import fields
from prepare_cmfgen_mom_capture import sha
from ptf_gated_metrics import ScrtempReader, read_dimensions


def exponential_cell_moments(a):
    """F_m(a)=int_0^1 y^m exp(-a*y)dy, m=0,1,2. 작은 a도 상쇄 없이 계산."""
    a = np.asarray(a, dtype=float)
    if not np.isfinite(a).all() or np.any(a < 0):
        raise ValueError('흡수 optical depth 오류')
    result = np.empty((3,) + a.shape)
    small = a <= .5
    x = a[small]
    for m in range(3):
        term = np.ones_like(x)
        total = term / (m + 1)
        for k in range(1, 20):
            term *= -x / k
            total += term / (m + k + 1)
        result[m][small] = total
    x = a[~small]
    e = np.exp(-x)
    result[0][~small] = -np.expm1(-x) / x
    result[1][~small] = (result[0][~small] - e) / x
    result[2][~small] = (2 * result[1][~small] - e) / x
    return result


def boundary_mode(chi, source, planck, dbdtau, radius, width, order=64):
    """고정 chi/S 정적 평면층. r^2는 (r_core+x)^2로 정확히 적분한다.

    delta I(x,mu)=[B-S+mu*dB/dtau]*exp(-chi*x/mu), 0<mu<1.
    반환 source는 eta-chi*J의 부호다. 실제 J/H는 입력으로 받지 않는다.
    """
    chi, source, planck, dbdtau = np.broadcast_arrays(chi, source, planck, dbdtau)
    if np.any(chi <= 0) or width <= 0 or radius <= 0:
        raise ValueError('정적 경계층 입력 오류')
    tau = chi * width
    x, w = np.polynomial.legendre.leggauss(order)
    mu = (x + 1) / 2; w = w / 2
    a = tau[..., None] / mu
    f0, f1, f2 = exponential_cell_moments(a)
    amplitude = (planck - source)[..., None] + dbdtau[..., None] * mu
    integral = -.5 * tau * np.sum(w * amplitude *
        (radius**2 * f0 + 2 * radius * width * f1 + width**2 * f2), axis=-1)
    j0 = .5 * (planck - source) + .25 * dbdtau
    j1 = .5 * np.sum(w * amplitude * np.exp(-a), axis=-1)
    trap = -.5 * width * chi * (radius**2 * j0 + (radius + width)**2 * j1)
    return integral, trap, j0, j1


def diffusion_boundary(nu15, temperature_1e4, dtdr):
    # native new_main/cmfgen.f:119--120, subs/phys_con.f, comp_j_blank.f:140--144.
    h, k, c = 6.62607015e-27, 1.380649e-16, 2.99792458e10
    hdkt = 1e11 * h / k; twohcsq = 2e45 * h / c**2
    x = hdkt * nu15 / temperature_1e4
    e = np.exp(-x); denominator = -np.expm1(-x)
    b = twohcsq * nu15**3 * e / denominator
    dbdr = twohcsq * nu15**3 * x * dtdr / temperature_1e4 * e / denominator**2
    return b, dbdr


def weighted_quantiles(values, weights):
    if np.any(weights < 0) or not np.isfinite(values).all() or not np.isfinite(weights).all():
        raise ValueError('경계층 구조 가중치 오류')
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    if cumulative[-1] == 0:
        return None
    return np.interp(np.array([.5, .95, .99]) * cumulative[-1], cumulative, values[order]).tolist()


def interior_structure(geometry, data):
    r, v, sigma, gamma = geometry
    dln = np.r_[np.log(data['freq'][0] / data['freq'][1]), -np.diff(np.log(data['freq']))]
    result = []
    for depth in (32, 35, 39, 49, 50, 51, 52, 89, 90):
        i = depth - 1
        j, h, k, nn, chi, eta, scat, source_j = np.array(data['field'][:, :, i]).T
        local = eta + scat * source_j - chi * j
        weights = data['weight'] * abs(local)
        adjacent = []
        for neighbor in (i - 1, i + 1):
            if 0 <= neighbor < len(r):
                adjacent.append(abs(r[i] - r[neighbor]) * .5 * (chi + data['field'][:, 4, neighbor]))
        tau = np.min(adjacent, axis=0)
        alpha = gamma[i]**3 * abs(v[i] / 299792.458) / (r[i] * dln * chi)
        result.append(dict(depth_1based=depth, signed_source_density=float(r[i]**2 * (data['weight'] @ local)),
            absolute_spectral_source_density=float(r[i]**2 * weights.sum()),
            weighting='모든 주파수의 FQW*abs(eta_total-chi*J); 공간적분/실제잔차 판정 아님',
            adjacent_tau_min_median_p95_p99=weighted_quantiles(tau, weights),
            scattering_fraction_median_p95_p99=weighted_quantiles(scat / chi, weights),
            homologous_alpha_over_chi_scale_median_p95_p99=weighted_quantiles(alpha, weights)))
    return result


def analyze(run, seed, energy, out):
    if out.exists():
        raise ValueError('기존 경계층 진단 보존')
    runtime = json.loads((run / 'seq_logs/whole-runtime.json').read_text())
    if not runtime.get('diagnostic_complete') or runtime.get('returncode') != 97:
        raise ValueError('부분 장 소비 금지')
    if sha(seed / 'SCRTEMP') != sha(run / 'SCRTEMP'):
        raise ValueError('새 장의 물질59 신원 변경')
    if not re.search(r'^\s*DIFFUSION\s+\[IB_METH\]', (run / 'MODEL').read_text(), re.M):
        raise ValueError('diffusion 내부경계만 지원')
    geometry, data = fields(run / 'FORMAL_PRIMARY_FIELDS.bin')
    r, v, sigma, gamma = geometry
    nd, nt = read_dimensions(seed)
    reader = ScrtempReader(seed / 'SCRTEMP', nd, nt)
    if reader.last_it != 59 or len(data) != 197291:
        raise ValueError('이번59/60 diagnostic 신원 불일치')
    dtdr = float(re.search(r'The value of DTDR is\s*:\s*(\S+)', (run / 'OUTGEN').read_text())[1])
    width = float(r[-2] - r[-1]); radius = float(r[-1])
    temperature = float(reader.state(59)[nt - 1, -1])
    b, dbdr = diffusion_boundary(data['freq'], temperature, dtdr)
    core = np.array(data['field'][:, :, -1])
    neighbor = np.array(data['field'][:, :, -2])
    j, h, k, nn, chi, eta, scat, source_j = core.T
    source = (eta + scat * source_j) / chi
    dbdtau = dbdr / chi
    estimates = {}; arrays = []
    for order in (32, 64):
        exact = np.empty(len(data)); trap = np.empty_like(exact)
        j0 = np.empty_like(exact); j1 = np.empty_like(exact)
        for start in range(0, len(data), 2048):
            sl = slice(start, start + 2048)
            exact[sl], trap[sl], j0[sl], j1[sl] = boundary_mode(
                chi[sl], source[sl], b[sl], dbdtau[sl], radius, width, order)
        estimates[str(order)] = dict(source_integral=float(data['weight'] @ exact),
            point_trapezoid=float(data['weight'] @ trap),
            trapezoid_minus_independent_integral=float(data['weight'] @ (trap - exact)))
        if order == 64:
            arrays = [exact, trap, j0, j1]
    exact, trap, j0, j1 = arrays
    predicted_j = source + j0
    actual_cooling = data['weight'] * chi * (j - source)
    norm = float(np.sum(abs(actual_cooling)))
    core_mismatch = float(np.sum(data['weight'] * chi * abs(j - predicted_j)))
    source_neighbor = (neighbor[:, 5] + neighbor[:, 6] * neighbor[:, 7]) / neighbor[:, 4]
    dln = np.r_[np.log(data['freq'][0] / data['freq'][1]), -np.diff(np.log(data['freq']))]
    alpha_over_chi = gamma[-1]**3 * abs(v[-1] / 299792.458) / (radius * dln * chi)
    significance = np.argsort(abs(data['weight'] * (trap - exact)))[-32:][::-1]
    points = [dict(index_1based=int(i + 1), freq_native=float(data['freq'][i]),
        tau_core=float(chi[i] * width), scattering_fraction=float(scat[i] / chi[i]),
        alpha_over_chi_scale=float(alpha_over_chi[i]),
        source_core=float(source[i]), source_neighbor=float(source_neighbor[i]),
        J_actual_core=float(j[i]), J_static_prediction_core=float(predicted_j[i]),
        signed_trapezoid_bias=float(data['weight'][i] * (trap[i] - exact[i]))) for i in significance]
    previous = json.loads((energy / 'energy-ledger.json').read_text())
    correction = estimates['64']['trapezoid_minus_independent_integral']
    signed_after = previous['summary']['signed_defect'] + correction
    out.mkdir()
    np.savez(out / 'core-layer.npz', frequency=data['freq'], weight=data['weight'], chi=chi,
        source=source, source_neighbor=source_neighbor, planck=b, dbdtau=dbdtau,
        actual_J=j, predicted_J=predicted_j, independent_integral=exact, point_trapezoid=trap)
    source_paths = [Path(__file__).resolve(), Path(fields.__code__.co_filename).resolve()]
    for p in source_paths:
        shutil.copy2(p, out / p.name)
    report = dict(scope='실제 local source/diffusion 경계로 정적 평면 흡수층의 적분 근사를 분리; 실제 수송 폐합 아님',
        assumptions='chi/S 고정·정적·평면·0<mu<1. 팽창/blue·source의 내부 변화·외부 incoming 보정은 포함하지 않음.',
        no_flux_used_to_determine_source=True, all_frequencies_included=len(data),
        radius=radius, width=width, beta_core=float(v[-1] / 299792.458),
        temperature_1e4_K=temperature, dtdr_native=dtdr, estimates=estimates,
        angular_Cauchy_difference=estimates['64']['source_integral'] - estimates['32']['source_integral'],
        core_local_residual_mismatch_L1_fraction=core_mismatch / norm,
        original_signed_defect=previous['summary']['signed_defect'],
        approximate_mode_quadrature_correction=correction,
        remaining_signed_defect_if_only_this_static_mode_is_corrected=signed_after,
        remaining_fraction_of_original_net_boundary=signed_after / abs(previous['summary']['net_boundary']),
        dominant_frequencies=points, energy_pass='NOT_JUDGED', final_goal='NOT_JUDGED',
        interior_structure=interior_structure(geometry, data),
        input_sha256={str(p): sha(p) for p in (run / 'FORMAL_PRIMARY_FIELDS.bin', run / 'MODEL', run / 'OUTGEN',
            seed / 'SCRTEMP', energy / 'energy-ledger.json', *source_paths)},
        output_sha256=sha(out / 'core-layer.npz'))
    with (out / 'core-layer.json').open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({key: report[key] for key in ('estimates', 'core_local_residual_mismatch_L1_fraction',
        'remaining_fraction_of_original_net_boundary')}, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'seed', 'energy', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    analyze(*(getattr(a, name).resolve() for name in ('run', 'seed', 'energy', 'out')))
