"""보존된 율행렬에서 III/II의 개별 선 복사장 민감도를 분리한다.

고정 T/ne/연속광/충돌/NT의 조건부 미분이다. 새 수송이나 생산 보정이 아니다.
수반 미분을 실제 비음수 GTH 해의 중앙차분으로 검증한다.
사용법: python3 cmf_ion_pump_sensitivity.py SAVED_RESPONSE UNIQUE_JSON [OTHER_RESPONSE]
"""
import csv
import hashlib
import json
import math
import re
import struct
import sys
from pathlib import Path

import numpy as np


def stationary(a):
    m = np.array(a, dtype=np.longdouble, copy=True)
    np.fill_diagonal(m, 0)
    assert np.all(np.isfinite(m)) and np.all(m >= 0)
    for k in range(len(m) - 1, 0, -1):
        out = m[:k, k].sum()
        assert out > 0
        m[k, :k] /= out
        m[:k, :k] += m[:k, k, None] * m[k, None, :k]
    x = np.ones(len(m), dtype=np.longdouble)
    for k in range(1, len(m)):
        x[k] = (m[k, :k] * x[:k]).sum()
    x /= x.sum()
    m = conservative(a)
    scale = abs(m) @ x
    used = scale > 0
    residual = float(np.max(abs(m @ x)[used] / scale[used]))
    assert residual < 1e-10
    return x, residual


def conservative(a):
    m = np.array(a, dtype=np.longdouble, copy=True)
    np.fill_diagonal(m, 0)
    np.fill_diagonal(m, -m.sum(axis=0))
    return m


assert len(sys.argv) in (3, 4)
root, output = map(Path, sys.argv[1:3])
other_root = Path(sys.argv[3]) if len(sys.argv) == 4 else None
assert not output.exists()
inputs = dict(line.split('=', 1) for line in (root / 'inputs.txt').read_text().splitlines())
atom_path = Path(inputs['reader']) / 'input/levels.csv'
atom = list(csv.DictReader(atom_path.open()))
log = (root / 'response.log').read_text()
temperature = float(re.search(r'SL_RESPONSE_RESULT[^\n]*T=([^ ]+)', log)[1])
files = [root / 'inputs.txt', root / 'response.log', atom_path]
result = {'scope': '고정 T/ne/연속광/충돌/NT; 선 J만 변경; 공동수렴/CMFGEN 인과 확인 아님',
          'root': str(root), 'T_K': temperature, 'ions': {}}
if other_root:
    result['other_root'] = str(other_root)
    files.extend((other_root / 'inputs.txt', other_root / 'response.log'))
for z in (14, 16):
    matrix_path = root / f'matrix_Z{z}.bin'
    level_path = root / f'levels_Z{z}.csv'
    line_path = root / f'lines_Z{z}.csv'
    files.extend((matrix_path, level_path, line_path))
    raw = matrix_path.read_bytes()
    assert raw[:8] == b'SLMAT01\0'
    n = struct.unpack_from('<Q', raw, 8)[0]
    total = struct.unpack_from('<d', raw, 16)[0]
    assert len(raw) == 24 + 8 * (n*n+n)
    a = np.frombuffer(raw, '<f8', n*n, 24).reshape(n, n, order='F')
    saved = np.frombuffer(raw, '<f8', n, 24+8*n*n) / total
    stage = np.full(n, -1)
    meta = list(csv.DictReader(level_path.open()))
    for row in meta:
        i, charge = int(row['matrix_index']), int(row['charge'])
        assert stage[i] in (-1, charge)
        stage[i] = charge
    assert stage[-1] == -1 and np.count_nonzero(stage < 0) == 1
    stage[-1] = stage.max() + 1
    ratio = lambda x: float(x[stage == 2].sum() / x[stage == 1].sum())
    x, residual = stationary(a)
    baseline = ratio(x)
    error = abs(baseline / ratio(saved) - 1)
    assert error < 1e-10

    # A dx = -dA x, sum(dx)=0. q dx = d ln(nIII/nII).
    # 행 스케일링/정규화 행 치환은 선형 풀이에만 쓰고 물리 율은 바꾸지 않는다.
    aa = np.array(conservative(a), dtype=float)
    scales = np.max(abs(aa), axis=1)
    assert np.all(scales > 0)
    normalized = aa / scales[:, None]
    pivot = int(np.argmax(x))
    normalized[pivot] = 1
    q = np.zeros(n)
    q[stage == 2] = 1 / float(x[stage == 2].sum())
    q[stage == 1] = -1 / float(x[stage == 1].sum())
    adjoint = np.linalg.solve(normalized.T, q)
    adjoint[pivot] = 0
    adjoint /= scales
    terms = []
    perturbation = np.zeros((n, n), dtype=np.longdouble)
    transplanted = np.array(a, dtype=np.longdouble, copy=True)
    field_prediction = 0.0
    if other_root:
        other_path = other_root / f'lines_Z{z}.csv'
        files.append(other_path)
        other_lines = {int(row['line']): row for row in csv.DictReader(other_path.open())}
    for row in csv.DictReader(line_path.open()):
        l, u = int(row['lower_matrix']), int(row['upper_matrix'])
        if l == u:
            continue
        wl, wu, j, blu, bul = [np.longdouble(row[k]) for k in
                              ('w_lower', 'w_upper', 'Jbar', 'B_lu', 'B_ul')]
        up, down = wl*blu*j, wu*bul*j
        net = up*x[l] - down*x[u]
        sensitivity = float(-(adjoint[u]-adjoint[l])*net)
        if other_root:
            other = other_lines[int(row['line'])]
            for key in ('charge', 'lower_global', 'upper_global', 'nu', 'B_lu', 'B_ul', 'A_ul'):
                assert row[key] == other[key], (z, row['line'], key)
            next_j = np.longdouble(other['Jbar'])
            transplanted[u, l] += wl*blu*(next_j-j)
            transplanted[l, u] += wu*bul*(next_j-j)
            if j > 0:
                field_prediction += sensitivity*float((next_j-j)/j)
            else:
                assert next_j == 0
        perturbation[u, l] += up
        perturbation[l, u] += down
        gl, gu = int(row['lower_global']), int(row['upper_global'])
        nu = float(row['nu'])
        exponent = 6.62607015e-27*nu/(1.380649e-16*temperature)
        planck = 2*6.62607015e-27*nu**3/2.99792458e10**2/math.expm1(exponent)
        terms.append(dict(line=int(row['line']), charge=int(row['charge']),
                          lower_global=gl, upper_global=gu, lower_matrix=l, upper_matrix=u,
                          lower=atom[gl]['configuration'], upper=atom[gu]['configuration'],
                          lower_eV=float(atom[gl]['energy_eV']), upper_eV=float(atom[gu]['energy_eV']),
                          wavelength_A=2.99792458e18/nu, J_over_B=float(j/planck),
                          dln_ratio_dln_J=sensitivity, upward=float(up), downward_stim=float(down)))
    terms.sort(key=lambda row: -abs(row['dln_ratio_dln_J']))
    checks = []
    for name, delta, predicted in [('all_lines', perturbation, sum(t['dln_ratio_dln_J'] for t in terms))] + [
        (f"line_{t['line']}", None, t['dln_ratio_dln_J']) for t in terms[:3]]:
        if delta is None:
            t = next(t for t in terms if name == f"line_{t['line']}")
            delta = np.zeros_like(perturbation)
            delta[t['upper_matrix'], t['lower_matrix']] = t['upward']
            delta[t['lower_matrix'], t['upper_matrix']] = t['downward_stim']
        epsilon = 1e-4
        plus, _ = stationary(a + epsilon*delta)
        minus, _ = stationary(a - epsilon*delta)
        measured = (math.log(ratio(plus))-math.log(ratio(minus))) / (2*epsilon)
        assert abs(measured-predicted) < 1e-6*max(1, abs(predicted)), (name, measured, predicted)
        checks.append(dict(path=name, adjoint=predicted, finite_difference=measured))
    grouped = {}
    for t in terms:
        key = str(t['upper_matrix'])
        if key not in grouped:
            grouped[key] = dict(upper_matrix=t['upper_matrix'], charge=t['charge'],
                                representative=t['upper'], sensitivity=0)
        grouped[key]['sensitivity'] += t['dln_ratio_dln_J']
    result['ions'][z] = dict(baseline_III_II=baseline, C_reproduction_error=error,
                             normalized_rate_residual=residual, derivative_checks=checks,
                             top_lines=terms[:20],
                             upper_groups=sorted(grouped.values(), key=lambda row: -abs(row['sensitivity']))[:12])
    if other_root:
        replaced, replacement_residual = stationary(transplanted)
        result['ions'][z]['other_BB_J_only'] = dict(
            III_II=ratio(replaced), relative_change=ratio(replaced)/baseline-1,
            linear_dln_prediction=field_prediction, normalized_rate_residual=replacement_residual,
            held_fixed='baseline populations within SL, T, ne, continuum, collision and NT rates')
        other_matrix = other_root / f'matrix_Z{z}.bin'
        other_levels = other_root / f'levels_Z{z}.csv'
        files.extend((other_matrix, other_levels))
        assert other_levels.read_bytes() == level_path.read_bytes()
        raw_other = other_matrix.read_bytes()
        assert raw_other[:8] == b'SLMAT01\0' and struct.unpack_from('<Q', raw_other, 8)[0] == n
        assert len(raw_other) == len(raw)
        aa_other = np.frombuffer(raw_other, '<f8', n*n, 24).reshape(n, n, order='F')
        same_ion = stage[:, None] == stage[None, :]
        swaps = []
        for label, selection in [('interstage_total', ~same_ion), ('intrastage_total', same_ion),
                                 ('entire_other_matrix', np.ones((n, n), dtype=bool))]:
            candidate = np.where(selection, aa_other, a)
            xx, rr = stationary(candidate)
            swaps.append(dict(path=label, III_II=ratio(xx), relative_change=ratio(xx)/baseline-1,
                              normalized_rate_residual=rr))
        result['ions'][z]['matrix_block_transfer'] = swaps
result['sha256'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
result['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
with output.open('x') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
print(json.dumps(result, indent=2, ensure_ascii=False))
