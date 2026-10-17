"""Ca II 역전의 두 BB 경로만 국소 Planck 율로 바꾸는 조건부 인과 대조.
실제 복사장/원자료를 수정하지 않으며, BF/충돌/NT와 나머지 BB는 고정한다.
변경 선의 흡수와 유도방출을 함께 바꾼다. 생산 보정안이나 새 수송해가 아니다.
"""
import csv
import json
import math
import struct
import sys
from pathlib import Path
import numpy as np

root = Path(sys.argv[1])
baseline = json.loads((root/'inversion_flow.json').read_text())
assert baseline['Z'] == 20 and baseline['line'] == 2139366
raw = (root/'matrix_Z20.bin').read_bytes()
n = struct.unpack_from('<Q', raw, 8)[0]
a = np.frombuffer(raw, '<f8', n*n, 24).reshape(n, n, order='F').astype(np.longdouble)
saved = np.frombuffer(raw, '<f8', n, 24+8*n*n).astype(np.longdouble)
lines = list(csv.DictReader((root/'lines_Z20.csv').open()))
T = baseline['T_K']

def stationary(matrix):
    m = matrix.copy()
    np.fill_diagonal(m, 0)
    assert np.all(m >= 0) and np.all(np.isfinite(m))
    for k in range(n-1, 0, -1):
        out = m[:k, k].sum()
        assert out > 0
        m[k, :k] /= out
        m[:k, :k] += m[:k, k, None]*m[k, None, :k]
    x = np.ones(n, dtype=np.longdouble)
    for k in range(1, n):
        x[k] = np.sum(m[k, :k]*x[:k])
    return x/x.sum()

def ratio(x):
    return float((x[22]/10)/(x[19]/14))

original = stationary(a)
assert abs(ratio(original)/ratio(saved)-1) < 1e-10
variants = []
for name, pairs in [('5p_to_8d', {(5,22)}), ('7f_to_3d', {(1,19)}),
                    ('both_paths', {(5,22), (1,19)})]:
    m = a.copy()
    changed = []
    for q in lines:
        l, u = int(q['lower_matrix']), int(q['upper_matrix'])
        if (l,u) not in pairs:
            continue
        nu = float(q['nu'])
        B = 2*6.62607015e-27*nu**3/2.99792458e10**2/math.expm1(6.62607015e-27*nu/(1.380649e-16*T))
        J = np.longdouble(q['Jbar'])
        for target, source, weight, coefficient in (
            (u,l,q['w_lower'],q['B_lu']), (l,u,q['w_upper'],q['B_ul'])):
            delta = np.longdouble(weight)*np.longdouble(coefficient)*(np.longdouble(B)-J)
            m[target,source] += delta
            m[source,source] -= delta
        changed.append(dict(line=int(q['line']), wavelength_A=2.99792458e18/nu,
                            Jbar_over_Bnu=float(J/B)))
    assert len(changed) == 2*len(pairs)
    x = stationary(m)
    off = m.copy()
    np.fill_diagonal(off, 0)
    np.fill_diagonal(m, -off.sum(axis=0))
    scale = abs(m)@x
    selected = scale > 0
    residual = float(np.max(abs(m@x)[selected]/scale[selected]))
    assert residual < 1e-10
    variants.append(dict(name=name, changed_lines=changed,
                         weighted_population_ratio=ratio(x),
                         normalized_rate_residual=residual))
output = dict(scope='conditional BB-only Planck replacement; frozen other rates; NOT a production correction or common-field solution',
              baseline_weighted_ratio=ratio(original), variants=variants)
with (root/'inversion_pump.json').open('x') as f:
    json.dump(output,f,indent=2)
print(json.dumps(output,indent=2))
