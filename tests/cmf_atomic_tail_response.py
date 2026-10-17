"""같은 실제 J/T/ne/비열율에서 고준위 전이망의 조건부 효과를 분리한다.
공개 마지막 준위에 해당하는 수까지 남기는 절단 시험이며 공개 원자모형의
완전한 재현이 아니다. 삭제 상태와 연결된 순·역전이는 함께 제거한다.
생산 원자료/해를 변경하지 않고 새 수송도 하지 않는다.
"""
import csv
import json
import struct
from pathlib import Path

import numpy as np


def stationary(a):
    """열 방향 생성행렬의 비음수 GTH 소거. 대각 상쇄를 풀이에 쓰지 않는다."""
    m = np.array(a, dtype=np.longdouble, copy=True)
    np.fill_diagonal(m, 0)
    assert np.all(m >= 0) and np.all(np.isfinite(m))
    n = len(m)
    for k in range(n - 1, 0, -1):
        total = np.sum(m[:k, k])
        assert total > 0, ('reducible', k)
        m[k, :k] /= total
        m[:k, :k] += m[:k, k, None] * m[k, None, :k]
    x = np.ones(n, dtype=np.longdouble)
    for k in range(1, n):
        x[k] = np.sum(m[k, :k] * x[:k])
    x /= np.sum(x)
    assert np.all(x >= 0) and abs(x.sum() - 1) < 1e-15
    return x


# 연결 제거 후의 대각은 남은 유출만으로 재구성한다.
def conservative(a):
    a = np.array(a, dtype=np.longdouble, copy=True)
    np.fill_diagonal(a, 0)
    np.fill_diagonal(a, -a.sum(axis=0))
    return a


root = Path('.')
prior = Path('/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/toy-material-425144')
atom = list(csv.DictReader((prior / 'input/levels.csv').open()))
output = {'scope': 'native iteration4 shell39; original BB domain; full levels; fixed J/T/ne/NT; conditional model-tail removal; NOT public model reproduction',
          'new_RTE': False, 'production_changed': False, 'ions': {}}
for z, cuts in ((14, {1: 62, 2: 61, 3: 48}), (16, {2: 98, 3: 67})):
    raw = (root / f'matrix_Z{z}.bin').read_bytes()
    assert raw[:7] == b'SLMAT01'
    n = struct.unpack_from('<Q', raw, 8)[0]
    total = struct.unpack_from('<d', raw, 16)[0]
    a = np.frombuffer(raw, '<f8', n*n, 24).reshape(n, n, order='F')
    saved = np.frombuffer(raw, '<f8', n, 24+8*n*n) / total
    stage = np.full(n, -1)
    ordinal = np.full(n, -1)
    meta = {}
    for row in csv.DictReader((root / f'levels_Z{z}.csv').open()):
        i, gl = int(row['matrix_index']), int(row['global_level'])
        assert stage[i] == -1 and float(row['within']) == 1
        assert int(atom[gl]['atomic_number']) == z
        stage[i] = int(row['charge'])
        ordinal[i] = int(atom[gl]['level_number'])
        meta[i] = atom[gl]['configuration']
    assert np.count_nonzero(stage < 0) == 1 and stage[-1] == -1
    stage[-1] = stage.max() + 1
    full = stationary(a)
    ratio = lambda x, st: float(x[st == 2].sum()/x[st == 1].sum())
    baseline = ratio(full, stage)
    error = abs(baseline / ratio(saved, stage) - 1)
    assert error < 1e-8, (z, error)
    assert abs(ratio(stationary(conservative(a)), stage)/baseline-1) < 1e-12
    variants = [(f'charge_{charge}_only', {charge: count}) for charge, count in cuts.items()]
    variants.append(('all_selected_tails', cuts))
    results = []
    for name, limits in variants:
        keep = np.ones(n, dtype=bool)
        endpoints = {}
        for charge, count in limits.items():
            indices = np.flatnonzero(stage == charge)
            assert np.array_equal(ordinal[indices], np.arange(len(indices)))
            keep &= ~((stage == charge) & (ordinal >= count))
            endpoints[charge] = {'retained_count': count, 'original_count': len(indices),
                                 'last_retained': meta[int(indices[count-1])],
                                 'first_removed': meta[int(indices[count])]}
        reduced = conservative(a[np.ix_(keep, keep)])
        x = stationary(reduced)
        flux = abs(reduced) @ x
        residual = reduced @ x
        valid = flux > 0
        normalized = float(np.max(abs(residual[valid])/flux[valid]))
        assert normalized < 1e-10
        value = ratio(x, stage[keep])
        results.append({'name': name, 'endpoints': endpoints, 'removed_states': int((~keep).sum()),
                        'removed_population_before': float(full[~keep].sum()),
                        'III_II': value, 'relative_change': value/baseline-1,
                        'normalized_rate_residual': normalized})
    output['ions'][z] = {'full_states': n, 'full_III_II': baseline,
                         'C_solution_relative_error': error, 'variants': results}
with (root / 'tail_summary.json').open('x') as f:
    json.dump(output, f, indent=2)
print(json.dumps(output, indent=2))
