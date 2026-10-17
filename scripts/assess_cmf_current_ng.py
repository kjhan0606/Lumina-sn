"""1252 물질 반복열의 Ng 후보 적합성만 평가한다. 생산 상태는 쓰지 않는다."""
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import struct

BASE = Path('/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results')
RUNS = [BASE / f'toy-material-{j}' for j in (426040, 426074, 426219, 426303)]
EXPECTED = (
    'af4f48c2c3f0ea74a0d9a9476bf76d006852cf40b7dffd6d751e6a9b1520da9c',
    '266d4bc7b1b0d886782224505aab3789d09c96f2d6b7aca378f2dafbd0d0065d',
    '6483fce8bfb5572e9a283b9e70747b36b0313a735529e69e825def211aa38a87',
    '2c1305b91aee47763a7060c4538ac6e65d3e27424128bf613074f3320d3879b1',
)
marker = 'target=[2.,3.,5.,7.]'
source = Path('source/original.py').read_text()
assert source.count(marker) == 1
helper = {}
exec(compile(source.split(marker)[0], 'source/original.py', 'exec'), helper)
runs = [helper['read'](r) for r in RUNS]
for root, digest in zip(RUNS, EXPECTED, strict=True):
    assert helper['hashes'][str(root / 'iterate_input.bin')] == digest
assert hashlib.sha256(runs[0]['identity']).hexdigest() == 'cf74990bb51488a0dce4b9cccafcdc3321ba2956b5dde56fda2d3829eb037b38'
ns, ni, nl, ne, nb = runs[0]['header'][:5]
assert (ns, ni, nl, ne, nb) == (113, 27, 19627, 6, 303316)
const = (4 + ne) * ns
level_start = const + (1 + ni) * ns
for i, run in enumerate(runs):
    assert run['header'][6] == (0, 0, 1, 2)[i]
    assert run['header'][:5] == runs[0]['header'][:5]
    for key in ('maps', 'par', 'identity', 'input_manifest'):
        assert run[key] == runs[0][key], key
    assert run['values'][:const] == runs[0]['values'][:const]
assert all(a['header'][5] < b['header'][5] for a, b in zip(runs, runs[1:]))

# 기존 보존/감쇠 조건을 그대로 재사용하고 과거 입력에 대한 실행부는 제외한다.
old_tree = ast.parse(Path('source/assess_damped.py').read_text())
selected = [n for n in old_tree.body if isinstance(n, ast.FunctionDef) and n.name in ('constraints', 'clean')]
assert len(selected) == 2
exec(compile(ast.Module(body=selected, type_ignores=[]), 'source/assess_damped.py', 'exec'))
maps = struct.unpack('<' + str(nl + 2 * ni) + 'i', runs[0]['maps'])
zs, stages = maps[nl:nl+ni], maps[nl+ni:]
with (RUNS[-1] / 'input/levels.csv').open() as f:
    levels = list(csv.DictReader(f))
groups = [[] for _ in range(ni)]
ion_keys = list(zip(zs, stages))
for fl, gl in enumerate(maps[:nl]):
    key = int(levels[gl]['atomic_number']), int(levels[gl]['ion_number'])
    groups[ion_keys.index(key)].append(fl)
elements = {z: [i for i in range(ni) if zs[i] == z] for z in set(zs)}
assert len(elements) == 6 and sum(not x for x in groups) == 6
latest = list(runs[-1]['values'][const:])
for run in runs:
    check = constraints(run['values'][const:], latest)
    assert check['nonnegative'] and check['level_sum_relative'] <= 1e-10
    assert check['nuclei_relative'] <= 1e-10 and check['charge_relative'] <= 1e-8

samples = []
for run in reversed(runs):
    v = run['values']
    samples.append([x for s in range(ns-1, -1, -1)
                    for x in [v[const+s], *(v[level_start+fl*ns+s] for fl in range(nl))]])
proposal = helper['ng'](samples)
assert all(math.isfinite(x) for x in proposal)
A1 = B1 = B2 = C1 = C2 = 0.
for x, y, z, u in zip(*samples, strict=True):
    w = 1/x if x else 1.
    d0 = x-y
    d1, d2 = d0+(z-y), d0+(u-z)
    A1 += w*d1*d1
    B1 += w*d1*d2
    C1 += w*d0*d1
    B2 += w*d2*d2
    C2 += w*d0*d2
div1, div2 = B2*A1, B1*B1
assert all(math.isfinite(x) for x in (div1, div2))
assert abs(div1-div2) > 1e-10*max(abs(div1), abs(div2)), '계수식 조건이 나쁨'
A, B = (B2*C1-B1*C2)/(div1-div2), (A1*C2-B1*C1)/(div1-div2)
formula_error = 0.
for x, y, z, p in zip(samples[0], samples[1], samples[2], proposal, strict=True):
    den = max(abs(p), abs(x))
    err = abs(x+A*(y-x)+B*(z-x)-p)
    formula_error = max(formula_error, err/den if den else err)
assert formula_error < 1e-10
del proposal, samples
previous, earlier = runs[-2]['values'][const:], runs[-3]['values'][const:]
direction = [A*(y-x)+B*(z-x) for x, y, z in zip(latest, previous, earlier, strict=True)]
records, accepted = [], None
for power in range(11):
    alpha = 2.**(-power)
    candidate = [x+alpha*d for x, d in zip(latest, direction, strict=True)]
    record = dict(alpha=alpha, **constraints(candidate, latest))
    records.append(record)
    if record['admissible']:
        accepted = dict(alpha=alpha)
        break
summary = dict(status='NUMERICAL_CANDIDATE_ONLY', goal_verdict='NOT_JUDGED',
    A=A, B=B, fortran_formula_relative=formula_error, records=records, accepted=accepted,
    candidate_written=False, production_changed=False, new_radiation=False, new_SE=False,
    physical_history=False, convergence_acceleration_measured=False,
    material_state_indices=[0, 1, 2, 3], completed_iterations=[r['header'][6] for r in runs],
    generations=[r['header'][5] for r in runs], input_sha256=helper['hashes'])
with Path('current_ng.json').open('x') as f:
    json.dump(clean(summary), f, indent=2, allow_nan=False)
    f.write('\n')
print(json.dumps(clean(summary), allow_nan=False))
