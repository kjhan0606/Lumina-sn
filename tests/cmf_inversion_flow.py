"""저장된 정상 율행렬에서 역전 선을 연결하는 인구 흐름을 읽는다.
새 인구해/수송/생산 변경은 없다. BB 이외 항은 합계로 남기며 광전리로 단정하지 않는다.
"""
import csv
import json
import math
import re
import struct
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
line_id = int(sys.argv[2])
z = int(sys.argv[3])
raw = (root / f'matrix_Z{z}.bin').read_bytes()
assert raw[:8] == b'SLMAT01\x00'
n = struct.unpack_from('<Q', raw, 8)[0]
total = struct.unpack_from('<d', raw, 16)[0]
assert len(raw) == 24 + 8 * (n*n+n)
a = np.frombuffer(raw, '<f8', n*n, 24).reshape(n, n, order='F').astype(np.longdouble)
x = np.frombuffer(raw, '<f8', n, 24+8*n*n).astype(np.longdouble)
assert np.all(np.isfinite(a)) and np.all(x >= 0) and np.all(np.isfinite(x))
assert abs(x.sum()/total-1) < 1e-10
meta = list(csv.DictReader((root / f'levels_Z{z}.csv').open()))
lines = list(csv.DictReader((root / f'lines_Z{z}.csv').open()))
parameters = re.search(r'SL_RESPONSE_RESULT[^\n]*T=([^ ]+) ne=([^\s]+)',
                       (root / 'response.log').read_text())
assert parameters
temperature, electron_density = map(float, parameters.groups())
inputs = dict(q.split('=', 1) for q in (root / 'inputs.txt').read_text().splitlines())
atomic = list(csv.DictReader((Path(inputs['reader']) / 'input/levels.csv').open()))
matches = [q for q in lines if int(q['line']) == line_id]
assert len(matches) == 1
line = matches[0]
lo, up = int(line['lower_matrix']), int(line['upper_matrix'])
assert lo != up
bb = np.zeros_like(a)
for q in lines:
    l, u = int(q['lower_matrix']), int(q['upper_matrix'])
    if l == u:
        continue
    wl, wu, j, blu, bul, au = [np.longdouble(q[k]) for k in
                               ('w_lower', 'w_upper', 'Jbar', 'B_lu', 'B_ul', 'A_ul')]
    bb[u, l] += wl * blu * j
    bb[l, u] += wu * (au + bul*j)
off = a.copy()
np.fill_diagonal(off, 0)
assert np.all(off >= 0)
rest = off-bb
flux = off*x[None, :]
flux_bb = bb*x[None, :]
flux_other = rest*x[None, :]
scale = (abs(a)*x[None, :]).sum(axis=1)
residual = a@x
active = scale > 0
max_residual = float(np.max(abs(residual[active])/scale[active]))
assert max_residual < 1e-10

def description(i):
    return [dict(global_level=int(q['global_level']), charge=int(q['charge']),
                 configuration=atomic[int(q['global_level'])]['configuration'],
                 energy_eV=float(q['energy_eV']), g=int(q['g']), within=float(q['within']))
            for q in meta if int(q['matrix_index']) == i]

def flows(i):
    net = flux[i, :]-flux[:, i]
    ordering = np.argsort(-abs(net))[:12]
    return dict(matrix_index=i, levels=description(i), population=float(x[i]),
                incoming=float(flux[i, :].sum()), outgoing=float(flux[:, i].sum()),
                BB_net_in=float(flux_bb[i, :].sum()-flux_bb[:, i].sum()),
                other_net_in=float(flux_other[i, :].sum()-flux_other[:, i].sum()),
                largest_pair_net_in=[dict(other_index=int(k), levels=description(int(k)),
                    net_in=float(net[k]), BB_net_in=float(flux_bb[i,k]-flux_bb[k,i]),
                    other_net_in=float(flux_other[i,k]-flux_other[k,i])) for k in ordering])

lower = np.longdouble(line['w_lower'])*x[lo]
upper = np.longdouble(line['w_upper'])*x[up]
gl = next(int(q['g']) for q in meta if int(q['global_level']) == int(line['lower_global']))
gu = next(int(q['g']) for q in meta if int(q['global_level']) == int(line['upper_global']))
j, blu, bul, au = [np.longdouble(line[k]) for k in ('Jbar', 'B_lu', 'B_ul', 'A_ul')]
nu = float(line['nu'])
planck = (2*6.62607015e-27*nu**3/2.99792458e10**2 /
          math.expm1(6.62607015e-27*nu/(1.380649e-16*temperature)))
result = dict(scope='fixed saved J/T/ne/NT, not the live electron-closed material or a coupled solution',
              Z=z, line=line_id, matrix_states=n, normalized_rate_residual=max_residual,
              T_K=temperature, ne=electron_density, Jbar_over_Bnu=float(j/planck),
              lower_population=float(lower), upper_population=float(upper),
              weighted_population_ratio=float((upper/gu)/(lower/gl)),
              Jbar=float(j), line_net_down=float(upper*(au+bul*j)-lower*blu*j),
              min_nonBB_offdiagonal_rate=float(rest.min()),
              lower=flows(lo), upper=flows(up))
with (root / 'inversion_flow.json').open('x') as output:
    json.dump(result, output, indent=2)
print(json.dumps(result, indent=2))
