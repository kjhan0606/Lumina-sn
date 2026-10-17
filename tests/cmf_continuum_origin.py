"""한 셸 실제 수송 입력을 광전리 기여로 가중해 소광/방출 성분을 읽는다.
국소 계수 비중이지 광자 기원 추적이나 CMFGEN과의 인과 대조가 아니다.
"""
import csv
import ctypes as ct
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

import numpy as np

root = Path(sys.argv[1])
base = Path('/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results')
prior = base/'toy-material-425642'
response = base/'saved-field-response-20261017.vNeZQn'
bridge = base/'prrr-comparison-20261017.1y1nJD'
raw = (root/'fine_shell39.bin').read_bytes()
assert raw[:8] == b'FNMAT01\0'
nf = struct.unpack_from('<Q', raw, 8)[0]
nu, chi, es, eta, j = np.frombuffer(raw, '<f8', 5*nf, 16).reshape(5, nf)
origin = (root/'origin_shell39.bin').read_bytes()
assert origin[:8] == b'ORIGIN01'
nb, nf2 = struct.unpack_from('<QQ', origin, 8)
lo, dl, T, ne, z2ni = struct.unpack_from('<5d', origin, 24)
assert nf == nf2 and len(origin) == 64+8*(14*nb+12*nf)
coarse = np.frombuffer(origin, '<f8', 14*nb, 64).reshape(14, nb)
lines = np.frombuffer(origin, '<f8', 12*nf, 64+8*14*nb).reshape(12, nf)
grid = lo*np.exp((np.arange(nb)+.5)*dl)
continuum = np.array([np.interp(np.log(nu), np.log(grid), row) for row in coarse])
red = nu < lo
continuum[:12, red] = 0
h, k, c, ev, me = 6.62607015e-27, 1.380649e-16, 2.99792458e10, 1.602176634e-12, 9.1093837015e-28
if np.any(red):
    coefficient = 3.6926e8/math.sqrt(T)*ne*z2ni
    continuum[12, red] = coefficient/nu[red]**3*(-np.expm1(-h*nu[red]/(k*T)))
    continuum[13, red] = coefficient*(2*h/c**2)*np.exp(-h*nu[red]/(k*T))
chi_parts = np.vstack([lines[::2], continuum[::2], es])
eta_parts = np.vstack([lines[1::2], continuum[1::2], es*j])
labels = [f'BB_Z{z}' for z in (14,16,20,26,27,28)] + [f'BF_Z{z}' for z in (14,16,20,26,27,28)] + ['FF', 'electron_scattering']
cs, ee = chi_parts.sum(axis=0), eta_parts.sum(axis=0)
assert np.allclose(cs, chi, rtol=1e-10, atol=0)
assert np.allclose(ee, eta+es*j, rtol=1e-10, atol=1e-300)

pack_path = prior/'input/bf_channels.bin'
pack = pack_path.read_bytes()
assert hashlib.sha256(pack).hexdigest() == '656329de676b540a08beb1e6652e839785f799428a40b31fd2774ce44d8af0cf'
magic, version, nlevels, nr, nbb, blo, bhi = struct.unpack_from('<IIiiidd', pack)
assert magic == 0x43424643 and version == 2 and nb == nbb and blo == lo
pos = 36
offset = np.frombuffer(pack, '<i4', nlevels+1, pos); pos += 4*(nlevels+1)
upper = np.frombuffer(pack, '<i4', nr, pos); pos += 4*nr
threshold = np.frombuffer(pack, '<f8', nr, pos); pos += 8*nr
sigma = np.frombuffer(pack, '<f8', nr*nb, pos).reshape(nr, nb)
edges = np.ascontiguousarray(lo*np.exp(np.arange(nb+1)*dl))
edges[-1] = bhi
fdl = math.log(nu[1]/nu[0])
fine_edges = np.r_[nu*np.exp(-.5*fdl), nu[-1]*np.exp(.5*fdl)]
union = np.unique(np.r_[edges, fine_edges[(fine_edges > blo) & (fine_edges < bhi)]])
left, right = union[:-1], union[1:]
mid = .5*(left+right)
cb = np.searchsorted(edges, mid, side='right')-1
fb = np.searchsorted(fine_edges, mid, side='right')-1
assert np.all((fb >= 0) & (fb < nf))
atom = list(csv.DictReader((prior/'input/levels.csv').open()))
library = bridge/'comparison.so'
assert hashlib.sha256(library.read_bytes()).hexdigest() == (bridge/'binary.sha256').read_text().split()[0]
lib = ct.CDLL(str(library))
ptr = ct.POINTER(ct.c_double)
lib.bf_vector.argtypes = [ct.c_int, ptr, ct.c_double, ptr]+[ct.c_double]*4+[ptr]*3
lib.bf_vector.restype = ct.c_int
def p(x):
    assert x.dtype == np.float64 and x.flags.c_contiguous
    return x.ctypes.data_as(ptr)
ones = np.ones(nb)
result = {'scope': '셸39 첫 저장장/물질의 국소 계수, 고정장 SE 광전리 기여 가중; 새 공동해/광자 기원/CMFGEN 장 비교 아님', 'T': T, 'ne': ne, 'ions': {}}
for z in (14, 16):
    buf = (response/f'matrix_Z{z}.bin').read_bytes()
    n = struct.unpack_from('<Q', buf, 8)[0]
    x = np.frombuffer(buf, '<f8', n, 24+8*n*n)
    meta = {int(row['global_level']): row for row in csv.DictReader((response/f'levels_Z{z}.csv').open())}
    kernel = np.zeros(nb)
    for lower, row in meta.items():
        if int(row['charge']) != 1:
            continue
        pop = float(x[int(row['matrix_index'])])*float(row['within'])
        for r in range(offset[lower], offset[lower+1]):
            u = int(upper[r]); assert u >= 0
            su = np.ascontiguousarray(sigma[r]); up, down = np.empty(nb), np.empty(nb)
            assert lib.bf_vector(nb, p(edges), float(threshold[r]), p(su), T, ne,
                                 float(atom[lower]['g']), float(atom[u]['g']), p(ones), p(up), p(down)) == 0
            kernel += pop*up
    weight = np.bincount(fb, weights=kernel[cb]/np.diff(edges)[cb]*j[fb]*(right-left), minlength=nf)
    energy = nu*h/ev
    rows = []
    for emin, emax in ((0, 5), (5, 10), (10, 16), (16, 24), (24, 40)):
        chosen = (energy >= emin) & (energy < emax)
        w = weight[chosen]; total = w.sum()
        assert total > 0 and np.all(chi[chosen] > 0) and np.all(ee[chosen] > 0)
        fractions = (chi_parts[:, chosen]/chi[chosen] * w).sum(axis=1)/total
        sources = (eta_parts[:, chosen]/ee[chosen] * w).sum(axis=1)/total
        rows.append(dict(eV=[emin, emax], ionization_weight=float(total/weight.sum()),
                         mean_chi=float(np.dot(w, chi[chosen])/total),
                         opacity_shares=dict(zip(labels, map(float, fractions))),
                         source_shares=dict(zip(labels, map(float, sources)))))
    result['ions'][z] = rows
files = [root/'fine_shell39.bin', root/'origin_shell39.bin', root/'origin.log', library, pack_path]
result['sha256'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
result['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
with (root/'origin_summary.json').open('x') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
print(json.dumps(result, indent=2, ensure_ascii=False))
