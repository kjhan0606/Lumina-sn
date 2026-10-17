"""이전 완료 반복0→1의 BF 장 변화와 준위·광자 에너지 기여를 분해한다.
같은 물질/T/ne에서 생산 bf_channel_bin을 호출한다. 새 SE/RTE는 없다.
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

base = Path('/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results')
prior = base/'toy-material-425626'
response = base/'saved-field-response-20261017.6I2cmg'
bridge = base/'prrr-comparison-20261017.1y1nJD'
destination = Path(sys.argv[1])
assert not destination.exists()
sources = {}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    sources[str(path)] = h.hexdigest()
    return h.hexdigest()


pack_path = prior/'input/bf_channels.bin'
assert digest(pack_path) == '656329de676b540a08beb1e6652e839785f799428a40b31fd2774ce44d8af0cf'
atom_path = prior/'input/levels.csv'
digest(atom_path)
atom = list(csv.DictReader(atom_path.open()))
pack = pack_path.read_bytes()
magic, version, nlevels, nr, nb, blo, bhi = struct.unpack_from('<IIiiidd', pack)
assert magic == 0x43424643 and version == 2 and nlevels == len(atom)
pos = 36
offset = np.frombuffer(pack, '<i4', nlevels+1, pos); pos += 4*(nlevels+1)
upper = np.frombuffer(pack, '<i4', nr, pos); pos += 4*nr
threshold = np.frombuffer(pack, '<f8', nr, pos); pos += 8*nr
sigma = np.frombuffer(pack, '<f8', nr*nb, pos).reshape(nr, nb)
edges = np.ascontiguousarray(blo*np.exp(np.arange(nb+1)*math.log(bhi/blo)/nb))
edges[-1] = bhi
width = np.diff(edges)
fields = []
headers = []
for path, expected, iteration in (
    (prior/'iterate_input.bin', '45c27c94608dca44602562a38ccab26f69bde36b693aa66ddc96413e744cee31', 0),
    (base/'baseline-425609-20261017.45Maft/work/initial_iterate.bin',
     'bff7f172e3efb158b2efa3bcda39613b102d2ee0a99c7ba5fbe3118447b364a2', 1)):
    assert digest(path) == expected
    buf = np.memmap(path, dtype='u1', mode='r')
    assert bytes(buf[:8]) == b'CMFITER1'
    ns, ni, nl, nelem, nf, generation, it = struct.unpack_from('<7Q', buf, 72)
    epoch, lo, dl = struct.unpack_from('<3d', buf, 128)
    assert ns == 113 and it == iteration and epoch == 172800
    start = 152+4*(nl+2*ni)
    T = struct.unpack_from('<d', buf, start+8*(3*ns+39))[0]
    ne = struct.unpack_from('<d', buf, start+8*((4+nelem)*ns+39))[0]
    headers.append(dict(iteration=it, T=T, ne=ne, identity=bytes(buf[8:72]).decode()))
    j = np.frombuffer(buf, '<f8', ns*nf, start+8*ns*(5+nelem+ni+nl)).reshape(ns, nf)[39]
    fe = lo*np.exp(np.arange(nf+1)*dl)
    assert fe[0] <= blo and fe[-1] >= bhi
    union = np.unique(np.r_[edges, fe[(fe > blo) & (fe < bhi)]])
    left, right = union[:-1], union[1:]
    mid = .5*(left+right)
    cb = np.searchsorted(edges, mid, side='right')-1
    fb = np.searchsorted(fe, mid, side='right')-1
    fields.append(np.ascontiguousarray(np.bincount(cb, weights=j[fb]*(right-left), minlength=nb)/width))
    assert np.max(abs(np.bincount(cb, weights=right-left, minlength=nb)/width-1)) < 1e-12
assert headers[0]['identity'] == headers[1]['identity']
assert headers[0]['T'] == headers[1]['T']
T, ne = headers[0]['T'], headers[0]['ne']
library_path = bridge/'comparison.so'
expected_library = (bridge/'binary.sha256').read_text().split()[0]
assert digest(library_path) == expected_library
lib = ct.CDLL(str(library_path))
ptr = ct.POINTER(ct.c_double)
lib.bf_vector.argtypes = [ct.c_int, ptr, ct.c_double, ptr]+[ct.c_double]*4+[ptr]*3
lib.bf_vector.restype = ct.c_int


def pointer(x):
    assert x.dtype == np.float64 and x.flags.c_contiguous
    return x.ctypes.data_as(ptr)


h, ev = 6.62607015e-27, 1.602176634e-12
energies = np.sqrt(edges[:-1]*edges[1:])*h/ev
bands = [(0, 5), (5, 10), (10, 16), (16, 24), (24, 40), (40, math.inf)]
out = dict(scope='동일 이전 바이너리 반복0→1; 고정 반복0 T/ne/개체수; BF 연속광만 교환',
           shell=39, fields=headers, T_fixed=T, ne_fixed=ne, ions={})
for z, expected_channels in ((14, 250), (16, 322)):
    matrix_path, level_path = response/f'matrix_Z{z}.bin', response/f'levels_Z{z}.csv'
    digest(matrix_path); digest(level_path)
    raw = matrix_path.read_bytes()
    n = struct.unpack_from('<Q', raw, 8)[0]
    x = np.frombuffer(raw, '<f8', n, 24+8*n*n)
    meta = {int(row['global_level']): row for row in csv.DictReader(level_path.open())}
    pop = {i: float(x[int(row['matrix_index'])])*float(row['within']) for i, row in meta.items()}
    ni = [sum(pop[i] for i, row in meta.items() if int(row['charge']) == charge) for charge in (1, 2)]
    flow = np.zeros((2, 2, nb))
    levels = []
    count = 0
    for lower, row in meta.items():
        if int(row['charge']) != 1:
            continue
        contribution = np.zeros((2, 2))
        for r in range(offset[lower], offset[lower+1]):
            u = int(upper[r])
            assert u in meta and int(meta[u]['charge']) == 2
            su = np.ascontiguousarray(sigma[r])
            for index, field in enumerate(fields):
                up, down = np.empty(nb), np.empty(nb)
                assert lib.bf_vector(nb, pointer(edges), float(threshold[r]), pointer(su), T, ne,
                                     float(atom[lower]['g']), float(atom[u]['g']),
                                     pointer(field), pointer(up), pointer(down)) == 0
                flow[index, 0] += pop[lower]*up
                flow[index, 1] += pop[u]*down
                contribution[index] += [pop[lower]*up.sum(), pop[u]*down.sum()]
            count += 1
        levels.append(dict(lower_global=lower, configuration=atom[lower]['configuration'],
                           energy_eV=float(atom[lower]['energy_eV']),
                           photo_before_after=contribution[:, 0].tolist(),
                           recombination_before_after=contribution[:, 1].tolist()))
    assert count == expected_channels
    totals = flow.sum(axis=2)
    effective = (totals[:, 0]/ni[0])/(totals[:, 1]/ni[1])
    band_results = []
    for low, high in bands:
        selection = (energies >= low) & (energies < high)
        values = flow[:, :, selection].sum(axis=2)
        band_results.append(dict(eV=[low, high if math.isfinite(high) else None],
                                 photo_before_after=values[:, 0].tolist(),
                                 recombination_before_after=values[:, 1].tolist(),
                                 photo_fraction_before=float(values[0, 0]/totals[0, 0]),
                                 contribution_to_total_photo_change=float((values[1, 0]-values[0, 0])/totals[0, 0])))
    out['ions'][z] = dict(channels=count, flow_before_after=totals.tolist(),
                          photo_relative_change=float(totals[1, 0]/totals[0, 0]-1),
                          recombination_relative_change=float(totals[1, 1]/totals[0, 1]-1),
                          fixed_distribution_BF_ratio=effective.tolist(), bands=band_results,
                          top_levels=sorted(levels, key=lambda row: -row['photo_before_after'][0])[:12])
out['sha256'] = sources
out['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
with destination.open('x') as f:
    json.dump(out, f, indent=2, ensure_ascii=False, allow_nan=False)
print(json.dumps(out, indent=2, ensure_ascii=False, allow_nan=False))
