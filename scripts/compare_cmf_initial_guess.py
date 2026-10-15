#!/usr/bin/env python3
"""동일 자체 물질에서 binary 수정 뒤 다시 푼 장과 BF 광전리 유출을 비교한다.

미수렴 초기해 진단이다. 물리 시간/인구 수렴/CMFGEN 일치를 판정하지 않는다.
BF 적분은 bf-field-resolution-424404에서 C 소비율과 대조한 식을 재사용한다.
"""
import argparse
import csv
import hashlib
import json
import math
import struct
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('old', type=Path)
    parser.add_argument('new', type=Path)
    parser.add_argument('production', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    hashes = {}

    def read(path):
        data = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(data).hexdigest()
        return data

    def checkpoint(root):
        raw = read(root/'iterate_input.bin')
        assert raw[:8] == b'CMFITER1' and raw[-8:] == b'ITEREND1'
        h = struct.unpack_from('<7Q', raw, 72)
        ns, ni, nl, ne, nf, generation, iteration = h
        assert 0 < ns <= 256 and min(ni, nl, ne, nf, generation) > 0
        parameters = struct.unpack_from('<3d', raw, 128)
        start = 152+4*(nl+2*ni)
        assert len(raw) == start+8*ns*(5+ne+ni+nl+nf)+8
        state_end = start+8*ns*(5+ne+ni+nl)
        maps = np.frombuffer(raw, '<i4', nl+2*ni, 152)
        pop = np.frombuffer(raw, '<f8', ns*nl, start+8*ns*(5+ne+ni)).reshape(nl, ns)
        ions = np.frombuffer(raw, '<f8', ns*ni, start+8*ns*(5+ne)).reshape(ni, ns)
        J = np.frombuffer(raw, '<f8', ns*nf, state_end).reshape(ns, nf)
        assert np.all(np.isfinite(J)) and np.all(J >= 0)
        identity = read(root/'iterate_identity.txt')
        assert raw[8:72].decode() == hashlib.sha256(identity).hexdigest()
        return dict(h=h, parameters=parameters, maps=maps, material=raw[152:state_end],
                    identity=identity, pop=pop, ions=ions, J=J)

    a, b = checkpoint(args.old), checkpoint(args.new)
    assert a['h'][:5] == b['h'][:5] and b['h'][6] == 0
    assert a['parameters'] == b['parameters'] and a['material'] == b['material']
    assert a['identity'] != b['identity']
    assert a['identity'].splitlines()[1:] == b['identity'].splitlines()[1:]
    assert read(args.old/'input.sha256') == read(args.new/'input.sha256')
    assert read(args.production/'guess_source_identity.txt') == a['identity']
    assert read(args.production/'iterate_identity.txt') == b['identity']
    read(args.production/'restart_input.bin')
    assert hashes[str(args.production/'restart_input.bin')] == hashes[str(args.old/'iterate_input.bin')]
    provenance = dict(line.split('=', 1) for line in read(args.production/'start_provenance.txt').decode().splitlines())
    assert provenance['start'] == 'guess'
    assert provenance['source_identity'] == hashlib.sha256(a['identity']).hexdigest()
    assert provenance['run_identity'] == hashlib.sha256(b['identity']).hexdigest()
    ns, ni, nl, ne, nf = a['h'][:5]
    epoch, fmin, dlog = a['parameters']
    assert epoch == 172800 and fmin > 0 and dlog > 0
    fe = fmin*np.exp(np.arange(nf+1)*dlog)
    widths = np.diff(fe)
    old_J, new_J = a['J'], b['J']
    difference = np.abs(new_J-old_J)
    scale = np.maximum(old_J, new_J)
    relative = np.divide(difference, scale, out=np.zeros_like(scale), where=scale > 0)
    shells = []
    for s in range(ns):
        x, y = float(old_J[s]@widths), float(new_J[s]@widths)
        norm = max(x, y)
        shells.append(dict(shell=s, old_integral=x, new_integral=y,
            relative_integral_change=abs(y-x)/norm if norm else 0.,
            relative_L1=float(difference[s]@widths)/norm if norm else 0.))
    s, k = np.unravel_index(np.argmax(relative), relative.shape)
    field = dict(max_relative=dict(shell=int(s), bin=int(k), value=float(relative[s,k]),
        old=float(old_J[s,k]), new=float(new_J[s,k]), frequency_hz=float(math.sqrt(fe[k]*fe[k+1]))),
        max_absolute=float(np.max(difference)), shells=shells)

    inputs = args.new/'input'
    level_data = read(inputs/'levels.csv').decode()
    levels = list(csv.DictReader(level_data.splitlines()))
    assert len(levels) == nl
    raw = read(inputs/'bf_channels.bin')
    magic, version, count, nr, nb, lo, hi = struct.unpack_from('<IIiiidd', raw)
    assert magic == 0x43424643 and version == 2 and count == nl and nb == 1234
    pos = 36
    offset = np.frombuffer(raw, '<i4', nl+1, pos); pos += 4*(nl+1)
    pos += 4*nr  # 광전리만 비교하므로 목적 준위는 소비하지 않는다.
    threshold = np.frombuffer(raw, '<f8', nr, pos); pos += 8*nr
    sigma = np.frombuffer(raw, '<f8', nr*nb, pos).reshape(nr, nb); pos += 8*nr*nb
    assert pos+8*nr == len(raw) and offset[0] == 0 and offset[-1] == nr
    edges = lo*np.exp(np.arange(nb+1)*math.log(hi/lo)/nb); edges[0] = lo; edges[-1] = hi
    assert fe[0] <= lo and fe[-1] >= hi
    union = np.unique(np.concatenate((edges, fe[(fe > lo)&(fe < hi)])))
    left, right = union[:-1], union[1:]
    mid = (left+right)*.5
    cb = np.searchsorted(edges, mid, side='right')-1
    fb = np.searchsorted(fe, mid, side='right')-1
    width = np.diff(edges)
    coarse = [np.array([np.bincount(cb, weights=j[fb]*(right-left), minlength=nb)/width for j in J])
              for J in (old_J, new_J)]
    gx, gw = np.polynomial.legendre.leggauss(8); gw *= .5
    index = {int(gl): i for i, gl in enumerate(a['maps'][:nl])}
    totals = {z: np.zeros((2, ns)) for z in (14, 16)}
    for gl, row in enumerate(levels):
        z, q = int(row['atomic_number']), int(row['ion_number'])
        if z not in totals or q != 1:
            continue
        coefficient = np.zeros(nb)
        for r in range(int(offset[gl]), int(offset[gl+1])):
            valid = edges[1:] > threshold[r]
            assert not np.any(sigma[r, ~valid])
            bins = np.flatnonzero(valid)
            l, u = np.maximum(edges[bins], threshold[r]), edges[bins+1]
            frequency = (l+u)[:,None]*.5+(u-l)[:,None]*.5*gx
            coefficient[bins] += 4*math.pi/6.62607015e-27*sigma[r,bins]*width[bins]*np.sum(gw/frequency, axis=1)
        for i in range(2):
            totals[z][i] += a['pop'][index[gl]]*(coarse[i]@coefficient)
    rates = []
    for z, values in totals.items():
        for s in range(ns):
            x, y = map(float, values[:,s]); norm = max(x, y)
            assert math.isfinite(x) and math.isfinite(y) and min(x,y) >= 0
            rates.append(dict(Z=z, shell=s, old_cm3_s=x, new_cm3_s=y,
                relative_change=y/x-1 if x else None, symmetric_relative_change=abs(y-x)/norm if norm else 0.))
    read(Path(__file__))
    result = dict(status='SAME_MATERIAL_NEW_POLICY_FIELD', physical_history=False,
        goal_verdict='NOT_JUDGED', completed_iterations=[a['h'][6], b['h'][6]],
        material_byte_identical=True, radiation_recomputed_by_production=True,
        photoionization_only=True, recombination_or_population_solved=False,
        field=field, photoionization=rates, sha256=hashes)
    args.output.mkdir(exist_ok=False)
    with (args.output/'comparison.json').open('x') as out:
        json.dump(result, out, indent=2, allow_nan=False); out.write('\n')
    print('SAME_MATERIAL_NEW_POLICY_FIELD')
    print('J_integral', max(shells, key=lambda x:x['relative_integral_change']))
    print('J_L1', max(shells, key=lambda x:x['relative_L1']))
    for r in rates:
        if 39 <= r['shell'] <= 44:
            print(json.dumps(r))


if __name__ == '__main__':
    main()
