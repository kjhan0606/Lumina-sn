"""두 보존 저장점의 J 변화 크기/위치만 읽는다. 수송이나 인구해는 없다."""
import hashlib
import json
import math
import struct
import sys
from pathlib import Path
import numpy as np

def read(path):
    path = Path(path)
    with path.open('rb') as f:
        head = f.read(152)
        f.seek(-8, 2)
        assert f.read() == b'ITEREND1'
    assert head[:8] == b'CMFITER1'
    ns, ni, nl, nelem, nf, generation, iteration = struct.unpack_from('<7Q',head,72)
    epoch, nu_min, dlog = struct.unpack_from('<3d',head,128)
    offset = 152+4*(nl+2*ni)+8*ns*(5+nelem+ni+nl)
    assert path.stat().st_size == offset+8*ns*nf+8
    j = np.memmap(path,dtype='<f8',mode='r',offset=offset,shape=(ns,nf))
    with path.open('rb') as f:
        h = hashlib.file_digest(f,'sha256').hexdigest()
    return j, dict(path=str(path), sha256=h, ns=ns,nf=nf,epoch=epoch,
                   nu_min=nu_min,dlog=dlog,generation=generation,iteration=iteration)

old, h0 = read(sys.argv[1])
new, h1 = read(sys.argv[2])
for key in ('ns','nf','epoch','nu_min','dlog'):
    assert h0[key] == h1[key]
nu = h0['nu_min']*np.exp((np.arange(h0['nf'])+.5)*h0['dlog'])
dnu = nu*h0['dlog']
energy = 6.62607015e-27*nu/1.602176634e-12
worst_rel = None
worst_abs = None
peak = None
shells = []
for s in range(h0['ns']):
    a,b = old[s],new[s]
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and np.all(a>=0) and np.all(b>=0)
    diff = abs(b-a)
    relative = np.zeros_like(a)
    np.divide(diff,a,out=relative,where=a>0)
    relative[(a==0)&(diff>0)] = np.inf
    k = int(np.argmax(relative))
    if worst_rel is None or relative[k] > worst_rel['relative_change']:
        worst_rel = dict(shell=s,bin=k,relative_change=float(relative[k]),
                         old_J=float(a[k]),new_J=float(b[k]),frequency_Hz=float(nu[k]),
                         energy_eV=float(energy[k]))
    k = int(np.argmax(diff))
    if worst_abs is None or diff[k] > worst_abs['absolute_change']:
        worst_abs = dict(shell=s,bin=k,absolute_change=float(diff[k]),
                         old_J=float(a[k]),new_J=float(b[k]),frequency_Hz=float(nu[k]))
    k = int(np.argmax(b))
    if peak is None or b[k] > peak['J']:
        peak = dict(shell=s,bin=k,J=float(b[k]),frequency_Hz=float(nu[k]))
    if s in (39,43):
        bands = []
        for label,lo,hi in [('all',0,math.inf),('IR',0,.5),('optical_nearUV',.5,6),
                           ('6_16eV',6,16),('16_24eV',16,24),('24_50eV',24,50)]:
            selected = (energy>=lo)&(energy<hi)
            before = float(np.sum(a[selected]*dnu[selected]))
            after = float(np.sum(b[selected]*dnu[selected]))
            bands.append(dict(band=label,integral_old=before,integral_new=after,
                              ratio=after/before if before else None))
        shells.append(dict(shell=s,bands=bands))
output = dict(scope='saved J comparison only; not luminosity, common convergence, or CMFGEN equivalence',
              old=h0,new=h1,worst_relative=worst_rel,worst_absolute=worst_abs,new_peak=peak,shells=shells)
outpath=Path(sys.argv[3])
with outpath.open('x') as f:
    json.dump(output,f,indent=2)
print(json.dumps(output,indent=2))
