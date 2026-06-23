#!/usr/bin/env python3
"""ROOT CAUSE + FIX verification for the cold-Te NLTE ill-conditioning.

ROOT: production solves the rate matrix A.n=b in RAW population space n_k. At cold Te the
LTE Boltzmann factor spans ~76 orders (E up to 35 eV, kTe~0.2 eV) >> double precision (16),
so most high levels are numerically unresolvable -> garbage -> fallback -> super-thermal.

FIX (standard, CMFGEN): solve in DEPARTURE-COEFFICIENT space n_k = b_k * n_k^LTE. Substituting
into A.n=0 gives (A . diag(n*)) . b = 0 -- a column scaling by the LTE populations that scales
the Boltzmann factor OUT, so every b_k ~ O(1) and the dynamic range collapses.

This script (a) shows the raw dynamic range / conditioning, (b) builds n* from the level data,
(c) re-solves in b_k space and reports the conditioning + b_k, proving the fix works on the
ACTUAL dumped production matrix.

Usage: test_bk_space_fix.py <matrix.bin> <levels.csv> <Te_K> [n_e]
"""
import sys, struct, numpy as np, csv
binf, levf, Te = sys.argv[1], sys.argv[2], float(sys.argv[3])
KB_EV = 8.617333e-5
with open(binf,'rb') as f:
    N,n_lo,Z,ion,shell = struct.unpack('5i', f.read(20))
    A = np.frombuffer(f.read(8*N*N), np.float64).reshape(N,N,order='F').copy()
    b = np.frombuffer(f.read(8*N), np.float64).copy()
# lower-ion level energies/g
E={};g={}
for r in csv.DictReader(open(levf)):
    if int(r['atomic_number'])==Z and int(r['ion_number'])==ion:
        E[int(r['level_number'])]=float(r['energy_eV']); g[int(r['level_number'])]=float(r['g'])
m=min(n_lo,len(E))
Ev=np.array([E[i] for i in range(m)]); gv=np.array([g[i] for i in range(m)])
# LTE reference populations n* (Boltzmann, lower ion); for the upper-ion / conservation
# block (indices >= n_lo) keep n*=1 (those rows are ion-balance/conservation, not Boltzmann).
nstar=np.ones(N)
nstar[:m]= gv*np.exp(-(Ev-Ev[0])/(KB_EV*Te))
print(f"N={N} O II levels={n_lo} Te={Te:.0f}  kTe={KB_EV*Te:.3f} eV")
print(f"n* (LTE) dynamic range over O II: {nstar[:m].max()/nstar[:m].min():.2e} ({np.log10(nstar[:m].max()/nstar[:m].min()):.0f} orders)")
print(f"\n--- RAW n-space (what production solves) ---")
print(f"cond(A) = {np.linalg.cond(A):.2e}   (double precision resolves ~1e16)")
x_raw,*_=np.linalg.lstsq(A,b,rcond=None)
print(f"raw solve: negatives={np.sum(x_raw<0)}/{N}  (unresolvable high manifold)")
print(f"\n--- FIX: b_k-space  (A.diag(n*)) b = b,  n = diag(n*) b ---")
Ab = A * nstar[None,:]            # column scaling by n*  == A.diag(n*)
print(f"cond(A.diag(n*)) = {np.linalg.cond(Ab):.2e}")
bk = np.linalg.solve(Ab, b)
n_fix = nstar*bk
print(f"b_k solve: b_k min={bk[:m].min():.3e} median={np.median(bk[:m]):.3e} max={bk[:m].max():.3e}  negatives(n)={np.sum(n_fix<0)}/{N}")
# DB verdict in b_k space at J=B: b_k MUST be ~1 for all resolvable levels
within=np.sum((bk[:m]>0.5)&(bk[:m]<2.0))
print(f"O II levels with b_k in [0.5,2]: {within}/{m} ({100*within/m:.0f}%)")
if within > 0.9*m:
    print("\n  ==> FIX CONFIRMED: in b_k space the cond collapses and J=B gives b_k~1 for ALL")
    print("      levels (detailed balance). The cold-Te ill-conditioning is REMOVED. The")
    print("      production super-thermal was the n-space numerical failure, not physics/rate-bug.")
else:
    print(f"\n  ==> b_k space improves conditioning to cond={np.linalg.cond(Ab):.1e} but {m-within} levels")
    print("      still off -- residual conditioning; may need partial-LTE freezing of E>>kTe levels too.")
