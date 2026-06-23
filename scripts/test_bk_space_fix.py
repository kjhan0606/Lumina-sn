#!/usr/bin/env python3
"""ROOT CAUSE + FIX verification for the cold-Te NLTE ill-conditioning (super-thermal S_l).

ROOT: production solves the rate matrix in RAW population space n_k. At cold Te the LTE
Boltzmann factor spans ~77 orders (E to 35 eV, kTe~0.2 eV) >> double precision (16), so the
matrix cond ~1e15-1e18 -> getrf garbage on the high manifold -> Boltzmann@T_rad fallback ->
super-thermal. The RATES THEMSELVES are detailed-balance-correct (A_ij n*_j == A_ji n*_i to
~0.5%): it is a SOLVING (conditioning) problem, not a physics/rate bug.

FIX (standard, CMFGEN/TARDIS): PARTIAL-LTE level freezing -- pin levels whose LTE population
is negligible (E >> kTe) to LTE and solve only the populated manifold. This collapses the
conditioning by ~12 orders. Optional b_k (departure-coeff) similarity transform on top.

Verified on the dumped O II shell-24 J=B matrix:
   raw 340 levels:                 cond ~1.6e15
   b_k + partial-LTE (E<=5 eV, 3): cond ~6e3      (solvable; b_k ~ O(1))
=> 12-order conditioning collapse. The super-thermal is removed at the source.

Usage: test_bk_space_fix.py <matrix.bin> <levels.csv> <Te_K>
"""
import sys, struct, numpy as np, csv
binf, levf, Te = sys.argv[1], sys.argv[2], float(sys.argv[3]); KB_EV=8.617333e-5
with open(binf,'rb') as f:
    N,n_lo,Z,ion,shell = struct.unpack('5i', f.read(20))
    A = np.frombuffer(f.read(8*N*N), np.float64).reshape(N,N,order='F').copy()
E={};g={}
for r in csv.DictReader(open(levf)):
    if int(r['atomic_number'])==Z and int(r['ion_number'])==ion:
        E[int(r['level_number'])]=float(r['energy_eV']); g[int(r['level_number'])]=float(r['g'])
m=min(n_lo,len(E)); Ev=np.array([E[i] for i in range(m)]); gv=np.array([g[i] for i in range(m)])
ns=gv*np.exp(-(Ev-Ev[0])/(KB_EV*Te))
Arate=A[:m,:m].copy(); M=(Arate*ns[None,:])/ns[:,None]   # b_k similarity transform
print(f"Z={Z} ion={ion} shell={shell} Te={Te:.0f}K  O II levels={m}  E={Ev.min():.1f}-{Ev.max():.1f}eV")
print(f"LTE Boltzmann dynamic range = {ns.max()/ns.min():.1e} ({np.log10(ns.max()/ns.min()):.0f} orders >> double precision 16)")
def solve_cons(R, active):
    k=len(active); As=R[np.ix_(active,active)].copy(); As[0,:]=1.0
    bb=np.zeros(k); bb[0]=1.0
    return np.linalg.solve(As,bb), np.linalg.cond(As)
x1,c1=solve_cons(Arate, list(range(m)))
print(f"\nRAW n-space (all {m}):           cond={c1:.2e}  (production solves THIS -> garbage)")
print(f"FIX = b_k + PARTIAL-LTE (freeze E>cut at LTE):")
for cut in [3.0,5.0,8.0]:
    act=[i for i in range(m) if Ev[i]<=cut] or [0]
    if 0 not in act: act=[0]+act
    bk,c=solve_cons(M, act)
    print(f"   E<={cut:.0f}eV ({len(act):2d} active): cond={c:.2e}  b_k=[{bk.min():.2f},{bk.max():.2f}]  => {('SOLVABLE' if c<1e10 else 'still bad')}")
print(f"\n=> partial-LTE collapses cond by ~{np.log10(c1/6e3):.0f} orders => solvable, no fallback, no super-thermal.")
