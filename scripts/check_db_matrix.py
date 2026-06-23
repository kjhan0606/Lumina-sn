#!/usr/bin/env python3
"""Detailed-balance verdict on a dumped J=B rate matrix (LUMINA_NLTE_MATDUMP+JEQB).
THEOREM: with J=B(Te) and DB collisions/recomb, the SE solution MUST be Boltzmann@Te
(within-ion levels) + Saha@Te (ion ratio). If not, a rate VIOLATES detailed balance = bug.

Usage: check_db_matrix.py <matrix.bin> <levels.csv> <Te_K> [n_e_cm3]
"""
import sys, struct, numpy as np

binf, levf, Te = sys.argv[1], sys.argv[2], float(sys.argv[3])
ne = float(sys.argv[4]) if len(sys.argv) > 4 else None
KB_EV = 8.617333e-5  # eV/K

with open(binf, 'rb') as f:
    N, n_lo, Z, ion, shell = struct.unpack('5i', f.read(20))
    A = np.frombuffer(f.read(8*N*N), dtype=np.float64).reshape(N, N, order='F').copy()
    b = np.frombuffer(f.read(8*N), dtype=np.float64).copy()
print(f"matrix: N={N} n_lo(lower-ion levels)={n_lo} Z={Z} ion={ion} shell={shell} Te={Te:.0f}K")

# solve A x = b  -> equilibrium populations
try:
    x = np.linalg.solve(A, b)
except np.linalg.LinAlgError:
    x, *_ = np.linalg.lstsq(A, b, rcond=None); print("  (singular -> lstsq)")
print(f"  solution: min={x.min():.3e} max={x.max():.3e}  negatives={np.sum(x<0)}")

# load lower-ion level energies/g (Z, ion)
import csv
E = []; g = []
for r in csv.DictReader(open(levf)):
    if int(r['atomic_number']) == Z and int(r['ion_number']) == ion:
        E.append(float(r['energy_eV'])); g.append(float(r['g']))
E = np.array(E); g = np.array(g)
m = min(n_lo, len(E))
print(f"  lower-ion levels with atomic data: {len(E)} (using {m})")

# within-ion DB check: x_i/x_0 vs Boltzmann (g_i/g_0) exp(-(E_i-E_0)/kTe)
xi = x[:m].copy()
ref = xi[0] if xi[0] != 0 else (xi[xi>0][0] if np.any(xi>0) else 1.0)
boltz = (g[:m]/g[0]) * np.exp(-(E[:m]-E[0])/(KB_EV*Te))
pop_ratio = xi/ (xi[0] if xi[0]!=0 else 1.0)
# departure coefficient b_k = (x_i/x_0) / boltz_i  (should be ~1 for ALL if DB clean)
with np.errstate(divide='ignore', invalid='ignore'):
    bk = pop_ratio / boltz
good = np.isfinite(bk) & (boltz > 0) & (xi > 0)
if good.sum() > 1:
    bkv = bk[good]
    print(f"\n  === DEPARTURE b_k = (x_i/x_0)/Boltzmann_i  (MUST be ~1 if DB clean) ===")
    print(f"  b_k: min={bkv.min():.3e}  median={np.median(bkv):.3e}  max={bkv.max():.3e}  (n={good.sum()})")
    nbad = np.sum((bkv < 0.5) | (bkv > 2.0))
    print(f"  levels with |log b_k|>0.3 (b_k outside [0.5,2]): {nbad}/{len(bkv)} ({100*nbad/len(bkv):.0f}%)")
    print(f"  worst-departed levels (level_idx, E_eV, g, b_k):")
    order = np.argsort(-np.abs(np.log(np.maximum(bkv,1e-30))))
    idxs = np.where(good)[0]
    for k in order[:6]:
        i = idxs[k]
        print(f"    lev {i:3d}  E={E[i]:7.3f}  g={g[i]:.0f}  b_k={bk[i]:.3e}")
    print()
    if np.median(bkv) > 0.5 and np.median(bkv) < 2.0 and nbad < 0.1*len(bkv):
        print("  VERDICT: b_k ~ 1 => matrix respects DETAILED BALANCE (no rate bug; super-thermal in")
        print("           the pipeline must come from OUTSIDE this matrix = real physics/ion-balance).")
    else:
        print("  VERDICT: b_k NOT ~1 at J=B => a RATE in this matrix VIOLATES detailed balance = BUG.")
        print("           (the worst-departed levels above localize the offending channel.)")
