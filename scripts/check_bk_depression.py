#!/usr/bin/env python3
"""DEPRECATED (2026-07-15, Fable adversarial verify): this summarizes b_k with
G-WEIGHTING ("use g as proxy"), which does NOT reflect which levels carry the
photoionization RATE (that is sigma*J-weighted). The "66x" IME depression this
tool reported is a g-weighting artifact -- the levels dominating S II/Si II
photoion have b_k ~ 0.2-1.3, not 0.015. Use scripts/db_photoion_calc.py, which
computes G_nlte/G_boltz with the REAL sigma_bf x J integral (the correct metric;
IME correction is only 1.6-2.7x, insufficient to fix over-ionization).

--- original (invalid) intent below ---
Test refined diagnosis: are IME (Si II, S II) excited levels NLTE-depressed
(b_k<<1) while IGE (Fe III, Co III) excited levels are less so?
If yes -> Boltzmann@T_e all-level Gph over-populates IME excited levels ->
over-ionizes IME. Correct fix = weight Gph by actual b_k.

Usage: check_bk_depression.py <levelpop.csv> [label]
levelpop cols: shell,Z,ion,level_num,E_eV,g,n_k,n_ground,b_k,has_sigma,n_sig_pos
"""
import sys, csv
import numpy as np
from collections import defaultdict

path=sys.argv[1]; label=sys.argv[2] if len(sys.argv)>2 else path
# targets: (Z, ion, shell, name)  ion=charge(0-index): Fe III=(26,2), Co III=(27,2),
#          Si II=(14,1), S II=(16,1)
TARGETS=[(26,2,0,'Fe III s0'),(27,2,0,'Co III s0'),(28,2,0,'Ni III s0'),
         (14,1,6,'Si II s6'),(16,1,6,'S II s6'),(16,1,10,'S II s10'),
         (14,2,6,'Si III s6'),(16,2,10,'S III s10')]

# gather rows per (shell,Z,ion)
rows=defaultdict(list)
with open(path) as f:
    for r in csv.DictReader(f):
        key=(int(r['shell']),int(r['Z']),int(r['ion']))
        rows[key].append((int(r['level_num']),float(r['E_eV']),float(r['g']),
                          float(r['b_k']),int(r['has_sigma'])))

print(f"### {label}")
print(f"{'species':>12} {'nlev':>5} {'nsig':>5} {'b_k(gnd)':>9} {'b_k_exc(sig,med)':>16} {'b_k_exc(sig,wtd)':>16}")
for Z,ion,sh,name in TARGETS:
    r=rows.get((sh,Z,ion))
    if not r:
        print(f"{name:>12}  (absent)"); continue
    r.sort()
    gnd=[x for x in r if x[0]==0]
    bk_gnd=gnd[0][3] if gnd else float('nan')
    # excited levels that can photoionize (has_sigma) and above ground
    exc=[x for x in r if x[0]>0 and x[4]==1]
    if exc:
        bks=np.array([x[3] for x in exc])
        # weight by Boltzmann factor g*exp(-E/kT)? use g as proxy weight for photoion contribution
        gs=np.array([x[2] for x in exc])
        med=np.median(bks)
        wtd=np.average(bks,weights=gs) if gs.sum()>0 else float('nan')
    else:
        med=wtd=float('nan')
    print(f"{name:>12} {len(r):>5} {len(exc):>5} {bk_gnd:>9.3f} {med:>16.3f} {wtd:>16.3f}")
print("\ninterpretation: b_k_exc << 1 => excited depressed => Boltzmann(b=1) over-ionizes.")
print("if IME(Si/S II) b_k_exc << IGE(Fe/Co III) b_k_exc => over-ionization is IME-selective artifact.")
