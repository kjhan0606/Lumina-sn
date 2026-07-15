#!/usr/bin/env python3
"""UV field audit: is MC J over-amplified at 900-1500A (the IME excited-level
photoion thresholds that all-level Gph newly exposes)?
Compare per shell: mc_J vs dilute-BB (W*B(nu,T_rad)) vs cs_J (deterministic).
If mc_J >> W*B AND truth (blanketed) should be < W*B, mc_J is over-amplified.
Usage: uv_field_audit.py <field.csv dir>
"""
import sys, csv, math
import numpy as np
H=6.62607015e-27; KB=1.380649e-16; C=2.99792458e10
d=sys.argv[1] if len(sys.argv)>1 else 'logs/coevolve_consume_a10_kx_gphground'

# plasma W, T_rad per shell
ps={}
for r in csv.DictReader(open(f'{d}/lumina_plasma_state.csv')):
    ps[int(r['shell_id'])]={'W':float(r['W']),'T_rad':float(r['T_rad']),'T_e':float(r['T_e'])}

# field: shell,bin,wavelength_A,cs_J,mc_J
fld={}
for r in csv.DictReader(open(f'{d}/lumina_coevolve_field.csv')):
    s=int(r['shell']); fld.setdefault(s,[]).append(
        (float(r['wavelength_A']),float(r['cs_J']),float(r['mc_J'])))

def Bnu(lam_A,T):
    nu=C/(lam_A*1e-8)
    x=H*nu/(KB*T)
    if x>700: return 0.0
    return 2*H*nu**3/C**2/math.expm1(x)

# target wavelengths (S II / Si II excited-level thresholds region)
WLS=[913,1000,1100,1200,1300,1500]
print(f"# UV field audit: {d}")
print(f"# ratio = mc_J / (W*B(nu,T_rad)); >1 means MC exceeds dilute-Planck (truth should be BELOW)")
print(f"{'sh':>3} {'v?':>5} {'W':>7} {'Trad':>6} {'Te':>6} | "+" ".join(f"{str(w)+chr(65):>10}" for w in WLS))
for s in [4,6,8,10,12,15,20]:
    if s not in fld: continue
    rows=fld[s]; W=ps[s]['W']; Tr=ps[s]['T_rad']; Te=ps[s]['T_e']
    lam=np.array([x[0] for x in rows]); mcj=np.array([x[2] for x in rows]); csj=np.array([x[1] for x in rows])
    cells=[]
    for w in WLS:
        i=int(np.argmin(np.abs(lam-w)))
        wb=W*Bnu(lam[i],Tr)
        ratio=mcj[i]/wb if wb>0 else float('nan')
        cells.append(f"{ratio:>10.1f}")
    print(f"{s:>3} {'':>5} {W:>7.4f} {Tr:>6.0f} {Te:>6.0f} | "+" ".join(cells))

print(f"\n# mc_J / cs_J ratio (which field is hot? Gph uses alpha*mc+((1-a)cs, alpha=1 => pure mc):")
print(f"{'sh':>3} | "+" ".join(f"{str(w)+chr(65):>9}" for w in WLS))
for s in [4,6,8,10,12,15,20]:
    if s not in fld: continue
    rows=fld[s]
    lam=np.array([x[0] for x in rows]); mcj=np.array([x[2] for x in rows]); csj=np.array([x[1] for x in rows])
    cells=[]
    for w in WLS:
        i=int(np.argmin(np.abs(lam-w)))
        r=mcj[i]/csj[i] if csj[i]>0 else float('nan')
        cells.append(f"{r:>9.2f}")
    print(f"{s:>3} | "+" ".join(cells))
