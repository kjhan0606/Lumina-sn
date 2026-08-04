#!/usr/bin/env python3
"""Judge whether Lumina's MC-TRANSPORT field is super-thermal in the UV line-forming
region (the Stage-0b falsifier). Complements judge_field_color_falsifier.py, which
tested the DETERMINISTIC field.

Reads lumina_mc_jnu.csv (shell,bin,nu,J_mc; written under LUMINA_MC_JDUMP=1 in a
THEN_MC run) + a plasma_state.csv for T_e per shell. Computes J_mc/B(nu,T_e) in the
UV band of the line-forming shells.

  GATE: median J_mc/B > ~1.2 in UV(2500-3500A), line-forming shells 5-15?
    PASS -> the MC transport field IS super-thermal (ARTIS-like) even though the
            deterministic field is not -> the fix is to feed the MC field to the
            macro-atom/rates (Design A core; IUP_TRAD was weak only because it used
            the tuned dilute-BB, not the true MC field).
    FAIL -> the MC field is also cool -> the hot deep-interior UV is not being
            transported (injection at s5 samples a cool source / boundary too cool)
            -> the fix is the injection/boundary/deep-layer UV, not the rates.

Usage: python3 scripts/judge_mc_field.py [mc_jnu.csv] [plasma_state.csv]
"""
import sys, csv, os, math
import numpy as np

H = 6.62607015e-27; KB = 1.380649e-16; C = 2.99792458e10; C_A = 2.99792458e18
mcp = sys.argv[1] if len(sys.argv) > 1 else 'lumina_mc_jnu.csv'
psp = sys.argv[2] if len(sys.argv) > 2 else 'logs/stage1_toy06_epay27j/lumina_plasma_state.csv'
if not os.path.exists(mcp): sys.exit(f"{mcp} not found (run with LUMINA_MC_JDUMP=1)")
if not os.path.exists(psp):
    for c in ('lumina_plasma_state.csv', 'logs/stage1_toy06_epay27/lumina_plasma_state.csv'):
        if os.path.exists(c): psp = c; break

Te = {int(r['shell_id']): float(r['T_e']) for r in csv.DictReader(open(psp))}
def B(nu, T):
    x = H * nu / (KB * T)
    return 2 * H * nu**3 / C**2 / math.expm1(x) if 1e-6 < x < 500 else 0.0

d = np.genfromtxt(mcp, delimiter=',', names=True)
byshell = {}
for row in d:
    s = int(row['shell']); nu = row['nu']; J = row['J_mc']
    lam = C_A / nu
    if 2500 <= lam < 3500 and s in Te and J > 0:
        b = B(nu, Te[s])
        if b > 0: byshell.setdefault(s, []).append(J / b)

print(f"MC-field falsifier: {mcp} (Te from {psp})")
print(f"{'shell':>5} {'Te':>7} {'medianJ/B':>10} {'max':>8} {'n':>4}")
allv = []
for s in range(5, 16):
    if s in byshell:
        v = np.array(byshell[s]); allv += list(v)
        print(f"{s:5d} {Te.get(s,0):7.0f} {np.median(v):10.3f} {v.max():8.3f} {len(v):4d}")
if not allv: sys.exit("no UV bins in line-forming shells")
allv = np.array(allv); med = float(np.median(allv))
print(f"\n=== VERDICT: MC-field line-forming UV median J/B = {med:.3f} "
      f"({100*np.mean(allv>1.2):.0f}% bins > 1.2) ===")
if med > 1.2:
    print("  PASS: MC transport field IS super-thermal -> feed the MC field to the")
    print("        macro-atom/rates (Design A core). IUP_TRAD was weak only because it")
    print("        used the tuned dilute-BB T_rad, not this true MC field.")
else:
    print("  FAIL: MC field also cool -> hot deep-interior UV not transported.")
    print("        Fix = injection/boundary/deep-layer UV color, not the rate field.")
