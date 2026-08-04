#!/usr/bin/env python3
"""Stage-0 field-color falsifier judge (fluorescence fix campaign, 2026-07-07).

Reads lumina_fine_jmap.csv (cols: shell,lambda_A,Te,J_over_B; written by the fine
producer under LUMINA_CMF_FINE_JMAP=1) and decides the go/no-go gate for the whole
fluorescence-fix family (design workflow wf_e8bd26c2):

  GATE: in the UV band (2500-3500 A) of the line-forming shells, is J/B > ~1.2?
    PASS -> a super-thermal UV pump exists in the (eps<1) scattering field ->
            the "b_k>1 in the deterministic solve" family is viable -> build Stage 1-3.
    FAIL (J/B ~ 1.0) -> the field is NOT super-thermal in the UV -> the family is
            falsified in place -> the real cap is the bf/continuum blanket
            (chi_abs*B(Te) thermalization); pivot there before writing rate code.

Rationale: b_k>1 (the fluorescence pump) can only form where J/B>1. This measures,
read-only, whether Lumina's fine field with a scattering line source (FINE_LINE_EPS<1)
actually carries a diluted-hot photospheric UV color into the line-forming region.

Usage:
  python3 scripts/judge_field_color_falsifier.py [jmap.csv] \
          [--uv LO HI] [--shells S0 S1] [--gate 1.2]
Defaults: jmap = logs/stage1_toy06_s0falsifier/lumina_fine_jmap.csv (or ./lumina_fine_jmap.csv);
          UV 2500-3500 A; line-forming shells 5-15; gate 1.2.
"""
import sys, csv, os, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('jmap', nargs='?', default=None)
ap.add_argument('--uv', nargs=2, type=float, default=[2500.0, 3500.0])
ap.add_argument('--shells', nargs=2, type=int, default=[5, 15])
ap.add_argument('--gate', type=float, default=1.2)
a = ap.parse_args()

path = a.jmap
if path is None:
    for c in ('logs/stage1_toy06_s0falsifier/lumina_fine_jmap.csv', 'lumina_fine_jmap.csv'):
        if os.path.exists(c):
            path = c; break
if not path or not os.path.exists(path):
    sys.exit(f"jmap not found (looked for {path}); run with LUMINA_CMF_FINE_JMAP=1 first")

rows = list(csv.DictReader(open(path)))
Te = {int(r['shell']): float(r['Te']) for r in rows}
lo_s, hi_s = a.shells
uv_lo, uv_hi = a.uv

byshell = {}
for r in rows:
    s = int(r['shell']); lam = float(r['lambda_A']); jb = float(r['J_over_B'])
    if uv_lo <= lam < uv_hi and np.isfinite(jb) and jb > 0:
        byshell.setdefault(s, []).append(jb)

print(f"field-color falsifier: {path}")
print(f"  UV band {uv_lo:.0f}-{uv_hi:.0f} A ; line-forming shells {lo_s}-{hi_s} ; gate J/B>{a.gate}")
print(f"\n{'shell':>5} {'Te':>7} {'medianJ/B':>10} {'max':>8} {'n':>4}")
allv = []
for s in range(lo_s, hi_s + 1):
    if s in byshell:
        v = np.array(byshell[s]); allv += list(v)
        print(f"{s:5d} {Te.get(s,0):7.0f} {np.median(v):10.3f} {v.max():8.3f} {len(v):4d}")
if not allv:
    sys.exit("no UV bins in the line-forming shell range — check band/shell args")
allv = np.array(allv)
med = float(np.median(allv))
frac_super = float(np.mean(allv > a.gate))
print(f"\n=== VERDICT: line-forming UV median J/B = {med:.3f} "
      f"({100*frac_super:.0f}% of bins > {a.gate}) ===")
if med > a.gate:
    print("  PASS: a super-thermal UV pump exists -> b_k>1 family VIABLE -> build Stage 1")
    print("        (Stage 1 = physical-eps + multi-bin (W,T_R) producer; Stage 2 = drive b_k in")
    print("         the deterministic NLTE solve under ION_LOCK; Stage 3 = THEN_MC NITER=2.)")
else:
    print("  FAIL: field NOT super-thermal in the UV -> fluorescence-field family FALSIFIED.")
    print("        Real cap = bf/continuum blanket (chi_abs*B(Te) thermalizes the field).")
    print("        Pivot to the bf/recombination continuum treatment (FINE_BF_OPAC) before")
    print("        writing any rate-matrix code.")
