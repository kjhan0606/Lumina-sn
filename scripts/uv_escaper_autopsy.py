#!/usr/bin/env python3
"""UV-escaper offline autopsy (Fable falsifier, 2026-07-06).

Question: the residual UV excess (epay27: 43% vs CMFGEN 24%). Is it
  (a) OPACITY deficit  — UV packets escape without ever interacting, or
  (b) BRANCHING defect — they interact but re-emit in UV / thermalize to
      red/NIR instead of cascading down to the blue (Fe/Mg) fluorescence band?

Three offline probes (no MC re-run; uses frozen-plasma dumps):
  1. FATE matrix from ma_fate_zihist.csv: UV-entry packets' exit-band
     distribution, global + per species. Fluorescence yield P(UV->optical).
  2. OPACITY depth from lumina_cmfgen_jnu.csv: radial tau_UV per shell
     (line + abs) vs electron scattering -> do UV photons get absorbed?
  3. Cross-check: which species carry the UV->UV re-emission (branching
     defect carriers) vs which cascade (working fluorescers).

Usage: python3 scripts/uv_escaper_autopsy.py [run=logs/stage1_toy06_epay27]
"""
import sys, csv
import numpy as np

RUN = sys.argv[1] if len(sys.argv) > 1 else 'logs/stage1_toy06_epay27'
FATE = sys.argv[2] if len(sys.argv) > 2 else 'ma_fate_zihist.csv'
GEO = 'data/tardis_reference_toy06_19p48d/geometry.csv'
JNU = f'{RUN}/lumina_cmfgen_jnu.csv'
C_A = 2.99792458e18  # Angstrom / s

BANDS = {0: 'UVblnk[1700,3000)', 1: 'CaIIKb[3000,3300)', 2: 'UVtgt[3300,3700)',
         3: 'fluor[3700,4400)', 4: 'green[4400,5500)', 5: 'red[5500,7000)',
         6: 'NIR1[7000,10000)', 7: 'NIR2[>=10000)'}
UV_ENTRY = {0, 1, 2}          # UV bands
OPTICAL = {3, 4, 5}           # fluorescence target: fluor+green+red
NIR = {6, 7}

# ---------------------------------------------------------------- 1. FATE
rows = []
lines = [L for L in open(FATE) if not L.startswith('#') and L.strip()]
for r in csv.DictReader(lines):
    rows.append((int(r['Z']), r['Z_name'], int(r['ion']),
                 int(r['entry_band']), int(r['exit_band']), int(r['count'])))

# global exit distribution for UV-entry packets
tot_uv = sum(c for Z, n, i, eb, xb, c in rows if eb in UV_ENTRY)
exit_hist = {}
for Z, n, i, eb, xb, c in rows:
    if eb in UV_ENTRY:
        exit_hist[xb] = exit_hist.get(xb, 0) + c
print("=" * 72)
print(f"[1] FATE of UV-entry macro-atom packets (bands 0+1+2), N={tot_uv:,}")
print("    exit band                     count        %")
for xb in sorted(exit_hist):
    print(f"    {BANDS[xb]:26s} {exit_hist[xb]:12,} {100*exit_hist[xb]/tot_uv:7.2f}%")
uv_out = sum(exit_hist.get(b, 0) for b in UV_ENTRY)
opt_out = sum(exit_hist.get(b, 0) for b in OPTICAL)
nir_out = sum(exit_hist.get(b, 0) for b in NIR)
print(f"    --> UV re-emit  {100*uv_out/tot_uv:6.2f}%   "
      f"optical(fluor)  {100*opt_out/tot_uv:6.2f}%   NIR {100*nir_out/tot_uv:6.2f}%")
print(f"    FLUORESCENCE YIELD P(UV-entry -> optical) = {100*opt_out/tot_uv:.2f}%  "
      f"(CMFGEN physics wants this LARGE)")

# per-species: who re-emits UV vs who cascades
print("\n    per-species UV-entry fate (species with >0.5% of UV-entry flux):")
print("    Z ion    name    N_uv-entry   ->UV%   ->opt%  ->NIR%")
spec = {}
for Z, n, i, eb, xb, c in rows:
    if eb in UV_ENTRY:
        k = (Z, i, n)
        d = spec.setdefault(k, [0, 0, 0, 0])  # tot, uv, opt, nir
        d[0] += c
        if xb in UV_ENTRY: d[1] += c
        elif xb in OPTICAL: d[2] += c
        elif xb in NIR: d[3] += c
for (Z, i, n), (t, u, o, nr) in sorted(spec.items(), key=lambda x: -x[1][0]):
    if t < 0.005 * tot_uv:
        continue
    print(f"    {Z:2d} {i:2d}  {n:6s} {t:12,}  {100*u/t:6.1f} {100*o/t:6.1f} {100*nr/t:6.1f}")

# ---------------------------------------------------------------- 2. OPACITY
geo = {int(r['shell_id']): (float(r['r_inner']), float(r['r_outer']))
       for r in csv.DictReader(open(GEO))}
d = np.genfromtxt(JNU, delimiter=',', names=True)
print("\n" + "=" * 72)
print("[2] UV (2500-3500A) radial optical depth per shell (frozen plasma)")
print("    shell   dr[cm]     tau_line   tau_abs   tau_es   (tau_line>1 => UV absorbed)")
cum_line = 0.0
shells = sorted(set(int(x) for x in d['shell']))
for s in shells:
    sub = d[d['shell'] == s]
    lam = C_A / sub['nu']
    uv = (lam >= 2500) & (lam < 3500)
    if uv.sum() == 0:
        continue
    dr = geo[s][1] - geo[s][0]
    tl = np.mean(sub['chi_line'][uv]) * dr
    ta = np.mean(sub['chi_abs'][uv]) * dr
    te = np.mean(sub['chi_es'][uv]) * dr
    cum_line += tl
    if s % 5 == 0 or s < 6:
        print(f"    {s:3d}   {dr:.3e}  {tl:9.3f}  {ta:8.2e}  {te:7.3f}")
print(f"    cumulative radial tau_line(UV) over all shells = {cum_line:.2f}")
print(f"    => UV packets {'DO' if cum_line > 1 else 'do NOT'} get absorbed "
      f"(opacity {'present' if cum_line > 1 else 'DEFICIENT'})")

# ---------------------------------------------------------------- VERDICT
print("\n" + "=" * 72)
print("[VERDICT]")
fl = 100 * opt_out / tot_uv
uvre = 100 * uv_out / tot_uv
if cum_line > 1 and uvre > 50:
    print(f"  Opacity present (tau_UV={cum_line:.1f}) but UV-entry re-emits UV "
          f"{uvre:.0f}% and fluoresces only {fl:.0f}% -> BRANCHING DEFECT.")
    print("  UV photons ARE absorbed; the macro-atom sends them back out in UV")
    print("  instead of cascading down. Attack = internal-down branch ratios")
    print("  (Fe/Co II-III) vs ARTIS down-jump formula.")
elif cum_line < 1:
    print(f"  tau_UV={cum_line:.2f} < 1 -> OPACITY DEFICIT: UV escapes unabsorbed.")
    print("  Attack = Fe II/III UV line count / Sobolev tau vs CMFGEN forest.")
else:
    print(f"  Mixed: tau_UV={cum_line:.1f}, fluor yield {fl:.0f}%. Inspect species table.")
