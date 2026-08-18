#!/usr/bin/env python3
"""M3: the two formal zero points, verified from first principles, then applied.

  A = ray-sample bias   : legacy midpoint impact grid charges the annulus that
                          straddles r_phot fully to the core (lumina.h:1130-1167)
  B = window truncation : FORMAL-CONS numerator is windowed, denominator is the
                          FULL injected Planck L_inj (lumina_plasma.c:16482-16489)
Both are recomputed here from parity42's own r_phot / r_outer / T_inner / window.
"""
import csv, math, os
import numpy as np
from scipy.integrate import quad

ROOT = '/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn'
OUT = os.path.join(ROOT, 'validation/cmfgen_toy06_19p48d/analysis/metering_batch1')
H, C, KB = 6.62607015e-27, 2.99792458e10, 1.380649e-16

# --- run constants, quoted verbatim from logs/coevolve_consume_parity42/stdout.log
r_phot, r_outer, n_imp = 6.563981e14, 6.782780e15, 100      # :122 / :41845
T_inner, L_inj = 10020.00, 3.094761e42                      # :124 / :41846
lam_lo, lam_hi = 504.875, 19995.125                         # :41846 window

# A -------------------------------------------------------------------------
dp = r_outer / n_imp
core = [(i + 0.5) * dp for i in range(n_imp) if (i + 0.5) * dp < r_phot]
w_disc = sum(p * dp for p in core)
w_exact = r_phot ** 2 / 2.0
A = w_disc / w_exact

# B -------------------------------------------------------------------------
def planck_frac(T, lo_A, hi_A):
    f = lambda x: x ** 3 / math.expm1(x)
    x1, x2 = H * C / (hi_A * 1e-8 * KB * T), H * C / (lo_A * 1e-8 * KB * T)
    return quad(f, x1, x2, limit=400)[0] / (math.pi ** 4 / 15)

B = planck_frac(T_inner, lam_lo, lam_hi)
f_win_LLt = planck_frac(T_inner, 2500.0, 19995.2)

print(f'A (ray sample)  : {len(core)} core rays, sum p dp = {w_disc/dp**2:.4f} dp^2, '
      f'exact = {w_exact/dp**2:.4f} dp^2  ->  A = {A:.6f}')
print(f'B (window)      : Planck({T_inner:.0f} K) inside {lam_lo}-{lam_hi} A '
      f'= {B:.6f}  (cut {100*(1-B):.4f}%)')
print(f'combined FORMAL-CONS zero point A*B = {A*B:.6f}')
print(f'Planck inside the L/L_truth window 2500-19995.2 A = {f_win_LLt:.6f}')

# --- apply ------------------------------------------------------------------
REC = [('parity36a', 0.7905, 3.4841), ('parity37', 1.0176, 34.8950),
       ('parity38', 1.0145, 4.3698), ('parity39', 1.0145, 4.3698),
       ('parity41', 0.9604, 4.1580), ('parity42', 5.4260, 24.9857)]
LT_over_Linj = 3.9911          # L_truth in the 2500-19995 A window, in L_inj

rows = []
for run, v, fc in REC:
    # defensible upper bound on the ray-sample bias of the TOTAL emergent L:
    #   |dL/L| <= (dW/W_core) * L_backlight_in_window / L_measured_in_window
    #   with L_backlight_in_window <= f_win * L_inj (direct B e^-tau <= Planck)
    bound = (A - 1.0) * f_win_LLt / (LT_over_Linj * v)
    rows.append(dict(run=run, L_over_Ltruth_recorded=v,
                     corrected_ray_only=v / A, corrected_both=v / (A * B),
                     defensible_bound_frac=bound,
                     defensible_lo=v * (1 - bound), defensible_hi=v * (1 + bound),
                     formal_cons_recorded=fc, formal_cons_corrected=fc / (A * B)))
with open(os.path.join(OUT, 'm3_zeropoint_corrected.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow(r)
for r in rows:
    print(f'  {r["run"]:<10s} {r["L_over_Ltruth_recorded"]:.4f} -> /A '
          f'{r["corrected_ray_only"]:.4f} -> /(A*B) {r["corrected_both"]:.4f} '
          f'| defensible +-{100*r["defensible_bound_frac"]:.2f}% '
          f'[{r["defensible_lo"]:.4f},{r["defensible_hi"]:.4f}]')
