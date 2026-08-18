"""Y2 step 12 (NEW in the formal cert): the amended P1-P4 brackets.

Inputs (both written by step10):
  y2_prereg_allshells.csv       iter 11 (Y2_IT=-1, default)
  y2_prereg_allshells_it-2.csv  iter 10 (Y2_IT=-2)

Rule, fixed in advance and applied mechanically:
  amended_lo = floor_1dp( min over the registered domain at iter 11 )
  amended_hi = ceil_1dp ( max over the registered domain at iter 11
                          x  the it10->it11 drift factor at the shell that
                             carries that max )
The drift multiplier is measured, not chosen: the field is still moving
(step1: median |dJ|/J = 0.19-0.31 in shells >= 20 at it10->11), so a bracket
built on a single iteration would be re-falsified by the next one.
"""
import os
import numpy as np
import pandas as pd
import y2_common as Y

f11 = pd.read_csv(os.path.join(Y.OUT, "y2_prereg_allshells.csv"))
f10 = pd.read_csv(os.path.join(Y.OUT, "y2_prereg_allshells_it-2.csv"))
BAND = list(range(20, 36))


def item(name, ion, shells, col, reg_lo, reg_hi):
    a = f11[(f11.ion == ion) & f11.shell.isin(shells)][["shell", col]].dropna()
    b = f10[(f10.ion == ion) & f10.shell.isin(shells)][["shell", col]].dropna()
    j = a.merge(b, on="shell", suffixes=("_11", "_10"))
    j["drift"] = j[f"{col}_11"] / j[f"{col}_10"]
    mn, mx = j[f"{col}_11"].min(), j[f"{col}_11"].max()
    sh_mx = int(j.loc[j[f"{col}_11"].idxmax(), "shell"])
    sh_mn = int(j.loc[j[f"{col}_11"].idxmin(), "shell"])
    dr_mx = float(j.loc[j[f"{col}_11"].idxmax(), "drift"])
    dr_mn = float(j.loc[j[f"{col}_11"].idxmin(), "drift"])
    inside = (mn >= reg_lo) and (mx <= reg_hi)
    lo_new = np.floor(min(mn, mn * min(dr_mn, 1.0)) * 10) / 10
    hi_new = np.ceil(max(mx, mx * max(dr_mx, 1.0)) * 10) / 10
    return dict(item=name, registered=f"x{reg_lo}-{reg_hi}",
                it11_min=mn, it11_min_shell=sh_mn,
                it11_max=mx, it11_max_shell=sh_mx,
                drift_at_max_it10_11=dr_mx,
                band_drift_median=float(j.drift.median()),
                verdict="WITHIN-RANGE" if inside else "OUTSIDE-RANGE",
                exceed_hi_by=max(0.0, mx - reg_hi),
                below_lo_by=max(0.0, reg_lo - mn),
                amended=f"x{lo_new:.1f}-{hi_new:.1f}")


rows = [
    item("P1 Si II s20-35 pop-wtd(Boltz)", "Si II", BAND, "ratio_boltz", 1.6, 3.5),
    item("P1 Si II s20-35 ground", "Si II", BAND, "ratio_ground", 2.0, 6.5),
    item("P2 Fe II s20-35 pop-wtd(Boltz)", "Fe II", BAND, "ratio_boltz", 1.4, 3.5),
    item("P2 Fe II s20-35 ground", "Fe II", BAND, "ratio_ground", 1.2, 6.5),
    item("P3 Mg II s30", "Mg II", [30], "ratio_boltz", 1.4, 1.7),
    item("P3 Co II s30", "Co II", [30], "ratio_boltz", 1.4, 1.7),
    item("P4 Ni II s20", "Ni II", [20], "ratio_boltz", 0.75, 0.98),
]
am = pd.DataFrame(rows)
am.to_csv(os.path.join(Y.OUT, "y2_amended_brackets.csv"), index=False)
pd.set_option("display.width", 260)
print("=== P1-P4 amended brackets (iter-11 observed + measured it10->11 drift) ===")
print(am.round(4).to_string(index=False))

# the true population-weighted variant, where the dump supports it
print("\n=== Si II s20-35, population-weighted (resolve_raw / resolve_ema) ===")
s = f11[(f11.ion == "Si II") & f11.shell.isin(BAND)]
for c in ("ratio_pop_raw", "ratio_pop_ema"):
    v = s[c].dropna()
    print(f"  {c}: min={v.min():.3f} max={v.max():.3f} "
          f"(vs Boltzmann {s.ratio_boltz.min():.3f}-{s.ratio_boltz.max():.3f}, "
          f"ground {s.ratio_ground.min():.3f}-{s.ratio_ground.max():.3f})")
print("\n  bracketing claim of PRELIM_REPORT.md S2.2 ('realized effect lies "
      "BETWEEN\n  ground and Boltzmann') -- test at every shell 20-35:")
ok = ((s.ratio_pop_raw >= np.minimum(s.ratio_boltz, s.ratio_ground)) &
      (s.ratio_pop_raw <= np.maximum(s.ratio_boltz, s.ratio_ground)))
print(f"    shells inside the bracket: {int(ok.sum())} / {len(s)}   "
      f"outside: {sorted(s.loc[~ok, 'shell'].tolist())}")
