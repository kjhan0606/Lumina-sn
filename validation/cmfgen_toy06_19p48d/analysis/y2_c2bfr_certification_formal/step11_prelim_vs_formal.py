"""Y2 step 11 (NEW in the formal cert): prelim(it10, killed-partial) vs
formal(it11, clean), plus a DETERMINISM check on the iterations the two runs
share (0..10).

Reads the PRELIM directory read-only.  Nothing is written there.
"""
import os
import numpy as np
import pandas as pd
import y2_common as Y

PRE = os.path.join(os.path.dirname(Y.OUT.rstrip("/")), "y2_c2bfr_certification")
print(f"prelim dir: {PRE}\nformal dir: {Y.OUT}")

pd.set_option("display.width", 240)

# ------------------------------------------- determinism at the DUMP-BYTE level
PART = os.path.join(os.path.dirname(Y.RUN.rstrip("/")),
                    "coevolve_consume_parity46_killed_partial")
print("\n=== DETERMINISM at the raw-dump LINE level (killed-partial vs clean) ===")
ndiff, first, ntot = 0, None, 0
with open(os.path.join(PART, "lumina_c2_bfr_dump.csv"), "rb") as fa, \
     open(os.path.join(Y.RUN, "lumina_c2_bfr_dump.csv"), "rb") as fb:
    for i, (x, yy) in enumerate(zip(fa, fb)):
        ntot += 1
        if x != yy:
            ndiff += 1
            if first is None:
                first = (i, x.decode().strip(), yy.decode().strip())
print(f"  common prefix lines compared : {ntot} (1 header + 550,000 data)")
print(f"  lines that differ            : {ndiff}")
if first:
    print(f"  first differing line (idx {first[0]}):\n    partial: {first[1]}\n"
          f"    clean  : {first[2]}")

# ---------------------------------------------------------- determinism 0..10
pr = pd.read_csv(os.path.join(PRE, "y2_ratio_by_iter.csv"))
fr = pd.read_csv(os.path.join(Y.OUT, "y2_ratio_by_iter.csv"))
m = pr.merge(fr, on=["ion", "iter", "shell"], suffixes=("_pre", "_for"))
m["d_ratio"] = (m.ratio_for - m.ratio_pre).abs()
m["rel_G2"] = (m.Gamma_C2_for - m.Gamma_C2_pre).abs() / m.Gamma_C2_pre.replace(0, np.nan)
print("\n=== DETERMINISM: shared iters 0..10, Boltzmann ratio and Gamma_C2 ===")
print(f"  rows compared          : {len(m)}  (iters {sorted(m['iter'].unique())})")
print(f"  max |ratio_for - ratio_pre|          : {m.d_ratio.max():.3e}")
print(f"  max relative |dGamma_C2|             : {np.nanmax(m.rel_G2.values):.3e}")

pj = pd.read_csv(os.path.join(PRE, "y2_Jraw_stability.csv"))
fj = pd.read_csv(os.path.join(Y.OUT, "y2_Jraw_stability.csv"))
j = pj[["shell", "med_rel_d_9_10", "Jint_i10"]].merge(
    fj[["shell", "med_rel_d_9_10", "med_rel_d_10_11", "Jint_i10", "Jint_i11"]],
    on="shell", suffixes=("_pre", "_for"))
print("\n=== DETERMINISM: J_raw stability it9->10 (identical window in both) ===")
print(f"  max |med_rel_d_9_10 diff| : "
      f"{(j.med_rel_d_9_10_for - j.med_rel_d_9_10_pre).abs().max():.3e}")
print(f"  max rel |Jint_i10 diff|   : "
      f"{((j.Jint_i10_for - j.Jint_i10_pre).abs() / j.Jint_i10_pre).max():.3e}")
print("\n  the NEW information (it10->11) by shell:")
print(j[j.shell.isin([0, 8, 20, 30, 45, 49])][
    ["shell", "med_rel_d_9_10_for", "med_rel_d_10_11"]].round(4).to_string(index=False))

# ------------------------------------------------------ headline side by side
pg = pd.read_csv(os.path.join(PRE, "y2_gamma_per_ion.csv"))
fg = pd.read_csv(os.path.join(Y.OUT, "y2_gamma_per_ion.csv"))
h = pg[["ion", "shell", "ratio_C2_over_GEMM"]].merge(
    fg[["ion", "shell", "ratio_C2_over_GEMM"]], on=["ion", "shell"],
    suffixes=("_it10_prelim", "_it11_formal"))
h["delta"] = h.ratio_C2_over_GEMM_it11_formal - h.ratio_C2_over_GEMM_it10_prelim
h["sign_flip_vs_1"] = ((h.ratio_C2_over_GEMM_it10_prelim - 1.0) *
                       (h.ratio_C2_over_GEMM_it11_formal - 1.0)) < 0
h.to_csv(os.path.join(Y.OUT, "y2_prelim_vs_formal_headline.csv"), index=False)
print("\n=== HEADLINE Boltzmann ratio: prelim it10 (partial) vs formal it11 (clean) ===")
print(h.pivot(index="ion", columns="shell",
              values="ratio_C2_over_GEMM_it10_prelim").round(3).to_string())
print("\n  --- formal it11 ---")
print(h.pivot(index="ion", columns="shell",
              values="ratio_C2_over_GEMM_it11_formal").round(3).to_string())
print("\n  --- delta (formal - prelim) ---")
print(h.pivot(index="ion", columns="shell", values="delta").round(3).to_string())
print(f"\n  ion x shell cells where the ratio crosses 1.0 (sign flip): "
      f"{int(h.sign_flip_vs_1.sum())}")
print(h[h.sign_flip_vs_1].round(3).to_string(index=False))

# ---------------------------------------------------------------- ground table
pgr = pd.read_csv(os.path.join(PRE, "y2_ratio_by_iter.csv"))
fgr = pd.read_csv(os.path.join(Y.OUT, "y2_ratio_by_iter.csv"))
p10 = pgr[pgr["iter"] == pgr["iter"].max()].pivot(index="ion", columns="shell",
                                                  values="ground_ratio")
f11 = fgr[fgr["iter"] == fgr["iter"].max()].pivot(index="ion", columns="shell",
                                                  values="ground_ratio")
print("\n=== GROUND ratio: prelim it10 ---")
print(p10.round(3).to_string())
print("\n=== GROUND ratio: formal it11 ---")
print(f11.round(3).to_string())

# ------------------------------------------------------------------- hygiene
print("\n=== HYGIENE side by side ===")
pf = pd.read_csv(os.path.join(PRE, "y2_finiteness.csv"), index_col=0)
ff = pd.read_csv(os.path.join(Y.OUT, "y2_finiteness.csv"), index_col=0)
print("  prelim:\n" + pf.to_string())
print("  formal:\n" + ff.to_string())
pu = pd.read_csv(os.path.join(PRE, "y2_unsampled_fraction.csv"))
fu = pd.read_csv(os.path.join(Y.OUT, "y2_unsampled_fraction.csv"))
for nm, d in (("prelim", pu), ("formal", fu)):
    print(f"  {nm}: n_cnt0_bfr_pos={d.n_cnt0_bfr_pos.sum()} "
          f"n_cnt_pos_bfr0={d.n_cnt_pos_bfr0.sum()} rows={len(d)*Y.NB}")
pb = pd.read_csv(os.path.join(PRE, "y2_bfr_identity.csv"))
fb = pd.read_csv(os.path.join(Y.OUT, "y2_bfr_identity.csv"))
for nm, d in (("prelim", pb), ("formal", fb)):
    print(f"  {nm} bfr-identity last iter: n={d.n.sum()} Rmin={d.Rmin.min():.6f} "
          f"Rmax={d.Rmax.max():.6f} n_out={d.n_out.sum()}")
