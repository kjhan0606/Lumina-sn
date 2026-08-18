"""Y2 step 1: instrument hygiene + cross-dump roundtrip identity.

FORMAL (clean complete run, iters 0-11).  The three-iteration stability window
is therefore (9, 10, 11); array indices -3/-2/-1 are unchanged.
"""
import os
import numpy as np
import pandas as pd
import y2_common as Y

d = Y.load_dumps()
J_raw, bfr, cnt = d["J_raw"], d["bfr"], d["cnt"]
iters, shells = d["iters"], d["shells"]
ni, ns = len(iters), len(shells)
print(f"iters={list(iters)}  n_shells={ns}  n_bins={Y.NB}")

# --- grid self-check: does the writer's nu_mid match the header constants? ---
dev = np.abs(d["nu_mid_file"] / Y.NU_MID - 1.0)
print(f"[grid] max |nu_mid_file/nu_mid_hdr - 1| = {dev.max():.3e}  "
      f"(NU_MIN={Y.NU_MIN:.6e} DLOG={Y.DLOG:.9e})")
assert dev.max() < 1e-6

# --- 1a finiteness / sign ---
rep = {}
for nm, a in (("J_raw", J_raw), ("bfr", bfr)):
    rep[nm] = dict(n=a.size, n_nonfinite=int((~np.isfinite(a)).sum()),
                   n_negative=int((a < 0).sum()), n_zero=int((a == 0).sum()),
                   vmin=float(np.nanmin(a)), vmax=float(np.nanmax(a)))
    print(f"[finite] {nm}: nonfinite={rep[nm]['n_nonfinite']} "
          f"neg={rep[nm]['n_negative']} zero={rep[nm]['n_zero']} "
          f"range=[{rep[nm]['vmin']:.4e},{rep[nm]['vmax']:.4e}]")
print(f"[count] j_nu_count: min={cnt.min()} max={cnt.max()} "
      f"n_missing_array(-1)={int((cnt < 0).sum())}")

# --- 1b MC-unsampled fraction per (iter, shell) ---
rows = []
for i in range(ni):
    for s in range(ns):
        c = cnt[i, s]
        b = bfr[i, s]
        rows.append(dict(iter=int(iters[i]), shell=int(shells[s]),
                         n_cnt0=int((c == 0).sum()),
                         frac_cnt0=float((c == 0).mean()),
                         n_bfr0=int((b <= 0).sum()),
                         frac_bfr0=float((b <= 0).mean()),
                         n_cnt0_bfr_pos=int(((c == 0) & (b > 0)).sum()),
                         n_cnt_pos_bfr0=int(((c > 0) & (b <= 0)).sum()),
                         packets=int(c.sum())))
uns = pd.DataFrame(rows)
uns.to_csv(os.path.join(Y.OUT, "y2_unsampled_fraction.csv"), index=False)
piv = uns.pivot(index="shell", columns="iter", values="frac_cnt0")
print("\n[unsampled] fraction of fine bins with j_nu_count==0, by shell x iter")
print(piv.loc[[0, 8, 20, 30, 45, 49]].round(3).to_string())
print(f"\n[unsampled] all-shell mean frac(cnt==0) per iter:")
print(uns.groupby('iter')['frac_cnt0'].mean().round(4).to_string())
print(f"[consistency] bins with cnt==0 but bfr>0 : {uns['n_cnt0_bfr_pos'].sum()}")
print(f"[consistency] bins with cnt >0 but bfr<=0: {uns['n_cnt_pos_bfr0'].sum()}")

# --- 1c iteration-to-iteration stability of J_raw over the final 3 iters ---
# clean run: indices -3/-2/-1 == iters 9 / 10 / 11
IA, IB, IC = (int(iters[-3]), int(iters[-2]), int(iters[-1]))
stab = []
for s in range(ns):
    a, b, c = J_raw[-3, s], J_raw[-2, s], J_raw[-1, s]
    m = (a > 0) & (b > 0) & (c > 0)
    w = c  # energy weight = the field itself
    def relchg(x, y):
        mm = m
        if mm.sum() == 0:
            return np.nan, np.nan
        r = np.abs(y[mm] - x[mm]) / x[mm]
        wr = np.average(r, weights=(x[mm] * Y.DNU[mm]))
        return float(np.median(r)), float(wr)
    m1, w1 = relchg(a, b)
    m2, w2 = relchg(b, c)
    stab.append({"shell": int(shells[s]), "n_common_pos": int(m.sum()),
                 f"med_rel_d_{IA}_{IB}": m1, f"wmean_rel_d_{IA}_{IB}": w1,
                 f"med_rel_d_{IB}_{IC}": m2, f"wmean_rel_d_{IB}_{IC}": w2,
                 f"Jint_i{IA}": float((J_raw[-3, s] * Y.DNU).sum()),
                 f"Jint_i{IB}": float((J_raw[-2, s] * Y.DNU).sum()),
                 f"Jint_i{IC}": float((J_raw[-1, s] * Y.DNU).sum())})
stab = pd.DataFrame(stab)
stab[f"Jint_rel_d_{IA}_{IB}"] = stab[f"Jint_i{IB}"] / stab[f"Jint_i{IA}"] - 1.0
stab[f"Jint_rel_d_{IB}_{IC}"] = stab[f"Jint_i{IC}"] / stab[f"Jint_i{IB}"] - 1.0
stab.to_csv(os.path.join(Y.OUT, "y2_Jraw_stability.csv"), index=False)
print(f"\n[stability] J_raw, last 3 iters ({IA},{IB},{IC})")
print(stab.loc[stab.shell.isin([0, 8, 20, 30, 45, 49]),
               ["shell", "n_common_pos",
                f"med_rel_d_{IA}_{IB}", f"wmean_rel_d_{IA}_{IB}",
                f"med_rel_d_{IB}_{IC}", f"wmean_rel_d_{IB}_{IC}",
                f"Jint_rel_d_{IA}_{IB}",
                f"Jint_rel_d_{IB}_{IC}"]].round(4).to_string(index=False))
print(f"[stability] CONVERGING? median |dJ|/J  it{IB}->{IC} < it{IA}->{IB} :",
      {int(r.shell): bool(getattr(r, f"med_rel_d_{IB}_{IC}") <
                          getattr(r, f"med_rel_d_{IA}_{IB}"))
       for r in stab.loc[stab.shell.isin([0, 8, 20, 30, 45, 49])].itertuples()})
print(f"[stability] all-shell median of med_rel_d_{IA}_{IB} = "
      f"{stab[f'med_rel_d_{IA}_{IB}'].median():.4f}   "
      f"med_rel_d_{IB}_{IC} = {stab[f'med_rel_d_{IB}_{IC}'].median():.4f}")

# --- 1d bfr stability, same window ---
bstab = []
for s in range(ns):
    b9, b10 = bfr[-2, s], bfr[-1, s]
    m = (b9 > 0) & (b10 > 0)
    r = np.abs(b10[m] - b9[m]) / b9[m] if m.sum() else np.array([np.nan])
    bstab.append({"shell": int(shells[s]), "n_common_pos": int(m.sum()),
                  "med_rel_d": float(np.median(r)),
                  f"bfr_sum_i{IB}": float(b9.sum()),
                  f"bfr_sum_i{IC}": float(b10.sum())})
bstab = pd.DataFrame(bstab)
bstab["bfr_sum_rel_d"] = bstab[f"bfr_sum_i{IC}"] / bstab[f"bfr_sum_i{IB}"] - 1.0
bstab.to_csv(os.path.join(Y.OUT, "y2_bfr_stability.csv"), index=False)
print(f"\n[stability] bfr, iter {IB}->{IC}")
print(bstab.loc[bstab.shell.isin([0, 8, 20, 30, 45, 49])].round(4).to_string(index=False))
# bfr must track J_raw (step2 identity): report the discrepancy explicitly
_chk = stab[["shell", f"med_rel_d_{IB}_{IC}"]].merge(
    bstab[["shell", "med_rel_d"]], on="shell", suffixes=("_J", "_bfr"))
_chk["abs_diff"] = (_chk[f"med_rel_d_{IB}_{IC}"] - _chk["med_rel_d"]).abs()
print(f"[stability] max |median dJ_raw - median dbfr| over 50 shells = "
      f"{_chk.abs_diff.max():.5f}")

# --- 1e ROUNDTRIP: sum_f J_raw*dnu over a coarse bin == c1 J_bin ---
c1 = d["c1"]
W, TR, mode = Y.build_JC1(c1, iters, shells)
Jint_fine = np.zeros((ni, ns, Y.NC))
for c in range(Y.NC):
    sel = Y.CIDX == c
    Jint_fine[:, :, c] = (J_raw[:, :, sel] * Y.DNU[sel]).sum(axis=-1)
c1s = c1.sort_values(["iter", "shell", "bin"])
Jbin_c1 = c1s["J_bin"].values.reshape(ni, ns, Y.NC)
both = (Jint_fine > 0) & (Jbin_c1 > 0)
ratio = np.where(both, Jint_fine / np.where(Jbin_c1 > 0, Jbin_c1, 1.0), np.nan)
print("\n[ROUNDTRIP] sum_f J_raw*dnu (C2 dump) vs J_bin (C1 dump), per coarse bin")
print(f"  compared bins: {int(both.sum())} / {both.size}; "
      f"disagree-on-zero: {int(((Jint_fine>0)!=(Jbin_c1>0)).sum())}")
print(f"  ratio: median={np.nanmedian(ratio):.9f} "
      f"min={np.nanmin(ratio):.9f} max={np.nanmax(ratio):.9f} "
      f"max|1-r|={np.nanmax(np.abs(ratio-1)):.3e}")

pd.DataFrame(rep).T.to_csv(os.path.join(Y.OUT, "y2_finiteness.csv"))
np.save(os.path.join(Y.OUT, "_cache_W.npy"), W)
np.save(os.path.join(Y.OUT, "_cache_TR.npy"), TR)
np.save(os.path.join(Y.OUT, "_cache_mode.npy"), mode.astype(str))
np.save(os.path.join(Y.OUT, "_cache_Jraw.npy"), J_raw)
np.save(os.path.join(Y.OUT, "_cache_bfr.npy"), bfr)
np.save(os.path.join(Y.OUT, "_cache_cnt.npy"), cnt)
print("\ncached arrays written to", Y.OUT)
