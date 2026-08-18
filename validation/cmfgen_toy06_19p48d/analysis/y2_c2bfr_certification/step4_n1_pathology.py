"""Y2 step 4: N1 pathology census -- where and how much does the C1 refit
distort the raw MC field that the C2 path would use?

Distortion metric per fine bin:  D[f] = J_C1[f] / J_raw[f]  (both erg/cm2/s/Hz/sr)
Energy share metric: the bin's share of  sum_f J*dnu  and of the *photoion
kernel* weight  sum_f (4pi/(h*nu))*dnu  (the measure R_bf actually integrates).

Coarse-bin mode taxonomy is READ from lumina_c1_bins.csv (written at
lumina_plasma.c:1038-1046): fit | pin | degen | empty.

PRELIMINARY-on-partial.
"""
import os
import numpy as np
import pandas as pd
import y2_common as Y

IT = -1
J_raw = np.load(os.path.join(Y.OUT, "_cache_Jraw.npy"))
J_C1 = np.load(os.path.join(Y.OUT, "_cache_JC1.npy"))
bfr = np.load(os.path.join(Y.OUT, "_cache_bfr.npy"))
cnt = np.load(os.path.join(Y.OUT, "_cache_cnt.npy"))
W = np.load(os.path.join(Y.OUT, "_cache_W.npy"))
TR = np.load(os.path.join(Y.OUT, "_cache_TR.npy"))
mode = np.load(os.path.join(Y.OUT, "_cache_mode.npy"), allow_pickle=True).astype(str)
ni, ns, nb = J_raw.shape

# ---------------------------------------------------------------- mode census
print("=== C1 coarse-bin mode census (all iters x 50 shells x 24 bins) ===")
vals, counts = np.unique(mode, return_counts=True)
for v, c in zip(vals, counts):
    print(f"  {v:6s}: {c:6d}  ({100*c/mode.size:5.2f}%)")
print("\n=== mode by coarse bin, iter 10 (lam range from the C1 dump) ===")
c1 = pd.read_csv(os.path.join(Y.RUN, "lumina_c1_bins.csv"))
c1l = c1[c1["iter"] == c1["iter"].max()]
lam = c1l.groupby("bin")[["lam_lo_A", "lam_hi_A"]].max()
rows = []
for c in range(Y.NC):
    m = mode[IT, :, c]
    rows.append(dict(cbin=c, lam_lo_A=lam.lam_lo_A.get(c, np.nan),
                     lam_hi_A=lam.lam_hi_A.get(c, np.nan),
                     n_fit=int((m == "fit").sum()), n_pin=int((m == "pin").sum()),
                     n_degen=int((m == "degen").sum()),
                     n_empty=int((m == "empty").sum()),
                     TR_med=float(np.median(TR[IT, :, c])),
                     W_med=float(np.median(W[IT, :, c]))))
mc = pd.DataFrame(rows)
mc.to_csv(os.path.join(Y.OUT, "y2_c1_mode_census.csv"), index=False)
print(mc.to_string(index=False))

# 250 kK rail firing
rail = (TR >= 0.95 * 250000.0)
print(f"\n[rail] coarse bins with T_R >= 0.95*250kK: {int(rail.sum())} / {rail.size}"
      f"  ({100*rail.mean():.2f}%)  |  exactly 250000.00 K: {int((TR==250000.0).sum())}")
print(f"[rail] of those, mode==degen (raw published instead): "
      f"{int((rail & (mode=='degen')).sum())};  mode==pin: {int((rail & (mode=='pin')).sum())};"
      f"  mode==fit (rail SURVIVES into the field): {int((rail & (mode=='fit')).sum())}")

# ------------------------------------------------------- fine-bin distortion
pk = 4.0 * np.pi / (Y.H_PLANCK * Y.NU_MID) * Y.DNU     # photoion kernel measure
both = (J_raw > 0) & (J_C1 > 0)
fill = (J_raw <= 0) & (J_C1 > 0)      # refit CREATES flux where MC saw none
kill = (J_raw > 0) & (J_C1 <= 0)      # refit DESTROYS flux the MC saw
D = np.where(both, J_C1 / np.where(J_raw > 0, J_raw, 1.0), np.nan)

BANDS = [("NIR/opt 20000-4000A", 4000., 20000.),
         ("near-UV 4000-2000A", 2000., 4000.),
         ("mid-UV 2000-1200A", 1200., 2000.),
         ("FUV 1200-912A", 912., 1200.),
         ("EUV 912-500A", 500., 912.),
         ("deep-EUV 500-100A", 100., 500.)]

rows = []
for nm, l0, l1 in BANDS:
    sel = (Y.LAM_MID_A >= l0) & (Y.LAM_MID_A < l1)
    d = D[IT][:, sel]
    b_ = both[IT][:, sel]
    f_ = fill[IT][:, sel]
    k_ = kill[IT][:, sel]
    # energy-weighted (photoion-kernel-weighted) share of each pathology
    wk = pk[sel][None, :]
    wraw = J_raw[IT][:, sel] * wk
    wc1 = J_C1[IT][:, sel] * wk
    tot_c1 = wc1.sum()
    rows.append(dict(
        band=nm, n_fine_bins=int(sel.sum()),
        frac_MC_unsampled=float((cnt[IT][:, sel] == 0).mean()),
        n_fill=int(f_.sum()), n_kill=int(k_.sum()),
        kernelw_share_fill=float(wc1[f_].sum() / tot_c1) if tot_c1 > 0 else np.nan,
        kernelw_share_D_gt_2=float(wc1[b_ & (d > 2)].sum() / tot_c1) if tot_c1 > 0 else np.nan,
        kernelw_share_D_lt_0p5=float(wc1[b_ & (d < 0.5)].sum() / tot_c1) if tot_c1 > 0 else np.nan,
        D_p10=float(np.nanpercentile(d[b_], 10)) if b_.sum() else np.nan,
        D_med=float(np.nanmedian(d[b_])) if b_.sum() else np.nan,
        D_p90=float(np.nanpercentile(d[b_], 90)) if b_.sum() else np.nan,
        kernel_ratio_C1_over_raw=float(wc1.sum() / wraw.sum()) if wraw.sum() > 0 else np.nan))
bd = pd.DataFrame(rows)
bd.to_csv(os.path.join(Y.OUT, "y2_n1_band_pathology.csv"), index=False)
print("\n=== N1 pathology by band (iter 10, all 50 shells pooled) ===")
print(bd.round(4).to_string(index=False))

# per-shell, the two bands that drive photoionization
rows = []
for s in range(ns):
    for nm, l0, l1 in BANDS:
        sel = (Y.LAM_MID_A >= l0) & (Y.LAM_MID_A < l1)
        wk = pk[sel]
        wraw = (J_raw[IT, s, sel] * wk).sum()
        wc1 = (J_C1[IT, s, sel] * wk).sum()
        f_ = fill[IT, s, sel]
        rows.append(dict(shell=s, band=nm,
                         kernel_C1=wc1, kernel_raw=wraw,
                         ratio_C1_over_raw=(wc1 / wraw) if wraw > 0 else np.nan,
                         n_fill=int(f_.sum()),
                         share_fill=(J_C1[IT, s, sel][f_] * wk[f_]).sum() / wc1
                         if wc1 > 0 else np.nan,
                         frac_unsampled=float((cnt[IT, s, sel] == 0).mean())))
ps = pd.DataFrame(rows)
ps.to_csv(os.path.join(Y.OUT, "y2_n1_shell_band.csv"), index=False)
print("\n=== kernel-weighted C1/raw ratio by shell x band (iter 10) ===")
print(ps.pivot(index="shell", columns="band",
               values="ratio_C1_over_raw").loc[[0, 8, 20, 30, 45, 49]]
      .round(3).to_string())
print("\n=== share of the C1 photoion kernel that comes from FILLED (MC-empty) bins ===")
print(ps.pivot(index="shell", columns="band",
               values="share_fill").loc[[0, 8, 20, 30, 45, 49]].round(3).to_string())
