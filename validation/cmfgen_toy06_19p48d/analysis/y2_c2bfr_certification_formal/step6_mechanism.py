"""Y2 step 6: mechanism of the C2-vs-GEMM gap + exact decomposition.

Two claims to falsify, not assume:
 (M1) the gap lives in the T_e-PINNED coarse bins 14/15 (728-905 A, 583-729 A),
      where LUMINA_C1_SUPERBIN_TEPIN replaces the MC shape by W*B_nu(T_e).  A
      Wien exponential across a 1.24:1 frequency bin at T_e~8.4kK suppresses the
      blue edge by ~e^{-3.7}, so an edge near the blue end of the bin is starved.
 (M2) the C2-vs-GEMM difference is confined to MC-SAMPLED bins; MC-unsampled
      bins are byte-identical on both paths (both take pref*J_C1), so Y3 cannot
      touch the empty-bin FILL pathology.

FORMAL (clean complete run): IT = -1 == dump iter 11.
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
TR = np.load(os.path.join(Y.OUT, "_cache_TR.npy"))
W = np.load(os.path.join(Y.OUT, "_cache_W.npy"))
Te = TR[IT, :, 14].copy()

# ---- M1: shape inside the pinned coarse bin 14 -----------------------------
print("=== M1: field shape inside T_e-pinned coarse bin 14 (728.8-905.6 A) ===")
sel = np.where(Y.CIDX == 14)[0]
print(f"  fine bins {sel[0]}..{sel[-1]}  lam {Y.LAM_MID_A[sel[-1]]:.1f}-{Y.LAM_MID_A[sel[0]]:.1f} A")
for s in (0, 8, 20, 30, 45, 49):
    jr, jc = J_raw[IT, s, sel], J_C1[IT, s, sel]
    # normalise both to unit bin energy so only SHAPE is compared
    nr = jr * Y.DNU[sel] / (jr * Y.DNU[sel]).sum() if (jr * Y.DNU[sel]).sum() > 0 else jr * 0
    nc = jc * Y.DNU[sel] / (jc * Y.DNU[sel]).sum() if (jc * Y.DNU[sel]).sum() > 0 else jc * 0
    # sel is ordered by INCREASING nu: sel[0] = red edge (903 A), sel[-1] = blue (731 A)
    n3 = len(sel) // 3
    print(f"  s{s:02d} T_e={Te[s]:6.0f}K W14={W[IT,s,14]:.3e} | "
          f"RED-third share raw={nr[:n3].sum():.4f} C1={nc[:n3].sum():.4f} "
          f"(C1/raw={nc[:n3].sum()/nr[:n3].sum() if nr[:n3].sum()>0 else np.nan:5.3f}) | "
          f"BLUE-third raw={nr[-n3:].sum():.4f} C1={nc[-n3:].sum():.4f} "
          f"(C1/raw={nc[-n3:].sum()/nr[-n3:].sum() if nr[-n3:].sum()>0 else np.nan:5.3f})")

print("\n  predicted Wien blue/red contrast inside the bin, exp(-h*dnu/kT_e):")
nu_b, nu_r = Y.NU_MID[sel[0]], Y.NU_MID[sel[-1]]
for s in (0, 8, 20, 30, 45, 49):
    x = Y.H_PLANCK * (nu_r - nu_b) / (Y.K_B * Te[s])
    print(f"    s{s:02d} T_e={Te[s]:6.0f}K -> B(blue)/B(red) suppression = {np.exp(-x):.4f}")

# where do the judgement edges sit inside their coarse bin?
print("\n  edge placement (fraction of the way from the RED to the BLUE edge of its coarse bin):")
for nm, lam in (("Si II", 758.506), ("Fe II", 765.914), ("Co II", 725.716),
                ("Ni II", 682.400), ("Cr II", 752.072), ("Mg II", 824.623),
                ("Ca II", 1044.365), ("Ti II", 913.292), ("S II", 531.257)):
    nu = Y.C_LIGHT / (lam * 1e-8)
    f = int(np.searchsorted(Y.NU_MID, nu))
    c = Y.CIDX[min(f, Y.NB - 1)]
    ss = np.where(Y.CIDX == c)[0]
    pos = (np.log(nu) - np.log(Y.NU_MID[ss[0]])) / (np.log(Y.NU_MID[ss[-1]]) - np.log(Y.NU_MID[ss[0]]))
    print(f"    {nm:6s} {lam:8.2f} A -> fine bin {f}, coarse bin {c:2d}, "
          f"position {pos:.2f} (0=red edge, 1=blue edge)")

# ---- M2: exact decomposition of the C2-GEMM difference ---------------------
print("\n=== M2: is the C2-GEMM gap confined to MC-sampled bins? ===")
samp = bfr[IT] > 0
# For a level with sigma>0 everywhere above threshold, the exact difference is
#   R_C2 - R_GEMM = sum_{sampled, above thr} (sigma*bfr - pref*J_C1)
#                 - sum_{below thr}          (pref*J_C1)      <- GEMM-only term
# Quantify the second (the GEMM K build omits the nu_thresh cut on cmfgen rows).
lv = Y.load_levels()
chi = Y.load_ioniz()
sg = Y.SigmaBF(Y.sigma_path())
SIG = np.memmap(Y.sigma_path(), dtype="<f8", mode="r", offset=sg.data_off,
                shape=(sg.n_lev, sg.n_freq))
PB = 4.0 * np.pi / (Y.H_PLANCK * Y.NU_MID) * Y.DNU
rows = []
for Z, ion, nm in [(14, 1, "Si II"), (26, 1, "Fe II"), (27, 1, "Co II"),
                   (16, 1, "S II"), (28, 1, "Ni II"), (20, 1, "Ca II")]:
    sub = lv[(lv.atomic_number == Z) & (lv.ion_number == ion)]
    chi_eV = chi[(Z, ion)]
    gidx = sub.gidx.values
    nu_th = (chi_eV - sub.energy_eV.values) * Y.EV_TO_ERG / Y.H_PLANCK
    sigma = np.array(SIG[gidx, :])
    below = (Y.NU_MID[None, :] < nu_th[:, None]) & (sigma > 0)
    for s in (0, 8, 20, 30, 45, 49):
        pref = sigma * PB[None, :]
        leak = (pref * J_C1[IT, s][None, :] * below).sum()
        tot = (pref * J_C1[IT, s][None, :] * (sigma > 0)).sum()
        d_samp = ((sigma * bfr[IT, s][None, :] - pref * J_C1[IT, s][None, :]) *
                  (samp[s][None, :] & (Y.NU_MID[None, :] >= nu_th[:, None]) &
                   (sigma > 0))).sum()
        rows.append(dict(ion=nm, shell=s, n_sigma_below_thresh=int(below.sum()),
                         GEMM_leak_below_thresh=leak, GEMM_total=tot,
                         leak_frac=leak / tot if tot > 0 else np.nan,
                         delta_from_sampled_bins=d_samp))
dc = pd.DataFrame(rows)
dc.to_csv(os.path.join(Y.OUT, "y2_decomposition.csv"), index=False)
print(dc.pivot(index="ion", columns="shell", values="leak_frac").round(6).to_string())
print("\n  (leak_frac = share of the GEMM integral that comes from bins BELOW the\n"
      "   level threshold, which the CMFGEN-sigma branch of the K build does not\n"
      "   cut -- lumina_nlte_gemm.cu:219-221 vs lumina_plasma.c:14608)")
print(f"\n  n_levels with any sigma>0 below their own threshold: "
      f"{dc.groupby('ion').n_sigma_below_thresh.first().to_dict()}")

# ---- empty-bin FILL survives Y3 -------------------------------------------
fill = (J_raw[IT] <= 0) & (J_C1[IT] > 0)
print(f"\n=== FILL bins (MC saw nothing, C1 publishes flux) survive Y3 unchanged ===")
print(f"  total fill bins at iter 11: {int(fill.sum())} of {fill.size}")
print(f"  every one has bfr==0 -> BOTH paths take pref*J_C1: "
      f"{bool((bfr[IT][fill] == 0).all())}")
for s in (0, 8, 20, 30, 45, 49):
    ss = fill[s]
    kern = (J_C1[IT, s] * PB)
    print(f"  s{s:02d}: n_fill={int(ss.sum()):4d}  fill share of the total "
          f"photoion kernel = {kern[ss].sum()/kern.sum():.4f}")
