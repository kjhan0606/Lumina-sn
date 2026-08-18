"""Y2 step 8: independent re-derivation of the path ratio (does not reuse
step5's code path), plus the headline summary table.

Closed form implied by steps 2 + 6:
    R_bf(C2)   ~=  sum_{sampled} pref*J_raw  +  sum_{unsampled} pref*J_C1
    R_bf(GEMM) ==  sum_{all}     pref*J_C1
i.e. Y3 is exactly "use the raw MC field wherever the MC sampled it".
If this reproduces step5's ratio to <0.3% the two computations agree and the
interpretation is certified.

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
TR = np.load(os.path.join(Y.OUT, "_cache_TR.npy"))
Te = TR[IT, :, 14]
SHELLS = [0, 8, 20, 30, 45, 49]

lv, chi = Y.load_levels(), Y.load_ioniz()
sg = Y.SigmaBF(Y.sigma_path())
SIG = np.memmap(Y.sigma_path(), dtype="<f8", mode="r", offset=sg.data_off,
                shape=(sg.n_lev, sg.n_freq))
PB = 4.0 * np.pi / (Y.H_PLANCK * Y.NU_MID) * Y.DNU

IONS = [(14, 1, "Si II"), (26, 1, "Fe II"), (27, 1, "Co II"), (28, 1, "Ni II"),
        (16, 1, "S II"), (20, 1, "Ca II"), (12, 1, "Mg II"), (24, 1, "Cr II"),
        (22, 1, "Ti II"), (25, 1, "Mn II"), (21, 1, "Sc II"), (8, 0, "O I"),
        (13, 1, "Al II"), (6, 1, "C II"), (8, 1, "O II")]

rows = []
for Z, ion, nm in IONS:
    sub = lv[(lv.atomic_number == Z) & (lv.ion_number == ion)]
    if len(sub) == 0:
        continue
    chi_eV = chi[(Z, ion)]
    g, E, gw = sub.gidx.values, sub.energy_eV.values, sub.g.values.astype(float)
    nu_th = (chi_eV - E) * Y.EV_TO_ERG / Y.H_PLANCK
    sigma = np.array(SIG[g, :])
    has = sg.has[g].astype(bool)
    s0 = 7.91e-18 / float(ion + 1) ** 2
    kr = np.where(Y.NU_MID[None, :] >= nu_th[:, None],
                  s0 * (nu_th[:, None] / Y.NU_MID[None, :]) ** 3, 0.0)
    sigma = np.where(has[:, None], sigma, kr)
    pref = sigma * PB[None, :]
    above = (Y.NU_MID[None, :] >= nu_th[:, None]) & (sigma > 0)
    ok = nu_th > 0
    for s in SHELLS:
        samp = bfr[IT, s] > 0
        # closed form -- built ONLY from J_raw / J_C1, never touching bfr values
        A = np.where(above & samp[None, :], pref * J_raw[IT, s][None, :], 0.0).sum(1)
        B = np.where(above & ~samp[None, :], pref * J_C1[IT, s][None, :], 0.0).sum(1)
        c2_cf = A + B
        gm = np.where(sigma > 0, pref * J_C1[IT, s][None, :], 0.0).sum(1)
        # direct form -- uses bfr, as the consumer does
        c2_dir = np.where(above, np.where(samp[None, :], sigma * bfr[IT, s][None, :],
                                          pref * J_C1[IT, s][None, :]), 0.0).sum(1)
        c2_cf[~ok] = c2_dir[~ok] = gm[~ok] = 0.0
        x = E * Y.EV_TO_ERG / (Y.K_B * Te[s])
        w = np.where(x < 500.0, gw * np.exp(-np.clip(x, 0, 500)), 0.0)
        f = w / w.sum() if w.sum() > 0 else w
        G_cf, G_dir, G_gm = (f * c2_cf).sum(), (f * c2_dir).sum(), (f * gm).sum()
        rows.append(dict(ion=nm, shell=s, T_e=float(Te[s]),
                         lam_edge_A=Y.C_LIGHT / (chi_eV * Y.EV_TO_ERG / Y.H_PLANCK) * 1e8,
                         ratio_direct=G_dir / G_gm if G_gm > 0 else np.nan,
                         ratio_closedform=G_cf / G_gm if G_gm > 0 else np.nan,
                         agree_rel=abs(G_cf - G_dir) / G_dir if G_dir > 0 else np.nan,
                         Gamma_C2=G_dir, Gamma_GEMM=G_gm,
                         sampled_share_of_kernel=float(
                             (pref[0] * J_C1[IT, s] * (above[0] & samp)).sum() /
                             (pref[0] * J_C1[IT, s] * above[0]).sum())
                         if (pref[0] * J_C1[IT, s] * above[0]).sum() > 0 else np.nan))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(Y.OUT, "y2_headline_summary.csv"), index=False)
pd.set_option("display.width", 240)
print("=== closed-form vs direct C2 integral (must agree; only in-bin nu weighting differs) ===")
print(f"  max relative disagreement over all ion x shell: {df.agree_rel.max():.3e}")
print(f"  median: {df.agree_rel.median():.3e}")
print("\n=== HEADLINE: Boltzmann-weighted Gamma_phot ratio C2/GEMM, dump iter 11 ===")
print(df.pivot(index="ion", columns="shell", values="ratio_direct").round(3).to_string())
print("\n=== share of each ground edge's photoion kernel that the MC actually sampled ===")
print(df.pivot(index="ion", columns="shell",
               values="sampled_share_of_kernel").round(3).to_string())
