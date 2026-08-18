"""Y2 step 5: full consumed-set census + Boltzmann-weighted per-ion Gamma_phot
on BOTH paths.

Weighting = exactly parity_gamma_phot (lumina_plasma.c:1746-1786):
    f_lev = g_l * exp(-E_l/kT_e) / Z_part,  Z_part = sum_l g_l exp(-E_l/kT_e)
    (B3 under parity: partition functions at T_e, undiluted W=1 -> this closed
     form is the code's own definition, plasma.c B3 banner)
T_e is READ OUT of the C1 dump: LUMINA_C1_SUPERBIN_TEPIN sets T_R := T_e(shell)
for the pinned coarse bins (lumina_plasma.c:952-956); cross-checked against
stdout.log:36005 '[CMFGEN] iter 11: T_e[0]=21203K T_e[25]=8501K T_e[49]=13066K'
(step9 does this crosscheck mechanically).

This is the internal-inconsistency measure of the run as configured:
  * R1 ionization closure   -> already integrates the C2 estimator (plasma.c:1731)
  * NLTE matrix R_bf        -> integrates the C1 refit via GEMM (plasma.c:14601)
Y3 (LUMINA_C2_MATRIX_BF=1) moves the second onto the first's field.

FORMAL (clean complete run): IT = -1 == dump iter 11.
"""
import os
import numpy as np
import pandas as pd
import y2_common as Y

IT = -1
SHELLS = [0, 8, 20, 30, 45, 49]

J_raw = np.load(os.path.join(Y.OUT, "_cache_Jraw.npy"))
J_C1 = np.load(os.path.join(Y.OUT, "_cache_JC1.npy"))
bfr = np.load(os.path.join(Y.OUT, "_cache_bfr.npy"))
TR = np.load(os.path.join(Y.OUT, "_cache_TR.npy"))
Te = TR[IT, :, 14].copy()           # pinned coarse bin == T_e(shell)
print("T_e (from pinned C1 bin 14):", " ".join(f"s{s}={Te[s]:.0f}" for s in SHELLS))

lv = Y.load_levels()
chi = Y.load_ioniz()
sg = Y.SigmaBF(Y.sigma_path())
# whole-file memmap for bulk row access
SIG = np.memmap(Y.sigma_path(), dtype="<f8", mode="r",
                offset=sg.data_off, shape=(sg.n_lev, sg.n_freq))

CONSUMED = [(14, 1, "Si II"), (20, 1, "Ca II"), (26, 1, "Fe II"), (16, 1, "S II"),
            (27, 1, "Co II"), (28, 1, "Ni II"), (6, 1, "C II"), (12, 1, "Mg II"),
            (22, 1, "Ti II"), (24, 1, "Cr II"), (13, 1, "Al II"), (21, 1, "Sc II"),
            (23, 1, "V II"), (25, 1, "Mn II"), (8, 0, "O I"), (8, 1, "O II")]
# not consumed in this configuration, reported for reference only
REFONLY = [(14, 2, "Si III"), (16, 2, "S III"), (26, 2, "Fe III"),
           (26, 3, "Fe IV"), (27, 2, "Co III")]

PREF_BASE = 4.0 * np.pi / (Y.H_PLANCK * Y.NU_MID) * Y.DNU     # x sigma = pref

lev_rows, ion_rows = [], []
for Z, ion, name, consumed in ([(*t, True) for t in CONSUMED] +
                               [(*t, False) for t in REFONLY]):
    sub = lv[(lv.atomic_number == Z) & (lv.ion_number == ion)]
    if len(sub) == 0:
        continue
    chi_eV = chi.get((Z, ion))
    if chi_eV is None:
        continue
    g = sub.gidx.values
    E = sub.energy_eV.values
    gw = sub.g.values.astype(float)
    nu_th = (chi_eV - E) * Y.EV_TO_ERG / Y.H_PLANCK
    ok = nu_th > 0
    sigma = np.array(SIG[g, :])
    has = sg.has[g].astype(bool)
    # Kramers fallback for levels without CMFGEN data (lumina_plasma.c:14617)
    s0 = 7.91e-18 / float(ion + 1) ** 2
    kr = np.where(Y.NU_MID[None, :] >= nu_th[:, None],
                  s0 * (nu_th[:, None] / Y.NU_MID[None, :]) ** 3, 0.0)
    sigma = np.where(has[:, None], sigma, kr)
    above = Y.NU_MID[None, :] >= nu_th[:, None]
    m_c2 = (sigma > 0) & above
    m_gm = (sigma > 0)                    # GEMM K has no nu_th cut on cmfgen rows
    pref = sigma * PREF_BASE[None, :]

    for s in SHELLS:
        Jc, bf = J_C1[IT, s], bfr[IT, s]
        use = bf > 0
        c2 = np.where(m_c2, np.where(use[None, :], sigma * bf[None, :],
                                     pref * Jc[None, :]), 0.0).sum(1)
        gm = np.where(m_gm, pref * Jc[None, :], 0.0).sum(1)
        c2mc = np.where(m_c2 & use[None, :], sigma * bf[None, :], 0.0).sum(1)
        c2[~ok] = 0.0
        gm[~ok] = 0.0
        # Boltzmann weights at T_e (parity_gamma_phot / B3)
        x = E * Y.EV_TO_ERG / (Y.K_B * Te[s])
        w = np.where(x < 500.0, gw * np.exp(-np.clip(x, 0, 500)), 0.0)
        Zp = w.sum()
        f = w / Zp if Zp > 0 else w * 0
        G_c2, G_gm = float((f * c2).sum()), float((f * gm).sum())
        ion_rows.append(dict(
            ion=name, Z=Z, ion_number=ion, consumed_in_matrix=consumed, shell=s,
            T_e=float(Te[s]), n_levels=int(len(sub)), n_with_cmfgen=int(has.sum()),
            lam_edge_A=Y.C_LIGHT / (chi_eV * Y.EV_TO_ERG / Y.H_PLANCK) * 1e8,
            Gamma_C2=G_c2, Gamma_GEMM=G_gm,
            ratio_C2_over_GEMM=(G_c2 / G_gm if G_gm > 0 else np.nan),
            frac_Gamma_from_MC=(float((f * c2mc).sum()) / G_c2 if G_c2 > 0 else np.nan),
            R_bf_ground_C2=float(c2[0]), R_bf_ground_GEMM=float(gm[0]),
            # unweighted spread over the ion's levels
            lev_ratio_p10=float(np.nanpercentile(np.where(gm > 0, c2 / np.where(gm > 0, gm, 1), np.nan), 10)),
            lev_ratio_med=float(np.nanmedian(np.where(gm > 0, c2 / np.where(gm > 0, gm, 1), np.nan))),
            lev_ratio_p90=float(np.nanpercentile(np.where(gm > 0, c2 / np.where(gm > 0, gm, 1), np.nan), 90)),
            n_lev_gm_pos=int((gm > 0).sum()),
            n_lev_c2_pos=int((c2 > 0).sum())))

ion = pd.DataFrame(ion_rows)
ion.to_csv(os.path.join(Y.OUT, "y2_gamma_per_ion.csv"), index=False)

pd.set_option("display.width", 240)
print("\n=== Boltzmann(T_e)-weighted Gamma_phot ratio  C2 / GEMM  (dump iter 11) ===")
print("    (consumed_in_matrix=True -> Y3 changes this rate; False -> reference only)")
p = ion[ion.consumed_in_matrix].pivot(index="ion", columns="shell",
                                      values="ratio_C2_over_GEMM")
print(p.round(3).to_string())
print("\n  reference-only ions (no matrix R_bf in this config):")
print(ion[~ion.consumed_in_matrix].pivot(index="ion", columns="shell",
                                         values="ratio_C2_over_GEMM").round(3).to_string())

print("\n=== absolute Gamma_phot [s^-1], C2 path (dump iter 11) ===")
print(ion[ion.consumed_in_matrix].pivot(index="ion", columns="shell",
                                        values="Gamma_C2").map(
    lambda v: f"{v:.3e}").to_string())
print("\n=== absolute Gamma_phot [s^-1], GEMM path ===")
print(ion[ion.consumed_in_matrix].pivot(index="ion", columns="shell",
                                        values="Gamma_GEMM").map(
    lambda v: f"{v:.3e}").to_string())
print("\n=== fraction of Gamma_C2 delivered by MC-sampled bins ===")
print(ion[ion.consumed_in_matrix].pivot(index="ion", columns="shell",
                                        values="frac_Gamma_from_MC").round(3).to_string())
print("\n=== per-level ratio spread (p10 / median / p90) at shell 30 ===")
d = ion[(ion.shell == 30) & ion.consumed_in_matrix]
print(d[["ion", "lam_edge_A", "n_levels", "n_lev_gm_pos", "lev_ratio_p10",
         "lev_ratio_med", "lev_ratio_p90"]].round(3).to_string(index=False))
