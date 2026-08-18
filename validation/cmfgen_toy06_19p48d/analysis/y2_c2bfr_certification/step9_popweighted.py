"""Y2 step 9 (FORMAL only -- needs the clean run's lumina_levelpop.csv).

PRELIM_REPORT.md sec 4.4 item 3 asked for a level-population dump so that the
Boltzmann bracket (sec 2.1 pop-wtd / sec 2.2 ground) could be replaced by the
ACTUAL population weighting the matrix uses.  The clean parity46 run carries
LUMINA_LEVELPOP_DUMP=1, so that measurement is now possible.

Weighting derived from the consumer, not assumed
(src/lumina_plasma.c:14686-14689):
    ACM(ground_hi, sl) += R_bf * f_lev ;  f_lev = FRAC_OF = within-SL pop fraction
so the total ionization flux out of the ion is
    Phi_ion = sum_sl n_sl * sum_{lev in sl} R_bf(lev) * f_lev
            = sum_lev n_lev * R_bf(lev)                       (exact)
and the population-weighted mean rate is Phi_ion / sum_lev n_lev.
n_lev is read from lumina_levelpop.csv column n_k (final iteration state).

Outputs y2_gamma_popweighted.csv.
"""
import os
import numpy as np
import pandas as pd
import y2_common as Y

IT = -1
SHELLS = [0, 8, 20, 30, 45, 49]

J_C1 = np.load(os.path.join(Y.OUT, "_cache_JC1.npy"))
bfr = np.load(os.path.join(Y.OUT, "_cache_bfr.npy"))
TR = np.load(os.path.join(Y.OUT, "_cache_TR.npy"))
Te = TR[IT, :, 14].copy()

lv, chi = Y.load_levels(), Y.load_ioniz()
sg = Y.SigmaBF(Y.sigma_path())
SIG = np.memmap(Y.sigma_path(), dtype="<f8", mode="r", offset=sg.data_off,
                shape=(sg.n_lev, sg.n_freq))
PB = 4.0 * np.pi / (Y.H_PLANCK * Y.NU_MID) * Y.DNU

lp = pd.read_csv(os.path.join(Y.RUN, "lumina_levelpop.csv"))

IONS = [(14, 1, "Si II"), (20, 1, "Ca II"), (26, 1, "Fe II"), (16, 1, "S II"),
        (27, 1, "Co II"), (28, 1, "Ni II"), (6, 1, "C II"), (12, 1, "Mg II"),
        (22, 1, "Ti II"), (24, 1, "Cr II"), (13, 1, "Al II"), (21, 1, "Sc II"),
        (25, 1, "Mn II"), (8, 0, "O I"), (8, 1, "O II")]

rows = []
n_align_bad = 0
for Z, ion, nm in IONS:
    sub = lv[(lv.atomic_number == Z) & (lv.ion_number == ion)]
    if len(sub) == 0 or (Z, ion) not in chi:
        continue
    chi_eV = chi[(Z, ion)]
    g, E, gw = sub.gidx.values, sub.energy_eV.values, sub.g.values.astype(float)
    nu_th = (chi_eV - E) * Y.EV_TO_ERG / Y.H_PLANCK
    ok = nu_th > 0
    sigma = np.array(SIG[g, :])
    has = sg.has[g].astype(bool)
    s0 = 7.91e-18 / float(ion + 1) ** 2
    kr = np.where(Y.NU_MID[None, :] >= nu_th[:, None],
                  s0 * (nu_th[:, None] / Y.NU_MID[None, :]) ** 3, 0.0)
    sigma = np.where(has[:, None], sigma, kr)
    pref = sigma * PB[None, :]
    above = (Y.NU_MID[None, :] >= nu_th[:, None]) & (sigma > 0)

    lpi = lp[(lp.Z == Z) & (lp.ion == ion)]
    for s in SHELLS:
        d = lpi[lpi.shell == s].sort_values("level_num")
        if len(d) != len(sub):
            continue
        # ---- alignment audit: levels.csv vs levelpop.csv must be the same table
        bad = (int((np.abs(d.E_eV.values - np.round(E, 4)) > 1e-3).sum())
               + int((d.g.values != gw).sum())
               + int((d.has_sigma.values.astype(bool) != has).sum()))
        n_align_bad += bad
        nk = d.n_k.values.astype(float)

        bf = bfr[IT, s]
        Jc = J_C1[IT, s]
        use = bf > 0
        c2 = np.where(above, np.where(use[None, :], sigma * bf[None, :],
                                      pref * Jc[None, :]), 0.0).sum(1)
        gm = np.where(sigma > 0, pref * Jc[None, :], 0.0).sum(1)
        c2[~ok] = gm[~ok] = 0.0

        # (a) realized population weighting  (the matrix's own)
        Ntot = nk.sum()
        fp = nk / Ntot if Ntot > 0 else nk * 0.0
        # (b) Boltzmann weighting at T_e  (PRELIM sec 2.1, parity_gamma_phot)
        x = E * Y.EV_TO_ERG / (Y.K_B * Te[s]) if Te[s] > 0 else np.full_like(E, 1e3)
        w = np.where(x < 500.0, gw * np.exp(-np.clip(x, 0, 500)), 0.0)
        fb = w / w.sum() if w.sum() > 0 else w

        rows.append(dict(
            ion=nm, Z=Z, ion_number=ion, shell=s, T_e=float(Te[s]),
            n_levels=len(sub), align_mismatches=bad, n_tot=float(Ntot),
            Gamma_C2_pop=float((fp * c2).sum()), Gamma_GEMM_pop=float((fp * gm).sum()),
            ratio_pop=float((fp * c2).sum() / (fp * gm).sum())
            if (fp * gm).sum() > 0 else np.nan,
            Gamma_C2_boltz=float((fb * c2).sum()), Gamma_GEMM_boltz=float((fb * gm).sum()),
            ratio_boltz=float((fb * c2).sum() / (fb * gm).sum())
            if (fb * gm).sum() > 0 else np.nan,
            ratio_ground=float(c2[0] / gm[0]) if gm[0] > 0 else np.nan,
            ground_pop_share=float(nk[0] / Ntot) if Ntot > 0 else np.nan))

df = pd.DataFrame(rows)
df.to_csv(os.path.join(Y.OUT, "y2_gamma_popweighted.csv"), index=False)
pd.set_option("display.width", 240)
print(f"[align] levels.csv vs levelpop.csv mismatches (E_eV/g/has_sigma): {n_align_bad}")
print("\n=== REALIZED population-weighted Gamma_phot ratio C2/GEMM (final iter) ===")
print(df.pivot(index="ion", columns="shell", values="ratio_pop").round(3).to_string())
print("\n=== same rows, Boltzmann(T_e) weighting (PRELIM method) ===")
print(df.pivot(index="ion", columns="shell", values="ratio_boltz").round(3).to_string())
print("\n=== same rows, GROUND level only ===")
print(df.pivot(index="ion", columns="shell", values="ratio_ground").round(3).to_string())
print("\n=== ground level's share of the ion population ===")
print(df.pivot(index="ion", columns="shell", values="ground_pop_share").round(3).to_string())
print("\n=== is the realized value inside the PRELIM bracket [ground, boltz]? ===")
lo = np.minimum(df.ratio_boltz, df.ratio_ground)
hi = np.maximum(df.ratio_boltz, df.ratio_ground)
df["inside_bracket"] = (df.ratio_pop >= lo - 1e-9) & (df.ratio_pop <= hi + 1e-9)
ok = df.dropna(subset=["ratio_pop", "ratio_boltz", "ratio_ground"])
print(f"  inside: {int(ok.inside_bracket.sum())} / {len(ok)}")
print(ok[~ok.inside_bracket][["ion", "shell", "ratio_ground", "ratio_pop",
                              "ratio_boltz"]].round(3).to_string(index=False))
