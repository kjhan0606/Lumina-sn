"""Y2 step 3: R_bf(C2 path) vs R_bf(GEMM path), per level x shell.

C2   path (armed by LUMINA_C2_MATRIX_BF, i.e. Y3): lumina_plasma.c:14621-14636
GEMM path (this run's actual matrix route):        lumina_plasma.c:14601-14604
                                                   lumina_nlte_gemm.cu:217-229,416-443
Field staged into the GEMM = nlte->J_nu == the C1 dilute-BB refit written at
lumina_plasma.c:1049-1067 of the SAME call that normalised bfr (both live in
nlte_build_perbin_dilute_field), so pairing them at a fixed dump `iter` is exact.

PRELIMINARY-on-partial (iter 0-10 of a walltime-killed run).
"""
import os
import numpy as np
import pandas as pd
import y2_common as Y

IT = -1          # dump iter 10 = the last complete block
SHELLS = [0, 8, 20, 30, 45, 49]

J_raw = np.load(os.path.join(Y.OUT, "_cache_Jraw.npy"))
bfr = np.load(os.path.join(Y.OUT, "_cache_bfr.npy"))
W = np.load(os.path.join(Y.OUT, "_cache_W.npy"))
TR = np.load(os.path.join(Y.OUT, "_cache_TR.npy"))
mode = np.load(os.path.join(Y.OUT, "_cache_mode.npy"), allow_pickle=True)

J_C1 = Y.assemble_JC1(W, TR, mode, J_raw)
np.save(os.path.join(Y.OUT, "_cache_JC1.npy"), J_C1)

lv = Y.load_levels()
chi = Y.load_ioniz()
sig = Y.SigmaBF(Y.sigma_path())
print(f"[sigma] n_lev={sig.n_lev} n_freq={sig.n_freq} "
      f"nu=[{sig.nu_min:.4e},{sig.nu_max:.4e}] has_sigma={int(sig.has.sum())}")
assert sig.n_lev == len(lv)

# --- which ions actually receive matrix R_bf in THIS run ---------------------
# nlte_get_pairs base table (lumina_plasma.c:7268-7273) over the 31-slot ion
# list printed at stdout_partial.log:278-308.  The LOWER member of each pair is
# the only ion whose levels enter the R_bf loop (lumina_plasma.c:14564).
CONSUMED = [(14, 1), (20, 1), (26, 1), (16, 1), (27, 1), (28, 1), (6, 1),
            (12, 1), (22, 1), (24, 1), (13, 1), (21, 1), (23, 1), (25, 1),
            (8, 0), (8, 1)]

# requested judgement-relevant set + the consumed lower ions
TARGETS = [(14, 1, "Si II"), (14, 2, "Si III"), (16, 1, "S II"), (16, 2, "S III"),
           (26, 2, "Fe III"), (26, 3, "Fe IV"), (27, 2, "Co III"),
           (26, 1, "Fe II"), (27, 1, "Co II"), (28, 1, "Ni II"),
           (20, 1, "Ca II"), (12, 1, "Mg II"), (6, 1, "C II"), (8, 1, "O II"),
           (24, 1, "Cr II"), (22, 1, "Ti II")]

rows = []
levrows = []
for Z, ion, name in TARGETS:
    sub = lv[(lv.atomic_number == Z) & (lv.ion_number == ion)]
    if len(sub) == 0:
        print(f"[skip] {name}: no levels in levels.csv")
        continue
    chi_eV = chi.get((Z, ion))
    if chi_eV is None:
        print(f"[skip] {name}: no ionization energy entry")
        continue
    chi_erg = chi_eV * Y.EV_TO_ERG
    consumed = (Z, ion) in CONSUMED
    # ground + the lowest excited levels that carry CMFGEN sigma
    gidx_all = sub.gidx.values
    pick = [gidx_all[0]]
    for g in gidx_all[1:]:
        if len(pick) >= 4:
            break
        if sig.has[g] and lv.energy_eV.values[g] > 0:
            pick.append(g)
    for g in pick:
        row = lv.iloc[g]
        E = float(row.energy_eV)
        nu_th = (chi_erg - E * Y.EV_TO_ERG) / Y.H_PLANCK
        if nu_th <= 0:
            continue
        if sig.has[g]:
            s_row = sig.row(g)
            src = "cmfgen"
        else:
            # Kramers fallback, lumina_plasma.c:14617 (uses get_bf_sigma0; the
            # generic 7.91e-18/(stage+1)^2 branch is the documented default)
            s0 = 7.91e-18 / float(ion + 1) ** 2
            s_row = np.where(Y.NU_MID >= nu_th, s0 * (nu_th / Y.NU_MID) ** 3, 0.0)
            src = "kramers"
        for s in SHELLS:
            c2, gm, c2mc, c2fb = Y.R_bf_paths(s_row, nu_th, J_C1[IT, s], bfr[IT, s])
            c2_t, gm_t, _, _ = Y.R_bf_paths(s_row, nu_th, J_C1[IT, s], bfr[IT, s],
                                            apply_thresh_to_gemm=True)
            rows.append(dict(
                ion=name, Z=Z, ion_number=ion, consumed_in_matrix=consumed,
                level_number=int(row.level_number), gidx=int(g),
                E_eV=E, g=int(row.g), sigma_src=src,
                lam_thresh_A=Y.C_LIGHT / nu_th * 1e8,
                shell=s, R_bf_C2=c2, R_bf_GEMM=gm,
                ratio_C2_over_GEMM=(c2 / gm if gm > 0 else np.nan),
                ratio_vs_GEMM_thrcut=(c2_t / gm_t if gm_t > 0 else np.nan),
                C2_from_MC=c2mc, C2_from_fallback=c2fb,
                frac_C2_from_MC=(c2mc / c2 if c2 > 0 else np.nan)))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(Y.OUT, "y2_path_ratio_levels.csv"), index=False)

pd.set_option("display.width", 200)
print("\n=== R_bf(C2) / R_bf(GEMM), dump iter 10, GROUND levels ===")
gd = df[df.level_number == 0].pivot(index="ion", columns="shell",
                                    values="ratio_C2_over_GEMM")
print(gd.round(3).to_string())
print("\n=== fraction of R_bf(C2) supplied by MC-sampled bins (ground) ===")
print(df[df.level_number == 0].pivot(index="ion", columns="shell",
                                     values="frac_C2_from_MC").round(3).to_string())
print("\n=== absolute R_bf [s^-1], ground ===")
for nm in ["Si II", "S II", "Fe II", "Co II", "Fe III", "Co III", "Si III"]:
    d = df[(df.ion == nm) & (df.level_number == 0)]
    if len(d) == 0:
        continue
    print(f"  {nm:7s} lam_th={d.lam_thresh_A.iloc[0]:7.1f}A  " +
          "  ".join(f"s{int(r.shell):02d}:C2={r.R_bf_C2:.3e}/GEMM={r.R_bf_GEMM:.3e}"
                    for r in d.itertuples()))

print("\n=== excited levels, shell 8 / 45 ===")
ex = df[(df.level_number > 0) & (df.shell.isin([8, 45]))]
print(ex[["ion", "level_number", "E_eV", "lam_thresh_A", "shell",
          "R_bf_C2", "R_bf_GEMM", "ratio_C2_over_GEMM",
          "frac_C2_from_MC"]].round(4).to_string(index=False))
