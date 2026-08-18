"""Y2 step 10 (FORMAL only): composition scoping of the path-ratio table.

PRELIM_REPORT.md sec 2.1 tabulates C2/GEMM for 15 lower ions x 6 shells without
asking whether each ion HAS any population in that shell.  toy06 is a stratified
model, so most of those cells are empty ions.  This step establishes, from the
run's own dumps:
  (a) which of the 16 matrix-consumed lower ions carry n_ion > 0, per shell
      (lumina_ion_pops.csv, byte-identical to parity44 per VERDICT_NOTE.md)
  (b) the realized population-weighted C2/GEMM ratio on every shell where the
      ion actually exists (weighting derived in step9)
  (c) the ion-population-weighted TOTAL photoionization flux ratio per shell
      -- the single number that says what Y3 does to that shell.

Outputs y2_composition_scope.csv, y2_popratio_allshell.csv.
"""
import os
import numpy as np
import pandas as pd
import y2_common as Y

IT = -1
J_C1 = np.load(os.path.join(Y.OUT, "_cache_JC1.npy"))
bfr = np.load(os.path.join(Y.OUT, "_cache_bfr.npy"))
TR = np.load(os.path.join(Y.OUT, "_cache_TR.npy"))
Te = TR[IT, :, 14].copy()
NS = J_C1.shape[1]

lv, chi = Y.load_levels(), Y.load_ioniz()
sg = Y.SigmaBF(Y.sigma_path())
SIG = np.memmap(Y.sigma_path(), dtype="<f8", mode="r", offset=sg.data_off,
                shape=(sg.n_lev, sg.n_freq))
PB = 4.0 * np.pi / (Y.H_PLANCK * Y.NU_MID) * Y.DNU

ip = pd.read_csv(os.path.join(Y.RUN, "lumina_ion_pops.csv"))
lp = pd.read_csv(os.path.join(Y.RUN, "lumina_levelpop.csv"))

# the 16 lower ions that receive matrix R_bf (nlte_get_pairs base table)
CONSUMED = [(14, 1, "Si II"), (20, 1, "Ca II"), (26, 1, "Fe II"), (16, 1, "S II"),
            (27, 1, "Co II"), (28, 1, "Ni II"), (6, 1, "C II"), (12, 1, "Mg II"),
            (22, 1, "Ti II"), (24, 1, "Cr II"), (13, 1, "Al II"), (21, 1, "Sc II"),
            (23, 1, "V II"), (25, 1, "Mn II"), (8, 0, "O I"), (8, 1, "O II")]

# ---------------------------------------------------------------- (a) scoping
rows = []
for Z, ion, nm in CONSUMED:
    d = ip[(ip.Z == Z) & (ip.stage == ion)]
    per = d.set_index("shell_id").n_ion.reindex(range(NS)).fillna(0.0).values
    zel = ip[ip.Z == Z].groupby("shell_id").n_ion.sum().reindex(range(NS)).fillna(0.0).values
    live = np.where(per > 0)[0]
    rows.append(dict(ion=nm, Z=Z, ion_number=ion,
                     n_shells_elem_present=int((zel > 0).sum()),
                     n_shells_ion_pos=len(live),
                     shell_first=int(live[0]) if len(live) else -1,
                     shell_last=int(live[-1]) if len(live) else -1,
                     n_ion_s0=per[0], n_ion_s8=per[8], n_ion_s20=per[20],
                     n_ion_s30=per[30], n_ion_s45=per[45], n_ion_s49=per[49]))
sc = pd.DataFrame(rows)
sc.to_csv(os.path.join(Y.OUT, "y2_composition_scope.csv"), index=False)
pd.set_option("display.width", 240)
print("=== (a) which matrix-consumed lower ions actually exist, per shell ===")
print(sc.to_string(index=False))

REAL = [(Z, i, n) for Z, i, n in CONSUMED
        if sc.loc[sc.ion == n, "n_shells_ion_pos"].iloc[0] > 0]
print(f"\n  ions with ANY population anywhere: {[n for _, _, n in REAL]}")
print(f"  ions with ZERO population in all 50 shells: "
      f"{[n for _, _, n in CONSUMED if (Z_ := n) not in [x[2] for x in REAL]]}")

# --------------------------------------------- (b)(c) pop-weighted, all shells
rows = []
tot_c2 = np.zeros(NS)
tot_gm = np.zeros(NS)
for Z, ion, nm in REAL:
    sub = lv[(lv.atomic_number == Z) & (lv.ion_number == ion)]
    if len(sub) == 0 or (Z, ion) not in chi:
        continue
    chi_eV = chi[(Z, ion)]
    g, E = sub.gidx.values, sub.energy_eV.values
    nu_th = (chi_eV - E) * Y.EV_TO_ERG / Y.H_PLANCK
    ok = nu_th > 0
    sigma = np.array(SIG[g, :])
    has = sg.has[g].astype(bool)
    s0c = 7.91e-18 / float(ion + 1) ** 2
    kr = np.where(Y.NU_MID[None, :] >= nu_th[:, None],
                  s0c * (nu_th[:, None] / Y.NU_MID[None, :]) ** 3, 0.0)
    sigma = np.where(has[:, None], sigma, kr)
    pref = sigma * PB[None, :]
    above = (Y.NU_MID[None, :] >= nu_th[:, None]) & (sigma > 0)
    lpi = lp[(lp.Z == Z) & (lp.ion == ion)]
    for s in range(NS):
        d = lpi[lpi.shell == s].sort_values("level_num")
        if len(d) != len(sub):
            continue
        nk = d.n_k.values.astype(float)
        if nk.sum() <= 0:
            continue
        bf, Jc = bfr[IT, s], J_C1[IT, s]
        use = bf > 0
        c2 = np.where(above, np.where(use[None, :], sigma * bf[None, :],
                                      pref * Jc[None, :]), 0.0).sum(1)
        gm = np.where(sigma > 0, pref * Jc[None, :], 0.0).sum(1)
        c2[~ok] = gm[~ok] = 0.0
        # absolute ionization FLUX out of the ion [cm^-3 s^-1] = sum_lev n_lev*R_bf
        F2, FG = float((nk * c2).sum()), float((nk * gm).sum())
        tot_c2[s] += F2
        tot_gm[s] += FG
        rows.append(dict(ion=nm, Z=Z, ion_number=ion, shell=s, T_e=float(Te[s]),
                         n_ion=float(nk.sum()), flux_C2=F2, flux_GEMM=FG,
                         ratio_pop=F2 / FG if FG > 0 else np.nan,
                         Gamma_C2_pop=F2 / nk.sum(), Gamma_GEMM_pop=FG / nk.sum()))
pr = pd.DataFrame(rows)
pr.to_csv(os.path.join(Y.OUT, "y2_popratio_allshell.csv"), index=False)

print("\n=== (b) REALIZED pop-weighted C2/GEMM, only where the ion exists ===")
SH = [0, 4, 8, 10, 12, 16, 20, 25, 30, 35, 40, 45, 49]
pv = pr[pr.shell.isin(SH)].pivot(index="ion", columns="shell", values="ratio_pop")
print(pv.round(3).to_string())
print("\n=== Si II pop-weighted ratio, every shell it exists in ===")
si = pr[pr.ion == "Si II"].sort_values("shell")
print("  " + "  ".join(f"s{int(r.shell)}:{r.ratio_pop:.2f}" for r in si.itertuples()))
p1 = si[(si.shell >= 20) & (si.shell <= 35)]
print(f"  P1 window s20-s35: min={p1.ratio_pop.min():.3f} "
      f"median={p1.ratio_pop.median():.3f} max={p1.ratio_pop.max():.3f} "
      f"(n={len(p1)}); registered band was 1.6-3.5")
print(f"  fraction of the s20-s35 window ABOVE 3.5: "
      f"{(p1.ratio_pop > 3.5).mean():.2f}")

print("\n=== (c) shell-total photoionization flux ratio C2/GEMM (all real ions) ===")
tt = pd.DataFrame(dict(shell=np.arange(NS), flux_C2=tot_c2, flux_GEMM=tot_gm))
tt["ratio"] = np.where(tt.flux_GEMM > 0, tt.flux_C2 / tt.flux_GEMM, np.nan)
print(tt[tt.shell.isin(SH)].round(4).to_string(index=False))

print("\n=== P5 NULL check (s0, s8): |ratio-1| for every REAL ion with Gamma>1 ===")
for s in (0, 8):
    d = pr[(pr.shell == s) & (pr.Gamma_C2_pop > 1.0)]
    for r in d.itertuples():
        flag = "PASS" if abs(r.ratio_pop - 1) <= 0.035 else "**FAIL**"
        print(f"  s{s:02d} {r.ion:6s} Gamma_C2={r.Gamma_C2_pop:.3e} "
              f"ratio={r.ratio_pop:.4f} |d|={abs(r.ratio_pop-1)*100:5.2f}%  {flag}")
    d0 = pr[(pr.shell == s) & (pr.Gamma_C2_pop <= 1.0)]
    for r in d0.itertuples():
        print(f"  s{s:02d} {r.ion:6s} Gamma_C2={r.Gamma_C2_pop:.3e} "
              f"ratio={r.ratio_pop:.4f}  (excluded by the Gamma>1 filter)")
