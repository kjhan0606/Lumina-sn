"""Y2 step 7: is the headline ratio a stable feature or MC noise?
Repeat the per-ion Gamma ratio for every dumped iteration, and measure the
empty-bin FILL share over the bins each edge actually integrates.

PRELIMINARY-on-partial.
"""
import os
import numpy as np
import pandas as pd
import y2_common as Y

J_raw = np.load(os.path.join(Y.OUT, "_cache_Jraw.npy"))
J_C1 = np.load(os.path.join(Y.OUT, "_cache_JC1.npy"))
bfr = np.load(os.path.join(Y.OUT, "_cache_bfr.npy"))
TR = np.load(os.path.join(Y.OUT, "_cache_TR.npy"))
ni, ns, nb = J_raw.shape
SHELLS = [0, 8, 20, 30, 45, 49]

lv = Y.load_levels()
chi = Y.load_ioniz()
sg = Y.SigmaBF(Y.sigma_path())
SIG = np.memmap(Y.sigma_path(), dtype="<f8", mode="r", offset=sg.data_off,
                shape=(sg.n_lev, sg.n_freq))
PB = 4.0 * np.pi / (Y.H_PLANCK * Y.NU_MID) * Y.DNU

IONS = [(14, 1, "Si II"), (26, 1, "Fe II"), (27, 1, "Co II"), (28, 1, "Ni II"),
        (16, 1, "S II"), (20, 1, "Ca II"), (12, 1, "Mg II"), (24, 1, "Cr II"),
        (22, 1, "Ti II"), (25, 1, "Mn II"), (21, 1, "Sc II"), (8, 0, "O I")]

rows, fillrows = [], []
for Z, ion, nm in IONS:
    sub = lv[(lv.atomic_number == Z) & (lv.ion_number == ion)]
    if len(sub) == 0:
        continue
    chi_eV = chi[(Z, ion)]
    gidx = sub.gidx.values
    E = sub.energy_eV.values
    gw = sub.g.values.astype(float)
    nu_th = (chi_eV - E) * Y.EV_TO_ERG / Y.H_PLANCK
    sigma = np.array(SIG[gidx, :])
    has = sg.has[gidx].astype(bool)
    s0 = 7.91e-18 / float(ion + 1) ** 2
    kr = np.where(Y.NU_MID[None, :] >= nu_th[:, None],
                  s0 * (nu_th[:, None] / Y.NU_MID[None, :]) ** 3, 0.0)
    sigma = np.where(has[:, None], sigma, kr)
    above = Y.NU_MID[None, :] >= nu_th[:, None]
    ok = nu_th > 0
    pref = sigma * PB[None, :]
    # ground-edge kernel support: bins the GROUND level actually integrates
    gmask = (sigma[0] > 0) & above[0]

    for it in range(ni):
        Te = TR[it, :, 14]
        for s in SHELLS:
            Jc, bf = J_C1[it, s], bfr[it, s]
            use = bf > 0
            m = (sigma > 0) & above
            c2 = np.where(m, np.where(use[None, :], sigma * bf[None, :],
                                      pref * Jc[None, :]), 0.0).sum(1)
            gm = np.where(sigma > 0, pref * Jc[None, :], 0.0).sum(1)
            c2[~ok] = 0.0
            gm[~ok] = 0.0
            x = E * Y.EV_TO_ERG / (Y.K_B * Te[s]) if Te[s] > 0 else np.full_like(E, 1e3)
            w = np.where(x < 500.0, gw * np.exp(-np.clip(x, 0, 500)), 0.0)
            f = w / w.sum() if w.sum() > 0 else w
            G2, GG = float((f * c2).sum()), float((f * gm).sum())
            rows.append(dict(ion=nm, iter=it, shell=s,
                             ratio=G2 / GG if GG > 0 else np.nan,
                             ground_ratio=c2[0] / gm[0] if gm[0] > 0 else np.nan,
                             Gamma_C2=G2, Gamma_GEMM=GG))
            if it == ni - 1 and gmask.sum() > 0:
                fillsel = gmask & (J_raw[it, s] <= 0) & (Jc > 0)
                kern = pref[0] * Jc
                fillrows.append(dict(
                    ion=nm, shell=s, n_edge_bins=int(gmask.sum()),
                    n_fill_in_edge=int(fillsel.sum()),
                    fill_share_of_edge_kernel=(kern[fillsel].sum() / kern[gmask].sum()
                                               if kern[gmask].sum() > 0 else np.nan),
                    unsampled_share_of_edge_kernel=(
                        kern[gmask & ~use].sum() / kern[gmask].sum()
                        if kern[gmask].sum() > 0 else np.nan)))

rb = pd.DataFrame(rows)
rb.to_csv(os.path.join(Y.OUT, "y2_ratio_by_iter.csv"), index=False)
fr = pd.DataFrame(fillrows)
fr.to_csv(os.path.join(Y.OUT, "y2_fill_share_at_edge.csv"), index=False)

pd.set_option("display.width", 240)
print("=== Boltzmann-weighted Gamma ratio C2/GEMM by iteration, shell 30 ===")
print(rb[rb.shell == 30].pivot(index="ion", columns="iter", values="ratio")
      .round(2).to_string())
print("\n=== same, shell 20 ===")
print(rb[rb.shell == 20].pivot(index="ion", columns="iter", values="ratio")
      .round(2).to_string())
print("\n=== same, shell 45 ===")
print(rb[rb.shell == 45].pivot(index="ion", columns="iter", values="ratio")
      .round(2).to_string())
print("\n=== GROUND-level ratio by iteration, shell 30 ===")
print(rb[rb.shell == 30].pivot(index="ion", columns="iter", values="ground_ratio")
      .round(2).to_string())

print("\n=== stability of the ratio over the last 3 iters (8,9,10) ===")
last = rb[rb["iter"] >= ni - 3]
agg = last.groupby(["ion", "shell"]).ratio.agg(["mean", "std", "min", "max"])
agg["spread"] = agg["max"] / agg["min"]
print(agg.round(3).to_string())

print("\n=== empty-bin FILL + MC-unsampled share of the GROUND edge kernel (iter 10) ===")
print(fr.pivot(index="ion", columns="shell",
               values="fill_share_of_edge_kernel").round(4).to_string())
print("\n  MC-unsampled share of the same kernel (this is the part Y3 leaves on C1):")
print(fr.pivot(index="ion", columns="shell",
               values="unsampled_share_of_edge_kernel").round(4).to_string())
