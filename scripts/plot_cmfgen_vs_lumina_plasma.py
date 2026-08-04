#!/usr/bin/env python3
"""Per-shell physical-quantity comparison CMFGEN reference vs LUMINA, DDC15 0.976d.
Panels: (A) temperature [CMFGEN gas T_rad vs LUMINA T_e + T_rad + blanket MC],
(B) electron density n_e, (C) dilution factor W, (D) LUMINA/CMFGEN ratios.
LUMINA primary = pure-CMFGEN deterministic-J run 164263 (post factor-2 fix);
blanket MC 164032 shown for context. CMFGEN reference 'T_rad' column IS the
CMFGEN gas/electron temperature; n_e from electron_densities.csv."""
import numpy as np, csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
REF  = f"{ROOT}/data/tardis_reference_ddc15_0p976d"
PC   = f"{ROOT}/logs/ddc15_pure_cmfgen_164263"
BLK  = f"{ROOT}/logs/ddc15_radeq_blanket_164032"

def rd(p):
    r = csv.reader(open(p)); h = next(r); c = {k: [] for k in h}
    for row in r:
        for k, v in zip(h, row): c[k].append(float(v))
    return {k: np.array(v) for k, v in c.items()}

geo = rd(f"{REF}/geometry.csv")
v   = 0.5 * (geo["v_inner"] + geo["v_outer"]) / 1e5   # km/s

ref   = rd(f"{REF}/plasma_state.csv")
ref_ne = rd(f"{REF}/electron_densities.csv")["n_e"]
ref_T  = ref["T_rad"]      # CMFGEN gas temperature
ref_W  = ref["W"]

pc  = rd(f"{PC}/lumina_plasma_state.csv")
blk = rd(f"{BLK}/lumina_plasma_state.csv")

fig, ax = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("DDC15 0.976d — CMFGEN reference vs LUMINA per-shell plasma state",
             fontsize=14, weight="bold")
oo = slice(24, 49)

# A. temperature
a = ax[0, 0]
a.plot(v, ref_T, "k", marker="o", ms=3, lw=1.8, label="CMFGEN gas T (truth)")
a.plot(v, pc["T_e"], "#D97757", marker="s", ms=3, lw=1.2, label="LUMINA T_e (pure-CMFGEN)")
a.plot(v, pc["T_rad"], "#8888cc", marker=".", ms=3, lw=0.9, alpha=0.7, label="LUMINA T_rad (pure-CMFGEN)")
a.plot(v, blk["T_e"], "#3898EC", marker="D", ms=3, lw=0.9, alpha=0.6, label="LUMINA T_e (blanket MC)")
a.set_ylabel("temperature [K]"); a.set_ylim(0, 7000)
a.set_title("A. temperature"); a.legend(fontsize=8); a.grid(alpha=0.3)

# B. n_e
b = ax[0, 1]
b.plot(v, ref_ne, "k", marker="o", ms=3, lw=1.8, label="CMFGEN n_e (truth)")
b.plot(v, pc["n_e"], "#D97757", marker="s", ms=3, lw=1.2, label="LUMINA n_e (pure-CMFGEN)")
b.plot(v, blk["n_e"], "#3898EC", marker="D", ms=3, lw=0.9, alpha=0.6, label="LUMINA n_e (blanket MC)")
b.set_yscale("log"); b.set_ylabel("electron density n_e [cm^-3]")
b.set_title("B. electron density"); b.legend(fontsize=8); b.grid(alpha=0.3, which="both")

# C. W
c = ax[1, 0]
c.plot(v, ref_W, "k", marker="o", ms=3, lw=1.8, label="CMFGEN W")
c.plot(v, pc["W"], "#D97757", marker="s", ms=3, lw=1.2, label="LUMINA W (pure-CMFGEN)")
c.set_xlabel("velocity [km/s]"); c.set_ylabel("dilution factor W")
c.set_title("C. dilution factor"); c.legend(fontsize=8); c.grid(alpha=0.3)

# D. ratios
d = ax[1, 1]
d.axhline(1.0, color="k", ls="--", lw=1.2)
d.plot(v, pc["T_e"]/ref_T, "#D97757", marker="s", ms=3, lw=1.2, label="T_e ratio (pure-CMFGEN/CMFGEN)")
d.plot(v, pc["n_e"]/ref_ne, "#4EC9B0", marker="^", ms=3, lw=1.2, label="n_e ratio (pure-CMFGEN/CMFGEN)")
d.plot(v, blk["n_e"]/ref_ne, "#3898EC", marker="D", ms=3, lw=0.8, alpha=0.5, label="n_e ratio (blanket MC/CMFGEN)")
d.set_xlabel("velocity [km/s]"); d.set_ylabel("LUMINA / CMFGEN")
d.set_yscale("log"); d.set_title("D. ratio to CMFGEN (1.0 = perfect)")
d.legend(fontsize=8); d.grid(alpha=0.3, which="both")
for axx in ax.flat:
    axx.axvspan(v[24], v[48], color="purple", alpha=0.05)

plt.tight_layout(rect=[0, 0, 1, 0.96])
out = f"{ROOT}/figures/2026-06-07_ddc15_cmfgen_vs_lumina_plasma.png"
plt.savefig(out, dpi=130); print("saved", out)

def stats(name, lum, ref_):
    r = lum / ref_
    print(f"{name}: sh0 {lum[0]:.3g} vs {ref_[0]:.3g} ({100*(r[0]-1):+.1f}%)  "
          f"outer24-48 mean ratio={np.mean(r[oo]):.3f}  RMS%={100*np.sqrt(np.mean((r[oo]-1)**2)):.1f}")
print("--- pure-CMFGEN (164263) ---")
stats("T_e ", pc["T_e"], ref_T)
stats("n_e ", pc["n_e"], ref_ne)
print("--- blanket MC (164032) ---")
stats("T_e ", blk["T_e"], ref_T)
stats("n_e ", blk["n_e"], ref_ne)
