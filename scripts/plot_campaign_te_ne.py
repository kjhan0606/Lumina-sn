#!/usr/bin/env python3
"""Campaign checkpoint figure: T_e / n_e profiles vs CMFGEN gold.

Usage:
  plot_campaign_te_ne.py OUT.png LABEL=jobid[:refdir] [LABEL=jobid ...]

jobid resolves logs/ddc15_pc_phase3_*_<jobid>/lumina_plasma_state.csv.
refdir (optional, after ':') = reference dir name under data/ for the shell
grid (default tardis_reference_ddc15_0p976d; use ..._v15k for extended runs).
Gold is velocity-interpolated from the DDC15 hydro file, so mixed grids
overlay correctly.
"""
import re, sys, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
HYDRO = f"{ROOT}/data/ddc15_hydro/DDC15_SN_HYDRO_DATA_0.976d"


def hydro(key, n=115):
    f = open(HYDRO).read()
    blk = f[f.find(key):].split("\n", 1)[1]
    return np.array([float(x) for x in
                     re.findall(r"[-+]?\d+\.\d+E[-+]\d+", blk)[:n]])


v_h = hydro("Velocity (km/s)")
o = np.argsort(v_h)
v_h = v_h[o]
T_h = (hydro("Temperature (10^4 K)") * 1e4)[o]
ne_h = hydro("Electron density")[o]

out = sys.argv[1]
runs = []
colors = ["tab:red", "tab:orange", "tab:green", "tab:blue", "tab:purple"]
for i, a in enumerate(sys.argv[2:]):
    lab, spec = a.split("=", 1)
    jid, _, ref = spec.partition(":")
    ref = ref or "tardis_reference_ddc15_0p976d"
    p = glob.glob(f"{ROOT}/logs/ddc15_pc_phase3_*_{jid}/lumina_plasma_state.csv")
    if not p:
        print(f"skip {lab}: no CSV for job {jid}")
        continue
    geo = pd.read_csv(f"{ROOT}/data/{ref}/geometry.csv")
    vc = 0.5 * (geo.v_inner + geo.v_outer).values / 1e5
    df = pd.read_csv(p[0])
    runs.append((lab, vc, df, colors[i % len(colors)]))

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
ax = axes[0]
mwin = v_h >= 19000
ax.plot(v_h[mwin] / 1e3, T_h[mwin], "k-", lw=2.6, label="CMFGEN gold")
for lab, vc, df, c in runs:
    ax.plot(vc / 1e3, df.T_e, color=c, lw=1.4, label=lab)
ax.set_xlim(19, 71); ax.set_ylim(1500, 5500)
ax.set_xlabel("v (10³ km/s)"); ax.set_ylabel("T_e (K)")
ax.set_title("T_e vs gold"); ax.legend(fontsize=8.5); ax.grid(alpha=0.25)

ax = axes[1]
ax.semilogy(v_h[mwin] / 1e3, ne_h[mwin], "k-", lw=2.6, label="CMFGEN gold")
for lab, vc, df, c in runs:
    ax.semilogy(vc / 1e3, df.n_e, color=c, lw=1.4, label=lab)
ax.set_xlim(19, 71); ax.set_ylim(2e4, 2e11)
ax.set_xlabel("v (10³ km/s)"); ax.set_ylabel("n_e (cm⁻³)")
ax.set_title("n_e vs gold"); ax.legend(fontsize=8.5)
ax.grid(alpha=0.25, which="both")

# stats footer (velocity bands, old-grid definitions)
lines = []
for lab, vc, df, c in runs:
    neC = np.interp(vc, v_h, ne_h); TC = np.interp(vc, v_h, T_h)
    d = np.log10(df.n_e.values / neC); t = (df.T_e.values - TC) / TC
    m_in = (vc >= 19312) & (vc < 27000)
    m_tr = (vc >= 27000) & (vc < 48000)
    m_ou = vc >= 48000
    rms = lambda x, m: np.sqrt(np.mean(x[m] ** 2)) if m.sum() else float("nan")
    lines.append(f"{lab}: ne dex inner {rms(d,m_in):.3f} / trans {rms(d,m_tr):.3f}"
                 f" / outer {rms(d,m_ou):.3f} | Te {100*np.sqrt(np.mean(t**2)):.1f}%")
fig.suptitle("DDC15 0.976d checkpoint\n" + "   ||   ".join(lines), fontsize=9, y=1.06)
fig.tight_layout()
fig.savefig(out, dpi=115, bbox_inches="tight")
print(out)
