#!/usr/bin/env python3
"""Pure-CMFGEN post-fix result figure (job 164263). Three panels:
  A. T_e(v): CMFGEN gas T vs blanket MC vs pure-CMFGEN (deterministic J).
  B. photosphere T_e[0] convergence across iters (the factor-2 fix climb).
  C. thick/thin-limit validation J/S vs radial tau (proof J->S at depth).
The factor-2 normalization fix made the thick limit exact (J/S->1) and pulled
the photosphere to -1.2% vs CMFGEN; the outer stays the shared RADEQ closure gap."""
import numpy as np, csv, re, os

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
v   = 0.5 * (geo["v_inner"] + geo["v_outer"]) / 1e5
ref_te = rd(f"{REF}/plasma_state.csv")["T_rad"]   # CMFGEN gas temperature
nsh = len(ref_te)
pc_te  = rd(f"{PC}/lumina_plasma_state.csv")["T_e"]
blk_te = rd(f"{BLK}/lumina_plasma_state.csv")["T_e"]

# --- parse per-iter T_e[0/24/48] from the [CMFGEN] iter trace ---
it_n, it0, it24, it48 = [], [], [], []
for line in open(f"{PC}/stdout.log"):
    m = re.search(r"\[CMFGEN\] iter\s+(\d+): T_e\[0\]=(\d+)K T_e\[24\]=(\d+)K T_e\[48\]=(\d+)K", line)
    if m:
        it_n.append(int(m.group(1))); it0.append(float(m.group(2)))
        it24.append(float(m.group(3))); it48.append(float(m.group(4)))
it_n = np.array(it_n)

# --- parse [CMFGEN-VALID] thick/thin rows ---
vs, vtau, vjb, vjs = [], [], [], []
for line in open(f"{PC}/stdout.log"):
    m = re.search(r"\[CMFGEN-VALID\] s=\s*(\d+) b=\s*\d+ .*tau_r=([\d.eE+-]+).*J/B=([\d.eE+-]+) J/S=([\d.eE+-]+)", line)
    if m:
        vs.append(int(m.group(1))); vtau.append(float(m.group(2)))
        vjb.append(float(m.group(3))); vjs.append(float(m.group(4)))
vs, vtau, vjs = np.array(vs), np.array(vtau), np.array(vjs)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(19, 6))
fig.suptitle("DDC15 0.976d pure-CMFGEN deterministic J — post factor-2 fix (job 164263): "
             "photosphere nailed (-1.2%), thick limit exact",
             fontsize=13, weight="bold")

# ----- Panel A: T_e(v) -----
a = ax[0]
a.plot(v, ref_te, "k", marker="o", ms=3, lw=1.8, label="CMFGEN gas T (truth)")
a.plot(v, blk_te, "#3898EC", marker="D", ms=3, lw=1.0, alpha=0.8, label="blanket MC")
a.plot(v, pc_te,  "#D97757", marker="s", ms=3, lw=1.1, alpha=0.9, label="pure CMFGEN (det. J)")
a.axvspan(v[24], v[48], color="purple", alpha=0.07)
a.set_xlabel("velocity [km/s]"); a.set_ylabel("T_e [K]"); a.set_ylim(0, 7000)
a.set_title("A. T_e(v) — photosphere fixed, outer = shared RADEQ gap")
a.legend(fontsize=9); a.grid(alpha=0.3)
a.annotate(f"sh0: {pc_te[0]:.0f}K (-1.2%)\nCMFGEN {ref_te[0]:.0f}K",
           xy=(v[0], pc_te[0]), xytext=(v[5], 5600), fontsize=9,
           arrowprops=dict(arrowstyle="->", color="#D97757"))

# ----- Panel B: photosphere convergence -----
b = ax[1]
b.axhline(ref_te[0], color="k", ls="--", lw=1.5, label=f"CMFGEN sh0 {ref_te[0]:.0f}K")
b.plot(it_n, it0, "#D97757", marker="s", ms=5, lw=1.6, label="pure CMFGEN T_e[0]")
b.plot(it_n, it24, "#4EC9B0", marker="^", ms=4, lw=1.0, alpha=0.8, label="T_e[24] (mid-outer)")
b.plot(it_n, it48, "#7E84A3", marker="v", ms=4, lw=1.0, alpha=0.8, label="T_e[48] (far outer)")
b.axhline(3991, color="#D97757", ls=":", lw=1.0, alpha=0.6)
b.set_xlabel("iteration"); b.set_ylabel("T_e [K]"); b.set_ylim(800, 4700)
b.set_title("B. convergence — sh0 climbs 3991->4379K (factor-2 fix)")
b.legend(fontsize=9); b.grid(alpha=0.3)

# ----- Panel C: thick-limit validation -----
c = ax[2]
c.axhline(1.0, color="k", ls="--", lw=1.5, label="ideal thick limit J/S=1")
# colour by shell
cmap = {0: "#D97757", 12: "#4EC9B0", 24: "#3898EC", 48: "#7E84A3"}
lab  = {0: "s=0 (photosphere)", 12: "s=12", 24: "s=24", 48: "s=48 (outer)"}
for s in [0, 12, 24, 48]:
    mask = vs == s
    order = np.argsort(vtau[mask])
    c.plot(vtau[mask][order], vjs[mask][order], marker="o", ms=7, lw=1.0,
           color=cmap[s], label=lab[s])
c.set_xscale("log"); c.set_yscale("log")
c.set_xlabel(r"radial optical depth $\tau_r$"); c.set_ylabel("J / S  (formal-solve closure)")
c.set_title("C. thick(J/S->1) / thin(scatter floor) validation")
c.axvspan(1.0, 1e6, color="green", alpha=0.06)
c.text(50, 1.4, "thick: J/S=1.00 (FIXED)\nwas 0.50 pre-fix", fontsize=9, color="green")
c.text(2e-6, 1e4, "thin-UV J floor\n(bug #2, open)", fontsize=8, color="#aa3333")
c.legend(fontsize=8, loc="lower left"); c.grid(alpha=0.3, which="both")

plt.tight_layout(rect=[0, 0, 1, 0.95])
out = f"{ROOT}/figures/2026-06-07_ddc15_pure_cmfgen_result.png"
plt.savefig(out, dpi=130); print("saved", out)

# console recap
oo = slice(24, 49)
rms = np.sqrt(np.mean(((pc_te[oo]-ref_te[oo])/ref_te[oo])**2))*100
print(f"pure CMFGEN: T_e[0]={pc_te[0]:.0f}K ({100*(pc_te[0]/ref_te[0]-1):+.1f}%)  "
      f"outer sh24-48 mean={np.mean(pc_te[oo]):.0f}K (CMFGEN {np.mean(ref_te[oo]):.0f}K)  RMS={rms:.1f}%")
