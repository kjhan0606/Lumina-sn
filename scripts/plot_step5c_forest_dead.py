#!/usr/bin/env python3
"""Step-5c: forest-overlap DEAD + floor NOT the cap.
Left: super-thermal optical count flat at 8 across 3.4x line range.
Right: Fe II 5170 sub-thermal (NOT floored, NOT pure-scatter W*B) -> rate-network problem."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

runs = {  # TAUMIN -> (job, surviving lines)
    "0.1": ("169884", 8983), "0.01": ("169911", 16441), "1e-3": ("169914", 30547)}
base = "logs/ddc15_pc_phase3_jnul0_radls1_linere1_ratio1.0_pi1_fz1_"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

# Panel 1: super-thermal count vs surviving lines (flat = forest-overlap dead)
xs, ys = [], []
for tm, (job, nlines) in runs.items():
    f = np.genfromtxt(f"{base}{job}/lumina_sl_vs_B.csv", delimiter=",", names=True)
    m = (f["lambda_A"] >= 4000) & (f["lambda_A"] <= 7000) & (f["Sl_over_B"] > 1.05)
    n = int(m.sum())
    xs.append(nlines); ys.append(n)
    ax1.annotate(f"TAUMIN={tm}", (nlines, n), textcoords="offset points", xytext=(0, 12),
                 ha="center", fontsize=8)
ax1.plot(xs, ys, "o-", color="#D97757", lw=2, ms=10)
ax1.set_ylim(0, 20)
ax1.set_xlabel("UV pump lines deposited (surviving TAUMIN)")
ax1.set_ylabel("# super-thermal optical lines (S_l/B>1.05)")
ax1.set_title("forest-overlap DEAD: 3.4x more UV lines -> ZERO change\n(8 lines, max 1.45, byte-identical)")
ax1.grid(alpha=0.3)
ax1.text(0.5, 0.5, "Fe II 2382A pump tau=1.2e5\nALWAYS in -> not a line-count issue",
         transform=ax1.transAxes, ha="center", fontsize=9,
         bbox=dict(boxstyle="round", fc="#FFF3CD", ec="#FFC107"))

# Panel 2: 5170 sub-thermal vs W*B scatter and vs B thermal
f = np.genfromtxt(f"{base}169914/lumina_sl_vs_B.csv", delimiter=",", names=True)
m = (f["lambda_A"] >= 5169) & (f["lambda_A"] <= 5172)
sh = f["shell"][m].astype(int); slb = f["Sl_over_B"][m]
o = np.argsort(sh); sh, slb = sh[o], slb[o]
ps = np.genfromtxt(f"{base}169914/lumina_plasma_state.csv", delimiter=",", names=True)
W = ps["W"]
sel = (sh >= 8) & (sh <= 20)
ax2.axhline(1.0, color="gray", ls=":", label="thermal S_l=B (need pump ABOVE this)")
ax2.plot(sh[sel], slb[sel], "s-", color="#3898EC", lw=2, ms=6, label="Fe II 5170 S_l/B (actual)")
ax2.plot(sh[sel], W[sh[sel]], "^--", color="#4EC9B0", lw=1.5, label="W (pure-scatter floor W*B/B)")
ax2.fill_between(sh[sel], W[sh[sel]], slb[sel], color="#3898EC", alpha=0.12)
ax2.set_xlabel("shell"); ax2.set_ylabel("S_l / B(T_e)")
ax2.set_title("Fe II 5170 (main green carrier) = SUB-thermal\nNOT floored (would be =1.0), NOT pumped (would be >1.0)")
ax2.legend(fontsize=8.5, loc="upper right"); ax2.grid(alpha=0.3)
ax2.annotate("dilute J + collisions\n(~2.5x W), still << B\n= NO fluorescence pump",
             xy=(14, 0.55), xytext=(15, 0.85), fontsize=8, color="#3898EC",
             arrowprops=dict(arrowstyle="->", color="#3898EC"))

fig.suptitle("Step-5c (169914): forest-overlap REFUTED + floor NOT the cap -> rate-network is the bottleneck",
             fontsize=12, weight="bold")
fig.tight_layout()
out = "figures/2026-06-26_step5c_forest_dead.png"
fig.savefig(out, dpi=110, bbox_inches="tight")
print("wrote", out)
print("super-thermal counts vs lines:", list(zip(xs, ys)))
