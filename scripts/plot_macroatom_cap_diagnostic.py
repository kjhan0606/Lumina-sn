#!/usr/bin/env python3
"""Diagnostic figure for the macroatom iter3-hang root cause + M2 cap fix.
Left: the DDC15 iron-curtain tau distribution that first appears after the
end-of-iter1 nebular/Saha recompute (mode-independent). Right: per-iteration
transport time + cap-hit fraction from the cap=200 smoke (job 160855), showing
the ~340x cost jump when macroatom packets traverse the dense forest, and that
the M2 cap bounds it (iter2 completes, 4% truncation).
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"

# --- TAU-DIAG (post-iter1 nebular recompute) ---
tau_bins = [">1", ">10", ">100", ">1e3"]
tau_counts = [268772, 148487, 72534, 30135]   # tau_max = 7.71e8

# --- cap=200 smoke (job 160855), per-iter transport ms + cap-hit ---
iters = [0, 1, 2]
trans_ms = [3323.9, 3236.3, 1137262.9]
caphit_pct = [0.0, 0.0, 4.014]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

# Panel A: iron-curtain
ax = axes[0]
bars = ax.bar(tau_bins, tau_counts, color=["#3898EC", "#4EC9B0", "#FFC107", "#D97757"])
ax.set_yscale("log")
ax.set_ylabel("number of (line x shell) optical depths")
ax.set_xlabel(r"Sobolev $\tau$ threshold")
ax.set_title("DDC15 iron-curtain after first nebular recompute\n"
             r"(mode-independent; $\tau_{max}=7.7\times10^{8}$)", fontsize=11)
for b, c in zip(bars, tau_counts):
    ax.text(b.get_x()+b.get_width()/2, c*1.15, f"{c:,}", ha="center", fontsize=9)
ax.grid(True, axis="y", alpha=0.25, which="both")

# Panel B: transport cost + cap-hit
ax = axes[1]
trans_min = np.array(trans_ms)/1000.0/60.0
bars = ax.bar([f"iter{i}" for i in iters], trans_min,
              color=["#3898EC", "#3898EC", "#D97757"])
ax.set_ylabel("transport wall time [min]")
ax.set_xlabel("iteration (transport uses previous-iter tau)")
ax.set_title("macroatom transport cost, cap=200 smoke (200k pkt, a40)\n"
             "sparse tau ~3.3s -> dense forest ~19 min (~340x)", fontsize=11)
for b, m, p in zip(bars, trans_min, caphit_pct):
    lbl = f"{m*60:.1f}s" if m < 1 else f"{m:.1f} min"
    ax.text(b.get_x()+b.get_width()/2, m+max(trans_min)*0.02, lbl,
            ha="center", fontsize=9)
    if p > 0:
        ax.text(b.get_x()+b.get_width()/2, m*0.5,
                f"cap-hit\n{p:.2f}%", ha="center", color="white",
                fontsize=10, fontweight="bold")
ax.grid(True, axis="y", alpha=0.25)
ax.annotate("iter0/1: 0% cap-hit\n(as-loaded sparse tau)",
            xy=(0.5, trans_min[0]), xytext=(0.3, max(trans_min)*0.45),
            fontsize=9, ha="left",
            arrowprops=dict(arrowstyle="->", color="#3898EC"))

fig.suptitle("Why macroatom hangs at 'Iteration 3' and how the M2 interaction cap fixes it",
             fontsize=13, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])
out = f"{ROOT}/figures/2026-05-31_macroatom_iter3_hang_cap_diagnostic.png"
plt.savefig(out, dpi=130)
print(f"saved: {out}")
