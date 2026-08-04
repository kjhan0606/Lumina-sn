#!/usr/bin/env python3
"""Step-5 fluorescence pump diagnosis (169884): UV pump -> optical Fe II S_l/B by shell.
Shows the mechanism is LIVE (5031 super-thermal) but STARVED (97.9% forest skipped)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = "logs/ddc15_pc_phase3_jnul0_radls1_linere1_ratio1.0_pi1_fz1_169884"
rows = np.genfromtxt(f"{d}/lumina_sl_vs_B.csv", delimiter=",", names=True)
lam, sh, slb, Te, tau = rows["lambda_A"], rows["shell"], rows["Sl_over_B"], rows["Te"], rows["tau"]

def by_shell(lo, hi):
    m = (lam >= lo) & (lam <= hi)
    s = sh[m].astype(int); v = slb[m]; t = Te[m]
    order = np.argsort(s)
    return s[order], v[order], t[order]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

# Panel 1: S_l/B by shell for pumped (5031) vs carrier (5170)
s1, v1, T1 = by_shell(5030, 5032)
s2, v2, T2 = by_shell(5169, 5172)
ax1.axhline(1.0, color="gray", ls=":", lw=1, label="thermal S_l=B")
ax1.plot(s1, v1, "o-", color="#D97757", lw=2, ms=6, label="Fe II 5031 (PUMPED)")
ax1.plot(s2, v2, "s-", color="#3898EC", lw=2, ms=5, label="Fe II 5170 (main carrier)")
ax1.set_xlabel("shell (inner→outer)"); ax1.set_ylabel("S_l / B(T_e)")
ax1.set_title("Fluorescence mechanism LIVE but selective\n(UV pump lifts 5031 super-thermal in thin shells)")
ax1.legend(loc="upper left", fontsize=9); ax1.grid(alpha=0.3); ax1.set_xlim(-0.5, 24)
ax1.annotate("super-thermal\n(S_l/B=1.34)", xy=(12, 1.343), xytext=(14, 1.45),
             fontsize=8, color="#D97757", arrowprops=dict(arrowstyle="->", color="#D97757"))

# Panel 2: super-thermal line COUNT by shell (the starvation)
mopt = (lam >= 4000) & (lam <= 7000) & (slb > 1.05)
shc = sh[mopt].astype(int)
cnt = np.bincount(shc, minlength=25)[:25]
ax2.bar(range(25), cnt, color="#4EC9B0")
ax2.set_xlabel("shell"); ax2.set_ylabel("# optical lines with S_l/B > 1.05")
ax2.set_title("STARVED: only ~6 fluorescent lines total\n(TAUMIN=0.1 skipped 586763/599100 = 97.9% UV forest)")
ax2.grid(alpha=0.3, axis="y")
ax2.text(0.5, 0.95, "P3a CONFIRMED\nensemble pump starved\n→ lower TAUMIN to restore forest",
         transform=ax2.transAxes, fontsize=9, va="top",
         bbox=dict(boxstyle="round", fc="#FFF3CD", ec="#FFC107"))

fig.suptitle("Step-5 (169884): Deterministic fluorescence pump — LIVE chain, starved by TAUMIN",
             fontsize=12, weight="bold")
fig.tight_layout()
out = "figures/2026-06-25_step5_fluor_pump.png"
fig.savefig(out, dpi=110, bbox_inches="tight")
print("wrote", out)
print(f"5031: max S_l/B = {v1.max():.3f} at shell {s1[np.argmax(v1)]}")
print(f"5170 carrier: max S_l/B = {v2.max():.3f} (stays thermal/sub-thermal)")
print(f"total super-thermal optical lines (>1.05) = {int(cnt.sum())}")
