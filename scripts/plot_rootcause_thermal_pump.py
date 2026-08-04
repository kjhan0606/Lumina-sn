#!/usr/bin/env python3
"""ROOT CAUSE: the UV pump field is THERMAL in the green-emitting shells.
producer (lumina_cmfgen.c:1736) deposits forest as thermal emitters -> J_bar -> B(T_e)
-> no super-thermal UV -> no fluorescence, regardless of line count/floor/consumer."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = "logs/ddc15_pc_phase3_jnul0_radls1_linere1_ratio1.0_pi1_fz1_169914"
f = np.genfromtxt(f"{d}/lumina_sl_vs_B.csv", delimiter=",", names=True)
# UV pump lines 2300-2600A: Jline/B by shell
m = (f["lambda_A"] >= 2300) & (f["lambda_A"] <= 2600)
sh = f["shell"][m].astype(int); jb = f["Jline"][m] / f["B_Te"][m]
# mean per shell
shs = np.arange(0, 30)
mean_jb = np.array([jb[sh == s].mean() if (sh == s).any() else np.nan for s in shs])

fig, ax = plt.subplots(figsize=(11.5, 6))
ax.axhline(1.0, color="gray", ls=":", lw=1.5, label="thermal J=B(T_e)  (pump DEAD here)")
ax.plot(shs, mean_jb, "o-", color="#D97757", lw=2, ms=7, label="UV pump field  mean J̄/B(T_e)  (2300-2600Å)")
ax.axvspan(11, 17, color="#4EC9B0", alpha=0.15)
ax.text(14, 1.05, "GREEN-EMITTING\nshells\nJ̄/B = 1.001\n(THERMAL pump\n= no fluorescence)",
        ha="center", va="bottom", fontsize=9, color="#2a8", weight="bold")
ax.annotate("only cold-thin outer shells\nsee diluted hot field\n(but emit no green)",
            xy=(24, mean_jb[24]), xytext=(20, 1.18), fontsize=8.5, color="#D97757",
            arrowprops=dict(arrowstyle="->", color="#D97757"))
ax.set_xlabel("shell (inner → outer)"); ax.set_ylabel("UV pump field  J̄ / B(T_e)")
ax.set_title("ROOT CAUSE: the UV pump field is THERMAL where green forms\n"
             "producer (lumina_cmfgen.c:1736) emits the forest as thermal S_l=B(T_e) → no super-thermal UV to fluoresce\n"
             "(this is why 3.4× more pump lines = byte-identical: adding thermal emitters, not pumpers)",
             fontsize=11)
ax.legend(loc="upper left", fontsize=10); ax.grid(alpha=0.3)
ax.set_xlim(-0.5, 28); ax.set_ylim(0.9, 1.25)
fig.tight_layout()
out = "figures/2026-06-26_rootcause_thermal_pump.png"
fig.savefig(out, dpi=115, bbox_inches="tight")
print("wrote", out)
print(f"green-emitting shells 11-17 mean J̄/B = {np.nanmean(mean_jb[11:18]):.4f}")
print(f"cold outer shell 24 mean J̄/B = {mean_jb[24]:.4f}")
