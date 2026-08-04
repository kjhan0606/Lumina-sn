#!/usr/bin/env python3
"""J/B(lambda, shell) map (169972): locate the fluorescence pump wavelength.
VERDICT: super-thermal pump is confined to FAR-UV 2000-2500A (NOT blue 3000-4500);
optical field COLLAPSES to ~0 at green shells (fine-producer wide-window transport flag)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = "logs/ddc15_pc_phase3_jnul0_radls1_linere1_ratio1.0_pi1_fz1_169972"
m = np.genfromtxt(f"{d}/lumina_fine_jmap.csv", delimiter=",", names=True)

fig, ax = plt.subplots(figsize=(12, 6.2))
shells = [(0, "#FFC107", "shell 0 (inner, Te 4434K)"),
          (6, "#D97757", "shell 6 (Te 3408K)"),
          (13, "#3898EC", "shell 13 (GREEN-forming, Te 2920K)"),
          (15, "#4EC9B0", "shell 15 (Te 3037K)")]
for sh, c, lab in shells:
    sel = m["shell"].astype(int) == sh
    lam = m["lambda_A"][sel]; jb = m["J_over_B"][sel]
    o = np.argsort(lam)
    ax.semilogy(lam[o], np.clip(jb[o], 1e-5, None), "-", color=c, lw=1.8, label=lab)

ax.axhline(1.0, color="gray", ls=":", lw=1.5, label="thermal J=B")
ax.axvspan(2000, 2500, color="#D97757", alpha=0.12)
ax.axvspan(3000, 4500, color="#3898EC", alpha=0.10)
ax.text(2250, 1e2, "FAR-UV\npump EXISTS\nJ/B 2-160\n(green shells)", ha="center", fontsize=8.5,
        color="#a64", weight="bold")
ax.text(3750, 3e-3, "blue 3000-4500\nNOT super-thermal\n(branch-1 REFUTED)", ha="center",
        fontsize=8.5, color="#36c")
ax.annotate("optical field COLLAPSES to ~0\nat green shells (sh13: J/B~1e-4)\nvs inner sh0 ~0.5\n→ fine-producer wide-window\ntransport flag for orthodox build",
            xy=(5000, 1.5e-4), xytext=(4300, 1.2e-2), fontsize=8, color="#3898EC",
            arrowprops=dict(arrowstyle="->", color="#3898EC"))

ax.set_xlabel("wavelength (A)"); ax.set_ylabel("J / B(T_e)  (log)")
ax.set_title("J̄/B(λ,shell) map (169972): the fluorescence pump field lives in FAR-UV 2000-2500Å, not blue\n"
             "the cheap line-J̄ window pump can't harvest it (strong lines thermalize cores) → orthodox full-coupled needed")
ax.legend(loc="center right", fontsize=8.5); ax.grid(alpha=0.3, which="both")
ax.set_xlim(2000, 5500); ax.set_ylim(1e-5, 3e2)
fig.tight_layout()
out = "figures/2026-06-26_jbmap_pump_location.png"
fig.savefig(out, dpi=115, bbox_inches="tight")
print("wrote", out)
for sh in [0, 13]:
    sel = m["shell"].astype(int) == sh
    uv = m["J_over_B"][sel & (m["lambda_A"]<2500)].max()
    opt = m["J_over_B"][sel & (m["lambda_A"]>=4000)].mean()
    print(f"shell {sh}: far-UV max J/B={uv:.1f}  optical mean J/B={opt:.4f}")
