#!/usr/bin/env python3
"""[MA-FATE] λ_abs → λ_emit transfer matrix heatmap (DDC15 0.976d macroatom 162998).
Row-normalized fractions; annotates the net redward drain (blue/optical -> NIR)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

bands = ["UVblnk\n1700-3000", "CaIIKb\n3000-3300", "UVtgt\n3300-3700",
         "fluor\n3700-4400", "green\n4400-5500", "red\n5500-7000",
         "NIR1\n7000-10000", "NIR2\n>10000"]

# raw counts from logs/...kp1_162998/stdout.log [MA-FATE] final iteration
M = np.array([
    [41138,  4542,  4019,  4835,  5114,  9377,  4895,  19135],
    [10545, 41956,  6344, 28478, 11981,  9847,  5823,  15943],
    [ 4307,  4699, 22464, 10146, 10373,  8923,  6734,  13480],
    [ 6660, 10159, 14823, 88226, 39668, 22336, 17667,  42791],
    [ 5335,  7907,  8596, 36364,183027, 45421, 38295, 103174],
    [ 3337, 14032,  8550, 21050, 54655,177465, 79665, 225841],
    [ 5431, 11992,  5699, 15241, 42017, 84652,371712, 293078],
    [13699, 30976,  3076, 19750, 56960,201937,258354,1066370],
], dtype=float)

row_tot = M.sum(1)
col_tot = M.sum(0)
F = M / row_tot[:, None]          # row-normalized: P(exit | entry)

fig, ax = plt.subplots(figsize=(9.5, 8.2))
im = ax.imshow(F, cmap="magma", vmin=0, vmax=0.65, aspect="equal")
for i in range(8):
    for j in range(8):
        v = F[i, j] * 100
        ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                color="white" if F[i, j] < 0.40 else "black", fontsize=8)

# diagonal box
for k in range(8):
    ax.add_patch(plt.Rectangle((k-0.5, k-0.5), 1, 1, fill=False,
                               edgecolor="cyan", lw=1.2))

ax.set_xticks(range(8)); ax.set_xticklabels(bands, fontsize=7.5)
ax.set_yticks(range(8)); ax.set_yticklabels(bands, fontsize=7.5)
ax.set_xlabel("EMITTED band (λ_emit)", fontsize=11)
ax.set_ylabel("ABSORBED band (λ_abs)", fontsize=11)
ax.set_title("[MA-FATE] macroatom λ_abs → λ_emit transfer (row %)\n"
             "DDC15 0.976d  macroatom 162998  (cyan = no-shift diagonal)",
             fontsize=11)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("P(emit band | absorbed band)")

# net-flux annotation
net = (col_tot - row_tot) / row_tot * 100
txt = "net flux (exit-entry)/entry:\n" + "\n".join(
    f"  {bands[i].splitlines()[0]:7s} {net[i]:+5.1f}%" for i in range(8))
ax.text(1.32, 0.5, txt, transform=ax.transAxes, fontsize=8.5,
        family="monospace", va="center",
        bbox=dict(boxstyle="round", fc="#140D44", ec="#4EC9B0", alpha=0.9),
        color="white")

fig.tight_layout()
out = "figures/2026-06-04_ddc15_ma_fate_transfer_matrix.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
print("\nnet (exit-entry)/entry per band:")
for i in range(8):
    print(f"  {bands[i].splitlines()[0]:8s} {net[i]:+6.1f}%")
# optical->NIR leakage
for name, i in [("fluor", 3), ("green", 4), ("red", 5)]:
    nir = (M[i, 6] + M[i, 7]) / row_tot[i] * 100
    print(f"  {name}-entry -> NIR(>7000): {nir:.1f}%")
