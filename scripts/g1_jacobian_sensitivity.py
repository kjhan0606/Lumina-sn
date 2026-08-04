#!/usr/bin/env python3
"""(G1) Jacobian sensitivity map of emulator at MAP/Mode-B point.

Compute J[i, k] = ∂f_i / ∂θ_k where f_i is the emulator-predicted spectrum
at wavelength bin i (asinh space) and θ_k is the k-th physical parameter.

Outputs:
  - figures/g1_jacobian_heatmap.png         (67 × 1101 heatmap)
  - data/g1_line_param_attribution.csv       (24 lines × 67 params, signed J̄)
  - data/g1_line_param_attribution_abs.csv   (24 lines × 67 params, |J̄|)
  - prints top-5 sensitive params per diagnostic line
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ML_ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-ML")
SN_ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
sys.path.insert(0, str(ML_ROOT))

from lumina_ml.emulator import Emulator
from lumina_ml import config as cfg

MODELS_DIR = ML_ROOT / "models_fullphys_v2_uvfilt"
PROCESSED_DIR = ML_ROOT / "data" / "processed_fullphys_v2_uvfilt"
MAP_NPY = ML_ROOT / "results_uv_methods_compare" / "mode_b_params_67.npy"

OUT_FIG = SN_ROOT / "figures" / "g1_jacobian_heatmap.png"
OUT_FIG_LINES = SN_ROOT / "figures" / "g1_jacobian_lines_only.png"
OUT_CSV = SN_ROOT / "data" / "g1_line_param_attribution.csv"
OUT_CSV_ABS = SN_ROOT / "data" / "g1_line_param_attribution_abs.csv"
OUT_CSV_LINES = SN_ROOT / "data" / "g1_line_param_attribution_residual.csv"

# 24 standard SN Ia diagnostic lines (rest λ in Å)
DIAG = [
    ("Fe II 2382", 2382.0), ("Fe II 2600", 2600.2),
    ("Mg II 2796", 2795.5), ("Mg II 2803", 2802.7),
    ("Mn II 2576", 2576.0), ("Mn II 2594", 2594.0),
    ("Co/Fe III 3070", 3070.0),
    ("Ca II K 3934", 3933.7), ("Ca II H 3968", 3968.5),
    ("Fe III 4404", 4404.0), ("Mg II 4481", 4481.3), ("Si III 4553", 4552.6),
    ("Fe II 4924", 4923.9), ("Fe II 5018", 5018.4),
    ("Fe III 5129", 5129.2), ("Fe II 5169", 5169.0),
    ("S II W 5454", 5454.0), ("S II W 5640", 5640.0),
    ("Si II 5972", 5971.8), ("Si II 6355", 6355.0),
    ("O I 7774", 7773.4),
    ("Ca II 8498", 8498.0), ("Ca II 8542", 8542.0), ("Ca II 8662", 8662.0),
]
LINE_HALFWIDTH_AA = 50.0  # collapse window around each line


def main():
    print("Loading emulator...")
    emu = Emulator.load(MODELS_DIR, PROCESSED_DIR, device="cpu")
    grid = cfg.SPECTRUM_GRID  # 2000-10000 Å, 5 Å step → 1601 bins
    n_bins = len(grid)
    param_names = cfg.STAGE2_PARAM_NAMES   # 67 names
    param_ranges = np.array(cfg.STAGE2_PARAM_RANGES, dtype=float)  # (67,2)
    n_params = len(param_names)
    print(f"  emulator OK; n_params={n_params}  n_bins={n_bins}")

    print(f"Loading MAP point: {MAP_NPY}")
    theta0 = np.load(MAP_NPY).astype(float)
    assert theta0.shape == (n_params,), f"expected ({n_params},), got {theta0.shape}"
    f0 = emu.predict_spectrum(theta0)  # (n_bins,) asinh-space
    print(f"  spectrum at MAP: peak={f0.max():.3f}  median={np.median(f0):.3f}")

    print("Computing Jacobian via central finite differences (eps = 1% range)...")
    eps_frac = 0.01
    eps_vec = eps_frac * (param_ranges[:, 1] - param_ranges[:, 0])
    J = np.zeros((n_params, n_bins), dtype=float)

    # Vectorize: build batch of perturbations
    batch_plus = np.tile(theta0, (n_params, 1))
    batch_minus = np.tile(theta0, (n_params, 1))
    for k in range(n_params):
        batch_plus[k, k] += eps_vec[k]
        batch_minus[k, k] -= eps_vec[k]
        # clip to range to avoid extrapolation
        lo, hi = param_ranges[k]
        batch_plus[k, k] = min(batch_plus[k, k], hi)
        batch_minus[k, k] = max(batch_minus[k, k], lo)

    f_plus = emu.predict_spectrum(batch_plus)    # (n_params, n_bins)
    f_minus = emu.predict_spectrum(batch_minus)
    h = (batch_plus[np.arange(n_params), np.arange(n_params)]
         - batch_minus[np.arange(n_params), np.arange(n_params)])
    h = h.reshape(-1, 1)
    J = (f_plus - f_minus) / h
    # Normalize to dimensionless: ∂f / ∂(θ_norm) where θ_norm ∈ [0,1]
    J_norm = J * (param_ranges[:, 1] - param_ranges[:, 0]).reshape(-1, 1)
    print(f"  J: shape={J.shape}  |J_norm|.max={np.abs(J_norm).max():.4f}")

    # Heatmap
    print(f"Writing heatmap → {OUT_FIG}")
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(15, 14))
    vmax = np.percentile(np.abs(J_norm), 99)
    im = ax.imshow(J_norm, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax,
                   extent=[grid[0], grid[-1], n_params - 0.5, -0.5])
    ax.set_yticks(np.arange(n_params))
    ax.set_yticklabels(param_names, fontsize=7)
    ax.set_xlabel("Wavelength (Å)")
    ax.set_title("(G1) Emulator Jacobian at Mode-B point  ·  ∂F̂(λ) / ∂θ_norm   (asinh flux units)")
    cb = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.01)
    cb.set_label("∂F̂ / ∂θ_norm   (positive = θ↑ raises flux)")
    # Mark diagnostic lines
    for label, lam_rest in DIAG:
        ax.axvline(lam_rest, color="black", lw=0.3, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_FIG, dpi=120)
    plt.close(fig)
    print(f"  saved {OUT_FIG.stat().st_size//1024} KB")

    # Per-line attribution: collapse |J_norm| in ±LINE_HALFWIDTH around each line
    print(f"\nBuilding per-line attribution (±{LINE_HALFWIDTH_AA:.0f} Å window)...")
    rows_signed, rows_abs = {}, {}
    for label, lam_rest in DIAG:
        m = (grid >= lam_rest - LINE_HALFWIDTH_AA) & (grid <= lam_rest + LINE_HALFWIDTH_AA)
        if m.sum() == 0:
            continue
        # signed: mean over window (preserves direction; absorption depth grows when J<0)
        signed = J_norm[:, m].mean(axis=1)
        # abs: RMS magnitude over window
        abs_ = np.sqrt(np.mean(J_norm[:, m] ** 2, axis=1))
        rows_signed[label] = signed
        rows_abs[label] = abs_

    df_signed = pd.DataFrame(rows_signed, index=param_names).T
    df_abs = pd.DataFrame(rows_abs, index=param_names).T
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_signed.to_csv(OUT_CSV)
    df_abs.to_csv(OUT_CSV_ABS)
    print(f"  wrote {OUT_CSV}")
    print(f"  wrote {OUT_CSV_ABS}")

    # ===== Continuum-normalized Jacobian (isolates LINE-SHAPE sensitivity) =====
    print("\nComputing continuum-normalized Jacobian (line-only sensitivity)...")
    def remove_cont(spec, win=201, poly=3):
        return spec - savgol_filter(spec, win, poly)
    f0_res = remove_cont(f0)
    f_plus_res = np.array([remove_cont(s) for s in f_plus])
    f_minus_res = np.array([remove_cont(s) for s in f_minus])
    J_res = (f_plus_res - f_minus_res) / h
    J_res_norm = J_res * (param_ranges[:, 1] - param_ranges[:, 0]).reshape(-1, 1)
    print(f"  J_res: |J_res_norm|.max={np.abs(J_res_norm).max():.4f}")

    # Heatmap (residual)
    print(f"Writing residual heatmap → {OUT_FIG_LINES}")
    fig, ax = plt.subplots(figsize=(15, 14))
    vmax = np.percentile(np.abs(J_res_norm), 99)
    im = ax.imshow(J_res_norm, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax,
                   extent=[grid[0], grid[-1], n_params - 0.5, -0.5])
    ax.set_yticks(np.arange(n_params))
    ax.set_yticklabels(param_names, fontsize=7)
    ax.set_xlabel("Wavelength (Å)")
    ax.set_title("(G1) Continuum-removed Jacobian at Mode-B  ·  ∂(F̂ − F̂_cont) / ∂θ_norm  (line-shape sensitivity)")
    cb = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.01)
    cb.set_label("∂F̂_residual / ∂θ_norm")
    for label, lam_rest in DIAG:
        ax.axvline(lam_rest, color="black", lw=0.3, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_FIG_LINES, dpi=120)
    plt.close(fig)

    # Per-line attribution on residual
    rows_res = {}
    for label, lam_rest in DIAG:
        m = (grid >= lam_rest - LINE_HALFWIDTH_AA) & (grid <= lam_rest + LINE_HALFWIDTH_AA)
        if m.sum() == 0:
            continue
        rows_res[label] = np.sqrt(np.mean(J_res_norm[:, m] ** 2, axis=1))
    df_res = pd.DataFrame(rows_res, index=param_names).T
    df_res.to_csv(OUT_CSV_LINES)
    print(f"  wrote {OUT_CSV_LINES}")

    # Print top-5 sensitive params per line (by |J̄|)
    print("\n=== Top-5 sensitive params per diagnostic line (|J̄|, signed sign in parens) ===")
    print(f"{'Line':<16} | {'Top-5 params (|J̄| signed)'}")
    print("-" * 96)
    for label, lam_rest in DIAG:
        if label not in df_abs.index:
            continue
        a = df_abs.loc[label].values
        s = df_signed.loc[label].values
        order = np.argsort(-a)[:5]
        cells = []
        for k in order:
            sgn = "+" if s[k] >= 0 else "−"
            cells.append(f"{param_names[k]} ({sgn}{a[k]:.3f})")
        print(f"{label:<16} | " + ", ".join(cells))

    # Same table but using continuum-removed Jacobian (line-shape only)
    print("\n=== Top-5 LINE-SHAPE sensitive params (continuum-removed) ===")
    print(f"{'Line':<16} | {'Top-5 params (|J_res|)'}")
    print("-" * 96)
    for label, lam_rest in DIAG:
        if label not in df_res.index:
            continue
        a = df_res.loc[label].values
        order = np.argsort(-a)[:5]
        cells = [f"{param_names[k]} ({a[k]:.3f})" for k in order]
        print(f"{label:<16} | " + ", ".join(cells))


if __name__ == "__main__":
    main()
