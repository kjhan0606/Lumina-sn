#!/usr/bin/env python3
"""(G2) Inverse linear regression: solve  J^T · Δθ ≈ (f_HST − f_MAP)

Given the emulator Jacobian J at Mode-B and the HST−emulator residual,
compute a parameter update Δθ that minimizes the line-shape residual,
with Tikhonov regularization (ridge) and truncated SVD as alternatives.

Outputs:
  figures/g2_inverse_regression.png    — 3-panel: Δθ bars, before/after spectrum, SVD
  data/g2_dtheta_recommendation.csv    — sorted Δθ table with bounds-clipping
  prints top-10 ΔΘ recommendations + recovered RMS reduction
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
from lumina_ml import preprocessing as pp

MODELS_DIR = ML_ROOT / "models_fullphys_v2_uvfilt"
PROCESSED_DIR = ML_ROOT / "data" / "processed_fullphys_v2_uvfilt"
MAP_NPY = ML_ROOT / "results_uv_methods_compare" / "mode_b_params_67.npy"
HST_CSV = SN_ROOT / "data/sn2011fe/hst_uv/sn2011fe_hst_bmax_stitched.csv"

OUT_FIG = SN_ROOT / "figures" / "g2_inverse_regression.png"
OUT_CSV = SN_ROOT / "data" / "g2_dtheta_recommendation.csv"

# Fit window: cap UV below 2000 Å (HST below this is noisier and emulator weaker)
FIT_LAM_MIN, FIT_LAM_MAX = 2000.0, 9000.0

# 24 diagnostic-line windows: collapse fit onto ±50 Å around each line (line-only mode)
DIAG_LINES = [2382.0, 2600.2, 2795.5, 2802.7, 2576.0, 2594.0, 3070.0,
              3933.7, 3968.5, 4404.0, 4481.3, 4552.6, 4923.9, 5018.4,
              5129.2, 5169.0, 5454.0, 5640.0, 5971.8, 6355.0, 7773.4,
              8498.0, 8542.0, 8662.0]
DIAG_HALFWIDTH = 50.0


def load_hst_on_grid(grid):
    df = pd.read_csv(HST_CSV)
    lam = df.iloc[:, 0].values
    flu = df.iloc[:, 1].values
    m = np.isfinite(flu) & (flu > 0) & (lam >= 1700) & (lam <= 10500)
    lam, flu = lam[m], flu[m]
    # Interpolate (HST is already per-Å; do NOT apply /1e8)
    grid_flu = np.interp(grid, lam, flu, left=0.0, right=0.0)
    smoothed = pp.adaptive_smooth(grid_flu, grid=grid)
    normed, peak = pp.peak_normalize(smoothed, grid=grid)
    asinh_obs = pp.asinh_transform(normed)
    return asinh_obs, peak


def main():
    print("Loading emulator + MAP...")
    emu = Emulator.load(MODELS_DIR, PROCESSED_DIR, device="cpu")
    grid = cfg.SPECTRUM_GRID
    n_bins = len(grid)
    param_names = cfg.STAGE2_PARAM_NAMES
    param_ranges = np.array(cfg.STAGE2_PARAM_RANGES, dtype=float)
    n_params = len(param_names)

    theta0_raw = np.load(MAP_NPY).astype(float)
    # Clip theta0 to prior bounds (Mode-B was found before prior tightening)
    theta0 = np.clip(theta0_raw, param_ranges[:, 0], param_ranges[:, 1])
    n_oob = int(np.sum(theta0 != theta0_raw))
    if n_oob > 0:
        oob = [(param_names[k], theta0_raw[k], theta0[k])
               for k in range(n_params) if theta0[k] != theta0_raw[k]]
        print(f"  WARN: {n_oob} MAP params clipped to bounds: "
              f"{', '.join(f'{n}({a:.3g}→{b:.3g})' for n,a,b in oob[:5])}")
    f_map = emu.predict_spectrum(theta0)
    print(f"  emu: ({n_params}D → {n_bins} bins). f_map peak={f_map.max():.3f}")

    print("Loading HST → asinh space...")
    f_hst, peak_hst = load_hst_on_grid(grid)
    print(f"  f_hst peak={f_hst.max():.3f}  (peak-norm anchor in [4000,7000])")

    # Residual
    fit_mask = (grid >= FIT_LAM_MIN) & (grid <= FIT_LAM_MAX)
    delta_f = f_hst - f_map
    print(f"  residual RMS (asinh, {FIT_LAM_MIN:.0f}-{FIT_LAM_MAX:.0f} Å): "
          f"{np.sqrt(np.mean(delta_f[fit_mask]**2)):.4f}")

    # Compute Jacobian (NORMALIZED params: θ̂_k = θ_k / range_k, so Δθ̂ ∈ [-1,1] units)
    print("Computing Jacobian via finite difference (eps=1% range)...")
    eps_frac = 0.01
    eps_vec = eps_frac * (param_ranges[:, 1] - param_ranges[:, 0])
    plus = np.tile(theta0, (n_params, 1))
    minus = np.tile(theta0, (n_params, 1))
    for k in range(n_params):
        lo, hi = param_ranges[k]
        plus[k, k] = min(theta0[k] + eps_vec[k], hi)
        minus[k, k] = max(theta0[k] - eps_vec[k], lo)
    f_plus = emu.predict_spectrum(plus)
    f_minus = emu.predict_spectrum(minus)
    h = (plus[np.arange(n_params), np.arange(n_params)]
         - minus[np.arange(n_params), np.arange(n_params)]).reshape(-1, 1)
    J_phys = (f_plus - f_minus) / h                            # (n_params, n_bins)
    width = (param_ranges[:, 1] - param_ranges[:, 0]).reshape(-1, 1)
    J_norm = J_phys * width                                    # ∂f / ∂θ̂   (∈ [-1,1] norm)
    print(f"  J_norm: shape={J_norm.shape}  |J|.max={np.abs(J_norm).max():.3f}")

    # Build line-only mask: union of ±DIAG_HALFWIDTH around each diagnostic line
    line_mask = np.zeros_like(grid, dtype=bool)
    for lr in DIAG_LINES:
        line_mask |= (grid >= lr - DIAG_HALFWIDTH) & (grid <= lr + DIAG_HALFWIDTH)
    line_mask &= fit_mask
    print(f"  diagnostic-line mask: {line_mask.sum()} bins of {fit_mask.sum()} in fit window")

    # Continuum-removed (line-shape only) target — subtract local pseudo-continuum
    f_hst_res = f_hst - savgol_filter(f_hst, 201, 3)
    f_map_res = f_map - savgol_filter(f_map, 201, 3)
    delta_f_res = f_hst_res - f_map_res
    J_phys_res = J_phys - savgol_filter(J_phys, 201, 3, axis=1)
    J_norm_res = J_phys_res * width

    # Design matrix in fit window: A · Δθ̂ = b   where A = J_norm^T (rows=bins, cols=params)
    A = J_norm[:, fit_mask].T                                  # (M, P)
    b = delta_f[fit_mask]                                      # (M,)
    M, P = A.shape
    print(f"  fit dims: A {A.shape}  b {b.shape}")
    print(f"  baseline RMS in line-only mask    : {np.sqrt(np.mean(delta_f[line_mask]**2)):.4f}")
    print(f"  baseline RMS in continuum-removed : {np.sqrt(np.mean(delta_f_res[fit_mask]**2)):.4f}")

    # SVD
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    print(f"  SVD (full): σ_max={s[0]:.3f}  σ_min={s[-1]:.3e}  cond={s[0]/s[-1]:.2e}")

    # 1) Tikhonov on full-spectrum residual
    lam_ridge = 0.10 * s[0]
    Δθ̂_ridge = np.linalg.solve(A.T @ A + (lam_ridge ** 2) * np.eye(P), A.T @ b)

    # 2) Truncated SVD on full-spectrum residual
    K = int(np.sum(s > 0.05 * s[0]))
    s_inv = np.zeros_like(s); s_inv[:K] = 1.0 / s[:K]
    Δθ̂_tsvd = Vt.T @ (s_inv * (U.T @ b))
    print(f"  truncated SVD: K={K} of {len(s)} components retained")

    # 3) LINE-ONLY ridge solve (continuum-removed J on diagnostic-line bins)
    A_line = J_norm_res[:, line_mask].T
    b_line = delta_f_res[line_mask]
    U_l, s_l, Vt_l = np.linalg.svd(A_line, full_matrices=False)
    lam_l = 0.10 * s_l[0]
    Δθ̂_lineridge = np.linalg.solve(A_line.T @ A_line + (lam_l**2) * np.eye(P),
                                     A_line.T @ b_line)
    K_l = int(np.sum(s_l > 0.05 * s_l[0]))
    s_l_inv = np.zeros_like(s_l); s_l_inv[:K_l] = 1.0 / s_l[:K_l]
    Δθ̂_linetsvd = Vt_l.T @ (s_l_inv * (U_l.T @ b_line))
    print(f"  line-only SVD: σ_max={s_l[0]:.3f}  K={K_l} of {len(s_l)}")

    # Convert normalized Δθ̂ back to physical Δθ
    Δθ_ridge = Δθ̂_ridge * width.flatten()
    Δθ_tsvd = Δθ̂_tsvd * width.flatten()
    Δθ_lineridge = Δθ̂_lineridge * width.flatten()
    Δθ_linetsvd = Δθ̂_linetsvd * width.flatten()

    # Use ridge as recommendation; clip to bounds; report bounded version
    theta_new_ridge = np.clip(theta0 + Δθ_ridge, param_ranges[:, 0], param_ranges[:, 1])
    Δθ_eff_ridge = theta_new_ridge - theta0          # post-clip
    theta_new_tsvd = np.clip(theta0 + Δθ_tsvd, param_ranges[:, 0], param_ranges[:, 1])
    Δθ_eff_tsvd = theta_new_tsvd - theta0

    # Validate: recompute spectrum at new params, measure residual reduction
    rms0 = np.sqrt(np.mean(delta_f[fit_mask] ** 2))
    print(f"\n  RMS@MAP   = {rms0:.4f}")

    # Line search: scale Δθ by α to find optimal step (linearity breaks at α=1)
    print("  line-searching α for all 4 candidate updates...")
    alphas = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00])
    candidates = {"ridge": Δθ_ridge, "tsvd": Δθ_tsvd,
                  "lineridge": Δθ_lineridge, "linetsvd": Δθ_linetsvd}
    rms_grid = {tag: [] for tag in candidates}
    for a in alphas:
        for tag, dtheta in candidates.items():
            t_try = np.clip(theta0 + a * dtheta, param_ranges[:, 0], param_ranges[:, 1])
            f_try = emu.predict_spectrum(t_try)
            # Score against full residual on the line-only mask (apples-to-apples)
            rms_grid[tag].append(np.sqrt(np.mean((f_hst - f_try)[line_mask] ** 2)))
    rms0_line = np.sqrt(np.mean(delta_f[line_mask] ** 2))
    a_best, rms_best, tag_best = 0.0, rms0_line, "MAP"
    for tag in candidates:
        a = alphas[int(np.argmin(rms_grid[tag]))]
        r = min(rms_grid[tag])
        improvement = 100 * (rms0_line - r) / rms0_line
        print(f"  {tag:<10}: best α={a:.2f} → RMS_line={r:.4f}  "
              f"({improvement:+.1f}%)")
        if r < rms_best:
            rms_best, a_best, tag_best = r, a, tag
    print(f"\n  WINNER: {tag_best} @ α={a_best:.2f} → RMS={rms_best:.4f}")

    # Apply best step (the winner)
    Δθ_winner = candidates[tag_best]
    theta_new = np.clip(theta0 + a_best * Δθ_winner, param_ranges[:, 0], param_ranges[:, 1])
    f_new = emu.predict_spectrum(theta_new)
    Δθ_eff = theta_new - theta0
    Δθ̂_eff = Δθ_eff / width.flatten()

    # For backward-compat with downstream plotting / CSV
    theta_new_ridge = theta_new
    Δθ̂_ridge = Δθ̂_eff
    Δθ_eff_ridge = Δθ_eff
    f_ridge = f_new
    rms_ridge = rms_best
    a_best_ridge = a_best
    a_best_tsvd = alphas[int(np.argmin(rms_grid["tsvd"]))]
    theta_new_tsvd = np.clip(theta0 + a_best_tsvd * Δθ_tsvd,
                             param_ranges[:, 0], param_ranges[:, 1])
    f_tsvd = emu.predict_spectrum(theta_new_tsvd)
    rms_tsvd = np.sqrt(np.mean((f_hst - f_tsvd)[line_mask] ** 2))
    Δθ̂_tsvd = (theta_new_tsvd - theta0) / width.flatten()
    rms0 = rms0_line

    # Per-param normalized magnitude (use larger-impact ridge)
    Δθ̂_eff = Δθ_eff_ridge / width.flatten()
    rec = pd.DataFrame({
        "param": param_names,
        "MAP_value": theta0,
        "Δθ_normalized": Δθ̂_ridge,            # raw recommendation in [0,1] units
        "Δθ_norm_clipped": Δθ̂_eff,            # after bound clipping
        "Δθ_physical": Δθ_eff_ridge,
        "new_value": theta_new_ridge,
        "lo_bound": param_ranges[:, 0],
        "hi_bound": param_ranges[:, 1],
        "Δθ_tsvd_norm": Δθ̂_tsvd,
    }).sort_values("Δθ_norm_clipped", key=lambda x: x.abs(), ascending=False)
    rec.to_csv(OUT_CSV, index=False)
    print(f"\n  wrote {OUT_CSV}")

    # Top-15 print
    print("\n=== Top-15 Δθ recommendation (Tikhonov ridge λ=10% σ_max) ===")
    print(f"{'param':<22} {'MAP':>10} → {'new':>10}   "
          f"{'Δθ̂(norm)':>10}  {'physical Δ':>12}")
    print("-" * 80)
    for _, r in rec.head(15).iterrows():
        clipped = "*" if abs(r["Δθ_normalized"]) - abs(r["Δθ_norm_clipped"]) > 1e-4 else " "
        print(f"{r['param']:<22} {r['MAP_value']:10.4g} → {r['new_value']:10.4g}   "
              f"{r['Δθ_norm_clipped']:+10.3f}{clipped}  {r['Δθ_physical']:+12.4g}")
    print("(* = hit prior bound after clipping)")

    # ===== Plot =====
    print(f"\nWriting figure → {OUT_FIG}")
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.5, 0.9],
                          width_ratios=[3, 1], hspace=0.35, wspace=0.18)

    # (a) Δθ̂ bar chart top-15
    ax1 = fig.add_subplot(gs[0, :])
    top = rec.head(20)
    colors = ["tab:red" if v > 0 else "tab:blue" for v in top["Δθ_norm_clipped"]]
    ax1.barh(np.arange(len(top)), top["Δθ_norm_clipped"], color=colors, alpha=0.7)
    ax1.set_yticks(np.arange(len(top)))
    ax1.set_yticklabels(top["param"], fontsize=9)
    ax1.invert_yaxis()
    ax1.axvline(0, color="k", lw=0.6)
    ax1.set_xlabel("Δθ̂ (units of prior range; + = increase, − = decrease)")
    ax1.set_title(f"(G2) Top-20 parameter updates  (Tikhonov ridge, λ={lam_ridge:.3f})")
    ax1.grid(alpha=0.3, axis="x")

    # (b) Spectrum: HST vs MAP vs MAP+Δθ
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(grid, f_hst, color="black", lw=1.0, label=f"HST (asinh, peak-norm)")
    ax2.plot(grid, f_map, color="tab:orange", lw=0.8, alpha=0.8,
             label=f"emulator @ MAP (RMS={rms0:.4f})")
    ax2.plot(grid, f_ridge, color="tab:green", lw=0.8, alpha=0.9,
             label=f"emulator @ MAP+Δθ_ridge (RMS={rms_ridge:.4f})")
    ax2.plot(grid, f_tsvd, color="tab:purple", lw=0.8, alpha=0.7, ls="--",
             label=f"emulator @ MAP+Δθ_tSVD (RMS={rms_tsvd:.4f})")
    ax2.axvspan(FIT_LAM_MIN, FIT_LAM_MAX, color="yellow", alpha=0.05)
    ax2.set_xlim(grid[0], grid[-1])
    ax2.set_xlabel("Wavelength (Å)")
    ax2.set_ylabel("asinh flux (peak-norm)")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.set_title("Spectrum: HST vs emulator (before/after Δθ)")
    ax2.grid(alpha=0.3)

    # (c) Singular values + (d) residual reduction
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.semilogy(np.arange(1, len(s) + 1), s / s[0], "o-", ms=3)
    ax3.axhline(0.05, color="r", ls="--", lw=0.8, label=f"5% cutoff (K={K})")
    ax3.set_xlabel("singular value index")
    ax3.set_ylabel("σ_i / σ_max")
    ax3.set_title(f"SVD spectrum  (cond={s[0]/s[-1]:.1e})")
    ax3.legend()
    ax3.grid(alpha=0.3, which="both")

    ax4 = fig.add_subplot(gs[2, 1])
    ax4.plot(alphas, rms_grid["ridge"], "o-", color="tab:green", label="ridge")
    ax4.plot(alphas, rms_grid["tsvd"], "s-", color="tab:purple", label="tSVD")
    ax4.axhline(rms0, color="tab:orange", ls="--", lw=0.8, label=f"MAP ({rms0:.3f})")
    ax4.axvline(a_best_ridge, color="tab:green", lw=0.5, alpha=0.5)
    ax4.axvline(a_best_tsvd, color="tab:purple", lw=0.5, alpha=0.5)
    ax4.set_xlabel("step size α")
    ax4.set_ylabel("RMS")
    ax4.set_title("Line search")
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=120)
    plt.close(fig)
    print(f"  saved {OUT_FIG.stat().st_size//1024} KB")


if __name__ == "__main__":
    main()
