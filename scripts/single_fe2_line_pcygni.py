#!/usr/bin/env python3
"""Pseudo-continuum + ONE Fe II line (Sobolev P-Cygni) overplotted on HST.

Pseudo-continuum: savgol-smoothed champion LUMINA spectrum (152761).
Single line: Fe II 5169 (Multiplet 42, canonical SN Ia photospheric indicator).
Sobolev P-Cygni in homologous expansion, constant τ across line-forming region.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT  = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
HST   = ROOT / "data/sn2011fe/hst_uv/sn2011fe_hst_bmax_stitched.csv"
CHAMP = ROOT / "logs/PROD_L19_W2x010_W3x065wide_152761/lumina_spectrum_formal.csv"
OUT   = ROOT / "figures/single_fe2_line_vs_hst.png"

# Pick a representative Fe II line — 5169 Å (Multiplet 42, classic SN Ia velocity diagnostic)
LINE_LAM_REST = 5169.03
LINE_LABEL    = "Fe II 5169"

# Atmosphere parameters (SN 2011fe B-max, t_exp ≈ 17.8d)
V_PHOT = 11000.0   # km/s — photospheric velocity
V_MAX  = 22000.0   # km/s — outer atmosphere
TAU_LIST = [0.5, 2.0, 8.0, 30.0]
C_KMS  = 299792.458


def load(p):
    df = pd.read_csv(p)
    lam = df.iloc[:, 0].values
    flu_col = "flux" if "flux" in df.columns else df.columns[1]
    return lam, df[flu_col].values


def pcygni_homologous(lam, lam_rest, v_phot, v_max, tau0, F_c, n_p=400):
    """Sobolev P-Cygni in homologous expansion v(r) = (r/R_phot) v_phot.

    Convention: observer at z = +∞. z_res in R_phot units; positive = blueshift.
    Pure scattering: source S = W(r) × F_c (W = geometric dilution factor).
    """
    F_obs   = np.zeros_like(F_c)
    p_max   = v_max / v_phot
    p_grid  = np.linspace(1e-6, p_max, n_p)
    dp      = p_grid[1] - p_grid[0]
    w_p     = 2.0 * np.pi * p_grid * dp           # annular area weight
    A_total = np.pi                               # photospheric disc area (R_phot=1)

    for i, l in enumerate(lam):
        # z_res in R_phot units; positive for blueshift (material approaching observer)
        z_res = (lam_rest - l) / lam_rest * (C_KMS / v_phot)
        r_res = np.sqrt(p_grid**2 + z_res**2)

        in_atm  = (r_res >= 1.0) & (r_res <= p_max)
        in_core = p_grid < 1.0
        z_phot_front = np.sqrt(np.maximum(0.0, 1.0 - p_grid**2))
        absorbed   = in_atm & in_core & (z_res > -z_phot_front)
        emit_only  = in_atm & ~in_core
        clear_core = in_core & ~absorbed

        W = np.zeros_like(p_grid)
        valid = in_atm
        W[valid] = 0.5 * (1.0 - np.sqrt(np.maximum(0.0, 1.0 - 1.0 / (r_res[valid]**2))))
        S = F_c[i] * W

        I = np.zeros_like(p_grid)
        I[clear_core] = F_c[i]
        I[absorbed]   = F_c[i] * np.exp(-tau0) + S[absorbed] * (1.0 - np.exp(-tau0))
        I[emit_only]  = S[emit_only] * (1.0 - np.exp(-tau0))

        F_obs[i] = np.sum(w_p * I) / A_total

    return F_obs


def main():
    print("Loading HST...")
    hlam, hflu = load(HST)
    m = (hlam >= 1500) & (hlam <= 9500) & np.isfinite(hflu) & (hflu > 0)
    hlam, hflu = hlam[m], hflu[m]

    print(f"Loading LUMINA champion: {CHAMP.name}")
    llam, lflu = load(CHAMP)
    # Continuum-band normalize champion to HST (4500–5800 Å)
    band_h = np.trapezoid(hflu[(hlam >= 4500) & (hlam <= 5800)], hlam[(hlam >= 4500) & (hlam <= 5800)])
    band_l = np.trapezoid(lflu[(llam >= 4500) & (llam <= 5800)], llam[(llam >= 4500) & (llam <= 5800)])
    lflu *= band_h / band_l

    print("Computing pseudo-continuum (savgol win=301, poly=3)...")
    F_c = savgol_filter(lflu, 301, 3)

    print(f"Computing P-Cygni for {LINE_LABEL} (λ_rest={LINE_LAM_REST:.2f} Å)")
    print(f"  v_phot={V_PHOT:.0f} km/s, v_max={V_MAX:.0f} km/s, τ ∈ {TAU_LIST}")
    profiles = {tau: pcygni_homologous(llam, LINE_LAM_REST, V_PHOT, V_MAX, tau, F_c)
                for tau in TAU_LIST}

    # ===== Figure: 2 panels =====
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))

    # (a) full spectrum view
    ax = axes[0]
    ax.plot(hlam, hflu, "k-", lw=0.7, alpha=0.75, label="HST B-max (SN 2011fe)")
    ax.plot(llam, F_c, "C0--", lw=1.4, label="LUMINA pseudo-continuum (savgol champ 152761)")
    colors = ["#5DADE2", "#F39C12", "#E74C3C", "#8E44AD"]
    for tau, col in zip(TAU_LIST, colors):
        ax.plot(llam, profiles[tau], color=col, lw=1.0, alpha=0.85, label=f"+ {LINE_LABEL} (τ={tau})")
    ax.axvline(LINE_LAM_REST, color="gray", ls=":", lw=0.6)
    ymax = 1.35 * np.max(hflu[(hlam >= 3500) & (hlam <= 7000)])
    ax.set_xlim(2000, 8000)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Wavelength (Å)")
    ax.set_ylabel("Flux (erg/s/cm²/Å)")
    ax.set_title(f"Pseudo-continuum + ONLY {LINE_LABEL} (Sobolev P-Cygni) vs HST")
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    # (b) zoom around the line
    ax = axes[1]
    win = 600
    ax.plot(hlam, hflu, "k-", lw=0.9, alpha=0.75, label="HST")
    ax.plot(llam, F_c, "C0--", lw=1.6, label="pseudo-continuum")
    for tau, col in zip(TAU_LIST, colors):
        ax.plot(llam, profiles[tau], color=col, lw=1.4, label=f"τ={tau}")
    ax.axvline(LINE_LAM_REST, color="gray", ls=":", lw=0.8)
    for v, lab in [(V_PHOT, f"−v_phot ({V_PHOT/1000:.0f}k)"),
                   (V_MAX,  f"−v_max ({V_MAX/1000:.0f}k)")]:
        l_blue = LINE_LAM_REST * (1.0 - v / C_KMS)
        ax.axvline(l_blue, color="red", ls="--", lw=0.6, alpha=0.55)
        ax.text(l_blue, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0 else 1, lab,
                fontsize=7, color="red", rotation=90, ha="right", va="top")
    ax.set_xlim(LINE_LAM_REST - win, LINE_LAM_REST + win)
    mask = (llam >= LINE_LAM_REST - win) & (llam <= LINE_LAM_REST + win)
    ax.set_ylim(0, 1.25 * max(F_c[mask].max(), hflu[(hlam >= LINE_LAM_REST - win) & (hlam <= LINE_LAM_REST + win)].max()))
    ax.set_xlabel("Wavelength (Å)")
    ax.set_ylabel("Flux")
    ax.set_title(f"Zoom: λ ∈ [{LINE_LAM_REST - win:.0f}, {LINE_LAM_REST + win:.0f}] Å")
    ax.legend(fontsize=8, loc="lower left")

    plt.tight_layout()
    fig.savefig(OUT, dpi=130)
    plt.close(fig)
    print(f"Saved: {OUT}  ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
