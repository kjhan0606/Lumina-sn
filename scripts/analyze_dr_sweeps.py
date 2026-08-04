#!/usr/bin/env python3
"""Analyze DR boost (154211) and DR floor (154215) sweeps.

For each completed array task:
  1. Parse final iteration W err / T_rad err from stdout.log
  2. Load lumina_spectrum_formal.csv if present
  3. Compute baseline-normalized RMS vs SN 2011fe HST B-max stitched
  4. Tabulate -> CSV at logs/dr_sweep_summary.csv
  5. Render comparison plot to figures/dr_sweep_compare.png
     (one panel per sweep type; overlay HST in black)
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT   = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
LOGS   = ROOT / "logs"
HST    = ROOT / "data/sn2011fe/hst_uv/sn2011fe_hst_bmax_stitched.csv"
OUTCSV = LOGS / "dr_sweep_summary.csv"
OUTPNG = ROOT / "figures/dr_sweep_compare.png"
C_KMS  = 299792.458
FWHM_KMS = 40000.0


def gauss_smooth(lam: np.ndarray, flu: np.ndarray, fwhm_kms: float = FWHM_KMS) -> np.ndarray:
    """Velocity-space Gaussian smoothing for pseudo-continuum."""
    cont = np.zeros_like(flu)
    beta = fwhm_kms / C_KMS
    for i in range(len(lam)):
        sigma = beta * lam[i] / 2.3548
        win = 4.0 * sigma
        sel = (lam >= lam[i] - win) & (lam <= lam[i] + win)
        if sel.sum() < 2:
            cont[i] = flu[i]
            continue
        w = np.exp(-0.5 * ((lam[sel] - lam[i]) / sigma) ** 2)
        cont[i] = np.sum(w * flu[sel]) / np.sum(w)
    return cont


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.is_file() or path.stat().st_size < 100:
        return None
    df = pd.read_csv(path)
    lam = df.iloc[:, 0].to_numpy()
    col = "flux" if "flux" in df.columns else df.columns[1]
    flu = df[col].to_numpy()
    m = np.isfinite(lam) & np.isfinite(flu) & (lam > 0)
    return lam[m], flu[m]


def parse_convergence(stdout: Path) -> dict:
    """Pull last W err / T_rad err / iter count from stdout.log."""
    if not stdout.is_file():
        return {}
    text = stdout.read_text(errors="replace")
    werr_m = re.findall(r"Mean \|W error\|.*?(\d+\.\d+)%", text)
    terr_m = re.findall(r"Mean \|T_rad error\|.*?(\d+\.\d+)%", text)
    iter_m = re.findall(r"--- Iteration (\d+)/\d+ ---", text)
    return {
        "W_err_pct": float(werr_m[-1]) if werr_m else np.nan,
        "T_rad_err_pct": float(terr_m[-1]) if terr_m else np.nan,
        "n_iters_done": int(iter_m[-1]) if iter_m else 0,
    }


def baseline_norm_rms(lam: np.ndarray, flu: np.ndarray,
                      hst_lam: np.ndarray, hst_flu: np.ndarray,
                      lo: float = 1700.0, hi: float = 9000.0) -> float:
    m = (lam >= lo) & (lam <= hi)
    if m.sum() < 50:
        return np.nan
    lam, flu = lam[m], flu[m]
    flu_interp = np.interp(hst_lam, lam, flu, left=0.0, right=0.0)
    sel = (hst_lam >= lo) & (hst_lam <= hi) & (flu_interp > 0)
    if sel.sum() < 50:
        return np.nan
    a, b = flu_interp[sel], hst_flu[sel]
    a_cont = gauss_smooth(hst_lam[sel], a)
    b_cont = gauss_smooth(hst_lam[sel], b)
    a_safe = np.where(a_cont > 0, a / a_cont, 0.0)
    b_safe = np.where(b_cont > 0, b / b_cont, 0.0)
    return float(np.sqrt(np.mean((a_safe - b_safe) ** 2)))


def discover_runs() -> list[dict]:
    """Find dr_boost_b*_<jobid> and dr_floor_e*_<jobid> directories."""
    rows = []
    for kind, glob, axis_re in [
        ("boost", "dr_boost_b*_*", r"dr_boost_b(\d+)_(\d+)$"),
        ("floor", "dr_floor_e*_*", r"dr_floor_e(\d+)_(\d+)$"),
    ]:
        for d in sorted(LOGS.glob(glob)):
            if not d.is_dir():
                continue
            m = re.match(axis_re, d.name)
            if not m:
                continue
            axis_val, jobid = m.group(1), m.group(2)
            rows.append({"kind": kind, "axis": int(axis_val), "jobid": jobid, "dir": d})
    return rows


def main() -> int:
    if not HST.is_file():
        print(f"HST stitched file missing: {HST}", file=sys.stderr)
        return 1
    hlam, hflu = load_csv(HST)
    m = (hlam >= 1700) & (hlam <= 9000) & (hflu > 0)
    hlam, hflu = hlam[m], hflu[m]

    runs = discover_runs()
    if not runs:
        print("No DR sweep run directories found yet.")
        return 0

    summary = []
    spectra = []
    for r in runs:
        d = r["dir"]
        conv = parse_convergence(d / "stdout.log")
        spec = load_csv(d / "lumina_spectrum_formal.csv")
        rms = np.nan
        if spec is not None:
            llam, lflu = spec
            rms = baseline_norm_rms(llam, lflu, hlam, hflu)
            spectra.append((r, llam, lflu))
        summary.append({
            **r,
            **conv,
            "rms_baseline_norm": rms,
            "spec_present": spec is not None,
        })
        del r["dir"]  # don't write Path to CSV

    df = pd.DataFrame(summary)
    df = df.drop(columns=["dir"], errors="ignore")
    df = df.sort_values(["kind", "axis"])
    OUTCSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTCSV, index=False)
    print(df.to_string(index=False))
    print(f"\nSummary written to {OUTCSV}")

    if not spectra:
        print("No spectra to plot yet.")
        return 0

    by_kind: dict[str, list] = {"boost": [], "floor": []}
    for run, lam, flu in spectra:
        by_kind[run["kind"]].append((run["axis"], lam, flu))

    n_panels = sum(1 for k in by_kind if by_kind[k])
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 4 * n_panels), squeeze=False)
    axes = axes.ravel()
    panel_i = 0
    for kind in ("boost", "floor"):
        items = sorted(by_kind[kind])
        if not items:
            continue
        ax = axes[panel_i]
        ax.plot(hlam, hflu, color="black", lw=1.0, label="SN2011fe HST stitched", alpha=0.7)
        cmap = plt.get_cmap("viridis")
        for j, (ax_val, lam, flu) in enumerate(items):
            color = cmap(j / max(1, len(items) - 1))
            label = (f"DR×{ax_val}" if kind == "boost"
                     else f"floor=1e-{ax_val} cm³/s")
            ax.plot(lam, flu, color=color, lw=0.8, label=label, alpha=0.85)
        ax.set_xlim(1700, 9000)
        ax.set_xlabel("Wavelength (Å)")
        ax.set_ylabel("F_λ")
        ax.set_yscale("log")
        ax.set_title(f"DR {kind} sweep vs SN 2011fe (B-max)")
        ax.legend(loc="upper right", fontsize=8)
        panel_i += 1

    OUTPNG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPNG, dpi=120)
    print(f"Plot written to {OUTPNG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
