#!/usr/bin/env python3
"""Patch macro_atom_data.csv's transition_probability column from the flat 1/N
placeholder to A_ul-weighted radiative rates at reference photospheric
conditions, then regenerate transition_probabilities.npy via finalize.

Static weights (finalize normalizes each source-level block to Σp=1):
  ttype -1 (emission)      ∝ A_ul
  ttype  0 (internal-down) ∝ A_ul
  ttype +1 (internal-up)   ∝ B_lu · W_ref · B_ν(T_rad_ref)

This is the static fallback consumed when LUMINA_DYNAMIC_TRANSPROB is off; the
old flat 1/N over-weighted internal transitions (red-pileup placeholder, #257).
The per-shell dynamic recompute (lumina_plasma.compute_transition_probabilities)
overrides this when enabled.

Usage:
  patch_transprob_aul_weighted.py SRC_REF DST_REF [n_shells]
DST_REF is created with symlinks to every SRC_REF file EXCEPT
macro_atom_data.csv (patched) and transition_probabilities.npy (regenerated).
"""
import sys, subprocess, shutil
from pathlib import Path
import numpy as np
import pandas as pd

C_CGS = 2.99792458e10
H_CGS = 6.62607015e-27
K_CGS = 1.380649e-16
T_RAD_REF = 10000.0   # K, ≈ converged DDC15 T_inner
W_REF     = 0.5

def main(src: Path, dst: Path, n_shells: int):
    dst.mkdir(parents=True, exist_ok=True)
    # finalize REWRITES these (line_list.csv re-sorted + 3 npy outputs); never
    # symlink them or np.save/to_csv would clobber the keeper through the link.
    regen = {"macro_atom_data.csv", "transition_probabilities.npy",
             "line_list.csv", "line2macro_level_upper.npy", "tau_sobolev.npy",
             "kshape_contract.txt"}
    for f in src.iterdir():
        if f.name in regen:
            continue
        link = dst / f.name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(f.resolve())
    # line_list.csv must be a REAL file in dst (finalize reads then rewrites it)
    shutil.copy2(src / "line_list.csv", dst / "line_list.csv")

    # line_id -> A_ul, B_lu, nu
    ll = pd.read_csv(src / "line_list.csv", usecols=["line_id", "nu", "B_lu", "A_ul"])
    n_lines = int(ll["line_id"].max()) + 1
    A_ul = np.zeros(n_lines); B_lu = np.zeros(n_lines); nu = np.zeros(n_lines)
    lid = ll["line_id"].to_numpy(np.int64)
    A_ul[lid] = ll["A_ul"].to_numpy()
    B_lu[lid] = ll["B_lu"].to_numpy()
    nu[lid]   = ll["nu"].to_numpy()
    x = H_CGS * nu / (K_CGS * T_RAD_REF)
    B_nu = (2 * H_CGS * nu ** 3 / C_CGS ** 2) / np.expm1(np.clip(x, 1e-30, 700.0))
    w_iup_line = B_lu * W_REF * B_nu

    ma = pd.read_csv(src / "macro_atom_data.csv")
    tt  = ma["transition_type"].to_numpy(np.int64)
    tlid = ma["transition_line_id"].to_numpy(np.int64)
    w = np.where(tt == 1, w_iup_line[tlid], A_ul[tlid]).astype(np.float64)
    ma["transition_probability"] = w
    ma.to_csv(dst / "macro_atom_data.csv", index=False)
    print(f"[patch] wrote macro_atom_data.csv  ({len(ma):,} rows)  "
          f"emit/idn=A_ul, iup=B_lu·W·B_ν  (T_rad_ref={T_RAD_REF:.0f}K W_ref={W_REF})")
    print(f"[patch]   weight stats: iup nonzero={int((w_iup_line[tlid][tt==1]>0).sum()):,}, "
          f"global min={w[w>0].min():.3e} max={w.max():.3e}")

    # regenerate transition_probabilities.npy (+ tau_sobolev / line2macro) via finalize
    rc = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "finalize_cmfgen_ref_npy.py"),
         str(dst), str(n_shells)], check=True)
    print(f"[patch] done -> {dst}")

if __name__ == "__main__":
    src = Path(sys.argv[1]).resolve()
    dst = Path(sys.argv[2]).resolve()
    n_sh = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    main(src, dst, n_sh)
