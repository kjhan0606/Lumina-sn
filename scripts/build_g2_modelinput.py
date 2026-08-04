#!/usr/bin/env python3
"""Build LUMINA input directory for the (G2) inverse-regression prescription.

Reads g2_dtheta_recommendation.csv (winner: lineridge α=0.75), constructs
Stage2Params, materializes a TARDIS-format ref directory at a fixed path,
and saves the 67D theta vector for reproducibility.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ML_ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-ML")
SN_ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
sys.path.insert(0, str(ML_ROOT))

from lumina_ml.data_utils import LuminaRunner, Stage2Params
from lumina_ml import config as cfg

CSV = SN_ROOT / "data" / "g2_dtheta_recommendation.csv"
OUT_NPY = SN_ROOT / "data" / "g2_theta_new_67.npy"
OUT_REF = SN_ROOT / "data" / "tardis_reference_g2_prescription"


def main():
    df = pd.read_csv(CSV)
    # Build 67D theta_new in canonical Stage2 order
    name_to_new = dict(zip(df["param"], df["new_value"]))
    theta_new = np.array([name_to_new[n] for n in cfg.STAGE2_PARAM_NAMES], dtype=float)

    # Enforce velocity hierarchy (G2 linear solve doesn't constrain v_core/v_wall/v_break)
    idx = {n: i for i, n in enumerate(cfg.STAGE2_PARAM_NAMES)}
    v_in = theta_new[idx["v_inner"]]
    v_core_min = v_in + 1000.0
    if theta_new[idx["v_core"]] < v_core_min:
        print(f"  v_core {theta_new[idx['v_core']]:.0f} < v_inner+1000 → bumped to {v_core_min:.0f}")
        theta_new[idx["v_core"]] = v_core_min
    v_wall_min = theta_new[idx["v_core"]] + 1000.0
    if theta_new[idx["v_wall"]] < v_wall_min:
        print(f"  v_wall {theta_new[idx['v_wall']]:.0f} < v_core+1000 → bumped to {v_wall_min:.0f}")
        theta_new[idx["v_wall"]] = v_wall_min
    v_break_min = v_in + 1500.0
    v_break_max = cfg.V_OUTER - 1000.0
    if theta_new[idx["v_break"]] < v_break_min:
        print(f"  v_break {theta_new[idx['v_break']]:.0f} < v_inner+1500 → bumped to {v_break_min:.0f}")
        theta_new[idx["v_break"]] = v_break_min
    if theta_new[idx["v_break"]] > v_break_max:
        theta_new[idx["v_break"]] = v_break_max

    print(f"theta_new (67D): {theta_new[:5]}... (first 5)")
    np.save(OUT_NPY, theta_new)
    print(f"  saved {OUT_NPY}")

    params = Stage2Params.from_array(theta_new)
    print(f"\n  is_valid() = {params.is_valid()}")
    print(f"\nKey physical params:")
    print(f"  log_L            = {params.log_L:.3f}")
    print(f"  v_inner          = {params.v_inner:.0f} km/s")
    print(f"  v_core           = {params.v_core:.0f} km/s")
    print(f"  t_exp            = {params.t_exp:.2f} d")
    print(f"  density_exp      = {params.density_exp:.2f}")
    print(f"  log_rho_0        = {params.log_rho_0:.3f}")
    print(f"  T_e_ratio        = {params.T_e_ratio:.3f}")
    print(f"  X_Fe_core/wall/outer = {params.X_Fe_core:.3f} / {params.X_Fe_wall:.3f} / {params.X_Fe_outer:.3f}")
    print(f"  X_Si_wall, X_Ni  = {params.X_Si_wall:.3f}, {params.X_Ni:.3f}")

    # Use LuminaRunner.create_model_dir but redirect to permanent path
    runner = LuminaRunner(nlte=True, nlte_start_iter=5)
    print(f"\nUsing ref_dir: {runner.ref_dir}")
    print(f"Building model dir at {OUT_REF}...")
    OUT_REF.parent.mkdir(parents=True, exist_ok=True)
    if OUT_REF.exists():
        import shutil
        shutil.rmtree(OUT_REF)

    # Replicate create_model_dir logic but at fixed path
    OUT_REF.mkdir(parents=True)
    for fname in runner.ref_files:
        if fname not in cfg.REGEN_FILES:
            src = runner.ref_dir / fname
            dst = OUT_REF / fname
            dst.symlink_to(src)
    runner._write_config_json(OUT_REF, params)
    runner._write_geometry_csv(OUT_REF, params)
    runner._write_density_csv(OUT_REF, params)
    runner._write_abundances_csv(OUT_REF, params)
    runner._write_electron_densities_csv(OUT_REF, params)
    runner._write_plasma_state_csv(OUT_REF, params)

    files = sorted(p.name for p in OUT_REF.iterdir())
    print(f"  {len(files)} files: {files[:8]}...")
    print(f"\nReady for: ./lumina_cuda {OUT_REF} <n_pkt> <n_iter> spectrum nlte")


if __name__ == "__main__":
    main()
