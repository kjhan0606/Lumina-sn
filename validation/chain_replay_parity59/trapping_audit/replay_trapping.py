#!/usr/bin/env python3
"""Parameterized copy/recomposition of Audit U and electron-scattering Audit T.

Definitions match audit_u_{cmfgen,lumina}.py and audit_t_es.py.  Expansion
opacity is replayed separately by this directory's parameterized
audit_t_expop.py from lumina_levelpop.csv; no lumina_line.csv is required.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import C_A, C_CM, FOURPI_OVER_C, cmfgen_field, integrate_j, load_field, parse_rvtj_block, write_csv

SIGMA_T = 6.6524587e-25
TARGET_V = [4264 + 728 * i for i in range(11)]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--cmfgen-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    field_path = args.input_dir / "lumina_coevolve_field.csv"
    plasma_path = args.input_dir / "lumina_plasma_state.csv"
    levelpop_path = args.input_dir / "lumina_levelpop.csv"
    for required in (field_path, plasma_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    field = load_field(field_path)
    wavelength_c, j_c, velocity_c = cmfgen_field(args.cmfgen_dir)
    nu_c = C_A / wavelength_c
    freq_order = np.argsort(nu_c)
    depth_order = np.argsort(velocity_c)
    u_c_depth = FOURPI_OVER_C * np.trapezoid(j_c[freq_order, :], nu_c[freq_order], axis=0)
    if np.any(u_c_depth <= 0):
        raise ValueError("CMFGEN energy density contains non-positive values")
    u_c_targets = 10.0 ** np.interp(TARGET_V, velocity_c[depth_order], np.log10(u_c_depth[depth_order]))

    u_rows: list[list[object]] = []
    for shell, target_v in enumerate(TARGET_V):
        data = field[shell]
        u_mc = integrate_j(data["wavelength_A"], data["mc_J"])
        u_cs = integrate_j(data["wavelength_A"], data["cs_J"])
        u_cmf = float(u_c_targets[shell])
        u_rows.append([shell, target_v, u_cmf, u_mc, u_cs, u_mc / u_cmf, np.log10(u_mc / u_cmf),
                       str(field_path), "mc_J; cs_J", str(args.cmfgen_dir / "EDDFACTOR"), "J_nu"])
    write_csv(args.output_dir / "audit_U_energy_density.csv",
              ["shell", "v_kms", "u_cmfgen_full", "u_lumina_mc", "u_lumina_cs",
               "mc_over_cmfgen", "log10_mc_over_cmfgen", "lumina_source_file",
               "lumina_fields", "cmfgen_source_file", "cmfgen_field"], u_rows)

    geometry_path = args.repo / "data/tardis_reference_toy06_19p48d/geometry.csv"
    geometry: dict[int, tuple[float, float]] = {}
    with geometry_path.open() as handle:
        for row in csv.DictReader(handle):
            geometry[int(row["shell_id"])] = (float(row["r_inner"]), float(row["r_outer"]))
    ne: dict[int, float] = {}
    with plasma_path.open() as handle:
        for row in csv.DictReader(handle):
            ne[int(row["shell_id"])] = float(row["n_e"])
    if set(geometry) != set(ne):
        raise ValueError("geometry/plasma shell sets differ")
    dtau = np.asarray([ne[s] * SIGMA_T * (geometry[s][1] - geometry[s][0]) for s in sorted(geometry)])
    tau_es = np.cumsum(dtau[::-1])[::-1]

    mean_path = args.cmfgen_dir / "MEANOPAC"
    v_mean: list[float] = []
    tau_ross: list[float] = []
    tau_c_es: list[float] = []
    with mean_path.open() as handle:
        for line in handle:
            tokens = line.split()
            if len(tokens) < 15:
                continue
            try:
                float(tokens[0]); int(tokens[1])
            except ValueError:
                continue
            tau_ross.append(float(tokens[2]))
            tau_c_es.append(float(tokens[10]))
            v_mean.append(float(tokens[14]))
    order = np.argsort(v_mean)
    v_m = np.asarray(v_mean)[order]
    tr_m = np.asarray(tau_ross)[order]
    te_m = np.asarray(tau_c_es)[order]
    tau_rows: list[list[object]] = []
    for shell, target_v in enumerate(TARGET_V):
        cr = float(np.interp(target_v, v_m, tr_m))
        ce = float(np.interp(target_v, v_m, te_m))
        le = float(tau_es[shell])
        tau_rows.append([shell, target_v, cr, ce, le, le / ce, str(mean_path), "Tau_Ross; Tau_es",
                         str(plasma_path), "n_e", str(geometry_path), "r_inner; r_outer"])
    write_csv(args.output_dir / "audit_T_electron_scattering.csv",
              ["shell", "v_kms", "cmfgen_tau_ross", "cmfgen_tau_es", "lumina_tau_es",
               "lumina_over_cmfgen_es", "cmfgen_source_file", "cmfgen_fields",
               "plasma_source_file", "plasma_field", "geometry_source_file", "geometry_fields"], tau_rows)

    unresolved = []
    tau_output = args.output_dir / "tau_lumina_line.csv"
    if not levelpop_path.is_file():
        unresolved.append(["tau_FUV and Lumina expansion/Rosseland line opacity", "UNRESOLVED",
                           str(levelpop_path), "required level-population capture is absent"])
    elif tau_output.is_file():
        unresolved.append(["tau_FUV and Lumina expansion/Rosseland line opacity", "RESOLVED",
                           str(tau_output), "persistent output from parameterized audit_t_expop.py"])
        with tau_output.open() as handle:
            tau_row = next(csv.DictReader(handle))
        lower_hit = float(tau_row["lower_level_hit_fraction"])
        upper_hit = float(tau_row["upper_level_hit_fraction"])
        if lower_hit < 1.0 or upper_hit < 1.0:
            unresolved.append(["line-list levels absent from level-population dump", "UNRESOLVED",
                               str(levelpop_path),
                               f"lower hit fraction={lower_hit}; upper hit fraction={upper_hit}; "
                               "see tau_lumina_line.csv for the historical convention"])
    else:
        unresolved.append(["tau_FUV and Lumina expansion/Rosseland line opacity", "NOT_RUN",
                           str(levelpop_path), "run this directory's parameterized audit_t_expop.py"])
    write_csv(args.output_dir / "UNRESOLVED.csv", ["quantity", "status", "required_file", "reason"], unresolved)


if __name__ == "__main__":
    main()
