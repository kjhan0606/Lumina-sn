#!/usr/bin/env python3
"""Parameterized replay of the historical Lumina expansion-opacity audit.

The numerical calculation reproduces
validation/cmfgen_toy06_19p48d/analysis/trapping_audit/audit_t_expop.py.  Input
and output paths, validation, provenance columns, and Lumina's defined
zero-population Sobolev branch are explicit here.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd


SOB = 2.6540281e-2
SIGMA_T = 6.6524587e-25
C = 2.99792458e10
C_A = 2.99792458e18
H = 6.62607015e-27
KB = 1.380649e-16
T_EXP = 19.48 * 86400.0
NSH = 7
T_R = 13120.0


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay the historical s0-s6 Lumina expansion-opacity audit."
    )
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--plasma-state", type=Path, required=True)
    parser.add_argument("--levelpop", type=Path, required=True)
    parser.add_argument("--line-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def key(z: np.ndarray | int, ion: np.ndarray | int, lev: np.ndarray | int):
    return (z * 100 + ion) * 100000 + lev


def d_bd_t(nu: np.ndarray, temperature: float) -> np.ndarray:
    x = H * nu / (KB * temperature)
    ex = np.exp(x)
    return (2 * H * nu**3 / C**2) * (x * ex / ((ex - 1) ** 2)) / temperature


def require_input(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def validate_persistent_output(path: Path) -> Path:
    resolved = path.resolve()
    if resolved == Path("/tmp") or Path("/tmp") in resolved.parents:
        raise ValueError(f"refusing ephemeral /tmp output: {resolved}")
    if "scratchpad" in resolved.parts:
        raise ValueError(f"refusing scratchpad output: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def main() -> None:
    args = arguments()
    geometry_path = require_input(args.geometry)
    plasma_path = require_input(args.plasma_state)
    levelpop_path = require_input(args.levelpop)
    line_list_path = require_input(args.line_list)
    output_path = validate_persistent_output(args.output)
    t0 = time.time()

    geometry: dict[int, tuple[float, float]] = {}
    with geometry_path.open() as handle:
        for row in csv.DictReader(handle):
            geometry[int(row["shell_id"])] = (
                float(row["r_inner"]),
                float(row["r_outer"]),
            )

    ne: dict[int, float] = {}
    with plasma_path.open() as handle:
        for row in csv.DictReader(handle):
            ne[int(row["shell_id"])] = float(row["n_e"])

    if set(geometry) != set(ne):
        raise ValueError("geometry and plasma-state shell sets differ")
    if set(range(NSH)) - set(geometry):
        raise ValueError(f"geometry/plasma state do not contain every shell s0..s{NSH - 1}")

    nshell_tot = len(geometry)
    expected_shells = set(range(nshell_tot))
    if set(geometry) != expected_shells:
        raise ValueError("shell ids must be contiguous from zero")
    dr = np.array(
        [geometry[s][1] - geometry[s][0] for s in range(nshell_tot)],
        dtype=np.float64,
    )

    pop: dict[int, np.ndarray] = {}
    gof: dict[int, int] = {}
    print(f"[{time.time() - t0:.0f}s] loading levelpop: {levelpop_path}")
    with levelpop_path.open() as handle:
        reader = csv.reader(handle)
        header = next(reader)
        columns = {name: index for index, name in enumerate(header)}
        required = {"shell", "Z", "ion", "level_num", "g", "n_k"}
        missing = required - set(columns)
        if missing:
            raise ValueError(f"levelpop missing columns: {sorted(missing)}")
        for row in reader:
            shell = int(row[columns["shell"]])
            if shell >= NSH:
                continue
            z = int(row[columns["Z"]])
            ion = int(row[columns["ion"]])
            level = int(row[columns["level_num"]])
            population_key = key(z, ion, level)
            if population_key not in pop:
                pop[population_key] = np.zeros(NSH)
                gof[population_key] = int(row[columns["g"]])
            pop[population_key][shell] = float(row[columns["n_k"]])
    if not pop:
        raise ValueError("levelpop contains no populations for s0..s6")
    print(
        f"[{time.time() - t0:.0f}s] levelpop: {len(pop)} unique levels "
        f"(shells 0..{NSH - 1})"
    )

    print(f"[{time.time() - t0:.0f}s] loading line list: {line_list_path}")
    lines = pd.read_csv(
        line_list_path,
        usecols=[
            "atomic_number",
            "ion_number",
            "level_number_lower",
            "level_number_upper",
            "f_lu",
            "wavelength",
            "nu",
        ],
    )
    z = lines["atomic_number"].to_numpy()
    ion = lines["ion_number"].to_numpy()
    lower = lines["level_number_lower"].to_numpy()
    upper = lines["level_number_upper"].to_numpy()
    f_lu = lines["f_lu"].to_numpy()
    wavelength_a = lines["wavelength"].to_numpy()
    nu = lines["nu"].to_numpy()
    wavelength_cm = wavelength_a * 1e-8
    lower_keys = key(z, ion, lower)
    upper_keys = key(z, ion, upper)
    n_lines = len(z)
    if n_lines == 0:
        raise ValueError("line list is empty")
    print(f"[{time.time() - t0:.0f}s] {n_lines} lines")

    keys = np.fromiter(pop.keys(), dtype=np.int64)
    order = np.argsort(keys)
    sorted_keys = keys[order]
    populations = np.array([pop[k] for k in keys], dtype=np.float64)[order]
    statistical_weights = np.array([gof[k] for k in keys], dtype=np.float64)[order]

    def lookup(query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        indices = np.searchsorted(sorted_keys, query)
        in_range = indices < len(sorted_keys)
        safe_indices = np.where(in_range, indices, 0)
        hits = in_range & (sorted_keys[safe_indices] == query)
        return safe_indices, hits

    lower_indices, lower_hits = lookup(lower_keys)
    upper_indices, upper_hits = lookup(upper_keys)
    # The 1e-30 missing-level convention is part of the historical audit being
    # replayed.  It is disclosed here and in every output row; no new fallback,
    # cap, or result adjustment is introduced for the capture.
    n_lower = np.where(lower_hits[:, None], populations[lower_indices], 1e-30)
    n_upper = np.where(upper_hits[:, None], populations[upper_indices], 1e-30)
    g_lower = np.where(lower_hits, statistical_weights[lower_indices], 1.0)
    g_upper = np.where(upper_hits, statistical_weights[upper_indices], 1.0)
    lower_hit_fraction = float(lower_hits.mean())
    upper_hit_fraction = float(upper_hits.mean())
    print(
        f"[{time.time() - t0:.0f}s] lower-level pop hits: "
        f"{lower_hit_fraction * 100:.1f}%  upper: {upper_hit_fraction * 100:.1f}%"
    )

    nu_lo, nu_hi = 1.5e14, 3.0e16
    nbin = 2000
    edges = np.logspace(np.log10(nu_lo), np.log10(nu_hi), nbin + 1)
    nu_center = np.sqrt(edges[:-1] * edges[1:])
    delta_nu = np.diff(edges)
    bin_index = np.searchsorted(edges, nu) - 1
    in_band_grid = (bin_index >= 0) & (bin_index < nbin)
    fuv = (wavelength_a >= 918.0) & (wavelength_a <= 1290.0)
    delta_nu_fuv = C_A / 918.0 - C_A / 1290.0

    results: list[tuple[int, float, float, float, float]] = []
    zero_lower_counts: list[int] = []
    zero_upper_counts: list[int] = []
    for shell in range(NSH):
        # lumina_plasma.c defines stim_corr=1 unless both populations are
        # positive, then applies the non-negative population-inversion rule.
        # Writing the branch explicitly avoids an artificial 0/0 for captured
        # zero populations.  Unlike production, this audit applies no 1e-100
        # floor to tau; n_lower=0 therefore gives tau=0 exactly.
        stimulated = np.ones(n_lines, dtype=np.float64)
        both_positive = (n_lower[:, shell] > 0.0) & (n_upper[:, shell] > 0.0)
        stimulated[both_positive] = 1.0 - (
            g_lower[both_positive]
            * n_upper[both_positive, shell]
            / (g_upper[both_positive] * n_lower[both_positive, shell])
        )
        inverted = both_positive & (stimulated < 0.0)
        stimulated[inverted] = 0.0
        zero_lower_counts.append(int(np.count_nonzero(n_lower[:, shell] == 0.0)))
        zero_upper_counts.append(int(np.count_nonzero(n_upper[:, shell] == 0.0)))
        tau_sobolev = (
            SOB * f_lu * wavelength_cm * T_EXP * n_lower[:, shell] * stimulated
        )
        one_minus_exp = 1.0 - np.exp(-tau_sobolev)
        weights = np.zeros(n_lines)
        weights[in_band_grid] = (
            nu[in_band_grid]
            / delta_nu[bin_index[in_band_grid]]
            * one_minus_exp[in_band_grid]
        )
        kappa_line = np.zeros(nbin)
        np.add.at(kappa_line, bin_index[in_band_grid], weights[in_band_grid])
        kappa_line /= C * T_EXP
        kappa_es = ne[shell] * SIGMA_T
        kappa_total = kappa_line + kappa_es
        rosseland_weights = d_bd_t(nu_center, T_R) * delta_nu
        kappa_rosseland = rosseland_weights.sum() / np.sum(
            rosseland_weights / kappa_total
        )
        kappa_fuv = (
            np.sum((nu[fuv] / delta_nu_fuv) * one_minus_exp[fuv])
            / (C * T_EXP)
            + kappa_es
        )
        results.append(
            (shell, kappa_rosseland, kappa_es, kappa_fuv, kappa_line.max())
        )
        print(
            f"  s{shell}: kap_Ross={kappa_rosseland:.3e}  "
            f"kap_es={kappa_es:.3e}  kap_FUV={kappa_fuv:.3e} cm^-1"
        )

    tau_es_out = np.array(
        [
            (
                np.array([ne[k] * SIGMA_T for k in range(nshell_tot)])[shell:]
                * dr[shell:]
            ).sum()
            for shell in range(nshell_tot)
        ]
    )
    kappa_rosseland = np.array([results[shell][1] for shell in range(NSH)])
    kappa_fuv = np.array([results[shell][3] for shell in range(NSH)])
    tau_rosseland_out = np.array(
        [
            (kappa_rosseland[shell:] * dr[shell:NSH]).sum() + tau_es_out[NSH]
            for shell in range(NSH)
        ]
    )
    tau_fuv_out = np.array(
        [
            (kappa_fuv[shell:] * dr[shell:NSH]).sum() + tau_es_out[NSH]
            for shell in range(NSH)
        ]
    )

    tau_sobolev_definition = (
        "tau_S=2.6540281e-2*f_lu*wavelength_cm*(19.48d)*n_lower*stim; "
        "stim=1 when n_lower<=0 or n_upper<=0, otherwise "
        "max(1-g_lower*n_upper/(g_upper*n_lower),0); no tau floor; missing "
        "line-list levels use the historical audit value n=1e-30,g=1"
    )
    kappa_rosseland_definition = (
        "2000 log-nu bins over 1.5e14..3.0e16 Hz; "
        "k_line=sum[(nu/dnu)*(1-exp(-tau_S))]/(c*t_exp); "
        "harmonic dBnu/dT mean of k_line+n_e*sigma_T at T=13120 K"
    )
    kappa_fuv_definition = (
        "single 918..1290 Angstrom rest-wavelength bin: "
        "sum[(nu/dnu_FUV)*(1-exp(-tau_S))]/(c*t_exp)+n_e*sigma_T"
    )
    outward_definition = (
        "sum kappa*dr over s(current)..s6 plus electron-scattering tau from "
        "s7 to surface; tau_es_out sums n_e*sigma_T*dr from current shell to surface"
    )

    header = [
        "shell",
        "v_kms",
        "tau_Ross_out",
        "tau_FUV_out",
        "tau_es_out",
        "kap_Ross",
        "kap_FUV",
        "kap_es",
        "lower_level_hit_fraction",
        "upper_level_hit_fraction",
        "zero_n_lower_line_count",
        "zero_n_upper_line_count",
        "geometry_source_file",
        "geometry_fields",
        "plasma_source_file",
        "plasma_field",
        "levelpop_source_file",
        "levelpop_fields",
        "line_list_source_file",
        "line_list_fields",
        "tau_sobolev_definition",
        "kap_Ross_definition",
        "kap_FUV_definition",
        "outward_tau_definition",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for shell in range(NSH):
            writer.writerow(
                [
                    shell,
                    4264 + 728 * shell,
                    tau_rosseland_out[shell],
                    tau_fuv_out[shell],
                    tau_es_out[shell],
                    results[shell][1],
                    results[shell][3],
                    results[shell][2],
                    lower_hit_fraction,
                    upper_hit_fraction,
                    zero_lower_counts[shell],
                    zero_upper_counts[shell],
                    geometry_path,
                    "shell_id;r_inner;r_outer",
                    plasma_path,
                    "shell_id;n_e",
                    levelpop_path,
                    "shell;Z;ion;level_num;g;n_k",
                    line_list_path,
                    "atomic_number;ion_number;level_number_lower;"
                    "level_number_upper;f_lu;wavelength;nu",
                    tau_sobolev_definition,
                    kappa_rosseland_definition,
                    kappa_fuv_definition,
                    outward_definition,
                ]
            )

    print("\n# Lumina outward optical depth (line part s0..s6 + es beyond s6)")
    print(f"{'shell':>5} {'tau_Ross':>9} {'tau_FUV':>9} {'tau_es':>8}")
    for shell in range(NSH):
        print(
            f"s{shell:>4} {tau_rosseland_out[shell]:>9.3f} "
            f"{tau_fuv_out[shell]:>9.3f} {tau_es_out[shell]:>8.3f}"
        )
    print(f"\n[{time.time() - t0:.0f}s] [wrote] {output_path}")


if __name__ == "__main__":
    main()
