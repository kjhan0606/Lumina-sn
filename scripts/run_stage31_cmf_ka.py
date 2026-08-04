#!/usr/bin/env python3
"""Stage 3.1 preregistered KA runner.  No model or GPU execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import subprocess
import sys

import mpmath as mp
import numpy as np
from scipy.interpolate import CubicSpline

from stage31_ka2_oracle import relative_difference as oracle_relative_difference
from stage31_ka2_oracle import solve as solve_ka2_oracle

ROOT = pathlib.Path(__file__).resolve().parents[1]
GRIDS_KA1 = ((128, 32), (256, 64), (512, 128))
GRIDS_KA2 = ((128, 32), (256, 64), (512, 128))
GRIDS_KA3 = ((32, 128), (64, 256), (128, 512), (256, 1024), (512, 2048),
             (1024, 4096))
KA3_OFFICIAL_TRIPLE = GRIDS_KA3[-3:]
TAU_CASES = (1.0e-3, 1.0, 100.0)
PREDICTED_P_WINDOWS = {
    1.0e-3: (1.90, 2.06),
    1.0: (1.90, 2.08),
    100.0: (1.85, 2.15),
}
KA3_SUPERSEDED_REV2_WINDOWS = {
    "p_obs_profile_l2": (1.25, 1.45),
    "finest_centroid_shift_error": (2.5e-5, 3.8e-5),
    "finest_invariant_area_relative_error": (4.0e-5, 5.5e-5),
    "finest_profile_relative_l2": (2.5e-3, 3.5e-3),
}
KA3_REV3_PREREGISTERED = {
    "scheme": "trapezoidal-start + branch-local quadratic-exact SC",
    "original_family_p_center": 2.00097,
    "official_triple_p_window": (1.96, 2.04),
    "finest_profile_relative_l2_center": 9.59e-5,
    "finest_profile_relative_l2_window": (8.8e-5, 1.08e-4),
    "extrapolation": "128x512 rev3 center 1.53593e-3 divided by 4^2",
}
KA3_REV4_PREREGISTERED = {
    "finest_profile_relative_l1_center": 2.64e-5,
    "finest_profile_relative_l1_window": (2.50e-5, 2.80e-5),
}


def compile_driver(executable: pathlib.Path, sanitize: bool = False) -> None:
    flags = ["-std=c11", "-O2", "-Wall", "-Wextra", "-Wpedantic", "-Werror"]
    if sanitize:
        flags = ["-std=c11", "-O1", "-g", "-Wall", "-Wextra", "-Wpedantic", "-Werror",
                 "-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
    command = ["gcc", *flags, "-fopenmp", "-Isrc", "scripts/stage31_cmf_ka_driver.c",
               "src/lumina_cmf_field.c", "-lm", "-o", str(executable)]
    subprocess.run(command, cwd=ROOT, check=True)


def read_table(path: pathlib.Path) -> tuple[dict[str, float], list[dict[str, float]]]:
    lines = path.read_text().splitlines()
    metadata: dict[str, float] = {}
    for token in lines[0][2:].split():
        key, value = token.split("=", 1)
        metadata[key] = float(value)
    names = lines[1].split("\t")
    rows = [dict(zip(names, map(float, line.split("\t")))) for line in lines[2:]]
    return metadata, rows


def exact_intensity(r: mp.mpf, mu: mp.mpf, chi: mp.mpf) -> mp.mpf:
    length = r * mu + mp.sqrt(1 - r * r * (1 - mu * mu))
    total_tau = chi * length
    m0 = -mp.expm1(-total_tau) / chi
    m1 = (1 - mp.exp(-total_tau) * (1 + total_tau)) / (chi * chi)
    m2 = (2 - mp.exp(-total_tau) * (total_tau * total_tau + 2 * total_tau + 2)) / (chi**3)
    return chi * (m0 + mp.mpf("0.5") * (r * r * m0 - 2 * r * mu * m1 + m2))


def relative_l2(values: list[float], exact: list[float]) -> float:
    numerator = math.fsum((a - b) ** 2 for a, b in zip(values, exact))
    denominator = math.fsum(b * b for b in exact)
    return math.sqrt(numerator / denominator)


def difference_norm(coarse: dict, fine: dict) -> float:
    if len(fine["J"]) != 2 * (len(coarse["J"]) - 1) + 1:
        raise ValueError("Richardson grids are not nested by a factor of two")
    restricted = [fine["J"][2*i] for i in range(len(coarse["J"]))]
    for i, (coarse_r, fine_r) in enumerate(zip(coarse["r"], fine["r"][::2])):
        if coarse_r != fine_r and not math.isclose(coarse_r, fine_r, rel_tol=1e-14, abs_tol=0.0):
            raise ValueError(f"non-nested evaluation radius at coarse index {i}")
    return math.sqrt(math.fsum((a - b) ** 2 for a, b in zip(coarse["J"], restricted))
                     / len(coarse["J"]))


def run_ka1(executable: pathlib.Path, work: pathlib.Path) -> dict:
    mp.mp.dps = 80
    cases = []
    overall = True
    for tau in TAU_CASES:
        levels = []
        for nr, nmu in GRIDS_KA1:
            table = work / f"ka1_tau{tau:g}_{nr}_{nmu}.tsv"
            subprocess.run([str(executable), "ka1", str(nr), str(nmu), repr(tau), str(table)],
                           cwd=ROOT, check=True)
            metadata, rows = read_table(table)
            radii, j_numeric, j_exact = [], [], []
            i_numeric, i_exact = [], []
            for i in range(nr + 1):
                shell_rows = rows[i * nmu:(i + 1) * nmu]
                r = shell_rows[0]["r"]
                radii.append(r)
                exact_j = mp.quad(lambda angle: exact_intensity(mp.mpf(str(r)), angle, mp.mpf(str(tau))), [-1, 1]) / 2
                j_exact.append(float(exact_j)); j_numeric.append(shell_rows[0]["J"])
                for row in shell_rows:
                    mu = mp.mpf(str(row["mu"]))
                    i_numeric.extend((row["Iminus"], row["Iplus"]))
                    i_exact.extend((float(exact_intensity(mp.mpf(str(r)), -mu, mp.mpf(str(tau)))),
                                    float(exact_intensity(mp.mpf(str(r)), mu, mp.mpf(str(tau))))))
            level = {
                "grid": {"nr": nr, "nmu": nmu}, "r": radii, "J": j_numeric,
                "J_relative_l2": relative_l2(j_numeric, j_exact),
                "I_relative_l2": relative_l2(i_numeric, i_exact),
                "max_scaled_error": max(max(abs(a-b) for a,b in zip(j_numeric,j_exact)),
                                        max(abs(a-b) for a,b in zip(i_numeric,i_exact))),
                "transport_residual": metadata["residual"],
                "clamp_count": int(metadata["clamp"]),
                "bdf_eta_negative_count": int(metadata["bdf_eta_negative"]),
                "solution_negative_excess_count": int(metadata["solution_negative_excess"]),
                "solution_subtruncation_count": int(metadata["solution_subtruncation"]),
                "solution_sign_indeterminate_subtruncation_count":
                    int(metadata["solution_sign_indeterminate_subtruncation"]),
                "solution_roundoff_enclosure_restart_count":
                    int(metadata["solution_roundoff_enclosure_restart"]),
                "sign_uncertain_count": int(metadata["sign_uncertain"]),
                "nonfinite_count": int(metadata["nonfinite"]),
            }
            levels.append(level)
        d12 = difference_norm(levels[0], levels[1])
        d24 = difference_norm(levels[1], levels[2])
        p_obs = math.log2(d12 / d24)
        finest = levels[-1]
        predicted_window = PREDICTED_P_WINDOWS[tau]
        prediction_pass = predicted_window[0] <= p_obs <= predicted_window[1]
        passed = (finest["I_relative_l2"] <= 1e-4 and finest["J_relative_l2"] <= 1e-4
                  and finest["max_scaled_error"] <= 3e-4 and 1.8 <= p_obs <= 2.2
                  and finest["transport_residual"] <= 1e-4
                  and finest["clamp_count"] == finest["solution_negative_excess_count"] == 0
                  and finest["sign_uncertain_count"] == finest["nonfinite_count"] == 0
                  and prediction_pass)
        overall = overall and passed
        for level in levels:
            del level["r"]
            del level["J"]
        cases.append({"tau_radius": tau, "levels": levels, "p_obs_J": p_obs,
                      "preregistered_p_window": predicted_window,
                      "preregistered_prediction_status": "PASS" if prediction_pass else "FAIL",
                      "outer_incoming_error": 0.0, "center_symmetry_error": 0.0,
                      "status": "PASS" if passed else "FAIL"})
    digest_runs = []
    deterministic_table = work / "ka1_determinism.tsv"
    for _ in range(3):
        subprocess.run([str(executable), "ka1", "512", "128", "1.0", str(deterministic_table)], cwd=ROOT, check=True)
        digest_runs.append(hashlib.sha256(deterministic_table.read_bytes()).hexdigest())
    deterministic = len(set(digest_runs)) == 1
    overall = overall and deterministic
    return {"ka": "KA1", "mpmath_dps": mp.mp.dps, "grids": GRIDS_KA1,
            "seed": None, "excluded_cells": 0, "cases": cases,
            "determinism_sha256": digest_runs, "determinism_pass": deterministic,
            "status": "PASS" if overall else "FAIL"}


def run_ka2(executable: pathlib.Path, work: pathlib.Path) -> dict:
    run_environment = {**os.environ, "OMP_NUM_THREADS": "32", "OMP_DYNAMIC": "FALSE"}
    levels = []
    output_targets = {i / nr for nr, _ in GRIDS_KA2 for i in range(nr + 1)}
    convergence_targets = {(i + 0.5) / GRIDS_KA2[-1][0]
                           for i in range(GRIDS_KA2[-1][0])}
    target_union = sorted(output_targets | convergence_targets)
    oracle_2048 = solve_ka2_oracle(2048, target_union)
    oracle_4096 = solve_ka2_oracle(4096, target_union)
    oracle_2048_map = dict(zip(target_union, oracle_2048["J"]))
    oracle_4096_map = dict(zip(target_union, oracle_4096["J"]))
    reference_difference = oracle_relative_difference(
        [oracle_2048_map[target] for target in sorted(convergence_targets)],
        [oracle_4096_map[target] for target in sorted(convergence_targets)],
    )
    oracle_map = oracle_4096_map
    for nr, nmu in GRIDS_KA2:
        table = work / f"ka2_{nr}_{nmu}.tsv"
        subprocess.run([str(executable), "ka2", str(nr), str(nmu), str(table)],
                       cwd=ROOT, check=True, env=run_environment)
        metadata, rows = read_table(table)
        radius = [row["r"] for row in rows]
        numeric = [row["J"] for row in rows]
        exact = [oracle_map[value] for value in radius]
        level = {
            "grid": {"nr": nr, "nmu": nmu},
            "r": radius,
            "J": numeric,
            "J_oracle_relative_l2": relative_l2(numeric, exact),
            "max_scaled_error": max(abs(left - right) for left, right in zip(numeric, exact)),
            "source_iterations": int(metadata["source_iterations"]),
            "source_residual": metadata["source_residual"],
            "transport_residual": metadata["transport_residual"],
            "energy_closure": metadata["energy_closure"],
            "L_thermal": metadata["Lthermal"],
            "L_escape": metadata["Lescape"],
            "L_absorbed": metadata["Labsorbed"],
            "clamp_count": int(metadata["clamp"]),
            "solution_negative_excess_count": int(metadata["solution_negative_excess"]),
            "solution_subtruncation_count": int(metadata["solution_subtruncation"]),
            "solution_sign_indeterminate_subtruncation_count":
                int(metadata["solution_sign_indeterminate_subtruncation"]),
            "solution_roundoff_enclosure_restart_count":
                int(metadata["solution_roundoff_enclosure_restart"]),
            "sign_uncertain_count": int(metadata["sign_uncertain"]),
            "nonfinite_count": int(metadata["nonfinite"]),
        }
        levels.append(level)
    differences = []
    for coarse, fine in zip(levels, levels[1:]):
        restricted = np.asarray(fine["J"])[::2]
        differences.append(float(np.sqrt(np.mean((np.asarray(coarse["J"]) - restricted) ** 2))))
    p_obs = math.log2(differences[0] / differences[1])
    finest = levels[-1]
    checks = {
        "oracle_nref_relative_difference_lt_1e-9": reference_difference < 1.0e-9,
        "oracle_full_arithmetic_is_80_digit":
            oracle_4096["matrix_storage"] == "80-digit",
        "finest_J_relative_l2_le_1e-4": finest["J_oracle_relative_l2"] <= 1.0e-4,
        "finest_max_scaled_error_le_3e-4": finest["max_scaled_error"] <= 3.0e-4,
        "p_obs_in_1p7_2p3": 1.7 <= p_obs <= 2.3,
        "finest_source_residual_le_1e-10": finest["source_residual"] <= 1.0e-10,
        "finest_transport_residual_le_1e-4": finest["transport_residual"] <= 1.0e-4,
        "finest_energy_closure_le_1e-4": finest["energy_closure"] <= 1.0e-4,
        "all_converged_within_max_iterations":
            all(level["source_iterations"] <= 500 for level in levels),
        "all_clamp_zero": all(level["clamp_count"] == 0 for level in levels),
        "all_solution_negative_zero":
            all(level["solution_negative_excess_count"] == 0 for level in levels),
        "all_sign_uncertain_zero": all(level["sign_uncertain_count"] == 0 for level in levels),
        "all_nonfinite_zero": all(level["nonfinite_count"] == 0 for level in levels),
    }
    passed = all(checks.values())
    for level in levels:
        del level["r"]
        del level["J"]
        level["status"] = "PASS" if passed else "FAIL"
    return {
        "ka": "KA2", "grids": GRIDS_KA2, "seed": None, "excluded_cells": 0,
        "parameters": {"chi0_R": 1.0, "epsilon": 0.2, "B0": 1.0},
        "cpu_threads": 32,
        "oracle": {
            "method": "Gauss-Legendre Nystrom with analytic logarithmic singularity subtraction",
            "mpmath_dps": 80,
            "Nref": [2048, 4096],
            "relative_difference": reference_difference,
            "Nref_2048_iterations": oracle_2048["iterations"],
            "Nref_4096_iterations": oracle_4096["iterations"],
            "matrix_storage": oracle_4096["matrix_storage"],
        },
        "levels": levels, "p_obs_J": p_obs, "difference_norms": differences,
        "acceptance_checks": checks, "acceptance_unchanged": True,
        "status": "PASS" if passed else "FAIL",
    }


def gaussian_cell_average(x: float, dx: float, center: float) -> float:
    sigma = 0.04
    root_two_sigma = math.sqrt(2.0) * sigma
    high = (x + 0.5 * dx - center) / root_two_sigma
    low = (x - 0.5 * dx - center) / root_two_sigma
    if low >= 0.0:
        difference = math.erfc(low) - math.erfc(high)
    elif high <= 0.0:
        difference = math.erfc(-high) - math.erfc(-low)
    else:
        difference = math.erf(high) - math.erf(low)
    return sigma * math.sqrt(math.pi / 2.0) * difference / dx


def gaussian_cell_average_oracle(x: float, dx: float, center: float) -> float:
    sigma = mp.mpf("0.04")
    x_mp, dx_mp, center_mp = map(lambda value: mp.mpf(str(value)), (x, dx, center))
    high = (x_mp + dx_mp / 2 - center_mp) / (mp.sqrt(2) * sigma)
    low = (x_mp - dx_mp / 2 - center_mp) / (mp.sqrt(2) * sigma)
    return float(sigma * mp.sqrt(mp.pi / 2) * (mp.erf(high) - mp.erf(low)) / dx_mp)


def profile_metrics(rows: list[dict[str, float]], metadata: dict[str, float]) -> dict[str, float]:
    shift = metadata["A"]
    dx = metadata["dx"]
    x = [row["x"] for row in rows]
    incoming = [row["Iin"] for row in rows]
    output = [row["Iout"] for row in rows]
    exact = [math.exp(-3.0 * shift) * gaussian_cell_average(value, dx, -shift) for value in x]
    l1 = math.fsum(abs(a - b) for a, b in zip(output, exact)) / math.fsum(abs(b) for b in exact)
    l2 = relative_l2(output, exact)
    input_area = dx * math.fsum(incoming)
    output_area = dx * math.fsum(output)
    centroid_in = math.fsum(a * b for a, b in zip(x, incoming)) / math.fsum(incoming)
    centroid_out = math.fsum(a * b for a, b in zip(x, output)) / math.fsum(output)
    return {
        "profile_relative_l1": l1,
        "profile_relative_l2": l2,
        "centroid_shift_error": abs((centroid_out - centroid_in) + shift),
        "invariant_area_relative_error": abs(math.exp(3.0 * shift) * output_area - input_area) / input_area,
        "blue_boundary_fraction": max(incoming[0], output[0]),
        "red_boundary_fraction": max(incoming[-1], output[-1]),
        "solution_min": min(output),
    }


def load_certificate(path: pathlib.Path) -> dict[tuple[int, int], dict]:
    report = json.loads(path.read_text())
    if report.get("status") != "PASS":
        raise ValueError(f"MPFR certificate report is not PASS: {path}")
    return {(int(level["ns"]), int(level["nnu"])): level for level in report["levels"]}


def run_ka3(executable: pathlib.Path, work: pathlib.Path,
            certificate_path: pathlib.Path) -> dict:
    mp.mp.dps = 80
    certificates = load_certificate(certificate_path)
    levels: list[dict] = []
    for ns, nnu in GRIDS_KA3:
        table = work / f"ka3_{ns}_{nnu}.tsv"
        completed = subprocess.run([str(executable), "ka3", str(ns), str(nnu), str(table)],
                                   cwd=ROOT, text=True, capture_output=True)
        if completed.returncode != 0:
            levels.append({
                "grid": {"ns": ns, "nnu": nnu},
                "status": "FAIL",
                "exit_code": completed.returncode,
                "stderr": completed.stderr.strip(),
                "failure_class": "DRIVER_FAILURE",
            })
            return {
                "ka": "KA3", "grids": GRIDS_KA3, "seed": None,
                "excluded_cells": 0, "levels": levels,
                "remaining_levels_status": "NOT RUN after first failed grid",
                "acceptance_unchanged": True, "status": "FAIL",
            }
        metadata, rows = read_table(table)
        metrics = profile_metrics(rows, metadata)
        oracle_values = [gaussian_cell_average_oracle(row["x"], metadata["dx"], 0.0)
                         for row in rows]
        stable_values = [gaussian_cell_average(row["x"], metadata["dx"], 0.0)
                         for row in rows]
        naive_values = []
        for row in rows:
            sigma = 0.04
            root_two_sigma = math.sqrt(2.0) * sigma
            high = (row["x"] + 0.5 * metadata["dx"]) / root_two_sigma
            low = (row["x"] - 0.5 * metadata["dx"]) / root_two_sigma
            naive_values.append(sigma * math.sqrt(math.pi / 2.0) *
                                (math.erf(high) - math.erf(low)) / metadata["dx"])
        tail_relative = [abs(value - oracle) / oracle
                         for value, oracle in zip(stable_values, oracle_values) if oracle > 0.0]
        first_eta = None
        if int(metadata["bdf_eta_negative"]) > 0:
            first_eta = {
                "evaluation_index": int(metadata["first_eval"]),
                "frequency_index": int(metadata["first_k"]),
                "ray_index": int(metadata["first_ray"]),
                "segment_index": int(metadata["first_segment"]),
                "substep_index": int(metadata["first_substep"]),
                "endpoint_index": int(metadata["first_endpoint"]),
                "eta_eff": metadata["first_eta"],
                "term_previous": metadata["first_prev"],
                "term_previous2": metadata["first_prev2"],
                "decay_ratio": metadata["first_decay_ratio"],
                "theoretical_limit": metadata["first_theoretical_limit"],
            }
        certificate = certificates.get((ns, nnu))
        if (ns, nnu) in KA3_OFFICIAL_TRIPLE and certificate is None:
            raise ValueError(f"missing MPFR certificate for official grid {ns}x{nnu}")
        expected_bits = 4096 if (ns, nnu) == (1024, 4096) else 2048
        if certificate is not None and int(certificate["certificate_bits"]) != expected_bits:
            raise ValueError(f"wrong certificate precision for {ns}x{nnu}")
        metrics.update({
            "grid": {"ns": ns, "nnu": nnu},
            "transport_residual": metadata["residual"],
            "clamp_count": int(metadata["clamp"]),
            "bdf_eta_negative_count": int(metadata["bdf_eta_negative"]),
            "bdf_eta_negative_plane_count": int(metadata["bdf_eta_negative_planes"]),
            "bdf_eta_min": metadata["bdf_eta_min"]
                if math.isfinite(metadata["bdf_eta_min"]) else None,
            "solution_negative_excess_count": int(metadata["solution_negative_excess"]),
            "solution_subtruncation_count": int(metadata["solution_subtruncation"]),
            "solution_sign_indeterminate_subtruncation_count":
                int(metadata["solution_sign_indeterminate_subtruncation"]),
            "solution_roundoff_enclosure_restart_count":
                int(metadata["solution_roundoff_enclosure_restart"]),
            "legacy_sign_uncertain_count": int(metadata["sign_uncertain"]),
            "legacy_nonfinite_count": int(metadata["nonfinite"]),
            "certificate_bits": int(certificate["certificate_bits"])
                if certificate is not None else None,
            "certified_sign_uncertain_count": int(certificate["certified_sign_uncertain"])
                if certificate is not None else None,
            "certified_nonfinite_count": int(certificate["certified_nonfinite"])
                if certificate is not None else None,
            "certified_negative_count": int(certificate["certified_negative"])
                if certificate is not None else None,
            "certified_min_lower": certificate["certified_min_lower"]
                if certificate is not None else None,
            "certified_max_width": certificate["certified_max_width"]
                if certificate is not None else None,
            "solver_status": int(metadata["solver_status"]),
            "driver_stderr": completed.stderr.strip(),
            "first_bdf_eta_negative": first_eta,
            "stable_tail_oracle_max_relative_error": max(tail_relative),
            "stable_tail_positive_cells": sum(value > 0.0 for value in stable_values),
            "naive_erf_zero_cells": sum(value == 0.0 for value in naive_values),
            "status": "PENDING_RICHARDSON",
        })
        levels.append(metrics)
    official_levels = levels[-3:]
    p_obs = math.log2(official_levels[1]["profile_relative_l2"] /
                      official_levels[2]["profile_relative_l2"])
    finest = levels[-1]
    measured_for_windows = {
        "p_obs_profile_l2": p_obs,
        "finest_centroid_shift_error": finest["centroid_shift_error"],
        "finest_invariant_area_relative_error": finest["invariant_area_relative_error"],
        "finest_profile_relative_l2": finest["profile_relative_l2"],
    }
    window_status = {
        name: "PASS" if bounds[0] <= measured_for_windows[name] <= bounds[1] else "FAIL"
        for name, bounds in KA3_SUPERSEDED_REV2_WINDOWS.items()
    }
    rev3_measurements = {
        "official_triple_p_obs_profile_l2": p_obs,
        "finest_profile_relative_l2": finest["profile_relative_l2"],
    }
    rev3_window_status = {
        "official_triple_p_obs_profile_l2": "PASS"
            if KA3_REV3_PREREGISTERED["official_triple_p_window"][0] <= p_obs <=
               KA3_REV3_PREREGISTERED["official_triple_p_window"][1] else "FAIL",
        "finest_profile_relative_l2": "PASS"
            if KA3_REV3_PREREGISTERED["finest_profile_relative_l2_window"][0] <=
               finest["profile_relative_l2"] <=
               KA3_REV3_PREREGISTERED["finest_profile_relative_l2_window"][1] else "FAIL",
    }
    rev4_window = KA3_REV4_PREREGISTERED["finest_profile_relative_l1_window"]
    rev4_measurements = {
        "finest_profile_relative_l1": finest["profile_relative_l1"],
        "official_triple_p_obs_profile_l2": p_obs,
        "finest_profile_relative_l2": finest["profile_relative_l2"],
        "finest_centroid_shift_error": finest["centroid_shift_error"],
        "finest_invariant_area_relative_error": finest["invariant_area_relative_error"],
    }
    rev4_window_status = {
        "finest_profile_relative_l1": "PASS"
            if rev4_window[0] <= finest["profile_relative_l1"] <= rev4_window[1]
            else "FAIL",
    }
    acceptance_checks = {
        "finest_profile_relative_l1_le_1e-4": finest["profile_relative_l1"] <= 1.0e-4,
        "finest_profile_relative_l2_le_1e-4": finest["profile_relative_l2"] <= 1.0e-4,
        "finest_centroid_shift_error_le_1e-4": finest["centroid_shift_error"] <= 1.0e-4,
        "finest_invariant_area_relative_error_le_1e-4":
            finest["invariant_area_relative_error"] <= 1.0e-4,
        "p_obs_in_1p8_2p2": 1.8 <= p_obs <= 2.2,
        "finest_transport_residual_le_1e-4": finest["transport_residual"] <= 1.0e-4,
        "finest_blue_boundary_fraction_lt_1e-12": finest["blue_boundary_fraction"] < 1.0e-12,
        "finest_red_boundary_fraction_lt_1e-12": finest["red_boundary_fraction"] < 1.0e-12,
        "official_triple_clamp_zero":
            all(level["clamp_count"] == 0 for level in official_levels),
        "official_triple_solution_negative_excess_zero":
            all(level["solution_negative_excess_count"] == 0 for level in official_levels),
        "official_triple_certified_sign_uncertain_zero":
            all(level["certified_sign_uncertain_count"] == 0 for level in official_levels),
        "official_triple_certified_nonfinite_zero":
            all(level["certified_nonfinite_count"] == 0 for level in official_levels),
        "official_triple_certified_negative_zero":
            all(level["certified_negative_count"] == 0 for level in official_levels),
        "rev4_finest_l1_preregistered_window":
            rev4_window_status["finest_profile_relative_l1"] == "PASS",
    }
    passed = all(acceptance_checks.values())
    for level in levels: level["status"] = "PASS" if passed else "FAIL"
    return {"ka": "KA3", "grids": GRIDS_KA3, "seed": None, "excluded_cells": 0,
            "levels": levels, "p_obs_profile_l2": p_obs,
            "mpmath_dps": mp.mp.dps,
            "superseded_rev2_preregistered_windows": KA3_SUPERSEDED_REV2_WINDOWS,
            "superseded_rev2_measurements": measured_for_windows,
            "superseded_rev2_window_status": window_status,
            "superseded_rev2_applicability":
                "NOT_APPLICABLE_AFTER_REV3_SCHEME_AND_GRID_EXTENSION",
            "rev3_preregistered": KA3_REV3_PREREGISTERED,
            "rev3_measurements": rev3_measurements,
            "rev3_window_status": rev3_window_status,
            "rev4_preregistered": KA3_REV4_PREREGISTERED,
            "rev4_measurements": rev4_measurements,
            "rev4_window_status": rev4_window_status,
            "mpfr_certificate_report": str(certificate_path),
            "official_triple_grids": KA3_OFFICIAL_TRIPLE,
            "acceptance_checks": acceptance_checks,
            "acceptance_unchanged": True, "status": "PASS" if passed else "FAIL"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ka", choices=("ka1", "ka2", "ka3"), default="ka1")
    parser.add_argument("--work", type=pathlib.Path, default=pathlib.Path("/tmp/stage31_ka"))
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--certificate", type=pathlib.Path,
                        default=ROOT / "docs/s31_results/mpfr_cert_rung8_fine.json")
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    executable = args.work / "stage31_cmf_ka_driver"
    compile_driver(executable)
    if args.ka == "ka1":
        report = run_ka1(executable, args.work)
    elif args.ka == "ka2":
        report = run_ka2(executable, args.work)
    else:
        report = run_ka3(executable, args.work, args.certificate)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
