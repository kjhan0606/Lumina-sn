#!/usr/bin/env python3
"""Build and judge the preregistered T5 pure-rank-1 fluor-matrix proxy.

This is an offline analysis/fixture tool.  It reads an existing LFMAT001
capture, constructs R*[j,i] = q[j] without a floor, clamp, or missing-bin
fallback, and quantifies the information removed by that replacement.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from cmf_chieta_check import check_artifact  # noqa: E402
from emiss_e11_fluor_matrix import (  # noqa: E402
    CONTRACT_ITERATION, FluorMatrixError, read_fluor_matrix,
    write_fixture_matrix)
import stage31_cmf_field_bench as bench  # noqa: E402


class T5Error(RuntimeError):
    pass


BANDS = (
    ("EUV", 100.0, 600.0),
    ("B0", 600.0, 1000.0),
    ("B1", 1000.0, 1500.0),
    ("B2", 1500.0, 2000.0),
    ("B3", 2000.0, 2500.0),
    ("B4", 2500.0, 3000.0),
    ("OPTICAL", 3000.0, 10000.0),
    ("IR", 10000.0, 20000.0),
)


def classify(wavelength: float) -> str:
    for name, lo, hi in BANDS:
        if lo <= wavelength < hi or (name == "IR" and wavelength <= hi):
            return name
    return "OUTSIDE"


def normalize(values: np.ndarray, name: str) -> tuple[np.ndarray, float]:
    """Energy-normalize without clipping; return the final roundoff closure."""
    values = np.asarray(values, dtype=np.float64).copy()
    if np.any(values < 0.0) or not np.isfinite(values).all():
        raise T5Error(f"negative/nonfinite {name}; no clamp allowed")
    total = math.fsum(float(value) for value in values)
    if not total > 0.0:
        raise T5Error(f"zero {name}")
    values /= total
    positive = np.flatnonzero(values > 0.0)
    if not len(positive):
        raise T5Error(f"empty normalized {name}")
    # Enforce the stated energy-conservation normalization in binary64.  This
    # is a signed roundoff residual, not a floor/clamp or a fitted correction.
    closure = 1.0 - math.fsum(float(value) for value in values)
    values[int(positive[-1])] += closure
    if values[int(positive[-1])] <= 0.0:
        raise T5Error(f"normalization residual invalidated {name}")
    final = math.fsum(float(value) for value in values) - 1.0
    if final != 0.0:
        raise T5Error(f"{name} did not close exactly: {final}")
    return values, closure


def weighted_quantile(values: np.ndarray, weights: np.ndarray,
                      probability: float) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cdf = np.cumsum(sorted_weights) / math.fsum(float(x) for x in sorted_weights)
    return float(sorted_values[min(int(np.searchsorted(cdf, probability)),
                                   len(sorted_values) - 1)])


def variant_metrics(name: str, raw: np.ndarray, active: np.ndarray,
                    wavelength: np.ndarray, labels: np.ndarray,
                    q: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]],
                                            list[dict[str, Any]], np.ndarray]:
    row_energy = np.sum(raw, axis=1)
    if np.any(row_energy <= 0.0):
        raise T5Error(f"{name} contains an empty active row")
    probability = raw / row_energy[:, None]
    tvd = 0.5 * np.sum(np.abs(probability - q[None, :]), axis=1)
    weights = row_energy / math.fsum(float(value) for value in row_energy)
    residual = probability - q[None, :]
    frob_total = float(np.sum(probability * probability))
    frob_residual = float(np.sum(residual * residual))
    weighted_total = float(np.sum(weights[:, None] * probability * probability))
    weighted_residual = float(np.sum(weights[:, None] * residual * residual))

    singular = np.linalg.svd(probability, compute_uv=False, full_matrices=False)
    singular_energy = singular * singular
    singular_total = float(np.sum(singular_energy))
    spectrum_rows = [{
        "variant": name,
        "index": index + 1,
        "singular_value": float(value),
        "fraction_of_frobenius_energy": float(singular_energy[index] / singular_total),
        "cumulative_fraction_of_frobenius_energy": float(
            np.sum(singular_energy[:index + 1]) / singular_total),
    } for index, value in enumerate(singular)]

    input_rows = [{
        "variant": name,
        "input_bin": int(ib),
        "input_wavelength_A": float(wavelength[ib]),
        "input_band": str(labels[ib]),
        "input_energy_weight": float(weights[index]),
        "TVD_row_to_q": float(tvd[index]),
    } for index, ib in enumerate(active)]

    band_rows: list[dict[str, Any]] = []
    for band, _, _ in BANDS:
        select = labels[active] == band
        if not np.any(select):
            continue
        band_raw = np.sum(raw[select], axis=0)
        band_q, _ = normalize(band_raw, f"{name}-{band}-output")
        band_rows.append({
            "variant": name,
            "input_band": band,
            "active_input_bins": int(np.sum(select)),
            "input_energy_fraction": float(np.sum(row_energy[select]) /
                                             np.sum(row_energy)),
            "TVD_band_output_to_q": float(0.5 * np.sum(np.abs(band_q - q))),
        })

    summary = {
        "active_rows": int(len(active)),
        "output_bins": int(probability.shape[1]),
        "SVD": {
            "top_10_singular_values": [float(x) for x in singular[:10]],
            "optimal_rank1_fraction_of_frobenius_energy": float(
                singular_energy[0] / singular_total),
            "optimal_rank2_fraction_of_frobenius_energy": float(
                np.sum(singular_energy[:2]) / singular_total),
            "optimal_rank1_relative_frobenius_residual": float(
                math.sqrt(max(0.0, 1.0 - singular_energy[0] / singular_total))),
            "q_proxy_unweighted_relative_frobenius_residual": float(
                math.sqrt(frob_residual / frob_total)),
            "q_proxy_input_energy_weighted_relative_frobenius_residual": float(
                math.sqrt(weighted_residual / weighted_total)),
        },
        "TVD_row_to_q": {
            "input_energy_weighted_mean": float(np.sum(weights * tvd)),
            "input_energy_weighted_p50": weighted_quantile(tvd, weights, 0.50),
            "input_energy_weighted_p95": weighted_quantile(tvd, weights, 0.95),
            "minimum": float(np.min(tvd)),
            "maximum": float(np.max(tvd)),
        },
    }
    return summary, input_rows, band_rows, singular


def source_contract(q: np.ndarray, active: np.ndarray, labels: np.ndarray,
                    widths: np.ndarray, e9: Any, source: Any, shell: int,
                    eps_mc: float) -> tuple[list[dict[str, Any]], dict[str, float]]:
    nr, nnu = int(e9.header[3]), int(e9.header[4])
    e9_eta = np.asarray(e9.arrays[7]).reshape(nr, nnu)[:, ::-1]
    e9_j = np.asarray(e9.arrays[8]).reshape(nr, nnu)[:, ::-1]
    chi = np.asarray(source.arrays[4]).reshape(nr, nnu)[:, ::-1]
    chi_line = chi - np.min(chi, axis=1)[:, None]
    eta_line = (1.0 - eps_mc) * chi_line * e9_j
    if (np.any(chi_line < 0.0) or np.any(eta_line < 0.0) or
            not np.isfinite(eta_line).all()):
        raise T5Error("invalid line-return source; no clamp allowed")
    removed = eta_line[shell] * widths
    total_removed = float(np.sum(removed[active]))
    if not total_removed > 0.0:
        raise T5Error("zero active line-return power")

    predictions: list[dict[str, Any]] = []
    for band, lo, hi in BANDS:
        mask = labels == band
        baseline = float(np.sum(e9_eta[shell, mask] * widths[mask]))
        removed_band = float(np.sum(removed[active[labels[active] == band]]))
        added = float(np.sum(q[mask]) * total_removed)
        ratio = (baseline - removed_band + added) / baseline
        predictions.append({
            "band": band,
            "lambda_lo_A": lo,
            "lambda_hi_A": hi,
            "baseline_source_energy": baseline,
            "same_bin_line_return_removed": removed_band,
            "coarse_redistributed_energy_added": added,
            "coarse_predicted_source_energy": baseline - removed_band + added,
            "coarse_predicted_source_ratio_to_E9": ratio,
            "registered_direction": "down" if ratio < 1.0 else "up",
            "registered_ratio_low_minus25pct": 0.75 * ratio,
            "registered_ratio_high_plus25pct": 1.25 * ratio,
        })
    by_input = {
        band: float(np.sum(removed[active[labels[active] == band]]) / total_removed)
        for band, _, _ in BANDS
    }
    return predictions, by_input


def full_matrix_source_contract(matrix: Any, dense: np.ndarray,
                                active: np.ndarray, labels: np.ndarray,
                                widths: np.ndarray, e9: Any, source: Any,
                                shell: int, eps_mc: float
                                ) -> tuple[list[dict[str, Any]], float]:
    """Compute the exact-bin source contract for the current full-R control."""
    nr, nnu = int(e9.header[3]), int(e9.header[4])
    e9_eta = np.asarray(e9.arrays[7]).reshape(nr, nnu)[:, ::-1]
    e9_j = np.asarray(e9.arrays[8]).reshape(nr, nnu)[:, ::-1]
    chi = np.asarray(source.arrays[4]).reshape(nr, nnu)[:, ::-1]
    chi_line = chi - np.min(chi, axis=1)[:, None]
    eta_line = (1.0 - eps_mc) * chi_line * e9_j
    if (np.any(chi_line < 0.0) or np.any(eta_line < 0.0) or
            not np.isfinite(eta_line).all()):
        raise T5Error("invalid full-R line-return source; no clamp allowed")
    removed = eta_line[shell] * widths
    probability = dense / matrix.terminal_energy[active, None]
    redistributed = removed[active] @ probability
    predictions: list[dict[str, Any]] = []
    for band, lo, hi in BANDS:
        mask = labels == band
        baseline = float(np.sum(e9_eta[shell, mask] * widths[mask]))
        selected_active = active[labels[active] == band]
        removed_band = float(np.sum(removed[selected_active]))
        added = float(np.sum(redistributed[mask]))
        ratio = (baseline - removed_band + added) / baseline
        predictions.append({
            "band": band,
            "lambda_lo_A": lo,
            "lambda_hi_A": hi,
            "baseline_source_energy": baseline,
            "same_bin_line_return_removed": removed_band,
            "coarse_redistributed_energy_added": added,
            "coarse_predicted_source_energy": baseline - removed_band + added,
            "coarse_predicted_source_ratio_to_E9": ratio,
            "registered_direction": "down" if ratio < 1.0 else "up",
            "registered_ratio_low_minus25pct": 0.75 * ratio,
            "registered_ratio_high_plus25pct": 1.25 * ratio,
        })
    b0 = labels == "B0"
    b2_rows = labels[active] == "B2"
    total_b0 = float(np.sum(removed[active, None] * probability[:, b0]))
    b2_b0 = float(np.sum(
        removed[active[b2_rows], None] * probability[b2_rows][:, b0]))
    if not total_b0 > 0.0:
        raise T5Error("zero full-R B0 inflow")
    return predictions, b2_b0 / total_b0


def build(args: argparse.Namespace) -> int:
    matrix_path = args.matrix.resolve()
    # [N4] the generation contract is enforced INSIDE the reader so no caller
    # can read first and check later; deviating from the campaign contract
    # iteration must be stated explicitly.
    matrix = read_fluor_matrix(
        matrix_path, expected_iteration=args.expected_iteration,
        expected_sha256=args.expected_sha256,
        non_contract_override=(args.expected_iteration != CONTRACT_ITERATION
                               and args.matrix_non_contract_override))

    nb = matrix.header["n_bins"]
    active = np.flatnonzero(matrix.terminal_energy > 0.0)
    dense = np.zeros((len(active), nb), dtype=np.float64)
    row_lookup = np.full(nb, -1, dtype=np.int64)
    row_lookup[active] = np.arange(len(active))
    for row in matrix.edges:
        mapped = int(row_lookup[int(row["input_bin"])])
        if mapped < 0:
            raise T5Error("edge belongs to an inactive input")
        dense[mapped, int(row["output_bin"])] = float(row["output_energy"])
    on_grid = np.sum(dense, axis=1)
    if np.any(on_grid <= 0.0):
        raise T5Error("active input lacks on-grid output")
    offdiag = dense.copy()
    offdiag[np.arange(len(active)), active] = 0.0
    offdiag_energy = np.sum(offdiag, axis=1)
    if np.any(offdiag_energy <= 0.0):
        raise T5Error("active input lacks off-diagonal output")

    q_inclusive, inc_roundoff = normalize(
        np.sum(dense, axis=0), "q-diagonal-inclusive")
    q_exclusive, exc_roundoff = normalize(
        np.sum(offdiag, axis=0), "q-diagonal-exclusive")

    grid_edges, _, _ = bench.canonical_grid()
    if len(grid_edges) != nb + 1:
        raise T5Error("canonical grid/matrix dimension mismatch")
    wavelength = bench.C_ANGSTROM / (0.5 * (grid_edges[:-1] + grid_edges[1:]))
    labels = np.asarray([classify(float(value)) for value in wavelength])
    widths = np.diff(grid_edges)

    variants = {
        "diagonal_inclusive": (dense, q_inclusive, inc_roundoff),
        "diagonal_exclusive": (offdiag, q_exclusive, exc_roundoff),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    q_rows: list[dict[str, Any]] = []
    input_tvd_rows: list[dict[str, Any]] = []
    band_tvd_rows: list[dict[str, Any]] = []
    spectrum_rows: list[dict[str, Any]] = []
    residual_summary: dict[str, Any] = {}

    e9 = check_artifact(args.e9_payload.resolve())
    source = check_artifact(args.source_payload.resolve())
    if e9.header != source.header:
        raise T5Error("E9/source payload headers differ")
    if (int(e9.header[3]) != matrix.header["n_shells"] or
            int(e9.header[4]) != nb):
        raise T5Error("matrix/payload dimension mismatch")
    base_prereg = json.loads(args.base_preregistration.read_text())
    eps_mc = float(base_prereg["eps_MC"])

    for name, (raw, q, norm_roundoff) in variants.items():
        summary, input_rows, band_rows, singular = variant_metrics(
            name, raw, active, wavelength, labels, q)
        residual_summary[name] = summary
        input_tvd_rows.extend(input_rows)
        band_tvd_rows.extend(band_rows)
        singular_energy = singular * singular
        singular_total = float(np.sum(singular_energy))
        spectrum_rows.extend({
            "variant": name,
            "index": index + 1,
            "singular_value": float(value),
            "fraction_of_frobenius_energy": float(
                singular_energy[index] / singular_total),
            "cumulative_fraction_of_frobenius_energy": float(
                np.sum(singular_energy[:index + 1]) / singular_total),
        } for index, value in enumerate(singular))
        for ib in range(nb):
            q_rows.append({
                "variant": name,
                "output_bin": ib,
                "output_wavelength_A": float(wavelength[ib]),
                "output_band": str(labels[ib]),
                "q": float(q[ib]),
            })

        fixture = args.out_dir / f"rank1_{name}.lfmat"
        fixture_edges = [
            (int(ib), int(ob), float(matrix.terminal_energy[ib] * q[ob]))
            for ib in active for ob in np.flatnonzero(q > 0.0)
        ]
        write_fixture_matrix(
            fixture,
            nb=nb,
            ns=matrix.header["n_shells"],
            iteration=matrix.header["iteration"],
            numin=matrix.header["nu_min"],
            numax=matrix.header["nu_max"],
            input_count=matrix.input_count,
            input_energy=matrix.input_energy,
            terminal_energy=matrix.terminal_energy,
            outside_energy=np.zeros(nb, dtype=np.float64),
            shell_count=matrix.shell_count,
            shell_kpacket_count=matrix.shell_kpacket_count,
            shell_absorbed=matrix.shell_absorbed_energy,
            shell_reemitted=matrix.shell_reemitted_energy,
            edges=fixture_edges,
            kpacket_events=matrix.header["kpacket_events"],
            kpacket_absorbed=matrix.header["kpacket_absorbed_energy"],
            kpacket_reemitted=matrix.header["kpacket_reemitted_energy"],
        )
        checked_fixture = read_fluor_matrix(
            fixture, expected_iteration=matrix.header["iteration"],
            non_contract_override=True)
        predictions, input_source_fraction = source_contract(
            q, active, labels, widths, e9, source, args.shell, eps_mc)
        q_band = {
            band: float(np.sum(q[labels == band])) for band, _, _ in BANDS
        }
        contract = {
            "schema": "lumina-uv-t5-rank1-application-contract-v1",
            "status": "FROZEN-BEFORE-EXACT-BIN-APPLICATION-AND-STAGE31",
            "shell": args.shell,
            "variant": name,
            "definition": (
                "R*[j,i]=q[j] for every active input i; q is the input-energy-"
                "weighted mean on-grid output SED from the measured current matrix"),
            "diagonal_treatment_in_q": (
                "included" if name == "diagonal_inclusive" else
                "removed before each input-energy contribution is aggregated"),
            "energy_normalization": {
                "sum_q": math.fsum(float(x) for x in q),
                "signed_binary64_roundoff_residual_applied": norm_roundoff,
                "fixture_column_closure_max_abs":
                    checked_fixture.column_closure_max_abs,
                "outside_probability": 0.0,
            },
            "input_generation": {
                "source_matrix_path": str(matrix_path),
                "source_matrix_iteration": matrix.header["iteration"],
                "source_matrix_edges": matrix.header["sparse_nonzero_edges"],
                "source_matrix_sha256": matrix.sha256,
                "fixture_sha256": checked_fixture.sha256,
                "generation_mismatch_to_E12": True,
                "E12_matrix_iteration": 10,
                "E12_matrix_edges": 473045,
                "E12_matrix_sha256":
                    "2b65dba6f0d0ad5edf42739e445d20a9b4ac892cf180ae5791eee30dcd01c99b",
            },
            "formal_matrix_sha256": checked_fixture.sha256,
            "payload_sha256": e9.manifest["sha256"],
            "source_payload_sha256": source.manifest["sha256"],
            "matrix_iteration": matrix.header["iteration"],
            "source_iteration": int(source.header[5]),
            "eps_MC": eps_mc,
            "predictions": predictions,
            "q_band_fraction": q_band,
            "source_weighted_B2_fraction_of_all_B0_inflow":
                input_source_fraction["B2"],
            "source_weight_by_input_band": input_source_fraction,
            "acceptance": {
                "relative_tolerance": 0.15,
                "E12_B0_J_det_over_CMFGEN": 26.43249460092573,
                "E12_B1_J_det_over_CMFGEN": 5.65886463035863,
                "E12_B2_fraction_of_all_B0_inflow": 0.5492453247038226,
                "E12_optical_fractional_change_artifact": -0.07901687336697438,
                "request_literal_optical_fractional_change": -0.0079,
            },
            "production_code_modified": False,
            "new_model_or_GPU_run": False,
            "clamp": 0,
            "fallback": 0,
        }
        contract_path = args.out_dir / f"preregistration_{name}.json"
        contract_path.write_text(json.dumps(
            contract, indent=2, sort_keys=True, allow_nan=False) + "\n")
        residual_summary[name].update({
            "q_band_fraction": q_band,
            "source_weight_by_input_band": input_source_fraction,
            "fixture": str(fixture),
            "fixture_sha256": checked_fixture.sha256,
            "fixture_sparse_edges": checked_fixture.header["sparse_nonzero_edges"],
            "fixture_column_closure_max_abs":
                checked_fixture.column_closure_max_abs,
        })

    # Same-generation full-R control: this is not a third proxy variant.  It
    # isolates generation drift from information removed by R*.
    full_predictions, full_b2_fraction = full_matrix_source_contract(
        matrix, dense, active, labels, widths, e9, source, args.shell, eps_mc)
    full_contract = json.loads((
        args.out_dir / "preregistration_diagonal_inclusive.json").read_text())
    full_contract.update({
        "schema": "lumina-uv-t5-same-generation-full-R-control-v1",
        "variant": "current_full_R_control",
        "definition": (
            "Current measured full R, applied only as a same-generation control; "
            "it is not part of the pure-rank-1 acceptance pair."),
        "formal_matrix_sha256": matrix.sha256,
        "predictions": full_predictions,
        "source_weighted_B2_fraction_of_all_B0_inflow": full_b2_fraction,
    })
    full_contract["input_generation"]["fixture_sha256"] = None
    (args.out_dir / "preregistration_current_full_R.json").write_text(
        json.dumps(full_contract, indent=2, sort_keys=True,
                   allow_nan=False) + "\n")

    def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(args.out_dir / "q_by_bin.csv", q_rows)
    write_csv(args.out_dir / "input_row_tvd.csv", input_tvd_rows)
    write_csv(args.out_dir / "input_band_tvd.csv", band_tvd_rows)
    write_csv(args.out_dir / "svd_spectrum.csv", spectrum_rows)
    residual_summary["schema"] = "lumina-uv-t5-rank1-residual-v1"
    residual_summary["input_generation"] = {
        "path": str(matrix_path),
        "iteration": matrix.header["iteration"],
        "edges": matrix.header["sparse_nonzero_edges"],
        "sha256": matrix.sha256,
        "active_input_bins": int(len(active)),
        "diagonal_energy_fraction_of_on_grid": float(
            (np.sum(dense) - np.sum(offdiag)) / np.sum(dense)),
        "q_inclusive_vs_exclusive_TVD": float(
            0.5 * np.sum(np.abs(q_inclusive - q_exclusive))),
    }
    out_summary = args.out_dir / "rank1_residual_summary.json"
    out_summary.write_text(json.dumps(
        residual_summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(residual_summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


def relative_test(value: float, target: float, tolerance: float) -> dict[str, Any]:
    error = abs(value - target) / abs(target)
    return {"value": value, "target": target, "relative_error": error,
            "within_tolerance": error <= tolerance}


def judge(args: argparse.Namespace) -> int:
    full_summary = json.loads((
        args.out_dir / "current_full_R" / "stage31_summary.json").read_text())
    full_by_band = {row["band"]: row for row in full_summary["bands"]}
    results: dict[str, Any] = {}
    for variant in ("diagonal_inclusive", "diagonal_exclusive"):
        summary_path = args.out_dir / variant / "stage31_summary.json"
        contract_path = args.out_dir / f"preregistration_{variant}.json"
        summary = json.loads(summary_path.read_text())
        contract = json.loads(contract_path.read_text())
        by_band = {row["band"]: row for row in summary["bands"]}
        tolerance = float(contract["acceptance"]["relative_tolerance"])
        metrics = {
            "B0": relative_test(
                float(by_band["B0"]["E10_J_det_over_CMFGEN"]),
                float(contract["acceptance"]["E12_B0_J_det_over_CMFGEN"]),
                tolerance),
            "B1": relative_test(
                float(by_band["B1"]["E10_J_det_over_CMFGEN"]),
                float(contract["acceptance"]["E12_B1_J_det_over_CMFGEN"]),
                tolerance),
            "B2_to_B0": relative_test(
                float(contract["source_weighted_B2_fraction_of_all_B0_inflow"]),
                float(contract["acceptance"]["E12_B2_fraction_of_all_B0_inflow"]),
                tolerance),
            "optical_artifact_target": relative_test(
                float(by_band["OPTICAL"]["fractional_change_from_E9"]),
                float(contract["acceptance"]["E12_optical_fractional_change_artifact"]),
                tolerance),
            "optical_E12_over_E9_ratio_level": relative_test(
                float(by_band["OPTICAL"]["E10_over_E9_J_det"]),
                0.9209831266330256, tolerance),
            "optical_request_literal_target": relative_test(
                float(by_band["OPTICAL"]["fractional_change_from_E9"]),
                float(contract["acceptance"]["request_literal_optical_fractional_change"]),
                tolerance),
        }
        guards = summary["driver_metadata"]
        guard_pass = (
            int(guards["clamp"]) == 0 and int(guards["nonfinite"]) == 0 and
            int(guards["sign_uncertain"]) == 0 and
            int(guards["solution_negative_excess"]) == 0 and
            int(summary["trip_count"]) == 0)
        canonical_pass = all(metrics[key]["within_tolerance"]
                             for key in ("B0", "B1", "B2_to_B0"))
        requested_artifact_pass = all(metrics[key]["within_tolerance"]
                                      for key in ("B0", "B1",
                                                  "optical_artifact_target"))
        requested_literal_pass = all(metrics[key]["within_tolerance"]
                                     for key in ("B0", "B1",
                                                 "optical_request_literal_target"))
        results[variant] = {
            "metrics": metrics,
            "relative_error_to_same_generation_full_R_J_det_over_CMFGEN": {
                band: abs(float(by_band[band]["E10_J_det_over_CMFGEN"]) -
                          float(full_by_band[band]["E10_J_det_over_CMFGEN"])) /
                      abs(float(full_by_band[band]["E10_J_det_over_CMFGEN"]))
                for band in ("B0", "B1", "B2", "B3", "B4", "BALL", "OPTICAL")
            },
            "same_E10_stage31_guard_pass": guard_pass,
            "preregistered_T5_canonical_pass_B0_B1_B2_to_B0":
                canonical_pass and guard_pass,
            "requested_pass_using_E12_artifact_optical_minus7p90pct":
                requested_artifact_pass and guard_pass,
            "requested_pass_using_literal_optical_minus0p79pct":
                requested_literal_pass and guard_pass,
        }
    full_control = {
        "B0": relative_test(
            float(full_by_band["B0"]["E10_J_det_over_CMFGEN"]),
            26.43249460092573, 0.15),
        "B1": relative_test(
            float(full_by_band["B1"]["E10_J_det_over_CMFGEN"]),
            5.65886463035863, 0.15),
        "optical_fractional_change_artifact": relative_test(
            float(full_by_band["OPTICAL"]["fractional_change_from_E9"]),
            -0.07901687336697438, 0.15),
        "optical_E12_over_E9_ratio_level": relative_test(
            float(full_by_band["OPTICAL"]["E10_over_E9_J_det"]),
            0.9209831266330256, 0.15),
    }
    final = {
        "schema": "lumina-uv-t5-verdict-v1",
        "generation_caveat": (
            "T5 uses current iteration-11/08ff3312 matrix; E12 targets use the "
            "overwritten, unavailable iteration-10/2b65dba6 generation"),
        "optical_target_caveat": (
            "The request says -0.79%, while the canonical E12 artifact records "
            "-7.901687%; both readings are judged separately."),
        "route_disposition": "UNRESOLVED",
        "route_disposition_reason": (
            "The authoritative preregistration tests B0/B1/B2-to-B0 and passes, "
            "while the task text substitutes an optical fractional-change metric "
            "that fails under both its literal and artifact-corrected values. "
            "Selecting one after observing the result would be post-hoc."),
        "variants": results,
        "same_generation_current_full_R_control_vs_E12": full_control,
        "route_close_canonical": all(
            row["preregistered_T5_canonical_pass_B0_B1_B2_to_B0"]
            for row in results.values()),
        "route_close_requested_with_artifact_optical": all(
            row["requested_pass_using_E12_artifact_optical_minus7p90pct"]
            for row in results.values()),
        "route_close_requested_literal_optical": all(
            row["requested_pass_using_literal_optical_minus0p79pct"]
            for row in results.values()),
    }
    path = args.out_dir / "verdict.json"
    path.write_text(json.dumps(final, indent=2, sort_keys=True,
                               allow_nan=False) + "\n")
    print(json.dumps(final, indent=2, sort_keys=True, allow_nan=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--matrix", type=Path, required=True)
    build_parser.add_argument("--expected-iteration", type=int, required=True)
    build_parser.add_argument("--expected-sha256")
    build_parser.add_argument(
        "--matrix-non-contract-override", action="store_true",
        help=f"required when --expected-iteration != {CONTRACT_ITERATION}")
    build_parser.add_argument("--e9-payload", type=Path, required=True)
    build_parser.add_argument("--source-payload", type=Path, required=True)
    build_parser.add_argument("--base-preregistration", type=Path, required=True)
    build_parser.add_argument("--out-dir", type=Path, required=True)
    build_parser.add_argument("--shell", type=int, default=8)
    build_parser.set_defaults(func=build)
    judge_parser = sub.add_parser("judge")
    judge_parser.add_argument("--out-dir", type=Path, required=True)
    judge_parser.set_defaults(func=judge)
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except (T5Error, FluorMatrixError, OSError, ValueError, KeyError) as exc:
        print(f"UNRESOLVED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
