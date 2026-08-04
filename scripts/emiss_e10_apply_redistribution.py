#!/usr/bin/env python3
"""Apply the recovered E9 energy-redistribution matrix to a frozen E9 source.

Only shell 8 and observed matrix input columns are changed.  No identity,
smoothing, nearest-bin, or missing-edge fallback is permitted.  Energy outside
the 100--20000 A diagnostic grid is retained in a side ledger, not clamped or
silently re-injected.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from cmf_chieta_check import check_artifact  # noqa: E402
import stage31_cmf_field_bench as bench  # noqa: E402
from emiss_e11_fluor_matrix import (  # noqa: E402
    FluorMatrixError, add_matrix_contract_args, read_fluor_matrix_from_args)


HEADER = struct.Struct("<8sIIQQQQIId")


class E10ApplyError(RuntimeError):
    pass


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise E10ApplyError(f"empty CSV: {path}")
    return rows


def serialize(header: tuple, arrays: list[np.ndarray]) -> bytes:
    pieces = [HEADER.pack(*header)]
    for values in arrays:
        pieces.append(np.asarray(values, dtype="<f8").reshape(-1).tobytes(order="C"))
    return b"".join(pieces)


def band_mask(wavelength: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return ((wavelength >= lo) &
            ((wavelength < hi) if hi < 20000.0 else (wavelength <= hi)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--e9-payload", type=Path,
        default=ROOT / "validation/emiss_e9/emiss_e9_effective_iter10")
    parser.add_argument(
        "--source-payload", type=Path,
        default=Path("/gpfs/kjhan/lumina_runner2/scratch/"
                     "emiss_ab2_capture_188766/emiss_ab_iter10.A"))
    parser.add_argument(
        "--matrix", type=Path,
        default=ROOT / "validation/emiss_e9/redistribution_matrix_s8_sparse.csv")
    parser.add_argument(
        "--matrix-format", choices=("auto", "prefix", "formal"), default="auto",
        help="auto recognizes LFMAT001; prefix preserves the E9 CSV path")
    parser.add_argument(
        "--normalization", type=Path,
        default=ROOT / "validation/emiss_e9/redistribution_input_normalization_s8.csv")
    parser.add_argument(
        "--matrix-summary", type=Path,
        default=ROOT / "validation/emiss_e9/redistribution_summary.json")
    parser.add_argument(
        "--preregistration", type=Path,
        default=ROOT / "validation/emiss_e10/preregistration.json")
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "validation/emiss_e10")
    parser.add_argument("--shell", type=int, default=8)
    parser.add_argument(
        "--column-closure-tolerance", type=float, default=2.0e-13,
        help="operator audit tolerance; E10 prefix default is unchanged")
    add_matrix_contract_args(parser)   # [N4] formal-matrix generation contract
    args = parser.parse_args()
    try:
        e9 = check_artifact(args.e9_payload.resolve())
        source = check_artifact(args.source_payload.resolve())
        if e9.header != source.header:
            raise E10ApplyError("E9/source payload headers differ")
        with args.preregistration.open() as stream:
            prereg = json.load(stream)
        if prereg.get("status") != "FROZEN-BEFORE-EXACT-BIN-APPLICATION-AND-STAGE31":
            raise E10ApplyError("preregistration is absent or not frozen")
        matrix_raw = args.matrix.read_bytes()
        matrix_format = args.matrix_format
        if matrix_format == "auto":
            matrix_format = "formal" if matrix_raw[:8] == b"LFMAT001" else "prefix"
        expected_hashes = {
            "payload_sha256": e9.manifest["sha256"],
            "source_payload_sha256": source.manifest["sha256"],
        }
        if matrix_format == "prefix":
            expected_hashes.update({
                "matrix_sha256": hashlib.sha256(matrix_raw).hexdigest(),
                "normalization_sha256": hashlib.sha256(
                    args.normalization.read_bytes()).hexdigest(),
            })
        elif prereg.get("formal_matrix_sha256") is not None:
            expected_hashes["formal_matrix_sha256"] = hashlib.sha256(
                matrix_raw).hexdigest()
        for key, value in expected_hashes.items():
            if prereg.get(key) != value:
                raise E10ApplyError(f"preregistration input hash mismatch: {key}")

        nr, nnu = int(e9.header[3]), int(e9.header[4])
        if nnu != 1000 or not (0 <= args.shell < nr):
            raise E10ApplyError("unexpected payload dimensions")
        e9_arrays = [np.asarray(x).copy() for x in e9.arrays]
        source_arrays = [np.asarray(x) for x in source.arrays]
        e9_fields = {
            "chi_coherent": e9_arrays[4].reshape(nr, nnu)[:, ::-1],
            "eta_fixed": e9_arrays[5].reshape(nr, nnu)[:, ::-1],
            "eta_coherent": e9_arrays[6].reshape(nr, nnu)[:, ::-1],
            "eta_total": e9_arrays[7].reshape(nr, nnu)[:, ::-1],
            "J": e9_arrays[8].reshape(nr, nnu)[:, ::-1],
        }
        for name, values in e9_fields.items():
            if not np.isfinite(values).all() or np.any(values <= 0.0):
                raise E10ApplyError(f"invalid E9 {name}; no clamp allowed")
        chi_original = source_arrays[4].reshape(nr, nnu)[:, ::-1]
        chi_es = np.min(chi_original, axis=1)[:, None]
        chi_line = chi_original - chi_es
        if np.any(chi_line < 0.0) or not np.isfinite(chi_line).all():
            raise E10ApplyError("line-opacity proxy invalid; no clamp allowed")
        eps_mc = float(prereg["eps_MC"])
        eta_line_return = (1.0 - eps_mc) * chi_line * e9_fields["J"]
        eta_electron_return = chi_es * e9_fields["J"]
        reconstructed = (e9_fields["eta_fixed"] + eta_electron_return +
                         eta_line_return)
        reconstruction_max = float(np.max(np.abs(
            reconstructed - e9_fields["eta_total"])))
        reconstruction_rel = reconstruction_max / float(np.max(
            np.abs(e9_fields["eta_total"])))
        if reconstruction_rel > 2.0e-15:
            raise E10ApplyError(
                f"E9 component reconstruction failed: {reconstruction_rel}")

        formal = None
        if matrix_format == "formal":
            formal = read_fluor_matrix_from_args(args.matrix, args)
            if (formal.header["n_bins"] != nnu or
                    formal.header["n_shells"] != nr):
                raise E10ApplyError("formal matrix/payload dimensions differ")
            expected_edges = bench.canonical_grid()[0]
            expected_dlog = math.log(expected_edges[-1] / expected_edges[0]) / nnu
            if not (math.isclose(formal.header["nu_min"], float(expected_edges[0]),
                                 rel_tol=2.0e-15) and
                    math.isclose(formal.header["nu_max"], float(expected_edges[-1]),
                                 rel_tol=2.0e-15) and
                    math.isclose(formal.header["d_log_nu"], expected_dlog,
                                 rel_tol=2.0e-15)):
                raise E10ApplyError("formal matrix/payload frequency grids differ")
            norm_rows = []
            for ib in np.flatnonzero(formal.terminal_energy > 0.0):
                denom_in = float(formal.input_energy[ib])
                denom_out = float(formal.terminal_energy[ib])
                norm_rows.append({
                    "input_bin": str(int(ib)),
                    "input_events_seen": str(int(formal.input_count[ib])),
                    "paired_terminals": str(int(formal.input_count[ib])),
                    "unpaired_prefix_tail": "0",
                    "matrix_plus_outside_count": str(int(formal.input_count[ib])),
                    "paired_input_energy": repr(denom_in),
                    "terminal_output_energy": repr(denom_out),
                    "energy_closure_output_over_input": repr(
                        denom_out / denom_in if denom_in else 0.0),
                    "outside_grid_count": "0",
                    "outside_grid_energy": repr(float(formal.outside_energy[ib])),
                })
            matrix_rows = [{
                "input_bin": str(int(row["input_bin"])),
                "output_bin": str(int(row["output_bin"])),
                "output_energy": repr(float(row["output_energy"])),
            } for row in formal.edges]
        else:
            norm_rows = read_csv(args.normalization)
            matrix_rows = read_csv(args.matrix)
        norm = {int(row["input_bin"]): row for row in norm_rows}
        if len(norm) != len(norm_rows):
            raise E10ApplyError("duplicate input normalization row")
        active = np.asarray(sorted(norm), dtype=np.int64)
        if np.any(active < 0) or np.any(active >= nnu):
            raise E10ApplyError("matrix input bin outside payload grid")

        redistributed_power = np.zeros(nnu, dtype=np.float64)
        outside_power_by_input = np.zeros(nnu, dtype=np.float64)
        probability_sum = np.zeros(nnu, dtype=np.float64)
        widths = np.diff(bench.canonical_grid()[0])
        s = args.shell
        removed_power_by_input = eta_line_return[s] * widths
        for ib, row in norm.items():
            denominator = float(row["terminal_output_energy"])
            if not (denominator > 0.0 and math.isfinite(denominator)):
                raise E10ApplyError(f"invalid probability denominator in bin {ib}")
            p_outside = float(row["outside_grid_energy"]) / denominator
            probability_sum[ib] = p_outside
            outside_power_by_input[ib] = removed_power_by_input[ib] * p_outside

        duplicate_edges = 0
        seen_edges: set[tuple[int, int]] = set()
        for row in matrix_rows:
            ib = int(row["input_bin"])
            ob = int(row["output_bin"])
            if ib not in norm or not (0 <= ob < nnu):
                raise E10ApplyError("matrix edge outside normalization/grid")
            edge = (ib, ob)
            if edge in seen_edges:
                duplicate_edges += 1
            seen_edges.add(edge)
            probability = (float(row["output_energy"]) /
                           float(norm[ib]["terminal_output_energy"]))
            if probability < 0.0 or not math.isfinite(probability):
                raise E10ApplyError("negative/nonfinite edge probability")
            probability_sum[ib] += probability
            redistributed_power[ob] += removed_power_by_input[ib] * probability
        if duplicate_edges:
            raise E10ApplyError(f"duplicate sparse edges: {duplicate_edges}")
        column_closure = probability_sum[active] - 1.0
        column_closure_max = float(np.max(np.abs(column_closure)))
        if (not math.isfinite(args.column_closure_tolerance) or
                args.column_closure_tolerance <= 0.0):
            raise E10ApplyError("invalid column-closure tolerance")
        if column_closure_max > args.column_closure_tolerance:
            raise E10ApplyError(
                f"operator column closure failed: {column_closure_max}")

        # The observed columns are the entire authorized application domain.
        # Other bins and all other shells remain the frozen E9 baseline; this is
        # explicitly a partial diagnostic, not an identity estimate for missing R.
        eta_new = e9_fields["eta_total"].copy()
        eta_new[s, active] -= eta_line_return[s, active]
        eta_new[s] += redistributed_power / widths
        eta_coherent_new = eta_new - e9_fields["eta_fixed"]
        # LCMFCE01 requires eta_total == eta_fixed + eta_coherent bitwise.
        # Re-form the total from the serialized components; this is a single
        # round-to-binary64 operation, not a clamp or normalization repair.
        eta_new = e9_fields["eta_fixed"] + eta_coherent_new
        negatives = {
            "eta_total": int(np.sum(eta_new < 0.0)),
            "eta_coherent": int(np.sum(eta_coherent_new < 0.0)),
            "redistributed_power": int(np.sum(redistributed_power < 0.0)),
        }
        nonfinite = {
            "eta_total": int(np.sum(~np.isfinite(eta_new))),
            "eta_coherent": int(np.sum(~np.isfinite(eta_coherent_new))),
            "redistributed_power": int(np.sum(~np.isfinite(redistributed_power))),
        }
        if any(negatives.values()) or any(nonfinite.values()):
            raise E10ApplyError(
                f"invalid constructed source; negative={negatives}, nonfinite={nonfinite}")

        removed = float(np.sum(removed_power_by_input[active]))
        injected = float(np.sum(redistributed_power))
        outside = float(np.sum(outside_power_by_input[active]))
        apply_relative_error = (injected + outside) / removed - 1.0
        if abs(apply_relative_error) > 1.0e-12:
            raise E10ApplyError(
                f"application energy gate failed: {apply_relative_error}")
        baseline_total = float(np.sum(e9_fields["eta_total"][s] * widths))
        new_total = float(np.sum(eta_new[s] * widths))
        full_source_relative_error = (new_total + outside) / baseline_total - 1.0

        e9_arrays[6] = eta_coherent_new[:, ::-1].reshape(-1)
        e9_arrays[7] = eta_new[:, ::-1].reshape(-1)
        raw = serialize(e9.header, e9_arrays)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        payload_path = args.out_dir / "emiss_e10_redistributed_iter10"
        payload_path.write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        manifest = {
            "schema": "LCMFCE01-v1",
            "sha256": digest,
            "iteration": int(e9.header[5]),
            "field_generation": int(e9.header[6]),
            "post_damping": True,
            "coherent_frozen": True,
            "frequency_descending": True,
            "eta_decomposition_bitwise": True,
            "eta_decomposition_max_abs": 0,
            "e10_diagnostic_only": True,
            "source_payload_sha256": e9.manifest["sha256"],
            "preregistration_sha256": hashlib.sha256(
                args.preregistration.read_bytes()).hexdigest(),
            "construction": (
                "shell-8 observed-column line return removed; energy-normalized "
                "sparse R output added by bin width; outside-grid energy side-ledgered"),
            "unsupported_bins_unchanged_not_inferred_identity": True,
            "repair_implemented": False,
            "clamp_or_floor_added": False,
            "fallback_added": False,
        }
        Path(str(payload_path) + ".manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        checked_out = check_artifact(payload_path)
        if checked_out.manifest["sha256"] != digest:
            raise E10ApplyError("written payload hash verification failed")

        edges = bench.canonical_grid()[0]
        wavelength = bench.C_ANGSTROM / (0.5 * (edges[:-1] + edges[1:]))
        bands = (
            ("B0", 600.0, 1000.0), ("B1", 1000.0, 1500.0),
            ("B2", 1500.0, 2000.0), ("B3", 2000.0, 2500.0),
            ("B4", 2500.0, 3000.0),
            ("OPTICAL", 3000.0, 10000.0),
            ("EUV", 100.0, 600.0), ("IR", 10000.0, 20000.0),
        )
        band_rows: list[dict[str, Any]] = []
        prereg_by_band = {row["band"]: row for row in prereg["predictions"]}
        for name, lo, hi in bands:
            mask = band_mask(wavelength, lo, hi)
            base = float(np.sum(e9_fields["eta_total"][s, mask] * widths[mask]))
            measured = float(np.sum(eta_new[s, mask] * widths[mask]))
            predicted_ratio = float(
                prereg_by_band[name]["coarse_predicted_source_ratio_to_E9"])
            band_rows.append({
                "band": name, "lambda_lo_A": lo, "lambda_hi_A": hi,
                "E9_source_energy": base,
                "E10_source_energy": measured,
                "E10_over_E9_source_energy": measured / base,
                "preregistered_coarse_ratio": predicted_ratio,
                "ratio_to_preregistered_prediction": measured / base / predicted_ratio,
                "hit_preregistered_plusminus25pct": abs(
                    measured / base / predicted_ratio - 1.0) <= 0.25,
                "redistributed_energy_added": float(np.sum(
                    redistributed_power[mask])),
            })
        with (args.out_dir / "source_band_measurement.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(band_rows[0]))
            writer.writeheader()
            writer.writerows(band_rows)

        if formal is not None:
            matrix_summary = {
                "paired_input_energy": formal.header["absorbed_energy"],
                "terminal_output_energy": formal.header["reemitted_energy"],
                "energy_conservation_relative_error":
                    formal.header["energy_conservation_relative_error"],
            }
        else:
            with args.matrix_summary.open() as stream:
                matrix_summary = json.load(stream)
        construction_error = float(
            matrix_summary["energy_conservation_relative_error"])
        construction_column_errors = [
            float(row["energy_closure_output_over_input"]) - 1.0
            for row in norm_rows
        ]
        unpaired = [int(row["unpaired_prefix_tail"]) for row in norm_rows]
        used_outputs = {int(row["output_bin"]) for row in matrix_rows}
        straddling_inputs = [
            int(ib) for ib in active
            if ((bench.C_ANGSTROM / edges[ib] > 3000.0 and
                 bench.C_ANGSTROM / edges[ib + 1] < 3000.0) or
                (bench.C_ANGSTROM / edges[ib] > 600.0 and
                 bench.C_ANGSTROM / edges[ib + 1] < 600.0))
        ]
        centered_uv_bins = {
            int(i) for i, value in enumerate(wavelength)
            if 600.0 <= value < 3000.0
        }
        active_set = set(int(i) for i in active)
        summary: dict[str, Any] = {
            "schema": "lumina-emiss-e10-redistribution-application-v1",
            "matrix_source_schema": ("LFMAT001-v1" if formal is not None
                                      else "lumina-emiss-e9-prefix-csv-v1"),
            "matrix_sha256": hashlib.sha256(matrix_raw).hexdigest(),
            "shell": s,
            "payload_sha256": digest,
            "preregistration_sha256": manifest["preregistration_sha256"],
            "construction_energy": {
                "paired_input_energy": matrix_summary["paired_input_energy"],
                "terminal_output_energy": matrix_summary["terminal_output_energy"],
                "output_over_input_relative_error": construction_error,
                "max_abs_input_column_relative_error": float(max(
                    abs(value) for value in construction_column_errors)),
            },
            "operator_normalization": {
                "max_abs_column_sum_minus_one_including_outside": column_closure_max,
                "acceptance_tolerance": args.column_closure_tolerance,
                "negative_probability_count": 0,
                "nonfinite_probability_count": 0,
            },
            "application_energy": {
                "removed_same_bin_line_return": removed,
                "injected_on_grid": injected,
                "outside_grid_side_ledger": outside,
                "relative_error_to_removed": apply_relative_error,
                "full_source_relative_error_including_outside": full_source_relative_error,
            },
            "coverage_and_missing": {
                "matrix_input_bins": len(active),
                "matrix_uncovered_bins_of_1000": nnu - len(active),
                "geometric_center_UV_bins_600_3000": len(centered_uv_bins),
                "uncovered_geometric_center_UV_bins": len(
                    centered_uv_bins - active_set),
                "active_boundary_bins_outside_center_UV_selection": len(
                    active_set - centered_uv_bins),
                "unsupported_shells_of_50": nr - 1,
                "matrix_shell_resolution": (
                    "global_plus_deep_photospheric_envelope_groups_and_per_shell_ledgers"
                    if formal is not None else "shell_8_prefix"),
                "sparse_observed_edges": len(seen_edges),
                "unobserved_edges_within_supported_columns": len(active) * nnu - len(seen_edges),
                "used_output_bins": len(used_outputs),
                "unused_output_bins": nnu - len(used_outputs),
                "outside_grid_terminal_count": (None if formal is not None else int(sum(
                    int(row["outside_grid_count"]) for row in norm_rows))),
                "unpaired_prefix_tail_count": int(sum(unpaired)),
                "input_bins_with_unpaired_prefix_tail": int(sum(x > 0 for x in unpaired)),
                "boundary_straddling_active_input_bins": straddling_inputs,
                "policy": (
                    "only observed columns applied; unsupported bins/shells remain "
                    "the named frozen E9 baseline and are not claimed as identity R; "
                    "unobserved edges receive exactly zero, with no smoothing/fill"),
            },
            "source_guards": {
                "negative": negatives,
                "nonfinite": nonfinite,
                "clamp": 0,
                "fallback": 0,
                "E9_component_reconstruction_max_abs": reconstruction_max,
                "E9_component_reconstruction_relative": reconstruction_rel,
            },
            "bands": band_rows,
            "EPAY_interpretation": (
                ("formal matrix carries independent absorbed/reemitted energy ledgers; "
                 "no normalization repair applied") if formal is not None else
                ("UNRESOLVED: event terminal/input float-energy closure is 7.5e-7, "
                 "and no independent serialized EPAY field exists in LCMFCE01")),
            "production_code_modified": False,
            "new_model_or_GPU_run": False,
            "repair_implemented": False,
        }
        (args.out_dir / "redistribution_application_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
        print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (E10ApplyError, FluorMatrixError, OSError, ValueError, KeyError) as exc:
        print(f"UNRESOLVED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
