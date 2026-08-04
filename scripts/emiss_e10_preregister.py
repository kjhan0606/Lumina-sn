#!/usr/bin/env python3
"""Preregister the E10 coarse-band redistribution prediction.

This offline predictor deliberately does not apply the 1000-bin operator.  It
collapses the already recovered E9 event ledger to broad bands, applies those
broad-band probabilities to broad-band absorbed-line power, and freezes the
expected direction and magnitude before the exact-bin payload is constructed.
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
import stage31_cmf_field_bench as bench  # noqa: E402


class E10PredictionError(RuntimeError):
    pass


UV_BANDS = (
    ("B0", 600.0, 1000.0),
    ("B1", 1000.0, 1500.0),
    ("B2", 1500.0, 2000.0),
    ("B3", 2000.0, 2500.0),
    ("B4", 2500.0, 3000.0),
)
OUTPUT_BANDS = UV_BANDS + (
    ("OPTICAL", 3000.0, 10000.0),
    ("EUV", 100.0, 600.0),
    ("IR", 10000.0, 20000.0),
)


def band_name(wavelength: float) -> str:
    for name, lo, hi in OUTPUT_BANDS:
        if lo <= wavelength < hi or (name == "IR" and wavelength <= hi):
            return name
    return "OUTSIDE"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise E10PredictionError(f"empty CSV: {path}")
    return rows


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
        "--normalization", type=Path,
        default=ROOT / "validation/emiss_e9/redistribution_input_normalization_s8.csv")
    parser.add_argument(
        "--e9-stage31", type=Path,
        default=ROOT / "validation/emiss_e9/stage31_measurement.csv")
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "validation/emiss_e10")
    parser.add_argument("--shell", type=int, default=8)
    args = parser.parse_args()
    try:
        e9 = check_artifact(args.e9_payload.resolve())
        source = check_artifact(args.source_payload.resolve())
        if e9.header != source.header:
            raise E10PredictionError("E9/source payload headers differ")
        nr, nnu = int(e9.header[3]), int(e9.header[4])
        if not (0 <= args.shell < nr) or nnu != 1000:
            raise E10PredictionError("unexpected shell or frequency-grid size")

        e9_arrays = [np.asarray(x) for x in e9.arrays]
        source_arrays = [np.asarray(x) for x in source.arrays]
        eta_e9 = e9_arrays[7].reshape(nr, nnu)[:, ::-1]
        j_e9 = e9_arrays[8].reshape(nr, nnu)[:, ::-1]
        chi_original = source_arrays[4].reshape(nr, nnu)[:, ::-1]
        chi_es = np.min(chi_original, axis=1)[:, None]
        chi_line = chi_original - chi_es
        if np.any(chi_line < 0.0) or not np.isfinite(chi_line).all():
            raise E10PredictionError("line-opacity proxy invalid; no clamp allowed")

        with (ROOT / "validation/emiss_e9/summary.json").open() as stream:
            eps_mc = float(json.load(stream)["eps_MC"])
        eta_line_return = (1.0 - eps_mc) * chi_line * j_e9
        if np.any(eta_line_return < 0.0) or not np.isfinite(eta_line_return).all():
            raise E10PredictionError("same-bin line-return source invalid")

        edges, _, _ = bench.canonical_grid()
        widths = np.diff(edges)
        centers_lambda = bench.C_ANGSTROM / (0.5 * (edges[:-1] + edges[1:]))
        bin_band = [band_name(float(value)) for value in centers_lambda]
        norm_rows = read_csv(args.normalization)
        norm = {int(row["input_bin"]): row for row in norm_rows}
        if len(norm) != len(norm_rows):
            raise E10PredictionError("duplicate matrix normalization column")

        event_denominator: dict[str, float] = {name: 0.0 for name, _, _ in UV_BANDS}
        event_output: dict[str, dict[str, float]] = {
            name: {out: 0.0 for out in [x[0] for x in OUTPUT_BANDS] + ["OUTSIDE"]}
            for name, _, _ in UV_BANDS
        }
        for ib, row in norm.items():
            input_band = bin_band[ib]
            if input_band not in event_denominator:
                continue
            event_denominator[input_band] += float(row["terminal_output_energy"])
            event_output[input_band]["OUTSIDE"] += float(row["outside_grid_energy"])
        matrix_rows = read_csv(args.matrix)
        for row in matrix_rows:
            ib = int(row["input_bin"])
            ob = int(row["output_bin"])
            input_band = bin_band[ib]
            if input_band not in event_denominator:
                continue
            event_output[input_band][bin_band[ob]] += float(row["output_energy"])

        transition: dict[str, dict[str, float]] = {}
        for input_band, denominator in event_denominator.items():
            if not (denominator > 0.0 and math.isfinite(denominator)):
                raise E10PredictionError(f"no event energy in {input_band}")
            transition[input_band] = {
                output_band: energy / denominator
                for output_band, energy in event_output[input_band].items()
            }
            closure = math.fsum(transition[input_band].values())
            if abs(closure - 1.0) > 2.0e-14:
                raise E10PredictionError(
                    f"coarse probability closure failed in {input_band}: {closure}")

        s = args.shell
        baseline_energy: dict[str, float] = {}
        line_energy: dict[str, float] = {}
        for name, lo, hi in OUTPUT_BANDS:
            mask = ((centers_lambda >= lo) &
                    ((centers_lambda < hi) if hi < 20000.0
                     else (centers_lambda <= hi)))
            baseline_energy[name] = float(np.sum(eta_e9[s, mask] * widths[mask]))
            line_energy[name] = float(np.sum(eta_line_return[s, mask] * widths[mask]))

        predicted_output = {name: 0.0 for name, _, _ in OUTPUT_BANDS}
        predicted_outside = 0.0
        for input_band, _, _ in UV_BANDS:
            absorbed = line_energy[input_band]
            for output_band, probability in transition[input_band].items():
                if output_band == "OUTSIDE":
                    predicted_outside += absorbed * probability
                elif output_band in predicted_output:
                    predicted_output[output_band] += absorbed * probability

        e9_stage31 = {row["band"]: row for row in read_csv(args.e9_stage31)}
        predictions: list[dict[str, Any]] = []
        for name, _, _ in OUTPUT_BANDS:
            removed = line_energy[name] if name in event_denominator else 0.0
            predicted = baseline_energy[name] - removed + predicted_output[name]
            ratio = predicted / baseline_energy[name]
            row: dict[str, Any] = {
                "band": name,
                "baseline_source_energy": baseline_energy[name],
                "same_bin_line_return_removed": removed,
                "coarse_redistributed_energy_added": predicted_output[name],
                "coarse_predicted_source_energy": predicted,
                "coarse_predicted_source_ratio_to_E9": ratio,
                "registered_ratio_low_minus25pct": 0.75 * ratio,
                "registered_ratio_high_plus25pct": 1.25 * ratio,
                "registered_direction": "down" if ratio < 1.0 else "up",
            }
            if name in ("B0", "B1"):
                old_residual = float(e9_stage31[name]["J_det_over_CMFGEN"])
                row.update({
                    "E9_J_det_over_CMFGEN": old_residual,
                    "coarse_predicted_J_det_over_CMFGEN": old_residual * ratio,
                    "CMFGEN_shape_target": 1.0,
                    "fractional_drop_required_to_reach_CMFGEN": (
                        1.0 - 1.0 / old_residual),
                    "registered_minimum_fractional_drop": 0.10,
                    "shape_toward_CMFGEN_requires": (
                        "J_det/CMFGEN below E9 by at least 10%; exact result "
                        "also compared with coarse prediction +/-25%"),
                })
            predictions.append(row)

        total_removed = math.fsum(line_energy[name] for name, _, _ in UV_BANDS)
        total_added_grid = math.fsum(predicted_output.values())
        applied_prediction_closure = (
            (total_added_grid + predicted_outside) / total_removed - 1.0)
        prereg = {
            "schema": "lumina-emiss-e10-preregistration-v1",
            "status": "FROZEN-BEFORE-EXACT-BIN-APPLICATION-AND-STAGE31",
            "shell": s,
            "predictor": (
                "event-energy transition matrix collapsed to broad input/output "
                "bands before multiplication by broad-band E9 line-return power"),
            "known_limit": (
                "band collapse discards within-band source/matrix covariance and "
                "does not model nonlocal formal-transport response"),
            "premeasurement_hypothesis_readout": (
                "B1 is predicted to decrease by about 26%, but B0 is predicted "
                "to increase by about 56% because coarse B2-to-B0 inflow exceeds "
                "the removed B0 same-bin return.  Therefore this recovered "
                "prefix matrix is preregistered to fail the two-band shape gate "
                "unless exact-bin covariance and formal transport reverse B0."
            ),
            "acceptance": {
                "B0_B1": (
                    "both exact stage31 J_det/CMFGEN values must fall at least "
                    "10% below E9; +/-25% coarse-prediction agreement is secondary"),
                "optical": (
                    "3000-10000 A source energy and stage31 band-integrated J_det "
                    "must both increase; magnitude is compared with the frozen "
                    "coarse ratios at +/-25%"),
                "energy": (
                    "event construction error is reported independently; exact "
                    "application requires abs((grid+outside)/removed-1) <= 1e-12"),
                "verdict": (
                    "shape-to-CMFGEN only if B0 and B1 meet direction/minimum-drop "
                    "and optical rises without energy loss; otherwise identify "
                    "bin width, EPAY, boundary, or coverage as residual cause, or "
                    "UNRESOLVED when discriminating evidence is absent"),
            },
            "payload_sha256": e9.manifest["sha256"],
            "source_payload_sha256": source.manifest["sha256"],
            "matrix_sha256": hashlib.sha256(args.matrix.read_bytes()).hexdigest(),
            "normalization_sha256": hashlib.sha256(
                args.normalization.read_bytes()).hexdigest(),
            "eps_MC": eps_mc,
            "coarse_transition_probability": transition,
            "predictions": predictions,
            "predicted_redistribution": {
                "removed_UV_line_return_energy": total_removed,
                "added_on_grid_energy": total_added_grid,
                "outside_energy": predicted_outside,
                "relative_closure_error": applied_prediction_closure,
            },
            "new_model_or_GPU_run": False,
            "production_code_modified": False,
            "clamp_or_fallback": False,
        }
        args.out_dir.mkdir(parents=True, exist_ok=True)
        out = args.out_dir / "preregistration.json"
        out.write_text(json.dumps(prereg, indent=2, sort_keys=True,
                                  allow_nan=False) + "\n")
        print(json.dumps({
            "status": prereg["status"],
            "predictions": predictions,
            "predicted_redistribution": prereg["predicted_redistribution"],
            "preregistration_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        }, indent=2, allow_nan=False))
        return 0
    except (E10PredictionError, OSError, ValueError, KeyError) as exc:
        print(f"UNRESOLVED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
