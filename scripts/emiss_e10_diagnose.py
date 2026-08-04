#!/usr/bin/env python3
"""Diagnose E10 band flows and bound the emergent implication offline."""
from __future__ import annotations

import argparse
import csv
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


class DiagnosisError(RuntimeError):
    pass


BANDS = (
    ("B0", 600.0, 1000.0), ("B1", 1000.0, 1500.0),
    ("B2", 1500.0, 2000.0), ("B3", 2000.0, 2500.0),
    ("B4", 2500.0, 3000.0),
    ("OPTICAL", 3000.0, 10000.0),
    ("EUV", 100.0, 600.0), ("IR", 10000.0, 20000.0),
)


def classify(wavelength: float) -> str:
    for name, lo, hi in BANDS:
        if lo <= wavelength < hi or (name == "IR" and wavelength <= hi):
            return name
    return "OUTSIDE"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise DiagnosisError(f"empty CSV: {path}")
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
        "--preregistration", type=Path,
        default=ROOT / "validation/emiss_e10/preregistration.json")
    parser.add_argument(
        "--application-summary", type=Path,
        default=ROOT / "validation/emiss_e10/redistribution_application_summary.json")
    parser.add_argument(
        "--stage31-summary", type=Path,
        default=ROOT / "validation/emiss_e10/stage31_summary.json")
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "validation/emiss_e10")
    parser.add_argument("--shell", type=int, default=8)
    args = parser.parse_args()
    try:
        e9 = check_artifact(args.e9_payload.resolve())
        source = check_artifact(args.source_payload.resolve())
        nr, nnu = int(e9.header[3]), int(e9.header[4])
        if e9.header != source.header or nnu != 1000:
            raise DiagnosisError("payload/header mismatch")
        with args.preregistration.open() as stream:
            prereg = json.load(stream)
        with args.application_summary.open() as stream:
            application = json.load(stream)
        with args.stage31_summary.open() as stream:
            stage31 = json.load(stream)

        e9_arrays = [np.asarray(x) for x in e9.arrays]
        source_arrays = [np.asarray(x) for x in source.arrays]
        j = e9_arrays[8].reshape(nr, nnu)[:, ::-1]
        chi = source_arrays[4].reshape(nr, nnu)[:, ::-1]
        chi_line = chi - np.min(chi, axis=1)[:, None]
        eta_line = (1.0 - float(prereg["eps_MC"])) * chi_line * j
        edges, _, _ = bench.canonical_grid()
        widths = np.diff(edges)
        wavelength = bench.C_ANGSTROM / (0.5 * (edges[:-1] + edges[1:]))
        labels = [classify(float(value)) for value in wavelength]

        norm_rows = read_csv(args.normalization)
        norm = {int(row["input_bin"]): row for row in norm_rows}
        removed = eta_line[args.shell] * widths
        flow: dict[tuple[str, str], float] = {}
        for ib, row in norm.items():
            input_label = labels[ib]
            denominator = float(row["terminal_output_energy"])
            value = (removed[ib] * float(row["outside_grid_energy"]) /
                     denominator)
            flow[(input_label, "OUTSIDE")] = (
                flow.get((input_label, "OUTSIDE"), 0.0) + value)
        matrix_rows = read_csv(args.matrix)
        for row in matrix_rows:
            ib, ob = int(row["input_bin"]), int(row["output_bin"])
            value = (removed[ib] * float(row["output_energy"]) /
                     float(norm[ib]["terminal_output_energy"]))
            key = (labels[ib], labels[ob])
            flow[key] = flow.get(key, 0.0) + value

        flow_rows: list[dict[str, Any]] = []
        total_removed = float(np.sum(removed[list(norm)]))
        for (input_band, output_band), energy in sorted(flow.items()):
            flow_rows.append({
                "input_band": input_band,
                "output_band": output_band,
                "redistributed_energy": energy,
                "fraction_of_all_removed_line_return": energy / total_removed,
            })
        args.out_dir.mkdir(parents=True, exist_ok=True)
        with (args.out_dir / "band_flow_measurement.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flow_rows[0]))
            writer.writeheader()
            writer.writerows(flow_rows)

        output_totals = {name: 0.0 for name, _, _ in BANDS}
        output_totals["OUTSIDE"] = 0.0
        for (_, output_band), energy in flow.items():
            output_totals[output_band] += energy
        output_fractions = {
            name: value / total_removed for name, value in output_totals.items()
        }
        uv_retention = math.fsum(output_fractions[name]
                                 for name in ("B0", "B1", "B2", "B3", "B4"))
        optical_fraction = output_fractions["OPTICAL"]

        b0_inflow = {input_band: energy for (input_band, output_band), energy
                     in flow.items() if output_band == "B0"}
        b0_total = math.fsum(b0_inflow.values())
        b0_contributions = {
            name: {
                "energy": energy,
                "fraction_of_B0_redistributed_inflow": energy / b0_total,
            } for name, energy in sorted(b0_inflow.items())
        }
        boundary_bins = application["coverage_and_missing"][
            "boundary_straddling_active_input_bins"]
        boundary_removed = float(np.sum(removed[boundary_bins]))
        boundary_to_b0 = math.fsum(
            energy for (input_band, output_band), energy in flow.items()
            if input_band == "OPTICAL" and output_band == "B0")

        current_uv, target_uv = 42.9, 23.8
        current_blue, target_blue = 5.8, 14.5
        single_uv = current_uv * uv_retention
        single_blue_upper = current_blue + current_uv * optical_fraction
        interactions_to_uv_target = (
            math.log(target_uv / current_uv) / math.log(uv_retention))
        emergent = {
            "method": (
                "indirect energy ledger only; assumes every current emergent-UV "
                "unit experiences this shell-8 branching once"),
            "current_UV_percent": current_uv,
            "target_UV_percent": target_uv,
            "current_blue_percent": current_blue,
            "target_blue_percent": target_blue,
            "source_weighted_UV_retention_probability": uv_retention,
            "source_weighted_optical_probability": optical_fraction,
            "single_pass_implied_UV_percent": single_uv,
            "single_pass_UV_reduction_points": current_uv - single_uv,
            "single_pass_blue_upper_bound_percent": single_blue_upper,
            "single_pass_blue_upper_increase_points": single_blue_upper - current_blue,
            "effective_repeated_interactions_to_UV_target": interactions_to_uv_target,
            "direction_consistent": bool(
                single_uv < current_uv and single_blue_upper > current_blue),
            "magnitude_sufficient_single_pass": bool(
                single_uv <= target_uv and single_blue_upper >= target_blue),
            "limitations": (
                "stage31 does not compute emergent flux; 3000-10000 A is wider "
                "than the historical blue diagnostic; escape, shell migration, "
                "population feedback, the iteration-11 capped prefix, and repeated "
                "interactions are absent"),
        }

        b0_stage = next(row for row in stage31["bands"] if row["band"] == "B0")
        b1_stage = next(row for row in stage31["bands"] if row["band"] == "B1")
        diagnosis = {
            "schema": "lumina-emiss-e10-diagnosis-v1",
            "source_weighted_destination_fraction": output_fractions,
            "source_weighted_destination_closure_error": (
                math.fsum(output_fractions.values()) - 1.0),
            "B0_redistributed_inflow": {
                "total": b0_total,
                "by_input_band": b0_contributions,
                "dominant_input_band": max(b0_contributions,
                                           key=lambda key: b0_contributions[key][
                                               "energy"]),
            },
            "boundary_audit": {
                "straddling_input_bins": boundary_bins,
                "straddling_removed_energy_fraction": boundary_removed / total_removed,
                "3000A_boundary_to_B0_fraction_of_B0_inflow": (
                    boundary_to_b0 / b0_total),
            },
            "emergent_indirect": emergent,
            "residual_cause_readout": {
                "observed_immediate_cause": (
                    "source-weighted cross-band matrix inflow makes B0 grow; B2 is "
                    "the dominant B0 input contributor"),
                "bin_width": (
                    "NOT-SUPPORTED as arithmetic cause: eta*dnu was redistributed "
                    "and application closure is at roundoff"),
                "EPAY": (
                    "UNRESOLVED as an independent physical owner: LCMFCE01 has no "
                    "serialized EPAY field; captured terminal/input float-energy "
                    "closure is only a 7.5e-7 discrepancy, far below the B0 change"),
                "boundary": (
                    "quantified by the two straddling columns; its contribution is "
                    "reported and is not the dominant B0 inflow"),
                "production_relevance": (
                    "UNRESOLVED because R is an iteration-11 non-random 41.2% event "
                    "prefix applied to an iteration-10 frozen source"),
            },
            "verdict": {
                "B0_stage31_change_fraction": b0_stage["fractional_change_from_E9"],
                "B1_stage31_change_fraction": b1_stage["fractional_change_from_E9"],
                "shape_gate_pass": stage31["shape_gate"]["both_pass"],
                "optical_gate_pass": stage31["optical_gate"]["both_pass"],
                "structural_repair_design_basis_complete": False,
                "readout": (
                    "movement is mixed, not a CMFGEN-directed shape repair: B1 and "
                    "optical improve in the registered direction, but B0 worsens"),
            },
        }
        (args.out_dir / "diagnosis_summary.json").write_text(
            json.dumps(diagnosis, indent=2, sort_keys=True, allow_nan=False) + "\n")
        print(json.dumps(diagnosis, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (DiagnosisError, OSError, ValueError, KeyError) as exc:
        print(f"UNRESOLVED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
