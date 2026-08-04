#!/usr/bin/env python3
"""Freeze E12 hypotheses before exact-bin redistribution and stage31.

The predictor follows E10: collapse the event-energy matrix to broad bands,
then multiply by broad-band frozen line-return power.  It deliberately does
not construct the exact 1000-bin source or read an E12 stage31 result.
"""
from __future__ import annotations

import argparse
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
    add_matrix_contract_args, read_fluor_matrix_from_args)
import stage31_cmf_field_bench as bench  # noqa: E402


class E12PreregisterError(RuntimeError):
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument(
        "--e9-payload", type=Path,
        default=ROOT / "validation/emiss_e9/emiss_e9_effective_iter10")
    parser.add_argument("--source-payload", type=Path, required=True)
    parser.add_argument(
        "--e9-stage31", type=Path,
        default=ROOT / "validation/emiss_e9/stage31_measurement.csv")
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "validation/emiss_e12")
    parser.add_argument("--shell", type=int, default=8)
    add_matrix_contract_args(parser)   # [N4] matrix generation contract
    args = parser.parse_args()
    try:
        matrix = read_fluor_matrix_from_args(args.matrix, args)
        e9 = check_artifact(args.e9_payload.resolve())
        source = check_artifact(args.source_payload.resolve())
        if e9.header != source.header:
            raise E12PreregisterError("E9/source payload headers differ")
        nr, nnu = int(e9.header[3]), int(e9.header[4])
        if (matrix.header["n_bins"] != nnu or
                matrix.header["n_shells"] != nr or
                not 0 <= args.shell < nr):
            raise E12PreregisterError("iteration/dimension/shell mismatch")

        edges, _, _ = bench.canonical_grid()
        widths = np.diff(edges)
        wavelength = bench.C_ANGSTROM / (0.5 * (edges[:-1] + edges[1:]))
        labels = np.asarray([classify(float(value)) for value in wavelength])

        with (ROOT / "validation/emiss_e9/summary.json").open() as stream:
            eps_mc = float(json.load(stream)["eps_MC"])
        e9_arrays = [np.asarray(x) for x in e9.arrays]
        source_arrays = [np.asarray(x) for x in source.arrays]
        eta_e9 = e9_arrays[7].reshape(nr, nnu)[:, ::-1]
        j_e9 = e9_arrays[8].reshape(nr, nnu)[:, ::-1]
        chi = source_arrays[4].reshape(nr, nnu)[:, ::-1]
        chi_line = chi - np.min(chi, axis=1)[:, None]
        eta_line = (1.0 - eps_mc) * chi_line * j_e9
        if (np.any(chi_line < 0.0) or np.any(eta_line < 0.0) or
                not np.isfinite(eta_line).all()):
            raise E12PreregisterError("invalid line-return source; no clamp allowed")

        active = np.flatnonzero(matrix.terminal_energy > 0.0)
        transition_energy: dict[str, dict[str, float]] = {}
        denominator: dict[str, float] = {}
        for ib in active:
            iband = str(labels[ib])
            denominator[iband] = denominator.get(iband, 0.0) + float(
                matrix.terminal_energy[ib])
            transition_energy.setdefault(iband, {})
            transition_energy[iband]["OUTSIDE"] = (
                transition_energy[iband].get("OUTSIDE", 0.0) +
                float(matrix.outside_energy[ib]))
        for row in matrix.edges:
            iband = str(labels[int(row["input_bin"])])
            oband = str(labels[int(row["output_bin"])])
            transition_energy.setdefault(iband, {})
            transition_energy[iband][oband] = (
                transition_energy[iband].get(oband, 0.0) +
                float(row["output_energy"]))
        transition = {
            iband: {oband: energy / denominator[iband]
                    for oband, energy in outputs.items()}
            for iband, outputs in transition_energy.items()
        }
        for iband, outputs in transition.items():
            if abs(math.fsum(outputs.values()) - 1.0) > 2.0e-12:
                raise E12PreregisterError(f"coarse closure failed: {iband}")

        baseline: dict[str, float] = {}
        line_power: dict[str, float] = {}
        for name, lo, hi in BANDS:
            mask = ((wavelength >= lo) &
                    ((wavelength < hi) if hi < 20000.0 else wavelength <= hi))
            baseline[name] = float(np.sum(
                eta_e9[args.shell, mask] * widths[mask]))
            line_power[name] = float(np.sum(
                eta_line[args.shell, mask] * widths[mask]))

        added = {name: 0.0 for name, _, _ in BANDS}
        added["OUTSIDE"] = 0.0
        for iband, power in line_power.items():
            for oband, probability in transition.get(iband, {}).items():
                added[oband] = added.get(oband, 0.0) + power * probability

        predictions: list[dict[str, Any]] = []
        for name, lo, hi in BANDS:
            removed = line_power[name] if name in transition else 0.0
            predicted = baseline[name] - removed + added[name]
            ratio = predicted / baseline[name]
            predictions.append({
                "band": name, "lambda_lo_A": lo, "lambda_hi_A": hi,
                "baseline_source_energy": baseline[name],
                "same_bin_line_return_removed": removed,
                "coarse_redistributed_energy_added": added[name],
                "coarse_predicted_source_energy": predicted,
                "coarse_predicted_source_ratio_to_E9": ratio,
                "registered_direction": "down" if ratio < 1.0 else "up",
                "registered_ratio_low_minus25pct": 0.75 * ratio,
                "registered_ratio_high_plus25pct": 1.25 * ratio,
            })

        kpacket_scale = (matrix.header["kpacket_absorbed_energy"] /
                         matrix.header["absorbed_energy"])
        prereg = {
            "schema": "lumina-emiss-e12-preregistration-v1",
            "status": "FROZEN-BEFORE-EXACT-BIN-APPLICATION-AND-STAGE31",
            "shell": args.shell,
            "predictor": (
                "LFMAT001 global event-energy matrix collapsed to broad bands "
                "before multiplication by broad-band frozen E9 line-return power"),
            "known_limit": (
                "band collapse discards within-band source/matrix covariance and "
                "does not model nonlocal formal-transport response"),
            "formal_matrix_sha256": matrix.sha256,
            "payload_sha256": e9.manifest["sha256"],
            "source_payload_sha256": source.manifest["sha256"],
            "matrix_iteration": matrix.header["iteration"],
            "source_iteration": int(source.header[5]),
            "eps_MC": eps_mc,
            "kpacket_absorbed_energy_fraction": kpacket_scale,
            "predictions": predictions,
            "acceptance": {
                "H1_B2_to_B0_dominance_disappears": {
                    "metric": "source-weighted B2 fraction of all B0 redistributed inflow",
                    "maximum": kpacket_scale,
                    "charter_readout": "at or below the measured k-packet 2%-scale",
                },
                "H2_B0_worsening_disappears": {
                    "metric": "stage31 B0 J_det/CMFGEN",
                    "maximum": 8.290551056587633,
                    "contrast": "E10 prefix result 20.909501676590434",
                },
                "H3_B0_and_B1_fall": {
                    "metric": "E12/E9 stage31 band-integrated J_det",
                    "B0_maximum": 1.0, "B1_maximum": 1.0,
                },
                "H4_optical_rises": {
                    "metric": "E12/E9 source energy and stage31 J_det in 3000-10000 A",
                    "source_minimum": 1.0, "stage31_minimum": 1.0,
                },
                "energy": "abs((grid+outside)/removed-1) <= 1e-12",
                "guards": (
                    "same E10 stage31 driver, shell=8, nmu=16, T_inner=10020 K, "
                    "bb_scale=1; trip/nonfinite/solution_negative_excess/clamp must be zero"),
            },
            "production_code_modified": False,
            "new_model_or_GPU_run": False,
            "clamp": 0,
            "fallback": 0,
        }
        args.out_dir.mkdir(parents=True, exist_ok=True)
        out = args.out_dir / "preregistration.json"
        out.write_text(json.dumps(prereg, indent=2, sort_keys=True,
                                  allow_nan=False) + "\n")
        print(json.dumps({
            "path": str(out),
            "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            "kpacket_scale": kpacket_scale,
            "coarse_ratios": {row["band"]:
                               row["coarse_predicted_source_ratio_to_E9"]
                               for row in predictions},
        }, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (E12PreregisterError, OSError, ValueError, KeyError) as exc:
        print(f"UNRESOLVED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
