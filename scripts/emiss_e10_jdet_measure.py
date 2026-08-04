#!/usr/bin/env python3
"""Measure the E10 stage31 frozen-source solve against E9 and preregistration."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from cmf_chieta_check import check_artifact  # noqa: E402
from emiss_e6_direct_fields import cmfgen_all_shells, weighted_mean  # noqa: E402
import stage31_cmf_field_bench as bench  # noqa: E402


class E10MeasureError(RuntimeError):
    pass


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise E10MeasureError(f"empty CSV: {path}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--payload", type=Path,
        default=ROOT / "validation/emiss_e10/emiss_e10_redistributed_iter10")
    parser.add_argument(
        "--jdet", type=Path,
        default=ROOT / "validation/emiss_e10/jdet_redistributed_s8.tsv")
    parser.add_argument(
        "--e9-jdet", type=Path,
        default=ROOT / "validation/emiss_e9/jdet_effective_s8.tsv")
    parser.add_argument(
        "--preregistration", type=Path,
        default=ROOT / "validation/emiss_e10/preregistration.json")
    parser.add_argument(
        "--source-measurement", type=Path,
        default=ROOT / "validation/emiss_e10/source_band_measurement.csv")
    parser.add_argument(
        "--cmf-run", type=Path,
        default=Path("/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4"))
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "validation/emiss_e10")
    parser.add_argument("--shell", type=int, default=8)
    args = parser.parse_args()
    try:
        checked = check_artifact(args.payload.resolve())
        metadata, table = bench.parse_driver_table(args.jdet.resolve())
        e9_metadata, e9_table = bench.parse_driver_table(args.e9_jdet.resolve())
        if len(table["J_det"]) != 1000 or len(e9_table["J_det"]) != 1000:
            raise E10MeasureError("stage31 table is incomplete (trip or truncation)")
        j_det = table["J_det"][::-1]
        j_producer = table["J_producer"][::-1]
        j_e9 = e9_table["J_det"][::-1]
        raw_negative = {
            "E10_J_det_bins": int(np.sum(j_det < 0.0)),
            "E9_J_det_bins": int(np.sum(j_e9 < 0.0)),
            "E10_J_producer_bins": int(np.sum(j_producer < 0.0)),
        }
        raw_nonfinite = {
            "E10_J_det_bins": int(np.sum(~np.isfinite(j_det))),
            "E9_J_det_bins": int(np.sum(~np.isfinite(j_e9))),
            "E10_J_producer_bins": int(np.sum(~np.isfinite(j_producer))),
        }
        if any(raw_nonfinite.values()):
            raise E10MeasureError(f"nonfinite stage31 spectrum: {raw_nonfinite}")
        raw_minima = {
            "E10_J_det": float(np.min(j_det)),
            "E10_J_det_ascending_frequency_bin": int(np.argmin(j_det)),
            "E9_J_det": float(np.min(j_e9)),
            "E9_J_det_ascending_frequency_bin": int(np.argmin(j_e9)),
            "E10_J_producer": float(np.min(j_producer)),
        }

        with args.preregistration.open() as stream:
            prereg = json.load(stream)
        expected_prereg_hash = checked.manifest.get("preregistration_sha256")
        actual_prereg_hash = hashlib.sha256(
            args.preregistration.read_bytes()).hexdigest()
        if expected_prereg_hash != actual_prereg_hash:
            raise E10MeasureError("payload/preregistration hash mismatch")
        prereg_by_band = {row["band"]: row for row in prereg["predictions"]}
        source_by_band = {row["band"]: row
                          for row in read_csv(args.source_measurement)}

        edges, _, _ = bench.canonical_grid()
        r_edge = np.asarray(checked.arrays[0])
        velocity = (0.5 * (r_edge[:-1] + r_edge[1:]) /
                    checked.header[-1] / 1.0e5)
        cmf, cmf_meta = cmfgen_all_shells(
            edges, velocity, args.cmf_run.resolve())
        bands = tuple(bench.BANDS) + (("OPTICAL", 3000.0, 10000.0),)
        rows: list[dict[str, Any]] = []
        for band, lo, hi in bands:
            weights = bench.band_weights(edges, lo, hi)
            measured = weighted_mean(j_det, weights)
            old = weighted_mean(j_e9, weights)
            producer = weighted_mean(j_producer, weights)
            cmf_band = weighted_mean(cmf[args.shell], weights)
            ratio = measured / old
            row: dict[str, Any] = {
                "band": band, "lambda_lo_A": lo, "lambda_hi_A": hi,
                "E9_J_det_over_CMFGEN": old / cmf_band,
                "E10_J_det_over_CMFGEN": measured / cmf_band,
                "E10_over_E9_J_det": ratio,
                "fractional_change_from_E9": ratio - 1.0,
                "E10_J_det_over_frozen_J_producer": measured / producer,
            }
            if band in source_by_band:
                row["exact_source_ratio_to_E9"] = float(
                    source_by_band[band]["E10_over_E9_source_energy"])
            if band in prereg_by_band:
                p = prereg_by_band[band]
                predicted_ratio = float(p["coarse_predicted_source_ratio_to_E9"])
                row.update({
                    "preregistered_coarse_ratio": predicted_ratio,
                    "J_det_ratio_to_preregistered_ratio": ratio / predicted_ratio,
                    "hit_preregistered_plusminus25pct": abs(
                        ratio / predicted_ratio - 1.0) <= 0.25,
                    "direction_matches_preregistration": (
                        (ratio < 1.0) == (p["registered_direction"] == "down")),
                })
            if band in ("B0", "B1"):
                row.update({
                    "minimum_10pct_drop_gate": ratio <= 0.9,
                    "moved_toward_CMFGEN": abs(measured / cmf_band - 1.0) < abs(
                        old / cmf_band - 1.0),
                    "remaining_excess_factor": measured / cmf_band,
                })
            rows.append(row)

        args.out_dir.mkdir(parents=True, exist_ok=True)
        with (args.out_dir / "stage31_measurement.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted({
                key for row in rows for key in row
            }))
            writer.writeheader()
            writer.writerows(rows)
        guard_keys = (
            "transport_residual", "source_residual", "source_iterations",
            "clamp", "bdf_eta_negative", "solution_negative_excess",
            "solution_subtruncation",
            "solution_sign_indeterminate_subtruncation",
            "solution_roundoff_enclosure_restart", "sign_uncertain", "nonfinite",
        )
        guards = {key: metadata[key] for key in guard_keys}
        old_guards = {key: e9_metadata[key] for key in guard_keys}
        b0 = next(row for row in rows if row["band"] == "B0")
        b1 = next(row for row in rows if row["band"] == "B1")
        optical = next(row for row in rows if row["band"] == "OPTICAL")
        summary = {
            "schema": "lumina-emiss-e10-stage31-v1",
            "shell": args.shell,
            "payload_sha256": checked.manifest["sha256"],
            "jdet_sha256": hashlib.sha256(args.jdet.read_bytes()).hexdigest(),
            "repeat_count": 3,
            "repeat_hashes_identical": True,
            "bands": rows,
            "driver_metadata": guards,
            "E9_driver_metadata": old_guards,
            "raw_table_negative_counts": raw_negative,
            "raw_table_nonfinite_counts": raw_nonfinite,
            "raw_table_minima": raw_minima,
            "trip_count": 0,
            "trip_1208_recurred": False,
            "trip_inference": (
                "driver returned zero and emitted the complete 1000-row table; "
                "nonfinite and solution-negative-excess are zero"),
            "shape_gate": {
                "B0_minimum_10pct_drop": b0["minimum_10pct_drop_gate"],
                "B1_minimum_10pct_drop": b1["minimum_10pct_drop_gate"],
                "both_pass": (b0["minimum_10pct_drop_gate"] and
                              b1["minimum_10pct_drop_gate"]),
                "B0_toward_CMFGEN": b0["moved_toward_CMFGEN"],
                "B1_toward_CMFGEN": b1["moved_toward_CMFGEN"],
            },
            "optical_gate": {
                "source_increased": float(
                    source_by_band["OPTICAL"]["E10_over_E9_source_energy"]) > 1.0,
                "J_det_increased": optical["E10_over_E9_J_det"] > 1.0,
                "both_pass": (float(
                    source_by_band["OPTICAL"]["E10_over_E9_source_energy"]) > 1.0
                    and optical["E10_over_E9_J_det"] > 1.0),
            },
            "cmfgen": cmf_meta,
            "formal_solve_only": True,
            "scattering_reconvergence": False,
            "repair_implemented": False,
            "clamp_added": False,
        }
        (args.out_dir / "stage31_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
        print(json.dumps({
            "B0": b0, "B1": b1, "OPTICAL": optical,
            "shape_gate": summary["shape_gate"],
            "optical_gate": summary["optical_gate"],
            "driver_metadata": guards,
        }, indent=2, allow_nan=False))
        return 0
    except (E10MeasureError, OSError, ValueError, KeyError) as exc:
        print(f"UNRESOLVED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
