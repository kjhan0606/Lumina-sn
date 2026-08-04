#!/usr/bin/env python3
"""Score the existing stage31 CPU solve against the E9 preregistration."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from cmf_chieta_check import check_artifact  # noqa: E402
from emiss_e6_direct_fields import cmfgen_all_shells, weighted_mean  # noqa: E402
import stage31_cmf_field_bench as bench  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", type=Path,
                        default=ROOT / "validation/emiss_e9/emiss_e9_effective_iter10")
    parser.add_argument("--jdet", type=Path,
                        default=ROOT / "validation/emiss_e9/jdet_effective_s8.tsv")
    parser.add_argument("--prediction", type=Path,
                        default=ROOT / "validation/emiss_e9/prediction_measurement.csv")
    parser.add_argument("--cmf-run", type=Path,
                        default=Path("/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4"))
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "validation/emiss_e9")
    parser.add_argument("--shell", type=int, default=8)
    args = parser.parse_args()
    try:
        checked = check_artifact(args.payload.resolve())
        metadata, table = bench.parse_driver_table(args.jdet.resolve())
        j_det = table["J_det"][::-1]
        j_effective = table["J_producer"][::-1]
        edges, _, _ = bench.canonical_grid()
        r_edge = np.asarray(checked.arrays[0])
        velocity = (0.5 * (r_edge[:-1] + r_edge[1:]) /
                    checked.header[-1] / 1.0e5)
        cmf, cmf_meta = cmfgen_all_shells(edges, velocity,
                                          args.cmf_run.resolve())
        with args.prediction.open() as stream:
            prereg = {r["band"]: r for r in csv.DictReader(stream)}
        rows = []
        for band, lo, hi in bench.BANDS:
            weights = bench.band_weights(edges, lo, hi)
            det = weighted_mean(j_det, weights)
            producer = weighted_mean(j_effective, weights)
            cmf_band = weighted_mean(cmf[args.shell], weights)
            prediction = float(prereg[band]["preregistered_J_over_CMFGEN"])
            ratio = det / cmf_band
            rows.append({
                "band": band, "lambda_lo_A": lo, "lambda_hi_A": hi,
                "preregistered_J_over_CMFGEN": prediction,
                "J_det_over_CMFGEN": ratio,
                "J_det_relative_to_prediction": ratio / prediction,
                "hit_within_10pct": abs(ratio / prediction - 1.0) <= 0.1,
                "J_det_over_arithmetic_effective_J": det / producer,
            })
        args.out_dir.mkdir(parents=True, exist_ok=True)
        with (args.out_dir / "stage31_measurement.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        acceptance = {
            key: metadata[key] for key in (
                "transport_residual", "source_residual", "source_iterations",
                "clamp", "bdf_eta_negative", "solution_negative_excess",
                "solution_subtruncation",
                "solution_sign_indeterminate_subtruncation",
                "solution_roundoff_enclosure_restart", "sign_uncertain", "nonfinite")
        }
        summary = {
            "schema": "lumina-emiss-e9-stage31-v1", "shell": args.shell,
            "payload_sha256": checked.manifest["sha256"],
            "jdet_sha256": hashlib.sha256(args.jdet.read_bytes()).hexdigest(),
            "bands": rows, "driver_metadata": acceptance,
            "cmfgen": cmf_meta,
            "formal_solve_only": True,
            "scattering_reconvergence": False,
            "repair_implemented": False,
            "trip_1208_recurred": False,
        }
        (args.out_dir / "stage31_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
        print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"UNRESOLVED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
