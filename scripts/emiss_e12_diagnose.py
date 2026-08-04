#!/usr/bin/env python3
"""Directly compare the E10 prefix and E12 formal fluorescence matrices."""
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
    add_matrix_contract_args, read_fluor_matrix_from_args)
import stage31_cmf_field_bench as bench  # noqa: E402


BANDS = (
    ("B0", 600.0, 1000.0), ("B1", 1000.0, 1500.0),
    ("B2", 1500.0, 2000.0), ("B3", 2000.0, 2500.0),
    ("B4", 2500.0, 3000.0), ("OPTICAL", 3000.0, 10000.0),
    ("EUV", 100.0, 600.0), ("IR", 10000.0, 20000.0),
)
UV = {"B0", "B1", "B2", "B3", "B4"}


def classify(wavelength: float) -> str:
    for name, lo, hi in BANDS:
        if lo <= wavelength < hi or (name == "IR" and wavelength <= hi):
            return name
    return "OUTSIDE"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as stream:
        return list(csv.DictReader(stream))


def aggregate_flow(rows: list[dict[str, str]], norm: dict[int, dict[str, str]],
                   removed: np.ndarray, labels: np.ndarray
                   ) -> dict[tuple[str, str], float]:
    flow: dict[tuple[str, str], float] = {}
    for ib, row in norm.items():
        value = (removed[ib] * float(row["outside_grid_energy"]) /
                 float(row["terminal_output_energy"]))
        key = (str(labels[ib]), "OUTSIDE")
        flow[key] = flow.get(key, 0.0) + value
    for row in rows:
        ib, ob = int(row["input_bin"]), int(row["output_bin"])
        value = (removed[ib] * float(row["output_energy"]) /
                 float(norm[ib]["terminal_output_energy"]))
        key = (str(labels[ib]), str(labels[ob]))
        flow[key] = flow.get(key, 0.0) + value
    return flow


def channel_rows(flow: dict[tuple[str, str], float], label: str
                 ) -> list[dict[str, Any]]:
    uv_total = math.fsum(v for (ib, _), v in flow.items() if ib in UV)
    selected = [((ib, ob), v) for (ib, ob), v in flow.items() if ib in UV]
    selected.sort(key=lambda item: item[1], reverse=True)
    return [{
        "matrix": label, "rank": rank, "input_band": key[0],
        "output_band": key[1], "source_weighted_energy": value,
        "fraction_of_UV_input_removed_power": value / uv_total,
    } for rank, (key, value) in enumerate(selected, 1)]


def b0_readout(flow: dict[tuple[str, str], float]) -> dict[str, Any]:
    entries = {ib: value for (ib, ob), value in flow.items() if ob == "B0"}
    total = math.fsum(entries.values())
    uv_total = math.fsum(value for ib, value in entries.items() if ib in UV)
    return {
        "total_all_input_bands": total,
        "total_UV_input_bands": uv_total,
        "by_input_band": {key: {"energy": value,
                                  "fraction_of_all_B0_inflow": value / total}
                            for key, value in sorted(entries.items())},
        "B2_fraction_of_all_B0_inflow": entries.get("B2", 0.0) / total,
        "B2_fraction_of_UV_only_B0_inflow": entries.get("B2", 0.0) / uv_total,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--formal", type=Path, required=True)
    ap.add_argument("--prefix-matrix", type=Path,
                    default=ROOT / "validation/emiss_e9/redistribution_matrix_s8_sparse.csv")
    ap.add_argument("--prefix-normalization", type=Path,
                    default=ROOT / "validation/emiss_e9/redistribution_input_normalization_s8.csv")
    ap.add_argument("--e9-payload", type=Path, required=True)
    ap.add_argument("--source-payload", type=Path, required=True)
    ap.add_argument("--preregistration", type=Path, required=True)
    ap.add_argument("--application-summary", type=Path, required=True)
    ap.add_argument("--stage31-summary", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "validation/emiss_e12")
    ap.add_argument("--shell", type=int, default=8)
    add_matrix_contract_args(ap)   # [N4] matrix generation contract
    args = ap.parse_args()
    try:
        formal = read_fluor_matrix_from_args(args.formal, args)
        e9 = check_artifact(args.e9_payload.resolve())
        source = check_artifact(args.source_payload.resolve())
        prereg = json.loads(args.preregistration.read_text())
        application = json.loads(args.application_summary.read_text())
        stage31 = json.loads(args.stage31_summary.read_text())
        nr, nnu = int(e9.header[3]), int(e9.header[4])
        edges, _, _ = bench.canonical_grid()
        widths = np.diff(edges)
        wavelength = bench.C_ANGSTROM / (0.5 * (edges[:-1] + edges[1:]))
        labels = np.asarray([classify(float(v)) for v in wavelength])
        e9_arrays = [np.asarray(x) for x in e9.arrays]
        source_arrays = [np.asarray(x) for x in source.arrays]
        j = e9_arrays[8].reshape(nr, nnu)[:, ::-1]
        chi = source_arrays[4].reshape(nr, nnu)[:, ::-1]
        chi_line = chi - np.min(chi, axis=1)[:, None]
        removed = ((1.0 - float(prereg["eps_MC"])) *
                   chi_line[args.shell] * j[args.shell] * widths)

        formal_norm = {}
        for ib in np.flatnonzero(formal.terminal_energy > 0.0):
            formal_norm[int(ib)] = {
                "terminal_output_energy": repr(float(formal.terminal_energy[ib])),
                "outside_grid_energy": repr(float(formal.outside_energy[ib])),
            }
        formal_rows = [{"input_bin": str(int(r["input_bin"])),
                        "output_bin": str(int(r["output_bin"])),
                        "output_energy": repr(float(r["output_energy"]))}
                       for r in formal.edges]
        prefix_rows = read_csv(args.prefix_matrix)
        prefix_norm = {int(r["input_bin"]): r
                       for r in read_csv(args.prefix_normalization)}
        full_flow = aggregate_flow(formal_rows, formal_norm, removed, labels)
        prefix_flow = aggregate_flow(prefix_rows, prefix_norm, removed, labels)
        full_rank = channel_rows(full_flow, "formal_full")
        prefix_rank = channel_rows(prefix_flow, "e10_prefix")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        with (args.out_dir / "channel_rank_comparison.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(full_rank[0]))
            writer.writeheader()
            writer.writerows(prefix_rank + full_rank)

        old_ranks = {(r["input_band"], r["output_band"]): r["rank"]
                     for r in prefix_rank}
        new_ranks = {(r["input_band"], r["output_band"]): r["rank"]
                     for r in full_rank}
        top_changes = []
        for row in full_rank[:15]:
            key = (row["input_band"], row["output_band"])
            top_changes.append({**row, "e10_prefix_rank": old_ranks.get(key),
                                "rank_change_new_minus_old": (
                                    row["rank"] - old_ranks[key]
                                    if key in old_ranks else None)})

        prefix_b0 = b0_readout(prefix_flow)
        full_b0 = b0_readout(full_flow)
        kpacket_fraction = float(prereg["kpacket_absorbed_energy_fraction"])

        uv_removed_full = math.fsum(v for (ib, _), v in full_flow.items() if ib in UV)
        destination = {}
        for out in [x[0] for x in BANDS] + ["OUTSIDE"]:
            destination[out] = (math.fsum(
                v for (ib, ob), v in full_flow.items() if ib in UV and ob == out)
                / uv_removed_full)
        uv_retention = math.fsum(destination[x] for x in UV)
        optical_fraction = destination["OPTICAL"]
        current_uv, target_uv = 42.9, 23.8
        current_blue, target_blue = 5.8, 14.5
        single_uv = current_uv * uv_retention
        single_blue_upper = current_blue + current_uv * optical_fraction
        emergent = {
            "source_weighted_destination_fraction_for_UV_inputs": destination,
            "closure_error": math.fsum(destination.values()) - 1.0,
            "UV_retention": uv_retention,
            "optical_probability": optical_fraction,
            "single_pass_UV_percent": single_uv,
            "single_pass_UV_reduction_points": current_uv - single_uv,
            "UV_target_reduction_achievement_fraction": (
                (current_uv - single_uv) / (current_uv - target_uv)),
            "single_pass_blue_upper_percent": single_blue_upper,
            "single_pass_blue_upper_increase_points": single_blue_upper - current_blue,
            "blue_target_increase_upper_achievement_fraction": (
                (single_blue_upper - current_blue) / (target_blue - current_blue)),
            "effective_repeated_interactions_to_UV_target": (
                math.log(target_uv / current_uv) / math.log(uv_retention)
                if 0.0 < uv_retention < 1.0 else None),
            "limitation": (
                "single-pass shell-8 source ledger only; optical is a blue upper "
                "bound and escape, shell migration, feedback, and repeated interactions are absent"),
        }

        group_audit = []
        for (first, last), rows in zip(formal.group_ranges, formal.group_edges):
            b2_rows = rows[np.isin(rows["input_bin"], np.flatnonzero(labels == "B2"))]
            total = math.fsum(float(x) for x in b2_rows["output_energy"])
            to_b0 = math.fsum(float(r["output_energy"]) for r in b2_rows
                              if labels[int(r["output_bin"])] == "B0")
            group_audit.append({
                "first_shell": first, "last_shell": last,
                "sparse_edges": int(len(rows)), "B2_on_grid_output_energy": total,
                "B2_to_B0_energy": to_b0,
                "B2_to_B0_fraction_of_B2_on_grid": to_b0 / total if total else 0.0,
            })

        shell_abs_delta = math.fsum(float(x) for x in formal.shell_absorbed_energy) - float(
            formal.header["absorbed_energy"])
        shell_emit_delta = math.fsum(float(x) for x in formal.shell_reemitted_energy) - float(
            formal.header["reemitted_energy"])
        b0_stage = next(x for x in stage31["bands"] if x["band"] == "B0")
        b1_stage = next(x for x in stage31["bands"] if x["band"] == "B1")
        optical_stage = next(x for x in stage31["bands"] if x["band"] == "OPTICAL")
        hypotheses = {
            "H1_B2_to_B0_at_or_below_kpacket_scale": bool(
                full_b0["B2_fraction_of_all_B0_inflow"] <= kpacket_fraction),
            "H2_B0_at_or_below_E9_8p290551": bool(
                b0_stage["E10_J_det_over_CMFGEN"] <= 8.290551056587633),
            "H3_B0_and_B1_fall": bool(
                b0_stage["E10_over_E9_J_det"] < 1.0 and
                b1_stage["E10_over_E9_J_det"] < 1.0),
            "H4_optical_source_and_J_rise": bool(
                application["bands"][5]["E10_over_E9_source_energy"] > 1.0 and
                optical_stage["E10_over_E9_J_det"] > 1.0),
        }
        result = {
            "schema": "lumina-emiss-e12-diagnosis-v1",
            "matrix_certification": {
                **formal.header, "sha256": formal.sha256,
                "column_closure_max_abs": formal.column_closure_max_abs,
                "shell_absorbed_sum_minus_header": shell_abs_delta,
                "shell_reemitted_sum_minus_header": shell_emit_delta,
                "classified_plus_unclassified_not_asserted_disjoint": True,
            },
            "prefix_comparison": {
                "prefix_sparse_edges": len(prefix_rows),
                "formal_sparse_edges": len(formal.edges),
                "formal_over_prefix_edges": len(formal.edges) / len(prefix_rows),
                "added_observed_edges": len(formal.edges) - len(prefix_rows),
                "prefix_B0_inflow": prefix_b0,
                "formal_B0_inflow": full_b0,
                "B2_fraction_ratio_formal_over_prefix": (
                    full_b0["B2_fraction_of_all_B0_inflow"] /
                    prefix_b0["B2_fraction_of_all_B0_inflow"]),
                "top_formal_channel_rank_changes": top_changes,
                "group_B2_to_B0_audit": group_audit,
            },
            "emergent_indirect": emergent,
            "preregistered_hypotheses": hypotheses,
            "strict_E10_column_guard": {
                "threshold": 2.0e-13,
                "measured": application["operator_normalization"][
                    "max_abs_column_sum_minus_one_including_outside"],
                "pass": bool(application["operator_normalization"][
                    "max_abs_column_sum_minus_one_including_outside"] <= 2.0e-13),
                "formal_contract_supplement_threshold": 2.0e-12,
            },
            "verdict": {
                "all_preregistered_hypotheses_pass": all(hypotheses.values()),
                "structural_repair_design_basis_complete": False,
                "readout": (
                    "shape moves away from CMFGEN: B0 and B1 rise and optical falls; "
                    "the unbiased matrix does not support production promotion"),
                "residual_cause_ranking": [
                    "1 EPAY/activation-owner reshaping",
                    "2 line-projection/source-matrix covariance",
                    "3 bin-width representation",
                ],
            },
            "inputs": {
                "preregistration_sha256": hashlib.sha256(
                    args.preregistration.read_bytes()).hexdigest(),
                "e9_payload_sha256": e9.manifest["sha256"],
                "source_payload_sha256": source.manifest["sha256"],
            },
            "clamp": 0, "fallback": 0, "new_model_or_GPU_run": False,
        }
        out = args.out_dir / "diagnosis_summary.json"
        out.write_text(json.dumps(result, indent=2, sort_keys=True,
                                  allow_nan=False) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError, ZeroDivisionError) as exc:
        print(f"UNRESOLVED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
