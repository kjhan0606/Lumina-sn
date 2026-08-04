#!/usr/bin/env python3
"""Rejudge Stage 31 rung10/11 from a qualified HP oracle and KA2 results.

The judge never relaxes a threshold.  Existing round-5B solver JSON files only
retain their already-computed J-vs-binary64-oracle errors, not the raw J vector.
Those legacy error fields are therefore preserved and explicitly labelled; all
other rung11 quantities are recomputed from the stored primitive metrics.  The
new full-precision oracle qualification replaces only the two old oracle gates.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "s31_results"
DEFAULT_BINARY64 = RESULTS / "ka2_oracle_rung10.json"
SCHEMA = "s31-ka2-judge-v1"


class InputError(ValueError):
    pass


def read_json(path: pathlib.Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise InputError(f"JSON root must be an object: {path}")
    return payload


def finite_number(value: object, field: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise InputError(f"{field} is not numeric: {value!r}") from exc
    if not math.isfinite(numeric):
        raise InputError(f"{field} is not finite: {value!r}")
    return numeric


def judge_oracle(oracle: dict[str, Any]) -> dict[str, Any]:
    contract = oracle.get("contract")
    audit = oracle.get("arithmetic_audit")
    self_check = oracle.get("self_check")
    references = oracle.get("references")
    if not all(isinstance(value, dict)
               for value in (contract, audit, self_check, references)):
        raise InputError("oracle is missing contract/audit/self_check/references objects")
    assert isinstance(contract, dict)
    assert isinstance(audit, dict)
    assert isinstance(self_check, dict)
    assert isinstance(references, dict)
    required_audit = {
        "nodes_and_weights_mpmath_80d",
        "E1_kernel_assembly_mpmath_80d",
        "log_singularity_subtraction_mpmath_80d",
        "dense_operator_storage_mpmath_80d",
        "dense_operator_solve_mpmath_80d",
        "target_evaluation_mpmath_80d",
        "comparison_norm_mpmath_80d",
    }
    relative_l2 = finite_number(self_check.get("relative_l2"),
                                "oracle.self_check.relative_l2")
    reference_metadata_valid = all(
        isinstance(references.get(order), dict)
        and references[order].get("order") == int(order)
        and references[order].get("mpmath_dps") == 80
        and references[order].get("arithmetic") ==
            "mpmath mpf, 80 decimal digits end-to-end"
        and references[order].get("operator_storage") ==
            "complete block-distributed dense mpf rows"
        for order in ("2048", "4096")
    )
    checks = {
        "schema_version_exact": oracle.get("schema_version") == "s31-ka2-oracle-hp-v1",
        "acceptance_declared_unchanged": oracle.get("acceptance_unchanged") is True,
        "production_run_declares_PASS": oracle.get("status") == "PASS",
        "production_run_declares_oracle_qualified": oracle.get("oracle_qualified") is True,
        "mpmath_dps_eq_80": contract.get("mpmath_dps") == 80,
        "required_nref_exact_2048_4096": contract.get("required_nref") == [2048, 4096],
        "requested_nref_exact_2048_4096": oracle.get("requested_nref") == [2048, 4096],
        "computed_nref_contains_2048_4096": {"2048", "4096"}.issubset(references),
        "reference_metadata_is_full_80digit_dense": reference_metadata_valid,
        "all_numerical_stages_are_mpmath_80d":
            required_audit.issubset(audit) and all(audit.get(key) is True
                                                   for key in required_audit),
        "Nref_relative_l2_lt_1e-9": relative_l2 < 1.0e-9,
        "self_check_evaluated": self_check.get("evaluated") is True,
        "self_check_orders_exact_2048_4096": self_check.get("orders") == [2048, 4096],
        "self_check_threshold_exact_1e-9": self_check.get("threshold") == "1e-9",
        "self_check_declares_pass": self_check.get("pass") is True,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    return {
        "status": status,
        "checks": checks,
        "relative_l2": relative_l2,
        "threshold": "< 1e-9",
    }


def binary64_comparison(binary64: dict[str, Any], hp: dict[str, Any]) -> dict[str, Any]:
    old = finite_number(binary64.get("relative_difference"),
                        "binary64.relative_difference")
    new = finite_number(hp["relative_l2"], "hp.relative_l2")
    return {
        "quantity": "Nref=2048/4096 self-agreement relative L2",
        "note": "This is a convergence-metric comparison, not a pointwise oracle-vector delta.",
        "binary64_relative_l2": old,
        "binary64_matrix_storage": binary64.get("matrix_storage"),
        "binary64_full_arithmetic_is_80_digit":
            binary64.get("acceptance", {}).get("full_oracle_arithmetic_is_80_digit"),
        "hp80_relative_l2": new,
        "absolute_metric_difference": abs(new - old),
        "hp80_over_binary64_ratio": new / old if old else None,
        "binary64_status": binary64.get("status"),
        "hp80_status": hp["status"],
    }


def solver_identity(path: pathlib.Path, payload: dict[str, Any]) -> str:
    return str(payload.get("ka") or payload.get("rung") or path.stem)


def judge_solver(path: pathlib.Path, payload: dict[str, Any],
                 oracle_pass: bool) -> dict[str, Any]:
    levels = payload.get("levels")
    if not isinstance(levels, list) or not levels:
        return {
            "input": str(path),
            "identity": solver_identity(path, payload),
            "status": "NOT_APPLICABLE",
            "reason": "no non-empty levels array",
        }
    if not all(isinstance(level, dict) for level in levels):
        raise InputError(f"{path}: every levels entry must be an object")
    finest = levels[-1]
    assert isinstance(finest, dict)
    p_obs = finite_number(payload.get("p_obs_J"), f"{path}.p_obs_J")
    checks = {
        # The archived JSON lacks raw J.  Keep its binary64-oracle metric visible.
        "finest_J_relative_l2_le_1e-4_legacy_binary64_basis":
            finite_number(finest.get("J_oracle_relative_l2"),
                          f"{path}.finest.J_oracle_relative_l2") <= 1.0e-4,
        "finest_max_scaled_error_le_3e-4_legacy_binary64_basis":
            finite_number(finest.get("max_scaled_error"),
                          f"{path}.finest.max_scaled_error") <= 3.0e-4,
        "p_obs_in_1p7_2p3": 1.7 <= p_obs <= 2.3,
        "finest_source_residual_le_1e-10":
            finite_number(finest.get("source_residual"),
                          f"{path}.finest.source_residual") <= 1.0e-10,
        "finest_transport_residual_le_1e-4":
            finite_number(finest.get("transport_residual"),
                          f"{path}.finest.transport_residual") <= 1.0e-4,
        "finest_energy_closure_le_1e-4":
            finite_number(finest.get("energy_closure"),
                          f"{path}.finest.energy_closure") <= 1.0e-4,
        "all_converged_within_500_iterations":
            all(int(level.get("source_iterations", 501)) <= 500 for level in levels),
        "all_clamp_zero": all(int(level.get("clamp_count", -1)) == 0 for level in levels),
        "all_solution_negative_zero":
            all(int(level.get("solution_negative_excess_count", -1)) == 0
                for level in levels),
        "all_sign_uncertain_zero":
            all(int(level.get("sign_uncertain_count", -1)) == 0 for level in levels),
        "all_nonfinite_zero":
            all(int(level.get("nonfinite_count", -1)) == 0 for level in levels),
        "full_80digit_oracle_qualified": oracle_pass,
    }
    numeric_without_oracle = all(value for key, value in checks.items()
                                 if key != "full_80digit_oracle_qualified")
    return {
        "input": str(path),
        "identity": solver_identity(path, payload),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "numeric_status_without_oracle_qualification":
            "PASS" if numeric_without_oracle else "FAIL",
        "checks": checks,
        "metrics": {
            "p_obs_J": p_obs,
            "finest_J_oracle_relative_l2_legacy_binary64_basis":
                finest.get("J_oracle_relative_l2"),
            "finest_max_scaled_error_legacy_binary64_basis":
                finest.get("max_scaled_error"),
        },
        "raw_J_recomparison": "UNAVAILABLE_IN_ARCHIVED_SOLVER_JSON",
    }


def default_solver_paths() -> list[pathlib.Path]:
    paths = sorted(RESULTS.glob("*ka2*.json"))
    scattering = RESULTS / "scattering_rung11.json"
    if scattering.exists():
        paths.append(scattering)
    return paths


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", required=True, type=pathlib.Path)
    parser.add_argument("--solver", type=pathlib.Path, nargs="*",
                        help="default: KA2/scattering JSON under docs/s31_results")
    parser.add_argument("--binary64-oracle", type=pathlib.Path,
                        default=DEFAULT_BINARY64)
    parser.add_argument("--out", type=pathlib.Path,
                        help="optional judgment JSON; report is always printed")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    try:
        oracle_payload = read_json(args.oracle)
        oracle_judgment = judge_oracle(oracle_payload)
        binary64_payload = read_json(args.binary64_oracle)
        paths = args.solver if args.solver else default_solver_paths()
        # Do not mistake oracle inputs for production-solver results.
        paths = [path for path in paths
                 if "oracle" not in path.name.lower() and path.resolve() != args.oracle.resolve()]
        if not paths:
            raise InputError("no solver result JSON inputs")
        solver_judgments = [judge_solver(path, read_json(path),
                                         oracle_judgment["status"] == "PASS")
                            for path in paths]
        applicable = [item for item in solver_judgments
                      if item["status"] != "NOT_APPLICABLE"]
        report = {
            "schema_version": SCHEMA,
            "acceptance_unchanged": True,
            "oracle_input": str(args.oracle),
            "rung10_oracle": oracle_judgment,
            "binary64_oracle_comparison":
                binary64_comparison(binary64_payload, oracle_judgment),
            "rung11_solver_results": solver_judgments,
            "overall_status": ("PASS" if applicable
                               and oracle_judgment["status"] == "PASS"
                               and all(item["status"] == "PASS" for item in applicable)
                               else "FAIL"),
            "limitations": [
                "Archived solver JSON stores J error scalars but not raw J vectors.",
                "Those two J-error gates retain their binary64-oracle basis; the judge does not invent a pointwise HP delta.",
            ],
        }
    except InputError as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.out.with_suffix(args.out.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.out)
    print(rendered, end="")
    return 0 if report["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
