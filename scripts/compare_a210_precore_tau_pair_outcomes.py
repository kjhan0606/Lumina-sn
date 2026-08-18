#!/usr/bin/env python3
"""Combine the two sealed A2-10 pre-core tau A/B configuration outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


class VerdictError(RuntimeError):
    pass


SCHEMA = "LUMINA_A210_PRECORE_TAU_AB_COMPARISON_V2"
OUTCOMES = {
    "BRACKET_RESTORED_GATE_PASS",
    "NO_BRACKET_PERSISTS",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_report(path: Path, expected_single_total: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise VerdictError(f"missing or unsafe pair report: {path}")
    try:
        report = json.loads(path.read_text(encoding="utf-8", errors="strict"))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise VerdictError(f"invalid pair report JSON: {path}") from exc
    if not isinstance(report, dict):
        raise VerdictError(f"pair report is not an object: {path}")
    if report.get("schema") != SCHEMA or report.get("status") != "PASS":
        raise VerdictError(f"pair report is not a V2 PASS: {path}")
    if report.get("outcome") not in OUTCOMES:
        raise VerdictError(f"invalid pair outcome: {path}")
    if report.get("exact_r6_identity") != "BIT_EXACT":
        raise VerdictError(f"pair report lacks exact/R6 identity: {path}")
    seal = report.get("sealed_pair")
    if not isinstance(seal, dict):
        raise VerdictError(f"pair report lacks sealed_pair: {path}")
    controls = seal.get("controls")
    if (not isinstance(controls, dict) or
            controls.get("input/single_total.txt") != expected_single_total):
        raise VerdictError(
            f"pair report single_total is not {expected_single_total}: {path}"
        )
    if seal.get("environment_identity") != "ONLY_PRECORE_TAU_REFRESH_DIFFERS":
        raise VerdictError(f"pair environments are not isolated: {path}")
    if (seal.get("baseline_precore_tau_refresh") != 0 or
            seal.get("candidate_precore_tau_refresh") != 1):
        raise VerdictError(f"pair seed direction is invalid: {path}")
    if report.get("physical_values_modified") is not False:
        raise VerdictError(f"pair modified physical values: {path}")
    for field in ("floor", "cap", "clamp", "jitter", "repair"):
        if report.get(field) != 0:
            raise VerdictError(f"pair {field} is nonzero: {path}")
    return report


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-total-zero-report", type=Path, required=True)
    parser.add_argument("--single-total-one-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        st0 = load_report(args.single_total_zero_report, "0")
        st1 = load_report(args.single_total_one_report, "1")
        st0_outcome = str(st0["outcome"])
        st1_outcome = str(st1["outcome"])
        consistency = (
            "CONSISTENT" if st0_outcome == st1_outcome else "CONFIG_DEPENDENT"
        )
        gate_causality = (
            "SUPPORTED"
            if st1_outcome == "BRACKET_RESTORED_GATE_PASS"
            else "NOT_SUPPORTED"
        )
        payload = {
            "schema": "LUMINA_A210_PRECORE_TAU_CROSS_CONFIG_VERDICT_V1",
            "status": "PASS",
            "gate_configuration": "single_total=1",
            "gate_causality": gate_causality,
            "cross_configuration_consistency": consistency,
            "single_total_zero_outcome": st0_outcome,
            "single_total_one_outcome": st1_outcome,
            "single_total_zero_report": str(
                args.single_total_zero_report.resolve()
            ),
            "single_total_zero_sha256": digest(
                args.single_total_zero_report
            ),
            "single_total_one_report": str(
                args.single_total_one_report.resolve()
            ),
            "single_total_one_sha256": digest(
                args.single_total_one_report
            ),
            "decision_rule": (
                "single_total=1 controls the production-gate branch; "
                "single_total=0 is cross-configuration evidence only"
            ),
            "physical_values_modified": False,
            "floor": 0,
            "cap": 0,
            "clamp": 0,
            "jitter": 0,
            "repair": 0,
        }
        atomic_write(args.report, payload)
        print(
            "PASS A210_PRECORE_TAU_CROSS_CONFIG "
            f"gate_causality={gate_causality} consistency={consistency} "
            f"st0={st0_outcome} st1={st1_outcome} repair=0"
        )
        return 0
    except (OSError, VerdictError) as exc:
        atomic_write(args.report, {
            "schema": "LUMINA_A210_PRECORE_TAU_CROSS_CONFIG_VERDICT_V1",
            "status": "FAIL",
            "error": str(exc),
        })
        print(f"FAIL A210_PRECORE_TAU_CROSS_CONFIG reason={exc}")
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
