#!/usr/bin/env python3
"""Fail-closed audit of the deterministic local-Jbar error envelope.

This checker is deliberately separate from the material-convergence checker.
It certifies that every requested DET outer iteration used the positive formal
operator, a verified componentwise envelope, a profile-local projection of that
envelope, and complete canonical Q_E coverage.  It never interprets the old
global fixed-point bound as a line-cell uncertainty.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path


EXACT_PREFIX = "[cmf_fine][EXACT-POSITIVE-SLIDING]"
IDENTITY_PREFIX = "[R6][LINE-IDENTITY]"
COVERAGE_PREFIX = "[R6][LINE-COVERAGE]"
TARGET_PREFIX = "[A2-10][LINE-NET-CELL-FINITE]"
TARGET_CELLS = ((15, 4), (17794, 26))
FORBIDDEN = (
    "[cmf_fine][BLOCKED]",
    "[R6][BLOCKED]",
    "[A2-10][LINE-NET-CELL-BLOCKED]",
    "[A2-10][LINE-NET-BLOCKED]",
)


class AuditError(RuntimeError):
    pass


def field(line: str, name: str) -> str:
    match = re.search(rf"(?:^| ){re.escape(name)}=([^ ]+)", line)
    if not match:
        raise AuditError(f"missing {name} in: {line[:240]}")
    return match.group(1)


def integer(line: str, name: str) -> int:
    value = field(line, name)
    if not re.fullmatch(r"[0-9]+", value):
        raise AuditError(f"invalid integer {name}={value!r}")
    return int(value)


def finite(line: str, name: str) -> float:
    value = field(line, name)
    try:
        result = float(value)
    except ValueError as exc:
        raise AuditError(f"invalid float {name}={value!r}") from exc
    if not math.isfinite(result):
        raise AuditError(f"non-finite {name}={value!r}")
    return result


def interval(line: str, name: str) -> tuple[float, float]:
    value = field(line, name)
    match = re.fullmatch(r"\[([^,]+),([^]]+)\]", value)
    if not match:
        raise AuditError(f"invalid interval {name}={value!r}")
    try:
        lower, upper = float(match.group(1)), float(match.group(2))
    except ValueError as exc:
        raise AuditError(f"invalid interval {name}={value!r}") from exc
    if not (math.isfinite(lower) and math.isfinite(upper) and
            0.0 <= lower <= upper):
        raise AuditError(f"unqualified interval {name}={value!r}")
    return lower, upper


def audit(log_path: Path, expected_iterations: int,
          expected_refinements: int,
          require_target_cells: bool = False) -> dict[str, object]:
    if not log_path.is_file() or log_path.is_symlink():
        raise AuditError(f"missing or non-regular stderr log: {log_path}")
    lines = log_path.read_text(encoding="utf-8", errors="strict").splitlines()
    forbidden = [line for line in lines if any(token in line for token in FORBIDDEN)]
    if forbidden:
        raise AuditError(f"blocked marker present: {forbidden[0][:300]}")

    exact = [line for line in lines if EXACT_PREFIX in line]
    identity = [line for line in lines if IDENTITY_PREFIX in line]
    coverage = [line for line in lines if COVERAGE_PREFIX in line]
    counts = (len(exact), len(identity), len(coverage))
    if counts != (expected_iterations,) * 3:
        raise AuditError(
            "iteration evidence count mismatch: "
            f"exact/identity/coverage={counts}, expected={expected_iterations}"
        )

    component_min = math.inf
    component_max = 0.0
    profile_min = math.inf
    profile_max = 0.0
    generations: list[int] = []
    for iteration, (solve, ident, cover) in enumerate(zip(exact, identity, coverage)):
        if field(solve, "status") != "OK":
            raise AuditError(f"iteration {iteration}: exact solve not OK")
        if integer(solve, "component_envelope") != 1:
            raise AuditError(f"iteration {iteration}: component envelope unverified")
        if integer(solve, "refinements") != expected_refinements:
            raise AuditError(f"iteration {iteration}: refinement count mismatch")
        residual = finite(solve, "residual")
        tolerance = finite(solve, "tolerance")
        if not (residual >= 0.0 and tolerance > 0.0 and residual < tolerance):
            raise AuditError(f"iteration {iteration}: exact residual not qualified")
        if integer(solve, "negative_recurrence") != 0:
            raise AuditError(f"iteration {iteration}: negative recurrence observed")
        solve_component = interval(solve, "component_error")

        if field(ident, "lane") != "DET" or \
                field(ident, "statistic_kind") != "DETERMINISTIC":
            raise AuditError(f"iteration {iteration}: wrong R6 identity lane")
        if integer(ident, "component_envelope") != 1 or \
                integer(ident, "refinements") != expected_refinements:
            raise AuditError(f"iteration {iteration}: R6 envelope unqualified")
        ident_component = interval(ident, "component_error")
        ident_profile = interval(ident, "profile_error")
        if solve_component != ident_component:
            raise AuditError(f"iteration {iteration}: component evidence changed at R6")

        generation = integer(cover, "generation")
        generations.append(generation)
        expected_coverage = {
            "all_lines": 2_588_798,
            "q_lines": 1_603_732,
            "e_lines": 2_180_286,
            "valid_lines": 2_180_286,
            "partial_lines": 0,
            "unsampled_lines": 0,
            "valid_cells": 109_014_300,
        }
        for name, expected in expected_coverage.items():
            observed = integer(cover, name)
            if observed != expected:
                raise AuditError(
                    f"iteration {iteration}: {name}={observed}, expected={expected}"
                )

        component_min = min(component_min, solve_component[0])
        component_max = max(component_max, solve_component[1])
        profile_min = min(profile_min, ident_profile[0])
        profile_max = max(profile_max, ident_profile[1])

    if generations != list(range(1, expected_iterations + 1)):
        raise AuditError(f"non-canonical radiation generations: {generations}")
    target_summary: dict[str, dict[str, float | int]] = {}
    if require_target_cells:
        target_lines = [line for line in lines if TARGET_PREFIX in line]
        expected_keys = {
            (phase, line, shell)
            for phase in ("LOWER", "UPPER")
            for line, shell in TARGET_CELLS
        }
        observed: dict[tuple[str, int, int], list[str]] = {
            key: [] for key in expected_keys
        }
        for witness in target_lines:
            key = (field(witness, "phase"), integer(witness, "line"),
                   integer(witness, "shell"))
            if key not in observed:
                raise AuditError(f"unexpected target-cell witness: {key}")
            observed[key].append(witness)
        for key in sorted(expected_keys):
            witnesses = observed[key]
            if len(witnesses) != expected_iterations:
                raise AuditError(
                    f"target-cell witness count {key}={len(witnesses)}, "
                    f"expected={expected_iterations}"
                )
            max_uncertainty_ratio = 0.0
            for witness in witnesses:
                jbar = finite(witness, "Jbar")
                bound = finite(witness, "Jbar_local_bound")
                signed_rate = finite(witness, "signed_rate")
                uncertainty = finite(witness, "uncertainty")
                for name in ("T_e_K", "tau_raw", "n_upper", "A_ul", "nu",
                             "chi_raw", "chi_effective", "eta_per_sr",
                             "absorption_per_sr", "net_per_sr",
                             "cancellation_condition"):
                    finite(witness, name)
                if jbar < 0.0 or bound < 0.0 or uncertainty < 0.0:
                    raise AuditError(f"negative target-cell measure at {key}")
                status = field(witness, "status")
                if status not in {"OK_COOLING", "OK_HEATING"}:
                    raise AuditError(f"unqualified target-cell status {key}={status}")
                if signed_rate == 0.0 or abs(signed_rate) <= uncertainty:
                    raise AuditError(f"target-cell sign not resolved at {key}")
                if any(integer(witness, name) != 0
                       for name in ("clamp", "floor", "jitter")):
                    raise AuditError(f"numerical repair observed at {key}")
                max_uncertainty_ratio = max(
                    max_uncertainty_ratio, uncertainty / abs(signed_rate))
            label = f"{key[0]}:line{key[1]}:shell{key[2]}"
            target_summary[label] = {
                "count": len(witnesses),
                "max_uncertainty_to_abs_rate": max_uncertainty_ratio,
            }
    return {
        "schema": "LUMINA_DET_LOCAL_JBAR_AUDIT_V1",
        "status": "PASS",
        "stderr_log": str(log_path.resolve()),
        "iterations": expected_iterations,
        "refinements": expected_refinements,
        "generations": generations,
        "component_error_min": component_min,
        "component_error_max": component_max,
        "profile_error_min": profile_min,
        "profile_error_max": profile_max,
        "valid_cells_per_iteration": 109_014_300,
        "global_bound_used_for_cell_qualification": False,
        "target_cells_required": require_target_cells,
        "target_cells": target_summary,
        "floor": 0,
        "clamp": 0,
        "jitter": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stderr", required=True, type=Path)
    parser.add_argument("--expected-iterations", required=True, type=int)
    parser.add_argument("--expected-refinements", default=8, type=int)
    parser.add_argument("--require-target-cells", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.expected_iterations <= 0 or args.expected_refinements <= 0:
        parser.error("expected counts must be positive")
    return args


def main() -> int:
    args = parse_args()
    try:
        report = audit(args.stderr, args.expected_iterations,
                       args.expected_refinements,
                       args.require_target_cells)
    except (AuditError, OSError, UnicodeError) as exc:
        print(f"DET_LOCAL_JBAR_AUDIT FAIL {exc}", file=sys.stderr)
        return 1
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    print("DET_LOCAL_JBAR_AUDIT PASS "
          f"iterations={report['iterations']} "
          f"component_error=[{report['component_error_min']:.17g},"
          f"{report['component_error_max']:.17g}] "
          f"profile_error=[{report['profile_error_min']:.17g},"
          f"{report['profile_error_max']:.17g}] "
          f"valid_cells={report['valid_cells_per_iteration']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
