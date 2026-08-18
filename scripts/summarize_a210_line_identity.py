#!/usr/bin/env python3
"""Validate and summarize A2-10 three-way line-identity diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path


PREFIX = "[A2-10][LINE-COEFFICIENT-IDENTITY] "
ENDPOINT_PREFIX = "[A2-10][ENDPOINT-FINITE] "
INTERIOR_PREFIX = "[A2-10][VECTOR-INTERIOR-SCAN] "
TOKEN = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")
FLOAT_FIELDS = (
    "sobolev_signed_rate",
    "exact_constant_rate",
    "einstein_consistent_rate",
    "constant_delta",
    "serialization_delta",
    "delta",
    "scaled_emission",
    "scaled_absorption",
    "cancellation_condition",
    "positive_tau_delta",
    "negative_tau_delta",
    "constant_positive_tau_delta",
    "constant_negative_tau_delta",
    "serialization_positive_tau_delta",
    "serialization_negative_tau_delta",
)
INTEGER_FIELDS = ("shell", "raw_cells", "srce_chk_cells", "repair")
ENDPOINT_FLOAT_FIELDS = ("T_e_K", "heating", "cooling", "residual")
INTERIOR_FLOAT_FIELDS = (
    "T_mid",
    "res_lo",
    "res_mid",
    "res_hi",
    "heat_mid",
    "cool_mid",
    "line_emit_mid",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def subtraction_close(minuend: float, subtrahend: float, result: float) -> bool:
    """Binary64 roundoff bound for a reported subtraction identity.

    The bound scales with the operands, not the possibly tiny residual.  This
    is an arithmetic serialization check, never a physical sign tolerance.
    """
    computed = minuend - subtrahend
    magnitude = abs(minuend) + abs(subtrahend) + abs(result)
    scale = max(abs(minuend), abs(subtrahend), abs(result), sys.float_info.min)
    bound = 64.0 * sys.float_info.epsilon * magnitude + 8.0 * math.ulp(scale)
    return abs(computed - result) <= bound


def partition_close(first: float, second: float, total: float) -> bool:
    computed = first + second
    magnitude = abs(first) + abs(second) + abs(total)
    scale = max(abs(first), abs(second), abs(total), sys.float_info.min)
    bound = 64.0 * sys.float_info.epsilon * magnitude + 8.0 * math.ulp(scale)
    return abs(computed - total) <= bound


def parse_record(line: str, line_number: int) -> dict[str, object]:
    tokens = dict(TOKEN.findall(line[len(PREFIX):]))
    required = {"phase", "interpretation", *FLOAT_FIELDS, *INTEGER_FIELDS}
    missing = required.difference(tokens)
    if missing:
        raise ValueError(
            f"line {line_number}: missing diagnostic fields {sorted(missing)}"
        )
    record: dict[str, object] = {
        "phase": tokens["phase"],
        "interpretation": tokens["interpretation"],
    }
    try:
        for name in FLOAT_FIELDS:
            record[name] = float(tokens[name])
        for name in INTEGER_FIELDS:
            record[name] = int(tokens[name])
    except ValueError as exc:
        raise ValueError(f"line {line_number}: invalid numeric field") from exc
    if any(not math.isfinite(float(record[name])) for name in FLOAT_FIELDS):
        raise ValueError(f"line {line_number}: nonfinite diagnostic value")
    if record["interpretation"] != "DIAGNOSTIC_ONLY" or record["repair"] != 0:
        raise ValueError(f"line {line_number}: mutation/repair contract violated")

    current = float(record["sobolev_signed_rate"])
    exact = float(record["exact_constant_rate"])
    einstein = float(record["einstein_consistent_rate"])
    constant_delta = float(record["constant_delta"])
    serialization_delta = float(record["serialization_delta"])
    total_delta = float(record["delta"])
    if not subtraction_close(exact, current, constant_delta):
        raise ValueError(f"line {line_number}: constant delta identity failed")
    if not subtraction_close(einstein, exact, serialization_delta):
        raise ValueError(f"line {line_number}: serialization delta identity failed")
    if not subtraction_close(einstein, current, total_delta):
        raise ValueError(f"line {line_number}: total delta identity failed")

    positive = float(record["positive_tau_delta"])
    negative = float(record["negative_tau_delta"])
    constant_positive = float(record["constant_positive_tau_delta"])
    constant_negative = float(record["constant_negative_tau_delta"])
    serialization_positive = float(record["serialization_positive_tau_delta"])
    serialization_negative = float(record["serialization_negative_tau_delta"])
    # The positive/negative arrays accumulate per-cell deltas in long double,
    # whereas the three shell rates are independently accumulated and cast to
    # binary64 before their reported subtraction.  Their tiny closure is thus
    # an observed summation-path difference, not a two-operand subtraction
    # identity and not a physical tolerance.  Validate the exact coefficient
    # decomposition within each tau-sign partition, then report all three
    # cross-path closures without hiding or repairing them.
    if not partition_close(
        constant_positive, serialization_positive, positive
    ):
        raise ValueError(f"line {line_number}: positive-tau decomposition failed")
    if not partition_close(
        constant_negative, serialization_negative, negative
    ):
        raise ValueError(f"line {line_number}: negative-tau decomposition failed")
    record["tau_partition_closure"] = math.fsum(
        (positive, negative, -total_delta)
    )
    record["constant_tau_partition_closure"] = math.fsum(
        (constant_positive, constant_negative, -constant_delta)
    )
    record["serialization_tau_partition_closure"] = math.fsum(
        (serialization_positive, serialization_negative, -serialization_delta)
    )
    return record


def parse_endpoint_record(line: str, line_number: int) -> dict[str, object]:
    tokens = dict(TOKEN.findall(line[len(ENDPOINT_PREFIX):]))
    required = {"phase", "shell", *ENDPOINT_FLOAT_FIELDS}
    missing = required.difference(tokens)
    if missing:
        raise ValueError(
            f"line {line_number}: missing endpoint fields {sorted(missing)}"
        )
    try:
        record: dict[str, object] = {
            "phase": tokens["phase"],
            "shell": int(tokens["shell"]),
        }
        for name in ENDPOINT_FLOAT_FIELDS:
            record[name] = float(tokens[name])
    except ValueError as exc:
        raise ValueError(f"line {line_number}: invalid endpoint field") from exc
    if any(
        not math.isfinite(float(record[name])) for name in ENDPOINT_FLOAT_FIELDS
    ):
        raise ValueError(f"line {line_number}: nonfinite endpoint value")
    if not subtraction_close(
        float(record["heating"]),
        float(record["cooling"]),
        float(record["residual"]),
    ):
        raise ValueError(f"line {line_number}: endpoint residual identity failed")
    return record


def parse_interior_record(line: str, line_number: int) -> dict[str, object]:
    tokens = dict(TOKEN.findall(line[len(INTERIOR_PREFIX):]))
    if tokens.get("action") != "DIAGNOSTIC_ONLY":
        raise ValueError(
            f"line {line_number}: interior scan is not diagnostic-only"
        )
    phase = tokens.get("phase")
    if not phase:
        raise ValueError(f"line {line_number}: interior scan has no phase")
    try:
        valid = int(tokens["valid"]) if "valid" in tokens else None
    except ValueError as exc:
        raise ValueError(f"line {line_number}: invalid interior valid flag") from exc
    if valid == 0:
        return {
            "kind": "invalid",
            "phase": phase,
            "status": tokens.get("status"),
            "reason": tokens.get("reason"),
            "valid": 0,
            "action": "DIAGNOSTIC_ONLY",
        }
    if "shell" in tokens:
        required = {
            "shell",
            *INTERIOR_FLOAT_FIELDS,
            "lo_mid_bracket",
            "mid_hi_bracket",
        }
        missing = required.difference(tokens)
        if missing:
            raise ValueError(
                f"line {line_number}: missing interior shell fields "
                f"{sorted(missing)}"
            )
        try:
            record: dict[str, object] = {
                "kind": "shell",
                "phase": phase,
                "shell": int(tokens["shell"]),
                "lo_mid_bracket": int(tokens["lo_mid_bracket"]),
                "mid_hi_bracket": int(tokens["mid_hi_bracket"]),
                "action": "DIAGNOSTIC_ONLY",
            }
            for name in INTERIOR_FLOAT_FIELDS:
                record[name] = float(tokens[name])
        except ValueError as exc:
            raise ValueError(f"line {line_number}: invalid interior shell field") from exc
        if any(
            not math.isfinite(float(record[name]))
            for name in INTERIOR_FLOAT_FIELDS
        ):
            raise ValueError(f"line {line_number}: nonfinite interior shell value")
        if record["lo_mid_bracket"] not in (0, 1) or record[
            "mid_hi_bracket"
        ] not in (0, 1):
            raise ValueError(f"line {line_number}: invalid interior bracket flag")
        if not subtraction_close(
            float(record["heat_mid"]),
            float(record["cool_mid"]),
            float(record["res_mid"]),
        ):
            raise ValueError(f"line {line_number}: interior residual identity failed")
        return record
    required = {
        "valid",
        "endpoint_no_bracket",
        "interior_bracket",
        "still_same_sign",
        "solver_result",
    }
    missing = required.difference(tokens)
    if missing:
        raise ValueError(
            f"line {line_number}: missing interior summary fields {sorted(missing)}"
        )
    try:
        summary = {
            "kind": "summary",
            "phase": phase,
            "valid": int(tokens["valid"]),
            "endpoint_no_bracket": int(tokens["endpoint_no_bracket"]),
            "interior_bracket": int(tokens["interior_bracket"]),
            "still_same_sign": int(tokens["still_same_sign"]),
            "solver_result": tokens["solver_result"],
            "action": "DIAGNOSTIC_ONLY",
        }
    except ValueError as exc:
        raise ValueError(f"line {line_number}: invalid interior summary field") from exc
    if summary["valid"] != 1 or summary["solver_result"] != "RADEQ_NO_BRACKET":
        raise ValueError(f"line {line_number}: invalid interior summary contract")
    if (
        summary["interior_bracket"] + summary["still_same_sign"]
        != summary["endpoint_no_bracket"]
    ):
        raise ValueError(f"line {line_number}: interior summary count mismatch")
    return summary


def finish_interior_scan(
    record: dict[str, object],
    pending: dict[str, list[dict[str, object]]],
    scans: list[dict[str, object]],
    line_number: int,
) -> None:
    phase = str(record["phase"])
    rows = pending.pop(phase, [])
    if record["kind"] == "invalid":
        if rows:
            raise ValueError(
                f"line {line_number}: invalid interior scan has shell rows"
            )
        scans.append(record)
        return
    expected = int(record["endpoint_no_bracket"])
    if len(rows) != expected:
        raise ValueError(
            f"line {line_number}: interior phase {phase} has {len(rows)} shell "
            f"rows, expected {expected}"
        )
    recovered = sum(
        int(row["lo_mid_bracket"] or row["mid_hi_bracket"])
        for row in rows
    )
    if recovered != int(record["interior_bracket"]):
        raise ValueError(
            f"line {line_number}: interior phase {phase} bracket count mismatch"
        )
    scans.append({**record, "shells": rows})


def sign(value: float) -> str:
    if value > 0.0:
        return "positive"
    if value < 0.0:
        return "negative"
    return "zero"


def append_ordered_record(
    batches: dict[str, list[list[dict[str, object]]]],
    record: dict[str, object],
    line_number: int,
    kind: str,
) -> None:
    phase = str(record["phase"])
    shell = int(record["shell"])
    phase_batches = batches.setdefault(phase, [])
    if shell == 0:
        phase_batches.append([])
    if not phase_batches:
        raise ValueError(
            f"line {line_number}: {kind} phase {phase} starts at shell {shell}"
        )
    batch = phase_batches[-1]
    if shell != len(batch):
        raise ValueError(
            f"line {line_number}: {kind} phase {phase} expected shell "
            f"{len(batch)}, got {shell}"
        )
    batch.append(record)


def validate_complete_batches(
    batches: dict[str, list[list[dict[str, object]]]],
    expected_shells: int,
    kind: str,
) -> None:
    for phase, phase_batches in batches.items():
        for batch_index, batch in enumerate(phase_batches):
            if len(batch) != expected_shells:
                raise ValueError(
                    f"{kind} phase {phase} batch {batch_index}: "
                    f"{len(batch)} records, expected {expected_shells}"
                )


def has_bracket(lower: float, upper: float) -> bool:
    return lower == 0.0 or upper == 0.0 or (lower < 0.0) != (upper < 0.0)


def endpoint_counterfactual(
    identity_batches: dict[str, list[list[dict[str, object]]]],
    endpoint_batches: dict[str, list[list[dict[str, object]]]],
    expected_shells: int,
) -> dict[str, object]:
    for phase in ("LOWER", "UPPER"):
        if len(endpoint_batches.get(phase, [])) != 1:
            raise ValueError(
                f"exactly one complete {phase} endpoint batch is required"
            )
        if len(identity_batches.get(phase, [])) != 1:
            raise ValueError(
                f"exactly one complete {phase} line-identity batch is required"
            )

    phase_shells: dict[str, list[dict[str, object]]] = {}
    for phase in ("LOWER", "UPPER"):
        identity = identity_batches[phase][0]
        endpoints = endpoint_batches[phase][0]
        phase_shells[phase] = []
        for shell in range(expected_shells):
            line = identity[shell]
            endpoint = endpoints[shell]
            current = float(endpoint["residual"])
            exact = current - float(line["constant_delta"])
            einstein = current - float(line["delta"])
            if not math.isfinite(exact) or not math.isfinite(einstein):
                raise ValueError(
                    f"phase {phase} shell {shell}: nonfinite counterfactual"
                )
            phase_shells[phase].append(
                {
                    "shell": shell,
                    "T_e_K": endpoint["T_e_K"],
                    "current_residual": current,
                    "exact_constant_residual": exact,
                    "einstein_consistent_residual": einstein,
                    "constant_delta": line["constant_delta"],
                    "serialization_delta": line["serialization_delta"],
                    "total_delta": line["delta"],
                }
            )

    bracket_counts = {name: 0 for name in ("current", "exact_constant", "einstein")}
    recovered = {name: [] for name in ("exact_constant", "einstein")}
    lost = {name: [] for name in ("exact_constant", "einstein")}
    shells: list[dict[str, object]] = []
    for shell in range(expected_shells):
        lower = phase_shells["LOWER"][shell]
        upper = phase_shells["UPPER"][shell]
        flags = {
            "current": has_bracket(
                float(lower["current_residual"]),
                float(upper["current_residual"]),
            ),
            "exact_constant": has_bracket(
                float(lower["exact_constant_residual"]),
                float(upper["exact_constant_residual"]),
            ),
            "einstein": has_bracket(
                float(lower["einstein_consistent_residual"]),
                float(upper["einstein_consistent_residual"]),
            ),
        }
        for name, flag in flags.items():
            bracket_counts[name] += int(flag)
        for name in ("exact_constant", "einstein"):
            if flags[name] and not flags["current"]:
                recovered[name].append(shell)
            if flags["current"] and not flags[name]:
                lost[name].append(shell)
        shells.append(
            {
                "shell": shell,
                "lower": lower,
                "upper": upper,
                "bracket": flags,
            }
        )

    return {
        "interpretation": "DIAGNOSTIC_COUNTERFACTUAL_ONLY",
        "residual_transform": "new_residual=current_residual-(new_line_rate-current_line_rate)",
        "physical_mutation": 0,
        "repair": 0,
        "bracket_counts": bracket_counts,
        "recovered_shells": recovered,
        "lost_shells": lost,
        "phases": phase_shells,
        "shells": shells,
        "verdict": "COMPLETE",
    }


def summarize(
    stderr_path: Path,
    expected_shells: int,
    require_endpoints: bool = False,
) -> dict[str, object]:
    if expected_shells <= 0:
        raise ValueError("expected_shells must be positive")
    batches: dict[str, list[list[dict[str, object]]]] = {}
    endpoint_batches: dict[str, list[list[dict[str, object]]]] = {}
    interior_pending: dict[str, list[dict[str, object]]] = {}
    interior_scans: list[dict[str, object]] = []
    records = 0
    endpoint_records = 0
    with stderr_path.open("r", encoding="utf-8", errors="replace") as stream:
        for line_number, raw in enumerate(stream, start=1):
            line = raw.rstrip("\n")
            if line.startswith(PREFIX):
                record = parse_record(line, line_number)
                append_ordered_record(batches, record, line_number, "identity")
                records += 1
            elif line.startswith(ENDPOINT_PREFIX):
                endpoint = parse_endpoint_record(line, line_number)
                append_ordered_record(
                    endpoint_batches, endpoint, line_number, "endpoint"
                )
                endpoint_records += 1
            elif line.startswith(INTERIOR_PREFIX):
                interior = parse_interior_record(line, line_number)
                if interior["kind"] == "shell":
                    interior_pending.setdefault(
                        str(interior["phase"]), []
                    ).append(interior)
                else:
                    finish_interior_scan(
                        interior, interior_pending, interior_scans, line_number
                    )

    if records == 0:
        raise ValueError("no line-identity records found")
    validate_complete_batches(batches, expected_shells, "identity")
    if len(batches.get("LOWER", [])) != 1 or len(batches.get("UPPER", [])) != 1:
        raise ValueError("exactly one complete LOWER and UPPER batch is required")
    if endpoint_records:
        validate_complete_batches(endpoint_batches, expected_shells, "endpoint")
    elif require_endpoints:
        raise ValueError("no endpoint records found")
    if interior_pending:
        raise ValueError(
            f"unterminated interior scans: {sorted(interior_pending)}"
        )
    valid_interior_scans = [
        scan for scan in interior_scans if scan["valid"] == 1
    ]
    interior_identity_batches = batches.get("INTERIOR", [])
    if len(valid_interior_scans) != len(interior_identity_batches):
        raise ValueError(
            "valid interior scan count does not match INTERIOR identity batches"
        )

    rendered_batches: dict[str, list[dict[str, object]]] = {}
    for phase, phase_batches in batches.items():
        rendered_batches[phase] = []
        for batch_index, batch in enumerate(phase_batches):
            counts = {
                owner: {"positive": 0, "negative": 0, "zero": 0}
                for owner in ("current", "exact_constant", "einstein")
            }
            for record in batch:
                counts["current"][sign(float(record["sobolev_signed_rate"]))] += 1
                counts["exact_constant"][sign(float(record["exact_constant_rate"]))] += 1
                counts["einstein"][sign(float(record["einstein_consistent_rate"]))] += 1
            rendered_batches[phase].append(
                {
                    "batch_index": batch_index,
                    "sign_counts": counts,
                    "shells": batch,
                }
            )

    report: dict[str, object] = {
        "schema": "lumina-a210-line-identity-summary-v1",
        "stderr": str(stderr_path.resolve()),
        "stderr_sha256": sha256(stderr_path),
        "expected_shells": expected_shells,
        "records": records,
        "phase_batch_counts": {
            phase: len(phase_batches) for phase, phase_batches in batches.items()
        },
        "physical_mutation": 0,
        "repair": 0,
        "arithmetic_identity": (
            "PASS_RATE_SUBTRACTIONS_AND_TAU_SIGN_DECOMPOSITION; "
            "CROSS_ACCUMULATION_CLOSURES_REPORTED"
        ),
        "batches": rendered_batches,
        "interior_scans": interior_scans,
        "interior_identity_phase_map": [
            {"batch_index": index, "phase": scan["phase"]}
            for index, scan in enumerate(valid_interior_scans)
        ],
        "verdict": "COMPLETE",
    }
    if endpoint_records:
        report["endpoint_records"] = endpoint_records
        report["endpoint_phase_batch_counts"] = {
            phase: len(phase_batches)
            for phase, phase_batches in endpoint_batches.items()
        }
        report["endpoint_counterfactual"] = endpoint_counterfactual(
            batches, endpoint_batches, expected_shells
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stderr", type=Path)
    parser.add_argument("--expected-shells", type=int, default=50)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--require-endpoints", action="store_true")
    args = parser.parse_args()
    try:
        report = summarize(
            args.stderr, args.expected_shells, args.require_endpoints
        )
    except (OSError, ValueError) as exc:
        print(f"[a210-line-identity-summary] ERROR: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
