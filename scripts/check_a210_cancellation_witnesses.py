#!/usr/bin/env python3
"""Audit A2-10 cancellation witnesses without changing physical values.

The input is a Lumina stderr log containing
``[A2-10][LINE-NET-CELL-BLOCKED]`` records.  Decimal arithmetic reconstructs
the line-net identity, its propagated absolute uncertainty, and the Jbar
error bound required to prove the recorded sign.  This is an offline judge;
it does not provide a tolerance or repair path to the producer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any


MARKER = "[A2-10][LINE-NET-CELL-BLOCKED]"
FOUR_PI = Decimal("12.56637061435917295385057353311801153679")
REQUIRED_FIELDS = (
    "status",
    "line",
    "shell",
    "eta_per_sr",
    "chi_effective",
    "Jbar",
    "Jbar_bound",
    "deck_scale",
    "absorption_per_sr",
    "net_per_sr",
    "signed_rate",
    "uncertainty",
    "cancellation_condition",
)
PAIR_RE = re.compile(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=([^\s]+)")


class AuditError(RuntimeError):
    """Input or identity failure that invalidates the witness audit."""


def decimal_field(fields: dict[str, str], name: str) -> Decimal:
    try:
        value = Decimal(fields[name])
    except (KeyError, ArithmeticError) as exc:
        raise AuditError(f"invalid or missing decimal field {name}") from exc
    if not value.is_finite():
        raise AuditError(f"non-finite field {name}={value}")
    return value


def relative_error(got: Decimal, expected: Decimal) -> Decimal:
    difference = abs(got - expected)
    if expected == 0:
        return Decimal(0) if difference == 0 else Decimal("Infinity")
    return difference / abs(expected)


def binary64_field(fields: dict[str, str], name: str) -> float:
    """Parse one producer value back to the binary64 printed by %.17g."""
    try:
        value = float(fields[name])
    except (KeyError, ValueError, OverflowError) as exc:
        raise AuditError(f"invalid or missing binary64 field {name}") from exc
    if not math.isfinite(value):
        raise AuditError(f"non-finite field {name}={value}")
    return value


def exact_decimal(value: float) -> Decimal:
    """Represent a finite binary64 exactly for proof-ratio arithmetic."""
    if not math.isfinite(value):
        raise AuditError(f"cannot convert non-finite binary64 value={value}")
    return Decimal.from_float(value)


def audit_record(fields: dict[str, str], maximum_relative_error: Decimal) -> dict[str, Any]:
    missing = [name for name in REQUIRED_FIELDS if name not in fields]
    if missing:
        raise AuditError(f"missing fields: {','.join(missing)}")
    if fields["status"] != "UNRESOLVED_CANCELLATION":
        raise AuditError(f"unexpected status={fields['status']}")

    with localcontext() as context:
        context.prec = 100
        # The producer contract is binary64 and deliberately uses one fused
        # eta-chi*Jbar rounding.  Reconstructing the %.17g strings with exact
        # decimal multiply/subtract is a different arithmetic program and can
        # falsely reject the most strongly cancelled cells.
        eta_f = binary64_field(fields, "eta_per_sr")
        chi_f = binary64_field(fields, "chi_effective")
        jbar_f = binary64_field(fields, "Jbar")
        jbar_bound_f = binary64_field(fields, "Jbar_bound")
        deck_scale_f = binary64_field(fields, "deck_scale")
        logged_absorption_f = binary64_field(fields, "absorption_per_sr")
        logged_net_f = binary64_field(fields, "net_per_sr")
        logged_rate_f = binary64_field(fields, "signed_rate")
        logged_uncertainty_f = binary64_field(fields, "uncertainty")
        logged_condition_f = binary64_field(fields, "cancellation_condition")

        if (eta_f < 0.0 or chi_f == 0.0 or jbar_f < 0.0 or
                jbar_bound_f < 0.0 or deck_scale_f <= 0.0):
            raise AuditError("witness violates finite material/radiation preconditions")

        reconstructed_absorption_f = chi_f * jbar_f
        reconstructed_net_f = math.fma(-chi_f, jbar_f, eta_f)
        factor_f = float(FOUR_PI) * deck_scale_f
        reconstructed_rate_f = reconstructed_net_f * factor_f
        uncertainty_per_sr_f = math.fma(abs(chi_f), jbar_bound_f, 0.0)
        reconstructed_uncertainty_f = uncertainty_per_sr_f * factor_f
        component_magnitude_f = abs(eta_f) + abs(reconstructed_absorption_f)
        if reconstructed_net_f == 0.0:
            raise AuditError("non-proven exact cancellation has no finite required bound")
        reconstructed_condition_f = (
            component_magnitude_f / abs(reconstructed_net_f)
        )

        eta = exact_decimal(eta_f)
        chi = exact_decimal(chi_f)
        jbar = exact_decimal(jbar_f)
        jbar_bound = exact_decimal(jbar_bound_f)
        deck_scale = exact_decimal(deck_scale_f)
        reconstructed_absorption = exact_decimal(reconstructed_absorption_f)
        reconstructed_net = exact_decimal(reconstructed_net_f)
        reconstructed_rate = exact_decimal(reconstructed_rate_f)
        reconstructed_uncertainty = exact_decimal(reconstructed_uncertainty_f)
        reconstructed_condition = exact_decimal(reconstructed_condition_f)
        required_symmetric_jbar_bound = abs(reconstructed_net) / abs(chi)
        current_to_required = jbar_bound / required_symmetric_jbar_bound
        one_sided_jbar_threshold = eta / chi if chi > 0 else None
        rate_margin = abs(reconstructed_rate) / reconstructed_uncertainty \
            if reconstructed_uncertainty > 0 else Decimal("Infinity")

        checks = {
            "absorption_identity": relative_error(
                exact_decimal(logged_absorption_f), reconstructed_absorption
            ),
            "net_fma_identity": relative_error(
                exact_decimal(logged_net_f), reconstructed_net
            ),
            "signed_rate_identity": relative_error(
                exact_decimal(logged_rate_f), reconstructed_rate
            ),
            "uncertainty_identity": relative_error(
                exact_decimal(logged_uncertainty_f), reconstructed_uncertainty
            ),
            "cancellation_condition_identity": relative_error(
                exact_decimal(logged_condition_f), reconstructed_condition
            ),
        }
        failed_checks = [
            name for name, error in checks.items()
            if not error.is_finite() or error > maximum_relative_error
        ]
        unresolved_consistent = abs(logged_rate_f) <= logged_uncertainty_f
        if failed_checks or not unresolved_consistent:
            detail = ",".join(failed_checks) or "unresolved_inequality"
            raise AuditError(f"witness identity failure: {detail}")

        return {
            "phase": fields.get("phase", "UNSPECIFIED"),
            "line": int(fields["line"]),
            "shell": int(fields["shell"]),
            "status": fields["status"],
            "signed_direction": (
                "COOLING" if reconstructed_rate_f > 0.0 else "HEATING"
            ),
            "identity_max_relative_error": float(max(checks.values())),
            "identity_relative_errors": {
                name: float(error) for name, error in checks.items()
            },
            "reconstructed": {
                "absorption_per_sr": str(reconstructed_absorption),
                "net_per_sr": str(reconstructed_net),
                "signed_rate": str(reconstructed_rate),
                "absolute_uncertainty": str(reconstructed_uncertainty),
                "cancellation_condition": str(reconstructed_condition),
            },
            "inputs": {
                "eta_per_sr": str(eta),
                "chi_effective": str(chi),
                "jbar": str(jbar),
                "jbar_bound": str(jbar_bound),
                "deck_scale": str(deck_scale),
            },
            "proof_requirement": {
                "required_symmetric_jbar_bound_strictly_below": str(
                    required_symmetric_jbar_bound
                ),
                "current_jbar_bound": str(jbar_bound),
                "current_to_required_bound_ratio": str(current_to_required),
                "signed_rate_to_uncertainty_ratio": str(rate_margin),
                "one_sided_jbar_threshold_eta_over_chi": (
                    str(one_sided_jbar_threshold)
                    if one_sided_jbar_threshold is not None else None
                ),
            },
            "repair_counters": {
                "clamp": int(fields.get("clamp", "-1")),
                "floor": int(fields.get("floor", "-1")),
                "jitter": int(fields.get("jitter", "-1")),
            },
        }


def parse_witnesses(log_path: Path, maximum_relative_error: Decimal) -> list[dict[str, Any]]:
    witnesses: list[dict[str, Any]] = []
    for line_number, text in enumerate(
        log_path.read_text(encoding="utf-8", errors="strict").splitlines(), start=1
    ):
        if MARKER not in text:
            continue
        fields = dict(PAIR_RE.findall(text))
        try:
            witness = audit_record(fields, maximum_relative_error)
        except AuditError as exc:
            raise AuditError(f"{log_path}:{line_number}: {exc}") from exc
        witness["source_line_number"] = line_number
        witnesses.append(witness)
    if not witnesses:
        raise AuditError(f"no {MARKER} records in {log_path}")
    return witnesses


def parse_expected(values: list[str]) -> set[tuple[int, int]]:
    expected: set[tuple[int, int]] = set()
    for value in values:
        try:
            line_text, shell_text = value.split(":", 1)
            key = (int(line_text), int(shell_text))
        except (ValueError, TypeError) as exc:
            raise AuditError(f"invalid --expect-witness {value!r}; use LINE:SHELL") from exc
        if key in expected:
            raise AuditError(f"duplicate --expect-witness {value!r}")
        expected.add(key)
    return expected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expect-witness", action="append", default=[])
    parser.add_argument(
        "--maximum-relative-error", default="1e-12",
        help="offline identity-judge threshold; never passed to the producer",
    )
    args = parser.parse_args()

    try:
        maximum_relative_error = Decimal(args.maximum_relative_error)
        if not maximum_relative_error.is_finite() or maximum_relative_error <= 0:
            raise AuditError("--maximum-relative-error must be finite and positive")
        if not args.log.is_file():
            raise AuditError(f"missing or non-regular log: {args.log}")
        witnesses = parse_witnesses(args.log, maximum_relative_error)
        expected = parse_expected(args.expect_witness)
        observed = {(item["line"], item["shell"]) for item in witnesses}
        if expected and observed != expected:
            raise AuditError(
                f"witness set mismatch expected={sorted(expected)} observed={sorted(observed)}"
            )
        if any(any(value != 0 for value in item["repair_counters"].values())
               for item in witnesses):
            raise AuditError("nonzero or missing repair counter in witness")
        payload = {
            "schema": "lumina-a2-10-cancellation-witness-audit-v1",
            "status": "PASS",
            "reason_code": "LINE_NET_IDENTITIES_AND_REQUIRED_BOUNDS_REPRODUCED",
            "source_log": str(args.log.resolve()),
            "source_sha256": hashlib.sha256(args.log.read_bytes()).hexdigest(),
            "maximum_relative_error": str(maximum_relative_error),
            "witness_count": len(witnesses),
            "observed_line_shell": [list(key) for key in sorted(observed)],
            "physical_values_modified": False,
            "floor": 0,
            "cap": 0,
            "clamp": 0,
            "jitter": 0,
            "witnesses": witnesses,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        ratios = ",".join(
            f"{item['line']}:{item['shell']}="
            f"{float(Decimal(item['proof_requirement']['current_to_required_bound_ratio'])):.9g}"
            for item in witnesses
        )
        print(
            f"PASS A2_10_CANCELLATION_WITNESS_AUDIT witnesses={len(witnesses)} "
            f"bound_ratios={ratios} repair=0"
        )
        return 0
    except (AuditError, OSError, ValueError) as exc:
        print(f"FAIL A2_10_CANCELLATION_WITNESS_AUDIT reason={exc}")
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
