#!/usr/bin/env python3
"""Audit whether a blocked A2-10 Sobolev proof bound has a fixed component."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


FOUR_PI = 12.56637061435917295385


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def fields(line: str) -> dict[str, str]:
    return dict(token.split("=", 1) for token in line.split() if "=" in token)


def cmfgen_beta(tau: float) -> float:
    if abs(tau) < 1.0e-3:
        companion = 0.5 - tau / 6.0 * (1.0 - tau / 4.0)
        return 1.0 - tau * companion
    if tau < 40.0:
        return (1.0 - math.exp(-tau)) / tau
    return 1.0 / tau


def close_ulp(actual: float, expected: float, operations: int = 8) -> bool:
    bound = operations * max(math.ulp(actual), math.ulp(expected))
    return abs(actual - expected) <= bound


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--line-net-source", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--line", type=int, required=True)
    parser.add_argument("--shell", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    source = args.line_net_source.read_text()
    match = re.search(
        r"int line_net_sobolev_radiation\(.*?\n\}", source, re.DOTALL)
    if not match:
        raise SystemExit("Sobolev radiation function not found")
    body = match.group(0)
    assignment = re.findall(
        r"double uncertainty\s*=\s*([^;]+);", body)
    if assignment != ["beta * continuum_j_absolute_uncertainty"]:
        raise SystemExit(f"unexpected propagated-bound assignment: {assignment}")
    if re.search(r"uncertainty\s*[+\-*/]=", body):
        raise SystemExit("propagated uncertainty has an additional component")

    witness = None
    with args.log.open(errors="replace") as stream:
        for raw in stream:
            if "[A2-10][LINE-NET-CELL-BLOCKED]" not in raw:
                continue
            item = fields(raw)
            if int(item.get("line", -1)) == args.line and int(
                    item.get("shell", -1)) == args.shell:
                witness = item
    if witness is None:
        raise SystemExit("requested blocked witness not found")

    tau = float(witness["tau_raw"])
    beta = cmfgen_beta(tau)
    jbar_bound = float(witness["Jbar_bound"])
    chi = float(witness["chi_effective"])
    deck_scale = float(witness["deck_scale"])
    actual_uncertainty = float(witness["uncertainty"])
    expected_uncertainty = abs(chi) * jbar_bound * FOUR_PI * deck_scale
    if not close_ulp(actual_uncertainty, expected_uncertainty):
        raise SystemExit(
            "line-rate uncertainty does not close from chi*Jbar_bound*4pi*scale")
    actual_ratio = float(witness["uncertainty_to_abs_rate"])
    expected_ratio = actual_uncertainty / abs(float(witness["signed_rate"]))
    if not close_ulp(actual_ratio, expected_ratio, 4):
        raise SystemExit("uncertainty/rate ratio does not close")
    if not (tau > 0.0 and 0.0 < beta < 1.0):
        raise SystemExit("witness is not a strictly contracting positive-tau cell")

    inferred_continuum_bound = jbar_bound / beta
    report = {
        "schema": "a210-sobolev-bound-decomposition-v1",
        "status": "PASS_CONTRACTING_PROPAGATED_INPUT_BOUND_ONLY",
        "line_net_source": str(args.line_net_source.resolve()),
        "line_net_source_sha256": digest(args.line_net_source),
        "source_log": str(args.log.resolve()),
        "source_log_sha256": digest(args.log),
        "witness": {
            "line": args.line,
            "shell": args.shell,
            "logged_phase": witness.get("phase"),
            "tau_raw": tau,
            "beta": beta,
            "strictly_contracting": True,
            "jbar_bound": jbar_bound,
            "inferred_pre_sobolev_continuum_bound": inferred_continuum_bound,
            "beta_times_inferred_bound": beta * inferred_continuum_bound,
            "signed_rate": float(witness["signed_rate"]),
            "line_rate_uncertainty": actual_uncertainty,
            "reconstructed_line_rate_uncertainty": expected_uncertainty,
            "uncertainty_to_abs_rate": actual_ratio,
        },
        "decomposition": {
            "propagated_continuum_input_bound": "beta * continuum_j_absolute_uncertainty",
            "continuum_multiplier": "beta",
            "local_emission_propagated_input_bound": 0,
            "additional_noncontracting_propagated_component": 0,
            "rounding_scope": (
                "The operator contract explicitly does not claim a complete "
                "rounding enclosure for beta/companion/Jbar arithmetic."),
        },
        "decision": {
            "blind_rung_escalation": False,
            "minimum_next_use": (
                "A stronger rung is justified only for the explicitly requested "
                "private-temperature callback because no fixed propagated input-bound "
                "component was found."),
            "physical_no_bracket_changed": False,
        },
        "physical_values_modified": 0,
        "floor": 0,
        "cap": 0,
        "clamp": 0,
        "jitter": 0,
        "repair": 0,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "report": str(args.report)},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
