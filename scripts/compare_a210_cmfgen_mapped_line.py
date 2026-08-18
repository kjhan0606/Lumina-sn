#!/usr/bin/env python3
"""Compare one requested Lumina A2-10 line cell with finite CMFGEN records.

This is a diagnostic comparison, not a parity gate: the current CMFGEN and
Lumina states are not identical.  It preserves every signed value and reports
ratios without tolerance, clipping, flooring, or repair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any


PREFIX = "[A2-10][LINE-NET-CELL-FINITE]"
FIELDS = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")
INTERIOR_NAMES = ("PUBLIC_SEED", "GEOMETRIC_MID")


class ComparisonError(RuntimeError):
    pass


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def finite(fields: dict[str, str], name: str) -> float:
    try:
        value = float(fields[name])
    except (KeyError, ValueError) as exc:
        raise ComparisonError(f"missing or invalid finite field: {name}") from exc
    if not math.isfinite(value):
        raise ComparisonError(f"nonfinite field: {name}={value}")
    return value


def optional_finite(fields: dict[str, str], name: str) -> float | None:
    return finite(fields, name) if name in fields else None


def parse_records(path: Path, line_id: int, shell: int) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise ComparisonError(f"missing or unsafe stderr: {path}")
    records: list[dict[str, Any]] = []
    interior_index = 0
    with path.open("rt", encoding="utf-8", errors="strict") as stream:
        for text in stream:
            if not text.startswith(PREFIX):
                continue
            fields = dict(FIELDS.findall(text))
            try:
                record_line = int(fields["line"])
                record_shell = int(fields["shell"])
            except (KeyError, ValueError) as exc:
                raise ComparisonError("malformed line-cell record") from exc
            if record_line != line_id or record_shell != shell:
                continue
            if fields.get("requested_cell") != "1":
                raise ComparisonError("selected record lacks requested_cell=1")
            for name in ("clamp", "floor", "jitter"):
                if fields.get(name) != "0":
                    raise ComparisonError(f"forbidden numerical repair: {name}")
            phase = fields.get("phase", "")
            if phase == "INTERIOR":
                phase = (
                    INTERIOR_NAMES[interior_index]
                    if interior_index < len(INTERIOR_NAMES)
                    else f"INTERIOR_{interior_index + 1}"
                )
                interior_index += 1
            elif phase not in ("LOWER", "UPPER"):
                raise ComparisonError(f"unexpected phase: {phase!r}")

            eta = finite(fields, "eta_per_sr")
            absorption = finite(fields, "absorption_per_sr")
            net = finite(fields, "net_per_sr")
            signed_rate = finite(fields, "signed_rate")
            condition = finite(fields, "cancellation_condition")
            if eta < 0.0 or absorption < 0.0 or condition < 0.0:
                raise ComparisonError("invalid signed-line components")
            if not math.isclose(net, eta - absorption, rel_tol=0.0,
                                abs_tol=8.0 * math.ulp(max(abs(eta), abs(absorption)))):
                raise ComparisonError("printed eta-absorption closure failed")
            inferred_scale = (
                signed_rate / (4.0 * math.pi * net) if net != 0.0 else None
            )
            jbar_over_source = absorption / eta if eta != 0.0 else None
            logged_jbar_over_source = optional_finite(
                fields, "jbar_over_source"
            )
            if (logged_jbar_over_source is not None and
                jbar_over_source is not None and
                abs(logged_jbar_over_source - jbar_over_source) >
                    4.0 * math.ulp(max(abs(logged_jbar_over_source),
                                       abs(jbar_over_source)))):
                raise ComparisonError("logged Jbar/source closure failed")
            try:
                exact_zero = int(fields.get("exact_zero", "-1"))
            except ValueError as exc:
                raise ComparisonError("invalid exact_zero provenance") from exc
            if exact_zero not in (0, 1):
                raise ComparisonError("invalid exact_zero provenance")
            records.append({
                "phase": phase,
                "temperature_K": finite(fields, "T_e_K"),
                "eta_per_sr_cgs": eta,
                "absorption_per_sr_cgs": absorption,
                "net_per_sr_cgs": net,
                "signed_rate_cgs": signed_rate,
                "jbar": finite(fields, "Jbar"),
                "jbar_absolute_error_bound": finite(fields, "Jbar_local_bound"),
                "jbar_over_source": jbar_over_source,
                "cancellation_condition": condition,
                "inferred_deck_scale": inferred_scale,
                "logged_deck_scale": optional_finite(fields, "deck_scale"),
                "electron_density_cm3": optional_finite(fields, "n_e_cm3"),
                "atom_density_cm3": optional_finite(fields, "n_atom_cm3"),
                "status": fields.get("status"),
                "exact_zero": exact_zero,
            })
    if len(records) != 4:
        raise ComparisonError(
            f"expected LOWER, UPPER, PUBLIC_SEED, GEOMETRIC_MID records; "
            f"found {len(records)}"
        )
    phases = [record["phase"] for record in records]
    if phases != ["LOWER", "UPPER", "PUBLIC_SEED", "GEOMETRIC_MID"]:
        raise ComparisonError(f"unexpected requested-cell phase order: {phases}")
    return records


def compare(reference: Path, stderr: Path, line_id: int, shell: int) -> dict[str, Any]:
    try:
        source = json.loads(reference.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ComparisonError(f"invalid CMFGEN reference: {reference}") from exc
    mapping = source.get("mapping", {})
    if mapping.get("lumina_line_id_zero_based") != line_id:
        raise ComparisonError("reference Lumina line identity mismatch")
    if source.get("physical_mutation") != 0 or any(
        source.get(name) != 0 for name in ("floor", "cap", "clamp", "jitter", "repair")
    ):
        raise ComparisonError("reference contains a forbidden repair")
    cmfgen_depths = source.get("depths")
    if not isinstance(cmfgen_depths, dict) or not cmfgen_depths:
        raise ComparisonError("reference has no CMFGEN depth records")
    lumina = parse_records(stderr, line_id, shell)
    comparisons: list[dict[str, Any]] = []
    for record in lumina:
        by_depth: dict[str, dict[str, float | None]] = {}
        for depth, cmfgen in cmfgen_depths.items():
            if not isinstance(cmfgen, dict):
                raise ComparisonError(f"invalid CMFGEN depth: {depth}")
            try:
                ref_eta = float(cmfgen["eta_per_sr_cgs"])
                ref_rate = float(cmfgen["signed_rate_cgs"])
                ref_js = float(cmfgen["jbar_over_source"])
                ref_condition = float(cmfgen["cancellation_condition"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ComparisonError(
                    f"invalid finite CMFGEN depth fields: {depth}"
                ) from exc
            if not all(math.isfinite(value) for value in
                       (ref_eta, ref_rate, ref_js, ref_condition)):
                raise ComparisonError(f"nonfinite CMFGEN depth: {depth}")
            by_depth[depth] = {
                "eta_ratio_lumina_over_cmfgen": (
                    record["eta_per_sr_cgs"] / ref_eta if ref_eta != 0.0 else None
                ),
                "signed_rate_ratio_lumina_over_cmfgen": (
                    record["signed_rate_cgs"] / ref_rate
                    if ref_rate != 0.0 else None
                ),
                "jbar_over_source_difference": (
                    record["jbar_over_source"] - ref_js
                    if record["jbar_over_source"] is not None else None
                ),
                "cancellation_condition_ratio": (
                    record["cancellation_condition"] / ref_condition
                    if ref_condition != 0.0 else None
                ),
            }
        comparisons.append({"lumina": record, "against_cmfgen_depths": by_depth})
    return {
        "schema": "lumina-a210-cmfgen-mapped-line-comparison-v1",
        "status": "PASS",
        "verdict": "FINITE_SAME_TRANSITION_SCALE_COMPARISON_NOT_STATE_PARITY",
        "reference": str(reference.resolve()),
        "reference_sha256": digest(reference),
        "lumina_stderr": str(stderr.resolve()),
        "lumina_stderr_sha256": digest(stderr),
        "line_id": line_id,
        "shell": shell,
        "comparisons": comparisons,
        "same_transition": True,
        "same_signed_observable": True,
        "same_state": False,
        "parity_claim": False,
        "physical_values_modified": False,
        "floor": 0,
        "cap": 0,
        "clamp": 0,
        "jitter": 0,
        "repair": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--stderr", type=Path, required=True)
    parser.add_argument("--line", type=int, required=True)
    parser.add_argument("--shell", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = compare(args.reference, args.stderr, args.line, args.shell)
        atomic_write(args.report, payload)
        print(
            "PASS A210_CMFGEN_MAPPED_LINE_COMPARISON "
            f"line={args.line} shell={args.shell} records=4 parity=0 repair=0"
        )
        return 0
    except (ComparisonError, OSError, UnicodeError) as exc:
        atomic_write(args.report, {
            "schema": "lumina-a210-cmfgen-mapped-line-comparison-v1",
            "status": "FAIL",
            "error": str(exc),
        })
        print(f"FAIL A210_CMFGEN_MAPPED_LINE_COMPARISON reason={exc}")
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
