#!/usr/bin/env python3
"""Stream CMFGEN LINEHEAT into transition-label ion-owner totals."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from cmfgen_ion_identity import parse_cmfgen_ion_id
from extract_cmfgen_line_net_fixture import (
    CMFGEN_RE_INTERNAL_TO_CGS,
    Header,
    lineheat_header,
    vectors,
)
from summarize_cmfgen_lineheat_depths import OnlineFsum, digest


def transition_owner(transition: str) -> tuple[str, str]:
    raw = transition.split("(", 1)[0]
    _, _, normalized = parse_cmfgen_ion_id(raw)
    return raw, normalized


def summarize(path: Path, depth_count: int, depths: list[int],
              model_spec: Path | None, finite_reference: Path | None) -> dict:
    if not depths or len(depths) != len(set(depths)):
        raise ValueError("depths must be nonempty and unique")
    if any(depth <= 0 or depth > depth_count for depth in depths):
        raise ValueError("selected depth is outside LINEHEAT")
    owners: dict[int, dict[str, dict[str, object]]] = {
        depth: {} for depth in depths
    }
    total = {
        depth: {"signed": OnlineFsum(), "absolute": OnlineFsum()}
        for depth in depths
    }
    records = 0
    for header, values in vectors(path, depth_count, lineheat_header):
        if not isinstance(header, Header):
            raise ValueError("unexpected LINEHEAT header type")
        raw, normalized = transition_owner(header.transition)
        records += 1
        for depth in depths:
            value = values[depth - 1]
            if not math.isfinite(value):
                raise ValueError(
                    f"line {header.line_id} depth {depth}: nonfinite LINEHEAT")
            owner = owners[depth].setdefault(raw, {
                "normalized_species": normalized,
                "signed": OnlineFsum(), "absolute": OnlineFsum(),
                "line_records": 0, "positive_count": 0,
                "negative_count": 0, "zero_count": 0,
            })
            owner["signed"].add(value)
            owner["absolute"].add(abs(value))
            owner["line_records"] += 1
            if value > 0.0:
                owner["positive_count"] += 1
            elif value < 0.0:
                owner["negative_count"] += 1
            else:
                owner["zero_count"] += 1
            total[depth]["signed"].add(value)
            total[depth]["absolute"].add(abs(value))
    if records == 0:
        raise ValueError("LINEHEAT yielded no numbered Sobolev records")

    reference = json.loads(finite_reference.read_text()) \
        if finite_reference else None
    rendered: dict[str, object] = {}
    for depth in depths:
        rows: list[dict[str, object]] = []
        grouped_signed = OnlineFsum()
        grouped_absolute = OnlineFsum()
        for raw, owner in owners[depth].items():
            signed_internal = owner["signed"].total()
            absolute_internal = owner["absolute"].total()
            grouped_signed.add(signed_internal)
            grouped_absolute.add(absolute_internal)
            rows.append({
                "cmfgen_label": raw,
                "normalized_species": owner["normalized_species"],
                "signed_internal": signed_internal,
                "signed_cgs_erg_cm3_s": (
                    signed_internal * CMFGEN_RE_INTERNAL_TO_CGS),
                "absolute_internal": absolute_internal,
                "absolute_cgs_erg_cm3_s": (
                    absolute_internal * CMFGEN_RE_INTERNAL_TO_CGS),
                "cancellation_condition": (
                    absolute_internal / abs(signed_internal)
                    if signed_internal != 0.0 else
                    (0.0 if absolute_internal == 0.0 else "infinite")),
                "line_records": owner["line_records"],
                "positive_count": owner["positive_count"],
                "negative_count": owner["negative_count"],
                "zero_count": owner["zero_count"],
            })
        signed = total[depth]["signed"].total()
        absolute = total[depth]["absolute"].total()
        ref_check = None
        if reference is not None:
            expected = reference["depths"][str(depth)]
            ref_check = {
                "signed_internal_expected": expected["signed_internal"],
                "absolute_internal_expected": expected["absolute_internal"],
                "signed_bit_exact": signed == expected["signed_internal"],
                "absolute_bit_exact": absolute == expected["absolute_internal"],
            }
            if not ref_check["signed_bit_exact"] or not ref_check[
                    "absolute_bit_exact"]:
                raise ValueError(f"depth {depth}: finite reference drift")
        rows.sort(key=lambda row: abs(float(row["signed_internal"])), reverse=True)
        rendered[str(depth)] = {
            "line_order_signed_internal": signed,
            "line_order_signed_cgs_erg_cm3_s": (
                signed * CMFGEN_RE_INTERNAL_TO_CGS),
            "line_order_absolute_internal": absolute,
            "grouped_signed_internal": grouped_signed.total(),
            "signed_grouping_delta_internal": grouped_signed.total() - signed,
            "grouped_absolute_internal": grouped_absolute.total(),
            "absolute_grouping_delta_internal": (
                grouped_absolute.total() - absolute),
            "owner_count": len(rows),
            "top_by_abs_signed_ion_total": rows,
            "finite_reference_check": ref_check,
        }

    return {
        "schema": "cmfgen-lineheat-ion-owner-summary-v1",
        "source": str(path.resolve()),
        "source_sha256": digest(path),
        "model_spec": ({
            "path": str(model_spec.resolve()), "sha256": digest(model_spec),
            "ion_label_contract": (
                "CMFGEN 2/SIX/SEV -> II/VI/VII; Sk/Nk -> chemical Si/Ni"),
        } if model_spec else None),
        "finite_reference": ({
            "path": str(finite_reference.resolve()),
            "sha256": digest(finite_reference),
        } if finite_reference else None),
        "depth_indexing": "CMFGEN_ONE_BASED",
        "depth_count": depth_count,
        "line_records": records,
        "equation": "sum_scaled_LINEHEAT*4*pi*1e-10",
        "summation": "ONLINE_NONOVERLAPPING_BINARY64_PARTIALS",
        "physical_mutation": 0,
        "floor": 0,
        "cap": 0,
        "clamp": 0,
        "jitter": 0,
        "repair": 0,
        "depths": rendered,
        "verdict": "COMPLETE_DIAGNOSTIC_OWNER_DECOMPOSITION",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lineheat", required=True, type=Path)
    parser.add_argument("--depth-count", type=int, default=90)
    parser.add_argument("--depth", action="append", type=int, required=True)
    parser.add_argument("--model-spec", type=Path)
    parser.add_argument("--finite-reference", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    try:
        report = summarize(args.lineheat, args.depth_count, args.depth,
                           args.model_spec, args.finite_reference)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"[cmfgen-lineheat-ion-owner-summary] ERROR: {exc}")
        return 2
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
