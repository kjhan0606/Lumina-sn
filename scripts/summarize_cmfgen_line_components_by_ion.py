#!/usr/bin/env python3
"""Join CMFGEN LINEHEAT/NETRATE into signed line components by ion owner."""

from __future__ import annotations

import argparse
import json
import math
from itertools import zip_longest
from pathlib import Path

from extract_cmfgen_line_net_fixture import (
    CMFGEN_RE_INTERNAL_TO_CGS,
    Header,
    NetHeader,
    lineheat_header,
    netrate_header,
    vectors,
)
from summarize_cmfgen_lineheat_depths import OnlineFsum, digest
from summarize_cmfgen_lineheat_ion_owners import transition_owner


def compatible(line: Header, net: NetHeader) -> bool:
    frequency_scale = max(abs(line.frequency_field), 1.0)
    return (
        line.line_id == net.line_id and
        line.transition == net.transition and
        line.lower == net.lower and line.upper == net.upper and
        abs(line.frequency_field - net.frequency_field) <=
        1.0e-9 * frequency_scale
    )


def accumulator() -> dict[str, object]:
    return {
        "signed": OnlineFsum(),
        "absolute_signed": OnlineFsum(),
        "emission": OnlineFsum(),
        "absorption": OnlineFsum(),
        "recomposed_signed": OnlineFsum(),
        "component_closure_bound": OnlineFsum(),
        "records": 0,
        "cooling": 0,
        "heating": 0,
        "exact_zero": 0,
    }


def summarize(lineheat: Path, netrate: Path, depth_count: int,
              depths: list[int], scale_threshold: float,
              finite_reference: Path | None) -> dict:
    if depth_count <= 0 or not depths or len(depths) != len(set(depths)):
        raise ValueError("invalid depth selection")
    if any(depth <= 0 or depth > depth_count for depth in depths):
        raise ValueError("selected depth outside CMFGEN vector")
    if not math.isfinite(scale_threshold) or scale_threshold < 0.0:
        raise ValueError("invalid CMFGEN SCL_LN_FAC")

    by_owner: dict[int, dict[str, dict[str, object]]] = {
        depth: {} for depth in depths
    }
    totals = {depth: accumulator() for depth in depths}
    record_count = 0
    line_iter = vectors(lineheat, depth_count, lineheat_header)
    net_iter = vectors(netrate, depth_count, netrate_header)
    for pair_index, pair in enumerate(zip_longest(line_iter, net_iter), 1):
        line_record, net_record = pair
        if line_record is None or net_record is None:
            raise ValueError("LINEHEAT/NETRATE record-count mismatch")
        line_header, line_values = line_record
        net_header, znet_values = net_record
        if not isinstance(line_header, Header) or not isinstance(
                net_header, NetHeader):
            raise ValueError("unexpected CMFGEN header type")
        if not compatible(line_header, net_header):
            raise ValueError(
                f"LINEHEAT/NETRATE identity mismatch at pair {pair_index}: "
                f"{line_header.line_id}/{net_header.line_id}")
        if not math.isfinite(line_header.scale):
            raise ValueError(f"line {line_header.line_id}: invalid raw scale")
        effective_scale = line_header.scale
        if abs(effective_scale - 1.0) > scale_threshold:
            effective_scale = 1.0
        if effective_scale == 0.0:
            raise ValueError(f"line {line_header.line_id}: zero effective scale")
        raw_owner, normalized_owner = transition_owner(
            line_header.transition)
        record_count += 1
        for depth in depths:
            signed = line_values[depth - 1]
            znet = znet_values[depth - 1]
            if not math.isfinite(signed) or not math.isfinite(znet):
                raise ValueError(
                    f"line {line_header.line_id} depth {depth}: nonfinite")
            if znet == 0.0:
                raise ValueError(
                    f"line {line_header.line_id} depth {depth}: "
                    "zero ZNET cannot identify finite components")
            # LINEHEAT stores scale*ETAL_MAT*ZNET.  Therefore division by the
            # independently printed NETRATE ZNET yields scale*ETAL_MAT, the
            # scaled emission component in the same internal units.  The
            # absorption component follows from the exact signed equation.
            # No clipping, absolute-value sign repair, or replacement occurs.
            emission = signed / znet
            absorption = emission - signed
            unscaled_emission = emission / effective_scale
            if (not math.isfinite(emission) or
                    not math.isfinite(unscaled_emission) or
                    unscaled_emission <= 0.0 or
                    not math.isfinite(absorption)):
                raise ValueError(
                    f"line {line_header.line_id} depth {depth}: "
                    "invalid derived component")
            recomposed = math.fsum((emission, -absorption))
            local_closure = math.fsum((recomposed, -signed))
            local_bound = 2.0 * (
                math.ulp(emission) + math.ulp(absorption) +
                math.ulp(recomposed) + math.ulp(signed))
            if (not math.isfinite(recomposed) or
                    not math.isfinite(local_closure) or
                    not math.isfinite(local_bound) or local_bound < 0.0 or
                    abs(local_closure) > local_bound):
                raise ValueError(
                    f"line {line_header.line_id} depth {depth}: "
                    "component recomposition exceeds binary64 bound")
            owner = by_owner[depth].setdefault(
                raw_owner,
                {**accumulator(), "normalized_species": normalized_owner},
            )
            for destination in (owner, totals[depth]):
                destination["signed"].add(signed)
                destination["absolute_signed"].add(abs(signed))
                destination["emission"].add(emission)
                destination["absorption"].add(absorption)
                destination["recomposed_signed"].add(recomposed)
                destination["component_closure_bound"].add(local_bound)
                destination["records"] += 1
                if signed > 0.0:
                    destination["cooling"] += 1
                elif signed < 0.0:
                    destination["heating"] += 1
                else:
                    destination["exact_zero"] += 1

    if record_count == 0:
        raise ValueError("no paired CMFGEN line records")

    finite = json.loads(finite_reference.read_text()) \
        if finite_reference else None
    rendered: dict[str, object] = {}
    for depth in depths:
        rows: list[dict[str, object]] = []
        grouped = {
            "signed": OnlineFsum(), "absolute_signed": OnlineFsum(),
            "emission": OnlineFsum(), "absorption": OnlineFsum(),
        }
        for raw_owner, owner in by_owner[depth].items():
            signed = owner["signed"].total()
            absolute = owner["absolute_signed"].total()
            emission = owner["emission"].total()
            absorption = owner["absorption"].total()
            recomposed = owner["recomposed_signed"].total()
            component_delta = math.fsum((recomposed, -signed))
            component_bound = (
                owner["component_closure_bound"].total() +
                math.ulp(recomposed) + math.ulp(signed))
            if abs(component_delta) > component_bound:
                raise ValueError(
                    f"depth {depth} owner {raw_owner}: component closure drift")
            grouped["signed"].add(signed)
            grouped["absolute_signed"].add(absolute)
            grouped["emission"].add(emission)
            grouped["absorption"].add(absorption)
            rows.append({
                "cmfgen_label": raw_owner,
                "normalized_species": owner["normalized_species"],
                "signed_internal": signed,
                "signed_cgs_erg_cm3_s": signed * CMFGEN_RE_INTERNAL_TO_CGS,
                "absolute_signed_internal": absolute,
                "scaled_emission_internal": emission,
                "scaled_emission_cgs_erg_cm3_s": (
                    emission * CMFGEN_RE_INTERNAL_TO_CGS),
                "scaled_absorption_internal": absorption,
                "scaled_absorption_cgs_erg_cm3_s": (
                    absorption * CMFGEN_RE_INTERNAL_TO_CGS),
                "cellwise_recomposed_signed_internal": recomposed,
                "cellwise_component_closure_internal": component_delta,
                "cellwise_component_closure_bound_internal": component_bound,
                "cellwise_component_closure_verified": True,
                "line_records": owner["records"],
                "cooling_count": owner["cooling"],
                "heating_count": owner["heating"],
                "exact_zero_count": owner["exact_zero"],
            })
        line_signed = totals[depth]["signed"].total()
        line_absolute = totals[depth]["absolute_signed"].total()
        line_emission = totals[depth]["emission"].total()
        line_absorption = totals[depth]["absorption"].total()
        line_recomposed = totals[depth]["recomposed_signed"].total()
        line_component_delta = math.fsum((line_recomposed, -line_signed))
        line_component_bound = (
            totals[depth]["component_closure_bound"].total() +
            math.ulp(line_recomposed) + math.ulp(line_signed))
        if abs(line_component_delta) > line_component_bound:
            raise ValueError(f"depth {depth}: component closure drift")
        reference_check = None
        if finite is not None:
            expected = finite["depths"][str(depth)]
            reference_check = {
                "signed_internal_expected": expected["signed_internal"],
                "absolute_internal_expected": expected["absolute_internal"],
                "signed_bit_exact": line_signed == expected["signed_internal"],
                "absolute_bit_exact": (
                    line_absolute == expected["absolute_internal"]),
            }
            if not all((reference_check["signed_bit_exact"],
                        reference_check["absolute_bit_exact"])):
                raise ValueError(f"depth {depth}: finite reference drift")
        rows.sort(key=lambda row: abs(float(row["signed_internal"])),
                  reverse=True)
        rendered[str(depth)] = {
            "line_order_signed_internal": line_signed,
            "line_order_absolute_signed_internal": line_absolute,
            "line_order_scaled_emission_internal": line_emission,
            "line_order_scaled_absorption_internal": line_absorption,
            "cellwise_recomposed_signed_internal": line_recomposed,
            "cellwise_component_closure_internal": line_component_delta,
            "cellwise_component_closure_bound_internal": line_component_bound,
            "cellwise_component_closure_verified": True,
            "separately_grouped_component_delta_internal": (
                line_emission - line_absorption - line_signed),
            "grouped_signed_internal": grouped["signed"].total(),
            "grouped_absolute_signed_internal": (
                grouped["absolute_signed"].total()),
            "grouped_scaled_emission_internal": grouped["emission"].total(),
            "grouped_scaled_absorption_internal": (
                grouped["absorption"].total()),
            "signed_grouping_delta_internal": (
                grouped["signed"].total() - line_signed),
            "emission_grouping_delta_internal": (
                grouped["emission"].total() - line_emission),
            "absorption_grouping_delta_internal": (
                grouped["absorption"].total() - line_absorption),
            "owner_count": len(rows),
            "owners_by_abs_signed_ion_total": rows,
            "finite_reference_check": reference_check,
        }

    return {
        "schema": "cmfgen-line-components-ion-owner-v1",
        "lineheat": {"path": str(lineheat.resolve()), "sha256": digest(lineheat)},
        "netrate": {"path": str(netrate.resolve()), "sha256": digest(netrate)},
        "finite_reference": ({
            "path": str(finite_reference.resolve()),
            "sha256": digest(finite_reference),
        } if finite_reference else None),
        "depth_count": depth_count,
        "paired_line_records": record_count,
        "equations": {
            "signed": "LINEHEAT = scale * ETAL_MAT * ZNET",
            "scaled_emission": "LINEHEAT / NETRATE_ZNET",
            "scaled_absorption": "scaled_emission - LINEHEAT",
            "cgs": "internal * 4*pi*1e-10",
        },
        "line_cooling_scale_contract": {
            "SCL_LN": True,
            "SCL_SL_OPAC": False,
            "SCL_LN_FAC": scale_threshold,
            "header_field": "raw (E_lower-E_upper)/line_frequency",
            "effective_scale": (
                "raw when abs(raw-1)<=SCL_LN_FAC, otherwise 1"),
            "source": "new_main/cmfgen_sub.f:2754-2762",
        },
        "printed_source_precision": {
            "LINEHEAT": "1P,5E12.4", "NETRATE": "1P,5E14.6",
        },
        "interpretation": (
            "Finite diagnostic decomposition of independently printed CMFGEN "
            "ledgers; component precision is limited by their serialization. "
            "The authoritative signed net is independently retained, while "
            "cellwise emission-minus-absorption recomposition must close "
            "inside an explicit binary64 ulp bound."),
        "physical_values_modified": 0,
        "floor": 0, "cap": 0, "clamp": 0, "jitter": 0, "repair": 0,
        "depths": rendered,
        "verdict": "COMPLETE_FINITE_COMPONENT_DECOMPOSITION",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lineheat", required=True, type=Path)
    parser.add_argument("--netrate", required=True, type=Path)
    parser.add_argument("--depth-count", type=int, default=90)
    parser.add_argument("--depth", action="append", required=True, type=int)
    parser.add_argument("--scale-threshold", required=True, type=float)
    parser.add_argument("--finite-reference", type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = summarize(args.lineheat, args.netrate, args.depth_count,
                           args.depth, args.scale_threshold,
                           args.finite_reference)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"[cmfgen-line-components-ion-owner] ERROR: {exc}")
        return 2
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["verdict"],
        "paired_line_records": report["paired_line_records"],
        "json_out": str(args.json_out),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
