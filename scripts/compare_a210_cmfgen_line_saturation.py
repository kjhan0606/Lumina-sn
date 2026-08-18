#!/usr/bin/env python3
"""Cross-match the sealed Lumina line-saturation rows to CMFGEN NETRATE.

The join is exact in ion and full-level identity.  Frequency comparison uses
only the half-unit interval implied by NETRATE's printed six-decimal frequency
field (in 1e15 Hz); multiple candidates inside that interval are rejected as
ambiguous.  The report preserves signed values and makes no state-parity or
physical-cause claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

from extract_cmfgen_line_net_fixture import (
    NetHeader,
    netrate_header,
    vectors,
)
from summarize_cmfgen_lineheat_ion_owners import transition_owner
from cmfgen_ion_identity import parse_cmfgen_ion_id


FREQUENCY_PRINT_HALF_UNIT = 0.5e-6  # NETRATE field units: 1e15 Hz


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


def load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ComparisonError(f"missing or unsafe {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ComparisonError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ComparisonError(f"non-object {label}: {path}")
    return value


def finite(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ComparisonError(f"invalid finite {label}") from exc
    if not math.isfinite(parsed):
        raise ComparisonError(f"nonfinite {label}")
    return parsed


def arithmetic_proof_bound(*values: float) -> float:
    """Conservative binary64 evaluation/serialization bound.

    This is evidence uncertainty only.  It never changes a physical value or
    supplies a physical acceptance tolerance.
    """
    if not values or not all(math.isfinite(value) for value in values):
        raise ComparisonError("nonfinite arithmetic-proof operand")
    magnitude = math.fsum(abs(value) for value in values)
    scale = max((abs(value) for value in values), default=sys.float_info.min)
    scale = max(scale, sys.float_info.min)
    bound = 128.0 * sys.float_info.epsilon * magnitude + \
        16.0 * math.ulp(scale)
    if not math.isfinite(bound) or bound < 0.0:
        raise ComparisonError("invalid arithmetic-proof bound")
    return bound


def validate_zero_repairs(document: dict[str, Any], label: str) -> None:
    mutation = document.get(
        "physical_values_modified", document.get("physical_mutation")
    )
    if mutation not in (False, 0):
        raise ComparisonError(f"{label}: physical mutation present")
    for name in ("floor", "cap", "clamp", "jitter", "repair"):
        if document.get(name) != 0:
            raise ComparisonError(f"{label}: forbidden {name}")


def selected_rows(document: dict[str, Any]) -> list[dict[str, Any]]:
    if document.get("schema") != "lumina-a210-line-saturation-summary-v1" or \
       document.get("status") != "PASS":
        raise ComparisonError("unsealed Lumina saturation summary")
    validate_zero_repairs(document, "Lumina summary")
    summary = document.get("summary")
    rows = document.get("rows")
    if not isinstance(summary, dict) or not isinstance(rows, list) or not rows:
        raise ComparisonError("Lumina summary has no selected rows")
    try:
        selected = int(summary["selected_rows"])
        candidate = int(summary["candidate_rows"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ComparisonError("invalid Lumina selection counts") from exc
    if selected != len(rows) or candidate < selected:
        raise ComparisonError("Lumina selection count mismatch")
    mode = summary.get("selection_mode", "COMBINED_PREFIX")
    if mode not in ("COMBINED_PREFIX", "PER_ION_UNION"):
        raise ComparisonError("Lumina selection mode is invalid")
    if finite(summary.get("selection_target_fraction"), "selection target") != 0.9 or \
       (mode == "COMBINED_PREFIX" and
        finite(summary.get("selected_fraction"), "selected fraction") < 0.9):
        raise ComparisonError("Lumina selection does not cover 90 percent")
    if mode == "PER_ION_UNION":
        metadata = document.get("union_metadata")
        ion_summaries = document.get("union_ion_summaries")
        if not isinstance(metadata, list) or len(metadata) != selected or \
           not isinstance(ion_summaries, list) or len(ion_summaries) != 3:
            raise ComparisonError("Lumina per-ion union proof is incomplete")
    try:
        target_ion = int(summary["target_ion_zero_based"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ComparisonError("Lumina summary target ion is invalid") from exc
    if not 0 <= target_ion <= 10:
        raise ComparisonError("Lumina summary target ion is outside schema")
    identities: set[tuple[int, int, int, int, int]] = set()
    output: list[dict[str, Any]] = []
    previous_rank = 0
    for index, raw in enumerate(rows, 1):
        if not isinstance(raw, dict):
            raise ComparisonError(f"Lumina row {index}: non-object")
        try:
            line = int(raw["line"])
            z = int(raw["Z"])
            ion = int(raw["ion"])
            lower = int(raw["lower_level"])
            upper = int(raw["upper_level"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ComparisonError(f"Lumina row {index}: invalid identity") from exc
        nu = finite(raw.get("nu"), f"Lumina row {index} frequency")
        rank = int(raw.get("rank", -1))
        if (mode == "COMBINED_PREFIX" and rank != index) or \
           rank <= previous_rank or \
           line < 0 or z not in (26, 27, 28) or \
           ion != target_ion or lower < 0 or upper < 0 or nu <= 0.0:
            raise ComparisonError(f"Lumina row {index}: scope/identity mismatch")
        identity = (z, ion + 1, lower + 1, upper + 1, line)
        if identity in identities:
            raise ComparisonError(f"Lumina row {index}: duplicate identity")
        identities.add(identity)
        # Revalidate the fields needed for the physical evidence.  The source
        # summarizer already proved their detailed arithmetic identities.
        for name in (
            "tau_raw", "tau_effective", "beta", "one_minus_beta",
            "jbar_over_source", "Jbar_absolute_bound", "scaled_emission",
        ):
            finite(raw.get(name), f"Lumina row {index} {name}")
        output.append(raw)
        previous_rank = rank
    return output


def coordinate_contract(
    reference: dict[str, Any], netrate: Path, depth_a: int, depth_b: int,
) -> tuple[float, dict[str, Any]]:
    validate_zero_repairs(reference, "coordinate reference")
    source = reference.get("source")
    interpolation = reference.get("shell_zero_velocity_interpolation")
    depths = reference.get("depths")
    if not isinstance(source, dict) or not isinstance(interpolation, dict) or \
       not isinstance(depths, dict):
        raise ComparisonError("coordinate reference missing provenance")
    expected_path = source.get("netrate")
    expected_sha = source.get("netrate_sha256")
    if not isinstance(expected_path, str) or Path(expected_path).resolve() != netrate.resolve():
        raise ComparisonError("coordinate reference NETRATE path mismatch")
    actual_sha = digest(netrate)
    if not isinstance(expected_sha, str) or actual_sha != expected_sha:
        raise ComparisonError("coordinate reference NETRATE SHA mismatch")
    if str(depth_a) not in depths or str(depth_b) not in depths:
        raise ComparisonError("coordinate reference lacks requested depths")
    fraction_key = f"fraction_from_depth_{depth_a}_to_{depth_b}"
    fraction = finite(
        interpolation.get(fraction_key),
        "coordinate interpolation fraction",
    )
    if not (0.0 < fraction < 1.0):
        raise ComparisonError("interpolation fraction outside adjacent depths")
    return fraction, {
        "reference_netrate_sha256": actual_sha,
        "lumina_velocity_km_s": finite(
            interpolation.get("lumina_velocity_km_s"), "Lumina velocity"
        ),
        "fraction_from_depth_a_to_b": fraction,
        "depth_a": depth_a,
        "depth_b": depth_b,
        "scope": interpolation.get("line_interpolation_scope"),
        "state_interpretation": interpolation.get("interpretation"),
    }


def pair_key(z: int, stage: int, lower_full: int, upper_full: int) -> tuple[int, int, int, int]:
    return z, stage, lower_full, upper_full


def match_netrate(
    netrate: Path, depth_count: int, depth_a: int, depth_b: int,
    rows: list[dict[str, Any]],
) -> dict[int, tuple[NetHeader, list[float]]]:
    if depth_count <= 0 or not (1 <= depth_a <= depth_count) or \
       not (1 <= depth_b <= depth_count) or depth_a == depth_b:
        raise ComparisonError("invalid NETRATE depth selection")
    by_pair: dict[tuple[int, int, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = pair_key(
            int(row["Z"]), int(row["ion"]) + 1,
            int(row["lower_level"]) + 1, int(row["upper_level"]) + 1,
        )
        by_pair.setdefault(key, []).append(row)
    matches: dict[int, tuple[NetHeader, list[float]]] = {}
    matched_cmfgen_ids: set[int] = set()
    for header, values in vectors(netrate, depth_count, netrate_header):
        if not isinstance(header, NetHeader):
            raise ComparisonError("unexpected NETRATE header type")
        raw_owner, _ = transition_owner(header.transition)
        try:
            z, stage, _ = parse_cmfgen_ion_id(raw_owner)
        except ValueError as exc:
            raise ComparisonError(
                f"NETRATE line {header.line_id}: invalid ion label"
            ) from exc
        candidates = by_pair.get(pair_key(
            z, stage, header.lower_full, header.upper_full
        ), [])
        if not candidates:
            continue
        inside = [
            row for row in candidates
            if abs(finite(row["nu"], "Lumina frequency") / 1.0e15 -
                   header.frequency_field) <= FREQUENCY_PRINT_HALF_UNIT
        ]
        if not inside:
            continue
        if len(inside) != 1:
            raise ComparisonError(
                f"CMFGEN line {header.line_id}: ambiguous Lumina frequency join"
            )
        line = int(inside[0]["line"])
        if line in matches:
            raise ComparisonError(
                f"Lumina line {line}: multiple CMFGEN transitions in print interval"
            )
        if header.line_id in matched_cmfgen_ids:
            raise ComparisonError(f"duplicate CMFGEN line id {header.line_id}")
        for depth in (depth_a, depth_b):
            if not math.isfinite(values[depth - 1]):
                raise ComparisonError(
                    f"CMFGEN line {header.line_id} depth {depth}: nonfinite ZNET"
                )
        matches[line] = (header, values)
        matched_cmfgen_ids.add(header.line_id)
    missing = sorted(int(row["line"]) for row in rows if int(row["line"]) not in matches)
    if missing:
        raise ComparisonError(
            f"NETRATE missing {len(missing)} unambiguous selected transitions: {missing[:10]}"
        )
    return matches


def interpolate(first: float, second: float, fraction: float) -> float:
    value = math.fsum((first, fraction * (second - first)))
    if not math.isfinite(value):
        raise ComparisonError("nonfinite coordinate interpolation")
    return value


def compare(
    summary_path: Path, netrate: Path, coordinate_path: Path,
    depth_count: int, depth_a: int, depth_b: int,
) -> dict[str, Any]:
    summary_document = load_json(summary_path, "Lumina saturation summary")
    coordinate_document = load_json(coordinate_path, "coordinate reference")
    if not netrate.is_file() or netrate.is_symlink() or netrate.stat().st_size == 0:
        raise ComparisonError(f"missing or unsafe NETRATE: {netrate}")
    rows = selected_rows(summary_document)
    fraction, coordinate = coordinate_contract(
        coordinate_document, netrate, depth_a, depth_b
    )
    matches = match_netrate(netrate, depth_count, depth_a, depth_b, rows)

    compared: list[dict[str, Any]] = []
    selected_emission = finite(
        summary_document["summary"]["selected_scaled_emission"],
        "selected emission",
    )
    if selected_emission <= 0.0:
        raise ComparisonError("nonpositive selected emission")
    tau_gt_one_emission: list[float] = []
    tau_le_one_emission: list[float] = []
    certified_below_emission: list[float] = []
    thick_certified_below_emission: list[float] = []
    certified_nonnegative_external_emission: list[float] = []
    certified_negative_external_emission: list[float] = []
    external_sign_indeterminate_emission: list[float] = []
    thin_cmf_above_lumina_emission: list[float] = []
    undefined_source_emission: list[float] = []
    nonpositive_tau_emission: list[float] = []

    for raw in rows:
        line = int(raw["line"])
        header, znet = matches[line]
        znet_a = znet[depth_a - 1]
        znet_b = znet[depth_b - 1]
        cmf_ratio_a = math.fsum((1.0, -znet_a))
        cmf_ratio_b = math.fsum((1.0, -znet_b))
        cmf_ratio_interp = interpolate(cmf_ratio_a, cmf_ratio_b, fraction)
        lumina_ratio = finite(raw["jbar_over_source"], "Lumina Jbar/source")
        local_term = finite(raw["one_minus_beta"], "Lumina one-minus-beta")
        beta = finite(raw["beta"], "Lumina beta")
        tau_effective = finite(raw["tau_effective"], "Lumina effective tau")
        emission = finite(raw["scaled_emission"], "Lumina scaled emission")
        if emission <= 0.0:
            raise ComparisonError(f"Lumina line {line}: nonpositive selected emission")

        source = raw.get("source_function")
        if source is None:
            ratio_bound = None
            proof_roundoff_bound = None
            certified_below = None
            certified_nonnegative_external = None
            certified_negative_external = None
            implied_continuum_over_source = None
            implied_continuum_over_source_lower = None
            implied_continuum_over_source_upper = None
            undefined_source_emission.append(emission)
        else:
            source_value = finite(source, "Lumina source function")
            if source_value <= 0.0:
                if tau_effective > 0.0:
                    raise ComparisonError(
                        f"Lumina line {line}: nonpositive source at positive tau"
                    )
                ratio_bound = None
                proof_roundoff_bound = None
                certified_below = None
                certified_nonnegative_external = None
                certified_negative_external = None
                implied_continuum_over_source = None
                implied_continuum_over_source_lower = None
                implied_continuum_over_source_upper = None
                nonpositive_tau_emission.append(emission)
            else:
                if tau_effective <= 0.0 or not (0.0 < beta <= 1.0) or \
                   not (0.0 <= local_term < 1.0):
                    raise ComparisonError(
                        f"Lumina line {line}: invalid positive-tau trapping domain"
                    )
                ratio_bound = finite(
                    raw["Jbar_absolute_bound"], "Lumina Jbar bound"
                ) / source_value
                proof_roundoff_bound = arithmetic_proof_bound(
                    lumina_ratio, ratio_bound, local_term, beta
                )
                upper_external_ratio = math.fsum((
                    lumina_ratio, ratio_bound, proof_roundoff_bound,
                    -local_term,
                ))
                lower_external_ratio = math.fsum((
                    lumina_ratio, -ratio_bound, -proof_roundoff_bound,
                    -local_term,
                ))
                implied_continuum_over_source = math.fsum((
                    lumina_ratio, -local_term,
                )) / beta
                implied_continuum_over_source_lower = \
                    lower_external_ratio / beta
                implied_continuum_over_source_upper = \
                    upper_external_ratio / beta
                if not all(math.isfinite(value) for value in (
                    ratio_bound, proof_roundoff_bound,
                    implied_continuum_over_source,
                    implied_continuum_over_source_lower,
                    implied_continuum_over_source_upper,
                )):
                    raise ComparisonError(
                        f"Lumina line {line}: nonfinite trapping proof"
                    )
                certified_below = upper_external_ratio < 0.0
                certified_negative_external = certified_below
                certified_nonnegative_external = lower_external_ratio >= 0.0
                if certified_below:
                    certified_below_emission.append(emission)
                    certified_negative_external_emission.append(emission)
                elif certified_nonnegative_external:
                    certified_nonnegative_external_emission.append(emission)
                else:
                    external_sign_indeterminate_emission.append(emission)
        if tau_effective > 1.0:
            tau_gt_one_emission.append(emission)
            if certified_below:
                thick_certified_below_emission.append(emission)
        elif tau_effective > 0.0:
            tau_le_one_emission.append(emission)
            if ratio_bound is not None and cmf_ratio_interp > lumina_ratio + ratio_bound:
                thin_cmf_above_lumina_emission.append(emission)

        compared.append({
            "lumina": raw,
            "cmfgen_mapping": {
                "line_id_one_based": header.line_id,
                "transition": header.transition,
                "frequency_1e15_hz_printed": header.frequency_field,
                "frequency_print_half_unit_1e15_hz": FREQUENCY_PRINT_HALF_UNIT,
                "lower_full_one_based": header.lower_full,
                "upper_full_one_based": header.upper_full,
                "identity": "EXACT_ION_AND_FULL_LEVELS_WITHIN_PRINTED_FREQUENCY_INTERVAL",
            },
            "cmfgen": {
                str(depth_a): {
                    "znet": znet_a,
                    "jbar_over_source_1_minus_znet": cmf_ratio_a,
                },
                str(depth_b): {
                    "znet": znet_b,
                    "jbar_over_source_1_minus_znet": cmf_ratio_b,
                },
                "shell_zero_velocity_interpolation": {
                    "fraction_from_depth_a_to_b": fraction,
                    "jbar_over_source": cmf_ratio_interp,
                },
            },
            "dimensionless_evidence": {
                "tau_effective_gt_one": tau_effective > 1.0,
                "lumina_jbar_over_source": lumina_ratio,
                "lumina_one_minus_beta": local_term,
                "lumina_local_term_minus_jbar_over_source": (
                    local_term - lumina_ratio
                ),
                "lumina_jbar_over_source_absolute_bound": ratio_bound,
                "arithmetic_proof_roundoff_bound": proof_roundoff_bound,
                "lumina_certified_below_one_minus_beta": certified_below,
                "lumina_certified_negative_external_continuum_component": (
                    certified_negative_external
                ),
                "lumina_certified_nonnegative_external_continuum_component": (
                    certified_nonnegative_external
                ),
                "lumina_implied_continuum_jbar_over_source": (
                    implied_continuum_over_source
                ),
                "lumina_implied_continuum_jbar_over_source_lower": (
                    implied_continuum_over_source_lower
                ),
                "lumina_implied_continuum_jbar_over_source_upper": (
                    implied_continuum_over_source_upper
                ),
                "cmfgen_interpolated_minus_lumina_jbar_over_source": (
                    cmf_ratio_interp - lumina_ratio
                ),
                "cmfgen_interpolated_to_lumina_ratio": (
                    cmf_ratio_interp / lumina_ratio
                    if lumina_ratio != 0.0 else None
                ),
            },
        })

    def evidence(values: list[float]) -> dict[str, Any]:
        value = math.fsum(values)
        return {
            "line_count": len(values),
            "selected_scaled_emission": value,
            "fraction_of_selected_emission": value / selected_emission,
        }

    return {
        "schema": "lumina-a210-cmfgen-line-saturation-comparison-v1",
        "status": "PASS",
        "verdict": "FINITE_TRANSITION_MATCH_DIAGNOSTIC_NOT_STATE_PARITY_NOT_CAUSE_CLAIM",
        "lumina_summary": str(summary_path.resolve()),
        "lumina_summary_sha256": digest(summary_path),
        "cmfgen_netrate": str(netrate.resolve()),
        "cmfgen_netrate_sha256": coordinate["reference_netrate_sha256"],
        "coordinate_reference": str(coordinate_path.resolve()),
        "coordinate_reference_sha256": digest(coordinate_path),
        "coordinate": coordinate,
        "matched_transition_count": len(compared),
        "selected_emission_evidence": {
            "tau_effective_gt_one": evidence(tau_gt_one_emission),
            "tau_effective_le_one": evidence(tau_le_one_emission),
            "certified_jbar_over_source_below_one_minus_beta": evidence(
                certified_below_emission
            ),
            "tau_gt_one_and_certified_below_local_term": evidence(
                thick_certified_below_emission
            ),
            "certified_nonnegative_external_continuum_component": evidence(
                certified_nonnegative_external_emission
            ),
            "certified_negative_external_continuum_component": evidence(
                certified_negative_external_emission
            ),
            "external_continuum_component_sign_indeterminate": evidence(
                external_sign_indeterminate_emission
            ),
            "tau_le_one_and_cmfgen_above_lumina_bound": evidence(
                thin_cmf_above_lumina_emission
            ),
            "source_function_undefined_chi_zero": evidence(
                undefined_source_emission
            ),
            "nonpositive_effective_tau_not_trapping_testable": evidence(
                nonpositive_tau_emission
            ),
        },
        "comparisons": compared,
        "same_transition": True,
        "same_velocity_coordinate_by_interpolation": True,
        "same_temperature": False,
        "same_population_and_radiation_state": False,
        "parity_claim": False,
        "physical_cause_claim": False,
        "arithmetic_proof_bound_is_physical_tolerance": False,
        "physical_values_modified": False,
        "floor": 0,
        "cap": 0,
        "clamp": 0,
        "jitter": 0,
        "repair": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--netrate", type=Path, required=True)
    parser.add_argument("--coordinate-reference", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--depth-count", type=int, default=90)
    parser.add_argument("--depth-a", type=int, default=67)
    parser.add_argument("--depth-b", type=int, default=68)
    args = parser.parse_args()
    try:
        payload = compare(
            args.summary, args.netrate, args.coordinate_reference,
            args.depth_count, args.depth_a, args.depth_b,
        )
        atomic_write(args.report, payload)
        print(
            "PASS A210_CMFGEN_LINE_SATURATION "
            f"matched={payload['matched_transition_count']} "
            "parity=0 cause_claim=0 repair=0"
        )
        return 0
    except (ComparisonError, OSError, UnicodeError, ValueError) as exc:
        atomic_write(args.report, {
            "schema": "lumina-a210-cmfgen-line-saturation-comparison-v1",
            "status": "FAIL",
            "error": str(exc),
        })
        print(f"FAIL A210_CMFGEN_LINE_SATURATION reason={exc}")
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
