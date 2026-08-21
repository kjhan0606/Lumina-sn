#!/usr/bin/env python3
"""Fail-closed log judge for the A100x2 A2-10 structural gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any


class GateError(RuntimeError):
    pass


KEY_VALUE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")
REPAIR_FIELD = re.compile(r"\b(floor|cap|clamp|jitter|repair)=([^\s]+)")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def indexed_lines(
    lines: list[str], prefix: str, count: int
) -> list[tuple[int, str, dict[str, str]]]:
    found = [
        (index, line, dict(KEY_VALUE.findall(line)))
        for index, line in enumerate(lines)
        if line.startswith(prefix)
    ]
    if len(found) != count:
        raise GateError(
            f"expected exactly {count} {prefix!r} lines, found {len(found)}"
        )
    return found


def only_line(lines: list[str], prefix: str) -> tuple[str, dict[str, str]]:
    (_, line, fields), = indexed_lines(lines, prefix, 1)
    return line, fields


def integer(fields: dict[str, str], key: str) -> int:
    try:
        return int(fields[key])
    except (KeyError, ValueError) as exc:
        raise GateError(f"missing/invalid integer {key}: {fields.get(key)!r}") from exc


def finite(fields: dict[str, str], key: str) -> float:
    try:
        value = float(fields[key])
    except (KeyError, ValueError) as exc:
        raise GateError(f"missing/invalid float {key}: {fields.get(key)!r}") from exc
    if not math.isfinite(value):
        raise GateError(f"non-finite {key}: {fields[key]!r}")
    return value


def require(fields: dict[str, str], **expected: str) -> None:
    for key, value in expected.items():
        if fields.get(key) != value:
            raise GateError(
                f"field mismatch {key}: got={fields.get(key)!r} expected={value!r}"
            )


def judge(args: argparse.Namespace) -> dict[str, Any]:
    stdout_text = args.stdout.read_text(encoding="utf-8", errors="strict")
    stderr_text = args.stderr.read_text(encoding="utf-8", errors="strict")
    combined = stdout_text + "\n" + stderr_text
    lines = combined.splitlines()

    forbidden = ("[FATAL]", "[BLOCKED]", "BLOCKED_", "DET_FLIGHT_FATAL")
    hits = [token for token in forbidden if token in combined]
    if hits:
        raise GateError(f"fatal/blocked markers present: {hits}")
    if "[A2-10][CANCELLATION-CENSUS]" in combined:
        raise GateError("cancellation census contaminated the non-census gate")

    repair_observations = 0
    for line in lines:
        for match in REPAIR_FIELD.finditer(line):
            # This record's unqualified ``cap`` is the exact solver's maximum
            # iteration count.  It is execution metadata, not a cap applied
            # to any physical value.  Physical cap observations on every
            # other record remain fail-closed below.
            if (match.group(1) == "cap" and
                    line.startswith("[cmf_fine][EXACT-MULTIGPU-EPOCH]")):
                continue
            repair_observations += 1
            if match.group(2).rstrip(",") != "0":
                raise GateError(
                    f"nonzero numerical repair field "
                    f"{match.group(1)}={match.group(2)}"
                )
    if repair_observations == 0:
        raise GateError("no numerical-repair audit fields were observed")

    # A2-INIT publishes R1 and one seed-material predictor.  Each requested
    # outer iteration then publishes one further exact/R6 epoch before its
    # R7 and physics-comparison commits.  The repair scan above intentionally
    # remains over the complete log for every iteration.
    outer_iterations = args.expected_outer_iterations
    publications = outer_iterations + 1
    exact_entries = indexed_lines(
        lines, "[cmf_fine][EXACT-MULTIGPU-EPOCH]", publications
    )
    for _, _, exact in exact_entries:
        require(
            exact,
            status="OK",
            devices=f"{args.expected_devices}/{args.expected_devices}",
            component_envelope="1",
            refinements=str(args.expected_refinements),
            failure_phase="0",
            failure_iteration="-1",
            floor="0",
            clamp="0",
            jitter="0",
        )
        exact_residual = finite(exact, "residual")
        exact_tolerance = finite(exact, "tolerance")
        if exact_residual > exact_tolerance:
            raise GateError(
                f"exact residual {exact_residual} exceeds tolerance {exact_tolerance}"
            )

    material_entries = indexed_lines(
        lines, "[cmf_fine][SIGNED-MATERIAL-CENSUS]", publications
    )
    for _, _, material in material_entries:
        require(
            material,
            raw_preserved="1",
            floor="0",
            clamp="0",
            jitter="0",
        )
    require(material_entries[0][2],
        line_shells=str(args.expected_r1_signed_cells),
        exact_zero_tau=str(args.expected_r1_exact_zero_tau),
        raw_negative="0",
        mild_negative="0",
        srce_chk="0",
    )
    require(material_entries[1][2],
        line_shells=str(args.expected_r2_signed_cells),
        exact_zero_tau=str(args.expected_r2_exact_zero_tau),
        raw_negative=str(args.expected_r2_raw_negative),
        mild_negative=str(args.expected_r2_mild_negative),
        srce_chk=str(args.expected_r2_srce_chk),
    )
    if (
        args.expected_r2_mild_negative + args.expected_r2_srce_chk
        != args.expected_r2_raw_negative
    ):
        raise GateError("invalid pre-registered R2 negative-opacity partition")
    for material_index, (_, _, material) in enumerate(material_entries):
        if (
            integer(material, "line_shells")
            + integer(material, "exact_zero_tau")
            != args.expected_valid_cells
        ):
            raise GateError("material census does not cover the complete Q_E slab")
        if material_index >= 2:
            raw_negative = integer(material, "raw_negative")
            mild_negative = integer(material, "mild_negative")
            srce_chk = integer(material, "srce_chk")
            if min(raw_negative, mild_negative, srce_chk) < 0:
                raise GateError("negative signed-material census count")
            if mild_negative + srce_chk != raw_negative:
                raise GateError(
                    "iteration signed-material negative-opacity partition mismatch"
                )

    policy_entries = indexed_lines(
        lines, "[cmf_fine][SIGNED-MATERIAL-POLICY]", publications
    )
    require(
        policy_entries[0][2],
        operator="INIT_SHARED_GAUSSIAN",
        srce_chk_expected="0",
        srce_chk_material="0",
        raw_preserved="1",
        floor="0",
        clamp="0",
        jitter="0",
        repair="0",
    )
    require(
        policy_entries[1][2],
        operator="CMFGEN_NONOVERLAP_SOBOLEV",
        srce_chk_expected=str(args.expected_r2_srce_chk),
        srce_chk_material=str(args.expected_r2_srce_chk),
        raw_preserved="1",
        floor="0",
        clamp="0",
        jitter="0",
        repair="0",
    )
    for material_entry, policy_entry in zip(
        material_entries[2:], policy_entries[2:]
    ):
        material = material_entry[2]
        policy = policy_entry[2]
        require(
            policy,
            operator="CMFGEN_NONOVERLAP_SOBOLEV",
            srce_chk_expected=material["srce_chk"],
            srce_chk_material=material["srce_chk"],
            raw_preserved="1",
            floor="0",
            clamp="0",
            jitter="0",
            repair="0",
        )

    sobolev_entries = indexed_lines(
        lines, "[cmf_fine][SOBOLEV-LINE-OPERATOR]", outer_iterations
    )
    for iteration, (_, _, sobolev) in enumerate(sobolev_entries):
        material = material_entries[iteration + 1][2]
        require(
            sobolev,
            status="PASS",
            mode="CMFGEN_NONOVERLAP_HOMOLOGY_SIGMA0",
            continuum_sampling="GAUSSIAN_PROFILE",
            jbar_cells=str(args.expected_valid_cells),
            raw_negative=material["raw_negative"],
            mild_negative=material["mild_negative"],
            srce_chk_expected=material["srce_chk"],
            srce_chk_applied=material["srce_chk"],
            all_jbar_finite="1",
            raw_preserved="1",
            floor="0",
            cap="0",
            clamp="0",
            jitter="0",
            repair="0",
        )
        beta_min = finite(sobolev, "beta_min")
        beta_max = finite(sobolev, "beta_max")
        if beta_min <= 0.0 or beta_max < beta_min:
            raise GateError(
                f"invalid Sobolev beta range [{beta_min}, {beta_max}]"
            )

    identity_entries = indexed_lines(
        lines, "[R6][LINE-IDENTITY]", publications
    )
    for _, _, identity in identity_entries:
        require(
            identity,
            lane="DET",
            component_envelope="1",
            refinements=str(args.expected_refinements),
        )
        identity_residual = finite(identity, "exact_residual")
        identity_tolerance = finite(identity, "exact_tolerance")
        if identity_residual > identity_tolerance:
            raise GateError("R6 exact residual exceeds its sealed tolerance")
        for key in ("q_set_hash", "e_set_hash", "domain_hash", "profile_hash"):
            if not SHA256.fullmatch(identity.get(key, "")):
                raise GateError(f"invalid R6 {key}")
    for key in ("q_set_hash", "e_set_hash", "domain_hash", "profile_hash"):
        before = identity_entries[0][2][key]
        for publication, (_, _, identity) in enumerate(
            identity_entries[1:], start=2
        ):
            after = identity[key]
            if before != after:
                raise GateError(
                    f"R1/R{publication} line identity changed for {key}: "
                    f"{before} != {after}"
                )

    coverage_entries = indexed_lines(
        lines, "[R6][LINE-COVERAGE]", publications
    )
    expected = {
        "q_lines": args.expected_q_lines,
        "e_lines": args.expected_e_lines,
        "valid_lines": args.expected_e_lines,
        "partial_lines": 0,
        "unsampled_lines": 0,
        "valid_cells": args.expected_valid_cells,
        "exact_zero_cells": 0,
    }
    for _, _, coverage in coverage_entries:
        for key, value in expected.items():
            observed = integer(coverage, key)
            if observed != value:
                raise GateError(f"R6 {key}={observed}, expected {value}")
    coverage_generations = [
        integer(coverage, "generation") for _, _, coverage in coverage_entries
    ]
    expected_generations = list(range(1, publications + 1))
    if coverage_generations != expected_generations:
        raise GateError(
            f"R6 radiation generations {coverage_generations}, "
            f"expected {expected_generations}"
        )

    predictor_index, predictor_line, predictor = indexed_lines(
        lines,
        "[A2-INIT][SEED-MATERIAL] event=INIT_SEED_MATERIAL_PREDICTOR",
        1,
    )[0]
    require(
        predictor,
        lane="DET",
        r1_generation="1",
        te_manifest_preserved="1",
        te_publication_preserved="1",
        floor="0",
        cap="0",
        clamp="0",
        jitter="0",
        repair="0",
    )
    te_transition = predictor.get("te_generation", "")
    if "->" not in te_transition:
        raise GateError("predictor commit lacks a Te generation transition")
    te_before, _, te_after = te_transition.partition("->")
    if te_before != te_after:
        raise GateError(
            f"predictor changed the Te generation: {te_transition!r}"
        )
    population_transition = predictor.get("population_generation", "")
    pop_before, arrow, pop_after = population_transition.partition("->")
    try:
        if arrow != "->" or int(pop_after) != int(pop_before) + 1:
            raise GateError(
                "predictor population generation did not advance exactly once: "
                f"{population_transition!r}"
            )
    except ValueError as exc:
        raise GateError(
            f"invalid predictor population transition: {population_transition!r}"
        ) from exc

    r7_entries = indexed_lines(
        lines, "[R7][PHASE] event=R7_MATERIAL_PHASE_COMMITTED",
        outer_iterations
    )
    comparison_entries = indexed_lines(
        lines, "[PHYSICS-COMPARISON] lane=DET", outer_iterations
    )
    for iteration, ((_, r7_line, r7), (_, comparison_line, _)) in enumerate(
        zip(r7_entries, comparison_entries)
    ):
        require(
            r7,
            lane="DET",
            iter=str(iteration),
            phase="A2-10",
            te_generation=f"{iteration + 1}->{iteration + 2}",
        )
        if "status=COMMITTED" not in comparison_line:
            raise GateError(f"invalid physics comparison commit: {comparison_line}")
        comparison = comparison_entries[iteration][2]
        require(comparison, lane="DET", iter=str(iteration), status="COMMITTED")

    ordering_values = [coverage_entries[0][0], predictor_index]
    for iteration in range(outer_iterations):
        ordering_values.extend((
            material_entries[iteration + 1][0],
            exact_entries[iteration + 1][0],
            sobolev_entries[iteration][0],
            coverage_entries[iteration + 1][0],
            r7_entries[iteration][0],
            comparison_entries[iteration][0],
        ))
    ordering = tuple(ordering_values)
    if list(ordering) != sorted(ordering) or len(set(ordering)) != len(ordering):
        raise GateError(
            "R1 -> predictor -> R2 -> R7 -> comparison ordering violated: "
            f"line indices {ordering}"
        )

    return {
        "schema": "LUMINA_A210_TARGETED_GATE_V3",
        "status": "PASS",
        "mode": "A210_TARGETED_GATE",
        "expected_outer_iterations": outer_iterations,
        "expected_devices": args.expected_devices,
        "expected_refinements": args.expected_refinements,
        "stdout_sha256": sha256(args.stdout),
        "stderr_sha256": sha256(args.stderr),
        "repair_audit_observations": repair_observations,
        "exact_publications": publications,
        "exact": [
            {
                "iterations": integer(exact, "iterations"),
                "residual": finite(exact, "residual"),
                "tolerance": finite(exact, "tolerance"),
                "devices": exact["devices"],
            }
            for _, _, exact in exact_entries
        ],
        "r6": expected,
        "r6_radiation_generations": coverage_generations,
        "seed_material_predictor_commit": True,
        "seed_material_predictor_te_transition": te_transition,
        "seed_material_predictor_population_transition": population_transition,
        "r2_signed_material": {
            "signed_cells": args.expected_r2_signed_cells,
            "exact_zero_tau": args.expected_r2_exact_zero_tau,
            "raw_negative": args.expected_r2_raw_negative,
            "mild_negative": args.expected_r2_mild_negative,
            "srce_chk": args.expected_r2_srce_chk,
        },
        "line_operator": "CMFGEN_NONOVERLAP_HOMOLOGY_SIGMA0",
        "sobolev_jbar_cells": args.expected_valid_cells,
        "r7_material_commit": True,
        "physics_comparison_commit": True,
        "cancellation_census_present": False,
        "initialization_material_changed": True,
        "physical_values_modified_by_numerical_repair": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stdout", type=Path, required=True)
    parser.add_argument("--stderr", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-devices", type=int, default=2)
    parser.add_argument("--expected-outer-iterations", type=int, default=1)
    parser.add_argument("--expected-refinements", type=int, required=True)
    parser.add_argument("--expected-q-lines", type=int, default=1_391_131)
    parser.add_argument("--expected-e-lines", type=int, default=2_180_286)
    parser.add_argument("--expected-valid-cells", type=int, default=109_014_300)
    parser.add_argument("--expected-r1-signed-cells", type=int, default=27_748_410)
    parser.add_argument("--expected-r1-exact-zero-tau", type=int, default=81_265_890)
    parser.add_argument("--expected-r2-signed-cells", type=int, default=22_866_166)
    parser.add_argument("--expected-r2-exact-zero-tau", type=int, default=86_148_134)
    parser.add_argument("--expected-r2-raw-negative", type=int, default=4_246_581)
    parser.add_argument("--expected-r2-mild-negative", type=int, default=4_246_577)
    parser.add_argument("--expected-r2-srce-chk", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.expected_devices != 2:
            raise GateError("the sealed targeted gate requires exactly two devices")
        if not 1 <= args.expected_outer_iterations <= 64:
            raise GateError("expected outer iterations must be in 1..64")
        if not 1 <= args.expected_refinements <= 64:
            raise GateError("expected refinements must be in 1..64")
        for path in (args.stdout, args.stderr):
            if not path.is_file() or path.is_symlink():
                raise GateError(f"missing or unsafe log: {path}")
        report = judge(args)
    except (GateError, OSError, UnicodeError) as exc:
        report = {
            "schema": "LUMINA_A210_TARGETED_GATE_V3",
            "status": "FAIL",
            "error": str(exc),
        }
        atomic_write_json(args.report, report)
        print(f"A210_TARGETED_GATE_FAIL {exc}", file=sys.stderr)
        return 4
    atomic_write_json(args.report, report)
    print(
        "A210_TARGETED_GATE_PASS "
        f"devices={args.expected_devices} refinements={args.expected_refinements} "
        f"report={args.report} floor=0 cap=0 clamp=0 jitter=0 repair=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
