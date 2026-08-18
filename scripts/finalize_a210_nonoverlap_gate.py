#!/usr/bin/env python3
"""Fail-closed completion audit for the A2-10 non-overlap Sobolev gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable


class AuditError(RuntimeError):
    pass


REPAIR_KEYS = ("floor", "cap", "clamp", "jitter", "repair")
ZERO_ENV = (
    "LUMINA_NLTE_LTE_FLOOR",
    "LUMINA_NLTE_FLOOR_MODE",
    "LUMINA_NLTE_FLOOR_REG",
    "LUMINA_NLTE_BK_CEIL",
    "LUMINA_NLTE_INV_CEIL",
    "LUMINA_NLTE_COLL_FLOOR",
    "LUMINA_DR_FLOOR_CMS",
    "LUMINA_STAGE4_BK_CAP",
    "LUMINA_HRESP_CLAMP",
    "LUMINA_TE_STEP_CLAMP",
    "LUMINA_J_CAP_FACTOR",
    "LUMINA_J_FLOOR_FACTOR",
    "LUMINA_RADEQ_LINE_CULL",
    "LUMINA_NLTE_GREY_TAU",
    "LUMINA_NLTE_ASSEMBLE_GPU",
    "LUMINA_NLTE_FALLBACK_TE",
)
KV = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")
FOUR_PI = 12.56637061435917295385057353311801153679
PROOF_WITNESS_MARKER = "[A2-10][LINE-NET-CELL-FINITE]"
EXPECTED_PROOF_WITNESSES = {
    ("LOWER", 1_154_618, 5),
    ("UPPER", 894_169, 27),
}
EXPECTED_PROOF_FIELDS = {
    ("[cmf_fine][EXACT-MULTIGPU-EPOCH]", "component_error"),
    ("[cmf_fine][EXACT-MULTIGPU-EPOCH]", "refinements"),
    ("[R6][LINE-IDENTITY]", "component_error"),
    ("[R6][LINE-IDENTITY]", "profile_error"),
    ("[R6][LINE-IDENTITY]", "refinements"),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def regular_file(path: Path, label: str) -> Path:
    require(path.is_file() and not path.is_symlink(), f"missing or unsafe {label}: {path}")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path, label: str) -> dict[str, Any]:
    regular_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8", errors="strict"))
    except (json.JSONDecodeError, UnicodeError, OSError) as exc:
        raise AuditError(f"invalid {label}: {path}: {exc}") from exc
    require(isinstance(value, dict), f"{label} is not a JSON object: {path}")
    return value


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def numeric_zero(value: Any) -> bool:
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value == 0
    if isinstance(value, str):
        try:
            return float(value) == 0.0
        except ValueError:
            return False
    return False


def audit_repair_fields(value: Any, location: str = "$") -> int:
    observations = 0
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{location}.{key}"
            if key in REPAIR_KEYS:
                observations += 1
                require(numeric_zero(item), f"nonzero numerical repair at {child}: {item!r}")
            observations += audit_repair_fields(item, child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            observations += audit_repair_fields(item, f"{location}[{index}]")
    return observations


def parse_exports(path: Path) -> dict[str, str]:
    exports: dict[str, str] = {}
    pattern = re.compile(r'^declare -x ([A-Za-z_][A-Za-z0-9_]*)="(.*)"$')
    for line in regular_file(path, "resolved environment").read_text(
        encoding="utf-8", errors="strict"
    ).splitlines():
        match = pattern.fullmatch(line)
        require(match is not None, f"malformed resolved export: {line!r}")
        name, value = match.groups()
        require(name not in exports, f"duplicate resolved export: {name}")
        exports[name] = value
    return exports


def verify_hash_manifest(manifest: Path, root: Path, label: str) -> dict[str, str]:
    regular_file(manifest, f"{label} manifest")
    require(root.is_dir() and not root.is_symlink(), f"missing or unsafe {label} root: {root}")
    observed: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8", errors="strict").splitlines():
        parts = line.split(maxsplit=1)
        require(len(parts) == 2, f"malformed {label} hash line: {line!r}")
        expected, relative = parts
        relative = relative.lstrip("*")
        require(re.fullmatch(r"[0-9a-f]{64}", expected) is not None,
                f"malformed {label} digest: {expected!r}")
        candidate = root / relative
        regular_file(candidate, f"{label} member")
        actual = sha256(candidate)
        require(actual == expected, f"{label} SHA mismatch: {relative}")
        observed[relative] = actual
    require(bool(observed), f"empty {label} manifest")
    return observed


def csv_data_rows(path: Path) -> int:
    regular_file(path, "cancellation census CSV")
    with path.open("r", encoding="utf-8", errors="strict", newline="") as stream:
        rows = list(csv.reader(stream))
    require(bool(rows), f"empty cancellation census CSV: {path}")
    require(rows[0][:4] == ["phase", "line", "shell", "status"],
            f"unexpected cancellation census header: {path}")
    return len(rows) - 1


def audit_refinement(path: Path) -> dict[str, Any]:
    report = load_json(path, "K12-to-K18 comparison")
    require(report.get("schema") == "lumina-a2-10-refinement-only-comparison-v1",
            "unexpected K12-to-K18 schema")
    require(report.get("status") == "PASS", "K12-to-K18 comparison did not pass")
    require(report.get("physical_values_modified") is False,
            "K12-to-K18 comparison modified physical values")
    require(report.get("surviving_count") == 0 and report.get("surviving") == [],
            "K18 cancellation census still has unresolved witnesses")
    repair_observations = audit_repair_fields(report, "$.refinement")
    require(repair_observations >= 4, "K12-to-K18 report lacks repair audit fields")
    baseline = report.get("baseline")
    candidate = report.get("candidate")
    require(isinstance(baseline, dict) and isinstance(candidate, dict),
            "K12-to-K18 report lacks endpoints")
    require((baseline.get("refinements"), baseline.get("unresolved")) == (12, 19),
            "unexpected K12 cancellation endpoint")
    require((candidate.get("refinements"), candidate.get("unresolved")) == (18, 0),
            "unexpected K18 cancellation endpoint")
    endpoints: dict[str, Any] = {}
    for name, endpoint in (("baseline", baseline), ("candidate", candidate)):
        csv_path = Path(str(endpoint.get("csv", "")))
        expected_sha = endpoint.get("sha256")
        regular_file(csv_path, f"{name} cancellation CSV")
        actual_sha = sha256(csv_path)
        require(actual_sha == expected_sha, f"{name} cancellation CSV SHA mismatch")
        rows = csv_data_rows(csv_path)
        require(rows == endpoint.get("unresolved"),
                f"{name} unresolved count differs from CSV rows")
        endpoints[name] = {
            "path": str(csv_path.resolve()),
            "sha256": actual_sha,
            "refinements": endpoint["refinements"],
            "unresolved": rows,
        }
    return {
        "status": "PASS",
        "report": str(path.resolve()),
        "report_sha256": sha256(path),
        "endpoints": endpoints,
        "repair_audit_observations": repair_observations,
    }


def audit_log_repairs(paths: Iterable[Path]) -> int:
    observations = 0
    for path in paths:
        text = regular_file(path, "model log").read_text(
            encoding="utf-8", errors="strict"
        )
        for line_number, line in enumerate(text.splitlines(), start=1):
            for key, value in KV.findall(line):
                if key in REPAIR_KEYS:
                    # ``cap`` on the exact multi-GPU record is the solver's
                    # iteration limit.  It never changes a physical value and
                    # must not be conflated with a numerical value cap.
                    if (key == "cap" and line.startswith(
                            "[cmf_fine][EXACT-MULTIGPU-EPOCH]")):
                        continue
                    observations += 1
                    require(numeric_zero(value),
                            f"nonzero {key} in {path}:{line_number}: {value}")
    require(observations > 0, "model logs contain no repair audit observations")
    return observations


def finite_float(fields: dict[str, str], name: str, label: str) -> float:
    try:
        value = float(fields[name])
    except (KeyError, ValueError, OverflowError) as exc:
        raise AuditError(f"invalid {name} in {label}") from exc
    require(math.isfinite(value), f"non-finite {name} in {label}")
    return value


def proof_bound_pair(value: Any, label: str) -> tuple[float, float]:
    require(isinstance(value, str), f"missing proof bound {label}")
    match = re.fullmatch(r"\[([^,]+),([^\]]+)\]", value)
    require(match is not None, f"malformed proof bound {label}: {value!r}")
    try:
        lower, upper = (float(item) for item in match.groups())
    except (ValueError, OverflowError) as exc:
        raise AuditError(f"non-numeric proof bound {label}: {value!r}") from exc
    require(math.isfinite(lower) and math.isfinite(upper) and
            lower >= 0.0 and upper >= lower,
            f"invalid proof bound {label}: {value!r}")
    return lower, upper


def audit_proof_reference_changes(
    changes: Any,
    expected_refinements: int,
) -> list[dict[str, Any]]:
    require(isinstance(changes, list) and len(changes) == len(EXPECTED_PROOF_FIELDS),
            "R1 proof-field change cardinality differs")
    observed: dict[tuple[str, str], dict[str, Any]] = {}
    for item in changes:
        require(isinstance(item, dict), "malformed R1 proof-field change")
        key = (str(item.get("record")), str(item.get("field")))
        require(key in EXPECTED_PROOF_FIELDS and key not in observed,
                f"unexpected or duplicate R1 proof-field change: {key}")
        if key[1] == "refinements":
            require(item.get("reference") == "24" and
                    item.get("candidate") == str(expected_refinements) and
                    set(item) == {"record", "field", "reference", "candidate"},
                    f"R1 proof refinement change differs: {key}")
        else:
            reference_pair = proof_bound_pair(item.get("reference"), f"reference {key}")
            candidate_pair = proof_bound_pair(item.get("candidate"), f"candidate {key}")
            require(candidate_pair[0] <= reference_pair[0] and
                    candidate_pair[1] < reference_pair[1],
                    f"R1 proof bound did not contract: {key}")
            ratio = item.get("upper_bound_ratio")
            require(isinstance(ratio, (int, float)) and not isinstance(ratio, bool) and
                    math.isfinite(float(ratio)) and
                    float(ratio) == candidate_pair[1] / reference_pair[1],
                    f"R1 proof-bound ratio differs: {key}")
            require(set(item) == {
                "record", "field", "reference", "candidate", "upper_bound_ratio"
            }, f"R1 proof-bound change has unauthorized fields: {key}")
        observed[key] = item
    require(set(observed) == EXPECTED_PROOF_FIELDS,
            "R1 proof-field change set differs")
    return [observed[key] for key in sorted(observed)]


def audit_proof_witnesses(run_root: Path, baseline_path: Path) -> dict[str, Any]:
    baseline = load_json(baseline_path, "K24 local proof-witness audit")
    require(baseline.get("schema") ==
            "lumina-a2-10-cancellation-witness-audit-v1" and
            baseline.get("status") == "PASS",
            "K24 local proof-witness audit did not pass")
    require(baseline.get("witness_count") == 2 and
            {tuple(item) for item in baseline.get("observed_line_shell", [])} ==
            {(line, shell) for _, line, shell in EXPECTED_PROOF_WITNESSES},
            "K24 local proof-witness set differs")
    require(baseline.get("physical_values_modified") is False,
            "K24 proof-witness audit modified physical values")
    baseline_repairs = audit_repair_fields(baseline, "$.proof_baseline")
    require(baseline_repairs >= 4,
            "K24 proof-witness audit lacks repair observations")
    source_log = Path(str(baseline.get("source_log", "")))
    regular_file(source_log, "K24 proof-witness source log")
    require(baseline.get("source_sha256") == sha256(source_log),
            "K24 proof-witness source log changed")

    witness_objects = baseline.get("witnesses")
    require(isinstance(witness_objects, list) and len(witness_objects) == 2,
            "K24 proof-witness records differ")
    baseline_by_key: dict[tuple[str, int, int], dict[str, Any]] = {}
    for witness in witness_objects:
        require(isinstance(witness, dict), "malformed K24 proof witness")
        key = (str(witness.get("phase")), int(witness.get("line", -1)),
               int(witness.get("shell", -1)))
        require(key in EXPECTED_PROOF_WITNESSES and key not in baseline_by_key,
                f"unexpected or duplicate K24 proof witness: {key}")
        require(witness.get("status") == "UNRESOLVED_CANCELLATION",
                f"K24 witness was not fail-closed: {key}")
        baseline_by_key[key] = witness
    require(set(baseline_by_key) == EXPECTED_PROOF_WITNESSES,
            "K24 phase-qualified proof-witness set differs")

    candidate_log = regular_file(run_root / "stderr.log", "K30 model stderr")
    candidate_by_key: dict[tuple[str, int, int], list[dict[str, str]]] = {
        key: [] for key in EXPECTED_PROOF_WITNESSES
    }
    for line in candidate_log.read_text(encoding="utf-8", errors="strict").splitlines():
        if not line.startswith(PROOF_WITNESS_MARKER):
            continue
        fields = dict(KV.findall(line))
        try:
            key = (fields["phase"], int(fields["line"]), int(fields["shell"]))
        except (KeyError, ValueError) as exc:
            raise AuditError("malformed finite proof-witness record") from exc
        if key in candidate_by_key:
            candidate_by_key[key].append(fields)

    results: list[dict[str, Any]] = []
    for key in sorted(EXPECTED_PROOF_WITNESSES):
        records = candidate_by_key[key]
        require(len(records) == 1,
                f"expected exactly one finite K30 proof witness {key}; got {len(records)}")
        fields = records[0]
        baseline_witness = baseline_by_key[key]
        inputs = baseline_witness.get("inputs")
        reconstructed = baseline_witness.get("reconstructed")
        requirement = baseline_witness.get("proof_requirement")
        require(isinstance(inputs, dict) and isinstance(reconstructed, dict) and
                isinstance(requirement, dict), f"incomplete K24 witness {key}")
        label = f"K30 witness {key}"
        eta = finite_float(fields, "eta_per_sr", label)
        chi = finite_float(fields, "chi_effective", label)
        jbar = finite_float(fields, "Jbar", label)
        bound = finite_float(fields, "Jbar_local_bound", label)
        deck_scale = finite_float(fields, "deck_scale", label)
        signed_rate = finite_float(fields, "signed_rate", label)
        uncertainty = finite_float(fields, "uncertainty", label)

        require(eta == float(inputs["eta_per_sr"]) and
                chi == float(inputs["chi_effective"]) and
                jbar == float(inputs["jbar"]) and
                deck_scale == float(inputs["deck_scale"]) and
                signed_rate == float(reconstructed["signed_rate"]),
                f"physical value changed from K24 at {key}")
        factor = FOUR_PI * deck_scale
        calculated_rate = math.fma(-chi, jbar, eta) * factor
        calculated_uncertainty = math.fma(abs(chi), bound, 0.0) * factor
        require(calculated_rate == signed_rate and
                calculated_uncertainty == uncertainty,
                f"K30 witness identity differs at {key}")
        required_bound = float(
            requirement["required_symmetric_jbar_bound_strictly_below"]
        )
        require(bound < required_bound and uncertainty < abs(signed_rate),
                f"K30 local proof bound did not resolve {key}")
        expected_status = "OK_COOLING" if signed_rate > 0.0 else "OK_HEATING"
        require(fields.get("status") == expected_status and
                fields.get("requested_cell") == "0",
                f"K30 proof witness status/provenance differs at {key}")
        results.append({
            "phase": key[0],
            "line": key[1],
            "shell": key[2],
            "status": fields["status"],
            "physical_values_bit_exact_to_k24": True,
            "jbar_absolute_uncertainty": bound,
            "required_jbar_bound": required_bound,
            "bound_to_required_ratio": bound / required_bound,
            "signed_rate_to_uncertainty_ratio":
                abs(signed_rate) / uncertainty if uncertainty > 0.0 else math.inf,
        })
    return {
        "status": "PASS",
        "baseline": str(baseline_path.resolve()),
        "baseline_sha256": sha256(baseline_path),
        "baseline_source_sha256": baseline["source_sha256"],
        "candidate_log_sha256": sha256(candidate_log),
        "witnesses": results,
        "physical_values_modified": False,
        "repair_audit_observations": baseline_repairs,
    }


def audit_run(run_root: Path, expected_refinements: int) -> dict[str, Any]:
    require(run_root.is_dir() and not run_root.is_symlink(),
            f"missing or unsafe run root: {run_root}")
    input_dir = run_root / "input"
    manual = run_root / "manual_control"
    require(regular_file(run_root / "model.rc", "model return code").read_text().strip() == "0",
            "model did not exit with rc=0")
    verdict = regular_file(run_root / "TARGETED_GATE_VERDICT.txt", "targeted verdict")
    require("A210_TARGETED_GATE_ACCEPT status=PASS" in verdict.read_text(),
            "targeted gate verdict is not PASS")

    targeted = load_json(run_root / "a210_targeted_gate_report.json", "targeted gate report")
    require(targeted.get("schema") == "LUMINA_A210_TARGETED_GATE_V3" and
            targeted.get("status") == "PASS", "targeted log gate did not pass")
    require(targeted.get("expected_devices") == 2 and
            targeted.get("expected_refinements") == expected_refinements,
            "targeted hardware/refinement contract differs")
    require(targeted.get("exact_publications") == 2 and
            targeted.get("r6_radiation_generations") == [1, 2],
            "targeted coevolution generation barrier differs")
    require(targeted.get("seed_material_predictor_commit") is True,
            "seed-material predictor commit is missing")
    require(targeted.get("line_operator") == "CMFGEN_NONOVERLAP_HOMOLOGY_SIGMA0" and
            targeted.get("sobolev_jbar_cells") == 109_014_300,
            "non-overlap Sobolev line operator coverage differs")
    require(targeted.get("r7_material_commit") is True and
            targeted.get("physics_comparison_commit") is True,
            "R7/comparison commit is missing")
    require(targeted.get("cancellation_census_present") is False,
            "non-census gate was contaminated by census mode")
    require(targeted.get("physical_values_modified_by_numerical_repair") is False,
            "targeted gate reports numerical repair")
    r2 = targeted.get("r2_signed_material")
    require(r2 == {
        "signed_cells": 22_866_166,
        "exact_zero_tau": 86_148_134,
        "raw_negative": 4_246_581,
        "mild_negative": 4_246_577,
        "srce_chk": 4,
    }, "R2 signed-material census differs")
    targeted_repairs = audit_repair_fields(targeted, "$.targeted")

    snapshot = load_json(run_root / "a210_targeted_snapshot_report.json",
                         "targeted snapshot report")
    require(snapshot.get("schema") == "LUMINA_DET_CONVERGENCE_V1" and
            snapshot.get("status") == "CONVERGED", "targeted snapshot gate did not pass")
    require(snapshot.get("expected_iterations") == 1 and
            snapshot.get("tail_transitions") == 0 and
            snapshot.get("expected_bins") == 1234 and
            snapshot.get("transitions") == [], "targeted snapshot contract differs")

    reference = load_json(run_root / "r1_k24_reference_comparison.json",
                          "R1 reference comparison")
    require(reference.get("schema") == "LUMINA_A210_TARGETED_REFERENCE_COMPARISON_V1" and
            reference.get("status") == "PASS", "R1 reference comparison did not pass")
    proof_changes_verified = 0
    if expected_refinements == 24:
        require(reference.get("reason") == "EXACT_AND_R6_FIELDS_BIT_EXACT" and
                reference.get("comparison_mode") in (None, "STRICT_BIT_EXACT") and
                reference.get("differences") == [],
                "R1 is not bit-exact to its sealed reference")
        r1_identity = "BIT_EXACT"
    else:
        proof_changes = audit_proof_reference_changes(
            reference.get("proof_field_changes"), expected_refinements
        )
        proof_changes_verified = len(proof_changes)
        require(
            reference.get("reason") ==
            "PHYSICAL_AND_SOLVER_FIELDS_BIT_EXACT_PROOF_BOUNDS_CONTRACTED" and
            reference.get("comparison_mode") == "PROOF_REFINEMENT_ONLY" and
            reference.get("reference_refinements") == 24 and
            reference.get("candidate_refinements") == expected_refinements and
            reference.get("proof_bounds_nonincreasing") is True and
            reference.get("differences") == [],
            "R1 physical/solver identity or proof-bound contraction differs",
        )
        r1_identity = "PHYSICAL_SOLVER_BIT_EXACT_PROOF_BOUNDS_CONTRACTED"
    require(reference.get("reference_occurrence") == 0 and
            reference.get("candidate_occurrence") == 0,
            "R1 comparison did not select the first pass")
    candidate_stderr = Path(str(reference.get("candidate_stderr", "")))
    require(candidate_stderr.resolve() == (run_root / "stderr.log").resolve(),
            "R1 comparison candidate is not this run")
    require(reference.get("candidate_sha256") == sha256(candidate_stderr),
            "R1 comparison candidate log changed after comparison")
    reference_stderr = Path(str(reference.get("reference_stderr", "")))
    regular_file(reference_stderr, "R1 sealed reference")
    require(reference.get("reference_sha256") == sha256(reference_stderr),
            "R1 sealed reference changed after comparison")
    reference_repairs = audit_repair_fields(reference, "$.reference")

    require((manual / "COMPLETED").is_file(), "tripwire supervisor did not complete")
    require(not (manual / "FAILED").exists() and not (manual / "YIELDED").exists(),
            "tripwire supervisor failed or yielded")
    require(regular_file(manual / "child.rc", "tripwire child return code").read_text().strip() == "0",
            "tripwire child return code is not zero")
    supervisor_text = regular_file(manual / "supervisor.log", "tripwire log").read_text(
        encoding="utf-8", errors="strict"
    )
    require(supervisor_text.count("status=START") == 1 and
            supervisor_text.count("status=CHILD_STARTED") == 1 and
            supervisor_text.count("status=COMPLETED child_rc=0") == 1,
            "tripwire lifecycle differs")
    require("action=YIELD" not in supervisor_text and "status=FAILED" not in supervisor_text and
            "FATAL" not in supervisor_text, "tripwire reports a conflict or failure")
    preflights = [line for line in supervisor_text.splitlines() if "gpu_preflight=" in line]
    require(len(preflights) == 2 and all("A100" in line for line in preflights),
            "tripwire did not preflight two A100 GPUs")

    footer = dict(KV.findall(regular_file(run_root / "RUN_FOOTER.txt", "run footer").read_text()))
    require(footer.get("diagnostic_mode") == "A210_TARGETED_GATE" and
            footer.get("outer_iterations") == "1" and
            footer.get("envelope_refinements") == str(expected_refinements) and
            footer.get("LUMINA_CMF_FINE_MGPU_DEVICES") == "2",
            "run footer differs from the targeted A100x2 contract")
    require(regular_file(input_dir / "precore_tau_refresh.txt", "pre-core refresh seal")
            .read_text().strip() == "0", "rejected pre-core tau refresh was enabled")
    exports = parse_exports(input_dir / "resolved_lumina.exports")
    for name in ZERO_ENV:
        require(exports.get(name) == "0", f"numerical repair env is not zero: {name}")
    require(exports.get("LUMINA_CMF_FINE_MGPU_DEVICES") == "2",
            "resolved environment is not two-device")
    require(exports.get("LUMINA_CMF_FINE_ENVELOPE_REFINEMENTS") ==
            str(expected_refinements),
            "resolved environment refinement count differs")

    binary = regular_file(input_dir / "lumina_cuda", "staged binary")
    sealed_binary = regular_file(input_dir / "binary.sha256", "binary seal").read_text().split()[0]
    require(sha256(binary) == sealed_binary, "staged binary SHA mismatch")
    deck = verify_hash_manifest(input_dir / "deck.sha256", input_dir / "model", "deck")
    topion = verify_hash_manifest(input_dir / "topion.sha256", input_dir / "global_atomic", "topion")
    log_repairs = audit_log_repairs((run_root / "stdout.log", run_root / "stderr.log"))

    return {
        "status": "PASS",
        "run_root": str(run_root.resolve()),
        "binary_sha256": sealed_binary,
        "deck_files_verified": len(deck),
        "topion_files_verified": len(topion),
        "targeted_gate_report_sha256": sha256(run_root / "a210_targeted_gate_report.json"),
        "targeted_snapshot_report_sha256": sha256(
            run_root / "a210_targeted_snapshot_report.json"
        ),
        "r1_reference_comparison_sha256": sha256(
            run_root / "r1_k24_reference_comparison.json"
        ),
        "stdout_sha256": sha256(run_root / "stdout.log"),
        "stderr_sha256": sha256(run_root / "stderr.log"),
        "tripwire": {
            "status": "COMPLETED_WITHOUT_CONFLICT",
            "gpu_count": len(preflights),
            "gpu_family": "A100",
            "supervisor_log_sha256": sha256(manual / "supervisor.log"),
        },
        "physics": {
            "line_operator": "CMFGEN_NONOVERLAP_HOMOLOGY_SIGMA0",
            "proof_envelope_refinements": expected_refinements,
            "sobolev_jbar_cells": 109_014_300,
            "r2_signed_material": r2,
            "r1_reference_identity": r1_identity,
            "r1_proof_fields_verified": proof_changes_verified,
            "coevolution_generations": [1, 2],
        },
        "repair": {
            "physical_values_modified": False,
            "floor": 0,
            "cap": 0,
            "clamp": 0,
            "jitter": 0,
            "repair": 0,
            "structured_observations": targeted_repairs + reference_repairs,
            "log_observations": log_repairs,
            "zero_environment_variables": list(ZERO_ENV),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--refinement-comparison", type=Path, required=True)
    parser.add_argument("--proof-witness-baseline", type=Path, required=True)
    parser.add_argument("--expected-refinements", type=int, default=24)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.expected_refinements <= 64:
        parser.error("--expected-refinements must be in 1..64")
    return args


def main() -> int:
    args = parse_args()
    try:
        refinement = audit_refinement(args.refinement_comparison)
        proof_witnesses = audit_proof_witnesses(
            args.run_root, args.proof_witness_baseline
        )
        run = audit_run(args.run_root, args.expected_refinements)
        payload = {
            "schema": "LUMINA_A210_NONOVERLAP_GATE_COMPLETION_V1",
            "status": "PASS",
            "requirements": {
                "k18_cancellation_census": refinement,
                "k30_local_proof_witnesses": proof_witnesses,
                "a100x2_non_census_gate": run,
            },
            "physical_values_modified": False,
            "floor": 0,
            "cap": 0,
            "clamp": 0,
            "jitter": 0,
            "repair": 0,
        }
    except (AuditError, OSError, UnicodeError) as exc:
        payload = {
            "schema": "LUMINA_A210_NONOVERLAP_GATE_COMPLETION_V1",
            "status": "FAIL",
            "error": str(exc),
        }
        atomic_write(args.report, payload)
        print(f"FAIL A210_NONOVERLAP_GATE_COMPLETION reason={exc}", file=sys.stderr)
        return 4
    atomic_write(args.report, payload)
    print(
        "PASS A210_NONOVERLAP_GATE_COMPLETION "
        "k12_unresolved=19 k18_unresolved=0 devices=2 line_operator=SOBOLEV "
        "floor=0 cap=0 clamp=0 jitter=0 repair=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
