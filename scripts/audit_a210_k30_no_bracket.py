#!/usr/bin/env python3
"""Seal the K30 proof resolution and the subsequent physical no-bracket branch.

This is deliberately not a gate-success judge.  It proves that the two K24
endpoint proof failures resolved at K30, while keeping the later four-shell
thermal no-bracket and the optional GEOMETRIC_MID proof failure distinct.
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


class AuditError(RuntimeError):
    pass


KV = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")
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
EXPECTED_EXACT = (
    (45, 9.6662782724980344e-09),
    (52, 8.1222406993212508e-09),
)
EXPECTED_DOMAIN = (
    "3278062cf80281ffdcc4eb74ffc37e743cbdc51a128da5a319bfba7d3a6416c4"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def regular(path: Path, label: str) -> Path:
    require(path.is_file() and not path.is_symlink(),
            f"missing or unsafe {label}: {path}")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(regular(path, label).read_text(
            encoding="utf-8", errors="strict"))
    except (json.JSONDecodeError, UnicodeError, OSError) as exc:
        raise AuditError(f"invalid {label}: {path}: {exc}") from exc
    require(isinstance(value, dict), f"{label} is not an object")
    return value


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def fields(line: str) -> dict[str, str]:
    return dict(KV.findall(line))


def records(lines: list[str], prefix: str) -> list[dict[str, str]]:
    return [fields(line) for line in lines if line.startswith(prefix)]


def one(lines: list[str], prefix: str) -> dict[str, str]:
    found = records(lines, prefix)
    require(len(found) == 1,
            f"expected one {prefix!r} record, found {len(found)}")
    return found[0]


def integer(item: dict[str, str], name: str, label: str) -> int:
    try:
        return int(item[name])
    except (KeyError, ValueError) as exc:
        raise AuditError(f"invalid {name} in {label}") from exc


def finite(item: dict[str, str], name: str, label: str) -> float:
    try:
        value = float(item[name])
    except (KeyError, ValueError, OverflowError) as exc:
        raise AuditError(f"invalid {name} in {label}") from exc
    require(math.isfinite(value), f"nonfinite {name} in {label}")
    return value


def exact_fields(item: dict[str, str], expected: dict[str, str], label: str) -> None:
    for key, value in expected.items():
        require(item.get(key) == value,
                f"{label} {key}={item.get(key)!r}, expected {value!r}")


def zero_repair_object(value: Any, label: str = "$") -> int:
    count = 0
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"floor", "cap", "clamp", "jitter", "repair"}:
                count += 1
                require(not isinstance(child, bool) and float(child) == 0.0,
                        f"nonzero repair field {label}.{key}={child!r}")
            count += zero_repair_object(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            count += zero_repair_object(child, f"{label}[{index}]")
    return count


def parse_exports(path: Path) -> dict[str, str]:
    pattern = re.compile(r'^declare -x ([A-Za-z_][A-Za-z0-9_]*)="(.*)"$')
    result: dict[str, str] = {}
    for line in regular(path, "resolved environment").read_text(
            encoding="utf-8", errors="strict").splitlines():
        match = pattern.fullmatch(line)
        require(match is not None, f"malformed resolved export: {line!r}")
        name, value = match.groups()
        require(name not in result, f"duplicate resolved export: {name}")
        result[name] = value
    return result


def audit_comparison(path: Path, stderr: Path, occurrence: int) -> dict[str, Any]:
    report = load_json(path, f"R{occurrence + 1} proof comparison")
    require(report.get("schema") ==
            "LUMINA_A210_TARGETED_REFERENCE_COMPARISON_V1" and
            report.get("status") == "PASS",
            f"R{occurrence + 1} proof comparison did not pass")
    require(report.get("comparison_mode") == "PROOF_REFINEMENT_ONLY" and
            report.get("reason") ==
            "PHYSICAL_AND_SOLVER_FIELDS_BIT_EXACT_PROOF_BOUNDS_CONTRACTED" and
            report.get("reference_refinements") == 24 and
            report.get("candidate_refinements") == 30 and
            report.get("reference_occurrence") == occurrence and
            report.get("candidate_occurrence") == occurrence and
            report.get("differences") == [] and
            report.get("physical_values_modified") is False,
            f"R{occurrence + 1} proof-only contract differs")
    candidate = Path(str(report.get("candidate_stderr", "")))
    reference = Path(str(report.get("reference_stderr", "")))
    require(candidate.resolve() == stderr.resolve() and
            report.get("candidate_sha256") == sha256(stderr),
            f"R{occurrence + 1} candidate log differs")
    require(regular(reference, "K24 reference log") and
            report.get("reference_sha256") == sha256(reference),
            f"R{occurrence + 1} reference log differs")
    changes = report.get("proof_field_changes")
    require(isinstance(changes, list) and len(changes) == 5,
            f"R{occurrence + 1} proof change set differs")
    ratios = [float(change["upper_bound_ratio"]) for change in changes
              if change.get("field") != "refinements"]
    require(len(ratios) == 3 and all(math.isfinite(value) and 0.0 <= value < 1.0
                                    for value in ratios),
            f"R{occurrence + 1} proof bounds did not contract")
    repairs = zero_repair_object(report, f"$.r{occurrence + 1}")
    require(repairs >= 5, f"R{occurrence + 1} repair audit fields absent")
    return {
        "status": "PASS",
        "report": str(path.resolve()),
        "report_sha256": sha256(path),
        "occurrence": occurrence,
        "upper_bound_ratios": ratios,
        "physical_and_solver_fields_bit_exact": True,
    }


def audit_proof_report(path: Path, stderr: Path) -> dict[str, Any]:
    report = load_json(path, "K30 proof witness report")
    require(report.get("schema") ==
            "lumina-a2-10-k30-proof-witness-resolution-v1" and
            report.get("status") == "PASS" and
            report.get("candidate_log_sha256") == sha256(stderr) and
            report.get("physical_values_modified") is False,
            "K30 endpoint proof report differs")
    witnesses = report.get("witnesses")
    require(isinstance(witnesses, list) and len(witnesses) == 2,
            "K30 endpoint proof witness count differs")
    observed = {(row.get("phase"), row.get("line"), row.get("shell"))
                for row in witnesses}
    require(observed == {("LOWER", 1_154_618, 5), ("UPPER", 894_169, 27)},
            "K30 endpoint proof witness set differs")
    for row in witnesses:
        require(row.get("physical_values_bit_exact_to_k24") is True and
                0.0 <= float(row["bound_to_required_ratio"]) < 1.0 and
                float(row["signed_rate_to_uncertainty_ratio"]) > 1.0,
                "K30 endpoint proof witness did not resolve")
    require(zero_repair_object(report, "$.proof") >= 5,
            "K30 endpoint proof report lacks repair fields")
    return {
        "status": "PASS",
        "report": str(path.resolve()),
        "report_sha256": sha256(path),
        "witnesses": witnesses,
    }


def audit_line_identity(path: Path, stderr: Path) -> dict[str, Any]:
    report = load_json(path, "K30 line identity report")
    require(report.get("schema") == "lumina-a210-line-identity-summary-v1" and
            report.get("verdict") == "COMPLETE" and
            Path(str(report.get("stderr", ""))).resolve() == stderr.resolve() and
            report.get("stderr_sha256") == sha256(stderr) and
            report.get("records") == 150 and
            report.get("phase_batch_counts") ==
            {"LOWER": 1, "UPPER": 1, "INTERIOR": 1} and
            report.get("physical_mutation") == 0 and report.get("repair") == 0,
            "K30 line identity report differs")
    counterfactual = report.get("endpoint_counterfactual")
    require(isinstance(counterfactual, dict) and
            counterfactual.get("bracket_counts") ==
            {"current": 46, "exact_constant": 46, "einstein": 46} and
            counterfactual.get("recovered_shells") ==
            {"exact_constant": [], "einstein": []} and
            counterfactual.get("lost_shells") ==
            {"exact_constant": [], "einstein": []},
            "endpoint coefficient counterfactual differs")
    scans = report.get("interior_scans")
    require(isinstance(scans, list) and len(scans) == 2,
            "interior diagnostic scan set differs")
    public, geometric = scans
    require(public.get("phase") == "PUBLIC_SEED" and
            public.get("valid") == 1 and
            public.get("endpoint_no_bracket") == 4 and
            public.get("interior_bracket") == 0 and
            public.get("still_same_sign") == 4,
            "PUBLIC_SEED no-bracket evidence differs")
    require(geometric == {
        "kind": "invalid", "phase": "GEOMETRIC_MID",
        "status": "RADEQ_SIGN_MISMATCH", "reason": None, "valid": 0,
        "action": "DIAGNOSTIC_ONLY",
    }, "GEOMETRIC_MID proof-failure evidence differs")
    return {
        "status": "PASS",
        "report": str(path.resolve()),
        "report_sha256": sha256(path),
        "endpoint_brackets": 46,
        "endpoint_no_bracket_shells": [0, 1, 2, 3],
        "public_seed": {
            "temperature_K": public["shells"][0]["T_mid"],
            "still_same_sign": 4,
        },
        "geometric_mid": "DIAGNOSTIC_PROOF_BLOCKED_NOT_A_PHYSICAL_VERDICT",
    }


def audit_run(run_root: Path) -> dict[str, Any]:
    require(run_root.is_dir() and not run_root.is_symlink(),
            f"missing or unsafe run root: {run_root}")
    stderr = regular(run_root / "stderr.log", "model stderr")
    stdout = regular(run_root / "stdout.log", "model stdout")
    lines = stderr.read_text(encoding="utf-8", errors="strict").splitlines()
    text = "\n".join(lines)
    require(regular(run_root / "model.rc", "model rc").read_text().strip() == "1",
            "model rc is not the expected fail-closed value 1")
    manual = run_root / "manual_control"
    require(regular(manual / "child.rc", "tripwire child rc").read_text().strip()
            == "70", "wrapper rc is not 70")
    require((manual / "FAILED").is_file() and not (manual / "YIELDED").exists()
            and not (manual / "COMPLETED").exists(),
            "tripwire terminal markers differ")
    supervisor = regular(manual / "supervisor.log", "tripwire log").read_text(
        encoding="utf-8", errors="strict")
    require(supervisor.count("status=START") == 1 and
            supervisor.count("status=CHILD_STARTED") == 1 and
            supervisor.count("status=FAILED child_rc=70") == 1 and
            "action=YIELD" not in supervisor and "COLLISION" not in supervisor,
            "tripwire did not fail naturally without a resource conflict")
    preflights = [line for line in supervisor.splitlines()
                  if "gpu_preflight=" in line]
    require(len(preflights) == 2 and all("A100" in line for line in preflights),
            "tripwire did not preflight two A100 devices")

    input_dir = run_root / "input"
    binary = regular(input_dir / "lumina_cuda", "staged binary")
    sealed_binary = regular(input_dir / "binary.sha256", "binary seal").read_text(
        encoding="ascii", errors="strict").split()[0]
    require(sha256(binary) == sealed_binary,
            "staged binary differs from its seal")
    require(regular(input_dir / "envelope_refinements.txt", "refinement seal")
            .read_text().strip() == "30", "refinement seal is not K30")
    require(regular(input_dir / "precore_tau_refresh.txt", "pre-core seal")
            .read_text().strip() == "0", "rejected pre-core refresh was enabled")
    require(regular(input_dir / "diagnostic_mode.txt", "diagnostic seal")
            .read_text().strip() == "A210_TARGETED_GATE",
            "diagnostic mode differs")
    exports = parse_exports(input_dir / "resolved_lumina.exports")
    for name in ZERO_ENV:
        require(exports.get(name) == "0", f"numerical repair env differs: {name}")
    require(exports.get("LUMINA_CMF_FINE_MGPU_DEVICES") == "2" and
            exports.get("LUMINA_CMF_FINE_ENVELOPE_REFINEMENTS") == "30" and
            exports.get("LUMINA_A210_CANCELLATION_CENSUS") in (None, "0"),
            "A100x2 K30 non-census execution contract differs")
    require("[A2-10][CANCELLATION-CENSUS]" not in text,
            "census contaminated the K30 non-census branch")

    exact = records(lines, "[cmf_fine][EXACT-MULTIGPU-EPOCH]")
    require(len(exact) == 2, "expected two exact publications")
    for index, (item, expected) in enumerate(zip(exact, EXPECTED_EXACT), 1):
        exact_fields(item, {
            "status": "OK", "devices": "2/2", "refinements": "30",
            "component_envelope": "1", "failure_phase": "0",
            "failure_iteration": "-1", "floor": "0", "clamp": "0",
            "jitter": "0", "domain_hash": EXPECTED_DOMAIN,
        }, f"R{index} exact")
        require(integer(item, "iterations", f"R{index} exact") == expected[0] and
                finite(item, "residual", f"R{index} exact") == expected[1] and
                finite(item, "residual", f"R{index} exact") <=
                finite(item, "tolerance", f"R{index} exact"),
                f"R{index} exact solve differs")

    material = records(lines, "[cmf_fine][SIGNED-MATERIAL-CENSUS]")
    require(len(material) == 2, "expected two signed material records")
    exact_fields(material[1], {
        "line_shells": "22866166", "exact_zero_tau": "86148134",
        "raw_negative": "4246581", "mild_negative": "4246577",
        "srce_chk": "4", "raw_preserved": "1", "floor": "0",
        "clamp": "0", "jitter": "0",
    }, "R2 signed material")
    sobolev = one(lines, "[cmf_fine][SOBOLEV-LINE-OPERATOR]")
    exact_fields(sobolev, {
        "status": "PASS", "mode": "CMFGEN_NONOVERLAP_HOMOLOGY_SIGMA0",
        "jbar_cells": "109014300", "raw_negative": "4246581",
        "mild_negative": "4246577", "srce_chk_expected": "4",
        "srce_chk_applied": "4", "all_jbar_finite": "1",
        "raw_preserved": "1", "floor": "0", "cap": "0",
        "clamp": "0", "jitter": "0", "repair": "0",
    }, "R2 Sobolev operator")
    identities = records(lines, "[R6][LINE-IDENTITY]")
    coverage = records(lines, "[R6][LINE-COVERAGE]")
    require(len(identities) == 2 and len(coverage) == 2,
            "R6 publication count differs")
    require([integer(row, "generation", "R6 identity") for row in identities]
            == [1, 2] and
            [integer(row, "generation", "R6 coverage") for row in coverage]
            == [1, 2], "coevolution generation order differs")
    for row in coverage:
        exact_fields(row, {"valid_cells": "109014300", "partial_lines": "0",
                           "unsampled_lines": "0"}, "R6 coverage")
    seed = one(lines,
        "[A2-INIT][SEED-MATERIAL] event=INIT_SEED_MATERIAL_PREDICTOR")
    exact_fields(seed, {
        "r1_generation": "1", "te_generation": "1->1",
        "population_generation": "1->2", "te_manifest_preserved": "1",
        "te_publication_preserved": "1", "floor": "0", "cap": "0",
        "clamp": "0", "jitter": "0", "repair": "0",
    }, "seed material commit")

    no_bracket_headers = [row for row in records(
        lines, "[A2-10][VECTOR-NOBRACKET]") if "count" in row]
    require(len(no_bracket_headers) == 1, "vector no-bracket header differs")
    no_bracket = no_bracket_headers[0]
    exact_fields(no_bracket, {
        "count": "4", "first_shell": "0", "same_positive": "0",
        "same_negative": "4", "endpoint_zero": "0", "T_lo": "3500",
        "T_hi": "140000",
    }, "vector no-bracket")
    require(finite(no_bracket, "res_lo", "vector no-bracket") < 0.0 and
            finite(no_bracket, "res_hi", "vector no-bracket") < 0.0,
            "first no-bracket shell is not cooling-dominated at both endpoints")
    line_term = one(lines,
        "[A2-10][VECTOR-NOBRACKET] first_shell=0 C_line_emit_lo=")
    require(finite(line_term, "C_line_emit_lo", "line term") > 0.0 and
            finite(line_term, "C_line_emit_hi", "line term") > 0.0,
            "line cooling is not finite positive")
    public_rows = records(lines,
        "[A2-10][VECTOR-INTERIOR-SCAN] phase=PUBLIC_SEED shell=")
    require(len(public_rows) == 4 and
            [integer(row, "shell", "PUBLIC_SEED") for row in public_rows]
            == [0, 1, 2, 3], "PUBLIC_SEED shell set differs")
    for row in public_rows:
        require(finite(row, "res_mid", "PUBLIC_SEED") < 0.0 and
                finite(row, "line_emit_mid", "PUBLIC_SEED") > 0.0 and
                row.get("lo_mid_bracket") == "0" and
                row.get("mid_hi_bracket") == "0" and
                row.get("action") == "DIAGNOSTIC_ONLY",
                "PUBLIC_SEED no-bracket record differs")
    public_summary = one(lines,
        "[A2-10][VECTOR-INTERIOR-SCAN] phase=PUBLIC_SEED valid=")
    exact_fields(public_summary, {
        "valid": "1", "endpoint_no_bracket": "4", "interior_bracket": "0",
        "still_same_sign": "4", "action": "DIAGNOSTIC_ONLY",
        "solver_result": "RADEQ_NO_BRACKET",
    }, "PUBLIC_SEED summary")
    geometric = one(lines,
        "[A2-10][VECTOR-INTERIOR-SCAN] phase=GEOMETRIC_MID status=")
    exact_fields(geometric, {
        "status": "RADEQ_SIGN_MISMATCH", "valid": "0",
        "action": "DIAGNOSTIC_ONLY",
    }, "GEOMETRIC_MID summary")
    blocked_cell = one(lines,
        "[A2-10][LINE-NET-CELL-BLOCKED] status=UNRESOLVED_CANCELLATION ")
    exact_fields(blocked_cell, {
        "phase": "INTERIOR", "line": "894169", "shell": "11",
        "status": "UNRESOLVED_CANCELLATION", "clamp": "0",
        "floor": "0", "jitter": "0",
    }, "GEOMETRIC_MID proof witness")
    require(finite(blocked_cell, "uncertainty", "GEOMETRIC_MID proof witness") >
            abs(finite(blocked_cell, "signed_rate", "GEOMETRIC_MID proof witness")),
            "GEOMETRIC_MID witness was not proof-unresolved")
    blocked = one(lines,
        "[A2-10][BLOCKED] event=R7_MATERIAL_UPDATE_BLOCKED")
    exact_fields(blocked, {
        "reason": "RADEQ_NO_BRACKET", "te_generation_before": "1",
        "te_generation_after": "1", "te_manifest_preserved": "1",
        "generation_preserved": "1", "material_update": "BLOCKED",
        "action": "TERMINATE", "no_bracket_delta": "1",
    }, "R7 fail-closed publication")
    fatal = one(lines, "[R7][FATAL]")
    exact_fields(fatal, {"lane": "DET", "iter": "0", "rc": "4"},
                 "R7 fatal")

    post = regular(manual / "post_gate_monitor.log", "post-gate monitor")
    completion = regular(manual / "completion_monitor.log", "completion monitor")
    require("status=NO_GATE_PASS" in post.read_text() and
            "status=NO_REFERENCE_PASS" in completion.read_text(),
            "automatic monitors did not reject the failed gate")
    return {
        "status": "PASS",
        "run_root": str(run_root.resolve()),
        "binary_sha256": sealed_binary,
        "stdout_sha256": sha256(stdout),
        "stderr_sha256": sha256(stderr),
        "model_rc": 1,
        "wrapper_rc": 70,
        "tripwire": {
            "status": "NATURAL_CHILD_FAILURE_WITHOUT_CONFLICT",
            "gpu_count": 2,
            "gpu_family": "A100",
            "supervisor_log_sha256": sha256(manual / "supervisor.log"),
        },
        "physics": {
            "line_operator": "CMFGEN_NONOVERLAP_HOMOLOGY_SIGMA0",
            "refinements": 30,
            "coevolution_generations": [1, 2],
            "r2_signed_material": {
                "signed_cells": 22_866_166, "exact_zero_tau": 86_148_134,
                "raw_negative": 4_246_581, "mild_negative": 4_246_577,
                "srce_chk": 4,
            },
            "endpoint_no_bracket_shells": [0, 1, 2, 3],
            "first_shell_endpoint_residuals": {
                "lower": finite(no_bracket, "res_lo", "vector no-bracket"),
                "upper": finite(no_bracket, "res_hi", "vector no-bracket"),
            },
            "public_seed_residuals": [
                finite(row, "res_mid", "PUBLIC_SEED") for row in public_rows
            ],
            "geometric_mid": {
                "physical_verdict": "NOT_AVAILABLE",
                "reason": "LOCAL_CANCELLATION_PROOF_UNRESOLVED",
                "line": 894_169, "shell": 11,
            },
            "te_publication": "PRESERVED_UNCHANGED",
        },
        "repair": {
            "physical_values_modified": False,
            "floor": 0, "cap": 0, "clamp": 0, "jitter": 0, "repair": 0,
            "zero_environment_variables": list(ZERO_ENV),
        },
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    run = audit_run(args.run_root)
    stderr = args.run_root / "stderr.log"
    r1 = audit_comparison(args.r1_comparison, stderr, 0)
    r2 = audit_comparison(args.r2_comparison, stderr, 1)
    proof = audit_proof_report(args.proof_witness_report, stderr)
    identity = audit_line_identity(args.line_identity_report, stderr)
    return {
        "schema": "lumina-a210-k30-no-bracket-branch-audit-v1",
        "status": "PASS",
        "interpretation": (
            "K30_ENDPOINT_PROOF_RESOLVED; PHYSICAL_ENDPOINT_AND_PUBLIC_SEED_"
            "NO_BRACKET_PRESERVED; GEOMETRIC_MID_PROOF_BLOCKED_SEPARATELY"
        ),
        "requirements": {
            "run": run,
            "r1_reference_identity": r1,
            "r2_reference_identity": r2,
            "k30_endpoint_proof": proof,
            "line_identity_and_no_bracket": identity,
        },
        "gate_pass": False,
        "next_action": "ROOT_CAUSE_MATCHED_STATE_LINE_RATE_WITHOUT_PHYSICAL_REPAIR",
        "physical_values_modified": False,
        "floor": 0, "cap": 0, "clamp": 0, "jitter": 0, "repair": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--r1-comparison", type=Path, required=True)
    parser.add_argument("--r2-comparison", type=Path, required=True)
    parser.add_argument("--proof-witness-report", type=Path, required=True)
    parser.add_argument("--line-identity-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = audit(args)
    except (AuditError, OSError, UnicodeError, ValueError, KeyError) as exc:
        report = {
            "schema": "lumina-a210-k30-no-bracket-branch-audit-v1",
            "status": "FAIL",
            "error": str(exc),
        }
        atomic_write(args.report, report)
        print(f"FAIL A210_K30_NO_BRACKET_BRANCH reason={exc}")
        return 4
    atomic_write(args.report, report)
    print(
        "PASS A210_K30_NO_BRACKET_BRANCH endpoint_proof=RESOLVED "
        "physical_no_bracket=4 geometric_mid=PROOF_BLOCKED "
        "floor=0 cap=0 clamp=0 jitter=0 repair=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
