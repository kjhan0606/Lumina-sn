#!/usr/bin/env python3
"""A2-07 frozen census and production population call-graph guard."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLASMA = ROOT / "src/lumina_plasma.c"

LEDGER_IDS = [
    "A2-05:old9160:T_rad", "A2-05:old9162:W",
    "A2-05:old11943:W", "A2-05:old11943:T_rad",
    "A2-05:old13672:W", "A2-05:old13672:T_rad",
    "A2-06:old4879:T_rad", "A2-06:old4880:W",
    "A2-06:old12093:W", "A2-06:old12100:W",
    "A2-06:old13739:W", "A2-06:old13743:W",
    "BASE:old2081:T_rad", "BASE:old2082:W",
    "BASE:old7402:T_rad", "BASE:old7403:W",
    "BASE:old17832:T_rad", "BASE:old17833:W",
]

GROUPS = [
    "partition canonical/fallback", "duplicate partition_functions_Te",
    "Sobolev LTE synthesis", "macro/k-packet fallback",
    "recombination destination", "BF opacity/rate population", "bf_rate_pop",
    "all-level Gamma weight", "RADEQ untracked population",
    "coupled untracked population", "isolated-row anchor",
    "force/singular fallback", "within-superlevel distribution",
    "Boltzmann dump", "ion rate selection duplication",
    "level matrix BF/BB assembly", "transactional publish/counters",
]

PRODUCTION_FUNCTIONS = {
    "compute_partition_functions": {
        "forbid": ("plasma->T_rad", "plasma->W", "T_e_T_rad_ratio"),
        "require": ("population_partition_build",),
    },
    "bf_rate_pop": {
        "forbid": ("T_rad", "plasma->W", "partition_functions_Te"),
        "require": ("population_lte_level_fraction",),
    },
    "compute_tau_sobolev": {
        "forbid": ("plasma->T_rad", "plasma->W", "T_e_T_rad_ratio"),
        "require": ("population_lte_level_fraction",),
    },
    "compute_bf_opacity": {
        "forbid": ("plasma->T_rad", "plasma->W", "beta_rad"),
        "require": ("population_lte_level_fraction",
                    "population_committed_generation"),
    },
    "parity_gamma_phot_checked": {
        "forbid": ("plasma->T_rad", "plasma->W", "T_e_T_rad_ratio"),
        "require": ("population_lte_level_fraction",
                    "nlte_bf_gamma_canonical"),
    },
    "compute_ion_populations_shell": {
        "forbid": ("T_e_T_rad_ratio", "parity_rate_se_ratio(",
                   "n_ion = 1e-300", "product = 1e30"),
        "require": ("parity_rate_se_ratio_checked", "log-sum-exp",
                    "POP_BF_STALE"),
    },
    "compute_electron_density": {
        "forbid": ("plasma->T_rad", "plasma->W", "T_e_T_rad_ratio",
                   "parity_rate_se_ratio"),
        "require": ("compute_ion_populations_shell", "shell_converged"),
    },
    "nlte_precompute_within_sl_frac_checked": {
        "forbid": ("plasma->T_rad", "plasma->W", "T_e_T_rad_ratio"),
        "require": ("population_partition_view_check", "within_sl_stamp"),
    },
    "nlte_solve_ion_shell": {
        "forbid": ("T_e_T_rad_ratio",),
        "require": ("gauss_solve", "b[i] < 0.0", "return -1"),
    },
    "nlte_solve_all": {
        "forbid": ("T_e_T_rad_ratio",),
        "require": ("population_transaction_begin", "plasma->n_electron",
                    "atom->partition_functions", "ce_converged",
                    "POP_FORBIDDEN_FALLBACK"),
    },
}

REQUIRED_TOKENS = (
    "population_partition_build", "population_partition_view_check",
    "population_lte_level_fraction", "population_transaction_begin",
    "population_transaction_commit", "POP_INVALID_TE",
    "POP_STALE_DERIVED_TEMPERATURE", "POP_FORBIDDEN_FALLBACK",
    "[A2-07][POP-VIEW]",
)


def function_body(text: str, name: str) -> tuple[int, str]:
    match = re.search(rf"^[^\n]*\b{name}\s*\([^;]*?\)\s*\{{", text, re.M | re.S)
    if not match:
        raise ValueError(f"function not found: {name}")
    start = match.start()
    brace = text.find("{", match.start(), match.end())
    depth = 0
    for pos in range(brace, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text.count("\n", 0, start) + 1, text[start:pos + 1]
    raise ValueError(f"unterminated function: {name}")


def code_only(body: str) -> str:
    """Blank comments/strings while retaining newlines for line accounting."""
    token = re.compile(r"/\*.*?\*/|//[^\n]*|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'",
                       re.S)
    return token.sub(lambda m: "".join("\n" if c == "\n" else " "
                                       for c in m.group(0)), body)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()

    text = PLASMA.read_text(encoding="utf-8")
    header = (ROOT / "src/lumina.h").read_text(encoding="utf-8")
    atomic = (ROOT / "src/lumina_atomic.c").read_text(encoding="utf-8")
    contract = (ROOT / "src/population_contract.c").read_text(encoding="utf-8")
    all_text = text + header + atomic + contract
    violations: list[dict[str, object]] = []
    spans = {}
    missing_function_tokens: list[dict[str, object]] = []
    for function, rules in PRODUCTION_FUNCTIONS.items():
        line, body = function_body(text, function)
        spans[function] = {"line": line, "lines": body.count("\n") + 1}
        executable = code_only(body)
        for pattern in rules["forbid"]:
            for hit in re.finditer(re.escape(pattern), executable):
                violations.append({
                    "function": function,
                    "pattern": pattern,
                    "line": line + body.count("\n", 0, hit.start()),
                })
        for token in rules["require"]:
            if token not in body:
                missing_function_tokens.append({
                    "function": function, "token": token, "line": line,
                })

    # The enormous transition/matrix routines retain explicitly marked legacy
    # observers, but the physical assignment must be overwritten by checked views
    # before the first matrix consumer.
    transition_line, transition_body = function_body(
        text, "compute_transition_probabilities")
    spans["compute_transition_probabilities"] = {
        "line": transition_line, "lines": transition_body.count("\n") + 1}
    transition_code = code_only(transition_body)
    for pattern in ("beta_rad", "level_metastable[rl->",
                    "plasma->T_e ?", "T_e_T_rad_ratio"):
        if pattern in transition_code:
            violations.append({"function": "compute_transition_probabilities",
                               "pattern": pattern, "line": transition_line})
    for token in ("population_lte_level_fraction", "n_j_from_solve"):
        if token not in transition_body:
            missing_function_tokens.append({
                "function": "compute_transition_probabilities", "token": token})

    matrix_line, matrix_body = function_body(text, "nlte_assemble_rate_matrix")
    spans["nlte_assemble_rate_matrix"] = {
        "line": matrix_line, "lines": matrix_body.count("\n") + 1}
    shadow_begin = matrix_body.find("A2_06_DIAGNOSTIC_SHADOW_BEGIN")
    shadow_end = matrix_body.find("A2_06_DIAGNOSTIC_SHADOW_END")
    canonical = matrix_body.find("nlte_bb_jbar_canonical", shadow_end)
    first_matrix_write = matrix_body.find("ACM(", canonical)
    if not (0 <= shadow_begin < shadow_end < canonical < first_matrix_write):
        violations.append({"function": "nlte_assemble_rate_matrix",
                           "pattern": "shadow/canonical ordering",
                           "line": matrix_line})
    for token in ("nlte_bf_gamma_canonical", "nlte_bb_jbar_canonical"):
        if token not in matrix_body:
            missing_function_tokens.append({
                "function": "nlte_assemble_rate_matrix", "token": token,
                "line": matrix_line})

    missing = [token for token in REQUIRED_TOKENS if token not in all_text]
    duplicate_store = "partition_functions_Te" in all_text
    bf_signature_ok = bool(re.search(
        r"bf_rate_pop\([^)]*int n_shells,\s*double T_e\)", text, re.S))
    ledger_unique = len(LEDGER_IDS) == len(set(LEDGER_IDS)) == 18
    groups_unique = len(GROUPS) == len(set(GROUPS)) == 17
    status = "PASS" if not violations and not missing and not missing_function_tokens and not duplicate_store \
        and bf_signature_ok and ledger_unique and groups_unique else "FAIL_STATIC_CENSUS"
    report = {
        "schema": "A2_07_STATIC_CENSUS_V1",
        "status": status,
        "reason_code": "OK" if status == "PASS" else "CONTRACT_MISMATCH",
        "child_rc": 0 if status == "PASS" else 5,
        "wrapper_rc": 0 if status == "PASS" else 5,
        "ledger_ids": LEDGER_IDS,
        "outside_groups": GROUPS,
        "function_spans": spans,
        "violations": violations,
        "missing_required_tokens": missing,
        "missing_function_tokens": missing_function_tokens,
        "duplicate_partition_store": duplicate_store,
        "bf_rate_pop_te_only_signature": bf_signature_ok,
        "allowlist": [
            {
                "file": "src/lumina_plasma.c",
                "function": "compute_radiative_equilibrium_te",
                "root": "A2-10 temperature solve",
                "symbols": ["plasma->T_rad", "plasma->W"],
                "reason": "temperature/energy equation, not a population fallback",
                "followup": "A2-10",
            },
            {
                "file": "src/lumina_cuda.cu",
                "function": "nlte_solve_all_gpu",
                "root": "nlte_solve_all_gpu",
                "symbols": ["plasma->T_rad", "plasma->W",
                            "T_e_T_rad_ratio"],
                "reason": "GPU ownership is outside A2-07",
                "followup": "A2-13",
            },
            {
                "file": "src/lumina_plasma.c",
                "function": "nlte_assemble_rate_matrix:A2_06_DIAGNOSTIC_SHADOW",
                "root": "nlte_solve_all -> nlte_assemble_rate_matrix",
                "symbols": ["opacity->jbar_line", "plasma->T_rad",
                            "plasma->W"],
                "reason": "observer-only shadow overwritten by checked line view before ACM",
                "followup": "none",
            },
            {
                "file": "src/lumina_plasma.c",
                "function": "compute_ion_populations_shell:legacy_phi_shadow",
                "root": "compute_plasma_state -> compute_ion_populations_shell",
                "symbols": ["plasma->T_rad", "plasma->W"],
                "reason": "legacy Saha diagnostic shadow; ratio supplier is checked helper",
                "followup": "none",
            },
        ],
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if status == "PASS" else 5


if __name__ == "__main__":
    raise SystemExit(main())
