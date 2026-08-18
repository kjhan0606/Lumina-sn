#!/usr/bin/env python3
"""Static gate for the A2-09 population-native line-emission transaction.

The gate enforces the Fable-reviewed contract: line emission uses the direct
n_u*A_ul*h*nu*beta/(4*pi*dnu) form, never line_source_S; the mutable raw tau
slab is generation-bracketed at both ends; writer/reader share one NLTE
authority predicate and one LTE population routine; and an unqualified in-band
line cell aborts the private candidate before exact-zero promotion or commit.

The negative controls mutate source text in memory; the working tree is never
changed.
"""

from pathlib import Path
import re
import sys


SOURCE = Path(__file__).resolve().parent.parent / "src" / "lumina_plasma.c"
FORMULA_SOURCE = (
    Path(__file__).resolve().parent.parent / "src" / "emissivity_publication.c"
)


def function_body(text: str, name: str) -> str:
    match = re.search(r"\b" + re.escape(name) + r"\s*\([^;]*?\)\s*\{", text, re.S)
    if not match:
        raise RuntimeError(f"function not found: {name}")
    start = match.end() - 1
    depth = 0
    for pos in range(start, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[start : pos + 1]
    raise RuntimeError(f"unterminated function: {name}")


def inspect_source(text: str, formula_text: str) -> list[str]:
    failures: list[str] = []
    body = function_body(text, "a209_publish_cpu_emissivity")

    required = (
        "blocked_line_cells",
        "[A2-09][BLOCKED]",
        "first_tau_status",
        "first_population_status",
        "a209_upper_population_for_tau",
        "a209_sobolev_line_eta",
        "a209_line_generation_snapshot",
        "a209_line_generation_bracket(&line_generation_begin,NULL)",
        "&line_generation_begin,&line_generation_end",
        "EMISS_TAU_MUTATED_DURING_CONSUME",
        "nlte_tau_line_authority",
        "if(blocked_line_cells){",
        "a209_publication_free(&c);return invalid_eta_cells?5:3;",
    )
    for token in required:
        if token not in body:
            failures.append(f"missing fail-closed token {token!r}")

    if "tv!=A208_VALID&&tv!=A208_EXACT_ZERO" not in body:
        failures.append("tau qualification predicate is incomplete")

    if re.search(r"opacity\s*->\s*line_source_(?:S|validity)", body):
        failures.append("A2-09 production reads forbidden quotient source")

    begin_check = body.find("a209_line_generation_bracket(&line_generation_begin,NULL)")
    consume = body.find("for(int l=0;l<opacity->n_lines;l++)")
    end_check = body.find("&line_generation_begin,&line_generation_end")
    if min(begin_check, consume, end_check) < 0 or not begin_check < consume < end_check:
        failures.append("raw tau consumption is not bracketed at both ends")

    upper = function_body(text, "a209_upper_population_for_tau")
    writer = function_body(text, "nlte_update_tau_sobolev")
    bulk = function_body(text, "compute_tau_sobolev")
    if "nlte_tau_line_uses_nlte" not in upper:
        failures.append("A2-09 does not use shared NLTE tau authority")
    for token in ("nlte_tau_line_authority", "nlte_tau_line_shell_authorized"):
        if token not in writer:
            failures.append(f"NLTE tau writer missing shared authority token {token!r}")
    if "g_ew_tau_authority" in upper or "g_ew_tau_authority" in writer:
        failures.append("writer/reader rederive element-wide tau authority")
    if "population_line_level_number_density" not in upper or \
       "population_line_level_number_density" not in bulk:
        failures.append("bulk tau and A2-09 do not share LTE line population routine")

    formula = function_body(formula_text, "a209_sobolev_line_eta")
    for token in ("-expm1(-tau)/tau", "n_upper*A_ul*h*nu*beta",
                  "(tau==0.0)?1.0"):
        if token not in formula:
            failures.append(f"missing direct-formula token {token!r}")

    abort_at = body.find("if(blocked_line_cells){")
    promote_at = body.find("c.component_status[i]=c.eta_bb[i]==0?")
    commit_at = body.find("a209_publication_commit(&opacity->cpu_emissivity,&c)")
    if min(abort_at, promote_at, commit_at) < 0:
        failures.append("cannot locate abort/promotion/commit ordering")
    elif not abort_at < promote_at < commit_at:
        failures.append("candidate can be promoted or committed before blocked-source abort")

    abort_block = body[abort_at:promote_at] if abort_at >= 0 and promote_at >= 0 else ""
    if "a209_publication_commit" in abort_block:
        failures.append("blocked-source branch commits the private candidate")
    if "EMISS_EXACT_ZERO" in abort_block:
        failures.append("blocked-source branch converts missing state to exact zero")

    return failures


def run_negative_controls(source: str, formula_source: str) -> list[str]:
    mutations = (
        (source.replace("if(blocked_line_cells){",
                        "if(0&&blocked_line_cells){", 1), formula_source),
        (source.replace(
             "a209_publication_free(&c);return invalid_eta_cells?5:3;",
             "a209_publication_free(&c);", 1), formula_source),
        (source.replace("a209_sobolev_line_eta(", "a209_removed_line_eta(", 1),
         formula_source),
        (source.replace(
             "a209_line_generation_bracket(&line_generation_begin,NULL)",
             "a209_line_generation_bracket(&line_generation_end,NULL)", 1),
         formula_source),
        (source, formula_source.replace("(tau==0.0)?1.0",
                                        "(tau==0.0)?0.0", 1)),
        (source.replace("&line_generation_begin,&line_generation_end",
                        "&line_generation_end,&line_generation_end", 1),
         formula_source),
        (source.replace("nlte_tau_line_uses_nlte(authority,shell,n_shells)",
                        "0", 1), formula_source),
        (source.replace("population_line_level_number_density(",
                        "population_lte_level_fraction("), formula_source),
    )
    missed: list[str] = []
    for index, (mutated, mutated_formula) in enumerate(mutations, start=1):
        if mutated == source and mutated_formula == formula_source:
            missed.append(f"injection-{index}-not-applied")
        elif not inspect_source(mutated, mutated_formula):
            missed.append(f"injection-{index}-not-detected")
    return missed


def main() -> int:
    source = SOURCE.read_text()
    formula_source = FORMULA_SOURCE.read_text()
    failures = inspect_source(source, formula_source)
    if failures:
        for failure in failures:
            print(f"[SH-RADEQ-0][STATIC][FAIL] {failure}", file=sys.stderr)
        return 1
    print("[SH-RADEQ-0][STATIC][PASS] population-native direct line eta is "
          "generation-bound and aborts the private candidate on invalid input")

    missed = run_negative_controls(source, formula_source)
    if missed:
        print("[SH-RADEQ-0][NEGATIVE-CONTROL][FAIL] " + ",".join(missed),
              file=sys.stderr)
        return 1
    print("[SH-RADEQ-0][NEGATIVE-CONTROL][PASS] injections=8 detected=8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
