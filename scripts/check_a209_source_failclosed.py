#!/usr/bin/env python3
"""Static gate for the A2-09 population-native line-emission transaction.

The gate enforces the Fable-reviewed contract: line emission uses the direct
n_u*A_ul*h*nu*beta/(4*pi*dnu) form, never line_source_S; the mutable raw tau
slab is generation-bracketed at both ends; writer/reader share one NLTE
authority predicate and one LTE population routine; and an unqualified in-band
line cell aborts the private candidate before exact-zero promotion or commit.

The forbidden quotient-source scan covers the text of exactly three functions:
a209_publish_cpu_emissivity_impl, a209_upper_population_for_tau, and
a209_sobolev_line_eta.  The a209_publish_cpu_emissivity and
nlte_update_tau_sobolev wrappers are separately pinned by exact body matches.

The negative controls mutate source text in memory; the working tree is never
changed.
"""

from pathlib import Path
import re
import sys

from gate_source_lib import body, body_raw, find_definition


SOURCE = Path(__file__).resolve().parent.parent / "src" / "lumina_plasma.c"
FORMULA_SOURCE = (
    Path(__file__).resolve().parent.parent / "src" / "emissivity_publication.c"
)

PUBLISH_WRAPPER = """{
 return a209_publish_cpu_emissivity_impl(
     opacity,bf,atom,plasma,nlte,epoch,a209_counters(),
     g_ew_tau_authority,g_ew_tau_authority_nshells);
}"""
TAU_WRAPPER = """{
    nlte_update_tau_sobolev_with_authority(
        nlte, atom, opacity, time_explosion, n_shells,
        g_ew_tau_authority, g_ew_tau_authority_nshells);
}"""
COUNTED_COMMIT = (
    "a209_publication_commit_counted(&opacity->cpu_emissivity,&c,ctr)"
)
COUNTED_NAME = "a209_publication_commit_counted("
EXPECTED_REASONS = (
    "missing fail-closed token 'if(blocked_line_cells){'",
    "missing fail-closed token "
    "'a209_publication_free(&c);return invalid_eta_cells?5:3;'",
    "missing fail-closed token 'a209_sobolev_line_eta'",
    "raw tau consumption is not bracketed at both ends",
    "missing direct-formula token '(tau==0.0)?1.0'",
    "raw tau consumption is not bracketed at both ends",
    "A2-09 does not use shared NLTE tau authority",
    "bulk tau and A2-09 do not share LTE line population routine",
    "A2-09 upper-population producer reads forbidden quotient source",
    "A2-09 line-eta formula reads forbidden quotient source",
)


def inspect_source(text: str, formula_text: str) -> list[str]:
    failures: list[str] = []
    publish = body(text, "a209_publish_cpu_emissivity_impl")
    upper = body(text, "a209_upper_population_for_tau")
    formula = body(formula_text, "a209_sobolev_line_eta")

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
        if token not in publish:
            failures.append(f"missing fail-closed token {token!r}")

    if "tv!=A208_VALID&&tv!=A208_EXACT_ZERO" not in publish:
        failures.append("tau qualification predicate is incomplete")

    if re.search(r"opacity\s*->\s*line_source_(?:S|validity)", publish):
        failures.append("A2-09 production reads forbidden quotient source")
    if re.search(r"opacity\s*->\s*line_source_(?:S|validity)", upper):
        failures.append(
            "A2-09 upper-population producer reads forbidden quotient source"
        )
    if re.search(r"opacity\s*->\s*line_source_(?:S|validity)", formula):
        failures.append(
            "A2-09 line-eta formula reads forbidden quotient source"
        )

    begin_check = publish.find(
        "a209_line_generation_bracket(&line_generation_begin,NULL)"
    )
    consume = publish.find("for(int l=0;l<opacity->n_lines;l++)")
    end_check = publish.find("&line_generation_begin,&line_generation_end")
    if min(begin_check, consume, end_check) < 0 or not begin_check < consume < end_check:
        failures.append("raw tau consumption is not bracketed at both ends")

    writer = body(text, "nlte_update_tau_sobolev_with_authority")
    bulk = body(text, "compute_tau_sobolev")
    if "nlte_tau_line_uses_nlte_by(" not in upper:
        failures.append("A2-09 does not use shared NLTE tau authority")
    for token in ("nlte_tau_line_authority", "nlte_tau_line_shell_authorized"):
        if token not in writer:
            failures.append(f"NLTE tau writer missing shared authority token {token!r}")
    if "g_ew_tau_authority" in upper or "g_ew_tau_authority" in writer:
        failures.append("writer/reader rederive element-wide tau authority")
    if "population_line_level_number_density" not in upper or \
       "population_line_level_number_density" not in bulk:
        failures.append("bulk tau and A2-09 do not share LTE line population routine")

    for token in ("-expm1(-tau)/tau", "n_upper*A_ul*h*nu*beta",
                  "(tau==0.0)?1.0"):
        if token not in formula:
            failures.append(f"missing direct-formula token {token!r}")

    abort_at = publish.find("if(blocked_line_cells){")
    promote_at = publish.find("c.component_status[i]=c.eta_bb[i]==0?")
    commit_at = publish.find(COUNTED_COMMIT)
    if min(abort_at, promote_at, commit_at) < 0:
        failures.append("cannot locate abort/promotion/commit ordering")
    elif not abort_at < promote_at < commit_at:
        failures.append("candidate can be promoted or committed before blocked-source abort")

    abort_block = (publish[abort_at:promote_at]
                   if abort_at >= 0 and promote_at >= 0 else "")
    if "a209_publication_commit" in abort_block:
        failures.append("blocked-source branch commits the private candidate")
    if "EMISS_EXACT_ZERO" in abort_block:
        failures.append("blocked-source branch converts missing state to exact zero")

    if body_raw(text, "a209_publish_cpu_emissivity") != PUBLISH_WRAPPER:
        failures.append("a209_publish_cpu_emissivity wrapper is not exact pure delegation")
    if body_raw(text, "nlte_update_tau_sobolev") != TAU_WRAPPER:
        failures.append("nlte_update_tau_sobolev wrapper is not exact pure delegation")
    if "if(!ctr)return 5;" not in publish:
        failures.append("A2-09 impl is missing exact counted-counter null guard")

    legacy_wrapper = body(formula_text, "a209_publication_commit")
    impl_call = COUNTED_COMMIT
    legacy_call = "a209_publication_commit_counted(pub,c,&g_ctr)"
    if (publish.count(impl_call) != 1
            or legacy_wrapper.count(legacy_call) != 1
            or text.count(COUNTED_NAME) != 1
            or formula_text.count(COUNTED_NAME) != 2):
        failures.append(
            "counted publication caller census is not plasma-impl=1 "
            "legacy-wrapper=1"
        )

    return failures


def replace_in_definition(text: str, name: str, old: str, new: str) -> str:
    start, end = find_definition(text, name)
    replaced = text[start:end].replace(old, new, 1)
    return text[:start] + replaced + text[end:]


def inject_at_unique_anchor(
        text: str, name: str, anchor: str, statement: str) -> str:
    start, end = find_definition(text, name)
    definition = text[start:end]
    if definition.count(anchor) != 1:
        return text
    injected = definition.replace(anchor, anchor + "\n " + statement, 1)
    return text[:start] + injected + text[end:]


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
        (replace_in_definition(
             source, "a209_upper_population_for_tau",
             "nlte_tau_line_uses_nlte_by(", "a209_alt_authority_stub("),
         formula_source),
        (source.replace("population_line_level_number_density(",
                        "population_lte_level_fraction("), formula_source),
        (inject_at_unique_anchor(
             source, "a209_upper_population_for_tau",
             "if(used_nlte)*used_nlte=0;",
             "(void)opacity->line_source_S[0];"),
         formula_source),
        (source, inject_at_unique_anchor(
             formula_source, "a209_sobolev_line_eta",
             "double beta=(tau==0.0)?1.0:-expm1(-tau)/tau;",
             "(void)opacity->line_source_S[0];")),
    )
    missed: list[str] = []
    for index, ((mutated, mutated_formula), expected) in enumerate(
            zip(mutations, EXPECTED_REASONS), start=1):
        if mutated == source and mutated_formula == formula_source:
            missed.append(f"injection-{index}-not-applied")
            continue
        failures = inspect_source(mutated, mutated_formula)
        if expected not in failures:
            missed.append(f"injection-{index}-wrong-reason")
            continue
        print("[SH-RADEQ-0][NEGATIVE-CONTROL][DETECTED] "
              f"injection={index} reason={expected}")
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
    print("[SH-RADEQ-0][NEGATIVE-CONTROL][PASS] injections=10 detected=10")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
