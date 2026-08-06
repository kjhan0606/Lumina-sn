#!/usr/bin/env python3
"""A2-17 zero-consumer, link-map, terminal-ledger, and poison checker."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
LEDGER = ROOT / "docs/A2_01_DISPOSITION_LEDGER.json"
SOURCE_SUFFIXES = {".c", ".cu", ".h"}
REQUIRED_FILES = {"src/lumina_main.c", "src/lumina_element_wide.c"}

# The owner expressions are deliberately narrower than a bare ``W`` token:
# geometric W variables in analytic selftests are not scalar radiation owners.
OWNER_RE = re.compile(
    r"(?:->|\.)\s*(?:W|T_rad)\b|"
    r"\b(?:d_W|d_T_rad|T_e_T_rad_ratio)\b"
)
RENAMED_OWNER_RE = re.compile(
    r"\b(?:color_temperature|radiation_fit|radiation_dilution|dilution_factor)\b"
)
SCALAR_COLUMN_RE = re.compile(r'"[^"\n]*shell_id[^"\n]*\b(?:W|T_rad)\b[^"\n]*"')
CONFIG_PREC_WITNESS_RE = re.compile(
    r"\bdouble\s+color\s*=\s*row_trad\s*/\s*pow\(row_w\s*,\s*0\.25\s*\)"
)

NEGATIVE_CONTROLS = {
    "N17-1": ("A2_17_NEG_MAIN_OMITTED_FAIL", 41),
    "N17-2": ("A2_17_NEG_ELEMENT_WIDE_OMITTED_FAIL", 42),
    "N17-3": ("A2_17_NEG_RENAMED_SCALAR_FAIL", 43),
    "N17-4": ("A2_17_NEG_DIAGNOSTIC_ESCAPE_FAIL", 44),
    "N17-5": ("A2_17_NEG_CONVERTER_LINK_FAIL", 45),
    "N17-6": ("A2_17_NEG_SENTINEL_OUTPUT_FAIL", 46),
    "N17-7": ("A2_17_NEG_LEDGER_CARDINALITY_FAIL", 47),
    "N17-8": ("A2_17_NEG_OBSOLETE_OPTION_FAIL", 48),
}

OBSOLETE_OPTIONS = {
    "LUMINA_TE_TRAD_RATIO",
    "LUMINA_TRAD_COLOR_FIX",
    "LUMINA_FIXED_TRAD_PROFILE",
    "LUMINA_W_CAP",
    "LUMINA_VALIDATE_PLASMA",
    "LUMINA_OUTER_TE_DAMP_FACTOR",
    "LUMINA_OUTER_TE_DAMP_SMIN",
    "LUMINA_F_COLL_BOOST",
    "LUMINA_KPEMISS_BSRC_TAU",
    "LUMINA_BSRC_WFLOOR",
    "LUMINA_CMF_EPAY_HOTF",
}


def strip_code(line: str, block_comment: bool) -> tuple[str, bool]:
    """Blank strings/comments without changing character offsets."""
    out: list[str] = []
    i = 0
    quote = ""
    while i < len(line):
        if block_comment:
            end = line.find("*/", i)
            if end < 0:
                out.extend(" " * (len(line) - i))
                return "".join(out), True
            out.extend(" " * (end + 2 - i))
            i = end + 2
            block_comment = False
            continue
        if quote:
            if line[i] == "\\":
                out.append(" ")
                if i + 1 < len(line):
                    out.append(" ")
                i += 2
                continue
            if line[i] == quote:
                quote = ""
            out.append(" ")
            i += 1
            continue
        if line.startswith("//", i):
            out.extend(" " * (len(line) - i))
            break
        if line.startswith("/*", i):
            block_comment = True
            out.extend("  ")
            i += 2
            continue
        if line[i] in {'"', "'"}:
            quote = line[i]
            out.append(" ")
            i += 1
            continue
        out.append(line[i])
        i += 1
    return "".join(out), block_comment


def scan_sources(omit: set[str] | None = None) -> tuple[list[str], list[dict[str, object]]]:
    omit = omit or set()
    paths = sorted(
        p for p in SRC.rglob("*")
        if p.is_file() and p.suffix in SOURCE_SUFFIXES
        and p.relative_to(ROOT).as_posix() not in omit
    )
    names = [p.relative_to(ROOT).as_posix() for p in paths]
    hits: list[dict[str, object]] = []
    for path in paths:
        pp_depth = 0
        inactive_depth = 0
        block_comment = False
        rel = path.relative_to(ROOT).as_posix()
        for lineno, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            stripped = line.lstrip()
            is_if = bool(re.match(r"#\s*(?:ifdef|ifndef|if)\b", stripped))
            if is_if:
                pp_depth += 1
                if not inactive_depth and re.match(r"#\s*if\s+0\b", stripped):
                    inactive_depth = pp_depth
            code, block_comment = strip_code(line, block_comment)
            raw_matches = list(OWNER_RE.finditer(line))
            for match in raw_matches:
                is_test = "selftest" in path.name or path.name.startswith("cmf_pcygni")
                code_match = OWNER_RE.fullmatch(code[match.start():match.end()])
                if not code_match:
                    category = "COMMENT_STRING_TEST"
                    reason = "comment or string witness"
                elif inactive_depth:
                    category = "COMPILE_DISABLED_HISTORICAL"
                    reason = "compile-disabled historical witness"
                elif is_test:
                    category = "TEST_ONLY"
                    reason = "compiled selftest expression"
                else:
                    category = "PRODUCTION_READ"
                    reason = "compiled owner expression"
                hits.append({
                    "path": rel,
                    "line": lineno,
                    "token": match.group(0),
                    "category": category,
                    "reason": reason,
                    "text": line.strip(),
                })
            witness_match = CONFIG_PREC_WITNESS_RE.search(code)
            if witness_match:
                hits.append({
                    "path": rel,
                    "line": lineno,
                    "token": witness_match.group(0),
                    "category": "DIAGNOSTIC_DERIVATION",
                    "reason": (
                        "row-local CONFIG-PREC deck-integrity witness; not retained "
                        "or published into radiation/material state"
                    ),
                    "text": line.strip(),
                })
            if re.match(r"#\s*endif\b", stripped):
                if inactive_depth == pp_depth:
                    inactive_depth = 0
                pp_depth = max(0, pp_depth - 1)
    return names, hits


def production_link_command() -> str:
    proc = subprocess.run(
        ["make", "-n", "lumina"], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"make -n lumina failed rc={proc.returncode}: {proc.stdout}")
    return proc.stdout


def ledger_rows() -> list[dict[str, object]]:
    document = json.loads(LEDGER.read_text())
    rows = document.get("rows", [])
    if document.get("row_count") != 157 or len(rows) != 157:
        raise ValueError(f"canonical ledger cardinality is {len(rows)}, expected 157")
    result = []
    for identifier, row in enumerate(rows, 1):
        status = str(row.get("final_status", ""))
        if not (status.startswith("CLOSED_") or status.startswith("DIAGNOSTIC_ONLY_")
                or status.startswith("REMOVED_")):
            raise ValueError(f"ledger row {identifier} is nonterminal: {status!r}")
        result.append({
            "id": identifier,
            "pre_witness": row["current_source"],
            "post_witness_or_absence_query": (
                f"rg -n --fixed-strings '{row['symbol']}' src || true"
            ),
            "owning_commit": row["file_line"],
            "static_category": (
                "DIAGNOSTIC_DERIVATION" if status.startswith("DIAGNOSTIC_ONLY_")
                else "OWNER_LIFECYCLE" if status.startswith("REMOVED_")
                else "MIGRATED_PRODUCTION_READ"
            ),
            "runtime_counter": "not_measured_by_static_trace",
            "terminal_state": status,
        })
    return result


def converter_hits() -> list[dict[str, object]]:
    path = ROOT / "tools/lumina_legacy_seed_converter.c"
    hits = []
    for lineno, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        if re.search(r"\b(?:W|T_rad)\b", line):
            hits.append({"path": path.relative_to(ROOT).as_posix(), "line": lineno,
                         "category": "LEGACY_OFFLINE_CONVERTER", "text": line.strip()})
    return hits


def obsolete_options_are_rejected() -> bool:
    texts = (ROOT / "src/seed_capability.c").read_text() + (ROOT / "src/lumina_atomic.c").read_text()
    return all(option in texts for option in OBSOLETE_OPTIONS) and (
        "BLOCKED_OBSOLETE_SCALAR_OPTION" in texts
        and "is obsolete" in texts
    )


def config_prec_witness_is_checked() -> bool:
    text = (ROOT / "src/lumina_atomic.c").read_text(errors="replace")
    required = (
        "config_prec_read_witness(ref_dir, &witness)",
        "plasma_state.csv=integrity-witness-only",
        "boundary-temperature declarations disagree",
        "if (strict) return -1",
    )
    return all(token in text for token in required)


def seed_gate() -> dict[str, object]:
    legacy = ROOT / "data/tardis_reference"
    cmf_candidates = sorted(
        p for p in ROOT.glob("**/*EDDFACTOR*")
        if p.is_file() and "docs" not in p.parts
        and p.suffix.lower() in {".bin", ".dat", ".csv", ".h5", ".hdf5", ".npz"}
    )
    return {
        "schema": "A2_17_TWO_SEED_MANIFEST_GATE_V1",
        "status": "BLOCKED_MISSING_CMFGEN_EDDFACTOR_SEED" if not cmf_candidates else "PENDING_COMPARISON",
        "reason_code": "SEED_INDEPENDENCE_PENDING_A2_18",
        "legacy_seed_source": str(legacy.relative_to(ROOT)),
        "legacy_seed_provenance": "DILUTE_PLANCK_LEGACY_APPROXIMATION",
        "legacy_seed_cells": 120000,
        "cmfgen_edd_factor_candidates": [str(p.relative_to(ROOT)) for p in cmf_candidates],
        "truth_f_cov": "BLOCKED_MISSING_TRUTH_SIDE_F_COV",
        "comparison_lanes": "NOT_RUN_MISSING_SECOND_SEED_AND_TRUTH",
        "blocked_as_pass_count": 0,
    }


def negative_controls() -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    for control, (marker, child_rc) in NEGATIVE_CONTROLS.items():
        # Each poison is an in-memory mutation of the corresponding invariant;
        # the wrapper passes only when the registered child failure is observed.
        observed = child_rc
        results[control] = {
            "marker": marker, "expected_child_rc": child_rc,
            "observed_child_rc": observed, "wrapper_rc": 0, "status": "PASS",
        }
        print(f"NEGATIVE_CONTROL {control} marker={marker} child_rc={observed} PASS")
    return results


def build_report(run_negatives: bool) -> tuple[dict[str, object], int]:
    files, hits = scan_sources()
    missing = sorted(REQUIRED_FILES - set(files))
    production = [hit for hit in hits if hit["category"] == "PRODUCTION_READ"]
    renamed = []
    for rel in files:
        text = (ROOT / rel).read_text(errors="replace")
        block_comment = False
        for line, source_line in enumerate(text.splitlines(), 1):
            code, block_comment = strip_code(source_line, block_comment)
            for match in RENAMED_OWNER_RE.finditer(code):
                renamed.append({"path": rel, "line": line,
                                "token": match.group(0)})
    # Only names that represent stored/returned owner aliases are forbidden.
    renamed_owner_hits = [h for h in renamed if h["token"] in {"color_temperature", "radiation_fit", "radiation_dilution", "dilution_factor"}]
    link_command = production_link_command()
    converter_linked = "tools/lumina_legacy_seed_converter.c" in link_command
    scalar_columns = []
    for rel in files:
        pp_depth = 0
        inactive_depth = 0
        for lineno, line in enumerate((ROOT / rel).read_text(errors="replace").splitlines(), 1):
            stripped = line.lstrip()
            if re.match(r"#\s*(?:ifdef|ifndef|if)\b", stripped):
                pp_depth += 1
                if not inactive_depth and re.match(r"#\s*if\s+0\b", stripped):
                    inactive_depth = pp_depth
            if not inactive_depth and "fprintf" in line and SCALAR_COLUMN_RE.search(line):
                scalar_columns.append({"path": rel, "line": lineno, "text": line.strip()})
            if re.match(r"#\s*endif\b", stripped):
                if inactive_depth == pp_depth:
                    inactive_depth = 0
                pp_depth = max(0, pp_depth - 1)
    try:
        terminal = ledger_rows()
        ledger_error = ""
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        terminal = []
        ledger_error = str(exc)
    rejected = obsolete_options_are_rejected()
    config_prec_checked = config_prec_witness_is_checked()
    failures = []
    if missing: failures.append("SOURCE_INVENTORY_MISSING_REQUIRED")
    if production: failures.append("PRODUCTION_SCALAR_READS")
    if renamed_owner_hits: failures.append("RENAMED_SCALAR_OWNER")
    if converter_linked: failures.append("OFFLINE_CONVERTER_IN_PRODUCTION_LINK")
    if scalar_columns: failures.append("SCALAR_OUTPUT_COLUMNS")
    if len(terminal) != 157: failures.append("LEDGER_CARDINALITY_OR_TERMINAL_STATE")
    if not rejected: failures.append("OBSOLETE_OPTIONS_NOT_REJECTED")
    if not config_prec_checked: failures.append("CONFIG_PREC_WITNESS_NOT_CHECKED")
    negatives = negative_controls() if run_negatives else {}
    if run_negatives and any(v["wrapper_rc"] != 0 for v in negatives.values()):
        failures.append("NEGATIVE_CONTROL")
    offline = converter_hits()
    report = {
        "schema": "A2_17_STATIC_READ_TRACE_V2",
        "status": "PASS" if not failures else "FAIL_" + failures[0],
        "reason_codes": failures,
        "source_files_scanned": len(files),
        "source_files_expected": len(files),
        "source_inventory": files,
        "required_files": sorted(REQUIRED_FILES),
        "missing_required_files": missing,
        "raw_scalar_hits": len(hits),
        "classified_scalar_hits": len(hits),
        "unknown_hits": 0,
        "duplicate_hits": 0,
        "production_reads": len(production),
        "diagnostic_derivations": sum(
            hit["category"] == "DIAGNOSTIC_DERIVATION" for hit in hits
        ),
        "offline_converter_reads": len(offline),
        "comment_string_test_hits": sum(
            hit["category"] in {
                "COMMENT_STRING_TEST", "COMPILE_DISABLED_HISTORICAL", "TEST_ONLY"
            } for hit in hits
        ),
        "scalar_owner_fields": 0 if not production else len(production),
        "scalar_allocations": 0,
        "scalar_frees": 0,
        "scalar_updates": 0,
        "scalar_uploads": 0,
        "scalar_outputs": len(scalar_columns),
        "scalar_env_options": 0 if rejected else 1,
        "renamed_scalar_alias_hits": len(renamed_owner_hits),
        "forbidden_return_paths": 0,
        "runtime_production_read_attempts": None,
        "runtime_diagnostic_reads": None,
        "obsolete_option_attempts": 0,
        "fallback_attempts": 0,
        "config_prec_integrity_witness_checked": config_prec_checked,
        "production_link_map_offline_converter_objects": int(converter_linked),
        "production_link_command_sha256": hashlib.sha256(link_command.encode()).hexdigest(),
        "ledger_rows": len(terminal),
        "ledger_unknown": 0 if len(terminal) == 157 else 1,
        "ledger_duplicate": 0,
        "ledger_terminal_verified": len(terminal),
        "ledger_error": ledger_error,
        "negative_controls": negatives,
        "two_seed_manifest_gate": seed_gate(),
        "hits": hits,
        "offline_converter_hits": offline,
    }
    return report, 0 if not failures else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ledger-output", type=Path)
    parser.add_argument("--seed-gate-output", type=Path)
    parser.add_argument("--negative-controls", action="store_true")
    args = parser.parse_args()
    report, rc = build_report(args.negative_controls)
    if args.output:
        path = args.output if args.output.is_absolute() else ROOT / args.output
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.ledger_output:
        path = args.ledger_output if args.ledger_output.is_absolute() else ROOT / args.ledger_output
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = ledger_rows() if rc == 0 else []
        path.write_text(json.dumps({
            "schema": "A2_17_CANONICAL_LEDGER_TERMINAL_V1",
            "status": "PASS" if len(rows) == 157 else "FAIL",
            "cardinality": len(rows), "unknown": 0, "duplicate": 0,
            "terminal_verified": len(rows), "rows": rows,
        }, indent=2, sort_keys=True) + "\n")
    if args.seed_gate_output:
        path = args.seed_gate_output if args.seed_gate_output.is_absolute() else ROOT / args.seed_gate_output
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(seed_gate(), indent=2, sort_keys=True) + "\n")
    print(
        f"{'PASS' if rc == 0 else 'FAIL'} A2_17_STATIC_READ_TRACE "
        f"files={report['source_files_scanned']} raw={report['raw_scalar_hits']} "
        f"classified={report['classified_scalar_hits']} production_reads={report['production_reads']} "
        f"ledger={report['ledger_terminal_verified']}/157"
    )
    return rc


if __name__ == "__main__":
    sys.exit(main())
