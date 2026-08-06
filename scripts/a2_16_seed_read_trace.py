#!/usr/bin/env python3
"""A2-16 conservative scalar-seed read trace and start-barrier checker.

This checker deliberately reports BLOCKED (rc 3) while any production scalar
read remains outside the generation-zero seed helper.  It never treats a
missing/invalid input as zero and never upgrades an upstream state to PASS.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
REQUIRED_FILES = {"src/lumina_main.c", "src/lumina_element_wide.c"}
SOURCE_SUFFIXES = {".c", ".cu", ".h"}
TOKEN_RE = re.compile(
    r"\b(?:W|T_rad|d_W|d_T_rad|T_e_T_rad_ratio)\b"
    r"|LUMINA_TRAD_COLOR_FIX|LUMINA_TE_TRAD_RATIO"
)
FIELD_READ_RE = re.compile(
    r"(?:->|\.)\s*(?:W|T_rad)\b|\b(?:W|T_rad|d_W|d_T_rad)\s*\["
)

NEGATIVE_CONTROLS = {
    "N16-1": {"marker": "A2_16_NEG_POSTCOMMIT_SEED_MUTATION_FAIL", "child_rc": 41},
    "N16-2": {"marker": "A2_16_NEG_POSTG0_TRAD_READ_FAIL", "child_rc": 42},
    "N16-3": {"marker": "A2_16_NEG_POSTG0_W_READ_FAIL", "child_rc": 43},
    "N16-4": {"marker": "A2_16_NEG_SEED_HOLD_FAIL", "child_rc": 44},
    "N16-5": {"marker": "A2_16_NEG_RUNTIME_LEGACY_LOAD_FAIL", "child_rc": 45},
    "N16-6": {"marker": "A2_16_NEG_OBSOLETE_TRADFIX_FAIL", "child_rc": 46},
    "N16-7": {"marker": "A2_16_NEG_TE_SEED_FALLBACK_FAIL", "child_rc": 47},
    "N16-8": {"marker": "A2_16_NEG_CAPABILITY_LIFETIME_FAIL", "child_rc": 48},
}


def source_inventory() -> list[Path]:
    return sorted(
        p for p in SRC.rglob("*") if p.is_file() and p.suffix in SOURCE_SUFFIXES
    )


def classify_line(path: Path, line: str, inactive: bool) -> str:
    stripped = line.strip()
    if inactive or stripped.startswith(("//", "/*", "*")):
        return "COMMENT_STRING_TEST"
    if "DIAG" in line or "diagnostic" in line.lower() or "comparator" in line.lower():
        return "DIAGNOSTIC_SHADOW_CANDIDATE"
    if path.name == "lumina_atomic.c" and "opacity->t_electrons[i]" in line:
        return "UNISOLATED_G0_TE_SEED_TARGET"
    if FIELD_READ_RE.search(line) or "getenv(" in line or "read_csv_column" in line:
        return "PRODUCTION_READ"
    return "DEFINITION_ASSIGNMENT_ARGUMENT_OR_STRING"


def scan() -> tuple[list[str], list[dict[str, object]]]:
    files = source_inventory()
    names = [p.relative_to(ROOT).as_posix() for p in files]
    hits: list[dict[str, object]] = []
    for path in files:
        depth = 0
        for lineno, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            directive = line.lstrip()
            if directive.startswith("#if 0"):
                depth += 1
            if TOKEN_RE.search(line):
                hits.append(
                    {
                        "path": path.relative_to(ROOT).as_posix(),
                        "line": lineno,
                        "category": classify_line(path, line, depth > 0),
                        "text": line.strip(),
                    }
                )
            if directive.startswith("#endif") and depth:
                depth -= 1
    return names, hits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    files, hits = scan()
    categories: dict[str, int] = {}
    for hit in hits:
        key = str(hit["category"])
        categories[key] = categories.get(key, 0) + 1
    missing_required = sorted(REQUIRED_FILES - set(files))
    production = categories.get("PRODUCTION_READ", 0)
    g0_target = categories.get("UNISOLATED_G0_TE_SEED_TARGET", 0)
    blockers = []
    if missing_required:
        blockers.append("BLOCKED_SOURCE_INVENTORY_INCOMPLETE")
    if production:
        blockers.append("BLOCKED_PRODUCTION_SCALAR_READS_REMAIN")
    if g0_target != 0:
        blockers.append("BLOCKED_G0_HELPER_NOT_ISOLATED")
    blockers.append("BLOCKED_UPSTREAM_A2_13_15_NOT_CLOSED")

    report = {
        "schema": "A2_16_SEED_READ_TRACE_V1",
        "status": "BLOCKED_UPSTREAM_NOT_CLOSED",
        "reason_codes": blockers,
        "source_files_scanned": len(files),
        "source_files_expected": len(files),
        "required_files": sorted(REQUIRED_FILES),
        "missing_required_files": missing_required,
        "raw_scalar_hits": len(hits),
        "category_counts": categories,
        "runtime_counters": {
            "seed_files_opened": None,
            "seed_cells_loaded": None,
            "seed_invalid_cells": None,
            "legacy_converter_rows": None,
            "legacy_converter_invalid_rows": None,
            "g0_seed_reads": None,
            "g0_te_seed_writes": None,
            "first_commit_successes": None,
            "seed_capability_revokes": None,
            "seed_buffers_freed": None,
            "post_g0_seed_read_attempts": None,
            "post_g0_scalar_read_attempts": None,
            "seed_reopen_attempts": None,
            "tradfix_option_attempts": None,
            "ratio_option_attempts": None,
            "hold_attempts": None,
            "extrapolation_attempts": None,
            "neighbor_copy_attempts": None,
            "seed_fallback_attempts": None,
            "partial_seed_publish_attempts": None,
        },
        "negative_controls_preregistered": NEGATIVE_CONTROLS,
        "negative_controls_execution": "NOT_RUN_BLOCKED_START_BARRIER",
        "hits": hits,
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        target = args.output if args.output.is_absolute() else ROOT / args.output
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    print(
        "A2_16_BLOCKED_UPSTREAM_NOT_CLOSED "
        f"files={len(files)} raw_hits={len(hits)} production_reads={production} "
        f"g0_unisolated={g0_target}"
    )
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
