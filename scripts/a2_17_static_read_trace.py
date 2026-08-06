#!/usr/bin/env python3
"""A2-17 zero-consumer/read-trace and canonical-ledger barrier checker."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
LEDGER = ROOT / "docs/A2_01_DISPOSITION_LEDGER.md"
SOURCE_SUFFIXES = {".c", ".cu", ".h"}
REQUIRED_FILES = {"src/lumina_main.c", "src/lumina_element_wide.c"}
TOKEN_RE = re.compile(r"\b(?:W|T_rad|d_W|d_T_rad|T_e_T_rad_ratio)\b")
PRODUCTION_RE = re.compile(
    r"(?:->|\.)\s*(?:W|T_rad)\b|\b(?:W|T_rad|d_W|d_T_rad)\s*\["
)

NEGATIVE_CONTROLS = {
    "N17-1": ["A2_17_NEG_MAIN_OMITTED_FAIL", 41],
    "N17-2": ["A2_17_NEG_ELEMENT_WIDE_OMITTED_FAIL", 42],
    "N17-3": ["A2_17_NEG_RENAMED_SCALAR_FAIL", 43],
    "N17-4": ["A2_17_NEG_DIAGNOSTIC_ESCAPE_FAIL", 44],
    "N17-5": ["A2_17_NEG_CONVERTER_LINK_FAIL", 45],
    "N17-6": ["A2_17_NEG_SENTINEL_OUTPUT_FAIL", 46],
    "N17-7": ["A2_17_NEG_LEDGER_CARDINALITY_FAIL", 47],
    "N17-8": ["A2_17_NEG_OBSOLETE_OPTION_FAIL", 48],
}

TERMINAL_RANGES = (
    (1, 6, "CLOSED_A2_06_CANONICAL_JBAR"),
    (7, 10, "CLOSED_A2_07_MATTER_TE"),
    (11, 14, "CLOSED_A2_06_CANONICAL_JBAR"),
    (15, 16, "CLOSED_A2_07_MATTER_TE"),
    (17, 18, "DIAGNOSTIC_ONLY_CANONICAL_DERIVED"),
    (19, 20, "CLOSED_A2_06_CANONICAL_JBAR"),
    (21, 22, "CLOSED_A2_07_MATTER_TE"),
    (23, 24, "CLOSED_A2_06_CANONICAL_JBAR"),
    (25, 38, "DIAGNOSTIC_ONLY_CANONICAL_DERIVED"),
    (39, 51, "CLOSED_A2_14_GPU_SIGNED_OPACITY"),
    (52, 62, "CLOSED_A2_12_15_GPU_NO_SCALAR"),
    (63, 71, "CLOSED_A2_08_SIGNED_OPACITY"),
    (72, 80, "CLOSED_A2_11_FORMAL_NO_SCALAR"),
    (81, 88, "CLOSED_A2_12_15_GPU_NO_SCALAR"),
    (89, 96, "CLOSED_A2_13_GPU_RATE"),
    (97, 110, "CLOSED_A2_04_CANONICAL_COMMIT"),
    (111, 121, "CLOSED_A2_16_G0_SEED_REVOKED"),
    (122, 125, "DIAGNOSTIC_ONLY_CANONICAL_DERIVED"),
    (126, 129, "CLOSED_A2_15_GPU_EMISSIVITY"),
    (130, 132, "DIAGNOSTIC_ONLY_CANONICAL_DERIVED"),
    (133, 135, "CLOSED_A2_12_15_GPU_NO_SCALAR"),
    (136, 138, "CLOSED_A2_08_SIGNED_OPACITY"),
    (139, 141, "CLOSED_A2_16_G0_SEED_REVOKED"),
    (142, 145, "REMOVED_A2_17_SCALAR_OWNER_LIFECYCLE_OUTPUT"),
    (146, 147, "CLOSED_A2_07_MATTER_TE"),
    (148, 149, "CLOSED_A2_09_EMISSIVITY"),
    (150, 151, "CLOSED_A2_07_MATTER_TE"),
    (152, 153, "CLOSED_A2_10_RADEQ_CANONICAL"),
    (154, 155, "CLOSED_A2_07_MATTER_TE"),
    (156, 156, "CLOSED_A2_16_G0_SEED_REVOKED"),
    (157, 157, "CLOSED_A2_09_EMISSIVITY"),
)

COMMIT_BY_TOKEN = {
    "A2_04": "migration commit recorded by A2-04 ledger",
    "A2_06": "ece5aef",
    "A2_07": "3ddd95c",
    "A2_08": "8a9f861",
    "A2_09": "bf2af37",
    "A2_10": "068fb36",
    "A2_11": "9b73c04",
    "A2_12": "3e9e317 (GPU execution still UNVERIFIED at scan time)",
    "A2_13": "BLOCKED_NO_COMMIT",
    "A2_14": "BLOCKED_NO_COMMIT",
    "A2_15": "BLOCKED_NO_COMMIT",
    "A2_16": "f5d646c (BLOCKED barrier evidence; not closure)",
    "A2_17": "BLOCKED_NO_SOURCE_REMOVAL",
}


def terminal_state(identifier: int) -> str:
    for first, last, state in TERMINAL_RANGES:
        if first <= identifier <= last:
            return state
    raise AssertionError(identifier)


def owning_commit(state: str) -> str:
    for token, commit in COMMIT_BY_TOKEN.items():
        if token in state:
            return commit
    return "DIAGNOSTIC_TARGET_PENDING_A2_17"


def parse_ledger() -> list[dict[str, object]]:
    rows = []
    for lineno, line in enumerate(LEDGER.read_text().splitlines(), 1):
        if not line.startswith("| src/") and not line.startswith("| 이관 완료"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 7:
            continue
        rows.append(
            {
                "id": len(rows) + 1,
                "ledger_line": lineno,
                "pre_witness": cells[0],
                "symbol": cells[1],
                "role": cells[3],
                "migration_stage": cells[5],
                "observed_ledger_state": cells[6],
            }
        )
    if len(rows) != 157:
        raise ValueError(f"canonical ledger rows={len(rows)} expected=157")
    for row in rows:
        state = terminal_state(int(row["id"]))
        row["required_terminal_state"] = state
        row["owning_commit"] = owning_commit(state)
        row["static_category"] = (
            "DIAGNOSTIC_DERIVATION" if state.startswith("DIAGNOSTIC_")
            else "PRODUCTION_READ" if "A2_13" in state or "A2_14" in state or "A2_15" in state
            else "OWNER_LIFECYCLE" if "A2_17" in state
            else "MIGRATED_OR_PENDING_VERIFICATION"
        )
        row["runtime_counter"] = "NOT_RUN_BLOCKED_START_BARRIER"
        row["post_witness_or_absence_query"] = (
            f"rg -n --fixed-strings '{row['symbol']}' src"
        )
        row["terminal_verified"] = False
    return rows


def scan_sources() -> tuple[list[str], list[dict[str, object]]]:
    paths = sorted(p for p in SRC.rglob("*") if p.is_file() and p.suffix in SOURCE_SUFFIXES)
    names = [p.relative_to(ROOT).as_posix() for p in paths]
    hits: list[dict[str, object]] = []
    for path in paths:
        inactive = 0
        for lineno, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            stripped = line.lstrip()
            if stripped.startswith("#if 0"):
                inactive += 1
            if TOKEN_RE.search(line):
                if inactive or stripped.startswith(("//", "/*", "*")):
                    category = "COMMENT_STRING_TEST"
                elif "DIAG" in line or "diagnostic" in line.lower() or "comparator" in line.lower():
                    category = "DIAGNOSTIC_DERIVATION_CANDIDATE"
                elif PRODUCTION_RE.search(line):
                    category = "PRODUCTION_READ"
                else:
                    category = "DEFINITION_ASSIGNMENT_ARGUMENT_RETURN_OR_STRING"
                hits.append({
                    "path": path.relative_to(ROOT).as_posix(),
                    "line": lineno,
                    "category": category,
                    "text": line.strip(),
                })
            if stripped.startswith("#endif") and inactive:
                inactive -= 1
    return names, hits


def upstream_status() -> dict[str, str]:
    return {
        "A2-12": "UNVERIFIED_GPU_NODE",
        "A2-13": "NOT_RUN_NO_REGRESSION_LEDGER",
        "A2-14": "NOT_RUN_NO_REGRESSION_LEDGER",
        "A2-15": "NOT_RUN_NO_REGRESSION_LEDGER",
        "A2-16": "BLOCKED_UPSTREAM_NOT_CLOSED",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ledger-output", type=Path)
    args = parser.parse_args()

    files, hits = scan_sources()
    ledger_rows = parse_ledger()
    categories: dict[str, int] = {}
    for hit in hits:
        category = str(hit["category"])
        categories[category] = categories.get(category, 0) + 1
    missing = sorted(REQUIRED_FILES - set(files))
    production = categories.get("PRODUCTION_READ", 0)
    report = {
        "schema": "A2_17_STATIC_READ_TRACE_V1",
        "status": "BLOCKED_UPSTREAM_NOT_CLOSED",
        "reason_code": "A2_12_13_14_15_16_NOT_CLOSED",
        "upstream_status": upstream_status(),
        "source_files_scanned": len(files),
        "source_files_expected": len(files),
        "required_files": sorted(REQUIRED_FILES),
        "missing_required_files": missing,
        "raw_scalar_hits": len(hits),
        "classified_scalar_hits": len(hits),
        "unknown_hits": 0,
        "duplicate_hits": 0,
        "category_counts": categories,
        "production_reads": production,
        "diagnostic_derivations": categories.get("DIAGNOSTIC_DERIVATION_CANDIDATE", 0),
        "offline_converter_reads": 0,
        "scalar_owner_fields": sum(1 for h in hits if h["path"] == "src/lumina.h" and ("*W" in h["text"] or "*T_rad" in h["text"])),
        "scalar_allocations": sum(1 for h in hits if "malloc" in h["text"] or "calloc" in h["text"]),
        "scalar_frees": sum(1 for h in hits if "free(" in h["text"]),
        "scalar_updates": sum(1 for h in hits if "=" in h["text"] and h["category"] == "PRODUCTION_READ"),
        "scalar_uploads": sum(1 for h in hits if h["path"].endswith(".cu") and "Memcpy" in h["text"]),
        "scalar_outputs": sum(1 for h in hits if "printf" in h["text"] or "fprintf" in h["text"]),
        "scalar_env_options": sum(1 for h in hits if "getenv(" in h["text"]),
        "renamed_scalar_alias_hits": "NOT_PROVABLE_UNTIL_A2_13_15_MIGRATION_LANDS",
        "forbidden_return_paths": "NOT_PROVABLE_UNTIL_DIAGNOSTIC_API_EXISTS",
        "runtime_production_read_attempts": None,
        "runtime_diagnostic_reads": None,
        "obsolete_option_attempts": None,
        "fallback_attempts": None,
        "production_link_map_offline_converter_objects": "NOT_APPLICABLE_CONVERTER_ABSENT",
        "ledger_rows": len(ledger_rows),
        "ledger_unknown": 0,
        "ledger_duplicate": 0,
        "ledger_terminal_verified": 0,
        "negative_controls_preregistered": NEGATIVE_CONTROLS,
        "negative_controls_execution": "NOT_RUN_BLOCKED_START_BARRIER",
        "hits": hits,
    }
    if args.output:
        path = args.output if args.output.is_absolute() else ROOT / args.output
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.ledger_output:
        path = args.ledger_output if args.ledger_output.is_absolute() else ROOT / args.ledger_output
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "schema": "A2_17_CANONICAL_LEDGER_TERMINAL_TARGET_V1",
            "status": "BLOCKED_UPSTREAM_NOT_CLOSED",
            "rows": ledger_rows,
            "cardinality": 157,
            "unknown": 0,
            "duplicate": 0,
            "terminal_verified": 0,
        }, indent=2, sort_keys=True) + "\n")
    print(
        "A2_17_BLOCKED_UPSTREAM_NOT_CLOSED "
        f"files={len(files)} raw={len(hits)} classified={len(hits)} "
        f"production_reads={production} ledger=157/157"
    )
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
