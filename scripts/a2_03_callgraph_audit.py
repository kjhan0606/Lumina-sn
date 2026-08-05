#!/usr/bin/env python3
"""Compiler-callgraph audit for the A2-03 producer-only shadow boundary."""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    "src/lumina_main.c",
    "src/lumina_transport.c",
    "src/a2_02c_segment_capture.c",
    "src/radiation_field.c",
    "src/lumina_plasma.c",
    "src/lumina_element_wide.c",
    "src/lumina_atomic.c",
    "src/lumina_cmfgen.c",
]

EXPECTED_FIELDS = [
    "shell_boundaries",
    "frequency_bin_edges",
    "J_nu",
    "units",
    "frame",
    "epoch",
    "generation",
    "provenance",
    "validity",
    "estimator_count_or_variance",
]

EXPECTED_EXTERNAL_CALLERS = {
    "radiation_field_shadow_gate_enabled": {"radiation_field_shadow_init"},
    "radiation_field_shadow_init": {"main"},
    "radiation_field_shadow_free": {"main", "radiation_field_shadow_init"},
    "radiation_field_shadow_begin_mc": {"main"},
    "radiation_field_accumulator_create": {"main"},
    "radiation_field_accumulator_free": {
        "main",
        "radiation_field_accumulator_create",
    },
    "radiation_field_accumulator_add": {"update_base_estimators"},
    "radiation_field_accumulator_reduce": {"main"},
    "radiation_field_shadow_commit_mc": {"main"},
    "radiation_field_shadow_validate_owner": {"radiation_field_shadow_commit_mc"},
    "radiation_field_shadow_dump_if_requested": {"radiation_field_shadow_commit_mc"},
}


def fail(message: str) -> None:
    raise SystemExit(f"A2_03_CALLGRAPH_AUDIT FAIL: {message}")


def canonical_fields() -> list[str]:
    text = (ROOT / "src/radiation_field.h").read_text(encoding="utf-8")
    matches = list(re.finditer(
        r"typedef\s+struct\s*\{(?P<body>[^{}]*)\}\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*;",
        text, flags=re.S))
    match = next((item for item in matches if item.group("name") == "RadiationField"), None)
    if match is None:
        fail("RadiationField definition not found")
    fields: list[str] = []
    for declaration in match.group("body").split(";"):
        declaration = re.sub(r"/\*.*?\*/", "", declaration, flags=re.S).strip()
        if not declaration:
            continue
        name = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*$", declaration)
        if not name:
            fail(f"unparsed RadiationField declaration: {declaration!r}")
        fields.append(name.group(1))
    return fields


def parse_cgraph(path: Path, result: dict[str, set[str]]) -> None:
    current: str | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        symbol = re.match(r"^([A-Za-z_][A-Za-z0-9_.$]*)/\d+ \(", line)
        if symbol:
            current = symbol.group(1).split(".", 1)[0]
            continue
        if current and line.startswith("  Called by:"):
            for caller in re.findall(r"([A-Za-z_][A-Za-z0-9_.$]*)/\d+", line):
                result.setdefault(current, set()).add(caller.split(".", 1)[0])


def compiler_callgraph() -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    with tempfile.TemporaryDirectory(prefix="a2_03_cgraph_") as temporary:
        outdir = Path(temporary)
        for index, source in enumerate(SOURCES):
            obj = outdir / f"unit_{index}.o"
            command = [
                "gcc",
                "-O0",
                "-std=c11",
                "-Isrc",
                "-fdump-ipa-cgraph",
                "-c",
                source,
                "-o",
                str(obj),
            ]
            completed = subprocess.run(
                command, cwd=ROOT, text=True, capture_output=True, check=False
            )
            if completed.returncode != 0:
                fail(f"compiler rejected {source}: {completed.stderr[-2000:]}")
        dumps = sorted(outdir.glob("*.cgraph"))
        if len(dumps) != len(SOURCES):
            fail(f"expected {len(SOURCES)} cgraph dumps, found {len(dumps)}")
        for dump in dumps:
            parse_cgraph(dump, result)
    return result


def main() -> int:
    fields = canonical_fields()
    if fields != EXPECTED_FIELDS:
        fail(f"canonical fields {fields!r} != {EXPECTED_FIELDS!r}")

    graph = compiler_callgraph()
    observed: dict[str, list[str]] = {}
    for callee, expected in EXPECTED_EXTERNAL_CALLERS.items():
        callers = graph.get(callee, set())
        if callers != expected:
            fail(f"{callee} callers={sorted(callers)} expected={sorted(expected)}")
        observed[callee] = sorted(callers)

    header = (ROOT / "src/radiation_field.h").read_text(encoding="utf-8")
    public_symbols = sorted(set(re.findall(r"\b(radiation_field_[a-z0-9_]+)\s*\(", header)))
    forbidden = [
        symbol for symbol in public_symbols
        if re.search(r"(?:get|read|query|lookup|sample|consume)", symbol)
    ]
    if forbidden:
        fail(f"consumer-like public API exported: {forbidden}")

    payload = {
        "schema": "lumina-a2-03-callgraph-audit-v1",
        "canonical_field_count": len(fields),
        "canonical_fields": fields,
        "compiler": "gcc -fdump-ipa-cgraph",
        "compiled_translation_units": SOURCES,
        "radiation_field_callers": observed,
        "physics_consumer_callers": [],
        "public_consumer_api": [],
        "verdict": "PASS",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
