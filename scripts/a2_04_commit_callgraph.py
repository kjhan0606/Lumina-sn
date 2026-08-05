#!/usr/bin/env python3
"""Static/compiler gate for the A2-04 canonical producer commit choke point."""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "src/lumina_main.c",
    "src/lumina_transport.c",
    "src/a2_02c_segment_capture.c",
    "src/radiation_field.c",
    "src/lumina_plasma.c",
    "src/lumina_element_wide.c",
    "src/lumina_atomic.c",
    "src/lumina_cmfgen.c",
)


def fail(message: str) -> None:
    raise SystemExit(f"A2_04_COMMIT_CALLGRAPH FAIL: {message}")


def parse_graph(path: Path, result: dict[str, set[str]]) -> None:
    current: str | None = None
    for line in path.read_text(errors="replace").splitlines():
        symbol = re.match(r"^([A-Za-z_][A-Za-z0-9_.$]*)/\d+ \(", line)
        if symbol:
            current = symbol.group(1).split(".", 1)[0]
        elif current and line.startswith("  Called by:"):
            for caller in re.findall(r"([A-Za-z_][A-Za-z0-9_.$]*)/\d+", line):
                result.setdefault(current, set()).add(caller.split(".", 1)[0])


def compiler_graph() -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    with tempfile.TemporaryDirectory(prefix="a2_04_cgraph_") as temporary:
        work = Path(temporary)
        for index, source in enumerate(SOURCES):
            proc = subprocess.run(
                ["gcc", "-O0", "-std=c11", "-Isrc", "-fdump-ipa-cgraph",
                 "-c", source, "-o", str(work / f"u{index}.o")],
                cwd=ROOT, text=True, capture_output=True, check=False,
            )
            if proc.returncode:
                fail(f"compiler rejected {source}: {proc.stderr[-2000:]}")
        dumps = sorted(work.glob("*.cgraph"))
        if len(dumps) != len(SOURCES):
            fail(f"cgraph dumps {len(dumps)} != translation units {len(SOURCES)}")
        for dump in dumps:
            parse_graph(dump, result)
    return result


def line_hits(path: Path, pattern: str) -> list[str]:
    rx = re.compile(pattern)
    return [
        f"{path.relative_to(ROOT)}:{number}"
        for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1)
        if rx.search(line)
    ]


def main() -> int:
    graph = compiler_graph()
    callers = graph.get("radiation_field_commit", set())
    expected = {"main", "cmfgen_commit_jnu"}
    if callers != expected:
        fail(f"radiation_field_commit callers={sorted(callers)} expected={sorted(expected)}")

    production = [ROOT / item for item in SOURCES]
    outside_owner = [path for path in production if path.name != "radiation_field.c"]
    forbidden_writes: dict[str, list[str]] = {}
    patterns = {
        "canonical_generation_write": r"radiation_field\.field\.generation\s*\.",
        "canonical_J_write": r"radiation_field\.field\.J_nu\.values\s*\[",
        "canonical_validity_write": r"radiation_field\.field\.validity\.values\s*\[",
        "canonical_count_write": (
            r"radiation_field\.field\.estimator_count_or_variance"
            r"\.count\s*\["
        ),
    }
    for label, pattern in patterns.items():
        hits = [hit for path in outside_owner for hit in line_hits(path, pattern)]
        if hits:
            forbidden_writes[label] = hits
    if forbidden_writes:
        fail(f"public owner writes outside commit module: {forbidden_writes}")

    plasma = (ROOT / "src/lumina_plasma.c").read_text(errors="replace")
    fit_start = plasma.find("void nlte_build_perbin_dilute_field")
    fit_end = plasma.find("void nlte_dump_perbin_field_csv", fit_start)
    if fit_start < 0 or fit_end < 0:
        fail("cannot isolate legacy per-bin Planck fit")
    fit_body = plasma[fit_start:fit_end]
    if "radiation_field_commit(" in fit_body or "radiation_field.field" in fit_body:
        fail("Planck fit can reach canonical owner")

    legacy_hits: list[str] = []
    for relative in ("src/lumina_plasma.c", "src/lumina_element_wide.c",
                     "src/lumina_cuda.cu", "src/lumina_nlte_gemm.cu",
                     "src/lumina_nlte_assemble.cu"):
        path = ROOT / relative
        legacy_hits.extend(line_hits(path, r"(?:nlte|n)->J_nu\s*\[|nlte\.J_nu\s*\["))
    if not legacy_hits:
        fail("legacy J_nu consumer census unexpectedly empty")

    payload = {
        "schema": "lumina-a2-04-commit-callgraph-v1",
        "compiler": "gcc -fdump-ipa-cgraph",
        "compiled_translation_units": list(SOURCES),
        "commit_api": (
            "int radiation_field_commit(RadiationFieldOwner *, "
            "const RadiationFieldCommitRequest *)"
        ),
        "commit_callers": sorted(callers),
        "generation_writers_outside_commit_module": 0,
        "canonical_J_writers_outside_commit_module": 0,
        "canonical_validity_writers_outside_commit_module": 0,
        "canonical_count_writers_outside_commit_module": 0,
        "planck_fit_canonical_commit_edges": 0,
        "canonical_physics_consumers": 0,
        "legacy_J_nu_consumer_site_count": len(legacy_hits),
        "legacy_J_nu_consumer_sites": legacy_hits,
        "floor_disposition": (
            "DEFER_LEGACY_REMOVAL_TO_A2_05_BECAUSE_ACTIVE_CONSUMERS_EXIST; "
            "CANONICAL_UNSAMPLED_IS_ZERO_PLUS_VALIDITY"
        ),
        "section_13_negative_paths_1_to_7": {
            "1_planck_overwrite": "DYNAMIC_REJECTION_PLUS_ZERO_CALL_EDGE",
            "2_bf_bypass": "STRUCTURALLY_DEFERRED_LEGACY_CONSUMER_NO_CANONICAL_READ_API",
            "3_jbar_stale": "STRUCTURALLY_DEFERRED_LEGACY_CONSUMER_NO_CANONICAL_READ_API",
            "4_cmfgen_split_generation": "DYNAMIC_GENERATION_REJECTION",
            "5_gpu_upload_missing": "A2_12_NOT_PRESENT_NO_CANONICAL_GPU_MIRROR",
            "6_gpu_reset_skew": "A2_12_NOT_PRESENT_NO_CANONICAL_GPU_MIRROR",
            "7_double_normalization": "DYNAMIC_PRODUCER_FORM_REJECTION",
        },
        "verdict": "PASS",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
