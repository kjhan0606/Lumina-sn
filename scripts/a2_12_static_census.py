#!/usr/bin/env python3
"""Freeze and verify the A2-12 130-row CUDA consumer census."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "068fb36"
FILES = (
    "src/lumina_cuda.cu",
    "src/lumina_bf_gemm.cu",
    "src/lumina_nlte_assemble.cu",
    "src/lumina_cmf_solve.cu",
    "src/lumina_nlte_gemm.cu",
)
EXPECTED = {
    "src/lumina_cuda.cu": 86,
    "src/lumina_bf_gemm.cu": 13,
    "src/lumina_nlte_assemble.cu": 14,
    "src/lumina_cmf_solve.cu": 10,
    "src/lumina_nlte_gemm.cu": 7,
}
PATTERN = re.compile(
    r"\bd_T_rad\b|\bd_W\b|\bd_jbar_line\b|\bd_jbar_count\b|"
    r"\bd_jblue_line\b|\bd_j_nu_estimator\b|\bd_j_nu_count\b|"
    r"\bd_J_nu\b|\bd_J\b|\bd_Jnew\b|RadiationField|LineJbar|"
    r"line_jbar|radiation_field"
)


def frozen_text(path: str) -> str:
    proc = subprocess.run(
        ("git", "show", f"{BASE}:{path}"), cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if proc.returncode:
        raise RuntimeError(proc.stderr.strip())
    return proc.stdout


def disposition(path: str, text: str) -> tuple[str, str]:
    if path.endswith("lumina_cmf_solve.cu"):
        return "CMF_J_GPU_TRANSFER", "BLOCKED_GPU_FALLBACK_FORBIDDEN"
    if path.endswith("lumina_nlte_gemm.cu"):
        return "NLTE_RATE_GEMM", "GPU_RATE_NOT_MIGRATED:A2-13_DEFER"
    if path.endswith("lumina_nlte_assemble.cu"):
        return "NLTE_ASSEMBLY_LEGACY_FIELD", "GPU_RATE_NOT_MIGRATED:A2-13_DEFER"
    if path.endswith("lumina_bf_gemm.cu"):
        return "BF_GEMM_SCALAR", "GPU_OPACITY_NOT_MIGRATED:A2-14_DEFER"
    if "d_jbar_line" in text or "d_jbar_count" in text:
        return "LEGACY_LINE_PRODUCER", "PRODUCER_ONLY:NOT_CANONICAL_MIRROR"
    if "d_jblue_line" in text:
        return "LEGACY_BLUE_WING_PRODUCER", "PRODUCER_ONLY:NOT_CANONICAL_MIRROR"
    if "d_j_nu_estimator" in text or "d_j_nu_count" in text:
        return "LEGACY_GLOBAL_J_PRODUCER", "PRODUCER_ONLY:NOT_CANONICAL_MIRROR"
    if "d_T_rad" in text or "d_W" in text:
        return "TRANSPORT_SCALAR", "GPU_EMISSIVITY_NOT_MIGRATED:A2-15_DEFER"
    return "CANONICAL_NAME_REFERENCE", "LIFECYCLE_ONLY"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    counts: dict[str, int] = {}
    for path in FILES:
        text = frozen_text(path)
        matched = []
        for lineno, line in enumerate(text.splitlines(), 1):
            if PATTERN.search(line):
                group, action = disposition(path, line)
                matched.append({
                    "census_id": f"C{len(rows) + len(matched) + 1:03d}",
                    "file": path,
                    "baseline_line": lineno,
                    "source_sha256": hashlib.sha256(line.encode()).hexdigest(),
                    "group": group,
                    "disposition": action,
                })
        rows.extend(matched)
        counts[path] = len(matched)
    failures = []
    if counts != EXPECTED or len(rows) != 130:
        failures.append(f"count mismatch counts={counts} total={len(rows)}")
    current = {p.name for p in (ROOT / "src").glob("*.cu")}
    required = {Path(p).name for p in FILES} | {"gpu_radiation_field.cu"}
    outside = sorted(current - required)
    mirror = (ROOT / "src/gpu_radiation_field.cu").read_text()
    if mirror.count("GpuRadiationFieldMirror *gpu_radiation_field_create") != 1:
        failures.append("canonical mirror owner/create count is not one")
    wiring = (ROOT / "scripts/run_gate_battery.py").read_text()
    if wiring.count('"src/gpu_radiation_field_contract.c"') != 5:
        failures.append("Z four-link plus Z-a2-12 TU wiring incomplete")
    zrunner = (ROOT / "scripts/run_zinert_selftest.py").read_text()
    for token in ("--a2-12-contract", "args.a2_12_contract",
                  "a2-12-gpu-lifecycle-contract"):
        if token not in zrunner:
            failures.append(f"Z runner missing {token}")
    cmf = (ROOT / "src/lumina_cmfgen.c").read_text()
    if "-> CPU fallback" in cmf or "run_cpu = 1" in cmf:
        failures.append("CMF same-attempt CPU fallback remains")
    for token in ("fallback_attempts=1", "physical_launches=0",
                  "BLOCKED_GPU_FALLBACK_FORBIDDEN"):
        if token not in cmf:
            failures.append(f"CMF failure evidence missing {token}")
    payload = {
        "schema": "A2_12_CUDA_CONSUMER_CENSUS_V1",
        "baseline": BASE,
        "tracked_build_authoritative_files": list(FILES),
        "counts": counts,
        "total_rows": len(rows),
        "rows": rows,
        "discovered_outside_census": outside,
        "archival_untracked_excluded": [
            "backup_groupA_1422/lumina_cuda.cu",
            "backup_groupA_1422/lumina_nlte_assemble.cu",
            "impl_withParityAA/orig/lumina_cuda.cu",
            "impl_withParityW/orig/lumina_cuda.cu",
            "impl_withParityY/orig/lumina_cuda.cu",
        ],
        "status": "FAIL" if failures else "PASS",
        "failures": failures,
    }
    if args.write:
        out = ROOT / "validation/a2_12/cuda_consumer_census.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"A2_12_STATIC_CENSUS {payload['status']} rows={len(rows)} counts={counts}")
    for failure in failures:
        print(f"FAIL {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
