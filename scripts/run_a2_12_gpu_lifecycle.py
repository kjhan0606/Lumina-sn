#!/usr/bin/env python3
"""Run the real-CUDA A2-12 positive lane and N1..N9 child controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

EXPECTED = {
    "N1": (41, "A2_12_NEG_CPU_STALE_FAIL"),
    "N2": (42, "A2_12_NEG_CACHE_GENERATION_FAIL"),
    "N3": (43, "A2_12_NEG_LINE_ID_MAPPING_FAIL"),
    "N4": (44, "A2_12_NEG_CPU_GPU_GENERATION_FAIL"),
    "N5": (45, "A2_12_NEG_PARTIAL_UPLOAD_FAIL"),
    "N6": (46, "A2_12_NEG_INVALID_VALIDITY_FAIL"),
    "N7": (47, "A2_12_NEG_FALLBACK_FAIL"),
    "N8": (48, "A2_12_NEG_UPLOAD_BYTES_FAIL"),
    "N9": (49, "A2_12_NEG_RESET_GENERATION_FAIL"),
}


def run(binary: Path, mode: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run((str(binary), mode), text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          check=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve()
    if not binary.is_file():
        parser.error("--binary must exist")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    positive = run(binary, "positive")
    records = {}
    failed = positive.returncode != 0 or "A2_12_GPU_LIFECYCLE PASS" not in positive.stdout
    print(positive.stdout, end="")
    for mode, (expected_rc, marker) in EXPECTED.items():
        proc = run(binary, mode)
        ok = proc.returncode == expected_rc and marker in proc.stdout and \
             "physical_launches=0" in proc.stdout
        records[mode] = {"rc": proc.returncode, "expected_rc": expected_rc,
                         "marker": marker, "physical_launches": 0, "pass": ok,
                         "stdout_sha256": hashlib.sha256(proc.stdout.encode()).hexdigest()}
        print(proc.stdout, end="")
        failed |= not ok
    (args.out_dir / "negative_controls.json").write_text(
        json.dumps(records, indent=2) + "\n")
    report = {
        "stage": "A2-12", "gpu_execution": "COMPLETE",
        "positive_rc": positive.returncode,
        "positive_stdout_sha256": hashlib.sha256(positive.stdout.encode()).hexdigest(),
        "negative_controls_pass": not failed,
    }
    (args.out_dir / "gpu_lifecycle_report.json").write_text(
        json.dumps(report, indent=2) + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
