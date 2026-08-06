#!/usr/bin/env python3
"""Run the A2-13~15 GPU micro-oracle without claiming production closure."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(binary: Path) -> dict[str, object]:
    proc = subprocess.run([str(binary.resolve())], text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return {"rc": proc.returncode, "stdout": proc.stdout,
            "stderr": proc.stderr, "binary_sha256": sha256(binary)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--a2-12-report", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if not all(p.is_file() for p in (args.oracle, args.contract,
                                     args.a2_12_report)):
        parser.error("oracle, contract, and A2-12 report must exist")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    lifecycle = json.loads(args.a2_12_report.read_text())
    lifecycle_pass = (lifecycle.get("positive_rc") == 0 and
                      lifecycle.get("negative_controls_pass") is True)
    oracle = run(args.oracle) if lifecycle_pass else {
        "rc": 3, "stdout": "", "stderr": "A2_12_LIFECYCLE_NOT_PASS\n",
        "binary_sha256": sha256(args.oracle)}
    contract = run(args.contract)
    marker = "A2_13_15_BLOCKED_PRODUCTION_NOT_MIGRATED"
    micro_pass = (lifecycle_pass and oracle["rc"] == 0 and
                  "A2_13_15_GPU_ORACLE PASS" in str(oracle["stdout"]) and
                  contract["rc"] == 0)
    report = {
        "schema": "A2_13_15_GPU_ORACLE_REPORT_V1",
        "stage_status": "BLOCKED_PRODUCTION_NOT_MIGRATED",
        "marker": marker,
        "a2_12_lifecycle_pass": lifecycle_pass,
        "micro_oracle_pass": micro_pass,
        "bf_oracle": "PASS" if micro_pass else "FAIL_OR_BLOCKED",
        "bb_oracle": "PASS" if micro_pass else "FAIL_OR_BLOCKED",
        "bf_bb_conjunction": "PASS" if micro_pass else "FAIL_OR_BLOCKED",
        "opacity_oracle": "PASS" if micro_pass else "FAIL_OR_BLOCKED",
        "emissivity_oracle": "PASS" if micro_pass else "FAIL_OR_BLOCKED",
        "production_guard_removed": False,
        "full_nlte_integration_run": False,
        "oracle": oracle,
        "contract": contract,
    }
    out = args.out_dir / "gpu_oracle_report.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (args.out_dir / "gpu_oracle_report.json.sha256").write_text(
        f"{sha256(out)}  gpu_oracle_report.json\n")
    print(f"{marker} micro_oracle_pass={str(micro_pass).lower()} "
          "production_guard_removed=false")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
