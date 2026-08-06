#!/usr/bin/env python3
"""Run A2-13~15 GPU oracles and preserve every independent lane verdict."""

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
    stdout = str(oracle["stdout"])
    bf_pass = lifecycle_pass and oracle["rc"] == 0 and \
        "A2_13_BF_CPU_GPU_ORACLE PASS" in stdout
    bb_pass = lifecycle_pass and oracle["rc"] == 0 and \
        "A2_13_BB_CPU_GPU_ORACLE PASS" in stdout
    opacity_pass = lifecycle_pass and oracle["rc"] == 0 and \
        "A2_14_OPACITY_CPU_GPU_ORACLE PASS" in stdout
    emissivity_pass = lifecycle_pass and oracle["rc"] == 0 and \
        "A2_15_EMISSIVITY_CPU_GPU_ORACLE PASS" in stdout
    half_oracle_controls = (contract["rc"] == 0 and
        stdout.count("A2_13_BF_CPU_GPU_ORACLE PASS") == 1 and
        stdout.count("A2_13_BB_CPU_GPU_ORACLE PASS") == 1 and
        "A2_13_NEG_HALF_ORACLE_FAIL" in str(contract["stdout"]))
    a2_13_pass = bf_pass and bb_pass and half_oracle_controls
    micro_pass = a2_13_pass and opacity_pass and emissivity_pass
    marker = ("A2_13_15_BLOCKED_A2_14_15_PRODUCTION_NOT_MIGRATED"
              if micro_pass else "A2_13_15_GPU_ORACLE_FAIL")
    report = {
        "schema": "A2_13_15_GPU_ORACLE_REPORT_V1",
        "stage_status": ("BLOCKED_A2_14_15_PRODUCTION_NOT_MIGRATED"
                         if micro_pass else "FAIL"),
        "marker": marker,
        "a2_12_lifecycle_pass": lifecycle_pass,
        "micro_oracle_pass": micro_pass,
        "bf_oracle": "PASS" if bf_pass else "FAIL_OR_BLOCKED",
        "bb_oracle": "PASS" if bb_pass else "FAIL_OR_BLOCKED",
        "bf_bb_conjunction": "PASS" if a2_13_pass else "FAIL_OR_BLOCKED",
        "half_oracle_negative_controls": "PASS" if half_oracle_controls else "FAIL",
        "opacity_oracle": "PASS" if opacity_pass else "FAIL_OR_BLOCKED",
        "emissivity_oracle": "PASS" if emissivity_pass else "FAIL_OR_BLOCKED",
        "production_guard_removed": {"A2_13_BF": True, "A2_13_BB": True,
                                      "A2_14": False, "A2_15": False},
        "full_nlte_integration_run": False,
        "oracle": oracle,
        "contract": contract,
    }
    out = args.out_dir / "gpu_oracle_report.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (args.out_dir / "gpu_oracle_report.json.sha256").write_text(
        f"{sha256(out)}  gpu_oracle_report.json\n")
    print(f"{marker} bf={bf_pass} bb={bb_pass} conjunction={a2_13_pass} "
          f"opacity={opacity_pass} emissivity={emissivity_pass}")
    return 3 if micro_pass else 4


if __name__ == "__main__":
    raise SystemExit(main())
