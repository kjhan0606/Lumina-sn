#!/usr/bin/env python3
"""Positive and negative controls for the cross-configuration A/B verdict."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts/compare_a210_precore_tau_pair_outcomes.py"


def pair_report(single_total: int, outcome: str) -> dict[str, object]:
    return {
        "schema": "LUMINA_A210_PRECORE_TAU_AB_COMPARISON_V2",
        "status": "PASS",
        "outcome": outcome,
        "exact_r6_identity": "BIT_EXACT",
        "sealed_pair": {
            "controls": {"input/single_total.txt": str(single_total)},
            "environment_identity": "ONLY_PRECORE_TAU_REFRESH_DIFFERS",
            "baseline_precore_tau_refresh": 0,
            "candidate_precore_tau_refresh": 1,
        },
        "physical_values_modified": False,
        "floor": 0,
        "cap": 0,
        "clamp": 0,
        "jitter": 0,
        "repair": 0,
    }


def run(directory: Path, case: str, st0: dict[str, object],
        st1: dict[str, object]) -> tuple[int, dict[str, object]]:
    case_dir = directory / case
    case_dir.mkdir()
    path0 = case_dir / "st0.json"
    path1 = case_dir / "st1.json"
    output = case_dir / "verdict.json"
    path0.write_text(json.dumps(st0) + "\n", encoding="utf-8")
    path1.write_text(json.dumps(st1) + "\n", encoding="utf-8")
    result = subprocess.run(
        (sys.executable, str(CHECKER),
         "--single-total-zero-report", str(path0),
         "--single-total-one-report", str(path1),
         "--report", str(output)),
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    return result.returncode, json.loads(output.read_text(encoding="utf-8"))


def main() -> int:
    restored = "BRACKET_RESTORED_GATE_PASS"
    persists = "NO_BRACKET_PERSISTS"
    with tempfile.TemporaryDirectory(prefix="a210-precore-cross-") as raw:
        directory = Path(raw)
        expected = {
            "both_restored": (restored, restored, "SUPPORTED", "CONSISTENT"),
            "st1_restored": (persists, restored, "SUPPORTED", "CONFIG_DEPENDENT"),
            "both_persist": (persists, persists, "NOT_SUPPORTED", "CONSISTENT"),
            "st1_persist": (restored, persists, "NOT_SUPPORTED", "CONFIG_DEPENDENT"),
        }
        for name, (out0, out1, causal, consistency) in expected.items():
            rc, verdict = run(
                directory, name, pair_report(0, out0), pair_report(1, out1)
            )
            if (rc != 0 or verdict.get("status") != "PASS" or
                    verdict.get("gate_causality") != causal or
                    verdict.get("cross_configuration_consistency") != consistency):
                print(f"FAIL A2_10_PRECORE_TAU_PAIR_OUTCOMES positive={name}")
                return 4

        negatives: dict[str, tuple[dict[str, object], dict[str, object]]] = {}
        bad_schema = pair_report(0, persists)
        bad_schema["schema"] = "OLD"
        negatives["schema"] = (bad_schema, pair_report(1, persists))
        bad_status = pair_report(1, persists)
        bad_status["status"] = "FAIL"
        negatives["status"] = (pair_report(0, persists), bad_status)
        bad_control = pair_report(1, persists)
        bad_control["sealed_pair"]["controls"]["input/single_total.txt"] = "0"  # type: ignore[index]
        negatives["control"] = (pair_report(0, persists), bad_control)
        bad_repair = pair_report(0, persists)
        bad_repair["repair"] = 1
        negatives["repair"] = (bad_repair, pair_report(1, persists))
        for name, (st0, st1) in negatives.items():
            rc, verdict = run(directory, f"negative_{name}", st0, st1)
            if rc != 4 or verdict.get("status") != "FAIL":
                print(f"FAIL A2_10_PRECORE_TAU_PAIR_OUTCOMES negative={name}")
                return 4
    print(
        "PASS A2_10_PRECORE_TAU_PAIR_OUTCOMES_SELFTEST "
        "positive=4 negative=4 repair=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
