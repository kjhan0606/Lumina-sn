#!/usr/bin/env python3
"""Fail-closed verdict for the production CMF exact CPU/multi-GPU A/B flight."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


def field(line: str, name: str) -> str:
    match = re.search(rf"(?:^| ){re.escape(name)}=([^ ]+)", line)
    if not match:
        raise ValueError(f"missing {name}")
    return match.group(1)


def finite_float(line: str, name: str) -> float:
    value = float(field(line, name))
    if not math.isfinite(value):
        raise ValueError(f"nonfinite {name}={value}")
    return value


def finite_range(line: str, name: str) -> tuple[float, float]:
    match = re.search(
        rf"(?:^| ){re.escape(name)}=\[([^,\]]+),([^\]]+)\]", line
    )
    if not match:
        raise ValueError(f"missing range {name}")
    lo, hi = float(match.group(1)), float(match.group(2))
    if not math.isfinite(lo) or not math.isfinite(hi) or lo > hi:
        raise ValueError(f"invalid range {name}=[{lo},{hi}]")
    return lo, hi


def only_line(text: str, marker: str) -> tuple[int, str]:
    matches = [(m.start(), line) for m in re.finditer(r"^.*$", text, re.M)
               if marker in (line := m.group(0))]
    if len(matches) != 1:
        raise ValueError(f"expected one {marker!r} line, found {len(matches)}")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stderr", type=Path, required=True)
    parser.add_argument("--model-rc", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-cells", type=int, default=100_655_650)
    parser.add_argument("--expected-bins", type=int, default=2_013_113)
    parser.add_argument("--expected-q-lines", type=int, default=1_603_732)
    parser.add_argument("--expected-e-lines", type=int, default=2_180_286)
    parser.add_argument("--expected-devices", type=int, default=4)
    parser.add_argument("--require-exit-after-r6", action="store_true")
    parser.add_argument("--require-external-fixture", action="store_true")
    args = parser.parse_args()
    if args.expected_devices <= 0:
        parser.error("--expected-devices must be positive")

    text = args.stderr.read_text(errors="replace")
    failures: list[str] = []
    report: dict[str, object] = {
        "schema": "lumina-cmf-exact-cmfgen-ab-v1",
        "verdict": "BLOCKED",
    }
    try:
        epoch_pos, epoch = only_line(text, "[cmf_fine][EXACT-MULTIGPU-EPOCH]")
        ab_pos, ab = only_line(text, "[cmf_fine][EXACT-MULTIGPU-AB]")
        r6_pos, r6 = only_line(text, "[R6][LINE-IDENTITY]")
        coverage_pos, coverage = only_line(text, "[R6][LINE-COVERAGE]")
        if not (epoch_pos < ab_pos < r6_pos < coverage_pos):
            raise ValueError("producer/AB/R6 marker order mismatch")
        expected_device_field = (
            f"{args.expected_devices}/{args.expected_devices}"
        )
        if (field(epoch, "status") != "OK" or
                field(epoch, "devices") != expected_device_field):
            raise ValueError(
                "multi-GPU epoch did not report status=OK "
                f"devices={expected_device_field}"
            )
        if "floor=0" not in epoch or "clamp=0" not in epoch or "jitter=0" not in epoch:
            raise ValueError("multi-GPU epoch repair contract is not zero")
        if " PASS " not in f" {ab} ":
            raise ValueError("A/B marker is not PASS")
        if int(field(ab, "cells")) != args.expected_cells:
            raise ValueError("A/B cell count mismatch")
        if int(field(ab, "devices")) != args.expected_devices:
            raise ValueError("A/B device count mismatch")
        if not all(token in ab for token in
                   ("floor=0", "clamp=0", "jitter=0", "repair=0")):
            raise ValueError("A/B repair contract is not zero")

        cpu_range = finite_range(ab, "finite_cpu_J")
        gpu_range = finite_range(ab, "finite_gpu_J")
        cpu_error = finite_range(ab, "cpu_component_error")
        gpu_error = finite_range(ab, "gpu_component_error")
        if cpu_range[0] < 0.0 or gpu_range[0] < 0.0:
            raise ValueError("negative finite J range")
        if not (cpu_range[1] > 0.0 and gpu_range[1] > 0.0):
            raise ValueError("CPU or GPU failed to reproduce a positive finite J")
        if cpu_error[0] < 0.0 or gpu_error[0] < 0.0:
            raise ValueError("negative component error range")
        max_relative_j = finite_float(ab, "max_relative_J")
        max_relative_error_width = finite_float(
            ab, "max_relative_error_width"
        )
        max_distance = finite_float(
            ab, "max_distance_over_combined_envelope"
        )
        if max_relative_j < 0.0 or max_relative_j > 1.0e-12:
            raise ValueError(f"max_relative_J out of contract: {max_relative_j}")
        if max_distance < 0.0 or max_distance > 1.0:
            raise ValueError(
                "CPU/GPU point estimates are outside their combined envelopes"
            )

        if int(field(r6, "fine_bins")) != args.expected_bins:
            raise ValueError("R6 fine-bin count mismatch")
        if int(field(r6, "q_lines")) != args.expected_q_lines:
            raise ValueError("R6 Q-line count mismatch")
        if int(field(r6, "e_lines")) != args.expected_e_lines:
            raise ValueError("R6 E-line count mismatch")
        if int(field(coverage, "valid_lines")) != args.expected_e_lines:
            raise ValueError("R6 valid-line count mismatch")
        if int(field(coverage, "partial_lines")) != 0:
            raise ValueError("R6 partial lines are present")
        if int(field(coverage, "unsampled_lines")) != 0:
            raise ValueError("R6 unsampled lines are present")

        if args.require_exit_after_r6:
            exit_pos, exit_line = only_line(
                text, "[VALIDATION][EXIT-AFTER-R6]"
            )
            if coverage_pos >= exit_pos or " PASS " not in f" {exit_line} ":
                raise ValueError("exit-after-R6 marker/order mismatch")
            if ("downstream_r7=NOT_RUN" not in exit_line or
                    "a2_10=NOT_RUN" not in exit_line):
                raise ValueError("exit-after-R6 downstream scope mismatch")
            if "[R7][" in text or "[A2-10][" in text:
                raise ValueError("post-R6 stage ran despite validation exit")

        if args.require_external_fixture:
            fixture_pos, fixture_line = only_line(
                text, "[cmf_fine][EXTERNAL-JNU-FIXTURE]"
            )
            if not (ab_pos < fixture_pos < r6_pos):
                raise ValueError("external fixture marker/order mismatch")
            if (" PASS " not in f" {fixture_line} " or
                    int(field(fixture_line, "rows")) != 400 or
                    field(fixture_line, "quantity") != "J_nu"):
                raise ValueError("external J_nu fixture contract mismatch")

        if "[cmf_fine][BLOCKED]" in text:
            raise ValueError("fine-grid producer emitted BLOCKED")
        model_rc: int | None = None
        downstream = "NOT_EVALUATED"
        if args.model_rc is not None:
            model_rc = int(args.model_rc.read_text().strip())
            downstream = "COMPLETE" if model_rc == 0 else "BLOCKED_AFTER_R6"
            if model_rc != 0 and not (
                "[A2-10][BLOCKED] event=R7_MATERIAL_UPDATE_BLOCKED" in text
                and "[R7][FATAL] lane=DET iter=0 rc=4" in text
            ):
                raise ValueError(
                    f"unexpected post-R6 model failure, process rc={model_rc}"
                )

        report.update({
            "verdict": "PASS",
            "cells": args.expected_cells,
            "fine_bins": args.expected_bins,
            "q_lines": args.expected_q_lines,
            "e_lines": args.expected_e_lines,
            "devices": args.expected_devices,
            "cpu_iterations": int(field(ab, "cpu_iterations")),
            "gpu_iterations": int(field(ab, "gpu_iterations")),
            "finite_cpu_J": cpu_range,
            "finite_gpu_J": gpu_range,
            "cpu_component_error": cpu_error,
            "gpu_component_error": gpu_error,
            "max_relative_J": max_relative_j,
            "max_relative_error_width": max_relative_error_width,
            "max_distance_over_combined_envelope": max_distance,
            "downstream_status": downstream,
            "exit_after_r6": args.require_exit_after_r6,
            "external_fixture": args.require_external_fixture,
            "repair": {"floor": 0, "clamp": 0, "jitter": 0},
        })
        if model_rc is not None:
            report["model_process_rc"] = model_rc
    except (OSError, ValueError) as exc:
        failures.append(str(exc))

    if failures:
        report["failures"] = failures
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
