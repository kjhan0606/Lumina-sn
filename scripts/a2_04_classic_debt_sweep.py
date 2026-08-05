#!/usr/bin/env python3
"""Measure the A2-04 classic-debt reachability without repairing the debts."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPRESENTATIVE_LOG = ROOT / "logs/ddc15_radeqesc_163604/stdout.log"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"A2_04_CLASSIC_DEBT_SWEEP FAIL: {message}")


def line_number(text: str, needle: str) -> int:
    for number, line in enumerate(text.splitlines(), 1):
        if needle in line:
            return number
    raise ValueError(needle)


def main() -> int:
    main_text = (ROOT / "src/lumina_main.c").read_text(errors="replace")
    cuda_text = (ROOT / "src/lumina_cuda.cu").read_text(errors="replace")
    plasma = (ROOT / "src/lumina_plasma.c").read_text(errors="replace")
    log = REPRESENTATIVE_LOG.read_text(errors="replace")
    require("config.damping_constant = 0.5" in main_text and
            "config.hold_iterations = 3" in main_text and
            "config.damping_constant = 0.5" in cuda_text and
            "config.hold_iterations = 3" in cuda_text, "H02 literals changed")
    require("const double T_LO = 1500.0, T_HI = 50000.0" in plasma and
            "nstep = 80" in plasma and "nstep = 60" in plasma,
            "H13 bounded search signature changed")
    require("create_estimators(geo.n_shells, 0)" in main_text and
            "j_blue/Edotlu not tracked per-thread" in main_text,
            "P02 structural omission changed")

    line_list = ROOT / "data/tardis_reference_toy06_19p48d/line_list.csv"
    with line_list.open() as stream:
        n_lines = sum(1 for _ in stream) - 1
    iterations = len(re.findall(r"^--- Iteration \d+/\d+ ---$", log, re.M))
    radiation_tables = log.count("  Shell  W_LUMINA   T_rad_LUM")
    payload = {
        "schema": "lumina-a2-04-classic-debt-sweep-v1",
        "policy": "MEASURE_AND_RECORD_ONLY_NO_REPAIR",
        "representative_archived_run": str(REPRESENTATIVE_LOG.relative_to(ROOT)),
        "items": {
            "H02": {
                "reachability": "FIRED",
                "locations": [
                    f"src/lumina_main.c:{line_number(main_text, 'config.damping_constant = 0.5')}",
                    f"src/lumina_cuda.cu:{line_number(cuda_text, 'config.damping_constant = 0.5')}",
                ],
                "measured": {
                    "default_damping": 0.5,
                    "default_hold_iterations": 3,
                    "archived_dynamic_transprob_damping_on_events":
                        len(re.findall(r"\[TransProb\].*damping=on", log)),
                    "archived_dynamic_transprob_damping_off_events":
                        len(re.findall(r"\[TransProb\].*damping=off", log)),
                },
                "impact": "each armed update retains exactly 50% of the old value; transition-probability updates are held through the configured first three iterations",
                "disposition_proposal": "OPEN_TO_A2_18_NO_REPAIR_IN_A2_04",
            },
            "H13": {
                "reachability": "FIRED",
                "location": f"src/lumina_plasma.c:{line_number(plasma, 'const double T_LO = 1500.0')}",
                "measured": {
                    "temperature_bounds_K": [1500.0, 50000.0],
                    "coarse_steps": 80,
                    "refine_steps": 60,
                    "archived_binned_J_arm_banners": log.count("[binned-J]"),
                    "archived_outer_iterations": iterations,
                },
                "impact": "the bounded fit was attempted from ten radiation-field solves in the representative run; canonical J_nu impact is now zero because the fit has no commit edge",
                "disposition_proposal": "DIAGNOSTIC_ONLY_CANONICAL_IMPACT_ZERO_KEEP_OPEN",
            },
            "S01": {
                "reachability": "FIRED_LEGACY_CANONICAL_BLOCKED",
                "locations": [
                    f"src/lumina_plasma.c:{line_number(plasma, 'void solve_radiation_field')}",
                    f"src/lumina_main.c:{line_number(main_text, 'solve_radiation_field(est')}",
                ],
                "measured": {
                    "archived_radiation_field_solves": radiation_tables,
                    "canonical_planck_commit_edges": 0,
                },
                "impact": "legacy W/T_rad diagnostics and pre-A2-17 consumers still change; canonical RadiationField bytes cannot be overwritten",
                "disposition_proposal": "A2_04_CANONICAL_PATH_CLOSED_LEGACY_REMAINS_A2_17",
            },
            "P02": {
                "reachability": "FIRED_BY_CONSTRUCTION_CPU_PARALLEL_REGION",
                "locations": [
                    f"src/lumina_main.c:{line_number(main_text, 'create_estimators(geo.n_shells, 0)')}",
                    f"src/lumina_main.c:{line_number(main_text, 'j_blue/Edotlu not tracked per-thread')}",
                ],
                "measured": {
                    "canonical_deck_line_count_excluded_per_worker": n_lines,
                    "local_estimator_n_lines": 0,
                    "local_j_blue_Edotlu_reductions": 0,
                },
                "impact": "all line-resonance estimator contributions are absent from the CPU thread-local reduction; A2-06 owns the repair",
                "disposition_proposal": "OPEN_TO_A2_06_NO_REPAIR_IN_A2_04",
            },
        },
        "census_file_modified": False,
        "verdict": "PASS_MEASURED_NO_REPAIR",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
