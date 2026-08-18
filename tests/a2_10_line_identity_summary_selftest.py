#!/usr/bin/env python3
"""Known-answer test for summarize_a210_line_identity.py."""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "summarize_a210_line_identity.py"
SPEC = importlib.util.spec_from_file_location("line_identity_summary", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def record(phase: str, shell: int, current: float) -> str:
    exact = current + 0.25
    einstein = exact - 0.0625
    constant_delta = exact - current
    serialization_delta = einstein - exact
    total_delta = einstein - current
    return (
        f"{MODULE.PREFIX}phase={phase} shell={shell} "
        f"sobolev_signed_rate={current:.17g} "
        f"exact_constant_rate={exact:.17g} "
        f"einstein_consistent_rate={einstein:.17g} "
        f"constant_delta={constant_delta:.17g} "
        f"serialization_delta={serialization_delta:.17g} "
        f"delta={total_delta:.17g} scaled_emission=10 scaled_absorption=9 "
        f"cancellation_condition=19 positive_tau_delta={total_delta:.17g} "
        "negative_tau_delta=0 "
        f"constant_positive_tau_delta={constant_delta:.17g} "
        "constant_negative_tau_delta=0 "
        f"serialization_positive_tau_delta={serialization_delta:.17g} "
        "serialization_negative_tau_delta=0 raw_cells=7 srce_chk_cells=1 "
        "interpretation=DIAGNOSTIC_ONLY repair=0\n"
    )


def endpoint(phase: str, shell: int, heating: float, cooling: float) -> str:
    residual = heating - cooling
    return (
        f"{MODULE.ENDPOINT_PREFIX}phase={phase} shell={shell} "
        f"T_e_K={3500.0 if phase == 'LOWER' else 140000.0:.17g} "
        f"heating={heating:.17g} cooling={cooling:.17g} "
        f"residual={residual:.17g}\n"
    )


def interior_shell(phase: str, shell: int, residual: float) -> str:
    cooling = 10.0
    heating = cooling + residual
    return (
        f"{MODULE.INTERIOR_PREFIX}phase={phase} shell={shell} T_mid=10020 "
        f"res_lo=-2 res_mid={residual:.17g} res_hi=-4 "
        f"heat_mid={heating:.17g} cool_mid={cooling:.17g} "
        "line_emit_mid=8 lo_mid_bracket=0 mid_hi_bracket=0 "
        "action=DIAGNOSTIC_ONLY\n"
    )


def interior_summary(phase: str, shell_count: int) -> str:
    return (
        f"{MODULE.INTERIOR_PREFIX}phase={phase} valid=1 "
        f"endpoint_no_bracket={shell_count} interior_bracket=0 "
        f"still_same_sign={shell_count} action=DIAGNOSTIC_ONLY "
        "solver_result=RADEQ_NO_BRACKET\n"
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="lumina-a210-line-identity-") as tmp:
        path = Path(tmp) / "stderr.log"
        path.write_text(
            "unrelated line\n"
            + record("LOWER", 0, -2.0)
            + record("LOWER", 1, 3.0)
            + record("UPPER", 0, -4.0)
            + record("UPPER", 1, -5.0),
            encoding="utf-8",
        )
        report = MODULE.summarize(path, 2)
        assert report["verdict"] == "COMPLETE"
        assert report["records"] == 4
        assert report["phase_batch_counts"] == {"LOWER": 1, "UPPER": 1}
        lower = report["batches"]["LOWER"][0]
        assert lower["sign_counts"]["current"] == {
            "positive": 1,
            "negative": 1,
            "zero": 0,
        }
        assert lower["shells"][0]["repair"] == 0
        assert lower["shells"][0]["tau_partition_closure"] == 0.0

        path.write_text(
            record("LOWER", 0, -2.0)
            + record("LOWER", 1, 3.0)
            + endpoint("LOWER", 0, 0.0, 0.1)
            + endpoint("LOWER", 1, 1.0, 0.0)
            + record("UPPER", 0, -4.0)
            + record("UPPER", 1, -5.0)
            + endpoint("UPPER", 0, 0.0, 1.0)
            + endpoint("UPPER", 1, 0.0, 1.0),
            encoding="utf-8",
        )
        report = MODULE.summarize(path, 2, require_endpoints=True)
        counterfactual = report["endpoint_counterfactual"]
        assert counterfactual["bracket_counts"] == {
            "current": 1,
            "exact_constant": 1,
            "einstein": 1,
        }
        assert counterfactual["physical_mutation"] == 0
        assert counterfactual["repair"] == 0
        shell0 = counterfactual["shells"][0]
        assert shell0["lower"]["current_residual"] == -0.1
        assert shell0["lower"]["exact_constant_residual"] == -0.35
        assert shell0["lower"]["einstein_consistent_residual"] == -0.2875

        path.write_text(
            record("LOWER", 0, -2.0)
            + record("LOWER", 1, -3.0)
            + endpoint("LOWER", 0, 0.0, 2.0)
            + endpoint("LOWER", 1, 0.0, 3.0)
            + record("UPPER", 0, -4.0)
            + record("UPPER", 1, -5.0)
            + endpoint("UPPER", 0, 0.0, 4.0)
            + endpoint("UPPER", 1, 0.0, 5.0)
            + record("INTERIOR", 0, -3.0)
            + record("INTERIOR", 1, -4.0)
            + interior_shell("PUBLIC_SEED", 0, -3.0)
            + interior_shell("PUBLIC_SEED", 1, -4.0)
            + interior_summary("PUBLIC_SEED", 2),
            encoding="utf-8",
        )
        report = MODULE.summarize(path, 2, require_endpoints=True)
        assert report["interior_identity_phase_map"] == [
            {"batch_index": 0, "phase": "PUBLIC_SEED"}
        ]
        scan = report["interior_scans"][0]
        assert scan["valid"] == 1
        assert scan["endpoint_no_bracket"] == 2
        assert scan["interior_bracket"] == 0
        assert scan["shells"][0]["res_mid"] == -3.0
    print("a2_10_line_identity_summary_selftest: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
