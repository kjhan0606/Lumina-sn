#!/usr/bin/env python3
"""Known-answer tests for check_sobolev_einstein_identity.py."""

from __future__ import annotations

import csv
import importlib.util
import math
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_sobolev_einstein_identity.py"
SPEC = importlib.util.spec_from_file_location("sobolev_identity_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_fixture(path: Path, coefficient_factors: list[float]) -> None:
    fieldnames = ["line_id", "f_lu", "nu", "B_lu", "wavelength_cm"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for line_id, factor in enumerate(coefficient_factors):
            f_lu = 0.25 + 0.125 * line_id
            nu = 4.0e14 + 1.0e14 * line_id
            coefficient = MODULE.EXACT_SOBOLEV_COEFFICIENT * factor
            B_lu = coefficient * MODULE.FOUR_PI * f_lu / (MODULE.H_CGS * nu)
            writer.writerow(
                {
                    "line_id": line_id,
                    "f_lu": f_lu,
                    "nu": nu,
                    "B_lu": B_lu,
                    "wavelength_cm": MODULE.C_CGS / nu,
                }
            )


def main() -> int:
    exact = MODULE.EXACT_SOBOLEV_COEFFICIENT
    expected = math.pi * MODULE.E_ESU**2 / (
        MODULE.M_ELECTRON_CGS * MODULE.C_CGS
    )
    assert exact == expected

    with tempfile.TemporaryDirectory(prefix="lumina-sobolev-einstein-") as tmp:
        tmp_path = Path(tmp)
        header = tmp_path / "lumina.h"
        header.write_text("#define SOBOLEV_COEFF 2.6540281e-02\n", encoding="utf-8")
        runtime = MODULE.load_runtime_coefficient(header)
        assert runtime == 2.6540281e-2

        exact_csv = tmp_path / "exact.csv"
        write_fixture(exact_csv, [1.0, 1.0])
        exact_report = MODULE.audit(exact_csv, exact)
        exact_deck = exact_report["deck_identity"]
        assert exact_deck["evaluated_lines"] == 2
        assert exact_deck["invalid_lines"] == 0
        assert abs(exact_deck["einstein_over_exact_sobolev_minus_one_min"]) <= 4e-16
        assert abs(exact_deck["einstein_over_exact_sobolev_minus_one_max"]) <= 4e-16

        drift_csv = tmp_path / "drift.csv"
        write_fixture(drift_csv, [1.0 - 2.0e-6, 1.0 + 3.0e-6])
        report = MODULE.audit(drift_csv, runtime)
        deck = report["deck_identity"]
        constants = report["constants"]
        assert report["verdict"] == "RUNTIME_CONSTANT_MISMATCH"
        assert deck["minimum_line_id"] == 0
        assert deck["maximum_line_id"] == 1
        assert abs(deck["einstein_over_exact_sobolev_minus_one_min"] + 2e-6) < 5e-16
        assert abs(deck["einstein_over_exact_sobolev_minus_one_max"] - 3e-6) < 5e-16
        assert deck["absolute_relative_counts"]["1e-6"] == 2
        assert constants["runtime_over_exact_minus_one"] > 7.0e-6
        assert constants["runtime_over_exact_minus_one"] < 7.5e-6

    print("sobolev_einstein_identity_selftest: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
