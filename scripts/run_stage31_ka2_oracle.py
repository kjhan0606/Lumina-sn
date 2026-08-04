#!/usr/bin/env python3
"""Run the preregistered KA2 Nyström reference-convergence gate."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from stage31_ka2_oracle import relative_difference, solve


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    targets = [(i + 0.5) / 512.0 for i in range(512)]
    reference_2048 = solve(2048, targets)
    reference_4096 = solve(4096, targets)
    difference = relative_difference(reference_2048["J"], reference_4096["J"])
    acceptance = {
        "Nref_relative_difference_lt_1e-9": difference < 1.0e-9,
        "full_oracle_arithmetic_is_80_digit":
            reference_4096["matrix_storage"] == "80-digit",
    }
    report = {
        "rung": 10,
        "method": "Gauss-Legendre Nystrom with analytic logarithmic singularity subtraction",
        "mpmath_dps": 80,
        "Nref": [2048, 4096],
        "relative_difference": difference,
        "acceptance": acceptance,
        "Nref_2048_iterations": reference_2048["iterations"],
        "Nref_4096_iterations": reference_4096["iterations"],
        "matrix_storage": reference_4096["matrix_storage"],
        "status": "PASS" if all(acceptance.values()) else "FAIL",
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
