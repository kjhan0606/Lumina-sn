#!/usr/bin/env python3
"""Audit the bound-bound opacity identity without changing physical values.

For the J_nu convention used by Lumina's statistical-equilibrium matrix,

    pi e^2/(m_e c) f_lu == h nu B_lu/(4 pi).

The left-hand side is the coefficient used by the Sobolev optical-depth
writer; the right-hand side is the coefficient implied by the Einstein
absorption rate consumed by SE.  A mismatch is especially dangerous when
the radiative-equilibrium line term subtracts two large, nearly equal
quantities.  This tool streams line_list.csv and reports both the global
runtime-constant error and the residual per-line deck serialization error.

It is an audit only: no value is floored, capped, clamped, jittered, repaired,
or written back to the deck.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path


C_CGS = 2.99792458e10
H_CGS = 6.62607015e-27
M_ELECTRON_CGS = 9.1093837015e-28
E_SI_COULOMB = 1.602176634e-19
E_ESU = E_SI_COULOMB * C_CGS / 10.0
EXACT_SOBOLEV_COEFFICIENT = (
    math.pi * E_ESU * E_ESU / (M_ELECTRON_CGS * C_CGS)
)
FOUR_PI = 4.0 * math.pi


def load_runtime_coefficient(header: Path) -> float:
    pattern = re.compile(
        r"^\s*#\s*define\s+SOBOLEV_COEFF\s+"
        r"(?P<value>[+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?)"
    )
    with header.open("r", encoding="utf-8") as stream:
        for line in stream:
            match = pattern.match(line)
            if match:
                value = float(match.group("value"))
                if not math.isfinite(value) or value <= 0.0:
                    raise ValueError("SOBOLEV_COEFF must be finite and positive")
                return value
    raise ValueError(f"SOBOLEV_COEFF not found in {header}")


def audit(line_list: Path, runtime_coefficient: float) -> dict[str, object]:
    required = {"line_id", "f_lu", "nu", "B_lu", "wavelength_cm"}
    count = 0
    invalid = 0
    total_relative = 0.0
    compensation = 0.0
    minimum = math.inf
    maximum = -math.inf
    minimum_line = None
    maximum_line = None
    thresholds = {"1e-5": 0, "1e-6": 0, "1e-7": 0}
    transport_minimum = math.inf
    transport_maximum = -math.inf
    transport_total = 0.0
    transport_compensation = 0.0
    wavelength_frequency_minimum = math.inf
    wavelength_frequency_maximum = -math.inf

    with line_list.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"missing line-list columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            try:
                line_id = int(row["line_id"])
                f_lu = float(row["f_lu"])
                nu = float(row["nu"])
                B_lu = float(row["B_lu"])
                wavelength_cm = float(row["wavelength_cm"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid numeric value at CSV row {row_number}") from exc
            if not (
                math.isfinite(f_lu)
                and math.isfinite(nu)
                and math.isfinite(B_lu)
                and math.isfinite(wavelength_cm)
                and f_lu > 0.0
                and nu > 0.0
                and B_lu > 0.0
                and wavelength_cm > 0.0
            ):
                invalid += 1
                continue

            einstein_coefficient = H_CGS * nu * B_lu / (FOUR_PI * f_lu)
            relative = einstein_coefficient / EXACT_SOBOLEV_COEFFICIENT - 1.0
            wavelength_frequency = wavelength_cm * nu / C_CGS
            runtime_transport_coefficient = (
                runtime_coefficient * wavelength_frequency
            )
            transport_relative = (
                einstein_coefficient / runtime_transport_coefficient - 1.0
            )
            if not (
                math.isfinite(relative)
                and math.isfinite(wavelength_frequency)
                and math.isfinite(transport_relative)
            ):
                invalid += 1
                continue

            # Kahan accumulation keeps the signed mean diagnostic stable even
            # when deck serialization errors are nearly symmetric.
            corrected = relative - compensation
            updated = total_relative + corrected
            compensation = (updated - total_relative) - corrected
            total_relative = updated
            transport_corrected = transport_relative - transport_compensation
            transport_updated = transport_total + transport_corrected
            transport_compensation = (
                (transport_updated - transport_total) - transport_corrected
            )
            transport_total = transport_updated
            count += 1
            if relative < minimum:
                minimum = relative
                minimum_line = line_id
            if relative > maximum:
                maximum = relative
                maximum_line = line_id
            transport_minimum = min(transport_minimum, transport_relative)
            transport_maximum = max(transport_maximum, transport_relative)
            wavelength_frequency_minimum = min(
                wavelength_frequency_minimum, wavelength_frequency - 1.0
            )
            wavelength_frequency_maximum = max(
                wavelength_frequency_maximum, wavelength_frequency - 1.0
            )
            absolute = abs(relative)
            if absolute > 1.0e-5:
                thresholds["1e-5"] += 1
            if absolute > 1.0e-6:
                thresholds["1e-6"] += 1
            if absolute > 1.0e-7:
                thresholds["1e-7"] += 1

    if count == 0:
        raise ValueError("line list contains no finite positive identity rows")

    runtime_relative = runtime_coefficient / EXACT_SOBOLEV_COEFFICIENT - 1.0
    return {
        "schema": "lumina-sobolev-einstein-identity-audit-v1",
        "line_list": str(line_list.resolve()),
        "physical_mutation": 0,
        "repair": 0,
        "constants": {
            "c_cgs": C_CGS,
            "h_cgs": H_CGS,
            "m_electron_cgs": M_ELECTRON_CGS,
            "e_si_coulomb": E_SI_COULOMB,
            "e_esu": E_ESU,
            "exact_sobolev_coefficient": EXACT_SOBOLEV_COEFFICIENT,
            "runtime_sobolev_coefficient": runtime_coefficient,
            "runtime_over_exact_minus_one": runtime_relative,
        },
        "deck_identity": {
            "evaluated_lines": count,
            "invalid_lines": invalid,
            "einstein_over_exact_sobolev_minus_one_min": minimum,
            "minimum_line_id": minimum_line,
            "einstein_over_exact_sobolev_minus_one_max": maximum,
            "maximum_line_id": maximum_line,
            "einstein_over_exact_sobolev_minus_one_mean": total_relative / count,
            "wavelength_nu_over_c_minus_one_min": wavelength_frequency_minimum,
            "wavelength_nu_over_c_minus_one_max": wavelength_frequency_maximum,
            "einstein_over_runtime_transport_minus_one_min": transport_minimum,
            "einstein_over_runtime_transport_minus_one_max": transport_maximum,
            "einstein_over_runtime_transport_minus_one_mean": (
                transport_total / count
            ),
            "absolute_relative_counts": thresholds,
        },
        "verdict": (
            "RUNTIME_CONSTANT_MISMATCH"
            if runtime_coefficient != EXACT_SOBOLEV_COEFFICIENT
            else ("DECK_SERIALIZATION_MISMATCH" if minimum != 0.0 or maximum != 0.0
                  else "EXACT_IDENTITY")
        ),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("line_list", type=Path)
    parser.add_argument("--header", type=Path, default=root / "src" / "lumina.h")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    try:
        runtime = load_runtime_coefficient(args.header)
        report = audit(args.line_list, runtime)
    except (OSError, ValueError) as exc:
        print(f"[sobolev-einstein-audit] ERROR: {exc}", file=sys.stderr)
        return 2

    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
