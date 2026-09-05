#!/usr/bin/env python3
"""Measure the preregistered E-3 formal-spectrum arm/base contrast."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import struct
import sys
import tempfile
from pathlib import Path


SCHEMA = "DUAL_AUTHORITY_SPECTRUM_CONTRAST_E3_V1"
EXPECTED_BINS = 1234
FIELDS = ("wavelength_angstrom", "flux")
HC_EV_ANGSTROM = 12398.419843320026
BANDS = (
    ("[0,2)", 0.0, 2.0),
    ("[2,4)", 2.0, 4.0),
    ("[4,6)", 4.0, 6.0),
    ("[6,12)", 6.0, 12.0),
    ("[12,inf)", 12.0, math.inf),
)


class ContrastError(RuntimeError):
    """Input violates the E-3 measurement contract."""


def resolve_spectrum(path: Path) -> Path:
    return path / "work" / "lumina_spectrum.csv" if path.is_dir() else path


def float_bits(value: float) -> str:
    return struct.pack(">d", value).hex()


def load_spectrum(path: Path) -> list[tuple[float, float]]:
    if path.is_symlink() or not path.is_file():
        raise ContrastError(f"spectrum is not a regular non-symlink file: {path}")
    rows: list[tuple[float, float]] = []
    try:
        with path.open(encoding="utf-8", errors="strict", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != FIELDS:
                raise ContrastError(
                    f"CSV header mismatch: actual={reader.fieldnames!r} expected={FIELDS!r}"
                )
            for line_number, raw in enumerate(reader, 2):
                try:
                    wavelength = float(raw["wavelength_angstrom"])
                    flux = float(raw["flux"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ContrastError(
                        f"invalid numeric field at CSV line {line_number}"
                    ) from exc
                if not math.isfinite(wavelength) or wavelength <= 0.0:
                    raise ContrastError(
                        f"wavelength must be finite and positive at CSV line {line_number}"
                    )
                if not math.isfinite(flux):
                    raise ContrastError(
                        f"flux must be finite at CSV line {line_number}"
                    )
                rows.append((wavelength, flux))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ContrastError(f"cannot read spectrum {path}: {exc}") from exc
    if len(rows) != EXPECTED_BINS:
        raise ContrastError(
            f"spectrum bin count is {len(rows)}, expected {EXPECTED_BINS}: {path}"
        )
    return rows


def band_index(energy_ev: float) -> int:
    for index, (_, lower, upper) in enumerate(BANDS):
        if lower <= energy_ev < upper:
            return index
    raise ContrastError(f"photon energy is outside registered bands: {energy_ev!r}")


def measure(base_path: Path, arm_path: Path) -> dict[str, object]:
    base_csv = resolve_spectrum(base_path)
    arm_csv = resolve_spectrum(arm_path)
    base = load_spectrum(base_csv)
    arm = load_spectrum(arm_csv)
    values: list[list[float]] = [[] for _ in BANDS]
    excluded_by_band = [0 for _ in BANDS]
    excluded_base = 0
    excluded_arm = 0
    excluded_either = 0

    for index, ((base_wave, base_flux), (arm_wave, arm_flux)) in enumerate(
        zip(base, arm, strict=True)
    ):
        if struct.pack(">d", base_wave) != struct.pack(">d", arm_wave):
            raise ContrastError(
                "wavelength grid bit mismatch "
                f"at bin {index}: base={float_bits(base_wave)} arm={float_bits(arm_wave)}"
            )
        base_nonpositive = base_flux <= 0.0
        arm_nonpositive = arm_flux <= 0.0
        energy_ev = HC_EV_ANGSTROM / base_wave
        energy_band = band_index(energy_ev)
        excluded_base += int(base_nonpositive)
        excluded_arm += int(arm_nonpositive)
        if base_nonpositive or arm_nonpositive:
            excluded_either += 1
            excluded_by_band[energy_band] += 1
            continue
        ratio = arm_flux / base_flux
        if not math.isfinite(ratio) or ratio <= 0.0:
            raise ContrastError(
                f"arm/base flux ratio is nonpositive or nonfinite at bin {index}"
            )
        dlog_flux = math.log(ratio)
        if not math.isfinite(dlog_flux):
            raise ContrastError(f"dlog flux is nonfinite at bin {index}")
        values[energy_band].append(dlog_flux)

    bands = []
    for (label, lower, upper), samples, excluded in zip(
        BANDS, values, excluded_by_band, strict=True
    ):
        bands.append(
            {
                "energy_eV": label,
                "lower_eV_inclusive": lower,
                "upper_eV_exclusive": None if math.isinf(upper) else upper,
                "included_bin_count": len(samples),
                "excluded_nonpositive_flux_count": excluded,
                "dlog_flux_median": statistics.median(samples) if samples else None,
                "dlog_flux_maximum": max(samples) if samples else None,
                "dlog_flux_minimum": min(samples) if samples else None,
            }
        )

    return {
        "schema": SCHEMA,
        "status": "PASS",
        "base_spectrum": str(base_csv),
        "arm_spectrum": str(arm_csv),
        "bin_count": len(base),
        "wavelength_grid_bit_identical": True,
        "included_bin_count": len(base) - excluded_either,
        "excluded_nonpositive_flux_count": excluded_either,
        "nonpositive_flux_count_by_input": {
            "base": excluded_base,
            "arm": excluded_arm,
        },
        "bands": bands,
    }


def write_fixture(path: Path, rows: list[tuple[float, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(FIELDS)
        writer.writerows(rows)


def selftest() -> int:
    energies = (1.0, 3.0, 5.0, 9.0, 15.0)
    deltas = (-0.2, -0.1, 0.0, 0.1, 0.2)
    base_rows: list[tuple[float, float]] = []
    arm_rows: list[tuple[float, float]] = []
    for index in range(EXPECTED_BINS):
        slot = index % len(energies)
        wavelength = HC_EV_ANGSTROM / energies[slot]
        base_flux = 2.0 + index / EXPECTED_BINS
        base_rows.append((wavelength, base_flux))
        arm_rows.append((wavelength, base_flux * math.exp(deltas[slot])))

    with tempfile.TemporaryDirectory(prefix="spectrum-contrast-selftest-") as raw:
        root = Path(raw)
        base_path = root / "base.csv"
        arm_path = root / "arm.csv"
        write_fixture(base_path, base_rows)
        write_fixture(arm_path, arm_rows)
        positive = measure(base_path, arm_path)
        expected_by_band: list[list[float]] = [[] for _ in BANDS]
        for (wavelength, base_flux), (_, arm_flux) in zip(
            base_rows, arm_rows, strict=True
        ):
            expected_by_band[band_index(HC_EV_ANGSTROM / wavelength)].append(
                math.log(arm_flux / base_flux)
            )
        direct_formula_ok = all(
            measured["dlog_flux_median"] == statistics.median(expected)
            and measured["dlog_flux_maximum"] == max(expected)
            and measured["dlog_flux_minimum"] == min(expected)
            for measured, expected in zip(
                positive["bands"], expected_by_band, strict=True
            )
        )
        if (
            positive["status"] != "PASS"
            or positive["bin_count"] != EXPECTED_BINS
            or positive["included_bin_count"] != EXPECTED_BINS
            or positive["excluded_nonpositive_flux_count"] != 0
            or any(band["included_bin_count"] == 0 for band in positive["bands"])
            or not direct_formula_ok
        ):
            raise SystemExit("positive E-3 fixture failed")

        mismatch_rows = list(arm_rows)
        wavelength, flux = mismatch_rows[17]
        mismatch_rows[17] = (math.nextafter(wavelength, math.inf), flux)
        write_fixture(arm_path, mismatch_rows)
        try:
            measure(base_path, arm_path)
        except ContrastError as exc:
            if "wavelength grid bit mismatch" not in str(exc):
                raise SystemExit(f"wrong grid-mismatch failure: {exc}") from exc
        else:
            raise SystemExit("grid mismatch was accepted")

        nonpositive_base = list(base_rows)
        nonpositive_arm = list(arm_rows)
        nonpositive_base[7] = (nonpositive_base[7][0], 0.0)
        nonpositive_arm[11] = (nonpositive_arm[11][0], -1.0)
        write_fixture(base_path, nonpositive_base)
        write_fixture(arm_path, nonpositive_arm)
        excluded = measure(base_path, arm_path)
        if (
            excluded["status"] != "PASS"
            or excluded["excluded_nonpositive_flux_count"] != 2
            or excluded["included_bin_count"] != EXPECTED_BINS - 2
            or excluded["nonpositive_flux_count_by_input"] != {"base": 1, "arm": 1}
        ):
            raise SystemExit("nonpositive-flux exclusion fixture failed")

    print(
        "SPECTRUM_CONTRAST_SELFTEST_PASS positive=1 negative=2 "
        "controls=GRID_BIT_MISMATCH_REJECTED,NONPOSITIVE_FLUX_EXCLUDED"
    )
    return 0


def arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", nargs="?", type=Path)
    parser.add_argument("arm", nargs="?", type=Path)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)
    if args.selftest:
        if args.base is not None or args.arm is not None:
            parser.error("--selftest does not accept spectrum paths")
    elif args.base is None or args.arm is None:
        parser.error("BASE and ARM spectrum paths or run roots are required")
    return args


def main(argv: list[str] | None = None) -> int:
    args = arguments(sys.argv[1:] if argv is None else argv)
    if args.selftest:
        return selftest()
    try:
        payload = measure(args.base, args.arm)
        rc = 0
    except ContrastError as exc:
        payload = {"schema": SCHEMA, "status": "FAIL", "error": str(exc)}
        rc = 2
    json.dump(payload, sys.stdout, indent=2, sort_keys=True, allow_nan=False)
    sys.stdout.write("\n")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
