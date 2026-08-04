#!/usr/bin/env python3
"""Read-only six-gate verifier for the toy06 CMFGEN-composition deck."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

from toy06_cmfgen_composition import (
    FOUR_PI_OVER_THREE,
    KM_CM,
    MSUN_G,
    conservative_map,
    parse_sn_hydro_data,
    parse_species_masses,
    read_geometry,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_DEFAULT = ROOT / "data/tardis_reference_toy06_19p48d_sivcaiv_ftos"
DECK_DEFAULT = ROOT / "data/tardis_reference_toy06_19p48d_cmfgencomp"
RUN_DEFAULT = Path("/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4")
EXPECTED_Z = (14, 16, 20, 26, 27, 28)
SUM_TOL = 5.0e-9
MASS_TOTAL_TOL_MSUN = 5.0e-6  # SPECIES_MASSES total is printed to 1e-5 Msun.
MASS_SPECIES_TOL_MSUN = 5.0e-5  # Per-species entries are printed to 1e-4 Msun.
VALUE_TOL = 5.0e-10
MUTABLE = {
    "abundances.csv", "density.csv", "isotopes.csv", "verification.log",
    "composition_mapping_report.json", "composition_preregistration.json",
    "COMPOSITION_INVALID",
}


def read_abundances(path: Path) -> tuple[list[str], dict[int, list[float]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        rows: dict[int, list[float]] = {}
        for row in reader:
            z = int(row[0])
            if z in rows:
                raise ValueError(f"duplicate abundance row Z={z}")
            rows[z] = [float(value) for value in row[1:]]
    return header, rows


def read_isotopes(path: Path) -> tuple[list[str], dict[str, tuple[int, int, list[float]]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        rows: dict[str, tuple[int, int, list[float]]] = {}
        for row in reader:
            name = row[0]
            if name in rows:
                raise ValueError(f"duplicate isotope row {name}")
            rows[name] = (int(row[1]), int(row[2]), [float(value) for value in row[3:]])
    return header, rows


def read_density(path: Path) -> list[float]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if [int(row["shell_id"]) for row in rows] != list(range(len(rows))):
        raise ValueError("density shell_id is not contiguous")
    return [float(row["rho"]) for row in rows]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gate1(deck: Path, failures: list[str]):
    print("GATE 1 — exact elemental abundance set")
    try:
        header, rows = read_abundances(deck / "abundances.csv")
        passed = tuple(sorted(rows)) == EXPECTED_Z
        print(f"  observed Z={sorted(rows)}; expected={list(EXPECTED_Z)} "
              f"{'PASS' if passed else 'FAIL'}")
        if not passed:
            failures.append(f"gate1 element set={sorted(rows)}")
        return header, rows
    except Exception as exc:
        print(f"  abundance input unavailable/invalid: {exc} FAIL")
        failures.append(f"gate1 abundance input: {exc}")
        return [], {}


def gate2(header, rows, deck: Path, expected, failures: list[str]):
    print(f"GATE 2 — shell sums and isotope subsets (abs tolerance={SUM_TOL:.1e})")
    ok = len(header) == 51 and header == ["atomic_number", *map(str, range(50))]
    max_error = math.inf
    if rows and all(len(values) == 50 for values in rows.values()):
        sums = [sum(rows[z][s] for z in EXPECTED_Z) for s in range(50)]
        finite = all(math.isfinite(value) for values in rows.values() for value in values)
        max_error = max(abs(value - 1.0) for value in sums)
        ok = ok and finite and max_error <= SUM_TOL
    else:
        ok = False
    print(f"  50 shell columns={len(header) == 51}; max|sumX-1|={max_error:.6e} "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append(f"gate2 elemental sum/shape max_error={max_error}")

    isotope_ok = True
    try:
        iso_header, isotopes = read_isotopes(deck / "isotopes.csv")
        identities = {"Ni56": (28, 56), "Co56": (27, 56), "Fe56": (26, 56)}
        isotope_ok = (iso_header == ["isotope", "atomic_number", "mass_number",
                                     *map(str, range(50))] and
                      set(isotopes) == set(identities))
        for name, (z, mass) in identities.items():
            actual_z, actual_mass, values = isotopes[name]
            isotope_ok &= actual_z == z and actual_mass == mass and len(values) == 50
            isotope_ok &= all(math.isfinite(value) and value >= 0.0 for value in values)
            if z in rows and len(rows[z]) == 50:
                isotope_ok &= all(value <= element + VALUE_TOL
                                  for value, element in zip(values, rows[z]))
            isotope_ok &= all(abs(a - b) <= VALUE_TOL
                              for a, b in zip(values, expected.isotopes[name]))
        print(f"  isotope rows={sorted(isotopes)}; exact mapped subsets "
              f"{'PASS' if isotope_ok else 'FAIL'}")
    except Exception as exc:
        isotope_ok = False
        print(f"  isotope input unavailable/invalid: {exc} FAIL")
    if not isotope_ok:
        failures.append("gate2 isotope set/subset/mapping")


def gate3(deck: Path, shells, expected, species, species_total, failures: list[str]):
    print("GATE 3 — finite-volume mass conservation against SPECIES_MASSES")
    try:
        _header, rows = read_abundances(deck / "abundances.csv")
        density = read_density(deck / "density.csv")
        if len(density) != 50 or len(shells) != 50:
            raise ValueError(f"density/geometry shell counts={len(density)}/{len(shells)}")
        masses = {name: 0.0 for name in species}
        z_name = {14: "Si", 16: "S", 20: "Ca", 26: "Fe", 27: "Co", 28: "Ni"}
        for s, shell in enumerate(shells):
            volume = FOUR_PI_OVER_THREE * (shell.r_outer_cm ** 3 - shell.r_inner_cm ** 3)
            shell_mass = density[s] * volume
            for z in EXPECTED_Z:
                masses[z_name[z]] += shell_mass * rows[z][s] / MSUN_G
        total = sum(masses.values())
        per_species = {name: abs(masses[name] - species[name]) for name in species}
        passed = (abs(total - species_total) <= MASS_TOTAL_TOL_MSUN and
                  all(delta <= MASS_SPECIES_TOL_MSUN for delta in per_species.values()))
        print(f"  mapped total={total:.9f} Msun, table={species_total:.9f}, "
              f"delta={total-species_total:+.3e}; per-species max delta="
              f"{max(per_species.values()):.3e} {'PASS' if passed else 'FAIL'}")
        if not passed:
            failures.append(f"gate3 total={total:.9f}, species deltas={per_species}")
    except Exception as exc:
        print(f"  mass inputs unavailable/invalid: {exc} FAIL")
        failures.append(f"gate3 mass input: {exc}")
    integrated_total = sum(expected.source_species_mass_msun.values())
    overlap_total = sum(expected.mapped_species_mass_msun.values())
    print(f"  canonical integrated full={integrated_total:.9f} Msun; "
          f"geometry overlap={overlap_total:.9f} Msun")


def gate4(rows, expected, failures: list[str]):
    print("GATE 4 — representative deep/outer stratification")
    fully_covered = [s for s, value in enumerate(expected.volume_coverage)
                     if math.isclose(value, 1.0, rel_tol=0.0, abs_tol=2.0e-12)]
    passed = bool(rows and fully_covered)
    representatives = [0, fully_covered[-1]] if fully_covered else []
    max_delta = math.inf
    if passed:
        deltas = [abs(rows[z][s] - expected.abundances[z][s])
                  for s in representatives for z in EXPECTED_Z]
        max_delta = max(deltas)
        passed = max_delta <= VALUE_TOL
        for s in representatives:
            print("  shell %d: Si=%.9g S=%.9g Ca=%.9g Fe=%.9g Co=%.9g Ni=%.9g" %
                  (s, *(rows[z][s] for z in EXPECTED_Z)))
    print(f"  representative max delta={max_delta:.3e} "
          f"{'PASS' if passed else 'FAIL'}")
    if not passed:
        failures.append(f"gate4 stratification max_delta={max_delta}")


def gate5(header, rows, shells, expected, failures: list[str]):
    print("GATE 5 — all 50 shells populated and canonically covered")
    incomplete = [s for s, value in enumerate(expected.volume_coverage)
                  if not math.isclose(value, 1.0, rel_tol=0.0, abs_tol=2.0e-12)]
    zero_shells = []
    if rows and all(len(values) == 50 for values in rows.values()):
        zero_shells = [s for s in range(50)
                       if sum(abs(rows[z][s]) for z in EXPECTED_Z) == 0.0]
    passed = (len(shells) == 50 and len(header) == 51 and not incomplete and
              rows and not zero_shells)
    print(f"  shells={len(shells)}, abundance columns={max(0, len(header)-1)}, "
          f"incomplete={incomplete}, zero={zero_shells} "
          f"{'PASS' if passed else 'FAIL'}")
    if not passed:
        failures.append(f"gate5 incomplete={incomplete}, zero={zero_shells}")


def gate6(deck: Path, base: Path, cmf_run: Path, off_control: Path | None,
          skip: bool, failures: list[str]):
    print("GATE 6 — retain certified atomic bytes and R1/R4 gates")
    if skip:
        print("  SKIP requested (self-test only)")
        return
    mismatches = []
    for old in sorted(path for path in base.rglob("*") if path.is_file()):
        relative = old.relative_to(base)
        if relative.as_posix() in MUTABLE:
            continue
        new = deck / relative
        if not new.is_file() or old.stat().st_size != new.stat().st_size or sha256(old) != sha256(new):
            mismatches.append(str(relative))
    byte_ok = not mismatches
    print(f"  immutable base files byte mismatches={len(mismatches)} "
          f"{'PASS' if byte_ok else 'FAIL'}")
    if not byte_ok:
        failures.append(f"gate6 byte mismatches={mismatches[:20]}")
    if off_control is None:
        print("  R1/R4 verifier not run: --r4-off-control is required FAIL")
        failures.append("gate6 missing R4 OFF-control")
        return
    command = [
        sys.executable, str(ROOT / "scripts/verify_deck_r4_ftos.py"),
        "--new", str(deck),
        "--links", str(cmf_run / "atomic_links.txt"),
        "--cmf-run", str(cmf_run),
        "--links-deck", str(ROOT / "data/tardis_reference_toy06_19p48d_sivcaiv_links"),
        "--off-control", str(off_control),
    ]
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode != 0:
        failures.append(f"gate6 R1/R4 verifier exit={result.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck", type=Path, default=DECK_DEFAULT)
    parser.add_argument("--base", type=Path, default=BASE_DEFAULT)
    parser.add_argument("--cmf-run", type=Path, default=RUN_DEFAULT)
    parser.add_argument("--r4-off-control", type=Path)
    parser.add_argument("--skip-atomic-gates", action="store_true",
                        help="only for isolated synthetic self-tests")
    args = parser.parse_args()
    if not args.deck.is_dir():
        print(f"ERROR: candidate deck absent: {args.deck}", file=sys.stderr)
        return 2
    failures: list[str] = []
    try:
        hydro = parse_sn_hydro_data(args.cmf_run / "SN_HYDRO_DATA")
        shells = read_geometry(args.deck / "geometry.csv", hydro.age_days)
        expected = conservative_map(hydro, shells)
        species, species_total = parse_species_masses(args.cmf_run / "SPECIES_MASSES")
    except Exception as exc:
        print(f"ERROR: canonical/geometry read failed: {exc}", file=sys.stderr)
        return 2

    header, rows = gate1(args.deck, failures)
    gate2(header, rows, args.deck, expected, failures)
    gate3(args.deck, shells, expected, species, species_total, failures)
    gate4(rows, expected, failures)
    gate5(header, rows, shells, expected, failures)
    gate6(args.deck, args.base, args.cmf_run, args.r4_off_control,
          args.skip_atomic_gates, failures)
    if failures:
        print("VERDICT: FAIL — no adjustment was made", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    print("VERDICT: all six gates PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
