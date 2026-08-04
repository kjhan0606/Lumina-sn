#!/usr/bin/env python3
"""Read-only seven-gate verifier for the StaNdaRT-canonical toy06 deck."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path
import subprocess
import sys

from standart_toy06_composition import (
    EXPECTED_Z, Z_NAME, conservative_map, core_decay_fractions, decay_to_epoch,
    parse_standart_model, read_geometry,
)
from toy06_cmfgen_composition import (
    conservative_map as map_cmfgen,
    parse_sn_hydro_data,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_DEFAULT = ROOT / "data/standart_data1/input_models/snia_toy06_1h_lowres.dat"
BASE_DEFAULT = ROOT / "data/tardis_reference_toy06_19p48d_sivcaiv_ftos"
DECK_DEFAULT = ROOT / "data/tardis_reference_toy06_19p48d_standart"
RUN_DEFAULT = Path("/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4")
TARGET_D = 19.48
SUM_TOL = 2.0e-12
VALUE_TOL = 5.0e-13
CMF_TOL = 5.0e-5
MASS_TOL_MSUN = 2.0e-12
MUTABLE = {"abundances.csv", "density.csv", "isotopes.csv", "verification.log",
           "standart_mapping_report.json", "STANDART_DECK_INVALID"}


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


def read_density(path: Path) -> list[float]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if [int(row["shell_id"]) for row in rows] != list(range(len(rows))):
        raise ValueError("density shell ids are not contiguous")
    return [float(row["rho"]) for row in rows]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gate1(header, rows, failures):
    print("GATE 1 — exact six-element deck set")
    ok = (header == ["atomic_number", *map(str, range(50))] and
          tuple(sorted(rows)) == EXPECTED_Z and
          all(len(values) == 50 for values in rows.values()))
    print(f"  Z={sorted(rows)}, columns={max(0, len(header)-1)} "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append("gate1 element set/header/shape")


def gate2(model, failures):
    print("GATE 2 — canonical Ti/O/C exact-zero census")
    observed = {name: model.zero_counts[name] for name in ("Ti", "O", "C")}
    ok = observed == {"Ti": 202, "O": 202, "C": 202}
    print(f"  zero shells={observed} {'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append(f"gate2 zero census={observed}")


def gate3(rows, density, shells, expected, failures):
    print(f"GATE 3 — shell sums and mass conservation (sum tol={SUM_TOL:.1e})")
    sums = [sum(rows.get(z, [math.nan] * 50)[s] for z in EXPECTED_Z)
            for s in range(50)]
    max_sum = max(abs(value - 1.0) for value in sums)
    finite = all(math.isfinite(value) and value >= 0.0
                 for values in rows.values() for value in values)
    value_delta = max(abs(rows[z][s] - expected.abundances[z][s])
                      for z in EXPECTED_Z for s in range(50))
    density_delta = max(abs(density[s] - expected.density_g_cm3[s]) /
                        expected.density_g_cm3[s] for s in range(50))
    mass_delta = max(
        [abs(expected.source_overlap_species_msun[name] -
             expected.mapped_species_msun[name])
         for name in expected.mapped_species_msun] +
        [abs(expected.source_overlap_isotope_msun[name] -
             expected.mapped_isotope_msun[name])
         for name in expected.mapped_isotope_msun])
    ok = (finite and max_sum <= SUM_TOL and value_delta <= VALUE_TOL and
          density_delta <= 2.0e-15 and mass_delta <= MASS_TOL_MSUN and
          len(density) == len(shells) == 50)
    print(f"  max|sumX-1|={max_sum:.3e}; value_delta={value_delta:.3e}; "
          f"rho_rel_delta={density_delta:.3e}; mass_delta={mass_delta:.3e} Msun "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append("gate3 normalization/mapping/mass conservation")


def gate4(model, expected, shells, cmf_run, skip, failures):
    print(f"GATE 4 — primary decay chain vs secondary SN_HYDRO_DATA (tol={CMF_TOL:.1e})")
    ni, co, fe = core_decay_fractions(TARGET_D)
    core_expected = (0.1083233103068866, 0.7937131695869826, 0.09796352010613085)
    analytic_delta = max(abs(a - b) for a, b in zip((ni, co, fe), core_expected))
    if skip:
        ok = analytic_delta <= 5.0e-12
        print(f"  fixture-only analytic core delta={analytic_delta:.3e}; secondary SKIP "
              f"{'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append("gate4 analytic decay fixture")
        return
    hydro = parse_sn_hydro_data(cmf_run / "SN_HYDRO_DATA")
    cmf_mapped = map_cmfgen(hydro, shells)
    common = [s for s, coverage in enumerate(cmf_mapped.volume_coverage)
              if abs(coverage - 1.0) <= 2.0e-12]
    outside = [s for s, coverage in enumerate(cmf_mapped.volume_coverage)
               if abs(coverage - 1.0) > 2.0e-12]
    if not common:
        failures.append("gate4 no common fully-covered shells")
        print("  no common fully-covered shells FAIL")
        return
    deltas = [abs(expected.abundances[z][s] - cmf_mapped.abundances[z][s])
              for z in EXPECTED_Z for s in common]
    max_delta = max(deltas)
    ok = (abs(hydro.age_days - TARGET_D) <= 1.0e-12 and
          analytic_delta <= 5.0e-12 and max_delta <= CMF_TOL)
    print(f"  common shells={common[0]}..{common[-1]} ({len(common)}); "
          f"secondary_outside={outside}; "
          f"analytic core delta={analytic_delta:.3e}; mapped max delta={max_delta:.3e} "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append(f"gate4 decay/CMFGEN max_delta={max_delta:.6e}")


def gate5(header, rows, expected, failures):
    print("GATE 5 — all 50 shells covered and nonzero")
    incomplete = [s for s, value in enumerate(expected.volume_coverage)
                  if abs(value - 1.0) > 2.0e-12]
    zero = [s for s in range(50)
            if sum(abs(rows.get(z, [0.0] * 50)[s]) for z in EXPECTED_Z) == 0.0]
    ok = len(header) == 51 and not incomplete and not zero
    print(f"  incomplete={incomplete}, zero={zero} {'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append(f"gate5 incomplete={incomplete} zero={zero}")


def gate6(rows, failures):
    print("GATE 6 — stratification: deep Co-dominant, outer Si/S/Ca")
    deep = 0
    outer = 49
    ok = (rows[27][deep] > rows[28][deep] > rows[26][deep] and
          rows[14][outer] > rows[16][outer] > rows[20][outer] and
          abs(rows[14][outer] - 0.55) <= 5.0e-12 and
          abs(rows[16][outer] - 0.35) <= 5.0e-12 and
          abs(rows[20][outer] - 0.10) <= 5.0e-12)
    print("  deep Ni/Co/Fe=%.6f/%.6f/%.6f; outer Si/S/Ca=%.6f/%.6f/%.6f %s" %
          (rows[28][deep], rows[27][deep], rows[26][deep],
           rows[14][outer], rows[16][outer], rows[20][outer],
           "PASS" if ok else "FAIL"))
    if not ok:
        failures.append("gate6 layer structure")


def gate7(deck, base, cmf_run, off_control, skip, failures):
    print("GATE 7 — immutable atomic bytes plus retained R1/R4")
    if skip:
        print("  fixture-only atomic gates SKIP")
        return
    mismatches = []
    for old in sorted(path for path in base.rglob("*") if path.is_file()):
        relative = old.relative_to(base)
        if relative.as_posix() in MUTABLE:
            continue
        new = deck / relative
        if (not new.is_file() or old.stat().st_size != new.stat().st_size or
                sha256(old) != sha256(new)):
            mismatches.append(str(relative))
    print(f"  immutable byte mismatches={len(mismatches)} "
          f"{'PASS' if not mismatches else 'FAIL'}")
    if mismatches:
        failures.append(f"gate7 byte mismatches={mismatches[:20]}")
    if off_control is None:
        failures.append("gate7 --r4-off-control missing")
        return
    command = [
        sys.executable, str(ROOT / "scripts/verify_deck_r4_ftos.py"),
        "--new", str(deck), "--links", str(cmf_run / "atomic_links.txt"),
        "--cmf-run", str(cmf_run),
        "--links-deck", str(ROOT / "data/tardis_reference_toy06_19p48d_sivcaiv_links"),
        "--off-control", str(off_control),
    ]
    if subprocess.run(command, cwd=ROOT, check=False).returncode != 0:
        failures.append("gate7 retained R1/R4 verifier failed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck", type=Path, default=DECK_DEFAULT)
    parser.add_argument("--base", type=Path, default=BASE_DEFAULT)
    parser.add_argument("--model", type=Path, default=MODEL_DEFAULT)
    parser.add_argument("--cmf-run", type=Path, default=RUN_DEFAULT)
    parser.add_argument("--r4-off-control", type=Path)
    parser.add_argument("--skip-secondary", action="store_true",
                        help="fixture self-test only")
    parser.add_argument("--skip-atomic-gates", action="store_true",
                        help="fixture self-test only")
    args = parser.parse_args()
    if not args.deck.is_dir():
        print(f"ERROR: candidate deck absent: {args.deck}", file=sys.stderr)
        return 2
    failures: list[str] = []
    try:
        model = parse_standart_model(args.model)
        decayed = decay_to_epoch(model, TARGET_D)
        shells = read_geometry(args.deck / "geometry.csv", TARGET_D)
        expected = conservative_map(model, decayed, shells)
        header, rows = read_abundances(args.deck / "abundances.csv")
        density = read_density(args.deck / "density.csv")
    except Exception as exc:
        print(f"ERROR: verifier input failed: {exc}", file=sys.stderr)
        return 2
    gate1(header, rows, failures)
    gate2(model, failures)
    gate3(rows, density, shells, expected, failures)
    gate4(model, expected, shells, args.cmf_run, args.skip_secondary, failures)
    gate5(header, rows, expected, failures)
    gate6(rows, failures)
    gate7(args.deck, args.base, args.cmf_run, args.r4_off_control,
          args.skip_atomic_gates, failures)
    if failures:
        print("VERDICT: FAIL — no adjustment was made", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    print("VERDICT: all seven StaNdaRT composition gates PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
