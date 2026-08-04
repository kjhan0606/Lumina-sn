#!/usr/bin/env python3
"""Prepare a new toy06 CMFGEN-composition deck; never overwrite a deck.

This is a CPU-only data preparation program.  On incomplete source coverage it
still creates an auditable candidate directory and mapping report, but omits
abundances.csv/isotopes.csv/density.csv.  The final validator must then fail.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

from toy06_cmfgen_composition import (
    ELEMENT_BLOCKS,
    ISOTOPE_BLOCKS,
    conservative_map,
    mapping_report,
    parse_sn_hydro_data,
    parse_species_masses,
    read_geometry,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_DEFAULT = ROOT / "data/tardis_reference_toy06_19p48d_sivcaiv_ftos"
OUT_DEFAULT = ROOT / "data/tardis_reference_toy06_19p48d_cmfgencomp"
RUN_DEFAULT = Path("/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4")
OMIT_FROM_COPY = {"abundances.csv", "density.csv", "isotopes.csv", "verification.log",
                  "composition_mapping_report.json", "composition_preregistration.json",
                  "COMPOSITION_INVALID"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_abundances(path: Path, abundances: dict[int, tuple[float, ...]]) -> None:
    shell_count = len(next(iter(abundances.values())))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["atomic_number", *range(shell_count)])
        for z in (14, 16, 20, 26, 27, 28):
            writer.writerow([z, *(format(value, ".17g") for value in abundances[z])])


def write_isotopes(path: Path, isotopes: dict[str, tuple[float, ...]]) -> None:
    shell_count = len(next(iter(isotopes.values())))
    identities = {name: (z, mass) for _label, (z, mass, name) in ISOTOPE_BLOCKS.items()}
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["isotope", "atomic_number", "mass_number", *range(shell_count)])
        for name in ("Ni56", "Co56", "Fe56"):
            z, mass = identities[name]
            writer.writerow([name, z, mass,
                             *(format(value, ".17g") for value in isotopes[name])])


def write_density(path: Path, density: tuple[float, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["shell_id", "rho"])
        for shell_id, rho in enumerate(density):
            writer.writerow([shell_id, format(rho, ".17g")])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=BASE_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--cmf-run", type=Path, default=RUN_DEFAULT)
    parser.add_argument("--geometry", type=Path,
                        help="read-only geometry override; default is BASE/geometry.csv")
    args = parser.parse_args()
    hydro_path = args.cmf_run / "SN_HYDRO_DATA"
    species_path = args.cmf_run / "SPECIES_MASSES"
    geometry_path = args.geometry or (args.base / "geometry.csv")
    for required in (args.base, hydro_path, species_path, geometry_path):
        if not required.exists():
            raise SystemExit(f"required input absent: {required}")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")

    hydro = parse_sn_hydro_data(hydro_path)
    shells = read_geometry(geometry_path, hydro.age_days)
    mapped = conservative_map(hydro, shells)
    species, species_total = parse_species_masses(species_path)
    report = mapping_report(hydro_path, geometry_path, hydro, shells, mapped,
                            species_path, species, species_total)
    report["canonical_sha256"] = sha256(hydro_path)
    report["species_masses_sha256"] = sha256(species_path)
    report["base_deck"] = str(args.base)
    report["output_deck"] = str(args.output)

    shutil.copytree(args.base, args.output,
                    ignore=lambda _directory, names: [name for name in names
                                                       if name in OMIT_FROM_COPY])
    if geometry_path.resolve() != (args.base / "geometry.csv").resolve():
        shutil.copy2(geometry_path, args.output / "geometry.csv")
    write_json(args.output / "composition_mapping_report.json", report)
    prereg = {
        "schema": "lumina.toy06_cmfgencomp.preregistration.v1",
        "before": "15 positive uniform elemental rows in the certified ftos provenance deck; only 30 shell columns",
        "expected_after": [
            "deep shells Co-dominated near X(Co)=0.7937 instead of Fe=0.5",
            "outer shells Si/S/Ca near 0.55/0.35/0.10",
            "C,O,Mg,Al,Sc,Ti,V,Cr,Mn absent from abundances.csv and therefore zero",
            "all spectra, ionization, and temperatures may change materially",
        ],
        "postrun_metrics": "PENDING: model/GPU run forbidden in this task; driving seat must record before/after after a validated deck exists",
    }
    write_json(args.output / "composition_preregistration.json", prereg)

    if report["can_emit_deck_composition"]:
        write_abundances(args.output / "abundances.csv", mapped.abundances)
        write_isotopes(args.output / "isotopes.csv", mapped.isotopes)
        write_density(args.output / "density.csv", mapped.density_g_cm3)
        print(f"composition candidate emitted: {args.output}")
    else:
        message = (
            "INVALID / DO NOT RUN\n"
            "Canonical velocity coverage does not cover all 50 Lumina shells.\n"
            "No abundance, isotope, or density file was emitted; no extrapolation was used.\n"
            "Run the final verifier for the numerical failure report.\n"
        )
        (args.output / "COMPOSITION_INVALID").write_text(message, encoding="utf-8")
        print(f"incomplete-coverage candidate prepared (composition omitted): {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
