#!/usr/bin/env python3
"""Build the new StaNdaRT-canonical toy06 deck; never overwrite a path.

This CPU-only program is intentionally not invoked by repository preparation.
The driving seat runs it, then runs the independent seven-gate verifier.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

from standart_toy06_composition import (
    EXPECTED_Z, Z_NAME, conservative_map, decay_to_epoch, parse_standart_model,
    read_geometry,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_DEFAULT = ROOT / "data/standart_data1/input_models/snia_toy06_1h_lowres.dat"
BASE_DEFAULT = ROOT / "data/tardis_reference_toy06_19p48d_sivcaiv_ftos"
OUT_DEFAULT = ROOT / "data/tardis_reference_toy06_19p48d_standart"
TARGET_D = 19.48
OMIT = {"abundances.csv", "density.csv", "isotopes.csv", "verification.log",
        "standart_mapping_report.json", "STANDART_DECK_INVALID"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_abundances(path: Path, values: dict[int, tuple[float, ...]]) -> None:
    nshell = len(next(iter(values.values())))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["atomic_number", *range(nshell)])
        for z in EXPECTED_Z:
            writer.writerow([z, *(format(value, ".17g") for value in values[z])])


def write_isotopes(path: Path, values: dict[str, tuple[float, ...]]) -> None:
    identities = {"Ni56": (28, 56), "Co56": (27, 56), "Fe56": (26, 56)}
    nshell = len(next(iter(values.values())))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["isotope", "atomic_number", "mass_number", *range(nshell)])
        for name in ("Ni56", "Co56", "Fe56"):
            writer.writerow([name, *identities[name],
                             *(format(value, ".17g") for value in values[name])])


def write_density(path: Path, values: tuple[float, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["shell_id", "rho"])
        for shell, value in enumerate(values):
            writer.writerow([shell, format(value, ".17g")])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=MODEL_DEFAULT)
    parser.add_argument("--base", type=Path, default=BASE_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--target-days", type=float, default=TARGET_D)
    args = parser.parse_args()
    for required in (args.model, args.base, args.base / "geometry.csv"):
        if not required.exists():
            raise SystemExit(f"required input absent: {required}")
    if args.output.exists() or args.output.is_symlink():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")

    model = parse_standart_model(args.model)
    decayed = decay_to_epoch(model, args.target_days)
    shells = read_geometry(args.base / "geometry.csv", args.target_days)
    mapped = conservative_map(model, decayed, shells)
    incomplete = [s for s, value in enumerate(mapped.volume_coverage)
                  if abs(value - 1.0) > 2.0e-12]
    zero_shells = [s for s, mass in enumerate(mapped.shell_mass_g) if mass <= 0.0]
    if len(shells) != 50 or incomplete or zero_shells:
        raise SystemExit("mapping coverage FAIL: "
                         f"shells={len(shells)} incomplete={incomplete} zero={zero_shells}")

    # Dereference the certified base's absolute symlinks so the new deck follows
    # REPO_ROOT on a compute node instead of retaining a login-host absolute path.
    shutil.copytree(args.base, args.output, symlinks=False,
                    ignore=lambda _directory, names: [name for name in names if name in OMIT])
    write_abundances(args.output / "abundances.csv", mapped.abundances)
    write_isotopes(args.output / "isotopes.csv", mapped.isotopes)
    write_density(args.output / "density.csv", mapped.density_g_cm3)

    norm_delta = [factor - 1.0 for factor in decayed.normalization_factors]
    report = {
        "schema": "lumina.toy06.standart.mapping.v1",
        "canonical_source": str(args.model),
        "canonical_sha256": sha256(args.model),
        "base_deck": str(args.base),
        "output_deck": str(args.output),
        "source_shape": [202, 21],
        "source_age_days": model.source_age_days,
        "target_age_days": args.target_days,
        "source_velocity_centres_km_s": [model.velocity_km_s[0], model.velocity_km_s[-1]],
        "source_velocity_edges_km_s": [model.velocity_edges_km_s[0],
                                         model.velocity_edges_km_s[-1]],
        "lumina_velocity_edges_km_s": [shells[0].v_inner_cm_s / 1e5,
                                         shells[-1].v_outer_cm_s / 1e5],
        "mapping": "source dmass times exact spherical-v^3 overlap fraction; mass-weighted finite-volume restriction",
        "outside_source_policy": "none; incomplete coverage is fatal",
        "renormalization": "explicit six-species per-source-cell normalization before restriction; Lumina runtime does not normalize",
        "normalization_factor_minmax": [min(decayed.normalization_factors),
                                          max(decayed.normalization_factors)],
        "normalization_max_abs_delta": max(abs(value) for value in norm_delta),
        "zero_census": model.zero_counts,
        "positive_census": model.positive_counts,
        "volume_coverage_by_shell": list(mapped.volume_coverage),
        "source_overlap_species_msun": mapped.source_overlap_species_msun,
        "mapped_species_msun": mapped.mapped_species_msun,
        "source_overlap_isotope_msun": mapped.source_overlap_isotope_msun,
        "mapped_isotope_msun": mapped.mapped_isotope_msun,
        "element_rows": [{"Z": z, "name": Z_NAME[z]} for z in EXPECTED_Z],
        "atomic_topology_note": "atom_masses and atomic tables retain 15 species; the other nine calloc abundance rows remain exact zero",
        "required_patched_runtime": {
            "LUMINA_REBUILD_INITIAL_PLASMA": "1 (invalidate copied iteration-0 opacity)",
            "LUMINA_GAMMA_DEP": "1",
            "LUMINA_DEPOSITION_FILE": "<deck>/deposition_cmfgen.csv (avoid a second Ni/Co decay)",
        },
    }
    (args.output / "standart_mapping_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"StaNdaRT candidate emitted: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
