#!/usr/bin/env python3
"""Conservative SN_HYDRO_DATA -> Lumina shell composition mapping.

The CMFGEN arrays are zone-centred and ordered outer-to-inner.  toy06 uses
50 km/s zones whose centres are 1025..35975 km/s, so their physical edges are
1000..36000 km/s.  Within each source zone density and mass fractions are
treated as cell averages.  A Lumina shell receives

    X[k,s] = sum_j rho[j] X[k,j] DeltaV[j,s]
             / sum_j rho[j] DeltaV[j,s]

over exact spherical-volume intersections.  The same numerator/denominator is
used for isotope subsets.  This is a mass-weighted conservative finite-volume
restriction; volume weighting would not conserve species mass when rho varies.
No value is evaluated outside the source cell edges.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import math
from pathlib import Path
import re
from typing import Iterable


DAY_S = 86400.0
KM_CM = 1.0e5
MSUN_G = 1.989e33
FOUR_PI_OVER_THREE = 4.0 * math.pi / 3.0

ELEMENT_BLOCKS = {
    "SIL mass fraction": (14, "Si"),
    "SUL mass fraction": (16, "S"),
    "CAL mass fraction": (20, "Ca"),
    "IRON mass fraction": (26, "Fe"),
    "COB mass fraction": (27, "Co"),
    "NICK mass fraction": (28, "Ni"),
}
ISOTOPE_BLOCKS = {
    "NICK 56 mass fraction": (28, 56, "Ni56"),
    "COB 56 mass fraction": (27, 56, "Co56"),
    "IRON 56 mass fraction": (26, 56, "Fe56"),
}
CORE_BLOCKS = {
    "Radius grid (10^10cm)": "radius_1e10_cm",
    "Velocity (km/s)": "velocity_km_s",
    "Density (g/cm^3)": "density_g_cm3",
}
EXPECTED_Z = (14, 16, 20, 26, 27, 28)


class CompositionError(RuntimeError):
    """An observational contract failed; callers must not repair the input."""


@dataclass(frozen=True)
class HydroData:
    n_points: int
    age_days: float
    radius_cm: tuple[float, ...]
    velocity_km_s: tuple[float, ...]
    density_g_cm3: tuple[float, ...]
    elements: dict[int, tuple[float, ...]]
    isotopes: dict[str, tuple[float, ...]]
    v_edges_km_s: tuple[float, ...]


@dataclass(frozen=True)
class GeometryShell:
    shell_id: int
    r_inner_cm: float
    r_outer_cm: float
    v_inner_cm_s: float
    v_outer_cm_s: float


@dataclass(frozen=True)
class MappingResult:
    abundances: dict[int, tuple[float, ...]]
    isotopes: dict[str, tuple[float, ...]]
    density_g_cm3: tuple[float, ...]
    covered_mass_g: tuple[float, ...]
    volume_coverage: tuple[float, ...]
    source_species_mass_msun: dict[str, float]
    mapped_species_mass_msun: dict[str, float]
    source_isotope_mass_msun: dict[str, float]
    mapped_isotope_mass_msun: dict[str, float]


def _header_number(text: str, label: str, cast):
    match = re.search(rf"^{re.escape(label)}\s*:\s*([^\s]+)", text, re.MULTILINE)
    if not match:
        raise CompositionError(f"SN_HYDRO_DATA header lacks {label!r}")
    return cast(match.group(1))


def _read_block(lines: list[str], start: int, count: int, label: str) -> tuple[float, ...]:
    values: list[float] = []
    for line in lines[start + 1:]:
        stripped = line.strip()
        if not stripped:
            if values:
                break
            continue
        try:
            row = [float(field.replace("D", "E").replace("d", "e"))
                   for field in stripped.split()]
        except ValueError:
            break
        values.extend(row)
        if len(values) >= count:
            break
    if len(values) != count:
        raise CompositionError(f"{label}: expected {count} values, found {len(values)}")
    return tuple(values)


def _centres_to_edges(centres: tuple[float, ...]) -> tuple[float, ...]:
    if len(centres) < 2 or any(b <= a for a, b in zip(centres, centres[1:])):
        raise CompositionError("source velocity centres are not strictly increasing")
    mids = [(a + b) * 0.5 for a, b in zip(centres, centres[1:])]
    first = centres[0] - 0.5 * (centres[1] - centres[0])
    last = centres[-1] + 0.5 * (centres[-1] - centres[-2])
    return tuple([first, *mids, last])


def parse_sn_hydro_data(path: Path) -> HydroData:
    text = path.read_text(encoding="latin-1")
    lines = text.splitlines()
    n_points = _header_number(text, "Number of data points", int)
    n_fractions = _header_number(text, "Number of mass fractions", int)
    n_isotopes = _header_number(text, "Number of isotopes", int)
    age_days = _header_number(text, "Time(days) since explosion", float)
    if (n_points, n_fractions, n_isotopes) != (700, 6, 3):
        raise CompositionError(
            f"truth shape is {(n_points, n_fractions, n_isotopes)}, expected (700, 6, 3)")

    labels = {**CORE_BLOCKS, **ELEMENT_BLOCKS, **ISOTOPE_BLOCKS}
    positions: dict[str, int] = {}
    for index, line in enumerate(lines):
        label = line.strip()
        if label in labels:
            if label in positions:
                raise CompositionError(f"duplicate SN_HYDRO_DATA block {label!r}")
            positions[label] = index
    missing = set(labels) - set(positions)
    if missing:
        raise CompositionError(f"missing SN_HYDRO_DATA blocks: {sorted(missing)}")
    observed_fraction_labels = {line.strip() for line in lines
                                if line.strip().endswith("mass fraction")}
    expected_fraction_labels = set(ELEMENT_BLOCKS) | set(ISOTOPE_BLOCKS)
    if observed_fraction_labels != expected_fraction_labels:
        raise CompositionError(
            "mass-fraction block set differs: "
            f"extra={sorted(observed_fraction_labels - expected_fraction_labels)}, "
            f"missing={sorted(expected_fraction_labels - observed_fraction_labels)}")

    blocks = {label: _read_block(lines, positions[label], n_points, label)
              for label in labels}
    # All integration is inner->outer.  The canonical file is outer->inner.
    velocity = blocks["Velocity (km/s)"]
    order = sorted(range(n_points), key=velocity.__getitem__)
    ordered = lambda values: tuple(values[i] for i in order)
    velocity = ordered(velocity)
    radius_cm = tuple(value * 1.0e10
                      for value in ordered(blocks["Radius grid (10^10cm)"]))
    density = ordered(blocks["Density (g/cm^3)"])
    elements = {z: ordered(blocks[label]) for label, (z, _name) in ELEMENT_BLOCKS.items()}
    isotopes = {name: ordered(blocks[label])
                for label, (_z, _a, name) in ISOTOPE_BLOCKS.items()}
    edges = _centres_to_edges(velocity)

    if not math.isclose(edges[0], 1000.0, abs_tol=1.0e-9) or \
            not math.isclose(edges[-1], 36000.0, abs_tol=1.0e-9):
        raise CompositionError(
            f"measured source edges are [{edges[0]}, {edges[-1]}] km/s, expected [1000, 36000]")
    time_s = age_days * DAY_S
    max_homology_error = max(abs(r - v * KM_CM * time_s)
                             for r, v in zip(radius_cm, velocity))
    # Radius is printed with only ~5 significant digits; 6e10 cm bounds the
    # observed rounding while remaining <0.01 source-zone width.
    if max_homology_error > 6.0e10:
        raise CompositionError(
            f"SN radius/v homology mismatch max={max_homology_error:.6e} cm")
    if any(rho <= 0.0 or not math.isfinite(rho) for rho in density):
        raise CompositionError("SN density contains non-positive/non-finite values")
    for index in range(n_points):
        total = sum(elements[z][index] for z in EXPECTED_Z)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=5.0e-9):
            raise CompositionError(f"source mass fractions sum to {total:.17g} at point {index}")
        for name, z in (("Ni56", 28), ("Co56", 27), ("Fe56", 26)):
            if isotopes[name][index] > elements[z][index] + 5.0e-12:
                raise CompositionError(f"{name} exceeds its elemental fraction at point {index}")
    return HydroData(n_points, age_days, radius_cm, velocity, density,
                     elements, isotopes, edges)


def read_geometry(path: Path, age_days: float) -> tuple[GeometryShell, ...]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {"shell_id", "r_inner", "r_outer", "v_inner", "v_outer"}
    if not rows or not required.issubset(rows[0]):
        raise CompositionError(f"{path}: geometry columns must include {sorted(required)}")
    shells = tuple(GeometryShell(
        int(row["shell_id"]), float(row["r_inner"]), float(row["r_outer"]),
        float(row["v_inner"]), float(row["v_outer"])) for row in rows)
    if [shell.shell_id for shell in shells] != list(range(len(shells))):
        raise CompositionError("geometry shell_id must be contiguous 0..N-1")
    time_s = age_days * DAY_S
    previous_outer = None
    for shell in shells:
        if shell.r_outer_cm <= shell.r_inner_cm or shell.v_outer_cm_s <= shell.v_inner_cm_s:
            raise CompositionError(f"geometry shell {shell.shell_id} has reversed edges")
        if previous_outer is not None and not math.isclose(
                shell.v_inner_cm_s, previous_outer, rel_tol=0.0, abs_tol=1.0e-4):
            raise CompositionError(f"geometry velocity gap/overlap before shell {shell.shell_id}")
        previous_outer = shell.v_outer_cm_s
        for radius, velocity, which in (
                (shell.r_inner_cm, shell.v_inner_cm_s, "inner"),
                (shell.r_outer_cm, shell.v_outer_cm_s, "outer")):
            error = abs(radius - velocity * time_s)
            if error > max(1.0, abs(radius) * 2.0e-12):
                raise CompositionError(
                    f"geometry shell {shell.shell_id} {which} r/v mismatch {error:.6e} cm")
    return shells


def _volume(time_s: float, vlo_km_s: float, vhi_km_s: float) -> float:
    lo = vlo_km_s * KM_CM * time_s
    hi = vhi_km_s * KM_CM * time_s
    return FOUR_PI_OVER_THREE * (hi ** 3 - lo ** 3)


def conservative_map(hydro: HydroData, shells: Iterable[GeometryShell]) -> MappingResult:
    shells = tuple(shells)
    time_s = hydro.age_days * DAY_S
    nshell = len(shells)
    numerators = {z: [0.0] * nshell for z in EXPECTED_Z}
    isotope_numerators = {name: [0.0] * nshell for name in hydro.isotopes}
    covered_mass = [0.0] * nshell
    coverage = [0.0] * nshell

    source_species = {name: 0.0 for _z, name in ELEMENT_BLOCKS.values()}
    source_isotopes = {name: 0.0 for name in hydro.isotopes}
    z_name = {z: name for z, name in ELEMENT_BLOCKS.values()}
    for j in range(hydro.n_points):
        cell_volume = _volume(time_s, hydro.v_edges_km_s[j], hydro.v_edges_km_s[j + 1])
        cell_mass = hydro.density_g_cm3[j] * cell_volume
        for z in EXPECTED_Z:
            source_species[z_name[z]] += cell_mass * hydro.elements[z][j] / MSUN_G
        for name in hydro.isotopes:
            source_isotopes[name] += cell_mass * hydro.isotopes[name][j] / MSUN_G

    for s, shell in enumerate(shells):
        slo = shell.v_inner_cm_s / KM_CM
        shi = shell.v_outer_cm_s / KM_CM
        full_volume = _volume(time_s, slo, shi)
        overlap_lo = max(slo, hydro.v_edges_km_s[0])
        overlap_hi = min(shi, hydro.v_edges_km_s[-1])
        if overlap_hi > overlap_lo:
            coverage[s] = _volume(time_s, overlap_lo, overlap_hi) / full_volume
        for j in range(hydro.n_points):
            lo = max(slo, hydro.v_edges_km_s[j])
            hi = min(shi, hydro.v_edges_km_s[j + 1])
            if hi <= lo:
                continue
            mass = hydro.density_g_cm3[j] * _volume(time_s, lo, hi)
            covered_mass[s] += mass
            for z in EXPECTED_Z:
                numerators[z][s] += mass * hydro.elements[z][j]
            for name in hydro.isotopes:
                isotope_numerators[name][s] += mass * hydro.isotopes[name][j]

    abundances = {z: tuple((numerators[z][s] / covered_mass[s]
                            if covered_mass[s] > 0.0 else math.nan)
                           for s in range(nshell)) for z in EXPECTED_Z}
    isotopes = {name: tuple((isotope_numerators[name][s] / covered_mass[s]
                             if covered_mass[s] > 0.0 else math.nan)
                            for s in range(nshell)) for name in hydro.isotopes}
    density = tuple((covered_mass[s] / _volume(
        time_s, shells[s].v_inner_cm_s / KM_CM, shells[s].v_outer_cm_s / KM_CM)
        if covered_mass[s] > 0.0 else math.nan) for s in range(nshell))
    mapped_species = {z_name[z]: sum(numerators[z]) / MSUN_G for z in EXPECTED_Z}
    mapped_isotopes = {name: sum(values) / MSUN_G
                       for name, values in isotope_numerators.items()}
    return MappingResult(abundances, isotopes, density, tuple(covered_mass), tuple(coverage),
                         source_species, mapped_species, source_isotopes, mapped_isotopes)


def parse_species_masses(path: Path) -> tuple[dict[str, float], float]:
    aliases = {"SIL": "Si", "SUL": "S", "CAL": "Ca", "IRON": "Fe",
               "COB": "Co", "NICK": "Ni"}
    result: dict[str, float] = {}
    total = None
    for line in path.read_text(encoding="latin-1").splitlines():
        fields = line.split()
        if fields and fields[0] in aliases and len(fields) >= 2:
            result.setdefault(aliases[fields[0]], float(fields[1]))
        match = re.search(r"Total ejecta mass of model is\s+([0-9.Ee+\-]+)", line)
        if match:
            total = float(match.group(1))
    if set(result) != set(aliases.values()) or total is None:
        raise CompositionError(f"could not parse six species and total from {path}")
    return result, total


def mapping_report(hydro_path: Path, geometry_path: Path, hydro: HydroData,
                   shells: tuple[GeometryShell, ...], mapped: MappingResult,
                   species_path: Path, species: dict[str, float], total: float) -> dict:
    incomplete = [shell.shell_id for shell, fraction in zip(shells, mapped.volume_coverage)
                  if not math.isclose(fraction, 1.0, rel_tol=0.0, abs_tol=2.0e-12)]
    return {
        "schema": "lumina.toy06_cmfgencomp.mapping.v1",
        "canonical_source": str(hydro_path),
        "species_masses_source": str(species_path),
        "geometry_source": str(geometry_path),
        "age_days": hydro.age_days,
        "source_points": hydro.n_points,
        "source_velocity_edges_km_s": [hydro.v_edges_km_s[0], hydro.v_edges_km_s[-1]],
        "lumina_shells": len(shells),
        "lumina_velocity_edges_km_s": [shells[0].v_inner_cm_s / KM_CM,
                                        shells[-1].v_outer_cm_s / KM_CM],
        "mapping": "rho*dV mass-weighted exact spherical-volume intersections",
        "outside_source_policy": "NONE: no extrapolation, clamp, or replacement",
        "volume_coverage_by_shell": list(mapped.volume_coverage),
        "incomplete_shell_ids": incomplete,
        "source_species_mass_msun_integrated": mapped.source_species_mass_msun,
        "mapped_overlap_species_mass_msun": mapped.mapped_species_mass_msun,
        "source_isotope_mass_msun_integrated": mapped.source_isotope_mass_msun,
        "mapped_overlap_isotope_mass_msun": mapped.mapped_isotope_mass_msun,
        "species_masses_table_msun": species,
        "species_masses_total_msun": total,
        "can_emit_deck_composition": not incomplete and len(shells) == 50,
        "unresolved": ([] if not incomplete else [
            "Lumina geometry is not fully covered by SN_HYDRO_DATA; composition files were not emitted.",
            "The missing inner CMFGEN range is not represented by any Lumina shell, so the 0.99393 Msun full-ejecta gate cannot pass.",
        ]),
    }


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
