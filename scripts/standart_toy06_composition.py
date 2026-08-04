#!/usr/bin/env python3
"""StaNdaRT toy06 truth, decay, and conservative Lumina-shell restriction.

The input rows are zone-centred cell averages.  Cell edges are measured from
the uniform velocity-centre grid.  The printed ``dmass`` is authoritative for
mass: a partial-cell contribution is dmass times its exact spherical v^3
volume fraction.  No value is evaluated outside a source cell.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import math
from pathlib import Path
import re
from typing import Iterable


DAY_S = 86400.0
KM_CM = 1.0e5
MSUN_G = 1.989e33
FOUR_PI_OVER_THREE = 4.0 * math.pi / 3.0
HALF_NI56_D = 6.075
HALF_CO56_D = 77.236
EXPECTED_Z = (14, 16, 20, 26, 27, 28)
Z_NAME = {14: "Si", 16: "S", 20: "Ca", 26: "Fe", 27: "Co", 28: "Ni"}


class CompositionError(RuntimeError):
    pass


@dataclass(frozen=True)
class GeometryShell:
    shell_id: int
    r_inner_cm: float
    r_outer_cm: float
    v_inner_cm_s: float
    v_outer_cm_s: float


@dataclass(frozen=True)
class StandartModel:
    source_age_days: float
    velocity_km_s: tuple[float, ...]
    velocity_edges_km_s: tuple[float, ...]
    dmass_msun: tuple[float, ...]
    density_g_cm3: tuple[float, ...]
    x_ige0: tuple[float, ...]
    x_ni560: tuple[float, ...]
    source_elements: dict[int, tuple[float, ...]]
    source_ni56: tuple[float, ...]
    zero_counts: dict[str, int]
    positive_counts: dict[str, int]


@dataclass(frozen=True)
class DecayedModel:
    target_age_days: float
    elements: dict[int, tuple[float, ...]]
    isotopes: dict[str, tuple[float, ...]]
    normalization_factors: tuple[float, ...]


@dataclass(frozen=True)
class MappingResult:
    abundances: dict[int, tuple[float, ...]]
    isotopes: dict[str, tuple[float, ...]]
    density_g_cm3: tuple[float, ...]
    shell_mass_g: tuple[float, ...]
    volume_coverage: tuple[float, ...]
    source_overlap_species_msun: dict[str, float]
    mapped_species_msun: dict[str, float]
    source_overlap_isotope_msun: dict[str, float]
    mapped_isotope_msun: dict[str, float]


def _centres_to_edges(centres: tuple[float, ...]) -> tuple[float, ...]:
    if len(centres) < 2 or any(b <= a for a, b in zip(centres, centres[1:])):
        raise CompositionError("velocity centres are not strictly increasing")
    mids = [(a + b) * 0.5 for a, b in zip(centres, centres[1:])]
    return tuple([centres[0] - (centres[1] - centres[0]) * 0.5,
                  *mids,
                  centres[-1] + (centres[-1] - centres[-2]) * 0.5])


def parse_standart_model(path: Path) -> StandartModel:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"tend\s*=\s*([0-9.eE+\-]+)\s*DAYS", text)
    if not match:
        raise CompositionError("missing tend age in StaNdaRT header")
    age = float(match.group(1))
    rows: list[list[float]] = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) == 21 and fields[0].isdigit():
            rows.append([float(value) for value in fields])
    if len(rows) != 202 or any(len(row) != 21 for row in rows):
        raise CompositionError(f"truth shape is {len(rows)}x21, expected 202x21")
    if [int(row[0]) for row in rows] != list(range(1, 203)):
        raise CompositionError("zone indices are not contiguous 1..202")

    velocity = tuple(row[1] for row in rows)
    edges = _centres_to_edges(velocity)
    if edges != tuple(float(value) for value in range(0, 40401, 200)):
        raise CompositionError(
            f"measured velocity edges are [{edges[0]}, {edges[-1]}], not [0,40400]")
    time_s = age * DAY_S
    max_homology = max(abs(row[9] - row[1] * KM_CM * time_s) for row in rows)
    # The header prints 0.041667 d while radii use exactly 3600 s.  Their
    # 0.0288-s rounding difference reaches 1.16064e8 cm at 40300 km/s.
    if max_homology > 1.17e8:
        raise CompositionError(f"radius/velocity homology error={max_homology:.6e} cm")

    names_cols = {"Ti": 7, "Ni": 13, "Co": 14, "Fe": 15,
                  "Ca": 16, "S": 17, "Si": 18, "O": 19, "C": 20}
    zeros = {name: sum(row[col] == 0.0 for row in rows)
             for name, col in names_cols.items()}
    positive = {name: sum(row[col] > 0.0 for row in rows)
                for name, col in names_cols.items()}
    if any(zeros[name] != 202 for name in ("Ti", "O", "C")):
        raise CompositionError(f"canonical Ti/O/C are not exact zero: {zeros}")
    expected_positive = {"Ni": 62, "Co": 62, "Fe": 62,
                         "Ca": 169, "S": 169, "Si": 169}
    if any(positive[name] != count for name, count in expected_positive.items()):
        raise CompositionError(f"canonical positive-shell census differs: {positive}")
    if any(row[4] != 0.0 for row in rows):
        raise CompositionError("toy06 stable-IGE column is nonzero; stable split is unspecified")

    source_elements = {
        28: tuple(row[13] for row in rows),
        27: tuple(row[14] for row in rows),
        26: tuple(row[15] for row in rows),
        20: tuple(row[16] for row in rows),
        16: tuple(row[17] for row in rows),
        14: tuple(row[18] for row in rows),
    }
    return StandartModel(
        age, velocity, edges, tuple(row[2] for row in rows),
        tuple(row[10] for row in rows), tuple(row[4] for row in rows),
        tuple(row[5] for row in rows), source_elements,
        tuple(row[12] for row in rows), zeros, positive)


def decay_to_epoch(model: StandartModel, target_age_days: float) -> DecayedModel:
    if target_age_days < model.source_age_days:
        raise CompositionError("target age predates the canonical composition snapshot")
    lni = math.log(2.0) / HALF_NI56_D
    lco = math.log(2.0) / HALF_CO56_D
    fni = math.exp(-lni * target_age_days)
    fco = lni / (lni - lco) * (
        math.exp(-lco * target_age_days) - math.exp(-lni * target_age_days))
    ffe = 1.0 - fni - fco

    raw = {z: [] for z in EXPECTED_Z}
    raw_iso = {"Ni56": [], "Co56": [], "Fe56": []}
    factors: list[float] = []
    for j, x0 in enumerate(model.x_ni560):
        values = {
            28: x0 * fni,
            27: x0 * fco,
            26: x0 * ffe + model.x_ige0[j],
            20: model.source_elements[20][j],
            16: model.source_elements[16][j],
            14: model.source_elements[14][j],
        }
        total = sum(values.values())
        if not math.isfinite(total) or total <= 0.0:
            raise CompositionError(f"non-positive/non-finite raw sum in zone {j + 1}")
        # The source is printed to 5-6 significant digits.  This explicit,
        # reported six-species renormalization mirrors CMFGEN; Lumina has none.
        factor = 1.0 / total
        factors.append(factor)
        for z in EXPECTED_Z:
            raw[z].append(values[z] * factor)
        raw_iso["Ni56"].append(values[28] * factor)
        raw_iso["Co56"].append(values[27] * factor)
        raw_iso["Fe56"].append(x0 * ffe * factor)
    return DecayedModel(target_age_days,
                        {z: tuple(values) for z, values in raw.items()},
                        {name: tuple(values) for name, values in raw_iso.items()},
                        tuple(factors))


def read_geometry(path: Path, target_age_days: float) -> tuple[GeometryShell, ...]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {"shell_id", "r_inner", "r_outer", "v_inner", "v_outer"}
    if not rows or not required.issubset(rows[0]):
        raise CompositionError(f"geometry lacks {sorted(required)}")
    shells = tuple(GeometryShell(int(row["shell_id"]), float(row["r_inner"]),
                                 float(row["r_outer"]), float(row["v_inner"]),
                                 float(row["v_outer"])) for row in rows)
    if [shell.shell_id for shell in shells] != list(range(len(shells))):
        raise CompositionError("geometry shell ids are not contiguous")
    t = target_age_days * DAY_S
    prior = None
    for shell in shells:
        if shell.v_outer_cm_s <= shell.v_inner_cm_s:
            raise CompositionError(f"reversed geometry shell {shell.shell_id}")
        if prior is not None and not math.isclose(shell.v_inner_cm_s, prior,
                                                  rel_tol=0.0, abs_tol=1.0e-4):
            raise CompositionError(f"geometry gap/overlap before shell {shell.shell_id}")
        prior = shell.v_outer_cm_s
        for radius, velocity in ((shell.r_inner_cm, shell.v_inner_cm_s),
                                 (shell.r_outer_cm, shell.v_outer_cm_s)):
            if abs(radius - velocity * t) > max(1.0, abs(radius) * 2.0e-12):
                raise CompositionError(f"geometry is not homologous at shell {shell.shell_id}")
    return shells


def _v3(vlo: float, vhi: float) -> float:
    return vhi ** 3 - vlo ** 3


def conservative_map(model: StandartModel, decayed: DecayedModel,
                     shells: Iterable[GeometryShell]) -> MappingResult:
    shells = tuple(shells)
    nshell = len(shells)
    elem_num = {z: [0.0] * nshell for z in EXPECTED_Z}
    iso_num = {name: [0.0] * nshell for name in decayed.isotopes}
    shell_mass = [0.0] * nshell
    coverage = [0.0] * nshell
    source_elem = {Z_NAME[z]: 0.0 for z in EXPECTED_Z}
    source_iso = {name: 0.0 for name in decayed.isotopes}

    for s, shell in enumerate(shells):
        slo = shell.v_inner_cm_s / KM_CM
        shi = shell.v_outer_cm_s / KM_CM
        overlap_lo = max(slo, model.velocity_edges_km_s[0])
        overlap_hi = min(shi, model.velocity_edges_km_s[-1])
        if overlap_hi > overlap_lo:
            coverage[s] = _v3(overlap_lo, overlap_hi) / _v3(slo, shi)
        for j in range(len(model.velocity_km_s)):
            cell_lo = model.velocity_edges_km_s[j]
            cell_hi = model.velocity_edges_km_s[j + 1]
            lo, hi = max(slo, cell_lo), min(shi, cell_hi)
            if hi <= lo:
                continue
            frac = _v3(lo, hi) / _v3(cell_lo, cell_hi)
            mass = model.dmass_msun[j] * MSUN_G * frac
            shell_mass[s] += mass
            for z in EXPECTED_Z:
                value = mass * decayed.elements[z][j]
                elem_num[z][s] += value
                source_elem[Z_NAME[z]] += value / MSUN_G
            for name in decayed.isotopes:
                value = mass * decayed.isotopes[name][j]
                iso_num[name][s] += value
                source_iso[name] += value / MSUN_G

    abundance = {z: tuple(elem_num[z][s] / shell_mass[s]
                          if shell_mass[s] > 0.0 else math.nan
                          for s in range(nshell)) for z in EXPECTED_Z}
    isotopes = {name: tuple(iso_num[name][s] / shell_mass[s]
                            if shell_mass[s] > 0.0 else math.nan
                            for s in range(nshell)) for name in decayed.isotopes}
    t = decayed.target_age_days * DAY_S
    density = tuple(shell_mass[s] /
                    (FOUR_PI_OVER_THREE *
                     ((shells[s].v_outer_cm_s * t) ** 3 -
                      (shells[s].v_inner_cm_s * t) ** 3))
                    if shell_mass[s] > 0.0 else math.nan
                    for s in range(nshell))
    mapped_elem = {Z_NAME[z]: sum(elem_num[z]) / MSUN_G for z in EXPECTED_Z}
    mapped_iso = {name: sum(values) / MSUN_G for name, values in iso_num.items()}
    return MappingResult(abundance, isotopes, density, tuple(shell_mass),
                         tuple(coverage), source_elem, mapped_elem,
                         source_iso, mapped_iso)


def core_decay_fractions(target_age_days: float) -> tuple[float, float, float]:
    """Exact unit-56Ni chain, used for the independent CMFGEN core check."""
    lni = math.log(2.0) / HALF_NI56_D
    lco = math.log(2.0) / HALF_CO56_D
    ni = math.exp(-lni * target_age_days)
    co = lni / (lni - lco) * (math.exp(-lco * target_age_days) - ni)
    return ni, co, 1.0 - ni - co
