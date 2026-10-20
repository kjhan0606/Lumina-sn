#!/usr/bin/env python3
"""Offline P-TF gate/physics battery for a CMFGEN relT3-family rundir.

This program is deliberately read-only with respect to every rundir.  For each
preserved STEQ solution it reports

  * the unfiltered correction distribution over every population equation,
  * gated MAXCH = 100 * max(abs(correction)) after excluding population
    variables with POPS < threshold * species total,
  * the un-gated SOLVEBA_V13 returned-MAXCH semantics, and
  * the gated correction owner (ion, superlevel, depth).

It also decodes SCRTEMP and reports physical state metrics for every requested
iteration, including iterations for which STEQ_VALS was not preserved.

Typical retrospective invocation:

  python3 scripts/ptf_gated_metrics.py \
    /gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT3 \
    /gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT3_probe1 \
    --from-it 41 --to-it 55 --format markdown

The decoder follows CMFGEN's direct-access SCRTEMP layout documented by
dir_acc_pars_gen.f and scr_read_v2.f.  STEQ sign follows solveba_v13.f:
positive means a proposed decrease and negative a proposed increase.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REC_BYTES = 16_376
DOUBLES_PER_REC = 2_047
SCRTEMP_HEADER_RECS = 2
DEFAULT_THRESHOLD = 1.0e-20
SOURCE_NV = 10

SOLUTION_LINE_RE = re.compile(r"\s*(\d+)\(\s*(\d+)\)#\s+(.*)")
OUTGEN_INC_RE = re.compile(
    r"Maximum % increase at depth\s+\d+\s+is\s*([+\-0-9.EeDd]+).*iteration\s+(\d+)"
)
OUTGEN_DEC_RE = re.compile(
    r"Maximum % decrease at depth\s+\d+\s+is\s*([+\-0-9.EeDd]+).*iteration\s+(\d+)"
)
OUTGEN_RETURN_RE = re.compile(
    r"Maximm changes as returned by SOLVEBA_V13 is\s*([+\-0-9.EeDd]+)"
)
LINK_HEADER_RE = re.compile(r"5 largest (reductions|increases) at depth:\s*(\d+)")
LINK_ROW_RE = re.compile(
    r"\s*([+\-]?\d+(?:\.\d*)?[EeDd][+\-]?\d+)\s+(\S+)\s+(\d+)\s+(\d+)\s*$"
)


class BatteryError(RuntimeError):
    """Input is incomplete or violates the expected CMFGEN layout."""


@dataclass(frozen=True)
class Variable:
    index: int                 # one-based I(STEQ)
    ion: str
    species_internal: str
    species: str
    superlevel: int
    charge: int
    level_name: str
    terminal: bool = False


@dataclass
class OutgenRecord:
    increase_percent: float | None = None
    decrease_percent: float | None = None
    returned_maxch_percent: float | None = None


@dataclass(frozen=True)
class LinkRecord:
    kind: str
    depth: int
    correction: float
    ion: str
    superlevel: int
    variable: int


def ffloat(text: str) -> float:
    return float(text.replace("D", "E").replace("d", "e"))


def species_prefix(ion: str) -> str:
    match = re.fullmatch(r"(.+?)(?:2|III|IV|V|SIX|SEV)", ion)
    if not match:
        raise BatteryError(f"cannot infer species from ion label {ion!r}")
    return match.group(1)


def physical_species(internal: str) -> str:
    # Historical CMFGEN aliases in this model: Sk=Si and Nk=Ni.
    return {"Sk": "Si", "Nk": "Ni"}.get(internal, internal)


def ion_charge(ion: str) -> int:
    for suffix, charge in (
        ("SEV", 6), ("SIX", 5), ("III", 2), ("IV", 3), ("V", 4), ("2", 1)
    ):
        if ion.endswith(suffix):
            return charge
    raise BatteryError(f"cannot infer charge from ion label {ion!r}")


def next_ion(ion: str) -> str:
    internal = species_prefix(ion)
    suffix = ion[len(internal):]
    nxt = {"2": "III", "III": "IV", "IV": "V", "V": "SIX", "SIX": "SEV"}
    if suffix not in nxt:
        raise BatteryError(f"cannot infer terminal stage after {ion!r}")
    return internal + nxt[suffix]


def read_dimensions(rundir: Path) -> tuple[int, int]:
    model = rundir / "MODEL"
    model_spec = rundir / "MODEL_SPEC"
    nd = nt = None
    if model.exists():
        with model.open(errors="replace") as handle:
            for line in handle:
                if "Number of depth points" in line:
                    match = re.match(r"\s*(\d+)", line)
                    if match:
                        nd = int(match.group(1))
                elif "Total number of variables" in line:
                    match = re.match(r"\s*(\d+)", line)
                    if match:
                        nt = int(match.group(1))
                if nd is not None and nt is not None:
                    break
    if nd is None and model_spec.exists():
        with model_spec.open(errors="replace") as handle:
            for line in handle:
                match = re.match(r"\s*(\d+)\s+\[ND\]", line)
                if match:
                    nd = int(match.group(1))
                    break
    if nd is None or nt is None:
        raise BatteryError(f"could not read ND/NT from {model}")
    if nt < 3:
        raise BatteryError(f"invalid NT={nt} in {model}")
    return nd, nt


def read_variables(rundir: Path, population_variables: int) -> list[Variable]:
    path = rundir / "LEVEL_SL_STEQ_LINKS"
    if not path.exists():
        raise BatteryError(f"missing {path}")

    raw: dict[int, tuple[str, int, str]] = {}
    with path.open(errors="replace") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 6 or not all(x.isdigit() for x in fields[1:5]):
                continue
            ion = fields[0]
            sl = int(fields[3])
            eq = int(fields[4])
            if not 1 <= eq <= population_variables:
                continue
            raw.setdefault(eq, (ion, sl, " ".join(fields[5:])))

    if not raw:
        raise BatteryError(f"no STEQ links parsed from {path}")

    variables: list[Variable] = []
    for eq in range(1, population_variables + 1):
        terminal = False
        if eq in raw:
            ion, sl, level_name = raw[eq]
        else:
            prev = raw.get(eq - 1)
            following = raw.get(eq + 1)
            if prev is None:
                raise BatteryError(f"unexplained I(STEQ) gap at variable {eq}")
            prev_ion = prev[0]
            if following is not None and species_prefix(following[0]) == species_prefix(prev_ion):
                raise BatteryError(f"internal I(STEQ) gap at variable {eq}")
            ion = next_ion(prev_ion)
            sl = 1
            level_name = "terminal closure"
            terminal = True
        internal = species_prefix(ion)
        variables.append(
            Variable(
                index=eq,
                ion=ion,
                species_internal=internal,
                species=physical_species(internal),
                superlevel=sl,
                charge=ion_charge(ion),
                level_name=level_name,
                terminal=terminal,
            )
        )
    return variables


class ScrtempReader:
    def __init__(
        self, path: Path, nd: int, nt: int, state_first_it: int = 1
    ) -> None:
        self.path = path
        self.nd = nd
        self.nt = nt
        self.state_first_it = state_first_it
        self.records_per_state = math.ceil(nd * nt / DOUBLES_PER_REC)
        size = path.stat().st_size
        if size % REC_BYTES:
            raise BatteryError(f"{path}: size {size} is not a multiple of {REC_BYTES}")
        records = size // REC_BYTES
        payload_records = records - SCRTEMP_HEADER_RECS
        if payload_records < 0 or payload_records % self.records_per_state:
            raise BatteryError(
                f"{path}: {records} records do not fit 2 + N*{self.records_per_state}"
            )
        self.nstates = payload_records // self.records_per_state
        self.last_it = state_first_it + self.nstates - 1
        self._raw = np.memmap(path, dtype="<f8", mode="r")
        self.radius = np.asarray(self._raw[:nd], dtype=np.float64).copy()

    def has_iteration(self, iteration: int) -> bool:
        return self.state_first_it <= iteration <= self.last_it

    def state(self, iteration: int) -> np.ndarray:
        if not self.has_iteration(iteration):
            raise BatteryError(
                f"{self.path}: iteration {iteration} outside "
                f"[{self.state_first_it}, {self.last_it}]"
            )
        local = iteration - self.state_first_it
        first_record = SCRTEMP_HEADER_RECS + local * self.records_per_state
        flat = np.empty(self.nt * self.nd, dtype=np.float64)
        filled = 0
        for record in range(self.records_per_state):
            take = min(DOUBLES_PER_REC, flat.size - filled)
            start = (first_record + record) * DOUBLES_PER_REC
            flat[filled:filled + take] = self._raw[start:start + take]
            filled += take
        return flat.reshape(self.nd, self.nt).T


def parse_solutions(path: Path, nt: int, nd: int) -> list[np.ndarray]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    solutions: list[np.ndarray] = []
    current: np.ndarray | None = None
    filled: np.ndarray | None = None
    in_solution = False

    def finish() -> None:
        nonlocal current, filled
        if current is None or filled is None:
            return
        if not bool(filled.all()):
            missing = int(filled.size - filled.sum())
            raise BatteryError(f"{path}: incomplete STEQ solution block ({missing} cells missing)")
        solutions.append(current)
        current = None
        filled = None

    with path.open(errors="replace") as handle:
        for line in handle:
            if "STEQ SOLUTION ARRAY" in line:
                finish()
                current = np.zeros((nt, nd), dtype=np.float64)
                filled = np.zeros((nt, nd), dtype=bool)
                in_solution = True
                continue
            if not in_solution:
                continue
            match = SOLUTION_LINE_RE.match(line)
            if not match:
                if "Equlibrium Equation" in line or "STEQ ARRAY" in line:
                    finish()
                    in_solution = False
                continue
            variable = int(match.group(1)) - 1
            depth = int(match.group(2)) - 1
            values = [ffloat(value) for value in match.group(3).split()]
            if not 0 <= variable < nt or depth < 0 or depth + len(values) > nd:
                raise BatteryError(f"{path}: invalid STEQ coordinates in {line.rstrip()!r}")
            assert current is not None and filled is not None
            current[variable, depth:depth + len(values)] = values
            filled[variable, depth:depth + len(values)] = True
    finish()
    return solutions


def parse_outgen(path: Path) -> dict[int, OutgenRecord]:
    records: dict[int, OutgenRecord] = {}
    pending_iteration: int | None = None
    if not path.exists():
        return records
    with path.open(errors="replace") as handle:
        for line in handle:
            match = OUTGEN_INC_RE.search(line)
            if match:
                value, iteration = ffloat(match.group(1)), int(match.group(2))
                records.setdefault(iteration, OutgenRecord()).increase_percent = value
                pending_iteration = iteration
                continue
            match = OUTGEN_DEC_RE.search(line)
            if match:
                value, iteration = ffloat(match.group(1)), int(match.group(2))
                records.setdefault(iteration, OutgenRecord()).decrease_percent = value
                pending_iteration = iteration
                continue
            match = OUTGEN_RETURN_RE.search(line)
            if match and pending_iteration is not None:
                records.setdefault(pending_iteration, OutgenRecord()).returned_maxch_percent = ffloat(
                    match.group(1)
                )
    return records


def parse_correction_link(path: Path) -> list[LinkRecord]:
    if not path.exists():
        return []
    records: list[LinkRecord] = []
    kind = None
    depth = None
    with path.open(errors="replace") as handle:
        for line in handle:
            header = LINK_HEADER_RE.search(line)
            if header:
                kind = header.group(1)
                depth = int(header.group(2))
                continue
            row = LINK_ROW_RE.match(line)
            if row and kind is not None and depth is not None:
                records.append(
                    LinkRecord(
                        kind=kind,
                        depth=depth,
                        correction=ffloat(row.group(1)),
                        ion=row.group(2),
                        superlevel=int(row.group(3)),
                        variable=int(row.group(4)),
                    )
                )
    return records


def read_rvtj_block(path: Path, header: str, nd: int) -> np.ndarray | None:
    if not path.exists():
        return None
    lines = path.read_text(errors="replace").splitlines()
    for line_index, line in enumerate(lines):
        if not line.strip().startswith(header):
            continue
        values: list[float] = []
        cursor = line_index + 1
        while cursor < len(lines) and len(values) < nd:
            try:
                values.extend(ffloat(item) for item in lines[cursor].split())
            except ValueError:
                break
            cursor += 1
        if len(values) >= nd:
            return np.asarray(values[:nd], dtype=np.float64)
    return None


def validate_rvtj(reader: ScrtempReader, path: Path) -> dict[str, Any]:
    anchors = {
        "radius": (read_rvtj_block(path, "Radius (10^10 cm)", reader.nd), reader.radius),
        "electron_density": (
            read_rvtj_block(path, "Electron density", reader.nd),
            reader.state(reader.last_it)[reader.nt - 2],
        ),
        "temperature": (
            read_rvtj_block(path, "Temperature (10^4K)", reader.nd),
            reader.state(reader.last_it)[reader.nt - 1],
        ),
    }
    if any(observed is None for observed, _ in anchors.values()):
        return {"present": path.exists(), "complete": False, "max_relative_errors": None}
    errors = {}
    for name, (observed, decoded) in anchors.items():
        assert observed is not None
        denominator = np.maximum(np.abs(observed), np.finfo(np.float64).tiny)
        errors[name] = float(np.max(np.abs(decoded - observed) / denominator))
    if max(errors.values()) > 1.0e-5:
        raise BatteryError(
            f"{path}: final SCRTEMP state does not match RVTJ anchors: {errors}"
        )
    return {"present": True, "complete": True, "max_relative_errors": errors}


def source_returned_maxch(solution: np.ndarray) -> float:
    """Reproduce solveba_v13.f MAXCH, including sentinel and NV fallback."""
    flat = solution.ravel()
    increases = np.sort(flat[flat <= 0.0])
    decreases = np.sort(flat[flat > 0.0])[::-1]
    increase = float(increases[0]) if increases.size else 0.0
    decrease = float(decreases[0]) if decreases.size else 0.0

    decrease_percent = 100.0 * decrease
    converted_decrease = (
        100.0 * decrease_percent / (100.0 - decrease_percent)
        if decrease_percent < 99.999
        else 1.0e7
    )
    maxch = max(-100.0 * increase, converted_decrease)

    inc10 = float(increases[SOURCE_NV - 1]) if increases.size >= SOURCE_NV else 0.0
    dec10 = float(decreases[SOURCE_NV - 1]) if decreases.size >= SOURCE_NV else 0.0
    if inc10 > -0.1 and dec10 < 0.1:
        maxch = 100.0 * max(abs(inc10), dec10 / (1.0 - min(dec10, 0.9999)))
    return maxch


def _steq_rounding_half_unit(value: float) -> float:
    """Half-unit uncertainty for WR2D_V2's ES12.4 solution serialization."""
    magnitude = abs(float(value))
    if magnitude == 0.0:
        return 0.5e-4
    exponent = math.floor(math.log10(magnitude))
    return 0.5 * (10.0 ** (exponent - 4))


def _source_decrease_conversion(fractional_correction: float) -> float:
    decrease_percent = 100.0 * fractional_correction
    if decrease_percent < 99.999:
        return 100.0 * decrease_percent / (100.0 - decrease_percent)
    return 1.0e7


def returned_maxch_rounding_interval(solution: np.ndarray) -> tuple[float, float]:
    """Bound source MAXCH from the printed STEQ solution's decimal precision.

    The source reports `SOL` through WR2D_V2's ES12.4 fields but prints the
    returned scalar at higher precision. Propagate each serialized value's
    half-unit interval through the extrema and 10-variable fallback. This
    avoids rejecting a correctly decoded solution when the decrease transform
    amplifies the five-significant-digit STEQ serialization near 100%.
    """
    flat = np.asarray(solution, dtype=np.float64).ravel()
    if flat.size == 0 or not np.isfinite(flat).all():
        raise BatteryError("cannot bound returned MAXCH for empty/non-finite STEQ")
    half_units = np.asarray(
        [_steq_rounding_half_unit(value) for value in flat], dtype=np.float64
    )
    lower = flat - half_units
    upper = flat + half_units

    increase_low = 100.0 * max(0.0, -float(np.min(upper)))
    increase_high = 100.0 * max(0.0, -float(np.min(lower)))
    decrease_low = _source_decrease_conversion(max(0.0, float(np.max(lower))))
    decrease_high = _source_decrease_conversion(max(0.0, float(np.max(upper))))
    raw_low = max(increase_low, decrease_low)
    raw_high = max(increase_high, decrease_high)

    fallback_possible = False
    fallback_always = False
    fallback_low = fallback_high = 0.0
    if flat.size >= SOURCE_NV:
        inc_low = float(np.sort(np.minimum(lower, 0.0))[SOURCE_NV - 1])
        inc_high = float(np.sort(np.minimum(upper, 0.0))[SOURCE_NV - 1])
        dec_low = float(np.sort(np.maximum(lower, 0.0))[::-1][SOURCE_NV - 1])
        dec_high = float(np.sort(np.maximum(upper, 0.0))[::-1][SOURCE_NV - 1])
        fallback_possible = inc_high > -0.1 and dec_low < 0.1
        fallback_always = inc_low > -0.1 and dec_high < 0.1
        if fallback_possible:
            def fallback_value(inc: float, dec: float) -> float:
                return 100.0 * max(
                    abs(inc), dec / (1.0 - min(dec, 0.9999))
                )

            fallback_low = fallback_value(inc_high, dec_low)
            fallback_high = fallback_value(inc_low, dec_high)

    if fallback_always:
        return fallback_low, fallback_high
    if fallback_possible:
        return min(raw_low, fallback_low), max(raw_high, fallback_high)
    return raw_low, raw_high


def species_slices(variables: list[Variable]) -> dict[str, np.ndarray]:
    grouped: dict[str, list[int]] = {}
    for variable in variables:
        grouped.setdefault(variable.species, []).append(variable.index - 1)
    return {key: np.asarray(value, dtype=np.int64) for key, value in grouped.items()}


def state_species_totals(
    state: np.ndarray, grouped: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    return {species: state[indexes].sum(axis=0) for species, indexes in grouped.items()}


def population_correction_summary(
    solution: np.ndarray, variables: list[Variable]
) -> dict[str, Any]:
    """Summarize every population-equation correction without activity filtering.

    These unfiltered quantiles are diagnostic only. They do not implement the
    v2 physically-active set, which also depends on ion fractions and rate/
    opacity/emissivity contributions.
    """
    npop = len(variables)
    if solution.ndim != 2 or solution.shape[0] < npop or npop == 0:
        raise BatteryError("invalid STEQ solution dimensions for population summary")
    values = np.abs(np.asarray(solution[:npop], dtype=np.float64)).ravel() * 100.0
    if values.size == 0 or not np.isfinite(values).all():
        raise BatteryError("population corrections are empty or non-finite")
    flat_index = int(np.argmax(np.abs(solution[:npop])))
    variable_index, depth_index = np.unravel_index(
        flat_index, solution[:npop].shape
    )
    owner = variables[int(variable_index)]
    return {
        "population_cells": int(values.size),
        "p95_percent": float(np.percentile(values, 95.0)),
        "max_percent": float(values.max()),
        "max_owner": {
            "variable": owner.index,
            "ion": owner.ion,
            "superlevel": owner.superlevel,
            "depth": int(depth_index) + 1,
        },
        "selection": "all population STEQ variables and depths; no activity filtering",
    }


def ion_fraction_active_correction_summary(
    solution: np.ndarray,
    pre_state: np.ndarray,
    variables: list[Variable],
    grouped: dict[str, np.ndarray],
    threshold: float = 1.0e-8,
) -> dict[str, Any]:
    """Measure corrections for ions active by fraction alone.

    This is a strict subset of the v2 active union: it includes every
    population equation for an ion whose fraction exceeds `threshold` at that
    depth, but does not add ions active only through rate/opacity/emissivity
    contributions. A max above the v2 limit is therefore decisive failure;
    a value below the limit is not sufficient for acceptance.
    """
    npop = len(variables)
    if (
        solution.ndim != 2
        or pre_state.ndim != 2
        or solution.shape[0] < npop
        or pre_state.shape[0] < npop
        or solution.shape[1] != pre_state.shape[1]
        or npop == 0
        or not math.isfinite(threshold)
        or threshold <= 0.0
    ):
        raise BatteryError("invalid dimensions or threshold for ion-fraction activity summary")

    species_totals = state_species_totals(pre_state, grouped)
    ion_rows: dict[str, list[int]] = {}
    for row, variable in enumerate(variables):
        ion_rows.setdefault(variable.ion, []).append(row)

    active = np.zeros((npop, pre_state.shape[1]), dtype=bool)
    ion_fraction_by_row_depth = np.zeros_like(active, dtype=np.float64)
    for ion, rows in ion_rows.items():
        species = variables[rows[0]].species
        total = species_totals.get(species)
        if total is None:
            raise BatteryError(f"missing species total for active-set ion {ion}")
        ion_population = pre_state[np.asarray(rows, dtype=np.int64)].sum(axis=0)
        fraction = np.zeros_like(total, dtype=np.float64)
        positive_total = total > 0.0
        fraction[positive_total] = ion_population[positive_total] / total[positive_total]
        active_depths = fraction > threshold
        active[np.ix_(np.asarray(rows, dtype=np.int64), active_depths)] = True
        ion_fraction_by_row_depth[np.ix_(np.asarray(rows, dtype=np.int64), np.arange(total.size))] = fraction

    if not bool(active.any()):
        raise BatteryError("no ion-fraction-active population cells")
    selected = np.abs(np.asarray(solution[:npop], dtype=np.float64))[active] * 100.0
    if not np.isfinite(selected).all():
        raise BatteryError("ion-fraction-active corrections contain non-finite values")
    active_rows, active_depths = np.nonzero(active)
    selected_flat = np.abs(solution[:npop][active])
    maximum_index = int(np.argmax(selected_flat))
    owner_row = int(active_rows[maximum_index])
    owner_depth = int(active_depths[maximum_index])
    owner = variables[owner_row]
    return {
        "threshold": threshold,
        "active_cells": int(selected.size),
        "p95_percent": float(np.percentile(selected, 95.0)),
        "max_percent": float(selected.max()),
        "max_owner": {
            "variable": owner.index,
            "ion": owner.ion,
            "superlevel": owner.superlevel,
            "depth": owner_depth + 1,
            "ion_fraction": float(ion_fraction_by_row_depth[owner_row, owner_depth]),
        },
        "contribution_channels_included": False,
        "selection": "ion fraction > threshold only; contribution-active ions not added",
    }


def shell_weights(radius: np.ndarray) -> np.ndarray:
    dr = np.empty_like(radius)
    if radius.size == 1:
        dr[0] = 1.0
    else:
        dr[0] = abs(radius[0] - radius[1])
        dr[-1] = abs(radius[-2] - radius[-1])
        if radius.size > 2:
            dr[1:-1] = 0.5 * np.abs(radius[:-2] - radius[2:])
    return radius * radius * dr


def mean_charges(
    state: np.ndarray,
    variables: list[Variable],
    grouped: dict[str, np.ndarray],
    weights: np.ndarray,
) -> dict[str, float]:
    charge = np.asarray([variable.charge for variable in variables], dtype=np.float64)
    result: dict[str, float] = {}
    for species, indexes in grouped.items():
        populations = state[indexes]
        numerator = float((populations * charge[indexes, None] * weights[None, :]).sum())
        denominator = float((populations * weights[None, :]).sum())
        result[species] = numerator / denominator
    return result


def find_variable(variables: list[Variable], ion: str, superlevel: int) -> int:
    hits = [v.index - 1 for v in variables if v.ion == ion and v.superlevel == superlevel]
    if len(hits) != 1:
        raise BatteryError(f"expected one {ion} SL{superlevel}; found {len(hits)}")
    return hits[0]


def physical_metrics(
    reader: ScrtempReader,
    iteration: int,
    variables: list[Variable],
    grouped: dict[str, np.ndarray],
) -> dict[str, Any]:
    state = reader.state(iteration)
    previous = reader.state(iteration - 1) if reader.has_iteration(iteration - 1) else None
    npop = len(variables)
    ne_index = reader.nt - 2
    weights = shell_weights(reader.radius)
    totals = state_species_totals(state, grouped)
    charges = mean_charges(state, variables, grouped, weights)

    siiii_indexes = np.asarray(
        [v.index - 1 for v in variables if v.ion == "SkIII"], dtype=np.int64
    )
    if siiii_indexes.size == 0:
        raise BatteryError("tracked ion SkIII (Si III) is absent")
    caiv_ground_index = find_variable(variables, "CaIV", 1)
    d21, d27 = 20, 26
    if reader.nd <= d27:
        raise BatteryError("P-TF tracked depths d21/d27 are absent")

    siiii_fraction = float(state[siiii_indexes, d27].sum() / totals["Si"][d27])
    caiv_ground = float(state[caiv_ground_index, d21])
    caiv_indexes = grouped["Ca"][[
        variables[index].ion == "CaIV" for index in grouped["Ca"]
    ]]
    caiv_fraction = float(state[caiv_indexes, d21].sum() / totals["Ca"][d21])

    output: dict[str, Any] = {
        "ne_d21": float(state[ne_index, d21]),
        "ne_d27": float(state[ne_index, d27]),
        "mean_charge": charges,
        "siiii_fraction_d27": siiii_fraction,
        "caiv_ground_d21": caiv_ground,
        "caiv_fraction_d21": caiv_fraction,
    }
    if previous is None:
        output.update(
            {
                "ne_max_abs_dln": None,
                "ne_max_depth": None,
                "pop_weighted_step_max": None,
                "pop_weighted_step_depth": None,
                "pop_weighted_step_d26": None,
                "pop_weighted_step_d27": None,
                "mean_charge_max_abs_step": None,
                "mean_charge_max_rel_step": None,
                "siiii_fraction_step_percent": None,
                "caiv_ground_step_percent": None,
            }
        )
        return output

    dne = np.abs(np.log(state[ne_index] / previous[ne_index]))
    popw = np.abs(state[:npop] - previous[:npop]).sum(axis=0) / previous[:npop].sum(axis=0)
    previous_totals = state_species_totals(previous, grouped)
    previous_charges = mean_charges(previous, variables, grouped, weights)
    previous_siiii = float(previous[siiii_indexes, d27].sum() / previous_totals["Si"][d27])
    previous_caiv_ground = float(previous[caiv_ground_index, d21])
    charge_abs = {key: charges[key] - previous_charges[key] for key in charges}
    charge_rel = {
        key: math.log(charges[key] / previous_charges[key]) for key in charges
    }
    output.update(
        {
            "ne_max_abs_dln": float(dne.max()),
            "ne_max_depth": int(dne.argmax() + 1),
            "pop_weighted_step_max": float(popw.max()),
            "pop_weighted_step_depth": int(popw.argmax() + 1),
            "pop_weighted_step_d26": float(popw[25]),
            "pop_weighted_step_d27": float(popw[26]),
            "mean_charge_abs_step": charge_abs,
            "mean_charge_max_abs_step": float(max(abs(x) for x in charge_abs.values())),
            "mean_charge_max_rel_step": float(max(abs(x) for x in charge_rel.values())),
            "siiii_fraction_step_percent": 100.0 * (siiii_fraction / previous_siiii - 1.0),
            "caiv_ground_step_percent": 100.0 * (caiv_ground / previous_caiv_ground - 1.0),
        }
    )
    return output


def gate_metrics(
    solution: np.ndarray,
    pre_state: np.ndarray,
    variables: list[Variable],
    grouped: dict[str, np.ndarray],
    threshold: float,
) -> dict[str, Any]:
    npop = len(variables)
    totals = state_species_totals(pre_state, grouped)
    include = np.zeros((npop, pre_state.shape[1]), dtype=bool)
    for species, indexes in grouped.items():
        include[indexes] = pre_state[indexes] >= threshold * totals[species][None, :]
    if not bool(include.any()):
        raise BatteryError("P-TF gate excluded every population variable")

    magnitude = np.abs(solution[:npop]).copy()
    magnitude[~include] = -np.inf
    variable_index, depth_index = (
        int(value) for value in np.unravel_index(int(np.argmax(magnitude)), magnitude.shape)
    )
    variable = variables[variable_index]
    population = float(pre_state[variable_index, depth_index])
    species_total = float(totals[variable.species][depth_index])
    raw_variable, raw_depth = (
        int(value)
        for value in np.unravel_index(int(np.argmax(np.abs(solution))), solution.shape)
    )
    raw_meta = variables[raw_variable] if raw_variable < npop else None
    return {
        "all_population_correction": population_correction_summary(solution, variables),
        "ion_fraction_active_correction": ion_fraction_active_correction_summary(
            solution, pre_state, variables, grouped
        ),
        "threshold": threshold,
        "included_cells": int(include.sum()),
        "excluded_cells": int(include.size - include.sum()),
        "gated_maxch_percent": 100.0 * float(magnitude[variable_index, depth_index]),
        "gated_owner": {
            "variable": variable.index,
            "ion": variable.ion,
            "species": variable.species,
            "superlevel": variable.superlevel,
            "level_name": variable.level_name,
            "depth": depth_index + 1,
            "correction": float(solution[variable_index, depth_index]),
            "population_before": population,
            "species_total_before": species_total,
            "species_fraction_before": population / species_total,
            "terminal": variable.terminal,
            "in_d21_d32": 21 <= depth_index + 1 <= 32,
        },
        "ungated_raw_max_abs_percent": 100.0 * float(abs(solution[raw_variable, raw_depth])),
        "ungated_raw_owner": {
            "variable": raw_variable + 1,
            "ion": raw_meta.ion if raw_meta else ("Ne" if raw_variable == npop else "T"),
            "superlevel": raw_meta.superlevel if raw_meta else None,
            "depth": raw_depth + 1,
            "correction": float(solution[raw_variable, raw_depth]),
        },
        "ungated_returned_maxch_percent": source_returned_maxch(solution),
    }


def validate_solution(
    iteration: int,
    solution: np.ndarray,
    outgen: dict[int, OutgenRecord],
) -> dict[str, Any]:
    record = outgen.get(iteration)
    computed_inc = -100.0 * float(solution.min())
    computed_dec = 100.0 * float(solution.max())
    computed_return = source_returned_maxch(solution)
    output: dict[str, Any] = {
        "outgen_present": record is not None,
        "computed_increase_percent": computed_inc,
        "computed_decrease_percent": computed_dec,
        "computed_returned_maxch_percent": computed_return,
    }
    if record is None:
        return output
    checks = []
    for label, computed, observed in (
        ("increase", computed_inc, record.increase_percent),
        ("decrease", computed_dec, record.decrease_percent),
        ("returned", computed_return, record.returned_maxch_percent),
    ):
        if observed is None:
            continue
        # OUTGEN extrema have only three significant digits; the returned line
        # is more precise, while STEQ_VALS stores five significant digits.
        tolerance = 6.0e-3 if label != "returned" else 1.0e-4
        if label == "returned":
            interval = returned_maxch_rounding_interval(solution)
            rounding = tolerance * max(abs(computed), abs(observed), 1.0e-10)
            ok = interval[0] - rounding <= observed <= interval[1] + rounding
        else:
            interval = None
            ok = math.isclose(computed, observed, rel_tol=tolerance, abs_tol=1.0e-10)
        check = {"field": label, "computed": computed, "observed": observed, "ok": ok}
        if interval is not None:
            check["printed_solution_interval"] = list(interval)
        checks.append(check)
        if not ok:
            raise BatteryError(
                f"iteration {iteration}: STEQ/OUTGEN {label} mismatch "
                f"({computed} vs {observed})"
            )
    output["checks"] = checks
    return output


def validate_correction_link(
    records: list[LinkRecord],
    solution: np.ndarray,
    variables: list[Variable],
) -> dict[str, Any]:
    if not records:
        return {"present": False, "rows_checked": 0, "max_relative_value_error": None}
    errors = []
    for record in records:
        if not 1 <= record.variable <= solution.shape[0] or not 1 <= record.depth <= solution.shape[1]:
            raise BatteryError(f"CORRECTION_LINK coordinate out of range: {record}")
        observed = float(solution[record.variable - 1, record.depth - 1])
        scale = max(abs(observed), abs(record.correction), 1.0e-300)
        relative = abs(observed - record.correction) / scale
        errors.append(relative)
        if relative > 6.0e-5:
            raise BatteryError(
                f"CORRECTION_LINK value mismatch at var {record.variable}, d{record.depth}: "
                f"{record.correction} vs {observed}"
            )
        if record.variable <= len(variables):
            meta = variables[record.variable - 1]
            if meta.ion != record.ion or meta.superlevel != record.superlevel:
                raise BatteryError(
                    f"CORRECTION_LINK metadata mismatch at var {record.variable}: "
                    f"{record.ion} SL{record.superlevel} vs {meta.ion} SL{meta.superlevel}"
                )
    return {
        "present": True,
        "rows_checked": len(records),
        "max_relative_value_error": max(errors),
    }


def run_label_and_path(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        label, raw_path = spec.split("=", 1)
        if not label:
            raise BatteryError(f"empty rundir label in {spec!r}")
    else:
        raw_path = spec
        label = Path(raw_path.rstrip("/")).name
    path = Path(raw_path).expanduser().resolve()
    if not path.is_dir():
        raise BatteryError(f"not a rundir: {path}")
    return label, path


def solution_iterations(
    solutions: list[np.ndarray], outgen: dict[int, OutgenRecord], reader: ScrtempReader
) -> list[int]:
    if not solutions:
        return []
    ids = sorted(outgen)
    if ids:
        # Restarted runs may retain older STEQ blocks while OUTGEN contains
        # only the current run's iterations.  OUTGEN IDs therefore map to the
        # newest corresponding STEQ blocks, not to the start of the STEQ file.
        count = min(len(ids), len(solutions))
        mapped_ids = ids[-count:]
        if all(reader.has_iteration(value) for value in mapped_ids):
            return mapped_ids
        raise BatteryError(
            "OUTGEN iterations do not map to the SCRTEMP state range: "
            f"{mapped_ids} vs {reader.state_first_it}..{reader.last_it}"
        )
    first = reader.last_it - len(solutions) + 1
    return list(range(first, reader.last_it + 1))


def mapped_solution_pairs(
    solutions: list[np.ndarray], iterations: list[int]
) -> list[tuple[int, np.ndarray]]:
    if not iterations:
        return []
    if len(iterations) > len(solutions):
        raise BatteryError("more mapped STEQ iterations than solution blocks")
    return list(zip(iterations, solutions[-len(iterations):]))


def signatures_match(left: list[Variable], right: list[Variable]) -> bool:
    return [
        (v.index, v.ion, v.superlevel, v.charge) for v in left
    ] == [
        (v.index, v.ion, v.superlevel, v.charge) for v in right
    ]


def collect(args: argparse.Namespace) -> dict[str, Any]:
    run_inputs = [run_label_and_path(spec) for spec in args.rundirs]
    prepared = []
    canonical_variables: list[Variable] | None = None
    canonical_nd_nt: tuple[int, int] | None = None
    available_solution_iterations: list[int] = []

    for label, path in run_inputs:
        nd, nt = read_dimensions(path)
        variables = read_variables(path, nt - 2)
        if canonical_nd_nt is None:
            canonical_nd_nt = (nd, nt)
            canonical_variables = variables
        elif canonical_nd_nt != (nd, nt) or not signatures_match(canonical_variables or [], variables):
            raise BatteryError(f"{path}: model/variable map differs from the first rundir")
        reader = ScrtempReader(path / "SCRTEMP", nd, nt, args.state_first_it)
        solutions = parse_solutions(path / "STEQ_VALS", nt, nd)
        outgen = parse_outgen(path / "OUTGEN")
        sol_ids = solution_iterations(solutions, outgen, reader)
        available_solution_iterations.extend(sol_ids)
        prepared.append((label, path, reader, variables, solutions, sol_ids, outgen))

    if canonical_variables is None:
        raise BatteryError("no rundirs supplied")
    grouped = species_slices(canonical_variables)
    state_first = min(item[2].state_first_it for item in prepared)
    state_last = max(item[2].last_it for item in prepared)
    default_first = min(available_solution_iterations) if available_solution_iterations else state_first + 1
    default_last = max(available_solution_iterations) if available_solution_iterations else state_last
    first_it = args.from_it if args.from_it is not None else default_first
    last_it = args.to_it if args.to_it is not None else default_last
    if first_it > last_it:
        raise BatteryError("--from-it must not exceed --to-it")

    rows: dict[int, dict[str, Any]] = {
        iteration: {
            "iteration": iteration,
            "state_source": None,
            "solution_source": None,
            "gate": None,
            "physical": None,
            "validation": None,
            "note": None,
        }
        for iteration in range(first_it, last_it + 1)
    }
    run_validation = []

    for label, path, reader, variables, solutions, sol_ids, outgen in prepared:
        for iteration in range(first_it, last_it + 1):
            if rows[iteration]["physical"] is None and reader.has_iteration(iteration):
                rows[iteration]["physical"] = physical_metrics(
                    reader, iteration, variables, grouped
                )
                rows[iteration]["state_source"] = label

        for iteration, solution in mapped_solution_pairs(solutions, sol_ids):
            validation = validate_solution(iteration, solution, outgen)
            if iteration not in rows:
                continue
            if not reader.has_iteration(iteration - 1):
                raise BatteryError(
                    f"{path}: gated iteration {iteration} has no pre-correction state {iteration - 1}"
                )
            computed_gate = gate_metrics(
                solution,
                reader.state(iteration - 1),
                variables,
                grouped,
                args.threshold,
            )
            if rows[iteration]["gate"] is not None:
                old = rows[iteration]["gate"]
                if not math.isclose(
                    old["gated_maxch_percent"],
                    computed_gate["gated_maxch_percent"],
                    rel_tol=1.0e-12,
                ):
                    raise BatteryError(f"conflicting STEQ blocks for iteration {iteration}")
            else:
                rows[iteration]["gate"] = computed_gate
                rows[iteration]["solution_source"] = label
                rows[iteration]["validation"] = validation

        links = parse_correction_link(path / "CORRECTION_LINK")
        link_result = (
            validate_correction_link(links, solutions[-1], variables)
            if solutions
            else {"present": bool(links), "rows_checked": 0, "max_relative_value_error": None}
        )
        run_validation.append(
            {
                "label": label,
                "rundir": str(path),
                "nd": reader.nd,
                "nt": reader.nt,
                "scrtemp_states": reader.nstates,
                "scrtemp_iteration_range": [reader.state_first_it, reader.last_it],
                "steq_blocks": len(solutions),
                "steq_unmapped_prefix_blocks": len(solutions) - len(sol_ids),
                "steq_iterations": sol_ids,
                "correction_link": link_result,
                "rvtj": validate_rvtj(reader, path / "RVTJ"),
            }
        )

    for iteration, row in rows.items():
        if row["physical"] is None:
            row["note"] = "SCRTEMP state unavailable"
        elif row["gate"] is None:
            row["note"] = "STEQ solution unavailable; physical metrics only"

    return {
        "schema": "ptf-gated-metrics-v2",
        "gate_definition": {
            "all_population_correction": (
                "diagnostic p95/max over every population STEQ variable and depth; "
                "not the v2 physically-active population acceptance metric"
            ),
            "ion_fraction_active_correction": (
                "p95/max for ions with fraction >1e-8 only; rate/opacity/emissivity-active ions "
                "are not added, so this is a strict subset diagnostic, not acceptance"
            ),
            "threshold": args.threshold,
            "exclude_if": "POPS(variable,depth) < threshold * species_total(species,depth)",
            "gated_maxch": "100 * max(abs(STEQ correction)) over included population variables",
            "pre_correction_state": "SCRTEMP iteration N-1 for STEQ iteration N",
            "ungated_return": "solveba_v13.f MAXCH including 1e7-percent decrease sentinel and NV=10 fallback",
        },
        "iteration_range": [first_it, last_it],
        "runs": run_validation,
        "rows": [rows[key] for key in sorted(rows)],
    }


def sci(value: float | None, digits: int = 3) -> str:
    return "N/A" if value is None else f"{value:.{digits}e}"


def markdown_output(result: dict[str, Any]) -> str:
    lines = [
        "# P-TF offline gated-metrics battery",
        "",
        f"Gate: `POPS >= {result['gate_definition']['threshold']:.1e} × species_total`; "
        "the state and corrections are not modified.",
        "",
        "## Gate yardstick",
        "",
        "| it | source | gated MAXCH (%) | owner | d | owner/species | ungated raw (%) | returned MAXCH (%) | E1 1e2–1e4 |",
        "|---:|:---|---:|:---|---:|---:|---:|---:|:---:|",
    ]
    for row in result["rows"]:
        gate = row["gate"]
        if gate is None:
            lines.append(
                f"| {row['iteration']} | {row['solution_source'] or '—'} | N/A | — | — | — | — | — | N/A |"
            )
            continue
        owner = gate["gated_owner"]
        e1 = "PASS" if 1.0e2 <= gate["gated_maxch_percent"] <= 1.0e4 else "FAIL"
        lines.append(
            f"| {row['iteration']} | {row['solution_source']} | {gate['gated_maxch_percent']:.4e} | "
            f"{owner['ion']} SL{owner['superlevel']} (v{owner['variable']}) | {owner['depth']} | "
            f"{owner['species_fraction_before']:.3e} | {gate['ungated_raw_max_abs_percent']:.4e} | "
            f"{gate['ungated_returned_maxch_percent']:.4e} | {e1} |"
        )

    lines.extend(
        [
            "",
            "## Population-correction diagnostics",
            "",
            "The ion-fraction subset uses `ion fraction > 1e-8` only and omits contribution-active ions.",
            "Neither diagnostic alone is the full v2 acceptance metric.",
            "",
            "| it | all-pop cells | all-pop p95 (%) | all-pop max (%) | ion-fraction-active cells | active p95 (%) | active max (%) | active max owner |",
            "|---:|---:|---:|---:|---:|---:|---:|:---|",
        ]
    )
    for row in result["rows"]:
        gate = row["gate"]
        if gate is None:
            lines.append(f"| {row['iteration']} | N/A | N/A | N/A | N/A | N/A | N/A | — |")
            continue
        summary = gate["all_population_correction"]
        fraction_active = gate["ion_fraction_active_correction"]
        owner = fraction_active["max_owner"]
        lines.append(
            f"| {row['iteration']} | {summary['population_cells']} | "
            f"{summary['p95_percent']:.4e} | {summary['max_percent']:.4e} | "
            f"{fraction_active['active_cells']} | {fraction_active['p95_percent']:.4e} | "
            f"{fraction_active['max_percent']:.4e} | {owner['ion']} SL{owner['superlevel']} "
            f"(v{owner['variable']}, d{owner['depth']}; ion fraction {owner['ion_fraction']:.3e}) |"
        )

    lines.extend(
        [
            "",
            "## Physical trajectory",
            "",
            "| it | state | max abs(dln ne) | popw max | d | max abs(d<q>) | Si III frac@d27 | step (%) | Ca IV gs@d21 | step (%) |",
            "|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["rows"]:
        phys = row["physical"]
        if phys is None:
            lines.append(f"| {row['iteration']} | — | N/A | N/A | — | N/A | N/A | N/A | N/A | N/A |")
            continue
        lines.append(
            f"| {row['iteration']} | {row['state_source']} | {sci(phys['ne_max_abs_dln'])} | "
            f"{sci(phys['pop_weighted_step_max'])} | {phys['pop_weighted_step_depth'] or '—'} | "
            f"{sci(phys['mean_charge_max_abs_step'])} | {phys['siiii_fraction_d27']:.6f} | "
            f"{sci(phys['siiii_fraction_step_percent'])} | {phys['caiv_ground_d21']:.6e} | "
            f"{sci(phys['caiv_ground_step_percent'])} |"
        )

    lines.extend(["", "## Decoder validation", ""])
    for run in result["runs"]:
        link = run["correction_link"]
        rvtj = run["rvtj"]
        rvtj_text = "RVTJ unavailable"
        if rvtj["complete"]:
            worst = max(rvtj["max_relative_errors"].values())
            rvtj_text = f"RVTJ worst relative error {worst:.3e}"
        lines.append(
            f"- `{run['label']}`: SCRTEMP {run['scrtemp_states']} states; "
            f"STEQ {run['steq_blocks']} blocks at {run['steq_iterations']}; "
            f"CORRECTION_LINK {link['rows_checked']} rows checked"
            + (
                f" (max relative rounding error {link['max_relative_value_error']:.3e})."
                if link["max_relative_value_error"] is not None
                else "."
            )
            + f" {rvtj_text}."
        )
    return "\n".join(lines) + "\n"


def csv_output(result: dict[str, Any]) -> str:
    stream = io.StringIO()
    fieldnames = [
        "iteration", "state_source", "solution_source", "gated_maxch_percent",
        "all_population_correction_cells", "all_population_correction_p95_percent",
        "all_population_correction_max_percent",
        "all_population_correction_max_owner_ion", "all_population_correction_max_owner_superlevel",
        "all_population_correction_max_owner_depth",
        "ion_fraction_active_cells", "ion_fraction_active_p95_percent",
        "ion_fraction_active_max_percent", "ion_fraction_active_max_owner_ion",
        "ion_fraction_active_max_owner_superlevel", "ion_fraction_active_max_owner_depth",
        "ion_fraction_active_max_owner_fraction",
        "owner_variable", "owner_ion", "owner_superlevel", "owner_depth",
        "owner_species_fraction", "ungated_raw_max_abs_percent",
        "ungated_returned_maxch_percent", "ne_max_abs_dln", "ne_max_depth",
        "pop_weighted_step_max", "pop_weighted_step_depth", "mean_charge_max_abs_step",
        "mean_charge_max_rel_step", "siiii_fraction_d27", "siiii_fraction_step_percent",
        "caiv_ground_d21", "caiv_ground_step_percent", "caiv_fraction_d21", "note",
    ]
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    for row in result["rows"]:
        gate = row["gate"] or {}
        owner = gate.get("gated_owner", {})
        all_population = gate.get("all_population_correction", {})
        all_population_owner = all_population.get("max_owner", {})
        fraction_active = gate.get("ion_fraction_active_correction", {})
        fraction_active_owner = fraction_active.get("max_owner", {})
        phys = row["physical"] or {}
        writer.writerow(
            {
                "iteration": row["iteration"],
                "state_source": row["state_source"],
                "solution_source": row["solution_source"],
                "gated_maxch_percent": gate.get("gated_maxch_percent"),
                "all_population_correction_cells": all_population.get("population_cells"),
                "all_population_correction_p95_percent": all_population.get("p95_percent"),
                "all_population_correction_max_percent": all_population.get("max_percent"),
                "all_population_correction_max_owner_ion": all_population_owner.get("ion"),
                "all_population_correction_max_owner_superlevel": all_population_owner.get("superlevel"),
                "all_population_correction_max_owner_depth": all_population_owner.get("depth"),
                "ion_fraction_active_cells": fraction_active.get("active_cells"),
                "ion_fraction_active_p95_percent": fraction_active.get("p95_percent"),
                "ion_fraction_active_max_percent": fraction_active.get("max_percent"),
                "ion_fraction_active_max_owner_ion": fraction_active_owner.get("ion"),
                "ion_fraction_active_max_owner_superlevel": fraction_active_owner.get("superlevel"),
                "ion_fraction_active_max_owner_depth": fraction_active_owner.get("depth"),
                "ion_fraction_active_max_owner_fraction": fraction_active_owner.get("ion_fraction"),
                "owner_variable": owner.get("variable"),
                "owner_ion": owner.get("ion"),
                "owner_superlevel": owner.get("superlevel"),
                "owner_depth": owner.get("depth"),
                "owner_species_fraction": owner.get("species_fraction_before"),
                "ungated_raw_max_abs_percent": gate.get("ungated_raw_max_abs_percent"),
                "ungated_returned_maxch_percent": gate.get("ungated_returned_maxch_percent"),
                "ne_max_abs_dln": phys.get("ne_max_abs_dln"),
                "ne_max_depth": phys.get("ne_max_depth"),
                "pop_weighted_step_max": phys.get("pop_weighted_step_max"),
                "pop_weighted_step_depth": phys.get("pop_weighted_step_depth"),
                "mean_charge_max_abs_step": phys.get("mean_charge_max_abs_step"),
                "mean_charge_max_rel_step": phys.get("mean_charge_max_rel_step"),
                "siiii_fraction_d27": phys.get("siiii_fraction_d27"),
                "siiii_fraction_step_percent": phys.get("siiii_fraction_step_percent"),
                "caiv_ground_d21": phys.get("caiv_ground_d21"),
                "caiv_ground_step_percent": phys.get("caiv_ground_step_percent"),
                "caiv_fraction_d21": phys.get("caiv_fraction_d21"),
                "note": row["note"],
            }
        )
    return stream.getvalue()


def write_result(text: str, output: str) -> None:
    if output == "-":
        sys.stdout.write(text)
        return
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def reject_rundir_output(output: str, rundirs: Iterable[str]) -> None:
    """Keep the CLI's direct output writes outside all input rundirs."""
    if output == "-":
        return
    target = Path(output).expanduser().resolve()
    for spec in rundirs:
        _, rundir = run_label_and_path(spec)
        try:
            target.relative_to(rundir)
        except ValueError:
            continue
        raise BatteryError(f"refusing to write output inside input rundir: {target}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only P-TF gated MAXCH and physical-metric battery"
    )
    parser.add_argument(
        "rundirs",
        nargs="+",
        metavar="[LABEL=]RUNDIR",
        help="one or more relT3-family CMFGEN rundirs in lineage order",
    )
    parser.add_argument("--from-it", type=int, default=None, help="first global iteration to report")
    parser.add_argument("--to-it", type=int, default=None, help="last global iteration to report")
    parser.add_argument(
        "--state-first-it",
        type=int,
        default=1,
        help="global iteration represented by the first SCRTEMP state (default: 1)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="species-relative population gate (default: 1e-20)",
    )
    parser.add_argument(
        "--format", choices=("markdown", "json", "csv"), default="markdown"
    )
    parser.add_argument("--output", default="-", help="output path, or - for stdout")
    args = parser.parse_args(argv)
    if not math.isfinite(args.threshold) or args.threshold < 0.0:
        parser.error("--threshold must be finite and non-negative")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        reject_rundir_output(args.output, args.rundirs)
        result = collect(args)
        if args.format == "json":
            rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
        elif args.format == "csv":
            rendered = csv_output(result)
        else:
            rendered = markdown_output(result)
        write_result(rendered, args.output)
    except (BatteryError, OSError, ValueError) as exc:
        print(f"ptf_gated_metrics.py: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
