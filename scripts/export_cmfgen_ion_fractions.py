#!/usr/bin/env python3
"""Export all-shell Si/S ion fractions from preserved CMFGEN SCRTEMP states.

Fractions use the same represented-population denominator as
``ptf_gated_metrics.state_species_totals``.  In particular, an inferred
terminal closure variable is retained and identified in the output rather
than silently dropped.  The input run is read-only; output files are created
exclusively and are never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.ptf_gated_metrics import (
        BatteryError,
        ScrtempReader,
        Variable,
        ffloat,
        ion_charge,
        physical_species,
        read_dimensions,
        read_variables,
        read_rvtj_block,
        species_prefix,
        validate_rvtj,
    )
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from ptf_gated_metrics import (  # direct ``python scripts/...`` invocation
        BatteryError,
        ScrtempReader,
        Variable,
            ffloat,
            ion_charge,
            physical_species,
        read_dimensions,
        read_variables,
        read_rvtj_block,
            species_prefix,
        validate_rvtj,
    )


TARGET_ELEMENTS = ("Si", "S")
POP_SPECIES_NAMES = {"Si": ("SIL", "Si"), "S": ("SUL", "S")}
STAGE_NAMES = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII"}
POP_REL_TOL = 1.0e-7  # E16.7 output rounding: at most half a unit in the 8th digit.
POP_FLOAT_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?")
POP_LEVEL_RE = re.compile(r"^\s*Number of (.+?) levels:\s*(\d+)\s*$", re.IGNORECASE)
CSV_FIELDS = (
    "iteration",
    "depth_index_1based",
    "velocity_km_s",
    "radius_1e10_cm",
    "element",
    "ion_stage",
    "charge",
    "ion_population_sum",
    "element_population_sum",
    "fraction_of_element",
    "superlevel_count",
    "includes_terminal_closure",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def target_groups(variables: list[Variable]) -> dict[tuple[str, int], list[int]]:
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for variable in variables:
        if variable.species in TARGET_ELEMENTS:
            grouped[(variable.species, variable.charge)].append(variable.index - 1)
    missing = [element for element in TARGET_ELEMENTS
               if not any(key[0] == element for key in grouped)]
    if missing:
        raise BatteryError(f"requested element(s) absent from model: {', '.join(missing)}")
    return dict(grouped)


def extract_rows(
    reader: ScrtempReader,
    variables: list[Variable],
    iterations: range,
    velocity_km_s: np.ndarray,
) -> list[dict[str, Any]]:
    groups = target_groups(variables)
    species_rows: dict[str, list[int]] = {
        element: [variable.index - 1 for variable in variables
                  if variable.species == element]
        for element in TARGET_ELEMENTS
    }
    result: list[dict[str, Any]] = []

    for iteration in iterations:
        if not reader.has_iteration(iteration):
            raise BatteryError(
                f"iteration {iteration} is outside SCRTEMP range "
                f"[{reader.state_first_it}, {reader.last_it}]"
            )
        state = reader.state(iteration)
        if state.shape[0] < len(variables):
            raise BatteryError("SCRTEMP has fewer population rows than LEVEL_SL_STEQ_LINKS")
        if not np.isfinite(state[:len(variables)]).all():
            raise BatteryError(f"non-finite population in SCRTEMP iteration {iteration}")
        if np.any(state[:len(variables)] < 0.0):
            raise BatteryError(f"negative population in SCRTEMP iteration {iteration}")
        radius = np.asarray(reader.radius, dtype=np.float64)
        if radius.size != reader.nd or not np.isfinite(radius).all() or np.any(radius <= 0.0):
            raise BatteryError("SCRTEMP radius vector is invalid")
        if (velocity_km_s.size != reader.nd or not np.isfinite(velocity_km_s).all()
                or np.any(velocity_km_s <= 0.0)):
            raise BatteryError("RVTJ velocity vector is invalid")

        totals: dict[str, np.ndarray] = {}
        for element in TARGET_ELEMENTS:
            total = state[np.asarray(species_rows[element], dtype=np.int64)].sum(axis=0)
            if not np.isfinite(total).all() or np.any(total <= 0.0):
                raise BatteryError(
                    f"non-positive or non-finite {element} population total "
                    f"in SCRTEMP iteration {iteration}"
                )
            totals[element] = total

        ordered_groups = sorted(
            groups.items(), key=lambda item: (TARGET_ELEMENTS.index(item[0][0]), item[0][1])
        )
        for (element, charge), indexes in ordered_groups:
            stage_number = charge + 1
            if stage_number not in STAGE_NAMES:
                raise BatteryError(f"unsupported ion stage charge {charge} for {element}")
            index_array = np.asarray(indexes, dtype=np.int64)
            ion_population = state[index_array].sum(axis=0)
            fraction = ion_population / totals[element]
            for depth in range(reader.nd):
                result.append({
                    "iteration": iteration,
                    "depth_index_1based": depth + 1,
                    "velocity_km_s": float(velocity_km_s[depth]),
                    "radius_1e10_cm": float(radius[depth]),
                    "element": element,
                    "ion_stage": STAGE_NAMES[stage_number],
                    "charge": charge,
                    "ion_population_sum": float(ion_population[depth]),
                    "element_population_sum": float(totals[element][depth]),
                    "fraction_of_element": float(fraction[depth]),
                    "superlevel_count": len(indexes),
                    "includes_terminal_closure": any(
                        variables[index].terminal for index in indexes
                    ),
                })
    return result


def render_csv(rows: list[dict[str, Any]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def write_new(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(contents)


def read_numeric_block(
    lines: list[str], start: int, count: int, context: str
) -> tuple[np.ndarray, int]:
    values: list[float] = []
    cursor = start
    while len(values) < count:
        if cursor >= len(lines):
            raise BatteryError(f"{context}: truncated numeric block ({len(values)}/{count})")
        tokens = POP_FLOAT_RE.findall(lines[cursor])
        remaining = count - len(values)
        if not tokens or len(tokens) > remaining:
            raise BatteryError(f"{context}: malformed numeric row at line {cursor + 1}")
        values.extend(ffloat(token) for token in tokens)
        cursor += 1
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise BatteryError(f"{context}: non-finite value")
    return array, cursor


def read_pop_species_output(
    path: Path, element: str, nd: int, variables: list[Variable]
) -> tuple[np.ndarray, np.ndarray, int]:
    """Read independent POP_SPECIES total and the top active ion's DI vector.

    CMFGEN writes POP{species} with POP_SPECIES first, then each active ion's
    XzV_F matrix and DXzV_F vector.  When the next ion is absent, source code
    identifies that last DXzV_F as the next ion's ground-state population.
    """
    lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    aliases = "|".join(re.escape(name) for name in POP_SPECIES_NAMES[element])
    abundance_re = re.compile(rf"^\s*(?:{aliases})/He abundance:", re.IGNORECASE)
    abundance_headers = [i for i, line in enumerate(lines) if abundance_re.match(line)]
    if len(abundance_headers) != 1:
        raise BatteryError(
            f"{path}: expected one {element}/He abundance header, got {len(abundance_headers)}"
        )
    species_total, _ = read_numeric_block(
        lines, abundance_headers[0] + 1, nd, f"{path} POP_SPECIES[{element}]"
    )
    if np.any(species_total <= 0.0):
        raise BatteryError(f"{path}: non-positive POP_SPECIES density for {element}")

    active = [v for v in variables if v.species == element and not v.terminal]
    terminal = [v for v in variables if v.species == element and v.terminal]
    if not active or len(terminal) != 1:
        raise BatteryError(
            f"{path}: expected active populations and one terminal closure row for {element}"
        )
    top_charge = max(v.charge for v in active)
    terminal_var = terminal[0]
    if terminal_var.charge != top_charge + 1 or terminal_var.superlevel != 1:
        raise BatteryError(
            f"{path}: terminal row {terminal_var.ion} does not immediately follow "
            f"top active charge {top_charge}"
        )

    top_di: list[np.ndarray] = []
    for index, line in enumerate(lines):
        match = POP_LEVEL_RE.match(line)
        if not match:
            continue
        ion = match.group(1).strip()
        nlevels = int(match.group(2))
        if nlevels <= 0:
            continue
        if index + 1 >= len(lines) or "Oscillator date:" not in lines[index + 1]:
            raise BatteryError(f"{path}: missing oscillator-date row for {ion}")
        try:
            ion_species = physical_species(species_prefix(ion))
            charge = ion_charge(ion)
        except BatteryError as exc:
            raise BatteryError(f"{path}: cannot identify ion block {ion!r}: {exc}") from exc
        cursor = index + 2
        _, cursor = read_numeric_block(
            lines, cursor, nlevels * nd, f"{path} XzV_F[{ion}]"
        )
        di_values, _ = read_numeric_block(lines, cursor, nd, f"{path} DXzV_F[{ion}]")
        if ion_species == element and charge == top_charge:
            top_di.append(di_values)

    if len(top_di) != 1:
        raise BatteryError(
            f"{path}: expected one top-stage ({element}, charge {top_charge}) DI vector, "
            f"got {len(top_di)}"
        )
    if np.any(top_di[0] < 0.0):
        raise BatteryError(f"{path}: negative terminal DI population for {element}")
    return species_total, top_di[0], terminal_var.index - 1


def audit_population_closure(
    rundir: Path, reader: ScrtempReader, variables: list[Variable]
) -> dict[str, Any]:
    """Cross-check final SCRTEMP sums and terminal DI against CMFGEN POP files."""
    state = reader.state(reader.last_it)
    result: dict[str, Any] = {}
    for element in TARGET_ELEMENTS:
        rows = [v.index - 1 for v in variables if v.species == element]
        if not rows:
            raise BatteryError(f"no SCRTEMP variables found for {element}")
        candidates = [rundir / f"POP{name}" for name in POP_SPECIES_NAMES[element]
                      if (rundir / f"POP{name}").is_file()]
        if len(candidates) != 1:
            raise BatteryError(f"{rundir}: expected one unambiguous {element} POP file")
        path = candidates[0]
        pop_total, pop_di, terminal_index = read_pop_species_output(
            path, element, reader.nd, variables
        )
        scrtemp_total = state[np.asarray(rows, dtype=np.int64)].sum(axis=0)
        scrtemp_di = state[terminal_index]
        total_rel = np.abs(scrtemp_total - pop_total) / np.maximum(
            np.maximum(np.abs(scrtemp_total), np.abs(pop_total)), np.finfo(np.float64).tiny
        )
        di_rel = np.abs(scrtemp_di - pop_di) / np.maximum(
            np.maximum(np.abs(scrtemp_di), np.abs(pop_di)), np.finfo(np.float64).tiny
        )
        if not np.isfinite(total_rel).all() or not np.isfinite(di_rel).all():
            raise BatteryError(f"non-finite {element} population closure residual")
        max_total_rel = float(np.max(total_rel))
        max_di_rel = float(np.max(di_rel))
        if max_total_rel > POP_REL_TOL:
            raise BatteryError(
                f"{element} SCRTEMP represented total differs from {path.name} "
                f"POP_SPECIES by {max_total_rel:.6e} (limit {POP_REL_TOL:.1e})"
            )
        if max_di_rel > POP_REL_TOL:
            raise BatteryError(
                f"{element} SCRTEMP terminal closure differs from top-ion DXzV_F "
                f"by {max_di_rel:.6e} (limit {POP_REL_TOL:.1e})"
            )
        result[element] = {
            "pop_species_file": path.name,
            "pop_species_sha256": sha256(path),
            "final_iteration": reader.last_it,
            "species_total_max_relative_error": max_total_rel,
            "terminal_di_stage": STAGE_NAMES[terminal_var_charge(variables, element)],
            "terminal_di_max_relative_error": max_di_rel,
            "terminal_di_fraction_min": float(np.min(pop_di / pop_total)),
            "terminal_di_fraction_max": float(np.max(pop_di / pop_total)),
            "relative_tolerance": POP_REL_TOL,
            "tolerance_basis": "CMFGEN E16.7 text serialization (8 significant digits)",
        }
    return result


def terminal_var_charge(variables: list[Variable], element: str) -> int:
    terminal = [v for v in variables if v.species == element and v.terminal]
    if len(terminal) != 1:
        raise BatteryError(f"expected one terminal closure variable for {element}")
    return terminal[0].charge + 1


def export(
    rundir: Path,
    output_csv: Path,
    output_json: Path,
    first_it: int | None,
    last_it: int | None,
    state_first_it: int,
) -> dict[str, Any]:
    rundir = rundir.resolve()
    output_csv = output_csv.resolve()
    output_json = output_json.resolve()
    if output_csv == output_json:
        raise BatteryError("CSV and JSON output paths must differ")
    if output_csv.exists() or output_json.exists():
        raise FileExistsError("refusing to overwrite an existing output")

    nd, nt = read_dimensions(rundir)
    variables = read_variables(rundir, nt - 2)
    reader = ScrtempReader(rundir / "SCRTEMP", nd, nt, state_first_it)
    rvtj_path = rundir / "RVTJ"
    rvtj_validation = validate_rvtj(reader, rvtj_path)
    if not rvtj_validation["complete"]:
        raise BatteryError("RVTJ anchors are incomplete; refusing depth-coordinate export")
    velocity = read_rvtj_block(rvtj_path, "Velocity (km/s)", nd)
    if velocity is None:
        raise BatteryError("RVTJ velocity block is missing")
    velocity_delta = np.diff(velocity)
    if not (np.all(velocity_delta > 0.0) or np.all(velocity_delta < 0.0)):
        raise BatteryError("RVTJ velocity grid is not strictly monotonic")
    first = reader.state_first_it if first_it is None else first_it
    last = reader.last_it if last_it is None else last_it
    if first > last:
        raise BatteryError("--from-it must not exceed --to-it")
    if first < reader.state_first_it or last > reader.last_it:
        raise BatteryError(
            f"requested iteration range [{first}, {last}] outside SCRTEMP range "
            f"[{reader.state_first_it}, {reader.last_it}]"
        )

    population_closure = audit_population_closure(rundir, reader, variables)
    rows = extract_rows(reader, variables, range(first, last + 1), velocity)
    csv_text = render_csv(rows)
    input_paths = [rundir / name for name in
                   ("MODEL", "LEVEL_SL_STEQ_LINKS", "SCRTEMP", "RVTJ")]
    input_paths.extend(rundir / population_closure[element]["pop_species_file"]
                       for element in TARGET_ELEMENTS)
    if (rundir / "MODEL_SPEC").is_file():
        input_paths.append(rundir / "MODEL_SPEC")
    metadata = {
        "schema": "CMFGEN_SI_S_ION_FRACTIONS_V1",
        "rundir": str(rundir),
        "dimensions": {"depth_points": nd, "total_variables": nt,
                       "population_variables": len(variables)},
        "iteration_range": {"first": first, "last": last,
                             "count": last - first + 1},
        "fraction_definition": (
            "sum of SCRTEMP population variables for an element/ion stage divided by "
            "the sum of all represented ion-stage population variables for that element; "
            "inferred terminal closure variables are included"
        ),
        "depth_coordinate": {
            "velocity_source": "RVTJ Velocity (km/s)",
            "radius_source": "SCRTEMP header, units 1e10 cm; cross-checked against RVTJ",
            "rvtj_final_state_validation": rvtj_validation,
            "velocity_order": "decreasing" if velocity[0] > velocity[-1] else "increasing",
        },
        "terminal_closure_rows": [
            {"variable": v.index, "element": v.species, "charge": v.charge,
             "ion": v.ion}
            for v in variables if v.terminal and v.species in TARGET_ELEMENTS
        ],
        "population_conservation_and_di_crosscheck": population_closure,
        "input_sha256": {path.name: sha256(path) for path in input_paths},
        "code_sha256": {
            "export_cmfgen_ion_fractions.py": sha256(Path(__file__).resolve()),
            "ptf_gated_metrics.py": sha256(
                Path(__file__).resolve().with_name("ptf_gated_metrics.py")
            ),
        },
        "csv_sha256": hashlib.sha256(csv_text.encode("utf-8")).hexdigest(),
        "row_count": len(rows),
        "outputs": {"csv": str(output_csv), "metadata": str(output_json)},
    }
    metadata_text = json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    write_new(output_csv, csv_text)
    try:
        write_new(output_json, metadata_text)
    except Exception:
        # The CSV is a newly created evidence artifact; do not silently remove it.
        # Report its path so the user can decide how to handle the incomplete pair.
        raise RuntimeError(f"metadata write failed; CSV remains at {output_csv}")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rundir", type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--from-it", type=int)
    parser.add_argument("--to-it", type=int)
    parser.add_argument("--state-first-it", type=int, default=1)
    args = parser.parse_args()
    try:
        metadata = export(
            args.rundir, args.output_csv, args.output_json,
            args.from_it, args.to_it, args.state_first_it,
        )
    except (BatteryError, FileExistsError, OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": "EXPORTED",
        "csv": metadata["outputs"]["csv"],
        "metadata": metadata["outputs"]["metadata"],
        "iterations": metadata["iteration_range"],
        "rows": metadata["row_count"],
        "csv_sha256": metadata["csv_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
