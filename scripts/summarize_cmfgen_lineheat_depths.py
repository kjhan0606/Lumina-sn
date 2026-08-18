#!/usr/bin/env python3
"""Stream finite CMFGEN LINEHEAT totals at selected one-based depths."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from extract_cmfgen_line_net_fixture import (
    CMFGEN_RE_INTERNAL_TO_CGS,
    Header,
    lineheat_header,
    vectors,
)


class OnlineFsum:
    """Online form of the non-overlapping-partials summation algorithm."""

    def __init__(self) -> None:
        self.partials: list[float] = []

    def add(self, value: float) -> None:
        partials = self.partials
        index = 0
        for partial in partials:
            if abs(value) < abs(partial):
                value, partial = partial, value
            high = value + partial
            low = partial - (high - value)
            if low != 0.0:
                partials[index] = low
                index += 1
            value = high
        partials[index:] = [value]

    def total(self) -> float:
        return math.fsum(self.partials)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def rvtj_vector(path: Path, label: str, depth_count: int) -> list[float]:
    """Read one named finite depth vector from CMFGEN's RVTJ ledger."""
    active = False
    values: list[float] = []
    with path.open("rt", encoding="ascii", errors="strict") as stream:
        for line_number, line in enumerate(stream, 1):
            if not active:
                if line.strip() == label:
                    active = True
                continue
            for token in line.split():
                try:
                    value = float(token.replace("D", "E").replace("d", "e"))
                except ValueError as exc:
                    raise ValueError(
                        f"{path}:{line_number}: nonnumeric {label} token {token!r}"
                    ) from exc
                if not math.isfinite(value):
                    raise ValueError(
                        f"{path}:{line_number}: nonfinite {label} value"
                    )
                values.append(value)
                if len(values) == depth_count:
                    return values
                if len(values) > depth_count:
                    raise ValueError(f"{path}: {label} exceeds {depth_count} values")
    if not active:
        raise ValueError(f"{path}: RVTJ section not found: {label}")
    raise ValueError(
        f"{path}: RVTJ section {label} has {len(values)}, expected {depth_count}"
    )


def rvtj_state(path: Path, depth_count: int) -> dict[str, object]:
    velocity = rvtj_vector(path, "Velocity (km/s)", depth_count)
    temperature_1e4 = rvtj_vector(path, "Temperature (10^4K)", depth_count)
    electron_density = rvtj_vector(path, "Electron density", depth_count)
    atom_density = rvtj_vector(path, "Atom Density", depth_count)
    ion_density = rvtj_vector(path, "Ion Density", depth_count)
    mass_density = rvtj_vector(path, "Mass Density (gm/cm^3)", depth_count)
    return {
        "source": str(path.resolve()),
        "sha256": digest(path),
        "velocity_km_s": velocity,
        "temperature_K": [value * 1.0e4 for value in temperature_1e4],
        "electron_density_cm3": electron_density,
        "atom_density_cm3": atom_density,
        "ion_density_cm3": ion_density,
        "mass_density_g_cm3": mass_density,
    }


def summarize(
    path: Path,
    depth_count: int,
    depths: list[int],
    rvtj_path: Path | None = None,
) -> dict[str, object]:
    if depth_count <= 0:
        raise ValueError("depth_count must be positive")
    if not depths or len(depths) != len(set(depths)):
        raise ValueError("depths must be nonempty and unique")
    if any(depth <= 0 or depth > depth_count for depth in depths):
        raise ValueError("selected depth is outside the LINEHEAT vector")

    accumulators = {
        depth: {
            "signed": OnlineFsum(),
            "absolute": OnlineFsum(),
            "positive_count": 0,
            "negative_count": 0,
            "zero_count": 0,
            "max_absolute": 0.0,
            "max_absolute_line_id": -1,
        }
        for depth in depths
    }
    line_records = 0
    for header, values in vectors(path, depth_count, lineheat_header):
        if not isinstance(header, Header):
            raise ValueError("unexpected LINEHEAT header type")
        line_records += 1
        for depth in depths:
            value = values[depth - 1]
            if not math.isfinite(value):
                raise ValueError(
                    f"line {header.line_id} depth {depth}: nonfinite LINEHEAT"
                )
            accumulator = accumulators[depth]
            accumulator["signed"].add(value)
            accumulator["absolute"].add(abs(value))
            if value > 0.0:
                accumulator["positive_count"] += 1
            elif value < 0.0:
                accumulator["negative_count"] += 1
            else:
                accumulator["zero_count"] += 1
            if abs(value) > accumulator["max_absolute"]:
                accumulator["max_absolute"] = abs(value)
                accumulator["max_absolute_line_id"] = header.line_id

    if line_records == 0:
        raise ValueError("LINEHEAT yielded no numbered Sobolev records")

    state = rvtj_state(rvtj_path, depth_count) if rvtj_path else None
    rendered: dict[str, object] = {}
    for depth in depths:
        accumulator = accumulators[depth]
        signed_internal = accumulator["signed"].total()
        absolute_internal = accumulator["absolute"].total()
        if not math.isfinite(signed_internal) or not math.isfinite(absolute_internal):
            raise ValueError(f"depth {depth}: nonfinite aggregate")
        rendered[str(depth)] = {
            "signed_internal": signed_internal,
            "signed_cgs_erg_cm3_s": signed_internal * CMFGEN_RE_INTERNAL_TO_CGS,
            "absolute_internal": absolute_internal,
            "cancellation_condition": (
                absolute_internal / abs(signed_internal)
                if signed_internal != 0.0
                else (0.0 if absolute_internal == 0.0 else "infinite")
            ),
            "positive_count": accumulator["positive_count"],
            "negative_count": accumulator["negative_count"],
            "zero_count": accumulator["zero_count"],
            "max_absolute_internal": accumulator["max_absolute"],
            "max_absolute_line_id": accumulator["max_absolute_line_id"],
        }
        if state is not None:
            rendered[str(depth)]["velocity_km_s"] = state["velocity_km_s"][
                depth - 1
            ]
            rendered[str(depth)]["temperature_K"] = state["temperature_K"][
                depth - 1
            ]
            rendered[str(depth)]["electron_density_cm3"] = state[
                "electron_density_cm3"
            ][depth - 1]
            rendered[str(depth)]["atom_density_cm3"] = state[
                "atom_density_cm3"
            ][depth - 1]
            rendered[str(depth)]["ion_density_cm3"] = state[
                "ion_density_cm3"
            ][depth - 1]
            rendered[str(depth)]["mass_density_g_cm3"] = state[
                "mass_density_g_cm3"
            ][depth - 1]

    report = {
        "schema": "lumina-cmfgen-lineheat-depth-summary-v1",
        "source": str(path.resolve()),
        "depth_indexing": "CMFGEN_ONE_BASED",
        "depth_count": depth_count,
        "line_records": line_records,
        "equation": "sum_scaled_LINEHEAT*4*pi*1e-10",
        "summation": "ONLINE_NONOVERLAPPING_BINARY64_PARTIALS",
        "physical_mutation": 0,
        "repair": 0,
        "depths": rendered,
        "verdict": "COMPLETE",
    }
    if state is not None:
        report["state_source"] = {
            "path": state["source"],
            "sha256": state["sha256"],
            "format": "CMFGEN_RVTJ",
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lineheat", type=Path, required=True)
    parser.add_argument("--depth-count", type=int, default=90)
    parser.add_argument("--depth", type=int, action="append", required=True)
    parser.add_argument("--rvtj", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    try:
        report = summarize(
            args.lineheat, args.depth_count, args.depth, args.rvtj
        )
    except (OSError, ValueError) as exc:
        print(f"[cmfgen-lineheat-depth-summary] ERROR: {exc}")
        return 2
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
