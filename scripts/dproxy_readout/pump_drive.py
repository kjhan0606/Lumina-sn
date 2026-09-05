#!/usr/bin/env python3
"""Read-only, full-line-list pump-drive readout.

The large line_list.csv is opened exactly once and processed with csv.DictReader.
Only descriptive products are written, and only beneath this script's directory.
Run with ``python3 -B`` to suppress bytecode files.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import os
import platform
import socket
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable


sys.dont_write_bytecode = True

SCHEMA = "pump-drive-readout-v1"
BASE_DIR = Path(__file__).resolve().parent
TARGET_Z = frozenset((26, 27, 28))
TARGET_ION = 1
SHELL_ID = 0
TEMPERATURE_K = 10020.0
K_B_EV_K = 8.617333262e-5
H_EV_S = 4.135667696e-15
PHOTON_SPLIT_EV = 4.0

COMPONENTS = (
    "all",
    "ge4eV",
    "lt4eV",
    "ground",
    "metastable",
    "other",
)

PAIR_FIELDS = (
    "shell",
    "Z",
    "ion",
    "upper_level",
    "runtime_sl_id_0",
    "runtime_sl_id_1",
)
SL_FIELDS = ("shell", "Z", "ion", "runtime_sl_id", "Y_s")
LEVEL_FIELDS = (
    "atomic_number",
    "ion_number",
    "level_number",
    "energy_eV",
    "g",
    "metastable",
)
LINE_FIELDS = (
    "atomic_number",
    "ion_number",
    "level_number_lower",
    "level_number_upper",
    "line_id",
    "f_lu",
    "nu",
)
SPECTRAL_FIELDS = ("shell_id", "bin_id", "nu_lo_Hz", "nu_hi_Hz", "J_nu")


class PumpDriveError(RuntimeError):
    """Fail-closed input/schema error."""


def require_fields(actual: Iterable[str] | None, required: Iterable[str], label: str) -> None:
    names = set(actual or ())
    missing = sorted(set(required) - names)
    if missing:
        raise PumpDriveError(f"{label}: missing required columns: {','.join(missing)}")


def parse_int(raw: str, label: str) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise PumpDriveError(f"{label}: invalid integer {raw!r}") from exc


def parse_float(raw: str, label: str, *, finite: bool = True) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise PumpDriveError(f"{label}: invalid float {raw!r}") from exc
    if finite and not math.isfinite(value):
        raise PumpDriveError(f"{label}: non-finite float {raw!r}")
    return value


def safe_log_ratio(value1: float | None, value0: float | None) -> float | None:
    if (
        value0 is None
        or value1 is None
        or not math.isfinite(value0)
        or not math.isfinite(value1)
        or value0 <= 0.0
        or value1 <= 0.0
    ):
        return None
    result = math.log(value1) - math.log(value0)
    return result if math.isfinite(result) else None


@dataclass(frozen=True)
class Target:
    Z: int
    ion: int
    upper_level: int
    runtime_sl_id: int

    @property
    def key(self) -> tuple[int, int, int]:
        return (self.Z, self.ion, self.upper_level)

    @property
    def sl_key(self) -> tuple[int, int, int]:
        return (self.Z, self.ion, self.runtime_sl_id)


@dataclass(frozen=True)
class LowerLevel:
    energy_eV: float
    g: float
    metastable: int


@dataclass
class StableSum:
    """Neumaier compensated streaming sum."""

    total: float = 0.0
    correction: float = 0.0

    def add(self, value: float) -> None:
        updated = self.total + value
        if abs(self.total) >= abs(value):
            self.correction += (self.total - updated) + value
        else:
            self.correction += (value - updated) + self.total
        self.total = updated

    def value(self) -> float | None:
        value = self.total + self.correction
        return value if math.isfinite(value) else None


@dataclass
class UpperAccumulator:
    selected_line_count: int = 0
    used_line_count: int = 0
    excluded_line_count: int = 0
    component_line_counts: Counter[str] = field(default_factory=Counter)
    sums: dict[str, tuple[StableSum, StableSum]] = field(
        default_factory=lambda: {
            component: (StableSum(), StableSum()) for component in COMPONENTS
        }
    )

    def add(self, components: Iterable[str], contribution0: float, contribution1: float) -> None:
        self.used_line_count += 1
        for component in components:
            self.component_line_counts[component] += 1
            self.sums[component][0].add(contribution0)
            self.sums[component][1].add(contribution1)


@dataclass(frozen=True)
class SpectralBin:
    bin_id: int
    lo: float
    hi: float
    J: float


@dataclass
class SpectralGrid:
    bins: list[SpectralBin]
    rows_total: int
    rows_shell0: int
    rows_other_shell: int
    nonfinite_J_rows_shell0: int
    nonpositive_J_rows_shell0: int
    lows: list[float] = field(init=False)

    def __post_init__(self) -> None:
        self.lows = [item.lo for item in self.bins]

    def lookup(self, nu: float) -> tuple[bool, float | None, int | None]:
        position = bisect.bisect_right(self.lows, nu) - 1
        if position < 0:
            return False, None, None
        item = self.bins[position]
        if item.lo <= nu < item.hi:
            return True, item.J, item.bin_id
        return False, None, None


def load_targets(
    pairs_path: Path, sl_groups_path: Path
) -> tuple[dict[tuple[int, int, int], Target], dict[tuple[int, int, int], float], dict[str, int]]:
    targets: dict[tuple[int, int, int], Target] = {}
    pair_rows = 0
    with pairs_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require_fields(reader.fieldnames, PAIR_FIELDS, str(pairs_path))
        for row_number, row in enumerate(reader, 2):
            pair_rows += 1
            prefix = f"{pairs_path}:row {row_number}"
            shell = parse_int(row["shell"], f"{prefix}:shell")
            Z = parse_int(row["Z"], f"{prefix}:Z")
            ion = parse_int(row["ion"], f"{prefix}:ion")
            upper = parse_int(row["upper_level"], f"{prefix}:upper_level")
            sl0 = parse_int(row["runtime_sl_id_0"], f"{prefix}:runtime_sl_id_0")
            sl1 = parse_int(row["runtime_sl_id_1"], f"{prefix}:runtime_sl_id_1")
            if shell != SHELL_ID or Z not in TARGET_Z or ion != TARGET_ION:
                raise PumpDriveError(
                    f"{prefix}: expected shell=0, Z in 26/27/28, ion=1"
                )
            if sl0 != sl1:
                raise PumpDriveError(f"{prefix}: runtime SL differs between iterations")
            candidate = Target(Z, ion, upper, sl0)
            previous = targets.get(candidate.key)
            if previous is not None and previous.runtime_sl_id != sl0:
                raise PumpDriveError(
                    f"{prefix}: upper target maps to conflicting runtime SL IDs"
                )
            targets[candidate.key] = candidate
    if not targets:
        raise PumpDriveError(f"{pairs_path}: no target upper levels")

    y_by_sl: dict[tuple[int, int, int], float] = {}
    sl_rows = 0
    with sl_groups_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require_fields(reader.fieldnames, SL_FIELDS, str(sl_groups_path))
        for row_number, row in enumerate(reader, 2):
            sl_rows += 1
            prefix = f"{sl_groups_path}:row {row_number}"
            shell = parse_int(row["shell"], f"{prefix}:shell")
            Z = parse_int(row["Z"], f"{prefix}:Z")
            ion = parse_int(row["ion"], f"{prefix}:ion")
            runtime_sl_id = parse_int(
                row["runtime_sl_id"], f"{prefix}:runtime_sl_id"
            )
            Y_s = parse_float(row["Y_s"], f"{prefix}:Y_s")
            if shell != SHELL_ID or Z not in TARGET_Z or ion != TARGET_ION:
                raise PumpDriveError(
                    f"{prefix}: expected shell=0, Z in 26/27/28, ion=1"
                )
            key = (Z, ion, runtime_sl_id)
            if key in y_by_sl:
                raise PumpDriveError(f"{prefix}: duplicate runtime SL key {key}")
            y_by_sl[key] = Y_s

    mapped_sl = {target.sl_key for target in targets.values()}
    missing = sorted(mapped_sl - set(y_by_sl))
    if missing:
        raise PumpDriveError(
            f"{sl_groups_path}: missing Y_s for {len(missing)} mapped runtime SLs; "
            f"first={missing[0]}"
        )
    return targets, y_by_sl, {
        "pairs_rows": pair_rows,
        "distinct_target_upper_levels": len(targets),
        "mapped_runtime_sl_count": len(mapped_sl),
        "sl_groups_rows": sl_rows,
        "unreferenced_sl_groups_rows": len(set(y_by_sl) - mapped_sl),
    }


def load_levels(path: Path) -> tuple[dict[tuple[int, int, int], LowerLevel], dict[str, int]]:
    levels: dict[tuple[int, int, int], LowerLevel] = {}
    counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require_fields(reader.fieldnames, LEVEL_FIELDS, str(path))
        for row_number, row in enumerate(reader, 2):
            counts["rows_total"] += 1
            prefix = f"{path}:row {row_number}"
            Z = parse_int(row["atomic_number"], f"{prefix}:atomic_number")
            ion = parse_int(row["ion_number"], f"{prefix}:ion_number")
            if Z not in TARGET_Z or ion != TARGET_ION:
                counts["rows_other_species"] += 1
                continue
            level_number = parse_int(row["level_number"], f"{prefix}:level_number")
            energy = parse_float(row["energy_eV"], f"{prefix}:energy_eV")
            g = parse_float(row["g"], f"{prefix}:g")
            metastable = parse_int(row["metastable"], f"{prefix}:metastable")
            if g <= 0.0:
                raise PumpDriveError(f"{prefix}: g must be positive")
            if metastable not in (0, 1):
                raise PumpDriveError(f"{prefix}: metastable must be 0 or 1")
            key = (Z, ion, level_number)
            if key in levels:
                raise PumpDriveError(f"{prefix}: duplicate level key {key}")
            levels[key] = LowerLevel(energy, g, metastable)
            counts["rows_target_species"] += 1
    return levels, dict(counts)


def load_spectral(path: Path) -> SpectralGrid:
    bins: list[SpectralBin] = []
    bin_ids: set[int] = set()
    counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require_fields(reader.fieldnames, SPECTRAL_FIELDS, str(path))
        for row_number, row in enumerate(reader, 2):
            counts["rows_total"] += 1
            prefix = f"{path}:row {row_number}"
            shell = parse_int(row["shell_id"], f"{prefix}:shell_id")
            if shell != SHELL_ID:
                counts["rows_other_shell"] += 1
                continue
            counts["rows_shell0"] += 1
            bin_id = parse_int(row["bin_id"], f"{prefix}:bin_id")
            lo = parse_float(row["nu_lo_Hz"], f"{prefix}:nu_lo_Hz")
            hi = parse_float(row["nu_hi_Hz"], f"{prefix}:nu_hi_Hz")
            # A non-finite J is retained so every affected selected line is
            # explicitly excluded and counted instead of poisoning a sum.
            J = parse_float(row["J_nu"], f"{prefix}:J_nu", finite=False)
            if not lo < hi:
                raise PumpDriveError(f"{prefix}: spectral bin must satisfy lo < hi")
            if bin_id in bin_ids:
                raise PumpDriveError(f"{prefix}: duplicate shell-0 bin_id {bin_id}")
            bin_ids.add(bin_id)
            if not math.isfinite(J):
                counts["nonfinite_J_rows_shell0"] += 1
            elif J <= 0.0:
                counts["nonpositive_J_rows_shell0"] += 1
            bins.append(SpectralBin(bin_id, lo, hi, J))
    if not bins:
        raise PumpDriveError(f"{path}: no shell_id=0 spectral bins")
    bins.sort(key=lambda item: (item.lo, item.hi, item.bin_id))
    for previous, current in zip(bins, bins[1:]):
        if current.lo < previous.hi:
            raise PumpDriveError(
                f"{path}: overlapping shell-0 bins {previous.bin_id} and {current.bin_id}"
            )
    return SpectralGrid(
        bins=bins,
        rows_total=counts["rows_total"],
        rows_shell0=counts["rows_shell0"],
        rows_other_shell=counts["rows_other_shell"],
        nonfinite_J_rows_shell0=counts["nonfinite_J_rows_shell0"],
        nonpositive_J_rows_shell0=counts["nonpositive_J_rows_shell0"],
    )


def lower_class(level: LowerLevel) -> str:
    # Ground wins the intentional overlap with metastable=1.
    if level.energy_eV < 0.1:
        return "ground"
    if level.metastable == 1:
        return "metastable"
    return "other"


def stream_line_list(
    path: Path,
    targets: dict[tuple[int, int, int], Target],
    levels: dict[tuple[int, int, int], LowerLevel],
    spectral0: SpectralGrid,
    spectral1: SpectralGrid,
) -> tuple[dict[tuple[int, int, int], UpperAccumulator], dict[str, Any]]:
    accumulators = {key: UpperAccumulator() for key in targets}
    counts: Counter[str] = Counter()
    exclusions: Counter[str] = Counter()

    # This is the sole open/read pass over line_list.csv.
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require_fields(reader.fieldnames, LINE_FIELDS, str(path))
        for row_number, row in enumerate(reader, 2):
            counts["rows_total"] += 1
            prefix = f"{path}:row {row_number}"
            try:
                Z = parse_int(row["atomic_number"], f"{prefix}:atomic_number")
                ion = parse_int(row["ion_number"], f"{prefix}:ion_number")
                upper = parse_int(
                    row["level_number_upper"], f"{prefix}:level_number_upper"
                )
            except PumpDriveError:
                counts["rows_with_malformed_identity"] += 1
                raise
            if Z not in TARGET_Z or ion != TARGET_ION:
                counts["rows_skipped_atomic_number_or_ion"] += 1
                continue
            key = (Z, ion, upper)
            if key not in targets:
                counts["rows_skipped_upper_outside_target"] += 1
                continue

            acc = accumulators[key]
            counts["selected_target_lines"] += 1
            acc.selected_line_count += 1
            reasons: set[str] = set()
            try:
                lower = parse_int(
                    row["level_number_lower"], f"{prefix}:level_number_lower"
                )
                if not row["line_id"]:
                    raise PumpDriveError(f"{prefix}: empty line_id")
                f_lu = parse_float(row["f_lu"], f"{prefix}:f_lu", finite=False)
                nu = parse_float(row["nu"], f"{prefix}:nu", finite=False)
            except PumpDriveError:
                reasons.add("malformed_selected_line")
                lower = -1
                f_lu = math.nan
                nu = math.nan

            level = levels.get((Z, ion, lower))
            if level is None:
                reasons.add("lower_level_not_in_levels")
            if not math.isfinite(f_lu):
                reasons.add("nonfinite_f_lu")
            elif f_lu <= 0.0:
                reasons.add("nonpositive_f_lu")
            if not math.isfinite(nu):
                reasons.add("nonfinite_nu")
            elif nu <= 0.0:
                reasons.add("nonpositive_nu")

            weight: float | None = None
            if level is not None and math.isfinite(f_lu) and f_lu > 0.0:
                try:
                    boltzmann = math.exp(-level.energy_eV / (K_B_EV_K * TEMPERATURE_K))
                    weight = f_lu * level.g * boltzmann
                except OverflowError:
                    weight = None
                if weight is None or not math.isfinite(weight):
                    reasons.add("nonfinite_weight")
                elif weight <= 0.0:
                    reasons.add("nonpositive_weight")

            found0 = found1 = False
            J0 = J1 = None
            if math.isfinite(nu) and nu > 0.0:
                found0, J0, _ = spectral0.lookup(nu)
                found1, J1, _ = spectral1.lookup(nu)
                if not found0:
                    reasons.add("frequency_outside_spectral_bins_iter0")
                if not found1:
                    reasons.add("frequency_outside_spectral_bins_iter1")
                if found0 and J0 is not None:
                    if not math.isfinite(J0):
                        reasons.add("nonfinite_J_iter0")
                    elif J0 <= 0.0:
                        reasons.add("nonpositive_J_iter0")
                if found1 and J1 is not None:
                    if not math.isfinite(J1):
                        reasons.add("nonfinite_J_iter1")
                    elif J1 <= 0.0:
                        reasons.add("nonpositive_J_iter1")

            contribution0 = contribution1 = None
            if not reasons:
                assert weight is not None and J0 is not None and J1 is not None
                contribution0 = weight * J0
                contribution1 = weight * J1
                if not math.isfinite(contribution0) or not math.isfinite(contribution1):
                    reasons.add("nonfinite_weighted_contribution")
                elif contribution0 <= 0.0 or contribution1 <= 0.0:
                    reasons.add("nonpositive_weighted_contribution")

            if reasons:
                counts["selected_lines_excluded"] += 1
                acc.excluded_line_count += 1
                for reason in sorted(reasons):
                    exclusions[reason] += 1
                continue

            assert level is not None and contribution0 is not None and contribution1 is not None
            photon_component = (
                "ge4eV" if H_EV_S * nu >= PHOTON_SPLIT_EV else "lt4eV"
            )
            acc.add(
                ("all", photon_component, lower_class(level)),
                contribution0,
                contribution1,
            )
            counts["selected_lines_used"] += 1

    counts["line_list_stream_passes"] = 1
    for name in (
        "rows_total",
        "rows_with_malformed_identity",
        "rows_skipped_atomic_number_or_ion",
        "rows_skipped_upper_outside_target",
        "selected_target_lines",
        "selected_lines_used",
        "selected_lines_excluded",
    ):
        counts.setdefault(name, 0)
    return accumulators, {
        **dict(counts),
        "exclusions_by_reason": dict(sorted(exclusions.items())),
    }


def component_values(acc: UpperAccumulator, component: str) -> dict[str, Any]:
    value0 = acc.sums[component][0].value()
    value1 = acc.sums[component][1].value()
    return {
        f"used_line_count_{component}": acc.component_line_counts[component],
        f"P_{component}_0": value0,
        f"P_{component}_1": value1,
        f"dlog_P_{component}": safe_log_ratio(value1, value0),
    }


def build_upper_rows(
    targets: dict[tuple[int, int, int], Target],
    accumulators: dict[tuple[int, int, int], UpperAccumulator],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(targets):
        target = targets[key]
        acc = accumulators[key]
        row: dict[str, Any] = {
            "Z": target.Z,
            "ion": target.ion,
            "upper_level": target.upper_level,
            "runtime_sl_id": target.runtime_sl_id,
            "selected_line_count": acc.selected_line_count,
            "used_line_count": acc.used_line_count,
            "excluded_line_count": acc.excluded_line_count,
        }
        for component in COMPONENTS:
            row.update(component_values(acc, component))
        row["X_u"] = row["dlog_P_all"]
        rows.append(row)
    return rows


def finite_sum_or_none(values: list[float | None]) -> float | None:
    if any(value is None for value in values):
        return None
    result = math.fsum(value for value in values if value is not None)
    return result if math.isfinite(result) else None


def build_sl_rows(
    targets: dict[tuple[int, int, int], Target],
    y_by_sl: dict[tuple[int, int, int], float],
    upper_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows_by_key = {
        (row["Z"], row["ion"], row["upper_level"]): row for row in upper_rows
    }
    members: dict[tuple[int, int, int], list[Target]] = defaultdict(list)
    for target in targets.values():
        members[target.sl_key].append(target)

    result: list[dict[str, Any]] = []
    for sl_key in sorted(members):
        targets_here = sorted(members[sl_key], key=lambda item: item.upper_level)
        upper_here = [rows_by_key[item.key] for item in targets_here]
        row: dict[str, Any] = {
            "Z": sl_key[0],
            "ion": sl_key[1],
            "runtime_sl_id": sl_key[2],
            "Y_s": y_by_sl[sl_key],
            "member_upper_count": len(targets_here),
            "selected_line_count": sum(item["selected_line_count"] for item in upper_here),
            "used_line_count": sum(item["used_line_count"] for item in upper_here),
            "excluded_line_count": sum(item["excluded_line_count"] for item in upper_here),
        }
        for component in COMPONENTS:
            value0 = finite_sum_or_none(
                [item[f"P_{component}_0"] for item in upper_here]
            )
            value1 = finite_sum_or_none(
                [item[f"P_{component}_1"] for item in upper_here]
            )
            row.update(
                {
                    f"used_line_count_{component}": sum(
                        item[f"used_line_count_{component}"] for item in upper_here
                    ),
                    f"P_{component}_0": value0,
                    f"P_{component}_1": value1,
                    f"dlog_P_{component}": safe_log_ratio(value1, value0),
                }
            )
        # The order names this pump coordinate X_u even after lump members
        # have been summed to one runtime-SL row.
        row["X_u"] = row["dlog_P_all"]
        result.append(row)
    return result


def midranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def pearson(first: list[float], second: list[float]) -> float | None:
    if len(first) != len(second) or len(first) < 2:
        return None
    mean_first = math.fsum(first) / len(first)
    mean_second = math.fsum(second) / len(second)
    centered_first = [value - mean_first for value in first]
    centered_second = [value - mean_second for value in second]
    denominator = math.sqrt(
        math.fsum(value * value for value in centered_first)
        * math.fsum(value * value for value in centered_second)
    )
    if denominator == 0.0 or not math.isfinite(denominator):
        return None
    value = math.fsum(
        x * y for x, y in zip(centered_first, centered_second)
    ) / denominator
    return value if math.isfinite(value) else None


def correlation_summary(sl_rows: list[dict[str, Any]]) -> dict[str, Any]:
    blocks: list[dict[str, Any]] = []
    pooled_x: list[float] = []
    pooled_y: list[float] = []
    for Z in sorted(TARGET_Z):
        rows = [
            row
            for row in sl_rows
            if row["Z"] == Z and row["X_u"] is not None and math.isfinite(row["Y_s"])
        ]
        xs = [row["X_u"] for row in rows]
        ys = [row["Y_s"] for row in rows]
        distinct_x = len(set(xs))
        distinct_y = len(set(ys))
        ranks_x = midranks(xs)
        ranks_y = midranks(ys)
        eligible = len(rows) >= 2 and distinct_x >= 2 and distinct_y >= 2
        rho = pearson(ranks_x, ranks_y) if eligible else None
        if eligible:
            mean_x = math.fsum(ranks_x) / len(ranks_x)
            mean_y = math.fsum(ranks_y) / len(ranks_y)
            pooled_x.extend(value - mean_x for value in ranks_x)
            pooled_y.extend(value - mean_y for value in ranks_y)
        blocks.append(
            {
                "Z": Z,
                "valid_sl_count": len(rows),
                "distinct_X_count": distinct_x,
                "distinct_Y_count": distinct_y,
                "X_tie_fraction": (
                    1.0 - distinct_x / len(rows) if rows else None
                ),
                "Y_tie_fraction": (
                    1.0 - distinct_y / len(rows) if rows else None
                ),
                "rho_midrank_spearman": rho,
                "included_in_centered_pool": eligible,
            }
        )
    included = sum(bool(item["included_in_centered_pool"]) for item in blocks)
    return {
        "blocks": blocks,
        "block_count": len(blocks),
        "blocks_in_centered_pool": included,
        "centered_pool_sl_count": len(pooled_x),
        "rho_block_centered_pooled_midrank_spearman": pearson(pooled_x, pooled_y),
    }


def quantile(sorted_values: list[float], fraction: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = fraction * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def distribution(values: Iterable[float | int | None]) -> dict[str, Any]:
    data = sorted(float(value) for value in values if value is not None and math.isfinite(value))
    if not data:
        return {
            "count": 0,
            "distinct_count": 0,
            "min": None,
            "q25": None,
            "median": None,
            "q75": None,
            "max": None,
        }
    return {
        "count": len(data),
        "distinct_count": len(set(data)),
        "min": data[0],
        "q25": quantile(data, 0.25),
        "median": median(data),
        "q75": quantile(data, 0.75),
        "max": data[-1],
    }


def row_distributions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {
        "selected_line_count": distribution(row["selected_line_count"] for row in rows),
        "used_line_count": distribution(row["used_line_count"] for row in rows),
        "excluded_line_count": distribution(row["excluded_line_count"] for row in rows),
    }
    for component in COMPONENTS:
        result[f"dlog_P_{component}"] = distribution(
            row[f"dlog_P_{component}"] for row in rows
        )
    return result


def spectral_metadata(grid: SpectralGrid) -> dict[str, Any]:
    return {
        "rows_total": grid.rows_total,
        "rows_shell0": grid.rows_shell0,
        "rows_other_shell": grid.rows_other_shell,
        "nonfinite_J_rows_shell0": grid.nonfinite_J_rows_shell0,
        "nonpositive_J_rows_shell0": grid.nonpositive_J_rows_shell0,
        "shell0_frequency_min_Hz": grid.bins[0].lo,
        "shell0_frequency_max_exclusive_Hz": max(item.hi for item in grid.bins),
    }


def csv_fields(identifier_fields: tuple[str, ...], coordinate_fields: tuple[str, ...]) -> list[str]:
    fields = list(identifier_fields)
    fields.extend(coordinate_fields)
    fields.extend(("selected_line_count", "used_line_count", "excluded_line_count"))
    for component in COMPONENTS:
        fields.extend(
            (
                f"used_line_count_{component}",
                f"P_{component}_0",
                f"P_{component}_1",
                f"dlog_P_{component}",
            )
        )
    return fields


UPPER_CSV_FIELDS = csv_fields(
    ("Z", "ion", "upper_level", "runtime_sl_id"), ("X_u",)
)
SL_CSV_FIELDS = csv_fields(("Z", "ion", "runtime_sl_id"), ("Y_s", "X_u"))


def analyze(
    run_root: Path, pairs_path: Path, sl_groups_path: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    line_list_path = run_root / "input/model/line_list.csv"
    levels_path = run_root / "input/model/levels.csv"
    spectral_paths = (
        run_root / "work/physics_comparison/physics_DET_iter0000.spectral.csv",
        run_root / "work/physics_comparison/physics_DET_iter0001.spectral.csv",
    )
    for path in (line_list_path, levels_path, *spectral_paths, pairs_path, sl_groups_path):
        if not path.is_file():
            raise PumpDriveError(f"required input is not a file: {path}")

    targets, y_by_sl, mapping_counts = load_targets(pairs_path, sl_groups_path)
    levels, level_counts = load_levels(levels_path)
    spectral0 = load_spectral(spectral_paths[0])
    spectral1 = load_spectral(spectral_paths[1])
    accumulators, line_counts = stream_line_list(
        line_list_path, targets, levels, spectral0, spectral1
    )
    upper_rows = build_upper_rows(targets, accumulators)
    sl_rows = build_sl_rows(targets, y_by_sl, upper_rows)

    summary = {
        "schema": SCHEMA,
        "readout_kind": "descriptive_pump_drive_proxy",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "execution": {
            "hostname": socket.gethostname(),
            "python": platform.python_version(),
        },
        "inputs": {
            "run_root": str(run_root.resolve()),
            "line_list": str(line_list_path.resolve()),
            "levels": str(levels_path.resolve()),
            "spectral_iter0": str(spectral_paths[0].resolve()),
            "spectral_iter1": str(spectral_paths[1].resolve()),
            "pairs": str(pairs_path.resolve()),
            "sl_groups": str(sl_groups_path.resolve()),
        },
        "method": {
            "temperature_K": TEMPERATURE_K,
            "k_B_eV_per_K": K_B_EV_K,
            "h_eV_s": H_EV_S,
            "weight": "f_lu * g_lower * exp(-E_lower_eV / (k_B_eV_per_K * temperature_K))",
            "spectral_bin_interval": "[nu_lo_Hz, nu_hi_Hz)",
            "photon_energy_split_eV": PHOTON_SPLIT_EV,
            "lower_level_classes": (
                "ground if E_lower_eV < 0.1; else metastable if metastable == 1; "
                "else other (mutually exclusive, ground precedence)"
            ),
            "runtime_sl_aggregation": (
                "sum P over all mapped member upper levels before taking log(P1/P0)"
            ),
            "rank_method": "exact-equality midranks",
            "distribution_quartiles": "linear interpolation at (n-1)*q",
        },
        "counts": {
            **mapping_counts,
            "levels": level_counts,
            "spectral_iter0": spectral_metadata(spectral0),
            "spectral_iter1": spectral_metadata(spectral1),
            "line_list": line_counts,
            "upper_rows": len(upper_rows),
            "runtime_sl_rows": len(sl_rows),
            "upper_levels_without_used_lines": sum(
                row["used_line_count"] == 0 for row in upper_rows
            ),
            "runtime_sl_without_X": sum(row["X_u"] is None for row in sl_rows),
        },
        "line_counts_per_upper": [
            {
                "Z": row["Z"],
                "ion": row["ion"],
                "upper_level": row["upper_level"],
                "runtime_sl_id": row["runtime_sl_id"],
                "selected": row["selected_line_count"],
                "used": row["used_line_count"],
                "excluded": row["excluded_line_count"],
            }
            for row in upper_rows
        ],
        "correlation": correlation_summary(sl_rows),
        "distributions": {
            "per_upper": row_distributions(upper_rows),
            "per_runtime_sl": row_distributions(sl_rows),
        },
    }
    # Ensure non-finite values can never leak into the JSON product.
    json.dumps(summary, allow_nan=False)
    return upper_rows, sl_rows, summary


def ensure_output_directory(path: Path, run_root: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(BASE_DIR)
    except ValueError as exc:
        raise PumpDriveError(
            f"output directory must be beneath audit directory {BASE_DIR}: {resolved}"
        ) from exc
    run_resolved = run_root.resolve()
    if resolved == run_resolved or run_resolved in resolved.parents:
        raise PumpDriveError("output directory must not be the run root or beneath it")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def atomic_write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, document: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def execute(
    run_root: Path,
    pairs_path: Path,
    sl_groups_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    output = ensure_output_directory(out_dir, run_root)
    upper_rows, sl_rows, summary = analyze(run_root, pairs_path, sl_groups_path)
    atomic_write_csv(output / "pump_drive_per_upper.csv", UPPER_CSV_FIELDS, upper_rows)
    atomic_write_csv(output / "pump_drive_sl.csv", SL_CSV_FIELDS, sl_rows)
    atomic_write_json(output / "pump_drive_summary.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream the complete line deck once and compute pump-drive readouts."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--pairs", type=Path, default=BASE_DIR / "out_331050/pairs.csv"
    )
    parser.add_argument(
        "--sl-groups", type=Path, default=BASE_DIR / "out_331050/sl_groups.csv"
    )
    parser.add_argument("--out-dir", type=Path, default=BASE_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = execute(args.run_root, args.pairs, args.sl_groups, args.out_dir)
    except (OSError, PumpDriveError, csv.Error) as exc:
        print(f"pump_drive: ERROR: {exc}", file=sys.stderr)
        return 2
    line_counts = summary["counts"]["line_list"]
    correlation = summary["correlation"]
    print("pump_drive: descriptive readout complete")
    print(
        "line_list rows={rows_total} selected={selected_target_lines} "
        "used={selected_lines_used} excluded={selected_lines_excluded}".format(
            rows_total=line_counts.get("rows_total", 0),
            selected_target_lines=line_counts.get("selected_target_lines", 0),
            selected_lines_used=line_counts.get("selected_lines_used", 0),
            selected_lines_excluded=line_counts.get("selected_lines_excluded", 0),
        )
    )
    print(
        f"upper={summary['counts']['upper_rows']} "
        f"runtime_sl={summary['counts']['runtime_sl_rows']}"
    )
    print(
        "rho_block_centered_pooled_midrank_spearman="
        f"{correlation['rho_block_centered_pooled_midrank_spearman']} "
        f"blocks={correlation['blocks_in_centered_pool']}/{correlation['block_count']}"
    )
    print("wrote pump_drive_per_upper.csv pump_drive_sl.csv pump_drive_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
