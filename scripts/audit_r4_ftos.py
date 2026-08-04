#!/usr/bin/env python3
"""R4 canonical f_to_s census and pre-implementation GPU resource gate."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import re
import shlex

import numpy as np

from cmfgen_parser import parse_f_to_s, parse_osc


ROOT = Path(__file__).resolve().parents[1]
LINKS_DEFAULT = Path("/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/atomic_links.txt")
DECK_DEFAULT = ROOT / "data/tardis_reference_toy06_19p48d_sivcaiv_links"
SOURCE = ROOT / "src/lumina_cuda.cu"
PLASMA_SOURCE = ROOT / "src/lumina_plasma.c"

BASELINE_GPU_BYTES = 110_569_158_168
H200_BYTES = 150_754_820_096
SHELLS = 50

DIR_TO_Z = {"SIL": 14, "SUL": 16, "CA": 20, "FE": 26, "COB": 27, "NICK": 28}
SYMBOL = {14: "Si", 16: "S", 20: "Ca", 26: "Fe", 27: "Co", 28: "Ni"}
ROMAN = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII"]

# src/lumina_plasma.c:nlte_get_pairs(), default-OFF 16-pair layout.
BASE_PAIRS = [
    (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13),
    (14, 15), (16, 17), (18, 19), (20, 21), (22, 23), (24, 25),
    (26, 27), (28, 29), (29, 30),
]


@dataclass(frozen=True)
class Entry:
    key: tuple[int, int]
    ion: str
    path: Path
    vintage: str
    ftos: object
    osc_path: Path


def link_kind(target: str) -> str | None:
    if target.endswith("_F_TO_S"):
        return "ftos"
    if target.endswith("_F_OSCDAT"):
        return "osc"
    return None


def path_key(path: Path) -> tuple[tuple[int, int], str]:
    parts = path.parts
    try:
        i = parts.index("atomic")
        element, roman, vintage = parts[i + 1:i + 4]
        return (DIR_TO_Z[element], ROMAN.index(roman)), vintage
    except (ValueError, KeyError) as exc:
        raise RuntimeError(f"unrecognised CMFGEN atomic path {path}") from exc


def read_entries(path: Path) -> list[Entry]:
    sources: dict[tuple[int, int], dict[str, Path]] = {}
    order: list[tuple[int, int]] = []
    for lineno, line in enumerate(path.read_text(encoding="latin-1").splitlines(), 1):
        fields = shlex.split(line, comments=True)
        if not fields or fields[0] != "ln":
            continue
        operands = [field for field in fields[1:] if not field.startswith("-")]
        if len(operands) != 2:
            raise RuntimeError(f"{path}:{lineno}: malformed ln command")
        source, target = Path(operands[0]), operands[1]
        kind = link_kind(target)
        if kind is None:
            continue
        key, _vintage = path_key(source)
        slot = sources.setdefault(key, {})
        if kind in slot:
            raise RuntimeError(f"{path}:{lineno}: duplicate {kind} for {key}")
        slot[kind] = source
        if kind == "ftos":
            order.append(key)

    if len(order) != len(set(order)):
        raise RuntimeError("duplicate f_to_s ion links")
    entries: list[Entry] = []
    for key in order:
        if set(sources[key]) != {"osc", "ftos"}:
            raise RuntimeError(f"incomplete osc/f_to_s links for {key}")
        ftos_path = sources[key]["ftos"]
        osc_path = sources[key]["osc"]
        ftos = parse_f_to_s(ftos_path)
        osc = parse_osc(osc_path)
        if osc.n_levels != ftos.n_levels:
            raise RuntimeError(
                f"{key}: osc/f_to_s declared FL mismatch "
                f"{osc.n_levels}/{ftos.n_levels}"
            )
        vintage = path_key(ftos_path)[1]
        z, stage = key
        entries.append(Entry(key, f"{SYMBOL[z]} {ROMAN[stage]}", ftos_path,
                             vintage, ftos, osc_path))
    return entries


def read_level_counts(deck: Path) -> tuple[dict[tuple[int, int], int],
                                            dict[tuple[int, int], int]]:
    full: dict[tuple[int, int], int] = {}
    supers: dict[tuple[int, int], set[int]] = {}
    with (deck / "levels.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            key = (int(row["atomic_number"]), int(row["ion_number"]))
            full[key] = full.get(key, 0) + 1
            supers.setdefault(key, set()).add(int(row["super_level"]))
    return full, {key: len(value) for key, value in supers.items()}


def source_default_targets() -> list[tuple[int, int]]:
    text = PLASMA_SOURCE.read_text()
    values = []
    for name in ("NLTE_TARGET_Z", "NLTE_TARGET_ION"):
        match = re.search(
            rf"static const int {name}\[\]\s*=\s*\{{(.*?)\}};", text, re.S
        )
        if not match:
            raise RuntimeError(f"cannot parse {name} from {PLASMA_SOURCE}")
        values.append([int(item) for item in re.findall(r"\d+", match.group(1))])
    if len(values[0]) != len(values[1]) or len(values[0]) != 31:
        raise RuntimeError("unexpected default NLTE target layout")
    return list(zip(values[0], values[1], strict=True))


def count_deck_lines(deck: Path) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    with (deck / "line_list.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            key = (int(row["atomic_number"]), int(row["ion_number"]))
            counts[key] = counts.get(key, 0) + 1
    return counts


def planned_line_count(entry: Entry) -> int:
    osc = parse_osc(entry.osc_path)
    lo = np.minimum(osc.transitions["i"], osc.transitions["j"])
    hi = np.maximum(osc.transitions["i"], osc.transitions["j"])
    return int(np.count_nonzero(
        (lo >= 1) & (hi <= entry.ftos.n_levels) &
        (osc.transitions["lam_A"] != 0.0)
    ))


def solver_bytes(max_n: int, shells: int) -> dict[str, int]:
    # Exact device allocations in src/lumina_cuda.cu:599-613.
    return {
        "d_matrices": shells * max_n * max_n * 8,
        "d_rhs": shells * max_n * 8,
        "d_Aarray+d_Barray": 2 * shells * 8,
        "d_pivot": shells * max_n * 4,
        "d_info": shells * 4,
    }


def fmt_mib(value: int) -> str:
    return f"{value / 2**20:,.3f} MiB"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--links", type=Path, default=LINKS_DEFAULT)
    parser.add_argument("--deck", type=Path, default=DECK_DEFAULT)
    args = parser.parse_args()

    entries = read_entries(args.links)
    if len(entries) != 27:
        raise RuntimeError(f"expected all 27 linked ions, got {len(entries)}")
    formats: dict[str, int] = {}
    for entry in entries:
        formats[entry.ftos.format_name] = formats.get(entry.ftos.format_name, 0) + 1

    print("R4 STEP 1 — COMPLETE f_to_s FORMAT CENSUS")
    print("| ion | canonical path | vintage | format | declared FL | declared SL |")
    print("|---|---|---:|---|---:|---:|")
    for entry in entries:
        print(f"| {entry.ion} | {entry.path} | {entry.vintage} | "
              f"{entry.ftos.format_name} | {entry.ftos.n_levels} | "
              f"{entry.ftos.n_super} |")
    print(f"FORMAT CLASSES ({len(formats)}): " +
          ", ".join(f"{key}={value}" for key, value in sorted(formats.items())))
    for key in sorted(formats):
        basis = next(entry.ftos.format_basis for entry in entries
                     if entry.ftos.format_name == key)
        print(f"FORMAT BASIS {key}: {basis}")

    current_full, current_super = read_level_counts(args.deck)
    targets = source_default_targets()
    planned_full = dict(current_full)
    planned_super = dict(current_super)
    by_key = {entry.key: entry for entry in entries}
    for entry in entries:
        z, stage = entry.key
        deck_key = (z, stage - 1)
        planned_full[deck_key] = entry.ftos.n_levels
        planned_super[deck_key] = entry.ftos.n_super

    pair_rows = []
    for pair_id, (lo, hi) in enumerate(BASE_PAIRS):
        low_key, high_key = targets[lo], targets[hi]
        logical = planned_super.get(low_key, 0) + planned_super.get(high_key, 0)
        allocated = planned_full.get(low_key, 0) + planned_full.get(high_key, 0)
        pair_rows.append((pair_id, low_key, high_key, logical, allocated))
    max_logical = max(row[3] for row in pair_rows)
    max_allocated = max(row[4] for row in pair_rows)

    source = SOURCE.read_text()
    allocation_is_full = (
        "nlte.nlte_ion_level_offset[hi + 1]" in source and
        "cuda_nlte_solver_init(&nlte_solver, max_N, geo.n_shells);" in source
    )
    if not allocation_is_full:
        raise RuntimeError("NLTE max_N allocation source contract changed")

    print("R4 STEP 3 — MEASURED SL COUNTS AND GPU GATE")
    for entry in entries:
        print(f"  {entry.ion:7s}: {entry.ftos.n_levels:4d} FL -> "
              f"{entry.ftos.n_super:3d} SL")
    print("DEFAULT NLTE PAIRS (logical SL N / current allocated FL N)")
    for pair_id, low_key, high_key, logical, allocated in pair_rows:
        print(f"  pair {pair_id:2d} {low_key}+{high_key}: "
              f"logical={logical:4d} allocated={allocated:4d}")

    logical_bytes = solver_bytes(max_logical, SHELLS)
    allocated_bytes = solver_bytes(max_allocated, SHELLS)
    print(f"LOGICAL max_N={max_logical}; d_matrices={logical_bytes['d_matrices']} "
          f"({fmt_mib(logical_bytes['d_matrices'])}); "
          f"solver device total={sum(logical_bytes.values())} "
          f"({fmt_mib(sum(logical_bytes.values()))})")
    print(f"CURRENT SOURCE ALLOCATION max_N={max_allocated}; "
          f"d_matrices={allocated_bytes['d_matrices']} "
          f"({fmt_mib(allocated_bytes['d_matrices'])}); "
          f"solver device total={sum(allocated_bytes.values())} "
          f"({fmt_mib(sum(allocated_bytes.values()))})")
    print("ARRAY CONTRACT: src/lumina_cuda.cu:605 d_matrices = "
          "batch_size * max_N * max_N * sizeof(double); current max_N "
          "precompute uses nlte_ion_level_offset (FL), while each solve uses "
          "nlte_ion_super_offset (SL) at src/lumina_cuda.cu:1147-1152.")

    current_lines = count_deck_lines(args.deck)
    delta_levels = 0
    delta_lines = 0
    for entry in entries:
        z, stage = entry.key
        deck_key = (z, stage - 1)
        delta_levels += entry.ftos.n_levels - current_full.get(deck_key, 0)
        delta_lines += planned_line_count(entry) - current_lines.get(deck_key, 0)
    if delta_levels < 0 or delta_lines < 0:
        raise RuntimeError(
            f"planned linked deck unexpectedly shrinks: dN={delta_levels} dL={delta_lines}"
        )
    delta_edges = 3 * delta_lines
    identified_delta = (
        16 * delta_edges * SHELLS + 12 * delta_edges +
        8 * delta_lines * SHELLS + 20 * delta_lines * SHELLS +
        20 * delta_lines + 16 * delta_levels * SHELLS +
        4 * delta_levels + 8 * delta_levels * SHELLS +
        4 * delta_levels + 4224 * delta_levels
    )
    projected = BASELINE_GPU_BYTES + identified_delta
    margin = H200_BYTES - projected
    print(f"PLANNED vs _links: delta levels={delta_levels}, "
          f"delta physical lines={delta_lines}, delta macro edges={delta_edges}")
    print(f"IDENTIFIED GPU delta={identified_delta} ({fmt_mib(identified_delta)}); "
          f"matrix allocation delta=0 because max allocated FL pair stays "
          f"{max_allocated}")
    print(f"CONSERVATIVE TOTAL={projected} ({fmt_mib(projected)}) / "
          f"H200={H200_BYTES} ({fmt_mib(H200_BYTES)}); "
          f"margin={margin} ({fmt_mib(margin)})")
    if margin < 0:
        raise SystemExit("R4 RESOURCE GATE FAIL — H200 exceeded")
    print("R4 RESOURCE GATE PASS — H200 FIT")


if __name__ == "__main__":
    main()
