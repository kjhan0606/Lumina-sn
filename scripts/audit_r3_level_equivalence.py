#!/usr/bin/env python3
"""Read-only R3a/R3b level audit against the CMFGEN run's linked inputs.

This script deliberately keeps the thresholds used by the pre-R1 audit:

* energy mismatch: abs(delta E) > 1e-6 cm^-1;
* statistical weight: exact integer mismatch;
* sigma attachment: CMFGEN flag plus a positive finite value in its grid row;
* Upsilon attachment: the level occurs in a transition in a status=OK CMFGEN
  collision binary (including the dedicated Fe III binary when present).

It never writes a deck.  JSON/Markdown products are written only when the
corresponding command-line path is explicitly supplied.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import re
import struct
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OLD_DEFAULT = ROOT / "data/tardis_reference_toy06_19p48d_sivcaiv"
NEW_DEFAULT = ROOT / "data/tardis_reference_toy06_19p48d_sivcaiv_links"
RUN_DEFAULT = Path("/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4")
CM_TO_EV = 1.239841984e-4
ENERGY_TOL_CM = 1.0e-6
LOW_LEVEL_CM = 14_753.0
INT_RE = re.compile(r"^[+-]?\d+$")


@dataclass(frozen=True)
class DeckLevel:
    z: int
    ion0: int
    number: int
    energy_cm: float
    g: int
    configuration: str
    csv_path: Path
    csv_line: int
    global_index: int


@dataclass(frozen=True)
class SourceLevel:
    rank: int
    energy_cm: float
    g: int
    configuration: str
    osc_path: Path
    osc_line: int


def load_r1_verifier():
    path = ROOT / "scripts/verify_deck_r1_vintage.py"
    spec = importlib.util.spec_from_file_location("r3_r1_verifier", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def quantiles(values: list[float]) -> dict[str, float | None]:
    return {name: percentile(values, q) for name, q in
            (("p50", 50), ("p90", 90), ("p99", 99), ("max", 100))}


def normalize_configuration(value: str) -> str:
    return "".join(ch.casefold() for ch in value if ch.isascii() and ch.isalnum())


def source_line_numbers(path: Path, n_levels: int) -> list[int]:
    """Reproduce cmfgen_parser's sequential level-row recognition."""
    result: list[int] = []
    want = 1
    for lineno, line in enumerate(path.read_text(
            encoding="latin-1", errors="replace").splitlines(), 1):
        tokens = line.split()
        if len(tokens) < 4:
            continue
        level_id = None
        for token in tokens[3:]:
            if INT_RE.match(token):
                level_id = abs(int(token))
                break
        if level_id != want:
            continue
        try:
            float(tokens[1].replace("D", "E").replace("d", "e"))
            float(tokens[2].replace("D", "E").replace("d", "e"))
        except ValueError:
            continue
        result.append(lineno)
        want += 1
        if want > n_levels:
            break
    if len(result) != n_levels:
        raise RuntimeError(
            f"{path}: located {len(result)}/{n_levels} source level lines")
    return result


def resolve_source(path: Path) -> Path:
    if path.is_file():
        return path
    marker = "/atomic/"
    text = str(path)
    if marker in text:
        relative = text.split(marker, 1)[1]
        local = ROOT / "data/atomic/cmfgen" / relative
        if local.is_file():
            return local
    raise FileNotFoundError(path)


def read_source_levels(ref: Any) -> list[SourceLevel]:
    path = resolve_source(ref.osc_path)
    line_numbers = source_line_numbers(path, ref.osc.n_levels)
    result = []
    for rank in range(ref.nf):
        row = ref.osc.levels[rank]
        result.append(SourceLevel(
            rank=rank,
            energy_cm=float(row["E_cm"]),
            g=int(round(float(row["g"]))),
            configuration=str(row["config"]),
            osc_path=ref.osc_path,
            osc_line=line_numbers[rank],
        ))
    return result


def deck_osc_paths(deck: Path) -> dict[tuple[int, int], Path]:
    vintage = deck / "atomic_vintage_manifest.csv"
    collision = deck / "coldata_cmfgen_manifest.csv"
    source = vintage if vintage.is_file() else collision
    result: dict[tuple[int, int], Path] = {}
    with source.open(newline="") as stream:
        for row in csv.DictReader(stream):
            z = int(row.get("atomic_number", row.get("Z", "-1")))
            if "ion_number" in row:
                ion0 = int(row["ion_number"])
            else:
                ion0 = int(row["ion0"])
            osc = row.get("osc_path", row.get("osc", ""))
            if osc:
                result[(z, ion0)] = Path(osc)
    return result


def read_deck_levels(deck: Path) -> dict[tuple[int, int], list[DeckLevel]]:
    path = deck / "levels.csv"
    osc_paths = deck_osc_paths(deck)
    raw: list[tuple[int, dict[str, str]]] = []
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        for csv_line, row in enumerate(reader, 2):
            raw.append((csv_line, row))

    configuration_by_ion: dict[tuple[int, int], list[str]] = {}
    if "configuration" not in (raw[0][1] if raw else {}):
        verifier = load_r1_verifier()
        expand = verifier.load_expand_module()
        for key, osc_path in osc_paths.items():
            osc = expand.parse_osc(resolve_source(osc_path))
            configuration_by_ion[key] = [str(row["config"]) for row in osc.levels]

    result: dict[tuple[int, int], list[DeckLevel]] = {}
    for global_index, (csv_line, row) in enumerate(raw):
        z = int(row["atomic_number"])
        ion0 = int(row["ion_number"])
        number = int(row["level_number"])
        key = (z, ion0)
        configuration = row.get("configuration", "")
        if not configuration:
            configs = configuration_by_ion.get(key, [])
            if number < len(configs):
                configuration = configs[number]
        result.setdefault(key, []).append(DeckLevel(
            z=z, ion0=ion0, number=number,
            energy_cm=float(row["energy_eV"]) / CM_TO_EV,
            g=int(round(float(row["g"]))),
            configuration=configuration,
            csv_path=path, csv_line=csv_line, global_index=global_index,
        ))
    for key, rows in result.items():
        rows.sort(key=lambda item: item.number)
        if [row.number for row in rows] != list(range(len(rows))):
            raise RuntimeError(f"{path}: non-contiguous level numbers for {key}")
    return result


def build_mapping(source: list[SourceLevel], deck: list[DeckLevel]) -> tuple[list[int], int]:
    """Map source ranks using the pre-R1 audit's three-pass rules.

    Exact E+g is strongest, normalized configuration+g is second, and rank is
    the final supplement.  Already-used destinations make a rank supplement a
    collision; the mapping is still retained so E/g diagnostics cover every
    existing source rank, while the ion is reported as nonidentity.
    """
    mapping: list[int | None] = [None] * len(source)
    used: set[int] = set()

    # Prefer an exact identity candidate before searching elsewhere.  This is
    # immaterial to physical matching but makes duplicate E/g groups stable.
    for src in source:
        candidates = [dst.number for dst in deck
                      if abs(dst.energy_cm - src.energy_cm) <= ENERGY_TOL_CM
                      and dst.g == src.g]
        if src.rank in candidates and src.rank not in used:
            mapping[src.rank] = src.rank
            used.add(src.rank)
        elif len([item for item in candidates if item not in used]) == 1:
            choice = [item for item in candidates if item not in used][0]
            mapping[src.rank] = choice
            used.add(choice)

    cfg_index: dict[tuple[str, int], list[int]] = {}
    for dst in deck:
        cfg_index.setdefault((normalize_configuration(dst.configuration), dst.g), []).append(
            dst.number)
    for src in source:
        if mapping[src.rank] is not None:
            continue
        candidates = [item for item in cfg_index.get(
            (normalize_configuration(src.configuration), src.g), []) if item not in used]
        if src.rank in candidates:
            choice = src.rank
        elif len(candidates) == 1:
            choice = candidates[0]
        else:
            continue
        mapping[src.rank] = choice
        used.add(choice)

    collisions = 0
    for src in source:
        if mapping[src.rank] is not None or src.rank >= len(deck):
            continue
        mapping[src.rank] = src.rank
        if src.rank in used:
            collisions += 1
        else:
            used.add(src.rank)

    prefix = next((index for index, item in enumerate(mapping) if item is None),
                  len(mapping))
    if any(item is not None for item in mapping[prefix:]):
        raise RuntimeError("non-prefix mapping hole; refusing to shift source ranks")
    return [int(item) for item in mapping[:prefix]], collisions


def sigma_presence(deck: Path, levels: dict[tuple[int, int], list[DeckLevel]]) -> set[int]:
    path = deck / "cmfgen_sigma_bf.bin"
    with path.open("rb") as stream:
        header = stream.read(32)
        magic, version, n_levels, n_freq = struct.unpack("<IIii", header[:16])
        if magic != 0x434D4644 or version != 1:
            raise RuntimeError(f"{path}: bad magic/version")
        flags = np.frombuffer(stream.read(n_levels), dtype=np.int8).copy()
    padding = (8 - n_levels % 8) % 8
    grid_offset = 32 + n_levels + padding
    expected_size = grid_offset + 8 * n_levels * n_freq
    if path.stat().st_size != expected_size:
        raise RuntimeError(f"{path}: size {path.stat().st_size}/{expected_size}")
    grid = np.memmap(path, mode="r", dtype="<f8", offset=grid_offset,
                     shape=(n_levels, n_freq))
    actual_flags = np.zeros(n_levels, dtype=bool)
    for start in range(0, n_levels, 1024):
        block = np.asarray(grid[start:start + 1024])
        actual_flags[start:start + len(block)] = np.any(
            np.isfinite(block) & (block > 0.0), axis=1)
    disagreement = np.flatnonzero((flags != 0) != actual_flags)
    if disagreement.size:
        raise RuntimeError(
            f"{path}: flag/grid disagreement at global level {int(disagreement[0])}")
    present = set(np.flatnonzero(actual_flags).tolist())
    expected = sum(len(rows) for rows in levels.values())
    if n_levels != expected:
        raise RuntimeError(f"{path}: n_levels={n_levels}, levels.csv={expected}")
    return present


def collision_endpoints(path: Path) -> tuple[tuple[int, int], set[int]]:
    with path.open("rb") as stream:
        raw = stream.read(28)
        if len(raw) != 28:
            raise RuntimeError(f"{path}: short header")
        magic, version, z, ion0, n_trans, n_temp, _n_levels = struct.unpack(
            "<IIiiiii", raw)
        # Generic tables use IGC1; the dedicated Fe III table uses FEC3 but
        # has the same header and record layout.
        if magic not in (0x49474331, 0x46454333) or version != 1:
            raise RuntimeError(f"{path}: bad magic/version")
        stream.seek(8 * n_temp, 1)
        endpoints: set[int] = set()
        for _ in range(n_trans):
            lower, upper = struct.unpack("<ii", stream.read(8))
            endpoints.update((lower, upper))
            stream.seek(8 * n_temp, 1)
        if stream.read(1):
            raise RuntimeError(f"{path}: trailing bytes")
    return (z, ion0), endpoints


def upsilon_presence(deck: Path) -> dict[tuple[int, int], set[int]]:
    manifest = deck / "coldata_cmfgen_manifest.csv"
    result: dict[tuple[int, int], set[int]] = {}
    with manifest.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["status"] != "OK" or not row["out_bin"]:
                continue
            path = deck / row["out_bin"]
            key, endpoints = collision_endpoints(path)
            result.setdefault(key, set()).update(endpoints)
    dedicated = deck / "feiii_col_zhang.bin"
    if dedicated.is_file():
        key, endpoints = collision_endpoints(dedicated)
        result.setdefault(key, set()).update(endpoints)
    return result


def analyze_deck(deck: Path, refs: dict[tuple[int, int], Any]) -> dict[str, Any]:
    deck_levels = read_deck_levels(deck)
    sigma = sigma_presence(deck, deck_levels)
    upsilon = upsilon_presence(deck)
    ions: list[dict[str, Any]] = []
    g_mismatches: list[dict[str, Any]] = []
    energy_rows: list[dict[str, Any]] = []
    nonidentity = 0
    present_total = sigma_absent = upsilon_absent = 0
    present_keys: list[list[int]] = []
    sigma_absent_keys: list[list[int]] = []
    upsilon_absent_keys: list[list[int]] = []

    for key, ref in sorted(refs.items()):
        z, stage = key
        ion0 = stage - 1
        source = read_source_levels(ref)
        actual = deck_levels.get((z, ion0), [])
        mapping, collisions = build_mapping(source, actual)
        if len(mapping) != min(len(source), len(actual)):
            raise RuntimeError(
                f"{deck}: incomplete mapping for Z={z} stage={stage}: "
                f"{len(mapping)}/{min(len(source), len(actual))}")
        # A missing suffix is structural coverage, not a mapping
        # nonidentity.  This preserves the pre-R1 convention: an existing
        # identity prefix counts as identity, while a wholly absent ion is not
        # entered in the mapping denominator.
        identity = (collisions == 0 and mapping == list(range(len(mapping))))
        if not identity and actual:
            nonidentity += 1

        ion_g = ion_e = ion_sigma_absent = ion_upsilon_absent = 0
        # Existing pre-R1 decks may contain only a prefix, or no rows at all,
        # for an active CMFGEN ion.  Metrics here intentionally cover the
        # existing/mapped subset; structural absence was R1's separate scope.
        for src, destination in zip(source, mapping):
            dst = actual[destination]
            present_total += 1
            present_keys.append([z, ion0, src.rank])
            delta = dst.energy_cm - src.energy_cm
            absolute = abs(delta)
            relative = absolute / max(abs(src.energy_cm), 1.0)
            energy_rows.append({
                "z": z, "ion0": ion0, "stage": stage, "rank": src.rank,
                "deck_level": dst.number, "cmfgen_energy_cm": src.energy_cm,
                "lumina_energy_cm": dst.energy_cm, "delta_energy_cm": delta,
                "absolute_delta_cm": absolute, "relative_delta": relative,
                "mismatch": absolute > ENERGY_TOL_CM,
            })
            if absolute > ENERGY_TOL_CM:
                ion_e += 1
            if src.g != dst.g:
                ion_g += 1
                g_mismatches.append({
                    "z": z, "ion0": ion0, "stage": stage, "rank": src.rank,
                    "deck_level": dst.number, "energy_cm": src.energy_cm,
                    "cmfgen_g": src.g, "lumina_g": dst.g,
                    "cmfgen_source": str(src.osc_path),
                    "cmfgen_line": src.osc_line,
                    "lumina_source": str(dst.csv_path),
                    "lumina_line": dst.csv_line,
                })
            if dst.global_index not in sigma:
                sigma_absent += 1
                ion_sigma_absent += 1
                sigma_absent_keys.append([z, ion0, src.rank])
            if dst.number not in upsilon.get((z, ion0), set()):
                upsilon_absent += 1
                ion_upsilon_absent += 1
                upsilon_absent_keys.append([z, ion0, src.rank])
        ions.append({
            "z": z, "ion0": ion0, "stage": stage,
            "active": len(source), "present": len(mapping),
            "mapping_identity": identity, "mapping_collisions": collisions,
            "g_mismatch": ion_g, "energy_mismatch": ion_e,
            "sigma_absent": ion_sigma_absent,
            "upsilon_absent": ion_upsilon_absent,
        })

    mismatch_energy = [row for row in energy_rows if row["mismatch"]]
    absolute_all = [row["absolute_delta_cm"] for row in energy_rows]
    relative_all = [row["relative_delta"] for row in energy_rows]
    absolute_bad = [row["absolute_delta_cm"] for row in mismatch_energy]
    relative_bad = [row["relative_delta"] for row in mismatch_energy]
    g_keys = {(row["z"], row["ion0"], row["rank"]) for row in g_mismatches}
    e_keys = {(row["z"], row["ion0"], row["rank"]) for row in mismatch_energy}
    low_g = sum(row["energy_cm"] <= LOW_LEVEL_CM for row in g_mismatches)
    return {
        "deck": str(deck),
        "criteria": {
            "energy_tolerance_cm": ENERGY_TOL_CM,
            "low_level_cutoff_cm": LOW_LEVEL_CM,
            "g": "exact integer",
            "sigma": "flag != 0 and positive finite grid value",
            "upsilon": "endpoint in status=OK collision binary or Fe III dedicated binary",
        },
        "summary": {
            "active_levels": sum(ref.nf for ref in refs.values()),
            "present_levels": present_total,
            "mapping_nonidentity_ions": nonidentity,
            "g_mismatch_levels": len(g_mismatches),
            "energy_mismatch_levels": len(mismatch_energy),
            "sigma_absent_present_levels": sigma_absent,
            "upsilon_absent_present_levels": upsilon_absent,
            "g_low_levels": low_g,
            "g_high_levels": len(g_mismatches) - low_g,
            "g_energy_absolute_cm_quantiles": quantiles(
                [row["energy_cm"] for row in g_mismatches]),
            "energy_all_absolute_cm_quantiles": quantiles(absolute_all),
            "energy_all_relative_quantiles": quantiles(relative_all),
            "energy_mismatch_absolute_cm_quantiles": quantiles(absolute_bad),
            "energy_mismatch_relative_quantiles": quantiles(relative_bad),
            "energy_ge_1000_cm_levels": sum(
                row["absolute_delta_cm"] >= 1000.0 for row in mismatch_energy),
            "g_energy_same_level_overlap": len(g_keys & e_keys),
        },
        "ions": ions,
        "g_mismatches": g_mismatches,
        "energy_mismatches": mismatch_energy,
        "present_keys": present_keys,
        "sigma_absent_keys": sigma_absent_keys,
        "upsilon_absent_keys": upsilon_absent_keys,
    }


def issue_transition(old: dict[str, Any], new: dict[str, Any], field: str) -> dict[str, int]:
    old_issues = {tuple(item) for item in old[field]}
    new_issues = {tuple(item) for item in new[field]}
    old_present = {tuple(item) for item in old["present_keys"]}
    new_present = {tuple(item) for item in new["present_keys"]}
    common = old_present & new_present
    return {
        "old_issues": len(old_issues),
        "old_resolved": len((old_issues - new_issues) & common),
        "old_persisting": len(old_issues & new_issues & common),
        "common_new_regressions": len((new_issues - old_issues) & common),
        "newly_present_issues": len(new_issues - old_present),
        "new_issues": len(new_issues),
    }


def format_number(value: float | None) -> str:
    return "—" if value is None else f"{value:.9g}"


def markdown_report(old: dict[str, Any], new: dict[str, Any]) -> str:
    lines = [
        "# R3 level-equivalence audit (machine-generated appendix)", "",
        "The same frozen criteria were applied to both decks.", "",
        "| metric | pre-R1 `_sivcaiv` | post-R1 `_sivcaiv_links` |",
        "|---|---:|---:|",
    ]
    keys = [
        ("g mismatch", "g_mismatch_levels"),
        ("E mismatch", "energy_mismatch_levels"),
        ("mapping nonidentity ions", "mapping_nonidentity_ions"),
        ("Upsilon absent among present", "upsilon_absent_present_levels"),
        ("sigma absent among present", "sigma_absent_present_levels"),
    ]
    for label, key in keys:
        lines.append(f"| {label} | {old['summary'][key]:,} | {new['summary'][key]:,} |")
    lines.extend(["", "## Post-R1 g mismatches (complete list)", ""])
    if not new["g_mismatches"]:
        lines.append("None (0 levels).")
    else:
        lines.extend([
            "| ion | rank | E (cm^-1) | CMFGEN g | Lumina g | sources |",
            "|---|---:|---:|---:|---:|---|",
        ])
        for row in new["g_mismatches"]:
            lines.append(
                f"| Z={row['z']} ion0={row['ion0']} | {row['rank']} | "
                f"{row['energy_cm']:.10g} | {row['cmfgen_g']} | {row['lumina_g']} | "
                f"`{row['cmfgen_source']}:{row['cmfgen_line']}`; "
                f"`{row['lumina_source']}:{row['lumina_line']}` |")
    lines.extend(["", "## Post-R1 E quantiles (all mapped levels)", "",
                  "| statistic | absolute (cm^-1) | relative |",
                  "|---|---:|---:|"])
    aq = new["summary"]["energy_all_absolute_cm_quantiles"]
    rq = new["summary"]["energy_all_relative_quantiles"]
    for key in ("p50", "p90", "p99", "max"):
        lines.append(f"| {key} | {format_number(aq[key])} | {format_number(rq[key])} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, default=OLD_DEFAULT)
    parser.add_argument("--new", type=Path, default=NEW_DEFAULT)
    parser.add_argument("--cmf-run", type=Path, default=RUN_DEFAULT)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    for path in (args.old, args.new, args.cmf_run):
        if not path.exists():
            print(f"ERROR: missing input {path}", file=sys.stderr)
            return 2

    try:
        verifier = load_r1_verifier()
        _expand, _links, refs = verifier.cmf_references(args.cmf_run)
        old = analyze_deck(args.old, refs)
        new = analyze_deck(args.new, refs)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    result = {
        "comparison": {
            "sigma": issue_transition(old, new, "sigma_absent_keys"),
            "upsilon": issue_transition(old, new, "upsilon_absent_keys"),
        },
        "old": old,
        "new": new,
    }
    print(markdown_report(old, new), end="")
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n")
    if args.markdown:
        args.markdown.write_text(markdown_report(old, new))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
