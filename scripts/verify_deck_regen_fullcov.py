#!/usr/bin/env python3
"""Read-only four-gate verifier for a regenerated CMFGEN deck.

It reports every failed gate and exits nonzero.  It never changes the deck.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import re
import shlex
import struct
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OLD_DEFAULT = ROOT / "data/tardis_reference_toy06_19p48d_sivcaiv"
NEW_DEFAULT = ROOT / "data/tardis_reference_toy06_19p48d_sivcaiv_fullcov"
RUN_DEFAULT = Path("/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4")
KEY_COLUMNS = ["atomic_number", "ion_number", "level_number_lower", "level_number_upper"]
VALUE_COLUMNS = ["wavelength_cm", "f_lu", "A_ul"]
ELEMENT_DIR_Z = {
    "SIL": 14, "SUL": 16, "CA": 20, "FE": 26, "COB": 27, "NICK": 28,
}
ROMAN_STAGE = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6,
}


def load_expand_module():
    path = ROOT / "scripts/expand_atomic_data_cmfgen.py"
    spec = importlib.util.spec_from_file_location("deck_verify_expand", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pack_key(z, ion, lower, upper) -> np.ndarray:
    z = np.asarray(z, dtype=np.uint64)
    ion = np.asarray(ion, dtype=np.uint64)
    lower = np.asarray(lower, dtype=np.uint64)
    upper = np.asarray(upper, dtype=np.uint64)
    return (z << 56) | (ion << 48) | (lower << 24) | upper


def deck_lines(path: Path, values: bool = False) -> pd.DataFrame:
    columns = KEY_COLUMNS + (VALUE_COLUMNS if values else [])
    frame = pd.read_csv(path / "line_list.csv", usecols=columns)
    frame["key"] = pack_key(*(frame[column] for column in KEY_COLUMNS))
    if frame["key"].duplicated().any():
        raise RuntimeError(f"duplicate physical-line identity in {path}")
    return frame


def cmf_reference(run: Path):
    expand = load_expand_module()
    model_spec = run / "MODEL_SPEC"
    atomic_links = run / "atomic_links.txt"
    if not model_spec.is_file() or not atomic_links.is_file():
        raise RuntimeError(f"missing CMFGEN reference files below {run}")

    nf_values: list[int] = []
    isf_pattern = re.compile(r"^\s*\d+\s*,\s*\d+\s*,\s*(\d+)\s+\[[^]]+_ISF\]")
    for line in model_spec.read_text(encoding="latin-1").splitlines():
        match = isf_pattern.match(line)
        if match:
            nf_values.append(int(match.group(1)))

    links: list[tuple[tuple[int, int], Path]] = []
    for line in atomic_links.read_text(encoding="latin-1").splitlines():
        if "_F_OSCDAT" not in line:
            continue
        fields = shlex.split(line)
        if len(fields) < 4 or fields[0] != "ln":
            raise RuntimeError(f"unexpected atomic_links row: {line}")
        osc_path = Path(fields[2])
        parts = osc_path.parts
        atomic_at = parts.index("atomic")
        element_dir, roman = parts[atomic_at + 1:atomic_at + 3]
        key = (ELEMENT_DIR_Z[element_dir], ROMAN_STAGE[roman])
        links.append((key, osc_path))
    if len(nf_values) != len(links):
        raise RuntimeError(f"MODEL_SPEC/atomic_links ion count differs: {len(nf_values)}/{len(links)}")

    by_ion: dict[tuple[int, int], np.ndarray] = {}
    for nf, ((z, stage), osc_path) in zip(nf_values, links, strict=True):
        ion0 = stage - 1
        osc = expand.parse_osc(osc_path)
        lower = np.minimum(osc.transitions["i"], osc.transitions["j"])
        upper = np.maximum(osc.transitions["i"], osc.transitions["j"])
        mask = (lower >= 1) & (upper <= nf) & (osc.transitions["lam_A"] != 0.0)
        keys = pack_key(
            np.full(mask.sum(), z), np.full(mask.sum(), ion0),
            lower[mask] - 1, upper[mask] - 1,
        )
        by_ion[(z, ion0)] = np.unique(keys)
    all_keys = np.unique(np.concatenate(list(by_ion.values())))
    return by_ion, all_keys


def sigma_counts(path: Path) -> tuple[int, int]:
    with (path / "cmfgen_sigma_bf.bin").open("rb") as stream:
        header = stream.read(32)
        if len(header) != 32:
            raise RuntimeError(f"bad sigma header in {path}")
        magic, version, n_levels, _n_freq = struct.unpack("<IIii", header[:16])
        if magic != 0x434D4644 or version != 1 or n_levels < 0:
            raise RuntimeError(f"bad sigma header in {path}")
        flags = np.fromfile(stream, dtype=np.uint8, count=n_levels)
    return int(n_levels), int(flags.sum())


def collision_count(path: Path) -> int:
    manifest = pd.read_csv(path / "coldata_cmfgen_manifest.csv")
    if "n_mapped" not in manifest:
        raise RuntimeError(f"n_mapped absent from collision manifest in {path}")
    return int(pd.to_numeric(manifest["n_mapped"], errors="raise").sum())


def audit_coiv_proxy() -> None:
    """Show whether the shipped Co IV source still consists of Fe III rows."""
    expand = load_expand_module()
    co_path = ROOT / "data/atomic/cmfgen/COB/IV/19apr23/col_data"
    fe_path = ROOT / "data/atomic/cmfgen/FE/III/19apr23/col_data"
    co = expand.parse_col(co_path)
    fe = expand.parse_col(fe_path)
    fe_rows = {omega.tobytes() for _, _, omega in fe.entries}
    row_matches = sum(omega.tobytes() in fe_rows for _, _, omega in co.entries)
    same_grid = np.array_equal(co.T_grid_kK.view(np.uint64), fe.T_grid_kK.view(np.uint64))
    print(
        "Co IV proxy audit — "
        f"Co rows found verbatim in Fe III={row_matches}/{len(co.entries)}, "
        f"temperature grid bit-identical={same_grid}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, default=OLD_DEFAULT)
    parser.add_argument("--new", type=Path, default=NEW_DEFAULT)
    parser.add_argument("--cmf-run", type=Path, default=RUN_DEFAULT)
    args = parser.parse_args()

    by_ion, reference = cmf_reference(args.cmf_run)
    old = deck_lines(args.old, values=True)
    new = deck_lines(args.new, values=True)
    old_keys = np.sort(old["key"].to_numpy(dtype=np.uint64))
    new_keys = np.sort(new["key"].to_numpy(dtype=np.uint64))
    failures: list[str] = []

    print("GATE 1 — per-ion CMFGEN coverage")
    for (z, ion0), required in sorted(by_ion.items()):
        present = np.intersect1d(required, new_keys, assume_unique=True).size
        ratio = present / required.size if required.size else float("nan")
        status = "PASS" if present == required.size else "FAIL"
        print(f"  Z={z:2d} ion={ion0 + 1:2d}: {present}/{required.size} = {ratio:.9f} {status}")
        if status == "FAIL":
            failures.append(f"gate1 Z={z} ion={ion0 + 1}: missing {required.size - present}")

    common = np.intersect1d(old_keys, reference, assume_unique=True)
    missing_old = np.setdiff1d(common, new_keys, assume_unique=True)
    gate2 = common.size == 881_085 and missing_old.size == 0
    print(f"GATE 2 — old CMF-common lines retained: common={common.size}, "
          f"missing={missing_old.size} "
          f"{'PASS' if gate2 else 'FAIL'}")
    if not gate2:
        failures.append(
            f"gate2: common={common.size}, missing {missing_old.size} old CMF-common lines"
        )

    old_index = old.set_index("key", verify_integrity=True).loc[common]
    new_indexed = new.set_index("key", verify_integrity=True)
    absent_common = np.setdiff1d(common, new_keys, assume_unique=True)
    mismatches = 0
    if absent_common.size == 0:
        new_index = new_indexed.loc[common]
        for column in VALUE_COLUMNS:
            left = old_index[column].to_numpy(dtype=np.float64).view(np.uint64)
            right = new_index[column].to_numpy(dtype=np.float64).view(np.uint64)
            count = int(np.count_nonzero(left != right))
            print(f"  gate3 {column}: bit mismatches={count}")
            mismatches += count
    else:
        mismatches = int(absent_common.size)
    gate3 = common.size == 881_085 and absent_common.size == 0 and mismatches == 0
    print(f"GATE 3 — common={common.size}, absent={absent_common.size}, "
          f"mismatches={mismatches} {'PASS' if gate3 else 'FAIL'}")
    if not gate3:
        failures.append(
            f"gate3: common={common.size}, absent={absent_common.size}, mismatches={mismatches}"
        )

    old_sigma_levels, old_sigma_present = sigma_counts(args.old)
    new_sigma_levels, new_sigma_present = sigma_counts(args.new)
    old_collision = collision_count(args.old)
    new_collision = collision_count(args.new)
    gate4 = (
        len(new) > len(old)
        and new_sigma_levels > old_sigma_levels
        and new_sigma_present > old_sigma_present
        and new_collision > old_collision
    )
    print(
        "GATE 4 — coupled growth: "
        f"lines {len(old)}->{len(new)}, sigma levels {old_sigma_levels}->{new_sigma_levels}, "
        f"sigma present {old_sigma_present}->{new_sigma_present}, "
        f"collision rows {old_collision}->{new_collision} "
        f"{'PASS' if gate4 else 'FAIL'}"
    )
    if not gate4:
        failures.append("gate4: line/sigma/collision data did not all grow")

    audit_coiv_proxy()

    if failures:
        print("VERDICT: FAIL — no adjustment was made", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    print("VERDICT: all four gates PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
