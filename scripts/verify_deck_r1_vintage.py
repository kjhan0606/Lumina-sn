#!/usr/bin/env python3
"""Read-only verifier for the R1 CMFGEN-link-pinned atomic deck.

The four gates are deliberately observational: failures are reported and make
the process exit nonzero; this program never edits or repairs a deck.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import re
import shlex
import struct
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_DEFAULT = ROOT / "data/tardis_reference_toy06_19p48d_sivcaiv"
FULLCOV_DEFAULT = ROOT / "data/tardis_reference_toy06_19p48d_sivcaiv_fullcov"
NEW_DEFAULT = ROOT / "data/tardis_reference_toy06_19p48d_sivcaiv_links"
RUN_DEFAULT = Path("/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4")

KEY_COLUMNS = [
    "atomic_number", "ion_number", "level_number_lower", "level_number_upper",
]
VALUE_COLUMNS = ["f_lu", "A_ul", "wavelength_cm"]
CM_TO_EV = 1.239841984e-4


@dataclass(frozen=True)
class IonReference:
    z: int
    stage: int
    nf: int
    osc_path: Path
    osc: object
    line_keys: np.ndarray
    linked_vintage: str
    latest_vintage: str

    @property
    def ion0(self) -> int:
        return self.stage - 1


def load_expand_module():
    path = ROOT / "scripts/expand_atomic_data_cmfgen.py"
    spec = importlib.util.spec_from_file_location("r1_verify_expand", path)
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


def ordered_osc_link_keys(expand, atomic_links: Path) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for lineno, line in enumerate(
            atomic_links.read_text(encoding="latin-1").splitlines(), 1):
        if "_F_OSCDAT" not in line:
            continue
        fields = shlex.split(line, comments=True)
        operands = [field for field in fields[1:] if not field.startswith("-")]
        if not fields or fields[0] != "ln" or len(operands) != 2:
            raise RuntimeError(f"{atomic_links}:{lineno}: malformed osc link")
        result.append(expand._atomic_path_identity(Path(operands[0]))[0])
    if len(result) != len(set(result)):
        raise RuntimeError(f"duplicate osc ion in {atomic_links}")
    return result


def cmf_references(run: Path):
    expand = load_expand_module()
    model_spec = run / "MODEL_SPEC"
    atomic_links = run / "atomic_links.txt"
    if not model_spec.is_file() or not atomic_links.is_file():
        raise RuntimeError(f"missing MODEL_SPEC/atomic_links.txt below {run}")

    links = expand.load_cmfgen_links(atomic_links)
    ordered_keys = ordered_osc_link_keys(expand, atomic_links)
    nf_pattern = re.compile(r"^\s*\d+\s*,\s*\d+\s*,\s*(\d+)\s+\[[^]]+_ISF\]")
    nf_values = [int(match.group(1)) for line in
                 model_spec.read_text(encoding="latin-1").splitlines()
                 if (match := nf_pattern.match(line))]
    if len(nf_values) != len(ordered_keys):
        raise RuntimeError(
            f"MODEL_SPEC/atomic_links ion count differs: "
            f"{len(nf_values)}/{len(ordered_keys)}")
    if set(ordered_keys) != set(links):
        raise RuntimeError("osc-link ion set differs from complete link-map ion set")

    refs: dict[tuple[int, int], IonReference] = {}
    for nf, key in zip(nf_values, ordered_keys, strict=True):
        z, stage = key
        osc_path = links[key]["osc"]
        osc = expand.parse_osc(osc_path)
        if osc.n_levels < nf:
            raise RuntimeError(
                f"linked osc {osc_path} has {osc.n_levels} levels below NF={nf}")
        lower = np.minimum(osc.transitions["i"], osc.transitions["j"])
        upper = np.maximum(osc.transitions["i"], osc.transitions["j"])
        mask = ((lower >= 1) & (upper <= nf) &
                (osc.transitions["lam_A"] != 0.0))
        keys = np.unique(pack_key(
            np.full(mask.sum(), z), np.full(mask.sum(), stage - 1),
            lower[mask] - 1, upper[mask] - 1,
        ))
        ion_dir = (expand.CMFGEN_ROOT / expand.CMFGEN_DIRS[z] /
                   expand.ROMAN[stage])
        latest = expand._pick_latest(ion_dir)
        if latest is None:
            raise RuntimeError(f"no local latest vintage below {ion_dir}")
        linked_vintage = expand._atomic_path_identity(osc_path)[1]
        refs[key] = IonReference(
            z, stage, nf, osc_path, osc, keys, linked_vintage, latest.name)
    return expand, links, refs


def read_vintage_manifest(path: Path) -> dict[tuple[int, int], dict[str, str]]:
    manifest_path = path / "atomic_vintage_manifest.csv"
    required = {
        "atomic_number", "ion_stage", "selection_source", "latest_vintage",
        "osc_vintage", "osc_path", "f_to_s_path", "phot_path", "col_path",
    }
    result: dict[tuple[int, int], dict[str, str]] = {}
    with manifest_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise RuntimeError(f"{manifest_path} missing columns {sorted(missing)}")
        for row in reader:
            key = (int(row["atomic_number"]), int(row["ion_stage"]))
            if key in result:
                raise RuntimeError(f"duplicate manifest ion {key} in {manifest_path}")
            result[key] = row
    return result


def read_levels(path: Path, require_configuration: bool = True):
    rows: dict[tuple[int, int], list[dict[str, str]]] = {}
    with (path / "levels.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "atomic_number", "ion_number", "level_number", "energy_eV", "g",
        }
        if require_configuration:
            required.add("configuration")
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise RuntimeError(f"{path}/levels.csv missing columns {sorted(missing)}")
        for row in reader:
            key = (int(row["atomic_number"]), int(row["ion_number"]) + 1)
            rows.setdefault(key, []).append(row)
    for key, ion_rows in rows.items():
        ion_rows.sort(key=lambda row: int(row["level_number"]))
        got = [int(row["level_number"]) for row in ion_rows]
        if got != list(range(len(got))):
            raise RuntimeError(f"non-contiguous level_number for {key} in {path}")
    return rows


def gate1(refs, new_lines: pd.DataFrame, failures: list[str]) -> None:
    new_keys = np.sort(new_lines["key"].to_numpy(dtype=np.uint64))
    print("GATE 1 — per-ion CMFGEN active-line coverage")
    for key, ref in sorted(refs.items()):
        present = np.intersect1d(ref.line_keys, new_keys, assume_unique=True).size
        ratio = present / ref.line_keys.size if ref.line_keys.size else float("nan")
        passed = ref.line_keys.size > 0 and present == ref.line_keys.size
        print(f"  Z={ref.z:2d} ion={ref.stage:2d}: {present}/{ref.line_keys.size} "
              f"= {ratio:.9f} {'PASS' if passed else 'FAIL'}")
        if not passed:
            failures.append(
                f"gate1 Z={ref.z} ion={ref.stage}: "
                f"{ref.line_keys.size - present} active lines absent")


def gate2(expand, links, refs, new: Path, failures: list[str]) -> None:
    manifest = read_vintage_manifest(new)
    levels = read_levels(new)
    bad_ions: list[tuple[tuple[int, int], str]] = []
    print("GATE 2 — CMFGEN-rank to deck-level identity")
    for key, ref in sorted(refs.items()):
        reasons: list[str] = []
        row = manifest.get(key)
        if row is None:
            reasons.append("vintage manifest row absent")
        else:
            if row["selection_source"] != "links":
                reasons.append(f"selection_source={row['selection_source']!r}")
            if row["latest_vintage"] != ref.latest_vintage:
                reasons.append("latest_vintage mismatch")
            for kind in expand._LINK_KINDS:
                actual = Path(row[f"{kind}_path"])
                if actual != links[key][kind]:
                    reasons.append(f"{kind}_path is not atomic_links source")

        ion_rows = levels.get(key, [])
        if len(ion_rows) < ref.nf:
            reasons.append(f"deck levels={len(ion_rows)} below NF={ref.nf}")
        else:
            for rank in range(ref.nf):
                actual = ion_rows[rank]
                source = ref.osc.levels[rank]
                expected_energy = f"{float(source['E_cm']) * CM_TO_EV:.10f}"
                if int(actual["level_number"]) != rank:
                    reasons.append(f"rank {rank}: level_number differs")
                    break
                if actual["energy_eV"] != expected_energy:
                    reasons.append(f"rank {rank}: energy differs")
                    break
                if float(actual["g"]) != float(source["g"]):
                    reasons.append(f"rank {rank}: g differs")
                    break
                if actual["configuration"] != str(source["config"]):
                    reasons.append(f"rank {rank}: configuration differs")
                    break
        if reasons:
            bad_ions.append((key, "; ".join(reasons[:4])))
            print(f"  Z={ref.z:2d} ion={ref.stage:2d}: NONIDENTITY — "
                  f"{bad_ions[-1][1]}")
        else:
            print(f"  Z={ref.z:2d} ion={ref.stage:2d}: identity "
                  f"({ref.nf} active levels) PASS")
    print(f"  mapping nonidentity ions = {len(bad_ions)} "
          f"{'PASS' if not bad_ions else 'FAIL'}")
    if bad_ions:
        failures.append(f"gate2: mapping nonidentity ions={len(bad_ions)}")


def gate3(refs, fullcov: Path, new: Path, failures: list[str]) -> None:
    old_lines = deck_lines(fullcov, values=True)
    new_lines = deck_lines(new, values=True)
    stable = [ref for ref in refs.values()
              if ref.linked_vintage == ref.latest_vintage]
    total_mismatches = 0
    print("GATE 3 — same-vintage ions bit-identical to fullcov")
    for ref in sorted(stable, key=lambda item: (item.z, item.stage)):
        old = old_lines[(old_lines.atomic_number == ref.z) &
                        (old_lines.ion_number == ref.ion0)].set_index(
                            "key", verify_integrity=True).sort_index()
        fresh = new_lines[(new_lines.atomic_number == ref.z) &
                          (new_lines.ion_number == ref.ion0)].set_index(
                              "key", verify_integrity=True).sort_index()
        reasons: list[str] = []
        if not old.index.equals(fresh.index):
            missing = old.index.difference(fresh.index).size
            added = fresh.index.difference(old.index).size
            reasons.append(f"line identities missing={missing} added={added}")
        else:
            for column in VALUE_COLUMNS:
                left = old[column].to_numpy(dtype=np.float64).view(np.uint64)
                right = fresh[column].to_numpy(dtype=np.float64).view(np.uint64)
                count = int(np.count_nonzero(left != right))
                if count:
                    reasons.append(f"{column} bit mismatches={count}")
                    total_mismatches += count
        if reasons:
            total_mismatches += 1
            print(f"  Z={ref.z:2d} ion={ref.stage:2d} "
                  f"{ref.linked_vintage}: FAIL — {'; '.join(reasons)}")
        else:
            print(f"  Z={ref.z:2d} ion={ref.stage:2d} "
                  f"{ref.linked_vintage}: {len(old)} lines, f/A/lambda bits PASS")
    passed = bool(stable) and total_mismatches == 0
    print(f"  same-vintage ions={len(stable)}, mismatches={total_mismatches} "
          f"{'PASS' if passed else 'FAIL'}")
    if not passed:
        failures.append(
            f"gate3: same-vintage ions={len(stable)}, mismatches={total_mismatches}")


def sigma_info(path: Path) -> tuple[int, int, int]:
    sigma = path / "cmfgen_sigma_bf.bin"
    with sigma.open("rb") as stream:
        header = stream.read(32)
        if len(header) != 32:
            raise RuntimeError(f"short sigma header in {sigma}")
        magic, version, n_levels, n_freq = struct.unpack("<IIii", header[:16])
        if magic != 0x434D4644 or version != 1 or n_levels < 0 or n_freq <= 0:
            raise RuntimeError(f"bad sigma header in {sigma}")
        flags = stream.read(n_levels)
        if len(flags) != n_levels:
            raise RuntimeError(f"short sigma flags in {sigma}")
    padding = (8 - n_levels % 8) % 8
    expected_size = 32 + n_levels + padding + 8 * n_levels * n_freq
    if sigma.stat().st_size != expected_size:
        raise RuntimeError(
            f"sigma size mismatch in {sigma}: {sigma.stat().st_size}/{expected_size}")
    return n_levels, sum(flags), n_freq


def collision_info(path: Path, levels) -> tuple[int, int]:
    manifest_path = path / "coldata_cmfgen_manifest.csv"
    frame = pd.read_csv(manifest_path, keep_default_na=False)
    required = {"Z", "ion0", "n_levels_ref", "n_mapped", "out_bin", "status"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"{manifest_path} missing columns {sorted(missing)}")
    expected = {(z, stage - 1): len(rows) for (z, stage), rows in levels.items()}
    seen: set[tuple[int, int]] = set()
    extent = 0
    mapped = 0
    for row in frame.to_dict("records"):
        key = (int(row["Z"]), int(row["ion0"]))
        if key in seen or key not in expected:
            raise RuntimeError(f"collision manifest duplicate/foreign ion {key} in {path}")
        seen.add(key)
        nlev = int(row["n_levels_ref"])
        nmap = int(row["n_mapped"])
        if nlev != expected[key]:
            raise RuntimeError(
                f"collision level extent mismatch for {key}: {nlev}/{expected[key]}")
        extent += nlev
        mapped += nmap
        if row["status"] == "OK":
            binary = path / str(row["out_bin"])
            with binary.open("rb") as stream:
                header = stream.read(28)
            if len(header) != 28:
                raise RuntimeError(f"short collision header in {binary}")
            magic, version, z, ion0, ntr, _nt, nlev_bin = struct.unpack(
                "<IIiiiii", header)
            if (magic != 0x49474331 or version != 1 or (z, ion0) != key or
                    ntr != nmap or nlev_bin != nlev):
                raise RuntimeError(f"collision header/manifest mismatch in {binary}")
        elif row["out_bin"]:
            raise RuntimeError(f"non-OK collision row names binary for {key}")
    if seen != set(expected):
        absent = sorted(set(expected) - seen)
        raise RuntimeError(f"collision manifest lacks deck ions: {absent}")
    return extent, mapped


def gate4(base: Path, new: Path, failures: list[str]) -> None:
    base_levels = read_levels(base, require_configuration=False)
    new_levels = read_levels(new, require_configuration=True)
    base_nlev = sum(map(len, base_levels.values()))
    new_nlev = sum(map(len, new_levels.values()))
    base_sigma_n, base_sigma_present, _ = sigma_info(base)
    new_sigma_n, new_sigma_present, _ = sigma_info(new)
    base_col_extent, base_mapped = collision_info(base, base_levels)
    new_col_extent, new_mapped = collision_info(new, new_levels)
    passed = (
        new_nlev > base_nlev
        and base_sigma_n == base_nlev
        and new_sigma_n == new_nlev
        and new_sigma_present > base_sigma_present
        and base_col_extent == base_nlev
        and new_col_extent == new_nlev
        and new_col_extent > base_col_extent
        and new_mapped > 0
    )
    print("GATE 4 — sigma/Upsilon sidecars retain coupled level expansion")
    print(f"  deck levels             {base_nlev} -> {new_nlev}")
    print(f"  sigma addressable       {base_sigma_n} -> {new_sigma_n}")
    print(f"  sigma present           {base_sigma_present} -> {new_sigma_present}")
    print(f"  Upsilon addressable     {base_col_extent} -> {new_col_extent}")
    print(f"  Upsilon mapped rows     {base_mapped} -> {new_mapped} (informational)")
    print(f"  {'PASS' if passed else 'FAIL'}")
    if not passed:
        failures.append("gate4: sigma/Upsilon sidecars do not retain expanded level extent")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=BASE_DEFAULT)
    parser.add_argument("--fullcov", type=Path, default=FULLCOV_DEFAULT)
    parser.add_argument("--new", type=Path, default=NEW_DEFAULT)
    parser.add_argument("--cmf-run", type=Path, default=RUN_DEFAULT)
    args = parser.parse_args()

    for label, path in (("base", args.base), ("fullcov", args.fullcov),
                        ("new", args.new)):
        if not path.is_dir():
            print(f"ERROR: {label} deck absent: {path}", file=sys.stderr)
            return 2

    try:
        expand, links, refs = cmf_references(args.cmf_run)
        new_lines = deck_lines(args.new)
        failures: list[str] = []
        gate1(refs, new_lines, failures)
        gate2(expand, links, refs, args.new, failures)
        gate3(refs, args.fullcov, args.new, failures)
        gate4(args.base, args.new, failures)
    except Exception as exc:
        print(f"ERROR: verifier input/contract failure: {exc}", file=sys.stderr)
        return 2

    if failures:
        print("VERDICT: FAIL — read-only; no adjustment was made", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    print("VERDICT: all four R1 vintage gates PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
