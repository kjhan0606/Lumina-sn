#!/usr/bin/env python3
"""Read-only identity gates for the R4 all-linked-f_to_s deck."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
NEW_DEFAULT = ROOT / "data/tardis_reference_toy06_19p48d_sivcaiv_ftos"
LINKS_DEFAULT = Path("/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/atomic_links.txt")
RUN_DEFAULT = LINKS_DEFAULT.parent


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    # exec_module() does not perform the sys.modules insertion done by the
    # normal import machinery.  Dataclasses resolve annotations through that
    # entry while the module is executing, so it must exist before exec_module.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if sys.modules.get(name) is not module:
        raise RuntimeError(f"dynamic import registration failed for {path}")
    return module


def read_levels(path: Path) -> dict[tuple[int, int], list[dict[str, str]]]:
    result: dict[tuple[int, int], list[dict[str, str]]] = {}
    with (path / "levels.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "atomic_number", "ion_number", "level_number", "super_level",
            "configuration", "energy_eV", "g",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise RuntimeError(f"levels.csv missing columns {sorted(missing)}")
        for row in reader:
            key = (int(row["atomic_number"]), int(row["ion_number"]) + 1)
            result.setdefault(key, []).append(row)
    for key, rows in result.items():
        rows.sort(key=lambda row: int(row["level_number"]))
        ranks = [int(row["level_number"]) for row in rows]
        if ranks != list(range(len(rows))):
            raise RuntimeError(f"non-contiguous deck level ranks for {key}")
    return result


def read_manifest(path: Path) -> dict[tuple[int, int], dict[str, str]]:
    result: dict[tuple[int, int], dict[str, str]] = {}
    with (path / "atomic_vintage_manifest.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "atomic_number", "ion_stage", "f_to_s_path", "f_to_s_format",
            "f_to_s_format_basis", "f_to_s_declared_full_levels",
            "f_to_s_declared_super_levels",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise RuntimeError(
                f"atomic_vintage_manifest.csv missing columns {sorted(missing)}"
            )
        for row in reader:
            key = (int(row["atomic_number"]), int(row["ion_stage"]))
            if key in result:
                raise RuntimeError(f"duplicate manifest ion {key}")
            result[key] = row
    return result


def gate_ftos(new: Path, links_path: Path, failures: list[str]) -> None:
    audit = load_script("r4_audit_for_verify", ROOT / "scripts/audit_r4_ftos.py")
    entries = audit.read_entries(links_path)
    levels = read_levels(new)
    manifest = read_manifest(new)
    bad_count = 0
    bad_membership = 0
    bad_provenance = 0
    print("R4 GATE 1/2 — all-ion SL counts and exact FL membership")
    for entry in entries:
        rows = levels.get(entry.key, [])
        actual = np.asarray([int(row["super_level"]) for row in rows], dtype="i4")
        expected = entry.ftos.sl_of_fl
        count_ok = (len(rows) == entry.ftos.n_levels and
                    len(set(actual.tolist())) == entry.ftos.n_super)
        membership_ok = np.array_equal(actual, expected)
        row = manifest.get(entry.key)
        provenance_ok = (
            row is not None and Path(row["f_to_s_path"]) == entry.path and
            row["f_to_s_format"] == entry.ftos.format_name and
            row["f_to_s_format_basis"] == entry.ftos.format_basis and
            int(row["f_to_s_declared_full_levels"]) == entry.ftos.n_levels and
            int(row["f_to_s_declared_super_levels"]) == entry.ftos.n_super
        )
        bad_count += not count_ok
        bad_membership += not membership_ok
        bad_provenance += not provenance_ok
        status = "PASS" if count_ok and membership_ok and provenance_ok else "FAIL"
        print(f"  {entry.ion:7s}: {len(rows)}/{entry.ftos.n_levels} FL, "
              f"{len(set(actual.tolist())) if actual.size else 0}/"
              f"{entry.ftos.n_super} SL; membership="
              f"{'exact' if membership_ok else 'DIFF'}; format="
              f"{entry.ftos.format_name} {status}")
    if len(entries) != 27 or bad_count or bad_membership or bad_provenance:
        failures.append(
            f"f_to_s identity: ions={len(entries)}, count_fail={bad_count}, "
            f"membership_fail={bad_membership}, provenance_fail={bad_provenance}"
        )


def gate_r1(new: Path, cmf_run: Path, failures: list[str]) -> None:
    command = [
        sys.executable, str(ROOT / "scripts/verify_deck_r1_vintage.py"),
        "--new", str(new), "--cmf-run", str(cmf_run),
    ]
    print("R4 GATE 3 — retain all R1 gates")
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode != 0:
        failures.append(f"R1 verifier exit={result.returncode}")


def gate_off_contract(links_deck: Path, failures: list[str]) -> None:
    """Prove the default-OFF branch still denotes the existing R1 bake path.

    A full OFF rebake would require a second multi-GiB deck.  The driver is
    structurally isolated instead: the R4 environment variable defaults false,
    appears only in the parser-selection OR and conditional manifest extension,
    and the immutable comparison target remains the already certified _links
    deck.  This gate fails if that narrow source contract changes.  The sbatch
    additionally snapshots and rechecks every _links byte across the R4 job.
    """
    source_path = ROOT / "scripts/expand_atomic_data_cmfgen.py"
    source = source_path.read_text()
    occurrences = source.count("LINK_FTOS_ENABLED")
    required_fragments = [
        "LINK_FTOS_ENABLED = os.environ.get('CMFGEN_LINK_FTOS', '0')",
        "use_link_ftos = LINK_FTOS_ENABLED and linked_ftos is not None",
        "use_superlev = (use_link_ftos or",
        "if LINK_FTOS_ENABLED:",
    ]
    passed = (
        links_deck.is_dir() and occurrences == 5 and
        all(fragment in source for fragment in required_fragments)
    )
    print("R4 GATE 4A — gate-OFF source contract")
    print(f"  CMFGEN_LINK_FTOS default=0; guarded references={occurrences}; "
          f"_links present={links_deck.is_dir()} {'PASS' if passed else 'FAIL'}")
    if not passed:
        failures.append("gate-OFF source contract changed")


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def gate_off_bytes(links_deck: Path, off_control: Path,
                   failures: list[str]) -> None:
    print("R4 GATE 4B — CMFGEN_LINK_FTOS=0 byte identity to _links")
    if not links_deck.is_dir() or not off_control.is_dir():
        failures.append(
            f"OFF byte inputs absent: _links={links_deck.is_dir()}, "
            f"OFF={off_control.is_dir()}"
        )
        print(f"  input directories: _links={links_deck.is_dir()}, "
              f"OFF={off_control.is_dir()} FAIL")
        return
    if links_deck.samefile(off_control):
        failures.append("OFF control aliases _links; byte gate was not tested")
        print("  OFF control aliases _links FAIL")
        return
    left = {path.relative_to(links_deck) for path in links_deck.rglob("*")
            if path.is_file()}
    right = {path.relative_to(off_control) for path in off_control.rglob("*")
             if path.is_file()}
    if left != right:
        failures.append(
            f"OFF file set differs: missing={sorted(left-right)}, "
            f"added={sorted(right-left)}"
        )
        print(f"  file set FAIL: _links={len(left)}, OFF={len(right)}")
        return
    mismatches = []
    for relative in sorted(left):
        old = links_deck / relative
        fresh = off_control / relative
        if old.stat().st_size != fresh.stat().st_size or \
                file_digest(old) != file_digest(fresh):
            mismatches.append(str(relative))
    print(f"  files={len(left)}, byte mismatches={len(mismatches)} "
          f"{'PASS' if not mismatches else 'FAIL'}")
    if mismatches:
        failures.append(f"OFF byte mismatches={mismatches}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new", type=Path, default=NEW_DEFAULT)
    parser.add_argument("--links", type=Path, default=LINKS_DEFAULT)
    parser.add_argument("--cmf-run", type=Path, default=RUN_DEFAULT)
    parser.add_argument(
        "--links-deck", type=Path,
        default=ROOT / "data/tardis_reference_toy06_19p48d_sivcaiv_links",
    )
    parser.add_argument("--off-control", type=Path, required=True)
    args = parser.parse_args()
    if not args.new.is_dir():
        print(f"ERROR: R4 deck absent: {args.new}", file=sys.stderr)
        return 2

    failures: list[str] = []
    gate_ftos(args.new, args.links, failures)
    gate_r1(args.new, args.cmf_run, failures)
    gate_off_contract(args.links_deck, failures)
    gate_off_bytes(args.links_deck, args.off_control, failures)
    if failures:
        print("R4 VERDICT: FAIL — no adjustment was made", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    print("R4 VERDICT: all f_to_s membership, R1, and OFF-contract gates PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
