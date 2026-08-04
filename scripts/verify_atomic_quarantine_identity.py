#!/usr/bin/env python3
"""Read-only bidirectional identity and quarantine-integrity verifier.

Mismatch details are written exhaustively outside the deck.  There is no
tolerance, sampling, repair, fallback, or exception swallowing.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
import re
import struct
import sys

import h5py
import numpy as np

from atomic_quarantine_contract import (
    ACTIVE_ROOT_FILES,
    AtomicContractError,
    compare_ion_inventory,
    compare_multiset,
    guard_active_path,
    open_active,
    read_active_ions,
    read_quarantine_ions,
    sha256_file,
)
from kshape_contract import check_contract


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DECK = ROOT / "data/tardis_reference_toy06_19p48d_sivcaiv_active"
DEFAULT_RUN = Path("/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4")
CM_TO_EV = 1.239841984e-4


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def model_rows(path: Path) -> list[tuple[int, int]]:
    pattern = re.compile(r"^\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s+\[[^]]+_ISF\]")
    result = []
    for line in path.read_text(encoding="latin-1").splitlines():
        match = pattern.match(line)
        if match:
            nsl, repeated, nf = map(int, match.groups())
            if nsl != repeated:
                raise RuntimeError(f"MODEL_SPEC N_SL fields disagree: {line}")
            result.append((nsl, nf))
    return result


def report_writer(report_dir: Path, name: str, fields: list[str]):
    path = report_dir / name
    stream = path.open("w", newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    return path, stream, writer


def gate_seal(deck: Path, failures: list[str]) -> dict:
    path = deck / "quarantine/manifest.json"
    manifest = json.loads(path.read_text())
    print("GATE Q1 — sealed archive and active-root byte integrity")
    if manifest.get("state") != "sealed":
        failures.append(f"manifest state={manifest.get('state')!r}, expected sealed")
    snapshot = deck / "quarantine/source_deck_snapshot"
    bad_archive = 0
    for name, record in manifest["archive_files"].items():
        source = snapshot / name
        if (
            not source.is_file()
            or source.stat().st_size != record["bytes"]
            or sha256_file(source) != record["sha256"]
        ):
            bad_archive += 1
    actual_names = {
        path.name for path in deck.iterdir()
        if path.is_file() and path.name != "verification.log"
    }
    sealed_names = set(manifest.get("active_files", {}))
    bad_active = 0
    if actual_names != sealed_names:
        failures.append(
            f"active root file set changed: missing={sorted(sealed_names-actual_names)}, "
            f"extra={sorted(actual_names-sealed_names)}"
        )
    for name, record in manifest.get("active_files", {}).items():
        source = deck / name
        if (
            not source.is_file()
            or source.stat().st_size != record["bytes"]
            or sha256_file(source) != record["sha256"]
        ):
            bad_active += 1
    print(
        f"  archive files={len(manifest['archive_files'])} bad={bad_archive}; "
        f"active files={len(sealed_names)} bad={bad_active} "
        f"{'PASS' if not bad_archive and not bad_active and actual_names == sealed_names else 'FAIL'}"
    )
    if bad_archive or bad_active:
        failures.append(
            f"sealed hash mismatch: archive={bad_archive}, active={bad_active}"
        )
    return manifest


def archived_inventory(deck: Path) -> set[tuple[int, int]]:
    result = set()
    with (deck / "quarantine/source_deck_snapshot/levels.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            result.add((int(row["atomic_number"]), int(row["ion_number"])))
    return result


def loaded_inventory(deck: Path) -> set[tuple[int, int]]:
    result = set()
    with open_active(deck, "levels.csv", newline="") as stream:
        for row in csv.DictReader(stream):
            result.add((int(row["atomic_number"]), int(row["ion_number"])))
    return result


def assert_csv_subset(
    deck: Path, name: str, z_column: str, ion_column: str,
    active: set[tuple[int, int]], failures: list[str],
) -> None:
    bad = Counter()
    with open_active(deck, name, newline="") as stream:
        for row in csv.DictReader(stream):
            key = (int(row[z_column]), int(row[ion_column]))
            if key not in active:
                bad[key] += 1
    if bad:
        failures.append(f"{name} rows outside active set: {dict(bad)}")


def csv_ion_set(deck: Path, name: str, z_column: str, ion_column: str):
    result = set()
    with open_active(deck, name, newline="") as stream:
        for row in csv.DictReader(stream):
            result.add((int(row[z_column]), int(row[ion_column])))
    return result


def gate_inventory(
    deck: Path,
    expected: set[tuple[int, int]],
    failures: list[str],
) -> set[tuple[int, int]]:
    print("GATE Q2 — Lumina_active = CMFGEN_linked, both directions")
    declared = read_active_ions(deck)
    loaded = loaded_inventory(deck)
    quarantined = read_quarantine_ions(deck)
    preserved = archived_inventory(deck)
    result = compare_ion_inventory(expected, loaded, quarantined, preserved)
    if declared != loaded:
        failures.append(
            f"active_ions.csv/runtime levels differ: "
            f"declared-only={sorted(declared-loaded)}, loaded-only={sorted(loaded-declared)}"
        )
    for line in result.diagnostics:
        failures.append(line)
    print(
        f"  expected={len(expected)} loaded={len(loaded)} quarantine={len(quarantined)} "
        f"preserved={len(preserved)} {'PASS' if result.passed and declared == loaded else 'FAIL'}"
    )

    tables = [
        ("atomic_vintage_manifest.csv", "atomic_number", "ion_number"),
        ("ionization_energies.csv", "atomic_number", "ion_number"),
        ("level_multiplicity.csv", "atomic_number", "ion_number"),
        ("line_list.csv", "atomic_number", "ion_number"),
        ("macro_atom_data.csv", "atomic_number", "ion_number"),
        ("macro_atom_references.csv", "atomic_number", "ion_number"),
        ("zeta_ions.csv", "atomic_number", "ion_number"),
    ]
    for args in tables:
        assert_csv_subset(deck, *args, active=expected, failures=failures)
    for name in ("atomic_vintage_manifest.csv", "ionization_energies.csv"):
        inventory = csv_ion_set(deck, name, "atomic_number", "ion_number")
        if inventory != expected:
            failures.append(
                f"{name} ion set differs: missing={sorted(expected-inventory)}, "
                f"extra={sorted(inventory-expected)}"
            )
    expected_z = {z for z, _ in expected}
    for name in ("atom_masses.csv", "abundances.csv"):
        with open_active(deck, name, newline="") as stream:
            actual_z = {int(row["atomic_number"]) for row in csv.DictReader(stream)}
        if actual_z != expected_z:
            failures.append(
                f"{name} element set differs: missing={sorted(expected_z-actual_z)}, "
                f"extra={sorted(actual_z-expected_z)}"
            )
    with open_active(deck, "ma_radrecomb_target_manifest.csv", newline="") as stream:
        bad_ma = Counter()
        for row in csv.DictReader(stream):
            key = (int(row["Z"]), int(row["stage"]) - 1)
            if key not in expected:
                bad_ma[key] += 1
    if bad_ma:
        failures.append(f"ma_radrecomb manifest outside active set: {dict(bad_ma)}")
    with h5py.File(deck / "atomic_data_cmfgen.h5", "r") as h5:
        h5_ions = {
            (int(name[1:3]), int(name.split("ion", 1)[1])) for name in h5.keys()
        }
    if h5_ions != expected:
        failures.append(
            f"HDF5 ion set differs: missing={sorted(expected-h5_ions)}, "
            f"extra={sorted(h5_ions-expected)}"
        )
    return loaded


def expected_level_records(refs, ftos_by_key):
    identities = Counter()
    values = {}
    for key, ref in refs.items():
        z, stage = key
        osc = ref.osc
        nf = ref.nf
        transitions = osc.transitions
        mask = (
            (transitions["i"] >= 1) & (transitions["j"] >= 1)
            & (transitions["i"] <= nf) & (transitions["j"] <= nf)
            & (transitions["lam_A"] != 0.0)
        )
        radiative = transitions[mask]
        a_sum = [0.0] * nf
        for transition in radiative:
            upper = max(int(transition["i"]), int(transition["j"])) - 1
            a_sum[upper] += float(transition["A"])
        membership = ftos_by_key[key].ftos.sl_of_fl[:nf]
        for rank in range(nf):
            identity = (z, stage - 1, rank)
            identities[identity] += 1
            values[identity] = (
                f"{float(osc.levels['E_cm'][rank]) * CM_TO_EV:.10f}",
                str(int(round(float(osc.levels["g"][rank])))),
                "1" if a_sum[rank] == 0.0 else "0",
                str(int(membership[rank])),
                str(osc.levels["config"][rank]),
            )
    return identities, values


def loaded_level_records(deck: Path):
    identities = Counter()
    values = {}
    with open_active(deck, "levels.csv", newline="") as stream:
        for row in csv.DictReader(stream):
            identity = (
                int(row["atomic_number"]), int(row["ion_number"]),
                int(row["level_number"]),
            )
            identities[identity] += 1
            values[identity] = (
                row["energy_eV"], row["g"], row["metastable"],
                row["super_level"], row["configuration"],
            )
    return identities, values


def gate_levels(deck: Path, refs, ftos_by_key, report_dir: Path, failures: list[str]):
    print("GATE Q3 — level identity/membership/value, both directions")
    expected_keys, expected_values = expected_level_records(refs, ftos_by_key)
    loaded_keys, loaded_values = loaded_level_records(deck)
    key_result = compare_multiset("LEVEL", expected_keys, loaded_keys)
    for line in key_result.diagnostics:
        failures.append(line)
    path, stream, writer = report_writer(
        report_dir, "level_identity_mismatches.csv",
        ["kind", "atomic_number", "ion_number", "level_number", "expected", "loaded"],
    )
    value_bad = 0
    for key in sorted(set(expected_values) & set(loaded_values)):
        if expected_values[key] != loaded_values[key]:
            value_bad += 1
            writer.writerow({
                "kind": "FAIL_VALUE_LEVEL", "atomic_number": key[0],
                "ion_number": key[1], "level_number": key[2],
                "expected": repr(expected_values[key]),
                "loaded": repr(loaded_values[key]),
            })
    stream.close()
    if value_bad:
        failures.append(f"FAIL_VALUE_LEVEL count={value_bad}; details={path}")
    print(
        f"  expected={sum(expected_keys.values())} loaded={sum(loaded_keys.values())} "
        f"missing={sum((expected_keys-loaded_keys).values())} "
        f"extra={sum((loaded_keys-expected_keys).values())} values={value_bad} "
        f"{'PASS' if key_result.passed and not value_bad else 'FAIL'}"
    )


def expected_line_records(refs):
    records = defaultdict(list)
    for key, ref in refs.items():
        z, stage = key
        for transition in ref.osc.transitions:
            i, j = int(transition["i"]), int(transition["j"])
            if i < 1 or j < 1 or i > ref.nf or j > ref.nf:
                continue
            if float(transition["lam_A"]) == 0.0:
                continue
            lower, upper = sorted((i - 1, j - 1))
            identity = (z, stage - 1, lower, upper)
            records[identity].append((
                float(transition["f"]), float(transition["A"]),
                abs(float(transition["lam_A"])),
            ))
    for values in records.values():
        values.sort()
    return records


def loaded_line_records(deck: Path):
    records = defaultdict(list)
    n_rows = 0
    with open_active(deck, "line_list.csv", newline="") as stream:
        for row in csv.DictReader(stream):
            identity = (
                int(row["atomic_number"]), int(row["ion_number"]),
                int(row["level_number_lower"]), int(row["level_number_upper"]),
            )
            records[identity].append((
                float(row["f_lu"]), float(row["A_ul"]), float(row["wavelength"]),
            ))
            n_rows += 1
    for values in records.values():
        values.sort()
    return records, n_rows


def gate_lines(deck: Path, refs, report_dir: Path, failures: list[str]):
    print("GATE Q4 — line multiset and f/A/lambda value identity, both directions")
    expected = expected_line_records(refs)
    loaded, n_loaded = loaded_line_records(deck)
    path, stream, writer = report_writer(
        report_dir, "line_identity_mismatches.csv",
        ["kind", "atomic_number", "ion_number", "lower", "upper",
         "occurrence", "expected", "loaded"],
    )
    missing = extra = value_bad = 0
    for key in sorted(set(expected) | set(loaded)):
        left = expected.get(key, [])
        right = loaded.get(key, [])
        common = min(len(left), len(right))
        for occurrence in range(common):
            if left[occurrence] != right[occurrence]:
                value_bad += 1
                writer.writerow({
                    "kind": "FAIL_VALUE_LINE", "atomic_number": key[0],
                    "ion_number": key[1], "lower": key[2], "upper": key[3],
                    "occurrence": occurrence, "expected": repr(left[occurrence]),
                    "loaded": repr(right[occurrence]),
                })
        for occurrence in range(common, len(left)):
            missing += 1
            writer.writerow({
                "kind": "FAIL_MISSING_LINE", "atomic_number": key[0],
                "ion_number": key[1], "lower": key[2], "upper": key[3],
                "occurrence": occurrence, "expected": repr(left[occurrence]),
                "loaded": "",
            })
        for occurrence in range(common, len(right)):
            extra += 1
            writer.writerow({
                "kind": "FAIL_EXTRA_LINE", "atomic_number": key[0],
                "ion_number": key[1], "lower": key[2], "upper": key[3],
                "occurrence": occurrence, "expected": "",
                "loaded": repr(right[occurrence]),
            })
    stream.close()
    n_expected = sum(map(len, expected.values()))
    if missing or extra or value_bad:
        failures.append(
            f"line identity missing={missing} extra={extra} values={value_bad}; details={path}"
        )
    config = json.loads((deck / "config.json").read_text())
    if config.get("n_lines") != n_loaded:
        failures.append(
            f"config n_lines={config.get('n_lines')} != line_list rows={n_loaded}"
        )
    print(
        f"  expected={n_expected} loaded={n_loaded} missing={missing} extra={extra} "
        f"values={value_bad} config={config.get('n_lines')} "
        f"{'PASS' if not (missing or extra or value_bad) and config.get('n_lines') == n_loaded else 'FAIL'}"
    )


def gate_derived_contract(deck: Path, failures: list[str]) -> None:
    """Check every global offset/reference array before a runtime can load it."""
    print("GATE Q5 — derived arrays, offsets, references, and dynamic inputs")
    with open_active(deck, "levels.csv", newline="") as stream:
        n_levels = sum(1 for _ in csv.DictReader(stream))
    with open_active(deck, "line_list.csv", newline="") as stream:
        n_lines = sum(1 for _ in csv.DictReader(stream))
    with open_active(deck, "macro_atom_data.csv", newline="") as stream:
        reader = csv.DictReader(stream)
        n_macro = 0
        bad_level_ref = bad_line_ref = 0
        for row in reader:
            n_macro += 1
            source = int(row["source_level_idx"])
            destination = int(row["destination_level_idx"])
            if not (0 <= source < n_levels and 0 <= destination < n_levels):
                bad_level_ref += 1
            line = int(row["lines_idx"])
            if line < 0 or line >= n_lines:
                bad_line_ref += 1
    with open_active(deck, "macro_atom_references.csv", newline="") as stream:
        references = list(csv.DictReader(stream))
    bad_refs = 0
    if len(references) != n_levels:
        bad_refs += abs(len(references) - n_levels) or 1
    for index, row in enumerate(references):
        if int(row["references_idx"]) != index:
            bad_refs += 1
        start = int(row["block_references"])
        count = int(row["count_total"])
        if start < 0 or count < 0 or start + count > n_macro:
            bad_refs += 1

    line2macro = np.load(
        guard_active_path(deck, deck / "line2macro_level_upper.npy"),
        allow_pickle=False, mmap_mode="r",
    )
    tau = np.load(
        guard_active_path(deck, deck / "tau_sobolev.npy"),
        allow_pickle=False, mmap_mode="r",
    )
    probabilities = np.load(
        guard_active_path(deck, deck / "transition_probabilities.npy"),
        allow_pickle=False, mmap_mode="r",
    )
    config = json.loads((deck / "config.json").read_text())
    n_shells = int(config["n_shells"])
    shape_bad = []
    if line2macro.shape != (n_lines,):
        shape_bad.append(f"line2macro={line2macro.shape}")
    elif np.any((line2macro < 0) | (line2macro >= n_levels)):
        shape_bad.append("line2macro has orphan/out-of-range level")
    if tau.shape != (n_lines, n_shells):
        shape_bad.append(f"tau={tau.shape}")
    if probabilities.shape != (n_macro, n_shells):
        shape_bad.append(f"transition_probabilities={probabilities.shape}")
    try:
        check_contract(deck)
    except (OSError, ValueError) as exc:
        shape_bad.append(f"K-SHAPE contract: {exc}")

    sigma_path = guard_active_path(deck, deck / "cmfgen_sigma_bf.bin")
    with sigma_path.open("rb") as stream:
        header = stream.read(32)
    if len(header) != 32:
        shape_bad.append("sigma header truncated")
    else:
        magic, version, sigma_levels, _, _, _ = struct.unpack("<IIiidd", header)
        if magic != 0x434D4644 or version != 1 or sigma_levels != n_levels:
            shape_bad.append(
                f"sigma header magic/version/n_levels={magic:#x}/{version}/{sigma_levels}"
            )
    target_path = guard_active_path(deck, deck / "ma_radrecomb_target.bin")
    with target_path.open("rb") as stream:
        header = stream.read(20)
    if len(header) != 20:
        shape_bad.append("ma_radrecomb header truncated")
    else:
        magic, version, target_levels, _, _ = struct.unpack("<IIiii", header)
        if magic != 0x4D415254 or version != 2 or target_levels != n_levels:
            shape_bad.append(
                f"ma_radrecomb magic/version/n_levels={magic:#x}/{version}/{target_levels}"
            )

    dynamic_bad = 0
    with open_active(deck, "coldata_cmfgen_manifest.csv", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["status"] != "OK":
                continue
            name = row["out_bin"]
            if not re.fullmatch(r"ige_col_\d+_\d+_cmfgen\.bin", name):
                dynamic_bad += 1
                continue
            try:
                approved = guard_active_path(deck, deck / name, {name})
            except AtomicContractError:
                dynamic_bad += 1
                continue
            if not approved.is_file():
                dynamic_bad += 1

    if bad_level_ref or bad_line_ref or bad_refs or shape_bad or dynamic_bad:
        failures.append(
            "derived contract: "
            f"level_refs={bad_level_ref}, line_refs={bad_line_ref}, "
            f"block_refs={bad_refs}, shapes={shape_bad}, dynamic_inputs={dynamic_bad}"
        )
    print(
        f"  levels={n_levels} lines={n_lines} macro={n_macro}; "
        f"level_refs={bad_level_ref} line_refs={bad_line_ref} block_refs={bad_refs} "
        f"shape_fail={len(shape_bad)} dynamic_fail={dynamic_bad} "
        f"{'PASS' if not (bad_level_ref or bad_line_ref or bad_refs or shape_bad or dynamic_bad) else 'FAIL'}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck", type=Path, default=DEFAULT_DECK)
    parser.add_argument("--cmf-run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--links", type=Path)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    deck = args.deck.resolve()
    report_dir = args.report_dir.resolve()
    try:
        report_dir.relative_to(deck)
    except ValueError:
        pass
    else:
        raise SystemExit("report-dir must be outside the read-only deck")
    report_dir.mkdir(parents=True, exist_ok=False)
    links = args.links or args.cmf_run / "atomic_links.txt"
    if not deck.is_dir() or not links.is_file() or not (args.cmf_run / "MODEL_SPEC").is_file():
        raise SystemExit("deck/MODEL_SPEC/atomic_links input absent")

    failures: list[str] = []
    try:
        manifest = gate_seal(deck, failures)
        expand = load_module(
            "expand_for_bidirectional_identity", ROOT / "scripts/expand_atomic_data_cmfgen.py"
        )
        r1 = load_module(
            "r1_for_bidirectional_identity", ROOT / "scripts/verify_deck_r1_vintage.py"
        )
        audit = load_module(
            "r4_for_bidirectional_identity", ROOT / "scripts/audit_r4_ftos.py"
        )
        _, _, refs = r1.cmf_references(args.cmf_run)
        ordered = r1.ordered_osc_link_keys(expand, links)
        model = model_rows(args.cmf_run / "MODEL_SPEC")
        if len(ordered) != len(model) or set(ordered) != set(refs):
            failures.append("CMFGEN MODEL_SPEC/link/reference ion sets differ")
        for key, (_, nf) in zip(ordered, model, strict=True):
            if refs[key].nf != nf:
                failures.append(f"MODEL_SPEC NF mismatch for {key}")
        expected = {(z, stage - 1) for z, stage in refs}
        gate_inventory(deck, expected, failures)
        ftos_by_key = {entry.key: entry for entry in audit.read_entries(links)}
        if set(ftos_by_key) != set(refs):
            failures.append("R4 f_to_s ion set differs from linked osc ion set")
        gate_levels(deck, refs, ftos_by_key, report_dir, failures)
        gate_lines(deck, refs, report_dir, failures)
        gate_derived_contract(deck, failures)
        if manifest.get("classification_counts") != {"a": 6, "b": 0, "c": 26}:
            failures.append("quarantine classification count differs from pre-registration")
    except (AtomicContractError, KeyError, ValueError, RuntimeError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    if failures:
        print("ATOMIC QUARANTINE IDENTITY VERDICT: FAIL", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    print("ATOMIC QUARANTINE IDENTITY VERDICT: PASS (R1/R4 active scope exact)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
