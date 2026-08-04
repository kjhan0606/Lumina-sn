#!/usr/bin/env python3
"""Generate/run the self-contained 1% and negative-control fixtures for L1-A."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile

import numpy as np

import l1a_collision
import l1a_lines
import l1a_sigma
from l1a_common import (
    ContractError, endpoint, evidence, load_golden, make_record, validate_record,
)


N_LEVELS = 292                 # two decks: 584,000 points ~= 1% of 58,384,000
N_FREQ = 1000
N_LINES = 24_025               # two decks: 48,050 ~= 1% of 4,805,085 rows


def _levels(path: Path) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["atomic_number", "ion_number", "level_number", "energy_eV",
                         "g", "metastable", "super_level", "configuration"])
        for level in range(N_LEVELS):
            writer.writerow([27, 3, level, f"{0.01*level:.10f}", 2+(level % 5),
                             int(level < 2), level % 100, f"fixture_{level}"])


def _lines(path: Path, changed: bool) -> list[tuple[int, int]]:
    selected: list[tuple[int, int]] = []
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["atomic_number", "ion_number", "level_number_lower",
                         "level_number_upper", "line_id", "wavelength", "f_ul", "f_lu",
                         "nu", "B_lu", "B_ul", "A_ul", "wavelength_cm"])
        line_id = 0
        for lo in range(N_LEVELS):
            for up in range(lo+1, N_LEVELS):
                if line_id >= N_LINES:
                    return selected
                f_lu = 1e-4 if line_id % 3 else 1e-6
                a_ul = 1e7 + line_id
                if changed and line_id % 97 == 0:
                    a_ul *= 1.0001
                wavelength = 1000.0 + line_id*0.001
                writer.writerow([27, 3, lo, up, line_id, wavelength, -f_lu, f_lu,
                                 2.99792458e18/wavelength, 1.0, 1.0, a_ul,
                                 wavelength*1e-8])
                if line_id < 100:
                    selected.append((lo, up))
                line_id += 1
    return selected


def _sigma(path: Path, changed: bool) -> None:
    flags = bytes([1])*N_LEVELS
    padding = b"\0"*((8-N_LEVELS % 8) % 8)
    with path.open("wb") as stream:
        stream.write(struct.pack("<IIiidd", 0x434D4644, 1, N_LEVELS, N_FREQ,
                                 1.0e14, 1.0e17))
        stream.write(flags)
        stream.write(padding)
        base = np.linspace(1e-20, 1e-18, N_FREQ, dtype="<f8")
        for level in range(N_LEVELS):
            row = base*(level+1)
            if changed and level % 101 == 0:
                row = row.copy()
                row[::100] *= 1.01
            row.tofile(stream)


def _collision_binary(path: Path, pairs: list[tuple[int, int]]) -> None:
    temperatures = (5000.0, 10000.0, 20000.0)
    with path.open("wb") as stream:
        stream.write(struct.pack("<IIiiiii", 0x49474331, 1, 27, 3,
                                 len(pairs), len(temperatures), N_LEVELS))
        stream.write(struct.pack("<3d", *temperatures))
        for index, pair in enumerate(pairs):
            stream.write(struct.pack("<ii", *pair))
            stream.write(struct.pack("<3d", 0.1+index/1000, 0.2+index/1000,
                                     0.3+index/1000))


def _manifest(path: Path, current: bool, binary: str) -> None:
    columns = ["ion", "Z", "ion0", "osc", "col", "n_levels_osc", "n_levels_ref",
               "n_trans_source", "n_temp", "n_mapped", "n_dropped", "drop_reasons",
               "omega_min", "omega_max", "omega_median", "max_sumrule_err",
               "cmfgen_quality_note", "out_bin", "status"]
    row = {key: "" for key in columns}
    row.update(ion="Co IV", Z=27, ion0=3, osc="fixture/osc_data",
               col="fixture/col_guess.dat" if current else "fixture/col_data",
               n_levels_osc=N_LEVELS, n_levels_ref=N_LEVELS,
               n_trans_source=0 if current else 100, n_temp=0 if current else 3,
               n_mapped=0 if current else 100, n_dropped=0,
               out_bin="" if current else binary,
               status="SKIP: 0 bound-bound pairs" if current else "OK")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerow(row)


def _osc(path: Path, pairs: list[tuple[int, int]]) -> None:
    lines = [
        f"{N_LEVELS} !Number of energy levels",
        "1000000.0 !Ionization energy",
        "4.0 !Screened nuclear charge",
        f"{len(pairs)} !Number of transitions",
    ]
    for level in range(N_LEVELS):
        e_cm = (0.01*level)/1.239841984e-4
        lines.append(f"fixture_{level} {2+(level%5):.1f} {e_cm:.10f} 1.0 0.0 1000.0 "
                     f"{level+1} 0.0")
    for index, (lo, up) in enumerate(pairs, 1):
        f_lu = 1e-4 if (index-1) % 3 else 1e-6
        a_ul = 1e7+(index-1)
        lines.append(f"fixture_{lo} -fixture_{up} {f_lu:.5E} {a_ul:.5E} "
                     f"{1000.0+(index-1)*.001:.6f} {lo+1}-{up+1} {index}")
    path.write_text("\n".join(lines)+"\n", encoding="latin-1")


def _phot(path: Path) -> None:
    lines = [
        f"{N_LEVELS} !Number of energy levels",
        "1 !Number of photoionization routes",
        "4.0 !Screened nuclear charge",
        "fixture_final !Final state in ion",
        "0.0 !Excitation energy of final state",
        "1.0 !Statistical weight of ion",
        "Megabarns !Cross-section unit",
        "False !Split J levels",
        f"{N_LEVELS} !Total number of data pairs",
    ]
    for level in range(N_LEVELS):
        lines.extend((f"fixture_{level} !Configuration name",
                      "1 !Type of cross-section",
                      "1 !Number of cross-section points",
                      "1.0 0.5 3.0"))
    path.write_text("\n".join(lines)+"\n", encoding="latin-1")


def _ftos(path: Path) -> None:
    lines = [f"{N_LEVELS} !Number of energy levels",
             "6 !Entry number of link to super level"]
    for level in range(N_LEVELS):
        lines.append(f"fixture_{level} {2+(level%5):.1f} {level:.4f} 1.0 1000.0 "
                     f"{min(level,100)+1} 0")
    path.write_text("\n".join(lines)+"\n", encoding="latin-1")


def _col(path: Path, pairs: list[tuple[int, int]]) -> None:
    selected = pairs[:100]
    lines = [f"{len(selected)} !Number of transitions",
             "3 !Number of T values", "1.0 !Scaling factor",
             "0.1 !Value for OMEGA if f=0", "Transition\\T 0.5 1.0 2.0"]
    for index, (lo, up) in enumerate(selected):
        lines.append(f"fixture_{lo} - fixture_{up} {0.1+index/1000:.6f} "
                     f"{0.2+index/1000:.6f} {0.3+index/1000:.6f}")
    path.write_text("\n".join(lines)+"\n", encoding="latin-1")


def generate(root: Path) -> None:
    legacy = root/"l1a_fixture"
    current = root/"l1a_fixture_ftos"
    tree = root/"cmfgen_tree"
    run = root/"cmfgen_run"
    for directory in (legacy, current, tree, run):
        directory.mkdir(parents=True, exist_ok=True)
    for deck in (legacy, current):
        _levels(deck/"levels.csv")
    pairs = _lines(legacy/"line_list.csv", changed=False)
    _lines(current/"line_list.csv", changed=True)
    _sigma(legacy/"cmfgen_sigma_bf.bin", changed=False)
    _sigma(current/"cmfgen_sigma_bf.bin", changed=True)
    binary = "ige_col_27_3_cmfgen.bin"
    _collision_binary(legacy/binary, pairs)
    _manifest(legacy/"coldata_cmfgen_manifest.csv", current=False, binary=binary)
    _manifest(current/"coldata_cmfgen_manifest.csv", current=True, binary=binary)
    atomic = tree/"COB/IV/01jan00"
    atomic.mkdir(parents=True)
    _osc(atomic/"osc_data", pairs)
    _phot(atomic/"phot_data_A")
    _ftos(atomic/"f_to_s")
    _col(atomic/"col_data", pairs)
    source = Path("/l1a_fixture_authority/atomic/COB/IV/01jan00")
    (run/"atomic_links.txt").write_text("\n".join((
        f"ln -sf {source/'osc_data'} CoIV_F_OSCDAT",
        f"ln -sf {source/'f_to_s'} CoIV_F_TO_S",
        f"ln -sf {source/'phot_data_A'} PHOTCoIV_A",
        f"ln -sf {source/'col_data'} CoIV_COL_DATA",
    ))+"\n")
    (run/"MODEL_SPEC").write_text(f"100,100,{N_LEVELS} [CoIV_ISF]\n")
    print(root)


def _valid_record(root: Path, *, join_keys: list[str], denominator: int,
                  states: dict[str, int]):
    path = root/"l1a_fixture/coldata_cmfgen_manifest.csv"
    side = endpoint(path, "fixture-role", "fixture-stage", "fixture quantity", "fixture")
    ev = evidence("l1a_fixture negative", Path(__file__), [path], 1)
    return make_record(
        item_id="I1", metric="negative fixture", left=side, right=copy.deepcopy(side),
        denominator=denominator, cardinality=denominator, selection="synthetic defect",
        member_sha=side["sha256"], states=states, threshold_mode="abs", threshold=1e-6,
        digits_left=5, digits_right=5, error_abs=0.0, error_rel=0.0,
        error_ulp=0, zero_rule="skip", join_keys=join_keys, duplicate_count=0,
        duplicate_policy="reject", policy_result="fixture deliberately mutates after construction",
        evidence_obj=ev, processed=1, unsupported=states["unsupported"],
        outcome="MATCH", kind=["COVERAGE"], disposition=["NONE"])


def _kwargs(root: Path) -> dict:
    return {"deck": root/"l1a_fixture_ftos", "peer": root/"l1a_fixture",
            "cmfgen_tree": root/"cmfgen_tree", "cmfgen_run": root/"cmfgen_run",
            "threshold_mode": "rel", "threshold": 1e-6,
            "command": "l1a fixture child", "super_cutoff": 100}


def _rewrite_csv(path: Path, mutate) -> None:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        rows, fields = list(reader), list(reader.fieldnames or ())
    mutate(rows)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _instrument_argv(root: Path, *, same_peer: bool = False) -> list[str]:
    deck = root/"l1a_fixture_ftos"
    peer = deck if same_peer else root/"l1a_fixture"
    return [sys.executable, str(Path(__file__).with_name("l1a_instrument.py")),
            "--deck", str(deck), "--epoch-peer", str(peer),
            "--cmfgen-tree", str(root/"cmfgen_tree"),
            "--cmfgen-run", str(root/"cmfgen_run"), "--super-cutoff", "100",
            "--engine", "all", "--chunk-points", "1048576", "--threshold-mode", "rel"]


def child(case: str, root: Path) -> int:
    kwargs = _kwargs(root)
    if case == "P01":
        links = root/"cmfgen_run/atomic_links.txt"
        direct = root/"direct/atomic/COB/IV/01jan00/osc_data"
        direct.parent.mkdir(parents=True)
        shutil.copyfile(root/"cmfgen_tree/COB/IV/01jan00/osc_data", direct)
        direct.write_text(direct.read_text(encoding="latin-1")+"# SHA fault\n", encoding="latin-1")
        text = links.read_text().replace(
            "/l1a_fixture_authority/atomic/COB/IV/01jan00/osc_data", str(direct))
        links.write_text(text)
        l1a_lines.run(**kwargs)
    elif case == "P02":
        return subprocess.run(_instrument_argv(root, same_peer=True),
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
    elif case == "P03":
        path = root/"l1a_fixture_ftos/levels.csv"
        _rewrite_csv(path, lambda rows: rows[7].update(configuration="rank_7_wrong_config"))
        records = l1a_lines.run(**kwargs)
        if records[0]["measurements"]["semantic_join"]["unmatched"]:
            raise ContractError("P03 semantic config damage was rejected from the join")
        return 0
    elif case == "P04":
        records = l1a_sigma.run(chunk_points=1048576, **kwargs)
        record = copy.deepcopy(records[0])
        if record["states"]["present"]:
            record["states"]["present"] -= 1
        elif record["states"]["missing"]:
            record["states"]["missing"] -= 1
        else:
            record["states"]["unsupported"] -= 1
        validate_record(record)
    elif case == "P05":
        path = root/"l1a_fixture_ftos/levels.csv"
        def duplicate(rows):
            for key in ("configuration", "g", "energy_eV"):
                rows[-1][key] = rows[0][key]
        _rewrite_csv(path, duplicate)
        records = l1a_lines.run(**kwargs)
        if any(record["join"]["duplicate_count"] for record in records):
            raise ContractError("P05 ambiguous duplicate was not selected")
        return 0
    elif case == "P06":
        path = root/"l1a_fixture_ftos/line_list.csv"
        _rewrite_csv(path, lambda rows: rows[0].update(A_ul="1000000"))
        records = l1a_lines.run(**kwargs)
        if records[0]["quantization"]["non_overlap_count"]:
            raise ContractError("P06 raw-token quantization detected non-overlap")
        return 0
    elif case == "P07":
        source = Path(__file__).resolve().parents[1]/"docs/L1_GOLDEN_MANIFEST.json"
        data = json.loads(source.read_text())
        first = next(iter(data["registrations"].values()))
        first["denominator_rule"] += "x"
        damaged = root/"damaged_golden.json"
        damaged.write_text(json.dumps(data))
        load_golden(damaged)
    elif case == "P08":
        path = root/"l1a_fixture_ftos/line_list.csv"
        _rewrite_csv(path, lambda rows: rows[0].update(f_lu="0.123456"))
        records = l1a_lines.run(**kwargs)
        record = next(row for row in records if row["metric"] == "line-bit identity (partial)")
        if record["evidence"]["exit_code"] != 0 and record["evidence"]["stdout_sha256"]:
            raise ContractError("P08 direct validator retained its real nonzero exit/stdout SHA")
        return 0
    elif case == "P09":
        (root/"l1a_fixture_ftos/cmfgen_sigma_bf.bin").rename(
            root/"l1a_fixture_ftos/cmfgen_sigma_bf.bin.missing")
        return subprocess.run(_instrument_argv(root), stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL).returncode
    elif case == "P10":
        record = _valid_record(root, join_keys=["key"], denominator=1,
                               states={"present": 1, "missing": 0, "zero": 0, "unsupported": 0})
        record["resources"]["peak_rss_bytes"] = (1 << 30)+1
        validate_record(record)
    else:
        raise ContractError(f"unknown child case {case}")
    return 0


def negative(root: Path) -> None:
    collision_records = l1a_collision.run(**_kwargs(root))
    if len(collision_records) != 5:
        raise AssertionError(f"collision baseline metric count {len(collision_records)} != 5")
    labels = {
        "P01": "authority_resolution", "P02": "comparison_axis",
        "P03": "semantic_identity", "P04": "state_exhaustion",
        "P05": "duplicate_policy", "P06": "quantization",
        "P07": "golden_binding", "P08": "validator_evidence",
        "P09": "run_completeness", "P10": "resource_limit",
    }
    for code, label in labels.items():
        with tempfile.TemporaryDirectory(prefix=f"l1a_{code.lower()}_") as tmp:
            case_root = Path(tmp)/"fixture"
            shutil.copytree(root, case_root)
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                     "--child", code, str(case_root)],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode == 0:
                raise AssertionError(f"{code} child unexpectedly passed: {result.stdout} {result.stderr}")
            print(f"EXPECTED_FAIL {code} {label} child_exit={result.returncode}")

    rows = l1a_collision._manifest(root/"l1a_fixture/coldata_cmfgen_manifest.csv")
    classified = l1a_collision.classify_requested_ion(rows, (99, 99))
    assert classified["states"]["unsupported"] == 1 and classified["states"]["missing"] == 0

    baseline = _valid_record(root, join_keys=["key"], denominator=1,
                             states={"present": 1, "missing": 0, "zero": 0, "unsupported": 0})
    validate_record(baseline)
    mutations = {
        "C01": lambda r: r["states"].update(present=0),
        "C02": lambda r: r["right"].update(role="wrong-role"),
        "C03": lambda r: r["sampling"].update(sensitive=True, alternatives=["one"]),
        "C04": lambda r: r["quantization"].update(reason=""),
        "C05": lambda r: r["precision"].update(threshold_mode="ulp", dtype=""),
        "C06": lambda r: r["error"].update(zero_denominator_rule="NA", relative=0.0),
        "C07": lambda r: r["schema_flags"].append("EPOCH_MIXED"),
        "C08": lambda r: r["join"].update(keys=[]),
        "C09": lambda r: r.update(id="I404"),
        "C10": lambda r: r["evidence"].update(record_count=0),
        "C11": lambda r: r["resources"].update(peak_rss_bytes=(1 << 30)+1),
        "C12": lambda r: r["build_attestation"].update(binary_sha=""),
    }
    for code, mutate in mutations.items():
        candidate = copy.deepcopy(baseline)
        mutate(candidate)
        try:
            validate_record(candidate)
        except ContractError as exc:
            if not str(exc).startswith(code):
                raise AssertionError((code, exc)) from exc
        else:
            raise AssertionError(f"{code} negative control did not fail")
        print(f"EXPECTED_FAIL {code} contract_guard child_exit=1")
    print("NEGATIVE lines damaged-semantic-key PASS")
    print("NEGATIVE sigma states-denominator PASS")
    print("NEGATIVE collision unsupported-not-missing PASS")
    print("NEGATIVE SUITE PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", type=Path)
    parser.add_argument("--negative", type=Path)
    parser.add_argument("--child", nargs=2, metavar=("CASE", "ROOT"))
    args = parser.parse_args()
    if sum(map(bool, (args.generate, args.negative, args.child))) != 1:
        parser.error("choose exactly one of --generate, --negative, or --child")
    if args.generate:
        generate(args.generate.resolve())
    elif args.negative:
        negative(args.negative.resolve(strict=True))
    else:
        return child(args.child[0], Path(args.child[1]).resolve(strict=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
