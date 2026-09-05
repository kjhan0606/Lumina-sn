#!/usr/bin/env python3
"""Synthetic tests for pump_drive.py; never uses the system temporary directory."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any


sys.dont_write_bytecode = True

import pump_drive


BASE_DIR = Path(__file__).resolve().parent

LEVEL_FIELDS = (
    "atomic_number",
    "ion_number",
    "level_number",
    "energy_eV",
    "g",
    "metastable",
    "super_level",
    "configuration",
)
LINE_FIELDS = (
    "atomic_number",
    "ion_number",
    "level_number_lower",
    "level_number_upper",
    "line_id",
    "wavelength",
    "f_ul",
    "f_lu",
    "nu",
    "B_lu",
    "B_ul",
    "A_ul",
    "wavelength_cm",
)
SPECTRAL_FIELDS = (
    "shell_id",
    "bin_id",
    "nu_lo_Hz",
    "nu_hi_Hz",
    "J_nu",
    "chi_es_cm1",
    "chi_bb_cm1",
    "chi_bf_cm1",
    "chi_ff_cm1",
    "chi_total_cm1",
    "eta_bb",
    "eta_bf",
    "eta_ff",
    "eta_true_total",
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


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def base_levels() -> list[dict[str, Any]]:
    return [
        {
            "atomic_number": 26,
            "ion_number": 1,
            "level_number": 0,
            "energy_eV": 0.0,
            "g": 2.0,
            "metastable": 1,
            "super_level": 0,
            "configuration": "ground",
        },
        {
            "atomic_number": 26,
            "ion_number": 1,
            "level_number": 1,
            "energy_eV": 1.0,
            "g": 4.0,
            "metastable": 1,
            "super_level": 1,
            "configuration": "metastable",
        },
        {
            "atomic_number": 26,
            "ion_number": 1,
            "level_number": 2,
            "energy_eV": 2.0,
            "g": 6.0,
            "metastable": 0,
            "super_level": 2,
            "configuration": "other",
        },
    ]


def line(lower: int, upper: int, line_id: int, photon_eV: float) -> dict[str, Any]:
    nu = photon_eV / pump_drive.H_EV_S
    return {
        "atomic_number": 26,
        "ion_number": 1,
        "level_number_lower": lower,
        "level_number_upper": upper,
        "line_id": line_id,
        "wavelength": 1.0,
        "f_ul": 0.5,
        "f_lu": 1.0,
        "nu": format(nu, ".17g"),
        "B_lu": 1.0,
        "B_ul": 1.0,
        "A_ul": 1.0,
        "wavelength_cm": 1.0,
    }


def base_lines() -> list[dict[str, Any]]:
    return [
        line(0, 1, 101, 3.0),
        line(1, 2, 102, 5.0),
        line(2, 3, 103, 6.0),
    ]


def spectral_rows(iteration: int, *, nonfinite_middle: bool = False) -> list[dict[str, Any]]:
    values0 = (1.0, 2.0, 1.5)
    values1 = (2.0, 6.0, 6.0)
    values = list(values0 if iteration == 0 else values1)
    if iteration == 1 and nonfinite_middle:
        values[1] = math.nan
    edges = ((6.0e14, 9.0e14), (1.0e15, 1.3e15), (1.3e15, 1.6e15))
    rows: list[dict[str, Any]] = []
    for bin_id, ((lo, hi), J) in enumerate(zip(edges, values)):
        row = {field: 0 for field in SPECTRAL_FIELDS}
        row.update(
            {
                "shell_id": 0,
                "bin_id": bin_id,
                "nu_lo_Hz": lo,
                "nu_hi_Hz": hi,
                "J_nu": J,
            }
        )
        rows.append(row)
    # This row proves shell filtering and cannot cover any shell-0 lookup.
    other = {field: 0 for field in SPECTRAL_FIELDS}
    other.update(
        {
            "shell_id": 1,
            "bin_id": 99,
            "nu_lo_Hz": 0.0,
            "nu_hi_Hz": 9.0e99,
            "J_nu": 999.0,
        }
    )
    rows.append(other)
    return rows


def write_fixture(
    root: Path,
    *,
    extra_line: dict[str, Any] | None = None,
    nonfinite_middle: bool = False,
) -> tuple[Path, Path, Path]:
    run_root = root / "synthetic_run_read_only"
    pairs_path = root / "pairs.csv"
    sl_path = root / "sl_groups.csv"
    write_csv(run_root / "input/model/levels.csv", LEVEL_FIELDS, base_levels())
    lines = base_lines()
    if extra_line is not None:
        lines.append(extra_line)
    write_csv(run_root / "input/model/line_list.csv", LINE_FIELDS, lines)
    for iteration in (0, 1):
        write_csv(
            run_root
            / f"work/physics_comparison/physics_DET_iter000{iteration}.spectral.csv",
            SPECTRAL_FIELDS,
            spectral_rows(iteration, nonfinite_middle=nonfinite_middle),
        )
    write_csv(
        pairs_path,
        PAIR_FIELDS,
        [
            {
                "shell": 0,
                "Z": 26,
                "ion": 1,
                "upper_level": 1,
                "runtime_sl_id_0": 10,
                "runtime_sl_id_1": 10,
            },
            {
                "shell": 0,
                "Z": 26,
                "ion": 1,
                "upper_level": 2,
                "runtime_sl_id_0": 10,
                "runtime_sl_id_1": 10,
            },
            {
                "shell": 0,
                "Z": 26,
                "ion": 1,
                "upper_level": 3,
                "runtime_sl_id_0": 11,
                "runtime_sl_id_1": 11,
            },
        ],
    )
    write_csv(
        sl_path,
        SL_FIELDS,
        [
            {"shell": 0, "Z": 26, "ion": 1, "runtime_sl_id": 10, "Y_s": 0.1},
            {"shell": 0, "Z": 26, "ion": 1, "runtime_sl_id": 11, "Y_s": 0.2},
        ],
    )
    return run_root, pairs_path, sl_path


def tree_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        with item.open("rb") as handle:
            while chunk := handle.read(65536):
                digest.update(chunk)
    return digest.hexdigest()


def assert_close(actual: float | None, expected: float, label: str) -> None:
    if actual is None or not math.isclose(actual, expected, rel_tol=1.0e-13, abs_tol=1.0e-13):
        raise AssertionError(f"{label}: got {actual!r}, expected {expected!r}")


def run_case(
    root: Path,
    *,
    extra_line: dict[str, Any] | None = None,
    nonfinite_middle: bool = False,
) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]]]:
    run_root, pairs_path, sl_path = write_fixture(
        root, extra_line=extra_line, nonfinite_middle=nonfinite_middle
    )
    before = tree_digest(run_root)
    out_dir = root / "products"
    summary = pump_drive.execute(run_root, pairs_path, sl_path, out_dir)
    after = tree_digest(run_root)
    if before != after:
        raise AssertionError("analyzer modified the synthetic read-only run root")
    expected_names = {
        "pump_drive_per_upper.csv",
        "pump_drive_sl.csv",
        "pump_drive_summary.json",
    }
    if {path.name for path in out_dir.iterdir()} != expected_names:
        raise AssertionError("unexpected product set")
    with (out_dir / "pump_drive_per_upper.csv").open(newline="") as handle:
        upper_rows = list(csv.DictReader(handle))
    with (out_dir / "pump_drive_sl.csv").open(newline="") as handle:
        sl_rows = list(csv.DictReader(handle))
    with (out_dir / "pump_drive_summary.json").open(encoding="utf-8") as handle:
        serialized = json.load(handle)
    if serialized["schema"] != pump_drive.SCHEMA:
        raise AssertionError("summary schema mismatch")
    return summary, upper_rows, sl_rows


def positive_case(root: Path) -> None:
    summary, upper_rows, sl_rows = run_case(root)
    counts = summary["counts"]["line_list"]
    if counts.get("line_list_stream_passes") != 1:
        raise AssertionError("line list was not reported as one streaming pass")
    if (counts.get("selected_target_lines"), counts.get("selected_lines_used")) != (3, 3):
        raise AssertionError("positive selected/used counts mismatch")
    by_upper = {int(row["upper_level"]): row for row in upper_rows}
    assert_close(float(by_upper[1]["dlog_P_all"]), math.log(2.0), "ground upper")
    assert_close(float(by_upper[2]["dlog_P_all"]), math.log(3.0), "metastable upper")
    assert_close(float(by_upper[3]["dlog_P_all"]), math.log(4.0), "other upper")
    if int(by_upper[1]["used_line_count_ground"]) != 1:
        raise AssertionError("ground class precedence failed")
    if int(by_upper[2]["used_line_count_metastable"]) != 1:
        raise AssertionError("metastable class failed")
    if int(by_upper[3]["used_line_count_other"]) != 1:
        raise AssertionError("other class failed")
    if int(by_upper[1]["used_line_count_lt4eV"]) != 1:
        raise AssertionError("<4 eV split failed")
    if int(by_upper[2]["used_line_count_ge4eV"]) != 1:
        raise AssertionError(">=4 eV split failed")
    by_sl = {int(row["runtime_sl_id"]): row for row in sl_rows}
    expected0 = float(by_upper[1]["P_all_0"]) + float(by_upper[2]["P_all_0"])
    expected1 = float(by_upper[1]["P_all_1"]) + float(by_upper[2]["P_all_1"])
    assert_close(float(by_sl[10]["P_all_0"]), expected0, "lump P0 sum")
    assert_close(float(by_sl[10]["P_all_1"]), expected1, "lump P1 sum")
    assert_close(
        summary["correlation"]["rho_block_centered_pooled_midrank_spearman"],
        1.0,
        "pooled rho",
    )


def outside_bin_case(root: Path) -> None:
    summary, _, _ = run_case(root, extra_line=line(0, 1, 201, 8.0))
    counts = summary["counts"]["line_list"]
    reasons = counts["exclusions_by_reason"]
    if counts.get("selected_target_lines") != 4 or counts.get("selected_lines_excluded") != 1:
        raise AssertionError("outside-bin line was not selected then excluded")
    if reasons.get("frequency_outside_spectral_bins_iter0") != 1:
        raise AssertionError("iteration-0 outside-bin exclusion was not counted")
    if reasons.get("frequency_outside_spectral_bins_iter1") != 1:
        raise AssertionError("iteration-1 outside-bin exclusion was not counted")


def outside_target_case(root: Path) -> None:
    summary, _, _ = run_case(root, extra_line=line(0, 99, 202, 3.0))
    counts = summary["counts"]["line_list"]
    if counts.get("rows_skipped_upper_outside_target") != 1:
        raise AssertionError("U-outside line was not counted as skipped")
    if counts.get("selected_target_lines") != 3:
        raise AssertionError("U-outside line entered the target selection")


def nonfinite_J_case(root: Path) -> None:
    summary, _, _ = run_case(root, nonfinite_middle=True)
    counts = summary["counts"]["line_list"]
    reasons = counts["exclusions_by_reason"]
    if reasons.get("nonfinite_J_iter1") != 1:
        raise AssertionError("non-finite J exclusion was not counted")
    if counts.get("selected_lines_excluded") != 1:
        raise AssertionError("non-finite J did not exclude exactly one selected line")
    if summary["counts"]["spectral_iter1"]["nonfinite_J_rows_shell0"] != 1:
        raise AssertionError("non-finite spectral row was not counted")
    json.dumps(summary, allow_nan=False)


def main() -> int:
    scratch = Path(tempfile.mkdtemp(prefix=".pump_drive_selftest_", dir=BASE_DIR))
    positive = negative = 0
    try:
        cases = (
            ("positive", positive_case),
            ("outside_bin", outside_bin_case),
            ("outside_target_U", outside_target_case),
            ("nonfinite_J", nonfinite_J_case),
        )
        for name, function in cases:
            case_root = scratch / name
            case_root.mkdir()
            function(case_root)
            if name == "positive":
                positive += 1
            else:
                negative += 1
            print(f"PASS {name}")
    finally:
        shutil.rmtree(scratch)
    print(f"selftest_pump_drive: PASS positive={positive}/1 negative={negative}/3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
