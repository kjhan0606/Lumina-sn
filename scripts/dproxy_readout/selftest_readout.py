#!/usr/bin/env python3
"""Synthetic fail-closed tests for readout_dproxy_sl_331050.py.

All temporary files are made beneath the invocation directory.  The system
temporary directory is deliberately not used.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


H_EV_S = 4.135667696e-15
ANALYZER = Path(__file__).resolve().with_name("readout_dproxy_sl_331050.py")


def number(value: float) -> str:
    return format(value, ".17g")


def base_specs() -> dict[int, list[dict[str, Any]]]:
    definitions = [
        # line, upper level/global, SL/anchor, nu, A, n0/n1, J0/J1, frac
        (100, 1, 1001, 10, 1001, 1.0e15, 2.0e7, 2.0, 4.0, 1.0, 2.0, 0.5),
        (101, 2, 1002, 10, 1001, 1.5e15, 3.0e7, 1.0, 2.0, 1.0, 3.0, 0.25),
        (102, 3, 1003, 11, 1003, 2.0e15, 4.0e7, 3.0, 9.0, 1.0, 4.0, 1.0),
        (103, 4, 1004, 11, 1003, 2.5e15, 5.0e7, 2.0, 6.0, 1.0, 5.0, 0.5),
    ]
    result: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    for publication in (0, 1):
        for (
            line,
            upper_level,
            upper_global,
            runtime_sl_id,
            super_anchor_global,
            nu,
            A_ul,
            n0,
            n1,
            J0,
            J1,
            frac,
        ) in definitions:
            result[publication].append(
                {
                    "publication": publication,
                    "line": line,
                    "upper_level": upper_level,
                    "upper_global": upper_global,
                    "runtime_sl_id": runtime_sl_id,
                    "super_anchor_global": super_anchor_global,
                    "nu": nu,
                    "A_ul": A_ul,
                    "n_upper": n0 if publication == 0 else n1,
                    "Jbar": J0 if publication == 0 else J1,
                    "frac": frac,
                }
            )
    return result


def render_row(spec: dict[str, Any]) -> str:
    publication = spec["publication"]
    n_upper = spec["n_upper"]
    frac = spec["frac"]
    n_decimal = spec.get("n_decimal", number(n_upper))
    n_hex = spec.get("n_hex", float(n_upper).hex())
    frac_decimal = spec.get("frac_decimal", number(frac))
    frac_hex = spec.get("frac_hex", float(frac).hex())
    Jbar = spec.get("Jbar_raw", number(spec["Jbar"]))
    return (
        "[A2-10][LINE-SATURATION-ROW] "
        "phase=REQUESTED_TE shell=0 "
        f"line={spec['line']} Z=26 ion=1 "
        "lower_global=1000 "
        f"upper_global={spec['upper_global']} "
        "lower_level=0 "
        f"upper_level={spec['upper_level']} "
        f"nu={number(spec['nu'])} A_ul={number(spec['A_ul'])} "
        f"n_upper={n_decimal} n_upper_hex={n_hex} "
        f"Jbar={Jbar} "
        f"within_sl_frac={frac_decimal} within_sl_frac_hex={frac_hex} "
        f"runtime_sl_id={spec['runtime_sl_id']} "
        f"super_anchor_global={spec['super_anchor_global']} "
        "population_source=NLTE_COMMITTED used_nlte=1 super_mode=1 "
        f"outer_iter={publication} "
        f"te_generation={20 + publication} "
        f"nlte_population_generation={30 + publication} "
        "population_provenance_status=OK "
        f"ledger_join_key={publication}:30:20:0:26:1:0:{spec['runtime_sl_id']}"
    )


def render_semi_state_row(spec: dict[str, Any]) -> str:
    return (
        "[A2-10][LINE-SATURATION-ROW] "
        "phase=REQUESTED_TE shell=0 "
        f"line={spec['line']} Z=26 ion=1 "
        "lower_global=1000 "
        f"upper_global={spec['upper_global']} "
        "lower_level=0 "
        f"upper_level={spec['upper_level']} "
        f"nu={number(spec['nu'])} A_ul={number(spec['A_ul'])} "
        f"n_upper={number(spec['n_upper'])} "
        f"Jbar={number(spec['Jbar'])}"
    )


def render_log(
    specs: dict[int, list[dict[str, Any]]],
    *,
    boundaries: tuple[int, ...] = (0, 1),
    duplicate_publication_0_first_row: bool = False,
    semi_state: bool = False,
) -> str:
    lines: list[str] = ["synthetic preamble"]
    for publication in (0, 1):
        if publication in boundaries:
            lines.append(
                f"[R7][PHASE] lane=DET iter={publication} phase=view "
                "event=R7_MATERIAL_PHASE_COMMITTED"
            )
        lines.append(
            "[A2-10][LINE-ION-OWNER] phase=REQUESTED_TE shell=0 "
            f"T_e_K=10000 ion_slot=0 outer_iter={publication}"
        )
        for spec in specs[publication]:
            lines.append(
                render_semi_state_row(spec) if semi_state else render_row(spec)
            )
        if publication == 0 and duplicate_publication_0_first_row:
            lines.append(
                render_semi_state_row(specs[0][0])
                if semi_state
                else render_row(specs[0][0])
            )
    return "\n".join(lines) + "\n"


def write_levels(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "atomic_number",
                "ion_number",
                "level_number",
                "energy_eV",
                "g",
                "metastable",
                "super_level",
                "configuration",
            ]
        )
        writer.writerow([26, 1, 0, number(0.0), 2, 1, 0, "lower"])
        for level_number, nu in enumerate((1.0e15, 1.5e15, 2.0e15, 2.5e15), 1):
            writer.writerow(
                [
                    26,
                    1,
                    level_number,
                    number(H_EV_S * nu),
                    2 + 2 * level_number,
                    0,
                    level_number,
                    f"upper_{level_number}",
                ]
            )


def create_case(case_dir: Path, log: str) -> tuple[Path, Path]:
    run_root = case_dir / "run"
    out_dir = case_dir / "out"
    run_root.mkdir(parents=True)
    out_dir.mkdir(parents=True)
    (run_root / "stderr.log").write_text(log, encoding="utf-8")
    write_levels(run_root / "input" / "global_atomic" / "synthetic" / "levels.csv")
    return run_root, out_dir


def write_sl_map(
    path: Path,
    specs: dict[int, list[dict[str, Any]]],
    *,
    omit_upper_global: int | None = None,
    runtime_override: dict[int, int] | None = None,
) -> None:
    runtime_override = runtime_override or {}
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "upper_global",
                "runtime_sl_id",
                "super_anchor_global",
                "within_sl_frac_hex",
            ),
        )
        writer.writeheader()
        for spec in specs[0]:
            if spec["upper_global"] == omit_upper_global:
                continue
            writer.writerow(
                {
                    "upper_global": spec["upper_global"],
                    "runtime_sl_id": runtime_override.get(
                        spec["upper_global"], spec["runtime_sl_id"]
                    ),
                    "super_anchor_global": spec["super_anchor_global"],
                    "within_sl_frac_hex": float(spec["frac"]).hex(),
                }
            )


def write_promoted_pairs(
    path: Path,
    specs: dict[int, list[dict[str, Any]]],
    *,
    omit_line: int | None = None,
) -> None:
    by_publication = {
        publication: {spec["line"]: spec for spec in values}
        for publication, values in specs.items()
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "eligible_for_sl",
                "shell",
                "line",
                "upper_global",
                "runtime_sl_id",
                "dlogJ",
                "dlogn",
            ),
        )
        writer.writeheader()
        for line in sorted(by_publication[0]):
            if line == omit_line:
                continue
            row0 = by_publication[0][line]
            row1 = by_publication[1][line]
            writer.writerow(
                {
                    "eligible_for_sl": 1,
                    "shell": 0,
                    "line": line,
                    "upper_global": row0["upper_global"],
                    "runtime_sl_id": row0["runtime_sl_id"],
                    "dlogJ": number(
                        math.log(row1["Jbar"]) - math.log(row0["Jbar"])
                    ),
                    "dlogn": number(
                        math.log(row1["n_upper"])
                        - math.log(row0["n_upper"])
                    ),
                }
            )


def write_seal(run_root: Path) -> None:
    relative_paths = (
        Path("stderr.log"),
        Path("input/global_atomic/synthetic/levels.csv"),
    )
    with (run_root / "RII_RUN_SEAL.sha").open("w", encoding="utf-8") as handle:
        for relative in relative_paths:
            digest = hashlib.sha256((run_root / relative).read_bytes()).hexdigest()
            handle.write(f"{digest}  {relative.as_posix()}\n")


def invoke(
    run_root: Path, out_dir: Path, *extra_args: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-B",
            str(ANALYZER),
            "--run-root",
            str(run_root),
            "--out-dir",
            str(out_dir),
            *extra_args,
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def load_readout(out_dir: Path) -> dict[str, Any]:
    path = out_dir / "readout.json"
    if not path.is_file():
        raise AssertionError(f"readout.json missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def assert_positive(case_dir: Path) -> None:
    specs = base_specs()
    run_root, out_dir = create_case(case_dir, render_log(specs))
    completed = invoke(run_root, out_dir)
    if completed.returncode != 0:
        raise AssertionError(
            f"positive exit={completed.returncode}\nstdout={completed.stdout}\n"
            f"stderr={completed.stderr}"
        )
    result = load_readout(out_dir)
    checks = {
        "validation error count": result["validation"]["error_count"] == 0,
        "matched pair count": result["pairs"]["matched_pair_count"] == 4,
        "eligible pair count": result["pairs"]["eligible_pair_count"] == 4,
        "runtime SL count": result["runtime_sl_summary"]["valid_runtime_sl_count"] == 2,
        "rho_SL defined": result["sl_block_covariation"]["rho_SL"] is not None,
        "D3 exact strings": result["D3_temperature"]["strings_identical"] is True,
        "unsealed": result["sealed"] is False,
        "single stderr pass": result["inputs"]["stderr_log"]["stream_passes"] == 1,
        "line list untouched": result["inputs"]["line_list_csv_read"] is False,
    }
    missing_artifacts = [
        name
        for name in (
            "readout.json",
            "pairs.csv",
            "sl_groups.csv",
            "exclusions.csv",
            "sl_map.csv",
        )
        if not (out_dir / name).is_file()
    ]
    failed = [name for name, condition in checks.items() if not condition]
    if failed or missing_artifacts:
        raise AssertionError(
            f"positive assertions failed={failed} missing_artifacts={missing_artifacts}"
        )


def negative_logs() -> list[tuple[str, str, str, str | None]]:
    cases: list[tuple[str, str, str, str | None]] = []

    specs = base_specs()
    cases.append(
        (
            "duplicate_pair",
            render_log(specs, duplicate_publication_0_first_row=True),
            "DUPLICATE_PAIR_KEY",
            None,
        )
    )

    specs = base_specs()
    cases.append(
        (
            "publication_boundary_missing",
            render_log(specs, boundaries=(1,)),
            "SCHEMA_PUBLICATION_BOUNDARY_COUNT",
            None,
        )
    )

    specs = base_specs()
    specs[0][0]["Jbar_raw"] = "nan"
    cases.append(
        (
            "nonfinite_numeric",
            render_log(specs),
            "NUMERIC_NONFINITE",
            None,
        )
    )

    specs = base_specs()
    specs[1][0]["runtime_sl_id"] = 99
    cases.append(
        (
            "runtime_sl_repetition_drift",
            render_log(specs),
            "PAIR_INVARIANT_VIOLATION",
            "runtime_sl_id",
        )
    )

    specs = base_specs()
    specs[0][0]["frac_decimal"] = "0.59999999999999998"
    cases.append(
        (
            "within_sl_frac_hex_decimal_mismatch",
            render_log(specs),
            "HEX_DECIMAL_ULP_MISMATCH",
            "within_sl_frac",
        )
    )

    specs = base_specs()
    specs[0][0]["nu"] = 1.2e15
    specs[1][0]["nu"] = 1.2e15
    cases.append(
        (
            "nu_level_energy_mismatch",
            render_log(specs),
            "NU_LEVEL_ENERGY_MISMATCH",
            None,
        )
    )
    return cases


def assert_negative(
    case_dir: Path,
    name: str,
    log: str,
    expected_reason: str,
    expected_detail: str | None,
) -> None:
    run_root, out_dir = create_case(case_dir, log)
    completed = invoke(run_root, out_dir)
    if completed.returncode != 2:
        raise AssertionError(
            f"{name}: expected exit 2, got {completed.returncode}\n"
            f"stdout={completed.stdout}\nstderr={completed.stderr}"
        )
    result = load_readout(out_dir)
    serialized_errors = json.dumps(
        result["validation"]["errors"], sort_keys=True, ensure_ascii=True
    )
    if result["validation"]["error_count"] < 1:
        raise AssertionError(f"{name}: no validation error was retained")
    if expected_reason not in serialized_errors:
        raise AssertionError(
            f"{name}: expected reason {expected_reason!r} absent from {serialized_errors}"
        )
    if expected_detail is not None and expected_detail not in serialized_errors:
        raise AssertionError(
            f"{name}: expected detail {expected_detail!r} absent from {serialized_errors}"
        )


def assert_sl_map_missing_exclusion(case_dir: Path) -> None:
    specs = base_specs()
    run_root, out_dir = create_case(
        case_dir, render_log(specs, semi_state=True)
    )
    sl_map_path = case_dir / "sl_map_missing.csv"
    promoted_path = case_dir / "promoted_pairs.csv"
    write_sl_map(sl_map_path, specs, omit_upper_global=1001)
    write_promoted_pairs(promoted_path, specs, omit_line=100)
    write_seal(run_root)
    completed = invoke(
        run_root,
        out_dir,
        "--sl-map",
        str(sl_map_path),
        "--semi-state",
        "--promoted-pairs",
        str(promoted_path),
    )
    if completed.returncode != 0:
        raise AssertionError(
            "sl_map_missing_upper_global: expected successful descriptive "
            f"exclusion, exit={completed.returncode}\nstdout={completed.stdout}\n"
            f"stderr={completed.stderr}"
        )
    result = json.loads(
        (out_dir / "readout_rii_all.json").read_text(encoding="utf-8")
    )
    checks = {
        "matched pairs": result["pairs"]["matched_pair_count"] == 4,
        "eligible pairs": result["pairs"]["eligible_pair_count"] == 3,
        "missing pair exclusion": result["exclusions"]["by_reason"].get(
            "sl_map_upper_global_missing"
        )
        == 1,
        "missing ROW count": result["sl_map_application"][
            "application_counts"
        ]["upper_global_missing"]
        == 2,
        "fraction exact-one count": result["sl_map_application"][
            "application_counts"
        ]["within_sl_frac_set_to_one"]
        == 8,
        "fraction not transplanted": result["sl_map_application"][
            "within_sl_frac_transplanted"
        ]
        is False,
        "seal pre": result["seal_verification"]["pre"]["exit_status"] == 0,
        "seal post": result["seal_verification"]["post"]["exit_status"] == 0,
    }
    failed = [name for name, condition in checks.items() if not condition]
    if failed:
        raise AssertionError(
            f"sl_map_missing_upper_global assertions failed={failed}"
        )


def assert_sl_map_provenance_mismatch(case_dir: Path) -> None:
    specs = base_specs()
    run_root, out_dir = create_case(case_dir, render_log(specs))
    sl_map_path = case_dir / "sl_map_mismatch.csv"
    write_sl_map(sl_map_path, specs, runtime_override={1001: 99})
    completed = invoke(
        run_root,
        out_dir,
        "--sl-map",
        str(sl_map_path),
    )
    if completed.returncode != 2:
        raise AssertionError(
            "sl_map_provenance_mismatch: expected exit 2, got "
            f"{completed.returncode}\nstdout={completed.stdout}\n"
            f"stderr={completed.stderr}"
        )
    result = load_readout(out_dir)
    serialized_errors = json.dumps(
        result["validation"]["errors"], sort_keys=True, ensure_ascii=True
    )
    checks = {
        "reason retained": "SL_MAP_PROVENANCE_MISMATCH" in serialized_errors,
        "mismatch rows counted": result["sl_map_application"][
            "application_counts"
        ]["provenance_mismatch"]
        == 2,
        "self-validation failed": result["sl_map_application"][
            "self_validation"
        ]["all_rows_match"]
        is False,
    }
    failed = [name for name, condition in checks.items() if not condition]
    if failed:
        raise AssertionError(
            f"sl_map_provenance_mismatch assertions failed={failed}"
        )


def main() -> int:
    invocation_dir = Path.cwd().resolve()
    if not ANALYZER.is_file():
        print(f"analyzer missing: {ANALYZER}", file=sys.stderr)
        return 2

    scratch = Path(
        tempfile.mkdtemp(prefix=".selftest_readout_", dir=str(invocation_dir))
    ).resolve()
    if scratch.parent != invocation_dir:
        print(f"unsafe selftest scratch path: {scratch}", file=sys.stderr)
        return 2

    positive_count = 0
    negative_count = 0
    cases = negative_logs()
    expected_negative_count = len(cases) + 2
    try:
        assert_positive(scratch / "positive")
        positive_count = 1
        print("positive synthetic_readout: ok")
        for index, (name, log, expected_reason, expected_detail) in enumerate(cases):
            assert_negative(
                scratch / f"negative_{index}_{name}",
                name,
                log,
                expected_reason,
                expected_detail,
            )
            negative_count += 1
            detail = f" detail={expected_detail}" if expected_detail else ""
            print(f"negative {name}: expected={expected_reason}{detail} observed=ok")
        assert_sl_map_missing_exclusion(scratch / "negative_6_sl_map_missing")
        negative_count += 1
        print(
            "negative sl_map_missing_upper_global: "
            "expected=sl_map_upper_global_missing exclusion observed=ok"
        )
        assert_sl_map_provenance_mismatch(
            scratch / "negative_7_sl_map_provenance_mismatch"
        )
        negative_count += 1
        print(
            "negative sl_map_provenance_mismatch: "
            "expected=SL_MAP_PROVENANCE_MISMATCH observed=ok"
        )
    except AssertionError as exc:
        print(f"SELFTEST_ERROR {exc}", file=sys.stderr)
        print(
            f"SELFTEST positive={positive_count}/1 "
            f"negative={negative_count}/{expected_negative_count}"
        )
        return 1
    finally:
        shutil.rmtree(scratch)

    print(
        f"SELFTEST positive=1/1 "
        f"negative={negative_count}/{expected_negative_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
