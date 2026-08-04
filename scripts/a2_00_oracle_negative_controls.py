#!/usr/bin/env python3
"""Run the six A2-00 injected-fault controls on a small synthetic CMFGEN fixture."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile


def direct_records(path: Path, record_size: int, records: list[bytes]) -> None:
    with path.open("wb") as stream:
        for value in records:
            if len(value) > record_size:
                raise ValueError("record exceeds RECL")
            stream.write(value)
            stream.write(b"\0" * (record_size - len(value)))


def write_fixture(root: Path, variant: float = 0.0) -> None:
    root.mkdir(parents=True)
    nd = 4
    ncf = 100
    edd_recl = 8 * (nd + 1)
    jh_recl = 8 * (2 * nd + 2)
    (root / "EDDFACTOR_INFO").write_text(
        " 12-Apr-2017 !INFO format date\n"
        " 23-Jan-2017 !File format date\n"
        f" {nd:11d}{edd_recl:12d}{8:12d}{1:12d}{4:12d}           T\n"
        "          ND        RECL   WORD_SIZE   UNIT_SIZE    INT_SIZE     LIT_END\n",
        encoding="ascii",
    )
    (root / "JH_AT_CURRENT_TIME_INFO").write_text(
        " 12-Apr-2017 !INFO format date\n"
        " 10-Jul-2006 !File format date\n"
        f" {nd:11d}{jh_recl:12d}{8:12d}{1:12d}{4:12d}           T\n"
        "          ND        RECL   WORD_SIZE   UNIT_SIZE    INT_SIZE     LIT_END\n",
        encoding="ascii",
    )
    radii = [10.0, 9.0, 8.0, 7.0]
    velocity = [4.0, 3.0, 2.0, 1.0]
    edd = [b""] * (14 + ncf)
    edd[1] = struct.pack("<i", 10)
    edd[4] = struct.pack("<d", 1.0)
    edd[9] = struct.pack("<4d", *radii)
    edd[10] = struct.pack("<4d", *velocity)
    edd[11] = struct.pack("<4d", 0.0, 0.0, 0.0, 0.0)
    for index in range(ncf):
        values = [1.0 + variant + index * 0.01 + depth * 0.001 for depth in range(nd)]
        frequency = 20.0 - index * 0.1
        edd[14 + index] = struct.pack("<5d", *values, frequency)
    direct_records(root / "EDDFACTOR", edd_recl, edd)
    jh = [b""] * (7 + ncf)
    jh[2] = struct.pack("<3i", 6, ncf, nd)
    jh[5] = struct.pack("<8d", *(radii + velocity))
    jh[6] = struct.pack("<10d", *([0.0] * 10))
    for index in range(ncf):
        values = [1.0 + variant + index * 0.01 + depth * 0.001 for depth in range(nd)]
        rsqj = [values[depth] * radii[depth] * radii[depth] for depth in range(nd)]
        frequency = 20.0 - index * 0.1
        jh[7 + index] = struct.pack(
            "<10d", *(rsqj + [0.0] * (nd - 1) + [0.0, 0.0, frequency])
        )
    direct_records(root / "JH_AT_CURRENT_TIME", jh_recl, jh)
    (root / "OUTGEN").write_text(
        " Number of frequencies is:                 100\n"
        " Current great iteration count is  63\n"
        " Maximum % increase at depth 1 is 7.50E+06 (LAMBDA) --- iteration 63\n"
        " Current great iteration count is  64\n"
        " Maximum % increase at depth 1 is 7.97E+05 (LAMBDA) --- iteration 64\n"
        " Current great iteration count is  65\n"
        " Maximum % increase at depth 1 is 3.69E+05 (LAMBDA) --- iteration 65\n"
        " Current great iteration count is  66\n"
        " Maximum % increase at depth 1 is 3.52E+05 (LAMBDA) --- iteration 66\n",
        encoding="ascii",
    )
    completion = "04-Aug-2026 00:00:00"
    vector = lambda values: " ".join(f"{value:.10E}" for value in values) + "\n"
    (root / "RVTJ").write_text(
        " Output format date:         15-Aug-2019\n"
        f" Completion of Model:        {completion}\n"
        f" ND:                         {nd}\n"
        " NCF:                        5\n"
        "Radius (10^10 cm)\n"
        + vector(radii)
        + "Velocity (km/s)\n"
        + vector(velocity)
        + "Electron density\n"
        + vector([4.0e8, 3.0e8, 2.0e8, 1.0e8])
        + "Temperature (10^4K)\n"
        + vector([1.4, 1.3, 1.2, 1.1]),
        encoding="ascii",
    )
    (root / "POPCAL").write_text(
        " Output format date:         27-JAN-1992\n"
        f" Completion of Model:        {completion}\n"
        f" ND:                         {nd}\n"
        " CAL/He abundance:           1.0E-3\n"
        + vector([1.0, 1.0, 1.0, 1.0]),
        encoding="ascii",
    )
    common = (
        "   Depth index\n"
        "            1           2           3           4\n"
        "   Radius [1.0E+10cm]\n"
        + vector(radii)
        + "   Temperature [1.0E+4K]\n"
        + vector([1.4, 1.3, 1.2, 1.1])
        + "   Electron Density\n"
        + vector([4.0e8, 3.0e8, 2.0e8, 1.0e8])
    )
    (root / "Ca2PRRR").write_text(common, encoding="ascii")
    (root / "GENCOOL").write_text(
        "   Depth\n"
        "            1           2           3           4\n"
        "   Radius [1.0E+10cm]\n"
        + vector(radii)
        + "   Velocity [km/s]\n"
        + vector(velocity)
        + "   Temperature [1.0E+4K]\n"
        + vector([1.4, 1.3, 1.2, 1.1])
        + "   Electron Density\n"
        + vector([4.0e8, 3.0e8, 2.0e8, 1.0e8]),
        encoding="ascii",
    )
    obs = [9.0, 8.0, 7.0, 6.0, 5.0]
    (root / "OBSFLUX").write_text(
        "\n Continuum Frequencies ( 5 )\n" + vector(obs), encoding="ascii"
    )
    (root / "OBS_FREQ").write_text(
        "".join(f" {obs[index]:.6E}   1.000\n" for index in range(4)),
        encoding="ascii",
    )
    (root / "run_jnu4.info").write_text(
        "RUN=synthetic (NUM_ITS=4, restart record 62, stop before it67 NaN)\n"
        "T [FIX_T]\n"
        "4 [NUM_ITS]\n"
        "T [DO_LAM_IT]\n"
        "POINT1: 62 62 1 -1000 F\n",
        encoding="ascii",
    )


def invoke(tool: Path, mode: str, root: Path, manifest: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(tool), mode, str(root), "--manifest", str(manifest), *extra],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scratch-root",
        type=Path,
        help="optional parent; a unique a2_00_* directory is still created",
    )
    args = parser.parse_args()
    tool = Path(__file__).with_name("cmfgen_oracle_contract.py").resolve()
    parent = args.scratch_root.resolve() if args.scratch_root else None
    scratch = Path(tempfile.mkdtemp(prefix="a2_00_oracle_controls_", dir=parent))
    print(f"SCRATCH_COPY_ROOT={scratch}")
    base = scratch / "baseline_run"
    alternate = scratch / "alternate_run"
    write_fixture(base)
    write_fixture(alternate, variant=0.25)
    manifest = scratch / "baseline_manifest.json"
    written = invoke(tool, "write", base, manifest, "--no-generation-scan")
    if written.returncode != 0:
        print(written.stdout, end="")
        print(f"FAIL setup write rc={written.returncode}")
        return 1
    baseline = invoke(tool, "check", base, manifest)
    if baseline.returncode != 0:
        print(baseline.stdout, end="")
        print(f"FAIL baseline check rc={baseline.returncode}")
        return 1

    cases: list[tuple[str, int, list[str]]] = []

    def fresh(name: str) -> Path:
        target = scratch / name
        shutil.copytree(base, target, copy_function=shutil.copy2)
        return target

    case = fresh("neg01_deleted")
    (case / "POPCAL").unlink()
    result = invoke(tool, "check", case, manifest)
    cases.append(("delete_one_file", result.returncode, ["MISSING_PATH POPCAL"]))

    case = fresh("neg02_truncated")
    target = case / "EDDFACTOR"
    with target.open("r+b") as stream:
        stream.truncate(target.stat().st_size - 1024)
    result = invoke(tool, "check", case, manifest)
    cases.append(
        (
            "truncate_1024_bytes",
            result.returncode,
            [
                "SIZE_MISMATCH EDDFACTOR",
                "HASH_MISMATCH EDDFACTOR",
                "RECORD_SCHEMA_FATAL",
            ],
        )
    )

    case = fresh("neg03_other_run")
    shutil.copy2(alternate / "EDDFACTOR", case / "EDDFACTOR")
    result = invoke(tool, "check", case, manifest)
    cases.append(("replace_from_other_run", result.returncode, ["HASH_MISMATCH EDDFACTOR"]))

    case = fresh("neg04_mtime_only")
    target = case / "POPCAL"
    old = target.stat()
    os.utime(target, ns=(old.st_atime_ns, old.st_mtime_ns + 86_400_000_000_000))
    result = invoke(tool, "check", case, manifest)
    cases.append(("mtime_only_must_pass", result.returncode, ["MTIME_CHANGED_IGNORED POPCAL"]))

    case = fresh("neg05_info_nd_plus_one")
    info = case / "EDDFACTOR_INFO"
    lines = info.read_text(encoding="ascii").splitlines()
    values = lines[2].split()
    values[0] = str(int(values[0]) + 1)
    lines[2] = " ".join(values)
    info.write_text("\n".join(lines) + "\n", encoding="ascii")
    result = invoke(tool, "check", case, manifest)
    cases.append(("info_declared_nd_plus_one", result.returncode, ["RECORD_SCHEMA_FATAL"]))

    case = fresh("neg06_unclassified")
    (case / "A2_UNKNOWN_PAYLOAD").write_bytes(b"unknown\n")
    result = invoke(tool, "check", case, manifest)
    cases.append(("add_unclassified_file", result.returncode, ["UNCLASSIFIED_EXTRA"]))

    expected_codes = [11, 14, 13, 0, 14, 15]
    failures = 0
    for (name, observed, markers), expected in zip(cases, expected_codes):
        case_root = scratch / {
            "delete_one_file": "neg01_deleted",
            "truncate_1024_bytes": "neg02_truncated",
            "replace_from_other_run": "neg03_other_run",
            "mtime_only_must_pass": "neg04_mtime_only",
            "info_declared_nd_plus_one": "neg05_info_nd_plus_one",
            "add_unclassified_file": "neg06_unclassified",
        }[name]
        rerun = invoke(tool, "check", case_root, manifest)
        marker_ok = all(marker in rerun.stdout for marker in markers)
        passed = observed == expected and marker_ok
        failures += not passed
        print(
            f"CONTROL {name} expected_rc={expected} observed_rc={observed} "
            f"markers={'OK' if marker_ok else 'MISSING'} {'PASS' if passed else 'FAIL'}"
        )

    ophys = invoke(tool, "check", base, manifest, "--profile", "ophys")
    ophys_ok = ophys.returncode == 16 and "MISSING_REQUIRED_FILE:NETRATE" in ophys.stdout
    failures += not ophys_ok
    print(
        f"POSITIVE_CONTROL current_like_snapshot_ophys_failure expected_rc=16 "
        f"observed_rc={ophys.returncode} {'PASS' if ophys_ok else 'FAIL'}"
    )
    print(f"SUMMARY controls_passed={7 - failures}/7 failures={failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
