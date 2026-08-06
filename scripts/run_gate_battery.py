#!/usr/bin/env python3
"""Build and run the D + K + Z-INERT + CONFIG-PREC gate battery."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from generate_composition_d_fixtures import cached_materialize


ROOT = Path(__file__).resolve().parents[1]
BATTERY_ORDER = ("D", "K", "Z", "CP")
EXPECTED_ROWS = {"D": 19, "K": 7, "Z": 10, "CP": 4}


def run_census_preflight() -> int:
    """Fail before expensive builds when the A2-01 anchor ledger is stale."""
    command = (
        sys.executable,
        str(ROOT / "scripts/a2_01_census_contract.py"),
        "check",
    )
    proc = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=False
    )
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    print(f"PREFLIGHT name=A2_01_CENSUS rc={proc.returncode}")
    return proc.returncode


def cpu_link_sources() -> tuple[str, ...]:
    """All src/*.c except translation units that define their own main()."""
    import glob as _glob
    import re as _re
    sources = []
    for path in sorted(_glob.glob("src/*.c")):
        text = open(path, encoding="utf-8", errors="replace").read()
        if path.endswith("/lumina_main.c"):
            continue
        if _re.search(r"^\s*int\s+main\s*\(", text, _re.M):
            continue
        sources.append(path)
    return tuple(sources)


class Tee:
    def __init__(self, streams: tuple[TextIO, ...], lock: threading.Lock):
        self.streams = streams
        self.lock = lock

    def write(self, value: str) -> int:
        with self.lock:
            for stream in self.streams:
                stream.write(value)
                stream.flush()
        return len(value)

    def flush(self) -> None:
        with self.lock:
            for stream in self.streams:
                stream.flush()


@dataclass(frozen=True)
class Build:
    name: str
    command: tuple[str, ...]
    output: Path


@dataclass(frozen=True)
class Run:
    battery: str
    command: tuple[str, ...]


@dataclass(frozen=True)
class RunResult:
    battery: str
    returncode: int
    lines: tuple[str, ...]


def build_specs(build: Path, cc: str) -> tuple[Build, ...]:
    common_z = (
        cc, "-O2", "-w", "-std=gnu11", "-D_GNU_SOURCE",
        "-ffunction-sections", "-fdata-sections", "-Isrc",
        "-Wl,--gc-sections",
    )
    return (
        Build(
            "D",
            (
                cc, "-O2", "-Wall", "-Wextra", "-std=c11",
                "-ffunction-sections", "-fdata-sections", "-Isrc",
                "-Wl,--gc-sections", "-o", str(build / "composition_d_harness"),
                "scripts/composition_d_harness.c", "src/lumina_atomic.c", "-lm",
            ),
            build / "composition_d_harness",
        ),
        Build(
            "K",
            (
                cc, "-O2", "-std=c11", "-Isrc", "-o",
                str(build / "kshape_harness"), "scripts/kshape_harness.c",
                # driver fix 2026-08-06: plasma/cmfgen now reference the
                # radiation-field commit API; link the derived non-main set.
                *cpu_link_sources(),
                "-lm", "-fopenmp",
            ),
            build / "kshape_harness",
        ),
        Build(
            "CP",
            (
                cc, "-O2", "-Wall", "-Wextra", "-std=c11", "-o",
                str(build / "lumina"), "src/lumina_main.c",
                # 2026-08-06 driver fix (3rd recurrence of stale hardcoded
                # lists): link every non-main C translation unit, derived at
                # runtime, so new A-2 stages cannot silently break this gate.
                *cpu_link_sources(),
                "-lm", "-fopenmp",
            ),
            build / "lumina",
        ),
        Build(
            "Z-validator",
            common_z + (
                "tests/abundance_zero_nlte_fixture.c", "src/lumina_plasma.c",
                "src/bf_rate_jnu.c", "src/population_contract.c",
                "src/opacity_publication.c",
                "src/emissivity_publication.c",
                "src/radeq_publication.c",
                "src/gpu_radiation_field_contract.c",
                "-lm", "-o", str(build / "zinert_validator"),
            ),
            build / "zinert_validator",
        ),
        Build(
            "Z-tau",
            common_z + (
                "-DLUMINA_FROZEN_ORACLE", "tests/zinert_tau_fixture.c",
                "src/lumina_plasma.c", "src/bf_rate_jnu.c", "src/population_contract.c", "src/opacity_publication.c", "src/emissivity_publication.c", "src/radeq_publication.c", "src/gpu_radiation_field_contract.c", "-lm", "-o", str(build / "zinert_tau"),
            ),
            build / "zinert_tau",
        ),
        Build(
            "Z-population",
            common_z + (
                "tests/zinert_population_fixture.c", "src/lumina_plasma.c",
                "src/bf_rate_jnu.c", "src/population_contract.c",
                "src/opacity_publication.c",
                "src/emissivity_publication.c",
                "src/radeq_publication.c",
                "src/gpu_radiation_field_contract.c",
                "-lm", "-o", str(build / "zinert_population"),
            ),
            build / "zinert_population",
        ),
        Build(
            "Z-canonical",
            common_z + (
                "-DLUMINA_FROZEN_ORACLE", "tests/zinert_canonical_tau_fixture.c",
                "src/lumina_plasma.c", "src/lumina_element_wide.c",
                "src/bf_rate_jnu.c", "src/population_contract.c",
                "src/lumina_atomic.c", "src/opacity_publication.c", "src/emissivity_publication.c", "src/radeq_publication.c", "src/gpu_radiation_field_contract.c", "-lm", "-o",
                str(build / "zinert_canonical_tau"),
            ),
            build / "zinert_canonical_tau",
        ),
        Build(
            "Z-a2-12",
            common_z + (
                "tests/a2_12_contract_selftest.c",
                "src/gpu_radiation_field_contract.c", "-o",
                str(build / "a2_12_contract"),
            ),
            build / "a2_12_contract",
        ),
        Build(
            "Z-a2-08",
            common_z + (
                "tests/a2_08_signed_opacity_selftest.c",
                "src/opacity_publication.c", "-lm", "-o",
                str(build / "a2_08_signed_opacity"),
            ),
            build / "a2_08_signed_opacity",
        ),
        Build(
            "Z-a2-09",
            common_z + (
                "tests/a2_09_emissivity_selftest.c",
                "src/emissivity_publication.c", "-lm", "-o",
                str(build / "a2_09_emissivity"),
            ),
            build / "a2_09_emissivity",
        ),
        Build(
            "Z-a2-10",
            common_z + (
                "tests/a2_10_radeq_selftest.c", "src/radeq_publication.c",
                "src/population_contract.c", "-lm", "-o",
                str(build / "a2_10_radeq"),
            ),
            build / "a2_10_radeq",
        ),
    )


def compile_one(spec: Build) -> tuple[Build, subprocess.CompletedProcess[str]]:
    proc = subprocess.run(
        spec.command, cwd=ROOT, text=True, capture_output=True, check=False
    )
    return spec, proc


def build_all(build: Path, base: Path, cache_root: Path) -> Path:
    specs = build_specs(build, os.environ.get("CC", "gcc"))
    fixture_path: Path | None = None
    failures = 0
    with ThreadPoolExecutor(max_workers=len(specs) + 1) as pool:
        futures = {pool.submit(compile_one, spec): spec.name for spec in specs}
        cache_future = pool.submit(cached_materialize, base, cache_root)
        for future in as_completed(futures):
            spec, proc = future.result()
            print(f"BUILD name={spec.name} rc={proc.returncode}")
            if proc.returncode != 0:
                failures += 1
                print((proc.stdout + proc.stderr).rstrip())
        fixture_path, hit, key = cache_future.result()
        print(
            f"FIXTURE_CACHE status={'HIT' if hit else 'MISS'} "
            f"key={key} path={fixture_path}"
        )
    if failures:
        raise RuntimeError(f"{failures} gate build(s) failed")
    return fixture_path


def runner_specs(
    build: Path, fixtures: Path, scratch: Path, *, serial: bool
) -> tuple[Run, ...]:
    serial_arg = ("--serial",) if serial else ()
    deck = ROOT / "data/tardis_reference_toy06_19p48d"
    return (
        Run(
            "D",
            (
                sys.executable, str(ROOT / "scripts/run_composition_d_gate.py"),
                "--harness", str(build / "composition_d_harness"),
                "--fixtures", str(fixtures), "--canonical", str(deck),
                "--scratch-root", str(scratch / "D"),
            ) + serial_arg,
        ),
        Run(
            "K",
            (
                sys.executable, str(ROOT / "scripts/run_k_gate.py"),
                "--loader", str(build / "kshape_harness"),
                "--positive-deck",
                str(ROOT / "data/tardis_reference_toy06_19p48d_sivcaiv"),
                "--negative-deck", str(deck),
                "--scratch-root", str(scratch / "K"),
            ) + serial_arg,
        ),
        Run(
            "Z",
            (
                sys.executable, str(ROOT / "scripts/run_zinert_selftest.py"),
                "--validator", str(build / "zinert_validator"),
                "--tau", str(build / "zinert_tau"),
                "--population", str(build / "zinert_population"),
                "--canonical-tau", str(build / "zinert_canonical_tau"),
                "--a2-08", str(build / "a2_08_signed_opacity"),
                "--a2-09", str(build / "a2_09_emissivity"),
                "--a2-10", str(build / "a2_10_radeq"),
                "--a2-12-contract", str(build / "a2_12_contract"),
                "--deck", str(deck), "--verify", str(ROOT / "scripts/verify_zinert.py"),
                "--scratch-root", str(scratch / "Z"),
            ) + serial_arg,
        ),
        Run(
            "CP",
            (
                sys.executable,
                str(ROOT / "scripts/run_config_prec_negative_controls.py"),
                "--binary", str(build / "lumina"), "--deck", str(deck),
                "--scratch-root", str(scratch / "CP"),
            ) + serial_arg,
        ),
    )


def collect(run: Run, output_lock: threading.Lock) -> RunResult:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        run.command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert proc.stdout is not None
    lines: list[str] = []
    for line in proc.stdout:
        lines.append(line)
        if line.startswith("PROGRESS "):
            with output_lock:
                print(line, end="")
    return RunResult(run.battery, proc.wait(), tuple(lines))


def run_all(
    build: Path,
    fixtures: Path,
    scratch: Path,
    *,
    serial: bool,
    render: bool,
) -> tuple[int, dict[str, RunResult]]:
    runs = runner_specs(build, fixtures, scratch, serial=serial)
    lock = threading.Lock()
    results: dict[str, RunResult] = {}
    if serial:
        for run in runs:
            results[run.battery] = collect(run, lock)
    else:
        with ThreadPoolExecutor(max_workers=len(runs)) as pool:
            futures = {pool.submit(collect, run, lock): run.battery for run in runs}
            for future in as_completed(futures):
                result = future.result()
                results[result.battery] = result

    if render:
        for battery in BATTERY_ORDER:
            result = results[battery]
            print(f"===== BATTERY {battery} rc={result.returncode} =====")
            for line in result.lines:
                if not line.startswith("PROGRESS "):
                    print(line, end="")
    complete = all(
        sum(line.startswith("RESULT battery=") for line in results[battery].lines)
        == EXPECTED_ROWS[battery]
        for battery in BATTERY_ORDER
    )
    return (
        0 if complete and all(result.returncode == 0 for result in results.values())
        else 1,
        results,
    )


def result_table(results: dict[str, RunResult]) -> str:
    return "".join(
        line
        for battery in BATTERY_ORDER
        for line in results[battery].lines
        if line.startswith("RESULT battery=")
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", action="store_true")
    parser.add_argument("--verify-equivalence", action="store_true")
    parser.add_argument(
        "--cache-root", type=Path,
        default=Path("/tmp/lumina-composition-d-fixture-cache"),
    )
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()
    if args.serial and args.verify_equivalence:
        parser.error("--serial and --verify-equivalence are mutually exclusive")

    log_stream = None
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        log_stream = args.log.open("w", encoding="utf-8")
        lock = threading.Lock()
        sys.stdout = Tee((sys.stdout, log_stream), lock)  # type: ignore[assignment]
        sys.stderr = Tee((sys.stderr, log_stream), lock)  # type: ignore[assignment]

    scratch_context = None
    try:
        if args.scratch_root:
            scratch = args.scratch_root.resolve()
            scratch.mkdir(parents=True, exist_ok=False)
        else:
            scratch_context = tempfile.TemporaryDirectory(
                prefix="lumina-gate-battery-", dir="/tmp"
            )
            scratch = Path(scratch_context.name)
        build = scratch / "build"
        build.mkdir()
        base = ROOT / "data/tardis_reference_toy06_19p48d"
        print(
            f"GATE_BATTERY node={os.uname().nodename} cpu_count={os.cpu_count()} "
            f"scratch={scratch}"
        )
        census_rc = run_census_preflight()
        if census_rc != 0:
            print("GATE_BATTERY_SUMMARY verdict=FAIL rc=1 preflight=A2_01_CENSUS")
            return 1
        fixtures = build_all(build, base, args.cache_root.resolve())

        if args.verify_equivalence:
            serial_rc, serial_results = run_all(
                build, fixtures, scratch / "serial", serial=True, render=False
            )
            parallel_rc, parallel_results = run_all(
                build, fixtures, scratch / "parallel", serial=False, render=False
            )
            serial_table = result_table(serial_results)
            parallel_table = result_table(parallel_results)
            identical = serial_table == parallel_table
            print("===== SERIAL/PARALLEL RESULT TABLE =====")
            print(serial_table, end="")
            print(
                "EQUIVALENCE "
                f"serial_rc={serial_rc} parallel_rc={parallel_rc} "
                f"table={'IDENTICAL' if identical else 'DIFFERENT'}"
            )
            if not identical:
                print("===== PARALLEL RESULT TABLE =====")
                print(parallel_table, end="")
            return 0 if serial_rc == parallel_rc == 0 and identical else 1

        rc, _results = run_all(
            build, fixtures, scratch / ("serial" if args.serial else "parallel"),
            serial=args.serial,
            render=True,
        )
        print(f"GATE_BATTERY_SUMMARY verdict={'PASS' if rc == 0 else 'FAIL'} rc={rc}")
        return rc
    finally:
        if scratch_context is not None:
            scratch_context.cleanup()
        if log_stream is not None:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_stream.close()


if __name__ == "__main__":
    raise SystemExit(main())
