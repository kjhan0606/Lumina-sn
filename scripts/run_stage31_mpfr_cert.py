#!/usr/bin/env python3
"""Fixed-precision directed-rounding replay for the Stage 3.1 KA3 recurrence."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE_GRIDS = ((128, 512, 2048), (256, 1024, 2048), (512, 2048, 2048))
FINE_GRID = (1024, 4096, 4096)


def read_metadata(path: pathlib.Path) -> dict[str, object]:
    line = path.read_text().strip()
    if not line.startswith("# "):
        raise ValueError(f"invalid certificate output: {path}")
    raw = dict(token.split("=", 1) for token in line[2:].split())
    integer_fields = {
        "ns", "nnu", "certificate_bits", "certified_sign_uncertain",
        "certified_nonfinite", "certified_negative", "first_k", "first_segment",
    }
    result: dict[str, object] = {}
    for key, value in raw.items():
        if key in integer_fields:
            result[key] = int(value)
        elif key in {"mpfr_version", "certified_min_lower", "certified_max_width"}:
            result[key] = value
        else:
            result[key] = float(value)
    return result


def compile_certificate(executable: pathlib.Path, prefix: pathlib.Path) -> list[str]:
    include = prefix / "include"
    library = prefix / "lib"
    header = include / "mpfr.h"
    if not header.is_file():
        raise FileNotFoundError(f"MPFR header not found: {header}")
    command = [
        "gcc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Wpedantic", "-Werror",
        "-Wconversion", "-Wshadow", "-isystem", str(include),
        "scripts/stage31_cmf_mpfr_cert.c", "-L" + str(library),
        "-Wl,-rpath," + str(library), "-lmpfr", "-lgmp", "-lm", "-o", str(executable),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    return command


def run(include_fine: bool, work: pathlib.Path, prefix: pathlib.Path) -> dict[str, object]:
    executable = work / "stage31_cmf_mpfr_cert"
    command = compile_certificate(executable, prefix)
    grids = BASE_GRIDS + ((FINE_GRID,) if include_fine else ())
    levels = []
    for ns, nnu, bits in grids:
        output = work / f"cert_{ns}_{nnu}_{bits}.txt"
        completed = subprocess.run(
            [str(executable), str(ns), str(nnu), str(bits), str(output)],
            cwd=ROOT, text=True, capture_output=True,
        )
        metadata = read_metadata(output)
        metadata.update({
            "exit_code": completed.returncode,
            "stderr": completed.stderr.strip(),
            "status": "PASS" if completed.returncode == 0 else "FAIL",
        })
        levels.append(metadata)
        if completed.returncode != 0:
            break
    checks = {
        "all_grids_completed": len(levels) == len(grids),
        "all_certified_sign_uncertain_zero":
            len(levels) == len(grids) and all(level["certified_sign_uncertain"] == 0 for level in levels),
        "all_certified_nonfinite_zero":
            len(levels) == len(grids) and all(level["certified_nonfinite"] == 0 for level in levels),
        "all_certified_negative_zero":
            len(levels) == len(grids) and all(level["certified_negative"] == 0 for level in levels),
        "fixed_precision_exact":
            len(levels) == len(grids) and
            all(level["certificate_bits"] == grid[2] for level, grid in zip(levels, grids)),
    }
    passed = all(checks.values())
    return {
        "rung": 8,
        "method": "MPFR directed-rounding interval replay",
        "production_double_touched": False,
        "grids": [{"ns": ns, "nnu": nnu, "bits": bits} for ns, nnu, bits in grids],
        "levels": levels,
        "checks": checks,
        "environment": {
            "prefix": str(prefix),
            "mpfr_header": str(prefix / "include" / "mpfr.h"),
            "compile_command": command,
            "rpath": str(prefix / "lib"),
        },
        "acceptance_unchanged": True,
        "status": "PASS" if passed else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-fine", action="store_true")
    parser.add_argument("--work", type=pathlib.Path, default=pathlib.Path("/tmp/stage31_mpfr_cert"))
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--mpfr-prefix", type=pathlib.Path,
                        default=pathlib.Path.home() / "local")
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    report = run(args.include_fine, args.work, args.mpfr_prefix)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
