#!/usr/bin/env python3
"""Dump and compare the archived pair baseline used by Wave-3 replay."""

import argparse
import csv
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--armed-dir", type=Path, required=True)
    parser.add_argument("--unarmed-dir", type=Path, required=True)
    parser.add_argument("--shell", type=int, required=True)
    parser.add_argument("--z", required=True, help="comma-separated Z list")
    return parser.parse_args()


def resolved_source(frozen):
    values = {}
    active = False
    with (frozen / "stdout.log").open() as handle:
        for raw in handle:
            if "=== RESOLVED CONFIG" in raw:
                active = True
                continue
            if not active:
                continue
            if "argv:" in raw:
                break
            line = raw.strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            # Match bench_frozen_oracle.c's sscanf parser and overrides.
            if not value or key in {"LUMINA_BIN", "OMP_NUM_THREADS"}:
                continue
            values[key] = value
    values["OMP_NUM_THREADS"] = "1"
    return values


def write_env(outdir, frozen, shell, zlist, armed):
    values = resolved_source(frozen)
    values["LUMINA_FROZEN_ORACLE_ONLY_SHELL"] = str(shell)
    if armed:
        values.update({
            "LUMINA_NLTE_ELEMENT_WIDE": "1",
            "LUMINA_NLTE_ELEMENT_WIDE_Z": zlist,
            "LUMINA_NLTE_ELEMENT_WIDE_SHELL": str(shell),
            "LUMINA_NLTE_ELEMENT_WIDE_COMMIT": "0",
            "LUMINA_NLTE_ELEMENT_WIDE_DUMP": "1",
            "LUMINA_NLTE_ELEMENT_WIDE_DUMP_DIR": str(outdir),
        })
    with (outdir / "resolved_env.txt").open("w", newline="") as handle:
        for key in sorted(values):
            handle.write(f"{key}={values[key]}\n")


def write_ions(outdir, frozen, shell, targets):
    selected = []
    totals = {}
    with (frozen / "lumina_ion_pops.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["shell_id"]) != shell:
                continue
            z = int(row["Z"])
            value = float(row["n_ion"])
            totals[z] = totals.get(z, 0.0) + value
            if z in targets and int(row["stage"]) in (1, 2, 3):
                selected.append((z, int(row["stage"]), value))
    with (outdir / "pair_ion_fractions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["shell", "Z", "stage", "n_ion_cm-3",
                         "n_element_cm-3", "ion_fraction"])
        for z, stage, value in sorted(selected):
            total = totals[z]
            writer.writerow([shell, z, stage, f"{value:.17e}",
                             f"{total:.17e}", f"{value / total:.17e}"])


def write_levels(outdir, frozen, shell, targets):
    with (frozen / "lumina_levelpop.csv").open(newline="") as source, \
         (outdir / "pair_level_populations.csv").open("w", newline="") as dest:
        reader = csv.DictReader(source)
        fields = ["shell", "Z", "ion", "level_num", "E_eV", "g", "n_k",
                  "n_ground", "b_k", "has_sigma", "n_sig_pos"]
        writer = csv.DictWriter(dest, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in reader:
            if (int(row["shell"]) != shell or
                    int(row["Z"]) not in targets or
                    int(row["ion"]) not in (1, 2)):
                continue
            writer.writerow({field: row[field] for field in fields})


def compare(armed_dir, unarmed_dir, shell):
    results = []
    specs = (
        ("pair_ion_fractions.csv",
         ("n_ion_cm-3", "n_element_cm-3", "ion_fraction")),
        ("pair_level_populations.csv", ("n_k", "n_ground", "b_k")),
    )
    for name, columns in specs:
        with (armed_dir / name).open(newline="") as handle:
            left = list(csv.DictReader(handle))
        with (unarmed_dir / name).open(newline="") as handle:
            right = list(csv.DictReader(handle))
        if left != right:
            byte_equal = 0
        else:
            byte_equal = int((armed_dir / name).read_bytes() ==
                             (unarmed_dir / name).read_bytes())
        max_abs = max_rel = max_dex = 0.0
        positive = 0
        if len(left) != len(right):
            raise ValueError(f"row-count mismatch for {name}")
        for a, b in zip(left, right):
            for column in columns:
                av, bv = float(a[column]), float(b[column])
                max_abs = max(max_abs, abs(av - bv))
                scale = max(abs(av), abs(bv))
                if scale > 0.0:
                    max_rel = max(max_rel, abs(av - bv) / scale)
                if av > 0.0 and bv > 0.0:
                    max_dex = max(max_dex, abs(math.log10(av / bv)))
                    positive += 1
        results.append((shell, name, len(left), len(right), byte_equal,
                        max_abs, max_rel, max_dex, positive))
    with (armed_dir / "comparison_summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["shell", "artifact", "armed_rows", "unarmed_rows",
                         "byte_equal", "max_abs_diff", "max_relative_diff",
                         "max_abs_dex", "positive_dex_values"])
        for row in results:
            writer.writerow([*row[:5], *(f"{value:.17e}" for value in row[5:8]),
                             row[8]])


def main():
    args = parse_args()
    targets = {int(item) for item in args.z.split(",")}
    for outdir, armed in ((args.armed_dir, True), (args.unarmed_dir, False)):
        if not outdir.is_dir():
            raise FileNotFoundError(outdir)
        write_env(outdir, args.frozen, args.shell, args.z, armed)
        write_ions(outdir, args.frozen, args.shell, targets)
        write_levels(outdir, args.frozen, args.shell, targets)
    compare(args.armed_dir, args.unarmed_dir, args.shell)


if __name__ == "__main__":
    main()
