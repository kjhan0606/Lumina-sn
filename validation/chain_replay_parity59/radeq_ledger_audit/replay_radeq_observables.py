#!/usr/bin/env python3
"""Replay directly observable parity59 RADEQ facts and field-side ledger values.

This is a parameterized, current-safe replacement for the light portion of
radeq_ledger.py.  It does not claim to reconstruct parity59's internal
counterfactual roots because the legacy script predates DB_FB/BF_RATE_POPS and
the capture does not dump the trial-temperature term tables.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import integrate_j, load_field, write_csv

A_RAD = 7.5657e-15


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plasma_path = args.input_dir / "lumina_plasma_state.csv"
    field_path = args.input_dir / "lumina_coevolve_field.csv"
    stdout_path = args.input_dir / "stdout.log"
    env_path = args.input_dir / "PARITY59_INSTR.env"
    for path in (plasma_path, field_path, stdout_path, env_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    with plasma_path.open() as handle:
        plasma = {int(row["shell_id"]): row for row in csv.DictReader(handle)}
    p0 = plasma[0]
    te = float(p0["T_e"])
    ne = float(p0["n_e"])
    field = load_field(field_path)[0]
    u_mc = integrate_j(field["wavelength_A"], field["mc_J"])
    u_cs = integrate_j(field["wavelength_A"], field["cs_J"])
    bath_t = (u_mc / A_RAD) ** 0.25

    history_rows: list[list[object]] = []
    pattern = re.compile(r"\[TEHOLD\] s0: T_e=(\d+)K \(prev=(\d+)K\) radeq_root=(\S+)")
    all_root_lines = 0
    all_root_found = 0
    for line_number, line in enumerate(stdout_path.read_text().splitlines(), 1):
        if "[TEHOLD] s" in line:
            all_root_lines += 1
            if "radeq_root=root-found" in line:
                all_root_found += 1
        match = pattern.search(line)
        if match:
            history_rows.append([len(history_rows) + 1, int(match.group(2)), int(match.group(1)), match.group(3),
                                 str(stdout_path), line_number, "[TEHOLD] s0 T_e; prev; radeq_root"])
    if not history_rows:
        raise ValueError("no s0 TEHOLD history found")

    env: dict[str, str] = {}
    for line in env_path.read_text().splitlines():
        if line.startswith("export ") and "=" in line:
            key, value = line[7:].split("=", 1)
            env[key] = value
    gates = ["LUMINA_RADEQ_TE", "LUMINA_RADEQ_SIMUL", "LUMINA_RADEQ_DAMP",
             "LUMINA_RADEQ_FB_RATE", "LUMINA_RADEQ_VR_STD", "LUMINA_RADEQ_DB_FB",
             "LUMINA_BF_RATE_POPS", "LUMINA_TE_STEP_CLAMP", "LUMINA_ETLA_ALLOW_HEAT"]
    gate_rows = [[key, env.get(key, "ABSENT"), str(env_path), f"export {key}"] for key in gates]

    summary_rows = [["s0_T_e", te, "K", str(plasma_path), "T_e"],
                    ["s0_n_e", ne, "cm^-3", str(plasma_path), "n_e"],
                    ["s0_u_mc", u_mc, "erg cm^-3", str(field_path), "wavelength_A; mc_J"],
                    ["s0_u_cs", u_cs, "erg cm^-3", str(field_path), "wavelength_A; cs_J"],
                    ["s0_bath_equivalent_T_from_mc", bath_t, "K", str(field_path), "wavelength_A; mc_J"],
                    ["TEHOLD_records_all_shells", all_root_lines, "count", str(stdout_path), "[TEHOLD]"],
                    ["TEHOLD_root_found_all_shells", all_root_found, "count", str(stdout_path), "radeq_root"]]
    write_csv(args.output_dir / "radeq_observables.csv", ["quantity", "value", "unit", "source_file", "source_field"], summary_rows)
    write_csv(args.output_dir / "s0_te_history.csv", ["solve_index", "prev_T_e_K", "committed_T_e_K", "radeq_root",
                                                        "source_file", "source_line", "source_fields"], history_rows)
    write_csv(args.output_dir / "radeq_gate_snapshot.csv", ["gate", "value", "source_file", "source_field"], gate_rows)
    unresolved = [
        ["parity59 zero-pump coupled root", "UNRESOLVED", "trial-T nion/Gph/Hex/emit_bf/ETLA table",
         "not dumped; 07-15 estimator predates DB_FB and BF_RATE_POPS"],
        ["parity59 own-cs.J coupled root", "UNRESOLVED", "trial-T nion/Gph/Hex/emit_bf/ETLA table",
         "not dumped; no substitute or legacy-formula relabeling"],
        ["parity59 CMFGEN-J coupled root", "UNRESOLVED", "same internal trial table plus field swap",
         "cannot be derived exactly from final CSV observables"],
        ["current decomposition levers for the historical 5640 K deficit", "NOT_APPLICABLE",
         "current deficit is -2467.639444 K (truth minus capture)",
         "endpoint sign reversed; historical lever magnitudes are retained only as 07-15 values"],
    ]
    write_csv(args.output_dir / "UNRESOLVED.csv", ["quantity", "status", "required_input", "reason"], unresolved)


if __name__ == "__main__":
    main()
