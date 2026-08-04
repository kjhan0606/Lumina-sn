#!/usr/bin/env python3
"""Audit the E1 assembler replay against the parity59 chi/eta capture.

This is deliberately an audit, not a repair-by-subtraction.  It reports the
channels actually serialized by LCMFCE01 and refuses to claim that the missing
pre-EPAY line/continuum terms can be recovered from their aggregate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from cmf_chieta_check import check_artifact  # noqa: E402

C_ANGSTROM = 2.99792458e18
DEFAULT_RUN = Path(
    "/gpfs/kjhan/lumina_runner2/scratch/chieta_capture_parity59_188605"
)
CAPTURE_SOURCE_HASHES = (
    ROOT / "docs/CODEX_CAPTURE_BINARY_CERT_LOGS_2026-08-01/"
    "src_sha256_before.txt"
)
FIELDS = {
    "chi_total": 3,
    "chi_coherent": 4,
    "eta_fixed": 5,
    "eta_coherent": 6,
    "eta_total": 7,
    "J": 8,
}
BANDS = (
    ("100-600", 100.0, 600.0),
    ("600-1000", 600.0, 1000.0),
    ("1000-1500", 1000.0, 1500.0),
    ("1500-2000", 1500.0, 2000.0),
    ("2000-2500", 2000.0, 2500.0),
    ("2500-3000", 2500.0, 3000.0),
    ("3000-20000", 3000.0, 20000.0),
    ("all", 0.0, float("inf")),
)


def rel_l1(candidate: np.ndarray, authority: np.ndarray) -> float:
    denominator = np.abs(authority).sum(dtype=np.longdouble)
    numerator = np.abs(candidate - authority).sum(dtype=np.longdouble)
    return float(numerator / denominator) if denominator > 0 else float("nan")


def compare(candidate: np.ndarray, authority: np.ndarray) -> dict:
    delta = np.abs(candidate - authority)
    peak = float(np.max(np.abs(authority)))
    return {
        "relative_l1": rel_l1(candidate, authority),
        "relative_linf_over_authority_peak": (
            float(np.max(delta)) / peak if peak > 0.0 else float("nan")
        ),
        "max_absolute": float(np.max(delta)),
        "bitwise_equal_cells": int(np.count_nonzero(
            candidate.view(np.uint64) == authority.view(np.uint64)
        )),
        "cells": int(authority.size),
    }


def parse_resolved_config(stdout: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    active = False
    for line in stdout.read_text(errors="strict").splitlines():
        if line == "=== RESOLVED CONFIG (env seen by binary) ===":
            active = True
            continue
        if active and line.startswith("argv:"):
            break
        if active:
            match = re.fullmatch(r"  ([A-Z][A-Z0-9_]*)=(.*)", line)
            if match:
                values[match.group(1)] = match.group(2)
    return values


def source_hash_audit() -> dict:
    expected: dict[str, str] = {}
    for line in CAPTURE_SOURCE_HASHES.read_text().splitlines():
        digest, path = line.split(maxsplit=1)
        expected[path] = digest
    result = {}
    for path in ("src/lumina_cmfgen.c", "src/lumina_cuda.cu",
                 "src/lumina_atomic.c", "src/lumina_plasma.c", "src/lumina.h"):
        current = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        result[path] = {
            "capture_sha256": expected[path],
            "current_sha256": current,
            "equal": current == expected[path],
        }
    return result


def epoch_temperature_audit(stdout_lines: list[str], plasma_csv: Path) -> dict:
    """Compare the rounded state feeding iter 10 with the post-loop CSV."""
    iter9 = next(i for i, line in enumerate(stdout_lines)
                 if "[CMFGEN] iter  9:" in line)
    capture_rounded: dict[int, float] = {}
    pattern = re.compile(r"\[TEHOLD\] s(\d+): T_e=([0-9]+)K")
    for line in reversed(stdout_lines[:iter9]):
        match = pattern.search(line)
        if match:
            capture_rounded.setdefault(int(match.group(1)), float(match.group(2)))
        if len(capture_rounded) == 50:
            break
    if set(capture_rounded) != set(range(50)):
        raise ValueError("could not recover the 50 rounded temperatures feeding iter 10")
    final = np.loadtxt(plasma_csv, delimiter=",", skiprows=1)
    if final.shape != (50, 5) or not np.array_equal(final[:, 0], np.arange(50)):
        raise ValueError("unexpected final plasma CSV")
    before = np.asarray([capture_rounded[s] for s in range(50)])
    after = final[:, 4]
    relative = np.abs(after - before) / before
    return {
        "capture_input": "rounded TEHOLD state printed before [CMFGEN] iter 9; this state feeds iter 10",
        "replay_input": "post-loop lumina_plasma_state.csv",
        "relative_l1_final_vs_capture_rounded": rel_l1(after, before),
        "max_cell_relative": float(relative.max()),
        "max_cell_shell": int(relative.argmax()),
        "samples": {
            str(shell): {"capture_rounded_K": before[shell],
                         "final_csv_K": after[shell]}
            for shell in (0, 11, 25, 49)
        },
        "precision_note": "capture-side values are logged to the nearest kelvin",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument(
        "--replay", type=Path,
        default=ROOT / "validation/emiss_e1/chieta_A_replay")
    parser.add_argument(
        "--out", type=Path,
        default=ROOT / "validation/emiss_e2e3/assembly_identity_audit.json")
    parser.add_argument(
        "--bands", type=Path,
        default=ROOT / "validation/emiss_e2e3/term_gap_by_band.csv")
    args = parser.parse_args()

    capture_path = args.run / "chieta_iter10"
    authority_checked = check_artifact(capture_path)
    replay_checked = check_artifact(args.replay)
    authority = [np.asarray(x, dtype=np.float64) for x in authority_checked.arrays]
    replay = [np.asarray(x, dtype=np.float64) for x in replay_checked.arrays]
    if authority[3].size != 50_000 or replay[3].size != 50_000:
        raise ValueError("E2 contract requires 50 shells x 1000 bins")

    arrays_a = {name: authority[index] for name, index in FIELDS.items()}
    arrays_r = {name: replay[index] for name, index in FIELDS.items()}
    # This residual is the only opacity split recoverable from LCMFCE01.  It is
    # chi_abs + chi_line_th; the two summands are not separately serialized.
    arrays_a["chi_noncoherent"] = authority[3] - authority[4]
    arrays_r["chi_noncoherent"] = replay[3] - replay[4]

    field_comparison = {
        name: compare(arrays_r[name], arrays_a[name]) for name in arrays_a
    }
    gate = {
        "metric": "global relative L1 against authority",
        "threshold": 1.0e-10,
        "chi_total": field_comparison["chi_total"]["relative_l1"],
        "eta_fixed": field_comparison["eta_fixed"]["relative_l1"],
    }
    gate["pass"] = gate["chi_total"] <= gate["threshold"] and gate[
        "eta_fixed"] <= gate["threshold"]

    grid = {
        name: {
            "bitwise_equal": bool(np.array_equal(authority[index], replay[index])),
            "max_absolute": float(np.max(np.abs(authority[index] - replay[index]))),
        }
        for name, index in (("r_edge", 0), ("nu", 1), ("dnu", 2))
    }

    eta_rebuilt = authority[5] + authority[6]
    eta_payload_identity = {
        "bitwise_equal": bool(np.array_equal(
            eta_rebuilt.view(np.uint64), authority[7].view(np.uint64))),
        "max_absolute": float(np.max(np.abs(eta_rebuilt - authority[7]))),
    }

    lam = C_ANGSTROM / authority[1]
    rows: list[dict] = []
    for band, lo, hi in BANDS:
        mask = (lam >= lo) & (lam < hi)
        for name in ("chi_total", "chi_coherent", "chi_noncoherent",
                     "eta_fixed", "eta_coherent", "eta_total"):
            aa = arrays_a[name].reshape(50, 1000)[:, mask]
            rr = arrays_r[name].reshape(50, 1000)[:, mask]
            rows.append({
                "band_A": band,
                "field": name,
                "relative_l1": rel_l1(rr, aa),
                "authority_l1": float(np.abs(aa).sum(dtype=np.longdouble)),
                "replay_l1": float(np.abs(rr).sum(dtype=np.longdouble)),
            })

    resolved = parse_resolved_config(args.run / "stdout.log")
    excluded = ("LUMINA_BIN", "OMP_NUM_THREADS", "LUMINA_CMF_SOLVE_GPU")
    replay_env = {key: value for key, value in resolved.items() if key not in excluded}
    stdout_lines = (args.run / "stdout.log").read_text(errors="strict").splitlines()
    marker_lines = {}
    for marker in ("[CMFGEN] iter 10:", "[CMFGEN] iter 11:",
                   "Per-level departure b_k dump -> lumina_levelpop.csv",
                   "Pure-CMFGEN plasma state written to lumina_plasma_state.csv",
                   "Pure-CMFGEN ion populations written to lumina_ion_pops.csv"):
        matches = [i + 1 for i, line in enumerate(stdout_lines) if marker in line]
        marker_lines[marker] = matches

    audit = {
        "schema": "emiss-assembly-identity-e2-v1",
        "authority": {
            "path": str(capture_path.resolve()),
            "sha256": authority_checked.manifest["sha256"],
            "iteration": authority_checked.header[5],
            "field_generation": authority_checked.header[6],
            "post_damping": bool(authority_checked.header[7] & 1),
        },
        "replay": {
            "path": str(args.replay.resolve()),
            "sha256": replay_checked.manifest["sha256"],
        },
        "e2_gate": gate,
        "grid_and_bin_boundaries": grid,
        "field_comparison": field_comparison,
        "authority_eta_payload_identity": eta_payload_identity,
        "environment": {
            "authority_resolved_variable_count": len(resolved),
            "replay_imported_variable_count": len(replay_env),
            "intentionally_overridden": {
                "LUMINA_BIN": "offline CPU driver",
                "OMP_NUM_THREADS": "1",
                "LUMINA_CMF_SOLVE_GPU": "0",
            },
            "physics_gate_values_are_imported_from_authority_stdout": True,
        },
        "source_hashes": source_hash_audit(),
        "epoch_evidence_stdout_lines": marker_lines,
        "temperature_epoch_skew": epoch_temperature_audit(
            stdout_lines, args.run / "lumina_plasma_state.csv"),
        "serialized_term_identifiability": {
            "direct": ["chi_total", "chi_coherent", "eta_fixed",
                       "eta_coherent", "eta_total", "J"],
            "derived": ["chi_noncoherent = chi_total - chi_coherent"],
            "not_identifiable": [
                "chi_abs versus chi_line_th inside chi_noncoherent",
                "pre-EPAY eta_line versus continuum eta",
                "per-shell EPAY scale/shape inputs",
                "per-line tau, eps_l, A_ul*n_u contribution",
            ],
        },
        "verdict": "PASS" if gate["pass"] else "UNRESOLVED",
        "e3_permitted": bool(gate["pass"]),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    args.bands.parent.mkdir(parents=True, exist_ok=True)
    with args.bands.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"e2_gate": gate, "verdict": audit["verdict"],
                      "e3_permitted": audit["e3_permitted"]}, indent=2))
    return 0 if gate["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
