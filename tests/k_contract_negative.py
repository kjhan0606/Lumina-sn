#!/usr/bin/env python3
"""Material K-SHAPE negatives plus the K-FRESH CPU call-chain proof."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
import sys

sys.path.insert(0, str(SCRIPTS))
from kshape_contract import CONTRACT_NAME, write_contract  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_raw_contract(deck: Path, n_lines: int, n_trans: int, n_shells: int) -> None:
    values = {
        "schema": "lumina-kshape-v1",
        "line_list_sha256": sha256(deck / "line_list.csv"),
        "tau_sobolev_sha256": sha256(deck / "tau_sobolev.npy"),
        "transition_probabilities_sha256": sha256(
            deck / "transition_probabilities.npy"
        ),
        "n_lines": str(n_lines),
        "n_macro_transitions": str(n_trans),
        "n_shells": str(n_shells),
        "dtype": "<f8",
        "byte_order": "little",
        "array_order": "C",
    }
    (deck / CONTRACT_NAME).write_text(
        "".join(f"{key}={value}\n" for key, value in values.items()),
        encoding="ascii",
    )


def base_deck(deck: Path, tau_value: float = 0.0) -> None:
    deck.mkdir()
    (deck / "geometry.csv").write_text(
        "shell_id,r_inner,r_outer,v_inner,v_outer\n0,1e14,2e14,1e8,2e8\n"
    )
    (deck / "config.json").write_text(
        '{"time_explosion_s":86400,"T_inner_K":10000,'
        '"luminosity_inner_erg_s":1e42,"n_packets":1,'
        '"n_iterations":1,"seed":1}\n'
    )
    (deck / "electron_densities.csv").write_text("shell_id,n_e\n0,1e8\n")
    (deck / "plasma_state.csv").write_text("shell_id,W,T_rad\n0,0.5,10000\n")
    (deck / "density.csv").write_text("shell_id,rho\n0,1e-14\n")
    (deck / "line_list.csv").write_text("nu\n3e15\n")
    (deck / "macro_atom_data.csv").write_text(
        "transition_type,destination_level_idx,lines_idx\n-1,0,0\n"
    )
    (deck / "macro_atom_references.csv").write_text("block_references\n0\n")
    np.save(deck / "line2macro_level_upper.npy", np.array([0], dtype=np.int64))
    np.save(deck / "tau_sobolev.npy", np.array([[tau_value]], dtype=np.float64))
    np.save(
        deck / "transition_probabilities.npy",
        np.array([[1.0]], dtype=np.float64),
    )
    write_contract(deck)


def run_case(loader: Path, name: str, deck: Path, expected: str) -> str:
    proc = subprocess.run(
        [str(loader), str(deck)], text=True, capture_output=True, check=False
    )
    combined = (proc.stdout + proc.stderr).strip()
    print(f"CASE {name} rc={proc.returncode}")
    print(combined)
    if proc.returncode == 0 or expected not in combined:
        raise RuntimeError(f"{name}: expected nonzero and {expected!r}")
    return combined


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--loader", type=Path, required=True)
    parser.add_argument("--fresh", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="lumina-k-contract-") as tmp_text:
        tmp = Path(tmp_text)

        col30 = tmp / "30-columns"
        base_deck(col30)
        np.save(col30 / "tau_sobolev.npy", np.zeros((1, 30), dtype=np.float64))
        np.save(
            col30 / "transition_probabilities.npy",
            np.ones((1, 30), dtype=np.float64),
        )
        write_contract(col30)
        run_case(args.loader, "30-columns", col30, "[K-SHAPE][FATAL]")

        tau_rows = tmp / "wrong-tau-rows"
        base_deck(tau_rows)
        np.save(tau_rows / "tau_sobolev.npy", np.zeros((2, 1), dtype=np.float64))
        write_contract(tau_rows)
        run_case(args.loader, "wrong-tau-rows", tau_rows, "[K-SHAPE][FATAL]")

        trans_rows = tmp / "wrong-transition-rows"
        base_deck(trans_rows)
        np.save(
            trans_rows / "transition_probabilities.npy",
            np.ones((2, 1), dtype=np.float64),
        )
        write_contract(trans_rows)
        run_case(
            args.loader,
            "wrong-transition-rows",
            trans_rows,
            "[K-SHAPE][FATAL]",
        )

        epoch = tmp / "wrong-line-epoch"
        base_deck(epoch)
        (epoch / "line_list.csv").write_text("nu\n2.9e15\n")
        run_case(args.loader, "wrong-line-epoch", epoch, "line_list hash/line-epoch")

        missing = tmp / "missing"
        base_deck(missing)
        (missing / "tau_sobolev.npy").unlink()
        run_case(args.loader, "missing", missing, "tau_sobolev hash/line-epoch")

        missing_trans = tmp / "missing-transition-probabilities"
        base_deck(missing_trans)
        (missing_trans / "transition_probabilities.npy").unlink()
        run_case(
            args.loader,
            "missing-transition-probabilities",
            missing_trans,
            "transition_probabilities hash/line-epoch",
        )

        truncated = tmp / "truncated"
        base_deck(truncated)
        raw = (truncated / "tau_sobolev.npy").read_bytes()
        (truncated / "tau_sobolev.npy").write_bytes(raw[:-4])
        write_raw_contract(truncated, 1, 1, 1)
        run_case(args.loader, "truncated", truncated, "truncated NPY payload")

        truncated_trans = tmp / "truncated-transition-probabilities"
        base_deck(truncated_trans)
        raw = (truncated_trans / "transition_probabilities.npy").read_bytes()
        (truncated_trans / "transition_probabilities.npy").write_bytes(raw[:-4])
        write_raw_contract(truncated_trans, 1, 1, 1)
        run_case(
            args.loader,
            "truncated-transition-probabilities",
            truncated_trans,
            "truncated NPY payload",
        )

        dtype = tmp / "wrong-dtype"
        base_deck(dtype)
        np.save(dtype / "tau_sobolev.npy", np.zeros((1, 1), dtype=np.float32))
        write_raw_contract(dtype, 1, 1, 1)
        run_case(args.loader, "wrong-dtype", dtype, "dtype/byte-order")

        endian = tmp / "wrong-byte-order"
        base_deck(endian)
        np.save(endian / "tau_sobolev.npy", np.zeros((1, 1), dtype=">f8"))
        write_raw_contract(endian, 1, 1, 1)
        run_case(args.loader, "wrong-byte-order", endian, "dtype/byte-order")

        trans_dtype = tmp / "wrong-transition-dtype"
        base_deck(trans_dtype)
        np.save(
            trans_dtype / "transition_probabilities.npy",
            np.ones((1, 1), dtype=np.float32),
        )
        write_raw_contract(trans_dtype, 1, 1, 1)
        run_case(
            args.loader,
            "wrong-transition-dtype",
            trans_dtype,
            "dtype/byte-order",
        )

        trans_endian = tmp / "wrong-transition-byte-order"
        base_deck(trans_endian)
        np.save(
            trans_endian / "transition_probabilities.npy",
            np.ones((1, 1), dtype=">f8"),
        )
        write_raw_contract(trans_endian, 1, 1, 1)
        run_case(
            args.loader,
            "wrong-transition-byte-order",
            trans_endian,
            "dtype/byte-order",
        )

        valid = tmp / "valid-sentinel-seed"
        base_deck(valid, 12345.6789)
        valid_proc = subprocess.run(
            [str(args.loader), str(valid)], text=True, capture_output=True, check=False
        )
        print(f"CASE valid-sentinel-seed rc={valid_proc.returncode}")
        print((valid_proc.stdout + valid_proc.stderr).strip())
        if valid_proc.returncode != 0 or "computed_generation=0 required_generation=1" not in valid_proc.stdout:
            raise RuntimeError("valid sentinel seed did not remain explicitly stale")

        fresh_proc = subprocess.run(
            [str(args.fresh)], text=True, capture_output=True, check=False
        )
        print(f"CASE stale-sentinel-first-consumer rc={fresh_proc.returncode}")
        print((fresh_proc.stdout + fresh_proc.stderr).strip())
        if fresh_proc.returncode != 0 or "sentinel_reached=NO" not in fresh_proc.stdout:
            raise RuntimeError("stale sentinel reached the CPU first consumer")

    print("K-CONTRACT NEGATIVE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
