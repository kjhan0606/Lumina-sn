#!/usr/bin/env python3
"""Write/check the fail-closed K-SHAPE sidecar for runtime NPY arrays."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np


CONTRACT_NAME = "kshape_contract.txt"
SCHEMA = "lumina-kshape-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array(path: Path) -> np.ndarray:
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if value.ndim != 2:
        raise ValueError(f"{path.name}: expected 2 dimensions, got {value.ndim}")
    if value.dtype.str != "<f8":
        raise ValueError(
            f"{path.name}: expected little-endian float64 (<f8), got {value.dtype.str}"
        )
    if not value.flags.c_contiguous:
        raise ValueError(f"{path.name}: expected C-order array")
    return value


def build_contract(deck: Path) -> dict[str, str]:
    deck = deck.resolve()
    line_list = deck / "line_list.csv"
    tau_path = deck / "tau_sobolev.npy"
    trans_path = deck / "transition_probabilities.npy"
    for path in (line_list, tau_path, trans_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    tau = _array(tau_path)
    trans = _array(trans_path)
    if tau.shape[1] != trans.shape[1]:
        raise ValueError(
            "tau_sobolev/transition_probabilities shell counts differ: "
            f"{tau.shape[1]} != {trans.shape[1]}"
        )

    return {
        "schema": SCHEMA,
        "line_list_sha256": sha256(line_list),
        "tau_sobolev_sha256": sha256(tau_path),
        "transition_probabilities_sha256": sha256(trans_path),
        "n_lines": str(tau.shape[0]),
        "n_macro_transitions": str(trans.shape[0]),
        "n_shells": str(tau.shape[1]),
        "dtype": "<f8",
        "byte_order": "little",
        "array_order": "C",
    }


def write_contract(deck: Path) -> Path:
    deck = deck.resolve()
    values = build_contract(deck)
    target = deck / CONTRACT_NAME
    temporary = deck / f".{CONTRACT_NAME}.tmp"
    temporary.write_text(
        "".join(f"{key}={value}\n" for key, value in values.items()),
        encoding="ascii",
    )
    temporary.replace(target)
    return target


def read_contract(deck: Path) -> dict[str, str]:
    path = deck.resolve() / CONTRACT_NAME
    values: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="ascii").splitlines(), 1):
        if not raw or "=" not in raw:
            raise ValueError(f"{path}:{line_number}: malformed contract line")
        key, value = raw.split("=", 1)
        if key in values:
            raise ValueError(f"{path}:{line_number}: duplicate key {key}")
        values[key] = value
    return values


def check_contract(deck: Path) -> dict[str, str]:
    expected = build_contract(deck)
    actual = read_contract(deck)
    if actual != expected:
        keys = sorted(set(actual) | set(expected))
        defects = [
            f"{key}: got {actual.get(key)!r}, expected {expected.get(key)!r}"
            for key in keys
            if actual.get(key) != expected.get(key)
        ]
        raise ValueError("K-SHAPE contract mismatch: " + "; ".join(defects))
    return actual


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("write", "check"))
    parser.add_argument("deck", type=Path)
    args = parser.parse_args()
    values = (
        {"path": str(write_contract(args.deck))}
        if args.mode == "write"
        else check_contract(args.deck)
    )
    print(
        f"K-SHAPE {args.mode.upper()}: {args.deck} "
        f"line_list_sha256={values.get('line_list_sha256', 'written')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
