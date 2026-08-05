#!/usr/bin/env python3
"""Materialize and preflight the 18 order-D negative/warning overlay decks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


NSHELLS = 2
HEADER = b"atomic_number,0,1\n"
MASS_ONE = b"atomic_number,mass_amu\n6,12\n"
MASS_TWO = b"atomic_number,mass_amu\n6,12\n8,16\n"
AB_ONE = HEADER + b"6,1,1\n"
AB_TWO = HEADER + b"6,0.5,0.5\n8,0.5,0.5\n"
CACHE_INPUTS = (
    "line2macro_level_upper.npy",
    "tau_sobolev.npy",
    "transition_probabilities.npy",
    "zeta_data.npy",
)


def specs() -> dict[str, tuple[bytes, bytes | None]]:
    huge_z = b"9" * 200
    overlong = b"6,0." + b"0" * 8200 + b",0.5\n"
    return {
        "D1": (MASS_ONE, None),
        "D2": (MASS_ONE, b"atomic_number,0\n6,1\n"),
        "D3": (MASS_ONE, HEADER + b"6,1\n"),
        "D4": (MASS_ONE, HEADER + b"8,1,1\n"),
        "D7a": (MASS_ONE, HEADER + b"6,nan,1\n"),
        "D7b": (MASS_ONE, HEADER + b"6,inf,1\n"),
        "D7c": (MASS_ONE, HEADER + b"6,-0.1,1\n"),
        # One ID, all three required expressions: Z suffix, X suffix, NUL.
        "D8": (
            b"atomic_number,mass_amu\n6,12\n8,16\n10,20\n",
            HEADER + b"6junk,0.5,0.5\n8,0.5junk,0.5\n10,0.5,\x000.5\n",
        ),
        # One ID, both required sides are duplicated in the same invocation.
        "D9": (
            b"atomic_number,mass_amu\n6,12\n6,12\n",
            HEADER + b"6,0.5,0.5\n6,0.5,0.5\n",
        ),
        "D10": (MASS_ONE, HEADER + overlong),
        "D12": (MASS_ONE, b"atomic_number,1,0\n6,1,1\n"),
        "D13": (MASS_ONE, HEADER),
        # D14 aggregates invalid integer forms/ranges, invalid masses, and a
        # missing mass field so physical and column-loader row counts differ.
        "D14": (
            b"atomic_number,mass_amu\n1.5,0\n0,nan\n2147483648,inf\n6\n",
            AB_ONE,
        ),
        "D15": (MASS_ONE, HEADER + b"6,0,0\n"),
        "D16": (MASS_ONE, HEADER + b"6,1.2,1\n"),
        # Exercise ERANGE from both strtol (huge Z) and strtod (underflow).
        "D17": (
            MASS_TWO,
            HEADER + huge_z + b",0.5,0.5\n6,1e-9999,0.5\n8,1,0.5\n",
        ),
        "D5": (MASS_TWO, AB_ONE),
        "D6": (MASS_ONE, HEADER + b"6,0.9,1\n"),
    }


def fields(line: bytes) -> list[bytes]:
    return line.rstrip(b"\r\n").split(b",")


def data_lines(blob: bytes) -> list[bytes]:
    return blob.splitlines()[1:]


def preflight(cases: dict[str, tuple[bytes, bytes | None]]) -> None:
    expected = {
        "D1", "D2", "D3", "D4", "D7a", "D7b", "D7c", "D8",
        "D9", "D10", "D12", "D13", "D14", "D15", "D16", "D17",
        "D5", "D6",
    }
    assert set(cases) == expected and len(cases) == 18
    assert cases["D1"][1] is None
    assert len(fields(cases["D2"][1].splitlines()[0])) - 1 != NSHELLS
    assert len(fields(data_lines(cases["D3"][1])[0])) - 1 != NSHELLS
    assert data_lines(cases["D4"][1])[0].startswith(b"8,")
    assert b",nan," in cases["D7a"][1]
    assert b",inf," in cases["D7b"][1]
    assert b",-0.1," in cases["D7c"][1]
    assert b"6junk," in cases["D8"][1]
    assert b"0.5junk" in cases["D8"][1]
    assert b"\x00" in cases["D8"][1]
    assert [r.split(b",", 1)[0] for r in data_lines(cases["D9"][0])] == [b"6", b"6"]
    assert [r.split(b",", 1)[0] for r in data_lines(cases["D9"][1])] == [b"6", b"6"]
    assert max(map(len, cases["D10"][1].splitlines(keepends=True))) >= 8192
    assert cases["D12"][1].splitlines()[0] == b"atomic_number,1,0"
    assert data_lines(cases["D13"][1]) == []
    assert b"1.5,0" in cases["D14"][0]
    assert b"0,nan" in cases["D14"][0]
    assert b"2147483648,inf" in cases["D14"][0]
    assert data_lines(cases["D14"][0])[-1] == b"6"
    assert fields(data_lines(cases["D15"][1])[0])[1:] == [b"0", b"0"]
    assert b",1.2," in cases["D16"][1]
    assert b"9" * 200 in cases["D17"][1]
    assert b"1e-9999" in cases["D17"][1]
    assert {6, 8} - {6} == {8}  # D5 mass-set minus abundance-set.
    assert b"6,0.9,1" in cases["D6"][1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_binding(base: Path) -> dict[str, object]:
    inputs = [base / name for name in CACHE_INPUTS]
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"fixture cache inputs missing: {missing}")
    with ThreadPoolExecutor(max_workers=len(inputs)) as pool:
        hashes = list(pool.map(sha256, inputs))
    generator = Path(__file__).resolve()
    return {
        "base": str(base.resolve()),
        "generator_sha256": sha256(generator),
        "deck_sha256": dict(zip(CACHE_INPUTS, hashes, strict=True)),
    }


def binding_key(binding: dict[str, object]) -> str:
    encoded = json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def materialize(
    base: Path, output: Path, *, cache_metadata: dict[str, object] | None = None
) -> None:
    cases = specs()
    preflight(cases)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    for case_id, (mass_blob, abundance_blob) in cases.items():
        deck = output / case_id
        deck.mkdir()
        for entry in base.iterdir():
            if entry.name in {"atom_masses.csv", "abundances.csv"}:
                continue
            os.symlink(entry.resolve(), deck / entry.name)
        (deck / "atom_masses.csv").write_bytes(mass_blob)
        if abundance_blob is not None:
            (deck / "abundances.csv").write_bytes(abundance_blob)

    manifest = {
        "n_shells": NSHELLS,
        "case_count": len(cases),
        "cases": list(cases),
        "base": str(base.resolve()),
    }
    if cache_metadata is not None:
        manifest["cache"] = cache_metadata
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"fixture preflight PASS: {len(cases)} cases -> {output}")


def cache_valid(path: Path, key: str) -> bool:
    try:
        manifest = json.loads((path / "manifest.json").read_text())
    except (OSError, ValueError, TypeError):
        return False
    expected = set(specs())
    return (
        manifest.get("cache", {}).get("key") == key
        and manifest.get("case_count") == len(expected)
        and all((path / case_id).is_dir() for case_id in expected)
    )


def cached_materialize(base: Path, cache_root: Path) -> tuple[Path, bool, str]:
    """Return an immutable, content-addressed node-local fixture directory."""
    base = base.resolve()
    binding = cache_binding(base)
    key = binding_key(binding)
    cache_root.mkdir(parents=True, exist_ok=True)
    target = cache_root / key
    if cache_valid(target, key):
        return target, True, key

    if target.exists():
        stale = cache_root / f".{key}.stale.{os.getpid()}"
        target.replace(stale)
        shutil.rmtree(stale)

    temporary = Path(tempfile.mkdtemp(prefix=f".{key}.", dir=cache_root))
    try:
        materialize(
            base,
            temporary,
            cache_metadata={"key": key, "binding": binding},
        )
        try:
            temporary.replace(target)
        except OSError:
            if not cache_valid(target, key):
                raise
            shutil.rmtree(temporary)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return target, False, key


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path)
    destination.add_argument("--cache-root", type=Path)
    args = parser.parse_args()
    if not args.base.is_dir():
        parser.error(f"base deck does not exist: {args.base}")
    if args.cache_root is not None:
        path, hit, key = cached_materialize(args.base, args.cache_root)
        print(
            f"fixture cache {'HIT' if hit else 'MISS'} key={key} path={path}"
        )
    else:
        materialize(args.base.resolve(), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
