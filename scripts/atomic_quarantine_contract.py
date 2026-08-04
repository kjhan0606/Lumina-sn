#!/usr/bin/env python3
"""Fail-closed loader contract shared by the atomic quarantine tools.

Only explicitly named files immediately below the deck root are active inputs.
Anything below ``quarantine/`` is an archive and is rejected before an OS open
is attempted.  This module is deliberately independent of the model runtime so
its fixtures can exercise leak failures without running LUMINA.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TextIO


Ion = tuple[int, int]  # (atomic number, zero-based ion number)

LEAK_TAG = "[ATOMIC-ACTIVE-SET-LEAK]"
CONTRACT_TAG = "[ATOMIC-LOADER-CONTRACT]"

# This is an allowlist, not a discovery glob.  A newly introduced runtime
# atomic input must be registered here and in the sealed manifest or validation
# fails instead of silently consuming it.
ACTIVE_ROOT_FILES = frozenset({
    "active_ions.csv",
    "abundances.csv",
    "atom_masses.csv",
    "atomic_data_cmfgen.h5",
    "atomic_vintage_manifest.csv",
    "cmfgen_sigma_bf.bin",
    "coldata_cmfgen_manifest.csv",
    "config.json",
    "ionization_energies.csv",
    "level_multiplicity.csv",
    "levels.csv",
    "line2macro_level_upper.npy",
    "line_list.csv",
    "kshape_contract.txt",
    "ma_radrecomb_target.bin",
    "ma_radrecomb_target_manifest.csv",
    "macro_atom_data.csv",
    "macro_atom_references.csv",
    "tau_sobolev.npy",
    "transition_probabilities.npy",
    "zeta_data.npy",
    "zeta_ions.csv",
    "zeta_temps.csv",
    "feiii_col_zhang.bin",
})


class AtomicContractError(RuntimeError):
    """A deck violated the fail-closed active-loader contract."""


@dataclass(frozen=True)
class InventoryVerdict:
    passed: bool
    diagnostics: tuple[str, ...]


def ion_text(ion: Ion) -> str:
    return f"Z={ion[0]},ion0={ion[1]}"


def _lexical_relative(deck: Path, candidate: Path) -> Path:
    deck_abs = deck.absolute()
    path_abs = candidate.absolute()
    try:
        return path_abs.relative_to(deck_abs)
    except ValueError as exc:
        raise AtomicContractError(
            f"{CONTRACT_TAG} path escapes deck root: {candidate}"
        ) from exc


def guard_active_path(
    deck: Path, candidate: Path, registered_names: Iterable[str] = (),
) -> Path:
    """Return an approved root input or fail before touching the filesystem."""
    relative = _lexical_relative(deck, candidate)
    if "quarantine" in relative.parts:
        raise AtomicContractError(
            f"{LEAK_TAG} refused quarantine consumption: {relative.as_posix()}"
        )
    allowed = ACTIVE_ROOT_FILES | frozenset(registered_names)
    if len(relative.parts) != 1 or relative.name not in allowed:
        raise AtomicContractError(
            f"{CONTRACT_TAG} unregistered/non-root input: {relative.as_posix()}"
        )
    return deck / relative.name


def open_active(deck: Path, name: str, *args, **kwargs) -> TextIO:
    """Open one registered deck-root input; recursive discovery is impossible."""
    return guard_active_path(deck, deck / name).open(*args, **kwargs)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_active_ions(deck: Path) -> set[Ion]:
    result: set[Ion] = set()
    with open_active(deck, "active_ions.csv", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"atomic_number", "ion_number"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise AtomicContractError(
                f"{CONTRACT_TAG} active_ions.csv missing {sorted(missing)}"
            )
        for row in reader:
            key = (int(row["atomic_number"]), int(row["ion_number"]))
            if key in result:
                raise AtomicContractError(
                    f"{CONTRACT_TAG} duplicate active ion {ion_text(key)}"
                )
            result.add(key)
    return result


def read_quarantine_ions(deck: Path) -> set[Ion]:
    # Manifest access is an archive-validation operation, never an active load.
    path = deck / "quarantine" / "manifest.json"
    payload = json.loads(path.read_text())
    return {
        (int(item["ion"]["Z"]), int(item["ion"]["ion0"]))
        for item in payload["ions"]
        if item["status"] == "quarantined"
    }


def compare_ion_inventory(
    expected: Iterable[Ion],
    loaded: Iterable[Ion],
    quarantined: Iterable[Ion],
    preserved: Iterable[Ion],
) -> InventoryVerdict:
    """Apply both directions plus the quarantine partition invariants."""
    expected_set = set(expected)
    loaded_set = set(loaded)
    quarantine_set = set(quarantined)
    preserved_set = set(preserved)
    diagnostics: list[str] = []

    for key in sorted(expected_set - loaded_set):
        diagnostics.append(f"FAIL_MISSING_ION {ion_text(key)}")
    for key in sorted(loaded_set - expected_set):
        diagnostics.append(f"FAIL_EXTRA_ION {ion_text(key)}")
    for key in sorted(loaded_set & quarantine_set):
        diagnostics.append(f"{LEAK_TAG} loaded quarantined ion {ion_text(key)}")
    union = loaded_set | quarantine_set
    for key in sorted(preserved_set - union):
        diagnostics.append(f"FAIL_ARCHIVE_PARTITION_MISSING {ion_text(key)}")
    for key in sorted(union - preserved_set):
        diagnostics.append(f"FAIL_ARCHIVE_PARTITION_EXTRA {ion_text(key)}")
    return InventoryVerdict(not diagnostics, tuple(diagnostics))


def compare_multiset(
    label: str,
    expected: Counter,
    loaded: Counter,
) -> InventoryVerdict:
    """Exact two-way multiset comparison used by level and line gates."""
    diagnostics: list[str] = []
    for key, count in sorted((expected - loaded).items(), key=lambda item: repr(item[0])):
        diagnostics.append(f"FAIL_MISSING_{label} count={count} key={key!r}")
    for key, count in sorted((loaded - expected).items(), key=lambda item: repr(item[0])):
        diagnostics.append(f"FAIL_EXTRA_{label} count={count} key={key!r}")
    return InventoryVerdict(not diagnostics, tuple(diagnostics))
