#!/usr/bin/env python3
"""CPU-only self-check and the three required negative controls."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import tempfile

from atomic_quarantine_contract import (
    AtomicContractError,
    compare_ion_inventory,
    guard_active_path,
)


ACTIVE = {(14, 1), (16, 1)}
QUARANTINED = {(6, 0)}
PRESERVED = ACTIVE | QUARANTINED


def verdict(name: str, loaded: set[tuple[int, int]], expect_pass: bool) -> None:
    result = compare_ion_inventory(ACTIVE, loaded, QUARANTINED, PRESERVED)
    state = "PASS" if result.passed else "FAIL"
    print(f"{name}: {state}")
    for line in result.diagnostics:
        print(f"  {line}")
    if result.passed != expect_pass:
        raise AssertionError(f"{name}: unexpected verdict")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="atomic-quarantine-fixture-") as raw:
        deck = Path(raw) / "deck"
        archive = deck / "quarantine"
        archive.mkdir(parents=True)
        with (deck / "active_ions.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("atomic_number", "ion_number"))
            writer.writerows(sorted(ACTIVE))
        (archive / "manifest.json").write_text(json.dumps({"ions": []}))
        sentinel = archive / "DO_NOT_LOAD"
        sentinel.write_text("A quarantine traversal is a loader contract violation.\n")
        sentinel.chmod(0)

        verdict("CONTROL baseline exact active set", set(ACTIVE), True)
        verdict("NEGATIVE 1 hidden extra ion", ACTIVE | {(6, 0)}, False)
        verdict("NEGATIVE 2 missing active ion", ACTIVE - {(16, 1)}, False)

        print("NEGATIVE 3 loader reads quarantine: FAIL")
        try:
            guard_active_path(deck, sentinel)
        except AtomicContractError as exc:
            print(f"  {exc}")
            if "[ATOMIC-ACTIVE-SET-LEAK]" not in str(exc):
                raise AssertionError("leak did not use mandatory fatal tag")
        else:
            raise AssertionError("quarantine path was silently accepted")
        finally:
            sentinel.chmod(0o600)

    print("FIXTURE SELF-CHECK: PASS (all three negative controls failed closed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
