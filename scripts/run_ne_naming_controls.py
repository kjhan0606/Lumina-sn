#!/usr/bin/env python3
"""Independent five-case NE-NAMING injected-defect battery.

Every fixture is a ``copy2`` scratch copy below
``/tmp/lumina_ne_naming_controls_*``.  The source deck is read only.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


COPY_FILES = (
    "config.json",
    "geometry.csv",
    "electron_densities.csv",
    "plasma_state.csv",
    "density.csv",
    "abundances.csv",
)
APPROVAL_TOKEN = "NE-NAMING-A-LEGACY-READONLY-2026-08-05"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_fixture(source: Path, root: Path, name: str) -> Path:
    fixture = root / name
    fixture.mkdir()
    for filename in COPY_FILES:
        shutil.copy2(source / filename, fixture / filename)
    return fixture


def write_manifest(fixture: Path, manifest: dict) -> Path:
    path = fixture / "ne_naming_manifest.json"
    with path.open("w") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    return path


def replace_electron_generation(fixture: Path, manifest: dict) -> None:
    path = fixture / "electron_densities.csv"
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or ())
        rows = list(reader)
    rows[0]["n_e"] = format(float(rows[0]["n_e"]) * 1.01, ".17g")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    manifest["companions"]["electron_densities.csv"] = {
        "sha256": sha256_file(path),
        "generation_id": "NE-INJECTED-NEW-ELECTRON-GENERATION",
    }


def run_child(checker: Path, fixture: Path, manifest_path: Path, claim: str,
              token: str | None, marker: str, expected_rc: int) -> bool:
    command = [
        sys.executable, str(checker), "--deck", str(fixture),
        "--manifest", str(manifest_path), "--claim", claim,
    ]
    if token is not None:
        command.extend(("--approval-token", token))
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    found = marker in result.stdout
    passed = result.returncode == expected_rc and found
    print(
        f"NE_NAMING_CONTROL case={fixture.name} child_rc={result.returncode} "
        f"expected_rc={expected_rc} marker={'yes' if found else 'no'} "
        f"verdict={'PASS' if passed else 'FAIL'} fixture={fixture}"
    )
    if not passed:
        print(result.stdout)
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checker", type=Path, default="scripts/check_ne_naming.py")
    parser.add_argument(
        "--deck", type=Path, default="data/tardis_reference_toy06_19p48d"
    )
    parser.add_argument(
        "--manifest", type=Path,
        default="docs/manifests/ne_naming_toy06_19p48d_legacy.json",
    )
    args = parser.parse_args()

    checker = args.checker.resolve()
    source = args.deck.resolve()
    source_manifest = args.manifest.resolve()
    missing = [name for name in COPY_FILES if not (source / name).is_file()]
    if not checker.is_file() or not source_manifest.is_file() or missing:
        print(
            f"NE_NAMING_CONTROL setup_error checker={checker} "
            f"manifest={source_manifest} missing={missing}",
            file=sys.stderr,
        )
        return 1
    with source_manifest.open() as stream:
        baseline = json.load(stream)

    root = Path(tempfile.mkdtemp(prefix="lumina_ne_naming_controls_", dir="/tmp"))
    print(f"NE_NAMING_CONTROL scratch={root}")
    cases: list[tuple[Path, Path, str, str | None, str, int]] = []

    fixture = copy_fixture(source, root, "missing_mode")
    manifest = copy.deepcopy(baseline)
    manifest.pop("electron_density_mode", None)
    cases.append((
        fixture, write_manifest(fixture, manifest), "legacy-read-only", APPROVAL_TOKEN,
        "NE-NAMING][FATAL] missing mode", 1,
    ))

    fixture = copy_fixture(source, root, "unapproved_placeholder")
    cases.append((
        fixture, write_manifest(fixture, copy.deepcopy(baseline)), "production", None,
        "NE-NAMING][FATAL] unapproved placeholder", 1,
    ))

    fixture = copy_fixture(source, root, "epoch_mismatch")
    manifest = copy.deepcopy(baseline)
    manifest["electron_density_mode"] = "CMFGEN_CHARGE_BALANCE"
    manifest["formula"] = "n_e(v) = sum_Z sum_q q * n_Z_q(v)"
    manifest["builder"]["producer_status"] = "REGISTERED_TRUE_PATH_SPECIFICATION"
    manifest["source"]["epoch_days"] = 18.0
    cases.append((
        fixture, write_manifest(fixture, manifest), "production", None,
        "NE-NAMING][FATAL] epoch mismatch", 1,
    ))

    fixture = copy_fixture(source, root, "generation_mismatch")
    manifest = copy.deepcopy(baseline)
    replace_electron_generation(fixture, manifest)
    cases.append((
        fixture, write_manifest(fixture, manifest), "legacy-read-only", APPROVAL_TOKEN,
        "NE-NAMING][FATAL] generation mismatch", 1,
    ))

    fixture = copy_fixture(source, root, "approved_legacy")
    cases.append((
        fixture, write_manifest(fixture, copy.deepcopy(baseline)),
        "legacy-read-only", APPROVAL_TOKEN, "NE-NAMING][WARN]", 0,
    ))

    passed = sum(run_child(checker, *case) for case in cases)
    print(
        f"NE_NAMING_CONTROL_SUMMARY passed={passed} total={len(cases)} "
        f"scratch={root}"
    )
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
