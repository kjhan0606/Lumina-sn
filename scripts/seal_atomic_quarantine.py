#!/usr/bin/env python3
"""Seal hashes and row counts after all active-deck sidecars are baked."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from atomic_quarantine_contract import ACTIVE_ROOT_FILES, sha256_file


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DECK = ROOT / "data/tardis_reference_toy06_19p48d_sivcaiv_active"


def csv_rows(path: Path) -> int:
    with path.open(newline="") as stream:
        reader = csv.reader(stream)
        next(reader)
        return sum(1 for _ in reader)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck", type=Path, default=DEFAULT_DECK)
    args = parser.parse_args()
    deck = args.deck.resolve()
    manifest_path = deck / "quarantine/manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"manifest absent: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("state") != "draft":
        raise SystemExit(f"refusing to reseal state={manifest.get('state')!r}")

    # feiii_col_zhang.bin is a feature-gated auxiliary, not mandatory for a
    # generic fixture, but it is present and sealed in this production deck.
    required = ACTIVE_ROOT_FILES - {"feiii_col_zhang.bin"}
    missing = sorted(name for name in required if not (deck / name).is_file())
    if missing:
        raise SystemExit(f"active root is incomplete; missing={missing}")

    snapshot = deck / "quarantine/source_deck_snapshot"
    for name, record in manifest["archive_files"].items():
        path = snapshot / name
        if not path.is_file():
            raise SystemExit(f"archived source file absent: {path}")
        actual = sha256_file(path)
        if actual != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise SystemExit(f"archived source hash/size mismatch: {path}")

    active_files = {}
    for path in sorted(deck.iterdir()):
        if not path.is_file() or path.name == "verification.log":
            continue
        record = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        if path.suffix == ".csv":
            record["rows"] = csv_rows(path)
        active_files[path.name] = record
    manifest["active_files"] = active_files
    manifest["state"] = "sealed"
    manifest["seal_contract"] = {
        "all_active_root_hashes_required": True,
        "all_archive_snapshot_hashes_required": True,
        "row_counts_required_for_csv": True,
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, manifest_path)
    print(
        f"SEALED atomic quarantine: active_files={len(active_files)} "
        f"archive_files={len(manifest['archive_files'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
