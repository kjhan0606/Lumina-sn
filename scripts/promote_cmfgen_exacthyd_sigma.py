#!/usr/bin/env python3
"""Promote an exact CMFGEN type-2/3/8/9 sigma asset into the active deck.

The existing 1178-bin stand-in asset is moved to a loader-forbidden,
recoverable archive.  Header, flags, extent, finite/nonnegative values and the
sealed-deck manifest are checked before either canonical file or provenance is
changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct

import numpy as np


N_BINS = 1178
NU_MIN = 5.8412785919616062e13
NU_MAX = 3.0e16
OLD_STANDIN_SHA256 = "4772cdad1ad75f6a409e1e38732b8b94f1ba741921f716330a2c02e37089847e"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def header(path: Path):
    with path.open("rb") as stream:
        raw = stream.read(32)
        if len(raw) != 32:
            raise SystemExit(f"short sigma header: {path}")
        values = struct.unpack("<IIiidd", raw)
        flags = stream.read(values[2])
        pad = (8 - values[2] % 8) % 8
        padding = stream.read(pad)
        offset = stream.tell()
    return (*values, flags, padding, offset)


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".exacthyd.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    args = parser.parse_args()
    deck = args.deck.resolve()
    candidate = args.candidate.resolve()
    canonical = deck / "cmfgen_sigma_bf.bin"
    provenance_path = deck / "DECK_PROVENANCE.json"
    manifest_path = deck / "quarantine/manifest.json"
    archive_dir = deck / "quarantine/exacthyd_promotion_2026-08-08"
    backup = archive_dir / "cmfgen_sigma_bf.pre_exacthyd_standin_1178.bin"
    if not candidate.is_file():
        raise SystemExit("candidate sigma missing")
    resume_partial = (not canonical.exists() and backup.is_file() and
                      sha256(backup) == OLD_STANDIN_SHA256)
    if not canonical.is_file() and not resume_partial:
        raise SystemExit("canonical sigma missing outside a recognized partial promotion")
    old_source = backup if resume_partial else canonical

    provenance = json.loads(provenance_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("state") != "sealed":
        raise SystemExit("deck is not sealed")
    for name, record in manifest.get("active_files", {}).items():
        path = old_source if resume_partial and name == canonical.name else deck / name
        if (not path.is_file() or path.stat().st_size != record["bytes"] or
                sha256(path) != record["sha256"]):
            raise SystemExit(f"sealed active file drifted: {path}")

    old_sha = sha256(old_source)
    if old_sha != OLD_STANDIN_SHA256:
        raise SystemExit(f"canonical is not registered stand-in asset: {old_sha}")
    old = header(old_source)
    new = header(candidate)
    if new[:6] != (0x434D4644, 1, old[2], N_BINS, NU_MIN, NU_MAX):
        raise SystemExit(f"candidate header mismatch: {new[:6]}")
    if old[6] != new[6]:
        raise SystemExit("candidate changed has_cmfgen flags")
    if any(new[7]):
        raise SystemExit("candidate has nonzero padding")
    expected_size = new[8] + new[2] * new[3] * 8
    if candidate.stat().st_size != expected_size:
        raise SystemExit("candidate extent mismatch")
    grid = np.memmap(candidate, dtype="<f8", mode="r", offset=new[8],
                     shape=(new[2], new[3]))
    if not np.isfinite(grid).all() or not (grid >= 0.0).all():
        raise SystemExit("candidate has nonfinite/negative sigma")
    changed_rows = int(np.any(grid != np.memmap(
        old_source, dtype="<f8", mode="r", offset=old[8],
        shape=(old[2], old[3])), axis=1).sum())
    del grid
    if changed_rows == 0:
        raise SystemExit("candidate did not change any sigma row")

    if not resume_partial:
        if archive_dir.exists() or backup.exists():
            raise SystemExit(f"refusing to reuse exacthyd archive: {archive_dir}")
        archive_dir.mkdir()

    new_sha = sha256(candidate)
    old_record = {
        "path": str(backup.relative_to(deck)),
        "bytes": old_source.stat().st_size,
        "sha256": old_sha,
        "evaluator": "params[0]-as-sigma_0 stand-in for CMFGEN types 2/3/8/9",
    }
    new_record = {
        "path": canonical.name,
        "bytes": candidate.stat().st_size,
        "sha256": new_sha,
        "evaluator": "CMFGEN SUB_PHOT_GEN exact types 2/3/8/9",
        "changed_rows": changed_rows,
        "frequency_grid_unchanged": True,
    }

    # The candidate may live on /tmp (a different filesystem), so first copy it
    # beside the canonical file, fsync, and verify the staged hash.  Only then do
    # same-filesystem atomic renames.  resume_partial recognizes the sole safe
    # interrupted state left by the pre-transaction implementation.
    stage = canonical.with_suffix(canonical.suffix + ".exacthyd.stage")
    shutil.copyfile(candidate, stage)
    with stage.open("rb") as stream:
        os.fsync(stream.fileno())
    if stage.stat().st_size != candidate.stat().st_size or sha256(stage) != new_sha:
        raise SystemExit("same-filesystem candidate staging verification failed")
    if not resume_partial:
        os.replace(canonical, backup)
    os.replace(stage, canonical)

    provenance.setdefault("env", {})["CMFGEN_EXACT_HYD"] = "1"
    provenance["frequency_grid"]["sigma_sha256"] = new_sha
    provenance["frequency_grid"]["sigma_evaluator"] = \
        "CMFGEN SUB_PHOT_GEN exact types 1/2/3/7/8/9/20/21/22"
    write_json_atomic(provenance_path, provenance)

    manifest.setdefault("sigma_migrations", []).append({
        "schema": "lumina-cmfgen-exacthyd-promotion-v1",
        "date": "2026-08-08",
        "old": old_record,
        "new": new_record,
    })
    manifest["active_files"]["cmfgen_sigma_bf.bin"] = {
        "bytes": canonical.stat().st_size,
        "sha256": new_sha,
    }
    manifest["active_files"]["DECK_PROVENANCE.json"] = {
        "bytes": provenance_path.stat().st_size,
        "sha256": sha256(provenance_path),
    }
    write_json_atomic(manifest_path, manifest)
    print("[CMFGEN][EXACTHYD][PROMOTE][PASS] "
          f"canonical={canonical} sha256={new_sha} changed_rows={changed_rows} "
          f"recoverable_old={backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
