#!/usr/bin/env python3
"""Independent five-case DECK-FOSSIL injected-defect battery.

Fixtures are ``copy2`` scratch copies below
``/tmp/lumina_deck_fossil_controls_*``.  No canonical or derived deck is
written, and this runner does not invoke the atomic generation writer.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


COMPANIONS = (
    "config.json",
    "geometry.csv",
    "electron_densities.csv",
    "plasma_state.csv",
)
CONFIG_KEYS = (
    "time_explosion_s",
    "T_inner_K",
    "luminosity_inner_erg_s",
    "n_shells",
    "v_inner_min_cm_s",
    "v_outer_max_cm_s",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def profile_sha256(values: list[float]) -> str:
    return sha256_json([float(value).hex() for value in values])


def read_column(path: Path, *names: str) -> list[list[float]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return [[float(row[name]) for row in rows] for name in names]


def copy_fixture(source: Path, root: Path, name: str) -> Path:
    fixture = root / name
    fixture.mkdir()
    for filename in COMPANIONS:
        shutil.copy2(source / filename, fixture / filename)
    return fixture


def baseline_manifest(fixture: Path, quarantine: Path) -> dict:
    with (fixture / "config.json").open() as stream:
        config = json.load(stream)
    geometry_r, = read_column(fixture / "geometry.csv", "r_inner")
    w_values, trad_values = read_column(fixture / "plasma_state.csv", "W", "T_rad")
    ne_values, = read_column(fixture / "electron_densities.csv", "n_e")
    generation_id = "DECK-FOSSIL-CONTROL-GENERATION"
    writer = Path(__file__).resolve()
    producer = {
        "writer": {"path": str(writer), "sha256": sha256_file(writer)},
        "argv": [
            sys.executable, str(writer), "--fixture-replay",
            "/tmp/lumina_deck_fossil_controls_registered_generation",
        ],
        "argv_template": [
            sys.executable, str(writer), "--fixture-replay", "{output}",
        ],
        "environment": {"LC_ALL": "C", "PYTHONHASHSEED": "0"},
        "working_directory": str(Path.cwd().resolve()),
    }
    inputs = {str(quarantine): {"sha256": sha256_file(quarantine)}}
    epoch = {"value": float(config["target_epoch_d"]), "unit": "day"}
    constants = {"sigma_SB": 5.670374e-5}
    units = {
        "L": "erg/s", "r_inner": "cm", "T_inner": "K", "W": "dimensionless",
        "T_rad": "K", "n_e": "cm^-3",
    }
    registration = {
        "writer": producer["writer"],
        "argv": producer["argv"],
        "argv_template": producer["argv_template"],
        "environment": producer["environment"],
        "working_directory": producer["working_directory"],
        "inputs": inputs,
        "epoch": epoch,
        "constants": constants,
        "units": units,
    }
    producer["registration_sha256"] = sha256_json(registration)
    deck_l = float(config["luminosity_inner_erg_s"])
    deck_t = float(config["T_inner_K"])
    t_sb = (
        deck_l / (4.0 * math.pi * geometry_r[0] ** 2 * constants["sigma_SB"])
    ) ** 0.25
    timestamp = datetime.now(timezone.utc).isoformat()
    return {
        "schema": "lumina.deck-generation/v1",
        "generation_id": generation_id,
        "producer": producer,
        "inputs": inputs,
        "epoch": epoch,
        "constants": constants,
        "units": units,
        "config_keys": {key: config[key] for key in CONFIG_KEYS},
        "companions": {
            name: {"sha256": sha256_file(fixture / name), "generation_id": generation_id}
            for name in COMPANIONS
        },
        "generation": {
            "started_at": timestamp,
            "committed_at": timestamp,
            "atomic_commit": True,
        },
        "replay": {
            "observed": {
                "luminosity_inner_erg_s": deck_l,
                "r_inner_cm": geometry_r[0],
                "T_inner_K": deck_t,
                "W_profile_sha256": profile_sha256(w_values),
                "T_rad_profile_sha256": profile_sha256(trad_values),
                "n_e_profile_sha256": profile_sha256(ne_values),
            },
            "metrics": {
                "R_L": 1.0,
                "epsilon_L": 0.0,
                "Delta_SB_K": abs(deck_t - t_sb),
            },
            "thresholds": {"epsilon_L_max": 1.0e-6, "Delta_SB_K_max": 5.0},
        },
    }


def write_manifest(fixture: Path, manifest: dict) -> Path:
    path = fixture / "generation_manifest.json"
    with path.open("w") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    return path


def mutate_config_generation(fixture: Path) -> None:
    path = fixture / "config.json"
    with path.open() as stream:
        config = json.load(stream)
    config["T_inner_K"] = float(config["T_inner_K"]) + 10.0
    with path.open("w") as stream:
        json.dump(config, stream, indent=2)
        stream.write("\n")


def run_child(checker: Path, fixture: Path, quarantine: Path,
              manifest: Path | None, mode: str, marker: str,
              expected_rc: int) -> bool:
    command = [sys.executable, str(checker), "--deck", str(fixture), "--mode", mode]
    if manifest is not None:
        command.extend(("--manifest", str(manifest)))
    if mode == "legacy-read-only":
        command.extend(("--quarantine", str(quarantine)))
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
        f"DECK_FOSSIL_CONTROL case={fixture.name} child_rc={result.returncode} "
        f"expected_rc={expected_rc} marker={'yes' if found else 'no'} "
        f"verdict={'PASS' if passed else 'FAIL'} fixture={fixture}"
    )
    if not passed:
        print(result.stdout)
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checker", type=Path, default="scripts/check_deck_fossil.py")
    parser.add_argument(
        "--deck", type=Path, default="data/tardis_reference_toy06_19p48d"
    )
    parser.add_argument(
        "--quarantine", type=Path,
        default="docs/manifests/deck_fossil_toy06_19p48d_quarantine.json",
    )
    args = parser.parse_args()
    checker = args.checker.resolve()
    source = args.deck.resolve()
    quarantine = args.quarantine.resolve()
    missing = [name for name in COMPANIONS if not (source / name).is_file()]
    if not checker.is_file() or not quarantine.is_file() or missing:
        print(
            f"DECK_FOSSIL_CONTROL setup_error checker={checker} "
            f"quarantine={quarantine} missing={missing}", file=sys.stderr,
        )
        return 1

    root = Path(tempfile.mkdtemp(prefix="lumina_deck_fossil_controls_", dir="/tmp"))
    print(f"DECK_FOSSIL_CONTROL scratch={root}")
    cases: list[tuple[Path, Path | None, str, str, int]] = []

    fixture = copy_fixture(source, root, "missing_manifest")
    cases.append((
        fixture, None, "canonical-production",
        "DECK-FOSSIL][FATAL] missing manifest", 1,
    ))

    fixture = copy_fixture(source, root, "generation_mismatch")
    manifest = baseline_manifest(fixture, quarantine)
    manifest_path = write_manifest(fixture, manifest)
    mutate_config_generation(fixture)
    cases.append((
        fixture, manifest_path, "canonical-production",
        "DECK-FOSSIL][FATAL] generation mismatch", 1,
    ))

    fixture = copy_fixture(source, root, "companion_hash_mismatch")
    manifest = baseline_manifest(fixture, quarantine)
    manifest_path = write_manifest(fixture, manifest)
    with (fixture / "plasma_state.csv").open("a") as stream:
        stream.write("\n")
    cases.append((
        fixture, manifest_path, "canonical-production",
        "DECK-FOSSIL][FATAL] companion hash mismatch", 1,
    ))

    fixture = copy_fixture(source, root, "writer_replay_mismatch")
    manifest = baseline_manifest(fixture, quarantine)
    manifest["producer"]["argv"].append("--injected-argv")
    manifest_path = write_manifest(fixture, manifest)
    cases.append((
        fixture, manifest_path, "canonical-production",
        "DECK-FOSSIL][FATAL] writer replay mismatch", 1,
    ))

    fixture = copy_fixture(source, root, "approved_fossil_legacy")
    cases.append((
        fixture, None, "legacy-read-only", "DECK-FOSSIL][WARN]", 0,
    ))

    passed = sum(
        run_child(checker, fixture, quarantine, manifest, mode, marker, expected_rc)
        for fixture, manifest, mode, marker, expected_rc in cases
    )
    print(
        f"DECK_FOSSIL_CONTROL_SUMMARY passed={passed} total={len(cases)} "
        f"scratch={root}"
    )
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
