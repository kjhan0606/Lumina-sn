#!/usr/bin/env python3
"""Read-only DECK-FOSSIL generation/replay and fossil-quarantine checker.

This program has no dependency on NE-NAMING.  Fossil mode is strictly read
only.  Canonical-generation mode replays the registered writer only into a
``/tmp/lumina_deck_fossil_controls_*`` scratch directory and never writes the
inspected deck.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile


TAG = "DECK-FOSSIL"
REPO_ROOT = Path(__file__).resolve().parents[1]
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
EPSILON_L_LIMIT = 1.0e-6
DELTA_SB_LIMIT_K = 5.0


class ContractFailure(RuntimeError):
    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


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


def sha256_path(path: Path) -> str:
    path = path.resolve()
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise ContractFailure("writer replay mismatch", f"input absent: {path}")
    records: list[dict[str, str]] = []
    for child in sorted(path.rglob("*")):
        relative = child.relative_to(path).as_posix()
        if child.is_symlink():
            records.append({"path": relative, "kind": "symlink", "target": os.readlink(child)})
        elif child.is_file():
            records.append({"path": relative, "kind": "file", "sha256": sha256_file(child)})
        elif child.is_dir():
            records.append({"path": relative, "kind": "directory"})
        else:
            raise ContractFailure(
                "writer replay mismatch", f"unsupported input tree entry: {child}"
            )
    return sha256_json(records)


def read_json(path: Path, missing_reason: str) -> dict:
    try:
        with path.open() as stream:
            result = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractFailure(missing_reason, f"cannot read {path}: {exc}") from exc
    if not isinstance(result, dict):
        raise ContractFailure(missing_reason, f"{path} is not an object")
    return result


def require_mapping(parent: dict, key: str, reason: str) -> dict:
    value = parent.get(key)
    if not isinstance(value, dict) or not value:
        raise ContractFailure(reason, f"missing/empty {key}")
    return value


def registered_path(value: str, repo_root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def read_column(path: Path, *names: str) -> list[list[float]]:
    try:
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        ids = [int(row["shell_id"]) for row in rows]
        values = [[float(row[name]) for row in rows] for name in names]
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ContractFailure("writer replay mismatch", f"bad {path.name}: {exc}") from exc
    if ids != list(range(len(rows))) or not rows:
        raise ContractFailure("writer replay mismatch", f"bad shell ids in {path.name}")
    if any(not math.isfinite(value) for column in values for value in column):
        raise ContractFailure("writer replay mismatch", f"non-finite {path.name}")
    return values


def profile_sha256(values: list[float]) -> str:
    return sha256_json([float(value).hex() for value in values])


def registration_payload(manifest: dict) -> dict:
    producer = require_mapping(manifest, "producer", "writer replay mismatch")
    return {
        "writer": producer.get("writer"),
        "argv": producer.get("argv"),
        "argv_template": producer.get("argv_template"),
        "environment": producer.get("environment"),
        "working_directory": producer.get("working_directory"),
        "inputs": manifest.get("inputs"),
        "epoch": manifest.get("epoch"),
        "constants": manifest.get("constants"),
        "units": manifest.get("units"),
    }


def verify_manifest_shape(manifest: dict) -> None:
    required = (
        "schema", "generation_id", "producer", "inputs", "epoch",
        "constants", "units", "config_keys", "companions", "generation",
        "replay",
    )
    missing = [key for key in required if key not in manifest]
    if missing or manifest.get("schema") != "lumina.deck-generation/v1":
        raise ContractFailure("missing manifest", f"schema/fields differ: {missing}")
    generation = require_mapping(manifest, "generation", "missing manifest")
    if (
        not generation.get("started_at")
        or not generation.get("committed_at")
        or generation.get("atomic_commit") is not True
    ):
        raise ContractFailure("missing manifest", "atomic start/commit record absent")
    if not isinstance(manifest.get("inputs"), dict):
        raise ContractFailure("missing manifest", "input hashes absent")
    if not isinstance(manifest.get("constants"), dict) or not isinstance(
        manifest.get("units"), dict
    ):
        raise ContractFailure("missing manifest", "constants/units absent")


def verify_generation(manifest: dict, deck: Path) -> dict:
    generation_id = manifest["generation_id"]
    companions = require_mapping(manifest, "companions", "generation mismatch")
    declared_config = require_mapping(manifest, "config_keys", "generation mismatch")
    config = read_json(deck / "config.json", "generation mismatch")
    if set(declared_config) != set(CONFIG_KEYS) or any(
        config.get(key) != declared_config.get(key) for key in CONFIG_KEYS
    ):
        raise ContractFailure("generation mismatch", "six config keys differ")
    if any(name not in companions for name in COMPANIONS):
        raise ContractFailure("generation mismatch", "companion record absent")
    generations = {
        companions[name].get("generation_id")
        for name in COMPANIONS
        if isinstance(companions.get(name), dict)
    }
    if generations != {generation_id}:
        raise ContractFailure(
            "generation mismatch", f"top={generation_id!r} companions={sorted(map(str, generations))}"
        )
    return config


def verify_companion_hashes(manifest: dict, deck: Path) -> None:
    companions = manifest["companions"]
    for name in COMPANIONS:
        path = deck / name
        if not path.is_file() or companions[name].get("sha256") != sha256_file(path):
            raise ContractFailure("companion hash mismatch", name)


def verify_writer_registration(manifest: dict, repo_root: Path) -> None:
    producer = require_mapping(manifest, "producer", "writer replay mismatch")
    writer = producer.get("writer")
    argv = producer.get("argv")
    argv_template = producer.get("argv_template")
    environment = producer.get("environment")
    working_directory = producer.get("working_directory")
    if (
        not isinstance(writer, dict)
        or not writer.get("path")
        or not writer.get("sha256")
        or not isinstance(argv, list)
        or not argv
        or not all(isinstance(item, str) for item in argv)
        or not isinstance(argv_template, list)
        or "{output}" not in argv_template
        or not all(isinstance(item, str) for item in argv_template)
        or not isinstance(environment, dict)
        or not all(isinstance(key, str) and isinstance(value, str)
                   for key, value in environment.items())
        or not isinstance(working_directory, str)
        or not Path(working_directory).is_dir()
    ):
        raise ContractFailure("writer replay mismatch", "writer/argv/env declaration absent")
    writer_path = registered_path(str(writer["path"]), repo_root)
    if not writer_path.is_file() or sha256_file(writer_path) != writer["sha256"]:
        raise ContractFailure("writer replay mismatch", "writer hash differs")
    template_paths = {
        str((Path(working_directory) / item).resolve())
        if not Path(item).is_absolute() else str(Path(item).resolve())
        for item in argv_template if item != "{output}"
    }
    if str(writer_path) not in template_paths:
        raise ContractFailure("writer replay mismatch", "writer absent from replay argv")
    if producer.get("registration_sha256") != sha256_json(registration_payload(manifest)):
        raise ContractFailure("writer replay mismatch", "registration attestation differs")
    inputs = manifest["inputs"]
    for label, record in inputs.items():
        if not isinstance(record, dict) or not record.get("sha256"):
            raise ContractFailure("writer replay mismatch", f"input hash absent: {label}")
        path = registered_path(label, repo_root)
        if not path.exists() or sha256_path(path) != record["sha256"]:
            raise ContractFailure("writer replay mismatch", f"input hash differs: {label}")


def replay_registered_writer(manifest: dict) -> Path:
    producer = manifest["producer"]
    template = list(producer["argv_template"])
    if template and template[0] == "--":
        template.pop(0)
    scratch = Path(tempfile.mkdtemp(
        prefix="lumina_deck_fossil_controls_replay_", dir="/tmp"
    ))
    output = scratch / "generation"
    command = [str(output) if item == "{output}" else item for item in template]
    try:
        result = subprocess.run(
            command,
            cwd=producer["working_directory"],
            env=producer["environment"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except OSError as exc:
        raise ContractFailure("writer replay mismatch", f"cannot invoke writer: {exc}") from exc
    print(f"[{TAG}][TRACE] replay_scratch={scratch} writer_rc={result.returncode}")
    if result.returncode != 0 or not output.is_dir():
        detail = result.stdout[-4000:] if result.stdout else "writer produced no output"
        raise ContractFailure("writer replay mismatch", detail)
    return output


def verify_replay(manifest: dict, deck: Path, config: dict,
                  replay_deck: Path) -> tuple[float, float]:
    replay = require_mapping(manifest, "replay", "writer replay mismatch")
    observed = require_mapping(replay, "observed", "writer replay mismatch")
    thresholds = require_mapping(replay, "thresholds", "writer replay mismatch")
    constants = manifest["constants"]
    units = manifest["units"]
    required_units = {"L", "r_inner", "T_inner", "W", "T_rad", "n_e"}
    if required_units - set(units):
        raise ContractFailure("writer replay mismatch", "replay units absent")
    try:
        sigma_sb = float(constants["sigma_SB"])
        expected_l = float(observed["luminosity_inner_erg_s"])
        expected_r = float(observed["r_inner_cm"])
        expected_t = float(observed["T_inner_K"])
        deck_l = float(config["luminosity_inner_erg_s"])
        deck_t = float(config["T_inner_K"])
        manifest_epoch = float(manifest["epoch"]["value"])
        deck_epoch = float(config["target_epoch_d"])
        epsilon_limit = float(thresholds["epsilon_L_max"])
        delta_limit = float(thresholds["Delta_SB_K_max"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractFailure("writer replay mismatch", f"scalar declaration absent: {exc}") from exc
    if not all(math.isfinite(value) and value > 0.0 for value in (
        sigma_sb, expected_l, expected_r, expected_t, deck_l, deck_t
    )):
        raise ContractFailure("writer replay mismatch", "replay scalar is not finite positive")
    if (
        manifest["epoch"].get("unit") != "day"
        or not math.isclose(manifest_epoch, deck_epoch, rel_tol=0.0, abs_tol=1.0e-12)
        or epsilon_limit != EPSILON_L_LIMIT
        or delta_limit != DELTA_SB_LIMIT_K
    ):
        raise ContractFailure("writer replay mismatch", "epoch/replay thresholds differ")

    replay_config = read_json(replay_deck / "config.json", "writer replay mismatch")
    geometry_r, = read_column(deck / "geometry.csv", "r_inner")
    replay_geometry_r, = read_column(replay_deck / "geometry.csv", "r_inner")
    w_values, trad_values = read_column(deck / "plasma_state.csv", "W", "T_rad")
    replay_w, replay_trad = read_column(
        replay_deck / "plasma_state.csv", "W", "T_rad"
    )
    ne_values, = read_column(deck / "electron_densities.csv", "n_e")
    replay_ne, = read_column(replay_deck / "electron_densities.csv", "n_e")
    if len(geometry_r) != len(w_values) or len(w_values) != len(ne_values):
        raise ContractFailure("writer replay mismatch", "companion shell counts differ")
    profiles = {
        "W_profile_sha256": profile_sha256(w_values),
        "T_rad_profile_sha256": profile_sha256(trad_values),
        "n_e_profile_sha256": profile_sha256(ne_values),
    }
    if any(observed.get(key) != value for key, value in profiles.items()):
        raise ContractFailure("writer replay mismatch", "registered W/T_rad/n_e differs")
    replay_profiles = {
        "W_profile_sha256": profile_sha256(replay_w),
        "T_rad_profile_sha256": profile_sha256(replay_trad),
        "n_e_profile_sha256": profile_sha256(replay_ne),
    }
    try:
        replay_l = float(replay_config["luminosity_inner_erg_s"])
        replay_t = float(replay_config["T_inner_K"])
        replay_r = float(replay_geometry_r[0])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractFailure("writer replay mismatch", "replay scalar absent") from exc
    if (
        not math.isclose(expected_l, deck_l, rel_tol=0.0, abs_tol=0.0)
        or not math.isclose(expected_r, geometry_r[0], rel_tol=0.0, abs_tol=0.0)
        or not math.isclose(expected_t, deck_t, rel_tol=0.0, abs_tol=0.0)
        or replay_profiles != profiles
        or any(replay_config.get(key) != config.get(key) for key in CONFIG_KEYS)
    ):
        raise ContractFailure("writer replay mismatch", "registered/replayed deck differs")
    if not math.isclose(replay_r, geometry_r[0], rel_tol=0.0, abs_tol=0.0):
        raise ContractFailure("writer replay mismatch", "r_inner replay differs")
    if not math.isclose(replay_t, deck_t, rel_tol=0.0, abs_tol=0.0):
        raise ContractFailure("writer replay mismatch", "T_inner replay differs")

    ratio_l = replay_l / deck_l
    epsilon_l = abs(ratio_l - 1.0)
    t_sb = (deck_l / (4.0 * math.pi * geometry_r[0] ** 2 * sigma_sb)) ** 0.25
    delta_sb = abs(deck_t - t_sb)
    if epsilon_l > EPSILON_L_LIMIT or delta_sb > DELTA_SB_LIMIT_K:
        raise ContractFailure(
            "writer replay mismatch",
            f"R_L={ratio_l:.17g} epsilon_L={epsilon_l:.17g} Delta_SB={delta_sb:.17g} K",
        )
    recorded = replay.get("metrics")
    if not isinstance(recorded, dict):
        raise ContractFailure("writer replay mismatch", "replay metrics absent")
    expected_metrics = {
        "R_L": ratio_l,
        "epsilon_L": epsilon_l,
        "Delta_SB_K": delta_sb,
    }
    for key, value in expected_metrics.items():
        try:
            declared = float(recorded[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractFailure("writer replay mismatch", f"metric absent: {key}") from exc
        if not math.isclose(value, declared, rel_tol=2.0e-13, abs_tol=1.0e-12):
            raise ContractFailure("writer replay mismatch", f"metric differs: {key}")
    return epsilon_l, delta_sb


def verify_fossil_quarantine(record: dict, deck: Path) -> tuple[float, float]:
    if (
        record.get("schema") != "lumina.deck-fossil-quarantine/v1"
        or record.get("state") != "APPROVED_FOSSIL_QUARANTINE"
        or record.get("producer") != "UNRESOLVED"
        or record.get("canonical_production_eligible") is not False
        or "legacy-read-only" not in record.get("allowed_modes", ())
    ):
        raise ContractFailure("missing manifest", "fossil approval/state differs")
    sealed = require_mapping(record, "exact_sha256", "companion hash mismatch")
    for name in COMPANIONS:
        path = deck / name
        if not path.is_file() or sealed.get(name) != sha256_file(path):
            raise ContractFailure("companion hash mismatch", f"fossil seal differs: {name}")
    hypotheses = record.get("rejected_hypotheses")
    expected = {
        "epoch_misselection", "wavelength_truncation", "fitted_quarter_constant",
        "git_untracked_history",
    }
    if not isinstance(hypotheses, list) or {
        item.get("id") for item in hypotheses if isinstance(item, dict)
    } != expected:
        raise ContractFailure("missing manifest", "rejected-hypothesis census differs")
    metrics = require_mapping(record, "metrics", "missing manifest")
    try:
        ratio_l = float(metrics["R_L"])
        epsilon_l = float(metrics["epsilon_L"])
        delta_sb = float(metrics["Delta_SB_K"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractFailure("missing manifest", "fossil metrics absent") from exc
    if not (
        math.isclose(ratio_l, 4.005038, rel_tol=0.0, abs_tol=5.0e-7)
        and math.isclose(epsilon_l, 3.005038, rel_tol=0.0, abs_tol=5.0e-7)
        and math.isclose(delta_sb, 1.65, rel_tol=0.0, abs_tol=5.0e-3)
    ):
        raise ContractFailure("missing manifest", "fossil preregistered metrics differ")
    return epsilon_l, delta_sb


def run(args: argparse.Namespace) -> int:
    deck = args.deck.resolve()
    if not deck.is_dir():
        raise ContractFailure("missing manifest", f"deck absent: {deck}")
    if args.mode == "legacy-read-only":
        if args.quarantine is None:
            raise ContractFailure("missing manifest", "--quarantine is required")
        record = read_json(args.quarantine.resolve(), "missing manifest")
        epsilon_l, delta_sb = verify_fossil_quarantine(record, deck)
        print(
            f"[{TAG}][WARN] producer=UNRESOLVED mode=legacy-read-only "
            f"epsilon_L={epsilon_l:.6f} Delta_SB={delta_sb:.2f}K "
            "canonical_production_eligible=no"
        )
        return 0

    manifest_path = (
        args.manifest.resolve()
        if args.manifest is not None
        else deck / "generation_manifest.json"
    )
    if not manifest_path.is_file():
        raise ContractFailure("missing manifest", str(manifest_path))
    manifest = read_json(manifest_path, "missing manifest")
    verify_manifest_shape(manifest)
    config = verify_generation(manifest, deck)
    verify_companion_hashes(manifest, deck)
    verify_writer_registration(manifest, args.repo_root.resolve())
    replay_deck = replay_registered_writer(manifest)
    epsilon_l, delta_sb = verify_replay(manifest, deck, config, replay_deck)
    print(
        f"[{TAG}][PASS] generation={manifest['generation_id']} "
        f"epsilon_L={epsilon_l:.17g} Delta_SB={delta_sb:.17g}K"
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("canonical-production", "legacy-read-only"),
        default="canonical-production",
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--quarantine", type=Path)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except ContractFailure as exc:
        suffix = f": {exc.detail}" if exc.detail else ""
        print(f"[{TAG}][FATAL] {exc.reason}{suffix}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[{TAG}][FATAL] missing manifest: unexpected {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
