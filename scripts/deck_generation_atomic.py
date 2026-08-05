#!/usr/bin/env python3
"""Atomic DECK-FOSSIL generation transaction.

The registered writer receives a new ``{output}`` path inside a sibling staging
directory.  Only a complete, validated four-companion generation is renamed to
the requested target.  The current toy06 canonical tree is always refused.

The writer subprocess receives exactly the JSON environment supplied with
``--environment-json``; inherited login-node variables are not hidden inputs.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid


REPO_ROOT = Path(__file__).resolve().parents[1]
CURRENT_CANONICAL = REPO_ROOT / "data/tardis_reference_toy06_19p48d"
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
REPLAY_UNITS = frozenset(("L", "r_inner", "T_inner", "W", "T_rad", "n_e"))
EPSILON_L_LIMIT = 1.0e-6
DELTA_SB_LIMIT_K = 5.0


class GenerationError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    """Hash a file or a directory tree without following directory symlinks."""
    path = path.resolve()
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise GenerationError(f"input is not a file/directory: {path}")
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
            raise GenerationError(f"unsupported input tree entry: {child}")
    return sha256_json(records)


def parse_pairs(values: list[str], label: str, numeric: bool) -> dict:
    result = {}
    for item in values:
        if "=" not in item:
            raise GenerationError(f"{label} must be KEY=VALUE: {item!r}")
        key, raw = item.split("=", 1)
        if not key or key in result:
            raise GenerationError(f"duplicate/empty {label}: {key!r}")
        if numeric:
            try:
                value = float(raw)
            except ValueError as exc:
                raise GenerationError(f"non-numeric {label}: {item!r}") from exc
            if not math.isfinite(value):
                raise GenerationError(f"non-finite {label}: {item!r}")
            result[key] = value
        else:
            result[key] = raw
    return result


def read_column(path: Path, *names: str) -> list[list[float]]:
    try:
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        ids = [int(row["shell_id"]) for row in rows]
        values = [[float(row[name]) for row in rows] for name in names]
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise GenerationError(f"bad {path.name}: {exc}") from exc
    if ids != list(range(len(rows))) or not rows:
        raise GenerationError(f"bad shell sequence in {path.name}")
    if any(not math.isfinite(value) for column in values for value in column):
        raise GenerationError(f"non-finite value in {path.name}")
    return values


def profile_sha256(values: list[float]) -> str:
    return sha256_json([float(value).hex() for value in values])


def fsync_tree(root: Path) -> None:
    """Durably flush regular files and directories before the namespace commit."""
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_symlink():
            continue
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    descriptor = os.open(root, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_target(target: Path) -> None:
    resolved = target.resolve(strict=False)
    canonical = CURRENT_CANONICAL.resolve()
    if resolved == canonical or canonical in resolved.parents:
        raise GenerationError(f"current canonical tree is immutable: {target}")
    if target.exists() or target.is_symlink():
        raise GenerationError(f"target must not exist: {target}")


def load_environment(path: Path) -> dict[str, str]:
    try:
        with path.open() as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise GenerationError(f"cannot read environment JSON: {exc}") from exc
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise GenerationError("environment JSON must be a string-to-string object")
    return value


def build_manifest(output: Path, args: argparse.Namespace, writer: Path,
                   executed_argv: list[str], environment: dict[str, str],
                   constants: dict, units: dict, started_at: str,
                   committed_at: str) -> dict:
    for name in COMPANIONS:
        if not (output / name).is_file():
            raise GenerationError(f"writer omitted companion: {name}")
    try:
        with (output / "config.json").open() as stream:
            config = json.load(stream)
        config_keys = {key: config[key] for key in CONFIG_KEYS}
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise GenerationError(f"six config keys unavailable: {exc}") from exc
    if not math.isclose(
        float(config.get("target_epoch_d", math.nan)), args.epoch_days,
        rel_tol=0.0, abs_tol=1.0e-12,
    ):
        raise GenerationError("writer output epoch differs from registered epoch")
    if "sigma_SB" not in constants:
        raise GenerationError("--constant sigma_SB=... is required")
    if REPLAY_UNITS - set(units):
        raise GenerationError(
            "missing replay units: " + ",".join(sorted(REPLAY_UNITS - set(units)))
        )

    geometry_r, = read_column(output / "geometry.csv", "r_inner")
    w_values, trad_values = read_column(output / "plasma_state.csv", "W", "T_rad")
    ne_values, = read_column(output / "electron_densities.csv", "n_e")
    if len(geometry_r) != len(w_values) or len(w_values) != len(ne_values):
        raise GenerationError("companion shell counts differ")
    generation_id = args.generation_id or f"deck-{uuid.uuid4()}"
    companions = {
        name: {
            "sha256": sha256_file(output / name),
            "generation_id": generation_id,
        }
        for name in COMPANIONS
    }
    inputs = {
        str(path.resolve()): {"sha256": sha256_path(path)} for path in args.input
    }
    producer = {
        "writer": {"path": str(writer), "sha256": sha256_file(writer)},
        "argv": executed_argv,
        "argv_template": args.writer_argv,
        "environment": environment,
        "working_directory": str(Path.cwd().resolve()),
    }
    registration = {
        "writer": producer["writer"],
        "argv": producer["argv"],
        "argv_template": producer["argv_template"],
        "environment": producer["environment"],
        "working_directory": producer["working_directory"],
        "inputs": inputs,
        "epoch": {"value": args.epoch_days, "unit": "day"},
        "constants": constants,
        "units": units,
    }
    producer["registration_sha256"] = sha256_json(registration)

    deck_l = float(config_keys["luminosity_inner_erg_s"])
    deck_t = float(config_keys["T_inner_K"])
    r_inner = geometry_r[0]
    sigma_sb = float(constants["sigma_SB"])
    t_sb = (deck_l / (4.0 * math.pi * r_inner ** 2 * sigma_sb)) ** 0.25
    delta_sb = abs(deck_t - t_sb)
    epsilon_l = 0.0
    if epsilon_l > EPSILON_L_LIMIT or delta_sb > DELTA_SB_LIMIT_K:
        raise GenerationError(
            f"replay thresholds fail: epsilon_L={epsilon_l:.17g} "
            f"Delta_SB={delta_sb:.17g} K"
        )
    return {
        "schema": "lumina.deck-generation/v1",
        "generation_id": generation_id,
        "producer": producer,
        "inputs": inputs,
        "epoch": {"value": args.epoch_days, "unit": "day"},
        "constants": constants,
        "units": units,
        "config_keys": config_keys,
        "companions": companions,
        "generation": {
            "started_at": started_at,
            "committed_at": committed_at,
            "atomic_commit": True,
            "commit_operation": "same-filesystem directory rename",
        },
        "replay": {
            "observed": {
                "luminosity_inner_erg_s": deck_l,
                "r_inner_cm": r_inner,
                "T_inner_K": deck_t,
                "W_profile_sha256": profile_sha256(w_values),
                "T_rad_profile_sha256": profile_sha256(trad_values),
                "n_e_profile_sha256": profile_sha256(ne_values),
            },
            "metrics": {"R_L": 1.0, "epsilon_L": epsilon_l, "Delta_SB_K": delta_sb},
            "thresholds": {
                "epsilon_L_max": EPSILON_L_LIMIT,
                "Delta_SB_K_max": DELTA_SB_LIMIT_K,
            },
        },
    }


def run(args: argparse.Namespace) -> int:
    target = args.target.resolve(strict=False)
    validate_target(target)
    writer = args.writer.resolve()
    if not writer.is_file():
        raise GenerationError(f"writer absent: {writer}")
    if not args.writer_argv:
        raise GenerationError("writer command is required after --")
    writer_argv = list(args.writer_argv)
    if writer_argv and writer_argv[0] == "--":
        writer_argv.pop(0)
    if "{output}" not in writer_argv:
        raise GenerationError("writer argv must contain a standalone {output}")
    resolved_arguments = {
        str(Path(item).resolve()) for item in writer_argv if item != "{output}"
    }
    if str(writer) not in resolved_arguments:
        raise GenerationError("registered writer path is absent from writer argv")
    args.writer_argv = writer_argv
    if not all(path.exists() for path in args.input):
        raise GenerationError("one or more registered inputs are absent")
    environment = load_environment(args.environment_json.resolve())
    constants = parse_pairs(args.constant, "constant", numeric=True)
    units = parse_pairs(args.unit, "unit", numeric=False)

    target.parent.mkdir(parents=True, exist_ok=True)
    stage_root = Path(tempfile.mkdtemp(prefix=f".{target.name}.stage-", dir=target.parent))
    output = stage_root / "generation"
    executed_argv = [str(output) if item == "{output}" else item for item in writer_argv]
    started_at = utc_now()
    print(f"[DECK-FOSSIL][ATOMIC] stage={stage_root} target={target}")
    try:
        result = subprocess.run(executed_argv, env=environment, check=False)
    except OSError as exc:
        raise GenerationError(f"cannot invoke writer: {exc}") from exc
    if result.returncode != 0:
        raise GenerationError(
            f"writer rc={result.returncode}; retained stage for inspection: {stage_root}"
        )
    if not output.is_dir():
        raise GenerationError(f"writer did not create {output}")
    committed_at = utc_now()
    manifest = build_manifest(
        output, args, writer, executed_argv, environment, constants, units,
        started_at, committed_at,
    )
    with (output / "generation_manifest.json").open("w") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    fsync_tree(output)
    if target.exists() or target.is_symlink():
        raise GenerationError(f"target appeared before commit: {target}")
    os.rename(output, target)
    stage_root.rmdir()
    descriptor = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    print(
        f"[DECK-FOSSIL][ATOMIC] committed generation={manifest['generation_id']} "
        f"target={target}"
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--writer", type=Path, required=True)
    parser.add_argument("--environment-json", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", default=[], required=True)
    parser.add_argument("--epoch-days", type=float, required=True)
    parser.add_argument("--constant", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--unit", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--generation-id")
    parser.add_argument("writer_argv", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except GenerationError as exc:
        print(f"[DECK-FOSSIL][FATAL] atomic generation aborted: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
