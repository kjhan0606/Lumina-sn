#!/usr/bin/env python3
"""Read-only NE-NAMING contract checker.

This checker is intentionally independent of DECK-FOSSIL.  It validates the
electron-density mode/provenance/approval before evaluating the registered
``tau_i`` and ``i_phot`` formulae.  It never writes the inspected deck.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys


TAG = "NE-NAMING"
PLACEHOLDER_MODE = "PLACEHOLDER_ZBAR_ONE"
TRUE_MODE = "CMFGEN_CHARGE_BALANCE"
SUPPORTED_MODES = frozenset((PLACEHOLDER_MODE, TRUE_MODE))
REQUIRED_COMPANIONS = (
    "config.json",
    "geometry.csv",
    "electron_densities.csv",
    "plasma_state.csv",
)
REPO_ROOT = Path(__file__).resolve().parents[1]


class ContractFailure(RuntimeError):
    """A fail-closed NE-NAMING verdict with a preregistered reason."""

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


def registered_path(value: str, repo_root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def read_json(path: Path) -> dict:
    try:
        with path.open() as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractFailure("missing provenance", f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractFailure("missing provenance", f"{path} is not a JSON object")
    return value


def require_mapping(parent: dict, key: str) -> dict:
    value = parent.get(key)
    if not isinstance(value, dict) or not value:
        raise ContractFailure("missing provenance", f"missing/empty {key}")
    return value


def verify_registered_provenance(manifest: dict, repo_root: Path,
                                 production: bool) -> None:
    required = (
        "formula", "applicable_zones", "builder", "inputs", "tau_phot",
        "approved_disposition",
    )
    missing = [key for key in required if key not in manifest]
    if missing:
        raise ContractFailure("missing provenance", ",".join(missing))
    if manifest.get("formula") != "n_e = n_atom * 1.0" and (
        manifest.get("electron_density_mode") == PLACEHOLDER_MODE
    ):
        raise ContractFailure("missing provenance", "placeholder formula differs")
    builder = require_mapping(manifest, "builder")
    if not builder.get("path") or not builder.get("sha256"):
        raise ContractFailure("missing provenance", "builder path/hash absent")
    status = str(builder.get("producer_status", ""))
    if production and not status.startswith("REGISTERED"):
        raise ContractFailure("missing provenance", f"producer_status={status!r}")
    builder_path = registered_path(str(builder["path"]), repo_root)
    if not builder_path.is_file() or sha256_file(builder_path) != builder["sha256"]:
        raise ContractFailure("missing provenance", "builder hash differs")
    inputs = require_mapping(manifest, "inputs")
    for label, record in inputs.items():
        if not isinstance(record, dict) or not record.get("sha256"):
            raise ContractFailure("missing provenance", f"input hash absent: {label}")
        path = registered_path(label, repo_root)
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise ContractFailure("missing provenance", f"input hash differs: {label}")
    try:
        tau_phot = float(manifest["tau_phot"])
    except (TypeError, ValueError) as exc:
        raise ContractFailure("missing provenance", "tau_phot is not numeric") from exc
    if not math.isfinite(tau_phot) or tau_phot <= 0.0:
        raise ContractFailure("missing provenance", "tau_phot is not finite positive")
    if manifest["approved_disposition"] not in ("A", "B"):
        raise ContractFailure("missing provenance", "unknown disposition")


def verify_true_path_epoch(manifest: dict, deck: Path) -> None:
    source = require_mapping(manifest, "source")
    specification = require_mapping(manifest, "true_path_specification")
    required_metadata = require_mapping(specification, "required_metadata")
    needed = {
        "units", "ND", "interpolation", "duplicates", "non_monotonic",
        "coverage", "outside_grid_policy",
    }
    if needed - set(required_metadata):
        raise ContractFailure(
            "missing provenance",
            "true-path metadata absent: " + ",".join(sorted(needed - set(required_metadata))),
        )
    config = read_json(deck / "config.json")
    try:
        source_epoch = float(source["epoch_days"])
        deck_epoch = float(config["target_epoch_d"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractFailure("epoch mismatch", "epoch declaration absent") from exc
    if not (math.isfinite(source_epoch) and math.isfinite(deck_epoch)) or not math.isclose(
        source_epoch, deck_epoch, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ContractFailure(
            "epoch mismatch", f"source={source_epoch!r} deck={deck_epoch!r}"
        )


def verify_generation_and_hashes(manifest: dict, deck: Path) -> None:
    generation_id = manifest.get("generation_id")
    companions = require_mapping(manifest, "companions")
    if not generation_id or any(name not in companions for name in REQUIRED_COMPANIONS):
        raise ContractFailure("generation mismatch", "generation/companion record absent")
    generations = {
        companions[name].get("generation_id")
        for name in REQUIRED_COMPANIONS
        if isinstance(companions.get(name), dict)
    }
    if generations != {generation_id}:
        raise ContractFailure(
            "generation mismatch", f"top={generation_id!r} companions={sorted(map(str, generations))}"
        )
    for name in REQUIRED_COMPANIONS:
        record = companions[name]
        path = deck / name
        if not path.is_file() or record.get("sha256") != sha256_file(path):
            raise ContractFailure("generation mismatch", f"hash differs: {name}")


def verify_legacy_approval(manifest: dict, deck: Path, token: str | None) -> None:
    approval = require_mapping(manifest, "approval")
    if approval.get("scope") != "legacy-read-only" or token != approval.get("token"):
        raise ContractFailure("unapproved placeholder", "legacy scope/token differs")
    sealed = require_mapping(approval, "exact_deck_sha256")
    for name in REQUIRED_COMPANIONS:
        path = deck / name
        if sealed.get(name) != sha256_file(path):
            raise ContractFailure("generation mismatch", f"legacy seal differs: {name}")


def read_column(path: Path, name: str) -> tuple[list[int], list[float]]:
    try:
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        ids = [int(row["shell_id"]) for row in rows]
        values = [float(row[name]) for row in rows]
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ContractFailure("missing provenance", f"cannot parse {path.name}:{name}: {exc}") from exc
    if ids != list(range(len(rows))) or not values or not all(map(math.isfinite, values)):
        raise ContractFailure("missing provenance", f"invalid shell sequence in {path.name}")
    return ids, values


def reproduce_zbar(manifest: dict, deck: Path, n_e: list[float]) -> list[float]:
    record = require_mapping(manifest, "zbar_reproduction")
    masses_raw = require_mapping(record, "molar_mass_g_mol")
    try:
        avogadro = float(record["N_A_mol_inverse"])
        masses = {int(z): float(value) for z, value in masses_raw.items()}
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractFailure("missing provenance", "invalid Zbar constants") from exc
    _, density = read_column(deck / "density.csv", "rho")
    try:
        with (deck / "abundances.csv").open(newline="") as stream:
            abundance_rows = list(csv.DictReader(stream))
        abundance = {
            int(row["atomic_number"]): [float(row[str(shell)]) for shell in range(len(n_e))]
            for row in abundance_rows
        }
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ContractFailure("missing provenance", f"cannot reproduce n_atom: {exc}") from exc
    if len(density) != len(n_e) or set(abundance) - set(masses):
        raise ContractFailure("missing provenance", "n_atom input shape/atomic masses differ")
    n_atom = [
        density[shell] * avogadro * sum(
            abundance[z][shell] / masses[z] for z in abundance
        )
        for shell in range(len(n_e))
    ]
    if any((not math.isfinite(value) or value <= 0.0) for value in n_atom):
        raise ContractFailure("missing provenance", "n_atom is not finite positive")
    return [n_e[shell] / n_atom[shell] for shell in range(len(n_e))]


def reproduce_tau(radius: list[float], n_e: list[float], sigma_t: float,
                  tau_phot: float) -> tuple[list[float], int]:
    if len(radius) != len(n_e) or len(radius) < 2:
        raise ContractFailure("missing provenance", "tau grid shape differs")
    tau = [0.0] * len(n_e)
    for index in range(len(n_e) - 2, -1, -1):
        dr = radius[index + 1] - radius[index]
        if not math.isfinite(dr) or dr <= 0.0:
            raise ContractFailure("missing provenance", "tau radius is non-monotonic")
        tau[index] = tau[index + 1] + 0.5 * (
            n_e[index] + n_e[index + 1]
        ) * sigma_t * dr
    above = [index for index, value in enumerate(tau) if value >= tau_phot]
    return tau, max(above) if above else 0


def close_enough(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=2.0e-13, abs_tol=1.0e-12)


def verify_embedded_boundary(manifest: dict) -> None:
    boundary = manifest.get("boundary_reproduction")
    if not isinstance(boundary, dict) or "radius_cm" not in boundary:
        return
    try:
        radius = list(map(float, boundary["radius_cm"]))
        n_e = list(map(float, boundary["n_e_cm3"]))
        n_atom = list(map(float, boundary["n_atom_cm3"]))
        recorded_zbar = list(map(float, boundary["Zbar_s"]))
        recorded_tau = list(map(float, boundary["tau_i"]))
        sigma_t = float(manifest["sigma_T_cm2"])
        tau_phot = float(manifest["tau_phot"])
        recorded_i = int(boundary["i_phot"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractFailure("missing provenance", f"bad embedded boundary: {exc}") from exc
    if not (len(radius) == len(n_e) == len(n_atom) == len(recorded_zbar)):
        raise ContractFailure("missing provenance", "embedded boundary shape differs")
    zbar = [electron / atom for electron, atom in zip(n_e, n_atom, strict=True)]
    tau, i_phot = reproduce_tau(radius, n_e, sigma_t, tau_phot)
    if (
        len(tau) != len(recorded_tau)
        or any(not close_enough(a, b) for a, b in zip(zbar, recorded_zbar, strict=True))
        or any(not close_enough(a, b) for a, b in zip(tau, recorded_tau, strict=True))
        or i_phot != recorded_i
    ):
        raise ContractFailure("missing provenance", "embedded tau/Zbar/i_phot replay differs")


def reproduce_deck_boundary(manifest: dict, deck: Path) -> None:
    _, n_e = read_column(deck / "electron_densities.csv", "n_e")
    _, radius = read_column(deck / "geometry.csv", "r_inner")
    _, velocity = read_column(deck / "geometry.csv", "v_inner")
    zbar = reproduce_zbar(manifest, deck, n_e)
    try:
        sigma_t = float(manifest["sigma_T_cm2"])
        tau_phot = float(manifest["tau_phot"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractFailure("missing provenance", "sigma_T/tau_phot absent") from exc
    tau, i_phot = reproduce_tau(radius, n_e, sigma_t, tau_phot)
    config = read_json(deck / "config.json")
    chain_exact = math.isclose(
        float(config["v_inner_min_cm_s"]), velocity[0], rel_tol=0.0, abs_tol=0.0
    )
    print(f"[{TAG}][TRACE] Zbar_s={json.dumps(zbar, separators=(',', ':'))}")
    print(f"[{TAG}][TRACE] tau_i={json.dumps(tau, separators=(',', ':'))}")
    print(
        f"[{TAG}][TRACE] i_phot={i_phot} tau_phot={tau_phot:.17g} "
        f"chain_case_A_v_inner_exact={'yes' if chain_exact else 'no'} "
        "impact=UNQUANTIFIED_PENDING_CLEAN_ZBAR"
    )
    verify_embedded_boundary(manifest)


def run(args: argparse.Namespace) -> int:
    deck = args.deck.resolve()
    if not deck.is_dir():
        raise ContractFailure("missing provenance", f"deck absent: {deck}")
    manifest = read_json(args.manifest.resolve())

    # Preregistered order: mode and authorization/provenance checks happen
    # before reproduce_deck_boundary(), which is the only i_phot evaluation.
    mode = manifest.get("electron_density_mode")
    if not mode:
        raise ContractFailure("missing mode")
    if mode not in SUPPORTED_MODES:
        raise ContractFailure("missing provenance", f"unsupported mode={mode!r}")

    production = args.claim in ("production", "canonical")
    if mode == PLACEHOLDER_MODE and production:
        approval = manifest.get("approval", {})
        if args.approval_token is None or args.approval_token != approval.get("token"):
            raise ContractFailure("unapproved placeholder")
        raise ContractFailure("placeholder production blocked", "disposition A")

    verify_registered_provenance(manifest, args.repo_root.resolve(), production)
    if mode == TRUE_MODE:
        verify_true_path_epoch(manifest, deck)
        if manifest.get("approved_disposition") != "B":
            raise ContractFailure("true path not enabled", "disposition A")

    if args.claim == "legacy-read-only":
        if mode != PLACEHOLDER_MODE:
            raise ContractFailure("unapproved placeholder", "legacy mode requires placeholder")
        verify_legacy_approval(manifest, deck, args.approval_token)
    elif args.claim == "diagnostic" and mode == PLACEHOLDER_MODE:
        approval = require_mapping(manifest, "approval")
        if (
            approval.get("scope") != "diagnostic"
            or args.approval_token is None
            or args.approval_token != approval.get("token")
        ):
            raise ContractFailure("unapproved placeholder", "diagnostic scope/token differs")

    verify_generation_and_hashes(manifest, deck)
    reproduce_deck_boundary(manifest, deck)

    if args.claim in ("legacy-read-only", "diagnostic"):
        print(
            f"[{TAG}][WARN] mode={mode} claim={args.claim} disposition="
            f"{manifest['approved_disposition']} read_only=yes"
        )
        return 0
    print(f"[{TAG}][PASS] mode={mode} claim={args.claim}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--claim",
        choices=("production", "canonical", "diagnostic", "legacy-read-only"),
        default="production",
    )
    parser.add_argument("--approval-token")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except ContractFailure as exc:
        suffix = f": {exc.detail}" if exc.detail else ""
        print(f"[{TAG}][FATAL] {exc.reason}{suffix}", file=sys.stderr)
        return 1
    except Exception as exc:  # Fail closed with the contract's fixed rc space.
        print(f"[{TAG}][FATAL] missing provenance: unexpected {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
