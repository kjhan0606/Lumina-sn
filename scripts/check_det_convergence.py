#!/usr/bin/env python3
"""Fail-closed convergence gate for committed Lumina DET comparison dumps.

The production DET driver runs a fixed number of outer iterations.  This tool
therefore treats process success and solver convergence as separate facts.  A
flight passes only when every one of the requested final consecutive
transitions satisfies the declared material, energy-ledger, and spectral
stability limits.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "LUMINA_PHYSICS_COMPARISON_V1"
MANIFEST_RE = re.compile(r"^physics_([A-Za-z0-9_-]+)_iter([0-9]{4})\.manifest\.json$")

SHELL_MAX_LIMITS = {
    "T_e_K": 5.0e-3,
    "n_e_cm3": 1.0e-2,
    "u_atom_erg": 1.0e-2,
}
ENERGY_L1_LIMITS = {
    "q_ad_temperature_gradient": 1.0e-2,
    "q_ad_velocity_divergence": 1.0e-2,
    "q_ad_electron_fraction_gradient": 1.0e-2,
    "q_ad_internal_energy_gradient": 1.0e-2,
    "q_ad_signed_total": 1.0e-2,
    "sum_heating": 1.0e-2,
    "sum_cooling": 1.0e-2,
}
SPECTRAL_L1_LIMITS = {
    "J_nu": 1.0e-2,
    "chi_total_cm1": 2.0e-2,
    "eta_true_total": 2.0e-2,
}

SHELL_REQUIRED = {
    "shell_id", "r_inner_cm", "r_outer_cm", "v_inner_cm_s",
    "v_outer_cm_s", "T_e_K", "n_e_cm3", "n_atom_cm3", "u_atom_erg",
    *ENERGY_L1_LIMITS,
    "residual",
}
SPECTRAL_REQUIRED = {
    "shell_id", "bin_id", "nu_lo_Hz", "nu_hi_Hz", *SPECTRAL_L1_LIMITS,
}


class GateError(RuntimeError):
    pass


def finite(value: str, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise GateError(f"non-numeric {context}: {value!r}") from exc
    if not math.isfinite(result):
        raise GateError(f"non-finite {context}: {value!r}")
    return result


def safe_member(base: Path, name: Any, label: str) -> Path:
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise GateError(f"unsafe {label}: {name!r}")
    path = base / name
    if not path.is_file() or path.is_symlink():
        raise GateError(f"missing or non-regular {label}: {path}")
    return path


def load_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        missing = sorted(required - fields)
        if missing:
            raise GateError(f"{path.name}: missing columns {missing}")
        rows = list(reader)
    if not rows:
        raise GateError(f"{path.name}: empty CSV")
    return rows


def load_snapshot(manifest_path: Path, lane: str, iteration: int,
                  expected_bins: int) -> dict[str, Any]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"cannot read manifest {manifest_path}: {exc}") from exc
    required_manifest = {
        "schema", "transaction_status", "code", "lane", "iteration",
        "epoch_s", "n_shells", "n_bins", "atomic_model_sha256",
        "geometry_sha256", "grid_manifest_sha256", "radiation_generation",
        "population_generation", "te_generation", "opacity_generation",
        "emissivity_generation", "shell_file", "spectral_file",
    }
    missing = sorted(required_manifest - set(manifest))
    if missing:
        raise GateError(f"{manifest_path.name}: missing keys {missing}")
    if manifest["schema"] != SCHEMA or manifest["transaction_status"] != "COMMITTED":
        raise GateError(f"{manifest_path.name}: transaction is not committed {SCHEMA}")
    if manifest["code"] != "LUMINA" or manifest["lane"] != lane:
        raise GateError(f"{manifest_path.name}: wrong code/lane")
    if manifest["iteration"] != iteration:
        raise GateError(f"{manifest_path.name}: iteration mismatch")
    ns = manifest["n_shells"]
    nb = manifest["n_bins"]
    if not isinstance(ns, int) or ns < 2 or nb != expected_bins:
        raise GateError(f"{manifest_path.name}: shape ns={ns!r} nb={nb!r}")
    shell_path = safe_member(manifest_path.parent, manifest["shell_file"], "shell_file")
    spectral_path = safe_member(manifest_path.parent, manifest["spectral_file"],
                                "spectral_file")
    shell = load_csv(shell_path, SHELL_REQUIRED)
    spectral = load_csv(spectral_path, SPECTRAL_REQUIRED)
    if len(shell) != ns or len(spectral) != ns * nb:
        raise GateError(
            f"{manifest_path.name}: row count shell={len(shell)}/{ns} "
            f"spectral={len(spectral)}/{ns * nb}"
        )
    for index, row in enumerate(shell):
        if int(row["shell_id"]) != index:
            raise GateError(f"{shell_path.name}: non-canonical shell row {index}")
        rin = finite(row["r_inner_cm"], f"shell {index} r_inner")
        rout = finite(row["r_outer_cm"], f"shell {index} r_outer")
        if not rout > rin:
            raise GateError(f"{shell_path.name}: invalid shell volume {index}")
        for field in SHELL_REQUIRED - {"shell_id"}:
            finite(row[field], f"shell {index} {field}")
    for index, row in enumerate(spectral):
        shell_id, bin_id = divmod(index, nb)
        if int(row["shell_id"]) != shell_id or int(row["bin_id"]) != bin_id:
            raise GateError(f"{spectral_path.name}: non-canonical spectral row {index}")
        lo = finite(row["nu_lo_Hz"], f"cell {index} nu_lo")
        hi = finite(row["nu_hi_Hz"], f"cell {index} nu_hi")
        if not hi > lo:
            raise GateError(f"{spectral_path.name}: invalid frequency cell {index}")
        for field in SPECTRAL_L1_LIMITS:
            finite(row[field], f"cell {index} {field}")
    return {"manifest": manifest, "shell": shell, "spectral": spectral}


def max_symmetric_relative(old: Iterable[float], new: Iterable[float]) -> float:
    result = 0.0
    for a, b in zip(old, new):
        scale = max(abs(a), abs(b))
        value = abs(b - a) / scale if scale else 0.0
        result = max(result, value)
    return result


def weighted_l1(old: Iterable[float], new: Iterable[float],
                weights: Iterable[float]) -> float:
    numerator = 0.0
    denominator = 0.0
    for a, b, weight in zip(old, new, weights):
        numerator += weight * abs(b - a)
        denominator += weight * max(abs(a), abs(b))
    return numerator / denominator if denominator else (0.0 if numerator == 0.0 else math.inf)


def energy_scaled_l1(old_rows: list[dict[str, str]],
                     new_rows: list[dict[str, str]], field: str,
                     volumes: list[float]) -> float:
    numerator = 0.0
    denominator = 0.0
    for old, new, volume in zip(old_rows, new_rows, volumes):
        a = finite(old[field], f"old {field}")
        b = finite(new[field], f"new {field}")
        numerator += volume * abs(b - a)
        scale = max(
            abs(finite(old["sum_heating"], "old sum_heating")),
            abs(finite(old["sum_cooling"], "old sum_cooling")),
            abs(finite(new["sum_heating"], "new sum_heating")),
            abs(finite(new["sum_cooling"], "new sum_cooling")),
        )
        denominator += volume * scale
    return numerator / denominator if denominator else (0.0 if numerator == 0.0 else math.inf)


def compare_pair(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    old_shell = old["shell"]
    new_shell = new["shell"]
    old_spectral = old["spectral"]
    new_spectral = new["spectral"]
    nb = old["manifest"]["n_bins"]
    volumes: list[float] = []
    for index, (a, b) in enumerate(zip(old_shell, new_shell)):
        identity = ("r_inner_cm", "r_outer_cm", "v_inner_cm_s", "v_outer_cm_s")
        if any(a[field] != b[field] for field in identity):
            raise GateError(f"shell geometry changed at shell {index}")
        rin = finite(a["r_inner_cm"], "r_inner")
        rout = finite(a["r_outer_cm"], "r_outer")
        volumes.append(rout ** 3 - rin ** 3)
    cell_weights: list[float] = []
    for index, (a, b) in enumerate(zip(old_spectral, new_spectral)):
        if a["nu_lo_Hz"] != b["nu_lo_Hz"] or a["nu_hi_Hz"] != b["nu_hi_Hz"]:
            raise GateError(f"frequency grid changed at cell {index}")
        shell_id = index // nb
        dnu = finite(a["nu_hi_Hz"], "nu_hi") - finite(a["nu_lo_Hz"], "nu_lo")
        cell_weights.append(volumes[shell_id] * dnu)

    metrics: dict[str, float] = {}
    limits: dict[str, float] = {}
    for field, limit in SHELL_MAX_LIMITS.items():
        metrics[f"{field}.max_symmetric_relative"] = max_symmetric_relative(
            (finite(row[field], field) for row in old_shell),
            (finite(row[field], field) for row in new_shell),
        )
        limits[f"{field}.max_symmetric_relative"] = limit
    for field, limit in ENERGY_L1_LIMITS.items():
        key = f"{field}.energy_scaled_l1"
        if field in {"sum_heating", "sum_cooling"}:
            metrics[key] = weighted_l1(
                (finite(row[field], field) for row in old_shell),
                (finite(row[field], field) for row in new_shell), volumes,
            )
        else:
            metrics[key] = energy_scaled_l1(old_shell, new_shell, field, volumes)
        limits[key] = limit
    for field, limit in SPECTRAL_L1_LIMITS.items():
        key = f"{field}.volume_frequency_weighted_l1"
        metrics[key] = weighted_l1(
            (finite(row[field], field) for row in old_spectral),
            (finite(row[field], field) for row in new_spectral), cell_weights,
        )
        limits[key] = limit

    final_balance = 0.0
    for row in new_shell:
        heating = abs(finite(row["sum_heating"], "sum_heating"))
        cooling = abs(finite(row["sum_cooling"], "sum_cooling"))
        residual = abs(finite(row["residual"], "residual"))
        final_balance = max(final_balance, residual / max(heating + cooling, sys.float_info.min))
    metrics["final_energy_balance.max_relative"] = final_balance
    limits["final_energy_balance.max_relative"] = 1.0e-3
    failures = [key for key, value in metrics.items()
                if not math.isfinite(value) or value > limits[key]]
    return {"metrics": metrics, "limits": limits, "failures": failures,
            "pass": not failures}


def run_gate(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    dump_dir = args.dump_dir.resolve()
    if not dump_dir.is_dir():
        raise GateError(f"dump directory does not exist: {dump_dir}")
    if args.expected_iterations < args.tail_transitions + 1:
        raise GateError("expected iterations must exceed tail transitions")
    leftovers = sorted(path.name for path in dump_dir.iterdir()
                       if ".tmp." in path.name)
    if leftovers:
        raise GateError(f"temporary comparison files remain: {leftovers[:8]}")
    manifests: dict[int, Path] = {}
    for path in dump_dir.glob(f"physics_{args.lane}_iter*.manifest.json"):
        match = MANIFEST_RE.fullmatch(path.name)
        if not match or match.group(1) != args.lane:
            raise GateError(f"malformed manifest name: {path.name}")
        manifests[int(match.group(2))] = path
    expected = set(range(args.expected_iterations))
    if set(manifests) != expected:
        raise GateError(
            f"iteration set mismatch: got={sorted(manifests)} expected={sorted(expected)}"
        )
    tail_start = args.expected_iterations - args.tail_transitions - 1
    needed = range(tail_start, args.expected_iterations)
    snapshots = {iteration: load_snapshot(manifests[iteration], args.lane,
                                           iteration, args.expected_bins)
                 for iteration in needed}

    invariant_keys = (
        "epoch_s", "n_shells", "n_bins", "atomic_model_sha256",
        "geometry_sha256", "grid_manifest_sha256",
    )
    baseline = snapshots[tail_start]["manifest"]
    generation_keys = (
        "radiation_generation", "population_generation", "te_generation",
        "opacity_generation", "emissivity_generation",
    )
    prior_generations: dict[str, int] | None = None
    for iteration in needed:
        manifest = snapshots[iteration]["manifest"]
        for key in invariant_keys:
            if manifest[key] != baseline[key]:
                raise GateError(f"manifest invariant {key} changed at iteration {iteration}")
        generations = {key: manifest[key] for key in generation_keys}
        if any(not isinstance(value, int) or value <= 0 for value in generations.values()):
            raise GateError(f"invalid generation at iteration {iteration}: {generations}")
        if prior_generations is not None:
            for key in generation_keys:
                if generations[key] <= prior_generations[key]:
                    raise GateError(f"non-increasing {key} at iteration {iteration}")
        prior_generations = generations

    transitions: list[dict[str, Any]] = []
    for old_iteration in range(tail_start, args.expected_iterations - 1):
        new_iteration = old_iteration + 1
        result = compare_pair(snapshots[old_iteration], snapshots[new_iteration])
        result["from_iteration"] = old_iteration
        result["to_iteration"] = new_iteration
        transitions.append(result)
    converged = all(item["pass"] for item in transitions)
    report = {
        "schema": "LUMINA_DET_CONVERGENCE_V1",
        "status": "CONVERGED" if converged else "NOT_CONVERGED",
        "lane": args.lane,
        "expected_iterations": args.expected_iterations,
        "tail_transitions": args.tail_transitions,
        "expected_bins": args.expected_bins,
        "dump_directory": str(dump_dir),
        "manifest_invariants": {key: baseline[key] for key in invariant_keys},
        "criteria": {
            "rule": "every metric passes on every final consecutive transition",
            "shell_max_limits": SHELL_MAX_LIMITS,
            "energy_l1_limits": ENERGY_L1_LIMITS,
            "spectral_l1_limits": SPECTRAL_L1_LIMITS,
            "final_energy_balance_max_relative": 1.0e-3,
        },
        "transitions": transitions,
    }
    return (0 if converged else 2), report


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--lane", default="DET")
    parser.add_argument("--expected-iterations", type=int, default=20)
    parser.add_argument("--tail-transitions", type=int, default=3)
    parser.add_argument("--expected-bins", type=int, default=1234)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_path = args.report or args.dump_dir / "det_convergence_report.json"
    try:
        code, report = run_gate(args)
    except GateError as exc:
        report = {
            "schema": "LUMINA_DET_CONVERGENCE_V1",
            "status": "INPUT_ERROR",
            "error": str(exc),
        }
        atomic_write_json(report_path, report)
        print(f"DET_CONVERGENCE_INPUT_ERROR {exc}", file=sys.stderr)
        return 3
    atomic_write_json(report_path, report)
    print(
        f"DET_CONVERGENCE_{report['status']} "
        f"iterations={args.expected_iterations} tail_transitions={args.tail_transitions} "
        f"report={report_path}"
    )
    if code:
        for transition in report["transitions"]:
            if transition["failures"]:
                print(
                    f"  iter {transition['from_iteration']}->{transition['to_iteration']}: "
                    + ", ".join(transition["failures"]),
                    file=sys.stderr,
                )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
