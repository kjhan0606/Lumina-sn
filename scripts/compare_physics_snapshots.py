#!/usr/bin/env python3
"""Conservative CMFGEN/ARTIS/Lumina physics-snapshot comparator.

The input contract is docs/CMFGEN_ARTIS_PHYSICS_COMPARISON_READINESS_2026-08-08.md.
Both shell and frequency fields are treated as piecewise-constant intensive
quantities.  Differences are integrated only over the common spherical volume
and common frequency domain; no extrapolation or centre interpolation exists.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SCHEMA = "LUMINA_PHYSICS_COMPARISON_V1"
FOUR_PI = 4.0 * math.pi

SHELL_COLUMNS = (
    "T_e_K", "n_e_cm3", "n_atom_cm3", "u_atom_erg",
    "q_ad_temperature_gradient", "q_ad_velocity_divergence",
    "q_ad_electron_fraction_gradient", "q_ad_internal_energy_gradient",
    "q_ad_signed_total", "q_ad_heating", "q_ad_cooling",
    "photo_heat", "line_abs_heat", "ff_abs_heat", "compton_heat",
    "gamma_heat", "nonthermal_heat", "recomb_cool", "line_emit_cool",
    "coll_line_cool", "ff_emit_cool", "compton_cool", "sum_heating",
    "sum_cooling", "residual",
)

SPECTRAL_COLUMNS = (
    "J_nu", "chi_es_cm1", "chi_bb_cm1", "chi_bf_cm1", "chi_ff_cm1",
    "chi_total_cm1", "eta_bb", "eta_bf", "eta_ff", "eta_true_total",
)

DERIVED_COLUMNS = (
    "four_pi_chi_bb_J", "four_pi_chi_bf_J", "four_pi_chi_ff_J",
    "four_pi_eta_bb", "four_pi_eta_bf", "four_pi_eta_ff",
)


class SnapshotError(ValueError):
    pass


@dataclass(frozen=True)
class Snapshot:
    manifest_path: Path
    manifest: dict[str, object]
    shell_edges: tuple[float, ...]
    frequency_edges: tuple[float, ...]
    shell_rows: tuple[dict[str, float], ...]
    spectral_rows: tuple[tuple[dict[str, float], ...], ...]


def _finite(value: str, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise SnapshotError(f"{label}: not a number") from exc
    if not math.isfinite(parsed):
        raise SnapshotError(f"{label}: nonfinite")
    return parsed


def _integer(value: object, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool):
        raise SnapshotError(f"{label}: boolean is not an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise SnapshotError(f"{label}: not an integer") from exc
    if str(parsed) != str(value) and not isinstance(value, int):
        raise SnapshotError(f"{label}: noncanonical integer")
    if positive and parsed <= 0:
        raise SnapshotError(f"{label}: must be positive")
    return parsed


def _close(a: float, b: float, tolerance: float = 2.0e-12) -> bool:
    return abs(a - b) <= tolerance * max(abs(a), abs(b), 1.0)


def _safe_relative_file(manifest_path: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise SnapshotError(f"{label}: unsafe or missing filename")
    path = manifest_path.parent / value
    if not path.is_file():
        raise SnapshotError(f"{label}: file not found: {path}")
    return path


def _validate_manifest(path: Path) -> dict[str, object]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SnapshotError(f"manifest unreadable: {path}") from exc
    if not isinstance(manifest, dict):
        raise SnapshotError("manifest root must be an object")
    required = {
        "schema": SCHEMA,
        "transaction_status": "COMMITTED",
        "frame": "SHELL_COMOVING",
        "frequency_coordinate": "HZ",
        "opacity_units": "CM^-1",
        "emissivity_units": "ERG_S^-1_CM^-3_HZ^-1_SR^-1",
        "volume_rate_units": "ERG_S^-1_CM^-3",
        "shell_weight": "SPHERICAL_VOLUME",
        "frequency_regrid": "INTEGRAL_PRESERVING_PIECEWISE_CONSTANT",
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise SnapshotError(f"manifest {key}: expected {expected!r}")
    if manifest.get("eta_is_per_sr") is not True:
        raise SnapshotError("manifest eta_is_per_sr must be true")
    if manifest.get("adiabatic_positive_is_cooling") is not True:
        raise SnapshotError("manifest adiabatic sign convention mismatch")
    factor = _finite(str(manifest.get("radiative_integral_factor")),
                     "radiative_integral_factor")
    if not _close(factor, FOUR_PI, 1.0e-14):
        raise SnapshotError("manifest radiative_integral_factor is not 4*pi")
    epoch = _finite(str(manifest.get("epoch_s")), "epoch_s")
    if epoch <= 0.0:
        raise SnapshotError("epoch_s must be positive")
    for key in ("n_shells", "n_bins", "radiation_generation",
                "population_generation", "te_generation",
                "opacity_generation", "emissivity_generation"):
        _integer(manifest.get(key), key, positive=True)
    if manifest["opacity_generation"] != manifest["emissivity_generation"]:
        raise SnapshotError("opacity/emissivity generation mismatch")
    for key in ("atomic_model_sha256", "geometry_sha256",
                "te_manifest_sha256", "grid_manifest_sha256"):
        value = manifest.get(key)
        if (not isinstance(value, str) or len(value) != 64 or
                any(ch not in "0123456789abcdefABCDEF" for ch in value)):
            raise SnapshotError(f"manifest {key}: not SHA-256 hex")
    return manifest


def _read_shells(path: Path, n_shells: int) -> tuple[tuple[float, ...],
                                                     tuple[dict[str, float], ...]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"shell_id", "r_inner_cm", "r_outer_cm", *SHELL_COLUMNS}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise SnapshotError("shell CSV schema incomplete")
        rows: list[dict[str, float]] = []
        edges: list[float] = []
        for expected_id, raw in enumerate(reader):
            shell_id = _integer(raw.get("shell_id"), "shell_id")
            if shell_id != expected_id:
                raise SnapshotError("shell_id must be contiguous from zero")
            converted = {column: _finite(raw[column], f"shell {shell_id} {column}")
                         for column in required if column != "shell_id"}
            ri, ro = converted["r_inner_cm"], converted["r_outer_cm"]
            if not (ri >= 0.0 and ro > ri):
                raise SnapshotError(f"shell {shell_id}: invalid radius interval")
            if edges and not _close(edges[-1], ri):
                raise SnapshotError(f"shell {shell_id}: noncontiguous radius")
            if not edges:
                edges.append(ri)
            edges.append(ro)
            if not _close(converted["q_ad_signed_total"],
                          converted["q_ad_cooling"]-converted["q_ad_heating"]):
                raise SnapshotError(f"shell {shell_id}: adiabatic sign closure")
            if not _close(converted["residual"],
                          converted["sum_heating"]-converted["sum_cooling"]):
                raise SnapshotError(f"shell {shell_id}: H-C closure")
            rows.append(converted)
    if len(rows) != n_shells:
        raise SnapshotError("shell CSV row count mismatch")
    return tuple(edges), tuple(rows)


def _read_spectral(path: Path, n_shells: int, n_bins: int) -> tuple[
        tuple[float, ...], tuple[tuple[dict[str, float], ...], ...]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"shell_id", "bin_id", "nu_lo_Hz", "nu_hi_Hz",
                    *SPECTRAL_COLUMNS}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise SnapshotError("spectral CSV schema incomplete")
        shell_rows: list[list[dict[str, float]]] = [[] for _ in range(n_shells)]
        canonical_edges: list[float] | None = None
        for raw in reader:
            shell_id = _integer(raw.get("shell_id"), "spectral shell_id")
            bin_id = _integer(raw.get("bin_id"), "bin_id")
            if shell_id < 0 or shell_id >= n_shells:
                raise SnapshotError("spectral shell_id out of range")
            if bin_id != len(shell_rows[shell_id]):
                raise SnapshotError("bin_id must be contiguous per shell")
            converted = {column: _finite(raw[column],
                         f"spectral {shell_id}/{bin_id} {column}")
                         for column in required
                         if column not in {"shell_id", "bin_id"}}
            lo, hi = converted["nu_lo_Hz"], converted["nu_hi_Hz"]
            if not (lo > 0.0 and hi > lo):
                raise SnapshotError("invalid frequency interval")
            if not _close(converted["chi_total_cm1"],
                          converted["chi_es_cm1"]+converted["chi_bb_cm1"]+
                          converted["chi_bf_cm1"]+converted["chi_ff_cm1"]):
                raise SnapshotError("opacity component closure")
            if not _close(converted["eta_true_total"],
                          converted["eta_bb"]+converted["eta_bf"]+
                          converted["eta_ff"]):
                raise SnapshotError("emissivity component closure")
            if converted["J_nu"] < 0.0 or any(converted[name] < 0.0 for name in
                    ("eta_bb", "eta_bf", "eta_ff", "eta_true_total")):
                raise SnapshotError("negative J or emissivity")
            shell_rows[shell_id].append(converted)
        for shell_id, rows in enumerate(shell_rows):
            if len(rows) != n_bins:
                raise SnapshotError("spectral row count mismatch")
            edges = [rows[0]["nu_lo_Hz"], *(row["nu_hi_Hz"] for row in rows)]
            for b in range(1, n_bins):
                if not _close(rows[b-1]["nu_hi_Hz"], rows[b]["nu_lo_Hz"]):
                    raise SnapshotError("noncontiguous frequency grid")
            if canonical_edges is None:
                canonical_edges = edges
            elif any(not _close(a,b) for a,b in zip(canonical_edges,edges)):
                raise SnapshotError(f"shell {shell_id}: frequency grid differs")
    assert canonical_edges is not None
    return tuple(canonical_edges), tuple(tuple(rows) for rows in shell_rows)


def load_snapshot(manifest_path: Path) -> Snapshot:
    manifest_path = manifest_path.resolve()
    manifest = _validate_manifest(manifest_path)
    n_shells = int(manifest["n_shells"])
    n_bins = int(manifest["n_bins"])
    shell_path = _safe_relative_file(manifest_path, manifest.get("shell_file"),
                                     "shell_file")
    spectral_path = _safe_relative_file(manifest_path,
                                        manifest.get("spectral_file"),
                                        "spectral_file")
    shell_edges, shell_rows = _read_shells(shell_path,n_shells)
    frequency_edges, spectral_rows = _read_spectral(
        spectral_path,n_shells,n_bins)
    return Snapshot(manifest_path,manifest,shell_edges,frequency_edges,
                    shell_rows,spectral_rows)


def _overlaps(left: tuple[float, ...], right: tuple[float, ...],
              weight_kind: str) -> tuple[list[tuple[int,int,float]],float,float,float]:
    i=j=0; result: list[tuple[int,int,float]]=[]; common=0.0
    def measure(lo: float, hi: float) -> float:
        if weight_kind == "volume":
            return FOUR_PI/3.0*(hi**3-lo**3)
        return hi-lo
    while i+1 < len(left) and j+1 < len(right):
        lo=max(left[i],right[j]); hi=min(left[i+1],right[j+1])
        if hi > lo:
            weight=measure(lo,hi); result.append((i,j,weight)); common+=weight
        if left[i+1] <= right[j+1]: i+=1
        else: j+=1
    left_total=measure(left[0],left[-1]); right_total=measure(right[0],right[-1])
    return result,common,left_total,right_total


def _value(row: dict[str,float], column: str) -> float:
    if column == "four_pi_chi_bb_J":
        return FOUR_PI*row["chi_bb_cm1"]*row["J_nu"]
    if column == "four_pi_chi_bf_J":
        return FOUR_PI*row["chi_bf_cm1"]*row["J_nu"]
    if column == "four_pi_chi_ff_J":
        return FOUR_PI*row["chi_ff_cm1"]*row["J_nu"]
    if column == "four_pi_eta_bb": return FOUR_PI*row["eta_bb"]
    if column == "four_pi_eta_bf": return FOUR_PI*row["eta_bf"]
    if column == "four_pi_eta_ff": return FOUR_PI*row["eta_ff"]
    return row[column]


def _metric(samples: Iterable[tuple[float,float,float]]) -> dict[str,float]:
    difference=norm=0.0; max_abs=0.0
    for a,b,weight in samples:
        delta=abs(a-b); difference+=delta*weight
        norm+=0.5*(abs(a)+abs(b))*weight; max_abs=max(max_abs,delta)
    relative = difference/norm if norm > 0.0 else (0.0 if difference == 0.0 else math.inf)
    return {"relative_l1":relative,"absolute_l1":difference,"max_abs":max_abs}


def compare_snapshots(left: Snapshot, right: Snapshot, *, rtol: float,
                      atol: float, allow_atomic_model_mismatch: bool=False) -> dict[str,object]:
    if not _close(float(left.manifest["epoch_s"]),float(right.manifest["epoch_s"]),
                  1.0e-12):
        raise SnapshotError("epoch mismatch")
    if (not allow_atomic_model_mismatch and
        left.manifest["atomic_model_sha256"] !=
            right.manifest["atomic_model_sha256"]):
        raise SnapshotError("atomic model mismatch")
    for key in ("frame","frequency_coordinate","opacity_units",
                "emissivity_units","volume_rate_units","eta_is_per_sr",
                "radiative_integral_factor","adiabatic_positive_is_cooling"):
        if left.manifest[key] != right.manifest[key]:
            raise SnapshotError(f"cross-snapshot convention mismatch: {key}")
    shell_overlap,common_volume,left_volume,right_volume = _overlaps(
        left.shell_edges,right.shell_edges,"volume")
    frequency_overlap,common_frequency,left_frequency,right_frequency = _overlaps(
        left.frequency_edges,right.frequency_edges,"frequency")
    if not shell_overlap or not frequency_overlap:
        raise SnapshotError("snapshots have no common shell/frequency domain")
    metrics: dict[str,dict[str,float]]={}
    for column in SHELL_COLUMNS:
        metrics[column]=_metric(
            (left.shell_rows[i][column],right.shell_rows[j][column],weight)
            for i,j,weight in shell_overlap)
    for column in (*SPECTRAL_COLUMNS,*DERIVED_COLUMNS):
        metrics[column]=_metric(
            (_value(left.spectral_rows[si][fi],column),
             _value(right.spectral_rows[sj][fj],column),volume_weight*frequency_weight)
            for si,sj,volume_weight in shell_overlap
            for fi,fj,frequency_weight in frequency_overlap)
    failed=[name for name,metric in metrics.items()
            if metric["relative_l1"] > rtol and metric["max_abs"] > atol]
    return {
        "schema":"LUMINA_PHYSICS_COMPARISON_RESULT_V1",
        "left_manifest":str(left.manifest_path),
        "right_manifest":str(right.manifest_path),
        "common_shell_volume_cm3":common_volume,
        "left_shell_coverage_fraction":common_volume/left_volume,
        "right_shell_coverage_fraction":common_volume/right_volume,
        "common_frequency_width_Hz":common_frequency,
        "left_frequency_coverage_fraction":common_frequency/left_frequency,
        "right_frequency_coverage_fraction":common_frequency/right_frequency,
        "rtol":rtol,"atol":atol,"metrics":metrics,"failed_columns":failed,
        "verdict":"PASS" if not failed else "DIFFERENT",
    }


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("left_manifest",type=Path)
    parser.add_argument("right_manifest",type=Path)
    parser.add_argument("--rtol",type=float,default=1.0e-8)
    parser.add_argument("--atol",type=float,default=0.0)
    parser.add_argument("--allow-atomic-model-mismatch",action="store_true")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.rtol < 0.0 or args.atol < 0.0:
        parser.error("tolerances must be nonnegative")
    try:
        result=compare_snapshots(load_snapshot(args.left_manifest),
                                 load_snapshot(args.right_manifest),
                                 rtol=args.rtol,atol=args.atol,
                                 allow_atomic_model_mismatch=
                                     args.allow_atomic_model_mismatch)
    except SnapshotError as exc:
        print(f"[PHYSICS-COMPARE][BLOCKED] reason={exc}",file=sys.stderr)
        return 2
    rendered=json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n"
    if args.output:
        args.output.write_text(rendered,encoding="utf-8")
    else:
        print(rendered,end="")
    print(f"[PHYSICS-COMPARE] verdict={result['verdict']} "
          f"failed={len(result['failed_columns'])}",file=sys.stderr)
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
