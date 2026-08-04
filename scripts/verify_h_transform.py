#!/usr/bin/env python3
"""Read-only H-TRANSFORM audit for the toy06 CMFGEN hydro generator.

The audit never imports or executes ``mk_sn_hydro.py`` and never writes below
its input trees.  It independently reconstructs three composition lanes:

  raw              source mass fractions, unchanged
  floor_only       the generator's 1e-10 Fe/Co/Ni floor
  floor_renorm     floor_only divided by its six-element sum

The 700 selected source zones are the preregistered decision grid.  The same
lanes are also linearly interpolated to the current 90-depth CMFGEN grid so the
actual consumer sampling is visible rather than inferred from the 700 zones.
All evidence is written to a caller-selected, previously nonexistent directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import re
import socket
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path(
    "/gpfs/kjhan/cmfgen_runs/toy06_19.48d/snia_toy06_19.48d.dat"
)
DEFAULT_GRID = Path("/gpfs/kjhan/cmfgen_runs/toy06_19.48d/NEW_SN_R_GRID")
DEFAULT_GENERATOR = Path("/gpfs/kjhan/cmfgen_runs/toy06_19.48d/mk_sn_hydro.py")
DEFAULT_LINE_LIST = (
    REPO_ROOT / "data/tardis_reference_toy06_19p48d_sivcaiv/line_list.csv"
)

FLOOR = 1.0e-10
V_LO_KMS = 1000.0
V_HI_KMS = 36000.0
MSUN_G = 1.989e33

ELEMENTS = ("SIL", "SUL", "CAL", "IRON", "COB", "NICK")
ATOMIC_NUMBER = {"SIL": 14, "SUL": 16, "CAL": 20, "IRON": 26, "COB": 27, "NICK": 28}
ATOMIC_MASS = {"SIL": 28.085, "SUL": 32.06, "CAL": 40.078,
               "IRON": 56.0, "COB": 56.0, "NICK": 56.0}
SOURCE_COLUMN = {"SIL": 18, "SUL": 17, "CAL": 16,
                 "IRON": 15, "COB": 14, "NICK": 13}
IGE = ("IRON", "COB", "NICK")

# These half-open bands reproduce the preregistered ~1.68e-9 maximum with the
# production toy06 line list.  Narrower post-hoc sub-bands are intentionally not
# introduced after seeing the result.
BANDS = (
    ("EUV_450_918", 450.0, 918.0),
    ("FUV_918_1290", 918.0, 1290.0),
    ("NUV_1290_2000", 1290.0, 2000.0),
    ("UVOPT_2000_10000", 2000.0, 10000.0),
    ("IR_10000_25000", 10000.0, 25000.0),
)

PREREG_LIMITS = {
    "raw_to_final_max_abs_delta_x": 6.0e-6,
    "sum_x_over_a_max_relative_change": 1.3e-5,
    "floor_only_proxy_max_relative_change": 2.0e-9,
    "floor_renorm_proxy_max_relative_change": 1.3e-5,
}
PRELIMINARY = {
    "raw_sum_min": 0.999992,
    "raw_sum_max": 1.000012,
    "elemental_floor_injection_sum": 1.416e-7,
    "raw_to_final_max_abs_delta_x": 5.15e-6,
    "sum_x_over_a_max_relative_change": 1.20e-5,
    "floor_only_proxy_max_relative_change": 1.68e-9,
    "floor_renorm_proxy_max_relative_change": 1.20e-5,
}


class AuditError(RuntimeError):
    """Fail-closed input or contract error."""


@dataclass(frozen=True)
class Lane:
    name: str
    composition: np.ndarray
    x56ni: np.ndarray
    normalization_factor: np.ndarray


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_finite_nonnegative(name: str, values: np.ndarray) -> None:
    if not np.all(np.isfinite(values)):
        raise AuditError(f"{name} contains NaN or Inf")
    if np.any(values < 0.0):
        raise AuditError(f"{name} contains negative values")


def require_unit_sum(values: np.ndarray, *, atol: float = 1.0e-12) -> None:
    sums = np.sum(values, axis=0)
    bad = np.flatnonzero(np.abs(sums - 1.0) > atol)
    if bad.size:
        i = int(bad[0])
        raise AuditError(
            f"off-sum composition requires explicit projection: shell={i} "
            f"sum={sums[i]:.17g} atol={atol:.1e}"
        )


def zero_resurrections(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    return np.argwhere((before == 0.0) & (after != 0.0))


def build_lanes(raw: np.ndarray, raw_x56ni: np.ndarray) -> tuple[Lane, Lane, Lane]:
    require_finite_nonnegative("raw composition", raw)
    require_finite_nonnegative("raw X56Ni", raw_x56ni)
    if raw.shape[0] != len(ELEMENTS) or raw.shape[1] != raw_x56ni.size:
        raise AuditError("composition/X56Ni shape mismatch")

    floor_only = raw.copy()
    for element in IGE:
        floor_only[ELEMENTS.index(element)] = np.maximum(
            floor_only[ELEMENTS.index(element)], FLOOR
        )
    floor_x56ni = np.maximum(raw_x56ni, FLOOR)
    floor_sum = np.sum(floor_only, axis=0)
    if np.any(floor_sum <= 0.0):
        raise AuditError("cannot explicitly project a nonpositive six-element sum")
    factor = 1.0 / floor_sum
    final = floor_only * factor
    final_x56ni = floor_x56ni * factor
    ones = np.ones(raw.shape[1], dtype=np.float64)
    return (
        Lane("raw", raw.copy(), raw_x56ni.copy(), ones),
        Lane("floor_only", floor_only, floor_x56ni, ones),
        Lane("floor_renorm", final, final_x56ni, factor),
    )


def read_source(path: Path) -> dict[str, np.ndarray]:
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] != 21:
        raise AuditError(f"expected 21-column StaNdaRT input, got {data.shape}")
    selected = (data[:, 1] >= V_LO_KMS) & (data[:, 1] <= V_HI_KMS)
    # Match mk_sn_hydro.py exactly: the selected source is written outermost
    # first, even though the StaNdaRT table itself is velocity-ascending.
    data = data[selected][::-1]
    if data.shape[0] != 700:
        raise AuditError(f"expected 700 selected zones, got {data.shape[0]}")
    raw = np.vstack([data[:, SOURCE_COLUMN[name]] for name in ELEMENTS])
    return {
        "velocity": data[:, 1].astype(np.float64),
        "dmass_msun": data[:, 2].astype(np.float64),
        "radius_cm": data[:, 9].astype(np.float64),
        "density_g_cm3": data[:, 10].astype(np.float64),
        "composition": raw.astype(np.float64),
        "x56ni": data[:, 12].astype(np.float64),
    }


def read_cmfgen_grid(path: Path) -> dict[str, np.ndarray]:
    row_pattern = re.compile(
        r"^\s*(\d+)\s+([0-9.Ee+-]+)\s+([0-9.Ee+-]+)\s+"
        r"([0-9.Ee+-]+)\s+([0-9.Ee+-]+)"
    )
    rows = []
    for line in path.read_text().splitlines():
        match = row_pattern.match(line)
        if match:
            rows.append(tuple(float(match.group(i)) for i in range(2, 6)))
    if len(rows) != 90:
        raise AuditError(f"expected 90 CMFGEN depths, got {len(rows)} in {path}")
    array = np.asarray(rows, dtype=np.float64)
    # NEW_SN_R_GRID radius is in 1e10 cm; columns are R, velocity, sigma, tau.
    return {"radius_cm": array[:, 0] * 1.0e10, "velocity": array[:, 1]}


def interpolate_lane(lane: Lane, source_velocity: np.ndarray,
                     target_velocity: np.ndarray) -> Lane:
    order = np.argsort(source_velocity)
    velocity_ascending = source_velocity[order]
    if not np.all(np.diff(velocity_ascending) > 0.0):
        raise AuditError("source velocity must be unique")
    composition = np.vstack([
        np.interp(target_velocity, velocity_ascending, row[order])
        for row in lane.composition
    ])
    x56ni = np.interp(target_velocity, velocity_ascending, lane.x56ni[order])
    factor = np.interp(
        target_velocity, velocity_ascending, lane.normalization_factor[order]
    )
    return Lane(lane.name, composition, x56ni, factor)


def line_strengths(path: Path) -> tuple[np.ndarray, dict[str, int]]:
    strengths = np.zeros((len(ELEMENTS), len(BANDS)), dtype=np.float64)
    z_to_index = {ATOMIC_NUMBER[name]: i for i, name in enumerate(ELEMENTS)}
    counts = {"rows": 0, "nonpositive_wavelength_excluded": 0,
              "nonfinite_excluded": 0}
    for chunk in pd.read_csv(
        path,
        usecols=["atomic_number", "wavelength", "f_lu"],
        chunksize=500_000,
    ):
        counts["rows"] += int(len(chunk))
        z = chunk["atomic_number"].to_numpy()
        wavelength = chunk["wavelength"].to_numpy(dtype=np.float64)
        f_lu = chunk["f_lu"].to_numpy(dtype=np.float64)
        finite = np.isfinite(wavelength) & np.isfinite(f_lu)
        counts["nonfinite_excluded"] += int(np.count_nonzero(~finite))
        counts["nonpositive_wavelength_excluded"] += int(
            np.count_nonzero(finite & (wavelength <= 0.0))
        )
        for zi, element_i in z_to_index.items():
            species = finite & (wavelength > 0.0) & (z == zi)
            for band_i, (_, lo, hi) in enumerate(BANDS):
                mask = species & (wavelength >= lo) & (wavelength < hi)
                strengths[element_i, band_i] += math.fsum(
                    np.abs(f_lu[mask]) * wavelength[mask]
                )
    if counts["rows"] == 0:
        raise AuditError(f"empty line list: {path}")
    return strengths, counts


def proxy(lane: Lane, strengths: np.ndarray) -> np.ndarray:
    masses = np.asarray([ATOMIC_MASS[name] for name in ELEMENTS])[:, None, None]
    # result shape: band, shell
    return np.sum(
        lane.composition[:, None, :] * strengths[:, :, None] / masses,
        axis=0,
    )


def relative_change(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    result = np.full(reference.shape, np.nan, dtype=np.float64)
    nonzero = reference != 0.0
    result[nonzero] = np.abs(candidate[nonzero] / reference[nonzero] - 1.0)
    both_zero = (reference == 0.0) & (candidate == 0.0)
    result[both_zero] = 0.0
    result[(reference == 0.0) & (candidate != 0.0)] = np.inf
    return result


def max_finite(values: np.ndarray) -> float:
    if np.any(np.isinf(values)):
        return float("inf")
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise AuditError("metric has no finite values")
    return float(np.max(finite))


def selected_mass(lane: Lane, dmass_msun: np.ndarray) -> np.ndarray:
    return np.sum(lane.composition * dmass_msun[None, :], axis=1)


def depth_integrated_mass(lane: Lane, radius_cm: np.ndarray,
                          density_g_cm3: np.ndarray) -> np.ndarray:
    order = np.argsort(radius_cm)
    r = radius_cm[order]
    rho = density_g_cm3[order]
    answer = []
    for row in lane.composition:
        integrand = 4.0 * np.pi * r * r * rho * row[order]
        answer.append(float(np.trapezoid(integrand, r) / MSUN_G))
    return np.asarray(answer)


def write_shell_csv(path: Path, grid_name: str, velocity: np.ndarray,
                    lanes: tuple[Lane, ...]) -> None:
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if handle.tell() == 0:
            writer.writerow([
                "grid", "shell", "velocity_km_s", "lane", "element",
                "X", "delta_X_from_raw", "sum6", "normalization_factor",
                "sum_X_over_A", "relative_change_sum_X_over_A_from_raw",
                "X56Ni", "X56Ni_minus_element_Ni",
                "isotope_chain_minus_element_IGE",
            ])
        raw = lanes[0]
        ni_index = ELEMENTS.index("NICK")
        fe_index = ELEMENTS.index("IRON")
        co_index = ELEMENTS.index("COB")
        atomic_mass = np.asarray([ATOMIC_MASS[name] for name in ELEMENTS])[:, None]
        raw_inv_a = np.sum(raw.composition / atomic_mass, axis=0)
        for lane in lanes:
            sums = np.sum(lane.composition, axis=0)
            inv_a = np.sum(lane.composition / atomic_mass, axis=0)
            inv_a_rel = relative_change(inv_a, raw_inv_a)
            for shell, vel in enumerate(velocity):
                for element_i, element in enumerate(ELEMENTS):
                    writer.writerow([
                        grid_name, shell, f"{vel:.17g}", lane.name, element,
                        f"{lane.composition[element_i, shell]:.17g}",
                        f"{lane.composition[element_i, shell] - raw.composition[element_i, shell]:.17g}",
                        f"{sums[shell]:.17g}",
                        f"{lane.normalization_factor[shell]:.17g}",
                        f"{inv_a[shell]:.17g}",
                        f"{inv_a_rel[shell]:.17g}",
                        f"{lane.x56ni[shell]:.17g}",
                        f"{lane.x56ni[shell] - lane.composition[ni_index, shell]:.17g}",
                        f"{(lane.x56ni[shell] + lane.composition[co_index, shell] + lane.composition[fe_index, shell]) - (lane.composition[ni_index, shell] + lane.composition[co_index, shell] + lane.composition[fe_index, shell]):.17g}",
                    ])


def write_proxy_csv(path: Path, grid_name: str, velocity: np.ndarray,
                    lanes: tuple[Lane, ...], strengths: np.ndarray) -> None:
    values = [proxy(lane, strengths) for lane in lanes]
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if handle.tell() == 0:
            writer.writerow([
                "grid", "shell", "velocity_km_s", "lane", "band",
                "lambda_lo_A", "lambda_hi_A", "proxy", "relative_change_from_raw",
            ])
        for lane, lane_proxy in zip(lanes, values):
            rel = relative_change(lane_proxy, values[0])
            for band_i, (band, lo, hi) in enumerate(BANDS):
                for shell, vel in enumerate(velocity):
                    writer.writerow([
                        grid_name, shell, f"{vel:.17g}", lane.name, band,
                        f"{lo:.17g}", f"{hi:.17g}",
                        f"{lane_proxy[band_i, shell]:.17g}",
                        f"{rel[band_i, shell]:.17g}",
                    ])


def lane_metrics(lanes: tuple[Lane, ...], strengths: np.ndarray) -> dict[str, float]:
    raw, floor_only, final = lanes
    atomic_mass = np.asarray([ATOMIC_MASS[name] for name in ELEMENTS])[:, None]
    inv_a = [np.sum(lane.composition / atomic_mass, axis=0) for lane in lanes]
    proxies = [proxy(lane, strengths) for lane in lanes]
    return {
        "raw_sum_min": float(np.min(np.sum(raw.composition, axis=0))),
        "raw_sum_max": float(np.max(np.sum(raw.composition, axis=0))),
        "floor_sum_min": float(np.min(np.sum(floor_only.composition, axis=0))),
        "floor_sum_max": float(np.max(np.sum(floor_only.composition, axis=0))),
        "normalization_factor_min": float(np.min(final.normalization_factor)),
        "normalization_factor_max": float(np.max(final.normalization_factor)),
        "raw_to_final_max_abs_delta_x": float(
            np.max(np.abs(final.composition - raw.composition))
        ),
        "sum_x_over_a_floor_max_relative_change": max_finite(
            relative_change(inv_a[1], inv_a[0])
        ),
        "sum_x_over_a_max_relative_change": max_finite(
            relative_change(inv_a[2], inv_a[0])
        ),
        "floor_only_proxy_max_relative_change": max_finite(
            relative_change(proxies[1], proxies[0])
        ),
        "floor_renorm_proxy_max_relative_change": max_finite(
            relative_change(proxies[2], proxies[0])
        ),
        "raw_x56ni_minus_ni_max_abs": float(
            np.max(np.abs(raw.x56ni - raw.composition[ELEMENTS.index("NICK")]))
        ),
        "floor_x56ni_minus_ni_max_abs": float(
            np.max(np.abs(floor_only.x56ni - floor_only.composition[ELEMENTS.index("NICK")]))
        ),
        "final_x56ni_minus_ni_max_abs": float(
            np.max(np.abs(final.x56ni - final.composition[ELEMENTS.index("NICK")]))
        ),
    }


def prereg_comparison(metrics: dict[str, float]) -> dict[str, dict[str, object]]:
    return {
        name: {
            "measured": metrics[name],
            "limit": limit,
            "pass": bool(metrics[name] <= limit),
            "margin": limit - metrics[name],
        }
        for name, limit in PREREG_LIMITS.items()
    }


def run_self_test() -> int:
    # A physically valid Si-only row has exact-zero IGE.  The raw lane preserves
    # it, while the current floor lanes resurrect it; the verifier must see that.
    raw = np.asarray([[1.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
    lanes = build_lanes(raw, np.asarray([0.0]))
    if zero_resurrections(raw, lanes[0].composition).size:
        raise AuditError("self-test failure: raw lane changed an exact zero")
    resurrected = zero_resurrections(raw, lanes[1].composition)
    if resurrected.shape[0] != 3:
        raise AuditError(
            f"self-test failure: expected three IGE resurrections, got {resurrected.shape[0]}"
        )
    print("PASS exact-zero control: raw kept exact zeros; floor violation detected (3 IGE entries)")

    off_sum = np.asarray([[0.8], [0.1], [0.0], [0.0], [0.0], [0.0]])
    try:
        require_unit_sum(off_sum)
    except AuditError as exc:
        projected = off_sum / np.sum(off_sum, axis=0)
        if not np.array_equal(off_sum, np.asarray([[0.8], [0.1], [0.0], [0.0], [0.0], [0.0]])):
            raise AuditError("self-test failure: off-sum source was modified")
        print(f"PASS off-sum control: explicit failure: {exc}")
        print(
            "PASS off-sum control: separate projection emitted "
            f"(source_sum={np.sum(off_sum):.17g}, projected_sum={np.sum(projected):.17g})"
        )
    else:
        raise AuditError("self-test failure: off-sum row was silently accepted")
    return 0


def run(args: argparse.Namespace) -> int:
    inputs = {
        "source": args.source.resolve(strict=True),
        "cmfgen_grid": args.cmfgen_grid.resolve(strict=True),
        "generator_read_only": args.generator.resolve(strict=True),
        "line_list": args.line_list.resolve(strict=True),
    }
    requested_output = args.output_dir.expanduser()
    output_dir = requested_output.resolve(strict=False)
    if (requested_output.exists() or requested_output.is_symlink()
            or output_dir.exists()):
        raise AuditError(f"output directory must not already exist: {requested_output}")
    input_parents = {path.parent for path in inputs.values()}
    if any(parent == output_dir or parent in output_dir.parents for parent in input_parents):
        raise AuditError(
            f"output directory must not be inside an input tree: {output_dir}"
        )

    input_hashes = {name: sha256(path) for name, path in inputs.items()}

    source = read_source(inputs["source"])
    grid = read_cmfgen_grid(inputs["cmfgen_grid"])
    source_lanes = build_lanes(source["composition"], source["x56ni"])
    grid_lanes = tuple(
        interpolate_lane(lane, source["velocity"], grid["velocity"])
        for lane in source_lanes
    )
    strengths, line_counts = line_strengths(inputs["line_list"])
    post_hashes = {name: sha256(path) for name, path in inputs.items()}
    changed_inputs = [name for name in inputs if input_hashes[name] != post_hashes[name]]
    if changed_inputs:
        raise AuditError(f"inputs changed during read-only audit: {changed_inputs}")

    # No input mutation has occurred.  Create only the new evidence directory.
    output_dir.mkdir(parents=True, exist_ok=False)
    shell_csv = output_dir / "shell_element_metrics.csv"
    proxy_csv = output_dir / "line_proxy_metrics.csv"
    write_shell_csv(shell_csv, "selected_700", source["velocity"], source_lanes)
    write_shell_csv(shell_csv, "cmfgen_90", grid["velocity"], grid_lanes)
    write_proxy_csv(proxy_csv, "selected_700", source["velocity"], source_lanes, strengths)
    write_proxy_csv(proxy_csv, "cmfgen_90", grid["velocity"], grid_lanes, strengths)

    metrics_700 = lane_metrics(source_lanes, strengths)
    metrics_700["elemental_floor_injection_sum"] = float(
        np.sum(source_lanes[1].composition - source_lanes[0].composition)
    )
    metrics_700["x56ni_floor_injection_sum"] = float(
        np.sum(source_lanes[1].x56ni - source_lanes[0].x56ni)
    )
    metrics_90 = lane_metrics(grid_lanes, strengths)

    mass_rows = []
    density_90 = np.interp(
        grid["velocity"], source["velocity"][::-1], source["density_g_cm3"][::-1]
    )
    selected_masses = [selected_mass(lane, source["dmass_msun"]) for lane in source_lanes]
    depth_masses = [
        depth_integrated_mass(lane, grid["radius_cm"], density_90)
        for lane in grid_lanes
    ]
    for grid_name, lanes, masses in (
        ("selected_700_dmass", source_lanes, selected_masses),
        ("cmfgen_90_trapezoid", grid_lanes, depth_masses),
    ):
        for lane, values in zip(lanes, masses):
            raw_values = masses[0]
            for element, value, raw_value in zip(ELEMENTS, values, raw_values):
                mass_rows.append({
                    "grid": grid_name,
                    "lane": lane.name,
                    "element": element,
                    "integrated_mass_msun": float(value),
                    "delta_mass_msun_from_raw": float(value - raw_value),
                })
    pd.DataFrame(mass_rows).to_csv(output_dir / "integrated_element_masses.csv", index=False)

    line_weight_rows = []
    for element_i, element in enumerate(ELEMENTS):
        for band_i, (band, lo, hi) in enumerate(BANDS):
            line_weight_rows.append({
                "element": element,
                "atomic_number": ATOMIC_NUMBER[element],
                "band": band,
                "lambda_lo_A": lo,
                "lambda_hi_A": hi,
                "sum_abs_f_lu_lambda_A": float(strengths[element_i, band_i]),
            })
    pd.DataFrame(line_weight_rows).to_csv(output_dir / "line_strength_by_element_band.csv", index=False)

    comparison = prereg_comparison(metrics_700)
    preliminary_comparison = {
        name: {
            "measured": metrics_700[name],
            "preliminary": expected,
            "difference": metrics_700[name] - expected,
        }
        for name, expected in PRELIMINARY.items()
    }
    zero_counts = {}
    for lane in source_lanes[1:]:
        locations = zero_resurrections(source_lanes[0].composition, lane.composition)
        zero_counts[lane.name] = int(locations.shape[0])

    summary = {
        "contract": "H-TRANSFORM",
        "status": "PASS" if all(item["pass"] for item in comparison.values()) else "FAIL",
        "disposition": {
            "floor": "REMOVE: small effect does not authorize a forbidden floor",
            "exact_zero": "must remain exactly zero",
            "renormalization": (
                "do not silently mutate provenance; retain raw values and emit any "
                "explicit conservation projection as a separate lane"
            ),
            "generator_modified": False,
        },
        "inputs": {
            name: {"path": str(path), "sha256": input_hashes[name]}
            for name, path in inputs.items()
        },
        "environment": {
            "hostname": socket.gethostname(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "argv": sys.argv,
        },
        "grids": {
            "selected_source_zones": int(source["velocity"].size),
            "cmfgen_depths": int(grid["velocity"].size),
            "selected_velocity_min_km_s": float(np.min(source["velocity"])),
            "selected_velocity_max_km_s": float(np.max(source["velocity"])),
        },
        "bands": [
            {"name": name, "lambda_lo_A": lo, "lambda_hi_A": hi,
             "interval": "half-open"}
            for name, lo, hi in BANDS
        ],
        "line_list_counts": line_counts,
        "selected_700_metrics": metrics_700,
        "cmfgen_90_metrics": metrics_90,
        "preregistered_limit_comparison_selected_700": comparison,
        "preliminary_value_comparison_selected_700": preliminary_comparison,
        "zero_resurrections_selected_700": zero_counts,
        "x56ni_consistency_definition": (
            "max absolute X56Ni-elemental_Ni; toy06 stable Ni is zero, so equality "
            "must survive floor and the shared explicit normalization factor"
        ),
        "artifacts": [
            "summary.json", "shell_element_metrics.csv", "integrated_element_masses.csv",
            "line_strength_by_element_band.csv", "line_proxy_metrics.csv",
            "artifact_sha256.json",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    artifact_hashes = {
        path.name: sha256(path)
        for path in sorted(output_dir.iterdir())
        if path.name != "artifact_sha256.json"
    }
    (output_dir / "artifact_sha256.json").write_text(
        json.dumps(artifact_hashes, indent=2, sort_keys=True) + "\n"
    )

    print(f"H-TRANSFORM {summary['status']}")
    print(f"evidence_dir={output_dir}")
    print(
        "raw_sum_range="
        f"[{metrics_700['raw_sum_min']:.12g}, {metrics_700['raw_sum_max']:.12g}]"
    )
    print(f"elemental_floor_injection_sum={metrics_700['elemental_floor_injection_sum']:.12g}")
    for name, result in comparison.items():
        verdict = "PASS" if result["pass"] else "FAIL"
        print(
            f"{verdict} {name}: measured={result['measured']:.12g} "
            f"limit={result['limit']:.12g}"
        )
    for name, result in preliminary_comparison.items():
        print(
            f"PRELIM {name}: measured={result['measured']:.12g} "
            f"preregistered_preview={result['preliminary']:.12g} "
            f"difference={result['difference']:.12g}"
        )
    print(
        "DISPOSITION floor=REMOVE exact-zero=KEEP_ZERO "
        "renormalization=SEPARATE_EXPLICIT_PROJECTION"
    )
    return 0 if summary["status"] == "PASS" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--cmfgen-grid", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--generator", type=Path, default=DEFAULT_GENERATOR)
    parser.add_argument("--line-list", type=Path, default=DEFAULT_LINE_LIST)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test and args.output_dir is None:
        parser.error("--output-dir is required unless --self-test is used")
    return args


if __name__ == "__main__":
    try:
        cli_args = parse_args()
        raise SystemExit(run_self_test() if cli_args.self_test else run(cli_args))
    except (AuditError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"FAIL H-TRANSFORM: {exc}", file=sys.stderr)
        raise SystemExit(2)
