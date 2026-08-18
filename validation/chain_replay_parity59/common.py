#!/usr/bin/env python3
"""Shared readers for the parity59 chain replay.

All paths are supplied by callers.  The routines intentionally preserve the
definitions used by the 2026-07-19 audits; they do not invent missing bins or
physical values.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

C_CM = 2.99792458e10
C_A = 2.99792458e18
FOURPI_OVER_C = 4.0 * np.pi / C_CM


def read_info(path: Path) -> dict[str, int | bool]:
    lines = path.read_text().splitlines()
    values = lines[2].split()
    return {
        "ND": int(values[0]),
        "RECL": int(values[1]),
        "WORD": int(values[2]),
        "little": values[5] == "T",
    }


def read_eddfactor(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    info = read_info(Path(str(path) + "_INFO"))
    nd = int(info["ND"])
    nwr = int(info["RECL"]) // int(info["WORD"])
    dtype = "<f8" if info["little"] else ">f8"
    raw = np.fromfile(path, dtype=dtype)
    complete = (raw.size // nwr) * nwr
    raw = raw[:complete].reshape(-1, nwr)
    data = raw[14:]
    good = np.isfinite(data[:, :nd]).all(axis=1) & (data[:, nd] > 0)
    return data[good, :nd], data[good, nd], nd


def parse_rvtj_block(path: Path, label: str, nd: int) -> np.ndarray:
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.strip() != label:
            continue
        values: list[float] = []
        for candidate in lines[i + 1 :]:
            if len(values) >= nd:
                break
            try:
                values.extend(float(token) for token in candidate.split())
            except ValueError:
                break
        if len(values) < nd:
            raise ValueError(f"{path}: block {label!r} has {len(values)} values, expected {nd}")
        return np.asarray(values[:nd])
    raise KeyError(f"{path}: no RVTJ block {label!r}")


def load_field(path: Path) -> dict[int, dict[str, np.ndarray]]:
    by_shell: dict[int, list[tuple[int, float, float, float]]] = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            shell = int(row["shell"])
            by_shell.setdefault(shell, []).append(
                (int(row["bin"]), float(row["wavelength_A"]), float(row["cs_J"]), float(row["mc_J"]))
            )
    result: dict[int, dict[str, np.ndarray]] = {}
    for shell, rows in by_shell.items():
        rows.sort(key=lambda item: item[0])
        values = np.asarray(rows)
        bins = values[:, 0].astype(int)
        if not np.array_equal(bins, np.arange(len(bins))):
            raise ValueError(f"{path}: shell {shell} has missing or reordered bins")
        result[shell] = {
            "bin": bins,
            "wavelength_A": values[:, 1],
            "cs_J": values[:, 2],
            "mc_J": values[:, 3],
        }
    return result


def integrate_j(wavelength_a: np.ndarray, jnu: np.ndarray, lo_a: float | None = None,
                hi_a: float | None = None) -> float:
    mask = np.ones(len(wavelength_a), dtype=bool)
    if lo_a is not None:
        mask &= wavelength_a >= lo_a
    if hi_a is not None:
        mask &= wavelength_a <= hi_a
    if mask.sum() < 2:
        raise ValueError(f"only {mask.sum()} samples in requested wavelength band")
    nu = C_A / wavelength_a[mask]
    order = np.argsort(nu)
    return float(FOURPI_OVER_C * np.trapezoid(jnu[mask][order], nu[order]))


def cmfgen_field(cmfgen_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    jnu, fl, nd = read_eddfactor(cmfgen_dir / "EDDFACTOR")
    wavelength_a = 2997.92458 / fl
    velocity = parse_rvtj_block(cmfgen_dir / "RVTJ", "Velocity (km/s)", nd)
    return wavelength_a, jnu, velocity


def cmfgen_j_at_velocity(wavelength_a: np.ndarray, jnu: np.ndarray,
                         velocity: np.ndarray, target: float) -> np.ndarray:
    order = np.argsort(velocity)
    v = velocity[order]
    if not (v[0] <= target <= v[-1]):
        raise ValueError(f"target velocity {target} outside [{v[0]}, {v[-1]}]")
    right = int(np.searchsorted(v, target))
    if right == 0:
        return jnu[:, order[0]].copy()
    if right == len(v):
        return jnu[:, order[-1]].copy()
    left = right - 1
    v0, v1 = v[left], v[right]
    a, b = jnu[:, order[left]], jnu[:, order[right]]
    if np.any(a <= 0) or np.any(b <= 0):
        bad = int(np.sum((a <= 0) | (b <= 0)))
        raise ValueError(f"cannot log-interpolate {bad} non-positive CMFGEN J values at v={target}")
    fraction = (target - v0) / (v1 - v0)
    return 10.0 ** ((1.0 - fraction) * np.log10(a) + fraction * np.log10(b))


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
