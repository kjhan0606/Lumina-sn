#!/usr/bin/env python3
"""Build the CMFGEN-population input for emissivity campaign E1.

This is an import/audit tool, not a model run.  It reads the converged CMFGEN
RVTJ, *OUT departure coefficients, *PRRR Saha reference densities, oscillator
files, and POP* direct populations.  Covered CMFGEN fine levels are mapped to
Lumina levels only after (Z, ion, zero-based level, g, energy) identity checks.

The binary output is intentionally small and simple for the CPU C consumer:

  8s magic E1POP001, uint32 version, uint32 n_shells, uint64 n_levels,
  uint8 covered[n_levels], float64 n_k[n_levels,n_shells] (little endian).

Uncovered entries are quiet NaNs.  No floor, clamp, extrapolation, or guessed
ion is permitted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from oracle_compare_cmfgen import (  # noqa: E402
    model_superlevels,
    parse_out_bk,
    parse_prrr,
)
from cmfgen_extract.parse_pops import parse as parse_pop  # noqa: E402
sys.path.insert(0, str(ROOT / "validation/cmfgen_toy06_19p48d/analysis"))
from cmp_rvtj_T_ne_vs_published import read_rvtj  # noqa: E402


MAGIC = b"E1POP001"
HEADER = struct.Struct("<8sIIQ")
VERSION = 1
SAHA_COEFF = 2.07078e-22  # LTEPOP_WLD_V2, T is in 10^4 K
HNU_OVER_KT4 = 1.0e11 * 6.62607015e-27 / 1.380649e-16
CMINV_TO_EV = 1.2398419843320026e-4

# CMFGEN calls nickel "Nk".  POP file, OUT/PRRR token, Z, charge.
ELEMENTS = {
    "Sk": (14, "POPSIL", ("Sk2", "SkIII", "SkIV", "SkV")),
    "S": (16, "POPSUL", ("S2", "SIII", "SIV", "SV")),
    "Ca": (20, "POPCAL", ("Ca2", "CaIII", "CaIV", "CaV")),
    "Fe": (26, "POPIRON", ("Fe2", "FeIII", "FeIV", "FeV")),
    "Co": (27, "POPCOB", ("Co2", "CoIII", "CoIV", "CoV")),
    "Nk": (28, "POPNICK", ("Nk2", "NkIII", "NkIV", "NkV")),
}

STAGE = {"2": 1, "III": 2, "IV": 3, "V": 4, "SIX": 5}


def stage_of(label: str) -> int:
    for suffix in ("SIX", "III", "IV", "V", "2"):
        if label.endswith(suffix):
            return STAGE[suffix]
    raise ValueError(f"unrecognized CMFGEN ion label {label}")


def numbers(line: str) -> list[float] | None:
    toks = line.split()
    if not toks:
        return None
    try:
        return [float(x.replace("D", "E").replace("d", "e")) for x in toks]
    except ValueError:
        return None


def parse_out_headers(path: Path, nd: int, nlev: int) -> dict[str, np.ndarray]:
    """Read the eight-value depth headers and prove every b block width."""
    lines = path.read_text(errors="strict").splitlines()
    header = numbers(lines[3])
    if header is None or int(header[2]) != nlev or int(header[3]) != nd:
        raise ValueError(f"{path}: fourth-line dimensions disagree")
    raw = np.full((nd, 8), np.nan)
    k = 4
    for depth in range(1, nd + 1):
        while k < len(lines) and numbers(lines[k]) is None:
            k += 1
        row = numbers(lines[k]) if k < len(lines) else None
        if row is None or len(row) != 8 or int(round(row[-1])) != depth:
            raise ValueError(f"{path}:{k + 1}: invalid depth header {depth}")
        raw[depth - 1] = row
        k += 1
        got = 0
        while got < nlev:
            row = numbers(lines[k]) if k < len(lines) else None
            if row is None or got + len(row) > nlev:
                raise ValueError(f"{path}:{k + 1}: b block width drift")
            got += len(row)
            k += 1
    return {
        "radius_1e10cm": raw[:, 0],
        "next_ion_reference_density": raw[:, 1],
        "electron_density": raw[:, 2],
        "temperature_1e4K": raw[:, 3],
        "ion_fraction_header": raw[:, 4],
        "velocity_kms": raw[:, 5],
        "clump_factor": raw[:, 6],
        "depth_1based": raw[:, 7],
    }


def parse_fosc_levels(path: Path, wanted: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return g, excitation energy [eV], edge frequency [1e15 Hz]."""
    lines = path.read_text(errors="strict").splitlines()
    marker = next(i for i, line in enumerate(lines)
                  if "!Number of energy levels" in line)
    declared = int(lines[marker].split()[0])
    if declared < wanted:
        raise ValueError(f"{path}: declared {declared} < OUT NLEV {wanted}")
    vals: list[tuple[float, float, float]] = []
    # The first numeric atomic record follows the transition-count header.
    for line in lines[marker + 4:]:
        tok = line.split()
        if len(tok) < 6:
            continue
        try:
            g = float(tok[1])
            energy_cm = float(tok[2])
            edge = float(tok[3])
        except ValueError:
            continue
        expected = len(vals) + 1
        ids = [abs(int(x)) for x in tok[4:]
               if re.fullmatch(r"[+-]?\d+", x)]
        if expected not in ids:
            continue
        vals.append((g, energy_cm * CMINV_TO_EV, edge))
        if len(vals) == wanted:
            break
    if len(vals) != wanted:
        raise ValueError(f"{path}: parsed {len(vals)} != wanted {wanted}")
    arr = np.asarray(vals)
    return arr[:, 0], arr[:, 1], arr[:, 2]


def read_geometry(path: Path) -> np.ndarray:
    rows = list(csv.DictReader(path.open()))
    shell = np.asarray([int(r["shell_id"]) for r in rows])
    if not np.array_equal(shell, np.arange(len(rows))):
        raise ValueError(f"{path}: shell identity is not 0..N-1")
    return np.asarray([
        0.5 * (float(r["v_inner"]) + float(r["v_outer"])) / 1e5
        for r in rows
    ])


def read_lumina_levels(path: Path) -> list[dict[str, str]]:
    rows = list(csv.DictReader(path.open()))
    keys = [(int(r["atomic_number"]), int(r["ion_number"]),
             int(r["level_number"])) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{path}: duplicate (Z,ion,level)")
    return rows


def log_interp(v_native: np.ndarray, values: np.ndarray,
               v_shell: np.ndarray) -> np.ndarray:
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("population interpolation received nonpositive/nonfinite input")
    order = np.argsort(v_native)
    x = v_native[order]
    if np.any(np.diff(x) <= 0.0):
        raise ValueError("CMFGEN velocity grid is not unique")
    inside = (v_shell >= x[0]) & (v_shell <= x[-1])
    out = np.full((values.shape[1], len(v_shell)), np.nan)
    for lev in range(values.shape[1]):
        out[lev, inside] = np.exp(np.interp(
            v_shell[inside], x, np.log(values[:, lev][order])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmfgen-dir", type=Path,
                    default=Path("/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern"))
    ap.add_argument("--model-dir", type=Path,
                    default=ROOT / "data/tardis_reference_toy06_19p48d_sivcaiv")
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "validation/emiss_e1")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    velocity, ne, te = read_rvtj(str(args.cmfgen_dir / "RVTJ"))
    nd = len(velocity)
    te4 = te / 1e4
    shell_velocity = read_geometry(args.model_dir / "geometry.csv")
    shell_in_rvtj = ((shell_velocity >= velocity.min()) &
                     (shell_velocity <= velocity.max()))
    levels = read_lumina_levels(args.model_dir / "levels.csv")
    lookup = {(int(r["atomic_number"]), int(r["ion_number"]),
               int(r["level_number"])): i for i, r in enumerate(levels)}
    population = np.full((len(levels), len(shell_velocity)), np.nan)
    covered = np.zeros(len(levels), dtype=np.uint8)
    nsl = model_superlevels(args.cmfgen_dir / "MODEL_SPEC")
    audit: list[dict[str, object]] = []
    roundtrip: list[dict[str, object]] = []

    for prefix, (z, pop_name, ions) in ELEMENTS.items():
        pop_rows = parse_pop(str(args.cmfgen_dir / pop_name))
        direct = {}
        for ion in ions:
            stage = stage_of(ion)
            vals = np.asarray([r[4] for r in pop_rows
                               if r[0] == z and r[1] == stage])
            if vals.size:
                direct[stage] = vals.reshape(nd, -1)
        del pop_rows

        for ion in ions:
            stage = stage_of(ion)
            out_path = args.cmfgen_dir / f"{ion}OUT"
            b = parse_out_bk(out_path, nd)
            nlev = b.shape[1]
            hdr = parse_out_headers(out_path, nd, nlev)
            g, energy, edge = parse_fosc_levels(
                args.cmfgen_dir / f"{ion}_F_OSCDAT", nlev)

            next_label = {
                1: f"{prefix}III", 2: f"{prefix}IV",
                3: f"{prefix}V", 4: f"{prefix}SIX",
            }[stage]
            try:
                gion = float(parse_fosc_levels(
                    args.cmfgen_dir / f"{next_label}_F_OSCDAT", 1)[0][0])
            except (OSError, StopIteration, ValueError):
                audit.append({"Z": z, "ion": stage, "label": ion,
                              "status": "UNRESOLVED", "reason":
                              f"missing next-ion ground g for {next_label}"})
                continue

            if ion not in nsl:
                raise ValueError(f"MODEL_SPEC lacks {ion}_ISF")
            pr = parse_prrr(args.cmfgen_dir / f"{ion}PRRR", ion, nd, nsl[ion])
            dic = hdr["next_ion_reference_density"]
            # PRRR and OUT are compared as an audit only.  Some depths disagree
            # far beyond text precision; never use PRRR to alter the validated
            # OUT-header DIC input.
            pr_rel = np.max(np.abs(pr["ion_density"] / dic - 1.0))
            ne_rel = np.max(np.abs(hdr["electron_density"] / ne - 1.0))
            te_rel = np.max(np.abs(hdr["temperature_1e4K"] / te4 - 1.0))

            log_lte = (
                np.log(SAHA_COEFF) + np.log(ne)[:, None] + np.log(dic)[:, None]
                - 1.5 * np.log(te4)[:, None] - math.log(gion)
                + np.log(g)[None, :] + edge[None, :] * HNU_OVER_KT4 / te4[:, None]
            )
            native = b * np.exp(log_lte)
            if np.any(~np.isfinite(native)) or np.any(native <= 0.0):
                raise ValueError(f"{ion}: reconstructed population invalid")

            mapped = 0
            max_e_abs = 0.0
            g_mismatch = 0
            for lev in range(nlev):
                key = (z, stage, lev)
                if key not in lookup:
                    continue
                gl = lookup[key]
                e_lum = float(levels[gl]["energy_eV"])
                g_lum = int(levels[gl]["g"])
                de = abs(e_lum - energy[lev])
                max_e_abs = max(max_e_abs, de)
                if g_lum != int(round(g[lev])):
                    g_mismatch += 1
                    continue
                # The Lumina CSV carries 10 decimals, while the CMF oscillator
                # energy is printed to 6 decimals in cm^-1.  2e-6 eV safely
                # envelopes only that textual precision; it is not a physics cut.
                if de > 2.0e-6:
                    continue
                population[gl] = log_interp(velocity, native[:, lev:lev + 1],
                                            shell_velocity)[0]
                covered[gl] = 1
                mapped += 1

            rt = direct.get(stage)
            if rt is None or rt.shape[1] < nlev:
                rt_stats = {"direct_POP_available": False}
            else:
                ratio = native / rt[:, :nlev]
                finite = np.isfinite(ratio) & (ratio > 0.0)
                lr = np.log10(ratio[finite])
                rt_stats = {
                    "direct_POP_available": True,
                    "roundtrip_positive_count": int(finite.sum()),
                    "roundtrip_median_ratio": float(np.median(ratio[finite])),
                    "roundtrip_p01_ratio": float(10.0 ** np.quantile(lr, 0.01)),
                    "roundtrip_p99_ratio": float(10.0 ** np.quantile(lr, 0.99)),
                    "roundtrip_max_abs_dex": float(np.max(np.abs(lr))),
                }
            row = {
                "Z": z, "ion": stage, "label": ion,
                "status": "COVERED" if mapped else "UNRESOLVED",
                "OUT_NLEV": nlev, "Lumina_mapped_levels": mapped,
                "g_mismatch": g_mismatch, "max_energy_abs_eV": max_e_abs,
                "gion_next": gion, "PRRR_DIC_max_rel": float(pr_rel),
                "OUT_RVTJ_ne_max_rel": float(ne_rel),
                "OUT_RVTJ_T_max_rel": float(te_rel),
                **rt_stats,
            }
            audit.append(row)
            roundtrip.append(row)

    out_bin = args.out_dir / "cmfgen_b_populations.bin"
    with out_bin.open("wb") as fh:
        fh.write(HEADER.pack(MAGIC, VERSION, len(shell_velocity), len(levels)))
        fh.write(covered.tobytes(order="C"))
        fh.write(population.astype("<f8", copy=False).tobytes(order="C"))

    out_csv = args.out_dir / "population_import_audit.csv"
    fields = sorted({key for row in audit for key in row})
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(audit)

    summary = {
        "schema": "E1POP001-v1",
        "cmfgen_dir": str(args.cmfgen_dir),
        "model_dir": str(args.model_dir),
        "n_shells": len(shell_velocity),
        "n_shells_inside_rvtj": int(shell_in_rvtj.sum()),
        "unresolved_shells_outside_rvtj": np.flatnonzero(~shell_in_rvtj).tolist(),
        "rvtj_velocity_range_km_s": [float(velocity.min()), float(velocity.max())],
        "lumina_shell_velocity_range_km_s": [float(shell_velocity.min()),
                                               float(shell_velocity.max())],
        "n_levels": len(levels),
        "covered_levels": int(covered.sum()),
        "uncovered_levels": int((covered == 0).sum()),
        "covered_fraction": float(covered.mean()),
        "population_binary_sha256": hashlib.sha256(out_bin.read_bytes()).hexdigest(),
        "no_floor_or_clamp": True,
        "interpolation": ("linear in velocity and log(population); outside-RVTJ "
                          "shells are NaN, with no extrapolation or endpoint hold"),
        "saha_source": "CMFGEN LTEPOP_WLD_V2: 2.07078e-22*ne*DIC*T4^-1.5*g/gion*exp(edge*HDKT/T4)",
    }
    (args.out_dir / "population_import_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
