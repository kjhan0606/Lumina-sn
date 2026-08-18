#!/usr/bin/env python3
"""Close the newly opened SH-GRID band against CMFGEN-native atomic physics.

This is an offline discretisation test, not a convergence claim.  It pins one
CMFGEN EDDFACTOR/RVTJ/POP snapshot and evaluates every CMFGEN-backed active
level whose bound-free threshold lies in the 178 bins added below the former
1.5e14-Hz floor in two independent ways:

  native  raw CMFGEN photoionisation model on the CMFGEN EDDFACTOR grid;
  binned  the production cmfgen_sigma_bf.bin row and K=2 canonical J_nu rule.

Both lanes consume the same J_nu, T_e, n_e and level/ion populations.  The
reported emissivity is spontaneous Milne eta_nu integrated over frequency,
per steradian, matching src/lumina_plasma.c's BF-MILNE expression.  *PRRR is
deliberately not used: it is known to be state-inconsistent with POP at some
depths in this run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import sys
from typing import Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DECK = ROOT / "data/tardis_reference_toy06_19p48d_sivcaiv_active"
DEFAULT_RUN = Path("/gpfs/kjhan/cmfgen_runs/toy06_19p48d_ophys")
DEFAULT_OUT = ROOT / "validation/sh_grid_low_band_closure_2026-08-08"
EXPAND_PATH = ROOT / "scripts/expand_atomic_data_cmfgen.py"
PARSER_DIR = (ROOT / "validation/cmfgen_toy06_19p48d/analysis/"
              "gamma_coiii_alllevel")

H = 6.62607015e-27
KB = 1.380649e-16
C = 2.99792458e10
EV = 1.602176634e-12
ME = 9.1093837015e-28
FOUR_PI = 4.0 * math.pi
OLD_NU_MIN_NOMINAL = 1.5e14
EXPECTED_NBIN = 1234
EXPECTED_ADDED_BINS = 178
EXPECTED_NLEV = 24542
CMFGEN_MAGIC = 0x434D4644
CMFGEN_VERSION = 1

POP_FILE = {
    14: "POPSIL",
    16: "POPSUL",
    20: "POPCAL",
    26: "POPIRON",
    27: "POPCOB",
    28: "POPNICK",
}

# Acceptance gates declared independently of the candidate result.  Global and
# ion sums test the quantities the rate equations consume; the level-weighted
# L1 gates prevent cancellation.  A maximum single level/cell relative error is
# retained as a diagnostic, not a gate: separate sigma/J bin averages cannot
# reconstruct their covariance in an arbitrarily small threshold sliver.
GATE = {
    "sigma_reconstruction_max_rel": 5.0e-10,
    "global_depth_rate_max_rel": 2.0e-2,
    "global_depth_eta_max_rel": 2.0e-2,
    "ion_depth_rate_max_rel": 5.0e-2,
    "ion_depth_eta_max_rel": 5.0e-2,
    "level_weighted_l1_rate_max_rel": 3.0e-2,
    "level_weighted_l1_eta_max_rel": 3.0e-2,
}
SIGNIFICANT_FRACTION = 1.0e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_expand(links: Path, caps: dict[tuple[int, int], int]):
    # These are physics/provenance switches, not optional tuning knobs here.
    os.environ["CMFGEN_FULL_LEVELS"] = "1"
    os.environ["CMFGEN_SUPER_LEVELS"] = "1"
    os.environ["CMFGEN_LINK_FTOS"] = "1"
    os.environ["CMFGEN_EXACT_HYD"] = "1"
    os.environ["CMFGEN_LINKS"] = str(links)
    spec = importlib.util.spec_from_file_location("sh_grid_low_band_expand", EXPAND_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {EXPAND_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    module.CMFGEN_ROOT = ROOT / "data/atomic/cmfgen"
    module.ION_LEVEL_CAPS = caps
    return module


def load_snapshot_parsers():
    sys.path.insert(0, str(PARSER_DIR))
    from gamma_coiii_alllevel import (  # pylint: disable=import-error
        parse_popcob, read_eddfactor, rvtj_block,
    )
    return parse_popcob, read_eddfactor, rvtj_block


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_sigma(path: Path) -> dict[str, object]:
    with path.open("rb") as stream:
        header = stream.read(32)
        if len(header) != 32:
            raise RuntimeError("short sigma header")
        magic, version, nlev, nbin, numin, numax = struct.unpack("<IIiidd", header)
        flags = np.frombuffer(stream.read(nlev), dtype="i1").copy()
        pad = (8 - nlev % 8) % 8
        padding = stream.read(pad)
        offset = stream.tell()
    expected = offset + nlev * nbin * 8
    if path.stat().st_size != expected:
        raise RuntimeError(f"sigma size mismatch got={path.stat().st_size} expected={expected}")
    if padding != bytes(pad):
        raise RuntimeError("non-zero sigma alignment padding")
    sigma = np.memmap(path, dtype="<f8", mode="r", offset=offset,
                      shape=(nlev, nbin))
    return {
        "magic": magic, "version": version, "nlev": nlev, "nbin": nbin,
        "numin": numin, "numax": numax, "flags": flags, "sigma": sigma,
    }


def cum_from_below(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.zeros(x.size)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return out


def eval_cum(x: np.ndarray, y: np.ndarray, cumulative: np.ndarray,
             at: np.ndarray) -> np.ndarray:
    at = np.clip(np.asarray(at, dtype=float), x[0], x[-1])
    idx = np.clip(np.searchsorted(x, at), 1, x.size - 1)
    x0, x1 = x[idx - 1], x[idx]
    frac = (at - x0) / (x1 - x0)
    yat = y[idx - 1] + frac * (y[idx] - y[idx - 1])
    return cumulative[idx - 1] + 0.5 * (y[idx - 1] + yat) * (at - x0)


def bin_average(x: np.ndarray, y: np.ndarray, edges: np.ndarray) -> np.ndarray:
    cumulative = cum_from_below(x, y)
    return np.diff(eval_cum(x, y, cumulative, edges)) / np.diff(edges)


def local_bin_average(x: np.ndarray, y: np.ndarray,
                      edges: np.ndarray) -> np.ndarray:
    """Bin averages without subtracting a large below-band cumulative sum."""
    out = np.empty(edges.size - 1)
    for index, (lo_edge, hi_edge) in enumerate(zip(edges[:-1], edges[1:])):
        inside = (x > lo_edge) & (x < hi_edge)
        xb = np.concatenate(([lo_edge], x[inside], [hi_edge]))
        yb = np.interp(xb, x, y)
        out[index] = np.trapezoid(yb, xb) / (hi_edge - lo_edge)
    return out


def trap_weights(x: np.ndarray) -> np.ndarray:
    out = np.zeros(x.size)
    dx = np.diff(x)
    out[:-1] += 0.5 * dx
    out[1:] += 0.5 * dx
    return out


def relative_error(value: np.ndarray, reference: np.ndarray) -> np.ndarray:
    out = np.full(np.broadcast_shapes(np.shape(value), np.shape(reference)), np.nan)
    v, r = np.broadcast_arrays(value, reference)
    good = np.isfinite(v) & np.isfinite(r) & (r != 0.0)
    out[good] = np.abs(v[good] / r[good] - 1.0)
    both_zero = (v == 0.0) & (r == 0.0)
    out[both_zero] = 0.0
    return out


def max_finite(value: np.ndarray) -> float:
    finite = np.asarray(value)[np.isfinite(value)]
    return float(np.max(finite)) if finite.size else 0.0


def native_model(entry, nu_threshold: float, bake_model):
    """CMFGEN SUB_PHOT_GEN semantics; correct the tabulated high-u tail.

    The baker intentionally uses its own stored-bin convention.  CMFGEN types
    20/21/22 instead continue past the final node as u^-3.  The new low band is
    normally within the tables, but retaining this distinction makes the two
    lanes genuinely independent.
    """
    if entry.cs_type not in (20, 21, 22):
        return bake_model[0]
    u_node = np.asarray(entry.energy, dtype=float)
    s_node = np.asarray(entry.sigma_Mb, dtype=float) * 1.0e-18
    order = np.argsort(u_node)
    u_node, s_node = u_node[order], s_node[order]

    def evaluate(nu):
        nu = np.asarray(nu, dtype=float)
        out = np.zeros(nu.shape)
        mask = nu >= nu_threshold
        if not np.any(mask):
            return out
        u = nu[mask] / nu_threshold
        val = np.interp(u, u_node, s_node, left=s_node[0], right=0.0)
        tail = u >= u_node[-1]
        if np.any(tail):
            val[tail] = s_node[-1] * (u_node[-1] / u[tail]) ** 3
        out[mask] = val
        return out

    return evaluate


def selftest() -> int:
    edges = np.exp(np.linspace(math.log(2.0), math.log(8.0), 17))
    x = np.unique(np.concatenate((np.linspace(2.0, 8.0, 100001), edges)))
    sigma = np.full(x.size, 3.0)
    j = np.full(x.size, 5.0)
    sb = bin_average(x, sigma, edges)
    jb = bin_average(x, j, edges)
    native = np.trapezoid(sigma * j, x)
    binned = float(np.sum(sb * jb * np.diff(edges)))
    rel = abs(binned / native - 1.0)
    fourpi = FOUR_PI * binned
    ok = rel < 2e-12 and abs(fourpi / binned - FOUR_PI) < 2e-15
    print(f"[SH-GRID][LOW-BAND][SELFTEST] rel={rel:.3e} "
          f"fourpi_ratio={fourpi/binned:.17g} status={'PASS' if ok else 'FAIL'}")
    return 0 if ok else 4


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", type=Path, default=DEFAULT_DECK)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sigma", type=Path,
                        help="candidate sigma asset (default: deck canonical)")
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    if args.chunk < 1:
        parser.error("--chunk must be positive")

    deck = args.deck.resolve()
    run = args.run.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    provenance = json.loads((deck / "DECK_PROVENANCE.json").read_text())
    active = read_rows(deck / "active_ions.csv")
    levels = read_rows(deck / "levels.csv")
    ionization = read_rows(deck / "ionization_energies.csv")
    caps = {(int(r["atomic_number"]), int(r["ion_stage"])): int(r["n_full"])
            for r in active}
    if len(caps) != len(active):
        raise SystemExit("duplicate active ion")
    links = Path(provenance["cmfgen_links"])
    if sha256(links) != provenance["cmfgen_links_sha256"]:
        raise SystemExit("CMFGEN links hash changed")

    sigma_path = args.sigma.resolve() if args.sigma else deck / "cmfgen_sigma_bf.bin"
    binary = read_sigma(sigma_path)
    if (binary["magic"] != CMFGEN_MAGIC or binary["version"] != CMFGEN_VERSION or
            binary["nlev"] != EXPECTED_NLEV or binary["nlev"] != len(levels) or
            binary["nbin"] != EXPECTED_NBIN):
        raise SystemExit("active sigma header/level contract mismatch")
    if not np.all((binary["flags"] == 0) | (binary["flags"] == 1)):
        raise SystemExit("sigma flag outside {0,1}")

    nbin = int(binary["nbin"])
    numin, numax = float(binary["numin"]), float(binary["numax"])
    dlog = math.log(numax / numin) / nbin
    edges = numin * np.exp(np.arange(nbin + 1) * dlog)
    centers = numin * np.exp((np.arange(nbin) + 0.5) * dlog)
    widths = np.diff(edges)
    band_edges = edges[:EXPECTED_ADDED_BINS + 1]
    band_centers = centers[:EXPECTED_ADDED_BINS]
    band_widths = widths[:EXPECTED_ADDED_BINS]
    canonical_edges = np.empty(2 * EXPECTED_ADDED_BINS + 1)
    canonical_edges[0::2] = band_edges
    canonical_edges[1::2] = np.sqrt(band_edges[:-1] * band_edges[1:])
    canonical_bf_bin = np.arange(2 * EXPECTED_ADDED_BINS) // 2
    band_hi = float(band_edges[-1])
    if abs(band_hi / OLD_NU_MIN_NOMINAL - 1.0) > 5e-15:
        raise SystemExit("178-bin boundary no longer recovers old grid floor")

    zlev = np.array([int(r["atomic_number"]) for r in levels], dtype=int)
    ilev = np.array([int(r["ion_number"]) for r in levels], dtype=int)
    nlev = np.array([int(r["level_number"]) for r in levels], dtype=int)
    elev = np.array([float(r["energy_eV"]) for r in levels])
    glev = np.array([float(r["g"]) for r in levels])
    chi = {(int(r["atomic_number"]), int(r["ion_number"])):
           float(r["ionization_energy_eV"]) for r in ionization}
    threshold = np.array([(chi[(int(z), int(i))] - e) * EV / H
                          for z, i, e in zip(zlev, ilev, elev)])
    selected = np.where((ilev >= 1) & (threshold >= numin) &
                        (threshold <= OLD_NU_MIN_NOMINAL) &
                        (binary["flags"] == 1))[0]
    if selected.size != 707:
        raise SystemExit(f"new-band active level census changed: {selected.size} != 707")

    expand = load_expand(links, caps)
    ion_data = expand.parse_all_ions()
    if set(ion_data) != set(caps):
        raise SystemExit("parsed ion set differs from active deck")
    for key, data in ion_data.items():
        nf = caps[key]
        data["n_kept"] = nf
        data["levels"] = data["osc"].levels[:nf]
    rebuilt, lookup, _ = expand.build_global_levels(ion_data)
    if len(rebuilt) != len(levels):
        raise SystemExit("rebuilt level count differs")
    for idx, row in enumerate(rebuilt):
        if (row[0], row[1], row[2]) != (zlev[idx], ilev[idx], nlev[idx]):
            raise SystemExit(f"rebuilt level identity differs at global={idx}")

    selected_set = set(int(x) for x in selected)
    models: dict[int, tuple[Callable, Callable, np.ndarray, float, float, int, str]] = {}
    phot_paths: set[Path] = set()
    for (z, stage), data in sorted(ion_data.items()):
        phot = data.get("phot")
        if phot is None or not phot.entries:
            continue
        phot_path = data["provenance"].get("phot_path")
        if phot_path is not None:
            phot_paths.add(Path(phot_path))
        levs = data["levels"]
        nf = data["n_kept"]
        cfg = {expand._norm_cfg(levs["config"][k]): k for k in range(nf)}
        terms: dict[str, list[int]] = {}
        for k in range(nf):
            terms.setdefault(expand._term_cfg(levs["config"][k]), []).append(k)
        zion = float(data["osc"].z_screen)
        for entry in phot.entries:
            target = cfg.get(expand._norm_cfg(entry.config))
            targets = [target] if target is not None else \
                terms.get(expand._term_cfg(entry.config), [])
            for lev in targets:
                global_index = lookup.get((z, stage, lev + 1))
                if global_index not in selected_set:
                    continue
                eth = data["osc"].ionization_eV - \
                    float(levs["E_cm"][lev]) * 1.239841984e-4
                nth = eth * EV / H
                nef = expand._cmfgen_nef(z, zion, nth)
                bake = expand._sigma_model(entry.cs_type, entry.energy,
                                           entry.sigma_Mb, nth,
                                           zion=zion, nef=nef)
                if bake is None:
                    continue
                native = native_model(entry, nth, bake)
                models[global_index] = (bake[0], native, bake[1], bake[2], nth,
                                        int(entry.cs_type), str(bake[3]))
    if set(models) != selected_set:
        missing = sorted(selected_set - set(models))
        raise SystemExit(f"raw CMFGEN evaluator missing for {len(missing)} rows: {missing[:8]}")

    # Reconstruct the exact added-band rows with the baker before comparing any
    # rates.  This prevents a stale/stand-in asset from passing through a
    # fortuitous population weighting.
    reconstructed = np.zeros((selected.size, EXPECTED_ADDED_BINS))
    for q, global_index in enumerate(selected):
        fn, _, nodes, start, _, _, _ = models[int(global_index)]
        reconstructed[q] = expand._bin_average_sigma(
            fn, nodes, band_edges, band_widths, start)
    stored = np.asarray(binary["sigma"][selected, :EXPECTED_ADDED_BINS])
    sigma_abs = np.abs(stored - reconstructed)
    sigma_scale = np.maximum(np.abs(reconstructed), 1e-300)
    sigma_rel = np.where((stored == 0.0) & (reconstructed == 0.0), 0.0,
                         sigma_abs / sigma_scale)
    sigma_max_rel = max_finite(sigma_rel)
    sigma_max_abs = max_finite(sigma_abs)

    parse_pop, read_edd, rvtj_block = load_snapshot_parsers()
    J, nu_cmf, nd, finish = read_edd(str(run / "EDDFACTOR"))
    if not np.isfinite(finish) or finish == 0.0:
        raise SystemExit("EDDFACTOR is incomplete")
    rvtj_text = (run / "RVTJ").read_text()
    velocity = rvtj_block(rvtj_text, "Velocity (km/s)", nd)
    temperature = rvtj_block(rvtj_text, "Temperature (10^4K)", nd) * 1.0e4
    electron = rvtj_block(rvtj_text, "Electron density", nd)
    if min(velocity.size, temperature.size, electron.size) != nd:
        raise SystemExit("short RVTJ block")

    active_by_z: dict[int, list[dict[str, str]]] = {}
    for row in active:
        active_by_z.setdefault(int(row["atomic_number"]), []).append(row)
    pop_by_ion: dict[tuple[int, int], np.ndarray] = {}
    ion_total: dict[tuple[int, int], np.ndarray] = {}
    pop_hash_paths: list[Path] = []
    roundtrip_bad = 0
    for z, rows in sorted(active_by_z.items()):
        pop_path = run / POP_FILE[z]
        pop_hash_paths.append(pop_path)
        _, ions, order, nd_pop = parse_pop(str(pop_path))
        if nd_pop != nd:
            raise SystemExit(f"{pop_path.name}: ND mismatch")
        expected_labels = [r["model_label"] for r in sorted(
            rows, key=lambda x: int(x["ion_stage"]))]
        if order != expected_labels:
            raise SystemExit(f"{pop_path.name}: ion order {order} != {expected_labels}")
        for row in rows:
            stage = int(row["ion_stage"])
            ion_csv = int(row["ion_number"])
            arr = ions[row["model_label"]][0]
            if arr.shape != (nd, int(row["n_full"])):
                raise SystemExit(f"{row['model_label']}: population shape mismatch")
            pop_by_ion[(z, ion_csv)] = arr
            ion_total[(z, ion_csv)] = arr.sum(axis=1)
        for lower, upper in zip(order[:-1], order[1:]):
            dci = ions[lower][1]
            ground = ions[upper][0][:, 0]
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(ground > 0.0, dci / ground, 1.0)
            roundtrip_bad += int(np.sum(~np.isfinite(ratio) |
                                        (np.abs(ratio - 1.0) > 1e-6)))
    if roundtrip_bad:
        raise SystemExit(f"POP DCI round-trip failures={roundtrip_bad}")

    level_population = np.empty((selected.size, nd))
    upper_density = np.zeros((selected.size, nd))
    upper_partition = np.ones((selected.size, nd))
    eta_eligible = np.zeros(selected.size, dtype=bool)
    for q, global_index in enumerate(selected):
        z, ion, lev = int(zlev[global_index]), int(ilev[global_index]), int(nlev[global_index])
        level_population[q] = pop_by_ion[(z, ion)][:, lev]
        upper_key = (z, ion + 1)
        if upper_key not in pop_by_ion:
            continue
        upper_density[q] = ion_total[upper_key]
        rows_upper = np.where((zlev == z) & (ilev == ion + 1))[0]
        xu = elev[rows_upper, None] * EV / (KB * temperature[None, :])
        weights = np.where(xu < 50.0, glev[rows_upper, None] * np.exp(-xu), 0.0)
        part = weights.sum(axis=0)
        fallback = max(1.0, float(glev[rows_upper[0]]))
        upper_partition[q] = np.where(part >= 1.0, part, fallback)
        eta_eligible[q] = True

    Jbar = np.empty((EXPECTED_ADDED_BINS, nd))
    Jcanonical = np.empty((2 * EXPECTED_ADDED_BINS, nd))
    for depth in range(nd):
        Jbar[:, depth] = local_bin_average(
            nu_cmf, J[:, depth], band_edges)
        Jcanonical[:, depth] = local_bin_average(
            nu_cmf, J[:, depth], canonical_edges)
    if (np.any(~np.isfinite(Jbar)) or np.any(Jbar < 0.0) or
            np.any(~np.isfinite(Jcanonical)) or np.any(Jcanonical < 0.0)):
        raise SystemExit("non-finite/negative J in the new band")

    mask_native = (nu_cmf > band_edges[0]) & (nu_cmf < band_edges[-1])
    nu_native = np.concatenate(([band_edges[0]], nu_cmf[mask_native],
                                [band_edges[-1]]))
    J_native = np.empty((nu_native.size, nd))
    J_native[1:-1] = J[mask_native]
    for depth in range(nd):
        J_native[0, depth] = np.interp(nu_native[0], nu_cmf, J[:, depth])
        J_native[-1, depth] = np.interp(nu_native[-1], nu_cmf, J[:, depth])
    w_native = trap_weights(nu_native)

    y_rate_native = J_native / (H * nu_native[:, None])
    x_native = H * nu_native[:, None] / (KB * temperature[None, :])
    y_eta_native = (2.0 * H * nu_native[:, None] ** 3 / C ** 2) * np.exp(-x_native)
    projection_native = np.concatenate((y_rate_native, y_eta_native), axis=1)

    nsel = selected.size
    gamma_native = np.zeros((nsel, nd))
    gamma_binned = np.zeros((nsel, nd))
    eta_integral_native = np.zeros((nsel, nd))
    eta_integral_binned = np.zeros((nsel, nd))
    for q0 in range(0, nsel, args.chunk):
        q1 = min(q0 + args.chunk, nsel)
        raw_sigma = np.empty((q1 - q0, nu_native.size))
        for local, global_index in enumerate(selected[q0:q1]):
            raw_sigma[local] = models[int(global_index)][1](nu_native)
        projected = (raw_sigma * w_native[None, :]) @ projection_native
        gamma_native[q0:q1] = FOUR_PI * projected[:, :nd]
        eta_integral_native[q0:q1] = projected[:, nd:]

        # Production photo-rate consumption.  The stored CMFGEN value is a
        # full-BF-bin linear-frequency average, including zero support below
        # the physical edge.  Relocate that conserved sigma*dnu mass onto the
        # active sub-interval, clamp both K=2 canonical halves to the edge,
        # and integrate the resulting step exactly against 1/nu.
        binned_sigma = stored[q0:q1].copy()
        edge_chunk = np.array([models[int(index)][4]
                               for index in selected[q0:q1]])
        sigma_step = binned_sigma[:, canonical_bf_bin].copy()
        bf_lo = band_edges[:-1][canonical_bf_bin]
        bf_hi = band_edges[1:][canonical_bf_bin]
        partial_canonical = ((bf_lo[None, :] < edge_chunk[:, None]) &
                             (bf_hi[None, :] > edge_chunk[:, None]) &
                             (sigma_step > 0.0))
        active_width = bf_hi[None, :] - edge_chunk[:, None]
        sigma_step = np.where(
            partial_canonical,
            sigma_step * (bf_hi - bf_lo)[None, :] / active_width,
            sigma_step)
        active_lo = np.maximum(canonical_edges[:-1][None, :],
                               edge_chunk[:, None])
        active_hi = canonical_edges[1:][None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            log_width = np.where(active_hi > active_lo,
                                 np.log(active_hi / active_lo), 0.0)
        photo_kernel = sigma_step * log_width / H
        gamma_binned[q0:q1] = FOUR_PI * (photo_kernel @ Jcanonical)

        # Production Milne consumption retains the conserved full-bin mass;
        # threshold-sensitive factors use a representative point in the
        # active part of the one partial BF bin.
        weighted_bin = binned_sigma * band_widths[None, :]
        partial = ((band_centers[None, :] < edge_chunk[:, None]) &
                   (band_edges[1:][None, :] > edge_chunk[:, None]))
        nu_eta = np.where(partial,
                          0.5 * (edge_chunk[:, None] + band_edges[1:][None, :]),
                          band_centers[None, :])
        x_eta = (H * nu_eta[:, :, None] /
                 (KB * temperature[None, None, :]))
        y_eta = ((2.0 * H / C ** 2) * nu_eta[:, :, None] ** 3 *
                 np.exp(-x_eta))
        eta_integral_binned[q0:q1] = np.sum(
            weighted_bin[:, :, None] * y_eta, axis=1)

    nu_edge = np.array([models[int(index)][4] for index in selected])
    lam3 = (H * H / (2.0 * math.pi * ME * KB * temperature)) ** 1.5
    eta_prefactor = (electron[None, :] * upper_density * glev[selected, None] *
                     lam3[None, :] / (2.0 * upper_partition))
    edge_boltz = np.exp(H * nu_edge[:, None] / (KB * temperature[None, :]))
    eta_native = eta_integral_native * edge_boltz * eta_prefactor
    eta_binned = eta_integral_binned * edge_boltz * eta_prefactor
    eta_native[~eta_eligible] = 0.0
    eta_binned[~eta_eligible] = 0.0

    rate_volume_native = gamma_native * level_population
    rate_volume_binned = gamma_binned * level_population
    rate_total_native = rate_volume_native.sum(axis=0)
    rate_total_binned = rate_volume_binned.sum(axis=0)
    eta_total_native = eta_native.sum(axis=0)
    eta_total_binned = eta_binned.sum(axis=0)
    rate_depth_rel = relative_error(rate_total_binned, rate_total_native)
    eta_depth_rel = relative_error(eta_total_binned, eta_total_native)

    ions = sorted(set((int(zlev[i]), int(ilev[i])) for i in selected))
    ion_records = []
    ion_rate_rel_values = []
    ion_eta_rel_values = []
    for z, ion in ions:
        take = (zlev[selected] == z) & (ilev[selected] == ion)
        rn = rate_volume_native[take].sum(axis=0)
        rb = rate_volume_binned[take].sum(axis=0)
        en = eta_native[take].sum(axis=0)
        eb = eta_binned[take].sum(axis=0)
        rr, er = relative_error(rb, rn), relative_error(eb, en)
        rate_sig = rn >= SIGNIFICANT_FRACTION * rate_total_native
        eta_sig = en >= SIGNIFICANT_FRACTION * eta_total_native
        ion_rate_rel_values.extend(rr[rate_sig].tolist())
        ion_eta_rel_values.extend(er[eta_sig].tolist())
        for depth in range(nd):
            ion_records.append((z, ion, depth + 1, velocity[depth], rn[depth],
                                rb[depth], rr[depth], en[depth], eb[depth], er[depth],
                                int(rate_sig[depth]), int(eta_sig[depth])))

    level_rate_rel = relative_error(rate_volume_binned, rate_volume_native)
    level_eta_rel = relative_error(eta_binned, eta_native)
    level_rate_sig = rate_volume_native >= \
        SIGNIFICANT_FRACTION * rate_total_native[None, :]
    level_eta_sig = eta_native >= \
        SIGNIFICANT_FRACTION * eta_total_native[None, :]
    sig_level_rate_max = max_finite(level_rate_rel[level_rate_sig])
    sig_level_eta_max = max_finite(level_eta_rel[level_eta_sig])
    ion_rate_max = max_finite(np.asarray(ion_rate_rel_values))
    ion_eta_max = max_finite(np.asarray(ion_eta_rel_values))
    level_rate_l1 = (np.sum(np.abs(rate_volume_binned - rate_volume_native), axis=0) /
                     np.maximum(rate_total_native, 1e-300))
    level_eta_l1 = (np.sum(np.abs(eta_binned - eta_native), axis=0) /
                    np.maximum(eta_total_native, 1e-300))

    metrics = {
        "sigma_reconstruction_max_abs_cm2": sigma_max_abs,
        "sigma_reconstruction_max_rel": sigma_max_rel,
        "global_depth_rate_max_rel": max_finite(rate_depth_rel),
        "global_depth_eta_max_rel": max_finite(eta_depth_rel),
        "ion_depth_rate_max_rel": ion_rate_max,
        "ion_depth_eta_max_rel": ion_eta_max,
        "level_weighted_l1_rate_max_rel": max_finite(level_rate_l1),
        "level_weighted_l1_eta_max_rel": max_finite(level_eta_l1),
        "significant_level_rate_max_rel": sig_level_rate_max,
        "significant_level_eta_max_rel": sig_level_eta_max,
        "significant_level_rate_cells": int(level_rate_sig.sum()),
        "significant_level_eta_cells": int(level_eta_sig.sum()),
    }
    checks = {
        name: metrics[name] <= limit for name, limit in GATE.items()
    }
    checks.update({
        "finite_nonnegative_rate": bool(np.all(np.isfinite(gamma_native)) and
                                         np.all(np.isfinite(gamma_binned)) and
                                         np.all(gamma_native >= 0.0) and
                                         np.all(gamma_binned >= 0.0)),
        "finite_nonnegative_eta": bool(np.all(np.isfinite(eta_native)) and
                                        np.all(np.isfinite(eta_binned)) and
                                        np.all(eta_native >= 0.0) and
                                        np.all(eta_binned >= 0.0)),
        "pop_roundtrip": roundtrip_bad == 0,
        "edd_complete": bool(np.isfinite(finish) and finish != 0.0),
        "level_coverage_707": selected.size == 707 and len(models) == 707,
    })
    verdict = "PASS" if all(checks.values()) else "FAIL"

    depth_path = out / "low_band_depth_summary.csv"
    with depth_path.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("depth", "velocity_km_s", "T_e_K", "n_e_cm3",
                         "photo_rate_native_cm3_s1", "photo_rate_binned_cm3_s1",
                         "photo_rate_relerr", "eta_native_erg_s1_cm3_sr1",
                         "eta_binned_erg_s1_cm3_sr1", "eta_relerr"))
        for depth in range(nd):
            writer.writerow((depth + 1, f"{velocity[depth]:.10e}",
                             f"{temperature[depth]:.10e}", f"{electron[depth]:.10e}",
                             f"{rate_total_native[depth]:.17e}",
                             f"{rate_total_binned[depth]:.17e}",
                             f"{rate_depth_rel[depth]:.17e}",
                             f"{eta_total_native[depth]:.17e}",
                             f"{eta_total_binned[depth]:.17e}",
                             f"{eta_depth_rel[depth]:.17e}"))

    ion_path = out / "low_band_ion_depth.csv"
    with ion_path.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("atomic_number", "ion_number", "depth", "velocity_km_s",
                         "photo_rate_native_cm3_s1", "photo_rate_binned_cm3_s1",
                         "photo_rate_relerr", "eta_native_erg_s1_cm3_sr1",
                         "eta_binned_erg_s1_cm3_sr1", "eta_relerr",
                         "rate_significant", "eta_significant"))
        writer.writerows(ion_records)

    level_path = out / "low_band_level_depth.csv"
    with level_path.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("global_index", "atomic_number", "ion_number", "level_number",
                         "cs_type", "sigma_path", "nu_edge_hz", "depth",
                         "velocity_km_s", "level_population_cm3", "gamma_native_s1",
                         "gamma_binned_s1", "photo_rate_native_cm3_s1",
                         "photo_rate_binned_cm3_s1", "photo_rate_relerr",
                         "eta_native_erg_s1_cm3_sr1", "eta_binned_erg_s1_cm3_sr1",
                         "eta_relerr", "rate_significant", "eta_significant"))
        for q, global_index in enumerate(selected):
            model = models[int(global_index)]
            for depth in range(nd):
                writer.writerow((int(global_index), int(zlev[global_index]),
                                 int(ilev[global_index]), int(nlev[global_index]),
                                 model[5], model[6], f"{model[4]:.17e}", depth + 1,
                                 f"{velocity[depth]:.10e}",
                                 f"{level_population[q, depth]:.17e}",
                                 f"{gamma_native[q, depth]:.17e}",
                                 f"{gamma_binned[q, depth]:.17e}",
                                 f"{rate_volume_native[q, depth]:.17e}",
                                 f"{rate_volume_binned[q, depth]:.17e}",
                                 f"{level_rate_rel[q, depth]:.17e}",
                                 f"{eta_native[q, depth]:.17e}",
                                 f"{eta_binned[q, depth]:.17e}",
                                 f"{level_eta_rel[q, depth]:.17e}",
                                 int(level_rate_sig[q, depth]),
                                 int(level_eta_sig[q, depth])))

    formal_path = run / "PROVENANCE_FORMAL.json"
    formal = json.loads(formal_path.read_text()) if formal_path.is_file() else {}
    input_paths = [run / "EDDFACTOR", run / "EDDFACTOR_INFO", run / "RVTJ",
                   *pop_hash_paths, deck / "DECK_PROVENANCE.json",
                   deck / "active_ions.csv", deck / "levels.csv",
                   deck / "ionization_energies.csv", sigma_path, links,
                   EXPAND_PATH, Path(__file__).resolve(), *sorted(phot_paths)]
    hashes = {str(path): sha256(path) for path in input_paths}
    manifest = {
        "schema": "lumina-sh-grid-low-band-closure-v1",
        "verdict": verdict,
        "scope": "offline discretisation closure; not a solver-convergence claim",
        "cmfgen_snapshot_status": "intentionally unconverged capture",
        "cmfgen_prrr_used": False,
        "cmfgen_prrr_reason": "known POP/PRRR state mismatch at some depths",
        "snapshot": {
            "run": str(run), "nd": nd, "edd_finish_record": float(finish),
            "formal_provenance": formal,
            "rvtj_hash_matches_formal": formal.get("files", {}).get("RVTJ") == hashes[str(run / "RVTJ")],
        },
        "grid": {
            "n_bins": nbin, "nu_min_hz": numin, "nu_max_hz": numax,
            "added_bins": EXPECTED_ADDED_BINS, "band_lo_hz": float(band_edges[0]),
            "band_hi_hz": band_hi, "old_nu_min_nominal_hz": OLD_NU_MIN_NOMINAL,
            "dlog_nu": dlog,
        },
        "units": {
            "J_nu": "erg s^-1 cm^-2 Hz^-1 sr^-1",
            "gamma_level": "s^-1",
            "photo_rate_volume": "cm^-3 s^-1",
            "eta_bf_integrated": "erg s^-1 cm^-3 sr^-1",
        },
        "conventions": {
            "photo_rate": "4pi integral sigma J_nu/(h nu) dnu",
            "emissivity": "spontaneous Milne; eta_nu per sr; no 4pi factor",
            "binned_photo_consumption": "K=2 canonical J_nu; sigma full-bin-average mass relocated to active support; exact step integral against 1/nu",
            "binned_milne_consumption": "sigma full-bin-average mass retained; factors use active partial-bin representative frequency",
            "native_consumption": "raw CMFGEN sigma on EDDFACTOR frequency samples, trapezoid",
            "single_level_max_role": "diagnostic only; sigma/J sub-bin covariance is not identifiable from separate bin averages",
        },
        "coverage": {
            "levels": int(selected.size), "depths": nd,
            "eta_eligible_levels": int(eta_eligible.sum()),
            "eta_ineligible_top_stage_levels": int((~eta_eligible).sum()),
            "native_frequency_points_in_band": int(nu_native.size),
            "canonical_frequency_bins_in_band": int(Jcanonical.shape[0]),
        },
        "gates": GATE,
        "diagnostic_significant_fraction": SIGNIFICANT_FRACTION,
        "metrics": metrics,
        "checks": checks,
        "artifacts": {
            "depth_summary": depth_path.name,
            "ion_depth": ion_path.name,
            "level_depth": level_path.name,
        },
        "input_sha256": hashes,
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    verdict_path = out / "VERDICT.md"
    lines = [
        "# SH-GRID 신규 저주파 band 동일-snapshot 폐합",
        "",
        f"판정: **{verdict}**",
        "",
        "이 시험은 미수렴 CMFGEN capture의 수렴도를 판정하지 않는다. 같은 고정",
        "`EDDFACTOR/RVTJ/POP*`를 두 적분 경로가 소비할 때 새 178개 bin의 BF",
        "광이온화율과 자발 Milne 방출률이 닫히는지만 판정한다.",
        "",
        f"- 대상: {selected.size} levels x {nd} depths",
        f"- sigma 재구성 max rel: `{sigma_max_rel:.6e}`",
        f"- depth 합계 photo-rate max rel: `{metrics['global_depth_rate_max_rel']:.6e}`",
        f"- depth 합계 eta max rel: `{metrics['global_depth_eta_max_rel']:.6e}`",
        f"- significant ion-depth photo/eta max rel: `{ion_rate_max:.6e}` / `{ion_eta_max:.6e}`",
        f"- level-weighted L1 photo/eta max rel: `{metrics['level_weighted_l1_rate_max_rel']:.6e}` / `{metrics['level_weighted_l1_eta_max_rel']:.6e}`",
        f"- significant level-depth photo/eta max rel: `{sig_level_rate_max:.6e}` / `{sig_level_eta_max:.6e}`",
        "",
        "마지막 level-depth 최대값은 진단치다. 임계면의 매우 얇은 한 빈에서",
        "서로 따로 평균된 sigma와 J만으로 sub-bin 공분산을 복원할 수 없으므로,",
        "수용 판정은 전역/ion 합과 기여도 가중 L1 폐합으로 한다.",
        "",
        "`*PRRR`은 사용하지 않았다. 이 run에서는 일부 깊이에서 POP와 상태가",
        "다르므로 동일-snapshot truth가 아니기 때문이다.",
    ]
    verdict_path.write_text("\n".join(lines) + "\n")

    print(f"[SH-GRID][LOW-BAND] levels={selected.size} depths={nd} "
          f"native_nu={nu_native.size} eta_eligible={int(eta_eligible.sum())}")
    for name, value in metrics.items():
        print(f"[SH-GRID][LOW-BAND][METRIC] {name}={value}")
    for name, passed in checks.items():
        print(f"[SH-GRID][LOW-BAND][CHECK] {name}={'PASS' if passed else 'FAIL'}")
    print(f"[SH-GRID][LOW-BAND][{verdict}] manifest={manifest_path}")
    return 0 if verdict == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
