#!/usr/bin/env python3
"""CMFGEN의 동일 Jν·상태에서 Si/S 정·역 bound-free 율을 대조한다.

이 진단은 전체 자기일관 분포 판정이 아니다. 같은 CMFGEN snapshot의
EDDFACTOR, POP, RVTJ, PRRR을 고정하고 Lumina 생산 경로의 광이온화와
v4 route Milne 역재결합을 재생해 주파수 적분·상위 SL 점유율·FL→SL 집계를
비교한다. 고정-T 또는 상태 불일치 자료의 PASS는 물리 동등성 승인이 아니다.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CERT = ROOT / "validation/cmfgen_toy06_19p48d/analysis/rates_certification"
GAMMA = ROOT / "validation/cmfgen_toy06_19p48d/analysis/gamma_coiii_alllevel"
ANALYSIS = ROOT / "validation/cmfgen_toy06_19p48d/analysis"
sys.path[:0] = [str(ROOT / "scripts"), str(CERT), str(GAMMA), str(ANALYSIS)]

import a2_05_l1bf_gate as l1bf  # noqa: E402
import expand_atomic_data_cmfgen as atomic  # noqa: E402
from cmfgen_parser import parse_osc  # noqa: E402
from cmfgen_phot_smoothing import PhotSmoother  # noqa: E402
from gamma_coiii_alllevel import (  # noqa: E402
    parse_f_to_s, parse_popcob, read_eddfactor,
)
from cmp_rvtj_T_ne_vs_published import read_rvtj  # noqa: E402
from oracle_compare_cmfgen import (  # noqa: E402
    numeric_line, parse_prrr as parse_prrr_summary,
)
from ptf_gated_metrics import (  # noqa: E402
    ScrtempReader, read_dimensions, read_variables,
)

TARGETS = {(14, 2): "POPSIL", (14, 3): "POPSIL",
           (16, 2): "POPSUL", (16, 3): "POPSUL"}
RATE_MEDIAN_LIMIT_DEX = 0.03
RATE_P95_LIMIT_DEX = 0.10
RATE_MAX_LIMIT_DEX = 0.20
CM_TO_EV = 1.239841984e-4
H_CGS = 6.62607015e-27
C_CGS = 2.99792458e10
M_E_CGS = 9.1093837015e-28
K_B_CGS = 1.380649e-16
EV_TO_ERG = 1.602176634e-12
BF_CHANNEL_MAGIC = 0x434D4643
BF_CHANNEL_VERSION = 4
ELEMENT_POP_CLOSURE_TOL_REL = 2.0e-6
PRRR_NEXT_SL1_TOL_REL = 1.1e-4
PRRR_RVTJ_ELECTRON_TOL_REL = 5.0e-5
PRRR_RVTJ_TEMPERATURE_TOL_REL = 5.0e-5
POST_POP_SCRTEMP_TOL_REL = 2.0e-5
SUP_TO_FULL_POP_TOL_REL = 2.0e-5
VALID, EXACT_ZERO = 1, 2
SCRTEMP_TEMPERATURE_SCALE_K = 1.0e4
CMFGEN_HDKT = 1.0e11 * H_CGS / K_B_CGS
CMFGEN_LEVEL_DISSOLUTION_RYD_HZ15 = 3.289395
CMFGEN_LTE_POPULATION_FACTOR = 2.07078e-22


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_sealed_deck(run: Path, deck: Path) -> tuple[dict, dict, dict]:
    stamp_path = deck / "DECK_PROVENANCE.json"
    identity_path = deck / "atomic_source_identity.json"
    stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    require(stamp.get("schema") == "lumina-deck-provenance-v1",
            "deck provenance schema mismatch")
    require(Path(stamp["deck"]).resolve() == deck.resolve(),
            "deck provenance points to another output deck")
    phot_smoother = None
    if stamp.get("phot_preprocessing"):
        require(Path(stamp["phot_preprocessing"]["model"]["path"]).resolve() ==
                (run / "MODEL").resolve(), "PHOT controls belong to another run")
        phot_smoother = PhotSmoother.from_provenance(stamp["phot_preprocessing"])
    paths = {
        "cmfgen_links": run / "atomic_links.txt",
        "cmfgen_model_spec": run / "MODEL_SPEC",
        "level_sl_steq_links": run / "LEVEL_SL_STEQ_LINKS",
    }
    hashes = {
        "cmfgen_links": "cmfgen_links_sha256",
        "cmfgen_model_spec": "cmfgen_model_spec_sha256",
        "level_sl_steq_links": "level_sl_steq_links_sha256",
    }
    for key, path in paths.items():
        require(Path(stamp[key]).resolve() == path.resolve(),
                f"deck provenance {key} path mismatch")
        require(sha256_file(path) == stamp[hashes[key]],
                f"sealed CMFGEN input changed: {path}")
    require(identity_path.resolve() ==
            Path(stamp["atomic_source_identity"]["path"]).resolve(),
            "source identity path mismatch")
    require(sha256_file(identity_path) == stamp["atomic_source_identity"]["sha256"],
            "source identity checksum mismatch")
    manifest = deck / "atomic_vintage_manifest.csv"
    require(sha256_file(manifest) == identity["atomic_vintage_manifest_sha256"],
            "atomic vintage manifest checksum mismatch")

    link_map = atomic.load_cmfgen_links(paths["cmfgen_links"])
    model_map = atomic.load_cmfgen_model_spec(
        paths["cmfgen_model_spec"], paths["cmfgen_links"])
    runtime_map = atomic.load_cmfgen_level_sl_map(paths["level_sl_steq_links"])
    require(set(model_map) == set(link_map), "MODEL_SPEC/link ion-set mismatch")
    expected_names = {
        f"{atomic.SYM[z]} {atomic.ROMAN[stage]}": (key, spec)
        for key, spec in model_map.items()
        for z, stage in (key,)
    }
    identity_rows = {row["ion"]: row for row in identity["ions"]}
    require(set(identity_rows) == set(expected_names),
            "deck active ion set differs from MODEL_SPEC")
    require(set(stamp["active_ions"]) == set(expected_names),
            "provenance ion set differs from MODEL_SPEC")
    for name, (key, spec) in expected_names.items():
        row = identity_rows[name]
        require(row["model_spec_nf"] == int(spec["n_full"]),
                f"{name}: source identity NF mismatch")
        require(row["active_superlevels"] == int(spec["n_super"]),
                f"{name}: source identity NS mismatch")
        require(stamp["active_ions"][name]["model_spec_nf"] == int(spec["n_full"]),
                f"{name}: provenance NF mismatch")
        require(stamp["active_ions"][name]["active_superlevels"] == int(spec["n_super"]),
                f"{name}: provenance NS mismatch")

    source_rows = {}
    for ion in identity["ions"]:
        for role, source in ion["sources"].items():
            path = Path(source["resolved_path"])
            source_rows[str(path)] = source["sha256"]
    for path_text, expected_sha in source_rows.items():
        path = Path(path_text)
        require(path.is_file(), f"atomic source disappeared: {path}")
        require(sha256_file(path) == expected_sha,
                f"atomic source changed after deck generation: {path}")

    route_manifest_path = deck / "cmfgen_bf_channels_manifest.json"
    route_manifest = json.loads(route_manifest_path.read_text(encoding="utf-8"))
    require(route_manifest.get("schema") in ("lumina-cmfgen-bf-channels-v4", "lumina-cmfgen-bf-channels-v5"),
            "BF channel manifest is not a supported grouped-SL contract")
    channel_path = deck / "cmfgen_bf_channels.bin"
    sigma_path = deck / "cmfgen_sigma_bf.bin"
    require(sha256_file(channel_path) == route_manifest.get("channel_bin_sha256"),
            "BF channel binary checksum differs from manifest")
    require(sha256_file(sigma_path) == route_manifest.get("sigma_bin_sha256"),
            "BF sigma binary checksum differs from route manifest")
    for ion_record in route_manifest.get("ions", {}).values():
        for source in ion_record.get("sources", []):
            path = Path(source["path"])
            require(path.is_file(), f"BF route source disappeared: {path}")
            require(sha256_file(path) == source["sha256"],
                    f"BF route source checksum mismatch: {path}")
    return stamp, identity, {"links": link_map, "model": model_map,
                             "runtime": runtime_map,
                             "route_manifest": route_manifest,
                             "phot_smoother": phot_smoother}


def parse_cmfgen_bf_channels(path: Path, expected_levels: int,
                             expected_bins: int, manifest: dict) -> dict:
    """Read and validate the sealed v4 CSR channel table used by Lumina."""
    raw = path.read_bytes()
    header_size = struct.calcsize("<IIiiidd")
    if len(raw) < header_size:
        raise ValueError(f"{path}: truncated BF channel header")
    magic, version, n_levels, n_bins, n_routes, nu_min, nu_max = struct.unpack_from(
        "<IIiiidd", raw, 0)
    if (magic != BF_CHANNEL_MAGIC or version not in (4, 5) or
            manifest.get("schema", "lumina-cmfgen-bf-channels-v4") != f"lumina-cmfgen-bf-channels-v{version}" or
            n_levels != expected_levels or n_bins != expected_bins or
            n_routes != int(manifest.get("n_channels", -1)) or
            n_levels != int(manifest.get("n_levels", -1)) or
            n_bins != int(manifest.get("n_freq_bins", -1)) or
            n_routes <= 0 or not (0.0 < nu_min < nu_max)):
        raise ValueError(f"{path}: BF channel header/manifest mismatch")
    offset = header_size

    def take(dtype: str, count: int) -> np.ndarray:
        nonlocal offset
        dt = np.dtype(dtype).newbyteorder("<")
        end = offset + dt.itemsize * count
        if end > len(raw):
            raise ValueError(f"{path}: truncated BF channel array {dtype}[{count}]")
        values = np.frombuffer(raw, dtype=dt, count=count, offset=offset).copy()
        offset = end
        return values

    offsets = take("i4", n_levels + 1)
    targets = take("i4", n_routes)
    probability = take("f8", n_routes)
    phot_grid = take("i4", n_routes)
    target_energy_eV = take("f8", n_routes)
    target_g = take("f8", n_routes)
    sigma_edge = take("f8", n_routes)
    sigma = take("f8", n_routes * n_bins).reshape(n_routes, n_bins)
    if version == 5:
        from cmfgen_recombination_policy import validate_tail
        validate_tail(raw[offset:], manifest.get("recombination_policy"))
    elif offset != len(raw):
        raise ValueError(f"{path}: trailing bytes after BF channel arrays")
    if (offsets[0] != 0 or offsets[-1] != n_routes or
            np.any(np.diff(offsets) < 0) or
            np.any((targets < -1) | (targets >= n_levels)) or
            not np.isfinite(probability).all() or np.any(probability <= 0.0) or
            np.any(phot_grid < 0) or
            not np.isfinite(target_energy_eV).all() or
            np.any(target_energy_eV < 0.0) or
            not np.isfinite(target_g).all() or np.any(target_g <= 0.0) or
            not np.isfinite(sigma_edge).all() or np.any(sigma_edge < 0.0) or
            not np.isfinite(sigma).all() or np.any(sigma < 0.0)):
        raise ValueError(f"{path}: invalid BF channel values")
    return {
        "n_levels": n_levels, "n_bins": n_bins, "n_routes": n_routes,
        "nu_min": nu_min, "nu_max": nu_max, "offsets": offsets,
        "targets": targets, "probability": probability,
        "phot_grid": phot_grid, "target_energy_eV": target_energy_eV,
        "target_g": target_g, "sigma_edge": sigma_edge, "sigma": sigma,
    }


def parse_prrr_superlevel_rates(path: Path, ion_label: str,
                                nd: int, nsl: int,
                                process: str = "photoionization") -> np.ndarray:
    """엄격히 PRRR의 광이온화 또는 복사재결합 SL 율밀도 블록을 읽는다."""
    labels = {
        "photoionization": f"{ion_label} Photoionization Rates",
        "recombination": f"{ion_label} Recombination Rates",
    }
    if process not in labels:
        raise ValueError(f"unknown PRRR rate process: {process}")
    block_label = labels[process]
    lines = path.read_text(errors="strict").replace("\f", "\n").splitlines()
    result = np.full((nsl, nd), np.nan, dtype=float)
    chunks: set[int] = set()
    i = 0
    while i < len(lines):
        if lines[i].strip() != "Depth index":
            i += 1
            continue
        if i + 1 >= len(lines):
            raise ValueError(f"{path}: missing depth row")
        depth_values = numeric_line(lines[i + 1])
        if not depth_values:
            raise ValueError(f"{path}:{i + 2}: invalid depth row")
        depths = [int(value) - 1 for value in depth_values]
        if any(float(value) != depth + 1
               for value, depth in zip(depth_values, depths)):
            raise ValueError(f"{path}: non-integral depth index {depth_values}")
        if any(depth < 0 or depth >= nd for depth in depths):
            raise ValueError(f"{path}: depth outside 1..{nd}: {depths}")
        start = depths[0]
        expected_depths = list(range(start, min(start + 10, nd)))
        if (start in chunks or depths != list(range(start, start + len(depths))) or
                depths != expected_depths):
            raise ValueError(f"{path}: duplicate/non-contiguous depth block {depths}")
        chunks.add(start)
        width = len(depths)
        block = None
        k = i + 2
        while k < len(lines) and lines[k].strip() != "Depth index":
            if lines[k].strip() != block_label:
                k += 1
                continue
            if block is not None:
                raise ValueError(f"{path}: duplicate {ion_label} rate block at depths {depths}")
            rows = []
            for sl in range(nsl):
                row = numeric_line(lines[k + 1 + sl]) if k + 1 + sl < len(lines) else None
                if row is None or len(row) != width:
                    raise ValueError(
                        f"{path}:{k + 2 + sl}: SL row {sl + 1}/{nsl} width mismatch")
                rows.append(row)
            after = k + nsl + 1
            after_rates = after
            while after_rates < len(lines) and not lines[after_rates].strip():
                after_rates += 1
            if (after_rates < len(lines) and
                    numeric_line(lines[after_rates]) is not None):
                raise ValueError(f"{path}:{after + 1}: more than MODEL_SPEC N_SL={nsl} rows")
            block = np.asarray(rows, dtype=float)
            k = after
        if block is None:
            raise ValueError(f"{path}: {ion_label} rates missing at depths {depths}")
        result[:, depths] = block
        i = k
    expected = set(range(0, nd, 10))
    if chunks != expected:
        raise ValueError(f"{path}: depth chunks {sorted(chunks)} != {sorted(expected)}")
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError(f"{path}: rates incomplete, non-finite, or negative")
    return result


def parse_prrr_temperature(path: Path, nd: int) -> np.ndarray:
    """Read the PRRR temperature profile (printed in units of 1e4 K)."""
    lines = path.read_text(errors="strict").replace("\f", "\n").splitlines()
    result = np.full(nd, np.nan, dtype=float)
    chunks: set[int] = set()
    i = 0
    while i < len(lines):
        if lines[i].strip() != "Depth index":
            i += 1
            continue
        if i + 1 >= len(lines):
            raise ValueError(f"{path}: missing depth row")
        depth_values = numeric_line(lines[i + 1])
        if not depth_values:
            raise ValueError(f"{path}:{i + 2}: invalid depth row")
        depths = [int(value) - 1 for value in depth_values]
        if any(float(value) != depth + 1
               for value, depth in zip(depth_values, depths)):
            raise ValueError(f"{path}: non-integral depth index {depth_values}")
        if any(depth < 0 or depth >= nd for depth in depths):
            raise ValueError(f"{path}: depth outside 1..{nd}: {depths}")
        start = depths[0]
        expected_depths = list(range(start, min(start + 10, nd)))
        if (start in chunks or depths != list(range(start, start + len(depths))) or
                depths != expected_depths):
            raise ValueError(f"{path}: duplicate/non-contiguous depth block {depths}")
        chunks.add(start)
        width = len(depths)
        temperature = None
        k = i + 2
        while k < len(lines) and lines[k].strip() != "Depth index":
            if lines[k].strip() != "Temperature [1.0E+4K]":
                k += 1
                continue
            if temperature is not None:
                raise ValueError(f"{path}: duplicate temperature block at depths {depths}")
            values = numeric_line(lines[k + 1]) if k + 1 < len(lines) else None
            if values is None or len(values) != width:
                raise ValueError(f"{path}:{k + 2}: temperature width mismatch")
            temperature = np.asarray(values, dtype=float)
            k += 2
        if temperature is None:
            raise ValueError(f"{path}: temperature missing at depths {depths}")
        if not np.isfinite(temperature).all() or np.any(temperature <= 0.0):
            raise ValueError(f"{path}: invalid temperature at depths {depths}")
        result[depths] = temperature * 1.0e4
        i = k
    expected = set(range(0, nd, 10))
    if chunks != expected:
        raise ValueError(f"{path}: depth chunks {sorted(chunks)} != {sorted(expected)}")
    if not np.isfinite(result).all() or np.any(result <= 0.0):
        raise ValueError(f"{path}: incomplete or invalid temperature profile")
    return result


def rate_metrics(reference: np.ndarray, lumina: np.ndarray) -> dict:
    if not (np.isfinite(reference).all() and np.isfinite(lumina).all()):
        raise ValueError("non-finite rate")
    if np.any(reference < 0) or np.any(lumina < 0):
        raise ValueError("negative rate")
    ref_zero = reference == 0.0
    lum_zero = lumina == 0.0
    both_zero = ref_zero & lum_zero
    asymmetric = ref_zero ^ lum_zero
    positive = ~(ref_zero | lum_zero)
    dex = np.abs(np.log10(lumina[positive] / reference[positive]))
    if dex.size:
        median = float(np.median(dex))
        p95 = float(np.quantile(dex, 0.95))
        maximum = float(dex.max())
    else:
        median = p95 = maximum = None
    return {
        "n_compared_positive": int(positive.sum()),
        "n_both_zero": int(both_zero.sum()),
        "n_asymmetric_zero": int(asymmetric.sum()),
        "median_abs_delta_dex": median,
        "p95_abs_delta_dex": p95,
        "max_abs_delta_dex": maximum,
    }


def milne_recombination_rate_density(
        sigma_row: np.ndarray, j_nu: np.ndarray, j_state: np.ndarray,
        nu_min: float, nu_max: float,
        threshold_eV: float, temperature: np.ndarray,
        electron_density: np.ndarray, g_lower: float, g_upper: float | np.ndarray,
        upper_population: np.ndarray, probability: float) -> np.ndarray:
    """Reproduce Lumina's explicit-channel v4 Milne rate for one route.

    The returned quantity is the route's recombination rate density. The
    channel coefficient is multiplied by the actual target-superlevel
    population, matching the production owner where v4 assigns the whole
    grouped superlevel (not an individual fine-member fraction) to the route.
    """
    sigma = np.asarray(sigma_row, dtype=float)
    field = np.asarray(j_nu, dtype=float)
    states = np.asarray(j_state, dtype=np.int32)
    temp = np.asarray(temperature, dtype=float)
    ne = np.asarray(electron_density, dtype=float)
    upper = np.asarray(upper_population, dtype=float)
    upper_weight = np.broadcast_to(np.asarray(g_upper, dtype=float), temp.shape)
    if (sigma.ndim != 1 or field.ndim != 2 or field.shape[1] != sigma.size or
            states.shape != field.shape or
            temp.ndim != 1 or ne.shape != temp.shape or upper.shape != temp.shape or
            field.shape[0] != temp.size):
        raise ValueError("Milne rate dimensions disagree")
    if (not np.isfinite(sigma).all() or np.any(sigma < 0.0) or
            not np.isfinite(field).all() or np.any(field < 0.0) or
            not np.isfinite(temp).all() or np.any(temp <= 0.0) or
            not np.isfinite(ne).all() or np.any(ne <= 0.0) or
            not np.isfinite(upper).all() or np.any(upper < 0.0) or
            not (0.0 < nu_min < nu_max) or
            not (threshold_eV > 0.0) or
            not (g_lower > 0.0) or not np.isfinite(upper_weight).all() or
            np.any(upper_weight <= 0.0) or
            not (probability > 0.0 and probability <= 1.0)):
        raise ValueError("invalid Milne rate input")

    threshold_hz = threshold_eV * EV_TO_ERG / H_CGS
    if threshold_hz >= nu_max:
        return np.zeros_like(temp)
    d_log_nu = math.log(nu_max / nu_min) / sigma.size
    log_nu_min = math.log(nu_min)
    milne = np.zeros_like(temp)
    for index, sigma_average in enumerate(sigma):
        if sigma_average <= 0.0:
            continue
        log_lo = log_nu_min + index * d_log_nu
        nu_lo = math.exp(log_lo)
        nu_hi = math.exp(log_lo + d_log_nu)
        if nu_hi <= threshold_hz:
            continue
        active_lo = max(nu_lo, threshold_hz)
        active_width = nu_hi - active_lo
        if active_width <= 0.0:
            continue
        sigma_active = sigma_average
        if nu_lo < threshold_hz:
            sigma_active *= (nu_hi - nu_lo) / active_width
        sigma_log_weight = sigma_active * math.log(nu_hi / active_lo)
        if sigma_log_weight <= 0.0:
            continue
        if np.any((states[:, index] != VALID) & (states[:, index] != EXACT_ZERO)):
            raise ValueError(
                f"Milne field validity is missing in active bin {index}; "
                "do not substitute an unobserved J_nu with zero")
        nu_center = math.exp(log_lo + 0.5 * d_log_nu)
        nu_milne = (0.5 * (threshold_hz + nu_hi)
                    if nu_center < threshold_hz < nu_hi else nu_center)
        x = H_CGS * nu_milne / (K_B_CGS * temp)
        active = x < 700.0
        if not np.any(active):
            continue
        spontaneous = 2.0 * H_CGS * nu_milne**3 / C_CGS**2
        prefactor = 4.0 * math.pi / H_CGS * sigma_log_weight
        milne[active] += (prefactor * (spontaneous + field[active, index]) *
                          np.exp(-x[active]))

    kT = K_B_CGS * temp
    log_nstar = (np.log(ne) + 1.5 * np.log(
        H_CGS * H_CGS / (2.0 * math.pi * M_E_CGS * kT)) +
        np.log(g_lower / (2.0 * upper_weight)) +
        threshold_eV * EV_TO_ERG / kT)
    if not np.isfinite(log_nstar).all() or np.any(
            log_nstar >= math.log(np.finfo(float).max)):
        raise ValueError("non-finite/overflowing Milne Saha factor")
    nstar = np.exp(log_nstar)
    rate_density = probability * nstar * milne * upper
    if not np.isfinite(rate_density).all() or np.any(rate_density < 0.0):
        raise ValueError("invalid Milne rate density")
    return rate_density


def lumina_recombination_rate_density(
        channels: dict, Z: int, stage: int, n_full: int, n_super: int,
        lower_sl_map: np.ndarray, lower_energy_eV: np.ndarray,
        lower_g: np.ndarray, ionization_eV: float,
        upper_stage: int, upper_sl_map: np.ndarray,
        upper_energy_eV: np.ndarray, upper_g: np.ndarray,
        upper_population_by_sl: np.ndarray, level_index: dict,
        j_nu: np.ndarray, j_state: np.ndarray, temperature: np.ndarray,
        electron_density: np.ndarray) -> np.ndarray:
    """Assemble v4 route Milne densities into the lower-ion superlevels."""
    result = np.zeros((n_super, len(temperature)), dtype=float)
    if (lower_sl_map.shape != (n_full,) or lower_energy_eV.shape != (n_full,) or
            lower_g.shape != (n_full,) or upper_population_by_sl.ndim != 2 or
            upper_sl_map.ndim != 1 or upper_sl_map.size == 0 or
            upper_energy_eV.shape != upper_sl_map.shape or
            upper_g.shape != upper_sl_map.shape or
            upper_population_by_sl.shape[0] != len(temperature) or
            upper_population_by_sl.shape[1] != int(np.max(upper_sl_map))):
        raise ValueError("lower/upper superlevel data dimensions disagree")
    if upper_stage != stage + 1:
        raise ValueError("BF route target stage is not the adjacent ion")
    if (not np.isfinite(upper_energy_eV).all() or
            not np.isfinite(upper_g).all() or np.any(upper_g <= 0.0)):
        raise ValueError("invalid target-superlevel energies/statistical weights")

    global_to_level = {index: key for key, index in level_index.items()}
    partition_cache = {}
    for lower_level in range(n_full):
        lower_global = level_index.get((Z, stage, lower_level + 1))
        if lower_global is None:
            raise ValueError(f"missing deck global index for Z={Z} stage={stage} FL={lower_level + 1}")
        begin, end = (int(channels["offsets"][lower_global]),
                      int(channels["offsets"][lower_global + 1]))
        if begin == end:
            raise ValueError(f"no v4 BF channel for Z={Z} stage={stage} FL={lower_level + 1}")
        probability_by_grid: dict[int, float] = {}
        for route in range(begin, end):
            target_global = int(channels["targets"][route])
            target_key = global_to_level.get(target_global)
            if target_key is None or target_key[:2] != (Z, upper_stage):
                raise ValueError(f"BF route {route}: target is not an active adjacent-ion level")
            target_level = int(target_key[2])
            if target_level < 1 or target_level > len(upper_sl_map):
                raise ValueError(f"BF route {route}: upper target level outside MODEL_SPEC NF")
            upper_sl = int(upper_sl_map[target_level - 1]) - 1
            if upper_sl < 0 or upper_sl >= upper_population_by_sl.shape[1]:
                raise ValueError(f"BF route {route}: target superlevel outside active model")
            grid = int(channels["phot_grid"][route])
            probability = float(channels["probability"][route])
            probability_by_grid[grid] = probability_by_grid.get(grid, 0.0) + probability
            threshold_eV = (ionization_eV - float(lower_energy_eV[lower_level]) +
                            float(channels["target_energy_eV"][route]))
            anchor_energy = float(upper_energy_eV[target_level - 1])
            if not math.isclose(anchor_energy,
                                float(channels["target_energy_eV"][route]),
                                rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(f"BF route {route}: target anchor energy mismatch")
            partition_key = (upper_sl, anchor_energy)
            if partition_key not in partition_cache:
                members = upper_sl_map == upper_sl + 1
                partition_cache[partition_key] = np.sum(
                    upper_g[members][None, :] * np.exp(
                        -(upper_energy_eV[members] - anchor_energy)[None, :] *
                        EV_TO_ERG / (K_B_CGS * temperature[:, None])), axis=1)
            route_rate = milne_recombination_rate_density(
                channels["sigma"][route], j_nu, j_state, channels["nu_min"],
                channels["nu_max"], threshold_eV, temperature,
                electron_density, float(lower_g[lower_level]),
                partition_cache[partition_key],
                upper_population_by_sl[:, upper_sl], probability)
            lower_sl = int(lower_sl_map[lower_level]) - 1
            if lower_sl < 0 or lower_sl >= n_super:
                raise ValueError(f"lower FL {lower_level + 1}: invalid F_TO_S superlevel")
            result[lower_sl] += route_rate
        if any(abs(total - 1.0) > 1.0e-12
               for total in probability_by_grid.values()):
            raise ValueError(
                f"Z={Z} stage={stage} FL={lower_level + 1}: route probabilities "
                f"do not normalize per PHOT grid: {probability_by_grid}")
    return result


def sum_full_level_populations_by_superlevel(
        full_populations: np.ndarray, full_level_superlevels: np.ndarray,
        n_superlevels: int) -> np.ndarray:
    """Recover each CMFGEN SL population by summing its printed full levels."""
    populations = np.asarray(full_populations, dtype=float)
    superlevels = np.asarray(full_level_superlevels, dtype=int)
    if (populations.ndim != 2 or superlevels.shape != (populations.shape[1],) or
            n_superlevels < 1 or np.any(superlevels < 1) or
            np.any(superlevels > n_superlevels) or
            set(np.unique(superlevels)) != set(range(1, n_superlevels + 1)) or
            not np.isfinite(populations).all() or np.any(populations < 0.0)):
        raise ValueError("invalid full-level population/superlevel map")
    grouped = np.zeros((populations.shape[0], n_superlevels), dtype=float)
    for level, superlevel in enumerate(superlevels):
        grouped[:, superlevel - 1] += populations[:, level]
    return grouped


def prrr_next_stage_sl1_population(
        populations_by_ion: dict, ion_label: str,
        next_ion_label: str | None,
        next_superlevel_populations: np.ndarray | None = None) -> np.ndarray:
    """PRRR DXzV is next-ion SL1, not an individual full-level population.

    SUM_STEQ_SOL assigns the extra DXzV unknown to ION_ID(ID+1), VEC_SL=1.
    POPCOB stores full-level populations, so callers must first sum all full
    levels mapped to SL1. At the top active ion DXzV is the terminal DCI.
    """
    if next_ion_label is not None:
        grouped = np.asarray(next_superlevel_populations, dtype=float)
        levels = np.asarray(populations_by_ion[next_ion_label][0], dtype=float)
        if (levels.ndim != 2 or grouped.ndim != 2 or
                grouped.shape[0] != levels.shape[0] or grouped.shape[1] < 1):
            raise ValueError(f"{next_ion_label}: missing next-stage SL1 population")
        return grouped[:, 0]
    terminal = np.asarray(populations_by_ion[ion_label][1], dtype=float)
    if terminal.ndim != 1:
        raise ValueError(f"{ion_label}: invalid terminal ground population")
    return terminal


def match_prrr_scrtemp_state(
        reader, variable_indices: dict[tuple[str, int], int],
        prrr_profiles: dict[str, dict], electron_density_index: int,
        temperature_index: int) -> tuple[int, np.ndarray, dict[str, dict]]:
    """Find the unique SCRTEMP record matching PRRR's pre-solve state."""
    candidates = []
    for record_index, iteration in enumerate(
            range(reader.state_first_it, reader.last_it + 1), start=1):
        state = reader.state(iteration)
        if (state.ndim != 2 or electron_density_index >= state.shape[0] or
                temperature_index >= state.shape[0]):
            raise ValueError("SCRTEMP state dimensions do not contain ED/T variables")
        electron_density = np.asarray(state[electron_density_index], dtype=float)
        temperature = (np.asarray(state[temperature_index], dtype=float) *
                       SCRTEMP_TEMPERATURE_SCALE_K)
        if (not np.isfinite(electron_density).all() or
                np.any(electron_density <= 0.0) or
                not np.isfinite(temperature).all() or np.any(temperature <= 0.0)):
            continue
        per_ion = {}
        matches = True
        for ion_label, profile in prrr_profiles.items():
            next_key = (str(profile["next_ion"]), 1)
            if next_key not in variable_indices:
                raise ValueError(f"SCRTEMP is missing next-ion SL1 equation {next_key}")
            sl1 = np.asarray(state[variable_indices[next_key]], dtype=float)
            prrr_sl1 = np.asarray(profile["ion_density"], dtype=float)
            prrr_ne = np.asarray(profile["electron_density"], dtype=float)
            prrr_temperature = np.asarray(profile["temperature"], dtype=float)
            if any(values.shape != electron_density.shape for values in
                   (sl1, prrr_sl1, prrr_ne, prrr_temperature)):
                raise ValueError(f"{ion_label}: PRRR/SCRTEMP depth dimensions disagree")
            if (not np.isfinite(sl1).all() or np.any(sl1 <= 0.0) or
                    not np.isfinite(prrr_sl1).all() or np.any(prrr_sl1 <= 0.0) or
                    not np.isfinite(prrr_ne).all() or np.any(prrr_ne <= 0.0) or
                    not np.isfinite(prrr_temperature).all() or
                    np.any(prrr_temperature <= 0.0)):
                raise ValueError(f"{ion_label}: invalid PRRR/SCRTEMP matching profile")
            errors = {
                "next_SL1_max_rel": float(np.max(np.abs(sl1 / prrr_sl1 - 1.0))),
                "electron_density_max_rel": float(
                    np.max(np.abs(electron_density / prrr_ne - 1.0))),
                "temperature_max_rel": float(
                    np.max(np.abs(temperature / prrr_temperature - 1.0))),
            }
            per_ion[ion_label] = errors
            matches &= (
                errors["next_SL1_max_rel"] <= PRRR_NEXT_SL1_TOL_REL and
                errors["electron_density_max_rel"] <= PRRR_RVTJ_ELECTRON_TOL_REL and
                errors["temperature_max_rel"] <= PRRR_RVTJ_TEMPERATURE_TOL_REL
            )
        if matches:
            candidates.append((record_index, state, per_ion))
    if len(candidates) != 1:
        indices = [candidate[0] for candidate in candidates]
        raise ValueError(
            "PRRR must match exactly one SCRTEMP pre-solve record; "
            f"matched records={indices}")
    return candidates[0]


def scrtemp_superlevel_populations(
        state: np.ndarray, ion_label: str, n_superlevels: int,
        variable_indices: dict[tuple[str, int], int]) -> np.ndarray:
    """Extract one ion's SL populations from a decoded SCRTEMP state."""
    rows = []
    for superlevel in range(1, n_superlevels + 1):
        key = (ion_label, superlevel)
        if key not in variable_indices:
            raise ValueError(f"SCRTEMP is missing population equation {key}")
        rows.append(np.asarray(state[variable_indices[key]], dtype=float))
    result = np.stack(rows, axis=1)
    if not np.isfinite(result).all() or np.any(result < 0.0):
        raise ValueError(f"{ion_label}: invalid SCRTEMP superlevel populations")
    return result


def match_scrtemp_state_to_final_populations(
        reader, variable_indices: dict[tuple[str, int], int],
        final_by_ion: dict[str, np.ndarray], electron_density_index: int,
        temperature_index: int, rvtj_electron_density: np.ndarray,
        rvtj_temperature: np.ndarray) -> tuple[int, np.ndarray, dict[str, dict]]:
    """Find the unique SCRTEMP record underlying final POP and RVTJ output."""
    candidates = []
    for record_index, iteration in enumerate(
            range(reader.state_first_it, reader.last_it + 1), start=1):
        state = reader.state(iteration)
        if (state.ndim != 2 or electron_density_index >= state.shape[0] or
                temperature_index >= state.shape[0]):
            raise ValueError("SCRTEMP state dimensions do not contain ED/T variables")
        ne = np.asarray(state[electron_density_index], dtype=float)
        temperature = (np.asarray(state[temperature_index], dtype=float) *
                       SCRTEMP_TEMPERATURE_SCALE_K)
        if not np.isfinite(ne).all() or not np.isfinite(temperature).all():
            continue
        ne_error = float(np.max(np.abs(ne / rvtj_electron_density - 1.0)))
        t_error = float(np.max(np.abs(temperature / rvtj_temperature - 1.0)))
        ion_errors = {}
        matches = (ne_error <= PRRR_RVTJ_ELECTRON_TOL_REL and
                   t_error <= PRRR_RVTJ_TEMPERATURE_TOL_REL)
        for ion_label, reference in final_by_ion.items():
            actual = scrtemp_superlevel_populations(
                state, ion_label, reference.shape[1], variable_indices)
            reference = np.asarray(reference, dtype=float)
            if actual.shape != reference.shape:
                raise ValueError(f"{ion_label}: final POP/SCRTEMP shape mismatch")
            both_zero = (actual == 0.0) & (reference == 0.0)
            both_positive = (actual > 0.0) & (reference > 0.0)
            rel = np.full(reference.shape, np.inf, dtype=float)
            rel[both_zero] = 0.0
            rel[both_positive] = np.abs(
                actual[both_positive] / reference[both_positive] - 1.0)
            maximum = float(np.max(rel))
            ion_errors[ion_label] = maximum
            matches &= maximum <= POST_POP_SCRTEMP_TOL_REL
        if matches:
            candidates.append((record_index, state, {
                "electron_density_max_rel": ne_error,
                "temperature_max_rel": t_error,
                "per_ion_max_rel": ion_errors,
            }))
    if len(candidates) != 1:
        indices = [candidate[0] for candidate in candidates]
        raise ValueError(
            "final POP/RVTJ must match exactly one SCRTEMP record; "
            f"matched records={indices}")
    return candidates[0]


def population_identity_metrics(reference: np.ndarray,
                                reconstructed: np.ndarray) -> dict:
    """Compare full-level populations without hiding asymmetric zeros."""
    ref = np.asarray(reference, dtype=float)
    calc = np.asarray(reconstructed, dtype=float)
    if (ref.ndim != 2 or ref.shape != calc.shape or not np.isfinite(ref).all() or
            not np.isfinite(calc).all() or np.any(ref < 0.0) or
            np.any(calc < 0.0)):
        raise ValueError("invalid full-level population comparison arrays")
    both_zero = (ref == 0.0) & (calc == 0.0)
    both_positive = (ref > 0.0) & (calc > 0.0)
    rel = np.full(ref.shape, np.inf, dtype=float)
    rel[both_zero] = 0.0
    rel[both_positive] = np.abs(calc[both_positive] / ref[both_positive] - 1.0)
    finite = rel[np.isfinite(rel)]
    max_witness = None
    if finite.size:
        finite_indices = np.flatnonzero(np.isfinite(rel))
        flat_index = finite_indices[int(np.argmax(finite))]
        depth, level = np.unravel_index(flat_index, rel.shape)
        max_witness = {
            "depth_1based": int(depth + 1),
            "full_level_1based": int(level + 1),
            "reference_population": float(ref[depth, level]),
            "reconstructed_population": float(calc[depth, level]),
            "abs_relative": float(rel[depth, level]),
        }
    return {
        "n_both_zero": int(np.count_nonzero(both_zero)),
        "n_asymmetric_zero": int(rel.size - np.count_nonzero(both_zero) -
                                  np.count_nonzero(both_positive)),
        "n_compared_positive": int(np.count_nonzero(both_positive)),
        "max_abs_relative": float(finite.max()) if finite.size else None,
        "p95_abs_relative": float(np.quantile(finite, 0.95)) if finite.size else None,
        "max_witness": max_witness,
    }


def scrtemp_positive_ion_population(state: np.ndarray, variables: list) -> np.ndarray:
    """Reproduce SUP_TO_FULL_V4's POPION sum over charged STEQ variables."""
    charged = [variable for variable in variables if variable.charge > 0]
    if not charged:
        raise ValueError("SCRTEMP contains no charged population equations")
    values = np.stack([state[variable.index - 1] for variable in charged])
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("invalid charged SCRTEMP populations for POPION")
    return values.sum(axis=0)


def parse_f_to_s_int_seq(
        path: Path, n_full: int
        ) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """기존 비교 도구 API. 생산 파서를 사용하되 과거 호출 형식을 유지한다."""
    from cmfgen_parser import parse_f_to_s_int_seq as parse
    return parse(path,n_full)


def parse_model_do_lev_dissolution(path: Path) -> bool:
    """Read CMFGEN MODEL's [DO_DIS] flag without guessing its default."""
    found = []
    with path.open(errors="replace") as handle:
        for line in handle:
            if "[DO_DIS]" not in line:
                continue
            token = line.split()[0].upper()
            if token not in {"T", "F", ".TRUE.", ".FALSE."}:
                raise ValueError(f"{path}: invalid [DO_DIS] value {token!r}")
            found.append(token in {"T", ".TRUE."})
    if len(found) != 1:
        raise ValueError(f"{path}: expected exactly one [DO_DIS] flag")
    return found[0]


def cmfgen_runtime_edge_nu15(ionization_energy_cm: float,
                             excitation_energy_cm: np.ndarray) -> np.ndarray:
    """Reproduce GENOSC_V9's runtime edge, not rounded osc_data nu15 text."""
    excitation = np.asarray(excitation_energy_cm, dtype=float)
    if (not math.isfinite(ionization_energy_cm) or ionization_energy_cm <= 0.0 or
            excitation.ndim != 1 or not np.isfinite(excitation).all()):
        raise ValueError("invalid GENOSC_V9 ionization/excitation energies")
    edge = (ionization_energy_cm - excitation) * C_CGS * 1.0e-15
    if not np.isfinite(edge).all() or np.any(edge <= 0.0):
        raise ValueError("GENOSC_V9 produced a non-positive runtime edge")
    return edge


def cmfgen_sup_to_full_v3b(
        superlevel_population: np.ndarray, lower_edge_nu15: np.ndarray,
        lower_g: np.ndarray, lower_f_to_s: np.ndarray, int_seq: np.ndarray,
        upper_edge_nu15: np.ndarray, upper_g: np.ndarray,
        upper_f_to_s: np.ndarray, upper_sl1_population: np.ndarray,
        temperature_1e4: np.ndarray, electron_density: np.ndarray,
        positive_ion_density: np.ndarray, ion_charge: float,
        do_lev_dissolution: bool,
        return_lte_full_on_super: bool = False
        ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Reproduce CMFGEN SUP_TO_FULL_V3b full-level distribution and closure."""
    superlevels = np.asarray(superlevel_population, dtype=float)
    lower_edge = np.asarray(lower_edge_nu15, dtype=float)
    lower_g = np.asarray(lower_g, dtype=float)
    lower_map = np.asarray(lower_f_to_s, dtype=int)
    sequence = np.asarray(int_seq, dtype=int)
    upper_edge = np.asarray(upper_edge_nu15, dtype=float)
    upper_g = np.asarray(upper_g, dtype=float)
    upper_map = np.asarray(upper_f_to_s, dtype=int)
    upper_sl1 = np.asarray(upper_sl1_population, dtype=float)
    temperature = np.asarray(temperature_1e4, dtype=float)
    electron_density = np.asarray(electron_density, dtype=float)
    positive_ion_density = np.asarray(positive_ion_density, dtype=float)
    if (superlevels.ndim != 2 or lower_edge.ndim != 1 or
            lower_g.shape != lower_edge.shape or lower_map.shape != lower_edge.shape or
            sequence.shape != lower_edge.shape or upper_edge.ndim != 1 or
            upper_g.shape != upper_edge.shape or upper_map.shape != upper_edge.shape):
        raise ValueError("SUP_TO_FULL input array dimensions disagree")
    nd, n_super = superlevels.shape
    if (nd == 0 or lower_edge.size == 0 or upper_edge.size == 0 or
            upper_sl1.shape != (nd,) or temperature.shape != (nd,) or
            electron_density.shape != (nd,) or positive_ion_density.shape != (nd,) or
            not math.isfinite(ion_charge) or ion_charge <= 0.0):
        raise ValueError("SUP_TO_FULL profile dimensions/ion charge are invalid")
    for values, name in ((superlevels, "superlevel populations"),
                         (lower_edge, "lower edge frequencies"),
                         (lower_g, "lower statistical weights"),
                         (upper_edge, "upper edge frequencies"),
                         (upper_g, "upper statistical weights"),
                         (upper_sl1, "upper SL1 population"),
                         (temperature, "temperature"),
                         (electron_density, "electron density"),
                         (positive_ion_density, "positive-ion density")):
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite SUP_TO_FULL {name}")
    if (np.any(superlevels < 0.0) or
            np.any(lower_g <= 0.0) or upper_edge[0] <= 0.0 or
            np.any(upper_g <= 0.0) or np.any(upper_sl1 <= 0.0) or
            np.any(temperature <= 0.0) or np.any(electron_density <= 0.0) or
            np.any(positive_ion_density <= 0.0)):
        raise ValueError("non-positive SUP_TO_FULL physical input")
    if (np.any(lower_map < 1) or np.any(lower_map > n_super) or
            set(np.unique(lower_map)) != set(range(1, n_super + 1)) or
            np.any(sequence < 0) or np.any(upper_map < 1) or
            not np.any(upper_map == 1)):
        raise ValueError("invalid SUP_TO_FULL superlevel/sequence mapping")

    # GION is the LTE partition function of the adjacent ion's ground SL.
    upper_ground = upper_map == 1
    partition_exponent = (
        (upper_edge[upper_ground, None] - upper_edge[0]) * CMFGEN_HDKT /
        temperature[None, :])
    partition = np.sum(upper_g[upper_ground, None] *
                       np.exp(partition_exponent), axis=0)
    if not np.isfinite(partition).all() or np.any(partition <= 0.0):
        raise ValueError("invalid adjacent-ion ground-SL partition function")

    if do_lev_dissolution:
        b_lev_dis = (8.589e14 * np.cbrt(positive_ion_density) /
                     electron_density) ** 1.5
        a_lev_dis = (0.0009 * electron_density ** 0.1667 /
                     np.sqrt(temperature))
        x_lev_dis = (1.0 + a_lev_dis) ** 3.15
    else:
        b_lev_dis = np.ones(nd, dtype=float)
        a_lev_dis = np.zeros(nd, dtype=float)
        x_lev_dis = np.ones(nd, dtype=float)
    occupation = np.ones((nd, lower_edge.size), dtype=float)
    if do_lev_dissolution:
        for level, edge in enumerate(lower_edge):
            t1 = (CMFGEN_LEVEL_DISSOLUTION_RYD_HZ15 * ion_charge**2 / edge
                  if edge != 0.0 else 0.0)
            neff = math.sqrt(t1) if 0.0 < t1 < 961.0 else 0.0
            if neff <= 2.01 * ion_charge:
                continue
            real_k = 16.0 * neff / (1.0 + neff)**2 / 3.0
            if neff <= 3.0:
                real_k = 1.0
            real_k = (real_k / ion_charge *
                      (ion_charge / neff)**4) ** 1.5
            y = 1.091 * (x_lev_dis + 4.0 * (ion_charge - 1.0) * a_lev_dis)
            beta = real_k * b_lev_dis
            f = y * beta**2 / (7.782 + x_lev_dis * beta)
            values = f / (1.0 + f)
            values[values > 0.99999999] = 1.0
            occupation[:, level] = values

    log_rhs = (math.log(CMFGEN_LTE_POPULATION_FACTOR) +
               np.log(electron_density) + np.log(upper_sl1) -
               1.5 * np.log(temperature) - np.log(partition))
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        log_lte_full = (
            np.log(occupation * lower_g[None, :]) +
            lower_edge[None, :] * (CMFGEN_HDKT / temperature[:, None]) +
            log_rhs[:, None])
    if not np.isfinite(log_lte_full).all():
        raise ValueError("invalid CMFGEN full-level LTE populations")
    log_lte_super = np.empty((nd, n_super), dtype=float)
    lte_full_on_super = np.empty_like(log_lte_full)
    for sl in range(1, n_super + 1):
        members = np.flatnonzero(lower_map == sl)
        log_lte_super[:, sl - 1] = np.logaddexp.reduce(
            log_lte_full[:, members], axis=1)
        lte_full_on_super[:, members] = np.exp(
            log_lte_full[:, members] - log_lte_super[:, sl - 1, None])
    lte_full = np.zeros_like(log_lte_full)
    full_under_limit = log_lte_full < 600.0
    lte_full[full_under_limit] = np.exp(log_lte_full[full_under_limit])
    lte_super = np.zeros_like(log_lte_super)
    super_under_limit = log_lte_super < 600.0
    lte_super[super_under_limit] = np.exp(log_lte_super[super_under_limit])
    if not np.isfinite(lte_full_on_super).all() or np.any(lte_super <= 0.0):
        raise ValueError("invalid grouped CMFGEN LTE populations")
    edge_super = np.empty((nd, n_super), dtype=float)
    member_count = np.empty(n_super, dtype=int)
    for sl in range(1, n_super + 1):
        members = np.flatnonzero(lower_map == sl)
        member_count[sl - 1] = members.size
        edge_super[:, sl - 1] = np.sum(
            lower_edge[members][None, :] * lte_full_on_super[:, members], axis=1)

    full_population = np.zeros((nd, lower_edge.size), dtype=float)
    for depth in range(nd):
        for level, sl_one_based in enumerate(lower_map):
            sl = int(sl_one_based) - 1
            if member_count[sl] == 1 or sequence[level] == 0:
                full_population[depth, level] = (
                    superlevels[depth, sl] * lte_full_on_super[depth, level])
                continue
            match = -1
            if lower_edge[level] <= edge_super[depth, sl]:
                candidate_ranges = (
                    range(level + 1, lower_edge.size),
                    range(level - 1, -1, -1),
                )
            else:
                candidate_ranges = (range(level - 1, -1, -1),)
            for candidates in candidate_ranges:
                for other in candidates:
                    if (sequence[other] == sequence[level] and
                            lower_map[other] != lower_map[level]):
                        match = other
                        break
                if match >= 0:
                    break
            if match < 0:
                full_population[depth, level] = (
                    superlevels[depth, sl] * lte_full_on_super[depth, level])
                continue
            match_sl = int(lower_map[match]) - 1
            numerator = edge_super[depth, sl] / lower_edge[level]
            denominator = edge_super[depth, sl] / edge_super[depth, match_sl]
            if numerator <= 0.0 or denominator <= 0.0 or denominator == 1.0:
                raise ValueError("invalid SUP_TO_FULL interpolation energy bracket")
            t1 = math.log(numerator) / math.log(denominator)
            b1 = superlevels[depth, match_sl] / lte_super[depth, match_sl]
            b2 = superlevels[depth, sl] / lte_super[depth, sl]
            departure = t1 * b1 + (1.0 - t1) * b2
            if b1 <= 1.0 and b2 <= 1.0 and departure > 1.0:
                departure = 1.0
            if b1 >= 1.0 and b2 >= 1.0 and departure < 1.0:
                departure = 1.0
            if departure < 0.0:
                departure = b1
            full_population[depth, level] = lte_full[depth, level] * departure

        for sl in range(1, n_super + 1):
            members = np.flatnonzero(lower_map == sl)
            total = float(full_population[depth, members].sum())
            target = float(superlevels[depth, sl - 1])
            if total <= 0.0:
                if target > 0.0:
                    raise ValueError(
                        f"CMFGEN full-level distribution has zero sum in SL{sl}")
                full_population[depth, members] = 0.0
            else:
                full_population[depth, members] *= target / total
    if (not np.isfinite(full_population).all() or
            np.any(full_population < 0.0)):
        raise ValueError("invalid reconstructed CMFGEN full-level populations")
    if return_lte_full_on_super:
        return full_population, lte_full_on_super
    return full_population


def lte_weighted_superlevel_photo_rate_density(
        superlevel_population: np.ndarray, lte_full_on_super: np.ndarray,
        full_level_superlevels: np.ndarray, full_level_gamma: np.ndarray
        ) -> np.ndarray:
    """Apply CMFGEN WSE's LTE-within-superlevel weighting to Lumina rates.

    This is a forensic kernel/weighting diagnostic, not the primary rate
    acceptance metric. CMFGEN QUAD_MULTI_V10 builds WSE from HNST_F_ON_S,
    then multiplies WSE by the current superlevel population.
    """
    superlevels = np.asarray(superlevel_population, dtype=float)
    fractions = np.asarray(lte_full_on_super, dtype=float)
    sl_map = np.asarray(full_level_superlevels, dtype=int)
    gamma = np.asarray(full_level_gamma, dtype=float)
    if (superlevels.ndim != 2 or fractions.ndim != 2 or gamma.ndim != 2 or
            fractions.shape[0] != superlevels.shape[0] or
            gamma.shape != (fractions.shape[1], superlevels.shape[0]) or
            sl_map.shape != (fractions.shape[1],) or
            np.any(sl_map < 1) or np.any(sl_map > superlevels.shape[1]) or
            set(np.unique(sl_map)) != set(range(1, superlevels.shape[1] + 1)) or
            not np.isfinite(superlevels).all() or
            not np.isfinite(fractions).all() or not np.isfinite(gamma).all() or
            np.any(superlevels < 0.0) or np.any(fractions < 0.0) or
            np.any(gamma < 0.0)):
        raise ValueError("invalid LTE-weighted superlevel photo-rate inputs")
    for sl in range(1, superlevels.shape[1] + 1):
        weight_sum = fractions[:, sl_map == sl].sum(axis=1)
        if not np.allclose(weight_sum, 1.0, rtol=2e-12, atol=2e-15):
            raise ValueError(f"LTE-on-super fractions do not normalize for SL{sl}")
    rate_density = np.zeros((superlevels.shape[1], superlevels.shape[0]),
                            dtype=float)
    for level, sl in enumerate(sl_map):
        rate_density[sl - 1] += (
            superlevels[:, sl - 1] * fractions[:, level] * gamma[level])
    if not np.isfinite(rate_density).all() or np.any(rate_density < 0.0):
        raise ValueError("invalid LTE-weighted superlevel photo-rate density")
    return rate_density


def raw_field_step_sigma_gamma(
        raw_nu: np.ndarray, raw_j_nu: np.ndarray,
        field_bin_edges: np.ndarray, field_states: np.ndarray,
        sigma_bin_edges: np.ndarray, sigma_bin_average: np.ndarray,
        nu_threshold: float) -> float:
    """Integrate the production step-sigma model against raw EDDFACTOR J(nu).

    This single-cell forensic replay removes only the conservative field-bin
    averaging. It retains the baked sigma bins, threshold-bin mass convention,
    field validity states, and the same linear interpolation between native J
    samples. It is not a separate physical acceptance calculation.
    """
    nu = np.asarray(raw_nu, dtype=float)
    j_nu = np.asarray(raw_j_nu, dtype=float)
    j_edges = np.asarray(field_bin_edges, dtype=float)
    states = np.asarray(field_states, dtype=int)
    sigma_edges = np.asarray(sigma_bin_edges, dtype=float)
    sigma = np.asarray(sigma_bin_average, dtype=float)
    if (nu.ndim != 1 or nu.size < 2 or j_nu.shape != nu.shape or
            j_edges.ndim != 1 or j_edges.size < 2 or
            states.shape != (j_edges.size - 1,) or
            sigma_edges.ndim != 1 or sigma_edges.size < 2 or
            sigma.shape != (sigma_edges.size - 1,) or
            not math.isfinite(nu_threshold) or nu_threshold <= 0.0 or
            not np.isfinite(nu).all() or not np.isfinite(j_nu).all() or
            not np.isfinite(j_edges).all() or not np.isfinite(sigma_edges).all() or
            not np.isfinite(sigma).all() or np.any(j_nu < 0.0) or
            np.any(sigma < 0.0) or np.any(np.diff(nu) <= 0.0) or
            np.any(np.diff(j_edges) <= 0.0) or
            np.any(np.diff(sigma_edges) <= 0.0)):
        raise ValueError("invalid raw-field step-sigma integration inputs")

    def integrate_linear_j_over_nu(lo: float, hi: float) -> float:
        if hi <= lo:
            return 0.0
        if lo < nu[0] or hi > nu[-1]:
            raise ValueError("valid field bin extends outside raw EDDFACTOR support")
        total = 0.0
        index = max(int(np.searchsorted(nu, lo, side="right")) - 1, 0)
        while index < nu.size - 1 and nu[index] < hi:
            x0 = max(lo, float(nu[index]))
            x1 = min(hi, float(nu[index + 1]))
            if x1 > x0:
                slope = ((j_nu[index + 1] - j_nu[index]) /
                         (nu[index + 1] - nu[index]))
                y0 = j_nu[index] + slope * (x0 - nu[index])
                y1 = j_nu[index] + slope * (x1 - nu[index])
                local_slope = (y1 - y0) / (x1 - x0)
                intercept = y0 - local_slope * x0
                total += (intercept * math.log(x1 / x0) +
                          local_slope * (x1 - x0))
            index += 1
        return total

    integral = 0.0
    for sigma_index, bin_sigma in enumerate(sigma):
        if bin_sigma <= 0.0:
            continue
        sigma_lo = float(sigma_edges[sigma_index])
        sigma_hi = float(sigma_edges[sigma_index + 1])
        active_lo = max(sigma_lo, nu_threshold)
        if sigma_hi <= active_lo:
            continue
        active_sigma = float(bin_sigma)
        if sigma_lo < nu_threshold:
            active_sigma *= (sigma_hi - sigma_lo) / (sigma_hi - active_lo)
        first_field_bin = max(
            int(np.searchsorted(j_edges, active_lo, side="right")) - 1, 0)
        last_field_bin = min(
            int(np.searchsorted(j_edges, sigma_hi, side="left")),
            states.size - 1)
        for field_bin in range(first_field_bin, last_field_bin + 1):
            if states[field_bin] != 1:  # VALID; exact-zero and missing bins add zero.
                continue
            lo = max(active_lo, float(j_edges[field_bin]))
            hi = min(sigma_hi, float(j_edges[field_bin + 1]))
            if hi > lo:
                integral += active_sigma * integrate_linear_j_over_nu(lo, hi)
    return 4.0 * math.pi * integral / 6.62607015e-27


def raw_field_continuous_sigma_gamma(
        raw_nu: np.ndarray, raw_j_nu: np.ndarray,
        field_bin_edges: np.ndarray, field_states: np.ndarray,
        sigma_terms: list[tuple[float, Callable[[np.ndarray], np.ndarray],
                                np.ndarray, float]]) -> float:
    """Integrate exact PHOT evaluator terms against raw J on native nodes.

    Each term is (route_probability, sigma(nu), sigma_structure_nodes,
    sigma_support_start). The union of field, sigma, and J nodes is integrated
    with 8-point Gauss-Legendre on every piece, retaining only valid field bins.
    """
    nu = np.asarray(raw_nu, dtype=float)
    j_nu = np.asarray(raw_j_nu, dtype=float)
    edges = np.asarray(field_bin_edges, dtype=float)
    states = np.asarray(field_states, dtype=int)
    if (nu.ndim != 1 or nu.size < 2 or j_nu.shape != nu.shape or
            edges.ndim != 1 or edges.size < 2 or
            states.shape != (edges.size - 1,) or
            not np.isfinite(nu).all() or not np.isfinite(j_nu).all() or
            not np.isfinite(edges).all() or np.any(j_nu < 0.0) or
            np.any(np.diff(nu) <= 0.0) or np.any(np.diff(edges) <= 0.0)):
        raise ValueError("invalid raw-field continuous-sigma integration inputs")
    quad_nodes, quad_weights = np.polynomial.legendre.leggauss(8)
    integral = 0.0
    for probability, sigma_fn, sigma_nodes, sigma_start in sigma_terms:
        knots = np.asarray(sigma_nodes, dtype=float)
        if (not math.isfinite(probability) or probability < 0.0 or
                not math.isfinite(sigma_start) or sigma_start <= 0.0 or
                knots.ndim != 1 or not np.isfinite(knots).all()):
            raise ValueError("invalid raw PHOT sigma term")
        lo = max(float(nu[0]), float(edges[0]), sigma_start)
        hi = min(float(nu[-1]), float(edges[-1]))
        if hi <= lo or probability == 0.0:
            continue
        parts = [np.array([lo, hi]),
                 nu[(nu > lo) & (nu < hi)],
                 edges[(edges > lo) & (edges < hi)],
                 knots[(knots > lo) & (knots < hi)]]
        breaks = np.unique(np.concatenate(parts))
        left, right = breaks[:-1], breaks[1:]
        mid = 0.5 * (left + right)
        field_index = np.searchsorted(edges, mid, side="right") - 1
        valid = ((field_index >= 0) & (field_index < states.size))
        valid_indices = np.flatnonzero(valid)
        valid[valid_indices] = states[field_index[valid_indices]] == 1
        if not np.any(valid):
            continue
        left = left[valid]
        right = right[valid]
        half = 0.5 * (right - left)
        center = 0.5 * (right + left)
        sample_nu = center[:, None] + half[:, None] * quad_nodes[None, :]
        sample_sigma = np.asarray(sigma_fn(sample_nu), dtype=float)
        if (sample_sigma.shape != sample_nu.shape or
                not np.isfinite(sample_sigma).all() or
                np.any(sample_sigma < 0.0)):
            raise ValueError("CMFGEN PHOT evaluator returned invalid sigma")
        sample_j = np.interp(sample_nu.ravel(), nu, j_nu).reshape(sample_nu.shape)
        piece_integrals = half * np.sum(
            quad_weights[None, :] * sample_sigma * sample_j / sample_nu, axis=1)
        integral += probability * float(np.sum(piece_integrals))
    return 4.0 * math.pi * integral / 6.62607015e-27


def cmfgen_raw_sigma_terms_for_level(
        phot_source: Path, channels: dict, global_level_index: int,
        lower_config: str, lower_energy_eV: float, ionization_eV: float,
        atomic_number: int, z_screen: float,
        phot_smoother: PhotSmoother | None = None) -> list:
    """Rebuild exact linked PHOT evaluators for one sealed lower full level."""
    paths = atomic._phot_route_paths(phot_source)
    phot_grids = [(phot_smoother.read(path) if phot_smoother else atomic.parse_phot(path))
                 for path in paths]
    if not 0 <= global_level_index < channels["n_levels"]:
        raise ValueError("raw PHOT level index is outside channel table")
    first = int(channels["offsets"][global_level_index])
    last = int(channels["offsets"][global_level_index + 1])
    if first == last:
        raise ValueError("raw PHOT witness has no sealed channel routes")
    config = atomic._norm_cfg(lower_config)
    term = atomic._term_cfg(config)
    sigma_terms = []
    for route in range(first, last):
        grid_id = int(channels["phot_grid"][route])
        if grid_id >= len(phot_grids):
            raise ValueError("sealed PHOT grid index exceeds linked PHOT sources")
        phot = phot_grids[grid_id]
        if phot.split_J:
            entries = [entry for entry in phot.entries
                       if atomic._norm_cfg(entry.config) == config]
        else:
            entries = [entry for entry in phot.entries
                       if atomic._term_cfg(entry.config) == term]
        if not entries:
            raise ValueError(
                f"linked PHOT grid {grid_id} has no cross section for {lower_config!r}")
        threshold_eV = (
            ionization_eV - lower_energy_eV +
            float(channels["target_energy_eV"][route]))
        if threshold_eV <= 0.0:
            continue
        # SUB_PHOT_GEN uses (GS_EDGE + EXC_FREQ)/EQUAL_COR_FAC.
        threshold_nu = (threshold_eV * atomic.EV_TO_ERG / atomic.H_CGS /
                        (1.0 + 1.0e-14))
        probability = float(channels["probability"][route])
        for entry in entries:
            model = atomic._route_sigma_model(
                entry, threshold_nu, atomic_number, z_screen)
            sigma_fn, sigma_nodes, sigma_start, _tag = model
            sigma_terms.append((probability, sigma_fn,
                                np.asarray(sigma_nodes, dtype=float),
                                float(sigma_start)))
    if not sigma_terms:
        raise ValueError("raw PHOT witness produced no active sigma evaluators")
    return sigma_terms


def cmfgen_get_num_phot_routes_source_components(
        final_state: str, upper_level_names: list[str], upper_g: np.ndarray,
        upper_energy_eV: np.ndarray, upper_f_to_s: np.ndarray) -> list[dict]:
    """Reproduce GET_NUM_PHOT_ROUTES_V1's source-level route enumeration."""
    weights = np.asarray(upper_g, dtype=float)
    energies = np.asarray(upper_energy_eV, dtype=float)
    superlevels = np.asarray(upper_f_to_s, dtype=int)
    if (weights.ndim != 1 or energies.shape != weights.shape or
            superlevels.shape != weights.shape or len(upper_level_names) != weights.size or
            not np.isfinite(weights).all() or not np.isfinite(energies).all() or
            np.any(weights <= 0.0) or np.any(superlevels < 1)):
        raise ValueError("invalid CMFGEN GET_NUM_PHOT_ROUTES inputs")
    selected = []
    for alias in (part.strip() for part in final_state.split("/")):
        matches = []
        for index, name in enumerate(upper_level_names):
            before_j = str(name).split("[", 1)[0].strip()
            if before_j == alias:
                matches.append(index)
        if matches:
            selected = matches
            break
    if not selected:
        raise ValueError(f"CMFGEN route final state {final_state!r} did not resolve")

    route_weights = []
    routes = []
    found = False
    f_to_s_save = 0
    for index in selected:
        route_weights.append(float(weights[index]))
        if found:
            if int(superlevels[index]) != f_to_s_save:
                routes.append({
                    "anchor_level_1based": index + 1,
                    "target_superlevel_1based": int(superlevels[index]),
                    "target_energy_eV": float(energies[index]),
                    "statistical_weight": float(weights[index]),
                })
            else:
                routes[-1]["statistical_weight"] += float(weights[index])
        else:
            found = True
            f_to_s_save = int(superlevels[index])
            routes.append({
                "anchor_level_1based": index + 1,
                "target_superlevel_1based": int(superlevels[index]),
                "target_energy_eV": float(energies[index]),
                "statistical_weight": float(weights[index]),
            })
    gsum = float(sum(route_weights))
    for route in routes:
        route["probability"] = route["statistical_weight"] / gsum
    return routes


def element_population_closure(pop_species: np.ndarray,
                               active_ion_populations: list[np.ndarray],
                               terminal_population: np.ndarray) -> dict:
    """Close active full-level populations plus one terminal reservoir to POP_SPECIES."""
    total = np.asarray(pop_species, dtype=float)
    if total.ndim != 1 or not np.isfinite(total).all() or np.any(total <= 0.0):
        raise ValueError("invalid POP_SPECIES abundance vector")
    represented = np.zeros_like(total)
    if not active_ion_populations:
        raise ValueError("element closure requires at least one active ion")
    for populations in active_ion_populations:
        levels = np.asarray(populations, dtype=float)
        if levels.ndim != 2 or levels.shape[0] != total.size or not np.isfinite(levels).all() or np.any(levels < 0.0):
            raise ValueError("invalid active-ion full-level population array")
        represented += levels.sum(axis=1)
    terminal = np.asarray(terminal_population, dtype=float)
    if terminal.shape != total.shape or not np.isfinite(terminal).all() or np.any(terminal < 0.0):
        raise ValueError("invalid terminal population array")
    relative = (represented + terminal) / total - 1.0
    absolute = np.abs(relative)
    return {
        "max_abs_relative_residual": float(absolute.max()),
        "p95_abs_relative_residual": float(np.quantile(absolute, 0.95)),
        "tolerance_rel": ELEMENT_POP_CLOSURE_TOL_REL,
        "pass": bool(absolute.max() <= ELEMENT_POP_CLOSURE_TOL_REL),
    }


def total_photoionization_rate(rate_density: np.ndarray,
                               current_ion_density: np.ndarray) -> np.ndarray:
    """Normalize a photoionization rate density by the *same current ion* population."""
    rates = np.asarray(rate_density, dtype=float)
    density = np.asarray(current_ion_density, dtype=float)
    if rates.ndim != 2 or density.ndim != 1 or rates.shape[1] != density.size:
        raise ValueError("photoionization rate/population dimensions disagree")
    if not np.isfinite(rates).all() or np.any(rates < 0.0) or \
            not np.isfinite(density).all() or np.any(density <= 0.0):
        raise ValueError("invalid photoionization rate density or current-ion density")
    return rates.sum(axis=0) / density


def level_energies_and_thresholds(levels: np.ndarray, ionization_eV: float,
                                  n_full: int, ion_label: str
                                  ) -> tuple[np.ndarray, np.ndarray]:
    """Return excitation energies and CMFGEN photoionization thresholds.

    In modern CMFGEN OSC rows, E(cm) is excitation energy while the `eV`
    column is already the binding/photoionization threshold E_ion-E_level.
    The latter must not be reconstructed by subtracting it from E_ion.
    """
    require(len(levels) >= n_full, f"{ion_label}: OSC levels shorter than NF")
    excitation_eV = np.asarray(levels["E_cm"][:n_full], dtype=float) * CM_TO_EV
    threshold_eV = np.asarray(levels["E_eV"][:n_full], dtype=float)
    require(np.isfinite(excitation_eV).all() and np.isfinite(threshold_eV).all() and
            math.isfinite(ionization_eV),
            f"{ion_label}: non-finite OSC energy/threshold")
    expected_threshold = ionization_eV - excitation_eV
    max_residual = float(np.max(np.abs(threshold_eV - expected_threshold)))
    require(max_residual <= 2.0e-3,
            f"{ion_label}: OSC threshold inconsistent with E_ion-E_level; "
            f"max residual={max_residual:.6g} eV")
    threshold_hz = np.where(threshold_eV > 0.0,
                            threshold_eV * l1bf.EV / l1bf.H, 0.0)
    return excitation_eV, threshold_hz


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmfgen-run", type=Path, required=True)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run, deck, fixture = (p.resolve() for p in (args.cmfgen_run, args.deck, args.fixture))
    out = args.out.resolve()
    require(fixture.is_file() and os.access(fixture, os.X_OK),
            f"fixture must be executable: {fixture}")
    require(run.is_dir() and deck.is_dir(), "CMFGEN run or Lumina deck missing")
    require(not out.exists(), f"refusing existing output directory: {out}")

    stamp, identity, loaded = verify_sealed_deck(run, deck)
    model_map, link_map, runtime_map = (
        loaded["model"], loaded["links"], loaded["runtime"])
    target_keys = set(TARGETS)
    require(target_keys.issubset(model_map), "one or more Si/S target ions absent from MODEL_SPEC")

    field, nu, nd, finish = read_eddfactor(str(run / "EDDFACTOR"))
    require(finish == 1.0, "CMFGEN EDDFACTOR is not finished")
    require(field.shape == (len(nu), nd) and np.isfinite(field).all() and
            np.all(field >= 0.0), "invalid CMFGEN EDDFACTOR array")
    _, rvtj_electron_density, rvtj_temperature = read_rvtj(str(run / "RVTJ"))
    require(rvtj_electron_density.shape == (nd,) and
            rvtj_temperature.shape == (nd,) and
            np.isfinite(rvtj_electron_density).all() and
            np.all(rvtj_electron_density > 0.0) and
            np.isfinite(rvtj_temperature).all() and
            np.all(rvtj_temperature > 0.0),
            "RVTJ electron-density/temperature dimension/value mismatch")

    scrtemp_nd, scrtemp_nt = read_dimensions(run)
    require(scrtemp_nd == nd and scrtemp_nt >= 3,
            "MODEL/SCRTEMP dimensions disagree with EDDFACTOR")
    scrtemp_variables = read_variables(run, scrtemp_nt - 2)
    do_lev_dissolution = parse_model_do_lev_dissolution(run / "MODEL")
    scrtemp_variable_indices = {}
    for variable in scrtemp_variables:
        equation = (variable.ion, variable.superlevel)
        require(equation not in scrtemp_variable_indices,
                f"duplicate SCRTEMP population equation {equation}")
        scrtemp_variable_indices[equation] = variable.index - 1

    prrr_profiles = {}
    for Z, stage in sorted(target_keys):
        label = str(model_map[(Z, stage)]["label"])
        next_key = (Z, stage + 1)
        require(next_key in model_map,
                f"{label}: pre-solve state matching requires adjacent ion in MODEL_SPEC")
        prrr_path = run / f"{label}PRRR"
        summary = parse_prrr_summary(prrr_path, label, nd,
                                     int(model_map[(Z, stage)]["n_super"]))
        prrr_profiles[label] = {
            "next_ion": str(model_map[next_key]["label"]),
            "ion_density": np.asarray(summary["ion_density"], dtype=float),
            "electron_density": np.asarray(summary["electron_density"], dtype=float),
            "temperature": parse_prrr_temperature(prrr_path, nd),
        }
    scrtemp_reader = ScrtempReader(run / "SCRTEMP", nd, scrtemp_nt)
    matched_scrtemp_record, matched_scrtemp_state, matched_scrtemp_metrics = (
        match_prrr_scrtemp_state(
            scrtemp_reader, scrtemp_variable_indices, prrr_profiles,
            scrtemp_nt - 2, scrtemp_nt - 1))
    matched_temperature_1e4 = matched_scrtemp_state[scrtemp_nt - 1]
    matched_temperature = (matched_temperature_1e4 *
                           SCRTEMP_TEMPERATURE_SCALE_K)
    matched_electron_density = matched_scrtemp_state[scrtemp_nt - 2]
    matched_positive_ion_density = scrtemp_positive_ion_population(
        matched_scrtemp_state, scrtemp_variables)

    edges = l1bf.canonical_edges()
    j_rows = np.empty((nd, l1bf.CANON_N_BINS), dtype=float)
    states = np.empty((nd, l1bf.CANON_N_BINS), dtype=np.int32)
    for depth in range(nd):
        j_rows[depth], states[depth] = l1bf.conservative_rebin(
            nu, field[:, depth], edges)

    bake_path = deck / "cmfgen_sigma_bf.bin"
    baked = l1bf.read_bake_bin(str(bake_path))
    require(baked["magic"] == 0x434D4644 and baked["ver"] == 1 and
            baked["size_ok"] and baked["nbin"] == l1bf.BF_N_BINS and
            baked["numin"] == l1bf.BF_NU_MIN and baked["numax"] == l1bf.BF_NU_MAX,
            "Lumina BF sigma binary contract mismatch")
    channels = parse_cmfgen_bf_channels(
        deck / "cmfgen_bf_channels.bin", int(baked["nlev"]),
        int(baked["nbin"]), loaded["route_manifest"])
    channel_edges = channels["nu_min"] * np.exp(
        np.arange(channels["n_bins"] + 1) *
        (math.log(channels["nu_max"] / channels["nu_min"]) /
         channels["n_bins"]))
    j_milne_rows = np.empty((nd, channels["n_bins"]), dtype=float)
    states_milne = np.empty((nd, channels["n_bins"]), dtype=np.int32)
    for depth in range(nd):
        j_milne_rows[depth], states_milne[depth] = l1bf.conservative_rebin(
            nu, field[:, depth], channel_edges)
    with (deck / "levels.csv").open(newline="") as stream:
        all_level_rows = list(csv.DictReader(stream))
    require(len(all_level_rows) == baked["nlev"],
            "levels.csv and sigma binary level counts differ")
    level_index = {}
    for index, row in enumerate(all_level_rows):
        csv_level = int(row["level_number"])
        require(csv_level >= 0, f"negative levels.csv level_number: {csv_level}")
        # Lumina CSV level_number is zero-based; CMFGEN OSC/F_TO_S IDs are 1-based.
        key = (int(row["atomic_number"]), int(row["ion_number"]) + 1,
               csv_level + 1)
        if key in level_index:
            raise ValueError(f"duplicate deck level key {key}")
        level_index[key] = index
    require(len(level_index) == int(baked["nlev"]),
            "levels.csv key count differs from sealed sigma/channel binaries")

    pop_cache = {}
    element_closure = {}
    for Z, pop_file in ((14, "POPSIL"), (16, "POPSUL")):
        if pop_file not in pop_cache:
            pop_cache[pop_file] = parse_popcob(str(run / pop_file))
        pop_species, populations_by_ion, _, pop_nd = pop_cache[pop_file]
        require(pop_nd == nd, f"{pop_file}: POP depth count mismatch")
        active_stages = sorted(stage for (z, stage) in model_map if z == Z)
        require(active_stages and
                active_stages == list(range(active_stages[0], active_stages[-1] + 1)),
                f"{atomic.SYM[Z]}: active stages are not contiguous")
        active_labels = [str(model_map[(Z, stage)]["label"])
                         for stage in active_stages]
        require(all(label in populations_by_ion for label in active_labels),
                f"{pop_file}: an active MODEL_SPEC ion is absent from POP output")
        top_label = active_labels[-1]
        element_closure[atomic.SYM[Z]] = element_population_closure(
            np.asarray(pop_species, dtype=float),
            [populations_by_ion[label][0] for label in active_labels],
            populations_by_ion[top_label][1])

    ions = []
    route_expansion_by_ion = {}
    final_superlevel_by_ion = {}
    sigma_blocks, threshold_blocks = [], []
    offsets = [0]
    snapshot_rows = []
    snapshot_by_ion = {}
    for key in sorted(target_keys):
        Z, stage = key
        spec = model_map[key]
        label = str(spec["label"])
        n_full, n_super = int(spec["n_full"]), int(spec["n_super"])
        pop_file = TARGETS[key]
        if pop_file not in pop_cache:
            pop_cache[pop_file] = parse_popcob(str(run / pop_file))
        pop_species, populations_by_ion, _, pop_nd = pop_cache[pop_file]
        require(pop_nd == nd and label in populations_by_ion,
                f"{label}: POP depth or ion label mismatch")
        populations = np.asarray(populations_by_ion[label][0], dtype=float)
        require(populations.shape == (nd, n_full),
                f"{label}: POP shape {populations.shape} != ({nd},{n_full})")
        require(np.isfinite(populations).all() and np.all(populations >= 0),
                f"{label}: invalid FL populations")
        sources = link_map[key]
        osc = parse_osc(sources["osc"])
        require(osc.n_levels >= n_full, f"{label}: oscillator levels shorter than MODEL_SPEC NF")
        sl_map, sl_names = parse_f_to_s(str(sources["f_to_s"]), osc.n_levels)
        require(len(sl_map) >= n_full and len(sl_names) >= n_full,
                f"{label}: F_TO_S shorter than MODEL_SPEC NF")
        sl_map = np.asarray(sl_map[:n_full], dtype=int)
        f_to_s_map, f_to_s_names, int_seq, f_to_s_edge_nu15 = (
            parse_f_to_s_int_seq(sources["f_to_s"], n_full))
        require(np.array_equal(f_to_s_map, sl_map) and
                f_to_s_names == sl_names[:n_full],
                f"{label}: strict F_TO_S/INT_SEQ rows disagree with runtime parser")
        osc_data_edge_nu15 = np.asarray(osc.levels["nu15"][:n_full], dtype=float)
        f_to_s_edge_rel = np.abs(f_to_s_edge_nu15 - osc_data_edge_nu15) / np.maximum(
            np.abs(osc_data_edge_nu15), 1.0)
        require(np.isfinite(osc_data_edge_nu15).all() and
                float(np.max(f_to_s_edge_rel)) <= 2.0e-4,
                f"{label}: F_TO_S/OSC edge frequency mismatch")
        # GENOSC_V9 ignores the osc_data 10^15-Hz field for FEDGE. It uses
        # (IONIZATION_ENERGY-E(cm^-1))*SPEED_OF_LIGHT*1e-15 instead.
        lower_edge_nu15 = cmfgen_runtime_edge_nu15(
            float(osc.ionization_eV) / CM_TO_EV,
            np.asarray(osc.levels["E_cm"][:n_full], dtype=float))
        require(set(np.unique(sl_map)) == set(range(1, n_super + 1)),
                f"{label}: active F_TO_S prefix does not cover MODEL_SPEC N_SL")
        final_superlevel_by_ion[label] = sum_full_level_populations_by_superlevel(
            populations, sl_map, n_super)
        for level in range(1, n_full + 1):
            runtime = runtime_map[(label, level)]
            osc_config = " ".join(str(osc.levels["config"][level - 1]).split()).lower()
            map_config = " ".join(str(runtime["configuration"]).split()).lower()
            require(osc_config == map_config,
                    f"{label} FL {level}: oscillator/configuration map mismatch")
            require(int(sl_map[level - 1]) == int(runtime["superlevel"]),
                    f"{label} FL {level}: F_TO_S/runtime SL mismatch")
            row_key = (Z, stage, level)
            require(row_key in level_index, f"{label} FL {level}: level absent from deck")
            deck_row = all_level_rows[level_index[row_key]]
            # Lumina levels.csv stores zero-based SL IDs; CMFGEN F_TO_S and
            # LEVEL_SL_STEQ_LINKS store the same identity one-based.
            require(int(deck_row["super_level"]) + 1 == int(sl_map[level - 1]),
                    f"{label} FL {level}: deck/CMFGEN SL identity mismatch")
            level_energy_eV = float(osc.levels["E_cm"][level - 1]) * CM_TO_EV
            require(abs(float(deck_row["energy_eV"]) - level_energy_eV) <= 1e-9,
                    f"{label} FL {level}: deck energy mismatch")
            require(math.isclose(float(deck_row["g"]),
                                 float(osc.levels["g"][level - 1]),
                                 rel_tol=0.0, abs_tol=1e-12),
                    f"{label} FL {level}: deck statistical weight mismatch")
        indices = [level_index[(Z, stage, level)] for level in range(1, n_full + 1)]
        sigma = np.asarray(baked["sigma"][indices], dtype=float)
        level_energy_eV, nu_threshold = level_energies_and_thresholds(
            osc.levels, float(osc.ionization_eV), n_full, label)
        require(np.isfinite(sigma).all() and np.all(sigma >= 0),
                f"{label}: invalid baked cross sections")
        # CMFGEN can retain superlevel members at/above the continuum edge;
        # the route baker correctly gives them no bound-free cross section.
        prrr_path = run / f"{label}PRRR"
        cmfgen_rate_density = parse_prrr_superlevel_rates(
            prrr_path, label, nd, n_super)
        cmfgen_recombination_density = parse_prrr_superlevel_rates(
            prrr_path, label, nd, n_super, process="recombination")
        prrr_summary = prrr_profiles[label]
        prrr_temperature = prrr_summary["temperature"]
        # PRRR's printed Ion Density is DXzV, mapped to the next ion's SL1
        # equation. It is neither this ion's total nor next ion's one full level.
        prrr_next_sl1 = np.asarray(prrr_summary["ion_density"], dtype=float)
        cmfgen_electron_density = prrr_summary["electron_density"]
        require(np.isfinite(prrr_next_sl1).all() and
                np.all(prrr_next_sl1 > 0.0) and
                np.isfinite(cmfgen_electron_density).all() and
                np.all(cmfgen_electron_density > 0.0),
                f"{label}: invalid PRRR density arrays")
        population_density = populations.sum(axis=1)
        next_key = (Z, stage + 1)
        require(next_key in model_map,
                f"{label}: v4 route-rate comparison requires the adjacent ion in MODEL_SPEC")
        next_spec = model_map[next_key]
        next_label = str(next_spec["label"])
        upper_n_full, upper_n_super = (int(next_spec["n_full"]),
                                       int(next_spec["n_super"]))
        upper_population = np.asarray(populations_by_ion[next_label][0], dtype=float)
        require(upper_population.shape == (nd, upper_n_full) and
                np.isfinite(upper_population).all() and
                np.all(upper_population >= 0.0),
                f"{next_label}: invalid adjacent-ion full-level populations")
        upper_sl_map = np.asarray([
            int(runtime_map[(next_label, level)]["superlevel"])
            for level in range(1, upper_n_full + 1)
        ], dtype=int)
        upper_population_by_sl = sum_full_level_populations_by_superlevel(
            upper_population, upper_sl_map, upper_n_super)
        next_sl1 = prrr_next_stage_sl1_population(
            populations_by_ion, label, next_label, upper_population_by_sl)
        require(next_sl1.shape == (nd,) and np.isfinite(next_sl1).all() and
                np.all(next_sl1 > 0.0),
                f"{label}: invalid next-stage SL1 population")
        next_sl1_delta = prrr_next_sl1 / next_sl1 - 1.0
        prrr_rvtj_ne_delta = cmfgen_electron_density / rvtj_electron_density - 1.0
        prrr_rvtj_temperature_delta = prrr_temperature / rvtj_temperature - 1.0
        next_sl1_max = float(np.max(np.abs(next_sl1_delta)))
        prrr_rvtj_ne_max = float(np.max(np.abs(prrr_rvtj_ne_delta)))
        prrr_rvtj_temperature_max = float(
            np.max(np.abs(prrr_rvtj_temperature_delta)))
        matched_errors = matched_scrtemp_metrics[label]
        same_state = (
            matched_errors["next_SL1_max_rel"] <= PRRR_NEXT_SL1_TOL_REL and
            matched_errors["electron_density_max_rel"] <= PRRR_RVTJ_ELECTRON_TOL_REL and
            matched_errors["temperature_max_rel"] <= PRRR_RVTJ_TEMPERATURE_TOL_REL
        )

        upper_sources = link_map[next_key]
        upper_osc = parse_osc(upper_sources["osc"])
        require(upper_osc.n_levels >= upper_n_full,
                f"{next_label}: oscillator levels shorter than MODEL_SPEC NF")
        require(set(np.unique(upper_sl_map)) == set(range(1, upper_n_super + 1)),
                f"{next_label}: active F_TO_S prefix does not cover MODEL_SPEC N_SL")
        for upper_level, upper_sl in enumerate(upper_sl_map):
            runtime = runtime_map[(next_label, upper_level + 1)]
            require(int(runtime["superlevel"]) == int(upper_sl),
                    f"{next_label} FL {upper_level + 1}: F_TO_S/runtime SL mismatch")
            osc_config = " ".join(
                str(upper_osc.levels["config"][upper_level]).split()).lower()
            map_config = " ".join(str(runtime["configuration"]).split()).lower()
            require(osc_config == map_config,
                    f"{next_label} FL {upper_level + 1}: oscillator/configuration map mismatch")
            upper_global = level_index.get((Z, stage + 1, upper_level + 1))
            require(upper_global is not None,
                    f"{next_label} FL {upper_level + 1}: level absent from deck")
            deck_row = all_level_rows[upper_global]
            require(int(deck_row["super_level"]) + 1 == int(upper_sl),
                    f"{next_label} FL {upper_level + 1}: deck/CMFGEN SL mismatch")
            upper_energy = float(upper_osc.levels["E_cm"][upper_level]) * CM_TO_EV
            require(abs(float(deck_row["energy_eV"]) - upper_energy) <= 1e-9,
                    f"{next_label} FL {upper_level + 1}: deck energy mismatch")
            require(math.isclose(float(deck_row["g"]),
                                 float(upper_osc.levels["g"][upper_level]),
                                 rel_tol=0.0, abs_tol=1e-12),
                    f"{next_label} FL {upper_level + 1}: deck statistical weight mismatch")
        _, upper_names, _, _ = parse_f_to_s_int_seq(
            upper_sources["f_to_s"], upper_n_full)
        upper_data = {
            "levels": upper_osc.levels,
            "n_kept": upper_n_full,
            "ftos": SimpleNamespace(sl_of_fl=upper_sl_map),
        }
        grid_route_diagnostics = []
        for grid_id, phot_path in enumerate(
                atomic._phot_route_paths(Path(sources["phot"]))):
            phot = atomic.parse_phot(phot_path)
            python_routes = atomic._phot_target_components(
                phot, upper_data, level_index, Z, stage + 1)
            cmfgen_routes = cmfgen_get_num_phot_routes_source_components(
                phot.final_state, upper_names,
                np.asarray(upper_osc.levels["g"][:upper_n_full], dtype=float),
                np.asarray(upper_osc.levels["E_cm"][:upper_n_full], dtype=float) * CM_TO_EV,
                upper_sl_map)
            python_rows = [{
                "anchor_level_1based": int(route["upper_level_num"]) + 1,
                "target_superlevel_1based": int(route["upper_superlevel"]),
                "target_energy_eV": float(route["energy_eV"]),
                "statistical_weight": float(route["g"]),
                "probability": float(route["probability"]),
            } for route in python_routes]
            same_routes = len(python_rows) == len(cmfgen_routes)
            if same_routes:
                for python_row, cmfgen_row in zip(python_rows, cmfgen_routes):
                    same_routes = same_routes and (
                        python_row["anchor_level_1based"] ==
                        cmfgen_row["anchor_level_1based"] and
                        python_row["target_superlevel_1based"] ==
                        cmfgen_row["target_superlevel_1based"] and
                        math.isclose(python_row["target_energy_eV"],
                                     cmfgen_row["target_energy_eV"],
                                     rel_tol=0.0, abs_tol=1e-10) and
                        math.isclose(python_row["statistical_weight"],
                                     cmfgen_row["statistical_weight"],
                                     rel_tol=0.0, abs_tol=1e-12) and
                        math.isclose(python_row["probability"],
                                     cmfgen_row["probability"],
                                     rel_tol=0.0, abs_tol=1e-12))
            grid_route_diagnostics.append({
                "phot_grid_0based": grid_id,
                "phot_path": str(phot_path),
                "final_state": phot.final_state,
                "python_grouped_route_count": len(python_rows),
                "cmfgen_source_route_count": len(cmfgen_routes),
                "component_sequence_match": bool(same_routes),
                "python_grouped_components": python_rows,
                "cmfgen_source_components": cmfgen_routes,
            })
        route_expansion_by_ion[label] = grid_route_diagnostics
        matched_upper_population_by_sl = scrtemp_superlevel_populations(
            matched_scrtemp_state, next_label, upper_n_super,
            scrtemp_variable_indices)
        upper_edge_nu15 = cmfgen_runtime_edge_nu15(
            float(upper_osc.ionization_eV) / CM_TO_EV,
            np.asarray(upper_osc.levels["E_cm"][:upper_n_full], dtype=float))
        matched_lower_population_by_sl = scrtemp_superlevel_populations(
            matched_scrtemp_state, label, n_super, scrtemp_variable_indices)
        pre_solve_full_population, lte_full_on_super = cmfgen_sup_to_full_v3b(
            matched_lower_population_by_sl, lower_edge_nu15,
            np.asarray(osc.levels["g"][:n_full], dtype=float), sl_map,
            int_seq, upper_edge_nu15,
            np.asarray(upper_osc.levels["g"][:upper_n_full], dtype=float),
            upper_sl_map, matched_upper_population_by_sl[:, 0],
            matched_temperature_1e4, matched_electron_density,
            matched_positive_ion_density, float(stage - 1),
            do_lev_dissolution, return_lte_full_on_super=True)
        lumina_recombination_density = lumina_recombination_rate_density(
            channels, Z, stage, n_full, n_super, sl_map, level_energy_eV,
            np.asarray(osc.levels["g"][:n_full], dtype=float),
            float(osc.ionization_eV), stage + 1, upper_sl_map,
            np.asarray(upper_osc.levels["E_cm"][:upper_n_full], dtype=float) * CM_TO_EV,
            np.asarray(upper_osc.levels["g"][:upper_n_full], dtype=float),
            matched_upper_population_by_sl, level_index, j_milne_rows,
            states_milne, matched_temperature, matched_electron_density)
        snapshot_by_ion[f"{atomic.SYM[Z]} {atomic.ROMAN[stage]}"] = {
            "prrr_ion_density_meaning": "DXzV: next-stage SL1 population",
            "scrtemp_record_1based": matched_scrtemp_record,
            "max_abs_PRRR_DXzV_vs_final_POP_next_SL1_rel": next_sl1_max,
            "max_abs_PRRR_vs_final_RVTJ_electron_density_rel": prrr_rvtj_ne_max,
            "max_abs_PRRR_vs_final_RVTJ_temperature_rel": prrr_rvtj_temperature_max,
            "max_abs_PRRR_vs_matched_SCRTEMP_next_SL1_rel":
                matched_errors["next_SL1_max_rel"],
            "max_abs_PRRR_vs_matched_SCRTEMP_electron_density_rel":
                matched_errors["electron_density_max_rel"],
            "max_abs_PRRR_vs_matched_SCRTEMP_temperature_rel":
                matched_errors["temperature_max_rel"],
            "next_SL1_tolerance_rel": PRRR_NEXT_SL1_TOL_REL,
            "electron_density_tolerance_rel": PRRR_RVTJ_ELECTRON_TOL_REL,
            "temperature_tolerance_rel": PRRR_RVTJ_TEMPERATURE_TOL_REL,
            "element_population_closure_pass": element_closure[atomic.SYM[Z]]["pass"],
            "prrr_pre_solve_state_match": bool(same_state),
        }
        for depth in range(nd):
            snapshot_rows.append({
                "ion": f"{atomic.SYM[Z]} {atomic.ROMAN[stage]}",
                "depth_1based": depth + 1,
                "current_ion_population_cm-3": float(population_density[depth]),
                "prrr_dxzv_next_SL1_cm-3": float(prrr_next_sl1[depth]),
                "next_stage_SL1_population_cm-3": float(next_sl1[depth]),
                "prrr_dxzv_vs_next_SL1_rel": float(next_sl1_delta[depth]),
                "prrr_electron_density_cm-3": float(cmfgen_electron_density[depth]),
                "rvtj_electron_density_cm-3": float(rvtj_electron_density[depth]),
                "prrr_vs_rvtj_electron_density_rel": float(prrr_rvtj_ne_delta[depth]),
                "prrr_temperature_K": float(prrr_temperature[depth]),
                "rvtj_temperature_K": float(rvtj_temperature[depth]),
                "prrr_vs_rvtj_temperature_rel": float(
                    prrr_rvtj_temperature_delta[depth]),
                "matched_scrtemp_record_1based": matched_scrtemp_record,
                "matched_scrtemp_next_SL1_cm-3": float(
                    matched_upper_population_by_sl[depth, 0]),
                "matched_scrtemp_electron_density_cm-3": float(
                    matched_electron_density[depth]),
                "matched_scrtemp_temperature_K": float(
                    matched_temperature[depth]),
            })
        sigma_blocks.append(sigma)
        threshold_blocks.append(nu_threshold)
        ions.append({
            "key": key, "label": label, "name": f"{atomic.SYM[Z]} {atomic.ROMAN[stage]}",
            "nf": n_full, "ns": n_super, "sl": sl_map,
            "population": pre_solve_full_population,
            "lte_full_on_super": lte_full_on_super,
            "final_population": populations,
            "pre_solve_superlevel_population": matched_lower_population_by_sl,
            "int_seq": int_seq,
            "lower_edge_nu15": lower_edge_nu15,
            "lower_g": np.asarray(osc.levels["g"][:n_full], dtype=float),
            "upper_label": next_label, "upper_n_super": upper_n_super,
            "upper_edge_nu15": upper_edge_nu15,
            "upper_g": np.asarray(upper_osc.levels["g"][:upper_n_full], dtype=float),
            "upper_sl_map": upper_sl_map,
            "ion_charge": float(stage - 1),
            "atomic_number": Z,
            "ionization_eV": float(osc.ionization_eV),
            "z_screen": float(osc.z_screen),
            "phot_source": Path(sources["phot"]),
            "lower_config": [str(value) for value in
                             osc.levels["config"][:n_full]],
            "lower_energy_eV": level_energy_eV,
            "sigma": sigma,
            "nu_threshold": nu_threshold,
            "cmfgen_rate_density": cmfgen_rate_density,
            "cmfgen_recombination_density": cmfgen_recombination_density,
            "lumina_recombination_density": lumina_recombination_density,
            "sigma_rows": indices, "prrr_path": prrr_path,
        })
        offsets.append(offsets[-1] + n_full)

    post_solve_scrtemp_record = None
    post_solve_scrtemp_metrics = {}
    sup_to_full_validation = {}
    sup_to_full_validation_error = None
    try:
        (post_solve_scrtemp_record, post_solve_scrtemp_state,
         post_solve_scrtemp_metrics) = match_scrtemp_state_to_final_populations(
             scrtemp_reader, scrtemp_variable_indices, final_superlevel_by_ion,
             scrtemp_nt - 2, scrtemp_nt - 1, rvtj_electron_density,
             rvtj_temperature)
        post_temperature_1e4 = post_solve_scrtemp_state[scrtemp_nt - 1]
        post_electron_density = post_solve_scrtemp_state[scrtemp_nt - 2]
        post_positive_ion_density = scrtemp_positive_ion_population(
            post_solve_scrtemp_state, scrtemp_variables)
        for ion in ions:
            post_lower = scrtemp_superlevel_populations(
                post_solve_scrtemp_state, ion["label"], ion["ns"],
                scrtemp_variable_indices)
            post_upper = scrtemp_superlevel_populations(
                post_solve_scrtemp_state, ion["upper_label"],
                ion["upper_n_super"], scrtemp_variable_indices)
            reconstructed = cmfgen_sup_to_full_v3b(
                post_lower, ion["lower_edge_nu15"], ion["lower_g"],
                ion["sl"], ion["int_seq"], ion["upper_edge_nu15"],
                ion["upper_g"], ion["upper_sl_map"], post_upper[:, 0],
                post_temperature_1e4, post_electron_density,
                post_positive_ion_density, ion["ion_charge"],
                do_lev_dissolution)
            metrics = population_identity_metrics(
                ion["final_population"], reconstructed)
            witness = metrics["max_witness"]
            if witness is not None:
                level_index = witness["full_level_1based"] - 1
                witness["superlevel_1based"] = int(ion["sl"][level_index])
                witness["int_seq"] = int(ion["int_seq"][level_index])
            metrics["tolerance_rel"] = SUP_TO_FULL_POP_TOL_REL
            metrics["pass"] = bool(
                metrics["n_asymmetric_zero"] == 0 and
                metrics["max_abs_relative"] is not None and
                metrics["max_abs_relative"] <= SUP_TO_FULL_POP_TOL_REL)
            sup_to_full_validation[ion["name"]] = metrics
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        sup_to_full_validation_error = str(exc)
    sup_to_full_validation_pass = (
        post_solve_scrtemp_record is not None and
        len(sup_to_full_validation) == len(ions) and
        all(row["pass"] for row in sup_to_full_validation.values()))

    out.mkdir(parents=True, exist_ok=False)
    fixture_output = l1bf.run_fixture(
        str(fixture), "main", str(out), 1, 19.48 * 86400.0,
        j_rows, states, int(baked["nbin"]), float(baked["numin"]),
        math.log(float(baked["numax"]) / float(baked["numin"])) /
        int(baked["nbin"]), np.concatenate(sigma_blocks),
        np.concatenate(threshold_blocks))

    detail_rows = []
    by_ion = {}
    all_sl_ref, all_sl_lum = [], []
    all_lte_weighted_sl_ref, all_lte_weighted_sl_lum = [], []
    all_total_ref, all_total_lum = [], []
    all_lte_weighted_total_ref, all_lte_weighted_total_lum = [], []
    all_recombination_ref, all_recombination_lum = [], []
    raw_field_witness_diagnostic = {}
    sigma_bin_edges = np.exp(
        math.log(float(baked["numin"])) +
        np.arange(int(baked["nbin"]) + 1) *
        math.log(float(baked["numax"]) / float(baked["numin"])) /
        int(baked["nbin"]))
    for index, ion in enumerate(ions):
        lo, hi = offsets[index:index + 2]
        gamma = fixture_output["gamma"][lo:hi]
        state = fixture_output["state"][lo:hi]
        wmiss = fixture_output["wmiss"][lo:hi]
        pop = ion["population"]
        lumina_sl = np.zeros((ion["ns"], nd), dtype=float)
        invalid_active = 0
        weighted_wmiss = np.zeros(nd, dtype=float)
        ion_population = pop.sum(axis=1)
        require(np.all(ion_population > 0), f"{ion['label']}: non-positive ion population")
        for depth in range(nd):
            for level in range(ion["nf"]):
                sl = int(ion["sl"][level]) - 1
                lumina_sl[sl, depth] += pop[depth, level] * gamma[level, depth]
                if pop[depth, level] > 0 and state[level, depth] not in (VALID, EXACT_ZERO):
                    invalid_active += 1
            weighted_wmiss[depth] = float(
                np.dot(pop[depth], wmiss[:, depth]) / ion_population[depth])
        cmf_sl = ion["cmfgen_rate_density"]
        sl_metrics = rate_metrics(cmf_sl, lumina_sl)
        lumina_lte_weighted_sl = lte_weighted_superlevel_photo_rate_density(
            ion["pre_solve_superlevel_population"], ion["lte_full_on_super"],
            ion["sl"], gamma)
        lte_weighted_sl_metrics = rate_metrics(cmf_sl, lumina_lte_weighted_sl)
        cmf_recombination = ion["cmfgen_recombination_density"]
        lumina_recombination = ion["lumina_recombination_density"]
        recombination_metrics = rate_metrics(cmf_recombination,
                                            lumina_recombination)
        cmf_total = total_photoionization_rate(cmf_sl, ion_population)
        lum_total = total_photoionization_rate(lumina_sl, ion_population)
        lte_weighted_lum_total = total_photoionization_rate(
            lumina_lte_weighted_sl, ion_population)
        lte_weighted_total_metrics = rate_metrics(cmf_total, lte_weighted_lum_total)
        total_metrics = rate_metrics(cmf_total, lum_total)
        positive_cells = (cmf_sl > 0.0) & (lumina_sl > 0.0)
        if np.any(positive_cells):
            cell_delta = np.full(cmf_sl.shape, -np.inf, dtype=float)
            cell_delta[positive_cells] = np.abs(np.log10(
                lumina_sl[positive_cells] / cmf_sl[positive_cells]))
            witness_sl, witness_depth = np.unravel_index(
                int(np.argmax(cell_delta)), cell_delta.shape)
            raw_field_rate = 0.0
            raw_phot_rate = 0.0
            witness_levels = np.flatnonzero(ion["sl"] == witness_sl + 1)
            for level in witness_levels:
                gamma_raw = raw_field_step_sigma_gamma(
                    nu, field[:, witness_depth], edges, states[witness_depth],
                    sigma_bin_edges, ion["sigma"][level],
                    float(ion["nu_threshold"][level]))
                raw_field_rate += pop[witness_depth, level] * gamma_raw
                global_level = int(ion["sigma_rows"][level])
                raw_sigma_terms = cmfgen_raw_sigma_terms_for_level(
                    ion["phot_source"], channels, global_level,
                    ion["lower_config"][level],
                    float(ion["lower_energy_eV"][level]),
                    ion["ionization_eV"], ion["atomic_number"], ion["z_screen"],
                    loaded["phot_smoother"])
                gamma_raw_phot = raw_field_continuous_sigma_gamma(
                    nu, field[:, witness_depth], edges, states[witness_depth],
                    raw_sigma_terms)
                raw_phot_rate += pop[witness_depth, level] * gamma_raw_phot
            ref_witness = float(cmf_sl[witness_sl, witness_depth])
            standard_witness = float(lumina_sl[witness_sl, witness_depth])
            raw_field_witness_diagnostic[ion["name"]] = {
                "superlevel_1based": int(witness_sl + 1),
                "depth_1based": int(witness_depth + 1),
                "full_levels_1based": [int(level + 1) for level in witness_levels],
                "cmfgen_prrr_rate_density_cm-3_s-1": ref_witness,
                "canonical_field_rate_density_cm-3_s-1": standard_witness,
                "raw_edd_factor_field_rate_density_cm-3_s-1": float(raw_field_rate),
                "raw_linked_phot_sigma_rate_density_cm-3_s-1": float(raw_phot_rate),
                "canonical_abs_delta_dex": float(cell_delta[witness_sl, witness_depth]),
                "raw_field_abs_delta_dex": (
                    abs(math.log10(raw_field_rate / ref_witness))
                    if raw_field_rate > 0.0 and ref_witness > 0.0 else None),
                "raw_vs_canonical_abs_delta_dex": (
                    abs(math.log10(raw_field_rate / standard_witness))
                    if raw_field_rate > 0.0 and standard_witness > 0.0 else None),
                "raw_phot_sigma_abs_delta_dex": (
                    abs(math.log10(raw_phot_rate / ref_witness))
                    if raw_phot_rate > 0.0 and ref_witness > 0.0 else None),
                "raw_phot_sigma_vs_baked_sigma_abs_delta_dex": (
                    abs(math.log10(raw_phot_rate / raw_field_rate))
                    if raw_phot_rate > 0.0 and raw_field_rate > 0.0 else None),
                "interpretation": "diagnostic_only_raw_edd_factor_with_exact_linked_phot_sigma_evaluator",
            }
        all_sl_ref.extend(cmf_sl.ravel().tolist())
        all_sl_lum.extend(lumina_sl.ravel().tolist())
        all_lte_weighted_sl_ref.extend(cmf_sl.ravel().tolist())
        all_lte_weighted_sl_lum.extend(lumina_lte_weighted_sl.ravel().tolist())
        all_total_ref.extend(cmf_total.tolist())
        all_total_lum.extend(lum_total.tolist())
        all_lte_weighted_total_ref.extend(cmf_total.tolist())
        all_lte_weighted_total_lum.extend(lte_weighted_lum_total.tolist())
        photo_numeric_criteria = all(
            value is not None and value <= limit
            for value, limit in (
                (sl_metrics["median_abs_delta_dex"], RATE_MEDIAN_LIMIT_DEX),
                (sl_metrics["p95_abs_delta_dex"], RATE_P95_LIMIT_DEX),
                (sl_metrics["max_abs_delta_dex"], RATE_MAX_LIMIT_DEX),
                (total_metrics["median_abs_delta_dex"], RATE_MEDIAN_LIMIT_DEX),
                (total_metrics["p95_abs_delta_dex"], RATE_P95_LIMIT_DEX),
                (total_metrics["max_abs_delta_dex"], RATE_MAX_LIMIT_DEX),
            )) and sl_metrics["n_asymmetric_zero"] == 0 and \
            total_metrics["n_asymmetric_zero"] == 0 and invalid_active == 0
        photo_criteria = photo_numeric_criteria and sup_to_full_validation_pass
        recombination_criteria = all(
            value is not None and value <= limit
            for value, limit in (
                (recombination_metrics["median_abs_delta_dex"], RATE_MEDIAN_LIMIT_DEX),
                (recombination_metrics["p95_abs_delta_dex"], RATE_P95_LIMIT_DEX),
                (recombination_metrics["max_abs_delta_dex"], RATE_MAX_LIMIT_DEX),
            )) and recombination_metrics["n_asymmetric_zero"] == 0
        all_recombination_ref.extend(cmf_recombination.ravel().tolist())
        all_recombination_lum.extend(lumina_recombination.ravel().tolist())
        criteria = photo_criteria and recombination_criteria
        by_ion[ion["name"]] = {
            "model_spec_nf": ion["nf"], "model_spec_ns": ion["ns"],
            "superlevel_rate_density": sl_metrics,
            "lte_weighted_superlevel_photo_rate_density_diagnostic":
                lte_weighted_sl_metrics,
            "lte_weighted_ion_total_gamma_diagnostic":
                lte_weighted_total_metrics,
            "ion_total_gamma_s-1": total_metrics,
            "radiative_recombination_rate_density": recombination_metrics,
            "invalid_active_state_cells": invalid_active,
            "population_weighted_wmiss_max": float(weighted_wmiss.max()),
            "photo_numeric_criteria_met": bool(photo_numeric_criteria),
            "sup_to_full_prestate_population_validation_pass":
                bool(sup_to_full_validation_pass),
            "photo_criteria_met": bool(photo_criteria),
            "recombination_criteria_met": bool(recombination_criteria),
            "criteria_met": bool(criteria),
        }
        for superlevel in range(ion["ns"]):
            for depth in range(nd):
                ref_value = float(cmf_sl[superlevel, depth])
                lum_value = float(lumina_sl[superlevel, depth])
                status = ("BOTH_ZERO" if ref_value == 0.0 and lum_value == 0.0 else
                          "ASYMMETRIC_ZERO" if ref_value == 0.0 or lum_value == 0.0 else
                          "COMPARED")
                delta = (math.log10(lum_value / ref_value)
                         if status == "COMPARED" else "")
                rr_ref = float(cmf_recombination[superlevel, depth])
                rr_lum = float(lumina_recombination[superlevel, depth])
                rr_status = ("BOTH_ZERO" if rr_ref == 0.0 and rr_lum == 0.0 else
                             "ASYMMETRIC_ZERO" if rr_ref == 0.0 or rr_lum == 0.0 else
                             "COMPARED")
                rr_delta = (math.log10(rr_lum / rr_ref)
                            if rr_status == "COMPARED" else "")
                detail_rows.append({
                    "ion": ion["name"], "depth_1based": depth + 1,
                    "superlevel_1based": superlevel + 1,
                    "cmfgen_photo_rate_density_cm-3_s-1": ref_value,
                    "lumina_photo_rate_density_cm-3_s-1": lum_value,
                    "lumina_over_cmfgen_photo": (lum_value / ref_value
                                                  if status == "COMPARED" else ""),
                    "photo_delta_log10_dex": delta, "photo_status": status,
                    "lumina_lte_weighted_photo_rate_density_cm-3_s-1": float(
                        lumina_lte_weighted_sl[superlevel, depth]),
                    "lte_weighted_over_cmfgen_photo": (
                        float(lumina_lte_weighted_sl[superlevel, depth] / ref_value)
                        if ref_value > 0.0 else ""),
                    "cmfgen_recombination_rate_density_cm-3_s-1": rr_ref,
                    "lumina_recombination_rate_density_cm-3_s-1": rr_lum,
                    "lumina_over_cmfgen_recombination": (
                        rr_lum / rr_ref if rr_status == "COMPARED" else ""),
                    "recombination_delta_log10_dex": rr_delta,
                    "recombination_status": rr_status,
                })

    overall_sl = rate_metrics(np.asarray(all_sl_ref), np.asarray(all_sl_lum))
    overall_lte_weighted_sl = rate_metrics(
        np.asarray(all_lte_weighted_sl_ref),
        np.asarray(all_lte_weighted_sl_lum))
    overall_total = rate_metrics(np.asarray(all_total_ref), np.asarray(all_total_lum))
    overall_lte_weighted_total = rate_metrics(
        np.asarray(all_lte_weighted_total_ref),
        np.asarray(all_lte_weighted_total_lum))
    overall_recombination = rate_metrics(
        np.asarray(all_recombination_ref), np.asarray(all_recombination_lum))
    overall_pass = all(row["criteria_met"] for row in by_ion.values())
    same_state_pass = all(
        row["prrr_pre_solve_state_match"] for row in snapshot_by_ion.values())
    input_paths = [
        run / name for name in (
            "EDDFACTOR", "EDDFACTOR_INFO", "RVTJ", "MODEL", "MODEL_SPEC",
            "LEVEL_SL_STEQ_LINKS", "atomic_links.txt", "POPSIL", "POPSUL",
            "SCRTEMP",
        )
    ] + [
        deck / name for name in (
            "DECK_PROVENANCE.json", "atomic_source_identity.json",
            "atomic_vintage_manifest.csv", "cmfgen_sigma_bf.bin",
            "cmfgen_bf_channels.bin", "cmfgen_bf_channels_manifest.json",
            "levels.csv",
        )
    ] + [fixture, Path(__file__).resolve(),
         ROOT / "scripts/cmfgen_phot_smoothing.py",
         ROOT / "src/lumina_plasma.c", ROOT / "src/lumina_element_wide.c"]
    if stamp.get("phot_preprocessing"):
        preprocessing = stamp["phot_preprocessing"]
        input_paths.extend(Path(row["path"]) for row in
                           [preprocessing["library"], *preprocessing["sources"]]
                           if row is not None)
    input_paths.extend(ion["prrr_path"] for ion in ions)
    upper_keys = {(Z, stage + 1) for Z, stage in target_keys}
    input_paths.extend(link_map[key][kind] for key in target_keys
                       for kind in ("osc", "f_to_s", "phot", "col"))
    input_paths.extend(link_map[key][kind] for key in upper_keys
                       for kind in ("osc", "f_to_s"))
    inputs = {str(path.resolve()): sha256_file(path) for path in sorted(set(input_paths))}

    csv_path = out / "si_s_bf_superlevel_rates.csv"
    with csv_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)
    snapshot_csv_path = out / "snapshot_consistency.csv"
    with snapshot_csv_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(snapshot_rows[0]))
        writer.writeheader()
        writer.writerows(snapshot_rows)
    report = {
        "schema": "lumina-cmfgen-bf-superlevel-rate-comparison-v10",
        "scope": ("forensic fixed-state forward photoionization and reverse Milne "
                  "recombination; photo and RR use matched pre-solve SCRTEMP, with "
                  "full-level populations reconstructed by CMFGEN SUP_TO_FULL_V3b; "
                  "includes a diagnostic-only CMFGEN LTE-on-super WSE weighting "
                  "comparison; not an ion-fraction acceptance run"),
        "cmfgen_run": str(run), "lumina_deck": str(deck),
        "phot_preprocessing": stamp.get("phot_preprocessing"),
        "depths": nd, "frequency_bins_lumina": int(baked["nbin"]),
        "canonical_field_bins": int(l1bf.CANON_N_BINS),
        "frequency_coverage_counts": {
            "VALID": int(np.count_nonzero(states == l1bf.VALID)),
            "EXACT_ZERO": int(np.count_nonzero(states == l1bf.EXACT_ZERO)),
            "OUT_OF_GRID": int(np.count_nonzero(states == l1bf.OUT_OF_GRID)),
        },
        "thresholds_abs_log10_dex": {
            "median_max": RATE_MEDIAN_LIMIT_DEX,
            "p95_max": RATE_P95_LIMIT_DEX,
            "max_max": RATE_MAX_LIMIT_DEX,
        },
        "overall_superlevel_rate_density": overall_sl,
        "overall_lte_weighted_superlevel_photo_rate_density_diagnostic":
            overall_lte_weighted_sl,
        "overall_ion_total_gamma": overall_total,
        "overall_lte_weighted_ion_total_gamma_diagnostic":
            overall_lte_weighted_total,
        "overall_radiative_recombination_rate_density": overall_recombination,
        "per_ion": by_ion,
        "snapshot_consistency": {
            "per_ion": snapshot_by_ion,
            "element_population_closure": element_closure,
            "prrr_pre_solve_scrtemp_match_gate": (
                "PASS" if same_state_pass else "NOT_PASS"),
            "matched_scrtemp_record_1based": matched_scrtemp_record,
            "matched_scrtemp_match_metrics": matched_scrtemp_metrics,
            "post_solve_scrtemp_record_1based": post_solve_scrtemp_record,
            "post_solve_scrtemp_match_metrics": post_solve_scrtemp_metrics,
        },
        "raw_field_witness_diagnostic_only": raw_field_witness_diagnostic,
        "route_expansion_source_comparison_diagnostic_only":
            route_expansion_by_ion,
        "full_population_reconstruction": {
            "method": "source-faithful CMFGEN SUP_TO_FULL_V4 -> SUP_TO_FULL_V3b port",
            "do_lev_dissolution": do_lev_dissolution,
            "post_solve_validation_gate": (
                "PASS" if sup_to_full_validation_pass else "NOT_PASS"),
            "post_solve_validation_tolerance_rel": SUP_TO_FULL_POP_TOL_REL,
            "post_solve_validation_per_ion": sup_to_full_validation,
            "post_solve_validation_error": sup_to_full_validation_error,
        },
        "normalization_contract": {
            "prrr_ion_density": "next-stage SL1 DXzV; not current-ion total",
            "total_photoionization_rate_denominator":
                "sum of current-ion full-level populations reconstructed from matched pre-solve SCRTEMP via SUP_TO_FULL_V3b",
            "lte_weighted_photo_diagnostic":
                "sum over full levels of matched pre-solve superlevel population times LTE full-level fraction within that superlevel (HNST_F_ON_S) times Lumina per-level Gamma; diagnostic only, not acceptance",
            "milne_target_weight":
                "G_SL(T; E_anchor)=sum of g_f*exp[-(E_f-E_anchor)/kT] over every active full member of the target superlevel; equivalent to g_anchor/within_sl_fraction_anchor in production",
        },
        "photo_and_recombination_rate_gate": "PASS" if overall_pass else "NOT_PASS",
        "photo_rate_gate": ("PASS" if all(row["photo_criteria_met"]
                                           for row in by_ion.values()) else "NOT_PASS"),
        "recombination_rate_gate": (
            "PASS" if all(row["recombination_criteria_met"]
                          for row in by_ion.values()) else "NOT_PASS"),
        "interpretation_status": (
            "RR_AND_PHOTO_PRE_SOLVE_STATE_MATCHED"
            if same_state_pass and sup_to_full_validation_pass else
            "RR_PRE_SOLVE_STATE_MATCHED; PHOTO_FULL_POP_RECONSTRUCTION_NOT_VALIDATED"
            if same_state_pass else "PRRR_PRE_SOLVE_SCRTEMP_MATCH_FAILED"),
        "solver_phase_alignment": {
            "prrr": "WRRECOMCHK_V5 at cmfgen_sub.f:3033, before SOLVE_FOR_POPS at :4002",
            "radiation_field": "EDDFACTOR output finalized before SOLVE_FOR_POPS",
            "lumina_recombination_inputs":
                "same-run EDDFACTOR plus unique matched SCRTEMP pre-solve upper SL populations, ED, and T",
            "photo_rate_population_inputs":
                "matched pre-solve SCRTEMP superlevels reconstructed to full levels via SUP_TO_FULL_V3b",
            "cmfgen_sub_f_sha256":
                "092f8526661b1f9a5eaeb7a875f07f55623f427ad07c383f2659f2fb67143374",
        },
        "limitations": [
            "Lumina 정·역율은 같은 실행의 EDDFACTOR Jν와 matched pre-solve SCRTEMP 상태를 사용한다; photo full-level population port는 post-solve POP 비교 통과 전까지 잠정이다.",
            "LTE-on-super 가중 광율은 CMFGEN QUAD_MULTI_V10의 WSE population weighting을 분리하는 진단이다. 기존 full-population rate acceptance와 이 진단을 혼동하지 않는다.",
            "CMFGEN GET_NUM_PHOT_ROUTES_V1과 Lumina route grouping의 source comparison은 진단 전용이며 어떤 수용 기준도 바꾸지 않는다.",
            "이 고정-T 비수렴 run은 동일 상태율 커널 비교용이며 자기일관 이온분포 또는 최종 CMFGEN 동등성을 판정하지 않는다.",
        ],
        "input_sha256": inputs,
        "output_csv": str(csv_path),
        "snapshot_consistency_csv": str(snapshot_csv_path),
        "source_identity_sha256": sha256_file(deck / "atomic_source_identity.json"),
        "deck_provenance_sha256": sha256_file(deck / "DECK_PROVENANCE.json"),
    }
    summary_path = out / "summary.json"
    with summary_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({
        "photo_and_recombination_rate_gate": report["photo_and_recombination_rate_gate"],
        "photo_rate_gate": report["photo_rate_gate"],
        "recombination_rate_gate": report["recombination_rate_gate"],
        "overall_superlevel_rate_density": overall_sl,
        "overall_lte_weighted_superlevel_photo_rate_density_diagnostic":
            overall_lte_weighted_sl,
        "overall_ion_total_gamma": overall_total,
        "overall_lte_weighted_ion_total_gamma_diagnostic":
            overall_lte_weighted_total,
        "overall_radiative_recombination_rate_density": overall_recombination,
        "per_ion": by_ion,
        "snapshot_consistency": report["snapshot_consistency"],
        "raw_field_witness_diagnostic_only": raw_field_witness_diagnostic,
        "route_expansion_source_comparison_diagnostic_only":
            route_expansion_by_ion,
        "interpretation_status": report["interpretation_status"],
        "output": str(out),
    }, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
