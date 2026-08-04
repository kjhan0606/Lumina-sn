#!/usr/bin/env python3
"""Independent verifier for the pre-T-SEED TRAD-FIX contract.

This program deliberately does not import or execute a Lumina deck builder,
Lumina solver, or any existing CMFGEN/Lumina parser.  It reads the frozen CSV,
JSON, geometry, source text, and (optionally) CMFGEN binary records directly.

The normal result is a JSON record containing all 50 shells in four states:

1. deck_original_energy
2. gate_independent_reconstruction
3. deck_inferred_color
4. selected_final_contract

Use ``--self-test`` for the small gate-OFF positive control and the deliberately
damaged-W negative control.  The 143 MB CMFGEN comparison is opt-in through
``--cmfgen-dir`` and is intended for lageunha, not the syntax node.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import statistics
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - fail-closed environment check
    raise SystemExit(f"FAIL: numpy is required by this independent verifier: {exc}")


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_DECK = REPO_ROOT / "data" / "tardis_reference_toy06_19p48d_sivcaiv"

EXPECTED_FILE_SHA256 = {
    "plasma_state.csv": "45ccb86be1296e644f8491eb9b1346b2b5746f96055003d7f7009082a336a6f4",
    "config.json": "cf61ab7c880243ffa94bba95b55c3bb4c88e526bcdf1d9b76bd81f44ff81293b",
    "geometry.csv": "21bb9349c11bceb7c815ca6fd5b21a647bddf1ba3f3da4311b97da93dc3ce3d6",
}
EXPECTED_W_DIGEST = "d93cb82e99a205fd66cf4e7406975d59a0588beef095ca00d684c2b8b71e4809"
EXPECTED_CMFGEN_SHA256 = {
    "EDDFACTOR": "83acc14a35999aaf39cf728ce783308be31fa52676b1c5410b5e84f4cc009705",
    "EDDFACTOR_INFO": "2c032445a9483d5154c15cdac5c0f14dfbb3f45dbb628e2d4936c29b3efabd42",
    "RVTJ": "a042fd49c726dc1c2b710c997fa3d27780189e98edebc28f9d77c06ffe034f78",
}

# cgs constants.  They are written here rather than imported from Lumina.
H_PLANCK = 6.62607015e-27
K_BOLTZMANN = 1.380649e-16
C_LIGHT = 2.99792458e10
C_LIGHT_A_S = 2.99792458e18
SIGMA_SB = 5.670374419e-5
RADIATION_A = 4.0 * SIGMA_SB / C_LIGHT
FOUR_PI_OVER_C = 4.0 * math.pi / C_LIGHT
ZETA_4 = math.pi**4 / 90.0
ZETA_5 = 1.0369277551433699
MOMENT_PLANCK_FACTOR = 4.0 * ZETA_5 / ZETA_4

UNIQUE_TOLERANCE_K = 1.0e-6
EXPECTED_SHELL_COUNT = 50
EXPECTED_CMFGEN_ND = 90
EXPECTED_CMFGEN_VALID_RECORDS = 196185
EXPECTED_CMFGEN_COVERED_SHELLS = list(range(44))

BANDS_A = (
    ("450_918_A", 450.0, 918.0),
    ("918_1290_A", 918.0, 1290.0),
    ("1290_2000_A", 1290.0, 2000.0),
    ("2000_10000_A", 2000.0, 10000.0),
    ("10000_25000_A", 10000.0, 25000.0),
)

EXPECTED_GATE_METRICS = {
    "changed_shells": list(range(1, 50)),
    "maximum_absolute_change": {"shell": 49, "value_K": 7336.498847, "digits": 6},
    "maximum_temperature_ratio": {"shell": 49, "value": 3.341240737, "digits": 9},
    "maximum_relative_increase_percent": {"shell": 49, "value": 234.124074, "digits": 6},
    "median_relative_increase_percent": {"value": 148.205192, "digits": 6},
    "bolometric_ratio": {
        "s10": {"value": 10.131996, "digits": 6},
        "s25": {"value": 39.228813, "digits": 6},
        "s40": {"value": 87.001843, "digits": 6},
        "s49": {"value": 124.632432, "digits": 6},
    },
}

FINAL_CONTRACT_CHOICES = (
    "deck-inferred-color",
    "config-inner-color",
    "gate-copy-color",
    "raw-deck-diagnostic",
)


class ContractError(RuntimeError):
    """A fail-closed contract or input error."""


@dataclass(frozen=True)
class PlasmaRow:
    shell_id: int
    W: float
    T_rad: float


@dataclass(frozen=True)
class GeometryRow:
    shell_id: int
    velocity_midpoint_km_s: float


def finite_positive(value: float, label: str) -> float:
    if not math.isfinite(value):
        raise ContractError(f"{label} is NaN or Inf")
    if value <= 0.0:
        raise ContractError(f"{label} is nonpositive: {value!r}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_w(rows: Iterable[PlasmaRow]) -> str:
    """Canonical binary64 W fingerprint, independent of CSV formatting."""
    digest = hashlib.sha256()
    for row in rows:
        digest.update(f"{row.shell_id},{row.W.hex()}\n".encode("ascii"))
    return digest.hexdigest()


def require_columns(fieldnames: list[str] | None, required: Iterable[str], path: Path) -> None:
    present = set(fieldnames or [])
    missing = [name for name in required if name not in present]
    if missing:
        raise ContractError(f"{path}: missing columns {missing}")


def read_plasma(path: Path) -> list[PlasmaRow]:
    rows: list[PlasmaRow] = []
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        require_columns(reader.fieldnames, ("shell_id", "W", "T_rad"), path)
        for index, raw in enumerate(reader, start=2):
            try:
                shell = int(raw["shell_id"])
                W = finite_positive(float(raw["W"]), f"{path}:{index}:W")
                temperature = finite_positive(float(raw["T_rad"]), f"{path}:{index}:T_rad")
            except (TypeError, ValueError) as exc:
                raise ContractError(f"{path}:{index}: invalid plasma row: {exc}") from exc
            rows.append(PlasmaRow(shell, W, temperature))
    if len(rows) != EXPECTED_SHELL_COUNT:
        raise ContractError(f"{path}: {len(rows)} shells, expected {EXPECTED_SHELL_COUNT}")
    if [row.shell_id for row in rows] != list(range(EXPECTED_SHELL_COUNT)):
        raise ContractError(f"{path}: shell_id must be contiguous s0-s49 in file order")
    return rows


def read_geometry(path: Path) -> list[GeometryRow]:
    rows: list[GeometryRow] = []
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        require_columns(reader.fieldnames, ("shell_id", "v_inner", "v_outer"), path)
        for index, raw in enumerate(reader, start=2):
            try:
                shell = int(raw["shell_id"])
                v_inner = finite_positive(float(raw["v_inner"]), f"{path}:{index}:v_inner")
                v_outer = finite_positive(float(raw["v_outer"]), f"{path}:{index}:v_outer")
            except (TypeError, ValueError) as exc:
                raise ContractError(f"{path}:{index}: invalid geometry row: {exc}") from exc
            if v_outer <= v_inner:
                raise ContractError(f"{path}:{index}: v_outer <= v_inner")
            rows.append(GeometryRow(shell, 0.5 * (v_inner + v_outer) / 1.0e5))
    if [row.shell_id for row in rows] != list(range(EXPECTED_SHELL_COUNT)):
        raise ContractError(f"{path}: geometry shell_id must be contiguous s0-s49")
    return rows


def load_deck(deck_dir: Path, require_frozen_hashes: bool = True) -> tuple[
    list[PlasmaRow], dict[str, Any], list[GeometryRow], dict[str, str]
]:
    paths = {name: deck_dir / name for name in EXPECTED_FILE_SHA256}
    for path in paths.values():
        if not path.is_file():
            raise ContractError(f"missing deck input {path}")

    rows = read_plasma(paths["plasma_state.csv"])
    w_fingerprint = digest_w(rows)
    if require_frozen_hashes and w_fingerprint != EXPECTED_W_DIGEST:
        raise ContractError(
            "W fingerprint mismatch: "
            f"observed={w_fingerprint} expected={EXPECTED_W_DIGEST}"
        )

    observed_sha = {name: sha256_file(path) for name, path in paths.items()}
    if require_frozen_hashes:
        for name, expected in EXPECTED_FILE_SHA256.items():
            if observed_sha[name] != expected:
                raise ContractError(
                    f"{name} SHA-256 mismatch: observed={observed_sha[name]} expected={expected}"
                )

    with paths["config.json"].open() as stream:
        try:
            config = json.load(stream)
        except json.JSONDecodeError as exc:
            raise ContractError(f"{paths['config.json']}: invalid JSON: {exc}") from exc
    t_inner = finite_positive(float(config.get("T_inner_K", math.nan)), "config.T_inner_K")
    if int(config.get("n_shells", -1)) != EXPECTED_SHELL_COUNT:
        raise ContractError("config.n_shells does not equal 50")
    config["T_inner_K"] = t_inner
    geometry = read_geometry(paths["geometry.csv"])
    observed_sha["W_binary64_digest"] = w_fingerprint
    return rows, config, geometry, observed_sha


def reconstruct_gate(rows: list[PlasmaRow], enabled: bool) -> list[PlasmaRow]:
    """Independent transcription of the load-time gate; no production call."""
    if not rows:
        raise ContractError("cannot reconstruct gate for an empty profile")
    anchor = rows[0].T_rad
    result = [
        PlasmaRow(row.shell_id, row.W, anchor if enabled and row.shell_id >= 1 else row.T_rad)
        for row in rows
    ]
    for before, after in zip(rows, result):
        if before.W.hex() != after.W.hex():
            raise ContractError(f"gate reconstruction changed W in s{before.shell_id}")
    return result


def tolerance_unique_count(values: Iterable[float], tolerance: float) -> int:
    ordered = sorted(values)
    if not ordered:
        return 0
    count = 1
    anchor = ordered[0]
    for value in ordered[1:]:
        if abs(value - anchor) > tolerance:
            count += 1
            anchor = value
    return count


def planck_bnu_scalar(temperature: float, frequency_hz: float) -> float:
    finite_positive(temperature, "Planck temperature")
    finite_positive(frequency_hz, "Planck frequency")
    x = H_PLANCK * frequency_hz / (K_BOLTZMANN * temperature)
    prefactor = 2.0 * H_PLANCK * frequency_hz**3 / C_LIGHT**2
    if x > 700.0:
        return prefactor * math.exp(-x)
    return prefactor / math.expm1(x)


def planck_bnu_array(temperature: float, frequency_hz: np.ndarray) -> np.ndarray:
    finite_positive(temperature, "Planck temperature")
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    if not np.isfinite(frequency).all() or np.any(frequency <= 0.0):
        raise ContractError("Planck frequency array contains nonpositive/NaN/Inf")
    x = H_PLANCK * frequency / (K_BOLTZMANN * temperature)
    prefactor = 2.0 * H_PLANCK * frequency**3 / C_LIGHT**2
    result = np.empty_like(frequency)
    ordinary = x <= 700.0
    result[ordinary] = prefactor[ordinary] / np.expm1(x[ordinary])
    result[~ordinary] = prefactor[~ordinary] * np.exp(-x[~ordinary])
    return result


def log_planck_bnu_array(temperature: float, frequency_hz: np.ndarray) -> np.ndarray:
    finite_positive(temperature, "Planck fit temperature")
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    x = H_PLANCK * frequency / (K_BOLTZMANN * temperature)
    log_denom = np.empty_like(x)
    large = x > 50.0
    log_denom[large] = x[large] + np.log1p(-np.exp(-x[large]))
    log_denom[~large] = np.log(np.expm1(x[~large]))
    return math.log(2.0 * H_PLANCK / C_LIGHT**2) + 3.0 * np.log(frequency) - log_denom


_GL_X, _GL_W = np.polynomial.legendre.leggauss(96)


def planck_band_integral(temperature: float, dilution: float, lambda_lo_a: float,
                         lambda_hi_a: float) -> float:
    """Integral of W*B_nu over the wavelength-selected frequency interval."""
    finite_positive(temperature, "band temperature")
    finite_positive(dilution, "band W")
    if not (0.0 < lambda_lo_a < lambda_hi_a):
        raise ContractError("invalid wavelength band")
    nu_lo = C_LIGHT_A_S / lambda_hi_a
    nu_hi = C_LIGHT_A_S / lambda_lo_a
    midpoint = 0.5 * (nu_lo + nu_hi)
    halfwidth = 0.5 * (nu_hi - nu_lo)
    frequencies = midpoint + halfwidth * _GL_X
    values = planck_bnu_array(temperature, frequencies)
    integral = dilution * halfwidth * float(np.dot(_GL_W, values))
    return finite_positive(integral, "band integral")


def select_final_temperatures(contract: str, rows: list[PlasmaRow], config: dict[str, Any],
                              inferred_color: list[float]) -> tuple[list[float], str]:
    if contract == "deck-inferred-color":
        return inferred_color.copy(), (
            "mechanical T_energy/W^0.25 candidate; selection for verification output "
            "does not approve it as the physical canonical value"
        )
    if contract == "config-inner-color":
        return [float(config["T_inner_K"])] * len(rows), (
            "config.T_inner_K candidate; selection does not establish shared-field provenance"
        )
    if contract == "gate-copy-color":
        return [rows[0].T_rad] * len(rows), (
            "current gate-copy candidate; selection does not resolve the 10020/10470/14172 conflict"
        )
    if contract == "raw-deck-diagnostic":
        return [row.T_rad for row in rows], (
            "gate-OFF raw-energy diagnostic only; not a production color contract"
        )
    raise ContractError(f"unknown final contract {contract!r}")


def state_payload(name: str, temperature_role: str, semantics: str,
                  rows: list[PlasmaRow], temperatures: list[float],
                  original_temperatures: list[float]) -> dict[str, Any]:
    if len(temperatures) != len(rows):
        raise ContractError(f"{name}: temperature length mismatch")
    shells: list[dict[str, Any]] = []
    for row, temperature, original in zip(rows, temperatures, original_temperatures):
        finite_positive(row.W, f"{name}:s{row.shell_id}:W")
        finite_positive(temperature, f"{name}:s{row.shell_id}:temperature")
        delta = temperature - original
        ratio = temperature / original
        bolometric = row.W * SIGMA_SB * temperature**4 / math.pi
        original_bolometric = row.W * SIGMA_SB * original**4 / math.pi
        band_values: dict[str, Any] = {}
        for band_name, lo_a, hi_a in BANDS_A:
            pivot_a = math.sqrt(lo_a * hi_a)
            pivot_frequency = C_LIGHT_A_S / pivot_a
            wbnu = row.W * planck_bnu_scalar(temperature, pivot_frequency)
            original_wbnu = row.W * planck_bnu_scalar(original, pivot_frequency)
            integral = planck_band_integral(temperature, row.W, lo_a, hi_a)
            original_integral = planck_band_integral(original, row.W, lo_a, hi_a)
            band_values[band_name] = {
                "lambda_lo_A": lo_a,
                "lambda_hi_A": hi_a,
                "geometric_pivot_A": pivot_a,
                "W_Bnu_at_pivot": wbnu,
                "W_Bnu_at_pivot_ratio_to_deck_original": wbnu / original_wbnu,
                "integral_W_Bnu_dnu": integral,
                "integral_ratio_to_deck_original": integral / original_integral,
            }
        shells.append({
            "shell_id": row.shell_id,
            "W": row.W,
            "temperature_K": temperature,
            "delta_T_vs_deck_original_K": delta,
            "temperature_ratio_to_deck_original": ratio,
            "relative_change_vs_deck_original_percent": 100.0 * (ratio - 1.0),
            "initial_T_e_ratio_1p0_K": temperature,
            "initial_T_e_ratio_0p9_K": 0.9 * temperature,
            "bolometric_integral_W_Bnu_dnu": bolometric,
            "bolometric_ratio_to_deck_original": bolometric / original_bolometric,
            "radiation_energy_density_4pi_over_c_integral": FOUR_PI_OVER_C * bolometric,
            "bands": band_values,
        })
    return {
        "name": name,
        "temperature_role": temperature_role,
        "semantics": semantics,
        "temperature_unique_count_exact_binary64": len({value.hex() for value in temperatures}),
        "temperature_unique_count_at_1e-6_K": tolerance_unique_count(
            temperatures, UNIQUE_TOLERANCE_K
        ),
        "W_unique_count_exact_binary64": len({row.W.hex() for row in rows}),
        "shells": shells,
    }


def rounded_comparison(name: str, observed: float, expected: float, digits: int,
                       shell: int | None = None, expected_shell: int | None = None) -> dict[str, Any]:
    observed_rounded = round(observed, digits)
    expected_rounded = round(expected, digits)
    match = observed_rounded == expected_rounded
    if expected_shell is not None:
        match = match and shell == expected_shell
    return {
        "metric": name,
        "observed": observed,
        "observed_rounded": observed_rounded,
        "expected": expected,
        "expected_rounded": expected_rounded,
        "decimal_places": digits,
        "observed_shell": shell,
        "expected_shell": expected_shell,
        "match": match,
    }


def gate_metric_comparison(original: list[float], gate_on: list[float]) -> dict[str, Any]:
    deltas = [after - before for before, after in zip(original, gate_on)]
    ratios = [after / before for before, after in zip(original, gate_on)]
    relative = [100.0 * (ratio - 1.0) for ratio in ratios]
    changed = [index for index, delta in enumerate(deltas) if delta != 0.0]
    max_abs_shell = max(range(len(deltas)), key=lambda index: abs(deltas[index]))
    max_ratio_shell = max(range(len(ratios)), key=ratios.__getitem__)
    max_rel_shell = max(range(len(relative)), key=relative.__getitem__)

    exp_abs = EXPECTED_GATE_METRICS["maximum_absolute_change"]
    exp_ratio = EXPECTED_GATE_METRICS["maximum_temperature_ratio"]
    exp_rel = EXPECTED_GATE_METRICS["maximum_relative_increase_percent"]
    exp_median = EXPECTED_GATE_METRICS["median_relative_increase_percent"]
    comparisons = [
        {
            "metric": "changed_shells",
            "observed": changed,
            "expected": EXPECTED_GATE_METRICS["changed_shells"],
            "match": changed == EXPECTED_GATE_METRICS["changed_shells"],
        },
        rounded_comparison(
            "maximum_absolute_change_K", abs(deltas[max_abs_shell]), float(exp_abs["value_K"]),
            int(exp_abs["digits"]), max_abs_shell, int(exp_abs["shell"]),
        ),
        rounded_comparison(
            "maximum_temperature_ratio", ratios[max_ratio_shell], float(exp_ratio["value"]),
            int(exp_ratio["digits"]), max_ratio_shell, int(exp_ratio["shell"]),
        ),
        rounded_comparison(
            "maximum_relative_increase_percent", relative[max_rel_shell], float(exp_rel["value"]),
            int(exp_rel["digits"]), max_rel_shell, int(exp_rel["shell"]),
        ),
        rounded_comparison(
            "median_relative_increase_percent", statistics.median(relative),
            float(exp_median["value"]), int(exp_median["digits"]),
        ),
    ]
    for label, expected in EXPECTED_GATE_METRICS["bolometric_ratio"].items():
        shell = int(label[1:])
        comparisons.append(rounded_comparison(
            f"bolometric_ratio_{label}", ratios[shell] ** 4, float(expected["value"]),
            int(expected["digits"]), shell, shell,
        ))
    return {
        "comparison_rule": (
            "compare at the decimal precision preregistered in the order; if different, "
            "retain both observed and expected values and fail without changing the expectation"
        ),
        "comparisons": comparisons,
        "pass": all(item["match"] for item in comparisons),
        "H_scale_context": {
            "maximum_fractional_increase": max(relative) / 100.0,
            "H_reference": 1.2e-5,
            "ratio": (max(relative) / 100.0) / 1.2e-5,
            "use": "scale comparison only, never an allowance",
        },
    }


# Every direct persistent-field access is enumerated.  Ranges are intentionally
# fail-closed: a new or moved access becomes "UNCLASSIFIED" and fails the census.
CONSUMER_RANGES: dict[str, list[tuple[int, int, str, str]]] = {
    "lumina_atomic.c": [
        (600, 645, "input_owner", "CSV load and LUMINA_TRAD_COLOR_FIX load-time overwrite"),
        (646, 700, "seed", "initial opacity electron-temperature seed from T_rad"),
        (830, 870, "lifecycle", "persistent field destruction"),
    ],
    "lumina_plasma.c": [
        (840, 1000, "owner_update", "radiation-field estimator/fixed profile mutates W,T_rad; dump"),
        (2000, 2160, "Boltzmann_partition", "partition functions and diluted Boltzmann weights"),
        (2380, 2750, "rate", "nebular/Saha ionization balance using W,T_rad"),
        (2751, 2940, "opacity", "Sobolev lower/upper populations and opacity"),
        (2941, 3100, "seed_rate", "ratio*T_rad T_e seed and W*T_rad^4 energy coupling"),
        (3650, 5000, "transition_probability", "macro-atom rates and W*Bnu(T_rad) radiative pump"),
        (7150, 7550, "rate_Boltzmann", "bound-free population/rate fallback at W,T_rad"),
        (7800, 7980, "emissivity", "Planck(T_rad) packet re-emission sampling"),
        (10150, 10500, "rate", "stage/photoionization rate and W depth gate"),
        (10650, 11050, "rate", "pump/temperature fallback consumers"),
        (11500, 12360, "seed_radeq", "pre-NLTE fallback plus radiative-equilibrium rates"),
        (12361, 12900, "rate", "ionization/recombination rate closures"),
        (13500, 14400, "rate_radeq", "simultaneous/coupled radiative-equilibrium solve"),
        (14600, 14850, "opacity_rate", "super-Planckian UV Jnu cap comparator"),
        (14851, 15300, "rate", "NLTE rate assembly and dilute photospheric field"),
        (16350, 16800, "Boltzmann_diagnostic", "rate dump and isolated-level Boltzmann anchors"),
        (17050, 17300, "rate", "downstream NLTE rate consumer"),
        (17700, 17950, "rate", "downstream rate/temperature fallback"),
        (17951, 18150, "opacity", "downstream opacity consumer"),
        (18250, 18950, "formal_transfer", "continuum/line/electron-scattering source W*Bnu(T_rad)"),
    ],
    "lumina_cuda.cu": [
        (100, 180, "GPU_lifecycle", "device T_rad field declaration"),
        (250, 370, "GPU_lifecycle", "device T_rad allocation contract"),
        (480, 570, "GPU_transfer", "persistent T_rad upload to transport device state"),
        (1350, 1750, "opacity_rate", "GPU-side opacity/rate temperature fallbacks"),
        (1900, 2150, "opacity_rate", "GPU line/radiation consumers"),
        (2400, 2520, "GPU_transfer", "per-iteration W upload contract for GPU transport"),
        (3200, 3340, "GPU_lifecycle", "device T_rad destruction"),
        (3650, 3850, "GPU_emissivity", "device Planck(T_rad) frequency sampling"),
        (5250, 5820, "GPU_transport", "packet interaction consumers of device T_rad"),
        (5900, 6320, "GPU_transport", "transport kernel device T_rad plumbing"),
        (6400, 6620, "GPU_transport", "packet-loop device T_rad consumer"),
        (8400, 8650, "rate", "W depth-tier gate"),
        (8750, 8920, "GPU_transport", "transport launch passes device T_rad"),
        (9900, 10100, "diagnostic", "iteration state capture"),
        (10150, 10320, "GPU_transport", "formal transport launch passes device T_rad"),
        (10700, 10950, "comparator", "TARDIS W,T_rad comparison and banner"),
        (10951, 11100, "output", "final persistent plasma-state output"),
    ],
    "lumina_main.c": [
        (300, 380, "diagnostic", "per-iteration state output"),
        (580, 690, "owner_validation", "temporary reference override and restore"),
        (700, 800, "comparator", "TARDIS W,T_rad comparison"),
        (830, 880, "output", "final plasma-state output"),
    ],
    "lumina_nlte_assemble.cu": [
        (1, 520, "GPU_rate", "device W lifecycle, radiative-rate consumption, and T_rad color anchor"),
    ],
    "lumina_bf_gemm.cu": [
        (1, 420, "GPU_opacity_rate", "device T_rad,W lifecycle and bound-free GEMM consumers"),
    ],
    "lumina_element_wide.c": [
        (2250, 2380, "diagnostic", "element-wide input/provenance dump"),
    ],
    "lumina_cmfgen.c": [
        (620, 1010, "rate_diagnostic", "hot-regime comparator and field dump"),
        (1550, 1660, "diagnostic", "T_rad state hash"),
        (2080, 2180, "rate", "hot-regime rate selector"),
    ],
}


PERSISTENT_FIELD_RE = re.compile(
    r"\b(?:plasma|ps)(?:->|\.)\s*(?:T_rad|W)\b|\bd_T_rad\b|\bd_W\b"
)


def consumer_census(repo_root: Path) -> dict[str, Any]:
    src = repo_root / "src"
    if not src.is_dir():
        raise ContractError(f"consumer census cannot find {src}")
    entries: list[dict[str, Any]] = []
    unclassified: list[dict[str, Any]] = []
    suffixes = {".c", ".cu", ".h"}
    for path in sorted(item for item in src.iterdir() if item.suffix in suffixes):
        relative = str(path.relative_to(repo_root))
        rules = CONSUMER_RANGES.get(path.name, [])
        for line_number, line in enumerate(path.read_text(errors="replace").splitlines(), start=1):
            if not PERSISTENT_FIELD_RE.search(line):
                continue
            role = "UNCLASSIFIED"
            meaning = "new or moved direct persistent-field access requires review"
            for lo, hi, candidate_role, candidate_meaning in rules:
                if lo <= line_number <= hi:
                    role, meaning = candidate_role, candidate_meaning
                    break
            entry = {
                "location": f"{relative}:{line_number}",
                "role": role,
                "meaning": meaning,
                "source": line.strip(),
            }
            entries.append(entry)
            if role == "UNCLASSIFIED":
                unclassified.append(entry)
    required_roles = {
        "seed": any("seed" in entry["role"] for entry in entries),
        "Boltzmann_or_partition": any(
            "Boltzmann" in entry["role"] for entry in entries
        ),
        "rate": any("rate" in entry["role"] for entry in entries),
        "opacity": any("opacity" in entry["role"] for entry in entries),
        "transition_probability": any(
            entry["role"] == "transition_probability" for entry in entries
        ),
        "comparator": any("comparator" in entry["role"] for entry in entries),
    }
    passed = bool(entries) and not unclassified and all(required_roles.values())
    return {
        "method": (
            "lexical census of every direct plasma/ps persistent T_rad or W access and "
            "every device d_T_rad/d_W access in src/*.{c,cu,h}; no source is imported or executed"
        ),
        "entry_count": len(entries),
        "required_meaning_coverage": required_roles,
        "unclassified_count": len(unclassified),
        "pass": passed,
        "entries": entries,
    }


def parse_rvtj_block(path: Path, label: str, count: int) -> np.ndarray:
    lines = path.read_text(errors="replace").splitlines()
    for index, line in enumerate(lines):
        if line.strip() != label:
            continue
        values: list[float] = []
        for candidate in lines[index + 1:]:
            if len(values) >= count:
                break
            try:
                values.extend(float(token.replace("D", "E")) for token in candidate.split())
            except ValueError:
                break
        if len(values) < count:
            raise ContractError(
                f"{path}: block {label!r} has {len(values)} values; expected {count}"
            )
        result = np.asarray(values[:count], dtype=np.float64)
        if not np.isfinite(result).all():
            raise ContractError(f"{path}: block {label!r} contains NaN/Inf")
        return result
    raise ContractError(f"{path}: missing RVTJ block {label!r}")


def parse_eddfactor_info(path: Path) -> dict[str, Any]:
    lines = path.read_text(errors="strict").splitlines()
    if len(lines) < 3:
        raise ContractError(f"{path}: truncated EDDFACTOR_INFO")
    tokens = lines[2].split()
    if len(tokens) < 6:
        raise ContractError(f"{path}: malformed layout row")
    try:
        nd, recl, word_size, unit_size, int_size = map(int, tokens[:5])
    except ValueError as exc:
        raise ContractError(f"{path}: non-integer layout: {exc}") from exc
    little_endian = tokens[5].upper().startswith("T")
    if recl % word_size:
        raise ContractError(f"{path}: RECL is not divisible by WORD_SIZE")
    return {
        "ND": nd,
        "RECL": recl,
        "WORD_SIZE": word_size,
        "UNIT_SIZE": unit_size,
        "INT_SIZE": int_size,
        "little_endian": little_endian,
        "words_per_record": recl // word_size,
    }


def log_velocity_interpolate(jnu_by_frequency_depth: np.ndarray, velocities: np.ndarray,
                             target_velocity: float) -> tuple[np.ndarray, dict[str, Any]]:
    order = np.argsort(velocities)
    sorted_velocity = velocities[order]
    if not (sorted_velocity[0] <= target_velocity <= sorted_velocity[-1]):
        raise ContractError(
            f"target velocity {target_velocity} outside RVTJ "
            f"[{sorted_velocity[0]}, {sorted_velocity[-1]}]"
        )
    right = int(np.searchsorted(sorted_velocity, target_velocity, side="left"))
    if right == 0:
        left = right = 0
        fraction = 0.0
    elif right == len(sorted_velocity):
        left = right = len(sorted_velocity) - 1
        fraction = 0.0
    elif sorted_velocity[right] == target_velocity:
        left = right
        fraction = 0.0
    else:
        left = right - 1
        fraction = (
            (target_velocity - sorted_velocity[left]) /
            (sorted_velocity[right] - sorted_velocity[left])
        )
    a = np.asarray(jnu_by_frequency_depth[:, order[left]], dtype=np.float64)
    b = np.asarray(jnu_by_frequency_depth[:, order[right]], dtype=np.float64)
    if np.any(a <= 0.0) or np.any(b <= 0.0) or not np.isfinite(a).all() or not np.isfinite(b).all():
        bad = int(np.sum((a <= 0.0) | (b <= 0.0) | ~np.isfinite(a) | ~np.isfinite(b)))
        raise ContractError(
            f"cannot log-J interpolate target {target_velocity}: {bad} invalid endpoints"
        )
    if left == right:
        result = a.copy()
    else:
        result = np.exp((1.0 - fraction) * np.log(a) + fraction * np.log(b))
    return result, {
        "left_depth_zero_based": int(order[left]),
        "right_depth_zero_based": int(order[right]),
        "left_velocity_km_s": float(sorted_velocity[left]),
        "right_velocity_km_s": float(sorted_velocity[right]),
        "fraction": float(fraction),
        "interpolation": "linear in velocity, logarithmic in J_nu at each frequency",
    }


def amplitude_free_planck_fit(frequency: np.ndarray, jnu: np.ndarray,
                              temperature_lo: float = 1000.0,
                              temperature_hi: float = 100000.0) -> dict[str, float]:
    """Least-squares shape fit in ln J; ln W is eliminated analytically."""
    if np.any(jnu <= 0.0) or not np.isfinite(jnu).all():
        raise ContractError("Planck shape fit received nonpositive/NaN/Inf J_nu")
    log_j = np.log(jnu)

    def objective(log_temperature: float, with_residual: bool = False) -> Any:
        temperature = math.exp(log_temperature)
        log_b = log_planck_bnu_array(temperature, frequency)
        log_w = float(np.mean(log_j - log_b))
        residual = log_j - (log_w + log_b)
        mse = float(np.mean(residual * residual))
        if with_residual:
            return mse, log_w, residual
        return mse

    # Golden-section minimization in ln(T); fixed bounds are part of the output.
    a = math.log(temperature_lo)
    b = math.log(temperature_hi)
    invphi = (math.sqrt(5.0) - 1.0) / 2.0
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc, fd = objective(c), objective(d)
    for _ in range(48):
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = objective(c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = objective(d)
    log_temperature = 0.5 * (a + b)
    mse, log_w, residual = objective(log_temperature, with_residual=True)
    temperature = math.exp(log_temperature)
    dilution = math.exp(log_w)
    model = dilution * planck_bnu_array(temperature, frequency)
    relative_l2 = float(np.linalg.norm(jnu - model) / np.linalg.norm(jnu))
    return {
        "T_color_K": temperature,
        "W": dilution,
        "ln_residual_rms": math.sqrt(mse),
        "ln_residual_abs_p95": float(np.quantile(np.abs(residual), 0.95)),
        "linear_relative_L2": relative_l2,
        "temperature_search_lo_K": temperature_lo,
        "temperature_search_hi_K": temperature_hi,
        "definition": (
            "minimize mean[(ln J_nu - ln W - ln B_nu(T))^2]; "
            "ln W is eliminated as the mean log-amplitude residual"
        ),
    }


def trapz_band(frequency: np.ndarray, values: np.ndarray, wavelength_a: np.ndarray,
               lambda_lo_a: float, lambda_hi_a: float) -> float:
    mask = (wavelength_a >= lambda_lo_a) & (wavelength_a <= lambda_hi_a)
    if int(mask.sum()) < 2:
        raise ContractError(
            f"CMFGEN grid has fewer than two samples in {lambda_lo_a}-{lambda_hi_a} A"
        )
    result = float(np.trapezoid(values[mask], frequency[mask]))
    return finite_positive(result, f"CMFGEN {lambda_lo_a}-{lambda_hi_a} A integral")


def cmfgen_comparison(cmfgen_dir: Path, geometry: list[GeometryRow],
                      deck_rows: list[PlasmaRow]) -> dict[str, Any]:
    paths = {name: cmfgen_dir / name for name in EXPECTED_CMFGEN_SHA256}
    for path in paths.values():
        if not path.is_file():
            raise ContractError(f"missing CMFGEN input {path}")
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    for name, expected in EXPECTED_CMFGEN_SHA256.items():
        if hashes[name] != expected:
            raise ContractError(
                f"CMFGEN {name} SHA-256 mismatch: observed={hashes[name]} expected={expected}"
            )

    info = parse_eddfactor_info(paths["EDDFACTOR_INFO"])
    if info["ND"] != EXPECTED_CMFGEN_ND:
        raise ContractError(f"CMFGEN ND={info['ND']}, expected {EXPECTED_CMFGEN_ND}")
    if info["WORD_SIZE"] != 8 or info["words_per_record"] < info["ND"] + 1:
        raise ContractError("CMFGEN EDDFACTOR record cannot hold ND J values plus frequency")
    file_size = paths["EDDFACTOR"].stat().st_size
    if file_size % info["RECL"]:
        raise ContractError("EDDFACTOR byte size is not an integer number of RECL records")
    record_count = file_size // info["RECL"]
    dtype = "<f8" if info["little_endian"] else ">f8"
    raw = np.memmap(
        paths["EDDFACTOR"], dtype=dtype, mode="r",
        shape=(record_count, info["words_per_record"]),
    )
    finish = float(raw[4, 0])
    if not math.isfinite(finish) or finish != 1.0:
        raise ContractError(f"EDDFACTOR FINISH={finish!r}, expected exactly 1")
    data = raw[14:]
    nd = int(info["ND"])
    finite_j = np.isfinite(data[:, :nd]).all(axis=1)
    frequency_column = np.asarray(data[:, nd], dtype=np.float64)
    good = finite_j & np.isfinite(frequency_column) & (frequency_column > 0.0)
    good_count = int(np.sum(good))
    if good_count != EXPECTED_CMFGEN_VALID_RECORDS:
        raise ContractError(
            f"EDDFACTOR valid records={good_count}, expected {EXPECTED_CMFGEN_VALID_RECORDS}"
        )
    jnu = data[good, :nd]
    fl_1e15 = frequency_column[good]
    frequency = fl_1e15 * 1.0e15
    roundtrip_fl = frequency / 1.0e15
    fl_roundtrip_max_rel = float(np.max(np.abs(roundtrip_fl - fl_1e15) / fl_1e15))
    wavelength_a_direct = C_LIGHT_A_S / frequency
    wavelength_a_native = 2997.92458 / fl_1e15
    wavelength_roundtrip_max_rel = float(np.max(
        np.abs(wavelength_a_direct - wavelength_a_native) / wavelength_a_native
    ))
    order = np.argsort(frequency)
    frequency = np.asarray(frequency[order], dtype=np.float64)
    wavelength_a = np.asarray(wavelength_a_direct[order], dtype=np.float64)
    jnu = jnu[order, :]
    if np.any(np.diff(frequency) < 0.0):
        raise ContractError("sorted CMFGEN frequency grid is not monotone")

    velocity = parse_rvtj_block(paths["RVTJ"], "Velocity (km/s)", nd)
    covered = [
        row.shell_id for row in geometry
        if float(np.min(velocity)) <= row.velocity_midpoint_km_s <= float(np.max(velocity))
    ]
    uncovered = [row.shell_id for row in geometry if row.shell_id not in covered]
    if covered != EXPECTED_CMFGEN_COVERED_SHELLS:
        raise ContractError(
            f"CMFGEN coverage is {covered}; expected exactly s0-s43 without hold/clamp"
        )

    gate_on = reconstruct_gate(deck_rows, True)
    rows_out: list[dict[str, Any]] = []
    for geometry_row in geometry:
        shell = geometry_row.shell_id
        if shell not in covered:
            rows_out.append({
                "shell_id": shell,
                "velocity_midpoint_km_s": geometry_row.velocity_midpoint_km_s,
                "status": "UNAVAILABLE_OUTSIDE_RVTJ",
                "value": None,
                "policy": "no hold-last, nearest-depth clamp, or extrapolation",
            })
            continue
        field, bracket = log_velocity_interpolate(
            jnu, velocity, geometry_row.velocity_midpoint_km_s
        )
        integral_j = finite_positive(
            float(np.trapezoid(field, frequency)), f"CMFGEN s{shell} integral J_nu"
        )
        integral_nu_j = finite_positive(
            float(np.trapezoid(frequency * field, frequency)),
            f"CMFGEN s{shell} integral nu*J_nu",
        )
        energy_density = FOUR_PI_OVER_C * integral_j
        energy_temperature = (energy_density / RADIATION_A) ** 0.25
        mean_frequency = integral_nu_j / integral_j
        moment_color = (
            (H_PLANCK / K_BOLTZMANN) * mean_frequency / MOMENT_PLANCK_FACTOR
        )
        fit = amplitude_free_planck_fit(frequency, field)
        deck = deck_rows[shell]
        gate = gate_on[shell]
        band_rows: dict[str, Any] = {}
        for band_name, lo_a, hi_a in BANDS_A:
            cmf_integral = trapz_band(frequency, field, wavelength_a, lo_a, hi_a)
            mask = (wavelength_a >= lo_a) & (wavelength_a <= hi_a)
            deck_model = deck.W * planck_bnu_array(deck.T_rad, frequency[mask])
            gate_model = gate.W * planck_bnu_array(gate.T_rad, frequency[mask])
            deck_integral = finite_positive(
                float(np.trapezoid(deck_model, frequency[mask])),
                f"deck s{shell} {band_name} integral",
            )
            gate_integral = finite_positive(
                float(np.trapezoid(gate_model, frequency[mask])),
                f"gate s{shell} {band_name} integral",
            )
            band_rows[band_name] = {
                "CMFGEN_integral_Jnu_dnu": cmf_integral,
                "deck_integral_W_Bnu_dnu": deck_integral,
                "deck_over_CMFGEN": deck_integral / cmf_integral,
                "gate_on_integral_W_Bnu_dnu": gate_integral,
                "gate_on_over_CMFGEN": gate_integral / cmf_integral,
                "same_grid_definition": "trapezoid on the selected CMFGEN frequencies",
            }
        rows_out.append({
            "shell_id": shell,
            "velocity_midpoint_km_s": geometry_row.velocity_midpoint_km_s,
            "status": "MEASURED",
            "velocity_bracket": bracket,
            "CMFGEN_integral_Jnu_dnu": integral_j,
            "CMFGEN_energy_density_erg_cm3": energy_density,
            "CMFGEN_T_energy_K": energy_temperature,
            "CMFGEN_mean_frequency_Hz": mean_frequency,
            "CMFGEN_moment_T_color_K": moment_color,
            "CMFGEN_amplitude_free_Planck_fit": fit,
            "deck_T_energy_candidate_K": deck.T_rad,
            "deck_over_CMFGEN_T_energy": deck.T_rad / energy_temperature,
            "gate_on_T_color_candidate_K": gate.T_rad,
            "gate_over_CMFGEN_moment_color": gate.T_rad / moment_color,
            "gate_over_CMFGEN_fit_color": gate.T_rad / fit["T_color_K"],
            "registered_bands": band_rows,
        })

    measured = [row for row in rows_out if row["status"] == "MEASURED"]
    return {
        "status": "MEASURED_NO_ACCEPTANCE_THRESHOLD",
        "reason": (
            "the independent quantities are measurable, but the order defines no numerical "
            "closeness or fit-residual threshold for declaring a canonical scalar"
        ),
        "input_sha256": hashes,
        "layout": {
            **info,
            "file_size_bytes": file_size,
            "record_count": record_count,
            "metadata_records": 14,
            "valid_frequency_records": good_count,
            "FINISH": finish,
        },
        "unit_roundtrip": {
            "stored_frequency_unit": "10^15 Hz",
            "J_nu_unit": "erg s^-1 cm^-2 Hz^-1 sr^-1",
            "max_relative_FL_to_Hz_to_FL": fl_roundtrip_max_rel,
            "max_relative_lambda_two_formula_roundtrip": wavelength_roundtrip_max_rel,
            "lambda_formula_A": "c[A/s]/(FL*1e15) == 2997.92458/FL",
        },
        "velocity_mapping": {
            "covered_shells": covered,
            "uncovered_shells": uncovered,
            "RVTJ_velocity_min_km_s": float(np.min(velocity)),
            "RVTJ_velocity_max_km_s": float(np.max(velocity)),
            "outside_policy": "UNAVAILABLE; never hold, clamp, or extrapolate",
        },
        "definitions": {
            "energy": "u=(4*pi/c)*integral J_nu dnu; T_energy=(u/a)^0.25",
            "moment_color": (
                "T=(h/k)*<nu>/(4*zeta(5)/zeta(4)), where "
                "<nu>=integral nu*J_nu dnu/integral J_nu dnu"
            ),
            "shape_fit": (
                "amplitude-free log-space Planck fit over every valid EDDFACTOR frequency; "
                "residual is reported, not forced to pass"
            ),
        },
        "summary_s0_s43": {
            "deck_over_CMFGEN_T_energy_median": statistics.median(
                row["deck_over_CMFGEN_T_energy"] for row in measured
            ),
            "deck_over_CMFGEN_T_energy_min": min(
                row["deck_over_CMFGEN_T_energy"] for row in measured
            ),
            "deck_over_CMFGEN_T_energy_max": max(
                row["deck_over_CMFGEN_T_energy"] for row in measured
            ),
            "gate_over_CMFGEN_moment_color_median": statistics.median(
                row["gate_over_CMFGEN_moment_color"] for row in measured
            ),
            "gate_over_CMFGEN_fit_color_median": statistics.median(
                row["gate_over_CMFGEN_fit_color"] for row in measured
            ),
            "Planck_fit_ln_residual_rms_median": statistics.median(
                row["CMFGEN_amplitude_free_Planck_fit"]["ln_residual_rms"]
                for row in measured
            ),
        },
        "shells": rows_out,
    }


def contradiction_record(rows: list[PlasmaRow], config: dict[str, Any],
                         inferred_color: list[float]) -> dict[str, Any]:
    t_inner = float(config["T_inner_K"])
    gate_copy = rows[0].T_rad
    builder_s0 = t_inner * rows[0].W ** 0.25
    file_relative_mismatch = 100.0 * abs(gate_copy - builder_s0) / gate_copy
    return {
        "deck_T_energy_over_W_quarter": {
            "minimum_K": min(inferred_color),
            "maximum_K": max(inferred_color),
            "range_K": max(inferred_color) - min(inferred_color),
            "unique_count_at_1e-6_K": tolerance_unique_count(
                inferred_color, UNIQUE_TOLERANCE_K
            ),
        },
        "gate_copy_K": gate_copy,
        "config_T_inner_K": t_inner,
        "config_builder_formula_s0_K": builder_s0,
        "file_vs_config_builder_s0_percent_denominator_file": file_relative_mismatch,
        "expected_29p3_percent_at_one_decimal": round(file_relative_mismatch, 1) == 29.3,
        "verdict": (
            "three distinct values; lineage/semantic contract defect, not an approximate match"
        ),
    }


def verify(deck_dir: Path, gate_enabled: bool, final_contract: str,
           cmfgen_dir: Path | None = None, require_frozen_hashes: bool = True) -> dict[str, Any]:
    rows, config, geometry, hashes = load_deck(deck_dir, require_frozen_hashes)
    original_temperatures = [row.T_rad for row in rows]
    gate_state = reconstruct_gate(rows, gate_enabled)
    gate_temperatures = [row.T_rad for row in gate_state]
    inferred_color = [
        row.T_rad / row.W ** 0.25
        for row in rows
    ]
    for index, value in enumerate(inferred_color):
        finite_positive(value, f"inferred color s{index}")
    final_temperatures, final_semantics = select_final_temperatures(
        final_contract, rows, config, inferred_color
    )

    states = [
        state_payload(
            "deck_original_energy", "T_rad column interpreted as T_energy",
            "frozen deck W,T_rad before any gate", rows, original_temperatures,
            original_temperatures,
        ),
        state_payload(
            "gate_independent_reconstruction", "runtime T_rad after current load-time gate",
            f"gate {'ON: copy s0 T_rad to s1-s49' if gate_enabled else 'OFF: identity'}; W invariant",
            gate_state, gate_temperatures, original_temperatures,
        ),
        state_payload(
            "deck_inferred_color", "T_color=T_energy/W^0.25",
            "mechanical inverse candidate, not by itself a provenance or canonicality decision",
            rows, inferred_color, original_temperatures,
        ),
        state_payload(
            "selected_final_contract", "T_color",
            f"selection={final_contract}; {final_semantics}", rows, final_temperatures,
            original_temperatures,
        ),
    ]

    if gate_enabled:
        expectation = gate_metric_comparison(original_temperatures, gate_temperatures)
        gate_invariant_pass = expectation["pass"]
    else:
        identity_rows = [
            before.shell_id == after.shell_id
            and before.W.hex() == after.W.hex()
            and before.T_rad.hex() == after.T_rad.hex()
            for before, after in zip(rows, gate_state)
        ]
        gate_invariant_pass = all(identity_rows)
        expectation = {
            "mode": "gate_OFF_positive_control",
            "all_50_rows_binary64_identical": gate_invariant_pass,
            "per_shell_identity": identity_rows,
            "pass": gate_invariant_pass,
        }

    census = consumer_census(REPO_ROOT)
    report: dict[str, Any] = {
        "schema": "TRAD-FIX-independent-verifier-v1",
        "verifier": str(SCRIPT_PATH),
        "independence_contract": {
            "production_builder_imported_or_executed": False,
            "Lumina_solver_imported_or_executed": False,
            "existing_CMFGEN_parser_imported": False,
            "inputs_read_directly": ["CSV", "JSON", "source text", "optional EDDFACTOR/RVTJ"],
        },
        "input": {
            "deck_dir": str(deck_dir.resolve()),
            "sha256": hashes,
            "frozen_hashes_required": require_frozen_hashes,
            "shell_count": len(rows),
        },
        "gate": {
            "environment_contract_name": "LUMINA_TRAD_COLOR_FIX",
            "requested_state": "ON" if gate_enabled else "OFF",
            "independent_reconstruction": True,
            "W_binary64_invariant": all(
                before.W.hex() == after.W.hex() for before, after in zip(rows, gate_state)
            ),
        },
        "bands": [
            {"name": name, "lambda_lo_A": lo, "lambda_hi_A": hi}
            for name, lo, hi in BANDS_A
        ],
        "state_output_location": "states[*].shells[s0..s49] in this JSON record",
        "states": states,
        "preregistered_expectation_comparison": expectation,
        "three_way_contradiction": contradiction_record(rows, config, inferred_color),
        "persistent_consumer_census": census,
        "cmfgen_independent_comparison": (
            cmfgen_comparison(cmfgen_dir, geometry, rows)
            if cmfgen_dir is not None
            else {
                "status": "NOT_RUN",
                "reason": "supply --cmfgen-dir on lageunha for the 143 MB field pass",
            }
        ),
    }
    report["verifier_verdict"] = "PASS" if (
        gate_invariant_pass
        and report["gate"]["W_binary64_invariant"]
        and census["pass"]
    ) else "FAIL"
    report["TRAD_FIX"] = {
        "status": "UNRESOLVED",
        "reason": (
            "this verifier measures candidates but neither approves a disposition nor selects "
            "10020/10470.093/14172.549 K as the physical canonical contract"
        ),
    }
    return report


def print_summary(report: dict[str, Any], stream: Any = sys.stdout) -> None:
    print(f"TRAD-FIX VERIFIER: {report['verifier_verdict']}", file=stream)
    print(
        f"TRAD_FIX status={report['TRAD_FIX']['status']} reason={report['TRAD_FIX']['reason']}",
        file=stream,
    )
    print(
        "INPUT plasma_sha256="
        f"{report['input']['sha256']['plasma_state.csv']} "
        f"W_digest={report['input']['sha256']['W_binary64_digest']}",
        file=stream,
    )
    print(
        f"GATE state={report['gate']['requested_state']} "
        f"W_invariant={report['gate']['W_binary64_invariant']}",
        file=stream,
    )
    for state in report["states"]:
        first = state["shells"][0]
        last = state["shells"][-1]
        print(
            f"STATE {state['name']}: unique_T={state['temperature_unique_count_at_1e-6_K']} "
            f"s0=({first['W']:.12g},{first['temperature_K']:.12f}) "
            f"s49=({last['W']:.12g},{last['temperature_K']:.12f})",
            file=stream,
        )
    expectation = report["preregistered_expectation_comparison"]
    print(f"EXPECTATION comparison_pass={expectation['pass']}", file=stream)
    for item in expectation.get("comparisons", []):
        print(
            f"  {item['metric']}: observed={item['observed']} "
            f"expected={item['expected']} match={item['match']}",
            file=stream,
        )
    contradiction = report["three_way_contradiction"]
    inverse = contradiction["deck_T_energy_over_W_quarter"]
    print(
        "CONTRADICTION "
        f"inverse_color=[{inverse['minimum_K']:.12f},{inverse['maximum_K']:.12f}] "
        f"gate={contradiction['gate_copy_K']:.12f} "
        f"config={contradiction['config_T_inner_K']:.12f} "
        f"config_builder_s0={contradiction['config_builder_formula_s0_K']:.12f} "
        f"mismatch={contradiction['file_vs_config_builder_s0_percent_denominator_file']:.6f}%",
        file=stream,
    )
    census = report["persistent_consumer_census"]
    print(
        f"CONSUMERS persistent_accesses={census['entry_count']} "
        f"unclassified={census['unclassified_count']} pass={census['pass']}",
        file=stream,
    )
    cmfgen = report["cmfgen_independent_comparison"]
    print(f"CMFGEN status={cmfgen['status']}", file=stream)


def write_json(report: dict[str, Any], output: str | None) -> None:
    if output == "-" or output is None:
        json.dump(report, sys.stdout, indent=2, sort_keys=True, allow_nan=False)
        sys.stdout.write("\n")
        return
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(f"JSON {path.resolve()}")


def self_test(deck_dir: Path) -> int:
    with tempfile.TemporaryDirectory(prefix="trad_fix_fixture_") as tmp:
        fixture = Path(tmp)
        for name in EXPECTED_FILE_SHA256:
            shutil.copyfile(deck_dir / name, fixture / name)

        original = read_plasma(fixture / "plasma_state.csv")
        damaged_rows = list(original)
        target = damaged_rows[7]
        damaged_rows[7] = PlasmaRow(target.shell_id, target.W * 1.01, target.T_rad)
        independently_expresses_w_defect = (
            damaged_rows[7].W.hex() != original[7].W.hex()
            and damaged_rows[7].T_rad.hex() == original[7].T_rad.hex()
            and all(
                damaged_rows[index] == original[index]
                for index in range(len(original)) if index != 7
            )
        )
        print(
            "FIXTURE_CAPABILITY "
            f"{'PASS' if independently_expresses_w_defect else 'FAIL'}: "
            "s7 W can change independently while T_rad and the other 49 rows remain identical"
        )
        if not independently_expresses_w_defect:
            return 1

        positive = verify(
            fixture, gate_enabled=False, final_contract="deck-inferred-color",
            cmfgen_dir=None, require_frozen_hashes=True,
        )
        positive_ok = (
            positive["verifier_verdict"] == "PASS"
            and positive["preregistered_expectation_comparison"][
                "all_50_rows_binary64_identical"
            ]
        )
        print(
            "POSITIVE_GATE_OFF "
            f"{'PASS' if positive_ok else 'FAIL'}: all 50 W,T_rad rows reproduced bit-for-bit"
        )
        if not positive_ok:
            return 1

        synthetic_frequency = np.geomspace(1.0e13, 1.0e16, 1024)
        synthetic_temperature = 12000.0
        synthetic_w = 0.2
        synthetic_j = synthetic_w * planck_bnu_array(
            synthetic_temperature, synthetic_frequency
        )
        synthetic_fit = amplitude_free_planck_fit(synthetic_frequency, synthetic_j)
        interpolation_source = np.column_stack((synthetic_j, 4.0 * synthetic_j))
        interpolated, _ = log_velocity_interpolate(
            interpolation_source, np.asarray((1000.0, 3000.0)), 2000.0
        )
        cmf_kernel_ok = (
            abs(synthetic_fit["T_color_K"] / synthetic_temperature - 1.0) < 1.0e-9
            and abs(synthetic_fit["W"] / synthetic_w - 1.0) < 1.0e-9
            and synthetic_fit["ln_residual_rms"] < 1.0e-9
            and np.allclose(interpolated, 2.0 * synthetic_j, rtol=1.0e-13, atol=0.0)
        )
        print(
            "CMF_KERNEL_FIXTURE "
            f"{'PASS' if cmf_kernel_ok else 'FAIL'}: "
            f"T_fit={synthetic_fit['T_color_K']:.9f} "
            f"W_fit={synthetic_fit['W']:.12f} "
            f"ln_rms={synthetic_fit['ln_residual_rms']:.3e}"
        )
        if not cmf_kernel_ok:
            return 1

        with (fixture / "plasma_state.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("shell_id", "W", "T_rad"))
            for row in damaged_rows:
                writer.writerow((row.shell_id, repr(row.W), repr(row.T_rad)))

        negative_ok = False
        negative_message = "verifier unexpectedly accepted damaged W"
        try:
            verify(
                fixture, gate_enabled=False, final_contract="deck-inferred-color",
                cmfgen_dir=None, require_frozen_hashes=True,
            )
        except ContractError as exc:
            negative_message = str(exc)
            negative_ok = "W fingerprint mismatch" in negative_message
        print(
            "NEGATIVE_DAMAGED_W "
            f"{'PASS' if negative_ok else 'FAIL'}: {negative_message}"
        )
        print(f"SELF_TEST {'PASS' if negative_ok else 'FAIL'}")
        return 0 if negative_ok else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-dir", type=Path, default=DEFAULT_DECK)
    parser.add_argument("--gate", choices=("on", "off"), default="on")
    parser.add_argument("--final-contract", choices=FINAL_CONTRACT_CHOICES)
    parser.add_argument(
        "--cmfgen-dir", type=Path,
        help="enable the full independent EDDFACTOR/RVTJ comparison (lageunha)",
    )
    parser.add_argument(
        "--json-out", help="write the complete 50-shell JSON record; '-' writes stdout",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.self_test:
            return self_test(args.deck_dir)
        if args.final_contract is None:
            raise ContractError(
                "--final-contract is required so the fourth state cannot silently choose a canonical value"
            )
        report = verify(
            args.deck_dir,
            gate_enabled=args.gate == "on",
            final_contract=args.final_contract,
            cmfgen_dir=args.cmfgen_dir,
            require_frozen_hashes=True,
        )
        if args.json_out == "-" or args.json_out is None:
            print_summary(report, stream=sys.stderr)
        else:
            print_summary(report)
        write_json(report, args.json_out)
        return 0 if report["verifier_verdict"] == "PASS" else 1
    except (ContractError, OSError, ValueError, OverflowError) as exc:
        print(f"TRAD-FIX VERIFIER: FAIL\nERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
