#!/usr/bin/env python3
"""Offline s0 B3/B4 attribution of mapped/unmapped line-source fracture.

The real LINEPOP payload is parsed only by ``uv_t2n9_offline.parse_linepop``.
The production line assembly is replayed row by row, without bin
apportionment or substituted values.  No model, transport, or GPU code runs.
"""
from __future__ import annotations

import argparse
import csv
import io
import itertools
import json
import math
from pathlib import Path
import sys
from typing import Any, NamedTuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import uv_t2n9_offline as base  # noqa: E402


DEFAULT_CAPTURE = Path(
    "/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932"
)
SCHEMA = "lumina-scheme-fracture-s0-v1"
F_NLTE_ION = np.uint32(1 << 0)
F_SL_USED_PLANCK = np.uint32(1 << 3)
BANDS = (("B3", 2000.0, 2500.0), ("B4", 2500.0, 3000.0))
FLOAT64_EPS = float(np.finfo(np.float64).eps)
# Match the Stage 3.2 Rung-1 closure gate's guard factor.  The 64 covers two
# independently ordered float64 reductions (mapsplit and this consumer), while
# n_terms below supplies the actual operation count.  This is a roundoff bound,
# not a physical clamp/floor/cap/fallback or an empirical relative tolerance.
MAPSPLIT_ROUNDOFF_K = 64.0


class FractureError(RuntimeError):
    pass


class Arrays(NamedTuple):
    line_id: np.ndarray
    line_slot: np.ndarray
    wavelength_A: np.ndarray
    nu_l: np.ndarray
    bins: np.ndarray
    Z: np.ndarray
    ion_number: np.ndarray
    flags: np.ndarray
    tau_used: np.ndarray
    eps_l: np.ndarray
    S_l_used: np.ndarray
    w: np.ndarray
    dnu: np.ndarray
    eps_phys: int
    src_nlte: int
    electron_temperature: float


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FractureError(message)


def fraction(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0.0 else numerator / denominator


def ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0.0 else numerator / denominator


def roman(value: int) -> str:
    require(0 < value < 4000, f"Roman stage outside domain: {value}")
    table = ((1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
             (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
             (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"))
    out: list[str] = []
    left = value
    for number, token in table:
        count, left = divmod(left, number)
        out.extend([token] * count)
    return "".join(out)


def ion_label(z: int, ion: int) -> str:
    require(z > 0 and ion >= 0, f"invalid ion ({z}, {ion})")
    return f"{base.ELEMENT_SYMBOL.get(z, f'Z{z}')} {roman(ion + 1)}"


def arrays_from_linepop(linepop: base.LinePop) -> Arrays:
    shell_hits = np.flatnonzero(np.asarray(linepop.shells) == 0)
    require(shell_hits.size == 1, "LINEPOP does not contain exactly one s0 slot")
    slot = int(shell_hits[0])
    selected = np.asarray(linepop.rows["shell_slot"] == slot)
    rows = linepop.rows[selected]
    line_slot = np.asarray(rows["line_slot"], dtype=np.int64)
    lines = linepop.lines[line_slot]
    arrays = Arrays(
        line_id=np.asarray(lines["line_id"], dtype=np.int64),
        line_slot=line_slot,
        wavelength_A=np.asarray(lines["lambda_cm"], dtype=np.float64) * 1.0e8,
        nu_l=np.asarray(lines["nu_l"], dtype=np.float64),
        bins=np.asarray(lines["bin"], dtype=np.int64),
        Z=np.asarray(lines["Z"], dtype=np.int64),
        ion_number=np.asarray(lines["ion"], dtype=np.int64),
        flags=np.asarray(rows["flags"], dtype=np.uint32),
        tau_used=np.asarray(rows["tau_used"], dtype=np.float64),
        eps_l=np.asarray(rows["eps_l"], dtype=np.float64),
        S_l_used=np.asarray(rows["S_l_used"], dtype=np.float64),
        w=np.asarray(rows["w"], dtype=np.float64),
        dnu=np.asarray(linepop.dnu, dtype=np.float64),
        eps_phys=int(linepop.header["eps_phys"]),
        src_nlte=int(linepop.header["src_nlte"]),
        electron_temperature=float(linepop.shell_state[slot, 0]),
    )
    validate_arrays(arrays)
    return arrays


def validate_arrays(a: Arrays) -> None:
    n = a.line_id.size
    require(n > 0, "empty s0 row selection")
    for name in ("line_slot", "wavelength_A", "nu_l", "bins", "Z",
                 "ion_number", "flags", "tau_used", "eps_l",
                 "S_l_used", "w"):
        require(getattr(a, name).size == n, f"row length mismatch: {name}")
    require(np.unique(a.line_id).size == n,
            "s0 row selection contains duplicate line_id")
    require(np.all((a.bins >= 0) & (a.bins < a.dnu.size)),
            "line bin outside dnu")
    for name in ("wavelength_A", "nu_l", "tau_used", "S_l_used", "w", "dnu"):
        values = getattr(a, name)
        require(np.isfinite(values).all() and np.all(values >= 0.0),
                f"{name} contains negative or nonfinite values")
    require(np.all(a.nu_l > 0.0) and np.all(a.tau_used > 0.0)
            and np.all(a.dnu > 0.0), "required positive row/grid value is zero")
    require(a.eps_phys in (0, 1), f"unknown eps_phys={a.eps_phys}")
    require(a.src_nlte in (0, 1), f"unknown src_nlte={a.src_nlte}")
    if a.eps_phys:
        require(np.isfinite(a.eps_l).all() and np.all(a.eps_l > 0.0),
                "eps_l is invalid while eps_phys is active")
    require(np.all(a.Z > 0) and np.all(a.ion_number >= 0),
            "invalid line ion identity")
    require(math.isfinite(a.electron_temperature)
            and a.electron_temperature > 0.0, "invalid s0 electron temperature")


def band_mask(a: Arrays, band: str) -> np.ndarray:
    match = [item for item in BANDS if item[0] == band]
    require(len(match) == 1, f"unknown band {band}")
    _, lo, hi = match[0]
    if band == "B4":
        return (a.wavelength_A >= lo) & (a.wavelength_A <= hi)
    return (a.wavelength_A >= lo) & (a.wavelength_A < hi)


def measures(a: Arrays, *, inject_omit_eps: bool = False) -> dict[str, np.ndarray]:
    if a.eps_phys and not inject_omit_eps:
        eta = a.w * a.eps_l * a.S_l_used
    else:
        eta = a.w * a.S_l_used
    energy = eta * a.dnu[a.bins]
    eta_noeps = a.w * a.S_l_used
    energy_noeps = eta_noeps * a.dnu[a.bins]
    chi_integral = a.w * a.dnu[a.bins]
    for name, values in (("eta_l", eta), ("line_emission_energy", energy),
                         ("eta_l_noeps", eta_noeps),
                         ("line_emission_energy_noeps", energy_noeps),
                         ("chi_line_dnu_integral", chi_integral)):
        require(np.isfinite(values).all() and np.all(values >= 0.0),
                f"{name} contains negative or nonfinite values")
    return {"eta": eta, "energy": energy, "eta_noeps": eta_noeps,
            "energy_noeps": energy_noeps, "chi": a.w,
            "chi_integral": chi_integral, "dnu_row": a.dnu[a.bins]}


def sum64(values: np.ndarray, mask: np.ndarray) -> float:
    result = float(np.sum(values[mask], dtype=np.float64))
    require(math.isfinite(result) and result >= 0.0,
            "aggregate is negative or nonfinite")
    return result


def split_metrics(a: Arrays, m: dict[str, np.ndarray],
                  band: str) -> dict[str, Any]:
    selected = band_mask(a, band)
    mapped = (a.flags & F_NLTE_ION) != 0
    out: dict[str, Any] = {"shell": 0, "band": band,
                           "rows": int(np.count_nonzero(selected))}
    for key, values in (("line_emission_energy", m["energy"]),
                        ("line_emission_energy_noeps", m["energy_noeps"]),
                        ("chi_line_sum", m["chi"]),
                        ("chi_line_dnu_integral", m["chi_integral"])):
        total = sum64(values, selected)
        mapped_value = sum64(values, selected & mapped)
        unmapped_value = sum64(values, selected & ~mapped)
        out[f"{key}_total"] = total
        out[f"{key}_mapped"] = mapped_value
        out[f"{key}_unmapped"] = unmapped_value
        out[f"{key}_unmapped_fraction"] = fraction(unmapped_value, total)
    return out


def mapsplit_term_counts(a: Arrays, band: str) -> dict[str, int]:
    """Operation counts used by the float64 cross-replay error bound.

    A sum uses one budget unit per selected addend.  A reported fraction is
    made from independently accumulated numerator and denominator totals plus
    one division, so its n_terms is the sum of those two counts plus one.
    """
    selected = band_mask(a, band)
    mapped = (a.flags & F_NLTE_ION) != 0
    counts = {
        "total": int(np.count_nonzero(selected)),
        "mapped": int(np.count_nonzero(selected & mapped)),
        "unmapped": int(np.count_nonzero(selected & ~mapped)),
    }
    return {
        "line_emission_energy_total": counts["total"],
        "line_emission_energy_mapped": counts["mapped"],
        "line_emission_energy_unmapped": counts["unmapped"],
        "line_emission_energy_unmapped_fraction": (
            counts["unmapped"] + counts["total"] + 1),
        "line_emission_energy_noeps_total": counts["total"],
        "line_emission_energy_noeps_unmapped": counts["unmapped"],
        "line_emission_energy_noeps_unmapped_fraction": (
            counts["unmapped"] + counts["total"] + 1),
        "chi_line_sum_total": counts["total"],
        "chi_line_sum_unmapped": counts["unmapped"],
        "chi_line_sum_unmapped_fraction": (
            counts["unmapped"] + counts["total"] + 1),
    }


def read_mapsplit_rows(path: Path) -> dict[str, dict[str, float]]:
    try:
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
    except OSError as exc:
        raise FractureError(f"cannot read mapsplit evidence: {path}") from exc
    output: dict[str, dict[str, float]] = {}
    for band in ("B3", "B4"):
        selected = [row for row in rows
                    if int(row["shell"]) == 0 and row["band"] == band]
        require(len(selected) == 1,
                f"mapsplit evidence lacks unique s0 {band} row")
        output[band] = {key: float(value) for key, value in selected[0].items()
                        if key not in ("shell", "band")}
    return output


def verify_mapsplit(observed: dict[str, dict[str, float]],
                    calculated: dict[str, dict[str, Any]],
                    term_counts: dict[str, dict[str, int]]) -> dict[str, Any]:
    mapping = {
        "line_emission_energy_total": "line_emission_energy_total",
        "line_emission_energy_mapped": "line_emission_energy_mapped",
        "line_emission_energy_unmapped": "line_emission_energy_unmapped",
        "line_emission_energy_unmapped_fraction":
            "line_emission_energy_unmapped_fraction",
        "line_emission_energy_noeps_total":
            "line_emission_energy_noeps_total",
        "line_emission_energy_noeps_unmapped":
            "line_emission_energy_noeps_unmapped",
        "line_emission_energy_noeps_unmapped_fraction":
            "line_emission_energy_noeps_unmapped_fraction",
        "chi_line_sum_total": "chi_line_sum_total",
        "chi_line_sum_unmapped": "chi_line_sum_unmapped",
        "chi_line_sum_unmapped_fraction": "chi_line_sum_unmapped_fraction",
    }
    checks: dict[str, Any] = {}
    for band in ("B3", "B4"):
        band_checks: dict[str, Any] = {}
        for csv_key, calc_key in mapping.items():
            left = observed[band][csv_key]
            right = float(calculated[band][calc_key])
            require(math.isfinite(left) and math.isfinite(right),
                    f"mapsplit comparison is nonfinite s0 {band} {csv_key}")
            n_terms = term_counts[band][csv_key]
            require(n_terms >= 0,
                    f"mapsplit comparison has invalid term count s0 {band} "
                    f"{csv_key}")
            scale = max(abs(left), abs(right))
            difference = abs(left - right)
            relative_difference = (None if scale == 0.0
                                   else difference / scale)
            relative_limit = MAPSPLIT_ROUNDOFF_K * FLOAT64_EPS * n_terms
            absolute_limit = relative_limit * scale
            require(difference <= absolute_limit,
                    f"mapsplit mismatch s0 {band} {csv_key}: "
                    f"observed={left:.17g} recomputed={right:.17g} "
                    f"abs_diff={difference:.17g} abs_limit={absolute_limit:.17g} "
                    f"rel_diff={relative_difference!r} "
                    f"rel_limit={relative_limit:.17g} n_terms={n_terms}")
            band_checks[csv_key] = {"observed": left, "recomputed": right,
                                    "exact_float_match": left == right,
                                    "absolute_difference": difference,
                                    "actual_relative_difference":
                                        relative_difference,
                                    "absolute_roundoff_limit": absolute_limit,
                                    "relative_roundoff_limit": relative_limit,
                                    "roundoff_k": MAPSPLIT_ROUNDOFF_K,
                                    "float64_epsilon": FLOAT64_EPS,
                                    "n_terms": n_terms}
        checks[band] = band_checks
    return checks


def rank_rows(a: Arrays, m: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    mapped = (a.flags & F_NLTE_ION) != 0
    output: list[dict[str, Any]] = []
    for band, lo, hi in BANDS:
        selected = band_mask(a, band) & ~mapped
        total_unmapped = sum64(m["energy"], selected)
        total_band = sum64(m["energy"], band_mask(a, band))
        indices = np.flatnonzero(selected)
        indices = np.asarray(sorted(indices, key=lambda idx: (
            -float(m["energy"][idx]), int(a.line_id[idx]))), dtype=np.int64)
        for rank, idx in enumerate(indices, 1):
            energy = float(m["energy"][idx])
            z, ion = int(a.Z[idx]), int(a.ion_number[idx])
            output.append({
                "shell": 0, "band": band, "lambda_lo_A": lo,
                "lambda_hi_A": hi, "rank": rank,
                "line_id": int(a.line_id[idx]),
                "line_slot": int(a.line_slot[idx]),
                "lambda_A": float(a.wavelength_A[idx]),
                "Z": z, "ion_number": ion,
                "spectroscopic_ion": ion_label(z, ion),
                "tau_used": float(a.tau_used[idx]),
                "eps_l": float(a.eps_l[idx]),
                "S_l_used": float(a.S_l_used[idx]),
                "w": float(a.w[idx]),
                "dnu": float(m["dnu_row"][idx]),
                "eta_l": float(m["eta"][idx]),
                "line_emission_energy": energy,
                "fraction_of_unmapped_band_energy":
                    fraction(energy, total_unmapped),
                "fraction_of_total_band_energy": fraction(energy, total_band),
            })
    require(bool(output), "no unmapped s0 B3/B4 lines")
    return output


def distribution(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {"count": 0, "zero_count": 0, "minimum": None,
                "q25": None, "median": None, "q75": None,
                "maximum": None, "mean": None}
    require(np.isfinite(values).all() and np.all(values >= 0.0),
            "distribution contains negative or nonfinite values")
    quantiles = np.quantile(values, [0.25, 0.5, 0.75])
    return {"count": int(values.size),
            "zero_count": int(np.count_nonzero(values == 0.0)),
            "minimum": float(np.min(values)), "q25": float(quantiles[0]),
            "median": float(quantiles[1]), "q75": float(quantiles[2]),
            "maximum": float(np.max(values)),
            "mean": float(np.mean(values, dtype=np.float64))}


def distribution_rows(a: Arrays, m: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    mapped = (a.flags & F_NLTE_ION) != 0
    factors = (("eps_l", a.eps_l), ("S_l_used", a.S_l_used),
               ("w", a.w), ("dnu", m["dnu_row"]),
               ("line_emission_energy", m["energy"]))
    rows: list[dict[str, Any]] = []
    for band, _, _ in BANDS:
        bmask = band_mask(a, band)
        for mapping, mmap in (("mapped", mapped), ("unmapped", ~mapped)):
            selected = bmask & mmap
            for factor, values in factors:
                rows.append({"shell": 0, "band": band, "mapping": mapping,
                             "factor": factor,
                             **distribution(values[selected])})
    return rows


def _sum_product(w: np.ndarray, factors: dict[str, np.ndarray],
                 names: tuple[str, ...]) -> float:
    product = np.array(w, dtype=np.float64, copy=True)
    for name in names:
        product *= factors[name]
    result = float(np.sum(product, dtype=np.float64))
    require(math.isfinite(result) and result >= 0.0,
            "factor product sum is negative or nonfinite")
    return result


def factor_decomposition(a: Arrays, m: dict[str, np.ndarray]) -> tuple[
        list[dict[str, Any]], dict[str, Any]]:
    """Exact ratio decomposition, permutation-averaged on the log scale.

    For any factor order, E_u/E_m = W_u/W_m times successive ratios of
    weighted conditional means.  Averaging each factor's log multiplier over
    all orders is an order-neutral Shapley-style allocation; no values are
    clipped or replaced.
    """
    mapped = (a.flags & F_NLTE_ION) != 0
    detail: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for band, _, _ in BANDS:
        bmask = band_mask(a, band)
        gmasks = {"mapped": bmask & mapped, "unmapped": bmask & ~mapped}
        active_names = ["S_l_used", "dnu"]
        if a.eps_phys:
            active_names.insert(0, "eps_l")
        factors = {"eps_l": a.eps_l, "S_l_used": a.S_l_used,
                   "dnu": m["dnu_row"]}
        wsum = {group: sum64(a.w, mask) for group, mask in gmasks.items()}
        w_multiplier = ratio(wsum["unmapped"], wsum["mapped"])
        energy_sum = {group: sum64(m["energy"], mask)
                      for group, mask in gmasks.items()}
        energy_multiplier = ratio(energy_sum["unmapped"], energy_sum["mapped"])
        log_contributions: dict[str, list[float]] = {
            name: [] for name in active_names}
        permutations = list(itertools.permutations(active_names))
        for order in permutations:
            prior: tuple[str, ...] = ()
            row: dict[str, Any] = {
                "shell": 0, "band": band,
                "factor_order": ">".join(order),
                "w_unmapped_over_mapped": w_multiplier,
            }
            for name in order:
                means: dict[str, float | None] = {}
                for group, mask in gmasks.items():
                    denominator = _sum_product(a.w[mask],
                                               {key: val[mask]
                                                for key, val in factors.items()},
                                               prior)
                    numerator = _sum_product(a.w[mask],
                                             {key: val[mask]
                                              for key, val in factors.items()},
                                             prior + (name,))
                    means[group] = ratio(numerator, denominator)
                multiplier = (None if means["mapped"] is None
                              or means["unmapped"] is None
                              else ratio(float(means["unmapped"]),
                                         float(means["mapped"])))
                row[f"{name}_unmapped_conditional_mean"] = means["unmapped"]
                row[f"{name}_mapped_conditional_mean"] = means["mapped"]
                row[f"{name}_multiplier"] = multiplier
                if multiplier is not None and multiplier > 0.0:
                    log_contributions[name].append(math.log(multiplier))
                prior += (name,)
            row["energy_unmapped_over_mapped"] = energy_multiplier
            product = w_multiplier
            if product is not None:
                for name in order:
                    value = row[f"{name}_multiplier"]
                    if value is None:
                        product = None
                        break
                    product *= value
            row["decomposition_product"] = product
            row["product_minus_energy_ratio"] = (
                None if product is None or energy_multiplier is None
                else product - energy_multiplier)
            detail.append(row)

        shapley: dict[str, float | None] = {}
        for name in ("eps_l", "S_l_used", "dnu"):
            logs = log_contributions.get(name, [])
            shapley[name] = (math.exp(math.fsum(logs) / len(logs))
                             if len(logs) == len(permutations) else None)
        exact_product = w_multiplier
        if exact_product is not None:
            for name in active_names:
                value = shapley[name]
                if value is None:
                    exact_product = None
                    break
                exact_product *= value
        candidates = {"w": w_multiplier, **shapley}
        defined = [(name, value) for name, value in candidates.items()
                   if value is not None and value > 0.0]
        dominant = (max(defined, key=lambda item: abs(math.log(float(item[1]))))[0]
                    if defined else None)
        summaries[band] = {
            "chi_w_unmapped_fraction": fraction(
                wsum["unmapped"], wsum["unmapped"] + wsum["mapped"]),
            "energy_unmapped_fraction": fraction(
                energy_sum["unmapped"],
                energy_sum["unmapped"] + energy_sum["mapped"]),
            "w_unmapped_over_mapped": w_multiplier,
            "eps_l_shapley_multiplier": shapley["eps_l"],
            "S_l_used_shapley_multiplier": shapley["S_l_used"],
            "dnu_shapley_multiplier": shapley["dnu"],
            "energy_unmapped_over_mapped": energy_multiplier,
            "shapley_product": exact_product,
            "shapley_product_minus_energy_ratio": (
                None if exact_product is None or energy_multiplier is None
                else exact_product - energy_multiplier),
            "factor_with_largest_absolute_log_multiplier": dominant,
            "interpretation": (
                "The named factor is the largest multiplicative separator on "
                "the absolute log scale; this is a ranking, not a thresholded "
                "causal claim. All factor permutations are retained in CSV."
            ),
        }
    return detail, summaries


def source_audit(a: Arrays) -> dict[str, Any]:
    fallback = (a.flags & F_SL_USED_PLANCK) != 0
    expected = base.planck(a.nu_l, a.electron_temperature)
    difference = np.abs(a.S_l_used - expected)
    relative = difference / expected
    require(np.isfinite(relative).all(), "nonfinite S_l/B(T_e) comparison")
    if a.src_nlte == 0:
        require(np.all(fallback),
                "src_nlte=0 but not every s0 row records Planck fallback")
    return {
        "header_src_nlte": a.src_nlte,
        "s0_electron_temperature_K": a.electron_temperature,
        "rows": int(a.line_id.size),
        "rows_with_S_l_used_Planck_flag": int(np.count_nonzero(fallback)),
        "all_rows_with_S_l_used_Planck_flag": bool(np.all(fallback)),
        "S_l_used_bitwise_equal_to_numpy_B_Te_rows": int(np.count_nonzero(
            a.S_l_used.view(np.uint64) == expected.view(np.uint64))),
        "max_abs_S_l_used_minus_numpy_B_Te": float(np.max(difference)),
        "max_relative_abs_S_l_used_over_numpy_B_Te_minus_one":
            float(np.max(relative)),
        "conclusion": (
            "header src_nlte=0 and every row carries the writer's Planck-"
            "fallback flag" if a.src_nlte == 0 and np.all(fallback)
            else "not every row is established as the writer's B(T_e) path"
        ),
    }


def definitions(a: Arrays) -> dict[str, Any]:
    eta = "w * eps_l * S_l_used" if a.eps_phys else "w * S_l_used"
    return {
        "mapping_predicate": "(flags & (1 << 0)) != 0",
        "unmapped_predicate": "(flags & (1 << 0)) == 0",
        "eps_phys_header": a.eps_phys,
        "src_nlte_header": a.src_nlte,
        "eta_l": eta,
        "line_emission_energy": f"({eta}) * dnu[line.bin]",
        "line_emission_energy_noeps": (
            "w * S_l_used * dnu[line.bin]; diagnostic contrast only; equals "
            "production only when eps_phys == 0"
        ),
        "chi_line": "w",
        "factor_decomposition": (
            "E_unmapped/E_mapped = (sum(w)_unmapped/sum(w)_mapped) times "
            "successive ratios of weighted conditional means for active factors; "
            "reported Shapley multipliers are geometric means over every factor "
            "order and multiply exactly (up to floating arithmetic) to the energy ratio"
        ),
        "mapsplit_cross_check_roundoff_bound": (
            "abs(observed-recomputed) <= 64 * float64_epsilon * n_terms * "
            "max(abs(observed),abs(recomputed)); n_terms is the number of "
            "summed addends, or numerator addends + denominator addends + one "
            "division for a fraction"
        ),
        "production_site": "src/lumina_cmfgen.c:792-801",
        "bin_total_apportionment": False,
        "undefined": "null in JSON and UNDEFINED in CSV when denominator is zero",
        "clamp": 0, "floor": 0, "cap": 0, "fallback": 0,
        "substitution": 0,
    }


def aggregate(a: Arrays, evidence: dict[str, dict[str, float]],
              provenance: dict[str, Any], *, inject_omit_eps: bool = False
              ) -> dict[str, Any]:
    validate_arrays(a)
    m = measures(a, inject_omit_eps=inject_omit_eps)
    calculated = {band: split_metrics(a, m, band) for band, _, _ in BANDS}
    term_counts = {band: mapsplit_term_counts(a, band)
                   for band, _, _ in BANDS}
    evidence_checks = verify_mapsplit(evidence, calculated, term_counts)
    ranks = rank_rows(a, m)
    distributions = distribution_rows(a, m)
    decomposition_rows, decomposition_summary = factor_decomposition(a, m)
    return {
        "schema": SCHEMA, "status": "PASS", "provenance": provenance,
        "definitions": definitions(a),
        "mapsplit_evidence_float64_cross_check": evidence_checks,
        "headline": calculated,
        "S_l_used_source_audit": source_audit(a),
        "factor_decomposition_summary": decomposition_summary,
        "unmapped_line_rank": ranks,
        "factor_distributions": distributions,
        "factor_decomposition_orders": decomposition_rows,
    }


def csv_scalar(value: Any) -> Any:
    return "UNDEFINED" if value is None else value


def csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    require(bool(rows), "refusing empty CSV")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: csv_scalar(row.get(key)) for key in fields})
    return stream.getvalue().encode()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True,
                       allow_nan=False) + "\n").encode()


def payloads(result: dict[str, Any]) -> dict[str, bytes]:
    return {
        "scheme_fracture_s0.json": json_bytes(result),
        "scheme_fracture_s0_line_rank.csv":
            csv_bytes(result["unmapped_line_rank"]),
        "scheme_fracture_s0_factor_distribution.csv":
            csv_bytes(result["factor_distributions"]),
        "scheme_fracture_s0_factor_decomposition.csv":
            csv_bytes(result["factor_decomposition_orders"]),
    }


def fixture_arrays() -> Arrays:
    wavelengths = np.asarray([2100.0, 2200.0, 2300.0, 2400.0,
                              2600.0, 2700.0, 2800.0, 2900.0])
    nu = base.CM_C * 1.0e8 / wavelengths
    temperature = 12000.0
    source = base.planck(nu, temperature)
    return Arrays(
        line_id=np.arange(100, 108, dtype=np.int64),
        line_slot=np.arange(8, dtype=np.int64),
        wavelength_A=wavelengths, nu_l=nu,
        bins=np.arange(8, dtype=np.int64),
        Z=np.asarray([26, 27, 28, 26, 27, 28, 26, 27], dtype=np.int64),
        ion_number=np.asarray([2, 3, 3, 2, 3, 3, 2, 3], dtype=np.int64),
        flags=np.asarray([1 | 8, 1 | 8, 8, 8, 1 | 8, 1 | 8, 8, 8],
                         dtype=np.uint32),
        tau_used=np.asarray([1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5]),
        eps_l=np.asarray([0.1, 0.2, 2.0, 4.0, 0.1, 0.2, 3.0, 5.0]),
        S_l_used=source,
        w=np.asarray([10.0, 10.0, 0.01, 0.01, 10.0, 10.0, 0.01, 0.01]),
        dnu=np.asarray([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]),
        eps_phys=1, src_nlte=0, electron_temperature=temperature,
    )


def fixture_evidence(a: Arrays) -> dict[str, dict[str, float]]:
    m = measures(a)
    output: dict[str, dict[str, float]] = {}
    for band, _, _ in BANDS:
        calc = split_metrics(a, m, band)
        output[band] = {key: float(value) for key, value in calc.items()
                        if key not in ("shell", "band", "rows")
                        and value is not None}
    return output


def self_test() -> dict[str, Any]:
    fixture = fixture_arrays()
    evidence = fixture_evidence(fixture)
    # Positive control for the reported real-data failure: a one-ULP change in
    # an independently accumulated total must fit the derived roundoff bound.
    original_b3_total = evidence["B3"]["line_emission_energy_total"]
    evidence["B3"]["line_emission_energy_total"] = math.nextafter(
        original_b3_total, math.inf)
    provenance = {"fixture": True, "schema": "LCMFLP01-v1-fixture",
                  "sha256": "fixture"}
    first_result = aggregate(fixture, evidence, provenance)
    second_result = aggregate(fixture, evidence, provenance)
    first = payloads(first_result)
    second = payloads(second_result)
    require(first == second, "fixture payload repeat is not byte-identical")
    require(first_result["S_l_used_source_audit"]
            ["all_rows_with_S_l_used_Planck_flag"],
            "fixture Planck-source audit failed")
    ulp_check = first_result["mapsplit_evidence_float64_cross_check"]["B3"][
        "line_emission_energy_total"]
    require(not ulp_check["exact_float_match"]
            and ulp_check["absolute_difference"] > 0.0
            and ulp_check["absolute_difference"]
            <= ulp_check["absolute_roundoff_limit"],
            "one-ULP mapsplit positive control did not use the roundoff bound")

    negative_output = ""
    try:
        aggregate(fixture, evidence, provenance, inject_omit_eps=True)
    except FractureError as exc:
        negative_output = f"FAIL (expected): {exc}"
    require(bool(negative_output), "injected eps omission was not detected")
    return {
        "status": "PASS", "fixture_only": True,
        "repeat_payloads_byte_identical": True,
        "eps_phys_fixture_header": fixture.eps_phys,
        "src_nlte_fixture_header": fixture.src_nlte,
        "all_rows_Planck_flag": True,
        "one_ulp_mapsplit_positive_control": ulp_check,
        "negative_control": {
            "injection": "omit eps_l while eps_phys == 1",
            "observed": negative_output,
        },
        "clamp": 0, "floor": 0, "cap": 0, "fallback": 0,
        "substitution": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linepop", type=Path)
    parser.add_argument("--mapsplit-csv", type=Path)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            result = self_test()
            print(result["negative_control"]["observed"])
            print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
            return 0
        require(args.outdir is not None, "--outdir is required")
        linepop_path = (args.linepop or DEFAULT_CAPTURE / "linepop_iter10").resolve()
        mapsplit_path = (args.mapsplit_csv or DEFAULT_CAPTURE / "uv_mapsplit" /
                         "uv_mapsplit_shell_band.csv").resolve()
        linepop = base.parse_linepop(linepop_path)
        arrays = arrays_from_linepop(linepop)
        evidence = read_mapsplit_rows(mapsplit_path)
        provenance = {
            "linepop_path": str(linepop.path),
            "linepop_schema": linepop.manifest["schema"],
            "linepop_sha256": linepop.manifest["sha256"],
            "linepop_iteration": linepop.header["iteration"],
            "linepop_field_generation": linepop.header["field_generation"],
            "mapsplit_csv_path": str(mapsplit_path),
            "linepop_reader": "scripts/uv_t2n9_offline.py:parse_linepop",
        }
        first_result = aggregate(arrays, evidence, provenance)
        first = payloads(first_result)
        outdir = args.outdir.resolve()
        outdir.mkdir(parents=True, exist_ok=True)
        for name, content in first.items():
            (outdir / name).write_bytes(content)
        print(json.dumps({
            "schema": SCHEMA, "status": "PASS", "outdir": str(outdir),
            "headline": first_result["headline"],
            "S_l_used_source_audit": first_result["S_l_used_source_audit"],
            "factor_decomposition_summary":
                first_result["factor_decomposition_summary"],
            "negative_control": "covered by --self-test",
        }, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (base.OfflineError, FractureError, OSError, ValueError,
            KeyError, TypeError) as exc:
        print(f"UNRESOLVED-FAIL-CLOSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
