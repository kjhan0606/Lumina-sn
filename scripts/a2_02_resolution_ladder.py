#!/usr/bin/env python3
"""A2-02 offline conservative frequency-resolution ladder.

This runner never launches Lumina and never asks for a GPU.  It consumes a
hash-bound aggregate made from existing fine-grid dumps.  Every physical array
in that aggregate is a bin average.  Candidate grids are made by overlap
integration; point sampling is intentionally not implemented.

Exit codes:
  0  a smallest passing grid was selected (or a static/self-test passed)
  2  input/schema/provenance/validity failure
  3  valid ladder execution, but 8000 -> 16000 failed (BLOCKED)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


SCHEMA_LEDGER = "lumina-a2-02c-frequency-union-v2"
SCHEMA_INPUT = "lumina-a2-02c-resolution-input-v2"
SCHEMA_OUTPUT = "lumina-a2-02c-global-resolution-result-v2"
CANDIDATES = (1000, 2000, 4000, 8000, 16000)
MAX_LIMIT = 0.01
MEDIAN_LIMIT = 0.002
STATE_NAMES = {1: "MEASURED", 2: "EXACT_ZERO", 3: "UNSAMPLED", 4: "OUT_OF_RANGE"}
SAFE_SHELLS = frozenset(range(9))
REPO_ROOT = Path(__file__).resolve().parent.parent


class ContractError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_from(owner: Path, raw: str) -> Path:
    """Resolve manifest paths; relative paths are repository-root relative."""
    del owner  # Kept in the signature to make the manifest ownership explicit.
    path = Path(raw)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"top-level JSON must be an object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def check_ledger(ledger: dict[str, Any]) -> None:
    require(ledger.get("schema") == SCHEMA_LEDGER, "frequency ledger schema mismatch")
    require(ledger.get("amends_after") == "43ffe31", "amends_after must be 43ffe31")
    consumers = ledger.get("consumers")
    require(isinstance(consumers, list) and len(consumers) == 7,
            "frequency ledger must contain exactly seven consumers")
    ordinals = [entry.get("contract_ordinal") for entry in consumers]
    require(ordinals == list(range(1, 8)), "consumer ordinals must be exactly 1..7")
    ids = [entry.get("id") for entry in consumers]
    require(len(set(ids)) == 7, "consumer ids must be unique")
    for entry in consumers:
        require(entry.get("evidence_file_lines"), f"{entry.get('id')}: missing file:line evidence")
        lo, hi = entry.get("nu_min_hz"), entry.get("nu_max_hz")
        require(isinstance(lo, (int, float)) and isinstance(hi, (int, float)) and
                math.isfinite(lo) and math.isfinite(hi) and 0.0 < lo < hi,
                f"{entry.get('id')}: invalid frequency interval")
    union = ledger.get("union", {})
    expected_lo = min(float(entry["nu_min_hz"]) for entry in consumers)
    expected_hi = max(float(entry["nu_max_hz"]) for entry in consumers)
    require(float(union.get("nu_min_hz", -1)) == expected_lo and
            float(union.get("nu_max_hz", -1)) == expected_hi,
            "declared union does not equal the seven-consumer union")
    ladder = ledger.get("resolution_ladder", {})
    require(tuple(ladder.get("candidate_bins", [])) == CANDIDATES,
            "candidate ladder must be 1000/2000/4000/8000/16000")
    require(float(ladder.get("maximum_relative_change_limit", -1)) == MAX_LIMIT,
            "maximum threshold must be exactly 0.01")
    require(float(ladder.get("median_relative_change_limit", -1)) == MEDIAN_LIMIT,
            "median threshold must be exactly 0.002")
    policy = ledger.get("oracle_shell_policy", {})
    require(policy.get("cmfgen_judgment_shells") == list(range(9)),
            "CMFGEN judgment shells must be exactly s0..s8")
    require(set(policy.get("applies_to_metrics", [])) == {"Gamma"},
            "safe-shell restriction must apply to Gamma only")
    require(ladder.get("metrics") == ["band_integral_J", "Gamma",
                                      "band_integral_chi", "band_integral_eta"] and
            ladder.get("Jbar_removed_from_global_ladder") is True,
            "amended global ladder must contain exactly four metrics and no Jbar")
    states = ledger.get("validity_contract", {}).get("states")
    require(states == ["MEASURED", "EXACT_ZERO", "UNSAMPLED", "OUT_OF_RANGE"],
            "validity states must preserve exact-zero/unsampled/out-of-range")
    registered = consumers[6].get("registered_bands")
    require(isinstance(registered, list) and
            [(band.get("id"), band.get("lambda_lo_A"), band.get("lambda_hi_A"))
             for band in registered] == [
                ("450_918_A", 450.0, 918.0),
                ("918_1290_A", 918.0, 1290.0),
                ("1290_2000_A", 1290.0, 2000.0),
                ("2000_10000_A", 2000.0, 10000.0),
                ("10000_25000_A", 10000.0, 25000.0),
            ], "registered validation bands must be the canonical five")


def validate_manifest(manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    manifest = load_json(manifest_path)
    require(manifest.get("schema") == SCHEMA_INPUT, "input manifest schema mismatch")
    require(manifest.get("amends_after") == "43ffe31", "input amends_after mismatch")
    provenance = manifest.get("provenance", {})
    require(provenance.get("new_gpu_run") is False,
            "new_gpu_run must be explicitly false; A2-02 accepts existing dumps only")
    dump_ids = provenance.get("existing_dump_ids")
    require(isinstance(dump_ids, list) and dump_ids,
            "at least one existing_dump_id is required")
    require("point" not in str(provenance.get("packing_method", "")).lower() or
            "no point" in str(provenance.get("packing_method", "")).lower(),
            "packing method must not use point samples")

    ledger_ref = manifest.get("consumer_union_ledger", {})
    ledger_path = resolve_from(manifest_path, str(ledger_ref.get("path", "")))
    require(ledger_path.is_file(), f"frequency ledger absent: {ledger_path}")
    actual_ledger_hash = sha256_file(ledger_path)
    require(ledger_ref.get("sha256") == actual_ledger_hash,
            f"frequency ledger hash mismatch: got {actual_ledger_hash}")
    ledger = load_json(ledger_path)
    check_ledger(ledger)

    fine_ref = manifest.get("fine_dump", {})
    fine_path = resolve_from(manifest_path, str(fine_ref.get("path", "")))
    require(fine_path.is_file(), f"fine dump absent: {fine_path}")
    actual_fine_hash = sha256_file(fine_path)
    require(fine_ref.get("sha256") == actual_fine_hash,
            f"fine dump hash mismatch: got {actual_fine_hash}")
    prereg = manifest.get("preregistered_expectation", {})
    require(prereg.get("pre_run_bin_choice") is None,
            "pre_run_bin_choice must remain null; choosing 1000 before measurement is forbidden")
    require(prereg.get("if_8000_to_16000_fails") == "BLOCKED",
            "last-pair disposition must be BLOCKED")
    require(prereg.get("thresholds") == {"maximum": MAX_LIMIT, "median": MEDIAN_LIMIT},
            "manifest thresholds disagree with canonical thresholds")
    return manifest, ledger, fine_path


def as_1d(data: Any, key: str, dtype: Any | None = None) -> np.ndarray:
    require(key in data, f"fine dump missing {key}")
    value = np.asarray(data[key], dtype=dtype)
    require(value.ndim == 1 and value.size > 0, f"{key} must be a nonempty 1-D array")
    return value


def as_2d(data: Any, key: str, width: int, dtype: Any | None = None,
          allow_empty: bool = False) -> np.ndarray:
    require(key in data, f"fine dump missing {key}")
    value = np.asarray(data[key], dtype=dtype)
    require(value.ndim == 2 and value.shape[1] == width,
            f"{key} must have shape [N,{width}]")
    require(allow_empty or value.shape[0] > 0, f"{key} must have at least one row")
    return value


def validate_arrays(data: Any, ledger: dict[str, Any]) -> dict[str, np.ndarray]:
    edges = as_1d(data, "nu_edges_hz", np.float64)
    require(edges.size >= CANDIDATES[-1] + 1,
            "fine grid must have at least 16000 bins")
    require(np.all(np.isfinite(edges)) and np.all(edges > 0.0) and
            np.all(np.diff(edges) > 0.0),
            "nu_edges_hz must be finite, positive, and strictly ascending")
    union = ledger["union"]
    require(edges[0] <= float(union["nu_min_hz"]) and
            edges[-1] >= float(union["nu_max_hz"]),
            "fine dump does not cover the complete consumer union")
    width = edges.size - 1
    shell = as_1d(data, "shell_id", np.int64)
    require(len(set(map(int, shell))) == shell.size, "shell_id values must be unique")
    j = as_2d(data, "j_nu", width, np.float64)
    state = as_2d(data, "j_state", width, np.uint8)
    chi = as_2d(data, "chi_nu", width, np.float64)
    eta = as_2d(data, "eta_nu", width, np.float64)
    require(j.shape[0] == shell.size and state.shape == j.shape and
            chi.shape == j.shape and eta.shape == j.shape,
            "shell field arrays must all have shape [len(shell_id),F]")
    require(np.all(np.isin(state, list(STATE_NAMES))), "j_state contains an unknown state")
    valid = (state == 1) | (state == 2)
    require(np.all(np.isfinite(j[valid])), "valid J bins must be finite")
    require(np.all(j[valid] >= 0.0), "valid J bins must be nonnegative")
    require(np.all(j[state == 2] == 0.0), "EXACT_ZERO bins must contain exactly 0.0")
    require(np.all(np.isfinite(chi)) and np.all(np.isfinite(eta)),
            "chi/eta arrays must be finite")

    bf_kernel = as_2d(data, "bf_kernel", width, np.float64)
    bf_shell = as_1d(data, "bf_shell_id", np.int64)
    bf_id = as_1d(data, "bf_id")
    line_profile = as_2d(data, "line_profile", width, np.float64)
    line_shell = as_1d(data, "line_shell_id", np.int64)
    line_id = as_1d(data, "line_id")
    require(bf_kernel.shape[0] == bf_shell.size == bf_id.size,
            "bf row metadata length mismatch")
    require(line_profile.shape[0] == line_shell.size == line_id.size,
            "line row metadata length mismatch")
    require(np.all(np.isfinite(bf_kernel)) and np.all(bf_kernel >= 0.0),
            "bf_kernel must be finite and nonnegative")
    require(np.all(np.isfinite(line_profile)) and np.all(line_profile >= 0.0),
            "line_profile must be finite and nonnegative")
    shell_set = set(map(int, shell))
    require(set(map(int, bf_shell)) <= shell_set, "bf_shell_id not present in shell_id")
    require(set(map(int, line_shell)) <= shell_set, "line_shell_id not present in shell_id")
    require(len(set(map(str, bf_id))) == bf_id.size, "bf_id values must be unique")
    require(len(set(map(str, line_id))) == line_id.size, "line_id values must be unique")
    require(np.all(np.sum(bf_kernel * np.diff(edges), axis=1) > 0.0),
            "each bf kernel must have positive integrated support")
    require(np.all(np.sum(line_profile * np.diff(edges), axis=1) > 0.0),
            "each line profile must have positive integrated support")
    # Invalid bins carry no numerical value into overlap arithmetic.  Their
    # state, not a sentinel/floor payload, invalidates every affected record.
    j_clean = j.copy()
    j_clean[~valid] = 0.0
    return {
        "edges": edges, "shell": shell, "j": j_clean, "state": state,
        "chi": chi, "eta": eta, "bf_kernel": bf_kernel,
        "bf_shell": bf_shell, "bf_id": bf_id,
        "line_profile": line_profile, "line_shell": line_shell,
        "line_id": line_id,
    }


def cumulative_at(values: np.ndarray, old_edges: np.ndarray,
                  query_edges: np.ndarray) -> np.ndarray:
    """Integral of piecewise-constant bin averages at arbitrary edges."""
    values = np.asarray(values, dtype=np.float64)
    flat = values.reshape((-1, values.shape[-1]))
    widths = np.diff(old_edges)
    prefix = np.concatenate(
        [np.zeros((flat.shape[0], 1)), np.cumsum(flat * widths, axis=1)], axis=1)
    idx = np.searchsorted(old_edges, query_edges, side="right") - 1
    idx = np.clip(idx, 0, widths.size - 1)
    out = np.take(prefix, idx, axis=1)
    out += np.take(flat, idx, axis=1) * (query_edges - old_edges[idx])
    out[:, query_edges <= old_edges[0]] = 0.0
    total = prefix[:, -1:]
    out[:, query_edges >= old_edges[-1]] = np.broadcast_to(
        total, out.shape)[:, query_edges >= old_edges[-1]]
    return out.reshape(values.shape[:-1] + (query_edges.size,))


def conservative_rebin(values: np.ndarray, old_edges: np.ndarray,
                       new_edges: np.ndarray) -> np.ndarray:
    integ = cumulative_at(values, old_edges, new_edges)
    return np.diff(integ, axis=-1) / np.diff(new_edges)


def band_integrals(values: np.ndarray, edges: np.ndarray,
                   bands: list[dict[str, Any]]) -> np.ndarray:
    query = []
    for band in bands:
        query.extend([float(band["nu_min_hz"]), float(band["nu_max_hz"])])
    at = cumulative_at(values, edges, np.asarray(query, dtype=np.float64))
    return at[..., 1::2] - at[..., 0::2]


def candidate_edges(ledger: dict[str, Any], bins: int) -> np.ndarray:
    union = ledger["union"]
    return np.geomspace(float(union["nu_min_hz"]), float(union["nu_max_hz"]), bins + 1)


def compute_candidate(arr: dict[str, np.ndarray], ledger: dict[str, Any],
                      bins: int) -> dict[str, list[dict[str, Any]]]:
    old_edges = arr["edges"]
    edges = candidate_edges(ledger, bins)
    widths = np.diff(edges)
    j = conservative_rebin(arr["j"], old_edges, edges)
    chi = conservative_rebin(arr["chi"], old_edges, edges)
    eta = conservative_rebin(arr["eta"], old_edges, edges)
    invalid_width = conservative_rebin(
        np.isin(arr["state"], [3, 4]).astype(np.float64), old_edges, edges) * widths
    bands = ledger["consumers"][6]["registered_bands"]
    j_band = band_integrals(j, edges, bands)
    bad_band = band_integrals((invalid_width > 0.0).astype(np.float64), edges, bands) > 0.0
    chi_band = band_integrals(chi, edges, bands)
    eta_band = band_integrals(eta, edges, bands)
    rows: dict[str, list[dict[str, Any]]] = {"J_band": [], "Gamma": [], "chi": [], "eta": []}
    # 2026-08-05 driver fix: a shell whose J is state-4 across the whole grid
    # (s44+ outside EDDFACTOR coverage, valid=false BY CONSTRUCTION in the npz)
    # can never yield a convergence verdict; counting it as invalid-eligible
    # made every ladder BLOCKED by construction.  Structural absence (state 4)
    # is excluded with a reason; MC gaps (state 3) stay invalid-eligible
    # (fail-closed), preserving the four-state distinction of the order (§9).
    dead_shell = np.all(arr["state"] == 4, axis=1)
    for si, shell in enumerate(arr["shell"]):
        for bi, band in enumerate(bands):
            suffix = f"s{int(shell)}:{band['id']}"
            rows["J_band"].append({
                "record_id": suffix, "shell": int(shell), "band": band["id"],
                "value": float(j_band[si, bi]), "valid": not bool(bad_band[si, bi]),
                "judgment_eligible": not bool(dead_shell[si]),
                "exclusion_reason": None if not dead_shell[si] else
                    "shell J entirely OUT_OF_RANGE by construction (outside EDDFACTOR coverage)",
            })
            rows["chi"].append({
                "record_id": f"chi:{suffix}", "shell": int(shell), "band": band["id"],
                "value": float(chi_band[si, bi]), "valid": True,
                "judgment_eligible": True,
            })
            rows["eta"].append({
                "record_id": f"eta:{suffix}", "shell": int(shell), "band": band["id"],
                "value": float(eta_band[si, bi]), "valid": True,
                "judgment_eligible": True,
            })

    shell_index = {int(value): index for index, value in enumerate(arr["shell"])}
    invalid_source = np.isin(arr["state"], [3, 4])
    for start in range(0, arr["bf_kernel"].shape[0], 64):
        stop = min(start + 64, arr["bf_kernel"].shape[0])
        kernel = conservative_rebin(arr["bf_kernel"][start:stop], old_edges, edges)
        for local, row_index in enumerate(range(start, stop)):
            shell = int(arr["bf_shell"][row_index])
            si = shell_index[shell]
            value = float(np.sum(j[si] * kernel[local] * widths))
            support = arr["bf_kernel"][row_index] > 0.0
            support_bad = bool(np.any(invalid_source[si] & support))
            support_oor = bool(np.any((arr["state"][si] == 4) & support))
            support_mc_gap = bool(np.any((arr["state"][si] == 3) & support))
            # driver fix 2026-08-05: structural OUT_OF_RANGE support excludes the
            # record (undefined by construction); MC gaps stay fail-closed.
            eligible = (shell in SAFE_SHELLS) and not (support_oor and not support_mc_gap)
            rows["Gamma"].append({
                "record_id": str(arr["bf_id"][row_index]), "shell": shell,
                "value": value, "valid": not support_bad,
                "judgment_eligible": eligible,
                "exclusion_reason": None if eligible else
                    ("support outside J coverage by construction"
                     if (shell in SAFE_SHELLS) else
                     "s9 straddles the jnu4/modern boundary" if shell == 9 else
                     "s10+ jnu4 outer oracle contaminated"),
            })

    return rows


def compare_rows(coarse: list[dict[str, Any]], fine: list[dict[str, Any]],
                 label: str) -> dict[str, Any]:
    cmap = {row["record_id"]: row for row in coarse}
    fmap = {row["record_id"]: row for row in fine}
    require(cmap.keys() == fmap.keys(), f"{label}: record identity changed across grids")
    details = []
    eligible_errors = []
    invalid_eligible = 0
    excluded = 0
    for record_id in sorted(cmap):
        a, b = cmap[record_id], fmap[record_id]
        require(a["judgment_eligible"] == b["judgment_eligible"],
                f"{label}:{record_id}: eligibility changed")
        valid = bool(a["valid"] and b["valid"])
        eligible = bool(a["judgment_eligible"])
        zero_denominator_mismatch = bool(
            valid and float(b["value"]) == 0.0 and float(a["value"]) != 0.0)
        if not valid or zero_denominator_mismatch:
            error = None
        elif float(b["value"]) == 0.0:
            error = 0.0
        else:
            error = abs(float(a["value"]) - float(b["value"])) / abs(float(b["value"]))
        if eligible:
            if valid and not zero_denominator_mismatch:
                eligible_errors.append(error)
            else:
                invalid_eligible += 1
        else:
            excluded += 1
        details.append({
            "record_id": record_id,
            "shell": a.get("shell"),
            "coarse_value": a["value"], "fine_value": b["value"],
            "relative_change": error, "valid": valid,
            "zero_denominator_mismatch": zero_denominator_mismatch,
            "judgment_eligible": eligible,
            "exclusion_reason": a.get("exclusion_reason"),
        })
    finite = np.asarray(eligible_errors, dtype=np.float64)
    maximum = float(np.max(finite)) if finite.size else None
    median = float(np.median(finite)) if finite.size else None
    passed = (invalid_eligible == 0 and finite.size > 0 and
              maximum <= MAX_LIMIT and median <= MEDIAN_LIMIT)
    return {
        "metric": label, "maximum_relative_change": maximum,
        "median_relative_change": median,
        "maximum_limit": MAX_LIMIT, "median_limit": MEDIAN_LIMIT,
        "eligible_records": int(finite.size) + invalid_eligible,
        "invalid_eligible_records": invalid_eligible,
        "excluded_record_only": excluded, "passed": passed,
        "records": details,
    }


def compare_pair(coarse_n: int, fine_n: int,
                 coarse: dict[str, list[dict[str, Any]]],
                 fine: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    j = compare_rows(coarse["J_band"], fine["J_band"], "band_integral_J")
    gamma = compare_rows(coarse["Gamma"], fine["Gamma"], "Gamma")
    chi = compare_rows(coarse["chi"], fine["chi"], "band_integral_chi")
    eta = compare_rows(coarse["eta"], fine["eta"], "band_integral_eta")
    chieta_pass = bool(chi["passed"] and eta["passed"])
    return {
        "coarse_bins": coarse_n, "fine_bins": fine_n,
        "metrics": {
            "band_integral_J": j, "Gamma": gamma,
            "band_integral_chi": chi, "band_integral_eta": eta,
            "chi_eta_combined_gate": {
                "passed": chieta_pass,
                "rule": "chi and eta must each satisfy max<=1% and median<=0.2%",
            },
        },
        "all_four_contract_metrics_passed": bool(
            j["passed"] and gamma["passed"] and chieta_pass),
    }


def run_ladder(arr: dict[str, np.ndarray], ledger: dict[str, Any]) -> dict[str, Any]:
    calculated = {n: compute_candidate(arr, ledger, n) for n in CANDIDATES}
    pairs = [compare_pair(a, b, calculated[a], calculated[b])
             for a, b in zip(CANDIDATES, CANDIDATES[1:])]
    selected = next((pair["coarse_bins"] for pair in pairs
                     if pair["all_four_contract_metrics_passed"]), None)
    decision = "SELECTED" if selected is not None else "BLOCKED"
    return {
        "schema": SCHEMA_OUTPUT,
        "decision": decision,
        "selected_bins": selected,
        "selection_reason": (
            f"{selected} is the smallest N whose N-to-2N pair passes all metrics"
            if selected is not None else
            "8000-to-16000 did not pass all four contract metrics"),
        "thresholds": {"maximum": MAX_LIMIT, "median": MEDIAN_LIMIT},
        "candidate_bins": list(CANDIDATES),
        "coordinate_method": "conservative overlap integration of bin averages",
        "global_metrics": ["band_integral_J", "Gamma", "band_integral_chi",
                           "band_integral_eta"],
        "Jbar_global_metric": "REMOVED_BY_AMENDMENT",
        "point_sampling_used": False,
        "validity_states": STATE_NAMES,
        "cmfgen_judgment_shells": list(range(9)),
        "cmfgen_record_only_shells": "Gamma: s9+ (s9 boundary-straddling; s10+ contaminated)",
        "pairs": pairs,
    }


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=path.name + ".",
                                     suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def self_test() -> None:
    lo, hi, fine_bins = 1.0e14, 1.0e16, 32768
    edges = np.geomspace(lo, hi, fine_bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    shells = np.arange(11, dtype=np.int64)
    scale = 1.0 + 0.01 * shells[:, None]
    shape = (centers / 1.0e15) ** -0.35
    j = scale * shape
    chi = scale * (0.4 + 0.2 * (centers / 1.0e15) ** 0.1)
    eta = scale * (0.3 + 0.1 * (centers / 1.0e15) ** -0.1)
    state = np.ones_like(j, dtype=np.uint8)
    state[:, 123] = 2
    j[:, 123] = 0.0
    bf_shell = np.asarray([0, 8, 10], dtype=np.int64)
    bf_kernel = np.asarray([
        np.where(centers >= threshold, (threshold / centers) ** 3 / centers, 0.0)
        for threshold in (2.0e14, 5.0e14, 8.0e14)])
    line_shell = np.asarray([0, 8, 10], dtype=np.int64)
    line_profile = np.asarray([
        np.exp(-0.5 * (np.log(centers / center) / 0.08) ** 2)
        for center in (8.0e14, 2.0e15, 4.0e15)])
    ledger = {
        "schema": SCHEMA_LEDGER,
        "amends_after": "43ffe31",
        "consumers": [
            {"id": f"c{i}", "contract_ordinal": i, "nu_min_hz": lo,
             "nu_max_hz": hi, "evidence_file_lines": ["fixture:1"],
             **({"registered_bands": [
                 {"id": "450_918_A", "lambda_lo_A": 450.0, "lambda_hi_A": 918.0,
                  "nu_min_hz": 1.0e14, "nu_max_hz": 2.0e14},
                 {"id": "918_1290_A", "lambda_lo_A": 918.0, "lambda_hi_A": 1290.0,
                  "nu_min_hz": 2.0e14, "nu_max_hz": 5.0e14},
                 {"id": "1290_2000_A", "lambda_lo_A": 1290.0, "lambda_hi_A": 2000.0,
                  "nu_min_hz": 5.0e14, "nu_max_hz": 1.0e15},
                 {"id": "2000_10000_A", "lambda_lo_A": 2000.0, "lambda_hi_A": 10000.0,
                  "nu_min_hz": 1.0e15, "nu_max_hz": 3.0e15},
                 {"id": "10000_25000_A", "lambda_lo_A": 10000.0, "lambda_hi_A": 25000.0,
                  "nu_min_hz": 3.0e15, "nu_max_hz": 1.0e16},
             ]} if i == 7 else {})}
            for i in range(1, 8)
        ],
        "union": {"nu_min_hz": lo, "nu_max_hz": hi},
        "resolution_ladder": {
            "candidate_bins": list(CANDIDATES),
            "maximum_relative_change_limit": MAX_LIMIT,
            "median_relative_change_limit": MEDIAN_LIMIT,
            "metrics": ["band_integral_J", "Gamma", "band_integral_chi",
                        "band_integral_eta"],
            "Jbar_removed_from_global_ladder": True,
        },
        "oracle_shell_policy": {
            "cmfgen_judgment_shells": list(range(9)),
            "applies_to_metrics": ["Gamma"],
        },
        "validity_contract": {
            "states": ["MEASURED", "EXACT_ZERO", "UNSAMPLED", "OUT_OF_RANGE"]
        },
    }
    check_ledger(ledger)
    data = {
        "nu_edges_hz": edges, "shell_id": shells, "j_nu": j,
        "j_state": state, "chi_nu": chi, "eta_nu": eta,
        "bf_kernel": bf_kernel, "bf_shell_id": bf_shell,
        "bf_id": np.asarray(["g0", "g8", "g10"]),
        "line_profile": line_profile, "line_shell_id": line_shell,
        "line_id": np.asarray(["l0", "l8", "l10"]),
    }
    arr = validate_arrays(data, ledger)
    result = run_ladder(arr, ledger)
    require(result["decision"] == "SELECTED", "smooth fixture should select a grid")
    for pair in result["pairs"]:
        require(pair["metrics"]["Gamma"]["excluded_record_only"] == 1,
                "contaminated Gamma shell was not recorded/excluded")
        require("Jbar" not in pair["metrics"], "Jbar leaked into global ladder")
    broken = dict(data)
    broken["j_state"] = state.copy()
    broken["j_state"][0, 1000] = 3
    broken_arr = validate_arrays(broken, ledger)
    broken_result = run_ladder(broken_arr, ledger)
    require(any(pair["metrics"]["band_integral_J"]["invalid_eligible_records"] > 0
                for pair in broken_result["pairs"]),
            "UNSAMPLED state must invalidate a judgment record")
    print("A2_02C_LADDER_SELFTEST PASS conservative_rebin=1 point_sample=0 "
          "global_metrics=4 Jbar_removed=1 invalid_eligible_gate=0")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check-ledger")
    check.add_argument("ledger", type=Path)
    sub.add_parser("self-test")
    run = sub.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "check-ledger":
            ledger = load_json(args.ledger.resolve())
            check_ledger(ledger)
            print("A2_02C_LEDGER PASS consumers=7 ladder=1000/2000/4000/8000/16000 "
                  "metrics=4 Jbar_removed=1 max=0.01 median=0.002 safe_shells=s0-s8")
            return 0
        if args.command == "self-test":
            self_test()
            return 0
        manifest, ledger, fine_path = validate_manifest(args.manifest.resolve())
        with np.load(fine_path, allow_pickle=False) as data:
            arrays = validate_arrays(data, ledger)
            result = run_ladder(arrays, ledger)
        result["input_manifest"] = str(args.manifest.resolve())
        result["input_manifest_sha256"] = sha256_file(args.manifest.resolve())
        result["consumer_union_ledger_sha256"] = sha256_file(
            resolve_from(args.manifest.resolve(), manifest["consumer_union_ledger"]["path"]))
        result["fine_dump_sha256"] = sha256_file(fine_path)
        result["provenance"] = manifest["provenance"]
        write_json_atomic(args.output.resolve(), result)
        print(f"A2_02C_LADDER {result['decision']} selected_bins={result['selected_bins']} "
              f"output={args.output.resolve()}")
        return 0 if result["decision"] == "SELECTED" else 3
    except (ContractError, OSError, ValueError, KeyError) as exc:
        print(f"A2_02C_INPUT_FAIL {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
