#!/usr/bin/env python3
"""Fail-closed consumer check for an LCMFLP01 population-native line dump.

The artifact is the T2 companion of the LCMFCE01 chi/eta capture: the same
generation, but resolved per (line, shell) instead of per (shell, coarse bin).
This reader refuses anything that would let a T2 experiment be run on the wrong
generation or on a replay that did not reproduce the assembled forest:

  * the generation contract is iteration 10 (same as cmf_chieta_check); any
    other expectation needs --non-contract-override,
  * the manifest SHA-256 must match the payload,
  * chi_line_roundtrip_bitwise must be true — a replay that does not reproduce
    cs->chi_line bit for bit is not the assembled forest and must not be used
    as a single-factor baseline,
  * the EPAY disposition column must be present, because eta_line is DISCARDED
    (rate-shape replaced) in a large fraction of the thin bins and swapping the
    emissivity there measures nothing.

No clamp, no fill, no fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
from typing import NamedTuple

import numpy as np

MAGIC = b"LCMFLP01"
ENDIAN = 0x01020304
VERSION = 1
CONTRACT_ITERATION = 10
NON_CONTRACT_RC = 2
ROW_BYTES = 76
LINE_BYTES = 80

HEADER = struct.Struct("<8sIIQQIIII d d d IIII d d d d d d")
ROW_DTYPE = np.dtype([
    ("line_slot", "<u4"), ("shell_slot", "<u4"), ("flags", "<u4"),
    ("tau_used", "<f8"), ("tau_from_pops", "<f8"),
    ("n_lower", "<f8"), ("n_upper", "<f8"),
    ("S_l_pop", "<f8"), ("S_l_used", "<f8"),
    ("eps_l", "<f8"), ("w", "<f8"),
], align=False)
LINE_DTYPE = np.dtype([
    ("line_id", "<u4"), ("bin", "<u4"), ("Z", "<i4"), ("ion", "<i4"),
    ("g_lower", "<i4"), ("g_upper", "<i4"),
    ("nlte_lower", "<i4"), ("nlte_upper", "<i4"),
    ("nu_l", "<f8"), ("lambda_cm", "<f8"), ("A_ul", "<f8"), ("f_lu", "<f8"),
    ("E_lower_eV", "<f8"), ("E_upper_eV", "<f8"),
], align=False)

F_NLTE_ION = 1 << 0
F_POPS_DEFINED = 1 << 1
F_SL_POP = 1 << 2
F_SL_FALLBACK = 1 << 3
F_STIM_CLAMPED = 1 << 4
F_TAU_ROUNDTRIP = 1 << 5

DISPOSITION = {0: "legacy_source", 1: "thick_exempt",
               2: "rate_shape_replaced", 3: "scalar_rescaled"}


class LinePopError(ValueError):
    pass


class LinePop(NamedTuple):
    header: dict
    shells: np.ndarray
    shell_state: np.ndarray
    nu: np.ndarray
    dnu: np.ndarray
    chi_line: np.ndarray
    chi_line_th: np.ndarray
    eta_line: np.ndarray
    disposition: np.ndarray
    lines: np.ndarray
    rows: np.ndarray
    manifest: dict
    contract_status: str


def check_artifact(path: Path, expected_iteration: int = CONTRACT_ITERATION,
                   non_contract_override: bool = False,
                   require_roundtrip: bool = True) -> LinePop:
    path = Path(path)
    deviates = expected_iteration != CONTRACT_ITERATION
    if deviates and not non_contract_override:
        raise LinePopError("non-contract expectation requires "
                           "--non-contract-override")
    raw = path.read_bytes()
    if len(raw) < HEADER.size:
        raise LinePopError("truncated header")
    (magic, endian, version, iteration, generation, n_shells, n_bins,
     n_sel, n_lines_sel, n_rows, t_exp, lam_lo, lam_hi, eps_phys, src_nlte,
     epay, epay_smin, epay_taubin, epay_hotf, eps_floor, eps_cap,
     line_eps, eps_uv, line_gate) = _unpack(raw)
    if (magic, endian, version) != (MAGIC, ENDIAN, VERSION):
        raise LinePopError("schema identity mismatch")
    if not (n_shells > 0 and n_bins > 0 and 0 < n_sel <= n_shells and
            lam_hi > lam_lo > 0.0 and t_exp > 0.0):
        raise LinePopError("invalid dimensions/window")
    if iteration != expected_iteration:
        raise LinePopError(
            f"generation mismatch: got iteration {iteration}, "
            f"expected {expected_iteration}")
    if generation != iteration:
        raise LinePopError("field_generation disagrees with iteration")

    off = _header_size()
    shells, off = _take(raw, off, "<u4", n_sel)
    shell_state, off = _take(raw, off, "<f8", 4 * n_sel)
    nu, off = _take(raw, off, "<f8", n_bins)
    dnu, off = _take(raw, off, "<f8", n_bins)
    chi_line, off = _take(raw, off, "<f8", n_sel * n_bins)
    chi_line_th, off = _take(raw, off, "<f8", n_sel * n_bins)
    eta_line, off = _take(raw, off, "<f8", n_sel * n_bins)
    disposition, off = _take(raw, off, "u1", n_shells * n_bins)
    if off + n_lines_sel * LINE_BYTES + n_rows * ROW_BYTES != len(raw):
        raise LinePopError("payload length disagrees with the header counts")
    lines = np.frombuffer(raw, dtype=LINE_DTYPE, count=n_lines_sel,
                          offset=off).copy()
    off += n_lines_sel * LINE_BYTES
    rows = np.frombuffer(raw, dtype=ROW_DTYPE, count=n_rows, offset=off).copy()

    if np.any(shells >= n_shells) or np.unique(shells).size != n_sel:
        raise LinePopError("invalid/duplicated shell selection")
    if not all(nu[k] < nu[k + 1] for k in range(len(nu) - 1)):
        raise LinePopError("frequency is not ascending")
    if np.any(dnu <= 0.0):
        raise LinePopError("non-positive dnu")
    for name, values in (("chi_line", chi_line), ("chi_line_th", chi_line_th),
                         ("eta_line", eta_line)):
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise LinePopError(f"invalid {name}; no clamp allowed")
    if n_rows and (np.any(rows["shell_slot"] >= n_sel) or
                   np.any(rows["line_slot"] >= max(n_lines_sel, 1))):
        raise LinePopError("row references a slot outside the payload")
    if n_rows and (not np.isfinite(rows["w"]).all() or
                   np.any(rows["w"] < 0.0) or np.any(rows["tau_used"] <= 0.0)):
        raise LinePopError("invalid row tau/w")
    if np.any(disposition > 3):
        raise LinePopError("unknown EPAY disposition code")

    sidecar = Path(str(path) + ".manifest.json")
    try:
        manifest = json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise LinePopError(f"invalid/missing sidecar: {exc}") from exc
    if manifest.get("schema") != "LCMFLP01-v1":
        raise LinePopError("sidecar schema mismatch")
    if manifest.get("sha256") != hashlib.sha256(raw).hexdigest():
        raise LinePopError("sidecar sha256 mismatch")
    for key, value in (("iteration", iteration), ("field_generation", generation),
                       ("rows", int(n_rows)), ("selected_lines", int(n_lines_sel)),
                       ("selected_shells", int(n_sel))):
        if manifest.get(key) != value:
            raise LinePopError(f"sidecar {key} disagrees with payload")
    if require_roundtrip and not manifest.get("chi_line_roundtrip_bitwise"):
        raise LinePopError(
            "chi_line round trip is not bitwise: the replay did not reproduce "
            "the assembled forest, so this artifact is not a valid "
            "single-factor T2 baseline "
            f"(max_abs={manifest.get('chi_line_roundtrip_max_abs')})")

    header = {
        "schema": "LCMFLP01-v1", "iteration": int(iteration),
        "field_generation": int(generation), "n_shells": int(n_shells),
        "n_bins": int(n_bins), "selected_shells": int(n_sel),
        "selected_lines": int(n_lines_sel), "rows": int(n_rows),
        "time_explosion": t_exp, "lambda_window_A": [lam_lo, lam_hi],
        "eps_phys": int(eps_phys), "src_nlte": int(src_nlte),
        "epay": int(epay), "epay_smin": int(epay_smin),
        "epay_taubin": epay_taubin, "epay_hotf": epay_hotf,
        "eps_floor": eps_floor, "eps_cap": eps_cap,
        "line_eps": line_eps, "eps_uv": eps_uv, "line_gate": line_gate,
    }
    return LinePop(header, shells, shell_state.reshape(n_sel, 4), nu, dnu,
                   chi_line.reshape(n_sel, n_bins),
                   chi_line_th.reshape(n_sel, n_bins),
                   eta_line.reshape(n_sel, n_bins),
                   disposition.reshape(n_shells, n_bins), lines, rows,
                   manifest, "NON-CONTRACT" if deviates else "CONTRACT")


def _header_size() -> int:
    # 8 magic + 2*u32 + 2*u64 + 4*u32 + 3*f64 + 4*u32 + 8*f64
    return 8 + 8 + 16 + 16 + 24 + 16 + 64


def _unpack(raw: bytes):
    o = 0
    magic = raw[0:8]; o = 8
    endian, version = struct.unpack_from("<II", raw, o); o += 8
    iteration, generation = struct.unpack_from("<QQ", raw, o); o += 16
    n_shells, n_bins, n_sel, n_lines_sel = struct.unpack_from("<IIII", raw, o)
    o += 16
    n_rows, = struct.unpack_from("<Q", raw, o); o += 8
    t_exp, lam_lo, lam_hi = struct.unpack_from("<ddd", raw, o); o += 24
    eps_phys, src_nlte, epay, epay_smin = struct.unpack_from("<IIII", raw, o)
    o += 16
    (epay_taubin, epay_hotf, eps_floor, eps_cap, line_eps, eps_uv,
     line_gate) = struct.unpack_from("<7d", raw, o)
    return (magic, endian, version, iteration, generation, n_shells, n_bins,
            n_sel, n_lines_sel, n_rows, t_exp, lam_lo, lam_hi, eps_phys,
            src_nlte, epay, epay_smin, epay_taubin, epay_hotf, eps_floor,
            eps_cap, line_eps, eps_uv, line_gate)


def _take(raw: bytes, offset: int, dtype: str, count: int):
    dt = np.dtype(dtype)
    end = offset + count * dt.itemsize
    if end > len(raw):
        raise LinePopError("truncated array")
    return np.frombuffer(raw, dtype=dt, count=count, offset=offset).copy(), end


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--expected-iteration", type=int,
                        default=CONTRACT_ITERATION)
    parser.add_argument("--non-contract-override", action="store_true")
    parser.add_argument("--allow-roundtrip-failure", action="store_true",
                        help="inspect a replay that did NOT reproduce "
                             "chi_line bitwise (never valid for T2)")
    args = parser.parse_args()
    try:
        lp = check_artifact(args.input, args.expected_iteration,
                            args.non_contract_override,
                            not args.allow_roundtrip_failure)
    except (OSError, LinePopError) as exc:
        print(f"FAIL: {exc}")
        return 1
    total = lp.disposition.size
    disp = {name: int(np.count_nonzero(lp.disposition == code))
            for code, name in DISPOSITION.items()}
    live = disp["legacy_source"] + disp["thick_exempt"]
    flags = lp.rows["flags"] if lp.rows.size else np.zeros(0, dtype=np.uint32)
    summary = {
        **lp.header,
        "sha256": lp.manifest["sha256"],
        "contract_status": lp.contract_status,
        "chi_line_roundtrip_bitwise": lp.manifest["chi_line_roundtrip_bitwise"],
        "chi_line_th_comparable": lp.manifest["chi_line_th_comparable"],
        "epay_disposition_cells": disp,
        "eta_line_reaches_S_fixed_fraction": live / total if total else 0.0,
        "rows_with_defined_populations": int(
            np.count_nonzero(flags & F_POPS_DEFINED)),
        "rows_with_population_native_S": int(np.count_nonzero(flags & F_SL_POP)),
        "rows_using_planck_fallback_S": int(
            np.count_nonzero(flags & F_SL_FALLBACK)),
        "rows_with_tau_population_roundtrip": int(
            np.count_nonzero(flags & F_TAU_ROUNDTRIP)),
        "rows_with_stimulated_clamp": int(
            np.count_nonzero(flags & F_STIM_CLAMPED)),
        "clamp": 0, "fallback": 0,
    }
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return NON_CONTRACT_RC if lp.contract_status == "NON-CONTRACT" else 0


if __name__ == "__main__":
    raise SystemExit(main())
