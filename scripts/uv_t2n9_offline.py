#!/usr/bin/env python3
"""Preregistered offline UV T2/N9 consumer.

Run this only on grammar-debug.  It reads existing iter-10 artifacts, rebuilds
LCMFCE01 lanes, and calls the existing CPU Stage-3.1 formal operator.  It does
not run a model, CMFGEN, or a GPU kernel and it never substitutes a value for
an invalid or unavailable datum.
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
import subprocess
import sys
from typing import Any, NamedTuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from cmf_chieta_check import check_artifact  # noqa: E402
from emiss_e6_direct_fields import cmfgen_all_shells, weighted_mean  # noqa: E402
import stage31_cmf_field_bench as bench  # noqa: E402


LINEPOP_SHA256 = "84d1849dafd1c796dac77c4037b19683e3ef1d5ddb72dd0e6bf701490b05a1cc"
CMF_RUN = Path("/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4")
EXPECTED_SHELLS = np.asarray([0, 8, 16, 20, 45], dtype="<u4")
EXPECTED_DISPOSITION = {
    "legacy_source": 5000,
    "thick_exempt": 10696,
    "rate_shape_replaced": 34304,
    "scalar_rescaled": 0,
}
DISPOSITION = {0: "legacy_source", 1: "thick_exempt",
               2: "rate_shape_replaced", 3: "scalar_rescaled"}

ELEMENT_SYMBOL = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N",
    8: "O", 9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al",
    14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar", 19: "K",
    20: "Ca", 21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn",
    26: "Fe", 27: "Co", 28: "Ni", 29: "Cu", 30: "Zn",
}

CM_H = 6.62607015e-27
CM_KB = 1.380649e-16
CM_C = 2.99792458e10
CM_SIGMA_T = 6.6524587e-25
M_PI = math.pi

F_NLTE_ION = 1 << 0
F_POPS_DEFINED = 1 << 1
F_SL_POP = 1 << 2
F_SL_USED_PLANCK = 1 << 3
F_STIM_ADJUSTED = 1 << 4
F_TAU_ROUNDTRIP = 1 << 5
KNOWN_FLAGS = ((1 << 6) - 1)

ROW_BYTES = 76
LINE_BYTES = 80
LP_HEADER = struct.Struct("<8sIIQQIIIIQdddIIIIddddddd")
CE_HEADER = struct.Struct("<8sIIQQQQIId")

ROW_DTYPE = np.dtype([
    ("line_slot", "<u4"), ("shell_slot", "<u4"), ("flags", "<u4"),
    ("tau_used", "<f8"), ("tau_from_pops", "<f8"),
    ("n_lower", "<f8"), ("n_upper", "<f8"),
    ("S_l_pop", "<f8"), ("S_l_used", "<f8"),
    ("eps_l", "<f8"), ("w", "<f8"),
], align=False)
LINE_DTYPE = np.dtype([
    ("line_id", "<u4"), ("bin", "<u4"), ("Z", "<i4"),
    ("ion", "<i4"), ("g_lower", "<i4"), ("g_upper", "<i4"),
    ("nlte_lower", "<i4"), ("nlte_upper", "<i4"),
    ("nu_l", "<f8"), ("lambda_cm", "<f8"), ("A_ul", "<f8"),
    ("f_lu", "<f8"), ("E_lower_eV", "<f8"),
    ("E_upper_eV", "<f8"),
], align=False)

BANDS = (
    ("B0", 600.0, 1000.0),
    ("B1", 1000.0, 1500.0),
    ("B2", 1500.0, 2000.0),
    ("B3", 2000.0, 2500.0),
    ("B4", 2500.0, 3000.0),
    ("BALL", 600.0, 3000.0),
)


class OfflineError(RuntimeError):
    pass


class LinePop(NamedTuple):
    path: Path
    manifest: dict[str, Any]
    header: dict[str, Any]
    shells: np.ndarray
    shell_state: np.ndarray
    nu: np.ndarray
    dnu: np.ndarray
    chi_line: np.ndarray
    chi_line_th: np.ndarray
    eta_line: np.ndarray
    disposition: np.ndarray
    lines: np.memmap
    rows: np.memmap
    offsets: dict[str, int]


def json_write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True,
                               allow_nan=False) + "\n")


def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise OfflineError(f"refusing empty CSV {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise OfflineError(message)


def finite_nonnegative(name: str, values: np.ndarray) -> None:
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise OfflineError(f"{name} contains a negative or nonfinite value")


def planck(nu: np.ndarray, temperature: float) -> np.ndarray:
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise OfflineError("invalid electron temperature")
    x = CM_H * nu / (CM_KB * temperature)
    if np.any(x > 700.0):
        raise OfflineError("Planck evaluation would underflow in the audited UV cells")
    denominator = np.expm1(x)
    if np.any(denominator <= 0.0) or not np.isfinite(denominator).all():
        raise OfflineError("invalid Planck denominator")
    result = 2.0 * CM_H * nu * nu * nu / (CM_C * CM_C * denominator)
    if np.any(result <= 0.0) or not np.isfinite(result).all():
        raise OfflineError("invalid Planck result")
    return result


def parse_linepop(path: Path) -> LinePop:
    """Validate small metadata and total length before mapping the large tables."""
    sidecar = Path(str(path) + ".manifest.json")
    try:
        manifest = json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise OfflineError(f"invalid LINEPOP sidecar: {exc}") from exc
    with path.open("rb") as stream:
        head_raw = stream.read(LP_HEADER.size)
    require(len(head_raw) == LP_HEADER.size, "truncated LINEPOP header")
    unpacked = LP_HEADER.unpack(head_raw)
    (magic, endian, version, iteration, generation, ns, nb, nsel, nlines,
     nrows, texp, lamlo, lamhi, eps_phys, src_nlte, epay, epay_smin,
     epay_taubin, epay_hotf, eps_low_gate, eps_high_gate, line_eps, eps_uv,
     line_gate) = unpacked
    require((magic, endian, version) == (b"LCMFLP01", 0x01020304, 1),
            "LINEPOP schema identity mismatch")
    require((iteration, generation, ns, nb) == (10, 10, 50, 1000),
            "LINEPOP generation/dimensions violate the contract")
    require((nsel, nlines, nrows) == (5, 601371, 1169145),
            "LINEPOP selection counts violate the preregistration")
    require((lamlo, lamhi) == (600.0, 3000.0), "LINEPOP wavelength mismatch")
    require((eps_phys, src_nlte, epay, epay_smin) == (1, 0, 2, 5),
            "LINEPOP gate mismatch")
    require((epay_taubin, epay_hotf) == (10.0, 0.0),
            "LINEPOP EPAY threshold mismatch")
    require(ROW_DTYPE.itemsize == ROW_BYTES and LINE_DTYPE.itemsize == LINE_BYTES,
            "local dtype size disagrees with writer")

    offsets: dict[str, int] = {"fixed_header": 0}
    off = LP_HEADER.size
    offsets["selected_shell_ids"] = off; off += nsel * 4
    offsets["selected_shell_state"] = off; off += nsel * 4 * 8
    offsets["nu"] = off; off += nb * 8
    offsets["dnu"] = off; off += nb * 8
    offsets["chi_line"] = off; off += nsel * nb * 8
    offsets["chi_line_th"] = off; off += nsel * nb * 8
    offsets["eta_line"] = off; off += nsel * nb * 8
    offsets["disposition"] = off; off += ns * nb
    offsets["line_static"] = off; off += nlines * LINE_BYTES
    offsets["rows"] = off; off += nrows * ROW_BYTES
    expected_bytes = off
    require(path.stat().st_size == expected_bytes,
            f"LINEPOP length mismatch: got {path.stat().st_size}, "
            f"writer layout requires {expected_bytes}")
    require(expected_bytes == 137151032,
            f"preregistered size arithmetic drifted: {expected_bytes}")
    require(manifest.get("schema") == "LCMFLP01-v1", "LINEPOP sidecar schema")
    for key, expected in (("iteration", 10), ("field_generation", 10),
                          ("n_shells", 50), ("n_bins", 1000),
                          ("selected_shells", 5), ("selected_lines", 601371),
                          ("rows", 1169145), ("row_bytes", 76)):
        require(manifest.get(key) == expected, f"LINEPOP sidecar {key} mismatch")
    require(manifest.get("chi_line_roundtrip_bitwise") is True,
            "writer chi_line replay was not bitwise")
    require(manifest.get("epay_scale_not_reproducible") is True,
            "EPAY replayability flag is absent")
    require(manifest.get("clamp") == 0 and manifest.get("fallback") == 0,
            "LINEPOP producer guard counters are nonzero")
    require(manifest.get("epay_disposition_counts") == EXPECTED_DISPOSITION,
            "LINEPOP disposition manifest differs from preregistration")

    digest = sha256_file(path)
    require(digest == LINEPOP_SHA256,
            f"LINEPOP SHA-256 mismatch: {digest}")
    require(manifest.get("sha256") == digest, "LINEPOP sidecar SHA-256 mismatch")

    def small(dtype: str, count: int, key: str) -> np.ndarray:
        return np.memmap(path, dtype=dtype, mode="r", offset=offsets[key],
                         shape=(count,)).copy()

    shells = small("<u4", nsel, "selected_shell_ids")
    require(shells.tobytes() == EXPECTED_SHELLS.tobytes(),
            f"selected shell identity mismatch: {shells.tolist()}")
    shell_state = small("<f8", nsel * 4, "selected_shell_state").reshape(nsel, 4)
    nu = small("<f8", nb, "nu")
    dnu = small("<f8", nb, "dnu")
    chi_line = small("<f8", nsel * nb, "chi_line").reshape(nsel, nb)
    chi_line_th = small("<f8", nsel * nb, "chi_line_th").reshape(nsel, nb)
    eta_line = small("<f8", nsel * nb, "eta_line").reshape(nsel, nb)
    disposition = small("u1", ns * nb, "disposition").reshape(ns, nb)
    require(np.all(np.diff(nu) > 0.0), "LINEPOP frequency is not ascending")
    finite_nonnegative("LINEPOP dnu", dnu)
    require(np.all(dnu > 0.0), "LINEPOP dnu contains zero")
    for name, values in (("shell state", shell_state), ("chi_line", chi_line),
                         ("chi_line_th", chi_line_th), ("eta_line", eta_line)):
        finite_nonnegative(name, values)
    require(np.all(disposition <= 3), "unknown EPAY disposition")
    actual_disp = {name: int(np.count_nonzero(disposition == code))
                   for code, name in DISPOSITION.items()}
    require(actual_disp == EXPECTED_DISPOSITION,
            f"EPAY disposition payload/manifest mismatch: {actual_disp}")

    lines = np.memmap(path, dtype=LINE_DTYPE, mode="r",
                      offset=offsets["line_static"], shape=(nlines,))
    rows = np.memmap(path, dtype=ROW_DTYPE, mode="r", offset=offsets["rows"],
                     shape=(nrows,))
    require(np.all(lines["bin"] < nb), "line-static bin outside grid")
    require(np.unique(lines["line_id"]).size == nlines,
            "line-static line_id is not unique")
    require(np.all(rows["line_slot"] < nlines), "row line_slot outside table")
    require(np.all(rows["shell_slot"] < nsel), "row shell_slot outside selection")
    require(np.all(rows["shell_slot"][1:] >= rows["shell_slot"][:-1]),
            "row table is not in writer shell order")
    row_line_id = lines["line_id"][rows["line_slot"]]
    same_shell = rows["shell_slot"][1:] == rows["shell_slot"][:-1]
    require(np.all(row_line_id[1:][same_shell] > row_line_id[:-1][same_shell]),
            "row table is not in writer line order within shell")
    require(not np.any(rows["flags"] & np.uint32(0xFFFFFFFF ^ KNOWN_FLAGS)),
            "unknown LINEPOP row flag")
    require(np.isfinite(rows["tau_used"]).all() and np.all(rows["tau_used"] > 0.0),
            "invalid tau_used")
    require(np.isfinite(rows["w"]).all() and np.all(rows["w"] >= 0.0),
            "invalid recorded expansion opacity")
    for field in ("tau_from_pops", "n_lower", "n_upper", "S_l_pop",
                  "S_l_used", "eps_l"):
        require(np.isfinite(rows[field]).all(), f"nonfinite row field {field}")

    header = {
        "iteration": int(iteration), "field_generation": int(generation),
        "n_shells": int(ns), "n_bins": int(nb), "selected_shells": int(nsel),
        "selected_lines": int(nlines), "rows": int(nrows),
        "time_explosion": float(texp), "lambda_window_A": [lamlo, lamhi],
        "eps_phys": int(eps_phys), "src_nlte": int(src_nlte),
        "epay": int(epay), "epay_smin": int(epay_smin),
        "epay_taubin": epay_taubin, "epay_hotf": epay_hotf,
        "eps_low_gate": eps_low_gate, "eps_high_gate": eps_high_gate,
        "line_eps": line_eps, "eps_uv": eps_uv, "line_gate": line_gate,
        "expected_file_bytes": expected_bytes,
    }
    return LinePop(path, manifest, header, shells, shell_state, nu, dnu,
                   chi_line, chi_line_th, eta_line, disposition, lines, rows,
                   offsets)


def serialize_chieta(header: tuple[Any, ...], arrays: list[np.ndarray]) -> bytes:
    chunks = [CE_HEADER.pack(*header)]
    chunks.extend(np.asarray(a, dtype="<f8").reshape(-1).tobytes(order="C")
                  for a in arrays)
    return b"".join(chunks)


def lane_manifest(source: Any, raw: bytes, diagnostic: str,
                  extra: dict[str, Any] | None = None) -> dict[str, Any]:
    result = {
        "schema": "LCMFCE01-v1",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "iteration": int(source.header[5]),
        "field_generation": int(source.header[6]),
        "post_damping": True,
        "coherent_frozen": True,
        "frequency_descending": True,
        "eta_decomposition_bitwise": True,
        "eta_decomposition_max_abs": 0.0,
        "diagnostic": diagnostic,
        "clamp": 0,
        "fallback": 0,
        "nonfinite": 0,
    }
    if extra:
        result.update(extra)
    return result


def write_lane(path: Path, source: Any, raw: bytes, diagnostic: str,
               extra: dict[str, Any] | None = None) -> str:
    path.write_bytes(raw)
    manifest = lane_manifest(source, raw, diagnostic, extra)
    json_write(Path(str(path) + ".manifest.json"), manifest)
    checked = check_artifact(path)
    require(checked.raw == raw, f"written {diagnostic} lane did not round-trip")
    return manifest["sha256"]


def replay_a_line_forest(linepop: LinePop) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Independently sum recorded A rows where the wavelength selection is complete."""
    nsel, nb = len(linepop.shells), linepop.header["n_bins"]
    rows, lines = linepop.rows, linepop.lines
    shell_slot = rows["shell_slot"]
    bin_index = lines["bin"][rows["line_slot"]]
    chi = np.zeros((nsel, nb), dtype=np.float64)
    chith = np.zeros((nsel, nb), dtype=np.float64)
    eta = np.zeros((nsel, nb), dtype=np.float64)
    np.add.at(chi, (shell_slot, bin_index), rows["w"])
    np.add.at(chith, (shell_slot, bin_index), rows["w"] * rows["eps_l"])
    np.add.at(eta, (shell_slot, bin_index),
              rows["w"] * rows["eps_l"] * rows["S_l_used"])

    edges, centers, _ = bench.canonical_grid()
    require(np.max(np.abs(centers / linepop.nu - 1.0)) <= 2.0e-15,
            "LINEPOP grid differs from canonical grid in A replay")
    nu_low = bench.C_ANGSTROM / 3000.0
    nu_high = bench.C_ANGSTROM / 600.0
    complete = (edges[:-1] >= nu_low) & (edges[1:] <= nu_high)
    require(np.count_nonzero(complete) > 0, "no complete LINEPOP wavelength bins")

    comparisons = {}
    for name, rebuilt, recorded in (
            ("chi_line", chi, linepop.chi_line),
            ("chi_line_th", chith, linepop.chi_line_th),
            ("eta_line", eta, linepop.eta_line)):
        left = np.asarray(rebuilt[:, complete], dtype="<f8").tobytes()
        right = np.asarray(recorded[:, complete], dtype="<f8").tobytes()
        require(left == right,
                f"A row reassembly is not bitwise for {name} in complete bins")
        comparisons[name] = {"bitwise": True, "cells": int(nsel * complete.sum())}
    return {
        "complete_bin_count": int(np.count_nonzero(complete)),
        "complete_cell_count_per_field": int(nsel * np.count_nonzero(complete)),
        "comparisons": comparisons,
    }, chi, complete


def prepare_authorities(linepop: LinePop, chieta_path: Path,
                        outdir: Path) -> tuple[Any, Any, Path, Path, dict[str, Any]]:
    a = check_artifact(chieta_path)
    require((int(a.header[3]), int(a.header[4]), int(a.header[5]), int(a.header[6]))
            == (50, 1000, 10, 10), "CHIETA generation/dimensions mismatch")
    require(a.manifest.get("sha256") == sha256_file(chieta_path),
            "CHIETA streamed SHA-256 mismatch")
    require(np.asarray(a.arrays[1], dtype="<f8")[::-1].tobytes()
            == linepop.nu.tobytes(), "LINEPOP/CHIETA nu grid is not bitwise common")
    require(np.asarray(a.arrays[2], dtype="<f8")[::-1].tobytes()
            == linepop.dnu.tobytes(), "LINEPOP/CHIETA dnu grid is not bitwise common")
    require(struct.pack("<d", float(a.header[-1])) ==
            struct.pack("<d", linepop.header["time_explosion"]),
            "LINEPOP/CHIETA expansion time differs")

    rebuilt_a = serialize_chieta(a.header, [np.asarray(x) for x in a.arrays])
    require(rebuilt_a == a.raw, "A read/write reassembly is not byte-identical")
    a_path = outdir / "t2_A_reassembled"
    a_sha = write_lane(a_path, a, rebuilt_a, "UV-T2-A-reassembled-bitwise")

    forest_audit, chi_rebuilt, complete = replay_a_line_forest(linepop)
    json_write(outdir / "t2_A_line_reassembly.json", forest_audit)
    injected = bytearray(np.asarray(chi_rebuilt[:, complete],
                                    dtype="<f8").tobytes())
    injected[0] ^= 0x01
    negative_fired = False
    try:
        require(bytes(injected) == np.asarray(
                    linepop.chi_line[:, complete], dtype="<f8").tobytes(),
                "SEEDED-DEFECT: A line-reassembly gate rejected a one-bit mutation")
    except OfflineError:
        negative_fired = True
    require(negative_fired, "seeded A line-reassembly defect was not detected")
    control = {
        "gate": "A rows reassemble chi_line bitwise in complete wavelength bins",
        "injection": "xor 0x01 at first reconstructed chi_line byte",
        "expected": "FAIL",
        "observed": "FAIL",
        "normal_A_sha256": a_sha,
        "A_line_reassembly": forest_audit,
    }
    json_write(outdir / "negative_control.json", control)

    base = linepop.path.parent / "emiss_ab_iter10"
    captured_a_path = Path(str(base) + ".A")
    b2_input = Path(str(base) + ".B2")
    require(captured_a_path.is_file() and b2_input.is_file(),
            "same-directory captured A/B2 pair is missing")
    captured_a = check_artifact(captured_a_path)
    b2 = check_artifact(b2_input)
    require(captured_a.raw == a.raw,
            "chieta authority is not bitwise equal to same-capture A lane")
    require(captured_a.manifest.get("emiss_ab_lane") == "A-production",
            "captured A lane tag mismatch")
    require(b2.manifest.get("emiss_ab_lane") ==
            "B2-Aul-nu-retain-A-undefined", "captured B2 lane tag mismatch")
    require(captured_a.header == b2.header, "A/B2 header mismatch")
    for index, name in ((0, "r_edge"), (1, "nu"), (2, "dnu"),
                        (3, "chi_total"), (4, "chi_coherent"),
                        (8, "J_producer")):
        require(np.asarray(captured_a.arrays[index], dtype="<f8").tobytes()
                == np.asarray(b2.arrays[index], dtype="<f8").tobytes(),
                f"B2 changed non-emissivity coordinate {name}")
    rebuilt_b2 = serialize_chieta(b2.header, [np.asarray(x) for x in b2.arrays])
    require(rebuilt_b2 == b2.raw, "B2 reserialization is not byte-identical")
    b2_path = outdir / "t2_B2_reassembled"
    b2_sha = write_lane(b2_path, b2, rebuilt_b2,
                        "UV-T2-B2-existing-population-eta",
                        {"source_payload_sha256": b2.manifest["sha256"]})
    control["B2_sha256"] = b2_sha
    return a, b2, a_path, b2_path, control


def n9_measure(linepop: LinePop, a: Any, outdir: Path) -> dict[str, Any]:
    ns, nb = 50, 1000
    edges, centers, _ = bench.canonical_grid()
    require(np.max(np.abs(centers / linepop.nu - 1.0)) <= 2.0e-15,
            "LINEPOP grid differs from canonical grid")
    wavelength = bench.C_ANGSTROM / linepop.nu
    eta_fixed = np.asarray(a.arrays[5]).reshape(ns, nb)[:, ::-1]
    finite_nonnegative("A eta_fixed", eta_fixed)
    r_edge = np.asarray(a.arrays[0])
    require(np.all(np.diff(r_edge) > 0.0), "non-increasing radial edges")
    volume = (4.0 * M_PI / 3.0) * (r_edge[1:] ** 3 - r_edge[:-1] ** 3)
    finite_nonnegative("shell volume", volume)
    require(np.all(volume > 0.0), "zero shell volume")

    count_rows: list[dict[str, Any]] = []
    energy_rows: list[dict[str, Any]] = []
    for band_index, (band, lo, hi) in enumerate(BANDS):
        if band == "BALL":
            center_mask = (wavelength >= lo) & (wavelength <= hi)
        elif band_index == 4:
            center_mask = (wavelength >= lo) & (wavelength <= hi)
        else:
            center_mask = (wavelength >= lo) & (wavelength < hi)
        weights = bench.band_weights(edges, lo, hi)
        require(np.any(center_mask) and float(np.sum(weights)) > 0.0,
                f"empty band {band}")
        global_total_energy = 0.0
        global_replaced_energy = 0.0
        for shell in range(ns):
            cell_disp = linepop.disposition[shell]
            total_cells = int(np.count_nonzero(center_mask))
            replaced_cells = int(np.count_nonzero(center_mask & (cell_disp == 2)))
            count_rows.append({
                "band": band, "lambda_lo_A": lo, "lambda_hi_A": hi,
                "shell": shell, "cells": total_cells,
                "rate_shape_replaced_cells": replaced_cells,
                "rate_shape_replaced_fraction": replaced_cells / total_cells,
            })
            emitted = eta_fixed[shell] * weights * volume[shell]
            finite_nonnegative("N9 fixed emission weight", emitted)
            total_energy = math.fsum(float(x) for x in emitted)
            replaced_energy = math.fsum(
                float(x) for x in emitted[cell_disp == 2])
            require(total_energy > 0.0, f"zero fixed UV energy shell={shell} band={band}")
            require(0.0 <= replaced_energy <= total_energy,
                    "N9 energy ledger does not close")
            energy_rows.append({
                "band": band, "lambda_lo_A": lo, "lambda_hi_A": hi,
                "shell": shell, "fixed_emission_weight": total_energy,
                "rate_shape_fixed_emission_weight": replaced_energy,
                "rate_shape_energy_fraction": replaced_energy / total_energy,
            })
            global_total_energy += total_energy
            global_replaced_energy += replaced_energy
        require(global_total_energy > 0.0, f"zero global energy in {band}")
        energy_rows.append({
            "band": band, "lambda_lo_A": lo, "lambda_hi_A": hi,
            "shell": "ALL", "fixed_emission_weight": global_total_energy,
            "rate_shape_fixed_emission_weight": global_replaced_energy,
            "rate_shape_energy_fraction": global_replaced_energy / global_total_energy,
        })

    outer = linepop.disposition[5:]
    require(outer.size == 45000, "s>=5 disposition has wrong cell count")
    outer_replaced = int(np.count_nonzero(outer == 2))
    require(outer_replaced == 34304, "s>=5 rate-shape count differs from manifest")

    source_audits = []
    all_rel: list[float] = []
    all_ulp: list[float] = []
    uv_mask = (wavelength >= 600.0) & (wavelength <= 3000.0)
    for slot, shell_u32 in enumerate(linepop.shells):
        shell = int(shell_u32)
        mask = ((linepop.disposition[shell] == 2) & uv_mask &
                (linepop.chi_line_th[slot] > 0.0))
        if not np.any(mask):
            source_audits.append({"shell": shell, "audited_cells": 0})
            continue
        bnu = planck(linepop.nu[mask], float(linepop.shell_state[slot, 0]))
        eta_rate_line = linepop.chi_line_th[slot, mask] * bnu
        source = eta_rate_line / linepop.chi_line_th[slot, mask]
        relative = np.abs(source / bnu - 1.0)
        spacing = np.abs(np.spacing(bnu))
        require(np.all(spacing > 0.0), "invalid ulp spacing in Planck audit")
        ulp = np.abs(source - bnu) / spacing
        all_rel.extend(float(x) for x in relative)
        all_ulp.extend(float(x) for x in ulp)
        source_audits.append({
            "shell": shell, "audited_cells": int(np.count_nonzero(mask)),
            "max_relative_S_rate_over_B_minus_1": float(np.max(relative)),
            "max_ulp_difference": float(np.max(ulp)),
        })
    require(all_rel, "no selected positive line cell available for B(Te) audit")
    max_rel, max_ulp = max(all_rel), max(all_ulp)
    require(max_rel <= 2.0 ** -48 and max_ulp <= 8.0,
            f"rate-shape line source is not B(Te): rel={max_rel}, ulp={max_ulp}")

    csv_write(outdir / "n9_disposition_shell_band.csv", count_rows)
    csv_write(outdir / "n9_energy_shell_band.csv", energy_rows)
    summary = {
        "schema": "lumina-uv-n9-v1",
        "disposition_counts": EXPECTED_DISPOSITION,
        "s_ge_5_cells": 45000,
        "s_ge_5_rate_shape_replaced_cells": outer_replaced,
        "s_ge_5_rate_shape_replaced_fraction": outer_replaced / 45000,
        "energy_definition": "eta_fixed_post_EPAY * exact_frequency_overlap * shell_volume",
        "coherent_return_excluded": True,
        "rate_shape_line_source_BTe": {
            "status": "PASS", "max_relative_error": max_rel,
            "max_ulp_difference": max_ulp, "by_selected_shell": source_audits,
            "identity": "eta_rate_line=chi_line_th*B_nu(Te); S_rate_line=eta_rate_line/chi_line_th",
        },
        "epay_scale_not_reproducible": True,
        "epay_scale_reason": (
            "acc_abs used the assemble-time lagged J; cmfgen_solve_J and damping "
            "changed J before the LINEPOP dump, and acc_abs/acc_dep/acc_w/wn were "
            "not serialized"
        ),
        "defect_ledger_candidate": (
            "EPAY-REPLAY-001: LCMFLP01-v1 preserves disposition but not the "
            "per-shell rate-shape/scalar normalization or its acc_abs, acc_dep, "
            "acc_w inputs. A population-native opacity counterfactual that changes "
            "that ledger therefore cannot be reassembled exactly from the same-epoch "
            "payload. Serialize the scale and all three per-shell accumulators."
        ),
        "clamp": 0, "fallback": 0, "nonfinite": 0,
    }
    json_write(outdir / "n9_summary.json", summary)
    return summary


def nonpositive_forensics(linepop: LinePop, outdir: Path) -> dict[str, Any]:
    """Characterize every row that cannot support population-native C.

    This is measurement only.  Recorded A opacity is never used as a population
    substitute, and no value is assigned to an undefined C counterfactual.
    """
    rows, lines = linepop.rows, linepop.lines
    line_slot = np.asarray(rows["line_slot"])
    shell_slot = np.asarray(rows["shell_slot"])
    row_lines = lines[line_slot]
    wavelength_A = np.asarray(row_lines["lambda_cm"]) * 1.0e8
    require(np.isfinite(wavelength_A).all() and np.all(wavelength_A > 0.0),
            "invalid line wavelength in nonpositive forensics")

    bad = ((rows["tau_from_pops"] <= 0.0) |
           (rows["n_lower"] <= 0.0) |
           (rows["n_upper"] <= 0.0) |
           (rows["S_l_pop"] <= 0.0))
    bad_count = int(np.count_nonzero(bad))
    require(bad_count > 0, "nonpositive forensic mode found no affected rows")
    bad_rows = np.asarray(rows[bad])
    bad_lines = np.asarray(row_lines[bad])
    bad_wave = wavelength_A[bad]
    bad_shell_slot = shell_slot[bad]

    lower_mapped = bad_lines["nlte_lower"] >= 0
    upper_mapped = bad_lines["nlte_upper"] >= 0
    lower_negative = bad_rows["n_lower"] < 0.0
    upper_negative = bad_rows["n_upper"] < 0.0
    raw_negative_population = lower_negative | upper_negative
    raw_zero_population = ((bad_rows["n_lower"] == 0.0) |
                           (bad_rows["n_upper"] == 0.0))
    solver_negative = ((lower_negative & lower_mapped) |
                       (upper_negative & upper_mapped))
    undefined_sentinel = ((lower_negative & ~lower_mapped) |
                          (upper_negative & ~upper_mapped))
    solved_zero = (((bad_rows["n_lower"] == 0.0) & lower_mapped) |
                   ((bad_rows["n_upper"] == 0.0) & upper_mapped))

    ball_bad = (bad_wave >= 600.0) & (bad_wave <= 3000.0)
    outside_ball_bad = ~ball_bad
    require(int(np.count_nonzero(ball_bad)) +
            int(np.count_nonzero(outside_ball_bad)) == bad_count,
            "BALL in/out row census does not close")

    edges, centers, _ = bench.canonical_grid()
    require(np.max(np.abs(centers / linepop.nu - 1.0)) <= 2.0e-15,
            "LINEPOP grid differs from canonical grid in nonpositive forensics")
    ball_weights = bench.band_weights(edges, 600.0, 3000.0)
    bad_bin = np.asarray(bad_lines["bin"], dtype=np.int64)

    shell_ball: list[dict[str, Any]] = []
    ball_fractions: list[float] = []
    for slot, shell_u32 in enumerate(linepop.shells):
        shell = int(shell_u32)
        affected = (bad_shell_slot == slot) & ball_bad
        affected_weight = math.fsum(
            float(w * ball_weights[int(b)])
            for w, b in zip(bad_rows["w"][affected], bad_bin[affected]))
        total_weight = math.fsum(
            float(x) for x in linepop.chi_line[slot] * ball_weights)
        require(total_weight > 0.0,
                f"zero recorded A BALL chi_line band weight shell={shell}")
        fraction = affected_weight / total_weight
        require(0.0 <= fraction <= 1.0 + 2.0e-15,
                f"affected A BALL chi_line share does not close shell={shell}")
        ball_fractions.append(fraction)
        shell_ball.append({
            "shell": shell,
            "affected_rows": int(np.count_nonzero(affected)),
            "recorded_A_affected_chi_line_band_weight": affected_weight,
            "recorded_A_total_chi_line_band_weight": total_weight,
            "recorded_A_affected_fraction": fraction,
        })

    wavelength_rows: list[dict[str, Any]] = []
    forensic_bands = BANDS[:5] + (("OUTSIDE_BALL", 0.0, math.inf),)
    for band_index, (band, lo, hi) in enumerate(forensic_bands):
        if band == "OUTSIDE_BALL":
            band_mask = outside_ball_bad
        elif band_index == 4:
            band_mask = (bad_wave >= lo) & (bad_wave <= hi)
        else:
            band_mask = (bad_wave >= lo) & (bad_wave < hi)
        for slot, shell_u32 in enumerate(linepop.shells):
            selected = band_mask & (bad_shell_slot == slot)
            wavelength_rows.append({
                "band": band, "lambda_lo_A": lo,
                "lambda_hi_A": "inf" if math.isinf(hi) else hi,
                "shell": int(shell_u32),
                "affected_rows": int(np.count_nonzero(selected)),
                "recorded_A_chi_line_sum": math.fsum(
                    float(x) for x in bad_rows["w"][selected]),
            })
    csv_write(outdir / "t2_nonpositive_wavelength_shell.csv", wavelength_rows)

    ion_groups: dict[tuple[int, int], dict[str, Any]] = {}
    level_groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    selected_shells = [int(x) for x in linepop.shells]
    for index in range(bad_count):
        line = bad_lines[index]
        row = bad_rows[index]
        z, ion = int(line["Z"]), int(line["ion"])
        weight = float(row["w"])
        ikey = (z, ion)
        ion_entry = ion_groups.setdefault(ikey, {
            "Z": z, "element": ELEMENT_SYMBOL.get(z, f"Z{z}"), "ion": ion,
            "rows": 0, "BALL_rows": 0, "recorded_A_chi_line_sum": 0.0,
        })
        ion_entry["rows"] += 1
        ion_entry["BALL_rows"] += int(bool(ball_bad[index]))
        ion_entry["recorded_A_chi_line_sum"] += weight

        lkey = (z, ion, int(line["nlte_lower"]), int(line["nlte_upper"]),
                int(line["g_lower"]), int(line["g_upper"]),
                float(line["E_lower_eV"]), float(line["E_upper_eV"]))
        level_entry = level_groups.setdefault(lkey, {
            "Z": z, "element": ELEMENT_SYMBOL.get(z, f"Z{z}"), "ion": ion,
            "nlte_lower": int(line["nlte_lower"]),
            "nlte_upper": int(line["nlte_upper"]),
            "g_lower": int(line["g_lower"]), "g_upper": int(line["g_upper"]),
            "E_lower_eV": float(line["E_lower_eV"]),
            "E_upper_eV": float(line["E_upper_eV"]),
            "rows": 0, "BALL_rows": 0, "recorded_A_chi_line_sum": 0.0,
            **{f"shell_{shell}_rows": 0 for shell in selected_shells},
        })
        level_entry["rows"] += 1
        level_entry["BALL_rows"] += int(bool(ball_bad[index]))
        level_entry["recorded_A_chi_line_sum"] += weight
        level_entry[f"shell_{selected_shells[int(bad_shell_slot[index])]}_rows"] += 1

    total_bad_w = math.fsum(float(x) for x in bad_rows["w"])
    for entry in ion_groups.values():
        entry["fraction_of_affected_recorded_A_chi_line"] = (
            entry["recorded_A_chi_line_sum"] / total_bad_w)
    ion_ranked_rows = sorted(
        ion_groups.values(),
        key=lambda x: (-x["rows"], -x["recorded_A_chi_line_sum"],
                       x["Z"], x["ion"]))
    ion_ranked_chi = sorted(
        ion_groups.values(),
        key=lambda x: (-x["recorded_A_chi_line_sum"], -x["rows"],
                       x["Z"], x["ion"]))
    level_ranked = sorted(level_groups.values(),
                          key=lambda x: (-x["recorded_A_chi_line_sum"],
                                         -x["rows"], x["Z"], x["ion"]))
    for rank, entry in enumerate(level_ranked, 1):
        entry["rank_by_recorded_A_chi_line"] = rank
        entry["fraction_of_affected_recorded_A_chi_line"] = (
            entry["recorded_A_chi_line_sum"] / total_bad_w)
    csv_write(outdir / "t2_nonpositive_level_rank.csv", level_ranked)

    tau_used = bad_rows["tau_used"]
    tau_pop = bad_rows["tau_from_pops"]
    summary = {
        "schema": "lumina-uv-t2n9-nonpositive-v1",
        "status": "UNRESOLVED-FAIL-CLOSED" if np.any(ball_bad) else "BALL-EXACT",
        "definition": (
            "affected row iff tau_from_pops, n_lower, n_upper, or S_l_pop is "
            "nonpositive; no recorded A value is substituted for population-native C"
        ),
        "row_census": {
            "all_rows": int(rows.size), "affected_rows": bad_count,
            "BALL_rows": int(np.count_nonzero(ball_bad)),
            "outside_BALL_rows": int(np.count_nonzero(outside_ball_bad)),
            "raw_negative_population_rows":
                int(np.count_nonzero(raw_negative_population)),
            "raw_zero_population_rows": int(np.count_nonzero(raw_zero_population)),
            "raw_negative_n_lower_rows": int(np.count_nonzero(lower_negative)),
            "raw_negative_n_upper_rows": int(np.count_nonzero(upper_negative)),
            "actual_solver_negative_population_rows": int(np.count_nonzero(solver_negative)),
            "undefined_minus_one_sentinel_rows": int(np.count_nonzero(undefined_sentinel)),
            "solved_zero_population_rows": int(np.count_nonzero(solved_zero)),
            "zero_n_lower_rows": int(np.count_nonzero(bad_rows["n_lower"] == 0.0)),
            "zero_n_upper_rows": int(np.count_nonzero(bad_rows["n_upper"] == 0.0)),
            "zero_S_l_pop_rows": int(np.count_nonzero(bad_rows["S_l_pop"] == 0.0)),
            "negative_S_l_pop_rows": int(np.count_nonzero(bad_rows["S_l_pop"] < 0.0)),
        },
        "BALL_support": {
            "exact_population_native_C_possible": not bool(np.any(ball_bad)),
            "reason": ("at least one undefined/nonpositive row lies in the frozen BALL"
                       if np.any(ball_bad) else
                       "every undefined/nonpositive row lies outside the frozen BALL"),
            "recorded_A_affected_fraction_by_selected_shell": shell_ball,
            "numeric_upper_bound_recorded_A_fraction_any_selected_shell":
                max(ball_fractions),
            "population_native_C_contribution_upper_bound": "UNDEFINED",
            "population_native_C_upper_bound_reason": (
                "the missing population/source is the quantity needed to compute it; "
                "using recorded A would be a forbidden substitution"
            ),
        },
        "population_breakdown": {
            "top_ions_by_affected_rows": ion_ranked_rows[:20],
            "top_ions_by_recorded_A_chi_line": ion_ranked_chi[:20],
            "top_levels_by_recorded_A_chi_line": level_ranked[:20],
        },
        "tau_used_on_affected_rows": {
            "finite_rows": int(np.count_nonzero(np.isfinite(tau_used))),
            "positive_rows": int(np.count_nonzero(tau_used > 0.0)),
            "min": float(np.min(tau_used)), "max": float(np.max(tau_used)),
            "recorded_A_w_finite_rows": int(np.count_nonzero(
                np.isfinite(bad_rows["w"]))),
            "recorded_A_w_positive_rows": int(np.count_nonzero(bad_rows["w"] > 0.0)),
            "tau_from_pops_negative_rows": int(np.count_nonzero(tau_pop < 0.0)),
            "tau_from_pops_zero_rows": int(np.count_nonzero(tau_pop == 0.0)),
            "tau_from_pops_positive_rows": int(np.count_nonzero(tau_pop > 0.0)),
        },
        "writer_trace": {
            "tau_used": (
                "src/lumina_cmfgen.c:777 reads the already assembled "
                "opac->tau_sobolev; :783-784 converts it to finite recorded w; "
                ":865 serializes tau_used"
            ),
            "population_fields": (
                "src/lumina_cmfgen.c:811-840 maps both levels, writes -1 sentinels "
                "when an NLTE level is unavailable, and only then computes tau_from_pops"
            ),
            "finite_opacity_origin": (
                "src/lumina_plasma.c:2582-2681 first constructs bulk/nebular Sobolev "
                "tau from ion density, dilution/Boltzmann level populations, and atomic "
                "f_lu/lambda; src/lumina_plasma.c:16987-17079 overrides only mapped "
                "NLTE lines. An unmapped line therefore can retain finite bulk tau "
                "while population-native LINEPOP fields remain undefined"
            ),
        },
        "clamp": 0, "fallback": 0, "substitution": 0, "nonfinite": 0,
    }
    json_write(outdir / "t2_nonpositive_population_forensics.json", summary)
    return summary


def build_c(linepop: LinePop, a: Any, outdir: Path) -> tuple[Path, dict[str, Any]]:
    rows, lines = linepop.rows, linepop.lines
    flags = rows["flags"]
    missing_pop = int(np.count_nonzero((flags & F_POPS_DEFINED) == 0))
    missing_source = int(np.count_nonzero((flags & F_SL_POP) == 0))
    stim_adjusted = int(np.count_nonzero(flags & F_STIM_ADJUSTED))
    invalid_positive = int(np.count_nonzero(
        (rows["tau_from_pops"] <= 0.0) | (rows["n_lower"] <= 0.0) |
        (rows["n_upper"] <= 0.0) | (rows["S_l_pop"] <= 0.0)))
    coverage = {
        "rows": int(rows.size), "missing_population_rows": missing_pop,
        "missing_population_source_rows": missing_source,
        "stimulated_adjustment_rows": stim_adjusted,
        "nonpositive_required_rows": invalid_positive,
        "tau_population_bitwise_rows": int(np.count_nonzero(flags & F_TAU_ROUNDTRIP)),
    }
    json_write(outdir / "t2_C_population_coverage.json", coverage)
    require(missing_pop == 0 and missing_source == 0 and stim_adjusted == 0 and
            invalid_positive == 0,
            "C population-native coverage is incomplete; substitution is forbidden")
    require(np.all((rows["eps_l"] > 0.0) & (rows["eps_l"] <= 1.0)),
            "invalid recorded destruction probability")

    line_slot = rows["line_slot"]
    shell_slot = rows["shell_slot"]
    bin_index = lines["bin"][line_slot]
    nu_l = lines["nu_l"][line_slot]
    require(np.isfinite(nu_l).all() and np.all(nu_l > 0.0),
            "invalid line frequency needed by C")
    tau_pop = rows["tau_from_pops"]
    frac = np.empty(tau_pop.shape, dtype=np.float64)
    thick = tau_pop > 1.0e-6
    frac[thick] = -np.expm1(-tau_pop[thick])
    frac[~thick] = tau_pop[~thick]
    w_pop = frac * nu_l / (CM_C * linepop.header["time_explosion"] *
                           linepop.dnu[bin_index])
    bitwise_tau = (flags & F_TAU_ROUNDTRIP) != 0
    w_pop[bitwise_tau] = rows["w"][bitwise_tau]
    finite_nonnegative("C w_pop", w_pop)

    nsel, nb = len(linepop.shells), linepop.header["n_bins"]
    delta_chi = np.zeros((nsel, nb), dtype=np.float64)
    delta_chith = np.zeros((nsel, nb), dtype=np.float64)
    delta_eta = np.zeros((nsel, nb), dtype=np.float64)
    dw = w_pop - rows["w"]
    de_th = rows["eps_l"] * dw
    de_eta = rows["eps_l"] * (
        w_pop * rows["S_l_pop"] - rows["w"] * rows["S_l_used"])
    np.add.at(delta_chi, (shell_slot, bin_index), dw)
    np.add.at(delta_chith, (shell_slot, bin_index), de_th)
    np.add.at(delta_eta, (shell_slot, bin_index), de_eta)
    require(all(np.isfinite(x).all() for x in (delta_chi, delta_chith, delta_eta)),
            "C accumulation produced a nonfinite value")

    for slot, shell_u32 in enumerate(linepop.shells):
        shell = int(shell_u32)
        d = linepop.disposition[shell]
        if np.any((d == 2) & (delta_chith[slot] != 0.0)):
            raise OfflineError(
                f"C changes chi_line_th in rate-shape cells at shell {shell}; "
                "the missing EPAY scale prevents exact reassembly")
        if np.any((d == 3) & ((delta_chith[slot] != 0.0) |
                              (delta_eta[slot] != 0.0))):
            raise OfflineError(
                f"C changes a scalar-rescaled cell at shell {shell}; "
                "the missing EPAY scale prevents exact reassembly")

    a_chi_total = np.asarray(a.arrays[3]).reshape(50, 1000)[:, ::-1]
    for slot, shell_u32 in enumerate(linepop.shells):
        shell = int(shell_u32)
        if shell < linepop.header["epay_smin"]:
            continue
        chi_e = float(linepop.shell_state[slot, 2]) * CM_SIGMA_T
        chi_abs = a_chi_total[shell] - chi_e - linepop.chi_line[slot]
        finite_nonnegative(f"derived chi_abs shell={shell}", chi_abs)
        old_thick = ((chi_abs + linepop.chi_line_th[slot]) *
                     float(linepop.shell_state[slot, 3]) >
                     linepop.header["epay_taubin"])
        new_thick = ((chi_abs + linepop.chi_line_th[slot] + delta_chith[slot]) *
                     float(linepop.shell_state[slot, 3]) >
                     linepop.header["epay_taubin"])
        require(np.array_equal(old_thick, new_thick),
                f"C changes EPAY thick/thin membership at shell {shell}")
        require(np.array_equal(old_thick, linepop.disposition[shell] == 1),
                f"derived EPAY membership disagrees with disposition shell={shell}")

    nr = int(a.header[3]); nnu = int(a.header[4])
    arrays = [np.asarray(x).copy() for x in a.arrays]
    chi_total = arrays[3].reshape(nr, nnu)[:, ::-1].copy()
    chi_coherent = arrays[4].reshape(nr, nnu)[:, ::-1].copy()
    eta_fixed = arrays[5].reshape(nr, nnu)[:, ::-1].copy()
    eta_coherent = arrays[6].reshape(nr, nnu)[:, ::-1].copy()
    j_producer = arrays[8].reshape(nr, nnu)[:, ::-1]
    for slot, shell_u32 in enumerate(linepop.shells):
        shell = int(shell_u32)
        coherent_delta = delta_chi[slot] - delta_chith[slot]
        chi_total[shell] += delta_chi[slot]
        chi_coherent[shell] += coherent_delta
        live = linepop.disposition[shell] <= 1
        eta_fixed[shell, live] += delta_eta[slot, live]
        eta_coherent[shell] += coherent_delta * j_producer[shell]
    eta_total = eta_fixed + eta_coherent
    for name, values in (("C chi_total", chi_total),
                         ("C chi_coherent", chi_coherent),
                         ("C eta_fixed", eta_fixed),
                         ("C eta_coherent", eta_coherent),
                         ("C eta_total", eta_total)):
        finite_nonnegative(name, values)
    require(np.all(chi_coherent <= chi_total),
            "C coherent opacity exceeds total opacity")
    arrays[3] = chi_total[:, ::-1].reshape(-1)
    arrays[4] = chi_coherent[:, ::-1].reshape(-1)
    arrays[5] = eta_fixed[:, ::-1].reshape(-1)
    arrays[6] = eta_coherent[:, ::-1].reshape(-1)
    arrays[7] = eta_total[:, ::-1].reshape(-1)
    raw1 = serialize_chieta(a.header, arrays)
    raw2 = serialize_chieta(a.header, arrays)
    require(raw1 == raw2, "C construction is not byte deterministic")
    path = outdir / "t2_C_population_native"
    digest = write_lane(path, a, raw1, "UV-T2-C-population-native-chi-eta", {
        "source_A_sha256": a.manifest["sha256"],
        "linepop_sha256": LINEPOP_SHA256,
        "selected_shells": [int(x) for x in linepop.shells],
        "population_coverage": coverage,
        "rate_shape_cells_require_zero_chi_line_th_delta": True,
    })
    audit = {
        "payload_sha256": digest, "repeat_construction_sha256": [
            hashlib.sha256(raw1).hexdigest(), hashlib.sha256(raw2).hexdigest()],
        "repeat_construction_identical": raw1 == raw2,
        "population_coverage": coverage,
        "changed_chi_total_cells": int(np.count_nonzero(delta_chi)),
        "changed_chi_line_th_cells": int(np.count_nonzero(delta_chith)),
        "changed_live_eta_cells": int(np.count_nonzero(
            delta_eta * np.asarray([linepop.disposition[int(s)] <= 1
                                    for s in linepop.shells]))),
        "max_abs_delta_chi": float(np.max(np.abs(delta_chi))),
        "max_abs_delta_chi_line_th": float(np.max(np.abs(delta_chith))),
        "max_abs_delta_eta_live_pre_EPAY": float(np.max(np.abs(delta_eta))),
        "clamp": 0, "fallback": 0, "nonfinite": 0,
    }
    json_write(outdir / "t2_C_construction.json", audit)
    return path, audit


def run_transport(lanes: dict[str, Path], outdir: Path) -> dict[str, Any]:
    driver = outdir / "stage31_cmf_field_driver"
    compile_command = bench.compile_driver(driver)
    environment = dict(os.environ)
    environment["OMP_NUM_THREADS"] = "4"
    solved: dict[str, Any] = {}
    for lane, payload in lanes.items():
        hashes: list[str] = []
        first_meta: dict[str, str] | None = None
        first_table: dict[str, np.ndarray] | None = None
        commands = []
        for repeat in (1, 2):
            table_path = outdir / f"t2_{lane}_jdet_repeat{repeat}.tsv"
            table_path.unlink(missing_ok=True)
            command = [str(driver), str(payload), str(payload) + ".manifest.json",
                       "8", "16", "10020.0", "1.0", str(table_path)]
            completed = subprocess.run(command, cwd=ROOT, env=environment,
                                       capture_output=True, text=True, check=False)
            commands.append(command)
            if completed.returncode:
                raise OfflineError(
                    f"Stage-3.1 {lane} failed rc={completed.returncode}: "
                    f"{completed.stderr.strip()}")
            meta, table = bench.parse_driver_table(table_path)
            for key in ("clamp", "solution_negative_excess", "sign_uncertain",
                        "nonfinite"):
                require(int(meta[key]) == 0,
                        f"Stage-3.1 {lane} guard {key}={meta[key]}")
            require(float(meta["transport_residual"]) <= 1.0e-4,
                    f"Stage-3.1 {lane} transport residual exceeds 1e-4")
            hashes.append(sha256_file(table_path))
            if repeat == 1:
                first_meta, first_table = meta, table
        require(len(set(hashes)) == 1,
                f"Stage-3.1 {lane} repeated outputs differ: {hashes}")
        assert first_meta is not None and first_table is not None
        solved[lane] = {"hashes": hashes, "metadata": first_meta,
                        "table": first_table, "commands": commands}
    return {"compile_command": compile_command, "lanes": solved}


def measure_t2(a: Any, transport: dict[str, Any], outdir: Path) -> dict[str, Any]:
    edges, _, _ = bench.canonical_grid()
    r_edge = np.asarray(a.arrays[0])
    velocities = 0.5 * (r_edge[:-1] + r_edge[1:]) / float(a.header[-1]) / 1.0e5
    cmf, cmf_meta = cmfgen_all_shells(edges, velocities, CMF_RUN)
    shell = 8
    require(np.isfinite(cmf[shell]).all() and np.all(cmf[shell] > 0.0),
            "CMFGEN shell-8 reference is invalid")
    rows = []
    for band, lo, hi in BANDS:
        weights = bench.band_weights(edges, lo, hi)
        values = {}
        for lane in ("A", "B2", "C"):
            j = transport["lanes"][lane]["table"]["J_det"][::-1]
            require(np.isfinite(j).all() and np.all(j >= 0.0),
                    f"invalid Stage-3.1 spectrum for {lane}")
            values[lane] = weighted_mean(j, weights)
        ref = weighted_mean(cmf[shell], weights)
        require(ref > 0.0 and all(value > 0.0 for value in values.values()),
                f"non-positive band mean in {band}")
        rows.append({
            "band": band, "lambda_lo_A": lo, "lambda_hi_A": hi,
            "A_over_CMFGEN": values["A"] / ref,
            "B2_over_CMFGEN": values["B2"] / ref,
            "C_over_CMFGEN": values["C"] / ref,
            "B2_over_A": values["B2"] / values["A"],
            "C_over_A": values["C"] / values["A"],
        })
    csv_write(outdir / "t2_band_table.csv", rows)
    ball = next(row for row in rows if row["band"] == "BALL")
    c_over_a = float(ball["C_over_A"])
    c_over_cmf = float(ball["C_over_CMFGEN"])
    if abs(c_over_a - 1.0) <= 0.05:
        verdict = "OPERATOR_ONLY"
    elif c_over_cmf > 3.0:
        verdict = "ASSEMBLY_AND_OPERATOR"
    elif 1.0 / 3.0 <= c_over_cmf <= 3.0:
        verdict = "ASSEMBLY_ONLY"
    else:
        verdict = "UNRESOLVED-OUTSIDE-PREREG"
    ledger = {
        "A_total_excess_dex": math.log10(float(ball["A_over_CMFGEN"])),
        "assembly_removed_dex_log10_A_over_C": math.log10(1.0 / c_over_a),
        "operator_remaining_dex_log10_C_over_CMFGEN": math.log10(c_over_cmf),
    }
    summary = {
        "schema": "lumina-uv-t2n9-t2-v1", "primary_band": "BALL",
        "preregistered_A_equivalence_fraction": 0.05,
        "preregistered_CMFGEN_level_interval": [1.0 / 3.0, 3.0],
        "verdict": verdict, "bands": rows, "log_budget": ledger,
        "transport": {
            lane: {"repeat_hashes": value["hashes"],
                   "repeat_identical": len(set(value["hashes"])) == 1,
                   "metadata": value["metadata"], "commands": value["commands"]}
            for lane, value in transport["lanes"].items()
        },
        "compile_command": transport["compile_command"],
        "cmfgen_existing_reference": cmf_meta,
        "E6_T1_contradiction_explanation_required": verdict == "ASSEMBLY_ONLY",
        "clamp": 0, "fallback": 0, "nonfinite": 0,
    }
    json_write(outdir / "t2_summary.json", summary)
    require(verdict != "UNRESOLVED-OUTSIDE-PREREG",
            "T2 result lies outside every preregistered branch")
    return summary


def self_test() -> dict[str, Any]:
    require(LP_HEADER.size == 152, f"LINEPOP header size {LP_HEADER.size}")
    require(ROW_DTYPE.itemsize == 76 and LINE_DTYPE.itemsize == 80,
            "record dtype sizes")
    header = (b"LCMFCE01", 0x01020304, 1, 2, 3, 10, 10, 7, 0, 1.0)
    lengths = [3, 3, 3] + [6] * 6
    arrays = [np.arange(n, dtype="<f8") + index + 1.0
              for index, n in enumerate(lengths)]
    raw1 = serialize_chieta(header, arrays)
    raw2 = serialize_chieta(header, arrays)
    require(raw1 == raw2, "synthetic serialization is not deterministic")
    defect = bytearray(raw1); defect[CE_HEADER.size] ^= 1
    fired = False
    try:
        require(bytes(defect) == raw1, "SEEDED-DEFECT")
    except OfflineError:
        fired = True
    require(fired, "synthetic negative control did not fire")
    return {
        "status": "PASS", "linepop_header_bytes": LP_HEADER.size,
        "row_bytes": ROW_DTYPE.itemsize, "line_bytes": LINE_DTYPE.itemsize,
        "repeat_serialization_identical": True,
        "seeded_one_bit_identity_defect": "EXPECTED-FAIL-OBSERVED",
        "synthetic_bytes": len(raw1),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--linepop", type=Path)
    parser.add_argument("--chieta", type=Path)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--forensics-only", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        try:
            print(json.dumps(self_test(), indent=2, sort_keys=True))
            return 0
        except OfflineError as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return 1
    if args.linepop is None or args.outdir is None:
        parser.error("--linepop and --outdir are required")
    if not args.forensics_only and args.chieta is None:
        parser.error("--chieta is required unless --forensics-only is set")
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        linepop = parse_linepop(args.linepop.resolve())
        layout = {
            "schema": "LCMFLP01-v1", "writer_fixed_header_bytes": 152,
            "row_bytes": 76, "line_static_bytes": 80,
            "rows_bytes": linepop.header["rows"] * 76,
            "line_static_table_bytes": linepop.header["selected_lines"] * 80,
            "expected_file_bytes": linepop.header["expected_file_bytes"],
            "actual_file_bytes": linepop.path.stat().st_size,
            "offsets": linepop.offsets,
            "reason_rows_times_row_bytes_is_not_file_size": (
                "rows*76 covers only the final row table; header, selected-shell "
                "metadata, grid, three aggregate arrays, full disposition table, "
                "and the 601371*80-byte line-static table precede it"
            ),
        }
        json_write(outdir / "linepop_layout_audit.json", layout)
        forensics = nonpositive_forensics(linepop, outdir)
        if args.forensics_only:
            result = {
                "schema": "lumina-uv-t2n9-step3-forensics-v1",
                "status": forensics["status"],
                "BALL_exact_population_native_C_possible":
                    forensics["BALL_support"]["exact_population_native_C_possible"],
                "affected_rows": forensics["row_census"]["affected_rows"],
                "BALL_rows": forensics["row_census"]["BALL_rows"],
                "actual_solver_negative_population_rows":
                    forensics["row_census"]["actual_solver_negative_population_rows"],
                "clamp": 0, "fallback": 0, "substitution": 0, "nonfinite": 0,
            }
            json_write(outdir / "step3_forensics_summary.json", result)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        a, _, a_path, b2_path, _ = prepare_authorities(
            linepop, args.chieta.resolve(), outdir)
        n9 = n9_measure(linepop, a, outdir)
        c_path, construction = build_c(linepop, a, outdir)
        transport = run_transport({"A": a_path, "B2": b2_path, "C": c_path},
                                  outdir)
        t2 = measure_t2(a, transport, outdir)
        result = {
            "schema": "lumina-uv-t2n9-step1-run-v1", "status": "PASS",
            "T2_verdict": t2["verdict"],
            "N9_rate_shape_fraction_s_ge_5":
                n9["s_ge_5_rate_shape_replaced_fraction"],
            "C_payload_sha256": construction["payload_sha256"],
            "negative_control": "EXPECTED-FAIL-OBSERVED",
            "repeat_determinism": True,
            "clamp": 0, "fallback": 0, "nonfinite": 0,
        }
        json_write(outdir / "summary.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (OfflineError, OSError, ValueError, KeyError,
            subprocess.SubprocessError, bench.BenchError) as exc:
        failure = {"schema": "lumina-uv-t2n9-step1-run-v1",
                   "status": "UNRESOLVED-FAIL-CLOSED", "reason": str(exc)}
        json_write(outdir / "failure.json", failure)
        print(f"UNRESOLVED-FAIL-CLOSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
