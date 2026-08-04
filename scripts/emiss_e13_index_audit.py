#!/usr/bin/env python3
"""E13 offline index/mirror and radiative macro-atom branch audit.

This tool never changes production state.  It reads an LFMAT001 capture and
LCMFCE01 frozen payloads, writes a counterfactual matrix with both frequency
indices mirrored, and measures native/mirrored frequency moments and frozen
source flows.  It also computes radiative-only Lucy macro-atom cascades for
Fe II/III directly from the selected atomic CSV files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from cmf_chieta_check import check_artifact  # noqa: E402
from emiss_e11_fluor_matrix import (  # noqa: E402
    EDGE_DTYPE, HEADER as LFMAT_HEADER, FluorMatrix,
    add_matrix_contract_args, read_fluor_matrix, read_fluor_matrix_from_args)
import stage31_cmf_field_bench as bench  # noqa: E402

H = 6.62607015e-27
EV = 1.602176634e-12
GROUP_HEADER = struct.Struct("<IIQ")
BANDS = (
    ("EUV", 100.0, 600.0),
    ("B0", 600.0, 1000.0),
    ("B1", 1000.0, 1500.0),
    ("B2", 1500.0, 2000.0),
    ("B3", 2000.0, 2500.0),
    ("B4", 2500.0, 3000.0),
    ("OPTICAL", 3000.0, 10000.0),
    ("IR", 10000.0, 20000.0),
)


class E13Error(RuntimeError):
    pass


def band_mask(wavelength: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (wavelength >= lo) & (wavelength < hi)


def mirror_edges(rows: np.ndarray, nbins: int) -> np.ndarray:
    result = np.asarray(rows, dtype=EDGE_DTYPE).copy()
    result["input_bin"] = nbins - 1 - result["input_bin"]
    result["output_bin"] = nbins - 1 - result["output_bin"]
    return result


def write_mirrored_matrix(matrix: FluorMatrix, path: Path) -> str:
    """Write R'[N-1-i,N-1-j]=R[i,j], including mirrored column ledgers."""
    h = matrix.header
    header = LFMAT_HEADER.pack(
        b"LFMAT001", 0x01020304, 1, int(h["flags"]), int(h["n_bins"]),
        int(h["n_shells"]), int(h["iteration"]), int(h["n_shell_groups"]),
        float(h["nu_min"]), float(h["nu_max"]), float(h["d_log_nu"]),
        int(h["events_total"]), int(h["classified_events"]),
        int(h["unclassified_input"]), int(h["unclassified_output"]),
        int(h["unclassified_energy"]), int(h["unclassified_route"]),
        int(h["kpacket_events"]), float(h["absorbed_energy"]),
        float(h["reemitted_energy"]), float(h["kpacket_absorbed_energy"]),
        float(h["kpacket_reemitted_energy"]), len(matrix.edges))
    arrays = (
        matrix.input_count[::-1].astype("<u8"),
        matrix.input_energy[::-1].astype("<f8"),
        matrix.terminal_energy[::-1].astype("<f8"),
        matrix.outside_energy[::-1].astype("<f8"),
        matrix.shell_count.astype("<u8"),
        matrix.shell_kpacket_count.astype("<u8"),
        matrix.shell_absorbed_energy.astype("<f8"),
        matrix.shell_reemitted_energy.astype("<f8"),
        mirror_edges(matrix.edges, int(h["n_bins"])),
    )
    pieces = [header, *(values.tobytes(order="C") for values in arrays)]
    for (first, last), rows in zip(matrix.group_ranges, matrix.group_edges):
        mirrored = mirror_edges(rows, int(h["n_bins"]))
        pieces.extend((GROUP_HEADER.pack(first, last, len(mirrored)),
                       mirrored.tobytes(order="C")))
    raw = b"".join(pieces)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    Path(str(path) + ".sha256").write_text(f"{digest}  {path}\n")
    checked = read_fluor_matrix(path, expected_iteration=int(h["iteration"]),
                                expected_sha256=digest,
                                non_contract_override=True)
    if checked.sha256 != digest:
        raise E13Error("mirrored LFMAT001 round-trip hash mismatch")
    return digest


def matrix_view(matrix: FluorMatrix, mirrored: bool) -> tuple[np.ndarray, ...]:
    nbins = int(matrix.header["n_bins"])
    ib = matrix.edges["input_bin"].astype(np.int64)
    ob = matrix.edges["output_bin"].astype(np.int64)
    terminal = matrix.terminal_energy
    outside = matrix.outside_energy
    if mirrored:
        ib = nbins - 1 - ib
        ob = nbins - 1 - ob
        terminal = terminal[::-1]
        outside = outside[::-1]
    return ib, ob, matrix.edges["output_energy"], terminal, outside


def frequency_metrics(matrix: FluorMatrix, mirrored: bool,
                      nu: np.ndarray, wavelength: np.ndarray) -> dict[str, Any]:
    ib, ob, energy, terminal, outside = matrix_view(matrix, mirrored)
    total = float(np.sum(energy))
    mean_in = float(np.sum(energy * nu[ib]) / total)
    mean_out = float(np.sum(energy * nu[ob]) / total)
    result: dict[str, Any] = {
        "on_grid_energy": total,
        "energy_weighted_mean_input_nu_Hz": mean_in,
        "energy_weighted_mean_output_nu_Hz": mean_out,
        "mean_output_minus_input_nu_Hz": mean_out - mean_in,
        "mean_output_over_input_nu": mean_out / mean_in,
        "lower_nu_energy_fraction": float(np.sum(energy[nu[ob] < nu[ib]]) / total),
        "same_nu_bin_energy_fraction": float(np.sum(energy[ob == ib]) / total),
        "higher_nu_energy_fraction": float(np.sum(energy[nu[ob] > nu[ib]]) / total),
    }
    for name, lo, hi in (("UV", 600.0, 3000.0), ("B2", 1500.0, 2000.0)):
        input_bins = band_mask(wavelength, lo, hi)
        selected = input_bins[ib]
        denominator = float(np.sum(energy[selected]) + np.sum(outside[input_bins]))
        result[f"{name}_input_terminal_energy"] = denominator
        for out_name, out_lo, out_hi in (
                ("UV", 600.0, 3000.0), ("B0", 600.0, 1000.0),
                ("OPTICAL", 3000.0, 10000.0)):
            out = band_mask(wavelength, out_lo, out_hi)
            result[f"{name}_to_{out_name}_fraction"] = float(
                np.sum(energy[selected & out[ob]]) / denominator)
        result[f"{name}_to_lower_nu_fraction"] = float(
            np.sum(energy[selected & (nu[ob] < nu[ib])]) / denominator)
        result[f"{name}_to_higher_nu_fraction"] = float(
            np.sum(energy[selected & (nu[ob] > nu[ib])]) / denominator)
    return result


def source_flow(matrix: FluorMatrix, mirrored: bool, e9_path: Path,
                source_path: Path, prereg: dict[str, Any], nu_edges: np.ndarray,
                wavelength: np.ndarray, shell: int) -> dict[str, Any]:
    e9 = check_artifact(e9_path.resolve())
    source = check_artifact(source_path.resolve())
    if e9.header != source.header:
        raise E13Error("E9/source LCMFCE01 headers differ")
    nr, nnu = int(e9.header[3]), int(e9.header[4])
    if nnu != int(matrix.header["n_bins"]) or not 0 <= shell < nr:
        raise E13Error("matrix/payload dimensions differ")
    e9_eta = np.asarray(e9.arrays[7]).reshape(nr, nnu)[:, ::-1]
    e9_j = np.asarray(e9.arrays[8]).reshape(nr, nnu)[:, ::-1]
    source_chi = np.asarray(source.arrays[4]).reshape(nr, nnu)[:, ::-1]
    chi_es = np.min(source_chi, axis=1)[:, None]
    chi_line = source_chi - chi_es
    if np.any(chi_line < 0.0):
        raise E13Error("negative line-opacity proxy")
    eta_line_return = (1.0 - float(prereg["eps_MC"])) * chi_line * e9_j
    widths = np.diff(nu_edges)
    removed = eta_line_return[shell] * widths
    ib, ob, energy, terminal, outside = matrix_view(matrix, mirrored)
    active = np.flatnonzero(terminal > 0.0)
    redistributed = np.zeros(nnu, dtype=np.float64)
    outside_power = np.zeros(nnu, dtype=np.float64)
    probability_sum = np.zeros(nnu, dtype=np.float64)
    probability_sum[active] = outside[active] / terminal[active]
    outside_power[active] = removed[active] * probability_sum[active]
    probability = energy / terminal[ib]
    np.add.at(probability_sum, ib, probability)
    np.add.at(redistributed, ob, removed[ib] * probability)
    b0 = band_mask(wavelength, 600.0, 1000.0)
    b2 = band_mask(wavelength, 1500.0, 2000.0)
    b0_total = float(np.sum(removed[ib] * probability * b0[ob]))
    b2_b0 = float(np.sum(removed[ib] * probability * b2[ib] * b0[ob]))
    eta_new = e9_eta.copy()
    eta_new[shell, active] -= eta_line_return[shell, active]
    eta_new[shell] += redistributed / widths
    band_ratios = {}
    for name, lo, hi in BANDS:
        mask = band_mask(wavelength, lo, hi)
        old = float(np.sum(e9_eta[shell, mask] * widths[mask]))
        new = float(np.sum(eta_new[shell, mask] * widths[mask]))
        band_ratios[name] = new / old
    removed_total = float(np.sum(removed[active]))
    return {
        "active_input_bins": int(len(active)),
        "operator_column_closure_max_abs": float(
            np.max(np.abs(probability_sum[active] - 1.0))),
        "removed_line_return_power": removed_total,
        "injected_on_grid_power": float(np.sum(redistributed)),
        "outside_grid_power": float(np.sum(outside_power)),
        "application_closure_relative": float(
            (np.sum(redistributed) + np.sum(outside_power)) / removed_total - 1.0),
        "B0_inflow_power": b0_total,
        "B2_to_B0_power": b2_b0,
        "B2_share_of_B0_inflow": b2_b0 / b0_total,
        "source_ratio_to_E9": band_ratios,
    }


def load_atomic_data(atomic_dir: Path, ion: int) -> tuple[np.ndarray, ...]:
    levels: dict[int, float] = {}
    with (atomic_dir / "levels.csv").open() as stream:
        for row in csv.DictReader(stream):
            if int(row["atomic_number"]) == 26 and int(row["ion_number"]) == ion:
                levels[int(row["level_number"])] = float(row["energy_eV"])
    if not levels:
        raise E13Error(f"no Fe ion={ion} levels")
    level_energy = np.asarray([levels.get(i, 0.0) for i in range(max(levels) + 1)])
    columns: list[list[float]] = [[] for _ in range(6)]
    with (atomic_dir / "line_list.csv").open() as stream:
        for row in csv.DictReader(stream):
            if int(row["atomic_number"]) != 26 or int(row["ion_number"]) != ion:
                continue
            values = (int(row["level_number_lower"]),
                      int(row["level_number_upper"]), float(row["nu"]),
                      float(row["A_ul"]), float(row["B_lu"]), float(row["B_ul"]))
            for column, value in zip(columns, values):
                column.append(value)
    return (level_energy, np.asarray(columns[0], dtype=np.int64),
            np.asarray(columns[1], dtype=np.int64),
            *(np.asarray(column, dtype=np.float64) for column in columns[2:]))


def ionization_offset(atomic_dir: Path, ion: int) -> float:
    values: dict[tuple[int, int], float] = {}
    with (atomic_dir / "ionization_energies.csv").open() as stream:
        for row in csv.DictReader(stream):
            values[(int(row["atomic_number"]), int(row["ion_number"]))] = float(
                row["ionization_energy_eV"])
    return sum(values[(26, stage)] for stage in range(ion))


def theoretical_cascade(atomic_dir: Path, ion: int, field_nu: np.ndarray,
                        field_j: np.ndarray) -> dict[str, Any]:
    energy_ev, lower, upper, line_nu, aul, blu, bul = load_atomic_data(atomic_dir, ion)
    offset_ev = ionization_offset(atomic_dir, ion)
    neutral_energy = (energy_ev + offset_ev) * EV
    jbar = np.interp(line_nu, field_nu, field_j, left=0.0, right=0.0)
    rate_down = aul + bul * jbar
    rate_up = blu * jbar
    w_emit = rate_down * H * line_nu
    w_idown = rate_down * neutral_energy[lower]
    w_iup = rate_up * neutral_energy[lower]
    total = np.zeros(len(energy_ev), dtype=np.float64)
    np.add.at(total, upper, w_emit + w_idown)
    np.add.at(total, lower, w_iup)
    p_emit = w_emit / np.where(total[upper] > 0.0, total[upper], 1.0)
    p_idown = w_idown / np.where(total[upper] > 0.0, total[upper], 1.0)
    p_iup = w_iup / np.where(total[lower] > 0.0, total[lower], 1.0)
    wavelength = bench.C_ANGSTROM / line_nu
    uv_entry = band_mask(wavelength, 600.0, 3000.0)
    # Energy-activation proxy.  Lower-level populations and stimulated-opacity
    # correction are unavailable in the atomic CSVs and intentionally omitted.
    entry_weight = blu * jbar * H * line_nu * uv_entry
    state = np.zeros(len(energy_ev), dtype=np.float64)
    np.add.at(state, upper, entry_weight)
    if not state.sum() > 0.0:
        raise E13Error(f"Fe ion={ion} has no UV entry weight")
    state /= state.sum()
    first = {
        "emit": float(np.sum(state[upper] * p_emit)),
        "internal_down": float(np.sum(state[upper] * p_idown)),
        "internal_up": float(np.sum(state[lower] * p_iup)),
    }
    exits = {name: 0.0 for name, _, _ in BANDS}
    exits["OUTSIDE_100_20000"] = 0.0
    emitted_total = 0.0
    residual = float(state.sum())
    iterations = 0
    for iteration in range(5000):
        contribution = state[upper] * p_emit
        emitted_total += float(np.sum(contribution))
        assigned = np.zeros(len(contribution), dtype=bool)
        for name, lo, hi in BANDS:
            mask = band_mask(wavelength, lo, hi)
            exits[name] += float(np.sum(contribution[mask]))
            assigned |= mask
        exits["OUTSIDE_100_20000"] += float(np.sum(contribution[~assigned]))
        next_state = np.zeros_like(state)
        np.add.at(next_state, lower, state[upper] * p_idown)
        np.add.at(next_state, upper, state[lower] * p_iup)
        residual = float(np.sum(next_state))
        iterations = iteration + 1
        if residual < 2.0e-13:
            break
        state = next_state
    if not math.isclose(emitted_total + residual, 1.0, rel_tol=0.0, abs_tol=5.0e-12):
        raise E13Error(f"Fe ion={ion} cascade closure failed")
    exit_fraction = {key: value / emitted_total for key, value in exits.items()}
    uv_fraction = sum(exit_fraction[name] for name in ("B0", "B1", "B2", "B3", "B4"))
    return {
        "species": "Fe II" if ion == 1 else "Fe III",
        "ion_number_zero_based": ion,
        "n_levels": int(len(energy_ev)),
        "n_lines": int(len(line_nu)),
        "neutral_ground_offset_eV": offset_ev,
        "field_shell": None,
        "entry_weight": "B_lu*J_producer*h*nu; lower populations omitted",
        "radiative_weights": {
            "emission": "(A_ul+B_ul*J)*h*nu",
            "internal_down": "(A_ul+B_ul*J)*E_lower_neutral",
            "internal_up": "B_lu*J*E_lower_neutral",
        },
        "first_action_probability": first,
        "iterations": iterations,
        "residual_probability": residual,
        "emitted_probability": emitted_total,
        "exit_fraction": exit_fraction,
        "UV_600_3000_exit_fraction": uv_fraction,
        "B0_600_1000_exit_fraction": exit_fraction["B0"],
        "excluded_physics": [
            "Sobolev beta", "collisions", "k-packet", "bound-free ion changes",
            "lower-level population weighting", "production probability damping"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--chieta", type=Path, required=True)
    parser.add_argument("--e9-payload", type=Path, required=True)
    parser.add_argument("--source-payload", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--atomic-dir", type=Path,
                        default=ROOT / "data/tardis_reference_toy06_19p48d")
    parser.add_argument("--shell", type=int, default=8)
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "validation/emiss_e13")
    add_matrix_contract_args(parser)   # [N4] matrix generation contract
    args = parser.parse_args()
    try:
        matrix = read_fluor_matrix_from_args(args.matrix, args)
        chieta = check_artifact(args.chieta.resolve())
        with args.preregistration.open() as stream:
            prereg = json.load(stream)
        nbins = int(matrix.header["n_bins"])
        nu_edges = matrix.header["nu_min"] * np.exp(
            np.arange(nbins + 1) * matrix.header["d_log_nu"])
        nu = np.sqrt(nu_edges[:-1] * nu_edges[1:])
        wavelength = bench.C_ANGSTROM / nu
        args.out_dir.mkdir(parents=True, exist_ok=True)
        mirrored_path = args.out_dir / "fluor_matrix_iter10_mirror_both_axes"
        mirrored_sha = write_mirrored_matrix(matrix, mirrored_path)
        contract = dict(prereg)
        contract["formal_matrix_sha256"] = mirrored_sha
        contract["e13_posthoc_counterfactual"] = True
        contract["e13_transform"] = "input_bin'=N-1-input_bin; output_bin'=N-1-output_bin"
        contract_path = args.out_dir / "mirror_application_contract.json"
        contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
        nr, nnu = int(chieta.header[3]), int(chieta.header[4])
        if not 0 <= args.shell < nr or nnu != nbins:
            raise E13Error("chieta shell/grid mismatch")
        field_nu = np.asarray(chieta.arrays[1])[::-1]
        field_j = np.asarray(chieta.arrays[8]).reshape(nr, nnu)[args.shell, ::-1]
        theory = [theoretical_cascade(args.atomic_dir.resolve(), ion, field_nu, field_j)
                  for ion in (1, 2)]
        for row in theory:
            row["field_shell"] = args.shell
        summary = {
            "schema": "lumina-emiss-e13-index-branch-audit-v1",
            "inputs": {
                "matrix": str(args.matrix.resolve()),
                "matrix_sha256": matrix.sha256,
                "chieta": str(args.chieta.resolve()),
                "chieta_sha256": chieta.manifest["sha256"],
                "e9_payload": str(args.e9_payload.resolve()),
                "source_payload": str(args.source_payload.resolve()),
                "atomic_dir": str(args.atomic_dir.resolve()),
            },
            "grid": {
                "n_bins": nbins,
                "nu_ascending": True,
                "wavelength_descending": True,
                "nu_center_first_Hz": float(nu[0]),
                "nu_center_last_Hz": float(nu[-1]),
                "lambda_center_first_A": float(wavelength[0]),
                "lambda_center_last_A": float(wavelength[-1]),
            },
            "native": {
                "frequency": frequency_metrics(matrix, False, nu, wavelength),
                "source_flow": source_flow(
                    matrix, False, args.e9_payload, args.source_payload,
                    prereg, nu_edges, wavelength, args.shell),
            },
            "mirror_both_axes": {
                "matrix": str(mirrored_path.resolve()),
                "matrix_sha256": mirrored_sha,
                "application_contract": str(contract_path.resolve()),
                "frequency": frequency_metrics(matrix, True, nu, wavelength),
                "source_flow": source_flow(
                    matrix, True, args.e9_payload, args.source_payload,
                    prereg, nu_edges, wavelength, args.shell),
            },
            "macro_atom_radiative_only_theory": theory,
            "kpacket_separation": {
                "matrix_has_edge_level_kpacket_tag": False,
                "kpacket_absorbed_fraction": (
                    matrix.header["kpacket_absorbed_energy"] /
                    matrix.header["absorbed_energy"]),
                "exact_non_kpacket_frequency_moment": "UNRESOLVED",
            },
            "production_code_modified": False,
            "new_model_or_GPU_run": False,
            "clamp_or_floor_added": False,
        }
        out = args.out_dir / "index_branch_audit.json"
        out.write_text(json.dumps(summary, indent=2, sort_keys=True,
                                  allow_nan=False) + "\n")
        print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (E13Error, OSError, ValueError, KeyError) as exc:
        print(f"UNRESOLVED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
