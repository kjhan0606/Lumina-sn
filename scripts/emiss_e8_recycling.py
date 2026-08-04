#!/usr/bin/env python3
"""E8 arithmetic-only audit of deterministic recycling and MC branching.

Consumes the frozen emissivity payload, CMFGEN jnu4/RVTJ, and archived MC
censuses/event records.  It never invokes a Lumina transport or plasma solve.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from emiss_ab_insitu_e4 import validate_lanes  # noqa: E402
from emiss_e6_direct_fields import (  # noqa: E402
    cmfgen_all_shells, weighted_integral, weighted_mean,
)
from emiss_e7_arithmetic import planck  # noqa: E402
import stage31_cmf_field_bench as bench  # noqa: E402
import w3_gamma_triple_compare as gamma  # noqa: E402


DEFAULT_RUN = Path("/gpfs/kjhan/lumina_runner2/scratch/emiss_ab2_capture_188766")
DEFAULT_CMF = Path("/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4")
SHELLS = (0, 8, 20)


class E8Error(RuntimeError):
    pass


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise E8Error(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ratio(a: float, b: float) -> float | None:
    return a / b if math.isfinite(a) and math.isfinite(b) and b != 0.0 else None


def rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks, sufficient for dependency-free Spearman coefficients."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def correlations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    # Exclude BALL because it duplicates the same bins already represented by B0--B4.
    use = [r for r in rows if r["band"] != "BALL" and r["J_ours_over_CMFGEN"] is not None]
    eps = np.asarray([r["eps_eff_source"] for r in use])
    gain = np.asarray([r["recycle_gain_source"] for r in use])
    jratio = np.asarray([r["J_ours_over_CMFGEN"] for r in use])
    chif = np.asarray([r["chi_coherent_over_total"] for r in use])

    def corr(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        return {
            "pearson": float(np.corrcoef(x, y)[0, 1]),
            "spearman": float(np.corrcoef(rankdata(x), rankdata(y))[0, 1]),
        }

    return {
        "sample_count_B0_to_B4_resolved_CMF": len(use),
        "log10_eps_vs_log10_recycle_gain": corr(np.log10(eps), np.log10(gain)),
        "log10_eps_vs_log10_J_ours_over_CMFGEN": corr(np.log10(eps), np.log10(jratio)),
        "chi_coherent_fraction_vs_log10_recycle_gain": corr(chif, np.log10(gain)),
        "identity_note": "recycle_gain_source=1/eps_eff_source by definition; its anti-correlation is algebraic",
    }


def cmf_temperature(cmf_run: Path, velocities: np.ndarray) -> np.ndarray:
    _, _, nd, _, _ = gamma.read_eddfactor(cmf_run / "EDDFACTOR")
    text = (cmf_run / "RVTJ").read_text()
    v = gamma.rvtj_block(text, "Velocity (km/s)", nd)
    t = 1.0e4 * gamma.rvtj_block(text, "Temperature (10^4K)", nd)
    order = np.argsort(v)
    result = np.full(velocities.shape, np.nan)
    inside = (velocities > v[order][0]) & (velocities < v[order][-1])
    result[inside] = np.interp(velocities[inside], v[order], t[order])
    return result


def read_ma_destruction(path: Path, iteration: int) -> list[dict[str, Any]]:
    rows = []
    with path.open() as stream:
        for row in csv.DictReader(stream):
            if int(row["iter"]) != iteration:
                continue
            terminal = int(row["terminals"])
            destroyed = int(row["destroyed"])
            rows.append({
                "iteration": iteration, "shell": int(row["shell"]),
                "terminals": terminal, "destroyed": destroyed,
                "thermal_destruction_fraction": ratio(destroyed, terminal),
                "heating_erg_s_cm3": float(row["heating_erg_s_cm3"]),
            })
    if len(rows) != 50:
        raise E8Error(f"expected 50 ma-line-destruction rows for iteration {iteration}")
    return rows


def census_identity(run: Path, ma_final: list[dict[str, Any]]) -> dict[str, Any]:
    census = {}
    with (run / "lumina_census_ma_fate.csv").open() as stream:
        for row in csv.DictReader(stream):
            census[int(row["shell"])] = {key: int(row[key]) for key in (
                "rad_deexc", "col_deexc", "rad_recomb", "total")}
    if sorted(census) != list(range(50)):
        raise E8Error("macro-atom fate census is not a complete 50-shell table")
    mismatches = []
    for row in ma_final:
        shell = row["shell"]
        # A terminal selected for epsilon destruction is not emitted at that
        # selection.  The final event census must therefore contain exactly the
        # complementary terminal count as radiative de-excitations.
        expected = row["terminals"] - row["destroyed"]
        if census[shell]["rad_deexc"] != expected:
            mismatches.append(shell)
    if mismatches:
        raise E8Error(f"terminal-destruction/census identity fails in shells {mismatches}")
    return {
        "shells_checked": 50, "mismatch_count": 0,
        "identity": "census.rad_deexc == ma_line_destruct.terminals - destroyed",
        "global_terminals": sum(r["terminals"] for r in ma_final),
        "global_destroyed": sum(r["destroyed"] for r in ma_final),
        "global_rad_deexc": sum(census[s]["rad_deexc"] for s in census),
        "global_col_deexc": sum(census[s]["col_deexc"] for s in census),
        "global_rad_recomb": sum(census[s]["rad_recomb"] for s in census),
    }


def event_branch_rows(run: Path, frequency_edges: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pair bb activation with its next terminal event, preserving per-packet order.

    CUDA atomics interleave packets globally, but calls made by one packet thread keep
    program order.  A per-packet last-activation table therefore pairs records without
    sorting the 400M-record archive.
    """
    try:
        from numba import njit
    except ImportError as exc:
        raise E8Error("numba is required to stream the 7.5-GB event archive") from exc

    event_dtype = np.dtype([
        ("pkt", "<u4"), ("line", "<i4"), ("nu", "<f4"), ("energy", "<f4"),
        ("etype", "u1"), ("shell", "u1"), ("iteration", "u1"), ("chan", "u1"),
    ])
    line_dtype = np.dtype([("lambda_A", "<f4"), ("Z", "<u2"), ("ion", "<u2")])
    events_path = run / "lumina_events.bin"
    lines_path = run / "lumina_events_lines.bin"
    events = np.memmap(events_path, dtype=event_dtype, mode="r", offset=32)
    lines = np.memmap(lines_path, dtype=line_dtype, mode="r", offset=8)
    line_frequency = bench.C_ANGSTROM / np.asarray(lines["lambda_A"], dtype=np.float64)
    line_bins = np.searchsorted(frequency_edges, line_frequency, side="right") - 1
    line_bins[(line_bins < 0) | (line_bins >= frequency_edges.size - 1)] = -1
    line_bins = np.asarray(line_bins, dtype=np.int32)
    max_packet = int(events["pkt"].max())

    @njit
    def consume(pkt, line, etype, shell, chan, wavelengths, coarse_bins, npackets):
        last_line = np.full(npackets, -1, np.int32)
        last_shell = np.zeros(npackets, np.uint8)
        # paired, resonant, different-line, redward, blueward, same-band,
        # emitted-outside-UV, continuum/thermal terminal, same coarse bin,
        # different coarse bin
        counts = np.zeros((6, 50, 10), np.int64)
        for i in range(pkt.size):
            p = pkt[i]
            lid = line[i]
            if etype[i] == 1:  # EVCH_MA_ACT_BB
                last_line[p] = lid
                last_shell[p] = shell[i]
                continue
            absorbed = last_line[p]
            if absorbed < 0:
                continue
            line_terminal = (chan[i] == 0x38 or chan[i] == 0x12 or
                             chan[i] == 0x16 or chan[i] == 0x15)
            cont_terminal = (chan[i] == 0x10 or chan[i] == 0x11 or
                             chan[i] == 0x14 or chan[i] == 0x24 or
                             chan[i] == 0x3A or chan[i] == 0x51)
            if not (line_terminal or cont_terminal):
                continue
            absorbed_lambda = wavelengths[absorbed]
            band = -1
            if 600.0 <= absorbed_lambda < 1000.0:
                band = 0
            elif 1000.0 <= absorbed_lambda < 1500.0:
                band = 1
            elif 1500.0 <= absorbed_lambda < 2000.0:
                band = 2
            elif 2000.0 <= absorbed_lambda < 2500.0:
                band = 3
            elif 2500.0 <= absorbed_lambda <= 3000.0:
                band = 4
            if band >= 0:
                s = last_shell[p]
                for outband in (band, 5):
                    counts[outband, s, 0] += 1
                    if line_terminal and lid >= 0:
                        emitted_lambda = wavelengths[lid]
                        if lid == absorbed:
                            counts[outband, s, 1] += 1
                        else:
                            counts[outband, s, 2] += 1
                        if emitted_lambda > absorbed_lambda:
                            counts[outband, s, 3] += 1
                        elif emitted_lambda < absorbed_lambda:
                            counts[outband, s, 4] += 1
                        emitted_band = -1
                        if 600.0 <= emitted_lambda < 1000.0:
                            emitted_band = 0
                        elif 1000.0 <= emitted_lambda < 1500.0:
                            emitted_band = 1
                        elif 1500.0 <= emitted_lambda < 2000.0:
                            emitted_band = 2
                        elif 2000.0 <= emitted_lambda < 2500.0:
                            emitted_band = 3
                        elif 2500.0 <= emitted_lambda <= 3000.0:
                            emitted_band = 4
                        if emitted_band == band:
                            counts[outband, s, 5] += 1
                        if emitted_band < 0:
                            counts[outband, s, 6] += 1
                        if coarse_bins[lid] == coarse_bins[absorbed]:
                            counts[outband, s, 8] += 1
                        else:
                            counts[outband, s, 9] += 1
                    else:
                        counts[outband, s, 7] += 1
            last_line[p] = -1
        return counts

    counts = consume(events["pkt"], events["line"], events["etype"],
                     events["shell"], events["chan"], lines["lambda_A"], line_bins,
                     max_packet + 1)
    names = [b[0] for b in bench.BANDS]
    rows = []
    for band_index, band in enumerate(names):
        for shell in range(50):
            c = counts[band_index, shell]
            paired = int(c[0])
            rows.append({
                "event_iteration": int(events["iteration"][0]) if events.size else None,
                "band": band, "shell": shell, "paired": paired,
                "resonant_same_line": int(c[1]), "different_line": int(c[2]),
                "redward": int(c[3]), "blueward": int(c[4]),
                "same_absorption_band": int(c[5]), "emitted_outside_600_3000": int(c[6]),
                "continuum_or_thermal_terminal": int(c[7]),
                "same_coarse_bin": int(c[8]), "different_coarse_bin": int(c[9]),
                "resonant_same_line_fraction": ratio(int(c[1]), paired),
                "different_line_fraction": ratio(int(c[2]), paired),
                "local_coherence_destruction_fraction": ratio(int(c[2] + c[7]), paired),
                "same_coarse_bin_fraction": ratio(int(c[8]), paired),
                "coarse_bin_coherence_destruction_fraction": ratio(int(c[9] + c[7]), paired),
                "same_absorption_band_fraction": ratio(int(c[5]), paired),
                "emitted_outside_600_3000_fraction": ratio(int(c[6]), paired),
            })

    log = (run / "stdout.log").read_text(errors="replace")
    match = re.search(r"\[EVENT-LOG\] it(\d+): (\d+) events \((\d+) dropped\)", log)
    meta = {
        "event_file": str(events_path), "event_records_stored": int(events.size),
        "line_records": int(lines.size), "max_packet_id": max_packet,
        "pairing_contract": "per-packet file order: bb activation -> next terminal channel",
    }
    if match:
        attempted = int(match.group(2))
        meta.update({"event_iteration": int(match.group(1)), "event_records_attempted": attempted,
                     "event_records_dropped": int(match.group(3)),
                     "stored_fraction": events.size / attempted,
                     "status": "TRUNCATED_PREFIX-not-an-unbiased-random-sample"})
    return rows, meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--cmf-run", type=Path, default=DEFAULT_CMF)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "validation/emiss_e8")
    args = parser.parse_args()
    try:
        run = args.run.resolve()
        a, b, b2 = validate_lanes(run / "emiss_ab_iter10", allow_seeded=False)
        nr, nnu = a.header[3], a.header[4]
        fields = {
            "chi_total": np.asarray(a.arrays[3]).reshape(nr, nnu)[:, ::-1],
            "chi_coherent": np.asarray(a.arrays[4]).reshape(nr, nnu)[:, ::-1],
            "eta_fixed": np.asarray(a.arrays[5]).reshape(nr, nnu)[:, ::-1],
            "eta_coherent": np.asarray(a.arrays[6]).reshape(nr, nnu)[:, ::-1],
            "eta_total": np.asarray(a.arrays[7]).reshape(nr, nnu)[:, ::-1],
            "J": np.asarray(a.arrays[8]).reshape(nr, nnu)[:, ::-1],
        }
        if not np.array_equal(fields["eta_fixed"] + fields["eta_coherent"], fields["eta_total"]):
            raise E8Error("payload eta decomposition is not bitwise exact")
        if np.any(fields["chi_total"] <= 0.0):
            raise E8Error("nonpositive chi_total prevents source decomposition")
        if np.any(fields["eta_fixed"] < 0.0) or np.any(fields["eta_coherent"] < 0.0):
            raise E8Error("negative emissivity encountered; no clamp is permitted")

        edges, centers, _ = bench.canonical_grid()
        r_edge = np.asarray(a.arrays[0])
        widths_r = np.diff(r_edge)
        velocities = 0.5 * (r_edge[:-1] + r_edge[1:]) / a.header[-1] / 1.0e5
        cmf_j, cmf_meta = cmfgen_all_shells(edges, velocities, args.cmf_run.resolve())
        cmf_te = cmf_temperature(args.cmf_run.resolve(), velocities)
        tau_out = np.cumsum((fields["chi_total"] * widths_r[:, None])[::-1], axis=0)[::-1]

        band_rows = []
        for band, lo, hi in bench.BANDS:
            weights = bench.band_weights(edges, lo, hi)
            for shell in range(nr):
                chi_i = weighted_integral(fields["chi_total"][shell], weights)
                chic_i = weighted_integral(fields["chi_coherent"][shell], weights)
                sf_i = weighted_integral(fields["eta_fixed"][shell] / fields["chi_total"][shell], weights)
                sc_i = weighted_integral(fields["eta_coherent"][shell] / fields["chi_total"][shell], weights)
                st_i = sf_i + sc_i
                ef_i = weighted_integral(fields["eta_fixed"][shell], weights)
                et_i = weighted_integral(fields["eta_total"][shell], weights)
                j_ours = weighted_mean(fields["J"][shell], weights)
                j_cmf = (weighted_mean(cmf_j[shell], weights)
                         if np.isfinite(cmf_j[shell]).all() else None)
                b_cmf = (weighted_mean(np.asarray([planck(nu, cmf_te[shell]) for nu in centers]), weights)
                         if math.isfinite(cmf_te[shell]) else None)
                eps_source = sf_i / st_i
                band_rows.append({
                    "band": band, "lambda_lo_A": lo, "lambda_hi_A": hi,
                    "shell": shell, "velocity_kms": velocities[shell],
                    "chi_coherent_over_total": chic_i / chi_i,
                    "chi_noncoherent_over_total": 1.0 - chic_i / chi_i,
                    "eps_eff_source": eps_source,
                    "recycle_gain_source": 1.0 / eps_source,
                    "coherent_source_fraction": sc_i / st_i,
                    "eps_eff_literal_eta_integral": ef_i / et_i,
                    "J_ours": j_ours, "J_CMFGEN": j_cmf,
                    "J_ours_over_CMFGEN": ratio(j_ours, j_cmf) if j_cmf is not None else None,
                    "S_fixed_over_CMFGEN": ratio(sf_i / math.fsum(weights), j_cmf) if j_cmf is not None else None,
                    "S_total_over_CMFGEN": ratio(st_i / math.fsum(weights), j_cmf) if j_cmf is not None else None,
                    "CMFGEN_T_K": cmf_te[shell] if math.isfinite(cmf_te[shell]) else None,
                    "CMFGEN_J_over_B_T": ratio(j_cmf, b_cmf) if j_cmf is not None and b_cmf is not None else None,
                    "tau_out_ge1_fraction": weighted_mean((tau_out[shell] >= 1.0).astype(float), weights),
                })

        macro_rows = read_ma_destruction(run / "lumina_ma_line_destruct.csv", int(a.manifest["iteration"]))
        macro_final_rows = read_ma_destruction(run / "lumina_ma_line_destruct.csv", 11)
        macro_census_identity = census_identity(run, macro_final_rows)
        event_rows, event_meta = event_branch_rows(run, edges)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        write_csv(args.out_dir / "band_shell_recycling.csv", band_rows)
        write_csv(args.out_dir / "macro_thermal_destruction_iter10.csv", macro_rows)
        write_csv(args.out_dir / "event_fluorescence_branch_iter11.csv", event_rows)

        def brow(band: str, shell: int) -> dict[str, Any]:
            return next(r for r in band_rows if r["band"] == band and r["shell"] == shell)

        def erow(band: str, shell: int) -> dict[str, Any]:
            return next(r for r in event_rows if r["band"] == band and r["shell"] == shell)

        def mrow(shell: int) -> dict[str, Any]:
            return next(r for r in macro_rows if r["shell"] == shell)

        s8 = brow("BALL", 8)
        e8 = erow("BALL", 8)
        required_gain = s8["J_ours_over_CMFGEN"] / s8["S_fixed_over_CMFGEN"]
        summary = {
            "schema": "lumina-emiss-e8-recycling-v1",
            "arithmetic_only_no_transport_solve": True,
            "no_new_clamp": True,
            "payload_sha256": a.manifest["sha256"],
            "payload_iteration": int(a.manifest["iteration"]),
            "cmfgen": cmf_meta,
            "definitions": {
                "cell_eps_eff": "eta_fixed/(eta_fixed+eta_coherent)",
                "band_eps_eff_source": "integral[(eta_fixed/chi_total)dnu]/integral[(eta_total/chi_total)dnu]",
                "literal_eta_integral_check": "integral eta_fixed dnu/integral eta_total dnu",
                "recycle_gain_source": "1/eps_eff_source=S_total_band/S_fixed_band",
            },
            "canonical_shells": {str(shell): {band: {
                key: brow(band, shell)[key] for key in (
                    "chi_coherent_over_total", "eps_eff_source", "recycle_gain_source",
                    "eps_eff_literal_eta_integral", "J_ours_over_CMFGEN", "CMFGEN_J_over_B_T")
            } for band, _, _ in bench.BANDS} for shell in SHELLS},
            "correlations": correlations(band_rows),
            "macro_comparison": {str(shell): {
                "eps_eff_source_BALL_payload_iter10": brow("BALL", shell)["eps_eff_source"],
                "MC_thermal_destruction_iter10": mrow(shell)["thermal_destruction_fraction"],
                "MC_over_deterministic_eps_eff": ratio(
                    mrow(shell)["thermal_destruction_fraction"], brow("BALL", shell)["eps_eff_source"]),
                "event_local_coherence_destruction_iter11_truncated": erow("BALL", shell)["local_coherence_destruction_fraction"],
                "event_coarse_bin_coherence_destruction_iter11_truncated": erow("BALL", shell)["coarse_bin_coherence_destruction_fraction"],
                "event_different_line_fraction_iter11_truncated": erow("BALL", shell)["different_line_fraction"],
            } for shell in SHELLS},
            "macro_final_census_identity": macro_census_identity,
            "event_log": event_meta,
            "s8_amplitude_closure": {
                "J_ours_over_CMFGEN": s8["J_ours_over_CMFGEN"],
                "S_fixed_over_CMFGEN": s8["S_fixed_over_CMFGEN"],
                "required_gain_J_over_Sfixed": required_gain,
                "measured_recycle_gain_Stotal_over_Sfixed": s8["recycle_gain_source"],
                "required_eps_inverse_gain": 1.0 / required_gain,
                "measured_eps_eff_source": s8["eps_eff_source"],
                "S_total_over_J_ours": s8["S_total_over_CMFGEN"] / s8["J_ours_over_CMFGEN"],
                "event_same_line_fraction_iter11_truncated": e8["resonant_same_line_fraction"],
                "event_local_coherence_destruction_iter11_truncated": e8["local_coherence_destruction_fraction"],
                "event_same_coarse_bin_fraction_iter11_truncated": e8["same_coarse_bin_fraction"],
                "event_coarse_bin_coherence_destruction_iter11_truncated": e8["coarse_bin_coherence_destruction_fraction"],
            },
            "CMFGEN_equivalent_eps": "UNRESOLVED-no frequency/depth ETA/CHI line-source dump; J/B and J~=S do not uniquely invert epsilon",
            "CMFGEN_eps_multiplier": "UNRESOLVED",
        }
        (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        print(f"PASS E8: {len(band_rows)} band-shell rows, {len(event_rows)} event-branch rows")
        print(f"s8 BALL eps={s8['eps_eff_source']:.9g} gain={s8['recycle_gain_source']:.6g} "
              f"J/CMF={s8['J_ours_over_CMFGEN']:.6g}")
        print(f"outputs: {args.out_dir.resolve()}")
        return 0
    except (E8Error, OSError, ValueError, KeyError) as exc:
        print(f"UNRESOLVED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
