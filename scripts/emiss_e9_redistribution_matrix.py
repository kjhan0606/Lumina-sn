#!/usr/bin/env python3
"""Measure a sparse MC bin-to-bin redistribution matrix from archived events.

The archive is read-only and may be a capped prefix.  Packet energy, rather
than photon count times frequency, is the conserved quantity checked here.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import stage31_cmf_field_bench as bench  # noqa: E402

DEFAULT_RUN = Path("/gpfs/kjhan/lumina_runner2/scratch/emiss_ab2_capture_188766")

EVENT_DTYPE = np.dtype([
    ("pkt", "<u4"), ("line", "<i4"), ("nu", "<f4"), ("energy", "<f4"),
    ("etype", "u1"), ("shell", "u1"), ("iteration", "u1"), ("chan", "u1"),
])
LINE_DTYPE = np.dtype([("lambda_A", "<f4"), ("Z", "<u2"), ("ion", "<u2")])


class MatrixError(RuntimeError):
    pass


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise MatrixError(f"refusing empty output {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "validation/emiss_e9")
    parser.add_argument("--shell", type=int, default=8)
    args = parser.parse_args()
    try:
        from numba import njit
        run = args.run.resolve()
        events_path = run / "lumina_events.bin"
        lines_path = run / "lumina_events_lines.bin"
        events = np.memmap(events_path, dtype=EVENT_DTYPE, mode="r", offset=32)
        lines = np.memmap(lines_path, dtype=LINE_DTYPE, mode="r", offset=8)
        if events.size == 0 or lines.size == 0:
            raise MatrixError("empty event/line archive")
        edges, _, _ = bench.canonical_grid()
        line_nu = bench.C_ANGSTROM / np.asarray(lines["lambda_A"], dtype=np.float64)
        line_bin = np.searchsorted(edges, line_nu, side="right") - 1
        line_bin[(line_bin < 0) | (line_bin >= edges.size - 1)] = -1
        line_bin = np.asarray(line_bin, dtype=np.int32)
        max_packet = int(events["pkt"].max())

        @njit
        def consume(pkt, line, nu, energy, etype, shell, chan,
                    line_bins, line_wavelengths, freq_edges, npackets,
                    target_shell):
            last_line = np.full(npackets, -1, np.int32)
            last_bin = np.full(npackets, -1, np.int32)
            last_energy = np.zeros(npackets, np.float32)
            count = np.zeros((1000, 1000), np.int64)
            energy_out = np.zeros((1000, 1000), np.float64)
            input_count = np.zeros(1000, np.int64)
            paired_count = np.zeros(1000, np.int64)
            input_energy = np.zeros(1000, np.float64)
            paired_input_energy = np.zeros(1000, np.float64)
            terminal_energy = np.zeros(1000, np.float64)
            outside_count = np.zeros(1000, np.int64)
            outside_energy = np.zeros(1000, np.float64)
            channel_count = np.zeros(256, np.int64)
            channel_energy = np.zeros(256, np.float64)
            for i in range(pkt.size):
                p = pkt[i]
                lid = line[i]
                if etype[i] == 1:
                    if (shell[i] == target_shell and lid >= 0 and
                            line_wavelengths[lid] >= 600.0 and
                            line_wavelengths[lid] <= 3000.0):
                        ib = line_bins[lid]
                        if ib >= 0:
                            last_line[p] = lid
                            last_bin[p] = ib
                            last_energy[p] = energy[i]
                            input_count[ib] += 1
                            input_energy[ib] += energy[i]
                    else:
                        last_line[p] = -1
                        last_bin[p] = -1
                    continue
                ib = last_bin[p]
                if ib < 0:
                    continue
                line_terminal = (chan[i] == 0x38 or chan[i] == 0x12 or
                                 chan[i] == 0x16 or chan[i] == 0x15)
                continuum_terminal = (chan[i] == 0x10 or chan[i] == 0x11 or
                                      chan[i] == 0x14 or chan[i] == 0x24 or
                                      chan[i] == 0x3A or chan[i] == 0x51)
                if not (line_terminal or continuum_terminal):
                    continue
                ob = -1
                if line_terminal and lid >= 0:
                    ob = line_bins[lid]
                elif nu[i] > 0.0:
                    ob = np.searchsorted(freq_edges, float(nu[i]), side="right") - 1
                    if ob < 0 or ob >= 1000:
                        ob = -1
                paired_count[ib] += 1
                paired_input_energy[ib] += float(last_energy[p])
                terminal_energy[ib] += float(energy[i])
                channel_count[chan[i]] += 1
                channel_energy[chan[i]] += float(energy[i])
                if ob >= 0:
                    count[ib, ob] += 1
                    energy_out[ib, ob] += float(energy[i])
                else:
                    outside_count[ib] += 1
                    outside_energy[ib] += float(energy[i])
                last_line[p] = -1
                last_bin[p] = -1
            return (count, energy_out, input_count, paired_count, input_energy,
                    paired_input_energy, terminal_energy, outside_count,
                    outside_energy, channel_count, channel_energy)

        result = consume(events["pkt"], events["line"], events["nu"],
                         events["energy"], events["etype"], events["shell"],
                         events["chan"], line_bin, lines["lambda_A"], edges,
                         max_packet + 1,
                         args.shell)
        (count, energy_out, input_count, paired_count, input_energy,
         paired_input_energy, terminal_energy, outside_count, outside_energy,
         channel_count, channel_energy) = result

        uv_in = paired_count > 0
        active_in = np.flatnonzero(uv_in & (paired_count > 0))
        sparse_rows: list[dict[str, Any]] = []
        for ib in active_in:
            denom_n = int(paired_count[ib])
            denom_e = float(terminal_energy[ib])
            for ob in np.flatnonzero(count[ib] > 0):
                sparse_rows.append({
                    "shell": args.shell, "input_bin": int(ib),
                    "output_bin": int(ob), "count": int(count[ib, ob]),
                    "count_probability": float(count[ib, ob] / denom_n),
                    "output_energy": float(energy_out[ib, ob]),
                    "energy_probability": (float(energy_out[ib, ob] / denom_e)
                                           if denom_e > 0.0 else None),
                })
        args.out_dir.mkdir(parents=True, exist_ok=True)
        write_csv(args.out_dir / "redistribution_matrix_s8_sparse.csv", sparse_rows)

        input_rows: list[dict[str, Any]] = []
        for ib in active_in:
            output_n = int(np.sum(count[ib])) + int(outside_count[ib])
            output_e = float(np.sum(energy_out[ib])) + float(outside_energy[ib])
            input_rows.append({
                "shell": args.shell, "input_bin": int(ib),
                "input_events_seen": int(input_count[ib]),
                "paired_terminals": int(paired_count[ib]),
                "unpaired_prefix_tail": int(input_count[ib] - paired_count[ib]),
                "matrix_plus_outside_count": output_n,
                "paired_input_energy": float(paired_input_energy[ib]),
                "terminal_output_energy": output_e,
                "energy_closure_output_over_input": (
                    output_e / paired_input_energy[ib]
                    if paired_input_energy[ib] > 0.0 else None),
                "outside_grid_count": int(outside_count[ib]),
                "outside_grid_energy": float(outside_energy[ib]),
            })
        write_csv(args.out_dir / "redistribution_input_normalization_s8.csv", input_rows)

        # Energy destination fractions for UV activations in the target shell.
        selected_energy = np.sum(energy_out[uv_in], axis=0)
        selected_count = np.sum(count[uv_in], axis=0)
        total_terminal_e = float(np.sum(terminal_energy[uv_in]))
        total_terminal_n = int(np.sum(paired_count[uv_in]))
        categories = [
            ("EUV_100_600", 100.0, 600.0),
            ("UV_600_3000", 600.0, 3000.0),
            ("optical_3000_10000", 3000.0, 10000.0),
            ("IR_10000_20000", 10000.0, 20000.0),
        ]
        center_nu = 0.5 * (edges[:-1] + edges[1:])
        center_lambda = bench.C_ANGSTROM / center_nu
        destination = {}
        for name, lo, hi in categories:
            mask = ((center_lambda >= lo) &
                    (center_lambda < hi if hi < 20000.0 else center_lambda <= hi))
            destination[name] = {
                "count": int(np.sum(selected_count[mask])),
                "count_fraction_of_paired": float(
                    np.sum(selected_count[mask]) / total_terminal_n),
                "energy": float(np.sum(selected_energy[mask])),
                "energy_fraction_of_terminal": float(
                    np.sum(selected_energy[mask]) / total_terminal_e),
            }
        out_n = int(np.sum(outside_count[uv_in]))
        out_e = float(np.sum(outside_energy[uv_in]))
        destination["outside_100_20000_or_unmapped"] = {
            "count": out_n, "count_fraction_of_paired": out_n / total_terminal_n,
            "energy": out_e, "energy_fraction_of_terminal": out_e / total_terminal_e,
        }

        paired_in_e = float(np.sum(paired_input_energy[uv_in]))
        terminal_out_e = float(np.sum(terminal_energy[uv_in]))
        log = (run / "stdout.log").read_text(errors="replace")
        match = re.search(r"\[EVENT-LOG\] it(\d+): (\d+) events \((\d+) dropped\)", log)
        event_meta: dict[str, Any] = {
            "stored_records": int(events.size),
            "line_records": int(lines.size),
            "status": "TRUNCATED_PREFIX-not-an-unbiased-random-sample",
        }
        if match:
            attempted = int(match.group(2))
            event_meta.update({
                "iteration": int(match.group(1)), "attempted_records": attempted,
                "dropped_records": int(match.group(3)),
                "stored_fraction": float(events.size / attempted),
            })
        channels = {f"0x{i:02X}": {
            "count": int(channel_count[i]), "energy": float(channel_energy[i])}
            for i in range(256) if channel_count[i] > 0}
        summary = {
            "schema": "lumina-emiss-e9-redistribution-v1",
            "shell": args.shell, "input_wavelength_A": [600.0, 3000.0],
            "matrix_shape": [1000, 1000],
            "sparse_nonzero_edges": len(sparse_rows),
            "active_input_bins": len(active_in),
            "paired_terminals": total_terminal_n,
            "paired_input_energy": paired_in_e,
            "terminal_output_energy": terminal_out_e,
            "energy_conservation_output_over_input": terminal_out_e / paired_in_e,
            "energy_conservation_relative_error": terminal_out_e / paired_in_e - 1.0,
            "destination": destination,
            "terminal_channels": channels,
            "event_archive": event_meta,
            "normalization": "column/input-bin stochastic; count and packet-energy probabilities both retained",
            "update_cadence_observed": "one archived matrix from final iteration 11 only",
            "iteration10_matrix": "UNRESOLVED-no iteration-10 raw event archive",
            "full_iteration11_matrix": "UNRESOLVED-event cap retained a non-random prefix",
        }
        (args.out_dir / "redistribution_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
        print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (ImportError, MatrixError, OSError, ValueError, KeyError) as exc:
        print(f"UNRESOLVED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
