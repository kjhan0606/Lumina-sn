#!/usr/bin/env python3
"""Chunked, parameterized copy of the historical event-forensics analysis.

This script performs a complete pass over lumina_events.bin and therefore MUST
be run on a compute node, not during the lightweight replay.  Definitions match
the historical taskB_event_forensics.py: EventRec schema, event-type sets, shell
groups, wavelength bands, packet-energy weighting, and s0-2 emission mean.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

EVENT_DTYPE = np.dtype([("pkt_id", "<u4"), ("line_id", "<i4"), ("nu", "<f4"),
                        ("energy", "<f4"), ("etype", "u1"), ("shell", "u1"),
                        ("iter", "u1"), ("pad", "u1")])
LINE_DTYPE = np.dtype([("lam", "<f4"), ("Z", "<u2"), ("ion", "<u2")])
C_A = 2.99792458e18
EMIT = (2, 4, 5)
ABSORB = (1, 3)
EDGES = np.asarray([100, 300, 450, 918, 1290, 2000, 3000, 4500, 7000, 10000, 19933, 1e12])
LABELS = ["soft_100_300", "EUV_300_450", "xuv_450_918", "FUV_918_1290",
          "NUV_1290_2000", "UV_2000_3000", "blue_3000_4500", "opt_4500_7000",
          "red_7000_10000", "NIR_10000_19933", "beyond_19933"]
GROUPS = {"s0-2": (0, 2, 3), "s7-8": (7, 8, 2)}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-events", type=int, default=5_000_000)
    return parser.parse_args()


def write(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(header); writer.writerows(rows)


def main() -> None:
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    event_path = args.input_dir / "lumina_events.bin"
    lines_path = args.input_dir / "lumina_events_lines.bin"
    for path in (event_path, lines_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    with event_path.open("rb") as handle:
        header = handle.read(32)
    if header[:8] != b"LUMEVT01":
        raise ValueError(f"{event_path}: bad event magic")
    with lines_path.open("rb") as handle:
        if handle.read(8) != b"LUMLIN01":
            raise ValueError(f"{lines_path}: bad line magic")
        line_records = np.frombuffer(handle.read(), dtype=LINE_DTYPE)
    line_z = line_records["Z"].astype(np.int32)
    line_ion = line_records["ion"].astype(np.int32)

    events = np.memmap(event_path, dtype=EVENT_DTYPE, mode="r", offset=32)
    emit_e = {group: np.zeros(len(LABELS)) for group in GROUPS}
    absorb_e = {group: np.zeros(len(LABELS)) for group in GROUPS}
    blue = {group: np.zeros(7) for group in GROUPS}  # n_emit,E_emit,n_abs,E_abs,n_kp,E_kp,net
    deep_emit_energy = 0.0
    deep_emit_lambda_energy = 0.0
    deep_emit_band = np.zeros(4)  # <1290,1290-2000,2000-4500,4500-20000
    ion_energy: dict[tuple[str, str, int, int], float] = defaultdict(float)
    iter_seen: set[int] = set()
    etype_hist = np.zeros(256, dtype=np.int64)

    for start in range(0, len(events), args.chunk_events):
        block = events[start : min(start + args.chunk_events, len(events))]
        et = np.asarray(block["etype"])
        sh = np.asarray(block["shell"])
        nu = np.asarray(block["nu"], dtype=np.float64)
        energy = np.asarray(block["energy"], dtype=np.float64)
        line_id = np.asarray(block["line_id"])
        np.add.at(etype_hist, et, 1)
        iter_seen.update(int(value) for value in np.unique(np.asarray(block["iter"])))
        positive_nu = nu > 0
        wavelength = np.empty(len(block), dtype=np.float64)
        wavelength[:] = np.nan
        wavelength[positive_nu] = C_A / nu[positive_nu]
        band = np.digitize(wavelength, EDGES) - 1
        is_emit = np.isin(et, EMIT)
        is_absorb = np.isin(et, ABSORB)
        for group, (lo_shell, hi_shell, _) in GROUPS.items():
            group_mask = (sh >= lo_shell) & (sh <= hi_shell)
            for b in range(len(LABELS)):
                in_band = group_mask & (band == b)
                emit_e[group][b] += energy[in_band & is_emit].sum()
                absorb_e[group][b] += energy[in_band & is_absorb].sum()
            blue_mask = group_mask & (wavelength < 1290.0)
            em = blue_mask & is_emit
            ab = blue_mask & is_absorb
            kp = blue_mask & np.isin(et, (4, 5))
            blue[group][0] += int(em.sum()); blue[group][1] += energy[em].sum()
            blue[group][2] += int(ab.sum()); blue[group][3] += energy[ab].sum()
            blue[group][4] += int(kp.sum()); blue[group][5] += energy[kp].sum()

        deep = (sh <= 2) & is_emit & (wavelength >= 100) & (wavelength <= 20000)
        deep_emit_energy += energy[deep].sum()
        deep_emit_lambda_energy += (wavelength[deep] * energy[deep]).sum()
        for i, (lo, hi) in enumerate(((100, 1290), (1290, 2000), (2000, 4500), (4500, 20000))):
            deep_emit_band[i] += energy[deep & (wavelength >= lo) & (wavelength < hi)].sum()

        # Historical top-ion roles, accumulated without retaining event arrays.
        roles = [("EMIT_NUVpile_1290_2000", (1290, 2000), EMIT),
                 ("EMIT_red_7000_19933", (7000, 19933), EMIT),
                 ("EMIT_optblue_3000_7000", (3000, 7000), EMIT),
                 ("ABS_FUVxuv_450_1290", (450, 1290), ABSORB),
                 ("EMIT_FUVxuv_450_1290", (450, 1290), EMIT)]
        for role, (lo, hi), types in roles:
            mask = (sh <= 2) & np.isin(et, types) & (wavelength >= lo) & (wavelength < hi) & (line_id >= 0)
            ids = line_id[mask]
            if ids.size == 0:
                continue
            if int(ids.max()) >= len(line_records):
                raise ValueError("event line_id exceeds lumina_events_lines table")
            keys = line_z[ids] * 100 + line_ion[ids]
            unique, inverse = np.unique(keys, return_inverse=True)
            sums = np.bincount(inverse, weights=energy[mask])
            for key, value in zip(unique, sums):
                ion_energy[("s0-2", role, int(key // 100), int(key % 100))] += float(value)

    ledger_rows = []
    for group in GROUPS:
        for b, label in enumerate(LABELS):
            ledger_rows.append([group, label, EDGES[b], EDGES[b + 1], emit_e[group][b],
                                absorb_e[group][b], emit_e[group][b] - absorb_e[group][b],
                                str(event_path), "energy; etype; shell; nu"])
    write(args.output_dir / "taskB_band_ledger.csv",
          ["group", "band", "lo_A", "hi_A", "emitE", "absE", "netE", "source_file", "source_fields"], ledger_rows)

    up_rows = []
    for group, (_, _, nshell) in GROUPS.items():
        values = blue[group]
        values[6] = values[1] - values[3]
        up_rows.append([group, nshell, *values, values[1] / nshell, values[6] / nshell,
                        str(event_path), "energy; etype; shell; nu"])
    write(args.output_dir / "taskB_upconversion.csv",
          ["group", "nshells", "n_emit_blue", "E_emit_blue", "n_abs_blue", "E_abs_blue",
           "n_kpkt_blue", "E_kpkt_blue", "net_blue", "E_emit_blue_per_shell", "net_blue_per_shell",
           "source_file", "source_fields"], up_rows)

    mean_lambda = deep_emit_lambda_energy / deep_emit_energy
    color_rows = [[mean_lambda, *list(deep_emit_band / deep_emit_energy), str(event_path),
                   "energy; etype in {2,4,5}; shell 0..2; nu"]]
    write(args.output_dir / "taskB_emission_color.csv",
          ["emission_weighted_mean_A", "frac_100_1290", "frac_1290_2000", "frac_2000_4500",
           "frac_4500_20000", "source_file", "source_fields"], color_rows)

    ion_rows = []
    for group, role in sorted(set((key[0], key[1]) for key in ion_energy)):
        items = [(z, ion, value) for (g, r, z, ion), value in ion_energy.items() if g == group and r == role]
        total = sum(item[2] for item in items)
        for z, ion, value in sorted(items, key=lambda item: item[2], reverse=True)[:12]:
            ion_rows.append([group, role, z, ion, value, value / total, str(event_path), "energy; line_id",
                             str(lines_path), "Z; ion"])
    write(args.output_dir / "taskB_top_ions.csv",
          ["group", "role", "Z", "ion_idx", "E", "frac_of_role", "event_source_file",
           "event_source_fields", "line_source_file", "line_source_fields"], ion_rows)
    coverage = [[len(events), list(np.nonzero(etype_hist)[0]), list(etype_hist[etype_hist > 0]), sorted(iter_seen),
                 str(event_path), "etype; iter"]]
    write(args.output_dir / "taskB_coverage.csv",
          ["n_events", "etype_values", "etype_counts", "iterations", "source_file", "source_fields"], coverage)


if __name__ == "__main__":
    main()
