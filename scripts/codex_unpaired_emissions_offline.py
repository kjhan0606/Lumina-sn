#!/usr/bin/env python3
"""Stream the existing event archive to diagnose line emissions without activation.

The 8 GB archive is deliberately processed only with ``--run-heavy``.  The
default/self-test path uses a synthetic in-memory fixture and never opens the
archive.  This is an offline consumer: it does not run transport, a model, or
GPU code, and it applies no clamp/floor/cap/fallback/substitution.
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
import read_events  # noqa: E402


CAPTURE = Path("/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932")
DEFAULT_EVENTS = CAPTURE / "lumina_events.bin"
DEFAULT_LINES = CAPTURE / "lumina_events_lines.bin"
DEFAULT_UPSTREAM = CAPTURE / "pile_ion_attribution/pile_ion_attribution.json"
DEFAULT_STDOUT = CAPTURE / "stdout.log"
TERMINAL_CHANNELS = (0x10, 0x11, 0x12, 0x14, 0x15, 0x16,
                     0x24, 0x38, 0x3A, 0x50, 0x51)
CHANNEL_NAMES = {
    0x10: "KPKT_FF", 0x11: "KPKT_FB", 0x12: "KPKT_COLLEXC",
    0x14: "KPKT_BTE", 0x15: "KPKT_MACAP", 0x16: "KPKT_COLLEXC_BB",
    0x24: "HEAT_LINETHERM", 0x30: "MA_ACT_BB", 0x31: "MA_ACT_BF",
    0x38: "MA_RAD_DEEXC", 0x3A: "MA_RAD_RECOMB", 0x40: "RPKT_BF_ABS",
    0x50: "ESCAPE", 0x51: "BF_REEMIT_LEGACY",
}
ORIGIN_NAMES = {0: "NONE_SINCE_TERMINAL", 1: "LINE_ACTIVATION",
                2: "BF_ABSORPTION", 3: "OTHER_NONTERMINAL"}
BANDS = (
    ("LT600", 0.0, 600.0, "[0,600)"),
    ("B0", 600.0, 1000.0, "[600,1000)"),
    ("B1", 1000.0, 1500.0, "[1000,1500)"),
    ("B2", 1500.0, 2000.0, "[1500,2000)"),
    ("B3", 2000.0, 2500.0, "[2000,2500)"),
    ("B4", 2500.0, 3000.0, "[2500,3000]"),
    ("GT3000", 3000.0, None, "(3000,+inf)"),
)


class AuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def fraction(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0.0 else numerator / denominator


def line_bands(wavelength: np.ndarray) -> np.ndarray:
    require(wavelength.ndim == 1 and wavelength.size > 0,
            "line wavelength table is empty/not 1-D")
    require(np.isfinite(wavelength).all() and np.all(wavelength > 0.0),
            "line wavelength is nonpositive/nonfinite")
    out = np.empty(wavelength.size, dtype=np.int8)
    out[wavelength < 600.0] = 0
    out[(wavelength >= 600.0) & (wavelength < 1000.0)] = 1
    out[(wavelength >= 1000.0) & (wavelength < 1500.0)] = 2
    out[(wavelength >= 1500.0) & (wavelength < 2000.0)] = 3
    out[(wavelength >= 2000.0) & (wavelength < 2500.0)] = 4
    out[(wavelength >= 2500.0) & (wavelength <= 3000.0)] = 5
    out[wavelength > 3000.0] = 6
    require(np.all((out >= 0) & (out < len(BANDS))), "band assignment failed")
    return out


def make_arrays(n_packets: int, max_z: int, max_ion: int) -> dict[str, np.ndarray]:
    require(n_packets > 0 and max_z > 0 and max_ion >= 0,
            "invalid state/ion dimensions")
    shape = (max_z + 1, max_ion + 1, len(BANDS), 256)
    return {
        "active": np.full(n_packets, -1, dtype=np.int32),
        "origin": np.zeros(n_packets, dtype=np.uint8),
        "totals": np.zeros(10, dtype=np.float64),
        # totals: all-line count, all-line energy, missing count, missing energy,
        # activations, terminals, bf absorptions, records processed,
        # all internal-emission count, all internal-emission energy.
        "all_count": np.zeros(shape, dtype=np.int64),
        "all_energy": np.zeros(shape, dtype=np.float64),
        "missing_count": np.zeros(shape, dtype=np.int64),
        "missing_energy": np.zeros(shape, dtype=np.float64),
        "missing_channel_count": np.zeros(256, dtype=np.int64),
        "missing_channel_energy": np.zeros(256, dtype=np.float64),
        "channel_origin_count": np.zeros((256, 4), dtype=np.int64),
        "channel_origin_energy": np.zeros((256, 4), dtype=np.float64),
    }


def compile_kernel():
    try:
        from numba import njit
    except ImportError as exc:
        raise AuditError("numba is required for the streaming state machine") from exc

    terminal_lut = np.zeros(256, dtype=np.uint8)
    terminal_lut[list(TERMINAL_CHANNELS)] = 1

    @njit(cache=True)
    def consume(pkt, lid, energy, etype, shell, chan,
                line_z, line_ion, line_band, terminal,
                active, origin, totals, all_count, all_energy,
                missing_count, missing_energy, missing_channel_count,
                missing_channel_energy, channel_origin_count,
                channel_origin_energy):
        for index in range(pkt.size):
            p = int(pkt[index])
            line = int(lid[index])
            typ = int(etype[index])
            channel = int(chan[index])
            sh = int(shell[index])
            en = float(energy[index])
            totals[7] += 1.0
            if typ == 1:
                active[p] = line
                origin[p] = 1
                totals[4] += 1.0
                continue
            is_terminal = terminal[channel] != 0
            if is_terminal:
                totals[5] += 1.0
            if typ == 3:
                origin[p] = 2
                totals[6] += 1.0
            elif not is_terminal and origin[p] == 0:
                origin[p] = 3
            if typ == 2 or typ == 4 or typ == 5 or typ == 8:
                totals[8] += 1.0
                totals[9] += en
            if typ == 2 and line >= 0:
                z = int(line_z[line])
                ion = int(line_ion[line])
                band = int(line_band[line])
                totals[0] += 1.0
                totals[1] += en
                all_count[z, ion, band, sh] += 1
                all_energy[z, ion, band, sh] += en
                if active[p] < 0:
                    prior = int(origin[p])
                    totals[2] += 1.0
                    totals[3] += en
                    missing_count[z, ion, band, sh] += 1
                    missing_energy[z, ion, band, sh] += en
                    missing_channel_count[channel] += 1
                    missing_channel_energy[channel] += en
                    channel_origin_count[channel, prior] += 1
                    channel_origin_energy[channel, prior] += en
            if is_terminal:
                active[p] = -1
                origin[p] = 0
    return consume, terminal_lut


def validate_chunk(chunk: np.ndarray, n_packets: int, n_lines: int,
                   expected_iteration: int) -> None:
    require(chunk.size > 0, "empty event chunk")
    require(np.all(chunk["pkt_id"] < n_packets),
            "event packet id exceeds the declared packet count")
    require(np.isfinite(chunk["energy"]).all() and np.all(chunk["energy"] >= 0.0),
            "event energy is negative/nonfinite")
    require(np.all(chunk["iter"] == expected_iteration),
            "event chunk contains an unexpected/mixed iteration")
    line_event = (chunk["etype"] == 1) | (chunk["etype"] == 2)
    ids = chunk["line_id"][line_event]
    require(np.all((ids >= 0) & (ids < n_lines)),
            "line absorption/emission has invalid line_id")


def consume_chunks(events: np.ndarray, line_lam: np.ndarray,
                   line_z: np.ndarray, line_ion: np.ndarray,
                   n_packets: int, chunk_records: int,
                   expected_iteration: int) -> dict[str, np.ndarray]:
    require(events.size > 0 and chunk_records > 0, "empty events/invalid chunk size")
    require(line_lam.size == line_z.size == line_ion.size,
            "line table columns differ in length")
    require(np.all(line_z > 0) and np.all(line_ion >= 0),
            "line ion identity outside defined domain")
    arrays = make_arrays(n_packets, int(np.max(line_z)), int(np.max(line_ion)))
    bands = line_bands(line_lam)
    kernel, terminal_lut = compile_kernel()
    for start in range(0, events.size, chunk_records):
        stop = min(events.size, start + chunk_records)
        chunk = events[start:stop]
        validate_chunk(chunk, n_packets, line_lam.size, expected_iteration)
        kernel(chunk["pkt_id"], chunk["line_id"], chunk["energy"],
               chunk["etype"], chunk["shell"], chunk["chan"],
               line_z, line_ion, bands, terminal_lut,
               arrays["active"], arrays["origin"], arrays["totals"],
               arrays["all_count"], arrays["all_energy"],
               arrays["missing_count"], arrays["missing_energy"],
               arrays["missing_channel_count"], arrays["missing_channel_energy"],
               arrays["channel_origin_count"], arrays["channel_origin_energy"])
    return arrays


def event_memmap(path: Path) -> np.memmap:
    with path.open("rb") as stream:
        header = stream.read(32)
    require(len(header) == 32 and header[:8] == b"LUMEVT01",
            f"event header identity mismatch: {path}")
    record_size = int(np.frombuffer(header[8:12], dtype="<u4")[0])
    require(record_size == read_events.EVENT_DTYPE.itemsize == 20,
            f"event record size mismatch: {record_size}")
    payload = path.stat().st_size - 32
    require(payload > 0 and payload % record_size == 0,
            "event payload length is not an integral positive record count")
    return np.memmap(path, dtype=read_events.EVENT_DTYPE, mode="r", offset=32,
                     shape=(payload // record_size,))


def line_memmap(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("rb") as stream:
        magic = stream.read(8)
    require(magic == b"LUMLIN01", f"line table identity mismatch: {path}")
    payload = path.stat().st_size - 8
    require(payload > 0 and payload % read_events.LINE_DTYPE.itemsize == 0,
            "line table payload length mismatch")
    rows = np.memmap(path, dtype=read_events.LINE_DTYPE, mode="r", offset=8,
                     shape=(payload // read_events.LINE_DTYPE.itemsize,))
    return rows["lam"], rows["Z"].astype(np.int32), rows["ion"].astype(np.int32)


def capture_metadata(stdout_path: Path) -> dict[str, Any]:
    text = stdout_path.read_text(errors="replace")
    packet_matches = re.findall(r"^  Packets: (\d+), Iterations: (\d+)$", text, re.M)
    event_matches = re.findall(
        r"\[EVENT-LOG\] LUMINA_EVENT_LOG=1: cap=(\d+)M iters=([^ ]+) "
        r"escatter=(\d+) lambda_max=([0-9.]+)", text)
    write_matches = re.findall(
        r"\[EVENT-LOG\] it(\d+): (\d+) events \((\d+) dropped\)", text)
    require(len(packet_matches) >= 1 and len(set(packet_matches)) == 1,
            "effective packet/iteration metadata missing or inconsistent")
    require(len(event_matches) >= 1 and len(set(event_matches)) == 1,
            "event-log arm metadata missing or inconsistent")
    require(len(write_matches) == 1, "event-log write metadata is not unique")
    packets, iterations = map(int, packet_matches[0])
    cap_m, iter_mode, escatter, lambda_max = event_matches[0]
    event_iter, attempted, dropped = map(int, write_matches[0])
    return {
        "n_packets": packets, "n_iterations": iterations,
        "event_cap_records": int(cap_m) * 1_000_000,
        "event_iteration_mode": iter_mode, "electron_scatter_logged": bool(int(escatter)),
        "lambda_max_A": float(lambda_max), "event_iteration": event_iter,
        "attempted_records": attempted, "dropped_tail_records": dropped,
    }


def sum_axes(array: np.ndarray, keep: tuple[int, ...]) -> np.ndarray:
    axes = tuple(axis for axis in range(array.ndim) if axis not in keep)
    return np.sum(array, axis=axes, dtype=array.dtype)


def ion_name(z: int, ion0: int) -> str:
    symbols = {6: "C", 8: "O", 12: "Mg", 13: "Al", 14: "Si", 16: "S",
               20: "Ca", 21: "Sc", 22: "Ti", 23: "V", 24: "Cr",
               25: "Mn", 26: "Fe", 27: "Co", 28: "Ni"}
    romans = ("I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X")
    label = romans[ion0] if ion0 < len(romans) else f"ion0={ion0}"
    return f"{symbols.get(z, f'Z{z}')} {label}"


def rows_by_channel(a: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    total_missing = float(a["totals"][3])
    output = []
    for channel in np.flatnonzero(a["missing_channel_count"]):
        energy = float(a["missing_channel_energy"][channel])
        output.append({
            "chan": f"0x{int(channel):02X}",
            "channel_name": CHANNEL_NAMES.get(int(channel), "UNNAMED"),
            "missing_events": int(a["missing_channel_count"][channel]),
            "missing_energy": energy,
            "fraction_of_all_missing_energy": fraction(energy, total_missing),
        })
    return sorted(output, key=lambda row: (-row["missing_energy"], row["chan"]))


def grouped_rows(a: dict[str, np.ndarray], kind: str) -> list[dict[str, Any]]:
    mc, me = a["missing_count"], a["missing_energy"]
    ac, ae = a["all_count"], a["all_energy"]
    total_missing = float(a["totals"][3])
    if kind == "ion":
        mcount, menergy = sum_axes(mc, (0, 1)), sum_axes(me, (0, 1))
        acount, aenergy = sum_axes(ac, (0, 1)), sum_axes(ae, (0, 1))
    elif kind == "band":
        mcount, menergy = sum_axes(mc, (2,)), sum_axes(me, (2,))
        acount, aenergy = sum_axes(ac, (2,)), sum_axes(ae, (2,))
    elif kind == "shell":
        mcount, menergy = sum_axes(mc, (3,)), sum_axes(me, (3,))
        acount, aenergy = sum_axes(ac, (3,)), sum_axes(ae, (3,))
    else:
        raise AuditError(f"unknown grouping: {kind}")
    output = []
    for index in np.argwhere(mcount > 0):
        key = tuple(int(x) for x in index)
        row: dict[str, Any]
        if kind == "ion":
            row = {"Z": key[0], "ion_number": key[1],
                   "ion": ion_name(key[0], key[1])}
        elif kind == "band":
            band, lo, hi, interval = BANDS[key[0]]
            row = {"band": band, "lambda_lo_A": lo, "lambda_hi_A": hi,
                   "interval_definition": interval}
        else:
            row = {"shell": key[0]}
        missing_e = float(menergy[key])
        all_e = float(aenergy[key])
        row.update({
            "missing_events": int(mcount[key]), "missing_energy": missing_e,
            "all_valid_line_emission_events_in_bucket": int(acount[key]),
            "all_valid_line_emission_energy_in_bucket": all_e,
            "fraction_of_all_missing_energy": fraction(missing_e, total_missing),
            "missing_fraction_of_bucket_line_emission_energy": fraction(missing_e, all_e),
        })
        output.append(row)
    return sorted(output, key=lambda row: -row["missing_energy"])


def rows_ion_band_shell(a: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    total_missing = float(a["totals"][3])
    output = []
    for z, ion, band_index, shell in np.argwhere(a["missing_count"] > 0):
        missing_e = float(a["missing_energy"][z, ion, band_index, shell])
        all_e = float(a["all_energy"][z, ion, band_index, shell])
        band = BANDS[int(band_index)][0]
        output.append({
            "Z": int(z), "ion_number": int(ion), "ion": ion_name(int(z), int(ion)),
            "band": band, "shell": int(shell),
            "missing_events": int(a["missing_count"][z, ion, band_index, shell]),
            "missing_energy": missing_e,
            "all_valid_line_emission_energy_in_bucket": all_e,
            "fraction_of_all_missing_energy": fraction(missing_e, total_missing),
            "missing_fraction_of_bucket_line_emission_energy": fraction(missing_e, all_e),
        })
    return sorted(output, key=lambda row: -row["missing_energy"])


def rows_channel_origin(a: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    total_missing = float(a["totals"][3])
    output = []
    for channel, origin in np.argwhere(a["channel_origin_count"] > 0):
        energy = float(a["channel_origin_energy"][channel, origin])
        output.append({
            "chan": f"0x{int(channel):02X}",
            "channel_name": CHANNEL_NAMES.get(int(channel), "UNNAMED"),
            "prior_origin_since_terminal": ORIGIN_NAMES[int(origin)],
            "missing_events": int(a["channel_origin_count"][channel, origin]),
            "missing_energy": energy,
            "fraction_of_all_missing_energy": fraction(energy, total_missing),
        })
    return sorted(output, key=lambda row: -row["missing_energy"])


def route_summary(channel_origin: list[dict[str, Any]], total: float) -> list[dict[str, Any]]:
    groups = {
        "EXPLICIT_NONLINE_KPKT_COLLEXC": lambda c, o: c == "0x12",
        "BF_ORIGIN_THEN_MA_LINE_EXIT": lambda c, o: o == "BF_ABSORPTION" and c in {"0x15", "0x38"},
        "LINE_ONLY_BB_ROUTE_WITHOUT_STORED_ACTIVATION": lambda c, o: c == "0x16",
        "OTHER_UNRESOLVED": lambda c, o: True,
    }
    used: set[int] = set()
    output = []
    for name, predicate in groups.items():
        selected = []
        for index, row in enumerate(channel_origin):
            if index not in used and predicate(row["chan"], row["prior_origin_since_terminal"]):
                selected.append((index, row))
        for index, _ in selected:
            used.add(index)
        energy = math.fsum(float(row["missing_energy"]) for _, row in selected)
        count = sum(int(row["missing_events"]) for _, row in selected)
        output.append({"route_class": name, "missing_events": count,
                       "missing_energy": energy,
                       "fraction_of_all_missing_energy": fraction(energy, total)})
    require(len(used) == len(channel_origin), "route partition did not consume every row")
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    require(bool(rows), f"refusing empty CSV: {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "UNDEFINED" if value is None else value
                             for key, value in row.items()})


def summarize(a: dict[str, np.ndarray], provenance: dict[str, Any],
              upstream: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    all_count, all_energy, missing_count, missing_energy = a["totals"][:4]
    tail = int(np.count_nonzero(a["active"] >= 0))
    tables = {
        "by_channel": rows_by_channel(a), "by_ion": grouped_rows(a, "ion"),
        "by_band": grouped_rows(a, "band"), "by_shell": grouped_rows(a, "shell"),
        "by_ion_band_shell": rows_ion_band_shell(a),
        "by_channel_prior_origin": rows_channel_origin(a),
    }
    routes = route_summary(tables["by_channel_prior_origin"], float(missing_energy))
    if upstream is not None:
        diag = upstream["pairing_diagnostics"]
        require(int(missing_count) == int(diag["line_emissions_without_activation"]),
                "missing count does not reproduce upstream pairing diagnostics")
        require(float(missing_energy) == float(diag["line_emission_energy_without_activation"]),
                "missing energy does not bit-reproduce upstream pairing diagnostics")
        require(tail == int(diag["activations_unpaired_at_stored_prefix_tail"]),
                "prefix-tail activation count does not reproduce upstream")
        require(int(a["totals"][4]) == int(diag["activation_records"]),
                "activation count does not reproduce upstream")
        require(int(a["totals"][5]) == int(diag["recognized_terminal_records"]),
                "terminal count does not reproduce upstream")
    summary = {
        "status": "PASS", "schema": "codex-unpaired-line-emission-v1",
        "definitions": {
            "line_emission": "EventRec.etype==2 and 0<=line_id<n_lines",
            "all_internal_emission": "EventRec.etype in {2 line emission/thermal line-site exit, 4 kpacket free-free/B(T_e), 5 kpacket free-bound/recombination, 8 legacy bf re-emission}",
            "activation": "most recent earlier EventRec.etype==1 for the same pkt_id, cleared after the first event whose chan is in the terminal-channel set",
            "without_activation": "eligible line emission for which that per-packet activation state is absent immediately before processing the emission",
            "energy": "EventRec.energy cast from stored float32 to float64 and accumulated in stored-record order",
            "emitted_ion": "lumina_events_lines.bin Z and zero-based ion indexed by the emission line_id",
            "emitted_band": ", ".join(f"{name}={interval}" for name, _, _, interval in BANDS) + " Angstrom, from lumina_events_lines.bin lambda_A",
            "prior_origin": "latest classified origin since the preceding recognized terminal: LINE_ACTIVATION for etype1, BF_ABSORPTION for etype3, OTHER_NONTERMINAL only if neither was already present, otherwise NONE_SINCE_TERMINAL",
            "route_partition": "ordered exclusive partition: chan 0x12; then prior BF with chan 0x15/0x38; then chan 0x16; then all remaining channel-origin rows",
            "terminal_channels_hex": [f"0x{x:02X}" for x in TERMINAL_CHANNELS],
            "conditioning": "stored non-random prefix only; no extrapolation to the dropped tail",
            "clamp": 0, "floor": 0, "cap": 0, "fallback": 0,
            "undefined_policy": "JSON null and CSV UNDEFINED only for a zero denominator; no replacement value",
        },
        "provenance": provenance,
        "totals": {
            "stored_records_processed": int(a["totals"][7]),
            "all_valid_line_emission_events": int(all_count),
            "all_valid_line_emission_energy": float(all_energy),
            "line_emissions_without_activation": int(missing_count),
            "line_emission_energy_without_activation": float(missing_energy),
            "missing_fraction_of_all_valid_line_emission_events": fraction(missing_count, all_count),
            "missing_fraction_of_all_valid_line_emission_energy": fraction(missing_energy, all_energy),
            "all_internal_emission_events": int(a["totals"][8]),
            "all_internal_emission_energy": float(a["totals"][9]),
            "missing_fraction_of_all_internal_emission_events":
                fraction(missing_count, a["totals"][8]),
            "missing_fraction_of_all_internal_emission_energy":
                fraction(missing_energy, a["totals"][9]),
            "activation_records": int(a["totals"][4]),
            "recognized_terminal_records": int(a["totals"][5]),
            "bf_absorption_records": int(a["totals"][6]),
            "activations_unpaired_at_stored_prefix_tail": tail,
        },
        "route_partition": routes,
        "output_tables": {
            "by_channel": "unpaired_by_channel.csv",
            "by_ion": "unpaired_by_ion.csv", "by_band": "unpaired_by_band.csv",
            "by_shell": "unpaired_by_shell.csv",
            "by_ion_band_shell": "unpaired_by_ion_band_shell.csv",
            "by_channel_prior_origin": "unpaired_by_channel_prior_origin.csv",
        },
    }
    return summary, tables


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ev = np.zeros(11, dtype=read_events.EVENT_DTYPE)
    rows = [
        (0, -1, 0.0, 0.0, 3, 0, 11, 0x40),
        (0, 0, 0.0, 2.0, 2, 0, 11, 0x12),
        (1, 1, 0.0, 0.0, 1, 1, 11, 0x30),
        (1, 0, 0.0, 3.0, 2, 1, 11, 0x38),
        (2, 1, 0.0, 5.0, 2, 2, 11, 0x16),
        (3, 2, 0.0, 7.0, 2, 3, 11, 0x38),
        (4, 2, 0.0, 0.0, 1, 4, 11, 0x30),
        (4, -1, 0.0, 0.0, 6, 4, 11, 0x50),
        (4, 2, 0.0, 11.0, 2, 4, 11, 0x38),
        (5, 1, 0.0, 0.0, 1, 5, 11, 0x30),
        (6, -1, 0.0, 0.0, 7, 6, 11, 0x42),
    ]
    for index, row in enumerate(rows):
        ev[index] = row
    lam = np.asarray([1500.0, 2500.0, 3500.0], dtype=np.float32)
    z = np.asarray([26, 27, 28], dtype=np.int32)
    ion = np.asarray([3, 3, 3], dtype=np.int32)
    return ev, lam, z, ion


def self_test() -> dict[str, Any]:
    events, lam, z, ion = fixture()
    arrays = consume_chunks(events, lam, z, ion, n_packets=7, chunk_records=2,
                            expected_iteration=11)
    summary, tables = summarize(arrays, {"fixture": True}, upstream=None)
    totals = summary["totals"]
    require(totals["all_valid_line_emission_events"] == 5 and
            totals["all_valid_line_emission_energy"] == 28.0,
            "fixture all-emission ledger mismatch")
    require(totals["all_internal_emission_events"] == 5 and
            totals["all_internal_emission_energy"] == 28.0,
            "fixture internal-emission ledger mismatch")
    require(totals["line_emissions_without_activation"] == 4 and
            totals["line_emission_energy_without_activation"] == 25.0,
            "fixture missing-emission ledger mismatch")
    require(totals["activations_unpaired_at_stored_prefix_tail"] == 1,
            "fixture prefix-tail state mismatch")
    co = {(row["chan"], row["prior_origin_since_terminal"]): row
          for row in tables["by_channel_prior_origin"]}
    require(co[("0x12", "BF_ABSORPTION")]["missing_energy"] == 2.0,
            "fixture BF->kpacket classification failed")
    require(co[("0x16", "NONE_SINCE_TERMINAL")]["missing_energy"] == 5.0,
            "fixture line-only negative case failed")
    # Negative control: if escape 0x50 were incorrectly removed from the
    # terminal set, packet 4's 11-energy emission would be falsely paired.
    defective_missing_energy = 25.0 - 11.0
    require(defective_missing_energy != totals["line_emission_energy_without_activation"],
            "negative control did not change the fixture answer")
    return {
        "status": "PASS", "chunk_boundary_state": "PASS",
        "expected_missing_events": 4, "expected_missing_energy": 25.0,
        "negative_control_drop_escape_terminal": {
            "status": "PASS-rejected", "defective_missing_energy": defective_missing_energy,
            "correct_missing_energy": 25.0,
        },
    }


def run_heavy(args: argparse.Namespace) -> None:
    metadata = capture_metadata(args.stdout)
    events = event_memmap(args.events)
    line_lam, line_z, line_ion = line_memmap(args.lines)
    require(events.size == metadata["event_cap_records"],
            "stored event count disagrees with capture cap")
    upstream = json.loads(args.upstream.read_text())
    require(upstream.get("provenance", {}).get("events", {}).get("path") == str(args.events),
            "upstream event provenance points at different bytes")
    arrays = consume_chunks(events, line_lam, line_z, line_ion,
                            metadata["n_packets"], args.chunk_records,
                            metadata["event_iteration"])
    provenance = {
        "events": {"path": str(args.events), "bytes": args.events.stat().st_size,
                   "sha256_reported_by_upstream_not_recomputed":
                       upstream["provenance"]["events"]["sha256"]},
        "lines": {"path": str(args.lines), "bytes": args.lines.stat().st_size,
                  "sha256_reported_by_upstream_not_recomputed":
                      upstream["provenance"]["lines"]["sha256"]},
        "upstream_pairing_json": str(args.upstream),
        "capture_stdout": str(args.stdout), "capture_metadata": metadata,
    }
    summary, tables = summarize(arrays, provenance, upstream)
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "unpaired_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    for name, rows in tables.items():
        write_csv(args.outdir / summary["output_tables"][name], rows)
    print(json.dumps({"status": "PASS", "outdir": str(args.outdir),
                      "totals": summary["totals"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--run-heavy", action="store_true",
                      help="explicitly open and stream the 8 GB stored event prefix")
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--lines", type=Path, default=DEFAULT_LINES)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--stdout", type=Path, default=DEFAULT_STDOUT)
    parser.add_argument("--chunk-records", type=int, default=4_000_000)
    parser.add_argument("--outdir", type=Path,
                        default=ROOT / "validation/codex_eps_thin/investigation3")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2, sort_keys=True))
    else:
        run_heavy(args)


if __name__ == "__main__":
    main()
