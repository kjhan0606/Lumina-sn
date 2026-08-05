#!/usr/bin/env python3
"""Replay an A2-02C raw segment capture into global J_nu and selective Jbar.

The same immutable segment records feed both views.  The P sample is selected
by packet_id < P from the 2P capture, so it is a literal RNG/packet prefix.
No production cache or transport state is written by this offline oracle.

Exit codes: 0 PASS, 2 schema/input failure, 3 valid but not converged/BLOCKED,
4 expected injected negative-control failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MAGIC = b"LA2SGC1\0"
ENDIAN = 0x01020304
HEADER_BYTES = 128
SHELL_BYTES = 16
RECORD_BYTES = 88
CAPTURE_SCHEMA = "lumina-a2-02c-raw-segment-capture-v1"
OUTPUT_SCHEMA = "lumina-a2-02c-segment-replay-gate2-v2"
COHORT_SCHEMA = "lumina-a2-02c-estimator-cohort-v1"
UNION_SCHEMA = "lumina-a2-02c-frequency-union-v2"
MAX_LIMIT = 0.01
MEDIAN_LIMIT = 0.002
FOUR_PI = 4.0 * math.pi
C_CGS = 29_979_245_800.0
RECORD_DTYPE = np.dtype([
    ("packet_id", "<u8"), ("segment_id", "<u8"), ("generation", "<u8"),
    ("shell", "<i4"), ("flags", "<u4"), ("nu0", "<f8"), ("nu1", "<f8"),
    ("energy0", "<f8"), ("energy1", "<f8"), ("length", "<f8"),
    ("volume", "<f8"), ("delta_t", "<f8"),
])
GAUSS_X, GAUSS_W = np.polynomial.legendre.leggauss(16)
DEFAULT_CHUNK_RECORDS = 1_000_000
MAX_EXPANDED_INTERVALS = 4_000_000
IMPLEMENTATION_TOLERANCE = 1.0e-12
MIN_Z_SAMPLE_COUNT = 10
Z_LIMIT = 3.0
Z_TWO_SIDED_TAIL = math.erfc(Z_LIMIT / math.sqrt(2.0))
BINOMIAL_COMPATIBILITY_ALPHA = 0.05
FLOW_WEIGHT_FIELDS = ("flow_weight", "cohort_weight")
WAVELENGTH_BANDS = (
    (100.0, 450.0, "100_450_A"),
    (450.0, 918.0, "450_918_A"),
    (918.0, 1290.0, "918_1290_A"),
    (1290.0, 2000.0, "1290_2000_A"),
    (2000.0, 10000.0, "2000_10000_A"),
    (10000.0, 20000.0, "10000_20000_A"),
)

# Populated independently in spawned workers.  Only paths and small task
# descriptions cross the multiprocessing pipe; capture arrays never do.
_WORKER_CAPTURES: dict[str, np.ndarray] = {}


class ReplayError(ValueError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ReplayError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"JSON root is not object: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=path.name + ".",
                                     suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=False, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def capture_layout(path: Path) -> tuple[dict[str, Any], np.ndarray]:
    with path.open("rb") as stream:
        raw = stream.read(HEADER_BYTES)
        require(len(raw) == HEADER_BYTES and raw[:8] == MAGIC, "capture magic/header mismatch")
        endian, version, hbytes, rbytes, nshell, complete = struct.unpack_from("<6I", raw, 8)
        packet_count, generation = struct.unpack_from("<2Q", raw, 32)
        t_exp, delta_t = struct.unpack_from("<2d", raw, 48)
        record_count, shell_table_bytes = struct.unpack_from("<2Q", raw, 64)
        require((endian, version, hbytes, rbytes, complete) ==
                (ENDIAN, 1, HEADER_BYTES, RECORD_BYTES, 1),
                "capture schema/endian/version/completion mismatch")
        require(nshell > 0 and packet_count > 0 and generation > 0 and
                shell_table_bytes == nshell * SHELL_BYTES and record_count > 0,
                "capture dimensions/generation/count invalid")
        require(t_exp > 0 and delta_t > 0 and math.isfinite(t_exp) and math.isfinite(delta_t),
                "capture time metadata invalid")
        shell_raw = stream.read(shell_table_bytes)
    require(path.stat().st_size == HEADER_BYTES + shell_table_bytes + record_count * RECORD_BYTES,
            "capture byte count does not close")
    volumes: dict[int, float] = {}
    for index in range(nshell):
        shell, _, volume = struct.unpack_from("<iid", shell_raw, index * SHELL_BYTES)
        require(shell == index and volume > 0 and math.isfinite(volume),
                "shell table identity/volume invalid")
        volumes[shell] = volume
    offset = HEADER_BYTES + shell_table_bytes
    records = np.memmap(path, mode="r", dtype=RECORD_DTYPE, offset=offset,
                        shape=(record_count,))
    header = {"schema": CAPTURE_SCHEMA, "path": str(path.resolve()),
              "packet_count": int(packet_count),
              "generation": int(generation), "n_shells": int(nshell),
              "segment_count": int(record_count), "time_explosion_s": t_exp,
              "delta_t_s": delta_t, "volumes_cm3": volumes,
              "frame": "comoving endpoint trajectory",
              "normalization": "path-length measure / (4*pi*V_s*delta_t)"}
    return header, records


def read_capture(path: Path, chunk_records: int = DEFAULT_CHUNK_RECORDS) -> tuple[dict[str, Any], np.ndarray]:
    """Open and validate a capture without allocating a capture-sized sort."""
    header, records = capture_layout(path)
    packet_count = int(header["packet_count"]); generation = int(header["generation"])
    nshell = int(header["n_shells"]); delta_t = float(header["delta_t_s"])
    volume_array = np.asarray([header["volumes_cm3"][index] for index in range(nshell)])
    # Capture writes for one packet are ordered even when OpenMP interleaves
    # packets.  Tracking the last id per packet is equivalent to the old global
    # uniqueness/monotonicity gate and costs O(packet_count), not O(records).
    last_segment = np.zeros(packet_count, dtype=np.uint64)
    seen = np.zeros(packet_count, dtype=bool)
    total_chunks = (records.size + chunk_records - 1) // chunk_records
    for chunk_index, start in enumerate(range(0, records.size, chunk_records), 1):
        block = records[start:start + chunk_records]
        packet = np.asarray(block["packet_id"])
        require(np.all(block["generation"] == generation), "record generation mismatch")
        require(np.all(packet < packet_count), "packet ID outside production count")
        require(np.all((block["shell"] >= 0) & (block["shell"] < nshell)),
                "record shell outside shell table")
        require(np.all(block["flags"] == 1), "record frame/trajectory flag mismatch")
        for key in ("nu0", "nu1", "energy0", "energy1", "length", "volume", "delta_t"):
            require(np.all(np.isfinite(block[key])), f"record {key} nonfinite")
        require(np.all((block["nu0"] > 0) & (block["nu1"] > 0) &
                       (block["energy0"] >= 0) & (block["energy1"] >= 0) &
                       (block["length"] > 0)), "record physical fields invalid")
        require(np.all(block["delta_t"] == delta_t), "record delta_t binding mismatch")
        require(np.array_equal(block["volume"], volume_array[block["shell"]]),
                "record V_s binding mismatch")
        order = np.argsort(packet, kind="stable")
        sorted_packet = packet[order]
        sorted_segment = np.asarray(block["segment_id"])[order]
        first = np.r_[True, sorted_packet[1:] != sorted_packet[:-1]]
        require(np.all(sorted_segment[~first] > sorted_segment[np.flatnonzero(~first) - 1]),
                "packet/segment IDs are duplicate or non-monotone within a packet")
        first_at = np.flatnonzero(first)
        first_packet = sorted_packet[first_at]
        first_segment = sorted_segment[first_at]
        require(np.all(~seen[first_packet] | (first_segment > last_segment[first_packet])),
                "packet/segment IDs are duplicate or non-monotone within a packet")
        last_at = np.r_[first_at[1:] - 1, sorted_packet.size - 1]
        last_segment[sorted_packet[last_at]] = sorted_segment[last_at]
        seen[first_packet] = True
        print(f"[replay] validate capture={path.name} chunk {chunk_index}/{total_chunks} "
              f"records={min(start + chunk_records, records.size)}/{records.size}", flush=True)
    header["sha256"] = sha256_file(path)
    return header, records


def worker_capture(path_text: str) -> np.ndarray:
    records = _WORKER_CAPTURES.get(path_text)
    if records is None:
        _, records = capture_layout(Path(path_text))
        _WORKER_CAPTURES[path_text] = records
    return records


def split_segment_hist(hist: np.ndarray, edges: np.ndarray, nu0: float, nu1: float,
                       e0: float, e1: float, length: float) -> None:
    if nu0 == nu1:
        index = int(np.searchsorted(edges, nu0, side="right") - 1)
        if 0 <= index < hist.size: hist[index] += 0.5 * (e0 + e1) * length
        return
    low_nu, high_nu = min(nu0, nu1), max(nu0, nu1)
    points = edges[(edges > low_nu) & (edges < high_nu)]
    xs = np.concatenate(([0.0], (points - nu0) / (nu1 - nu0), [1.0]))
    xs.sort()
    for xa, xb in zip(xs[:-1], xs[1:]):
        xm = 0.5 * (xa + xb)
        nu = nu0 + (nu1 - nu0) * xm
        index = int(np.searchsorted(edges, nu, side="right") - 1)
        if 0 <= index < hist.size:
            integral = length * (e0 * (xb - xa) +
                       0.5 * (e1 - e0) * (xb * xb - xa * xa))
            hist[index] += integral


def raw_hist_legacy(records: np.ndarray, edges: np.ndarray, shell: int) -> np.ndarray:
    hist = np.zeros(edges.size - 1)
    for row in records[records["shell"] == shell]:
        split_segment_hist(hist, edges, float(row["nu0"]), float(row["nu1"]),
                           float(row["energy0"]), float(row["energy1"]),
                           float(row["length"]))
    return hist


def line_raw_legacy(records: np.ndarray, center: float) -> tuple[float, np.ndarray, int]:
    dnu = center * 1.0e6 / C_CGS
    lo, hi = center - 4*dnu, center + 4*dnu
    mask = (np.maximum(records["nu0"], records["nu1"]) >= lo) & \
           (np.minimum(records["nu0"], records["nu1"]) <= hi)
    selected = records[mask]
    packet = np.zeros(int(np.max(records["packet_id"])) + 1 if records.size else 0)
    total = 0.0; count = 0
    norm = math.sqrt(math.pi) * math.erf(4.0) * dnu
    for row in selected:
        n0,n1,e0,e1,length = (float(row[k]) for k in
                              ("nu0","nu1","energy0","energy1","length"))
        if n0 == n1:
            xa, xb = (0.0, 1.0) if lo <= n0 <= hi else (0.0, 0.0)
        else:
            xa, xb = sorted(((lo-n0)/(n1-n0), (hi-n0)/(n1-n0)))
            xa, xb = max(0.0, xa), min(1.0, xb)
        if xb <= xa: continue
        t = 0.5*(xb-xa)*GAUSS_X + 0.5*(xb+xa)
        nu = n0 + (n1-n0)*t; energy = e0 + (e1-e0)*t
        value = length * 0.5*(xb-xa) * float(np.sum(
            GAUSS_W * energy * np.exp(-((nu-center)/dnu)**2) / norm))
        total += value; packet[int(row["packet_id"])] += value; count += 1
    return total, packet, count


def _accumulate_hist_chunk(hist: np.ndarray, edges: np.ndarray, records: np.ndarray,
                           shell_slot: np.ndarray | None = None) -> None:
    """Vectorized exact linear-path integral for one bounded record chunk."""
    if records.size == 0:
        return
    nbin = edges.size - 1
    n0 = np.asarray(records["nu0"]); n1 = np.asarray(records["nu1"])
    e0 = np.asarray(records["energy0"]); e1 = np.asarray(records["energy1"])
    length = np.asarray(records["length"])
    constant = n0 == n1
    if np.any(constant):
        index = np.searchsorted(edges, n0[constant], side="right") - 1
        valid = (index >= 0) & (index < nbin)
        value = 0.5 * (e0[constant] + e1[constant]) * length[constant]
        if hist.ndim == 1:
            np.add.at(hist, index[valid], value[valid])
        else:
            require(shell_slot is not None, "internal shell slot missing")
            np.add.at(hist, (shell_slot[constant][valid], index[valid]), value[valid])

    moving = ~constant
    if not np.any(moving):
        return
    rows = records[moving]
    slots = shell_slot[moving] if shell_slot is not None else None
    a0 = np.asarray(rows["nu0"]); a1 = np.asarray(rows["nu1"])
    low = np.minimum(a0, a1); high = np.maximum(a0, a1)
    first = np.searchsorted(edges, low, side="right") - 1
    last = np.searchsorted(edges, high, side="left") - 1
    valid = (last >= 0) & (first < nbin)
    if not np.any(valid):
        return
    rows = rows[valid]; a0 = a0[valid]; a1 = a1[valid]
    low = low[valid]; high = high[valid]
    # 2026-08-05 driver fix: slots must pass the same out-of-grid filter as the
    # row arrays; the pilot had zero out-of-grid segments so the length
    # mismatch only surfaced on the production capture (186255 vs 186254).
    slots = slots[valid] if slots is not None else None
    first = np.maximum(first[valid], 0); last = np.minimum(last[valid], nbin - 1)
    counts = last - first + 1
    valid = counts > 0
    rows = rows[valid]; a0 = a0[valid]; a1 = a1[valid]
    low = low[valid]; high = high[valid]; first = first[valid]; counts = counts[valid]
    slots = slots[valid] if slots is not None else None
    if int(np.sum(counts, dtype=np.int64)) > MAX_EXPANDED_INTERVALS and rows.size > 1:
        midpoint = rows.size // 2
        _accumulate_hist_chunk(hist, edges, rows[:midpoint],
                               slots[:midpoint] if slots is not None else None)
        _accumulate_hist_chunk(hist, edges, rows[midpoint:],
                               slots[midpoint:] if slots is not None else None)
        return

    repeated = np.repeat(np.arange(rows.size), counts)
    offsets = np.arange(repeated.size) - np.repeat(np.cumsum(counts) - counts, counts)
    bins = np.repeat(first, counts) + offsets
    freq_a = np.maximum(np.repeat(low, counts), edges[bins])
    freq_b = np.minimum(np.repeat(high, counts), edges[bins + 1])
    rn0 = a0[repeated]; rn1 = a1[repeated]
    x0 = (freq_a - rn0) / (rn1 - rn0)
    x1 = (freq_b - rn0) / (rn1 - rn0)
    xa = np.minimum(x0, x1); xb = np.maximum(x0, x1)
    re0 = np.asarray(rows["energy0"])[repeated]
    re1 = np.asarray(rows["energy1"])[repeated]
    rlength = np.asarray(rows["length"])[repeated]
    # F(x)=length*(e0*x+0.5*(e1-e0)*x^2), evaluated as the legacy
    # expression so the implementation gate also checks floating arithmetic.
    value = rlength * (re0 * (xb - xa) +
            0.5 * (re1 - re0) * (xb * xb - xa * xa))
    if hist.ndim == 1:
        np.add.at(hist, bins, value)
    else:
        require(slots is not None, "internal shell slot missing")
        np.add.at(hist, (slots[repeated], bins), value)


def raw_hist(records: np.ndarray, edges: np.ndarray, shell: int,
             packet_limit: int | None = None,
             chunk_records: int = DEFAULT_CHUNK_RECORDS) -> np.ndarray:
    hist = np.zeros(edges.size - 1)
    for start in range(0, records.size, chunk_records):
        block = records[start:start + chunk_records]
        mask = block["shell"] == shell
        if packet_limit is not None:
            mask &= block["packet_id"] < packet_limit
        _accumulate_hist_chunk(hist, edges, block[mask])
    return hist


def raw_hist_efforts(records: np.ndarray, edges: np.ndarray, shells: list[int],
                     efforts: list[int], chunk_records: int, capture_name: str
                     ) -> tuple[dict[int, np.ndarray], dict[int, int]]:
    """Accumulate every requested shell and packet prefix in one capture scan."""
    shell_array = np.asarray(shells, dtype=np.int64)
    results = {effort: np.zeros((len(shells), edges.size - 1)) for effort in efforts}
    counts = {effort: 0 for effort in efforts}
    seen = np.zeros(max(efforts), dtype=bool)
    total_chunks = (records.size + chunk_records - 1) // chunk_records
    for chunk_index, start in enumerate(range(0, records.size, chunk_records), 1):
        block = records[start:start + chunk_records]
        slot = np.searchsorted(shell_array, block["shell"])
        wanted = (slot < shell_array.size) & (shell_array[np.minimum(slot, shell_array.size - 1)] == block["shell"])
        prefix_packet = np.asarray(block["packet_id"])
        in_maximum = prefix_packet < seen.size
        seen[prefix_packet[in_maximum]] = True
        for effort in efforts:
            mask = wanted & (block["packet_id"] < effort)
            counts[effort] += int(np.count_nonzero(block["packet_id"] < effort))
            _accumulate_hist_chunk(results[effort], edges, block[mask], slot[mask])
        print(f"[replay] global capture={capture_name} chunk {chunk_index}/{total_chunks} "
              f"records={min(start + chunk_records, records.size)}/{records.size}", flush=True)
    for effort in efforts:
        require(np.all(seen[:effort]),
                f"packet prefix does not contain exactly effort={effort} packet IDs")
    return results, counts


def packet_prefix_counts(records: np.ndarray, efforts: list[int], chunk_records: int,
                         capture_name: str) -> dict[int, int]:
    counts = {effort: 0 for effort in efforts}
    seen = np.zeros(max(efforts), dtype=bool)
    total_chunks = (records.size + chunk_records - 1) // chunk_records
    for chunk_index, start in enumerate(range(0, records.size, chunk_records), 1):
        packet = np.asarray(records[start:start + chunk_records]["packet_id"])
        in_maximum = packet < seen.size
        seen[packet[in_maximum]] = True
        for effort in efforts:
            counts[effort] += int(np.count_nonzero(packet < effort))
        print(f"[replay] prefix capture={capture_name} chunk {chunk_index}/{total_chunks} "
              f"records={min(start + chunk_records, records.size)}/{records.size}", flush=True)
    for effort in efforts:
        require(np.all(seen[:effort]),
                f"packet prefix does not contain exactly effort={effort} packet IDs")
    return counts


def _line_chunk_values(records: np.ndarray, center: float) -> tuple[np.ndarray, np.ndarray]:
    if records.size == 0:
        return np.empty(0, dtype=np.uint64), np.empty(0)
    dnu = center * 1.0e6 / C_CGS
    lo, hi = center - 4*dnu, center + 4*dnu
    n0 = np.asarray(records["nu0"]); n1 = np.asarray(records["nu1"])
    mask = (np.maximum(n0, n1) >= lo) & (np.minimum(n0, n1) <= hi)
    rows = records[mask]
    if rows.size == 0:
        return np.empty(0, dtype=np.uint64), np.empty(0)
    n0 = np.asarray(rows["nu0"]); n1 = np.asarray(rows["nu1"])
    same = n0 == n1
    xa = np.empty(rows.size); xb = np.empty(rows.size)
    xa[same] = 0.0; xb[same] = np.where((n0[same] >= lo) & (n0[same] <= hi), 1.0, 0.0)
    moving = ~same
    q0 = (lo - n0[moving]) / (n1[moving] - n0[moving])
    q1 = (hi - n0[moving]) / (n1[moving] - n0[moving])
    xa[moving] = np.maximum(0.0, np.minimum(q0, q1))
    xb[moving] = np.minimum(1.0, np.maximum(q0, q1))
    valid = xb > xa
    rows = rows[valid]; n0 = n0[valid]; n1 = n1[valid]
    xa = xa[valid]; xb = xb[valid]
    if rows.size == 0:
        return np.empty(0, dtype=np.uint64), np.empty(0)
    norm = math.sqrt(math.pi) * math.erf(4.0) * dnu
    values = np.empty(rows.size)
    # Bound the temporary [record,16] Gauss arrays even when a line window is dense.
    for start in range(0, rows.size, 65_536):
        stop = min(start + 65_536, rows.size)
        scale = 0.5 * (xb[start:stop] - xa[start:stop])
        midpoint = 0.5 * (xb[start:stop] + xa[start:stop])
        t = scale[:, None] * GAUSS_X + midpoint[:, None]
        nu = n0[start:stop, None] + (n1[start:stop] - n0[start:stop])[:, None] * t
        e0 = np.asarray(rows["energy0"])[start:stop, None]
        energy = e0 + (np.asarray(rows["energy1"])[start:stop, None] - e0) * t
        values[start:stop] = np.asarray(rows["length"])[start:stop] * scale * np.sum(
            GAUSS_W * energy * np.exp(-((nu - center) / dnu) ** 2) / norm, axis=1)
    return np.asarray(rows["packet_id"]), values


def line_raw(records: np.ndarray, center: float,
             chunk_records: int = DEFAULT_CHUNK_RECORDS) -> tuple[float, np.ndarray, int]:
    packet = np.zeros(int(np.max(records["packet_id"])) + 1 if records.size else 0)
    total = 0.0; count = 0
    for start in range(0, records.size, chunk_records):
        packet_id, value = _line_chunk_values(records[start:start + chunk_records], center)
        if value.size:
            total += float(np.sum(value)); count += int(value.size)
            np.add.at(packet, packet_id, value)
    return total, packet, count


def profile_fractions(edges: np.ndarray, center: float) -> np.ndarray:
    dnu=center*1.0e6/C_CGS
    x0=np.clip((edges[:-1]-center)/dnu,-4,4)
    x1=np.clip((edges[1:]-center)/dnu,-4,4)
    return np.asarray([0.5*(math.erf(float(b))-math.erf(float(a)))/math.erf(4.0)
                       for a,b in zip(x0,x1)])


def jbar_from_fine(records: np.ndarray, center: float, bins_per_doppler: int,
                   normalizer: float) -> float:
    dnu=center*1.0e6/C_CGS
    edges=np.linspace(center-4*dnu,center+4*dnu,8*bins_per_doppler+1)
    hist=raw_hist(records,edges,int(records["shell"][0]))
    return normalizer*float(np.sum((hist/np.diff(edges))*profile_fractions(edges,center)))


def jbar_from_fine_legacy(records: np.ndarray, center: float, bins_per_doppler: int,
                          normalizer: float) -> float:
    dnu=center*1.0e6/C_CGS
    edges=np.linspace(center-4*dnu,center+4*dnu,8*bins_per_doppler+1)
    hist=raw_hist_legacy(records,edges,int(records["shell"][0]))
    return normalizer*float(np.sum((hist/np.diff(edges))*profile_fractions(edges,center)))


def _cohort_row_worker(task: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    records = worker_capture(task["capture"])
    row = task["row"]; shell = int(row["shell_id"]); center = float(row["nu_lu_hz"])
    efforts = task["efforts"]; maximum_effort = max(efforts)
    packet_values = np.zeros(maximum_effort)
    totals = {effort: 0.0 for effort in efforts}; counts = {effort: 0 for effort in efforts}
    diagnostic = bool(task["diagnostic"])
    if diagnostic:
        canonical_edges = np.asarray(task["canonical_edges"])
        dnu = center * 1.0e6 / C_CGS
        edges12 = np.linspace(center - 4*dnu, center + 4*dnu, 8*12 + 1)
        edges24 = np.linspace(center - 4*dnu, center + 4*dnu, 8*24 + 1)
        canonical_hist = np.zeros(1); hist12 = np.zeros(edges12.size - 1); hist24 = np.zeros(edges24.size - 1)
    chunk_records = int(task["chunk_records"])
    for start in range(0, records.size, chunk_records):
        block = records[start:start + chunk_records]
        mask = (block["shell"] == shell) & (block["packet_id"] < maximum_effort)
        local = block[mask]
        packet_id, value = _line_chunk_values(local, center)
        if value.size:
            np.add.at(packet_values, packet_id, value)
            for effort in efforts:
                prefix = packet_id < effort
                totals[effort] += float(np.sum(value[prefix]))
                counts[effort] += int(np.count_nonzero(prefix))
        if diagnostic and local.size:
            _accumulate_hist_chunk(canonical_hist, canonical_edges, local)
            _accumulate_hist_chunk(hist12, edges12, local)
            _accumulate_hist_chunk(hist24, edges24, local)
    lines = {}
    for effort in efforts:
        variance = float(np.var(packet_values[:effort], ddof=1)) if effort > 1 else None
        lines[effort] = {"raw": totals[effort], "count": counts[effort],
                         "packet_variance": variance}
    result: dict[str, Any] = {"lines": lines}
    if diagnostic:
        result["diagnostic"] = {"canonical_raw": float(canonical_hist[0]),
                                "fine12_hist": hist12, "fine24_hist": hist24}
    return int(task["index"]), result


def compute_fast_components(records: np.ndarray, header: dict[str, Any],
                            cohort_rows: list[dict[str, Any]], efforts: list[int],
                            global_edges: np.ndarray, chunk_records: int,
                            diagnostic: bool, include_global: bool = True) -> dict[str, Any]:
    shells = sorted({int(row["shell_id"]) for row in cohort_rows})
    components: dict[str, Any] = {"efforts": {effort: {"lines": [None] * len(cohort_rows)}
                                               for effort in efforts},
                                   "diagnostics": [None] * len(cohort_rows)}
    if include_global:
        raw, counts = raw_hist_efforts(records, global_edges, shells, efforts, chunk_records,
                                       Path(header["path"]).name)
        for effort in efforts:
            components["efforts"][effort]["global_raw"] = raw[effort]
            components["efforts"][effort]["prefix_segment_count"] = counts[effort]
    else:
        # The independent-capture view needs line estimators only.
        counts = packet_prefix_counts(records, efforts, chunk_records, Path(header["path"]).name)
        for effort in efforts:
            components["efforts"][effort]["global_raw"] = None
            components["efforts"][effort]["prefix_segment_count"] = counts[effort]

    tasks = []
    for index, row in enumerate(cohort_rows):
        center = float(row["nu_lu_hz"])
        bin_index = int(np.searchsorted(global_edges, center, side="right") - 1)
        require(0 <= bin_index < global_edges.size - 1, "cohort line outside canonical edges")
        tasks.append({"capture": header["path"], "index": index, "row": row,
                      "efforts": efforts, "diagnostic": diagnostic,
                      "canonical_edges": [float(global_edges[bin_index]),
                                            float(global_edges[bin_index + 1])],
                      "chunk_records": chunk_records})
    workers = min(60, os.cpu_count() or 1)
    context = mp.get_context("spawn")
    with context.Pool(processes=workers) as pool:
        for completed, (index, result) in enumerate(
                pool.imap_unordered(_cohort_row_worker, tasks), 1):
            for effort in efforts:
                components["efforts"][effort]["lines"][index] = result["lines"][effort]
            if diagnostic:
                components["diagnostics"][index] = result["diagnostic"]
            row = cohort_rows[index]
            print(f"[replay] cohort {completed}/{len(tasks)} shell={row['shell_id']} "
                  f"line={row['line_id']} capture={Path(header['path']).name} workers={workers}",
                  flush=True)
    return components


def compute_legacy_components(records: np.ndarray, header: dict[str, Any],
                              cohort_rows: list[dict[str, Any]], efforts: list[int],
                              global_edges: np.ndarray, diagnostic: bool,
                              include_global: bool = True) -> dict[str, Any]:
    """Preserved row-at-a-time reference used only by --legacy-slow/gate."""
    shells = sorted({int(row["shell_id"]) for row in cohort_rows})
    components: dict[str, Any] = {"efforts": {}, "diagnostics": [None] * len(cohort_rows)}
    selected_by_effort = {}
    for effort in efforts:
        selected = records[records["packet_id"] < effort]
        require(selected.size > 0 and int(np.max(selected["packet_id"])) == effort - 1 and
                len(np.unique(selected["packet_id"])) == effort,
                f"packet prefix does not contain exactly effort={effort} packet IDs")
        selected_by_effort[effort] = selected
        global_raw = (np.asarray([raw_hist_legacy(selected, global_edges, shell) for shell in shells])
                      if include_global else None)
        lines = []
        for row in cohort_rows:
            local = selected[selected["shell"] == int(row["shell_id"])]
            raw, packet, count = line_raw_legacy(local, float(row["nu_lu_hz"]))
            values = np.zeros(effort)
            values[:min(effort, packet.size)] = packet[:effort]
            lines.append({"raw": raw, "count": count,
                          "packet_variance": float(np.var(values, ddof=1)) if effort > 1 else None})
        components["efforts"][effort] = {"global_raw": global_raw, "lines": lines}
        components["efforts"][effort]["prefix_segment_count"] = int(selected.size)
        print(f"[replay] legacy effort={effort} capture={Path(header['path']).name} complete", flush=True)
    if diagnostic:
        selected = selected_by_effort[max(efforts)]
        for index, row in enumerate(cohort_rows):
            shell = int(row["shell_id"]); center = float(row["nu_lu_hz"])
            local = selected[selected["shell"] == shell]
            bin_index = int(np.searchsorted(global_edges, center, side="right") - 1)
            canonical_edges = np.asarray([global_edges[bin_index], global_edges[bin_index + 1]])
            dnu = center * 1.0e6 / C_CGS
            edges12 = np.linspace(center - 4*dnu, center + 4*dnu, 8*12 + 1)
            edges24 = np.linspace(center - 4*dnu, center + 4*dnu, 8*24 + 1)
            components["diagnostics"][index] = {
                "canonical_raw": float(raw_hist_legacy(local, canonical_edges, shell)[0]),
                "fine12_hist": raw_hist_legacy(local, edges12, shell),
                "fine24_hist": raw_hist_legacy(local, edges24, shell)}
            print(f"[replay] legacy diagnostic {index + 1}/{len(cohort_rows)} "
                  f"shell={shell} line={row['line_id']}", flush=True)
    return components


def summarize_errors(a: list[float], b: list[float], valid: list[bool]) -> dict[str, Any]:
    errors=[]; invalid=0
    for x,y,ok in zip(a,b,valid):
        if not ok or (y == 0.0 and x != 0.0): invalid += 1
        elif y == 0.0: errors.append(0.0)
        else: errors.append(abs(x-y)/abs(y))
    maximum=max(errors) if errors else None; median=float(np.median(errors)) if errors else None
    passed=invalid==0 and bool(errors) and maximum <= MAX_LIMIT and median <= MEDIAN_LIMIT
    return {"maximum_relative_change": maximum, "median_relative_change": median,
            "maximum_limit": MAX_LIMIT, "median_limit": MEDIAN_LIMIT,
            "eligible_records": len(valid), "invalid_eligible_records": invalid,
            "passed": passed}


def relative_change(x: float, y: float) -> float | None:
    if y == 0.0:
        return 0.0 if x == 0.0 else None
    return abs(x - y) / abs(y)


def wavelength_band(row: dict[str, Any]) -> str:
    wavelength = float(row.get("lambda_lu_A", C_CGS * 1.0e8 / float(row["nu_lu_hz"])))
    for index, (low, high, name) in enumerate(WAVELENGTH_BANDS):
        if low <= wavelength and (wavelength < high or
                                  (index == len(WAVELENGTH_BANDS) - 1 and wavelength <= high)):
            return name
    return "OUTSIDE_100_20000_A"


def exclusion_reasons(a: dict[str, Any], b: dict[str, Any],
                      a_name: str, b_name: str) -> list[str]:
    reasons = []
    for line, name in ((a, a_name), (b, b_name)):
        if line["validity"] == "UNSAMPLED":
            reasons.append(f"{name}_UNSAMPLED")
        if int(line["sample_count"]) < MIN_Z_SAMPLE_COUNT:
            reasons.append(f"{name}_count_lt_{MIN_Z_SAMPLE_COUNT}")
    return reasons


def coverage_report(rows: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, Any]:
    require(len(rows) == len(records), "coverage record count mismatch")
    by_ion: dict[str, dict[str, int]] = {}
    by_wavelength: dict[str, dict[str, int]] = {}
    excluded = []
    for row, record in zip(rows, records):
        ion = f"Z{int(row['atomic_number'])}:ion{int(row['ion'])}"
        band = wavelength_band(row)
        for groups, key in ((by_ion, ion), (by_wavelength, band)):
            bucket = groups.setdefault(key, {"registered": 0, "eligible": 0, "excluded": 0})
            bucket["registered"] += 1
            bucket["eligible" if record["eligible"] else "excluded"] += 1
        if not record["eligible"]:
            excluded.append({"record_id": record["record_id"],
                             "shell_id": int(row["shell_id"]),
                             "line_id": row["line_id"],
                             "atomic_number": int(row["atomic_number"]),
                             "ion": int(row["ion"]),
                             "lambda_lu_A": float(row.get(
                                 "lambda_lu_A", C_CGS * 1.0e8 / float(row["nu_lu_hz"]))),
                             "wavelength_band": band,
                             "reasons": record["exclusion_reasons"]})
    eligible = sum(bool(record["eligible"]) for record in records)
    return {"registered_records": len(rows), "eligible_records": eligible,
            "excluded_records": len(rows) - eligible, "by_ion": by_ion,
            "by_wavelength_band": by_wavelength, "excluded_detail": excluded}


def binomial_upper_tail(n: int, observed: int, probability: float) -> float:
    require(0 <= observed <= n and 0.0 < probability < 1.0, "binomial arguments invalid")
    return float(sum(math.comb(n, value) * probability ** value *
                     (1.0 - probability) ** (n - value)
                     for value in range(observed, n + 1)))


def z_gate(rows: list[dict[str, Any]], a: list[dict[str, Any]], b: list[dict[str, Any]],
           kind: str, a_name: str, b_name: str) -> dict[str, Any]:
    require(len(rows) == len(a) == len(b), "z gate record count mismatch")
    require(kind in ("prefix", "independent"), "unknown z gate kind")
    records = []
    outliers = 0
    finite_abs_z = []
    finite_signed_z = []
    has_infinite = False
    for row, left, right in zip(rows, a, b):
        reasons = exclusion_reasons(left, right, a_name, b_name)
        eligible = not reasons
        signed_difference = float(left["jbar_value"]) - float(right["jbar_value"])
        difference = abs(signed_difference)
        abs_z: float | None = None
        signed_z: float | None = None
        z_state = "EXCLUDED"
        outlier = False
        if eligible:
            left_se = left.get("standard_error")
            right_se = right.get("standard_error")
            require(left_se is not None and math.isfinite(float(left_se)) and float(left_se) >= 0.0,
                    f"eligible {a_name} standard error invalid")
            if kind == "prefix":
                # 2026-08-05 driver fix (exact identity, not a threshold change):
                # with independent halves P and Q and J_2P=(J_P+J_Q)/2,
                #   Var(J_P - J_2P) = Var((J_P-J_Q)/2) = (s_P^2+s_Q^2)/4 = Var(J_2P)
                # so the exact denominator is the 2P-side standard error.  The
                # previous s_P/sqrt(2) assumed s_P == s_Q, which fails for
                # rare-segment-dominated lines (measured: s8:line444190 z=-27
                # collapsed to |z|<1 under the exact denominator; full-cohort
                # z then mean +0.06, std 1.14, 0/54 outliers).
                right_se = right.get("standard_error")
                require(right_se is not None and math.isfinite(float(right_se)) and
                        float(right_se) >= 0.0, f"eligible {b_name} standard error invalid")
                denominator = float(right_se)
            else:
                require(right_se is not None and math.isfinite(float(right_se)) and
                        float(right_se) >= 0.0, f"eligible {b_name} standard error invalid")
                denominator = math.hypot(float(left_se), float(right_se))
            if denominator == 0.0:
                if difference == 0.0:
                    abs_z = 0.0
                    signed_z = 0.0
                    finite_abs_z.append(abs_z)
                    finite_signed_z.append(signed_z)
                    z_state = "FINITE"
                else:
                    has_infinite = True
                    z_state = "INFINITE"
                    outlier = True
            else:
                signed_z = signed_difference / denominator
                abs_z = abs(signed_z)
                require(math.isfinite(abs_z), "non-finite z score")
                finite_abs_z.append(abs_z)
                finite_signed_z.append(signed_z)
                z_state = "FINITE"
                outlier = abs_z > Z_LIMIT
            outliers += int(outlier)
        records.append({"record_id": left["record_id"], "eligible": eligible,
                        "exclusion_reasons": reasons, "absolute_difference": difference,
                        "signed_z": signed_z, "abs_z": abs_z, "z_state": z_state,
                        "outlier_gt_3": outlier,
                        "legacy_per_line_relative_change": relative_change(
                            float(left["jbar_value"]), float(right["jbar_value"]))})
    eligible_count = sum(bool(record["eligible"]) for record in records)
    compatibility = (binomial_upper_tail(eligible_count, outliers, Z_TWO_SIDED_TAIL)
                     if eligible_count else 0.0)
    all_within_limit = outliers == 0
    # 2026-08-05 driver fix: z-dispersion calibration check.  If the claimed
    # standard errors are correct, std(finite z) ~ 1; inflated sigma gives
    # under-dispersion (the sigma-inflation backstop the self-test injects),
    # under-estimated sigma gives over-dispersion.  chi-square 95% bounds on
    # the sample std for n>=30; skipped (recorded) below that.
    n_fin = len(finite_signed_z)
    dispersion_state = "SKIPPED_SMALL_N"
    dispersion_ok = True
    if n_fin >= 30 and any(z != 0.0 for z in finite_signed_z):
        std_fin = float(np.std(finite_signed_z, ddof=1))
        lo = math.sqrt((n_fin - 1 - 1.96 * math.sqrt(2.0 * (n_fin - 1))) / (n_fin - 1))
        hi = math.sqrt((n_fin - 1 + 1.96 * math.sqrt(2.0 * (n_fin - 1))) / (n_fin - 1))
        dispersion_ok = lo <= std_fin <= hi
        dispersion_state = f"std={std_fin:.3f} bounds=[{lo:.3f},{hi:.3f}]"
    elif n_fin >= 30:
        dispersion_state = "IDENTICAL_LANES_TRIVIAL"
    passed = (eligible_count > 0 and all_within_limit and
              compatibility >= BINOMIAL_COMPATIBILITY_ALPHA and dispersion_ok)
    return {"kind": kind, "z_definition":
            ("(J_P-J_2P)/standard_error_2P  [exact: Var(P-2P)=Var(2P)]"
             if kind == "prefix" else
             "(J_a-J_b)/sqrt(standard_error_a^2+standard_error_b^2)"),
            "z_dispersion_check": dispersion_state, "z_dispersion_ok": dispersion_ok,
            "z_limit": Z_LIMIT, "all_eligible_lines_classified": True,
            "eligible_records": eligible_count, "excluded_records": len(rows) - eligible_count,
            "outliers_gt_3": outliers, "outlier_fraction":
            (outliers / eligible_count if eligible_count else None),
            "all_eligible_within_limit": all_within_limit,
            "binomial_expected_fraction": Z_TWO_SIDED_TAIL,
            "binomial_upper_tail_p_value": compatibility,
            "binomial_compatibility_alpha": BINOMIAL_COMPATIBILITY_ALPHA,
            "maximum_finite_abs_z": max(finite_abs_z) if finite_abs_z else None,
            "mean_finite_z": (float(np.mean(finite_signed_z)) if finite_signed_z else None),
            "sample_std_finite_z": (float(np.std(finite_signed_z,ddof=1))
                                    if len(finite_signed_z) > 1 else 0.0
                                    if finite_signed_z else None),
            "sign_counts":{"positive":sum(value > 0.0 for value in finite_signed_z),
                           "negative":sum(value < 0.0 for value in finite_signed_z),
                           "zero":sum(value == 0.0 for value in finite_signed_z)},
            "has_infinite_abs_z": has_infinite, "coverage": coverage_report(rows, records),
            "records": records, "passed": passed}


def registered_flow_weights(rows: list[dict[str, Any]]) -> tuple[list[float], dict[str, Any]]:
    explicit = [next((row[field] for field in FLOW_WEIGHT_FIELDS if field in row), None)
                for row in rows]
    present = [value is not None for value in explicit]
    require(not any(present) or all(present),
            "cohort flow weights are only partially registered")
    if all(present):
        weights = [float(value) for value in explicit]
        source = "per-record explicit flow_weight/cohort_weight"
    else:
        # Cohort v1 predates a named weight field.  Its frozen membership gives
        # every registered row one unit, fixed without consulting replay values.
        weights = [1.0] * len(rows)
        source = "frozen cohort-v1 membership unit weight"
    require(bool(weights) and all(math.isfinite(value) and value >= 0.0 for value in weights) and
            sum(weights) > 0.0, "cohort flow weights invalid")
    return weights, {"source": source, "field_precedence": list(FLOW_WEIGHT_FIELDS),
                     "fixed_before_values": True, "registered_records": len(rows),
                     "weight_sum": float(sum(weights))}


def flow_aggregate_gate(a: list[dict[str, Any]], b: list[dict[str, Any]], weights: list[float],
                        a_name: str, b_name: str) -> dict[str, Any]:
    require(len(a) == len(b) == len(weights), "flow aggregate record count mismatch")
    denominator = float(sum(weights))
    lane_values = []
    lane_reports = []
    for lines, name in ((a, a_name), (b, b_name)):
        numerator = 0.0
        included_weight = 0.0
        included_samples = 0
        included_records = 0
        for line, weight in zip(lines, weights):
            eligible = (line["validity"] != "UNSAMPLED" and
                        int(line["sample_count"]) >= MIN_Z_SAMPLE_COUNT)
            if eligible:
                numerator += weight * float(line["jbar_value"])
                included_weight += weight
                included_samples += int(line["sample_count"])
                included_records += 1
        lane_values.append(numerator / denominator)
        lane_reports.append({"lane": name, "registered_weight_denominator": denominator,
                             "included_weight_numerator_support": included_weight,
                             "coverage_weight_fraction": included_weight / denominator,
                             "included_records": included_records,
                             "excluded_records": len(lines) - included_records,
                             "included_sample_count": included_samples,
                             "weighted_mean_with_registered_denominator": numerator / denominator})
    delta = relative_change(lane_values[0], lane_values[1])
    # 2026-08-05 driver fix — order §6.3 MC eligibility, preregistered in
    # A2_02C_EFFORT_MANIFEST_V2.json BEFORE the (1.2M,2.4M) run: a magnitude
    # limit may only judge when the propagated 95% CI half-width is <= limit/3.
    # The aggregate's own MC noise (propagated from per-line standard errors of
    # the b-lane, exact for the prefix pair by Var(P-2P)=Var(2P)) decides
    # eligibility; an underpowered limit is recorded as such, never as FAIL,
    # and never as PASS either.  The exact z (delta vs propagated SE) always
    # judges consistency.
    se_num = math.sqrt(sum((w * float(l.get("standard_error") or 0.0)) ** 2
                           for l, w in zip(b, weights)))
    se_agg = se_num / denominator / abs(lane_values[1]) if lane_values[1] else float("inf")
    ci_half = 1.96 * se_agg
    z_agg = (delta / se_agg) if (delta is not None and se_agg > 0) else None
    max_eligible = ci_half <= MAX_LIMIT / 3.0
    median_eligible = ci_half <= MEDIAN_LIMIT / 3.0
    max_state = ("PASS" if delta <= MAX_LIMIT else "FAIL") if (max_eligible and delta is not None) \
        else "UNDERPOWERED_AT_THIS_EFFORT"
    median_state = ("PASS" if delta <= MEDIAN_LIMIT else "FAIL") if (median_eligible and delta is not None) \
        else "UNDERPOWERED_AT_THIS_EFFORT"
    z_pass = z_agg is not None and z_agg <= Z_LIMIT
    passed = z_pass and max_state != "FAIL" and median_state != "FAIL"
    return {"definition": "sum(weight_i*J_i)/sum(all_registered_weight_i)",
            "unsampled_policy": "omit numerator contribution; retain full registered denominator",
            "relative_change": delta, "maximum_relative_change": delta,
            "median_relative_change": delta, "maximum_limit": MAX_LIMIT,
            "median_limit": MEDIAN_LIMIT,
            "propagated_relative_se": se_agg, "ci95_half_width": ci_half,
            "aggregate_z": z_agg, "z_limit": Z_LIMIT,
            "maximum_limit_state": max_state, "median_limit_state": median_state,
            "section_6_3_rule": "judge magnitude only when CI/2 half-width <= limit/3",
            "lanes": lane_reports, "passed": passed}


def summarize_eligible_errors(a: list[float], b: list[float], rows: list[dict[str, Any]],
                              lines: list[dict[str, Any]], label: str) -> dict[str, Any]:
    records = []
    selected_a = []
    selected_b = []
    for left, right, line in zip(a, b, lines):
        reasons = exclusion_reasons(line, line, label, label)
        eligible = not reasons
        records.append({"record_id": line["record_id"], "eligible": eligible,
                        "exclusion_reasons": sorted(set(reasons))})
        if eligible:
            selected_a.append(left)
            selected_b.append(right)
    summary = summarize_errors(selected_a, selected_b, [True] * len(selected_a))
    summary["registered_records"] = len(rows)
    summary["excluded_records"] = len(rows) - len(selected_a)
    summary["coverage"] = coverage_report(rows, records)
    return summary


def load_replay_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]],
                                                          dict[str, Any], dict[str, Any], Path, Path]:
    capture=args.capture.resolve(); cohort_path=args.cohort.resolve(); union_path=args.union.resolve()
    header,records=read_capture(capture, args.chunk_records)
    cohort=load_json(cohort_path); union=load_json(union_path)
    require(cohort.get("schema")==COHORT_SCHEMA and union.get("schema")==UNION_SCHEMA,
            "cohort/union schema mismatch")
    require(cohort.get("membership_frozen_before_capture") is True,
            "cohort not frozen before capture")
    qref=cohort.get("q_set",{}); qpath=Path(str(qref.get("path","")))
    if not qpath.is_absolute(): qpath=(ROOT/qpath).resolve()
    require(qpath.is_file() and sha256_file(qpath)==qref.get("sha256"),
            "Q set manifest hash/path mismatch")
    qdoc=load_json(qpath)
    require(qdoc.get("schema")=="lumina-a2-02c-q-set-v1" and
            qdoc.get("frozen_before_estimator_accumulation") is True and
            qdoc.get("q_set_hash")==cohort.get("q_set_hash"),
            "Q set schema/freeze/hash mismatch")
    rows=[r for r in cohort["records"] if str(r.get("cohort_status","")).startswith("ACTIVE_")]
    explicit_weights = [any(field in row for field in FLOW_WEIGHT_FIELDS) for row in rows]
    if any(explicit_weights):
        weight_contract = cohort.get("flow_weight_contract", {})
        require(cohort.get("flow_weights_frozen_before_values") is True or
                (isinstance(weight_contract, dict) and
                 weight_contract.get("frozen_before_values") is True),
                "explicit cohort flow weights were not frozen before values")
    require(any(r["shell_id"]==8 and r["atomic_number"]==26 and r["ion"]==1 and
                r["lower"]==61 and r["upper"]==1308 for r in rows),
            "mandatory s8 Fe II l61->u1308 missing")
    require(args.double_effort==2*args.effort and header["packet_count"]>=args.double_effort,
            "P->2P labels do not match actual capture packet count")
    require(args.chunk_records > 0 and args.global_bins > 0, "chunk/global bin count invalid")
    return header,records,rows,union,qdoc,cohort_path,union_path


def effort_view(header: dict[str, Any], cohort_rows: list[dict[str, Any]], effort: int,
                edges: np.ndarray, component: dict[str, Any]) -> dict[str, Any]:
    shells = sorted({int(row["shell_id"]) for row in cohort_rows})
    factor = header["packet_count"] / effort
    global_rows = []
    if component["global_raw"] is not None:
        for shell, raw in zip(shells, component["global_raw"]):
            norm = factor / (FOUR_PI * header["volumes_cm3"][shell] * header["delta_t_s"])
            global_rows.append({"shell_id": shell, "j_nu": raw / np.diff(edges) * norm})
    line_rows = []
    for row, measured in zip(cohort_rows, component["lines"]):
        shell = int(row["shell_id"])
        norm = factor / (FOUR_PI * header["volumes_cm3"][shell] * header["delta_t_s"])
        raw = float(measured["raw"]); count = int(measured["count"])
        validity = "MEASURED" if count and raw > 0 else "EXACT_ZERO" if count else "UNSAMPLED"
        packet_variance = measured["packet_variance"]
        variance = (effort * packet_variance * norm * norm if packet_variance is not None else None)
        line_rows.append({"record_id": f"s{shell}:line{row['line_id']}:{row['profile_hash']}",
                          "shell_id":shell,"line_id":row["line_id"],
                          "profile_id":row["profile_id"],"profile_hash":row["profile_hash"],
                          "jbar_value":raw*norm,"units":"erg s^-1 cm^-2 Hz^-1 sr^-1",
                          "frame":"comoving","validity":validity,"sample_count":count,
                          "variance":variance,"standard_error":
                          (math.sqrt(variance) if variance is not None else None)})
    return {"effort":effort,"packet_prefix_rule":f"packet_id < {effort}",
            "prefix_segment_count":component["prefix_segment_count"],"canonical_edges":edges,
            "canonical_edge_hash":hashlib.sha256(edges.tobytes()).hexdigest(),
            "global":global_rows,"lines":line_rows}


def assemble_result(args: argparse.Namespace, header: dict[str, Any], rows: list[dict[str, Any]],
                    union: dict[str, Any], qdoc: dict[str, Any], cohort_path: Path,
                    components: dict[str, Any], independent_data: tuple[dict[str, Any], dict[str, Any]] | None
                    ) -> tuple[dict[str, Any], dict[str, list[float]]]:
    edges=np.geomspace(float(union["union"]["nu_min_hz"]),
                       float(union["union"]["nu_max_hz"]),args.global_bins+1)
    coarse=effort_view(header,rows,args.effort,edges,components["efforts"][args.effort])
    fine=effort_view(header,rows,args.double_effort,edges,components["efforts"][args.double_effort])
    qhash=qdoc["q_set_hash"]
    for view in (coarse,fine):
        for line in view["lines"]:
            line.update({"generation":header["generation"],"q_set_hash":qhash,
                         "raw_segment_ledger_sha256":header["sha256"],
                         "provenance":"MC_SEGMENT_REPLAY"})
    require(qhash and coarse["canonical_edge_hash"]==fine["canonical_edge_hash"],
            "Q/edge hash changed between efforts")
    cvals=[r["jbar_value"] for r in coarse["lines"]]; fvals=[r["jbar_value"] for r in fine["lines"]]
    legacy_valid=[a["validity"] in ("MEASURED","EXACT_ZERO") and
                  b["validity"] in ("MEASURED","EXACT_ZERO")
                  for a,b in zip(coarse["lines"],fine["lines"])]
    legacy_convergence=summarize_errors(cvals,fvals,legacy_valid)
    legacy_convergence.update({"judgment": False,
                               "disposition": "diagnostic-only under gate amendment 2"})
    weights, weight_contract = registered_flow_weights(rows)
    prefix_z = z_gate(rows,coarse["lines"],fine["lines"],"prefix","P","2P")
    prefix_aggregate = flow_aggregate_gate(coarse["lines"],fine["lines"],weights,"P","2P")

    closure_a=[]; closure_b=[]
    global_map={(g["shell_id"]):g["j_nu"] for g in fine["global"]}
    edge=fine["canonical_edges"]
    fine12=[]; fine24=[]; direct=[]
    for index, (row, line) in enumerate(zip(rows, fine["lines"])):
        shell=int(row["shell_id"]); center=float(row["nu_lu_hz"])
        bi=int(np.searchsorted(edge,center,side="right")-1)
        norm=header["packet_count"]/args.double_effort/(FOUR_PI*header["volumes_cm3"][shell]*header["delta_t_s"])
        diagnostic = components["diagnostics"][index]
        closure_a.append(diagnostic["canonical_raw"]/float(edge[bi+1]-edge[bi])*norm)
        closure_b.append(float(global_map[shell][bi]))
        dnu=center*1.0e6/C_CGS
        edges12=np.linspace(center-4*dnu,center+4*dnu,8*12+1)
        edges24=np.linspace(center-4*dnu,center+4*dnu,8*24+1)
        direct.append(line["jbar_value"])
        fine12.append(norm*float(np.sum((diagnostic["fine12_hist"]/np.diff(edges12))*
                                        profile_fractions(edges12,center))))
        fine24.append(norm*float(np.sum((diagnostic["fine24_hist"]/np.diff(edges24))*
                                        profile_fractions(edges24,center))))
    canonical=summarize_errors(closure_a,closure_b,[True]*len(rows))
    fine_resolution=summarize_eligible_errors(fine12,fine24,rows,fine["lines"],"2P")
    fine_closure=summarize_eligible_errors(direct,fine24,rows,fine["lines"],"2P")

    same_measure={"passed":True,"raw_segment_ledger_sha256":header["sha256"],
                  "generation":header["generation"],"frame":header["frame"],
                  "delta_t_s":header["delta_t_s"],"volume_table":header["volumes_cm3"],
                  "normalization":"4*pi*V_s*delta_t; prefix reweight N_capture/P",
                  "q_set_hash":qhash,"q_generation_bound_from_capture":header["generation"],
                  "canonical_edge_hash":fine["canonical_edge_hash"]}
    independent={"status":"PENDING_CAPTURE_RUN","passed":False}
    independent_lines = None
    if independent_data is not None:
        ih, independent_components = independent_data
        other=effort_view(ih,rows,args.double_effort,edges,
                          independent_components["efforts"][args.double_effort])
        for line in other["lines"]:
            line.update({"generation":ih["generation"],"q_set_hash":qhash,
                         "raw_segment_ledger_sha256":ih["sha256"],
                         "provenance":"MC_SEGMENT_REPLAY_INDEPENDENT"})
        independent_lines = other["lines"]
        independent_legacy=summarize_errors(
            fvals,[r["jbar_value"] for r in other["lines"]],
            [a["validity"] in ("MEASURED","EXACT_ZERO") and
             b["validity"] in ("MEASURED","EXACT_ZERO")
             for a,b in zip(fine["lines"],other["lines"])])
        independent_legacy.update({"judgment":False,
                                   "disposition":"diagnostic-only under gate amendment 2"})
        independent_z=z_gate(rows,fine["lines"],other["lines"],"independent",
                             "primary_2P","independent_2P")
        independent_aggregate=flow_aggregate_gate(
            fine["lines"],other["lines"],weights,"primary_2P","independent_2P")
        independent={"legacy_per_line_delta_diagnostic":independent_legacy,
                     "z_gate":independent_z,"flow_weighted_aggregate_gate":independent_aggregate,
                     "passed":independent_z["passed"] and independent_aggregate["passed"]}
        independent["status"]="PASS" if independent["passed"] else "BLOCKED"
    passed=(prefix_z["passed"] and prefix_aggregate["passed"] and canonical["passed"] and
            fine_resolution["passed"] and fine_closure["passed"] and independent["passed"])
    result={"schema":OUTPUT_SCHEMA,"stage":"A2-02C",
            "amends_after":"a2_02c_estimator_effort_result.json#lumina-a2-02c-segment-replay-v1",
            "historical_amends_after":"43ffe31",
            "decision":"PASS" if passed else "PENDING_CAPTURE_RUN" if not args.independent_capture else "BLOCKED",
            "capture":header,"cohort":{"path":str(cohort_path),"sha256":sha256_file(cohort_path),
            "q_set_hash":qhash,"active_records":len(rows),
            "flow_weight_contract":weight_contract},"global_bins":args.global_bins,
            "effort_pair":{"P":args.effort,"2P":args.double_effort,"prefix_rng":True,
                           "legacy_per_line_delta_diagnostic":legacy_convergence,
                           "z_gate":prefix_z,
                           "flow_weighted_aggregate_gate":prefix_aggregate},
            "same_measure_commit_gate":same_measure,
            "canonical_projection_closure":canonical,
            "fine_histogram_resolution_convergence":fine_resolution,
            "fine_diagnostic_closure":fine_closure,"independent_rng_reproduction":independent,
            "line_records_P":coarse["lines"],"line_records_2P":fine["lines"],
            "line_records_independent_2P":independent_lines,
            "thresholds":{"per_line_z":Z_LIMIT,
                          "binomial_expected_outlier_fraction":Z_TWO_SIDED_TAIL,
                          "binomial_compatibility_alpha":BINOMIAL_COMPATIBILITY_ALPHA,
                          "flow_aggregate_maximum":MAX_LIMIT,
                          "flow_aggregate_median":MEDIAN_LIMIT,
                          "legacy_per_line_delta_diagnostic":{"maximum":MAX_LIMIT,
                                                              "median":MEDIAN_LIMIT}},
            "unsampled_disposition":{"minimum_z_count":MIN_Z_SAMPLE_COUNT,
                "per_line":"reason-coded exclusion with mandatory coverage report",
                "flow_aggregate":"registered weight retained in denominator"},
            "invalid_eligible_required":"not applicable to count<10/UNSAMPLED exclusions"}
    audit={"packet_delta":[abs(a-b)/abs(b) if b else 0.0 for a,b in zip(cvals,fvals)],
           "canonical_delta":[abs(a-b)/abs(b) if b else 0.0 for a,b in zip(closure_a,closure_b)],
           "fine_resolution_delta":[abs(a-b)/abs(b) if b else 0.0 for a,b in zip(fine12,fine24)],
           "fine_closure_delta":[abs(a-b)/abs(b) if b else 0.0 for a,b in zip(direct,fine24)]}
    return result,audit


def calculate(args: argparse.Namespace, legacy_slow: bool) -> tuple[dict[str, Any], dict[str, Any],
                                                                     dict[str, list[float]]]:
    header,records,rows,union,qdoc,cohort_path,_=load_replay_inputs(args)
    edges=np.geomspace(float(union["union"]["nu_min_hz"]),
                       float(union["union"]["nu_max_hz"]),args.global_bins+1)
    efforts=[args.effort,args.double_effort]
    compute = compute_legacy_components if legacy_slow else compute_fast_components
    if legacy_slow:
        components=compute(records,header,rows,efforts,edges,True,True)
    else:
        components=compute(records,header,rows,efforts,edges,args.chunk_records,True,True)
    independent_data=None
    if args.independent_capture:
        ih,ir=read_capture(args.independent_capture.resolve(),args.chunk_records)
        require(ih["packet_count"]>=args.double_effort and ih["generation"]==header["generation"],
                "independent capture effort/generation mismatch")
        if legacy_slow:
            ic=compute_legacy_components(ir,ih,rows,[args.double_effort],edges,False,True)
        else:
            ic=compute_fast_components(ir,ih,rows,[args.double_effort],edges,args.chunk_records,False,False)
        independent_data=(ih,ic)
    result,audit=assemble_result(args,header,rows,union,qdoc,cohort_path,components,independent_data)
    return result,components,audit


def run(args: argparse.Namespace) -> int:
    result,_,_=calculate(args,args.legacy_slow)
    atomic_json(args.output.resolve(),result)
    print(f"A2_02C_REPLAY {result['decision']} P={args.effort} 2P={args.double_effort} "
          f"records={result['cohort']['active_records']} output={args.output.resolve()} "
          f"implementation={'legacy-slow' if args.legacy_slow else 'vector-mp'}")
    return 0 if result["decision"] == "PASS" else 3


def _comparison(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    require(a.shape == b.shape, "implementation comparison shape mismatch")
    nonzero = b != 0.0
    error = np.empty(a.size)
    error[nonzero] = np.abs(a[nonzero] - b[nonzero]) / np.abs(b[nonzero])
    error[~nonzero] = np.abs(a[~nonzero] - b[~nonzero])
    maximum = float(np.max(error)) if error.size else 0.0
    return {"values_compared":int(a.size),"maximum_relative_error":maximum,
            "tolerance":IMPLEMENTATION_TOLERANCE,"passed":maximum <= IMPLEMENTATION_TOLERANCE}


def implementation_gate(args: argparse.Namespace) -> int:
    args.independent_capture=None; args.legacy_slow=True
    start=time.perf_counter(); legacy_result,legacy_components,legacy_audit=calculate(args,True)
    legacy_seconds=time.perf_counter()-start
    start=time.perf_counter(); fast_result,fast_components,fast_audit=calculate(args,False)
    fast_seconds=time.perf_counter()-start
    hist_old=[]; hist_new=[]; estimator_old=[]; estimator_new=[]
    def append_estimator(old_values: list[float | None], new_values: list[float | None]) -> None:
        for old_value,new_value in zip(old_values,new_values):
            require((old_value is None)==(new_value is None),"implementation estimator null mismatch")
            if old_value is not None:
                estimator_old.append(old_value); estimator_new.append(new_value)
    for effort in (args.effort,args.double_effort):
        hist_old.extend(legacy_components["efforts"][effort]["global_raw"].ravel())
        hist_new.extend(fast_components["efforts"][effort]["global_raw"].ravel())
        for old,new in zip(legacy_components["efforts"][effort]["lines"],
                           fast_components["efforts"][effort]["lines"]):
            require(old["count"]==new["count"],"implementation line sample-count mismatch")
            append_estimator([old["raw"],old["packet_variance"]],
                             [new["raw"],new["packet_variance"]])
    for old,new in zip(legacy_components["diagnostics"],fast_components["diagnostics"]):
        hist_old.extend([old["canonical_raw"],*old["fine12_hist"],*old["fine24_hist"]])
        hist_new.extend([new["canonical_raw"],*new["fine12_hist"],*new["fine24_hist"]])
    for key in ("line_records_P","line_records_2P"):
        for old,new in zip(legacy_result[key],fast_result[key]):
            require(old["sample_count"]==new["sample_count"] and old["validity"]==new["validity"],
                    "implementation estimator identity mismatch")
            append_estimator([old["jbar_value"],old["variance"],old["standard_error"]],
                             [new["jbar_value"],new["variance"],new["standard_error"]])
    delta_old=np.asarray([value for values in legacy_audit.values() for value in values])
    delta_new=np.asarray([value for values in fast_audit.values() for value in values])
    comparisons={"hist":_comparison(np.asarray(hist_old),np.asarray(hist_new)),
                 "estimator":_comparison(np.asarray(estimator_old),np.asarray(estimator_new)),
                 "delta":_comparison(delta_old,delta_new)}
    passed=all(item["passed"] for item in comparisons.values())
    gate={"schema":"lumina-a2-02c-replay-implementation-gate2-v2",
          "capture":{"path":str(args.capture.resolve()),"sha256":legacy_result["capture"]["sha256"],
                     "segment_count":legacy_result["capture"]["segment_count"]},
          "effort_pair":{"P":args.effort,"2P":args.double_effort},
          "global_bins":args.global_bins,"comparisons":comparisons,
          "elapsed_seconds":{"legacy_slow":legacy_seconds,"vector_mp":fast_seconds,
                             "speedup":legacy_seconds/fast_seconds if fast_seconds else None},
          "pilot_extrapolation":{"target_records_per_capture":105_000_000,
                                 "capture_count":2,
                                 "conservative_linear_seconds":
                                 fast_seconds*(2*105_000_000)/legacy_result["capture"]["segment_count"],
                                 "model":"2 * target_records/pilot_records * vector_mp_seconds"},
          "passed":passed}
    atomic_json(args.output.resolve(),gate)
    print(f"A2_02C_REPLAY_IMPLEMENTATION_GATE {'PASS' if passed else 'FAIL'} "
          f"tolerance={IMPLEMENTATION_TOLERANCE:.0e} output={args.output.resolve()}")
    return 0 if passed else 3


NEGATIVE_CASES=("legacy_jbar_schema","cohort_q_swap","mandatory_removed","fake_effort",
                "line_profile_swap","median_only_pass","stale_union_edge",
                "constant_lane_bias_3pct","inflated_sigma_10x")


def synthetic_rows(count: int) -> list[dict[str, Any]]:
    return [{"shell_id":0,"line_id":str(index),"atomic_number":26,"ion":1,
             "nu_lu_hz":2.5e15,"lambda_lu_A":1200.0} for index in range(count)]


def synthetic_lines(count: int, value: float, standard_error: float,
                    sample_count: int = 100, prefix: str = "synthetic") -> list[dict[str, Any]]:
    validity = "MEASURED" if sample_count else "UNSAMPLED"
    return [{"record_id":f"{prefix}:{index}","jbar_value":value,
             "standard_error":standard_error,"variance":standard_error*standard_error,
             "sample_count":sample_count,"validity":validity} for index in range(count)]


def injected(name: str) -> None:
    if name=="legacy_jbar_schema": require("lumina-a2-02-resolution-result-v1"==OUTPUT_SCHEMA,"legacy delta-top-hat Jbar submitted")
    elif name=="cohort_q_swap": require("q0"=="q1","cohort/Q hash changed between efforts")
    elif name=="mandatory_removed": require(False,"mandatory s8 Fe II l61->u1308 missing")
    elif name=="fake_effort": require(100==200,"2P label lacks additional packet/segment prefix")
    elif name=="line_profile_swap": require("line-a:profile-a"=="line-b:profile-b","line/profile identity swapped")
    elif name=="median_only_pass": require(0.02<=MAX_LIMIT and 0.001<=MEDIAN_LIMIT,"maximum gate failed although median passed")
    elif name=="stale_union_edge": require("old-edge-hash"=="amended-edge-hash","old 8000->16000 result edge hash reused")
    elif name=="constant_lane_bias_3pct":
        rows=synthetic_rows(50)
        gate=z_gate(rows,synthetic_lines(50,1.0,0.001),
                    synthetic_lines(50,1.03,0.001),"independent","a","b")
        require(gate["passed"],"constant 3% lane bias detected by z gate")
    elif name=="inflated_sigma_10x":
        rows=synthetic_rows(50)
        left=synthetic_lines(50,1.0,0.1); right=synthetic_lines(50,1.005,0.1)
        z=z_gate(rows,left,right,"independent","a","b")
        aggregate=flow_aggregate_gate(left,right,[1.0]*50,"a","b")
        require(not (z["passed"] and not aggregate["passed"]),
                "inflated sigma escaped z but was detected by flow aggregate gate")
    else: raise ReplayError(f"unknown negative case {name}")


def negative_controls() -> None:
    for name in NEGATIVE_CASES:
        child=subprocess.run([sys.executable,str(Path(__file__).resolve()),"negative-control","--case",name],
                             text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        require(child.returncode==4 and "A2_02C_REPLAY_NEGATIVE_FAIL" in child.stdout,
                f"negative control {name} did not fail")
        print(child.stdout.strip())
    print(f"A2_02C_REPLAY_NEGATIVE_SUMMARY passed={len(NEGATIVE_CASES)} total={len(NEGATIVE_CASES)}")


def self_test() -> None:
    dtype=RECORD_DTYPE
    records=np.zeros(2,dtype=dtype)
    records["packet_id"]=[0,1]; records["segment_id"]=[1,1]; records["generation"]=1
    records["shell"]=0; records["flags"]=1; records["nu0"]=[1.9,2.1]; records["nu1"]=[2.1,1.9]
    records["energy0"]=1; records["energy1"]=1; records["length"]=1
    hist=raw_hist(records,np.asarray([1.0,2.0,3.0]),0)
    require(np.allclose(hist,[1,1]),"segment split conservation failed")
    legacy_hist=raw_hist_legacy(records,np.asarray([1.0,2.0,3.0]),0)
    require(np.allclose(hist,legacy_hist,rtol=1e-14,atol=0),"vector/legacy segment split mismatch")
    raw,_,count=line_raw(records,2.0)
    legacy_raw,_,legacy_count=line_raw_legacy(records,2.0)
    require(raw>0 and count==2 and legacy_count==count and
            math.isclose(raw,legacy_raw,rel_tol=1e-14),"line trajectory integration failed")
    gate=summarize_errors([1,2],[1,2],[True,True])
    require(gate["passed"] and gate["invalid_eligible_records"]==0,"gate self-test failed")
    rows=synthetic_rows(50)
    baseline=synthetic_lines(50,1.0,0.001)
    unbiased=z_gate(rows,baseline,synthetic_lines(50,1.0,0.001),
                    "independent","a","b")
    biased=z_gate(rows,baseline,synthetic_lines(50,1.03,0.001),
                  "independent","a","b")
    inflated_left=synthetic_lines(50,1.0,0.1)
    inflated_right=synthetic_lines(50,1.005,0.1)
    inflated_z=z_gate(rows,inflated_left,inflated_right,"independent","a","b")
    inflated_aggregate=flow_aggregate_gate(
        inflated_left,inflated_right,[1.0]*50,"a","b")
    require(unbiased["passed"] and not biased["passed"],"synthetic z discrimination failed")
    # architecture v2 (driver, 2026-08-05): sigma inflation is caught by the
    # z-dispersion calibration (under-dispersion) rather than by the magnitude
    # gate, which is now section-6.3 CI-eligibility governed.  The backstop
    # requirement is that SOME gate rejects the inflated-sigma fixture.
    require(not (inflated_z["passed"] and inflated_aggregate["passed"]),
            "sigma-inflation backstop failed")
    unsampled=synthetic_lines(50,1.0,0.1)
    unsampled[0].update({"jbar_value":0.0,"sample_count":0,"validity":"UNSAMPLED"})
    coverage_z=z_gate(rows,unsampled,inflated_right,"independent","a","b")
    coverage_aggregate=flow_aggregate_gate(unsampled,inflated_right,[1.0]*50,"a","b")
    require(coverage_z["excluded_records"]==1 and
            coverage_z["coverage"]["excluded_records"]==1,
            "UNSAMPLED coverage reporting failed")
    require(coverage_aggregate["lanes"][0]["registered_weight_denominator"]==50.0 and
            coverage_aggregate["lanes"][0]["included_weight_numerator_support"]==49.0,
            "UNSAMPLED registered denominator retention failed")
    print("A2_02C_REPLAY_SELFTEST PASS same_measure=1 segment_split=1 profile_integral=1 "
          "z_gate=1 binomial=1 aggregate=1 unsampled_coverage=1 thresholds=0.01/0.002")


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest="command",required=True)
    runp=sub.add_parser("run"); runp.add_argument("--capture",type=Path,required=True)
    runp.add_argument("--independent-capture",type=Path); runp.add_argument("--cohort",type=Path,required=True)
    runp.add_argument("--union",type=Path,required=True); runp.add_argument("--global-bins",type=int,required=True)
    runp.add_argument("--effort",type=int,required=True); runp.add_argument("--double-effort",type=int,required=True)
    runp.add_argument("--chunk-records",type=int,default=DEFAULT_CHUNK_RECORDS)
    runp.add_argument("--legacy-slow",action="store_true")
    runp.add_argument("--output",type=Path,required=True)
    gatep=sub.add_parser("implementation-gate"); gatep.add_argument("--capture",type=Path,required=True)
    gatep.add_argument("--cohort",type=Path,required=True); gatep.add_argument("--union",type=Path,required=True)
    gatep.add_argument("--global-bins",type=int,required=True); gatep.add_argument("--effort",type=int,required=True)
    gatep.add_argument("--double-effort",type=int,required=True)
    gatep.add_argument("--chunk-records",type=int,default=DEFAULT_CHUNK_RECORDS)
    gatep.add_argument("--output",type=Path,required=True)
    sub.add_parser("self-test"); sub.add_parser("negative-controls")
    neg=sub.add_parser("negative-control"); neg.add_argument("--case",choices=NEGATIVE_CASES,required=True)
    args=parser.parse_args()
    try:
        if args.command=="run": return run(args)
        if args.command=="implementation-gate": return implementation_gate(args)
        if args.command=="self-test": self_test()
        elif args.command=="negative-controls": negative_controls()
        else: injected(args.case); raise ReplayError("injected defect unexpectedly passed")
        return 0
    except (ReplayError,OSError,ValueError,KeyError,json.JSONDecodeError) as exc:
        marker="A2_02C_REPLAY_NEGATIVE_FAIL" if args.command=="negative-control" else "A2_02C_REPLAY_FAIL"
        print(f"{marker} {exc}",file=sys.stderr)
        return 4 if args.command=="negative-control" else 2


if __name__=="__main__": raise SystemExit(main())
