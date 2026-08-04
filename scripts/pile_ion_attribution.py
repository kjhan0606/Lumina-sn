#!/usr/bin/env python3
"""Offline attribution of MC line-pile emission to activating ions.

This consumer reuses ``scripts/read_events.py`` for both binary event inputs
and ``scripts/emiss_e11_fluor_matrix.py`` for LFMAT001.  It does not run a
model, transport, or a GPU kernel.  Real inputs are accepted only when every
existing SHA-256 sidecar verifies; payloads without sidecars are hashed at
analysis time and explicitly marked as such in provenance.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
from typing import Any, NamedTuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import read_events as event_reader  # noqa: E402
from emiss_e11_fluor_matrix import (  # noqa: E402
    FluorMatrixError,
    read_fluor_matrix,
    write_fixture_matrix,
)
import uv_t2n9_offline as base  # noqa: E402


DEFAULT_CAPTURE = Path(
    "/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932"
)
SCHEMA = "lumina-pile-ion-attribution-v1"
BANDS = (
    ("B0", 600.0, 1000.0),
    ("B1", 1000.0, 1500.0),
    ("B2", 1500.0, 2000.0),
    ("B3", 2000.0, 2500.0),
    ("B4", 2500.0, 3000.0),
    ("BALL", 600.0, 3000.0),
    ("LEGACY_PILE", 1290.0, 2000.0),
)
TERMINAL_CHANNELS = frozenset(
    (0x10, 0x11, 0x12, 0x14, 0x15, 0x16, 0x24, 0x38, 0x3A, 0x50, 0x51)
)
UNATTRIBUTED_Z = -1
UNATTRIBUTED_ION = -1


class AttributionError(RuntimeError):
    pass


class EventTallies(NamedTuple):
    # [scope: all/deep, band, Z-slot, ion-slot]
    totals: np.ndarray
    counts: np.ndarray
    # [scope: all/deep, quantile, band, Z-slot, ion-slot]
    quartile_totals: np.ndarray
    quartile_counts: np.ndarray
    eligible_energy_by_quartile: np.ndarray
    eligible_count_by_quartile: np.ndarray
    # Legacy 1290-2000 A emitted-line-ion control, [scope, Z-slot, ion-slot].
    legacy_emitted_ion_totals: np.ndarray
    legacy_emitted_ion_counts: np.ndarray
    emission_without_activation: int
    emission_without_activation_energy: float
    activation_count: int
    terminal_count: int
    unpaired_prefix_tail: int


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AttributionError(message)


def fraction(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0.0 else numerator / denominator


def relative_change(maximum: float, minimum: float,
                    reference: float) -> float | None:
    return None if reference == 0.0 else (maximum - minimum) / reference


def max_over_min_minus_one(maximum: float, minimum: float) -> float | None:
    return None if minimum == 0.0 else maximum / minimum - 1.0


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
    if (z, ion) == (UNATTRIBUTED_Z, UNATTRIBUTED_ION):
        return "UNATTRIBUTED"
    require(z > 0 and ion >= 0, f"invalid ion key ({z}, {ion})")
    return f"{base.ELEMENT_SYMBOL.get(z, f'Z{z}')} {roman(ion + 1)}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_provenance(path: Path) -> dict[str, Any]:
    """Verify an existing sidecar, or record a digest computed for this read.

    ``sidecar_verified`` establishes agreement with digest evidence created
    outside this analysis.  ``computed_now`` only identifies the bytes this
    analysis read; it is deliberately not represented as historical integrity
    evidence.  A present but malformed, unreadable, or mismatching sidecar is
    always fail-closed.
    """
    sidecar = Path(str(path) + ".sha256")
    try:
        fields = sidecar.read_text().split()
    except FileNotFoundError:
        measured = sha256_file(path)
        return {
            "path": str(path),
            "sha256": measured,
            "sha256_source": "computed_now",
            "sha256_sidecar_path": None,
            "sha256_evidence": "identifies bytes read by this analysis only",
        }
    except OSError as exc:
        raise AttributionError(f"cannot read SHA-256 sidecar: {sidecar}") from exc
    require(bool(fields) and len(fields[0]) == 64,
            f"malformed SHA-256 sidecar: {sidecar}")
    expected = fields[0].lower()
    require(all(ch in "0123456789abcdef" for ch in expected),
            f"non-hex SHA-256 sidecar: {sidecar}")
    measured = sha256_file(path)
    require(measured == expected,
            f"SHA-256 mismatch for {path}: expected {expected}, got {measured}")
    return {
        "path": str(path),
        "sha256": measured,
        "sha256_source": "sidecar_verified",
        "sha256_sidecar_path": str(sidecar),
        "sha256_evidence": "payload matched pre-existing sidecar digest",
    }


def band_index(wavelength_A: float) -> int:
    if 600.0 <= wavelength_A < 1000.0:
        return 0
    if 1000.0 <= wavelength_A < 1500.0:
        return 1
    if 1500.0 <= wavelength_A < 2000.0:
        return 2
    if 2000.0 <= wavelength_A < 2500.0:
        return 3
    if 2500.0 <= wavelength_A <= 3000.0:
        return 4
    return -1


def _consume_python(events: np.ndarray, line_lam: np.ndarray,
                    line_z: np.ndarray, line_ion: np.ndarray,
                    quantiles: int, inject_emitted_ion: bool = False) -> EventTallies:
    """Reference one-pass state machine; fixture-sized inputs only."""
    max_z = int(np.max(line_z))
    max_ion = int(np.max(line_ion))
    nz, ni = max_z + 2, max_ion + 2
    uz, ui = max_z + 1, max_ion + 1
    totals = np.zeros((2, len(BANDS), nz, ni), dtype=np.float64)
    counts = np.zeros_like(totals, dtype=np.int64)
    qtot = np.zeros((2, quantiles, len(BANDS), nz, ni), dtype=np.float64)
    qcount = np.zeros_like(qtot, dtype=np.int64)
    qenergy = np.zeros((2, quantiles), dtype=np.float64)
    qevents = np.zeros((2, quantiles), dtype=np.int64)
    emitted_totals = np.zeros((2, nz, ni), dtype=np.float64)
    emitted_counts = np.zeros((2, nz, ni), dtype=np.int64)
    active: dict[int, int] = {}
    activation_count = terminal_count = missing_count = 0
    missing_energy = 0.0
    n = len(events)
    for index, row in enumerate(events):
        packet = int(row["pkt_id"])
        lid = int(row["line_id"])
        etype = int(row["etype"])
        if etype == 1:
            active[packet] = lid
            activation_count += 1
            continue
        terminal = int(row["chan"]) in TERMINAL_CHANNELS
        if terminal:
            terminal_count += 1
        if etype == 2 and lid >= 0:
            emitted_lam = float(line_lam[lid])
            b = band_index(emitted_lam)
            legacy = 1290.0 <= emitted_lam < 2000.0
            activation_lid = active.get(packet)
            if activation_lid is None:
                zslot, islot = uz, ui
                missing_count += 1
                missing_energy += float(row["energy"])
            else:
                source_lid = lid if inject_emitted_ion else activation_lid
                zslot = int(line_z[source_lid])
                islot = int(line_ion[source_lid])
            energy = float(row["energy"])
            deep = int(row["shell"]) <= 2
            q = min(quantiles - 1, index * quantiles // n)
            band_slots: list[int] = []
            if b >= 0:
                band_slots.extend((b, 5))
            if legacy:
                band_slots.append(6)
            for scope in (0, 1) if deep else (0,):
                for bs in band_slots:
                    totals[scope, bs, zslot, islot] += energy
                    counts[scope, bs, zslot, islot] += 1
                if legacy:
                    ez, ei = int(line_z[lid]), int(line_ion[lid])
                    emitted_totals[scope, ez, ei] += energy
                    emitted_counts[scope, ez, ei] += 1
            for scope in (0, 1) if deep else (0,):
                for bs in band_slots:
                    qtot[scope, q, bs, zslot, islot] += energy
                    qcount[scope, q, bs, zslot, islot] += 1
                if b >= 0:
                    qenergy[scope, q] += energy
                    qevents[scope, q] += 1
        if terminal:
            active.pop(packet, None)
    return EventTallies(totals, counts, qtot, qcount, qenergy, qevents,
                        emitted_totals, emitted_counts, missing_count,
                        missing_energy, activation_count, terminal_count,
                        len(active))


def consume_events(events: np.ndarray, line_lam: np.ndarray,
                   line_z: np.ndarray, line_ion: np.ndarray,
                   quantiles: int) -> EventTallies:
    """Production entry point.

    Numba is required for the 400-million-record archive.  Parsing remains in
    ``read_events``; compilation only accelerates the causal state machine.
    """
    try:
        from numba import njit
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise AttributionError("numba is required for the real event pass") from exc

    terminal_lookup = np.zeros(256, dtype=np.uint8)
    terminal_lookup[list(TERMINAL_CHANNELS)] = 1
    max_z = int(np.max(line_z))
    max_ion = int(np.max(line_ion))

    @njit
    def kernel(pkt, lid, energy, etype, shell, chan, lam, ztable, itable,
               qn, terminal_lut, mz, mi):
        nz, ni = mz + 2, mi + 2
        uz, ui = mz + 1, mi + 1
        nbands = 7
        totals = np.zeros((2, nbands, nz, ni), np.float64)
        counts = np.zeros((2, nbands, nz, ni), np.int64)
        qtot = np.zeros((2, qn, nbands, nz, ni), np.float64)
        qcount = np.zeros((2, qn, nbands, nz, ni), np.int64)
        qenergy = np.zeros((2, qn), np.float64)
        qevents = np.zeros((2, qn), np.int64)
        emitted_totals = np.zeros((2, nz, ni), np.float64)
        emitted_counts = np.zeros((2, nz, ni), np.int64)
        max_packet = int(np.max(pkt))
        last_activation = np.full(max_packet + 1, -1, np.int32)
        missing_count = 0
        missing_energy = 0.0
        activation_count = 0
        terminal_count = 0
        n = pkt.size
        for index in range(n):
            p = int(pkt[index])
            line = int(lid[index])
            typ = int(etype[index])
            if typ == 1:
                last_activation[p] = line
                activation_count += 1
                continue
            terminal = terminal_lut[int(chan[index])] != 0
            if terminal:
                terminal_count += 1
            if typ == 2 and line >= 0:
                wl = float(lam[line])
                b = -1
                if 600.0 <= wl < 1000.0:
                    b = 0
                elif 1000.0 <= wl < 1500.0:
                    b = 1
                elif 1500.0 <= wl < 2000.0:
                    b = 2
                elif 2000.0 <= wl < 2500.0:
                    b = 3
                elif 2500.0 <= wl <= 3000.0:
                    b = 4
                legacy = 1290.0 <= wl < 2000.0
                actline = int(last_activation[p])
                if actline < 0:
                    zs, ions = uz, ui
                    missing_count += 1
                    missing_energy += float(energy[index])
                else:
                    zs = int(ztable[actline])
                    ions = int(itable[actline])
                en = float(energy[index])
                deep = int(shell[index]) <= 2
                q = index * qn // n
                if q >= qn:
                    q = qn - 1
                for scope in range(2 if deep else 1):
                    if b >= 0:
                        totals[scope, b, zs, ions] += en
                        counts[scope, b, zs, ions] += 1
                        totals[scope, 5, zs, ions] += en
                        counts[scope, 5, zs, ions] += 1
                    if legacy:
                        totals[scope, 6, zs, ions] += en
                        counts[scope, 6, zs, ions] += 1
                        ez = int(ztable[line])
                        ei = int(itable[line])
                        emitted_totals[scope, ez, ei] += en
                        emitted_counts[scope, ez, ei] += 1
                for scope in range(2 if deep else 1):
                    if b >= 0:
                        qtot[scope, q, b, zs, ions] += en
                        qcount[scope, q, b, zs, ions] += 1
                        qtot[scope, q, 5, zs, ions] += en
                        qcount[scope, q, 5, zs, ions] += 1
                        qenergy[scope, q] += en
                        qevents[scope, q] += 1
                    if legacy:
                        qtot[scope, q, 6, zs, ions] += en
                        qcount[scope, q, 6, zs, ions] += 1
            if terminal:
                last_activation[p] = -1
        unpaired = int(np.count_nonzero(last_activation >= 0))
        return (totals, counts, qtot, qcount, qenergy, qevents,
                emitted_totals, emitted_counts, missing_count, missing_energy,
                activation_count, terminal_count, unpaired)

    result = kernel(events["pkt_id"], events["line_id"], events["energy"],
                    events["etype"], events["shell"], events["chan"],
                    line_lam, line_z, line_ion, quantiles, terminal_lookup,
                    max_z, max_ion)
    return EventTallies(*result)


def _iter_ions(values: np.ndarray):
    nz, ni = values.shape
    for zs in range(nz):
        for ions in range(ni):
            if values[zs, ions] != 0.0:
                yield zs, ions


def _decode_slot(zslot: int, ionslot: int, max_z: int,
                 max_ion: int) -> tuple[int, int]:
    if (zslot, ionslot) == (max_z + 1, max_ion + 1):
        return UNATTRIBUTED_Z, UNATTRIBUTED_ION
    return zslot, ionslot


def prefix_audit(t: EventTallies, max_z: int, max_ion: int,
                 quantiles: int, stored_records: int) -> dict[str, Any]:
    """This function is intentionally called before ``crosstab_rows``."""
    rows: list[dict[str, Any]] = []
    drift: list[dict[str, Any]] = []
    for scope_slot, scope in enumerate(("ALL_STORED_SHELLS", "DEEP_S0_S2")):
        for band_slot, (band, lo, hi) in enumerate(BANDS):
            overall_grid = t.totals[scope_slot, band_slot]
            overall_total = float(np.sum(overall_grid, dtype=np.float64))
            keys = set(_iter_ions(overall_grid))
            for q in range(quantiles):
                keys.update(_iter_ions(
                    t.quartile_totals[scope_slot, q, band_slot]))
            for zs, ions in sorted(keys):
                z, ion = _decode_slot(zs, ions, max_z, max_ion)
                shares: list[float | None] = []
                for q in range(quantiles):
                    qgrid = t.quartile_totals[scope_slot, q, band_slot]
                    qden = float(np.sum(qgrid, dtype=np.float64))
                    value = float(qgrid[zs, ions])
                    share = fraction(value, qden)
                    shares.append(share)
                    rows.append({
                        "quantile": q + 1,
                        "record_start_inclusive": q * stored_records // quantiles,
                        "record_stop_exclusive":
                            (q + 1) * stored_records // quantiles,
                        "scope": scope, "band": band,
                        "lambda_lo_A": lo, "lambda_hi_A": hi,
                        "activation_Z": None if z < 0 else z,
                        "activation_ion_number": None if ion < 0 else ion,
                        "activation_ion": ion_label(z, ion),
                        "line_emission_events": int(
                            t.quartile_counts[
                                scope_slot, q, band_slot, zs, ions]),
                        "line_emission_energy": value,
                        "share_of_quantile_band_line_emission_energy": share,
                        "quantile_band_denominator": qden,
                    })
                defined = [x for x in shares if x is not None]
                minimum = min(defined) if defined else None
                maximum = max(defined) if defined else None
                overall_value = float(overall_grid[zs, ions])
                overall_share = fraction(overall_value, overall_total)
                drift.append({
                    "scope": scope, "band": band,
                    "activation_Z": None if z < 0 else z,
                    "activation_ion_number": None if ion < 0 else ion,
                    "activation_ion": ion_label(z, ion),
                    "overall_share": overall_share,
                    "minimum_quantile_share": minimum,
                    "maximum_quantile_share": maximum,
                    "max_minus_min": (
                        None if minimum is None or maximum is None
                        else maximum - minimum),
                    "relative_change_range_over_overall_share": (
                        None if minimum is None or maximum is None
                        or overall_share is None
                        else relative_change(maximum, minimum, overall_share)),
                    "relative_change_max_over_min_minus_one": (
                        None if minimum is None or maximum is None
                        else max_over_min_minus_one(maximum, minimum)),
                })
    ranked = sorted(
        (row for row in drift if row["max_minus_min"] is not None),
        key=lambda row: (-float(row["max_minus_min"]), row["band"],
                         row["activation_ion"]),
    )
    return {
        "status": "MEASURED-NO-PREDECLARED-STABILITY-THRESHOLD",
        "calculation_order": "BEFORE_ION_X_BAND_CROSSTAB",
        "partition": (f"{quantiles} contiguous equal-count ranges of stored records; "
                      "sizes differ by at most one record"),
        "conditioning": "TRUNCATED_PREFIX-not-an-unbiased-random-sample",
        "interpretation": (
            "Shares are prefix-conditional. Movement is reported numerically as "
            "max-minus-min and two explicitly defined relative changes; no stable/"
            "unstable cutoff is imposed after seeing the data."
        ),
        "top_observed_movements": ranked[:25],
        "rows": rows,
        "drift": drift,
    }


def crosstab_rows(t: EventTallies, max_z: int,
                  max_ion: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for scope_slot, scope in enumerate(("ALL_STORED_SHELLS", "DEEP_S0_S2")):
        for band_slot, (band, lo, hi) in enumerate(BANDS):
            grid = t.totals[scope_slot, band_slot]
            denominator = float(np.sum(grid, dtype=np.float64))
            ranked = sorted(_iter_ions(grid), key=lambda key: (
                -float(grid[key]), key[0], key[1]))
            for rank, (zs, ions) in enumerate(ranked, 1):
                z, ion = _decode_slot(zs, ions, max_z, max_ion)
                energy = float(grid[zs, ions])
                output.append({
                    "scope": scope, "band": band,
                    "lambda_lo_A": lo, "lambda_hi_A": hi, "rank": rank,
                    "activation_Z": None if z < 0 else z,
                    "activation_ion_number": None if ion < 0 else ion,
                    "activation_ion": ion_label(z, ion),
                    "line_emission_events": int(t.counts[scope_slot, band_slot, zs, ions]),
                    "line_emission_energy": energy,
                    "fraction_of_scope_band_line_emission_energy":
                        fraction(energy, denominator),
                    "denominator_line_emission_energy": denominator,
                })
    require(bool(output), "empty activation-ion crosstab")
    return output


def legacy_coiv_check(rows: list[dict[str, Any]], tallies: EventTallies) -> dict[str, Any]:
    selected = [row for row in rows
                if row["scope"] == "DEEP_S0_S2"
                and row["band"] == "LEGACY_PILE"
                and row["activation_Z"] == 27
                and row["activation_ion_number"] == 3]
    co_energy = math.fsum(float(row["line_emission_energy"]) for row in selected)
    denom_rows = [row for row in rows
                  if row["scope"] == "DEEP_S0_S2"
                  and row["band"] == "LEGACY_PILE"]
    denominator = (float(denom_rows[0]["denominator_line_emission_energy"])
                   if denom_rows else 0.0)
    share = fraction(co_energy, denominator)
    emitted_denominator = float(np.sum(
        tallies.legacy_emitted_ion_totals[1], dtype=np.float64))
    emitted_co_energy = float(tallies.legacy_emitted_ion_totals[1, 27, 3])
    emitted_share = fraction(emitted_co_energy, emitted_denominator)
    return {
        "claim_under_test": "Co IV = 84% of the 1290-2000 A deep pile emission",
        "old_claim_attribution_axis": "EMITTED_LINE_ION",
        "this_analysis_attribution_axis": "ACTIVATING_LINE_ION",
        "denominator_definition": (
            "sum(EventRec.energy) over stored-prefix records with etype == 2, "
            "valid emitted line_id, emitted-line wavelength 1290 <= lambda_A < "
            "2000, and emission-event shell in 0..2; includes unattributed rows "
            "in the denominator"
        ),
        "numerator_definition": (
            "the denominator subset whose same-packet most recent etype == 1 "
            "activation line has Z == 27 and zero-based ion_number == 3 (Co IV)"
        ),
        "co_iv_activation_energy": co_energy,
        "denominator_line_emission_energy": denominator,
        "co_iv_activation_share": share,
        "activation_share_minus_0p84": None if share is None else share - 0.84,
        "old_axis_emitted_co_iv_energy": emitted_co_energy,
        "old_axis_emitted_line_energy_denominator": emitted_denominator,
        "old_axis_emitted_co_iv_share": emitted_share,
        "old_axis_emitted_share_minus_0p84": (
            None if emitted_share is None else emitted_share - 0.84),
        "old_axis_denominator_definition": (
            "sum(EventRec.energy) over stored-prefix records with etype == 2, "
            "valid emitted line_id, emitted-line wavelength 1290 <= lambda_A < "
            "2000, and emission-event shell in 0..2"
        ),
        "old_axis_numerator_definition": (
            "the old-axis denominator subset whose emitted line has Z == 27 "
            "and zero-based ion_number == 3 (Co IV)"
        ),
        "verdict_rule": (
            "No tolerance is invented: both numeric shares and their signed "
            "differences from 0.84 are reported. The emitted-ion reproduction "
            "control and activating-ion attribution are not conflated."
        ),
    }


def csv_scalar(value: Any) -> Any:
    return "UNDEFINED" if value is None else value


def csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    require(bool(rows), "refusing empty CSV")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    for row in rows:
        writer.writerow({key: csv_scalar(value) for key, value in row.items()})
    return stream.getvalue().encode()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True,
                       allow_nan=False) + "\n").encode()


def aggregate(events: np.ndarray, lines: tuple[np.ndarray, np.ndarray, np.ndarray],
              matrix: Any, quantiles: int, provenance: dict[str, Any],
              prefix_meta: dict[str, Any],
              use_accelerated: bool = True) -> dict[str, Any]:
    require(quantiles >= 4, "prefix audit requires at least four quantiles")
    line_lam, line_z, line_ion = lines
    require(events.size > 0 and line_lam.size > 0, "empty event/line fixture")
    require(line_lam.size == line_z.size == line_ion.size,
            "event line arrays have different lengths")
    require(np.isfinite(line_lam).all() and np.all(line_lam > 0.0),
            "invalid line-table wavelength")
    require(np.all(line_z > 0) and np.all(line_ion >= 0),
            "invalid line-table ion identity")
    line_event = (events["etype"] == 1) | (events["etype"] == 2)
    ids = events["line_id"][line_event]
    require(np.all((ids >= 0) & (ids < line_lam.size)),
            "line absorption/emission has invalid line_id")
    require(np.isfinite(events["energy"]).all() and np.all(events["energy"] >= 0.0),
            "event energy is negative or nonfinite")
    unique_iteration = np.unique(events["iter"])
    require(unique_iteration.size == 1, "event prefix mixes iterations")
    require(prefix_meta.get("status") ==
            "TRUNCATED_PREFIX-not-an-unbiased-random-sample",
            "prefix metadata lacks the required bias status")
    require(int(unique_iteration[0]) == int(prefix_meta["iteration"]),
            "event payload iteration disagrees with prefix metadata")
    consumer = consume_events if use_accelerated else _consume_python
    tallies = consumer(events, line_lam, line_z, line_ion, quantiles)

    # Required ordering: finish prefix-bias audit before constructing crosstab.
    audit = prefix_audit(tallies, int(np.max(line_z)), int(np.max(line_ion)),
                         quantiles, int(events.size))
    crosstab = crosstab_rows(tallies, int(np.max(line_z)), int(np.max(line_ion)))
    legacy = legacy_coiv_check(crosstab, tallies)
    attempted = int(prefix_meta["attempted_records"])
    require(int(prefix_meta["stored_records"]) == int(events.size),
            "prefix metadata stored_records disagrees with event payload")
    stored_fraction = fraction(float(events.size), float(attempted))
    require(stored_fraction == float(prefix_meta["stored_fraction"]),
            "prefix metadata stored_fraction disagrees with exact record ratio")
    result = {
        "schema": SCHEMA, "status": "PASS",
        "provenance": provenance,
        "definitions": {
            "conditioning": (
                "Every event-derived count, energy, and fraction is conditional "
                "on the non-random stored prefix. It is not an unbiased full-pass "
                "estimate."
            ),
            "event_parser": "scripts/read_events.py load_events/load_lines",
            "activation": (
                "most recent earlier etype == 1 line-abs record for the same "
                "pkt_id, cleared at the first recognized terminal channel"
            ),
            "line_emission": "etype == 2 and line_id >= 0",
            "emission_band": "band of lumina_events_lines[emitted line_id].lambda_A",
            "energy": "EventRec.energy (packet comoving energy weight)",
            "B0_B4": (
                "B0=[600,1000), B1=[1000,1500), B2=[1500,2000), "
                "B3=[2000,2500), B4=[2500,3000] Angstrom"
            ),
            "prefix_quantile": (
                "quantile of the emission record's zero-based stored-record index; "
                "contiguous equal-count record ranges"
            ),
            "ion_share": (
                "activating-ion line-emission energy divided by all eligible line-"
                "emission energy in the same scope/band/quantile, including "
                "UNATTRIBUTED in the denominator"
            ),
            "relative_change_range_over_overall_share": (
                "(maximum quantile share - minimum quantile share) / overall "
                "stored-prefix share; null when overall share is zero"
            ),
            "relative_change_max_over_min_minus_one": (
                "maximum quantile share / minimum quantile share - 1; null when "
                "minimum share is zero"
            ),
            "undefined": "null in JSON and UNDEFINED in CSV when denominator is zero",
            "clamp": 0, "floor": 0, "cap": 0, "fallback": 0,
            "substitution": 0,
        },
        "archive": {
            "status": "TRUNCATED_PREFIX-not-an-unbiased-random-sample",
            "stored_records": int(events.size),
            "attempted_event_log_records": attempted,
            "stored_fraction_of_attempted_event_log_records": stored_fraction,
            "matrix_full_pass_line_interactions":
                int(matrix.header["events_total"]),
            "matrix_events_semantics": (
                "full-pass line interactions recorded by LFMAT001; not the "
                "attempted event-log-record denominator"
            ),
            "event_iteration": int(unique_iteration[0]),
            "matrix_iteration": int(matrix.header["iteration"]),
            "cross_generation_direct_comparison": False,
            "cross_generation_reason": (
                "the event archive is iteration 11 while the independently "
                "verified full LFMAT001 reference is iteration 10"
                if int(unique_iteration[0]) != int(matrix.header["iteration"])
                else "event and matrix iterations match"
            ),
        },
        "prefix_bias_audit": audit,
        "ion_band_crosstab": crosstab,
        "legacy_co_iv_84_percent_check": legacy,
        "pairing_diagnostics": {
            "activation_records": tallies.activation_count,
            "recognized_terminal_records": tallies.terminal_count,
            "line_emissions_without_activation": tallies.emission_without_activation,
            "line_emission_energy_without_activation":
                tallies.emission_without_activation_energy,
            "activations_unpaired_at_stored_prefix_tail": tallies.unpaired_prefix_tail,
        },
    }
    return result


def payloads(result: dict[str, Any]) -> dict[str, bytes]:
    audit = result["prefix_bias_audit"]
    return {
        "pile_ion_attribution.json": json_bytes(result),
        "pile_ion_band.csv": csv_bytes(result["ion_band_crosstab"]),
        "pile_prefix_quantile_ion_share.csv": csv_bytes(audit["rows"]),
        "pile_prefix_ion_drift.csv": csv_bytes(audit["drift"]),
    }


def _write_event_fixture(directory: Path) -> tuple[Path, Path]:
    lines = np.asarray([
        (1400.0, 27, 3),  # Co IV activation
        (1400.0, 26, 2),  # Fe III activation
        (1400.0, 28, 3),  # Ni IV emitted line: deliberately different
    ], dtype=event_reader.LINE_DTYPE)
    events = np.zeros(32, dtype=event_reader.EVENT_DTYPE)
    events["etype"] = 7
    events["chan"] = 0x42
    events["iter"] = 10
    energies = ((8.0, 2.0), (6.0, 4.0), (4.0, 6.0), (2.0, 8.0))
    packet = 0
    for q, pair in enumerate(energies):
        start = q * 8
        for j, (activation_lid, energy) in enumerate(zip((0, 1), pair)):
            ai = start + 2 * j
            ei = ai + 1
            events[ai] = (packet, activation_lid, 2.1e15, energy, 1, 1, 10, 0x30)
            events[ei] = (packet, 2, 2.1e15, energy, 2, 1, 10, 0x38)
            packet += 1
    event_path = directory / "lumina_events.bin"
    line_path = directory / "lumina_events_lines.bin"
    event_path.write_bytes(b"LUMEVT01" + struct.pack("<I", 20) + bytes(20)
                           + events.tobytes())
    line_path.write_bytes(b"LUMLIN01" + lines.tobytes())
    for path in (event_path, line_path):
        digest = sha256_file(path)
        Path(str(path) + ".sha256").write_text(f"{digest}  {path.name}\n")
    return event_path, line_path


def _write_matrix_fixture(directory: Path) -> Path:
    path = directory / "fluor_matrix_iter10.iter010"
    input_count = np.asarray([8, 8, 8, 4, 4], dtype=np.uint64)
    terminal = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    write_fixture_matrix(
        path, nb=5, ns=3, iteration=10, numin=1.0, numax=32.0,
        input_count=input_count, input_energy=terminal,
        terminal_energy=terminal, outside_energy=np.zeros(5),
        shell_count=np.asarray([10, 11, 11], dtype=np.uint64),
        shell_kpacket_count=np.zeros(3, dtype=np.uint64),
        shell_absorbed=np.asarray([1.0, 2.0, 3.0]),
        shell_reemitted=np.asarray([1.0, 2.0, 3.0]),
        edges=[(i, i, float(terminal[i])) for i in range(5)],
    )
    return path


def read_prefix_metadata(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
        meta = value["event_archive"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise AttributionError(f"invalid prefix metadata: {path}") from exc
    for key, expected in (
            ("status", "TRUNCATED_PREFIX-not-an-unbiased-random-sample"),
            ("iteration", 11),
            ("attempted_records", 970557187),
            ("stored_records", 400000000)):
        require(meta.get(key) == expected,
                f"prefix metadata {key} mismatch: {meta.get(key)!r}")
    exact_fraction = float(meta["stored_records"] / meta["attempted_records"])
    require(float(meta.get("stored_fraction")) == exact_fraction,
            "prefix metadata stored_fraction arithmetic mismatch")
    return dict(meta)


def self_test() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="pile-ion-fixture-") as tmp:
        directory = Path(tmp)
        event_path, line_path = _write_event_fixture(directory)
        matrix_path = _write_matrix_fixture(directory)
        verified_probe = sha256_provenance(event_path)
        # Exercise the real capture's two no-sidecar event inputs.  Their
        # digests must remain explicit computed-now evidence, never masquerade
        # as sidecar verified.
        Path(str(event_path) + ".sha256").unlink()
        Path(str(line_path) + ".sha256").unlink()
        event_provenance = sha256_provenance(event_path)
        line_provenance = sha256_provenance(line_path)
        events = event_reader.load_events(event_path)
        lines = event_reader.load_lines(line_path)
        matrix = read_fluor_matrix(matrix_path, expected_iteration=10)
        matrix_provenance = {
            "path": str(matrix_path),
            "sha256": matrix.sha256,
            "sha256_source": "sidecar_verified",
            "sha256_sidecar_path": str(Path(str(matrix_path) + ".sha256")),
            "sha256_evidence": "payload matched pre-existing sidecar digest",
            "schema": matrix.header["schema"],
            "contract_status": matrix.contract_status,
        }
        provenance = {
            "events": event_provenance,
            "lines": line_provenance,
            "matrix": matrix_provenance,
            "fixture": True,
        }
        require(verified_probe["sha256_source"] == "sidecar_verified"
                and event_provenance["sha256_source"] == "computed_now"
                and line_provenance["sha256_source"] == "computed_now"
                and matrix_provenance["sha256_source"] == "sidecar_verified",
                "fixture did not preserve distinct SHA-256 evidence sources")

        sidecar_negative_output = ""
        mismatch_path = directory / "mismatching_sidecar_payload.bin"
        mismatch_path.write_bytes(event_path.read_bytes())
        Path(str(mismatch_path) + ".sha256").write_text(
            f"{'0' * 64}  {mismatch_path.name}\n")
        try:
            sha256_provenance(mismatch_path)
        except AttributionError as exc:
            sidecar_negative_output = f"FAIL (expected): {exc}"
        require(bool(sidecar_negative_output),
                "mismatching SHA-256 sidecar was not detected")
        prefix_meta = {
            "status": "TRUNCATED_PREFIX-not-an-unbiased-random-sample",
            "iteration": 10,
            "attempted_records": 64, "stored_records": 32,
            "stored_fraction": 0.5,
        }
        first_result = aggregate(events, lines, matrix, 4, provenance, prefix_meta,
                                 use_accelerated=False)
        second_result = aggregate(events, lines, matrix, 4, provenance, prefix_meta,
                                  use_accelerated=False)
        first = payloads(first_result)
        second = payloads(second_result)
        require(first == second, "fixture payload repeat is not byte-identical")
        co = [row for row in first_result["prefix_bias_audit"]["drift"]
              if row["scope"] == "DEEP_S0_S2" and row["band"] == "BALL"
              and row["activation_ion"] == "Co IV"]
        require(len(co) == 1 and math.isclose(
                    float(co[0]["max_minus_min"]), 0.6,
                    rel_tol=0.0, abs_tol=2.0e-16),
                "fixture did not recover the seeded Co IV prefix drift")

        # Negative control: attribute by emitted-line ion instead of activating ion.
        defective = _consume_python(events, *lines, 4, inject_emitted_ion=True)
        negative_output = ""
        try:
            bad_audit = prefix_audit(defective, int(np.max(lines[1])),
                                     int(np.max(lines[2])), 4, len(events))
            bad_co = [row for row in bad_audit["drift"]
                      if row["scope"] == "DEEP_S0_S2"
                      and row["band"] == "BALL"
                      and row["activation_ion"] == "Co IV"]
            require(len(bad_co) == 1 and math.isclose(
                        float(bad_co[0]["max_minus_min"]), 0.6,
                        rel_tol=0.0, abs_tol=2.0e-16),
                    "INJECTED-DEFECT emitted-ion attribution failed fixture oracle")
        except AttributionError as exc:
            negative_output = f"FAIL (expected): {exc}"
        require(bool(negative_output), "injected emitted-ion defect was not detected")
        return {
            "status": "PASS",
            "fixture_only": True,
            "reader_reuse": "scripts/read_events.py load_events/load_lines",
            "sha256_source_coverage": {
                "sidecar_verified": 1,
                "computed_now": 2,
            },
            "optional_sidecar_branch_verified": True,
            "repeat_payloads_byte_identical": True,
            "expected_co_iv_ball_quantile_shares": [0.8, 0.6, 0.4, 0.2],
            "expected_co_iv_max_minus_min": 0.6,
            "negative_control": {
                "injection": "use emitted-line ion instead of activating-line ion",
                "observed": negative_output,
            },
            "sha256_mismatch_negative_control": {
                "injection": "replace an existing sidecar digest with zeros",
                "observed": sidecar_negative_output,
            },
            "clamp": 0, "floor": 0, "cap": 0, "fallback": 0,
            "substitution": 0,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--lines", type=Path)
    parser.add_argument("--matrix", type=Path)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--quantiles", type=int, default=4)
    parser.add_argument("--expected-matrix-sha256")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            result = self_test()
            print(result["negative_control"]["observed"])
            print(result["sha256_mismatch_negative_control"]["observed"])
            print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
            return 0
        require(args.outdir is not None, "--outdir is required")
        capture = args.capture_dir.resolve()
        events_path = (args.events or capture / "lumina_events.bin").resolve()
        lines_path = (args.lines or capture / "lumina_events_lines.bin").resolve()
        matrix_path = (args.matrix or capture /
                       "fluor_matrix_iter10.iter010").resolve()
        event_provenance = sha256_provenance(events_path)
        line_provenance = sha256_provenance(lines_path)
        # The matrix reader independently verifies its mandatory sidecar.
        matrix = read_fluor_matrix(
            matrix_path, expected_iteration=10,
            expected_sha256=args.expected_matrix_sha256)
        events = event_reader.load_events(events_path)
        lines = event_reader.load_lines(lines_path)
        provenance = {
            "events": event_provenance,
            "lines": line_provenance,
            "matrix": {
                "path": str(matrix_path),
                "sha256": matrix.sha256,
                "sha256_source": "sidecar_verified",
                "sha256_sidecar_path":
                    str(Path(str(matrix_path) + ".sha256")),
                "sha256_evidence":
                    "payload matched pre-existing sidecar digest",
                "schema": matrix.header["schema"],
                "contract_status": matrix.contract_status,
            },
        }
        prefix_summary = ROOT / "validation/emiss_e9/redistribution_summary.json"
        prefix_meta = read_prefix_metadata(prefix_summary)
        provenance["prefix_metadata_path"] = str(prefix_summary.resolve())
        first_result = aggregate(events, lines, matrix, args.quantiles, provenance,
                                 prefix_meta)
        first = payloads(first_result)
        outdir = args.outdir.resolve()
        outdir.mkdir(parents=True, exist_ok=True)
        for name, content in first.items():
            (outdir / name).write_bytes(content)
        print(json.dumps({
            "schema": SCHEMA, "status": "PASS", "outdir": str(outdir),
            "provenance": first_result["provenance"],
            "archive": first_result["archive"],
            "legacy_co_iv_84_percent_check":
                first_result["legacy_co_iv_84_percent_check"],
            "negative_control": "covered by --self-test",
        }, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (AttributionError, FluorMatrixError, OSError, ValueError,
            KeyError, TypeError) as exc:
        print(f"UNRESOLVED-FAIL-CLOSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
