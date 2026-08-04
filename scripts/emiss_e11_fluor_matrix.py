#!/usr/bin/env python3
"""Read, validate, inspect, and fixture-write LFMAT001 v1 matrices.

Production writes one uncapped, in-transport matrix per MC pass.  The device
accumulator is dense, while this file stores only nonzero (input, output)
edges.  No normalization, clamp, missing-edge fill, or identity fallback is
performed here.

GENERATION CONTRACT (N4).  A SHA-256 sidecar that is recomputed together with
its payload proves integrity, not identity: the production writer used to
overwrite one path every MC pass and refresh the sidecar with it, so
`sha256sum -c` passed while E12/E13 silently consumed a different matrix
(header iteration 11 under a file named ...iter10).  `read_fluor_matrix`
therefore REQUIRES an explicit `expected_iteration`; the campaign contract is
iteration 10 and any other expectation must pass `non_contract_override`,
exactly like `cmf_chieta_check.check_artifact`.  `expected_sha256` additionally
pins the byte identity independently of the sidecar.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any

import numpy as np


MAGIC = b"LFMAT001"
ENDIAN = 0x01020304
VERSION = 1
FLAGS = 0xF  # sparse, global, per-shell ledgers, representative shell groups
HEADER = struct.Struct("<8s7I3d7Q4dQ")
EDGE_DTYPE = np.dtype([("input_bin", "<u4"), ("output_bin", "<u4"),
                       ("output_energy", "<f8")])
CONTRACT_ITERATION = 10
NON_CONTRACT_RC = 2
CONTRACT_STATUS = "CONTRACT"
NON_CONTRACT_STATUS = "NON-CONTRACT"


class FluorMatrixError(RuntimeError):
    pass


@dataclass(frozen=True)
class FluorMatrix:
    path: Path
    header: dict[str, Any]
    input_count: np.ndarray
    input_energy: np.ndarray
    terminal_energy: np.ndarray
    outside_energy: np.ndarray
    shell_count: np.ndarray
    shell_kpacket_count: np.ndarray
    shell_absorbed_energy: np.ndarray
    shell_reemitted_energy: np.ndarray
    edges: np.ndarray
    group_ranges: tuple[tuple[int, int], ...]
    group_edges: tuple[np.ndarray, ...]
    sha256: str
    column_closure_max_abs: float
    contract_status: str = CONTRACT_STATUS


def _take(raw: bytes, offset: int, dtype: str, count: int) -> tuple[np.ndarray, int]:
    dt = np.dtype(dtype)
    end = offset + count * dt.itemsize
    if end > len(raw):
        raise FluorMatrixError("truncated LFMAT001 array")
    return np.frombuffer(raw, dtype=dt, count=count, offset=offset).copy(), end


def _verify_sha(path: Path, raw: bytes) -> str:
    sidecar = Path(str(path) + ".sha256")
    try:
        fields = sidecar.read_text().split()
    except OSError as exc:
        raise FluorMatrixError(f"missing SHA-256 sidecar: {sidecar}") from exc
    if not fields or len(fields[0]) != 64:
        raise FluorMatrixError("malformed SHA-256 sidecar")
    measured = hashlib.sha256(raw).hexdigest()
    if fields[0].lower() != measured:
        raise FluorMatrixError("SHA-256 sidecar mismatch")
    return measured


def read_fluor_matrix(path: Path, *, expected_iteration: int,
                       expected_sha256: str | None = None,
                       non_contract_override: bool = False,
                       verify_column_closure: bool = True,
                       closure_tolerance: float = 2.0e-12) -> FluorMatrix:
    """Read one LFMAT001 matrix under an explicit generation contract.

    `expected_iteration` is keyword-only and has NO default: a consumer that
    forgets it raises TypeError instead of silently accepting whatever
    generation happens to be on disk.
    """
    deviates_from_contract = expected_iteration != CONTRACT_ITERATION
    if deviates_from_contract and not non_contract_override:
        raise FluorMatrixError(
            "non-contract matrix expectation requires non_contract_override "
            f"(expected_iteration={expected_iteration}, "
            f"contract={CONTRACT_ITERATION})")
    path = path.resolve()
    raw = path.read_bytes()
    digest = _verify_sha(path, raw)
    if expected_sha256 is not None and digest != expected_sha256.lower():
        raise FluorMatrixError(
            f"matrix identity changed: expected sha256={expected_sha256}, "
            f"measured={digest}")
    if len(raw) < HEADER.size:
        raise FluorMatrixError("truncated LFMAT001 header")
    values = HEADER.unpack_from(raw)
    (magic, endian, version, flags, nb, ns, iteration, ngroups,
     numin, numax, dlognu, events, classified, unclassified_input,
     unclassified_output, unclassified_energy, unclassified_route,
     kpacket_events, absorbed,
     reemitted, kpacket_absorbed, kpacket_reemitted, nnz) = values
    if (magic, endian, version, flags) != (MAGIC, ENDIAN, VERSION, FLAGS):
        raise FluorMatrixError("bad LFMAT001 magic/endian/version/flags")
    if not (0 < nb <= 4096 and 0 < ns <= 4096 and numin > 0.0 and
            numax > numin and dlognu > 0.0 and nnz <= nb * nb and
            0 < ngroups <= 3):
        raise FluorMatrixError("invalid LFMAT001 dimensions/grid/nnz")
    if int(iteration) != expected_iteration:
        raise FluorMatrixError(
            f"matrix generation mismatch: header iteration={int(iteration)}, "
            f"expected {expected_iteration} (the SHA-256 sidecar is "
            "recomputed with the payload and cannot detect this)")
    offset = HEADER.size
    input_count, offset = _take(raw, offset, "<u8", nb)
    input_energy, offset = _take(raw, offset, "<f8", nb)
    terminal_energy, offset = _take(raw, offset, "<f8", nb)
    outside_energy, offset = _take(raw, offset, "<f8", nb)
    shell_count, offset = _take(raw, offset, "<u8", ns)
    shell_kpacket_count, offset = _take(raw, offset, "<u8", ns)
    shell_absorbed, offset = _take(raw, offset, "<f8", ns)
    shell_reemitted, offset = _take(raw, offset, "<f8", ns)
    edge_end = offset + int(nnz) * EDGE_DTYPE.itemsize
    if edge_end > len(raw):
        raise FluorMatrixError("LFMAT001 global edge section truncated")
    edges = np.frombuffer(raw, dtype=EDGE_DTYPE, count=int(nnz), offset=offset).copy()
    offset = edge_end
    group_ranges: list[tuple[int, int]] = []
    group_edges: list[np.ndarray] = []
    group_header = struct.Struct("<IIQ")
    for group in range(ngroups):
        if offset + group_header.size > len(raw):
            raise FluorMatrixError("LFMAT001 group header truncated")
        first, last, group_nnz = group_header.unpack_from(raw, offset)
        offset += group_header.size
        if not (0 <= first <= last < ns and group_nnz <= nb * nb):
            raise FluorMatrixError(f"invalid LFMAT001 shell group {group}")
        end = offset + int(group_nnz) * EDGE_DTYPE.itemsize
        if end > len(raw):
            raise FluorMatrixError("LFMAT001 group edge section truncated")
        group_ranges.append((int(first), int(last)))
        group_edges.append(np.frombuffer(raw, dtype=EDGE_DTYPE,
                                         count=int(group_nnz), offset=offset).copy())
        offset = end
    if offset != len(raw):
        raise FluorMatrixError("LFMAT001 trailing bytes")
    finite_arrays = (input_energy, terminal_energy, outside_energy,
                     shell_absorbed, shell_reemitted, edges["output_energy"])
    if any(not np.isfinite(x).all() for x in finite_arrays):
        raise FluorMatrixError("nonfinite LFMAT001 energy")
    if any(np.any(x < 0.0) for x in finite_arrays):
        raise FluorMatrixError("negative LFMAT001 energy")
    if (np.any(edges["input_bin"] >= nb) or
            np.any(edges["output_bin"] >= nb)):
        raise FluorMatrixError("edge outside LFMAT001 grid")
    pairs = (edges["input_bin"].astype(np.uint64) * np.uint64(nb) +
             edges["output_bin"].astype(np.uint64))
    if np.unique(pairs).size != pairs.size:
        raise FluorMatrixError("duplicate LFMAT001 edge")
    group_dense_sum = np.zeros((nb, nb), dtype=np.float64)
    for group, group_rows in enumerate(group_edges):
        if (np.any(group_rows["input_bin"] >= nb) or
                np.any(group_rows["output_bin"] >= nb) or
                not np.isfinite(group_rows["output_energy"]).all() or
                np.any(group_rows["output_energy"] < 0.0)):
            raise FluorMatrixError(f"invalid LFMAT001 group edge {group}")
        gpairs = (group_rows["input_bin"].astype(np.uint64) * np.uint64(nb) +
                  group_rows["output_bin"].astype(np.uint64))
        if np.unique(gpairs).size != gpairs.size:
            raise FluorMatrixError(f"duplicate LFMAT001 group edge {group}")
        np.add.at(group_dense_sum,
                  (group_rows["input_bin"], group_rows["output_bin"]),
                  group_rows["output_energy"])
    global_dense = np.zeros((nb, nb), dtype=np.float64)
    np.add.at(global_dense, (edges["input_bin"], edges["output_bin"]),
              edges["output_energy"])
    group_delta = float(np.max(np.abs(group_dense_sum - global_dense)))
    group_scale = max(float(np.max(np.abs(global_dense))), 1.0)
    if group_delta / group_scale > closure_tolerance:
        raise FluorMatrixError("representative shell groups do not sum to global matrix")
    on_grid = np.zeros(nb, dtype=np.float64)
    np.add.at(on_grid, edges["input_bin"], edges["output_energy"])
    column_delta = on_grid + outside_energy - terminal_energy
    scale = np.maximum(np.abs(terminal_energy), 1.0)
    column_rel = np.abs(column_delta) / scale
    column_max = float(np.max(column_rel))
    if verify_column_closure and column_max > closure_tolerance:
        ib = int(np.argmax(column_rel))
        raise FluorMatrixError(
            f"column energy ledger mismatch input_bin={ib}: "
            f"matrix+outside={on_grid[ib] + outside_energy[ib]:.17g} "
            f"terminal={terminal_energy[ib]:.17g} rel={column_rel[ib]:.17g}")
    header = {
        "schema": "LFMAT001-v1", "flags": int(flags), "n_bins": int(nb),
        "n_shells": int(ns), "iteration": int(iteration),
        "n_shell_groups": int(ngroups),
        "nu_min": numin, "nu_max": numax, "d_log_nu": dlognu,
        "events_total": int(events), "classified_events": int(classified),
        "unclassified_input": int(unclassified_input),
        "unclassified_output": int(unclassified_output),
        "unclassified_energy": int(unclassified_energy),
        "unclassified_route": int(unclassified_route),
        "kpacket_events": int(kpacket_events),
        "absorbed_energy": absorbed, "reemitted_energy": reemitted,
        "energy_conservation_relative_error": (
            reemitted / absorbed - 1.0 if absorbed else 0.0),
        "kpacket_absorbed_energy": kpacket_absorbed,
        "kpacket_reemitted_energy": kpacket_reemitted,
        "kpacket_energy_relative_error": (
            kpacket_reemitted / kpacket_absorbed - 1.0
            if kpacket_absorbed else 0.0),
        "sparse_nonzero_edges": int(nnz),
    }
    return FluorMatrix(path, header, input_count, input_energy,
                       terminal_energy, outside_energy, shell_count,
                       shell_kpacket_count, shell_absorbed, shell_reemitted,
                       edges, tuple(group_ranges), tuple(group_edges), digest,
                       column_max,
                       NON_CONTRACT_STATUS if deviates_from_contract
                       else CONTRACT_STATUS)


def write_fixture_matrix(path: Path, *, nb: int, ns: int, iteration: int,
                         numin: float, numax: float,
                         input_count: np.ndarray, input_energy: np.ndarray,
                         terminal_energy: np.ndarray, outside_energy: np.ndarray,
                         shell_count: np.ndarray, shell_kpacket_count: np.ndarray,
                         shell_absorbed: np.ndarray, shell_reemitted: np.ndarray,
                         edges: list[tuple[int, int, float]],
                         kpacket_events: int = 0,
                         kpacket_absorbed: float = 0.0,
                         kpacket_reemitted: float = 0.0) -> None:
    """Fixture-only canonical writer; production uses the C/CUDA writer."""
    dlognu = math.log(numax / numin) / nb
    edge_array = np.asarray(edges, dtype=EDGE_DTYPE)
    if ns <= 5:
        group_ranges = [(0, ns - 1)]
    elif ns <= 13:
        group_ranges = [(0, 4), (5, ns - 1)]
    else:
        group_ranges = [(0, 4), (5, 12), (13, ns - 1)]
    events = int(np.sum(input_count))
    classified = events  # fixtures below have no outside/unclassified events
    absorbed = float(np.sum(shell_absorbed))
    reemitted = float(np.sum(shell_reemitted))
    header = HEADER.pack(
        MAGIC, ENDIAN, VERSION, FLAGS, nb, ns, iteration, len(group_ranges),
        numin, numax, dlognu, events, classified, 0, 0, 0, 0,
        kpacket_events, absorbed, reemitted, kpacket_absorbed,
        kpacket_reemitted, len(edge_array))
    arrays = (np.asarray(input_count, dtype="<u8"),
              np.asarray(input_energy, dtype="<f8"),
              np.asarray(terminal_energy, dtype="<f8"),
              np.asarray(outside_energy, dtype="<f8"),
              np.asarray(shell_count, dtype="<u8"),
              np.asarray(shell_kpacket_count, dtype="<u8"),
              np.asarray(shell_absorbed, dtype="<f8"),
              np.asarray(shell_reemitted, dtype="<f8"), edge_array)
    pieces = [header, *(x.tobytes(order="C") for x in arrays)]
    group_header = struct.Struct("<IIQ")
    for group, (first, last) in enumerate(group_ranges):
        rows = edge_array if group == 0 else np.empty(0, dtype=EDGE_DTYPE)
        pieces.extend((group_header.pack(first, last, len(rows)),
                       rows.tobytes(order="C")))
    raw = b"".join(pieces)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    Path(str(path) + ".sha256").write_text(f"{digest}  {path}\n")


def add_matrix_contract_args(parser: argparse.ArgumentParser) -> None:
    """Attach the shared LFMAT001 generation-contract flags to a consumer."""
    parser.add_argument(
        "--expected-matrix-iteration", type=int, default=CONTRACT_ITERATION,
        help=f"consumer contract is {CONTRACT_ITERATION}; other values "
             "require --matrix-non-contract-override")
    parser.add_argument("--expected-matrix-sha256", type=str, default=None,
                        help="pin the matrix byte identity independently of "
                             "the co-updated .sha256 sidecar")
    parser.add_argument("--matrix-non-contract-override", action="store_true",
                        help="accept a non-contract matrix generation")


def read_fluor_matrix_from_args(path: Path, args: argparse.Namespace,
                                **kwargs) -> FluorMatrix:
    return read_fluor_matrix(
        path,
        expected_iteration=args.expected_matrix_iteration,
        expected_sha256=args.expected_matrix_sha256,
        non_contract_override=args.matrix_non_contract_override,
        **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path)
    add_matrix_contract_args(parser)
    args = parser.parse_args()
    try:
        matrix = read_fluor_matrix_from_args(args.matrix, args)
        print(json.dumps({**matrix.header, "sha256": matrix.sha256,
                          "column_closure_max_abs": matrix.column_closure_max_abs,
                          "contract_status": matrix.contract_status,
                          "clamp": 0, "fallback": 0},
                         indent=2, sort_keys=True, allow_nan=False))
        return (NON_CONTRACT_RC
                if matrix.contract_status == NON_CONTRACT_STATUS else 0)
    except (OSError, ValueError, FluorMatrixError) as exc:
        print(f"FAIL: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
