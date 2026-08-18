#!/usr/bin/env python3
"""Extract one finite CMFGEN ETAL_MAT*ZNET known-answer cell.

CMFGEN's final-iteration LINEHEAT and NETRATE files are multi-gigabyte text
ledgers.  This reader streams them: LINEHEAT supplies the scaled line energy
vector and NETRATE supplies ZNET for the same line.  The selected cell must
have finite nonzero ETAL_MAT*ZNET, finite nonzero ZNET, and positive derived
ETAL_MAT.  No near-zero or missing value is promoted to a fixture.

Only the numbered Sobolev records written by cmfgen_sub.f:2739-2762 are
eligible.  Text-only/dielectronic records and cumulative STEQ_T vectors are
ignored.  CMFGEN depth indices in the output are one-based.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
from pathlib import Path
from typing import Iterator, NamedTuple


SCHEMA = "lumina-cmfgen-line-net-known-answer-v2"
CMFGEN_RE_INTERNAL_TO_CGS = 4.0 * math.pi * 1.0e-10


class Header(NamedTuple):
    line_id: int
    transition: str
    frequency_field: float
    lower: int
    upper: int
    scale: float


class NetHeader(NamedTuple):
    line_id: int
    transition: str
    frequency_field: float
    lower: int
    upper: int
    lower_full: int
    upper_full: int


def number(text: str) -> float:
    return float(text.replace("D", "E").replace("d", "e"))


def lineheat_header(line: str) -> Header | None:
    fields = line.split()
    if len(fields) < 6 or not fields[0].isdigit():
        return None
    try:
        line_id = int(fields[0])
        frequency = number(fields[-4])
        lower = int(fields[-3])
        upper = int(fields[-2])
        scale = number(fields[-1])
    except ValueError:
        return None
    transition = " ".join(fields[1:-4])
    if not transition or not math.isfinite(frequency) or not math.isfinite(scale):
        return None
    return Header(line_id, transition, frequency, lower, upper, scale)


def netrate_header(line: str) -> NetHeader | None:
    fields = line.split()
    # The transition label has no whitespace in the current ledger, so the
    # minimal valid record has seven fields: id, label, frequency, and four
    # level indices.  Longer labels remain valid because parsing is from the
    # right-hand numeric tail.
    if len(fields) < 7 or not fields[0].isdigit():
        return None
    try:
        line_id = int(fields[0])
        frequency = number(fields[-5])
        lower = int(fields[-4])
        upper = int(fields[-3])
        lower_full = int(fields[-2])
        upper_full = int(fields[-1])
    except ValueError:
        return None
    transition = " ".join(fields[1:-5])
    if not transition or not math.isfinite(frequency):
        return None
    return NetHeader(
        line_id, transition, frequency, lower, upper, lower_full, upper_full
    )


def vectors(
    path: Path,
    depth_count: int,
    parser,
    selected: set[int] | None = None,
) -> Iterator[tuple[Header | NetHeader, list[float]]]:
    """Yield the first depth vector following each eligible numbered header."""
    active: Header | NetHeader | None = None
    values: list[float] = []
    with path.open("rt", encoding="ascii", errors="strict", buffering=4 << 20) as stream:
        for line_number, line in enumerate(stream, 1):
            header = parser(line)
            if header is not None:
                if active is not None and len(values) != depth_count:
                    raise ValueError(
                        f"{path}:{line_number}: new header before {depth_count} values"
                    )
                active = header
                values = []
                continue
            if active is None:
                continue
            for token in line.split():
                try:
                    value = number(token)
                except ValueError as exc:
                    raise ValueError(
                        f"{path}:{line_number}: nonnumeric vector token {token!r}"
                    ) from exc
                values.append(value)
                if len(values) == depth_count:
                    if selected is None or active.line_id in selected:
                        yield active, values
                    active = None
                    values = []
                    break
                if len(values) > depth_count:
                    raise ValueError(
                        f"{path}:{line_number}: vector exceeds {depth_count} values"
                    )
    if active is not None:
        raise ValueError(f"{path}: truncated vector with {len(values)} values")


def source_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def scan_lineheat(path: Path, depth_count: int, keep: int) -> list[dict]:
    heap: list[tuple[float, int, int, dict]] = []
    sequence = 0
    for header, values in vectors(path, depth_count, lineheat_header):
        assert isinstance(header, Header)
        if header.scale == 0.0:
            continue
        for depth_zero, scaled in enumerate(values):
            raw = scaled / header.scale
            if not math.isfinite(raw) or raw == 0.0:
                continue
            row = {
                "line_id": header.line_id,
                "transition": header.transition,
                "frequency_field": header.frequency_field,
                "lower": header.lower,
                "upper": header.upper,
                "scale_factor": header.scale,
                "depth_index": depth_zero + 1,
                "scaled_lineheat": scaled,
                "q_line": raw,
            }
            item = (abs(raw), sequence, header.line_id, row)
            sequence += 1
            if len(heap) < keep:
                heapq.heappush(heap, item)
            elif item[0] > heap[0][0]:
                heapq.heapreplace(heap, item)
    return [item[3] for item in sorted(heap, reverse=True)]


def scan_netrate(
    path: Path, depth_count: int, selected: set[int]
) -> dict[int, tuple[NetHeader, list[float]]]:
    result: dict[int, tuple[NetHeader, list[float]]] = {}
    for header, values in vectors(path, depth_count, netrate_header, selected):
        assert isinstance(header, NetHeader)
        previous = result.get(header.line_id)
        if previous is not None and previous != (header, values):
            raise ValueError(f"duplicate inconsistent NETRATE line {header.line_id}")
        result[header.line_id] = (header, values)
    missing = selected - result.keys()
    if missing:
        preview = sorted(missing)[:10]
        raise ValueError(f"NETRATE missing {len(missing)} selected lines: {preview}")
    return result


def choose(candidates: list[dict], net: dict[int, tuple[NetHeader, list[float]]]) -> dict:
    for row in candidates:
        header, znet_vector = net[row["line_id"]]
        depth = row["depth_index"] - 1
        znet = znet_vector[depth]
        if not math.isfinite(znet) or znet == 0.0:
            continue
        etal = row["q_line"] / znet
        if not math.isfinite(etal) or etal <= 0.0:
            continue
        if row["lower"] != header.lower or row["upper"] != header.upper:
            raise ValueError(f"LINEHEAT/NETRATE level mismatch line {row['line_id']}")
        if row["transition"] != header.transition:
            raise ValueError(f"LINEHEAT/NETRATE label mismatch line {row['line_id']}")
        frequency_delta = abs(row["frequency_field"] - header.frequency_field)
        frequency_scale = max(abs(row["frequency_field"]), 1.0)
        if frequency_delta > 1.0e-9 * frequency_scale:
            raise ValueError(f"LINEHEAT/NETRATE frequency mismatch line {row['line_id']}")
        return {
            **row,
            "q_line_internal_unscaled": row["q_line"],
            "q_line_internal_scaled": row["scaled_lineheat"],
            "q_line_cgs_unscaled": row["q_line"] * CMFGEN_RE_INTERNAL_TO_CGS,
            "q_line_cgs_scaled": row["scaled_lineheat"] * CMFGEN_RE_INTERNAL_TO_CGS,
            "znet": znet,
            "etal_mat_derived": etal,
            "etal_mat_internal_derived": etal,
            "integrated_emissivity_cgs_per_sr_derived": etal * 1.0e-10,
            "lower_full": header.lower_full,
            "upper_full": header.upper_full,
            "identity_checks": {
                "transition": "MATCH",
                "frequency_field": "MATCH",
                "lower_upper": "MATCH",
            },
        }
    raise ValueError("no candidate has finite nonzero ZNET and positive ETAL_MAT")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lineheat", type=Path, required=True)
    parser.add_argument("--netrate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--depth-count", type=int, default=90)
    parser.add_argument("--candidate-count", type=int, default=2048)
    parser.add_argument("--hash-sources", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.depth_count <= 0 or args.candidate_count <= 0:
        raise SystemExit("depth-count and candidate-count must be positive")
    for path in (args.lineheat, args.netrate):
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"missing or empty input: {path}")

    candidates = scan_lineheat(args.lineheat, args.depth_count, args.candidate_count)
    if not candidates:
        raise SystemExit("LINEHEAT yielded no finite nonzero candidate")
    selected_ids = {row["line_id"] for row in candidates}
    net = scan_netrate(args.netrate, args.depth_count, selected_ids)
    witness = choose(candidates, net)

    sources = {}
    for label, path in (("LINEHEAT", args.lineheat), ("NETRATE", args.netrate)):
        sources[label] = {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": source_digest(path) if args.hash_sources else None,
        }
    document = {
        "schema": SCHEMA,
        "equation": {
            "unscaled_internal": "q_line = ETAL_MAT * ZNET",
            "deck_internal": "q_line_re = SCL_FAC * ETAL_MAT * ZNET",
            "cgs": "q_line_cgs = q_line_re * 4*pi*1e-10",
        },
        "units": {
            "LINEHEAT": "CMFGEN radiative-equilibrium internal units",
            "q_line_cgs": "erg cm^-3 s^-1",
            "etal_mat_internal_to_integrated_emissivity_cgs_per_sr": 1.0e-10,
            "radiative_equilibrium_internal_to_cgs": CMFGEN_RE_INTERNAL_TO_CGS,
            "derivation": [
                "EMLIN = 1e25*h/(4*pi), frequency is in 1e15 Hz, and number populations are in cm^-3",
                "the extra 1e10 is CMFGEN's opacity/radius scaling; a volumetric rate restores 4*pi",
            ],
        },
        "sign": "positive=cooling, negative=heating in CMFGEN STEQ_T convention",
        "depth_indexing": "CMFGEN one-based",
        "source_writer": {
            "file": "new_main/cmfgen_sub.f",
            "header_lines": "2739-2749",
            "znet_line": "2750",
            "lineheat_line": "2762",
            "production_re_lines": "2469-2480",
            "emlin_definition": "new_main/cmfgen.f:121-122",
            "cgs_conversion_evidence": "new_main/subs/eval_adiabatic_v3.f:143-170",
        },
        "printed_precision": {
            "LINEHEAT": "Fortran 1P,5E12.4",
            "NETRATE": "Fortran 1P,5E14.6",
            "fixture_relative_tolerance": 2.0e-4,
        },
        "selection": {
            "rule": "largest abs(unscaled LINEHEAT) among streamed candidates with finite nonzero ZNET and positive finite derived ETAL_MAT",
            "candidate_count": args.candidate_count,
            "candidate_distinct_lines": len(selected_ids),
        },
        "deck_policy": {
            "SCL_LN": True,
            "SCL_LN_FAC": 0.5,
            "SCL_DEN_LIM": "default 1e30; not reached by the witness",
            "production_comparison_quantity": "q_line_cgs_scaled",
        },
        "witness": witness,
        "sources": sources,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(document, indent=2, allow_nan=False) + "\n"
    args.out.write_text(encoded, encoding="utf-8")
    print(
        "CMFGEN_LINE_NET_FIXTURE "
        f"line={witness['line_id']} depth={witness['depth_index']} "
        f"q_internal={witness['q_line_internal_scaled']:.17g} "
        f"q_cgs={witness['q_line_cgs_scaled']:.17g} "
        f"znet={witness['znet']:.17g} "
        f"etal={witness['etal_mat_derived']:.17g} out={args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
