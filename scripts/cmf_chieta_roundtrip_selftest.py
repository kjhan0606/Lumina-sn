#!/usr/bin/env python3
"""Offline LCMFCE01 v1 write/read/bitwise round-trip self-test."""

import argparse
from pathlib import Path
import struct
import subprocess
import tempfile

from cmf_chieta_check import check_artifact


ROOT = Path(__file__).resolve().parents[1]
HEADER = struct.Struct("<8sIIQQQQIId")


def parse(path: Path):
    raw = path.read_bytes()
    if len(raw) < HEADER.size:
        raise ValueError("truncated header")
    header = HEADER.unpack_from(raw)
    magic, endian, version, nr, nnu, iteration, generation, flags, reserved, texp = header
    if (magic, endian, version, reserved) != (b"LCMFCE01", 0x01020304, 1, 0):
        raise ValueError("schema identity mismatch")
    if flags & ~0x7 or nr == 0 or nnu == 0:
        raise ValueError("header contract mismatch")
    count = nr * nnu
    lengths = [nr + 1, nnu, nnu] + [count] * 6
    arrays = []
    off = HEADER.size
    for length in lengths:
        size = 8 * length
        if off + size > len(raw):
            raise ValueError("truncated array")
        arrays.append(struct.unpack_from(f"<{length}d", raw, off))
        off += size
    if off != len(raw):
        raise ValueError("trailing bytes")
    r_edge, nu, dnu, chi, chic, etaf, etac, etat, jprod = arrays
    if not all(nu[k] > nu[k + 1] for k in range(len(nu) - 1)):
        raise ValueError("frequency is not descending")
    if not all(x > 0 for x in dnu):
        raise ValueError("non-positive dnu")
    eta_max_abs = 0.0
    eta_bitwise = True
    for fixed, coherent, total in zip(etaf, etac, etat):
        eta_max_abs = max(eta_max_abs, abs(total - (fixed + coherent)))
        if struct.pack("<d", fixed + coherent) != struct.pack("<d", total):
            eta_bitwise = False
    if not eta_bitwise:
        raise ValueError("eta decomposition is not bitwise exact")
    return raw, header, arrays, eta_max_abs, eta_bitwise


def serialize(header, arrays):
    pieces = [HEADER.pack(*header)]
    for values in arrays:
        pieces.append(struct.pack(f"<{len(values)}d", *values))
    return b"".join(pieces)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path,
                    help="validate an existing runtime dump instead of the fixture")
    ap.add_argument("--no-build", action="store_true")
    args = ap.parse_args()
    if args.input:
        path = args.input.resolve()
    else:
        if not args.no_build:
            subprocess.run(["make", "-B", "selftest_cmf_chieta_dump"],
                           cwd=ROOT, check=True)
        out = Path(tempfile.mkdtemp(prefix="cmf_chieta_rt_", dir="/tmp"))
        path = out / "fixture.lcmfce"
        subprocess.run([str(ROOT / "selftest_cmf_chieta_dump"), str(path)],
                       cwd=ROOT, check=True)
    result = check_artifact(path)
    if result.contract_status != "CONTRACT":
        raise ValueError("round-trip fixture is outside the consumer contract")
    raw, header, arrays, manifest = (
        result.raw, result.header, result.arrays, result.manifest)
    rebuilt = serialize(header, arrays)
    if rebuilt != raw:
        raise ValueError("write-read-write bytes differ")
    digest = manifest["sha256"]
    print(f"PASS LCMFCE01 write-read-write bitwise roundtrip: {path}")
    print(f"sha256={digest} bytes={len(raw)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
