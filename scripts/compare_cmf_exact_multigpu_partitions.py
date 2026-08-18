#!/usr/bin/env python3
"""Compare equal-ray and weighted-segment results from one benchmark binary."""

from __future__ import annotations

import array
import hashlib
import math
import pathlib
import struct
import sys


MAGIC = b"LUMINA_MGPU_S2\0\0"


def fail(message: str) -> "NoReturn":
    raise SystemExit(f"CMF_MGPU_PARTITION_COMPARE_FAIL {message}")


def read_result(path: pathlib.Path) -> tuple[int, int, int, array.array, array.array, bytes]:
    payload = path.read_bytes()
    if len(payload) < 40 or payload[:16] != MAGIC:
        fail(f"invalid-header path={path}")
    shells, bins, devices = struct.unpack_from("=QQQ", payload, 16)
    cells = shells * bins
    expected = 40 + 2 * cells * 8
    if shells == 0 or bins < 2 or len(payload) != expected:
        fail(f"invalid-shape-or-size path={path}")
    values = array.array("d")
    values.frombytes(payload[40:])
    if sys.byteorder != "little":
        values.byteswap()
    return shells, bins, devices, values[:cells], values[cells:], payload


def main() -> int:
    if len(sys.argv) != 3:
        fail("usage: compare_cmf_exact_multigpu_partitions.py EQUAL.bin WEIGHTED.bin")
    equal_path, weighted_path = map(pathlib.Path, sys.argv[1:])
    ens, enb, edev, equal_j, equal_error, equal_payload = read_result(equal_path)
    wns, wnb, wdev, weighted_j, weighted_error, weighted_payload = read_result(weighted_path)
    if (ens, enb) != (wns, wnb):
        fail("shape-mismatch")
    if edev != 4 or wdev != 4:
        fail(f"device-contract equal/weighted={edev}/{wdev}")

    covered = 0
    max_relative = 0.0
    max_absolute = 0.0
    max_envelope_ratio = 0.0
    max_index = 0
    for index, (a, ua, b, ub) in enumerate(
            zip(equal_j, equal_error, weighted_j, weighted_error)):
        if not (math.isfinite(a) and math.isfinite(ua) and
                math.isfinite(b) and math.isfinite(ub)):
            fail(f"nonfinite-result index={index}")
        if a < 0.0 or ua < 0.0 or b < 0.0 or ub < 0.0:
            fail(f"negative-result index={index}")
        difference = abs(a - b)
        denominator = max(abs(a), abs(b))
        relative = difference / denominator if denominator > 0.0 else 0.0
        if relative > max_relative:
            max_relative = relative
            max_index = index
        max_absolute = max(max_absolute, difference)
        combined = math.nextafter(ua + ub, math.inf)
        if difference <= combined:
            covered += 1
        if combined > 0.0:
            max_envelope_ratio = max(max_envelope_ratio,
                                     difference / combined)

    cells = ens * enb
    if covered != cells:
        fail(f"envelope-coverage covered={covered}/{cells}")
    equal_sha = hashlib.sha256(equal_payload).hexdigest()
    weighted_sha = hashlib.sha256(weighted_payload).hexdigest()
    print(
        "CMF_MGPU_PARTITION_COMPARE PASS "
        f"shells={ens} bins={enb} cells={cells} devices=4/4 "
        f"max_rel_equal_weighted={max_relative:.17g} "
        f"max_abs_equal_weighted={max_absolute:.17g} "
        f"max_shell/bin={max_index // enb}/{max_index % enb} "
        f"envelope_ratio={max_envelope_ratio:.17g} "
        f"covered={covered}/{cells} "
        f"byte_identical={int(equal_payload == weighted_payload)} "
        f"equal_sha256={equal_sha} weighted_sha256={weighted_sha} "
        "numerical_repairs=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
