#!/usr/bin/env python3
"""Compare independently scheduled 1-GPU and 4-GPU reduced CMF results."""

from __future__ import annotations

import array
import math
import pathlib
import struct
import sys


MAGIC = b"LUMINA_MGPU_S2\0\0"


def fail(message: str) -> "NoReturn":
    raise SystemExit(f"CMF_MGPU_REDUCED_SPLIT_COMPARE_FAIL {message}")


def read_result(path: pathlib.Path) -> tuple[int, int, int, array.array, array.array]:
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
    return shells, bins, devices, values[:cells], values[cells:]


def main() -> int:
    if len(sys.argv) != 3:
        fail("usage: compare_cmf_exact_multigpu_reduced.py ONE.bin FOUR.bin")
    one_path, four_path = map(pathlib.Path, sys.argv[1:])
    one_ns, one_nb, one_devices, one_j, one_error = read_result(one_path)
    four_ns, four_nb, four_devices, four_j, four_error = read_result(four_path)
    if (one_ns, one_nb) != (four_ns, four_nb):
        fail("shape-mismatch")
    if one_devices != 1 or four_devices != 4:
        fail(f"device-contract one/four={one_devices}/{four_devices}")

    covered = 0
    max_relative = 0.0
    max_envelope_ratio = 0.0
    for a, ua, b, ub in zip(one_j, one_error, four_j, four_error):
        if not (math.isfinite(a) and math.isfinite(ua) and
                math.isfinite(b) and math.isfinite(ub)):
            fail("nonfinite-result")
        if a < 0.0 or ua < 0.0 or b < 0.0 or ub < 0.0:
            fail("negative-result")
        difference = abs(a - b)
        denominator = max(abs(a), abs(b))
        relative = difference / denominator if denominator > 0.0 else 0.0
        max_relative = max(max_relative, relative)
        combined = math.nextafter(ua + ub, math.inf)
        if difference <= combined:
            covered += 1
        if combined > 0.0:
            max_envelope_ratio = max(max_envelope_ratio,
                                     difference / combined)

    cells = one_ns * one_nb
    if covered != cells:
        fail(f"envelope-coverage covered={covered}/{cells}")
    print(
        "CMF_MGPU_REDUCED_SPLIT_COMPARE PASS "
        f"shells={one_ns} bins={one_nb} cells={cells} devices=1/4 "
        f"max_rel_one_four={max_relative:.17g} "
        f"envelope_ratio={max_envelope_ratio:.17g} "
        f"covered={covered}/{cells} numerical_repairs=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
