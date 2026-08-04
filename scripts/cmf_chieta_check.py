#!/usr/bin/env python3
"""Fail-closed consumer check for an LCMFCE01 frozen chi/eta sidecar.

The Stage-3.1 contract is producer iteration/field generation 10 after damping.
Writers record metadata; this consumer rejects caller-selected expectations
unless the caller explicitly enters the NON-CONTRACT validation lane.
"""

import argparse
import hashlib
import json
from pathlib import Path
import struct
from typing import NamedTuple


HEADER = struct.Struct("<8sIIQQQQIId")
FLAG_POST_DAMP = 1
FLAG_COHERENT_FROZEN = 2
FLAG_FREQUENCY_DESCENDING = 4
CONTRACT_ITERATION = 10
NON_CONTRACT_RC = 2
CONTRACT_STATUS = "CONTRACT"
NON_CONTRACT_STATUS = "NON-CONTRACT"


class CheckError(ValueError):
    pass


class CheckResult(NamedTuple):
    raw: bytes
    header: tuple
    arrays: list
    eta_max_abs: float
    eta_bitwise: bool
    manifest: dict
    contract_status: str


def check_artifact(path: Path, expected_iteration: int = 10,
                   expected_generation: int | None = None,
                   require_post_damp: bool = True,
                   non_contract_override: bool = False):
    path = Path(path)
    expected_generation = (expected_iteration if expected_generation is None
                           else expected_generation)
    deviates_from_contract = (
        expected_iteration != CONTRACT_ITERATION or
        expected_generation != CONTRACT_ITERATION or
        not require_post_damp)
    if deviates_from_contract and not non_contract_override:
        raise CheckError(
            "non-contract expectation requires --non-contract-override")
    raw = path.read_bytes()
    if len(raw) < HEADER.size:
        raise CheckError("truncated header")
    header = HEADER.unpack_from(raw)
    magic, endian, version, nr, nnu, iteration, generation, flags, reserved, texp = header
    if (magic, endian, version, reserved) != (
            b"LCMFCE01", 0x01020304, 1, 0):
        raise CheckError("schema identity mismatch")
    if nr == 0 or nnu == 0 or not (texp > 0.0):
        raise CheckError("invalid dimensions/time")
    if flags & ~(FLAG_POST_DAMP | FLAG_COHERENT_FROZEN |
                 FLAG_FREQUENCY_DESCENDING):
        raise CheckError("unknown header flag")
    if not flags & FLAG_COHERENT_FROZEN:
        raise CheckError("field is not coherent-frozen")
    if not flags & FLAG_FREQUENCY_DESCENDING:
        raise CheckError("frequency ordering flag is absent")
    if iteration != expected_iteration:
        raise CheckError(
            f"iteration mismatch: got {iteration}, expected {expected_iteration}")
    if generation != expected_generation:
        raise CheckError(
            f"field_generation mismatch: got {generation}, "
            f"expected {expected_generation}")
    post_damp = bool(flags & FLAG_POST_DAMP)
    if require_post_damp and not post_damp:
        raise CheckError("post-damping field required")

    count = nr * nnu
    lengths = [nr + 1, nnu, nnu] + [count] * 6
    arrays = []
    off = HEADER.size
    for length in lengths:
        size = 8 * length
        if off + size > len(raw):
            raise CheckError("truncated array")
        arrays.append(struct.unpack_from(f"<{length}d", raw, off))
        off += size
    if off != len(raw):
        raise CheckError("trailing bytes")
    _, nu, dnu, _, _, eta_fixed, eta_coherent, eta_total, _ = arrays
    if not all(nu[k] > nu[k + 1] for k in range(len(nu) - 1)):
        raise CheckError("frequency is not descending")
    if not all(value > 0.0 for value in dnu):
        raise CheckError("non-positive dnu")
    eta_max_abs = 0.0
    eta_bitwise = True
    for fixed, coherent, total in zip(eta_fixed, eta_coherent, eta_total):
        split = fixed + coherent
        eta_max_abs = max(eta_max_abs, abs(total - split))
        if struct.pack("<d", split) != struct.pack("<d", total):
            eta_bitwise = False
    if not eta_bitwise:
        raise CheckError("eta decomposition is not bitwise exact")

    sidecar_path = Path(str(path) + ".manifest.json")
    try:
        manifest = json.loads(sidecar_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckError(f"invalid/missing sidecar: {exc}") from exc
    expected_sidecar = {
        "schema": "LCMFCE01-v1",
        "iteration": iteration,
        "field_generation": generation,
        "post_damping": post_damp,
        "coherent_frozen": True,
        "frequency_descending": True,
        "eta_decomposition_bitwise": eta_bitwise,
        "eta_decomposition_max_abs": eta_max_abs,
    }
    for key, value in expected_sidecar.items():
        if manifest.get(key) != value:
            raise CheckError(f"sidecar {key} disagrees with payload")
    digest = hashlib.sha256(raw).hexdigest()
    if manifest.get("sha256") != digest:
        raise CheckError("sidecar sha256 mismatch")
    contract_status = (NON_CONTRACT_STATUS if deviates_from_contract
                       else CONTRACT_STATUS)
    return CheckResult(raw, header, arrays, eta_max_abs, eta_bitwise,
                       manifest, contract_status)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--expected-iteration", type=int, default=10,
                        help="consumer contract is 10; other values require "
                             "--non-contract-override")
    parser.add_argument("--expected-field-generation", type=int,
                        help="default: same as expected iteration")
    parser.add_argument("--allow-pre-damp", action="store_true")
    parser.add_argument(
        "--non-contract-override", action="store_true",
        help="validate alternate metadata, print NON-CONTRACT, and return 2")
    args = parser.parse_args()
    try:
        result = check_artifact(
            args.input, args.expected_iteration,
            args.expected_field_generation, not args.allow_pre_damp,
            args.non_contract_override)
    except (OSError, CheckError) as exc:
        print(f"FAIL: {exc}")
        return 1
    raw, header = result.raw, result.header
    if result.contract_status == NON_CONTRACT_STATUS:
        print(f"NON-CONTRACT: iteration={header[5]} "
              f"field_generation={header[6]} "
              f"post_damp={int(bool(header[7] & FLAG_POST_DAMP))} "
              f"bytes={len(raw)}")
        return NON_CONTRACT_RC
    print(f"PASS: iteration={header[5]} field_generation={header[6]} "
          f"post_damp={int(bool(header[7] & FLAG_POST_DAMP))} bytes={len(raw)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
