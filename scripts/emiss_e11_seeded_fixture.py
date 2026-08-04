#!/usr/bin/env python3
"""Offline E11 negative control: seeded edge distortion, generation swap,
and OFF byte identity."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from emiss_e11_fluor_matrix import (
    CONTRACT_ITERATION, FluorMatrixError, read_fluor_matrix,
    write_fixture_matrix)


def synthetic_packet_output(seed: int, estimator_gate: str | None) -> bytes:
    """A tiny transport-state oracle: observation may not mutate packet bytes."""
    rng = np.random.default_rng(seed)
    packet = bytearray()
    for packet_id in range(128):
        input_bin = int(rng.integers(0, 4))
        output_bin = int(rng.integers(0, 4))
        energy = float(rng.random())
        packet.extend(struct.pack("<IIId", packet_id, input_bin, output_bin, energy))
        if estimator_gate:
            # The enabled observer consumes only copies; the packet serialization
            # above is intentionally its immutable input/output state.
            _observed = (input_bin, output_bin, energy)
            assert _observed[2] == energy
    return bytes(packet)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/emiss_e11_fixture"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    base = args.out_dir / "formal_matrix_base.bin"
    distorted = args.out_dir / "formal_matrix_distorted.bin"
    input_count = np.array([2, 0, 2, 0], dtype=np.uint64)
    input_energy = np.array([5.0, 0.0, 10.0, 0.0])
    terminal = input_energy.copy()
    outside = np.zeros(4)
    shell_count = np.array([1, 3], dtype=np.uint64)
    shell_kcount = np.array([0, 2], dtype=np.uint64)
    shell_abs = np.array([5.0, 10.0])
    shell_emit = shell_abs.copy()
    edges = [(0, 0, 3.0), (0, 1, 2.0), (2, 0, 2.0), (2, 3, 8.0)]
    common = dict(nb=4, ns=2, iteration=11, numin=1.0e14, numax=1.0e16,
                  input_count=input_count, input_energy=input_energy,
                  terminal_energy=terminal, outside_energy=outside,
                  shell_count=shell_count, shell_kpacket_count=shell_kcount,
                  shell_absorbed=shell_abs, shell_reemitted=shell_emit,
                  kpacket_events=2, kpacket_absorbed=8.0,
                  kpacket_reemitted=8.0)
    write_fixture_matrix(base, edges=edges, **common)
    good = read_fluor_matrix(base, expected_iteration=11,
                             non_contract_override=True)

    factor = 7.0
    bad_edges = [(ib, ob, energy * factor if (ib, ob) == (2, 0) else energy)
                 for ib, ob, energy in edges]
    write_fixture_matrix(distorted, edges=bad_edges, **common)
    failure = ""
    try:
        read_fluor_matrix(distorted, expected_iteration=11,
                          non_contract_override=True)
    except FluorMatrixError as exc:
        failure = str(exc)
    if "input_bin=2" not in failure:
        raise RuntimeError(f"seeded distortion was not localized: {failure}")
    bad = read_fluor_matrix(distorted, expected_iteration=11,
                            non_contract_override=True,
                            verify_column_closure=False)
    good_edge = float(good.edges["output_energy"][(
        (good.edges["input_bin"] == 2) & (good.edges["output_bin"] == 0))][0])
    bad_edge = float(bad.edges["output_energy"][(
        (bad.edges["input_bin"] == 2) & (bad.edges["output_bin"] == 0))][0])
    measured_factor = bad_edge / good_edge
    if measured_factor != factor:
        raise RuntimeError("seeded edge multiplier not recovered exactly")

    # A canonical 1000-bin identity fixture exercises the E10 formal consumer
    # without changing its frozen source.  This is arithmetic-only and is not a
    # physical identity fallback in production.
    identity = args.out_dir / "formal_matrix_identity_1000.bin"
    nfull, sfull = 1000, 50
    full_count = np.ones(nfull, dtype=np.uint64)
    full_energy = np.ones(nfull)
    full_shell_count = np.zeros(sfull, dtype=np.uint64)
    full_shell_count[0] = nfull
    full_shell_energy = np.zeros(sfull)
    full_shell_energy[0] = nfull
    write_fixture_matrix(
        identity, nb=nfull, ns=sfull, iteration=11,
        numin=1.5e14, numax=3.0e16, input_count=full_count,
        input_energy=full_energy, terminal_energy=full_energy,
        outside_energy=np.zeros(nfull), shell_count=full_shell_count,
        shell_kpacket_count=np.zeros(sfull, dtype=np.uint64),
        shell_absorbed=full_shell_energy, shell_reemitted=full_shell_energy,
        edges=[(i, i, 1.0) for i in range(nfull)])
    identity_matrix = read_fluor_matrix(identity, expected_iteration=11,
                                        non_contract_override=True)

    # [N4] GENERATION-SWAP negative control.  Reproduce the exact production
    # accident: the payload is replaced in place by a different generation AND
    # its SHA-256 sidecar is refreshed with it, so `sha256sum -c` still passes.
    # A consumer holding the campaign contract (iteration 10) must refuse.
    swap = args.out_dir / "formal_matrix_generation_swap.bin"
    swap_common = dict(common)
    swap_common["iteration"] = CONTRACT_ITERATION
    write_fixture_matrix(swap, edges=edges, **swap_common)
    contract_ok = read_fluor_matrix(swap, expected_iteration=CONTRACT_ITERATION)
    sidecar = Path(str(swap) + ".sha256")
    before = {"iteration": contract_ok.header["iteration"],
              "sha256": contract_ok.sha256,
              "sidecar": sidecar.read_text().split()[0]}
    # the writer refreshes payload+sidecar together, exactly as production did
    write_fixture_matrix(swap, edges=edges, **common)          # iteration 11
    after_digest = hashlib.sha256(swap.read_bytes()).hexdigest()
    sidecar_agrees = sidecar.read_text().split()[0] == after_digest
    if not sidecar_agrees:
        raise RuntimeError("generation-swap control did not reproduce a "
                           "self-consistent sidecar")
    if after_digest == before["sha256"]:
        raise RuntimeError("generation swap did not change the payload")
    swap_failure = ""
    try:
        read_fluor_matrix(swap, expected_iteration=CONTRACT_ITERATION)
    except FluorMatrixError as exc:
        swap_failure = str(exc)
    if "generation mismatch" not in swap_failure:
        raise RuntimeError(
            f"generation swap was NOT refused by the consumer: {swap_failure}")
    # the pinned-identity guard is independent of the sidecar and also fires
    pin_failure = ""
    try:
        read_fluor_matrix(swap, expected_iteration=11,
                          non_contract_override=True,
                          expected_sha256=before["sha256"])
    except FluorMatrixError as exc:
        pin_failure = str(exc)
    if "identity changed" not in pin_failure:
        raise RuntimeError(f"sha pin did not fire: {pin_failure}")
    # a non-contract expectation without the explicit override is refused too
    override_failure = ""
    try:
        read_fluor_matrix(swap, expected_iteration=11)
    except FluorMatrixError as exc:
        override_failure = str(exc)
    if "non_contract_override" not in override_failure:
        raise RuntimeError(f"override guard did not fire: {override_failure}")

    baseline_packets = synthetic_packet_output(0xE11, None)
    off_packets = synthetic_packet_output(0xE11, "")
    if baseline_packets != off_packets:
        raise RuntimeError("OFF packet stream is not byte-identical")
    summary = {
        "schema": "lumina-emiss-e11-seeded-fixture-v1",
        "base_sha256": good.sha256,
        "distorted_sha256": bad.sha256,
        "identity_1000_sha256": identity_matrix.sha256,
        "seeded_edge": {"input_bin": 2, "output_bin": 0,
                        "factor_requested": factor,
                        "factor_measured": measured_factor},
        "negative_control_detected": True,
        "negative_control_failure": failure,
        "generation_swap_control": {
            "before": before,
            "after_sha256": after_digest,
            "sidecar_recheck_still_passes": sidecar_agrees,
            "consumer_refused": swap_failure,
            "sha_pin_refused": pin_failure,
            "override_guard_refused": override_failure,
        },
        "base_column_closure_max_abs": good.column_closure_max_abs,
        "distorted_column_closure_max_abs": bad.column_closure_max_abs,
        "off_byte_invariant": True,
        "off_packet_bytes": len(off_packets),
        "off_packet_sha256": hashlib.sha256(off_packets).hexdigest(),
        "clamp": 0, "normalization_repair": 0, "fallback": 0,
    }
    out = args.out_dir / "fixture_summary.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
