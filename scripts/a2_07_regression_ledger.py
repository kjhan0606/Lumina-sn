#!/usr/bin/env python3
"""Write the single A2-07 §16 regression-ledger object."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chain", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--self-check", type=Path, required=True)
    parser.add_argument("--classic", type=Path, required=True)
    parser.add_argument("--source-hash", required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        chain, oracle = load(args.chain), load(args.oracle)
        selfcheck, classic = load(args.self_check), load(args.classic)
    except (OSError, json.JSONDecodeError) as exc:
        print(exc)
        return 2
    neg = selfcheck.get("negative_control_status", {})
    entry = {
        "stage_id": "A2-07", "contract": "SPEC_A2_07_V1",
        "source_tree_hash": args.source_hash,
        "input_manifest_hash": {"CHAIN": sha(args.chain), "ORACLE_INPUT": sha(args.oracle)},
        "oracle_id": "CMFGEN_POP_AND_RATE_EXPORT_SAME_GENERATION",
        "node": platform.node(), "command": args.command,
        "exit_status": {"CHAIN": chain.get("child_rc"),
                        "ORACLE_INPUT": oracle.get("child_rc")},
        "new_layer_status": {
            "L2ION": {"CHAIN": chain.get("new_layer_status", {}).get("L2ION"),
                      "ORACLE_INPUT": oracle.get("new_layer_status", {}).get("L2ION")},
            "L2LEVEL": {"CHAIN": chain.get("new_layer_status", {}).get("L2LEVEL"),
                        "ORACLE_INPUT": oracle.get("new_layer_status", {}).get("L2LEVEL"),
                        "PARTITION_CPU": oracle.get("new_layer_status", {}).get("PARTITION_CPU")},
        },
        "all_previous_layer_statuses": {
            "A2_03_A2_06_GRAMMAR_SELFTESTS": "SEE_GRAMMAR_DEBUG_ARTIFACTS",
            "A2_05": "PRESERVE_UPSTREAM_ARTIFACT",
            "A2_06": "PRESERVE_UPSTREAM_ARTIFACT"
        },
        "negative_control_status": {
            "stage_swap": neg.get("N1"), "neighbor_ne": neg.get("N2"),
            "trad_for_te": neg.get("N3"), "level_shuffle": neg.get("N4")},
        "coverage": {lane: doc.get("metrics", {}).get("truth_f_cov")
                     for lane, doc in (("CHAIN", chain), ("ORACLE_INPUT", oracle))},
        "metric_values": {lane: doc.get("metrics", {})
                          for lane, doc in (("CHAIN", chain), ("ORACLE_INPUT", oracle))},
        "changed_output_allowlist": ["ion_population", "level_population",
                                     "electron_density", "partition", "population_status"],
        "guard_hits": {"CHAIN": chain.get("checks", {}),
                       "ORACLE_INPUT": oracle.get("checks", {})},
        "fallback_hits": {
            "CHAIN": chain.get("population_counters", {}).get("pop_fallback_attempts"),
            "ORACLE_INPUT": oracle.get("population_counters", {}).get("pop_fallback_attempts")},
        "rng_seed": {"CHAIN": chain.get("rng_seed"), "ORACLE_INPUT": oracle.get("rng_seed")},
        "mc_confidence": {"CHAIN": chain.get("mc_confidence"),
                          "ORACLE_INPUT": oracle.get("mc_confidence")},
        "classic_status": classic.get("status"),
        "artifact_paths": [str(args.chain), str(args.oracle), str(args.self_check),
                           str(args.classic)],
        "driver_signoff": "PENDING_FABLE"
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(entry, separators=(",", ":"), allow_nan=False) + "\n",
                           encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
