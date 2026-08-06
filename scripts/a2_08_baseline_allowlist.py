#!/usr/bin/env python3
"""Create the immutable A2-08 changed-output allowlist before source edits."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation/a2_08/A2_08_CHANGED_OUTPUT_ALLOWLIST.json"
SIDE = OUT.with_suffix(".sha256")


def git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def entry(entry_id: str, symbol: str, kind: str, identity: dict[str, int],
          before: str, change: str, after: str, reason: str) -> dict[str, object]:
    return {
        "id": entry_id,
        "artifact_or_symbol": symbol,
        "identity_kind": kind,
        "identity": identity,
        "before_value_or_sha256": before,
        "allowed_change_kind": change,
        "expected_after_constraint": after,
        "reason_code": reason,
        "owner_stage": "A2-08",
    }


def document() -> dict[str, object]:
    return {
        "schema": "lumina-a2-08-changed-output-allowlist-v1",
        "stage": "A2-08",
        "baseline_head": git("rev-parse", "HEAD"),
        "created_before_source_edit": True,
        "canonicalization": "utf8-sorted-keys-indent2-lf-final-lf",
        "entries": [
            entry("BB-NORMAL-L0-S0", "tau_sobolev", "line_shell",
                  {"line": 0, "shell": 0}, "fixture:+2.0",
                  "SIGNED_TAU_PUBLISH", "value==+2.0 && status==VALID",
                  "SIGNED_SOBOLEV_DIRECT_DIFFERENCE"),
            entry("BB-INVERSION-L1-S0", "tau_sobolev", "line_shell",
                  {"line": 1, "shell": 0}, "fixture:+1e-100(clamped)",
                  "SIGNED_TAU_PUBLISH", "value<0 && status==VALID",
                  "REMOVE_STIMULATED_EMISSION_CLAMP"),
            entry("BB-EXACT-ZERO-L2-S0", "tau_sobolev", "line_shell",
                  {"line": 2, "shell": 0}, "fixture:+1e-100(floor)",
                  "EXACT_ZERO", "bits==positive_zero && status==EXACT_ZERO",
                  "REMOVE_NUMERIC_FLOOR_EXACT_ZERO"),
            entry("BF-INVERSION-R0-S0-B1", "bf_net_route", "route_shell_bin",
                  {"route": 0, "shell": 0, "bin": 1}, "fixture:+0(clamped)",
                  "SIGNED_BF_NET_PUBLISH", "value<0 && status==VALID",
                  "REMOVE_STIMULATED_RECOMBINATION_CLAMP"),
            entry("SOURCE-NEGATIVE-L1-S0", "line_source_S", "line_shell",
                  {"line": 1, "shell": 0}, "fixture:+0(fallback-sentinel)",
                  "STATUS_BEARING_LINE_SOURCE", "value<0 && status==VALID",
                  "REMOVE_NUMERIC_SOURCE_SENTINEL"),
            entry("REPLAY-LINE-BLOCK-G1", "replay_line_block", "buffer",
                  {"buffer": 1}, "fixture:absent",
                  "ATOMIC_DUAL_VIEW_COMMIT",
                  "radiation_generation==line_generation && status==VALID",
                  "REPLAY_LINE_BLOCK_PUBLICATION"),
            entry("COMPONENT-S0-B0", "chi_total", "shell_bin",
                  {"shell": 0, "bin": 0}, "fixture:legacy-unsplit",
                  "SIGNED_COMPONENT_PUBLICATION",
                  "total==((es+bb)+bf)+ff && closure<=1e-10",
                  "CPU_OPACITY_COMPONENT_OWNER"),
            entry("OWNER-POINTER-G1", "CpuOpacityPublication", "diagnostic",
                  {"generation": 1}, "fixture:no-owner",
                  "OWNER_PROVENANCE", "committed_generation==1",
                  "ATOMIC_PUBLICATION_OWNER"),
        ],
        "forbidden_scopes": ["whole_spectrum", "all_opacity", "physics_changed"],
    }


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("write", "check"))
    args = parser.parse_args()
    expected = canonical_bytes(document())
    digest = hashlib.sha256(expected).hexdigest()
    side = f"{digest}  validation/a2_08/A2_08_CHANGED_OUTPUT_ALLOWLIST.json\n".encode()
    if args.command == "write":
        if OUT.exists() or SIDE.exists():
            raise SystemExit("refusing to replace an existing A2-08 allowlist seal")
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_bytes(expected)
        SIDE.write_bytes(side)
    else:
        if OUT.read_bytes() != expected or SIDE.read_bytes() != side:
            raise SystemExit("A2-08 allowlist canonical bytes or sidecar mismatch")
    print(f"PASS A2_08_ALLOWLIST baseline={document()['baseline_head']} sha256={digest} entries=8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
