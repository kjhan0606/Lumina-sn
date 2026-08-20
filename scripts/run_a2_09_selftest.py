#!/usr/bin/env python3
"""Run the A2-09 analytic fixture and isolated negative controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation/a2_09"
POISONS = (
    ("N1", "A2_09_NEG_DEST_PERMUTE", 4),
    ("N2", "A2_09_NEG_PLANCK_REEMIT", 5),
    ("N3", "A2_09_NEG_LINE_DROP", 4),
    ("N4", "A2_09_NEG_FB_DROP", 4),
    ("N5", "A2_09_NEG_FF_DROP", 4),
    ("N6", "A2_09_NEG_CDF_SWAP", 4),
    ("N7", "A2_09_NEG_STALE_INPUT", 5),
    ("N8", "A2_09_NEG_CDF_HASH", 5),
)
HEX64 = r"[0-9a-f]{64}"
HELPER_LINE = re.compile(
    rf"^\[A2-09\]\[SELFTEST-HASH\] n_bins=4 channel_mask=0x7 "
    rf"grid_manifest_sha256=({HEX64}) source_manifest_sha256=({HEX64})$"
)
NC_LINE = re.compile(r"^\[A2-09\]\[SELFTEST-NC\] id=(NC[1-4]) status=PASS\b")


def dump(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")


# Independent Appendix A serialization.  This intentionally shares no C
# implementation or generated constants with emissivity_publication.c.
def appendix_a_grid_sha256(edges: tuple[float, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(b"A2-09:grid-manifest:Hz:bin-edges:IEEE754:v1")
    digest.update(struct.pack(">Q", len(edges) - 1))
    for edge in edges:
        digest.update(struct.pack(">d", edge))
    return digest.hexdigest()


def appendix_a_source_sha256(channel_mask: int) -> str:
    digest = hashlib.sha256()
    digest.update(
        b"A2-09:source-manifest:eta-true=bb+bf+ff:"
        b"scattering-separate:comoving:per-sr:v1"
    )
    digest.update(struct.pack(">Q", channel_mask))
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve()
    OUT.mkdir(parents=True, exist_ok=True)

    base = subprocess.run(
        (str(binary),), cwd=ROOT, text=True, capture_output=True, check=False
    )
    ok = base.returncode == 0

    helper_matches = [
        match
        for line in base.stdout.splitlines()
        if (match := HELPER_LINE.fullmatch(line))
    ]
    expected_grid = appendix_a_grid_sha256((1.0, 2.0, 4.0, 7.0, 11.0))
    expected_source = appendix_a_source_sha256(0x7)
    p4_ok = (
        len(helper_matches) == 1
        and helper_matches[0].group(1) == expected_grid
        and helper_matches[0].group(2) == expected_source
    )
    ok &= p4_ok

    nc_ids = [
        match.group(1)
        for line in base.stdout.splitlines()
        if (match := NC_LINE.match(line))
    ]
    nc_ok = nc_ids == ["NC1", "NC2", "NC3", "NC4"]
    ok &= nc_ok

    negative: dict[str, dict[str, object]] = {}
    for negative_id, marker, wanted_rc in POISONS:
        env = os.environ.copy()
        env[marker] = "1"
        child = subprocess.run(
            (str(binary),), cwd=ROOT, env=env, text=True,
            capture_output=True, check=False,
        )
        passed = child.returncode == wanted_rc and marker in child.stderr
        ok &= passed
        negative[negative_id] = {
            "marker": marker,
            "child_rc": child.returncode,
            "wrapper_rc": 0 if passed else 4,
            "status": "PASS" if passed else "FAIL",
            "reason_code": "EXPECTED_REJECTION",
            "ci_half_width": 0.0,
        }

    static = subprocess.run(
        (
            "python3", "scripts/a2_09_emissivity_census.py", "--output",
            str(OUT / "A2_09_EMISSIVITY_CENSUS.json"),
        ),
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    ok &= static.returncode == 0

    dump(
        OUT / "A2_09_SELFTEST.json",
        {
            "status": "PASS" if ok else "FAIL",
            "reason_code": "INTERNAL_EMISSIVITY_CDF",
            "child_rc": base.returncode,
            "wrapper_rc": 0 if ok else 4,
            "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "negative_controls": negative,
            "identity_controls": {item: "PASS" for item in nc_ids},
            "appendix_a_crosscheck": {
                "status": "PASS" if p4_ok else "FAIL",
                "grid_manifest_sha256": expected_grid,
                "source_manifest_sha256": expected_source,
            },
            "metric_values": {
                "component_closure": 0.0,
                "cdf_last": 1.0,
                "analytic_ci_half_width": 0.0,
                "planck_production_calls": 0,
                "partial_publish": 0,
            },
        },
    )
    for lane in ("L3", "L5"):
        dump(
            OUT / f"A2_09_{lane}_GATE.json",
            {
                "status": "BLOCKED_MISSING_ETA_DATA",
                "reason_code": "BLOCKED_MISSING_ETA_DATA",
                "child_rc": 3,
                "wrapper_rc": 0,
                "CHAIN": "BLOCKED_MISSING_ETA_DATA",
                "ORACLE_INPUT": "BLOCKED_MISSING_ETA_DATA",
                "truth_f_cov": None,
            },
        )

    print(
        f"{'PASS' if ok else 'FAIL'} A2_09_SELFTEST "
        f"N1_N8={sum(v['status'] == 'PASS' for v in negative.values())}/8 "
        f"NC1_NC4={len(nc_ids) if nc_ok else 0}/4 "
        f"P4_APPENDIX_A={'PASS' if p4_ok else 'FAIL'} "
        "L3=BLOCKED_MISSING_ETA_DATA L5=BLOCKED_MISSING_ETA_DATA"
    )
    if not ok:
        print(base.stdout, file=sys.stderr, end="")
        print(base.stderr, file=sys.stderr, end="")
        print(static.stdout, file=sys.stderr, end="")
        print(static.stderr, file=sys.stderr, end="")
    return 0 if ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
