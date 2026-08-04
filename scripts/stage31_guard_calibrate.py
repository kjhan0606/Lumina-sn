#!/usr/bin/env python3
"""Derive the round-7D O(h^2) solution-sign guard coefficient from KA data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ka1", type=Path, required=True)
    parser.add_argument("--ka2", type=Path, required=True)
    parser.add_argument("--ka3", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    ka1 = json.loads(args.ka1.read_text())
    ka2 = json.loads(args.ka2.read_text())
    ka3 = json.loads(args.ka3.read_text())
    samples: list[dict[str, object]] = []
    for case in ka1["cases"]:
        for level in case["levels"]:
            n = int(level["grid"]["nmu"])
            samples.append({
                "ka": "KA1", "case": case["tau_radius"], "n": n,
                "relative_error": level["I_relative_l2"],
                "coefficient": float(level["I_relative_l2"]) * n * n,
            })
    for level in ka2["levels"]:
        n = int(level["grid"]["nmu"])
        samples.append({
            "ka": "KA2", "n": n,
            "relative_error": level["J_oracle_relative_l2"],
            "coefficient": float(level["J_oracle_relative_l2"]) * n * n,
        })
    for level in ka3["levels"]:
        n = int(level["grid"]["ns"])
        samples.append({
            "ka": "KA3", "n": n,
            "relative_error": level["profile_relative_l1"],
            "coefficient": float(level["profile_relative_l1"]) * n * n,
        })
    maximum = max(samples, key=lambda item: float(item["coefficient"]))
    report = {
        "schema": "stage31-sign-guard-calibration-v1",
        "formula": "C=max(E_relative/h^2), h=1/n",
        "safety_multiplier": 1.0,
        "coefficient": maximum["coefficient"],
        "attaining_sample": maximum,
        "per_ka_maximum": {
            name: max(float(item["coefficient"]) for item in samples if item["ka"] == name)
            for name in ("KA1", "KA2", "KA3")
        },
        "samples": samples,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
