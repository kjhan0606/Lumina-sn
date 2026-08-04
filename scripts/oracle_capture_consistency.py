#!/usr/bin/env python3
"""Compare archived production Jbar consumers with frozen-oracle replay values."""
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


SHELLS = (0, 8, 43)


def relerr(a: float, b: float) -> float:
    if a == b:
        return 0.0
    return abs(a - b) / max(abs(a), abs(b), 1.0e-300)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle-dir", required=True, type=Path)
    ap.add_argument("--capture", required=True, type=Path)
    ap.add_argument("--consumer-iter", required=True, type=int)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    expected: dict[tuple[int, int, int, int], dict[str, str]] = {}
    for shell in SHELLS:
        with (args.oracle_dir / f"lumina_oracle_cell_s{shell}.csv").open() as fh:
            rows = list(csv.DictReader(fh))
        by_key = {(r["quantity"], r["Z"], r["stage"]): r for r in rows}
        for row in rows:
            if row["quantity"] != "jbar_representative" or row["status"] != "available":
                continue
            match = re.match(r"line(\d+)_", row["transition"])
            if not match:
                raise ValueError(f"unparseable oracle transition: {row['transition']}")
            line_id = int(match.group(1))
            key = (shell, int(row["Z"]), int(row["stage"]), line_id)
            raw = by_key.get(("jbar_input_raw", row["Z"], row["stage"]))
            beta = by_key.get(("sobolev_beta", row["Z"], row["stage"]))
            if raw is None or beta is None:
                raise ValueError(f"missing oracle companion row for {key}")
            expected[key] = {
                "transition": row["transition"],
                "oracle_j": row["value"],
                "oracle_raw": raw["value"],
                "raw_status": raw["status"],
                "oracle_beta": beta["value"],
            }

    found: dict[tuple[int, int, int, int], dict[str, str]] = {}
    with args.capture.open() as fh:
        for row in csv.DictReader(fh):
            if int(row["iter"]) != args.consumer_iter:
                continue
            key = (int(row["shell"]), int(row["Z"]), int(row["ion"]),
                   int(row["line_idx"]))
            if key in expected:
                if key in found:
                    raise ValueError(f"duplicate captured representative: {key}")
                found[key] = row

    fields = [
        "shell", "Z", "stage", "line_id", "transition", "mode",
        "capture_jbar", "oracle_loaded_raw_jbar", "raw_exact",
        "oracle_production_J", "production_J_exact_when_direct",
        "capture_beta", "oracle_recomputed_beta", "beta_relative_error",
        "beta_within_capture_precision", "status", "note",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for key, oracle in sorted(expected.items()):
            capture = found.get(key)
            if capture is None:
                writer.writerow({
                    "shell": key[0], "Z": key[1], "stage": key[2],
                    "line_id": key[3], "transition": oracle["transition"],
                    "status": "unavailable",
                    "note": "representative line absent from selected capture",
                })
                continue
            cap_j = float(capture["jbar_line"])
            cap_beta = float(capture["beta"])
            oracle_j = float(oracle["oracle_j"])
            oracle_beta = float(oracle["oracle_beta"])
            oracle_raw = (float(oracle["oracle_raw"])
                          if oracle["raw_status"] == "available" and oracle["oracle_raw"]
                          else math.nan)
            mode = int(capture["mode"])
            direct = mode in (1, 2, 3) and cap_j > 0.0
            beta_err = relerr(cap_beta, oracle_beta)
            writer.writerow({
                "shell": key[0], "Z": key[1], "stage": key[2],
                "line_id": key[3], "transition": oracle["transition"],
                "mode": mode, "capture_jbar": f"{cap_j:.17e}",
                "oracle_loaded_raw_jbar": (f"{oracle_raw:.17e}"
                                           if math.isfinite(oracle_raw) else ""),
                "raw_exact": int(math.isfinite(oracle_raw) and cap_j == oracle_raw),
                "oracle_production_J": f"{oracle_j:.17e}",
                "production_J_exact_when_direct": int(direct and cap_j == oracle_j),
                "capture_beta": f"{cap_beta:.17e}",
                "oracle_recomputed_beta": f"{oracle_beta:.17e}",
                "beta_relative_error": f"{beta_err:.17e}",
                "beta_within_capture_precision": int(beta_err <= 5.0e-6),
                "status": "compared",
                "note": ("direct raw-Jbar production branch"
                         if direct else
                         "captured raw Jbar was not the direct production J branch"),
            })

    print(f"[GATEB-CONSISTENCY] expected={len(expected)} found={len(found)} "
          f"wrote={args.out}")


if __name__ == "__main__":
    main()
