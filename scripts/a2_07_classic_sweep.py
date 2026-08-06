#!/usr/bin/env python3
"""A2-07 classic-debt 17-item evidence collector (diagnostic shadow only)."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ITEMS = {
    "H01": (r"T_e/T_rad|TE_TRAD", "PARTIAL_MEASURED", "A2-10/A2-17"),
    "H04": (r"n_e\[|NE_NOT_CONVERGED", "MEASURED_OPEN", "A2-18"),
    "H05": (r"CE iter|CE converged", "MEASURED_OPEN", "A2-13"),
    "H11": (r"NLTE.*ions|pairs|DR", "MEASURED_OPEN", "A2-13"),
    "H15": (r"nonthermal|Gamma", "MEASURED_OPEN", "A2-10"),
    "S03": (r"partition Z\(T_e\)|A2-07", "PARTIAL_MEASURED", "A2-08"),
    "S04": (r"rate-SE closure|BF-VIEW", "PARTIAL_MEASURED", "A2-13"),
    "S09": (r"recomb|target", "MEASURED_OPEN", "A2-09/A2-13"),
    "S14": (r"super-level|SUPER", "MEASURED_OPEN", "A2-13"),
    "S15": (r"RANK_INCOMPLETE|isolated", "PARTIAL_MEASURED", "A2-13"),
    "S16": (r"timedep|backward-Euler", "MEASURED_OPEN", "A2-18"),
    "G01": (r"sentinel|atomic missing", "MEASURED_OPEN", "A2-13"),
    "G02": (r"zeta", "MEASURED_OPEN", "A2-13"),
    "G03": (r"SOLVE_FAILED|nonfinite|forbidden", "PARTIAL_MEASURED", "A2-13"),
    "G05": (r"synthetic O IV", "MEASURED_OPEN", "A2-13"),
    "G06": (r"SKIP_Z|skip mask", "PARTIAL_MEASURED", "A2-08/A2-13"),
    "G10": (r"start=|seed", "MEASURED_OPEN", "A2-18"),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, action="append", default=[])
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    text = ""
    evidence = []
    for path in args.log:
        try:
            text += path.read_text(encoding="utf-8", errors="replace") + "\n"
            evidence.append(str(path))
        except OSError as exc:
            print(exc)
            return 2
    metrics = {}
    if args.metrics:
        try:
            metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(exc)
            return 2
    if args.self_check:
        text = " ".join(pattern.replace("\\", "") for pattern, _, _ in ITEMS.values())
        evidence = ["synthetic-self-check"]
    rows = []
    for debt_id, (pattern, disposition, followup) in ITEMS.items():
        hits = list(re.finditer(pattern, text, re.I))
        metric = metrics.get(debt_id, {})
        measured = bool(metric)
        rows.append({
            "id": debt_id,
            "fired": bool(hits),
            "hit_count": len(hits),
            "affected_shells/species": metric.get("affected_shells/species", []),
            "impact_metric": metric.get("impact_metric", "BLOCKED_DRIVER_METRIC_NOT_PROVIDED"),
            "impact_value": metric.get("impact_value"),
            "population_path_live": metric.get("population_path_live", bool(hits)),
            "disposition": disposition if measured else "BLOCKED_DRIVER_METRIC_NOT_PROVIDED",
            "evidence": metric.get("evidence", evidence),
            "followup": followup,
        })
    report = {
        "schema": "A2_07_CLASSIC_SWEEP_V1",
        "status": "PASS_SELF_CHECK" if args.self_check else
                  ("PASS" if all(r["impact_value"] is not None for r in rows)
                   else "BLOCKED_CLASSIC_IMPACT_METRICS"),
        "reason_code": "OK" if args.self_check else
                       ("OK" if all(r["impact_value"] is not None for r in rows)
                        else "DRIVER_PAIRED_RUN_REQUIRED"),
        "items": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if args.self_check or report["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
