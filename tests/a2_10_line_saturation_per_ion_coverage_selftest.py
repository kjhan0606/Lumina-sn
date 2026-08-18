#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts/check_a210_line_saturation_per_ion_coverage.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_reports(root: Path, selected: dict[int, float], repair: int = 0,
                  different_source: bool = False) -> tuple[Path, Path]:
    source = root / "stderr.log"
    source.write_text("sealed source\n", encoding="utf-8")
    other = root / "other.log"
    other.write_text("other source\n", encoding="utf-8")
    totals = {26: 10.0, 27: 20.0, 28: 30.0}
    rows = []
    cumulative = 0.0
    for rank, z in enumerate((26, 27, 28), 1):
        cumulative += selected[z]
        rows.append({
            "rank": rank,
            "line": 100 + rank,
            "Z": z,
            "ion": 3,
            "scaled_emission": selected[z],
            "scaled_emission_serialized": str(selected[z]),
        })
    summary = root / "saturation.json"
    summary.write_text(json.dumps({
        "schema": "lumina-a210-line-saturation-summary-v1",
        "status": "PASS",
        "source_log": str(source.resolve()),
        "source_log_sha256": sha(source),
        "summary": {
            "candidate_rows": 100,
            "selected_rows": 3,
            "total_scaled_emission": 60.0,
            "total_scaled_emission_serialized": "60.0",
            "selected_scaled_emission": cumulative,
            "selected_scaled_emission_serialized": str(cumulative),
        },
        "rows": rows,
        "physical_values_modified": False,
        "floor": 0, "cap": 0, "clamp": 0, "jitter": 0, "repair": repair,
    }) + "\n", encoding="utf-8")
    owner_source = other if different_source else source
    owner = root / "owner.json"
    owner.write_text(json.dumps({
        "schema": "a210-line-ion-owner-diagnostic-v1",
        "status": "PASS",
        "complete": True,
        "phase": "REQUESTED_TE",
        "source_log": str(owner_source.resolve()),
        "source_log_sha256": sha(owner_source),
        "shells": [{
            "shell": 0,
            "owners_by_abs_signed_ion_total": [
                {"Z": z, "ion_stage": 3, "ion_label": 4,
                 "scaled_emission": str(totals[z])}
                for z in (26, 27, 28)
            ],
        }],
        "physical_values_modified": False,
        "floor": 0, "cap": 0, "clamp": 0, "jitter": 0, "repair": 0,
    }) + "\n", encoding="utf-8")
    return summary, owner


def run(summary: Path, owner: Path, report: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([
        "python3", str(SCRIPT),
        "--saturation-summary", str(summary),
        "--owner-report", str(owner),
        "--report", str(report),
    ], cwd=ROOT, text=True, capture_output=True)


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        summary, owner = write_reports(root, {26: 9.0, 27: 18.0, 28: 27.0})
        report = root / "pass.json"
        result = run(summary, owner, report)
        if result.returncode != 0:
            raise SystemExit(f"positive failed: {result.stdout} {result.stderr}")
        payload = json.loads(report.read_text(encoding="utf-8"))
        if payload["status"] != "PASS" or not all(
                item["coverage_pass"] for item in payload["per_ion"]):
            raise SystemExit("per-ion 90-percent positive mismatch")

        under = root / "under"
        under.mkdir()
        under_summary, under_owner = write_reports(
            under, {26: 8.0, 27: 20.0, 28: 30.0}
        )
        under_report = under / "report.json"
        if run(under_summary, under_owner, under_report).returncode != 4:
            raise SystemExit("combined-pass/per-ion-fail fixture accepted")
        under_payload = json.loads(under_report.read_text(encoding="utf-8"))
        if under_payload.get("verdict") != \
                "COMBINED_PREFIX_UNDERCOVERS_AT_LEAST_ONE_TARGET_ION":
            raise SystemExit("undercoverage verdict missing")

        mismatch = root / "mismatch"
        mismatch.mkdir()
        mismatch_summary, mismatch_owner = write_reports(
            mismatch, {26: 9.0, 27: 18.0, 28: 27.0},
            different_source=True,
        )
        if run(mismatch_summary, mismatch_owner,
               mismatch / "report.json").returncode != 4:
            raise SystemExit("different source logs accepted")

        repaired = root / "repaired"
        repaired.mkdir()
        repaired_summary, repaired_owner = write_reports(
            repaired, {26: 9.0, 27: 18.0, 28: 27.0}, repair=1
        )
        if run(repaired_summary, repaired_owner,
               repaired / "report.json").returncode != 4:
            raise SystemExit("repair marker accepted")

        missing = root / "missing"
        missing.mkdir()
        missing_summary, missing_owner = write_reports(
            missing, {26: 9.0, 27: 18.0, 28: 27.0}
        )
        missing_payload = json.loads(missing_owner.read_text(encoding="utf-8"))
        missing_payload["shells"][0]["owners_by_abs_signed_ion_total"].pop()
        missing_owner.write_text(json.dumps(missing_payload) + "\n", encoding="utf-8")
        if run(missing_summary, missing_owner,
               missing / "report.json").returncode != 4:
            raise SystemExit("missing target owner accepted")

    print("PASS a2_10_line_saturation_per_ion_coverage positive+4_negative")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
