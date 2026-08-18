#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts/compare_a210_cmfgen_line_saturation.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vector(z67: float, z68: float) -> str:
    values = [0.5] * 90
    values[66] = z67
    values[67] = z68
    lines = []
    for begin in range(0, 90, 5):
        lines.append(" ".join(f"{value:14.6E}" for value in values[begin:begin + 5]))
    return "\n".join(lines)


def record(line_id: int, label: str, frequency: float,
           lower_full: int, upper_full: int, z67: float, z68: float) -> str:
    return (
        f"{line_id:7d}  {label}(fixture)  {frequency:11.6f} "
        f"1 2 {lower_full} {upper_full}\n{vector(z67, z68)}\n"
    )


def lumina_row(rank: int, line: int, z: int, lower: int, upper: int,
               nu: float, emission: float) -> dict[str, object]:
    return {
        "rank": rank,
        "line": line,
        "Z": z,
        "ion": 3,
        "lower_level": lower,
        "upper_level": upper,
        "nu": nu,
        "tau_raw": 2.0,
        "tau_effective": 2.0,
        "beta": 0.43233235838169365,
        "one_minus_beta": 0.5676676416183064,
        "jbar_over_source": 0.1,
        "Jbar_absolute_bound": 0.01,
        "source_function": 2.0,
        "scaled_emission": emission,
    }


def summary(path: Path, repair: int = 0) -> None:
    payload = {
        "schema": "lumina-a210-line-saturation-summary-v1",
        "status": "PASS",
        "summary": {
            "candidate_rows": 2,
            "selected_rows": 2,
            "selection_target_fraction": 0.9,
            "selected_fraction": 0.95,
            "selected_scaled_emission": 95.0,
        },
        "rows": [
            lumina_row(1, 101, 26, 0, 10, 5.0e15, 60.0),
            lumina_row(2, 202, 27, 1, 11, 6.0e15, 35.0),
        ],
        "physical_values_modified": False,
        "floor": 0,
        "cap": 0,
        "clamp": 0,
        "jitter": 0,
        "repair": repair,
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def coordinate(path: Path, netrate: Path, source_sha: str | None = None) -> None:
    payload = {
        "source": {
            "netrate": str(netrate.resolve()),
            "netrate_sha256": source_sha or sha256(netrate),
        },
        "depths": {"67": {}, "68": {}},
        "shell_zero_velocity_interpolation": {
            "lumina_velocity_km_s": 4264.0,
            "fraction_from_depth_67_to_68": 0.4,
            "line_interpolation_scope": "fixture",
            "interpretation": "STATE_NOT_MATCHED",
        },
        "physical_mutation": 0,
        "floor": 0,
        "cap": 0,
        "clamp": 0,
        "jitter": 0,
        "repair": 0,
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def run(summary_path: Path, netrate: Path, coordinate_path: Path,
        report: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "python3", str(SCRIPT), "--summary", str(summary_path),
            "--netrate", str(netrate), "--coordinate-reference",
            str(coordinate_path), "--report", str(report),
        ],
        cwd=ROOT, text=True, capture_output=True,
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        netrate = root / "NETRATE"
        netrate.write_text(
            record(1, "FeIV", 5.0, 1, 11, 0.1, 0.2) +
            record(2, "CoIV", 6.0, 2, 12, 0.3, 0.4),
            encoding="ascii",
        )
        source = root / "summary.json"
        reference = root / "coordinate.json"
        summary(source)
        coordinate(reference, netrate)
        report = root / "positive.json"
        result = run(source, netrate, reference, report)
        if result.returncode != 0:
            raise SystemExit(f"positive failed: {result.stdout} {result.stderr}")
        payload = json.loads(report.read_text(encoding="utf-8"))
        if payload["status"] != "PASS" or payload["matched_transition_count"] != 2:
            raise SystemExit("positive report mismatch")
        evidence = payload["selected_emission_evidence"]
        if evidence[
            "certified_negative_external_continuum_component"
        ]["fraction_of_selected_emission"] != 1.0:
            raise SystemExit("negative external-component witness was not certified")

        union = root / "summary.union.json"
        union_payload = json.loads(source.read_text(encoding="utf-8"))
        union_payload["summary"]["selection_mode"] = "PER_ION_UNION"
        union_payload["rows"][1]["rank"] = 3
        union_payload["union_metadata"] = [{"line": 101}, {"line": 202}]
        union_payload["union_ion_summaries"] = [
            {"Z": 26}, {"Z": 27}, {"Z": 28},
        ]
        union.write_text(json.dumps(union_payload) + "\n", encoding="utf-8")
        union_result = run(union, netrate, reference, root / "union.json")
        if union_result.returncode != 0:
            raise SystemExit(
                f"union fixture failed: {union_result.stdout} "
                f"{union_result.stderr}"
            )

        consistent = root / "summary.consistent.json"
        consistent_payload = json.loads(source.read_text(encoding="utf-8"))
        for row in consistent_payload["rows"]:
            row["jbar_over_source"] = float(row["one_minus_beta"]) + \
                float(row["beta"]) * 0.2
        consistent.write_text(
            json.dumps(consistent_payload) + "\n", encoding="utf-8"
        )
        consistent_report = root / "consistent.json"
        consistent_result = run(
            consistent, netrate, reference, consistent_report
        )
        if consistent_result.returncode != 0:
            raise SystemExit(
                f"consistent fixture failed: {consistent_result.stdout} "
                f"{consistent_result.stderr}"
            )
        consistent_evidence = json.loads(
            consistent_report.read_text(encoding="utf-8")
        )["selected_emission_evidence"]
        if consistent_evidence[
            "certified_nonnegative_external_continuum_component"
        ]["fraction_of_selected_emission"] != 1.0:
            raise SystemExit("physical nonnegative external component not certified")

        # A value only infinitesimally below the raw Jbar bound must remain
        # indeterminate once evaluation/serialization roundoff is included.
        boundary = root / "summary.boundary.json"
        boundary_payload = json.loads(source.read_text(encoding="utf-8"))
        for row in boundary_payload["rows"]:
            local = float(row["one_minus_beta"])
            ratio_bound = float(row["Jbar_absolute_bound"]) / float(
                row["source_function"]
            )
            row["jbar_over_source"] = math.nextafter(
                local - ratio_bound, -math.inf
            )
        boundary.write_text(json.dumps(boundary_payload) + "\n", encoding="utf-8")
        boundary_report = root / "boundary.json"
        boundary_result = run(boundary, netrate, reference, boundary_report)
        if boundary_result.returncode != 0:
            raise SystemExit(
                f"boundary fixture failed: {boundary_result.stdout} "
                f"{boundary_result.stderr}"
            )
        boundary_evidence = json.loads(
            boundary_report.read_text(encoding="utf-8")
        )["selected_emission_evidence"]
        if boundary_evidence[
            "certified_negative_external_continuum_component"
        ]["line_count"] != 0 or boundary_evidence[
            "external_continuum_component_sign_indeterminate"
        ]["line_count"] != 2:
            raise SystemExit("roundoff-boundary fixture was falsely certified")

        missing = root / "NETRATE.missing"
        missing.write_text(
            record(1, "FeIV", 5.0, 1, 11, 0.1, 0.2), encoding="ascii"
        )
        missing_ref = root / "coordinate.missing.json"
        coordinate(missing_ref, missing)
        if run(source, missing, missing_ref, root / "missing.json").returncode != 4:
            raise SystemExit("missing transition accepted")

        ambiguous = root / "NETRATE.ambiguous"
        ambiguous.write_text(
            netrate.read_text(encoding="ascii") +
            record(3, "FeIV", 5.0, 1, 11, 0.15, 0.25),
            encoding="ascii",
        )
        ambiguous_ref = root / "coordinate.ambiguous.json"
        coordinate(ambiguous_ref, ambiguous)
        if run(
            source, ambiguous, ambiguous_ref, root / "ambiguous.json"
        ).returncode != 4:
            raise SystemExit("ambiguous transition accepted")

        repaired = root / "summary.repaired.json"
        summary(repaired, repair=1)
        if run(repaired, netrate, reference, root / "repair.json").returncode != 4:
            raise SystemExit("repaired summary accepted")

        wrong_sha = root / "coordinate.wrong-sha.json"
        coordinate(wrong_sha, netrate, source_sha="0" * 64)
        if run(source, netrate, wrong_sha, root / "wrong-sha.json").returncode != 4:
            raise SystemExit("wrong NETRATE SHA accepted")

    print(
        "PASS a2_10_cmfgen_line_saturation_comparison "
        "combined+union+physical_identity+roundoff_boundary+4_negative"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
