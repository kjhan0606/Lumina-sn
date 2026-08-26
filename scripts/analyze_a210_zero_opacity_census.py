#!/usr/bin/env python3
"""Parse the A210_ZO_CENSUS hazard-row census without inventing missing data."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


MODE = "A210_ZO_CENSUS"
ROW_PREFIX = "[A2-10][LINE-SATURATION-ZERO-OPACITY-ROW]"
SUMMARY_PREFIX = "[A2-10][LINE-SATURATION-SUMMARY]"
REQUIRED_ROW_FIELDS = (
    "phase", "shell", "line", "Z", "ion", "lower_global", "upper_global",
    "tau_validity", "tau_raw", "effective_tau", "n_upper", "A_ul", "nu",
    "emission_per_sr", "srce_chk", "exact_zero_provenance", "source_defined",
)
ANCHOR_LINE = 669992
ANCHOR_BITS = {
    "tau_validity": 2,
    "n_upper": float("4.437688873013011e-4"),
    "A_ul": float("0.5222556"),
    "emission_per_sr": float("2.7310010625545125e-16"),
}


class CensusError(Exception):
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


def fields(line: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in line.split()[1:]:
        if "=" not in token:
            continue
        name, value = token.split("=", 1)
        if name in result:
            raise CensusError("DUPLICATE_FIELD", name)
        result[name] = value
    return result


def integer(data: dict[str, str], name: str) -> int:
    raw = data.get(name)
    if raw is None:
        raise CensusError("ROW_FIELD_MISSING", name)
    try:
        value = int(raw)
    except ValueError as exc:
        raise CensusError("ROW_FIELD_INVALID", f"{name}={raw}") from exc
    if str(value) != raw:
        raise CensusError("ROW_FIELD_INVALID", f"{name}={raw}")
    return value


def finite(data: dict[str, str], name: str) -> float:
    raw = data.get(name)
    if raw is None:
        raise CensusError("ROW_FIELD_MISSING", name)
    try:
        value = float(raw)
    except ValueError as exc:
        raise CensusError("ROW_FIELD_INVALID", f"{name}={raw}") from exc
    if not math.isfinite(value):
        raise CensusError("ROW_FIELD_INVALID", f"{name}={raw}")
    return value


def parse_row(raw: dict[str, str], ordinal: int) -> dict[str, Any]:
    missing = [name for name in REQUIRED_ROW_FIELDS if name not in raw]
    if missing:
        return {
            "ordinal": ordinal,
            "raw": raw,
            "valid": False,
            "reason": "ROW_FIELD_MISSING",
            "detail": ",".join(missing),
        }
    try:
        parsed: dict[str, Any] = {"phase": raw["phase"]}
        for name in (
            "shell", "line", "Z", "ion", "lower_global", "upper_global",
            "tau_validity", "srce_chk", "exact_zero_provenance", "source_defined",
        ):
            parsed[name] = integer(raw, name)
        for name in ("tau_raw", "effective_tau", "n_upper", "A_ul", "nu", "emission_per_sr"):
            parsed[name] = finite(raw, name)
        if parsed["phase"] != "REQUESTED_TE":
            raise CensusError("ROW_PHASE_INVALID", parsed["phase"])
        if parsed["shell"] != 0:
            raise CensusError("ROW_SHELL_INVALID", str(parsed["shell"]))
        if parsed["ion"] != 1 or parsed["Z"] not in (26, 27, 28):
            raise CensusError("ROW_TARGET_IDENTITY_INVALID",
                              f"Z={parsed['Z']} ion={parsed['ion']}")
        if parsed["tau_raw"] != 0.0 or parsed["n_upper"] <= 0.0:
            raise CensusError("ROW_HAZARD_PREDICATE_INVALID",
                              f"tau_raw={raw['tau_raw']} n_upper={raw['n_upper']}")
    except CensusError as exc:
        return {
            "ordinal": ordinal,
            "raw": raw,
            "valid": False,
            "reason": exc.reason,
            "detail": exc.detail,
        }
    return {"ordinal": ordinal, "raw": raw, "parsed": parsed, "valid": True}


def parse_summary(raw: dict[str, str]) -> tuple[dict[str, int] | None, dict[str, str] | None]:
    required = ("phase", "shell", "target_ion", "candidate_rows", "complete",
                "zero_opacity_emitting_rows")
    missing = [name for name in required if name not in raw]
    if missing:
        return None, {"reason": "SUMMARY_FIELD_MISSING", "detail": ",".join(missing)}
    try:
        parsed = {name: integer(raw, name) for name in (
            "shell", "target_ion", "candidate_rows", "complete",
            "zero_opacity_emitting_rows",
        )}
    except CensusError as exc:
        return None, {"reason": "SUMMARY_FIELD_INVALID", "detail": exc.detail}
    if raw["phase"] != "REQUESTED_TE" or parsed["shell"] != 0 or parsed["target_ion"] != 1:
        return None, {
            "reason": "SUMMARY_IDENTITY_INVALID",
            "detail": f"phase={raw['phase']} shell={parsed['shell']} target_ion={parsed['target_ion']}",
        }
    if parsed["candidate_rows"] < 0 or parsed["zero_opacity_emitting_rows"] < 0:
        return None, {"reason": "SUMMARY_COUNT_INVALID", "detail": "negative count"}
    return parsed, None


def log_blocks(text: str) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    pending: list[dict[str, str]] = []
    blocks: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for line in text.splitlines():
        try:
            if line.startswith(ROW_PREFIX):
                pending.append(fields(line))
            elif line.startswith(SUMMARY_PREFIX):
                raw = fields(line)
                if raw.get("phase") != "REQUESTED_TE" or raw.get("target_ion") != "1":
                    continue
                blocks.append({"rows": pending, "summary_raw": raw})
                pending = []
        except CensusError as exc:
            errors.append({"reason": exc.reason, "detail": exc.detail})
    if pending:
        errors.append({"reason": "ORPHAN_ROWS", "detail": f"count={len(pending)}"})
    return blocks, errors


def analyze_text(text: str) -> dict[str, Any]:
    blocks, errors = log_blocks(text)
    public_blocks: list[dict[str, Any]] = []
    for iteration, block in enumerate(blocks):
        summary, summary_error = parse_summary(block["summary_raw"])
        rows = [parse_row(raw, ordinal) for ordinal, raw in enumerate(block["rows"])]
        reasons: list[str] = []
        if summary_error:
            errors.append({"iteration": iteration, **summary_error})
            reasons.append(summary_error["reason"])
        else:
            assert summary is not None
            if summary["complete"] != 1:
                errors.append({"iteration": iteration, "reason": "INCOMPLETE_SUMMARY",
                               "detail": f"complete={summary['complete']}"})
                reasons.append("INCOMPLETE_SUMMARY")
            observed = len(rows)
            expected = summary["zero_opacity_emitting_rows"]
            if expected > 0 and observed == 0:
                errors.append({"iteration": iteration, "reason": "COUNTER_WITHOUT_ROWS",
                               "detail": f"counter={expected}"})
                reasons.append("COUNTER_WITHOUT_ROWS")
            elif observed != expected:
                errors.append({"iteration": iteration, "reason": "ROW_COUNTER_MISMATCH",
                               "detail": f"rows={observed} counter={expected}"})
                reasons.append("ROW_COUNTER_MISMATCH")
        invalid = [row for row in rows if not row["valid"]]
        for row in invalid:
            errors.append({"iteration": iteration, "ordinal": row["ordinal"],
                           "reason": row["reason"], "detail": row.get("detail", "")})
            reasons.append(row["reason"])
        public_blocks.append({
            "iteration": iteration,
            "summary": summary if summary_error is None else None,
            "summary_raw": block["summary_raw"],
            "row_lines": len(rows),
            "valid_rows": len(rows) - len(invalid),
            "invalid_rows": len(invalid),
            "reasons": sorted(set(reasons)),
            "rows": rows,
        })

    if len(public_blocks) != 2:
        errors.append({"reason": "SUMMARY_ITERATION_COUNT",
                       "detail": f"observed={len(public_blocks)} expected=2"})

    anchor: dict[str, Any] = {"line": ANCHOR_LINE, "present": False, "matches": False,
                              "fields": {}}
    fa: float | None = None
    branch = "UNRESOLVED"
    # Anchor and branch meanings require the complete, parseable row census.
    # A malformed census is reported as such; it must not be reinterpreted as
    # an absent physical anchor or assigned a physical branch.
    if len(public_blocks) == 2 and not errors:
        iter0 = public_blocks[0]
        valid0 = [row["parsed"] for row in iter0["rows"] if row["valid"]]
        anchors = [row for row in valid0 if row["line"] == ANCHOR_LINE]
        anchor["present"] = bool(anchors)
        if not anchors:
            errors.append({"reason": "ANCHOR_ROW_MISSING", "detail": f"line={ANCHOR_LINE}"})
        elif len(anchors) != 1:
            errors.append({"reason": "ANCHOR_ROW_AMBIGUOUS", "detail": f"count={len(anchors)}"})
        else:
            observed = anchors[0]
            comparisons: dict[str, Any] = {}
            for name, expected in ANCHOR_BITS.items():
                actual = observed[name]
                if isinstance(expected, float):
                    matches = actual.hex() == expected.hex()
                    comparisons[name] = {
                        "expected_hex": expected.hex(), "actual_hex": actual.hex(), "matches": matches,
                    }
                else:
                    matches = actual == expected
                    comparisons[name] = {"expected": expected, "actual": actual, "matches": matches}
            anchor["fields"] = comparisons
            anchor["matches"] = all(item["matches"] for item in comparisons.values())
            if not anchor["matches"]:
                errors.append({"reason": "ANCHOR_FIELD_MISMATCH", "detail": f"line={ANCHOR_LINE}"})
        summary0 = iter0["summary"]
        if summary0 is not None and not iter0["invalid_rows"]:
            counter = summary0["zero_opacity_emitting_rows"]
            if counter == 0:
                branch = "ZC-Z"
                errors.append({"reason": "ZERO_OPACITY_ROWS_MISSING", "detail": "iter=0 counter=0"})
            elif counter == len(valid0):
                fa = sum(row["tau_validity"] == 2 for row in valid0) / len(valid0)
                branch = "ZC-A" if fa >= 1.0 else "ZC-B"

    reason_names = sorted({item["reason"] for item in errors})
    status = "PASS" if not reason_names else "FAIL"
    observations: dict[str, Any] = {}
    if public_blocks and public_blocks[0]["summary"] is not None:
        candidate_rows = public_blocks[0]["summary"]["candidate_rows"]
        observations["iter0_candidate_rows"] = candidate_rows
        observations["candidate_rows_exceeds_truncated_l6c"] = candidate_rows > 66110
    return {
        "schema": "A210_ZERO_OPACITY_CENSUS_V1",
        "mode": MODE,
        "status": status,
        "branch": branch,
        "fa": fa,
        "anchor": anchor,
        "iterations": public_blocks,
        "errors": errors,
        "blocking_reasons": reason_names,
        "observations": observations,
        "physical_values_modified": False,
        "clamp": 0,
        "floor": 0,
        "cap": 0,
    }


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    if path.exists() and path.is_symlink():
        raise CensusError("UNSAFE_REPORT_PATH", str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.tmp.", delete=False
    ) as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def synthetic_row() -> str:
    return (
        f"{ROW_PREFIX} phase=REQUESTED_TE shell=0 line={ANCHOR_LINE} Z=26 ion=1 "
        "lower_global=100 upper_global=2542 tau_validity=2 tau_raw=0 effective_tau=0 "
        "n_upper=0.0004437688873013011 A_ul=0.52225560000000004 nu=500000000000000 "
        "emission_per_sr=2.7310010625545125e-16 srce_chk=0 exact_zero_provenance=0 source_defined=0"
    )


def synthetic_log() -> str:
    lines: list[str] = []
    for _ in range(2):
        lines.append(synthetic_row())
        lines.append(
            f"{SUMMARY_PREFIX} phase=REQUESTED_TE shell=0 target_Z=26,27,28 target_ion=1 "
            "candidate_rows=66111 selected_rows=3 total_scaled_emission=1 "
            "selected_scaled_emission=0.9 selected_fraction=0.9 "
            "selection_mode=PER_ION_UNION complete=1 zero_opacity_emitting_rows=1"
        )
    return "\n".join(lines) + "\n"


def replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        raise CensusError("SELFTEST_FIXTURE_INVALID", old)
    return text.replace(old, new, 1)


def selftest() -> int:
    clean = synthetic_log()
    cases = [
        ("NC-Z0", "NONE", clean, None),
        ("NC-Z1", "COUNTER_GT_ZERO_ROW_ZERO",
         "\n".join(line for line in clean.splitlines() if not line.startswith(ROW_PREFIX)) + "\n",
         "COUNTER_WITHOUT_ROWS"),
        ("NC-Z2", "COUNTER_TWO_ROW_ONE",
         replace_once(clean, "zero_opacity_emitting_rows=1", "zero_opacity_emitting_rows=2"),
         "ROW_COUNTER_MISMATCH"),
        ("NC-Z3", "ANCHOR_A_UL_ONE_DIGIT",
         replace_once(clean, "A_ul=0.52225560000000004", "A_ul=0.52225560000000014"),
         "ANCHOR_FIELD_MISMATCH"),
        ("NC-Z4", "COMPLETE_ZERO",
         replace_once(clean, "complete=1", "complete=0"), "INCOMPLETE_SUMMARY"),
    ]
    failed = 0
    for name, injection, text, expected in cases:
        report = analyze_text(text)
        if expected is None:
            ok = report["status"] == "PASS" and report["branch"] == "ZC-A"
            reason = "PASS" if ok else ",".join(report["blocking_reasons"])
            action = "inject=NONE"
            result = "PASS" if ok else "FAIL"
        else:
            ok = report["status"] == "FAIL" and expected in report["blocking_reasons"]
            reason = ",".join(report["blocking_reasons"])
            action = f"inject={injection}"
            result = "FAIL" if ok else "UNEXPECTED_PASS"
        print(f"{name} {action} status={result} reason={reason}")
        if not ok:
            failed += 1
        if expected is not None:
            restored = analyze_text(clean)
            restored_ok = restored["status"] == "PASS" and restored["branch"] == "ZC-A"
            print(f"{name} remove=RESTORE_LOG status={'PASS' if restored_ok else 'FAIL'} "
                  f"reason={'PASS' if restored_ok else ','.join(restored['blocking_reasons'])}")
            if not restored_ok:
                failed += 1
    print(f"A210_ZERO_OPACITY_CENSUS_ANALYZER_SELFTEST {'PASS' if not failed else 'FAIL'} "
          f"controls=NC-Z0,NC-Z1,NC-Z2,NC-Z3,NC-Z4 failed={failed}")
    return 0 if not failed else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--stderr", type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.selftest:
        return selftest()
    try:
        if args.run_root is None and args.stderr is None:
            raise CensusError("RUN_ROOT_OR_STDERR_REQUIRED")
        stderr_path = args.stderr or args.run_root / "stderr.log"
        if not stderr_path.is_file() or stderr_path.is_symlink():
            raise CensusError("MISSING_SAFE_STDERR", str(stderr_path))
        report_path = args.report or (args.run_root / "a210_zero_opacity_census_report.json"
                                      if args.run_root is not None else None)
        if report_path is None:
            raise CensusError("REPORT_REQUIRED_WITH_STDERR")
        report = analyze_text(stderr_path.read_text(encoding="utf-8", errors="strict"))
        atomic_write_json(report_path, report)
    except (CensusError, OSError, UnicodeError) as exc:
        reason = exc.reason if isinstance(exc, CensusError) else type(exc).__name__
        detail = exc.detail if isinstance(exc, CensusError) else str(exc)
        print(f"A210_ZERO_OPACITY_CENSUS_ANALYZER_FAIL reason={reason}"
              + (f" detail={detail}" if detail else ""), file=sys.stderr)
        return 4
    print("A210_ZERO_OPACITY_CENSUS_ANALYZER_RESULT "
          f"status={report['status']} branch={report['branch']} fa={report['fa']} "
          f"report={report_path}")
    return 0 if report["status"] == "PASS" else 4


if __name__ == "__main__":
    raise SystemExit(main())
