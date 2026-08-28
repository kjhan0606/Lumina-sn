#!/usr/bin/env python3
"""Fail-closed A210_Z2A_PROBE diagnostic-isolation gate reader.

This reader consumes diagnostic records only.  It neither derives nor judges a
physical value: R-II compares the new capture-on diagnostic stream to the
sealed capture-off census, R-III inventories its prescribed diagnostic rows,
and R-IV checks the zero-hazard diagnostic expectation.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


MODE = "A210_Z2A_PROBE"
A210_INSTRUMENT_REPAIR = True
ZERO_ROW_PREFIX = "[A2-10][LINE-SATURATION-ZERO-OPACITY-ROW]"
ROW_PREFIX = "[A2-10][LINE-SATURATION-ROW]"
SUMMARY_PREFIX = "[A2-10][LINE-SATURATION-SUMMARY]"
BLOCKED_PREFIX = "[A2-10][LINE-SATURATION-BLOCKED]"
ISOLATED_PREFIX = "[A2-10][LINE-SATURATION-ISOLATED]"
EXPECTED_HAZARD_ROWS = 221_382
ANCHOR_LINE = 669_992
ANCHOR_BITS_BY_ITERATION = (
    {
        "tau_validity": "2",
        "n_upper": "0x1.d1534dba20c26p-12",
        "A_ul": "0x1.0b6516044e436p-1",
        "emission_per_sr": "0x1.3adcec9caadc0p-52",
    },
    {
        "tau_validity": "2",
        "n_upper": "0x1.daa141c1395e6p-12",
        "A_ul": "0x1.0b6516044e436p-1",
        "emission_per_sr": "0x1.4128acfd24836p-52",
    },
)


class GateError(RuntimeError):
    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


def fields(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in text.split()[1:]:
        if "=" not in token:
            continue
        name, value = token.split("=", 1)
        if name in result:
            raise GateError("DUPLICATE_FIELD", name)
        result[name] = value
    return result


def integer(record: dict[str, str], name: str) -> int:
    raw = record.get(name)
    if raw is None:
        raise GateError("FIELD_MISSING", name)
    try:
        value = int(raw)
    except ValueError as exc:
        raise GateError("FIELD_INVALID", f"{name}={raw}") from exc
    if str(value) != raw:
        raise GateError("FIELD_INVALID", f"{name}={raw}")
    return value


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() and path.is_symlink():
        raise GateError("UNSAFE_REPORT_PATH", str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.tmp.", delete=False,
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def read_safe(path: Path) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise GateError("MISSING_SAFE_INPUT", str(path))
    return path.read_bytes()


def parse_log(data: bytes, target_ion: int) -> dict[str, Any]:
    blocks: list[dict[str, Any]] = []
    pending_zero: list[bytes] = []
    pending_rows: list[dict[str, str]] = []
    isolated: list[dict[str, str]] = []
    blocked: list[dict[str, str]] = []
    parse_errors: list[str] = []
    try:
        lines = data.splitlines()
        for raw in lines:
            text = raw.decode("utf-8", errors="strict")
            try:
                if text.startswith(ISOLATED_PREFIX):
                    isolated.append(fields(text))
                elif text.startswith(BLOCKED_PREFIX):
                    blocked.append(fields(text))
                elif text.startswith(ZERO_ROW_PREFIX):
                    pending_zero.append(raw)
                elif text.startswith(ROW_PREFIX):
                    pending_rows.append(fields(text))
                elif text.startswith(SUMMARY_PREFIX):
                    summary = fields(text)
                    if (summary.get("phase") == "REQUESTED_TE" and
                            summary.get("target_ion") == str(target_ion)):
                        blocks.append({
                            "summary": summary,
                            "zero_rows": pending_zero,
                            "rows": pending_rows,
                        })
                        pending_zero = []
                        pending_rows = []
            except GateError as exc:
                parse_errors.append(f"{exc.reason}:{exc.detail}")
    except UnicodeError as exc:
        raise GateError("LOG_NOT_UTF8", str(exc)) from exc
    return {
        "blocks": blocks,
        "orphan_zero_rows": len(pending_zero),
        "orphan_rows": len(pending_rows),
        "isolated": isolated,
        "blocked": blocked,
        "parse_errors": parse_errors,
    }


def add_error(errors: list[str], reason: str) -> None:
    if reason not in errors:
        errors.append(reason)


def base_report(mode: str, errors: list[str], branch: str, **extra: Any) -> dict[str, Any]:
    return {
        "schema": "A210_Z2A_PROBE_V1",
        "mode": mode,
        "status": "PASS" if not errors else "FAIL",
        "branch": branch,
        "blocking_reasons": errors,
        "physical_values_modified": False,
        "clamp": 0,
        "floor": 0,
        "cap": 0,
        "jitter": 0,
        "repair": 0,
        **extra,
    }


def safe_number(record: dict[str, str], name: str) -> int | None:
    try:
        return integer(record, name)
    except GateError:
        return None


def validate_summary(summary: dict[str, str], target_ion: int) -> list[str]:
    errors: list[str] = []
    required = ("phase", "shell", "target_ion", "candidate_rows",
                "selected_rows", "complete", "zero_opacity_emitting_rows")
    for name in required:
        if name not in summary:
            add_error(errors, "SUMMARY_FIELD_MISSING")
    if errors:
        return errors
    try:
        if (summary["phase"] != "REQUESTED_TE" or
                integer(summary, "shell") != 0 or
                integer(summary, "target_ion") != target_ion):
            add_error(errors, "SUMMARY_IDENTITY_INVALID")
        if integer(summary, "complete") != 1:
            add_error(errors, "INCOMPLETE_SUMMARY")
        if (integer(summary, "candidate_rows") < 0 or
                integer(summary, "selected_rows") < 0 or
                integer(summary, "zero_opacity_emitting_rows") < 0):
            add_error(errors, "SUMMARY_COUNT_INVALID")
    except GateError:
        add_error(errors, "SUMMARY_FIELD_INVALID")
    return errors


def anchor_matches(raw_rows: list[bytes], iteration: int) -> tuple[bool, str]:
    anchors: list[dict[str, str]] = []
    for raw in raw_rows:
        record = fields(raw.decode("utf-8", errors="strict"))
        if record.get("line") == str(ANCHOR_LINE):
            anchors.append(record)
    if len(anchors) != 1:
        return False, "ANCHOR_MISSING" if not anchors else "ANCHOR_AMBIGUOUS"
    if iteration not in range(len(ANCHOR_BITS_BY_ITERATION)):
        return False, "ANCHOR_ITERATION_INVALID"
    anchor = anchors[0]
    for name, expected in ANCHOR_BITS_BY_ITERATION[iteration].items():
        actual = anchor.get(name)
        if actual is None:
            return False, "ANCHOR_FIELD_MISSING"
        if name == "tau_validity":
            if actual != expected:
                return False, "ANCHOR_FIELD_MISMATCH"
        else:
            try:
                if float(actual).hex() != expected:
                    return False, "ANCHOR_FIELD_MISMATCH"
            except ValueError:
                return False, "ANCHOR_FIELD_INVALID"
    return True, "PASS"


def selected_sequence(rows: list[dict[str, str]]) -> list[tuple[str, str]] | None:
    sequence: list[tuple[str, str]] = []
    for row in rows:
        rank, line = row.get("rank"), row.get("line")
        if rank is None or line is None:
            return None
        sequence.append((rank, line))
    return sequence


def analyze_r2(
    run_data: bytes, census_data: bytes, model_rc: int | None = None,
    expected_hazard_rows: int = EXPECTED_HAZARD_ROWS,
) -> dict[str, Any]:
    run = parse_log(run_data, 1)
    census = parse_log(census_data, 1)
    errors: list[str] = []
    if model_rc is not None and model_rc != 0:
        add_error(errors, "MODEL_NOT_COMPLETE")
    if run["isolated"]:
        add_error(errors, "ISOLATED_PRESENT")
        return base_report("R-II", errors, "ZI-X", f_bit=None,
                           isolated_count=len(run["isolated"]),
                           blocked_count=len(run["blocked"]))
    if run["blocked"]:
        add_error(errors, "BLOCKED_PRESENT")
        return base_report("R-II", errors, "ZI-D", f_bit=None,
                           isolated_count=0, blocked_count=len(run["blocked"]))
    if census["isolated"] or census["blocked"]:
        add_error(errors, "CENSUS_REFERENCE_NOT_COMPLETE")
    if run["parse_errors"] or census["parse_errors"]:
        add_error(errors, "LOG_PARSE_ERROR")
    if (len(run["blocks"]) != 2 or len(census["blocks"]) != 2 or
            run["orphan_zero_rows"] or run["orphan_rows"] or
            census["orphan_zero_rows"] or census["orphan_rows"]):
        add_error(errors, "SUMMARY_COUNT_OR_TRUNCATION")
        return base_report("R-II", errors, "ZI-D", f_bit=None,
                           isolated_count=0, blocked_count=0)

    matches = 0
    total = 0
    observations: list[dict[str, Any]] = []
    for iteration, (actual, reference) in enumerate(zip(run["blocks"], census["blocks"])):
        actual_summary, reference_summary = actual["summary"], reference["summary"]
        for reason in validate_summary(actual_summary, 1):
            add_error(errors, reason)
        for reason in validate_summary(reference_summary, 1):
            add_error(errors, f"CENSUS_{reason}")
        if "sprobe_undefined_rows" not in actual_summary:
            add_error(errors, "SPROBE_COUNTER_MISSING")
        else:
            sprobe = safe_number(actual_summary, "sprobe_undefined_rows")
            zero = safe_number(actual_summary, "zero_opacity_emitting_rows")
            if sprobe is None or zero is None:
                add_error(errors, "SPROBE_COUNTER_INVALID")
            elif sprobe != zero or sprobe != expected_hazard_rows:
                add_error(errors, "SPROBE_COUNTER_MISMATCH")
        shared_actual = {k: v for k, v in actual_summary.items()
                         if k != "sprobe_undefined_rows"}
        if shared_actual != reference_summary:
            add_error(errors, "SUMMARY_TOKEN_MISMATCH")
        actual_zero = actual["zero_rows"]
        reference_zero = reference["zero_rows"]
        if len(actual_zero) != expected_hazard_rows or len(reference_zero) != expected_hazard_rows:
            add_error(errors, "ROW_COUNT_MISMATCH")
        pair_total = max(len(actual_zero), len(reference_zero))
        pair_matches = sum(1 for left, right in zip(actual_zero, reference_zero)
                           if left == right)
        matches += pair_matches
        total += pair_total
        if pair_matches != pair_total:
            add_error(errors, "ROW_STREAM_MISMATCH")
        actual_selected = selected_sequence(actual["rows"])
        reference_selected = selected_sequence(reference["rows"])
        if (actual_selected is None or reference_selected is None or
                actual_selected != reference_selected):
            add_error(errors, "SELECTION_SEQUENCE_MISMATCH")
        for raw_rows in (actual_zero, reference_zero):
            ok, reason = anchor_matches(raw_rows, iteration)
            if not ok:
                add_error(errors, reason)
        observations.append({
            "iteration": iteration,
            "zero_rows": len(actual_zero),
            "selected_rows": len(actual["rows"]),
            "summary_sprobe_undefined_rows": actual_summary.get("sprobe_undefined_rows"),
        })
    f_bit = matches / total if total else 0.0
    branch = "ZI-A" if not errors and f_bit >= 1.0 else "ZI-B"
    if "MODEL_NOT_COMPLETE" in errors:
        branch = "ZI-D"
    return base_report("R-II", errors, branch, f_bit=f_bit,
                       isolated_count=0, blocked_count=0,
                       iterations=observations)


def analyze_r3_or_r4(run_data: bytes, mode: str, model_rc: int | None = None) -> dict[str, Any]:
    target_ion = 2 if mode == "R-III" else 3
    run = parse_log(run_data, target_ion)
    errors: list[str] = []
    if model_rc is not None and model_rc != 0:
        add_error(errors, "MODEL_NOT_COMPLETE")
    if run["isolated"]:
        add_error(errors, "ISOLATED_PRESENT")
        return base_report(mode, errors, "ZI-X", isolated_count=len(run["isolated"]),
                           blocked_count=len(run["blocked"]))
    if run["blocked"]:
        add_error(errors, "BLOCKED_PRESENT")
        return base_report(mode, errors, "ZI-D", isolated_count=0,
                           blocked_count=len(run["blocked"]))
    if run["parse_errors"]:
        add_error(errors, "LOG_PARSE_ERROR")
    if (len(run["blocks"]) != 2 or run["orphan_zero_rows"] or
            run["orphan_rows"]):
        add_error(errors, "SUMMARY_COUNT_OR_TRUNCATION")
        return base_report(mode, errors, "ZI-D", isolated_count=0, blocked_count=0)

    stage4_rows = 0
    hazards = 0
    hazard_by_z: dict[str, int] = {}
    anchor_262210 = False
    for block in run["blocks"]:
        summary = block["summary"]
        for reason in validate_summary(summary, target_ion):
            add_error(errors, reason)
        zero_count = safe_number(summary, "zero_opacity_emitting_rows")
        if zero_count is None or zero_count != len(block["zero_rows"]):
            add_error(errors, "HAZARD_COUNTER_MISMATCH")
        sprobe = safe_number(summary, "sprobe_undefined_rows")
        if sprobe is None:
            add_error(errors, "SPROBE_COUNTER_MISSING")
        elif zero_count is None or sprobe != zero_count:
            add_error(errors, "SPROBE_COUNTER_MISMATCH")
        for row in block["rows"]:
            required = ("producer_continuum_term", "producer_local_emission_term",
                        "producer_terms_defined")
            if all(name in row for name in required):
                stage4_rows += 1
        for raw in block["zero_rows"]:
            row = fields(raw.decode("utf-8", errors="strict"))
            hazards += 1
            z = row.get("Z", "UNAVAILABLE")
            hazard_by_z[z] = hazard_by_z.get(z, 0) + 1
            if row.get("line") == "262210":
                anchor_262210 = True
    if mode == "R-III" and stage4_rows < 1:
        add_error(errors, "STAGE4_ROW_MISSING")
    if mode == "R-IV":
        if hazards != 0:
            add_error(errors, "RIV_HAZARD_ROWS_NONZERO")
    branch = "ZI-A" if not errors else "ZI-P"
    if "MODEL_NOT_COMPLETE" in errors:
        branch = "ZI-D"
    return base_report(mode, errors, branch, isolated_count=0, blocked_count=0,
                       stage4_rows=stage4_rows, hazard_rows=hazards,
                       hazard_rows_by_Z=hazard_by_z,
                       line_262210_present=anchor_262210)


def synthetic_zero_row(iteration: int) -> str:
    if iteration == 0:
        n_upper = "0.0004437688873013011"
        emission_per_sr = "2.7310010625545125e-16"
    elif iteration == 1:
        n_upper = "0.00045264235457262295"
        emission_per_sr = "2.7856093265405119e-16"
    else:
        raise GateError("ANCHOR_ITERATION_INVALID", str(iteration))
    return (
        f"{ZERO_ROW_PREFIX} phase=REQUESTED_TE shell=0 line={ANCHOR_LINE} Z=26 ion=1 "
        "lower_global=1 upper_global=2 tau_validity=2 tau_raw=0 effective_tau=0 "
        f"n_upper={n_upper} A_ul=0.52225560000000004 "
        f"nu=5e14 emission_per_sr={emission_per_sr} srce_chk=0 "
        "exact_zero_provenance=1 source_defined=0"
    )


def synthetic_selected_row() -> str:
    return (
        f"{ROW_PREFIX} phase=REQUESTED_TE shell=0 rank=1 line={ANCHOR_LINE} "
        "Z=26 ion=1 producer_continuum_term=1 producer_local_emission_term=1 "
        "producer_terms_defined=1"
    )


def synthetic_summary(include_sprobe: bool) -> str:
    suffix = " sprobe_undefined_rows=1" if include_sprobe else ""
    return (
        f"{SUMMARY_PREFIX} phase=REQUESTED_TE shell=0 target_Z=26,27,28 target_ion=1 "
        "candidate_rows=1 selected_rows=1 total_scaled_emission=1 "
        "selected_scaled_emission=1 selected_fraction=1 selection_target_fraction=0.9 "
        "selection_mode=PER_ION_UNION complete=1 interpretation=DIAGNOSTIC_ONLY "
        "physical_values_modified=0 clamp=0 floor=0 cap=0 jitter=0 repair=0 "
        f"zero_opacity_emitting_rows=1{suffix}"
    )


def synthetic_logs() -> tuple[str, str]:
    run_lines: list[str] = []
    census_lines: list[str] = []
    for iteration in range(2):
        run_lines.extend((synthetic_zero_row(iteration), synthetic_selected_row(),
                          synthetic_summary(True)))
        census_lines.extend((synthetic_zero_row(iteration), synthetic_selected_row(),
                             synthetic_summary(False)))
    return "\n".join(run_lines) + "\n", "\n".join(census_lines) + "\n"


def report_outside_census(report: Path, census_stderr: Path) -> None:
    report_real = Path(os.path.realpath(report))
    census_parent_real = Path(os.path.realpath(census_stderr.parent))
    try:
        report_real.relative_to(census_parent_real)
    except ValueError:
        pass
    else:
        raise GateError("REPORT_INSIDE_SEALED_INPUT", str(report_real))
    if report.exists() and report.is_symlink():
        raise GateError("UNSAFE_REPORT_PATH", str(report))


def census_selfcheck(census_stderr: Path) -> dict[str, Any]:
    if not census_stderr.is_file() or census_stderr.is_symlink():
        raise GateError("MISSING_SAFE_INPUT", str(census_stderr))
    blocks: list[tuple[dict[str, str], list[bytes]]] = []
    anchors: list[bytes] = []
    anchor_token = f" line={ANCHOR_LINE} ".encode("ascii")
    try:
        with census_stderr.open("rb") as stream:
            for with_newline in stream:
                raw = with_newline.rstrip(b"\r\n")
                if raw.startswith(ZERO_ROW_PREFIX.encode("ascii")):
                    if anchor_token in raw:
                        anchors.append(raw)
                elif raw.startswith(SUMMARY_PREFIX.encode("ascii")):
                    summary = fields(raw.decode("utf-8", errors="strict"))
                    blocks.append((summary, anchors))
                    anchors = []
    except UnicodeError as exc:
        raise GateError("LOG_NOT_UTF8", str(exc)) from exc
    except OSError as exc:
        raise GateError("LOG_READ_FAILED", str(exc)) from exc
    if anchors:
        raise GateError("ANCHOR_AMBIGUOUS", "anchor_after_final_summary")
    if len(blocks) != 2:
        raise GateError("SUMMARY_COUNT_MISMATCH", f"count={len(blocks)}")
    observations: list[dict[str, Any]] = []
    for iteration, (summary, raw_rows) in enumerate(blocks):
        if summary.get("complete") != "1":
            raise GateError("SUMMARY_TOKEN_MISMATCH", f"iter={iteration},complete")
        ok, reason = anchor_matches(raw_rows, iteration)
        if not ok:
            raise GateError(reason, f"iter={iteration}")
        observations.append({"iteration": iteration, "complete": "1"})
    return base_report(
        "CENSUS-SELFCHECK", [], "ZI-A", summary_count=len(blocks),
        summaries=observations, anchor_line=ANCHOR_LINE,
        anchor_bits_by_iteration=ANCHOR_BITS_BY_ITERATION,
        input_opened_read_only=True, streaming_log_reader=True,
    )


def replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        raise GateError("SELFTEST_FIXTURE_INVALID", old)
    return text.replace(old, new, 1)


def control_line(name: str, action: str, report: dict[str, Any]) -> str:
    reasons = ",".join(report["blocking_reasons"]) or "PASS"
    return (f"{name} {action} status={report['status']} reason={reasons} "
            f"branch={report['branch']}")


def selftest() -> int:
    clean, census = synthetic_logs()
    cases = [
        ("NC-I0", "NONE", clean, "ZI-A", "PASS"),
        ("NC-I1", "ISOLATED", f"{ISOLATED_PREFIX} reason=ALLOCATION_FAILED phase=REQUESTED_TE "
         "rows_at_invalidation=0 zero_opacity_emitting_rows=0 sprobe_undefined_rows=0 "
         "physics_transaction=CONTINUED complete=0 interpretation=DIAGNOSTIC_ONLY "
         "physical_values_modified=0 clamp=0 floor=0 cap=0 jitter=0 repair=0\n{clean}",
         "ZI-X", "FAIL"),
        ("NC-I2", "ROW_FIELD_TAMPER", replace_once(clean, "n_upper=0.0004437688873013011",
         "n_upper=0.0004437688873013012"), "ZI-B", "FAIL"),
        ("NC-I3", "SUMMARY_TOKEN_TAMPER", replace_once(clean, "candidate_rows=1", "candidate_rows=2"),
         "ZI-B", "FAIL"),
        ("NC-I4", "TRUNCATE_FINAL_SUMMARY", clean.rsplit("\n", 2)[0] + "\n",
         "ZI-D", "FAIL"),
        ("NC-I5", "BLOCKED", f"{BLOCKED_PREFIX} reason=INVALID_ROW_PROVENANCE phase=REQUESTED_TE "
         "candidate_rows=0 target_Z=26,27,28 target_ion=1 complete=0\n{clean}",
         "ZI-D", "FAIL"),
        ("NC-I6", "REMOVE_ANCHOR", "\n".join(
         line for line in clean.splitlines() if not line.startswith(ZERO_ROW_PREFIX)) + "\n",
         "ZI-B", "FAIL"),
    ]
    failed = 0
    for name, injection, text, expected_branch, expected_status in cases:
        report = analyze_r2(text.encode(), census.encode(), expected_hazard_rows=1)
        print(control_line(name, f"inject={injection}", report))
        if report["status"] != expected_status or report["branch"] != expected_branch:
            failed += 1
        if injection != "NONE":
            restored = analyze_r2(clean.encode(), census.encode(), expected_hazard_rows=1)
            print(control_line(name, "remove=RESTORE_LOG", restored))
            if restored["status"] != "PASS" or restored["branch"] != "ZI-A":
                failed += 1

    iter1 = synthetic_zero_row(1)
    iter0 = synthetic_zero_row(0)
    injected_run = replace_once(clean, iter1, iter0)
    injected_census = replace_once(census, iter1, iter0)
    injected = analyze_r2(injected_run.encode(), injected_census.encode(),
                          expected_hazard_rows=1)
    print(control_line("NC-I7", "inject=ITER1_WITH_ITER0_ANCHOR", injected))
    if (injected["status"] != "FAIL" or
            injected["blocking_reasons"] != ["ANCHOR_FIELD_MISMATCH"]):
        failed += 1
    restored = analyze_r2(clean.encode(), census.encode(), expected_hazard_rows=1)
    print(control_line("NC-I7", "remove=RESTORE_ITER1_ANCHOR", restored))
    if restored["status"] != "PASS" or restored["branch"] != "ZI-A":
        failed += 1

    with tempfile.TemporaryDirectory(prefix="z2a-i8-") as directory:
        root = Path(directory)
        census_root = root / "census"
        census_root.mkdir()
        census_path = census_root / "stderr.log"
        report_path = root / "census-selfcheck.json"
        census_path.write_text(census, encoding="utf-8")
        command = [
            sys.executable, str(Path(__file__).resolve()),
            "--census-selfcheck", str(census_path), "--report", str(report_path),
        ]
        positive = subprocess.run(command, capture_output=True, text=True)
        positive_payload = json.loads(report_path.read_text(encoding="utf-8"))
        positive_unchanged = census_path.read_text(encoding="utf-8") == census
        positive_ok = (positive.returncode == 0 and
                       positive_payload.get("status") == "PASS" and positive_unchanged)
        if not positive_ok:
            failed += 1
        print("NC-I8 positive=DISTINCT_ITERATION_ANCHORS "
              f"status={positive_payload.get('status', 'FAIL')} "
              f"reason={'PASS' if positive_ok else 'CENSUS_SELFCHECK_FAILED'} "
              f"branch={positive_payload.get('branch', 'ABSENT')} "
              f"byte_unchanged={int(positive_unchanged)} streaming=1")
        mutated_census = replace_once(census, iter1, iter0)
        census_path.write_text(mutated_census, encoding="utf-8")
        negative = subprocess.run(command, capture_output=True, text=True)
        negative_payload = json.loads(report_path.read_text(encoding="utf-8"))
        negative_reasons = negative_payload.get("blocking_reasons", [])
        negative_reason = negative_reasons[0] if negative_reasons else "ABSENT"
        negative_unchanged = census_path.read_text(encoding="utf-8") == mutated_census
        if (negative.returncode != 4 or negative_payload.get("status") != "FAIL" or
                negative_reason != "ANCHOR_FIELD_MISMATCH" or not negative_unchanged):
            failed += 1
        print("NC-I8 inject=ITER1_WITH_ITER0_ANCHOR "
              f"status={negative_payload.get('status', 'FAIL')} "
              f"reason={negative_reason} branch={negative_payload.get('branch', 'ABSENT')} "
              f"byte_unchanged={int(negative_unchanged)} streaming=1")
        census_path.write_text(census, encoding="utf-8")
        restored_cli = subprocess.run(command, capture_output=True, text=True)
        restored_payload = json.loads(report_path.read_text(encoding="utf-8"))
        restored_unchanged = census_path.read_text(encoding="utf-8") == census
        restored_ok = (restored_cli.returncode == 0 and
                       restored_payload.get("status") == "PASS" and restored_unchanged)
        if not restored_ok:
            failed += 1
        print("NC-I8 remove=RESTORE_ITER1_ANCHOR "
              f"status={restored_payload.get('status', 'FAIL')} "
              f"reason={'PASS' if restored_ok else 'CENSUS_SELFCHECK_FAILED'} "
              f"branch={restored_payload.get('branch', 'ABSENT')} "
              f"byte_unchanged={int(restored_unchanged)} streaming=1")
    print("A210_Z2A_PROBE_ANALYZER_SELFTEST "
          f"{'PASS' if not failed else 'FAIL'} "
          "controls=NC-I0,NC-I1,NC-I2,NC-I3,NC-I4,NC-I5,NC-I6,NC-I7,NC-I8 "
          f"failed={failed}")
    return 0 if not failed else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--census-selfcheck", type=Path)
    parser.add_argument("--mode", choices=("R-II", "R-III", "R-IV"))
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--stderr", type=Path)
    parser.add_argument("--census-stderr", type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def model_rc(run_root: Path | None) -> int | None:
    if run_root is None:
        return None
    path = run_root / "model.rc"
    raw = read_safe(path).decode("ascii", errors="strict").strip()
    try:
        return int(raw)
    except ValueError as exc:
        raise GateError("MODEL_RC_INVALID", raw) from exc


def main() -> int:
    args = parse_args()
    census_mode_ambiguous = (
        args.census_selfcheck is not None and
        (args.selftest or args.mode is not None or args.run_root is not None or
         args.stderr is not None or args.census_stderr is not None)
    )
    if census_mode_ambiguous:
        print("A210_Z2A_PROBE_ANALYZER_FAIL reason=MODE_AMBIGUOUS", file=sys.stderr)
        return 4
    if args.selftest:
        return selftest()
    if args.census_selfcheck is not None:
        if args.report is None:
            print("A210_Z2A_PROBE_CENSUS_SELFCHECK_FAIL reason=REPORT_REQUIRED",
                  file=sys.stderr)
            return 4
        report_safe = False
        try:
            report_outside_census(args.report, args.census_selfcheck)
            report_safe = True
            report = census_selfcheck(args.census_selfcheck)
            atomic_write_json(args.report, report)
        except (GateError, OSError, UnicodeError) as exc:
            reason = exc.reason if isinstance(exc, GateError) else type(exc).__name__
            detail = exc.detail if isinstance(exc, GateError) else str(exc)
            if report_safe:
                failure = base_report(
                    "CENSUS-SELFCHECK", [reason], "ZI-B", detail=detail,
                    input_write_attempted=False, streaming_log_reader=True,
                )
                try:
                    atomic_write_json(args.report, failure)
                except (GateError, OSError):
                    pass
            print("A210_Z2A_PROBE_CENSUS_SELFCHECK_FAIL "
                  f"reason={reason}" + (f" detail={detail}" if detail else ""),
                  file=sys.stderr)
            return 4
        print("A210_Z2A_PROBE_CENSUS_SELFCHECK_RESULT "
              f"status=PASS anchor_line={ANCHOR_LINE} report={args.report} "
              "input_access=READ_ONLY streaming=1")
        return 0
    try:
        if args.mode is None or (args.run_root is None and args.stderr is None):
            raise GateError("MODE_AND_RUN_INPUT_REQUIRED")
        stderr_path = args.stderr or args.run_root / "stderr.log"
        report_path = args.report or (args.run_root / "a210_z2a_probe_report.json"
                                      if args.run_root is not None else None)
        if report_path is None:
            raise GateError("REPORT_REQUIRED_WITH_STDERR")
        data = read_safe(stderr_path)
        rc = model_rc(args.run_root)
        if args.mode == "R-II":
            if args.census_stderr is None:
                raise GateError("CENSUS_STDERR_REQUIRED")
            report = analyze_r2(data, read_safe(args.census_stderr), rc)
        else:
            report = analyze_r3_or_r4(data, args.mode, rc)
        atomic_write_json(report_path, report)
    except (GateError, OSError, UnicodeError) as exc:
        reason = exc.reason if isinstance(exc, GateError) else type(exc).__name__
        detail = exc.detail if isinstance(exc, GateError) else str(exc)
        print("A210_Z2A_PROBE_ANALYZER_FAIL "
              f"reason={reason}" + (f" detail={detail}" if detail else ""), file=sys.stderr)
        return 4
    print("A210_Z2A_PROBE_ANALYZER_RESULT "
          f"status={report['status']} branch={report['branch']} report={report_path}")
    return 0 if report["status"] == "PASS" else 4


if __name__ == "__main__":
    raise SystemExit(main())
