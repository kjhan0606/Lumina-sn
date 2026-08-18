#!/usr/bin/env python3
"""Eight required positive/negative controls for byte_parity_compare."""

import sys
import tempfile
from pathlib import Path

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "byte_parity_compare",
    Path(__file__).resolve().parents[1] / "scripts" / "byte_parity_compare.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
EXIT_DIFFERENT = _mod.EXIT_DIFFERENT
EXIT_FAIL_CLOSED = _mod.EXIT_FAIL_CLOSED
EXIT_IDENTICAL = _mod.EXIT_IDENTICAL
compare_runs = _mod.compare_runs


VALUE_PATH = "values/output.csv"
LOG_PATH = "logs/run.log"

BASE_VALUE = (
    b"wavelength_angstrom,flux\n"
    b"500.000000,1.0000000000000000\n"
    b"501.000000,2.0000000000000000\n"
)
BASE_LOG = "EVENT alpha\nEVENT beta\nEVENT gamma\n"


def _write_run(root, value=BASE_VALUE, log=BASE_LOG):
    value_path = root / VALUE_PATH
    log_path = root / LOG_PATH
    value_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    value_path.write_bytes(value)
    log_path.write_text(log, encoding="utf-8")


def _pair(
    parent,
    right_value=BASE_VALUE,
    right_log=BASE_LOG,
    left_value=BASE_VALUE,
    left_log=BASE_LOG,
):
    left = parent / "left"
    right = parent / "right"
    _write_run(left, left_value, left_log)
    _write_run(right, right_value, right_log)
    return left, right


def _assert(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    with tempfile.TemporaryDirectory(prefix="byte-parity-selftest-") as temp_name:
        temp_root = Path(temp_name)

        # P: identical declared value and log files must be an exact PASS.
        left, right = _pair(temp_root / "P")
        report = compare_runs(left, right, [VALUE_PATH], [LOG_PATH])
        _assert(report["exit_code"] == EXIT_IDENTICAL, "P did not pass")

        # B1: one flipped value byte must fail and identify the exact byte offset.
        b1_root = temp_root / "B1"
        left, right = _pair(b1_root)
        flip_offset = BASE_VALUE.index(b"2")
        flipped = bytearray(BASE_VALUE)
        flipped[flip_offset] ^= 1
        right_value_path = right / VALUE_PATH
        right_value_path.write_bytes(bytes(flipped))
        report = compare_runs(left, right, [VALUE_PATH], [LOG_PATH])
        _assert(report["exit_code"] == EXIT_DIFFERENT, "B1 did not fail")
        actual_offset = report["value"]["files"][0]["first_difference"]["offset"]
        _assert(actual_offset == flip_offset, "B1 reported the wrong offset")

        # B2: changing only the final printed numeric digit must fail byte parity.
        b2_value = BASE_VALUE.replace(b"2.0000000000000000", b"2.0000000000000001")
        left, right = _pair(temp_root / "B2", right_value=b2_value)
        report = compare_runs(left, right, [VALUE_PATH], [LOG_PATH])
        _assert(report["exit_code"] == EXIT_DIFFERENT, "B2 did not fail")

        # B3: swapping an otherwise identical multiset of log lines must fail in order.
        b3_log = "EVENT beta\nEVENT alpha\nEVENT gamma\n"
        left, right = _pair(temp_root / "B3", right_log=b3_log)
        report = compare_runs(left, right, [VALUE_PATH], [LOG_PATH])
        _assert(report["exit_code"] == EXIT_DIFFERENT, "B3 did not fail")
        _assert(
            report["log"]["files"][0]["first_difference"]["line"] == 1,
            "B3 did not report the first ordered line",
        )

        # B4: deleting a line must not be treated as filtering and must fail.
        b4_log = "EVENT alpha\nEVENT gamma\n"
        left, right = _pair(temp_root / "B4", right_log=b4_log)
        report = compare_runs(left, right, [VALUE_PATH], [LOG_PATH])
        _assert(report["exit_code"] == EXIT_DIFFERENT, "B4 did not fail")
        _assert(
            report["log"]["files"][0]["first_difference"]["line"] == 2,
            "B4 did not report the deleted-line position",
        )

        # B5: a widened normalization match over T=100.0 must fail closed via the guard.
        b5_log = "SEED=42 T=100.0\n"
        widened_rule = {
            "name": "widened-run-field",
            "regex": r"(?:SEED=\S+|T=\S+)",
            "replacement": "<volatile>",
            "scope": "all-lines",
            "whitelist": [r"re:SEED=\d+"],
        }
        left, right = _pair(
            temp_root / "B5", left_log=b5_log, right_log=b5_log
        )
        report = compare_runs(
            left, right, [VALUE_PATH], [LOG_PATH], [widened_rule]
        )
        _assert(report["exit_code"] == EXIT_FAIL_CLOSED, "B5 did not fail closed")
        _assert(
            report["log"]["files"][0]["normalization_guard"]["pass"] is False,
            "B5 guard did not record a violation",
        )
        b5_violations = report["log"]["files"][0]["normalization_guard"][
            "violations"
        ]
        _assert(
            b5_violations and all(item["token"] == "100.0" for item in b5_violations),
            "B5 did not identify the swallowed physical value 100.0",
        )
        b5_census = report["log"]["files"][0]["rule_census"][0]
        _assert(
            b5_census["left_whitelist_spans_removed"] == 1
            and b5_census["right_whitelist_spans_removed"] == 1,
            "B5 did not exercise whitelist span matching",
        )

        # B5b: a whitelisted timestamp may vary while the physical value stays equal.
        timestamp_rule = {
            "name": "timestamp",
            "pattern": r"2026-[0-9T:\-]+Z",
            "replacement": "<timestamp>",
            "scope": "all-lines",
            "whitelist": [r"re:2026-08-18T[0-9:]+Z"],
        }
        b5b_root = temp_root / "B5b"
        left = b5b_root / "left"
        right = b5b_root / "right"
        _write_run(left, log="ts=2026-08-18T10:00:00Z flux=1.2345\n")
        _write_run(right, log="ts=2026-08-18T11:00:00Z flux=1.2345\n")
        report = compare_runs(
            left, right, [VALUE_PATH], [LOG_PATH], [timestamp_rule]
        )
        _assert(report["exit_code"] == EXIT_IDENTICAL, "B5b did not pass")
        _assert(
            report["log"]["files"][0]["normalization_guard"]["pass"] is True,
            "B5b timestamp whitelist did not pass the guard",
        )
        b5b_census = report["log"]["files"][0]["rule_census"][0]
        _assert(
            b5b_census["left_whitelist_spans_removed"] == 1
            and b5b_census["right_whitelist_spans_removed"] == 1,
            "B5b did not record timestamp whitelist spans",
        )

        # B6: an empty value declaration is an undecidable, fail-closed comparison.
        left, right = _pair(temp_root / "B6")
        report = compare_runs(left, right, [], [])
        _assert(report["exit_code"] == EXIT_FAIL_CLOSED, "B6 did not fail closed")

        # B7: a value present on only one side is never skipped and fails closed.
        b7_root = temp_root / "B7"
        left = b7_root / "left"
        right = b7_root / "right"
        _write_run(left)
        right.mkdir(parents=True, exist_ok=True)
        report = compare_runs(left, right, [VALUE_PATH], [])
        _assert(report["exit_code"] == EXIT_FAIL_CLOSED, "B7 did not fail closed")

    print("PASS byte_parity_compare positive=2 negative_controls=7")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (AssertionError, OSError, ValueError) as exc:
        print("FAIL byte_parity_compare_selftest: %s" % exc, file=sys.stderr)
        sys.exit(1)
