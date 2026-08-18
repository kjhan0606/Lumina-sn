#!/usr/bin/env python3
"""Fail-closed byte-parity comparator for value files and ordered logs.

Value files are compared as bytes.  They are never interpreted as CSV (or as
any other structured format).  Logs are the only inputs which can be
normalized, and their normalized lines are compared in declaration order.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


EXIT_IDENTICAL = 0
EXIT_DIFFERENT = 1
EXIT_FAIL_CLOSED = 2

_NUMBER_TOKEN = re.compile(
    r"(?<![A-Za-z_])(?:[-+]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))"
    r"(?:[eE][-+]?\d+)?|[-+]?(?:nan|inf(?:inity)?))(?![A-Za-z_])",
    re.IGNORECASE,
)


class ComparatorConfigError(ValueError):
    """Raised when a declared comparison input is not a valid declaration."""


class NormalizationRule:
    """A named, scoped log normalization rule.

    A scope is either ``all-lines``/``line`` or an object containing a
    ``line_regex`` selector.  Whitelist entries are exact numeric tokens by
    default; entries beginning with ``re:`` are regular expressions.  Both
    forms are matched as spans inside each normalization match.  Those spans
    are removed before the numeric-token guard scans the remainder.
    """

    def __init__(self, name, pattern, replacement, scope, whitelist):
        if not isinstance(name, str) or not name:
            raise ComparatorConfigError("normalization rule name is required")
        if not isinstance(pattern, str) or not pattern:
            raise ComparatorConfigError(
                "normalization rule %r needs a non-empty regex" % name
            )
        if not isinstance(replacement, str):
            raise ComparatorConfigError(
                "normalization rule %r replacement must be a string" % name
            )
        if not isinstance(whitelist, list):
            raise ComparatorConfigError(
                "normalization rule %r whitelist must be a list" % name
            )

        try:
            compiled = re.compile(pattern)
        except re.error as exc:
            raise ComparatorConfigError(
                "invalid regex for rule %r: %s" % (name, exc)
            ) from exc

        whitelist_patterns = []
        for entry in whitelist:
            if not isinstance(entry, str) or not entry:
                raise ComparatorConfigError(
                    "rule %r has a non-string/empty whitelist entry" % name
                )
            expression = entry[3:] if entry.startswith("re:") else re.escape(entry)
            try:
                whitelist_patterns.append(re.compile(expression))
            except re.error as exc:
                raise ComparatorConfigError(
                    "invalid whitelist regex for rule %r: %s" % (name, exc)
                ) from exc

        self.name = name
        self.pattern_text = pattern
        self.regex = compiled
        self.replacement = replacement
        self.scope = scope
        self.whitelist = list(whitelist)
        self.whitelist_patterns = whitelist_patterns

        if isinstance(scope, dict):
            line_regex = scope.get("line_regex")
            if not isinstance(line_regex, str) or not line_regex:
                raise ComparatorConfigError(
                    "rule %r scope needs a non-empty line_regex" % name
                )
            try:
                self.scope_regex = re.compile(line_regex)
            except re.error as exc:
                raise ComparatorConfigError(
                    "invalid scope regex for rule %r: %s" % (name, exc)
                ) from exc
            self.scope_kind = "line_regex"
        elif isinstance(scope, str) and scope:
            if scope.startswith("line_regex:"):
                line_regex = scope[len("line_regex:") :]
                if not line_regex:
                    raise ComparatorConfigError(
                        "rule %r has an empty line_regex scope" % name
                    )
                try:
                    self.scope_regex = re.compile(line_regex)
                except re.error as exc:
                    raise ComparatorConfigError(
                        "invalid scope regex for rule %r: %s" % (name, exc)
                    ) from exc
                self.scope_kind = "line_regex"
            elif scope in ("all-lines", "line", "each-line"):
                self.scope_regex = None
                self.scope_kind = "all-lines"
            else:
                raise ComparatorConfigError(
                    "rule %r scope must be all-lines, line, each-line, "
                    "or a line_regex selector" % name
                )
        else:
            raise ComparatorConfigError("rule %r needs a declared scope" % name)

    @classmethod
    def from_mapping(cls, mapping):
        if not isinstance(mapping, dict):
            raise ComparatorConfigError("each normalization rule must be an object")
        name = mapping.get("name")
        pattern = mapping.get("regex", mapping.get("pattern"))
        replacement = mapping.get("replacement")
        if replacement is None:
            if isinstance(name, str) and name:
                replacement = "<%s>" % name
            else:
                replacement = "<NORMALIZED>"
        scope = mapping.get("scope")
        whitelist = mapping.get("whitelist", [])
        return cls(name, pattern, replacement, scope, whitelist)

    def applies_to(self, line):
        if self.scope_kind == "all-lines":
            return True
        return self.scope_regex.search(line) is not None

    def whitelist_spans(self, matched_text):
        """Return non-overlapping whitelist-covered character spans.

        A whitelist is a declaration of text regions, not a pre-tokenization
        allow-list.  Merge overlapping matches so census values describe the
        actual removed regions rather than counting the same bytes twice.
        """

        candidates = []
        for expression in self.whitelist_patterns:
            for match in expression.finditer(matched_text):
                if match.start() != match.end():
                    candidates.append((match.start(), match.end()))

        candidates.sort(key=lambda span: (span[0], span[1]))
        spans = []
        for start, end in candidates:
            if not spans or start >= spans[-1][1]:
                spans.append([start, end])
            elif end > spans[-1][1]:
                spans[-1][1] = end
        return [(start, end) for start, end in spans]

    def as_report(self):
        return {
            "name": self.name,
            "regex": self.pattern_text,
            "replacement": self.replacement,
            "scope": self.scope,
            "whitelist": list(self.whitelist),
        }


def _coerce_rules(raw_rules):
    if raw_rules is None:
        raw_rules = []
    if not isinstance(raw_rules, (list, tuple)):
        raise ComparatorConfigError("normalization rules must be a list")

    rules = []
    for raw_rule in raw_rules:
        if isinstance(raw_rule, NormalizationRule):
            rule = raw_rule
        else:
            rule = NormalizationRule.from_mapping(raw_rule)
        for previous in rules:
            if previous.name == rule.name:
                raise ComparatorConfigError(
                    "duplicate normalization rule name: %s" % rule.name
                )
        rules.append(rule)
    return rules


def _coerce_declared_paths(raw_paths, label):
    if raw_paths is None:
        return []
    if isinstance(raw_paths, (str, Path)):
        raw_paths = [raw_paths]
    try:
        paths = list(raw_paths)
    except TypeError as exc:
        raise ComparatorConfigError("%s file list is not iterable" % label) from exc

    result = []
    for raw_path in paths:
        if not isinstance(raw_path, (str, Path)):
            raise ComparatorConfigError("%s file paths must be strings" % label)
        path = Path(raw_path)
        if path.is_absolute() or not path.parts or path == Path("."):
            raise ComparatorConfigError(
                "%s paths must be non-empty relative paths: %r" % (label, raw_path)
            )
        if ".." in path.parts:
            raise ComparatorConfigError(
                "%s path escapes its declared root: %r" % (label, raw_path)
            )
        text_path = path.as_posix()
        for previous in result:
            if previous == text_path:
                raise ComparatorConfigError(
                    "duplicate declared %s path: %s" % (label, text_path)
                )
        result.append(text_path)
    return result


def _resolve_under_root(root, relative_path):
    try:
        root_path = Path(root).resolve(strict=True)
        if not root_path.is_dir():
            raise ComparatorConfigError("comparison root is not a directory: %s" % root)
        candidate = (root_path / relative_path).resolve(strict=False)
        candidate.relative_to(root_path)
        return candidate
    except ComparatorConfigError:
        raise
    except (OSError, ValueError) as exc:
        raise ComparatorConfigError(
            "cannot resolve %s under root %s: %s" % (relative_path, root, exc)
        ) from exc


def _read_bytes(path):
    try:
        return path.read_bytes(), None
    except OSError as exc:
        return None, "%s: %s" % (type(exc).__name__, exc)
    except Exception as exc:  # a comparator error is fail-closed
        return None, "%s: %s" % (type(exc).__name__, exc)


def _read_text_lines(path):
    data, error = _read_bytes(path)
    if error is not None:
        return None, None, error
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        return None, None, "UnicodeDecodeError: %s" % exc
    return data, text.splitlines(keepends=True), None


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _first_difference(left, right):
    common_length = len(left)
    if len(right) < common_length:
        common_length = len(right)
    offset = None
    for index in range(common_length):
        if left[index] != right[index]:
            offset = index
            break
    if offset is None and len(left) != len(right):
        offset = common_length
    return offset


def _hex_window(data, offset):
    start = offset - 32
    if start < 0:
        start = 0
    end = offset + 33
    if end > len(data):
        end = len(data)
    window = data[start:end]
    ascii_text = ""
    for byte in window:
        if 32 <= byte <= 126:
            ascii_text += chr(byte)
        else:
            ascii_text += "."
    return {
        "start": start,
        "end_exclusive": end,
        "hex": " ".join("%02x" % byte for byte in window),
        "ascii": ascii_text,
    }


def _text_context(data, offset):
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return None

    position = offset
    if position > len(data):
        position = len(data)
    line_start = data.rfind(b"\n", 0, position) + 1
    line_end = data.find(b"\n", position)
    if line_end < 0:
        line_end = len(data)
    line_number = data[:line_start].count(b"\n") + 1
    raw_line = data[line_start:line_end].decode("utf-8")
    prefix = data[line_start:position].decode("utf-8", errors="replace")
    return {
        "line": line_number,
        "column": len(prefix) + 1,
        "raw_line": raw_line,
    }


def _byte_difference_report(left, right):
    offset = _first_difference(left, right)
    if offset is None:
        return None
    report = {
        "offset": offset,
        "left_byte": left[offset] if offset < len(left) else None,
        "right_byte": right[offset] if offset < len(right) else None,
        "hexdump": {
            "left": _hex_window(left, offset),
            "right": _hex_window(right, offset),
        },
    }
    left_context = _text_context(left, offset)
    right_context = _text_context(right, offset)
    if left_context is not None or right_context is not None:
        report["text"] = {"left": left_context, "right": right_context}
    return report


def _new_census(rule):
    return {
        "name": rule.name,
        "regex": rule.pattern_text,
        "scope": rule.scope,
        "whitelist": list(rule.whitelist),
        "left_lines_touched": 0,
        "left_bytes_touched": 0,
        "right_lines_touched": 0,
        "right_bytes_touched": 0,
        "left_whitelist_spans_removed": 0,
        "left_whitelist_bytes_removed": 0,
        "right_whitelist_spans_removed": 0,
        "right_whitelist_bytes_removed": 0,
    }


def _normalize_lines(lines, rules, side):
    normalized = []
    census = []
    for rule in rules:
        census.append(_new_census(rule))
    guard_violations = []

    for line_number, raw_line in enumerate(lines, 1):
        working_line = raw_line
        for rule_index, rule in enumerate(rules):
            if not rule.applies_to(raw_line):
                continue
            matches = list(rule.regex.finditer(working_line))
            if not matches:
                continue
            census_entry = census[rule_index]
            census_entry["%s_lines_touched" % side] += 1
            for match in matches:
                matched_text = match.group(0)
                census_entry["%s_bytes_touched" % side] += len(
                    matched_text.encode("utf-8")
                )
                whitelist_spans = rule.whitelist_spans(matched_text)
                census_entry["%s_whitelist_spans_removed" % side] += len(
                    whitelist_spans
                )
                census_entry["%s_whitelist_bytes_removed" % side] += sum(
                    len(matched_text[start:end].encode("utf-8"))
                    for start, end in whitelist_spans
                )
                scan_text = list(matched_text)
                for start, end in whitelist_spans:
                    scan_text[start:end] = " " * (end - start)
                scan_text = "".join(scan_text)
                for number_match in _NUMBER_TOKEN.finditer(scan_text):
                    token = number_match.group(0)
                    guard_violations.append(
                        {
                            "rule": rule.name,
                            "side": side,
                            "line": line_number,
                            "token": token,
                            "matched_text": matched_text,
                        }
                    )
            try:
                working_line = rule.regex.sub(rule.replacement, working_line)
            except (re.error, IndexError, KeyError) as exc:
                raise ComparatorConfigError(
                    "replacement failed for rule %r: %s" % (rule.name, exc)
                ) from exc
        normalized.append(working_line)

    return normalized, census, guard_violations


def _first_line_difference(left_lines, right_lines):
    common_length = len(left_lines)
    if len(right_lines) < common_length:
        common_length = len(right_lines)
    for index in range(common_length):
        if left_lines[index] != right_lines[index]:
            return index
    if len(left_lines) != len(right_lines):
        return common_length
    return None


def _line_difference_report(
    line_number, left_lines, right_lines, normalized_left, normalized_right
):
    left_raw = left_lines[line_number] if line_number < len(left_lines) else None
    right_raw = right_lines[line_number] if line_number < len(right_lines) else None
    left_normalized = (
        normalized_left[line_number] if line_number < len(normalized_left) else None
    )
    right_normalized = (
        normalized_right[line_number] if line_number < len(normalized_right) else None
    )
    return {
        "line": line_number + 1,
        "left_original": left_raw,
        "right_original": right_raw,
        "left_normalized": left_normalized,
        "right_normalized": right_normalized,
    }


def _empty_report(left_root, right_root, value_files, log_files):
    return {
        "schema": "tool-bytepar-v1",
        "left_root": str(left_root),
        "right_root": str(right_root),
        "declared_value_files": list(value_files),
        "declared_log_files": list(log_files),
        "value": {"compared_files": 0, "files": [], "pass": False},
        "log": {"compared_files": 0, "files": [], "pass": False},
        "errors": [],
    }


def compare_runs(left_root, right_root, value_files, log_files=None, rules=None):
    """Compare two run roots and return a JSON-serializable report.

    ``value_files`` and ``log_files`` are relative paths declared by the
    caller.  A missing declaration, missing side, read error, invalid rule,
    or normalization guard violation is fail-closed (exit code 2).
    """

    try:
        declared_values = _coerce_declared_paths(value_files, "value")
        declared_logs = _coerce_declared_paths(log_files, "log")
        normalized_rules = _coerce_rules(rules)
        left_root_path = Path(left_root)
        right_root_path = Path(right_root)
        report = _empty_report(
            left_root_path, right_root_path, declared_values, declared_logs
        )
        report["normalization_rules"] = [rule.as_report() for rule in normalized_rules]
        _resolve_under_root(left_root_path, ".")
        _resolve_under_root(right_root_path, ".")
    except (ComparatorConfigError, OSError, TypeError, ValueError) as exc:
        report = _empty_report(left_root, right_root, [], [])
        report["errors"].append("configuration: %s" % exc)
        report["verdict"] = "FAIL_CLOSED"
        report["exit_code"] = EXIT_FAIL_CLOSED
        report["summary"] = "FAIL_CLOSED byte-parity values=0 logs=0"
        return report
    except Exception as exc:
        report = _empty_report(left_root, right_root, [], [])
        report["errors"].append("unexpected comparator error: %s: %s" % (type(exc).__name__, exc))
        report["verdict"] = "FAIL_CLOSED"
        report["exit_code"] = EXIT_FAIL_CLOSED
        report["summary"] = "FAIL_CLOSED byte-parity values=0 logs=0"
        return report

    fail_closed = False
    different = False

    if not declared_values:
        fail_closed = True
        report["errors"].append("no value files were declared")

    for relative_path in declared_values:
        file_report = {"path": relative_path}
        try:
            left_path = _resolve_under_root(left_root_path, relative_path)
            right_path = _resolve_under_root(right_root_path, relative_path)
        except ComparatorConfigError as exc:
            fail_closed = True
            file_report["status"] = "FAIL_CLOSED"
            file_report["error"] = str(exc)
            report["value"]["files"].append(file_report)
            continue

        left_data, left_error = _read_bytes(left_path)
        right_data, right_error = _read_bytes(right_path)
        file_report["left_path"] = str(left_path)
        file_report["right_path"] = str(right_path)
        file_report["left_size"] = len(left_data) if left_data is not None else None
        file_report["right_size"] = len(right_data) if right_data is not None else None
        file_report["left_sha256"] = _sha256(left_data) if left_data is not None else None
        file_report["right_sha256"] = _sha256(right_data) if right_data is not None else None

        if left_error is not None or right_error is not None:
            fail_closed = True
            file_report["status"] = "FAIL_CLOSED"
            file_report["left_error"] = left_error
            file_report["right_error"] = right_error
        else:
            report["value"]["compared_files"] += 1
            difference = _byte_difference_report(left_data, right_data)
            if difference is None:
                file_report["status"] = "IDENTICAL"
            else:
                different = True
                file_report["status"] = "DIFFERENT"
                file_report["first_difference"] = difference
        report["value"]["files"].append(file_report)

    if report["value"]["compared_files"] == 0:
        fail_closed = True
        report["errors"].append("no value files could be compared")

    report["value"]["pass"] = (
        not fail_closed
        and report["value"]["compared_files"] > 0
        and all(item.get("status") == "IDENTICAL" for item in report["value"]["files"])
    )

    if not declared_logs:
        report["log"]["pass"] = True
        report["log"]["note"] = "no log files declared; value tier remains the gate"

    aggregate_census = []
    for rule in normalized_rules:
        aggregate_census.append(_new_census(rule))

    for relative_path in declared_logs:
        file_report = {"path": relative_path}
        try:
            left_path = _resolve_under_root(left_root_path, relative_path)
            right_path = _resolve_under_root(right_root_path, relative_path)
        except ComparatorConfigError as exc:
            fail_closed = True
            file_report["status"] = "FAIL_CLOSED"
            file_report["error"] = str(exc)
            report["log"]["files"].append(file_report)
            continue
        left_data, left_lines, left_error = _read_text_lines(left_path)
        right_data, right_lines, right_error = _read_text_lines(right_path)
        file_report["left_path"] = str(left_path)
        file_report["right_path"] = str(right_path)
        file_report["left_size"] = len(left_data) if left_data is not None else None
        file_report["right_size"] = len(right_data) if right_data is not None else None

        if left_error is not None or right_error is not None:
            fail_closed = True
            file_report["status"] = "FAIL_CLOSED"
            file_report["left_error"] = left_error
            file_report["right_error"] = right_error
            report["log"]["files"].append(file_report)
            continue

        try:
            normalized_left, left_census, left_guard = _normalize_lines(
                left_lines, normalized_rules, "left"
            )
            normalized_right, right_census, right_guard = _normalize_lines(
                right_lines, normalized_rules, "right"
            )
        except ComparatorConfigError as exc:
            fail_closed = True
            file_report["status"] = "FAIL_CLOSED"
            file_report["error"] = str(exc)
            report["log"]["files"].append(file_report)
            continue
        except Exception as exc:
            fail_closed = True
            file_report["status"] = "FAIL_CLOSED"
            file_report["error"] = "unexpected normalization error: %s: %s" % (
                type(exc).__name__,
                exc,
            )
            report["log"]["files"].append(file_report)
            continue

        file_report["rule_census"] = []
        for rule_index, rule in enumerate(normalized_rules):
            file_census = {
                "name": rule.name,
                "regex": rule.pattern_text,
                "scope": rule.scope,
                "whitelist": list(rule.whitelist),
                "left_lines_touched": left_census[rule_index]["left_lines_touched"],
                "left_bytes_touched": left_census[rule_index]["left_bytes_touched"],
                "right_lines_touched": right_census[rule_index]["right_lines_touched"],
                "right_bytes_touched": right_census[rule_index]["right_bytes_touched"],
                "left_whitelist_spans_removed": left_census[rule_index][
                    "left_whitelist_spans_removed"
                ],
                "left_whitelist_bytes_removed": left_census[rule_index][
                    "left_whitelist_bytes_removed"
                ],
                "right_whitelist_spans_removed": right_census[rule_index][
                    "right_whitelist_spans_removed"
                ],
                "right_whitelist_bytes_removed": right_census[rule_index][
                    "right_whitelist_bytes_removed"
                ],
            }
            file_report["rule_census"].append(file_census)
            aggregate_census[rule_index]["left_lines_touched"] += file_census[
                "left_lines_touched"
            ]
            aggregate_census[rule_index]["left_bytes_touched"] += file_census[
                "left_bytes_touched"
            ]
            aggregate_census[rule_index]["right_lines_touched"] += file_census[
                "right_lines_touched"
            ]
            aggregate_census[rule_index]["right_bytes_touched"] += file_census[
                "right_bytes_touched"
            ]
            aggregate_census[rule_index]["left_whitelist_spans_removed"] += (
                file_census["left_whitelist_spans_removed"]
            )
            aggregate_census[rule_index]["left_whitelist_bytes_removed"] += (
                file_census["left_whitelist_bytes_removed"]
            )
            aggregate_census[rule_index]["right_whitelist_spans_removed"] += (
                file_census["right_whitelist_spans_removed"]
            )
            aggregate_census[rule_index]["right_whitelist_bytes_removed"] += (
                file_census["right_whitelist_bytes_removed"]
            )

        guards = []
        guards.extend(left_guard)
        guards.extend(right_guard)
        if guards:
            fail_closed = True
            file_report["status"] = "FAIL_CLOSED"
            file_report["normalization_guard"] = {
                "pass": False,
                "violations": guards,
            }
        else:
            file_report["normalization_guard"] = {"pass": True, "violations": []}
            report["log"]["compared_files"] += 1
            line_index = _first_line_difference(
                normalized_left, normalized_right
            )
            if line_index is None:
                file_report["status"] = "IDENTICAL"
            else:
                different = True
                file_report["status"] = "DIFFERENT"
                file_report["first_difference"] = _line_difference_report(
                    line_index,
                    left_lines,
                    right_lines,
                    normalized_left,
                    normalized_right,
                )
        report["log"]["files"].append(file_report)

    report["log"]["rule_census"] = aggregate_census
    if declared_logs and report["log"]["compared_files"] == 0:
        fail_closed = True
        report["errors"].append("no log files could be compared")
    report["log"]["pass"] = (
        not fail_closed
        and (
            not declared_logs
            or all(item.get("status") == "IDENTICAL" for item in report["log"]["files"])
        )
    )

    if fail_closed:
        verdict = "FAIL_CLOSED"
        exit_code = EXIT_FAIL_CLOSED
    elif different:
        verdict = "FAIL"
        exit_code = EXIT_DIFFERENT
    else:
        verdict = "PASS"
        exit_code = EXIT_IDENTICAL
    report["verdict"] = verdict
    report["exit_code"] = exit_code
    report["summary"] = (
        "%s byte-parity values=%d logs=%d"
        % (verdict, report["value"]["compared_files"], report["log"]["compared_files"])
    )
    return report


def _load_rule_file(path):
    try:
        content = Path(path).read_text(encoding="utf-8")
        value = json.loads(content)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ComparatorConfigError("cannot read rule file %s: %s" % (path, exc)) from exc
    if not isinstance(value, list):
        raise ComparatorConfigError("rule file must contain a JSON list")
    return value


def _parse_inline_rules(raw_rules):
    result = []
    for raw_rule in raw_rules:
        try:
            value = json.loads(raw_rule)
        except json.JSONDecodeError as exc:
            raise ComparatorConfigError("--rule is not valid JSON: %s" % exc) from exc
        result.append(value)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare declared value files byte-for-byte and logs in order."
    )
    parser.add_argument("left_root", type=Path)
    parser.add_argument("right_root", type=Path)
    parser.add_argument(
        "--value-file",
        "--value",
        dest="value_files",
        action="append",
        default=[],
        help="relative value path; repeat for every value file",
    )
    parser.add_argument(
        "--log-file",
        "--log",
        dest="log_files",
        action="append",
        default=[],
        help="relative log path; repeat for every log file",
    )
    parser.add_argument(
        "--rules",
        "--log-rules",
        dest="rule_file",
        type=Path,
        help="JSON list of named normalization rules",
    )
    parser.add_argument(
        "--rule",
        action="append",
        default=[],
        help="one normalization rule as a JSON object; repeatable",
    )
    parser.add_argument(
        "--output",
        "--report",
        dest="output",
        type=Path,
        required=True,
        help="JSON report path",
    )
    args = parser.parse_args(argv)

    try:
        raw_rules = []
        if args.rule_file is not None:
            raw_rules.extend(_load_rule_file(args.rule_file))
        raw_rules.extend(_parse_inline_rules(args.rule))
    except ComparatorConfigError as exc:
        report = _empty_report(args.left_root, args.right_root, [], [])
        report["errors"].append(str(exc))
        report["verdict"] = "FAIL_CLOSED"
        report["exit_code"] = EXIT_FAIL_CLOSED
        report["summary"] = "FAIL_CLOSED byte-parity values=0 logs=0"
    else:
        report = compare_runs(
            args.left_root,
            args.right_root,
            args.value_files,
            args.log_files,
            raw_rules,
        )

    try:
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        print("FAIL_CLOSED byte-parity report-write: %s" % exc)
        return EXIT_FAIL_CLOSED
    print(report["summary"])
    return report["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
