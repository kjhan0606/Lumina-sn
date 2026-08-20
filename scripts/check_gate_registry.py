#!/usr/bin/env python3
"""Check gate-registry completeness, wiring, known-red pins, and self-NCs."""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "scripts/gate_registry.json"
MAKEFILE_PATH = ROOT / "Makefile"
BATTERY_PATH = ROOT / "scripts/run_gate_battery.py"
SCHEMA = "lumina-gate-registry-v2"
GATE_SUFFIX = re.compile(r"-(?:check|census|gate)$")
TOKEN_CHARS = r"A-Za-z0-9_.-"
E11_WIRING = (
    "EMISS_E11_SEEDED_FIXTURE; recipe 2 commands: execution gate wired; "
    "py_compile 4 files unwired"
)
KNOWN_RED_FIELDS = {
    "failing_command_index",
    "rc",
    "first_fail_line",
    "fail_line_count",
    "registered",
    "repair_rung",
}
VALID_CATEGORIES = {
    "battery-core",
    "preflight",
    "collective",
    "milestone",
    "run-dependent",
    "manual-audit",
    "manual-lane",
    "known-red",
    "unwired",
}
UNWIRED_FIELDS = {"observed", "disposition_rung", "lexical_mentions", "note"}
UNWIRED_REQUIRED_FIELDS = {"observed", "disposition_rung"}


@dataclass(frozen=True)
class LogicalLine:
    text: str
    first_line: int
    last_line: int


def make_logical_lines(text: str) -> list[LogicalLine]:
    """Join Make backslash continuations before interpreting a line.

    In particular, every physical continuation of a ``.PHONY`` declaration
    remains part of that one logical declaration; none can masquerade as an
    executable reference.
    """
    result: list[LogicalLine] = []
    parts: list[str] = []
    first = 1
    physical = text.splitlines()
    for number, raw in enumerate(physical, 1):
        if not parts:
            first = number
        match = re.search(r"\\\s*$", raw)
        if match:
            parts.append(raw[:match.start()].rstrip())
            continue
        parts.append(raw)
        result.append(LogicalLine(" ".join(parts), first, number))
        parts = []
    if parts:
        result.append(LogicalLine(" ".join(parts), first, len(physical)))
    return result


def is_gate_target(name: str) -> bool:
    return name.startswith("selftest") or GATE_SUFFIX.search(name) is not None


def make_targets(lines: Iterable[LogicalLine]) -> dict[str, LogicalLine]:
    targets: dict[str, LogicalLine] = {}
    for logical in lines:
        line = logical.text
        if not line or line[0].isspace() or line.startswith("#") or ":" not in line:
            continue
        lhs = line.split(":", 1)[0]
        if "=" in lhs:
            continue
        for target in lhs.split():
            if is_gate_target(target):
                targets[target] = logical
    return targets


def make_rule_targets(line: str) -> tuple[str, ...]:
    """Return the targets on a Make rule definition, or no targets."""
    if not line or line[0].isspace() or line.startswith("#") or ":" not in line:
        return ()
    lhs = line.split(":", 1)[0]
    if "=" in lhs:
        return ()
    return tuple(lhs.split())


def make_references(lines: Iterable[LogicalLine], target: str) -> list[LogicalLine]:
    """Return lexical wiring references with the four contracted exclusions.

    The exclusions are exactly full comments, ``.PHONY`` logical lines, the
    target's own rule definitions and recipes, and recipes belonging to a
    ``clean`` rule.  References in every other rule remain visible.
    """
    token = re.compile(
        rf"(?<![{TOKEN_CHARS}]){re.escape(target)}(?![{TOKEN_CHARS}])"
    )
    references: list[LogicalLine] = []
    recipe_owners: tuple[str, ...] = ()
    for logical in lines:
        line = logical.text
        stripped = line.lstrip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        if line.startswith("\t"):
            if target in recipe_owners or "clean" in recipe_owners:
                continue
            candidate = line
        else:
            recipe_owners = make_rule_targets(line)
            if stripped.startswith(".PHONY:"):
                continue
            if target in recipe_owners:
                continue
            candidate = line
            if "clean" in recipe_owners and ";" in candidate:
                candidate = candidate.split(";", 1)[0]
        if token.search(candidate):
            references.append(logical)
    return references


def target_recipe_commands(
    lines: Iterable[LogicalLine], target: str
) -> list[str]:
    """Return one target's Make recipe as normalized command strings."""
    commands: list[str] = []
    recipe_owners: tuple[str, ...] = ()
    for logical in lines:
        line = logical.text
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        if line.startswith("\t"):
            if target in recipe_owners:
                commands.append(stripped)
            continue
        recipe_owners = make_rule_targets(line)
        if target in recipe_owners and ";" in line:
            inline_recipe = line.split(";", 1)[1].strip()
            if inline_recipe:
                commands.append(inline_recipe)
    return commands


def target_prerequisites(lines: Iterable[LogicalLine], target: str) -> tuple[str, ...]:
    for logical in lines:
        line = logical.text
        if not line or line[0].isspace() or line.startswith("#") or ":" not in line:
            continue
        lhs, rhs = line.split(":", 1)
        if target in lhs.split():
            return tuple(rhs.split())
    return ()


def python_constants(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    wanted = {"BATTERY_ORDER", "EXPECTED_ROWS", "PREFLIGHTS"}
    found: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in wanted:
            found[target.id] = ast.literal_eval(node.value)
    return found


def read_registry_text(text: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None, "registry-unreadable"
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        return None, "registry-unreadable"
    if not isinstance(value.get("entries"), list):
        return None, "registry-unreadable"
    return value, None


def split_wiring(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def command_shape_is_valid(commands: Any) -> bool:
    return (
        isinstance(commands, list)
        and bool(commands)
        and all(
            isinstance(command, list)
            and bool(command)
            and all(isinstance(arg, str) and arg for arg in command)
            for command in commands
        )
    )


def validate_known_red_row(entry: dict[str, Any]) -> str | None:
    name = entry.get("name", "<unnamed>")
    pin = entry.get("known_red")
    if not isinstance(pin, dict) or not KNOWN_RED_FIELDS.issubset(pin):
        return f"known-red-row-incomplete:{name}"
    if set(pin) != KNOWN_RED_FIELDS:
        return f"known-red-row-incomplete:{name}"
    commands = entry.get("commands")
    if not command_shape_is_valid(commands):
        return f"known-red-row-incomplete:{name}"
    index = pin["failing_command_index"]
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(commands):
        return f"known-red-row-incomplete:{name}"
    for field in ("rc", "fail_line_count"):
        if not isinstance(pin[field], int) or isinstance(pin[field], bool):
            return f"known-red-row-incomplete:{name}"
    if pin["rc"] == 0 or pin["fail_line_count"] < 1:
        return f"known-red-row-incomplete:{name}"
    if not isinstance(pin["first_fail_line"], str) or not pin["first_fail_line"]:
        return f"known-red-row-incomplete:{name}"
    try:
        date.fromisoformat(pin["registered"])
    except (TypeError, ValueError):
        return f"known-red-row-incomplete:{name}"
    if not isinstance(pin["repair_rung"], str) or not re.fullmatch(
        r"GR-[1-9][0-9]*", pin["repair_rung"]
    ):
        return f"known-red-row-incomplete:{name}"
    return None


def validate_unwired_row(entry: dict[str, Any]) -> str | None:
    name = entry.get("name", "<unnamed>")
    unwired = entry.get("unwired")
    if (
        not isinstance(unwired, dict)
        or not UNWIRED_REQUIRED_FIELDS.issubset(unwired)
        or not set(unwired).issubset(UNWIRED_FIELDS)
    ):
        return f"unwired-row-incomplete:{name}"
    observed = unwired["observed"]
    if not isinstance(observed, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}", observed
    ):
        return f"unwired-row-incomplete:{name}"
    try:
        date.fromisoformat(observed)
    except ValueError:
        return f"unwired-row-incomplete:{name}"
    if unwired["disposition_rung"] != "GR-7":
        return f"unwired-row-incomplete:{name}"
    lexical_mentions = unwired.get("lexical_mentions", [])
    if (
        not isinstance(lexical_mentions, list)
        or any(not isinstance(path, str) or not path for path in lexical_mentions)
    ):
        return f"unwired-row-incomplete:{name}"
    if len(set(lexical_mentions)) != len(lexical_mentions):
        return f"unwired-row-incomplete:{name}"
    for relative in lexical_mentions:
        path = ROOT / relative
        try:
            in_census = path.resolve().is_relative_to(ROOT.resolve()) and any(
                path.resolve().is_relative_to(directory.resolve())
                for directory in (ROOT / "scripts", ROOT / "tests")
            )
        except OSError:
            in_census = False
        if path.suffix not in {".sh", ".py"} or not path.is_file() or not in_census:
            return f"unwired-row-incomplete:{name}"
    if "note" in unwired and (
        not isinstance(unwired["note"], str) or not unwired["note"]
    ):
        return f"unwired-row-incomplete:{name}"
    return None


def source_token_references(target: str) -> list[str]:
    """Find target tokens in the contracted scripts/tests source census."""
    token = re.compile(
        rf"(?<![{TOKEN_CHARS}]){re.escape(target)}(?![{TOKEN_CHARS}])"
    )
    references: list[str] = []
    for directory in (ROOT / "scripts", ROOT / "tests"):
        for suffix in ("*.sh", "*.py"):
            for path in sorted(directory.rglob(suffix)):
                if path.is_file() and token.search(
                    path.read_text(encoding="utf-8", errors="replace")
                ):
                    references.append(path.relative_to(ROOT).as_posix())
    return references


def validate_registry(
    registry: dict[str, Any], make_text: str, battery_path: Path = BATTERY_PATH
) -> list[str]:
    errors: list[str] = []
    lines = make_logical_lines(make_text)
    targets = make_targets(lines)
    try:
        constants = python_constants(battery_path)
        battery_order = tuple(constants["BATTERY_ORDER"])
        expected_rows = dict(constants["EXPECTED_ROWS"])
        preflight_rows = tuple(constants["PREFLIGHTS"])
    except (KeyError, OSError, SyntaxError, TypeError, ValueError):
        return ["battery-contract-unreadable"]
    preflights: dict[str, tuple[str, tuple[str, ...]]] = {}
    for row in preflight_rows:
        if (
            not isinstance(row, tuple)
            or len(row) != 3
            or not isinstance(row[0], str)
            or not isinstance(row[1], str)
            or not isinstance(row[2], tuple)
        ):
            errors.append("preflight-row-unreadable")
            continue
        if row[0] in preflights:
            errors.append(f"duplicate-preflight-row:{row[0]}")
        preflights[row[0]] = (row[1], row[2])

    by_name: dict[str, dict[str, Any]] = {}
    for raw_entry in registry["entries"]:
        if not isinstance(raw_entry, dict):
            errors.append("registry-row-incomplete:<unnamed>")
            continue
        name = raw_entry.get("name", "<unnamed>")
        required = {"name", "kind", "commands", "category", "wiring"}
        if (
            not required.issubset(raw_entry)
            or not isinstance(name, str)
            or not name
            or raw_entry.get("kind") not in {"make-target", "script"}
            or not command_shape_is_valid(raw_entry.get("commands"))
            or not isinstance(raw_entry.get("wiring"), str)
            or not raw_entry.get("wiring")
        ):
            errors.append(f"registry-row-incomplete:{name}")
            continue
        if name in by_name:
            errors.append(f"duplicate-registry-entry:{name}")
            continue
        by_name[name] = raw_entry

        category = raw_entry["category"]
        if category not in VALID_CATEGORIES:
            errors.append(f"unknown-category:{name}")
            continue
        if category == "known-red":
            reason = validate_known_red_row(raw_entry)
            if reason:
                errors.append(reason)
            elif raw_entry["wiring"] != raw_entry["known_red"]["repair_rung"]:
                errors.append(f"known-red-wiring-drift:{name}")
        elif "known_red" in raw_entry:
            errors.append(f"known-red-field-forbidden:{name}")
        if category == "unwired":
            reason = validate_unwired_row(raw_entry)
            if reason:
                errors.append(reason)
            elif raw_entry["wiring"] != raw_entry["unwired"]["disposition_rung"]:
                errors.append(f"unwired-row-incomplete:{name}")
        elif "unwired" in raw_entry:
            errors.append(f"unwired-field-forbidden:{name}")

        if raw_entry["kind"] == "make-target" and name not in targets:
            errors.append(f"registry-target-missing:{name}")
        if raw_entry["kind"] == "script" and not (ROOT / name).is_file():
            errors.append(f"registry-script-missing:{name}")

        wiring_text = raw_entry["wiring"]
        wiring = split_wiring(wiring_text)
        if category == "battery-core":
            if wiring != battery_order or any(
                battery not in expected_rows or not isinstance(expected_rows[battery], int)
                for battery in battery_order
            ):
                errors.append(f"battery-core-wiring-invalid:{name}")
        elif category == "preflight":
            if name == "selftest_emiss_e11_fluor_matrix":
                if wiring_text != E11_WIRING:
                    errors.append(f"preflight-wiring-missing:{name}")
                wiring = split_wiring(wiring_text.split(";", 1)[0])
            expected_commands: list[list[str]] = []
            for row_name in wiring:
                if row_name not in preflights:
                    errors.append(f"preflight-wiring-missing:{name}")
                    continue
                script, args = preflights[row_name]
                expected_commands.append(["python3", script, *args])
            if expected_commands != raw_entry["commands"]:
                errors.append(f"preflight-command-drift:{name}")
        elif category == "collective":
            path = ROOT / raw_entry["wiring"]
            if not path.is_file():
                errors.append(f"collective-wiring-missing:{name}")
            elif path == MAKEFILE_PATH:
                if not make_references(lines, name):
                    errors.append(f"collective-reference-missing:{name}")
            else:
                text = path.read_text(encoding="utf-8", errors="replace")
                token = re.compile(
                    rf"(?<![{TOKEN_CHARS}]){re.escape(name)}(?![{TOKEN_CHARS}])"
                )
                if token.search(text) is None:
                    errors.append(f"collective-reference-missing:{name}")
        elif category == "unwired":
            unwired = raw_entry.get("unwired")
            allowed = (
                set(unwired.get("lexical_mentions", []))
                if isinstance(unwired, dict)
                else set()
            )
            make_reference = bool(make_references(lines, name))
            source_references = source_token_references(name)
            if make_reference or any(path not in allowed for path in source_references):
                errors.append(f"unwired-now-referenced:{name}")
        elif category == "milestone":
            groups = wiring
            if not groups or any(
                name not in target_prerequisites(lines, group) for group in groups
            ):
                errors.append(f"milestone-wiring-missing:{name}")
        elif category == "manual-lane":
            for lane in wiring:
                path = ROOT / lane
                if not path.is_file() or name not in path.read_text(
                    encoding="utf-8", errors="replace"
                ):
                    errors.append(f"manual-lane-wiring-missing:{name}")
        elif category in {"run-dependent", "manual-audit"}:
            if not wiring:
                errors.append(f"category-reason-missing:{name}")

        if category == "known-red" and raw_entry["kind"] == "make-target":
            recipe = target_recipe_commands(lines, name)
            commands = [" ".join(command) for command in raw_entry["commands"]]
            if recipe != commands:
                errors.append(f"known-red-recipe-drift:{name}")

    for target in sorted(targets):
        if target not in by_name:
            errors.append(f"unregistered-gate-target:{target}")
    return errors


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def observe_known_red(entry: dict[str, Any]) -> str | None:
    name = entry["name"]
    pin = entry["known_red"]
    failure_index: int | None = None
    failure: subprocess.CompletedProcess[str] | None = None
    for index, command in enumerate(entry["commands"]):
        try:
            proc = run_command(command)
        except OSError:
            return f"known-red-signature-drift:{name}"
        if proc.returncode != 0:
            failure_index = index
            failure = proc
            break
    if failure is None:
        return f"known-red-unexpected-pass:{name}"
    fail_lines = [line for line in failure.stdout.splitlines() if "FAIL" in line]
    actual = {
        "failing_command_index": failure_index,
        "rc": failure.returncode,
        "first_fail_line": fail_lines[0] if fail_lines else "",
        "fail_line_count": len(fail_lines),
    }
    expected = {key: pin[key] for key in actual}
    if actual != expected:
        return f"known-red-signature-drift:{name}"
    return None


def run_negative_controls() -> list[tuple[str, str, bool]]:
    empty = {"schema": SCHEMA, "entries": []}
    name_a = "synthetic-r1a-gate"
    make_a = f".PHONY: anchor\nanchor:\n\t@:\n{name_a}:\n\t@:\n"
    errors_a = validate_registry(empty, make_a)
    expected_a = f"unregistered-gate-target:{name_a}"

    name_b = "synthetic-r1b"
    fail_line_b = "[NC-R1b][FAIL] sentinel"
    entry_b = {
        "name": name_b,
        "kind": "make-target",
        "commands": [[sys.executable, "-c", f"print({fail_line_b!r});raise SystemExit(1)"]],
        "category": "known-red",
        "wiring": "GR-1",
        "known_red": {
            "failing_command_index": 0,
            "rc": 1,
            "first_fail_line": fail_line_b + "X",
            "fail_line_count": 1,
            "registered": "2026-08-20",
            "repair_rung": "GR-1",
        },
    }
    actual_b = observe_known_red(entry_b)
    expected_b = f"known-red-signature-drift:{name_b}"

    name_c = "synthetic-r1c"
    entry_c = {
        "name": name_c,
        "kind": "make-target",
        "commands": [[sys.executable, "-c", "raise SystemExit(0)"]],
        "category": "known-red",
        "wiring": "GR-1",
        "known_red": {
            "failing_command_index": 0,
            "rc": 1,
            "first_fail_line": "[NC-R1c][FAIL] sentinel",
            "fail_line_count": 1,
            "registered": "2026-08-20",
            "repair_rung": "GR-1",
        },
    }
    actual_c = observe_known_red(entry_c)
    expected_c = f"known-red-unexpected-pass:{name_c}"

    name_d = "synthetic-r1d-gate"
    make_d = (
        ".PHONY: anchor \\\n"
        f"\t{name_d}\n"
        "anchor:\n\t@:\n"
        f"{name_d}:\n\t@:\n"
    )
    lines_d = make_logical_lines(make_d)
    errors_d = validate_registry(empty, make_d)
    expected_d = f"unregistered-gate-target:{name_d}"
    phony_is_not_reference = make_references(lines_d, name_d) == []

    name_e = "synthetic-r1e-gate"
    make_e = (
        f"{name_e}: fixture.c\n"
        f"\tcc -o {name_e} fixture.c\n"
        f"\t./{name_e}\n"
    )
    entry_e = {
        "name": name_e,
        "kind": "make-target",
        "commands": [["make", name_e]],
        "category": "collective",
        "wiring": "Makefile",
    }
    errors_e = validate_registry({"schema": SCHEMA, "entries": [entry_e]}, make_e)
    expected_e = f"collective-reference-missing:{name_e}"

    name_f = "synthetic-r1f-gate"
    make_f = f"{name_f}:\n\t@:\nclean:\n\trm -f {name_f}\n"
    entry_f = {
        "name": name_f,
        "kind": "make-target",
        "commands": [["make", name_f]],
        "category": "collective",
        "wiring": "Makefile",
    }
    errors_f = validate_registry({"schema": SCHEMA, "entries": [entry_f]}, make_f)
    expected_f = f"collective-reference-missing:{name_f}"

    name_g = "synthetic-" + "r1g-gate"
    make_g = f"{name_g}:\n\t@:\nanchor: {name_g}\n\t@:\n"
    entry_g = {
        "name": name_g,
        "kind": "make-target",
        "commands": [["make", name_g]],
        "category": "unwired",
        "wiring": "GR-7",
        "unwired": {
            "observed": "2026-08-20",
            "disposition_rung": "GR-7",
        },
    }
    errors_g = validate_registry({"schema": SCHEMA, "entries": [entry_g]}, make_g)
    expected_g = f"unwired-now-referenced:{name_g}"

    name_h = "synthetic-r1h-gate"
    make_h = f"{name_h}:\n\tpython3 scripts/synthetic-r1h-recipe.py\n"
    entry_h = {
        "name": name_h,
        "kind": "make-target",
        "commands": [["python3", "scripts/synthetic-r1h-registry.py"]],
        "category": "known-red",
        "wiring": "GR-1",
        "known_red": {
            "failing_command_index": 0,
            "rc": 1,
            "first_fail_line": "[NC-R1h][FAIL] sentinel",
            "fail_line_count": 1,
            "registered": "2026-08-20",
            "repair_rung": "GR-1",
        },
    }
    errors_h = validate_registry({"schema": SCHEMA, "entries": [entry_h]}, make_h)
    expected_h = f"known-red-recipe-drift:{name_h}"

    return [
        ("NC-R1a", expected_a, errors_a == [expected_a]),
        ("NC-R1b", expected_b, actual_b == expected_b),
        ("NC-R1c", expected_c, actual_c == expected_c),
        (
            "NC-R1d",
            expected_d,
            errors_d == [expected_d] and phony_is_not_reference,
        ),
        ("NC-R1e", expected_e, errors_e == [expected_e]),
        ("NC-R1f", expected_f, errors_f == [expected_f]),
        ("NC-R1g", expected_g, errors_g == [expected_g]),
        ("NC-R1h", expected_h, errors_h == [expected_h]),
    ]


def main() -> int:
    try:
        registry_text = REGISTRY_PATH.read_text(encoding="utf-8")
        make_text = MAKEFILE_PATH.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        print("[GATE-REGISTRY][COMPLETENESS][FAIL] reason=registry-unreadable")
        return 1
    registry, reason = read_registry_text(registry_text)
    if reason is not None or registry is None:
        print(f"[GATE-REGISTRY][COMPLETENESS][FAIL] reason={reason}")
        return 1

    errors = validate_registry(registry, make_text)
    if errors:
        for error in errors:
            print(f"[GATE-REGISTRY][COMPLETENESS][FAIL] reason={error}")
        return 1
    target_count = len(make_targets(make_logical_lines(make_text)))
    print(
        "[GATE-REGISTRY][COMPLETENESS][PASS] "
        f"make_targets={target_count} registry_entries={len(registry['entries'])}"
    )

    known_red = [
        entry for entry in registry["entries"] if entry["category"] == "known-red"
    ]
    red_errors = [reason for entry in known_red if (reason := observe_known_red(entry))]
    if red_errors:
        for error in red_errors:
            print(f"[GATE-REGISTRY][KNOWN-RED][FAIL] reason={error}")
        return 1
    print(f"[GATE-REGISTRY][KNOWN-RED][PASS] entries={len(known_red)}")

    nc_results = run_negative_controls()
    nc_failed = False
    for nc_name, expected, passed in nc_results:
        state = "PASS" if passed else "FAIL"
        print(
            f"[GATE-REGISTRY][NEGATIVE-CONTROL][{state}] "
            f"name={nc_name} reason={expected}"
        )
        nc_failed = nc_failed or not passed
    return 1 if nc_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
