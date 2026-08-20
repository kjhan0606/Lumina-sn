#!/usr/bin/env python3
"""Check gate-registry completeness, wiring, known pins, and self-NCs."""

from __future__ import annotations

import ast
import importlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "scripts/gate_registry.json"
MAKEFILE_PATH = ROOT / "Makefile"
BATTERY_PATH = ROOT / "scripts/run_gate_battery.py"
FIXTURE_GENERATOR_PATH = ROOT / "scripts/generate_composition_d_fixtures.py"
SCHEMA = "lumina-gate-registry-v3"
RUNG_NAME = r"[A-Z][A-Z0-9]*(-[A-Z0-9]+)+"
GATE_SUFFIX = re.compile(r"-(?:check|census|gate)$")
TOKEN_CHARS = r"A-Za-z0-9_.-"
MAKE_VARIABLE = re.compile(
    r"\$\(([A-Za-z_][A-Za-z0-9_]*)\)|\$\{([A-Za-z_][A-Za-z0-9_]*)\}"
)
MAKE_ASSIGNMENT = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_]*)\s*(\?=|\+=|:=|=)\s*(.*)$"
)
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
KNOWN_DRIFT_REQUIRED_FIELDS = {
    "build",
    "make_target",
    "battery_only",
    "make_only",
    "shared",
    "registered",
    "repair_rung",
}
KNOWN_DRIFT_FIELDS = KNOWN_DRIFT_REQUIRED_FIELDS | {"note"}
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
DISPOSITION_REQUIRED_FIELDS = {
    "verdict",
    "verdict_doc",
    "target_category",
    "target_wiring",
    "execution_rung",
}
DISPOSITION_FIELDS = DISPOSITION_REQUIRED_FIELDS | {"preconditions", "decided"}
DISPOSITION_TARGET_CATEGORIES = {
    "preflight",
    "milestone",
    "run-dependent",
    "known-red",
}
DISPOSITION_TARGET_WIRINGS = {
    "selftest-registry-milestone",
    "PHYSICS_COMPARISON_REGRID",
    "GPU and NVCC required",
    "SH-UW-1",
}
DISPOSITION_PRECONDITIONS = {
    "forced-rerun(§8-1)",
    "run-step-required(§8-2)",
}


@dataclass(frozen=True)
class LogicalLine:
    text: str
    first_line: int
    last_line: int


@dataclass(frozen=True)
class BatteryBuild:
    name: str
    command: tuple[str, ...]


@dataclass(frozen=True)
class BuildRecipePair:
    build_name: str
    make_target: str
    battery_sources: frozenset[str]
    make_sources: frozenset[str]

    @property
    def reason(self) -> str:
        return f"build-spec-drift:{self.build_name}"

    @property
    def battery_only(self) -> tuple[str, ...]:
        return tuple(sorted(self.battery_sources - self.make_sources))

    @property
    def make_only(self) -> tuple[str, ...]:
        return tuple(sorted(self.make_sources - self.battery_sources))

    @property
    def shared(self) -> tuple[str, ...]:
        return tuple(sorted(self.battery_sources & self.make_sources))


@dataclass(frozen=True)
class BuildSpecFailure:
    reason: str
    pair: BuildRecipePair | None = None


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


def expand_make_variables(
    value: str,
    variables: dict[str, str],
    stack: tuple[str, ...] = (),
) -> str:
    """Expand the simple Make variables used by source-list recipes.

    Make functions and substitution references are deliberately unsupported:
    leaving one uninterpreted could hide a C input, so callers reject any such
    construct instead of silently certifying it.
    """

    def replace(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(2)
        if name in stack:
            raise ValueError("recursive Make variable")
        if name not in variables:
            raise ValueError(f"unknown Make variable: {name}")
        return expand_make_variables(variables[name], variables, stack + (name,))

    expanded = MAKE_VARIABLE.sub(replace, value)
    if "$((" in expanded or re.search(r"\$\([^)]*[: ,][^)]*\)", expanded):
        raise ValueError("unsupported Make expression")
    if re.search(r"\$\([^)]*\)|\$\{[^}]*\}", expanded):
        raise ValueError("unsupported Make variable reference")
    return expanded


def make_variable_values(lines: Iterable[LogicalLine]) -> dict[str, str]:
    variables = {"MAKE": "make"}  # GNU Make's built-in recursive invocation.
    simple: set[str] = set()
    for logical in lines:
        line = logical.text
        if not line or line[0].isspace() or line.startswith("#"):
            continue
        match = MAKE_ASSIGNMENT.fullmatch(line)
        if match is None:
            continue
        name, operator, value = match.groups()
        if operator == "?=" and name in variables:
            continue
        if operator == "+=":
            addition = (
                expand_make_variables(value, variables)
                if name in simple
                else value
            )
            variables[name] = " ".join(
                part for part in (variables.get(name, ""), addition) if part
            )
        elif operator == ":=":
            variables[name] = expand_make_variables(value, variables)
            simple.add(name)
        else:
            variables[name] = value
            simple.discard(name)
    return variables


def shell_words(command: str) -> tuple[str, ...]:
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|()")
    lexer.commenters = "#"
    lexer.whitespace_split = True
    return tuple(lexer)


def c_source_words(words: Iterable[str]) -> frozenset[str]:
    sources = frozenset(word for word in words if word.endswith(".c"))
    if any(any(marker in source for marker in "*?[") for source in sources):
        raise ValueError("dynamic C source expression")
    return sources


def make_recipe_c_sources(make_text: str) -> dict[str, frozenset[str]]:
    """Return each Make target's expanded recipe-level ``.c`` input set."""
    lines = make_logical_lines(make_text)
    variables = make_variable_values(lines)
    recipes: dict[str, list[tuple[str, tuple[str, ...]]]] = {}
    owners: tuple[str, ...] = ()
    prerequisites: tuple[str, ...] = ()
    for logical in lines:
        line = logical.text
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        if line.startswith("\t"):
            for owner in owners:
                recipes.setdefault(owner, []).append((stripped, prerequisites))
            continue

        owners = make_rule_targets(line)
        prerequisites = ()
        if not owners:
            continue
        rhs = line.split(":", 1)[1]
        prerequisite_text, separator, inline_recipe = rhs.partition(";")
        expanded_prerequisites = expand_make_variables(
            prerequisite_text, variables
        )
        prerequisites = tuple(
            word for word in shell_words(expanded_prerequisites) if word != "|"
        )
        if separator and inline_recipe.strip():
            for owner in owners:
                recipes.setdefault(owner, []).append(
                    (inline_recipe.strip(), prerequisites)
                )

    result: dict[str, frozenset[str]] = {}
    for target, commands in recipes.items():
        sources: set[str] = set()
        for command, command_prerequisites in commands:
            expanded = expand_make_variables(command, variables)
            unique_prerequisites = tuple(dict.fromkeys(command_prerequisites))
            automatic = {
                "$@": target,
                "$<": command_prerequisites[0] if command_prerequisites else "",
                "$^": " ".join(unique_prerequisites),
                "$+": " ".join(command_prerequisites),
            }
            for token, replacement in automatic.items():
                expanded = expanded.replace(token, replacement)
            sources.update(c_source_words(shell_words(expanded)))
        result[target] = frozenset(sources)
    return result


def load_battery_builds(
    battery_path: Path = BATTERY_PATH,
) -> tuple[BatteryBuild, ...]:
    """Import ``run_gate_battery`` normally and read its live Build objects."""
    battery_path = battery_path.resolve()
    scripts_dir = battery_path.parent
    root = scripts_dir.parent
    module_names = ("run_gate_battery", "generate_composition_d_fixtures")
    previous_modules = {
        name: sys.modules.pop(name) for name in module_names if name in sys.modules
    }
    old_cwd = Path.cwd()
    sys.path.insert(0, str(scripts_dir))
    try:
        os.chdir(root)
        module = importlib.import_module("run_gate_battery")
        if Path(module.__file__).resolve() != battery_path:
            raise ValueError("wrong battery module imported")
        raw_builds = module.build_specs(root / ".gate-registry-probe", "cc")
        if not isinstance(raw_builds, tuple):
            raise TypeError("build_specs did not return a tuple")
        builds: list[BatteryBuild] = []
        names: set[str] = set()
        for raw in raw_builds:
            name = getattr(raw, "name", None)
            command = getattr(raw, "command", None)
            if (
                not isinstance(name, str)
                or not name
                or name in names
                or not isinstance(command, tuple)
                or not command
                or any(not isinstance(word, str) or not word for word in command)
            ):
                raise TypeError("unreadable Build row")
            names.add(name)
            builds.append(BatteryBuild(name, command))
        return tuple(builds)
    finally:
        os.chdir(old_cwd)
        try:
            sys.path.remove(str(scripts_dir))
        except ValueError:
            pass
        for name in module_names:
            sys.modules.pop(name, None)
        sys.modules.update(previous_modules)


def compare_build_specs(
    builds: Iterable[BatteryBuild], make_text: str
) -> tuple[list[BuildRecipePair], tuple[str, ...]]:
    """Pair by shared exact ``tests/*.c`` tokens and compare all C inputs."""
    make_sources = make_recipe_c_sources(make_text)
    pairs: list[BuildRecipePair] = []
    unpaired: list[str] = []
    seen_names: set[str] = set()
    for build in builds:
        if build.name in seen_names:
            raise ValueError("duplicate Build name")
        seen_names.add(build.name)
        battery_sources = c_source_words(build.command)
        battery_tests = frozenset(
            source for source in battery_sources if source.startswith("tests/")
        )
        targets = [
            target
            for target, sources in sorted(make_sources.items())
            if battery_tests.intersection(
                source for source in sources if source.startswith("tests/")
            )
        ]
        if not targets:
            unpaired.append(build.name)
            continue
        for target in targets:
            pairs.append(
                BuildRecipePair(
                    build.name, target, battery_sources, make_sources[target]
                )
            )
    return pairs, tuple(unpaired)


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
    if not isinstance(value.get("entries"), list) or not isinstance(
        value.get("known_drifts"), list
    ):
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
        RUNG_NAME, pin["repair_rung"]
    ):
        return f"known-red-row-incomplete:{name}"
    return None


def validate_known_drift_row(row: Any) -> str | None:
    name = row.get("build", "<unnamed>") if isinstance(row, dict) else "<unnamed>"
    if (
        not isinstance(row, dict)
        or not KNOWN_DRIFT_REQUIRED_FIELDS.issubset(row)
        or not set(row).issubset(KNOWN_DRIFT_FIELDS)
    ):
        return f"known-drift-row-incomplete:{name}"
    if (
        not isinstance(row["build"], str)
        or not row["build"]
        or not isinstance(row["make_target"], str)
        or not row["make_target"]
    ):
        return f"known-drift-row-incomplete:{name}"
    partitions: list[list[str]] = []
    for field in ("battery_only", "make_only", "shared"):
        values = row[field]
        if (
            not isinstance(values, list)
            or any(not isinstance(value, str) or not value for value in values)
            or values != sorted(set(values))
        ):
            return f"known-drift-row-incomplete:{name}"
        partitions.append(values)
    if any(
        set(partitions[left]).intersection(partitions[right])
        for left, right in ((0, 1), (0, 2), (1, 2))
    ):
        return f"known-drift-row-incomplete:{name}"
    if not isinstance(row["registered"], str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}", row["registered"]
    ):
        return f"known-drift-row-incomplete:{name}"
    try:
        date.fromisoformat(row["registered"])
    except ValueError:
        return f"known-drift-row-incomplete:{name}"
    if not isinstance(row["repair_rung"], str) or not re.fullmatch(
        RUNG_NAME, row["repair_rung"]
    ):
        return f"known-drift-row-incomplete:{name}"
    if "note" in row and (not isinstance(row["note"], str) or not row["note"]):
        return f"known-drift-row-incomplete:{name}"
    return None


def validate_known_drift_rows(rows: Any) -> list[str]:
    if not isinstance(rows, list):
        return ["registry-unreadable"]
    errors: list[str] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        reason = validate_known_drift_row(row)
        if reason is not None:
            errors.append(reason)
        if not isinstance(row, dict):
            continue
        build = row.get("build")
        make_target = row.get("make_target")
        if not isinstance(build, str) or not isinstance(make_target, str):
            continue
        key = (build, make_target)
        if key in seen:
            errors.append(f"duplicate-known-drift:{build}")
        else:
            seen.add(key)
    return errors


def evaluate_known_drifts(
    pairs: Iterable[BuildRecipePair], rows: list[dict[str, Any]]
) -> tuple[
    list[BuildSpecFailure],
    list[tuple[BuildRecipePair, dict[str, Any]]],
]:
    """Compare every Build/recipe pair with the exact registered three-way pin."""
    registered = {(row["build"], row["make_target"]): row for row in rows}
    actual_pairs: set[tuple[str, str]] = set()
    failures: list[BuildSpecFailure] = []
    observed: list[tuple[BuildRecipePair, dict[str, Any]]] = []
    for pair in pairs:
        key = (pair.build_name, pair.make_target)
        actual_pairs.add(key)
        row = registered.get(key)
        drifted = pair.battery_sources != pair.make_sources
        if row is None:
            if drifted:
                failures.append(BuildSpecFailure(pair.reason, pair))
            continue
        if not drifted:
            failures.append(
                BuildSpecFailure(
                    f"known-drift-unexpected-resolve:{pair.build_name}"
                )
            )
            continue
        actual_pin = {
            "battery_only": pair.battery_only,
            "make_only": pair.make_only,
            "shared": pair.shared,
        }
        expected_pin = {
            field: tuple(row[field]) for field in ("battery_only", "make_only", "shared")
        }
        if actual_pin != expected_pin:
            failures.append(
                BuildSpecFailure(
                    f"known-drift-signature-drift:{pair.build_name}"
                )
            )
        else:
            observed.append((pair, row))
    for key, row in registered.items():
        if key not in actual_pairs:
            failures.append(
                BuildSpecFailure(f"known-drift-pair-missing:{row['build']}")
            )
    return failures, observed


def format_known_drift_observation(
    pair: BuildRecipePair, row: dict[str, Any]
) -> str:
    return (
        "[GATE-REGISTRY][BUILD-SPEC][KNOWN-DRIFT] "
        f"reason=known-drift-observed:{pair.build_name} "
        f"make_target={pair.make_target} "
        f"battery_only={','.join(pair.battery_only) or '-'} "
        f"make_only={','.join(pair.make_only) or '-'} "
        f"repair_rung={row['repair_rung']}"
    )


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


def validate_disposition_row(entry: dict[str, Any]) -> str | None:
    name = entry.get("name", "<unnamed>")
    disposition = entry.get("disposition")
    if (
        not isinstance(disposition, dict)
        or not DISPOSITION_REQUIRED_FIELDS.issubset(disposition)
        or not set(disposition).issubset(DISPOSITION_FIELDS)
    ):
        return f"disposition-invalid:{name}"
    verdict = disposition["verdict"]
    target_category = disposition["target_category"]
    if not isinstance(verdict, str) or verdict not in {"wire", "known-red"}:
        return f"disposition-invalid:{name}"
    if (
        not isinstance(target_category, str)
        or target_category not in DISPOSITION_TARGET_CATEGORIES
    ):
        return f"disposition-invalid:{name}"
    if (verdict == "known-red") != (target_category == "known-red"):
        return f"disposition-invalid:{name}"
    target_wiring = disposition["target_wiring"]
    if (
        not isinstance(target_wiring, str)
        or target_wiring not in DISPOSITION_TARGET_WIRINGS
    ):
        return f"disposition-invalid:{name}"
    execution_rung = disposition["execution_rung"]
    if not isinstance(execution_rung, str) or not re.fullmatch(
        RUNG_NAME, execution_rung
    ):
        return f"disposition-invalid:{name}"
    verdict_doc = disposition["verdict_doc"]
    if not isinstance(verdict_doc, str) or not verdict_doc:
        return f"disposition-invalid:{name}"
    verdict_path = ROOT / verdict_doc
    try:
        verdict_in_repo = verdict_path.resolve().is_relative_to(ROOT.resolve())
    except OSError:
        verdict_in_repo = False
    if (
        verdict_path.suffix != ".md"
        or not verdict_in_repo
        or not verdict_path.is_file()
    ):
        return f"disposition-invalid:{name}"
    if "preconditions" in disposition:
        preconditions = disposition["preconditions"]
        if (
            not isinstance(preconditions, str)
            or preconditions not in DISPOSITION_PRECONDITIONS
        ):
            return f"disposition-invalid:{name}"
    if "decided" in disposition:
        decided = disposition["decided"]
        if not isinstance(decided, str) or not re.fullmatch(
            r"\d{4}-\d{2}-\d{2}", decided
        ):
            return f"disposition-invalid:{name}"
        try:
            date.fromisoformat(decided)
        except ValueError:
            return f"disposition-invalid:{name}"
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
    errors.extend(validate_known_drift_rows(registry.get("known_drifts")))
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
        if category == "unwired" and "disposition" not in raw_entry:
            errors.append(f"disposition-missing:{name}")
        elif "disposition" in raw_entry:
            reason = validate_disposition_row(raw_entry)
            if reason:
                errors.append(reason)
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


def retro_a209_negative_control(make_text: str) -> bool:
    """Replay the pre-GR-0 battery omission in an isolated copy tree."""
    before = (
        '                "tests/a2_09_emissivity_selftest.c",\n'
        '                "src/emissivity_publication.c", '
        '"src/population_contract.c", "-lm", "-o",'
    )
    after = (
        '                "tests/a2_09_emissivity_selftest.c",\n'
        '                "src/emissivity_publication.c", "-lm", "-o",'
    )
    try:
        with tempfile.TemporaryDirectory(prefix="gate-registry-gr8-") as raw_tmp:
            copy_root = Path(raw_tmp)
            copy_scripts = copy_root / "scripts"
            copy_scripts.mkdir()
            copied_battery = copy_scripts / BATTERY_PATH.name
            shutil.copy2(BATTERY_PATH, copied_battery)
            shutil.copy2(
                FIXTURE_GENERATOR_PATH,
                copy_scripts / FIXTURE_GENERATOR_PATH.name,
            )
            text = copied_battery.read_text(encoding="utf-8")
            if text.count(before) != 1:
                return False
            copied_battery.write_text(text.replace(before, after), encoding="utf-8")
            copied_builds = load_battery_builds(copied_battery)
            a209 = tuple(build for build in copied_builds if build.name == "Z-a2-09")
            pairs, unpaired = compare_build_specs(a209, make_text)
    except Exception:
        return False
    drifts = [pair for pair in pairs if pair.battery_sources != pair.make_sources]
    return (
        len(a209) == 1
        and len(pairs) == 1
        and not unpaired
        and [pair.reason for pair in drifts] == ["build-spec-drift:Z-a2-09"]
        and drifts[0].battery_only == ()
        and drifts[0].make_only == ("src/population_contract.c",)
    )


def run_negative_controls(make_text: str | None = None) -> list[tuple[str, str, bool]]:
    empty = {"schema": SCHEMA, "known_drifts": [], "entries": []}
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
    errors_e = validate_registry({**empty, "entries": [entry_e]}, make_e)
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
    errors_f = validate_registry({**empty, "entries": [entry_f]}, make_f)
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
        "disposition": {
            "verdict": "wire",
            "verdict_doc": "docs/VERDICT_UNWIRED_GATES_2026-08-20.md",
            "target_category": "milestone",
            "target_wiring": "selftest-registry-milestone",
            "execution_rung": "SH-UW-3",
            "decided": "2026-08-20",
        },
    }
    errors_g = validate_registry({**empty, "entries": [entry_g]}, make_g)
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
    errors_h = validate_registry({**empty, "entries": [entry_h]}, make_h)
    expected_h = f"known-red-recipe-drift:{name_h}"

    name_r7ba = "synthetic-" + "r7b-a-gate"
    make_r7ba = f"{name_r7ba}:\n\t@:\n"
    entry_r7ba = {
        "name": name_r7ba,
        "kind": "make-target",
        "commands": [["make", name_r7ba]],
        "category": "unwired",
        "wiring": "GR-7",
        "unwired": {
            "observed": "2026-08-20",
            "disposition_rung": "GR-7",
        },
    }
    errors_r7ba = validate_registry(
        {**empty, "entries": [entry_r7ba]}, make_r7ba
    )
    expected_r7ba = f"disposition-missing:{name_r7ba}"

    name_r7bb = "synthetic-" + "r7b-b-gate"
    make_r7bb = f"{name_r7bb}:\n\t@:\n"
    entry_r7bb = {
        "name": name_r7bb,
        "kind": "make-target",
        "commands": [["make", name_r7bb]],
        "category": "unwired",
        "wiring": "GR-7",
        "unwired": {
            "observed": "2026-08-20",
            "disposition_rung": "GR-7",
        },
        "disposition": {
            "verdict": "wire",
            "verdict_doc": "docs/VERDICT_UNWIRED_GATES_2026-08-20.md",
            "target_category": "milestone",
            "target_wiring": "selftest-registry-milestone",
            "execution_rung": "sh-uw-3",
            "decided": "2026-08-20",
        },
    }
    errors_r7bb = validate_registry(
        {**empty, "entries": [entry_r7bb]}, make_r7bb
    )
    expected_r7bb = f"disposition-invalid:{name_r7bb}"

    name_r8a = "synthetic-r8a"
    test_r8a = "tests/synthetic_r8a.c"
    make_r8a = f"r8a-target:\n\tcc -o $@ {test_r8a}\n"
    builds_r8a = (
        BatteryBuild(name_r8a, ("cc", test_r8a, "src/battery_only.c")),
    )
    pairs_r8a, unpaired_r8a = compare_build_specs(builds_r8a, make_r8a)
    drifts_r8a = [
        pair.reason
        for pair in pairs_r8a
        if pair.battery_sources != pair.make_sources
    ]
    expected_r8a = f"build-spec-drift:{name_r8a}"

    name_r8b = "synthetic-r8b"
    test_r8b = "tests/synthetic_r8b.c"
    make_r8b = (
        "R8B_EXTRA = src/recipe_only.c\n"
        f"r8b-target:\n\tcc -o $@ {test_r8b} $(R8B_EXTRA)\n"
    )
    builds_r8b = (BatteryBuild(name_r8b, ("cc", test_r8b)),)
    pairs_r8b, unpaired_r8b = compare_build_specs(builds_r8b, make_r8b)
    drifts_r8b = [
        pair.reason
        for pair in pairs_r8b
        if pair.battery_sources != pair.make_sources
    ]
    expected_r8b = f"build-spec-drift:{name_r8b}"

    name_r8c = "synthetic-r8c"
    test_r8c = "tests/synthetic_r8c.c"
    shared_r8c = "src/shared_r8c.c"
    make_r8c = f"r8c-target:\n\tcc -o $@ {test_r8c} {shared_r8c}\n"
    builds_r8c = (BatteryBuild(name_r8c, ("cc", test_r8c, shared_r8c)),)
    pairs_r8c, unpaired_r8c = compare_build_specs(builds_r8c, make_r8c)
    drifts_r8c = [
        pair.reason
        for pair in pairs_r8c
        if pair.battery_sources != pair.make_sources
    ]
    expected_r8c = f"no-build-spec-drift:{name_r8c}"

    expected_r8d = "build-spec-drift:Z-a2-09"
    passed_r8d = retro_a209_negative_control(
        make_text
        if make_text is not None
        else MAKEFILE_PATH.read_text(encoding="utf-8")
    )

    name_r9 = "synthetic-r9"
    target_r9 = "synthetic-r9-target"
    test_r9 = "tests/synthetic_r9.c"
    shared_r9 = "src/shared_r9.c"
    battery_only_r9 = "src/battery_only_r9.c"
    make_r9 = f"{target_r9}:\n\tcc -o $@ {test_r9} {shared_r9}\n"
    builds_r9 = (
        BatteryBuild(name_r9, ("cc", test_r9, shared_r9, battery_only_r9)),
    )
    pairs_r9, unpaired_r9 = compare_build_specs(builds_r9, make_r9)
    row_r9 = {
        "build": name_r9,
        "make_target": target_r9,
        "battery_only": [battery_only_r9],
        "make_only": [],
        "shared": [shared_r9, test_r9],
        "registered": "2026-08-21",
        "repair_rung": "SH-BS-1",
    }

    failures_r9a, observed_r9a = evaluate_known_drifts(pairs_r9, [row_r9])
    expected_r9a = f"known-drift-observed:{name_r9}"
    expected_line_r9a = (
        "[GATE-REGISTRY][BUILD-SPEC][KNOWN-DRIFT] "
        f"reason={expected_r9a} make_target={target_r9} "
        f"battery_only={battery_only_r9} make_only=- repair_rung=SH-BS-1"
    )
    observed_lines_r9a = [
        format_known_drift_observation(pair, row) for pair, row in observed_r9a
    ]

    row_r9b = {
        **row_r9,
        "battery_only": [battery_only_r9, "src/mutated_r9.c"],
    }
    failures_r9b, observed_r9b = evaluate_known_drifts(pairs_r9, [row_r9b])
    row_r9b_shared = {**row_r9, "shared": [test_r9]}
    failures_r9b_shared, observed_r9b_shared = evaluate_known_drifts(
        pairs_r9, [row_r9b_shared]
    )
    expected_r9b = f"known-drift-signature-drift:{name_r9}"

    failures_r9c, observed_r9c = evaluate_known_drifts(pairs_r9, [])
    expected_r9c = f"build-spec-drift:{name_r9}"

    resolved_builds_r9 = (BatteryBuild(name_r9, ("cc", test_r9, shared_r9)),)
    resolved_pairs_r9, resolved_unpaired_r9 = compare_build_specs(
        resolved_builds_r9, make_r9
    )
    failures_r9d, observed_r9d = evaluate_known_drifts(
        resolved_pairs_r9, [row_r9]
    )
    expected_r9d = f"known-drift-unexpected-resolve:{name_r9}"

    row_r9e_missing = {
        key: value for key, value in row_r9.items() if key != "registered"
    }
    row_r9e_bad_rung = {**row_r9, "repair_rung": "sh-bs-1"}
    actual_r9e_missing = validate_known_drift_row(row_r9e_missing)
    actual_r9e_bad_rung = validate_known_drift_row(row_r9e_bad_rung)
    expected_r9e = f"known-drift-row-incomplete:{name_r9}"

    failures_r9f, observed_r9f = evaluate_known_drifts([], [row_r9])
    expected_r9f = f"known-drift-pair-missing:{name_r9}"

    name_r9g = "synthetic-r9g"
    entry_r9g = {
        "name": name_r9g,
        "kind": "make-target",
        "commands": [[sys.executable, "-c", "raise SystemExit(1)"]],
        "category": "known-red",
        "wiring": "SH-UW-1",
        "known_red": {
            "failing_command_index": 0,
            "rc": 1,
            "first_fail_line": "[NC-R9g][FAIL] sentinel",
            "fail_line_count": 1,
            "registered": "2026-08-21",
            "repair_rung": "SH-UW-1",
        },
    }
    actual_r9g = validate_known_red_row(entry_r9g)
    expected_r9g = "validate_known_red_row=None"

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
        ("NC-R7b-a", expected_r7ba, errors_r7ba == [expected_r7ba]),
        ("NC-R7b-b", expected_r7bb, errors_r7bb == [expected_r7bb]),
        (
            "NC-R8a",
            expected_r8a,
            len(pairs_r8a) == 1
            and not unpaired_r8a
            and drifts_r8a == [expected_r8a]
            and pairs_r8a[0].battery_only == ("src/battery_only.c",)
            and pairs_r8a[0].make_only == (),
        ),
        (
            "NC-R8b",
            expected_r8b,
            len(pairs_r8b) == 1
            and not unpaired_r8b
            and drifts_r8b == [expected_r8b]
            and pairs_r8b[0].battery_only == ()
            and pairs_r8b[0].make_only == ("src/recipe_only.c",),
        ),
        (
            "NC-R8c",
            expected_r8c,
            len(pairs_r8c) == 1 and not unpaired_r8c and drifts_r8c == [],
        ),
        ("NC-R8d", expected_r8d, passed_r8d),
        (
            "NC-R9a",
            expected_r9a,
            len(pairs_r9) == 1
            and not unpaired_r9
            and validate_known_drift_row(row_r9) is None
            and failures_r9a == []
            and observed_lines_r9a == [expected_line_r9a],
        ),
        (
            "NC-R9b",
            expected_r9b,
            [failure.reason for failure in failures_r9b] == [expected_r9b]
            and observed_r9b == []
            and [failure.reason for failure in failures_r9b_shared]
            == [expected_r9b]
            and observed_r9b_shared == [],
        ),
        (
            "NC-R9c",
            expected_r9c,
            [failure.reason for failure in failures_r9c] == [expected_r9c]
            and observed_r9c == [],
        ),
        (
            "NC-R9d",
            expected_r9d,
            len(resolved_pairs_r9) == 1
            and not resolved_unpaired_r9
            and [failure.reason for failure in failures_r9d] == [expected_r9d]
            and observed_r9d == [],
        ),
        (
            "NC-R9e",
            expected_r9e,
            actual_r9e_missing == expected_r9e
            and actual_r9e_bad_rung == expected_r9e,
        ),
        (
            "NC-R9f",
            expected_r9f,
            [failure.reason for failure in failures_r9f] == [expected_r9f]
            and observed_r9f == [],
        ),
        ("NC-R9g", expected_r9g, actual_r9g is None),
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

    build_spec_failed = False
    try:
        builds = load_battery_builds()
        build_pairs, unpaired = compare_build_specs(builds, make_text)
    except Exception:
        print(
            "[GATE-REGISTRY][BUILD-SPEC][FAIL] "
            "reason=battery-contract-unreadable"
        )
        build_spec_failed = True
    else:
        print(
            "[GATE-REGISTRY][BUILD-SPEC][SCOPE] "
            f"pairs={len(build_pairs)} unpaired={len(unpaired)}"
        )
        build_spec_failures, observed_drifts = evaluate_known_drifts(
            build_pairs, registry["known_drifts"]
        )
        for pair, row in observed_drifts:
            print(format_known_drift_observation(pair, row))
        for failure in build_spec_failures:
            if failure.pair is None:
                print(
                    "[GATE-REGISTRY][BUILD-SPEC][FAIL] "
                    f"reason={failure.reason}"
                )
                continue
            pair = failure.pair
            print(
                "[GATE-REGISTRY][BUILD-SPEC][FAIL] "
                f"reason={failure.reason} make_target={pair.make_target} "
                f"battery_only={','.join(pair.battery_only) or '-'} "
                f"make_only={','.join(pair.make_only) or '-'}"
            )
        if not build_spec_failures:
            print(
                "[GATE-REGISTRY][BUILD-SPEC][PASS] "
                f"pairs={len(build_pairs)} unpaired={len(unpaired)} "
                f"known_drifts={len(registry['known_drifts'])}"
            )
        build_spec_failed = bool(build_spec_failures)

    known_red = [
        entry for entry in registry["entries"] if entry["category"] == "known-red"
    ]
    red_errors = [reason for entry in known_red if (reason := observe_known_red(entry))]
    if red_errors:
        for error in red_errors:
            print(f"[GATE-REGISTRY][KNOWN-RED][FAIL] reason={error}")
        return 1
    print(f"[GATE-REGISTRY][KNOWN-RED][PASS] entries={len(known_red)}")

    nc_results = run_negative_controls(make_text)
    nc_failed = False
    for nc_name, expected, passed in nc_results:
        state = "PASS" if passed else "FAIL"
        print(
            f"[GATE-REGISTRY][NEGATIVE-CONTROL][{state}] "
            f"name={nc_name} reason={expected}"
        )
        nc_failed = nc_failed or not passed
    return 1 if build_spec_failed or nc_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
