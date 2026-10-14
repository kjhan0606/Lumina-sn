#!/usr/bin/env python3
"""No-GPU regression for generated env registration and strict preflight."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import re


REPO = pathlib.Path(__file__).resolve().parents[1]
DERIVER = REPO / "scripts/derive_env_universe.py"
HEADER = REPO / "src/env_universe.h"
REGISTRY = REPO / "validation/instrumentation_debt/ENV_UNIVERSE.json"
TARGET = "LUMINA_DPROXY_POPULATION_PROVENANCE"
TYPO = "LUMINA_DPROXY_POPULATION_PROVENANCE_TYPO"


def load_deriver():
    spec = importlib.util.spec_from_file_location("derive_env_universe", DERIVER)
    if spec is None or spec.loader is None:
        raise AssertionError("cannot load env-universe generator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def derive_in_memory(module):
    by_source = {
        "literal_getenv": set(),
        "env_arrays": set(),
        "constructed": set(),
        "wrapper_calls": set(),
    }
    per_file: dict[str, int] = {}
    for path in module.SRC:
        text = path.read_text(errors="ignore")
        branches = (
            module.literal_getenv(text),
            module.array_members(text),
            module.constructed(text),
            module.wrapper_calls(text),
        )
        for name, values in zip(by_source, branches):
            by_source[name] |= values
        count = len(set().union(*branches))
        if count:
            per_file[path.name] = count

    source_union = set().union(*by_source.values())
    nonsrc, set_only = module.nonsrc_consumers()
    by_source["nonsrc_consumers"] = nonsrc
    union = source_union | nonsrc
    lumina = sorted(name for name in union if name.startswith("LUMINA_"))
    return {
        "counts": {name: len(values) for name, values in by_source.items()},
        "union_total": len(union),
        "lumina_prefixed": len(lumina),
        "non_lumina": sorted(name for name in union if not name.startswith("LUMINA_")),
        "missed_without_wrapper_scan": sorted(
            by_source["wrapper_calls"] - by_source["literal_getenv"]
        ),
        "missed_without_nonsrc_scan": sorted(nonsrc - source_union),
        "set_only_never_read": sorted(set_only - source_union),
        "per_file_hits": dict(sorted(per_file.items(), key=lambda item: -item[1])),
        "env": lumina,
    }


def header_names() -> tuple[int, list[str]]:
    text = HEADER.read_text(encoding="utf-8")
    count_match = re.search(r"^#define LUMINA_ENV_UNIVERSE_COUNT ([0-9]+)$", text, re.M)
    array_match = re.search(
        r"LUMINA_ENV_UNIVERSE\[\]\s*=\s*\{(?P<body>.*?)\n\};",
        text,
        re.S,
    )
    if count_match is None or array_match is None:
        raise AssertionError("generated C env registry is malformed")
    names = re.findall(r'^\s*"(LUMINA_[A-Z0-9_]+)",$', array_match.group("body"), re.M)
    return int(count_match.group(1)), names


def strict_unknown(environment: dict[str, str], known: set[str]) -> list[str]:
    return sorted(
        name
        for name in environment
        if name.startswith("LUMINA_") and name not in known
    )


def main() -> int:
    expected = derive_in_memory(load_deriver())
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    for key, value in expected.items():
        if registry.get(key) != value:
            raise AssertionError(f"generated JSON registry drift: {key}")

    declared_count, declared_names = header_names()
    if declared_names != expected["env"] or declared_count != len(declared_names):
        raise AssertionError("generated C and JSON env registries differ")
    if TARGET not in declared_names:
        raise AssertionError(f"required runtime env is unregistered: {TARGET}")

    known = set(declared_names)
    valid = {"LUMINA_ENV_STRICT": "1", TARGET: "1"}
    if strict_unknown(valid, known):
        raise AssertionError("strict preflight rejected the registered d-proxy env")
    invalid = dict(valid, **{TYPO: "1"})
    if strict_unknown(invalid, known) != [TYPO]:
        raise AssertionError("strict preflight did not reject an unknown d-proxy env")

    print(
        "ENV_UNIVERSE_STRICT_SELFTEST_PASS "
        f"registered={TARGET} universe={declared_count} "
        "positive=1 unknown_negative=1 generated_artifacts_consistent=1"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
