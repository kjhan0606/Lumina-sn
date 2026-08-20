#!/usr/bin/env python3
"""Self-test the definition anchoring and lexical views used by source gates."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from gate_source_lib import body, body_raw, find_definition, inject_at_head


def require(condition: bool, case: str) -> None:
    if not condition:
        raise AssertionError(case)


def require_error(text: str, name: str, expected: str, case: str) -> None:
    try:
        find_definition(text, name)
    except RuntimeError as error:
        require(str(error) == expected, case)
    else:
        raise AssertionError(case)


def main() -> int:
    calls_and_definition = """
int choose(int value) {
    return value;
}
int caller(int value) {
    if (choose((value))) { return choose(value); }
    return 0;
}
"""
    start, end = find_definition(calls_and_definition, "choose")
    require(calls_and_definition[start:end] == body_raw(
        calls_and_definition, "choose"), "definition-not-selected")
    require(body_raw(calls_and_definition, "choose") == "{\n    return value;\n}",
            "call-site-selected")

    duplicate = "int wrapper(void) { return 1; }\nint wrapper(void) { return 2; }\n"
    require_error(duplicate, "wrapper", "ambiguous function anchor: wrapper",
                  "duplicate-definition-not-rejected")
    require_error("int caller(void) { return missing(); }", "missing",
                  "function not found: missing", "missing-definition-not-rejected")

    comments_and_string = """
int lexical(void) {
    // LINE_COMMENT_TOKEN
    /* BLOCK_COMMENT_TOKEN
       BLOCK_COMMENT_SECOND_LINE */
    const char *reason = "STRING_TOKEN /* string is preserved */";
    return 0;
}
"""
    raw = body_raw(comments_and_string, "lexical")
    cleaned = body(comments_and_string, "lexical")
    require("LINE_COMMENT_TOKEN" in raw and "BLOCK_COMMENT_TOKEN" in raw,
            "comment-fixture-invalid")
    require("LINE_COMMENT_TOKEN" not in cleaned
            and "BLOCK_COMMENT_TOKEN" not in cleaned,
            "comment-token-survived")
    require('"STRING_TOKEN /* string is preserved */"' in cleaned,
            "string-token-removed")
    require(len(cleaned) == len(raw) and cleaned.count("\n") == raw.count("\n"),
            "line-structure-not-preserved")

    injected = inject_at_head(calls_and_definition, "choose", "int injected = 1;")
    require("{\n    int injected = 1;\n    return value;" in
            body_raw(injected, "choose"), "head-injection-misplaced")

    print("PASS GATE_SOURCE_LIB_SELFTEST cases=definition-vs-calls,ambiguous,"
          "missing,comments,string,inject-at-head")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
