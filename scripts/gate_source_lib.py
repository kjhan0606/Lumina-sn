#!/usr/bin/env python3
"""Definition-anchored C source extraction helpers for structural gates."""

from __future__ import annotations

import re


def _blank_non_newline(buffer: list[str], start: int, end: int) -> None:
    for index in range(start, end):
        if buffer[index] != "\n":
            buffer[index] = " "


def _lexical_view(text: str, *, blank_literals: bool) -> str:
    """Blank comments, and optionally literals, without changing offsets."""
    result = list(text)
    index = 0
    while index < len(text):
        if text.startswith("//", index):
            end = text.find("\n", index + 2)
            if end < 0:
                end = len(text)
            _blank_non_newline(result, index, end)
            index = end
            continue
        if text.startswith("/*", index):
            close = text.find("*/", index + 2)
            end = len(text) if close < 0 else close + 2
            _blank_non_newline(result, index, end)
            index = end
            continue
        if text[index] in {'"', "'"}:
            quote = text[index]
            end = index + 1
            while end < len(text):
                if text[end] == "\\":
                    end = min(end + 2, len(text))
                    continue
                end += 1
                if text[end - 1] == quote:
                    break
            if blank_literals:
                _blank_non_newline(result, index, end)
            index = end
            continue
        index += 1
    return "".join(result)


def _definition_spans(text: str, name: str) -> list[tuple[int, int]]:
    structural = _lexical_view(text, blank_literals=True)
    candidate = re.compile(r"\b" + re.escape(name) + r"\s*\(")
    spans: list[tuple[int, int]] = []
    for match in candidate.finditer(structural):
        opening_parenthesis = match.end() - 1
        depth = 0
        closing_parenthesis: int | None = None
        for position in range(opening_parenthesis, len(structural)):
            character = structural[position]
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                if depth == 0:
                    closing_parenthesis = position
                    break
        if closing_parenthesis is None:
            raise RuntimeError(f"unterminated parameter list: {name}")

        opening_brace = closing_parenthesis + 1
        while (opening_brace < len(structural)
               and structural[opening_brace].isspace()):
            opening_brace += 1
        if opening_brace >= len(structural) or structural[opening_brace] != "{":
            continue

        depth = 0
        for position in range(opening_brace, len(structural)):
            character = structural[position]
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    spans.append((opening_brace, position + 1))
                    break
        else:
            raise RuntimeError(f"unterminated function: {name}")
    return spans


def find_definition(text: str, name: str) -> tuple[int, int]:
    """Return the unique definition's brace-delimited half-open span."""
    spans = _definition_spans(text, name)
    if not spans:
        raise RuntimeError(f"function not found: {name}")
    if len(spans) > 1:
        raise RuntimeError(f"ambiguous function anchor: {name}")
    return spans[0]


def body_raw(text: str, name: str) -> str:
    """Return the unique definition body exactly as written."""
    start, end = find_definition(text, name)
    return text[start:end]


def body(text: str, name: str) -> str:
    """Return its body with comments blanked and string literals preserved."""
    return _lexical_view(body_raw(text, name), blank_literals=False)


def inject_at_head(text: str, name: str, stmt: str) -> str:
    """Insert a statement immediately inside the unique definition body."""
    start, _ = find_definition(text, name)
    return text[:start + 1] + "\n    " + stmt + text[start + 1:]
