#!/usr/bin/env python3
"""Apply the sealed fixed-T Stage A controls to an isolated CMFGEN copy."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def set_control(text: str, key: str, value: str, *, append: bool = False) -> str:
    pattern = re.compile(
        rf"^(?![ \t]*[!#])([ \t]*)\S+([ \t]+\[{re.escape(key)}\].*)$",
        re.MULTILINE,
    )
    matches = list(pattern.finditer(text))
    if len(matches) > 1:
        raise ValueError(f"[{key}] occurs {len(matches)} times")
    if not matches:
        if not append:
            raise ValueError(f"missing [{key}]")
        return text.rstrip() + f"\n{value:<13}[{key}] ! Stage A\n"
    match = matches[0]
    return text[: match.start()] + match.group(1) + value + match.group(2) + text[match.end() :]


def configure(vadat_path: Path, in_its_path: Path) -> None:
    vadat = vadat_path.read_text()
    for key, value, append in (
        ("FIX_T", "T", False),
        ("FIX_T_AUTO", "F", False),
        ("DO_DDT", "F", False),
        ("TRT_NON_TE", "F", False),
        ("MAX_LIN", "3.0D0", False),
        ("MAX_LAM", "10.0D0", False),
        ("DO_NG", "F", False),
        ("WRITE_RATES", "T", True),
        ("WRITE_JH", "T", True),
    ):
        vadat = set_control(vadat, key, value, append=append)

    # Request full linearization at startup. DO_LAM_AUTO only controls
    # promotion out of explicitly requested RD_LAMBDA mode; solve_for_pops
    # can still schedule native BA/LAMBDA alternation when RD_LAMBDA is false.
    # Preserve this historical request; do not claim 19 full BA iterations.
    in_its = (
        "19           [NUM_ITS]       ! Stage A cap: 19 additional iterations\n"
        "F            [DO_LAM_IT]     ! full linearization from the first iteration\n"
        "F            [DO_LAM_AUTO]   ! prohibit automatic LAMBDA transition\n"
        "F            [DO_T_AUTO]     ! prohibit automatic temperature release\n"
    )
    vadat_path.write_text(vadat)
    in_its_path.write_text(in_its)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("vadat", type=Path)
    parser.add_argument("in_its", type=Path)
    args = parser.parse_args()
    configure(args.vadat, args.in_its)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
