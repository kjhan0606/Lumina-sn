#!/usr/bin/env python3
"""CPU-only positive/negative fixtures for the strict CMFGEN f_to_s parser."""

from __future__ import annotations

from pathlib import Path
import tempfile

from cmfgen_parser import parse_f_to_s


LINKS = Path("/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/atomic_links.txt")
SI2 = Path("/gpfs/kjhan/cmfgen_21jun23/atomic/SIL/II/19apr23/f_to_s_79")
SI4_OLD = Path(
    "/gpfs/kjhan/cmfgen_21jun23/atomic/SIL/IV/5dec96/f_to_s_split.dat"
)


def expect_fail(label: str, call) -> None:
    try:
        call()
    except ValueError as exc:
        print(f"NEGATIVE {label}: FAIL caught as required — {exc}")
    else:
        raise AssertionError(f"negative fixture {label} was silently accepted")


def write_fixture(path: Path, n_levels: int, rows: list[str]) -> None:
    path.write_text(
        f"{n_levels} !Number of energy levels\n"
        "6 !Entry number of link to super level\n" +
        "\n".join(rows) + "\n",
        encoding="latin-1",
    )


def main() -> None:
    if not LINKS.is_file() or not SI2.is_file() or not SI4_OLD.is_file():
        raise SystemExit("R4 canonical CMFGEN fixtures are unavailable")

    explicit = parse_f_to_s(SI2)
    implicit = parse_f_to_s(SI4_OLD)
    assert (explicit.n_levels, explicit.n_super, explicit.format_name) == (
        157, 79, "explicit_fl_id"
    )
    assert (implicit.n_levels, implicit.n_super, implicit.format_name) == (
        66, 55, "implicit_fl_row_order"
    )
    print("POSITIVE explicit Si II: 157 FL -> 79 SL PASS")
    print("POSITIVE implicit Si IV: 66 FL -> 55 SL PASS")

    # Exact regression for the failed R4 attempt: inject the old defect by
    # forcing the implicit file down the explicit-final-column lane.  Its last
    # column is the auxiliary zero, not an FL ID, so strict validation must stop.
    expect_fail(
        "prior bug: Si IV final zero misread as explicit FL ID",
        lambda: parse_f_to_s(SI4_OLD, _test_force_format="explicit_fl_id"),
    )

    # A second column-slip test on a genuine explicit file selects its
    # penultimate auxiliary-zero column.  It must not fall back to row order.
    expect_fail(
        "explicit FL column shifted final->penultimate",
        lambda: parse_f_to_s(SI2, _test_explicit_fl_column_offset=-1),
    )

    with tempfile.TemporaryDirectory(prefix="r4_ftos_") as tmp:
        root = Path(tmp)
        good = [
            "a 1.0 0.0 1.0 1.0 1 0 1",
            "b 1.0 1.0 1.0 1.0 2 0 2",
            "c 1.0 2.0 1.0 1.0 3 0 3",
        ]

        short = root / "short.dat"
        write_fixture(short, 3, good[:2])
        expect_fail("mapped FL count != declared FL count",
                    lambda: parse_f_to_s(short))

        duplicate = root / "duplicate.dat"
        duplicate_rows = good.copy()
        duplicate_rows[-1] = "c 1.0 2.0 1.0 1.0 3 0 2"
        write_fixture(duplicate, 3, duplicate_rows)
        expect_fail("FL ID not exactly once",
                    lambda: parse_f_to_s(duplicate))

        hole = root / "sl_hole.dat"
        hole_rows = good.copy()
        hole_rows[1] = "b 1.0 1.0 1.0 1.0 3 0 2"
        hole_rows[2] = "c 1.0 2.0 1.0 1.0 3 0 3"
        write_fixture(hole, 3, hole_rows)
        expect_fail("SL numbering hole",
                    lambda: parse_f_to_s(hole))

    print("R4 f_to_s fixture SELF-CHECK PASS")


if __name__ == "__main__":
    main()
