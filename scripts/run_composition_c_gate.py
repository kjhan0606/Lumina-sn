#!/usr/bin/env python3
"""Run the preregistered composition C gates from order v4, sections 8a-8c.

G6 uses the order's permitted static alternative.  Running the expander's
complete main path requires the external CMFGEN atomic-data tree and performs a
large atomic-data regeneration; the static check proves that the abundance
writer is absent and that main has no abundance-writing path.
"""

from __future__ import annotations

import ast
import csv
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "data/tardis_reference_toy06_19p48d/abundances.csv"
DECKS = (
    ROOT / "data/tardis_reference_toy06_19p48d_sivcaiv_fullcov",
    ROOT / "data/tardis_reference_toy06_19p48d_sivcaiv_links",
    ROOT / "data/tardis_reference_toy06_19p48d_sivcaiv_ftos",
)
DEFECTIVE_NAME = "abundances.csv.defective_20260803"
EXPANDER = ROOT / "scripts/expand_atomic_data_cmfgen.py"
G7_SEAL = ROOT / "docs/G7_PRE_HASHES_20260803.txt"
EXPECTED_ELEMENTS = {28, 27, 26, 20, 16, 14, 8, 6}
ZERO_ELEMENTS = {8, 6}
G7_NAMES = ("levels.csv", "line_list.csv", "cmfgen_sigma_bf.bin")


def display(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def cmp_equal(left: Path, right: Path) -> bool:
    return subprocess.run(
        ["cmp", "-s", str(left), str(right)], check=False
    ).returncode == 0


def check_g1(decks: Iterable[Path]) -> tuple[bool, str]:
    comparisons = [
        (deck, cmp_equal(deck / "abundances.csv", CANONICAL)) for deck in decks
    ]
    detail = ", ".join(
        f"{display(deck)}={'identical' if equal else 'different'}"
        for deck, equal in comparisons
    )
    return all(equal for _, equal in comparisons), detail


def composition_shape(deck: Path) -> tuple[bool, str]:
    with (deck / "abundances.csv").open(newline="") as stream:
        rows = csv.reader(stream)
        header = next(rows, [])
        abundance_columns = max(len(header) - 1, 0)
        data_widths = [max(len(row) - 1, 0) for row in rows if row]
    with (deck / "geometry.csv").open(newline="") as stream:
        rows = csv.reader(stream)
        next(rows, None)
        geometry_rows = sum(1 for row in rows if row)
    widths_match = all(width == abundance_columns for width in data_widths)
    passed = abundance_columns == geometry_rows == 50 and widths_match
    detail = (
        f"{display(deck)}: abundance_columns={abundance_columns}, "
        f"geometry_rows={geometry_rows}, data_widths_match={widths_match}"
    )
    return passed, detail


def check_g2(decks: Iterable[Path]) -> tuple[bool, str]:
    results = [composition_shape(deck) for deck in decks]
    return all(passed for passed, _ in results), "; ".join(
        detail for _, detail in results
    )


def read_abundances(deck: Path) -> list[tuple[int, list[float]]]:
    parsed: list[tuple[int, list[float]]] = []
    with (deck / "abundances.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            values = list(row.values())
            parsed.append((int(values[0]), [float(value) for value in values[1:]]))
    return parsed


def check_g3() -> tuple[bool, str]:
    details = []
    all_passed = True
    for deck in DECKS:
        rows = read_abundances(deck)
        elements = [z for z, _ in rows]
        zero_violations = [
            (z, shell, value)
            for z, values in rows if z in ZERO_ELEMENTS
            for shell, value in enumerate(values) if value != 0.0
        ]
        passed = (
            set(elements) == EXPECTED_ELEMENTS
            and len(elements) == len(EXPECTED_ELEMENTS)
            and not zero_violations
        )
        all_passed &= passed
        details.append(
            f"{display(deck)}: elements={sorted(elements)}, "
            f"O/C_nonzero={len(zero_violations)}"
        )
    return all_passed, "; ".join(details)


def check_g4() -> tuple[bool, str]:
    details = []
    all_passed = True
    for deck in DECKS:
        rows = read_abundances(deck)
        n_shells = len(rows[0][1]) if rows else 0
        deviations = []
        for shell in range(n_shells):
            total = 0.0
            for _, values in rows:
                total += values[shell]
            deviations.append(abs(total - 1.0))
        max_deviation = max(deviations, default=float("inf"))
        violations = sum(value > 1.0e-12 for value in deviations)
        all_passed &= violations == 0
        details.append(
            f"{display(deck)}: max_deviation={max_deviation:.17g}, "
            f"violations={violations}"
        )
    return all_passed, "; ".join(details)


def check_g5() -> tuple[bool, str]:
    details = []
    all_passed = True
    for deck in DECKS:
        constant = [
            z for z, values in read_abundances(deck)
            if z not in ZERO_ELEMENTS and values
            and all(value == values[0] for value in values[1:])
        ]
        all_passed &= not constant
        details.append(f"{display(deck)}: constant_non_O/C={constant}")
    return all_passed, "; ".join(details)


def check_g6() -> tuple[bool, str]:
    tree = ast.parse(EXPANDER.read_text(encoding="utf-8"), filename=str(EXPANDER))
    writer_definitions = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "write_abundances"
    ]
    main_nodes = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "main"
    ]
    main_calls_writer = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "write_abundances"
        for main in main_nodes for node in ast.walk(main)
    )
    main_names_abundance_output = any(
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and "abundances.csv" in node.value
        for main in main_nodes for node in ast.walk(main)
    )
    passed = (
        len(main_nodes) == 1
        and not writer_definitions
        and not main_calls_writer
        and not main_names_abundance_output
    )
    return passed, (
        "static alternative: "
        f"main_count={len(main_nodes)}, writer_definitions={len(writer_definitions)}, "
        f"main_calls_writer={main_calls_writer}, "
        f"main_names_abundance_output={main_names_abundance_output}"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_g7_seal() -> dict[Path, str]:
    sealed = {}
    for raw in G7_SEAL.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        digest, relative = line.split(maxsplit=1)
        sealed[ROOT / relative] = digest
    return sealed


def check_g7() -> tuple[bool, str]:
    sealed = read_g7_seal()
    expected_paths = {deck / name for deck in DECKS for name in G7_NAMES}
    path_set_matches = set(sealed) == expected_paths
    mismatches = [
        display(path) for path in sorted(expected_paths)
        if sealed.get(path) != sha256_file(path)
    ]
    passed = path_set_matches and not mismatches
    return passed, (
        f"sealed_paths={len(sealed)}, expected_paths={len(expected_paths)}, "
        f"path_set_matches={path_set_matches}, mismatches={mismatches}"
    )


def check_negative_controls() -> list[tuple[str, bool, str]]:
    defective = DECKS[0] / DEFECTIVE_NAME
    with tempfile.TemporaryDirectory(prefix="composition-c-negative-") as tmp:
        deck = Path(tmp) / "defective_30_column_deck"
        deck.mkdir()
        shutil.copy2(defective, deck / "abundances.csv")
        shutil.copy2(DECKS[0] / "geometry.csv", deck / "geometry.csv")
        g1_passed, g1_detail = check_g1((deck,))
        g2_passed, g2_detail = check_g2((deck,))
    return [
        ("NEG-G1", not g1_passed, f"candidate G1 must FAIL: {g1_detail}"),
        ("NEG-G2", not g2_passed, f"candidate G2 must FAIL: {g2_detail}"),
    ]


def main() -> int:
    checks = [
        ("G1", lambda: check_g1(DECKS)),
        ("G2", lambda: check_g2(DECKS)),
        ("G3", check_g3),
        ("G4", check_g4),
        ("G5", check_g5),
        ("G6", check_g6),
        ("G7", check_g7),
    ]
    results: list[tuple[str, bool, str]] = []
    for gate, check in checks:
        try:
            passed, detail = check()
        except Exception as exc:  # Keep later gates observable after one failure.
            passed, detail = False, f"{type(exc).__name__}: {exc}"
        results.append((gate, passed, detail))
    try:
        results.extend(check_negative_controls())
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        results.extend((gate, False, detail) for gate in ("NEG-G1", "NEG-G2"))

    for gate, passed, detail in results:
        print(f"{'PASS' if passed else 'FAIL'} {gate}: {detail}")
    passed_count = sum(passed for _, passed, _ in results)
    print(f"SUMMARY: {passed_count}/{len(results)} checks passed")
    return 0 if passed_count == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
