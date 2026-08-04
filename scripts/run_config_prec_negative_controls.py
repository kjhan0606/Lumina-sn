#!/usr/bin/env python3
"""CONFIG-PREC injected-defect controls.

Every child is expected to stop in the common reference-data loader with rc=1.
Fixtures are copies under /tmp; the source deck is never modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Callable


REQUIRED_FILES = (
    "geometry.csv",
    "config.json",
    "electron_densities.csv",
    "plasma_state.csv",
)


def inferred_color(deck: Path) -> float:
    with (deck / "plasma_state.csv").open(newline="") as stream:
        row = next(csv.DictReader(stream))
    return float(row["T_rad"]) / math.pow(float(row["W"]), 0.25)


def set_config_temperature(deck: Path, temperature: float) -> None:
    path = deck / "config.json"
    with path.open() as stream:
        config = json.load(stream)
    config["T_inner_K"] = temperature
    with path.open("w") as stream:
        json.dump(config, stream, indent=2)
        stream.write("\n")


def split_profile(deck: Path) -> None:
    set_config_temperature(deck, inferred_color(deck))
    path = deck / "plasma_state.csv"
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames or len(rows) < 2:
        raise RuntimeError("plasma_state.csv needs at least two rows")
    rows[1]["T_rad"] = format(float(rows[1]["T_rad"]) * 1.02, ".17g")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def no_mutation(_deck: Path) -> None:
    return


def run_case(binary: Path, source_deck: Path, scratch_root: Path,
             name: str, mutation: Callable[[Path], None],
             env_updates: dict[str, str], marker: str) -> bool:
    fixture = scratch_root / name
    fixture.mkdir()
    for filename in REQUIRED_FILES:
        shutil.copy2(source_deck / filename, fixture / filename)
    mutation(fixture)

    env = os.environ.copy()
    env.pop("LUMINA_CONFIG_PREC", None)
    env.pop("LUMINA_T_INNER_FIX", None)
    env.update(env_updates)
    result = subprocess.run(
        [str(binary), str(fixture)],
        cwd=binary.parent,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    ok = result.returncode == 1 and marker in result.stdout
    print(f"CONFIG_PREC_NEG case={name} child_rc={result.returncode} "
          f"marker={'yes' if marker in result.stdout else 'no'} "
          f"verdict={'PASS' if ok else 'FAIL'} fixture={fixture}")
    if not ok:
        print(result.stdout)
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default="./lumina")
    parser.add_argument(
        "--deck", default="data/tardis_reference_toy06_19p48d"
    )
    args = parser.parse_args()

    binary = Path(args.binary).resolve()
    source_deck = Path(args.deck).resolve()
    missing = [name for name in REQUIRED_FILES
               if not (source_deck / name).is_file()]
    if not binary.is_file() or missing:
        print(f"CONFIG_PREC_NEG setup_error binary={binary} missing={missing}",
              file=sys.stderr)
        return 2

    scratch_root = Path(tempfile.mkdtemp(prefix="lumina_config_prec_controls_"))
    print(f"CONFIG_PREC_NEG scratch={scratch_root}")
    color = inferred_color(source_deck)
    cases = (
        (
            "canonical_mismatch",
            no_mutation,
            {"LUMINA_CONFIG_PREC": "1"},
            "[CONFIG-PREC][FATAL] boundary-temperature declarations disagree",
        ),
        (
            "env_cannot_waive_deck_mismatch",
            no_mutation,
            {
                "LUMINA_CONFIG_PREC": "1",
                "LUMINA_T_INNER_FIX": format(color, ".17g"),
            },
            "[CONFIG-PREC][FATAL] boundary-temperature declarations disagree",
        ),
        (
            "split_inferred_color_profile",
            split_profile,
            {"LUMINA_CONFIG_PREC": "1"},
            "[CONFIG-PREC][FATAL] boundary-temperature declarations disagree",
        ),
        (
            "invalid_gate_value",
            no_mutation,
            {"LUMINA_CONFIG_PREC": "true"},
            "[CONFIG-PREC][FATAL] LUMINA_CONFIG_PREC='true' is invalid",
        ),
    )
    passed = sum(
        run_case(binary, source_deck, scratch_root, *case) for case in cases
    )
    print(f"CONFIG_PREC_NEG_SUMMARY passed={passed} total={len(cases)} "
          f"scratch={scratch_root}")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
