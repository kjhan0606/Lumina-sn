#!/usr/bin/env python3
"""Rebake only ``cmfgen_sigma_bf.bin`` for an existing, sealed deck.

The deck's level order and CMFGEN link provenance are treated as immutable
inputs.  The output path must not exist.  No row padding, first-bin fill, or
old-grid interpolation is available in this program: every row is evaluated
again from the linked CMFGEN photoionization source on the baker's canonical
grid.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
EXPAND = ROOT / "scripts/expand_atomic_data_cmfgen.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_expand():
    spec = importlib.util.spec_from_file_location("sh_grid_sigma_expand", EXPAND)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {EXPAND}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    deck = args.deck.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output}")
    provenance_path = deck / "DECK_PROVENANCE.json"
    active_path = deck / "active_ions.csv"
    levels_path = deck / "levels.csv"
    if not provenance_path.is_file() or not active_path.is_file() or not levels_path.is_file():
        raise SystemExit("deck must contain DECK_PROVENANCE.json, active_ions.csv and levels.csv")

    provenance = json.loads(provenance_path.read_text())
    env = provenance.get("env", {})
    required = {
        "CMFGEN_FULL_LEVELS": "1",
        "CMFGEN_SUPER_LEVELS": "1",
        "CMFGEN_LINK_FTOS": "1",
    }
    for name, expected in required.items():
        if env.get(name) != expected:
            raise SystemExit(f"deck provenance has {name}={env.get(name)!r}, expected {expected!r}")
        os.environ[name] = expected
    # The params[0]-as-sigma_0 path for CMFGEN types 2/3/8/9 is a historical
    # stand-in, not a physical cross-section.  A fresh canonical rebake must
    # never silently inherit that default.  Legacy deck provenance may omit
    # this key; a contradictory explicit value is rejected.
    exact_hyd = env.get("CMFGEN_EXACT_HYD")
    if exact_hyd not in (None, "1"):
        raise SystemExit(
            f"deck provenance has CMFGEN_EXACT_HYD={exact_hyd!r}, expected '1'"
        )
    os.environ["CMFGEN_EXACT_HYD"] = "1"
    links = Path(provenance.get("cmfgen_links", ""))
    if not links.is_file() or sha256(links) != provenance.get("cmfgen_links_sha256"):
        raise SystemExit(f"CMFGEN links missing or hash-stale: {links}")
    os.environ["CMFGEN_LINKS"] = str(links)

    with active_path.open(newline="") as stream:
        active_rows = list(csv.DictReader(stream))
    caps: dict[tuple[int, int], int] = {}
    for row in active_rows:
        key = (int(row["atomic_number"]), int(row["ion_stage"]))
        if key in caps:
            raise SystemExit(f"duplicate active ion {key}")
        caps[key] = int(row["n_full"])
        for field, hash_field in (("osc_path", "osc_sha256"),
                                  ("f_to_s_path", "f_to_s_sha256")):
            source = Path(row[field])
            if not source.is_file() or sha256(source) != row[hash_field]:
                raise SystemExit(f"active source missing or hash-stale: {source}")

    expand = load_expand()
    expand.ROOT = ROOT
    expand.CMFGEN_ROOT = ROOT / "data/atomic/cmfgen"
    expand.ION_LEVEL_CAPS = caps
    if set(expand.CMFGEN_LINK_MAP) != set(caps):
        raise SystemExit("active_ions.csv and CMFGEN_LINKS ion sets differ")
    ion_data = expand.parse_all_ions()
    if set(ion_data) != set(caps):
        raise SystemExit("parsed CMFGEN ion set differs from active deck")

    active_by_key = {
        (int(row["atomic_number"]), int(row["ion_stage"])): row
        for row in active_rows
    }
    for key, data in ion_data.items():
        nf = caps[key]
        row = active_by_key[key]
        if data["osc"].n_levels < nf or data.get("ftos") is None:
            raise SystemExit(f"linked source cannot supply NF/f_to_s for {key}")
        data["n_kept"] = nf
        data["levels"] = data["osc"].levels[:nf]
        trans = data["trans"]
        data["trans"] = trans[(trans["i"] <= nf) & (trans["j"] <= nf)]
        prefix = data["ftos"].sl_of_fl[:nf]
        if len(set(prefix.tolist())) != int(row["n_super"]):
            raise SystemExit(f"f_to_s prefix changed for {key}")

    levels_rows, lookup, _ = expand.build_global_levels(ion_data)
    with tempfile.TemporaryDirectory(prefix="lumina-sh-grid-levels-") as tmp:
        generated_levels = Path(tmp) / "levels.csv"
        expand.write_levels_csv(levels_rows, generated_levels)
        if generated_levels.read_bytes() != levels_path.read_bytes():
            raise SystemExit("generated level order is not byte-identical to the sealed deck")

    output.parent.mkdir(parents=True, exist_ok=True)
    expand.bake_sigma_bf_grid(ion_data, ion_data, levels_rows, lookup, output)
    print(
        "[SH-GRID][SIGMA-REBAKE][PASS] "
        f"deck={deck} output={output} levels={len(levels_rows)} "
        f"bins={expand.BF_N_FREQ_BIN} range=[{expand.BF_NU_MIN:.17g},"
        f"{expand.BF_NU_MAX:.17g}] sha256={sha256(output)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
