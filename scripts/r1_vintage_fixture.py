#!/usr/bin/env python3
"""Small CPU-only fixture for CMFGEN_LINKS parsing and selection semantics."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[1]
EXPAND = ROOT / "scripts/expand_atomic_data_cmfgen.py"
ATOMIC = ROOT / "data/atomic/cmfgen"


def load_expand():
    # Keep the fixture independent of an inherited production selection.
    for name in ("CMFGEN_LINKS", "CMFGEN_VINTAGE_MATCH",
                 "CMFGEN_FULL_LEVELS", "CMFGEN_SUPER_LEVELS"):
        os.environ.pop(name, None)
    spec = importlib.util.spec_from_file_location("r1_fixture_expand", EXPAND)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {EXPAND}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def symlink(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source.resolve(strict=True))


def main() -> int:
    expand = load_expand()
    with tempfile.TemporaryDirectory(prefix="r1_vintage_fixture_", dir="/tmp") as tmp:
        fixture = Path(tmp)
        atomic = fixture / "atomic"

        # Unlinked S II has an older directory but must fall through to latest.
        (atomic / "SUL/II/3oct00").mkdir(parents=True)
        for name in ("osc_data", "phot_data_A", "col_data"):
            symlink(ATOMIC / "SUL/II/19apr23" / name,
                    atomic / "SUL/II/19apr23" / name)

        # Linked S V deliberately points at 3oct00 while a valid 19apr23 exists.
        old_sources = {
            "osc": "svosc_fin.dat",
            "f_to_s": "f_to_s_50.dat",
            "phot": "phot_sm_3000.dat",
            "col": "col_sv.dat",
        }
        for name in old_sources.values():
            symlink(ATOMIC / "SUL/V/3oct00" / name,
                    atomic / "SUL/V/3oct00" / name)
        for name in ("osc_data", "phot_data_A", "col_data"):
            symlink(ATOMIC / "SUL/V/19apr23" / name,
                    atomic / "SUL/V/19apr23" / name)

        links_path = fixture / "atomic_links.txt"
        old = atomic / "SUL/V/3oct00"
        links_path.write_text(
            f"ln -sf {old / old_sources['osc']} SV_F_OSCDAT\n"
            f"ln -sf {old / old_sources['f_to_s']} SV_F_TO_S\n"
            f"ln -sf {old / old_sources['phot']} PHOTSV_A\n"
            f"ln -sf {old / old_sources['col']} SV_COL_DATA\n")

        link_map = expand.load_cmfgen_links(links_path)
        assert set(link_map) == {(16, 5)}
        assert set(link_map[(16, 5)]) == set(expand._LINK_KINDS)
        print("FIXTURE parse: represented ions=[(16, 5)], kinds=osc/f_to_s/phot/col PASS")

        expand.CMFGEN_ROOT = atomic
        expand.ION_LEVEL_CAPS = {(16, 2): 3, (16, 5): 3}
        expand.CMFGEN_LINK_MAP = link_map
        expand.SUPER_LEVEL_ENABLED = False
        expand.VINTAGE_MATCH = False
        parsed = expand.parse_all_ions()

        sii = parsed[(16, 2)]["provenance"]
        sv = parsed[(16, 5)]["provenance"]
        assert sii["selection_source"] == "auto"
        assert sii["latest_vintage"] == "19apr23"
        assert sii["osc_path"].parent.name == "19apr23"
        print("NEGATIVE absent-link fallback: S II selected auto/19apr23 PASS")

        assert sv["selection_source"] == "links"
        assert sv["latest_vintage"] == "19apr23"
        for kind in ("osc", "phot", "col"):
            assert sv[f"{kind}_path"] == link_map[(16, 5)][kind]
            assert sv[f"{kind}_path"].parent.name == "3oct00"
        assert sv["f_to_s_path"] == link_map[(16, 5)]["f_to_s"]
        print("POSITIVE linked vintage force: S V selected links/3oct00 PASS")
        print("NEGATIVE linked-no-latest-leak: S V latest=19apr23 but all linked inputs stay 3oct00 PASS")
        print("FIXTURE VERDICT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
