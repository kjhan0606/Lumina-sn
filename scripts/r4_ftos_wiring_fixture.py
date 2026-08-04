#!/usr/bin/env python3
"""CPU-only in-memory check that the R4 generator consumes every linked map."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPAND = ROOT / "scripts/expand_atomic_data_cmfgen.py"


def main() -> None:
    if os.environ.get("CMFGEN_LINK_FTOS") != "1":
        raise SystemExit("CMFGEN_LINK_FTOS=1 is required")
    spec = importlib.util.spec_from_file_location("r4_wiring_expand", EXPAND)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {EXPAND}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not module.LINK_FTOS_ENABLED or len(module.CMFGEN_LINK_MAP) != 27:
        raise RuntimeError("R4 link gate did not arm all 27 link sets")

    # This fixture audits selection/membership, not phot/col parsing.  Avoid the
    # unrelated large tables and restrict the generator census to linked ions.
    module.ION_LEVEL_CAPS = {key: None for key in module.CMFGEN_LINK_MAP}
    module.parse_phot = lambda _path: None
    module.parse_col = lambda _path: None
    data = module.parse_all_ions()
    if set(data) != set(module.CMFGEN_LINK_MAP):
        raise RuntimeError("generator linked-ion set differs from atomic_links")
    rows, _lookup, _g = module.build_global_levels(data)

    cursor = 0
    for key, item in sorted(data.items()):
        ftos = item.get("ftos")
        if ftos is None:
            raise RuntimeError(f"generator failed to consume f_to_s for {key}")
        n = ftos.n_levels
        actual = np.asarray([row[6] for row in rows[cursor:cursor + n]], dtype="i4")
        cursor += n
        if item["n_kept"] != n or not np.array_equal(actual, ftos.sl_of_fl):
            raise RuntimeError(f"generator FL->SL membership differs for {key}")
    if cursor != len(rows):
        raise RuntimeError("generator level-row extent mismatch")
    print(f"R4 generator wiring SELF-CHECK PASS: ions={len(data)}, "
          f"levels={len(rows)}, every linked FL->SL exact")


if __name__ == "__main__":
    main()
