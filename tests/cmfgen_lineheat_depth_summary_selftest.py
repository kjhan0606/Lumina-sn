#!/usr/bin/env python3
"""Known-answer test for summarize_cmfgen_lineheat_depths.py."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
SCRIPT = SCRIPT_DIR / "summarize_cmfgen_lineheat_depths.py"
SPEC = importlib.util.spec_from_file_location("lineheat_depth_summary", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="lumina-cmfgen-lineheat-") as tmp:
        source = Path(tmp) / "LINEHEAT"
        source.write_text(
            "1 X 1.0 1 2 1.0\n 1.0 2.0 -3.0\n"
            "2 Y 2.0 3 4 1.0\n -1.0 4.0 5.0\n",
            encoding="ascii",
        )
        state = Path(tmp) / "RVTJ"
        state.write_text(
            "Header\n"
            " Velocity (km/s)\n 3000 2000 1000\n"
            " Temperature (10^4K)\n 1.5 2.0 2.5\n"
            " Electron density\n 30 20 10\n"
            " Atom Density\n 15 10 5\n"
            " Ion Density\n 15 10 5\n"
            " Mass Density (gm/cm^3)\n 3e-13 2e-13 1e-13\n",
            encoding="ascii",
        )
        report = MODULE.summarize(source, 3, [1, 3], state)
        assert report["line_records"] == 2
        depth1 = report["depths"]["1"]
        assert depth1["signed_internal"] == 0.0
        assert depth1["absolute_internal"] == 2.0
        assert depth1["cancellation_condition"] == "infinite"
        assert depth1["velocity_km_s"] == 3000.0
        assert depth1["temperature_K"] == 15000.0
        assert depth1["electron_density_cm3"] == 30.0
        assert depth1["atom_density_cm3"] == 15.0
        assert depth1["ion_density_cm3"] == 15.0
        assert depth1["mass_density_g_cm3"] == 3.0e-13
        depth3 = report["depths"]["3"]
        assert depth3["signed_internal"] == 2.0
        assert depth3["absolute_internal"] == 8.0
        assert depth3["positive_count"] == 1
        assert depth3["negative_count"] == 1
        assert depth3["velocity_km_s"] == 1000.0
        assert depth3["temperature_K"] == 25000.0
        assert depth3["electron_density_cm3"] == 10.0
        assert depth3["mass_density_g_cm3"] == 1.0e-13
        assert report["state_source"]["format"] == "CMFGEN_RVTJ"
        assert len(report["state_source"]["sha256"]) == 64
        assert report["repair"] == 0
    print("cmfgen_lineheat_depth_summary_selftest: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
