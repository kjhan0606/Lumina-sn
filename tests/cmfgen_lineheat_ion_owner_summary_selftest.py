#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from cmfgen_ion_identity import parse_cmfgen_ion_id  # noqa: E402
from summarize_cmfgen_lineheat_ion_owners import summarize  # noqa: E402


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "LINEHEAT"
        path.write_text(
            "1 CoIV(a-b) 10.0 1 2 1.0\n1 -2 3\n"
            "2 CoIV(c-d) 11.0 2 3 1.0\n-.5 1 0\n"
            "3 NkIII(e-f) 12.0 3 4 1.0\n.25 .5 -.25\n"
            "4 FeSIX(g-h) 13.0 4 5 1.0\n0 0 0\n"
            "5 CoSIX(i-j) 14.0 5 6 1.0\n0 0 0\n"
            "6 NkSIX(k-l) 15.0 6 7 1.0\n0 0 0\n"
            "7 SkIV(m-n) 16.0 7 8 1.0\n0 0 0\n"
        )
        report = summarize(path, 3, [1, 2], None, None)
    if report["line_records"] != 7 or report["verdict"] != \
            "COMPLETE_DIAGNOSTIC_OWNER_DECOMPOSITION":
        raise SystemExit("summary header mismatch")
    depth1 = report["depths"]["1"]
    if depth1["line_order_signed_internal"] != 0.75 or \
            depth1["line_order_absolute_internal"] != 1.75 or \
            depth1["signed_grouping_delta_internal"] != 0.0:
        raise SystemExit("line-order closure mismatch")
    rows = {row["cmfgen_label"]: row
            for row in depth1["top_by_abs_signed_ion_total"]}
    if rows["CoIV"]["normalized_species"] != "Co IV" or \
            rows["CoIV"]["signed_internal"] != 0.5 or \
            rows["NkIII"]["normalized_species"] != "Ni III" or \
            rows["NkIII"]["signed_internal"] != 0.25 or \
            rows["FeSIX"]["normalized_species"] != "Fe VI" or \
            rows["CoSIX"]["normalized_species"] != "Co VI" or \
            rows["NkSIX"]["normalized_species"] != "Ni VI" or \
            rows["SkIV"]["normalized_species"] != "Si IV":
        raise SystemExit("owner identity or aggregation mismatch")
    expected = {
        "FeSIX": (26, 6, "Fe VI"),
        "CoSIX": (27, 6, "Co VI"),
        "NkSIX": (28, 6, "Ni VI"),
        "SkIV": (14, 4, "Si IV"),
        "SSIX": (16, 6, "S VI"),
        "SIX": (16, 9, "S IX"),
    }
    for label, identity in expected.items():
        if parse_cmfgen_ion_id(label) != identity:
            raise SystemExit(f"CMFGEN ion identity mismatch: {label}")
    for invalid in ("", "FeII", "FeSIXjunk", "XxIII", "FeSIX "):
        try:
            parse_cmfgen_ion_id(invalid)
        except ValueError:
            pass
        else:
            raise SystemExit(f"invalid CMFGEN ion id accepted: {invalid!r}")
    print("PASS cmfgen_lineheat_ion_owner_summary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
