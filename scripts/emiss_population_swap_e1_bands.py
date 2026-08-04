#!/usr/bin/env python3
"""Make the preregistered E1 600--3000 A table from stage31 outputs."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import stage31_cmf_field_bench as bench  # noqa: E402
from cmf_chieta_check import check_artifact  # noqa: E402

RUN = Path("/gpfs/kjhan/lumina_runner2/scratch/chieta_capture_parity59_188605")
TABLES = {
    "A_authority": ROOT / "docs/s31_results/stage31_jdet_s8_round7d.tsv",
    "A_replay": ROOT / "validation/emiss_e1/jdet_A_replay.tsv",
    "B": ROOT / "validation/emiss_e1/jdet_B.tsv",
}


def main() -> None:
    edges, _, _ = bench.canonical_grid()
    context, _ = bench.load_gamma_context(RUN, edges, None)
    cmf = context["cmf"]["J"]
    arrays = check_artifact(RUN / "chieta_iter10").arrays
    j_mc = np.asarray(arrays[8]).reshape(50, 1000)[8][::-1]
    result = {}
    for lane, path in TABLES.items():
        _, table = bench.parse_driver_table(path)
        result[lane] = bench.make_band_rows(
            edges, table["J_det"][::-1], j_mc, cmf)
    out = ROOT / "validation/emiss_e1/band_ratios.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    for i, row in enumerate(result["A_authority"]):
        a = row["J_det_over_J_CMFGEN"]
        ar = result["A_replay"][i]["J_det_over_J_CMFGEN"]
        b = result["B"][i]["J_det_over_J_CMFGEN"]
        print(f"{row['band']:4s} A={a:.8g} A_replay={ar:.8g} B={b:.8g} "
              f"B/A={b/a:.8g} excess_removed={(a-b)/(a-1):.8g}")


if __name__ == "__main__":
    main()
