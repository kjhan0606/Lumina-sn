#!/usr/bin/env python3
"""기존 CMFGEN 출력만 묶어 restart 선택에 쓸 단일 반복 표를 만든다."""
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

from compare_cmfgen_bf_superlevel_rates import sha256_file
from ptf_gated_metrics import ScrtempReader, read_dimensions, read_variables, parse_outgen

NUMBER = r"[+\-\d.EeDd]+"


def parse_controls(path):
    text = path.read_text(errors="strict")
    chunks = re.split(r"Current great iteration count is\s+(\d+)", text)
    result = {}
    for j in range(1, len(chunks), 2):
        iteration, body = int(chunks[j]), chunks[j + 1]
        luminosity = re.search(r"Luminosity of star.*?is:\s*("+NUMBER+r")\s+("+NUMBER+r")", body)
        scale = re.search(r"minimum value of scale for Major species is:\s*("+NUMBER+r")", body)
        def number(value): return float(value.replace("D", "E").replace("d", "e"))
        result[iteration] = dict(
            mode="LAMBDA" if "(LAMBDA)" in body else
                 "BA" if "(BA computed)" in body else "UNRECORDED",
            min_major_scale=number(scale[1]) if scale else None,
            moment_excessive_count=body.count("Error in MOM_J_REL_V9: excessive iteration count"),
            luminosity_d1=number(luminosity[1]) if luminosity else None,
            luminosity_dND=number(luminosity[2]) if luminosity else None,
            moment_before_population_update=(body.find("Error in MOM_J_REL_V9") <
                body.find("Maximum % increase") if "Error in MOM_J_REL_V9" in body else False),
            NG_text_present=bool(re.search(r"\bNG\b", body)))
    return result, dict(BAMATPNT_missing="Error opening BAMATPNT" in text,
                        JEW_missing="Error opening JEW" in text)


def history(run, start, stop, correction):
    nd, nt = read_dimensions(run)
    variables = read_variables(run, nt - 2)
    reader = ScrtempReader(run / "SCRTEMP", nd, nt)
    outgen = parse_outgen(run / "OUTGEN")
    controls, resets = parse_controls(run / "OUTGEN")
    rows = []
    for iteration in range(start, stop + 1):
        old, new = reader.state(iteration-1), reader.state(iteration)
        control = controls.get(iteration)
        if control is None:
            raise ValueError(f"OUTGEN iteration marker missing: {run} {iteration}")
        record = outgen[iteration]
        row = dict(run=run.name, iteration=iteration, **control,
                   raw_increase_percent=record.increase_percent,
                   raw_decrease_percent=record.decrease_percent,
                   returned_MAXCH_percent=record.returned_maxch_percent,
                   max_abs_delta_ln_ne=float(np.max(np.abs(np.log(new[nt-2]/old[nt-2])))))
        if row["luminosity_dND"]:
            row["luminosity_d1_over_dND"] = row["luminosity_d1"]/row["luminosity_dND"]
        weighted_changes = []
        for element in sorted({v.species for v in variables}):
            indices = [v.index-1 for v in variables if v.species == element]
            old_total = old[indices].sum(axis=0)
            if np.any(old_total <= 0):
                raise ValueError("nonpositive element total")
            weighted_changes.append(np.max(np.abs(new[indices]-old[indices])/old_total))
        row["max_population_change_over_element_total"] = float(max(weighted_changes))
        for element in ("Si", "S"):
            all_indices = [v.index-1 for v in variables if v.species == element]
            iii = [v.index-1 for v in variables if v.species == element and v.charge == 2]
            fa = old[iii].sum(axis=0)/old[all_indices].sum(axis=0)
            fb = new[iii].sum(axis=0)/new[all_indices].sum(axis=0)
            row[element+"_III_max_abs_fraction_step"] = float(np.max(np.abs(fb-fa)))
            row[element+"_III_depth27_fraction"] = float(fb[26])
            row[element+"_III_depth27_relative_step_percent"] = float(100*(fb[26]/fa[26]-1)) if fa[26]>0 else None
        gate = correction.get(iteration, {}).get("gate", {}).get("ion_fraction_active_correction", {})
        row["ion_fraction_active_subset_correction_p95_percent"] = gate.get("p95_percent")
        row["ion_fraction_active_subset_correction_max_percent"] = gate.get("max_percent")
        rows.append(row)
    return rows, resets


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modern", type=Path, required=True)
    ap.add_argument("--stage-a", type=Path, required=True)
    ap.add_argument("--stage-a-metrics", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a=ap.parse_args()
    if a.out.exists(): raise ValueError("기존 출력 보존")
    metric=json.loads(a.stage_a_metrics.read_text())
    corrections={r["iteration"]:r for r in metric["rows"]}
    modern,resets_m=history(a.modern,30,40,{})
    stage,resets_a=history(a.stage_a,41,59,corrections)
    rows=modern+stage
    inputs=[a.stage_a_metrics,Path(__file__)]
    for run in (a.modern,a.stage_a):
        inputs += [run/name for name in ("OUTGEN","SCRTEMP","MODEL","POINT1","LEVEL_SL_STEQ_LINKS")]
    summary=dict(scope="existing output retrospective; no simulation", iterations=len(rows),
                 source_resets={a.modern.name:resets_m,a.stage_a.name:resets_a},
                 modern_luminosity_ratio_range=[min(r["luminosity_d1_over_dND"] for r in modern),max(r["luminosity_d1_over_dND"] for r in modern)],
                 stage_a_luminosity_ratio_range=[min(r["luminosity_d1_over_dND"] for r in stage),max(r["luminosity_d1_over_dND"] for r in stage)],
                 phase="OUTGEN radiation pre-solve; state deltas from SCRTEMP post-solve",
                 absent="no saved iteration-by-iteration Jnu reconstructed; missing correction cells remain blank",
                 NG="OUTGEN NG-text flag is not proof of application; use POINT1 LST_NG and source scheduling",
                 acceptance="diagnostic only; no RE/charge/transport/CMFGEN-equivalence PASS",
                 input_sha256={str(p.resolve()):sha256_file(p) for p in inputs})
    a.out.mkdir()
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (a.out/"iteration_history.csv").open("x",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    (a.out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k!="input_sha256"},indent=2))


if __name__=="__main__": main()
