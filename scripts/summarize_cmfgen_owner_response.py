#!/usr/bin/env python3
"""생산 owner 응답·채택과 고정상태 참조의 차이를 구분해 요약한다."""
import argparse
import csv
import json
import math
import re
from pathlib import Path

from compare_cmfgen_bf_superlevel_rates import sha256_file


def summarize(root, response, log, output):
    if output.exists():
        raise ValueError("기존 요약 보존")
    text = log.read_text()
    manifest = json.loads((root / "input/input_manifest.json").read_text())
    nd = manifest["depths"]
    if f"CMF_OWNER_RESPONSE_COMPLETE depths={nd}" not in text:
        raise ValueError("생산 풀이 정상 완료 기록 없음")
    rows = list(csv.DictReader(response.open()))
    keys = [(int(r["depth_1based"]),int(r["Z"]),int(r["stage"])) for r in rows]
    expected = {(d,z,c) for d in range(1,nd+1) for z in (14,16) for c in range(2,7)}
    if len(keys)!=len(set(keys)) or set(keys)!=expected:
        raise ValueError("중복/누락 owner 분포 행")
    solves = {key[:2]: row for key,row in zip(keys,rows)}
    per_element = {}
    for Z in (14,16):
        part = [r for (d,z),r in solves.items() if z==Z]
        stages = {}
        for stage in range(2,7):
            part_stage = [r for r in rows if int(r["Z"])==Z and int(r["stage"])==stage
                          and r["candidate_fraction"]]
            if not part_stage:
                continue
            worst = max(part_stage,key=lambda r: abs(float(r["candidate_fraction"])-float(r["pre_fraction"])))
            a,b = float(worst["pre_fraction"]),float(worst["candidate_fraction"])
            stages[str(stage)] = dict(worst_abs_fraction_change=dict(
                depth_1based=int(worst["depth_1based"]),reference_pre_fraction=a,
                candidate_fraction=b,candidate_minus_pre=b-a,
                committed_fraction=float(worst["committed_fraction"]),
                actually_committed=worst["commit_performed"]=="1"))
        per_element[str(Z)] = dict(solves=len(part),
            completed=sum(r["completed"]=="1" for r in part),
            candidate_valid=sum(r["candidate_valid"]=="1" for r in part),
            committed=sum(r["commit_performed"]=="1" for r in part),
            topology_fail=sum(r["topology_pass"]=="0" for r in part),
            numerical_fail=sum(r["numerical_pass"]=="0" for r in part),
            boundary_fail=sum(r["boundary_pass"]=="0" for r in part),
            max_scaled_se_residual=max(float(r["scaled_se_residual"]) for r in part),
            max_conservation_residual=max(float(r["conservation_residual"]) for r in part),
            kappa_range=[min(float(r["kappa_2"]) for r in part),max(float(r["kappa_2"]) for r in part)],
            raw_rank_deficits=sorted({int(r["dimension"])-int(r["raw_rank"]) for r in part}),
            graph_components=sorted({int(r["graph_components"]) for r in part}),
            stages=stages)
    baseline = root / "owner_response_a5.csv"
    same_numeric = None
    if baseline.exists():
        previous = list(csv.DictReader(baseline.open()))
        comparison_columns = ("depth_1based","Z","stage","pre_fraction","candidate_fraction",
                              "committed_fraction","commit_performed","scaled_se_residual",
                              "conservation_residual","kappa_2")
        same_numeric = len(previous)==len(rows) and all(
            all(a[k]==b[k] for k in comparison_columns) for a,b in zip(previous,rows))
    source_root=Path(__file__).resolve().parents[1]
    paths = [response,log,root/"input/owner_input.bin",root/"input/input_manifest.json",
             root/"input/atomic_view_manifest.json",root/"bench_cmfgen_owner_a6"]
    for name in ("bench_cmfgen_owner.c","src/lumina.h","src/lumina_element_wide.c",
                 "src/lumina_plasma.c","src/lumina_atomic.c","src/bf_rate_jnu.c",
                 "src/line_jbar.c","src/radiation_field.c","src/population_contract.c",
                 "scripts/prepare_cmfgen_owner_response.py",str(Path(__file__).relative_to(source_root)),
                 "data/atomic/topion_levels.csv","data/atomic/ionization_reference.csv"):
        paths.append(source_root/name)
    view=root/"input/atomic_view"
    paths.extend(view/name for name in ("abundances.csv","ma_radrecomb_target.bin",
                                        "coldata_cmfgen_manifest.csv"))
    for row in csv.DictReader((view/"coldata_cmfgen_manifest.csv").open()):
        if row["status"]=="OK":
            p=Path(row["out_bin"]);paths.append(p if p.is_absolute() else view/p)
    summary=dict(schema="cmfgen-production-owner-response-v1",
        scope="fixed external CMFGEN J/Te/ne; original SL totals; not a coupled solution",
        cmfgen_run=manifest["cmfgen_run"],epoch_seconds=manifest["epoch_seconds"],
        matched_scrtemp_record=manifest["matched_scrtemp_record"],depths=nd,
        per_element=per_element,
        zero_curve_fix_numeric_rows_equal_to_a5=same_numeric,
        bf_view_counters=re.findall(r"\[A2-05\]\[BF-VIEW\].*",text),
        bb_view_counters=re.findall(r"\[A2-06\]\[BB-VIEW\].*",text),
        CMFGEN_converged=False,coupled_charge_RE_transport_converged=False,
        final_goal="NOT_JUDGED",reference_interpretation="pre-solve nonconverged CMFGEN population, not equilibrium truth",
        input_and_source_sha256={str(p.resolve()):sha256_file(p) for p in paths})
    output.write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:summary[k] for k in ("depths","per_element",
        "zero_curve_fix_numeric_rows_equal_to_a5","bf_view_counters")},indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",type=Path,required=True)
    a=ap.parse_args();r=a.root.resolve()
    summarize(r,r/"owner_response_a6.csv",r/"owner6.log",r/"owner_summary.json")


if __name__=="__main__":main()
