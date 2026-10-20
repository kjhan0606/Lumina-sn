#!/usr/bin/env python3
"""선택물리 수리의 실제 C 실행을 보존한 독립 진단과 대조한다."""
import argparse
import csv
import json
import hashlib
import math
from pathlib import Path
import re

import numpy as np

from repair_cmfgen_einstein_view import sha
from summarize_cmfgen_einstein_response import load_case, input_population


def parse_native_se_tail(text, models, nd, cases):
    marker = text.rfind("\n STEQ ARRAY\n")
    if marker < 0:
        raise ValueError("native 전체 SE 출력 없음")
    groups, current = [], []
    for line in text[marker:].splitlines()[2:]:
        match = re.match(r"^\s*(\d+)\s*(\*\*|\*)\s+(.+)$", line)
        if match:
            if match[2] == "**" and current:
                groups.append(current)
                current = []
            current.append((int(match[1]), match[3].split()))
        elif current and line.strip():
            groups.append(current)
            current = []
            break
    if current:
        groups.append(current)
    keys = list(models)
    if len(groups) != ((nd+9)//10)*len(keys):
        raise ValueError("native SE 이온/깊이 block 누락")
    results = []
    for Z, stage, depth in cases:
        ion = keys.index((Z, stage))
        group = groups[((depth-1)//10)*len(keys)+ion]
        nsl = int(models[(Z,stage)]["n_super"])
        if (len(group) not in (nsl+2, nsl+3) or
                [row[0] for row in group] != list(range(1,len(group)+1))):
            raise ValueError("native SE bound-level 행 정체성 불일치")
        strings = [values[(depth-1)%10] for _, values in group[:nsl]]
        values = [float(s.replace("D", "E")) for s in strings]
        if not np.isfinite(values).all():
            raise ValueError("native SE 비유한 출력")
        rounding = math.fsum(.5*10**(int(re.search(r"[ED]([+-]?\d+)$",s)[1])-4)
                            for s,v in zip(strings,values) if v != 0)
        results.append(dict(Z=Z, stage=stage, model_label=models[(Z,stage)]["label"],
            depth_1based=depth, bound_SL_rows=nsl, printed_rows=len(group),
            bound_SE_sum=math.fsum(values), bound_SE_sum_abs=math.fsum(abs(v) for v in values),
            decimal_print_rounding_sum_bound=rounding))
    return results


def native_se_snapshot(run, cases):
    from expand_atomic_data_cmfgen import load_cmfgen_model_spec
    from ptf_gated_metrics import read_dimensions
    path = run / "STEQ_VALS"
    size = path.stat().st_size
    offset = max(0,size-24*1024*1024)
    with path.open("rb") as stream:
        stream.seek(offset)
        raw = stream.read()
    models = load_cmfgen_model_spec(run / "MODEL_SPEC", run / "atomic_links.txt")
    nd, _ = read_dimensions(run)
    iteration = int((run / "POINT1").read_text().splitlines()[1].split()[0])
    return dict(scope="last native pre-SOLVEBA bound-level STEQ, not a complete process decomposition",
        last_native_iteration=iteration, expected_pre_solve_scrtemp_record=iteration-1,
        cases=parse_native_se_tail(raw.decode("ascii"),models,nd,cases),
        source=dict(path=str(path), bytes=size, tail_offset=offset,
                    tail_sha256=hashlib.sha256(raw).hexdigest(),
                    POINT1_sha256=sha(run / "POINT1")), final_goal="NOT_JUDGED")


def reference_balance(matrix_root, input_path, rates_path, cases):
    """비수렴 참조의 부분 BF 불균형과 실제 C의 같은 인구 유량을 구분한다."""
    reference = {}
    for row in csv.DictReader(rates_path.open()):
        key = (row["ion"], int(row["depth_1based"]))
        if key not in reference:
            reference[key] = [0., 0., 0]
        reference[key][0] += float(row["cmfgen_photo_rate_density_cm-3_s-1"])
        reference[key][1] += float(row["cmfgen_recombination_rate_density_cm-3_s-1"])
        reference[key][2] += 1
    result = []
    for Z, depth in cases:
        identity, planes, candidate, files = load_case(matrix_root, depth, Z)
        pre, profile = input_population(input_path, depth, Z, identity)
        charge = np.array([int(r["internal_stage"]) for r in identity])
        lower, upper = charge == 2, charge == 3
        flows = {}
        for name, A in planes.items():
            flows[name] = {}
            for label, pop in (("common_pre", pre), ("candidate", candidate)):
                forward = float(np.sum(A[np.ix_(upper, lower)] @ pop[lower]))
                reverse = float(np.sum(A[np.ix_(lower, upper)] @ pop[upper]))
                flows[name][label] = dict(forward=forward, reverse=reverse, net=forward-reverse)
        name = "Si III" if Z == 14 else "S III"
        photo, recomb, count = reference[(name, depth)]
        if count != int(np.sum(lower)):
            raise ValueError("PRRR/행렬 III superlevel 범위 불일치")
        pair_ratio = recomb/photo * float(pre[lower].sum()/pre[upper].sum())
        cflow = flows["rad_bf"]["common_pre"]
        result.append(dict(Z=Z, depth_1based=depth,
            native_PRRR=dict(photo=photo, recombination=recomb, reverse_over_forward=recomb/photo),
            C_common_pre_photo_over_native=cflow["forward"]/photo,
            C_common_pre_recombination_over_native=cflow["reverse"]/recomb,
            native_BF_only_pair_III_fraction_estimate=pair_ratio/(1+pair_ratio),
            C_candidate_pair_III_fraction=float(candidate[lower].sum()/(candidate[lower].sum()+candidate[upper].sum())),
            reference_pair_III_fraction=float(pre[lower].sum()/(pre[lower].sum()+pre[upper].sum())),
            C_cross_stage_flux=flows,
            evidence_sha256={str(p):sha(p) for p in files.values()}))
    return dict(scope="partial III/IV BF balance, not native full SE or a convergence verdict", cases=result,
                input_sha256={str(p):sha(p) for p in (input_path, rates_path)}, final_goal="NOT_JUDGED")


def summarize(root, previous):
    out = root / "policy_response_summary.json"
    if out.exists():
        raise ValueError("기존 요약 보존")
    diagnostics = json.loads((previous / "external_DR_diagnostic.json").read_text())
    before_path = root / "baseline/selected_response.csv"
    after_path = root / "policy/selected_response.csv"
    old_baseline_path = previous / "corrected/selected_response.csv"
    def read(path):
        rows = list(csv.DictReader(path.open()))
        data = {(int(r["Z"]), int(r["depth_1based"]), int(r["stage"])): r for r in rows}
        expected = {(z, d, c) for z, d in ((14, 61), (16, 27), (16, 53)) for c in range(2, 7)}
        if len(data) != len(rows) or set(data) != expected:
            raise ValueError("3점 비교 행 누락/중복")
        return data
    before, after, original = read(before_path), read(after_path), read(old_baseline_path)
    for key in before:
        if before[key]["candidate_fraction"] != original[key]["candidate_fraction"]:
            raise ValueError("신규 바이너리의 구형 정책 결과 변경")
        for col in ("Te_K", "ne_cm3", "n_element", "pre_fraction"):
            if before[key][col] != after[key][col]:
                raise ValueError("선택물리 수리 이외 입력 변경")
    cases = []
    for case in diagnostics["cases"]:
        Z, depth = case["Z"], case["depth_1based"]
        identity, old_planes, old_pop, files0 = load_case(root / "baseline", depth, Z)
        _, new_planes, new_pop, files1 = load_case(root / "policy", depth, Z)
        zero = np.zeros((len(identity), len(identity)))
        channels = {}
        for channel in old_planes.keys() | new_planes.keys():
            a, b = old_planes.get(channel, zero), new_planes.get(channel, zero)
            equal = bool(np.array_equal(a, b))
            if channel != "autoion_DR" and not equal:
                raise ValueError("외부 fit 이외 과정 변경: " + channel)
            if channel == "autoion_DR" and np.any(b != 0):
                raise ValueError("정책에서 금지한 외부 fit이 남음")
            channels[channel] = dict(byte_equal=equal, after_nonzero=int(np.count_nonzero(b)))
        fractions = {str(c): float(after[(Z, depth, c)]["candidate_fraction"]) for c in range(2, 7)}
        expected = case["without_external_DR_diagnostic"]["fractions"]
        delta = max(abs(fractions[k] - expected[k]) for k in fractions)
        if delta > 1e-10:
            raise ValueError("생산 후보가 사전 독립 진단을 재현하지 못함")
        cases.append(dict(Z=Z, depth_1based=depth, III_reference=float(after[(Z,depth,3)]["pre_fraction"]),
            III_production=fractions["3"], III_B_only=float(before[(Z,depth,3)]["candidate_fraction"]),
            max_fraction_delta_vs_offline=delta, channels=channels,
            commit_performed=int(after[(Z,depth,3)]["commit_performed"]),
            evidence_sha256={str(p): sha(p) for p in list(files0.values())+list(files1.values())}))
    result = dict(scope="actual production C policy vs independent fixed-input diagnostic; not coupled convergence",
        baseline_reproduced_exactly=True, cases=cases, final_goal="NOT_JUDGED",
        input_sha256={str(p): sha(p) for p in (before_path, after_path,
            previous / "external_DR_diagnostic.json", root / "atomic_view/cmfgen_bf_channels_manifest.json")},
        script_sha256=sha(Path(__file__)))
    out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    for case in cases:
        print(json.dumps({k:v for k,v in case.items() if k != "evidence_sha256"}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--previous", type=Path, required=True)
    args = ap.parse_args()
    summarize(args.root, args.previous)
