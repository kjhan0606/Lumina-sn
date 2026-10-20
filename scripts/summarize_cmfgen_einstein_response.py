#!/usr/bin/env python3
"""실제 C 전후 행렬에서 B 규약 수리의 범위와 여기·이온화 반응을 대조한다."""
import argparse
import csv
import ctypes as C
import json
from pathlib import Path
import struct

import numpy as np

from repair_cmfgen_einstein_view import sha
from analyze_cmfgen_owner_matrices import GeneratorDiagnostic, solve_gth


def input_population(path, depth, Z, identity):
    with path.open("rb") as stream:
        magic, nd, nn, nc, nb, epoch = struct.unpack("<8sIIIId", stream.read(32))
        if magic != b"CMFOWN01" or not 1 <= depth <= nd:
            raise ValueError("원래 입력 정체성 불일치")
        stream.seek(8 * nn, 1)
        columns = np.frombuffer(stream.read(12 * nc), dtype="<i4").reshape(nc, 3)
        stride = 40 + 8 * nc + 12 * nb + 8 * nn
        stream.seek((depth - 1) * stride, 1)
        profile = struct.unpack("<5d", stream.read(40))
        pop = np.frombuffer(stream.read(8 * nc), dtype="<f8")
    lookup = {tuple(key): float(value) for key, value in zip(columns, pop)}
    return np.array([lookup[(Z, int(row["internal_stage"]), int(row["sl_id"]))]
                     for row in identity]), profile


def load_case(root, depth, Z):
    stem = root / "matrices" / f"lumina_ew_iter{depth:04d}_ce00_z{Z:02d}_s{depth-1:03d}_"
    files = {role: Path(str(stem) + role + ".csv") for role in
             ("identity", "boundary_mass", "matrix_raw", "solution")}
    identity = list(csv.DictReader(files["identity"].open()))
    boundary = next(csv.DictReader(files["boundary_mass"].open()))
    identity.append(dict(matrix_index=str(len(identity)), internal_stage=boundary["terminal_charge"],
                         spectroscopic_stage=str(int(boundary["terminal_charge"]) + 1), sl_id="0"))
    n = len(identity)
    if [int(row["matrix_index"]) for row in identity] != list(range(n)):
        raise ValueError("행렬 행 정체성/순서 불일치")
    planes = {}
    for row in csv.DictReader(files["matrix_raw"].open()):
        if row["channel"] not in planes:
            planes[row["channel"]] = np.zeros((n, n))
        A = planes[row["channel"]]
        A[int(row["row"]), int(row["column"])] += float(row["value"])
    pop = np.empty(n)
    for row in csv.DictReader(files["solution"].open()):
        if row["record_type"] in ("SL", "BOUNDARY_MASS"):
            pop[int(row["matrix_index"])] = float(row["restored_population"])
    return identity, planes, pop, files


def row_flow(A, n, row):
    off = A[row].copy()
    off[row] = 0
    inflow, outflow = float(off @ n), float(-A[row, row] * n[row])
    return dict(inflow=inflow, outflow=outflow, net=inflow - outflow)


def summarize_full_response(before_path, after_path):
    def read(path):
        rows = list(csv.DictReader(path.open()))
        result = {(int(r["depth_1based"]), int(r["Z"]), int(r["stage"])): r for r in rows}
        if len(result) != len(rows):
            raise ValueError("중복 분포 행")
        return result
    old, new = read(before_path), read(after_path)
    expected = {(d, z, c) for d in range(1, 91) for z in (14, 16) for c in range(2, 7)}
    if set(old) != expected or set(new) != expected:
        raise ValueError("전90깊이 응답 누락")
    for key in expected:
        if any(old[key][k] != new[key][k] for k in ("Te_K", "ne_cm3", "n_element", "pre_fraction")):
            raise ValueError("B 수리 전후 고정 입력 변경")
    result = dict(scope="90 depths, equal node weights including trace abundance; candidate not coupled solution",
                  criteria_changed=False, rows=len(new), final_goal="NOT_JUDGED", per_element={})
    for Z in (14, 16):
        part = [new[(d, Z, 3)] for d in range(1, 91)]
        elem = dict(committed=sum(r["commit_performed"] == "1" for r in part),
                    candidate_valid=sum(r["candidate_valid"] == "1" for r in part),
                    kappa_range=[min(float(r["kappa_2"]) for r in part), max(float(r["kappa_2"]) for r in part)],
                    raw_ranks=sorted({int(r["raw_rank"]) for r in part}), stages={})
        for stage in (2, 3):
            keys = [(d, Z, stage) for d in range(1, 91)]
            reference = np.array([float(new[k]["pre_fraction"]) for k in keys])
            candidates = {"before": np.array([float(old[k]["candidate_fraction"]) for k in keys]),
                          "after": np.array([float(new[k]["candidate_fraction"]) for k in keys])}
            metrics = {}
            for label, values in candidates.items():
                delta = np.abs(values - reference)
                worst = int(np.argmax(delta))
                positive = (values > 0) & (reference > 0)
                logs = np.abs(np.log10(values[positive] / reference[positive]))
                metrics[label] = dict(abs_fraction_error_median_p95_max=np.percentile(delta, [50,95,100]).tolist(),
                    abs_log10_ratio_median_p95_max=np.percentile(logs, [50,95,100]).tolist(),
                    positive_ratio_samples=int(np.sum(positive)), worst_depth_1based=worst+1,
                    worst_n_element_cm3=float(new[keys[worst]]["n_element"]))
            elem["stages"][str(stage)] = metrics
        result["per_element"][str(Z)] = elem
    result["evidence_sha256"] = {str(p): sha(p) for p in (before_path, after_path)}
    return result


def compare(root, input_path, Z, depth, library=None):
    identity, old, x0, files0 = load_case(root / "baseline", depth, Z)
    identity1, new, x1, files1 = load_case(root / "corrected", depth, Z)
    fields = ("matrix_index", "internal_stage", "sl_id")
    if [[r[k] for k in fields] for r in identity] != [[r[k] for k in fields] for r in identity1]:
        raise ValueError("수리 전후 준위 정체성 변경")
    pre, profile = input_population(input_path, depth, Z, identity)
    if set(old) != set(new):
        raise ValueError("수리 전후 과정 집합 변경")
    channel_delta = {c: dict(byte_equal=bool(np.array_equal(old[c], new[c])),
                            max_abs_delta=float(np.max(np.abs(new[c] - old[c])))) for c in old}
    if any(not d["byte_equal"] for c, d in channel_delta.items() if c != "rad_bb"):
        raise ValueError("B 수리에 무관한 과정이 변경됨")
    charge = np.array([int(r["internal_stage"]) for r in identity])
    total = float(pre.sum())
    fractions = {name: {str(c + 1): float(x[charge == c].sum() / total) for c in range(1, 6)}
                 for name, x in (("cmfgen_pre", pre), ("old_candidate", x0), ("new_candidate", x1))}
    target = np.flatnonzero(charge == 2)
    gamma = np.sum(old["rad_bf"][np.ix_(charge == 3, charge == 2)], axis=0)
    weights = {name: x[target] / x[target].sum() for name, x in
               (("cmfgen_pre", pre), ("old_candidate", x0), ("new_candidate", x1))}
    mean_gamma = {name: float(w @ gamma) for name, w in weights.items()}
    rank = np.argsort(weights["cmfgen_pre"] * gamma)[::-1][:5]
    rows = []
    for k in rank:
        i = target[k]
        rows.append(dict(sl_0based=int(identity[i]["sl_id"]), gamma_s=float(gamma[k]),
                         within_ion_fraction={name: float(w[k]) for name, w in weights.items()},
                         mean_gamma_contribution={name: float(w[k] * gamma[k]) for name, w in weights.items()},
                         common_pre_flow_by_channel={c: dict(before=row_flow(old[c], pre, i),
                                                             after=row_flow(new[c], pre, i)) for c in old}))
    result = dict(Z=Z, depth_1based=depth, Te_K=profile[1], ne_cm3=profile[2], fractions=fractions,
                channel_delta=channel_delta, mean_III_photo_coefficient_s=mean_gamma,
                largest_pre_photo_contributors=rows,
                evidence_sha256={str(p): sha(p) for p in list(files0.values()) + list(files1.values())})
    if library is not None:
        # 원래 raw generator의 명명된 과정 평면을 빼는 독립 진단이다.
        # 생산 commit 또는 수용 기준 우회로 사용하지 않는다.
        A = sum(new.values()) - new["autoion_DR"]
        without_extra, diagnostic = solve_gth(library, A, total)
        result["without_external_DR_diagnostic"] = dict(
            scope="offline C GTH on corrected raw matrix minus external fit plane; NOT production",
            fractions={str(c+1): float(without_extra[charge == c].sum()/total) for c in range(1,6)},
            generator_residual=diagnostic.residual,
            III_mean_photo_coefficient_s=float(without_extra[target] @ gamma / without_extra[target].sum()),
            common_pre_cross_stage_flux={c: dict(
                forward_III_to_IV=float(np.sum(Ac[np.ix_(charge == 3, charge == 2)] @ pre[target])),
                reverse_IV_to_III=float(np.sum(Ac[np.ix_(charge == 2, charge == 3)] @ pre[charge == 3])))
                for c, Ac in new.items()})
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--library", type=Path)
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    out = a.output or a.root / "einstein_response_summary.json"
    if out.exists():
        raise ValueError("기존 요약 보존")
    library = None
    if a.library:
        library = C.CDLL(str(a.library.resolve()))
        library.population_generator_stationary_gth.argtypes = [C.POINTER(C.c_double), C.c_size_t,
            C.c_double, C.POINTER(C.c_double), C.POINTER(GeneratorDiagnostic)]
        library.population_generator_stationary_gth.restype = C.c_int
    cases = [compare(a.root, a.input, z, d, library) for z, d in ((14, 61), (16, 27), (16, 53))]
    result = dict(scope="actual C matrices, only B columns repaired; fixed nonconverged reference",
                  cases=cases, input_sha256=sha(a.input), script_sha256=sha(Path(__file__)),
                  gth_library_sha256=sha(a.library) if a.library else None,
                  final_goal="NOT_JUDGED")
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for case in cases:
        print(json.dumps({k: v for k, v in case.items() if k not in
                          ("largest_pre_photo_contributors", "evidence_sha256")}))


if __name__ == "__main__":
    main()
