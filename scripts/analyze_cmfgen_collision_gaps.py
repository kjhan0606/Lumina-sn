#!/usr/bin/env python3
"""보존한 생산 행렬에서 비복사 준위쌍 충돌과 표 보간 차이만 분리한다.

새 수렴해나 native 전체 충돌 연산자라고 주장하지 않는다. SAME_N, 선의
fallback 및 8.629/8.630 상수 차이는 이 진단에서 변경하지 않는다.
"""
import argparse
import csv
import ctypes as C
import json
import math
from pathlib import Path
import re
import struct

import numpy as np
from scipy.special import erf

from analyze_cmfgen_owner_matrices import GeneratorDiagnostic, solve_gth
from repair_cmfgen_einstein_view import sha
from summarize_cmfgen_einstein_response import input_population, load_case

KB_EV = 1.380649e-16 / 1.602176634e-12
NU_BB_MIN, NU_BB_MAX = 1.49896229e14, 2.99792458e16


def read_table(path, Z, charge, nlevel):
    """생산 로더가 소비하는 IGC1 표의 양수 Upsilon을 그대로 읽는다."""
    raw = path.read_bytes()
    if len(raw) < 28:
        raise ValueError("충돌표 header 누락")
    magic, version, z, q, nr, nt, nl = struct.unpack_from("<IIiiiii", raw)
    if ((magic, version, z, q, nl) != (0x49474331, 1, Z, charge, nlevel)
            or nr <= 0 or nt < 2 or len(raw) != 28+8*nt+nr*(8+8*nt)):
        raise ValueError("충돌표 정체성/차원 불일치")
    grid = np.frombuffer(raw, dtype="<f8", count=nt, offset=28).copy()
    if not np.isfinite(grid).all() or np.any(grid <= 0) or np.any(np.diff(grid) <= 0):
        raise ValueError("충돌표 온도축 손상")
    pairs = {}
    offset = 28+8*nt
    for _ in range(nr):
        lo, hi = struct.unpack_from("<ii", raw, offset)
        values = np.frombuffer(raw, dtype="<f8", count=nt, offset=offset+8).copy()
        if (not 0 <= lo < hi < nl or (lo, hi) in pairs
                or not np.isfinite(values).all() or np.any(values <= 0)):
            raise ValueError("충돌표 중복/범위/양수 규약 불일치")
        pairs[lo, hi] = values
        offset += 8+8*nt
    return grid, pairs


def thermal_fractions(levels, temperature):
    groups = {}
    for i, row in enumerate(levels):
        groups.setdefault(int(row["super_level"]), []).append(i)
    result = np.zeros(len(levels))
    for members in groups.values():
        energy = np.array([float(levels[i]["energy_eV"]) for i in members])
        g = np.array([float(levels[i]["g"]) for i in members])
        if not np.isfinite(energy).all() or np.any(g <= 0) or temperature <= 0:
            raise ValueError("준위/온도 불일치")
        weights = g*np.exp(-(energy-energy.min())/(KB_EV*temperature))
        result[members] = weights/weights.sum()
    return result


def add_pair(A, lo, hi, omega, levels, fractions, rows, temperature, ne):
    lower, upper = levels[lo], levels[hi]
    i, j = rows[int(lower["super_level"])], rows[int(upper["super_level"])]
    if i == j:
        return False
    de = float(upper["energy_eV"])-float(lower["energy_eV"])
    if de < 0 or not omega > 0:
        raise ValueError("에너지 순서 또는 Upsilon 손상")
    coeff = ne*8.629e-6*omega/math.sqrt(temperature)
    up = coeff/float(lower["g"])*math.exp(-de/(KB_EV*temperature))*fractions[lo]
    down = coeff/float(upper["g"])*fractions[hi]
    A[j, i] += up
    A[i, j] += down
    A[i, i] -= up
    A[j, j] -= down
    return True


def collision_changes(levels, radiative_pairs, tables, omega_set, rows, temperature, ne, n):
    """두 효과를 가산/차분 평면으로 분리하며 생산 표의 동작은 바꾸지 않는다."""
    fractions = thermal_fractions(levels, temperature)
    missing, old_table, log_table = (np.zeros((n, n)) for _ in range(3))
    grid, tabulated = tables
    counts = dict(missing_full_pairs=0, missing_cross_SL_pairs=0, degenerate_missing_pairs=0,
                  table_pairs=len(tabulated))
    for lo in range(len(levels)):
        for hi in range(lo+1, len(levels)):
            if (lo, hi) in radiative_pairs or (lo, hi) in tabulated:
                continue
            counts["missing_full_pairs"] += 1
            if float(levels[hi]["energy_eV"]) == float(levels[lo]["energy_eV"]):
                counts["degenerate_missing_pairs"] += 1
            if add_pair(missing, lo, hi, omega_set, levels, fractions, rows, temperature, ne):
                counts["missing_cross_SL_pairs"] += 1
    for (lo, hi), values in tabulated.items():
        # 생산 C 표 경로는 dE=0 쌍을 소비하지 않는다. 그 차이는 여기서 제외한다.
        if float(levels[hi]["energy_eV"]) <= float(levels[lo]["energy_eV"]):
            continue
        linear = float(np.interp(temperature, grid, values))
        logarithmic = math.exp(float(np.interp(math.log(temperature), np.log(grid), np.log(values))))
        add_pair(old_table, lo, hi, linear, levels, fractions, rows, temperature, ne)
        add_pair(log_table, lo, hi, logarithmic, levels, fractions, rows, temperature, ne)
    return missing, log_table-old_table, counts


def load_deck(deck):
    levels = {}
    for row in csv.DictReader((deck/"levels.csv").open()):
        key = int(row["atomic_number"]), int(row["ion_number"])
        if key[0] in (14, 16):
            levels.setdefault(key, []).append(row)
    for part in levels.values():
        part.sort(key=lambda row: int(row["level_number"]))
        if [int(r["level_number"]) for r in part] != list(range(len(part))):
            raise ValueError("준위 정체성 누락")
    lines = {key: set() for key in levels}
    excluded = []
    for row in csv.DictReader((deck/"line_list.csv").open()):
        key = int(row["atomic_number"]), int(row["ion_number"])
        if key in lines:
            lines[key].add(tuple(sorted((int(row["level_number_lower"]), int(row["level_number_upper"])))))
            if not NU_BB_MIN <= float(row["nu"]) <= NU_BB_MAX:
                excluded.append(row)
    metadata = {(int(r["Z"]), int(r["ion0"])): r for r in
                csv.DictReader((deck/"coldata_cmfgen_manifest.csv").open())}
    return levels, lines, metadata, excluded


def native_field(path, depth):
    with path.open("rb") as stream:
        magic, nd, nn, nc, nb, _ = struct.unpack("<8sIIIId", stream.read(32))
        if magic != b"CMFOWN01" or not 1 <= depth <= nd:
            raise ValueError("복사장 입력 정체성 불일치")
        nu = np.frombuffer(stream.read(8*nn), dtype="<f8").copy()
        stride = 40+8*nc+12*nb+8*nn
        stream.seek(12*nc+(depth-1)*stride+40+8*nc+12*nb, 1)
        J = np.frombuffer(stream.read(8*nn), dtype="<f8").copy()
    if (len(J) != nn or not np.isfinite(J).all() or np.any(J < 0)
            or not np.isfinite(nu).all() or np.any(np.diff(nu) <= 0)):
        raise ValueError("native J/주파수 입력 손상")
    return nu, J


def gaussian_mean(nu, J, center):
    """생산 고정장 adapter와 같은 10km/s, ±4 Gaussian의 선형 J 적분."""
    width = center*1e6/2.99792458e10
    lo, hi = center-4*width, center+4*width
    if lo < nu[0] or hi > nu[-1]:
        return None
    begin, end = np.searchsorted(nu, lo, side="right"), np.searchsorted(nu, hi, side="left")
    knots = np.r_[lo, nu[begin:end], hi]
    values = np.interp(knots, nu, J)
    a, b = knots[:-1], knots[1:]
    x, y = (a-center)/width, (b-center)/width
    slope = np.diff(values)/(b-a)
    intercept = values[:-1]+slope*(center-a)
    norm = math.erf(4.)
    integral = float(np.sum(intercept*(erf(y)-erf(x))/(2*norm)
                    + slope*width*(np.exp(-x*x)-np.exp(-y*y))/(2*math.sqrt(math.pi)*norm)))
    if not math.isfinite(integral) or integral < 0:
        raise ValueError("등록 Gaussian 적분 실패")
    return integral


def excluded_bb_plane(excluded, levels, identity, Z, temperature, nu, J):
    n = len(identity)
    result = np.zeros((n,n))
    mapping = {(int(r["internal_stage"]), int(r["sl_id"])): i for i,r in enumerate(identity)}
    fractions = {q: thermal_fractions(part, temperature) for (z,q),part in levels.items() if z == Z}
    census = dict(outside_lines=0, cross_SL_lines=0, native_field_uncovered=0, exact_zero_J=0)
    for row in excluded:
        if int(row["atomic_number"]) != Z:
            continue
        census["outside_lines"] += 1
        q = int(row["ion_number"])
        lo, hi = int(row["level_number_lower"]), int(row["level_number_upper"])
        part = levels[Z,q]
        i, j = mapping[q,int(part[lo]["super_level"])], mapping[q,int(part[hi]["super_level"])]
        if i == j:
            continue
        census["cross_SL_lines"] += 1
        mean = gaussian_mean(nu, J, float(row["nu"]))
        if mean is None:
            census["native_field_uncovered"] += 1
            continue
        if mean == 0:
            census["exact_zero_J"] += 1
        up = float(row["B_lu"])*mean*fractions[q][lo]
        down = float(row["B_ul"])*mean*fractions[q][hi]
        result[j,i] += up
        result[i,j] += down
        result[i,i] -= up
        result[j,j] -= down
    return result, census


def analyze(root, deck, input_path, library, native_source):
    lib = C.CDLL(str(library.resolve()))
    lib.population_generator_stationary_gth.argtypes = [C.POINTER(C.c_double), C.c_size_t, C.c_double,
        C.POINTER(C.c_double), C.POINTER(GeneratorDiagnostic)]
    lib.population_generator_stationary_gth.restype = C.c_int
    levels, lines, metadata, excluded = load_deck(deck)
    result, inputs = [], {input_path, library, deck/"levels.csv", deck/"line_list.csv",
        deck/"coldata_cmfgen_manifest.csv", native_source/"newsubs/omega_gen_v3.f"}
    for Z, depth in ((14, 28), (16, 35)):
        identity, planes, original, files = load_case(root, depth, Z)
        inputs.update(files.values())
        pre, profile = input_population(input_path, depth, Z, identity)
        temperature, ne = profile[1:3]
        nu, J = native_field(input_path, depth)
        n, total = len(identity), float(pre.sum())
        missing, interpolation = np.zeros((n, n)), np.zeros((n, n))
        census = []
        for charge in range(1, 5):
            key = Z, charge
            source = Path(metadata[key]["col"])
            inputs.add(source)
            text = source.read_text()
            match = re.search(r"(?m)^\s*([\d.]+)\s*!Value for OMEGA if f=0", text)
            if not match:
                raise ValueError("원래 col_data의 OMEGA_SET 불명")
            omega_set = float(match[1])
            rows = {int(r["sl_id"]): i for i, r in enumerate(identity)
                    if int(r["internal_stage"]) == charge}
            if set(rows) != {int(r["super_level"]) for r in levels[key]}:
                raise ValueError("생산 행렬과 준위 SL 매핑 불일치")
            name = metadata[key]["out_bin"]
            if name:
                path = deck/name
                inputs.add(path)
                tables = read_table(path, Z, charge, len(levels[key]))
            else:
                declared = re.search(r"(?m)^\s*(\d+)\s*!Number of transitions", text)
                if (not declared or int(declared[1]) != 0
                        or metadata[key]["n_trans_source"] not in ("", "0")):
                    raise ValueError("실제 충돌표 유실")
                tables = np.array([1., 2.]), {}
            a, b, counts = collision_changes(levels[key], lines[key], tables, omega_set,
                                            rows, temperature, ne, n)
            missing += a
            interpolation += b
            census.append(dict(charge=charge, omega_set=omega_set, **counts))
        A = sum(planes.values())
        bb_add, bb_census = excluded_bb_plane(excluded, levels, identity, Z, temperature, nu, J)
        charge = np.array([int(r["internal_stage"]) for r in identity])
        solutions = {}
        for name, change in (("baseline", np.zeros((n,n))), ("missing_only", missing),
                             ("log_interpolation_only", interpolation), ("both", missing+interpolation),
                             ("excluded_BB_pumping_only", bb_add),
                             ("all_three", missing+interpolation+bb_add)):
            x, diagnostic = solve_gth(lib, A+change, total)
            solutions[name] = dict(fractions={str(q+1): float(x[charge == q].sum()/total)
                                              for q in range(1,6)},
                                   componentwise_residual=diagnostic.residual)
            if name == "baseline" and np.max(np.abs(x-original))/total > 1e-10:
                raise ValueError("원래 생산 후보 재현 실패")
        reference = {str(q+1): float(pre[charge == q].sum()/total) for q in range(1,6)}
        result.append(dict(Z=Z, depth_1based=depth, temperature_K=temperature, ne_cm3=ne,
                           reference=reference, solutions=solutions, census=census,
                           excluded_BB_census=bb_census,
                           excluded_BB_plane_max_s=float(np.max(np.abs(bb_add))),
                           missing_plane_max_s=float(np.max(np.abs(missing))),
                           interpolation_delta_max_s=float(np.max(np.abs(interpolation)))))
    return dict(scope="두 고정입력 지점의 누락 BB 충돌/표 보간/제외 파장 BB 흡수·유도방출 민감도; 생산 변경·전체 native 등가성 아님",
                criteria_changed=False, final_goal="NOT_JUDGED", cases=result,
                evidence_sha256={str(path): sha(path) for path in sorted(inputs)},
                analyzer_sha256=sha(Path(__file__)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "deck", "input", "library", "native-source", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("기존 진단 보존")
    report = analyze(args.root, args.deck, args.input, args.library, args.native_source)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    for case in report["cases"]:
        print(json.dumps(case, ensure_ascii=False, allow_nan=False))
