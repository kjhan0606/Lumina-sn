#!/usr/bin/env python3
"""봉인 광선·물질의 공간 분할을 고정하고 내부 주파수 이동만 세분한다.

각 주파수의 최종 formal source를 쓴다. 첫 청색 구간은 원 간격을
유지하며 이후 구간의 χ/η와 core 경계를 logν에서 선형 보간한다.
새 SE/RE 또는 자기일관 산란 해를 계산하지 않는다.
"""
import argparse
from collections import deque
from itertools import groupby
import json
from pathlib import Path
import resource
import time

import numpy as np
from numba import njit

from analyze_cmfgen_formal_capture import geometry
from analyze_cmfgen_siii_compact_capture import blue_state, records
from cmfgen_siii_full_path_kernel import refine_nodes, sweep
from compare_cmfgen_siii_all_plus import production_weight
from prepare_cmfgen_mom_capture import sha
from prepare_cmfgen_siii_compact_capture import FIRST, LAST


@njit(cache=True)
def advance(previous_chi, previous_eta, chi, eta, s_m, s_p, b_m, b_p,
            prev_m, prev_p, dlognu, previous_boundary, boundary,
            core_ray, subdivisions, frequency_factor):
    """양 끝점을 보존하고 χ 하한 없이 고정 계수 사이를 보간한다."""
    minus, plus = prev_m, prev_p
    for step in range(1, frequency_factor+1):
        if step == frequency_factor:
            current_chi, current_eta, inner = chi, eta, boundary
        else:
            w = step/frequency_factor
            current_chi = (1.-w)*previous_chi+w*chi
            current_eta = (1.-w)*previous_eta+w*eta
            inner = (1.-w)*previous_boundary+w*boundary
        plus, minus = sweep(current_chi, current_eta, s_m, s_p, b_m, b_p,
                            minus, plus, dlognu/frequency_factor, inner,
                            core_ray, subdivisions)
    return plus, minus


def run(capture, out, transport_root, baseline_rates, subdivisions=64,
        frequency_factor=1, reference_result=None):
    started = time.monotonic()
    source_paths = (Path(__file__),
                    Path(__file__).with_name('cmfgen_siii_full_path_kernel.py'))
    source_sha256_at_start = {str(path): sha(path) for path in source_paths}
    delta_path = out.with_suffix('.npy')
    if out.exists() or delta_path.exists():
        raise FileExistsError('기존 공간·주파수 비교 결과 보존')
    if (type(subdivisions) is not int or not 2 <= subdivisions <= 64 or
            type(frequency_factor) is not int or frequency_factor not in (1, 2, 4)):
        raise ValueError('공간2–64·주파수1/2/4 분할 필요')
    field = capture/'field'
    candidates = list(field.glob(f'FORMAL_{FIRST}_*_geometry.bin'))
    if len(candidates) != 1:
        raise ValueError('첫 주파수 기하가 유일하지 않음')
    g = geometry(candidates[0])
    _, blue = blue_state(field/'siii_blue_in.bin', g, include_rays=True)
    nrays = g['NP']-49
    if nrays != 56 or len(blue) < nrays or g['RAY_POINTS_INSERTED']:
        raise ValueError('봉인 native49 광선·기하 불일치')
    geo = {pieces: [tuple(refine_nodes(ray[name], pieces)
                         for name in ('S_M', 'S_P', 'B_M', 'B_P'))
                    for ray in g['rays'][:nrays]]
           for pieces in (1, subdivisions)}
    state = {pieces: [dict(I_M=refine_nodes(ray['I_M'], pieces),
                          I_P=refine_nodes(ray['I_P'], pieces))
                      for ray in blue[:nrays]]
             for pieces in (1, subdivisions)}
    prior = None
    freq, original_j, fine_j = [], [], []
    maximum = 0.
    calls = 0
    for index, group in groupby(records(field/'siii_compact.bin', g, include_rays=True),
                                key=lambda row: row['index']):
        # 같은 주파수의 중간 반복은 청색 상태를 갱신하지 않는다.
        last = deque(maxlen=1)
        for row in group:
            last.append(row)
            calls += 1
        row = last[0]
        if index != FIRST+len(freq) or row['np_limit'] < nrays:
            raise ValueError('주파수 또는 목표 광선 누락')
        if prior is not None and not np.isclose(
                np.log(prior['freq']/row['freq']), row['dlognu'],
                rtol=1e-8, atol=1e-12):
            raise ValueError('원 비등간격 주파수 폭 불일치')
        moments = {}
        for pieces in (1, subdivisions):
            outputs = []
            total_j = 0.
            factor = frequency_factor if pieces != 1 and prior is not None else 1
            for r, ray in enumerate(g['rays'][:nrays]):
                n = len(ray['R'])
                prev = state[pieces][r]
                boundary = row['boundary'][r][2]
                old = prior if prior is not None else row
                plus, minus = advance(old['chi_ext'][:n], old['eta_ext'][:n],
                    row['chi_ext'][:n], row['eta_ext'][:n], *geo[pieces][r],
                    prev['I_M'], prev['I_P'], row['dlognu'],
                    old['boundary'][r][2], boundary, r < g['NC'], pieces, factor)
                if (not np.isfinite(plus).all() or not np.isfinite(minus).all() or
                        np.min(plus) < 0 or np.min(minus) < 0):
                    raise ValueError('주파수 세분 강도 비유한/음수')
                outputs.append(dict(I_M=minus, I_P=plus))
                link = (int(ray['LNK'][49])-1)*pieces
                total_j += (plus[link]*g['weights']['JQW_P'][49,r] +
                            minus[link]*g['weights']['JQW_M'][49,r])
                if pieces == 1:
                    pairs = [(plus[0], row['boundary'][r][0]),
                             (minus[0], row['boundary'][r][1]),
                             (plus[-1], boundary),
                             (minus[-1], row['boundary'][r][3])]
                    for offset in range(4):
                        if r < len(row['points'][offset]):
                            k = int(ray['LNK'][48+offset])-1
                            observed = row['points'][offset][r]
                            pairs.extend(((plus[k], observed[0]), (minus[k], observed[1]),
                                          (prev['I_P'][k], observed[2]),
                                          (prev['I_M'][k], observed[3])))
                    for a, b in pairs:
                        maximum = max(maximum, abs(a-b)/max(abs(a), abs(b), 1e-30))
            state[pieces] = outputs
            moments[pieces] = total_j
        if (maximum > 1e-10 or abs(moments[1]-row['native'][1]) >
                1e-10*max(abs(moments[1]), abs(row['native'][1]), 1e-30)):
            raise ValueError('원 최종 호출 강도·J 재현 실패')
        freq.append(row['freq']); original_j.append(row['native'][1])
        fine_j.append(moments[subdivisions])
        prior = row
    if len(freq) != LAST-FIRST+1 or calls != 15550:
        raise ValueError('원 대역 최종 source coverage 불일치')
    freq = np.asarray(freq[::-1])*1e15
    original_j = np.asarray(original_j[::-1])
    fine_j = np.asarray(fine_j[::-1])
    delta = fine_j-original_j
    agreement = None
    if reference_result is not None:
        reference = json.loads(reference_result.read_text())
        previous_delta = np.load(reference_result.with_suffix('.npy'), allow_pickle=False)
        if (frequency_factor != 1 or reference['subdivisions'] != subdivisions or
                sha(reference_result.with_suffix('.npy')) != reference['delta_j_sha256'] or
                previous_delta.shape != delta.shape):
            raise ValueError('기존 전체 경로 비교 신원 불일치')
        agreement = float(np.max(abs(delta-previous_delta)/np.maximum(
            np.maximum(abs(fine_j), abs(original_j)), 1e-30)))
        if agreement > 1e-10:
            raise ValueError('주파수1분할이 기존 전체 경로 결과와 불일치')
    weight, f_rate, j_rate, original, f_path = production_weight(
        freq, transport_root, baseline_rates, original_j)
    delta_rate = float(np.sum(weight*delta))
    np.save(delta_path, delta, allow_pickle=False)
    report = dict(status='FIXED_SOURCE_FREQUENCY_REFINEMENT_CONDITIONAL',
        subdivisions=subdivisions, frequency_factor=frequency_factor,
        frequencies=len(freq), source_calls_read=calls, rays=nrays,
        baseline_max_scaled_error=maximum, reference_result_max_scaled_error=agreement,
        first_blue_interval='원 간격 유지; 포획 앞 source를 외삽하지 않음',
        coefficient_interpolation='log(nu) 선형 χ/η·core 경계, signed χ, floor/clip 없음',
        photo_quadrature='기존 5196개 주파수와 동일 PHOT/Milne 가중치',
        original_F_photo_cm3_s=f_rate, original_J_photo_cm3_s=j_rate,
        refined_J_photo_cm3_s=j_rate+delta_rate,
        refined_F_minus_J_photo_cm3_s=f_rate-j_rate-delta_rate,
        delta_J_photo_cm3_s=delta_rate, delta_j_sha256=sha(delta_path),
        elapsed_seconds=time.monotonic()-started, new_SE_RE_updates=0,
        self_consistent_scattering=False, final_goal='NOT_MET',
        source_hash_timing='run 시작 시 source 해시; 종료 시 파일 변화는 별도 기록',
        source_files_changed_since_start=[str(path) for path in source_paths
            if sha(path) != source_sha256_at_start[str(path)]],
        file_sha256={**source_sha256_at_start, **{str(p): sha(p) for p in
            (candidates[0], field/'siii_blue_in.bin', field/'siii_compact.bin',
             original, f_path, baseline_rates)}})
    with out.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--transport-root', type=Path, required=True)
    parser.add_argument('--baseline-rates', type=Path, required=True)
    parser.add_argument('--subdivisions', type=int, default=64)
    parser.add_argument('--frequency-factor', type=int, choices=(1, 2, 4), default=1)
    parser.add_argument('--reference-result', type=Path)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (4*1024**3, 4*1024**3))
    run(args.capture.resolve(), args.out.resolve(), args.transport_root.resolve(),
        args.baseline_rates.resolve(), args.subdivisions, args.frequency_factor,
        args.reference_result.resolve() if args.reference_result else None)
