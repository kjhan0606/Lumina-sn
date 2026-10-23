#!/usr/bin/env python3
"""봉인 CMFGEN 광선의 원 공간경로와 세분 공간경로를 같은 χ/η에서 대조한다.

각 원 formal 호출의 계수·core 경계를 고정한다. 다른 주파수로의
광선 상태 전달은 계산된 전체 광선 출력을 사용한다. 새 SE/RE는 없다.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze_cmfgen_formal_capture import geometry
from analyze_cmfgen_siii_compact_capture import blue_state, records
from cmfgen_siii_full_path_kernel import refine_nodes, sweep
from compare_cmfgen_siii_all_plus import production_weight
from prepare_cmfgen_mom_capture import sha
from prepare_cmfgen_siii_compact_capture import FIRST, LAST


def run(capture, out, subdivisions=8, max_calls=6,
        transport_root=None, baseline_rates=None):
    full_band = max_calls is None
    delta_path = out.with_suffix('.npy')
    if out.exists() or (full_band and delta_path.exists()):
        raise FileExistsError('기존 전체 공간경로 결과 보존')
    if (type(subdivisions) is not int or not 2 <= subdivisions <= 128 or
            (not full_band and
             (type(max_calls) is not int or not 1 <= max_calls <= 12)) or
            (full_band and (transport_root is None or baseline_rates is None))):
        raise ValueError('제한된 광선 경로 세분·호출 횟수 필요')
    field = capture/'field'
    candidates = list(field.glob(f'FORMAL_{FIRST}_*_geometry.bin'))
    if len(candidates) != 1:
        raise ValueError('첫 주파수 봉인 기하가 유일하지 않음')
    g = geometry(candidates[0])
    initial_jit, blue = blue_state(field/'siii_blue_in.bin',g,include_rays=True)
    nrays = g['NP']-49
    if nrays != 56 or len(blue) < nrays or g['RAY_POINTS_INSERTED']:
        raise ValueError('native49 광선·청색 상태·기하 불일치')
    geo = {}
    state = {}
    for pieces in (1,subdivisions):
        geo[pieces] = [tuple(refine_nodes(g['rays'][r][name],pieces)
                             for name in ('S_M','S_P','B_M','B_P'))
                       for r in range(nrays)]
        state[pieces] = [dict(I_M=refine_nodes(blue[r]['I_M'],pieces),
                              I_P=refine_nodes(blue[r]['I_P'],pieces))
                         for r in range(nrays)]
    previous_index = None
    last = None
    summary = []
    final_entries = []
    last_entry = None
    baseline_max_scaled = 0.
    calls = 0
    for row in records(field/'siii_compact.bin',g,include_rays=True):
        if not full_band and calls >= max_calls:
            break
        if previous_index is None:
            if (row['index'],row['jit']) != (FIRST,initial_jit):
                raise ValueError('첫 청색 상태 신원 불일치')
        elif row['index'] != previous_index:
            if row['index'] != previous_index+1 or not row['new_freq']:
                raise ValueError('청색→적색 주파수 연쇄 누락')
            final_entries.append(last_entry)
            state = last
        elif row['new_freq']:
            raise ValueError('같은 주파수 반복에서 새 주파수 표시')
        if row['np_limit'] < nrays:
            raise ValueError('포획 목표 광선 수 부족')
        last = {}
        moments = {}
        for pieces in (1,subdivisions):
            outputs = []
            j49 = 0.
            for r in range(nrays):
                ray = g['rays'][r]
                n = len(ray['R'])
                previous = state[pieces][r]
                inner_plus = row['boundary'][r][2]
                plus,minus = sweep(row['chi_ext'][:n],row['eta_ext'][:n],
                                   *geo[pieces][r], previous['I_M'],
                                   previous['I_P'], row['dlognu'],inner_plus,
                                   r < g['NC'],pieces)
                if (not np.isfinite(plus).all() or not np.isfinite(minus).all() or
                        np.min(plus) < -1e-25 or np.min(minus) < -1e-25):
                    raise ValueError('세분 전체 광선의 유한·비음수 조건 실패')
                outputs.append(dict(I_M=minus,I_P=plus))
                link = (int(ray['LNK'][49])-1)*pieces
                j49 += (plus[link]*g['weights']['JQW_P'][49,r] +
                        minus[link]*g['weights']['JQW_M'][49,r])
                if pieces == 1:
                    comparison = ((plus[0],row['boundary'][r][0]),
                                  (minus[0],row['boundary'][r][1]),
                                  (plus[-1],row['boundary'][r][2]),
                                  (minus[-1],row['boundary'][r][3]))
                    for offset in range(4):
                        if r < len(row['points'][offset]):
                            k = int(ray['LNK'][48+offset])-1
                            observed = row['points'][offset][r]
                            comparison += ((plus[k],observed[0]),
                                           (minus[k],observed[1]),
                                           (previous['I_P'][k],observed[2]),
                                           (previous['I_M'][k],observed[3]))
                    for calculated, observed in comparison:
                        baseline_max_scaled = max(
                            baseline_max_scaled,
                            abs(calculated-observed)/
                            max(abs(calculated),abs(observed),1e-30))
            last[pieces] = outputs
            moments[pieces] = j49
        if baseline_max_scaled > 1e-10:
            raise ValueError(f'원 전체경로 재현 실패: {baseline_max_scaled}')
        baseline_j_error = abs(moments[1]-row['native'][1]) / max(
            abs(moments[1]),abs(row['native'][1]),1e-30)
        if baseline_j_error > 1e-10:
            raise ValueError('원 전체 광선 각도합 J 재현 실패')
        last_entry = dict(frequency_index=row['index'],jit=row['jit'],
                          freq=row['freq'],
                          native49_J_original=float(row['native'][1]),
                          original_ray_J=float(moments[1]),
                          refined_ray_J=float(moments[subdivisions]),
                          refined_minus_original_ray_J=float(
                              moments[subdivisions]-moments[1]))
        if not full_band:
            summary.append(last_entry)
        previous_index = row['index']
        calls += 1
    if (not full_band and calls != max_calls) or calls == 0:
        raise ValueError('요청한 formal 호출 부족')
    final_entries.append(last_entry)
    if full_band and (previous_index != LAST or
                      len(final_entries) != LAST-FIRST+1):
        raise ValueError('전체 생산 대역 호출/주파수 누락')
    report = dict(status='FULL_PATH_FIXED_MATERIAL_CONDITIONAL',
                  calls=calls, rays_per_call=nrays,
                  subdivisions=subdivisions,
                  baseline_max_scaled_error=baseline_max_scaled,
                  sampled_calls=summary if not full_band else
                  [final_entries[-1-k] for k in (2543,2542,839)],
                  fixed_core_boundary=True, fixed_material=True,
                  actual_new_SE_RE_updates=0, final_goal='NOT_MET',
                  file_sha256={str(p): sha(p) for p in
                               (field/'siii_blue_in.bin',
                                field/'siii_compact.bin',candidates[0],
                                Path(__file__))})
    if full_band:
        descending = final_entries[::-1]
        freq = np.asarray([item['freq'] for item in descending])
        original_j = np.asarray([item['native49_J_original'] for item in descending])
        refined_j = np.asarray([item['refined_ray_J'] for item in descending])
        if (not np.isfinite(refined_j).all() or np.any(refined_j < 0) or
                not np.all(np.diff(freq) > 0)):
            raise ValueError('전체 세분 주파수·J 비유한/역순')
        weight, f_rate, j_rate, original, f_path = production_weight(
            freq*1e15,transport_root,baseline_rates,original_j)
        delta = refined_j-original_j
        delta_rate = float(np.sum(weight*delta))
        np.save(delta_path,delta,allow_pickle=False)
        report.update(frequencies=len(freq),
                      original_F_photo_cm3_s=f_rate,
                      original_J_photo_cm3_s=j_rate,
                      refined_J_photo_cm3_s=j_rate+delta_rate,
                      refined_F_minus_J_photo_cm3_s=f_rate-j_rate-delta_rate,
                      full_path_delta_J_photo_cm3_s=delta_rate,
                      delta_j_sha256=sha(delta_path))
        report['file_sha256'].update({str(p):sha(p) for p in
                                      (original,f_path,baseline_rates)})
    with out.open('x') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(result=str(out),calls=calls,
                          baseline_max_scaled_error=baseline_max_scaled,
                          last=last_entry,
                          full_path_delta_J_photo_cm3_s=
                          report.get('full_path_delta_J_photo_cm3_s'),
                          refined_F_minus_J_photo_cm3_s=
                          report.get('refined_F_minus_J_photo_cm3_s')),
                     ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--subdivisions',type=int,default=8)
    parser.add_argument('--max-calls',type=int,default=6)
    parser.add_argument('--full-band',action='store_true')
    parser.add_argument('--transport-root',type=Path)
    parser.add_argument('--baseline-rates',type=Path)
    args = parser.parse_args()
    run(args.capture.resolve(),args.out.resolve(),args.subdivisions,
        None if args.full_band else args.max_calls,
        args.transport_root.resolve() if args.transport_root else None,
        args.baseline_rates.resolve() if args.baseline_rates else None)
