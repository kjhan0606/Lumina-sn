#!/usr/bin/env python3
"""봉인 S III 원광선 55개 outward 한 구간의 공간 세분을 J·PHOT율로 합산한다.

안쪽 입사·바깥 이웃 source·minus와 나머지 광선은 원 포획으로 고정한다.
이것은 국소 조건부 민감도이며 전체 formal/SE/RE 또는 새 스펙트럼이 아니다.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze_cmfgen_formal_capture import geometry
from analyze_cmfgen_siii_compact_capture import records
from compare_cmfgen_siii_ray_substeps import sc_sweep
from compare_lumina_checkpoint_rates import photon_weights
import compare_lumina_native_node_rates as N
from prepare_cmfgen_mom_capture import sha
from prepare_cmfgen_siii_compact_capture import FIRST, LAST
from run_cmfgen_formal_primary import fields


def collect_plus(capture):
    field = capture / 'field'
    candidates = list(field.glob(f'FORMAL_{FIRST}_*_geometry.bin'))
    if len(candidates) != 1:
        raise ValueError('첫 주파수 봉인 기하가 유일하지 않음')
    g = geometry(candidates[0])
    nrays = g['NP'] - 50  # Python native49→50, Fortran ID50→51
    if nrays != 55 or g['RAY_POINTS_INSERTED']:
        raise ValueError('봉인 native49 outward 광선 계약 불일치')
    links = np.array([[int(ray['LNK'][j])-1 for j in (48, 49, 50)]
                      for ray in g['rays'][:nrays]])
    far_link, outer_link, inner_link = links.T
    if np.any(far_link >= outer_link) or np.any(outer_link >= inner_link):
        raise ValueError('원 광선 native 링크 순서 오류')
    frame = []
    frequencies = []
    intervals = []
    native_j = []
    indices = []

    def keep(row):
        far = np.asarray(row['points'][0][:nrays], dtype='f8')
        outer = np.asarray(row['points'][1][:nrays], dtype='f8')
        inner = np.asarray(row['points'][2][:nrays], dtype='f8')
        arrays = np.stack((inner[:, 0], outer[:, 0], inner[:, 2],
                           outer[:, 2], far[:, 2],
                           row['chi_ext'][inner_link],
                           row['chi_ext'][outer_link],
                           row['chi_ext'][far_link],
                           row['eta_ext'][inner_link],
                           row['eta_ext'][outer_link],
                           row['eta_ext'][far_link]), axis=1)
        if not np.isfinite(arrays).all():
            raise ValueError('원 광선 장·계수 비유한')
        frame.append(arrays)
        frequencies.append(row['freq'])
        intervals.append(row['dlognu'])
        native_j.append(row['native'][1])
        indices.append(row['index'])

    prior = None
    for row in records(field / 'siii_compact.bin', g, include_rays=True):
        if prior is not None and row['index'] != prior['index']:
            keep(prior)
        prior = row
    if prior is not None:
        keep(prior)
    if (len(indices) != LAST-FIRST+1 or
            indices != list(range(FIRST, LAST+1))):
        raise ValueError('원 목표 대역 최종 formal 호출 누락')
    data = np.stack(frame[::-1])
    freq = np.asarray(frequencies[::-1], dtype='f8')
    dlognu = np.asarray(intervals[::-1], dtype='f8')
    j = np.asarray(native_j[::-1], dtype='f8')
    if (not np.isfinite(data).all() or
            np.any(data[:, :, [0, 1, 2, 3, 4, 8, 9, 10]] < 0) or
            not np.allclose(np.diff(np.log(freq)), dlognu[:-1],
                            rtol=1e-8, atol=1e-12) or
            np.max(np.abs(data[:-1, :, 3]-data[1:, :, 1])) != 0 or
            np.max(np.abs(data[:-1, :, 2]-data[1:, :, 0])) != 0):
        raise ValueError('원 광선 청색 연쇄 또는 주파수 계약 불일치')
    weights = np.asarray(g['weights']['JQW_P'][49, :nrays], dtype='f8')
    if np.any(weights < 0) or not np.isfinite(weights).all():
        raise ValueError('원 outward 각도 가중치 오류')
    rays = []
    names = ('incoming', 'observed', 'previous_inner', 'previous_outer',
             'previous_far', 'chi_inner', 'chi_outer', 'chi_far',
             'eta_inner', 'eta_outer', 'eta_far')
    for r in range(nrays):
        ray = g['rays'][r]
        f, o, i = links[r]
        length = float(ray['S_P'][o]-ray['S_P'][i])
        far_length = float(ray['S_P'][f]-ray['S_P'][o])
        if length <= 0 or far_length <= 0:
            raise ValueError('원 광선 공간 길이 오류')
        values = {name: data[:, r, k] for k, name in enumerate(names)}
        values.update(length=length, dlognu=dlognu,
                      b_inner=float(ray['B_P'][i]),
                      b_outer=float(ray['B_P'][o]),
                      far_neighbor=dict(previous=values['previous_far'],
                                        chi=values['chi_far'], eta=values['eta_far'],
                                        b=float(ray['B_P'][f]), length=far_length))
        rays.append(values)
    return dict(freq=freq, native_j=j, rays=rays, angle_weights=weights,
                geometry=candidates[0], compact=field/'siii_compact.bin')


def production_weight(nu_band, transport_root, baseline_rates, capture_j):
    record = json.loads((transport_root/'execution.json').read_text())
    rates = json.loads(baseline_rates.read_text())
    if (record['status'] != 'NATIVE_POSITION_F_COMPLETE' or
            record['native_indices'] != {'outer': 49, 'inner': 50} or
            record['shell'] != N.SHELL or
            record['material_record'] != 70 or record['reference_record'] != 71 or
            rates['material_record'] != 70 or rates['reference_record'] != 71):
        raise ValueError('봉인 F/J·물질 신원 불일치')
    original_paths = [Path(p) for p in record['input_sha256']
                      if Path(p).name == 'FORMAL_PRIMARY_FIELDS.bin']
    f_path = transport_root/'F-native-positions.npy'
    if (len(original_paths) != 1 or
            sha(original_paths[0]) != record['input_sha256'][str(original_paths[0])] or
            sha(f_path) != record['output']['sha256']):
        raise ValueError('원 장·완성 F 신원 불일치')
    config = json.loads(N.CONFIG.read_text())
    nu_own = np.load(config['grid'], mmap_mode='r', allow_pickle=False)
    f = np.load(f_path, mmap_mode='r', allow_pickle=False)
    g, original = fields(original_paths[0])
    try:
        nu = np.asarray(original['freq'][::-1])*1e15
        mask = (2.99792458e18/nu >= 600) & (2.99792458e18/nu < 750)
        if (int(mask.sum()) != len(nu_band) or
                not np.array_equal(nu[mask], nu_band) or
                not np.allclose(g[0, [49, 50]]*1e10,
                                record['native_radii_cm'], rtol=1e-12, atol=0) or
                not np.allclose(original['field'][::-1, 0, 49][mask],
                                capture_j, rtol=1e-10, atol=0)):
            raise ValueError('원 장·포획 native49 J·격자 대응 불일치')
        projected_f = np.interp(np.log(nu_band), np.log(nu_own), f[0])
        plan = json.loads(Path(config['plan']).read_text())
        rows, models, routes, policy, _, counts = N.B.load_model(
            Path(plan['initial_material']), Path(config['cells'])/'atomic',
            Path(config['phot_library']))
        source = json.loads((Path(config['reuse_sources']['cell_sources']['directory'])/
                             'sources.json').read_text())
        if policy != source['process_selection'] or counts != source['routes_by_ion']:
            raise ValueError('원 PHOT 경로 변경')
        material_path = Path(config['cells'])/'material_cells.bin'
        matter = N.B.read_material(material_path, rows, models)
        do_dis = N.B.dissolution_flag(Path(plan['initial_material'])/'MODEL')
        if do_dis:
            matter = N.B.read_material(material_path, rows, models,
                                       N.B.material_occupation(rows, models, matter))
        state = {key: (value[:, [N.SHELL]] if key in
                       ('full', 'superlevel', 'occupation') else
                       value[[N.SHELL]] if key in ('physical', 'temperature', 'ne')
                       else value)
                 for key, value in matter.items()}
        selected = [route for route in routes if (route.Z, route.stage) == (16, 3)]
        if not selected:
            raise ValueError('생산 S III PHOT route 없음')
        coeff = N.B.continuum_grid(selected, rows, state, nu_band, 512)
        if do_dis:
            N.B.add_dissolved_continuum(coeff, selected, rows, models,
                                        state, nu_band)
        weight = coeff['chi_abs'][0]*photon_weights(nu)[mask]
        f_rate = float(np.sum(projected_f*weight))
        j_rate = float(np.sum(capture_j*weight))
        expected = rates['grid_comparison']['original_grid_photo_cm3_s']
        if (abs(f_rate-expected['F_outer49_band']) > 1e-6 or
                abs(j_rate-expected['J_outer49_band']) > 1e-6):
            raise ValueError('봉인 S III 광흡수율 재현 실패')
        return weight, f_rate, j_rate, original_paths[0], f_path
    finally:
        original._mmap.close()
        nu_own._mmap.close()
        f._mmap.close()


def analyze(capture, transport_root, baseline_rates, out_root, subdivisions=8,
            initial_profile='linear'):
    if out_root.exists():
        raise FileExistsError('기존 outward 각도/율 결과 보존')
    if type(subdivisions) is not int or not 2 <= subdivisions <= 64:
        raise ValueError('한정된 조건부 공간 세분 수 필요')
    if initial_profile not in ('linear', 'inner', 'outer'):
        raise ValueError('새 내부 노드 첫 청색 강도 프로필 오류')
    captured = collect_plus(capture)
    photo_weight, f_rate, j_rate, original, f_path = production_weight(
        captured['freq']*1e15, transport_root, baseline_rates,
        captured['native_j'])
    delta = np.zeros(len(captured['freq']))
    baseline_max_abs = 0.
    for r, ray in enumerate(captured['rays']):
        baseline = sc_sweep(ray, 1, mode='bezier')
        error = np.max(np.abs(baseline-ray['observed']))
        baseline_max_abs = max(baseline_max_abs, float(error))
        if error > 1e-10*max(float(np.max(ray['observed'])), 1e-300):
            raise ValueError(f'원 outward 광선 {r+1} 전대역 재현 실패')
        refined = sc_sweep(ray, subdivisions, mode='bezier',
                           initial_profile=initial_profile)
        delta += captured['angle_weights'][r]*(refined-ray['observed'])
    j_adjusted = captured['native_j']+delta
    if not np.isfinite(j_adjusted).all() or np.any(j_adjusted < 0):
        raise ValueError('조건부 각도 적분 J 오류')
    delta_rate = float(np.sum(photo_weight*delta))
    selected = {str(k): dict(wavelength_A=float(2.99792458e18/
                                              (captured['freq'][k]*1e15)),
                             original_J=float(captured['native_j'][k]),
                             outward_delta_J=float(delta[k]),
                             conditional_J=float(j_adjusted[k]))
                for k in (2543, 2542, 839)}
    out_root.mkdir(parents=False)
    delta_path = out_root/'outward_delta_J.npy'
    np.save(delta_path, delta, allow_pickle=False)
    report = dict(scope=__doc__, status='LOCAL_CONDITIONAL_ONLY',
                  native_zero_index=49, outward_crossing_ray_count=len(captured['rays']),
                  frequencies=len(captured['freq']), subdivisions=subdivisions,
                  initial_profile=initial_profile,
                  original_bezier_baseline_max_abs=baseline_max_abs,
                  plus_angle_weight_sum=float(np.sum(captured['angle_weights'])),
                  original_F_photo_cm3_s=f_rate,
                  original_J_photo_cm3_s=j_rate,
                  original_F_minus_J_photo_cm3_s=f_rate-j_rate,
                  conditional_J_photo_cm3_s=j_rate+delta_rate,
                  conditional_F_minus_J_photo_cm3_s=f_rate-j_rate-delta_rate,
                  outward_delta_J_photo_cm3_s=delta_rate,
                  samples=selected,
                  delta_j_sha256=sha(delta_path),
                  file_sha256={str(p): sha(p) for p in
                               (captured['geometry'], captured['compact'],
                                original, f_path, baseline_rates,
                                Path(__file__))},
                  actual_new_transport_calls=0, SE_RE_updates=0,
                  final_goal='NOT_MET')
    report_path = out_root/'report.json'
    with report_path.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2,
                  allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(result=str(report_path),
                          baseline_max_abs=baseline_max_abs,
                          outward_delta_J_photo_cm3_s=delta_rate,
                          conditional_F_minus_J_photo_cm3_s=
                          report['conditional_F_minus_J_photo_cm3_s'],
                          samples=selected), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--transport-root', type=Path, required=True)
    parser.add_argument('--baseline-rates', type=Path, required=True)
    parser.add_argument('--out-root', type=Path, required=True)
    parser.add_argument('--subdivisions', type=int, default=8)
    parser.add_argument('--initial-profile', choices=('linear', 'inner', 'outer'),
                        default='linear')
    args = parser.parse_args()
    analyze(args.capture.resolve(), args.transport_root.resolve(),
            args.baseline_rates.resolve(), args.out_root.resolve(),
            args.subdivisions, args.initial_profile)
