#!/usr/bin/env python3
"""봉인 물질70/J71의 S III 율 잔차에서 실제 광선 비교 지점을 고른다.

완성 F/J와 원 생산 PHOT/Milne만 읽는다. 새 수송·SE/RE는 하지 않는다.
개별 주파수점의 율 기여는 원 격자 전체 사다리꼴 가중치를 쓴다.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze_cmfgen_formal_capture import geometry
from compare_lumina_checkpoint_rates import photon_weights
import compare_lumina_native_node_rates as N
from prepare_cmfgen_mom_capture import sha
from run_cmfgen_formal_primary import fields


def top_spots(contribution, tau, blue_margin, drift, count=8):
    contribution = np.asarray(contribution, dtype='f8')
    tau = np.asarray(tau, dtype='f8')
    blue_margin = np.asarray(blue_margin, dtype='f8')
    if (any(a.shape != contribution.shape for a in (tau, blue_margin)) or
            not all(np.isfinite(a).all() for a in
                    (contribution, tau, blue_margin)) or
            np.any(tau < 0) or drift <= 0 or count < 1):
        raise ValueError('S III 표본 선택 입력 오류')
    eligible = np.flatnonzero((tau < 100) & (blue_margin >= 2 * drift) &
                              (contribution > 0))
    chosen = eligible[np.argsort(-contribution[eligible], kind='stable')[:count]]
    return [dict(reverse_band_index=int(k),
                 signed_photo_residual_cm3_s=float(contribution[k]),
                 radial_tau=float(tau[k]),
                 blue_margin_log_frequency=float(blue_margin[k]))
            for k in chosen]


def analyze(transport_root, capture, baseline_rates, out):
    if out.exists():
        raise FileExistsError('기존 S III 주파수 순위 결과 보존')
    record = json.loads((transport_root / 'execution.json').read_text())
    rates = json.loads(baseline_rates.read_text())
    if (record['status'] != 'NATIVE_POSITION_F_COMPLETE' or
            record['shell'] != N.SHELL or record['native_indices'] !=
            {'outer': 49, 'inner': 50} or
            record['material_record'] != 70 or record['reference_record'] != 71 or
            rates['material_record'] != 70 or rates['reference_record'] != 71 or
            rates['cells_with_populations'] != [N.SHELL]):
        raise ValueError('봉인 F/J·셀39 신원 불일치')
    f_path = transport_root / 'F-native-positions.npy'
    own_grid_path = Path(json.loads(N.CONFIG.read_text())['grid'])
    formal_paths = [Path(p) for p in record['input_sha256']
                    if Path(p).name == 'FORMAL_PRIMARY_FIELDS.bin']
    if (len(formal_paths) != 1 or
            sha(f_path) != record['output']['sha256'] or
            sha(formal_paths[0]) != record['input_sha256'][str(formal_paths[0])] or
            sha(formal_paths[0]) !=
            rates['grid_comparison']['original_field_sha256'][str(formal_paths[0])]):
        raise ValueError('원 장 또는 완성 F 해시 불일치')
    field = capture / 'field'
    geom_files = list(field.glob('FORMAL_40286_*_geometry.bin'))
    if len(geom_files) != 1:
        raise ValueError('포획 기하가 유일하지 않음')
    g = geometry(geom_files[0])
    ray = g['rays'][0]
    outer, inner = (int(ray['LNK'][j]) - 1 for j in (49, 50))
    length = float(ray['S_P'][outer] - ray['S_P'][inner])
    drift = .5 * float(ray['B_P'][outer] + ray['B_P'][inner]) * length
    if length <= 0 or drift <= 0:
        raise ValueError('목표 광선의 길이·청색 이동량 오류')

    original_geometry, original = fields(formal_paths[0])
    own_nu = np.load(own_grid_path, mmap_mode='r', allow_pickle=False)
    f = np.load(f_path, mmap_mode='r', allow_pickle=False)
    try:
        if (original['field'].shape[2] <= 50 or f.shape[0] < 2 or
                f.shape[1] != len(own_nu) or not np.allclose(
                    original_geometry[0, [49, 50]] * 1e10,
                    record['native_radii_cm'], rtol=1e-12, atol=0)):
            raise ValueError('F/J 관측 반경 또는 주파수 형상 불일치')
        nu = np.asarray(original['freq'][::-1]) * 1e15
        wave = 2.99792458e18 / nu
        band = (wave >= 600.) & (wave < 750.)
        if band.sum() != 5196 or not np.all(np.diff(np.flatnonzero(band)) == 1):
            raise ValueError('원 600–750Å 격자 재고 불일치')
        nu_band = nu[band]
        own_f = np.asarray(f[0], dtype='f8')
        if nu_band[0] < own_nu[0] or nu_band[-1] > own_nu[-1]:
            raise ValueError('완성 F 보간 경계 부족')
        projected_f = np.interp(np.log(nu_band), np.log(own_nu), own_f)
        j = np.asarray(original['field'][::-1, 0, 49])[band]
        chi_outer = np.asarray(original['field'][::-1, 4, 49])[band]
        chi_inner = np.asarray(original['field'][::-1, 4, 50])[band]
        radial_tau = .5 * (chi_outer + chi_inner) * (
            original_geometry[0, 49] - original_geometry[0, 50])
        if np.any(radial_tau < 0) or not np.isfinite(radial_tau).all():
            raise ValueError('원 방사 χ 깊이 오류')
        config = json.loads(N.CONFIG.read_text())
        plan = json.loads(Path(config['plan']).read_text())
        rows, models, routes, policy, _, counts = N.B.load_model(
            Path(plan['initial_material']), Path(config['cells']) / 'atomic',
            Path(config['phot_library']))
        source = json.loads((Path(config['reuse_sources']['cell_sources']['directory']) /
                             'sources.json').read_text())
        if policy != source['process_selection'] or counts != source['routes_by_ion']:
            raise ValueError('원 PHOT route/선택 변경')
        material_path = Path(config['cells']) / 'material_cells.bin'
        matter = N.B.read_material(material_path, rows, models)
        do_dis = N.B.dissolution_flag(Path(plan['initial_material']) / 'MODEL')
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
            raise ValueError('S III PHOT route 없음')
        coefficients = N.B.continuum_grid(selected, rows, state, nu_band, 512)
        if do_dis:
            N.B.add_dissolved_continuum(coefficients, selected, rows, models,
                                        state, nu_band)
        absorption = coefficients['chi_abs'][0]
        weights = photon_weights(nu)[band]
        f_rate = float(np.sum(absorption * projected_f * weights))
        j_rate = float(np.sum(absorption * j * weights))
        reference = rates['grid_comparison']['original_grid_photo_cm3_s']
        if (abs(f_rate - reference['F_outer49_band']) > 1e-6 or
                abs(j_rate - reference['J_outer49_band']) > 1e-6):
            raise ValueError('봉인 원격자 S III 광흡수율 재현 실패')
        contribution = absorption * (projected_f - j) * weights
        margin = np.log(nu_band[-1] / nu_band)
        spots = top_spots(contribution, radial_tau, margin, drift)
        for spot in spots:
            k = spot['reverse_band_index']
            spot.update(wavelength_A=float(2.99792458e18 / nu_band[k]),
                        original_frequency_index_1based=int(len(nu) -
                                                             np.flatnonzero(band)[k]))
        result = dict(scope=__doc__, status='WEIGHTED_DIAGNOSTIC_ONLY',
                      original_grid_band_points=int(band.sum()),
                      band_photo_F_cm3_s=f_rate, band_photo_J_cm3_s=j_rate,
                      signed_residual_cm3_s=float(np.sum(contribution)),
                      positive_residual_tau_lt_100_cm3_s=float(np.sum(
                          contribution[(radial_tau < 100) & (contribution > 0)])),
                      signed_residual_tau_lt_100_cm3_s=float(np.sum(
                          contribution[radial_tau < 100])),
                      selection_rule='positive S III PHOT-weighted F-J, radial tau<100, two ray drifts of blue margin',
                      ray1_drift_log_frequency=drift, top_spots=spots,
                      file_sha256={str(p): sha(p) for p in
                                   (formal_paths[0], f_path, geom_files[0],
                                    baseline_rates, N.CONFIG, Path(__file__))},
                      actual_new_transport_calls=0, SE_RE_updates=0,
                      final_goal='NOT_MET')
        with out.open('x') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2,
                      allow_nan=False)
            stream.write('\n')
        print(json.dumps(dict(result=str(out), residual=result['signed_residual_cm3_s'],
                              top_spots=spots[:3]), ensure_ascii=False))
    finally:
        original._mmap.close()
        own_nu._mmap.close()
        f._mmap.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transport-root', type=Path, required=True)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--baseline-rates', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.transport_root.resolve(), args.capture.resolve(),
            args.baseline_rates.resolve(), args.out.resolve())
