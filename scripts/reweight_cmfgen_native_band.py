#!/usr/bin/env python3
"""완료 S III 대역장을 실제 native49 물질로 재가중한다. 새 수송/SE/RE는 없다.

과거 셀39 가중치는 그대로 보존하고 원70의 정확한 C full population을 쓴다.
600–750Å 부분 광흡수율이며 전주파수 순이온화율이나 최종 이온분율은 아니다.
"""
import argparse
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np

import cmfgen_model_continuum as B
import lumina_native90_rates as R
import lumina_native_columns as N
from compare_lumina_checkpoint_rates import photon_weights
from run_cmfgen_formal_primary import fields

BASE_SHA = 'f3b01d29c798e910fe42427283f1a9697189da45926d0bc4b77961e991909bab'
WEIGHTS_SHA = 'cc31382dba09deaeca2dd5e5938975cc1a697f119ae9dc675835377e6ca6441f'
REFINED_SHA = 'afcd477b2eae5d7406004482bfb1747cb2500f0d3d4e20381c10fdb2c3c546f5'


def band_weights(full_frequency, mask, absorption):
    """부분대역 끝점도 전 원격자의 사다리꼴 폭을 유지한다."""
    mask = np.asarray(mask)
    absorption = np.asarray(absorption, dtype='f8')
    measure = photon_weights(full_frequency)
    if (mask.dtype != np.dtype(bool) or mask.shape != measure.shape or
            absorption.shape != (int(mask.sum()),) or not mask.any() or
            not np.isfinite(absorption).all() or np.any(absorption < 0)):
        raise ValueError('같은 원 격자의 대역 마스크/양의 흡수계수 필요')
    return measure[mask]*absorption


def rates(weight, radiation, ion_density):
    """동일한 하위 이온 인구로만 부분대역 유효 광흡수율을 정규화한다."""
    weight = np.asarray(weight, dtype='f8')
    if (weight.ndim != 1 or not len(weight) or not np.isfinite(weight).all() or
            np.any(weight < 0) or not np.isfinite(ion_density) or ion_density <= 0):
        raise ValueError('부피 흐름과 같은 물질의 양의 이온 수밀도 필요')
    result = {}
    for name, value in radiation.items():
        value = np.asarray(value, dtype='f8')
        if value.shape != weight.shape or not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError('완성된 같은 대역의 유한 비음수 장 필요')
        photo = math.fsum(float(x) for x in weight*value)
        result[name] = dict(photo_cm3_s=photo, effective_photo_s=photo/ion_density)
    return result


def compare(base_path, weight_path, refined_path, out):
    if out.exists(): raise FileExistsError('기존 native 물질 대역율 보존')
    started = time.monotonic()
    for path, expected in ((base_path, BASE_SHA), (weight_path, WEIGHTS_SHA), (refined_path, REFINED_SHA)):
        if B.sha(path) != expected: raise ValueError('봉인 완료 입력 변경: '+path.name)
    base = json.loads(base_path.read_text()); cpath = Path(base['config'])
    c = json.loads(cpath.read_text()); plan = json.loads(Path(c['plan']).read_text())
    if (plan['reference_radiation_material_record'], plan['reference_radiation_record']) != (70, 71):
        raise ValueError('원70/J71 물질/장 세대 필요')
    source_path = Path(c['reuse_sources']['boundary_sources']['directory'])/'sources.json'
    if B.sha(source_path) != c['reuse_sources']['boundary_sources']['report_sha256']:
        raise ValueError('원 물질 계수 보고서 변경')
    source_report = json.loads(source_path.read_text())
    refined = json.loads(refined_path.read_text()); delta_path = refined_path.with_suffix('.npy')
    if (refined['status'] != 'FULL_PATH_FIXED_MATERIAL_CONDITIONAL' or
            refined['frequencies'] != 5196 or refined['subdivisions'] != 128 or
            B.sha(delta_path) != refined['delta_j_sha256']):
        raise ValueError('완료 공간128 대역 증거 필요')
    reference = Path(plan['same_material_reference_field'])/'FORMAL_PRIMARY_FIELDS.bin'
    if B.sha(reference) != plan['reference_field_sha256']:
        raise ValueError('원 주파수/복사장 변경')
    geometry, records = fields(reference)
    with N.open_sealed_native(base['config'], base['preparation'], base['preparation_sha256']) as source:
        if source.identity != base['source_identity']: raise ValueError('다른 native 물질 계수 세대')
        if not np.array_equal(geometry[0]*1.e10, source.radius):
            raise ValueError('원 물질·장 반경 주소 불일치')
        rows, models, routes, state, dissolved, full_check, _ = R.material_state(c, plan, source, source_report)
        node = 49
        frequency = np.asarray(records['freq'][::-1])*1.e15
        mask = (2.99792458e18/frequency >= 600.) & (2.99792458e18/frequency < 750.)
        nu = frequency[mask]
        with np.load(weight_path, allow_pickle=False) as old:
            if not np.array_equal(nu, old['freq_Hz']) or len(nu) != 5196:
                raise ValueError('완료 대역의 원 주파수 주소 불일치')
            original = np.asarray(records['field'][::-1, 0, node])[mask]
            if not np.allclose(original, old['original_J'], rtol=1.e-12, atol=0):
                raise ValueError('완료 대역의 원 native49 J 불일치')
            old_weight = old['photo_weight'].copy(); own = old['F8'].copy()
        delta = np.load(delta_path, allow_pickle=False)
        radiation = dict(original_CMFGEN=original, spatial128_CMFGEN=original+delta, legacy_Lumina_F=own)
        def point(matter, index):
            return {k: (v[:, [index]] if k in ('full','superlevel','occupation') else
                        v[[index]] if k in ('physical','temperature','ne') else v)
                    for k, v in matter.items()}
        selected = point(state, node)
        ion_rows = [i for i, r in enumerate(rows) if (int(r['atomic_number']), int(r['ion_number'])) == (16, 2)]
        population = float(selected['full'][ion_rows].sum())
        ion_routes = [r for r in routes if (r.Z, r.stage) == (16, 3)]
        if len(ion_routes) != 256: raise ValueError('S III 활성 PHOT 경로 누락')
        # 주소만 바꾼 대조인지 확인한다. 현재 같은 생산식으로 과거 셀39 가중치를 재생산한다.
        old_material = Path(c['cells'])/'material_cells.bin'
        old_source_path = Path(c['reuse_sources']['cell_sources']['directory'])/'sources.json'
        if B.sha(old_source_path) != c['reuse_sources']['cell_sources']['report_sha256']:
            raise ValueError('과거 셀 물질 계수 보고서 변경')
        old_source = json.loads(old_source_path.read_text())
        if B.sha(old_material) != old_source['input_sha256'].get(str(old_material)):
            raise ValueError('과거 셀39 물질 변경')
        old_state = B.read_material(old_material, rows, models)
        if dissolved:
            old_state = B.read_material(old_material, rows, models, B.material_occupation(rows, models, old_state))
        old_point = point(old_state, 39)
        old_coefficients = B.continuum_grid(ion_routes, rows, old_point, nu, 512)
        if dissolved: B.add_dissolved_continuum(old_coefficients, ion_routes, rows, models, old_point, nu)
        rebuilt_weight = band_weights(frequency, mask, old_coefficients['chi_abs'][0])
        if not np.allclose(rebuilt_weight, old_weight, rtol=5.e-11, atol=0):
            raise ValueError('주소 외에 PHOT/용해 재구성 차이가 있어 단순 재가중으로 비교 불가')
        relative = np.divide(abs(rebuilt_weight-old_weight), old_weight,
                             out=np.zeros_like(old_weight), where=old_weight > 0)
        coefficients = B.continuum_grid(ion_routes, rows, selected, nu, 512)
        bound_weight = band_weights(frequency, mask, coefficients['chi_abs'][0])
        if dissolved: B.add_dissolved_continuum(coefficients, ion_routes, rows, models, selected, nu)
        native_weight = band_weights(frequency, mask, coefficients['chi_abs'][0])
        native = rates(native_weight, radiation, population)
        bound = rates(bound_weight, radiation, population)
        legacy = {name: math.fsum(map(float, old_weight*j)) for name, j in radiation.items()}
        for label, key in (('original_CMFGEN','original_J_photo_cm3_s'),
                           ('spatial128_CMFGEN','refined_J_photo_cm3_s'),
                           ('legacy_Lumina_F','original_F_photo_cm3_s')):
            if not math.isclose(legacy[label], refined[key], rel_tol=2.e-12, abs_tol=0):
                raise ValueError('과거 셀39 가중 흐름 재현 실패')
        out.mkdir()
        np.savez(out/'native-band-weights.npz', frequency_Hz=nu, photo_weight=native_weight,
                 bound_photo_weight=bound_weight, ion_density_cm3=population,
                 radius_cm=float(source.radius[node]))
        inputs = {str(p): B.sha(p) for p in (base_path, cpath, Path(c['plan']), source_path,
            weight_path, refined_path, delta_path, old_material, old_source_path,
            Path(c['seed'])/'native_material_nodes.bin',
            Path(c['native_full_population']), Path(c['cells'])/'atomic/levels.csv',
            Path(__file__), Path(B.__file__), Path(R.__file__), Path(N.__file__))}
        inputs[str(reference)] = plan['reference_field_sha256']
        report = dict(status='COMPLETE_NATIVE49_BAND_REWEIGHT', material_record=70,
            reference_record=71, native_node_zero_based=node, native_radius_cm=float(source.radius[node]),
            ion='S III', ion_density_cm3=population, frequencies=len(nu), routes=len(ion_routes),
            native49_total=native, native49_bound=bound, legacy_cell39_photo_cm3_s=legacy,
            legacy_weight_reproduction_max_relative=float(relative.max()),
            exact_C_full_population=full_check, input_sha256=inputs,
            output_sha256=B.sha(out/'native-band-weights.npz'), elapsed_seconds=time.monotonic()-started,
            interpretation='같은 native49 원70 인구의 부분대역 광흡수. 순이온화/새 물질해/전대역 Γ가 아님',
            coefficient_sampling='actual native frequencies; opacity dissolution policy as legacy weight',
            new_transport_calls=0, new_SE_RE_updates=0, new_global_BF_integrations=0,
            native_band_absorption_weight_evaluations=1, final_goal='NOT_MET')
        encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        with (out/'native-band-rates.json').open('x') as stream: stream.write(encoded)
    records._mmap.close()
    print(encoded)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base','weights','refined','out'): parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    compare(args.base.resolve(), args.weights.resolve(), args.refined.resolve(), args.out.resolve())
