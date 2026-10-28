#!/usr/bin/env python3
"""저장된 S III 대역 비교의 계수 규약을 확인한다. 새 수송/SE/RE는 없다.

원 CONT_FREQ 참조, 완성된 직접평가 참조, legacy F의 실제 C 계수를 구분한다.
χ는 소광용 signed 값이며 광이온화율 가중치로 사용하지 않는다.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from prepare_lumina_outer_input import read_auxiliary, sha
from run_cmfgen_formal_primary import fields

DIRECT_SHA = '3a9dc92124479f0bb2e9adcd1e548885f4568fb9deb7143f2d20d18b148cc058'
WEIGHT_SHA = '4e7c219d0fedf4dc12f505407aecfc9ef8918596428d8784b322baaeead99f8b'


def log_sample(frequency, values, requested):
    frequency, values, requested = [np.asarray(x, dtype='f8') for x in (frequency, values, requested)]
    if (frequency.ndim != 1 or len(frequency) < 2 or values.shape != frequency.shape or
            requested.ndim != 1 or not len(requested) or
            not all(np.isfinite(x).all() for x in (frequency, values, requested)) or
            np.any(frequency <= 0) or np.any(np.diff(frequency) <= 0) or
            np.any(requested < frequency[0]) or np.any(requested > frequency[-1])):
        raise ValueError('같은 증가 주파수의 유한 점값과 외삽 없는 요청 필요')
    return np.interp(np.log(requested), np.log(frequency), values)


def contrast(value, reference):
    value, reference = [np.asarray(x, dtype='f8') for x in (value, reference)]
    if (value.ndim != 1 or not len(value) or reference.shape != value.shape or
            not np.isfinite(value).all() or not np.isfinite(reference).all()):
        raise ValueError('유한한 같은 크기 대역 배열 필요')
    scale = np.maximum(abs(value), abs(reference))
    symmetric = np.divide(abs(value-reference), scale, out=np.zeros_like(scale), where=scale > 0)
    positive = (value > 0) & (reference > 0)
    ratio = value[positive]/reference[positive]
    return dict(symmetric_max=float(symmetric.max()),
        symmetric_p95=float(np.quantile(symmetric, .95)),
        positive_ratio_min=float(ratio.min()) if len(ratio) else None,
        positive_ratio_median=float(np.median(ratio)) if len(ratio) else None,
        positive_ratio_max=float(ratio.max()) if len(ratio) else None,
        nonpositive_pairs=int((~positive).sum()))


def compare(legacy_run, direct_run, weights, out):
    if out.exists(): raise FileExistsError('기존 입력 규약 대조 보존')
    if any(os.environ.get(k) != '1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):
        raise ValueError('소형 CPU1 스레드 환경 필요')
    started = time.monotonic()
    execution_path = legacy_run/'execution.json'
    legacy = json.loads(execution_path.read_text())
    runtime_path = direct_run/'run/direct-runtime.json'
    direct = json.loads(runtime_path.read_text())
    inputs_path = direct_run/'run/direct-inputs.json'
    inputs = json.loads(inputs_path.read_text())
    if (legacy['status'] != 'NATIVE_POSITION_F_COMPLETE' or
            (legacy['material_record'], legacy['reference_record']) != (70, 71) or
            direct['state'] != 'FIELD_COMPLETE' or not direct['field_complete'] or
            direct['coverage']['count'] != 197291 or not direct['material_byte_unchanged'] or
            not direct['geometry_and_frequency_unchanged'] or inputs['input_state_record'] != 70):
        raise ValueError('완료된 동일 물질70 참조와 legacy F 필요')
    source_reports = [Path(p) for p in legacy['input_sha256'] if Path(p).name == 'sources.json']
    if len(source_reports) != 1: raise ValueError('legacy 실제 계수 출처가 유일하지 않음')
    source_path = source_reports[0]; source = json.loads(source_path.read_text())
    source_binary = source_path.with_name('sources.bin')
    grids = [Path(p) for p in source['input_sha256'] if Path(p).name == 'frequency.npy']
    if len(grids) != 1: raise ValueError('legacy 계수 실제 주파수 격자가 유일하지 않음')
    grid_path = grids[0]
    original_path = Path(inputs['reference'])/'field/FORMAL_PRIMARY_FIELDS.bin'
    direct_path = direct_run/'run/field/FORMAL_PRIMARY_FIELDS.bin'
    expected = {source_path: legacy['input_sha256'][str(source_path)],
        source_binary: legacy['input_sha256'][str(source_binary)],
        original_path: inputs['baseline_field_sha256'], direct_path: DIRECT_SHA,
        grid_path: source['input_sha256'][str(grid_path)], weights: WEIGHT_SHA}
    if (direct['evidence_sha256']['FORMAL_PRIMARY_FIELDS.bin'] != DIRECT_SHA or
            source['output_sha256']['sources.bin'] != expected[source_binary] or
            legacy['input_sha256'].get(str(original_path)) != expected[original_path]):
        raise ValueError('참조/계수 계보 불일치')
    for path, identity in expected.items():
        if sha(path) != identity: raise ValueError('봉인 입력 변경: '+path.name)
    old_geometry, old = fields(original_path)
    new_geometry, new = fields(direct_path)
    try:
        if (not np.array_equal(old_geometry, new_geometry) or
                not np.array_equal(old['freq'], new['freq']) or
                not np.array_equal(old['weight'], new['weight'])):
            raise ValueError('물질 기하·주파수·적분 주소 불일치')
        frequency = np.asarray(old['freq'][::-1])*1.e15
        mask = (2.99792458e18/frequency >= 600.) & (2.99792458e18/frequency < 750.)
        nu = frequency[mask]
        with np.load(weights, allow_pickle=False) as w:
            if not np.array_equal(nu, w['frequency_Hz']): raise ValueError('부분대역 가중치 주소 불일치')
            weight = w['photo_weight'].copy(); density = float(w['ion_density_cm3'])
        if len(nu) != 5196: raise ValueError('원 S III 대역 coverage 불일치')
        grid = np.load(grid_path, allow_pickle=False)
        nodes = [49, 50, 58]
        chi, es, eta = read_auxiliary(source_binary, source, grid, nodes)
        comparison = {}; arrays = {'frequency_Hz': nu, 'native49_photo_weight': weight}
        for at, node in enumerate(nodes):
            old_values = np.asarray(old['field'][::-1, :, node])[mask]
            new_values = np.asarray(new['field'][::-1, :, node])[mask]
            projected_j = log_sample(frequency, old['field'][::-1, 7, node], grid)
            legacy_values = dict(chi=log_sample(grid, chi[at], nu),
                true_eta=log_sample(grid, eta[at], nu), es=log_sample(grid, es[at], nu),
                total_eta=log_sample(grid, eta[at]+es[at]*projected_j, nu))
            original = dict(J=old_values[:, 0], chi=old_values[:, 4]/1.e10,
                true_eta=old_values[:, 5]/1.e10, es=old_values[:, 6]/1.e10,
                total_eta=(old_values[:, 5]+old_values[:, 6]*old_values[:, 7])/1.e10)
            direct_values = dict(J=new_values[:, 0], chi=new_values[:, 4]/1.e10,
                true_eta=new_values[:, 5]/1.e10, es=new_values[:, 6]/1.e10,
                total_eta=(new_values[:, 5]+new_values[:, 6]*new_values[:, 7])/1.e10)
            comparison[str(node)] = dict(
                direct_over_original={k: contrast(direct_values[k], original[k]) for k in original},
                legacy_over_original={k: contrast(v, original[k]) for k,v in legacy_values.items()},
                legacy_over_direct={k: contrast(v, direct_values[k]) for k,v in legacy_values.items()})
            for label, values in (('original',original),('direct',direct_values),('legacy',legacy_values)):
                for name, value in values.items(): arrays[f'{label}_{name}_node{node}'] = value
        photo = {label: float(np.dot(weight, arrays[label+'_J_node49'])) for label in ('original','direct')}
        photo['direct_over_original'] = photo['direct']/photo['original']
        photo['ion_density_cm3'] = density
        photo['scope'] = '같은 native49 인구의600–750Å 부분 흡수 흐름; 전주파수 Γ 아님'
        samples = []
        for wave in (611.125, 673.572, 714.9525535):
            i = int(np.argmin(abs(2.99792458e18/nu-wave)))
            row = dict(wavelength_A=float(2.99792458e18/nu[i]), native_node_zero_based=49)
            for label in ('original','direct','legacy'):
                row[label] = {name: float(arrays[f'{label}_{name}_node49'][i])
                              for name in ('chi','true_eta','es','total_eta')}
            row['original_J'] = float(arrays['original_J_node49'][i])
            row['direct_J'] = float(arrays['direct_J_node49'][i]); samples.append(row)
    finally:
        old._mmap.close(); new._mmap.close()
    out.mkdir()
    np.savez(out/'band-contract.npz', **arrays)
    producers = [Path(__file__), Path(fields.__code__.co_filename), Path(read_auxiliary.__code__.co_filename)]
    snapshot = out/'source_snapshot'; snapshot.mkdir()
    for p in producers: shutil.copy2(p, snapshot/p.name)
    identities = {str(p): sha(p) for p in (execution_path,runtime_path,inputs_path,*producers)}
    identities.update({str(p): v for p,v in expected.items()})
    report = dict(status='COMPLETE_STORED_BAND_CONTRACT_COMPARISON', scope=__doc__,
        frequencies=len(nu), nodes_zero_based=nodes, same_material_record=70,
        comparison=comparison, native49_photo_cm3_s=photo, samples=samples,
        legacy_coefficient_interpolation='각 C 계수는 log(nu) 선형; 총eta는 원 C 격자에서 ES*projected J 조립 후 표본화',
        interpretation='legacy F/공간128 잔차는 동일 입력의 순수 수송 오차가 아님. 계수 차이가 장 차이의 전부라는 증거도 아님',
        new_transport_calls=0,new_material_updates=0,final_goal='NOT_MET',
        input_sha256=identities,output_sha256=sha(out/'band-contract.npz'),
        elapsed_seconds=time.monotonic()-started)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    with (out/'band-contract.json').open('x') as stream: stream.write(encoded)
    print(json.dumps(dict(status=report['status'], photo=photo, samples=samples,
                         elapsed_seconds=report['elapsed_seconds']), ensure_ascii=False))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('legacy-run','direct-run','weights','out'): p.add_argument('--'+key,type=Path,required=True)
    a = p.parse_args()
    compare(a.legacy_run.resolve(),a.direct_run.resolve(),a.weights.resolve(),a.out.resolve())
