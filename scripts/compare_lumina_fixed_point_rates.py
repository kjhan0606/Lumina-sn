#!/usr/bin/env python3
"""봉인 F(Jref), F(J4)를 같은 물질70의 생산 Si/S PHOT율로 비교한다."""
import argparse
import csv
import json
from pathlib import Path
import resource
import time

import numpy as np

import cmfgen_model_continuum as B
from compare_lumina_checkpoint_rates import ion_rates_stream, TARGETS
from run_lumina_cell_candidate import read_material


ROOT = Path('/gpfs/kjhan/cmfgen_runs/lumina_fixed_point_pair_20261022-jXG2f6')
OLD = Path('/gpfs/kjhan/cmfgen_runs/lumina_rate_consumer_20261021-JmUIiq/production-grid-rates')
CONFIG = Path('/gpfs/kjhan/cmfgen_runs/lumina_field0_checkpoint_20261020-EoA5Bi/input/config.json')
J4 = Path('/gpfs/kjhan/cmfgen_runs/lumina_outer_campaign_20261021-FAmNPQ/checkpoint/J-000004.npy')


def compare(out):
    started = time.monotonic()
    out = Path(out)
    if out.exists():
        raise ValueError('기존 율 결과 보존')
    execution = json.loads((ROOT/'execution.json').read_text())
    old = json.loads((OLD/'rates.json').read_text())
    c = json.loads(CONFIG.read_text())
    plan = json.loads(Path(c['plan']).read_text())
    source = json.loads((Path(c['reuse_sources']['cell_sources']['directory'])/'sources.json').read_text())
    if (execution['status'] != 'PAIR_DIAGNOSTIC_COMPLETE' or
            execution['completed_F_calls'] != 2 or
            execution['material_record'] != old['material_record'] != 70 or
            execution['reference_record'] != old['reference_record'] != 71 or
            old['integration_grid'] != 'production' or
            old['status'] != 'UNVERIFIED_FIELD_RATE_COMPARISON' or
            old['rates_csv_sha256'] != B.sha(OLD/'rates.csv') or
            old['captured_sha256'][str(CONFIG)] != B.sha(CONFIG) or
            plan['reference_field_sha256'] != old['reference_sha256']):
        raise ValueError('같은 물질/생산율/완료 F 짝의 신원 불일치')
    for path, identity in old['input_sha256'].items():
        if B.sha(Path(path)) != identity:
            raise ValueError('기존 생산 원자/물질 입력 변경: '+path)
    for name in ('compare_lumina_checkpoint_rates.py', 'cmfgen_model_continuum.py',
                 'cmfgen_parser.py', 'expand_atomic_data_cmfgen.py', 'cmfgen_phot_smoothing.py'):
        expected = [value for path, value in old['source_sha256'].items()
                    if Path(path).name == name or path == name]
        if len(expected) != 1 or B.sha(Path(__file__).with_name(name)) != expected[0]:
            raise ValueError('기존 생산율 함수와 다른 소스: '+name)
    paths = dict(CMFGEN_projected=ROOT/'Jref.npy', J4=J4,
                 F_CMFGEN=ROOT/'Fref.npy', F_J4=ROOT/'F4.npy')
    expected = dict(CMFGEN_projected=execution['Jref']['sha256'],
                    J4=execution['input_sha256'][str(J4)],
                    F_CMFGEN=execution['F_reference']['sha256'],
                    F_J4=execution['F_J4']['sha256'])
    arrays = {}
    try:
        for name, path in paths.items():
            if B.sha(path) != expected[name]:
                raise ValueError('복사장 원본 변경: '+name)
            arrays[name] = np.load(path, mmap_mode='r', allow_pickle=False)
        nu = np.load(c['grid'], mmap_mode='r', allow_pickle=False)
        try:
            material_path = Path(c['cells'])/'material_cells.bin'
            material = read_material(material_path, b'LUMCEL01')
            if nu.shape != (1115020,) or material['ns'] != 89 or any(
                    value.shape != (91,len(nu)) for value in arrays.values()):
                raise ValueError('원래 89셀+보조2점/전주파수 차원 변경')
            run = Path(plan['initial_material'])
            atomic = Path(c['cells'])/'atomic'
            rows, models, routes, policy, smoothing, counts = B.load_model(
                run, atomic, Path(c['phot_library']))
            if policy != source['process_selection'] or counts != source['routes_by_ion']:
                raise ValueError('생산 PHOT route/과정 선택 불일치')
            matter = B.read_material(material_path, rows, models)
            do_dis = B.dissolution_flag(run/'MODEL')
            if do_dis:
                matter = B.read_material(material_path, rows, models,
                                         B.material_occupation(rows, models, matter))
            full_check = B.consume_c_full_population(Path(c['full_population']), matter)
            def block(sl):
                return {name: value[:89,sl] for name,value in arrays.items()}
            old_rows = {}
            with (OLD/'rates.csv').open() as stream:
                for row in csv.DictReader(stream):
                    key = int(row['Z']),int(row['stage']),int(row['shell'])
                    old_rows[key] = float(row['CMFGEN_photo_cm3_s'])
            if len(old_rows) != 4*89:
                raise ValueError('기존 율의 네 이온·89셀 범위 오류')
            rows_out = []
            old_relative_max = 0.
            for (Z,stage),result in ion_rates_stream(nu,block,routes,rows,models,matter,do_dis,
                                                     targets=TARGETS):
                for shell in range(89):
                    photo = {name:float(value['photo'][shell])
                             for name,value in result['total'].items()}
                    reverse = {name:float(value['reverse'][shell])
                               for name,value in result['total'].items()}
                    baseline = old_rows[Z,stage,shell]
                    current = photo['CMFGEN_projected']
                    difference = abs(current-baseline)/(baseline if baseline else 1.)
                    old_relative_max = max(old_relative_max,difference)
                    if baseline and difference > 1e-5:
                        raise ValueError('기존 생산 참조율 재현 실패')
                    if current <= 0:
                        raise ValueError('생산 광흡수율 분모0; 비율 출판 금지')
                    rows_out.append(dict(Z=Z,stage=stage,shell=shell,routes=result['routes'],
                        photo_cm3_s=photo,reverse_cm3_s=reverse,
                        photo_over_reference={name:value/current for name,value in photo.items()},
                        old_reference_photo_cm3_s=baseline))
            if len(rows_out) != 4*89:
                raise ValueError('율 비교 행 누락')
            result = dict(schema='lumina-fixed-point-production-phot-rates-v1',
                scope='same fixed material70/full C populations/raw PHOT/Milne; no SE/RE or accepted field',
                material_record=70,reference_record=71,cells=89,frequencies=len(nu),
                targets=[list(x) for x in TARGETS],rows=rows_out,
                old_CMFGEN_rate_max_relative_difference=old_relative_max,
                do_lev_dissolution=do_dis,C_full_population_comparison=full_check,
                phot_preprocessing=smoothing,field_sha256=expected,
                source_sha256={str(Path(__file__)):B.sha(Path(__file__))},
                elapsed_seconds=time.monotonic()-started,
                peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                physical_RT_executions=0,SE_RE_solved=False,
                accepted_physical_field=False,final_goal='NOT_MET')
            out.mkdir()
            with (out/'rates.json').open('x') as stream:
                json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
                stream.write('\n')
            return result
        finally:
            nu._mmap.close()
    finally:
        for array in arrays.values():
            array._mmap.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    result = compare(args.out)
    print(json.dumps(dict(status='RATES_COMPLETED',rows=len(result['rows']),
        old_CMFGEN_rate_max_relative_difference=result['old_CMFGEN_rate_max_relative_difference'],
        elapsed_seconds=result['elapsed_seconds']),ensure_ascii=False))
