#!/usr/bin/env python3
"""셀37 native 양 끝점 반사실험의 생산 Si/S PHOT·Milne 율을 비교한다."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np


SNAPSHOT = Path('/gpfs/kjhan/cmfgen_runs/lumina_fixed_point_rates_20261022-2mOuiA/scripts')
sys.path.insert(0, str(SNAPSHOT))
import cmfgen_model_continuum as B
from compare_lumina_checkpoint_rates import ion_rates_stream, TARGETS
from run_lumina_cell_candidate import read_material


PRIOR = Path('/gpfs/kjhan/cmfgen_runs/lumina_fixed_point_rates_20261022-2mOuiA/result/rates.json')
PAIR = Path('/gpfs/kjhan/cmfgen_runs/lumina_fixed_point_pair_20261022-jXG2f6')
OLD = Path('/gpfs/kjhan/cmfgen_runs/lumina_rate_consumer_20261021-JmUIiq/production-grid-rates')
CONFIG = Path('/gpfs/kjhan/cmfgen_runs/lumina_field0_checkpoint_20261020-EoA5Bi/input/config.json')
EXPECTED_PRIOR_SHA = '5aacafe6d7415e4bcf26e133572471881370207bb47aad8e38ed68f899a979aa'


def compare(transport_root, out):
    started = time.monotonic()
    transport_root, out = Path(transport_root), Path(out)
    if out.exists():
        raise ValueError('기존 율 결과 보존')
    if B.sha(PRIOR) != EXPECTED_PRIOR_SHA:
        raise ValueError('기존 생산 Si/S율 신원 변경')
    previous = json.loads(PRIOR.read_text())
    execution = json.loads((transport_root / 'execution.json').read_text())
    pair = json.loads((PAIR / 'execution.json').read_text())
    old = json.loads((OLD / 'rates.json').read_text())
    config = json.loads(CONFIG.read_text())
    plan = json.loads(Path(config['plan']).read_text())
    source = json.loads((Path(config['reuse_sources']['cell_sources']['directory']) / 'sources.json').read_text())
    if (execution['status'] != 'COUNTERFACTUAL_DIAGNOSTIC_COMPLETE' or
            execution['F_calls_completed'] != 2 or
            pair['status'] != 'PAIR_DIAGNOSTIC_COMPLETE' or
            previous['material_record'] != old['material_record'] != 70 or
            previous['reference_record'] != old['reference_record'] != 71 or
            previous['field_sha256']['CMFGEN_projected'] != pair['Jref']['sha256'] or
            previous['field_sha256']['F_CMFGEN'] != pair['F_reference']['sha256'] or
            execution['input_sha256'][str(PAIR / 'Jref.npy')] != pair['Jref']['sha256'] or
            execution['input_sha256'][str(PAIR / 'Fref.npy')] != pair['F_reference']['sha256'] or
            old['integration_grid'] != 'production' or
            old['rates_csv_sha256'] != B.sha(OLD / 'rates.csv') or
            old['captured_sha256'][str(CONFIG)] != B.sha(CONFIG) or
            plan['reference_field_sha256'] != old['reference_sha256']):
        raise ValueError('반사실험/기존 생산 기준의 물질·장 신원 불일치')
    for path, identity in old['input_sha256'].items():
        if B.sha(Path(path)) != identity:
            raise ValueError('기존 원자·물질 입력 변경: ' + path)
    for name in ('compare_lumina_checkpoint_rates.py', 'cmfgen_model_continuum.py',
                 'cmfgen_parser.py', 'expand_atomic_data_cmfgen.py', 'cmfgen_phot_smoothing.py'):
        expected = [value for path, value in old['source_sha256'].items()
                    if Path(path).name == name or path == name]
        if len(expected) != 1 or B.sha(SNAPSHOT / name) != expected[0]:
            raise ValueError('기존 생산 PHOT 소비자와 다른 소스: ' + name)
    fields = {}
    try:
        for label in ('outer', 'inner'):
            item = execution['endpoint_results'][label]['field']
            path = transport_root / f'F-{label}.npy'
            if item['path'] != str(path) or B.sha(path) != item['sha256']:
                raise ValueError('끝점 반사실험 F 변경: ' + label)
            fields[label] = np.load(path, mmap_mode='r', allow_pickle=False)
        nu = np.load(config['grid'], mmap_mode='r', allow_pickle=False)
        try:
            material_path = Path(config['cells']) / 'material_cells.bin'
            material = read_material(material_path, b'LUMCEL01')
            if (nu.shape != (1115020,) or material['ns'] != 89 or
                    any(value.shape != (91, len(nu)) for value in fields.values())):
                raise ValueError('같은 89셀/전체ν 범위가 아님')
            run = Path(plan['initial_material'])
            atomic = Path(config['cells']) / 'atomic'
            rows, models, routes, policy, smoothing, counts = B.load_model(
                run, atomic, Path(config['phot_library']))
            if policy != source['process_selection'] or counts != source['routes_by_ion']:
                raise ValueError('생산 PHOT 과정·route 불일치')
            matter = B.read_material(material_path, rows, models)
            do_dis = B.dissolution_flag(run / 'MODEL')
            if do_dis:
                matter = B.read_material(material_path, rows, models,
                                         B.material_occupation(rows, models, matter))
            full_check = B.consume_c_full_population(Path(config['full_population']), matter)
            baseline = {(int(row['Z']), int(row['stage']), int(row['shell'])): row
                        for row in previous['rows']}
            if len(baseline) != 4 * 89:
                raise ValueError('기존 네 이온·89셀 생산율 누락')

            def block(sl):
                return {name: value[:89, sl] for name, value in fields.items()}

            output_rows = []
            for (z, stage), result in ion_rates_stream(nu, block, routes, rows, models,
                                                        matter, do_dis, targets=TARGETS):
                for shell in range(89):
                    old_row = baseline[z, stage, shell]
                    reference = old_row['photo_cm3_s']['CMFGEN_projected']
                    p0 = old_row['photo_cm3_s']['F_CMFGEN']
                    if reference <= 0 or p0 <= 0 or result['routes'] != old_row['routes']:
                        raise ValueError('기존 생산율 분모/route 불일치')
                    photo = {name: float(value['photo'][shell])
                             for name, value in result['total'].items()}
                    reverse = {name: float(value['reverse'][shell])
                               for name, value in result['total'].items()}
                    output_rows.append(dict(Z=z, stage=stage, shell=shell,
                        routes=result['routes'], photo_cm3_s=photo, reverse_cm3_s=reverse,
                        photo_over_CMFGEN={name: value / reference for name, value in photo.items()},
                        photo_over_P0_Fref={name: value / p0 for name, value in photo.items()},
                        baseline_CMFGEN_photo_cm3_s=reference,
                        baseline_P0_Fref_photo_cm3_s=p0))
            if len(output_rows) != 4 * 89:
                raise ValueError('출력 이온·셀 수 부족')
            output = dict(schema='lumina-shell37-endpoint-production-rates-v1',
                scope='Fixed material70, full C population and raw PHOT/Milne; endpoint counterfactuals only',
                material_record=70, reference_record=71, cells=89, frequencies=len(nu),
                targets=[list(x) for x in TARGETS], rows=output_rows,
                do_lev_dissolution=do_dis, C_full_population_comparison=full_check,
                phot_preprocessing=smoothing,
                field_sha256={name: execution['endpoint_results'][name]['field']['sha256']
                              for name in fields},
                input_sha256={str(path): B.sha(path) for path in
                              (PRIOR, transport_root / 'execution.json', CONFIG)},
                source_sha256={str(Path(__file__)): B.sha(Path(__file__))},
                elapsed_seconds=time.monotonic() - started,
                peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                physical_RT_executions=0, SE_RE_solved=False,
                accepted_physical_field=False, final_goal='NOT_MET')
            out.mkdir()
            (out / 'rates.json').write_text(json.dumps(output, ensure_ascii=False,
                                                    indent=2, allow_nan=False) + '\n')
            return output
        finally:
            nu._mmap.close()
    finally:
        for value in fields.values():
            value._mmap.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transport-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.transport_root, args.out)
    print(json.dumps(dict(status='RATES_COMPLETED', rows=len(result['rows']),
                          elapsed_seconds=result['elapsed_seconds'],
                          peak_rss_KiB=result['peak_rss_KiB']), ensure_ascii=False))
