#!/usr/bin/env python3
"""미완 외부대기 J4를 진공 J37과 같은 CMFGEN 광자 대역에 대조한다."""
import argparse
import json
from pathlib import Path

import numpy as np

from compare_lumina_bridge_field0 import matched_shells
from run_cmfgen_formal_primary import fields
from run_lumina_cell_candidate import read_material, sha


BANDS = ((100., 400.), (400., 800.), (800., 1600.), (1600., 3000.))
C_ANGSTROM_HZ = 2.99792458e18
H_ERG_S = 6.62607015e-27


def compare(config_path, diagnostic_path, old_path, out):
    config_path, diagnostic_path, old_path, out = map(Path, (config_path, diagnostic_path, old_path, out))
    if out.exists():
        raise ValueError('기존 결과를 덮어쓰지 않는다')
    config = json.loads(config_path.read_text())
    diagnostic = json.loads(diagnostic_path.read_text())
    old = json.loads(old_path.read_text())
    plan_path = Path(config['plan'])
    plan = json.loads(plan_path.read_text())
    if (diagnostic['schema'] != 'lumina-outer-checkpoint-diagnostic-v1' or
            diagnostic['selection'] != 'current' or diagnostic['completed_iterations'] != 4 or
            diagnostic['accepted_physical_field'] is not False or
            diagnostic['captured_sha256'][str(config_path)] != sha(config_path) or
            diagnostic['captured_sha256'][str(plan_path)] != sha(plan_path) or
            config['plan_sha256'] != sha(plan_path) or
            old['config_sha256'] != sha(config_path) or
            old['plan_sha256'] != sha(plan_path) or
            plan['initial_native_record'] != 70 or
            plan['reference_radiation_material_record'] != 70 or
            plan['reference_radiation_record'] != 71):
        raise ValueError('J37/J4/참조의 봉인 세대가 다르다')
    checkpoint = diagnostic_path.parent.parent/'checkpoint'
    state_path = checkpoint/'checkpoint.json'
    state = json.loads(state_path.read_text())
    entry = diagnostic['comparison_J']
    j_path = checkpoint/entry['filename']
    if (diagnostic['captured_sha256'][str(state_path)] != sha(state_path) or
            state['J'] != entry or state['phase'] != 'ITERATION_COMPLETED' or
            state['accepted_physical_field'] is not False or
            entry['sha256'] != sha(j_path) or
            entry['sha256'] != diagnostic['radiation_sha256']['checkpoint']):
        raise ValueError('J4 저장장 신원 불일치')
    reference_path = Path(plan['same_material_reference_field'])/'FORMAL_PRIMARY_FIELDS.bin'
    if (plan['reference_field_sha256'] != old['reference_field_sha256'] or
            plan['reference_field_sha256'] != diagnostic['radiation_sha256']['cmfgen'] or
            plan['reference_field_sha256'] != sha(reference_path)):
        raise ValueError('참조 J71 신원 불일치')
    material = read_material(Path(config['cells'])/'material_cells.bin', b'LUMCEL01')
    own_nu = np.load(config['grid'], mmap_mode='r', allow_pickle=False)
    if sha(config['grid']) != config['grid_sha256']:
        raise ValueError('전체 주파수 격자 변경')
    j = np.load(j_path, mmap_mode='r', allow_pickle=False)
    if j.shape != (material['ns']+2, len(own_nu)):
        raise ValueError('보조 두 점을 포함한 J4 배열 차원 불일치')
    geometry, records = fields(reference_path)
    rows = []
    old_selected = {r['shell']: r for r in old['selected_shells']}
    maximum_reference_discrepancy = 0.
    try:
        for shell, target, candidate, reference in matched_shells(
                geometry, records, material, own_nu, j[:material['ns']]):
            wavelength = C_ANGSTROM_HZ/target
            bands = {}
            for low, high in BANDS:
                mask = (wavelength >= low) & (wavelength < high)
                if mask.sum() < 2:
                    raise ValueError('대역 주파수 표본 부족')
                scale = 4*np.pi/H_ERG_S
                own = float(np.trapezoid(scale*candidate[mask]/target[mask], target[mask]))
                cmf = float(np.trapezoid(scale*reference[mask]/target[mask], target[mask]))
                if not np.isfinite((own, cmf)).all() or own < 0 or cmf <= 0:
                    raise ValueError('광자 적분 비유한/음수/참조0')
                name = f'{low:g}-{high:g}A'
                bands[name] = dict(lumina=own, cmfgen=cmf, lumina_over_cmfgen=own/cmf)
                if shell in old_selected:
                    earlier = old_selected[shell]['bands'][name]
                    discrepancy = abs(cmf-earlier['cmfgen'])/cmf
                    maximum_reference_discrepancy = max(maximum_reference_discrepancy, discrepancy)
                    if discrepancy > 1e-12:
                        raise ValueError('J37과 J4의 참조 적분 규칙 불일치')
                    bands[name]['vacuum_J37_over_cmfgen'] = earlier['lumina_over_cmfgen']
            rows.append(dict(shell=shell, radius_cm=float(.5*sum(material['physical'][shell,:2])),
                             bands=bands))
    finally:
        records._mmap.close()
        j._mmap.close()
    if len(rows) != 89:
        raise ValueError('89셀 전체 대조가 아님')
    summary = {}
    for name in rows[0]['bands']:
        ratios = np.array([r['bands'][name]['lumina_over_cmfgen'] for r in rows])
        summary[name] = dict(min_ratio=float(ratios.min()), min_shell=int(ratios.argmin()),
                             median_ratio=float(np.median(ratios)), max_ratio=float(ratios.max()),
                             outer_ratio=float(ratios[-1]))
    result = dict(schema='lumina-outer-j4-band-contrast-v1', material_record=70,
                  reference_record=71, iterations=4, accepted_physical_field=False,
                  scope='photons only; no PHOT cross section, convergence or Si/S equivalence',
                  bands_angstrom=[list(x) for x in BANDS], cells=len(rows), summary=summary,
                  selected_shells=[rows[i] for i in (0, 15, 30, 45, 60, 75, 88)],
                  max_J37_J4_reference_relative_difference=maximum_reference_discrepancy,
                  identity=dict(config_sha256=sha(config_path), plan_sha256=sha(plan_path),
                                J4_sha256=entry['sha256'], reference_sha256=plan['reference_field_sha256'],
                                old_diagnostic_sha256=sha(old_path)),
                  final_goal='NOT_MET')
    out.mkdir()
    with (out/'diagnostic.json').open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'diagnostic', 'old', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.config, args.diagnostic, args.old, args.out),
                     ensure_ascii=False, allow_nan=False))
