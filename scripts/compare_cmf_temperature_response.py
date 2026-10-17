#!/usr/bin/env python3
"""같은 격자의 온도 표현 대조. 미수렴 한 번의 응답이며 최종 통과 판정이 아니다."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from compare_cmf_initial_iterate import material, fraction
from read_cmf_common_material import public_block, interpolate, write_csv


def parent_profiles(rows):
    """자체 자식 셀의 실제 이온 핵수를 합산한다. 공개 셀 평균은 아니다."""
    groups = {}
    for row in rows:
        groups.setdefault((row['parent'], row['Z'], row['stage']), []).append(row)
    result = []
    for (parent, z, stage), group in sorted(groups.items()):
        weight = math.fsum(r['nuclei_shell'] for r in group)
        entry = dict(parent=parent, Z=z, stage=stage, child_shells=len(group),
            velocity_inner_km_s=min(r['velocity_inner_km_s'] for r in group),
            velocity_outer_km_s=max(r['velocity_outer_km_s'] for r in group),
            nuclei_shell=weight)
        for mode in ('constant', 'reconstructed', 'reference'):
            key = mode + '_fraction'
            # 미포함 참조를0으로 채우거나 남은 셀만 재정규화하지 않는다.
            available = [r for r in group if r['nuclei_shell'] > 0]
            entry[key] = (math.fsum(r[key]*r['nuclei_shell'] for r in available)/weight
                if weight and all(r[key] is not None for r in available) else None)
        entry['fraction_change'] = (entry['reconstructed_fraction']-entry['constant_fraction']
            if entry['constant_fraction'] is not None and entry['reconstructed_fraction'] is not None else None)
        result.append(entry)
    return result


def compare(constant, reconstructed, public, out):
    runs = [constant, reconstructed]
    tables, predictors, paths, fields = [], [], [], []
    for mode, root in zip(('constant', 'reconstructed'), runs):
        log = root / 'test.log'
        if f'TEMPERATURE_RESPONSE_DONE mode={mode} ' not in log.read_text():
            raise ValueError('두 실제 공동 갱신의 완료가 필요함')
        _, before = material(root/'work/mapped_predictor.csv', 'DIAGNOSTIC_REMAPPED_GUESS;')
        _, first = material(root/'work/response_material_0.csv', 'DIAGNOSTIC_INITIAL_ITERATE;')
        _, after = material(root/'work/response_material_1.csv', 'DIAGNOSTIC_INITIAL_ITERATE;')
        if before != first or before.keys() != after.keys():
            raise ValueError('첫 장 계산의 물질 보존 또는 레이아웃 불일치')
        for key, row in before.items():
            if any(row[k] != after[key][k] for k in ('r_inner_cm', 'r_outer_cm', 'rho_g_cm3', 'T_K', 'mass_fraction')):
                raise ValueError('갱신에서 고정 물질 조건이 변함')
        with (root/'work/response_field_1.csv').open() as f:
            field = {int(r['shell']): r for r in csv.DictReader(f)}
        if set(field) != set(range(132)) or {s for s, z, q in after} != set(field):
            raise ValueError('132셸 누락')
        if {int(r['parent']) for r in field.values()} != set(range(113)):
            raise ValueError('원래113개 부모 셸 매핑 불일치')
        tables.append(after);predictors.append(before);fields.append(field)
        paths.extend([log, root/'source.sha256', root/'env_lumina_effective.txt', root/'output.sha256',
            root/'work/mapped_predictor.csv', root/'work/response_material_0.csv',
            root/'work/response_material_1.csv', root/'work/response_field_1.csv', root/'work/gamma.csv'])
    if (runs[0]/'work/gamma.csv').read_bytes() != (runs[1]/'work/gamma.csv').read_bytes():
        raise ValueError('두 대조의 감마 침적이 다름')
    if tables[0].keys() != tables[1].keys():
        raise ValueError('두 대조의 레이아웃 불일치')
    for key, row in predictors[0].items():
        if any(row[k] != predictors[1][key][k] for k in row if k != 'T_K'):
            raise ValueError('두 초기 추정에서 T 외의 입력이 다름')
    charge_error = [0., 0.]
    for branch, table in enumerate(tables):
        for s in range(132):
            selected = [r for (shell, z, q), r in table.items() if shell == s]
            charge = math.fsum(int(r['stage'])*float(r['n_ion_cm3']) for r in selected)
            density = float(selected[0]['n_e_cm3'])
            if not charge > 0 or any(float(r['n_e_cm3']) != density for r in selected):
                raise ValueError('전하/전자밀도 행 불일치')
            charge_error[branch] = max(charge_error[branch], abs(density/charge-1))
        if charge_error[branch] > 1e-8:
            raise ValueError('기존 전하 폐합 기준 미달')
    rows, metrics, missing = [], [], {}
    for z, name, mass in ((14, 'si', 28.0855), (16, 's', 32.065)):
        path = public / f'ionfrac_{name}_toy06_cmfgen.txt';paths.append(path)
        header, reference = public_block(path, 2.)
        columns = {int(k[len(name):]): i for i, k in enumerate(header) if k.startswith(name)}
        missing[str(z)] = sorted(set(columns)-{q for s, element, q in tables[0] if element == z})
        for (s, element, q), a in tables[0].items():
            if element != z:
                continue
            b = tables[1][s,z,q]
            fa, weight = fraction(a, mass);fb, wb = fraction(b, mass)
            if weight != wb or fields[0][s]['parent'] != fields[1][s]['parent']:
                raise ValueError('두 대조의 원소 핵수/부모 매핑 불일치')
            velocity = (float(a['r_inner_cm'])+float(a['r_outer_cm']))/2/172800/1e5
            ref = interpolate(reference, velocity, columns[q]) if q in columns else None
            rows.append(dict(shell=s,parent=int(fields[0][s]['parent']),Z=z,stage=q,velocity_km_s=velocity,
                velocity_inner_km_s=float(a['r_inner_cm'])/172800/1e5,
                velocity_outer_km_s=float(a['r_outer_cm'])/172800/1e5,
                constant_fraction=fa,reconstructed_fraction=fb,reference_fraction=ref,nuclei_shell=weight,
                fraction_change=fb-fa if fa is not None else None,
                constant_T_K=float(a['T_K']),reconstructed_T_K=float(b['T_K']),
                constant_ne=float(a['n_e_cm3']),reconstructed_ne=float(b['n_e_cm3']),
                J_integral_ratio=float(fields[1][s]['J_integral'])/float(fields[0][s]['J_integral'])))
        for q in sorted({r['stage'] for r in rows if r['Z']==z}):
            for region in ('all_comparable', '12100_12500'):
                group=[r for r in rows if r['Z']==z and r['stage']==q and r['reference_fraction'] is not None and
                    r['constant_fraction'] is not None and (region=='all_comparable' or 12100<r['velocity_km_s']<12500)]
                weight=math.fsum(r['nuclei_shell'] for r in group)
                entry=dict(Z=z,stage=q,region=region,shells=len(group))
                for mode in ('constant','reconstructed','reference'):
                    entry[mode+'_weighted_fraction']=math.fsum(r[mode+'_fraction']*r['nuclei_shell'] for r in group)/weight if weight else None
                    if mode!='reference':
                        entry[mode+'_weighted_absolute_error']=math.fsum(abs(r[mode+'_fraction']-r['reference_fraction'])*r['nuclei_shell'] for r in group)/weight if weight else None
                metrics.append(entry)
    paths += [Path(__file__), Path(__file__).with_name('compare_cmf_initial_iterate.py'),Path(__file__).with_name('read_cmf_common_material.py')]
    summary=dict(status='ONE_COUPLED_RESPONSE_NOT_CONVERGED',goal_verdict='NOT_JUDGED',physical_history=False,
        day=2.,full_SE_updates=1,new_radiation_each_branch=True,production_changed=False,
        reference_interpolation='linear at own centers; no extrapolation; not cell-averaged public solution',
        parent_reference='nuclei-weighted child-center interpolation; not public cell average',
        missing_stages=missing,missing_stages_are_not_zero=True,
        gamma_byte_identical=True,initial_guess_identical_except_temperature=True,
        charge_relative=charge_error,metrics=metrics,input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    out.mkdir(exist_ok=False)
    write_csv(out/'ionization.csv',rows)
    write_csv(out/'parent-ionization.csv',parent_profiles(rows))
    with (out/'summary.json').open('x') as f:
        json.dump(summary,f,indent=2,allow_nan=False);f.write('\n')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('constant','reconstructed','public','output'):
        parser.add_argument(name,type=Path)
    args=parser.parse_args()
    result=compare(args.constant,args.reconstructed,args.public,args.output)
    print(json.dumps({k:result[k] for k in ('status','charge_relative','metrics')},allow_nan=False))
