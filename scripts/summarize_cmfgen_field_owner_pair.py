#!/usr/bin/env python3
"""같은 물질59에서 두 완성 외부장의 실제 C 인구 응답을 비교한다."""
import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np

from prepare_cmfgen_mom_capture import sha


def read_response(path, log, reference, nd):
    if f'CMF_OWNER_RESPONSE_COMPLETE depths={nd}' not in log.read_text():
        raise ValueError('전90점 owner 완료 증거 없음')
    rows = list(csv.DictReader(path.open()))
    keys = [(int(r['depth_1based']), int(r['Z']), int(r['stage'])) for r in rows]
    expected = {(d, Z, s) for d in range(1, nd + 1) for Z in (14, 16) for s in range(2, 7)}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('인구 응답 행 누락/중복')
    result = dict(zip(keys, rows))
    for key, row in result.items():
        for name in ('pre_fraction', 'candidate_fraction', 'committed_fraction',
                     'Te_K', 'ne_cm3', 'n_element', 'scaled_se_residual',
                     'conservation_residual', 'kappa_2'):
            if not row[name] or not np.isfinite(float(row[name])):
                raise ValueError('owner 비유한/누락 수치')
        if row['completed'] != '1' or row['candidate_valid'] != '1':
            raise ValueError('owner 후보가 유효하지 않음')
        if not np.isclose(float(row['pre_fraction']), reference[key], rtol=2e-13, atol=1e-300):
            raise ValueError('owner의 초기 물질59 인구 불일치')
    for d in range(1, nd + 1):
        for Z in (14, 16):
            for name in ('pre_fraction', 'candidate_fraction', 'committed_fraction'):
                if abs(sum(float(result[d, Z, s][name]) for s in range(2, 7)) - 1) > 2e-12:
                    raise ValueError('이온분율 원소합 오류')
    return result


def verify_certificate_transition(current, legacy):
    if current.keys()!=legacy.keys():raise ValueError('인증 전후 coverage 변경')
    fixed=('Te_K','ne_cm3','n_element','pre_fraction','candidate_fraction',
           'scaled_se_residual','conservation_residual','kappa_2','dimension','raw_rank','graph_components')
    for key,row in current.items():
        previous=legacy[key]
        if any(row[name]!=previous[name] for name in fixed):
            raise ValueError('인증 전후 후보/기존 진단 변경: '+str(key))
        for new,old in (('legacy_topology_pass','topology_pass'),('legacy_numerical_pass','numerical_pass')):
            if row[new]!=previous[old]:raise ValueError('legacy 판정 변경')
        if row['adoption_path']!='CERTIFIED_NUMERICAL':raise ValueError('인증 경로 정체성 오류')
        if row['commit_performed']=='1':
            required=('completed','candidate_valid','topology_pass','numerical_pass','boundary_pass','certificate_pass')
            if any(row[name]!='1' for name in required) or row['certificate_status']!='0':
                raise ValueError('실패한 인증/물리 검사 뒤 채택')
            if abs(float(row['committed_fraction'])-float(row['candidate_fraction']))>1e-12:
                raise ValueError('반환 후보와 다른 인구 채택')
    cells=[r for (_,_,s),r in current.items() if s==3]
    return dict(cells=len(cells),candidate_and_original_diagnostics_byte_equal=True,legacy_flags_preserved=True,
        certified=sum(r['certificate_pass']=='1' for r in cells),
        maximum_probability_error_display=max(float(r['certificate_probability_error_display']) for r in cells),
        maximum_relative_error_display=max(float(r['certificate_relative_error_display']) for r in cells),
        warning='십진 최대오차는 표시용; 실제 판정은 C의 외향 longdouble 경계에서 수행')


def summarize(inputs, before, after, owner, out, legacy_before=None, legacy_after=None):
    if out.exists():
        raise ValueError('기존 인구 응답 요약 보존')
    manifest = json.loads((inputs / 'pair-inputs.json').read_text())
    if manifest['material_record'] != 59 or not manifest['same_T_ne_initial_population']:
        raise ValueError('물질59 짝비교 신원 불일치')
    for name, expected in manifest['output_sha256'].items():
        if sha(inputs / name) != expected:
            raise ValueError('실제 owner 입력 변경: ' + name)
    ref = {(int(r['depth_1based']), int(r['Z']), int(r['stage'])): float(r['fraction'])
           for r in csv.DictReader((inputs / 'reference59.csv').open())}
    nd = manifest['depths']
    a, b = [read_response(p / 'response.csv', p / 'owner.log', ref, nd) for p in (before, after)]
    transition={};legacy_files=[]
    if (legacy_before is None)!=(legacy_after is None):raise ValueError('legacy 짝 입력 모두 필요')
    if legacy_before is not None:
        for label,current,old_root in (('J59',a,legacy_before),('J60',b,legacy_after)):
            old=read_response(old_root/'response.csv',old_root/'owner.log',ref,nd)
            transition[label]=verify_certificate_transition(current,old)
            legacy_files.extend([old_root/'response.csv',old_root/'owner.log'])
    fixed = ('Te_K', 'ne_cm3', 'n_element', 'pre_fraction')
    if any(any(a[key][name] != b[key][name] for name in fixed) for key in a):
        raise ValueError('두 owner 풀이의 물질 상태가 다름')
    points = []; per_element = {}
    for Z in (14, 16):
        statuses = {}
        for label, rows in (('J59', a), ('J60', b)):
            group = [rows[d, Z, 3] for d in range(1, nd + 1)]
            statuses[label] = dict(committed=sum(r['commit_performed'] == '1' for r in group),
                numerical_fail=sum(r['numerical_pass'] == '0' for r in group),
                topology_fail=sum(r['topology_pass'] == '0' for r in group),
                boundary_fail=sum(r['boundary_pass'] == '0' for r in group),
                kappa_range=[min(float(r['kappa_2']) for r in group), max(float(r['kappa_2']) for r in group)],
                max_scaled_se_residual=max(float(r['scaled_se_residual']) for r in group),
                max_conservation_residual=max(float(r['conservation_residual']) for r in group))
        stages = {}
        for s in range(2, 7):
            previous = np.array([float(a[d, Z, s]['candidate_fraction']) for d in range(1, nd + 1)])
            current = np.array([float(b[d, Z, s]['candidate_fraction']) for d in range(1, nd + 1)])
            original = np.array([ref[d, Z, s] for d in range(1, nd + 1)])
            change = current - previous
            worst = int(np.argmax(abs(change)))
            stages[str(s)] = dict(all90_equal_weight=True,
                J60_minus_J59_abs_median_p95_max=np.quantile(abs(change), [.5, .95, 1]).tolist(),
                J59_vs_frozen59_abs_median_p95_max=np.quantile(abs(previous - original), [.5, .95, 1]).tolist(),
                J60_vs_frozen59_abs_median_p95_max=np.quantile(abs(current - original), [.5, .95, 1]).tolist(),
                worst_field_response_depth=worst + 1,
                frozen59_at_worst=float(original[worst]), J59_at_worst=float(previous[worst]),
                J60_at_worst=float(current[worst]))
        per_element[str(Z)] = dict(statuses=statuses, stages=stages)
    charge = []
    for d in range(1, nd + 1):
        delta = []
        for rows in (a, b):
            value = sum(float(rows[d, Z, 3]['n_element']) * sum((s - 1) *
                (float(rows[d, Z, s]['candidate_fraction']) - ref[d, Z, s]) for s in range(2, 7)) for Z in (14, 16))
            delta.append(value / float(rows[d, 14, 3]['ne_cm3']))
        charge.append(delta)
        for Z in (14, 16):
            for s in range(2, 7):
                key = (d, Z, s)
                points.append(dict(depth_1based=d, Z=Z, stage=s, frozen59=ref[key],
                    candidate_J59=float(a[key]['candidate_fraction']), candidate_J60=float(b[key]['candidate_fraction']),
                    committed_J59=float(a[key]['committed_fraction']), committed_J60=float(b[key]['committed_fraction']),
                    commit_J59=int(a[key]['commit_performed']), commit_J60=int(b[key]['commit_performed'])))
    out.mkdir()
    with (out / 'fractions.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(points[0])); writer.writeheader(); writer.writerows(points)
    source = Path(__file__).resolve(); shutil.copy2(source, out / source.name)
    files = [inputs / 'pair-inputs.json', inputs / 'reference59.csv',
             before / 'response.csv', before / 'owner.log', after / 'response.csv', after / 'owner.log', owner, source,*legacy_files]
    report = dict(scope='같은 물질59에서 실제 생산 C owner의 외부 J59/J60 조건부 응답',
        material_record=59, depths=nd, rows=len(points), per_element=per_element,
        input_pair_equal_except_radiation=True, criteria_changed=bool(transition),
        numerical_adoption_contract_changed=bool(transition),physical_criteria_changed=False,
        certified_numerical_transition=transition,
        target_SiS_charge_increment_over_frozen_ne=np.asarray(charge).tolist(),
        charge_warning='고정 T/ne의 Si/S 후보 전자수 증분이다. 실제 채택 여부는 statuses를 참조한다. 다른 원소/전하공동수렴은 포함하지 않는다.',
        CMFGEN_converged=False, population_and_T_updated_in_native_run=False,
        energy_pass='NOT_JUDGED', physical_equivalence='NOT_JUDGED', final_goal='NOT_JUDGED',
        phase_warning=manifest['field_phase_warning'],
        input_sha256={str(p): sha(p) for p in files}, fractions_sha256=sha(out / 'fractions.csv'))
    with (out / 'owner-pair.json').open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(dict(rows=len(points), per_element=per_element), ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs', 'before', 'after', 'owner', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    for name in ('legacy_before','legacy_after'):
        p.add_argument('--'+name.replace('_','-'),type=Path)
    args = p.parse_args()
    summarize(*(getattr(args, name).resolve() for name in ('inputs', 'before', 'after', 'owner', 'out')),
        legacy_before=args.legacy_before.resolve() if args.legacy_before else None,
        legacy_after=args.legacy_after.resolve() if args.legacy_after else None)
