#!/usr/bin/env python3
"""보존 Jbar·인구의 IR 열 기여와 선택 native BB 순율을 대조한다. SE/수송 없음."""
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

import cmfgen_model_continuum as B
import cmfgen_model_sources as S
from compare_cmfgen_fixedj_full import sha, read_matrix, read_state, compare_heat_row
from prepare_cmfgen_fixedj_heat import read_heat_row
from expand_atomic_data_cmfgen import load_cmfgen_model_spec
from lumina_model_energy import ModelThermalEnergy
from cmfgen_ion_identity import parse_cmfgen_ion_id
from extract_cmfgen_line_net_fixture import netrate_header, vectors


def add_bb_flow(target, lower, upper, downward):
    """이온 내부 전이의 유입−유출. 같은 SL 안의 전이는 풀 변수에 작용하지 않는다."""
    if lower != upper:
        target[lower] += downward
        target[upper] -= downward


def compare_native_bb_flows(path, lines, jbar, mass, pop, fraction, columns, rows, group, excluded,
                            full_population=None, vector_sink=None):
    targets = {(14, 2), (14, 3), (16, 2), (16, 3)}
    z = np.array([int(r['atomic_number']) for r in rows])
    stage = np.array([int(r['ion_number'])+1 for r in rows])
    full = np.array([int(r['level_number'])+1 for r in rows])
    keys = {}
    for i, (lo, up) in enumerate(zip(lines['lower'], lines['upper'])):
        if (z[lo], stage[lo]) in targets and not excluded[i]:
            key = (int(z[lo]), int(stage[lo]), int(full[lo]), int(full[up]))
            if key in keys:
                raise ValueError('중복 native BB 주소')
            keys[key] = i
    shape = pop.shape
    native, fine, gross, rounding = (np.zeros(shape, dtype=np.longdouble) for _ in range(4))
    seen = set()
    same_sl = 0
    def selected_header(line):
        header = netrate_header(line)
        if header is not None:
            raw = header.transition.split('(', 1)[0]
            if raw in ('Sk2', 'SkIII', 'S2', 'SIII', 'Si2', 'SiIII'):
                return header
        return None
    for header, values in vectors(path, pop.shape[1], selected_header):
        atomic, ion, _ = parse_cmfgen_ion_id(header.transition.split('(', 1)[0])
        key = (atomic, ion, header.lower_full, header.upper_full)
        if key not in keys or key in seen:
            raise ValueError('NETRATE/선 catalogue 주소 누락·중복')
        seen.add(key)
        i = keys[key]
        lo, up = lines['lower'][i], lines['upper'][i]
        lc, uc = int(group[lo]), int(group[up])
        nu, a, ratio = (lines[k][i] for k in ('nu0', 'A', 'g_ratio'))
        if abs(nu-header.frequency_field*1.e15) > 5.01e8:
            raise ValueError('native 출력 반올림보다 큰 전이 주파수 차이')
        lower = np.asarray(pop[lc]*fraction[lo] if full_population is None else full_population[lo], dtype=np.longdouble)
        upper = np.asarray(pop[uc]*fraction[up] if full_population is None else full_population[up], dtype=np.longdouble)
        bul = a*B.C**2/(2*B.H*nu**3)
        upward = lower*(bul/ratio)*jbar[i]
        downward = upper*(a*mass[i]+bul*jbar[i])
        znet = np.asarray(values, dtype=np.longdouble)
        if not np.isfinite(znet).all():
            raise ValueError('비유한 native ZNET')
        native_down = upper*a*znet
        add_bb_flow(native, lc, uc, native_down)
        add_bb_flow(fine, lc, uc, downward-upward)
        if lc != uc:
            gross[lc] += downward+upward
            gross[uc] += downward+upward
            nonzero = znet != 0
            quantum = np.zeros_like(znet)
            quantum[nonzero] = .5*10.**(np.floor(np.log10(abs(znet[nonzero])))-6)
            uncertainty = upper*a*quantum
            rounding[lc] += uncertainty
            rounding[uc] += uncertainty
        else:
            same_sl += 1
    if seen != set(keys):
        raise ValueError('일부 Si/S native 전이를 대조하지 못함')
    result = {}
    for atomic, ion in sorted(targets):
        ids = np.flatnonzero((columns[:,0] == atomic) & (columns[:,1] == ion-1))
        if not len(ids):
            raise ValueError('대상 이온 SL 누락')
        delta = native[ids]-fine[ids]
        l1 = np.sum(abs(delta), axis=0)
        fine_l1 = np.sum(abs(fine[ids]), axis=0)
        gross_sum = gross[ids].sum(axis=0)
        relative = np.divide(l1, gross_sum, out=np.zeros_like(l1), where=gross_sum>0)
        worst = np.unravel_index(np.argmax(abs(delta)), delta.shape)
        conserved = np.abs(delta.sum(axis=0))
        if vector_sink is not None:
            vector_sink[atomic,ion]=dict(columns=columns[ids].copy(),native=native[ids].copy(),
                                        gross=gross_sum.copy())
        result[f'Z{atomic}_stage{ion}'] = dict(
            matched_lines=sum(k[:2] == (atomic, ion) for k in keys),
            delta_over_BB_gross_median=float(np.median(relative)),
            delta_over_BB_gross_max=float(relative.max()),
            max_absolute_row_delta=float(abs(delta[worst])),
            max_absolute_row_depth=int(worst[1])+1,
            max_absolute_row_superlevel=int(columns[ids[worst[0]],2])+1,
            number_conservation_relative=float(np.max(np.divide(conserved,gross_sum,
                out=np.zeros_like(conserved),where=gross_sum>0))),
            depths=[dict(depth=d+1,L1_delta=float(l1[d]),fine_net_L1=float(fine_l1[d]),
                native_print_L1_bound=float(rounding[ids,d].sum()),
                delta_over_BB_gross=float(relative[d])) for d in (0,35,51,84,89)])
    return dict(scope='공통 A·인구에 native 출력 ZNET와 fine Jbar를 대입한 BB 블록 차이; native 전체 SE 잔차 아님',
                population='native full snapshot' if full_population is not None else 'Lumina response',
                current_units='cm^-3 s^-1',matched_lines=len(seen),same_superlevel_lines=same_sl,
                by_ion=result,ionization_solved=False)


def summarize(sources, response_root, *, netrate=None, expected_netrate_sha256=None,
              native_comparison=None, native_heat=None, expected_native_heat_sha256=None,
              native_vector_sink=None):
    record = json.loads((response_root / 'result.json').read_text())
    source_record = json.loads((sources / 'sources.json').read_text())
    if record['BB_contract'] != 'SOURCE_SAMPLED_TRAPEZOID_A_TIMES_MASS_V1':
        raise ValueError('보존 BB 계약은 A×profile mass 경로여야 함')
    if source_record['output_sha256']['sources.bin'] != record['source_identity']['source_sha256']:
        raise ValueError('선 catalogue와 응답의 source 신원이 다름')
    expected = {}
    def bind(path, digest):
        if sha(path) != digest:
            raise ValueError(path.name + ' 봉인 hash 불일치')
        expected[path] = digest
        return path
    def original(name):
        entries = [(Path(p), h) for p, h in record['input_sha256'].items() if Path(p).name == name]
        if len(entries) != 1:
            raise ValueError(name + ' 입력 신원 누락/중복')
        return bind(*entries[0])
    model = original('MODEL')
    spec = original('MODEL_SPEC')
    links = original('atomic_links.txt')
    levels = original('levels.csv')
    bind(sources / 'line_catalog.npz', source_record['output_sha256']['line_catalog.npz'])
    for name in ('jbar.npy', 'line-profile-mass.npy'):
        bind(response_root / name, record['output_sha256'][name])
    selected = record['selected_response']
    evaluation = response_root / Path(selected['root']).name
    bind(evaluation / 'response.npz', selected['response_sha256'])
    bind(evaluation / 'balance.npz', selected['balance_sha256'])
    rows = list(csv.DictReader(levels.open()))
    models = load_cmfgen_model_spec(spec, links)
    with np.load(evaluation / 'response.npz') as z:
        pop, fraction, columns = (z[k] for k in ('population', 'fraction', 'columns'))
    atomic_links = B.atomic.load_cmfgen_links(links)
    ions = {}
    for key, atom in models.items():
        path = Path(atomic_links[key]['osc'])
        bind(path, record['input_sha256'][str(path)])
        full = B.parse_osc(path, include_transitions=False).levels[:atom['n_full']]
        ionization = re.findall(r'^\s*(\S+)\s+!Ionization energy\b', path.read_text(), re.M)
        if len(ionization) != 1:
            raise ValueError('OSC ionization energy 누락/중복')
        edge = float(ionization[0].replace('D', 'E'))
        ions[key] = dict(edge_Hz=(edge-full['E_cm'])*B.C,
                         excitation_Hz=full['E_cm']*B.C, g=full['g'])
    # 이온 내부 BB 차이에는 중성 기준 상수가 소거된다. 물리 내부에너지는 평가하지 않는다.
    owner = ModelThermalEnergy(rows, columns, ions, {z: 0. for z, _ in models})
    with np.load(sources / 'line_catalog.npz') as z:
        lines = {k: z[k] for k in z.files}
    jbar = np.load(response_root / 'jbar.npy', mmap_mode='r')
    mass = np.load(response_root / 'line-profile-mass.npy', mmap_mode='r')
    count, nd = jbar.shape
    if count != len(lines['nu0']) or mass.shape != (count,) or fraction.shape[1] != nd:
        raise ValueError('catalogue/Jbar/인구 차원 불일치')
    domain = S.native_line_domain(model.parent)
    for path, digest in domain['source_sha256'].items():
        bind(Path(path), digest)
    excluded = ~S.native_grid_support_selection(lines['nu0'], domain)
    total = np.zeros((2, nd), dtype=np.longdouble)
    dropped = np.zeros_like(total)
    by_ion = {}
    nuclei = pop.sum(axis=0)
    group = owner.full_to_sl
    enabled = S.control(model, 'SCL_LN') == 'T'
    limit = S.number(model, 'SCL_LN_FAC')
    density_limit = S.number(model, 'SCL_DEN_LIM', '1e30')
    for first in range(0, count, 2048):
        last = min(count, first + 2048)
        lo, up = (lines[k][first:last] for k in ('lower', 'upper'))
        nu, a, ratio = (lines[k][first:last] for k in ('nu0', 'A', 'g_ratio'))
        lower = np.asarray(pop[group[lo]] * fraction[lo], dtype=np.longdouble)
        upper = np.asarray(pop[group[up]] * fraction[up], dtype=np.longdouble)
        bul = a * B.C**2 / (2 * B.H * nu**3)
        forward = lower * (bul / ratio)[:, None] * jbar[first:last] * (B.H * nu)[:, None]
        reverse = upper * (a * mass[first:last])[:, None] * (B.H * nu)[:, None]
        reverse += upper * bul[:, None] * jbar[first:last] * (B.H * nu)[:, None]
        factor = owner.line_scaling(lo, up, nu, nuclei, enabled=enabled,
                                   ratio_limit=limit, density_limit=density_limit)
        forward *= factor
        reverse *= factor
        total += np.stack([forward.sum(axis=0), reverse.sum(axis=0)])
        for offset in np.flatnonzero(excluded[first:last]):
            key = (int(rows[lo[offset]]['atomic_number']), int(rows[lo[offset]]['ion_number']) + 1)
            entry = by_ion.setdefault(key, [0, np.zeros((2, nd), dtype=np.longdouble)])
            entry[0] += 1
            values = np.stack([forward[offset], reverse[offset]])
            entry[1] += values
            dropped += values
    with np.load(evaluation / 'balance.npz') as z:
        stored = z['native_bb']
        old_net = z['native_stored_net']
    error = float(np.max(np.abs(total - stored)))
    if not np.allclose(total, stored, rtol=1.e-11, atol=1.e-13):
        raise ValueError('전체 BB 합이 보존 열장부를 재현하지 못함')
    heat_comparison=None
    if native_heat is not None:
        if expected_native_heat_sha256 is None or native_comparison is None:
            raise ValueError('native 열행과 same-state 비교 신원이 필요함')
        heat_evidence=json.loads(native_comparison.read_text())['input_sha256']
        if heat_evidence['accepted_Lumina_response']['sha256']!=selected['response_sha256']:
            raise ValueError('native 열행의 선택 응답 신원이 다름')
        row=read_heat_row(bind(native_heat,expected_native_heat_sha256))
        state=read_state(bind(native_heat.parent/'NATIVE_HEAT_INPUT.bin',
                              heat_evidence['expected_state']['sha256']))
        if (not np.array_equal(row['pops'],state['pops']) or
            not np.array_equal(row['radius_native'],state['radius'])):
            raise ValueError('포획 열행과 입력 state 불일치')
        heat_comparison=compare_heat_row(row,evaluation/'balance.npz',selected['balance_sha256'])
    native_flows = None
    if netrate is not None:
        if expected_netrate_sha256 is None:
            raise ValueError('NETRATE 기대 hash가 필요함')
        bind(netrate, expected_netrate_sha256)
        if native_comparison is None:
            raise ValueError('동일-state native comparison 신원이 필요함')
        bind(native_comparison, sha(native_comparison))
        evidence = json.loads(native_comparison.read_text())['input_sha256']
        if evidence['accepted_Lumina_response']['sha256'] != selected['response_sha256']:
            raise ValueError('native snapshot과 선택 응답 신원이 다름')
        native_full = read_matrix(bind(netrate.parent/'NATIVE_FULL_POP.bin',
                                      evidence['native_full_population']['sha256']))
        index_path = bind(netrate.parent/'NATIVE_FULL_INDEX.csv',evidence['native_index']['sha256'])
        lookup = {(int(r['atomic_number']),int(r['ion_number'])+1,int(r['level_number'])+1):i
                  for i,r in enumerate(rows)}
        aligned = np.empty_like(fraction)
        mapped = set()
        native_rows = list(csv.DictReader(index_path.open()))
        if len(native_rows) != len(rows) or native_full.shape != fraction.shape:
            raise ValueError('native full snapshot 형상 불일치')
        for i, row in enumerate(native_rows):
            atomic, ion, _ = parse_cmfgen_ion_id(row['ion_label'])
            if atomic != int(row['Z']):
                raise ValueError('native index의 원소 라벨/번호 불일치')
            target = lookup[atomic,ion,int(row['full_level'])]
            if target in mapped or int(row['superlevel']) != columns[group[target],2]+1:
                raise ValueError('native full-level 대응 중복/SL 불일치')
            aligned[target] = native_full[i]
            mapped.add(target)
        native_flows = compare_native_bb_flows(netrate, lines, jbar, mass, pop, fraction,
                                              columns, rows, group, excluded, aligned,native_vector_sink)
    for path, digest in expected.items():
        if sha(path) != digest:
            raise ValueError(path.name + ' 분석 중 변경')
    delta = np.asarray(-(dropped[0] - dropped[1]), dtype='f8')
    return dict(scope='고정-state 열 재합산; 새 SE/RE/수송/물질수용 없음',
        sign='양수=가열; delta=제외 후 열잔차−기존 열잔차',
        retained_lines=count-int(excluded.sum()), excluded_lines=int(excluded.sum()),
        grid_cutoff_Hz=domain['minimum_line_Hz_grid_guard_inclusive'],
        saved_native_bb_max_abs_reproduction_error=error,
        maximum_abs_heat_delta=float(np.max(abs(delta))),
        maximum_abs_heat_delta_depth=int(np.argmax(abs(delta)))+1,
        by_ion={f'Z{z}_stage{stage}':dict(lines=n,
            maximum_abs_net=float(np.max(abs(v[0]-v[1]))),
            net_at_depth36=float(v[0,35]-v[1,35]),
            net_at_depth85=float(v[0,84]-v[1,84])) for (z,stage),(n,v) in sorted(by_ion.items())},
        depths=[dict(depth=i+1,old_heat_residual=float(old_net[i]),heat_delta=float(delta[i]),
            frozen_state_new_heat_residual=float(old_net[i]+delta[i])) for i in (0,35,51,84,85,88,89)],
        input_sha256=[dict(name=path.name, sha256=digest) for path,digest in expected.items()],
        native_BB_flow_comparison=native_flows,
        native_heat_comparison=heat_comparison,
        final_goal='NOT_MET')


def pairwise_native_blocks(left,right):
    if set(left)!=set(right) or set(left)!={(14,2),(14,3),(16,2),(16,3)}:
        raise ValueError('native BB 블록 이온 누락')
    report={}
    for key in sorted(left):
        a,b=left[key],right[key]
        if not np.array_equal(a['columns'],b['columns']) or a['native'].shape!=b['native'].shape:
            raise ValueError('격자 사이 BB 블록 주소 불일치')
        l1=np.sum(abs(a['native']-b['native']),axis=0)
        scale=np.asarray(a['gross'])
        relative=np.divide(l1,scale,out=np.zeros_like(l1),where=scale>0)
        report[f'Z{key[0]}_stage{key[1]}']=dict(max_L1=float(l1.max()),
            max_L1_depth=int(np.argmax(l1))+1,max_over_fine_BB_gross=float(relative.max()),
            depths=[dict(depth=i+1,L1=float(l1[i])) for i in (0,35,51,84,89)])
    return report


def compare_grid_bundle(sources,response_root,bundle_root,native_comparison,*,
                        baseline_netrate_sha256,analyzer=None):
    """완료된 두 포획만 h=1과 대조한다. 미완료 파일을 분석하거나 계산을 재제출하지 않는다."""
    if analyzer is None:analyzer=summarize
    if native_comparison is None:raise ValueError('기준 same-state 비교가 필요함')
    plan_path=bundle_root/'bundle-inputs.json'
    plan_digest=sha(plan_path);comparison_digest=sha(native_comparison)
    plan=json.loads(plan_path.read_text())
    if [(a['root'],a['frac_dop']) for a in plan['attempts']]!=[('h05',.5),('h025',.25)]:
        raise ValueError('알려진 격자 묶음 순서와 다름')
    evidence=json.loads(native_comparison.read_text())
    baseline_build=evidence['input_sha256']['native_executable']['sha256']
    expected_build=plan.get('capture_build_sha256',baseline_build)
    bridge=plan.get('observer_build_bridge')
    if expected_build!=baseline_build and bridge is None:
        raise ValueError('수정 실행파일은 완료 h05 대조 근거가 필요함')
    runtime_by_root={}
    for attempt in plan['attempts']:
        path=bundle_root/attempt['root']/'native/seq_logs/fixedj-heat-runtime.json'
        if not path.is_file():
            return dict(status='WAITING_FOR_BOTH_CAPTURES',missing=attempt['root'],final_goal='NOT_MET')
        try:
            runtime=json.loads(path.read_text())
        except json.JSONDecodeError:
            return dict(status='WAITING_FOR_RUNTIME_PUBLICATION',capture=attempt['root'],final_goal='NOT_MET')
        if runtime.get('status')!='CAPTURED':
            return dict(status='CAPTURE_NOT_COMPLETE',capture=attempt['root'],
                        capture_status=runtime.get('status'),final_goal='NOT_MET')
        if (runtime.get('srun_exit_code')!=96 or runtime.get('captured_state_exact') is not True or
            runtime.get('fixed_j_sha256')!=evidence['fixed_J_source_sha256'] or
            runtime.get('accepted_response_sha256')!=evidence['input_sha256']['accepted_Lumina_response']['sha256'] or
            runtime.get('build_sha256')!=expected_build):
            raise ValueError('세분화 포획이 기준 same-state/J/실행파일과 다름')
        root=bundle_root/attempt['root']
        for name in ('VADAT','MODEL_SPEC','NATIVE_HEAT_INPUT.bin','EDDFACTOR','EDDFACTOR_INFO'):
            if sha(root/'native'/name)!=attempt['input_sha256'][name]:
                raise ValueError('세분화 포획의 초기 입력 변경: '+name)
        for name in ('NETRATE','NATIVE_HEAT_ROW.bin'):
            item=runtime['outputs'][name]
            if sha(root/'native'/name)!=item['sha256']:
                raise ValueError('세분화 포획 출력 변경: '+name)
        runtime_by_root[attempt['root']]=runtime
    bridge_report=None
    if bridge is not None:
        from run_cmfgen_fixedj_grid_bundle import (completed_bridge_reference,compare_bridge,
                                                  observer_only_build)
        reference=completed_bridge_reference(Path(bridge['root']))
        if (any(reference[k]!=bridge[k] for k in reference) or
            reference['build_sha256']!=baseline_build or
            reference['fixed_j_sha256']!=evidence['fixed_J_source_sha256'] or
            reference['accepted_response_sha256']!=evidence['input_sha256']['accepted_Lumina_response']['sha256']):
            raise ValueError('보존 h05 대조 기준과 같은-state/J/원 실행파일 신원이 다름')
        proof=observer_only_build(Path(plan['baseline'])/'build',Path(bridge['replacement_build']))
        if proof['output_sha256']['cmfgen_fixedj_heat.exe']!=expected_build:
            raise ValueError('세분화 포획이 검증한 관측 경로 제외 빌드와 다름')
        bridge_report=compare_bridge(bundle_root/'h05',bridge)
    baseline=Path(plan['baseline'])/'native'
    baseline_heat=evidence['input_sha256']['native_heat_row']['sha256']
    if not baseline_netrate_sha256 or sha(baseline/'NETRATE')!=baseline_netrate_sha256:
        raise ValueError('기준 NETRATE 봉인 hash 불일치')
    blocks=[{}]
    cases=[dict(frac_dop=1.,report=analyzer(sources,response_root,
        netrate=baseline/'NETRATE',expected_netrate_sha256=baseline_netrate_sha256,
        native_comparison=native_comparison,native_heat=baseline/'NATIVE_HEAT_ROW.bin',
        expected_native_heat_sha256=baseline_heat,native_vector_sink=blocks[0]))]
    for attempt in plan['attempts']:
        root=bundle_root/attempt['root']/'native';outputs=runtime_by_root[attempt['root']]['outputs']
        blocks.append({})
        cases.append(dict(frac_dop=attempt['frac_dop'],report=analyzer(sources,response_root,
            netrate=root/'NETRATE',expected_netrate_sha256=outputs['NETRATE']['sha256'],
            native_comparison=native_comparison,native_heat=root/'NATIVE_HEAT_ROW.bin',
            expected_native_heat_sha256=outputs['NATIVE_HEAT_ROW.bin']['sha256'],native_vector_sink=blocks[-1])))
    differences=[]
    for i in (0,1):
        heat=[]
        for case in cases[i:i+2]:
            points=case['report']['native_heat_comparison']['stored_policy_candidate']['points']
            heat.append(np.array([p['native_steq_t_cgs'] for p in points]))
        change=heat[1]-heat[0]
        differences.append(dict(from_frac_dop=cases[i]['frac_dop'],to_frac_dop=cases[i+1]['frac_dop'],
            native_heat_max_abs_change=float(np.max(abs(change))),
            native_heat_RMS_change=float(np.sqrt(np.mean(change*change))),
            native_BB_blocks=pairwise_native_blocks(blocks[i],blocks[i+1])))
    if sha(plan_path)!=plan_digest or sha(native_comparison)!=comparison_digest:
        raise ValueError('격자 비교 중 입력 계획/기준 비교 변경')
    return dict(status='SAME_STATE_GRID_COMPARISON',cases=cases,
        bundle_inputs_sha256=plan_digest,baseline_comparison_sha256=comparison_digest,
        successive_grid_differences=differences,
        observer_build_bridge=bridge_report,
        interpretation='열/BB 블록의 실제 격자 의존성만 비교; 단조 감소를 격자 수렴이나 이온화·스펙트럼 동질성으로 승격하지 않음',
        final_goal='NOT_MET')


def compare_saved_continuum_flows(response_root, native):
    """완료된 PRRR/GENCOOL만 읽는다. full-photon BF 열 일치로 해석하지 않는다."""
    from oracle_compare_cmfgen import parse_prrr, model_superlevels, parse_gencool
    from lumina_native_heat_sources import cooling_vector
    response_root, native = Path(response_root), Path(native)
    record = json.loads((response_root/'result.json').read_text())
    selected = record['selected_response']
    evaluation = response_root/Path(selected['root']).name
    runtime_path = native/'seq_logs/fixedj-heat-runtime.json'
    runtime = json.loads(runtime_path.read_text())
    if (runtime['status'] != 'CAPTURED' or not runtime['captured_state_exact'] or
            runtime['accepted_response_sha256'] != selected['response_sha256'] or
            runtime['material_update'] or runtime['new_transport']):
        raise ValueError('완료된 동일-state 고정장 포획이 아님')
    bound = {}
    def bind(path, expected=None):
        actual = sha(path)
        if expected is not None and actual != expected:
            raise ValueError(path.name+' 봉인 해시 불일치')
        bound[path] = actual
        return path
    bind(runtime_path)
    heat = read_heat_row(bind(native/'NATIVE_HEAT_ROW.bin',
                             runtime['outputs']['NATIVE_HEAT_ROW.bin']['sha256']))
    with np.load(bind(evaluation/'response.npz', selected['response_sha256'])) as data:
        columns, pop = data['columns'], data['population']
        if (not np.array_equal(data['ne'], heat['ne']) or
                not np.array_equal(data['temperature']*1.e-4, heat['temperature_native'])):
            raise ValueError('native/Lumina T/ne 동일-state 불일치')
    with np.load(bind(evaluation/'balance.npz', selected['balance_sha256'])) as data:
        balance = {key:data[key] for key in data.files}
    with bind(native/'NATIVE_FULL_INDEX.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    nsl = model_superlevels(bind(native/'MODEL_SPEC'))
    nd = pop.shape[1]
    def addresses(label):
        subset = [r for r in rows if r['ion_label'] == label]
        ids = sorted({int(r['eq_sl'])-1 for r in subset})
        if (not ids or len(ids) != nsl[label] or min(ids) < 0 or
                max(ids) >= heat['pops'].shape[0]-2):
            raise ValueError('native SL 주소/개수 오류: '+label)
        return ids
    def stats(reference, candidate):
        a, b = np.asarray(reference, dtype='f8'), np.asarray(candidate, dtype='f8')
        if (a.shape != (nd,) or b.shape != a.shape or not np.isfinite([a,b]).all()
                or np.any(a <= 0) or np.any(b <= 0)):
            raise ValueError('전체 깊이의 양의 BF 흐름이 필요함')
        ratio = a/b
        return dict(native_over_lumina=ratio.tolist(),
            minimum_ratio=float(ratio.min()), median_ratio=float(np.median(ratio)),
            maximum_ratio=float(ratio.max()), maximum_relative_difference=float(abs(ratio-1).max()),
            worst_depth=int(abs(ratio-1).argmax())+1,
            native=a.tolist(), lumina=b.tolist())
    results = {}
    for label in ('Sk2','SkIII','S2','SIII'):
        z, stage, _ = parse_cmfgen_ion_id(label)
        ids = addresses(label)
        native_lower = heat['pops'][ids].sum(axis=0)
        mask = (columns[:,0] == z) & (columns[:,1] == stage-1)
        if not np.array_equal(native_lower, pop[mask].sum(axis=0)):
            raise ValueError('선택 하위 이온 총수의 native/Lumina 불일치')
        path = bind(native/(label+'PRRR'))
        values = parse_prrr(path, label, nd, nsl[label], lower_ion_density=native_lower)
        parent = [r for r in rows if parse_cmfgen_ion_id(r['ion_label'])[:2] == (z,stage+1)
                  and int(r['full_level']) == 1]
        if len(parent) != 1:
            raise ValueError('상위 이온 바닥상태의 독립 주소 누락/중복')
        ground = heat['pops'][int(parent[0]['eq_sl'])-1]
        parent_error = float(abs(values['parent_ground_density']/ground-1).max())
        electron_error = float(abs(values['electron_density']/heat['ne']-1).max())
        if max(parent_error,electron_error) > 5.1e-5:
            raise ValueError('PRRR/native parent-ground/ne의 인쇄 정밀도 밖 불일치')
        i = np.flatnonzero((balance['bf_ion_Z_stage'] == [z,stage]).all(axis=1)).item()
        if not np.array_equal(native_lower, balance['bf_ion_number_density'][i]):
            raise ValueError('저장 BF 이온 총수 불일치')
        flows = balance['bf_ion_number_flows'][i]
        results[label] = dict(photo=stats(values['photo_flow'],flows[0]),
            recombination=stats(values['recombination_flow'],flows[1]+flows[2]),
            parent_ground_max_relative_difference=parent_error,
            electron_max_relative_difference=electron_error,
            gamma_native=values['gamma'].tolist(),
            gamma_lumina=np.asarray(flows[0]/native_lower,dtype='f8').tolist())
    path = bind(native/'GENCOOL')
    text = path.read_text()
    state_errors = {}
    for title, vector in [('Radius [1.0E+10cm]',heat['radius_native']),
                          ('Temperature [1.0E+4K]',heat['temperature_native']),
                          ('Electron Density',heat['ne'])]:
        error = float(abs(cooling_vector(text,title,nd)/vector-1).max())
        if error > 5.1e-5:
            raise ValueError('GENCOOL의 상태 불일치: '+title)
        state_errors[title] = error
    cool = parse_gencool(path,nd)
    native_ff = -cool['ff']
    ff = balance['ff_native_upper_ions']
    lumina_ff = np.asarray(ff[0]-ff[1],dtype='f8')
    delta = native_ff-lumina_ff
    for path, digest in bound.items():
        if sha(path) != digest:
            raise ValueError('판독 중 원자료 변경: '+path.name)
    return dict(status='SAME_STATE_CONTINUUM_NUMBER_FLOWS',depths=nd,
        input_sha256={str(p.relative_to(native)) if p.is_relative_to(native) else
                      'response/'+str(p.relative_to(response_root)):h for p,h in bound.items()},
        density_contract='PRRR DHYD=parent ground SL; Gamma=photo volume flow/same lower-ion total',
        bf_number_flows=results,
        ff=dict(native=native_ff.tolist(),lumina=lumina_ff.tolist(),delta=delta.tolist(),
                maximum_absolute_difference=float(abs(delta).max()),
                worst_depth=int(abs(delta).argmax())+1,
                gencool_state_max_relative_differences=state_errors),
        units=dict(number_flows='cm^-3 s^-1',gamma='s^-1',ff='erg cm^-3 s^-1'),
        interpretation='BF 수/FF 열 대조만 해당; GENCOOL BF 전자냉각을 full-photon RE와 비교하지 않음. 이온분포/열수렴/스펙트럼 동질성 미입증',
        invalid_predecessor='h025_prrr_vs_stored_BF_ion_flows.json: 잘못된 DHYD 분모의 Gamma 비교는 사용 금지',
        new_physical_evaluations=0,final_goal='NOT_MET')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path)
    parser.add_argument('--response-root', type=Path, required=True)
    parser.add_argument('--netrate', type=Path)
    parser.add_argument('--expected-netrate-sha256')
    parser.add_argument('--native-comparison', type=Path)
    parser.add_argument('--bundle-root', type=Path)
    parser.add_argument('--continuum-native-root', type=Path,
                        help='완료 PRRR/GENCOOL와 저장 응답만 비교; 새 BF 적분 없음')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.continuum_native_root and (args.bundle_root or args.netrate):
        parser.error('저장 연속 흐름 대조와 BB/격자 재생 옵션을 혼합하지 않음')
    if not args.continuum_native_root and not args.sources:
        parser.error('BB/격자 재생에는 --sources가 필요함')
    report=(compare_saved_continuum_flows(args.response_root,args.continuum_native_root)
        if args.continuum_native_root else compare_grid_bundle(args.sources,args.response_root,args.bundle_root,args.native_comparison,
                               baseline_netrate_sha256=args.expected_netrate_sha256)
        if args.bundle_root else summarize(args.sources, args.response_root,netrate=args.netrate,
        expected_netrate_sha256=args.expected_netrate_sha256,
        native_comparison=args.native_comparison))
    if args.output:
        if args.bundle_root and report['status']!='SAME_STATE_GRID_COMPARISON':
            raise SystemExit('두 포획 미완료: 결과 파일을 저장하지 않음')
        with args.output.open('x') as stream:
            json.dump(report,stream,ensure_ascii=False,indent=2)
            stream.write('\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))
