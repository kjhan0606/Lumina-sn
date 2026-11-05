#!/usr/bin/env python3
"""완료 bound-photo 계수 재합산: 새 적분/SE/수송 없이 J와 인구 가중치 분리.

stdout JSON만 생성한다. 실행 중 보고서에서는 완료 thermal_initial_response만
사용하며, 후행 온도 섭동이나 최종 선택 상태로 재명명하지 않는다.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def split_change(g00, g01, g10, g11):
    """첫 첨자는 인구, 둘째는 장. 두 갱신 순서를 평균한 대칭 분해."""
    field = .5*((g01-g00)+(g11-g10))
    population = .5*((g10-g00)+(g11-g01))
    return field, population


def analyze(before_path, accepted_path, current_path, reference_report):
    inputs = {}
    def checked(p, expected=None):
        p = Path(p).resolve(); h = digest(p)
        if expected is not None and h != expected:
            raise ValueError('저장 자료 해시 불일치: '+p.name)
        inputs[str(p)] = h
        return p
    def report(p):
        p = Path(p).resolve()
        raw = p.read_bytes()
        inputs[str(p)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    paths = [Path(p).resolve() for p in (before_path, accepted_path, current_path)]
    reports = [report(p) for p in paths]
    reference_report = Path(reference_report).resolve()
    ref = report(reference_report)
    if ref['state'] != 'COMPLETE_BOUND_PHOTO_COMPARISON_NOT_EQUIVALENCE':
        raise ValueError('완료된 과거 CMFGEN bound-photo 비교 필요')
    for p, r in zip(paths[:2], reports[:2]):
        if not r['state'].startswith('COMPLETE_'):
            raise ValueError('이전 응답 완료 필요')
    entries = [r['selected_response'] for r in reports[:2]] + [reports[2]['thermal_initial_response']]
    states = []
    for e in entries:
        p = checked(Path(e['root'])/'response.npz', e['response_sha256'])
        with np.load(p, allow_pickle=False) as a:
            states.append({k: a[k] for k in ('columns', 'population', 'fraction', 'temperature', 'ne')})
    balances=[]
    for e in entries[1:]:
        p=checked(Path(e['root'])/'balance.npz',e['balance_sha256'])
        with np.load(p,allow_pickle=False) as a:
            balances.append({k:a[k] for k in ('bf_ion_Z_stage','bf_ion_number_flows',
                'bf_ion_number_flow_channels')})
    if any(list(a['bf_ion_number_flow_channels'])!=['photo','spontaneous','stimulated','net'] for a in balances):
        raise ValueError('BF 수 흐름 채널 정의 불일치')
    levels = [Path(p) for p in reports[2]['input_sha256'] if Path(p).name == 'levels.csv']
    if len(levels) != 1: raise ValueError('full-level 주소 모호')
    level_path = checked(levels[0], reports[2]['input_sha256'][str(levels[0])])
    with level_path.open() as f: levels = list(csv.DictReader(f))
    columns = states[0]['columns']
    if any(not np.array_equal(s['columns'], columns) for s in states):
        raise ValueError('인구 원자 주소 불일치')
    if not np.array_equal(states[1]['temperature'], states[2]['temperature']):
        raise ValueError('온도 변화까지 섞인 새-J 초기 응답')
    index = {tuple(map(int,c)): i for i,c in enumerate(columns)}
    group = np.array([index[int(r['atomic_number']),int(r['ion_number']),int(r['super_level'])] for r in levels])
    rosters = []; rates = []
    for p, r in zip(paths, reports):
        for name in ('bound-rates.npy','route-roster.npz'):
            checked(p.parent/name, r.get('output_sha256',{}).get(name))
        with np.load(p.parent/'route-roster.npz',allow_pickle=False) as a:
            rosters.append({k:a[k] for k in a.files})
        rates.append(np.load(p.parent/'bound-rates.npy',mmap_mode='r',allow_pickle=False))
    roster = rosters[0]
    if any(set(q)!=set(roster) or any(not np.array_equal(q[k],v) for k,v in roster.items()) for q in rosters[1:]):
        raise ValueError('계수 경로/단면적 주소 불일치')
    if len({r['source_identity']['grid_sha256'] for r in reports}) != 1:
        raise ValueError('주파수 격자 불일치')
    # bound photo에는 T/ne/Milne pref가 들어가지 않는다. 실제 같은 J의 두
    # 저장본을 대조하여 온도 보정 전후의 재사용을 확인한다.
    if not np.array_equal(rates[0][:,:,0], rates[1][:,:,0]):
        raise ValueError('같은 이전 J의 저장 photo 계수가 다름')
    for name in ('src/bf_rate_jnu.c','scripts/lumina_model_material_rates.py','scripts/cmfgen_model_continuum.py'):
        files = [p.parent.parent/'source'/name for p in paths]
        if len({digest(checked(f)) for f in files}) != 1:
            raise ValueError('bound-photo 생산 구현 변경: '+name)
    # 원자 입력 해시를 세 생산자 사이에서 비교한다. 고정 인구 효과의 해석에
    # 필요 없는 BB/열 구현 변경은 이 조건에 섞지 않는다.
    for p,h in reports[0]['input_sha256'].items():
        if Path(p).name in ('levels.csv','MODEL_SPEC','atomic_links.txt','MODEL','cmfgen_phot_smoothing.so') or '/PHOT/' in p or Path(p).name.startswith('phot'):
            if any(r['input_sha256'].get(p) != h for r in reports[1:]):
                raise ValueError('BF 원자 입력 신원 변경: '+Path(p).name)
            checked(p,h)
    profile_path = checked(reference_report.parent/'profiles.csv',ref['profiles_sha256'])
    if ref['input_sha256'].get(str(paths[0])) != inputs[str(paths[0])]:
        raise ValueError('과거 비교의 이전 응답 신원 불일치')
    frozen_path = str(Path(entries[0]['root'])/'response.npz')
    if ref['input_sha256'].get(frozen_path) != entries[0]['response_sha256']:
        raise ValueError('과거 CMFGEN 비교의 고정 인구가 다름')
    with profile_path.open() as f:
        previous = {(int(x['Z']),int(x['stage']),int(x['point_1based'])-1):x for x in csv.DictReader(f)}
    if len(previous)!=360: raise ValueError('과거 비교의 4이온×90점 coverage 불일치')
    from prepare_lumina_outer_input import read_native
    radius=np.array([float(previous[14,2,n]['radius_cm']) for n in range(90)])
    for r in reports:
        prep_sha=r['source_identity']['preparation_sha256']
        prep_paths=[p for p,h in r['input_sha256'].items() if h==prep_sha]
        if len(prep_paths)!=1: raise ValueError('반경 준비 기록 모호')
        prep=json.loads(checked(prep_paths[0],prep_sha).read_text())
        material=checked(prep['material'],prep['input_sha256'][prep['material']])
        physical,_=read_native(material)
        if not np.array_equal(physical[:,0],radius): raise ValueError('실제 물질 반경 불일치')
    if any(len(s['temperature'])!=90 for s in states): raise ValueError('물질90점 누락')
    records = []; max_replay = 0.; max_closure = 0.
    for z,stage in ((14,2),(14,3),(16,2),(16,3)):
        ids = np.flatnonzero((roster['Z']==z)&(roster['stage']==stage))
        select = (columns[:,0]==z)&(columns[:,1]==stage-1)
        def gamma(s, table):
            density=s['population'][select].sum(axis=0)
            if np.any(density<=0): raise ValueError('이온 수밀도 비양수')
            lower=roster['lower'][ids]
            w=s['population'][group[lower]]*s['fraction'][lower]*roster['probability'][ids,None]/density
            return np.asarray(np.sum(np.asarray(w,dtype=np.longdouble)*table[ids,:,0],axis=0),dtype=float)
        frozen0=gamma(states[0],rates[0]); frozen1=gamma(states[0],rates[2])
        g00=gamma(states[1],rates[1]); g01=gamma(states[1],rates[2])
        g10=gamma(states[2],rates[1]); g11=gamma(states[2],rates[2])
        f,m=split_change(g00,g01,g10,g11)
        bf=[]
        for s,book,g in zip(states[1:],balances,(g00,g11)):
            pairs=np.flatnonzero(np.all(book['bf_ion_Z_stage']==[z,stage],axis=1))
            if len(pairs)!=1: raise ValueError('BF 이온 장부 주소 불일치')
            flows=np.asarray(book['bf_ion_number_flows'][pairs[0]],dtype=float)
            ni=s['population'][select].sum(axis=0)
            parent=s['population'][(columns[:,0]==z)&(columns[:,1]==stage)].sum(axis=0)
            if np.any(parent<=0) or np.any(flows[0]<=0): raise ValueError('BF 정규화 분모 비양수')
            total_gamma=flows[0]/ni
            pseudo=1-g/total_gamma
            if np.any(pseudo < -1.e-10): raise ValueError('bound photo가 총 BF photo보다 큼')
            bf.append(dict(gamma=total_gamma,alpha=(flows[1]+flows[2])/(s['ne']*parent),
                stimulated_share=flows[2]/(flows[1]+flows[2]),
                pseudo_share=pseudo,net_over_photo=flows[3]/flows[0]))
        max_closure=max(max_closure,float(np.max(abs((f+m)-(g11-g00))/np.maximum(abs(g00),abs(g11)))))
        for n in range(len(frozen0)):
            old=previous[z,stage,n]; saved=float(old['new_gamma']); baseline=float(old['CMFGEN_J_gamma'])
            max_replay=max(max_replay,abs(frozen0[n]-saved)/max(abs(saved),1.e-300))
            fractions=[float(s['population'][select,n].sum()/s['population'][columns[:,0]==z,n].sum()) for s in states[1:]]
            records.append(dict(node=n,Z=z,stage=stage,radius_cm=float(old['radius_cm']),
                frozen_previous_over_CMFGEN=frozen0[n]/baseline,frozen_current_over_CMFGEN=frozen1[n]/baseline,
                frozen_current_over_previous=frozen1[n]/frozen0[n],
                gamma_previous=g00[n],gamma_current=g11[n],
                field_effect_relative_to_previous=f[n]/g00[n],population_effect_relative_to_previous=m[n]/g00[n],
                total_change_relative_to_previous=(g11[n]-g00[n])/g00[n],
                fraction_previous=fractions[0],fraction_current=fractions[1],
                total_BF_gamma_current_over_previous=bf[1]['gamma'][n]/bf[0]['gamma'][n],
                effective_BF_alpha_current_over_previous=bf[1]['alpha'][n]/bf[0]['alpha'][n],
                ne_current_over_previous=states[2]['ne'][n]/states[1]['ne'][n],
                previous_pseudo_photo_share=bf[0]['pseudo_share'][n],
                current_pseudo_photo_share=bf[1]['pseudo_share'][n],
                previous_stimulated_RR_share=bf[0]['stimulated_share'][n],
                current_stimulated_RR_share=bf[1]['stimulated_share'][n],
                previous_BF_net_over_photo=bf[0]['net_over_photo'][n],
                current_BF_net_over_photo=bf[1]['net_over_photo'][n]))
    if max_replay>1.e-12 or max_closure>1.e-12:
        raise ValueError('기존 비교 재현 또는 대칭 분해 합 불일치')
    for p,h in inputs.items():
        # 실행 중 result는 합법적으로 갱신된다. 실제 사용한 완료 배열은 불변이어야 한다.
        if Path(p)!=paths[2] and digest(p)!=h:
            raise ValueError('판독 중 완료 입력 변경: '+Path(p).name)
    summaries=[]
    for z,stage in ((14,2),(14,3),(16,2),(16,3)):
        for label,lo,hi in (('all90',0,90),('transition30_42',30,43),('inner80_89',80,90)):
            v=[x for x in records if x['Z']==z and x['stage']==stage and lo<=x['node']<hi]
            keys=('frozen_previous_over_CMFGEN','frozen_current_over_CMFGEN','frozen_current_over_previous',
                  'field_effect_relative_to_previous','population_effect_relative_to_previous','total_change_relative_to_previous',
                  'total_BF_gamma_current_over_previous','effective_BF_alpha_current_over_previous',
                  'current_pseudo_photo_share','current_stimulated_RR_share','current_BF_net_over_photo')
            summaries.append(dict(Z=z,stage=stage,region=label,points=len(v),
                **{k:dict(median=float(np.median([x[k] for x in v])),min=min(x[k] for x in v),max=max(x[k] for x in v)) for k in keys}))
    return dict(scope='STORED_BOUND_PHOTO_ONLY_NOT_TOTAL_RATE_OR_SE_CAUSALITY',
        current_response_role='thermal_initial_response_not_final_selected',
        reference='CMFGEN70/J71 fixed-T nonconverged diagnostic; frozen pre-refinement population',
        alpha_scope='stored BF (spontaneous+stimulated)/(ne*parent ion density); includes population response, not independent equilibrium prediction',
        new_integrals=0,new_SE=0,new_transport=0,new_native=0,
        report_hash_scope='live result hash is observation only; completed arrays checked separately',
        input_sha256=inputs,temperature_bit_equal=True,
        previous_photo_replay_relative_max=max_replay,decomposition_closure_relative_max=max_closure,
        summaries=summaries,records=records)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('before','accepted','current','reference-report'):p.add_argument('--'+key,required=True,type=Path)
    a=p.parse_args()
    print(json.dumps(analyze(a.before,a.accepted,a.current,a.reference_report),indent=2,ensure_ascii=False,allow_nan=False))
