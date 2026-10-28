#!/usr/bin/env python3
"""원70·저장 J의 Si/S 지정점 입력 인구에 대한 C/native SE 작용 비교.

기존 BF90/BB90와 충돌 원자료를 재사용한다. 정상해/원자율 적분/수송은 없다.
생산 BB roster와 native 이산 BB roster가 같다고 주장하지 않는다.
"""
import argparse
import csv
import ctypes as C
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from lumina_model_collision_source import ModelCollisionSource
from lumina_model_material_bridge import MaterialBridge, DP, ptr
from ptf_gated_metrics import read_variables
from run_cmfgen_native_se_observation import read_witness
from run_lumina_full_material_response import BASE, BF_RESULT, BB_RESULT, PARAMETERS
import cmfgen_model_continuum as B
import cmfgen_model_sources as S
import lumina_native_line_quadrature as L
from run_cmfgen_formal_primary import fields


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def run(library,out,native_lines_library=None,stationary=False,element=16,node=33):
    start=time.monotonic();Z=int(element);node=int(node);out=Path(out).resolve()
    if Z not in (14,16) or not 0<=node<90:raise ValueError('비교 원소 또는 원래점 범위 오류')
    if out.exists():raise FileExistsError('기존 결과 보존')
    b=json.loads(BF_RESULT.read_text());bb=json.loads(BB_RESULT.read_text())
    baseline=Path('/gpfs/kjhan/cmfgen_runs/lumina_full_model_response_20261027-vzt2uX/run')
    native=Path('/gpfs/kjhan/cmfgen_runs/lumina_fixedj_exp70_20261026-phmMnx/run')
    base=json.loads(BASE.read_text());config=json.loads(Path(base['config']).read_text())
    plan=json.loads(Path(config['plan']).read_text());material=Path(plan['initial_material'])
    atomic=Path(config['seed'])/'atomic';nodes=Path(config['seed'])/'native_material_nodes.bin'
    evidence=json.loads((native/'seq_logs/fixedj-exponent-full-evidence.json').read_text())
    witness=native/'material/NATIVE_SE_BOUND_000071.bin'
    paths={BF_RESULT:'482b18c4d884984787c108575e096cd7ac18665aa5e85fda63a66f83f8d286c6',
      BB_RESULT:'b95b16e3401068bc8239116a87c872f392e970ec5780b7ce04d32cf51d1e9ec9',
      baseline/'response.npz':'48bdbe5b3959fc976e147741d60dab39bbe94a969232fe4cdc44956df50d59d6',
      witness:evidence['input_sha256'][str(witness)]}
    for name in ('material-snapshot.npz','route-roster.npz','total-rates.npy','routes-complete.npy'):
        paths[BF_RESULT.parent/name]=b['output_sha256'][name]
    for name in ('line-roster.npz','jbar.npy','status.npy','lines-complete.npy'):
        paths[BB_RESULT.parent/name]=bb['output_sha256'][name]
    for path,digest in paths.items():
        if sha(path)!=digest:raise ValueError('기존 입력 변경: '+path.name)
    if b['state']!='COMPLETE_FROZEN_MATERIAL_BF_90' or bb['state']!='COMPLETE_LINEAR_FIELD_BB90':
        raise ValueError('완성 BF/BB 입력 필요')
    field=Path(plan['same_material_reference_field'])/'FORMAL_PRIMARY_FIELDS.bin'
    if (b['data_sha256'].get(str(field))!=plan['reference_field_sha256'] or
            bb['input_sha256'].get(str(field))!=plan['reference_field_sha256'] or
            b['source_identity']!=base['source_identity']):
        raise ValueError('기존 BF/BB의 원70 공통장 계보 불일치')
    with np.load(BF_RESULT.parent/'material-snapshot.npz') as f:snapshot={k:f[k] for k in f.files}
    with np.load(BF_RESULT.parent/'route-roster.npz') as f:roster={k:f[k] for k in f.files}
    with np.load(BB_RESULT.parent/'line-roster.npz') as f:lines={k:f[k] for k in f.files}
    rates=np.load(BF_RESULT.parent/'total-rates.npy',mmap_mode='r')
    jbar=np.load(BB_RESULT.parent/'jbar.npy',mmap_mode='r');status=np.load(BB_RESULT.parent/'status.npy')
    if not np.load(BF_RESULT.parent/'routes-complete.npy').all() or not np.load(BB_RESULT.parent/'lines-complete.npy').all():
        raise ValueError('부분 율 입력 금지')
    head,native_values=read_witness(witness)
    if (head['parent'],head['counter'],head['nd'])!=(70,71,90):raise ValueError('다른 native 물질/시점')
    variables=read_variables(native/'material',head['nr'])
    native_indices={({'Si':14,'S':16}.get(v.species),v.charge,-1 if v.terminal else v.superlevel-1):v.index-1
                    for v in variables if v.species in ('Si','S')}
    cols=snapshot['columns'];selected=np.flatnonzero(cols[:,0]==Z)
    original=[];raw=[]
    for c in selected:
        key=tuple(map(int,cols[c]));idx=native_indices[key]
        original.append(native_values['pops'][idx,node]);raw.append(native_values['raw'][idx,node])
    np.testing.assert_allclose(snapshot['superlevel'][selected,node],original,rtol=2e-14,atol=0)
    np.testing.assert_allclose(snapshot['temperature'][node],native_values['pops'][-1,node]*1e4,rtol=3e-15,atol=0)
    np.testing.assert_array_equal(snapshot['ne'][node],native_values['pops'][-2,node])
    rows=list(csv.DictReader((atomic/'levels.csv').open()));zs=np.array([int(r['atomic_number']) for r in rows])
    ids=np.flatnonzero(zs[lines['lower']]==Z);sel={k:v[ids] for k,v in lines.items()}
    source=ModelCollisionSource(material,rows);state={k:snapshot[k] for k in ('columns','superlevel','temperature','ne','occupation')}
    nu=sel['nu0'];A=sel['A'];bul=A*B.C**2/(2*B.H*nu**3);J=jbar[ids,node]
    absorption,stimulated,spontaneous,bb_status=bul/sel['g_ratio']*J,bul*J,A,status[ids]
    native_line_info=None
    if native_lines_library is not None:
        if sha(field)!=plan['reference_field_sha256']:raise ValueError('원 복사장 변경')
        geometry,field_values=fields(field)
        try:
            frequency=np.asarray(field_values['freq'][::-1])*1e15
            field_j=np.asarray(field_values['field'][::-1,0,node])
            width=np.zeros_like(frequency);width[:-1]+=.5*np.diff(frequency);width[1:]+=.5*np.diff(frequency)
            producer=SimpleNamespace(lines=sel,profile=S.fixed_profile(material/'MODEL'),dll=C.CDLL(str(native_lines_library)))
            first,last,active,native_line_info=L.intervals(native/'material',rows,source.models,sel,len(frequency))
            values=L.contract(producer,frequency,width,field_j,first,last,active)
        finally:field_values._mmap.close()
        absorption,stimulated,spontaneous=values['absorption'],values['stimulated'],values['spontaneous_native_discrete']
        bb_status=values['status'].copy()
        if np.any((bb_status==5)&((absorption!=0)|(stimulated!=0)|(spontaneous!=0))):
            raise ValueError('native 비활성 선의 비영 잔차')
        bb_status[bb_status==5]=2
        native_line_info['library_sha256']=sha(native_lines_library)
        native_line_info['profile_mass_minmax']=[float(values['profile_mass'].min()),float(values['profile_mass'].max())]
    output=np.empty((11,len(cols)),dtype='f8');pairs=0
    with MaterialBridge(library,atomic,nodes,baseline/'occupation.bin',PARAMETERS) as bridge:
        bridge.prepare(state,snapshot['fraction'],snapshot['saha'],roster,rates,trial_id=2,radiation_generation=70)
        bridge.begin_element(Z,node,sel,absorption,stimulated,spontaneous,bb_status)
        for key in sorted(k for k in source.models if k[0]==Z):
            ion=source.load(*key)
            for lo,up,omega,_ in ion.blocks([state['temperature'][node]]):
                bridge.push(Z,node,lo,up,omega[0]);pairs+=len(lo)
            ion.verify_identity()
        probe=bridge.library.nlte_model_material_bridge_element_probe
        probe.argtypes=[C.c_void_p,C.c_uint64,DP,C.c_size_t];probe.restype=C.c_int
        output.fill(-123.)
        if probe(None,bridge.id,ptr(output),output.size)==0 or not np.all(output==-123.):
            raise ValueError('잘못된 비교 호출이 출력을 바꿈')
        if probe(bridge.elements[Z,node],bridge.id,ptr(output),output.size):raise ValueError('C 입력 연산자 평가 실패')
        bridge.free_element(bridge.elements.pop((Z,node)))
    c_raw=output[0,selected]; native_raw=np.asarray(raw)
    incoming=output[[1,3,5,7]][:,selected];outgoing=output[[2,4,6,8]][:,selected]
    combined=np.sum(incoming.astype(np.longdouble)-outgoing.astype(np.longdouble),axis=0)
    gross=np.sum(incoming.astype(np.longdouble)+outgoing.astype(np.longdouble),axis=0)
    if np.any(abs(combined-c_raw)>1e-12*gross):raise ValueError('과정별 흐름과 C 원시 잔차 불일치')
    scaled=np.divide(abs(c_raw),gross,out=np.zeros_like(c_raw),where=gross>0)
    with np.load(baseline/'response.npz') as f:
        old=float(f['ledger'][list(map(int,f['species'])).index(Z),node,5])
    if native_lines_library is None and not np.isclose(scaled.max(),old,rtol=2e-11,atol=0):raise ValueError('기존 C 입력 잔차 불일치')
    stationary_result=None
    if stationary:
        if native_lines_library is None:raise ValueError('native 선 비교에만 정상해 요청 가능')
        candidate=np.zeros_like(state['superlevel'])
        with MaterialBridge(library,atomic,nodes,baseline/'occupation.bin',PARAMETERS) as bridge:
            bridge.prepare(state,snapshot['fraction'],snapshot['saha'],roster,rates,trial_id=2,radiation_generation=70)
            bridge.begin_element(Z,node,sel,absorption,stimulated,spontaneous,bb_status)
            for key in sorted(k for k in source.models if k[0]==Z):
                ion=source.load(*key)
                for lo,up,omega,_ in ion.blocks([state['temperature'][node]]):bridge.push(Z,node,lo,up,omega[0])
                ion.verify_identity()
            candidate_ledger=bridge.finish(Z,node,candidate)
        with np.load(baseline/'response.npz') as f:
            if not np.array_equal(f['columns'],cols):raise ValueError('기존 정상해 원자 주소 불일치')
            production=f['population'][selected,node]
        raw_population=np.asarray(original);solved=candidate[selected,node]
        total=raw_population.sum()
        if not np.isfinite(solved).all() or np.any(solved<0) or not np.isclose(solved.sum(),total,rtol=1e-11):
            raise ValueError('고정장 후보 인구 보존 실패')
        stationary_result=dict(physical_scope='fixed_T_ne_J_single_element_not_charge_RE_or_transport',ledger=candidate_ledger.tolist(),
            stage_fractions=[dict(charge=int(q),native_input=float(raw_population[cols[selected,1]==q].sum()/total),
                production_BB_candidate=float(production[cols[selected,1]==q].sum()/total),
                native_BB_candidate=float(solved[cols[selected,1]==q].sum()/total)) for q in np.unique(cols[selected,1])])
    details=[]
    for j,c in enumerate(selected):
        details.append(dict(Z=Z,charge=int(cols[c,1]),superlevel=int(cols[c,2]),population=float(original[j]),
            native_raw=float(native_raw[j]),C_raw=float(c_raw[j]),gross_C=float(gross[j]),
            processes={name:dict(incoming=float(output[1+2*k,c]),outgoing=float(output[2+2*k,c]))
                for k,name in enumerate(('BF_radiative','BF_collisional','BB_radiative','collision_total','dissolution_subset'))}))
    result=dict(state='COMPLETE_SAME_INPUT_NATIVE_LINE_DIAGNOSTIC' if native_lines_library else 'COMPLETE_SAME_INPUT_PRODUCTION_ROSTER_DIAGNOSTIC',node_zero_based=node,element=Z,
        physical_equivalence='NOT_PROVEN_BF_CONTRACT_MAY_DIFFER',new_BF=0,new_BB=int(native_lines_library is not None),new_SE_solve=int(stationary),
        new_native=0,new_transport=0,rows=len(details),collision_pairs=pairs,
        native_line_info=native_line_info,
        stationary=stationary_result,
        C_max_scaled_residual=float(scaled.max()),baseline_C_max_scaled_residual=old,
        maximum_abs_row_difference=float(np.max(abs(c_raw-native_raw))),
        input_sha256={str(p):h for p,h in paths.items()},library_sha256=sha(library),
        elapsed_seconds=time.monotonic()-start,details=details)
    out.mkdir();(out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('details','input_sha256')},ensure_ascii=False))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--native-lines-library',type=Path)
    p.add_argument('--stationary',action='store_true')
    p.add_argument('--element',type=int,default=16);p.add_argument('--node',type=int,default=33)
    a=p.parse_args();run(a.library,a.out,a.native_lines_library,a.stationary,a.element,a.node)
