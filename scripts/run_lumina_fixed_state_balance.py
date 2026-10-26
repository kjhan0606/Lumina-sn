#!/usr/bin/env python3
"""완료 전원소 후보의 동일 상태 에너지 잔차. 온도/전하/수송 해를 새로 풀지 않는다.

BF·BB의 독립 열/냉각 합을 기존 C net과 대조하고 기존 FF 결과를 소비한다.
물리 광자/내부에너지와 native 호환 SCL_LN/정적 단열을 분리한다.
외부 붕괴열은 저장 native 프로필 조건이다. 이것이 물리 순간출력의 증거는 아니다.
"""
import argparse
import csv
import ctypes as C
import json
import os
from pathlib import Path
import struct
import time

import numpy as np

import cmfgen_model_continuum as B
import cmfgen_model_sources as S
from lumina_model_energy import ModelThermalEnergy
from lumina_model_balance import balance
from lumina_native_heat_sources import vector,cooling_vector,artificial_heat
from compare_lumina_internal_energy import AdiabaticInput,AdiabaticCell,DP


RUNS=Path('/gpfs/kjhan/cmfgen_runs')
BF=RUNS/'lumina_material_bf90_20261027-B64JUq/run'
BB=RUNS/'lumina_linear_BB90_20261027-gfIdax/run'
RESPONSE=RUNS/'lumina_full_model_response_20261027-vzt2uX/run'


def load_report(root,state,sha,inputs):
    path=root/'result.json'
    if B.sha(path)!=sha:raise ValueError('봉인 완료 보고서 변경')
    r=json.loads(path.read_text())
    if r['state']!=state:raise ValueError('미완료 입력')
    inputs[str(path)]=sha
    outputs=r['output_sha256']
    if isinstance(outputs,str):outputs={'response.npz':outputs}
    for name,h in outputs.items():inputs[str(root/name)]=h
    return r


def npz(path):
    with np.load(path) as f:return {k:f[k] for k in f.files}


def adiabatic(library,radius,epoch,t,ne,nuclei,energy):
    dll=C.CDLL(str(library));function=dll.cmfgen_adiabatic_v3_homologous_evaluate
    function.argtypes=[C.POINTER(AdiabaticInput),C.POINTER(AdiabaticCell)];function.restype=C.c_int
    arrays=[np.ascontiguousarray(a[::-1],dtype='f8') for a in
            (radius,radius/epoch,t,nuclei,ne,energy)]
    v=AdiabaticInput(len(t),epoch,*[a.ctypes.data_as(DP) for a in arrays]);cells=(AdiabaticCell*len(t))()
    if function(C.byref(v),cells):raise ValueError('C 깊이 결합 단열 소비 실패')
    return {name:np.array([getattr(cell,name) for cell in cells])[::-1] for name,_ in AdiabaticCell._fields_}


def run(args):
    if os.environ.get('SLURM_CPUS_PER_TASK')!='1' or os.environ.get('SLURM_MEM_PER_NODE')!='4096':
        raise ValueError('CPU1/4GiB Slurm 후처리 전용')
    start=time.monotonic();out=args.out.resolve()
    if out.exists():raise FileExistsError('기존 에너지 잔차 결과 보존')
    inputs={}
    br=load_report(BF,'COMPLETE_FROZEN_MATERIAL_BF_90',
        '482b18c4d884984787c108575e096cd7ac18665aa5e85fda63a66f83f8d286c6',inputs)
    load_report(BB,'COMPLETE_LINEAR_FIELD_BB90',
        'b95b16e3401068bc8239116a87c872f392e970ec5780b7ce04d32cf51d1e9ec9',inputs)
    rr=load_report(RESPONSE,'COMPLETE_PRIVATE_FIXED_FIELD_ALL_MODEL_RESPONSE',
        'a93e5127286bf0b3ef0e3bf41ba39549b4877675518f2bcffb95923cde46cf68',inputs)
    ffpath=args.ff_result.resolve();fr=json.loads(ffpath.read_text())
    if (fr['state']!='COMPLETE_FROZEN_FIELD_FF90_THERMAL_COEFFICIENTS' or fr['points']!=90 or
        fr['input_sha256'].get(str(RESPONSE/'result.json'))!=inputs[str(RESPONSE/'result.json')] or
        fr['input_sha256'].get(str(RESPONSE/'response.npz'))!=rr['output_sha256']):
        raise ValueError('같은 C 후보의 완료 FF 결과 필요')
    inputs[str(ffpath)]=B.sha(ffpath)
    for name,h in fr['output_sha256'].items():inputs[str(ffpath.parent/name)]=h
    base_path=RUNS/'lumina_extended_4h_20261022-rJGMVR/gpu-local-plan.json'
    config_path=Path(json.loads(base_path.read_text())['config']);config=json.loads(config_path.read_text())
    plan_path=Path(config['plan']);plan=json.loads(plan_path.read_text());run=Path(plan['initial_material'])
    common_field=str(Path(plan['same_material_reference_field'])/'FORMAL_PRIMARY_FIELDS.bin')
    if fr['input_sha256'].get(common_field)!=plan['reference_field_sha256']:
        raise ValueError('FF의 공통 복사장 신원 불일치')
    model=run/'MODEL';levels=Path(config['cells'])/'atomic/levels.csv'
    nodes=Path(config['seed'])/'native_material_nodes.bin';rvtj=run/'RVTJ';cool=run/'GENCOOL'
    project=Path(__file__).resolve().parents[1]
    for p in (base_path,config_path,plan_path,model,levels,nodes,rvtj,cool,run/'MODEL_SPEC',
              run/'atomic_links.txt',args.adiabatic_library,Path(__file__),
              project/'scripts/lumina_native_heat_sources.py',project/'scripts/lumina_model_balance.py',
              project/'scripts/lumina_model_energy.py',project/'scripts/compare_lumina_internal_energy.py',
              project/'data/atomic/ionization_reference.csv'):
        inputs[str(p)]=B.sha(p)
    native_sources={
        RUNS.parent/'cmfgen_src/cur_cmf/new_main/subs/prevent_low_t.f':
            '2082ca5b04b6e465fdbbe9e8abdf95b012d6256c4b252003aa10790099ca6629',
        RUNS.parent/'cmfgen_src/cur_cmf/subs/wr_cooling_term.f':
            'e443d6b4625bfccfefa3df80f557f22f1fec7d43e7ed6b67f46cd6081bcb1f50'}
    for p,h in native_sources.items():
        if B.sha(p)!=h:raise ValueError('native 열원/0 생략 writer 규약 변경')
        inputs[str(p)]=h
    for p in (model,levels,nodes,run/'MODEL_SPEC',run/'atomic_links.txt'):
        if inputs[str(p)]!=br['data_sha256'][str(p)]:raise ValueError('다른 물질/원자 모형')
    if any(B.sha(Path(p))!=h for p,h in inputs.items()):raise ValueError('완료 입력/배열 변경')
    snapshot=npz(BF/'material-snapshot.npz');candidate=npz(RESPONSE/'response.npz')
    roster=npz(BF/'route-roster.npz');lines=npz(BB/'line-roster.npz')
    if any(not np.array_equal(snapshot[k],candidate[k]) for k in ('columns','temperature','ne')):
        raise ValueError('BF/SE 후보의 T/ne/원자 주소 불일치')
    pop=candidate['population'];columns=candidate['columns'];t=candidate['temperature'];ne=candidate['ne']
    if pop.shape!=(1798,90) or not candidate['completed'].all() or np.any(columns[:,1]<=0):
        raise ValueError('전원소 양이온/terminal 후보 불완결')
    rows=list(csv.DictReader(levels.open()));models=B.atomic.load_cmfgen_model_spec(run/'MODEL_SPEC',run/'atomic_links.txt')
    owner=ModelThermalEnergy.from_model(run,rows,models,columns,project/'data/atomic/ionization_reference.csv')
    for p,h in owner.input_sha256.items():
        if p in br['data_sha256'] and h!=br['data_sha256'][p]:raise ValueError('에너지 원 OSC/원자자료 변경')
    inputs.update(owner.input_sha256)
    group=owner.full_to_sl;f=snapshot['fraction'];nuclei=pop.sum(axis=0)
    with nodes.open('rb') as stream:
        tag,ns,nc,nl,epoch=struct.unpack('<8sIIId',stream.read(28))
        key=np.fromfile(stream,dtype='<i4',count=3*nc).reshape(nc,3)
        physical=np.fromfile(stream,dtype='<f8',count=4*ns).reshape(ns,4)
    if tag!=b'LUMNOD01' or (ns,nc,nl)!=(90,1798,len(rows)) or not np.array_equal(key,columns):
        raise ValueError('에너지 물질/기하 원래점 정체성 오류')
    if not np.allclose(nuclei,candidate['energy'][:,0],rtol=1.e-12,atol=0):raise ValueError('C 원자 총수 불일치')
    radius=physical[:,0]
    text=rvtj.read_text()
    if not np.allclose(vector(text,'Radius (10^10 cm)',90)*1.e10,radius,rtol=8.e-11,atol=0):
        raise ValueError('외부 열원 반경/순서 불일치')
    # 이 deck은 LOCAL+instantaneous+thermal-only다. 같은 기하/동위원소의
    # 외부 가열은 stage/ne/T 후보에 맞추어 fitting하지 않는다.
    expected={'COH_ES':'T','TRT_NON_TE':'F','INS_DEP':'T','GAMRAY_TRANS':'LOCAL',
              'INC_XRAYS':'F','INC_RAY':'F','DO_DDT':'F','INC_ADV':'F','INC_AD':'T'}
    if any(S.control(model,k)!=v for k,v in expected.items()):raise ValueError('외부/산란/단열 과정 계약 변경')
    decay=vector(text,'Heating: radioactive decay (ergs/cm^3/s)',90)
    cooling_text=cool.read_text()
    # WR_COOLING_TERM은 전벡터 SUM=0이면 모든 Depth에서 제목을 생략한다.
    shock=cooling_vector(cooling_text,'Shock Power Term [ergs/cm**3/sec]',90,omitted_sum_zero=True)
    xray=cooling_vector(cooling_text,'X-ray heating term [ergs/cm**3/sec]',90,omitted_sum_zero=True)
    if any(np.any(v<0) for v in (decay,shock,xray)) or np.any(xray!=0):raise ValueError('미지원 부호/활성 외부 열원')
    external=decay+shock+xray
    art=artificial_heat(t,ne,nuclei,S.number(model,'T_MIN')*1.e4)
    bf_terms=np.zeros((2,90),dtype=np.longdouble);bb_terms=np.zeros_like(bf_terms);native_bb=np.zeros_like(bf_terms)
    rates=np.load(BF/'total-rates.npy',mmap_mode='r')
    index={tuple(map(int,k)):i for i,k in enumerate(columns)}
    for first in range(0,len(rates),1024):
        last=min(len(rates),first+1024);lo=roster['lower'][first:last];up=roster['target'][first:last]
        upper=[]
        for target,z,stage in zip(up,roster['Z'][first:last],roster['stage'][first:last]):
            upper.append(group[target] if target>=0 else index[int(z),int(stage),-1])
        nlo=np.asarray(pop[group[lo]]*f[lo],dtype=np.longdouble);nup=np.asarray(pop[upper],dtype=np.longdouble)
        unit=np.asarray(rates[first:last],dtype=np.longdouble);edge=roster['threshold'][first:last,None]*np.longdouble('6.62607015e-27')
        probability=roster['probability'][first:last,None]
        forward=probability*nlo*(unit[:,:,3]+edge*unit[:,:,0])
        reverse=probability*nup*(unit[:,:,4]+unit[:,:,5]+edge*(unit[:,:,1]+unit[:,:,2]))
        if np.any(forward<0) or np.any(reverse<0):raise ValueError('실제 BF 광자 에너지 음수')
        bf_terms+=np.stack([forward.sum(axis=0),reverse.sum(axis=0)])
    jbar=np.load(BB/'jbar.npy',mmap_mode='r')
    for first in range(0,len(jbar),4096):
        last=min(len(jbar),first+4096);lo=lines['lower'][first:last];up=lines['upper'][first:last]
        nu=lines['nu0'][first:last];A=lines['A'][first:last];bul=A*B.C**2/(2*B.H*nu**3)
        lower=np.asarray(pop[group[lo]]*f[lo],dtype=np.longdouble)
        upper=np.asarray(pop[group[up]]*f[up],dtype=np.longdouble);J=jbar[first:last]
        forward=lower*(bul/lines['g_ratio'][first:last])[:,None]*J*(B.H*nu)[:,None]
        reverse=upper*(A[:,None]+bul[:,None]*J)*(B.H*nu)[:,None]
        factor=owner.line_scaling(lo,up,nu,nuclei,enabled=S.control(model,'SCL_LN')=='T',
            ratio_limit=S.number(model,'SCL_LN_FAC'),density_limit=S.number(model,'SCL_DEN_LIM','1e30'))
        bb_terms+=np.stack([forward.sum(axis=0),reverse.sum(axis=0)])
        native_bb+=np.stack([(forward*factor).sum(axis=0),(reverse*factor).sum(axis=0)])
    independent={}
    for name,terms,slot in (('BF',bf_terms,0),('BB',bb_terms,1)):
        actual=candidate['ledger'][:,:,slot].sum(axis=0,dtype=np.longdouble)
        error=abs(terms[0]-terms[1]-actual);scale=terms.sum(axis=0)
        if np.any(error>1.e-10*scale):raise ValueError('완료 C와 독립 '+name+' 에너지 불일치')
        independent[name]=float(np.max(np.divide(error,scale,out=np.zeros_like(error),where=scale>0)))
    with (ffpath.parent/'thermal.csv').open() as stream:ffrows=list(csv.DictReader(stream))
    if len(ffrows)!=90 or [int(r['native_node']) for r in ffrows]!=list(range(90)):
        raise ValueError('FF 전90점 주소 누락/순서 오류')
    ff={policy:np.array([[float(r['fixed_T_ne_SE_response_'+policy+'_'+kind]) for r in ffrows]
                        for kind in ('heating','cooling')]) for policy in ('all_positive_ions','native_upper_ions')}
    energies={mode:np.asarray(owner.internal_energy(pop,f,mode),dtype='f8') for mode in owner.MODES}
    py_physical=energies['physical_neutral'].copy()
    energies['physical_neutral']=candidate['energy'][:,2].copy() # 실제 C 모델의 중성 기준
    ad={mode:adiabatic(args.adiabatic_library,radius,epoch,t,ne,nuclei,u) for mode,u in energies.items()}
    result_terms={
        'physical_photon_stored_decay':balance(bf_terms,bb_terms,ff['all_positive_ions'],ad['physical_neutral']['signed_total'],external,np.zeros(90)),
        'native_compatible_stored_decay':balance(bf_terms,native_bb,ff['native_upper_ions'],ad['native_static_raw']['signed_total'],external,art)}
    if any(B.sha(Path(p))!=h for p,h in inputs.items()):raise ValueError('잔차 평가 중 원 입력/소스 변경')
    out.mkdir();arrays=dict(bf=bf_terms,bb=bb_terms,native_bb=native_bb,external=external,artificial=art,
        decay=decay,shock=shock,physical_internal_energy=energies['physical_neutral'],
        native_internal_energy=energies['native_static_raw'],physical_adiabatic=ad['physical_neutral']['signed_total'],
        native_adiabatic=ad['native_static_raw']['signed_total'],charge_residual=candidate['charge_residual'])
    for policy,v in ff.items():arrays['ff_'+policy]=v
    for mode,values in result_terms.items():
        for k,v in values.items():arrays[mode+'_'+k]=v
    np.savez(out/'balance.npz',**arrays)
    summary={mode:dict(maximum_gross_normalized=float(values['gross_normalized'].max()),
        maximum_channel_normalized=float(values['channel_normalized'].max()),
        worst_channel_normalized_node=int(np.argmax(values['channel_normalized'])),
        maximum_abs_net=float(abs(values['net']).max())) for mode,values in result_terms.items()}
    report=dict(state='COMPLETE_FIXED_STATE_REGISTERED_ENERGY_BALANCES',input_sha256=inputs,
        source_commit=args.source_commit,points=90,independent_C_photon_checks=independent,summaries=summary,
        physical_internal_energy_C_vs_raw_OSC_relative_max=float(np.max(abs(py_physical/energies['physical_neutral']-1))),
        radioactive_heating_nonzero_points=int(np.count_nonzero(decay)),artificial_heating_nonzero_points=int(np.count_nonzero(art)),
        artificial_heating_max=float(art.max()),gross_metric_is_not_completion=True,
        external_decay_policy='stored native profile only; not proof of physical instantaneous decay power',
        physical_instantaneous_decay_validated=False,
        selected_nodes=[dict(node=i,**{mode:{k:float(v[i]) for k,v in values.items()} for mode,values in result_terms.items()})
                        for i in (0,22,35,49,65,89)],
        omitted_external_zero_rule='WR_COOLING_TERM SUM=0 skips all Depth blocks; frozen actual deck, no new shock/Xray model',
        output_sha256=B.sha(out/'balance.npz'),wall_seconds=time.monotonic()-start,
        temperature_ne_updated=False,charge_RE_solved=False,native_residual_Jv_equivalence=False,
        new_BF=0,new_BB=0,new_SE=0,new_transport=0,new_native=0,material_updates=0,final_goal='NOT_MET')
    (out/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(state=report['state'],summaries=summary,seconds=report['wall_seconds'])))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ff-result',type=Path,required=True)
    p.add_argument('--adiabatic-library',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source-commit',required=True)
    run(p.parse_args())
