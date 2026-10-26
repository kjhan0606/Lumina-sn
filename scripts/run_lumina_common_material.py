#!/usr/bin/env python3
"""동적 전모형 평가기의 공통 ne 초기화와 같은 후보의 전체 깊이 에너지 잔차.

원70/J70/T70을 보존하며 전하 해가 전체 RE/native/수송/스펙트럼 동등성은 아니다.
이 평가기는 이후 T 변화의 전적분을 지원하고 깊이 AD를 독립점으로 쪼개지 않는다.
"""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import struct
import time

import numpy as np

import cmfgen_model_continuum as B
import cmfgen_model_sources as S
from compare_native_prrr_common_grid import continuum_mapping
from lumina_common_charge import solve
from lumina_model_collision_source import ModelCollisionSource
from lumina_model_freefree import ModelFreeFree
from lumina_model_material_bridge import MaterialBridge
from lumina_model_material_rates import ModelMaterialRates
from lumina_model_response import ModelResponse
from lumina_model_response_balance import ResponseBalance
from run_cmfgen_formal_primary import fields
from run_lumina_fixed_state_balance import BF,BB,RESPONSE,npz,load_report
from run_lumina_full_material_response import BASE,PARAMETERS


def run(args):
    if (os.environ.get('SLURM_CPUS_PER_TASK')!='16' or os.environ.get('SLURM_MEM_PER_NODE')!='32768' or
        os.environ.get('OMP_NUM_THREADS')!='16'):raise ValueError('CPU16/32GiB/OpenMP16 전용')
    started=time.monotonic();deadline=started+6600;out=args.out.resolve()
    if out.exists():raise FileExistsError('기존 공통 물질 후보 보존')
    # 큰 입력을 읽거나 후보를 풀기 전에 Python에서 실제 소비할 공개 심볼 확인.
    library=C.CDLL(str(args.library))
    for name in ('cmfgen_adiabatic_v3_homologous_evaluate','bf_rate_material_block_contract',
                 'bf_rate_material_pseudo_block_contract','nlte_model_material_bridge_open',
                 'nlte_model_material_bridge_prepare','nlte_model_material_bridge_finalize'):
        getattr(library,name)
    inputs={}
    br=load_report(BF,'COMPLETE_FROZEN_MATERIAL_BF_90',
        '482b18c4d884984787c108575e096cd7ac18665aa5e85fda63a66f83f8d286c6',inputs)
    load_report(BB,'COMPLETE_LINEAR_FIELD_BB90',
        'b95b16e3401068bc8239116a87c872f392e970ec5780b7ce04d32cf51d1e9ec9',inputs)
    load_report(RESPONSE,'COMPLETE_PRIVATE_FIXED_FIELD_ALL_MODEL_RESPONSE',
        'a93e5127286bf0b3ef0e3bf41ba39549b4877675518f2bcffb95923cde46cf68',inputs)
    ffroot=BF.parent.parent/'lumina_FF90_20261027-aZEAs7/run'
    fr=load_report(ffroot,'COMPLETE_FROZEN_FIELD_FF90_THERMAL_COEFFICIENTS',
        '2ceed7d323612208d0e2ac23808e8d0f9fd6ce6f5b7feff47f2ff0981a3f3b58',inputs)
    config_path=Path(json.loads(BASE.read_text())['config']);config=json.loads(config_path.read_text())
    plan_path=Path(config['plan']);plan=json.loads(plan_path.read_text());material=Path(plan['initial_material'])
    field_path=Path(plan['same_material_reference_field'])/'FORMAL_PRIMARY_FIELDS.bin'
    evaluation_path=BF.parent.parent/'lumina_fixedj_exp70_20261026-phmMnx/run/material/CONT_FREQ'
    kernel=BF.parent.parent/'lumina_full_source_tests_20261020-0sY1jL/kernels/model_sources.so'
    project=Path(__file__).resolve().parents[1];atomic=Path(config['seed'])/'atomic'
    nodes=Path(config['seed'])/'native_material_nodes.bin'
    inputs.update(br['data_sha256'])
    inputs.update(fr['input_sha256'])
    for p in (BASE,config_path,plan_path,PARAMETERS,args.library,kernel,nodes,atomic/'levels.csv',
              material/'RVTJ',material/'GENCOOL',material/'SN_HYDRO_FOR_NEXT_MODEL',material/'NUC_DECAY_DATA',
              project/'data/atomic/ionization_reference.csv',project/'data/atomic/topion_levels.csv'):
        actual=B.sha(p)
        if str(p) in inputs and inputs[str(p)]!=actual:raise ValueError('봉인 입력 충돌')
        inputs[str(p)]=actual
    if (inputs[str(field_path)]!=plan['reference_field_sha256'] or inputs[str(kernel)]!=
            '0f89ccd19f0a2a7631308b0197213d41dee4b6420c0a24cecad2527d8ddc0386'):
        raise ValueError('다른 공통 J/Gaunt')
    if any(B.sha(Path(p))!=h for p,h in inputs.items()):raise ValueError('계산 전 입력 변경')
    snapshot=npz(BF/'material-snapshot.npz');initial=npz(RESPONSE/'response.npz')
    roster=npz(BF/'route-roster.npz');lines=npz(BB/'line-roster.npz')
    rows,models,routes,policy,smoothing,_=B.load_model(material,atomic,Path(config['phot_library']))
    source_report=json.loads((Path(config['reuse_sources']['boundary_sources']['directory'])/'sources.json').read_text())
    if policy!=source_report['process_selection']:raise ValueError('전 원자 과정 계약 변경')
    rows_hash=br['data_sha256'][str(Path(config['cells'])/'atomic/levels.csv')]
    if B.sha(atomic/'levels.csv')!=rows_hash:raise ValueError('FL 주소 변경')
    original=B.read_material(nodes,rows,models)
    if any(not np.array_equal(original[k],snapshot[k]) for k in ('columns','superlevel','temperature','ne')):
        raise ValueError('원 물질 snapshot 불일치')
    if any(not np.array_equal(initial[k],snapshot[k]) for k in ('columns','temperature','ne')):
        raise ValueError('초기 C 응답 신원 불일치')
    geometry,records=fields(field_path)
    try:
        nu=np.array(records['freq'][::-1]*1.e15);J=np.array(records['field'][::-1,0,:].T,order='C')
        if not np.array_equal(geometry[0]*1.e10,original['physical'][:,0]):raise ValueError('원90점 반경 불일치')
        width=np.zeros_like(nu);width[:-1]+=.5*np.diff(nu);width[1:]+=.5*np.diff(nu)
        evaluation=continuum_mapping(evaluation_path,nu)
    finally:records._mmap.close()
    provider=ModelMaterialRates(routes,rows,models,args.library,nu,width,J,evaluation=evaluation,
        radiation_generation=70,dissolution_enabled=B.dissolution_flag(material/'MODEL'))
    if provider.field_signature!=br['field_signature']:raise ValueError('BF 공통 J signature 변경')
    reference_trial=provider.prepare(original,material_generation=1)
    for name,actual in (('fraction',reference_trial.fraction),('saha',reference_trial.saha),
                        ('occupation',reference_trial.state['occupation']),('milne_prefactor',reference_trial.milne_prefactor)):
        if not np.array_equal(actual,snapshot[name]):raise ValueError('BF 원 trial의 '+name+' 변경')
    actual_roster=dict(lower=[r.lower for r in provider.routes],target=[r.target for r in provider.routes],
        Z=[r.Z for r in provider.routes],stage=[r.stage for r in provider.routes],
        phot_grid=[r.phot_grid for r in provider.routes],probability=[r.probability for r in provider.routes],
        threshold=provider.edge,sigma_edge=provider.sigma_edge)
    if any(not np.array_equal(v,roster[k]) for k,v in actual_roster.items()):raise ValueError('PHOT 모형/σ/목표 변경')
    ff=ModelFreeFree(nu,width,J,S.SourceKernels(kernel),models,snapshot['columns'],
                     radiation_generation=70,evaluation=evaluation)
    ffcoef=npz(ffroot/'coefficients.npz')
    if not np.array_equal(ffcoef['columns'],snapshot['columns']):raise ValueError('FF 원자주소 불일치')
    out.mkdir();occupation=out/'initial-occupation.bin'
    with occupation.open('xb') as stream:
        stream.write(struct.pack('<8sII',b'LMATOCC1',len(rows),90));stream.write(snapshot['occupation'].astype('<f8').tobytes())
    balance=ResponseBalance(material,rows,models,snapshot['columns'],project/'data/atomic/ionization_reference.csv',
        original['physical'][:,0],original['epoch_seconds'],args.library,roster,lines,
        np.load(BB/'jbar.npy',mmap_mode='r'),ff,ffcoef,snapshot['superlevel'])
    inputs.update(balance.owner.input_sha256);inputs.update(balance.decay_metadata['input_sha256'])
    (out/'inputs.json').write_text(json.dumps(dict(source_commit=args.source_commit,input_sha256=inputs,
        resources='CPU16/32GiB/2h/OpenMP16/GPU0/retry0',scope=__doc__,max_charge_evaluations=8,
        temperature_update=False,new_native=0,new_transport=0),indent=2)+'\n')
    evaluations=[];latest={}
    def record(result,metrics,trial,total):
        number=len(evaluations)+1;root=out/('evaluation-'+str(number));root.mkdir()
        # 후속 에너지 진단이 실패해도 이미 생산한 후보를 잃지 않는다.
        np.savez(root/'response.npz',**result)
        (root/'SE-result.json').write_text(json.dumps(dict(**metrics,
            response_sha256=B.sha(root/'response.npz')),indent=2,allow_nan=False)+'\n')
        energy,energy_summary=balance.evaluate(result,trial,total)
        np.savez(root/'balance.npz',**energy)
        entry=dict(**metrics,**energy_summary,output_sha256={name:B.sha(root/name) for name in ('response.npz','balance.npz')})
        (root/'result.json').write_text(json.dumps(entry,indent=2,allow_nan=False)+'\n')
        evaluations.append(entry);latest.update(result=result,trial=trial,rates=total,root=root)
        print(json.dumps(dict(evaluation=number,**entry)),flush=True)
    def decision(data,decision):
        (latest['root']/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    with MaterialBridge(args.library,atomic,nodes,occupation,PARAMETERS) as bridge:
        response=ModelResponse(provider,bridge,ModelCollisionSource(material,rows),lines,balance.jbar,
            np.load(BB/'status.npy'),reference_trial,np.load(BF/'bound-rates.npy',mmap_mode='r'),roster,deadline=deadline)
        def evaluate(ne):return response.evaluate(snapshot['temperature'],ne,save=record)[0]
        columns=snapshot['columns'];lower=np.zeros(90);upper=np.zeros(90)
        for Z in response.species:
            selected=columns[:,0]==Z;n=snapshot['superlevel'][selected].sum(axis=0)
            lower+=int(columns[selected,1].min())*n;upper+=int(columns[selected,1].max())*n
        solved=solve(evaluate,snapshot['ne'],initial['charge_residual'],lower,upper,on_evaluation=decision)
        collision_identities={k:v for ion in response.ions.values() for k,v in ion.identity.items()}
    inputs.update(collision_identities)
    if any(B.sha(Path(p))!=h for p,h in inputs.items()):raise ValueError('공통 물질응답 중 원 입력 변경')
    selected=solved['payload']
    stages=[];similarity={}
    if selected is not None:
        np.savez(out/'accepted-response.npz',**selected)
        for Z in response.species:
            total=snapshot['superlevel'][columns[:,0]==Z].sum(axis=0)
            for q in sorted(set(map(int,columns[columns[:,0]==Z,1]))):
                mask=(columns[:,0]==Z)&(columns[:,1]==q)
                old=snapshot['superlevel'][mask].sum(axis=0)/total
                fixed=initial['population'][mask].sum(axis=0)/total
                new=selected['population'][mask].sum(axis=0)/total
                stages.append(dict(Z=Z,charge=q,CMFGEN70_fraction=old.tolist(),
                                   initial_fixed_ne_fraction=fixed.tolist(),candidate_fraction=new.tolist()))
                if Z in (14,16) and q in (1,2):
                    similarity['Z'+str(Z)+'_charge'+str(q)]=dict(
                        max_abs_fraction_vs_CMFGEN70=float(abs(new-old).max()),
                        worst_node=int(np.argmax(abs(new-old))),
                        max_abs_fraction_change_from_fixed_ne=float(abs(new-fixed).max()))
    report=dict(state='COMPLETE_FIXED_T_COMMON_CHARGE_INITIALIZATION' if solved['converged'] else
        'INCOMPLETE_FIXED_T_COMMON_CHARGE_INITIALIZATION',source_commit=args.source_commit,
        input_sha256=inputs,history=solved['history'],evaluations=evaluations,
        initial_charge_relative=float(np.max(abs(initial['charge_residual']/initial['ne']))),
        final_charge_relative=float(np.max(abs(solved['charge']/solved['ne']))),
        max_ne_relative_change=float(np.max(abs(solved['ne']/snapshot['ne']-1))),
        stages=stages,Si_S_similarity=similarity,
        comparison_baseline='CMFGEN material70 fixed-T input; not converged matched released-T anchor',
        accepted_output_sha256=B.sha(out/'accepted-response.npz') if selected is not None else None,
        all_elements=6,bound_ions=27,points=90,terminal=6,charge_solved=solved['converged'],
        temperature_updated=False,RE_solved=False,native_E_Jv_verified=False,new_native=0,new_transport=0,
        radioactive_policy='stored/instantaneous_scenario kept separate; no reference installed',
        final_goal='NOT_MET',wall_seconds=time.monotonic()-started)
    (out/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('state','final_charge_relative','wall_seconds')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--library',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--source-commit',required=True)
    run(p.parse_args())
