#!/usr/bin/env python3
"""동적 전모형 공통 ne 초기화 또는 완료 후보에서 깊이 결합 T/ne Newton.

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
from lumina_coupled_material import packet,solve_coupled
from lumina_material_restart import common_start
import lumina_frozen_direction as frozen
from lumina_model_collision_source import ModelCollisionSource
from lumina_model_freefree import ModelFreeFree
from lumina_model_material_bridge import MaterialBridge
from lumina_model_material_rates import ModelMaterialRates
from lumina_model_response import ModelResponse
from lumina_model_response_balance import ResponseBalance
from run_cmfgen_formal_primary import fields
from run_lumina_fixed_state_balance import BF,BB,RESPONSE,npz,load_report,adiabatic
from run_lumina_full_material_response import BASE,PARAMETERS


def absent_partial_auto_inputs(material,models):
    # STEQ_AUTO_V2는 DIE 플래그와 별개로 파일 존재 시 처리한다. 없을 때만 RETURN.
    names=[m['label']+'_AUTO_DATA' for _,m in sorted(models.items())]
    for name in names:
        path=Path(material)/name
        if path.exists() or path.is_symlink():
            raise ValueError('부분 autoionization 입력을 비활성으로 간주할 수 없음: '+name)
    return names


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
    bb_report=load_report(BB,'COMPLETE_LINEAR_FIELD_BB90',
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
    coupled=bool(args.common_result)
    if coupled!=bool(args.common_result_sha) or coupled!=bool(args.thermal_policy):
        raise ValueError('공동 Newton은 완료 전하 보고서/hash/명시한 열정책이 함께 필요')
    frozen_args=(args.linearization,args.linearization_sha,args.linearization_inputs_sha,args.frozen_alpha)
    continuing=any(v is not None for v in frozen_args)
    if continuing and (not coupled or any(v is None for v in frozen_args)):
        raise ValueError('추가 감쇠는 같은 기준 후보와 방향/입력 hash/감쇠 비율이 함께 필요')
    basis,basis_identity=(frozen.load(args.linearization,args.linearization_sha,args.linearization_inputs_sha,
                         args.thermal_policy,inputs) if continuing else (None,None))
    start_candidate,start_energy,start_directory=(common_start(args.common_result,args.common_result_sha,snapshot,inputs)
        if coupled else (initial,None,None))
    roster=npz(BF/'route-roster.npz');lines=npz(BB/'line-roster.npz')
    rows,models,routes,policy,smoothing,_=B.load_model(material,atomic,Path(config['phot_library']))
    absent_auto=absent_partial_auto_inputs(material,models)
    source_report=json.loads((Path(config['reuse_sources']['boundary_sources']['directory'])/'sources.json').read_text())
    if policy!=source_report['process_selection']:raise ValueError('전 원자 과정 계약 변경')
    rows_hash=br['data_sha256'][str(Path(config['cells'])/'atomic/levels.csv')]
    if B.sha(atomic/'levels.csv')!=rows_hash:raise ValueError('FL 주소 변경')
    original=B.read_material(nodes,rows,models)
    if S.fixed_profile(material/'MODEL')!=bb_report['profile']:
        raise ValueError('T 갱신에 재사용할 수 없는 가변/변경 선 profile')
    boundary=dict(FIX_T_AUTO=S.control(material/'MODEL','FIX_T_AUTO'),
        TAU_SCL_T=S.number(material/'MODEL','TAU_SCL_T'),
        FIX_INB_T=S.control(material/'MODEL','FIX_INB_T','F'),
        FIX_X_DPTH=S.number(material/'MODEL','FIX_X_DPTH','0'),
        D2_EQ_D1=S.control(material/'MODEL','D2_EQ_D1','F'))
    if coupled and boundary!=dict(FIX_T_AUTO='F',TAU_SCL_T=0.,FIX_INB_T='F',FIX_X_DPTH=0.,D2_EQ_D1='F'):
        raise ValueError('현재90개 자유 T/ne 평가기가 지원하지 않는 경계 제약')
    if coupled:
        native=Path('/gpfs/kjhan/cmfgen_src/cur_cmf')
        boundary_sources={
            'new_main/cmfgen_sub.f':'092f8526661b1f9a5eaeb7a875f07f55623f427ad07c383f2659f2fb67143374',
            'new_main/mod_subs/rd_control_variables.f':'1f065997994a61681abc8e34426aa4f4cb95f5aaaaa8ec4518f27a9bbedc8e88',
            'new_main/mod_subs/solve_for_pops.f':'21b7398bcdb9f331639395320961dcae13d7c6c9abae560ed2060d09528357dc',
            'new_main/subs/solveba_v13.f':'8210020b3a3df9a0c1af31de26eedd73fefcdb365547a2677e21b3b83faa63e8'}
        for name,h in boundary_sources.items():
            p=native/name
            if B.sha(p)!=h:raise ValueError('기본 경계 제약의 native 근거 소스 변경')
            inputs[str(p)]=h
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
        temperature_update=coupled,thermal_policy=args.thermal_policy,max_coupled_evaluations=(1 if continuing else 5) if coupled else 0,
        reused_numerical_direction=basis_identity,frozen_alpha=args.frozen_alpha,
        absent_partial_auto_inputs=absent_auto,
        material_boundary=boundary,temperature_degrees='all90 free in private fixed-J trial; not native release' if coupled else 'fixed70',
        new_native=0,new_transport=0),indent=2)+'\n')
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
        evaluations.append(entry);latest.update(result=result,trial=trial,rates=total,root=root,energy=energy)
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
        if not coupled:
            solved=solve(evaluate,snapshot['ne'],initial['charge_residual'],lower,upper,on_evaluation=decision)
        else:
            base_packet=packet(start_candidate,start_energy,args.thermal_policy)
            packet_roots={id(base_packet):start_directory}
            def evaluate_coupled(t,ne):
                response.evaluate(t,ne,save=record)
                value=packet(latest['result'],latest['energy'],args.thermal_policy)
                packet_roots[id(value)]=latest['root']
                return value
            def coupled_decision(value,entry):
                (packet_roots[id(value)]/'phase.json').write_text(json.dumps(entry,indent=2)+'\n')
            def full_ad(t,ne,ni,u):
                return adiabatic(args.library,original['physical'][:,0],original['epoch_seconds'],t,ne,ni,u)['signed_total']
            def direction_record(base,j,f,direction,c,h):
                np.savez(out/'newton-linearization.npz',jacobian=j,scaled_residual=f,direction=direction,
                    temperature=base.temperature,ne=base.ne,charge_scale=c,thermal_scale=h)
            try:
                if continuing:
                    outer=frozen.trial(evaluate_coupled,base_packet,full_ad,basis,args.frozen_alpha,
                                       on_evaluation=coupled_decision)
                else:
                    outer=solve_coupled(evaluate_coupled,base_packet,full_ad,max_evaluations=5,
                        on_evaluation=coupled_decision,on_direction=direction_record)
            except TimeoutError as error:
                (out/'incomplete.json').write_text(json.dumps(dict(reason=str(error),final_goal='NOT_MET'))+'\n')
                raise
            accepted_root=packet_roots[id(outer['packet'])]
            selected_outer=npz(accepted_root/'response.npz')
            solved=dict(converged=outer['numerical_converged'],ne=outer['packet'].ne,
                charge=outer['packet'].charge,payload=selected_outer,history=outer['history'])
        collision_identities={k:v for ion in response.ions.values() for k,v in ion.identity.items()}
    inputs.update(collision_identities)
    absent_partial_auto_inputs(material,models)
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
    state=('COMPLETE_FIXED_T_COMMON_CHARGE_INITIALIZATION' if solved['converged'] else
        'INCOMPLETE_FIXED_T_COMMON_CHARGE_INITIALIZATION')
    if coupled:state='COMPLETE_DEPTH_COUPLED_MATERIAL_STEP' if outer['iterations'] else 'INCOMPLETE_DEPTH_COUPLED_MATERIAL_STEP'
    report=dict(state=state,source_commit=args.source_commit,
        input_sha256=inputs,history=solved['history'],evaluations=evaluations,
        initial_charge_relative=float(np.max(abs(start_candidate['charge_residual']/start_candidate['ne']))),
        final_charge_relative=float(np.max(abs(solved['charge']/solved['ne']))),
        max_ne_relative_change=float(np.max(abs(solved['ne']/snapshot['ne']-1))),
        stages=stages,Si_S_similarity=similarity,
        comparison_baseline='CMFGEN material70 fixed-T input; not converged matched released-T anchor',
        accepted_output_sha256=B.sha(out/'accepted-response.npz') if selected is not None else None,
        all_elements=6,bound_ions=27,points=90,terminal=6,
        charge_solved=bool(np.max(abs(solved['charge']/solved['ne']))<=1.e-10),
        temperature_updated=bool(selected is not None and not np.array_equal(selected['temperature'],snapshot['temperature'])),
        max_temperature_relative_change=float(np.max(abs(selected['temperature']/snapshot['temperature']-1))) if selected is not None else 0.,
        internal_fixed_field_numerical_convergence=solved['converged'] if coupled else False,
        thermal_policy=args.thermal_policy,RE_solved=False,native_E_Jv_verified=False,new_native=0,new_transport=0,
        reused_numerical_direction=basis_identity,frozen_alpha=args.frozen_alpha,
        accepted_source_directory=str(accepted_root) if coupled else None,
        newton_linearization_sha256=B.sha(out/'newton-linearization.npz') if coupled and (out/'newton-linearization.npz').exists() else None,
        radioactive_policy='stored/instantaneous_scenario kept separate; no reference installed',
        final_goal='NOT_MET',wall_seconds=time.monotonic()-started)
    (out/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('state','final_charge_relative','wall_seconds')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--library',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--source-commit',required=True)
    p.add_argument('--common-result',type=Path);p.add_argument('--common-result-sha')
    p.add_argument('--thermal-policy',choices=('physical_stored','physical_instantaneous_scenario',
        'native_stored','native_instantaneous_scenario'))
    p.add_argument('--linearization',type=Path);p.add_argument('--linearization-sha')
    p.add_argument('--linearization-inputs-sha');p.add_argument('--frozen-alpha',type=float)
    run(p.parse_args())
