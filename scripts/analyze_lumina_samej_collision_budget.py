#!/usr/bin/env python3
"""완료된 동일J 결과의 충돌 유량과 알려진 native 상수 차이를 독립 재구성한다.

새 율 적분/native/수송/물질 갱신은 없으며 비교 상수를 물리 구현에 피팅하지 않는다.
native witness의 충돌 유량은 하측 경계이지 정확한 항별 출력이 아님을 보존한다.
"""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

import cmfgen_model_continuum as B
import build_cmfgen_fullpair_omega as O
from run_lumina_pseudo_candidate_overlay import HEADER,RECORD
from run_cmfgen_native_se_observation import read_witness
from ptf_gated_metrics import read_variables


def direct_ci(temperature,ne,edge,sigma,charge,native=False):
    u=B.H*edge/(B.KB*temperature)
    g=.1*min(charge+1,3)
    if native:
        return ne*3.23e10*g*sigma*np.sqrt(temperature/1.e4)/(edge/1.e15)*np.exp(-u)
    return ne*1.55e13/np.sqrt(temperature)*g*sigma*np.exp(-u)/u


def run(args):
    if (os.environ.get('SLURM_CPUS_PER_TASK') != '1' or
        os.environ.get('SLURM_MEM_PER_NODE') != '4096'):
        raise ValueError('CPU1·4GiB Slurm 전용')
    if args.out.exists():
        raise FileExistsError('기존 충돌예산 결과 보존')
    response=json.loads(args.response.read_text())
    source=Path(response['source_result'])
    original=json.loads(source.read_text())
    if (B.sha(source)!=response['source_sha256'] or
        not response['source_unchanged'] or not response['control_exactly_reproduced'] or
        original['input_material']!=70 or original['native_node']!=49 or
        not response['cases']['CI1_D1']['dissolution_enabled']):
        raise ValueError('같은 원70/J70의 완료 대조가 아님')
    project=Path(__file__).resolve().parents[1]
    base_path=Path('/gpfs/kjhan/cmfgen_runs/lumina_extended_4h_20261022-rJGMVR/gpu-local-plan.json')
    base=json.loads(base_path.read_text())
    config=json.loads(Path(base['config']).read_text())
    levels_path=Path(config['seed'])/'atomic/levels.csv'
    occupation=next(Path(p) for p in response['input_sha256'] if p.endswith('/model-occupation-input.bin'))
    collision=next(Path(p) for p in response['input_sha256'] if p.endswith('/fullpair.bin'))
    bf_path=next(Path(p) for p in original['cases']['native_discrete']['scalar_files_sha256'] if p.endswith('-bf.bin'))
    witness_path=next(Path(p) for p in original['input_sha256'] if p.endswith('/NATIVE_SE_BOUND_000071.bin'))
    native=witness_path.parent
    native_sources=[Path('/gpfs/kjhan/cmfgen_src/cur_cmf')/name for name in
        ('newsubs/omega_gen_v3.f','newsubs/subcol_multi_v6.f','newsubs/genosc_v9.f','subs/ltepop_wld_v2.f',
         'subs/phys_con.f','new_main/cmfgen.f')]
    identities={str(path):B.sha(path) for path in
        (args.response,source,base_path,Path(base['config']),levels_path,occupation,collision,
         bf_path,witness_path,native/'LEVEL_SL_STEQ_LINKS',Path(__file__),*native_sources)}
    for path in (base_path,levels_path,occupation,collision):
        if identities[str(path)]!=response['input_sha256'][str(path)]:
            raise ValueError('봉인된 충돌 입력 변경')
    if (B.sha(bf_path)!=original['cases']['native_discrete']['scalar_files_sha256'][str(bf_path)] or
        B.sha(witness_path)!=original['input_sha256'][str(witness_path)]):
        raise ValueError('저장 native/BF 계수 변경')
    rows=list(csv.DictReader(levels_path.open()))
    z=np.array([int(r['atomic_number']) for r in rows])
    q=np.array([int(r['ion_number']) for r in rows])
    sl=np.array([int(r['super_level']) for r in rows])
    g=np.array([float(r['g']) for r in rows])
    energy=np.array([float(r['energy_eV']) for r in rows])
    blob=bf_path.read_bytes()
    tag,nl,nr,ns,node,temp,ne=HEADER.unpack_from(blob)
    if (tag,nl,ns,node)!=(b'LMBFFUL1',len(rows),90,49):
        raise ValueError('저장 BF 주소 오류')
    bf=np.frombuffer(blob,dtype=RECORD,offset=HEADER.size)
    w=np.frombuffer(occupation.read_bytes(),dtype='<f8',offset=16).reshape(nl,ns)[:,node]
    raw=response['cases']['CI1_D1']['profile']
    reference={(r['Z'],r['charge'],r['superlevel']):r for r in raw}
    material={(r['Z'],r['charge'],r['superlevel']):r['original']
              for r in original['cases']['native_discrete']['profile']}
    keys=list(reference);index={k:i for i,k in enumerate(keys)}
    row_index=np.full(nl,-1,dtype=int);fraction=np.zeros(nl)
    groups={}
    for lo in np.flatnonzero(np.isin(z,[14,16])):
        groups.setdefault((int(z[lo]),int(q[lo]),int(sl[lo])),[]).append(lo)
    for key,ids in groups.items():
        row_index[ids]=index[key]
        weights=g[ids]*w[ids]*np.exp(-(energy[ids]-energy[ids].min())*B.EV/(B.KB*temp))
        fraction[ids]=weights/weights.sum()
    # LTEPOP_WLD_V2의 명시 상수와 현재 물리 Saha 상수의 비. 결과 피팅이 아니다.
    physical_saha=.5*(B.H*B.H/(2*np.pi*B.ME*B.KB*1.e4))**1.5
    saha_scale=2.07078e-22/physical_saha
    plan=json.loads(Path(config['plan']).read_text())
    material_run=Path(plan['initial_material'])
    links=B.atomic.load_cmfgen_links(material_run/'atomic_links.txt')
    models=B.atomic.load_cmfgen_model_spec(material_run/'MODEL_SPEC',material_run/'atomic_links.txt')
    identities.update({str(p):B.sha(p) for p in (Path(O.__file__),material_run/'atomic_links.txt',material_run/'MODEL_SPEC')})
    native_omega=[];legacy_omega=[];primitive_info=[]
    native_edges=np.zeros(nl)
    for Z,ion0,name,expected in O.IONS:
        ids=np.flatnonzero((z==Z)&(q==ion0)).tolist()
        for kind in ('osc','col'):
            p=Path(links[Z,ion0+1][kind]);identities[str(p)]=B.sha(p)
        osc=B.parse_osc(links[Z,ion0+1]['osc'])
        native_edges[ids]=O.native_edge_frequencies(links[Z,ion0+1]['osc'],osc,len(ids))
        _,_,legacy,legacy_info=O.ion_strengths(Z,ion0,name,expected,ids,rows,models,links,temp)
        lo,up,precise,info=O.ion_strengths(Z,ion0,name,expected,ids,rows,models,links,temp,native_precision=True)
        info['legacy_branches']=legacy_info['branches']
        info['legacy_GF_culled_lines']=legacy_info['GF_culled_lines']
        info['omega_changed_pairs']=int(np.count_nonzero(precise!=legacy))
        info['omega_max_relative_change']=float(np.max(abs(precise/legacy-1.)))
        legacy_omega.append(legacy);native_omega.append(precise);primitive_info.append(info)
    native_omega=np.concatenate(native_omega);legacy_omega=np.concatenate(legacy_omega)
    native_fraction=np.zeros(nl)
    for key,ids in groups.items():
        weights=g[ids]*w[ids]*np.exp((native_edges[ids]-native_edges[ids].max())*1.e15*B.H/(B.KB*temp))
        native_fraction[ids]=weights/weights.sum()
    primary={}
    budgets={name:{kind:np.zeros((2,len(keys)),dtype=np.longdouble)
                   for kind in ('direct_CI','bound_BB','dissolution')}
             for name in ('physical_C','native_constants','native_omega_only','native_primitives')}

    def add(budget,lower,upper,forward,reverse):
        if lower==upper:
            return
        budget[0,upper]+=forward;budget[1,lower]+=forward
        budget[0,lower]+=reverse;budget[1,upper]+=reverse

    for r in bf:
        lo,target,zt,qt,grid=map(int,r['ids'])
        if z[lo] not in (14,16):
            continue
        upper_key=((int(z[target]),int(q[target]),int(sl[target])) if target>=0 else (zt,qt,-1))
        lower,upper=row_index[lo],index[upper_key]
        prob,edge,sigma,saha=map(float,r['values'][:4])
        if grid==0:
            if lo in primary:
                raise ValueError('primary PHOT 중복')
            native_saha=saha*saha_scale*np.exp((native_edges[lo]*1.e15-edge)*B.H/(B.KB*temp))
            if target>=0:
                native_saha*=native_fraction[target]/fraction[target]
            primary[lo]=(upper,saha,native_saha)
        if target>=0 and grid!=0:
            continue
        for name,native_constants in (('physical_C',False),('native_constants',True)):
            rate=prob*direct_ci(temp,ne,edge,sigma,int(q[lo]),native_constants)
            forward=np.longdouble(rate*fraction[lo])*material[keys[lower]]
            reverse=np.longdouble(rate*saha*(saha_scale if native_constants else 1.))*material[upper_key]
            add(budgets[name]['direct_CI'],lower,upper,forward,reverse)
        native_saha=saha*saha_scale*np.exp((native_edges[lo]*1.e15-edge)*B.H/(B.KB*temp))
        if target>=0:
            native_saha*=native_fraction[target]/fraction[target]
        rate=prob*direct_ci(temp,ne,native_edges[lo]*1.e15,sigma,int(q[lo]),True)
        add(budgets['native_primitives']['direct_CI'],lower,upper,
            np.longdouble(rate*native_fraction[lo])*material[keys[lower]],
            np.longdouble(rate*native_saha)*material[upper_key])
    budgets['native_omega_only']['direct_CI']=budgets['native_constants']['direct_CI'].copy()
    cb=collision.read_bytes()
    if HEADER.unpack_from(cb)!=(b'LMCBPTS1',nl,ns,node,157371,temp,ne):
        raise ValueError('충돌 전쌍 주소 오류')
    pairs=np.frombuffer(cb,dtype=[('lo','<i4'),('up','<i4'),('omega','<f8')],offset=HEADER.size)
    if not np.array_equal(legacy_omega,pairs['omega']):
        raise ValueError('현재 legacy 생산기가 저장된 전쌍 omega를 정확히 재현하지 못함')
    scale=8.63e-6*ne/np.sqrt(temp)
    for pair,r in enumerate(pairs):
        lo,up=int(r['lo']),int(r['up'])
        lower,upper=row_index[lo],row_index[up]
        excitation=scale*r['omega']*np.exp(-(energy[up]-energy[lo])*B.EV/(B.KB*temp))/g[lo]
        forward=np.longdouble(fraction[lo]*excitation*(w[up]/w[lo]))*material[keys[lower]]
        reverse=np.longdouble(fraction[up]*scale*r['omega']/g[up])*material[keys[upper]]
        for name in ('physical_C','native_constants'):
            add(budgets[name]['bound_BB'],lower,upper,forward,reverse)
        ground,saha,native_saha=primary[lo]
        lost=1.-w[up]/w[lo]
        if lost<0:
            raise ValueError('음의 occupation 용해확률')
        forward=np.longdouble(fraction[lo]*excitation*lost)*material[keys[lower]]
        reverse=np.longdouble(saha*excitation*lost)*material[keys[ground]]
        add(budgets['physical_C']['dissolution'],lower,ground,forward,reverse)
        add(budgets['native_constants']['dissolution'],lower,ground,forward,reverse*saha_scale)
        omega_ratio=native_omega[pair]/r['omega']
        add(budgets['native_omega_only']['bound_BB'],lower,upper,
            np.longdouble(fraction[lo]*excitation*(w[up]/w[lo]))*material[keys[lower]]*omega_ratio,
            np.longdouble(fraction[up]*scale*r['omega']/g[up])*material[keys[upper]]*omega_ratio)
        add(budgets['native_omega_only']['dissolution'],lower,ground,forward*omega_ratio,reverse*saha_scale*omega_ratio)
        native_excitation=scale*native_omega[pair]*np.exp(-(native_edges[lo]-native_edges[up])*1.e15*B.H/(B.KB*temp))/g[lo]
        native_forward=np.longdouble(native_fraction[lo]*native_excitation*(w[up]/w[lo]))*material[keys[lower]]
        native_reverse=np.longdouble(native_fraction[up]*scale*native_omega[pair]/g[up])*material[keys[upper]]
        add(budgets['native_primitives']['bound_BB'],lower,upper,native_forward,native_reverse)
        add(budgets['native_primitives']['dissolution'],lower,ground,
            np.longdouble(native_fraction[lo]*native_excitation*lost)*material[keys[lower]],
            np.longdouble(native_saha*native_excitation*lost)*material[keys[ground]])
    header,witness=read_witness(witness_path)
    if header['parent']!=70:
        raise ValueError('native witness 원 입력 오류')
    addresses={({'Si':14,'S':16}[v.species],v.charge,-1 if v.terminal else v.superlevel-1):v.index-1
               for v in read_variables(native,header['nr']) if v.species in ('Si','S')}
    if set(addresses)!=set(keys):
        raise ValueError('native witness518주소 불일치')
    physical=sum(budgets['physical_C'].values())
    matched=sum(budgets['native_constants'].values())
    primitive=sum(budgets['native_primitives'].values())
    omega_only=sum(budgets['native_omega_only'].values())
    electron_scale=(4.80320427e-10/B.atomic.E_CGS)**2
    result_rows=[]
    for i,key in enumerate(keys):
        r=reference[key];native_row=addresses[key]
        if material[key]!=witness['pops'][native_row,node] or witness['coll_valid'][native_row,node]!=1:
            raise ValueError('native 충돌 물질/유효성 오류')
        physical_net=physical[0,i]-physical[1,i]
        c_net=r['C']['bf_col']+r['C']['bb_col']
        physical_difference=float(physical_net-c_net)
        flow=float(physical[:,i].sum())
        if abs(physical_difference)>1.e-10*flow:
            raise ValueError('독립 충돌 예측이 실제 C를 재현하지 못함')
        native_in,native_out=(float(witness[k][native_row,node]) for k in ('inc','out'))
        adjustment=float((matched[0,i]-matched[1,i])-physical_net)
        corrected_difference=r['difference']+adjustment
        primitive_adjustment=float(primitive[0,i]-primitive[1,i]-physical_net)
        bb_adjustment=(electron_scale-1.)*r['C']['bb_rad']
        result_rows.append(dict(Z=key[0],charge=key[1],superlevel=key[2],
            C_raw=r['C']['total'],native_raw=r['native_raw'],physical_C_collision_net=c_net,
            independent_physical_difference=physical_difference,collision_absolute_flow=flow,
            native_collision_in_lower=native_in,native_collision_out_lower=native_out,
            predicted_native_in=float(matched[0,i]),predicted_native_out=float(matched[1,i]),
            predicted_native_minus_in_lower=float(matched[0,i])-native_in,
            predicted_native_minus_out_lower=float(matched[1,i])-native_out,
            known_collision_constant_adjustment=adjustment,
            native_primitive_in=float(primitive[0,i]),native_primitive_out=float(primitive[1,i]),
            native_omega_only_in=float(omega_only[0,i]),native_omega_only_out=float(omega_only[1,i]),
            native_primitive_minus_in_lower=float(primitive[0,i])-native_in,
            native_primitive_minus_out_lower=float(primitive[1,i])-native_out,
            known_BB_electron_charge_adjustment=bb_adjustment,
            native_primitive_collision_adjustment=primitive_adjustment,
            primitive_adjusted_raw_difference=r['difference']+primitive_adjustment+bb_adjustment,
            lower_bound_difference_diagnostic=(r['difference']+
                native_in-native_out-c_net+bb_adjustment),
            unadjusted_raw_difference=r['difference'],adjusted_raw_difference=corrected_difference,
            adjusted_raw_difference_over_total_flow=(abs(corrected_difference)/r['C']['absolute_flow']
                                                    if r['C']['absolute_flow'] else None)))
    if any(B.sha(Path(p))!=h for p,h in identities.items()):
        raise ValueError('충돌예산 분석 중 입력/소스 변경')
    result=dict(schema='lumina-sameJ-collision-budget-v1',state='COMPLETE_READ_ONLY_DECOMPOSITION',
        input_sha256=identities,rows=result_rows,material=70,native_node=49,
        constants=dict(native_direct_coefficient=3.23,C_direct_coefficient=1.55e13,
                       native_electron_charge=4.80320427e-10,C_electron_charge=B.atomic.E_CGS,
                       native_to_C_BB_electron_charge_square=electron_scale,
                       native_to_physical_saha=saha_scale,
                       native_to_C_direct=direct_ci(temp,ne,1.e15,1.e-18,1,True)/direct_ci(temp,ne,1.e15,1.e-18,1)),
        witness_interpretation='COLLISION_LOWER_BOUND_NOT_EXACT_MATRIX',
        native_primitive_omega=primitive_info,legacy_omega_exactly_reproduced=True,
        native_new_primitives_are_predictions_not_new_C_execution=True,
        no_fitted_scale=True,new_native=0,new_transport=0,new_BF_integrals=0,new_BB_integrals=0,
        public_material_updates=0,source_unchanged=True,full_arithmetic_certificate=False,
        final_goal='NOT_MET')
    with args.out.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(state=result['state'],rows=len(result_rows),constants=result['constants'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--response',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args())
