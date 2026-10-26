#!/usr/bin/env python3
"""전90점·전원소의 동일 T/ne 고정장 SE 응답. charge/RE 공동 해는 아니다.

완료 BF90과 선형장 BB90을 재사용하고 Ω(T)를 전 FL쌍 streaming으로 전달한다.
기존 C 조립기/GTH가 실제 후보와 에너지/전하 잔차를 소유한다. 새 적분/RT 없음.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import struct
import time

import numpy as np

import cmfgen_model_continuum as B
from lumina_model_collision_source import ModelCollisionSource
from lumina_model_material_bridge import MaterialBridge


BF_RESULT=Path('/gpfs/kjhan/cmfgen_runs/lumina_material_bf90_20261027-B64JUq/run/result.json')
BB_RESULT=Path('/gpfs/kjhan/cmfgen_runs/lumina_linear_BB90_20261027-gfIdax/run/result.json')
BASE=Path('/gpfs/kjhan/cmfgen_runs/lumina_extended_4h_20261022-rJGMVR/gpu-local-plan.json')
PARAMETERS=Path('/gpfs/kjhan/cmfgen_runs/lumina_direct_slprofile_20261026-UUiwsU/model-occupation-parameters.bin')


def checked_report(path,expected_hash,state,identities):
    if B.sha(path)!=expected_hash:raise ValueError('완료 원 입력 보고서 변경')
    r=json.loads(path.read_text())
    if r['state']!=state:raise ValueError('전모형 완료 입력이 아님')
    identities[str(path)]=expected_hash
    for name,sha in r['output_sha256'].items():identities[str(path.parent/name)]=sha
    return r


def run(args):
    if (os.environ.get('SLURM_CPUS_PER_TASK')!='16' or os.environ.get('SLURM_MEM_PER_NODE')!='32768' or
        os.environ.get('OMP_NUM_THREADS')!='16'):
        raise ValueError('CPU16/32GiB/OpenMP16 공통 응답 시험 전용')
    start=time.monotonic();out=args.out.resolve()
    if out.exists():raise FileExistsError('기존 전90점 물질응답 보존')
    identities={}
    bf=checked_report(BF_RESULT,'482b18c4d884984787c108575e096cd7ac18665aa5e85fda63a66f83f8d286c6',
                      'COMPLETE_FROZEN_MATERIAL_BF_90',identities)
    bb=checked_report(BB_RESULT,'b95b16e3401068bc8239116a87c872f392e970ec5780b7ce04d32cf51d1e9ec9',
                      'COMPLETE_LINEAR_FIELD_BB90',identities)
    base=json.loads(BASE.read_text());config_path=Path(base['config']);config=json.loads(config_path.read_text())
    plan_path=Path(config['plan']);plan=json.loads(plan_path.read_text())
    field_path=Path(plan['same_material_reference_field'])/'FORMAL_PRIMARY_FIELDS.bin'
    if (bf['data_sha256'].get(str(field_path))!=plan['reference_field_sha256'] or
        bb['input_sha256'].get(str(field_path))!=plan['reference_field_sha256'] or
        bf['source_identity']!=base['source_identity']):raise ValueError('공통 J/물질 계보 아님')
    identities.update(bf['data_sha256']);identities.update(bb['input_sha256'])
    atomic=Path(config['seed'])/'atomic';levels=atomic/'levels.csv'
    material=Path(config['seed'])/'native_material_nodes.bin';initial=Path(plan['initial_material'])
    project=Path(__file__).resolve().parents[1]
    code=[project/'src/nlte_model_material_bridge.c',project/'src/nlte_model_material_bridge.h',
          project/'scripts/lumina_model_material_bridge.py',Path(__file__),
          project/'scripts/lumina_model_collision_source.py']
    for p in (BASE,config_path,plan_path,levels,PARAMETERS,args.library,*code,
              project/'data/atomic/topion_levels.csv',project/'data/atomic/ionization_reference.csv'):
        identities[str(p)]=B.sha(p)
    if B.sha(levels)!=bf['data_sha256'][str(Path(config['cells'])/'atomic/levels.csv')]:
        raise ValueError('다른 FL 원자 주소')
    if any(B.sha(Path(p))!=sha for p,sha in identities.items()):raise ValueError('완료 입력 변경')
    with np.load(BF_RESULT.parent/'material-snapshot.npz') as f:snapshot={k:f[k] for k in f.files}
    with np.load(BF_RESULT.parent/'route-roster.npz') as f:roster={k:f[k] for k in f.files}
    with np.load(BB_RESULT.parent/'line-roster.npz') as f:lines={k:f[k] for k in f.files}
    rates=np.load(BF_RESULT.parent/'total-rates.npy',mmap_mode='r')
    jbar=np.load(BB_RESULT.parent/'jbar.npy',mmap_mode='r');status=np.load(BB_RESULT.parent/'status.npy')
    done_bf=np.load(BF_RESULT.parent/'routes-complete.npy');done_bb=np.load(BB_RESULT.parent/'lines-complete.npy')
    if (rates.shape!=(24667,90,6) or jbar.shape!=(790493,90) or not done_bf.all() or not done_bb.all() or
        status.shape!=(790493,) or not np.isin(status,[1,2]).all() or not np.isfinite(jbar).all() or np.any(jbar<0)):
        raise ValueError('전원소·전90점 BF/BB 결손')
    rows=list(csv.DictReader(levels.open()));source=ModelCollisionSource(initial,rows)
    z=np.array([int(r['atomic_number']) for r in rows]);species=sorted(set(map(int,z)))
    if len(source.models)!=27 or len(rows)!=24542 or len(species)!=6:raise ValueError('전 원자모형 누락')
    out.mkdir()
    occupation=out/'occupation.bin'
    with occupation.open('xb') as f:
        f.write(struct.pack('<8sII',b'LMATOCC1',len(rows),90))
        f.write(np.asarray(snapshot['occupation'],dtype='<f8').tobytes())
    (out/'inputs.json').write_text(json.dumps(dict(source_commit=args.source_commit,input_sha256=identities,
        resources='CPU16/32GiB/2h/OpenMP16/GPU0/retry0',scope=__doc__,
        preserved='native material70/J70/atomic data',new_BF=0,new_BB=0,new_native=0,new_transport=0),indent=2)+'\n')
    state={k:snapshot[k] for k in ('columns','superlevel','temperature','ne','occupation')}
    output=np.full(state['superlevel'].shape,np.nan);ledger=np.full((len(species),90,7),np.nan)
    completed=np.zeros((len(species),90),dtype=bool);pair_count=0;collision_identities={}
    def prepare(bridge,trial_id,s=state):
        bridge.prepare(s,snapshot['fraction'],snapshot['saha'],roster,rates,
                       trial_id=trial_id,radiation_generation=70)
    with MaterialBridge(args.library,atomic,material,occupation,PARAMETERS) as bridge:
        prepare(bridge,2)
        try:prepare(bridge,2)
        except ValueError:pass
        else:raise ValueError('동일 trial ID 재사용 허용 결함')
        wrong=dict(state,columns=state['columns'][::-1])
        try:prepare(bridge,3,wrong)
        except ValueError:pass
        else:raise ValueError('바뀐 SL/terminal 원자 주소 허용 결함')
        prepare(bridge,4)
        try:bridge.finalize(output)
        except ValueError:pass
        else:raise ValueError('전원소 미완결 후보 게시 결함')
        prepare(bridge,5)
        for iz,Z in enumerate(species):
            ids=np.flatnonzero(z[lines['lower']]==Z)
            selected={k:array[ids] for k,array in lines.items()}
            nu=selected['nu0'];A=selected['A'];bul=A*B.C**2/(2*B.H*nu**3)
            for node in range(90):
                values=jbar[ids,node]
                bridge.begin_element(Z,node,selected,bul/selected['g_ratio']*values,bul*values,A,status[ids])
            for key in sorted(k for k in source.models if k[0]==Z):
                ion=source.load(*key);collision_identities.update(ion.identity)
                for lo,up,omega,counts in ion.blocks(state['temperature']):
                    if time.monotonic()-start>6600:raise TimeoutError('전모형 응답 한도; 부분 출력 보존')
                    bridge.push_points(Z,lo,up,omega);pair_count+=len(lo)
                ion.verify_identity()
                print(json.dumps(dict(phase='COLLISION',Z=Z,stage=key[1],pairs_done=pair_count,
                                      seconds=time.monotonic()-start)),flush=True)
            ledger[iz]=bridge.finish_points(Z,output);completed[iz]=True
            np.savez(out/('partial-Z'+str(Z)+'.npz'),population=output,completed=completed,ledger=ledger,species=species)
        derived=bridge.finalize(output)
    if pair_count!=20323356 or not completed.all() or not np.isfinite(output).all() or np.any(output<=0):
        raise ValueError('전 FL쌍/SL/terminal 응답 누락')
    if ledger[:,:,6].max()>1.e-8:raise ValueError('전모형 고정장 SE 잔차 불충족')
    columns=state['columns'];original=state['superlevel'];stages=[]
    for Z in species:
        for q in sorted(set(map(int,columns[columns[:,0]==Z,1]))):
            select=(columns[:,0]==Z)&(columns[:,1]==q)
            old=original[select].sum(axis=0);new=output[select].sum(axis=0)
            total=original[columns[:,0]==Z].sum(axis=0)
            stages.append(dict(Z=Z,charge=q,original_fraction=(old/total).tolist(),candidate_fraction=(new/total).tolist()))
    if any(B.sha(Path(p))!=sha for p,sha in {**identities,**collision_identities}.items()):
        raise ValueError('전모형 응답 중 원 입력 변경')
    np.savez(out/'response.npz',population=output,columns=columns,temperature=state['temperature'],ne=state['ne'],
             species=species,ledger=ledger,completed=completed,**derived)
    result=dict(state='COMPLETE_PRIVATE_FIXED_FIELD_ALL_MODEL_RESPONSE',source_commit=args.source_commit,
        input_sha256=identities,collision_input_sha256=collision_identities,
        elements=6,bound_ions=27,points=90,SL=1792,terminal=6,pairs=pair_count,
        maximum_stationary_SE_residual=float(ledger[:,:,6].max()),
        maximum_charge_residual_relative=float(np.max(np.abs(derived['charge_residual']/state['ne']))),
        stages=stages,negative_controls=3,output_sha256=B.sha(out/'response.npz'),
        comparison_baseline='CMFGEN material70 input at fixed T/ne; not a converged released-T anchor',
        post_response_thermodynamics_refreshed=True,
        wall_seconds=time.monotonic()-start,charge_RE_solved=False,material_updates=0,
        new_BF=0,new_BB=0,new_native=0,new_transport=0,final_goal='NOT_MET')
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('state','maximum_stationary_SE_residual',
        'maximum_charge_residual_relative','wall_seconds')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source-commit',required=True)
    run(p.parse_args())
