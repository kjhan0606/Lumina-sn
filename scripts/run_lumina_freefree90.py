#!/usr/bin/env python3
"""원197291주파수 J70의 FF 열계수를90점에서 생산해 원/고정장 후보에 재사용한다.

독립 묶음 완료나 SE/전하/RE/수송 수렴이 아니다. 새 BF/BB·수송·native는0이다.
"""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

import cmfgen_model_continuum as B
import cmfgen_model_sources as S
from compare_native_prrr_common_grid import continuum_mapping
from lumina_model_freefree import ModelFreeFree
from run_cmfgen_formal_primary import fields


def run(args):
    started=time.monotonic();out=args.out.resolve()
    if out.exists():raise FileExistsError('기존 FF 결과 보존')
    out.mkdir()
    runs=Path('/gpfs/kjhan/cmfgen_runs')
    bf=runs/'lumina_material_bf90_20261027-B64JUq/run'
    response=runs/'lumina_full_model_response_20261027-vzt2uX/run'
    base_path=runs/'lumina_extended_4h_20261022-rJGMVR/gpu-local-plan.json'
    config_path=Path(json.loads(base_path.read_text())['config']);config=json.loads(config_path.read_text())
    plan_path=Path(config['plan']);plan=json.loads(plan_path.read_text())
    kernel=runs/'lumina_full_source_tests_20261020-0sY1jL/kernels/model_sources.so'
    bfreport=json.loads((bf/'result.json').read_text());rreport=json.loads((response/'result.json').read_text())
    if (B.sha(bf/'result.json')!='482b18c4d884984787c108575e096cd7ac18665aa5e85fda63a66f83f8d286c6' or
        B.sha(response/'result.json')!='a93e5127286bf0b3ef0e3bf41ba39549b4877675518f2bcffb95923cde46cf68' or
        B.sha(kernel)!='0f89ccd19f0a2a7631308b0197213d41dee4b6420c0a24cecad2527d8ddc0386' or
        bfreport['state']!='COMPLETE_FROZEN_MATERIAL_BF_90' or
        rreport['state']!='COMPLETE_PRIVATE_FIXED_FIELD_ALL_MODEL_RESPONSE'):
        raise ValueError('완료 BF/응답/기존 Gaunt 바이너리 정체성 불일치')
    if (B.sha(bf/'material-snapshot.npz')!=bfreport['output_sha256']['material-snapshot.npz'] or
        B.sha(response/'response.npz')!=rreport['output_sha256']):raise ValueError('완료 인구 자료 변경')
    with np.load(bf/'material-snapshot.npz') as f:state={k:f[k] for k in f.files}
    with np.load(response/'response.npz') as f:candidate={k:f[k] for k in f.files}
    if any(not np.array_equal(state[k],candidate[k]) for k in ('columns','temperature','ne')):
        raise ValueError('서로 다른 고정 T/ne/주소')
    material=Path(plan['initial_material']);field_path=Path(plan['same_material_reference_field'])/'FORMAL_PRIMARY_FIELDS.bin'
    evaluation_path=runs/'lumina_fixedj_exp70_20261026-phmMnx/run/material/CONT_FREQ'
    identities={str(p):B.sha(p) for p in (base_path,config_path,plan_path,kernel,
        bf/'result.json',bf/'material-snapshot.npz',response/'result.json',response/'response.npz',
        material/'MODEL',material/'MODEL_SPEC',material/'atomic_links.txt',field_path,evaluation_path)}
    if (identities[str(field_path)]!=plan['reference_field_sha256'] or
        identities[str(evaluation_path)]!=bfreport['data_sha256'][str(evaluation_path)]):
        raise ValueError('다른 장 또는 CONT_FREQ 대응')
    models=B.atomic.load_cmfgen_model_spec(material/'MODEL_SPEC',material/'atomic_links.txt')
    if any(p['external_dr_fit_allowed'] or p['die_auto'] or p['die_wi']
           for p in json.loads((Path(config['reuse_sources']['boundary_sources']['directory'])/'sources.json').read_text())['process_selection']['ions']):
        raise ValueError('미지원 FF resonance 활성')
    geometry,records=fields(field_path)
    code=[Path(__file__),Path(__file__).with_name('lumina_model_freefree.py'),
          Path(__file__).with_name('cmfgen_model_sources.py')]
    code_sha={str(p.resolve()):B.sha(p) for p in code}
    try:
        nu=np.array(records['freq'][::-1]*1.e15);weights=np.zeros_like(nu)
        weights[:-1]+=.5*np.diff(nu);weights[1:]+=.5*np.diff(nu)
        evaluation=continuum_mapping(evaluation_path,nu)
        provider=ModelFreeFree(nu,weights,records['field'][::-1,0,:].T,S.SourceKernels(kernel),
            models,state['columns'],radiation_generation=70,evaluation=evaluation)
        if provider.J.shape!=(90,197291) or len(models)!=27:raise ValueError('전90점/주파수/이온 누락')
        (out/'inputs.json').write_text(json.dumps(dict(input_sha256=identities,code_sha256=code_sha,
            resources='grammar-debug CPU1/4GiB/180s',new_transport=0,new_native=0),indent=2)+'\n')
        coefficients=provider.prepare(state['temperature'],radiation_generation=70)
        original=provider.contract(coefficients,state,radiation_generation=70,material_generation=1)['terms']
        trial={**state,'superlevel':candidate['population']}
        changed=provider.contract(coefficients,trial,radiation_generation=70,material_generation=2)['terms']
        np.savez(out/'coefficients.npz',temperature=coefficients.temperature,charges=provider.charges,
            heating=coefficients.heating,cooling=coefficients.cooling,columns=provider.columns,
            signature=provider.signature)
        data=[]
        for node in range(90):
            row=dict(native_node=node,radius_cm=float(geometry[0,node]*1.e10))
            for state_name,terms in (('original',original),('fixed_T_ne_SE_response',changed)):
                for policy,v in terms.items():
                    for name in ('heating','cooling','net'):row[state_name+'_'+policy+'_'+name]=float(v[name][node])
            data.append(row)
        with (out/'thermal.csv').open('x') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
        if any(B.sha(Path(p))!=h for p,h in {**identities,**code_sha}.items()):raise ValueError('실행 중 입력/코드 변경')
        differences={name:dict(maximum_abs_policy_net_difference=float(np.max(abs(
            terms['all_positive_ions']['net']-terms['native_upper_ions']['net']))),
            maximum_abs_all_ion_net=float(abs(terms['all_positive_ions']['net']).max()))
            for name,terms in (('original',original),('fixed_T_ne_SE_response',changed))}
        result=dict(state='COMPLETE_FROZEN_FIELD_FF90_THERMAL_COEFFICIENTS',input_sha256=identities,
            code_sha256=code_sha,field_signature=provider.signature,points=90,frequencies=len(nu),
            charges=provider.charges.tolist(),differences=differences,
            output_sha256={n:B.sha(out/n) for n in ('coefficients.npz','thermal.csv')},
            selected_nodes=[data[i] for i in (0,22,35,49,65,89)],
            coefficient_reused_for_pop_ne_change=True,coefficient_valid_for_changed_temperature=False,
            wall_seconds=time.monotonic()-started,new_BF=0,new_BB=0,new_transport=0,new_native=0,
            material_updates=0,charge_RE_solved=False,final_goal='NOT_MET')
        (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:result[k] for k in ('state','points','frequencies','differences','wall_seconds')}))
    finally:records._mmap.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    run(p.parse_args())
