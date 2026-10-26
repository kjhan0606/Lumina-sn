#!/usr/bin/env python3
"""실제90점의 전체 준위 점유확률과 private C 물질 후보를 확인한다.

원 물질은 읽기 전용이다. T/ne/Npositive 변경 사례는 fit 소비 시험이며
SE/전하/RE가 풀린 해로 채택하거나 게시하지 않는다.
"""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import cmfgen_model_continuum as B
import lumina_native_columns as N
import lumina_native90_rates as R


class Model(C.Structure):
    _fields_=[('n_levels',C.c_size_t),('edge_Hz',C.POINTER(C.c_double)),
              ('z_screen',C.POINTER(C.c_double))]


def run(base_path,library,projection,output):
    started=time.monotonic();base_path=Path(base_path);output=Path(output)
    if output.exists():raise FileExistsError('기존 실제 물질 시험 보존')
    if B.sha(base_path)!='f3b01d29c798e910fe42427283f1a9697189da45926d0bc4b77961e991909bab':
        raise ValueError('다른 원 native 계획')
    base=json.loads(base_path.read_text());c=json.loads(Path(base['config']).read_text())
    plan=json.loads(Path(c['plan']).read_text())
    report_path=Path(c['reuse_sources']['boundary_sources']['directory'])/'sources.json'
    if B.sha(report_path)!=c['reuse_sources']['boundary_sources']['report_sha256']:
        raise ValueError('다른 생산 계수 신원')
    report=json.loads(report_path.read_text())
    with N.open_sealed_native(base['config'],base['preparation'],base['preparation_sha256']) as source:
        rows,models,routes,state,do_dis,full_check,smoothing=R.material_state(c,plan,source,report)
        identity=dict(source.identity)
    if not do_dis:raise ValueError('실제 DO_DIS 물질이 아님')
    nl,ns=state['full'].shape
    edge=np.empty(nl);z=np.empty(nl)
    for l,row in enumerate(rows):
        model=models[int(row['atomic_number']),int(row['ion_number'])+1]
        edge[l]=(model['ionization_eV']-float(row['energy_eV']))*B.EV/B.H
        z[l]=model['z_screen']
    library=Path(library);projection=Path(projection);lib=C.CDLL(str(library.resolve()))
    ptr=C.POINTER(C.c_double);fn=lib.population_model_occupation_build
    fn.argtypes=[C.POINTER(Model),C.c_size_t,ptr,ptr,ptr,ptr];fn.restype=C.c_int
    model=Model(nl,edge.ctypes.data_as(ptr),z.ctypes.data_as(ptr))
    cases=[];original_w=None
    for name in ('original','T_ne_native49','Npositive_native49'):
        trial=dict(state)
        trial['temperature']=state['temperature'].copy();trial['ne']=state['ne'].copy()
        trial['superlevel']=state['superlevel'].copy()
        if name=='T_ne_native49':
            trial['temperature'][49]*=1.07;trial['ne'][49]*=1.11
        if name=='Npositive_native49':
            trial['superlevel'][state['columns'][:,1]>0,49]*=1.13
        positive=np.ascontiguousarray(trial['superlevel'][state['columns'][:,1]>0].sum(axis=0))
        expected=B.material_occupation(rows,models,trial)
        got=np.full_like(expected,-1.)
        args=[np.ascontiguousarray(a,dtype='f8') for a in (trial['temperature'],trial['ne'],positive)]
        rc=fn(C.byref(model),ns,*[a.ctypes.data_as(ptr) for a in args],got.ctypes.data_as(ptr))
        if rc:raise ValueError('실제 C occupation fit 실패: '+str(rc))
        np.testing.assert_allclose(got,expected,rtol=5.e-12,atol=0.)
        relative=np.divide(abs(got-expected),expected,out=np.zeros_like(expected),where=expected>0)
        cases.append(dict(name=name,values=got.size,max_relative=float(relative.max()),
            w_min=float(got.min()),w_max=float(got.max()),
            changed_from_original=0 if original_w is None else int(np.count_nonzero(got!=original_w))))
        if original_w is None:original_w=got.copy()
    saved=got.copy();args[1][0]=np.nan
    if fn(C.byref(model),ns,*[a.ctypes.data_as(ptr) for a in args],got.ctypes.data_as(ptr))==0 or not np.array_equal(got,saved):
        raise ValueError('실제 fit 실패의 부분 출력 변경')
    occupation=output.with_name('model-occupation-input.bin')
    parameters=output.with_name('model-occupation-parameters.bin')
    exported=output.with_name('model-imported-full.bin')
    with occupation.open('xb') as stream:
        stream.write(struct.pack('<8sII',b'LMATOCC1',nl,ns))
        stream.write(np.ascontiguousarray(state['occupation'],dtype='<f8').tobytes())
    with parameters.open('xb') as stream:
        stream.write(struct.pack('<8sI',b'LMOCPAR1',nl))
        stream.write(np.asarray(edge,dtype='<f8').tobytes());stream.write(np.asarray(z,dtype='<f8').tobytes())
    # 원래90점의 물질 입력을 사용한다. cells의89개 평균 조성은 이
    # native 후보에 맞춰 늘리거나 재해석하지 않는다.
    native_atomic=Path(c['seed'])/'atomic'
    seed=json.loads((Path(c['seed'])/'material-seed.json').read_text())
    for name in ('abundances.csv','element_number_densities.csv','electron_temperatures.csv'):
        if B.sha(native_atomic/name)!=seed['output_sha256']['atomic/'+name]:
            raise ValueError('native90 원 조성/고정 원자수/온도 신원 불일치')
    if B.sha(native_atomic/'levels.csv')!=B.sha(Path(c['cells'])/'atomic/levels.csv'):
        raise ValueError('원자 준위 순서 불일치')
    command=[str(projection.resolve()),str(native_atomic),str(ns),
        str(Path(c['seed'])/'native_material_nodes.bin'),str(exported),str(occupation),str(parameters)]
    project=Path(__file__).resolve().parents[1]
    catalog=project/'data/atomic/topion_levels.csv'
    reference=project/'data/atomic/ionization_reference.csv'
    env=dict(os.environ,LUMINA_TOPION_LEVELS_FILE=str(catalog),
             LUMINA_IONIZATION_REFERENCE_FILE=str(reference))
    subprocess.run(command,check=True,env=env,timeout=max(1.,250.-(time.monotonic()-started)))
    imported=B.consume_c_full_population(exported,dict(state))
    result=dict(state='PASS',scope='actual native90 full-model occupation and private thermodynamics',
        cases=cases,input_identity=identity,source_population_check=full_check,
        imported_population_check=imported,failed_fit_preserves_output=True,
        executable_sha256=B.sha(projection),library_sha256=B.sha(library),
        inputs_sha256={p.name:B.sha(p) for p in (occupation,parameters)},
        auxiliary_atomic_sha256={p.name:B.sha(p) for p in (catalog,reference)},
        script_sha256=B.sha(Path(__file__)),wall_seconds=time.monotonic()-started,
        fit_sensitivity_is_SE_RE_solution=False,public_material_updates=0,
        new_transport=0,final_goal='NOT_MET')
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('MODEL_OCCUPATION_NATIVE_TEST PASS values='+str(nl*ns),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True)
    p.add_argument('--library',required=True);p.add_argument('--projection',required=True)
    p.add_argument('--out',required=True);args=p.parse_args()
    run(args.base,args.library,args.projection,args.out)
