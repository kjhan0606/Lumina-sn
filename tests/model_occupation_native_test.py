#!/usr/bin/env python3
"""실제90점의 전체 준위 점유확률과 private C 물질 후보를 확인한다.

원 물질은 읽기 전용이다. T/ne/Npositive 변경 사례는 fit 소비 시험이며
SE/전하/RE가 풀린 해로 채택하거나 게시하지 않는다.
"""
import argparse
import ctypes as C
import json
import os
import re
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
from lumina_model_point_rates import ModelPointBoundRates


class Model(C.Structure):
    _fields_=[('n_levels',C.c_size_t),('edge_Hz',C.POINTER(C.c_double)),
              ('z_screen',C.POINTER(C.c_double))]


def run(base_path,library,projection,output,point_library=None,bb_result=None,col_result=None):
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
    point_test=None
    bb_test=None
    collision_test=None
    if point_library is not None:
        point=ModelPointBoundRates(routes,rows,models,state,49,Path(point_library).resolve())
        indices=np.unique(np.linspace(0,len(source.frequency)-1,33).astype('i8'))
        frequency=np.ascontiguousarray(source.frequency[indices]);width=np.zeros_like(frequency)
        left=indices>0;right=indices<len(source.frequency)-1
        width[left]+=.5*(source.frequency[indices[left]]-source.frequency[indices[left]-1])
        width[right]+=.5*(source.frequency[indices[right]+1]-source.frequency[indices[right]])
        j_path=Path('/gpfs/kjhan/cmfgen_runs/lumina_resident_field_20261026-qFP5rJ/run/J.npy')
        with N.FrozenNativeField(j_path,source,source.radius,source.frequency) as field:
            j=np.ascontiguousarray(field.read(np.array([49]),indices)[0])
        block=point.contract_bound_block(frequency,width,j)
        physical=point.physical_ion_rates(block)
        single=dict(state)
        for key in ('full','occupation','superlevel'):single[key]=state[key][:,49:50]
        for key in ('temperature','ne'):single[key]=state[key][49:50]
        coefficients=B.continuum_grid(routes,rows,single,frequency,33)
        photon=4*np.pi*width/(B.H*frequency)
        expected=np.array([np.sum(coefficients['chi_abs'][0]*j*photon),
            np.sum(coefficients['eta_spont'][0]*photon),np.sum(coefficients['chi_stim'][0]*j*photon)])
        got=np.sum(list(physical.values()),axis=0)
        np.testing.assert_allclose(got,expected,rtol=5.e-12,atol=0.)
        per_ion=[]
        for key,actual in sorted(physical.items()):
            selected=[r for r in routes if (r.Z,r.stage)==key]
            ion=B.continuum_grid(selected,rows,single,frequency,33)
            wanted=np.array([np.sum(ion['chi_abs'][0]*j*photon),np.sum(ion['eta_spont'][0]*photon),
                             np.sum(ion['chi_stim'][0]*j*photon)])
            np.testing.assert_allclose(actual,wanted,rtol=5.e-12,atol=0.)
            per_ion.append(dict(Z=key[0],stage=key[1],C=actual.tolist(),python=wanted.tolist()))
        contributions=output.with_name('bound-selected-frequency-contributions.bin')
        dtype=np.dtype([('ids','<i4',(5,)),('values','<f8',(10,))])
        records=np.zeros(len(point.routes),dtype=dtype)
        for i,r in enumerate(point.routes):
            records['ids'][i]=[r.lower,r.target,r.Z if r.target<0 else 0,r.stage if r.target<0 else 0,r.phot_grid]
            records['values'][i]=[r.probability,r.threshold_Hz,point.sigma_edge[i],point.saha[i],*block[i]]
        with contributions.open('xb') as stream:
            stream.write(struct.pack('<8sIIII2d',b'LMBFPTS1',nl,len(point.routes),ns,49,point.temperature,point.ne))
            stream.write(records.tobytes())
        point_command=list(command)
        point_command[4]=str(output.with_name('model-point-imported-full.bin'))
        point_command.append(str(contributions))
        if bb_result is not None:
            bb_result=Path(bb_result)
            sealed=json.loads(bb_result.read_text())
            bb_catalog=bb_result.with_name('point-lines.npz')
            if (B.sha(bb_result)!='94070d528e9e84de48711519d0b4ea38f76157177f6e077f3d5d3dad583b09fe' or
                B.sha(bb_catalog)!=sealed['artifact_sha256'] or not sealed['source_unchanged'] or
                sealed['native_node']!=49 or sealed['input_identity']!=identity or
                sealed['state']!='COMPLETE_RADIATIVE_BB_ONLY'):
                raise ValueError('서로 다른 BB 원래점/물질/장 또는 부분 선 자료')
            with np.load(bb_catalog,allow_pickle=False) as data:
                lower=np.asarray(data['lower'],dtype='i4');upper=np.asarray(data['upper'],dtype='i4')
                nu=np.asarray(data['nu0'],dtype='f8');absorption=np.asarray(data['absorption'],dtype='f8')
                stimulated=np.asarray(data['stimulated'],dtype='f8')
                spontaneous=np.asarray(data['spontaneous'],dtype='f8')
                status=np.asarray(data['status'],dtype='u1')
            if (len(lower)!=sealed['lines'] or any(len(v)!=len(lower) for v in
                 (upper,nu,absorption,stimulated,spontaneous,status)) or
                 not np.isin(status,[1,2]).all() or np.any(lower<0) or np.any(upper<0) or
                 np.any(lower>=nl) or np.any(upper>=nl) or
                 not all(np.isfinite(v).all() for v in (nu,absorption,stimulated,spontaneous))):
                raise ValueError('BB 실물 라인 계약 누락·수치 오류')
            line_z=np.array([int(r['atomic_number']) for r in rows],dtype='i4')
            if np.any(line_z[lower]!=line_z[upper]):raise ValueError('원소 간 BB 전이')
            bb_physical=(state['full'][lower,49]*absorption-
                state['full'][upper,49]*(stimulated+spontaneous))*B.H*nu
            bb_expected={int(Z):float(np.sum(bb_physical[line_z[lower]==Z],dtype=np.longdouble))
                         for Z in np.unique(line_z)}
            bb_binary=output.with_name('point-lines-contributions.bin')
            records=np.empty(len(lower),dtype=np.dtype([('ids','<i4',(2,)),('values','<f8',(4,))]))
            records['ids'][:,0]=lower;records['ids'][:,1]=upper
            for index,value in enumerate((nu,absorption,stimulated,spontaneous)):
                records['values'][:,index]=value
            with bb_binary.open('xb') as stream:
                stream.write(struct.pack('<8sIIII',b'LMBBPTS1',nl,len(lower),ns,49))
                stream.write(records.tobytes())
            point_command.append(str(bb_binary))
        if col_result is not None:
            if bb_result is None:
                raise ValueError('전쌍 충돌 C 시험에는 기존 BB 원래점 handoff가 필요')
            col_result=Path(col_result)
            sealed=json.loads(col_result.read_text())
            binary=col_result.with_suffix('')
            if (sealed['schema']!='lumina-cmfgen-fullpair-omega-v1' or
                sealed['state']!='PRIVATE_INPUT_ONLY' or sealed['material_record']!=70 or
                sealed['native_node']!=49 or sealed['levels']!=nl or
                sealed['temperature_K']!=float(state['temperature'][49]) or
                sealed['electron_density_cm3']!=float(state['ne'][49]) or
                B.sha(binary)!=sealed['output_sha256'] or
                not all(B.sha(Path(path))==digest for path,digest in
                        sealed['input_sha256'].items())):
                raise ValueError('전쌍 Ω 원래점/신원 불일치')
            raw=binary.read_bytes()
            magic,nlevels,nshells,node,count,temp,ne=struct.unpack_from('<8sIIII2d',raw)
            if (magic!=b'LMCBPTS1' or (nlevels,nshells,node,count)!=(nl,ns,49,sealed['pairs']) or
                (temp,ne)!=(float(state['temperature'][49]),float(state['ne'][49])) or
                len(raw)!=40+16*count or len(sealed['ions'])!=8):
                raise ValueError('전쌍 Ω binary 형상 불일치')
            records=np.frombuffer(raw,offset=40,
                dtype=np.dtype([('lower','<i4'),('upper','<i4'),('omega','<f8')]))
            lo=records['lower'];up=records['upper'];omega=records['omega']
            if (np.any(lo<0) or np.any(up>=nl) or np.any(lo>=up) or
                np.any(~np.isfinite(omega)) or np.any(omega<=0) or
                np.any(lo[1:]<lo[:-1]) or
                np.any((lo[1:]==lo[:-1])&(up[1:]<=up[:-1])) or
                sum(row['pairs'] for row in sealed['ions'])!=count):
                raise ValueError('전쌍 Ω 목록 누락/중복/강도 오류')
            z=np.array([int(row['atomic_number']) for row in rows],dtype='i4')
            stage=np.array([int(row['ion_number']) for row in rows],dtype='i4')
            sl=np.array([int(row['super_level']) for row in rows],dtype='i4')
            e=np.array([float(row['energy_eV']) for row in rows])
            g=np.array([float(row['g']) for row in rows])
            if np.any(z[lo]!=z[up]) or np.any(stage[lo]!=stage[up]):
                raise ValueError('원소/이온 간 충돌쌍')
            cross=sl[lo]!=sl[up]
            kt=8.617333262145e-5*temp
            factor=8.63e-6*ne/np.sqrt(temp)
            gap=e[up]-e[lo]
            w=state['occupation'][:,49];pop=state['full'][:,49]
            forward=factor*omega/g[lo]*np.exp(-gap/kt)*w[up]/w[lo]
            reverse=factor*omega/g[up]
            exchange=(pop[lo]*forward-pop[up]*reverse)*gap*B.EV
            expected_heat={int(Z):float(np.sum(exchange[cross&(z[lo]==Z)],dtype=np.longdouble))
                           for Z in (14,16)}
            point_command.append(str(binary))
        completed=subprocess.run(point_command,check=True,env=env,
            timeout=max(1.,250.-(time.monotonic()-started)),
            capture_output=bb_result is not None,text=bb_result is not None)
        if bb_result is not None:
            print(completed.stdout,end='',flush=True)
            matched=re.findall(r'^\[MODEL-BB\]\[ACTUAL\] Z=(\d+) lines=(\d+) intra=(\d+) heat_erg_cm3_s=([^\s]+)$',
                               completed.stdout,re.M)
            if len(matched)!=len(bb_expected) or sum(int(row[1]) for row in matched)!=len(lower):
                raise ValueError('BB 원소별 C handoff 행 누락')
            for Z,bb_count,intra,heat_string in matched:
                Z=int(Z);actual=float(heat_string);bb_reference=bb_expected[Z]
                if not np.isclose(actual,bb_reference,rtol=2.e-7,atol=1.e-30):
                    raise ValueError(f'원소 {Z} 복사 선 열량 불일치: C={actual} 원FL={bb_reference}')
            bb_test=dict(lines=len(lower),source_result_sha256=B.sha(bb_result),
                contributions_sha256=B.sha(bb_binary),C_heat_per_Z={Z:float(value) for Z,_,_,value in matched},
                full_level_heat_per_Z=bb_expected,within_SL_lines=sum(int(row[2]) for row in matched),
                C_all_elements_BB_assembly=True,collisional_BB_included=False,
                original_J_A_after_IR_fix_regenerated=False,SE_RE_solution=False)
        if col_result is not None:
            matched=re.findall(r'^\[MODEL-COLL-BB\]\[ACTUAL\] Z=(\d+) pairs=(\d+) cross=(\d+) intra=(\d+) heat_erg_cm3_s=([^\s]+)$',
                               completed.stdout,re.M)
            if (len(matched)!=2 or sum(int(row[1]) for row in matched)!=count or
                sum(int(row[2])+int(row[3]) for row in matched)!=count):
                raise ValueError('전쌍 C handoff 범위 누락')
            for Z,_,_,_,heat_string in matched:
                if not np.isclose(float(heat_string),expected_heat[int(Z)],rtol=2.e-7,atol=1.e-25):
                    raise ValueError(f'원소 {Z} 충돌 열량 C/원FL 불일치')
            collision_test=dict(native_node=49,source_result_sha256=B.sha(col_result),
                binary_sha256=B.sha(binary),pairs=count,
                C_heat_per_Z={int(Z):float(h) for Z,_,_,_,h in matched},
                full_level_heat_per_Z=expected_heat,
                C_all_Si_S_bound_ion_BB_collision_assembly=True,
                full_SE_RE_solution=False)
        point_test=dict(native_node=49,routes=len(point.routes),ions=len(physical),selected_frequencies=len(frequency),
            terminal_ions=[dict(Z=key[0],lower_stage=key[1],energy_eV=value[0],g=value[1])
                           for key,value in sorted(point.terminal.items())],
            terminal_inverse='native ground scalar; all PHOT EION=0 and GION identical per ion',
            C_photo_spont_stim=got.tolist(),python_photo_spont_stim=expected.tolist(),
            per_ion=per_ion,contributions_sha256=B.sha(contributions),
            actual_C_all_elements_BF_assembly=True,uncomputed_native_nodes='REJECTED',
            bound_only=True,pseudo_included=False,complete_frequency_integral=False,
            SE_RE_solution=False,point_library_sha256=B.sha(Path(point_library)))
    result=dict(state='PASS',scope='actual native90 full-model occupation and private thermodynamics',
        cases=cases,input_identity=identity,source_population_check=full_check,
        imported_population_check=imported,failed_fit_preserves_output=True,
        executable_sha256=B.sha(projection),library_sha256=B.sha(library),
        inputs_sha256={p.name:B.sha(p) for p in (occupation,parameters)},
        auxiliary_atomic_sha256={p.name:B.sha(p) for p in (catalog,reference)},
        script_sha256=B.sha(Path(__file__)),wall_seconds=time.monotonic()-started,
        fit_sensitivity_is_SE_RE_solution=False,public_material_updates=0,
        point_bound_test=point_test,point_bb_test=bb_test,new_transport=0,final_goal='NOT_MET')
    result['point_collision_test']=collision_test
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('MODEL_OCCUPATION_NATIVE_TEST PASS values='+str(nl*ns),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True)
    p.add_argument('--library',required=True);p.add_argument('--projection',required=True)
    p.add_argument('--point-library')
    p.add_argument('--bb-result')
    p.add_argument('--col-result')
    p.add_argument('--out',required=True);args=p.parse_args()
    run(args.base,args.library,args.projection,args.out,args.point_library,args.bb_result,args.col_result)
