#!/usr/bin/env python3
"""같은 입력 상태의 native FL/점유확률과 Lumina 저장 배열을 진단 비교한다.

이 도구는 사전 승인된 물리 동질성 기준이 아니다. 실제 native 실행의 상태가
입력/열행과 동일한지 확인하고, 전체 FL 값을 종별·깊이별로 나란히 기록한다.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from expand_atomic_data_cmfgen import load_cmfgen_model_spec
from prepare_cmfgen_fixedj_heat import read_heat_row


PHASE = 'AFTER_CURRENT_STATE_EVAL_LTE_BEFORE_RATE_AND_HEAT_EVALUATION'
SHA256_ARGUMENTS = {
    'expected-j-sha256','expected-vadat-sha256','expected-response-sha256',
    'expected-lumina-balance-sha256'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def argument_type(name):
    return str if name in SHA256_ARGUMENTS else Path


def identity(path):
    path=Path(path)
    return {'name':path.name,'sha256':sha(path)}


def verify_manifest_artifact(manifest_path,artifact_path,key):
    manifest_path=Path(manifest_path);artifact_path=Path(artifact_path)
    manifest=json.loads(manifest_path.read_text())
    outputs=manifest.get('output_sha256')
    if not isinstance(outputs,dict) or key not in outputs:
        raise ValueError(f'{manifest_path.name}에 {key} 산출물 해시가 없음')
    actual=sha(artifact_path)
    if actual!=outputs[key]:
        raise ValueError(f'{artifact_path.name}이 준비 manifest의 {key}와 불일치')
    return manifest


def verify_run_identity(args):
    build=verify_manifest_artifact(args.build_manifest,args.executable,
                                   'cmfgen_fixedj_heat.exe')
    field_manifest=Path(args.field_manifest)
    field=verify_manifest_artifact(field_manifest,Path(args.field_root)/'EDDFACTOR',
                                   'EDDFACTOR')
    verify_manifest_artifact(field_manifest,Path(args.field_root)/'EDDFACTOR_INFO',
                             'EDDFACTOR_INFO')
    j_inputs=[(name,digest) for name,digest in field.get('input_sha256',{}).items()
              if Path(name).name.lower() in ('j.npy','j.bin')]
    if len(j_inputs)!=1:
        raise ValueError('field manifest에서 고정 J 원본을 하나로 특정할 수 없음')
    expected_j=args.expected_j_sha256.lower()
    if len(expected_j)!=64 or any(ch not in '0123456789abcdef' for ch in expected_j):
        raise ValueError('기대 J SHA256 형식 오류')
    if j_inputs[0][1].lower()!=expected_j:
        raise ValueError('field manifest의 고정 J가 지정된 기준 J와 불일치')
    vadat=Path(args.native_root)/'VADAT'
    expected_vadat=args.expected_vadat_sha256.lower()
    if len(expected_vadat)!=64 or any(ch not in '0123456789abcdef' for ch in expected_vadat):
        raise ValueError('기대 VADAT SHA256 형식 오류')
    if sha(vadat)!=expected_vadat:
        raise ValueError('실제 native run의 VADAT가 준비한 제어 입력과 불일치')
    return dict(build_manifest=Path(args.build_manifest),
        executable=Path(args.executable),field_manifest=field_manifest,
        field_factor=Path(args.field_root)/'EDDFACTOR',
        field_info=Path(args.field_root)/'EDDFACTOR_INFO',
        native_vadat=vadat,
        fixed_J_source_name=Path(j_inputs[0][0]).name,
        fixed_J_source_sha256=j_inputs[0][1])


def read_matrix(path):
    raw=Path(path).read_bytes()
    if len(raw)<16:
        raise ValueError('LMATFL01 행렬 헤더가 잘림')
    magic,nfull,nd=struct.unpack('<8sii',raw[:16])
    if magic!=b'LMATFL01' or nfull<1 or nd<2 or nfull>100000 or nd>10000:
        raise ValueError('LMATFL01 행렬 형식/차원 오류')
    if len(raw)!=16+8*nfull*nd:
        raise ValueError('LMATFL01 행렬 길이 오류')
    data=np.frombuffer(raw[16:],dtype='<f8').reshape(nfull,nd).copy()
    if not np.isfinite(data).all() or np.any(data<0):
        raise ValueError('LMATFL01 행렬의 비유한/음수 값')
    return data


def read_state(path):
    raw=Path(path).read_bytes()
    if len(raw)<32 or raw[:16]!=b'NATIVEHEATINPV1 ':
        raise ValueError('native 고정 상태 입력 헤더 오류')
    version,bits,nt,nd=struct.unpack('<4i',raw[16:32])
    if version!=1 or bits!=64 or nt<3 or nd<2 or len(raw)!=32+8*nd*(nt+1):
        raise ValueError('native 고정 상태 입력 차원/길이 오류')
    values=np.frombuffer(raw[32:],dtype='<f8')
    radius=values[:nd].copy()
    pops=values[nd:].reshape((nt,nd),order='F').copy()
    if (not np.isfinite(values).all() or np.any(radius<=0) or
            np.any(np.diff(radius)>=0) or np.any(pops<0) or
            np.any(pops[-2:]<=0)):
        raise ValueError('native 고정 상태 입력 값 오류')
    return dict(raw=raw,nt=nt,nd=nd,radius=radius,pops=pops)


def read_occupation(path,nfull,nd):
    raw=Path(path).read_bytes()
    if len(raw)<16:
        raise ValueError('LMATOCC1 헤더가 잘림')
    magic,nrow,npoint=struct.unpack('<8sII',raw[:16])
    if magic!=b'LMATOCC1' or (nrow,npoint)!=(nfull,nd) or len(raw)!=16+8*nfull*nd:
        raise ValueError('LMATOCC1 형상/길이 오류')
    values=np.frombuffer(raw[16:],dtype='<f8').reshape(nfull,nd).copy()
    if not np.isfinite(values).all() or np.any(values<0):
        raise ValueError('LMATOCC1 값 오류')
    return values


def canonical_ion_label(label):
    value=''.join(ch for ch in str(label).upper() if ch.isalnum())
    if value.startswith('SK'):
        value='SI'+value[2:]
    if value.startswith('NK'):
        value='NI'+value[2:]
    return value


def level_layout(level_rows,models):
    lookup={}
    for i,row in enumerate(level_rows):
        key=(int(row['atomic_number']),int(row['ion_number'])+1,
             int(row['level_number'])+1)
        if key in lookup:
            raise ValueError('levels.csv 원자·전하·FL 키 중복')
        lookup[key]=i
    labels={}
    expected=set()
    for (z,stage),model in models.items():
        label=canonical_ion_label(model['label'])
        key=(int(z),label)
        if key in labels:
            raise ValueError('MODEL_SPEC native ion label 중복')
        labels[key]=int(stage)
        expected.update((int(z),int(stage),level)
                        for level in range(1,int(model['n_full'])+1))
    if set(lookup)!=expected:
        raise ValueError('MODEL_SPEC 전체 FL 집합과 levels.csv 불일치')
    return lookup,labels


def align_native(matrix,index_rows,level_rows,lookup,labels):
    if len(index_rows)!=matrix.shape[0] or len(level_rows)!=matrix.shape[0]:
        raise ValueError('native/C FL 행 수 불일치')
    aligned=np.empty((len(level_rows),matrix.shape[1]),dtype=np.float64)
    seen=set();native_groups=[]
    for i,row in enumerate(index_rows):
        z=int(row['Z']);label=canonical_ion_label(row['ion_label'])
        if int(row['native_ion'])<1 or int(row['full_level'])<1 or int(row['superlevel'])<1:
            raise ValueError('native ion/FL/SL index가 양수가 아님')
        edge=float(row['edge_native']);statistical_weight=float(row['g'])
        if (not np.isfinite(edge) or not np.isfinite(statistical_weight) or
                statistical_weight<=0):
            raise ValueError('native edge/g index 값 오류')
        if (z,label) not in labels:
            raise ValueError('native ion label을 MODEL_SPEC에 대응할 수 없음')
        stage=labels[z,label]
        key=(z,stage,int(row['full_level']))
        if key in seen or key not in lookup:
            raise ValueError('native FL 키가 중복되거나 levels.csv에 없음')
        seen.add(key)
        target=lookup[key]
        if int(row['superlevel'])!=int(level_rows[target]['super_level'])+1:
            raise ValueError('native F_TO_S와 levels.csv superlevel 불일치')
        if int(row['eq_sl'])<1:
            raise ValueError('native 전역 SL 식 번호가 유효하지 않음')
        aligned[target]=matrix[i]
        native_groups.append((target,z,stage,label,int(row['full_level'])))
    if len(seen)!=len(lookup):
        raise ValueError('native index가 MODEL_SPEC 전체 FL을 덮지 않음')
    return aligned,native_groups


def error_metrics(left,right):
    if left.shape!=right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('비교 배열 형상/유한성 오류')
    difference=np.abs(left-right)
    scale=np.maximum(np.abs(left),np.abs(right))
    relative=np.divide(difference,scale,out=np.zeros_like(difference),where=scale>0)
    flat=int(np.argmax(relative))
    row,point=np.unravel_index(flat,relative.shape)
    return dict(shape=list(left.shape),exact_values=int(np.count_nonzero(left==right)),
        values=int(left.size),max_absolute=float(difference.max()),
        max_scaled_relative=float(relative.max()),
        median_scaled_relative=float(np.median(relative)),
        p95_scaled_relative=float(np.quantile(relative,0.95)),
        worst_row=int(row),worst_point=int(point+1),
        worst_left=float(left[row,point]),worst_right=float(right[row,point]))


def compare_heat_row(native_row,balance_path,expected_balance_sha256):
    """Compare the captured native RE row with the selected Lumina native ledger.

    This is a raw, same-state diagnostic. The native and Lumina frequency
    quadratures are not assumed identical, and no acceptance threshold is added.
    """
    balance_path=Path(balance_path)
    expected=expected_balance_sha256.lower()
    if len(expected)!=64 or any(ch not in '0123456789abcdef' for ch in expected):
        raise ValueError('기대 Lumina balance SHA256 형식 오류')
    if sha(balance_path)!=expected:
        raise ValueError('Lumina balance가 선택 물질의 봉인 해시와 불일치')
    with np.load(balance_path,allow_pickle=False) as archive:
        required=('native_instantaneous_scenario_net','native_instantaneous_scenario_gross')
        if any(name not in archive.files for name in required):
            raise ValueError('Lumina balance에 native 순간 열정책 net/gross가 없음')
        lumina=np.asarray(archive[required[0]],dtype=np.float64)
        gross=np.asarray(archive[required[1]],dtype=np.float64)
    native=np.asarray(native_row['steq_t_native'],dtype=np.float64)
    if (native.ndim!=1 or lumina.shape!=native.shape or gross.shape!=native.shape or
            not np.isfinite([native,lumina,gross]).all() or np.any(gross<0)):
        raise ValueError('native/Lumina 실제 열행의 형상·유한성·부호 오류')
    delta=native-lumina
    scale=np.maximum.reduce((gross,np.abs(native),np.abs(lumina)))
    scaled=np.divide(np.abs(delta),scale,out=np.zeros_like(delta),where=scale>0)
    worst=int(np.argmax(scaled))
    points=[dict(point_1based=i+1,native_steq_t=float(native[i]),
                  lumina_native_net=float(lumina[i]),difference=float(delta[i]),
                  lumina_native_gross=float(gross[i]),scaled_absolute_difference=float(scaled[i]))
            for i in range(len(native))]
    return dict(scope='same-state raw native STEQ_T vs Lumina native_instantaneous_scenario_net; diagnostic only',
        native_frequency_count=int(native_row['frequencies']),
        lumina_balance_sha256=expected,
        max_absolute_difference=float(np.max(np.abs(delta))),
        rms_difference=float(np.sqrt(np.mean(delta*delta))),
        max_scaled_absolute_difference=float(scaled[worst]),
        median_scaled_absolute_difference=float(np.median(scaled)),
        p95_scaled_absolute_difference=float(np.quantile(scaled,0.95)),
        worst_point_1based=worst+1,points=points,
        interpretation='No pass threshold; native and Lumina frequency quadratures may differ.')


def read_index(path,nfull):
    with Path(path).open(newline='') as stream:
        reader=csv.DictReader(stream)
        required={'native_ion','Z','full_level','superlevel','eq_sl','ion_label','edge_native','g'}
        if set(reader.fieldnames or ())!=required:
            raise ValueError('native full-level index schema 오류')
        rows=list(reader)
    if len(rows)!=nfull:
        raise ValueError('native full-level index 행 수 오류')
    return rows


def compare(args):
    root=Path(args.native_root);out=Path(args.out)
    if out.exists():
        raise FileExistsError('기존 full-level 비교 결과를 덮어쓰지 않음')
    run_identity=verify_run_identity(args)
    native_full=read_matrix(root/'NATIVE_FULL_POP.bin')
    native_fraction=read_matrix(root/'NATIVE_LTE_FRACTION.bin')
    native_occupation=read_matrix(root/'NATIVE_OCCUPATION.bin')
    if not (native_full.shape==native_fraction.shape==native_occupation.shape):
        raise ValueError('native FL population/fraction/occupation 형상 불일치')
    nfull,nd=native_full.shape
    if nd!=90:
        raise ValueError('현재 비교 기준의 깊이점 수는 90이어야 함')
    index_rows=read_index(root/'NATIVE_FULL_INDEX.csv',nfull)
    phase=(root/'NATIVE_FULL_PHASE.txt').read_text().splitlines()
    if len(phase)<2 or phase[0]!=PHASE:
        raise ValueError('native full snapshot 단계 표식 오류')
    counter,nt,phase_nd,phase_nfull,npres=map(int,phase[1].split())
    if (nt<3 or phase_nd!=nd or phase_nfull!=nfull or npres<1 or counter<1):
        raise ValueError('native full snapshot 차원/평가 횟수 오류')

    supplied=read_state(args.expected_state)
    captured=read_state(root/'NATIVE_FULL_INPUT.bin')
    if captured['raw']!=supplied['raw']:
        raise ValueError('native full snapshot의 실제 입력 상태가 승인 입력과 바이트 불일치')
    if (captured['nt']!=nt or captured['nd']!=nd or
            captured['nt']!=supplied['nt'] or captured['nd']!=supplied['nd']):
        raise ValueError('native full snapshot의 입력 형상 불일치')
    heat=read_heat_row(root/'NATIVE_HEAT_ROW.bin')
    if (heat['evaluation_counter']!=counter or heat['pops'].shape!=captured['pops'].shape or
            not np.array_equal(heat['radius_native'],captured['radius']) or
            not np.array_equal(heat['pops'],captured['pops'])):
        raise ValueError('full snapshot과 열행이 같은 native 상태/평가가 아님')
    expected_response=args.expected_response_sha256.lower()
    if len(expected_response)!=64 or any(ch not in '0123456789abcdef' for ch in expected_response):
        raise ValueError('기대 수용 Lumina response SHA256 형식 오류')
    if sha(args.response)!=expected_response:
        raise ValueError('수용 Lumina response가 봉인된 선택 응답과 불일치')
    native_heat_comparison=compare_heat_row(
        heat,args.lumina_balance,args.expected_lumina_balance_sha256)

    with Path(args.levels).open(newline='') as stream:
        level_rows=list(csv.DictReader(stream))
    models=load_cmfgen_model_spec(Path(args.model_spec),Path(args.atomic_links))
    lookup,labels=level_layout(level_rows,models)
    aligned_full,groups=align_native(native_full,index_rows,level_rows,lookup,labels)
    if len({int(row['native_ion']) for row in index_rows})!=npres:
        raise ValueError('native active-ion 수와 full-level index 불일치')
    aligned_fraction,_=align_native(native_fraction,index_rows,level_rows,lookup,labels)
    aligned_occupation,_=align_native(native_occupation,index_rows,level_rows,lookup,labels)
    c_full=read_matrix(args.c_full)
    if c_full.shape!=native_full.shape:
        raise ValueError('C-imported full 형상이 native FL 행렬과 다름')
    c_occupation=read_occupation(args.c_occupation,nfull,nd)
    with np.load(args.response,allow_pickle=False) as response:
        fraction=np.asarray(response['fraction'],dtype=np.float64)
        occupation=np.asarray(response['occupation'],dtype=np.float64)
    if (fraction.shape!=native_full.shape or occupation.shape!=native_full.shape or
            not np.isfinite(fraction).all() or not np.isfinite(occupation).all() or
            np.any(fraction<0) or np.any(occupation<0)):
        raise ValueError('수용 Lumina fraction/occupation 형상 또는 값 오류')

    comparisons={
        'native_full_vs_C_imported_full':error_metrics(aligned_full,c_full),
        'native_LTE_fraction_vs_Lumina_fraction':error_metrics(aligned_fraction,fraction),
        'native_occupation_vs_Lumina_occupation':error_metrics(aligned_occupation,occupation),
        'C_material_occupation_vs_Lumina_response':error_metrics(c_occupation,occupation),
    }
    ion_groups={}
    for row,z,stage,label,full_level in groups:
        ion_groups.setdefault((z,stage,label),[]).append(row)
    by_ion={}
    for (z,stage,label),rows in sorted(ion_groups.items()):
        selected=np.asarray(rows,dtype=np.int64)
        by_ion[f'{label}:Z{z}:stage{stage}']=error_metrics(
            aligned_full[selected],c_full[selected])

    diff=np.abs(aligned_full-c_full)
    scale=np.maximum(np.abs(aligned_full),np.abs(c_full))
    relative=np.divide(diff,scale,out=np.zeros_like(diff),where=scale>0)
    flat=np.argpartition(relative.ravel(),-min(10,relative.size))[-min(10,relative.size):]
    worst=[]
    for offset in sorted(flat,key=lambda i:relative.ravel()[i],reverse=True):
        row,point=np.unravel_index(int(offset),relative.shape)
        item=level_rows[row]
        worst.append(dict(atomic_number=int(item['atomic_number']),
            charge=int(item['ion_number']),full_level=int(item['level_number'])+1,
            point_index=int(point+1),native=float(aligned_full[row,point]),
            lumina=float(c_full[row,point]),absolute=float(diff[row,point]),
            scaled_relative=float(relative[row,point])))

    input_paths={
        'native_full_population':root/'NATIVE_FULL_POP.bin',
        'native_lte_fraction':root/'NATIVE_LTE_FRACTION.bin',
        'native_occupation':root/'NATIVE_OCCUPATION.bin',
        'native_index':root/'NATIVE_FULL_INDEX.csv',
        'native_state_snapshot':root/'NATIVE_FULL_INPUT.bin',
        'native_heat_row':root/'NATIVE_HEAT_ROW.bin',
        'native_executable':run_identity['executable'],
        'build_manifest':run_identity['build_manifest'],
        'fixed_J_field_manifest':run_identity['field_manifest'],
        'fixed_J_field_factor':run_identity['field_factor'],
        'fixed_J_field_info':run_identity['field_info'],
        'native_VADAT':run_identity['native_vadat'],
        'expected_state':Path(args.expected_state),
        'C_imported_full':Path(args.c_full),
        'C_material_occupation':Path(args.c_occupation),
        'accepted_Lumina_response':Path(args.response),
        'selected_Lumina_balance':Path(args.lumina_balance),
        'atomic_levels':Path(args.levels),
        'MODEL_SPEC':Path(args.model_spec),
        'atomic_links':Path(args.atomic_links),
    }
    report=dict(schema='lumina-cmfgen-fixedj-full-comparison-v1',
        scope='실제 native fixed-J 상태의 전체 FL 진단 비교; 사전 동질성 합격선 없음',
        native_evaluation_counter=counter,active_ions=npres,full_levels=nfull,points=nd,
        fixed_J_source_name=run_identity['fixed_J_source_name'],
        fixed_J_source_sha256=run_identity['fixed_J_source_sha256'],
        exact_injected_state=True,heat_row_same_state=True,
        comparisons=comparisons,native_heat_comparison=native_heat_comparison,
        full_population_by_ion=by_ion,
        largest_full_population_differences=worst,
        physical_equivalence='DIAGNOSTIC_ONLY_NO_PREDECLARED_THRESHOLD',
        final_goal='NOT_MET',input_sha256={key:identity(path) for key,path in input_paths.items()})
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('native-root','build-manifest','executable','field-manifest',
                 'field-root','expected-j-sha256','expected-vadat-sha256',
                 'expected-state','c-full','c-occupation','response','expected-response-sha256',
                 'lumina-balance','expected-lumina-balance-sha256',
                 'levels','model-spec','atomic-links','out'):
        parser.add_argument('--'+name,required=True,
            type=argument_type(name))
    arguments=parser.parse_args()
    result=compare(arguments)
    print(json.dumps({'full_levels':result['full_levels'],'points':result['points'],
        'exact_injected_state':result['exact_injected_state'],
        'heat_row_same_state':result['heat_row_same_state'],
        'comparisons':result['comparisons'],
        'native_heat_comparison':result['native_heat_comparison'],
        'physical_equivalence':result['physical_equivalence'],
        'final_goal':result['final_goal']},ensure_ascii=False,indent=2))
