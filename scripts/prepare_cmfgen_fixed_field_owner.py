#!/usr/bin/env python3
"""완료된 native 고정-J 물질 record와 같은 J60의 새 C owner 입력을 만든다.

진행 중인 SCRTEMP 전체를 완료 자료로 취급하지 않는다. POINT1/2가 모두 가리킨
완료 record의 고정 byte 구간만 두 번 읽어 보존한다. 장 block은 이미 검증한
입력의 원래 bytes를 유지하며 새 인구/T/ne만 바꾼다. native는 읽기 전용이다.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import struct

import numpy as np

from configure_cmfgen_stageA import set_control
from ptf_gated_metrics import REC_BYTES,DOUBLES_PER_REC,SCRTEMP_HEADER_RECS,read_dimensions,read_variables
from run_cmfgen_lambda_probe import sha


def point_records(run):
    values={}
    for name in ('POINT1','POINT2'):
        fields=(run/name).read_text().splitlines()[1].split()
        if len(fields)!=5 or fields[2]!='1':raise ValueError('일반 선형 SCRTEMP 포인터 필요')
        values[name]=int(fields[0])
    return values


def record_bytes(path,record,nd,nt):
    if record<1:raise ValueError('양의 완료 record 필요')
    nrec=(nd*nt+DOUBLES_PER_REC-1)//DOUBLES_PER_REC
    offset=(SCRTEMP_HEADER_RECS+(record-1)*nrec)*REC_BYTES
    with path.open('rb') as stream:
        header=stream.read(SCRTEMP_HEADER_RECS*REC_BYTES)
        stream.seek(offset);raw=stream.read(nrec*REC_BYTES)
    if len(header)!=SCRTEMP_HEADER_RECS*REC_BYTES or len(raw)!=nrec*REC_BYTES:
        raise ValueError('완료 record 저장 구간이 짧음')
    state=np.frombuffer(raw,dtype='<f8')[:nd*nt].reshape(nd,nt).T.copy()
    if not np.isfinite(state).all() or np.any(state[:-2]<0) or np.any(state[-2:]<=0):
        raise ValueError('비유한/음수 완료 인구 또는 T/ne')
    return header,raw,state


def snapshot(run,seed,record):
    nd,nt=read_dimensions(run)
    before=point_records(run)
    if min(before.values())<record:raise ValueError('아직 완료되지 않은 물질 record')
    header,raw,state=record_bytes(run/'SCRTEMP',record,nd,nt)
    header2,raw2,_=record_bytes(run/'SCRTEMP',record,nd,nt)
    after=point_records(run)
    if min(after.values())<record or raw!=raw2 or header!=header2:
        raise ValueError('읽는 동안 완료 record가 변경됨')
    old_header,_,old=record_bytes(seed/'SCRTEMP',59,nd,nt)
    if header!=old_header:raise ValueError('고정 물질 격자 header 변경')
    if not np.array_equal(state[-1],old[-1]):raise ValueError('고정T가 아님')
    return state,old,dict(record=record,depths=nd,variables=nt,point_before=before,point_after=after,
        record_storage_sha256=hashlib.sha256(raw).hexdigest(),header_sha256=hashlib.sha256(header).hexdigest())


def replace_material(base,out,state,old,indexes):
    with base.open('rb') as src,out.open('xb') as dst:
        header=src.read(struct.calcsize('<8sIIIId'))
        magic,nd,nn,nc,nb,epoch=struct.unpack('<8sIIIId',header)
        if magic!=b'CMFOWN01' or nd!=state.shape[1] or nc!=len(indexes):raise ValueError('원래 입력 차원 오류')
        frequency=src.read(8*nn);colraw=src.read(12*nc)
        columns=np.frombuffer(colraw,dtype='<i4').reshape(nc,3).copy()
        if len(frequency)!=8*nn:raise ValueError('부분 주파수')
        dst.write(header);dst.write(frequency);dst.write(colraw)
        field_hash=hashlib.sha256();new_hash=hashlib.sha256();profiles=[]
        field_size=12*nb+8*nn
        for depth in range(nd):
            physical=src.read(40);pop_raw=src.read(8*nc);field=src.read(field_size)
            if len(physical)!=40 or len(pop_raw)!=8*nc or len(field)!=field_size:
                raise ValueError('부분 owner 입력 block')
            p=np.frombuffer(physical,dtype='<f8').copy();previous=np.frombuffer(pop_raw,dtype='<f8')
            if (not np.array_equal(previous,old[indexes,depth]) or
                p[1]!=old[-1,depth]*1e4 or p[2]!=old[-2,depth]):
                raise ValueError('원래59 인구/T/ne와 baseline 불일치')
            p[1]=state[-1,depth]*1e4;p[2]=state[-2,depth];profiles.append(p)
            dst.write(p.astype('<f8').tobytes());dst.write(state[indexes,depth].astype('<f8').tobytes())
            dst.write(field);field_hash.update(field)
        if src.read(1):raise ValueError('owner 입력 후행 데이터')
    # producer가 실제 쓴 장 block도 독립적으로 읽어 원래 J60 bytes와 비교한다.
    with out.open('rb') as stream:
        stream.seek(len(header)+len(frequency)+len(colraw))
        for depth in range(nd):
            stream.seek(40+8*nc,1);new_hash.update(stream.read(field_size))
    if new_hash.digest()!=field_hash.digest():raise ValueError('고정 복사장 변경')
    return columns,np.array(profiles),dict(native_frequencies=nn,epoch_seconds=epoch,
        fixed_radiation_blocks_sha256=field_hash.hexdigest(),radiation_blocks_byte_equal=True)


def prepare(run,base,record,out):
    if out.exists():raise ValueError('기존 완료 물질 입력 보존')
    old_manifest=json.loads((base/'pair-inputs.json').read_text())
    fixed=json.loads((run/'seq_logs/fixed-field-inputs.json').read_text());seed=Path(fixed['seed'])
    if Path(old_manifest['seed'])!=seed or fixed['field']!=old_manifest['run']:
        raise ValueError('다른 물질/복사장 계보')
    for name in ('MODEL_SPEC','LEVEL_SL_STEQ_LINKS'):
        expected=old_manifest['input_sha256'][str(seed/name)]
        if sha(run/name)!=expected:raise ValueError('원자모형/준위 대응 변경: '+name)
    for name,h in fixed['input_controls_sha256'].items():
        if sha(run/name)!=h:raise ValueError('실행 제어 변경: '+name)
    if set_control((run/'VADAT').read_text(),'USE_FIXED_J','F')!=(seed/'VADAT').read_text():
        raise ValueError('고정-J 이외 물리 제어 변경')
    for name,h in fixed['fixed_field_inputs'].items():
        if sha(run/name)!=h:raise ValueError('외부장 변경: '+name)
    original=base/'J60_owner_input.bin'
    if sha(original)!=old_manifest['output_sha256'][original.name]:raise ValueError('기존 입력 변경')
    deck=Path(old_manifest['deck']);verified_view={}
    for name,h in old_manifest['input_sha256'].items():
        path=Path(name)
        if path.parent==deck:
            if sha(path)!=h:raise ValueError('기존 원자 view 변경: '+name)
            verified_view[name]=h
    if not verified_view:raise ValueError('봉인 원자 view 신원 없음')
    state,previous,identity=snapshot(run,seed,record)
    variables=read_variables(run,state.shape[0]-2)
    lookup={((14 if v.species=='Si' else 16),v.charge,v.superlevel-1):v.index-1
            for v in variables if v.species in ('Si','S')}
    with original.open('rb') as stream:
        header=stream.read(struct.calcsize('<8sIIIId'));_,nd,nn,nc,nb,epoch=struct.unpack('<8sIIIId',header)
        stream.seek(8*nn,1);columns=np.frombuffer(stream.read(12*nc),dtype='<i4').reshape(nc,3)
    if len(lookup)!=len(columns) or set(lookup)!=set(map(tuple,columns)):
        raise ValueError('기존 SL/terminal 집합 불일치')
    indexes=np.array([lookup[tuple(col)] for col in columns],dtype=int)
    out.mkdir();columns,profiles,field_identity=replace_material(original,out/'owner_input.bin',state,previous,indexes)
    np.savez(out/'completed-state.npz',state=state,previous59=previous,owner_variable_indexes=indexes,profiles=profiles,columns=columns)
    expected_ne=sum(v.charge*state[v.index-1] for v in variables)
    charge=abs(expected_ne-state[-2])/state[-2]
    elements={}
    for species in sorted({v.species for v in variables}):
        selected=[v.index-1 for v in variables if v.species==species]
        elements[species]=float(np.max(abs(state[selected].sum(axis=0)/previous[selected].sum(axis=0)-1)))
    with (out/'reference.csv').open('x',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['depth_1based','Z','stage','population_cm3','fraction','Te_K','ne_cm3'])
        for depth in range(nd):
            for Z in (14,16):
                select=np.array([i for i,col in enumerate(columns) if col[0]==Z]);total=state[indexes[select],depth].sum()
                for charge_number in range(1,6):
                    selected=[indexes[i] for i in select if columns[i,1]==charge_number]
                    density=state[selected,depth].sum()
                    writer.writerow([depth+1,Z,charge_number+1,density,density/total,profiles[depth,1],profiles[depth,2]])
    source=Path(__file__).resolve();shutil.copy2(source,out/source.name)
    report=dict(scope=__doc__,material_snapshot=identity,field_identity=field_identity,
        completed_record_only=True,run=str(run),base_input=str(base),deck=old_manifest['deck'],
        max_ne_change_over59=float(np.max(abs(state[-2]/previous[-2]-1))),
        max_native_charge_residual=float(charge.max()),element_total_max_relative_change=elements,
        unchanged_atomic_view_sha256=verified_view,
        temperature_unchanged=True,conditional_fixed_J_not_joint_convergence=True,
        input_sha256={str(p):sha(p) for p in (base/'pair-inputs.json',original,run/'seq_logs/fixed-field-inputs.json',source)},
        output_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()},
        native_full_SE_residual='NOT_YET_EXTRACTED_NOT_REPLACED_BY_CORRECTION',physical_equivalence='NOT_JUDGED',final_goal='NOT_JUDGED')
    (out/'fixed-field-owner-inputs.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('material_snapshot','max_ne_change_over59','max_native_charge_residual','element_total_max_relative_change')},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run','base','out'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--record',type=int,required=True)
    a=parser.parse_args();prepare(a.run.resolve(),a.base.resolve(),a.record,a.out.resolve())
