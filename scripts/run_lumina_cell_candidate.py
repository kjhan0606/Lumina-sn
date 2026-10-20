#!/usr/bin/env python3
"""같은 물질/주파수의 실제 계수를 셀 수송 후보 실행기에 연결한다.

기본은 입력/메모리 사전 확인이다. 구현된 방정식은 정상 homologous Iν·3a이며,
native 상대론/이류 MODEL을 이 방정식으로 대체 실행할 수 없다. 후보 출력은
CMFGEN 동등성·전 모델 공동수렴·최종 스펙트럼 합격으로 표시하지 않는다.
"""
from __future__ import annotations
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from cmfgen_model_sources import control

FRAME='STATIONARY_HOMOLOGOUS_I_NU_3A'
COMPONENTS=('bf_chi','bf_eta','ff_chi','ff_eta','bb_chi','bb_eta','es_chi')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def identity(report,path):
    path=Path(path).resolve()
    if report['input_sha256'].get(str(path))!=sha(path):
        raise ValueError(f'계수와 입력 신원 불일치: {path}')


def read_material(path,magic):
    path=Path(path).resolve()
    with path.open('rb') as stream:
        tag,ns,nc,nl,epoch=struct.unpack('<8sIIId',stream.read(28))
    stride=6 if magic==b'LUMCEL01' else 4
    if tag!=magic or not 1<=ns<=2**31-18 or not nc or not nl or not np.isfinite(epoch) or epoch<=0:
        raise ValueError('물질 header 손상')
    if path.stat().st_size!=28+12*nc+8*ns*(stride+nc):raise ValueError('물질 길이 불일치')
    physical=np.memmap(path,mode='r',dtype='<f8',offset=28+12*nc,shape=(ns,stride))
    if not np.isfinite(physical).all() or np.any(physical<=0):raise ValueError('물질 물리값 손상')
    if magic==b'LUMCEL01':
        if np.any(physical[:,1]<=physical[:,0]) or not np.array_equal(physical[1:,0],physical[:-1,1]):
            raise ValueError('셀 경계 공백/중첩/순서 오류')
        if not np.all(physical[:,5]==1):raise ValueError('현재 후보는 smooth 물질만 지원')
        t,ne=physical[:,2],physical[:,3]
    else:
        if np.any(np.diff(physical[:,0])>=0):raise ValueError('원래 점의 반경 순서 오류')
        t,ne=physical[:,1],physical[:,2]
    return dict(path=path,ns=ns,nc=nc,nl=nl,epoch=epoch,physical=physical,t=t,ne=ne)


def read_sources(folder,material,model):
    folder=Path(folder).resolve();report=json.loads((folder/'sources.json').read_text())
    if report.get('schema')!='lumina-model-local-sources-v1':raise ValueError('계수 schema 불일치')
    identity(report,material['path']);identity(report,model)
    binary=folder/'sources.bin'
    if sha(binary)!=report['output_sha256']['sources.bin']:raise ValueError('계수 binary 신원 변경')
    with binary.open('rb') as stream:tag,ns,nf=struct.unpack('<8sII',stream.read(16))
    if tag!=b'LMPHYS01' or ns!=material['ns'] or not 2<=nf<=2**31-1:
        raise ValueError('계수/물질 차원 불일치')
    if report['cells']!=ns or report['frequencies']!=nf or report['full_levels']!=material['nl']:
        raise ValueError('계수 manifest 차원 불일치')
    size=nf+2*ns+7*ns*nf
    if binary.stat().st_size!=16+8*size:raise ValueError('계수 binary 길이 불일치')
    data=np.memmap(binary,mode='r',dtype='<f8',offset=16,shape=(size,))
    nu=data[:nf];t=data[nf:nf+ns];ne=data[nf+ns:nf+2*ns]
    if not np.array_equal(t,material['t']) or not np.array_equal(ne,material['ne']):
        raise ValueError('같은 물질의 T/ne가 아님')
    if not np.isfinite(nu).all() or np.any(nu<=0) or np.any(np.diff(nu)<=0):
        raise ValueError('주파수 순서/값 오류')
    dlog=float(np.log(nu[1]/nu[0]))
    if np.max(abs(np.log(nu[1:]/nu[:-1])-dlog))>64*np.finfo(float).eps:
        raise ValueError('현재 수송 API는 log-uniform 격자가 필요함')
    component=data[nf+2*ns:].reshape(7,ns,nf)
    for i,name in enumerate(COMPONENTS):
        flat=component[i].reshape(-1)
        for begin in range(0,len(flat),131072):
            block=flat[begin:begin+131072]
            if not np.isfinite(block).all() or (name not in ('bf_chi','bb_chi') and np.any(block<0)):
                raise ValueError(f'계수 성분 손상: {name}')
    return dict(folder=folder,binary=binary,report=report,ns=ns,nf=nf,nu=nu,dlog=dlog,component=component)


def memory_inventory(material,sources,boundary_material,boundary_sources):
    ns,nf=sources['ns'],sources['nf'];nr=ns+17;stride=ns+1;n=ns*nf
    batch=max(1,min(16,nr,(8*1024**3)//8//2//nf//stride))
    beta=2*float(material['physical'][-1,1])/(2.99792458e10*material['epoch']*sources['dlog'])
    if not np.isfinite(beta) or beta>=2**31-2:raise ValueError('주파수 이동 인덱스 표현 범위 초과')
    # 동시 보유를 보수적으로 합한다. 배열 크기 계산이지 실제 peak RSS 측정이 아니다.
    parts=dict(source_maps=sources['binary'].stat().st_size+boundary_sources['binary'].stat().st_size,
        material_maps=material['path'].stat().st_size+boundary_material['path'].stat().st_size,
        J_and_error=2*n*8,raw_assembly_with_private_workspace=4*n*8,
        solver_cell_work=5*n*8,previous_angular_samples=n*8,
        envelope_verification_and_refinement=3*n*8,
        directional_ray_buffers=2*batch*stride*nf*8,inner_ray_boundary=nr*nf*8,
        segment_temporaries=batch*(nf*8+64*(int(beta)+2)),
        boundary_work_and_quadrature=8*nf*8,geometry_reserve=64*nr*stride+64*(ns+nr),
        validation_chunk_reserve=8*1024**2,runtime_reserve=256*1024**2)
    return dict(known_allocation_bound_bytes=sum(parts.values()),parts=parts,ray_batch=batch,
        whole_process_peak_RSS_verified=False,includes_full_readonly_maps=True,
        approximation='conservative simultaneous known allocations plus explicit runtime reserve')


def prepare(cells,sources,seed,boundary_sources,model,memory_gib=32):
    model=Path(model).resolve();cells=Path(cells).resolve();seed=Path(seed).resolve()
    material=read_material(cells/'material_cells.bin',b'LUMCEL01')
    native=read_material(seed/'native_material_nodes.bin',b'LUMNOD01')
    manifest=json.loads((cells/'material-cells.json').read_text())
    if sha(material['path'])!=manifest['output_sha256']['material_cells.bin']:
        raise ValueError('셀 물질 manifest 신원 불일치')
    cell_source=read_sources(sources,material,model)
    inner_source=read_sources(boundary_sources,native,model)
    if material['epoch']!=native['epoch'] or not np.array_equal(cell_source['nu'],inner_source['nu']):
        raise ValueError('경계/셀의 epoch 또는 주파수 불일치')
    if float(material['physical'][0,0])!=float(native['physical'][-1,0]):
        raise ValueError('실제 내부 경계 반경 불일치')
    temperature=float(manifest['inner_gas_temperature_K']);luminosity=float(manifest['inner_luminosity_erg_s'])
    if manifest['boundary_method']!='DIFFUSION' or temperature!=float(native['t'][-1]):
        raise ValueError('원래 확산 경계/온도 불일치')
    if not np.isfinite(luminosity) or luminosity<=0:raise ValueError('경계 광도 손상')
    flags={key:control(model,key) for key in ('USE_J_REL','INCL_REL','INCL_ADV_TRANS','COH_ES')}
    if any(value not in ('T','F') for value in flags.values()):raise ValueError('MODEL 논리값 손상')
    reasons=[]
    if any(flags[key]=='T' for key in ('USE_J_REL','INCL_REL','INCL_ADV_TRANS')):
        reasons.append('FRAME_NOT_IMPLEMENTED: native 상대론/이류 MODEL과 정상 homologous 3a 후보는 다름')
    if flags['COH_ES']!='T':reasons.append('NONCOHERENT_SCATTERING_NOT_IMPLEMENTED')
    memory=memory_inventory(material,cell_source,native,inner_source)
    if not np.isfinite(memory_gib) or memory_gib<=0 or memory_gib>32:
        raise ValueError('봉인32GiB 이내의 양의 메모리 한도 필요')
    if memory['known_allocation_bound_bytes']>memory_gib*1024**3:reasons.append('MEMORY_BUDGET_EXCEEDED')
    input_paths=[model,material['path'],native['path'],cells/'material-cells.json',
        cell_source['folder']/'sources.json',cell_source['binary'],
        inner_source['folder']/'sources.json',inner_source['binary']]
    report=dict(schema='lumina-cell-candidate-preflight-v1',implemented_frame=FRAME,model_flags=flags,
        cells=material['ns'],frequencies=cell_source['nf'],epoch_seconds=material['epoch'],
        radius_domain_cm=[float(material['physical'][0,0]),float(material['physical'][-1,1])],
        boundary_temperature_K=temperature,boundary_luminosity_erg_s=luminosity,
        boundary_sample='original innermost native node; not first-cell averaged material',
        memory=memory,execution_eligible=not reasons,reasons=reasons,
        frequency_resolution_certified=False,physical_equivalence='NOT_JUDGED',final_goal='NOT_JUDGED',
        input_sha256={str(p):sha(p) for p in input_paths},authoritative_fields_executed=0)
    return dict(material=material,sources=cell_source,boundary=inner_source,
        temperature=temperature,luminosity=luminosity,report=report)


def execute_candidate(prepared,library,iteration_cap=200,tolerance=1e-10):
    if not prepared['report']['execution_eligible']:
        raise ValueError('; '.join(prepared['report']['reasons']))
    if iteration_cap<2 or not np.isfinite(tolerance) or tolerance<=0:raise ValueError('반복 기준 손상')
    material=prepared['material'];sources=prepared['sources'];ns,nf=sources['ns'],sources['nf']
    component=prepared['boundary']['component'][:,-1,:]
    chi=np.zeros(nf);eta=np.zeros(nf)
    for index in (0,2,4,6):np.add(chi,component[index],out=chi)
    for index in (1,3,5):np.add(eta,component[index],out=eta)
    if np.any(chi<=0):raise ValueError('확산 경계의 실제 총소광은 양수여야 함')
    ri=np.ascontiguousarray(material['physical'][:,0]);ro=np.ascontiguousarray(material['physical'][:,1])
    j=np.zeros((ns,nf));error=np.zeros_like(j);counts=np.zeros(4,dtype='u8');diagnostics=np.zeros(4)
    dll=ct.CDLL(str(Path(library).resolve(strict=True)))
    f=dll.lumina_cell_stationary_candidate;p=ct.POINTER(ct.c_double)
    f.argtypes=[ct.c_int,ct.c_int,ct.c_double,p,p,p,ct.c_double,ct.c_double,ct.c_double]+[p]*12+[
        ct.c_int,ct.c_double,ct.POINTER(ct.c_uint64),p]
    f.restype=ct.c_int
    def pointer(a):return a.ctypes.data_as(p)
    status=f(ns,nf,sources['dlog'],pointer(sources['nu']),pointer(ri),pointer(ro),
        material['epoch'],prepared['temperature'],prepared['luminosity'],pointer(chi),pointer(eta),
        pointer(component[6]),*[pointer(a) for a in sources['component']],pointer(j),pointer(error),
        iteration_cap,tolerance,counts.ctypes.data_as(ct.POINTER(ct.c_uint64)),pointer(diagnostics))
    if status:raise ValueError(f'후보 셀 수송 실패: status={status}')
    result=dict(frame=FRAME,iterations=int(counts[0]),rays=int(counts[1]),ray_buffer_slots=int(counts[2]),
        candidate_solver_calls=1,authoritative_field_role=False,
        componentwise_verified=bool(counts[3]),relative_change=float(diagnostics[0]),
        actual_K_norm_upper=float(diagnostics[1]),residual_upper=float(diagnostics[2]),
        error_upper_max=float(diagnostics[3]),library_sha256=sha(library),
        output_role='stationary candidate; not native relativistic field/equivalence proof')
    return j,error,result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('cells','sources','seed','boundary-sources','model','out'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--library',type=Path);parser.add_argument('--execute-candidate',action='store_true')
    parser.add_argument('--memory-GiB',type=float,default=32)
    a=parser.parse_args()
    if a.out.exists():raise ValueError('기존 후보 실행 산출물 보존')
    a.out.mkdir()
    result=dict(status='INVALID_INPUT',authoritative_fields_executed=0,final_goal='NOT_JUDGED')
    code=2
    try:
        prepared=prepare(a.cells,a.sources,a.seed,a.boundary_sources,a.model,a.memory_GiB)
        result=prepared['report'];result['status']='PREFLIGHT_ONLY'
        code=0 if result['execution_eligible'] else 3
        if a.execute_candidate:
            if not result['execution_eligible']:raise ValueError('; '.join(result['reasons']))
            if a.library is None:raise ValueError('후보 실행 library 필요')
            j,error,run=execute_candidate(prepared,a.library)
            np.savez(a.out/'candidate_field.npz',nu_Hz=prepared['sources']['nu'],J=j,error_upper=error)
            result.update(status='CANDIDATE_COMPLETED',candidate=run,
                output_sha256={'candidate_field.npz':sha(a.out/'candidate_field.npz')})
    except (ValueError,OSError,KeyError,struct.error) as exc:
        result.update(status='NOT_EXECUTED_OR_FAILED',failure=str(exc));code=2
    with (a.out/'preflight.json').open('x') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:result.get(k) for k in ('status','execution_eligible','reasons','failure')}))
    return code


if __name__=='__main__':raise SystemExit(main())
