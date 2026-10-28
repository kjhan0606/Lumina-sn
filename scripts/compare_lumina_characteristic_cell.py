#!/usr/bin/env python3
"""같은 생산 계수/반경 함수에서 실제 이동ν의 fullSR 셀을 독립 ODE와 대조한다.

legacy 물질70의 고정 J source다. 고정-blue SC식/CMFGEN 수치에 맞추지 않으며
새 산란/물질/전대역 수송이나 observer를 실행하지 않는다.
"""
import argparse
from contextlib import ExitStack
import ctypes as ct
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import time

import numpy as np
from scipy.integrate import solve_ivp

from analyze_cmfgen_formal_capture import geometry
from lumina_linear_native import DenseView
from lumina_fullsr_frozen import FrozenKernel
from lumina_fullsr_sc import SCALED
import lumina_outer_atmosphere as A
from prepare_lumina_outer_input import read_auxiliary, sha
from run_cmfgen_formal_primary import fields

ROOT = Path(__file__).resolve().parents[1]


class Sampler:
    def __init__(self, library, frequency, chi, es, eta, source_j):
        self.frequency = np.asarray(frequency, dtype='f8')
        n = len(self.frequency)
        self.canonical = np.ascontiguousarray([chi,es,eta],dtype='f8')
        self.j = np.ascontiguousarray(source_j,dtype='f8')
        if (n<2 or self.canonical.shape!=(3,2,n) or self.j.shape!=(2,n) or
                not np.isfinite(self.canonical).all() or not np.isfinite(self.j).all() or
                np.any(self.canonical[1]<=0) or np.any(self.canonical[2]<0) or np.any(self.j<0) or
                not np.isfinite(self.frequency).all() or np.any(self.frequency<=0) or
                np.any(np.diff(self.frequency)<=0)):
            raise ValueError('같은 두 native 점의 유한 계수/J와 증가 주파수 필요')
        self.log = np.log(self.frequency);self.h=(self.log[-1]-self.log[0])/(n-1)
        if np.max(abs(self.log-(self.log[0]+self.h*np.arange(n))))>2.e-13:
            raise ValueError('실제 생산의 균일 logν 격자 필요')
        self.outer=np.zeros(10);self.radius=np.array([2.,1.])
        pointer=lambda x:x.ctypes.data_as(ct.POINTER(ct.c_double))
        self.view=DenseView(2,n,2,1,1,pointer(self.canonical),pointer(self.outer),
                            pointer(self.j),pointer(self.radius),pointer(self.radius))
        self.library=ct.CDLL(str(library));self.function=self.library.lumina_characteristic_samples
        self.function.argtypes=[ct.c_int,ct.POINTER(DenseView)]+[ct.c_void_p]*4
        self.function.restype=ct.c_int

    def at(self, weight, nu):
        w,nu=np.broadcast_arrays(np.asarray(weight,dtype='f8'),np.asarray(nu,dtype='f8'))
        if (np.any(nu<self.frequency[0]) or np.any(nu>self.frequency[-1]) or
                not np.isfinite(nu).all() or np.any(nu<=0)):
            raise ValueError('주파수 외삽 없는 같은 계수 필요')
        w=np.ascontiguousarray(w.reshape(-1));column=np.ascontiguousarray((np.log(nu).reshape(-1)-self.log[0])/self.h)
        # 수학적으로 같은 원 끝점을 정확한 native index로 주소화한다. 외삽/clip 아님.
        column[nu.reshape(-1)==self.frequency[0]]=0.
        column[nu.reshape(-1)==self.frequency[-1]]=len(self.frequency)-1
        chi=np.empty(len(w));eta=np.empty(len(w))
        code=self.function(len(w),ct.byref(self.view),w.ctypes.data,column.ctypes.data,
                           chi.ctypes.data,eta.ctypes.data)
        if code:raise ValueError('생산 계수 표본화 오류: '+str(code))
        return chi,eta


@dataclass
class Cell:
    sampler: Sampler
    g: object
    inner: float
    outer: float
    impact: float
    exit_nu: float
    deadline: float = float('inf')

    def __post_init__(self):
        if not self.outer>self.inner>self.impact>=0:raise ValueError('외향 셀 기하 오류')
        self.z0=np.sqrt((self.inner-self.impact)*(self.inner+self.impact))
        self.z1=np.sqrt((self.outer-self.impact)*(self.outer+self.impact))
        self.exit_d=float(A.doppler(self.g,self.impact,self.z1,self.outer))
        self.slope=float((A.velocity_at(self.g,self.outer)-A.velocity_at(self.g,self.inner))/
                         (self.outer-self.inner)/A.C)

    def point(self,x):
        if time.monotonic()>=self.deadline:raise TimeoutError('셀 대조 전체 시간 한도')
        if not 0<=x<=1:raise ValueError('셀 좌표 범위 오류')
        z=self.z0+x*(self.z1-self.z0)
        r=self.inner if x==0 else self.outer if x==1 else float(np.hypot(self.impact,z))
        beta=float(A.velocity_at(self.g,r)/A.C);mu=z/r
        doppler=(1-beta*mu)/np.sqrt((1-beta)*(1+beta))
        log_gradient=beta*self.slope*mu/((1-beta)*(1+beta))-(self.slope*mu*mu+
            beta*self.impact**2/r**3)/(1-beta*mu)
        nu=self.exit_nu*doppler/self.exit_d
        chi,eta=self.sampler.at((r-self.inner)/(self.outer-self.inner),nu)
        return float(chi[0]),float(eta[0]),float(doppler),float(log_gradient),r,nu

    def mesh(self,n):
        x=np.linspace(0.,1.,n+1);values=[self.point(float(t)) for t in x]
        lengths=[];deltas=[]
        for k in range(n):
            z0=self.z0+x[k]*(self.z1-self.z0);z1=self.z0+x[k+1]*(self.z1-self.z0)
            length,delta=A.segment_geometry(self.g,self.impact,z0,z1,
                start_radius=self.inner if k==0 else None,end_radius=self.outer if k+1==n else None)
            lengths.append(length);deltas.append(delta)
        return np.array(lengths),np.array(deltas),np.array([[v[0]] for v in values]),np.array([[v[1]] for v in values])


def reference(cell,incoming,relative):
    """생산 맵을 호출하지 않는 실제 z좌표의 독립 적응 ODE 적분."""
    if not np.isfinite(incoming) or incoming<0 or not 0<relative<1:raise ValueError('독립 ODE 입력 오류')
    def rhs(x,y):
        chi,eta,doppler,gradient,_,_=cell.point(float(x));dz=cell.z1-cell.z0
        return [dz*(doppler*eta-(doppler*chi-3*gradient)*y[0])]
    result=solve_ivp(rhs,(0.,1.),[incoming],method='Radau',rtol=relative,
                     atol=max(incoming,1.e-8)*relative*1.e-4,max_step=1/128)
    if not result.success or not np.isfinite(result.y).all() or result.y[0,-1]<0:
        raise ValueError('독립 셀 ODE 미완료')
    return float(result.y[0,-1]),dict(evaluations=int(result.nfev),steps=len(result.t)-1)


def build(out):
    sources=[ROOT/'scripts/lumina_characteristic_cell.cpp',ROOT/'src/cmf_fullsr_frozen.cpp',ROOT/'src/cmf_fullsr_sc.cpp']
    library=out/'characteristic-cell.so'
    command=['c++','-O3','-std=c++17','-ffp-contract=off','-Wall','-Wextra','-Werror',
        '-fopenmp','-fPIC','-shared','-I'+str(ROOT/'src'),*map(str,sources),'-o',str(library)]
    result=subprocess.run(command,capture_output=True,text=True,timeout=45)
    if result.returncode:raise RuntimeError(result.stderr)
    return library,command,sources


def compare(legacy_run,capture,manifest,out,*,reuse_reference=None,geometry_library=None):
    if out.exists():raise FileExistsError('기존 실제 특성선 대조 보존')
    if any(os.environ.get(k)!='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):
        raise ValueError('소형 CPU1 환경 필요')
    if (reuse_reference is None)!=(geometry_library is None):raise ValueError('참조 재사용과 실제 phase 기하 바이너리가 함께 필요')
    started=time.monotonic();deadline=started+(75 if reuse_reference else 150)
    execution_path=legacy_run/'execution.json';execution=json.loads(execution_path.read_text())
    if (execution['status']!='NATIVE_POSITION_F_COMPLETE' or
        (execution['material_record'],execution['reference_record'])!=(70,71)):
        raise ValueError('완료 원70/J71 계수 기록 필요')
    sealed=execution['input_sha256'];sources=[Path(p) for p in sealed if Path(p).name=='sources.json']
    references=[Path(p) for p in sealed if Path(p).name=='FORMAL_PRIMARY_FIELDS.bin']
    if len(sources)!=1 or len(references)!=1:raise ValueError('원 계수/참조 출처 오류')
    source_path=sources[0];report=json.loads(source_path.read_text());reference_path=references[0]
    binary=source_path.with_name('sources.bin')
    grids=[Path(p) for p in report['input_sha256'] if Path(p).name=='frequency.npy']
    if len(grids)!=1:raise ValueError('원 주파수 출처 오류')
    grid_path=grids[0];geo_path=capture/'field/FORMAL_40286_00_geometry.bin'
    expected={source_path:sealed[str(source_path)],binary:sealed[str(binary)],
        reference_path:sealed[str(reference_path)],grid_path:report['input_sha256'][str(grid_path)],
        geo_path:json.loads(manifest.read_text())['file_sha256'][str(geo_path)]}
    for p,identity in expected.items():
        if sha(p)!=identity:raise ValueError('봉인 입력 변경: '+p.name)
    prior=None
    if reuse_reference is not None:
        if sha(reuse_reference)!='443ce33ba2cb7421390fb00396bd38c6372970a8d4f1a75e9910e2d21f5c76f2':
            raise ValueError('완료 독립 ODE 참조 변경')
        prior=json.loads(reuse_reference.read_text())
        if (prior['status']!='COMPLETE_FROZEN_SOURCE_CHARACTERISTIC_CELL' or
                prior['input_sha256']!={str(p):v for p,v in expected.items()}):
            raise ValueError('같은 입력의 완성 참조가 아님')
        if sha(geometry_library)!='3cddc57f07232c92af77d7647f137d6f0f63904024959b34345e487da9bec2d0':
            raise ValueError('현재 F2의 C phase 기하 바이너리와 다름')
    raw,field=fields(reference_path)
    try:
        grid=np.load(grid_path,allow_pickle=False);chi,es,eta=read_auxiliary(binary,report,grid,[49,50])
        log_ref=np.log(np.asarray(field['freq'][::-1])*1.e15)
        j=np.array([np.interp(np.log(grid),log_ref,field['field'][::-1,7,node]) for node in (49,50)])
    finally:field._mmap.close()
    geo=geometry(geo_path);ray=geo['rays'][53]
    if geo['ND']!=90 or not np.array_equal(geo['R_EXT'][-90:],raw[0]):raise ValueError('원90기하 주소 오류')
    impact=float(ray['R'][-1]*1.e10)
    out.mkdir();library,command,producers=build(out);sampler=Sampler(library,grid,chi,es,eta,j)
    kernel=FrozenKernel(library,linear_chi=True);g=A.geometry(raw[0]*1.e10,raw[1]*1.e5,raw[2])
    record=dict(status='STARTED',cases=[],scope=__doc__,input_sha256={str(p):identity for p,identity in expected.items()},
        incoming_scope='동일한 제어 입사1e-8. 이동ν의 실제 CMFGEN 입사 스펙트럼 포획을 주장하지 않음',
        new_global_transport=0,new_material_updates=0,final_goal='NOT_MET')
    stack=ExitStack()
    try:
        factory=None
        if prior is not None:
            from lumina_native_columns import NativeSourceColumns
            from lumina_phase_geometry import PhaseGeometry
            with binary.open('rb') as stream:
                _,ns,nf=struct.unpack('<8sII',stream.read(16));stream.seek(16+8*nf)
                temperature=np.frombuffer(stream.read(8*ns),dtype='<f8').copy()
                ne=np.frombuffer(stream.read(8*ns),dtype='<f8').copy()
            source=stack.enter_context(NativeSourceColumns(binary,grid,raw[0]*1.e10,temperature,ne))
            factory=stack.enter_context(PhaseGeometry(geometry_library,g,source))
            record.update(mesh_scope='실제 C phase 기하: 이동ν 한 bin당1/2/4/8분할',
                independent_ODE_calls=0,reused_reference_sha256=sha(reuse_reference),
                geometry_library_sha256=sha(geometry_library))
        for wavelength in (611.1250641898218,673.5722485905775,714.9525535140685):
            cell=Cell(sampler,g,float(raw[0,50]*1.e10),float(raw[0,49]*1.e10),impact,
                      2.99792458e18/wavelength,deadline)
            incoming=1.e-8
            if prior is None:
                loose,work_loose=reference(cell,incoming,1.e-8);tight,work_tight=reference(cell,incoming,1.e-10)
                reference_change=abs(loose-tight)/tight
            else:
                matched=[v for v in prior['cases'] if v['exit_wavelength_A']==wavelength]
                if len(matched)!=1:raise ValueError('완료 참조 파장 주소 오류')
                tight=matched[0]['reference'];reference_change=matched[0]['loose_tight_relative_change']
                work_loose=work_tight={'reused_not_recomputed':True}
            initial=np.zeros(1,dtype=SCALED);initial['mantissa'],initial['exponent']=np.frexp(incoming)
            mesh=[]
            for n in ((1,2,4,8) if factory else (1,4,16,64,256)):
                if factory:
                    phase=factory.grid([impact,cell.z0,cell.z1,cell.inner,cell.outer],n)
                    values=[cell.point(float(x)) for x in phase['x']]
                    length,delta=phase['length'],phase['drift']
                    c=np.array([[v[0]] for v in values]);e=np.array([[v[1]] for v in values])
                else:length,delta,c,e=cell.mesh(n)
                code,value,_,stats=kernel.chain(length,delta,c,e,initial,max_seconds=min(120.,deadline-time.monotonic()))
                if code:raise ValueError('공용 생산 chain 미완료: '+str(stats))
                value=float(np.ldexp(value['mantissa'][0],int(value['exponent'][0])))
                mesh.append(dict(pieces=len(length),intensity=value,relative_error=abs(value-tight)/tight,
                                 primitive_maps=stats['primitive_maps']))
                if factory:mesh[-1].update(subdivisions_per_frequency_bin=n,
                    max_phase_roundoff=phase['max_phase_roundoff'],
                    maximum_radius_difference_cm=float(np.max(abs(phase['radius']-np.array([v[4] for v in values])))))
            first,last=cell.point(0.),cell.point(1.)
            record['cases'].append(dict(exit_wavelength_A=wavelength,reference=tight,
                loose_tight_relative_change=reference_change,loose_work=work_loose,tight_work=work_tight,
                entrance_frequency_Hz=first[5],exit_frequency_Hz=last[5],
                doppler_log_drift=float(np.log(first[2]/last[2])),mesh=mesh))
        record['status']='COMPLETE_FROZEN_SOURCE_CHARACTERISTIC_CELL'
    except (TimeoutError,ValueError) as error:
        record['status']='INCOMPLETE_CHARACTERISTIC_CELL';record['error']=str(error)
    finally:stack.close()
    headers=[ROOT/'src'/name for name in ('cmf_linear_provider.hpp','cmf_nodal_input.h',
        'cmf_fullsr_linear_map.hpp','cmf_ordered_maps.hpp','cmf_positive_moments.h',
        'cmf_frequency_history.h','cmf_linear_affine.hpp','cmf_fullsr_entry.hpp',
        'cmf_fullsr_frozen.h','cmf_fullsr_sc.h')]
    producers.extend([*headers,Path(__file__),Path(A.__file__),Path(fields.__code__.co_filename),
                      Path(read_auxiliary.__code__.co_filename),Path(FrozenKernel.chain.__code__.co_filename)])
    if prior is not None:
        producers.extend([Path(NativeSourceColumns.__init__.__code__.co_filename),
            Path(PhaseGeometry.__init__.__code__.co_filename),
            Path(factory.provider.__class__.__init__.__code__.co_filename)])
    snapshot=out/'source_snapshot';snapshot.mkdir()
    for p in producers:shutil.copy2(p,snapshot/p.name)
    record.update(material_record=70,source_reference_record=71,native_nodes_zero_based=[50,49],ray_1based=54,
        impact_cm=impact,compiler_command=command,elapsed_seconds=time.monotonic()-started,
        source_sha256={str(p):sha(p) for p in producers},library_sha256=sha(library))
    record['input_sha256'].update({str(p):sha(p) for p in (execution_path,manifest)})
    if reuse_reference is not None:
        record['input_sha256'].update({str(p):sha(p) for p in (reuse_reference,geometry_library)})
        record['status']=('COMPLETE_FROZEN_SOURCE_PHASE_CELL' if record['status'].startswith('COMPLETE_') else record['status'])
    encoded=json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)
    with (out/'characteristic-cell.json').open('x') as stream:stream.write(encoded)
    print(encoded);return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('legacy-run','capture','manifest','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--reuse-reference',type=Path);p.add_argument('--geometry-library',type=Path)
    a=p.parse_args();r=compare(a.legacy_run.resolve(),a.capture.resolve(),a.manifest.resolve(),a.out.resolve(),
        reuse_reference=a.reuse_reference.resolve() if a.reuse_reference else None,
        geometry_library=a.geometry_library.resolve() if a.geometry_library else None)
    raise SystemExit(0 if r['status'].startswith('COMPLETE_') else 2)
