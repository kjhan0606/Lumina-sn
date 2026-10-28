#!/usr/bin/env python3
"""동일 native 광선/대역에서 s-선형 물질의 이차 방출항 수송과 PHOT 흐름을 대조한다.

5196개 원 주파수의 마지막 실제 총eta·core/외부 입사는 고정한다. 원 대역 앞의
실제 blue에서 시작해 새 광선을 다음 주파수로 전달한다. 새 산란/물질 고정점이나
Lumina 생산 특성선과 같은 이산식이라는 주장은 아니다.
"""
import argparse
import ctypes as ct
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from analyze_cmfgen_formal_capture import geometry
from analyze_cmfgen_siii_compact_capture import blue_state, records
from prepare_cmfgen_siii_compact_capture import FIRST, LAST
from prepare_cmfgen_mom_capture import sha

ROOT=Path(__file__).resolve().parents[1]
WEIGHTS_SHA='cc31382dba09deaeca2dd5e5938975cc1a697f119ae9dc675835377e6ca6441f'


class Stats(ct.Structure):
    _fields_=[('completed',ct.c_int),('failed_ray',ct.c_int),('kernel_status',ct.c_int),
              ('directional_cells',ct.c_uint64),('elapsed_seconds',ct.c_double)]


class BandKernel:
    def __init__(self,library):
        self.library=ct.CDLL(str(library));self.function=self.library.lumina_quadratic_band
        self.function.argtypes=[ct.c_int]*4+[ct.c_void_p]*15+[ct.c_double,ct.c_void_p,ct.POINTER(Stats)]
        self.function.restype=ct.c_int

    def advance(self, geometry_arrays, chi, eta, h, inner, outer, *, seconds=120.):
        a=geometry_arrays
        chi,eta,h,inner,outer=[np.ascontiguousarray(x,dtype='f8') for x in (chi,eta,h,inner,outer)]
        if chi.ndim!=2:raise ValueError('대역 계수 차원 오류')
        nf,ne=chi.shape;nr=len(a['obs']);total=int(a['ptr'][-1])
        if (eta.shape!=chi.shape or h.shape!=(nf,) or inner.shape!=(nf,nr) or outer.shape!=(nf,nr) or
            a['ptr'].dtype!=np.dtype('i4') or a['obs'].dtype!=np.dtype('i4') or
            a['ptr'].shape!=(nr+1,) or a['wp'].shape!=(nr,) or a['wm'].shape!=(nr,)):
            raise ValueError('대역 배열 shape/dtype 불일치')
        names=('lm','lp','bm','bp','wp','wm','initial_m','initial_p')
        for name in names:
            expected=(nr,) if name in ('wp','wm') else (total,)
            if a[name].shape!=expected or a[name].dtype!=np.dtype('f8') or not a[name].flags.c_contiguous:
                raise ValueError('대역 기하 배열 불일치: '+name)
        if not a['ptr'].flags.c_contiguous or not a['obs'].flags.c_contiguous:
            raise ValueError('대역 인덱스 연속성 오류')
        result=np.full(nf,np.nan);stats=Stats()
        arrays=[a['ptr'],a['obs'],a['lm'],a['lp'],a['bm'],a['bp'],a['wp'],a['wm'],
                chi,eta,h,inner,outer,a['initial_m'],a['initial_p']]
        code=self.function(nf,nr,int(a['nc']),ne,*[x.ctypes.data for x in arrays],seconds,
                           result.ctypes.data,ct.byref(stats))
        return code,result,{name:getattr(stats,name) for name,_ in Stats._fields_}


def build(out):
    sources=[ROOT/'scripts'/name for name in ('cmfgen_quadratic_band.cpp','cmfgen_fixed_blue_transport.cpp')]
    headers=[ROOT/'src'/name for name in ('cmf_fullsr_linear_map.hpp','cmf_ordered_maps.hpp',
                                        'cmf_positive_moments.h','cmf_frequency_history.h')]
    library=out/'quadratic-band.so'
    command=[os.environ.get('CXX','c++'),'-O3','-std=c++17','-ffp-contract=off','-Wall','-Wextra',
             '-Werror','-fPIC','-shared','-I'+str(ROOT/'src'),*map(str,sources),'-o',str(library)]
    result=subprocess.run(command,capture_output=True,text=True,timeout=45)
    if result.returncode:raise RuntimeError(result.stderr)
    return BandKernel(library),command,[*sources,*headers,library]


def refine(values, subdivisions):
    """원 index 구간의 선형 물질/기하를 유지하고 내부 blue 자유도만 추가한다."""
    a=np.asarray(values,dtype='f8')
    if a.ndim<1 or a.shape[-1]<2 or type(subdivisions) is not int or subdivisions not in (1,2,4):raise ValueError('공간 세분 입력 오류')
    if subdivisions==1:return np.ascontiguousarray(a)
    x=np.arange((a.shape[-1]-1)*subdivisions+1)/subdivisions
    left=np.minimum(x.astype('i8'),a.shape[-1]-2);fraction=x-left
    return np.ascontiguousarray((1-fraction)*a[...,left]+fraction*a[...,left+1])


def plan(g, initial, node, subdivisions=1):
    nr=g['NP']-node;ptr=[0];obs=[];arrays={k:[] for k in ('lm','lp','bm','bp','initial_m','initial_p')}
    for ip in range(nr):
        ray=g['rays'][ip];original_n=len(ray['R']);n=(original_n-1)*subdivisions+1
        if n>512:raise ValueError('소형 진단의 공간점 한도')
        if not np.array_equal(ray['R'],g['R_EXT'][:original_n]):raise ValueError('원 확장층과 다른 광선 좌표')
        k=int(ray['LNK'][node])-1
        if not 0<=k<original_n:raise ValueError('native 관측 링크 오류')
        obs.append(k*subdivisions);ptr.append(ptr[-1]+n)
        for name,value in dict(lm=np.r_[np.diff(refine(ray['S_M'],subdivisions)),0.],
            lp=np.r_[np.diff(refine(ray['S_P'][::-1],subdivisions)),0.],
            bm=refine(ray['B_M'],subdivisions),bp=refine(ray['B_P'][::-1],subdivisions),
            initial_m=refine(initial[ip]['I_M'],subdivisions),
            initial_p=refine(initial[ip]['I_P'][::-1],subdivisions)).items():arrays[name].append(value)
    arrays={k:np.ascontiguousarray(np.concatenate(v),dtype='f8') for k,v in arrays.items()}
    arrays.update(ptr=np.array(ptr,dtype='i4'),obs=np.array(obs,dtype='i4'),nc=min(g['NC'],nr),
        wp=np.ascontiguousarray(g['weights']['JQW_P'][node,:nr]),
        wm=np.ascontiguousarray(g['weights']['JQW_M'][node,:nr]))
    return arrays


def collect(field,g,initial_jit,deadline,count=None):
    requested=LAST-FIRST+1 if count is None else count
    if type(requested) is not int or not 1<=requested<=LAST-FIRST+1:raise ValueError('요청 주파수 수 오류')
    selected=[];last=None;calls=0
    def keep(row):
        return {k:row[k] for k in ('index','freq','dlognu','chi_ext','eta_ext','boundary','native')}
    for row in records(field/'siii_compact.bin',g,include_rays=True):
        if calls%128==0 and time.monotonic()>=deadline:raise TimeoutError('대역 입력 판독 시간 한도')
        if calls==0 and (row['index'],row['jit'])!=(FIRST,initial_jit):raise ValueError('첫 blue 호출 신원 오류')
        if last is not None and row['index']!=last['index']:
            if row['index']!=last['index']+1 or not row['new_freq']:raise ValueError('대역 연속 주파수 누락')
            selected.append(keep(last))
            if len(selected)==requested:break
        elif last is not None and (row['jit']!=last['jit']+1 or row['new_freq']):
            raise ValueError('같은 주파수의 실제 source 반복 순서 오류')
        last=row;calls+=1
    if last is not None and len(selected)<requested:selected.append(keep(last))
    if len(selected)!=requested or selected[-1]['index']!=FIRST+requested-1:raise ValueError('요청 대역 포획 부족')
    # 마지막 총eta는 고정 source다. 같은 주파수 이전 JIT는 새 산란 반복에 사용하지 않는다.
    return selected,calls


def compare(capture,weights,manifest,out,seconds=150,*,subdivisions=1,max_frequencies=None):
    if out.exists():raise FileExistsError('기존 대역 비교 보존')
    if not 0<seconds<=160 or subdivisions not in (1,2,4):raise ValueError('대역 비교 시간/공간 한도 오류')
    if any(os.environ.get(k)!='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):
        raise ValueError('소형 CPU1 환경 필요')
    started=time.monotonic();deadline=started+seconds;out.mkdir()
    field=capture/'field';geo=field/f'FORMAL_{FIRST}_00_geometry.bin'
    paths=[geo,field/'siii_blue_in.bin',field/'siii_compact.bin',weights]
    sealed=json.loads(manifest.read_text())['file_sha256']
    for path in paths:
        expected=WEIGHTS_SHA if path==weights else sealed.get(str(path))
        if expected is None or sha(path)!=expected:raise ValueError('봉인 입력 변경/누락: '+path.name)
    g=geometry(geo)
    if g['RAY_POINTS_INSERTED'] or g['ND']!=90:raise ValueError('현재 범위는 추가점 없는 원90기하')
    jit,initial=blue_state(paths[1],g,include_rays=True)
    rows,calls=collect(field,g,jit,deadline,max_frequencies);nr=g['NP']-49;arrays=plan(g,initial,49,subdivisions)
    chi=refine(np.array([r['chi_ext'] for r in rows]),subdivisions)
    eta=refine(np.array([r['eta_ext'] for r in rows]),subdivisions)
    frequency=np.array([r['freq']*1e15 for r in rows]);h=np.array([r['dlognu'] for r in rows])
    inner=np.array([[x[2] for x in r['boundary'][:nr]] for r in rows])
    outer=np.array([[x[1] for x in r['boundary'][:nr]] for r in rows])
    original=np.array([r['native'][1] for r in rows])
    nf=len(rows);whole_band=nf==LAST-FIRST+1
    with np.load(weights,allow_pickle=False) as data:
        if not np.array_equal(frequency[::-1],data['freq_Hz'][-nf:]):raise ValueError('PHOT 가중치 주파수 불일치')
        if not np.allclose(original[::-1],data['original_J'][-nf:],rtol=1e-10,atol=0):raise ValueError('원 native49 J 불일치')
        weight=data['photo_weight'][-nf:].copy();reference_f=data['F8'][-nf:].copy()
    kernel,command,producers=build(out);remaining=deadline-time.monotonic()
    if remaining<=1:raise TimeoutError('대역 수송 시간 잔여 부족')
    code,result,stats=kernel.advance(arrays,chi,eta,h,inner,outer,seconds=remaining)
    completed=stats['completed'];complete=code==0 and completed==len(rows)
    np.savez(out/'band.npz',frequency_Hz=frequency[:completed],J=result[:completed],
             original_J=original[:completed])
    rate=None
    if complete:
        rate=dict(original_J_photo_cm3_s=float(np.dot(weight,original[::-1])),
            quadratic_J_photo_cm3_s=float(np.dot(weight,result[::-1])),
            existing_F_photo_cm3_s=float(np.dot(weight,reference_f)),
            absolute_F_minus_quadratic_photo_cm3_s=float(np.dot(weight,abs(reference_f-result[::-1]))))
    files=[*paths,manifest,Path(__file__),Path(records.__code__.co_filename),
           Path(geometry.__code__.co_filename),*producers]
    source=out/'source_snapshot';source.mkdir()
    for path in files[5:]:
        if path.suffix in ('.py','.cpp','.h','.hpp'):shutil.copy2(path,source/path.name)
    report=dict(status=(('COMPLETE_FROZEN_ETA_BAND' if whole_band else 'COMPLETE_FROZEN_ETA_PREFIX')
        if complete else 'INCOMPLETE_FROZEN_ETA_BAND'),
        scope=__doc__,input_calls=calls,requested_frequencies=len(rows),stats=stats,returncode=code,
        spatial_subdivisions=subdivisions,whole_band=whole_band,
        requested_spectral_coverage_A=[float(2.99792458e18/frequency[0]),float(2.99792458e18/frequency[-1])],
        completed_spectral_coverage_A=([float(2.99792458e18/frequency[0]),
            float(2.99792458e18/frequency[completed-1])] if completed else None),
        phase=dict(material=70,reference=71,node_zero_based=49,weight_population_cell=39),
        new_self_consistent_scattering=False,new_material_updates=0,new_global_frequency_runs=0,
        accepted_physical_field=False,production_changed=False,
        rate=rate if whole_band else None,prefix_rate=rate if not whole_band else None,
        compiler_command=command,input_sha256={str(p):sha(p) for p in files},
        output_sha256=sha(out/'band.npz'),elapsed_seconds=time.monotonic()-started,final_goal='NOT_MET')
    encoded=json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)
    with (out/'quadratic-band.json').open('x') as stream:stream.write(encoded)
    print(json.dumps({k:report[k] for k in ('status','stats','rate','prefix_rate','elapsed_seconds')},ensure_ascii=False))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('capture','weights','manifest','out'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--spatial-subdivisions',type=int,choices=(1,2,4),default=1)
    p.add_argument('--max-frequencies',type=int)
    a=p.parse_args();r=compare(a.capture.resolve(),a.weights.resolve(),a.manifest.resolve(),a.out.resolve(),
        subdivisions=a.spatial_subdivisions,max_frequencies=a.max_frequencies)
    raise SystemExit(0 if r['status'].startswith('COMPLETE_') else 2)
