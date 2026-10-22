#!/usr/bin/env python3
"""실제 CMFGEN J를 보간 시험 자료로만 쓴다. 자체장 입력/물리 수송은 아니다."""
import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from run_cmfgen_formal_primary import fields

ROOT=Path(__file__).resolve().parents[1]
CONFIG=Path('/gpfs/kjhan/cmfgen_runs/lumina_field0_checkpoint_20261020-EoA5Bi/input/config.json')
CONFIG_SHA='5fe599e35c767659ac40fd80052250e6950615679eee8bd14cc6b2aef3997a82'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def run(root):
    root=Path(root).resolve();out=root/'native-sample.json';library=root/'probe.so'
    if out.exists() or library.exists():raise FileExistsError('기존 실제 재구성 대조 보존')
    # 원 config의 해시는 기존 봉인값을 그대로 확인한다.
    if sha(CONFIG)!=CONFIG_SHA:raise ValueError('원 실제 config 변경')
    c=json.loads(CONFIG.read_text());plan_path=Path(c['plan'])
    if sha(plan_path)!=c['plan_sha256']:raise ValueError('원 물질/참조 계획 변경')
    plan=json.loads(plan_path.read_text());grid_path=Path(c['grid'])
    if sha(grid_path)!=c['grid_sha256']:raise ValueError('원 실제 주파수 격자 변경')
    started=time.monotonic()
    source_paths=[Path(__file__),ROOT/'tests/monotone_reconstruction_probe.cpp',
        ROOT/'src/cmf_gpu_reconstruction.hpp',ROOT/'src/cmf_monotone_reconstruction.hpp',
        ROOT/'src/cmf_ordered_maps.hpp',ROOT/'scripts/run_cmfgen_formal_primary.py']
    identities={str(p):sha(p) for p in source_paths}
    subprocess.run([os.environ.get('CXX','c++'),'-O2','-std=c++17','-ffp-contract=off',
        '-Wall','-Wextra','-Werror','-fPIC','-shared','-I'+str(ROOT/'src'),str(source_paths[1]),
        '-o',str(library)],check=True,capture_output=True,text=True,timeout=60)
    fun=ct.CDLL(str(library)).reconstruction_native_sample_probe
    a=np.ctypeslib.ndpointer(dtype='f8',flags='C_CONTIGUOUS');i=np.ctypeslib.ndpointer(dtype='i4',flags='C_CONTIGUOUS')
    fun.argtypes=[ct.c_int,ct.c_int,i,a,a,a,i];fun.restype=ct.c_int
    nu=np.load(grid_path,allow_pickle=False);lognu=np.log(nu)
    if len(nu)!=1115020 or np.any(np.diff(lognu)<=0):raise ValueError('전 native 격자 오류')
    geometry,record=fields(Path(plan['same_material_reference_field'])/'FORMAL_PRIMARY_FIELDS.bin')
    rows=[]
    try:
        if geometry.shape!=(4,90):raise ValueError('참조의 원래점90 차원 변경')
        reference_nu=np.asarray(record['freq'][::-1])*1e15;reference_log=np.log(reference_nu)
        interior=np.flatnonzero((nu>reference_nu[0])&(nu<reference_nu[-1]))
        for node in (0,15,30,45,60,75,89):
            if time.monotonic()-started>170:raise TimeoutError('실제 재구성 대조 내부170초')
            values=np.ascontiguousarray(np.interp(lognu,reference_log,record['field'][::-1,0,node]))
            # 전대역 고정 표본과 큰 상대 굴곡 표본을 같이 조사한다. 밝은 열로 정규화하지 않는다.
            denominator=values[:-2]+values[1:-1]+values[2:]
            curvature=np.zeros(len(nu)-2)
            np.divide(abs(values[:-2]-2*values[1:-1]+values[2:]),denominator,
                out=curvature,where=denominator>0)
            sharp=np.argpartition(curvature,-256)[-256:]+1
            columns=np.unique(np.r_[np.linspace(interior[0],interior[-1],2048).astype('i4'),
                sharp,624712,624968,625223])
            columns=np.ascontiguousarray(columns[(columns>=3)&(columns<len(nu)-5)&
                (columns>=interior[0]+4)&(columns+5<=interior[-1])],dtype='i4')
            original=np.zeros(len(columns));candidate=np.zeros(len(columns));flags=np.zeros(len(columns),dtype='i4')
            if fun(len(nu),len(columns),columns,values,original,candidate,flags):raise ValueError('표본 재구성 호출 실패')
            target=.5*(lognu[columns]+lognu[columns+1])
            reference=np.interp(target,reference_log,record['field'][::-1,0,node])
            def summary(actual,valid):
                positive=valid&(reference>0)
                difference=abs(actual-reference);relative=np.zeros(len(reference))
                np.divide(difference,reference,out=relative,where=reference>0)
                worst=int(np.argmax(np.where(positive,relative,-1.))) if np.any(positive) else None
                return dict(maximum_sample_relative=float(relative[worst]) if worst is not None else None,
                    worst_column=int(columns[worst]) if worst is not None else None,
                    reference_at_worst=float(reference[worst]) if worst is not None else None,
                    reconstructed_at_worst=float(actual[worst]) if worst is not None else None,
                    wavelength_A_at_worst=float(2.99792458e18/np.exp(target[worst])) if worst is not None else None,
                    endpoint_values_at_worst=values[columns[worst]:columns[worst]+2].tolist() if worst is not None else None,
                    reference_points_inside_bin=int(np.searchsorted(reference_nu,nu[columns[worst]+1],side='right')-
                        np.searchsorted(reference_nu,nu[columns[worst]],side='left')) if worst is not None else None,
                    samples_over_1e4=int(np.sum(positive&(relative>1e-4))),
                    zero_reference_nonzero=int(np.sum(valid&(reference==0)&(actual!=0))))
            rows.append(dict(native_node=node,radius_cm=float(geometry[0,node]*1e10),samples=len(columns),
                failed_original=int(np.sum((flags&1)!=0)),failed_candidate=int(np.sum((flags&2)!=0)),
                original_limited=int(np.sum((flags&4)!=0)),original=summary(original,(flags&1)==0),
                candidate=summary(candidate,(flags&2)==0)))
    finally:record._mmap.close()
    if any(sha(p)!=h for p,h in identities.items()):raise ValueError('대조 중 소스 변경')
    result=dict(status='NATIVE_RADIATION_RECONSTRUCTION_SAMPLE_NOT_FIELD',rows=rows,
        input_sha256={str(CONFIG):CONFIG_SHA,str(plan_path):c['plan_sha256'],str(grid_path):c['grid_sha256']},
        source_sha256=identities,library_sha256=sha(library),
        reference_field_sha256=plan['reference_field_sha256'],reference_hash='INHERITED_SEALED_PLAN_NOT_REHASHED',
        reference='CMFGEN 원 J의 lognu 선형 보간을 시험 함수로 사용. 독립 연속 정답/PHOT/자체장 수용이 아니다.',
        sampling='각7개 원래점의 균일2048열+상대굴곡256열+기존3열. 반bin 이동만 조사하며 전주파수 최대가 아니다.',
        elapsed_seconds=time.monotonic()-started,GPU_seconds=0,actual_model_F_calls=0,final_goal='NOT_MET')
    with out.open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(result['rows'],ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');args=p.parse_args();run(args.root)
