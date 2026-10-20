#!/usr/bin/env python3
"""uniform angular 공통장 1회: 전주파수/소비자 신원과 에너지 원자료를 pre-pop에 보존한다."""
import argparse
import csv
import json
import os
from pathlib import Path
import resource
import struct
import subprocess
import time

import numpy as np

from run_cmfgen_lambda_probe import sha, stage, scan_logs, stop_child


def coverage(path):
    if not path.exists():
        return dict(count=0, max_source_residual=None, max_J_residual=None, max_H_change_over_J=None)
    if path.stat().st_size > 64*1024**2:
        raise ValueError('주파수 요약 64MiB 초과')
    maximum = np.zeros(3); count = 0; last_freq = np.inf; max_calls = 0
    with path.open() as f:
        for row in csv.DictReader(f):
            count += 1; freq = float(row['freq_native']); calls = int(row['formal_calls'])
            values = np.array([float(row[k]) for k in ('source_residual','J_residual','H_change_over_J')])
            if int(row['index']) != count or not 1 <= calls <= 20 or not 0 < freq < last_freq:
                raise ValueError('공통장 주파수/호출 coverage 불일치')
            if not np.isfinite(values).all() or np.any(values > 1e-8):
                raise ValueError('미수렴 주파수가 소비됨')
            maximum = np.maximum(maximum, values); last_freq = freq; max_calls = max(calls,max_calls)
    return dict(count=count,max_source_residual=float(maximum[0]),max_J_residual=float(maximum[1]),
                max_H_change_over_J=float(maximum[2]),max_formal_calls=max_calls)


def fields(path):
    with path.open('rb') as f:
        tag=f.read(16); header=f.read(24)
        if tag.rstrip() != b'FORMALPRIMARYV1' or len(header)!=24:
            raise ValueError('공통장 binary schema 불일치')
        kind,bits,integer_bits,n,count,nfield=struct.unpack('<6i',header)
        if (kind,bits,integer_bits,nfield)!=(8,64,32,8) or not 3<=n<=500 or not 1<=count<=1000000:
            raise ValueError('공통장 binary 차원/정밀도 불일치')
        geometry=np.frombuffer(f.read(4*n*8),dtype='<f8').copy()
    if geometry.size!=4*n:
        raise ValueError('공통장 기하가 잘림')
    geometry=geometry.reshape(4,n)
    dtype=np.dtype([('index','<i4'),('calls','<i4'),('freq','<f8'),('weight','<f8'),('field','<f8',(8,n))])
    offset=40+4*n*8
    if path.stat().st_size != offset+count*dtype.itemsize:
        raise ValueError('공통장 파일이 부분 출력이거나 record 수 불일치')
    a=np.memmap(path,mode='r',dtype=dtype,offset=offset,shape=(count,))
    if not np.isfinite(geometry).all() or np.any(np.diff(geometry[0])>=0):
        raise ValueError('공통장 기하 오류')
    previous=np.inf
    for start in range(0,count,1024):
        b=a[start:start+1024]; value=b['field']
        if not np.array_equal(b['index'],np.arange(start+1,start+len(b)+1)):
            raise ValueError('공통장 record 순서 오류')
        if not np.isfinite(value).all() or np.any(value[:,[0,4,5,6,7],:]<0):
            raise ValueError('공통장/소스 값 오류')
        if np.any(b['calls']<1) or np.any(b['calls']>20) or not np.isfinite(b['freq']).all() or not np.isfinite(b['weight']).all():
            raise ValueError('공통장 호출/주파수/가중치 오류')
        if np.any(b['freq']<=0) or np.any(b['weight']<=0) or np.any(np.diff(b['freq'])>=0) or b['freq'][0]>=previous:
            raise ValueError('공통장 주파수/가중치 순서 오류')
        previous=b['freq'][-1]
    return geometry,a


def read_edd(root):
    # 기존 CMFGEN reader와 같은 direct-access 형식. 단위는 명시적으로 Hz다.
    info=(root/'EDDFACTOR_INFO').read_text().splitlines()[2].split()
    nd,recl,word=map(int,info[:3]);little=info[5]=='T'
    if word!=8 or recl%word or recl//word<nd+1:
        raise ValueError('EDD record 형식 오류')
    nwr=recl//word;raw=np.fromfile(root/'EDDFACTOR',dtype='<f8' if little else '>f8')
    if raw.size%nwr or raw.size<15*nwr:
        raise ValueError('EDD 부분 record')
    raw=raw.reshape(-1,nwr);data=raw[14:]
    good=np.isfinite(data[:,:nd]).all(axis=1)&(data[:,nd]>0)
    j=data[good,:nd];nu=data[good,nd]*1e15;order=np.argsort(nu)
    return np.ascontiguousarray(j[order]),nu[order],nd,float(raw[4,0])


def validate_edd(root, a):
    j,nu,nd,_=read_edd(root)
    # 기존 reader의 주파수 정렬과 실제 stream 순서를 명시적으로 맞춘다.
    order=np.argsort(a['freq'])
    if len(nu)!=len(a) or nd!=a['field'].shape[2]:
        raise ValueError('EDD와 공통장 크기 불일치')
    native=a['freq'][order]
    if not np.array_equal(nu,native*1e15):
        raise ValueError('EDD와 공통장 주파수 불일치')
    expected=np.asarray(a['field'][order,0,:])
    if not np.array_equal(j,expected):
        raise ValueError('율 소비 EDD의 J가 angular producer와 다름')


def read_cell_energy(path, expected_n=None, expected_count=None):
    if path.stat().st_size > 1024**2:
        raise ValueError('cell 에너지 요약 1MiB 초과')
    with path.open('rb') as f:
        tag=f.read(16);header=f.read(16)
        if tag.rstrip()!=b'RAYENERGYV1' or len(header)!=16:
            raise ValueError('cell 에너지 형식 불일치')
        n,count,expected,order=struct.unpack('<4i',header)
        values=np.frombuffer(f.read(),dtype='<f8').copy()
    if not 3<=n<=500 or count!=expected or count<1 or order!=8 or len(values)!=12*n-5:
        raise ValueError('부분 cell 에너지 또는 coverage 불일치')
    if ((expected_n is not None and n!=expected_n) or
            (expected_count is not None and count!=expected_count)):
        raise ValueError('전체 field/cell 에너지 coverage 불일치')
    if not np.isfinite(values).all() or np.any(np.diff(values[:n])>=0):
        raise ValueError('cell 에너지 기하/비유한 값')
    first,last,endpoint,identity=values[n:n+4]
    if not first>=last>0 or endpoint<0 or identity<0:
        raise ValueError('cell 에너지 주파수/오차 표기')
    offset=n+4
    budget=values[offset:offset+5*(n-1)].reshape(n-1,5);offset+=5*(n-1)
    result=dict(r=values[:n],count=count,first_frequency=first,last_frequency=last,
        maximum_endpoint_error=endpoint,maximum_identity_relative=identity,budget=budget)
    for name,size in (('ray_flux',n),('native_flux',n),('work_first',n-1),('work_last',n-1),('identity_signed',n-1),('identity_L1',n-1)):
        result[name]=values[offset:offset+size];offset+=size
    return result


def validate_cell_energy(geometry,a,path):
    n=a['field'].shape[2];record=read_cell_energy(path,n,len(a))
    if not np.array_equal(record['r'],geometry[0]):raise ValueError('cell/field r 기하 불일치')
    r,v,_,gamma=geometry;total=np.zeros(n);absolute=np.zeros(n)
    for start in range(0,len(a),1024):
        b=a[start:start+1024]
        value=b['weight'][:,None]*r*r*gamma*(b['field'][:,1,:]+v/299792.458*b['field'][:,0,:])
        total+=value.sum(axis=0);absolute+=abs(value).sum(axis=0)
    # 두 합산 순서 및 gamma 연산의 IEEE 전방오차 상한. 물리 허용치가 아니다.
    bound=(20+2*len(a))*np.finfo(float).eps*absolute
    if np.any(abs(total-record['native_flux'])>bound):raise ValueError('cell/native field 소비자 광도 불일치')
    return dict(count=record['count'],maximum_endpoint_error=float(record['maximum_endpoint_error']),
        maximum_identity_relative=float(record['maximum_identity_relative']),physical_energy_pass='NOT_JUDGED')


def run(seed,root,native,build,baseline):
    started=time.monotonic(); manifest=json.loads((build/'formal-primary-build.json').read_text())
    if not manifest['whole_executable_released']:
        raise ValueError('전체 diagnostic 준비가 끝나지 않은 build')
    if os.environ.get('OMP_NUM_THREADS')!='16' or not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Slurm CPU16/OMP16 계산에서만 실행')
    for path,h in manifest['output_sha256'].items():
        if sha(Path(path))!=h:
            raise ValueError('봉인 공통장 build 변경: '+path)
    old=stage(seed,root,native/'exe/cmfgen_dev.exe',native)
    if old['seed_sha256']!=json.loads(baseline.read_text())['seed_sha256']:
        raise ValueError('봉인 StageA59 입력 불일치')
    env=dict(os.environ,LUMINA_FORMAL_PRIMARY='1',LUMINA_FORMAL_PRIMARY_WHOLE='1',
             LUMINA_FORMAL_CELL_BUDGET='1' if manifest.get('cell_energy_integral') else '',
             LUMINA_WHOLE_MODE='',LUMINA_MOM_CAPTURE_MODE='',LUMINA_FORMAL_CAPTURE_MODE='')
    resource.setrlimit(resource.RLIMIT_FSIZE,(8*1024**3,8*1024**3))
    record=dict(scope='StageA59의 새 angular J60/율·열/에너지 원자료 diagnostic; pre-pop 종료',
                transport_scheme='uniform_formal_angular_source_fixed_point',seed=str(seed),run=str(root),
                source_interpolation=manifest.get('source_interpolation','legacy_conditional_parabolic'),
                cell_energy_integral=manifest.get('cell_energy_integral'),
                frequency_native_to_Hz=1e15,frequency_weight_unit='Hz',
                build=str(build),build_manifest_sha256=sha(build/'formal-primary-build.json'),
                baseline_sha256=sha(baseline),expected_frequencies=197291,expected_depths=90,
                resources='CPU16/OMP16/16GiB/GPU0 parent2700s child2400s retry0 no node pin',
                source_fixed_point_tolerance=1e-8,formal_iteration_cap=20,
                MOM_equation_residual='NOT_APPLICABLE_NOT_ZERO',final_goal='NOT_JUDGED')
    with (root/'seq_logs/whole-inputs.json').open('x') as f:
        json.dump(record,f,ensure_ascii=False,indent=2)
    launch=time.monotonic();reason=None;cursors={}
    with (root/'seq_logs/cmfgen.stdout').open('xb') as out,(root/'seq_logs/cmfgen.stderr').open('xb') as err:
        proc=subprocess.Popen(['srun','--cpu-bind=cores',str(build/'cmfgen_formal_primary.exe')],cwd=root,env=env,
            stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
        while proc.poll() is None:
            remaining=2400-(time.monotonic()-launch)
            if remaining<=0:
                reason='child2400s';stop_child(proc);break
            try:proc.wait(timeout=min(60,remaining))
            except subprocess.TimeoutExpired:
                reason=scan_logs(root,cursors)
                if reason:stop_child(proc);break
    complete=False;validation_error=None;cov=dict(count=0);cell_summary=None
    try:
        cov=coverage(root/'FORMAL_PRIMARY_SWEEP.csv')
        if proc.returncode==97 and (root/'FORMAL_PRIMARY_COMPLETE.txt').is_file() and cov['count']==197291:
            g,a=fields(root/'FORMAL_PRIMARY_FIELDS.bin')
            if a['field'].shape!=(197291,8,90):raise ValueError('실제 전체 격자가 계약과 다름')
            if sha(root/'SCRTEMP')!=sha(seed/'SCRTEMP'):raise ValueError('pre-pop 물질 상태 변경')
            validate_edd(root,a)
            if manifest.get('cell_energy_integral'):
                cell_summary=validate_cell_energy(g,a,root/'FORMAL_CELL_ENERGY.bin')
            complete=True
    except (ValueError,OSError) as error:
        validation_error=str(error)
    evidence=[root/n for n in ('FORMAL_PRIMARY_SWEEP.csv','FORMAL_PRIMARY_FIELDS.bin','FORMAL_PRIMARY_COMPLETE.txt','FORMAL_CELL_ENERGY.bin',
        'FORMAL_PRIMARY_FAILURE.txt','FORMAL_PRIMARY_SOURCE.bin','FORMAL_PRIMARY_CONTEXT.txt','FORMAL_PRIMARY_CONTEXT.bin','FORMAL_PRIMARY_BLUE.bin',
        'FORMAL_PRIMARY_RAYS.bin','FORMAL_PRIMARY_GEOMETRY.bin') if (root/n).is_file()]
    snapshot_bytes=sum(p.stat().st_size for p in evidence if p.name.startswith(('FORMAL_PRIMARY_SOURCE','FORMAL_PRIMARY_CONTEXT','FORMAL_PRIMARY_BLUE','FORMAL_PRIMARY_RAYS','FORMAL_PRIMARY_GEOMETRY')))
    if snapshot_bytes>256*1024**2:raise ValueError('최초 실패 snapshot 256MiB 초과')
    result=dict(transport_scheme=record['transport_scheme'],source_interpolation=record['source_interpolation'],
        returncode=proc.returncode,diagnostic_complete=complete,
        coverage=cov,validation_error=validation_error,stop_reason='pre_population_sweep_complete' if complete else reason or 'incomplete_or_first_failure',
        cell_energy=cell_summary,
        elapsed_child_seconds=time.monotonic()-launch,elapsed_total_seconds=time.monotonic()-started,
        evidence_sha256={str(p):sha(p) for p in evidence},snapshot_bytes=snapshot_bytes,
        energy_closure='NOT_JUDGED_REQUIRES_ACTUAL_90_GRID_LEDGER',closure_pass='NOT_JUDGED',final_goal='NOT_JUDGED')
    with (root/'seq_logs/whole-runtime.json').open('x') as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))
    return 0 if complete else 1


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('seed','run','native','build','baseline'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();raise SystemExit(run(a.seed,a.run,a.native,a.build,a.baseline))
