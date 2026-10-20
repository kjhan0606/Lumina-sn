#!/usr/bin/env python3
"""원래 전체 복사장 범위와 실제 고정 Doppler 폭으로 후보 주파수 입력을 만든다.

선폭당 표본 수는 명시적 개발 격자 설정이지 해상도 수렴/동질성 판정이 아니다.
물리적 주파수 범위를 메모리에 맞춰 줄이거나 원래 장을 자체장으로 사용하지 않는다.
"""
import argparse
import json
from pathlib import Path
import struct

import numpy as np

from cmfgen_model_sources import fixed_profile
from analyze_cmfgen_material_inventory import sha


def reference_frequencies(path):
    # 모든 J/H/K를 읽지 않고 고정 형식의 실제 주파수 열만 읽는다.
    with path.open('rb') as stream:
        tag=stream.read(16);header=stream.read(24)
    if tag.rstrip()!=b'FORMALPRIMARYV1' or len(header)!=24:
        raise ValueError('완료 참조장 형식 불일치')
    kind,bits,integer_bits,ns,nf,nfield=struct.unpack('<6i',header)
    if (kind,bits,integer_bits,nfield)!=(8,64,32,8) or not 3<=ns<=500 or not 2<=nf<=1000000:
        raise ValueError('참조장 차원/정밀도 손상')
    offset=40+32*ns;stride=24+64*ns
    if path.stat().st_size!=offset+nf*stride:raise ValueError('미완료/후행 참조장')
    dtype=np.dtype(dict(names=['nu'],formats=['<f8'],offsets=[8],itemsize=stride))
    records=np.memmap(path,mode='r',dtype=dtype,offset=offset,shape=(nf,))
    frequency=np.array(records['nu'])*1e15
    if not np.isfinite(frequency).all() or np.any(frequency<=0) or not np.all(np.diff(frequency)<0):
        raise ValueError('참조 주파수 순서/값 손상')
    return frequency


def prepare(reference,expected_sha,model,points_per_width,out):
    reference=Path(reference).resolve();model=Path(model).resolve();out=Path(out)
    if out.exists():raise ValueError('기존 격자 산출물 보존')
    if not np.isfinite(points_per_width) or points_per_width<1:raise ValueError('선폭당 표본 수 손상')
    identity=sha(reference);model_identity=sha(model)
    if identity!=expected_sha:raise ValueError('봉인 참조장 SHA256 불일치')
    native=reference_frequencies(reference);profile=fixed_profile(model)
    target_step=profile['width_fraction']/points_per_width
    lo,hi=float(native[-1]),float(native[0]);span=np.log(hi/lo)
    count=int(np.ceil(span/target_step))+1
    if not 2<=count<=2**31-1:raise ValueError('후보 격자 차원 표현 범위 초과')
    step=span/(count-1);frequency=lo*np.exp(np.arange(count)*step)
    # 양 끝의 작은 지수 반올림도 원래 범위 안팎으로 바꾸지 않는다.
    frequency[0]=lo;frequency[-1]=hi
    if sha(reference)!=identity or sha(model)!=model_identity:raise ValueError('격자 생성 중 입력 변경')
    out.mkdir();np.save(out/'frequency.npy',frequency,allow_pickle=False)
    report=dict(schema='lumina-full-frequency-grid-v1',reference_field=str(reference),
        reference_field_sha256=identity,model_sha256=model_identity,native_frequency_unit_to_Hz=1e15,
        native_points=len(native),frequencies=count,frequency_range_Hz=[lo,hi],dlognu=step,
        points_per_Doppler_width=points_per_width,profile=profile,
        input_sha256={str(reference):identity,str(model):model_identity},
        output_sha256={'frequency.npy':sha(out/'frequency.npy')},
        frequency_resolution_certified=False,physical_RT_executions=0,final_goal='NOT_JUDGED')
    with (out/'grid.json').open('x') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-field',type=Path,required=True)
    p.add_argument('--reference-sha256',required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--points-per-width',type=float,default=4)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();r=prepare(a.reference_field,a.reference_sha256,a.model,a.points_per_width,a.out)
    print(json.dumps({k:r[k] for k in ('frequencies','frequency_range_Hz','dlognu','frequency_resolution_certified')}))
