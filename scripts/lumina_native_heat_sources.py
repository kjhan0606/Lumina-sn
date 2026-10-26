"""원 native의 외부열원 벡터와 PREVENT_LOW_T 규약. 장/물질 갱신은 없다.

Fortran E12.4의 세 자리 지수에서 E가 생략되는 형식을 보존해 읽는다.
출력에 없는 열원을 임의의0으로 대체하지 않으며, 알려진 writer의 SUM=0
생략 계약을 caller가 명시한 경우에만 전체0 벡터를 허용한다.
"""
import re

import numpy as np


def fortran_number(token):
    text=token.strip().replace('D','E').replace('d','e')
    if 'e' not in text.lower():
        text=re.sub(r'(?<=[0-9.])([+-]\d+)$',r'E\1',text)
    result=float(text)
    if not np.isfinite(result):raise ValueError('native 열원 비유한 수치')
    return result


def vector(text,label,count):
    lines=text.splitlines();indices=[i for i,line in enumerate(lines) if line.strip()==label]
    if len(indices)!=1:raise ValueError('native 벡터 제목 누락/중복: '+label)
    values=[]
    for line in lines[indices[0]+1:]:
        if not line.strip():
            if values:break
            continue
        try:row=[fortran_number(token) for token in line.split()]
        except ValueError:break
        values.extend(row)
    if len(values)!=count:raise ValueError('부분 native 벡터: '+label)
    return np.asarray(values)


def cooling_vector(text,label,count,*,omitted_sum_zero=False):
    """GENCOOL의 실제 Depth 주소를 사용한다. 깊이 블록 순서를 가정하지 않는다."""
    lines=text.splitlines();starts=[i for i,line in enumerate(lines) if line.strip()=='Depth']
    if not starts:raise ValueError('GENCOOL Depth 블록 누락')
    output=np.full(count,np.nan);covered=np.zeros(count,dtype=bool);present=[]
    for first,last in zip(starts,starts[1:]+[len(lines)]):
        chunk=lines[first+1:last];nonempty=[i for i,line in enumerate(chunk) if line.strip()]
        if not nonempty:raise ValueError('부분 GENCOOL Depth')
        try:indices=np.asarray([int(x)-1 for x in chunk[nonempty[0]].split()])
        except ValueError as error:raise ValueError('GENCOOL Depth 정수 주소 오류') from error
        if (not len(indices) or np.any(indices<0) or np.any(indices>=count) or
            len(np.unique(indices))!=len(indices) or covered[indices].any()):
            raise ValueError('GENCOOL Depth 누락/중복 주소')
        covered[indices]=True
        headings=[i for i,line in enumerate(chunk) if line.strip()==label]
        present.append(bool(headings))
        if len(headings)>1:raise ValueError('GENCOOL 열원 중복')
        if headings:output[indices]=vector('\n'.join(chunk),label,len(indices))
    if not covered.all():raise ValueError('부분 GENCOOL 전90점')
    if not any(present):
        if not omitted_sum_zero:raise ValueError('확인되지 않은 열원0 대체 거부')
        return np.zeros(count)
    if not all(present) or not np.isfinite(output).all():raise ValueError('부분 GENCOOL 열원 벡터')
    return output


def artificial_heat(temperature,electron_density,positive_ion_density,minimum_temperature):
    """PREVENT_LOW_T의 EHB cgs 출력. 임의 floor/clamp나 exp cutoff가 없다."""
    t,ne,ni=[np.asarray(a,dtype='f8') for a in (temperature,electron_density,positive_ion_density)]
    if (t.ndim!=1 or ne.shape!=t.shape or ni.shape!=t.shape or not len(t) or
        not np.isfinite([t,ne,ni]).all() or np.any(t<=0) or np.any(ne<=0) or np.any(ni<=0) or
        not np.isfinite(minimum_temperature) or minimum_temperature<0):
        raise ValueError('native 인공 가열 물질 오류')
    if minimum_temperature==0:return np.zeros_like(t)
    with np.errstate(over='raise',invalid='raise',under='ignore'):
        value=1.25663706e-19*ne*ni*np.exp(1000.*(minimum_temperature*1.e-4-t*1.e-4))
    if not np.isfinite(value).all():raise ValueError('native 인공 가열 비유한')
    return value
