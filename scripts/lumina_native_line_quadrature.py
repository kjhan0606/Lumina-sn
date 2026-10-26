"""저장된 native 선 식별자·정수 공명구간의 비교용 복원. 새 주파수 생성 없음."""
import ctypes as C
from pathlib import Path
import re

import numpy as np

import cmfgen_model_continuum as B
import cmfgen_model_sources as S


def intervals(run,rows,models,lines,n_frequency):
    run = Path(run)
    profile = S.fixed_profile(run/'MODEL')
    if (S.control(run/'MODEL','GLOBAL_LINE') != 'BLANK' or
            S.control(run/'MODEL','LAM_SET','F') != 'F'):
        raise ValueError('고정 DOP_FIX 전역 blanketed 선 비교만 지원')
    files = [run/name for name in ('MODEL','TRANS_INFO','fort.77')]
    before = {str(path):B.sha(path) for path in files}
    by_label = {value['label']:key for key,value in models.items()}
    full = {(int(r['atomic_number']),int(r['ion_number'])+1,int(r['level_number'])+1):i
            for i,r in enumerate(rows)}
    pairs = {(int(l),int(u)):i for i,(l,u) in enumerate(zip(lines['lower'],lines['upper']))}
    if len(pairs) != len(lines['lower']):
        raise ValueError('중복 C 선 원자 주소')
    native_ids = np.zeros(len(pairs),dtype='i8')
    with (run/'TRANS_INFO').open() as stream:
        for line in stream:
            parts = line.split()
            if len(parts) < 6 or not all(x.isdigit() for x in parts[:3]):
                continue
            label = re.search(r'([A-Za-z][A-Za-z0-9]*)\(',line)
            if label is None or label[1] not in by_label:
                continue
            Z,stage = by_label[label[1]]
            if Z not in (14,16):
                continue
            native_index,lower,upper = map(int,parts[:3])
            pair = full.get((Z,stage,lower)),full.get((Z,stage,upper))
            position = pairs.get(pair)
            if position is None:
                continue
            if native_ids[position] != 0:
                raise ValueError('native TRANS_INFO 원자 주소 중복')
            native_ids[position] = native_index
    if np.any(native_ids == 0):
        raise ValueError('native TRANS_INFO에서 선택 C 선의 정확한 원자 주소 누락')
    # FIX_DOP의 모든 시작 주파수는 동일 양의 배율×선 중심이다.
    # 두 정렬의 순서는 같고 같은 중심의 tie는 같은 공명구간을 가진다.
    table = np.loadtxt(run/'fort.77',usecols=(0,1,2,3),ndmin=2)
    nactive = len(table)
    ids,starts,ends = (table[:,i].astype('i8') for i in range(3))
    if (not np.array_equal(ids,np.arange(1,nactive+1)) or
            not np.array_equal(table[:,:3],np.c_[ids,starts,ends]) or
            np.any(starts < 1) or np.any(ends <= starts) or np.any(ends > n_frequency)):
        raise ValueError('native 정수 공명구간 표의 주소/순서 오류')
    active = native_ids <= nactive
    first = np.zeros(len(pairs),dtype=np.uintp)
    last = np.zeros_like(first)
    position = native_ids[active]-1
    first[active] = n_frequency-ends[position]
    last[active] = n_frequency-starts[position]+1
    native_center = table[position,3]*1e15
    center = np.asarray(lines['nu0'])[active]
    print_bound = .5*10.**(np.floor(np.log10(native_center))-5)
    arithmetic_allowance = 64*np.finfo(float).eps*1e18
    if np.any(abs(native_center-center) > print_bound+arithmetic_allowance):
        raise ValueError('정수 선 번호와 C 선 중심이 원 인쇄 정밀도에서 불일치')
    if any(B.sha(Path(path)) != digest for path,digest in before.items()):
        raise ValueError('native 선 주소 판독 도중 원 자료 변경')
    return first,last,active,dict(source_sha256=before,native_grid_active_lines=nactive,
        selected_active_lines=int(active.sum()),selected_native_inactive_lines=int((~active).sum()),
        inactive_native_ids=native_ids[~active].tolist(),profile=profile,
        index_rule='TRANS_INFO atomic addresses; constant DOP_FIX preserves start-frequency order',
        scope='native discrete operator only; inactivity is not physical absence of the transition')


def contract(producer,frequency,width,J,first,last,active):
    nu,width,j = [np.ascontiguousarray(v,dtype='f8') for v in (frequency,width,J)]
    first,last = [np.ascontiguousarray(v,dtype=np.uintp) for v in (first,last)]
    active = np.ascontiguousarray(active,dtype='u1')
    center = np.ascontiguousarray(producer.lines['nu0'],dtype='f8')
    n = len(center)
    if any(v.shape != (n,) for v in (first,last,active)):
        raise ValueError('native BB 정수 공명구간 차원 오류')
    ptr = C.POINTER(C.c_double)
    iptr = C.POINTER(C.c_size_t)
    bptr = C.POINTER(C.c_uint8)
    fn = producer.dll.lumina_model_point_gaussian_native_jbar
    fn.argtypes = [C.c_size_t,C.c_size_t]+[ptr]*4+[iptr,iptr,bptr,C.c_double,ptr,ptr,bptr]
    fn.restype = C.c_int
    jbar = np.empty(n);mass = np.empty(n);status = np.empty(n,dtype='u1')
    if fn(len(nu),n,*[v.ctypes.data_as(ptr) for v in (nu,width,j,center)],
          first.ctypes.data_as(iptr),last.ctypes.data_as(iptr),active.ctypes.data_as(bptr),
          producer.profile['width_fraction'],jbar.ctypes.data_as(ptr),mass.ctypes.data_as(ptr),
          status.ctypes.data_as(bptr)):
        raise ValueError('native 저장 공명구간 BB 적분 실패')
    bul = producer.lines['A']*B.C**2/(2*B.H*center**3)
    return dict(jbar=jbar,profile_mass=mass,status=status,
                absorption=bul/producer.lines['g_ratio']*jbar,stimulated=bul*jbar,
                spontaneous_native_discrete=producer.lines['A']*mass,
                native_active=active,first=first,last=last)
