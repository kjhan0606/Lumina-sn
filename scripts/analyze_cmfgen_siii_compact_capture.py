#!/usr/bin/env python3
"""압축 formal 광선 포획을 원 J장과 광선 각도합에 대조한다."""
import argparse
import json
import math
from pathlib import Path
import struct

import numpy as np

from analyze_cmfgen_formal_capture import geometry
from prepare_cmfgen_mom_capture import sha
from prepare_cmfgen_siii_compact_capture import FIRST, LAST
from run_cmfgen_formal_primary import fields


HEADER = struct.Struct('<16s9i2d')
BOUNDARY = struct.Struct('<2i4d')
POINT = struct.Struct('<3i6d')
BLUE_HEADER = struct.Struct('<16s6i')


def take(stream, size):
    raw = stream.read(size)
    if len(raw) != size:
        raise ValueError('압축 광선 기록이 잘림')
    return raw


def blue_state(path, g, include_rays=False):
    if path.stat().st_size > 64 * 1024**2:
        raise ValueError('청색 초기 광선 상태가 과대함')
    with path.open('rb') as stream:
        tag, outer, index, jit, np_, nd, np_limit = BLUE_HEADER.unpack(take(stream, BLUE_HEADER.size))
        if (tag.rstrip() != b'LUMINASIIIBLUE1' or
                (outer,index,np_,nd) != (71,FIRST,g['NP'],g['ND']) or
                np_limit not in (np_,np_-1)):
            raise ValueError('청색 초기 광선 헤더 불일치')
        rays=[]
        for ip in range(1,np_limit+1):
            written_ip, nz = struct.unpack('<2i', take(stream, 8))
            if written_ip != ip or nz != len(g['rays'][ip-1]['R']):
                raise ValueError('청색 초기 광선 기하 불일치')
            values = np.frombuffer(take(stream, 16*nz), dtype='<f8')
            if not np.isfinite(values).all():
                raise ValueError('청색 초기 강도 비유한')
            if include_rays:
                rays.append(dict(I_P=values[:nz].copy(),I_M=values[nz:].copy()))
        if take(stream,16).rstrip() != b'LUMINASIIIBLEND' or stream.read(1):
            raise ValueError('청색 초기 상태 끝/후행 오류')
    return (jit,rays) if include_rays else jit


def records(path, g, include_rays=False):
    if not 0 < path.stat().st_size <= 512 * 1024**2:
        raise ValueError('압축 광선 기록 크기 위반')
    with path.open('rb') as stream:
        while stream.read(1):
            stream.seek(-1,1)
            unpacked=HEADER.unpack(take(stream,HEADER.size))
            tag=unpacked[0]
            outer,index,jit,np_,nd,nd_ext,np_limit,new_freq,inserted=unpacked[1:10]
            freq,dlognu=unpacked[10:]
            if (tag.rstrip()!=b'LUMINASIIIRAYV1' or outer!=71 or
                not FIRST<=index<=LAST or np_!=g['NP'] or nd!=g['ND'] or
                not nd<=nd_ext<=nd+1000 or np_limit not in (np_,np_-1) or
                new_freq not in (0,1) or inserted not in (0,1) or
                inserted!=int(g['RAY_POINTS_INSERTED']) or
                not np.isfinite([freq,dlognu]).all() or freq<=0 or dlognu<=0):
                raise ValueError('압축 formal 헤더/기하 오류')
            arrays=np.frombuffer(take(stream,8*(4+2*nd_ext)),dtype='<f8')
            if not np.isfinite(arrays).all():
                raise ValueError('압축 formal 계수 비유한')
            native=arrays[:4]
            boundary=[]
            for ip in range(1,np_limit+1):
                row=BOUNDARY.unpack(take(stream,BOUNDARY.size))
                if row[:2]!=(ip,len(g['rays'][ip-1]['R'])) or not np.isfinite(row[2:]).all():
                    raise ValueError('경계 광선 불일치')
                if include_rays:
                    boundary.append(row[2:])
            observed=np.zeros(4)
            bound=np.zeros(4)
            point_rows=[]
            for id_ in range(49,53):
                terms=[];magnitudes=[]
                nodes=[]
                for ip in range(1,min(np_limit,np_-id_+1)+1):
                    row=POINT.unpack(take(stream,POINT.size))
                    expected_link=int(g['rays'][ip-1]['LNK'][id_-1])
                    if row[:3]!=(ip,id_,expected_link) or not np.isfinite(row[3:]).all():
                        raise ValueError('native 광선 위치/값 불일치')
                    plus=row[3]*g['weights']['JQW_P'][id_-1,ip-1]
                    minus=row[4]*g['weights']['JQW_M'][id_-1,ip-1]
                    terms.append(float(plus+minus))
                    magnitudes.extend((abs(float(plus)),abs(float(minus))))
                    if include_rays:
                        nodes.append(row[3:])
                if include_rays:
                    point_rows.append(nodes)
                k=id_-49
                observed[k]=math.fsum(terms)
                gamma=(3*len(terms)+8)*np.finfo(float).eps
                bound[k]=gamma/(1-gamma)*math.fsum(magnitudes)
            if take(stream,16).rstrip()!=b'LUMINASIIIEND1':
                raise ValueError('압축 formal 끝 표지 불일치')
            error=np.abs(native-observed)
            if np.any(error>bound):
                raise ValueError('원 formal 광선합 재현 실패')
            result=dict(index=index,jit=jit,freq=freq,native=native.copy(),
                        max_roundoff_fraction=float(np.max(np.divide(
                            error,bound,out=np.zeros_like(error),where=bound>0))))
            if include_rays:
                result.update(dlognu=dlognu,np_limit=np_limit,new_freq=bool(new_freq),
                              chi_ext=arrays[4:4+nd_ext].copy(),
                              eta_ext=arrays[4+nd_ext:].copy(),
                              boundary=boundary,points=point_rows)
            yield result


def analyze(capture, original_field, out):
    if out.exists():
        raise FileExistsError('기존 분석 보존')
    field=capture/'field'
    candidates=sorted(field.glob(f'FORMAL_{FIRST}_*_geometry.bin'))
    if len(candidates)!=1:
        raise ValueError('첫 대역 주파수의 단일 기하 누락')
    g=geometry(candidates[0])
    blue_jit=blue_state(field/'siii_blue_in.bin',g)
    _,original=fields(original_field)
    counts=np.zeros(LAST-FIRST+1,dtype='i4')
    maximum_roundoff=0.;maximum_j_rel=0.;previous=FIRST;seen=0
    for row in records(field/'siii_compact.bin',g):
        index=row['index']
        if seen==0 and (index,row['jit'])!=(FIRST,blue_jit):
            raise ValueError('청색 초기 상태와 첫 formal 호출 불일치')
        if index<previous or index>previous+1:
            raise ValueError('압축 주파수 순서/누락')
        previous=index
        offset=index-FIRST
        counts[offset]+=1
        seen+=1
        if counts[offset]>20 or row['freq']!=original['freq'][index-1]:
            raise ValueError('원 장 주파수/호출 신원 불일치')
        maximum_roundoff=max(maximum_roundoff,row['max_roundoff_fraction'])
        if counts[offset]==original['calls'][index-1]:
            truth=np.asarray(original['field'][index-1,0,48:52])
            error=np.abs(row['native']-truth)/np.maximum(np.abs(truth),1e-300)
            maximum_j_rel=max(maximum_j_rel,float(np.max(error)))
            if np.max(error)>1e-10:
                raise ValueError('최종 native J와 원 formal 장 불일치')
    expected=np.asarray(original['calls'][FIRST-1:LAST],dtype='i4')
    if previous!=LAST or not np.array_equal(counts,expected):
        raise ValueError('목표 대역 호출 수 누락/중복')
    report=dict(scope=__doc__,band_indices=[FIRST,LAST],calls=int(counts.sum()),
                frequencies=int(len(counts)),blue_jit=blue_jit,
                maximum_ray_moment_roundoff_fraction=maximum_roundoff,
                maximum_final_J_relative_to_original=maximum_j_rel,
                file_sha256={str(p):sha(p) for p in (
                    field/'siii_blue_in.bin',field/'siii_compact.bin',candidates[0],original_field)},
                final_goal='NOT_MET')
    with out.open('x') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(result=str(out),calls=report['calls'],
                          max_J_rel=maximum_j_rel),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--original-field',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    analyze(args.capture.resolve(),args.original_field.resolve(),args.out.resolve())
