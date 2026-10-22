#!/usr/bin/env python3
"""포획된 원 CMFGEN 광선을 청색 주파수부터 독립 연쇄 재현한다.

원 강도·J의 재현 검사이며 대체 수송의 원인 기여율이나 최종 물리 일치는 아니다.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze_cmfgen_formal_capture import geometry
from analyze_cmfgen_siii_compact_capture import blue_state, records
from check_cmfgen_local_sc_packet import replay_ray
from prepare_cmfgen_mom_capture import sha
from prepare_cmfgen_siii_compact_capture import FIRST, LAST


def scaled_error(actual, expected):
    actual=np.asarray(actual);expected=np.asarray(expected)
    return float(np.max(np.abs(actual-expected)/np.maximum(
        np.maximum(np.abs(actual),np.abs(expected)),1e-30)))


def replay_record(row,g,blue):
    """한 formal 호출을 같은 주파수의 고정 청색 상태에서 재현한다."""
    if len(blue)<row['np_limit']:
        raise ValueError('청색 광선 수가 현재 formal 호출보다 작음')
    limit=row['np_limit']
    output=[];maximum=0.
    for ip in range(1,limit+1):
        geom=g['rays'][ip-1]
        old=blue[ip-1]
        outer_p,outer_m,inner_p,inner_m=row['boundary'][ip-1]
        if outer_m != 0.:
            raise ValueError('독립 SC의 외측 음방향 경계가 0이 아님')
        inner_override=inner_p if ip<=g['NC'] else None
        plus,minus=replay_ray(dict(dLOG_NU=row['dlognu'],NC=g['NC']),
                              geom,old,row['chi_ext'],row['eta_ext'],ip,
                              inner_plus=inner_override)
        maximum=max(maximum,scaled_error(
            [plus[0],minus[0],plus[-1],minus[-1]],
            [outer_p,outer_m,inner_p,inner_m]))
        output.append(dict(I_P=plus,I_M=minus))
    for offset,nodes in enumerate(row['points']):
        id_=49+offset
        for ip,values in enumerate(nodes,1):
            link=int(g['rays'][ip-1]['LNK'][id_-1])-1
            if not 0<=link<len(output[ip-1]['I_P']):
                raise ValueError('native 광선 링크가 범위를 벗어남')
            got=[output[ip-1]['I_P'][link],output[ip-1]['I_M'][link],
                 blue[ip-1]['I_P'][link],blue[ip-1]['I_M'][link],
                 g['rays'][ip-1]['B_P'][link],g['rays'][ip-1]['B_M'][link]]
            maximum=max(maximum,scaled_error(got,values))
    if maximum>1e-10:
        raise ValueError(f'원 formal 광선·청색·계수 독립 재현 실패: {maximum:.3g}')
    return output,maximum


def replay(capture,out,max_calls=None):
    if out.exists():
        raise FileExistsError('기존 연쇄 재현 보고서 보존')
    if max_calls is not None and max_calls<1:
        raise ValueError('max_calls는 양수여야 함')
    field=capture/'field'
    candidates=sorted(field.glob(f'FORMAL_{FIRST}_*_geometry.bin'))
    if len(candidates)!=1:
        raise ValueError('대역 첫 호출의 단일 기하가 필요')
    g=geometry(candidates[0])
    initial_jit,carried=blue_state(field/'siii_blue_in.bin',g,include_rays=True)
    first=True;previous_index=None;last=None;seen=0;maximum=0.
    for row in records(field/'siii_compact.bin',g,include_rays=True):
        if first and (row['index'],row['jit'])!=(FIRST,initial_jit):
            raise ValueError('초기 청색 상태의 호출 신원 불일치')
        first=False
        if previous_index is not None and row['index']!=previous_index:
            if row['index']!=previous_index+1:
                raise ValueError('주파수 연쇄 누락')
            carried=last
        previous_index=row['index']
        last,error=replay_record(row,g,carried)
        maximum=max(maximum,error)
        seen+=1
        if max_calls is not None and seen>=max_calls:
            break
    if seen==0:
        raise ValueError('압축 광선 기록이 비어 있음')
    if max_calls is None and previous_index!=LAST:
        raise ValueError('전대역 주파수 끝까지 기록되지 않음')
    report=dict(scope=__doc__,first_frequency=FIRST,last_replayed_frequency=previous_index,
                formal_calls_replayed=seen,max_scaled_error=maximum,
                reached_band_end=max_calls is None,
                call_inventory_requires_separate_analyzer=True,
                file_sha256={str(path):sha(path) for path in (
                    field/'siii_blue_in.bin',field/'siii_compact.bin',candidates[0],Path(__file__))},
                final_goal='NOT_MET')
    with out.open('x') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(result=str(out),calls=seen,max_scaled_error=maximum),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--max-calls',type=int)
    args=parser.parse_args()
    replay(args.capture.resolve(),args.out.resolve(),args.max_calls)
