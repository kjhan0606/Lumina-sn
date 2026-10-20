#!/usr/bin/env python3
"""최초 MOM 사건에 붙인 원래 F-loop 제어·소스 입력을 엄격히 판독한다."""
from pathlib import Path

from analyze_cmfgen_mom_capture import Reader


def floop_state(path:Path):
    r=Reader(path.read_bytes())
    if r.take(16).rstrip()!=b'LUMINAFLOOP1':raise ValueError('F-loop header 오류')
    phase=r.integers(5);n,count=r.integers(2);compute,coherent=r.integers(2)
    if not 3<=n<=10000 or not 0<=count<=21 or compute not in (0,1) or coherent not in (0,1):
        raise ValueError('F-loop 차원/분기 오류')
    tolerance=float(r.reals()[0])
    arrays={key:r.reals(n) for key in ('FOLD','ETA_CLUMP','CHI_SCAT_CLUMP','RJ_ES')}
    if r.take(16).rstrip()!=b'LUMINAFLOOPEND1' or r.offset!=len(r.raw):raise ValueError('F-loop 끝/후행 오류')
    if count!=phase[2]:raise ValueError('F-loop counter/phase 불일치')
    return dict(phase=phase,ND=n,count=count,compute_eddfac=bool(compute),coherent_es=bool(coherent),tolerance=tolerance,**arrays)
