#!/usr/bin/env python3
"""실제 음의 총소광을 원인 선과 원래 두 깊이의 인구역전까지 추적한다.

완료 계수를 읽는 오프라인 분석이다. χ를 수정하거나 새 복사장을 풀지 않는다.
"""
import argparse
import csv
import json
from pathlib import Path
import struct

import numpy as np

import cmfgen_model_sources as S
B=S.B


def read_full(path):
    with path.open('rb') as stream:
        magic,nl,ns=struct.unpack('<8sII',stream.read(16))
        if magic!=b'LMATFL01':raise ValueError('C FL 형식 불일치')
        array=np.frombuffer(stream.read(8*nl*ns),dtype='<f8').reshape(nl,ns).copy()
        if stream.read(1):raise ValueError('C FL 후행 데이터')
    if not np.isfinite(array).all() or np.any(array<0):raise ValueError('C FL 입력 손상')
    return array


def analyze(sources,atomic,c_full,native_full,cell_map,out):
    if out.exists():raise ValueError('기존 인구역전 추적 보존')
    report=json.loads((sources/'sources.json').read_text())
    for name,identity in report['output_sha256'].items():
        if B.sha(sources/name)!=identity:raise ValueError('완료 소스 표본 신원 변경')
    data=np.load(sources/'sources.npz');lines=np.load(sources/'line_catalog.npz')
    full=read_full(c_full);native=read_full(native_full)
    levels=list(csv.DictReader((atomic/'levels.csv').open()))
    mapping=list(csv.DictReader(cell_map.open()))
    if len(mapping)!=full.shape[1] or full.shape[0]!=len(levels) or native.shape[0]!=len(levels):
        raise ValueError('C 인구/FL/셀 대응 불일치')
    nu=data['nu_Hz'];chi=sum(data[k] for k in ('bf_chi','ff_chi','bb_chi','es_chi'))
    negative=np.argwhere(chi<0);cases=[]
    profile=report['profile'];opacity=np.pi*B.atomic.E_CGS**2/(B.ME*B.C)
    for cell,bin_index in negative:
        freq=float(nu[bin_index]);center=lines['nu0']
        active=np.flatnonzero(abs(freq-center)<=center*profile['support_fraction'])
        lo=lines['lower'][active];up=lines['upper'][active]
        width=center[active]*profile['width_fraction'];x=(freq-center[active])/width
        phi=np.exp(-x*x)/(1.77245385095516*width)
        contributions=opacity*lines['f'][active]*(full[lo,cell]-lines['g_ratio'][active]*full[up,cell])*phi
        difference=abs(float(contributions.sum())-float(data['bb_chi'][cell,bin_index]))
        if difference>1e-12*max(float(abs(contributions).sum()),1e-300):
            raise ValueError('원인 선 합이 실제 생산 BB 계수와 다름')
        native_indices=[int(mapping[cell][k])-1 for k in
            ('inner_native_depth_1based','outer_native_depth_1based')]
        top=[]
        for at in np.argsort(-abs(contributions))[:8]:
            lower,upper=int(lo[at]),int(up[at]);gl=float(levels[lower]['g']);gu=float(levels[upper]['g'])
            top.append(dict(catalog_row=int(active[at]),Z=int(levels[lower]['atomic_number']),
                ion_number=int(levels[lower]['ion_number']),lower_FL_1based=int(levels[lower]['level_number'])+1,
                upper_FL_1based=int(levels[upper]['level_number'])+1,
                lower_configuration=levels[lower]['configuration'],upper_configuration=levels[upper]['configuration'],
                wavelength_vacuum_A=float(B.C/center[active[at]]*1e8),
                profile_x=float(x[at]),chi=float(contributions[at]),
                cell_lower_n_over_g=float(full[lower,cell]/gl),cell_upper_n_over_g=float(full[upper,cell]/gu),
                parent_native_depths_1based=[i+1 for i in native_indices],
                parent_lower_n_over_g=[float(native[lower,i]/gl) for i in native_indices],
                parent_upper_n_over_g=[float(native[upper,i]/gu) for i in native_indices],
                parent_inverted=[bool(native[lower,i]/gl<native[upper,i]/gu) for i in native_indices]))
        cases.append(dict(cell=int(cell),bin=int(bin_index),frequency_Hz=freq,
            wavelength_A=float(B.C/freq*1e8),total_chi=float(chi[cell,bin_index]),
            components={k:float(data[k][cell,bin_index]) for k in
                ('bf_chi','bf_eta','ff_chi','ff_eta','bb_chi','bb_eta','es_chi')},
            contributing_lines=len(active),line_sum_absolute_difference=difference,top_contributors=top))
    result=dict(schema='lumina-signed-source-attribution-v1',
        negative_samples=len(cases),cases=cases,
        parent_comparison='original90 C-imported full populations reconstructed from native SL; not a native saved FL dump',
        modified_opacity=False,physical_RT_executions=0,final_goal='NOT_JUDGED',
        input_sha256={str(p):B.sha(p) for p in (sources/'sources.json',c_full,native_full,cell_map,atomic/'levels.csv',Path(__file__).resolve())})
    with out.open('x') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    return dict(negative_samples=len(cases),dominant_ions=sorted({(r['top_contributors'][0]['Z'],
        r['top_contributors'][0]['ion_number']) for r in cases}),
        dominant_inverted_in_parent_count=sum(any(r['top_contributors'][0]['parent_inverted']) for r in cases))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('sources','atomic','C-full-population','native-full-population','cell-map','out'):
        parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args()
    print(json.dumps(analyze(a.sources,a.atomic,a.C_full_population,a.native_full_population,a.cell_map,a.out)))
