#!/usr/bin/env python3
"""물질70 원래90점의 국소 χ/η를 같은 물질에서 나온 J71 소스와 대조한다.

선택된 실제 native 주파수의 국소 계수 비교이며 새 복사장·스펙트럼·
수렴 또는 전 주파수 동등성의 증거로 사용하지 않는다.
"""
import argparse
import json
from pathlib import Path
import struct

import numpy as np

import cmfgen_model_sources as S
from run_cmfgen_formal_primary import fields
B=S.B

PLAN_SHA='51abafe6ae9e18be0736432eba8405a129fa9e68e2c8986ee3fde6b829c5c19f'


def compare(plan_path,seed,c_full,attribution,phot_library,kernel_library,out):
    if out.exists():raise ValueError('기존 국소 소스 대조 보존')
    if B.sha(plan_path)!=PLAN_SHA:raise ValueError('BRIDGE 봉인 계획 신원 불일치')
    plan=json.loads(plan_path.read_text())
    if plan['initial_native_record']!=70 or plan['reference_radiation_material_record']!=70:
        raise ValueError('동일 물질70/J71 대조가 아님')
    field=Path(plan['same_material_reference_field'])/'FORMAL_PRIMARY_FIELDS.bin'
    if B.sha(field)!=plan['reference_field_sha256']:raise ValueError('봉인 J71 소스 변경')
    geometry,records=fields(field)
    with (seed/'native_material_nodes.bin').open('rb') as stream:
        magic,ns,nc,nl,epoch=struct.unpack('<8sIIId',stream.read(28))
        stream.seek(12*nc,1)
        physical=np.frombuffer(stream.read(8*4*ns),dtype='<f8').reshape(ns,4)
    if magic!=b'LUMNOD01' or not np.allclose(physical[:,0],geometry[0]*1e10,rtol=2e-15,atol=0):
        raise ValueError('물질70 원래 깊이점과 J71 source geometry 불일치')
    trace=json.loads(attribution.read_text());requested=[r['frequency_Hz'] for r in trace['cases']]
    requested += [1e14,5e14,1e15,3e15,1e16]
    native_nu=np.asarray(records['freq'])*1e15
    indices=sorted({int(np.argmin(abs(np.log(native_nu/nu)))) for nu in requested},
                   key=lambda i:native_nu[i])
    nu=native_nu[indices]
    out.mkdir()
    summary=S.prepare(Path(plan['initial_material']),seed/'native_material_nodes.bin',seed/'atomic',
        phot_library,kernel_library,c_full,nu,out/'sources90')
    data=np.load(out/'sources90/sources.npz')
    total=sum(data[k] for k in ('bf_chi','ff_chi','bb_chi','es_chi'))
    eta=sum(data[k] for k in ('bf_eta','ff_eta','bb_eta'))
    # native의 공간 단위는10^10cm다. J 단위는 이미 cgs다.
    reference=np.asarray(records['field'][indices]).transpose(1,2,0)
    reference_chi=reference[4]*1e-10;reference_eta=reference[5]*1e-10;reference_es=reference[6]*1e-10
    cases=[]
    for depth,b in np.argwhere(total<0):
        cases.append(dict(depth_1based=int(depth)+1,native_frequency_record_1based=indices[b]+1,
            frequency_Hz=float(nu[b]),wavelength_A=float(B.C/nu[b]*1e8),
            raw_total_chi=float(total[depth,b]),reference_total_chi=float(reference_chi[depth,b]),
            raw_eta=float(eta[depth,b]),reference_eta=float(reference_eta[depth,b]),
            BF_chi=float(data['bf_chi'][depth,b]),FF_chi=float(data['ff_chi'][depth,b]),
            BB_chi=float(data['bb_chi'][depth,b]),es_chi=float(data['es_chi'][depth,b])))
    def relative(a,b):
        scale=np.maximum(abs(a),abs(b));return np.divide(abs(a-b),scale,out=np.zeros_like(a),where=scale>0)
    def stats(a):return dict(median=float(np.median(a)),p95=float(np.quantile(a,.95)),maximum=float(a.max()))
    np.savez(out/'comparison.npz',nu_Hz=nu,raw_chi=total,raw_eta=eta,
        reference_chi=reference_chi,reference_eta=reference_eta,reference_es=reference_es,
        all_positive_ff_chi=data['ff_chi'],native_target_ff_chi=data['ff_native_chi'])
    report=dict(schema='lumina-native70-local-source-comparison-v1',material_record=70,
        reference_radiation_record=71,reference_material_record=70,native_points=ns,
        selected_frequencies=len(nu),frequency_records_1based=[i+1 for i in indices],
        all_values=ns*len(nu),full_population_input=str(c_full),source_summary=summary,
        relative_error=dict(chi_total=stats(relative(total,reference_chi)),
            eta_true=stats(relative(eta,reference_eta)),es=stats(relative(data['es_chi'],reference_es))),
        signed_raw_vs_positive_reference=cases,
        native_source_changes='reference includes SRCE_CHK/zero-opacity replacement; raw coefficients do not',
        interpretation='same native depth POINTS/frequencies; not a field/ionization/spectrum comparison; no claim all residuals caused by SRCE_CHK',
        native_executions=0,physical_RT_executions=0,final_goal='NOT_JUDGED',
        input_sha256={str(p):B.sha(p) for p in (plan_path,seed/'native_material_nodes.bin',c_full,
            attribution,field,Path(__file__).resolve(),Path(__file__).with_name('run_cmfgen_formal_primary.py'))},
        output_sha256={'comparison.npz':B.sha(out/'comparison.npz'),
            'sources90/sources.json':B.sha(out/'sources90/sources.json')})
    with (out/'comparison.json').open('x') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    return {k:report[k] for k in ('native_points','selected_frequencies','relative_error','final_goal')}|dict(
        raw_negative_native_points=len(cases))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','seed','C-full-population','attribution','phot-library','kernel-library','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args()
    print(json.dumps(compare(a.plan,a.seed,a.C_full_population,a.attribution,a.phot_library,a.kernel_library,a.out)))
