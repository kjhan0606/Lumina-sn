#!/usr/bin/env python3
"""완료 BB90의 같은 순천이율에서 광자/원 native 선 열장부만 비교한다.

새 BF/BB 적분·물질해·수송 없이 저장 자료를 읽는다. 실제 온도해 영향이나
native 이산 연산자 동등성은 입증하지 않는다. 출력은 표준출력 JSON뿐이다.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np

import cmfgen_model_continuum as B
import cmfgen_model_sources as S


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def compare(bb,bf):
    bb=bb.resolve();bf=bf.resolve()
    bbreport=json.loads((bb/'result.json').read_text());bfreport=json.loads((bf/'result.json').read_text())
    if (bbreport['state']!='COMPLETE_LINEAR_FIELD_BB90' or
        bfreport['state']!='COMPLETE_FROZEN_MATERIAL_BF_90'):
        raise ValueError('완료한 동일 원물질 입력 필요')
    for root,report in ((bb,bbreport),(bf,bfreport)):
        for name,h in report['output_sha256'].items():
            if digest(root/name)!=h:raise ValueError('저장 결과 변경')
    with np.load(bb/'line-roster.npz') as f:lines={k:f[k] for k in f.files}
    with np.load(bf/'material-snapshot.npz') as f:snapshot={k:f[k] for k in f.files}
    base_path=Path('/gpfs/kjhan/cmfgen_runs/lumina_extended_4h_20261022-rJGMVR/gpu-local-plan.json')
    config_path=Path(json.loads(base_path.read_text())['config']);c=json.loads(config_path.read_text())
    plan_path=Path(c['plan']);plan=json.loads(plan_path.read_text())
    run=Path(plan['initial_material']);level_path=Path(c['cells'])/'atomic/levels.csv'
    common_field=str(Path(plan['same_material_reference_field'])/'FORMAL_PRIMARY_FIELDS.bin')
    if (bbreport['input_sha256'][common_field]!=plan['reference_field_sha256'] or
        bfreport['data_sha256'][common_field]!=plan['reference_field_sha256']):
        raise ValueError('서로 다른 복사장')
    identities={str(p):digest(p) for p in (base_path,config_path,plan_path,level_path,
        run/'MODEL',run/'MODEL_SPEC',run/'atomic_links.txt',Path(__file__))}
    for path in (level_path,run/'MODEL',run/'MODEL_SPEC',run/'atomic_links.txt'):
        if (identities[str(path)]!=bbreport['input_sha256'][str(path)] or
            identities[str(path)]!=bfreport['data_sha256'][str(path)]):
            raise ValueError('BF/BB 원자/모형 계보 불일치')
    rows=list(csv.DictReader(level_path.open()))
    models=B.atomic.load_cmfgen_model_spec(run/'MODEL_SPEC',run/'atomic_links.txt')
    links=B.atomic.load_cmfgen_links(run/'atomic_links.txt')
    lookup={(int(r['atomic_number']),int(r['ion_number'])+1,int(r['level_number'])+1):i
            for i,r in enumerate(rows)}
    mean_edge=np.full(len(rows),np.nan)
    for (z,stage),model in models.items():
        path=Path(links[z,stage]['osc']);identities[str(path)]=digest(path)
        if identities[str(path)]!=bbreport['input_sha256'][str(path)]:raise ValueError('OSC 원자료 변경')
        levels=B.parse_osc(path,include_transitions=False).levels[:model['n_full']]
        # GENOSC_V9는 보조 nu15 열을 버리고 원 파수에서 FEDGE를 다시 만든다.
        header=re.findall(r'^\s*(\S+)\s+!Ionization energy\b',path.read_text(),re.M)
        if len(header)!=1 or len(levels)!=model['n_full']:raise ValueError('원 FEDGE 입력 누락')
        ion_cm=float(header[0].replace('D','E'))
        edge=(ion_cm-levels['E_cm'])*B.C*1.e-15
        members=np.array([lookup[z,stage,l+1] for l in range(len(levels))])
        sl=np.array([int(rows[i]['super_level']) for i in members])
        # AVE_LEVEL_ENERGY: 온도·occupation 비가 아니라 g로 raw EDGE를 평균한다.
        for group in np.unique(sl):
            selected=sl==group
            mean=np.sum(levels['g'][selected]*edge[selected],dtype=np.longdouble)/np.sum(
                levels['g'][selected],dtype=np.longdouble)
            mean_edge[members[selected]]=float(mean*1.e15)
    if not np.isfinite(mean_edge).all():raise ValueError('SL 평균 문턱 누락')
    columns={tuple(map(int,key)):i for i,key in enumerate(snapshot['columns'])}
    full_to_sl=np.array([columns[int(r['atomic_number']),int(r['ion_number']),int(r['super_level'])] for r in rows])
    zfull=np.array([int(r['atomic_number']) for r in rows]);density=snapshot['superlevel'].sum(axis=0)
    if (len(rows),len(lines['nu0']),len(density))!=(24542,790493,90):raise ValueError('전모형/90점 누락')
    enabled=S.control(run/'MODEL','SCL_LN')=='T'
    if S.control(run/'MODEL','SCL_SL_OPAC','F')!='F':raise ValueError('다른 소광 재척도 계약')
    limit=S.number(run/'MODEL','SCL_LN_FAC');density_limit=S.number(run/'MODEL','SCL_DEN_LIM','1e30')
    jbar=np.load(bb/'jbar.npy',mmap_mode='r');status=np.load(bb/'status.npy');done=np.load(bb/'lines-complete.npy')
    if not done.all() or not np.isin(status,(1,2)).all():raise ValueError('BB 결손')
    totals={int(z):np.zeros((3,90),dtype=np.longdouble) for z in np.unique(zfull)}
    counts=dict(lines=0,within_native_ratio_window=0,changed_by_more_than_one_percent=0,same_SL=0)
    try:
        for first in range(0,len(jbar),4096):
            last=min(len(jbar),first+4096);lo=lines['lower'][first:last];up=lines['upper'][first:last]
            nu=lines['nu0'][first:last];A=lines['A'][first:last];gratio=lines['g_ratio'][first:last]
            ratio=(mean_edge[lo]-mean_edge[up])/nu
            selected=np.abs(ratio-1)<=limit
            factor=np.where(selected,ratio,1.) if enabled else np.ones_like(ratio)
            effective=np.where(density[None,:]>=density_limit,1.,factor[:,None])
            nl=snapshot['superlevel'][full_to_sl[lo]]*snapshot['fraction'][lo]
            nu_pop=snapshot['superlevel'][full_to_sl[up]]*snapshot['fraction'][up]
            bul=A*B.C**2/(2*B.H*nu**3)
            forward=np.asarray(nl*bul[:,None]/gratio[:,None]*jbar[first:last],dtype=np.longdouble)
            reverse=np.asarray(nu_pop*(A[:,None]+bul[:,None]*jbar[first:last]),dtype=np.longdouble)
            energy=np.asarray(B.H*nu,dtype=np.longdouble)[:,None]
            photon=(forward-reverse)*energy;native=photon*effective;gross=(forward+reverse)*energy
            if not all(np.isfinite(v).all() for v in (photon,native,gross)):raise ValueError('열장부 비유한')
            for z in np.unique(zfull[lo]):
                group=zfull[lo]==z
                totals[int(z)]+=np.stack([a[group].sum(axis=0) for a in (photon,native,gross)])
            counts['lines']+=len(lo);counts['within_native_ratio_window']+=int(selected.sum())
            counts['changed_by_more_than_one_percent']+=int(np.count_nonzero(np.abs(factor-1)>.01))
            counts['same_SL']+=int(np.count_nonzero(full_to_sl[lo]==full_to_sl[up]))
        by_element=[]
        for z,(physical,native,gross) in sorted(totals.items()):
            delta=native-physical;scaled=np.divide(abs(delta),gross,out=np.zeros_like(gross),where=gross>0)
            k=int(np.argmax(scaled))
            by_element.append(dict(Z=z,maximum_delta_over_gross=float(scaled[k]),worst_native_node=k,
                physical_at_worst=float(physical[k]),native_thermal_at_worst=float(native[k]),
                delta_at_worst=float(delta[k]),gross_at_worst=float(gross[k]),
                maximum_absolute_delta=float(abs(delta).max())))
        if any(digest(Path(p))!=h for p,h in identities.items()):raise ValueError('후처리 중 원자료 변경')
        return dict(scope=__doc__,source_BB_result_sha256=digest(bb/'result.json'),
            source_BF_result_sha256=digest(bf/'result.json'),input_sha256=identities,counts=counts,
            native_SCL_LN=enabled,native_ratio_window=limit,native_density_limit=density_limit,
            nuclei_density_min=float(density.min()),nuclei_density_max=float(density.max()),
            density_override_points=int(np.count_nonzero(density>=density_limit)),by_element=by_element,
            edge_rule='GENOSC_V9: (ionization_cm - E_cm) * c * 1e-15; ignore printed nu15',
            new_transport=0,new_native=0,new_BF_BB_integrals=0,material_updates=0,
            static_vs_dynamic_internal_energy_compared=False,final_goal='NOT_MET')
    finally:jbar._mmap.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bb',type=Path,required=True);p.add_argument('--bf',type=Path,required=True)
    args=p.parse_args();print(json.dumps(compare(args.bb,args.bf),indent=2,allow_nan=False))
