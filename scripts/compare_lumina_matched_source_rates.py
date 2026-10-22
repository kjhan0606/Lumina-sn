#!/usr/bin/env python3
"""셀37/39의 Si/S 생산 PHOT율을 같은 위치의 source 대리장과 대조한다.

CMFGEN native 계수/J_source의 두 끝점에서 총 source를 구성해 물리 반경
중간계수의 비로 계산한다. 이는 실제 CMFGEN 셀중심 J가 아니며 UV 밖은
기존 Jref를 유지한다. 기존 고정인구/PHOT/Milne 소비자를 그대로 사용한다.
"""
import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np

from compare_lumina_native_spatial_rates import (B, CONFIG, LABELS, OLD, PAIR,
                                                  PRIOR, SNAPSHOT, TARGETS)
from compare_lumina_checkpoint_rates import ion_rates_stream
from run_cmfgen_formal_primary import fields


SHELLS = (37,39)
BANDS = ((100,400),(400,800),(800,1600),(1600,3000))
FINE_BANDS = ((100,400),(400,800),*((lo,lo+100) for lo in range(800,1600,100)),
              (1600,3000))
TRANSPORT = Path('/gpfs/kjhan/cmfgen_runs/lumina_native_spatial_20261022-bq4ny1')
RATES = Path('/gpfs/kjhan/cmfgen_runs/lumina_native_spatial_rates_20261022-WS5BR0/result/rates.json')


def source_proxies(reference, own_nu, jref):
    geometry, formal = fields(reference)
    try:
        ref_nu = np.asarray(formal['freq'][::-1])*1e15
        if (len(ref_nu)!=197291 or ref_nu[0]!=own_nu[0] or ref_nu[-1]!=own_nu[-1] or
                not np.all(np.diff(ref_nu)>0)):
            raise ValueError('CMFGEN formal 주파수 대응 불일치')
        mask = (2.99792458e18/own_nu>=100)&(2.99792458e18/own_nu<3000)
        target = np.log(own_nu[mask]);origin = np.log(ref_nu)
        f = formal['field'][::-1]
        midpoint = np.asarray(jref[list(SHELLS)]).copy()
        endpoint = midpoint.copy()
        for row,shell in enumerate(SHELLS):
            outer,inner=88-shell,89-shell
            if not geometry[0,outer]>geometry[0,inner]:
                raise ValueError('native 반경/셀 대응 오류')
            items=[]
            for node in (outer,inner):
                chi=np.interp(target,origin,np.asarray(f[:,4,node]))/1e10
                eta=np.interp(target,origin,np.asarray(f[:,5,node]))/1e10
                es=np.interp(target,origin,np.asarray(f[:,6,node]))/1e10
                source_j=np.interp(target,origin,np.asarray(f[:,7,node]))
                if np.any(chi<=0) or np.any(eta<0) or np.any(es<0) or np.any(source_j<0):
                    raise ValueError('UV source proxy의 원계수 부호 오류')
                items.append((chi,eta+es*source_j))
            (a,x),(b,y)=items
            midpoint[row,mask]=(x+y)/(a+b)
            endpoint[row,mask]=.5*(x/a+y/b)
        if any(not np.isfinite(value).all() or np.any(value<0)
               for value in (midpoint,endpoint)):
            raise ValueError('source proxy 음수/비유한')
        return midpoint,endpoint,dict(uv_bins=int(np.count_nonzero(mask)),
            rule='100–3000A: native endpoint chi/eta/es/Jsource lognu-linear to Lumina nu, then chi-weighted midpoint or endpoint source average; outside UV: existing Jref')
    finally:
        formal._mmap.close()


def compare(out, bands=False, fine_bands=False):
    started=time.monotonic();out=Path(out)
    if out.exists():
        raise FileExistsError('기존 PHOT source 위치 판독 보존')
    old=json.loads((OLD/'rates.json').read_text())
    prior=json.loads(PRIOR.read_text())
    native=json.loads(RATES.read_text())
    pair=json.loads((PAIR/'execution.json').read_text())
    transport=json.loads((TRANSPORT/'execution.json').read_text())
    config=json.loads(CONFIG.read_text())
    plan=json.loads(Path(config['plan']).read_text())
    if (old['material_record']!=prior['material_record'] or
            prior['material_record']!=native['material_record']!=70 or
            prior['reference_record']!=native['reference_record']!=71 or
            pair['Jref']['sha256']!=prior['field_sha256']['CMFGEN_projected'] or
            plan['reference_field_sha256']!=old['reference_sha256'] or
            transport['F_calls_completed']!=3 or native['cells']!=89):
        raise ValueError('물질70/J71/생산율 비교 입력 세대 불일치')
    references=[Path(name) for name in pair['input_sha256']
                if Path(name).name=='FORMAL_PRIMARY_FIELDS.bin']
    if len(references)!=1 or B.sha(references[0])!=plan['reference_field_sha256']:
        raise ValueError('CMFGEN formal 원본 신원 변경')
    for name in ('compare_lumina_checkpoint_rates.py','cmfgen_model_continuum.py',
                 'cmfgen_parser.py','expand_atomic_data_cmfgen.py','cmfgen_phot_smoothing.py'):
        expected=[value for path,value in old['source_sha256'].items()
                  if Path(path).name==name or path==name]
        if len(expected)!=1 or B.sha(SNAPSHOT/name)!=expected[0]:
            raise ValueError('생산 PHOT 코드 신원 변경: '+name)
    jref=np.load(PAIR/'Jref.npy',mmap_mode='r',allow_pickle=False)
    native4=np.load(TRANSPORT/'native-4.npy',mmap_mode='r',allow_pickle=False)
    p0=np.load(TRANSPORT/'P0-4.npy',mmap_mode='r',allow_pickle=False)
    nu=np.load(config['grid'],mmap_mode='r',allow_pickle=False)
    try:
        if (B.sha(PAIR/'Jref.npy')!=pair['Jref']['sha256'] or
                B.sha(TRANSPORT/'native-4.npy')!=transport['native-4']['field']['sha256'] or
                B.sha(TRANSPORT/'P0-4.npy')!=transport['P0-4']['field']['sha256'] or
                jref.shape!=(91,len(nu)) or native4.shape!=jref.shape or p0.shape!=jref.shape):
            raise ValueError('봉인 장/격자 신원 변경')
        midpoint,endpoint,rule=source_proxies(references[0],nu,jref)
        material_path=Path(config['cells'])/'material_cells.bin'
        run=Path(plan['initial_material']);atomic=Path(config['cells'])/'atomic'
        rows,models,routes,policy,smoothing,counts=B.load_model(
            run,atomic,Path(config['phot_library']))
        source=json.loads((Path(config['reuse_sources']['cell_sources']['directory'])/
                            'sources.json').read_text())
        if policy!=source['process_selection'] or counts!=source['routes_by_ion']:
            raise ValueError('생산 PHOT route/선택 불일치')
        matter=B.read_material(material_path,rows,models)
        do_dis=B.dissolution_flag(run/'MODEL')
        if do_dis:
            matter=B.read_material(material_path,rows,models,
                B.material_occupation(rows,models,matter))
        full_check=B.consume_c_full_population(Path(config['full_population']),matter)
        selected={key:(value[:,SHELLS] if key in ('full','superlevel','occupation') else
                       value[list(SHELLS)] if key in ('physical','temperature','ne') else value)
                  for key,value in matter.items()}
        intervals=FINE_BANDS if fine_bands else BANDS if bands else ()
        band_masks={f'{lo}_{hi}':(2.99792458e18/nu>=lo)&(2.99792458e18/nu<hi)
                    for lo,hi in intervals}
        if intervals:
            band_masks['other']=~np.logical_or.reduce(list(band_masks.values()))
        def block(sl):
            radiation=dict(Jref=np.asarray(jref[list(SHELLS),sl]),
                native4=np.asarray(native4[list(SHELLS),sl]),
                P0=np.asarray(p0[list(SHELLS),sl]),
                midpoint_source=midpoint[:,sl],endpoint_source_average=endpoint[:,sl])
            for band,mask in band_masks.items():
                selected_mask=mask[sl][None,:]
                for label in ('Jref','native4','midpoint_source'):
                    radiation[label+'_'+band]=radiation[label]*selected_mask
            return radiation
        old_rows={(int(r['Z']),int(r['stage']),int(r['shell'])):r for r in native['rows']}
        output=[]
        for (z,stage),result in ion_rates_stream(nu,block,routes,rows,models,selected,
                                                   do_dis,targets=TARGETS):
            for index,shell in enumerate(SHELLS):
                previous=old_rows[z,stage,shell]
                all_photo={label:float(value['photo'][index])
                           for label,value in result['total'].items()}
                photo={label:all_photo[label] for label in
                       ('Jref','native4','P0','midpoint_source','endpoint_source_average')}
                for label,old_value in (('Jref',previous['baseline_CMFGEN_photo_cm3_s']),
                                        ('native4',previous['photo_cm3_s']['native-4']),
                                        ('P0',previous['photo_cm3_s']['P0-4'])):
                    if abs(photo[label]-old_value)>1e-9*max(photo[label],old_value):
                        raise ValueError('기존 생산율 재현 실패: '+str((z,stage,shell,label)))
                denom=photo['midpoint_source']
                if not denom>0:
                    raise ValueError('양의 중심 source 기준율 필요')
                band_result={}
                if intervals:
                    for band in band_masks:
                        part={label:all_photo[label+'_'+band]
                              for label in ('Jref','native4','midpoint_source')}
                        band_result[band]=dict(photo_cm3_s=part,
                            Jref_fraction_of_total=part['Jref']/photo['Jref'],
                            native4_over_midpoint_source=(part['native4']/part['midpoint_source']
                                if part['midpoint_source']>0 else None))
                    for label in ('Jref','native4','midpoint_source'):
                        total=sum(item['photo_cm3_s'][label] for item in band_result.values())
                        if abs(total-photo[label])>1e-8*photo[label]:
                            raise ValueError('대역별 생산율 합이 전율과 다름')
                output.append(dict(Z=z,stage=stage,shell=shell,routes=result['routes'],
                    photo_cm3_s=photo,
                    native4_over_midpoint_source=photo['native4']/denom,
                    P0_over_midpoint_source=photo['P0']/denom,
                    endpoint_average_over_midpoint_source=photo['endpoint_source_average']/denom,
                    Jref_over_midpoint_source=photo['Jref']/denom,
                    band_rates=band_result if intervals else None))
        if len(output)!=len(TARGETS)*len(SHELLS):
            raise ValueError('네 이온×두 셀 생산율 누락')
        schema=('lumina-matched-source-proxy-production-rates-fine-bands-v3' if fine_bands else
                'lumina-matched-source-proxy-production-rates-bands-v2' if bands else
                'lumina-matched-source-proxy-production-rates-v1')
        result=dict(schema=schema,
            scope='same material70/J71, original C full population/raw PHOT/Milne; UV-only source proxy is not center J',
            material_record=70,reference_record=71,shells=list(SHELLS),
            source_proxy=rule,band_intervals_A=[list(pair) for pair in intervals],rows=output,
            C_full_population_comparison=full_check,do_lev_dissolution=do_dis,
            phot_preprocessing=smoothing,elapsed_seconds=time.monotonic()-started,
            peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            input_sha256={str(path):B.sha(path) for path in
                (RATES,PRIOR,TRANSPORT/'execution.json',PAIR/'execution.json',CONFIG)},
            physical_RT_executions=0,SE_RE_updates=0,final_goal='NOT_MET')
        with out.open('x') as stream:
            json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
            stream.write('\n')
        return result
    finally:
        for item in (jref,native4,p0,nu):
            item._mmap.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--bands',action='store_true',help='원 PHOT율을 UV 네 대역과 나머지로 분해')
    parser.add_argument('--fine-bands',action='store_true',help='800–1600A만 100A씩 세분해 원 PHOT율 분해')
    a=parser.parse_args()
    if a.bands and a.fine_bands:
        parser.error('--bands와 --fine-bands는 동시에 지정하지 않는다')
    record=compare(a.out,bands=a.bands,fine_bands=a.fine_bands)
    print(json.dumps(dict(out=str(a.out),rows=record['rows'],
                          elapsed_seconds=record['elapsed_seconds']),ensure_ascii=False))
