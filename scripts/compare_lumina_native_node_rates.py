#!/usr/bin/env python3
"""물질70·같은 셀39 인구와 PHOT/Milne로 native49/50의 F와 J 광흡수율을 대조한다."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np


SNAPSHOT = Path('/gpfs/kjhan/cmfgen_runs/lumina_fixed_point_rates_20261022-2mOuiA/scripts')
sys.path.insert(0,str(SNAPSHOT))
import cmfgen_model_continuum as B
from compare_lumina_checkpoint_rates import ion_rates_stream, TARGETS
from run_cmfgen_formal_primary import fields

ROOT = Path('/gpfs/kjhan/cmfgen_runs/lumina_native_position39_20261022-Snv5iR')
CONFIG = Path('/gpfs/kjhan/cmfgen_runs/lumina_field0_checkpoint_20261020-EoA5Bi/input/config.json')
SHELL = 39


def project_rows(nu, values, target_nu):
    """같은 logν 선형 규칙으로 완성된 장만 다른 적분 격자에 읽어 온다."""
    nu=np.asarray(nu);values=np.asarray(values);target_nu=np.asarray(target_nu)
    if (nu.ndim!=1 or target_nu.ndim!=1 or values.ndim!=2 or
            values.shape[1]!=len(nu) or len(nu)<2 or len(target_nu)<2 or
            not np.isfinite(nu).all() or not np.isfinite(target_nu).all() or
            not np.isfinite(values).all() or np.any(nu<=0) or
            np.any(np.diff(nu)<=0) or np.any(np.diff(target_nu)<=0) or
            target_nu[0]<nu[0] or target_nu[-1]>nu[-1]):
        raise ValueError('전 주파수의 양의 증가 격자·유한 장·보간 범위 필요')
    source=np.log(nu);target=np.log(target_nu)
    return np.stack([np.interp(target,source,row) for row in values])


def compare(out, siii_detail=False, siii_profile=False, transport_root=ROOT,
            compare_grids=False):
    if out.exists():
        raise FileExistsError('기존 native-node 광흡수율 결과 보존')
    start=time.monotonic()
    transport_root=Path(transport_root).resolve()
    record=json.loads((transport_root/'execution.json').read_text())
    config=json.loads(CONFIG.read_text())
    plan=json.loads(Path(config['plan']).read_text())
    if record['status']!='NATIVE_POSITION_F_COMPLETE' or record['shell']!=SHELL or \
       record['native_indices']!={'outer':49,'inner':50} or \
       record['material_record']!=70 or record['reference_record']!=71:
        raise ValueError('같은 물질·반경의 실제 수송이 아님')
    f_path=transport_root/'F-native-positions.npy'
    j_path=transport_root/'CMFGEN-native-J.npy'
    if B.sha(f_path)!=record['output']['sha256'] or \
       B.sha(j_path)!=record['CMFGEN_native_J']['sha256']:
        raise ValueError('기존 F/J 전체 장 신원 변경')
    nu=np.load(config['grid'],mmap_mode='r',allow_pickle=False)
    f=np.load(f_path,mmap_mode='r',allow_pickle=False)
    j=np.load(j_path,mmap_mode='r',allow_pickle=False)
    try:
        if nu.shape!=(1115020,) or f.shape!=(4,len(nu)) or j.shape!=(2,len(nu)):
            raise ValueError('전체 주파수·두 native 점 배열 불일치')
        material_path=Path(config['cells'])/'material_cells.bin'
        run=Path(plan['initial_material'])
        rows,models,routes,policy,smoothing,counts=B.load_model(
            run,Path(config['cells'])/'atomic',Path(config['phot_library']))
        source=json.loads((Path(config['reuse_sources']['cell_sources']['directory'])/
                           'sources.json').read_text())
        if policy!=source['process_selection'] or counts!=source['routes_by_ion']:
            raise ValueError('원 PHOT route/선택 변경')
        matter=B.read_material(material_path,rows,models)
        do_dis=B.dissolution_flag(run/'MODEL')
        if do_dis:
            matter=B.read_material(material_path,rows,models,
                B.material_occupation(rows,models,matter))
        selected={key:(value[:,[SHELL]] if key in ('full','superlevel','occupation') else
                       value[[SHELL]] if key in ('physical','temperature','ne') else value)
                  for key,value in matter.items()}
        wave=2.99792458e18/nu
        masks={'100–400Å':(wave>=100)&(wave<400),
               '400–800Å':(wave>=400)&(wave<800),
               '800–1600Å':(wave>=800)&(wave<1600),
               '1400–1500Å':(wave>=1400)&(wave<1500)}
        if compare_grids:
            masks['600–750Å']=(wave>=600)&(wave<750)
        if siii_detail or siii_profile:
            for lo in range(400,800,50):
                masks[f'{lo}–{lo+50}Å']=(wave>=lo)&(wave<lo+50)
        if siii_profile:
            for lo in range(600,750,10):
                masks[f'{lo}–{lo+10}Å']=(wave>=lo)&(wave<lo+10)
        targets=((16,3),) if (siii_detail or siii_profile) else TARGETS

        def block(sl):
            radiation={}
            for index,label in enumerate(('outer49','inner50')):
                a=np.asarray(f[index:index+1,sl]);b=np.asarray(j[index:index+1,sl])
                radiation['F_'+label]=a
                radiation['J_'+label]=b
                for band,mask in masks.items():
                    radiation['F_'+label+'_'+band]=a*mask[sl][None,:]
                    radiation['J_'+label+'_'+band]=b*mask[sl][None,:]
            return radiation

        result=[]
        for (z,stage),value in ion_rates_stream(nu,block,routes,rows,models,selected,
                                                 do_dis,targets=targets):
            for label in ('outer49','inner50'):
                rates={name:float(item['photo'][0]) for name,item in value['total'].items()}
                pairs={}
                for band in ('full',*masks):
                    suffix='' if band=='full' else '_'+band
                    a,b=rates['F_'+label+suffix],rates['J_'+label+suffix]
                    pairs[band]=dict(F_photo_cm3_s=a,J_photo_cm3_s=b,
                                     F_over_J=a/b if b>0 else None)
                result.append(dict(Z=z,stage=stage,native_node=label,
                                   cell_population_shell=SHELL,routes=value['routes'],
                                   bands=pairs))
        if len(result)!=2*len(targets):
            raise ValueError('요청 이온·두 native 점 광흡수율 누락')
        grid_comparison=None
        if compare_grids:
            references=[Path(name) for name in record['input_sha256']
                        if Path(name).name=='FORMAL_PRIMARY_FIELDS.bin']
            if len(references)!=1 or B.sha(references[0])!=record['input_sha256'][str(references[0])]:
                raise ValueError('원 CMFGEN 완성 장 신원 불일치')
            _,original=fields(references[0])
            try:
                native_nu=np.asarray(original['freq'][::-1])*1e15
                native_j=np.asarray(original['field'][::-1,0,:][:,[49,50]]).T
                native_f=project_rows(nu,np.asarray(f[:2]),native_nu)
                native_mask=(2.99792458e18/native_nu>=600)&(2.99792458e18/native_nu<750)

                def native_block(sl):
                    ja,jb=native_j[0:1,sl],native_j[1:2,sl]
                    fa,fb=native_f[0:1,sl],native_f[1:2,sl]
                    mask=native_mask[sl][None,:]
                    return dict(J_outer49=ja,J_inner50=jb,F_outer49=fa,F_inner50=fb,
                                J_outer49_band=ja*mask,J_inner50_band=jb*mask,
                                F_outer49_band=fa*mask,F_inner50_band=fb*mask)

                native_rates=list(ion_rates_stream(native_nu,native_block,routes,rows,models,
                                selected,do_dis,targets=((16,3),)))
                if len(native_rates)!=1:
                    raise ValueError('원 격자 S III율 누락')
                grid_comparison=dict(scope='완성 F/J의 적분 격자와 logν 재표본 효과만; 새 수송 아님',
                    original_field_sha256={str(references[0]):record['input_sha256'][str(references[0])]},
                    original_frequency_count=len(native_nu),projected_frequency_count=len(nu),
                    original_grid_photo_cm3_s={key:float(value['photo'][0]) for key,value in
                        native_rates[0][1]['total'].items()},
                    projected_grid_photo_cm3_s={f'{kind}_{label}{suffix}':
                        next(row for row in result if row['native_node']==label)['bands'][band][field]
                        for kind,field in (('F','F_photo_cm3_s'),('J','J_photo_cm3_s'))
                        for label in ('outer49','inner50')
                        for suffix,band in (('', 'full'),('_band','600–750Å'))})
            finally:
                original._mmap.close()
        out.mkdir(parents=False)
        report=dict(scope=__doc__,material_record=70,reference_record=71,
                    cells_with_populations=[SHELL],native_nodes=[49,50],
                    transport_operator_scheme=record.get('operator_scheme',
                        'continuous characteristic raw positive; legacy record'),
                    frequencies=len(nu),ions=[list(key) for key in targets],
                    results=result,do_lev_dissolution=do_dis,
                    photo_preprocessing=smoothing,
                    input_sha256={str(p):B.sha(p) for p in (transport_root/'execution.json',
                        f_path,j_path,CONFIG,Path(config['grid']))},
                    elapsed_seconds=time.monotonic()-start,
                    peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    actual_new_transport_calls=0,SE_RE_updates=0,
                    grid_comparison=grid_comparison,
                    same_cell_population_not_native_physical_state=True,
                    final_goal='NOT_MET')
        (out/'rates.json').write_text(json.dumps(report,ensure_ascii=False,
                                                 indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(result=str(out/'rates.json'),rates=[
            dict(Z=r['Z'],stage=r['stage'],node=r['native_node'],
                 ratios={k:v['F_over_J'] for k,v in r['bands'].items()})
            for r in result],elapsed_seconds=report['elapsed_seconds']),ensure_ascii=False))
    finally:
        nu._mmap.close();f._mmap.close();j._mmap.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--siii-detail',action='store_true',
                        help='S III만 계산하고 400–800Å를 50Å씩 분해한다')
    parser.add_argument('--siii-profile',action='store_true',
                        help='S III만 계산하고 600–750Å를 10Å씩 추가 분해한다')
    parser.add_argument('--transport-root',type=Path,default=ROOT,
                        help='완료된 같은 물질70/J71 native 반경 F 실행 root')
    parser.add_argument('--compare-grids',action='store_true',
                        help='원 CMFGEN 격자와 C 격자에서 같은 S III F/J 광흡수율 재적분')
    args=parser.parse_args()
    compare(args.out.resolve(),args.siii_detail,args.siii_profile,args.transport_root,
            args.compare_grids)
