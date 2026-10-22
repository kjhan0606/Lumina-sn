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

ROOT = Path('/gpfs/kjhan/cmfgen_runs/lumina_native_position39_20261022-Snv5iR')
CONFIG = Path('/gpfs/kjhan/cmfgen_runs/lumina_field0_checkpoint_20261020-EoA5Bi/input/config.json')
SHELL = 39


def compare(out):
    if out.exists():
        raise FileExistsError('기존 native-node 광흡수율 결과 보존')
    start=time.monotonic()
    record=json.loads((ROOT/'execution.json').read_text())
    config=json.loads(CONFIG.read_text())
    plan=json.loads(Path(config['plan']).read_text())
    if record['status']!='NATIVE_POSITION_F_COMPLETE' or record['shell']!=SHELL or \
       record['native_indices']!={'outer':49,'inner':50} or \
       record['material_record']!=70 or record['reference_record']!=71:
        raise ValueError('같은 물질·반경의 실제 수송이 아님')
    f_path=ROOT/'F-native-positions.npy'
    j_path=ROOT/'CMFGEN-native-J.npy'
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
                                                 do_dis,targets=TARGETS):
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
        if len(result)!=8:
            raise ValueError('네 이온·두 native 점 광흡수율 누락')
        out.mkdir(parents=False)
        report=dict(scope=__doc__,material_record=70,reference_record=71,
                    cells_with_populations=[SHELL],native_nodes=[49,50],
                    frequencies=len(nu),ions=[list(key) for key in TARGETS],
                    results=result,do_lev_dissolution=do_dis,
                    photo_preprocessing=smoothing,
                    input_sha256={str(p):B.sha(p) for p in (ROOT/'execution.json',
                        f_path,j_path,CONFIG,Path(config['grid']))},
                    elapsed_seconds=time.monotonic()-start,
                    peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    actual_new_transport_calls=0,SE_RE_updates=0,
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
    args=parser.parse_args()
    compare(args.out.resolve())
