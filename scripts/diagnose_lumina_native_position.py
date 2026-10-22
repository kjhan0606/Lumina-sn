#!/usr/bin/env python3
"""물질70/J71의 Lumina F를 CMFGEN native 두 반경에 직접 표본화한다.

원 native-spatial C 물리식/계수/경계와 전주파수 고정 J를 유지한다.
선택한 상풍 비교기에서는 구간 주파수 연산자만 격리 교체한다. 한 번의 F이며,
두 코드의 각도·주파수 연산자 동등성이나
자기일관 Si/S 이온화 분포를 증명하지 않는다.
"""
import argparse
import json
from pathlib import Path
import resource
import shutil
import socket
import time

import numpy as np

from diagnose_lumina_native_spatial_fixed import (_operator,project_native_scattering)
from diagnose_lumina_outer_fixed_point import PROJECT,INPUT,LIBRARY
from diagnose_lumina_shell37_endpoint_sensitivity import PAIR,SOURCE,save_array,save_record
from lumina_outer_native_spatial import NativeSpatialGeometry,set_native_fixed
from lumina_upwind_outer_variant import build_variant as build_upwind_variant
from lumina_outer_prepared import PreparedInputs
from prepare_lumina_outer_input import read_auxiliary,sha
from run_cmfgen_formal_primary import fields


PRIOR=Path('/gpfs/kjhan/cmfgen_runs/lumina_native_spatial_20261022-bq4ny1')
BANDS=((100,400),(400,800),(800,1600),(1600,3000))


def photon_band(nu,value,mask):
    return float(np.trapezoid(value[mask]/nu[mask],nu[mask]))


def run(root,shell,subdivisions=4,upwind_p0=False):
    root=Path(root).resolve()
    if (socket.gethostname().lower()!='lageunha' or Path.cwd().resolve()!=PROJECT or
            shell not in (37,39) or subdivisions not in (4,8) or
            type(upwind_p0) is not bool or
            not root.is_dir() or list(root.iterdir())):
        raise ValueError('Lageunha·프로젝트·셀37/39·빈 새 root 필요')
    if shutil.disk_usage(root).free<2*1024**3:
        raise ValueError('진단 자료 저장 공간 부족')
    started=time.monotonic()
    prior=json.loads((PRIOR/'execution.json').read_text())
    pair=json.loads((PAIR/'execution.json').read_text())
    source=json.loads((SOURCE/'sources.json').read_text())
    library=PRIOR/'src/outer-native-fixed.so'
    references=[Path(name) for name in pair['input_sha256']
                if Path(name).name=='FORMAL_PRIMARY_FIELDS.bin']
    if (prior['status']!='FIXED_J_NATIVE_SPATIAL_DIAGNOSTIC_COMPLETE' or
            prior['F_calls_completed']!=3 or pair['material_record']!=70 or
            pair['reference_record']!=71 or len(references)!=1 or
            sha(library)!=prior['variant_library_sha256'] or
            sha(PAIR/'Jref.npy')!=pair['Jref']['sha256'] or
            sha(PRIOR/'native-4.npy')!=prior['native-4']['field']['sha256'] or
            sha(INPUT/'boundary/preparation.json')!=
                prior['input_sha256'][str(INPUT/'boundary/preparation.json')] or
            sha(INPUT/'cells/cells.json')!=
                prior['input_sha256'][str(INPUT/'cells/cells.json')] or
            sha(SOURCE/'sources.json')!=prior['input_sha256'][str(SOURCE/'sources.json')] or
            sha(SOURCE/'sources.bin')!=source['output_sha256']['sources.bin'] or
            sha(LIBRARY)!=pair['input_sha256'][str(LIBRARY)] or
            sha(references[0])!=pair['input_sha256'][str(references[0])]):
        raise ValueError('원 native 수송/물질70/J71 입력 신원 불일치')
    record=dict(schema='lumina-native-position-fixed-v1',status='PREPARING',shell=shell,
        native_indices=dict(outer=88-shell,inner=89-shell),subdivisions=subdivisions,
        material_record=70,
        reference_record=71,F_calls_completed=0,physical_RT_executions=0,SE_RE_updates=0,
        final_goal='NOT_MET',resources=dict(host='lageunha',CPU=16,GPU=0,
            memory_GiB=64,wall_seconds_max=3600,F_calls_max=1,automatic_retry=0),
        input_sha256={str(path):sha(path) for path in
            (PRIOR/'execution.json',PAIR/'execution.json',PAIR/'Jref.npy',
             PRIOR/'native-4.npy',SOURCE/'sources.json',SOURCE/'sources.bin',
             INPUT/'boundary/preparation.json',INPUT/'cells/cells.json',
             library,references[0],LIBRARY)},
        source_sha256={str(Path(__file__)):sha(Path(__file__)),
            str(Path(__file__).with_name('lumina_outer_native_spatial.py')):
                sha(Path(__file__).with_name('lumina_outer_native_spatial.py'))})
    save_record(root/'execution.json',record)
    if upwind_p0:
        library=build_upwind_variant(root/'src')
        record['operator_scheme']='C-path P0 CMF first-order blue upwind; linear blue intensity within each subsegment; not CMFGEN 3-point SC'
        record['operator_library_sha256']=sha(library)
        for path in (Path(__file__).with_name('lumina_upwind_outer_variant.py'),
                     Path(__file__).resolve().parents[1]/'src/lumina_upwind_p0.c'):
            record['source_sha256'][str(path)]=sha(path)
    else:
        record['operator_scheme']='continuous characteristic raw positive; original native-spatial comparison'
    save_record(root/'execution.json',record)
    with PreparedInputs(LIBRARY,INPUT/'boundary',INPUT/'cells') as loaded:
        if loaded.plan.n!=89 or len(loaded.nu)!=1115020:
            raise ValueError('전주파수/native90 원 계획이 아님')
        jref=np.load(PAIR/'Jref.npy',mmap_mode='r',allow_pickle=False)
        old=np.load(PRIOR/'native-4.npy',mmap_mode='r',allow_pickle=False)
        geometry,formal=fields(references[0])
        try:
            if (jref.shape!=(91,len(loaded.nu)) or old.shape!=jref.shape or
                    formal['field'].shape[1:]!=(8,90) or
                    not np.allclose(geometry[0]*1e10,loaded.plan.g.native_radius,
                                    rtol=1e-12,atol=0)):
                raise ValueError('참조/수송 native 반경 또는 주파수 차원 불일치')
            native_j,j_difference=project_native_scattering(formal,loaded.nu)
            ref_nu=np.asarray(formal['freq'][::-1])*1e15
            log_ref=np.log(ref_nu);log_own=np.log(loaded.nu)
            native_indices=(88-shell,89-shell)
            cmf_native=np.ascontiguousarray(np.stack([
                np.interp(log_own,log_ref,np.asarray(formal['field'][::-1,0,node]))
                for node in native_indices]))
        finally:
            formal._mmap.close()
        chi,es,true_eta=read_auxiliary(SOURCE/'sources.bin',source,loaded.nu,list(range(90)))
        total_eta=true_eta+es*native_j
        if not np.isfinite(total_eta).all() or np.any(total_eta<0):
            raise ValueError('원 native 방출률 오류')
        plan=NativeSpatialGeometry(loaded.plan.g,subdivisions=subdivisions,
                                   extra_native_indices=native_indices)
        record.update(status='RUNNING',native_radii_cm=plan.sample_radii[-2:].tolist(),
            actual_ray_count=len(plan.paths),half_segments=plan.half_segments,
            source_rule=prior['source_rule'],J_source_vs_J_formal_max_row_normalized=j_difference,
            output_row_order=['native_outer','native_inner','cell37','cell39'],
            input_J_last_two_rows='unchanged original reference_indices, not output native positions')
        save_record(root/'execution.json',record)
        with _operator(library,plan,loaded) as operator:
            set_native_fixed(operator,chi,total_eta,jref)
            output,info=operator.apply_j(jref)
        sampled=np.ascontiguousarray(output[[-2,-1,37,39]])
        record['output']=save_array(root/'F-native-positions.npy',sampled)
        record['CMFGEN_native_J']=save_array(root/'CMFGEN-native-J.npy',cmf_native)
        record['actual_ray_threads']=info['actual_ray_threads']
        bands={}
        wave=2.99792458e18/loaded.nu
        for lo,hi in BANDS:
            mask=(wave>=lo)&(wave<hi)
            label=f'{lo}-{hi}A'
            rows={}
            for index,name in enumerate(('outer','inner')):
                denominator=photon_band(loaded.nu,cmf_native[index],mask)
                if not denominator>0:
                    raise ValueError('CMFGEN native J 대역 광자 적분 없음')
                rows[name]=dict(F_over_CMFGEN_native_J=
                    photon_band(loaded.nu,sampled[index],mask)/denominator,
                    radius_cm=float(plan.sample_radii[89+index]))
            for index,center in enumerate((37,39)):
                denominator=photon_band(loaded.nu,old[center],mask)
                rows[f'center{center}']=dict(new_over_previous_native4=
                    photon_band(loaded.nu,sampled[2+index],mask)/denominator)
            bands[label]=rows
        record.update(status='NATIVE_POSITION_F_COMPLETE',bands=bands,
            F_calls_completed=1,physical_RT_executions=1,
            elapsed_seconds=time.monotonic()-started,
            peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        save_record(root/'execution.json',record)
        jref._mmap.close();old._mmap.close()
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-root',type=Path,required=True)
    parser.add_argument('--shell',type=int,choices=(37,39),required=True)
    parser.add_argument('--subdivisions',type=int,choices=(4,8),default=4)
    parser.add_argument('--upwind-p0',action='store_true',
                        help='같은 C 경로의 격리 1차 청색 상풍 P0 비교기; CMFGEN 3점 SC 아님')
    args=parser.parse_args()
    result=run(args.out_root,args.shell,args.subdivisions,args.upwind_p0)
    print(json.dumps(dict(root=str(args.out_root),status=result['status'],
        elapsed_seconds=result['elapsed_seconds'],bands=result['bands']),ensure_ascii=False))
