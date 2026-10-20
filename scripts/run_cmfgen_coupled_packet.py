#!/usr/bin/env python3
"""실제 두 blue 이력을 고정한 한 주파수의 비채택 공동 반응.

native 행렬은 풀이 전 포획한다. 독립 근의 J만 다음 source trial에 전달하고
H/PSD 위반을 숨기거나 생산 해·새 주파수 이력으로 반환하지 않는다.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import struct
import subprocess
import time

import numpy as np
from scipy.linalg import solve

from analyze_cmfgen_mom_capture import Reader,capture,first_event
from analyze_cmfgen_formal_capture import rays
from analyze_cmfgen_initial_closure import angular_support
from analyze_cmfgen_sweep_context import floop_state
from prepare_cmfgen_mom_capture import sha
from solve_cmfgen_mom_kernel import first_repair_equation,linmon,linmon_matrix


def matrix_packet(path):
    r=Reader(path.read_bytes())
    if r.take(16).rstrip()!=b'LUMINACOUPLEMAT1':raise ValueError('공동 행렬 header 오류')
    n=r.integers()[0];frequency=float(r.reals()[0])
    if not 3<=n<=10000:raise ValueError('공동 행렬 차원 오류')
    matrix=r.reals(4*n).reshape((4,n),order='F');slope=r.reals(n)
    if r.take(16).rstrip()!=b'LUMINACOUPLEEND1' or r.offset!=len(r.raw):raise ValueError('공동 행렬 끝 오류')
    return dict(ND=n,frequency=frequency,matrix=matrix,slope=slope)


def solve_trial(state,packet):
    """같은 homogeneous LINMON 원식의 분기 Newton. 선형 풀이 합계100 이하."""
    base,b,adv=first_repair_equation(state,packet)
    j=state['modules']['jrel'];r=j['R']['values'];n=len(r)
    scale=j['GAM_REL']['values']*r*r
    x=scale*state['modules']['ray']['JNU_STORE']['values']
    def residual(y):
        f=base@y+adv*linmon(y,r)-b
        den=abs(base)@abs(y)+abs(adv*linmon(y,r))+abs(b)
        return float(np.max(np.divide(abs(f),den,out=np.zeros(n),where=den>0)))
    count=0
    for _ in range(100):
        D,_=linmon_matrix(x,r)
        y=solve(base+adv[:,None]*D,b);count+=1
        if not np.isfinite(y).all() or np.any(y<=0):
            return y,count,residual(y),False
        error=residual(y)
        x=y
        if error<1e-10:return x,count,error,True
    return x,count,residual(x),False


def run(run_root,build,out,max_seconds=250,max_trials=20,source_update='moment'):
    if out.exists():raise ValueError('기존 공동 반응 보존')
    if not 1<=max_trials<=20 or not 0<max_seconds<=250:raise ValueError('공동 반응 예산 오류')
    if source_update not in ('moment','formal'):raise ValueError('공동 반응 source 갱신 모드 오류')
    if any(os.environ.get(k)!='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):
        raise ValueError('격리 CPU1 필수')
    started=time.monotonic();manifest=json.loads((build/'coupled-build.json').read_text())
    for name,h in manifest['output_sha256'].items():
        if sha(Path(name))!=h:raise ValueError('공동 fixture 변경')
    parent=Path(manifest['parent']);parent_record=json.loads((parent/'sweep-build.json').read_text())
    inventory=Path(parent_record['mom_build'])/'state_inventory.json'
    inv=json.loads(inventory.read_text());original=capture(run_root/'MOM_KERNEL.bin',inv)
    event=first_event(run_root/'MOM_FAULT.bin');control=floop_state(run_root/'FLOOP_STATE.bin')
    if event['kind']!=2 or event['iteration']!=1 or not control['coherent_es']:
        raise ValueError('이 fixture는 비초기 첫 H 사건과 coherent 산란에 한정')
    lib=ctypes.CDLL(str(Path(parent_record['mom_build'])/'analysis_v2/mon_mid.so'))
    ptr=np.ctypeslib.ndpointer(dtype=np.float64,ndim=1,flags='C_CONTIGUOUS')
    lib.native_mid.argtypes=[ctypes.c_int,ptr,ptr,ptr]
    out.mkdir();previous=None;rows=[];completed=False;reason='outer_trial_limit';snapshot_bytes=0
    for trial in range(max_trials):
        remaining=max_seconds-(time.monotonic()-started)
        if remaining<2:reason='offline_time_limit';break
        root=out/('trial_%02d'%trial);root.mkdir()
        args=[str(build/'coupled_packet.exe'),str(run_root)]
        if previous is not None:
            source=root/'SOURCE_J.bin';np.asarray(previous['J'],dtype='<f8').tofile(source);args.append(str(source))
        env=dict(os.environ,LUMINA_MOM_CAPTURE_MODE='CAPTURE_FIRST',LUMINA_WHOLE_MODE='',LUMINA_FORMAL_CAPTURE_MODE='')
        with (root/'native.stdout').open('xb') as stdout,(root/'native.stderr').open('xb') as stderr:
            proc=subprocess.run(args,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,timeout=min(30,remaining))
        if proc.returncode!=96:raise ValueError('첫 행렬 전용 포획 종료가 아님: '+str(proc.returncode))
        if (root/'TRIAL_BLUE_AFTER.bin').read_bytes()!=(run_root/'FORMAL_EVENT_rays_in.bin').read_bytes():
            raise ValueError('trial에서 blue 광선 이력이 전진함')
        state=capture(root/'TRIAL_KERNEL.bin',inv);packet=matrix_packet(root/'TRIAL_MATRIX.bin')
        j=state['modules']['jrel'];q=state['modules']['ray'];v=lambda k:j[k]['values']
        if state['phase']!=original['phase'] or packet['frequency']!=event['frequency']:raise ValueError('trial 입력 위상 오류')
        for group,names in (('jrel',('GAM_RSQJNU_PREV','GAM_RSQHNU_PREV')),('ray',('K_ON_J_PREV','KMID_ON_J_PREV','NMID_ON_J_PREV'))):
            for k in names:
                if not np.array_equal(state['modules'][group][k]['values'],original['modules'][group][k]['values']):
                    raise ValueError('MOM blue 이력 변경: '+k)
        n=packet['ND'];r=v('R');physical=v('GAM_REL')*r*r
        base,b,adv=first_repair_equation(state,packet)
        x,count,eq,ok=solve_trial(state,packet)
        if not ok:reason='moment_trial_not_solved';break
        bc=struct.unpack_from('=7d',(root/'TRIAL_BOUNDARY.bin').read_bytes(),36)
        flux_constant=v('BETA')[0]*physical[0]*bc[0]*v('Q')[0]/v('DTAU_H')[0]/(v('P_H')[0]+v('W')[0])
        jp,hp=v('GAM_RSQJNU_PREV'),v('GAM_RSQHNU_PREV')
        H=(v('HU')[:-1]*x[1:]-v('HL')[:-1]*x[:-1]+v('HS')[:-1]*hp[:-1]
           +v('EPS_PREV')[:-1]*(jp[:-1]+jp[1:])-v('EPS')[:-1]*(x[:-1]+x[1:]))
        H[0]+=flux_constant
        mid=np.zeros(n-1);lib.native_mid(n,r,np.ascontiguousarray(x),mid)
        support=angular_support(mid,H,q['KMID_ON_J']['values'][:-1]*(x[:-1]+x[1:]),q['NMID_ON_J']['values'][:-1]*(x[:-1]+x[1:]))
        J=x/physical;formal=q['JNU_STORE']['values'];source=capture(root/'TRIAL_ENTRY.bin',inv)['args']['JNU_SM']
        source_eta=control['ETA_CLUMP']+control['CHI_SCAT_CLUMP']*source
        primary_J=J if source_update=='moment' else formal
        matched_eta=control['ETA_CLUMP']+control['CHI_SCAT_CLUMP']*primary_J
        rel=lambda a,z:float(np.max(abs(a-z)/np.maximum(abs(a),abs(z))))
        lag=float(np.max(abs(matched_eta-source_eta)/matched_eta))
        formal_x=formal*physical
        f=base@formal_x+adv*linmon(formal_x,r)-b
        lag_term=v('DTAUONQ')*v('COH_VEC')*(formal_x-physical*source);lag_term[[0,-1]]=0
        den=abs(base)@abs(formal_x)+abs(adv*linmon(formal_x,r))+abs(b)+abs(lag_term)
        rayset=rays(root/'TRIAL_RAYS_OUT.bin')
        if not all(np.isfinite(a[k]).all() for a in rayset['rays'] for k in ('I_P','I_M')):
            raise ValueError('공동 반응 실제 광선 비유한값')
        ray_min=float(min(np.min(a[k]) for a in rayset['rays'] for k in ('I_P','I_M')))
        if ray_min<0:raise ValueError('공동 반응 실제 광선 음수')
        H_moment=H/np.asarray(.5*(v('GAM_REL')[:-1]+v('GAM_REL')[1:])*(.5*(r[:-1]+r[1:]))**2)
        H_primary=H_moment
        if source_update=='formal':
            H_primary=np.zeros(n-1)
            lib.native_mid(n,r,np.ascontiguousarray(physical*q['HNU_STORE']['values']),H_primary)
            H_primary/=.5*(v('GAM_REL')[:-1]+v('GAM_REL')[1:])*(.5*(r[:-1]+r[1:]))**2
        sample=dict(J=primary_J,H=H_primary,Q=v('Q'),F=q['K_ON_J']['values'])
        ray_support=angular_support(formal,q['HNU_STORE']['values'],q['KNU_STORE']['values'],q['NNU_STORE']['values'])
        row=dict(trial=trial,source_J_change=rel(primary_J,source),source_eta_lag=lag,moment_linear_solves=count,equation_residual=eq,
            formal_MOM_J_difference=rel(J,formal),formal_source_matched_equation_residual=float(np.max(abs(f+lag_term)/den)),
            H_MAX_VAL_edges_1based=(np.flatnonzero(abs(H)>np.maximum(x[:-1],x[1:]))+1).tolist(),angular_support=support,
            actual_ray_min=ray_min,blue_rays_equal=True,moment_blue_equal=True,
            formal_node_support=dict(minimum=ray_support['minimum'],node_1based=ray_support['edge_1based'],normalized_JHKN=ray_support['normalized_JHKN']))
        if previous is not None:
            row.update(J_change=rel(sample['J'],previous['J']),H_change_over_J=float(np.max(abs(sample['H']-previous['H'])/np.maximum(sample['J'][:-1],sample['J'][1:]))),
                       Q_change=rel(sample['Q'],previous['Q']),F_change=float(np.max(abs(sample['F']-previous['F']))))
            completed=(max(row['J_change'],row['H_change_over_J'],row['Q_change'],lag)<1e-8 and row['F_change']<=control['tolerance'])
        rows.append(row);previous={k:np.array(val,copy=True) for k,val in sample.items()}
        with (root/'trial-result.json').open('x') as f:json.dump(dict(metrics=row,nonaccepted_J=primary_J.tolist(),hypothetical_moment_J=J.tolist()),f,ensure_ascii=False,indent=2,allow_nan=False)
        snapshot_bytes+=sum((root/name).stat().st_size for name in ('TRIAL_ENTRY.bin','TRIAL_KERNEL.bin','TRIAL_MATRIX.bin','TRIAL_FORMAL.bin','TRIAL_RAYS_OUT.bin','TRIAL_BLUE_AFTER.bin','TRIAL_GEOMETRY.bin','TRIAL_BOUNDARY.bin'))
        if snapshot_bytes>256*1024**2:raise ValueError('공동 fixture 포획 예산 초과')
        if completed:reason='joint_fixed_point_not_physics_pass';break
    inputs=[run_root/name for name in ('MOM_CAPTURE.bin','MOM_KERNEL.bin','MOM_FAULT.bin','AFFINE_INPUT.bin','FORMAL_EVENT_in.bin','FORMAL_EVENT_rays_in.bin','FLOOP_STATE.bin')]
    report=dict(scope='실제 두 blue 이력을 고정한 같은 주파수의 비채택 공동 반응; 생산 반환/새 blue 아님',run=str(run_root),build=str(build),source_update=source_update,
        input_sha256={str(p):sha(p) for p in inputs},producer_sha256=sha(Path(__file__)),build_manifest_sha256=sha(build/'coupled-build.json'),
        actual_source_phase=event['phase'],trials=rows,joint_fixed_point=completed,stop_reason=reason,snapshot_bytes=snapshot_bytes,
        elapsed_seconds=time.monotonic()-started,production_physics='NOT_ACCEPTED',final_goal='NOT_JUDGED')
    with (out/'coupled-response.json').open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(out=str(out),joint_fixed_point=completed,stop_reason=reason,trials=len(rows),last=rows[-1] if rows else None),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('run','build','out'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--max-seconds',type=float,default=250);p.add_argument('--max-trials',type=int,default=20)
    p.add_argument('--source-update',choices=('moment','formal'),default='moment')
    a=p.parse_args();run(a.run,a.build,a.out,a.max_seconds,a.max_trials,a.source_update)
