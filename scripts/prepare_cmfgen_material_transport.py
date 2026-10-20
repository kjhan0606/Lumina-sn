#!/usr/bin/env python3
"""같은 실제 blue packet에서 물질 기반 cell 수송의 전체 angular 고정점을 비교한다.

새 사설 archive/packet만 생성한다. 원래 build/whole/원자자료를 바꾸거나
전주파수·스펙트럼·에너지/앵커 수용으로 승격하지 않는다.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from prepare_cmfgen_mom_capture import FLAGS,command,once,sha
from analyze_cmfgen_formal_capture import formal_state,geometry,rays,compensated_moments


def material_kernel(source):
    source=once(source,'      USE LUMINA_POSITIVE_SC,ONLY:SC_BEZIER_SOURCE\n',
        '      USE LUMINA_POSITIVE_SC,ONLY:SC_BEZIER_SOURCE\n      USE LUMINA_MATERIAL_CELL,ONLY:MATERIAL_CELL_TRANSFER\n')
    source=once(source,'      integer :: luer,error_lu\n',
        '      integer :: luer,error_lu\n      integer lum_status,lum_evaluations\n      real(ldp) lum_error\n')
    assignments=[
        ('\tray(ip)%I_m(iz)=ray(ip)%I_m(iz-1)*ee+t1\n','M','iz-1','iz'),
        ('      ray(ip)%I_m(nzz)=ray(ip)%I_m(nzz-1)*ee+\n     *     alpha*source_prime(nzz-1)+beta*source_prime(nzz)\n','M','nzz-1','nzz'),
        ('        ray(ip)%I_p(iz)=ray(ip)%I_p(iz+1)*ee+t1\n','P','iz+1','iz'),
        ('        ray(ip)%I_p(iz)=ray(ip)%I_p(iz+1)*ee+beta*source_prime(iz)+gamma*source_prime(iz+1)\n','P','iz+1','iz')]
    for marker,direction,up,down in assignments:
        sign=direction.lower();pair=f'[{up},{down}]'
        replacement=f'''      if(fp_enabled())then
        call material_cell_transfer(ray(ip)%i_{sign}({up}),chi({pair}),eta({pair}),
     *       ray(ip)%b_{sign}({pair}),ray(ip)%i_{sign}_prev({pair}),
     *       ray(ip)%s_{sign}({down})-ray(ip)%s_{sign}({up}),nu_dnu,1.e-10_ldp,
     *       ray(ip)%i_{sign}({down}),lum_error,lum_evaluations,lum_status)
        if(lum_status/=0)then
          write(6,*)'MATERIAL_CELL_FAIL',ip,{up},{down},lum_status,lum_error
          stop 87
        end if
      else
'''+marker+'      end if\n'
        source=once(source,marker,replacement)
    if source.count('call material_cell_transfer(')!=4:raise ValueError('네 RTE 진행 지점이 아님')
    return source


def prepare(base,out):
    manifest=json.loads((base/'formal-primary-build.json').read_text())
    if manifest.get('source_interpolation')!='positive_continuous_quadratic_Bezier':
        raise ValueError('검증된 v11 기준 필요')
    for name,h in manifest['output_sha256'].items():
        if sha(Path(name))!=h:raise ValueError('봉인 baseline 변경: '+name)
    parent=Path(manifest['parent']);transport=json.loads((parent/'sweep-build.json').read_text())
    native=Path(transport['native']);formal=Path(transport['formal_build']);mom=Path(transport['mom_build'])
    module=Path(__file__).with_name('cmfgen_material_cell.f90');shutil.copy2(module,out/module.name)
    (out/'solve_cmf_formal_v2.f').write_text(material_kernel((base/'solve_cmf_formal_v2.f').read_text()))
    includes=['-I'+str(p) for p in (out,base,parent,formal,mom,native/'mod')]
    command(['gfortran','-O','-fopenmp','-ffree-line-length-none',*includes,'-c',module.name],out)
    command(['gfortran',*FLAGS,*includes,'-c','solve_cmf_formal_v2.f'],out)
    shutil.copy2(base/'libsubs.a',out/'libsubs.a')
    command(['ar','r','libsubs.a','solve_cmf_formal_v2.o'],out)
    replay=Path(__file__).with_name('replay_cmfgen_formal_primary.f90');shutil.copy2(replay,out/replay.name)
    common=[out/'cmfgen_material_cell.o']+[base/(n+'.o') for n in
        ('cmfgen_formal_primary','cmfgen_positive_sc','cmfgen_formal_primary_whole')]
    common += [parent/(n+'.o') for n in ('cmfgen_mom_acceleration','lumina_affine_boundary',
        'lumina_mom_diag','lumina_formal_diag','lumina_whole_monitor')]
    common += [formal/'ext_rel_grid_v4.o',mom/'mod_jrel_v9.o']
    exe=out/'material-packet.exe'
    command(['gfortran','-O','-fopenmp','-ffree-line-length-none',*includes,str(out/replay.name),
        *map(str,common),'-L'+str(out),'-L'+str(native/'lib'),'-lsubs','-lpgplt','-lmod',
        str(native/'lib/libmy_lapack.a'),str(native/'lib/libmy_blas.a'),'-o',str(exe)],out)
    inventory=json.loads((formal/'formal-build.json').read_text())['ray_inventory']
    producers=[Path(__file__),module,replay,base/'formal-primary-build.json',base/'solve_cmf_formal_v2.f',*common]
    return exe,inventory,producers


def run(base,packet,reference,out):
    if out.exists():raise ValueError('기존 전체 angular packet 수리 증거 보존')
    out.mkdir();shutil.copy2(__file__,out/Path(__file__).name)
    exe,inventory,producers=prepare(base,out)
    outcomes=[];roots=[]
    for label,binary in (('reference',reference),('material',exe)):
        root=out/label;root.mkdir();roots.append(root)
        env=dict(os.environ,LUMINA_FORMAL_PRIMARY='1',LUMINA_FORMAL_PRIMARY_WHOLE='',
            LUMINA_FORMAL_CELL_BUDGET='',LUMINA_WHOLE_MODE='',LUMINA_MOM_CAPTURE_MODE='')
        start=time.monotonic()
        with (root/'stdout').open('xb') as stdout,(root/'stderr').open('xb') as stderr:
            proc=subprocess.run([str(binary),str(packet)],cwd=root,env=env,stdout=stdout,stderr=stderr,timeout=120)
        outcomes.append(dict(label=label,returncode=proc.returncode,seconds_CPU1=time.monotonic()-start))
        if proc.returncode:
            (out/'packet-failure.json').write_text(json.dumps(outcomes,indent=2)+'\n')
            raise ValueError('전체 angular packet 실패: '+label)
    names=('PRIMARY_GEOMETRY.bin','PRIMARY_BLUE_FINAL.bin')
    unchanged={name:(roots[0]/name).read_bytes()==(roots[1]/name).read_bytes() for name in names}
    if not all(unchanged.values()):raise ValueError('의도치 않은 geometry/blue 변경')
    results=[];states=[]
    for root in roots:
        g=geometry(root/'PRIMARY_GEOMETRY.bin');rs=rays(root/'PRIMARY_RAYS_FINAL.bin')
        state=formal_state(root/'PRIMARY_FORMAL_FINAL.bin',inventory)
        values=np.array([state['ray'][k+'NU_STORE']['values'] for k in 'JHKN']);states.append(values)
        independent,bound=compensated_moments(g,rs)
        if np.any(abs(values-independent)>bound):raise ValueError('angular 모멘트 재현 불일치')
        if any(np.any(v<0) for ray in rs['rays'] for v in ray.values()):raise ValueError('음의 실제 광선 강도')
        history=list(csv.DictReader((root/'PRIMARY_REPLAY.csv').open()))
        last=history[-1]
        if any(float(last[k])>1e-8 for k in ('J_residual','source_residual','H_change_over_J')):
            raise ValueError('source 고정점 미완료')
        results.append(dict(calls=len(history),last=last))
    scale=np.maximum(states[0][0],states[1][0]);change=abs(states[1]-states[0])/scale[None,:]
    report=dict(scope=__doc__,physical_reconstruction='각 native s cell의 양의 linear chi/eta/b/blue에서 직접 formal 적분',
        fixed_actual_blue_and_boundary=True,unchanged=unchanged,outcomes=outcomes,source_fixed_points=results,
        maximum_J_H_K_N_change_over_local_J=change.max(axis=1).tolist(),
        affected_J_depth_1based=int(np.argmax(change[0]))+1,
        input_sha256={str(p):sha(p) for p in [*producers,reference]},
        output_sha256={str(p):sha(p) for root in roots for p in (root/'PRIMARY_FORMAL_FINAL.bin',root/'PRIMARY_REPLAY.csv')},
        whole_executable_released=False,conditional_packet_not_whole=True,energy_pass='NOT_JUDGED',final_goal='NOT_JUDGED')
    (out/'material-transport.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('outcomes','maximum_J_H_K_N_change_over_local_J','affected_J_depth_1based')},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('base','packet','reference','out'):parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args();run(*(getattr(a,k).resolve() for k in ('base','packet','reference','out')))
