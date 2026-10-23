#!/usr/bin/env python3
"""봉인 BRIDGE의 첫 전체 계수 생산/자체장0을 같은 자원 예약에서 실행한다.

실제 field0는 후보 명칭과 무관하게 원래 장/실행 예산에서1회다. 자동 재시도나
후속 물질 갱신은 하지 않으며, 부분 실패와 시간 초과의 산출물을 보존한다.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

import run_lumina_cell_candidate as D


def effective_resources(c,plan,reservation):
    original=dict(CPU=16,OMP=16,memory_GiB=32,GPU=0,parent_seconds=2700,
                  child_seconds=2400,node_pinned=False,retry=0)
    if plan['resources_per_run']!=original:raise ValueError('기존 자원/재시도 계약 불일치')
    if 'resource_amendment' not in c:
        return original,8,None
    path=Path(c['resource_amendment'])
    if D.sha(path)!=c['resource_amendment_sha256']:raise ValueError('자원 변경 SHA256 불일치')
    a=json.loads(path.read_text())
    approved={**original,'memory_GiB':64,'parent_seconds':10800,'child_seconds':10200}
    recovery=a.get('schema')=='lumina-field0-checkpoint-amendment-v1'
    reuse_retry=recovery or a.get('schema')=='lumina-field0-reuse-amendment-v2'
    continuation=reuse_retry or a.get('schema')=='lumina-field0-reuse-amendment-v1'
    number=6 if recovery else 5 if reuse_retry else 4 if continuation else 3
    previous=[1199911,1199926,1200155] if continuation else [1199911,1199926]
    if reuse_retry:previous.append(1200286)
    if recovery:previous.append(1200364)
    if (a.get('schema') not in ('lumina-field0-resource-amendment-v1','lumina-field0-reuse-amendment-v1',
                              'lumina-field0-reuse-amendment-v2','lumina-field0-checkpoint-amendment-v1') or
        a.get('original_plan_sha256')!=c['plan_sha256'] or
        reservation!=number or a.get('reserved_run_number')!=number or
        a.get('previous_reservations')!=previous or
        a.get('maximum_runs')!=number or plan.get('maximum_runs',3)!=3 or
        a.get('maximum_parent_reservation_seconds')!=(48600 if recovery else 37800 if reuse_retry else 27000 if continuation else 16200) or
        a.get('resources_per_run')!=approved or a.get('ray_buffer_GiB')!=24):
        raise ValueError('승인된 마지막 예약 자원 변경 범위 불일치')
    if continuation:
        reuse=a.get('approved_reuse_sources',{})
        if (set(reuse)!= {'cell_sources','boundary_sources'} or
            c.get('reuse_sources')!=reuse or c.get('share_source_sigma',False) is not False):
            raise ValueError('추가 예약은 두 완성 계수 재사용만 허용; 재생산 금지')
        old_path=Path(a['previous_execution'])
        if D.sha(old_path)!=a['previous_execution_sha256']:
            raise ValueError('이전 실행 기록 신원 불일치')
        old=json.loads(old_path.read_text())
        if (old.get('slurm_job_id')!='1200155' or old.get('status')!='FAILED_PRESERVED' or
            old.get('physical_field_executions')!=0 or old.get('plan_sha256')!=c['plan_sha256'] or
            not any(s.get('step')=='paired_sources' and s.get('state')=='COMPLETED'
                    for s in old.get('steps',[]))):
            raise ValueError('이전 완료 계수/미실행 수송 이력 불일치')
        for label,entry in reuse.items():
            if Path(entry['directory']).resolve()!=old_path.parent/label:
                raise ValueError('1200155의 계수만 재사용 가능')
        if reuse_retry:
            failed_path=Path(a['previous_failed_execution'])
            if D.sha(failed_path)!=a['previous_failed_execution_sha256']:
                raise ValueError('직전 실패 실행 기록 신원 불일치')
            failed=json.loads(failed_path.read_text())
            if (failed.get('slurm_job_id')!='1200286' or failed.get('reserved_run_number')!=4 or
                failed.get('status')!='FAILED_PRESERVED' or failed.get('physical_field_executions')!=0 or
                failed.get('plan_sha256')!=c['plan_sha256']):
                raise ValueError('직전 미실행 수송/예약 이력 불일치')
        if recovery:
            if (c.get('checkpoint') is not True or c.get('resume_checkpoint') is not None or
                a.get('required_checkpoint') is not True or
                c.get('library_sha256')!=a.get('approved_library_sha256')):
                raise ValueError('추가1회는 검증된 새 바이너리의 J 저장 경로만 허용')
            previous_path=Path(a['previous_transport_execution'])
            if D.sha(previous_path)!=a['previous_transport_execution_sha256']:
                raise ValueError('직전 수송 시간 초과 기록 신원 불일치')
            previous_transport=json.loads(previous_path.read_text())
            if (previous_transport.get('slurm_job_id')!='1200364' or
                previous_transport.get('reserved_run_number')!=5 or
                previous_transport.get('status')!='FAILED_PRESERVED' or
                previous_transport.get('physical_field_executions')!=1 or
                previous_transport.get('plan_sha256')!=c['plan_sha256'] or
                not any(s.get('step')=='field0' and s.get('state')=='FAILED'
                        for s in previous_transport.get('steps',[]))):
                raise ValueError('직전 예약5/실패 수송1회 이력 불일치')
    return approved,24,a


def execute(config_path,config_sha,out):
    config_path=Path(config_path).resolve();out=Path(out)
    if D.sha(config_path)!=config_sha:raise ValueError('봉인 실행 구성 SHA256 불일치')
    c=json.loads(config_path.read_text());plan_path=Path(c['plan'])
    if D.sha(plan_path)!=c['plan_sha256']:raise ValueError('봉인 BRIDGE 계획 SHA256 불일치')
    plan=json.loads(plan_path.read_text())
    reservation=int(c.get('reserved_run_number',1))
    resources,ray_buffer_gib,amendment=effective_resources(c,plan,reservation)
    maximum=amendment['maximum_runs'] if amendment else plan.get('maximum_runs',3)
    if not 1<=reservation<=maximum:raise ValueError('봉인 예약 횟수 상한 초과')
    if os.environ.get('OMP_NUM_THREADS')!='16':raise ValueError('실제 OMP16 환경 필요')
    if out.exists():raise ValueError('기존 실행 산출물 보존; 같은 root 재실행 금지')
    checkpoint=c.get('checkpoint',False);resume=c.get('resume_checkpoint')
    if type(checkpoint) is not bool or (resume is not None and not checkpoint):
        raise ValueError('재개는 명시 J 저장 경로에서만 허용')
    resume_dir=None
    if resume is not None:
        if not isinstance(resume,dict) or set(resume)!={'directory','metadata_sha256'}:
            raise ValueError('봉인 재개 위치/메타데이터 신원 필요')
        resume_dir=Path(resume['directory']).resolve()
        if D.sha(resume_dir/'checkpoint.json')!=resume['metadata_sha256']:
            raise ValueError('재개 메타데이터 SHA256 불일치')
    seed=Path(c['seed']);cells=Path(c['cells']);material_run=Path(plan['initial_material'])
    seed_report=json.loads((seed/'material-seed.json').read_text())
    if (seed_report['plan_sha256']!=c['plan_sha256'] or seed_report['source_record']!=70 or
        seed_report['record_storage_sha256']!=plan['initial_record_storage_sha256']):
        raise ValueError('원래70 입력 세대 불일치')
    grid=Path(c['grid']);grid_report=json.loads(grid.with_name('grid.json').read_text())
    if D.sha(grid)!=c['grid_sha256'] or grid_report['reference_field_sha256']!=plan['reference_field_sha256']:
        raise ValueError('원래 전체 참조장/주파수 격자 신원 불일치')
    material=D.read_material(cells/'material_cells.bin',b'LUMCEL01')
    native=D.read_material(seed/'native_material_nodes.bin',b'LUMNOD01')
    nu=np.load(grid,allow_pickle=False);nf=len(nu)
    def source_size(ns):return 16+8*(nf+2*ns+7*ns*nf)
    memory=D.memory_inventory_sizes(material,material['ns'],nf,grid_report['dlognu'],native,
        source_size(material['ns']),source_size(native['ns']),True,ray_buffer_gib)
    if memory['known_allocation_bound_bytes']>resources['memory_GiB']*1024**3:
        raise ValueError('전체 격자의 알려진 할당이 승인 메모리 초과; 해상도 축소/실행하지 않음')
    library=Path(c['library'])
    if D.sha(library)!=c['library_sha256']:raise ValueError('선택한 C 수송 library 신원 불일치')
    out.mkdir();started=time.monotonic();scripts=Path(__file__).resolve().parent
    result=dict(schema='lumina-bridge-field0-execution-v1',status='PREPARING',
        config_sha256=config_sha,plan_sha256=c['plan_sha256'],resources=resources,
        resource_amendment=amendment,resource_amendment_sha256=c.get('resource_amendment_sha256'),
        ray_buffer_GiB=ray_buffer_gib,
        reserved_run_number=reservation,slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        frequency_grid=grid_report,memory=memory,physical_field_executions=0,
        matter_updates=0,native_executions=0,automatic_retry=0,steps=[],final_goal='NOT_JUDGED')
    if checkpoint:
        result['checkpoint']=dict(directory=str((out/'field0/checkpoint').resolve()),
            resume_origin=resume,role='unverified J only; not a completed physical field')
    def save():
        result['elapsed_seconds']=time.monotonic()-started
        if checkpoint:
            metadata=out/'field0/checkpoint/checkpoint.json'
            if metadata.is_file():
                result['checkpoint']['metadata_sha256']=D.sha(metadata)
        (out/'execution.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    def run(label,command):
        remaining=resources['child_seconds']-(time.monotonic()-started)
        if remaining<=0:raise TimeoutError('봉인 전체 자식 예산 소진')
        if label=='field0':
            result['physical_field_executions']=1;result['status']='FIELD0_STARTED'
        result['steps'].append(dict(step=label,state='RUNNING',command=list(map(str,command))))
        save()
        with (out/(label+'.log')).open('x') as log:
            subprocess.run(list(map(str,command)),check=True,stdout=log,stderr=subprocess.STDOUT,
                timeout=remaining,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
        result['steps'][-1]['state']='COMPLETED';save()
    save()
    try:
        D.identity(seed_report,material_run/'RVTJ')
        flags,rel,velocity,reasons=D.frame_contract(material_run/'MODEL',native,material['epoch'])
        result['early_frame_check']=dict(flags=flags,relativistic=rel,velocity=velocity,reasons=reasons)
        save()
        if not rel or reasons:raise ValueError('계수 생산 전 full-SR 확인 실패: '+str(reasons))
        reuse=c.get('reuse_sources',{})
        if not isinstance(reuse,dict) or set(reuse)-{'cell_sources','boundary_sources'}:
            raise ValueError('알 수 없는 계수 재사용 지정')
        source_dirs={}
        paired=c.get('share_source_sigma',False)
        if not isinstance(paired,bool) or (paired and reuse):
            raise ValueError('σ 공유와 완료 계수 재사용은 실행 구성에서 구분해야 함')
        if paired:
            source_dirs={label:out/label for label in ('cell_sources','boundary_sources')}
            run('paired_sources',[sys.executable,scripts/'cmfgen_model_sources.py','--run',material_run,
                '--material',material['path'],'--atomic',cells/'atomic','--phot-library',c['phot_library'],
                '--kernel-library',c['kernel_library'],'--C-full-population',c['full_population'],
                '--frequency-grid',grid,'--out',source_dirs['cell_sources'],
                '--peer-material',native['path'],'--peer-C-full-population',c['native_full_population'],
                '--peer-out',source_dirs['boundary_sources']])
        for label,mat,full in (('cell_sources',material['path'],c['full_population']),
                               ('boundary_sources',native['path'],c['native_full_population'])):
            if label in source_dirs:continue
            if label in reuse:
                entry=reuse[label];folder=Path(entry['directory']).resolve()
                report_path=folder/'sources.json'
                if D.sha(report_path)!=entry['report_sha256']:
                    raise ValueError('재사용 완료 계수 보고서 신원 불일치')
                source_report=json.loads(report_path.read_text())
                # 부분 출력은 sources.json/완료 binary hash를 갖추지 못한다.
                # 실제 binary·T/ne·격자는 아래 공통 prepare에서 재검증한다.
                for path in source_report['input_sha256']:
                    D.identity(source_report,path)
                for path in (mat,material_run/'MODEL',full,c['phot_library'],
                             c['kernel_library'],cells/'atomic/levels.csv'):
                    D.identity(source_report,path)
                source_dirs[label]=folder
                result['steps'].append(dict(step=label,state='REUSED_COMPLETED',
                    directory=str(folder),report_sha256=entry['report_sha256']))
                save()
                continue
            source_dirs[label]=out/label
            run(label,[sys.executable,scripts/'cmfgen_model_sources.py','--run',material_run,
                '--material',mat,'--atomic',cells/'atomic','--phot-library',c['phot_library'],
                '--kernel-library',c['kernel_library'],'--C-full-population',full,
                '--frequency-grid',grid,'--out',out/label])
        prepared=D.prepare(cells,source_dirs['cell_sources'],seed,
                           source_dirs['boundary_sources'],material_run/'MODEL',
                           resources['memory_GiB'],ray_buffer_gib)
        if not np.array_equal(prepared['sources']['nu'],nu):
            raise ValueError('실제 생산/재사용 계수가 봉인 전체 주파수 격자와 다름')
        if not prepared['report']['execution_eligible'] or prepared['report']['implemented_frame']!=D.REL_FRAME:
            raise ValueError('실제 full-SR 입력/메모리 사전 확인 실패: '+str(prepared['report']['reasons']))
        result['preflight']=prepared['report'];save()
        del prepared  # 검증용 매핑/임시 상태는 자식 수송 시작 전에 해제한다.
        command=[sys.executable,scripts/'run_lumina_cell_candidate.py','--cells',cells,
            '--sources',source_dirs['cell_sources'],'--seed',seed,
            '--boundary-sources',source_dirs['boundary_sources'],
            '--model',material_run/'MODEL','--library',library,
            '--memory-GiB',resources['memory_GiB'],'--ray-buffer-GiB',ray_buffer_gib,
            '--execute-candidate','--out',out/'field0']
        if checkpoint:command+=['--checkpoint']
        if resume_dir is not None:command+=['--resume-checkpoint',resume_dir]
        run('field0',command)
        field=json.loads((out/'field0/preflight.json').read_text())
        if field['status']!='CANDIDATE_COMPLETED':raise ValueError('실제 field0 완료 증거 없음')
        result['status']='FIELD0_COMPLETED';result['field0']=field
        result['next']='같은 물질70의 J71과 공간/주파수 함수를 맞춰 비교; 미설명 차이면 물질1 중단'
        save();return 0
    except (ValueError,OSError,KeyError,TypeError,subprocess.SubprocessError,TimeoutError) as exc:
        result['status']='FAILED_PRESERVED';result['failure']=str(exc)
        if result['steps'] and result['steps'][-1]['state']=='RUNNING':result['steps'][-1]['state']='FAILED'
        save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--config-sha256',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();raise SystemExit(execute(a.config,a.config_sha256,a.out))
