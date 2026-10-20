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


def execute(config_path,config_sha,out):
    config_path=Path(config_path).resolve();out=Path(out)
    if D.sha(config_path)!=config_sha:raise ValueError('봉인 실행 구성 SHA256 불일치')
    c=json.loads(config_path.read_text());plan_path=Path(c['plan'])
    if D.sha(plan_path)!=c['plan_sha256']:raise ValueError('봉인 BRIDGE 계획 SHA256 불일치')
    plan=json.loads(plan_path.read_text());resources=plan['resources_per_run']
    reservation=int(c.get('reserved_run_number',1))
    if not 1<=reservation<=plan.get('maximum_runs',3):raise ValueError('봉인 예약 횟수 상한 초과')
    if resources!=dict(CPU=16,OMP=16,memory_GiB=32,GPU=0,parent_seconds=2700,
                       child_seconds=2400,node_pinned=False,retry=0):
        raise ValueError('기존 자원/재시도 계약 불일치')
    if os.environ.get('OMP_NUM_THREADS')!='16':raise ValueError('실제 OMP16 환경 필요')
    if out.exists():raise ValueError('기존 실행 산출물 보존; 같은 root 재실행 금지')
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
        source_size(material['ns']),source_size(native['ns']),True)
    if memory['known_allocation_bound_bytes']>32*1024**3:
        raise ValueError('전체 격자의 알려진 할당이32GiB 초과; 해상도 축소/실행하지 않음')
    library=Path(c['library'])
    if D.sha(library)!=c['library_sha256']:raise ValueError('선택한 C 수송 library 신원 불일치')
    out.mkdir();started=time.monotonic();scripts=Path(__file__).resolve().parent
    result=dict(schema='lumina-bridge-field0-execution-v1',status='PREPARING',
        config_sha256=config_sha,plan_sha256=c['plan_sha256'],resources=resources,
        reserved_run_number=reservation,slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        frequency_grid=grid_report,memory=memory,physical_field_executions=0,
        matter_updates=0,native_executions=0,automatic_retry=0,steps=[],final_goal='NOT_JUDGED')
    def save():
        result['elapsed_seconds']=time.monotonic()-started
        (out/'execution.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    def run(label,command):
        remaining=2400-(time.monotonic()-started)
        if remaining<=0:raise TimeoutError('봉인 자식2400초 예산 소진')
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
        for label,mat,full in (('cell_sources',material['path'],c['full_population']),
                               ('boundary_sources',native['path'],c['native_full_population'])):
            run(label,[sys.executable,scripts/'cmfgen_model_sources.py','--run',material_run,
                '--material',mat,'--atomic',cells/'atomic','--phot-library',c['phot_library'],
                '--kernel-library',c['kernel_library'],'--C-full-population',full,
                '--frequency-grid',grid,'--out',out/label])
        prepared=D.prepare(cells,out/'cell_sources',seed,out/'boundary_sources',material_run/'MODEL')
        if not prepared['report']['execution_eligible'] or prepared['report']['implemented_frame']!=D.REL_FRAME:
            raise ValueError('실제 full-SR 입력/메모리 사전 확인 실패: '+str(prepared['report']['reasons']))
        result['preflight']=prepared['report'];save()
        run('field0',[sys.executable,scripts/'run_lumina_cell_candidate.py','--cells',cells,
            '--sources',out/'cell_sources','--seed',seed,'--boundary-sources',out/'boundary_sources',
            '--model',material_run/'MODEL','--library',library,'--execute-candidate','--out',out/'field0'])
        field=json.loads((out/'field0/preflight.json').read_text())
        if field['status']!='CANDIDATE_COMPLETED':raise ValueError('실제 field0 완료 증거 없음')
        result['status']='FIELD0_COMPLETED';result['field0']=field
        result['next']='같은 물질70의 J71과 공간/주파수 함수를 맞춰 비교; 미설명 차이면 물질1 중단'
        save();return 0
    except (ValueError,OSError,subprocess.SubprocessError,TimeoutError) as exc:
        result['status']='FAILED_PRESERVED';result['failure']=str(exc)
        if result['steps'] and result['steps'][-1]['state']=='RUNNING':result['steps'][-1]['state']='FAILED'
        save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--config-sha256',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();raise SystemExit(execute(a.config,a.config_sha256,a.out))
