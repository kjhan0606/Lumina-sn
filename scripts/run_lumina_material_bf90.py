#!/usr/bin/env python3
"""원70/J70의 전24667 PHOT 경로·90점 BF를 배치 생산한다. SE/RE/RT 해가 아니다."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np

import cmfgen_model_continuum as B
import lumina_native_columns as N
import lumina_native90_rates as R
from lumina_model_material_rates import ModelMaterialRates
from compare_native_prrr_common_grid import continuum_mapping
from run_cmfgen_formal_primary import fields


def run(args):
    if (os.environ.get('SLURM_CPUS_PER_TASK')!='16' or os.environ.get('SLURM_MEM_PER_NODE')!='32768' or
        os.environ.get('OMP_NUM_THREADS')!='16' or not re.fullmatch('[0-9a-f]{40}',args.source_commit)):
        raise ValueError('CPU16·32GiB·OpenMP16 및 봉인 소스 커밋 필요')
    started=time.monotonic();deadline=started+6600;out=args.out
    if out.exists():raise FileExistsError('기존 전90점 BF 보존')
    out.mkdir()
    project=Path(__file__).resolve().parents[1]
    base_path=Path('/gpfs/kjhan/cmfgen_runs/lumina_extended_4h_20261022-rJGMVR/gpu-local-plan.json')
    base=json.loads(base_path.read_text());config=Path(base['config']);c=json.loads(config.read_text())
    plan_path=Path(c['plan']);plan=json.loads(plan_path.read_text())
    source_report=Path(c['reuse_sources']['boundary_sources']['directory'])/'sources.json'
    if B.sha(source_report)!=c['reuse_sources']['boundary_sources']['report_sha256']:
        raise ValueError('원 계수 입력 보고서 변경')
    report=json.loads(source_report.read_text())
    direct_test=Path('/gpfs/kjhan/cmfgen_runs/lumina_material_batch_v2_20261027-cHHYRA/run/result.json')
    test=json.loads(direct_test.read_text())
    library=direct_test.parent.parent/'build/point.so'
    if (B.sha(direct_test)!='fbc5c5221b51a8895739fae2868d23b2242fb63faca8aec83a036a234fbb8e72' or
        B.sha(library)!=test['input_sha256'][str(library)] or test['source_identity']!=base['source_identity']):
        raise ValueError('전90점 직접시험 바이너리/물질 계보 변경')
    run_path=Path(plan['initial_material'])
    field_path=Path(plan['same_material_reference_field'])/'FORMAL_PRIMARY_FIELDS.bin'
    native=Path('/gpfs/kjhan/cmfgen_runs/lumina_fixedj_exp70_20261026-phmMnx/run/material')
    code=[project/name for name in ('scripts/run_lumina_material_bf90.py','scripts/lumina_model_material_rates.py',
        'scripts/lumina_model_point_rates.py','scripts/cmfgen_model_continuum.py',
        'scripts/expand_atomic_data_cmfgen.py','scripts/cmfgen_parser.py','scripts/cmfgen_phot_smoothing.py',
        'scripts/lumina_native90_rates.py','scripts/lumina_native_columns.py',
        'scripts/compare_native_prrr_common_grid.py','src/bf_rate_jnu.c','src/bf_rate_jnu.h')]
    code_sha={str(p):B.sha(p) for p in code}
    for p,h in code_sha.items():
        blob=subprocess.run(['git','show',args.source_commit+':'+str(Path(p).relative_to(project))],
                            cwd=project,capture_output=True,check=True).stdout
        if hashlib.sha256(blob).hexdigest()!=h:raise ValueError('커밋되지 않은 실행 소스')
    for name in ('src/bf_rate_jnu.c','src/bf_rate_jnu.h','scripts/lumina_model_material_rates.py'):
        p=str(project/name)
        if code_sha[p]!=test['input_sha256'][p]:
            raise ValueError('대표90점 검증과 다른 kernel/소비자')
    data=[base_path,config,plan_path,source_report,direct_test,library,field_path,native/'CONT_FREQ',
          Path(c['seed'])/'native_material_nodes.bin',Path(c['native_full_population']),
          Path(c['cells'])/'atomic/levels.csv',Path(c['phot_library']),
          *[run_path/name for name in ('MODEL','MODEL_SPEC','atomic_links.txt','LEVEL_SL_STEQ_LINKS')]]
    for item in report['process_selection']['ions']:
        for phot in item['phot_sources']:
            data.append(Path(phot['path']))
    links=B.atomic.load_cmfgen_links(run_path/'atomic_links.txt')
    data.extend(Path(link[kind]) for link in links.values() for kind in ('osc','f_to_s'))
    data_sha={str(p):B.sha(p) for p in data}
    if (data_sha[str(base_path)]!='f3b01d29c798e910fe42427283f1a9697189da45926d0bc4b77961e991909bab' or
        data_sha[str(field_path)]!=plan['reference_field_sha256']):
        raise ValueError('같은 원 물질70/원 J70 아님')
    (out/'inputs.json').write_text(json.dumps(dict(scope=__doc__,source_commit=args.source_commit,
        code_sha256=code_sha,data_sha256=data_sha,resources='CPU16/32GiB/2h/GPU0/retry0',
        internal_seconds=6600,material_updates=0,new_native=0,new_transport=0),indent=2)+'\n')
    geometry,field=fields(field_path)
    try:
        with N.open_sealed_native(base['config'],base['preparation'],base['preparation_sha256']) as source:
            rows,models,routes,state,do_dis,full_check,smoothing=R.material_state(c,plan,source,report)
            if source.identity!=base['source_identity'] or not np.array_equal(geometry[0]*1.e10,source.radius):
                raise ValueError('전90점 물질/장 반경 불일치')
        nu=np.array(field['freq'][::-1]*1.e15);j=np.array(field['field'][::-1,0,:].T,order='C')
        evaluation=continuum_mapping(native/'CONT_FREQ',nu)
        width=np.zeros_like(nu);width[:-1]+=.5*np.diff(nu);width[1:]+=.5*np.diff(nu)
        provider=ModelMaterialRates(routes,rows,models,library,nu,width,j,evaluation=evaluation,
            radiation_generation=70,dissolution_enabled=do_dis)
        if (len(provider.routes),len(models),provider.n_points,len(nu))!=(24667,27,90,197291):
            raise ValueError('전모형 BF 경로/이온/원90점/주파수 누락')
        trial=provider.prepare(state,material_generation=1);nr=len(provider.routes)
        np.savez(out/'material-snapshot.npz',temperature=trial.state['temperature'],ne=trial.state['ne'],
            columns=trial.state['columns'],superlevel=trial.state['superlevel'],occupation=trial.state['occupation'],
            fraction=trial.fraction,saha=trial.saha,milne_prefactor=trial.milne_prefactor)
        np.savez(out/'route-roster.npz',lower=[r.lower for r in provider.routes],target=[r.target for r in provider.routes],
            Z=[r.Z for r in provider.routes],stage=[r.stage for r in provider.routes],phot_grid=[r.phot_grid for r in provider.routes],
            probability=[r.probability for r in provider.routes],threshold=provider.edge,sigma_edge=provider.sigma_edge)
        shape=(nr,90,6)
        outputs=[np.lib.format.open_memmap(out/name,mode='w+',dtype='f8',shape=shape)
                 for name in ('bound-rates.npy','total-rates.npy')]
        done=np.lib.format.open_memmap(out/'routes-complete.npy',mode='w+',dtype='bool',shape=(nr,))
        for a in outputs:a[:]=np.nan
        done[:]=False;done.flush()
        calculation_start=time.monotonic()
        for first in range(0,nr,128):
            if time.monotonic()>deadline:
                print(json.dumps(dict(state='INCOMPLETE_TIME_LIMIT',routes_complete=int(done.sum()))),flush=True)
                return 3
            last=min(nr,first+128)
            values=provider.integrate(trial,route_indices=np.arange(first,last),material_generation=1,
                                      radiation_generation=70,route_block=128)
            for output,value in zip(outputs,values):output[first:last]=value;output.flush()
            done[first:last]=True;done.flush()
            if last%512==0 or last==nr:
                print(json.dumps(dict(phase='BF90',routes_complete=last,routes_total=nr,
                    seconds=time.monotonic()-calculation_start)),flush=True)
        if not done.all() or any(not np.isfinite(a).all() or np.any(a[:,:,:3]<0) for a in outputs):
            raise ValueError('완성 전90점 BF 수치/완료 mask 오류')
        if any(B.sha(Path(p))!=h for p,h in data_sha.items()):
            raise ValueError('BF90 실행 중 실제 입력 자료/바이너리 변경')
        changed_code=[p for p,h in code_sha.items() if B.sha(Path(p))!=h]
        # 실행 소스는 시작 때 고정 커밋과 일치했고 모든 모듈/바이너리를 이미 로드했다.
        # 다른 파일의 후속 편집은 이 결과의 버전과 혼합하지 않는다.
        result=dict(schema='lumina-frozen-material-all-routes-BF90-v1',state='COMPLETE_FROZEN_MATERIAL_BF_90',
            input_material=70,material_points=90,model_ions=27,model_levels=len(rows),routes_complete=nr,
            frequency_points=len(nu),field_signature=provider.field_signature,material_signature=trial.material_signature,
            source_identity=base['source_identity'],source_commit=args.source_commit,code_sha256=code_sha,
            current_code_changed_after_loading=changed_code,data_sha256=data_sha,
            output_sha256={name:B.sha(out/name) for name in ('bound-rates.npy','total-rates.npy','routes-complete.npy',
                'material-snapshot.npz','route-roster.npz')},
            BF_seconds=time.monotonic()-calculation_start,wall_seconds=time.monotonic()-started,
            data_unchanged=True,new_transport=0,new_native=0,public_material_updates=0,
            SE_charge_RE_solved=False,final_goal='NOT_MET')
        (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(state=result['state'],wall_seconds=result['wall_seconds'])),flush=True)
    finally:
        field._mmap.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--out',type=Path,required=True)
    raise SystemExit(run(parser.parse_args()))
