#!/usr/bin/env python3
"""원 물질70·직접 J의 Si/S 준위용해 SE율을 기존 bound율과 분리해 생산한다."""
import argparse
import csv
import json
import os
from pathlib import Path
import resource
import socket
import time

import numpy as np

import cmfgen_model_continuum as B
import lumina_native_columns as N
import lumina_native90_rates as R
from lumina_model_point_rates import ModelPointBoundRates
from lumina_model_point_dissolution import contract_dissolution


def run(args):
    started=time.monotonic();out=Path(args.out)
    if out.exists():raise FileExistsError('기존 준위용해 결과 보존')
    host=socket.gethostname().lower()
    grammar_slurm=(host.startswith('grammar') and os.environ.get('SLURM_JOB_ID') and
                   os.environ.get('SLURM_CPUS_PER_TASK')=='1' and
                   os.environ.get('SLURM_MEM_PER_NODE')=='4096')
    if (host!='lageunha' and not grammar_slurm) or (
        os.environ.get('OMP_NUM_THREADS')!='1' or
        os.environ.get('OPENBLAS_NUM_THREADS')!='1'):
        raise ValueError('Lageunha CPU1 수동 또는 grammar CPU1·4GiB Slurm 실행 필요')
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
    source_files=[Path(__file__),Path(B.__file__),Path(N.__file__),Path(R.__file__),
                  Path(__file__).with_name('lumina_model_point_rates.py'),
                  Path(__file__).with_name('lumina_model_point_dissolution.py'),Path(args.library)]
    source_sha={str(p):B.sha(p) for p in source_files}
    projection=Path(args.field_projection);seal=json.loads(projection.read_text())
    if (seal.get('schema')!='lumina-cmfgen-direct-native90-j-v1' or
        seal.get('status')!='FIELD_PROJECTED_NOT_SELF_CONSISTENT' or
        seal.get('material_record')!=70 or seal.get('native_shells')!=90 or
        seal.get('native_frequencies')!=1115020 or seal.get('extrapolated_frequency_count')!=0):
        raise ValueError('다른 세대의 직접 복사장')
    j_path=Path(seal['J_path'])
    if B.sha(j_path)!=seal['J_sha256']:raise ValueError('직접 J byte 신원 변경')
    bound=Path(args.full_bound_result);bound_result=json.loads(bound.read_text())
    attestation=json.loads(bound.with_name('radiation-input.json').read_text())
    if (bound_result.get('state')!='COMPLETE_BOUND_ONLY' or
        bound_result.get('routes_complete')!=24667 or
        bound_result.get('routes_expected')!=24667 or
        bound_result.get('complete_bound_frequency_integral') is not True or
        bound_result.get('source_unchanged') is not True or
        bound_result.get('pseudo_included') is not False or
        bound_result.get('native_node')!=args.node or
        bound_result.get('frequencies')!=1115020 or
        attestation.get('status')!='COMPLETE_INPUT_SEALED' or
        attestation.get('bound_result_sha256')!=B.sha(bound) or
        attestation.get('J_sha256_before')!=seal['J_sha256'] or
        attestation.get('J_sha256_after')!=seal['J_sha256'] or
        attestation.get('projection_sha256')!=B.sha(projection) or
        attestation.get('library_sha256')!=B.sha(Path(args.library)) or
        bound_result.get('source_sha256',{}).get(str(Path(B.__file__)))!=B.sha(Path(B.__file__))):
        raise ValueError('완료 bound/J/바이너리 입력의 다른 원래점 또는 신원')
    base_path=Path(args.base)
    if B.sha(base_path)!='f3b01d29c798e910fe42427283f1a9697189da45926d0bc4b77961e991909bab':
        raise ValueError('원 native 계획 불일치')
    base=json.loads(base_path.read_text());c=json.loads(Path(base['config']).read_text())
    plan=json.loads(Path(c['plan']).read_text())
    report_path=Path(c['reuse_sources']['boundary_sources']['directory'])/'sources.json'
    if B.sha(report_path)!=c['reuse_sources']['boundary_sources']['report_sha256']:
        raise ValueError('계수 신원 변경')
    report=json.loads(report_path.read_text())
    with N.open_sealed_native(base['config'],base['preparation'],base['preparation_sha256']) as source:
        if source.identity!=seal['source_identity'] or source.identity!=bound_result['input_identity'] or (
            source.ns,source.nf)!=(90,1115020):
            raise ValueError('복사장/율/물질 원래점 세대 불일치')
        rows,models,routes,state,do_dis,_,_=R.material_state(c,plan,source,report)
        if not do_dis:raise ValueError('원 물질에서 용해 과정 비활성')
        point=ModelPointBoundRates(routes,rows,models,state,args.node,Path(args.library))
        widths=np.zeros(source.nf,dtype='f8');delta=np.diff(source.frequency)
        widths[:-1]+=.5*delta;widths[1:]+=.5*delta
        with N.FrozenNativeField(j_path,source,source.radius,source.frequency) as field:
            j=np.concatenate([field.read(np.array([args.node]),np.arange(lo,min(source.nf,lo+65536),dtype='i8'))[0]
                              for lo in range(0,source.nf,65536)])
        unit=contract_dissolution(point,source.frequency,widths,j,dissolution_enabled=True)
    physical=point.physical_ion_rates(unit)
    reference_path=Path(args.reference_rates)
    if B.sha(reference_path)!='db6161c3cb2919ce264c276469093b73a19d76e2d76fd73310cc7287ac9af711':
        raise ValueError('독립 직접율 표의 신원 변경')
    with reference_path.open(newline='') as stream:
        reference={(int(r['native_node']),int(r['Z']),int(r['stage'])):r for r in csv.DictReader(stream)}
    if len(reference)!=360:raise ValueError('전90점×4이온 독립 적분표 누락')
    comparison=[];maximum=0.
    for Z,stage in ((14,2),(14,3),(16,2),(16,3)):
        measured=physical[Z,stage]
        row=reference[args.node,Z,stage]
        expected=np.array([float(row[f'CMFGEN_DIRECT_total_{name}_cm3_s'])-
                           float(row[f'CMFGEN_DIRECT_bound_{name}_cm3_s'])
                           for name in ('photo','spontaneous','stimulated')])
        relative=np.abs(measured-expected)/np.maximum(np.abs(expected),1.e-30)
        maximum=max(maximum,float(relative.max()))
        comparison.append(dict(Z=Z,stage=stage,C=measured.tolist(),independent=expected.tolist(),
                               relative=relative.tolist()))
    unchanged=(all(B.sha(Path(p))==digest for p,digest in source_sha.items()) and
               B.sha(j_path)==seal['J_sha256'])
    out.mkdir(parents=True)
    np.save(out/'pseudo-unit-rates.npy',unit,allow_pickle=False)
    result=dict(schema='lumina-model-point-native-dissolution-v1',
        state='COMPLETE_PRIVATE_RATES' if unchanged and maximum<=1.e-6 else 'MISMATCH',
        native_node=args.node,material_record=70,routes=len(point.routes),
        nonzero_routes=int(np.count_nonzero(np.any(unit[:,:3]!=0,axis=1))),
        radiation_J_sha256=seal['J_sha256'],source_unchanged=unchanged,
        independent_rate_max_relative=maximum,independent_rate_comparison=comparison,
        artifact_sha256=B.sha(out/'pseudo-unit-rates.npy'),
        input_sha256={str(p):B.sha(p) for p in (projection,bound,Path(args.reference_rates),base_path)},
        source_sha256=source_sha,wall_seconds=time.monotonic()-started,
        material_updates=0,SE_charge_RE_solved=False,new_transport=0,final_goal='NOT_MET')
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('state','native_node','nonzero_routes',
        'independent_rate_max_relative','wall_seconds')}),flush=True)
    return 0 if result['state']=='COMPLETE_PRIVATE_RATES' else 2


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base','field-projection','full-bound-result','library','reference-rates','out'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--node',type=int,default=49)
    raise SystemExit(run(p.parse_args()))
