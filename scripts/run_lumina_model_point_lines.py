#!/usr/bin/env python3
"""동결 자체장/native49의 전모형 BB Jbar와 Einstein 복사율만 생산한다.

같은 source의 raw OSC/GF 선택/고정 Doppler profile을 재사용한다.
전체 SE/RE·충돌율·물질 갱신·새 수송의 완료 판정은 하지 않는다.
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

import cmfgen_model_continuum as B
import lumina_native_columns as N
import lumina_native90_rates as R
from lumina_model_point_lines import ModelPointLines


def run(args):
    start=time.monotonic();out=Path(args.out)
    if out.exists():raise FileExistsError('기존 점값 BB 결과 보존')
    if os.environ.get('SLURM_CPUS_PER_TASK')!='1' or os.environ.get('SLURM_MEM_PER_NODE')!='4096':
        raise ValueError('CPU1/4GiB Slurm 직접시험 필요')
    out.mkdir()
    files=[Path(__file__),Path(__file__).with_name('lumina_model_point_lines.py'),
        Path(__file__).with_name('cmfgen_model_sources.py'),Path(B.__file__),Path(N.__file__),Path(R.__file__),Path(args.library)]
    identities={str(p):B.sha(p) for p in files}
    (out/'source-before.json').write_text(json.dumps(identities,indent=2)+'\n')
    base_path=Path(args.base)
    if B.sha(base_path)!='f3b01d29c798e910fe42427283f1a9697189da45926d0bc4b77961e991909bab':
        raise ValueError('원 native 계획 불일치')
    base=json.loads(base_path.read_text());c=json.loads(Path(base['config']).read_text())
    plan=json.loads(Path(c['plan']).read_text())
    report_path=Path(c['reuse_sources']['boundary_sources']['directory'])/'sources.json'
    if B.sha(report_path)!=c['reuse_sources']['boundary_sources']['report_sha256']:
        raise ValueError('원 계수 신원 변경')
    with N.open_sealed_native(base['config'],base['preparation'],base['preparation_sha256']) as source:
        rows,models,routes,state,do_dis,full_check,smoothing=R.material_state(c,plan,source,json.loads(report_path.read_text()))
        nu=np.array(source.frequency,copy=True);node=49
        with N.FrozenNativeField(args.j,source,source.radius,source.frequency) as field:
            j=np.concatenate([field.read(np.array([node]),np.arange(lo,min(len(nu),lo+65536),dtype='i8'))[0]
                              for lo in range(0,len(nu),65536)])
        identity=dict(source.identity);radius=float(source.radius[node])
    widths=np.zeros_like(nu);delta=np.diff(nu)
    widths[:-1]+=.5*delta;widths[1:]+=.5*delta
    producer=ModelPointLines(Path(plan['initial_material']),rows,models,Path(args.library).resolve())
    values=producer.contract(nu,widths,j);status=values['status'];valid=(status==1)|(status==2)
    np.savez(out/'point-lines.npz',**producer.lines,**values)
    unchanged=all(B.sha(Path(p))==sha for p,sha in identities.items())
    complete=bool(valid.all() and unchanged)
    result=dict(state='COMPLETE_RADIATIVE_BB_ONLY' if complete else 'INCOMPLETE_FIELD_COVERAGE',
        native_node=node,radius_cm=radius,frequency_points=len(nu),lines=len(status),
        status_counts={name:int(np.count_nonzero(status==code)) for code,name in
                       ((1,'VALID'),(2,'EXACT_ZERO'),(3,'OUT_OF_GRID'),(4,'UNSAMPLED'))},
        profile=producer.profile,line_domain=producer.domain,
        radiation_provenance='sealed J_A from old source without final native IR roster; not regenerated',
        profile_mass_range_valid=[float(values['profile_mass'][valid].min()),
            float(values['profile_mass'][valid].max())] if np.any(valid) else None,
        profile_mass_normalization_applied=False,spontaneous_A_profile_mass_scaled=False,
        raw_OSC_source_sha256=producer.identities,raw_selection_counts=producer.counts,
        source_sha256=identities,source_unchanged=unchanged,input_identity=identity,
        full_population_check=full_check,artifact_sha256=B.sha(out/'point-lines.npz'),
        collisional_BB_included=False,SE_RE_solution=False,material_updates=0,
        new_transport=0,wall_seconds=time.monotonic()-start,final_goal='NOT_MET')
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('state','lines','status_counts','profile_mass_range_valid','wall_seconds')}),flush=True)
    return 0 if complete else 2


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--j',required=True)
    p.add_argument('--library',required=True);p.add_argument('--out',required=True)
    raise SystemExit(run(p.parse_args()))
