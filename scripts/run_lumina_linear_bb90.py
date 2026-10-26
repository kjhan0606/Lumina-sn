#!/usr/bin/env python3
"""원 J70의 전모형 선형 ν BB 적분. 공통 물질응답용이며 새 수송/SE 해가 아니다."""
import argparse
import csv
import json
import os
from pathlib import Path
import time

import numpy as np

import cmfgen_model_continuum as B
import cmfgen_model_sources as S
from lumina_model_linear_lines import LinearModelLines
from run_cmfgen_formal_primary import fields


def run(args):
    if os.environ.get('SLURM_CPUS_PER_TASK')!='1' or os.environ.get('SLURM_MEM_PER_NODE')!='4096':
        raise ValueError('CPU1/4GiB Slurm 시험 전용')
    start=time.monotonic();out=args.out.resolve()
    if out.exists():raise FileExistsError('기존 BB 결과 보존')
    out.mkdir()
    base_path=Path('/gpfs/kjhan/cmfgen_runs/lumina_extended_4h_20261022-rJGMVR/gpu-local-plan.json')
    if B.sha(base_path)!='f3b01d29c798e910fe42427283f1a9697189da45926d0bc4b77961e991909bab':
        raise ValueError('원 입력 계획 변경')
    base=json.loads(base_path.read_text());config_path=Path(base['config'])
    config=json.loads(config_path.read_text());plan_path=Path(config['plan']);plan=json.loads(plan_path.read_text())
    material=Path(plan['initial_material']);level_path=Path(config['cells'])/'atomic/levels.csv'
    field_path=Path(plan['same_material_reference_field'])/'FORMAL_PRIMARY_FIELDS.bin'
    inputs={str(p):B.sha(p) for p in (base_path,config_path,plan_path,level_path,field_path,
        args.library,*[material/name for name in ('MODEL','MODEL_SPEC','atomic_links.txt','RVTJ')])}
    if inputs[str(field_path)]!=plan['reference_field_sha256']:raise ValueError('다른 복사장')
    rows=list(csv.DictReader(level_path.open()))
    models=B.atomic.load_cmfgen_model_spec(material/'MODEL_SPEC',material/'atomic_links.txt')
    lines,counts,identities=S.load_lines(material,rows,models)
    inputs.update(identities);profile=S.fixed_profile(material/'MODEL')
    geometry,field=fields(field_path)
    try:
        nu=np.array(field['freq'][::-1]*1.e15)
        provider=LinearModelLines(lines,profile,args.library,nu,field['field'][::-1,0,:],radiation_generation=70)
        nr=len(lines['nu0']);npnt=provider.J.shape[1]
        if (nr,npnt,len(nu),len(models),len(rows))!=(790493,90,197291,27,24542):
            raise ValueError('전모형 선/90점/원주파수 누락')
        source=Path(__file__).resolve().parents[1]
        code={str(source/name):B.sha(source/name) for name in (
            'scripts/run_lumina_linear_bb90.py','scripts/lumina_model_linear_lines.py',
            'scripts/cmfgen_model_sources.py','scripts/cmfgen_model_continuum.py',
            'scripts/run_cmfgen_formal_primary.py','src/model_line_linear_jbar.c')}
        (out/'inputs.json').write_text(json.dumps(dict(source_commit=args.source_commit,input_sha256=inputs,
            code_sha256=code,field_representation='piecewise linear in frequency',profile=profile,
            signature=provider.signature,resources='CPU1/4GiB/10min/GPU0/retry0',new_transport=0,
            new_native=0,material_updates=0),indent=2)+'\n')
        np.savez(out/'line-roster.npz',**lines)
        jbar=np.lib.format.open_memmap(out/'jbar.npy',mode='w+',dtype='f8',shape=(nr,npnt))
        mass=np.lib.format.open_memmap(out/'profile-mass.npy',mode='w+',dtype='f8',shape=(nr,))
        status=np.lib.format.open_memmap(out/'status.npy',mode='w+',dtype='u1',shape=(nr,))
        done=np.lib.format.open_memmap(out/'lines-complete.npy',mode='w+',dtype='bool',shape=(nr,))
        jbar[:]=np.nan;mass[:]=np.nan;status[:]=0;done[:]=False;done.flush()
        for first in range(0,nr,4096):
            if time.monotonic()-start>480:
                print(json.dumps(dict(state='INCOMPLETE_TIME_LIMIT',complete=int(done.sum()))),flush=True)
                return 3
            last=min(nr,first+4096);v=provider.contract(first,last,radiation_generation=70)
            rates=provider.rates(first,last,v);valid=(v['status']==1)|(v['status']==2)
            if any(not np.isfinite(rates[name][valid]).all() or np.any(rates[name][valid]<0)
                   for name in ('absorption','stimulated','spontaneous')):
                raise ValueError('BB Einstein 율 비유한/음수')
            jbar[first:last]=v['jbar'];mass[first:last]=v['profile_mass'];status[first:last]=v['status']
            for a in (jbar,mass,status):a.flush()
            done[first:last]=True;done.flush()
            if last%65536==0 or last==nr:
                print(json.dumps(dict(lines_complete=last,lines_total=nr,seconds=time.monotonic()-start)),flush=True)
        missing=np.flatnonzero(status==3)
        specials=[]
        for charge,lo,up in ((1,199,203),(2,138,139)):
            addresses={int(row['level_number'])+1:i for i,row in enumerate(rows)
                       if int(row['atomic_number'])==16 and int(row['ion_number'])==charge}
            indices=np.flatnonzero((lines['lower']==addresses[lo])&(lines['upper']==addresses[up]))
            if len(indices)!=1:raise ValueError('기존 미표본 S 선 주소 누락/중복')
            i=int(indices[0]);inside=int(np.count_nonzero((nu>=lines['nu0'][i]*(1-profile['support_fraction']))&
                                                       (nu<=lines['nu0'][i]*(1+profile['support_fraction']))))
            specials.append(dict(Z=16,ion_number=charge,lower_FL=lo,upper_FL=up,line_index=i,
                original_nodes_in_support=inside,status=int(status[i]),profile_mass=float(mass[i]),
                jbar90=jbar[i].tolist()))
        if any(B.sha(Path(p))!=h for p,h in {**inputs,**code}.items()):raise ValueError('실행 중 입력/소스 변경')
        complete=bool(done.all() and not len(missing))
        result=dict(state='COMPLETE_LINEAR_FIELD_BB90' if complete else 'INCOMPLETE_FIELD_COVERAGE',
            source_commit=args.source_commit,lines=nr,points=npnt,frequencies=len(nu),profile=profile,
            input_sha256=inputs,code_sha256=code,signature=provider.signature,radiation_generation=70,
            status_counts={str(i):int(np.count_nonzero(status==i)) for i in (1,2,3)},
            out_of_grid_indices=missing.tolist(),previous_unsampled_lines=specials,line_selection=counts,
            output_sha256={name:B.sha(out/name) for name in ('jbar.npy','profile-mass.npy','status.npy',
                                                         'lines-complete.npy','line-roster.npz')},
            wall_seconds=time.monotonic()-start,field_representation='piecewise linear in frequency',
            no_extrapolation=True,no_profile_renormalization=True,native_discrete_operator_replaced=False,
            new_transport=0,new_native=0,SE_charge_RE_solved=False,final_goal='NOT_MET')
        (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:result[k] for k in ('state','lines','points','status_counts','wall_seconds')}),flush=True)
        return 0 if complete else 4
    finally:
        field._mmap.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source-commit',required=True)
    raise SystemExit(run(p.parse_args()))
