#!/usr/bin/env python3
"""실제 angular 공통장의 source/팽창/경계 에너지 장부. 물리 PASS를 판정하지 않는다."""
import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np

from run_cmfgen_formal_primary import fields
from run_cmfgen_lambda_probe import sha


def mechanical_coefficient(r,beta,sigma,gamma,j,h,k):
    # native CMFGEN_SUB의 USE_J_REL .AND. INCL_REL_TERMS 분기와 같은 계수.
    return r*beta*gamma*(j-k+gamma*gamma*(sigma+1)*(k+beta*h))


def ledger(geometry,records):
    r,v,sigma,gamma=geometry;beta=v/299792.458
    n=len(r);jint=np.zeros(n);hint=np.zeros(n);kint=np.zeros(n)
    emission=np.zeros(n);absorption=np.zeros(n);scatter_lag=np.zeros(n);work=np.zeros(n)
    abs_absorption=np.zeros(n);abs_work=np.zeros(n)
    for start in range(0,len(records),1024):
        b=records[start:start+1024];w=b['weight'][:,None]
        j,h,k,nn,chi,eta,scat,source=np.moveaxis(b['field'],1,0)
        local_abs=(chi-scat)*j
        local_work=mechanical_coefficient(r,beta,sigma,gamma,j,h,k)
        jint+=np.sum(w*j,axis=0);hint+=np.sum(w*h,axis=0);kint+=np.sum(w*k,axis=0)
        emission+=np.sum(w*eta,axis=0);absorption+=np.sum(w*local_abs,axis=0)
        scatter_lag+=np.sum(w*scat*(source-j),axis=0)
        work+=np.sum(w*local_work,axis=0)
        abs_absorption+=np.sum(w*abs(local_abs),axis=0);abs_work+=np.sum(w*abs(local_work),axis=0)
    def endpoint(row):
        j,h,k=row['field'][:3]
        # FL은 10^15 Hz 단위지만 실제 FQW는 Hz다. 끝항도 반드시 Hz로 맞춘다.
        return row['freq']*1e15*mechanical_coefficient(r,beta,sigma,gamma,j,h,k)
    frequency_end=endpoint(records[0])-endpoint(records[-1])
    luminosity=r*r*gamma*(hint+beta*jint)
    source=r*r*(emission-absorption+scatter_lag)-work+frequency_end
    dr=r[:-1]-r[1:]
    cell_source=.5*dr*(source[:-1]+source[1:])
    cell_flux=luminosity[:-1]-luminosity[1:]
    cell_defect=cell_flux-cell_source
    integral=lambda y:float(np.sum(.5*dr*(y[:-1]+y[1:])))
    boundary=float(luminosity[0]-luminosity[-1]);volume=integral(source);defect=boundary-volume
    budgets=dict(true_emission=integral(r*r*emission),true_absorption=integral(r*r*absorption),
        scattering_source_lag=integral(r*r*scatter_lag),expansion_work=integral(work),frequency_end_term=integral(frequency_end))
    absolute_scale=(abs(float(luminosity[0]))+abs(float(luminosity[-1]))
        +integral(r*r*(abs(emission)+abs_absorption+abs(scatter_lag)))
        +integral(abs_work)+integral(abs(frequency_end)))
    net_scale=max(abs(boundary),abs(volume))
    summary=dict(boundary_outer=float(luminosity[0]),boundary_inner=float(luminosity[-1]),net_boundary=boundary,
        volume_rhs=volume,signed_defect=defect,absolute_defect=abs(defect),budgets=budgets,
        relative_to_net=abs(defect)/net_scale if net_scale else None,
        absolute_term_scale=absolute_scale,relative_to_absolute_terms=abs(defect)/absolute_scale if absolute_scale else None,
        sum_absolute_cell_defect=float(np.sum(abs(cell_defect))))
    columns=dict(r=r,v_km_s=v,sigma=sigma,gamma=gamma,J_integral=jint,H_integral=hint,K_integral=kint,
        eta_true_integral=emission,chi_abs_J_integral=absorption,scattering_source_lag=scatter_lag,
        expansion_work=work,frequency_end_term=frequency_end,luminosity=luminosity,volume_rhs_density=source)
    return summary,columns,dict(flux_difference=cell_flux,source_integral=cell_source,signed_defect=cell_defect)


def analyze(run,out):
    if out.exists():raise ValueError('기존 에너지 장부 보존')
    runtime=json.loads((run/'seq_logs/whole-runtime.json').read_text())
    if not runtime.get('diagnostic_complete') or runtime.get('returncode')!=97 or runtime['coverage']['count']!=197291:
        raise ValueError('부분 공통장을 전주파수 장부로 소비하지 않음')
    geometry,a=fields(run/'FORMAL_PRIMARY_FIELDS.bin')
    summary,columns,cells=ledger(geometry,a)
    out.mkdir()
    producers=[Path(__file__).resolve()]+[Path(__file__).with_name(name).resolve() for name in (
        'run_cmfgen_formal_primary.py','run_cmfgen_lambda_probe.py','configure_cmfgen_stageA.py','ptf_gated_metrics.py')]
    for source_path in producers:shutil.copy2(source_path,out/source_path.name)
    with (out/'depths.csv').open('x') as f:
        writer=csv.writer(f);writer.writerow(['depth_1based',*columns])
        for i in range(geometry.shape[1]):writer.writerow([i+1,*[v[i] for v in columns.values()]])
    with (out/'cells.csv').open('x') as f:
        writer=csv.writer(f);writer.writerow(['outer_depth_1based',*cells])
        for i in range(geometry.shape[1]-1):writer.writerow([i+1,*[v[i] for v in cells.values()]])
    report=dict(scope='실제 90격자의 연속 에너지식에 대한 trapezoid 장부; 원래 MOM 잔차/격자수렴/최종합격 대체 아님',
        equation='d[gamma*r^2*(H+beta*J)]/dr = r^2*(eta_true-chi_abs*J+scatter_source_lag)-work+[nu*work_nu]_low^high',
        quadrature='기록된 native 주파수 가중치 및 실제 r 격자의 cell trapezoid',
        units='native RTE r 단위; 기록된 FQW는 Hz, FL은 1e15 Hz 단위이므로 주파수 끝항에 명시적으로 1e15 적용',
        summary=summary,input_sha256={str(p):sha(p) for p in (run/'FORMAL_PRIMARY_FIELDS.bin',run/'seq_logs/whole-runtime.json',Path(__file__))},
        producer_sha256={str(p):sha(p) for p in producers},
        energy_pass='NOT_JUDGED',final_goal='NOT_JUDGED')
    with (out/'energy-ledger.json').open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();analyze(a.run,a.out)
