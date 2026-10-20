#!/usr/bin/env python3
"""완주한 실제 cell 장부의 물리 수지·보간 결함·수치 항등식을 분리한다."""
import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np

from run_cmfgen_formal_primary import read_cell_energy,fields
from prepare_cmfgen_mom_capture import sha


def energy_summary(record):
    b=record['budget'];flux=record['ray_flux'];native=record['native_flux']
    delta=flux[:-1]-flux[1:];end=record['work_first']-record['work_last']
    continuous_rhs=b[:,0]-b[:,1]+end
    discrete=delta-b[:,0]-b[:,2]
    continuous=delta-continuous_rhs
    identity=discrete-b[:,3]-b[:,4]
    net=float(flux[0]-flux[-1]);volume=float(continuous_rhs.sum());scale=max(abs(net),abs(volume))
    summary=dict(ray_boundary_outer=float(flux[0]),ray_boundary_inner=float(flux[-1]),
        ray_net_boundary=net,native_net_boundary=float(native[0]-native[-1]),
        physical_source=float(b[:,0].sum()),expansion_work=float(b[:,1].sum()),
        discrete_blue_exchange=float(b[:,2].sum()),frequency_end=float(end.sum()),
        geometry_interpolation_defect=float(b[:,3].sum()),SC_source_interpolation_defect=float(b[:,4].sum()),
        physical_continuous_rhs=volume,physical_continuous_signed_defect=float(continuous.sum()),
        physical_continuous_relative_to_net=abs(float(continuous.sum()))/scale if scale else None,
        physical_continuous_relative_to_native_outer=abs(float(continuous.sum()))/abs(float(native[0])) if native[0] else None,
        physical_continuous_cell_L1=float(abs(continuous).sum()),
        physical_discrete_signed_defect=float(discrete.sum()),physical_discrete_cell_L1=float(abs(discrete).sum()),
        numerical_identity_signed_defect=float(identity.sum()),numerical_identity_cell_L1=float(abs(identity).sum()),
        integrated_numerical_identity_L1=float(record['identity_L1'].sum()),
        frequency_discretization_difference=float((b[:,2]+b[:,1]-end).sum()),
        angular_boundary_difference=float(net-(native[0]-native[-1])),
        maximum_endpoint_error=float(record['maximum_endpoint_error']),
        maximum_frequency_numerical_identity_relative=float(record['maximum_identity_relative']))
    columns=dict(delta_ray_flux=delta,physical_source=b[:,0],expansion_work=b[:,1],
        discrete_blue_exchange=b[:,2],frequency_end=end,geometry_interpolation_defect=b[:,3],
        SC_source_interpolation_defect=b[:,4],physical_continuous_rhs=continuous_rhs,
        physical_continuous_defect=continuous,physical_discrete_defect=discrete,
        numerical_identity_defect=identity,integrated_numerical_identity_L1=record['identity_L1'])
    return summary,columns


def field_difference(run,baseline):
    g,a=fields(run/'FORMAL_PRIMARY_FIELDS.bin');oldg,b=fields(baseline/'FORMAL_PRIMARY_FIELDS.bin')
    if not np.array_equal(g,oldg) or len(a)!=len(b):raise ValueError('관측기 전후 격자 불일치')
    maximum=np.zeros(4);input_max=np.zeros(3)
    for start in range(0,len(a),1024):
        x=a[start:start+1024];y=b[start:start+1024]
        if not np.array_equal(x['freq'],y['freq']) or not np.array_equal(x['weight'],y['weight']):
            raise ValueError('관측기 전후 주파수/가중치 불일치')
        jscale=np.maximum(x['field'][:,0,:],y['field'][:,0,:])
        d=abs(x['field'][:,:4,:]-y['field'][:,:4,:])
        ratio=np.divide(d,jscale[:,None,:],out=np.zeros_like(d),where=jscale[:,None,:]>0)
        if np.any((jscale[:,None,:]==0)&(d!=0)):raise ValueError('영장 전후 비영 변화')
        maximum=np.maximum(maximum,ratio.max(axis=(0,2)))
        for i,field in enumerate((4,5,6)):
            xx=x['field'][:,field,:];yy=y['field'][:,field,:];scale=np.maximum(abs(xx),abs(yy))
            ratio=np.divide(abs(xx-yy),scale,out=np.zeros_like(xx),where=scale>0)
            input_max[i]=max(input_max[i],float(ratio.max()))
    return dict(all_frequency_depth_cells=len(a)*g.shape[1],
        maximum_J_H_K_N_change_over_local_J=maximum.tolist(),
        maximum_chi_eta_true_chi_scat_relative_change=input_max.tolist(),
        interpretation='모든 bin 포함. OpenMP 합산 순서/고정점 반올림은 byte 동일성으로 가정하지 않는다.')


def analyze(run,baseline,out):
    if out.exists():raise ValueError('기존 실제 cell 에너지 판독 보존')
    runtime=json.loads((run/'seq_logs/whole-runtime.json').read_text())
    if not runtime.get('diagnostic_complete') or runtime.get('returncode')!=97 or not runtime.get('cell_energy'):
        raise ValueError('완료된 전주파수 cell 진단이 아님')
    path=run/'FORMAL_CELL_ENERGY.bin'
    if sha(path)!=runtime['evidence_sha256'].get(str(path)):
        raise ValueError('cell 원본 신원 변경')
    record=read_cell_energy(path,90,197291)
    summary,columns=energy_summary(record)
    difference=field_difference(run,baseline)
    out.mkdir()
    with (out/'cells.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['outer_depth_1based',*columns])
        for i in range(89):w.writerow([i+1,*[v[i] for v in columns.values()]])
    sources=[Path(__file__).resolve(),Path(read_cell_energy.__code__.co_filename).resolve()]
    for source in sources:shutil.copy2(source,out/source.name)
    files=[path,run/'seq_logs/whole-runtime.json',run/'seq_logs/whole-inputs.json',
        run/'FORMAL_PRIMARY_FIELDS.bin',baseline/'FORMAL_PRIMARY_FIELDS.bin',*sources]
    report=dict(scope='실제90격자의 독립 ray 적분/보간 차이 분리; 수렴앵커/SiS 합격 아님',
        physical_reconstruction='native s에서 실제 ray chi/eta/b/blue/D 선형 보간; SC source/기하 결함은 물리 source에서 제외',
        warning='기하/SC source 결함을 더한 수치 항등식이 작아도 물리 에너지 보존 PASS가 아니다.',
        summary=summary,baseline_field_difference=difference,
        input_sha256={str(p):sha(p) for p in files},output_sha256=sha(out/'cells.csv'),
        energy_pass='NOT_JUDGED',final_goal='NOT_JUDGED')
    (out/'ray-energy.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(summary=summary,baseline_field_difference=difference),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','baseline','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();analyze(a.run.resolve(),a.baseline.resolve(),a.out.resolve())
