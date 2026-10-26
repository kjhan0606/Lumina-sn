#!/usr/bin/env python3
"""완성 BF90을 실제 C 원소 블록에서 소비한다. 공동 SE/전하/RE 해가 아니다.

동일 물질의 전 PHOT 경로를 재적분 없이 전달하고, 전체 광자/전자 에너지를
경로별 독립 합산과 비교한다. 부분 BF 입력을 완료 입력으로 바꾸지 않는다.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import struct
import subprocess

import numpy as np

import cmfgen_model_continuum as B


HEADER=struct.Struct('<8sIIII2d')


def addresses(rows,columns,roster):
    """FL→실제 SL/terminal 원자 주소. 가장 가까운 준위로 대체하지 않는다."""
    index={tuple(map(int,c)):i for i,c in enumerate(columns)}
    if len(index)!=len(columns):
        raise ValueError('중복 물질 SL/terminal')
    lower=[];upper=[]
    for lo,up,z,stage in zip(roster['lower'],roster['target'],roster['Z'],roster['stage']):
        lo,up,z,stage=map(int,(lo,up,z,stage))
        if not 0<=lo<len(rows) or up < -1 or up>=len(rows):
            raise ValueError('BF FL 주소 범위 오류')
        a=rows[lo];key=(int(a['atomic_number']),int(a['ion_number']),int(a['super_level']))
        if key[:2]!=(z,stage-1):
            raise ValueError('BF 하위 이온/원자 주소 오류')
        if up>=0:
            b=rows[up];target=(int(b['atomic_number']),int(b['ion_number']),int(b['super_level']))
            if target[:2]!=(z,stage):
                raise ValueError('BF 상위 이온/원자 주소 오류')
        else:
            target=(z,stage,-1)
        lower.append(index[key]);upper.append(index[target])
    return np.asarray(lower),np.asarray(upper)


def energy_prediction(snapshot,roster,rates,lower,upper):
    """원 FL 점유수와 단위 PHOT율로 계산한다. C SL 행렬/열계수는 읽지 않는다."""
    lo=np.asarray(roster['lower']);pop=snapshot['superlevel']
    nlo=np.asarray(pop[lower]*snapshot['fraction'][lo],dtype=np.longdouble)
    nup=np.asarray(pop[upper],dtype=np.longdouble)
    prob=np.asarray(roster['probability'],dtype=np.longdouble)[:,None]
    edge=np.asarray(roster['threshold'],dtype=np.longdouble)[:,None]*np.longdouble('6.62607015e-27')
    photo=nlo*rates[:,:,0];recomb=nup*(rates[:,:,1]+rates[:,:,2])
    electron=prob*(nlo*rates[:,:,3]-nup*(rates[:,:,4]+rates[:,:,5]))
    threshold=prob*edge*(photo-recomb)
    gross=prob*(np.abs(nlo*rates[:,:,3])+np.abs(nup*rates[:,:,4])+np.abs(nup*rates[:,:,5])+
                edge*(photo+recomb))
    return {int(z):dict(electron=electron[roster['Z']==z].sum(axis=0),
                       photon=(electron+threshold)[roster['Z']==z].sum(axis=0),
                       gross=gross[roster['Z']==z].sum(axis=0)) for z in np.unique(roster['Z'])}


def write_handoff(path,snapshot,roster,rates):
    nr,ns,nvalues=rates.shape;nl=len(snapshot['fraction'])
    if ns!=90 or nvalues!=6 or snapshot['saha'].shape!=(nr,ns):
        raise ValueError('전90점 BF handoff 형상 오류')
    with path.open('xb') as f:
        f.write(HEADER.pack(b'LMBF90V1',nl,nr,ns,2**32-1,0.,0.))
        for name in ('temperature','ne'):
            f.write(np.asarray(snapshot[name],dtype='<f8').tobytes())
        for r in range(nr):
            f.write(struct.pack('<5i',int(roster['lower'][r]),int(roster['target'][r]),
                                int(roster['Z'][r]),int(roster['stage'][r]),int(roster['phot_grid'][r])))
            f.write(struct.pack('<3d',roster['probability'][r],roster['threshold'][r],roster['sigma_edge'][r]))
            f.write(np.asarray(np.column_stack((snapshot['saha'][r],rates[r])),dtype='<f8').tobytes())


def run(args):
    if os.environ.get('SLURM_CPUS_PER_TASK')!='1' or os.environ.get('SLURM_MEM_PER_NODE')!='4096':
        raise ValueError('CPU1/4GiB Slurm 직접 소비 시험 전용')
    report_path=args.source.resolve();root=report_path.parent;out=args.out.resolve()
    if out.exists():raise FileExistsError('기존 BF90 소비 결과 보존')
    report=json.loads(report_path.read_text())
    if (report.get('state')!='COMPLETE_FROZEN_MATERIAL_BF_90' or
        (report['material_points'],report['model_ions'],report['model_levels'],report['routes_complete'],
         report['frequency_points'])!=(90,27,24542,24667,197291) or not report['data_unchanged']):
        raise ValueError('완료한 전모형 BF90 입력만 소비 가능')
    identities={str(report_path):B.sha(report_path),**report['data_sha256']}
    identities.update({str(root/name):sha for name,sha in report['output_sha256'].items()})
    project=Path(__file__).resolve().parents[1]
    old=Path('/gpfs/kjhan/cmfgen_runs/lumina_direct_slprofile_20261026-UUiwsU')
    parameters=old/'model-occupation-parameters.bin'
    base_path=Path('/gpfs/kjhan/cmfgen_runs/lumina_extended_4h_20261022-rJGMVR/gpu-local-plan.json')
    base=json.loads(base_path.read_text());config_path=Path(base['config']);config=json.loads(config_path.read_text())
    material=Path(config['seed'])/'native_material_nodes.bin'
    levels=Path(config['seed'])/'atomic/levels.csv'
    for p in (parameters,base_path,config_path,levels,args.projection,Path(__file__),
              args.projection.with_name('source-before.sha256'),project/'data/atomic/topion_levels.csv',
              project/'data/atomic/ionization_reference.csv'):
        identities[str(p.resolve())]=B.sha(p)
    if report['source_identity']!=base['source_identity']:
        raise ValueError('BF90 실제 물질 계보 불일치')
    if B.sha(levels)!=report['data_sha256'].get(str(Path(config['cells'])/'atomic/levels.csv')):
        raise ValueError('BF 적분과 C 원자모형의 FL roster 불일치')
    if any(B.sha(Path(p))!=sha for p,sha in identities.items()):
        raise ValueError('BF90 봉인 입력/완료 배열 변경')
    done=np.load(root/'routes-complete.npy')
    rates=np.load(root/'total-rates.npy',mmap_mode='r')
    with np.load(root/'material-snapshot.npz') as f:snapshot={k:f[k] for k in f.files}
    with np.load(root/'route-roster.npz') as f:roster={k:f[k] for k in f.files}
    if (rates.shape!=(24667,90,6) or done.shape!=(24667,) or done.dtype!=bool or not done.all() or
        not np.isfinite(rates).all() or np.any(rates[:,:,:3]<0)):
        raise ValueError('부분/비유한 BF 입력')
    rows=list(csv.DictReader(levels.open()))
    lower,upper=addresses(rows,snapshot['columns'],roster)
    # 원래점 물질 자체가 다르면 C 소비 전에 거부한다.
    with material.open('rb') as f:
        tag,ns,nc,nl,epoch=struct.unpack('<8sIIId',f.read(28))
        columns=np.fromfile(f,dtype='<i4',count=3*nc).reshape(nc,3)
        physical=np.fromfile(f,dtype='<f8',count=4*ns).reshape(ns,4)
        pop=np.fromfile(f,dtype='<f8',count=nc*ns).reshape(ns,nc).T
        if f.read(1):raise ValueError('물질 파일 잉여 bytes')
    if (tag!=b'LUMNOD01' or (ns,nl)!=(90,len(rows)) or
        not np.array_equal(columns,snapshot['columns']) or not np.array_equal(pop,snapshot['superlevel']) or
        not np.array_equal(physical[:,1],snapshot['temperature']) or not np.array_equal(physical[:,2],snapshot['ne'])):
        raise ValueError('BF90과 원 물질/점 주소 불일치')
    out.mkdir();(out/'inputs.json').write_text(json.dumps(dict(input_sha256=identities,
        scope=__doc__,resources='CPU1/4GiB/10min/GPU0/retry0'),indent=2)+'\n')
    handoff=out/'all-BF90.bin';write_handoff(handoff,snapshot,roster,rates)
    occupation=out/'occupation.bin'
    with occupation.open('xb') as f:
        f.write(struct.pack('<8sII',b'LMATOCC1',len(rows),90))
        f.write(np.asarray(snapshot['occupation'],dtype='<f8').tobytes())
    exported=out/'imported-full-population.bin'
    env=dict(os.environ,LUMINA_ARTIS_PARITY='0',LUMINA_NLTE_BF_COLLISION='1',
        LUMINA_TOPION_LEVELS_FILE=str(project/'data/atomic/topion_levels.csv'),
        LUMINA_IONIZATION_REFERENCE_FILE=str(project/'data/atomic/ionization_reference.csv'))
    command=[str(args.projection),str(levels.parent),'90',str(material),str(exported),
             str(occupation),str(parameters),str(handoff)]
    child=subprocess.run(command,env=env,capture_output=True,text=True,timeout=420)
    (out/'C.stdout').write_text(child.stdout);(out/'C.stderr').write_text(child.stderr)
    if child.returncode:raise ValueError('실제 C 전90점 BF 소비 실패')
    logs=re.findall(r'\[MODEL-SE\]\[BF-ENERGY\] Z=(\d+) native_node=(\d+) electron_erg_cm3_s=(\S+) photon_erg_cm3_s=(\S+)',child.stdout)
    actual={(int(z),int(s)):(float(e),float(p)) for z,s,e,p in logs}
    prediction=energy_prediction(snapshot,roster,rates,lower,upper)
    expected={(z,s) for z in prediction for s in range(90)}
    if len(actual)!=len(logs) or set(actual)!=expected:raise ValueError('전원소/전90점 에너지 소비 누락/중복')
    checks=[]
    for (z,s),(electron,photon) in sorted(actual.items()):
        p=prediction[z];scale=p['gross'][s]
        error=max(abs(electron-p['electron'][s]),abs(photon-p['photon'][s]))
        if not np.isfinite([electron,photon,error,scale]).all() or error>1.e-10*scale:
            raise ValueError('실제 C와 독립 광자/전자 에너지 장부 불일치')
        checks.append(dict(Z=z,node=s,electron=electron,photon=photon,
                           error_over_gross=float(error/scale) if scale else 0.))
    if (B.sha(exported)!=identities[config['native_full_population']] or
        any(B.sha(Path(p))!=sha for p,sha in identities.items())):
        raise ValueError('원 인구/입력 보존 실패')
    result=dict(state='COMPLETE_FROZEN_BF90_C_CONSUMPTION',source_result=str(report_path),
        input_sha256=identities,elements=len(prediction),points=90,routes=24667,
        checks=checks,command=command,CI_enabled=True,public_material_updates=0,
        new_BF=0,new_BB=0,new_native=0,new_transport=0,SE_charge_RE_solved=False,
        final_goal='NOT_MET')
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(state=result['state'],elements=result['elements'],points=90,
                         maximum_error_over_gross=max(c['error_over_gross'] for c in checks))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--projection',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args())
