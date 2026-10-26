"""완료 공통 전하 후보와 같은 수용 evaluation의 열장부를 읽기 전용 인수한다."""
import json
from pathlib import Path
import re

import numpy as np

from analyze_cmfgen_material_inventory import sha


def read_npz(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in f.files}


def common_start(path,expected_sha,reference,inputs):
    path=Path(path).resolve()
    if not re.fullmatch('[0-9a-f]{64}',expected_sha or '') or sha(path)!=expected_sha:
        raise ValueError('완료 공통 전하 보고서 신원 불일치')
    report=json.loads(path.read_text());root=path.parent
    if (report['state']!='COMPLETE_FIXED_T_COMMON_CHARGE_INITIALIZATION' or
        report['charge_solved'] is not True or report['temperature_updated'] is not False or
        report['points']!=len(reference['temperature'])):
        raise ValueError('완료 고정 T 공통 전하 초기화가 아님')
    accepted=[r['evaluation'] for r in report['history'] if r['accepted']]
    if not accepted:raise ValueError('수용된 공통 전하 evaluation 없음')
    i=max(accepted)
    if i<1 or i>len(report['evaluations']):raise ValueError('잘못된 수용 evaluation 주소')
    directory=root/('evaluation-'+str(i));entry=report['evaluations'][i-1]
    identities={str(path):expected_sha,str(root/'accepted-response.npz'):report['accepted_output_sha256']}
    for name in ('response.npz','balance.npz'):
        identities[str(directory/name)]=entry['output_sha256'][name]
    if any(sha(Path(p))!=h for p,h in identities.items()):raise ValueError('수용 후보/열장부 신원 불일치')
    candidate=read_npz(root/'accepted-response.npz');direct=read_npz(directory/'response.npz')
    if set(candidate)!=set(direct) or any(not np.array_equal(v,direct[k]) for k,v in candidate.items()):
        raise ValueError('공통 전하 후보와 수용 evaluation의 배열 불일치')
    columns=candidate['columns'];p=candidate['population'];ne=candidate['ne']
    if (not np.array_equal(columns,reference['columns']) or
        not np.array_equal(candidate['temperature'],reference['temperature']) or
        p.shape!=reference['superlevel'].shape or ne.shape!=reference['ne'].shape or
        not np.isfinite(p).all() or np.any(p<=0) or not np.isfinite(ne).all() or np.any(ne<=0) or
        candidate['ledger'].shape!=(6,len(ne),7) or not np.isfinite(candidate['ledger']).all() or
        candidate['charge_residual'].shape!=ne.shape or not np.isfinite(candidate['charge_residual']).all() or
        float(candidate['ledger'][:,:,6].max())>1.e-8):
        raise ValueError('공통 전하 후보의 모형/물질/SE 불일치')
    charge=np.sum(columns[:,1,None]*p,axis=0,dtype=np.longdouble)-ne
    if (np.max(abs(charge/ne))>1.e-10 or
        np.max(abs((charge-candidate['charge_residual'])/ne))>1.e-12):
        raise ValueError('공통 전하 후보의 실제 전하 잔차 실패')
    for Z in np.unique(columns[:,0]):
        mask=columns[:,0]==Z;old=reference['superlevel'][mask].sum(axis=0)
        if not np.allclose(p[mask].sum(axis=0),old,rtol=1.e-10,atol=0):
            raise ValueError('공통 전하 후보의 원소총수 변경')
    previous=report['input_sha256']
    if any(k in inputs and inputs[k]!=v for k,v in previous.items()):
        raise ValueError('원 복사장/물질과 다른 공통 전하 후보')
    identities.update(previous)
    if any(sha(Path(p))!=h for p,h in identities.items()):raise ValueError('인수 중 원 입력 변경')
    inputs.update(identities)
    return candidate,read_npz(directory/'balance.npz'),directory
