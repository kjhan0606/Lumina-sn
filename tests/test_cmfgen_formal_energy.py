"""장부 항의 단위·부호·상쇄 확인. 실제 수송의 독립 물리 정답은 아니다."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_cmfgen_formal_energy import ledger


def fixture(beta=0.):
    r=np.array([2.,1.5,1.]);gamma=1/np.sqrt(1-beta*beta)
    geometry=np.array([r,np.full(3,beta*299792.458),np.full(3,-1.),np.full(3,gamma)])
    dtype=np.dtype([('freq','f8'),('weight','f8'),('field','f8',(8,3))])
    # 실제 2/1 Hz를 native FL 단위로 주고 FQW는 Hz로 유지한다.
    a=np.zeros(2,dtype=dtype);a['freq']=[2e-15,1e-15];a['weight']=.5
    a['field'][:,0,:]=3;a['field'][:,1,:]=1/r;a['field'][:,2,:]=1
    a['field'][:,5,:]=gamma*(1+6*beta*r)/(r*r);a['field'][:,7,:]=3
    return geometry,a


def test_static_boundary_equals_volume_without_arbitrary_floor():
    g,a=fixture();summary,_,cells=ledger(g,a)
    assert abs(summary['net_boundary']-1)<1e-14
    assert summary['absolute_defect']<1e-14
    assert max(abs(cells['signed_defect']))<1e-14
    a['field'][:,5,:]+=.1/g[0]**2
    summary,_,_=ledger(g,a)
    assert abs(summary['signed_defect']+.1)<1e-14


def test_relativistic_work_frequency_endpoint_and_large_term_cancellation():
    g,a=fixture(.1);summary,_,_=ledger(g,a)
    assert summary['budgets']['expansion_work']>0
    assert abs(summary['budgets']['frequency_end_term']-summary['budgets']['expansion_work'])<1e-14
    assert summary['absolute_defect']<1e-14
    a['field'][:,4,:]=100
    a['field'][:,5,:]+=300
    summary,_,_=ledger(g,a)
    assert summary['absolute_defect']<1e-12
    assert summary['absolute_term_scale']>1000*abs(summary['net_boundary'])/2
