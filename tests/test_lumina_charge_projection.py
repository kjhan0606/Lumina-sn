from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from lumina_charge_projection import proposal,trial
from lumina_coupled_material import MaterialPacket,jacobian,newton_direction,residual


def case():
    nuclei=np.full(3,1.e8);target=np.array([1.06,1.08,1.09])*1.e4
    root=lambda t:2.e8+2.e8*np.log(t/1.e4)**2
    def ad(t,ne,ni,u):
        return 1.e-3*(np.log(t/1.e4)-np.roll(np.log(t/1.e4),1))+1.e-4*np.log(ne/2.e8)
    target_ad=ad(target,root(target),nuclei,nuclei*1.e-11)
    def evaluate(t,ne):
        return MaterialPacket(t,ne,nuclei,nuclei*1.e-11,
            target_ad+.1*np.log(t/target),root(t)-ne,np.ones(3))
    t=np.full(3,1.e4);ne=root(t);base=evaluate(t,ne);h=1.e-4
    j=jacobian(base,evaluate(t*np.exp(h),ne),evaluate(t*np.exp(-h),ne),
        evaluate(t,ne*np.exp(h)),evaluate(t,ne*np.exp(-h)),h,ad)
    f=residual(base,ad,ne,base.thermal_scale)
    basis=dict(temperature=t,ne=ne,charge_scale=ne,thermal_scale=base.thermal_scale,
               scaled_residual=f,jacobian=j,direction=newton_direction(j,f))
    return base,ad,basis,evaluate


def test_one_temperature_is_projected_with_actual_charge_and_depth_heat():
    base,ad,basis,evaluate=case();seen=[];records=[]
    def actual(t,ne):seen.append((t.copy(),ne.copy()));return evaluate(t,ne)
    result=trial(actual,base,ad,basis,1.,np.full(3,1.e8),np.full(3,4.e8),
                 on_evaluation=lambda p,e:records.append(e))
    assert result['iterations']==1 and result['history'][0]['charge_solved']
    assert 2<=len(seen)<=8 and len(seen)==result['evaluations_used']
    assert all(np.array_equal(t,seen[0][0]) for t,ne in seen)
    assert abs(result['packet'].charge/result['packet'].ne).max()<=1.e-10
    assert result['history'][0]['scaled_max']<result['history'][0]['scaled_start_max']
    assert records[-1]['phase']=='charge_projected_line'
    assert not result['numerical_converged'] # 한 번의 온도 제안을 공동수렴으로 표시하지 않는다.


def test_charge_budget_exhaustion_or_thermal_worsening_preserves_base():
    base,ad,basis,evaluate=case();seen=[]
    def actual(t,ne):seen.append(ne.copy());return evaluate(t,ne)
    result=trial(actual,base,ad,basis,1.,np.full(3,1.e8),np.full(3,4.e8),max_evaluations=1)
    assert len(seen)==1 and not result['history'][0]['charge_solved']
    assert result['packet'] is base and result['iterations']==0
    def worse(t,ne):
        p=evaluate(t,ne);return replace(p,local_heat=p.local_heat+.2)
    result=trial(worse,base,ad,basis,1.,np.full(3,1.e8),np.full(3,4.e8))
    assert result['history'][0]['charge_solved'] and not result['history'][0]['accepted']
    assert result['packet'] is base


def test_wrong_basis_bounds_and_actual_temperature_are_rejected():
    base,ad,basis,evaluate=case()
    with pytest.raises(ValueError,match='같은 선형계'):
        proposal(base,ad,{**basis,'direction':np.zeros(6)},1.)
    with pytest.raises(ValueError,match='수밀도'):
        trial(evaluate,base,ad,basis,1.,np.full(3,3.e8),np.full(3,4.e8))
    with pytest.raises(ValueError,match='T/ne'):
        trial(lambda t,ne:evaluate(t*1.01,ne),base,ad,basis,1.,np.full(3,1.e8),np.full(3,4.e8))
    with pytest.raises(ValueError,match='기준 잔차'):
        proposal(base,ad,{**basis,'scaled_residual':basis['scaled_residual']+1.e-5},1.)
