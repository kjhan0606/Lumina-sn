"""새 장 열보정 연결의 보존·수용·실제 callback 소비 시험. 실제 수송은 없다."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import lumina_fixed_field_thermal as T


def case(temperature, ne, number=0):
    temperature=np.array(temperature, copy=True);ne=np.array(ne, copy=True)
    net=np.log(temperature/np.array([2.,3.]))
    data=dict(temperature=temperature,ne=ne,population=np.ones((2,2)),
              charge_residual=np.log(ne/np.array([5.,7.])))
    heat=dict(physical_adiabatic=np.zeros(2),physical_stored_net=net,
              physical_stored_gross=np.full(2,2.),physical_internal_energy=np.ones(2))
    return data,heat,dict(number=number)


def adiabatic(t,ne,nuclei,u):
    return np.zeros_like(t)


def test_actual_new_field_callback_reaches_fixed_field_root_not_joint_convergence():
    first=case(np.array([2.,3.])*1.02,[5.,7.]);saved=first[0]['temperature'].copy();calls=[];decisions=[]
    def evaluate(t,ne):
        calls.append((t.copy(),ne.copy()))
        return case(t,ne,len(calls))
    result=T.solve(evaluate,first,adiabatic,policy='physical_stored',max_evaluations=12,
                   on_evaluation=lambda value,entry:decisions.append((value[2]['number'],entry.copy())))
    assert result['fixed_field_numerical_converged']
    assert result['accepted_iterations']>=1 and result['evaluations_used']==len(calls)
    np.testing.assert_allclose(result['selected'][0]['temperature'],[2.,3.],rtol=1e-10)
    np.testing.assert_array_equal(first[0]['temperature'],saved)
    assert result['selected'][2]['number']>0 and decisions
    assert result['new_transport']==0 and not result['joint_convergence']


def test_rejected_thermal_step_preserves_exact_initial_payload():
    first=case([2.04,3.06],[5.,7.]);calls=[]
    def evaluate(t,ne):
        calls.append(1);value=case(t,ne,len(calls))
        if len(calls)==5:value[1]['physical_stored_net'][:]=100.
        return value
    result=T.solve(evaluate,first,adiabatic,policy='physical_stored',max_evaluations=5)
    assert result['selected'] is first
    assert result['accepted_iterations']==0 and not result['fixed_field_numerical_converged']
    assert len(calls)==5


def test_temperature_callback_must_consume_requested_state():
    first=case([2.04,3.06],[5.,7.])
    with pytest.raises(ValueError,match='T/ne'):
        T.solve(lambda t,ne:case(t,ne*1.01),first,adiabatic,
                policy='physical_stored',max_evaluations=5)


@pytest.mark.parametrize('limit',[0,4,13,True])
def test_evaluation_limits_checked_before_expensive_callback(limit):
    def never(*args):raise AssertionError('실제 물질평가를 호출하면 안 됨')
    with pytest.raises(ValueError):
        T.solve(never,case([2.04,3.06],[5.,7.]),adiabatic,
                policy='physical_stored',max_evaluations=limit)
