import numpy as np
import pytest
import lumina_thermal_refinement as R
from test_lumina_thermal_refinement import fixture


def test_root_requires_fresh_check_after_actual_correction():
    evaluate,args=fixture()
    first=evaluate(np.array([2.,3.])*np.exp(.02),np.array([5.,7.]))
    # 먼저 같은 T의 전하를 닫은 실제 상태를 만든다.
    t=first[0]['temperature'];dt=np.log(t/args['first'][0]['temperature'])
    ne=np.array([5.,7.])*np.exp(.3*dt+.8*dt**2);first=evaluate(t,ne)
    calls=[]
    def actual(t,ne):
        value=evaluate(t,ne);calls.append(value);return value
    result=R.solve_root(actual,first,args['adiabatic'],args['lower'],args['upper'],policy=args['policy'])
    assert result['fixed_field_root_converged']
    assert result['accepted_iterations']==1
    assert len(calls)==result['evaluations_used']<=12
    assert len(result['centered_checks'])==2
    assert result['root_fresh_direction_max']<=1.e-3
    assert not result['joint_convergence']


def test_limit_is_not_root_even_when_heat_tolerance_passes():
    evaluate,a=fixture()
    result=R.solve_root(evaluate,a['first'],a['adiabatic'],a['lower'],a['upper'],policy=a['policy'],max_evaluations=4)
    assert result['evaluations_used']==4 and result['accepted_iterations']==0
    assert result['physical_tolerances_satisfied']
    assert not result['fixed_field_root_converged']


def test_failed_actual_candidate_preserves_base_and_budget():
    evaluate,a=fixture();count=0
    def bad(t,ne):
        nonlocal count
        count+=1;value=evaluate(t,ne)
        if count>4:value[1]['physical_stored_net'][:]=1.
        return value
    result=R.solve_root(bad,a['first'],a['adiabatic'],a['lower'],a['upper'],policy=a['policy'])
    assert result['selected'] is a['first']
    assert not result['fixed_field_root_converged']
    assert result['stop_reason']=='ACTUAL_BOX_CANDIDATE_NOT_ACCEPTED'
    assert count<=8
