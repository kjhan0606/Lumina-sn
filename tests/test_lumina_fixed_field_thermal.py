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


def projection_case():
    from test_lumina_charge_projection import case as build
    from lumina_charge_projection import proposal
    base, ad, basis, evaluate = build()
    def payload(t, ne, number=0):
        p = evaluate(t, ne); net = p.local_heat-ad(t, ne, p.nuclei, p.internal_energy)
        return (dict(temperature=t, ne=ne, population=p.nuclei[None,:], charge_residual=p.charge),
            dict(physical_adiabatic=ad(t,ne,p.nuclei,p.internal_energy),
                 physical_stored_net=net, physical_stored_gross=2*p.thermal_scale-abs(net),
                 physical_internal_energy=p.internal_energy), dict(number=number))
    first = payload(base.temperature, base.ne)
    t, ne = proposal(base, ad, basis, 1.)
    return first, payload(t, ne), ad, basis, payload


def test_same_temperature_charge_projection_reuses_saved_trial_then_evaluates_only_ne():
    first, saved, ad, basis, payload = projection_case(); fresh=[]; replayed=[]; decisions=[]
    def evaluate(t, ne):
        np.testing.assert_array_equal(t, saved[0]['temperature'])
        value=payload(t,ne,len(fresh)+1); fresh.append(value); return value
    result=T.project(evaluate,first,saved,ad,basis,1.,np.full(3,1.e8),np.full(3,4.e8),
        policy='physical_stored',max_evaluations=4,on_replay=replayed.append,
        on_evaluation=lambda value,record:decisions.append(record.copy()))
    assert replayed==[saved] and 1<=len(fresh)<=4
    assert result['replayed_evaluations']==1 and result['fresh_evaluations_used']==len(fresh)
    assert result['evaluations_used']==len(result['history'])==len(fresh)+1
    assert result['accepted_iterations']==1 and result['charge_relative']<=1.e-10
    assert result['history'][-1]['phase']=='charge_projected_line'
    assert result['history'][-1]['accepted'] is True
    assert result['selected'] is fresh[-1] and not result['joint_convergence']
    assert [e['evaluation'] for e in result['history']]==list(range(1,len(fresh)+2))


@pytest.mark.parametrize('fault',['charge','heat'])
def test_projection_cannot_accept_charge_failure_or_thermal_worsening(fault):
    first,saved,ad,basis,payload=projection_case();fresh=[]
    def evaluate(t,ne):
        value=payload(t,ne);fresh.append(value)
        if fault=='charge':value[0]['charge_residual']=saved[0]['charge_residual'].copy()
        else:value[1]['physical_stored_net']+=.2
        return value
    result=T.project(evaluate,first,saved,ad,basis,1.,np.full(3,1.e8),np.full(3,4.e8),
        policy='physical_stored',max_evaluations=1)
    assert len(fresh)==1 and result['selected'] is first
    assert result['accepted_iterations']==0 and not result['fixed_field_numerical_converged']


def test_projection_rejects_wrong_saved_trial_before_any_fresh_evaluation():
    first,saved,ad,basis,payload=projection_case()
    saved[0]['temperature']=saved[0]['temperature']*1.001
    def never(*args):raise AssertionError('다른 T 자료를 새 적분으로 대체하면 안 됨')
    with pytest.raises(ValueError,match='제안 T/ne'):
        T.project(never,first,saved,ad,basis,1.,np.full(3,1.e8),np.full(3,4.e8),
            policy='physical_stored',max_evaluations=4)


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


def test_direction_is_available_before_expensive_candidate_and_can_reproduce_it():
    first=case([2.04,3.06],[5.,7.]);calls=[];directions=[]
    def evaluate(t,ne):
        calls.append((t.copy(),ne.copy()))
        if len(calls)==5:
            assert directions, '방향 후보 평가 전에 원 선형계가 보존돼야 함'
        return case(t,ne,len(calls))
    def save(value,matrix,residual,delta,charge_scale,thermal_scale):
        assert len(calls)==4 and value is first
        np.testing.assert_allclose(matrix@delta,-residual,rtol=1e-12,atol=1e-16)
        directions.append((value,delta.copy()))
    result=T.solve(evaluate,first,adiabatic,policy='physical_stored',max_evaluations=5,
                   on_direction=save)
    value,delta=directions[0]
    alpha=min(1.,.1/np.max(abs(delta)))
    candidate=np.exp(np.r_[np.log(value[0]['temperature']),np.log(value[0]['ne'])]+alpha*delta)
    np.testing.assert_array_equal(candidate,np.r_[calls[4][0],calls[4][1]])
    assert result['accepted_iterations']==1 and len(directions)==1


def test_failure_after_direction_preserves_callback_output_without_returning_false_success():
    directions=[];calls=[]
    def evaluate(t,ne):
        calls.append(1)
        if len(calls)==5:raise TimeoutError('후보 평가 중 외부 시간 종료')
        return case(t,ne)
    with pytest.raises(TimeoutError):
        T.solve(evaluate,case([2.04,3.06],[5.,7.]),adiabatic,
            policy='physical_stored',max_evaluations=5,
            on_direction=lambda *args:directions.append(args))
    assert len(directions)==1 and len(calls)==5


def test_resume_reuses_rejected_trial_and_probes_then_evaluates_only_halved_candidate():
    first=case([2.04,3.06],[5.,7.]);saved=[]
    def original(t,ne):
        value=case(t,ne,len(saved)+1)
        if len(saved)==4:value[1]['physical_stored_net'][:]=100.
        saved.append(value);return value
    stopped=T.solve(original,first,adiabatic,policy='physical_stored',max_evaluations=5)
    assert stopped['selected'] is first and len(saved)==5
    snapshots=[v[0]['temperature'].copy() for v in saved];fresh=[];replayed=[]
    def additional(t,ne):
        fresh.append(case(t,ne,6));return fresh[-1]
    continued=T.solve(additional,first,adiabatic,policy='physical_stored',max_evaluations=1,
        replay=saved,on_replay=lambda v:replayed.append(v))
    assert len(fresh)==1 and len(replayed)==5
    assert continued['fresh_evaluations_used']==1 and continued['replayed_evaluations']==5
    assert continued['selected'] is fresh[0] and continued['accepted_iterations']==1
    assert continued['history'][-1]['alpha']==.5
    for value,snapshot in zip(saved,snapshots):np.testing.assert_array_equal(value[0]['temperature'],snapshot)


def test_replay_of_partial_derivatives_does_not_repeat_actual_evaluations():
    first=case([2.04,3.06],[5.,7.]);saved=[]
    def stopped(t,ne):
        if len(saved)==2:raise TimeoutError('네 섭동 중 두 개 완료 후 종료')
        saved.append(case(t,ne,len(saved)+1));return saved[-1]
    with pytest.raises(TimeoutError):
        T.solve(stopped,first,adiabatic,policy='physical_stored',max_evaluations=5)
    fresh=[]
    def additional(t,ne):fresh.append(case(t,ne));return fresh[-1]
    result=T.solve(additional,first,adiabatic,policy='physical_stored',max_evaluations=3,replay=saved)
    assert len(fresh)==3 and result['replayed_evaluations']==2
    assert result['fixed_field_numerical_converged']


def test_accepted_changed_state_builds_new_derivatives_not_old_newton_matrix():
    def nonlinear(t,ne):
        value=case(t,ne)
        value[1]['physical_stored_net']=np.asarray(t)/np.array([2.,3.])-1.
        return value
    first=nonlinear([2.04,3.06],[5.,7.]);saved=[]
    def original(t,ne):saved.append(nonlinear(t,ne));return saved[-1]
    stopped=T.solve(original,first,adiabatic,policy='physical_stored',max_evaluations=5)
    assert stopped['accepted_iterations']==1 and not stopped['fixed_field_numerical_converged']
    fresh=[]
    def additional(t,ne):fresh.append(nonlinear(t,ne));return fresh[-1]
    continued=T.solve(additional,first,adiabatic,policy='physical_stored',max_evaluations=5,replay=saved)
    assert len(fresh)==5 and continued['accepted_iterations']==2
    np.testing.assert_allclose(fresh[0][0]['temperature'],stopped['selected'][0]['temperature']*np.exp(1.e-4),rtol=1e-15)
    assert not np.array_equal(fresh[0][0]['temperature'],saved[0][0]['temperature'])


def test_other_energy_policy_reuses_four_SE_probes_but_builds_its_own_direction():
    def both(t,ne):
        value=case(t,ne)
        value[1].update(native_adiabatic=np.zeros(2),native_internal_energy=np.ones(2),
            native_instantaneous_scenario_net=value[1]['physical_stored_net']+.005,
            native_instantaneous_scenario_gross=np.full(2,2.005))
        return value
    first=both([2.04,3.06],[5.,7.]);saved=[]
    def original(t,ne):saved.append(both(t,ne));return saved[-1]
    old=T.solve(original,first,adiabatic,policy='physical_stored',max_evaluations=5)
    fresh=[]
    def additional(t,ne):fresh.append(both(t,ne));return fresh[-1]
    new=T.solve(additional,first,adiabatic,policy='native_instantaneous_scenario',
        max_evaluations=1,replay=saved[:4])
    assert new['fresh_evaluations_used']==1 and new['replayed_evaluations']==4
    assert new['fixed_field_numerical_converged']
    assert not np.array_equal(old['selected'][0]['temperature'],new['selected'][0]['temperature'])
    np.testing.assert_allclose(new['selected'][0]['temperature'],np.array([2.,3.])*np.exp(-.005),rtol=1e-10)


@pytest.mark.parametrize('fault',['order','temperature','ne'])
def test_mismatched_replay_never_falls_back_to_expensive_evaluation(fault):
    first=case([2.04,3.06],[5.,7.]);saved=[]
    def original(t,ne):saved.append(case(t,ne));return saved[-1]
    T.solve(original,first,adiabatic,policy='physical_stored',max_evaluations=5)
    if fault=='order':saved[0],saved[1]=saved[1],saved[0]
    else:saved[0][0][fault]*=1.000001
    def never(*args):raise AssertionError('재생 불일치를 새 적분으로 숨기면 안 됨')
    with pytest.raises(ValueError,match='호출 순서/T/ne'):
        T.solve(never,first,adiabatic,policy='physical_stored',max_evaluations=1,replay=saved)


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
