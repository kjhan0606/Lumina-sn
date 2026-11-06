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


def test_actual_new_field_secant_is_opt_in_and_still_rechecks_heat():
    first=case(np.array([2.,3.])*1.2,[5.,7.]);calls=[];directions=[]
    def evaluate(t,ne):
        value=case(t,ne,len(calls)+1);calls.append(value);return value
    result=T.solve(evaluate,first,adiabatic,policy='physical_stored',
                   max_evaluations=12,secant_after_accept=True,
                   on_direction=lambda *args:directions.append(args[-1]))
    assert result['secant_after_accept'] is True
    assert result['fresh_evaluations_used']==len(calls)==len(result['history'])
    assert result['fixed_field_numerical_converged']
    assert result['new_transport']==0 and not result['joint_convergence']
    assert directions[0]=='centered' and 'secant' in directions[1:]
    np.testing.assert_allclose(result['selected'][0]['temperature'],[2.,3.],rtol=1.e-10)


def test_v2_residual_tolerances_accept_base_without_new_material_callbacks():
    first=case(np.array([2.,3.])*1.00005,[5.,7.]);calls=[]
    result=T.solve(lambda t,ne:(calls.append(1),case(t,ne))[1],first,adiabatic,
        policy='physical_stored',max_evaluations=12,
        charge_tolerance=1.e-10,thermal_tolerance=1.e-3)
    assert result['fixed_field_numerical_converged']
    assert result['charge_scaled_max']<=1.e-10
    assert result['thermal_scaled_max']<=1.e-3
    assert result['evaluations_used']==0 and not calls
    assert result['selected'] is first
    assert result['new_transport']==0 and not result['joint_convergence']


def test_normal_response_root_option_survives_callback_adapter():
    def large_gross(t,ne):
        value=case(t,ne)
        value[1]['physical_stored_gross'][:]=1.e6
        return value
    first=large_gross(np.array([2.,3.])*1.02,np.array([5.,7.]))
    result=T.solve(large_gross,first,adiabatic,policy='physical_stored',
        max_evaluations=12,charge_tolerance=1.e-10,thermal_tolerance=1.e-3,
        root_log_step_tolerance=1.e-3,secant_after_accept=True)
    assert result['root_search_requested'] and result['fixed_field_root_converged']
    assert result['accepted_iterations']>0 and result['fresh_evaluations_used']>=9
    assert not result['joint_convergence'] and result['new_transport']==0
    np.testing.assert_allclose(result['selected'][0]['temperature'],[2.,3.],rtol=1.e-8)


def test_long_secant_root_continuation_reuses_secant_but_requires_fresh_centered_confirmation():
    target=np.array([2.,3.]);target_ne=np.array([5.,7.]);calls=[];directions=[]
    def state(t,ne,number=0):
        dt=np.log(t/target);dn=np.log(ne/target_ne)
        data=dict(temperature=np.array(t),ne=np.array(ne),population=np.ones((2,2)),
                  charge_residual=ne*(dn-.2*dt))
        heat=dict(physical_adiabatic=np.zeros(2),physical_stored_net=dt,
                  physical_stored_gross=np.full(2,2.),physical_internal_energy=np.ones(2))
        return data,heat,dict(number=number)
    first=state(target*np.exp([1.4,.28]),target_ne*np.exp([.28,-.14]))
    def evaluate(t,ne):
        value=state(t,ne,len(calls)+1);calls.append(value);return value
    result=T.solve(evaluate,first,adiabatic,policy='physical_stored',max_evaluations=48,
        charge_tolerance=1.e-10,thermal_tolerance=1.e-3,root_log_step_tolerance=1.e-3,
        secant_after_accept=True,on_direction=lambda *args:directions.append(args[-1]))
    assert result['fixed_field_root_converged']
    assert result['root_fresh_direction_max']<=1.e-3
    assert result['accepted_iterations']>1 and 4<len(calls)<=48
    assert directions[0]=='centered' and 'secant' in directions and directions[-1]=='centered'
    assert np.max(abs(result['selected'][0]['charge_residual']/result['selected'][0]['ne']))<=1.e-10
    assert result['thermal_physical_scaled_max']<=1.e-3
    assert result['new_transport']==0 and not result['joint_convergence']
    np.testing.assert_allclose(result['selected'][0]['temperature'],target,rtol=1.e-8)


@pytest.mark.parametrize('secant,root_check',[(False,True),(True,False)])
def test_48_evaluation_budget_requires_secant_and_fresh_root_check(secant,root_check):
    with pytest.raises(ValueError,match='secant root 최대48'):
        T.solve(lambda *a:pytest.fail('잘못된 48회 설정에서 응답 금지'),case([2.1,3.15],[5.,7.]),
            adiabatic,policy='physical_stored',max_evaluations=48,
            secant_after_accept=secant,root_log_step_tolerance=1.e-3 if root_check else None)


def test_four_probe_root_confirmation_exports_base_not_last_perturbation():
    from prepare_lumina_accepted_material import thermal_selection
    first=case(np.array([2.,3.])*np.exp(.0005),np.array([5.,7.]));order=[]
    first[2]['fixed_field_thermal_callback']=True
    def evaluate(t,ne):
        value=case(t,ne,len(order)+1)
        value[2]['fixed_field_thermal_callback']=True
        order.append(value[2]);return value
    result=T.solve(evaluate,first,adiabatic,policy='physical_stored',max_evaluations=4,
        charge_tolerance=1.e-10,thermal_tolerance=1.e-3,root_log_step_tolerance=1.e-3)
    assert result['fixed_field_root_converged'] and result['accepted_iterations']==0
    assert result['evaluations_used']==len(order)==4 and result['selected'] is first
    report=dict(temperature_block=result,thermal_evaluation_order=order,
        thermal_initial_response=first[2],selected_response=result['selected'][2])
    index,selected,kind=thermal_selection(report)
    assert index==0 and selected is first[2]
    assert kind=='FINITE_NEW_FIELD_TNE_CONVERGED_BASE'
    report['selected_response']=order[-1]
    with pytest.raises(ValueError,match='선택 결과'):
        thermal_selection(report)


def test_direction_only_resolves_temperature_sensitivity_inside_gross_tolerance():
    first=case(np.array([2.,3.])*1.2,[5.,7.]); saved=first[0]['temperature'].copy()
    first[1]['physical_stored_gross'][:]=1.e12
    calls=[]; directions=[]; phases=[]
    def evaluate(t,ne):
        calls.append(1); value=case(t,ne,len(calls))
        value[1]['physical_stored_gross'][:]=1.e12
        return value
    accepted=T.solve(evaluate,first,adiabatic,policy='physical_stored',max_evaluations=5,
        charge_tolerance=1.e-10,thermal_tolerance=1.e-3)
    assert accepted['fixed_field_numerical_converged'] and not calls
    result=T.linearize(evaluate,first,adiabatic,policy='physical_stored',
        on_evaluation=lambda value,entry:phases.append(entry['phase']),
        on_direction=lambda *args:directions.append(args))
    assert phases==['T+','T-','ne+','ne-'] and len(calls)==4
    assert result['evaluations_used']==result['fresh_evaluations_used']==4
    assert result['accepted_iterations']==0 and not result['material_published']
    assert 'selected' not in result and 'fixed_field_numerical_converged' not in result
    np.testing.assert_allclose(directions[0][3][:2],-np.log(1.2),rtol=1.e-10)
    np.testing.assert_array_equal(first[0]['temperature'],saved)
    assert result['linear_relative_residual']<1.e-12
    assert not result['joint_convergence'] and result['new_transport']==0


@pytest.mark.parametrize('limit',[0,3,5,True])
def test_direction_only_rejects_budget_before_material_callbacks(limit):
    first=case([2.,3.],[5.,7.])
    with pytest.raises(ValueError,match='실제평가4회'):
        T.linearize(lambda *a:pytest.fail('신규 평가 금지'),first,adiabatic,
                    policy='physical_stored',max_evaluations=limit)


def test_direction_only_requires_closed_charge_and_matching_probe_state():
    first=case([2.,3.],[5.,7.]); first[0]['charge_residual'][:]=1.
    with pytest.raises(ValueError,match='전하가 닫힌'):
        T.linearize(lambda *a:pytest.fail('신규 평가 금지'),first,adiabatic,policy='physical_stored')
    first=case([2.,3.],[5.,7.])
    def bad(t,ne):
        value=case(t,ne); value[0]['temperature'][0]*=1.01
        return value
    with pytest.raises(ValueError,match='요청 T/ne'):
        T.linearize(bad,first,adiabatic,policy='physical_stored')


def test_direction_only_preserves_base_and_never_publishes_after_probe_failure():
    first=case([2.,3.],[5.,7.]); before=first[0]['temperature'].copy(); calls=[]; published=[]
    def evaluate(t,ne):
        calls.append(1)
        if len(calls)==3:raise RuntimeError('합성 세 번째 응답 실패')
        return case(t,ne)
    with pytest.raises(RuntimeError,match='세 번째'):
        T.linearize(evaluate,first,adiabatic,policy='physical_stored',
            on_direction=lambda *a:published.append(a))
    assert len(calls)==3 and not published
    np.testing.assert_array_equal(first[0]['temperature'],before)


def test_already_converged_base_does_not_require_replaying_old_perturbations():
    first=case(np.array([2.,3.]),[5.,7.]);replayed=[]
    result=T.solve(lambda *args:pytest.fail('수렴된 기준에서 새 평가 불필요'),
        first,adiabatic,policy='physical_stored',max_evaluations=1,
        replay=(first,)*4,on_replay=replayed.append,
        charge_tolerance=1.e-10,thermal_tolerance=1.e-3)
    assert result['fixed_field_numerical_converged'] and result['selected'] is first
    assert result['evaluations_used']==result['fresh_evaluations_used']==0
    assert result['replayed_evaluations']==0 and not replayed
    assert not result['joint_convergence']


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


def test_normal_temperature_iteration_closes_charge_without_a_manual_replay_run():
    first, saved, ad, basis, payload=projection_case(); calls=[]; decisions=[]
    def evaluate(t,ne):
        value=payload(t,ne,len(calls)+1);calls.append(value);return value
    result=T.solve(evaluate,first,ad,policy='physical_stored',max_evaluations=8,
        charge_bounds=(np.full(3,1.e8),np.full(3,4.e8)),
        on_evaluation=lambda value,entry:decisions.append((value,entry.copy())))
    assert result['charge_projected_trials'] and result['accepted_iterations']==1
    assert result['charge_relative']<=1.e-10
    assert 6<=len(calls)<=8 and result['fresh_evaluations_used']==len(calls)
    assert len(result['history'])==len(calls)
    assert result['selected'] is not first and result['history'][-1]['accepted']
    assert result['history'][-1]['phase']=='charge_projected_line'
    assert all(np.array_equal(value[0]['temperature'],calls[4][0]['temperature']) for value in calls[5:])
    assert any(entry.get('accepted') is True and value is result['selected'] for value,entry in decisions)
    assert not result['joint_convergence']
    # Consume the producer's real mode key rather than a handcrafted alias.
    from prepare_lumina_accepted_material import thermal_selection
    order=[dict(value[2],fixed_field_thermal_callback=True) for value in calls]
    report=dict(temperature_block=result,thermal_evaluation_order=order,
        selected_response=dict(result['selected'][2],fixed_field_thermal_callback=True))
    index,selected,kind=thermal_selection(report)
    assert index==result['history'][-1]['evaluation']
    assert selected==order[index-1] and kind=='FINITE_NEW_FIELD_TNE_ACCEPTED_STEP'


def test_secant_after_charge_projection_uses_actual_closed_candidate():
    first, saved, ad, basis, payload=projection_case();calls=[];directions=[]
    def evaluate(t,ne):
        value=payload(t,ne,len(calls)+1);calls.append(value);return value
    result=T.solve(evaluate,first,ad,policy='physical_stored',max_evaluations=8,
        charge_bounds=(np.full(3,1.e8),np.full(3,4.e8)),secant_after_accept=True,
        on_direction=lambda *args:directions.append(args[-1]))
    assert directions==['centered','secant']
    assert result['accepted_iterations']==2 and result['fixed_field_numerical_converged']
    assert result['charge_relative']<=1.e-10 and result['secant_after_accept']
    assert len(calls)==result['fresh_evaluations_used']==len(result['history'])==8
    assert result['selected'] is calls[-1] and not result['joint_convergence']


@pytest.mark.parametrize('fault',['budget','charge','heat'])
def test_normal_projection_keeps_the_accepted_state_on_failure(fault):
    first, saved, ad, basis, payload=projection_case();calls=[]
    def evaluate(t,ne):
        value=payload(t,ne,len(calls)+1)
        if len(calls)>=4:
            if fault=='charge':value[0]['charge_residual']=np.full(3,1.e7)
            if fault=='heat':value[1]['physical_stored_net']+=.2
        calls.append(value);return value
    limit=5 if fault=='budget' else 8
    result=T.solve(evaluate,first,ad,policy='physical_stored',max_evaluations=limit,
                   charge_bounds=(np.full(3,1.e8),np.full(3,4.e8)))
    assert result['selected'] is first and result['accepted_iterations']==0
    assert not result['fixed_field_numerical_converged']
    assert len(calls)==limit and result['fresh_evaluations_used']==limit


def test_projected_iteration_resumes_its_exact_callbacks_without_repeating_a_temperature():
    first,saved,ad,basis,payload=projection_case();preserved=[]
    bounds=(np.full(3,1.e8),np.full(3,4.e8))
    def original(t,ne):preserved.append(payload(t,ne,len(preserved)+1));return preserved[-1]
    stopped=T.solve(original,first,ad,policy='physical_stored',max_evaluations=5,charge_bounds=bounds)
    assert len(preserved)==5 and stopped['accepted_iterations']==0
    fresh=[]
    def more(t,ne):
        np.testing.assert_array_equal(t,preserved[-1][0]['temperature'])
        fresh.append(payload(t,ne,len(preserved)+len(fresh)+1));return fresh[-1]
    continued=T.solve(more,first,ad,policy='physical_stored',max_evaluations=3,
                      charge_bounds=bounds,replay=preserved)
    assert 1<=len(fresh)<=3 and continued['accepted_iterations']==1
    assert continued['replayed_evaluations']==5 and continued['fresh_evaluations_used']==len(fresh)
    assert continued['selected'] is fresh[-1] and continued['charge_relative']<=1.e-10


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
