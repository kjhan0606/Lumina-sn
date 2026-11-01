"""허용 영역 안에서도 실제 T/전하/열을 함께 개선하는 선택 경로의 합성 시험."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import lumina_fixed_field_thermal as T
import lumina_thermal_refinement as R


def fixture():
    root = np.array([2., 3.]); base_T = root*np.exp([.2, .02]); base_ne = np.array([5., 7.])
    def evaluate(t, ne):
        dt = np.log(t/base_T)
        charge = base_ne*(np.exp(.3*dt+.8*dt**2)-ne/base_ne)
        net = 1.e-6*np.log(t/root)
        return (dict(temperature=np.array(t), ne=np.array(ne), population=np.ones((2, 2)),
                     charge_residual=charge),
                dict(physical_adiabatic=np.zeros(2), physical_stored_net=net,
                     physical_stored_gross=np.ones(2), physical_internal_energy=np.ones(2)),
                dict(SE_residual=0.))
    first = evaluate(base_T, base_ne); probes = []; saved = []
    def probe(t, ne):
        value = evaluate(t, ne); probes.append(value); return value
    ad = lambda t, ne, nuclei, u: np.zeros_like(t)
    T.linearize(probe, first, ad, policy='physical_stored', on_direction=lambda *a:saved.append(a))
    _, j, f, delta, c, h = saved[0]
    basis = dict(temperature=base_T, ne=base_ne, jacobian=j, scaled_residual=f,
                 log_direction=delta, charge_scale=c, thermal_scale=h)
    args = dict(first=first, probes=probes, adiabatic=ad, basis=basis,
                lower=np.ones(2), upper=np.full(2, 20.), policy='physical_stored')
    return evaluate, args


def test_explicit_refinement_moves_T_inside_existing_RE_tolerance_and_closes_charge():
    evaluate, args = fixture(); calls = []; decisions = []; proposals = []
    old = args['first'][0]['temperature'].copy()
    def actual(t, ne):
        value = evaluate(t, ne); calls.append(value); return value
    result = R.refine(actual, **args, on_evaluation=lambda v,e:decisions.append(e),
                      on_proposal=proposals.append)
    assert result['accepted_iterations'] == 1 and result['selected'] is not args['first']
    assert 2 <= len(calls) <= 6 and len(calls) == result['evaluations_used']
    assert result['charge_relative'] <= 1.e-10
    assert result['thermal_physical_scaled_max'] <= 1.e-3
    assert result['saved_centered_probes_reused'] == 4
    assert result['temperature_trials'] == 1 and proposals[-1]['accepted']
    assert all(np.array_equal(value[0]['temperature'], calls[0][0]['temperature']) for value in calls)
    assert result['history'][-1]['phase'] == 'charge_projected_line'
    assert not result['temperature_root_certified'] and not result['joint_convergence']
    assert result['new_transport'] == 0 and result['replayed_evaluations'] == 0
    np.testing.assert_array_equal(args['first'][0]['temperature'], old)
    assert result['proposals'][0]['actual_heat_max'] < result['history'][-1]['scaled_start_max']
    # 전달기는 실제 callback 순서와 수용된 상태만 소비해야 한다.
    from prepare_lumina_accepted_material import thermal_selection
    order = [dict(number=i, fixed_field_thermal_callback=True) for i in range(len(calls))]
    selected = next(i for i,v in enumerate(calls) if v is result['selected'])
    report = dict(temperature_block=result, thermal_evaluation_order=order, selected_response=order[selected])
    assert thermal_selection(report)[0] == selected+1


def test_RE_pass_alone_cannot_accept_an_actual_heat_worsening():
    evaluate, args = fixture(); calls = []
    def worse(t, ne):
        value = evaluate(t, ne); value[1]['physical_stored_net'][:] = 1.e-5
        calls.append(value); return value
    result = R.refine(worse, **args)
    assert result['selected'] is args['first'] and result['accepted_iterations'] == 0
    assert result['temperature_trials'] == 2 and len(calls) <= 12
    assert not result['fixed_field_numerical_converged']
    assert all(not p['accepted'] for p in result['proposals'])


def test_candidate_must_meet_RE_on_its_own_scale_not_the_base_scale():
    evaluate, args = fixture()
    def changed_scale(t, ne):
        value = evaluate(t, ne)
        value[1]['physical_stored_gross'][:] = 2*abs(value[1]['physical_stored_net'])+1.e-30
        return value
    result = R.refine(changed_scale, **args)
    assert result['accepted_iterations'] == 0 and result['selected'] is args['first']
    assert any(e.get('thermal_physical_scaled_max', 0) > 1.e-3 for e in result['history'])


def test_unclosed_baseline_is_rejected_before_reusing_direction():
    _, args = fixture(); args['first'][0]['charge_residual'][0] = 1.
    with pytest.raises(ValueError, match='기준 전하'):
        R.refine(lambda *a:pytest.fail('기준 전하 미폐합에서 새 평가 금지'), **args)


def test_backtracking_rechecks_actual_response_instead_of_accepting_linear_prediction():
    evaluate, args = fixture(); calls = []
    def nonlinear(t, ne):
        value = evaluate(t, ne)
        if np.max(abs(np.log(t/args['first'][0]['temperature']))) > .075:
            value[1]['physical_stored_net'][:] = 1.e-5
        calls.append(value); return value
    result = R.refine(nonlinear, **args)
    assert result['accepted_iterations'] == 1 and result['temperature_trials'] == 2
    assert [p['accepted'] for p in result['proposals']] == [False, True]
    assert result['proposals'][-1]['alpha'] == .5 and len(calls) <= 12
    assert result['charge_relative'] <= 1.e-10


def test_unclosed_charge_and_budget_exhaustion_preserve_base():
    evaluate, args = fixture(); calls = []
    def actual(t, ne):
        calls.append(1); return evaluate(t, ne)
    result = R.refine(actual, **args, max_evaluations=2)
    assert len(calls) == 2 and result['selected'] is args['first']
    assert result['accepted_iterations'] == 0
    assert not any(p['charge_solved'] for p in result['proposals'])


@pytest.mark.parametrize('key', ['temperature', 'ne', 'thermal_scale', 'jacobian', 'log_direction'])
def test_stale_basis_is_rejected_before_new_callbacks(key):
    _, args = fixture(); args['basis'][key] = np.array(args['basis'][key], copy=True)+1.e-6
    with pytest.raises(ValueError, match='불일치'):
        R.refine(lambda *a:pytest.fail('stale 방향의 실제평가 금지'), **args)


@pytest.mark.parametrize('fault', ['T', 'ne', 'nuclei', 'SE', 'missing_SE'])
def test_actual_callback_identity_and_SE_are_checked(fault):
    evaluate, args = fixture()
    def bad(t, ne):
        value = evaluate(t, ne)
        if fault == 'T': value[0]['temperature'][0] *= 1.01
        elif fault == 'ne': value[0]['ne'][0] *= 1.01
        elif fault == 'nuclei': value[0]['population'][0, 0] *= 1.1
        elif fault == 'SE': value[2]['SE_residual'] = 1.e-5
        else: value[2].pop('SE_residual')
        return value
    with pytest.raises(ValueError, match='callback'):
        R.refine(bad, **args)


@pytest.mark.parametrize('limit', [0, 1, 13, True])
def test_budget_is_checked_without_new_callbacks(limit):
    _, args = fixture()
    with pytest.raises(ValueError, match='한도'):
        R.refine(lambda *a:pytest.fail('잘못된 예산으로 실제평가 금지'), **args, max_evaluations=limit)


def test_no_root_or_material_is_published_after_actual_failure():
    _, args = fixture(); old = args['first'][0]['temperature'].copy()
    def fail(*a): raise RuntimeError('실제 응답 실패')
    with pytest.raises(RuntimeError, match='실제 응답 실패'): R.refine(fail, **args)
    np.testing.assert_array_equal(args['first'][0]['temperature'], old)
