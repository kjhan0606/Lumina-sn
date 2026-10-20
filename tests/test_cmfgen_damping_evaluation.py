"""작은 적용량·선택적 개선을 실제 결합 안정화나 물리 합격으로 바꾸지 않는다."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_cmfgen_damping_branch import OUTPUT_ROUNDOFF, decide, run


def frame(p=.02,q=.002,f=.02,ne=.004):
    return dict(ne_step=ne,elements={str(Z):dict(material=[p/10,p],
        conditional=[q/10,q],field=[f/10,f]) for Z in (14,16)})


def test_all_numeric_conditions_only_create_candidate_not_physical_acceptance():
    r = decide(frame(),frame(.01,.001,.01,.002),.001,.0123)
    assert r['numerical_adoption_conditions_observed']
    assert r['decision'] == 'EVALUATE_STABILIZATION_CANDIDATE'
    assert not r['automatically_adopted'] and not r['automatic_extension']
    assert not r['joint_convergence'] and not r['released_T']
    assert r['final_goal'] == r['physical_equivalence'] == 'NOT_JUDGED'


def test_smaller_ne_and_local_cell_do_not_override_worse_sulfur_distribution():
    old,new = frame(),frame(.01,.001,.01,.002)
    new['elements']['16']['material'][1] = .03
    r = decide(old,new,.001,.0123)
    assert not r['final_P_and_ne_all_no_worse']
    assert not r['numerical_adoption_conditions_observed']
    assert r['decision'] == 'MIXED_NOT_ADOPTED'


@pytest.mark.parametrize('component',['Si_p95','Si_max','S_p95','S_max','ne'])
def test_each_preregistered_P_or_ne_component_can_prevent_adoption(component):
    old,new = frame(),frame(.01,.001,.01,.002)
    if component == 'ne':new['ne_step'] = .005
    else:
        label,index = component.split('_')
        new['elements']['14' if label == 'Si' else '16']['material'][0 if index == 'p95' else 1] = .03
    assert not decide(old,new,.001,.0123)['numerical_adoption_conditions_observed']


@pytest.mark.parametrize('motion',[.0123,.02])
def test_equal_or_larger_same_cell_step_is_not_improvement(motion):
    r = decide(frame(),frame(.01,.001,.01,.002),motion,.0123)
    assert not r['local_SiV_depth10_step_below_preregistered']
    assert not r['numerical_adoption_conditions_observed']


def test_flat_or_one_ulp_change_is_not_beyond_output_precision():
    old = frame()
    new = deepcopy(old)
    new['ne_step'] = np.nextafter(old['ne_step'],0)
    r = decide(old,new,.001,.0123)
    assert r['final_P_and_ne_all_no_worse']
    assert not r['at_least_one_improvement_beyond_output_roundoff']
    assert not r['numerical_adoption_conditions_observed']
    assert OUTPUT_ROUNDOFF > np.finfo(float).eps


def test_all_Q_components_non_decreasing_with_one_increase_rejects_candidate():
    old,new = frame(),frame(.01,.002,.01,.002)
    new['elements']['16']['conditional'][0] *= 2
    r = decide(old,new,.001,.0123)
    assert r['generally_worse_Q'] and not r['numerical_adoption_conditions_observed']


def test_mixed_Q_or_F_remains_explicitly_mixed_not_automatic_adoption():
    for key in ('conditional','field'):
        old,new = frame(),frame(.01,.001,.01,.002)
        new['elements']['14'][key][0] = .03
        r = decide(old,new,.001,.0123)
        assert r['numerical_adoption_conditions_observed']
        assert r['decision'] == 'MIXED_REQUIRES_EVALUATION'
        assert not r['automatically_adopted']


@pytest.mark.parametrize('key',['material','conditional','field'])
@pytest.mark.parametrize('value',[np.nan,np.inf,-1.])
def test_nonfinite_or_negative_metrics_never_become_small_valid_values(key,value):
    old,new = frame(),frame(.01,.001,.01,.002)
    new['elements']['16'][key][0] = value
    with pytest.raises(ValueError):decide(old,new,.001,.0123)


@pytest.mark.parametrize('value',[np.nan,np.inf,-1.])
def test_invalid_ne_is_not_a_convergence_signal(value):
    with pytest.raises(ValueError):decide(frame(),frame(ne=value),.001,.0123)


def test_existing_output_is_preserved_before_any_source_reads(tmp_path):
    with pytest.raises(ValueError,match='기존'):
        run(Path('no'),Path('no'),Path('no'),Path('no'),Path('no'),'no',tmp_path)
