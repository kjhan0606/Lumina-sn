"""사전 중단 규칙과2주기 후보가 최종 물리 합격으로 바뀌지 않도록 확인한다."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_cmfgen_feedback_66_67 import decide,cycle_candidates,run


def frame(record,value=1.,ne=1.,full=True):
    return dict(record=record,ne_step=ne,full_step_premise=full,
        elements={str(Z):dict(field=[value,value],material=[value,value],conditional=[value,value]) for Z in (14,16)})


def test_joint_deterioration_skips67_without_claiming_goal_completion():
    r=decide([frame(65),frame(66,2.,2.)])
    assert r['decision']=='SKIP67_JOINT_DETERIORATION'
    assert r['final_goal']=='NOT_JUDGED' and not r['joint_convergence']


@pytest.mark.parametrize('component',['field','material','conditional','ne'])
def test_one_improvement_is_mixed_not_joint_deterioration(component):
    b=frame(66,2.,2.)
    if component=='ne':b['ne_step']=.5
    else:b['elements']['14'][component][0]=.5
    assert decide([frame(65),b])['decision']=='CONTINUE67_WITHIN_LIMIT'


def test_only_one_elements_deterioration_or_exact_plateau_does_not_trigger_joint_rule():
    b=frame(66,2.,2.);b['elements']['16']=deepcopy(frame(65,.5)['elements']['16'])
    assert decide([frame(65),b])['decision']=='CONTINUE67_WITHIN_LIMIT'
    assert decide([frame(65),frame(66)])['decision']=='CONTINUE67_WITHIN_LIMIT'


def test_changed_or_missing_full_step_premise_stops_but_keeps_completed66():
    b=frame(66,full=False)
    r=decide([frame(65),b])
    assert r['decision']=='STOP_BEFORE67_FULL_STEP_UNPROVEN' and r['completed66_preserved']


def test_baseline_wait_and_two_outer_limit_never_auto_extend():
    assert decide([frame(65)])['decision']=='WAIT_FOR_COMPLETED66'
    r=decide([frame(65),frame(66,.5),frame(67,.1)])
    assert r['decision']=='LIMIT_REACHED_EVALUATE' and not r['automatic_extension']
    with pytest.raises(ValueError):decide([frame(65),frame(67)])
    with pytest.raises(ValueError):decide([frame(65),frame(66),frame(67),frame(68)])


def test_cycle_candidates_compare_signed_same_cell_changes_without_tiny_value_floor():
    a=np.array([0.,.3,0.]);b=np.array([.1,.4,1e-300]);c=np.array([.001,.5,1e-302])
    r=cycle_candidates(a,b,c)
    assert r['sign_reversals']==2 and r['two_step_closer_than_both_one_steps']==2
    assert r['maximum_cycle_candidate_amplitude']==pytest.approx(.099)
    assert r['maximum_candidate_index_0based']==[0]
    assert '증명' in r['interpretation']


def test_existing_output_is_preserved_before_any_input_reads(tmp_path):
    with pytest.raises(ValueError,match='기존'):run([],tmp_path)


def test_new_coupling_range_is_opt_in_and_keeps_full_step_flag_honest():
    a,b=frame(67),frame(68,.5,.5,full=False)
    with pytest.raises(ValueError):decide([a,b])
    b['coupling_step_permitted']=True
    result=decide([a,b],coupling=True)
    assert result['decision']=='CONTINUE69_WITHIN_LIMIT'
    assert not b['full_step_premise'] and not result['joint_convergence']
    b['coupling_step_permitted']=False
    assert decide([a,b],coupling=True)['decision']=='STOP_BEFORE69_STEP_CAUSE_UNPROVEN'


def test_new_coupling_joint_worsening_and_final_limit_are_not_auto_extended():
    a,b,c=frame(67),frame(68,2.,2.),frame(69,.5,.5)
    b['coupling_step_permitted']=True
    assert decide([a,b],coupling=True)['decision']=='SKIP69_JOINT_DETERIORATION'
    result=decide([a,b,c],coupling=True)
    assert result['decision']=='LIMIT_REACHED_EVALUATE' and not result['automatic_extension']
    with pytest.raises(ValueError):decide([a,b,c,frame(70)],coupling=True)


def followup(record,value=1.,ne=.010741041305838639,full=True):
    r=frame(record,value,ne,full)
    r['coupling_step_permitted']=True
    return r


def test_followup_requires_explicit_registered_69_baseline():
    a,b=followup(69),followup(70)
    with pytest.raises(ValueError):decide([a,b],coupling=True)
    with pytest.raises(ValueError):decide([a,b],coupling_initial_record=69)
    assert decide([a],coupling=True,coupling_initial_record=69)['decision']=='WAIT_FOR_COMPLETED70'
    with pytest.raises(ValueError):decide([followup(71)],coupling=True,coupling_initial_record=71)


def test_followup_early_stop_ignores_Q_improvement_and_keeps_old_rule():
    a,b=followup(69),followup(70,2.)
    for e in b['elements'].values():e['conditional']=[.1,.1]
    r=decide([a,b],coupling=True,coupling_initial_record=69)
    assert r['decision']=='SKIP71_JOINT_DETERIORATION'
    assert not r['conditional_used_for_early_stop'] and r['final_goal']=='NOT_JUDGED'
    a['record']=67;b['record']=68
    assert decide([a,b],coupling=True)['decision']=='CONTINUE69_WITHIN_LIMIT'


def test_followup_some_global_increase_counts_even_if_other_element_is_flat():
    a,b=followup(69),followup(70)
    b['elements']['14']['field'][0]=2.
    assert decide([a,b],coupling=True,coupling_initial_record=69)['decision']=='SKIP71_JOINT_DETERIORATION'


@pytest.mark.parametrize('improvement',['ne','field','material','flat'])
def test_followup_mixed_or_exact_plateau_is_not_joint_worsening(improvement):
    a,b=followup(69),followup(70,2.)
    if improvement=='ne':b['ne_step']=np.nextafter(a['ne_step'],-np.inf)
    elif improvement=='flat':b=followup(70)
    else:b['elements']['16'][improvement][1]=.5
    assert decide([a,b],coupling=True,coupling_initial_record=69)['decision']=='CONTINUE71_WITHIN_LIMIT'


def test_followup_partial_step_permission_stays_distinct_from_full_step():
    a,b=followup(69),followup(70,.5,full=False)
    assert decide([a,b],coupling=True,coupling_initial_record=69)['decision']=='CONTINUE71_WITHIN_LIMIT'
    assert not b['full_step_premise']
    b['coupling_step_permitted']=False
    r=decide([a,b],coupling=True,coupling_initial_record=69)
    assert r['decision']=='STOP_BEFORE71_STEP_CAUSE_UNPROVEN' and r['completed70_preserved']


@pytest.mark.parametrize('failure',[None,'ne70','ne71','FP_equal','FP_larger'])
def test_followup_observation_uses_both_ne_steps_and_all_FP_without_physical_PASS(failure):
    frames=[followup(69),followup(70,.7,.004),followup(71,.5,.003)]
    if failure=='ne70':frames[1]['ne_step']=.005
    elif failure=='ne71':frames[2]['ne_step']=.005
    elif failure=='FP_equal':frames[2]['elements']['16']['material'][1]=1.
    elif failure=='FP_larger':frames[2]['elements']['14']['field'][0]=2.
    r=decide(frames,coupling=True,coupling_initial_record=69)
    assert r['decision']=='LIMIT_REACHED_EVALUATE' and not r['automatic_extension']
    assert r['attenuation_observation']['objective_observed']==(failure is None)
    assert not r['attenuation_observation']['physical_equivalence_acceptance']
    assert r['final_goal']=='NOT_JUDGED' and not r['joint_convergence']
    with pytest.raises(ValueError):decide(frames+[followup(72)],coupling=True,coupling_initial_record=69)


@pytest.mark.parametrize('bad',['ne','field','negative'])
def test_followup_nonfinite_or_negative_metrics_do_not_authorize_more_work(bad):
    a,b=followup(69),followup(70)
    if bad=='ne':b['ne_step']=float('nan')
    elif bad=='field':b['elements']['14']['field'][0]=float('inf')
    else:b['elements']['16']['material'][0]=-1.
    with pytest.raises(ValueError):decide([a,b],coupling=True,coupling_initial_record=69)
