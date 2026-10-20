"""기본 거부 규약을 보존하고 유한 양수성 부분-step 계약만 허용한다."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import cmfgen_coupling_contract as cc
import run_cmfgen_native_se_observation as obs
from test_cmfgen_native_se_observation import sample, save_witness


def test_positivity_decrease_is_not_the_old_growth_cap_or_SE_convergence():
    correction = np.array([[1.0108, .5], [-16.317, 0], [.03, .01], [0, 0]])
    r = cc.classify_step(correction, .98936, False)
    assert r['allowed'] and r['reason'] == 'POSITIVITY_DECREASE_LIMIT_WITH_GROWTH_CAP_EXCLUDED'
    assert r['maximum_scaled_variable_growth_upper'] < 18
    assert r['physical_SE_convergence'] == 'NOT_JUDGED'


def test_full_step_stays_full_step_but_growth_limit_is_never_hidden():
    a = np.zeros((4, 2))
    assert cc.classify_step(a, 1., True)['allowed']
    a[0, 0] = -1e6
    assert not cc.classify_step(a, .01, False)['allowed']
    assert not cc.classify_step(a, 1., True)['allowed']


@pytest.mark.parametrize('minimum,full', [(None, True), (0, False), (2, False), (1, False), (.5, False)])
def test_unknown_partial_cause_and_missing_scale_never_gain_permission(minimum, full):
    assert not cc.classify_step(np.zeros((4, 2)), minimum, full)['allowed']


def test_nonfinite_correction_is_not_permission():
    assert not cc.classify_step(np.full((4, 2), np.nan), .99, False)['allowed']


def test_ne_growth_limiter_is_not_misclassified_as_population_positivity():
    a = np.zeros((4, 2))
    a[0, 0] = 1.0108
    a[-2, 0] = -1e6
    assert not cc.classify_step(a, .98936, False)['allowed']


def test_nonzero_T_correction_is_not_a_fixed_T_permission():
    a = np.zeros((4, 2))
    a[0, 0] = 1.0108
    a[-1, 0] = .2
    assert cc.classify_step(a, .98936, False)['reason'] == 'FIXED_T_CORRECTION_NOT_ZERO'


def policy(tmp_path, initial=67):
    return dict(initial_record=initial, purpose_id=cc.PURPOSE if initial==67 else cc.FOLLOWUP_PURPOSE,
        allowed_source_records=[initial, initial+1], initial_source=str(tmp_path/'original/material'),
        initial_runtime_sha256='initial-runtime', initial_next_outer_eligible_unchanged=initial==69,
        observation68_checkpoint_forbidden=str(tmp_path/'observation/evaluation'), _sha256='plan')


def test_initial67_permission_does_not_mutate_old_nexteligible(tmp_path, monkeypatch):
    c = policy(tmp_path)
    runtime = dict(next_outer_eligible=False)
    before = deepcopy(runtime)
    monkeypatch.setattr(cc, 'sha', lambda _: 'initial-runtime')
    monkeypatch.setattr(cc, 'step_evidence', lambda *_: dict(allowed=True))
    assert cc.authorize_source(Path(c['initial_source']), 67, {}, runtime, c)['allowed']
    assert runtime == before
    with pytest.raises(ValueError, match='원래'):
        cc.authorize_source(tmp_path/'another/material', 67, {}, runtime, c)
    with pytest.raises(ValueError, match='메타데이터'):
        cc.authorize_source(Path(c['initial_source']), 67, {}, dict(next_outer_eligible=True), c)


def test_observation68_and_any_third_outer_are_rejected_before_file_reads(tmp_path):
    c = policy(tmp_path)
    for source, record in ((Path(c['observation68_checkpoint_forbidden']), 68), (tmp_path/'material', 69)):
        with pytest.raises(ValueError, match='금지'):
            cc.authorize_source(source, record, {}, {}, c)


@pytest.mark.parametrize('initial', [67, 69])
def test_first_new_state_requires_actual_parent_plan_and_selection(tmp_path, monkeypatch, initial):
    c = policy(tmp_path, initial)
    source = tmp_path/f'new{initial+1}/material'
    inputs = dict(coupling_contract_sha256='plan', coupling_purpose=c['purpose_id'],
        source=c['initial_source'], source_state=dict(record=initial))
    runtime = dict(coupling_next_input_permitted=True)
    monkeypatch.setattr(cc, 'step_evidence', lambda *_: dict(allowed=True))
    with pytest.raises(ValueError, match='묶음 판단'):
        cc.authorize_source(source, initial+1, inputs, runtime, c)
    selection = tmp_path/'selection.json'
    selection.write_text(json.dumps(dict(decision=f'CONTINUE{initial+2}_WITHIN_LIMIT', coupling_contract_sha256='plan',
        selected_material=str(source), input_sha256={})))
    assert cc.authorize_source(source, initial+1, inputs, runtime, c, selection)['allowed']
    changed = deepcopy(inputs)
    changed['source'] = c['observation68_checkpoint_forbidden']
    with pytest.raises(ValueError, match='계보'):
        cc.authorize_source(source, initial+1, changed, runtime, c, selection)
    changed=deepcopy(inputs);changed['source_state']['record']=initial-2
    with pytest.raises(ValueError, match='계보'):
        cc.authorize_source(source, initial+1, changed, runtime, c, selection)
    changed=deepcopy(inputs);changed['coupling_purpose']='wrong-purpose'
    with pytest.raises(ValueError, match='계보'):
        cc.authorize_source(source, initial+1, changed, runtime, c, selection)
    selection.write_text(json.dumps(dict(decision=f'SKIP{initial+2}_JOINT_DETERIORATION')))
    with pytest.raises(ValueError, match='선택'):
        cc.authorize_source(source, initial+1, inputs, runtime, c, selection)


def test_initial69_keeps_its_existing_full_step_metadata(tmp_path, monkeypatch):
    c=policy(tmp_path,69);runtime=dict(next_outer_eligible=True)
    monkeypatch.setattr(cc,'sha',lambda _: 'initial-runtime')
    monkeypatch.setattr(cc,'step_evidence',lambda *_: dict(allowed=True))
    assert cc.authorize_source(Path(c['initial_source']),69,{},runtime,c)['allowed']
    assert runtime==dict(next_outer_eligible=True)
    with pytest.raises(ValueError,match='메타데이터'):
        cc.authorize_source(Path(c['initial_source']),69,{},dict(next_outer_eligible=False),c)
    for record in (68,71,72):
        with pytest.raises(ValueError,match='금지'):
            cc.authorize_source(Path(c['initial_source']),record,{},{},c)


def loadable_contract(tmp_path, purpose):
    c=json.loads((Path(__file__).resolve().parents[1]/'docs'/f'{purpose}.json').read_text())
    evidence=tmp_path/'noninterference.json'
    evidence.write_text(json.dumps(dict(evaluation_complete=True,source_unchanged=True,
        noninterference='REPRODUCED',native_executions=2,RT_executions=0,outer_executions=0)))
    binary=tmp_path/'C1';binary.write_bytes(b'isolated fixture')
    c.update(noninterference_evidence=str(evidence),noninterference_evidence_sha256=cc.sha(evidence),
        C_binary=str(binary),C_binary_sha256=cc.sha(binary))
    return c,tmp_path/'contract.json'


@pytest.mark.parametrize('purpose,initial',[(cc.PURPOSE,67),(cc.FOLLOWUP_PURPOSE,69)])
def test_only_registered_contracts_load_with_exact_finite_ranges(tmp_path,purpose,initial):
    c,path=loadable_contract(tmp_path,purpose);path.write_text(json.dumps(c))
    result=cc.load_contract(path,cc.sha(path))
    assert result['initial_record']==initial and result['allowed_source_records']==[initial,initial+1]
    assert result['_sha256']==cc.sha(path) and not result['automatic_extension']


@pytest.mark.parametrize('change', ['purpose','initial','sources','third','threshold','reservation','acceptance'])
def test_followup_cannot_extend_or_weaken_its_registered_observation(tmp_path,change):
    c,path=loadable_contract(tmp_path,cc.FOLLOWUP_PURPOSE)
    if change=='purpose':c['purpose_id']='CMFGEN_COUPLING_72_73_UNREGISTERED'
    elif change=='initial':c['initial_record']=71
    elif change=='sources':c['allowed_source_records']=[69,70,71]
    elif change=='third':c['maximum_native_executions']=3
    elif change=='threshold':c['quantitative_observation']['each_new_record_ne_max_relative_step_below']=.01
    elif change=='reservation':c['maximum_parent_reservation_seconds']=10800
    elif change=='acceptance':c['quantitative_observation']['physical_equivalence_acceptance']=True
    path.write_text(json.dumps(c))
    with pytest.raises(ValueError,match='범위|목표'):
        cc.load_contract(path,cc.sha(path))


def test_claim_namespaces_separate_old_and_new_contracts(tmp_path,monkeypatch):
    base=tmp_path/'claims';base.mkdir()
    original_path=cc.Path
    monkeypatch.setattr(cc,'Path',lambda p: base if p=='/gpfs/kjhan/cmfgen_runs' else original_path(p))
    old=cc.claim_generation(policy(tmp_path),67,tmp_path/'old-run')
    new=cc.claim_generation(policy(tmp_path,69),69,tmp_path/'new-run')
    assert Path(old).parent.name=='lumina_'+cc.PURPOSE+'_claims'
    assert Path(new).parent.name=='lumina_'+cc.FOLLOWUP_PURPOSE+'_claims'
    assert Path(new).name=='70.json'
    with pytest.raises(FileExistsError):cc.claim_generation(policy(tmp_path,69),69,tmp_path/'retry')
    with pytest.raises(ValueError):cc.claim_generation(policy(tmp_path,69),71,tmp_path/'third')


def test_changed_contract_identity_is_rejected_before_json(tmp_path):
    path = tmp_path/'plan.json'
    path.write_text('not json')
    with pytest.raises(ValueError, match='신원'):
        cc.load_contract(path, 'wrong')


def test_third_run_budget_or_temperature_release_cannot_be_authorized(tmp_path):
    base = json.loads((Path(__file__).resolve().parents[1]/'docs/CMFGEN_COUPLING_68_69_2026-10-20.json').read_text())
    for key, value in (('maximum_outer_executions', 3), ('T_release', True), ('allowed_source_records', [67, 68, 69])):
        c = deepcopy(base)
        c[key] = value
        path = tmp_path/'plan.json'
        path.write_text(json.dumps(c))
        with pytest.raises(ValueError, match='범위'):
            cc.load_contract(path, cc.sha(path))


def test_each_generation_can_be_claimed_once_and_only_two_generations(tmp_path):
    c = policy(tmp_path)
    claims = tmp_path/'claims'
    first = cc.claim_generation(c, 67, tmp_path/'run1', claims_root=claims)
    assert json.loads(Path(first).read_text())['new_record'] == 68
    with pytest.raises(FileExistsError):
        cc.claim_generation(c, 67, tmp_path/'run2', claims_root=claims)
    cc.claim_generation(c, 68, tmp_path/'run3', claims_root=claims)
    with pytest.raises(ValueError):
        cc.claim_generation(c, 69, tmp_path/'run4', claims_root=claims)
    assert sorted(p.name for p in claims.iterdir()) == ['68.json', '69.json']


def test_same_capture_can_explicitly_name_real_new_field_generation(tmp_path, monkeypatch):
    fields = sample()
    save_witness(tmp_path/'NATIVE_SE_BOUND_000068.bin', fields)
    (tmp_path/'seq_logs').mkdir()
    monkeypatch.setattr(obs, 'read_variables', lambda *_: [
        SimpleNamespace(index=i, species='Si', ion='SiIII', superlevel=i, charge=1) for i in (1, 2)])
    result = obs.assess(tmp_path, fields['pops'], 67, radiation_produced_from_record=67)
    assert result['radiation_record'] == 68 and result['radiation_produced_from_record'] == 67
    assert result['phase']['parent'] == 67 and not result['RE_certified']
