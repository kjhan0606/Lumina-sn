"""열보정 재개 입력의 신원·실제 물리 코드·완료 배열·호출 목록 시험."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import lumina_thermal_replay as R
from test_lumina_fixed_field_thermal import case


@pytest.fixture
def preserved(tmp_path):
    root=tmp_path/'run';root.mkdir();(tmp_path/'build').mkdir()
    libraries=[tmp_path/'build'/name for name in ('bridge.so','lines.so')]
    for path in libraries:path.write_bytes(path.name.encode())
    model=tmp_path/'MODEL';model.write_bytes(b'model')
    columns=np.array([[14,1,0],[14,2,0]])
    state=dict(columns=columns,superlevel=np.ones((2,2)))
    entries=[]
    for i in range(3):
        data,heat,_=case(np.array([2.04,3.06])*np.exp(i*1.e-4),[5.,7.])
        data['columns']=columns
        output=root/('evaluation-'+str(i+1));output.mkdir()
        np.savez(output/'response.npz',**data);np.savez(output/'balance.npz',**heat)
        entries.append(dict(root=str(output),response_sha256=R.B.sha(output/'response.npz'),
            balance_sha256=R.B.sha(output/'balance.npz'),fixed_field_thermal_callback=i>0))
    implementation={'lumina_coupled_material.py':'same'}
    inputs={str(p):R.B.sha(p) for p in [*libraries,model]}
    report=dict(state='INCOMPLETE_PRESERVED',field={'J':'F2'},source_identity={'material':'accepted'},
        thermal_implementation_sha256=implementation.copy(),input_sha256=inputs.copy(),
        thermal_initial_response=entries[0],thermal_evaluation_order=entries[1:],charge_history=[{'accepted':True}])
    args=dict(field=report['field'].copy(),source_identity=report['source_identity'].copy(),
        implementation=implementation,state=state,library=libraries[0],line_library=libraries[1],inputs=inputs.copy())
    return root,report,args


def load(preserved):
    root,report,args=preserved;path=root/'result.json';path.write_text(json.dumps(report))
    return R.load(path,R.B.sha(path),**args)


def test_completed_values_are_loaded_without_changing_source_files(preserved):
    root,report,args=preserved
    paths=[Path(e['root'])/name for e in [report['thermal_initial_response'],*report['thermal_evaluation_order']]
           for name in ('response.npz','balance.npz')]
    before={str(p):R.B.sha(p) for p in paths}
    result=load(preserved)
    assert len(result['replay'])==2 and result['first'][2]==report['thermal_initial_response']
    assert result['charge_history']==[{'accepted':True}]
    assert result['provenance']['new_material_evaluations']==0
    for path,digest in before.items():
        assert R.B.sha(Path(path))==digest and args['inputs'][path]==digest


def test_verified_parent_exposes_optional_bound_tables_without_new_evaluations(preserved):
    root, report, args = preserved
    receipt = dict(path=str(root/'temperature-bound-cache/table.json'), sha256='table-sha')
    report['temperature_bound_cache'] = dict(completed_bound_tables=[receipt])
    result = load(preserved)
    assert result['bound_temperature_tables'] == [receipt]
    assert result['provenance']['new_material_evaluations'] == 0
    args['source_identity'] = {'material':'wrong'}
    with pytest.raises(ValueError, match='장/물질'): load(preserved)


def projection_source(preserved):
    root,report,args=preserved
    template=Path(report['thermal_evaluation_order'][-1]['root'])
    for i in (4,5,6):
        output=root/('evaluation-'+str(i));output.mkdir()
        for name in ('response.npz','balance.npz'):(output/name).write_bytes((template/name).read_bytes())
        report['thermal_evaluation_order'].append(dict(root=str(output),
            response_sha256=R.B.sha(output/'response.npz'),balance_sha256=R.B.sha(output/'balance.npz'),
            fixed_field_thermal_callback=True))
    report['temperature_block']=dict(accepted_iterations=0,history=[
        dict(phase=p,accepted=False,alpha=1.) for p in ('T+','T-','ne+','ne-','line_search')])
    output=root/'thermal-direction-1';output.mkdir();matrix=output/'linearization.npz'
    with np.load(Path(report['thermal_initial_response']['root'])/'response.npz') as first:
        np.savez(matrix,temperature=first['temperature'],ne=first['ne'],jacobian=np.eye(4),
            scaled_residual=np.zeros(4),log_direction=np.zeros(4),charge_scale=first['ne'],thermal_scale=np.ones(2))
    direction=dict(root=str(output),linearization_sha256=R.B.sha(matrix),energy_policy='physical_stored',
        field=report['field'].copy(),source_identity=report['source_identity'].copy(),
        base_response=report['thermal_initial_response'])
    (output/'result.json').write_text(json.dumps(direction));report['thermal_directions']=[direction]
    args['project_charge']=True
    return matrix


def test_projection_reuses_verified_direction_and_actual_rejected_trial(preserved):
    matrix=projection_source(preserved);result=load(preserved)
    assert result['projection']['trial'] is result['replay'][-1]
    assert result['projection']['alpha']==1.
    np.testing.assert_array_equal(result['projection']['basis']['direction'],np.zeros(4))
    assert str(matrix) in preserved[2]['inputs']
    assert result['provenance']['previous_direction_or_acceptance_reused'] is True
    assert result['provenance']['previous_acceptance_reused'] is False


@pytest.mark.parametrize('fault',['accepted','prefix','matrix','field','policy'])
def test_projection_rejects_other_branch_or_changed_direction(preserved,fault):
    matrix=projection_source(preserved);root,report,args=preserved
    if fault=='accepted':report['temperature_block']['accepted_iterations']=1
    elif fault=='prefix':args['prefix']=4
    elif fault=='matrix':matrix.write_bytes(b'changed')
    else:
        direction=report['thermal_directions'][0]
        direction['field' if fault=='field' else 'energy_policy']={} if fault=='field' else 'native_stored'
        (Path(direction['root'])/'result.json').write_text(json.dumps(direction))
    with pytest.raises(ValueError,match='전하 보정'):load(preserved)


def test_stopped_writer_can_reuse_only_explicit_completed_prefix_without_claiming_completion(preserved, monkeypatch):
    _, report, args = preserved
    report.update(state='STARTED', execution=dict(hostname=R.socket.gethostname(), pid=12345))
    args['prefix'] = 2
    def missing(pid, signal):
        assert (pid, signal) == (12345, 0)
        raise ProcessLookupError
    monkeypatch.setattr(R.os, 'kill', missing)
    result = load(preserved)
    assert len(result['replay']) == 2
    assert result['provenance']['source_process_missing']
    assert result['provenance']['source_report_state'] == 'STARTED'
    assert not result['provenance']['source_completion_certified']


@pytest.mark.parametrize('fault', ['live', 'foreign', 'permission', 'no-prefix'])
def test_unproven_stopped_writer_cannot_bypass_replay_source_lifecycle(preserved, monkeypatch, fault):
    _, report, args = preserved
    report.update(state='STARTED', execution=dict(hostname=R.socket.gethostname(), pid=12345))
    args['prefix'] = 2
    if fault == 'foreign': report['execution']['hostname'] = 'another-host'
    if fault == 'no-prefix': args.pop('prefix')
    def observe(pid, signal):
        if fault == 'permission': raise PermissionError
        if fault != 'live': raise ProcessLookupError
    monkeypatch.setattr(R.os, 'kill', observe)
    with pytest.raises(ValueError, match='종료된 물질'): load(preserved)


@pytest.mark.parametrize('fault',['live','field','material','implementation','library','input',
    'response','heat','nonthermal','empty','columns','nuclei','pin'])
def test_wrong_or_corrupted_replay_is_never_used(preserved,fault):
    root,report,args=preserved
    if fault=='live':report['state']='STARTED'
    elif fault=='field':args['field']={'J':'old'}
    elif fault=='material':args['source_identity']={'material':'old'}
    elif fault=='implementation':args['implementation']={'lumina_coupled_material.py':'changed'}
    elif fault=='library':args['library'].write_bytes(b'changed')
    elif fault=='input':(root.parent/'MODEL').write_bytes(b'changed')
    elif fault in ('response','heat'):
        name='response.npz' if fault=='response' else 'balance.npz'
        (Path(report['thermal_evaluation_order'][0]['root'])/name).write_bytes(b'changed')
    elif fault=='nonthermal':report['thermal_evaluation_order'][0]['fixed_field_thermal_callback']=False
    elif fault=='empty':report['thermal_evaluation_order']=[]
    elif fault=='columns':args['state']['columns']=args['state']['columns'].copy();args['state']['columns'][0,0]=16
    elif fault=='nuclei':args['state']['superlevel']*=1.01
    else:
        load(preserved)
        with pytest.raises(ValueError,match='보고서 변경'):
            R.load(root/'result.json','0'*64,**args)
        return
    with pytest.raises(ValueError):load(preserved)


def native_ledgers(preserved):
    root,report,args=preserved
    for entry in [report['thermal_initial_response'],*report['thermal_evaluation_order']]:
        path=Path(entry['root'])/'balance.npz'
        with np.load(path) as f:heat={k:f[k] for k in f.files}
        heat.update(native_adiabatic=np.zeros(2),native_internal_energy=np.ones(2),
            native_instantaneous_scenario_net=heat['physical_stored_net']+.005,
            native_instantaneous_scenario_gross=heat['physical_stored_gross']+.005,
            stored_decay=np.zeros(2),instantaneous_scenario=np.full(2,.005))
        np.savez(path,**heat);entry['balance_sha256']=R.B.sha(path)
    args['policy']='native_instantaneous_scenario'


def test_changed_energy_policy_cannot_replay_old_direction_candidates(preserved):
    native_ledgers(preserved)
    with pytest.raises(ValueError,match='열방정식 정책 불일치'):load(preserved)


def test_explicit_initial_probe_prefix_can_reuse_material_data_not_old_direction(preserved):
    native_ledgers(preserved);preserved[2]['prefix']=2
    result=load(preserved)
    assert len(result['replay'])==2
    p=result['provenance']
    assert p['previous_energy_policy']=='physical_stored' and p['energy_policy']=='native_instantaneous_scenario'
    assert p['initial_probe_prefix']==2 and not p['previous_direction_or_acceptance_reused']


@pytest.mark.parametrize('prefix',[0,5,True,3])
def test_probe_prefix_is_initial_bounded_and_complete(preserved,prefix):
    native_ledgers(preserved);preserved[2]['prefix']=prefix
    with pytest.raises(ValueError):load(preserved)


def test_changed_deposition_profile_is_not_a_policy_only_reuse(preserved):
    native_ledgers(preserved);_,report,args=preserved;args['prefix']=2
    entry=report['thermal_evaluation_order'][0];path=Path(entry['root'])/'balance.npz'
    with np.load(path) as f:heat={k:f[k] for k in f.files}
    heat['instantaneous_scenario']*=1.1;np.savez(path,**heat);entry['balance_sha256']=R.B.sha(path)
    with pytest.raises(ValueError,match='침적열 프로필'):load(preserved)


def test_equal_total_nuclei_does_not_hide_changed_local_element_inventories(preserved):
    _,report,args=preserved
    args['state']['columns'][1,0]=28
    for entry in [report['thermal_initial_response'],*report['thermal_evaluation_order']]:
        path=Path(entry['root'])/'response.npz'
        with np.load(path) as f:data={k:f[k] for k in f.files}
        data['columns']=args['state']['columns'].copy()
        if entry is report['thermal_evaluation_order'][0]:
            data['population'][0]+=.1;data['population'][1]-=.1
        np.savez(path,**data);entry['response_sha256']=R.B.sha(path)
    with pytest.raises(ValueError,match='깊이별 원소/핵종'):load(preserved)
