"""새 장의 소비 계약·좌표별 비교 시험. 실제 BF/SE/RT 생산은 없다."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import lumina_new_field_response as N
from test_lumina_resident_feedback import scenario
import lumina_resident_feedback as F


@pytest.fixture
def manual_resources(monkeypatch):
    monkeypatch.setenv('LUMINA_RESPONSE_EXECUTOR', 'manual-lageunha')
    monkeypatch.setenv('OMP_NUM_THREADS', '16')
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    monkeypatch.delenv('SLURM_CPUS_PER_TASK', raising=False)
    monkeypatch.delenv('SLURM_MEM_PER_NODE', raising=False)
    monkeypatch.setattr(N.socket, 'gethostname', lambda: 'LagEunha')
    monkeypatch.setattr(N.os, 'sched_getaffinity', lambda _: set(range(16, 32)))
    monkeypatch.setattr(N.resource, 'getrlimit', lambda _: (32*1024**3, N.resource.RLIM_INFINITY))


def test_manual_resources_use_real_limits_not_fake_slurm(manual_resources):
    result = N.execution_resources(13800)
    assert result['executor'] == 'manual-lageunha'
    assert result['cpu_affinity'] == list(range(16, 32))
    assert result['address_space_limit_bytes'] == 32*1024**3
    assert 'job_id' not in result


@pytest.mark.parametrize('fault', ['host', 'slurm_job', 'affinity', 'unlimited', 'too_large', 'threads', 'time', 'mode'])
def test_manual_resource_mismatch_is_rejected(manual_resources, monkeypatch, fault):
    seconds = 13800
    if fault == 'host': monkeypatch.setattr(N.socket, 'gethostname', lambda: 'syntax')
    elif fault == 'slurm_job': monkeypatch.setenv('SLURM_JOB_ID', '123')
    elif fault == 'affinity': monkeypatch.setattr(N.os, 'sched_getaffinity', lambda _: set(range(17)))
    elif fault in ('unlimited', 'too_large'):
        limit = N.resource.RLIM_INFINITY if fault == 'unlimited' else 33*1024**3
        monkeypatch.setattr(N.resource, 'getrlimit', lambda _: (limit, limit))
    elif fault == 'threads': monkeypatch.setenv('OMP_NUM_THREADS', '32')
    elif fault == 'time': seconds = 42601
    else: monkeypatch.setenv('LUMINA_RESPONSE_EXECUTOR', 'other')
    with pytest.raises(ValueError): N.execution_resources(seconds)


def test_existing_slurm_resources_remain_supported(monkeypatch):
    monkeypatch.delenv('LUMINA_RESPONSE_EXECUTOR', raising=False)
    monkeypatch.setenv('OMP_NUM_THREADS', '16')
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '16')
    monkeypatch.setenv('SLURM_MEM_PER_NODE', '32768')
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    assert N.execution_resources(13800)['job_id'] == '123'
    with pytest.raises(ValueError): N.execution_resources(42600)
    monkeypatch.setenv('SLURM_MEM_PER_NODE', '65536')
    with pytest.raises(ValueError): N.execution_resources(13800)


def test_long_thermal_requires_real_slurm_time_and_temperature_evaluations(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setenv('LUMINA_RESPONSE_EXECUTOR', 'slurm')
    monkeypatch.setenv('LUMINA_RESPONSE_LONG_THERMAL', '1')
    monkeypatch.setenv('OMP_NUM_THREADS', '16')
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '16')
    monkeypatch.setenv('SLURM_MEM_PER_NODE', '32768')
    monkeypatch.setenv('SLURM_JOB_ID', '456')
    shown={'value': 'JobId=456 JobState=RUNNING TimeLimit=1-06:00:00 NumCPUs=16'}
    def show(command, **kwargs):
        assert command == ['scontrol', 'show', 'job', '456', '-o']
        assert kwargs['check'] and kwargs['timeout'] == 10
        return SimpleNamespace(stdout=shown['value'])
    monkeypatch.setattr(N.subprocess, 'run', show)
    result=N.execution_resources(106200, 12)
    assert result['long_thermal'] and result['seconds']==106200
    assert result['allocated_seconds']==108000
    with pytest.raises(ValueError, match='온도 평가'):
        N.execution_resources(106200, 0)
    with pytest.raises(ValueError, match='시간 할당 부족'):
        shown['value']='JobId=456 JobState=RUNNING TimeLimit=1-05:00:00 NumCPUs=16'
        N.execution_resources(106200, 12)
    shown['value']='JobId=456 JobState=RUNNING TimeLimit=1-06:00:00 NumCPUs=16'
    with pytest.raises(ValueError): N.execution_resources(106201, 12)
    monkeypatch.setenv('LUMINA_RESPONSE_EXECUTOR', 'manual-lageunha')
    with pytest.raises(ValueError, match='Slurm 경로'):
        N.execution_resources(106200, 12)


@pytest.mark.parametrize('reported', ['JobId=456 TimeLimit=UNLIMITED',
    'JobId=456 TimeLimit=1-06:00', 'JobId=456'])
def test_long_thermal_rejects_missing_or_ambiguous_allocation(monkeypatch, reported):
    from types import SimpleNamespace
    monkeypatch.setattr(N.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(stdout=reported))
    with pytest.raises(ValueError, match='할당 시간 확인 실패'):
        N.slurm_allocation_seconds('456')


def test_approved_manual_duration_keeps_cpu_memory_and_host_limits(manual_resources):
    result=N.execution_resources(42600)
    assert result['seconds']==42600 and result['CPUs']==16
    assert result['memory_limit_bytes']==32*1024**3


def test_temperature_cache_memory_is_reserved_without_claiming_measured_full_RSS():
    from lumina_temperature_response_cache import extra_memory_reserve
    args = (90, 1115020, 24667, 790493, 24542)
    base = N.memory_inventory(*args)
    cached = N.memory_inventory(*args, temperature_cache=True)
    difference = cached['known_allocation_and_reserve_bytes']-base['known_allocation_and_reserve_bytes']
    assert difference == extra_memory_reserve(24667, 90, 24542)
    assert cached['known_allocation_and_reserve_bytes'] < 32*1024**3
    assert cached['whole_process_peak_RSS_verified'] is False


@pytest.mark.parametrize('limit', [4, 13, True])
def test_temperature_mode_rejects_bad_budget_before_creating_output(manual_resources, limit):
    from types import SimpleNamespace
    # out 등 실행 인수가 없어도 온도 예산 검증에서 먼저 종료돼야 한다.
    with pytest.raises(ValueError, match='온도 보정'):
        N.run(SimpleNamespace(seconds=13800, temperature_evaluations=limit))


@pytest.mark.parametrize('path,digest,limit', [('result.json', None, 5),
    (None, 'sha', 5), ('result.json', 'sha', 0)])
def test_resume_requires_pin_pair_and_explicit_temperature_mode(manual_resources, path, digest, limit):
    from types import SimpleNamespace
    with pytest.raises(ValueError, match='완료 계수 재개'):
        N.run(SimpleNamespace(seconds=13800, temperature_evaluations=limit,
                              resume_result=path, resume_result_sha=digest))


@pytest.mark.parametrize('path,digest,seed', [('result.json',None,True),
    (None,'sha',True),('result.json','sha',False)])
def test_thermal_resume_requires_pin_pair_and_seed(manual_resources,path,digest,seed):
    from types import SimpleNamespace
    with pytest.raises(ValueError,match='열평가 재개'):
        N.run(SimpleNamespace(seconds=13800,temperature_evaluations=5,
            resume_thermal_result=path,resume_thermal_result_sha=digest,
            resume_result='seed' if seed else None,resume_result_sha='seed-sha' if seed else None))


def test_temperature_correction_rejects_invalid_explicit_equation_before_output(manual_resources):
    from types import SimpleNamespace
    with pytest.raises(ValueError,match='열방정식 정책'):
        N.run(SimpleNamespace(seconds=13800,temperature_evaluations=5,thermal_policy='unknown'))


@pytest.mark.parametrize('policy', N.thermal_replay.ENERGY_POLICIES)
def test_fixed_temperature_continuation_inherits_the_accepted_material_equation(policy):
    manifest = dict(thermal_energy_policy=policy)
    assert N.continued_energy_policy(manifest, None) == policy
    assert N.continued_energy_policy(manifest, None, required=True) == policy
    assert N.continued_energy_policy(manifest, policy) == policy
    assert N.continued_energy_policy({}, policy) == policy


def test_continuation_does_not_silently_change_a_known_thermal_equation():
    with pytest.raises(ValueError, match='상위 수용 물질'):
        N.continued_energy_policy(dict(thermal_energy_policy='native_instantaneous_scenario'), 'physical_stored')
    with pytest.raises(ValueError, match='정책'):
        N.continued_energy_policy(dict(thermal_energy_policy='unknown'), None)
    assert N.continued_energy_policy({}, None) is None
    with pytest.raises(ValueError, match='상위 물질'):
        N.continued_energy_policy({}, None, required=True)


@pytest.mark.parametrize('resume,prefix,limit',[(False,None,5),(True,4,4),(True,None,5)])
def test_projection_needs_whole_completed_source_and_bounded_ne_budget(manual_resources,resume,prefix,limit):
    from types import SimpleNamespace
    with pytest.raises(ValueError,match='고정 T 전하'):
        N.run(SimpleNamespace(seconds=1200,temperature_evaluations=limit,charge_projection=True,
            resume_result='seed' if resume else None,resume_result_sha='seed-sha' if resume else None,
            resume_thermal_result='report' if resume else None,resume_thermal_result_sha='sha' if resume else None,
            thermal_replay_prefix=prefix,thermal_policy='native_instantaneous_scenario'))


@pytest.mark.parametrize('prefix',[0,5,True])
def test_invalid_thermal_probe_prefix_is_rejected_before_output(manual_resources,prefix):
    from types import SimpleNamespace
    with pytest.raises(ValueError,match='초기 열섭동'):
        N.run(SimpleNamespace(seconds=13800,temperature_evaluations=5,thermal_policy='physical_stored',
            resume_result='seed',resume_result_sha='sha',resume_thermal_result='report',
            resume_thermal_result_sha='sha',thermal_replay_prefix=prefix))


def test_saved_direction_keeps_exact_linear_system_and_never_marks_candidate_accepted(tmp_path):
    data=dict(temperature=np.array([9000.,10000.]),ne=np.array([20.,30.]))
    metadata=dict(root='evaluation-1',response_sha256='response',balance_sha256='balance')
    matrix=np.diag([1.,2.,3.,4.]);residual=np.array([.01,-.01,.02,-.02])
    delta=np.linalg.solve(matrix,-residual);charge_scale=data['ne'];thermal_scale=np.array([1.,2.])
    record=N.preserve_thermal_direction(tmp_path,1,(data,{},metadata),matrix,residual,delta,
        charge_scale,thermal_scale,field={'J':'F2'},source_identity={'material':'accepted'},
        probes=[{'root':'evaluation-'+str(i)} for i in range(2,6)])
    root=tmp_path/'thermal-direction-1';path=root/'linearization.npz'
    assert record['linearization_sha256']==N.B.sha(path)
    assert json.loads((root/'result.json').read_text())==record
    with np.load(path) as saved:
        for key,value in dict(temperature=data['temperature'],ne=data['ne'],jacobian=matrix,
            scaled_residual=residual,log_direction=delta,charge_scale=charge_scale,
            thermal_scale=thermal_scale).items():np.testing.assert_array_equal(saved[key],value)
    assert record['base_response']==metadata and len(record['probes'])==4
    assert record['linearization_kind']=='centered' and record['centered_log_step']==1.e-4
    assert record['state']=='SAVED_DIRECTION_NOT_ACCEPTED_CANDIDATE'
    assert not record['joint_convergence'] and record['new_material_evaluations']==0
    with pytest.raises(FileExistsError):
        N.preserve_thermal_direction(tmp_path,1,(data,{},metadata),matrix,residual,delta,
            charge_scale,thermal_scale,field={},source_identity={},probes=[])
    secant=N.preserve_thermal_direction(tmp_path,2,(data,{},metadata),matrix,residual,delta,
        charge_scale,thermal_scale,field={'J':'F2'},source_identity={'material':'accepted'},
        probes=[],linearization_kind='secant')
    assert secant['linearization_kind']=='secant' and secant['centered_log_step'] is None
    assert secant['probes']==[] and secant['base_response']==metadata
    with pytest.raises(ValueError,match='선형화'):
        N.preserve_thermal_direction(tmp_path,3,(data,{},metadata),matrix,residual,delta,
            charge_scale,thermal_scale,field={},source_identity={},probes=[{}],
            linearization_kind='secant')


def test_trapezoid_preserves_original_nonuniform_grid():
    nu = np.array([1., 2., 4., 9.]); saved = nu.copy()
    w = N.trapezoid_weights(nu)
    np.testing.assert_array_equal(nu, saved)
    assert w.sum() == 8.
    assert np.dot(w, 2.*nu+3.) == 104.


@pytest.mark.parametrize('nu', [[1.], [1., 1.], [0., 1.], [2., 1.], [1., np.nan]])
def test_bad_frequency_contract_rejected(nu):
    with pytest.raises(ValueError): N.trapezoid_weights(nu)


def test_ion_comparison_has_all_nodes_and_preserves_absolute_values():
    columns = np.array([[14, 1, 0], [14, 1, 1], [14, 2, 0], [16, 1, 0], [16, 2, 0]])
    old = np.array([[1., 2.], [1., 2.], [8., 6.], [2., 3.], [8., 7.]])
    candidate = old.copy(); candidate[0] += 1.; candidate[2] -= 1.
    summary, rows = N.comparison(columns, old, old, candidate, np.array([20., 10.]))
    assert len(summary) == 4 and len(rows) == 8
    si = next(v for v in summary if v['Z'] == 14 and v['charge'] == 1)
    assert si['max_absolute_fraction_difference'] == pytest.approx(.1)
    r = next(v for v in rows if v['Z'] == 14 and v['charge'] == 1 and v['node'] == 0)
    assert r['CMFGEN70_fraction'] == .2 and r['new_J_response_fraction'] == .3
    assert r['radius_cm'] == 20.


def test_followup_memory_inventory_is_separate_from_coefficient_inventory():
    memory = N.memory_inventory(90, 1115020, 24667, 790493, 24542)
    assert 10*1024**3 < memory['known_allocation_and_reserve_bytes'] < 32*1024**3
    assert not memory['whole_process_peak_RSS_verified']


def connect(scenario, monkeypatch):
    args, calls, behavior = scenario
    source = F.open_prepared(None, None)[1]
    source.radius = np.array([3., 2., 1.]); source.frequency = np.arange(1., 9.)
    monkeypatch.setattr(N, 'open_prepared', F.open_prepared)
    return args, behavior


def test_actual_inputfield_accepts_only_last_complete_new_J(scenario, monkeypatch):
    args, behavior = connect(scenario, monkeypatch)
    record = F.run(**args); feedback = args['out']/'feedback.json'
    with N.open_field(args['accepted_input'], args['accepted_input_sha256'], feedback, N.B.sha(feedback)) as (_, source, field, last):
        assert last['sweep'] == 2 and source.ns == 3
        np.testing.assert_array_equal(field.values, np.full((3, 8), 2.))
    assert field.closed


def test_partial_second_uses_completed_first_not_partial_J(scenario, monkeypatch):
    args, behavior = connect(scenario, monkeypatch); behavior['fail'] = 'partial'
    F.run(**args); feedback = args['out']/'feedback.json'
    with N.open_field(args['accepted_input'], args['accepted_input_sha256'], feedback, N.B.sha(feedback)) as (_, _, field, last):
        assert last['sweep'] == 1
        np.testing.assert_array_equal(field.values, np.ones((3, 8)))


@pytest.mark.parametrize('fault', ['J', 'moments', 'record', 'material', 'no_complete'])
def test_changed_or_incomplete_field_is_never_used(scenario, monkeypatch, fault):
    args, behavior = connect(scenario, monkeypatch)
    F.run(**args); feedback = args['out']/'feedback.json'; data = json.loads(feedback.read_text())
    if fault in ('J', 'moments'):
        path = Path(data['last_complete'][fault]['path']); path.write_bytes(b'changed')
    elif fault == 'record':
        (args['out']/'F-2/resident-field.json').write_text('{}')
    elif fault == 'material': data['source_identity'] = dict(material='old70')
    else: data['last_complete'] = None
    feedback.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        with N.open_field(args['accepted_input'], args['accepted_input_sha256'], feedback, N.B.sha(feedback)):
            pass
