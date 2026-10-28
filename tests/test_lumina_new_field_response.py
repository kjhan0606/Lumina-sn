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
    elif fault == 'time': seconds = 14400
    else: monkeypatch.setenv('LUMINA_RESPONSE_EXECUTOR', 'other')
    with pytest.raises(ValueError): N.execution_resources(seconds)


def test_existing_slurm_resources_remain_supported(monkeypatch):
    monkeypatch.delenv('LUMINA_RESPONSE_EXECUTOR', raising=False)
    monkeypatch.setenv('OMP_NUM_THREADS', '16')
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '16')
    monkeypatch.setenv('SLURM_MEM_PER_NODE', '32768')
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    assert N.execution_resources(13800)['job_id'] == '123'
    monkeypatch.setenv('SLURM_MEM_PER_NODE', '65536')
    with pytest.raises(ValueError): N.execution_resources(13800)


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
