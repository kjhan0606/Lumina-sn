"""합성 완성장으로 실제 CLI 전달·부분 결과 보존을 시험한다. GPU 실행은 없다."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import lumina_resident_feedback as F
import lumina_material_trial_field as T


@pytest.fixture
def scenario(tmp_path, monkeypatch):
    root = tmp_path/'stage'; (root/'code/scripts').mkdir(parents=True)
    worker = root/'code/scripts/lumina_resident_run.py'; worker.write_text('# synthetic worker\n')
    (root/'stage.json').write_text(json.dumps(dict(files={
        'scripts/lumina_resident_run.py': F.sha(worker)})))
    for name in ('chain_cuda.so', 'geometry.so'): (root/name).write_bytes(b'synthetic')
    prep = tmp_path/'preparation.json'; prep.write_text('{}')
    identity = dict(role='ACCEPTED_NODAL_MATERIAL_NEW_COEFFICIENTS', material='synthetic')
    class Source:
        ns, nf = 3, 8
        radius = np.array([3., 2., 1.])
        frequency = np.arange(1., 9.)
        def __enter__(self): return self
        def __exit__(self, *args): pass
    source = Source(); source.identity = identity
    monkeypatch.setattr(F, 'open_prepared', lambda *a: ({}, source, None, None))
    for name, value in (('SLURM_JOB_ID', 'test'), ('SLURM_CPUS_PER_TASK', '1'),
                        ('SLURM_MEM_PER_NODE', '16384')):
        monkeypatch.setenv(name, value)
    clock = [0.]; monkeypatch.setattr(F, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    calls = []; behavior = dict(fail=None, cost=200., source_identity=identity)
    def launch(command, *, stdout, stderr, timeout):
        calls.append(command)
        index = len(calls); child = Path(command[2]); clock[0] += behavior['cost']
        stdout.write('worker output\n'); stderr.write('worker diagnostics\n')
        if behavior['fail'] == 'timeout': raise subprocess.TimeoutExpired(command, timeout)
        J = child/'J.npy'; moments = child/'moments.npy'
        np.save(J, np.full((3, 8), float(index))); np.save(moments, np.ones((3, 3, 8)))
        prior = 'ZERO_OWN_START' if '--input-j' not in command else dict(
            path=command[command.index('--input-j')+1],
            sha256=command[command.index('--input-j-sha256')+1])
        record = dict(schema='lumina-resident-fullsr-native90-v1', complete_original_grid=True,
            completed_actual_F_calls=1,
            original_layout=dict(frequencies=8, paths=1, physical_intervals=2,
                                 step_observations=3, boundary_observations=1),
            field_report=dict(complete=True, failed=False, physical_intervals=2,
                              observations=4, next_action=4, actions=4),
            source_identity=identity, identity=dict(input_plan_sha256=F.sha(prep), input_J=prior),
            J=dict(path=str(J), sha256=F.sha(J)), moments=dict(path=str(moments), sha256=F.sha(moments)))
        if index == 2 or behavior['fail'] == 'partial-first':
            fault = behavior['fail']
            if fault in ('partial', 'partial-first'): record['complete_original_grid'] = False
            if fault == 'hash': record['J']['sha256'] = '0'*64
            if fault == 'input': record['identity']['input_J'] = 'ZERO_OWN_START'
            if fault == 'source': record['source_identity'] = {'material': 'wrong'}
            if fault == 'missing': return SimpleNamespace(returncode=1)
            if fault == 'nan':
                np.save(J, np.full((3, 8), np.nan)); record['J']['sha256'] = F.sha(J)
        (child/'resident-field.json').write_text(json.dumps(record))
        return SimpleNamespace(returncode=1 if index == 2 and behavior['fail'] == 'exit' else 0)
    monkeypatch.setattr(F.subprocess, 'run', launch)
    kwargs = dict(staged_root=root, out=tmp_path/'feedback', cache=tmp_path/'cache',
                  accepted_input=prep, accepted_input_sha256=F.sha(prep), seconds=1200)
    return kwargs, calls, behavior


def seed_field(args, identity):
    directory = args['out'].parent/'seed'; directory.mkdir()
    target = directory/'J.npy'; np.save(target, np.full((3, 8), .5))
    record = dict(schema='lumina-resident-fullsr-native90-v1', complete_original_grid=True,
        completed_actual_F_calls=1, source_identity=identity,
        original_layout=dict(frequencies=8, paths=1, physical_intervals=2,
                             step_observations=3, boundary_observations=1),
        field_report=dict(complete=True, failed=False, physical_intervals=2,
                          observations=4, next_action=4, actions=4),
        J=dict(path=str(target), sha256=F.sha(target)))
    report = directory/'resident-field.json'; report.write_text(json.dumps(record))
    return target, report, record


def test_complete_same_material_seed_is_forwarded_to_first_new_sweep(scenario):
    args, calls, behavior = scenario
    target, report, previous = seed_field(args, behavior['source_identity'])
    result = F.run(**args, input_j=target, input_j_sha256=F.sha(target))
    assert len(calls) == 2 and result['last_complete']['sweep'] == 2
    assert result['initial_complete_J'] == previous['J']
    assert result['sweeps'][0]['input_J'] == previous['J']
    assert calls[0][calls[0].index('--input-j')+1] == str(target)
    assert result['sweeps'][0]['J_gap']['max_norm_relative'] == .5
    assert not result['scattering_convergence_certified'] and result['final_goal'] == 'NOT_MET'


def test_previous_material_seed_is_only_initial_trial_for_first_sweep(scenario, monkeypatch):
    args, calls, behavior = scenario
    behavior['source_identity']['epoch_seconds'] = 100.
    old_identity = dict(material='previous', epoch_seconds=100.)
    target, _, previous = seed_field(args, old_identity)
    prep = args['out'].parent/'previous-preparation.json'; prep.write_text('{}')
    class Source:
        ns, nf = 3, 8
        radius = np.array([3., 2., 1.]); frequency = np.arange(1., 9.)
        identity = old_identity
        def __enter__(self): return self
        def __exit__(self, *args): pass
    monkeypatch.setattr(T, 'open_prepared', lambda *a: ({}, Source(), None, None))
    result = F.run(**args, input_j=target, input_j_sha256=F.sha(target),
        initial_preparation=prep, initial_preparation_sha256=F.sha(prep))
    assert len(calls) == 2 and result['last_complete']['sweep'] == 2
    assert calls[0][calls[0].index('--initial-preparation')+1] == str(prep)
    assert calls[0][calls[0].index('--initial-preparation-sha256')+1] == F.sha(prep)
    assert '--initial-preparation' not in calls[1]  # 다음에는 이번 물질의 완료 J만 전달한다.
    assert result['source_identity'] == behavior['source_identity']
    assert result['initial_field_role']['origin_source_identity'] == old_identity
    assert result['initial_field_role']['accepted_field_for_target_material'] is False
    assert result['initial_complete_J'] == previous['J']
    assert result['material_updates'] == 0 and result['final_goal'] == 'NOT_MET'


@pytest.mark.parametrize('fault', ['hash', 'partial', 'source'])
def test_invalid_initial_seed_cannot_launch_transport(scenario, fault):
    args, calls, behavior = scenario
    target, report, previous = seed_field(args, behavior['source_identity'])
    digest = F.sha(target)
    if fault == 'hash': digest = '0'*64
    if fault == 'partial': previous['complete_original_grid'] = False
    if fault == 'source': previous['source_identity'] = {'material':'wrong'}
    report.write_text(json.dumps(previous))
    with pytest.raises(ValueError):
        F.run(**args, input_j=target, input_j_sha256=digest)
    assert not calls and not args['out'].exists()


def test_seed_path_and_digest_must_be_supplied_together(scenario):
    args, calls, behavior = scenario
    with pytest.raises(ValueError): F.run(**args, input_j=Path('unused'))
    with pytest.raises(ValueError): F.run(**args, input_j_sha256='0'*64)
    assert not calls


def test_failed_seeded_sweep_does_not_count_the_seed_as_a_new_completed_field(scenario):
    args, calls, behavior = scenario
    target, report, previous = seed_field(args, behavior['source_identity'])
    behavior['fail'] = 'partial-first'
    result = F.run(**args, input_j=target, input_j_sha256=F.sha(target))
    assert len(calls) == 1 and result['last_complete'] is None
    assert result['initial_complete_J'] == previous['J']
    assert result['status'] == 'PARTIAL_SWEEP_STOP'


def test_two_complete_fields_pass_exact_previous_J_without_checkpoint(scenario):
    args, calls, behavior = scenario
    result = F.run(**args)
    assert len(calls) == 2 and result['last_complete']['sweep'] == 2
    assert result['status'] == 'FINITE_FEEDBACK_RESPONSE_NOT_CONVERGENCE'
    assert not result['scattering_convergence_certified'] and result['final_goal'] == 'NOT_MET'
    assert result['material_updates'] == result['new_BF_BB_integrals'] == 0
    first_J = args['out']/'F-1/J.npy'
    assert calls[1][calls[1].index('--input-j')+1] == str(first_J)
    assert calls[1][calls[1].index('--input-j-sha256')+1] == F.sha(first_J)
    assert '--resume' not in calls[1] and '--input-j' not in calls[0]
    assert int(calls[1][calls[1].index('--seconds')+1]) < int(calls[0][calls[0].index('--seconds')+1])
    assert result['sweeps'][1]['J_gap']['max_norm_relative'] == .5
    assert 'J_gap' not in result['sweeps'][0]
    assert (args['out']/'F-1/stdout.log').read_text() == 'worker output\n'
    assert json.loads((args['out']/'feedback.json').read_text()) == result


@pytest.mark.parametrize('fault', ['partial', 'hash', 'input', 'source', 'missing', 'nan', 'exit'])
def test_failed_second_sweep_preserves_first_and_never_retries(scenario, fault):
    args, calls, behavior = scenario; behavior['fail'] = fault
    result = F.run(**args)
    assert len(calls) == 2 and result['last_complete']['sweep'] == 1
    assert result['status'] in ('PARTIAL_SWEEP_STOP', 'ERROR_PRESERVED')
    assert F.sha(result['last_complete']['J']['path']) == result['last_complete']['J']['sha256']
    assert not result['scattering_convergence_certified']


@pytest.mark.parametrize('fault', ['partial-first', 'timeout'])
def test_failed_first_sweep_cannot_start_second(scenario, fault):
    args, calls, behavior = scenario; behavior['fail'] = fault
    result = F.run(**args)
    assert len(calls) == 1 and result['last_complete'] is None
    assert result['status'] in ('PARTIAL_SWEEP_STOP', 'ERROR_PRESERVED')


def test_shared_time_admission_stops_with_last_complete(scenario):
    args, calls, behavior = scenario; behavior['cost'] = 500.
    result = F.run(**args)
    assert len(calls) == 1 and result['last_complete']['sweep'] == 1
    assert result['status'] == 'SHARED_TIME_RESERVE_STOP'


def test_existing_results_are_never_overwritten(scenario):
    args, calls, behavior = scenario; args['out'].mkdir()
    with pytest.raises(FileExistsError): F.run(**args)
    assert not calls


@pytest.mark.parametrize('key,value', [('seconds', 239), ('seconds', 43201),
    ('max_sweeps', 3), ('max_sweeps', True)])
def test_execution_limits_are_enforced(scenario, key, value):
    args, calls, behavior = scenario; args[key] = value
    with pytest.raises(ValueError): F.run(**args)
    assert not calls


@pytest.mark.parametrize('bad', ['dtype', 'shape', 'negative', 'nan'])
def test_field_gap_rejects_invalid_full_field(tmp_path, bad):
    a = tmp_path/'a.npy'; b = tmp_path/'b.npy'; x = np.ones((3, 8))
    np.save(a, x)
    if bad == 'dtype': x = x.astype('f4')
    if bad == 'shape': x = x.T
    if bad == 'negative': x[0, 0] = -1.
    if bad == 'nan': x[0, 0] = np.nan
    np.save(b, x)
    with pytest.raises(ValueError): F.field_gap(a, b, (3, 8))
