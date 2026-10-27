import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import lumina_continue_accepted_field as C
import lumina_new_field_response as N


def test_completed_second_is_not_repeated(tmp_path, monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID', '1')
    staged = tmp_path/'staged'; feedback = staged/'feedback'; feedback.mkdir(parents=True)
    prep = tmp_path/'prep'; prep.write_text('input')
    (feedback/'feedback.json').write_text(json.dumps({'accepted_preparation_sha256': C.sha(prep)}))
    (feedback/'F-2').mkdir()
    (feedback/'F-2/resident-field.json').write_text(json.dumps({'complete': True}))
    monkeypatch.setattr(C, 'completed_transport', lambda record, nf: record.get('complete', False))
    monkeypatch.setattr(C.subprocess, 'run', lambda *a, **kw: (_ for _ in ()).throw(AssertionError('재실행')))
    parent = tmp_path/'out'; parent.mkdir()
    C.run(parent, staged, prep, C.sha(prep), tmp_path, 85800)
    assert json.loads((parent/'continuation.json').read_text())['state'] == 'ORIGINAL_F2_ALREADY_COMPLETE'


def test_second_uses_first_and_preserves_matching_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID', '1')
    staged = tmp_path/'staged'; feedback = staged/'feedback'; feedback.mkdir(parents=True)
    prep = tmp_path/'prep'; prep.write_text('input')
    (feedback/'feedback.json').write_text(json.dumps({'accepted_preparation_sha256': C.sha(prep)}))
    for n in ('code', 'stage.json', 'chain_cuda.so', 'geometry.so'):
        (staged/n).mkdir() if n == 'code' else (staged/n).write_text(n)
    (feedback/'F-1').mkdir(); (feedback/'F-2').mkdir()
    first = {'complete': True, 'J': {'path': '/original/J.npy', 'sha256': 'abc'}}
    (feedback/'F-1/resident-field.json').write_text(json.dumps(first))
    (feedback/'F-2/resident-field.json').write_text(json.dumps({'checkpoint': '/saved/checkpoint'}))
    monkeypatch.setattr(C, 'completed_transport', lambda record, nf: record.get('complete', False))
    def call(command, **kwargs):
        assert command[command.index('--input-j')+1] == '/original/J.npy'
        assert command[command.index('--resume-checkpoint')+1] == '/saved/checkpoint'
        child = Path(command[2]); j = child/'J.npy'; moments = child/'moments.npy'
        record = {'complete': True, 'source_identity': {'a': 1},
                  'J': {'path': str(j), 'sha256': 'j'}, 'moments': {'path': str(moments), 'sha256': 'm'}}
        (child/'resident-field.json').write_text(json.dumps(record))
        return type('Result', (), {'returncode': 0})()
    monkeypatch.setattr(C.subprocess, 'run', call)
    parent = tmp_path/'out'; parent.mkdir()
    C.run(parent, staged, prep, C.sha(prep), tmp_path, 85800)
    result = json.loads((parent/'continuation.json').read_text())
    assert result['state'] == 'SECOND_SWEEP_COMPLETE'
    report = json.loads((parent/'feedback/feedback.json').read_text())
    assert report['sweeps'][0]['actual_physical_sweep'] == 2
    assert report['original_first_J'] == first['J']


def test_continued_field_is_accepted_by_real_response_reader(tmp_path, monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID', '1')
    staged = tmp_path/'staged'; old = staged/'feedback'; old.mkdir(parents=True)
    prep = tmp_path/'prep'; prep.write_text('input'); prep_sha = C.sha(prep)
    (old/'feedback.json').write_text(json.dumps({'accepted_preparation_sha256': prep_sha}))
    for name in ('code', 'stage.json', 'chain_cuda.so', 'geometry.so'):
        (staged/name).mkdir() if name == 'code' else (staged/name).write_text(name)
    (old/'F-1').mkdir()
    first = {'complete': True, 'J': {'path': '/previous/J.npy', 'sha256': 'first'}}
    (old/'F-1/resident-field.json').write_text(json.dumps(first))
    monkeypatch.setattr(C, 'completed_transport', lambda record, nf: record.get('complete', False))
    identity = {'role': 'test-material'}
    def call(command, **kwargs):
        child = Path(command[2]); j = child/'J.npy'; moments = child/'moments.npy'
        np.save(j, np.ones((2, 3))); np.save(moments, np.ones((3, 2, 3)))
        record = {'complete': True, 'schema': 'lumina-resident-fullsr-native90-v1',
            'complete_original_grid': True, 'completed_actual_F_calls': 1, 'source_identity': identity,
            'original_layout': {'frequencies': 3, 'paths': 1, 'physical_intervals': 1,
                                'step_observations': 1, 'boundary_observations': 1},
            'field_report': {'complete': True, 'failed': False, 'physical_intervals': 1,
                             'observations': 2, 'next_action': 3, 'actions': 3},
            'J': {'path': str(j), 'sha256': C.sha(j)},
            'moments': {'path': str(moments), 'sha256': C.sha(moments)}}
        (child/'resident-field.json').write_text(json.dumps(record))
        return type('Result', (), {'returncode': 0})()
    monkeypatch.setattr(C.subprocess, 'run', call)
    parent = tmp_path/'out'; parent.mkdir()
    C.run(parent, staged, prep, prep_sha, tmp_path, 85800)
    class Source:
        ns, nf = 2, 3
        def __init__(self): self.identity = identity
        radius, frequency = np.array([2., 1.]), np.array([1., 2., 3.])
        def __enter__(self): return self
        def __exit__(self, *args): return None
    monkeypatch.setattr(N, 'open_prepared', lambda *args: ({}, Source(), None, None))
    feedback = parent/'feedback/feedback.json'
    with N.open_field(prep, prep_sha, feedback, C.sha(feedback)) as (_, _, field, last):
        np.testing.assert_array_equal(field.values, np.ones((2, 3)))
        assert last['sweep'] == 1
