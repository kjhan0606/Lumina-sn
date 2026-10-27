import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import lumina_continue_accepted_field as C


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
