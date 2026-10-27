import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import lumina_continue_response_dispatch as D


def test_original_response_or_uncertain_submission_never_duplicates(tmp_path):
    marker = tmp_path/'response-submission.json'
    assert D.may_submit(marker)
    for record in ({'state': 'SUBMITTED_ONCE', 'job_id': 1},
                   {'state': 'SUBMISSION_ATTEMPT_STARTED'},
                   {'state': 'SUBMISSION_UNCONFIRMED_NO_RETRY', 'returncode': 255,
                    'stderr': 'connection lost'}):
        marker.write_text(json.dumps(record))
        assert not D.may_submit(marker)
    marker.write_text(json.dumps({'state': 'SUBMISSION_UNCONFIRMED_NO_RETRY', 'returncode': 255,
                                  'stderr': 'ssh: Could not resolve hostname grammar'}))
    assert D.may_submit(marker)


def test_existing_response_skips_without_ssh(tmp_path, monkeypatch):
    control = tmp_path/'control'; control.mkdir()
    (control/'response-submission.json').write_text(json.dumps({'state': 'SUBMITTED_ONCE'}))
    parent = tmp_path/'parent'; parent.mkdir()
    monkeypatch.setattr(D.subprocess, 'run', lambda *a, **k: (_ for _ in ()).throw(AssertionError('중복 제출')))
    result = D.dispatch(parent, control, '127.0.0.1')
    assert result['state'] == 'ORIGINAL_RESPONSE_SUBMITTED_OR_UNCERTAIN_NO_DUPLICATE'
    assert not (parent/'response').exists()
    assert json.loads((parent/'continued-response-submission.json').read_text()) == result


def test_completed_second_submits_once(tmp_path, monkeypatch):
    parent = tmp_path/'parent'; parent.mkdir()
    control = tmp_path/'control'; control.mkdir()
    response_script = control/'submit_lumina_new_field_response.slurm'; response_script.write_text('job')
    coeff = tmp_path/'coeff'; (coeff/'boundary').mkdir(parents=True)
    prep = coeff/'boundary/preparation.json'; prep.write_text('prep')
    feedback = parent/'feedback.json'
    prep_hash = '4707762a819fee7f013fb8655ad5d6a738c9f64cfd1ddc897e5138caa9478887'
    feedback.write_text(json.dumps({'accepted_preparation_sha256': prep_hash,
        'sweeps': [{'state': 'COMPLETE_F_NOT_CONVERGENCE'}], 'last_complete': {'sweep': 1}}))
    (parent/'continuation.json').write_text(json.dumps({'state': 'SECOND_SWEEP_COMPLETE',
        'feedback': str(feedback), 'feedback_sha256': D.sha(feedback)}))
    config = {'control_sha256': {response_script.name: D.sha(response_script)},
              'coefficient_result': str(coeff/'result.json'), 'response_commit': 'fixed'}
    config_path = control/'config.json'; config_path.write_text(json.dumps(config))
    real_sha = D.sha
    monkeypatch.setattr(D, 'sha', lambda path: ('53783a45a1a28cb9ba5694a211e6437897de8e0c500d0a06b79223011eb73fda'
        if Path(path) == config_path else prep_hash if Path(path) == prep else real_sha(path)))
    def submit(command, **kwargs):
        assert command[-2] == '127.0.0.1' and str(feedback) in command[-1]
        return type('Result', (), {'returncode': 0, 'stdout': '1234\n', 'stderr': ''})()
    monkeypatch.setattr(D.subprocess, 'run', submit)
    result = D.dispatch(parent, control, '127.0.0.1')
    assert result['state'] == 'SUBMITTED_ONCE' and result['job_id'] == 1234
    assert (parent/'response').is_dir()
