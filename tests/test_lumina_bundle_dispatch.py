"""클러스터 간 제출 제어 시험. 실제 ssh/sbatch 호출은 없다."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import lumina_bundle_dispatch as D


@pytest.fixture
def setup(tmp_path, monkeypatch):
    control = tmp_path/'control'; control.mkdir()
    gpu = tmp_path/'gpu'; gpu.mkdir(); (gpu/'prepared.json').write_text('{}')
    run = tmp_path/'coeff'; (run/'boundary').mkdir(parents=True)
    prep = run/'boundary/preparation.json'; prep.write_text('{}')
    result = run/'result.json'; result.write_text(json.dumps(dict(
        state='COMPLETE_NEW_ACCEPTED_COEFFICIENTS_AND_BOUNDARY', source_commit='a'*40,
        preparation=dict(path=str(prep), sha256=D.sha(prep)))))
    c = dict(schema='lumina-accepted-bundle-dispatch-v1', control_sha256={},
        coefficient_result=str(result), coefficient_commit='a'*40, gpu_root=str(gpu),
        gpu_prepared_sha256=D.sha(gpu/'prepared.json'), response_parent=str(tmp_path/'response'), response_commit='b'*40)
    config = control/'config.json'; config.write_text(json.dumps(c)); calls = []
    def submit(*args, **kwargs):
        calls.append(args[0]); return SimpleNamespace(returncode=0, stdout='12345\n', stderr='')
    monkeypatch.setattr(D.subprocess, 'run', submit)
    return config, c, calls


def test_field_submitted_once_with_bounded_script(setup):
    config, c, calls = setup
    result = D.dispatch(config, D.sha(config), 'field')
    assert result['job_id'] == 12345 and len(calls) == 1 and calls[0][-2] == 'syntax'
    assert '--no-requeue' in calls[0][-1] and 'submit_lumina_accepted_feedback.slurm' in calls[0][-1]
    with pytest.raises(FileExistsError): D.dispatch(config, D.sha(config), 'field')
    assert len(calls) == 1


def test_uncertain_submission_is_preserved_and_never_retried(setup, monkeypatch):
    config, c, calls = setup
    def timeout(*args, **kwargs): raise subprocess.TimeoutExpired(args[0], 45)
    monkeypatch.setattr(D.subprocess, 'run', timeout)
    with pytest.raises(subprocess.TimeoutExpired): D.dispatch(config, D.sha(config), 'field')
    marker = json.loads((config.parent/'field-submission.json').read_text())
    assert marker['state'] == 'SUBMISSION_UNCONFIRMED_NO_RETRY'
    with pytest.raises(FileExistsError): D.dispatch(config, D.sha(config), 'field')


@pytest.mark.parametrize('complete', [True, False])
def test_response_requires_completed_new_field(setup, complete):
    config, c, calls = setup; gpu = Path(c['gpu_root']); (gpu/'feedback').mkdir()
    coefficient = json.loads(Path(c['coefficient_result']).read_text())
    (gpu/'feedback/feedback.json').write_text(json.dumps(dict(
        last_complete=dict(sweep=1) if complete else None,
        accepted_preparation_sha256=coefficient['preparation']['sha256'],
        sweeps=[dict(state='COMPLETE_F_NOT_CONVERGENCE')], status='PARTIAL_SWEEP_STOP')))
    if complete:
        result = D.dispatch(config, D.sha(config), 'response')
        assert result['job_id'] == 12345 and calls[0][-2] == 'grammar'
        assert '--feedback-sha' in calls[0][-1] and '--preparation-sha' in calls[0][-1]
    else:
        with pytest.raises(ValueError): D.dispatch(config, D.sha(config), 'response')
        assert not calls and not Path(c['response_parent']).exists()


def test_incomplete_coefficients_never_submit_GPU(setup):
    config, c, calls = setup
    result = Path(c['coefficient_result']); data = json.loads(result.read_text())
    data['state'] = 'INCOMPLETE'; result.write_text(json.dumps(data))
    with pytest.raises(ValueError): D.dispatch(config, D.sha(config), 'field')
    assert not calls
