"""클러스터 간 제출 제어 시험. 실제 ssh/sbatch 호출은 없다."""
import json
from pathlib import Path
import shlex
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


def test_response_uses_verified_IP_but_retains_alias_authentication_and_strict_host_key(setup):
    config,c,calls=setup;gpu=Path(c['gpu_root']);(gpu/'feedback').mkdir()
    coefficient=json.loads(Path(c['coefficient_result']).read_text())
    (gpu/'feedback/feedback.json').write_text(json.dumps(dict(last_complete=dict(sweep=1),
        accepted_preparation_sha256=coefficient['preparation']['sha256'],
        sweeps=[dict(state='COMPLETE_F_NOT_CONVERGENCE')])))
    c['submission_host_addresses']={'grammar':'192.0.2.20'}
    config.write_text(json.dumps(c))
    result=D.dispatch(config,D.sha(config),'response')
    ssh=calls[0]
    assert ssh[-2]=='grammar' and 'HostName=192.0.2.20' in ssh and 'HostKeyAlias=grammar' in ssh
    assert 'StrictHostKeyChecking=yes' in ssh and 'UpdateHostKeys=no' in ssh
    assert result['ssh_command']==ssh and result['job_id']==12345
    assert result['automatic_retries']==0
    with pytest.raises(FileExistsError):D.dispatch(config,D.sha(config),'response')
    assert len(calls)==1


@pytest.mark.parametrize('addresses',[
    [],{'other':'192.0.2.20'},{'grammar':True},{'grammar':'grammar'},
    {'grammar':'192.0.2.20 -oStrictHostKeyChecking=no'},
    {'grammar':'0.0.0.0'},{'grammar':'224.0.0.1'},{'grammar':'fe80::1%en0'}])
def test_bad_route_cannot_submit_or_create_result_roots(setup,addresses):
    config,c,calls=setup;c['submission_host_addresses']=addresses
    config.write_text(json.dumps(c))
    with pytest.raises(ValueError):D.dispatch(config,D.sha(config),'response')
    assert not calls and not Path(c['response_parent']).exists()
    assert not (config.parent/'response-submission.json').exists()


def test_IP_route_does_not_change_other_submission_host():
    command=D.submission_ssh({'submission_host_addresses':{'grammar':'192.0.2.20'}},'syntax')
    assert command==['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','syntax']


def test_incomplete_coefficients_never_submit_GPU(setup):
    config, c, calls = setup
    result = Path(c['coefficient_result']); data = json.loads(result.read_text())
    data['state'] = 'INCOMPLETE'; result.write_text(json.dumps(data))
    with pytest.raises(ValueError): D.dispatch(config, D.sha(config), 'field')
    assert not calls


def test_one_sweep_and_previous_material_trial_reach_the_submission_argv(setup):
    config,c,calls=setup
    c['limits']=dict(maximum_F=1)
    c['initial_field']=dict(J=dict(path='/old material/J.npy',sha256='a'*64),
        preparation=dict(path='/old material/preparation.json',sha256='b'*64))
    config.write_text(json.dumps(c))
    D.dispatch(config,D.sha(config),'field')
    command=shlex.split(calls[0][-1])
    assert command[command.index('--max-sweeps')+1]=='1'
    assert command[command.index('--input-j')+1]=='/old material/J.npy'
    assert command[command.index('--input-j-sha256')+1]=='a'*64
    assert command[command.index('--initial-preparation')+1]=='/old material/preparation.json'
    assert command[command.index('--initial-preparation-sha256')+1]=='b'*64
    assert '--resume-checkpoint' not in command


@pytest.mark.parametrize('fault',[0,3,True,'missing_J','bad_sha'])
def test_invalid_sweep_or_partial_warm_start_cannot_submit(setup,fault):
    config,c,calls=setup
    if fault in ('missing_J','bad_sha'):
        c['initial_field']=dict(preparation=dict(path='P',sha256='b'*64))
        if fault=='bad_sha':c['initial_field']['J']=dict(path='J',sha256='invalid')
    else:c['limits']=dict(maximum_F=fault)
    config.write_text(json.dumps(c))
    with pytest.raises(ValueError):D.dispatch(config,D.sha(config),'field')
    assert not calls and not (config.parent/'field-submission.json').exists()


def test_control_preparation_uses_the_actual_coefficient_commit(tmp_path,monkeypatch):
    import prepare_lumina_bundle_dispatch as P
    gpu=tmp_path/'gpu';gpu.mkdir()
    (gpu/'prepared.json').write_text(json.dumps(dict(source_commit='b'*40)))
    def git_show(*args,**kwargs):return SimpleNamespace(stdout=b'# sealed control\n')
    monkeypatch.setattr(P.subprocess,'run',git_show)
    prepared=P.prepare(tmp_path,tmp_path/'control','b'*40,gpu,tmp_path/'new-coeff/result.json',
        tmp_path/'response','c'*40,max_sweeps=1,input_j=tmp_path/'old/J.npy',input_j_sha256='a'*64,
        initial_preparation=tmp_path/'old/P.json',initial_preparation_sha256='d'*64)
    config=json.loads(Path(prepared['config']).read_text())
    assert config['coefficient_commit']=='c'*40 and config['response_commit']=='b'*40
    assert config['limits']['maximum_F']==1 and config['limits']['GPU_hours']==12
    assert config['initial_field']['J']['path']==str(tmp_path/'old/J.npy')
    assert prepared['jobs_submitted']==0


@pytest.mark.parametrize('control',[False,True])
def test_real_batch_wrapper_forwards_bounded_feedback_options_and_response(tmp_path,control):
    gpu=tmp_path/'gpu';(gpu/'code/scripts').mkdir(parents=True)
    runner=gpu/'code/scripts/run_lumina_resident_field.sh'
    runner.write_text("""#!/bin/bash
python3 -c 'import json,sys;from pathlib import Path;Path("field-args.json").write_text(json.dumps(sys.argv[1:]))' "$@"
""")
    args=[str(gpu),'new P.json','prep-sha'];options=['--max-sweeps','2']
    if control:
        folder=tmp_path/'control';folder.mkdir();config=folder/'config.json'
        (folder/'lumina_bundle_dispatch.py').write_text(
            'import json,sys\nfrom pathlib import Path\nPath(__file__).with_name("response-args.json").write_text(json.dumps(sys.argv[1:]))\n')
        options=['--max-sweeps','1','--input-j','old J.npy','--input-j-sha256','j-sha',
                 '--initial-preparation','old P.json','--initial-preparation-sha256','old-sha']
        args += [str(config),'config-sha',*options]
    script=Path(__file__).resolve().parents[1]/'scripts/submit_lumina_accepted_feedback.slurm'
    subprocess.run(['bash',str(script),*args],check=True,capture_output=True,text=True)
    forwarded=json.loads((gpu/'field-args.json').read_text())
    assert forwarded[forwarded.index('--max-sweeps'):]==options
    if control:
        assert json.loads((folder/'response-args.json').read_text())==[str(config),'config-sha','response']
