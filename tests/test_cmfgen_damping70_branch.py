"""별도 반-step 증명과 기본 full-step 적격성 분리를 확인한다."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_cmfgen_damping70_branch as branch
from run_cmfgen_lambda_probe import sha


def sample():
    before = np.array([[3.,4.],[7.,6.],[11.,12.],[1.,1.]])
    reference = np.array([[4.,5.],[6.,5.],[10.,11.],[1.,1.]])
    raw = 1-reference/before
    rows = [[1,1,.5,.5],[2,1,.5,.5]]
    return before,(before+reference)/2,reference,raw,raw.copy(),rows


def test_actual_midpoint_is_verified_without_default_full_step_permission():
    args = sample()
    proof = branch.midpoint_evidence(*args)
    assert proof['verified']
    assert proof['default_next_outer_eligibility_overridden'] is False
    assert not branch.applied_full_step(args[0],args[1],args[3])


@pytest.mark.parametrize('fault',['state','raw','scale','missing','repeated','unscaled'])
def test_faults_never_gain_branch_permission(fault):
    args = list(sample())
    if fault == 'state':args[1][0,0] += 1e-6
    if fault == 'raw':args[3][0,0] += 1e-4
    if fault == 'scale':args[5][0][1] = .9
    if fault == 'missing':args[5] = args[5][:1]
    if fault == 'repeated':args[5][1][0] = 1
    if fault == 'unscaled':args[1] = args[2].copy()
    assert not branch.midpoint_evidence(*args)['verified']


def test_existing_output_is_preserved_before_plan_reads(tmp_path,monkeypatch):
    monkeypatch.setenv('OMP_NUM_THREADS','16')
    monkeypatch.setenv('SLURM_JOB_ID','7')
    with pytest.raises(ValueError,match='기존'):
        branch.run(Path('not-read'),'not-read',tmp_path)


def test_unexpected_resource_or_root_is_rejected_before_plan_reads(tmp_path,monkeypatch):
    monkeypatch.setenv('OMP_NUM_THREADS','16')
    monkeypatch.setenv('SLURM_JOB_ID','7')
    monkeypatch.setenv('SLURM_CPUS_PER_TASK','1')
    monkeypatch.setenv('SLURM_MEM_PER_NODE','32768')
    with pytest.raises(ValueError,match='자원'):
        branch.run(Path('not-read'),'not-read',tmp_path/'new')


def fixture_plan():
    return dict(purpose_id=branch.PURPOSE,maximum_native_executions=2,
        maximum_RT_executions=1,automatic_extension=False,omega_sequence=[.5,1.],
        T_release=False,first_parent_seconds=1200,first_child_seconds=900,
        second_parent_seconds=2700,second_child_seconds=2400,sealed_inputs={})


def test_only_registered_finite_plan_is_accepted(tmp_path):
    path = tmp_path/'plan.json'
    plan = fixture_plan()
    path.write_text(json.dumps(plan))
    assert branch.load_plan(path,sha(path)) == plan
    with pytest.raises(ValueError,match='신원'):
        branch.load_plan(path,'wrong')
    for key,value in (('maximum_RT_executions',2),('maximum_native_executions',3),
        ('automatic_extension',True),('omega_sequence',[.25,1.]),('T_release',True),
        ('first_child_seconds',2400),('second_parent_seconds',5400)):
        changed = dict(plan,**{key:value})
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError,match='계획'):
            branch.load_plan(path,sha(path))
