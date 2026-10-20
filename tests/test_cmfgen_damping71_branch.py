"""완료 반-step 증명과 실제 C1 없이 후속 수송을 열지 않는지 확인한다."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_cmfgen_damping71_branch as branch
import run_cmfgen_feedback_c1 as C1
from run_cmfgen_lambda_probe import sha


def test_existing_outputs_are_refused_before_source_reads(tmp_path,monkeypatch):
    monkeypatch.setenv('OMP_NUM_THREADS','16')
    monkeypatch.setenv('SLURM_JOB_ID','7')
    with pytest.raises(ValueError,match='기존'):
        branch.run(Path('no'),'no',Path('no'),Path('no'),'no',tmp_path)
    with pytest.raises(ValueError,match='기존'):
        C1.run(Path('no'),Path('no'),70,tmp_path)


@pytest.mark.parametrize('key,value', [('material_complete',False),('failure','injected'),
    ('source_unchanged',False),('branch_next_permitted',False),('next_outer_eligible',True),
    ('native_executions',2),('RT_executions',1),('material_record',71)])
def test_failed_or_unproven_first_branch_never_reads_transport(tmp_path,monkeypatch,key,value):
    (tmp_path/'seq_logs').mkdir()
    claims = tmp_path/'claims';claims.mkdir()
    runtime = dict(branch_plan_sha256='plan',branch='70B',material_complete=True,
        field_complete=True,source_unchanged=True,branch_next_permitted=True,
        next_outer_eligible=False,RT_executions=0,native_executions=1,material_record=70,
        material=dict(returncode=0,stop_reason=None))
    runtime[key] = value
    path = tmp_path/'seq_logs/feedback-runtime.json'
    path.write_text(json.dumps(runtime))
    (tmp_path/'seq_logs/feedback-inputs.json').write_text(json.dumps(dict(
        branch_plan_sha256='plan',omega=.5,source='/original69')))
    (claims/'70B.json').write_text(json.dumps(dict(root=str(tmp_path),branch='70B',plan_sha256='plan')))
    monkeypatch.setattr(branch,'preflight',lambda *args:pytest.fail('실패한 첫 분기로 수송 준비'))
    with pytest.raises(ValueError,match='70B'):
        branch.verify_source(dict(claims_root=str(claims),source69='/original69'),tmp_path,Path('no'),sha(path))


def test_changed_first_runtime_is_rejected_before_other_reads(tmp_path):
    (tmp_path/'seq_logs').mkdir()
    (tmp_path/'seq_logs/feedback-runtime.json').write_text('{}')
    with pytest.raises(ValueError,match='신원'):
        branch.verify_source({},tmp_path,Path('no'),'wrong')


def test_failed_branch_never_creates_C1_output(tmp_path):
    (tmp_path/'seq_logs').mkdir()
    (tmp_path/'seq_logs/feedback-runtime.json').write_text(json.dumps(dict(
        branch='70B',source_unchanged=True,branch_next_permitted=False)))
    out = tmp_path/'new'
    with pytest.raises(ValueError,match='증명'):
        C1.run(tmp_path,Path('no'),70,out)
    assert not out.exists()
