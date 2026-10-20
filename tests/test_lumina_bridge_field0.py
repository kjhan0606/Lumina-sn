"""예산 계상/실패 보존의 제어 시험. 여기서 실제 물리장을 실행하지 않는다."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_lumina_bridge_field0 as R
from test_lumina_cell_candidate import verified_rel_fixture


def config(tmp_path,monkeypatch):
    args=verified_rel_fixture(tmp_path);monkeypatch.setenv('OMP_NUM_THREADS','16')
    plan=dict(initial_material=str(tmp_path),initial_record_storage_sha256='a'*64,
        reference_field_sha256='b'*64,resources_per_run=dict(CPU=16,OMP=16,memory_GiB=32,
            GPU=0,parent_seconds=2700,child_seconds=2400,node_pinned=False,retry=0))
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan));identity=R.D.sha(path)
    (args['seed']/'material-seed.json').write_text(json.dumps(dict(plan_sha256=identity,
        source_record=70,record_storage_sha256='a'*64)))
    grid=tmp_path/'frequency.npy';nu=1e14*np.exp(np.arange(32)*.001);np.save(grid,nu)
    (tmp_path/'grid.json').write_text(json.dumps(dict(dlognu=.001,reference_field_sha256='b'*64)))
    lib=tmp_path/'library.so';lib.write_bytes(b'control-test-placeholder-not-loaded')
    c=dict(plan=str(path),plan_sha256=identity,seed=str(args['seed']),cells=str(args['cells']),
        grid=str(grid),grid_sha256=R.D.sha(grid),library=str(lib),library_sha256=R.D.sha(lib),
        full_population='not-used',native_full_population='not-used',phot_library='not-used',
        kernel_library='not-used')
    cfg=tmp_path/'config.json';cfg.write_text(json.dumps(c))
    monkeypatch.setattr(R.D,'prepare',lambda *a:dict(report=dict(execution_eligible=True,
        implemented_frame=R.D.REL_FRAME,reasons=[])))
    return cfg


@pytest.mark.parametrize('failure_at',[None,1,3])
def test_one_reservation_and_no_hidden_retries(tmp_path,monkeypatch,failure_at):
    cfg=config(tmp_path,monkeypatch);out=tmp_path/'run';calls=[]
    def command(command,**kwargs):
        calls.append(command)
        if len(calls)==failure_at:raise subprocess.CalledProcessError(5,command)
        if len(calls)==3:
            field=out/'field0';field.mkdir()
            (field/'preflight.json').write_text(json.dumps(dict(status='CANDIDATE_COMPLETED')))
        assert 0<kwargs['timeout']<=2400
        return subprocess.CompletedProcess(command,0)
    monkeypatch.setattr(R.subprocess,'run',command)
    if failure_at is None:assert R.execute(cfg,R.D.sha(cfg),out)==0
    else:
        with pytest.raises(subprocess.CalledProcessError):R.execute(cfg,R.D.sha(cfg),out)
    result=json.loads((out/'execution.json').read_text())
    assert result['physical_field_executions']==(0 if failure_at==1 else 1)
    assert result['reserved_run_number']==1 and result['automatic_retry']==0
    assert result['matter_updates']==result['native_executions']==0
    assert len(calls)==(failure_at or 3)
    assert result['status']==('FIELD0_COMPLETED' if failure_at is None else 'FAILED_PRESERVED')
    with pytest.raises(ValueError,match='보존'):R.execute(cfg,R.D.sha(cfg),out)


def test_config_identity_failure_cannot_create_or_execute(tmp_path,monkeypatch):
    cfg=config(tmp_path,monkeypatch);out=tmp_path/'run'
    with pytest.raises(ValueError,match='SHA256'):R.execute(cfg,'0'*64,out)
    assert not out.exists()
