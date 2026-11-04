"""승인된 두 포획의 순서·실패 중단·중복 실행 방지. 실제 CMFGEN 실행 없음."""
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_cmfgen_fixedj_grid_bundle as B


@pytest.fixture
def bundle(tmp_path,monkeypatch):
    for k,v in [('SLURM_CPUS_PER_TASK','16'),('SLURM_MEM_PER_NODE','32768'),('SLURM_JOB_ID','test')]:
        monkeypatch.setenv(k,v)
    attempts=[]
    for name,h in [('h05',.5),('h025',.25)]:
        native=tmp_path/name/'native';native.mkdir(parents=True)
        (native/'VADAT').write_text('synthetic')
        attempts.append(dict(root=name,frac_dop=h,input_sha256={'VADAT':B.capture.sha(native/'VADAT')}))
    record=dict(schema='lumina-fixedj-grid-bundle-v1',resources=dict(cpus=16,memory_MiB=32768,
        wall_seconds=21600,captures=2,retries=0,gpus=0),attempts=attempts,source_sha256={})
    (tmp_path/'bundle-inputs.json').write_text(json.dumps(record))
    return tmp_path


def test_two_captures_are_sequential_and_not_a_final_physics_pass(bundle):
    called=[]
    assert B.run_bundle(bundle,lambda p:called.append(p.name) or 0)==0
    assert called==['h05','h025']
    result=json.loads((bundle/'bundle-runtime.json').read_text())
    assert result['final_goal']=='NOT_MET'
    assert result['status']=='CAPTURED_BOTH_NOT_PHYSICAL_EQUIVALENCE'
    with pytest.raises(FileExistsError):B.run_bundle(bundle,lambda p:0)


def test_first_failure_never_starts_second(bundle):
    called=[]
    assert B.run_bundle(bundle,lambda p:called.append(p.name) or 1)==1
    assert called==['h05']


def test_changed_second_input_is_not_consumed(bundle):
    called=[]
    (bundle/'h025/native/VADAT').write_text('changed')
    assert B.run_bundle(bundle,lambda p:called.append(p.name) or 0)==1
    assert called==['h05']


def test_insufficient_remaining_wall_time_preserves_first_only(bundle):
    times=iter([0.,17000.,17001.])
    called=[]
    assert B.run_bundle(bundle,lambda p:called.append(p.name) or 0,clock=lambda:next(times))==1
    assert called==['h05']
