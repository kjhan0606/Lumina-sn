"""전ν BF 부분 출력의 비파괴 메타데이터 복구 시험. 물리 재계산은 없다."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_lumina_model_point_bound as runner


def fixture(out,invalid=False):
    out.mkdir()
    values=np.full((2,6),np.nan);values[0]=[1,2,3,-4,-5,6]
    if invalid:values[0,0]=np.nan
    np.save(out/'unit-rates.npy',values)
    np.save(out/'routes-complete.npy',np.array([True,False]))
    np.savez(out/'roster.npz',Z=[14,16],stage=[2,3])


def test_saved_partial_recovery_preserves_bytes_and_does_not_certify_complete(tmp_path):
    out=tmp_path/'run';fixture(out)
    before={p.name:runner.B.sha(p) for p in out.iterdir()}
    assert runner.recover_saved(out)==0
    import json
    report=json.loads((out/'recovery.json').read_text())
    assert report['routes_complete']==1 and report['routes_expected']==2
    assert not report['complete_bound_frequency_integral']
    assert not report['start_manifest_available']
    assert report['original_stop_reason']=='UNKNOWN_FINAL_METADATA_FAILED'
    assert report['artifact_sha256']==before
    assert all(runner.B.sha(out/name)==digest for name,digest in before.items())
    with pytest.raises(FileExistsError):runner.recover_saved(out)


def test_recovery_rejects_nonfinite_completed_route_without_writing_report(tmp_path):
    out=tmp_path/'run';fixture(out,True)
    with pytest.raises(ValueError):runner.recover_saved(out)
    assert not (out/'recovery.json').exists()
