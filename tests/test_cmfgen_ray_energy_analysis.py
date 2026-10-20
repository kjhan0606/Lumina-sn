"""수치 항등식의 작은 잔차를 물리 보존으로 대체하지 않는다."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_cmfgen_ray_energy import energy_summary,analyze


def record():
    return dict(budget=np.array([[.5,0,.2,.1,.2],[.5,0,.2,.1,.2]]),
        ray_flux=np.array([3.,2.,1.]),native_flux=np.array([3.,2.,1.]),
        work_first=np.zeros(2),work_last=np.zeros(2),identity_L1=np.zeros(2),
        maximum_endpoint_error=0.,maximum_identity_relative=0.)


def test_interpolation_corrections_are_not_physical_energy_sources():
    summary,cells=energy_summary(record())
    assert abs(summary['numerical_identity_signed_defect'])<1e-14
    assert summary['physical_discrete_signed_defect']==pytest.approx(.6)
    assert summary['physical_continuous_signed_defect']==pytest.approx(1.)
    assert np.all(cells['physical_continuous_defect']>0)


def test_frequency_work_and_endpoint_signs_are_separate():
    a=record();a['budget'][:,1]=.3;a['work_first'][:]=.5
    summary,_=energy_summary(a)
    assert abs(summary['frequency_discretization_difference'])<1e-14
    assert summary['physical_continuous_signed_defect']==pytest.approx(summary['physical_discrete_signed_defect'])


def test_partial_radiation_cannot_be_consumed_as_energy_evidence(tmp_path):
    run=tmp_path/'partial';(run/'seq_logs').mkdir(parents=True)
    (run/'seq_logs/whole-runtime.json').write_text(json.dumps(dict(
        diagnostic_complete=False,returncode=97,cell_energy=dict(count=1))))
    out=tmp_path/'not-created'
    with pytest.raises(ValueError,match='완료'):
        analyze(run,tmp_path/'baseline',out)
    assert not out.exists()
