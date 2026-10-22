"""χ-only τ 실제구간이 원 정확도/한도와 미실행 범위를 유지하는지 확인한다."""
import importlib.util
from pathlib import Path
import sys

from test_lumina_gpu_measure_local import local
from lumina_gpu_compact import EndpointCompactDriver


def test_chi_tau_actual_preserves_full_interval_contract(local,monkeypatch,tmp_path):
    monkeypatch.setitem(sys.modules,'lumina_gpu_local',local)
    path=Path(__file__).resolve().parents[1]/'scripts/lumina_gpu_chi_tau_actual.py'
    spec=importlib.util.spec_from_file_location('isolated_chi_tau_actual',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    monkeypatch.setattr(m,'evidence',lambda root:None)
    calls=[];monkeypatch.setattr(local,'prepare',lambda root,**kw:calls.append(kw));m.prepare(tmp_path)
    assert calls[0]['interval_indices']==(13003,)
    assert local.LIMITS['already_allocated_gpu_seconds']==705 and local.LIMITS['prep_window_seconds']==1095
    assert local.LIMITS['original_preparation_budget_seconds']==1800
    for key,value in [('local_calls',1),('automatic_retries',0),('local_internal_seconds',140),
        ('interval_seconds',120),('external_wall_seconds',150),('slurm_allocation_seconds',180),
        ('cost_target_seconds',10),('local_full_vectors_total',512),('source_relative',1e-8),
        ('spatial_relative',1e-5),('host_GiB',16),('device_GiB',16),('GPU',1)]:
        assert local.LIMITS[key]==value
    assert local.LIMITS['fine_each_output'] and not local.LIMITS['shared_accuracy_only']
    assert not local.LIMITS['actual_model_F_calls'] and not local.LIMITS['SE_RE_updates']
    assert calls[0]['comparison_context']['workspace_expected_bytes']==1048771496
    assert issubclass(m.ChiTauIntervalDriver,EndpointCompactDriver)
    assert m.ChiTauIntervalDriver.interval_limit_seconds==120.
