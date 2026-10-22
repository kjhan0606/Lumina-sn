"""끝점 재사용 실제구간 어댑터가 원 물리·오차·한도를 보존하는지 확인한다."""
import importlib.util
from pathlib import Path
import sys

from test_lumina_gpu_measure_local import local
from lumina_gpu_compact import CompactDriver,EndpointCompactDriver


def test_endpoint_actual_preserves_original_contract_and_one_attempt(local,monkeypatch,tmp_path):
    monkeypatch.setitem(sys.modules,'lumina_gpu_local',local)
    path=Path(__file__).resolve().parents[1]/'scripts/lumina_gpu_endpoint_actual.py'
    spec=importlib.util.spec_from_file_location('isolated_endpoint_actual',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    monkeypatch.setattr(m,'evidence',lambda root:None)
    calls=[];monkeypatch.setattr(local,'prepare',lambda root,**kw:calls.append(kw));m.prepare(tmp_path)
    assert calls[0]['interval_indices']==(13003,)
    assert local.LIMITS['local_calls']==1 and local.LIMITS['automatic_retries']==0
    assert local.LIMITS['already_allocated_gpu_seconds']==564 and local.LIMITS['prep_window_seconds']==1236
    assert local.LIMITS['original_preparation_budget_seconds']==1800
    assert local.LIMITS['local_internal_seconds']==140 and local.LIMITS['interval_seconds']==120
    assert local.LIMITS['external_wall_seconds']==150 and local.LIMITS['slurm_allocation_seconds']==180
    assert local.LIMITS['cost_target_seconds']==10 and local.LIMITS['fine_each_output']
    assert local.LIMITS['source_relative']==1e-8 and local.LIMITS['spatial_relative']==1e-5
    assert local.LIMITS['local_full_vectors_total']==512 and not local.LIMITS['shared_accuracy_only']
    assert not local.LIMITS['actual_model_F_calls'] and not local.LIMITS['SE_RE_updates']
    assert calls[0]['comparison_context']['workspace_expected_bytes']==1048771496
    assert issubclass(m.EndpointIntervalDriver,EndpointCompactDriver)
    assert m.EndpointIntervalDriver.endpoint_cache and m.EndpointIntervalDriver.interval_limit_seconds==120.
    assert not CompactDriver.endpoint_cache and CompactDriver.interval_limit_seconds==60.
