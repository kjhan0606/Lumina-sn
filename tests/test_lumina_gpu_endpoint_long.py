"""사용자 시간확대가 정확도나 성능 판정을 조용히 바꾸지 않도록 확인한다."""
import importlib.util
from pathlib import Path
import sys

import pytest
from test_lumina_gpu_measure_local import local
from lumina_gpu_compact import CompactDriver,EndpointCompactDriver


def test_cost_is_required_by_default_but_can_be_reported_separately(local):
    entry=dict(accuracy_met=True,**local.completion_assessment(123.))
    assert not local.local_acceptance(entry)
    assert local.local_acceptance(entry,require_cost_target=False)
    result=local.local_completion([entry],require_cost_target=False)
    assert result['status']=='GPU_LOCAL_ACCURACY_MET_COST_REPORTED_NOT_FIELD'
    assert result['performance_qualified'] is False
    with pytest.raises(ValueError):local.local_completion([entry])
    with pytest.raises(ValueError):local.local_acceptance(entry,require_cost_target='false')
    entry['accuracy_met']=False
    assert not local.local_acceptance(entry,require_cost_target=False)
    with pytest.raises(ValueError):local.local_completion([entry],require_cost_target=False)


def test_original_short_and_shared_completion_remain_distinct(local):
    entry=dict(accuracy_met=True,**local.completion_assessment(3.))
    assert local.local_completion([entry])==dict(status='GPU_LOCAL_MET_NOT_FIELD',performance_qualified=True)
    shared=dict(accuracy_met=True,**local.completion_assessment(123.,shared_accuracy=True))
    result=local.local_completion([shared],shared_accuracy=True)
    assert result['status']=='GPU_LOCAL_SHARED_ACCURACY_MET_PERFORMANCE_UNASSESSED_NOT_FIELD'
    assert result['performance_qualified'] is False
    with pytest.raises(ValueError):local.local_completion([],require_cost_target=False)


def test_long_adapter_preserves_physics_and_extends_only_explicit_time_policy(local,monkeypatch,tmp_path):
    monkeypatch.setitem(sys.modules,'lumina_gpu_local',local)
    # 의존 어댑터의 import 부수효과도 격리된 local에만 적용한다.
    monkeypatch.delitem(sys.modules,'lumina_gpu_endpoint_actual',raising=False)
    path=Path(__file__).resolve().parents[1]/'scripts/lumina_gpu_endpoint_long.py'
    spec=importlib.util.spec_from_file_location('isolated_endpoint_long',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    monkeypatch.setattr(m,'evidence',lambda root:None)
    monkeypatch.setattr(m,'PROOFS',(str(tmp_path/'device-test.exit'),))
    (tmp_path/'device-test.exit').write_text('0\n')
    calls=[];monkeypatch.setattr(local,'prepare',lambda root,**kw:calls.append(kw));m.prepare(tmp_path)
    assert calls[0]['interval_indices']==(13003,)
    assert local.LIMITS['already_allocated_gpu_seconds']==847
    assert local.LIMITS['original_preparation_budget_seconds']==1800
    assert local.LIMITS['user_extended_time_budget'] and local.LIMITS['prep_window_seconds']==3600
    assert local.LIMITS['interval_seconds']==3400 and local.LIMITS['local_internal_seconds']==3500
    assert local.LIMITS['external_wall_seconds']==3550 and local.LIMITS['slurm_allocation_seconds']==3600
    assert local.LIMITS['require_cost_target'] is False and local.LIMITS['cost_target_seconds']==10
    assert local.LIMITS['local_full_vectors_total']==512 and local.LIMITS['fine_each_output']
    assert local.LIMITS['source_relative']==1e-8 and local.LIMITS['spatial_relative']==1e-5
    assert local.LIMITS['local_calls']==1 and local.LIMITS['automatic_retries']==0
    assert local.LIMITS['GPU']==1 and local.LIMITS['host_GiB']==local.LIMITS['device_GiB']==16
    assert not local.LIMITS['actual_model_F_calls'] and not local.LIMITS['SE_RE_updates']
    assert calls[0]['comparison_context']['workspace_expected_bytes']==1048771496
    assert issubclass(m.LongEndpointDriver,EndpointCompactDriver)
    assert m.LongEndpointDriver.interval_limit_seconds==3400.
    assert CompactDriver.interval_limit_seconds==60.
