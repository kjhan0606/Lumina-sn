"""A100 실제구간 어댑터가 원 기준·과거 비용·단회 범위를 보존하는지 확인한다."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from test_lumina_gpu_measure_local import local
from lumina_gpu_compact import CompactDriver


def adapter(local,monkeypatch):
    monkeypatch.setitem(sys.modules,'lumina_gpu_local',local)
    path=Path(__file__).resolve().parents[1]/'scripts/lumina_gpu_a100_actual.py'
    spec=importlib.util.spec_from_file_location('isolated_a100_actual',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_a100_reuses_same_physics_and_preserves_actual_interval_limits(local,monkeypatch,tmp_path):
    m=adapter(local,monkeypatch);monkeypatch.setattr(m,'evidence',lambda root:None)
    calls=[];monkeypatch.setattr(local,'prepare',lambda root,**kw:calls.append(kw));m.prepare(tmp_path)
    assert calls[0]['interval_indices']==(13003,) and local.LIMITS['local_calls']==1
    assert local.LIMITS['already_allocated_gpu_seconds']==422 and local.LIMITS['prep_window_seconds']==1378
    assert local.LIMITS['original_preparation_budget_seconds']==1800
    assert local.LIMITS['local_internal_seconds']==140 and local.LIMITS['interval_seconds']==120
    assert local.LIMITS['external_wall_seconds']==150 and local.LIMITS['slurm_allocation_seconds']==180
    assert local.LIMITS['cost_target_seconds']==10 and local.LIMITS['fine_each_output']
    assert local.LIMITS['source_relative']==1e-8 and local.LIMITS['spatial_relative']==1e-5
    assert local.LIMITS['local_full_vectors_total']==512 and local.LIMITS['automatic_retries']==0
    assert not local.LIMITS['shared_accuracy_only'] and not local.LIMITS['actual_model_F_calls']
    assert calls[0]['comparison_context']['workspace_expected_bytes']==1013090856
    assert CompactDriver.interval_limit_seconds==60. and m.A100IntervalDriver.interval_limit_seconds==120.


def test_a100_does_not_allow_silent_interval_extension(local,monkeypatch):
    m=adapter(local,monkeypatch);d=m.A100IntervalDriver.__new__(m.A100IntervalDriver)
    d.bound=lambda:None;d.provider=SimpleNamespace(_source=object())
    with pytest.raises(ValueError,match='시간 한도'):d.interval(None,13003,None,max_seconds=121.)
    with pytest.raises(ValueError,match='물리 구간'):
        d.interval(SimpleNamespace(source=object(),panels=0),13003,None,max_seconds=120.)


def test_a100_still_refuses_busy_gpu_and_does_not_weaken_10_seconds(local,monkeypatch):
    adapter(local,monkeypatch)
    assert local.device_ready(24*1024,0) and not local.device_ready(24*1024,6)
    assert not local.device_ready(24*1024-1,0)
    assert local.completion_assessment(10.)['performance_qualified']
    assert not local.completion_assessment(10.001)['performance_qualified']
