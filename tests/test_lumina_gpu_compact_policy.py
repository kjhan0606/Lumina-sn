"""공유 정확도 모드는 명시 선택이며 메모리와 성능 해석을 바꾸지 않는다."""
import json
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import test_lumina_gpu_compact_device as device
from test_lumina_gpu_measure_local import local


def setup(monkeypatch,tmp_path,free,util,shared):
    monkeypatch.setenv('LUMINA_COMPACT_GPU_LIBRARY','unused.so')
    monkeypatch.setenv('LUMINA_COMPACT_GPU_RECORD_ROOT',str(tmp_path))
    if shared:monkeypatch.setenv('LUMINA_COMPACT_SHARED_CORRECTNESS','1')
    else:monkeypatch.delenv('LUMINA_COMPACT_SHARED_CORRECTNESS',raising=False)
    monkeypatch.setattr(device.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout=f'{free}, {util}'))
    class Runtime:
        def __init__(self,*args):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def report(self):return {'live':4}
    monkeypatch.setattr(device,'Runtime',Runtime)
    return device.gpu.__wrapped__()


def test_default_device_test_retains_original_idle_requirement(monkeypatch,tmp_path):
    iterator=setup(monkeypatch,tmp_path,26188,14,False)
    with pytest.raises(AssertionError):next(iterator)
    assert not (tmp_path/'memory.json').exists()


def test_explicit_shared_mode_records_load_and_forbids_timing_acceptance(monkeypatch,tmp_path):
    iterator=setup(monkeypatch,tmp_path,26188,14,True)
    next(iterator)
    with pytest.raises(StopIteration):next(iterator)
    d=json.loads((tmp_path/'memory.json').read_text())
    assert d['shared_correctness'] and d['precheck_utilization']==14
    assert d['timing_acceptance'] is False


def test_shared_mode_does_not_relax_free_memory_requirement(monkeypatch,tmp_path):
    iterator=setup(monkeypatch,tmp_path,23000,0,True)
    with pytest.raises(AssertionError):next(iterator)
    assert not (tmp_path/'memory.json').exists()


def test_native_compact_adapter_preserves_original_cost_and_precision_limits(local,monkeypatch,tmp_path):
    monkeypatch.setitem(sys.modules,'lumina_gpu_local',local)
    path=Path(__file__).resolve().parents[1]/'scripts/lumina_gpu_compact_local.py'
    spec=importlib.util.spec_from_file_location('isolated_compact_local',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'evidence',lambda root:None)
    calls=[];monkeypatch.setattr(local,'prepare',lambda root,**kw:calls.append(kw))
    module.prepare(tmp_path)
    assert calls[0]['interval_indices']==(13003,)
    assert local.LIMITS['local_calls']==1 and local.LIMITS['automatic_retries']==0
    assert local.LIMITS['cost_target_seconds']==10 and local.LIMITS['fine_each_output']
    assert local.LIMITS['prep_window_seconds']==1722 and local.LIMITS['already_allocated_gpu_seconds']==78
    assert local.LIMITS['source_relative']==1e-8 and local.LIMITS['spatial_relative']==1e-5
    assert calls[0]['comparison_context']['workspace_expected_bytes']==1013090856
    assert local.LIMITS['actual_model_F_calls']==0
    module.configure_shared_accuracy()
    assert local.LIMITS['shared_accuracy_only'] and local.LIMITS['performance_qualification']=='UNASSESSED_SHARED'
    assert local.LIMITS['cost_target_seconds']==10 and local.LIMITS['external_wall_seconds']==75
    assert local.LIMITS['prep_window_seconds']==1722 and local.LIMITS['actual_model_F_calls']==0
    from lumina_gpu_compact import CompactDriver
    assert CompactDriver.interval_limit_seconds==60.
    module.configure_double_time()
    assert local.LIMITS['external_wall_seconds']==150 and local.LIMITS['local_internal_seconds']==140
    assert local.LIMITS['interval_seconds']==120 and local.LIMITS['user_approved_double_time']
    assert local.LIMITS['already_allocated_gpu_seconds']==148 and local.LIMITS['prep_window_seconds']==1652
    assert local.LIMITS['cost_target_seconds']==10 and local.LIMITS['automatic_retries']==0
    assert CompactDriver.interval_limit_seconds==60. and local.GpuDriver.interval_limit_seconds==120.
    driver=local.GpuDriver.__new__(local.GpuDriver);driver.bound=lambda:None
    driver.provider=SimpleNamespace(_source=object())
    invalid=SimpleNamespace(source=object(),panels=0)
    with pytest.raises(ValueError,match='물리 구간'):
        driver.interval(invalid,0,None,max_seconds=120.)
    with pytest.raises(ValueError,match='시간 한도'):
        driver.interval(invalid,0,None,max_seconds=121.)


def test_native_shared_accuracy_cannot_promote_even_a_fast_time_to_performance(local):
    assert not local.device_ready(26188,15)
    assert local.device_ready(26188,15,shared_accuracy=True)
    assert not local.device_ready(23000,0,shared_accuracy=True)
    for seconds in [.1,10.,30.]:
        d=local.completion_assessment(seconds,shared_accuracy=True)
        assert d['cost_met'] is None and not d['performance_qualified']
        assert d['observed_cost_target_met']==(seconds<=10.)
    assert local.completion_assessment(.1)['cost_met'] is True
    assert local.completion_assessment(30.)['cost_met'] is False
