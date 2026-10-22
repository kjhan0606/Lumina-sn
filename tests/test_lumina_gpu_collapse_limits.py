"""시간·누적 작업만 확대하고 봉인 실행·정확도·메모리를 보존한다."""
import importlib.util
from pathlib import Path
import sys
import time

import pytest

from lumina_gpu_driver import Work
from lumina_gpu_hybrid import HybridDriver
from test_lumina_gpu_measure_local import local


@pytest.fixture
def adapter(local,monkeypatch):
    monkeypatch.setitem(sys.modules,'lumina_gpu_local',local)
    for name in ('lumina_gpu_native_actual','lumina_gpu_prefix_actual','lumina_gpu_hybrid_actual'):
        monkeypatch.delitem(sys.modules,name,raising=False)
    path=Path(__file__).resolve().parents[1]/'scripts/lumina_gpu_collapse_actual.py'
    spec=importlib.util.spec_from_file_location('isolated_collapse_limits',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_old_sealed_execution_keeps_original_limits_by_default(adapter):
    limits=adapter.local.LIMITS
    assert (limits['interval_seconds'],limits['local_internal_seconds'],
        limits['external_wall_seconds'],limits['prep_window_seconds'])==(3200,3300,3350,3415)
    assert limits['local_full_vectors_total']==512
    assert limits['already_allocated_gpu_seconds']==4586
    assert limits['cumulative_preparation_limit_seconds']==8001
    assert adapter.local.GpuDriver is HybridDriver


def test_new_limits_change_time_and_cumulative_work_not_physics_or_memory(adapter):
    before=dict(adapter.local.LIMITS)
    adapter.extend_execution_limits(8001)
    limits=adapter.local.LIMITS
    assert (limits['interval_seconds'],limits['local_internal_seconds'],
        limits['external_wall_seconds'],limits['prep_window_seconds'],
        limits['slurm_allocation_seconds'])==(14400,14500,14600,14700,14700)
    assert limits['already_allocated_gpu_seconds']==8001
    assert limits['cumulative_preparation_limit_seconds']==22701
    assert limits['local_full_vectors_total']==4096
    for key in ('source_relative','spatial_relative','fine_each_output','require_cost_target',
        'cost_target_seconds','cpus','GPU','host_GiB','device_GiB','cache_GiB',
        'automatic_retries','local_calls','actual_model_F_calls','SE_RE_updates','carried_parallel_work'):
        assert limits[key]==before[key],key
    assert adapter.local.GpuDriver.interval_limit_seconds==14400.
    assert HybridDriver.interval_limit_seconds==3420.
    layout=adapter.local.GpuDriver.__new__(adapter.local.GpuDriver).layout_for(1115020,256)
    assert layout['workspace']==1070005192<1024**3


def test_new_work_limit_permits_more_work_without_allocating_a_larger_cache(adapter):
    adapter.extend_execution_limits(7000)
    limits=adapter.local.LIMITS;nf=1115020
    work=Work(limits['local_full_vectors_total']*nf-limits['carried_parallel_work'],
        time.monotonic()+2)
    work.take('map_queries',1024*nf)
    assert work.used==1024*nf>512*nf
    assert work.maximum+limits['carried_parallel_work']==4096*nf
    with pytest.raises(RuntimeError,match='작업량 한도'):
        work.take('map_queries',work.maximum-work.used+1)


def test_long_ceiling_preserves_physics_memory_and_sealed_four_hour_mode(adapter):
    before=dict(adapter.local.LIMITS)
    adapter.extend_execution_limits(22264,long_run=True)
    limits=adapter.local.LIMITS
    assert (limits['interval_seconds'],limits['local_internal_seconds'],
        limits['external_wall_seconds'],limits['prep_window_seconds'],
        limits['slurm_allocation_seconds'])==(86000,86100,86200,86400,86400)
    assert limits['local_full_vectors_total']==16384
    assert limits['already_allocated_gpu_seconds']==22264
    assert limits['cumulative_preparation_limit_seconds']==108664
    assert adapter.local.GpuDriver.interval_limit_seconds==86000.
    assert HybridDriver.interval_limit_seconds==3420.
    for key in ('source_relative','spatial_relative','fine_each_output','require_cost_target',
        'cost_target_seconds','cpus','GPU','host_GiB','device_GiB','cache_GiB',
        'automatic_retries','local_calls','actual_model_F_calls','SE_RE_updates','carried_parallel_work'):
        assert limits[key]==before[key],key
    layout=adapter.local.GpuDriver.__new__(adapter.local.GpuDriver).layout_for(1115020,256)
    assert layout['workspace']==1070005192<1024**3
    nf=1115020
    work=Work(limits['local_full_vectors_total']*nf-limits['carried_parallel_work'],
        time.monotonic()+2)
    work.take('map_queries',4096*nf)
    assert work.maximum+limits['carried_parallel_work']==16384*nf


@pytest.mark.parametrize('previous',[True,None,4586.,-1,0,22263])
def test_long_execution_requires_settled_prior_usage(adapter,previous):
    before=dict(adapter.local.LIMITS)
    with pytest.raises(ValueError):
        adapter.extend_execution_limits(previous,long_run=True)
    assert adapter.local.LIMITS==before


@pytest.mark.parametrize('previous',[True,None,4586.,-1,0,4585])
def test_extended_execution_cannot_reset_previous_usage(adapter,previous):
    before=dict(adapter.local.LIMITS)
    with pytest.raises(ValueError,match='실제 누적 GPU초'):
        adapter.extend_execution_limits(previous)
    assert adapter.local.LIMITS==before
