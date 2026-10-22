"""동일 ABI라도 서로 다른 원천 적분기의 조용한 혼용을 거부한다."""
from types import SimpleNamespace
import ctypes as ct
import importlib.util
import os
from pathlib import Path
import sys

import pytest
from lumina_gpu_compact import EndpointCompactDriver
from lumina_gpu_native_bin import NativeBinDriver
from test_lumina_gpu_measure_local import local


class Function:
    def __init__(self,value):self.value=value
    def __call__(self,*args):return self.value


@pytest.mark.parametrize('driver,marker',[(NativeBinDriver,None),(NativeBinDriver,0),(EndpointCompactDriver,1)])
def test_integrator_identity_mismatch_is_rejected_before_allocation(driver,marker):
    lib=SimpleNamespace(lumina_linear_cuda_node_bytes=Function(1128),
        lumina_linear_cuda_map_bytes=Function(56),lumina_linear_cuda_endpoint_bytes=Function(4*17*8))
    if marker is not None:lib.lumina_linear_cuda_source_integrator=Function(marker)
    with pytest.raises(ValueError,match='원천 적분기'):
        driver(SimpleNamespace(lib=lib),SimpleNamespace(nf=17),'unused.so')


def test_default_and_native_limits_and_memory_policy_are_not_silently_changed():
    assert NativeBinDriver.source_integrator==1 and EndpointCompactDriver.source_integrator==0
    assert NativeBinDriver.interval_limit_seconds==3400. and EndpointCompactDriver.interval_limit_seconds==60.
    assert NativeBinDriver.__new__(NativeBinDriver).layout_for(1115020,256)['workspace']==1048771496


def test_explicit_host_only_compiled_integrator_identity():
    path=os.environ.get('LUMINA_NATIVE_BIN_PLAN_LIBRARY')
    if not path:pytest.skip('별도 strict 빌드의 host ABI 선택 필요')
    lib=ct.CDLL(path);marker=lib.lumina_linear_cuda_source_integrator
    marker.argtypes=[];marker.restype=ct.c_int
    assert marker()==1


def test_native_actual_limits_and_final_physics_scope(local,monkeypatch,tmp_path):
    monkeypatch.setitem(sys.modules,'lumina_gpu_local',local)
    path=Path(__file__).resolve().parents[1]/'scripts/lumina_gpu_native_actual.py'
    spec=importlib.util.spec_from_file_location('isolated_native_actual',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    monkeypatch.setattr(m,'evidence',lambda root:None)
    monkeypatch.setattr(local.M,'N',SimpleNamespace(sha=lambda path:'sealed'),raising=False)
    calls=[];monkeypatch.setattr(local,'prepare',lambda root,**kw:calls.append(kw));m.prepare(tmp_path)
    assert calls[0]['interval_indices']==(13003,)
    assert local.LIMITS['already_allocated_gpu_seconds']==1051
    assert local.LIMITS['cumulative_preparation_limit_seconds']==4651
    assert local.LIMITS['original_preparation_budget_seconds']==1800 and local.LIMITS['user_extended_time_budget']
    assert local.LIMITS['interval_seconds']==3400 and local.LIMITS['local_internal_seconds']==3500
    assert local.LIMITS['external_wall_seconds']==3550 and local.LIMITS['slurm_allocation_seconds']==3600
    assert local.LIMITS['require_cost_target'] is False and local.LIMITS['cost_target_seconds']==10
    assert local.LIMITS['source_relative']==1e-8 and local.LIMITS['spatial_relative']==1e-5
    assert local.LIMITS['local_full_vectors_total']==512 and local.LIMITS['fine_each_output']
    assert local.LIMITS['host_GiB']==local.LIMITS['device_GiB']==16 and local.LIMITS['GPU']==1
    assert local.LIMITS['local_calls']==1 and local.LIMITS['automatic_retries']==0
    assert not local.LIMITS['actual_model_F_calls'] and not local.LIMITS['SE_RE_updates']
