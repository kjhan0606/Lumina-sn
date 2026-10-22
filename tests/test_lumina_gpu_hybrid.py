"""두 적분기의 실패 분류·합산 예산·미완 태그를 시험한다. 실제장 증거가 아니다."""
import ctypes as ct
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
from lumina_gpu_hybrid import HybridDriver
from lumina_gpu_compact import WarpCompactDriver
from test_lumina_gpu_native_bin import Function
from test_lumina_gpu_measure_local import local


@pytest.fixture(scope='module')
def policy(tmp_path_factory):
    root=Path(__file__).resolve().parents[1];out=tmp_path_factory.mktemp('hybrid')/'probe.so'
    p=subprocess.run([os.environ.get('CXX','c++'),'-O2','-std=c++17','-ffp-contract=off','-Wall','-Wextra','-Werror',
        '-fPIC','-shared','-I'+str(root/'src'),str(root/'tests/hybrid_measure_probe.cpp'),'-o',str(out)],
        capture_output=True,text=True,timeout=60)
    assert p.returncode==0,p.stderr
    f=ct.CDLL(str(out)).hybrid_policy_probe
    f.argtypes=[ct.c_int]*3+[np.ctypeslib.ndpointer(dtype='i4',flags='C_CONTIGUOUS')];f.restype=ct.c_int
    def call(total,code,damage=0):
        values=np.full(5,-17,dtype='i4');assert f(total,code,damage,values)==0;return values
    return call


def test_two_attempts_never_spend_more_than_original_panel_budget(policy):
    for total in (1,2,3,99,100,99999,100000):
        p,s,*_=policy(total,3)
        assert p>=1 and s>=0 and p+s==total
    for total in (-1,0,100001):assert policy(total,3)[:2].tolist()==[0,0]


@pytest.mark.parametrize('code',range(-1,10))
def test_only_explicit_limit_codes_are_pending_not_bad_input_or_numeric_range(policy,code):
    values=policy(100000,code);allowed=3<=code<=7
    assert bool(values[2])==bool(values[3])==allowed
    assert values[4]==(code if allowed else 0)


@pytest.mark.parametrize('damage',[1,2,3])
def test_incomplete_tag_damage_cannot_be_consumed_as_a_physical_map(policy,damage):
    assert policy(100000,5,damage)[4]==-1


def test_hybrid_state_and_cooperative_memory_are_counted():
    layout=HybridDriver.__new__(HybridDriver).layout_for(1115020,256)
    assert layout['hybrid_counters']==32 and layout['workspace']==1070005192<1024**3
    assert layout['cooperative_shared']==WarpCompactDriver.cooperative_shared_bytes
    assert HybridDriver.source_integrator==2


@pytest.mark.parametrize('marker',[None,0,1])
def test_wrong_integrator_binary_rejected_before_allocation(marker):
    lib=SimpleNamespace(lumina_linear_cuda_node_bytes=Function(1128),lumina_linear_cuda_map_bytes=Function(56),
        lumina_linear_cuda_endpoint_bytes=Function(4*17*8))
    if marker is not None:lib.lumina_linear_cuda_source_integrator=Function(marker)
    with pytest.raises(ValueError,match='원천 적분기'):
        HybridDriver(SimpleNamespace(lib=lib),SimpleNamespace(nf=17),'unused.so')


def test_actual_adapter_carries_cost_and_preserves_original_accuracy(local,monkeypatch,tmp_path):
    import importlib.util
    import sys
    monkeypatch.setitem(sys.modules,'lumina_gpu_local',local)
    for name in ('lumina_gpu_native_actual','lumina_gpu_prefix_actual'):
        monkeypatch.delitem(sys.modules,name,raising=False)
    path=Path(__file__).resolve().parents[1]/'scripts/lumina_gpu_hybrid_actual.py'
    spec=importlib.util.spec_from_file_location('isolated_hybrid_actual',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'evidence',lambda root:None)
    monkeypatch.setattr(local.M,'N',SimpleNamespace(sha=lambda path:'sealed'),raising=False)
    calls=[];monkeypatch.setattr(local,'prepare',lambda root,**kw:calls.append(kw));module.prepare(tmp_path)
    assert calls[0]['interval_indices']==(13003,)
    assert local.GpuDriver is HybridDriver
    assert local.LIMITS['already_allocated_gpu_seconds']==4341
    assert local.LIMITS['cumulative_preparation_limit_seconds']==4341+3600
    assert (local.LIMITS['interval_seconds'],local.LIMITS['local_internal_seconds'],
        local.LIMITS['external_wall_seconds'],local.LIMITS['slurm_allocation_seconds'])==(3420,3480,3500,3600)
    assert local.LIMITS['source_relative']==1e-8 and local.LIMITS['spatial_relative']==1e-5
    assert local.LIMITS['fine_each_output'] and not local.LIMITS['require_cost_target']
    assert local.LIMITS['local_full_vectors_total']==512 and local.LIMITS['carried_parallel_work']==8*1115020
    assert local.LIMITS['GPU']==1 and local.LIMITS['cpus']==1
    assert local.LIMITS['host_GiB']==local.LIMITS['device_GiB']==16
    assert local.LIMITS['automatic_retries']==local.LIMITS['actual_model_F_calls']==local.LIMITS['SE_RE_updates']==0
