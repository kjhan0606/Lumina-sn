"""실제 실패 수정의 한도·원 정확도·비용 분리 정책을 고정한다."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

from test_lumina_gpu_measure_local import local


def test_prefix_actual_preserves_original_physics_and_remaining_time(local,monkeypatch,tmp_path):
    monkeypatch.setitem(sys.modules,'lumina_gpu_local',local)
    monkeypatch.delitem(sys.modules,'lumina_gpu_native_actual',raising=False)
    path=Path(__file__).resolve().parents[1]/'scripts/lumina_gpu_prefix_actual.py'
    spec=importlib.util.spec_from_file_location('isolated_prefix_actual',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    monkeypatch.setattr(m,'evidence',lambda root:None)
    monkeypatch.setattr(local.M,'N',SimpleNamespace(sha=lambda path:'sealed'),raising=False)
    calls=[];monkeypatch.setattr(local,'prepare',lambda root,**kw:calls.append(kw));m.prepare(tmp_path)
    assert calls[0]['interval_indices']==(13003,)
    assert local.LIMITS['already_allocated_gpu_seconds']==1078
    assert local.LIMITS['slurm_allocation_seconds']==3480 and local.LIMITS['interval_seconds']==3240
    assert local.LIMITS['local_internal_seconds']==3340 and local.LIMITS['external_wall_seconds']==3420
    assert 1078+60+3480<=local.LIMITS['cumulative_preparation_limit_seconds']==4651
    assert local.LIMITS['require_cost_target'] is False and local.LIMITS['cost_target_seconds']==10
    assert local.LIMITS['source_relative']==1e-8 and local.LIMITS['spatial_relative']==1e-5
    assert local.LIMITS['fine_each_output'] and local.LIMITS['local_full_vectors_total']==512
    assert local.LIMITS['GPU']==1 and local.LIMITS['cpus']==1
    assert local.LIMITS['host_GiB']==local.LIMITS['device_GiB']==16
    assert not local.LIMITS['actual_model_F_calls'] and not local.LIMITS['SE_RE_updates']
    assert local.LIMITS['automatic_retries']==0 and local.LIMITS['local_calls']==1
