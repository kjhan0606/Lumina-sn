"""수치 GPU 호출 없이 참조 비교·선택 구간·작업량 연결만 확인한다."""
import importlib.util
from pathlib import Path
import sys
import time
from types import SimpleNamespace,ModuleType

import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from lumina_gpu_measure import MeasureDriver
from lumina_gpu_linear import Piece
from lumina_gpu_driver import Work


@pytest.fixture
def local(monkeypatch):
    for name,field in (('lumina_local_micro',None),('lumina_compiled_input','CompiledInput'),('lumina_nodal_driver','CompiledDriverPlan')):
        module=ModuleType(name)
        if field:setattr(module,field,object)
        monkeypatch.setitem(sys.modules,name,module)
    spec=importlib.util.spec_from_file_location('isolated_gpu_local',ROOT/'scripts/lumina_gpu_local.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def test_faint_point_mismatch_cannot_hide_behind_bright_maximum(local):
    fine=np.array([100.,1e-10]);actual=np.array([100.,1.01e-10])
    prior=local.fine_comparison(actual,fine);assert prior['accuracy_met']
    record=local.fine_comparison(actual,fine,each_output=True)
    assert record['fine_max_norm_difference']<1e-4
    assert record['fine_relative_differences'][1]>.009
    assert not record['accuracy_met']
    assert local.fine_comparison(fine,fine,each_output=True)['accuracy_met']


def test_measure_adapter_selects_only_13003_and_preserves_limits(local,monkeypatch,tmp_path):
    monkeypatch.setitem(sys.modules,'lumina_gpu_local',local)
    spec=importlib.util.spec_from_file_location('isolated_measure_local',ROOT/'scripts/lumina_gpu_measure_local.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    monkeypatch.setattr(m,'evidence',lambda root:None)
    calls=[];monkeypatch.setattr(local,'prepare',lambda root,**kw:calls.append(kw))
    m.prepare(tmp_path)
    assert calls[0]['interval_indices']==(13003,)
    assert local.LIMITS['local_calls']==1 and local.LIMITS['automatic_retries']==0
    assert local.LIMITS['local_full_vectors_total']==512 and local.LIMITS['cost_target_seconds']==10
    assert local.LIMITS['fine_each_output'] and local.LIMITS['actual_model_F_calls']==0
    assert local.LIMITS['prep_window_seconds']==1794


def test_measure_group_counts_both_source_and_chi_under_same_work_cap():
    class Fake:
        def lumina_linear_cuda_plan(self,*args):
            p=args[-1]._obj;p.leaves=2;p.bins=33;p.nf=35
            return 0
        def lumina_linear_cuda_build(self,*args):return 0
        def lumina_linear_cuda_query(self,*args):return 0
    class Target:
        address=1
        def part(self,*args):return self
    d=MeasureDriver.__new__(MeasureDriver);d.lib=Fake();d.view=Piece().view;d.rt=SimpleNamespace(status=1)
    d.provider=SimpleNamespace(_source=SimpleNamespace(dlognu=.001));d.tree=SimpleNamespace(size=100000,address=1)
    from lumina_gpu_driver import Location
    d.piece=lambda *a:([Location(1,1,0,.3)]*3,1.,.002,.001)
    w=Work(10000,time.monotonic()+60)
    d.group((None,None,None,None),0,0,8,Target(),w)
    assert w.counts['source_projected_vertices']==3*(34+2)
    assert w.counts['source_projection_nodes']==6 and w.counts['source_tree_compositions']==3
    assert w.counts['bin_maps']==3*34 and w.used==sum(w.counts.values())
