"""GPU나 수송 수치 없이 전체장 시간 전달·미완 원자성만 시험한다."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import lumina_gpu_driver as G
from lumina_gpu_native_bin import NativeBinDriver


@pytest.fixture
def field_double(monkeypatch):
    clock=SimpleNamespace(now=100.)
    monkeypatch.setattr(G,'time',SimpleNamespace(monotonic=lambda:clock.now))
    class Runtime:
        def upload(self,*args):pass
        def zero(self,*args):pass
        def sync(self):pass
        def download(self,buffer,shape):return np.zeros(shape)
        def report(self):return dict(test_double=True)
    class Driver(NativeBinDriver):
        interval_limit_seconds=3240.
        def __init__(self):
            self.nf,self.ns,self.width=8,2,4;self.provider=SimpleNamespace(_source=object())
            self.rt=Runtime();self.moments=None;self.ray=object();self.generation=7
            self.input_identity=dict(test_double=True);self.calls=[];self.allocations=0
            self.elapsed_per_interval=0.;self.fail_at=None
        def bound(self):pass
        def prepare_events(self,plan):pass
        def alloc(self,size):self.allocations+=1;return object()
        def accumulate(self,*args):pass
        def interval(self,plan,index,work,*,max_seconds):
            self.calls.append((index,max_seconds))
            if self.fail_at==index:raise RuntimeError('의도한 미완 구간')
            clock.now+=self.elapsed_per_interval
            return dict(limited_polynomials=0,rounds=1,max_mesh_leaves=1,global_tolerance=1.,global_error=0.)
    d=Driver()
    plan=SimpleNamespace(source=d.provider._source,panels=1,driver_block_width=4,
        arrays=dict(paths=np.array([[0,2,2,0,0]]),steps=np.zeros((2,1)),event_ptr=np.zeros(3,dtype='i4')))
    return d,plan,clock


def invoke(case,**kwargs):
    d,p,_clock=case
    return d.field(p,np.ones(8),np.zeros(8),**kwargs)


def test_existing_default_stays_at_sixty_seconds(field_double):
    d,_p,_clock=field_double;_moments,report=invoke(field_double)
    assert d.calls==[(0,60.),(1,60.)]
    assert report['interval_seconds_limit']==60. and report['total_seconds_limit']==5400.
    assert report['accepted_physical_field'] is False and report['final_goal']=='NOT_MET'


def test_long_interval_is_forwarded_without_extending_the_whole_field(field_double):
    d,_p,_clock=field_double;_moments,report=invoke(field_double,interval_seconds=3240.)
    assert d.calls==[(0,3240.),(1,3240.)] and report['total_seconds_limit']==5400.


def test_remaining_field_budget_clips_each_interval(field_double):
    d,_p,_clock=field_double;d.elapsed_per_interval=45.
    with pytest.raises(TimeoutError):invoke(field_double,seconds=70.,interval_seconds=3240.)
    assert d.calls==[(0,70.),(1,25.)]


def test_expired_field_does_not_start_another_interval(field_double):
    d,_p,_clock=field_double;d.elapsed_per_interval=70.
    with pytest.raises(TimeoutError):invoke(field_double,seconds=70.,interval_seconds=3240.)
    assert d.calls==[(0,70.)]


@pytest.mark.parametrize('limit',[0.,-1.,float('nan'),float('inf'),True,'3240',3241.])
def test_bad_or_driver_unsupported_interval_rejected_before_allocation(field_double,limit):
    d,_p,_clock=field_double
    with pytest.raises(ValueError,match='구간 시간'):invoke(field_double,interval_seconds=limit)
    assert not d.allocations and not d.calls


def test_whole_field_cap_and_failed_interval_still_do_not_return_a_field(field_double):
    d,_p,_clock=field_double
    with pytest.raises(ValueError):invoke(field_double,seconds=5401.,interval_seconds=3240.)
    assert not d.allocations
    d.fail_at=1
    with pytest.raises(RuntimeError,match='미완 구간'):invoke(field_double,interval_seconds=3240.)
    assert len(d.calls)==2
