"""입사 오차의 실제 replay 전파. 재구성/관측/전체장 수용을 대신하지 않는다."""
import ctypes as ct
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from lumina_gpu_compact import CompactMap
from lumina_linear_native import CPositive
from lumina_gpu_driver import Work
from lumina_gpu_linear import LinearDriver


@pytest.fixture(scope='module')
def replay(tmp_path_factory):
    library=tmp_path_factory.mktemp('ray-error-replay')/'probe.so'
    command=[os.environ.get('CXX','c++'),'-O2','-std=c++17','-ffp-contract=off','-Wall','-Wextra','-Werror',
        '-fPIC','-shared','-I'+str(ROOT/'src'),str(ROOT/'tests/linear_affine_probe.cpp'),'-o',str(library)]
    p=subprocess.run(command,capture_output=True,text=True,timeout=60)
    assert p.returncode==0,p.stderr
    lib=ct.CDLL(str(library));array=np.ctypeslib.ndpointer(dtype='f8',flags='C_CONTIGUOUS')
    original=lib.lumina_compact_affine_probe
    original.argtypes=[ct.c_int]*2+[ct.POINTER(CompactMap),array,ct.c_double]+[array]*4
    original.restype=ct.c_int
    uncertain=lib.lumina_compact_affine_uncertain_probe
    uncertain.argtypes=original.argtypes+[ct.POINTER(CPositive)];uncertain.restype=ct.c_int
    def call(maps,initial,incoming=None):
        initial=np.ascontiguousarray(initial,dtype='f8');width=len(initial);n=len(maps)//(3*width)
        output,error,ratio=[np.full(width,-17.) for _ in range(3)];scores=np.full(n*width,-17.)
        args=(n,width,maps,initial,1e-5,output,error,ratio,scores)
        code=original(*args) if incoming is None else uncertain(*args,incoming)
        return code,output,error,ratio,scores
    return call


def positive(value):
    mantissa,exponent=math.frexp(value);return CPositive(mantissa,exponent)


def transfer(value=1.,source=0.,tau_error=0.):
    return CompactMap(positive(value),positive(source),CPositive(),tau_error)


@pytest.mark.parametrize('intensity',[1e-280,1.,1e280])
def test_forty_locally_passing_intervals_retain_the_final_bias(replay,intensity):
    delta=3e-6;t=1.+delta;root=math.sqrt(t)
    maps=(CompactMap*3)(transfer(t,tau_error=math.log1p(delta)),
        transfer(root,tau_error=.5*math.log1p(delta)),transfer(root,tau_error=.5*math.log1p(delta)))
    value=intensity;incoming=0.
    for _ in range(40):
        code,out,error,ratio,_scores=replay(maps,[value],(CPositive*1)(positive(incoming)))
        assert code==0 and ratio[0]<1
        value,incoming=float(out[0]),float(error[0])
    assert abs(value-intensity)>intensity*1e-4
    assert abs(value-intensity)<=incoming


def test_zero_incident_is_bit_identical_and_inherited_error_does_not_force_local_refinement(replay):
    maps=(CompactMap*6)(transfer(2.),transfer(2.),transfer(4.),transfer(4.),
        transfer(.5),transfer(.5))
    baseline=replay(maps,[3.,0.]);zero=replay(maps,[3.,0.],(CPositive*2)())
    for a,b in zip(baseline[1:],zero[1:]):np.testing.assert_array_equal(a,b)
    incoming=(CPositive*2)(positive(7.),positive(1e-200))
    result=replay(maps,[3.,0.],incoming)
    assert result[0]==0
    np.testing.assert_array_equal(result[1],baseline[1])
    np.testing.assert_array_equal(result[3],baseline[3])
    np.testing.assert_array_equal(result[4],baseline[4])
    assert result[2][0]>=14. and result[2][1]>=2e-200
    assert result[1][1]==0. and result[2][1]>0.  # I=0이어도 E>0을 지우지 않는다.


@pytest.mark.parametrize('incoming',[CPositive(-.5,1),CPositive(float('nan'),0),CPositive(.25,0),
    CPositive(.5,1<<60)])
def test_invalid_incident_uncertainty_does_not_publish_a_partial_vector(replay,incoming):
    maps=(CompactMap*6)(*[transfer() for _ in range(6)])
    result=replay(maps,[1.,2.],(CPositive*2)(positive(0.),incoming))
    assert result[0]!=0
    for array in result[1:]:np.testing.assert_array_equal(array,np.full(len(array),-17.))


@pytest.mark.parametrize('with_incident',[False,True])
def test_python_dispatch_uses_the_same_frequency_slice_without_new_allocation(with_incident):
    class Function:
        def __init__(self):self.calls=[]
        def __call__(self,*args):self.calls.append(args);return 0
    runtime=SimpleNamespace(status=19,upload=lambda *_:None)
    class Region:
        def __init__(self,address,size=4096):self.address=address;self.size=size;self.owner=runtime
        def part(self,start,size):
            assert 0<=start<=self.size-size
            return Region(self.address+start,size)
    d=LinearDriver.__new__(LinearDriver);d.rt=runtime;d.width=4
    normal=Function();uncertain=Function()
    d.lib=SimpleNamespace(lumina_linear_cuda_replay=normal,lumina_linear_cuda_replay_uncertain=uncertain)
    for i,name in enumerate(('pointers','characteristic','candidate','results','scores','errors','frequency_errors'),1):
        setattr(d,name,Region(i*10000))
    incoming=Region(80000);current=[SimpleNamespace(cache=(0,Region(90000)))]
    work=Work(100,time.monotonic()+2)
    d.replay(current,4,3,work,incident_errors=incoming if with_incident else None)
    chosen=uncertain if with_incident else normal
    assert len(chosen.calls)==1 and not (normal if with_incident else uncertain).calls
    args=chosen.calls[0]
    assert args[:5]==(1,3,3,4,1e-5) and args[-1]==19
    assert args[6]==20032 and args[7]==30032 and args[11]==70064
    if with_incident:assert args[12]==80064
    assert work.used==3
