"""B GPU 공통 합성 수학·출력별 기준·메모리·ABI를 CPU에서 독립 대조한다."""
import ctypes as ct
import importlib
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from lumina_gpu_linear import MAP_BYTES, NODE_BYTES, LOCAL_BYTES, Piece, Runtime, memory_layout
from lumina_linear_native import CPositive, Map, NativeMap, Positive
import lumina_gpu_linear as gpu_linear


@pytest.fixture(scope='module')
def affine(tmp_path_factory):
    path=tmp_path_factory.mktemp('linear-affine')/'probe.so'
    p=subprocess.run([os.environ.get('CXX','c++'),'-O2','-std=c++17','-ffp-contract=off','-Wall','-Wextra','-Werror',
        '-fPIC','-shared','-I'+str(ROOT/'src'),str(ROOT/'tests/linear_affine_probe.cpp'),'-o',str(path)],
        capture_output=True,text=True,timeout=60)
    assert p.returncode==0,p.stderr
    f=ct.CDLL(str(path)).lumina_linear_affine_probe
    d=np.ctypeslib.ndpointer(dtype='f8',flags='C_CONTIGUOUS')
    f.argtypes=[ct.c_int,ct.c_int,ct.POINTER(NativeMap),d,ct.c_double,d,d,d,d];f.restype=ct.c_int
    return f


def raw(m):
    value=NativeMap()
    for key,part in (('transmission',m.t),('source',m.source),('source_error',m.source_error)):
        setattr(value,key,CPositive(part.m,part.e))
    value.tau_error=m.tau_error
    return value


def run(fun,groups,initial,relative=1e-5):
    n=len(groups);width=len(initial);data=(NativeMap*(3*n*width))()
    for row,g in enumerate(groups):
        for k in range(3):
            for f in range(width):data[(row*3+k)*width+f]=raw(g[k][f])
    initial=np.ascontiguousarray(initial,dtype='f8');o=np.full(width,-17.);e=o.copy();r=o.copy();s=np.full((n,width),-17.)
    code=fun(n,width,data,initial,relative,o,e,r,s)
    return code,o,e,r,s


def test_temporal_order_and_errors_match_separate_python_positive_composition(affine):
    rng=np.random.default_rng(102204);n,width=6,17;groups=[]
    for _ in range(n):
        left=[Map(Positive.of(float(rng.uniform(.3,1.4))),Positive.of(float(rng.uniform(.1,1.))),Positive.of(1e-12),2e-14) for _ in range(width)]
        right=[Map(Positive.of(float(rng.uniform(.3,1.4))),Positive.of(float(rng.uniform(.1,1.))),Positive.of(1e-12),3e-14) for _ in range(width)]
        groups.append(([a.then(b) for a,b in zip(left,right)],left,right))
    initial=rng.uniform(0.,1.,width);code,value,error,ratios,scores=run(affine,groups,initial);assert code==0
    for f in range(width):
        current=Positive.of(float(initial[f]));numerical=Positive();geometry=[];responses=[]
        for coarse,left,right in groups:
            fine=left[f].then(right[f]);cv,ce=coarse[f].apply(current);fv,fe=fine.apply(current)
            numerical=fine.upper_t*numerical+fe
            geometry.append(cv.difference(fv)+ce+fe+(cv+fv)*(16*np.finfo('f8').eps));responses.append(fine.upper_t);current=fv
        tail=Positive.of(1.);g=Positive()
        for j in range(n-1,-1,-1):g=g+geometry[j]*tail;tail=tail*responses[j]
        expected=(g+numerical+current*(64*(n+1)*np.finfo('f8').eps)).value()
        assert value[f]==pytest.approx(current.value(),rel=3e-15)
        assert error[f]==pytest.approx(expected,rel=3e-5,abs=1e-14)
        assert ratios[f]==pytest.approx(error[f]/(1e-5*value[f]),rel=2e-14)
    assert (scores>=0).all()


def test_intermediate_transmission_underflow_can_recover_later(affine):
    z=Positive();one=Positive.of(1.)
    a=Map(Positive(.5,-1599),z,z,0.);b=Map(Positive(.5,1601),z,z,0.)
    c=Map(one,z,z,0.);code,value,error,ratios,scores=run(affine,[([c],[a],[b])],[2.])
    assert code==0 and value[0]==2. and 0<error[0]<1e-10 and ratios[0]<1.


def test_weak_output_does_not_pass_using_another_bright_output(affine):
    z=Positive();one=Positive.of(1.)
    def m(s):return Map(one,Positive.of(s),z,0.)
    code,value,error,ratios,scores=run(affine,[([m(1.01e-8),m(1e10)],[m(0.),m(0.)],[m(1e-8),m(1e10)])],[0.,0.])
    assert code==0 and value.tolist()==[1e-8,1e10]
    assert ratios[0]>100 and ratios[1]<1 and scores[0,0]>100


def test_exact_zero_output_has_exact_zero_error_and_ratio(affine):
    m=Map(Positive.of(.8),Positive(),Positive(),0.)
    code,value,error,ratios,scores=run(affine,[([m],[m],[m])],[0.])
    assert code==0 and value[0]==error[0]==ratios[0]==scores[0,0]==0.


def test_invalid_map_does_not_publish_partial_cpu_results(affine):
    good=Map(Positive.of(1.),Positive.of(1.),Positive(),0.)
    # C ABI로 들어오는 비정규 양의 값을 직접 넣어 실패 원자성을 확인한다.
    data=(NativeMap*3)(raw(good),raw(good),raw(good));data[2].source.m=-1.
    a=np.zeros(1);o=np.full(1,-17.);e=o.copy();r=o.copy();s=o.copy()
    assert affine(1,1,data,a,1e-5,o,e,r,s)!=0
    assert o[0]==e[0]==r[0]==s[0]==-17.


@pytest.mark.parametrize('nf,width',[(8,1),(65,16),(1115020,256)])
def test_layout_includes_maps_tree_replay_rays_errors_and_weight_consumer(nf,width):
    layout=memory_layout(nf,width)
    assert layout['workspace']<=1024**3 and layout['arena']>=3*nf*MAP_BYTES
    assert layout['results']>=nf*LOCAL_BYTES and layout['slots']<=4096
    assert NODE_BYTES==272 and MAP_BYTES==144 and ct.sizeof(Piece)==192


@pytest.mark.parametrize('nf,width',[(2,256),(2000001,256),(1115020,0),(2000000,256)])
def test_impossible_layout_or_input_is_rejected(nf,width):
    with pytest.raises((ValueError,MemoryError)):memory_layout(nf,width)


def test_import_and_runtime_abi_definition_do_not_create_cuda_context(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('import 중 공유 라이브러리 로드 금지')
    monkeypatch.setattr(ct,'CDLL',forbidden)
    importlib.reload(gpu_linear)


@pytest.mark.parametrize('failure', ['bound','validation','geometry'])
def test_new_interval_failure_cannot_report_previous_interval_progress(failure):
    from types import SimpleNamespace
    driver = gpu_linear.LinearDriver.__new__(gpu_linear.LinearDriver)
    driver.last_completed_batch = dict(interval_index=99, maximum_error_ratio=.2)
    source = object()
    driver.provider = SimpleNamespace(_source=source)
    driver.nf = driver.width = 8

    def fail(*args):
        raise RuntimeError('새 구간 초기 실패')

    driver.bound = fail if failure == 'bound' else lambda: None
    driver.geometry = fail
    plan = SimpleNamespace(source=source, panels=0 if failure == 'validation' else 1,
                           arrays={'steps':np.zeros((1,5))})
    work = SimpleNamespace(deadline=float('inf'), check=lambda: None)
    with pytest.raises((RuntimeError,ValueError)):
        driver.interval(plan,0,work,max_seconds=1.)
    assert driver.last_completed_batch is None
