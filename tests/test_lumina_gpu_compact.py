"""축약 오차 소비와 독립 χ/η 트리를 CPU에서 대조한다. import는 GPU를 열지 않는다."""
import ctypes as ct
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
from lumina_gpu_compact import CompactMap,CompactDriver,MAP_BYTES,memory_layout
from lumina_gpu_linear import Piece
from lumina_linear_native import NativeMap,CPositive
from test_lumina_characteristic_linear import reference


@pytest.fixture(scope='module')
def kernels(tmp_path_factory):
    path=tmp_path_factory.mktemp('compact-affine')/'probe.so'
    p=subprocess.run([os.environ.get('CXX','c++'),'-O2','-std=c++17','-ffp-contract=off',
        '-Wall','-Wextra','-Werror','-fPIC','-shared','-I'+str(ROOT/'src'),
        str(ROOT/'tests/linear_affine_probe.cpp'),str(ROOT/'tests/source_measure_probe.cpp'),'-o',str(path)],
        capture_output=True,text=True,timeout=60)
    assert p.returncode==0,p.stderr
    lib=ct.CDLL(str(path));a=np.ctypeslib.ndpointer(dtype='f8',flags='C_CONTIGUOUS')
    for name,raw in [('lumina_linear_affine_probe',NativeMap),('lumina_compact_affine_probe',CompactMap)]:
        f=getattr(lib,name);f.argtypes=[ct.c_int,ct.c_int,ct.POINTER(raw),a,ct.c_double,a,a,a,a];f.restype=ct.c_int
    f=lib.source_measure_split_probe
    f.argtypes=[ct.c_int]*3+[a,a,ct.c_int,a,np.ctypeslib.ndpointer(dtype='i8',flags='C_CONTIGUOUS'),
                           np.ctypeslib.ndpointer(dtype='i4',flags='C_CONTIGUOUS')];f.restype=ct.c_int
    return lib


def packed(rich):
    return (CompactMap*len(rich))(*(CompactMap(x.transmission,x.source,x.source_error,x.tau_error) for x in rich))


def replay(fun,data,n,width,initial):
    arrays=[np.full(width,-17.) for _ in range(3)]+[np.full((n,width),-17.)]
    code=fun(n,width,data,np.ascontiguousarray(initial,dtype='f8'),1e-5,*arrays)
    return code,arrays


def test_compact_ABI_and_native_workspace_count_every_live_region():
    assert MAP_BYTES==56 and ct.sizeof(CompactMap)==56 and ct.sizeof(NativeMap)==144
    assert ct.sizeof(Piece)==192 and Piece.source_leaves.offset==188
    layout=memory_layout(1115020,256)
    assert layout['chi_tree']==570425344 and layout['source_tree']==112197632
    assert layout['arena']>=3*1115020*56 and layout['results']>=1115020*64
    assert layout['workspace']==1013090856<1024**3
    with pytest.raises(MemoryError):memory_layout(2000000,256)
    with pytest.raises(ValueError):memory_layout(True,256)


def test_wrong_rich_binary_is_rejected_before_allocation():
    lib=SimpleNamespace(lumina_linear_cuda_node_bytes=lambda:1128,lumina_linear_cuda_map_bytes=lambda:144)
    with pytest.raises(ValueError,match='맵 저장 구조'):
        CompactDriver(SimpleNamespace(lib=lib),SimpleNamespace(nf=17),'unused.so')


def test_split_group_uses_own_leaf_counts_and_compact_byte_offsets():
    from lumina_gpu_driver import Work,Location
    offsets=[]
    class Fake:
        def lumina_linear_cuda_plan(self,*args):
            assert args[-2]==2
            p=args[-1]._obj;p.leaves=32;p.source_leaves=2;p.bins=33;p.nf=35
            return 0
        def lumina_linear_cuda_build(self,*args):return 0
        def lumina_linear_cuda_query(self,*args):return 0
    class Target:
        address=1
        def part(self,offset,size):offsets.append((offset,size));return self
    d=CompactDriver.__new__(CompactDriver);d.lib=Fake();d.view=Piece().view;d.rt=SimpleNamespace(status=1)
    d.provider=SimpleNamespace(_source=SimpleNamespace(dlognu=.001));d.tree=SimpleNamespace(size=100000,address=1)
    d.piece=lambda *a:([Location(1,1,0,.3)]*3,1.,.002,.001)
    work=Work(10000,time.monotonic()+60);d.group((None,None,None,None),0,0,8,Target(),work)
    assert offsets==[(0,448),(448,448),(896,448)]
    assert work.counts['bin_maps']==102 and work.counts['tree_compositions']==93
    assert work.counts['padding_leaves']==45
    assert work.counts['source_projected_vertices']==108 and work.counts['source_tree_compositions']==3
    assert work.used==sum(work.counts.values())


@pytest.mark.parametrize('seed',[17,41,103])
def test_same_rich_map_retains_bitwise_values_total_errors_scores_and_decisions(kernels,seed):
    rng=np.random.default_rng(seed);n,width=7,19;data=(NativeMap*(3*n*width))()
    for x in data:
        # 차수/반올림/생략 진단은 일부러 유효하지 않게 둔다. 기존 소비자도 이 필드를 읽지 않는다.
        x.transmission=CPositive(float(rng.uniform(.5,1)),int(rng.integers(-4,4)))
        x.source=CPositive(float(rng.uniform(.5,1)),int(rng.integers(-800,400)))
        x.source_error=CPositive(x.source.m,x.source.e-45);x.tau_error=float(rng.uniform(0,1e-14))
        x.source_roundoff=CPositive(float('nan'),-123);x.panels=-999
    initial=np.exp2(rng.uniform(-800,400,width))
    a=replay(kernels.lumina_linear_affine_probe,data,n,width,initial)
    b=replay(kernels.lumina_compact_affine_probe,packed(data),n,width,initial)
    assert a[0]==b[0]==0
    for x,y in zip(a[1],b[1]):
        assert x.tobytes()==y.tobytes()
        np.testing.assert_array_equal(x<=1,y<=1)


@pytest.mark.parametrize('field',['transmission','source','source_error','tau_error'])
def test_bad_consumed_field_preserves_output_canaries_in_both_layouts(kernels,field):
    data=(NativeMap*3)()
    for x in data:x.transmission=CPositive(.5,1);x.source=CPositive(.5,1)
    if field=='tau_error':data[1].tau_error=float('nan')
    else:setattr(data[1],field,CPositive(-.5,1))
    for fun,raw in [(kernels.lumina_linear_affine_probe,data),(kernels.lumina_compact_affine_probe,packed(data))]:
        code,outputs=replay(fun,raw,1,1,[0.])
        assert code!=0
        for output in outputs:np.testing.assert_array_equal(output,np.full_like(output,-17.))


@pytest.mark.parametrize('kind',['zero','recoverable','weak_bright'])
def test_zero_and_individual_weak_error_without_bright_norm(kernels,kind):
    width=2 if kind=='weak_bright' else 1;data=(NativeMap*(3*width))()
    for x in data:x.transmission=CPositive(.5,1)
    initial=np.zeros(width)
    if kind=='recoverable':
        data[width].transmission=CPositive(.5,-1599);data[2*width].transmission=CPositive(.5,1601);initial[0]=2.
    if kind=='weak_bright':
        for f,value in enumerate([1e-80,1e10]):
            m,e=np.frexp(value);data[2*width+f].source=CPositive(m,e)
            m,e=np.frexp(value*(1.01 if f==0 else 1));data[f].source=CPositive(m,e)
    a=replay(kernels.lumina_linear_affine_probe,data,1,width,initial)
    b=replay(kernels.lumina_compact_affine_probe,packed(data),1,width,initial)
    assert a[0]==b[0]==0
    for x,y in zip(a[1],b[1]):assert x.tobytes()==y.tobytes()
    if kind=='zero':assert all(np.all(x==0) for x in b[1])
    elif kind=='recoverable':assert b[1][0][0]==2. and b[1][2][0]<1
    else:assert b[1][2][0]>100 and b[1][2][1]<1


@pytest.mark.parametrize('nf,shift,span',[(17,.375,0.),(35,.375,3.4),(70,.217,68.4),(70,30.9,3.4),(70,75.,3.4)])
def test_independent_chi2_eta32_matches_native_bin_reference(kernels,nf,shift,span):
    rng=np.random.default_rng(102222+nf)
    v=np.ascontiguousarray(np.vstack([rng.uniform(-.8,1.,(2,nf)),rng.uniform(.01,2.,(2,nf))]))
    params=np.array([1.7,shift,span,.021,.13,1e-8]);f=kernels.source_measure_split_probe
    results=[]
    expected=reference(v,length=1.7,shift=shift,span=span,delta=.021,incident=.13)
    for block in [32,2]:
        m=np.full(7,-17.);e=np.full(7,-17,dtype='i8');counts=np.full(5,-17,dtype='i4')
        assert f(nf,block,32,v,params,100000,m,e,counts)==0
        value=np.ldexp(m,e.astype('i4'));results.append(value)
        assert abs(value[0]-expected)<=value[3]+2e-13*abs(expected)
        assert value[3]<=1e-8*value[0]
    assert abs(results[0][0]-results[1][0])<=results[0][3]+results[1][3]


@pytest.fixture(scope='module')
def cuda_plan():
    # 함수 본문은 host의 계획 산술만 수행한다. runtime 생성/장치 API/커널은 호출하지 않는다.
    path=os.environ.get('LUMINA_COMPACT_PLAN_LIBRARY')
    if not path:pytest.skip('별도 컴파일된 CUDA host 계획 시험만 허용')
    lib=ct.CDLL(path)
    f=lib.lumina_linear_cuda_plan
    f.argtypes=[ct.c_void_p]*2+[ct.c_double]*4+[ct.c_int]*3+[ct.c_void_p];f.restype=ct.c_int
    for name in ('node_bytes','map_bytes','local_bytes'):
        getattr(lib,'lumina_linear_cuda_'+name).restype=ct.c_size_t
    assert lib.lumina_linear_cuda_node_bytes()==1128
    assert lib.lumina_linear_cuda_map_bytes()==56 and lib.lumina_linear_cuda_local_bytes()==64
    return f


@pytest.mark.parametrize('nf,begin,shift,span',[(35,0,.375,3.4),(70,0,.217,68.4),
    (70,31,2.31,3.4),(70,0,75.,3.4),(70,0,.375,0.)])
def test_cuda_host_plan_has_distinct_tree_coordinates_and_same_native_extent(cuda_plan,nf,begin,shift,span):
    from test_lumina_linear_native import inputs
    from lumina_linear_native import Location
    import math
    view,_arrays=inputs(nf=nf);loc=(Location*2)(Location(1,3,2,.23),Location(1,2,1,.72));p=Piece()
    assert cuda_plan(ct.byref(view),loc,1.1,.001,(shift+span)*.001,shift*.001,begin,nf-begin,2,ct.byref(p))==0
    assert p.base==min(nf-2,begin+math.floor(shift)) and p.nf==nf-p.base and p.block==2
    for block,leaves in [(2,p.leaves),(32,p.source_leaves)]:
        count=(p.bins+block-1)//block
        assert leaves==1<<max(0,count-1).bit_length()
    assert CompactDriver.__new__(CompactDriver).tree_bytes(p)<=memory_layout(nf,256)['tree']
    if nf==70 and span>60:assert p.leaves==64 and p.source_leaves==4
