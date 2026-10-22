"""끝점 값 재사용이 원 반경/ES×J 순서·실제 적응 맵을 바꾸지 않는지 대조한다."""
import ctypes as ct
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
from test_lumina_linear_native import inputs,endpoint
from test_lumina_native_cost import piece,run
from lumina_gpu_compact import CompactDriver,EndpointCompactDriver,memory_layout

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def kernels(tmp_path_factory):
    result=[]
    for cached in (False,True):
        path=tmp_path_factory.mktemp('endpoint-cache')/'probe.so'
        flags=['-DCMF_GPU_ENVELOPE_UPPER','-DCMF_GPU_ADAPTIVE_LINEAR_EXP']
        if cached:flags.append('-DCMF_GPU_ENDPOINT_CACHE')
        p=subprocess.run([os.environ.get('CXX','c++'),'-O2','-std=c++17','-ffp-contract=off',
            '-Wall','-Wextra','-Werror','-fPIC','-shared',*flags,'-I'+str(ROOT/'src'),
            str(ROOT/'tests/native_cost_probe.cpp'),str(ROOT/'tests/endpoint_cache_probe.cpp'),'-o',str(path)],
            capture_output=True,text=True,timeout=60)
        assert p.returncode==0,p.stderr
        lib=ct.CDLL(str(path))
        for name in ('native_cost_probe','native_cost_reference'):
            f=getattr(lib,name);f.argtypes=[ct.c_void_p,ct.c_int]+[ct.c_void_p]*3;f.restype=ct.c_int
        f=lib.endpoint_cache_probe;f.argtypes=[ct.c_void_p]*3+[ct.c_int];f.restype=ct.c_int
        f=lib.endpoint_cache_partial_probe
        f.argtypes=[ct.c_void_p]+[ct.c_int]*4+[ct.c_void_p];f.restype=ct.c_int
        result.append(lib)
    return result


@pytest.mark.parametrize('mode',[0,1,2])
@pytest.mark.parametrize('jmode',[0,1])
def test_exact_reader_order_and_refill_not_stale_generation(kernels,mode,jmode):
    view,arrays=inputs(nf=17,jmode=jmode)
    from lumina_gpu_driver import Location
    loc=Location(mode,3 if mode==0 else 2,3 if mode==0 else 1,.37 if mode else 0.)
    locations=(Location*2)(loc,loc);previous=None
    for _ in range(2):
        out=np.full((4,17),-17.)
        assert kernels[1].endpoint_cache_probe(ct.byref(view),locations,out.ctypes.data,out.size)==0
        for j in range(17):
            chi,eta=endpoint(view,arrays,loc,j)
            np.testing.assert_allclose(out[:,j],[chi,chi,eta,eta],rtol=3e-15,atol=0)
        if previous is not None:assert not np.array_equal(out,previous)
        previous=out.copy();view.generation+=1
        arrays[0][2]*=2.;arrays[1][:,view.ne:2*view.ne]+=np.log(2.)


@pytest.mark.parametrize('column,shift,span',[(0,.375,50.7),(31,.375,50.7),(64,3.375,50.7),
    (127,.375,50.7),(128,.375,50.7),(33,.375,0.),(15,140.,3.)])
def test_cached_maps_errors_panels_and_signed_source_bits_equal_original(kernels,column,shift,span):
    p,_host,_arrays=piece(129,shift,span)
    a,b=[run(lib.native_cost_probe,p,column) for lib in kernels]
    assert a[0]==b[0]==0 and bytes(a[1])==bytes(b[1])


@pytest.mark.parametrize('bad',['eta','ES'])
def test_invalid_original_input_preserves_entire_output(kernels,bad):
    view,arrays=inputs(nf=17,jmode=0)
    from lumina_gpu_driver import Location
    locations=(Location*2)(Location(1,3,2,.23),Location(1,2,1,.72))
    if bad=='eta':arrays[0][2,:]=-10.
    else:arrays[0][1,:]=0.
    out=np.full((4,17),-17.)
    assert kernels[1].endpoint_cache_probe(ct.byref(view),locations,out.ctypes.data,out.size)!=0
    assert np.all(out==-17.)


def test_unfilled_columns_refused_but_blue_edge_is_available(kernels):
    values=np.ones((4,17));f=kernels[1].endpoint_cache_partial_probe
    for filled,column,code in ((3,2,0),(3,3,2),(3,16,0),(0,0,2),(0,16,0)):
        out=np.full(2,-17.)
        assert f(values.ctypes.data,17,filled,0,column,out.ctypes.data)==code
        assert np.all(out==(1. if code==0 else -17.))


@pytest.mark.parametrize('nf,bins,offset,span,count,wanted',[
    (129,4,.375,3.4,1,33),(129,33,.375,32.4,1,65),
    (129,0,.375,0.,33,34),(35,34,.375,40.,1,35),(129,0,140.,3.,1,0)])
def test_partial_cache_covers_rounded_source_and_chi_tree_vertices(kernels,nf,bins,offset,span,count,wanted):
    f=kernels[1].endpoint_cache_prefix_probe
    f.argtypes=[ct.c_int]*4+[ct.c_double]*2+[ct.c_int];f.restype=ct.c_int
    filled=f(nf,bins,2,32,offset,span,count)
    assert filled==wanted
    from lumina_gpu_linear import endpoint_prefix
    p=SimpleNamespace(nf=nf,bins=bins,offset=offset,span=span,count=count)
    assert endpoint_prefix(p,2,32)==filled
    values=np.ones((4,nf));out=np.full(2,-17.)
    if filled:
        assert kernels[1].endpoint_cache_partial_probe(values.ctypes.data,nf,filled,0,filled-1,out.ctypes.data)==0
    assert kernels[1].endpoint_cache_partial_probe(values.ctypes.data,nf,filled,0,nf-1,out.ctypes.data)==0


def test_all_endpoint_bytes_counted_without_expanding_one_GiB():
    old=memory_layout(1115020,256);new=memory_layout(1115020,256,endpoint_cache=True)
    assert old['workspace']==1013090856 and old['endpoint_bytes']==0
    assert new['endpoint_bytes']==35680640 and new['workspace']==1048771496<1024**3
    assert new['tree']==old['tree']+35680640
    p=SimpleNamespace(leaves=1048576,source_leaves=65536,nf=1115020)
    d=EndpointCompactDriver.__new__(EndpointCompactDriver)
    assert d.tree_bytes(p)==new['tree']
    assert not CompactDriver.endpoint_cache and d.endpoint_cache
    with pytest.raises(MemoryError):memory_layout(2000000,256,endpoint_cache=True)


def test_mixed_cache_binary_and_driver_refused_before_allocation():
    def report(nf):return 4*nf*8
    lib=SimpleNamespace(lumina_linear_cuda_node_bytes=lambda:1128,
        lumina_linear_cuda_map_bytes=lambda:56,lumina_linear_cuda_endpoint_bytes=report)
    with pytest.raises(ValueError,match='끝점 캐시 비용'):
        CompactDriver(SimpleNamespace(lib=lib),SimpleNamespace(nf=17),'unused.so')
    del lib.lumina_linear_cuda_endpoint_bytes
    with pytest.raises(ValueError,match='끝점 캐시 비용'):
        EndpointCompactDriver(SimpleNamespace(lib=lib),SimpleNamespace(nf=17),'unused.so')


def test_device_refill_fixture_mutates_arrays_without_assigning_tuple_members():
    from test_lumina_gpu_compact_device import double_source_input
    _view,arrays=inputs(nf=17);data=SimpleNamespace(arrays=arrays)
    old=[a.copy() for a in arrays]
    double_source_input(data)
    np.testing.assert_array_equal(arrays[0][2],2*old[0][2])
    np.testing.assert_array_equal(arrays[2],2*old[2])
    np.testing.assert_array_equal(arrays[0][:2],old[0][:2])
    np.testing.assert_array_equal(arrays[1],old[1])
