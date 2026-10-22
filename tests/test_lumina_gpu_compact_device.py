"""축약 원천 경로의 작은 장치 직접시험. 실제13003/완성장은 실행하지 않는다."""
import ctypes as ct
import json
from math import comb
import os
from pathlib import Path
import subprocess
import sys
from decimal import Decimal,localcontext

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from lumina_gpu_compact import Runtime,CompactMap,CompactDriver,EndpointCompactDriver,MAP_BYTES
from lumina_gpu_linear import Piece,LOCAL_BYTES
from lumina_gpu_driver import check
from lumina_linear_native import NativeKernel,CPositive
from test_lumina_gpu_measure import Input,SourceNode
from test_lumina_linear_native import endpoint


def tree_bytes(piece):
    driver=EndpointCompactDriver if os.environ.get('LUMINA_ENDPOINT_CACHE')=='1' else CompactDriver
    return driver.__new__(driver).tree_bytes(piece)


def double_source_input(data):
    data.arrays[0][2]*=2.
    data.arrays[2][...]*=2.


@pytest.fixture(scope='session')
def gpu():
    path=os.environ.get('LUMINA_COMPACT_GPU_LIBRARY')
    if not path:pytest.skip('별도 봉인된 축약 원천 GPU 직접시험만 허용')
    row=subprocess.run(['nvidia-smi','--id=0','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],
        check=True,capture_output=True,text=True,timeout=15).stdout.strip().split(',')
    free,util=[int(v.strip()) for v in row]
    shared=os.environ.get('LUMINA_COMPACT_SHARED_CORRECTNESS')=='1'
    assert free>=24*1024
    if not shared:assert util<=5
    with Runtime(path,free*1024**2) as runtime:
        try:yield runtime
        finally:
            with (Path(os.environ['LUMINA_COMPACT_GPU_RECORD_ROOT'])/'memory.json').open('x') as f:
                json.dump(dict(runtime.report(),shared_correctness=shared,precheck_utilization=util,
                               timing_acceptance=False),f,indent=2,allow_nan=False)


@pytest.mark.parametrize('nf',[35,70])
def test_split_device_tree_has_correct_source_origin_padding_and_exact_moments(gpu,nf):
    data=Input(gpu,nf);p=Piece();nodes=None
    try:
        check(gpu.lib.lumina_linear_cuda_plan(ct.byref(data.device),data.locations,1.1,.001,.000575,.000375,0,nf,2,ct.byref(p)))
        nodes=gpu.allocate(tree_bytes(p))
        check(gpu.lib.lumina_linear_cuda_build(ct.byref(p),nodes.address,gpu.status))
        assert p.source_leaves!=p.leaves
        raw=gpu.download(nodes.part(2*p.leaves*272,2*p.source_leaves*856),(2*p.source_leaves,856),'u1')
        root=SourceNode.from_buffer_copy(raw[1].tobytes())
        assert root.first==0 and root.last==nf-1 and root.width==nf-1
        x,w=np.polynomial.legendre.leggauss(16);v=(x+1)/2;w=w/(2*(nf-1))
        for c in range(2):
            eta=np.array([endpoint(data.host,data.arrays,data.locations[c],i)[1] for i in range(nf)])
            expected=np.zeros(26)
            for j in range(nf-1):
                t=(j+v)/(nf-1);source=(1-v)*eta[j]+v*eta[j+1]
                for k in range(26):expected[k]+=np.sum(w*comb(25,k)*t**k*(1-t)**(25-k)*source)
            actual=np.array([z.positive().value() for z in root.m[c]])
            np.testing.assert_allclose(actual,expected,rtol=3e-12,atol=0)
    finally:
        if nodes is not None:nodes.close()
        data.close()


@pytest.mark.parametrize('shift,span,weak',[(.375,0.,False),(.375,.2,False),(.375,3.4,False),
    (40.,3.,False),(.375,3.4,True)])
def test_compact_device_map_keeps_CPU_source_transmission_and_total_error(gpu,shift,span,weak):
    nf=17;data=Input(gpu,nf,weak);p=Piece();buffers=[];length=1e300 if weak else 1.1;h=.001
    try:
        check(gpu.lib.lumina_linear_cuda_plan(ct.byref(data.device),data.locations,length,h,(shift+span)*h,shift*h,0,nf,2,ct.byref(p)))
        nodes=gpu.allocate(tree_bytes(p));buffers.append(nodes)
        maps=gpu.allocate(nf*MAP_BYTES);buffers.append(maps)
        check(gpu.lib.lumina_linear_cuda_build(ct.byref(p),nodes.address,gpu.status))
        check(gpu.lib.lumina_linear_cuda_query(ct.byref(p),nodes.address,1e-8,100000,maps.address,gpu.status))
        raw=gpu.download(maps,(nf,MAP_BYTES),'u1')
        cpu=NativeKernel(os.environ['LUMINA_COMPACT_NATIVE_LIBRARY'],source_measure=True)
        for column in range(nf):
            expected,_=cpu.map(data.host,data.locations,length,h,(shift+span)*h,shift*h,column)
            actual=CompactMap.from_buffer_copy(raw[column].tobytes())
            for a,b in [(actual.transmission.positive().value(),expected.t.value()),
                        (actual.source.positive().value(),expected.source.value())]:
                assert abs(a-b)<=3e-10*abs(b)
            error=actual.source_error.positive().value();value=actual.source.positive().value()
            assert 0<=error<=1e-8*value
            assert abs(value-expected.source.value())<=error+expected.source_error.value()
    finally:
        for b in reversed(buffers):b.close()
        data.close()


def test_compact_device_replay_preserves_order_recovery_and_weak_failure(gpu):
    one=CPositive(.5,1);zero=CPositive();identity=CompactMap(one,zero,zero,0.)
    a=CompactMap(CPositive(.5,-1599),zero,zero,0.);b=CompactMap(CPositive(.5,1601),zero,zero,0.)
    def source(v):
        m,e=np.frexp(v);return CompactMap(one,CPositive(m,e),zero,0.)
    maps=(CompactMap*6)(identity,source(1.01e-8),a,identity,b,source(1e-8))
    buffers=[]
    def allocate(size):
        result=gpu.allocate(size);buffers.append(result);return result
    try:
        raw=np.frombuffer(maps,dtype='u1').copy();m=allocate(raw.nbytes);gpu.upload(m,raw)
        pointers=allocate(8);gpu.upload(pointers,np.array([m.address],dtype='u8'))
        initial=allocate(16);gpu.upload(initial,np.array([2.,0.]));out=allocate(16)
        local=allocate(2*LOCAL_BYTES);scores=allocate(8);ratios=allocate(16);errors=allocate(32)
        check(gpu.lib.lumina_linear_cuda_replay(1,2,2,1,1e-5,pointers.address,initial.address,out.address,
              local.address,scores.address,ratios.address,errors.address,gpu.status))
        actual=gpu.download(out,2);ratio=gpu.download(ratios,2)
        assert actual.tolist()==[2.,1e-8] and ratio[0]<1 and ratio[1]>100
        assert gpu.download(scores,1)[0]>100
    finally:
        for b in reversed(buffers):b.close()


@pytest.mark.skipif(os.environ.get('LUMINA_COMPACT_RANGE_TEST')!='1',reason='범위 진단 바이너리 명시 선택 필요')
def test_range_query_retains_original_indices_and_does_not_publish_outside_range(gpu):
    data=Input(gpu,70);p=Piece();buffers=[]
    f=gpu.lib.lumina_linear_cuda_query_range
    f.argtypes=[ct.c_void_p]*2+[ct.c_double]+[ct.c_int]*3+[ct.c_void_p]*2;f.restype=ct.c_int
    try:
        check(gpu.lib.lumina_linear_cuda_plan(ct.byref(data.device),data.locations,1.1,.001,.003775,.000375,0,70,2,ct.byref(p)))
        nodes=gpu.allocate(tree_bytes(p));buffers.append(nodes)
        maps=gpu.allocate(70*MAP_BYTES);buffers.append(maps)
        gpu.upload(maps,np.full(70*MAP_BYTES,0xA5,dtype='u1'))
        check(gpu.lib.lumina_linear_cuda_build(ct.byref(p),nodes.address,gpu.status))
        check(f(ct.byref(p),nodes.address,1e-8,100000,31,6,maps.address,gpu.status))
        raw=gpu.download(maps,(70,MAP_BYTES),'u1')
        assert np.all(raw[:31]==0xA5) and np.all(raw[37:]==0xA5)
        cpu=NativeKernel(os.environ['LUMINA_COMPACT_NATIVE_LIBRARY'],source_measure=True)
        for column in range(31,37):
            expected,_=cpu.map(data.host,data.locations,1.1,.001,.003775,.000375,column)
            actual=CompactMap.from_buffer_copy(raw[column].tobytes())
            for a,b in [(actual.transmission.positive().value(),expected.t.value()),
                        (actual.source.positive().value(),expected.source.value())]:
                assert abs(a-b)<=3e-10*abs(b)
        for first,count in [(-1,1),(69,2),(0,0),(0,1025)]:
            assert f(ct.byref(p),nodes.address,1e-8,100000,first,count,maps.address,gpu.status)==1
        gpu.sync()
        np.testing.assert_array_equal(gpu.download(maps,(70,MAP_BYTES),'u1'),raw)
    finally:
        for b in reversed(buffers):b.close()
        data.close()


@pytest.mark.parametrize('begin,count,span',[(0,1,3.4),(31,3,32.4),(0,33,0.)])
def test_partial_piece_keeps_rounded_source_vertices_and_CPU_map(gpu,begin,count,span):
    # query 끝보다 멀리 읽는 마지막 η32 block도 원 Reader와 같아야 한다.
    nf=129;data=Input(gpu,nf);p=Piece();buffers=[];shift=.375;h=.001;length=1.1
    try:
        check(gpu.lib.lumina_linear_cuda_plan(ct.byref(data.device),data.locations,length,h,
            (shift+span)*h,shift*h,begin,count,2,ct.byref(p)))
        nodes=gpu.allocate(tree_bytes(p));buffers.append(nodes)
        maps=gpu.allocate(count*MAP_BYTES);buffers.append(maps)
        check(gpu.lib.lumina_linear_cuda_build(ct.byref(p),nodes.address,gpu.status))
        check(gpu.lib.lumina_linear_cuda_query(ct.byref(p),nodes.address,1e-8,100000,maps.address,gpu.status))
        raw=gpu.download(maps,(count,MAP_BYTES),'u1')
        cpu=NativeKernel(os.environ['LUMINA_COMPACT_NATIVE_LIBRARY'],source_measure=True)
        for index in range(count):
            expected,_=cpu.map(data.host,data.locations,length,h,(shift+span)*h,shift*h,begin+index)
            actual=CompactMap.from_buffer_copy(raw[index].tobytes())
            for a,b in [(actual.transmission.positive().value(),expected.t.value()),
                        (actual.source.positive().value(),expected.source.value())]:
                assert abs(a-b)<=3e-10*abs(b)
            assert abs(actual.source.positive().value()-expected.source.value())<=\
                actual.source_error.positive().value()+expected.source_error.value()
    finally:
        for b in reversed(buffers):b.close()
        data.close()


@pytest.mark.skipif(os.environ.get('LUMINA_ENDPOINT_CACHE')!='1',reason='끝점 캐시 선택 바이너리만 시험')
def test_endpoint_cache_refills_changed_input_for_each_piece(gpu):
    nf=35;data=Input(gpu,nf);p=Piece();buffers=[]
    try:
        check(gpu.lib.lumina_linear_cuda_plan(ct.byref(data.device),data.locations,1.1,.001,.003775,.000375,
            7,8,2,ct.byref(p)))
        nodes=gpu.allocate(tree_bytes(p));maps=gpu.allocate(8*MAP_BYTES);buffers=[nodes,maps]
        results=[]
        for _ in range(2):
            check(gpu.lib.lumina_linear_cuda_build(ct.byref(p),nodes.address,gpu.status))
            check(gpu.lib.lumina_linear_cuda_query(ct.byref(p),nodes.address,1e-8,100000,maps.address,gpu.status))
            raw=gpu.download(maps,(8,MAP_BYTES),'u1')
            results.append([CompactMap.from_buffer_copy(row.tobytes()).source.positive().value() for row in raw])
            double_source_input(data)
            gpu.upload(data.buffers[0],data.arrays[0]);gpu.upload(data.buffers[2],data.arrays[2])
            p.view.generation+=1
        np.testing.assert_array_equal(np.array(results[1]),2*np.array(results[0]))
    finally:
        for b in reversed(buffers):b.close()
        data.close()


@pytest.mark.skipif(os.environ.get('LUMINA_WARP_MEASURE')!='1',reason='협력 선택 바이너리만 시험')
@pytest.mark.parametrize('adaptive',[0,1])
@pytest.mark.parametrize('width',[0.,1e-8,1e-4,.01,.125])
def test_warp_polynomial_matches_independent_exponential_enclosure(gpu,adaptive,width):
    f=gpu.lib.lumina_warp_polynomial_probe
    f.argtypes=[ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_void_p];f.restype=ct.c_int
    phase=np.array([-.7*width,-.1*width,.3*width,-.4*width]);buffers=[]
    try:
        a=gpu.allocate(phase.nbytes);buffers.append(a);gpu.upload(a,phase)
        b=gpu.allocate(28*8);buffers.append(b)
        check(f(a.address,adaptive,b.address,gpu.status));gpu.sync();out=gpu.download(b,28)
        m=int(out[27]);assert m in range(0,25,3) and np.all(out[:m+1]>=1.)
        with localcontext() as context:
            context.prec=90;D=Decimal.from_float
            def basis(n,k,t):
                return Decimal(comb(n,k))*(t**k if k else Decimal(1))*((1-t)**(n-k) if n!=k else Decimal(1))
            for x in (0.,.03125,.27,.5,.91,1.):
                t=D(x);wanted=sum(D(float(phase[k]))*basis(3,k,t) for k in range(4))
                actual=sum(D(float(out[k]))*basis(m,k,t) for k in range(m+1))
                expected=(wanted-D(float(out[25]))).exp()
                assert abs(actual-expected)<=D(float(out[26]))+Decimal('3e-13')*expected
    finally:
        for b in reversed(buffers):b.close()


@pytest.mark.skipif(os.environ.get('LUMINA_WARP_MEASURE')!='1',reason='협력 선택 바이너리만 시험')
@pytest.mark.parametrize('case',['zero','flat','wide','weak'])
def test_warp_positive_reduction_keeps_weak_values_and_roundoff_bound(gpu,case):
    values=np.zeros(32,dtype=[('m','<f8'),('e','<i8')]);rng=np.random.default_rng(10222026)
    if case!='zero':
        values['m']=rng.uniform(.5,1.,32)
        values['e']=0 if case=='flat' else (rng.integers(-1800,1800,32) if case=='wide' else -1800)
    f=gpu.lib.lumina_warp_positive_sum_probe
    f.argtypes=[ct.c_void_p]*3;f.restype=ct.c_int;buffers=[]
    try:
        a=gpu.allocate(values.nbytes);buffers.append(a);gpu.upload(a,values)
        b=gpu.allocate(16);buffers.append(b)
        check(f(a.address,b.address,gpu.status));gpu.sync()
        out=CPositive.from_buffer_copy(gpu.download(b,(16,),'u1').tobytes())
        with localcontext() as context:
            context.prec=100;D=Decimal.from_float
            wanted=sum(D(float(v['m']))*Decimal(2)**int(v['e']) for v in values)
            out.positive()
            actual=D(out.m)*Decimal(2)**out.e
            assert abs(actual-wanted)<=Decimal('1e-12')*wanted
    finally:
        for b in reversed(buffers):b.close()
