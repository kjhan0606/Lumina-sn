"""원천 측도 GPU 후보의 별도 ABI·트리·맵 직접 시험. 큰 물리 실행이 아니다."""
import ctypes as ct
import json
from math import comb
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from lumina_gpu_measure import Runtime,MeasureDriver,memory_layout,NODE_BYTES,SOURCE_NODE_BYTES,PROJECTION_BYTES
from lumina_gpu_linear import Piece,MAP_BYTES
from lumina_gpu_driver import check
from lumina_linear_native import DenseView,Location,NativeKernel,NativeMap,CPositive
from test_lumina_linear_native import inputs,endpoint


class SourceNode(ct.Structure):
    _fields_=[('m',(CPositive*26)*2),('width',ct.c_double),('first',ct.c_int),('last',ct.c_int),('depth',ct.c_int)]


def test_source_measure_distinct_node_abi_and_complete_workspace():
    assert ct.sizeof(SourceNode)==SOURCE_NODE_BYTES==856
    assert NODE_BYTES==1128 and PROJECTION_BYTES==13728
    layout=memory_layout(1115020,256)
    assert layout['workspace']==973244968 and layout['workspace']<=1024**3
    assert MeasureDriver.node_bytes==NODE_BYTES
    assert MeasureDriver.projection_bytes==PROJECTION_BYTES
    with pytest.raises(MemoryError):memory_layout(2000000,256)


def test_driver_rejects_a_different_variant_before_any_allocation():
    rt=SimpleNamespace(lib=SimpleNamespace(lumina_linear_cuda_node_bytes=lambda:272))
    with pytest.raises(ValueError,match='저장 구조'):
        MeasureDriver(rt,SimpleNamespace(nf=17),'unused.so')


@pytest.fixture(scope='module')
def gpu():
    library=os.environ.get('LUMINA_MEASURE_GPU_LIBRARY')
    if not library:pytest.skip('별도 봉인된 원천 측도 GPU 직접 시험만 허용')
    row=subprocess.run(['nvidia-smi','--id=0','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],
        check=True,capture_output=True,text=True,timeout=15).stdout.strip().split(',')
    free,util=[int(v.strip()) for v in row]
    assert free>=24*1024 and util<=5
    with Runtime(library,free*1024**2) as runtime:
        try:yield runtime
        finally:
            memory=runtime.report()
            target=Path(os.environ['LUMINA_MEASURE_GPU_RECORD_ROOT'])/'memory.json'
            with target.open('x') as f:json.dump(memory,f,indent=2,allow_nan=False)


class Input:
    def __init__(self,gpu,nf,weak=False):
        self.host,self.arrays=inputs(nf=nf);self.buffers=[]
        if weak:
            self.arrays[0][0]=0.
            self.host.jmode=0  # 산란원만0으로 한다. 원 native 계약의 ES>0은 보존한다.
            self.arrays[0][2]=np.arange(nf)%3*np.nextafter(0.,1.)
        pointers=[]
        for array in self.arrays:
            b=gpu.allocate(array.nbytes);self.buffers.append(b);gpu.upload(b,array)
            pointers.append(ct.cast(b.address,ct.POINTER(ct.c_double)))
        self.device=DenseView(self.host.ns,self.host.nf,self.host.ne,self.host.jmode,self.host.generation,*pointers)
        self.locations=(Location*2)(Location(1,3,2,.23),Location(1,2,1,.72))
    def close(self):
        for b in reversed(self.buffers):b.close()


@pytest.mark.parametrize('nf,block',[(35,4),(70,32)])
def test_parallel_projected_tree_including_partial_last_block_and_padding(gpu,nf,block):
    data=Input(gpu,nf);piece=Piece();buffers=[]
    try:
        check(gpu.lib.lumina_linear_cuda_plan(ct.byref(data.device),data.locations,1.1,.001,.000575,.000375,0,nf,block,ct.byref(piece)))
        nodes=gpu.allocate(2*piece.leaves*NODE_BYTES);buffers.append(nodes)
        check(gpu.lib.lumina_linear_cuda_build(ct.byref(piece),nodes.address,gpu.status))
        offset=2*piece.leaves*272
        raw=gpu.download(nodes.part(offset,2*piece.leaves*SOURCE_NODE_BYTES),(2*piece.leaves,SOURCE_NODE_BYTES),'u1')
        root=SourceNode.from_buffer_copy(raw[1].tobytes())
        assert root.first==0 and root.last==nf-1 and root.width==nf-1
        x,w=np.polynomial.legendre.leggauss(16);v=(x+1)/2;w=w/(2*(nf-1))
        for c in range(2):
            eta=np.array([endpoint(data.host,data.arrays,data.locations[c],i)[1] for i in range(nf)])
            reference=np.zeros(26)
            for j in range(nf-1):
                t=(j+v)/(nf-1);source=(1-v)*eta[j]+v*eta[j+1]
                for k in range(26):reference[k]+=np.sum(w*comb(25,k)*t**k*(1-t)**(25-k)*source)
            actual=np.array([p.positive().value() for p in root.m[c]])
            np.testing.assert_allclose(actual,reference,rtol=3e-12,atol=0)
    finally:
        for b in reversed(buffers):b.close()
        data.close()


@pytest.mark.parametrize('shift,span,weak',[(.375,0.,False),(.375,.2,False),(.375,3.4,False),
    (40.,3.,False),(.375,3.4,True)])
def test_source_measure_device_maps_match_CPU_with_weak_recovery_and_blue(gpu,shift,span,weak):
    nf=17;data=Input(gpu,nf,weak);piece=Piece();buffers=[];length=1e300 if weak else 1.1;h=.001
    try:
        check(gpu.lib.lumina_linear_cuda_plan(ct.byref(data.device),data.locations,length,h,(shift+span)*h,shift*h,
                                             0,nf,4,ct.byref(piece)))
        nodes=gpu.allocate(2*piece.leaves*NODE_BYTES);maps=gpu.allocate(nf*MAP_BYTES);buffers=[nodes,maps]
        check(gpu.lib.lumina_linear_cuda_build(ct.byref(piece),nodes.address,gpu.status))
        check(gpu.lib.lumina_linear_cuda_query(ct.byref(piece),nodes.address,1e-8,100000,maps.address,gpu.status))
        raw=gpu.download(maps,(nf,MAP_BYTES),'u1')
        cpu=NativeKernel(os.environ['LUMINA_MEASURE_NATIVE_LIBRARY'],source_measure=True)
        for column in range(nf):
            expected,_=cpu.map(data.host,data.locations,length,h,(shift+span)*h,shift*h,column)
            actual=NativeMap.from_buffer_copy(raw[column].tobytes())
            for a,b in ((actual.transmission.positive().value(),expected.t.value()),
                        (actual.source.positive().value(),expected.source.value())):
                assert abs(a-b)<=3e-10*abs(b)
            assert actual.source_error.positive().value()<=1e-8*actual.source.positive().value()
    finally:
        for b in reversed(buffers):b.close()
        data.close()
