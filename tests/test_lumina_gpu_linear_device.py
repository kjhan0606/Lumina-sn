"""B 실제 장치 시험. 명시적 환경/봉인/계산 노드 실행 전에는 건너뛴다."""
import ctypes as ct
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from lumina_gpu_driver import Work,check
from lumina_gpu_linear import Runtime,LinearDriver,Piece,MAP_BYTES,NODE_BYTES,LOCAL_BYTES,POSITIVE_DTYPE
from lumina_compiled_input import DenseView,CompiledInput
from lumina_linear_native import Location,NativeKernel,NativeMap,Positive

pytestmark=pytest.mark.skipif(not os.environ.get('LUMINA_B_GPU_LIBRARY'),reason='명시적 B GPU 실행 계약 필요')


@pytest.fixture(scope='module')
def gpu():
    smi=subprocess.run(['nvidia-smi','--id=0','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],
        check=True,capture_output=True,text=True,timeout=15).stdout.strip().split(',')
    free,util=[int(v.strip()) for v in smi]
    assert free>=24*1024 and util<=5
    with Runtime(os.environ['LUMINA_B_GPU_LIBRARY'],free*1024**2) as runtime:yield runtime


class Input:
    def __init__(self,gpu,nf=33):
        from test_lumina_linear_native import inputs
        self.host,self.arrays=inputs(nf=nf);self.gpu=gpu;self.buffers=[]
        ptr=[]
        for a in self.arrays:
            block=gpu.allocate(a.nbytes);self.buffers.append(block);gpu.upload(block,a)
            ptr.append(ct.cast(block.address,ct.POINTER(ct.c_double)))
        self.device=DenseView(self.host.ns,self.host.nf,self.host.ne,self.host.jmode,self.host.generation,*ptr)
        self.locations=(Location*2)(Location(1,3,2,.23),Location(1,2,1,.72))
    def close(self):
        for b in reversed(self.buffers):b.close()


@pytest.mark.parametrize('shift,span',[(.375,0.),(.375,.2),(.375,1.25),(.375,9.4),(40.,3.)])
def test_device_native_moment_maps_match_cpu_B_including_blue_end(gpu,shift,span):
    data=Input(gpu);p=Piece();h=.001;length=1.1;da=(shift+span)*h;db=shift*h;buffers=[]
    try:
        check(gpu.lib.lumina_linear_cuda_plan(ct.byref(data.device),data.locations,length,h,da,db,0,data.host.nf,32,ct.byref(p)))
        nodes=gpu.allocate(2*p.leaves*NODE_BYTES);maps=gpu.allocate(data.host.nf*MAP_BYTES);buffers=[nodes,maps]
        check(gpu.lib.lumina_linear_cuda_build(ct.byref(p),nodes.address,gpu.status))
        check(gpu.lib.lumina_linear_cuda_query(ct.byref(p),nodes.address,1e-8,100000,maps.address,gpu.status))
        raw=gpu.download(maps,(data.host.nf,MAP_BYTES),'u1')
        cpu=NativeKernel(os.environ['LUMINA_B_NATIVE_LIBRARY'])
        for column in range(data.host.nf):
            expected,detail=cpu.map(data.host,data.locations,length,h,da,db,column)
            actual=NativeMap.from_buffer_copy(raw[column].tobytes())
            for a,b in ((actual.transmission.positive(),expected.t),(actual.source.positive(),expected.source)):
                av,bv=a.value(),b.value()
                assert abs(av-bv)<=max(1e-300,abs(bv))*3e-10
            assert actual.source_error.positive().value()>=0 and actual.panels>0
    finally:
        for b in reversed(buffers):b.close()
        data.close()


def test_device_replay_keeps_scaled_temporal_order_and_weak_output_failure(gpu):
    from test_lumina_gpu_linear_host import raw
    from lumina_linear_native import Map
    z=Positive();one=Positive.of(1.)
    a=Map(Positive(.5,-1599),z,z,0.);b=Map(Positive(.5,1601),z,z,0.)
    identity=Map(one,z,z,0.)
    # 서로 다른 두 열: 첫 열은 중간 underflow 회복, 둘째 열은 실제 coarse/fine 불일치.
    c=Map(one,Positive.of(1.01e-8),z,0.);r=Map(one,Positive.of(1e-8),z,0.)
    maps=(NativeMap*6)(raw(identity),raw(c),raw(a),raw(identity),raw(b),raw(r))
    host=np.frombuffer(maps,dtype='u1').copy();buffers=[]
    def allocate(size):
        result=gpu.allocate(size);buffers.append(result);return result
    try:
        m=allocate(host.nbytes);gpu.upload(m,host);p=allocate(8);gpu.upload(p,np.array([m.address],dtype='u8'))
        initial=allocate(16);gpu.upload(initial,np.array([2.,0.]));out=allocate(16);local=allocate(2*LOCAL_BYTES)
        scores=allocate(8);ratios=allocate(16);errors=allocate(32)
        check(gpu.lib.lumina_linear_cuda_replay(1,2,2,1,1e-5,p.address,initial.address,out.address,
            local.address,scores.address,ratios.address,errors.address,gpu.status))
        actual=gpu.download(out,2);ratio=gpu.download(ratios,2)
        assert actual[0]==2. and actual[1]==1e-8 and ratio[0]<1. and ratio[1]>100.
        assert gpu.download(scores,1)[0]>100.
    finally:
        for b in reversed(buffers):b.close()


def test_device_weighted_reduction_includes_every_block_and_zero_weights(gpu):
    nf=9001;weights=np.geomspace(1e-30,1e20,nf);intensity=np.geomspace(1e-10,1e10,nf)
    bounds=np.geomspace(1e-20,1e-5,nf);mantissa,exponent=np.frexp(bounds)
    encoded=np.empty(nf,dtype=POSITIVE_DTYPE);encoded['m']=mantissa;encoded['e']=exponent
    buffers=[]
    try:
        for values in (weights,intensity,encoded):
            b=gpu.allocate(values.nbytes);gpu.upload(b,values);buffers.append(b)
        pair=gpu.allocate(((nf+4095)//4096)*32);buffers.append(pair)
        for zero in (False,True):
            if zero:gpu.upload(buffers[0],np.zeros(nf))
            check(gpu.lib.lumina_linear_cuda_weighted(nf,*[b.address for b in buffers],gpu.status))
            parts=gpu.download(pair,((nf+4095)//4096,2),POSITIVE_DTYPE);value=error=Positive()
            for row in parts:
                value=value+Positive(float(row[0]['m']),int(row[0]['e']));error=error+Positive(float(row[1]['m']),int(row[1]['e']))
            if zero:assert value.value()==error.value()==0.
            else:
                assert value.value()==pytest.approx(float(np.sum(weights*intensity)),rel=3e-14)
                assert error.value()>=float(np.sum(weights*bounds))*(1-3e-14)
    finally:
        for b in reversed(buffers):b.close()


def test_small_complete_fields_and_actual_J1_rebind(gpu,tmp_path):
    from test_lumina_nodal_driver import setup,_TrialField
    from lumina_nodal_driver import CompiledDriverPlan,DriverKernel
    cpu=os.environ['LUMINA_B_CPU_LIBRARY'];g,source,_,boundary,saved=setup(tmp_path,cpu)
    plan=CompiledDriverPlan(g,source,panels=1,block_width=8);reference=DriverKernel(cpu);records=[]
    with source:
        field=_TrialField(source,saved)
        with CompiledInput(cpu,g,source,field) as provider:
            provider.bind_view();provider.prepare_dense();driver=LinearDriver(gpu,provider,cpu,block_width=8)
            try:
                for iteration in (1,2):
                    if iteration==2:
                        field.values=np.array(moments[0],copy=True);field.values.flags.writeable=False
                        field.identity=dict(role='OWN_SYNTHETIC_B_J1_REBOUND',iteration=1)
                        provider.bind_view();provider.prepare_dense();driver.upload_generation()
                    _,oracle=reference.execute_small(plan,provider,boundary.planck,boundary.gradient,max_seconds=120.,reuse_maps=True)
                    expected=oracle.pop('moments')
                    moments,report=driver.field(plan,boundary.planck,boundary.gradient,seconds=180.)
                    for a,b in zip(moments,expected):assert float(abs(a-b).max())<=1e-4*float(abs(b).max())
                    assert report['physical_intervals']==3302 and report['source_generation']==iteration
                    records.append(report)
            finally:driver.close()
    (tmp_path/'small-B-fields.json').write_text(json.dumps(records,indent=2,allow_nan=False))


def test_last_device_error_is_sticky_and_not_cleared(gpu):
    buffers=[]
    try:
        for values in (np.ones(8),np.ones(8),np.zeros(8,dtype=POSITIVE_DTYPE)):
            b=gpu.allocate(values.nbytes);gpu.upload(b,values);buffers.append(b)
        invalid=np.zeros(8,dtype=POSITIVE_DTYPE);invalid['m'][3]=-1.;gpu.upload(buffers[2],invalid)
        pairs=gpu.allocate(32);buffers.append(pairs)
        check(gpu.lib.lumina_linear_cuda_weighted(8,*[b.address for b in buffers],gpu.status))
        for _ in range(2):
            with pytest.raises(RuntimeError):gpu.sync()
    finally:
        for b in reversed(buffers):b.close()
