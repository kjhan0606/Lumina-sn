"""이미 검증한 B 바이너리의 큰 트리·padding·청색 경로만 추가 확인한다."""
import ctypes as ct
import os
import numpy as np
import pytest
from lumina_gpu_linear import Runtime,Piece,MAP_BYTES,NODE_BYTES
from lumina_linear_native import NativeMap
from lumina_gpu_driver import check
from test_lumina_gpu_linear_device import Input

pytestmark=pytest.mark.skipif(not os.environ.get('LUMINA_B_GPU_LIBRARY'),reason='봉인 GPU 실행만 허용')


@pytest.fixture(scope='module')
def gpu():
    import subprocess
    free,util=[int(v.strip()) for v in subprocess.run(['nvidia-smi','--id=0','--query-gpu=memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'],check=True,capture_output=True,text=True,timeout=15).stdout.strip().split(',')]
    assert free>=24*1024 and util<=5
    with Runtime(os.environ['LUMINA_B_GPU_LIBRARY'],free*1024**2) as r:yield r


@pytest.mark.parametrize('nf,span',[(129,97.4),(258,177.2)])
def test_same_global_tree_CPU_B_and_GPU_B_at_adversarial_columns(gpu,nf,span):
    data=Input(gpu,nf=nf);piece=Piece();shift=.375;h=.001;L=1.1;da=(shift+span)*h;db=shift*h;buffers=[]
    try:
        check(gpu.lib.lumina_linear_cuda_plan(ct.byref(data.device),data.locations,L,h,da,db,0,nf,32,ct.byref(piece)))
        nodes=gpu.allocate(2*piece.leaves*NODE_BYTES);maps=gpu.allocate(nf*MAP_BYTES);buffers=[nodes,maps]
        check(gpu.lib.lumina_linear_cuda_build(ct.byref(piece),nodes.address,gpu.status))
        check(gpu.lib.lumina_linear_cuda_query(ct.byref(piece),nodes.address,1e-8,100000,maps.address,gpu.status))
        raw=gpu.download(maps,(nf,MAP_BYTES),'u1');columns=np.array([0,17,31,32,63,nf-3,nf-2,nf-1],dtype='i4')
        f=ct.CDLL(os.environ['LUMINA_B_FULLTREE_CPU_LIBRARY']).selected_full_tree
        f.argtypes=[ct.c_void_p,ct.c_void_p]+[ct.c_double]*4+[ct.c_int,ct.c_void_p,ct.c_void_p];f.restype=ct.c_int
        reference=(NativeMap*len(columns))()
        assert f(ct.byref(data.host),data.locations,L,h,da,db,len(columns),columns.ctypes.data,reference)==0
        for i,column in enumerate(columns):
            actual=NativeMap.from_buffer_copy(raw[column].tobytes())
            for name in ('source','transmission'):
                a=getattr(actual,name).positive().value();b=getattr(reference[i],name).positive().value()
                assert abs(a-b)<=3e-12*abs(b)
    finally:
        for b in reversed(buffers):b.close()
        data.close()
