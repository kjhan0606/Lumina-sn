"""분리된 대체kernel이 두꺼운bin의 같은 물리해를 반환하는지 직접 확인한다."""
import ctypes as ct
import numpy as np
from test_lumina_gpu_compact_device import gpu,tree_bytes,test_compact_device_map_keeps_CPU_source_transmission_and_total_error as ordinary
from test_lumina_gpu_prefix_device import test_thick_native_bin_uses_prefix_not_whole_bin_uncertainty as thick
from test_lumina_gpu_measure import Input
from test_lumina_characteristic_linear import reference
from lumina_gpu_compact import CompactMap,MAP_BYTES
from lumina_gpu_linear import Piece
from lumina_gpu_driver import check


def stats(gpu,reset=False):
    f=gpu.lib.lumina_linear_cuda_hybrid_stats
    f.argtypes=[ct.c_int,ct.POINTER(ct.c_ulonglong)];f.restype=ct.c_int
    values=(ct.c_ulonglong*4)();assert f(int(reset),None if reset else values)==0
    return list(values)


def test_successful_primary_needs_no_scalar_fallback(gpu):
    stats(gpu,True);ordinary(gpu,.375,0.,False)
    assert stats(gpu)==[0,0,0,0]


def test_thick_bin_accuracy_does_not_assume_that_primary_fails(gpu):
    stats(gpu,True);thick(gpu,1e10)
    attempted,completed,unresolved,ticks=stats(gpu)
    assert completed==attempted and unresolved==0


def test_limit_fallback_returns_independent_bin_integral_without_unresolved_maps(gpu):
    # CPU 같은 입력: source31/native8패널. 전체20을10+10으로나누면
    # 기본work-limit가실제로대체분기를요구한다. 물리식/오차는바꾸지않는다.
    stats(gpu,True);data=Input(gpu,9);buffers=[];p=Piece();h=.001;shift=.37;span=6.8
    da=(shift+span)*h;db=shift*h
    try:
        data.arrays[0][0]=0.;data.arrays[0][0,:,0]=1.
        data.arrays[0][1]=1.;data.arrays[0][2]=2.;data.host.jmode=data.device.jmode=0
        gpu.upload(data.buffers[0],data.arrays[0])
        check(gpu.lib.lumina_linear_cuda_plan(ct.byref(data.device),data.locations,1.,h,da,db,0,9,2,ct.byref(p)))
        nodes=gpu.allocate(tree_bytes(p));maps=gpu.allocate(9*MAP_BYTES);buffers=[nodes,maps]
        check(gpu.lib.lumina_linear_cuda_build(ct.byref(p),nodes.address,gpu.status))
        check(gpu.lib.lumina_linear_cuda_query(ct.byref(p),nodes.address,1e-8,20,maps.address,gpu.status))
        raw=gpu.download(maps,(9,MAP_BYTES),'u1');actual=CompactMap.from_buffer_copy(raw[0].tobytes())
        values=np.zeros((4,9));values[:2,0]=1.;values[2:]=2.
        expected=reference(values,shift=p.offset,span=p.span,delta=da-db)
        value=actual.source.positive().value();bound=actual.source_error.positive().value()
        assert abs(value-expected)<=bound+2e-13*expected and 0<=bound<=1e-8*value
        attempted,completed,unresolved,ticks=stats(gpu)
        assert attempted>0 and completed==attempted and unresolved==0 and ticks>0
    finally:
        for buffer in reversed(buffers):buffer.close()
        data.close()
