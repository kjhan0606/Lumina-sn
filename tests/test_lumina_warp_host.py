"""협력 상태 비용을 숨기거나 다른 ABI를 섞지 않는 직접시험."""
from types import SimpleNamespace
import pytest
from lumina_gpu_compact import CompactDriver,EndpointCompactDriver,WarpCompactDriver


def runtime(cooperative,compatible=0):
    lib=SimpleNamespace(lumina_linear_cuda_node_bytes=lambda:1128,
        lumina_linear_cuda_map_bytes=lambda:56,lumina_linear_cuda_endpoint_bytes=lambda nf:4*nf*8,
        lumina_linear_cuda_cooperative_bytes=lambda:cooperative,
        lumina_linear_cuda_cooperative_device=lambda:compatible)
    return SimpleNamespace(lib=lib)


def test_concurrent_shared_memory_stays_inside_original_GiB():
    d=WarpCompactDriver.__new__(WarpCompactDriver);layout=d.layout_for(1115020,256)
    assert d.cooperative_shared_bytes==21233664 and layout['cooperative_shared']==21233664
    assert layout['workspace']==1070005160<1024**3
    assert layout['tree']==EndpointCompactDriver.__new__(EndpointCompactDriver).layout_for(1115020,256)['tree']


def test_cooperative_binary_and_scalar_driver_refused_before_allocation():
    with pytest.raises(ValueError,match='협력 상태 비용'):
        EndpointCompactDriver(runtime(WarpCompactDriver.cooperative_shared_bytes),SimpleNamespace(nf=17),'unused.so')


def test_scalar_binary_and_cooperative_driver_refused_before_allocation():
    with pytest.raises(ValueError,match='협력 상태 비용'):
        WarpCompactDriver(runtime(0),SimpleNamespace(nf=17),'unused.so')


def test_incompatible_device_refused_before_allocation():
    with pytest.raises(RuntimeError):
        WarpCompactDriver(runtime(WarpCompactDriver.cooperative_shared_bytes,3),SimpleNamespace(nf=17),'unused.so')
