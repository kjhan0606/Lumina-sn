"""두꺼운 native bin의 출구 원천을 독립 Decimal 해석해로 대조한다."""
import ctypes as ct
from decimal import Decimal,localcontext

import pytest
from test_lumina_gpu_compact_device import gpu,tree_bytes
from test_lumina_gpu_measure import Input
from lumina_gpu_compact import CompactMap,MAP_BYTES
from lumina_gpu_linear import Piece
from lumina_gpu_driver import check


@pytest.mark.parametrize('chi',[1e8,1e10])
def test_thick_native_bin_uses_prefix_not_whole_bin_uncertainty(gpu,chi):
    data=Input(gpu,9);buffers=[];p=Piece();h=.001;da=(.37+6.8)*h;db=.37*h
    try:
        data.arrays[0][0]=chi;data.arrays[0][1]=1.;data.arrays[0][2]=2.
        data.host.jmode=data.device.jmode=0
        gpu.upload(data.buffers[0],data.arrays[0])
        check(gpu.lib.lumina_linear_cuda_plan(ct.byref(data.device),data.locations,1.,h,da,db,0,9,2,ct.byref(p)))
        nodes=gpu.allocate(tree_bytes(p));maps=gpu.allocate(9*MAP_BYTES);buffers=[nodes,maps]
        check(gpu.lib.lumina_linear_cuda_build(ct.byref(p),nodes.address,gpu.status))
        check(gpu.lib.lumina_linear_cuda_query(ct.byref(p),nodes.address,1e-8,100000,maps.address,gpu.status))
        raw=gpu.download(maps,(9,MAP_BYTES),'u1')
        with localcontext() as c:
            c.prec=90;D=Decimal.from_float;depth=D(chi)+3*D(da-db)
            expected=D(2.)*(1-(-depth).exp())/depth
            for row in raw:
                value=CompactMap.from_buffer_copy(row.tobytes())
                source=value.source;bound=value.source_error
                actual=D(source.m)*Decimal(2)**source.e;error=D(bound.m)*Decimal(2)**bound.e
                assert actual>0 and abs(actual-expected)<=error and error<=D(1e-8)*actual
    finally:
        for block in reversed(buffers):block.close()
        data.close()
