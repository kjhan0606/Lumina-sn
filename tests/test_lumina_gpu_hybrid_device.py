"""분리된 대체kernel이 두꺼운bin의 같은 물리해를 반환하는지 직접 확인한다."""
import ctypes as ct
from test_lumina_gpu_compact_device import gpu,test_compact_device_map_keeps_CPU_source_transmission_and_total_error as ordinary
from test_lumina_gpu_prefix_device import test_thick_native_bin_uses_prefix_not_whole_bin_uncertainty as thick


def stats(gpu,reset=False):
    f=gpu.lib.lumina_linear_cuda_hybrid_stats
    f.argtypes=[ct.c_int,ct.POINTER(ct.c_ulonglong)];f.restype=ct.c_int
    values=(ct.c_ulonglong*4)();assert f(int(reset),None if reset else values)==0
    return list(values)


def test_successful_primary_needs_no_scalar_fallback(gpu):
    stats(gpu,True);ordinary(gpu,.375,0.,False)
    assert stats(gpu)==[0,0,0,0]


def test_limit_fallback_returns_decimal_thick_bin_solution_without_unresolved_maps(gpu):
    stats(gpu,True);thick(gpu,1e10)
    attempted,completed,unresolved,ticks=stats(gpu)
    assert attempted>0 and completed==attempted and unresolved==0 and ticks>0
