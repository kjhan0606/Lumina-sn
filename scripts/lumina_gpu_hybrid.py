"""같은 출력의 제한 실패만 별도 native kernel로 처리하는 명시적 선택."""
import ctypes as ct
from lumina_gpu_compact import WarpCompactDriver
from lumina_gpu_driver import CACHE_LIMIT,check


class HybridDriver(WarpCompactDriver):
    source_integrator=2
    interval_limit_seconds=3420.
    engine='SPACE_LINEAR_WARP_NATIVE_LIMIT_FALLBACK_GPU'

    def layout_for(self,nf,width):
        layout=super().layout_for(nf,width)
        layout['hybrid_counters']=32;layout['workspace']+=32
        if layout['workspace']>CACHE_LIMIT:raise MemoryError('대체 상태 포함 활성 작업공간1GiB 초과')
        return layout

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.stats=self.lib.lumina_linear_cuda_hybrid_stats
        self.stats.argtypes=[ct.c_int,ct.POINTER(ct.c_ulonglong)];self.stats.restype=ct.c_int
        check(self.stats(1,None))

    def integration_report(self):
        counters=(ct.c_ulonglong*4)();check(self.stats(0,counters))
        return dict(fallback_attempts=int(counters[0]),fallback_completed=int(counters[1]),
            fallback_unresolved=int(counters[2]),fallback_thread_clock_ticks=int(counters[3]),
            clock_role='sum of per-thread device cycles; not wall time or billed GPU seconds')


class SlabLocalHybridDriver(HybridDriver):
    """현재 주파수 묶음을 끝내어 맵 재생성을 줄이는 명시적 실행 선택."""
    slab_local_refinement=True
    engine='SPACE_LINEAR_WARP_NATIVE_SLAB_LOCAL_GPU'
