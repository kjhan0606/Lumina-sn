"""원천 측도 GPU 후보. B 기본 바이너리와 별도 플래그/ABI로 선택한다."""
from lumina_gpu_linear import Runtime as LinearRuntime,LinearDriver,memory_layout as linear_layout

SOURCE_NODE_BYTES=856
NODE_BYTES=272+SOURCE_NODE_BYTES
PROJECTION_BYTES=2*26*33*8


class Runtime(LinearRuntime):
    def __init__(self,library,baseline_free):
        super().__init__(library,baseline_free,_node_bytes=NODE_BYTES)


def memory_layout(nf,width):
    return linear_layout(nf,width,node_bytes=NODE_BYTES,extra_bytes=PROJECTION_BYTES)


class MeasureDriver(LinearDriver):
    node_bytes=NODE_BYTES
    projection_bytes=PROJECTION_BYTES
    engine='SPACE_LINEAR_NATIVE_SOURCE_MEASURE_GPU'
