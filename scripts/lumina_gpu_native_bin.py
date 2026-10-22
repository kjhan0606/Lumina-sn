"""원 native 경계별 정확한 적분·오차 배분 후보의 명시적 GPU 선택."""
from lumina_gpu_compact import EndpointCompactDriver


class NativeBinDriver(EndpointCompactDriver):
    source_integrator=1
    engine='SPACE_LINEAR_NATIVE_BIN_STREAM_GPU'
    interval_limit_seconds=3400.
