"""변경된 구간 알고리즘을 같은 실제 장치 산술로 두 순서에서 대조한다."""
import os
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from test_lumina_gpu_compact_device import gpu
from test_lumina_compiled_input import packet
from lumina_compiled_history import SmallDirectProviderTransport, _TrialField
from lumina_compiled_input import CompiledInput
from lumina_nodal_driver import CompiledDriverPlan, DriverKernel
from lumina_gpu_hybrid import HybridDriver, SlabLocalHybridDriver
from lumina_gpu_driver import Work
import time

pytestmark = pytest.mark.skipif(not os.environ.get('LUMINA_SLAB_CPU_LIBRARY'),
                                reason='명시적 Slurm 장치 실행에서만 수행')


def test_two_orders_produce_identical_complete_device_intervals(gpu, tmp_path):
    cpu = os.environ['LUMINA_SLAB_CPU_LIBRARY']
    g, source, _ = packet(tmp_path, physical=True)
    chi, es, eta = source.read(np.arange(24), np.arange(8))
    boundary = SmallDirectProviderTransport(g, source.frequency, chi, eta, es, 14000.,
        np.zeros(8), history=DriverKernel(cpu), panels=16, input_library=cpu, source=source)
    plan = CompiledDriverPlan(g, source, panels=1, block_width=2)
    outputs = []
    indices = (0,len(plan.arrays['steps'])//2,len(plan.arrays['steps'])-1)
    with source:
        field = _TrialField(source, np.zeros((source.ns,source.nf)))
        with CompiledInput(cpu, g, source, field) as provider:
            provider.bind_view(); provider.prepare_dense()
            for cls in (HybridDriver, SlabLocalHybridDriver):
                driver = cls(gpu, provider, cpu, block_width=2)
                try:
                    values = []
                    for index in indices:
                        gpu.upload(driver.ray,np.geomspace(1e-12,1e-6,source.nf))
                        work = Work(2000000*source.nf,time.monotonic()+10.)
                        report = driver.interval(plan,index,work,max_seconds=10.)
                        assert report['global_error'] <= 1.
                        values.append(gpu.download(driver.ray,source.nf))
                    outputs.append(np.array(values))
                finally:
                    driver.close()
    np.testing.assert_array_equal(outputs[0], outputs[1])
