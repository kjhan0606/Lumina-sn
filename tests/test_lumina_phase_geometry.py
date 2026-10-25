"""같은 기존 C 기하의 phase 절점을 Python 독립 root/proper 길이와 대조한다."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/'scripts'))
from lumina_phase_geometry import PhaseGeometry
from lumina_nodal_driver import CompiledDriverPlan
from lumina_local_micro import Interval
from test_lumina_nodal_driver import driver_library
from test_lumina_compiled_input import packet


@pytest.mark.parametrize('subdivisions', [1, 2, 4])
def test_c_phase_grid_preserves_real_doppler_radius_length_and_order(driver_library, tmp_path, subdivisions):
    g, source, _ = packet(tmp_path, physical=True)
    with source, PhaseGeometry(driver_library, g, source) as factory:
        plan = CompiledDriverPlan(g, source, panels=1)
        tangent = int(np.flatnonzero(plan.arrays['steps'][:, 1] == 0.)[0])
        for index in (0, 1000, tangent, len(plan.arrays['steps'])-1):
            step = plan.arrays['steps'][index]; interval = Interval(g, step)
            grid = factory.grid(step, subdivisions)
            span = interval.point(0.)[1]/source.dlognu
            expected_steps = max(1, int(np.ceil(span*subdivisions)))
            assert len(grid['length']) == expected_steps
            assert grid['x'][0] == 0 and grid['x'][-1] == 1
            assert np.all(np.diff(grid['x']) > 0) and np.all(grid['drift'] >= 0)
            expected_x = [0.]
            for k in range(1, expected_steps):
                target = (expected_steps-k)*source.dlognu/subdivisions
                expected_x.append(brentq(lambda x: interval.point(float(x))[1]-target, expected_x[-1], 1.,
                                         xtol=4*np.finfo(float).eps, rtol=4*np.finfo(float).eps))
            expected_x.append(1.)
            np.testing.assert_allclose(grid['x'], expected_x, rtol=0, atol=2e-12)
            points = [interval.point(float(x)) for x in expected_x]
            np.testing.assert_allclose(grid['radius'], [p[0] for p in points], rtol=2e-13)
            np.testing.assert_allclose(grid['phase'], np.array([p[1] for p in points])/source.dlognu, rtol=0, atol=2e-12)
            lengths = [interval.geometry(a, b) for a, b in zip(expected_x[:-1], expected_x[1:])]
            np.testing.assert_allclose(grid['length'], lengths, rtol=2e-12)


def test_zero_drift_retains_one_real_spatial_cell_and_native_endpoints(driver_library, tmp_path):
    g, source, _ = packet(tmp_path, physical=True); g = replace(g, velocity=np.zeros_like(g.velocity))
    with source, PhaseGeometry(driver_library, g, source) as factory:
        plan = CompiledDriverPlan(g, source, panels=1)
        for index in (0, len(plan.arrays['steps'])//2, len(plan.arrays['steps'])-1):
            step = plan.arrays['steps'][index]; grid = factory.grid(step)
            np.testing.assert_array_equal(grid['x'], [0., 1.])
            np.testing.assert_array_equal(grid['phase'], [0., 0.])
            np.testing.assert_array_equal(grid['drift'], [0.])
            assert np.isclose(grid['length'][0], step[2]-step[1], rtol=2e-15)
            np.testing.assert_array_equal(grid['radius'], step[3:5])


def test_bad_native_endpoint_is_rejected_before_grid_publication(driver_library, tmp_path):
    g, source, _ = packet(tmp_path, physical=True)
    with source, PhaseGeometry(driver_library, g, source) as factory:
        plan = CompiledDriverPlan(g, source, panels=1); bad = plan.arrays['steps'][0].copy()
        bad[1] = np.nextafter(bad[1], 0.)
        with pytest.raises(ValueError, match='조회 실패: 8'): factory.grid(bad)
