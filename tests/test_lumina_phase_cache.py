"""기하 재사용은 같은 C 결과 그대로이며 물질/J 갱신과 독립이다."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/'scripts'))
from lumina_phase_cache import CachedPhaseGeometry, build_cache
from lumina_phase_geometry import PhaseGeometry
from lumina_nodal_driver import CompiledDriverPlan
from lumina_native_columns import sha
from test_lumina_nodal_driver import driver_library
from test_lumina_compiled_input import packet


def small_plan(geometry, source):
    original = CompiledDriverPlan(geometry, source, panels=1)
    arrays = original.arrays.copy()
    arrays['steps'] = arrays['steps'][[0, 1000, len(arrays['steps'])//2, -1]].copy()
    return SimpleNamespace(source=source, arrays=arrays)


def test_cached_phase_is_byte_identical_readonly_and_reusable(driver_library, tmp_path):
    g, source, _ = packet(tmp_path/'input', physical=True)
    with source, PhaseGeometry(driver_library, g, source) as factory:
        plan = small_plan(g, source); target = tmp_path/'cache'
        record = build_cache(target, factory, plan)
        assert record['complete'] and record['actual_F_calls'] == 0 and record['completed_intervals'] == 4
        before = sha(target/'geometry.json')
        with CachedPhaseGeometry(target, factory, plan) as cached:
            for step in plan.arrays['steps']:
                expected, actual = factory.grid(step), cached.grid(step)
                assert factory.count(step) == len(expected['length'])
                for name in ('x', 'radius', 'phase', 'locations', 'length', 'drift'):
                    np.testing.assert_array_equal(actual[name], expected[name])
                    assert not actual[name].flags.writeable
            with pytest.raises(ValueError, match='세분'): cached.grid(plan.arrays['steps'][0], 2)
            outside = plan.arrays['steps'][0].copy(); outside[0] += 1
            with pytest.raises(ValueError, match='cache 밖'): cached.grid(outside)
        # 물질 자체의 정체성은 전체장 입력이 검증한다. 기하 cache는 물질 신원을 키로 삼지 않는다.
        old_identity = getattr(source, 'identity', None); source.identity = dict(role='OTHER_MATERIAL_SAME_GEOMETRY')
        with CachedPhaseGeometry(target, factory, plan): pass
        source.identity = old_identity
        assert sha(target/'geometry.json') == before
        with pytest.raises(FileExistsError): build_cache(target, factory, plan)


def test_cache_rejects_changed_grid_or_bytes(driver_library, tmp_path):
    g, source, _ = packet(tmp_path/'input', physical=True)
    with source, PhaseGeometry(driver_library, g, source) as factory:
        plan = small_plan(g, source); target = tmp_path/'cache'; build_cache(target, factory, plan)
        old_h = source.dlognu; source.dlognu *= 1.01
        with pytest.raises(ValueError, match='같은 원 phase'): CachedPhaseGeometry(target, factory, plan)
        source.dlognu = old_h
        with (target/'length.npy').open('ab') as stream: stream.write(b'bad')
        with pytest.raises(ValueError, match='원본 변경'): CachedPhaseGeometry(target, factory, plan)


def test_incomplete_cache_cannot_be_used(driver_library, tmp_path):
    g, source, _ = packet(tmp_path/'input', physical=True)
    with source, PhaseGeometry(driver_library, g, source) as factory:
        plan = small_plan(g, source); target = tmp_path/'cache'
        with pytest.raises(TimeoutError): build_cache(target, factory, plan, seconds=1e-12)
        record = json.loads((target/'geometry.json').read_text())
        assert record['complete'] is False and record['actual_F_calls'] == 0
        with pytest.raises(ValueError, match='완성된'): CachedPhaseGeometry(target, factory, plan)
