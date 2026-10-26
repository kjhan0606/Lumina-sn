"""원래점 준위용해 C 적분을 독립 continuum 계수와 비교한다."""
import ctypes as C
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import cmfgen_model_continuum as B
from lumina_model_point_rates import ModelPointBoundRates
from lumina_model_point_dissolution import contract_dissolution


def synthetic_point(tmp_path):
    library = tmp_path / 'point.so'
    subprocess.run(['gcc', '-O1', '-std=c11', '-fPIC', '-shared', '-Wl,-z,defs',
                    '-I' + str(ROOT / 'src'), '-o', str(library),
                    str(ROOT / 'src/bf_rate_jnu.c'),
                    str(ROOT / 'src/radiation_field.c'),
                    str(ROOT / 'src/seed_capability.c'), '-lm'], check=True)
    rows = [dict(atomic_number=14, ion_number=1, super_level=0, g='2', energy_eV='0'),
            dict(atomic_number=14, ion_number=2, super_level=0, g='2', energy_eV='0')]
    state = dict(columns=np.array([[14, 1, 0], [14, 2, 0], [14, 3, -1]]),
                 full=np.array([[1e7], [1e5]]),
                 superlevel=np.array([[1e7], [1e5], [1e-4]]),
                 occupation=np.ones((2, 1)),
                 sl_index={(14, 2, 0): 0, (14, 3, 0): 1, (14, 3, -1): 2},
                 temperature=np.array([1e4]), ne=np.array([1e8]))
    edge = 5e15
    route = B.Route(14, 2, 0, 1, 0, 0, .7, 2., 0., edge,
                    ((lambda nu: np.full_like(nu, 1e-18), np.array([edge]), edge, 'test'),),
                    dissolution_primary=True)
    point = object.__new__(ModelPointBoundRates)
    point.rows = rows
    point.models = {(14, 2): dict(z_screen=2.), (14, 3): dict(terminal_g=2.)}
    point.state = state
    point.node = 0
    point.temperature = 1e4
    point.routes = [route]
    point.sigma_edge = np.array([1e-18])
    point.upper_population = np.array([1e5])
    point.library = C.CDLL(str(library))
    point.ptr = C.POINTER(C.c_double)
    return point


def test_pseudo_rate_matches_independent_native_se_continuum(tmp_path):
    point = synthetic_point(tmp_path)
    edge = point.routes[0].threshold_Hz
    nu = edge * np.array([.75, .81, .9, .95, .99, .999, .9999, 1.1])
    width = np.full(len(nu), 1e12)
    j = np.full(len(nu), 1e-5)
    actual = contract_dissolution(point, nu, width, j, dissolution_enabled=True)
    assert actual.shape == (1, 6)
    assert np.all(actual[0, :3] > 0)
    direct = {key: np.zeros((1, len(nu))) for key in ('chi_abs', 'chi_stim', 'chi_net', 'eta_spont')}
    B.add_dissolved_continuum(direct, point.routes, point.rows, point.models,
                              point.state, nu, support_policy='native-SE')
    physical = point.physical_ion_rates(actual)[14, 2]
    weight = 4 * np.pi * width / (B.H * nu)
    expected = np.array([np.sum(weight * direct['chi_abs'][0] * j),
                         np.sum(weight * direct['eta_spont'][0]),
                         np.sum(weight * direct['chi_stim'][0] * j)])
    np.testing.assert_allclose(physical, expected, rtol=2e-14, atol=0)


def test_pseudo_requires_explicit_material_policy(tmp_path):
    point = synthetic_point(tmp_path)
    nu = np.array([4e15, 5e15])
    with pytest.raises(ValueError, match='활성화되지 않음'):
        contract_dissolution(point, nu, np.ones(2), np.ones(2))
