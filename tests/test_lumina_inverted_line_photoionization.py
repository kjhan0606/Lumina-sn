"""BF 직접 기여가 실제 Si II 하위준위와 경로확률을 사용하는지 확인한다."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_lumina_inverted_line_photoionization import (
    collision_pair_flow, parse_args, radiative_line_flow, superlevel_line_network, weighted_photoionization)


def test_siII_not_siIII_and_probability_applied_once():
    rows = [dict(atomic_number=14, ion_number=q, super_level=sl, level_number=level, configuration=str(level))
            for q, sl, level in [(1, 0, 123), (1, 1, 124), (1, 2, 5), (2, 0, 0)]]
    columns = np.array([[14, 1, 0], [14, 1, 1], [14, 1, 2], [14, 2, 0]])
    populations = np.ones((4, 2)); fractions = np.ones((4, 2))
    roster = dict(lower=np.array([0, 0, 1, 2, 3]), Z=np.full(5, 14), stage=np.array([2, 2, 2, 2, 3]),
                  probability=np.array([.25, .75, .5, 1., 1.]), threshold=np.arange(1., 6.))
    rates = np.array([[4., 4.], [4., 4.], [2., 2.], [5., 5.], [1000., 1000.]])
    result = weighted_photoionization(rows, columns, populations, fractions, roster, rates, node=0)
    assert result['SiII_routes'] == 4 and result['total_rate_cm3_s'] == 10.
    assert [x['node_rate_cm3_s'] for x in result['levels']] == [4., 1.]
    assert result['combined_node_fraction'] == result['combined_max_fraction'] == .5
    roster['stage'][3] = 3
    roster['stage'][4] = 2
    with pytest.raises(ValueError, match='Si II lower'):
        weighted_photoionization(rows, columns, populations, fractions, roster, rates, node=0)


def test_cli_names_match_actual_analysis_arguments():
    args = vars(parse_args(['--root', '/tmp/input', '--levels', '/tmp/levels.csv']))
    assert set(args) == {'root', 'levels_path', 'BB_root', 'original_full', 'collision_run', 'collision_reference'}
    assert args['levels_path'] == Path('/tmp/levels.csv')


def test_inverted_line_retains_stimulated_and_spontaneous_downward_flow():
    result = radiative_line_flow([2., 2.], [6., 6.], [1.e-7, 0.], 1.e13, 100., .5)
    assert result['stimulated'][0] > result['absorption'][0]
    assert np.all(result['net_up'] < 0) and result['net_up'][1] == -600.
    with pytest.raises(ValueError): radiative_line_flow([2.], [6.], [-1.], 1.e13, 100., .5)


def test_superlevel_network_excludes_internal_lines_and_signs_cancel():
    rows = [dict(atomic_number=14, ion_number=1, super_level=sl, level_number=i)
            for i, sl in enumerate([0, 0, 1])]
    lines = dict(lower=np.array([0, 0, 1]), upper=np.array([1, 2, 2]),
                 nu0=np.full(3, 1.e13), A=np.full(3, 100.), g_ratio=np.full(3, .5))
    full = np.array([[2.], [3.], [6.]])
    jbar = np.zeros((3, 1)); status = np.ones(3, dtype='u1')
    result = superlevel_line_network(rows, full, lines, jbar, status, [0, 1], 1, 0)
    assert result[0]['cross_lines'] == result[1]['cross_lines'] == 2
    assert result[0]['incoming_cm3_s'] == result[1]['outgoing_cm3_s'] == 1200.
    assert result[0]['net_cm3_s'] == -result[1]['net_cm3_s']
    assert result[0]['inverted_line_net_cm3_s'] == 600.


def test_collision_pair_uses_occupation_ratio_and_detailed_balance():
    T, ne, gl, gu, gap, wl, wu = 10000., 1.e8, 2., 4., .3, .9, .8
    lower = 3.; upper = lower*(gu/gl)*(wu/wl)*np.exp(-gap/(8.617333262145e-5*T))
    result = collision_pair_flow(lower, upper, T, ne, 5., gl, gu, gap, wl, wu, True)
    assert result['net_up_cm3_s'] == pytest.approx(0., abs=1.e-12)
    assert result['dissolution_forward_cm3_s'] > 0
    assert collision_pair_flow(lower, upper, T, ne, 5., gl, gu, gap, wl, wu, False)['dissolution_forward_cm3_s'] == 0
    with pytest.raises(ValueError): collision_pair_flow(lower, upper, T, ne, 5., gl, gu, gap, .8, .9, True)
