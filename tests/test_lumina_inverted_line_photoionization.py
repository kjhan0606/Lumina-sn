"""BF 직접 기여가 실제 Si II 하위준위와 경로확률을 사용하는지 확인한다."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_lumina_inverted_line_photoionization import parse_args, weighted_photoionization


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
    assert set(args) == {'root', 'levels_path'} and args['levels_path'] == Path('/tmp/levels.csv')
