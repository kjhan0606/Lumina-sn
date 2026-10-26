from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_lumina_reference_response import parse_raw


def test_input_residual_rows_are_complete_atomic_addresses():
    original = {(14,1,0):3.,(14,5,-1):1.}
    text = ('[MODEL-SE][INPUT-RESIDUAL-ROW] Z=14 stage0=1 super0=0 '
            'bf_rad=1 bf_col=-2 bb_rad=3 bb_col=-4 total=-2 absolute_flow=100\n'
            '[MODEL-SE][INPUT-RESIDUAL-ROW] Z=14 stage0=5 super0=-1 '
            'bf_rad=-1 bf_col=2 bb_rad=0 bb_col=0 total=1 absolute_flow=3\n')
    result = parse_raw(text,original)
    assert result[14,1,0]['total'] == -2.
    assert result[14,5,-1]['absolute_flow'] == 3.
    with pytest.raises(ValueError,match='누락'):
        parse_raw(text.splitlines()[0],original)
    with pytest.raises(ValueError,match='주소'):
        parse_raw(text+text,original)
