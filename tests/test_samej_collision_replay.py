from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from replay_lumina_samej_collisions import compare_rows


def test_compare_keeps_zero_flow_and_absolute_difference():
    key = (14,1,0)
    ref = {key:dict(native_raw=-3.)}
    result = compare_rows({key:dict(total=-2.,absolute_flow=10.)},ref)[0]
    assert result['difference'] == 1.
    assert result['difference_over_C_absolute_flow'] == .1
    zero = compare_rows({key:dict(total=0.,absolute_flow=0.)},ref)[0]
    assert zero['difference'] == 3.
    assert zero['difference_over_C_absolute_flow'] is None
