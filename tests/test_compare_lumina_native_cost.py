"""같은 coarse 비교의 표현 범위·표본 누락·오류 판정을 직접 확인한다."""
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from compare_lumina_native_cost import compare


def record():
    rows=[dict(column=k,code=0,t=dict(m=.75,e=-120000),source=dict(m=.75,e=-2000),
               error=dict(m=.5,e=-2040),panels=3,query_seconds=.1,reader_query=24) for k in (1,9)]
    return dict(status='CPU_SAMPLE_FINISHED_NO_MAP_FAILURES',baseline_sha256='sealed',source_identity={'J':0},
                interval=13003,coarse_only=True,full_frequency=False,piece={'span':1413.3},columns=[1,9],rows=rows)


def test_scaled_values_are_compared_without_final_double_underflow():
    original=record();updated=copy.deepcopy(original)
    assert compare(original,updated)['status']=='CPU_COARSE_COMPARISON_MET_NOT_FIELD'
    updated['rows'][0]['source']['m']=.76
    assert compare(original,updated)['status']=='CPU_COARSE_NOT_MET'


@pytest.mark.parametrize('bad',['identity','duplicate','missing','unfinished'])
def test_incompatible_or_incomplete_sample_is_rejected(bad):
    original=record();updated=copy.deepcopy(original)
    if bad=='identity':updated['source_identity']={'J':1}
    elif bad=='duplicate':updated['rows'][1]=copy.deepcopy(updated['rows'][0])
    elif bad=='missing':updated['rows'].pop()
    else:updated['status']='STARTED'
    with pytest.raises(ValueError):compare(original,updated)


def test_final_source_relative_error_is_still_required():
    original=record();updated=copy.deepcopy(original);updated['rows'][0]['error']={'m':.5,'e':-2000}
    assert compare(original,updated)['status']=='CPU_COARSE_NOT_MET'
