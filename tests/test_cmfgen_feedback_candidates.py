"""II/IV/V의 실패나 장 세대 차이를 III 한 단계의 개선으로 숨기지 않는다."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from compare_cmfgen_feedback_candidates import compare,run


def report(record,error):
    return dict(record=record,radiation_record=64,radiation_material_record=63,depths=90,rows=900,
        per_element={str(Z):dict(certified_and_committed=90,stages={str(s):dict(
            absolute_difference_mass_median_p95_max=[error/4,error/2,error],support_mismatch_count=0)
            for s in range(2,7)}) for Z in (14,16)})


def test_both_elements_and_all_stages_improve_without_claiming_final_convergence():
    r=compare(report(63,.3),report(64,.2),report(64,.01))
    assert r['decision']=='NEXT_OUTER_SUPPORTED' and r['selected_case']=='64B'
    assert not r['joint_convergence'] and not r['released_T']
    assert r['final_goal']=='NOT_JUDGED'


def test_good_III_cannot_hide_worse_V():
    B=report(64,.01)
    B['per_element']['14']['stages']['5']['absolute_difference_mass_median_p95_max']=[.1,.2,.4]
    assert compare(report(63,.3),report(64,.2),B)['decision']=='REVIEW_REQUIRED'


def test_support_mismatch_does_not_pass_with_small_numeric_errors():
    B=report(64,.01);B['per_element']['16']['stages']['2']['support_mismatch_count']=1
    assert compare(report(63,.3),report(64,.2),B)['selected_case'] is None


def test_different_field_or_uncommitted_C_is_refused():
    B=report(64,.01);B['radiation_record']=60
    with pytest.raises(ValueError):compare(report(63,.3),report(64,.2),B)
    B=report(64,.01);B['per_element']['14']['certified_and_committed']=89
    with pytest.raises(ValueError):compare(report(63,.3),report(64,.2),B)


def test_existing_output_is_preserved_before_inputs(tmp_path):
    with pytest.raises(ValueError,match='기존'):
        run(Path('not-read'),Path('not-read'),Path('not-read'),tmp_path)
