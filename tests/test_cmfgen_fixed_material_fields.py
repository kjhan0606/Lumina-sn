"""물질이 같은지 소형 합성 응답으로 확인한다."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from compare_cmfgen_fixed_material_fields import compare,run


def responses():
    return {(d,Z,s):dict(Te_K='10000',ne_cm3='100',n_element='2',pre_fraction='0.2',candidate_fraction='0.2')
            for d in (1,2) for Z in (14,16) for s in range(2,7)}


def test_all_stages_not_just_III_are_retained():
    old=responses();new=deepcopy(old);new[2,14,5]['candidate_fraction']='0.8'
    r=compare(old,new,np.array([1.,2.]),2)
    assert r['rows']==20
    assert r['per_element_stages']['14']['5']['absolute_change_mass_median_p95_max'][2]==pytest.approx(.6)
    assert r['per_element_stages']['14']['3']['absolute_change_mass_median_p95_max'][2]==0


@pytest.mark.parametrize('key',['Te_K','ne_cm3','n_element','pre_fraction'])
def test_non_field_input_changes_are_refused(key):
    old=responses();new=deepcopy(old);new[1,16,2][key]='changed'
    with pytest.raises(ValueError,match='물질 변화'):compare(old,new,np.ones(2),2)


def test_partial_depths_and_existing_output_are_refused(tmp_path):
    old=responses();new=deepcopy(old);new.pop((2,14,6))
    with pytest.raises(ValueError,match='coverage'):compare(old,new,np.ones(2),2)
    with pytest.raises(ValueError,match='기존'):run(Path('not-read'),Path('not-read'),tmp_path)
