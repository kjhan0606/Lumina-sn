"""완성장 신원과 동일 격자 BF 비교의 경계 시험."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import compare_cmfgen_direct_native_rates as D


def fixture():
    plan={'reference_field_sha256':'old-hash'}
    inputs=dict(schema='lumina-direct-continuum-reference-v1',input_state_record=70,
        next_native_iteration=71,expected_depths=90,expected_frequencies=197291,
        changed_controls={'CROSS':'F -> T'},baseline_field_sha256='old-hash')
    runtime=dict(state='FIELD_COMPLETE',field_complete=True,coverage={'count':197291},
        execution={'returncode':97,'stop_reason':None},material_byte_unchanged=True,
        geometry_and_frequency_unchanged=True)
    return inputs,runtime,plan


def test_partial_direct_field_never_becomes_comparison_reference():
    inputs,runtime,plan=fixture()
    D.verify_direct(inputs,runtime,plan)
    runtime['field_complete']=False
    with pytest.raises(ValueError,match='완성된 직접'):
        D.verify_direct(inputs,runtime,plan)
    runtime['field_complete']=True
    runtime['coverage']['count']=110263
    with pytest.raises(ValueError,match='완성된 직접'):
        D.verify_direct(inputs,runtime,plan)


def test_direct_and_original_use_the_same_log_frequency_rule():
    nu=np.array([1.,2.,4.]);own_nu=np.array([1.,1.5,2.,3.,4.])
    original=np.array([[2.,4.,8.],[1.,2.,4.]])
    direct=3.*original
    own=np.array([[2.,3.,4.,6.,8.],[1.,1.5,2.,3.,4.]])
    target=np.array([1.,1.5,2.,3.,4.])
    result=D.direct_radiation_block(target,nu,original,direct,own_nu,own,
                                    np.log(nu),np.log(own_nu))
    np.testing.assert_allclose(result['CMFGEN_DIRECT'],3*result['CMFGEN'],rtol=0.,atol=1.e-14)
    np.testing.assert_array_equal(result['J_A'],own)
