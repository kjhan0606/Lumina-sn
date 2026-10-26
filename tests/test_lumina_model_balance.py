import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from lumina_model_balance import balance


def test_signed_adiabatic_and_no_double_counted_collision_heat():
    bf=np.array([[10.,10.],[8.,8.]]);bb=np.array([[5.,5.],[6.,6.]])
    ff=np.array([[1.,1.],[2.,2.]])
    result=balance(bf,bb,ff,np.array([2.,-2.]),np.array([3.,3.]),np.zeros(2))
    np.testing.assert_array_equal(result['net'],[1.,5.])
    np.testing.assert_array_equal(result['gross'],[37.,37.])
    np.testing.assert_allclose(result['gross_normalized'],[1/37,5/37])
    np.testing.assert_allclose(result['channel_normalized'],[1/9,5/9])


def test_large_scattering_gross_is_not_a_hidden_completion_criterion():
    values=np.array([[1.e12],[1.e12]])
    result=balance(values,values,values,np.ones(1),np.zeros(1),np.zeros(1))
    assert result['gross_normalized'][0]<1.e-10
    assert result['channel_normalized'][0]==1.


def test_missing_negative_and_nonfinite_channel_rejected():
    x=np.ones((2,2));z=np.zeros(2)
    for bad in (np.full((2,2),np.nan),-x,np.zeros((2,1))):
        with pytest.raises(ValueError):balance(x,x,bad,z,z,z)
