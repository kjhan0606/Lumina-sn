import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from lumina_native_heat_sources import fortran_number,vector,cooling_vector,artificial_heat


def test_fortran_omitted_exponent_is_not_zero_or_mantissa():
    assert fortran_number('4.3606-173')==4.3606e-173
    assert fortran_number('-9.3428-205')==-9.3428e-205
    assert fortran_number('2.5D+02')==250.
    for text in ('nan','inf','************'):
        with pytest.raises(ValueError):fortran_number(text)


def test_vector_is_exact_length_and_not_padded():
    text='Heating\n 1. 2.D+00\n 4.3606-173\nNext header\n'
    np.testing.assert_array_equal(vector(text,'Heating',3),[1.,2.,4.3606e-173])
    with pytest.raises(ValueError):vector(text,'Heating',4)
    with pytest.raises(ValueError):vector(text+'Heating\n1\n','Heating',3)


def test_cooling_uses_depth_addresses_and_explicit_zero_writer_contract():
    text='Depth\n3 4\nSource\n3. 4.\n\nDepth\n1 2\nSource\n1. 2.\n'
    np.testing.assert_array_equal(cooling_vector(text,'Source',4),[1.,2.,3.,4.])
    with pytest.raises(ValueError):cooling_vector(text,'Missing',4)
    np.testing.assert_array_equal(cooling_vector(text,'Missing',4,omitted_sum_zero=True),np.zeros(4))
    for changed in (text.replace('1 2','1 3'),text.replace('1 2','1'),text.replace('Source\n1. 2.','Absent\n1. 2.')):
        with pytest.raises(ValueError):cooling_vector(changed,'Source',4,omitted_sum_zero=True)


def test_artificial_heat_keeps_tiny_nonzero_and_physical_density_factors():
    t=np.array([8878.,11000.]);ne=np.array([1.e8,2.e8]);ni=np.array([3.e7,4.e7])
    a=artificial_heat(t,ne,ni,5000.)
    assert 0<a[0]<1.e-170
    np.testing.assert_array_equal(artificial_heat(t,2*ne,ni,5000.),2*a)
    np.testing.assert_allclose(artificial_heat(t+10,ne,ni,5000.),a*np.exp(-1),rtol=1.e-13)
    np.testing.assert_array_equal(artificial_heat(t,ne,ni,0.),np.zeros(2))
