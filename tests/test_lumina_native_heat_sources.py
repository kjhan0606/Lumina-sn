import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from lumina_native_heat_sources import fortran_number,vector,cooling_vector,artificial_heat,instantaneous_profile


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
    # native는 먼저 double T4로 바꾼다. 두 반올림된 지수의 차이는 -1에서
    # 약1e-13 달라질 수 있으므로 같은 이진 입력의 지수 차이를 기준으로 쓴다.
    e0=1000.*(.5-t*1.e-4);e1=1000.*(.5-(t+10)*1.e-4)
    np.testing.assert_allclose(artificial_heat(t+10,ne,ni,5000.),a*np.exp(e1-e0),
                               rtol=8*np.finfo(float).eps)
    np.testing.assert_array_equal(artificial_heat(t,ne,ni,0.),np.zeros(2))


def test_instantaneous_uses_verified_pure_isotope_current_number_not_a_mass_conversion(tmp_path):
    header=('Time(days) since explosion: 1.0\n\nRadius grid (10^10 cm)\n1. 2.\n\n')
    for name in ('NICK','COB'):
        for suffix in (' mass fraction',' 56 mass fraction'):header+=name+suffix+'\n.1 .2\n\n'
    (tmp_path/'SN_HYDRO_FOR_NEXT_MODEL').write_text(header)
    (tmp_path/'NUC_DECAY_DATA').write_text('NICK 55.942132 1.0 COB 55.939839 1.0 F 6\n'
                                         'COB 55.938904 2.0 IRON 55.934937 2.0 S 15\n')
    columns=np.array([[28,1,0],[28,2,-1],[27,1,0]])
    pop=np.array([[1.,3.],[2.,4.],[5.,6.]])
    actual,meta=instantaneous_profile(tmp_path,columns,pop,np.array([1.e10,2.e10]),86400.)
    np.testing.assert_allclose(actual,np.array([8.,13.])*np.log(2)/86400*1.602176634e-6,rtol=2.e-15)
    assert not meta['density_mass_conversion'] and not meta['new_interpolation']
    (tmp_path/'SN_HYDRO_FOR_NEXT_MODEL').write_text(header.replace('NICK 56 mass fraction\n.1 .2',
                                                                 'NICK 56 mass fraction\n.05 .2'))
    with pytest.raises(ValueError):instantaneous_profile(tmp_path,columns,pop,np.array([1.e10,2.e10]),86400.)
