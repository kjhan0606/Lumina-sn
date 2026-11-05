import numpy as np
from analyze_lumina_stored_photo_feedback import split_change


def test_symmetric_split_is_exact_and_reverses_with_endpoints():
    a,b,c,d=(np.array(x,dtype=float) for x in ([2,7],[3,9],[5,4],[11,6]))
    f,m=split_change(a,b,c,d)
    np.testing.assert_array_equal(f+m,d-a)
    rf,rm=split_change(d,c,b,a)
    np.testing.assert_array_equal(rf,-f)
    np.testing.assert_array_equal(rm,-m)


def test_fixed_population_or_field_removes_only_its_term():
    a=np.array([2.,4.]);b=np.array([3.,1.])
    f,m=split_change(a,b,a,b)
    np.testing.assert_array_equal(f,b-a)
    np.testing.assert_array_equal(m,0.)
    f,m=split_change(a,a,b,b)
    np.testing.assert_array_equal(f,0.)
    np.testing.assert_array_equal(m,b-a)


def test_population_weighted_rate_change_is_not_field_change():
    w0=np.array([.8,.2]); w1=np.array([.2,.8])
    q0=np.array([1.,10.]);q1=q0*1.1
    g00,g01,g10,g11=w0@q0,w0@q1,w1@q0,w1@q1
    f,m=split_change(g00,g01,g10,g11)
    assert m>f>0
    np.testing.assert_allclose(f+m,g11-g00,rtol=1.e-15)
