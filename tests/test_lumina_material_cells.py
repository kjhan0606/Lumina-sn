"""선형 부피 복원과 질량가중 온도를 해석적 적분과 비교한다."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_lumina_material_cells import remap


def test_linear_volume_profile_has_exact_cell_average_and_integral():
    r=np.array([5.,3.,2.,1.])
    x=r**3
    n=2+3*x
    cells=remap(r,np.full(4,10000.),2*n,np.ones(4),n[:,None],np.ones((1,4)))
    expected=2+1.5*(x[:-1]+x[1:])
    np.testing.assert_allclose(cells['population'][:,0],expected[::-1],rtol=0,atol=0)
    analytic=(4*np.pi/3)*(2*(125-1)+1.5*(125**2-1))
    assert np.sum(cells['population'][:,0]*cells['volume_cm3'])==pytest.approx(analytic,rel=2e-15)
    assert cells['radius_edges'][0]==1 and cells['radius_edges'][-1]==5


def test_mass_weighted_temperature_and_composition_are_not_point_averages():
    cells=remap(np.array([2.,1.]),np.array([20000.,10000.]),np.array([3.,1.]),
        np.array([3.,1.]),np.array([[30.],[10.]]),np.array([[.75,.25],[.25,.75]]))
    assert cells['Te_K'][0]==17500
    assert cells['rho_g_cm3'][0]==2
    np.testing.assert_array_equal(cells['mass_fractions'][:,0],[.625,.375])
    assert cells['population'][0,0]==20


@pytest.mark.parametrize('radius',[[1,2],[2,2],[2,-1]])
def test_wrong_geometry_is_not_sorted_or_extrapolated(radius):
    with pytest.raises(ValueError):
        remap(np.array(radius,dtype=float),np.ones(2),np.ones(2),np.ones(2),
              np.ones((2,1)),np.ones((1,2)))
