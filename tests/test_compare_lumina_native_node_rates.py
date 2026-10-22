import numpy as np
import pytest

from compare_lumina_native_node_rates import project_rows


def test_project_rows_log_frequency_and_endpoints():
    nu=np.exp(np.array([0.,1.,2.]))
    target=np.exp(np.array([0.,.5,1.5,2.]))
    data=np.array([[1.,3.,5.],[0.,4.,8.]])
    np.testing.assert_allclose(project_rows(nu,data,target),
                               [[1.,2.,4.,5.],[0.,2.,6.,8.]],rtol=0,atol=1e-15)


@pytest.mark.parametrize('target',[
    np.exp(np.array([-.1,1.])),
    np.exp(np.array([1.,2.1])),
    np.exp(np.array([1.,.5])),
])
def test_project_rows_rejects_extrapolation_or_nonmonotone(target):
    with pytest.raises(ValueError):
        project_rows(np.exp(np.array([0.,1.,2.])),np.ones((2,3)),target)
