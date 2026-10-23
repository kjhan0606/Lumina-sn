"""공간 세분 광선 커널의 원 native 해와 일정 source 해를 확인한다."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from check_cmfgen_local_sc_packet import replay_ray
from cmfgen_siii_full_path_kernel import refine_nodes, sweep


def test_full_path_native_matches_existing_replay_and_refined_constant_source():
    n = 7
    chi = np.full(n, .2)
    eta = np.full(n, .6)
    geom = dict(R=np.ones(n), S_M=np.arange(n,dtype='f8'),
                S_P=np.arange(n,0,-1,dtype='f8'),
                B_M=np.zeros(n), B_P=np.zeros(n))
    previous = dict(I_M=np.zeros(n), I_P=np.zeros(n))
    native_p,native_m = replay_ray(dict(dLOG_NU=.01), geom, previous,
                                   chi, eta, 1, inner_plus=0.)
    for pieces in (1, 2, 8, 128):
        args = (chi, eta, refine_nodes(geom['S_M'],pieces),
                refine_nodes(geom['S_P'],pieces),
                refine_nodes(geom['B_M'],pieces),
                refine_nodes(geom['B_P'],pieces),
                refine_nodes(previous['I_M'],pieces),
                refine_nodes(previous['I_P'],pieces), .01, 0., True, pieces)
        plus,minus = sweep(*args)
        if pieces == 1:
            np.testing.assert_allclose(plus,native_p,rtol=1e-13)
            np.testing.assert_allclose(minus,native_m,rtol=1e-13)
        np.testing.assert_allclose(minus[-1],3.*(-np.expm1(-.2*(n-1))),rtol=1e-13)
        np.testing.assert_allclose(plus[0],3.*(-np.expm1(-.2*(n-1))),rtol=1e-13)
