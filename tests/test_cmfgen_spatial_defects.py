"""공간 source 차이의 전달 분해를 독립 주입·gain·경계 대조로 확인한다."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_cmfgen_spatial_defects import defect_terms, analyze, probe_cell, QuadraticBlueKernel
from compare_cmfgen_fixed_blue_transport import build_kernel
from check_cmfgen_local_sc_packet import local_sc
from compare_cmfgen_packet_cell_material import linear_material_reference


@pytest.mark.parametrize('opaque', [False, True])
def test_single_injection_has_analytic_downstream_transmission(opaque):
    t = np.array([.8, .6, .9, .7])
    if opaque: t[3] = 0.
    e = np.zeros(5); injection = .3
    for i in range(4): e[i+1] = t[i]*e[i]+(injection if i == 1 else 0.)
    terms, boundary, closure = defect_terms(e, t)
    np.testing.assert_allclose(terms, [0., injection*t[2]*t[3], 0., 0.], atol=5e-17)
    assert boundary == 0. and closure < 1e-14


def test_gain_and_signed_local_terms_do_not_get_clipped():
    t = np.array([1.2, .7, 1.1]); injection = np.array([.4, -.5, .2]); e = np.empty(4); e[0] = .8
    for i in range(3): e[i+1] = t[i]*e[i]+injection[i]
    original = e.copy(); terms, boundary, closure = defect_terms(e, t)
    np.testing.assert_allclose(terms, injection*np.array([t[1]*t[2], t[2], 1.]), rtol=1e-14)
    assert terms[1] < 0 and boundary == pytest.approx(e[0]*t.prod()) and closure < 1e-14
    np.testing.assert_array_equal(e, original)


def test_empty_path_is_only_the_boundary():
    terms, boundary, closure = defect_terms([.3], [])
    assert len(terms) == 0 and boundary == .3 and closure == 0.


@pytest.mark.parametrize('e,t', [([], []), ([1., 2.], []), ([1., np.nan], [.2]),
                              ([1., 2.], [-.2]), ([1., 2.], [np.inf])])
def test_malformed_paths_fail(e, t):
    with pytest.raises(ValueError): defect_terms(e, t)


def test_existing_analysis_is_not_overwritten(tmp_path):
    marker = tmp_path/'keep'; marker.write_text('원본')
    with pytest.raises(FileExistsError): analyze(tmp_path, tmp_path, tmp_path, tmp_path)
    assert marker.read_text() == '원본'


def test_local_probe_shares_incoming_and_matches_independent_material_integral(tmp_path):
    kernel = build_kernel(tmp_path)[0]; n = 6
    ray = dict(R=np.arange(n, 0, -1, dtype=float), S_M=np.arange(n, dtype=float),
        S_P=-np.arange(n, dtype=float), B_M=np.zeros(n), B_P=np.zeros(n))
    g = dict(rays=[ray], ND_ADD=0)
    chi = np.linspace(.2, .7, n); eta = np.array([.1, .5, .3, .9, .2, .7])
    blue = dict(rays=[dict(I_P=np.zeros(n), I_M=np.zeros(n))])
    current = np.full(n, .4)
    current[2] = local_sc(chi, eta, blue['rays'][0]['I_P'], ray, current, 2, .01, 'P')
    actual = dict(rays=[dict(I_P=current)])
    report = probe_cell(kernel, dict(dLOG_NU=.01), g, blue, actual, chi, eta, 0, np.int64(2), 'P', 64)
    assert report['SC_replay_absolute_difference'] < 1e-14
    assert report['production_core_reference_absolute_difference'] < 1e-13
    assert abs(report['local_injection']) > .001
    import json
    assert json.loads(json.dumps(report))['native_outer_zero_based'] == 2


@pytest.mark.parametrize('length', [.00001, 1., 10000.])
def test_quadratic_source_matches_independent_integral_without_subdivision(tmp_path, length):
    build_kernel(tmp_path); kernel = QuadraticBlueKernel(tmp_path/'fixed-blue.so')
    chi=np.array([.2, 1.1]);eta=np.array([.3, .8]);b=np.array([.07, .3]);blue=np.array([.9, .2])
    exact,_=linear_material_reference(.4, chi, eta, b, blue, length, 9.)
    actual=kernel.leg([length],chi,eta,b,blue,9.,.4)[-1]
    assert actual == pytest.approx(exact, rel=3e-11)


def test_quadratic_source_has_no_silent_gain_fallback(tmp_path):
    build_kernel(tmp_path);kernel=QuadraticBlueKernel(tmp_path/'fixed-blue.so')
    with pytest.raises(ValueError):kernel.leg([1.],[-1.,-1.],[.2,.2],[0.,0.],[.3,.3],4.,.4)
    with pytest.raises(ValueError):kernel.leg([1.],[.1,.1],[.2,.2],[0.,0.],[.3,.3],4.,.4,64)
