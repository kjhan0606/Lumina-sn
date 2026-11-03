import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from compare_lumina_native_grid_heat import (
    compare_fine_field_to_eddfactor,
    field_input,
    fine_grid_trapezoid_weights,
    heat_components,
    interpolate_native_j,
    native_trapuneq_weights,
    parse_native_cont_freq_text,
    read_native_heat_controls,
)


def test_native_grid_parser_uses_only_declared_ncf_rows_and_preserves_map():
    text = """!NCF,          3
! ML Freq(10^15Hz) Freq(cont) Lam(Ang) dV(km/s) dV(cont)
  1 3.0000000E+00 2.8000000E+00 9.9930819E+03
  2 2.0000000E+00 1.5000000E+00 1.4989623E+04
  3 1.0000000E+00 1.0000000E+00 2.9979246E+04
! diagnostic block, not part of NCF
  1 9.0000000E+00 9.0000000E+00 1.0
"""
    nu, evaluation, meta = parse_native_cont_freq_text(text)
    np.testing.assert_array_equal(nu, np.array([1., 2., 3.]) * 1.e15)
    np.testing.assert_array_equal(evaluation, np.array([1., 1.5, 2.8]) * 1.e15)
    assert meta['ncf'] == 3
    assert 'not binary' in meta['printed_frequency_precision']


def test_native_grid_parser_rejects_missing_or_nonmonotone_integration_rows():
    missing = "!NCF, 2\n 1 2.0 2.0 1.0\n"
    with pytest.raises(ValueError, match='일부 누락'):
        parse_native_cont_freq_text(missing)
    nonmonotone = "!NCF, 2\n 1 1.0 1.0 1.0\n 2 2.0 2.0 1.0\n"
    with pytest.raises(ValueError, match='내림차순'):
        parse_native_cont_freq_text(nonmonotone)


def test_native_trapuneq_weights_match_nonuniform_nodal_formula():
    nu = np.array([1., 3., 8., 10.]) * 1.e15
    np.testing.assert_array_equal(native_trapuneq_weights(nu),
                                  np.array([1., 3.5, 3.5, 1.]) * 1.e15)


def test_fine_grid_weights_match_composite_trapezoid_on_nonuniform_grid():
    nu = np.array([1., 3., 8., 10.]) * 1.e15
    np.testing.assert_array_equal(fine_grid_trapezoid_weights(nu),
                                  np.array([1., 3.5, 3.5, 1.]) * 1.e15)


def test_field_manifest_resolves_and_hashes_one_sealed_input(tmp_path):
    root = tmp_path/'field'
    root.mkdir()
    path = root/'J.npy'
    path.write_bytes(b'sealed-field')
    import hashlib
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    found, found_hash = field_input({'input_sha256': {'J.npy': digest}}, 'J.npy', root)
    assert found == path
    assert found_hash == digest
    with pytest.raises(ValueError, match='unchanged J.npy'):
        field_input({'input_sha256': {'J.npy': '0'*64}}, 'J.npy', root)


def test_fine_field_and_eddfactor_comparison_reports_exact_shared_J():
    nu = np.array([1., 2., 4., 8.])
    j = np.arange(90*4, dtype=np.float64).reshape(90, 4)
    got = compare_fine_field_to_eddfactor(nu, j, nu.copy(), j.T.copy())
    assert got['max_frequency_relative_difference'] == 0.
    assert got['max_J_absolute_difference'] == 0.
    assert got['max_J_symmetric_relative_difference'] == 0.
    assert got['exact_J_values'] == got['total_J_values'] == 360


def test_native_heat_controls_determine_scattering_and_line_handling(tmp_path):
    vadat = tmp_path/'VADAT'
    vadat.write_text("T [COH_ES] ! coherent\nBLANK [GLOBAL_LINE] ! line mode\n")
    assert read_native_heat_controls(vadat) == {'COH_ES': 'T', 'GLOBAL_LINE': 'BLANK'}


def test_rd_cont_j_interpolation_is_linear_in_frequency_and_clamps_endpoints():
    source_nu = np.array([2., 4., 8.])
    source_j = np.array([[2., 20.], [4., 40.], [8., 80.]])
    target = np.array([1., 3., 6., 9.])
    got = interpolate_native_j(source_nu, source_j, target)
    np.testing.assert_array_equal(got, np.array([[2., 3., 6., 8.],
                                                 [20., 30., 60., 80.]]))


def test_thermal_heat_excludes_coherent_es_and_reports_bound_bound_separately(tmp_path):
    path = tmp_path/'sources.npz'
    nu = np.array([1., 2., 4.])
    shape = (1, 3)
    np.savez(path, nu_Hz=nu,
        bf_chi=np.full(shape, 2.), bf_eta=np.full(shape, .5),
        ff_chi=np.zeros(shape), ff_eta=np.zeros(shape),
        bb_chi=np.full(shape, .25), bb_eta=np.full(shape, .1),
        es_chi=np.full(shape, 100.))
    j = np.array([[1., 2., 3.]])
    weights = np.array([.5, 1.5, 1.])
    result, gross = heat_components(path, j, weights, nu)
    expected_bf = 4.*math.pi*np.sum((2.*j[0]-.5)*weights)
    expected_bb = 4.*math.pi*np.sum((.25*j[0]-.1)*weights)
    assert result['BF'][0] == pytest.approx(expected_bf)
    assert result['BB'][0] == pytest.approx(expected_bb)
    assert result['total_no_coherent_ES'][0] == pytest.approx(expected_bf+expected_bb)
    assert gross[0] > result['total_no_coherent_ES'][0]
