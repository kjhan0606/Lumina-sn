import numpy as np
import pytest
import struct

from scripts.compare_cmfgen_bf_superlevel_rates import (
    CM_TO_EV,
    element_population_closure,
    population_identity_metrics,
    lumina_recombination_rate_density,
    milne_recombination_rate_density,
    parse_cmfgen_bf_channels,
    parse_f_to_s_int_seq,
    parse_model_do_lev_dissolution,
    cmfgen_runtime_edge_nu15,
    cmfgen_get_num_phot_routes_source_components,
    level_energies_and_thresholds,
    cmfgen_sup_to_full_v3b,
    lte_weighted_superlevel_photo_rate_density,
    raw_field_continuous_sigma_gamma,
    raw_field_step_sigma_gamma,
    match_scrtemp_state_to_final_populations,
    parse_prrr_temperature,
    parse_prrr_superlevel_rates,
    match_prrr_scrtemp_state,
    prrr_next_stage_sl1_population,
    rate_metrics,
    sum_full_level_populations_by_superlevel,
    total_photoionization_rate,
)


class _ScrtempFixture:
    state_first_it = 1

    def __init__(self, states):
        self.states = states
        self.last_it = len(states)

    def state(self, iteration):
        return self.states[iteration - 1].copy()


def test_match_prrr_to_unique_scrtemp_pre_solve_record():
    # State row 0 is next-ion SL1, row 1 is ED, row 2 is T in 1e4 K.
    states = np.array([
        [[9.0, 8.0], [1.0e9, 2.0e9], [1.0, 1.1]],
        [[5.0, 6.0], [3.0e9, 4.0e9], [1.2, 1.3]],
    ])
    profile = {
        "Si II": {
            "next_ion": "Si III", "ion_density": np.array([5.0, 6.0]),
            "electron_density": np.array([3.0e9, 4.0e9]),
            "temperature": np.array([12000.0, 13000.0]),
        },
    }

    record, state, metrics = match_prrr_scrtemp_state(
        _ScrtempFixture(states), {("Si III", 1): 0}, profile,
        electron_density_index=1, temperature_index=2)

    assert record == 2
    np.testing.assert_array_equal(state, states[1])
    assert metrics["Si II"]["next_SL1_max_rel"] == 0.0
    assert metrics["Si II"]["electron_density_max_rel"] == 0.0
    assert metrics["Si II"]["temperature_max_rel"] == pytest.approx(0.0)


def test_match_prrr_to_scrtemp_rejects_ambiguous_record():
    state = np.array([[5.0], [3.0e9], [1.2]])
    profile = {
        "Si II": {
            "next_ion": "Si III", "ion_density": np.array([5.0]),
            "electron_density": np.array([3.0e9]),
            "temperature": np.array([12000.0]),
        },
    }

    with pytest.raises(ValueError, match="exactly one SCRTEMP pre-solve record"):
        match_prrr_scrtemp_state(
            _ScrtempFixture(np.stack([state, state])),
            {("Si III", 1): 0}, profile,
            electron_density_index=1, temperature_index=2)


def test_match_final_pop_and_rvtj_to_unique_scrtemp_record():
    states = np.array([
        [[2.0, 3.0], [4.0, 5.0], [1.0e9, 2.0e9], [1.0, 1.1]],
        [[1.0, 1.0], [1.0, 1.0], [3.0e9, 4.0e9], [1.2, 1.3]],
    ])
    final = {"Sk2": np.array([[2.0, 4.0], [3.0, 5.0]])}

    record, state, metrics = match_scrtemp_state_to_final_populations(
        _ScrtempFixture(states), {("Sk2", 1): 0, ("Sk2", 2): 1},
        final, electron_density_index=2, temperature_index=3,
        rvtj_electron_density=np.array([1.0e9, 2.0e9]),
        rvtj_temperature=np.array([10000.0, 11000.0]))

    assert record == 1
    np.testing.assert_array_equal(state, states[0])
    assert metrics["per_ion_max_rel"]["Sk2"] == 0.0


def test_parse_f_to_s_int_seq_matches_cmfgen_row_contract(tmp_path):
    path = tmp_path / "SiII_F_TO_S"
    path.write_text(
        "header\n"
        "level_a 2.0 0.0 3.1 100.0 1 0 1\n"
        "level_b 4.0 20.0 2.9 110.0 1 7 2\n"
        "level_c 2.0 50.0 2.5 120.0 2 7 3\n")

    superlevels, names, sequence, edge_nu15 = parse_f_to_s_int_seq(
        path, n_full=3)

    np.testing.assert_array_equal(superlevels, [1, 1, 2])
    assert names == ["level_a", "level_b", "level_c"]
    np.testing.assert_array_equal(sequence, [0, 7, 7])
    np.testing.assert_allclose(edge_nu15, [3.1, 2.9, 2.5])


def test_parse_f_to_s_int_seq_accepts_legacy_implicit_row_order(tmp_path):
    path = tmp_path / "SiIV_F_TO_S_legacy"
    path.write_text(
        "header\n"
        "level_a 2.0 0.0 3.1 100.0 1 0\n"
        "level_b 4.0 20.0 2.9 110.0 1 0\n"
        "level_c 2.0 50.0 2.5 120.0 2 -7\n")

    superlevels, names, sequence, edge_nu15 = parse_f_to_s_int_seq(
        path, n_full=3)

    np.testing.assert_array_equal(superlevels, [1, 1, 2])
    assert names == ["level_a", "level_b", "level_c"]
    np.testing.assert_array_equal(sequence, [0, 0, -7])
    np.testing.assert_allclose(edge_nu15, [3.1, 2.9, 2.5])


def test_parse_f_to_s_int_seq_rejects_mixed_row_layouts(tmp_path):
    path = tmp_path / "mixed_F_TO_S"
    path.write_text(
        "header\n"
        "level_a 2.0 0.0 3.1 100.0 1 0\n"
        "level_b 4.0 20.0 2.9 110.0 1 0 2\n")

    with pytest.raises(ValueError, match="mixed F_TO_S row layouts"):
        parse_f_to_s_int_seq(path, n_full=2)


def test_parse_model_do_lev_dissolution_requires_explicit_single_flag(tmp_path):
    path = tmp_path / "MODEL"
    path.write_text("T [DO_DIS] Allow level dissolution\n")

    assert parse_model_do_lev_dissolution(path)

    path.write_text("T [DO_DIS] first\nF [DO_DIS] duplicate\n")
    with pytest.raises(ValueError, match="exactly one"):
        parse_model_do_lev_dissolution(path)


def test_cmfgen_runtime_edge_uses_genosc_ionization_minus_excitation_energy():
    edge = cmfgen_runtime_edge_nu15(
        100000.0, np.array([0.0, 25000.0, 50000.0]))

    np.testing.assert_allclose(
        edge, np.array([100000.0, 75000.0, 50000.0]) * 2.99792458e-5,
        rtol=2.0e-16, atol=0.0)
    with pytest.raises(ValueError, match="non-positive runtime edge"):
        cmfgen_runtime_edge_nu15(100000.0, np.array([100000.0]))


def test_cmfgen_route_enumerator_matches_source_save_variable_behavior():
    routes = cmfgen_get_num_phot_routes_source_components(
        final_state="TERM",
        upper_level_names=["TERM [1]", "TERM [2]", "TERM [3]", "TERM [4]"],
        upper_g=np.array([2.0, 4.0, 6.0, 8.0]),
        upper_energy_eV=np.array([0.0, 0.1, 0.2, 0.3]),
        upper_f_to_s=np.array([1, 2, 2, 3]))

    assert [route["target_superlevel_1based"] for route in routes] == [1, 2, 2, 3]
    assert [route["anchor_level_1based"] for route in routes] == [1, 2, 3, 4]
    np.testing.assert_allclose(
        [route["probability"] for route in routes], [0.1, 0.2, 0.3, 0.4],
        rtol=0.0, atol=1e-15)


def test_cmfgen_sup_to_full_v3b_returns_normalized_lte_on_super_weights():
    full, lte_on_super = cmfgen_sup_to_full_v3b(
        superlevel_population=np.array([[40.0, 60.0], [30.0, 70.0]]),
        lower_edge_nu15=np.array([10.0, 8.0, 6.0, 4.0]),
        lower_g=np.array([2.0, 4.0, 2.0, 4.0]),
        lower_f_to_s=np.array([1, 1, 2, 2]),
        int_seq=np.array([0, 7, 0, 7]),
        upper_edge_nu15=np.array([20.0, 19.0]),
        upper_g=np.array([2.0, 4.0]),
        upper_f_to_s=np.array([1, 1]),
        upper_sl1_population=np.array([50.0, 60.0]),
        temperature_1e4=np.array([1.0, 1.4]),
        electron_density=np.array([2.0e9, 3.0e9]),
        positive_ion_density=np.array([1.0e10, 2.0e10]),
        ion_charge=1.0,
        do_lev_dissolution=True,
        return_lte_full_on_super=True)

    grouped = np.column_stack((full[:, 0] + full[:, 1],
                               full[:, 2] + full[:, 3]))
    np.testing.assert_allclose(grouped, [[40.0, 60.0], [30.0, 70.0]],
                               rtol=2e-15, atol=0.0)
    assert np.isfinite(full).all() and np.all(full > 0.0)
    for sl in (1, 2):
        np.testing.assert_allclose(
            lte_on_super[:, np.array([1, 1, 2, 2]) == sl].sum(axis=1),
            np.ones(2), rtol=2e-15, atol=0.0)


def test_lte_weighted_superlevel_photo_rate_matches_wse_population_algebra():
    rate = lte_weighted_superlevel_photo_rate_density(
        superlevel_population=np.array([[40.0, 60.0]]),
        lte_full_on_super=np.array([[0.25, 0.75, 0.4, 0.6]]),
        full_level_superlevels=np.array([1, 1, 2, 2]),
        full_level_gamma=np.array([[2.0], [4.0], [5.0], [10.0]]))

    np.testing.assert_allclose(rate, [[140.0], [480.0]], rtol=0.0, atol=0.0)
    with pytest.raises(ValueError, match="do not normalize"):
        lte_weighted_superlevel_photo_rate_density(
            superlevel_population=np.array([[40.0, 60.0]]),
            lte_full_on_super=np.array([[0.25, 0.70, 0.4, 0.6]]),
            full_level_superlevels=np.array([1, 1, 2, 2]),
            full_level_gamma=np.array([[2.0], [4.0], [5.0], [10.0]]))


def test_raw_field_step_sigma_gamma_integrates_linear_j_with_threshold_mass_rule():
    gamma = raw_field_step_sigma_gamma(
        raw_nu=np.array([1.0, 2.0, 4.0]),
        raw_j_nu=np.array([3.0, 3.0, 3.0]),
        field_bin_edges=np.array([1.0, 2.0, 4.0]),
        field_states=np.array([1, 1]),
        sigma_bin_edges=np.array([1.0, 2.0, 4.0]),
        sigma_bin_average=np.array([1.0, 1.0]),
        nu_threshold=1.5)

    expected = (4.0 * np.pi / 6.62607015e-27 * 3.0 *
                (2.0 * np.log(2.0 / 1.5) + np.log(2.0)))
    assert gamma == pytest.approx(expected, rel=2e-15)


def test_raw_field_continuous_sigma_gamma_integrates_exact_sigma_nodes():
    def sigma(nu):
        return np.full(np.shape(nu), 2.0)

    gamma = raw_field_continuous_sigma_gamma(
        raw_nu=np.array([1.0, 2.0, 4.0]),
        raw_j_nu=np.array([3.0, 3.0, 3.0]),
        field_bin_edges=np.array([1.0, 4.0]),
        field_states=np.array([1]),
        sigma_terms=[(1.0, sigma, np.array([1.0, 4.0]), 1.0)])

    expected = 4.0 * np.pi / 6.62607015e-27 * 6.0 * np.log(4.0)
    assert gamma == pytest.approx(expected, rel=2e-12)


def test_cmfgen_sup_to_full_v3b_retries_backward_sequence_search():
    common = dict(
        superlevel_population=np.array([[10.0, 100.0]]),
        lower_edge_nu15=np.array([20.0, 15.0, 10.0]),
        lower_g=np.array([1.0, 1.0, 1.0]),
        lower_f_to_s=np.array([1, 2, 1]),
        upper_edge_nu15=np.array([21.0]),
        upper_g=np.array([1.0]),
        upper_f_to_s=np.array([1]),
        upper_sl1_population=np.array([50.0]),
        temperature_1e4=np.array([1.0]),
        electron_density=np.array([1.0e9]),
        positive_ion_density=np.array([1.0e10]),
        ion_charge=1.0,
        do_lev_dissolution=False,
    )
    with_sequence = cmfgen_sup_to_full_v3b(
        **common, int_seq=np.array([0, 7, 7]))
    without_sequence = cmfgen_sup_to_full_v3b(
        **common, int_seq=np.zeros(3, dtype=int))

    # For FL3, CMFGEN's forward search is empty; it must then retry backward
    # and interpolate from FL2, which belongs to a different superlevel.
    assert not np.isclose(with_sequence[0, 2], without_sequence[0, 2],
                          rtol=1e-10, atol=0.0)
    np.testing.assert_allclose(with_sequence.sum(axis=1), [110.0],
                               rtol=2e-15, atol=0.0)


def test_population_identity_metrics_reports_max_depth_and_level():
    metrics = population_identity_metrics(
        np.array([[1.0, 4.0], [2.0, 8.0]]),
        np.array([[1.1, 4.0], [2.0, 7.5]]))

    assert metrics["max_witness"] == {
        "depth_1based": 1,
        "full_level_1based": 1,
        "reference_population": 1.0,
        "reconstructed_population": 1.1,
        "abs_relative": pytest.approx(0.1),
    }


def _block(first_depth, n_superlevels=2, extra_row=False,
           process="Photoionization"):
    depths = range(first_depth, first_depth + 10)
    lines = ["Depth index", " ".join(str(depth) for depth in depths), "",
             f"Sk2 {process} Rates"]
    lines.extend(" ".join([f"{sl * 1e-8:.3E}"] * 10)
                 for sl in range(1, n_superlevels + 1))
    if extra_row:
        lines.extend(["", " ".join(["9.0E-99"] * 10)])
    return lines


def test_parse_prrr_superlevel_rates_reads_all_depth_chunks(tmp_path):
    path = tmp_path / "Sk2PRRR"
    path.write_text("\n".join(_block(1) + [""] + _block(11)) + "\n")

    rates = parse_prrr_superlevel_rates(path, "Sk2", nd=20, nsl=2)

    assert rates.shape == (2, 20)
    np.testing.assert_allclose(rates[0], 1e-8)
    np.testing.assert_allclose(rates[1], 2e-8)


def test_parse_prrr_superlevel_recombination_rates_reads_all_depth_chunks(tmp_path):
    path = tmp_path / "Sk2PRRR"
    path.write_text("\n".join(
        _block(1, process="Recombination") + [""] +
        _block(11, process="Recombination")) + "\n")

    rates = parse_prrr_superlevel_rates(
        path, "Sk2", nd=20, nsl=2, process="recombination")

    assert rates.shape == (2, 20)
    np.testing.assert_allclose(rates[0], 1e-8)
    np.testing.assert_allclose(rates[1], 2e-8)


def test_parse_prrr_superlevel_rates_rejects_extra_superlevel_row(tmp_path):
    path = tmp_path / "Sk2PRRR"
    path.write_text("\n".join(_block(1, extra_row=True)) + "\n")

    with pytest.raises(ValueError, match="more than MODEL_SPEC N_SL"):
        parse_prrr_superlevel_rates(path, "Sk2", nd=10, nsl=2)


def test_parse_prrr_superlevel_rates_rejects_incomplete_depth_chunk(tmp_path):
    path = tmp_path / "Sk2PRRR"
    lines = ["Depth index", "1 2 3 4 5 6 7 8 9", "",
             "Sk2 Photoionization Rates", "1 1 1 1 1 1 1 1 1",
             "2 2 2 2 2 2 2 2 2"]
    path.write_text("\n".join(lines) + "\n")

    with pytest.raises(ValueError, match="depth block"):
        parse_prrr_superlevel_rates(path, "Sk2", nd=10, nsl=2)


def _temperature_chunk(first_depth, value):
    depths = range(first_depth, first_depth + 10)
    return ["Depth index", " ".join(str(depth) for depth in depths), "",
            "Temperature [1.0E+4K]",
            " ".join([f"{value:.4E}"] * 10)]


def test_parse_prrr_temperature_reads_all_depth_chunks_and_converts_kelvin(tmp_path):
    path = tmp_path / "Sk2PRRR"
    path.write_text("\n".join(
        _temperature_chunk(1, 1.5) + [""] +
        _temperature_chunk(11, 2.0)) + "\n")

    temperature = parse_prrr_temperature(path, nd=20)

    np.testing.assert_allclose(temperature[:10], 15000.0)
    np.testing.assert_allclose(temperature[10:], 20000.0)


def test_parse_prrr_temperature_rejects_missing_depth_profile(tmp_path):
    path = tmp_path / "Sk2PRRR"
    path.write_text("\n".join(_temperature_chunk(1, 1.5)) + "\n")

    with pytest.raises(ValueError, match="depth chunks"):
        parse_prrr_temperature(path, nd=20)


def test_parse_prrr_temperature_rejects_missing_temperature_block(tmp_path):
    path = tmp_path / "Sk2PRRR"
    lines = ["Depth index", "1 2 3 4 5 6 7 8 9 10", "", "Electron Density",
             " ".join(["1.0E+05"] * 10)]
    path.write_text("\n".join(lines) + "\n")

    with pytest.raises(ValueError, match="temperature missing"):
        parse_prrr_temperature(path, nd=10)


def _channel_binary(path, *, version=4, trailing=b""):
    n_levels, n_bins, n_routes = 4, 2, 2
    data = bytearray(struct.pack("<IIiiidd", 0x434D4643, version,
                                 n_levels, n_bins, n_routes, 1.0e14, 2.0e14))
    data.extend(struct.pack("<5i", 0, 1, 2, 2, 2))
    data.extend(struct.pack("<2i", 2, 3))
    data.extend(struct.pack("<2d", 1.0, 1.0))
    data.extend(struct.pack("<2i", 0, 0))
    data.extend(struct.pack("<2d", 0.0, 0.0))
    data.extend(struct.pack("<2d", 4.0, 6.0))
    data.extend(struct.pack("<2d", 1.0e-18, 1.0e-18))
    data.extend(np.asarray([[2e-18, 1e-18], [1e-18, 3e-18]],
                           dtype="<f8").tobytes())
    data.extend(trailing)
    path.write_bytes(data)


def test_parse_cmfgen_bf_channels_reads_sealed_v4_arrays(tmp_path):
    path = tmp_path / "cmfgen_bf_channels.bin"
    _channel_binary(path)

    channels = parse_cmfgen_bf_channels(
        path, expected_levels=4, expected_bins=2,
        manifest={"n_channels": 2, "n_levels": 4, "n_freq_bins": 2})

    np.testing.assert_array_equal(channels["offsets"], [0, 1, 2, 2, 2])
    np.testing.assert_array_equal(channels["targets"], [2, 3])
    np.testing.assert_allclose(channels["sigma"], [[2e-18, 1e-18],
                                                   [1e-18, 3e-18]])


@pytest.mark.parametrize("version,trailing", [(3, b""), (4, b"x")])
def test_parse_cmfgen_bf_channels_rejects_wrong_version_or_trailing_bytes(
        tmp_path, version, trailing):
    path = tmp_path / "cmfgen_bf_channels.bin"
    _channel_binary(path, version=version, trailing=trailing)

    with pytest.raises(ValueError):
        parse_cmfgen_bf_channels(
            path, expected_levels=4, expected_bins=2,
            manifest={"n_channels": 2, "n_levels": 4, "n_freq_bins": 2})


def test_rate_metrics_reports_zero_mismatch_separately():
    metrics = rate_metrics(np.array([0.0, 1.0, 2.0]),
                           np.array([0.0, 1.1, 0.0]))

    assert metrics["n_both_zero"] == 1
    assert metrics["n_asymmetric_zero"] == 1
    assert metrics["n_compared_positive"] == 1
    assert metrics["median_abs_delta_dex"] == pytest.approx(np.log10(1.1))


def test_prrr_dxzv_is_next_stage_superlevel_one_not_single_full_level():
    populations = {
        "Sk2": (np.array([[100.0, 2.0], [200.0, 3.0]]), np.array([7.0, 8.0])),
        "SkIII": (np.array([[2.0, 3.0, 50.0], [4.0, 5.0, 60.0]]),
                  np.array([9.0, 10.0])),
    }
    grouped = sum_full_level_populations_by_superlevel(
        populations["SkIII"][0], np.array([1, 1, 2]), n_superlevels=2)

    reference = prrr_next_stage_sl1_population(
        populations, "Sk2", "SkIII", grouped)

    np.testing.assert_array_equal(reference, [5.0, 9.0])
    assert not np.array_equal(reference, populations["Sk2"][0].sum(axis=1))


def test_superlevel_population_sum_rejects_incomplete_or_invalid_mapping():
    with pytest.raises(ValueError, match="invalid full-level population/superlevel map"):
        sum_full_level_populations_by_superlevel(
            np.array([[1.0, 2.0]]), np.array([1, 3]), n_superlevels=2)


def test_top_ion_prrr_dxzv_uses_terminal_reservoir():
    populations = {
        "SkV": (np.array([[11.0, 12.0], [13.0, 14.0]]), np.array([0.5, 0.25])),
    }

    np.testing.assert_array_equal(
        prrr_next_stage_sl1_population(populations, "SkV", None), [0.5, 0.25])


def test_element_population_closure_includes_terminal_reservoir_once():
    result = element_population_closure(
        np.array([14.0, 20.0]),
        [np.array([[4.0, 3.0], [5.0, 8.0]]), np.array([[5.0], [6.0]])],
        np.array([2.0, 1.0]))

    assert result["pass"]
    assert result["max_abs_relative_residual"] == pytest.approx(0.0)


def test_total_photoionization_rate_uses_current_ion_density():
    rate_density = np.array([[3.0, 5.0], [1.0, 3.0]])
    current_ion_density = np.array([2.0, 4.0])

    np.testing.assert_allclose(
        total_photoionization_rate(rate_density, current_ion_density), [2.0, 2.0])


def test_total_photoionization_rate_rejects_nonpositive_current_ion_density():
    with pytest.raises(ValueError, match="current-ion density"):
        total_photoionization_rate(np.ones((2, 1)), np.array([0.0]))


def test_osc_e_ev_is_used_as_binding_threshold_not_subtracted_again():
    levels = np.array([
        (0.0, 16.0),
        (5.0 / CM_TO_EV, 11.0),
        (17.0 / CM_TO_EV, -1.0),
    ], dtype=[("E_cm", "f8"), ("E_eV", "f8")])

    excitation, threshold_hz = level_energies_and_thresholds(
        levels, ionization_eV=16.0, n_full=3, ion_label="Si II")

    np.testing.assert_allclose(excitation, [0.0, 5.0, 17.0])
    np.testing.assert_allclose(threshold_hz[:2],
                               np.array([16.0, 11.0]) * 1.602176634e-12 /
                               6.62607015e-27)
    assert threshold_hz[2] == 0.0


def test_osc_energy_threshold_identity_is_fail_closed():
    levels = np.array([(0.0, 12.0)], dtype=[("E_cm", "f8"), ("E_eV", "f8")])

    with pytest.raises(RuntimeError, match="threshold inconsistent"):
        level_energies_and_thresholds(levels, ionization_eV=16.0,
                                      n_full=1, ion_label="Si II")


@pytest.mark.parametrize("reference,lumina", [
    (np.array([-1.0]), np.array([1.0])),
    (np.array([1.0]), np.array([np.nan])),
])
def test_rate_metrics_rejects_invalid_rates(reference, lumina):
    with pytest.raises(ValueError):
        rate_metrics(reference, lumina)


def test_milne_rate_density_uses_group_population_and_rejects_missing_field():
    sigma = np.array([2.0e-18, 1.0e-18])
    field = np.array([[0.0, 3.0e-4], [0.0, 4.0e-4]])
    state = np.full((2, 2), 1, dtype=np.int32)
    temperature = np.array([1.2e4, 1.5e4])
    ne = np.array([2.0e9, 3.0e9])
    upper = np.array([7.0e5, 9.0e5])
    kwargs = dict(
        sigma_row=sigma, j_nu=field, j_state=state,
        nu_min=1.0e14, nu_max=2.0e14, threshold_eV=0.75,
        temperature=temperature, electron_density=ne,
        g_lower=2.0, g_upper=8.0, upper_population=upper,
        probability=0.25)

    rate = milne_recombination_rate_density(**kwargs)
    doubled = milne_recombination_rate_density(
        **{**kwargs, "upper_population": 2.0 * upper})

    assert np.isfinite(rate).all() and np.all(rate > 0.0)
    np.testing.assert_allclose(doubled, 2.0 * rate, rtol=2e-15)
    bad_state = state.copy()
    bad_state[0, 1] = 3  # OUT_OF_GRID in the L1BF state vocabulary
    with pytest.raises(ValueError, match="do not substitute"):
        milne_recombination_rate_density(**{**kwargs, "j_state": bad_state})


def test_v4_recombination_aggregates_two_upper_fine_levels_into_one_sl():
    channels = {
        "nu_min": 1.0e14, "nu_max": 2.0e14,
        "offsets": np.array([0, 1, 2, 2, 2]),
        "targets": np.array([2, 3]),
        "probability": np.array([1.0, 1.0]),
        "phot_grid": np.array([0, 0]),
        "target_energy_eV": np.array([0.0, 0.12]),
        "target_g": np.array([6.0, 6.0]),
        "sigma": np.array([[2.0e-18, 1.0e-18],
                            [1.0e-18, 3.0e-18]]),
    }
    level_index = {
        (14, 2, 1): 0, (14, 2, 2): 1,
        (14, 3, 1): 2, (14, 3, 2): 3,
    }
    field = np.array([[1.0e-4, 2.0e-4]])
    field_state = np.ones_like(field, dtype=np.int32)
    temp, ne = np.array([1.4e4]), np.array([1.0e9])
    grouped_upper = np.array([[5.0e6]])

    rates = lumina_recombination_rate_density(
        channels, 14, 2, 2, 1, np.array([1, 1]), np.array([0.0, 0.1]),
        np.array([2.0, 4.0]), 0.8, 3, np.array([1, 1]),
        np.array([0.0, 0.12]), np.array([2.0, 4.0]),
        grouped_upper, level_index, field, field_state, temp, ne)

    thermal_energy = 8.617333262145e-5 * temp
    partition = 2.0 + 4.0 * np.exp(-0.12 / thermal_energy)
    route1 = milne_recombination_rate_density(
        channels["sigma"][0], field, field_state, 1.0e14, 2.0e14,
        0.8, temp, ne, 2.0, partition, grouped_upper[:, 0], 1.0)
    route2 = milne_recombination_rate_density(
        channels["sigma"][1], field, field_state, 1.0e14, 2.0e14,
        0.82, temp, ne, 4.0, partition * np.exp(0.12 / thermal_energy),
        grouped_upper[:, 0], 1.0)

    np.testing.assert_allclose(rates[0], route1 + route2)
