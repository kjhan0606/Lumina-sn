"""공개 스펙트럼 비교에서 단위 변환과 형상 정규화를 구분한다."""
import pytest
import numpy as np
import plot_spectrum_vs_cmfgen as overlay


def test_formal_per_cm_to_per_angstrom(tmp_path, monkeypatch):
    path = tmp_path / "formal.csv"
    path.write_text("wavelength_angstrom,flux\n3000,2e45\n4000,4e45\n")
    monkeypatch.setattr(overlay, "LUM", path)
    wave, luminosity = overlay.read_lumina()
    assert wave == [3000.0, 4000.0]
    assert luminosity == pytest.approx([2e37, 4e37])
    # Å 기준 적분은 원래 cm 기준 적분과 같아야 한다.
    assert overlay.integral(wave, luminosity, 3000, 4000) == pytest.approx(3e40)


def test_public_reference_is_already_per_angstrom(tmp_path, monkeypatch):
    path = tmp_path / "cmfgen.txt"
    path.write_text("#NTIMES: 2\n#NWAVE: 2\n#TIMES[d]: 10 19.48\n"
                    "#wavelength[Ang] flux[erg/s/Ang]\n3000 9e36 2e37\n4000 9e36 4e37\n")
    monkeypatch.setattr(overlay, "CMF", path)
    wave, luminosity = overlay.read_cmfgen()
    assert wave == [3000.0, 4000.0]
    assert luminosity == [2e37, 4e37]


def test_unit_conversion_does_not_change_normalized_shape():
    wave = [3000.0, 4000.0, 5000.0]
    raw = [1e45, 4e45, 3e45]
    converted = [value * overlay.CM_PER_ANGSTROM for value in raw]
    target = 8e40
    before = target / overlay.integral(wave, raw, 3000, 5000)
    after = target / overlay.integral(wave, converted, 3000, 5000)
    assert after == pytest.approx(before * 1e8)
    assert [value * after for value in converted] == pytest.approx(
        [value * before for value in raw])


def test_observer_npz_is_already_per_angstrom_and_reverses_wave_order(tmp_path):
    path = tmp_path / 'observer-spectrum.npz'
    np.savez(path, wavelength_A=[5000., 4000., 3000.],
             luminosity_erg_s_A=[3.e37, 2.e37, 1.e37])
    wave, luminosity = overlay.read_observer(path)
    assert wave == [3000., 4000., 5000.]
    assert luminosity == [1.e37, 2.e37, 3.e37]


@pytest.mark.parametrize('wave,luminosity', [
    ([3000., 3000.], [1., 2.]),
    ([3000., float('nan')], [1., 2.]),
    ([3000., 4000.], [1., -2.]),
    ([3000., 4000.], [1.]),
])
def test_observer_rejects_invalid_or_duplicate_samples(tmp_path, wave, luminosity):
    path = tmp_path / 'observer-spectrum.npz'
    np.savez(path, wavelength_A=wave, luminosity_erg_s_A=luminosity)
    with pytest.raises(ValueError):
        overlay.read_observer(path)


def test_common_band_integrates_both_interpolated_edges_without_extrapolating():
    wave = [2000., 4000., 6000.]
    flux = [2., 4., 6.]
    assert overlay.integral_with_edges(wave, flux, 3000., 5000.) == pytest.approx(8000.)
    with pytest.raises(ValueError):
        overlay.integral_with_edges(wave, flux, 1000., 5000.)


def test_observer_npz_flows_through_plot_without_legacy_unit_conversion(tmp_path, monkeypatch):
    path = tmp_path / 'observer-spectrum.npz'
    np.savez(path, wavelength_A=[20000., 10000., 6000., 4000., 3000., 2000.],
             luminosity_erg_s_A=[2.e37, 4.e37, 3.e37, 2.e37, 1.e37, 1.e37])
    monkeypatch.setattr(overlay, 'read_cmfgen', lambda: (
        [2500., 3000., 4000., 6000., 10000., 20000.],
        [1.e37, 1.e37, 2.e37, 3.e37, 4.e37, 2.e37]))
    monkeypatch.setenv('MPLCONFIGDIR', str(tmp_path / 'matplotlib'))
    output = tmp_path / 'observer-comparison.png'
    overlay.main(output=output, observer=path)
    assert output.stat().st_size > 0
