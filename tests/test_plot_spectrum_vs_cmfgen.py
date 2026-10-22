"""공개 스펙트럼 비교에서 단위 변환과 형상 정규화를 구분한다."""
import pytest
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
