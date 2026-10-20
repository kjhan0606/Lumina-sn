import os
from pathlib import Path

import numpy as np
import pytest

from scripts.cmfgen_parser import parse_phot
from scripts.cmfgen_phot_smoothing import PhotSmoother, read_smoothing_controls


def model_file(tmp_path, text="3000 [SIG_GAU_KMS]\n"):
    path = tmp_path / "MODEL"
    path.write_text(text)
    return path


def phot_file(tmp_path, smoothing="0.0D0"):
    path = tmp_path / "PHOT_A"
    header = (f"{smoothing} !Sigma used for Gaussian smoothing (km/s)\n"
              if smoothing is not None else "")
    path.write_text(
        header + "3 !Total number of data pairs\n"
        "term !Configuration name\n20 !Type of cross-section\n"
        "3 !Number of cross-section points\n1 1\n1.1 10\n2 1\n")
    return path


def test_reads_runtime_defaults_and_phot_input_width(tmp_path):
    assert read_smoothing_controls(model_file(tmp_path)) == {
        "sigma_kms": 3000.0, "fraction_sigma": .25,
        "cut_accuracy": .02, "above_edge": True}
    assert parse_phot(phot_file(tmp_path)).smoothing_sigma_kms == 0.0


@pytest.mark.parametrize("input_width", [None, "3000"])
def test_matches_rdphot_skip_rule_without_library(tmp_path, input_width):
    source = phot_file(tmp_path, input_width)
    smoother = PhotSmoother(model_file(tmp_path))
    raw = parse_phot(source)
    result = smoother.apply(source, raw)
    np.testing.assert_array_equal(result.entries[0].energy, raw.entries[0].energy)
    np.testing.assert_array_equal(result.entries[0].sigma_Mb, raw.entries[0].sigma_Mb)
    assert smoother.files[str(source.resolve())]["smoothed_entries"] == 0


def test_does_not_silently_bake_unsmoothed_data_when_model_requires_smoothing(tmp_path):
    source = phot_file(tmp_path)
    with pytest.raises(RuntimeError, match="MODEL requires PHOT smoothing"):
        PhotSmoother(model_file(tmp_path)).read(source)


def test_replay_refuses_changed_model(tmp_path):
    model = model_file(tmp_path)
    metadata = PhotSmoother(model).provenance()
    model.write_text("2000 [SIG_GAU_KMS]\n")
    with pytest.raises(ValueError, match="preprocessing input changed"):
        PhotSmoother.from_provenance(metadata)


def test_original_fortran_resolves_real_silicon_ii_5g_witness(tmp_path):
    library = os.environ.get("CMFGEN_PHOT_SMOOTH_LIB")
    source = Path("/gpfs/kjhan/cmfgen_21jun23/atomic/SIL/II/19apr23/phot_data_A")
    if not library or not source.is_file():
        pytest.skip("requires the scoped CMFGEN Fortran smoothing witness")
    raw = parse_phot(source)
    entry = next(e for e in raw.entries if e.config == "3s2_5g_2Ge")
    raw.entries = [entry]
    result = PhotSmoother(model_file(tmp_path), Path(library)).apply(source, raw)
    actual = result.entries[0]
    # Independent direct SM_PHOT_V3 run, fixed source/version, 2026-10-19.
    assert entry.n_points == 1813
    assert actual.n_points == 291
    assert actual.energy[0] == 1.0
    assert actual.sigma_Mb[0] == pytest.approx(4.367588887658677, rel=2e-13)
