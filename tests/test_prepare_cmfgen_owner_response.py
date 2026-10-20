import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_cmfgen_owner_response import write_input


def test_binary_profiles_preserved_and_no_overwrite(tmp_path):
    path = tmp_path / "owner.bin"
    nu = np.array([1e10, 1e20])
    profiles = np.array([[1e-14, 1e4, 1e7, 1e8, 2e8]])
    populations = np.array([[3.0, 7.0]])
    write_input(path, nu, profiles, [(14, 1, 0), (14, 5, 0)],
                populations, np.ones((2, 1)), 1e6)
    raw = path.read_bytes()
    magic, nd, nn, nc, nb, epoch = struct.unpack_from("<8sIIIId", raw)
    assert (magic, nd, nn, nc, epoch) == (b"CMFOWN01", 1, 2, 2, 1e6)
    offset = struct.calcsize("<8sIIIId") + 8 * nn + 12 * nc
    assert np.array_equal(np.frombuffer(raw, "<f8", count=5, offset=offset), profiles[0])
    assert np.array_equal(np.frombuffer(raw, "<f8", count=nc, offset=offset+40), populations[0])
    assert len(raw) == offset + 40 + 8*nc + 12*nb + 8*nn
    with pytest.raises(FileExistsError):
        write_input(path, nu, profiles, [(14, 1, 0), (14, 5, 0)],
                    populations, np.ones((2, 1)), 1e6)


@pytest.mark.parametrize("bad", ["negative_J", "negative_population", "nan_T", "bad_bounds"])
def test_invalid_input_rejected_before_write(tmp_path, bad):
    profiles = np.array([[1e-14, 1e4, 1e7, 1e8, 2e8]])
    pop = np.array([[1.0]])
    field = np.ones((2, 1))
    if bad == "negative_J": field[0, 0] = -1
    if bad == "negative_population": pop[0, 0] = -1
    if bad == "nan_T": profiles[0, 1] = np.nan
    if bad == "bad_bounds": profiles[0, 4] = profiles[0, 3]
    path = tmp_path / "bad.bin"
    with pytest.raises(ValueError):
        write_input(path, np.array([1e10, 1e20]), profiles, [(14, 1, 0)], pop, field, 1e6)
    assert not path.exists()
