"""새59 상태의 외부장 짝비교와 기존 PRRR adapter의 계약을 구별한다."""
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import prepare_cmfgen_field_owner_pair as pair

SEED = Path('/gpfs/kjhan/cmfgen_runs/toy06_19p48d_stageA_fixedT_427702')


def test_actual_material_is_record59_not_historical_record39():
    models = pair.bf.atomic.load_cmfgen_model_spec(SEED / 'MODEL_SPEC', SEED / 'atomic_links.txt')
    columns, profiles, populations, epoch = pair.material_inputs(SEED, dict(model=models))
    nd, nt = pair.bf.read_dimensions(SEED)
    reader = pair.bf.ScrtempReader(SEED / 'SCRTEMP', nd, nt)
    variables = [v for v in pair.bf.read_variables(SEED, nt - 2) if v.species in ('Si', 'S')]
    state = reader.state(59)
    np.testing.assert_array_equal(populations, state[[v.index - 1 for v in variables]].T)
    np.testing.assert_array_equal(profiles[:, 1], state[nt - 1] * 1e4)
    np.testing.assert_array_equal(profiles[:, 2], state[nt - 2])
    assert not np.array_equal(populations, reader.state(39)[[v.index - 1 for v in variables]].T)
    assert populations.shape == (90, 518) and len(columns) == 518
    assert epoch == pytest.approx(19.48 * 86400, rel=1e-5)


def test_historical_state_is_refused_before_population_export(monkeypatch):
    monkeypatch.setattr(pair.bf, 'read_dimensions', lambda _: (90, 100))
    class OldReader:
        last_it = 39
        def __init__(self, *args):
            pass
    monkeypatch.setattr(pair.bf, 'ScrtempReader', OldReader)
    with pytest.raises(ValueError, match='59'):
        pair.material_inputs(Path('not-read'), {})


def test_pair_binary_changes_only_radiation_not_material(tmp_path):
    nu = np.array([1e10, 1e20])
    columns = [(14, 1, 0), (14, 5, 0)]
    profiles = np.array([[1e-14, 1e4, 1e7, 1e8, 2e8]])
    pop = np.array([[3., 7.]])
    for label, value in (('old', 1.), ('new', 2.)):
        pair.write_input(tmp_path / label, nu, profiles, columns, pop, np.full((2, 1), value), 1e6)
    a, b = [(tmp_path / label).read_bytes() for label in ('old', 'new')]
    _, _, nn, nc, _, _ = struct.unpack_from('<8sIIIId', a)
    material_end = struct.calcsize('<8sIIIId') + 8 * nn + 12 * nc + 40 + 8 * nc
    assert a[:material_end] == b[:material_end]
    assert a[material_end:] != b[material_end:]


def test_local_atomic_copy_requires_same_bytes_not_just_same_role(tmp_path):
    a, b = tmp_path / 'original', tmp_path / 'local_copy'
    a.write_bytes(b'same PHOT source'); b.write_bytes(a.read_bytes())
    expected = {(16, 3): dict(osc=a)}
    actual = {(16, 3): dict(osc=b)}
    assert pair.same_atomic_sources(expected, actual)[0]['material59_paths'] == [str(b)]
    b.write_bytes(b'different source')
    with pytest.raises(ValueError, match='byte'):
        pair.same_atomic_sources(expected, actual)
