"""원시 관찰 reader와 충분조건을 확인하며 큰 상한을 실패로 바꾸지 않는다."""
from pathlib import Path
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_cmfgen_native_se_observation as obs


def sample():
    x = dict(pops=np.array([[2., 3.], [4., 5.], [6., 7.], [1., 1.]]),
        raw_count=np.array([1, 1], dtype='<i4'))
    for key in ('raw', 'lo', 'hi', 'inc', 'out'):
        x[key] = np.zeros((2, 2))
    for key in ('raw_valid', 'coll_valid', 'coll_count'):
        x[key] = np.ones((2, 2), dtype='<i4')
    x['inc'][:] = 1
    for key in ('nc', 'nc_lo', 'nc_hi', 'nc_target'):
        x[key] = np.zeros((1, 2))
    x['nc_valid'] = np.ones((1, 2), dtype='<i4')
    x['charge_raw'] = np.zeros(2)
    return x


def save_witness(path, fields):
    content = b'NATIVESEBOUNDV1 '  # Fortran 문자열은 16 bytes다.
    assert len(content) == 16
    content += struct.pack('<9i', 1, 64, 2, 2, 4, 2, 68, 67, 1)
    for key in ('pops', 'raw', 'lo', 'hi', 'inc', 'out', 'raw_count', 'raw_valid',
                'coll_valid', 'coll_count', 'nc', 'nc_lo', 'nc_hi', 'nc_target', 'nc_valid', 'charge_raw'):
        content += fields[key].astype('<i4' if key in ('raw_count', 'raw_valid', 'coll_valid', 'coll_count', 'nc_valid') else '<f8').tobytes(order='F')
    path.write_bytes(content)


def test_reader_preserves_all_rows_and_phase(tmp_path):
    path = tmp_path/'witness.bin'
    expected = sample()
    save_witness(path, expected)
    header, actual = obs.read_witness(path)
    assert header['parent'] == 67 and header['counter'] == 68
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
    path.write_bytes(path.read_bytes()+b'!')
    with pytest.raises(ValueError, match='후행'):
        obs.read_witness(path)


def test_missing_and_large_bounds_are_unproven_not_failed():
    x = sample()
    x['lo'][0] = [0, -1e-7]
    x['hi'][0] = [0, 1e-7]
    x['inc'][1] = [0, 1]
    x['coll_valid'][1, 1] = 0
    upper, proven, reasons = obs.sufficient_bounds(x)
    np.testing.assert_array_equal(proven, [[True, False], [False, False]])
    assert upper[0, 0] == 0
    assert reasons[0, 1] == 'SUFFICIENT_UPPER_NOT_SMALL_ENOUGH'
    assert reasons[1, 0] == 'ZERO_COLLISION_LOWER_BOUND'
    assert reasons[1, 1] == 'INVALID_OR_MISSING_COLLISION_OR_RAW'
    assert all('FAIL' not in reason for reason in reasons.ravel())


def test_upper_boundary_is_not_rounded_into_pass():
    x = sample()
    x['hi'][:] = 1e-8
    upper, proven, _ = obs.sufficient_bounds(x)
    assert np.all(upper > 1e-8) and not proven.any()


@pytest.mark.parametrize('changed', ['storage', 'population', 'correction', 'scale', 'control'])
def test_noninterference_requires_original_bytes_and_scale(tmp_path, monkeypatch, changed):
    actual, reference = tmp_path/'actual', tmp_path/'reference'
    for directory in (actual, reference):
        directory.mkdir()
        (directory/'VADAT').write_text('original')
        (directory/'IN_ITS').write_text('original')
        (directory/'OUTGEN').write_text('The local minimum value of scale is: 0.98936\n')
    monkeypatch.setattr(obs, 'read_dimensions', lambda _: (2, 4))
    def records(path, *_):
        modified = path.parent == actual
        return (b'header', b'other' if modified and changed == 'storage' else b'original',
                np.ones((4, 2))*(2 if modified and changed == 'population' else 1))
    monkeypatch.setattr(obs, 'record_bytes', records)
    monkeypatch.setattr(obs, 'parse_solutions', lambda path, *_:
        [np.ones((4, 2))*(2 if path.parent == actual and changed == 'correction' else 1)])
    monkeypatch.setattr(obs, 'point_records', lambda _: dict(POINT1=67, POINT2=67))
    if changed == 'scale':
        (actual/'OUTGEN').write_text('The local minimum value of scale is: 1.00000\n')
    if changed == 'control':
        (actual/'VADAT').write_text('different')
    result = obs.reproduction(actual, reference)
    assert not result['reproduced'] and not result['tolerance_enlarged']


def test_phase_mismatch_does_not_create_assessment(tmp_path):
    fields = sample()
    save_witness(tmp_path/'NATIVE_SE_BOUND_000068.bin', fields)
    with pytest.raises(ValueError, match='갱신 전'):
        obs.assess(tmp_path, fields['pops']*2, 67)
    assert not (tmp_path/'seq_logs/native-se-assessment.json').exists()


def test_source_keeps_conditional_execution_and_byproduct_contract():
    # 실행 순서는 성공을 꾸미지 않는 명시적 조건으로 묶여 있어야 한다.
    source = Path(obs.__file__).read_text()
    assert source.index("if not report['reproduction_comparison']['reproduced']") < source.index("report[name+'_assessment'] = assess")
    assert "비간섭 재현 실패; 두 번째 native 실행 금지" in source
    assert "byproduct_record68_adopted_as_outer_checkpoint=False" in source
