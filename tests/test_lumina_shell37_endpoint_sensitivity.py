"""반사실험의 셀 계수 교체가 봉인 원본을 수정하지 않는지 확인한다."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from diagnose_lumina_shell37_endpoint_sensitivity import REFERENCE_NODES, SHELL, digest_row


def test_native_mapping_and_private_coefficient_views(tmp_path):
    assert SHELL == 37
    assert REFERENCE_NODES == (("outer", 51), ("inner", 52))
    original = np.arange(12., dtype='<f8').reshape(3, 4)
    coefficient = tmp_path / 'coefficient.npy'
    np.save(coefficient, original)
    before = coefficient.read_bytes()
    private = np.load(coefficient, mmap_mode='c', allow_pickle=False)
    private[1] = [-1., -2., -3., -4.]
    assert digest_row(private[1]) != digest_row(original[1])
    private._mmap.close()
    assert coefficient.read_bytes() == before

    raw = tmp_path / 'raw.bin'
    raw.write_bytes(b'0123456789ABCDEF' + original.tobytes())
    before = raw.read_bytes()
    private = np.memmap(raw, mode='c', dtype='<f8', offset=16, shape=(3, 4))
    private[1] = [10., 20., 30., 40.]
    private._mmap.close()
    assert raw.read_bytes() == before
