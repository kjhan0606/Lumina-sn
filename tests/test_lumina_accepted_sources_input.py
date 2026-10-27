"""새 후보의 실제 C 인구 전달 계약. 계수 생산이나 수송은 실행하지 않는다."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_lumina_accepted_sources as S


def test_default_retains_existing_sealed_candidate():
    manifest, mh, full, fh = S.material_inputs()
    assert manifest.name == 'accepted-material.json' and full.name == 'C-imported-full.bin'
    assert mh == '61e3090a8550bffb6e6f030a84be811407634299eefba93036d1fad30d492c1b'
    assert fh == '2ce239bd5d39e994e2cca0a46b283d8ccb68fc17a60bbaabcc58c9291de22ea6'


def test_new_candidate_and_C_full_are_explicit_and_pinned(tmp_path):
    manifest = tmp_path/'new'/'accepted-material.json'; full = tmp_path/'C-new.bin'
    result = S.material_inputs(manifest, 'a'*64, full, 'b'*64)
    assert result == (manifest.resolve(), 'a'*64, full.resolve(), 'b'*64)


@pytest.mark.parametrize('values', [
    ('m', None, None, None), ('m', 'a'*64, 'f', None),
    ('m', 'invalid', 'f', 'b'*64), ('m', 'A'*64, 'f', 'b'*64),
])
def test_partial_or_invalid_new_identity_rejected(values):
    with pytest.raises(ValueError): S.material_inputs(*values)
