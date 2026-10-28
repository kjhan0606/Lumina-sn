"""새 후보의 실제 C 인구 전달 계약. 계수 생산이나 수송은 실행하지 않는다."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_lumina_accepted_sources as S


def test_missing_current_material_cannot_silently_select_the_old_candidate():
    with pytest.raises(ValueError, match='과거 후보 기본값'):
        S.material_inputs()


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


@pytest.mark.parametrize('thermal,expected',[
    (None,'ALL_POSITIVE_IONS'),('physical_stored','ALL_POSITIVE_IONS'),
    ('physical_instantaneous_scenario','ALL_POSITIVE_IONS'),
    ('native_stored','NATIVE_UPPER_IONS'),('native_instantaneous_scenario','NATIVE_UPPER_IONS')])
def test_FF_source_selection_follows_declared_material_energy_policy(thermal,expected):
    manifest={} if thermal is None else dict(thermal_energy_policy=thermal)
    assert S.coefficient_free_free_policy(manifest)==expected
    assert S.coefficient_free_free_policy(manifest,expected)==expected


@pytest.mark.parametrize('thermal,requested',[
    ('native_instantaneous_scenario','ALL_POSITIVE_IONS'),
    ('physical_stored','NATIVE_UPPER_IONS'),('unknown',None),(None,'unknown')])
def test_mixed_heat_and_transport_FF_models_are_rejected(thermal,requested):
    with pytest.raises(ValueError):S.coefficient_free_free_policy(dict(thermal_energy_policy=thermal),requested)
