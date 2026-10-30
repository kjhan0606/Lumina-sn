"""새 후보의 실제 C 인구 전달 계약. 계수 생산이나 수송은 실행하지 않는다."""
from pathlib import Path
import json
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


def test_invalid_TNE_manifest_stops_before_frequency_coefficient_integration(tmp_path,monkeypatch):
    accepted=tmp_path/'material/accepted-material.json';accepted.parent.mkdir()
    accepted.write_text(json.dumps(dict(schema='lumina-accepted-nodal-material-v1',
        source_kind='FINITE_NEW_FIELD_FIXED_T_CHARGE_RESPONSE')))
    full=tmp_path/'C-full.bin';full.write_bytes(b'fixture')
    monkeypatch.setenv('SLURM_CPUS_PER_TASK','16')
    monkeypatch.setenv('SLURM_MEM_PER_NODE','32768')
    monkeypatch.setenv('OMP_NUM_THREADS','16')
    monkeypatch.setattr(S,'RUNS',tmp_path/'runs')
    expected={
        'grid':'c03b9fb565a215fd11ec70ab2574c70af7f09eab74669214638177c2bbfc0a6d',
        'geometry_preparation':'1fad50444de5db7a583befd8d59ae7970c5ebf63c3cd423273c341ddb21582f2',
        'cpu_library':'6f82306bff6170037a40f09303f33e1cb00d930ddbdeb0cbe38d7b8b61855be4',
        'boundary_library':'f88362d72782415cadd2066ff6a14d512dce5d62ec61db4c9f620c800a7fd396'}
    relative={
        'grid':'lumina_full_sr_verified_20261020-yb8k9y/grid/frequency.npy',
        'geometry_preparation':'lumina_outer_reuse_20261021-uTVbnT/boundary/preparation.json',
        'cpu_library':'lumina_extended_4h_20261022-rJGMVR/driver.so',
        'boundary_library':'lumina_outer_weighted_20261021-B0Fepu/pytest-threadcap-work/outer-c0/outer.so'}
    pinned={str(S.RUNS/relative[name]):digest for name,digest in expected.items()}
    pinned[str(accepted)]='a'*64;pinned[str(full)]='b'*64
    monkeypatch.setattr(S.B,'sha',lambda path:pinned[str(Path(path).resolve())])
    monkeypatch.setattr(S,'material_inputs',lambda *args:(accepted,'a'*64,full,'b'*64))
    monkeypatch.setattr(S.S,'prepare',lambda *args,**kwargs:pytest.fail('coefficient integration must not start'))
    with pytest.raises(ValueError,match='TNE 후보만 허용'):
        S.run(tmp_path/'coefficients','c'*40,accepted,'a'*64,full,'b'*64)
    assert not (tmp_path/'coefficients').exists()
