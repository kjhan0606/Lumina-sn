"""새 후보의 실제 C 인구 전달 계약. 계수 생산이나 수송은 실행하지 않는다."""
from pathlib import Path
import json
import sys
import os
import subprocess
from types import SimpleNamespace

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


def test_new_source_run_builds_current_occupation_kernel_without_replacing_existing(tmp_path):
    native=S.RUNS.parent/'cmfgen_src/cur_cmf'
    if not (native/'subs/gff.f').is_file():pytest.skip('native GFF 소스 필요')
    project=Path(__file__).resolve().parents[1];out=tmp_path/'kernels'
    library,record=S.build_current_source_kernel(project,native,out)
    assert S.S.SourceKernels(library).occupation_line_function is not None
    assert record['library_sha256']==S.B.sha(library)
    for name in ('model_source_kernels.c','model_gaussian_profile.h'):
        assert record['source_sha256'][name]==S.B.sha(project/'src'/name)
    assert record['source_sha256']['gff.f']==S.B.sha(native/'subs/gff.f')
    assert record['BB_occupation_contract']=='ABSORPTION_TIMES_W_UPPER_OVER_W_LOWER_V1'
    before=library.read_bytes()
    with pytest.raises(FileExistsError):S.build_current_source_kernel(project,native,out)
    assert library.read_bytes()==before


def test_builder_result_without_occupation_abi_cannot_reach_source_production(tmp_path,monkeypatch):
    monkeypatch.setattr(S.subprocess,'run',lambda *args,**kwargs:None)
    monkeypatch.setattr(S.S,'SourceKernels',lambda path:SimpleNamespace(occupation_line_function=None))
    with pytest.raises(ValueError,match='occupation 지원'):
        S.build_current_source_kernel(tmp_path,tmp_path/'native',tmp_path/'kernels')


@pytest.mark.parametrize('callback',['none','valid','bad_digest','missing_digest','source_failed','preflight_failed'])
def test_source_completion_callback_is_explicit_and_never_follows_a_failed_source(tmp_path,callback):
    control=tmp_path/'control';control.mkdir();config=control/'config.json';config.write_text('{}')
    (control/'lumina_bundle_dispatch.py').write_text('# 합성 제출기; 실제 SSH/Slurm 호출 없음\n')
    binaries=tmp_path/'bin';binaries.mkdir();parent=tmp_path/'parent';parent.mkdir()
    git=binaries/'git';git.write_text('#!/bin/bash\ntar -cf - -T /dev/null\n');git.chmod(0o700)
    python=binaries/'python3';python.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$*" >> "$FIXTURE_TRACE"\n'
        'if [[ "${@: -1}" = preflight-source ]]; then exit "$FIXTURE_PREFLIGHT_STATUS"; fi\n'
        'if [[ "$1" = scripts/run_lumina_accepted_sources.py ]]; then exit "$FIXTURE_SOURCE_STATUS"; fi\n')
    python.chmod(0o700)
    startup=tmp_path/'shell-env';startup.write_text('module() { :; }\n')
    trace=tmp_path/'trace'
    env={**os.environ,'PATH':str(binaries)+os.pathsep+os.environ['PATH'],'BASH_ENV':str(startup),
         'FIXTURE_TRACE':str(trace),'FIXTURE_SOURCE_STATUS':'7' if callback=='source_failed' else '0'}
    env['FIXTURE_PREFLIGHT_STATUS']='8' if callback=='preflight_failed' else '0'
    env.pop('LUMINA_ACCEPTED_BUNDLE_CONFIG',None);env.pop('LUMINA_ACCEPTED_BUNDLE_CONFIG_SHA256',None)
    if callback!='none':
        env['LUMINA_ACCEPTED_BUNDLE_CONFIG']=str(config)
        if callback!='missing_digest':
            env['LUMINA_ACCEPTED_BUNDLE_CONFIG_SHA256']='0'*64 if callback=='bad_digest' else S.B.sha(config)
    script=Path(S.__file__).with_name('submit_lumina_accepted_sources.slurm')
    result=subprocess.run(['bash',str(script),str(parent),'a'*40],env=env,capture_output=True,text=True,timeout=10)
    lines=trace.read_text().splitlines() if trace.exists() else []
    if callback in ('bad_digest','missing_digest'):
        assert result.returncode!=0 and not lines and not (parent/'source').exists()
    elif callback=='source_failed':
        assert result.returncode==7 and len(lines)==2
        assert lines[0].endswith(' preflight-source')
    elif callback=='preflight_failed':
        assert result.returncode==8 and len(lines)==1 and not (parent/'source').exists()
    else:
        assert result.returncode==0 and len(lines)==(3 if callback=='valid' else 1)
        if callback=='valid':assert lines[-1].endswith(' '+S.B.sha(config)+' field')


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
