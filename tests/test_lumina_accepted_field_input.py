"""합성 계수의 새 물질→실제 C 경계→상주 입력 연결. 실물 F/GPU 실행은 없다."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import lumina_accepted_field_input as A
import lumina_outer_atmosphere as O
from lumina_compiled_input import CompiledInput
from lumina_local_micro import ZeroTrial
from lumina_nodal_driver import CompiledDriverPlan
from lumina_resident_run import open_field_inputs
from lumina_material_trial_field import open_transport_trial
from lumina_fullsr_sc_actual import InputField
from test_lumina_compiled_input import packet
from test_lumina_resident_run import resident_record


@pytest.fixture(scope='module')
def field_library(tmp_path_factory):
    output = tmp_path_factory.mktemp('accepted-field')/'field.so'
    sources = ('cmf_outer_transport.c', 'cmf_exact_sliding.c', 'cmf_error_envelope.c',
        'cmf_relativistic_characteristic.c', 'cmf_frequency_history.c', 'cmf_nodal_input.c',
        'cmf_nodal_history.c', 'cmf_nodal_driver.c')
    subprocess.run([os.environ.get('CC', 'cc'), '-O2', '-std=c11', '-fopenmp', '-fPIC', '-shared',
        '-ffp-contract=off', *[str(ROOT/'src'/n) for n in sources], '-lm', '-o', str(output)],
        check=True, capture_output=True, timeout=60)
    return output


def fixture(root, library):
    g, original_source, parts = packet(root/'sources', physical=True)
    nu = original_source.frequency.copy(); radius = original_source.radius.copy(); original_source.close()
    ns = len(radius); te = 14100.+np.arange(ns); ne = 1.e8+np.arange(ns)
    source = root/'sources/sources.bin'
    with source.open('r+b') as f:
        f.seek(16+8*len(nu)); f.write(te.astype('<f8').tobytes()); f.write(ne.astype('<f8').tobytes())
    grid = root/'frequency.npy'; np.save(grid, nu, allow_pickle=False)
    material_dir = root/'accepted'; material_dir.mkdir()
    material = material_dir/'native_material_nodes.bin'; epoch = radius[0]/(.06*2.99792458e10)
    # 실제 준비자와 같은 반경/epoch 연산 순서. 곱셈으로 만든 속도와의
    # 1ULP 차이를 허용하려고 생산자의 기하 신원 조건을 완화하지 않는다.
    g = O.geometry(radius, radius/epoch, np.zeros(ns))
    physical = np.column_stack((radius, te, ne, np.full(ns, 1.e-14)))
    material.write_bytes(struct.pack('<8sIIId', b'LUMNOD01', ns, 1, 1, epoch)+
        struct.pack('<3i', 14, 1, 0)+physical.astype('<f8').tobytes()+np.ones((ns, 1), dtype='<f8').tobytes())
    full = material_dir/'C-imported-full.bin'
    full.write_bytes(struct.pack('<8sII', b'LMATFL01', 1, ns)+np.ones((1, ns), dtype='<f8').tobytes())
    old_seed = root/'original-seed.json'
    old_seed.write_text(json.dumps(dict(epoch_seconds=epoch, boundary_method='DIFFUSION',
        inner_radius_cm=float(radius[-1]), inner_luminosity_erg_s=1.e39)))
    model = root/'MODEL'; model.write_text('synthetic model')
    accepted = material_dir/'accepted-material.json'
    accepted.write_text(json.dumps(dict(schema='lumina-accepted-nodal-material-v1', points=ns,
        full_levels=1, accepted_response_sha256='c'*64,
        output_sha256={'native_material_nodes.bin': A.sha(material)},
        input_sha256={str(p): A.sha(p) for p in (old_seed, model)})))
    report = source.parent/'sources.json'
    report.write_text(json.dumps(dict(schema='lumina-model-local-sources-v1', cells=ns,
        frequencies=len(nu), full_levels=1,
        C_full_population_comparison=dict(max_relative_error=0.),
        input_sha256={str(p): A.sha(p) for p in (material, full, grid, model)},
        output_sha256={'sources.bin': A.sha(source)})))
    previous = root/'old-geometry'; previous.mkdir()
    np.savez(previous/'geometry.npz', radius=g.radius, velocity=g.velocity, sigma=g.sigma,
        native_radius=g.native_radius, reference_indices=g.reference_indices)
    geometry = previous/'preparation.json'
    geometry.write_text(json.dumps(dict(epoch_seconds=epoch,
        output_sha256={'geometry.npz': A.sha(previous/'geometry.npz')})))
    boundary_library = root/'boundary.so'; boundary_library.write_bytes(library.read_bytes())
    paths = dict(accepted_material=accepted, sources=report, full_population=full, grid=grid,
        geometry_preparation=geometry, cpu_library=library, boundary_library=boundary_library,
        original_seed=old_seed, model=model)
    config = root/'config.json'
    config.write_text(json.dumps(dict(schema='lumina-accepted-field-config-v1',
        **{k: dict(path=str(p), sha256=A.sha(p)) for k, p in paths.items()})))
    return config, paths, parts, te, ne


def repin(config, key, path):
    data = json.loads(config.read_text()); data[key] = dict(path=str(path), sha256=A.sha(path))
    config.write_text(json.dumps(data))


@pytest.mark.parametrize('policy,ff',[('native_instantaneous_scenario','NATIVE_UPPER_IONS'),
    ('physical_instantaneous_scenario','ALL_POSITIVE_IONS')])
def test_declared_heat_policy_and_actual_coefficient_policy_reach_boundary(field_library,tmp_path,policy,ff):
    config,paths,*_=fixture(tmp_path,field_library)
    m=json.loads(paths['accepted_material'].read_text());m['thermal_energy_policy']=policy
    paths['accepted_material'].write_text(json.dumps(m));repin(config,'accepted_material',paths['accepted_material'])
    s=json.loads(paths['sources'].read_text());s['free_free']=dict(policy=ff)
    paths['sources'].write_text(json.dumps(s));repin(config,'sources',paths['sources'])
    result=A.prepare(config,A.sha(config),tmp_path/'ready')
    assert result['thermal_energy_policy']==policy and result['source_free_free_policy']==ff
    assert result['actual_F_calls']==0


@pytest.mark.parametrize('ff',[None,'ALL_POSITIVE_IONS'])
def test_native_material_cannot_use_undeclared_or_all_ion_coefficients(field_library,tmp_path,ff):
    config,paths,*_=fixture(tmp_path,field_library)
    m=json.loads(paths['accepted_material'].read_text());m['thermal_energy_policy']='native_instantaneous_scenario'
    paths['accepted_material'].write_text(json.dumps(m));repin(config,'accepted_material',paths['accepted_material'])
    if ff is not None:
        s=json.loads(paths['sources'].read_text());s['free_free']=dict(policy=ff)
        paths['sources'].write_text(json.dumps(s));repin(config,'sources',paths['sources'])
    with pytest.raises(ValueError,match='FF 선택 불일치'):A.prepare(config,A.sha(config),tmp_path/'ready')
    assert not (tmp_path/'ready').exists()


def test_new_state_actual_C_boundary_and_dense_input_are_connected(field_library, tmp_path):
    config, paths, parts, te, ne = fixture(tmp_path, field_library); out = tmp_path/'ready'
    report = A.prepare(config, A.sha(config), out); path = out/'preparation.json'
    assert report['boundary']['temperature_K'] == te[-1]
    assert report['boundary_library'] == str(paths['boundary_library']) != report['cpu_library']
    assert report['boundary']['luminosity_relative_error'] < 2.e-13
    assert report['geometry_reused_but_old_boundary_not_reused']
    with ExitStack() as stack:
        c, source, geometry, boundary, digest = open_field_inputs(stack, path, A.sha(path))
        assert c['cpu_library'] == str(field_library) and digest == A.sha(path)
        assert source.identity['role'] == 'ACCEPTED_NODAL_MATERIAL_NEW_COEFFICIENTS'
        assert source.identity['accepted_response_sha256'] == 'c'*64
        np.testing.assert_array_equal(source.read([0, 23], [0, 7])[0],
            ((parts[6]+(parts[0]+parts[2]))+parts[4])[np.ix_([0, 23], [0, 7])])
        plan = CompiledDriverPlan(geometry, source, panels=1)
        assert plan.layout()['native_nodes'] == 24 and plan.layout()['frequencies'] == 8
        with CompiledInput(field_library, geometry, source, ZeroTrial(source),
                slab_width=8, cache_rows=16, profiles=64) as provider:
            provider.bind_zero_trial(); stats = provider.prepare_dense(max_seconds=10.)
            assert stats['ready']
            raw = provider.raw(np.array([0, 23]), np.array([0, 7]))
            for actual, expected in zip(raw, source.read([0, 23], [0, 7])):
                np.testing.assert_array_equal(actual, expected)
        old_planck, _, _ = A.diffusion_boundary(field_library, source.frequency, 14000., 1.e39,
            float(source.radius[-1]), np.full(source.nf, 1.e-14))
        assert np.any(boundary[0] != old_planck)
    assert source.closed
    assert report['actual_F_calls'] == report['new_SE_solves'] == report['new_BF_BB_integrals'] == 0


def test_previous_J_uses_new_coefficients_in_actual_C(field_library, tmp_path):
    old_root = tmp_path/'old'; old_root.mkdir()
    old_config, old_paths, old_parts, *_ = fixture(old_root, field_library)
    binary = old_paths['sources'].parent/'sources.bin'
    with binary.open('r+b') as stream:
        stream.seek(16+8*(8+2*24)+4*24*8*8)
        stream.write((2*old_parts[4]).astype('<f8').tobytes())
    source_report = json.loads(old_paths['sources'].read_text())
    source_report['output_sha256']['sources.bin'] = A.sha(binary)
    old_paths['sources'].write_text(json.dumps(source_report)); repin(old_config, 'sources', old_paths['sources'])
    A.prepare(old_config, A.sha(old_config), old_root/'ready')
    old_prep = old_root/'ready/preparation.json'
    _, old_source, _, _ = A.open_prepared(old_prep, A.sha(old_prep))
    with old_source:
        old_identity = old_source.identity
        old_chi = old_source.read([5], [3])[0].copy()
    jpath = old_root/'J.npy'; np.save(jpath, np.full((24, 8), .125))
    old_record = dict(resident_record(), source_identity=old_identity,
                      J=dict(path=str(jpath), sha256=A.sha(jpath)))
    (old_root/'resident-field.json').write_text(json.dumps(old_record))
    config, _, _, _, _ = fixture(tmp_path, field_library)
    A.prepare(config, A.sha(config), tmp_path/'ready'); prep = tmp_path/'ready/preparation.json'
    with ExitStack() as stack:
        _, source, geometry, boundary, _ = open_field_inputs(stack, prep, A.sha(prep))
        with pytest.raises(ValueError, match='동일 물질'): InputField(source, jpath, A.sha(jpath))
        field = open_transport_trial(source, jpath, A.sha(jpath),
            initial_preparation=old_prep, initial_preparation_sha256=A.sha(old_prep))
        stack.callback(field.close)
        with CompiledInput(field_library, geometry, source, field,
                slab_width=8, cache_rows=16, profiles=64) as provider:
            provider.bind_view(); provider.prepare_dense(max_seconds=10.)
            chi, es, eta = source.read([5], [3])
            actual = provider.at([source.radius[5]], [3.])
            np.testing.assert_allclose(actual[0], chi, rtol=1.e-12, atol=0)
            np.testing.assert_allclose(actual[1], eta+es*.125, rtol=1.e-12, atol=0)
            assert np.any(actual[0] != old_chi)
            assert provider.identity['field']['accepted_field_for_target_material'] is False
            assert boundary[0].shape == (8,)


@pytest.mark.parametrize('bad', ['old_material', 'old_full', 'old_grid', 'old_model', 'bad_C_comparison'])
def test_legacy_coefficients_cannot_be_relabeled_as_new_state(field_library, tmp_path, bad):
    config, paths, *_ = fixture(tmp_path, field_library)
    report = json.loads(paths['sources'].read_text())
    if bad == 'bad_C_comparison':
        report['C_full_population_comparison']['max_relative_error'] = float('nan')
    else:
        key = {'old_material': str(paths['accepted_material'].parent/'native_material_nodes.bin'),
            'old_full': str(paths['full_population']), 'old_grid': str(paths['grid']),
            'old_model': str(paths['model'])}[bad]
        report['input_sha256'][key] = '0'*64
    paths['sources'].write_text(json.dumps(report)); repin(config, 'sources', paths['sources'])
    with pytest.raises(ValueError):
        A.prepare(config, A.sha(config), tmp_path/'ready')


def test_binary_T_ne_and_geometry_must_be_actual_new_material(field_library, tmp_path):
    config, paths, *_ = fixture(tmp_path, field_library)
    binary = paths['sources'].parent/'sources.bin'
    with binary.open('r+b') as f:
        f.seek(16+8*8); f.write(np.full(24, 14000., dtype='<f8').tobytes())
    report = json.loads(paths['sources'].read_text()); report['output_sha256']['sources.bin'] = A.sha(binary)
    paths['sources'].write_text(json.dumps(report)); repin(config, 'sources', paths['sources'])
    with pytest.raises(ValueError, match='T/ne'):
        A.prepare(config, A.sha(config), tmp_path/'ready')


def test_modified_new_boundary_or_manifest_cannot_resume(field_library, tmp_path):
    config, paths, *_ = fixture(tmp_path, field_library); out = tmp_path/'ready'
    A.prepare(config, A.sha(config), out); path = out/'preparation.json'; digest = A.sha(path)
    np.save(out/'inner_gradient.npy', np.zeros(8))
    with pytest.raises(ValueError, match='경계 변경'):
        A.open_prepared(path, digest)
    with pytest.raises(ValueError, match='기록 변경'):
        A.open_prepared(path, '0'*64)
    with pytest.raises(FileExistsError, match='보존'):
        A.prepare(Path('not_read'), 'not_read', out)
    with ExitStack() as stack, pytest.raises(ValueError, match='함께'):
        open_field_inputs(stack, path, None)


def test_legacy_resident_default_keeps_original_plan_and_identity(field_library, tmp_path, monkeypatch):
    import lumina_resident_run as R
    preparation = tmp_path/'preparation.json'
    for name in ('inner_planck.npy', 'inner_gradient.npy'):
        np.save(tmp_path/name, np.ones(8))
    preparation.write_text(json.dumps(dict(output_sha256={n: A.sha(tmp_path/n)
        for n in ('inner_planck.npy', 'inner_gradient.npy')})))
    class Source:
        ns, nf, closed, identity = 90, 1115020, False, dict(material_record=70)
        def __enter__(self): return self
        def __exit__(self, *args): self.closed = True
    source = Source(); geometry = SimpleNamespace(legacy=True)
    config = tmp_path/'base.json'
    config.write_text(json.dumps(dict(cpu_library=str(field_library),
        sealed_sha256={str(field_library): A.sha(field_library)}, config='old-config',
        preparation=str(preparation), preparation_sha256=A.sha(preparation), source_identity=source.identity)))
    monkeypatch.setattr(R, 'BASE', config); monkeypatch.setattr(R, 'BASE_SHA', A.sha(config))
    monkeypatch.setattr(R.M.N, 'open_sealed_native', lambda *args: source)
    monkeypatch.setattr(R.M, 'open_geometry', lambda *args: geometry)
    with ExitStack() as stack:
        c, actual, g, boundary, digest = R.open_field_inputs(stack)
        assert actual is source and g is geometry and digest == A.sha(config)
        assert c['source_identity'] == dict(material_record=70) and len(boundary) == 2
    assert source.closed


def test_gpu_staging_includes_new_optional_input_dependency(tmp_path):
    from stage_lumina_fullsr_sc import stage
    root = tmp_path/'stage'
    stage(ROOT, root, resident=True)
    record = json.loads((root/'stage.json').read_text())
    for name in ('scripts/lumina_accepted_field_input.py', 'scripts/lumina_resident_run.py',
                 'scripts/lumina_resident_feedback.py',
                 'scripts/lumina_outer_c_backend.py'):
        assert record['files'][name] == A.sha(ROOT/name) == A.sha(root/'code'/name)
