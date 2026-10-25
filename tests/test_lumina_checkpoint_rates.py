"""미수렴 장 소비자의 물리 항·영점·저장 신원을 소형 입력으로 확인한다."""
from pathlib import Path
import json
import csv
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import compare_lumina_checkpoint_rates as R


def fixture():
    nu = np.geomspace(1e15, 4e15, 41)
    rows = [dict(atomic_number='14', ion_number='1', g='2', super_level='0', energy_eV='0'),
            dict(atomic_number='14', ion_number='2', g='6', super_level='0', energy_eV='0')]
    state = dict(temperature=np.array([7000., 12000.]), ne=np.array([1e7, 1e8]),
                 full=np.array([[1e5, 2e5], [3e5, 4e5]]),
                 occupation=np.array([[.5, .8], [.9, .95]]))
    route = R.B.Route(14, 2, 0, 1, 0, 0, .3, 19., 0., nu[0],
                     ((lambda q: 2e-18*(nu[0]/q)**3, [nu[0]], nu[0], 'test'),))
    return nu, rows, state, route


def test_production_milne_contraction_and_all_nodes():
    nu, rows, state, route = fixture()
    j = np.full((2, len(nu)), 1e-6)
    radiation = dict(reference=j, same=j.copy(), doubled=2*j, zero=np.zeros_like(j))
    [(key, result)] = list(R.ion_rates(nu, radiation, [route], rows, {}, state, False,
                                     targets=((14, 2),), block_bins=7))
    assert key == (14, 2) and result['routes'] == 1
    upper = state['full'][1]*state['occupation'][0]/state['occupation'][1]
    a, s, e = R.B.milne_route(nu, route.sigma(nu), state['full'][0], upper,
        state['temperature'], state['ne'], 2., 6., route.threshold_Hz, route.probability)
    weights = R.photon_weights(nu)
    expected = dict(photo=(a*j)@weights, stimulated=(s*j)@weights, spontaneous=e@weights)
    for process, value in expected.items():
        got = result['total']['reference'][process]
        np.testing.assert_allclose(got, value, rtol=3e-14)
        np.testing.assert_array_equal(got, result['total']['same'][process])
        np.testing.assert_allclose(result['total']['doubled'][process],
                                  value*(1 if process == 'spontaneous' else 2), rtol=3e-14)
    assert np.all(result['total']['zero']['photo'] == 0)
    np.testing.assert_array_equal(result['total']['zero']['reverse'],
                                  result['total']['reference']['spontaneous'])
    np.testing.assert_allclose(weights@nu, (nu[-1]-nu[0])*4*np.pi/R.B.H, rtol=3e-15)


def test_zero_lower_keeps_reverse_and_signed_net_opacity():
    nu, rows, state, route = fixture()
    state['full'][0] = 0
    [(key, rates)] = list(R.ion_rates(nu, {'J': np.ones((2, len(nu)))}, [route],
        rows, {}, state, False, targets=((14, 2),)))
    assert np.all(rates['total']['J']['photo'] == 0)
    assert np.all(rates['total']['J']['reverse'] > 0)
    assert R._ratio(0., 0.) == 'UNDEFINED_0_OVER_0'
    assert R._ratio(1., 0.) == 'Infinity'


def test_small_populations_are_not_removed_from_rates():
    nu, rows, state, route = fixture()
    state['full'][:] = [[1e-200, 2e-200], [3e-190, 4e-190]]
    [(key, rates)] = list(R.ion_rates(nu, {'J': np.full((2, len(nu)), 1e-6)},
        [route], rows, {}, state, False, targets=((14, 2),)))
    for values in rates['total']['J'].values():
        assert np.all(values > 0) and np.isfinite(values).all()


def test_lte_planck_rate_balance_and_induced_counted_once():
    nu, rows, state, route = fixture()
    state['occupation'][:] = 1
    t = state['temperature']
    nstar = state['full'][1]*state['ne']*(R.B.H**2/(2*np.pi*R.B.ME*R.B.KB*t))**1.5/6
    state['full'][0] = nstar*np.exp(R.B.H*route.threshold_Hz/(R.B.KB*t))
    j = (2*R.B.H*nu**3/R.B.C**2)[None, :]/np.expm1(R.B.H*nu[None, :]/(R.B.KB*t[:, None]))
    [(key, rates)] = list(R.ion_rates(nu, {'LTE': j}, [route], rows, {}, state, False,
                                     targets=((14, 2),)))
    np.testing.assert_allclose(rates['total']['LTE']['photo'], rates['total']['LTE']['reverse'], rtol=3e-14)


@pytest.mark.parametrize('block_bins', [None, 1, 7, 128])
def test_model_dissolution_uses_same_production_coefficients(block_bins):
    nu, rows, state, route = fixture()
    nu = np.geomspace(.99e15, 1.01e15, 25)
    route.dissolution_primary = True
    state.update(columns=np.array([[14, 1, 0], [14, 2, 0]]),
                 superlevel=state['full'].copy(), sl_index={(14, 1, 0): 0, (14, 2, 0): 1})
    models = {(14, 2): dict(z_screen=2.), (14, 3): dict(terminal_g=1.)}
    # ground_lte_reference는 위 이온의 terminal까지 실제 물질 위상을 필요로 한다.
    state['superlevel'] = np.vstack((state['superlevel'], [5e5, 6e5]))
    state['columns'] = np.vstack((state['columns'], [14, 3, -1]))
    state['sl_index'][14, 3, -1] = 2
    j = np.full((2, len(nu)), 1e-6)
    coefficient = R.B.continuum_grid([route], rows, state, nu)
    R.B.add_dissolved_continuum(coefficient, [route], rows, models, state, nu)
    direct = R.contract_rates(coefficient, {'J': j}, R.photon_weights(nu))
    if block_bins is None:
        iterator = R.ion_rates(nu, {'J': j}, [route], rows, models, state, True, targets=((14, 2),))
    else:
        iterator = R.ion_rates_stream(nu, lambda sl: {'J': j[:, sl]}, [route], rows, models,
            state, True, targets=((14, 2),), integration_block_bins=block_bins)
    [(key, rates)] = list(iterator)
    for process, values in direct['J'].items():
        np.testing.assert_allclose(rates['total']['J'][process], values, rtol=3e-14)
    assert np.all(rates['total']['J']['photo'] > rates['bound']['J']['photo'])


@pytest.mark.parametrize('block_bins', [1, 3, 16, 41, 64])
def test_streamed_rates_keep_global_endpoint_weights_and_every_field(block_bins):
    nu, rows, state, route = fixture()
    j = np.full((2, len(nu)), 1e-6)
    fields = dict(first=j, second=2*j, zero=j*0)
    [(key, expected)] = list(R.ion_rates(nu, fields, [route], rows, {}, state, False,
                                       targets=((14, 2),)))
    [(key, actual)] = list(R.ion_rates_stream(nu, lambda sl: {k: v[:, sl] for k, v in fields.items()},
        [route], rows, {}, state, False, targets=((14, 2),), integration_block_bins=block_bins))
    for scope in ('bound', 'total'):
        for name in fields:
            for process, value in expected[scope][name].items():
                np.testing.assert_allclose(actual[scope][name][process], value, rtol=3e-14, atol=0)


def test_production_frequency_keeps_fine_nodes_and_exact_common_band():
    own = np.array([1., 2., 3., 4., 5., 6.])
    native = np.array([1.5, 2.7, 5.5])
    np.testing.assert_array_equal(R.production_frequency(own, native), [1.5, 2., 3., 4., 5., 5.5])
    with pytest.raises(ValueError): R.production_frequency(own[1:], native)


@pytest.mark.parametrize('bad', ['negative', 'nan', 'shape', 'order', 'missing'])
def test_invalid_inputs_are_not_repaired(bad):
    nu, rows, state, route = fixture()
    j = np.ones((2, len(nu)))
    targets = ((14, 2),)
    if bad == 'negative': j[0, 4] = -1
    elif bad == 'nan': j[0, 4] = np.nan
    elif bad == 'shape': j = j[:1]
    elif bad == 'order': nu = nu[::-1]
    else: targets = ((16, 3),)
    with pytest.raises(ValueError):
        list(R.ion_rates(nu, {'J': j}, [route], rows, {}, state, False, targets=targets))


def test_entry_preserves_file_and_rejects_corruption_initial_and_wrong_shape(tmp_path):
    path = tmp_path/'J-000003.npy'
    np.save(path, np.ones((4, 7)))
    entry = dict(filename=path.name, iterations=3, sha256=R.B.sha(path))
    mapped = R._entry(entry, (4, 7), tmp_path)
    assert mapped.flags.writeable is False
    mapped._mmap.close()
    with pytest.raises(ValueError): R._entry(entry, (3, 7), tmp_path)
    with pytest.raises(ValueError): R._entry(dict(entry, iterations=0), (4, 7), tmp_path)
    with path.open('ab') as stream: stream.write(b'changed')
    with pytest.raises(ValueError): R._entry(entry, (4, 7), tmp_path)


@pytest.mark.parametrize('fault', [None, 'radius', 'range', 'unselected_negative'])
def test_actual_binary_projection_keeps_original_cells_and_every_native_frequency(tmp_path, fault):
    nu = np.geomspace(1e15, 4e15, 41)
    radius = np.array([4e14, 3e14, 2e14])
    geometry = np.zeros((4, 3)); geometry[0] = radius/1e10
    native = nu[[0, 7, 21, 40]][::-1]
    reference = tmp_path/'reference.bin'
    with reference.open('xb') as stream:
        stream.write(b'FORMALPRIMARYV1'.ljust(16, b' '))
        stream.write(struct.pack('<6i', 8, 64, 32, 3, len(native), 8))
        stream.write(geometry.astype('<f8').tobytes())
        for i, f in enumerate(native):
            values = np.zeros((8, 3)); values[0] = radius/1e14+np.log(f/nu[0])
            stream.write(struct.pack('<iidd', i+1, 1, f/1e15, 1.))
            stream.write(values.astype('<f8').tobytes())
    physical = np.zeros((2, 6)); physical[:, :2] = [[2e14, 3e14], [3e14, 4e14]]
    material = dict(ns=2, physical=physical)
    j = physical[:, :2].mean(axis=1)[:, None]/1e14+np.log(nu/nu[0])[None, :]
    j = np.vstack((j, np.ones((2, len(nu)))*17))
    if fault == 'radius': physical[0, 0] *= 1.01
    elif fault == 'range': nu = nu[1:]; j = j[:, 1:]
    elif fault == 'unselected_negative': j[0, 5] = -1
    entries = {}
    for n, factor in ((3, 1), (4, 2)):
        path = tmp_path/f'J-{n:06d}.npy'; np.save(path, factor*j)
        entries[f'J{n}'] = dict(filename=path.name, iterations=n, sha256=R.B.sha(path))
    if fault:
        with pytest.raises(ValueError):
            R.projected_fields(material, nu, tmp_path, entries, reference)
        return
    target, projected = R.projected_fields(material, nu, tmp_path, entries, reference)
    np.testing.assert_array_equal(target, native[::-1])
    assert projected['J3'].shape == (2, 4)
    np.testing.assert_allclose(projected['J3'], projected['CMFGEN'], rtol=2e-15)
    np.testing.assert_array_equal(projected['J4'], 2*projected['J3'])


def test_existing_output_is_preserved_before_input_read(tmp_path):
    marker = tmp_path/'marker'
    marker.write_text('preserved')
    with pytest.raises(ValueError, match='보존'):
        R.compare(tmp_path/'absent', tmp_path/'absent', tmp_path/'absent', tmp_path/'absent', tmp_path)
    assert marker.read_text() == 'preserved'


def driver_fixture(tmp_path, monkeypatch):
    """파일 정체성/출력은 실제 처리하고 원자 모형·물질 판독만 합성으로 분리한다."""
    nu, rows, matter, template = fixture()
    rows += [{**r, 'atomic_number': '16'} for r in rows]
    matter['full'] = np.vstack((matter['full'], matter['full']))
    matter['occupation'] = np.vstack((matter['occupation'], matter['occupation']))
    matter.update(superlevel=np.array([[1e5, 2e5], [3e5, 4e5]]),
                  sl_index={(14, 3, -1): 0, (16, 3, -1): 1})
    routes = []
    for Z, stage in R.TARGETS:
        lower = (0 if Z == 14 else 2)+(stage-2)
        routes.append(R.B.Route(Z, stage, lower, lower+1 if stage == 2 else -1,
            0, 0, .3, 6., 0., nu[0], template.models))
    counts = {f'Z{Z}_stage{stage}': 1 for Z, stage in R.TARGETS}
    policy = dict(test_only=True)
    run, cells, atomic, checkpoint = [tmp_path/n for n in ('material', 'cells', 'atomic', 'checkpoint')]
    for folder in (run, cells, atomic, checkpoint, cells/'atomic'):
        folder.mkdir()
    fixed = [run/n for n in ('MODEL', 'MODEL_SPEC', 'LEVEL_SL_STEQ_LINKS', 'atomic_links.txt')]
    fixed += [cells/'material_cells.bin', cells/'atomic/levels.csv', tmp_path/'full.bin', tmp_path/'phot.so']
    for path in fixed: path.write_bytes(b'only test input')
    grid = tmp_path/'nu.npy'; np.save(grid, nu); fixed.append(grid)
    source = dict(input_sha256={str(p): R.B.sha(p) for p in fixed},
                  process_selection=policy, routes_by_ion=counts)
    for name in ('cmfgen_model_continuum.py', 'cmfgen_parser.py', 'expand_atomic_data_cmfgen.py',
                 'cmfgen_phot_smoothing.py'):
        path = Path(R.__file__).with_name(name)
        source['input_sha256'][str(path)] = R.B.sha(path)
    (tmp_path/'sources.json').write_text(json.dumps(source))
    reference = tmp_path/'FORMAL_PRIMARY_FIELDS.bin'; reference.write_bytes(b'test reference identity')
    plan = dict(initial_material=str(run), initial_native_record=70, reference_radiation_material_record=70,
        initial_record_storage_sha256='a'*64, reference_radiation_record=71,
        same_material_reference_field=str(tmp_path), reference_field_sha256=R.B.sha(reference))
    plan_path = tmp_path/'plan.json'; plan_path.write_text(json.dumps(plan))
    seed = dict(plan_sha256=R.B.sha(plan_path), source_record=70, record_storage_sha256='a'*64)
    seed_path = tmp_path/'material-seed.json'; seed_path.write_text(json.dumps(seed))
    config = dict(plan=str(plan_path), plan_sha256=R.B.sha(plan_path), seed=str(tmp_path), cells=str(cells),
        grid=str(grid), full_population=str(tmp_path/'full.bin'), phot_library=str(tmp_path/'phot.so'),
        reuse_sources=dict(cell_sources=dict(directory=str(tmp_path), report_sha256=R.B.sha(tmp_path/'sources.json'))))
    config_path = tmp_path/'config.json'; config_path.write_text(json.dumps(config))
    entries = []
    for n in (3, 4):
        path = checkpoint/f'J-{n:06d}.npy'; np.save(path, np.full((4, len(nu)), float(n)))
        entries.append(dict(filename=path.name, iterations=n, sha256=R.B.sha(path)))
    saved = dict(schema='lumina-outer-nonlinear-checkpoint-v1', phase='ITERATION_COMPLETED',
        accepted_physical_field=False, componentwise_verified=False, J=entries[1], completed_iterations=4,
        previous_residual=dict(input_J=entries[0], output_J=entries[1]), contract=dict(shape=[4, len(nu)]))
    saved['metadata_sha256'] = R.digest(saved)
    saved_path = checkpoint/'checkpoint.json'; saved_path.write_text(json.dumps(saved))
    diagnostic_paths = []
    for selection, entry in zip(('previous', 'current'), entries, strict=True):
        d = dict(schema='lumina-outer-checkpoint-diagnostic-v1', status='UNVERIFIED_CHECKPOINT_DIAGNOSTIC',
            accepted_physical_field=False, componentwise_verified=False, selection=selection,
            comparison_J=entry, current_J=entries[1], material_record=70, reference_record=71,
            radiation_sha256=dict(cmfgen=plan['reference_field_sha256']),
            captured_sha256={str(p): R.B.sha(p) for p in (config_path, plan_path, seed_path, saved_path)})
        path = tmp_path/(selection+'.json'); path.write_text(json.dumps(d)); diagnostic_paths.append(path)
    radiation = dict(CMFGEN=np.full((2, len(nu)), 1e-6),
                     J3=np.full((2, len(nu)), 2e-6), J4=np.full((2, len(nu)), 3e-6))
    physical = np.zeros((2, 6)); physical[:, :2] = [[2., 3.], [3., 4.]]
    monkeypatch.setattr(R, 'projected_fields', lambda *args: (nu, radiation))
    monkeypatch.setattr(R, 'read_material', lambda *args: dict(ns=2, physical=physical))
    monkeypatch.setattr(R.B, 'load_model', lambda *args: (rows, {}, routes, policy, {}, counts))
    monkeypatch.setattr(R.B, 'read_material', lambda *args: matter)
    monkeypatch.setattr(R.B, 'dissolution_flag', lambda *args: False)
    monkeypatch.setattr(R.B, 'consume_c_full_population', lambda *args: dict(test_only=True))
    return config_path, checkpoint, *diagnostic_paths


@pytest.mark.parametrize('integration_grid', ['reference', 'production'])
@pytest.mark.parametrize('fault', [None, 'generation', 'metadata', 'fixed_input'])
def test_driver_publication_and_identity_checks(tmp_path, monkeypatch, fault, integration_grid):
    args = driver_fixture(tmp_path, monkeypatch)
    out = tmp_path/'out'
    if fault == 'generation':
        d = json.loads(args[2].read_text()); d['material_record'] = 13
        args[2].write_text(json.dumps(d))
    elif fault == 'metadata':
        path = args[1]/'checkpoint.json'
        d = json.loads(path.read_text()); d['completed_iterations'] = 99
        path.write_text(json.dumps(d))
    elif fault == 'fixed_input':
        (tmp_path/'cells/material_cells.bin').write_bytes(b'changed')
    if fault:
        with pytest.raises(ValueError): R.compare(*args, out, integration_grid)
        assert not out.exists()
        return
    result = R.compare(*args, out, integration_grid)
    assert result['material_record'] == 70 and result['reference_record'] == 71
    assert result['physical_RT_executions'] == result['native_executions'] == result['matter_updates'] == 0
    assert result['accepted_physical_field'] is False and result['final_goal'] == 'NOT_JUDGED'
    assert result['rates_csv_sha256'] == R.B.sha(out/'rates.csv')
    records = list(csv.DictReader((out/'rates.csv').open()))
    assert len(records) == 8
    for row in records:
        # production 모드에서는 실제 저장된3/4를 읽으며 합성 투영값을 재사용하지 않는다.
        ratios = (2., 3.) if integration_grid == 'reference' else (3e6, 4e6)
        assert float(row['J3_photo_over_CMFGEN']) == pytest.approx(ratios[0], rel=3e-14)
        assert float(row['J4_photo_over_CMFGEN']) == pytest.approx(ratios[1], rel=3e-14)
