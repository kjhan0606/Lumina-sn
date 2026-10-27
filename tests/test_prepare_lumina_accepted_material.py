"""수용 물질을 이전70과 혼동하지 않고 같은 점의 계수 입력으로 전달한다."""
import csv
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import prepare_lumina_accepted_material as E


def fixture(root, monkeypatch, change=None):
    seed = root/'seed'; atomic = seed/'atomic'; atomic.mkdir(parents=True)
    run = root/'native'; run.mkdir()
    for name, text in (('MODEL', 'F [DO_DIS]\n'), ('MODEL_SPEC', 'test'), ('atomic_links.txt', 'test')):
        (run/name).write_text(text)
    rows = [dict(atomic_number=14, ion_number=1, super_level=0, level_number=i, g=g, energy_eV=e)
            for i, g, e in ((0, 2, 0), (1, 4, 1))]
    with (atomic/'levels.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (atomic/'abundances.csv').write_text('atomic_number,0,1,2\n14,1,1,1\n')
    (atomic/'element_number_densities.csv').write_text(
        'atomic_number,shell_id,N_element_cm3\n14,0,14\n14,1,14\n14,2,14\n')
    (atomic/'electron_temperatures.csv').write_text('shell_id,T_e\n0,10000\n1,10000\n2,10000\n')
    columns = np.array([[14, 1, 0], [14, 2, -1]], dtype='<i4')
    physical = np.array([[3.e14, 1.e4, 16., 1.e-14], [2.e14, 1.e4, 18., 2.e-14],
                         [1.e14, 1.e4, 20., 3.e-14]])
    nodes = seed/'native_material_nodes.bin'
    with nodes.open('wb') as f:
        f.write(struct.pack('<8sIIId', b'LUMNOD01', 3, 2, 2, 1.e6))
        for a, dtype in ((columns, '<i4'), (physical, '<f8'), ([[12, 2], [10, 4], [8, 6]], '<f8')):
            f.write(np.asarray(a, dtype=dtype).tobytes())
    names = ['native_material_nodes.bin', 'atomic/abundances.csv',
             'atomic/element_number_densities.csv', 'atomic/electron_temperatures.csv']
    (seed/'material-seed.json').write_text(json.dumps(dict(output_sha256={n: E.B.sha(seed/n) for n in names})))
    models = {(14, 2): dict(n_super=1)}
    monkeypatch.setattr(E.B.atomic, 'load_cmfgen_model_spec', lambda *args: models)
    temperature = np.array([8000., 12000., 16000.]); pop = np.array([[9., 8., 7.], [5., 6., 7.]])
    ne = (pop*columns[:, 1, None]).sum(axis=0)
    weight = np.array([2., 4.])[:, None]*np.exp(-np.array([0., 1.])[:, None]*E.B.EV/(E.B.KB*temperature))
    candidate = dict(columns=columns, population=pop, temperature=temperature, ne=ne,
                     fraction=weight/weight.sum(axis=0), occupation=np.ones((2, 3)),
                     ledger=np.zeros((1, 3, 7)), charge_residual=np.zeros(3))
    if change is not None:
        change(candidate)
    directory = root/'evaluation-1'; directory.mkdir()
    for p in (root/'accepted-response.npz', directory/'response.npz'):
        np.savez(p, **candidate)
    np.savez(directory/'balance.npz', native_stored_net=np.zeros(3))
    report = dict(state='COMPLETE_DEPTH_COUPLED_MATERIAL_STEP', charge_solved=True, points=3,
        history=[dict(evaluation=1, accepted=True)], accepted_source_directory=str(directory),
        evaluations=[dict(output_sha256={n: E.B.sha(directory/n) for n in ('response.npz', 'balance.npz')})],
        accepted_output_sha256=E.B.sha(root/'accepted-response.npz'),
        input_sha256={str(p): E.B.sha(p) for p in (nodes, atomic/'levels.csv',
            *[run/n for n in ('MODEL', 'MODEL_SPEC', 'atomic_links.txt')])})
    result = root/'result.json'; result.write_text(json.dumps(report))
    return result, seed, run, models, physical, candidate


def test_roundtrip_uses_actual_accepted_state_and_preserves_original_nodes(tmp_path, monkeypatch):
    result, seed, run, models, physical, candidate = fixture(tmp_path, monkeypatch)
    original = {p: E.B.sha(p) for p in (seed/'native_material_nodes.bin', result, seed/'atomic/abundances.csv')}
    out = tmp_path/'export'; report = E.prepare(result, E.B.sha(result), seed, run, out)
    columns, actual, population, nl, epoch = E.read_nodes(out/'native_material_nodes.bin')
    np.testing.assert_array_equal(actual[:, [0, 3]], physical[:, [0, 3]])
    np.testing.assert_array_equal(actual[:, 1], candidate['temperature'])
    np.testing.assert_array_equal(actual[:, 2], candidate['ne'])
    np.testing.assert_array_equal(population.T, candidate['population'])
    assert (nl, epoch) == (2, 1.e6)
    assert report['checks']['restored_full_relative_max'] < 5.e-12
    assert report['new_transport'] == report['new_SE_solves'] == report['new_BF_BB_integrals'] == 0
    assert report['C_full_population_exported'] is False
    assert not (out/'material-seed.json').exists() and not (out/'sources.bin').exists()
    assert all(E.B.sha(p) == h for p, h in original.items())
    for name in ('abundances.csv', 'element_number_densities.csv'):
        assert (out/'atomic'/name).read_bytes() == (seed/'atomic'/name).read_bytes()


@pytest.mark.parametrize('change', [
    lambda c: c['ne'].__setitem__(0, c['ne'][0]*1.01),
    lambda c: c['population'].__setitem__((0, 0), c['population'][0, 0]+1),
    lambda c: c['columns'].__setitem__((0, 2), 99),
    lambda c: c['occupation'].__setitem__((0, 0), .5),
    lambda c: c['fraction'].__setitem__((0, 0), .5),
    lambda c: c['ledger'].__setitem__((0, 0, 6), np.nan),
])
def test_semantically_wrong_state_is_rejected_even_with_valid_hashes(tmp_path, monkeypatch, change):
    result, seed, run, *_ = fixture(tmp_path, monkeypatch, change)
    out = tmp_path/'export'
    with pytest.raises(ValueError):
        E.prepare(result, E.B.sha(result), seed, run, out)
    assert not (out/'accepted-material.json').exists()


def test_wrong_hash_stale_model_and_existing_output_are_not_consumed(tmp_path, monkeypatch):
    result, seed, run, *_ = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='신원'):
        E.prepare(result, '0'*64, seed, run, tmp_path/'out')
    with pytest.raises(FileExistsError, match='보존'):
        E.prepare(Path('not_read'), 'not_read', Path('not_read'), Path('not_read'), tmp_path)
    (run/'MODEL').write_text('modified')
    with pytest.raises(ValueError, match='입력 변경'):
        E.prepare(result, E.B.sha(result), seed, run, tmp_path/'out')


def test_accepted_copy_must_match_selected_evaluation(tmp_path, monkeypatch):
    result, seed, run, *_ = fixture(tmp_path, monkeypatch)
    report = json.loads(result.read_text()); report['evaluations'][0]['output_sha256']['response.npz'] = '0'*64
    result.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='evaluation'):
        E.prepare(result, E.B.sha(result), seed, run, tmp_path/'out')


def test_dissolution_uses_original_osc_metadata_not_model_spec_alone(tmp_path, monkeypatch):
    result, seed, run, models, *_ = fixture(tmp_path, monkeypatch)
    (run/'MODEL').write_text('T [DO_DIS]\n'); osc = run/'osc'; osc.write_text('sealed OSC')
    monkeypatch.setattr(E.B.atomic, 'load_cmfgen_links', lambda *args: {(14, 2): dict(osc=str(osc))})
    monkeypatch.setattr(E.B, 'parse_osc', lambda *args, **kwargs: SimpleNamespace(ionization_eV=20., z_screen=2.))
    report = json.loads(result.read_text())
    for path in (run/'MODEL', osc):
        report['input_sha256'][str(path)] = E.B.sha(path)
    result.write_text(json.dumps(report))
    exported = E.prepare(result, E.B.sha(result), seed, run, tmp_path/'export')
    assert exported['checks']['occupation_relative_max'] == 0
    assert str(osc) in exported['input_sha256']
    assert models[14, 2]['ionization_eV'] == 20.


def new_field_report(result, *, selected_index=1):
    old = json.loads(result.read_text()); root = result.parent
    entries = []
    for index in range(1, selected_index+1):
        directory = root/('evaluation-'+str(index))
        if index > 1:
            directory.mkdir()
            for name in ('response.npz', 'balance.npz'):
                (directory/name).write_bytes((root/'evaluation-1'/name).read_bytes())
        entries.append(dict(root=str(directory), response_sha256=E.B.sha(directory/'response.npz'),
                            balance_sha256=E.B.sha(directory/'balance.npz')))
    report = dict(state='COMPLETE_FINITE_NEW_FIELD_RESPONSE_NOT_JOINT_CONVERGENCE',
        charge_converged=True, fixed_accepted_temperature=True, points=old['points'],
        selected_response=entries[-1], evaluations=entries,
        charge_history=[dict(evaluation=i, accepted=True) for i in range(1, selected_index)],
        input_sha256=old['input_sha256'])
    result.write_text(json.dumps(report))
    return report


@pytest.mark.parametrize('index', [1, 2])
def test_new_field_converged_selected_response_roundtrips(tmp_path, monkeypatch, index):
    result, seed, run, models, physical, candidate = fixture(tmp_path, monkeypatch)
    new_field_report(result, selected_index=index)
    output = tmp_path/'new-export'; exported = E.prepare(result, E.B.sha(result), seed, run, output)
    _, actual, population, _, _ = E.read_nodes(output/'native_material_nodes.bin')
    np.testing.assert_array_equal(actual[:, [0, 3]], physical[:, [0, 3]])
    np.testing.assert_array_equal(population.T, candidate['population'])
    assert exported['source_kind'] == 'FINITE_NEW_FIELD_FIXED_T_CHARGE_RESPONSE'
    assert exported['accepted_evaluation'] == index
    assert exported['accepted_response_sha256'] == E.B.sha(result.parent/('evaluation-'+str(index))/'response.npz')
    assert exported['final_goal'] == 'NOT_MET' and exported['new_transport'] == 0


@pytest.mark.parametrize('fault', ['not_converged', 'bad_fixed_T', 'incorrect_selection',
    'wrong_history', 'changed_response', 'changed_balance', 'incomplete'])
def test_new_field_rejects_unconverged_or_mismatched_candidate(tmp_path, monkeypatch, fault):
    result, seed, run, *_ = fixture(tmp_path, monkeypatch)
    report = new_field_report(result, selected_index=2)
    if fault == 'not_converged': report['charge_converged'] = False
    if fault == 'bad_fixed_T': report['fixed_accepted_temperature'] = False
    if fault == 'incorrect_selection': report['selected_response']['root'] = str(tmp_path/'evaluation-1')
    if fault == 'wrong_history': report['charge_history'][0]['accepted'] = False
    if fault == 'incomplete': report['state'] = 'INCOMPLETE_PRESERVED'
    if fault in ('changed_response', 'changed_balance'):
        (tmp_path/'evaluation-2'/('response.npz' if fault == 'changed_response' else 'balance.npz')).write_bytes(b'changed')
    result.write_text(json.dumps(report))
    with pytest.raises(ValueError): E.prepare(result, E.B.sha(result), seed, run, tmp_path/'export')
    assert not (tmp_path/'export/accepted-material.json').exists()
