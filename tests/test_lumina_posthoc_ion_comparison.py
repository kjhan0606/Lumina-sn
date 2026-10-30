from pathlib import Path
import json

import numpy as np
import pytest

import compare_lumina_new_field_response_eval as C
import cmfgen_model_continuum as B


def test_input_path_requires_exactly_one_basename():
    assert C.input_path({'/a/preparation.json': 'x'}, 'preparation.json') == Path('/a/preparation.json')
    with pytest.raises(ValueError, match='주소가 하나로'):
        C.input_path({'/a/levels.csv': 'x', '/b/levels.csv': 'y'}, 'levels.csv')
    with pytest.raises(ValueError, match='주소가 하나로'):
        C.input_path({}, 'feedback.json')


def test_evaluation_files_binds_index_directory_and_content_hashes(tmp_path):
    root = tmp_path/'evaluation-1'; root.mkdir()
    response = root/'response.npz'; response.write_bytes(b'response fixture')
    balance = root/'balance.npz'; balance.write_bytes(b'balance fixture')
    entry = dict(root=str(root), response_sha256=B.sha(response), balance_sha256=B.sha(balance))
    selected, response_path, balance_path = C.evaluation_files(
        tmp_path/'result.json', {'evaluations': [entry]}, 1)
    assert selected == entry
    assert response_path == response and balance_path == balance
    with pytest.raises(ValueError, match='주소 불일치'):
        C.evaluation_files(tmp_path/'result.json', {'evaluations': [dict(entry, root=str(tmp_path))]}, 1)
    response.write_bytes(b'changed response')
    with pytest.raises(ValueError, match='해시 불일치'):
        C.evaluation_files(tmp_path/'result.json', {'evaluations': [entry]}, 1)


def test_evaluation_index_must_be_published(tmp_path):
    with pytest.raises(ValueError, match='인덱스'):
        C.evaluation_files(tmp_path/'result.json', {'evaluations': []}, 1)
    with pytest.raises(ValueError, match='인덱스'):
        C.evaluation_files(tmp_path/'result.json', {'evaluations': [{}]}, 0)


def test_accepted_material_hashes_are_read_from_its_manifest(tmp_path):
    manifest_dir = tmp_path/'accepted'; manifest_dir.mkdir()
    material = manifest_dir/'native_material_nodes.bin'; material.write_bytes(b'nodal material')
    manifest = manifest_dir/'accepted-material.json'
    manifest.write_text(__import__('json').dumps({
        'output_sha256': {'native_material_nodes.bin': B.sha(material)}}))
    preparation = dict(accepted_material=str(manifest), material=str(material),
        input_sha256={str(manifest): B.sha(manifest)})
    assert C.accepted_material_paths(preparation) == material
    material.write_bytes(b'modified material')
    with pytest.raises(ValueError, match='물질 파일의 봉인 해시'):
        C.accepted_material_paths(preparation)


def test_original_reference_must_match_manifest_and_run_hash(tmp_path):
    original_dir = tmp_path/'original'; original_dir.mkdir()
    original = original_dir/'native_material_nodes.bin'; original.write_bytes(b'original')
    digest = B.sha(original)
    manifest = tmp_path/'accepted-material.json'
    manifest.write_text(json.dumps({'input_sha256': {str(original): digest}}))
    preparation = dict(accepted_material=str(manifest), input_sha256={str(manifest): B.sha(manifest)})
    report = {'input_sha256': {str(original): digest}}
    assert C.original_reference_path(preparation, report) == original
    with pytest.raises(ValueError, match='manifest/실행 봉인 해시'):
        C.original_reference_path(preparation, {'input_sha256': {str(original): 'wrong'}})
    original.write_bytes(b'changed')
    with pytest.raises(ValueError, match='manifest/실행 봉인 해시'):
        C.original_reference_path(preparation, report)


def test_comparison_uses_manifest_original_not_previous_accepted_material(tmp_path, monkeypatch):
    original_dir = tmp_path/'original'; original_dir.mkdir()
    accepted_dir = tmp_path/'accepted'; accepted_dir.mkdir()
    original = original_dir/'native_material_nodes.bin'; original.write_bytes(b'CMFGEN original')
    accepted = accepted_dir/'native_material_nodes.bin'; accepted.write_bytes(b'previous Lumina')
    manifest = accepted_dir/'accepted-material.json'
    manifest.write_text(json.dumps(dict(input_sha256={str(original): B.sha(original)},
        output_sha256={'native_material_nodes.bin': B.sha(accepted)})))
    preparation = tmp_path/'preparation.json'
    preparation.write_text(json.dumps(dict(accepted_material=str(manifest), material=str(accepted),
        input_sha256={str(manifest): B.sha(manifest), str(accepted): B.sha(accepted)})))
    feedback = tmp_path/'feedback.json'
    feedback.write_text(json.dumps({'last_complete': {'J': {'sha256': 'a'*64}}}))
    files = {}
    for name in ('MODEL', 'MODEL_SPEC', 'atomic_links.txt', 'levels.csv'):
        path = tmp_path/name; path.write_text('header\n'); files[str(path)] = B.sha(path)
    columns = np.array([[14, 1, 0], [14, 2, 0]], dtype=np.int32)
    population = np.array([[.5, .5], [.5, .5]])
    evaluation = tmp_path/'evaluation-1'; evaluation.mkdir()
    response = evaluation/'response.npz'
    np.savez(response, columns=columns, population=population,
             temperature=np.array([100., 100.]), ne=np.array([10., 10.]))
    balance = evaluation/'balance.npz'; balance.write_bytes(b'balance')
    entry = dict(root=str(evaluation), generation=11, response_sha256=B.sha(response),
                 balance_sha256=B.sha(balance))
    report = dict(state='STARTED', final_goal='NOT_MET', selected_response=None,
        comparison_baseline='producer-declared baseline', evaluations=[entry],
        input_sha256={str(preparation): B.sha(preparation), str(feedback): B.sha(feedback),
                      str(original): B.sha(original), **files})
    result = tmp_path/'result.json'; result.write_text(json.dumps(report))
    monkeypatch.setattr(B.atomic, 'load_cmfgen_model_spec', lambda *_: {})
    def fake_material(path, _rows, _models):
        if path == original:
            pops = np.array([[.8, .8], [.2, .2]])
        elif path == accepted:
            pops = np.array([[.6, .6], [.4, .4]])
        else:
            raise AssertionError('unexpected material')
        return dict(columns=columns, physical=np.array([[1., 0., 0., 0.], [2., 0., 0., 0.]]),
                    superlevel=pops, temperature=np.array([100., 100.]), ne=np.array([10., 10.]))
    monkeypatch.setattr(B, 'read_material', fake_material)
    compared = C.compare_result(result, B.sha(result), 1)
    stage = next(row for row in compared['ion_summary'] if row['Z'] == 14 and row['charge'] == 2)
    assert stage['reference_fraction'] == pytest.approx(.2)
    assert stage['new_response_fraction'] == pytest.approx(.5)
    assert stage['max_absolute_fraction_difference'] == pytest.approx(.3)
    assert stage['max_change_from_accepted'] == pytest.approx(.1)
    assert compared['reference_material_sha256'] == B.sha(original)
    assert compared['accepted_material_sha256'] == B.sha(accepted)
    assert compared['baseline'] == 'ACCEPTED_MANIFEST_ORIGINAL_NATIVE_MATERIAL_NOT_INDEPENDENTLY_CERTIFIED'
    assert compared['producer_declared_baseline'] == 'producer-declared baseline'
