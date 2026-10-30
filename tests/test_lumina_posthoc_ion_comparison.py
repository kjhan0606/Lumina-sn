from pathlib import Path

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
