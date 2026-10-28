"""이전 물질의 완성장은 초기 추정값일 뿐 새 물질의 완료장이 아니다."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import lumina_material_trial_field as T
from lumina_fullsr_sc_actual import InputField
from lumina_native_columns import sha
from test_lumina_resident_feedback import seed_field


@pytest.fixture
def changed_material(tmp_path, monkeypatch):
    class Source:
        ns, nf = 3, 8
        radius = np.array([3., 2., 1.])
        frequency = np.arange(1., 9.)
        closed = False
        def __enter__(self): return self
        def __exit__(self, *args): self.closed = True
    old = Source(); new = Source()
    old.identity = dict(material='old', epoch_seconds=100.)
    new.identity = dict(material='new', epoch_seconds=100.)
    target, report, record = seed_field(dict(out=tmp_path/'unused'), old.identity)
    previous = tmp_path/'old-preparation.json'; previous.write_text('{}')
    calls = []
    def prepared(path, digest):
        calls.append((path, digest))
        if sha(path) != digest: raise ValueError('이전 준비 기록 변경')
        return {}, old, None, None
    monkeypatch.setattr(T, 'open_prepared', prepared)
    args = dict(initial_preparation=previous, initial_preparation_sha256=sha(previous))
    return new, old, target, report, record, args, calls


def test_previous_material_is_readonly_trial_and_strict_consumer_still_refuses_it(changed_material):
    new, old, path, _, _, args, calls = changed_material
    with pytest.raises(ValueError, match='동일 물질'): InputField(new, path, sha(path))
    field = T.open_transport_trial(new, path, sha(path), **args)
    try:
        np.testing.assert_array_equal(field.values, np.full((3, 8), .5))
        assert not field.values.flags.writeable
        assert field.identity['origin_source_identity'] == old.identity
        assert field.identity['target_source_identity'] == new.identity
        assert field.identity['self_consistent'] is False
        assert field.identity['accepted_field_for_target_material'] is False
        assert field.identity['initial_preparation_sha256'] == args['initial_preparation_sha256']
        assert old.closed and not new.closed and len(calls) == 1
    finally: field.close()
    # 파일 자체는 이전 물질의 완료장이다. 초기 추정값 연결로 계보를 덮어쓰지 않는다.
    with pytest.raises(ValueError, match='동일 물질'): InputField(new, path, sha(path))


@pytest.mark.parametrize('bad', ['radius', 'frequency', 'epoch', 'missing_epoch', 'shape'])
def test_changed_grid_or_epoch_is_not_interpolated(changed_material, bad):
    new, old, path, _, _, args, _ = changed_material
    if bad == 'radius': new.radius = new.radius.copy(); new.radius[0] += .01
    elif bad == 'frequency': new.frequency = new.frequency.copy(); new.frequency[0] += .01
    elif bad == 'epoch': new.identity['epoch_seconds'] += 1
    elif bad == 'missing_epoch': new.identity.pop('epoch_seconds')
    else: new.nf += 1
    with pytest.raises(ValueError, match='같은 반경/주파수/epoch'):
        T.open_transport_trial(new, path, sha(path), **args)
    assert old.closed and not new.closed


@pytest.mark.parametrize('bad', ['partial', 'wrong_source', 'digest', 'report_path',
                               'preparation_hash', 'negative', 'nan'])
def test_trial_does_not_bypass_original_completion_identity_or_values(changed_material, bad):
    new, old, path, report, record, args, _ = changed_material
    digest = sha(path)
    if bad == 'partial': record['complete_original_grid'] = False
    elif bad == 'wrong_source': record['source_identity'] = new.identity
    elif bad == 'digest': digest = '0'*64
    elif bad == 'report_path': record['J']['path'] = str(path.parent/'other.npy')
    elif bad == 'preparation_hash': args['initial_preparation_sha256'] = '0'*64
    else:
        values = np.full((3, 8), .5); values[0, 0] = -1 if bad == 'negative' else np.nan
        np.save(path, values); digest = sha(path); record['J']['sha256'] = digest
    report.write_text(json.dumps(record))
    with pytest.raises(ValueError): T.open_transport_trial(new, path, digest, **args)


@pytest.mark.parametrize('args', [dict(input_j='J'), dict(input_j_sha256='sha'),
    dict(initial_preparation='P'), dict(initial_preparation_sha256='sha'),
    dict(initial_preparation='P', initial_preparation_sha256='sha')])
def test_incomplete_cli_pairs_cannot_open_any_input(args):
    options = dict(input_j=None, input_j_sha256=None, initial_preparation=None,
                   initial_preparation_sha256=None)
    options.update(args)
    with pytest.raises(ValueError): T.validate_initial_arguments(**options)


def test_legacy_same_material_path_remains_strict_and_keeps_its_identity(changed_material):
    _, old, path, _, _, _, calls = changed_material
    field = T.open_transport_trial(old, path, sha(path))
    try: assert field.identity['role'] == 'OWN_NATIVE90_COMPLETED_FIELD'
    finally: field.close()
    assert not calls
