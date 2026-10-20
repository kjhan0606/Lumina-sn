"""수송 세대는 명시적으로 바꾸되 기존 물리/coverage/STOP은 유지한다."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_cmfgen_feedback_build import next_generation, prepare


def original():
    return (Path(__file__).resolve().parents[1]/'scripts/cmfgen_formal_primary_whole.f90').read_text()


@pytest.mark.parametrize('iteration', [64,65])
def test_generation_change_preserves_every_other_byte_and_stop(iteration):
    source = original()
    changed = next_generation(source, iteration)
    assert f'.or.lumina_outer/={iteration})' in changed
    assert f'write(lu,*){iteration},pw_count,pw_expected,pw_calls' in changed
    assert changed.count('stop 97') == 1
    restored = changed.replace(f'.or.lumina_outer/={iteration})', '.or.lumina_outer/=60)')
    restored = restored.replace(f'write(lu,*){iteration},pw_count', 'write(lu,*)60,pw_count')
    assert restored == source


@pytest.mark.parametrize('iteration', [0, 59, 60, 64.5])
def test_implicit_or_invalid_generation_is_refused(iteration):
    with pytest.raises(ValueError):
        next_generation(original(), iteration)


@pytest.mark.parametrize('old,new', [('.or.lumina_outer/=60)', ')'),
                                  ('stop 97', 'return'),
                                  ('write(lu,*)60,pw_count', 'write(lu,*)61,pw_count')])
def test_unexpected_or_unprotected_parent_is_refused(old, new):
    with pytest.raises((ValueError, RuntimeError)):
        next_generation(original().replace(old, new), 64)


def test_existing_build_is_preserved_before_input_reads(tmp_path):
    with pytest.raises(ValueError, match='기존'):
        prepare(Path('not-read'), tmp_path, 64)
