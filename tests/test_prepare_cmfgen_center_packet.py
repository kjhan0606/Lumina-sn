"""원 formal 커널을 바꾸지 않고 목표 주파수 계측만 선택한다."""
import pytest

from prepare_cmfgen_center_packet import NEW_SELECTION, OLD_SELECTION, selected_source


def test_exact_old_selection_only():
    source = 'before\n' + OLD_SELECTION + '\nafter\n'
    assert selected_source(source) == 'before\n' + NEW_SELECTION + '\nafter\n'
    with pytest.raises(ValueError):
        selected_source(source + OLD_SELECTION)
    with pytest.raises(ValueError):
        selected_source('FORMAL_SELECTED=.TRUE.')
