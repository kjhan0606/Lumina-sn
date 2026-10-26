import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_cmfgen_fixedj_exponent import OLD, NEW, patched


def test_exact_single_change_preserves_surrounding_source():
    assert patched('before\n'+OLD+'\nafter\n') == 'before\n'+NEW+'\nafter\n'
    # Only the actual-frequency factor leaves COMPUTE_NEW_CROSS. CONT_FREQ stays cached.
    assert NEW.index('EMHNUKT_CONT') < NEW.index('END IF') < NEW.index('T1=-HDKT*FL')
    assert NEW.count('EMHNUKT(1:ND)=EXP(T1/T(1:ND))') == 1


def test_missing_duplicate_and_already_patched_sources_are_rejected():
    for source in ('unrelated', OLD+OLD, NEW):
        with pytest.raises(ValueError, match='삽입점'):
            patched(source)
