"""직접평가 참조는 CROSS만 바꾸고 승인 밖 실행/물질 갱신을 하지 않는다."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_cmfgen_direct_continuum as D


def vadat():
    return 'F [CROSS]\n750.0 [V_CROSS]\nF [USE_FIXED_J]\nT [FIX_T]\nF [INC_GRID]\n'


def test_only_cross_changes_and_velocity_spacing_is_preserved():
    original=vadat();actual=D.direct_vadat(original)
    assert actual==original.replace('F [CROSS]','T [CROSS]')
    assert D.control(actual,'V_CROSS')=='750.0'


@pytest.mark.parametrize('old,new',[
    ('F [CROSS]','T [CROSS]'),('750.0','500.0'),('F [USE_FIXED_J]','T [USE_FIXED_J]'),
    ('T [FIX_T]','F [FIX_T]'),('F [INC_GRID]','T [INC_GRID]')])
def test_different_physical_baseline_is_rejected(old,new):
    with pytest.raises(ValueError):D.direct_vadat(vadat().replace(old,new))


def test_duplicate_cross_is_not_silently_rewritten():
    with pytest.raises(ValueError):D.direct_vadat(vadat()+'F [CROSS]\n')


def test_native_process_cannot_start_without_the_exact_approved_allocation(monkeypatch,tmp_path):
    monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    monkeypatch.setattr(D,'prepare',lambda *args:pytest.fail('승인 할당 밖 준비/실행'))
    with pytest.raises(ValueError):D.run(tmp_path,tmp_path/'new')
