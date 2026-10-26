from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_lumina_common_material import absent_partial_auto_inputs


def test_partial_auto_absence_and_later_file_appearance_are_distinguished(tmp_path):
    models={(14,2):dict(label='Sk2'),(16,3):dict(label='SIII')}
    assert absent_partial_auto_inputs(tmp_path,models)==['Sk2_AUTO_DATA','SIII_AUTO_DATA']
    (tmp_path/'SIII_AUTO_DATA').write_text('읽지 않은 자료를 자동0으로 간주하지 않는다')
    with pytest.raises(ValueError,match='SIII_AUTO_DATA'):absent_partial_auto_inputs(tmp_path,models)


def test_broken_auto_link_is_not_silently_treated_as_verified_absence(tmp_path):
    models={(14,2):dict(label='Sk2')}
    (tmp_path/'Sk2_AUTO_DATA').symlink_to(tmp_path/'missing-original')
    with pytest.raises(ValueError,match='Sk2_AUTO_DATA'):absent_partial_auto_inputs(tmp_path,models)
