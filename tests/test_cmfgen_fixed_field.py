"""새 J60과 native 원래 SE 갱신의 범위/원본 보존을 검사한다."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_cmfgen_fixed_field as fixed
from run_cmfgen_lambda_probe import control


def inputs():
    values=dict(FIX_T='T',FIX_T_AUTO='F',FIX_NE='F',DO_DDT='F',DO_NG='F',TRT_NON_TE='F',
                USE_FIXED_J='F',WRITE_RATES='T',WRITE_JH='T',INC_GRID='F')
    return ''.join(f'{v} [{k}]\n' for k,v in values.items()),'19 [NUM_ITS]\nF [DO_LAM_IT]\nF [DO_LAM_AUTO]\nF [DO_T_AUTO]\n'


def test_only_fixed_field_and_iteration_controls_change():
    vadat,it=inputs();new_v,new_it=fixed.controls(vadat,it)
    assert new_v==vadat.replace('F [USE_FIXED_J]','T [USE_FIXED_J]')
    assert new_it==it.replace('19 [NUM_ITS]','4 [NUM_ITS]').replace('F [DO_LAM_IT]','T [DO_LAM_IT]')
    assert control(new_v,'FIX_T')=='T' and control(new_v,'FIX_NE')=='F'


@pytest.mark.parametrize('key,original,new',[('FIX_T','T','F'),('FIX_NE','F','T'),
    ('DO_DDT','F','T'),('INC_GRID','F','T')])
def test_physics_and_unsupported_mode_changes_are_not_silently_accepted(key,original,new):
    vadat,it=inputs()
    with pytest.raises(ValueError):fixed.controls(vadat.replace(f'{original} [{key}]',f'{new} [{key}]'),it)


def test_existing_run_is_refused_before_field_reads(tmp_path):
    with pytest.raises(ValueError,match='기존 계산'):
        fixed.prepare(Path('not-read'),Path('not-read'),tmp_path,Path('not-read'))


def test_resource_count_does_not_follow_external_links_and_detects_new_output_directory(tmp_path):
    root=tmp_path/'run';root.mkdir();(root/'seq_logs').mkdir()
    outside=tmp_path/'external';outside.write_bytes(b'x'*10000)
    (root/'atom_link').symlink_to(outside)
    (root/'OUTGEN').write_bytes(b'normal')
    assert fixed.output_bytes(root)==6
    (root/'unexpected').mkdir()
    with pytest.raises(ValueError,match='하위 디렉터리'):fixed.output_bytes(root)


def test_native_error_and_root_limit_stop_run(tmp_path,monkeypatch):
    (tmp_path/'seq_logs').mkdir();(tmp_path/'OUTGEN').write_text('Error reading EDDFACTOR\n')
    assert fixed.scan(tmp_path,{})=='native_solver_error:OUTGEN:Error'
    (tmp_path/'OUTGEN').write_text('LAMBDA iteration used\n')
    monkeypatch.setattr(fixed,'output_bytes',lambda _:17*1024**3)
    assert fixed.scan(tmp_path,{})=='root_size_cap_16GiB'


def test_preserved_isotope_warning_does_not_kill_a_valid_solver(tmp_path):
    (tmp_path/'seq_logs').mkdir()
    (tmp_path/'OUTGEN').write_text(' Warning: Possible error in reading SN_HYDRO_DATA with RD_SN_DATA.\n'
        ' LAMBDA iteration used\n')
    assert fixed.scan(tmp_path,{}) is None
    with (tmp_path/'OUTGEN').open('a') as stream:stream.write(' Error reading EDDFACTOR in RD_CONT_J\n')
    assert fixed.scan(tmp_path,{}) is not None
