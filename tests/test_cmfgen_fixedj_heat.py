"""공통-J 실제 RE 준비 경로의 산술 보존, 갱신 전 종료, 직렬화 직접시험."""
import os
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(SCRIPTS))
from prepare_cmfgen_fixedj_heat import (COMP_OPAC, FINISH, FINISH_POINT, FIXED_SKIP,
                                      GUARD, GUARD_POINT, patched, read_heat_row)


def test_private_patch_restores_actual_coefficients_before_re_and_preserves_rest():
    original = ('start\n'+GUARD_POINT+'frequency start\n'+FIXED_SKIP+
                'continuum_RE\nadiabatic\nweak_line_RE\n'+FINISH_POINT+'solver\n')
    changed = patched(original)
    assert changed == ('start\n'+GUARD_POINT+GUARD+'frequency start\n'+COMP_OPAC+
                       'continuum_RE\nadiabatic\nweak_line_RE\n'+FINISH+FINISH_POINT+'solver\n')
    assert changed.count('CALL COMP_OPAC') == 1
    assert 'EMHNUKT' not in changed  # COMP_OPAC 자체가 actual-nu 지수를 매번 평가한다.
    assert changed.index('weak_line_RE') < changed.index('CALL LUMINA_FIXEDJ_HEAT_FINISH')
    assert changed.index('CALL LUMINA_FIXEDJ_HEAT_FINISH') < changed.index(FINISH_POINT)
    with pytest.raises(ValueError):
        patched(changed)


@pytest.mark.parametrize('marker', [FIXED_SKIP, GUARD_POINT, FINISH_POINT])
def test_patch_rejects_missing_or_duplicate_insertions(marker):
    original = GUARD_POINT+FIXED_SKIP+FINISH_POINT
    for source in (original.replace(marker, ''), original+marker):
        with pytest.raises(ValueError):
            patched(source)


@pytest.fixture(scope='module')
def native(tmp_path_factory):
    root = tmp_path_factory.mktemp('fixedj_actual_heat')
    (root/'kind.f90').write_text('module set_kind_module\ninteger,parameter::ldp=kind(1d0)\nend module\n')
    (root/'driver.f90').write_text('''program direct_heat_test
use set_kind_module
use ieee_arithmetic
implicit none
real(ldp)::pops(4,2),r(2),v(2),sigma(2),t(2),ed(2),atoms(2),clump(2),heat(2),charge(2)
logical::fixedj,ehb,coherent,newmod
integer::ndietot
character(16)::gamma_mode
character(32)::mode
call get_command_argument(1,mode)
fixedj=.true.;ehb=.false.;coherent=.true.;newmod=.false.;ndietot=0;gamma_mode='LOCAL'
if(mode=='not_fixed')fixedj=.false.
if(mode=='ehb')ehb=.true.
if(mode=='noncoherent')coherent=.false.
if(mode=='newmod')newmod=.true.
if(mode=='die_transfer')ndietot=1
if(mode=='gamma_transfer')gamma_mode='RAD_TRANS'
call lumina_fixedj_heat_guard(fixedj,.false.,coherent,ehb,.false.,.false.,.false., &
                             .false.,.false.,'BLANK',newmod,ndietot,gamma_mode)
r=[3d0,1d0];v=[30d0,10d0];sigma=0d0;t=[1d0,2d0];ed=[5d0,7d0]
atoms=[2d0,3d0];clump=1d0;heat=[-3d0,4d0];charge=[1d-16,-2d-16]
pops(1,:)=2d0;pops(2,:)=3d0;pops(3,:)=ed;pops(4,:)=t
if(mode=='nan')heat(1)=ieee_value(0d0,ieee_quiet_nan)
if(mode=='bad_state')pops(4,1)=2d0
if(mode=='bad_radius')r(1)=0.5d0
call lumina_fixedj_heat_finish(pops,r,v,sigma,t,ed,atoms,clump,heat,charge,4,2,11,65,7)
open(unit=71,file='FORBIDDEN_POP_UPDATE',status='new')
write(71,*)'must never reach population update'
end program
''')
    exe = root/'synthetic'
    subprocess.run(['gfortran', '-O', '-ffree-line-length-none', str(root/'kind.f90'),
                    str(SCRIPTS/'cmfgen_fixedj_heat.f90'), str(root/'driver.f90'), '-o', str(exe)],
                   cwd=root, check=True, capture_output=True, timeout=30)

    def run(out, mode='normal', enabled=True):
        out.mkdir()
        env = dict(os.environ)
        env.pop('LUMINA_FIXEDJ_HEAT_ONLY', None)
        if enabled:
            env['LUMINA_FIXEDJ_HEAT_ONLY'] = '1'
        result = subprocess.run([str(exe), mode], cwd=out, env=env,
                                capture_output=True, timeout=10)
        assert not (out/'FORBIDDEN_POP_UPDATE').exists()
        return result
    return run


def test_native_serialization_has_signed_actual_row_exact_state_and_no_update(tmp_path, native):
    output = tmp_path/'capture'
    result = native(output)
    assert result.returncode == 96
    row = read_heat_row(output/'NATIVE_HEAT_ROW.bin')
    np.testing.assert_array_equal(row['steq_t_native'], [-3., 4.])
    np.testing.assert_array_equal(row['temperature_native'], [1., 2.])
    np.testing.assert_array_equal(row['pops'][-2], [5., 7.])
    assert row['frequencies'] == 11 and row['weak_lines'] == 7
    assert row['evaluation_counter'] == 65
    assert row['phase'] == 'BEFORE_POP_UPDATE'
    assert row['kind'] == 'INTEGRAL_RE_NOT_EHB'
    assert row['joint_convergence'] is False


@pytest.mark.parametrize('mode', ['not_fixed', 'ehb', 'noncoherent', 'nan', 'bad_state', 'bad_radius',
                                  'newmod', 'die_transfer', 'gamma_transfer'])
def test_native_rejects_other_operators_and_invalid_state(tmp_path, native, mode):
    output = tmp_path/mode
    assert native(output, mode).returncode == 94
    assert not (output/'NATIVE_HEAT_ROW.bin').exists()


def test_explicit_activation_required(tmp_path, native):
    output = tmp_path/'disabled'
    assert native(output, enabled=False).returncode == 94
    assert not (output/'NATIVE_HEAT_ROW.bin').exists()


def test_reader_rejects_truncation_extra_data_and_bad_header(tmp_path, native):
    output = tmp_path/'original'
    native(output)
    raw = (output/'NATIVE_HEAT_ROW.bin').read_bytes()
    for i, data in enumerate((b'bad', raw[:-1], raw+b'x', b'wrong header!!!!'+raw[16:])):
        path = tmp_path/f'bad-{i}.bin'
        path.write_bytes(data)
        with pytest.raises(ValueError):
            read_heat_row(path)


@pytest.mark.parametrize('index,value', [(0, 2), (1, 32), (2, 2), (3, 1), (4, 0),
                                       (5, 0), (6, 8), (7, -1)])
def test_reader_rejects_changed_schema_dimensions(tmp_path, native, index, value):
    output = tmp_path/'original'
    native(output)
    raw = bytearray((output/'NATIVE_HEAT_ROW.bin').read_bytes())
    struct.pack_into('<i', raw, 16+index*4, value)
    path = tmp_path/'bad.bin'
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        read_heat_row(path)
