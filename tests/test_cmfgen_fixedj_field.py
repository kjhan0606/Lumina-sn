"""현재 J의 native EDD 직렬화와 원 RD_CONT_J 보간의 직접시험. 수송 시험은 아니다."""
import os
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(SCRIPTS))
from prepare_cmfgen_fixedj_field import write_field


def header(nd=2):
    raw = bytearray(14*8*(nd+1))
    struct.pack_into('<i', raw, 8*(nd+1), 10)
    return raw


def test_every_frequency_and_depth_value_is_preserved_without_resampling(tmp_path):
    nu = np.array([1.e15, 2.e15, 4.e15])
    J = np.array([[1., 4., 8.], [3., 5., 10.]])
    target = tmp_path/'EDDFACTOR'
    record = write_field(target, header(), nu, J, block=2)
    raw = np.fromfile(target, dtype='<f8').reshape(17, 3)
    assert raw[4, 0] == 1.
    np.testing.assert_array_equal(raw[14:, :2].T, J[:, ::-1])
    np.testing.assert_array_equal(raw[14:, 2], nu[::-1]*1.e-15)
    assert struct.unpack('<3i', target.read_bytes()[48:60]) == (15, 3, 2)
    assert record['native_quadrature_equal'] is False
    with pytest.raises(FileExistsError):
        write_field(target, header(), nu, J)


@pytest.mark.parametrize('mode', ['descending', 'zero_nu', 'duplicate_nu', 'nan_nu',
                                  'shape', 'precision', 'negative_J', 'nan_J', 'header'])
def test_invalid_field_is_not_published_as_complete(tmp_path, mode):
    nu = np.array([1.e15, 2.e15, 4.e15]); J = np.array([[1., 4., 8.], [3., 5., 10.]])
    h = header()
    if mode == 'descending': nu = nu[::-1]
    elif mode == 'zero_nu': nu[0] = 0.
    elif mode == 'duplicate_nu': nu[1] = nu[0]
    elif mode == 'nan_nu': nu[1] = np.nan
    elif mode == 'shape': J = J[:, :2]
    elif mode == 'precision': J = J.astype('f4')
    elif mode == 'negative_J': J[0, 0] = -1.
    elif mode == 'nan_J': J[0, 0] = np.nan
    elif mode == 'header': h = b'wrong'
    target = tmp_path/'EDDFACTOR'
    with pytest.raises(ValueError):
        write_field(target, h, nu, J, block=1)
    if target.exists():
        raw = target.read_bytes()
        assert struct.unpack_from('<d', raw, 4*24)[0] == 0.


def test_original_native_reader_consumes_current_field_knots_and_linear_intervals(tmp_path):
    source_root = os.environ.get('CMFGEN_SOURCE_ROOT')
    if not source_root:
        pytest.skip('실제 native RD_CONT_J 소스 경로 미지정')
    source = Path(source_root)/'new_main/mod_subs/rd_cont_j.f'
    # 원래 RD_CONT_J를 수정하지 않는다. 아래 모듈은 관찰용 배열 저장소만 제공한다.
    (tmp_path/'state.f90').write_text('''module set_kind_module
integer,parameter::ldp=kind(1d0)
end module
module ang_qw_mod
use set_kind_module
integer::np_obs
real(ldp)::p(2)=[0d0,1d0],p_obs(2),rmax_obs,v_at_rmax,iplus(2)
end module
module mod_cmfgen
use set_kind_module
real(ldp)::r(2)=[3d0,1d0],v(2)=[30d0,10d0]
end module
module opac_mod
use set_kind_module
real(ldp)::zeta(2)=0d0,theta(2)=0d0,source(2)
end module
module radiation_mod
use set_kind_module
real(ldp)::rj(2),k_mom(2),rsqhnu(2),hbc_cmf(2)
end module
''')
    (tmp_path/'reader.f90').write_text('''program reader_check
use set_kind_module
use radiation_mod
implicit none
integer::access_f,i
real(ldp)::query(4)=[4d0,3d0,2d0,1d0]
open(unit=17,file='EDDFACTOR',access='direct',form='unformatted',recl=24,status='old',action='read')
open(unit=18,file='reader-output.bin',access='stream',form='unformatted',status='new')
access_f=15
do i=1,4
call rd_cont_j(query(i),i,i==1,.true.,.false.,6,17,access_f,2,2)
write(18)rj
end do
close(18)
end program
''')
    subprocess.run(['gfortran', '-O', '-ffixed-line-length-none',
                    str(tmp_path/'state.f90'), str(source), str(tmp_path/'reader.f90'),
                    '-o', str(tmp_path/'reader')], cwd=tmp_path, check=True,
                   capture_output=True, timeout=30)
    nu = np.array([1.e15, 2.e15, 4.e15]); J = np.array([[1., 4., 8.], [3., 5., 10.]])
    write_field(tmp_path/'EDDFACTOR', header(), nu, J)
    subprocess.run([str(tmp_path/'reader')], cwd=tmp_path, check=True,
                   capture_output=True, timeout=10)
    actual = np.fromfile(tmp_path/'reader-output.bin', dtype='<f8').reshape(4, 2)
    np.testing.assert_array_equal(actual, [[8., 10.], [6., 7.5], [4., 5.], [1., 3.]])
