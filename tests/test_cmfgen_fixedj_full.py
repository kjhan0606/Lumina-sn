"""native full-level 관찰자의 행 순서와 공통 상태 직렬화를 직접 확인한다."""
import csv
from pathlib import Path
import struct
import subprocess

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'


@pytest.fixture(scope='module')
def full_capture(tmp_path_factory):
    root = tmp_path_factory.mktemp('fixedj_full_capture')
    (root/'set_kind_module.f90').write_text('''module set_kind_module
integer,parameter::ldp=kind(1d0)
end module
''')
    (root/'mod_cmfgen.f90').write_text('''module mod_cmfgen
use set_kind_module,only:ldp
implicit none
type model_atom_data
  logical::xzv_pres=.false.
  integer::nxzv_f=0,nxzv=0,eqxzv=0
  real(ldp),allocatable::xzv_f(:,:),xzvlte_f_on_s(:,:),w_xzv_f(:,:)
  real(ldp),allocatable::edgexzv_f(:),gxzv_f(:)
  integer,allocatable::f_to_s_xzv(:)
end type
integer::num_ions=1,species_lnk(1)=1
real(ldp)::at_no(1)=14d0
character(12)::ion_id(1)='SiIII'
type(model_atom_data)::atm(1)
end module
''')
    (root/'driver.f90').write_text('''program capture_test
use set_kind_module,only:ldp
use mod_cmfgen
implicit none
real(ldp)::pops(5,3),r(3)
character(32)::mode
call get_command_argument(1,mode)
atm(1)%xzv_pres=.true.
atm(1)%nxzv_f=3;atm(1)%nxzv=2;atm(1)%eqxzv=1
allocate(atm(1)%xzv_f(3,3),atm(1)%xzvlte_f_on_s(3,3),atm(1)%w_xzv_f(3,3))
allocate(atm(1)%f_to_s_xzv(3),atm(1)%edgexzv_f(3),atm(1)%gxzv_f(3))
atm(1)%xzv_f(1,:)=[11d0,12d0,13d0]
atm(1)%xzv_f(2,:)=[21d0,22d0,23d0]
atm(1)%xzv_f(3,:)=[31d0,32d0,33d0]
atm(1)%xzvlte_f_on_s=0.1d0
atm(1)%w_xzv_f=0.9d0
atm(1)%f_to_s_xzv=[1,1,2]
atm(1)%edgexzv_f=[1.5d0,1.0d0,0.5d0]
atm(1)%gxzv_f=[2d0,4d0,6d0]
if(mode=='negative')atm(1)%xzv_f(1,1)=-1d0
r=[3d0,2d0,1d0]
pops(1,:)=[1d0,2d0,3d0];pops(2,:)=[4d0,5d0,6d0]
pops(3,:)=[7d0,8d0,9d0];pops(4,:)=[10d0,11d0,12d0]
pops(5,:)=[13d0,14d0,15d0]
call lumina_fixedj_full_capture(pops,r,5,3,7)
end program
''')
    exe = root/'capture.exe'
    observer = SCRIPTS/'cmfgen_fixedj_full.f90'
    result = subprocess.run(['gfortran','-O','-ffree-line-length-none',
                             str(root/'set_kind_module.f90'),str(root/'mod_cmfgen.f90'),
                             str(observer),str(root/'driver.f90'),'-o',str(exe)],
                            cwd=root,text=True,capture_output=True,check=False)
    assert result.returncode == 0, result.stdout+result.stderr
    return root,exe


def read_matrix(path):
    raw=path.read_bytes()
    magic,nfull,nd=struct.unpack('<8sii',raw[:16])
    assert magic==b'LMATFL01' and (nfull,nd)==(3,3)
    assert len(raw)==16+8*nfull*nd
    return np.frombuffer(raw[16:],dtype='<f8').reshape(nfull,nd).copy()


def test_native_full_matrices_and_state_use_c_row_major_contract(tmp_path,full_capture):
    _,exe=full_capture
    result=subprocess.run([str(exe)],cwd=tmp_path,text=True,capture_output=True,check=False)
    assert result.returncode==0,result.stdout+result.stderr
    np.testing.assert_array_equal(read_matrix(tmp_path/'NATIVE_FULL_POP.bin'),
        [[11,12,13],[21,22,23],[31,32,33]])
    np.testing.assert_array_equal(read_matrix(tmp_path/'NATIVE_LTE_FRACTION.bin'),
        np.full((3,3),0.1))
    np.testing.assert_array_equal(read_matrix(tmp_path/'NATIVE_OCCUPATION.bin'),
        np.full((3,3),0.9))
    with (tmp_path/'NATIVE_FULL_INDEX.csv').open(newline='') as stream:
        rows=list(csv.DictReader(stream))
    assert [(int(row['Z']),int(row['full_level']),int(row['superlevel']),
             int(row['eq_sl']),row['ion_label']) for row in rows]==[
                 (14,1,1,1,'SiIII'),(14,2,1,1,'SiIII'),(14,3,2,2,'SiIII')]
    raw=(tmp_path/'NATIVE_FULL_INPUT.bin').read_bytes()
    assert raw[:16]==b'NATIVEHEATINPV1 '
    assert struct.unpack('<4i',raw[16:32])==(1,64,5,3)
    values=np.frombuffer(raw[32:],dtype='<f8')
    np.testing.assert_array_equal(values[:3],[3,2,1])
    np.testing.assert_array_equal(values[3:].reshape((5,3),order='F'),
        [[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]])
    phase=(tmp_path/'NATIVE_FULL_PHASE.txt').read_text()
    assert phase.startswith('AFTER_CURRENT_STATE_EVAL_LTE_BEFORE_RATE_AND_HEAT_EVALUATION')
    assert phase.splitlines()[1].split()==['7','5','3','3','1']


def test_native_full_capture_rejects_invalid_population_before_creating_outputs(tmp_path,full_capture):
    _,exe=full_capture
    result=subprocess.run([str(exe),'negative'],cwd=tmp_path,text=True,capture_output=True,check=False)
    assert result.returncode==94
    assert list(tmp_path.iterdir())==[]
