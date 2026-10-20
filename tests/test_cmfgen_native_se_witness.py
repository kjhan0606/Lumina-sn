"""소형 native 구조 모형에서 매핑·관찰 비간섭·갱신 전 phase를 확인한다."""
from pathlib import Path
import os
import struct
import subprocess
import sys

import numpy as np
import pytest

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
from prepare_cmfgen_se_witness import patched,INSERTIONS,NATIVE_SOURCES,prepare


@pytest.fixture(scope='module')
def capture(tmp_path_factory):
    root=tmp_path_factory.mktemp('native_se_model')
    stub=root/'stub.f90'
    stub.write_text('''module set_kind_module
integer,parameter::ldp=kind(1.0d0)
end module
module steq_data_mod
use set_kind_module
type ion_data
logical::xzv_pres=.false.
integer::n_se=3
real(ldp),allocatable::steq(:,:)
integer,allocatable::eq_in_ba(:)
end type
type(ion_data)::se(3)
real(ldp)::steq_ed(2)=0
end module
module mod_cmfgen
use set_kind_module
integer,parameter::num_species=1
integer::species_beg_id(1)=[1],species_end_id(1)=[3]
real(ldp)::pop_species(2,1)=17
end module
''')
    driver=root/'driver.f90'
    driver.write_text('''program synthetic_observer
use set_kind_module
use steq_data_mod
use ieee_exceptions
implicit none
integer::id,k
real(ldp)::pops(5,2),saved(5,2),c(1,1),h(1),hs(1),ed
real(ldp)::raw_before(3,2,2)
logical::flags(size(ieee_all))
character(16)::mode
call get_command_argument(1,mode)
do id=1,2
se(id)%xzv_pres=.true.
allocate(se(id)%steq(3,2),se(id)%eq_in_ba(3))
end do
se(1)%eq_in_ba=[1,2,3];se(2)%eq_in_ba=[2,3,3]
se(1)%steq(1,:)=28;se(1)%steq(2,:)=-28;se(1)%steq(3,:)=3
se(2)%steq(1,:)=7;se(2)%steq(2,:)=-7;se(2)%steq(3,:)=-3
raw_before(:,:,1)=se(1)%steq;raw_before(:,:,2)=se(2)%steq
pops(:,1)=[3.0_ldp,11.0_ldp,13.0_ldp,7.0_ldp,1.0_ldp]
pops(:,2)=pops(:,1);saved=pops
call lumina_native_se_reset(5,2,3)
do k=1,2
do id=1,2
if (trim(mode)=='missing'.and.id==2) cycle
if (id==1) then
c=2;h=3;hs=5
else
c=0.5_ldp;h=11;hs=13
end if
ed=7
if (trim(mode)=='negative'.and.id==1) c=-1
if (trim(mode)=='underflow') then
c=1e-300_ldp;h=1e-300_ldp;hs=1e-300_ldp
end if
flags=.false.;call ieee_set_flag(ieee_all,flags)
call lumina_native_se_collect(id,1,2,k,c,h,hs,ed)
call ieee_get_flag(ieee_all,flags)
if (any(flags)) stop 21
if (trim(mode)=='repeat'.and.id==1) call lumina_native_se_collect(id,1,2,k,c,h,hs,ed)
end do
end do
call lumina_native_se_write(pops,5,2,3,65)
if (any(pops/=saved)) stop 22
if (any(se(1)%steq/=raw_before(:,:,1))) stop 23
if (any(se(2)%steq/=raw_before(:,:,2))) stop 24
end program
''')
    exe=root/'synthetic'
    subprocess.run(['gfortran','-O','-ffree-line-length-none',str(stub),
        str(SCRIPTS/'cmfgen_se_collision_bound.f90'),str(SCRIPTS/'cmfgen_native_se_witness.f90'),
        str(driver),'-o',str(exe)],cwd=root,check=True,capture_output=True,timeout=30)
    def run(output,mode='normal',enabled=True):
        output.mkdir()
        env=dict(os.environ)
        env.pop('LUMINA_NATIVE_SE_WITNESS',None)
        if enabled:env['LUMINA_NATIVE_SE_WITNESS']='1'
        subprocess.run([str(exe),mode],cwd=output,env=env,check=True,capture_output=True,timeout=10)
        path=output/'NATIVE_SE_BOUND_000065.bin'
        if not path.exists():return None
        data=path.read_bytes();assert data[:16].rstrip()==b'NATIVESEBOUNDV1'
        header=struct.unpack_from('<9i',data,16);offset=52
        schema,bits,nr,nd,nt,ni,counter,parent,ns=header
        assert (schema,bits,nr,nd,nt,ni,counter,parent,ns)==(1,64,3,2,5,3,65,64,1)
        def array(dtype,shape):
            nonlocal offset
            count=int(np.prod(shape));a=np.frombuffer(data,dtype=dtype,count=count,offset=offset).copy()
            offset+=a.nbytes;return a.reshape(shape,order='F')
        fields={'pops':array('<f8',(nt,nd))}
        for key in ('raw','lo','hi','inc','out'):fields[key]=array('<f8',(nr,nd))
        fields['raw_count']=array('<i4',(nr,))
        for key in ('raw_valid','coll_valid','coll_count'):fields[key]=array('<i4',(nr,nd))
        for key in ('nc','nc_lo','nc_hi','nc_target'):fields[key]=array('<f8',(ns,nd))
        fields['nc_valid']=array('<i4',(ns,nd));fields['charge_raw']=array('<f8',(nd,))
        assert offset==len(data)
        return fields
    return run


def test_multiple_ion_contributions_are_mapped_before_any_row_replacement(capture,tmp_path):
    x=capture(tmp_path/'normal')
    np.testing.assert_array_equal(x['raw'],[[28,28],[-21,-21],[-7,-7]])
    np.testing.assert_array_equal(x['raw_count'],[1,2,1])
    assert np.all(x['raw_valid']==1) and np.all(x['coll_valid']==1)
    np.testing.assert_allclose(x['inc'],[[70,70],[42+45.5,42+45.5],[38.5,38.5]],rtol=1e-14)
    np.testing.assert_allclose(x['out'],[[42,42],[70+38.5,70+38.5],[45.5,45.5]],rtol=1e-14)
    assert np.all(x['lo']<=x['raw']) and np.all(x['raw']<=x['hi'])
    assert np.all(x['nc']==0) and np.all(x['nc_target']==17)
    assert np.all(x['nc_lo']<=0) and np.all(x['nc_hi']>=0)


@pytest.mark.parametrize('mode',['missing','negative','repeat'])
def test_missing_or_invalid_calls_never_gain_a_valid_bound(capture,tmp_path,mode):
    x=capture(tmp_path/mode,mode)
    assert not np.all(x['coll_valid']==1)
    assert np.all(x['raw_valid']==1)
    np.testing.assert_array_equal(x['raw_count'],[1,2,1])


def test_underflow_preserves_native_flags_and_never_invents_a_denominator(capture,tmp_path):
    x=capture(tmp_path/'underflow','underflow')
    assert np.all(x['inc']==0) and np.all(x['out']==0)
    assert np.all(x['coll_valid']==1)


def test_default_disabled_observer_creates_no_output(capture,tmp_path):
    assert capture(tmp_path/'disabled',enabled=False) is None


@pytest.mark.parametrize('role',list(NATIVE_SOURCES))
def test_only_three_exact_observation_insertions_change_native_source(role):
    native=Path('/gpfs/kjhan/cmfgen_src/cur_cmf')
    original=(native/NATIVE_SOURCES[role]).read_text()
    changed=patched(original,role)
    assert changed.replace(INSERTIONS[role][1],'')==original
    with pytest.raises((ValueError,RuntimeError)):
        patched(original.replace(INSERTIONS[role][0],''),role)


def test_existing_private_build_is_preserved_before_inputs(tmp_path):
    with pytest.raises(ValueError,match='기존'):
        prepare(Path('not-read'),Path('not-read'),tmp_path)
