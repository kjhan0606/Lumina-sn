"""물리 제조해와 별개로 전주파수 소비·직렬화·부분 출력 거부만 시험한다."""
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from test_cmfgen_formal_primary import native_link
from run_cmfgen_formal_primary import fields, coverage, validate_edd, read_edd


@pytest.fixture(scope='module')
def whole_fixture(tmp_path_factory):
    root=tmp_path_factory.mktemp('primary_whole_unit')
    source=root/'whole.f90'
    source.write_text('''program unit_whole
use lumina_formal_primary_whole
implicit none
integer i
character(len=50) mode
real(ldp) r(3),v(3),j(3),h(3),k(3),r2h(3)
call get_command_argument(1,mode)
nd_store=3
allocate(r_store(3),jnu_store(3),hnu_store(3),knu_store(3),nnu_store(3),gam_rel_store(3))
r=[3._ldp,2._ldp,1._ldp];v=0;j=1;h=.1_ldp;k=1._ldp/3;r2h=r*r*h
r_store=r;jnu_store=j;hnu_store=h;knu_store=k;nnu_store=.05_ldp;gam_rel_store=1
fp_source_j=j;fp_eta_true=j;fp_scattering=9*j
fp_ready=.true.;fp_in_call=.false.;fp_calls=2
fp_j_residual=0;fp_source_residual=0;fp_h_residual=0;fp_f_change=0
pw_input_ready=.true.;pw_input%chi=10*j;pw_input%sigma=0*j
lumina_outer=60
if(mode=='wrong-h')h=0
do i=1,2
 lumina_freq_index=i;fp_freq=3-i;pw_input%freq=fp_freq
 if(mode=='gap'.and.i==1)lumina_freq_index=2
 if(mode=='source')fp_j_residual=1
 call pw_frequency_end(i,2,fp_freq,1._ldp,r,v,j,h,k,r2h,3)
end do
call pw_finish()
end program
''')
    exe=root/'whole';native_link(source,exe)
    return exe


def execute(exe,out,mode=''):
    out.mkdir()
    return subprocess.run([str(exe),mode],cwd=out,capture_output=True,timeout=10,
        env=dict(os.environ,LUMINA_FORMAL_PRIMARY='1',LUMINA_FORMAL_PRIMARY_WHOLE='1',LUMINA_WHOLE_MODE=''))


def test_complete_field_has_exact_consumed_moments_and_partial_is_rejected(whole_fixture,tmp_path):
    out=tmp_path/'good';p=execute(whole_fixture,out)
    assert p.returncode==97,p.stderr
    assert coverage(out/'FORMAL_PRIMARY_SWEEP.csv')['count']==2
    geometry,a=fields(out/'FORMAL_PRIMARY_FIELDS.bin')
    np.testing.assert_array_equal(geometry[0],[3,2,1])
    np.testing.assert_array_equal(a['freq'],[2,1])
    np.testing.assert_array_equal(a['field'][:,0,:],np.ones((2,3)))
    np.testing.assert_array_equal(a['field'][:,1,:],np.full((2,3),.1))
    assert 'NOT_CLOSURE_OR_PHYSICS_PASS' in (out/'FORMAL_PRIMARY_COMPLETE.txt').read_text()
    partial=tmp_path/'partial.bin';partial.write_bytes((out/'FORMAL_PRIMARY_FIELDS.bin').read_bytes()[:-8])
    with pytest.raises(ValueError,match='부분 출력'):
        fields(partial)


@pytest.mark.parametrize('mode,reason',[('wrong-h','H/K/flux consumer'),('gap','coverage mismatch'),('source','unconverged source')])
def test_wrong_consumer_history_or_source_cannot_finish(whole_fixture,tmp_path,mode,reason):
    out=tmp_path/mode;p=execute(whole_fixture,out,mode)
    assert p.returncode==95,p.stderr
    assert reason in (out/'FORMAL_PRIMARY_FAILURE.txt').read_text()
    raw=(out/'FORMAL_PRIMARY_SOURCE.bin').read_bytes()
    assert raw[:16].rstrip()==b'FPRIMARYSOURCE1'
    np.testing.assert_array_equal(np.frombuffer(raw,dtype='<f8',offset=36),np.array([1.]*6+[9.]*3))
    assert not (out/'FORMAL_PRIMARY_COMPLETE.txt').exists()


def test_edd_reader_hz_and_field_stream_identity(whole_fixture,tmp_path,monkeypatch):
    out=tmp_path/'edd';assert execute(whole_fixture,out).returncode==97
    _,a=fields(out/'FORMAL_PRIMARY_FIELDS.bin')
    import run_cmfgen_formal_primary as runner
    monkeypatch.setattr(runner,'read_edd',lambda _: (np.ones((2,3)),np.array([1e15,2e15]),3,0))
    validate_edd(out,a)
    monkeypatch.setattr(runner,'read_edd',lambda _: (np.zeros((2,3)),np.array([1e15,2e15]),3,0))
    with pytest.raises(ValueError,match='producer'):
        validate_edd(out,a)


def test_actual_direct_access_edd_reader_does_not_guess_frequency_units(tmp_path):
    (tmp_path/'EDDFACTOR_INFO').write_text('format\nformat\n3 32 8 1 4 T\n')
    a=np.zeros((16,4));a[14]=[2,3,4,2];a[15]=[5,6,7,1]
    (tmp_path/'EDDFACTOR').write_bytes(a.astype('<f8').tobytes())
    j,nu,nd,_=read_edd(tmp_path)
    np.testing.assert_array_equal(j,[[5,6,7],[2,3,4]])
    np.testing.assert_array_equal(nu,[1e15,2e15])
    assert nd==3
