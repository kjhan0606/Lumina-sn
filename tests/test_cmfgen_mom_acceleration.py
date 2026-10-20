"""실제 cap 사건과 틀린 후보로 원식 보존·예산·거부 경로를 직접 시험한다."""
import json
import csv
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_cmfgen_mom_capture import capture, trace
from analyze_cmfgen_sweep_context import floop_state
from test_cmfgen_affine_sweep import BUILD, PARENT, native_link


@pytest.fixture(scope='module')
def candidate_fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp('gradient_candidate')
    source = root/'candidate.f90'
    source.write_text('''program candidate_test
use lumina_mom_acceleration
implicit none
integer,parameter::n=5
real(ldp) r(n),x(n),base(4,n),alpha(n),lower(n),diagonal(n),upper(n),d(n),slope(n),err
real(ldp) hu(n),hl(n),hs(n),eps(n),epsp(n),jp(n),hp(n),saved(n),tri(4,n)
integer branch(n),reason,i
logical ok
character(len=30) mode
call get_command_argument(1,mode)
r=[5._ldp,4._ldp,3._ldp,2._ldp,1._ldp];x=r*r+1; saved=x
base=0;base(2,:)=1;base(4,:)=x;alpha=0
hu=0;hl=0;hs=0;eps=0;epsp=0;jp=x;hp=0
call linmon_branch(n,r,x,lower,diagonal,upper,branch)
if(mode=='--proposal')then
 alpha=.02_ldp;alpha(1)=0;alpha(n)=0
 call derivchi(d,x,r,n,'LINMON')
 base(4,:)=x+alpha*r*d;tri=base
 call propose_gradient(n,r,tri,0*x,alpha,x,hu,hl,hs,eps,epsp,jp,hp,0._ldp,ok,slope)
 if(.not.ok.or.accel_proposals/=1.or.accel_accepted/=1)stop 20
 if(any(x/=saved))stop 21
 if(maxval(abs(slope-r*d/x))>1.e-12_ldp)stop 22
else
 if(mode=='--branch')branch(2)=branch(2)+1
 if(mode=='--negative')x(2)=-1
 if(mode=='--nonfinite')x(2)=ieee_value(x(2),ieee_quiet_nan)
 if(mode=='--wrong-equation')base(4,2)=1.2_ldp*base(4,2)
 if(mode=='--H')hu(1)=100
 call validate_candidate(n,r,base,alpha,x,branch,hu,hl,hs,eps,epsp,jp,hp,0._ldp,ok,slope,reason,err)
 select case(trim(mode))
 case('--branch'); if(ok.or.reason/=11)stop 11
 case('--negative','--nonfinite'); if(ok.or.reason/=10)stop 12
 case('--wrong-equation'); if(ok.or.reason/=12)stop 13
 case('--H'); if(ok.or.reason/=13)stop 14
 case default; if(.not.ok.or.reason/=0.or.err>1.e-15_ldp)stop 15
 end select
end if
end program
''')
    exe = root/'candidate';native_link(source,exe)
    return exe


@pytest.mark.parametrize('mode', ['--valid','--proposal','--branch','--negative','--nonfinite','--wrong-equation','--H'])
def test_candidate_is_only_a_verified_gradient_proposal(candidate_fixture, mode, tmp_path):
    subprocess.run([str(candidate_fixture),mode],cwd=tmp_path,check=True,capture_output=True,timeout=15)


def test_actual_cap_reaches_original_criterion_without_replacing_old_iterate(tmp_path):
    run=Path('/gpfs/kjhan/cmfgen_runs/toy06_19p48d_affine_sweep_428713')
    old_replay=Path('/gpfs/kjhan/cmfgen_runs/lumina_affine_5691_replay_20261019-lSVfbZ')
    result=subprocess.run([str(BUILD/'sweep_mom_replay.exe'),str(run/'MOM_CAPTURE.bin'),str(run/'AFFINE_INPUT.bin')],
        cwd=tmp_path,capture_output=True,timeout=30,
        env=dict(os.environ,LUMINA_MOM_CAPTURE_MODE='REPLAY_FIRST',LUMINA_WHOLE_MODE='1'))
    assert result.returncode==0,result.stderr
    old_bytes=(old_replay/'native_trace/MOM_TRACE.bin').read_bytes()
    new_bytes=(tmp_path/'MOM_TRACE.bin').read_bytes()
    # CALL 44bytes + 매회 TRI/RAW/ERR의 고정 길이: 제안 전90회 원래 산술은 불변이다.
    prefix=44+90*(44+64*90)
    assert new_bytes[:prefix]==old_bytes[:prefix]
    calls=trace(tmp_path/'MOM_TRACE.bin')[-1]['iterations']
    assert set(calls)==set(range(1,91))|{92,93}
    assert calls[93]['ERR0'][0]<1e-8
    # 후보 solve91은 예산에는 있지만 JOLD나 native 반환 J로 기록되지 않는다.
    np.testing.assert_array_equal(calls[92]['ERR0'][1][0],calls[90]['RAW0'])
    np.testing.assert_array_equal(calls[93]['ERR0'][1][0],calls[92]['RAW0'])
    inv=json.loads((PARENT/'capture_first_v1/state_inventory.json').read_text())
    state=capture(tmp_path/'SWEEP_REPLAY_RESULT.bin',inv)
    x=state['modules']['jrel']['GAM_RSQJNU_SAVE']['values']
    exact=np.array(json.loads((old_replay/'native-event-analysis.json').read_text())['root_checks'][0]['solution'])
    assert np.max(abs(x-exact)/exact)<1e-8
    summary=(tmp_path/'MOM_ACCEL_SUMMARY.txt').read_text().splitlines()[1].split()
    assert summary[:3]==['1','1','0']
    assert not (tmp_path/'MOM_FAULT.bin').exists()


def test_actual_cap_then_original_same_frequency_F_loop(tmp_path):
    run=Path('/gpfs/kjhan/cmfgen_runs/toy06_19p48d_affine_sweep_428713')
    exe=tmp_path/'event_floop'
    native_link(Path(__file__).resolve().parents[1]/'scripts/replay_cmfgen_event_floop.f90',exe)
    result=subprocess.run([str(exe),str(run)],cwd=tmp_path,capture_output=True,timeout=30,
        env=dict(os.environ,LUMINA_MOM_CAPTURE_MODE='CAPTURE_FIRST',LUMINA_WHOLE_MODE=''))
    assert result.returncode==0,result.stderr
    assert b'ORIGINAL_F_LOOP_TERMINATED_NOT_PHYSICS_PASS' in result.stdout
    assert (tmp_path/'replay_geometry.bin').read_bytes()==(run/'FORMAL_EVENT_geometry.bin').read_bytes()
    assert (tmp_path/'first_replay_rays_out.bin').read_bytes()==(run/'FORMAL_EVENT_rays_out.bin').read_bytes()
    control=floop_state(run/'FLOOP_STATE.bin')
    with (tmp_path/'FLOOP_REPLAY.csv').open() as f:rows=list(csv.DictReader(f))
    assert [int(row['jit']) for row in rows]==list(range(control['count'],control['count']+len(rows)))
    assert all(int(row['native_F_cap'])==0 for row in rows)
    assert all(float(row['F_change'])>control['tolerance'] for row in rows[:-1])
    assert float(rows[-1]['F_change'])<=control['tolerance']
    assert not (tmp_path/'MOM_FAULT.bin').exists()
