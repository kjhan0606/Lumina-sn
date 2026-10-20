from pathlib import Path
import csv
import os
import subprocess
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from prepare_cmfgen_affine_sweep import monitor_source,exact_replay_source
from run_cmfgen_affine_sweep import sweep_coverage

PARENT=Path("/gpfs/kjhan/cmfgen_runs/lumina_mom_capture_build_20261019-RbUeST")
BUILD=Path(os.environ.get("CMFGEN_SWEEP_TEST_BUILD",PARENT/"whole_observe_v2"))
NATIVE=Path("/gpfs/kjhan/cmfgen_src/cur_cmf")


def native_link(source,exe):
    common=[BUILD/(name+".o") for name in ("lumina_affine_boundary","lumina_mom_diag","lumina_formal_diag","lumina_whole_monitor")]
    if (BUILD/'cmfgen_mom_acceleration.o').exists():common.insert(0,BUILD/'cmfgen_mom_acceleration.o')
    subprocess.run(["gfortran","-O","-fopenmp","-ffree-line-length-none","-I"+str(BUILD),
        "-I"+str(PARENT/"formal_v2"),"-I"+str(PARENT/"capture_first_v1"),"-I"+str(NATIVE/"mod"),str(source),*map(str,common),
        str(PARENT/"formal_v2/ext_rel_grid_v4.o"),str(PARENT/"capture_first_v1/mod_jrel_v9.o"),
        "-L"+str(BUILD),"-L"+str(NATIVE/"lib"),"-lsubs","-lpgplt","-lmod",str(NATIVE/"lib/libmy_lapack.a"),
        str(NATIVE/"lib/libmy_blas.a"),"-o",str(exe)],check=True,capture_output=True,timeout=90)


@pytest.fixture(scope="module")
def monitor_fixture(tmp_path_factory):
    root=tmp_path_factory.mktemp("whole_monitor")
    source=root/"monitor_test.f90"
    source.write_text("""program monitor_test
use lumina_whole_monitor
implicit none
character(len=30) mode
real(ldp) s
call get_command_argument(1,mode)
wm_np=90
if(mode=='--coverage')then
 call wm_frequency_end(1,3)
 stop 10
else if(mode=='--empty')then
 call wm_finish()
 stop 11
else if(mode=='--complete')then
 wm_pending=.true.;wm_meta=[60,1,0,1,1];wm_row=0;wm_row(1)=1.e-290_ldp
 wm_iter=1
 call wm_frequency_end(1,1)
 call wm_finish()
 stop 12
else if(mode=='--f-cap')then
 call wm_note_f(21,.1_ldp,.001_ldp)
 if(wm_code/=7)stop 13
 stop
else if(mode=='--f-cap-with-PSD')then
 wm_code=3
 call wm_note_f(20,.1_ldp,.001_ldp)
 if(wm_code/=7.or..not.wm_should_stop(wm_code))stop 18
 stop
else if(mode=='--policy')then
 if(wm_should_stop(0))stop 19
 if(wm_should_stop(3).neqv.(.not.wm_nonfatal_psd))stop 20
 if(wm_should_stop(4).neqv.(.not.wm_nonfatal_psd))stop 21
 if(wm_should_stop(8).neqv.(.not.wm_nonfatal_psd))stop 22
 if(.not.all([wm_should_stop(1),wm_should_stop(2),wm_should_stop(5),wm_should_stop(6),wm_should_stop(7)]))stop 23
 stop
else if(mode=='--support-complete')then
 if(.not.wm_nonfatal_psd)stop 91
 wm_pending=.true.;wm_meta=[60,1,0,1,1];wm_row=0;wm_row(1)=1.e-290_ldp
 wm_iter=1;wm_code=3
 call wm_frequency_end(1,1)
 call wm_finish()
 stop 24
else if(mode=='--init-closure')then
 nd=3
 allocate(kmid_on_j(3),nmid_on_j(3))
 kmid_on_j=1._ldp/6;nmid_on_j=.05_ldp
 call wm_save_physical_closure()
 kmid_on_j=0;nmid_on_j=0
 if(any(wm_physical_kmid/=1._ldp/6).or.any(wm_physical_nmid/=.05_ldp))stop 17
 stop
end if
s=wm_support(1._ldp,.2_ldp,1._ldp/3,.1_ldp)
if(s<=0)stop 14
s=wm_support(1._ldp,-.851632_ldp,.272599_ldp,-.125253_ldp)
if(s>=-.4_ldp)stop 15
s=wm_support(1._ldp,1._ldp,1._ldp,1._ldp)
if(abs(s)>1.e-14_ldp)stop 16
end program
""")
    exe=root/"monitor_test";native_link(source,exe)
    return exe


def test_native_monitor_support_and_injected_unrealizable_moments(monitor_fixture,tmp_path):
    subprocess.run([str(monitor_fixture)],cwd=tmp_path,env=dict(os.environ,LUMINA_WHOLE_MODE="1"),check=True,capture_output=True)


@pytest.mark.parametrize("mode,code",[("--coverage",98),("--empty",98),("--complete",97),("--f-cap",0),("--init-closure",0),("--policy",0),("--f-cap-with-PSD",0)])
def test_native_monitor_coverage_pre_pop_and_f_cap(monitor_fixture,tmp_path,mode,code):
    result=subprocess.run([str(monitor_fixture),mode],cwd=tmp_path,env=dict(os.environ,LUMINA_WHOLE_MODE="1"),capture_output=True)
    assert result.returncode==code,result.stderr
    if mode=="--complete":
        coverage=sweep_coverage(tmp_path/"WHOLE_SWEEP.csv")
        assert coverage["count"]==1 and coverage["last"]["freq_native"]==pytest.approx(1e-290,rel=1e-7,abs=0)
        assert "NOT_CLOSURE_OR_PHYSICS_PASS" in (tmp_path/"WHOLE_COMPLETE.txt").read_text()


def test_observation_hooks_and_exact_replay_do_not_inject_formal_history():
    source=monitor_source();replay=exact_replay_source()
    assert "ES16.7E3" in source
    assert "GAM_RSQJNU_PREV=JP" not in replay
    assert "CALL LUMINA_JREL_RESTORE(J)" in replay
    assert "FILE_BYTES/=112" in replay
    assert "MAX(ABS(J),ABS(H),ABS(K),ABS(N))" in source


def test_PSD_flag_can_finish_diagnostic_without_physics_pass(monitor_fixture,tmp_path):
    import json
    if not json.loads((BUILD/'sweep-build.json').read_text()).get('nonfatal_psd'):pytest.skip('관측 계속 전용 빌드')
    result=subprocess.run([str(monitor_fixture),'--support-complete'],cwd=tmp_path,capture_output=True,
        env=dict(os.environ,LUMINA_WHOLE_MODE='1'))
    assert result.returncode==97,result.stderr
    coverage=sweep_coverage(tmp_path/'WHOLE_SWEEP.csv')
    assert coverage['flags']=={'3':1}
    assert 'NOT_CLOSURE_OR_PHYSICS_PASS' in (tmp_path/'WHOLE_COMPLETE.txt').read_text()
    row=(tmp_path/'WHOLE_SWEEP.csv').read_text().splitlines()[1]
    assert (len(row)+1)*197291+4096<64*1024**2


def test_actual_PSD_is_recorded_and_cannot_mask_equation_or_flux_failure(tmp_path):
    import json
    if not json.loads((BUILD/'sweep-build.json').read_text()).get('nonfatal_psd'):pytest.skip('관측 계속 전용 빌드')
    source=exact_replay_source().replace('INTEGER LU,IOS,META(5),FILE_BYTES',
        'INTEGER LU,IOS,META(5),FILE_BYTES,TEST_CODE,TEST_BAD,TEST_SLOT\nREAL(KIND=LDP) TEST_FLUX,TEST_DIAG')
    check="""TEST_FLUX=BETA(1)*GAM_REL(1)*R(1)**2*AFFINE_HINC*Q(1)/DTAU_H(1)/(P_H(1)+W(1))
IF(WM_CODE/=3)STOP 30
TEST_DIAG=WM_TRI(2,45)
WM_TRI(2,45)=TEST_DIAG*1.01_LDP
CALL WM_CHECK(A%FREQ,A%INIT,A%INCL_ADVEC_TERMS,1,WM_TRI,WM_X,WM_H,WM_SLOPE,TEST_FLUX,TEST_CODE,TEST_BAD)
IF(TEST_CODE/=5)STOP 31
WM_TRI(2,45)=TEST_DIAG
TEST_SLOT=MAXLOC(ABS(WM_H(1:ND-1)),1)
WM_H(TEST_SLOT)=WM_H(TEST_SLOT)*1.01_LDP
CALL WM_CHECK(A%FREQ,A%INIT,A%INCL_ADVEC_TERMS,1,WM_TRI,WM_X,WM_H,WM_SLOPE,TEST_FLUX,TEST_CODE,TEST_BAD)
IF(TEST_CODE/=6)STOP 32
"""
    source=source.replace('IF(WHOLE_ENABLED())CALL WM_CALL_SUMMARY()',check)
    code=tmp_path/'fatal_precedence.f90';code.write_text(source)
    exe=tmp_path/'fatal_precedence';native_link(code,exe)
    run=Path('/gpfs/kjhan/cmfgen_runs/toy06_19p48d_affine_sweep_428645')
    result=subprocess.run([str(exe),str(run/'MOM_CAPTURE.bin'),str(run/'AFFINE_INPUT.bin')],cwd=tmp_path,capture_output=True,timeout=30,
        env=dict(os.environ,LUMINA_MOM_CAPTURE_MODE='REPLAY_FIRST',LUMINA_WHOLE_MODE='1'))
    assert result.returncode==0,result.stderr


def test_existing_two_frequency_native_results_unchanged_with_observation(tmp_path):
    source=PARENT/"affine_spatial_v1/affine_mom_replay.f90"
    exe=tmp_path/"local_replay";native_link(source,exe)
    root=Path("/gpfs/kjhan/cmfgen_runs/lumina_formal_replay_20261019-j8PO9m/native_affine_spatial_v1")
    mom=Path("/gpfs/kjhan/cmfgen_runs/toy06_19p48d_mom_first_428588")
    for name,input_name in (("previous","MOM_PREVIOUS.bin"),("current","MOM_KERNEL.bin")):
        out=tmp_path/name;out.mkdir()
        result=subprocess.run([str(exe),str(mom/input_name),str(root/name/"boundary.txt")],cwd=out,
            env=dict(os.environ,LUMINA_MOM_CAPTURE_MODE="REPLAY_FIRST",LUMINA_WHOLE_MODE="1"),capture_output=True,timeout=30)
        assert result.returncode==0,result.stderr
        for filename in ("MOM_TRACE.bin","AFFINE_RESULT.bin"):
            assert (out/filename).read_bytes()==(root/name/filename).read_bytes()


def test_coverage_rejects_skipped_frequency(tmp_path):
    path=tmp_path/"bad.csv"
    path.write_text("outer,index,jit,mom_iterations,code\n60,2,0,1,0\n")
    with pytest.raises(ValueError,match="coverage"):sweep_coverage(path)


def test_actual_initial_frequency_is_not_rejected_using_disabled_coefficients(tmp_path):
    import json
    import numpy as np
    from analyze_cmfgen_mom_capture import capture,first_event
    run=Path('/gpfs/kjhan/cmfgen_runs/toy06_19p48d_affine_sweep_428641')
    inventory=json.loads((PARENT/'capture_first_v1/state_inventory.json').read_text())
    result=subprocess.run([str(BUILD/'sweep_mom_replay.exe'),str(run/'MOM_CAPTURE.bin'),str(run/'AFFINE_INPUT.bin')],
        cwd=tmp_path,env=dict(os.environ,LUMINA_MOM_CAPTURE_MODE='REPLAY_FIRST',LUMINA_WHOLE_MODE='1'),capture_output=True,timeout=30)
    assert result.returncode==0,result.stderr
    state=capture(tmp_path/'SWEEP_REPLAY_RESULT.bin',inventory)
    old=first_event(run/'MOM_FAULT.bin')
    np.testing.assert_array_equal(state['modules']['jrel']['GAM_RSQJNU_SAVE']['values'],old['raw'])
    np.testing.assert_array_equal(state['modules']['jrel']['GAM_RSQHNU_SAVE']['values'],old['hraw'])
    assert not (tmp_path/'MOM_FAULT.bin').exists()
    record=(tmp_path/'WHOLE_CALL_SUMMARY.txt').read_text().splitlines()[1]
    assert float(record[4*24:5*24])>0.12


def test_event_context_writes_actual_blue_rays_not_current_save(tmp_path):
    import json
    import numpy as np
    from analyze_cmfgen_sweep_context import floop_state
    if not json.loads((BUILD/'sweep-build.json').read_text()).get('capture_event'):pytest.skip('event 포획 전용 빌드')
    source=(PARENT/'formal_v2/formal_replay.f90').read_text().replace('USE LUMINA_FORMAL_DIAG\n','USE LUMINA_FORMAL_DIAG\nUSE LUMINA_WHOLE_MONITOR\n',1)
    source=source.replace("WRITE(6,'(A)')'LUMINA_FORMAL_REPLAY_COMPLETE'",
        "CALL WM_SAVE_LOOP(LUMINA_JIT,A%FEDD,1.0E-4_LDP,.TRUE.,.TRUE.,A%ETA,A%ESEC,A%ETA,A%ND)\nCALL WM_EVENT_HOOK()")
    code=tmp_path/'event_context.f90';code.write_text(source)
    exe=tmp_path/'event_context';native_link(code,exe)
    prefix=Path('/gpfs/kjhan/cmfgen_runs/toy06_19p48d_formal_capture_428619/FORMAL_23732_00')
    result=subprocess.run([str(exe),str(prefix)],cwd=tmp_path,capture_output=True,timeout=30,
        env=dict(os.environ,LUMINA_WHOLE_MODE='1',LUMINA_FORMAL_CAPTURE_MODE=''))
    assert result.returncode==0,result.stderr
    assert (tmp_path/'FORMAL_EVENT_in.bin').read_bytes()==Path(str(prefix)+'_in.bin').read_bytes()
    for suffix in ('_rays_in.bin','_rays_out.bin','_geometry.bin'):
        assert (tmp_path/('FORMAL_EVENT'+suffix)).read_bytes()==Path(str(prefix)+suffix).read_bytes()
    state=floop_state(tmp_path/'FLOOP_STATE.bin')
    assert state['phase']==(60,23732,0,0,1) and state['coherent_es']
    np.testing.assert_array_equal(state['RJ_ES'],np.zeros(state['ND']))
    bad=tmp_path/'bad_floop.bin';bad.write_bytes((tmp_path/'FLOOP_STATE.bin').read_bytes()+b'x')
    with pytest.raises(ValueError):floop_state(bad)


def test_actual_event_follows_original_f_loop_stop_without_hiding_PSD(tmp_path):
    import json
    import numpy as np
    from analyze_cmfgen_mom_capture import capture,first_event
    run=Path('/gpfs/kjhan/cmfgen_runs/toy06_19p48d_affine_sweep_428691')
    result=subprocess.run([str(PARENT/'event_v2/replay_event_floop.exe'),str(run)],cwd=tmp_path,capture_output=True,timeout=30,
        env=dict(os.environ,LUMINA_MOM_CAPTURE_MODE='CAPTURE_FIRST',LUMINA_WHOLE_MODE='',LUMINA_FORMAL_CAPTURE_MODE=''))
    assert result.returncode==0,result.stderr
    assert (tmp_path/'replay_geometry.bin').read_bytes()==(run/'FORMAL_EVENT_geometry.bin').read_bytes()
    assert (tmp_path/'first_replay_rays_out.bin').read_bytes()==(run/'FORMAL_EVENT_rays_out.bin').read_bytes()
    rows=list(csv.DictReader((tmp_path/'FLOOP_REPLAY.csv').open()))
    assert [int(r['jit']) for r in rows]==[0,1]
    assert float(rows[0]['F_change'])>1e-4 and float(rows[1]['F_change'])<1e-4
    assert all(int(r['native_F_cap'])==0 for r in rows)
    assert not (tmp_path/'JIT_02_MOM.bin').exists()
    inv=json.loads((PARENT/'capture_first_v1/state_inventory.json').read_text())
    state=capture(tmp_path/'JIT_00_MOM.bin',inv);event=first_event(run/'MOM_FAULT.bin')
    np.testing.assert_array_equal(state['modules']['jrel']['GAM_RSQJNU_SAVE']['values'],event['raw'])
    np.testing.assert_array_equal(state['modules']['jrel']['GAM_RSQHNU_SAVE']['values'],event['hraw'])
