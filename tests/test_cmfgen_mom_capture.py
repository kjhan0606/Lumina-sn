from pathlib import Path
import json
import hashlib
import os
import re
import struct
import subprocess
import sys

import pytest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from prepare_cmfgen_mom_capture import (ARGUMENTS, FLAGS, diagnostic_module, fields_of,
                                       instrument_module, instrument_mom, instrument_first_mom,once,capture_target,
                                       rebuild_unmodified_sources,prepare)
from run_cmfgen_mom_capture import capture_header,verify_build
from analyze_cmfgen_mom_capture import capture,first_event,residual,summary,trace
from analyze_cmfgen_owner_matrices import multiprecision_stationary
from solve_cmfgen_mom_kernel import (linmon,linmon_matrix,first_repair_equation,tri_matrix,
    affine_boundary_from_half,affine_outer_system,affine_outer_primitive,affine_spatial_terms)

NATIVE=Path(os.environ.get("CMFGEN_PHOT_SMOOTH_SOURCE_ROOT","/gpfs/kjhan/cmfgen_src/cur_cmf"))


def test_affine_incident_boundary_and_independent_primitive_equation():
    plus=np.array([2.,.6,.4,.1]);minus=np.array([3.,-1.5,1.,-.6])
    now=affine_boundary_from_half(plus,minus)
    assert now["h"]*(plus[0]+minus[0])-now["H_incid"]==pytest.approx(sum([plus[1],minus[1]]))
    assert now["n"]*5-now["N_incid"]==pytest.approx(-.5)
    prev=affine_boundary_from_half(plus*.7,minus*.8)
    fields=dict(R=np.array([4.,3.,2.]),GAM_REL=np.ones(3)*1.1,GAM_REL_SQ=np.ones(3)*1.21,
                SIGMA=np.zeros(3),BETA=np.ones(3)*.1,DELTA=np.ones(3)*2.5,P_H=np.ones(3),
                VdHdR_TERM=np.array([.02,.03,.04]),Q=np.array([.1,.2,1.]),DTAU_H=np.array([.3,.4,0.]),W=np.ones(3)*2)
    k=np.array([.4,.5,.6]);kp=np.array([.3,.4,.5]);old_h,old_n=.2,.05
    g=1.21
    fields["PSI"]=np.array([2.5*(old_h+(g-1)*old_n+g*.1*k[0]),0.,0.])
    state=dict(args=dict(INCL_ADVEC_TERMS=True),modules=dict(jrel={n:dict(values=x) for n,x in fields.items()},
        ray=dict(K_ON_J=dict(values=k),K_ON_J_PREV=dict(values=kp),HBC=old_h)))
    M=np.diag([0.,2.,3.]);b=np.array([7.,8.,9.])
    M[0,0]=(k[0]+.02)*.1/.3+fields["PSI"][0]+old_h
    M[0,1]=-(k[1]+.03)*.2/.3
    x=np.array([1.2,.7,.5]);jp=np.array([.8,.6,.4])
    changed,rhs=affine_outer_system(state,M,b,now,prev,jp)
    assert (changed@x-rhs)[0]==pytest.approx(affine_outer_primitive(state,x,jp,now,prev),abs=1e-14)
    np.testing.assert_array_equal(changed[1:],M[1:]);np.testing.assert_array_equal(rhs[1:],b[1:])
    bad=rhs.copy();bad[0]-=4.**2*1.1*2.5*(prev["H_incid"]+(g-1)*prev["N_incid"])
    assert abs((changed@x-bad)[0]-affine_outer_primitive(state,x,jp,now,prev))>1
    spatial,rhs_spatial=affine_outer_system(state,M,b,now,prev,jp,spatial=True)
    primitive_spatial=affine_outer_primitive(state,x,jp,now,prev,spatial=True)
    roundoff=8*np.finfo(float).eps*(abs(spatial[0])@abs(x)+abs(rhs_spatial[0])+abs(primitive_spatial))
    assert abs((spatial@x-rhs_spatial)[0]-primitive_spatial)<=roundoff
    d,inc,df,fi=affine_spatial_terms(state,now)
    assert (spatial@x-rhs_spatial)[1]-(changed@x-rhs)[1]==pytest.approx(-df*x[0]+fi)
    assert rhs_spatial[1]-rhs[1]==pytest.approx(-fi)
    assert spatial[1,0]-changed[1,0]==pytest.approx(-df)
    assert inc/(1+fields["W"][0])==pytest.approx(fi)
    state["args"]["INCL_ADVEC_TERMS"]=False
    assert affine_spatial_terms(state,now)==(0.,0.,0.,0.)
    state["args"]["INCL_ADVEC_TERMS"]=True
    vacuum=affine_boundary_from_half(plus,np.zeros(4))
    assert vacuum["H_incid"]==0 and vacuum["N_incid"]==0
    with pytest.raises(ValueError,match="지지"):
        affine_boundary_from_half(np.zeros(4),minus)


@pytest.mark.parametrize("advection",[False,True])
def test_first_repair_equation_uses_pre_repair_slope_and_preserves_boundaries(advection):
    r=np.array([9.,7.,5.,3.,1.])
    slope=np.array([11.,2.,-3.,4.,-15.])
    alpha=np.array([0.,.2,.3,.4,0.]) if advection else np.zeros(5)
    fixed=np.array([[0.,-1.,-1.,-1.,-1.],[5.,5.,5.,5.,5.],[-.5,-.5,-.5,-.5,0.],[1.,2.,3.,4.,5.]])
    event=dict(ND=5,matrix=fixed.copy(),slope=slope)
    event["matrix"][1]+=alpha*slope
    fields={k:dict(values=v) for k,v in dict(R=r,DTAUONQ=np.arange(1.,6.),
        CON_DELTA=np.ones(5),CHI_J=np.full(5,10.)).items()}
    state=dict(args=dict(INNER_BND_METH="DIFFUSION",OUTER_BND_METH="HONJ",ND_SM=5,
                         R_SM=r,INCL_ADVEC_TERMS=advection),modules=dict(jrel=fields))
    before=event["matrix"].copy()
    base,rhs,adv=first_repair_equation(state,event)
    np.testing.assert_allclose(base,tri_matrix(fixed),rtol=0,atol=1e-15)
    np.testing.assert_array_equal(rhs,fixed[3])
    np.testing.assert_allclose(adv,alpha*r,rtol=0,atol=1e-15)
    np.testing.assert_array_equal(event["matrix"],before)
    x=np.array([1.,2.,4.,3.,2.])
    derivative=linmon_matrix(x,r)[0]
    np.testing.assert_allclose((base+adv[:,None]*derivative)@x,
                               base@x+adv*linmon(x,r),rtol=1e-14,atol=1e-14)


def test_first_repair_equation_rejects_unimplemented_boundary():
    state=dict(args=dict(INNER_BND_METH="HOLLOW",OUTER_BND_METH="HONJ"),
               modules=dict(jrel=dict(R=dict(values=np.array([3.,2.,1.])))))
    with pytest.raises(ValueError,match="경계/격자"):
        first_repair_equation(state,{})


def test_complete_native_module_declarations_and_rejection():
    j=(NATIVE/"plane/mom_jrel_v9.f").read_text()
    j=j[:re.search(r"(?im)^\s*END MODULE MOD_JREL_V9\s*$",j).end()]
    f=fields_of(j)
    assert len({v["name"].upper() for v in f})==len(f)
    names={v["name"].upper():v for v in f}
    assert names["FREQ_SAVE"]["type"] == "REAL(KIND=LDP)"
    assert names["GAM_RSQJNU_PREV"]["alloc"]
    assert names["MOM_ERR_ON_FREQ"]["fixed"]=="N_ERR_MAX"
    with pytest.raises(ValueError,match="미지원"):
        fields_of("MODULE BAD\nCHARACTER*8 UNHANDLED\nEND MODULE BAD\n")
    with pytest.raises(ValueError):
        once("x x","x","y")


def test_hooks_keep_native_solver_calls_and_failure_accuracy_limit():
    text=(NATIVE/"plane/mom_jrel_v9.f").read_text()
    split=re.search(r"(?im)^\s*END MODULE MOD_JREL_V9\s*$",text)
    native=text[split.end():]
    result=instrument_mom(native)
    for call in ("CALL THOMAS(TA,TB,TC,XM,ND,1)","IF(MAX_ER .LT. 1.0E-08_LDP)ACCURATE=.TRUE.",
                 "IF(ICOUNT .EQ.  100)THEN","XM(I)=GAM_REL(I)*R(I)*R(I)*JNU_STORE(I)"):
        assert result.count(call)==native.count(call)
    assert result.index("CALL LUMINA_JREL_TAKE") < result.index("C_KMS=1.0E-05")
    assert "MAX_ER >= 1.0E-08_LDP" in result and "STOP 93" in result
    assert result.index("'RAW0'") < result.index("RECORDED_ERROR=.FALSE.")


def test_entry_capture_stops_before_selected_native_call_and_requires_target(tmp_path):
    text=(NATIVE/"plane/mom_jrel_v9.f").read_text()
    split=re.search(r"(?im)^\s*END MODULE MOD_JREL_V9\s*$",text)
    result=instrument_mom(text[split.end():],entry_only=True)
    assert result.index("CALL LUMINA_TAKE_ARGS") < result.index("LUMINA_SNAPSHOT_IO('MOM_CAPTURE.bin'")
    assert result.index("LUMINA_SNAPSHOT_IO('MOM_CAPTURE.bin'") < result.index("STOP 96")
    assert result.index("STOP 96") < result.index("C_KMS=1.0E-05")
    with pytest.raises(ValueError,match="진입 포획"):
        prepare(tmp_path/'parent',tmp_path/'native',tmp_path/'out',entry_only=True)
    with pytest.raises(ValueError,match="진입 포획"):
        prepare(tmp_path/'parent',tmp_path/'native',tmp_path/'out',True,entry_only=True)


def test_flux_trace_brackets_native_limiter_and_requires_selected_replay(tmp_path):
    text=(NATIVE/"plane/mom_jrel_v9.f").read_text()
    split=re.search(r"(?im)^\s*END MODULE MOD_JREL_V9\s*$",text)
    result=instrument_mom(text[split.end():],trace_flux=True)
    assert result.index("'HRAW'") < result.index("IF(H_CHK_OPTION .EQ. 'AV_VAL')THEN")
    assert result.index("'HFIX'") > result.index("IF(H_CHK_OPTION .EQ. 'MAX_VAL')THEN")
    assert result.index("'HFIX'") < result.index("IF(.NOT. INCL_ADVEC_TERMS)THEN")
    assert "CLOSE(LUM_H_TRACE)" in result
    with pytest.raises(ValueError,match='함께 사용할 수 없음'):
        instrument_mom(text[split.end():],entry_only=True,trace_flux=True)
    with pytest.raises(ValueError,match='flux 재생'):
        prepare(tmp_path/'parent',tmp_path/'native',tmp_path/'out',trace_flux=True)
    with pytest.raises(ValueError,match='flux 재생'):
        prepare(tmp_path/'parent',tmp_path/'native',tmp_path/'out',entry_only=True,trace_flux=True,
                target_frequency=6.1,target_outer=78)


def compile_io_test(tmp_path,first_repair=False,**selection):
    files=[]
    inventories=[]
    for filename,module,prefix in (("mom_jrel_v9.f","MOD_JREL_V9","LUMINA_JREL"),
                                  ("mod_ray_mom_store.f","MOD_RAY_MOM_STORE","LUMINA_RAY")):
        text=(NATIVE/"plane"/filename).read_text()
        end=re.search(r"(?im)^\s*END MODULE "+module+r"\s*$",text).end()
        transformed,fields=instrument_module(text[:end]+"\n",module,prefix)
        path=tmp_path/(module.lower()+".f")
        path.write_text(transformed)
        inventories.append(fields)
        files.append(path)
    diag=tmp_path/"diag.f90"
    diag.write_text(diagnostic_module(first_repair,**selection))
    files.append(diag)
    code=["program io_test", "use mod_jrel_v9", "use mod_ray_mom_store", "use lumina_mom_diag", "implicit none",
          "type(lumina_arg_state) a,b", "type(lumina_jrel_state) js,j2", "type(lumina_ray_state) rs,r2",
          "character(len=4096) input_path,selection_arg", "integer selected_outer",
          "real(ldp) selected_frequency"]
    if first_repair:code += ["integer event_i","real(ldp) event_tri(4,5),event_raw(5),event_h(5),event_slope(5)"]
    code += ["call get_command_argument(1,input_path)",
          "if(input_path=='--selected' .or. input_path=='--before-pop')then",
          "call get_command_argument(2,selection_arg)", "if(len_trim(selection_arg)>0)then",
          "read(selection_arg,*)selected_outer", "call lumina_mom_set_iteration(selected_outer)", "end if",
          "if(input_path=='--selected')then", "call get_command_argument(3,selection_arg)",
          "read(selection_arg,*)selected_frequency", "write(*,*)merge(1,0,lumina_selected(selected_frequency))",
          "else", "call lumina_mom_stop_before_pop()", "end if", "stop", "end if",
          "if(input_path/=' ' .and. input_path/='--event')then", "call lumina_snapshot_io(trim(input_path),a,js,rs,.true.)", "stop", "end if"]
    for fields in inventories:
        for i,f in enumerate(fields):
            name=f["name"]
            # 하나의 미할당 배열과 음수 lower bound도 그대로 보존해야 한다.
            if name=="ESEC":
                continue
            if f["alloc"]:
                code.append(f"allocate({name}(-2:4))")
            val=".true." if f["type"]=="LOGICAL" else str(i+2)+(".125_ldp" if f["type"].startswith("REAL") else "")
            code.append(name+"="+val)
    code += ["a%nd_sm=5", "lumina_outer=60", "lumina_freq_index=1234", "lumina_jit=4"]
    for name,kind,array in ARGUMENTS:
        if name=="ND_SM":
            continue
        if array:
            code.append(f"allocate(a%{name}(5))")
        val="'saved  string '" if kind=="C" else ".false." if kind=="L" else "3.25_ldp"
        code.append(f"a%{name}={val}")
    code += ["call lumina_jrel_take(js)", "call lumina_ray_take(rs)",
             "call lumina_snapshot_io('first.bin',a,js,rs,.false.)",
             "call lumina_snapshot_io('first.bin',b,j2,r2,.true.)",
             "nd=-999", "freq_save=-999._ldp", "gam_rsqjnu_prev=0._ldp",
             "call lumina_jrel_restore(j2)", "call lumina_ray_restore(r2)",
             "if(allocated(esec))stop 11", "if(lbound(gam_rsqjnu_prev,1)/=-2)stop 12",
             "call lumina_jrel_take(js)", "call lumina_ray_take(rs)",
             "call lumina_snapshot_io('second.bin',b,js,rs,.false.)"]
    if first_repair:
        code += ["call lumina_jrel_empty(j2)","if(j2%nd/=0 .or. allocated(j2%chi))stop 13",
                 "call lumina_ray_raw_take(r2)","if(allocated(r2%k_on_j))stop 14",
                 "if(any(r2%jnu_store/=jnu_store))stop 15","if(any(r2%rmid_store(2:4)/=0))stop 16",
                 "if(input_path=='--event')then","call lumina_first_complete(a,js,rs)",
                 "call lumina_mom_set_phase(1235,0,.false.,.true.)","call lumina_first_begin(3._ldp)",
                 "a%freq=3._ldp","event_tri=reshape([(real(event_i,ldp),event_i=1,20)],[4,5])",
                 "event_raw=1._ldp;event_raw(2)=-1._ldp;event_h=0;event_slope=.1_ldp",
                 "call lumina_first_event(1,2,2,5,3._ldp,event_tri,event_raw,event_h,event_slope,.false.,a,js,rs,js,rs)","end if"]
    code += ["end program"]
    prog=tmp_path/"io_test.f90"
    prog.write_text("\n".join(code)+"\n")
    for file in files+[prog]:
        flags=FLAGS if file.suffix==".f" else ["-O","-ffree-line-length-none"]
        subprocess.run(["gfortran",*flags,"-I.","-I"+str(NATIVE/"mod"),"-c",str(file)],cwd=tmp_path,check=True,capture_output=True)
    exe=tmp_path/"io_test"
    subprocess.run(["gfortran","mod_jrel_v9.o","mod_ray_mom_store.o","diag.o","io_test.o","-o",str(exe)],
                   cwd=tmp_path,check=True,capture_output=True)
    (tmp_path/"inventory.json").write_text(json.dumps(dict(jrel=inventories[0],ray=inventories[1])))
    return exe


def test_real_native_state_and_arguments_roundtrip_compiled(tmp_path):
    exe=compile_io_test(tmp_path)
    subprocess.run([str(exe)],cwd=tmp_path,check=True,capture_output=True)
    original=(tmp_path/"first.bin").read_bytes()
    assert original == (tmp_path/"second.bin").read_bytes()
    assert original[:16].rstrip()==b"LUMINAMOMCAPV1"
    decoded=capture(tmp_path/"first.bin",json.loads((tmp_path/"inventory.json").read_text()))
    assert decoded["phase"][:3]==(60,1234,4)
    assert decoded["args"]["ND_SM"]==5 and decoded["args"]["FREQ"]==3.25
    assert decoded["modules"]["jrel"]["ESEC"] is None
    assert decoded["modules"]["jrel"]["GAM_RSQJNU_PREV"]["bounds"]==(-2,4)
    for suffix,bad in (("kind",original[:16]+struct.pack("=i",99)+original[20:]),
                       ("truncated",original[:-1]),("trailing",original+b"x")):
        path=tmp_path/(suffix+".bin")
        path.write_bytes(bad)
        completed=subprocess.run([str(exe),str(path)],cwd=tmp_path,capture_output=True)
        assert completed.returncode != 0
    env=os.environ.copy()
    env["LUMINA_MOM_CAPTURE_MODE"]="CAPTURE"
    result=subprocess.run([str(exe),"--before-pop"],cwd=tmp_path,env=env,capture_output=True)
    assert result.returncode==94 and b"LUMINA_NO_CAPTURE_BEFORE_POP" in result.stdout
    env.pop("LUMINA_MOM_CAPTURE_MODE")
    assert subprocess.run([str(exe),"--before-pop"],cwd=tmp_path,env=env,capture_output=True).returncode==0


def test_actual_capture_header_rejects_wrong_phase_kind_and_size(tmp_path):
    p=tmp_path/"capture.bin"
    good=b"LUMINAMOMCAPV1  "+struct.pack("=4i6i",8,64,32,32,60,1234,2,0,1,90)
    # 실제 stream header는 16 bytes의 blank-padded CHARACTER다.
    assert len(good)==56
    p.write_bytes(good)
    assert capture_header(p)["J_IT_COUNTER"]==2
    for offset,value in ((16,99),(32,61),(36,0),(44,1),(52,89)):
        bad=bytearray(good)
        struct.pack_into("=i",bad,offset,value)
        p.write_bytes(bad)
        with pytest.raises(ValueError):
            capture_header(p)
    p.write_bytes(good[:-1])
    with pytest.raises(ValueError):
        capture_header(p)


def test_default_generated_diagnostics_remain_byte_identical():
    expected = {
        False: '4bd4ce8342df4931db56fd9a24ba6b1760ebd5011c229049c9e6230210efc97d',
        True: '6d78375b4c42028007609af9da48319499ffd31d122f37a895397660ec679a81',
    }
    for first, digest in expected.items():
        assert hashlib.sha256(diagnostic_module(first).encode()).hexdigest() == digest


@pytest.mark.parametrize('options', [
    dict(target_frequency=6.1), dict(target_outer=78),
    dict(target_frequency=float('nan'), target_outer=78),
    dict(target_frequency=float('inf'), target_outer=78),
    dict(target_frequency=0., target_outer=78),
    dict(target_frequency=-1., target_outer=78),
    dict(target_frequency=True, target_outer=78),
    dict(target_frequency='6.1', target_outer=78),
    dict(target_frequency=6.1, target_outer=True),
    dict(target_frequency=6.1, target_outer=0),
    dict(target_frequency=6.1, target_outer=78.5),
    dict(target_frequency=6.1, target_outer=2147483648),
    dict(first_repair=True, target_frequency=6.1, target_outer=78),
])
def test_new_capture_selection_rejects_partial_or_invalid_scope(options):
    with pytest.raises(ValueError, match='선택 포획'):
        diagnostic_module(**options)


def test_retargeted_native_selector_and_stop_phase_compiled(tmp_path):
    exe = compile_io_test(tmp_path, target_frequency=6.10732968, target_outer=78)
    env = dict(os.environ, LUMINA_MOM_CAPTURE_MODE='CAPTURE')
    for outer, frequency, selected in (
        (78, 6.10732968, 1), (78, 6.107329684, 1), (78, 6.10732969, 0),
        (77, 6.10732968, 0), (79, 6.10732968, 0), (78, 6.40196372, 0),
    ):
        result = subprocess.run([str(exe), '--selected', str(outer), str(frequency)],
                                cwd=tmp_path, env=env, capture_output=True, text=True)
        assert result.returncode == 0 and int(result.stdout.strip()) == selected, result.stderr
    # 75~77은 진행할 수 있어야 하며78에서 포획이 없으면 인구 갱신 전 중단한다.
    for outer, code in ((75, 0), (77, 0), (78, 94), (79, 94)):
        result = subprocess.run([str(exe), '--before-pop', str(outer)],
                                cwd=tmp_path, env=env, capture_output=True)
        assert result.returncode == code
    env['LUMINA_MOM_CAPTURE_MODE'] = 'REPLAY'
    result = subprocess.run([str(exe), '--selected', '78', '6.10732968'],
                            cwd=tmp_path, env=env, capture_output=True)
    assert result.returncode == 0 and int(result.stdout.strip()) == 1
    env.pop('LUMINA_MOM_CAPTURE_MODE')
    result = subprocess.run([str(exe), '--selected', '78', '6.10732968'],
                            cwd=tmp_path, env=env, capture_output=True)
    assert result.returncode == 0 and int(result.stdout.strip()) == 0


def test_header_requires_explicit_new_outer_and_legacy_launcher_rejects_it(tmp_path):
    p = tmp_path/'capture.bin'
    p.write_bytes(b'LUMINAMOMCAPV1  '+struct.pack('=4i6i', 8, 64, 32, 32, 78, 1234, 2, 0, 1, 90))
    with pytest.raises(ValueError):
        capture_header(p)
    assert capture_header(p, expected_outer=78)['outer_iteration'] == 78
    for bad in (77, True, 0, 2147483648):
        with pytest.raises(ValueError):
            capture_header(p, expected_outer=bad)
    (tmp_path/'capture-build.json').write_text(json.dumps(dict(
        capture_target=capture_target(target_frequency=6.10732968, target_outer=78))))
    with pytest.raises(ValueError, match='60회 전용'):
        verify_build(tmp_path)


@pytest.mark.parametrize('with_explicit', [False, True])
def test_unmodified_parent_link_keeps_explicit_physics_objects(tmp_path, monkeypatch, with_explicit):
    parent=tmp_path/'parent';native=tmp_path/'native';out=tmp_path/'out'
    parent.mkdir();out.mkdir()
    for name in ('plane/mod_ray_mom_store.f','plane/mom_jrel_v9.f',
                 'new_main/mod_subs/comp_j_blank.f','new_main/cmfgen_sub.f','new_main/GET_J_CHI.INC'):
        path=native/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('source')
    (parent/'libsubs.a').write_bytes(b'archive')
    (parent/'cmfgen_parent_rebuilt.exe').write_bytes(b'parent binary')
    objects=[tmp_path/'decay.o',tmp_path/'lazy.o'] if with_explicit else []
    for path in objects:path.write_bytes(b'explicit object')
    commands=[]
    def fake_command(args,cwd):
        commands.append(args)
        if '-o' in args:
            (cwd/args[args.index('-o')+1]).write_bytes(b'parent binary')
    monkeypatch.setattr('prepare_cmfgen_mom_capture.command',fake_command)
    rebuild_unmodified_sources(parent,native,out,explicit_objects=objects)
    link=commands[-1];start=link.index('cmfgen_sub.o')+1
    assert link[start:start+len(objects)]==objects
    assert link[start+len(objects)]=='-L.'
    assert (parent/'libsubs.a').read_bytes()==b'archive'


@pytest.mark.parametrize('objects', [['same.o','same.o'], ['not-an-object.a']])
def test_capture_build_rejects_ambiguous_explicit_object_manifest_before_build(tmp_path,objects):
    parent=tmp_path/'parent';parent.mkdir()
    (parent/'parent-build-identity.json').write_text(json.dumps(dict(explicit_objects=objects)))
    out=tmp_path/'out'
    with pytest.raises(ValueError,match='명시 object'):
        prepare(parent,tmp_path/'native',out)
    assert not out.exists()


@pytest.mark.parametrize('bad', ['missing_hash', 'changed_object'])
def test_explicit_physics_object_identity_is_checked_before_compilation(tmp_path,bad):
    parent=tmp_path/'parent';native=tmp_path/'native';parent.mkdir()
    names=('new_main/cmfgen.o','new_main/cmfgen_sub.o','plane/mom_jrel_v9.f',
           'plane/mod_ray_mom_store.f','new_main/mod_subs/comp_j_blank.f',
           'tools/set_kind_module.f','new_main/cmfgen_sub.f','new_main/GET_J_CHI.INC',
           'lib/libmod.a','lib/libpgplt.a','lib/libmy_lapack.a','lib/libmy_blas.a')
    inputs=[parent/'libsubs.a',parent/'cmfgen_parent_rebuilt.exe']+[native/n for n in names]
    physical=tmp_path/'physical.o';inputs.append(physical)
    for path in inputs:
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'sealed input')
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    if bad=='missing_hash':hashes.pop(str(physical))
    else:physical.write_bytes(b'changed physics object')
    (parent/'parent-build-identity.json').write_text(json.dumps(dict(
        explicit_objects=[str(physical)],evidence_sha256=hashes)))
    out=tmp_path/'out'
    with pytest.raises(ValueError,match='신원 변경'):
        prepare(parent,native,out)
    assert not out.exists()


def test_first_event_native_io_previous_frequency_and_initial_placeholders(tmp_path):
    exe=compile_io_test(tmp_path,True)
    env=os.environ.copy();env["LUMINA_MOM_CAPTURE_MODE"]="CAPTURE_FIRST"
    result=subprocess.run([str(exe),"--event"],cwd=tmp_path,env=env,capture_output=True)
    assert result.returncode==95, result.stderr.decode()
    event=first_event(tmp_path/"MOM_FAULT.bin")
    assert event["phase"][:3]==(60,1235,0)
    assert (event["kind"],event["iteration"],event["bad_1based"])==(1,2,2)
    assert event["matrix"].ravel(order="F").tolist()==list(range(1,21))
    assert event["raw"].tolist()==[1,-1,1,1,1]
    inv=json.loads((tmp_path/"inventory.json").read_text())
    previous=capture(tmp_path/"MOM_PREVIOUS.bin",inv)
    assert previous["phase"][:3]==(60,1234,4) and previous["args"]["FREQ"]==3.25
    assert capture(tmp_path/"MOM_CAPTURE.bin",inv)["args"]["FREQ"]==3
    assert not (tmp_path/"MOM_TRACE.bin").exists()
    raw=(tmp_path/"MOM_FAULT.bin").read_bytes()
    for bad in (raw[:-1],raw+b"x"):
        path=tmp_path/"bad_event.bin";path.write_bytes(bad)
        with pytest.raises(ValueError):first_event(path)


def test_first_event_hooks_precede_repairs_and_do_not_trace_normal_calls():
    text=(NATIVE/"plane/mom_jrel_v9.f").read_text()
    native=text[re.search(r"(?im)^\s*END MODULE MOD_JREL_V9\s*$",text).end():]
    result=instrument_first_mom(native)
    assert result.index("CALL LUMINA_FIRST_EVENT(1,")<result.index("RECORDED_ERROR=.FALSE.")
    assert result.index("CALL LUMINA_FIRST_EVENT(2,")<result.index("IF(H_CHK_OPTION .EQ. 'AV_VAL')")
    assert result.index("CALL LUMINA_FIRST_EVENT(3,")<result.index("excessive iteration count")
    assert "LUM_TRACE_ACTIVE=LUM_MODE=='REPLAY_FIRST'" in result
    assert "LUM_INITIAL_PENDING=INIT .AND. .NOT.ALLOCATED(AV_SIGMA)" in result
    assert "IF(ALLOCATED(K_ON_J))STOP 89" in result
    assert result.count("CALL THOMAS(TA,TB,TC,XM,ND,1)")==1
    assert "ABS(FREQ-6.40196372" not in diagnostic_module(True)


def test_trace_reader_distinguishes_accurate_linear_solve_from_replacement(tmp_path):
    path=tmp_path/"trace.bin"
    A=np.array([[0.,-1.],[2.,2.],[-1.,0.],[-4.,5.]])
    raw=np.array([-1.,2.]);new=np.array([1.,2.]);slope=np.array([-.5,.5])
    data=b"CALL"+struct.pack("=5i2di",60,123,2,0,1,6.40196372,6.5,2)
    for i in range(1,101):
        data+=b"TRI0"+struct.pack("=2i",i,2)+A.tobytes()
        data+=b"RAW0"+struct.pack("=2i",i,2)+raw.tobytes()
        data+=b"ERR0"+struct.pack("=2id",i,2,.25)+np.r_[raw,new,slope].tobytes()
    path.write_bytes(data)
    calls=trace(path)
    result=summary(calls[0],"DIFFUSION")
    assert len(result)==100 and len(calls[0]["records"])==301
    assert result[-1]["negative_before_replacement"]==1
    assert result[-1]["replaced_rows"]==1
    assert result[-1]["linear_residual_before"]==0
    assert result[-1]["linear_residual_after_replacement"]==pytest.approx(.5)
    assert residual(A,raw)==0
    for bad in (data[:-1],data+b"BAD!",data[44:]):
        path.write_bytes(bad)
        with pytest.raises(ValueError):
            trace(path)


def test_independent_multiprecision_keeps_weak_positive_rates_and_rejects_bad_generator():
    result=multiprecision_stationary(np.array([[-2.,1.],[2.,-1.]]),100)
    assert [float(v) for v in result["population"]]==pytest.approx([1/3,2/3])
    weak=multiprecision_stationary(np.array([[-1e-30,2.],[1e-30,-2.]]),100)
    assert weak["positive"] and float(weak["population"][1])==pytest.approx(5e-31,rel=1e-12,abs=0)
    with pytest.raises(ValueError):
        multiprecision_stationary(np.array([[-2.,-1.],[2.,1.]]),100)


def test_same_linmon_branches_and_exact_linear_data():
    r=np.array([9.,7.,5.,3.,2.,1.,.25])
    assert linmon(3*r+2,r)==pytest.approx(np.full(7,3.))
    x=np.array([6.,2.,8.,4.,3.,7.,5.])
    D,_=linmon_matrix(x,r)
    assert D@x==pytest.approx(linmon(x,r),abs=1e-14)
    numerical=np.column_stack([(linmon(x+np.eye(7)[i]*1e-6,r)-linmon(x-np.eye(7)[i]*1e-6,r))/2e-6
                               for i in range(7)])
    assert D==pytest.approx(numerical,abs=1e-8)
    assert linmon(x,r)[1]==linmon(x,r)[2]==0
