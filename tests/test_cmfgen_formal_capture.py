import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from prepare_cmfgen_formal_capture import ARGS,OUTPUTS,instrument_formal,module_source,replay_source,split_solver
from analyze_cmfgen_formal_capture import formal_state,rays,geometry,angular_integral

NATIVE=Path("/gpfs/kjhan/cmfgen_src/cur_cmf")
BUILD=Path(os.environ.get("CMFGEN_FORMAL_TEST_BUILD","/gpfs/kjhan/cmfgen_runs/lumina_mom_capture_build_20261019-RbUeST/formal_v1"))
MOM=Path("/gpfs/kjhan/cmfgen_runs/lumina_mom_capture_build_20261019-RbUeST/capture_first_v1")


@pytest.fixture(scope="module")
def compiled_fixture(tmp_path_factory):
    tmp=tmp_path_factory.mktemp("formal_io")
    code=["program io_test","use lumina_formal_diag","use,intrinsic::ieee_arithmetic","implicit none",
          "type(lumina_formal_args) a,b","type(lumina_ray_state) s,s2","character(len=4096) input_path",
          "integer ip,i,j,n","call get_command_argument(1,input_path)",
          "if(input_path=='--before-pop')then","call lumina_formal_stop_before_pop()","stop","end if",
          "if(input_path/=' ')then","call formal_state_io(trim(input_path),a,s,.true.)","stop","end if"]
    for name,kind,dim in ARGS:
        if kind=="I":code.append("a%"+name+"="+dict(NC="1",NP="5",ND="4")[name]);continue
        if dim:code.append("allocate(a%"+name+"("+dict(ND="4",NP="5")[dim]+"))")
        value="'DIFFUSION'" if name=="INNER_BND_METH" else "'LOGMON'" if kind=="C" else ".false." if kind=="L" else "7.25_ldp"
        code.append("a%"+name+"="+value)
    actual=",".join("b%"+x[0] for x in ARGS)
    code += ["b=a","call formal_args_take(a,.false.,"+actual+")"]
    for name in OUTPUTS:code.append("if(any([a%"+name+"/=0._ldp]))stop 12")
    code += ["nd_store=4","hn_def_on_nodes=.true.","hbc=0._ldp","nbc=0._ldp",
             "hnu_at_ob=1._ldp","hnu_at_ib=2._ldp","nnu_at_ob=3._ldp","nnu_at_ib=4._ldp",
             "hnu_at_ob_prev=5._ldp","hnu_at_ib_prev=6._ldp","nnu_at_ob_prev=7._ldp","nnu_at_ib_prev=8._ldp",
             "nplus_ob=999._ldp","nmin_ob=888._ldp"]
    for name in "r_store rmid_store ext_rmid_store gam_rel_store jnu_store hnu_store knu_store nnu_store".split():
        code += ["allocate("+name+"("+("5" if name=="ext_rmid_store" else "4")+"))",name+"=4.25_ldp"]
    code += ["rmid_store(4)=999._ldp","call formal_ray_take(s)",
             "if(s%nplus_ob/=0.or.s%nmin_ob/=0.or.s%rmid_store(4)/=0)stop 13",
             "if(allocated(s%k_on_j))stop 14","lumina_outer=60",
             "call lumina_mom_set_phase(23731,0,.false.,.true.)",
             "call formal_state_io('state.bin',a,s,.false.)",
             "call formal_state_io('state.bin',b,s2,.true.)",
             "call formal_state_io('state2.bin',b,s2,.false.)",
             "do ip=1,a%np","n=ip+2","ray(ip)%nz=n"]
    for name in "i_p i_m i_p_prev i_m_prev i_p_save i_m_save r_ray s_p s_m mu_p mu_m b_p b_m".split():
        code += ["allocate(ray(ip)%"+name+"(n))","ray(ip)%"+name+"=real(ip,ldp)+0.125_ldp"]
    code += ["ray(ip)%i_p_prev=20._ldp+ip","ray(ip)%i_m_prev=30._ldp+ip",
             "ray(ip)%i_p_save=40._ldp+ip","ray(ip)%i_m_save=50._ldp+ip",
             "allocate(ray(ip)%lnk(a%nd))","ray(ip)%lnk=1","end do",
             "a%new_freq=.true.","call formal_rays_io('rays_new.bin',a,.false.,.false.)",
             "a%new_freq=.false.","call formal_rays_io('rays_same.bin',a,.false.,.false.)",
             "call formal_rays_io('rays_new.bin',a,.true.,.false.)",
             "do ip=1,a%np","if(any(ray(ip)%i_p_prev/=40._ldp+ip))stop 15",
             "if(any(ray(ip)%i_m_save/=50._ldp+ip))stop 16","end do",
             "nd_ext=6","nd_add=2","idmin=1","idmax=4","ray_points_inserted=.true."]
    for name in "r_ext v_ext vdop_vec_ext sigma_ext".split():code += ["allocate("+name+"(6))",name+"=2._ldp"]
    for name in "jqw_p hqw_p kqw_p nqw_p jqw_m hqw_m kqw_m nqw_m".split():
        code += ["allocate("+name+"(a%nd,a%np))",name+"=ieee_value(0._ldp,ieee_quiet_nan)",
                 "do i=1,a%nd",name+"(i,1:a%np+1-i)=.125_ldp","end do"]
    code += ["call formal_geometry('geometry.bin',a)","end program"]
    source=tmp/"io_test.f90";source.write_text("\n".join(code)+"\n")
    # 과거 봉인 object가 아니라 현재 producer를 격리 재빌드한다.
    # formal_v1의 구형 EOF 검사를 현재 소스 시험으로 오인하지 않는다.
    diag=tmp/"lumina_formal_diag.f90";diag.write_text(module_source()+"\n")
    flags=["gfortran","-O","-fopenmp","-ffree-line-length-none","-J"+str(tmp),
           "-I"+str(tmp),"-I"+str(BUILD),"-I"+str(MOM),"-I"+str(NATIVE/"mod")]
    subprocess.run(flags+["-c",str(diag),"-o",str(tmp/"lumina_formal_diag.o")],
                   cwd=tmp,check=True,capture_output=True,text=True,timeout=60)
    command=flags+[str(source),str(BUILD/"ext_rel_grid_v4.o"),str(tmp/"lumina_formal_diag.o"),
             str(MOM/"mod_jrel_v9.o"),str(MOM/"lumina_mom_diag.o"),"-L"+str(MOM),"-L"+str(NATIVE/"lib"),
             "-lsubs","-lpgplt","-lmod",str(NATIVE/"lib/libmy_lapack.a"),str(NATIVE/"lib/libmy_blas.a"),"-o",str(tmp/"io_test")]
    subprocess.run(command,cwd=tmp,check=True,capture_output=True,text=True,timeout=60)
    subprocess.run([str(tmp/"io_test")],cwd=tmp,check=True,capture_output=True,text=True,timeout=20)
    return tmp,json.loads((BUILD/"formal-build.json").read_text())["ray_inventory"]


def test_hooks_preserve_native_calls_and_limit_scope():
    original=(NATIVE/"plane/cmf_formal_rel_v4.f").read_text()
    ext,sub=instrument_formal(original)
    assert "MODULE EXT_REL_GRID_V4" in ext
    for text in ("CALL SOLVE_CMF_FORMAL_V2","IF(HBC .LT. 0.0_LDP)HBC=0.0_LDP","NU_ON_dNU=1.0_LDP/dLOG_NU"):
        assert sub.count(text)==original.count(text)
    assert sub.index("CALL FORMAL_STATE_IO")<sub.index("NU_ON_dNU=0.0_LDP")
    assert sub.index("STOP 96")>sub.index("FIRST_TIME=.FALSE.")
    assert "23731" in module_source() and "23732" in module_source()
    assert "CALL LUMINA_RAY_RESTORE(S)" in replay_source()


def test_split_keeps_physics_and_changes_only_inward_surface_start():
    source=(NATIVE/"plane/solve_cmf_formal_v2.f").read_text()
    changed=split_solver(source)
    for name in ("CALL REL_VARIABLES","CALL OPTDEPTH_V4","exp(-dtau_loc","source_prime"):
        assert changed.count(name)==source.count(name)
    assert "do iz=lum_inward_start+1" in changed
    assert "if(lum_inward_start<nzz)then" in changed
    assert "FORMAL_SPLIT_INCOMING" in replay_source(True)


def test_native_io_roundtrip_validity_and_effective_previous_rays(compiled_fixture):
    tmp,inventory=compiled_fixture
    assert (tmp/"state.bin").read_bytes()==(tmp/"state2.bin").read_bytes()
    state=formal_state(tmp/"state.bin",inventory)
    assert state["phase"]==(60,23731,0,0,1)
    assert state["ray"]["NPLUS_OB"]==0 and state["ray"]["RMID_STORE"]["values"][-1]==0
    assert state["ray"]["K_ON_J"] is None
    for name in OUTPUTS:assert np.all(np.asarray(state["args"][name])==0)
    new,same=rays(tmp/"rays_new.bin"),rays(tmp/"rays_same.bin")
    assert np.all(new["rays"][2]["I_P"]==43)
    assert np.all(same["rays"][2]["I_M"]==33)
    g=geometry(tmp/"geometry.bin")
    assert g["ND_EXT"]==6
    assert all(np.all(np.isfinite(w)) for w in g["weights"].values())


@pytest.mark.parametrize("change",["tag","truncated","trailing","trailing_short"])
def test_state_corruption_rejected_by_python_and_native(compiled_fixture,change):
    tmp,inventory=compiled_fixture;raw=(tmp/"state.bin").read_bytes()
    raw=b"X"+raw[1:] if change=="tag" else raw[:-30] if change=="truncated" else raw+b"X" if change=="trailing_short" else raw+b"extra padding 1234567890"
    bad=tmp/(change+".bin");bad.write_bytes(raw)
    with pytest.raises(ValueError):formal_state(bad,inventory)
    result=subprocess.run([str(tmp/"io_test"),str(bad)],cwd=tmp,capture_output=True,timeout=20)
    assert result.returncode!=0


def test_before_population_guard(compiled_fixture):
    tmp,_=compiled_fixture
    result=subprocess.run([str(tmp/"io_test"),"--before-pop"],cwd=tmp,
        env={**os.environ,"LUMINA_FORMAL_CAPTURE_MODE":"CAPTURE_FORMAL"},capture_output=True,timeout=20)
    assert result.returncode==94


def test_fresh_half_moments_from_actual_ray_weights():
    g=dict(NC=1,NP=3,ND=3,rays=[dict(LNK=np.ones(3,dtype=int)) for _ in range(3)],weights={})
    mu=np.array([.8,.4,.1])
    for order,letter in enumerate("JHKN"):
        g["weights"][letter+"QW_P"]=np.tile(mu**order/2,(3,1))
        g["weights"][letter+"QW_M"]=np.tile((-mu)**order/2,(3,1))
    rs=dict(NC=1,NP=3,ND=3,rays=[dict(I_P=np.array([i+1.]),I_M=np.array([i+2.])) for i in range(3)])
    plus,minus=angular_integral(g,rs)
    assert plus[0,0]==3 and minus[0,0]==4.5
    assert plus[1,2]==pytest.approx(.4) and minus[1,2]==pytest.approx(-.8)
    assert plus[3,1]==pytest.approx((.8**3+2*.4**3)/2)
