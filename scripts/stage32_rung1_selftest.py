#!/usr/bin/env python3
"""Model-free Rung-1 controls, including required injected defects."""
from __future__ import annotations
import hashlib,json,os,shutil,subprocess,tempfile
from pathlib import Path
import numpy as np
from stage32_rung1_check import HEADER,ROW,R1Error,read_check

ITER=10;FIELD_GEN=37
def run(exe:Path,base:Path,extra=None):
    env=os.environ.copy()
    env.update({"LUMINA_CMFGEN_EPS_FLOOR":"0.2","LUMINA_CMFGEN_EPS_CAP":"0.7"})
    env.update(extra or {})
    return subprocess.run([str(exe),str(base)],env=env,text=True,capture_output=True)
def stamped(base:Path,it:int=ITER): return Path(f"{base}.iter{it:03d}")
def check(path:Path,*,expect_unscaled_energy=False):
    return read_check(path,expected_iteration=ITER,expected_field_generation=FIELD_GEN,
                      expect_unscaled_energy=expect_unscaled_energy)
def expect_fail(path:Path,needle:str,*,expect_unscaled_energy=False):
    try:
        check(path,expect_unscaled_energy=expect_unscaled_energy)
        raise RuntimeError(f"defect was accepted: {needle}")
    except R1Error as e:
        if needle not in str(e): raise
        return str(e)

def rehash(path:Path):
    mfpath=Path(str(path)+".manifest.json")
    mf=json.loads(mfpath.read_text())
    mf["sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
    mfpath.write_text(json.dumps(mf,indent=2,sort_keys=True)+"\n")

def main():
    root=Path(__file__).resolve().parents[1];exe=root/"selftest_stage32_rung1_writer"
    work=Path(tempfile.mkdtemp(prefix="stage32_r1_",dir="/tmp"));result={}
    try:
        good_base=work/"good.bin";p=run(exe,good_base)
        if p.returncode: raise RuntimeError(p.stderr)
        good=stamped(good_base)
        result["positive_eps_phys"]=check(good)
        result["eps_fixture_coverage"]=p.stderr.splitlines()[0]
        noeps_base=work/"noeps.bin";p=run(exe,noeps_base,{"S32_FIXTURE_EPS_PHYS":"0"})
        if p.returncode: raise RuntimeError(p.stderr)
        result["positive_eps_phys_zero"]=check(stamped(noeps_base),expect_unscaled_energy=True)
        undefined_base=work/"undefined.bin"
        p=run(exe,undefined_base,{"S32_SEED_CHI_TOT_ZERO":"1"})
        if p.returncode: raise RuntimeError(p.stderr)
        undefined=check(stamped(undefined_base))
        if undefined["rho_undefined_chi_tot_zero_rows"]<=0:
            raise RuntimeError("chi_tot==0 undefined fixture was not recorded")
        result["chi_tot_zero_recorded_undefined"]=undefined["rho_undefined_chi_tot_zero_rows"]
        try:
            read_check(good) # public reader must not supply a default generation
            raise RuntimeError("missing expected_iteration was accepted")
        except TypeError as e: result["mandatory_keyword_only_refused"]=str(e)

        cases=(("beta_defect","S32_SEED_BETA_DEFECT","KA-3.2.3 FAIL",False),
               ("disposition_acc_w_defect","S32_SEED_DISPOSITION_DEFECT","branch-site disposition",False),
               ("opacity_share_defect","S32_SEED_OPACITY_SHARE_DEFECT","line energy FAIL",True),
               ("thin_numerator_defect","S32_SEED_THIN_NUMERATOR_DEFECT","line energy FAIL",True),
               ("row_unscaled_defect","S32_SEED_ROW_UNSCALED_DEFECT","does not close",False),
               ("authoritative_unscaled_defect","S32_SEED_AUTHORITATIVE_UNSCALED_DEFECT","does not close",False))
        for name,envname,needle,noeps in cases:
            env={envname:"1"}
            if noeps: env["S32_FIXTURE_EPS_PHYS"]="0"
            base=work/f"{name}.bin";p=run(exe,base,env)
            if p.returncode: raise RuntimeError(f"{name} writer failed before checker: {p.stderr}")
            result[name+"_negative_control"]=expect_fail(
                stamped(base),needle,expect_unscaled_energy=noeps)

        olddef=work/"old_definition.bin.iter010"
        shutil.copy(good,olddef)
        shutil.copy(Path(str(good)+".manifest.json"),Path(str(olddef)+".manifest.json"))
        raw=bytearray(olddef.read_bytes())
        nrows=HEADER.unpack_from(raw)[10]
        rows=np.frombuffer(raw,dtype=ROW,count=nrows,offset=HEADER.size)
        rows["rho_local"]=(1.0-rows["eps0_raw"])*(1.0-rows["beta"])
        olddef.write_bytes(raw);rehash(olddef)
        result["old_sobolev_definition_negative_control"]=expect_fail(
            olddef,"rho_local production-array identity failure")

        gen_base=work/"generation_defect.bin"
        p=run(exe,gen_base,{"S32_SEED_LAMBDA_GENERATION_DEFECT":"1"})
        if p.returncode==0 or "chi/lambda generation mismatch" not in p.stderr:
            raise RuntimeError("cross-generation defect was not refused")
        result["cross_generation_negative_control"]=p.stderr.strip()

        both_base=work/"both_unscaled.bin"
        p=run(exe,both_base,{"S32_SEED_BOTH_UNSCALED_DEFECT":"1"})
        if p.returncode: raise RuntimeError(p.stderr)
        both=check(stamped(both_base))
        result["both_sides_unscaled_control"]="NOT DETECTED: closure closes when both owners omit epsilon"
        result["fixture_preregistered_v2_readout"]={
            "rho_energy_weighted_median":result["positive_eps_phys"]["rho_energy_weighted_median"],
            "geometric_amplification_1_over_1_minus_rho":result["positive_eps_phys"]["geometric_amplification_1_over_1_minus_rho"],
            "prediction_1_status":result["positive_eps_phys"]["prediction_1_status"],
            "prediction_2_status":result["positive_eps_phys"]["prediction_2_status"]}

        g11_base=work/"g11.bin";p=run(exe,g11_base,{"S32_FIXTURE_ITER":"11"})
        if p.returncode: raise RuntimeError(p.stderr)
        try:
            read_check(stamped(g11_base,11),expected_iteration=ITER,
                       expected_field_generation=FIELD_GEN)
            raise RuntimeError("iteration mismatch accepted")
        except R1Error as e: result["iteration_mismatch_refused"]=str(e)
        try:
            read_check(good,expected_iteration=ITER,expected_field_generation=FIELD_GEN+1)
            raise RuntimeError("independent field-generation mismatch accepted")
        except R1Error as e: result["field_generation_mismatch_refused"]=str(e)

        tamper=work/"tamper.bin.iter010";shutil.copy(good,tamper)
        shutil.copy(Path(str(good)+".manifest.json"),Path(str(tamper)+".manifest.json"))
        raw=bytearray(tamper.read_bytes());raw[-9]^=1;tamper.write_bytes(raw)
        try:
            read_check(tamper,expected_iteration=ITER,expected_field_generation=FIELD_GEN)
            raise RuntimeError("tamper accepted")
        except R1Error as e: result["payload_tamper_refused"]=str(e)
        p=run(exe,good_base)
        if p.returncode==0: raise RuntimeError("generation overwrite accepted")
        result["generation_overwrite_refused"]=p.stderr.strip()
        result["verdict"]="PASS";print(json.dumps(result,indent=2,sort_keys=True));return 0
    except Exception as e:
        result["verdict"]=f"FAIL: {e}";print(json.dumps(result,indent=2,sort_keys=True));return 1
    finally: shutil.rmtree(work,ignore_errors=True)
if __name__=="__main__": raise SystemExit(main())
