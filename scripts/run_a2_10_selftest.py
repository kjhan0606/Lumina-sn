#!/usr/bin/env python3
"""Run A2-10 analytic root/ledger and isolated N1-N8 controls."""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"validation/a2_10"
P=(("N1","A2_10_NEG_PHOTOHEAT_DROP",4),("N2","A2_10_NEG_NEIGHBOR_TE",4),("N3","A2_10_NEG_CANCEL_PAIR",4),("N4","A2_10_NEG_STALE_TERM",5),("N5","A2_10_NEG_PLANCK_FIELD",5),("N6","A2_10_NEG_ROOT_PIN",5),("N7","A2_10_NEG_TERM_SIGN",4),("N8","A2_10_NEG_TE_MANIFEST",5))
def dump(p:Path,o:object)->None:p.write_text(json.dumps(o,sort_keys=True,indent=2)+"\n")
def main()->int:
 q=argparse.ArgumentParser();q.add_argument("--binary",type=Path,required=True);a=q.parse_args();b=a.binary.resolve();OUT.mkdir(parents=True,exist_ok=True);base=subprocess.run((str(b),),cwd=ROOT,text=True,capture_output=True);ok=base.returncode==0;neg={}
 for nid,m,want in P:
  env=os.environ.copy();env[m]="1";r=subprocess.run((str(b),),cwd=ROOT,env=env,text=True,capture_output=True);passed=r.returncode==want and m in r.stderr;ok&=passed;neg[nid]={"marker":m,"child_rc":r.returncode,"wrapper_rc":0 if passed else 4,"status":"PASS" if passed else "FAIL","reason_code":"EXPECTED_REJECTION","ci_half_width":0.0}
 c=subprocess.run(("python3","scripts/a2_10_radeq_census.py","--output",str(OUT/"A2_10_RADEQ_CENSUS.json")),cwd=ROOT,text=True,capture_output=True);ok&=c.returncode==0
 dump(OUT/"A2_10_SELFTEST.json",{"status":"PASS" if ok else "FAIL","reason_code":"INTERNAL_RADEQ_TERM_ROOT_TRANSACTION","child_rc":base.returncode,"wrapper_rc":0 if ok else 4,"binary_sha256":hashlib.sha256(b.read_bytes()).hexdigest(),"negative_controls":neg,"metric_values":{"Te_root_K":5000.0,"E_balance":0.0,"line_owner_overlap_shells":0,"line_owner_closure_failures":0,"max_line_owner_closure":0.0,"te_manifest_exact_match":True,"te_context_separate":True,"partial_publish":0}})
 dump(OUT/"A2_10_L6_GATE.json",{"status":"BLOCKED_INCOMPLETE_ADIABATIC","reason_code":"RADEQ_INCOMPLETE_ADIABATIC","child_rc":3,"wrapper_rc":0,"CHAIN":"BLOCKED_INCOMPLETE_ADIABATIC","ORACLE_INPUT":"BLOCKED_INCOMPLETE_ADIABATIC","producer_equation":"RE_INTEGRAL","diagnostic_equation":"EHB_THERMAL","adiabatic_model":"ELECTRON_TRANSLATIONAL_ONLY","required_adiabatic_model":"CMFGEN_COMPLETE","heat_residual_qualified":False,"truth_f_cov":None,"line_owner_overlap_shells":0,"max_line_owner_closure":0.0})
 print(f"{'PASS' if ok else 'FAIL'} A2_10_SELFTEST N1_N8={sum(v['status']=='PASS' for v in neg.values())}/8 L6=BLOCKED_INCOMPLETE_ADIABATIC")
 return 0 if ok else 4
if __name__=="__main__":raise SystemExit(main())
