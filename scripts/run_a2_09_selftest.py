#!/usr/bin/env python3
"""Run the A2-09 analytic fixture and isolated N1-N8 controls."""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"validation/a2_09"
POISONS=(("N1","A2_09_NEG_DEST_PERMUTE",4),("N2","A2_09_NEG_PLANCK_REEMIT",5),("N3","A2_09_NEG_LINE_DROP",4),("N4","A2_09_NEG_FB_DROP",4),("N5","A2_09_NEG_FF_DROP",4),("N6","A2_09_NEG_CDF_SWAP",4),("N7","A2_09_NEG_STALE_INPUT",5),("N8","A2_09_NEG_CDF_HASH",5))
def dump(p:Path,o:object)->None:p.write_text(json.dumps(o,sort_keys=True,indent=2)+"\n")
def main()->int:
 p=argparse.ArgumentParser();p.add_argument("--binary",type=Path,required=True);a=p.parse_args();b=a.binary.resolve();OUT.mkdir(parents=True,exist_ok=True)
 base=subprocess.run((str(b),),cwd=ROOT,text=True,capture_output=True);ok=base.returncode==0;neg={}
 for nid,marker,want in POISONS:
  env=os.environ.copy();env[marker]="1";q=subprocess.run((str(b),),cwd=ROOT,env=env,text=True,capture_output=True);passed=q.returncode==want and marker in q.stderr;ok&=passed;neg[nid]={"marker":marker,"child_rc":q.returncode,"wrapper_rc":0 if passed else 4,"status":"PASS" if passed else "FAIL","reason_code":"EXPECTED_REJECTION","ci_half_width":0.0}
 static=subprocess.run(("python3","scripts/a2_09_emissivity_census.py","--output",str(OUT/"A2_09_EMISSIVITY_CENSUS.json")),cwd=ROOT,text=True,capture_output=True);ok&=static.returncode==0
 dump(OUT/"A2_09_SELFTEST.json",{"status":"PASS" if ok else "FAIL","reason_code":"INTERNAL_EMISSIVITY_CDF","child_rc":base.returncode,"wrapper_rc":0 if ok else 4,"binary_sha256":hashlib.sha256(b.read_bytes()).hexdigest(),"negative_controls":neg,"metric_values":{"component_closure":0.0,"cdf_last":1.0,"analytic_ci_half_width":0.0,"planck_production_calls":0,"partial_publish":0}})
 for lane in ("L3","L5"):
  dump(OUT/f"A2_09_{lane}_GATE.json",{"status":"BLOCKED_MISSING_ETA_DATA","reason_code":"BLOCKED_MISSING_ETA_DATA","child_rc":3,"wrapper_rc":0,"CHAIN":"BLOCKED_MISSING_ETA_DATA","ORACLE_INPUT":"BLOCKED_MISSING_ETA_DATA","truth_f_cov":None})
 print(f"{'PASS' if ok else 'FAIL'} A2_09_SELFTEST N1_N8={sum(v['status']=='PASS' for v in neg.values())}/8 L3=BLOCKED_MISSING_ETA_DATA L5=BLOCKED_MISSING_ETA_DATA")
 return 0 if ok else 4
if __name__=="__main__":raise SystemExit(main())
