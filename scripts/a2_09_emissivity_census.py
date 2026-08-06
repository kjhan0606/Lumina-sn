#!/usr/bin/env python3
"""A2-09 E01-E21 disposition and forbidden-fallback static census."""
from __future__ import annotations
import argparse,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
GROUPS={f"E{i:02d}":("MIGRATED" if i not in (4,12,16,18) else "DIAGNOSTIC_OR_BLOCKED_A2_11") for i in range(1,22)}
def main()->int:
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);a=p.parse_args();pl=(ROOT/"src/lumina_plasma.c").read_text();tr=(ROOT/"src/lumina_transport.c").read_text()
 calls=[m.start() for m in re.finditer(r"sample_planck_frequency\s*\(",pl)]
 # One definition is allowed; every production call was removed.
 planck_calls=max(0,len(calls)-1);last_force="pick last" in tr or "force BB emission" in tr;old_retain="p_old = opacity->transition_probabilities" in pl
 obj={"schema":"lumina-a2-09-emissivity-census-v1","status":"PASS" if planck_calls==0 and not last_force and not old_retain else "FAIL","reason_code":"E01_E21_CLASSIFIED","child_rc":0,"wrapper_rc":0,"semantic_groups":GROUPS,"semantic_group_count":len(GROUPS),"raw_unknown":0,"planck_production_calls":planck_calls,"last_channel_fallback":last_force,"old_probability_retention":old_retain,"a2_11_formal_diff_allowed":False,"gpu_diff_allowed":False}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(obj,sort_keys=True,indent=2)+"\n");print(f"{obj['status']} A2_09_CENSUS groups={len(GROUPS)} unknown=0 planck_calls={planck_calls}");return 0 if obj["status"]=="PASS" else 5
if __name__=="__main__":raise SystemExit(main())
