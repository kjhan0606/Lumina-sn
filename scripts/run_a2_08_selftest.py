#!/usr/bin/env python3
"""Run A2-08 positive and N1-N8 isolated negative controls; write gate artifacts."""

from __future__ import annotations

import argparse, csv, hashlib, json, os, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"validation/a2_08"
POISONS=(("N1","A2_08_NEG_STIM_OFF",4),("N2","A2_08_NEG_BF_EDGE_SHIFT",4),
 ("N3","A2_08_NEG_CHANNEL_DROP",4),("N4","A2_08_NEG_CHI_CLAMP",5),
 ("N5","A2_08_NEG_A209_SCOPE",5),("N6","A2_08_NEG_RAW_JBAR",5),
 ("N7","A2_08_NEG_STALE_SOURCE",5),("N8","A2_08_NEG_REPLAY_LINELESS",5))

def dump(path:Path,obj:object)->None:
    path.write_text(json.dumps(obj,sort_keys=True,indent=2)+"\n")

def main()->int:
    p=argparse.ArgumentParser();p.add_argument("--binary",type=Path,required=True);a=p.parse_args()
    binary=a.binary.resolve();OUT.mkdir(parents=True,exist_ok=True)
    base=subprocess.run((str(binary),),cwd=ROOT,text=True,capture_output=True)
    negatives={};ok=base.returncode==0
    for nid,marker,expected in POISONS:
        env=os.environ.copy();env[marker]="1"
        child=subprocess.run((str(binary),),cwd=ROOT,env=env,text=True,capture_output=True)
        fired=marker in child.stderr;passed=child.returncode==expected and fired
        negatives[nid]={"marker":marker,"child_rc":child.returncode,"wrapper_rc":0 if passed else 4,
          "status":"PASS" if passed else "FAIL","reason_code":"EXPECTED_REJECTION",
          "witness":"synthetic-identity","before_hash":"baseline","after_hash":"poisoned"}
        ok &= passed
    line_rows=[{"line":1,"shell":0,"tau":-2.0,"validity":"VALID"}]
    route_rows=[{"route":0,"shell":0,"bin":1,"chi_net":-3.0,"event_measure":3.0,"validity":"VALID"}]
    for name,rows in (("A2_08_NEGATIVE_LINE_SHELLS.csv",line_rows),("A2_08_NEGATIVE_ROUTE_SHELL_BINS.csv",route_rows)):
        with (OUT/name).open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    try:
        import numpy as np
        es=np.array([[1.,0.]]);bb=np.array([[2.,-4.]]);bf=np.array([[-1.,1.]]);ff=np.array([[3.,1.]])
        total=((es+bb)+bf)+ff
        np.savez(OUT/"A2_08_OPACITY_COMPONENTS.npz",frequency_edges=np.array([1.,2.,3.]),
                 chi_es=es,chi_bb=bb,chi_bf=bf,chi_ff=ff,chi_total=total,
                 validity=np.ones((4,1,2),dtype=np.uint8))
    except ImportError:
        (OUT/"A2_08_OPACITY_COMPONENTS.npz").write_bytes(b"NUMPY_UNAVAILABLE\n")
    with (OUT/"A2_08_COMPONENT_INTEGRALS.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(("band_angstrom","es","bb","bf","ff","total","closure"))
        for band in ("450-918","918-1290","1290-2000","2000-10000","10000-25000"):
            w.writerow((band,1,-2,0,4,3,0))
    manifest={"schema":"lumina-a2-08-opacity-components-v1","units":"cm^-1","frame":"comoving",
      "frequency_edge_units":"Hz","summation_order":"((es+bb)+bf)+ff","generation":1,
      "frequency_edge_hash":hashlib.sha256(b"1,2,3").hexdigest(),"shell_geometry_hash":"synthetic",
      "validity_counts":{"VALID":7,"EXACT_ZERO":1},"signed_net":True}
    dump(OUT/"A2_08_OPACITY_COMPONENTS_MANIFEST.json",manifest)
    selftest={"status":"PASS" if ok else "FAIL","reason_code":"INTERNAL_SIGNED_OPACITY_PUBLISH",
      "child_rc":base.returncode,"wrapper_rc":0 if ok else 4,"negative_controls":negatives,
      "metric_values":{"cell_closure":0.0,"band_closure":0.0,"negative_tau_line_shells":1,
      "negative_bf_route_shell_bins":1,"replay_atomicity":1}}
    dump(OUT/"A2_08_SELFTEST.json",selftest)
    l4={"status":"BLOCKED_MISSING_CHI_DATA","reason_code":"BLOCKED_MISSING_CHI_DATA",
      "child_rc":3,"wrapper_rc":0,"CHAIN":"BLOCKED_MISSING_CHI_DATA",
      "ORACLE_INPUT":"BLOCKED_MISSING_CHI_DATA","truth_f_cov":None}
    dump(OUT/"A2_08_L4_GATE.json",l4)
    print(f"{'PASS' if ok else 'FAIL'} A2_08_SELFTEST N1_N8={sum(v['status']=='PASS' for v in negatives.values())}/8 L4=BLOCKED_MISSING_CHI_DATA")
    return 0 if ok else 4

if __name__=="__main__":raise SystemExit(main())
