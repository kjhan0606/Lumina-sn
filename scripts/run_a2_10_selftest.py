#!/usr/bin/env python3
import argparse, os, subprocess, sys

p=argparse.ArgumentParser(); p.add_argument("--binary",required=True); a=p.parse_args()
cases=[
 ("A2_10_NEG_PHOTOHEAT_DROP",4),("A2_10_NEG_NEIGHBOR_TE",4),
 ("A2_10_NEG_CANCEL_PAIR",4),("A2_10_NEG_STALE_TERM",5),
 ("A2_10_NEG_PLANCK_FIELD",5),("A2_10_NEG_ROOT_PIN",5),
 ("A2_10_NEG_TERM_SIGN",4),("A2_10_NEG_TE_MANIFEST",5),
 *((x,6) for x in ("XSHELL_NEG_SELECT","XSHELL_NEG_FREEZE","XSHELL_NEG_INIT",
 "XSHELL_NEG_NULL","XSHELL_NEG_SH24","XSHELL_NEG_RATE","XSHELL_NEG_DEAD","XSHELL_NEG_DIR"))]
r=subprocess.run([a.binary])
baseline_pass=int(r.returncode==0)
if not baseline_pass:
 print(f"[A2-10][SELFTEST-SUMMARY] baseline_pass={baseline_pass} baseline_rc={r.returncode} negative_expected_fail=0 total=0",file=sys.stderr)
 sys.exit(1)
ok=baseline_pass
for name,want in cases:
 e=os.environ.copy(); e[name]="1"; q=subprocess.run([a.binary],env=e)
 if q.returncode!=want:
  print(f"[SELFTEST][FAIL] {name} rc={q.returncode} expected={want}",file=sys.stderr);sys.exit(1)
 ok+=1
print(f"[A2-10][SELFTEST-SUMMARY] baseline_pass={baseline_pass} negative_expected_fail={len(cases)} total={ok}")
