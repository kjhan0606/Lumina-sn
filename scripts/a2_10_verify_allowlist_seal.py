#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];D=ROOT/"validation/a2_10";J=D/"A2_10_CHANGED_OUTPUT_ALLOWLIST.json";S=D/"A2_10_CHANGED_OUTPUT_ALLOWLIST.sha256";I=D/"A2_10_IMPLEMENTATION_START.json"
def main()->int:
 try:
  m=json.loads(I.read_text());cur=hashlib.sha256(J.read_bytes()).hexdigest();side=S.read_text().split()[0];blob=subprocess.check_output(("git","show",f"{m['seal_commit']}:{J.relative_to(ROOT)}"),cwd=ROOT);sealed=hashlib.sha256(blob).hexdigest();ok=cur==side==sealed==m["json_sha256"] and subprocess.run(("git","merge-base","--is-ancestor",m["seal_commit"],"HEAD"),cwd=ROOT).returncode==0
 except Exception as e:print(f"FAIL A2_10_SEAL {e}",file=sys.stderr);return 2
 print(f"{'PASS' if ok else 'FAIL'} A2_10_SEAL current={cur} sidecar={side} blob={sealed}");return 0 if ok else 2
if __name__=="__main__":raise SystemExit(main())
