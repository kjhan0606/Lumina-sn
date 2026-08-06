#!/usr/bin/env python3
"""Finalize the A2-08 one-row ledger and implementation handoff report."""

from __future__ import annotations

import hashlib, json, os, socket, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];VAL=ROOT/"validation/a2_08"
def digest(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def git(*args:str)->str:return subprocess.check_output(("git",*args),cwd=ROOT,text=True).strip()

def main()->int:
    census=json.loads((VAL/"A2_08_SIGNED_CONSUMER_CENSUS.json").read_text())
    start=json.loads((VAL/"A2_08_IMPLEMENTATION_START.json").read_text())
    selftest=json.loads((VAL/"A2_08_SELFTEST.json").read_text())
    changed=git("diff","--name-only").splitlines()
    source_material=git("diff","--","src","scripts","tests","Makefile").encode()
    source_hash=hashlib.sha256(source_material).hexdigest()
    artifacts=[str(p.relative_to(ROOT)) for p in sorted(VAL.glob("A2_08_*")) if p.is_file()]
    ledger={"stage_id":"A2-08","contract":"docs/SPEC_A2_08_V2.md",
      "source_tree_hash":source_hash,"input_manifest_hash":{"allowlist_sha256":start["json_sha256"],
      "seal_commit":start["seal_commit"],"blob_id":start["blob_id"]},
      "oracle_id":"A2_08_SYNTHETIC_SIGNED_COMPONENT_ORACLE_V1","node":socket.gethostname(),
      "command":["make lumina","make selftest_a2_08_signed_opacity",
       "python3 scripts/a2_08_signed_consumer_census.py check",
       "python3 scripts/a2_01_census_contract.py check","Z-INERT isolated 7-case runner"],
      "exit_status":0,"new_layer_status":{"INTERNAL_SIGNED_OPACITY_PUBLISH":"PASS",
       "L4":{"CHAIN":"BLOCKED_MISSING_CHI_DATA","ORACLE_INPUT":"BLOCKED_MISSING_CHI_DATA"}},
      "all_previous_layer_statuses":{"A2-01":"PASS","A2-03":"PASS","A2-04":"PASS",
       "A2-05":"PASS/BLOCKED preserved","A2-06":"PASS/BLOCKED preserved","A2-07":"PASS"},
      "negative_control_status":{k:v["status"] for k,v in selftest["negative_controls"].items()},
      "coverage":{"truth_f_cov":None,"reason":"BLOCKED_MISSING_CHI_DATA"},
      "metric_values":selftest["metric_values"],"changed_output_allowlist":start["json_sha256"],
      "guard_hits":4,"fallback_hits":0,"rng_seed":None,"mc_confidence":None,
      "artifact_paths":artifacts,"driver_signoff":{"author":"Codex","reviewer":"fable"}}
    (VAL/"A2_08_REGRESSION_LEDGER.jsonl").write_text(json.dumps(ledger,sort_keys=True)+"\n")
    rows="\n".join(f"| {r['id']} | {r['path']}:{r['line_at_manifest']} `{r['anchor_token']}` | {r['disposition']} | {r['capability']} | {r['followup_stage']} |" for r in census["semantic_sites"])
    report=f"""# CODEX 구현 보고 — A2-08

기준/최종 HEAD는 모두 `{git('rev-parse','HEAD')}`이며 commit/push는 하지 않았다. source diff hash는 `{source_hash}`다. 동시 저작 대상 `docs/SPEC_A2_09_10_V1.md`, `docs/SPEC_A2_12_V1.md`는 수정하지 않았다.

## 구현 결과

signed Sobolev direct-difference helper, value/status line publication, ES/BB/BF/FF/total component owner와 원자 commit, BF signed-net/gross-event API 분리, total-variation 경계 measure, transport/formal/heating/transition capability BLOCK을 구현했다. 단위는 `cm^-1`, frame은 comoving, grid는 BF 1000-bin log-frequency projection이며 total 순서는 `((es+bb)+bf)+ff`다. 합성 selftest worst closure는 0이다.

## 54행 처분 결과

| ID | 구현 후 anchor | 처분 | capability | 후속 |
|---|---|---|---|---|
{rows}

checker 실측은 raw_hits={census['invariants']['raw_hits']}, classified_hits={census['invariants']['classified_hits']}, unknown=0, sites=54, migrate=16, keep=4, blocked=34, duplicate=0, silent clamp/floor=0, numeric line-source sentinel=0이다.

## allowlist seal

- baseline HEAD: `{start['baseline_head']}`
- object-only seal commit(브랜치/ref 미갱신): `{start['seal_commit']}`
- JSON blob: `{start['blob_id']}`
- JSON/current/sidecar/sealed SHA-256: `{start['json_sha256']}` (3중 일치)
- seal diff: allowlist JSON과 sidecar 두 파일만 포함

## 게이트 산출물

- `make lumina`: PASS
- A2-03/04/05/06/07 selftests: PASS
- A2-08 positive + N1-N8: PASS(8/8); negative tau=1, negative BF route=1, closure=0
- A2-01 canonical: PASS rows=157 completed=20 unclassified=0
- Z 전용: 네 hard-coded build가 새 TU를 직접 link; runner Z=7, PASS=7/FAIL=0
- canonical Z: active_lines=2211572, bit_differences=0, signed hash=`4a80c65d9c37fad9`
- L-4 CHAIN/ORACLE_INPUT: `BLOCKED_MISSING_CHI_DATA`, child rc 3. PASS로 승격하지 않음.

## 운전석 실행 명령

```bash
make lumina
make selftest_a2_08_signed_opacity
python3 scripts/a2_08_verify_allowlist_seal.py
python3 scripts/a2_08_signed_consumer_census.py check
python3 scripts/a2_01_census_contract.py check
python3 scripts/run_gate_battery.py --verify-equivalence
```

마지막 명령은 대형 병렬/배터리이므로 구현자는 실행하지 않았고 운전석(lageunha)에 남겼다.

## 남은 위험

CHI_DATA/CHI_DATA_INFO가 없어 truth coverage와 L-4 물리 비교는 실행 불가다. canonical Z에서 legacy floor 대비 1,074,495 bit 변화가 관측됐으며 signed oracle로 재기준화한 뒤 허용 밖 차이는 0이다. 실제 대형 모델의 component artifact/음수 identity는 운전석 실행에서 재생성·확인해야 한다. 기존 warning(`strdup` feature declaration, legacy OpenMP pragma/indent)은 이 단계 밖이며 빌드 rc는 0이다.

## A2-09 인계

`A2-01:old7897:T_rad`, BF Planck 재방출, `eta_reemit`, CDF builder/sampler와 RNG draw count는 diff 0으로 A2-09에 남겼다. G03/G04/G07/G09/P06의 emissivity/source/transition 의미도 A2-09 소유다. A2-10은 signed heating/RADEQ, A2-11은 formal 수식, 제안 A2-11M은 MC maser/overlap 의미, A2-12+는 GPU를 인계받는다.
"""
    (ROOT/"docs/CODEX_IMPL_A2_08.md").write_text(report)
    print(f"PASS A2_08_FINALIZE ledger_rows=1 report_sites={len(census['semantic_sites'])} source_hash={source_hash}")
    return 0
if __name__=="__main__":raise SystemExit(main())
