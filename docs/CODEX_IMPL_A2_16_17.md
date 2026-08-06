# A2-16 + A2-17 구현 보고서 — 시작 장벽 판정 (개정 11)

## 판정

두 계약 모두 **`BLOCKED_UPSTREAM_NOT_CLOSED`**다. 이는 구현 편의에 따른 재해석이
아니라 `docs/SPEC_A2_13_18_V1.md` §4.1의 강제 시작 장벽 결과다.

- 기준 HEAD: `9b73c042dcc5feae75c58ff8e00547b33ea36d93`
- A2-12 저장된 판정: `UNVERIFIED_GPU_NODE`
- A2-13/A2-14/A2-15 regression ledger와 구현 커밋: 없음(`NOT_RUN`)
- 운전석 제공 GPU job: SLURM `217317`, 진행 중이라고 보고됨. sandbox에서는 scheduler
  socket 접근이 거부되어 종결 상태를 독립 확인하지 못했다.
- A2-16 structural source 변경: 0. A2-13~15가 아직 소비하는 scalar를 먼저 제거하지 않았다.
- A2-17 owner/lifecycle/output source 변경: 0. zero-consumer가 아닌 상태에서 producer를
  철거하지 않았다.
- `.cu` 및 rate kernel 수식 변경: 0.

## 커밋별 소속

### A2-16

- bundle commit: `f5d646cba44b56ecd4e2f57ba9f1ace4b3e4c69d`
- 소속 파일: `scripts/a2_16_seed_read_trace.py`,
  `validation/a2_16/A2_16_SEED_READ_TRACE.json`,
  `validation/a2_16/A2_16_REGRESSION_LEDGER.jsonl`
- 결과: 전체 `src` 40파일, raw scalar/ratio/obsolete-option hit 604,
  보수적 production read 120, 격리되지 않은 `lumina_atomic.c` generation-0 직접 seed
  target 1. checker rc=3.
- N16-1~8 marker와 기대 child rc 41~48을 실행 전 고정했다. 시작 장벽 때문에 poison을
  실행해 PASS를 주장하지 않고 `NOT_RUN_BLOCKED_START_BARRIER`로 남겼다.

### A2-17

- bundle commit: 이 보고서를 포함하는 `A2_17_BUNDLE_TIP`(bundle 전달 SHA로 확정)
- 소속 파일: `scripts/a2_17_static_read_trace.py`,
  `validation/a2_17/A2_17_STATIC_READ_TRACE.json`,
  `validation/a2_17/A2_17_LEDGER_TERMINAL_TARGET.json`,
  `validation/a2_17/A2_17_REGRESSION_LEDGER.jsonl`, 이 보고서
- 결과: 전체 `src` 40파일(필수 `lumina_main.c`, `lumina_element_wide.c` 포함), raw 600,
  classified 600, unknown 0, duplicate 0, production read 119. checker rc=3.
- N17-1~8 marker와 기대 child rc 41~48을 실행 전 고정했다. 시작 장벽 때문에 실행 상태는
  `NOT_RUN_BLOCKED_START_BARRIER`다.

source 수정 전 allowlist seal은 만들지 않았다. 시작 장벽 뒤 source 수정 자체가 금지됐고,
물리 출력/owner/output schema 변경도 없기 때문이다. seal이 필요한 실제 구현은 upstream
폐합 후 첫 source 수정 전에 새로 만들어야 한다.

## canonical 원장 157행 최종 상태

`validation/a2_17/A2_17_LEDGER_TERMINAL_TARGET.json`에 ID 001~157 각각의
`pre_witness`, `post_witness_or_absence_query`, `required_terminal_state`, owning commit,
static category, runtime counter를 한 행씩 기록했다. cardinality 157, unknown 0,
duplicate 0이다. A2-05~12 폐합분은 명세의 terminal state와 기존 owning commit으로
반영했고, A2-13~16 미폐합분은 `BLOCKED_NO_COMMIT` 또는 blocked evidence로 명시했다.

현재 canonical markdown 원장의 옛 disposition을 terminal state로 덮어쓰지 않았다.
A2-13~16이 닫히지 않은 상태에서 157행을 모두 CLOSED/REMOVED로 쓰면 허위 폐합이기 때문이다.
따라서 terminal target은 전량 기재됐지만 `terminal_verified=0/157`이며, A2-17의 최종
zero-consumer 상태가 아니다.

목표 상태별 행 수는 다음과 같다.

| terminal owner/state | 행 수 | 현재 판정 |
|---|---:|---|
| A2-04 canonical commit | 14 | 기존 이관 반영 |
| A2-06 canonical Jbar | 14 | 기존 이관 반영 |
| A2-07 matter Te | 14 | 기존 이관 반영 |
| A2-08 signed opacity | 12 | 기존 이관 반영 |
| A2-09 emissivity | 3 | 기존 이관 반영 |
| A2-10 RADEQ canonical | 2 | 기존 이관 반영 |
| A2-11 formal no-scalar | 9 | 기존 이관 반영 |
| A2-12/15 GPU no-scalar | 22 | upstream 미폐합 |
| A2-13 GPU rate | 8 | `NOT_RUN` |
| A2-14 GPU signed opacity | 13 | `NOT_RUN` |
| A2-15 GPU emissivity | 4 | `NOT_RUN` |
| A2-16 g0 seed revoked | 15 | blocked |
| A2-17 owner/lifecycle/output removed | 4 | blocked |
| canonical diagnostic-derived only | 23 | A2-17 검증 대기 |
| 합계 | 157 | `BLOCKED_UPSTREAM_NOT_CLOSED` |

## static read-trace 증거

실행 명령:

```bash
python3 scripts/a2_16_seed_read_trace.py \
  --output validation/a2_16/A2_16_SEED_READ_TRACE.json
python3 scripts/a2_17_static_read_trace.py \
  --output validation/a2_17/A2_17_STATIC_READ_TRACE.json \
  --ledger-output validation/a2_17/A2_17_LEDGER_TERMINAL_TARGET.json
```

두 명령 모두 의도대로 rc=3이다. artifact SHA-256은 다음과 같다.

- A2-16 trace: `510e23fa494c9c25833469828ff1686cb4dcb8e0202dbfb800c1c27bc211327a`
- A2-17 trace: `21f538ff89d8ed5480e9e27d4662adfbcc40c6cb3a266723e193b75557262041`
- 157행 target: `bc7867f4506c0195203c993dabefcd5719dc5e49e5b5fc5a54278695554e54e4`

A2-17 category는 `PRODUCTION_READ=119`, `DIAGNOSTIC_DERIVATION_CANDIDATE=2`,
`COMMENT_STRING_TEST=231`, `DEFINITION_ASSIGNMENT_ARGUMENT_RETURN_OR_STRING=248`이다.
`raw_scalar_hits=classified_scalar_hits=600`이고 unknown/duplicate는 0이다. 다만 alias/callgraph
최종 증명은 upstream migration이 도착하기 전에는 성립하지 않으므로 renamed alias와
forbidden return path를 `NOT_PROVABLE`로 기록했다. zero-consumer를 증명했다는 주장은 없다.

## validity, fallback, counters, `nlte_free`

- native seed loader/capability가 설치되지 않았으므로 validity cell을 0/hold/extrapolation로
  세탁하지 않았다. 관련 runtime counter는 `null/NOT_RUN`이며 0으로 가장하지 않았다.
- fallback hit도 runtime 미실행이므로 `UNKNOWN_NOT_RUN`이다. PASS counter로 쓰지 않았다.
- 신규 allocation/TU가 없으므로 신규 `nlte_free` 배선도 없다. 기존 cleanup은
  `src/lumina_plasma.c`의 `nlte_free`, CPU main 두 호출, CUDA 한 호출을 그대로 보존했다.
- 신규 production TU가 없어 Z 네 hard-coded link와 `run_zinert_selftest.py` 신규 row 의무는
  발생하지 않았다.

## 검증

- `make lumina`: PASS(up to date)
- Makefile `selftest*` target 26/26: build/target rc=0
- 직접 binary 25개: 14 PASS, 6 usage rc=2(인자형), GPU lifecycle rc=70,
  기존 data/generation fixture 4개 nonzero
- 공식 wrapper 9개: 7 PASS; 기존 `cmf_linepop_roundtrip_selftest.py`는
  `invalid eta_line; no clamp allowed`, Z wrapper는 기존 link list에서
  `a208_signed_sobolev` 누락으로 FAIL
- `python3 scripts/a2_01_census_contract.py check`: PASS,
  `rows=157 completed=20 unclassified=0`, rc=0
- selftest inventory SHA-256:
  `ac0bf31fab7138bb99eaeca0ee0d4553947d760e604ed3e18c0d01363710a185`
- full gate battery: 운전석 지시대로 미실행
- static read-trace: A2-16 rc=3, A2-17 rc=3(정직한 BLOCKED)

## 남은 위험과 A2-18 인계

1. SLURM 217317 종결 artifact가 현재 작업트리의 A2-12
   `UNVERIFIED_GPU_NODE` report/ledger를 실제 PASS 또는 구체적 BLOCKED로 갱신해야 한다.
2. A2-13~15는 별도 계약 커밋, CPU/GPU oracle, poison, GPU read-trace까지 폐합돼야 한다.
3. 그 뒤 A2-16은 source 수정 전 allowlist seal을 만들고 native seed schema/offline converter,
   g0 capability, first-commit revoke/free, N16-1~8을 실제 구현·실행해야 한다.
4. A2-16 폐합 뒤 이 trace를 재실행해 production read 0을 먼저 증명하고서만 A2-17 producer,
   owner field, allocation/upload/free/output/env를 제거한다.
5. A2-18은 이 문서의 blocked evidence를 PASS로 승격하지 말고, 두 seed 전량 비교와 157행
   `terminal_verified=157/157`, runtime counter, driver signoff가 생긴 뒤 재판정해야 한다.
