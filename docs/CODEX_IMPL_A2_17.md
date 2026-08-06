# A2-17 구현 보고

## 판정

A2-17 구조 계약은 `PASS`다. production `W/T_rad` read, owner field,
allocation/free/update/upload/output/env option, renamed owner alias 및 offline converter
production link가 모두 0이다. canonical 원장은 157/157 terminal, unknown 0, duplicate 0이다.

물리 truth 판정은 구조 PASS와 합치지 않았다. 두 번째 명시적 CMFGEN EDDFACTOR seed와
truth-side `f_cov`가 작업트리에 없으므로 two-seed gate는
`BLOCKED_MISSING_CMFGEN_EDDFACTOR_SEED` /
`SEED_INDEPENDENCE_PENDING_A2_18`이다. `blocked_as_pass_count=0`이다.

## 단계별 변경

### 커밋 1 — native seed와 offline converter

- commit: `8f99e7f0e995e4b293fd395b5dbc3f54f315fbd7`
- `src/jnu_seed.{c,h}`: shell boundary/ID, frequency edges, bin-average `J_nu`,
  units, comoving frame, epoch, provenance, source hashes, per-cell validity를 가진
  native schema/loader.
- invalid coverage는 `valid=false`로 유지한다. hold, extrapolation, neighbor copy,
  zero publish는 없고 incomplete coverage는 `BLOCKED_INCOMPLETE_SEED_COVERAGE`다.
- `tools/lumina_legacy_seed_converter.c`: runtime과 분리된 converter. provenance는
  `DILUTE_PLANCK_LEGACY_APPROXIMATION`이다.
- 실측: `data/tardis_reference`, 30 shells × 4000 bins = 120,000 cells,
  output 1,112,824 bytes, SHA-256
  `65aba6c0ce863bb003a125b5ec6022bd07887c900fc4a75ac9fe7b4dccc31d08`.
- Makefile, Z battery의 `src/gpu_radiation_field_contract.c` 인접 5개 compile list,
  `run_zinert_selftest.py`와 `run_zinert_selftest.sh`에 신규 TU/selftest를 배선했다.

### 커밋 2 — production scalar 철거

- commit: `dd9cd398698914b5e004e33d1c8ec41fcd5f411f`
- 계약 census 109 reads를 철거했다:

| 파일 | 계약 read 수 | 처분 |
|---|---:|---|
| `lumina_plasma.c` | 51 | checked `J_nu`/material `T_e`, scalar fit·alternate Newton 퇴역 |
| `lumina_cuda.cu` | 20 | host/device owner, upload, packet/rate/output read 제거 |
| `lumina_atomic.c` | 15 | legacy deck column parser, color fix, scalar `T_e` seed 제거 |
| `lumina_main.c` | 15 | validation reload, comparator, scalar output 제거 |
| `lumina_cmfgen.c` | 6 | owner validation/hash/serialization/classifier 제거 |
| `lumina_cmf_selftest.c` | 2 | test-only local analytic witness로 분류 |

`PlasmaState.W`, `PlasmaState.T_rad`, `T_e_T_rad_ratio`, `d_W`, `d_T_rad`와
그 lifecycle은 없다. generation-0 `T_e`는 material seed인
`opacity.t_electrons`에서만 초기화한다. runtime은 `plasma_state.csv`의 legacy scalar
열을 열지 않는다. 폐기된 scalar option은 조용히 무시하지 않고 nonzero loader 오류다.

출력 schema는 scalar 열을 0/NaN sentinel로 보존하지 않고 `shell_id,T_e,n_e`로
올렸다. legacy per-bin dilute-Planck 압축 production path는 raw canonical fine-bin
estimator 게시로 교체했고, scalar-radiation coupled-Newton 대체 경로는 ABI no-op으로
퇴역했다. 컴파일 제외된 역사/oracle 블록과 CMF 자체검증 지역변수만
`COMMENT_STRING_TEST`로 남는다.

### 커밋 3 — zero-consumer와 원장

- `scripts/a2_17_static_read_trace.py`: `.c/.cu/.h` 52개 inventory, 필수
  `lumina_main.c`/`lumina_element_wide.c`, preprocessor-inactive 분리, production link,
  output schema, obsolete options, terminal ledger 및 N17-1..8을 검사한다.
- static 결과: raw 33, classified 33, `PRODUCTION_READ=0`,
  `COMMENT_STRING_TEST=33`, offline converter read 2, unknown/duplicate 0.
- owner/lifecycle/update/upload/output/env/renamed alias/forbidden return/fallback는 모두 0.
- `docs/A2_01_DISPOSITION_LEDGER.{json,md}`는 명세 §4.4의 157행 terminal state와
  stage별 owning commit을 전부 기재한다. census는 completed=157,
  unclassified=0, rc=0이다.
- N17-1..8은 등록 marker와 child rc 41..48을 모두 관측했고 wrapper rc=0이다.

## 검증

- `make -B -j2 lumina`: PASS.
- `make -B -j2 lumina_cuda`: PASS (build/link only; GPU execution은 운전석 몫).
- A2 CPU selftests 03, 04, 05, 06, 07, 08, 09, 10, 12, 13-15, 16, 17: PASS.
  A2-08/09/10의 기존 `BLOCKED_*` truth lane은 그대로 보존했다.
- `bash scripts/run_zinert_selftest.sh --serial`: 12/12 PASS.
- `python3 scripts/a2_01_census_contract.py check`: PASS, rc=0.
- static read trace + N17-1..8: PASS, rc=0.
- full battery와 GPU runtime은 사용자 지시대로 운전석 실행 범위다.

## 산출물

- `validation/a2_17/A2_17_STATIC_READ_TRACE.json`
- `validation/a2_17/A2_17_LEDGER_TERMINAL_TARGET.json`
- `validation/a2_17/A2_17_TWO_SEED_MANIFEST_GATE.json`
- `A2_17_STEP1.bundle` / SHA-256
  `a552a6ee1ab100804d6041c8ef4bd683aecfaebf8345a9881828cade7c50ec08`
- `A2_17_STEP2.bundle` / SHA-256
  `80723ef80d14b3cd900f6f38426ea264ed55dabf3a9752719fbae14c8e31cceb`

원본 `.git`은 `index.lock` 생성이 거부되어 쓰지 않았다. 별도 writable Git metadata에서
commit을 만들고 각 단계 bundle과 SHA-256을 작업트리에 남겼다. push는 수행하지 않았다.
