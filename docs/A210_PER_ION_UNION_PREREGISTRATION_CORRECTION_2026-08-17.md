# A2-10 Stage 3 per-ion union 사전등록 정정

이 문서는 `docs/FABLE_PLAN_ANALYSIS_TRANSFER_A210_2026-08-17.md` Stage 3의
`동일 binary SHA·동일 state 필수` 문구 중 literal binary SHA 요구만 정정한다.
V4 fail-closed, 물리 원인 claim 금지, union 정의와 모든 수치 안전 제약은 완화하지 않는다.

근거 판정:
`docs/FABLE_AUDIT_A210_PER_ION_UNION_BINARY_SHA_2026-08-17.md`

## 확정된 모순

봉인 K36 binary는 Fe/Co/Ni IV의 combined-emission global 90% prefix만 기록한다.
per-ion selector가 없고 미선택 candidate는 직렬화되지 않았으므로, 동일 binary SHA로
per-ion union을 생성하는 것은 불가능하다. V4는 Fe IV 0.738388..., Co IV 0.935772...,
Ni IV 0.710001...로 `UNDERCOVERED`다.

## 정정된 Stage 3 계약

- 동일 물리 baseline/source state를 유지한다.
- 코드 변경은 `src/lumina_plasma.c`의 saturation diagnostic 선택/기록부와 이를 검증하는
  strict option/test/comparator에만 한정한다.
- 새 binary SHA는 허용하되, 동일 input SHA chain과 R1/R2/LOWER/UPPER/REQUESTED_TE 공통
  baseline strict bit-exact 비교로 계보 동일성을 입증한다.
- 기존 `LUMINA_A210_LINE_SATURATION_DIAG=1`의 combined-prefix 동작은 바이트 동일하게 보존한다.
- 새 strict mode는 각 Z=26,27,28에 대해 scaled emission 내림차순, 동률 line id
  오름차순으로 정렬하고, 각 ion 총량의 0.9에 처음 도달하는 최소 prefix를 선택한다.
  출력 대상은 세 prefix의 union이다.
- 신규 union과 기존 global prefix의 교집합 line은 모든 인쇄 물리/진단 필드가 바이트 동일해야 한다.
- union 검사기는 행 삭제 또는 scaled-emission 섭동에 FAIL하는 음성 대조를 통과해야 한다.
- 재실행 자체의 parity claim=0, physical cause claim=0이다.
- 물리 producer/publication 비접촉, pre-core=0, coevolution generation/publication barrier
  보존, physical_values_modified=0, floor/cap/clamp/jitter/repair=0을 유지한다.
- A100×2 전체 재실행의 기대 종료는 `model.rc=1` + 자연 `RADEQ_NO_BRACKET`이다.
- ion별 coverage ≥0.9, prefix 최소성, 교집합 바이트 동일, owner closure, 공통 baseline strict
  비교가 전부 PASS한 뒤에만 Stage 4 J/O 귀속으로 진행한다.

## 기대 변경 집합

- `src/lumina_plasma.c`: diagnostic-only mode parsing, per-ion minimal-prefix 선택과 기록.
- strict env universe/knob registry의 새 값 또는 새 진단 option 등록.
- 해당 mode의 deterministic selftest와 union comparator/negative control.
- staging/monitoring/manifest는 새 binary와 검증 산출물을 봉인하는 범위에서만 변경한다.

이 정정은 물리 공식, solver, material, opacity, emissivity, population, radiation field,
publication 권한을 변경하지 않는다.
