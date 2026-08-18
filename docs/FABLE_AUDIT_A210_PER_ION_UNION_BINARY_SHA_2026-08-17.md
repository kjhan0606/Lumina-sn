# Fable 판정: A2-10 V4 per-ion union과 literal binary SHA 모순

질의: `docs/QUERY_FABLE_A210_PER_ION_UNION_BINARY_SHA_2026-08-17.md`

Claude Code CLI의 Fable 모델을 read-only 도구(`Read,Grep,Glob`)와 plan permission으로
실행했다. Fable은 지정된 코드·계획·봉인 산출물을 직접 읽고 다음 판정을 내렸다.

## 단일 판정

**선택지 1 — 새 diagnostic-only 바이너리 허용. 새 SHA 재실행이 유일한 적법 경로다.**

## 실측 근거

- `a210_line_saturation_log_complete`는 세 이온을 합친 단일 내림차순 정렬 뒤 combined
  90% prefix 하나만 기록한다. target fraction은 하드코딩이고 per-ion mode는 없다.
- 수집 candidate는 종료 때 메모리에서 소멸하며 직렬화되지 않는다.
- 봉인 `stderr.log`에는 211,887 candidate 중 `LINE-SATURATION-ROW` 929개와 summary
  1개만 있다. run root의 다른 파일에도 미기록 candidate가 없다.
- V4와 owner report SHA 및 Fe/Co/Ni coverage 수치를 재계산해 질의 값과 일치함을 확인했다.
- 따라서 기존 sealed 산출물로 per-ion union을 read-only 복원하는 것은 불가능하다.
- union 생략도 기각한다. Fe/Ni 표본은 Co가 지배한 global threshold 위의 편향 표본이며,
  누락 행이 J/O 판단에 중립이라는 증명은 누락 행 자체를 필요로 한다.

## 동일 SHA 문구 정정

Stage 3의 literal `동일 binary SHA` 요구는 다음으로 봉인 정정하는 것을 허용했다.

> 동일 물리 baseline/source state; 변경은 read-only 진단 선택/기록에만 한정하고,
> 계보는 동일 input SHA 체인과 공통 baseline strict bit-exact 비교로 입증한다.

Binary SHA 동일성의 증거 기능은 R1 계보 연결이며, SHA 그 자체가 물리 증거는 아니다.
현 문구를 문자 그대로 유지하면 union을 요구하면서 union을 낼 수 없는 바이너리를
요구하므로 Stage 3가 실행 불가능하다.

## 구현·검증 조건

질의에 제시한 증거는 아래 정정과 보강을 포함할 때 충분하다.

1. 새 mode는 별도 env 값으로 두고 기존 `=1` 경로를 비접촉으로 유지한다.
2. 각 ion 내부에서 scaled emission 내림차순, 동률은 line id 오름차순으로 정렬한다.
3. 각 ion 총량의 90%에 처음 도달하는 최소 prefix를 선택하고 세 prefix의 union만 기록한다.
4. “새 target row만 증가”는 불변량이 아니다. Co IV minimal prefix는 기존 617개보다
   줄 수 있다. 대신 기존·신규 **교집합 행의 인쇄 필드 전체 문자열이 바이트 동일**해야 한다.
5. comparator는 행 1개 삭제 또는 scaled-emission 섭동 주입으로 FAIL하는 음성 대조를
   시연해야 PASS 자격을 갖는다.
6. `=1` 바이트 동일 회귀는 실제 diagnostic path가 발화하는 기존 A2-10 selftest
   battery로 확인한다.
7. 물리 producer/publication은 비접촉이고, 동일 deck/input/Sigma/state 계보,
   pre-core=0, generation/publication barrier, physical-values-modified=0,
   floor/cap/clamp/jitter/repair=0을 유지한다.
8. 이 재실행 자체의 physical cause claim은 0이며 V4 PASS 뒤에만 Stage 4로 진행한다.

## 재실행 범위

**A100×2 전체 재실행 약 2.5시간이 필수다.** checkpoint/state restore 경로가 없고,
진단은 R1 envelope, R2 exact solve, Sobolev Jbar 109,014,300 cells의 전체 상류 상태를
소비한다. 또한 R1/R2/LOWER/UPPER/REQUESTED_TE strict 계보 증거는 실제 전 phase 실행으로만
생성된다. 봉인 supervisor 시간은 2시간 33분 38초였다.

재실행은 이번 진단의 기대 종료인 `model.rc=1` + 자연 `RADEQ_NO_BRACKET`을 검사해야 하며,
K-final 전용 `COMPLETED/child.rc=0`을 요구해서는 안 된다.

## Fable이 지정한 다음 다섯 단계

1. Stage 3 문구와 변경 범위·union 정의·cause claim 0을 사전등록 정정으로 봉인한다.
2. Codex가 새 mode를 구현하고 selftest, 기존 mode 바이트 동일, comparator 음성 대조를 수행한다.
3. 물리 producer/publication 비접촉 diff를 감사하고 새 바이너리·input SHA를 봉인한다.
4. 동일 input 사본과 새 바이너리로 A100×2 판정런을 한 번 수행하고 자연 rc=1을 봉인한다.
5. ion별 coverage ≥0.9, 교집합 바이트 동일, baseline strict 비교, owner closure가 모두
   PASS한 뒤에만 Stage 4 J/O 귀속 평가로 진행한다.
