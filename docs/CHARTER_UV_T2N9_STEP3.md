# 과업 T2/N9 3단 — 판정과 대장 기재

2단 실행 결과가 `/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/uv_t2n9_offline/`
에 있다. grammar-debug에서 18:36에 실행됐고 상태는
`UNRESOLVED-FAIL-CLOSED`(`failure.json`)다. N9는 산출됐고 T2는 C 커버리지에서
멈췄다.

## 0. 규율 (변경 없음)

- **사전등록을 완화하지 마라.** `validation/uv_t2n9/PREREG.md`가 정본이다.
  판독량은 shell 8 BALL band-mean이고 네 갈래는 그대로다. 경계값을 새로
  만들거나 판독량을 바꾸면 그 순간 이 과업은 무효다.
- clamp·floor·fallback·대체 금지. 정의되지 않는 곳은 정의되지 않는다고 적어라.
- `src/` 수정 금지. GPU·모델 런 금지. commit 금지.
- 무거운 연산은 직접 실행하지 마라. 실행이 필요하면 스크립트를 고치고
  **실행 명령 한 줄**을 보고하라. 투척은 운전석이 한다.

## 1. 첫 질문 — BALL 판독을 정확히 구성할 수 있는가

`t2_C_population_coverage.json`은 nonpositive 인구 행 28,949개 때문에 멈췄다.
그런데 `negative_control.json`은 complete bin이 303개이고 A 재조립이 그
안에서 bitwise 일치함을 보인다. 따라서 다음이 갈린다.

1. **28,949행이 전부 BALL(600–3000 Å) 밖에 있다면** C는 BALL에서 대체 없이
   정확히 구성된다. 그러면 C를 만들고 **사전등록 네 갈래를 그대로 적용**하라.
2. **일부라도 BALL 안에 있다면** 그 행들이 BALL의 chi_line에 기여하는 몫을
   정량하고, 대체 없이는 판독이 불가함을 명시한 뒤 `UNRESOLVED-FAIL-CLOSED`를
   유지하라. 기여 몫의 상한을 숫자로 내라.

이 판단은 사후 기준 선택이 아니다. 판독량은 이미 고정돼 있고, 묻는 것은
그 판독량을 데이터가 대체 없이 지지하는가 하나다. 어느 쪽이든 근거 수치를
내라.

## 2. 두 번째 질문 — nonpositive 인구 28,949행의 정체

이것은 T2의 장애물이자 그 자체로 발견이다. 다음을 특정하라.

- **0인가 음수인가.** 음수가 하나라도 있으면 솔버 결함이다. 개수를 분리하라.
- 어떤 원소·이온·준위인가. 상위 기여자를 순위로 내라.
- 파장 분포. BALL 안팎 분리.
- `tau_used`가 그 행들에서 무엇으로 계산됐는가. 인구가 nonpositive인데
  불투명도는 유한한가. 유한하다면 무엇이 그 값을 만들었는지 writer 소스에서
  추적하라.
- 대장 기재 문안을 작성하라. 수리안은 이 과업의 범위가 아니다.

## 3. 세 번째 — N9 판정문 확정

측정은 끝났다. 판정문만 쓰면 된다. 다음 수치를 근거로 인용하라.

- 셀 수 기준 s>=5 `rate_shape_replaced` 0.76231
- 에너지 기준 shell 8 BALL 0.9956, B1–B4 전 셸 1.0000
- `rate_shape_line_source_BTe` PASS, 최대 상대오차 2.22e-16, 1 ULP,
  항등식 `eta_rate_line = chi_line_th * B_nu(Te)`
- clamp 0, fallback 0, nonfinite 0, 음성 대조 expected FAIL = observed FAIL

판정문은 다음 물음에 답해야 한다. **s>=5에서 UV 선 방출의 실질 전부가
구성상 열적이라면, 인구·원자데이터·형광 행렬에 대한 어떤 개입이 UV 형상을
바꿀 수 있는가.** 답이 "없다"라면 그렇게 적어라.

## 4. 네 번째 — 대장 기재 2건

`EPAY-REPLAY-001`(계기 결함, 이미 문안 초안이 `n9_summary.json`에 있다)과
nonpositive 인구 건을 `docs/VERIFICATION_REGISTERS.md`의 형식에 맞춰
**기재용 문안으로** 작성하라. 파일을 직접 고치지 말고 문안만 보고서에 실어라.

## 5. 보고

전체는 `docs/CODEX_UV_T2N9_STEP3.md`. `-o` 요약에는 BALL 구성 가능 여부와
그 근거 수치, 사전등록 갈래 적용 결과 또는 UNRESOLVED 사유, nonpositive
인구의 0/음수 분리 개수, N9 판정문 한 문단, 추가 실행이 필요하면 명령 한 줄만
담아라.
