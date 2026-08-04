# Stage 3.2 Rung 1 — 독립 리뷰 REJECT에 대한 수리 발주

`patches/stage32_rung1_readonly_lambda.patch`
(sha256 `db400a22907f32b126fa9007972de4be8dbf76ad3297b8ff5bee99c04bf65bb8`)가
독립 리뷰에서 **REJECT**됐다. 판정 근거는 `docs/CODEX_STAGE32_RUNG1_REVIEW.md`다.
그 문서를 읽고 F1~F4를 수리하라.

## 0. 먼저 못박는 것

- **사전등록은 고정이다.** `patches/stage32_rung1_expected_changes.txt`의 문안과
  `rho_local` 예측 구간을 바꾸지 마라. 측정이 사전등록의 정의를 만족하도록
  **구현을 고치는 것**이지, 구현이 낸 값에 맞춰 정의를 고치는 것이 아니다.
- **1단 범위는 그대로다.** 읽기 전용 계측. 선원함수·불투명도·방출률·율·
  population·수송 상태를 바꾸지 마라. 2단 이후를 구현하지 마라.
- 리뷰가 무죄로 판정한 부분(기존 물리 상태 무변경, 산출물의 수송 미재주입)을
  수리 과정에서 깨뜨리지 마라.

## 1. F1 — disposition을 branch-site 실측으로

현재는 실제 분기 직전에 조건을 **재계산**하고, 그 재계산식이 실제 분기
(`src/lumina_cmfgen.c:1704`의 `epay >= 2 && acc_w > 0.0 && hot_regime`)에서
`acc_w > 0.0`을 누락했다.

**요구**: 처분 값을 **실제 분기가 실행되는 그 자리에서** 기록하라. 조건을 다시
쓰지 마라. 분기가 여러 곳이면 각 분기마다 기록하고, 어떤 분기도 타지 않는 경로가
있으면 그 사실 자체를 별도 표지로 남겨라. 재계산·근사·추론은 금지다.

**요구**: 체커가 manifest의 하드코딩 문자열이나 `output_discarded_rows: 0`을
신뢰하지 않도록 고쳐라. 처분이 branch-site에서 왔음을 **독립적으로** 검증하라.

**음성 대조 추가 의무**: 이번 결함을 재현하는 주입 시험을 넣어라. 즉 처분을
`acc_w` 조건을 뺀 식으로 재계산하도록 결함을 심었을 때 체커가 **FAIL하는 것을
시연**하라. 지금 체커는 이 결함을 못 잡았다. 못 잡는 체커는 게이트가 아니다.

## 2. F2 — 행별 에너지를 실제 방출 에너지로

현재는 bin 총 `eta_line`을 불투명도 몫 `w_l / chi_line_bin`으로 배분한다. 실제
조립은 선마다 `eta_l`을 더한다(`src/lumina_cmfgen.c:1371-1395`).

> **정정 (2026-08-03, 운전석 오류).** 이 절은 원래 조립식을 `eta_l = w_l * S_l`로
> 적었다. 틀렸다. production은 `!emiss_b && eps_phys`일 때
> `eta_l = w_l * el * S_l`이고(`src/lumina_cmfgen.c:1376`, writer 대응부 `:792-801`),
> `el`은 `radeq_line_eps_phys(...)`를 `eps_floor`/`eps_cap`으로 자른 값이다. 그렇지
> 않을 때만 `eta_l = w_l * S_l`이다. v2 패치는 이 틀린 명세를 충실히 구현했으므로
> 그 결함의 1차 책임은 이 발주서에 있다. 후속 수리 발주는
> `docs/CHARTER_STAGE32_RUNG1_REPAIR_F5.md`.

한 bin에 서로 다른 `S_l`이 둘 이상이면 두 값은 다르다. **불투명도 몫은 방출
에너지 몫이 아니다.**

**요구**: 선별 `w_l * S_l`을 실제로 누적해 행에 기록하라. 배분식을 쓰지 마라.

**요구**: 얇은 선의 분자 불일치를 없애라. production은 `tau <= 1e-6`에서
expansion-opacity 분자로 `tau`를 쓴다(`src/lumina_cmfgen.c:1369-1370`). 패치와
fixture는 항상 `-expm1(-tau)`를 쓴다. **production과 동일한 식**을 쓰고,
fixture가 비-production 식으로 `chi`를 만들어 불일치를 가리는 구조도 없애라.

**요구**: census 폐합을 증명하라. 자기 행들의 합만으로는 부족하다. 독립적으로
얻은 authoritative `sum(eta_pre_epay*dnu)` 총량, 선택-window 경계 bin의 비선택
선 기여, closure residual을 manifest에 기록하라.

## 3. F3 — 정확해를 누락·거부하는 가드 3건 제거

판별식은 "정확해가 위반할 수 있는 가드인가"이다. 셋 다 걸린다.

1. `tau <= 1e-12` 행을 조용히 건너뛰는 것. 양의 유한 `tau`에 beta 정확해가
   존재한다. 숨은 active-set floor다. manifest의 `floor: 0` 주장과도 모순이다.
2. `lambda_star < 1`과 `rho < 1`의 강제. binary64에서 `tau=2e16`이면
   `beta=5e-17`이고 `1-beta`는 **올바르게** 1.0으로 반올림된다. 정상 입력을
   실패시킨다.
3. 유한 `C_ul + A_ul`의 overflow 가능성에 대해 `denom` 비유한을 즉시 거부하는 것.
   정확한 비율 `C/(C+A)`가 정의되는 유한 입력을 거부한다. 안정적인 비율 계산으로
   바꿔라.

제거하되 **무엇으로도 대체하지 마라.** 값이 정의되지 않으면 정의되지 않는다고
기록하고, 계산이 불가능하면 중단하고 보고하라. 조용한 통과 금지.

## 4. F4 — R-N4 세대 규율에 정확히 맞추기

- writer가 실제 경로에 `.iter%03d` 세대 식별자를 붙이도록 하라. 세대가 **이름과
  header 양쪽**에 있어야 한다. 기존 경로 덮어쓰기는 FATAL이다.
- 공용 reader의 `expected_iteration`을 **필수 keyword-only 인자**로 바꿔라.
  소비자가 생략하면 `TypeError`로 막혀야 한다. 기본값 10은 규율 위반이다.
- `iteration`을 두 필드에 복제하지 마라. `field_generation`은 독립적으로 얻은
  metadata여야 한다. 독립 계보를 확보할 수 없으면 그 사실을 명시하고, 복제로
  위장하지 마라.

## 5. 산출물과 규율

- 수리된 패치를 **새 파일**로 내라: `patches/stage32_rung1_readonly_lambda_v2.patch`.
  v1을 덮어쓰지 마라. 두 sha256을 모두 보고하라.
- 패치를 트리에 적용하지 마라. commit 하지 마라.
- 무거운 연산·GPU·모델 런 금지. 빌드와 fixture 자기검사까지다.
- F1~F4 각각에 대해 **무엇을 어떻게 고쳤고 어느 시험이 그것을 잡는가**를
  파일:줄로 제시하라. "고쳤다"만으로는 접수하지 않는다.
- 리뷰가 지적하지 않은 부분을 함께 바꾸지 마라. 수리 범위를 넓히지 마라.

전체 보고는 `docs/CODEX_STAGE32_RUNG1_REPAIR.md`. `-o` 요약에는 F1~F4 각각의
수리 요지 한 줄, 새 패치 sha256, 새로 추가한 음성 대조와 그 시연 결과, 빌드
결과, 남은 미해결만 담아라.
