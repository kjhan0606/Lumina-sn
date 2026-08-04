# Stage 3.2 Rung 1 독립 리뷰 REJECT 수리 보고

상태: **F1--F4 수리 완료 / v2 패치 생성 및 경량 검증 PASS / production 모델 계측 미실행**  
기준 문서: `docs/CHARTER_STAGE32_RUNG1_REPAIR.md`, `docs/CODEX_STAGE32_RUNG1_REVIEW.md`  
고정 사전등록: `patches/stage32_rung1_expected_changes.txt`

## -o 요약

- F1: disposition을 EPAY의 legacy/thick/rate-shape/scalar 실제 분기 자리에서만 기록하고, 별도 branch evidence를 체커가 `acc_w`까지 독립 검산하도록 수리했다.
- F2: 선택 선의 production `w_l*S_l`을 조립 루프에서 직접 누적하고, production 얇은 선 분자와 독립 셀 총량·경계 비선택 기여·closure residual census를 봉인했다.
- F3: 양의 `tau<=1e-12` 누락과 `Lambda_star<1`/`rho<1` 강제를 제거하고, `C/(C+A)`는 합을 만들지 않는 log-domain 안정 비율로 바꿨다.
- F4: 실제 이름에 `.iter%03d`를 붙이고, reader 기대 세대를 필수 keyword-only로 만들며, field generation을 조립 스냅샷 독립 카운터로 분리했다.
- 새 패치 SHA-256: `60bc65c172f95da5c10a07f37743b1f51b007ab9645b503aa0a7ff1d028655be`.
- 새 음성 대조: `acc_w` 누락 disposition 재계산은 `hard gate FAIL: branch-site disposition mismatch (including acc_w)`로 FAIL; opacity-share와 얇은 선 분자 결함도 각각 `line energy FAIL`로 FAIL했다.
- 빌드: 격리 복사본 CPU `make -B lumina` PASS, `make selftest_stage32_rung1` PASS, Python bytecode 검사 PASS. GPU/모델 런은 하지 않았다.
- 남은 미해결: production payload와 동결된 `rho_local` 구간의 실제 판정은 금지된 모델 런을 하지 않아 미측정이다.

## 산출물과 불변 입력

- v1: `patches/stage32_rung1_readonly_lambda.patch`, SHA-256
  `db400a22907f32b126fa9007972de4be8dbf76ad3297b8ff5bee99c04bf65bb8`.
- v2: `patches/stage32_rung1_readonly_lambda_v2.patch`, SHA-256
  `60bc65c172f95da5c10a07f37743b1f51b007ab9645b503aa0a7ff1d028655be`.
- 고정 사전등록: SHA-256
  `e3c5c186a4617946b697b5368cf697d6329967d63e8c6f641a10470297515ed1`.
  문안과 `rho_local` 예측 구간은 변경하지 않았다.
- v2는 원본 기준의 완전 패치다. v1을 덮어쓰지 않았고 실제 소스 트리에 어느
  패치도 적용하지 않았으며 commit도 만들지 않았다.

## F1 — 실제 branch-site disposition과 독립 판정

`patches/stage32_rung1_readonly_lambda_v2.patch:151-158`은 disposition과 별개로
`eligible`, `thick`, `epay>=2`, `acc_w>0`, `hot`, 실제 branch 도달을 나타내는
evidence bit를 정의한다. disposition 자체는 조건을 재작성해 선계산하지 않는다.

- rate-shape 분기의 thick `continue` 직전과 실제 replacement 직전에서 각각 1과 2를
  기록한다(`:569-586`).
- scalar 분기의 실제 thick/non-thick 양쪽에서 각각 1과 3을 기록한다(`:594-606`).
- EPAY 대상 분기 자체를 타지 않은 경로는 disposition 0과 `eligible` bit가 없는 별도
  evidence로 기록한다(`:613-621`).

체커는 manifest의 `output_route` 문자열이나 `output_discarded_rows: 0`을 읽지 않는다.
행에 SHA로 결박된 evidence로 기대 disposition을 독립 계산하며, rate-shape 판정에는
반드시 `EV_ACCW`를 포함한다(`:849-862`). 따라서 처분을 branch 밖에서
`epay>=2 && hot`으로 다시 계산하는 결함은 통과할 수 없다.

fixture의 `S32_SEED_DISPOSITION_DEFECT`는 `acc_w==0` evidence인 scalar 행의 처분만
rate-shape로 잘못 재계산한다(`:738-752`). 자기검사에서 writer는 결함 payload를
정상 생성했고 체커가 다음으로 거부했다(`:973-980`).

```text
hard gate FAIL: branch-site disposition mismatch (including acc_w)
```

## F2 — 실제 `w_l*S_l`과 독립 energy census

선택 대역 line map과 전용 누적 배열은 gate ON에서만 준비되고 매 assembly generation에
초기화된다(`:430-458`). production line loop에서 source가 확정되고 production의
`w`가 계산된 바로 그 자리에서 선택 선별 `w*Sl`을 누적한다(`:516-537`). bin
`eta_line`이나 `chi_line`의 몫으로 행 에너지를 배분하는 코드는 없다. 행은 이 누적값에
해당 bin의 `dnu`만 곱한다(`:247-262`, `:329-351`).

얇은 선 oracle은 production과 똑같이 `tau<=1e-6`에서 분자로 `tau`를 사용한다
(`:839-847`). fixture도 정상 경로에서는 같은 piecewise 식을 쓴다(`:726-736`).
`S32_SEED_THIN_NUMERATOR_DEFECT`만 의도적으로 `-expm1(-tau)`를 심고 체커 FAIL을
확인한다. 같은 bin에 서로 다른 `S_l`을 둔 fixture에서
`S32_SEED_OPACITY_SHARE_DEFECT`는 이전의 `eta_bin*w_l/chi_bin` 배분을 재현하며
역시 `line energy FAIL: not production w_l*S_l*dnu`로 거부된다(`:753-759`,
`:973-980`).

독립 census는 다음 세 양을 서로 다른 owner에서 얻는다.

1. 선택 행 총량: branch-site line accumulator의 행 합.
2. authoritative 총량: pre-EPAY cell snapshot `eta_line*dnu`의 window-bin 합.
3. 경계 비선택 기여: production line loop에서 대역 밖 선이 양쪽 경계 bin에 더한
   `w_l*S_l`의 별도 누적 합(`:529-537`).

manifest는 세 총량과 `closure_residual = authoritative - selected - boundary`를
기록한다(`:270-289`, `:381-394`). 체커는 payload 행 합을 다시 만들고 residual
항등식과 binary64 누적 오차 한계 안의 폐합을 검증한다(`:889-904`). 정상 fixture는
다음으로 닫혔다.

```text
authoritative_pre_epay_window_energy = 1.1883768067151214e-06
boundary_nonselected_line_energy     = 4.833892931094202e-07
closure_residual                     = 0.0
```

## F3 — 정확해 누락·거부 가드 제거

- writer의 행 선택은 더 이상 `tau<=1e-12`를 건너뛰지 않는다. window 안의 모든 양의
  유한 `tau`를 beta/rho 행으로 처리한다(`:223-264`). production이 기존 active
  predicate 때문에 조립하지 않은 매우 얇은 선도 행은 남고, energy만 실제값 0이며
  `ASSEMBLED` evidence로 그 사실을 명시한다(`:319-351`, `:839-847`).
- writer와 reader 모두 `Lambda_star==1.0`, `rho_local==1.0`을 정상 binary64 값으로
  허용한다(`:247-258`, `:824-830`). fixture는 `tau=1e-16`과 `tau=2e16`을 포함해
  두 극한의 정상 통과를 검사한다(`:708-710`).
- `eps0=C/(C+A)`는 overflow 가능한 `C`와 `C+A`를 만들지 않는다. 양의 두 rate는
  log-domain logistic으로 비율을 계산하고, 정확히 정의된 0/1 endpoint만 직접
  처리한다(`:104-135`). `0/(0+0)`처럼 정의되지 않은 경우는 대체값 없이 실패한다.

새 clamp, floor, cap, fallback 또는 비유한값 대체는 넣지 않았다. 정의되지 않거나
계산 불가능한 값은 writer가 artifact 생성 전에 명시적으로 FAIL한다.

## F4 — 세대 이름, 필수 reader choke point, 독립 계보

- writer는 base path를 직접 쓰지 않고 실제 payload를 `base.iter%03d`에 쓴다. payload와
  manifest 중 하나라도 이미 있으면 overwrite를 거부한다(`:190-203`). 체커는 파일명의
  세대와 header iteration도 대조한다(`:803-816`).
- 공용 `read_check`는 `expected_iteration`과 독립 `expected_field_generation`을 모두
  필수 keyword-only로 받는다(`:803-816`). CLI도 두 옵션을 `required=True`로 둔다
  (`:922-929`). 생략 시험은 Python `TypeError`를 확인했다(`:968-971`).
- `field_generation`은 더 이상 호출부의 `iteration` 복제가 아니다. 진단 line field가
  assemble될 때마다 독립 카운터가 증가하고(`:430-458`), writer는 이 값을 header와
  manifest에 기록한다(`:299-311`, `:373-394`). CPU/CUDA 호출부는 iteration만
  넘긴다(`:624-632`, `:655-660`). fixture는 iteration 10과 독립 generation 37을
  사용하고, 잘못 기대한 generation 38을 reader가 거부하는 시험을 둔다(`:763-776`,
  `:982-992`).

## 검증 기록

- `git apply --check patches/stage32_rung1_readonly_lambda_v2.patch`: PASS.
- `python3 -m py_compile scripts/stage32_rung1_check.py scripts/stage32_rung1_selftest.py`:
  PASS(격리 적용본).
- `make selftest_stage32_rung1`: PASS. 정상 payload, beta 결함, `acc_w` disposition 결함,
  opacity-share 결함, 얇은 선 분자 결함, 필수 keyword-only, iteration mismatch,
  독립 field-generation mismatch, tamper, overwrite 거부를 모두 확인했다.
- 정상 fixture: 18행, `field_generation=37`, KA-3.2.3 PASS, disposition count
  `legacy=6`, `thick=3`, `rate_shape=6`, `scalar=3`.
- 격리 복사본 `make -B lumina`: PASS. 기존 warning은 있었으나 새 compile/link error는
  없었다.
- GPU 빌드, GPU 실행, 모델 실행은 하지 않았다.

검증 과정의 절차상 이탈을 숨기지 않는다. 첫 CPU build 호출의 working directory를
잘못 지정해 실제 작업공간의 ignored `lumina` 실행파일을 한 번 rebuild했다. 소스에는
패치를 적용하지 않았고 이후 정식 빌드와 모든 selftest는 `/tmp` 격리 복사본에서
재수행했다. rebuild 이전 ignored binary의 바이트는 보존돼 있지 않아 복원하지 못했다.

## 범위와 남은 미해결

추가 코드는 gate ON 진단 배열, writer, reader, fixture에 한정된다. 선원함수,
불투명도, 방출률, rate, population, 수송 상태를 새 코드가 소비하도록 연결하지 않았고
ALI resolvent, rate 정합, EPAY 은퇴, `SRC_NLTE` 소비 등 2단 이후 구현은 없다.

production 모델 census와 동결 예측
(`tau>=100` 에너지의 90% 이상 및 가중 중앙값이 `[0.99,1.0)`)은 모델 런 금지 때문에
여전히 미측정이다. 합성 fixture의 `rho_local` 결과는 production 예측 판정으로
사용하지 않았다.
