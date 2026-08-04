# Stage 3.2 Rung 1 v4 수행 보고

상태: **측정 대상 교체 완료 / v4 패치 생성 / fixture·격리 CPU build PASS / 모델·GPU 런 미실행 / 빌드 격리 규율 1회 위반**

기준 문서:

- `docs/CHARTER_STAGE32_RUNG1_V4.md`
- `docs/AUDIT_STAGE32_RUNG1_EPSILON_DISCREPANCY.md`
- `patches/stage32_rung1_expected_changes_v2.txt`

## 산출물과 불변 계보

- v3 `patches/stage32_rung1_readonly_lambda_v3.patch` SHA-256:
  `b41e991c5ff947e488c85b678b96bfc1f5ba4f88d78146e3978ab262e1728c30`
- v4 `patches/stage32_rung1_readonly_lambda_v4.patch` SHA-256:
  `7cba853937394b28c7f4d2fc54a0bf1b4de14b9276c456f4a4101d1fe2a2a644`
- 은퇴 사전등록 `patches/stage32_rung1_expected_changes.txt` SHA-256:
  `e3c5c186a4617946b697b5368cf697d6329967d63e8c6f641a10470297515ed1`
- 새 정본 `patches/stage32_rung1_expected_changes_v2.txt` SHA-256:
  `5abec18ccc9edc0232ae68de73c8a549c405d28309c6adffab0372304d11dde1`

v3와 은퇴 사전등록은 수정하지 않았다. v4는 현재 작업 트리에
`git apply --check --whitespace=error-all`로 cleanly 적용 가능하며 실제 트리에 적용하거나
commit하지 않았다. 모델 런과 GPU 빌드/실행은 하지 않았다.

## 1차 측정량 교체

LCMFR101 스키마를 v3으로 올리고 행에 raw production `chi_es`, `chi_tot`,
`lambda_star`, `rho_local`과 primary 상태를 싣는다
(`patches/stage32_rung1_readonly_lambda_v4.patch:32-40`). writer는 각 선택 행의 셀
인덱스에서 production 배열을 직접 읽고

```text
rho_local = (chi_es / chi_tot) * lambda_star
```

를 계산한다(`patches/stage32_rung1_readonly_lambda_v4.patch:653-660`, `:753-760`).
`radeq_line_local_response`, Sobolev `beta`, log-domain logistic, line2k는 이 계산에
들어가지 않는다. checker는 payload의 세 production 배열로만 식을 다시 계산해 bitwise
동일성을 요구한다(`:86-93`). 구 정의를 주입한 음성 대조가 이 지점에서 FAIL한다
(`:288-301`).

`chi_tot==0`이면 `rho_local`에 대체값을 넣지 않는다. 행의 `primary_status`를
`UNDEFINED_CHI_TOT_ZERO`로 쓰고 rho에는 NaN을 기록한다(`:655-660`, `:755-760`).
checker는 오직 이 상태와 `chi_tot==0`, NaN의 정확한 조합만 허용한다(`:86-90`).
fixture의 별도 양성 대조는 undefined 행 1개가 기록됨을 확인했다(`:267-272`).

`lambda_star`는 raw 값을 싣고 `[0,1)` 밖이면 writer가 실제 값을 FAIL 로그에 포함하며
중단한다(`:663-676`); checker도 `<0` 또는 `>=1`을 거부한다(`:70-78`). primary에는
clamp/floor/cap/fallback이 없고 manifest도 이를 0으로 선언한다(`:825`).

## 세대 정합

`stage32_field_generation`과 독립적인 `stage32_lambda_generation`을 추가했다
(`patches/stage32_rung1_readonly_lambda_v4.patch:1146-1149`). 정합 규율은 다음 순서다.

1. 조립 snapshot을 준비하면서 `stage32_field_generation`을 증가시키고, 직전 formal
   solve의 diagonal 세대를 0으로 무효화한다(`:898-905`). 이 조립이 이후
   `chi_es/chi_tot`을 채운다.
2. `cmfgen_solve_J`가 모든 bin의 formal solve를 마치고 `lambda_star` 기록을 완료한
   뒤에만 `stage32_lambda_generation=stage32_field_generation`으로 인증한다
   (`:1103-1111`). 조기 allocation 실패나 다른 solver 경로는 인증하지 않는다.
3. CPU와 CUDA의 dump 호출은 solve 뒤에 있다(`:1115-1123`, `:1187-1189`). writer는
   두 세대가 같지 않으면 payload open 전에 FAIL한다(`:566-584`). 두 값은 binary header와
   manifest에도 함께 기록되고 reader가 다시 동일성을 요구한다(`:52-62`, `:137-142`,
   `:725`).

이를 잡는 `S32_SEED_LAMBDA_GENERATION_DEFECT` fixture는 assembly 37과 lambda 36을
주입하며(`:486`), selftest가 writer 단계 거부를 요구한다(`:303-307`). 실제 출력은
아래 음성 대조 절에 기록한다.

## 2차 per-line view

동일한 `(line,shell,bin)` 행에 `beta`, `eps0_raw`, `eps_prime`, `eps_applied` 네 값을
모두 기록한다(`patches/stage32_rung1_readonly_lambda_v4.patch:34-40`, `:769-780`).
`eps_prime`은 production `radeq_line_eps_phys`에서 읽고, `eps_applied`는 production과
같은 floor-then-cap 비교를 보조 대장용으로만 재현한다(`:543-557`). checker는
`eps_prime = eps0/(eps0+(1-eps0)*beta)`와 applied 비교를 독립 검산한다(`:79-85`,
`:143-154`). manifest는 `eps_applied != eps_prime` 행 수를 싣는다(`:811-816`). 정상
fixture에서는 54행 중 38행이었다.

이 네 열에는 사전등록 판정을 걸지 않았다. 사전등록 요약은 shell 8의 정의된 primary
행만 line-emission-energy로 가중한다(`:181-201`).

## F1/F3/F4/F5 유지와 시험

### F1 — branch-site disposition과 독립 evidence

branch site에서 `acc_w>0`을 포함한 evidence bit를 기록한다
(`patches/stage32_rung1_readonly_lambda_v4.patch:1037-1042`)고 각 실제 분기에서
disposition을 쓴다(`:1049-1084`). checker는 evidence만으로 disposition을 독립 재구성해
`acc_w` 누락을 거부한다(`:110-123`). 실제 대조 출력은
`hard gate FAIL: branch-site disposition mismatch (including acc_w)`였다.

### F3 — 제거 상태 유지

primary 경로에는 clamp/floor/cap/fallback을 넣지 않았다(`:653-676`, `:825`). Sobolev
beta는 보조 view에서만 analytic oracle로 검사하며(`:79-85`), 기존 beta 결함은
`KA-3.2.3 FAIL: Sobolev beta differs from analytic oracle`로 계속 검출됐다.

### F4 — iteration/field generation 규율

reader의 `expected_iteration`과 `expected_field_generation`은 계속 필수 keyword-only다
(`:46-48`, `:203-212`). `.iter%03d`, header/filename 일치, 독립 field generation,
overwrite 거부, SHA tamper 거부를 selftest가 계속 시험한다(`:318-341`). 여기에 독립
lambda generation 동일성 검사를 추가했다(`:52-62`).

### F5 — production `eta_l` 재사용

production이 `eta_line`에 더하는 동일 지역변수 `eta_l`을 한 번 계산하고
(`patches/stage32_rung1_readonly_lambda_v4.patch:981-1000`), selected와 boundary
계측에 그대로 더한다(`:1008-1019`). 에너지 식을 writer에서 다시 계산하지 않는다.
checker의 authoritative/selected/boundary closure는 `:163-178`에 있다.
`S32_SEED_ROW_UNSCALED_DEFECT`가 selected/boundary 누적만 `w*Sl`로 되돌리면 이 closure가
FAIL한다(`:419-439`, `:278-283`).

## 필수 음성 대조 실제 출력

격리 fixture selftest에서 다음 세 주입 결함이 실제로 거부됐다.

1. 구 1차 정의 `(1-eps0)*(1-beta)`:

   ```text
   rho_local production-array identity failure
   ```

2. `chi_es/chi_tot` assembly generation 37과 `lambda_star` generation 36의 교차:

   ```text
   [fixture] eps_phys=1 floor_hits=18 interior_hits=16 cap_hits=29
   [STAGE32-R1][FAIL] invalid state/path or chi/lambda generation mismatch: assembly=37 lambda=36
   ```

3. F5 selected/boundary 누적을 `w*Sl`로 회귀:

   ```text
   authoritative pre-EPAY energy census does not close: residual=-2.7419668496724608e-06 tol=1.139929884932239e-18
   ```

## fixture 사전등록 v2 요약

정상 synthetic fixture의 shell 8, 600--3000 A 행에서 production `eta_l` 에너지 가중으로
계산한 값은 다음과 같다.

| 요약량 | fixture 값 | v2 예측 구간 | fixture 판독 |
|---|---:|---:|---|
| `rho_local` 가중 중앙값 | `0.9603999999999999` | `[0.90,0.98)` | MATCH |
| `1/(1-rho)` | `25.252525252525203` | `[10,50]` | MATCH |

이는 스키마와 배선의 자기검사일 뿐 production 측정이나 사전등록 결과 판정이 아니다.
모델 런 금지 때문에 production 값과 E8 5247.49의 disposition은 미측정이다.

## 빌드와 검증

- `git apply --check --whitespace=error-all patches/stage32_rung1_readonly_lambda_v4.patch`:
  PASS.
- 새 격리 base에 v4를 적용한 결과와 작성 격리본 9개 파일 byte 비교: PASS.
- 격리 `/tmp/stage32_r1_v4_retry.ga1cW5/work`에서
  `make selftest_stage32_rung1`: PASS.
- 같은 격리본에서 `make -B lumina`: PASS. 기존 warning은 있었고 새 compile/link error는
  없었다.
- GPU 빌드/실행 및 모델 런: 미실행.

## 남은 미해결과 규율 위반

1. v3에서 알려진 한계는 그대로다. selected/boundary와 authoritative 양쪽이 동시에
   epsilon을 누락해 같은 `w*Sl`을 기록하면 closure가 닫혀 검출하지 못했다. 실제 출력은
   `NOT DETECTED: closure closes when both owners omit epsilon`이다. 이를 숨기기 위한 ε 식
   복제나 억지 guard를 넣지 않았다.
2. 모델 런 금지로 production payload의 가중 중앙값, `1/(1-rho)`, E8 source composition
   ratio와의 실제 모순 여부는 미해결이다.
3. 빌드 입력을 격리본으로 복사하는 명령과 build를 한 셸에 묶는 과정에서 작업 디렉터리를
   원본으로 둔 실수로, 실제 작업 트리에서 CPU `make -B lumina`가 한 번 실행됐다. 소스
   패치·commit은 없었지만 기존 untracked `lumina` 바이너리를 덮어썼을 가능성이 있다.
   임의 삭제/복구는 하지 않았다. 이후 최종 CPU build는 격리본에서 다시 PASS했다.
