# Stage 3.2 Rung 1 — F5 수리 보고

상태: **F5 수리 완료 / v3 패치 생성 / 격리 selftest·CPU build PASS / 모델 런 미실행**  
기준 문서: `docs/CHARTER_STAGE32_RUNG1_REPAIR_F5.md`  
고정 사전등록: `patches/stage32_rung1_expected_changes.txt`

## -o 요약

- F5 production 행·경계 에너지: production이 `eta_line`에 더하는 `eta_l`을 한 번만 계산하고 같은 변수를 계측 누적에 재사용하게 했으며, 두 owner 중 하나만 `w*Sl`로 바꾸는 closure 음성 대조가 이를 잡는다.
- F5 fixture ε 구조: 선·셸별 비단위 유한 ε와 production floor/interior/cap 사례를 심고 별도 `eps_phys=0` 양성 경로를 추가했으며, fixture coverage와 ε OFF 독립 `w*Sl` oracle이 이를 확인한다.
- F5 checker·음성 대조: ε ON checker는 production ε 식을 복제하지 않고 authoritative/selected/boundary closure를 판정하며, 양쪽 동시 ε 제거는 검출하지 못함을 그대로 노출했다.
- SHA-256: v2 `60bc65c172f95da5c10a07f37743b1f51b007ab9645b503aa0a7ff1d028655be`; v3 `b41e991c5ff947e488c85b678b96bfc1f5ba4f88d78146e3978ab262e1728c30`.
- 음성 대조 1 실제 출력: `authoritative pre-EPAY energy census does not close: residual=-6.4529279658056094e-07 tol=1.5199065132429856e-19`.
- 음성 대조 2 실제 출력: `authoritative pre-EPAY energy census does not close: residual=6.4529279658056094e-07 tol=1.5199065132429856e-19`.
- 사전등록 가중 판독: 수리 전 v2 `w*Sl` 가중은 에너지 분율 `0.660377358490566`, 중앙값 `0.99`; 수리 후 production `eta_l` 가중은 에너지 분율 `0.8383233532934131`, 중앙값 `0.99`였다.
- 빌드: 격리 복사본에서 `make selftest_stage32_rung1` PASS, CPU `make -B lumina` PASS(기존 warning, compile/link error 없음).
- 남은 미해결: selected와 authoritative 양쪽에서 동시에 ε를 빼면 closure가 닫혀 checker가 검출하지 못한다. 모델 런 금지로 production payload의 동결 `rho_local` 예측 판정도 미측정이다.

## 산출물과 불변 조건

- v2: `patches/stage32_rung1_readonly_lambda_v2.patch`, SHA-256
  `60bc65c172f95da5c10a07f37743b1f51b007ab9645b503aa0a7ff1d028655be`.
- v3: `patches/stage32_rung1_readonly_lambda_v3.patch`, SHA-256
  `b41e991c5ff947e488c85b678b96bfc1f5ba4f88d78146e3978ab262e1728c30`.
- 사전등록: SHA-256
  `e3c5c186a4617946b697b5368cf697d6329967d63e8c6f641a10470297515ed1`.
  문안과 `[0.99, 1.0)` 예측 구간은 변경하지 않았다.
- v2는 덮어쓰지 않았다. v3는 현재 작업 트리에 cleanly 적용 가능한 완전 패치이며
  `git apply --check`가 PASS했다. 패치를 실제 작업 트리에 적용하지 않았고 commit도
  만들지 않았다.
- 빌드와 fixture 실행은 `/tmp/stage32_f5_repair.9TbAGH/final` 격리 복사본에서만
  수행했다. GPU 빌드·GPU 실행·모델 런은 하지 않았다.

## F5-1 — production `eta_l` 직접 재사용

v3는 production line loop에서 `eta_l`을 하나의 지역값으로 만든다
(`patches/stage32_rung1_readonly_lambda_v3.patch:829`). `eps_phys` 분기에서는 기존
production clamp를 거친 `el`로 `eta_l = w*el*Sl`을 계산하고 그 값을 `eta_line`에
더하며(`:830-838`), ε OFF 분기에서는 `eta_l = w*Sl`을 계산해 같은 방식으로
더한다(`:850-852`). 계측은 이 분기나 식을 복제하지 않고, 이미 계산된 동일
`eta_l`만 선택 행과 경계 owner에 더한다(`:856-866`).

따라서 새 계측 clamp/floor/cap/fallback/대체값은 없다. production의 기존 ε
처리 결과를 계측이 읽을 뿐이다. 구조체와 manifest의 정의도 `w_l*S_l`에서
production `eta_l`로 바로잡았다(`:674`, `:976`).

이를 잡는 시험은 다음 두 F5 전용 결함이다
(`patches/stage32_rung1_readonly_lambda_v3.patch:232-233`).

1. `S32_SEED_ROW_UNSCALED_DEFECT`: selected와 boundary 누적만 v2의 `w*Sl`로
   되돌린다. checker 출력은 다음과 같았다.

   ```text
   authoritative pre-EPAY energy census does not close: residual=-6.4529279658056094e-07 tol=1.5199065132429856e-19
   ```

2. `S32_SEED_AUTHORITATIVE_UNSCALED_DEFECT`: authoritative owner만 `w*Sl`로
   되돌린다. checker 출력은 다음과 같았다.

   ```text
   authoritative pre-EPAY energy census does not close: residual=6.4529279658056094e-07 tol=1.5199065132429856e-19
   ```

checker의 closure FAIL 지점은 `patches/stage32_rung1_readonly_lambda_v3.patch:142-149`다.

## F5-2 — ε 결함이 발현되는 fixture

fixture의 `radeq_line_eps_phys`는 더 이상 `-1.0`을 반환하지 않는다. 선마다 다른
기본값과 셸 배율로 항상 유한한 비단위 ε를 만든다
(`patches/stage32_rung1_readonly_lambda_v3.patch:297-306`). fixture가 production의
기존 설정을 모사하는 `eps_floor=0.2`, `eps_cap=0.7`을 적용한 실제 coverage는 다음과
같았다.

```text
[fixture] eps_phys=1 floor_hits=12 interior_hits=7 cap_hits=2
```

이는 fixture 안에서 production clamp 경로를 재현하기 위한 값이며 계측 코드에
새 clamp를 넣은 것이 아니다. 조립 fixture는 먼저 production 대응 `eta_l`을 만든 뒤
authoritative와 selected/boundary 두 owner에 넣는다(`:354-373`).

`S32_FIXTURE_EPS_PHYS=0`은 별도 양성 경로다(`:348-349`, `:220-222`). 이때는
`eta_l=w*Sl`이며, 외부에서 ε OFF임을 명시한 checker의 독립 oracle도 함께 PASS했다
(`:78-95`). 정상 ε OFF fixture 수치는 다음과 같다.

```text
authoritative_energy = 1.1883768067151214e-06
boundary_nonselected_energy = 4.833892931094202e-07
closure_residual = 0.0
```

## checker 한계와 억지 가드 부재

ε ON checker는 production의 `eps_phys` 분기나 clamp를 다시 쓰지 않는다. SHA로
결박된 selected 행 합, assembly의 authoritative pre-EPAY snapshot, boundary owner의
세 양만 독립 census로 닫는다. ε OFF라고 외부에서 명시한 시험에만 기존 `w*Sl`
closed-form oracle을 적용한다(`patches/stage32_rung1_readonly_lambda_v3.patch:78-95`).

`S32_SEED_BOTH_UNSCALED_DEFECT`는 selected/boundary와 authoritative 양쪽을 모두
`w*Sl`로 바꾼다(`:350-352`, `:368-369`). 이 경우 closure가 닫혀 checker가
**검출하지 못했다**. 자기검사는 이 사실을 다음 출력으로 명시한다(`:242-250`).

```text
NOT DETECTED: closure closes when both owners omit epsilon
```

이를 잡기 위한 ε 식 복제, 추가 schema field, clamp 재적용 또는 대체 가드는 넣지
않았다.

## 사전등록 가중 판독의 전후 변화

같은 비단위 ε fixture에서 양쪽 owner를 v2 `w*Sl`로 만든 control의 행 가중치를
수리 전 값으로, 정상 production `eta_l` fixture를 수리 후 값으로 읽었다
(`patches/stage32_rung1_readonly_lambda_v3.patch:242-253`).

| fixture 가중치 | `tau>=100` 중 `[0.99,1.0)` 에너지 분율 | 에너지 가중 중앙값 | 판정 |
|---|---:|---:|---|
| 수리 전 v2 `w*Sl` | `0.660377358490566` | `0.99` | `OUTSIDE_DISCOVERY` |
| 수리 후 production `eta_l` | `0.8383233532934131` | `0.99` | `OUTSIDE_DISCOVERY` |

문안이나 예측 구간을 결과에 맞춰 바꾸지 않았다. 두 fixture 판독 모두 동결된 90%
조건 밖이며, 합성 fixture 결과를 production 예측의 실측 판정으로 대신하지 않는다.

## F1/F3/F4 비회귀와 검증

v2의 F1/F3/F4 코드는 변경하지 않았다. 같은 selftest에서 기존 음성 대조와 세대
규율이 계속 PASS했다.

- F1 `acc_w` 누락 disposition: `hard gate FAIL: branch-site disposition mismatch (including acc_w)`.
- F3 beta 결함: `KA-3.2.3 FAIL: Sobolev beta differs from analytic oracle`.
- F4 reader 필수 세대, iteration/field-generation mismatch, overwrite 거부, payload
  tamper 시험: 모두 예상한 거부 결과.
- v2의 opacity-share와 얇은 선 분자 시험은 ε OFF fixture의 유효한 독립
  `w*Sl` oracle에서 계속 FAIL을 시연했다(`:230-240`).

검증 결과:

- `git apply --check patches/stage32_rung1_readonly_lambda_v3.patch`: PASS.
- 새 격리 복사본에 v3를 적용한 결과와 검증 소스의 바이트 비교: PASS.
- 격리 `make selftest_stage32_rung1`: PASS.
- 격리 CPU `make -B lumina`: PASS. 기존 warning은 있었지만 새 compile/link error는
  없었다.

## 남은 미해결

closure는 두 owner가 서로 독립적으로 같은 production 계측량을 받았는지는 판별하지만,
둘이 공모해 같은 잘못된 `w*Sl`을 기록하면 판별하지 못한다. 발주서 지시대로 이를
숨기거나 잡기 위한 억지 가드를 추가하지 않았다.

또한 모델 런 금지 때문에 production payload의 실제 ε 분포와 동결 사전등록
(`tau>=100` 에너지 90% 이상, 가중 중앙값 `[0.99,1.0)`) 판정은 여전히 미측정이다.
