# Stage 3.2 ALI — Rung 1 구현 보고

상태: **Rung 1 패치·경량 검증 완료 / 실제 모델 census는 운전자 실행 대기**  
동결 시각: 2026-08-02 (Asia/Seoul)  
정본: `docs/CODEX_STAGE32_ALI_DESIGN.md`  
과업문: `docs/CHARTER_STAGE32_RUNG1.md`

## -o 사전등록 요약

- 기대 변경집합, gate OFF: 기존 산출물은 전부 byte-identical이다. 예외는 0개다.
- 기대 변경집합, gate ON: Stage 3.2 Rung 1 전용 Sobolev 국소응답 덤프 1개와 그 덤프의
  SHA-256/세대 manifest만 추가한다. 그 밖의 산출물은 전부 byte-identical이다.
- 전용 덤프의 각 `(line, shell)` 행은 `tau_sobolev`, 정본의 안정식으로 계산한
  `beta=(1-exp(-tau))/tau`, `Lambda_star=1-beta`, 원시 열화확률
  `eps_l=C/(C+A)`, `rho_local=(1-eps_l)*Lambda_star`, pre-EPAY 선 에너지와
  그 행의 EPAY disposition을 가진다.
- `rho_local` 수치 예측: UV 600–3000 Å에서 `tau_sobolev >= 100`인 유효 행의
  **pre-EPAY 선 에너지 가중치 중 90% 이상**이 `0.99 <= rho_local < 1.0`에 있고,
  에너지 가중 중앙값도 `[0.99, 1.0)`에 있을 것으로 예측한다. 이 구간은
  구현 전에 동결했으며, 측정 이탈은 clamp/floor/cap/fallback으로 숨기지 않고
  발견으로 보고한다.
- 근거: 두꺼운 선은 `beta ~ 1/tau <= 0.01`이고 UV 선의 국소 trapping mode는
  산란 지배(`eps_l << 1`)일 것으로 예상되므로 `(1-eps_l)(1-beta)`가 1에
  접근해야 한다. 다만 이것은 실측 전 가설이지 acceptance를 맞추기 위한 가드가 아니다.
- 하드 게이트 증거: 행의 마지막 열에 실제 coarse-cell EPAY disposition을 쓰고,
  manifest에 disposition별 `(line,shell)` 수와 `eta_pre_epay*dnu` 에너지를 함께
  쓴다. Rung 1 산출값은 EPAY의 `eta_line` 입력이 아니라 EPAY 완료 뒤의 읽기 전용
  side-band 덤프로 직접 기록되므로 어느 disposition에서도 폐기되지 않는다. 판정기는
  행 수·에너지 census가 전체와 닫히고 `rate_shape_replaced`에도 실제 행과 에너지가
  존재하는지 검사한다.
- 예정 패치: `patches/stage32_rung1_readonly_lambda.patch`
- 빌드 결과: 구현 뒤 기록한다.
- 모델 실행이 필요한 명령: 구현 뒤 정확히 한 줄만 기록한다. Codex는 실행하지 않는다.

## 범위 봉인

허용되는 변경은 기본 OFF인 경로에서 기존 상태를 읽어 위 전용 덤프를 만드는 것과,
그 binary/manifest를 검증하는 경량 fixture 및 KA-3.2.3 검사뿐이다. 선원함수,
불투명도, 방출률, 율, population, 수송 배열 또는 solver 갱신은 금지한다. Stage 2의
resolvent/ALI 갱신, Stage 3의 rate 정합, Stage 4의 EPAY 은퇴, Stage 5의
`SRC_NLTE` 소비는 구현하지 않는다.

금지된 수치 처리는 clamp, floor, cap, fallback 및 비유한값 대체 전부다. 입력이나
계산 결과가 정의역을 벗어나거나 비유한이면 artifact를 만들지 않고 실패한다.

## 구현·검증 결과

### 동결 이력과 산출물

구현을 시작하기 전에 위 사전등록 본문을 동결했다. 최초 보고서의 SHA-256은
`84554beae08122646bf39f4f1d6d005ea1e7b024df2e16e9bfc104302b4c1430`이고,
별도 불변 사전등록 파일 `patches/stage32_rung1_expected_changes.txt`의 SHA-256은
`e3c5c186a4617946b697b5368cf697d6329967d63e8c6f641a10470297515ed1`이다.

구현물은 `patches/stage32_rung1_readonly_lambda.patch` 한 파일이며 SHA-256은
`db400a22907f32b126fa9007972de4be8dbf76ad3297b8ff5bee99c04bf65bb8`이다.
패치는 작업 트리에 적용하지 않았고 commit도 만들지 않았다. 모든 적용·빌드·시험은
`/tmp`의 임시 복제본에서만 수행했다.

### Rung 1 구현 범위

- `radeq_line_local_response()`는 기존 Sobolev `tau`, 원시 충돌률 `C`와 자발률 `A`를
  읽어 `beta=-expm1(-tau)/tau`, `Lambda_star=1-beta`, `eps0=C/(C+A)`,
  `rho_local=(1-eps0)*Lambda_star`만 계산한다. 기존의 beta가 들어간
  `eps_eff=C/(C+A*beta)`는 쓰지 않는다.
- gate는 기본 OFF이고, ON일 때만 pre-EPAY `eta_line`과 실제 EPAY 분기 disposition을
  진단 전용 배열에 복사한다. 이 배열은 선원함수, 불투명도, 방출률, 율, population,
  수송 배열 또는 solver 입력으로 되먹임되지 않는다.
- 덤프는 기존 EPAY 조립과 수송 solve가 끝난 뒤 직접 기록되는 side-band이다. 행은
  `(line,shell,bin,lambda,tau,beta,Lambda_star,eps0,rho,eta_pre_epay*dnu,
  disposition)`을 가지며 manifest는 세대, payload SHA-256, disposition별 행 수와
  pre-EPAY 에너지 합, `output_discarded_rows=0`을 기록한다.
- 유효하지 않은 정의역, 비유한값, 율 부재, 세대 불일치, 기존 파일 덮어쓰기 또는
  payload 변조는 즉시 실패한다. clamp, floor, cap, fallback과 비유한값 대체는 없다.
- ALI resolvent/갱신, rate 정합, EPAY 제거, `SRC_NLTE` 소비를 추가하지 않았다.

### 0절 하드 게이트: disposition census

정적 경로와 독립 fixture를 함께 확인했다. 이 덤프는 EPAY가 소비하는 `eta_line`이나
`S_fixed`에 삽입되지 않고 EPAY 이후 별도 파일로 직행한다. 따라서 실제 EPAY 분기가
선 에너지를 교체하더라도 Rung 1 행은 그 분기 표지와 pre-EPAY 에너지를 그대로 가진다.
검사기는 disposition별 부분합이 payload 전체 행 수·에너지와 닫히는지, 특히
`rate_shape_replaced` 행과 에너지가 실제로 남았는지 검사한다.

경량 positive fixture의 census는 다음과 같이 닫혔다.

| disposition | 행 수 | `eta_pre_epay*dnu` 합 |
|---|---:|---:|
| `legacy` | 9 | `2.2096360342235936e-07` |
| `thick_thermalized` | 9 | `2.3727478279870206e-07` |
| `rate_shape_replaced` | 9 | `2.5935248264035173e-07` |
| `scalar_thermalized` | 8 | `2.0406021179375697e-07` |

총 35행이며 `output_discarded_rows=0`이다. 특히 EPAY가 교체한
`rate_shape_replaced` 9행도 Rung 1 payload에 남아 있으므로, Rung 1 출력이 EPAY에
폐기되는 경로가 아님을 disposition census로 입증했다. 과업문이 인용한 실제 선행
계측(job 188932)의 UV 폐기율은 shell 8 BALL 에너지 99.563%, B1--B4의 `s>=5`
에너지 100%였으며, 본 side-band 설계는 바로 그 폐기 분기를 행의 마지막 열로 보존한다.

다만 위 수치는 model-free fixture의 경로 적격성 증거다. 실제 parity59 모델의 Rung 1
payload와 그 `rho_local` 분포는 무거운 GPU/모델 실행 금지 때문에 이 작업에서 만들지
않았다. 실제 모델 acceptance는 아래 한 줄을 운전자가 실행해 생성한 census가 닫힌 뒤에만
확정한다.

### KA-3.2.3와 음성 대조

정상 payload에서 판정기는 모든 행에 대해 독립 analytic oracle의
`beta=-expm1(-tau)/tau`와 일치함을 확인하여 `KA-3.2.3 PASS`했다. 필수 음성 대조는
유한하고 정의역 안인 beta를 의도적으로 `0.5*beta`로 바꾸었다. 판정기는 성공으로
통과시키지 않고 정확히 다음 실패를 냈다.

```text
KA-3.2.3 FAIL: Sobolev beta differs from analytic oracle
```

fixture는 `rho_local` 가중비 0.4527669585, 가중 중앙값
0.98999999999901로 사전등록 구간 밖이었다. 이는 일부러 `eps0` 전 범위를 넣은 합성
fixture의 `OUTSIDE_DISCOVERY`이며 보정하지 않았다. 사전등록 수치 예측의 실제 판정은
production payload에만 적용한다.

### 경량 검증 결과

- `git apply --check patches/stage32_rung1_readonly_lambda.patch`: PASS.
- 임시 복제본에서 `make selftest_stage32_rung1`: PASS. 정상 KA-3.2.3, beta 결함
  음성 대조, 세대 불일치 거부, 덮어쓰기 거부, payload 변조 거부를 모두 확인했다.
- fixture payload SHA-256:
  `7fffacae37d5cd7f2aaa6cde83d3eb90f9629a31ddb7fa795a07499525857588`.
- `python3 -m py_compile`로 두 검사 스크립트: PASS.
- 임시 복제본 CPU `make lumina`: PASS. 기존 경고는 있었으나 새 compile/link error는
  없었다. GPU 빌드와 GPU/모델 실행은 하지 않았다.
- 기존 `selftest_cmf_linepop_dump` 및 roundtrip selftest: PASS.
- gate OFF의 대표 기존 linepop fixture는 원본과 패치 적용 임시 복제본 모두
  SHA-256 `a82edf4e05ff3c0e6728518a7e8dd71ea1ebd656bca125ccca6ce6bd014e5f40`으로
  byte-identical이었다.

### 실제 모델 계측 명령(미실행)

전제는 패치를 driver 배포본에 적용·빌드하고, job-private `RUN_DIR`에서 parity59 인증
환경을 이미 source한 상태다. Codex는 다음 한 줄을 실행하지 않았다.

```bash
LUMINA_STAGE32_RUNG1_DUMP="$RUN_DIR/stage32_rung1_iter10" LUMINA_STAGE32_RUNG1_ITER=10 ./"$LUMINA_BIN" data/tardis_reference_toy06_19p48d_sivcaiv 100000 12 spectrum nlte
```

## -o 최종 요약

- 사전등록 `rho_local`: UV 600--3000 Å, `tau>=100` 유효 행의 pre-EPAY 에너지
  90% 이상이 `[0.99,1.0)`, 가중 중앙값도 `[0.99,1.0)`.
- 하드 게이트: EPAY 이후 직행 side-band이며 35행 census가 완전히 닫혔다.
  `rate_shape_replaced` 9행/`2.5935248264035173e-07`도 출력에 존재했고 폐기 행은 0이다.
- 패치: `patches/stage32_rung1_readonly_lambda.patch`, SHA-256
  `db400a22907f32b126fa9007972de4be8dbf76ad3297b8ff5bee99c04bf65bb8`.
- 빌드/시험: CPU 빌드 PASS, model-free selftest PASS, KA-3.2.3 정상 PASS와 beta 결함
  음성 대조 FAIL을 확인했다.
- 실제 모델: 금지 지시에 따라 미실행. 위 한 줄이 운전자용 production capture 명령이다.
