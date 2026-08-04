# Codex A-S31 — Stage 3.1 구현 보고서

상태: **중단 — KA1 FAIL**  
정본 설계: `docs/CODEX_STAGE3_1_CMF_FIELD_DESIGN_2026-08-01.md`  
차터: `docs/STAGE3_CMF_FIELD_CHARTER_2026-08-01.md`

## 결론

골격 gate는 통과했으나 첫 수치 gate인 KA1이 사전등록 acceptance를 통과하지 못했다. 발주의 “실패 시 정직 기록 후 중단” 규율에 따라 KA3, KA2 Nyström oracle, 무가속 산란 반복에는 진입하지 않았다. 모델/GPU 실행, 물리 clamp/floor, 기존 `src/` 수정, 커밋은 모두 0이다. 기존 변경이 있던 `Makefile`도 건드리지 않았다.

| 단계 | 판정 | 근거 |
|---|---|---|
| 0 골격/API/schema/ray/SC | **PASS** | strict compile, valid/malformed frozen v1, SHA-256, ASan/UBSan, GL, SC gate 통과 |
| 1 KA1 pure absorption | **FAIL** | 세 경우 모두 `p_obs` 창 밖; `chi R=100`은 h4 I L2와 max 오차도 초과 |
| 2 KA3 homologous redshift | **NOT RUN** | KA1 단계 gate 중단 |
| 3 KA2 Nyström oracle | **NOT RUN** | KA1 단계 gate 중단 |
| 4 무가속 coherent 반복 | **NOT RUN** | KA1 단계 gate 중단 |
| redistribution ABI 자리 | **PASS (structure only)** | enum/callback 존재, 호출은 `LCMF_EUNSUPPORTED` |
| parity59 판별 벤치 | **UNRESOLVED-INPUT-1** | authoritative total `chi_nu,eta_nu` capture 부재; C1/C2 eta 추측 금지 |

명시된 발주 순서인 골격→KA1→KA3→KA2→산란 반복을 적용했다. 설계 §9 표의 KA2/KA3 순서와 다른 부분은 이번 발주의 더 구체적인 지시를 우선했다.

## 구현된 범위

- 독립 double-only C ABI, 오류 좌표/항 기록, overflow-checked allocation.
- 엄격 radial/descending uniform-log-frequency validator와 coherent-opacity 범위 검사.
- 평가 shell별 `[0,1]` Gauss-Legendre `mu`와 `p=r sqrt(1-mu^2)` cache.
- full-sphere two-leg 및 inner-core 경로 골격, linear radial source SC, `J/H/K` 결정론 적분.
- `expm1`/급수 기반 small-tau 선형 SC와 진공 exact branch. 물리값 clamp/floor 없음.
- little-endian `LCMFCE01` frozen field reader, payload SHA-256 manifest, redundant eta bitwise audit.
- KA1 C fixture + Python 80-digit `mpmath` exact oracle + h/h2/h4 runner.
- parity bench fail-closed 자리: capture가 없으면 `UNRESOLVED-INPUT-1`, C1/C2 재구성은 `FORBIDDEN`.

Frequency-advection 및 nonzero coherent solve는 단계 gate를 넘지 않았으므로 현재 명시적으로 `LCMF_EUNSUPPORTED`를 반환한다. 이를 구현 완료로 표시하지 않는다.

## KA1 수치표

격자는 사전등록 그대로 `(Nr,Nmu)=(32,8),(64,16),(128,32)`이며, `S=1+0.5r^2`, `R=1`이다. `I`는 식 (12), `J`는 식 (13)의 80-digit adaptive quadrature다. `p_obs`는 fine pair-average restriction을 쓴 discrete L2로 계산했다. 제외 cell은 0이다.

| chi R | grid | I rel L2 | J rel L2 | max scaled | residual |
|---:|---:|---:|---:|---:|---:|
| 1e-3 | 32x8 | 3.4567847845e-4 | 2.2914132495e-4 | 1.7365941999e-6 | 3.5104893630e-9 |
| 1e-3 | 64x16 | 1.2872901099e-4 | 9.1224885624e-5 | 7.7823813845e-7 | 5.9264833279e-8 |
| 1e-3 | 128x32 | 4.8804293214e-5 | 3.5573175374e-5 | 3.0107020563e-7 | 1.8810338682e-7 |
| 1 | 32x8 | 4.7035832132e-4 | 2.9830377380e-4 | 1.4385783217e-3 | 1.3468021319e-9 |
| 1 | 64x16 | 1.7500581562e-4 | 1.1553680879e-4 | 5.9919485044e-4 | 3.0877329129e-8 |
| 1 | 128x32 | 6.6364596396e-5 | 4.3958688679e-5 | 2.4087205732e-4 | 5.4865965715e-7 |
| 100 | 32x8 | 4.6933300169e-4 | 3.0977408171e-4 | 4.6048328788e-3 | 9.8275056040e-11 |
| 100 | 64x16 | 2.1607325137e-4 | 1.3574874962e-4 | 2.3009764070e-3 | 5.7155159695e-10 |
| 100 | 128x32 | 1.0858289314e-4 | 5.2339583028e-5 | 1.1414637337e-3 | 5.2425858876e-9 |

| chi R | p_obs(J) | h4 I/J <=1e-4 | h4 max <=3e-4 | p 창 1.8--2.2 | residual <=1e-4 | 판정 |
|---:|---:|---:|---:|---:|---:|---:|
| 1e-3 | 1.5031936869 | PASS | PASS | **FAIL** | PASS | **FAIL** |
| 1 | 1.5035826886 | PASS | PASS | **FAIL** | PASS | **FAIL** |
| 100 | 0.4640614107 | **FAIL (I)** | **FAIL** | **FAIL** | PASS | **FAIL** |

세 경우 모두 clamp/negative/non-finite count는 `0/0/0`, outer incoming 및 중심 대칭 구성 오차는 0이다. 동일 executable/input 출력 SHA-256은 3회 모두 `8b0b02d4b346affe9c8d0cdf7e9b9d7b3158668194fa142241d62b164c5f866d`였다.

## 실패 원인 분석

`chi R=100`의 최대 오차는 마지막 radial cell `i=127`, GL `m=9`, `r=0.99609375`, `mu=0.20614212137961885`의 incoming intensity다.

```text
I_numeric = 1.253246069822307
I_exact   = 1.2543875335559775
abs error = 1.1414637336706335e-3
```

설계 §2.3은 마지막 shell center부터 outer boundary까지 source를 constant extension하도록 구속한다. h4에서도 이 half-cell의 optical depth가 `100/(2*128)=0.390625`라서 quadratic exact source의 바깥쪽 증가를 충분히 해상하지 못하고, 수치해가 위 좌표에서 낮게 나온다. 이 non-asymptotic boundary layer가 thick case의 max 오차와 Richardson 붕괴의 직접 관측 원인이다.

optically thin/moderate 두 경우도 h4 절대오차 문턱은 통과하지만 joint radial/GL refinement의 관측 차수는 약 1.50에 머문다. outer cells를 임의 제외해도 사전등록 창에 들어오지 않았으며, acceptance 뒤에 norm이나 제외 집합을 바꾸지 않았다. 경계 반-cell 재구성 변경, 고정 grid 변경, 혹은 별도 manufactured boundary 재승인 없이는 이 gate를 정직하게 PASS로 만들 수 없다.

## 패치 사다리와 로그

| rung | 패치 | 기대 변경집합/로그 | 판정 |
|---:|---|---|---|
| 1 | `patches/s31_rung1_skeleton.patch` | `docs/s31_logs/rung1_skeleton.log` | PASS |
| 2 | `patches/s31_rung2_ka1_pure_absorption.patch` | `docs/s31_logs/rung2_ka1_pure_absorption.log` | FAIL / STOP |
| 3 KA3 | 생성 안 함 | KA1 gate에 의해 미진입 | NOT RUN |
| 4 KA2 | 생성 안 함 | KA1 gate에 의해 미진입 | NOT RUN |
| 5 산란 반복 | 생성 안 함 | KA1 gate에 의해 미진입 | NOT RUN |

## 재현 명령

```bash
gcc -std=c11 -O2 -Wall -Wextra -Wpedantic -Werror -Wconversion -Wshadow \
  -Isrc tests/stage31_cmf_skeleton_selftest.c src/lumina_cmf_field.c -lm \
  -o /tmp/stage31_skeleton
/tmp/stage31_skeleton

python3 scripts/run_stage31_cmf_ka.py --ka ka1 \
  --work /tmp/stage31_ka1_repro --output /tmp/stage31_ka1.json
# expected exit: 1, report status: FAIL

python3 scripts/stage31_cmf_field_bench.py \
  --status-json /tmp/stage31_parity_status.json
# expected exit: 3, status: UNRESOLVED-INPUT-1
```

상세 machine-readable 수치는 `docs/s31_results/ka1.json`, 벤치 입력 상태는 `docs/s31_results/parity_bench.json`에 있다.
