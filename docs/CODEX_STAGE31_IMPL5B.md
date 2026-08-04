# Codex A-S31 round 5B — MPFR 인증 재생 + KA3 최종

상태: **rung8 PASS, rung9 KA3 PASS, rung10 FAIL(strict oracle arithmetic), rung11 conditional numeric PASS / overall STOP**  
정본: `docs/CODEX_STAGE31_DESIGN_REV4.md`, 원설계 `docs/CODEX_STAGE3_1_CMF_FIELD_DESIGN_2026-08-01.md` §6  
실행일: 2026-08-01

## 1. 결론

환경 blocker는 해소됐다. `/home/kjhan/local/include/mpfr.h`가 존재하고, strict C11 compile 및 rpath executable이 `/home/kjhan/local/lib/libmpfr.so.6`과 `libgmp.so.10`을 실제로 연결했다. runtime MPFR version은 `4.2.1`이다.

rung8은 생산 double 계산을 바꾸지 않는 별도 MPFR directed-rounding replay로 기존 세 격자를 고정 2048 bit에서 인증했다. `certified sign-uncertain/non-finite/negative`는 전부 0이다. rung9의 신규 `(1024,4096)`은 정본대로 4096 bit에서 다시 인증했고 세 counter가 모두 0이었다.

KA3는 새 공식 triple `(256,1024)/(512,2048)/(1024,4096)`에서 전 항 PASS했다. fine L1은 `2.6403943801e-5`로 등록 중심 `2.64e-5`, 창 `[2.50,2.80]e-5`에 들어왔다. L2, 차수, centroid, area, residual, boundary, clamp 및 certified guard도 모두 기존 문턱을 통과했다.

KA3 gate가 열려 rung10과 rung11을 실행했다. coherent-scattering 생산 solver의 plain fixed point는 damping/ALI 없이 32회에 수렴했고, 모든 수치 acceptance는 통과했다. 그러나 독립 Nyström 구현은 singular primitive를 80 decimal digit로 평가했어도 dense operator 저장과 matvec를 binary64로 수행했다. `Nref=2048/4096` 상대차 `3.6445e-10 < 1e-9`는 통과하지만, 원설계의 **oracle 전체 80-digit arithmetic** 계약은 충족하지 않는다. 이 사실을 숨기지 않고 rung10을 FAIL, oracle에 의존하는 rung11 최종 인증을 conditional로 고정한다.

| rung | 내용 | 판정 |
|---:|---|---|
| 8 | MPFR directed-rounding certificate replay | **PASS** |
| 9 | `(1024,4096)` 포함 KA3 최종 | **PASS** |
| 10 | KA2 Nyström oracle | **FAIL — full 80-digit arithmetic 미충족** |
| 11 | 무가속 coherent scattering | **수치 전 항 PASS / oracle qualification 때문에 최종 PASS 보류** |

## 2. rung8 MPFR 인증

compile/link 실측은 다음 계약을 사용했다.

```text
gcc -std=c11 -O2 -Wall -Wextra -Wpedantic -Werror \
  -Wconversion -Wshadow \
  -isystem /home/kjhan/local/include \
  scripts/stage31_cmf_mpfr_cert.c \
  -L/home/kjhan/local/lib -Wl,-rpath,/home/kjhan/local/lib \
  -lmpfr -lgmp -lm

ldd:
  libmpfr.so.6 => /home/kjhan/local/lib/libmpfr.so.6
  libgmp.so.10 => /home/kjhan/local/lib/libgmp.so.10
```

replay는 binary64 fixture 입력을 `mpfr_set_d` 점구간으로 올린 뒤 BDF, branch-local 3점 Lagrange, formal-integral weight, `exp/expm1`, 사칙연산을 각각 RNDD/RNDU로 평가한다. `tau<0.25` 급수에는 24항 다음 항의 geometric remainder를 외향으로 더한다. 생산 중심 double 배열과 기존 production `src`는 건드리지 않는다.

| grid | bits | certified uncertain | certified non-finite | certified negative | certified min lower | max width |
|---:|---:|---:|---:|---:|---:|---:|
| 128x512 | 2048 | 0 | 0 | 0 | `1.26859066172249716e-24` | `8.45355857491272719e-544` |
| 256x1024 | 2048 | 0 | 0 | 0 | `1.20744714349472628e-24` | `3.99376903758235988e-461` |
| 512x2048 | 2048 | 0 | 0 | 0 | `1.17665071518847714e-24` | `4.81788188902006415e-293` |
| 1024x4096 | 4096 | 0 | 0 | 0 | `1.16159671501369656e-24` | `1.73118362323440330e-572` |

첫 세 격자의 최초 rung8 결과와 rung9 geometric-grid schema 수리 후 재인증 결과는 각각 `mpfr_cert_rung8.json`, `mpfr_cert_rung8_fine.json`에 분리했다. 두 실행 모두 0-counter PASS다.

## 3. rung9 KA3 최종

### 3.1 4096-bin schema 수리

기존 `nu[k]=nu[k-1]*exp(-dx)`는 4096회 binary64 재귀 뒤 인접 `log(nu[k-1]/nu[k])` 편차가 기존 `1e-12` relative schema를 `1.2286e-12`로 넘었다. tolerance를 바꾸지 않고 수학적으로 같은

```text
nu[k] = nu[k-1] / exp(dx)
```

를 사용했다. 이 방식은 4096개 인접 log ratio가 binary64에서 동일하다. domain, `dx`, cell 수, physics 및 acceptance는 바뀌지 않았다. 인증 replay도 같은 binary64 입력으로 재실행했다.

### 3.2 공식 triple 수치

| grid | profile L1 | profile L2 | centroid error | area error | residual |
|---:|---:|---:|---:|---:|---:|
| 256x1024 | `4.2297852432e-4` | `3.8350100059e-4` | `1.3306754518e-8` | `1.3314833277e-8` | `2.6550151753e-11` |
| 512x2048 | `1.0565893313e-4` | `9.5801490424e-5` | `3.3380858078e-9` | `3.3390294475e-9` | `3.1032097242e-11` |
| 1024x4096 | `2.6403943801e-5` | `2.3940548910e-5` | `8.3595148859e-10` | `8.3601323326e-10` | `3.5783219290e-11` |

공식 `p_obs(L2)=2.000591869903567`이다. fine L1 등록 창, L1/L2 `<=1e-4`, centroid/area/residual `<=1e-4`, `p in [1.8,2.2]`, boundary `<1e-12`, clamp/solution-negative 0 및 certified 3-counter 0을 전부 통과했다.

legacy binary64 독립-radius 진단은 acceptance에서 삭제하지 않고 이름을 명확히 바꿔 보존했다. fine에서 legacy sign-uncertain `4,144,614`, legacy non-finite `2,363,058`이지만 MPFR certificate는 각각 0이다. REV4가 지시한 대로 acceptance는 후자의 실제 outward-rounded certificate를 사용한다.

## 4. rung10 KA2 Nyström

구현한 독립 reference는 Gauss-Legendre nodes, `r=r'` logarithmic singularity subtraction, `Nref=2048/4096`, 80-dps analytic singular primitive를 사용한다. 두 reference의 512개 shell-center 상대차는 `3.6445269209e-10`으로 등록 문턱 `<1e-9`를 통과했다.

그러나 operator matrix와 fixed-point matvec storage/arithmetic는 numpy/scipy binary64다. 따라서 다음처럼 수치 convergence와 arithmetic contract를 분리한다.

| 항목 | 실측 | 요구 | 판정 |
|---|---:|---:|---|
| Nref relative difference | `3.6445269209e-10` | `<1e-9` | PASS |
| singular primitive precision | 80 dps | 80 digit | PASS |
| 전체 operator/solve arithmetic | binary64 | 80 digit | **FAIL** |

`Nref` agreement를 full high-precision oracle로 위장하지 않았다. 이 한 항 때문에 strict rung10 verdict는 FAIL이다.

## 5. rung11 plain coherent scattering

solver patch는 `eta_total=eta_fixed+chi_coherent*J_old`를 damping 없이 직접 갱신한다. ALI, Ng acceleration, floor, clamp 또는 freeze는 없다. CPU OpenMP는 서로 독립인 evaluation-radius formal sweeps만 병렬화하며 source iteration sequence를 바꾸지 않는다. `(256,64)`의 32-thread 출력은 직렬 출력과 byte-identical했고, `(128,32)` 2회 SHA-256도 동일했다.

정본 개정 계보의 KA1 공통 grids `(128,32)/(256,64)/(512,128)`을 사용했다.

| grid | J oracle rel L2 | max error | iterations | source residual | transport residual | energy closure |
|---:|---:|---:|---:|---:|---:|---:|
| 128x32 | `1.2964054513e-5` | `2.8934981368e-6` | 32 | `4.0876540514e-13` | `5.6139410444e-8` | `1.0439545437e-5` |
| 256x64 | `3.7746224104e-6` | `9.0581229199e-7` | 32 | `4.0851572460e-13` | `1.2240375716e-6` | `3.7861477207e-6` |
| 512x128 | `1.0791297038e-6` | `2.8254280991e-7` | 32 | `4.0973860214e-13` | `8.1272609432e-6` | `1.2468279151e-6` |

`p_obs(J)=1.7695427052`로 `[1.7,2.3]` 안이다. finest L2/max/source residual/transport/energy, max iterations, clamp/negative/non-finite counter는 모두 PASS다. 다만 J error의 비교 정본인 rung10 oracle가 full 80-digit 계약을 닫지 못했으므로 rung11은 **conditional numeric PASS**이며 최종 certified PASS는 아니다.

## 6. 검증, 규율, 산출물

- 기존 production `src/` 수정 0. `src/lumina_cmf_field.c` SHA-256은 작업 전후 `9a9e781602ed3e959db19705f939ccb8fcd1a1e04220fd310df8e4a2f9a080d0`이다.
- 구현은 `impl_s31_round5b/` 격리본에서 검증하고 순차 patch로 납품했다.
- strict C11 + `-Wconversion -Wshadow` compile PASS, 확장 skeleton/coherent self-test PASS.
- OpenMP 32-thread `(128,32)` 두 번의 TSV SHA-256은 모두 `7a76950c884df92c1f9527a26ce0330981b263342c4fedf5131623bb468a258e`다.
- acceptance 완화, physical clamp/floor/tail 제외, 신규 model/GPU run, commit 모두 0.
- 전체 기계판독 표는 `docs/s31_results/round5b_verdict_table.csv`다.

| patch | SHA-256 | 내용 |
|---|---|---|
| `patches/s31_rung8.patch` | `e639eeb19b7e534650479fc43da2023df75d47bd561f4bfe5f0f4b5031ec1129` | MPFR certificate replay |
| `patches/s31_rung9.patch` | `29fae10176d170dfbf854fac15d8b7e583f70ca6ce939ca51b2a7aa0bf2b9591` | 4096 grid + KA3/certificate merge |
| `patches/s31_rung10.patch` | `1a799979e0772e10be609462cab39da4bef52eec80d305f6072abed3fe4a0f58` | Nyström oracle probe |
| `patches/s31_rung11.patch` | `14d64a65cd437cab15bdb4bf2df0d2bc5da5e1043543f5402c09fe2444f864e8` | plain source iteration + KA2 ledger |

주요 결과:

- `docs/s31_results/mpfr_cert_rung8_fine.json`
- `docs/s31_results/ka3_rev4.json`
- `docs/s31_results/ka2_oracle_rung10.json`
- `docs/s31_results/scattering_rung11.json`
- `docs/s31_results/round5b_verdict_table.csv`

최종 해제 조건은 rung10 dense operator와 solve 전체를 실제 80-digit arithmetic으로 재실행해 같은 `Nref` gate를 닫는 것이다. 그 전에는 KA3 PASS는 유효하지만 KA2/산란의 최종 certified PASS를 선언하지 않는다.
