# P-TF 오프라인 게이트-잣대 배터리

작성: 2026-08-01. 구현: `scripts/ptf_gated_metrics.py`. 정본: `docs/FABLE_RELT3_ANALYSIS.md` §5.2의 **통계만 게이트** 보수판.

## 결론

배터리를 읽기 전용 CLI로 구현하고 기존 relT3 계보의 it41–55에 소급 적용했다. CMFGEN 소스, rundir, 보존 상태와 correction은 하나도 변경하지 않았다.

- relT3 P0에서 STEQ가 보존된 it46–50의 게이트 MAXCH는 **3.2054e3–9.3308e3%**다. 비게이트 반환 MAXCH `1.0e7%` 고착을 걷어내고 사전등록 E1의 `1e2–1e4%` 창에 5/5 들어온다.
- 폐기된 full-step probe1 분기에서는 it51–55의 게이트 MAXCH도 **1.1510e5–4.5327e5%**다. 따라서 이 분기의 큰 correction을 terminal sentinel 하나로만 설명할 수는 없다.
- E2의 bulk 물리 지표는 두 구간 모두 유계다. it51–55의 `max |Δln n_e|`는 2.22e-3–3.15e-3, pop-weighted step은 1.23e-2–1.56e-2이며, Fe/Co/Ni의 it50→55 평균전하 변화는 최대 0.0663%다.
- E3는 **미충족**이다. relT3 P0 주인은 주로 d2–20의 Ni VI/Co V이고, probe1의 주인은 d21–31로 옮겨 감쇠하지만 종 총수 대비 1e-15–1e-17인 Ca IV/Co IV superlevel이다. 지정된 `1e-20` 게이트는 terminal 유령을 제거하지만 “물리적으로 큰 준위”까지 보장하지는 않는다.
- 신규 스틴트 398961은 이 보고서 작성 시점에 착지 전이므로 분석하지 않았다. CLI는 해당 rundir가 생기면 같은 명령 형태로 즉시 재실행할 수 있다.

## 1. 범위와 불변 규율

이번 작업은 다음 파일만 읽었다.

- `/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT3/{SCRTEMP,STEQ_VALS,CORRECTION_LINK,OUTGEN,RVTJ,MODEL,MODEL_SPEC,LEVEL_SL_STEQ_LINKS}`
- `/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT3_probe1/`의 동명 파일
- 판정 정본 `docs/FABLE_RELT3_ANALYSIS.md`와 거기에 인용된 CMFGEN 포맷·SOLVEBA 의미론

신규 CMFGEN 런은 실행하지 않았고 소스·rundir를 수정하지 않았다. CLI 자체도 `--output`이 입력 rundir 안을 가리키면 실패하도록 막는다. 분석 산출물은 표준출력 또는 사용자가 지정한 rundir 밖 경로에만 쓸 수 있다.

`relT3_plin`은 별도 P-lin 분기이며 P-TF 소급 대상이 아니다. 398961의 완료 산출물도 아직 없으므로 현재 표에는 기존 relT3와 폐기된 probe1만 들어간다.

## 2. 배터리 정의

### 2.1 게이트 MAXCH

STEQ iteration `N`의 correction을 평가할 때 상태는 correction 적용 전인 SCRTEMP `N-1`을 사용한다. 변수 `j`가 속한 원소의 그 깊이 총수를

\[
N_{Z,d}=\sum_{j\in Z} \mathrm{POPS}_{j,d}
\]

라 두고, 다음 조건의 population 변수만 판정 집합에 포함한다.

\[
\mathrm{POPS}_{j,d}\ge 10^{-20}N_{Z,d}.
\]

통계용 게이트 MAXCH는

\[
\mathrm{MAXCH}_{\rm gate}=100\max_{j,d\;\mathrm{included}}|\mathrm{SOL}_{j,d}|
\]

이다. 전자밀도와 온도 변수는 species population gate의 대상이 아니므로 이 최대값에서 제외한다. SOL의 부호는 CMFGEN 정의 그대로 `+`가 감소, `-`가 증가 제안이다.

대조값은 두 가지다.

1. `ungated raw`: 모든 STEQ 변수의 `100×max|SOL|`.
2. `returned MAXCH`: `solveba_v13.f`의 실제 반환 의미론. 감소가 99.999% 이상이면 `1e7%` sentinel을 쓰며, 활성화되는 경우 NV=10 fallback도 그대로 재현한다.

게이트는 **측정만** 바꾼다. SOL을 0으로 만들거나 population 업데이트를 동결하지 않는다. 따라서 본 결과는 원래 제안과 상태를 사후에 다른 잣대로 읽은 결과다.

### 2.2 물리 지표

각 상태 iteration마다 다음을 계산한다.

- `max |Δln n_e|`: 전 깊이의 iteration 간 전자밀도 최대 로그 변화.
- `popw max`: `Σ_j |N_j(k)-N_j(k-1)| / Σ_j N_j(k-1)`의 깊이별 최대값. 정본 E2의 d26–27 band 확인을 위해 d26과 d27 값도 JSON에 보존한다.
- 원소별 shell-weighted 평균전하: 가중치 `r²|Δr|`. 한 원소 안에서는 원자질량이 상수라 정본의 질량/수 가중 평균전하와 동일하다.
- Si III@d27 분율: 내부 CMFGEN 명칭 `SkIII`의 원소 총수 대비 분율.
- Ca IV@d21 바닥: `CaIV SL1`, 즉 `I(STEQ)=603`의 실제 population과 Ca 원소 내 ion fraction.

CLI의 JSON에는 원소별 평균전하 전부와 step 변화, owner의 pre-step population·종 총수·종분율·대표 level 이름이 포함된다.

## 3. 디코더와 검증

구현은 fable의 `scrtemp_lib.py`/`parse_steq.py`에서 검증된 레이아웃을 독립된 단일 스크립트로 옮겼다. scratchpad 경로에 런타임 의존하지 않는다.

- SCRTEMP: 16,376 B direct-access record, record당 2,047 double, R/V 2 records 뒤에 iteration당 `ceil(NT×ND/2047)=80` records. 파일 크기에서 상태 수를 자동 산출한다.
- 변수 맵: 고정된 1,798개 표를 내장하지 않고 rundir의 `LEVEL_SL_STEQ_LINKS`에서 I(STEQ), ion, SL과 대표 level 이름을 읽는다. 원소 경계의 빈 I(STEQ)는 terminal closure로 복원한다.
- STEQ: `STEQ SOLUTION ARRAY`를 전수 파싱하고 모든 `NT×ND` cell이 실제로 채워졌는지 확인한다.
- iteration 번호: OUTGEN의 `--- iteration N`을 우선 사용한다. 없을 때만 SCRTEMP 말단과 STEQ block 수로 추론한다.

현재 두 rundir의 검증은 다음과 같이 닫혔다.

| rundir | SCRTEMP 상태 | STEQ block | OUTGEN extrema/returned | CORRECTION_LINK | final SCRTEMP↔RVTJ |
|:---|---:|:---|:---|:---|:---|
| relT3 | 50 | it46–50, 5개 | 5/5 일치 | 마지막 block 100행 일치 | 최대 상대오차 4.2e-8 |
| probe1 | 55 | it51–55, 5개 | 5/5 일치 | 마지막 block 90행 일치 | 최대 상대오차 4.2e-8 |

CORRECTION_LINK 값과 STEQ 값은 파일에 인쇄된 자릿수에서 전부 같았다. RVTJ anchor 오차도 정본 [§0.2](./FABLE_RELT3_ANALYSIS.md#02-scrtemp-디코더-독립-재구성)의 fable 검증값과 같다. STEQ_VALS 자체가 약 5 유효숫자로 저장되므로 아래 게이트 MAXCH 정밀도도 그 범위로 제한된다.

중요한 결측이 하나 있다. relT3의 it41–45 STEQ_VALS/OUTGEN은 재발주 때 소실됐고 SCRTEMP 상태만 남았다. 적용 step은 clipping된 상태차이이므로 raw SOL을 역산할 수 없다. 배터리는 이 다섯 iteration의 게이트값을 추정하지 않고 `N/A`로 둔다.

## 4. 소급 결과

### 4.1 iteration별 게이트 잣대

`owner/species`는 correction 적용 전 해당 변수 population을 같은 원소 총수로 나눈 값이다.

| it | 분기 | 게이트 MAXCH (%) | 게이트 주인 | depth | owner/species | 비게이트 raw (%) | 반환 MAXCH (%) | E1 창 |
|---:|:---|---:|:---|---:|---:|---:|---:|:---:|
| 41 | relT3 | N/A | STEQ 미보존 | — | — | — | — | N/A |
| 42 | relT3 | N/A | STEQ 미보존 | — | — | — | — | N/A |
| 43 | relT3 | N/A | STEQ 미보존 | — | — | — | — | N/A |
| 44 | relT3 | N/A | STEQ 미보존 | — | — | — | — | N/A |
| 45 | relT3 | N/A | STEQ 미보존 | — | — | — | — | N/A |
| 46 | relT3 | 9.3308e3 | NkSIX SL1, v1736 | 20 | 1.083e-20 | 1.6483e5 | 1.0000e7 | PASS |
| 47 | relT3 | 3.2054e3 | CoV SL12, v1448 | 9 | 5.214e-8 | 3.2054e3 | 1.0000e7 | PASS |
| 48 | relT3 | 7.2889e3 | NkSIX SL1, v1736 | 20 | 1.102e-20 | 1.4118e5 | 1.0000e7 | PASS |
| 49 | relT3 | 3.3717e3 | CoV SL12, v1448 | 2 | 6.118e-8 | 3.3717e3 | 1.0000e7 | PASS |
| 50 | relT3 | 5.1286e3 | CoV SL12, v1448 | 8 | 5.828e-8 | 5.1286e3 | 1.0000e7 | PASS |
| 51 | full | 4.5327e5 | CaIV SL3, v605 | 21 | 7.486e-15 | 3.8393e7 | 3.8393e7 | FAIL |
| 52 | probe1 Λ | 3.1276e5 | CoIV SL50, v1430 | 31 | 1.778e-17 | 3.1276e5 | 1.0000e7 | FAIL |
| 53 | probe1 Λ | 2.0843e5 | CoIV SL50, v1430 | 31 | 1.956e-17 | 2.0843e5 | 1.0000e7 | FAIL |
| 54 | probe1 Λ | 1.2564e5 | CoIV SL50, v1430 | 30 | 2.934e-17 | 2.8240e5 | 1.0000e7 | FAIL |
| 55 | probe1 Λ | 1.1510e5 | CoIV SL50, v1430 | 30 | 3.228e-17 | 7.4830e5 | 1.0000e7 | FAIL |

해석:

- it46/48의 Ca V SL70 raw spike는 게이트에서 사라지지만, 새 주인 NkSIX SL1은 문턱의 1.08–1.10배에 불과하다. `1e-20` 경계에 결과가 민감하다는 직접 증거다.
- it47/49/50은 게이트와 raw가 같아진다. 그러나 주인은 d2/8/9의 Co V SL12이지 사전등록된 d21–32 실물 크롤이 아니다.
- it51은 raw `3.8393e7%`에서 84.7배 낮아지지만 여전히 `4.5327e5%`다. 게이트 주인은 Ca IV 바닥 SL1이 아니라 SL3이다.
- it52–55에는 Co IV SL50@d30–31이 계속 주인이며 게이트 MAXCH가 3.1276e5→1.1510e5, 즉 2.72배 감쇠한다. 다만 해당 준위의 종분율은 1e-17대다.

### 4.2 iteration별 물리 궤적

`max |Δ<q>|`는 그 iteration에서 여섯 원소 평균전하 step 중 최대 절대값이다. `Si step`과 `Ca step`은 직전 상태 대비 상대 percent다.

| it | 분기 | max \|Δln n_e\| | popw max | depth | max \|Δ<q>\| | Si III frac@d27 | Si step (%) | Ca IV gs@d21 | Ca step (%) |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| 41 | relT3 | 6.471e-3 | 3.387e-2 | 26 | 4.832e-4 | 0.577834 | -3.820 | 8.590921e-2 | +8.166 |
| 42 | relT3 | 5.764e-3 | 3.312e-2 | 26 | 7.090e-4 | 0.555018 | -3.948 | 9.380902e-2 | +9.196 |
| 43 | relT3 | 5.781e-3 | 2.888e-2 | 27 | 6.613e-4 | 0.531378 | -4.259 | 9.995237e-2 | +6.549 |
| 44 | relT3 | 2.951e-3 | 2.137e-2 | 8 | 6.873e-4 | 0.521346 | -1.888 | 1.099476e-1 | +10.000 |
| 45 | relT3 | 3.208e-3 | 2.345e-2 | 3 | 8.251e-4 | 0.518041 | -0.634 | 1.193340e-1 | +8.537 |
| 46 | relT3 | 5.551e-3 | 2.797e-2 | 27 | 7.621e-4 | 0.496111 | -4.233 | 1.237280e-1 | +3.682 |
| 47 | relT3 | 6.903e-3 | 3.713e-2 | 27 | 7.897e-4 | 0.526384 | +6.102 | 1.266662e-1 | +2.375 |
| 48 | relT3 | 5.776e-3 | 2.906e-2 | 27 | 6.346e-4 | 0.502489 | -4.539 | 1.314606e-1 | +3.785 |
| 49 | relT3 | 7.278e-3 | 3.837e-2 | 27 | 6.029e-4 | 0.534056 | +6.282 | 1.344356e-1 | +2.263 |
| 50 | relT3 | 6.870e-3 | 3.717e-2 | 27 | 7.270e-4 | 0.564155 | +5.636 | 1.376357e-1 | +2.380 |
| 51 | full | 2.219e-3 | 1.565e-2 | 10 | 1.808e-3 | 0.562410 | -0.309 | 1.445175e-1 | +5.000 |
| 52 | probe1 Λ | 2.557e-3 | 1.233e-2 | 26 | 1.018e-3 | 0.556907 | -0.978 | 1.479255e-1 | +2.358 |
| 53 | probe1 Λ | 2.764e-3 | 1.414e-2 | 26 | 1.178e-3 | 0.550021 | -1.236 | 1.504159e-1 | +1.684 |
| 54 | probe1 Λ | 2.929e-3 | 1.255e-2 | 34 | 1.224e-3 | 0.543141 | -1.251 | 1.545809e-1 | +2.769 |
| 55 | probe1 Λ | 3.150e-3 | 1.406e-2 | 27 | 1.311e-3 | 0.534876 | -1.522 | 1.588180e-1 | +2.741 |

평균전하 endpoint는 다음과 같다.

| it | Si | S | Ca | Fe | Co | Ni |
|---:|---:|---:|---:|---:|---:|---:|
| 40 | 2.034876 | 2.037522 | 2.000105 | 2.891171 | 2.826049 | 2.761111 |
| 50 | 2.034781 | 2.030640 | 2.000110 | 2.891195 | 2.826052 | 2.761117 |
| 55 | 2.035110 | 2.035443 | 2.000107 | 2.889474 | 2.824180 | 2.759357 |

it40→55에서 Ca IV 바닥은 0.079423→0.158818 cm⁻³로 **1.9996배**가 된다. Si III@d27은 0.496–0.601 범위의 왕복 후 probe1에서 0.5624→0.5349로 단조 감소한다. bulk 지표가 작다는 사실과 미시 자유도가 고정점에 도달하지 않았다는 사실은 동시에 성립한다.

## 5. E1–E3 예비 판정표

| 구간 | E1: 게이트 잣대 개방 | E2: 물리 band | E3: 실물 주인 이동·감쇠 | 예비 판정 |
|:---|:---|:---|:---|:---|
| relT3 P0 it41–50 | **PASS(관측 가능분 5/5)**. it46–50 3.21e3–9.33e3% | **기준 band 성립**. `max|Δln n_e|` 2.95e-3–7.28e-3, popw 2.14e-2–3.84e-2 | **FAIL**. 주인이 d2–20 Ni VI/Co V로 교대하고 monotonic 감쇠 없음 | 통계 붕괴 제거는 성공, 실물 표적 노출은 실패 |
| it51 full | **FAIL**, 4.53e5% | 추적 bulk 지표는 유계지만 정본의 RE ×32 악화를 무효화하지 않음 | Ca IV SL3@d21로 이동했으나 종분율 7.49e-15 | 폐기 분기 유지 |
| probe1 Λ it52–55 | **FAIL**, 3.13e5→1.15e5% | **PASS**. `max|Δln n_e|` 2.56e-3–3.15e-3, popw 1.23e-2–1.41e-2; Fe/Co/Ni 누적전하 변화 ≤0.0663% | **부분/최종 FAIL**. d30–31 Co IV SL50로 고정되고 2.72배 감쇠하지만 종분율 1.78e-17–3.23e-17 | 감쇠 신호는 있으나 “실물 주인” 반증 못 함 |

E2는 이번 보수판이 상태를 바꾸지 않으므로 인과적 A/B 판정이 아니라 **원 궤적이 사전등록 물리 band에 있는지 확인하는 sanity gate**다. 특히 it51의 radiative-equilibrium 후퇴는 이 소형 배터리의 지정 지표 밖이며, 정본 판정을 뒤집지 않는다.

E1과 E3를 함께 보면 `1e-20×종총수`는 필요한 게이트지만 충분한 “실물성” 정의는 아니다. 문턱을 임의로 다시 튜닝하지 않고 정본값을 유지했다. 신규 it51′–55′에서도 owner의 종분율과 depth를 반드시 MAXCH 옆에 읽어야 한다.

## 6. CLI와 398961 착지 후 명령

CLI는 같은 model family의 rundir 하나 이상을 lineage 순서로 받는다. 중복된 SCRTEMP 상태는 앞 rundir를 쓰고, 각 STEQ block은 OUTGEN iteration tag로 합친다. 출력은 Markdown, JSON, CSV를 지원한다.

현재 소급표 전체 재현:

```bash
python3 scripts/ptf_gated_metrics.py \
  relT3=/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT3 \
  probe1=/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT3_probe1 \
  --from-it 41 --to-it 55 --threshold 1e-20 --format markdown
```

기계 판독 JSON 재현:

```bash
python3 scripts/ptf_gated_metrics.py \
  relT3=/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT3 \
  probe1=/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT3_probe1 \
  --from-it 41 --to-it 55 --threshold 1e-20 \
  --format json --output /tmp/ptf_retro_it41_55.json
```

398961의 실제 rundir가 착지한 뒤에는 경로만 바꾼다.

```bash
PTF_398961_RUNDIR=/gpfs/kjhan/cmfgen_runs/ACTUAL_398961_RUNDIR
python3 scripts/ptf_gated_metrics.py \
  "ptf398961=${PTF_398961_RUNDIR}" \
  --from-it 51 --to-it 55 --threshold 1e-20 --format markdown
```

기본값은 SCRTEMP 첫 상태가 global it1이라고 가정한다. 새 파일이 전체 history가 아니라 it50 checkpoint부터만 담는 예외 형식이면 `--state-first-it 50`을 명시한다. STEQ iteration은 가능한 한 OUTGEN tag를 사용하므로 정상 continuation에서는 별도 offset이 필요 없다.

입력 바이트 provenance 재현:

```bash
sha256sum \
  /gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT3/{SCRTEMP,STEQ_VALS,CORRECTION_LINK} \
  /gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT3_probe1/{SCRTEMP,STEQ_VALS,CORRECTION_LINK}
```

| 파일 | SHA-256 |
|:---|:---|
| relT3/SCRTEMP | `3b4cdb118de971a145f5c9ae20dfbe87ef8b8e1a664b3f737044a03f7374ed57` |
| relT3/STEQ_VALS | `46218dfa30a33b00ca18d2916e1e2d89bc2a06b094e7594f746a07ddec81d091` |
| relT3/CORRECTION_LINK | `11cd2bab60db1fa6ff2ddbb5eb5caa40a62a2dd911b506307ee7787ae6ef04b0` |
| probe1/SCRTEMP | `799620df11d863c9910cc916b001b758bd70dffe77c0e691ca03a885e1784114` |
| probe1/STEQ_VALS | `75229d6ca1ffbe6d2e8f80994b30387921b7acc2d25287db019887b6820e644c` |
| probe1/CORRECTION_LINK | `812af8e3978c5c6fd4f75ecbf318c5e5c37ce781948921e1b3bcaa756595af45` |

## 7. 제한과 다음 판독 규칙

1. it41–45의 correction은 원자료 결측이므로 복원하지 않는다. 물리 상태표만 재현 가능하다.
2. 통계 게이트는 solver가 적용한 correction을 바꾸지 않는다. 따라서 “trace가 상태를 끌지 않았다”는 인과 명제는 이 오프라인판 단독으로 검증할 수 없다.
3. 게이트 생존은 물리적 major와 동의어가 아니다. 현재 소급에서 1e-20 문턱 바로 위 변수와 종분율 1e-17 변수도 주인이 됐다.
4. 신규 398961 판정은 E1만 보지 않는다. `owner depth`, `owner/species`, Si III@d27, Ca IV 바닥, 평균전하, `max|Δln n_e|`를 같은 행에서 함께 확인한다.
5. 신규 스틴트에서도 E1이 1e5% 이상이고 owner/species가 실제 population-weighted major라면 정본의 반증 조건을 발동한다. 반대로 owner/species가 다시 1e-17 수준이면 “실체 발산”으로 즉시 재격상하지 않고 게이트의 실물성 부족으로 구분한다.

이번 작업은 구현 파일과 본 보고서만 추가했으며 커밋하지 않았다.
