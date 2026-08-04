# E9 — 예측 검증과 구조 수리 설계

판정일: 2026-08-02 (Asia/Seoul)  
범위: 기존 E7/E8 payload·MC 장부의 오프라인 산술 소비와, 가능할 경우 기존
stage31 CPU 드라이버의 유효장 solve. 신규 모델/GPU 런, 신규 clamp, 물리 수리 구현,
commit 없음.

## 1. 사전등록 — 측정 전에 고정

이 절의 수치와 판독 문턱을 유효장 측정 전에 고정한다. s8 iteration-10 MC terminal
열적 파괴율은

```text
eps_MC = 12,766 / 5,238,790 = 0.0024368222433042742
G_MC   = 1 / eps_MC          = 410.3705154316153
```

이다. E8과 같은 국소·광학적으로 두꺼운 재순환 관계를 대역별로 적용한다.

```text
P_b = (J_old,b / J_CMFGEN,b) * G_MC / G_old,b
G_old,b = 1 / eps_eff(source,b)
```

여기서 하나의 s8 MC terminal 확률을 B1--B4와 BALL에 공통으로 쓴다. 이것은
대역별 MC 파괴율이 측정됐다는 뜻이 아니라, 요청문의 BALL `410/5247` 치환을 각
대역의 기존 이득에 확장한 사전등록 가정이다.

| band [Å] | 기존 `J/CMF` | 기존 `G_b` | 사전등록 `P_b` | 적중 창 (±10%) |
|---|---:|---:|---:|---:|
| B1 1000--1500 | 32.32305790 | 2,686.496447 | **4.93744555** | 4.44370100--5.43119011 |
| B2 1500--2000 | 7.37578857 | 1,631.408853 | **1.85533268** | 1.66979941--2.04086595 |
| B3 2000--2500 | 6.91222871 | 13,410.283591 | **0.21152236** | 0.19037013--0.23267460 |
| B4 2500--3000 | 16.29216747 | 20,178.261318 | **0.33133802** | 0.29820422--0.36447182 |
| BALL 600--3000 | 11.97709747 | 5,247.490389 | **0.93664729** | 0.84298256--1.03031202 |

요청문에서 반올림한 `11.977*(410/5247)=0.936`과 같은 예측이다. 주 판정은 BALL:

- 유효장 산술 측정 `J_eff/CMFGEN`이 0.84298--1.03031이면 **기전 확정(진폭 수준)**.
- 창 밖이면 기전 자체를 즉시 부정하지 않고, E8 관계식의 광학 두께·국소성·공통
  대역 epsilon 가정을 재검토한다.
- B1--B4는 같은 ±10% 창으로 보조 판정하고, 일부만 적중하면 대역 의존 재분배가
  scalar epsilon 치환보다 우선이라는 판독을 사전등록한다.

## 2. 측정

### 2.1 산술 유효장

사전등록을 닫은 뒤, A payload의 각 셀에서 E8 관계식을 그대로 확장했다.

```text
eps_old(q) = eta_fixed(q) / eta_total(q)
G_old(q)   = 1 / eps_old(q)
J_eff(q)   = J_old(q) * G_MC / G_old(q)
           = J_old(q) * eps_old(q) / eps_MC
```

LCMFCE01에는 순수 `chi_line`과 `chi_es`가 직렬화되지 않는다. final-state
`n_e sigma_T`는 capture epoch의 일부 외곽 셀 `chi_coherent`보다 커서 전 payload에서
10,430개 음수 line remainder를 만든다. 이를 0으로 clamp하지 않았다. 대신 각 shell의
payload 내 최소 `chi_coherent`를 capture-epoch line-free `chi_es` 대리값으로 사용했다.
s8에서 이 값은 `4.97809226e-16 cm^-1`이고 final-state `n_e sigma_T`의 0.9983008배다.

```text
chi_es,proxy(s)   = min_b chi_coherent(s,b)
chi_line,proxy    = chi_coherent - chi_es,proxy       >= 0  (실측)
chi_coherent,eff  = chi_es,proxy + (1-eps_MC) chi_line,proxy
eta_coherent,eff  = chi_coherent,eff * J_eff
eta_total,eff     = eta_fixed + eta_coherent,eff
```

이것은 누락된 성분의 유일한 exact 분해가 아니다. exact capture-epoch `chi_es`와
`chi_line`은 **UNRESOLVED**다. 그러나 s8의 전자항 대리값 차이는 0.17%이고 결과는
그보다 훨씬 작은 차이만 보인다.

### 2.2 사전등록 대조

| band | 사전등록 | 산술 `J_eff/CMF` | 재조립 source/CMF | stage31 `J_det/CMF` | `J_det/예측-1` | 판정 |
|---|---:|---:|---:|---:|---:|---|
| B1 | 4.93744555 | 4.93321108 | 4.88417171 | 4.91614286 | -0.4315% | **HIT** |
| B2 | 1.85533268 | 1.85368717 | 1.83336724 | 1.83988084 | -0.8328% | **HIT** |
| B3 | 0.21152236 | 0.21151423 | 0.21143348 | 0.20836087 | -1.4946% | **HIT** |
| B4 | 0.33133802 | 0.33130945 | 0.33108052 | 0.33680469 | +1.6499% | **HIT** |
| BALL | **0.93664729** | **0.93595715** | **0.92790972** | **0.93228813** | **-0.4654%** | **HIT** |

전 대역이 사전등록 ±10% 창 안이다. BALL의 산술 유효장, 재조립 source, formal solve는
예측에서 각각 -0.0737%, -0.9329%, -0.4654%다. 따라서 차터의 판독 규칙에 따라
**산란 재순환 기전은 진폭 수준에서 확정**한다. frozen source를 실제 formal transport에
통과시켜도 국소 관계식의 오차가 0.5% 수준이므로, 이 s8 진폭에서는 광학 두께와
비국소 수송이 우선 재검토 항목이 아니다.

다만 범위를 정확히 제한해야 한다.

- 산술 `J_eff`는 E8 관계식으로 만든 값이므로 그 자체는 대수적이다. 독립성이 더 큰
  부분은 같은 source를 받은 stage31 formal solve가 예측을 유지했다는 것이다.
- 이것은 인구·rate·재분배 행렬까지 다시 수렴시킨 물리 수리 run이 아니다.
- 대역별 적중은 모두 1로 간다는 뜻이 아니다. B1은 여전히 4.92배, B2는 1.84배이고,
  보조 B0도 예측 8.4558배/실측 8.2906배다. scalar epsilon은 BALL 진폭을 닫지만
  스펙트럼 형상을 수리하지 않는다.
- `chi_coherent`만 교체하고 기존 `J`와 `eta_fixed`를 유지한 opacity-only 대조는 BALL
  `11.94875465`다. 즉 11.98→0.93 감소는 opacity를 0.24% 낮춘 효과가 아니라
  `5247→410` 재순환 이득을 바꾼 효과다.

### 2.3 stage31 실행 판독

기존 CPU driver는 진단 payload의 `eta_total,eff`를 frozen source로 한 formal solve를
수행했다. scattering 재수렴이나 population update는 없었다.

```text
transport_residual          = 8.180569987551006e-7
source_iterations           = 1 (frozen source)
clamp                       = 0
solution_negative_excess    = 0
sign_uncertain              = 0
nonfinite                   = 0
```

1208 Å trip은 재발하지 않았고 3회 출력 SHA-256이 모두
`59ea65a2...a473`로 같았다. 기존 인증 guard가 허용하는 sub-truncation은
118,679건, sign-indeterminate sub-truncation은 974,903건이며 원래 bit pattern을
고치지 않았다. `bdf_eta_negative=358,662`도 기록됐지만 solution-negative excess는 0이다.

## 3. 정본 선 처리 설계 — 구현하지 않음

### 3.1 물리 source와 가속기의 분리

선 `l:u->d`의 정본 extinction/emission owner는 population이다.

```text
chi_l(nu) = (h nu_l / 4 pi) [n_d B_du - n_u B_ud] phi_l(nu)
eta_l(nu) = (h nu_l / 4 pi) n_u A_ud phi_l(nu)
S_l       = eta_l / chi_l
          = (2 h nu_l^3 / c^2) /
            [(g_u n_d)/(g_d n_u) - 1]
```

denominator가 0 이하인 inversion/maser 셀은 floor나 clamp로 숨기지 않고 별도 물리
분기로 fail/진단한다. transfer 조립은 다음처럼 바꾼다.

```text
chi_total = chi_cont,abs + chi_es + sum_l chi_l
eta_phys  = eta_cont,thermal + chi_es J + sum_l eta_l
```

즉 `chi_line`은 `chi_coherent`에서 완전히 빠지고 **line extinction**으로 남는다.
선 흡수 후 에너지가 열로 반드시 파괴된다는 뜻은 아니다. 재방출은 population source와
아래 주파수 재분배가 담당한다.

형광을 명시하려면 흡수 에너지 벡터 `a_i`와 energy-normalized 분기 행렬을 둔다.

```text
a_i                 = line-absorption power in input bin i
eta_fluor,j          = sum_i R^E[j,i] a_i
sum_j R^E[j,i] + p_heat[i] + p_outside[i] = 1
```

여기서 `R^E`는 photon-count 확률이 아니라 **흡수 packet energy 중 출력 빈으로 간
비율**이다. 따라서 서로 다른 `nu_i,nu_j` 사이에서 별도 `h nu` 보정 없이 열 합계가
닫힌다. count matrix는 통계 오차용 보조량으로만 보존한다.

population emission과 `R a`를 둘 다 source에 더하면 이중계상이다. 정본 계약은

```text
sum_l eta_l^pop  == eta_primary(coll/recomb/dep) + R^E a
```

를 coupled fixed point의 일치식으로 삼고, formal solve에는 한 번만 넣는다. 구현 선택은
(A) `eta_l^pop=chi_l S_l`를 직접 소비하고 `R`을 응답/검증 operator로 쓰거나,
(B) 우변의 primary+redistribution 분해를 소비하되 수렴 시 population emissivity와
일치시킨다. 추적성이 좋은 B를 권고하지만 이 항등 gate 없이 B를 켜면 안 된다.

물리 잔차는 다음 하나다.

```text
F(J,n,T) = J - Lambda[chi(n,T)] eta_phys(J,n,T) = 0
```

`Lambda*`는 이 잔차의 근사 Jacobian/preconditioner일 뿐이다. `Lambda*`의 diagonal,
band block, damping 계수는 `eta_phys`, `chi`, `R`에 저장하거나 더하지 않는다.
가속기를 바꿔도 수렴한 `J,n,S_l`이 같아야 하며, `Lambda*` on/off의 최종 물리 잔차
동일성이 acceptance다.

### 3.2 자료 구조와 갱신 계약

권고 자료 구조는 shell별 energy-CSR이다.

```text
RedistributionSnapshot {
  schema, model_sha256, atomic_sha256, bin_edges_sha256,
  population_epoch, rate_epoch, source_iteration,
  shell_ptr[n_shell+1], input_ptr[n_active_input+1],
  output_bin[nnz], energy_weight[nnz], count[nnz],
  input_energy[n_active], output_energy[n_active],
  p_heat[n_active], p_outside[n_active], sample_count[n_active]
}
```

- 열은 input bin, 행은 output bin이다. `input_ptr`로 column normalization을 직접
  검사하고, deterministic matvec에는 전치/CSR view를 한 번 생성한다.
- 물리적으로 1000x1000x50 dense double은 400 MB라 가능하지만, 관측 s8은
  92,287/1,000,000 edge만 비영이므로 sparse가 추적·I/O에 낫다.
- inner formal/ALI iteration 중에는 `R`과 population을 동결한다. 외부 nonlinear
  iteration에서 `J -> rates -> n -> S_l/R`가 갱신될 때 epoch를 하나 올리고 다음 inner
  solve가 그 snapshot만 소비한다. 서로 다른 epoch 혼합은 fail-closed한다.
- MC 직접 조달 lane은 packet별 `(shell,input_bin,output_bin,input_energy,output_energy,
  terminal_channel)`을 reduce해 snapshot을 만든다. 표본 부족 열은 identity fallback이나
  smoothing을 하지 않고 **UNRESOLVED/insufficient-sample**로 닫는다.
- 정본 생산에서는 동일 rate owner로 기대 분기율을 결정론적으로 계산하고, MC 표본
  snapshot을 shadow oracle로 쓰는 편이 noise가 없다. MC snapshot을 직접 소비하는
  replay lane도 같은 schema로 유지하여 두 경로를 byte/energy 대조한다.

필수 보존 gate는 다음이다.

1. 각 input column에서 `sum R^E + p_heat + p_outside = 1`.
2. 전 shell에서 `sum input packet energy = sum output + heat + outside`.
3. count matrix와 energy matrix를 분리하고 photon 수 보존을 요구하지 않음.
4. population emissivity와 `primary+R a`의 frequency-integrated 합 일치.
5. 같은 snapshot 재소비의 deterministic hash와, bin projection 전/후 총 에너지 일치.

### 3.3 기존 산출물로 조달 가능한가 — 실측

**부분적으로 YES, 정본에는 아직 NO**다. iteration-11 raw event prefix에서 s8의
600--3000 Å line activation을 다음처럼 실제 복원했다.

| 항목 | 실측 |
|---|---:|
| paired terminals | 1,856,667 |
| active input bins | 305 |
| sparse nonzero bin->bin edges | 92,287 |
| matrix size | 1000 x 1000 |
| packet-energy output/input | 1.00000075032 |
| global relative closure error | 7.5032e-7 |
| input-column closure max `abs(ratio-1)` | 8.15e-5 |

따라서 raw `lumina_events.bin`과 `lumina_events_lines.bin`만으로 CSR prototype,
channel sink, count/energy normalization을 모두 만들 수 있다. 기존
`lumina_census_emission.csv`는 출력 파장/채널만 20개 bin으로 합치며 입력 빈이 없고,
`lumina_census_ma_fate.csv`는 shell terminal fate만 주므로 단독으로 full `R`을 만들 수
없다.

제한은 결정적이다. raw archive는 시도 970,557,187건 중 400,000,000건
(41.2134%)인 **비무작위 prefix**이고 iteration 11만 담는다. payload와 같은
iteration-10 full matrix, iteration-11 full matrix, 표본 bias는 모두 **UNRESOLVED**다.
그러므로 이번 CSR은 경로/보존 가능성의 실측 인증이지 production 확률의 정본이 아니다.

## 4. 수리 후 정량 사전등록

### 4.1 내부장

이번 amplitude gate는 이미 다음을 사전등록·적중했다.

```text
s8 600--3000 A J/CMFGEN: 11.9771 -> predicted 0.93665 -> stage31 0.93229
```

그러나 B0/B1 residual 8.29/4.92배는 scalar epsilon이 주파수 이동을 만들지 못함을
보인다. 정본 `R` 수리 acceptance는 BALL뿐 아니라 각 대역과 emergent color를 본다.

### 4.2 emergent energy ledger

기지 결함은 emergent UV 42.9% 대 CMFGEN 23.8%, blue 5.8% 대 14.5%다. 보존적
수리의 정량 목표를 다음처럼 고정한다.

| 점유율 | 현재 | 수리 목표 | ±10% acceptance |
|---|---:|---:|---:|
| UV | 42.9% | **23.8%** | 21.42--26.18% |
| blue/optical diagnostic | 5.8% | **14.5%** | 13.05--15.95% |

UV는 19.1 percentage point, 현재 UV 에너지의 44.52%가 감소해야 한다. 총 에너지
보존 시 non-UV는 57.1→76.2%, 즉 상대 +33.45%다. blue 증가가 8.7 point이므로 나머지
10.4 point는 red optical/NIR/기타 non-UV로 가야 한다. bolometric 합이 줄어 UV를
맞추는 결과는 실패다.

event prefix의 단일 s8 UV 상호작용 destination은 다음이다.

| 출력 | energy fraction |
|---|---:|
| UV 600--3000 Å | 96.3854% |
| optical 3000--10000 Å | **3.56746%** |
| EUV 100--600 Å | 0.02575% |
| IR 10000--20000 Å | 0.01903% |
| grid 밖/미매핑 | 0.00234% |

한 번의 형광만으로는 19.1 point 이동을 만들지 못한다. 그러나 같은 분기가 매번
적용되고 UV 출력이 다시 흡수된다는 단순 반복 한계에서는 평균 UV 이탈 시간이
`1/(1-0.963854)=27.67`회이고, 현재 UV의 44.52%를 optical로 옮기는 데 필요한 유효
상호작용은

```text
N = ln(1-0.4452) / ln(1-0.0356746) = 16.22
```

회다. UV를 이탈한 에너지 중 optical 비율은 98.70%다. 따라서 광학 증가의 방향과
필요 크기는 event 통계와 양립한다. 다만 escape와 shell 이동, iteration-11 prefix bias,
population feedback을 뺀 반복 모델이고, event의 3000--10000 Å optical 구간은 역사적
`blue` 진단보다 넓다. 따라서 실제 emergent UV/blue 점유율은 신규 coupled run 전까지
**UNRESOLVED**다. 위 표는 향후 수리 run의 사전등록 acceptance다.

## 5. 결합 고정점 위험과 Stage 4 의존성

**frozen-population transfer probe는 Stage 4 없이 가능하며 이번 E9가 그것이다. 그러나
정본 `J <-> n <-> S_l/R` 고정점은 현 상태에서 Stage 4 없이 승인할 수 없다.** 현재
NLTE/SE 집합 밖인 stage-IV Fe/Co/Ni 선은 population source owner가 완전하지 않고,
이 선을 population 기반으로 바꾸면 fallback source와 macro-atom branch가 다시
갈라진다. 이를 덮고 반복 수렴만 시키는 것은 물리적 고정점이 아니다.

수치 위험도 크다. scalar local 모델에서 `eps_MC=0.00243682`이면 무가속 iteration의
오차 e-fold는 약 `1/eps=410`회이고 `1e-4` 감소에는 약 3,780회가 필요하다.
관측 coarse-bin coherence 파괴 95.1%는 frequency-diagonal spectral radius를 크게
낮출 수 있지만, 공간적으로 두꺼운 선 숲과 population feedback은 남는다.

선행조건을 다음 순서로 둔다.

1. Stage 4의 relevant-ion population/rate ownership을 닫고, 모든 소비자가 같은
   `A/B/C/beta`와 epoch를 사용한다.
2. 작은 고정 population fixture에서 무가속 `Lambda` 반복과 direct/high-precision
   oracle을 먼저 일치시킨다.
3. `R` column/energy 및 `eta_pop == eta_primary+R a` gate를 통과시킨다.
4. 그 뒤 block-ALI 또는 Newton-Krylov를 **가속기로만** 추가한다. `Lambda*` on/off가
   같은 물리 해에 도달해야 한다.
5. 외부 잔차를 `||Delta J||`, population conservation/SE residual, `||Delta S_l||`,
   bolometric energy residual로 동시에 판정한다. 하나만 수렴하면 실패다.
6. full coupled solve가 무가속으로 비현실적으로 느리면 Stage 3.2 ALI와 Stage 4
   population closure를 선행한다. 신규 clamp/damping으로 해를 바꾸지 않는다.

따라서 “Stage 4 없이 가능한가”의 답은 **진단/동결 solve=YES,
물리 정본 고정점=NO (선행조건 미충족)**다.

## 6. 최종 판독

1. **예측 적중:** BALL 산술 0.93596, 재조립 source 0.92791, stage31 0.93229로
   사전등록 0.93665를 모두 ±1% 안에서 재현했다. 차터 규칙상 산란 재순환은
   **진폭 수준에서 확정**이다.
2. **형상은 미수리:** B0/B1이 8.29/4.92배 남고 opacity-only 치환은 11.95배다.
   scalar epsilon 또는 `chi` 축소는 정본 수리가 아니며, 주파수 비대각 `R`이 필요하다.
3. **조달 경로:** 기존 raw event/census로 1000-bin sparse 구조와 에너지 보존은
   실측 가능하다. cap 없는 동시대 matrix 확률은 **UNRESOLVED**다.
4. **구조 수리:** line extinction/population source/redistribution을 하나의 coupled
   물리 항등으로 두고, ALI를 별도 preconditioner로 제한한다. 이번에는 설계만 했고
   source physics 구현은 0건이다.
5. **emergent 기대:** UV -19.1 point와 blue +8.7 point를 보존 gate와 함께
   사전등록했다. 실제 emergent 수치는 coupled run 금지 때문에 **UNRESOLVED**다.

## 7. 재현 명령

아래 명령은 기존 payload/CMFGEN/event archive를 읽고 CPU diagnostic payload/formal
solve를 만든다. Lumina 모델·plasma·GPU transport는 실행하지 않는다.

```bash
E9_RUN=/gpfs/kjhan/lumina_runner2/scratch/emiss_ab2_capture_188766
E9_CMF=/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4

python3 -m py_compile \
  scripts/emiss_e9_prediction_design.py \
  scripts/emiss_e9_jdet_measure.py \
  scripts/emiss_e9_redistribution_matrix.py

python3 scripts/emiss_e9_prediction_design.py \
  --run "$E9_RUN" --cmf-run "$E9_CMF" \
  --out-dir validation/emiss_e9

python3 scripts/cmf_chieta_check.py \
  validation/emiss_e9/emiss_e9_effective_iter10

gcc -std=c11 -O2 -Wall -Wextra -Wpedantic -Werror \
  -D_POSIX_C_SOURCE=200809L -Isrc \
  scripts/stage31_cmf_field_driver.c src/lumina_cmf_field.c -lm \
  -o /tmp/stage31_cmf_field_driver_e9

/tmp/stage31_cmf_field_driver_e9 \
  validation/emiss_e9/emiss_e9_effective_iter10 \
  validation/emiss_e9/emiss_e9_effective_iter10.manifest.json \
  8 16 10020 1 validation/emiss_e9/jdet_effective_s8.tsv

python3 scripts/emiss_e9_jdet_measure.py \
  --payload validation/emiss_e9/emiss_e9_effective_iter10 \
  --jdet validation/emiss_e9/jdet_effective_s8.tsv \
  --prediction validation/emiss_e9/prediction_measurement.csv \
  --cmf-run "$E9_CMF" --out-dir validation/emiss_e9

timeout 60s python3 scripts/emiss_e9_redistribution_matrix.py \
  --run "$E9_RUN" --out-dir validation/emiss_e9

jq '{eps_MC,gain_MC,bands}' validation/emiss_e9/summary.json
jq '{bands,driver_metadata,trip_1208_recurred}' \
  validation/emiss_e9/stage31_summary.json
jq '{paired_terminals,sparse_nonzero_edges,
     energy_conservation_output_over_input,destination,event_archive}' \
  validation/emiss_e9/redistribution_summary.json

sha256sum \
  scripts/emiss_e9_prediction_design.py \
  scripts/emiss_e9_jdet_measure.py \
  scripts/emiss_e9_redistribution_matrix.py \
  validation/emiss_e9/prediction_measurement.csv \
  validation/emiss_e9/stage31_measurement.csv \
  validation/emiss_e9/redistribution_matrix_s8_sparse.csv \
  validation/emiss_e9/redistribution_input_normalization_s8.csv \
  validation/emiss_e9/summary.json \
  validation/emiss_e9/stage31_summary.json \
  validation/emiss_e9/redistribution_summary.json \
  validation/emiss_e9/emiss_e9_effective_iter10 \
  validation/emiss_e9/jdet_effective_s8.tsv
```

주요 산출물:

- `prediction_measurement.csv`: 사전등록/산술/재조립 대조 6대역.
- `stage31_measurement.csv`: CPU formal `J_det` 대조 6대역.
- `redistribution_matrix_s8_sparse.csv`: 92,287개 s8 bin-to-bin edge.
- `redistribution_input_normalization_s8.csv`: 305개 input column 보존 장부.
- 세 JSON: 정의, driver guard, event cap/energy 보존의 기계 판독.
- `emiss_e9_effective_iter10`: source-physics 수리가 아닌 frozen diagnostic payload.

핵심 SHA-256:

```text
a760ba50771b...c1c240b  scripts/emiss_e9_prediction_design.py
7c630694be52...0f964bc  scripts/emiss_e9_jdet_measure.py
e341597f6084...e09415c  scripts/emiss_e9_redistribution_matrix.py
89415d0b1a4a...273339  validation/emiss_e9/prediction_measurement.csv
d6f2776971f9...f95a8  validation/emiss_e9/stage31_measurement.csv
29c432f0638b...086140  validation/emiss_e9/redistribution_matrix_s8_sparse.csv
79720089f957...13f211  validation/emiss_e9/stage31_summary.json
afbeef86fdaa...1010  validation/emiss_e9/redistribution_summary.json
f67c31c0920e...600b2  validation/emiss_e9/emiss_e9_effective_iter10
59ea65a2ac71...a473  validation/emiss_e9/jdet_effective_s8.tsv
```

규율 장부: 신규 모델/GPU 런 0, 신규 clamp/floor 0, 구조 수리 구현 0, commit 0.
