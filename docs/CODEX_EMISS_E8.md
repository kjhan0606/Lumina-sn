# E8 — 산란 재순환 고리의 해부 (파괴 확률의 실체)

판정일: 2026-08-02 (Asia/Seoul)  
범위: 기존 `emiss_ab2_capture_188766` payload, CMFGEN jnu4/현재 source tree,
기존 MC census/event log의 읽기·산술 소비만 수행. 신규 런, 수송 solve, 신규 clamp,
commit 없음.

## 0. 결론

**s8의 600–3000 Å 장 11.9771×는 payload source 좌표에서 산란 재순환으로
수치상 완전히 닫힌다.** 고정 source는 CMFGEN의 `0.00228248`인데 측정
`eps_eff(source)=1.90567286e-4`, 즉 `S_total/S_fixed=5247.4904`다. 실제 장에
필요한 이득은 `11.9770975/0.00228247768=5247.4106`이며 두 값의 차이는
0.00152%다. `S_total/J_ours=1.00001521`도 같은 폐합을 보인다.

그러나 이것은 `eta_coherent=chi_coherent*J_ours`를 사용한 **동일 시점의
대수 폐합**이지 독립적인 인과 실험은 아니다. 인과 기전의 강한 증거는 별도로 있다.

- s8 BALL opacity의 **97.7713%**가 coherent 채널이다. 이 채널은 전자산란만이
  아니라 `(1-eps_l) chi_line`을 포함한다.
- 같은 iteration-10 MC 레인의 실제 terminal 열적 파괴율은 **0.243682%**로,
  결정론 `eps_eff(source)=0.0190567%`보다 12.787배다.
- 더 중요한 차이는 열적 파괴가 아니라 **형광 분기**다. iteration-11 이벤트
  prefix에서 s8 UV line absorption 1,856,667쌍 중 같은 line 재방출은 3.35784%,
  다른 line 재방출은 96.5952%였다. 1000-bin transfer grid에서 같은 coarse bin에
  남은 것은 4.88359%뿐이고, continuum/thermal terminal까지 합친 **coarse-bin
  coherence 파괴율은 95.1164%**다. 단 이 로그는 970,557,187건 중
  400,000,000건만 저장한 prefix라 비무작위 truncation 표본이라는 제한이 있다.

따라서 두 레인은 같은 “파괴”를 말하지 않는다. 결정론 `eps_eff`는 band source에서
고정 emissivity가 차지하는 몫이고, MC `destroyed/terminals`는 충돌 열화 확률이며,
MC 다른-line 분기는 광자 에너지를 보존하면서 **주파수 coherence를 파괴**한다.
현재 결정론 ALI는 이 마지막 94–98% 규모의 coarse-bin 이탈을 같은 빈의
`chi_coherent*J`로 되돌린다.

**수리 표적은 scalar epsilon 증대나 clamp가 아니라 multi-line redistribution
operator다.** `(1-eps) chi_line J_same-bin`을 그대로 쓰지 말고, 같은 원자
upper-state의 radiative branch matrix와 열적 sink를 분리하여
`eta_scat,b_out = sum_b_in R[b_out,b_in] chi_line,b_in J_b_in` 형태로 조립해야 한다.
line-level 대각은 같은-line 생존이고, coarse-bin 대각에는 같은 빈으로 끝난 형광도
합쳐지며, 비대각은 빈을 건넌 형광 분기다. 별도 sink는 동일한
`C_ul/(C_ul+A_ul beta)` rate owner를 써야 한다. CMFGEN 등가 epsilon의 숫자와
그 배율은 필요한 `ETA/CHI` frequency-depth dump가 없어 **UNRESOLVED**다.

## 1. 입력, epoch, 정의

권위 payload는 다음이다.

```text
/gpfs/kjhan/lumina_runner2/scratch/emiss_ab2_capture_188766/
  emiss_ab_iter10.{A,B,B2}
```

A의 SHA-256은 `ac62eae5...4011`이고 iteration/generation은 10/10,
post-damping이다. A/B/B2의 `chi_total`, `chi_coherent`, `J_producer`는 bitwise
공통이고 A의 `eta_total=eta_fixed+eta_coherent`도 bitwise exact다. 모든 band
산술은 E6/E7과 동일한 exact-edge overlap Hz 가중을 사용했다.

셀의 정의는 요청문 그대로다.

```text
eps_eff(cell) = eta_fixed / (eta_fixed + eta_coherent)
eta_coherent  = chi_coherent * J_producer
```

band에서는 수송 source에 직접 대응하도록 다음을 주 정의로 삼았다.

```text
eps_eff(source,band)
  = integral[(eta_fixed/chi_total) dnu]
    / integral[(eta_total/chi_total) dnu]

G_recycle = 1 / eps_eff(source,band) = S_total,band / S_fixed,band
```

요청문의 literal emissivity 합산도 별도 열로 보존했다.

```text
eps_eff(eta-integral,band) = integral eta_fixed dnu / integral eta_total dnu
```

두 band 정의는 서로 다른 opacity 빈을 합칠 때 가중이 달라질 뿐 셀에서는 같다.
어떤 음수 제거, floor, cap, clamp도 사후 산술에 적용하지 않았다. 생산 조립기 자체는
per-line `eps_l`에 기존 `EPS_FLOOR/EPS_CAP`을 적용한다
(`src/lumina_cmfgen.c:787-792`); 이것은 새 조치가 아니라 payload producer의 일부다.

MC 비교 epoch는 두 종류다.

- `lumina_ma_line_destruct.csv`의 iteration 10은 payload dump 직후 같은 loop의
  transport가 만든 동시대 열적 파괴 장부다. assemble/solve/dump는
  `src/lumina_cuda.cu:7582-7667`, 뒤이은 transport와 counter는
  `src/lumina_cuda.cu:8245-8283`에 있다.
- raw event/census는 final iteration 11만 기록했다. 그러므로 형광 분기는 한
  iteration 뒤의 독립 표본이며 exact iteration-10 branch 비는 **UNRESOLVED**다.

## 2. `chi_coherent` 성분 동정

### 2.1 Lumina 조립 코드

선 opacity는 각 Sobolev line을 coarse bin에 넣어

```text
chi_line[bin] += (1-exp(-tau_l)) nu_l/(c t_exp dnu_bin)
```

로 조립한다(`src/lumina_cmfgen.c:764-785`). 물리 epsilon gate에서는
`eps_l=C_ul/(C_ul+A_ul beta_esc)`를 구하고 `chi_line_th += eps_l*w`,
`eta_line += eps_l*w*S_l`로 thermal share를 따로 쌓는다
(`src/lumina_cmfgen.c:724-731,786-802`).

이 payload의 실제 환경은 `LUMINA_CMFGEN_LINE_EPS_PHYS=1`이었다
(`emiss_ab2_capture_188766/stdout.log:15`, `PARITY59_EMISS_AB2.env:27`). 따라서
legacy 전열화가 아니라 `chi_ln_th=chi_line_th`인 물리 split 분기가 실행됐다
(`src/lumina_cmfgen.c:999-1031`, 특히 `1014-1016`).

continuum combine은 명확하다.

```c
chi_e  = n_e * CM_SIGMA_T;                // 963
chi_a  = chi_bf + chi_ff;                 // 991, true absorption
chi_t  = chi_e + chi_a + chi_ln;          // 997
chi_es = chi_e + (chi_ln - chi_ln_th);    // 1051
```

따라서 payload의 `chi_coherent`(내부 이름 `chi_es`)는

```text
chi_coherent = chi_electron + sum_l (1-eps_l) chi_line,l
```

이다. **전자산란 전용이 아니다.** bound-free/free-free는 `chi_abs`에 들어가고
coherent continuum으로 들어가지 않는다(`src/lumina_cmfgen.c:979-997,1051-1053`).
이 assembler에는 Rayleigh/dust 같은 다른 continuum scattering 항이 없다. 즉 Lumina의
연속 산란은 이 경로상 전자산란만이고, 선의 scattering remainder가 같은 배열에 섞인다.

formal solve도 `r=chi_es/chi_tot`를 만들고
`S=S_fixed+r*J`로 그대로 소비한다(`src/lumina_cmfgen.c:1505-1516`). payload는 solve와
damping 후 `eta_total=chi_tot*S_fixed+chi_es*J`를 기록한다
(`src/lumina_cuda.cu:7636-7665`).

### 2.2 payload 비율

아래 `chi_c/chi_t`는 `integral chi_coherent dnu / integral chi_total dnu`,
`eps_S`는 주 band 정의, `eps_eta`는 literal emissivity 적분 정의다. 전 300개
band-shell 행은 `validation/emiss_e8/band_shell_recycling.csv`에 있다.

| shell | band | `chi_c/chi_t` | `eps_S` | `eps_eta` | `1/eps_S` |
|---:|---|---:|---:|---:|---:|
| 0 | B0 | 97.9420% | 6.66302e-3 | 9.70094e-3 | 150.08 |
| 0 | B1 | 99.9771% | 2.49995e-4 | 1.65205e-4 | 4,000.08 |
| 0 | B2 | 99.9508% | 6.83651e-4 | 4.43798e-4 | 1,462.73 |
| 0 | B3 | 99.9825% | 1.52138e-4 | 1.45571e-4 | 6,572.96 |
| 0 | B4 | 99.9860% | 1.01260e-4 | 9.53336e-5 | 9,875.53 |
| 0 | BALL | 98.6132% | 6.86329e-4 | 1.10831e-3 | 1,457.03 |
| 8 | B0 | 97.2000% | 6.10272e-4 | 9.65638e-4 | 1,638.61 |
| 8 | B1 | 99.8812% | 3.72232e-4 | 3.60406e-4 | 2,686.50 |
| 8 | B2 | 99.9364% | 6.12967e-4 | 7.28567e-4 | 1,631.41 |
| 8 | B3 | 99.9940% | 7.45696e-5 | 5.97145e-5 | 13,410.28 |
| 8 | B4 | 99.9800% | 4.95583e-5 | 7.20524e-5 | 20,178.26 |
| 8 | BALL | **97.7713%** | **1.90567e-4** | 4.04620e-4 | **5,247.49** |
| 20 | B0 | 71.4399% | 1.23430e-2 | 4.49422e-3 | 81.02 |
| 20 | B1 | 99.9426% | 2.68607e-3 | 3.25195e-4 | 372.29 |
| 20 | B2 | 99.4684% | 5.90076e-3 | 1.32165e-2 | 169.47 |
| 20 | B3 | 99.9977% | 1.34724e-4 | 4.67689e-5 | 7,422.56 |
| 20 | B4 | 99.9952% | 7.95023e-5 | 3.24680e-5 | 12,578.26 |
| 20 | BALL | 74.8951% | 1.19074e-3 | 4.08756e-3 | 839.82 |

s8에서는 모든 sub-band가 97.2% 이상 coherent이며 B1–B4는 99.88–99.994%다.
BALL 값이 97.77%로 낮아지는 것은 Hz 폭이 큰 B0가 적분을 지배하기 때문이다.

## 3. CMFGEN 대조

### 3.1 source 정식의 구조

현재 `/gpfs/kjhan/cmfgen_src/cur_cmf/` source tree에서 continuum scattering은
`ESEC+CHI_RAY`로 만들고 total continuum opacity에 더한다
(`new_main/mod_subs/comp_opac.f:166-185`). continuum source는

```fortran
ZETA  = ETA/CHI
THETA = CHI_SCAT/CHI
S_cont = ZETA + THETA*J
```

이다(`comp_opac.f:364-369`). 즉 CMFGEN 쪽 연속 산란에는 전자뿐 아니라 선택적
Rayleigh도 들어갈 수 있다.

하지만 bound-bound line은 Lumina처럼 line opacity의 대부분을 `THETA*J`에 접어
넣지 않는다. 각 line의 population으로

```fortran
CHIL_MAT = const_opac * (n_lower - g_ratio*n_upper)
ETAL_MAT = const_emis * n_upper
```

를 별도로 만든다(`new_main/cmfgen_sub.f:3477-3488`; 같은 생산 helper는
`new_main/mod_subs/set_line_opac.f:294-312`). resonance zone에서 이 둘을
`CHI`와 `ETA`에 독립적으로 더한다(`cmfgen_sub.f:2194-2205`). formal solution의
line source도

```fortran
TCHI   = CHI + CHIL*profile
SOURCE = (ETA + ETAL*profile)/TCHI
```

다(`/gpfs/.../subs/formsol.f:192-201`). 즉 radiative/collisional branching은
다준위 statistical-equilibrium population과 `ETAL/CHIL`에 함축되며, 선 전체를
local same-frequency `(1-eps)J`로 지정하지 않는다.

이 source tree의 관련 SHA-256은 `cmfgen_sub.f=092f8526...43374`,
`comp_opac.f=c065c49f...bb99a`, `formsol.f=bca3416b...417ee`다. 현재 tree가
jnu4를 만든 executable과 bitwise 동일하다는 build manifest는 찾지 못했으므로
원문 인용은 구조 증거이고 exact binary provenance는 **UNRESOLVED**다.

### 3.2 `J_CMFGEN/B(T_e)`와 epsilon 역산 불능

RVTJ의 CMFGEN `T_e(v)`를 같은 shell velocity에 보간해 band 평균을 냈다.
s0의 `J/B`는 B0–B4에서 0.737–0.815, s8은 0.706–0.919이고 BALL은 각각
0.8000, 0.7815다. s20은 일부 UV에서 `B(T_e)`가 극소여서 0.193–93.1로 넓다.

이 수치로 CMFGEN epsilon을 유일하게 역산할 수 없다. 두 준위 가정
`S=(1-eps)J+eps B`에 `J≈S`만 넣으면 `eps(B-J)≈0`이 되어, `S` 또는 line별
`ETAL/CHIL` 없이 epsilon 정보가 사라진다. 현재 jnu4는 `J`, RVTJ는 `T_e`를 주지만
frequency-depth `ETA`, `CHI`, line별 `ETAL/CHIL`을 주지 않는다.

따라서 다음 두 요구값은 정직하게 남긴다.

```text
CMFGEN equivalent eps_eff = UNRESOLVED
our eps_eff / CMFGEN equivalent multiplier = UNRESOLVED
```

## 4. 기전 검증과 macro-atom 비교

### 4.1 payload 내부 상관

`recycle_gain=1/eps_eff`이므로 둘의 log Pearson/Spearman은 정의상 정확히 -1이다.
이것만으로 인과를 주장하지 않았다. BALL 중복을 빼고 B0–B4, CMF mapping이 가능한
220 band-shell을 사용한 독립 field 지표는 다음과 같다.

| 비교 | Pearson | Spearman |
|---|---:|---:|
| `log eps_eff` vs `log(J_ours/J_CMFGEN)` | -0.5693 | -0.5788 |
| `chi_coherent/chi_total` vs `log recycle_gain` | +0.3827 | +0.7081 |

즉 작은 `eps_eff`일수록 실제 CMFGEN 대비 장도 커지는 중간 강도의 음의 상관이 있고,
coherent opacity 비율이 클수록 재순환 이득이 큰 순위 상관도 확인된다. 완전 상관이 아닌
것은 경계장, 기하 희석, depth, distributed source가 함께 변하기 때문이다.

### 4.2 MC의 열적 파괴

MC terminal에서 쓰는 미시 확률은 결정론 per-line thermal split과 같은 형태다.

```text
eps_line = C_ul / (C_ul + A_ul*beta_Sobolev)
```

host table은 `src/lumina_plasma.c:4616-4631`, device lottery와 k-packet sink는
`src/lumina_cuda.cu:4648-4673`이다. 최종 counter의 항등
`rad_deexc = terminals-destroyed`는 50/50 shell에서 정확히 닫혔다
(`src/lumina_cuda.cu:4661-4669`; census 분류 `src/lumina_cuda.cu:4238-4271`).

iteration 10의 실제 terminal count 결과는 다음과 같다.

| shell | terminals | destroyed | MC thermal fraction | BALL `eps_eff(source)` | MC/ours |
|---:|---:|---:|---:|---:|---:|
| 0 | 258,558,461 | 264,509 | 1.02301e-3 | 6.86329e-4 | 1.491 |
| 8 | 5,238,790 | 12,766 | **2.43682e-3** | **1.90567e-4** | **12.787** |
| 20 | 223,214 | 309 | 1.38432e-3 | 1.19074e-3 | 1.163 |

s8 producer banner의 rate-level 참고값도 Co III UV mean 0.0002, Fe III 0.0031,
Ni III 약 0.0000이며, 대표 deepest line은 각각 0.0002/0.0007/약 0이다
(`stdout.log:447-450`). 따라서 BALL source aggregate와 개별 line/terminal 확률은
같은 양이 아니며, shell에 따라 우연히 비슷하거나 12.8배 벌어진다.

### 4.3 형광 분기는 열적 파괴보다 훨씬 크다

event log에서 각 packet의 bb activation(`etype=1`)과 다음 terminal channel을
program order로 연결했다. s8 결과는 다음과 같다.

| absorption band | paired | same line | different line | same coarse bin | coarse-bin coherence destroyed¹ | same broad band emission |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 108,123 | 1.9237% | 98.0226% | 2.4315% | 97.5685% | 10.7239% |
| B1 | 319,504 | 3.6566% | 96.2479% | 4.9805% | 95.0195% | 24.2510% |
| B2 | 885,942 | 2.5125% | 97.4496% | 4.4095% | 95.5905% | 53.1139% |
| B3 | 450,077 | 4.8898% | 95.0795% | 6.2612% | 93.7388% | 33.8131% |
| B4 | 93,021 | 4.6377% | 95.3247% | 5.2504% | 94.7496% | 14.8483% |
| BALL | **1,856,667** | **3.3578%** | **96.5952%** | **4.8836%** | **95.1164%** | **39.0826%** |

¹ `(different coarse bin + continuum/thermal terminal) / paired`, 즉 현재 1000-bin
formal solve에 직접 대응하는 정의다. exact-line coherence 파괴율
`(different line + continuum/thermal)/paired`는 BALL 96.6422%다. BALL의
continuum/thermal terminal은 872건(0.04697%)이다.

이것은 “열로 사라지는 에너지”가 아니라 “같은 UV line/빈으로 즉시 되돌아갈
확률”의 파괴다. CMFGEN의 다준위 `ETAL/CHIL`과 MC macro-atom은 이 비대각 분기를
표현하지만, 현 결정론 source는 선 scattering remainder 전부를 local diagonal
`chi_coherent,b*J_b`로 놓는다. 두 레인은 같은 파괴 물리를 말하지 않는다.

단, 이벤트 파일은 final iteration 11의 970,557,187 attempted record 중 cap 4억
(41.2134%)만 저장한 prefix이며 570,557,187건이 dropped됐다. 따라서 위 branch
비율은 큰-N이지만 unbiased random estimator로 인증할 수 없고, exact 전체 비율은
**UNRESOLVED**다. 열적 `destroyed/terminals`는 cap 바깥에서도 세는 별도 counter라
이 제한을 받지 않는다.

## 5. 12× 판독과 수리 표적

s8 BALL의 amplitude 장부는 다음 한 줄로 닫힌다.

```text
S_fixed/CMFGEN              = 0.00228247768
J_ours/CMFGEN               = 11.9770975
required J/S_fixed gain     = 5247.4106
measured S_total/S_fixed    = 5247.4904
required eps=1/gain         = 1.90570185e-4
measured eps_eff(source)    = 1.90567286e-4
S_total/J_ours              = 1.00001521
```

따라서 “12×를 만들 수 있는 크기인가”에는 **YES — amplitude는 충분하며 사실상
정확히 그 크기**라고 답한다. 다만 “CMFGEN보다 파괴 확률이 몇 배 부족한가”에는
CMFGEN 등가량 부재로 **UNRESOLVED**다. 이 둘을 섞어 `eps`를 임의 조정하면 안 된다.

수리의 정확한 표적은 다음 계약이다.

1. `C_ul/(C_ul+A_ul beta)`는 결정론/MC가 동일한 collision data, real-Upsilon,
   escape probability를 한 owner에서 소비하도록 한다. 기존 floor/cap을 수리로
   추가하지 않는다.
2. 선 absorption 후 upper-state의 `A_uk beta_uk` 및 collisional/k-packet branch로
   frequency redistribution matrix `R[b_out,b_in,s]`를 만든다. line-level에서
   same-line과 different-line을 구분한 뒤 coarse grid에 보존적으로 사영하여,
   bin 대각에는 실제 same-bin 종착만 놓는다.
3. deterministic source를
   `eta_sc,b_out=sum_b_in R[b_out,b_in] chi_line,b_in J_b_in`으로 조립하고,
   thermal sink와 fixed continuum/dep source를 별도 보존한다.
4. `sum_b_out R + P_thermal = 1`과 frequency-integrated energy conservation을
   payload 단계에서 검증한다. CMFGEN 대조는 line `ETAL/CHIL` dump가 생기기 전까지
   숫자 epsilon matching이 아니라 source-function/branch 구조 비교로 제한한다.

남은 후보 순위는 이번 직접 증거로 갱신한다.

| 순위 | 후보 | 상태 |
|---:|---|---|
| 1 | same-bin coherent 대신 macro/CMFGEN형 비대각 선 분기 | **직접 표적** — event prefix에서 93.7–97.6%가 다른 coarse bin/terminal |
| 2 | inner boundary/산란 depth mapping | 여전히 장의 seed와 depth 형상을 바꿀 수 있음 |
| 3 | EPAY post-shape | payload가 post-EPAY만 저장하여 pre/post shape 미분해 |
| 4 | coarse 빈 폭/선 projection | 1208 Å 등 국소 형상에는 영향, BALL amplitude 우선도는 하향 |

## 6. 한계와 UNRESOLVED

- CMFGEN frequency-depth `ETA/CHI`, line별 `ETAL/CHIL` dump가 없어 등가 epsilon과
  배율은 **UNRESOLVED**다.
- 현재 CMFGEN source tree와 jnu4 executable의 binary provenance 연결도
  **UNRESOLVED**다.
- raw fluorescence branch는 iteration 11 truncated prefix다. iteration-10 전체
  branch rate는 **UNRESOLVED**다.
- payload는 `chi_abs`, `chi_line_th`, pure `chi_e`를 직렬화하지 않는다. 따라서
  payload만으로 coherent opacity를 electron과 line-scatter로 exact 수치 분해하는
  것은 **UNRESOLVED**다. 성분 포함 여부는 source로 확정되고, E7의 final-state
  `n_e sigma_T` proxy만 별도 존재한다.
- band `eps_eff(source)`는 transfer source aggregate이지 한 line의 microscopic
  collision probability가 아니다.

## 7. 재현 명령

아래는 기존 파일 소비만 하며 Lumina executable이나 transport solver를 호출하지 않는다.
event archive 7.5 GB를 한 번 순차 소비하므로 현재 노드에서 약 30초가 걸렸다.

```bash
E8_RUN=/gpfs/kjhan/lumina_runner2/scratch/emiss_ab2_capture_188766
E8_CMF=/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4

python3 -m py_compile scripts/emiss_e8_recycling.py
timeout 60s python3 scripts/emiss_e8_recycling.py \
  --run "$E8_RUN" \
  --cmf-run "$E8_CMF" \
  --out-dir validation/emiss_e8

jq '{canonical_shells,correlations,macro_comparison,
     macro_final_census_identity,event_log,s8_amplitude_closure,
     CMFGEN_equivalent_eps,CMFGEN_eps_multiplier}' \
  validation/emiss_e8/summary.json

python3 - <<'PY'
import csv
for row in csv.DictReader(open('validation/emiss_e8/event_fluorescence_branch_iter11.csv')):
    if row['shell'] == '8':
        print(row)
PY

sha256sum \
  scripts/emiss_e8_recycling.py \
  validation/emiss_e8/band_shell_recycling.csv \
  validation/emiss_e8/macro_thermal_destruction_iter10.csv \
  validation/emiss_e8/event_fluorescence_branch_iter11.csv \
  validation/emiss_e8/summary.json
```

산출물:

- `validation/emiss_e8/band_shell_recycling.csv`: 6 band × 50 shell = 300행.
- `validation/emiss_e8/macro_thermal_destruction_iter10.csv`: 50 shell exact counter.
- `validation/emiss_e8/event_fluorescence_branch_iter11.csv`: 6 band × 50 shell = 300행.
- `validation/emiss_e8/summary.json`: 정의, canonical table, correlation, census identity,
  amplitude closure, UNRESOLVED 장부.

입력/소스 인증용 핵심 SHA-256:

```text
Lumina cmfgen assembler  434a41f3...beff  src/lumina_cmfgen.c
Lumina plasma/rates      37eae294...cd7   src/lumina_plasma.c
Lumina CUDA/MC           75c5e3d7...4c51  src/lumina_cuda.cu
MC event records         63e0fdb8...5546  lumina_events.bin
MC line dictionary       cc882433...5a6d  lumina_events_lines.bin
MC destruction ledger    704792a5...d997  lumina_ma_line_destruct.csv
```
