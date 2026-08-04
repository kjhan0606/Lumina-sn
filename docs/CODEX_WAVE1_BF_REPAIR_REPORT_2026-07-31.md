# Codex A Wave 1 — bf 영역 수리 보고서

작성일: 2026-07-31
범위: D-3 stimulated-recombination, neutral bf, B18-ⓐ `eta_bf` spin 일관성,
B형 C59/C61의 bf·free-bound 방출 부분
금지 준수: Git 명령 미사용, GPU 실행 미사용

## Wave 1.1 보수 (C 리뷰 FAIL 2건)

`docs/CODEX_WAVE1_C_REVIEW.md`의 두 FAIL에 따라 다음 계약이 이 문서의
초기 Wave 1 설명을 대체한다.

- Stimulated recombination은 ARTIS `rpkt.cc:733-765`의
  `sigma_bf * target_probability * corrfactor`를 항별로 구현한다. target map은
  이 게이트 단독으로 로드되며, v2 map은 multi-target CSR과 per-route probability를
  저장하여 `sum_target p_target*corrfactor_target`을 축약 없이 계산한다.
  기존 v1 CMFGEN map은 조사된 단일 target route이므로 명시적으로 `p=1`로
  호환된다. `n_e`는 셸별 `clump_factor`를 곱한 `clumpednne`이며 smooth model은
  정확히 1이다. edge/exponent에는 ARTIS `H`, `KB` 값을 사용한다.
- `LUMINA_FIX_BF_MULTI_EDGE`가 환경에 존재하면 `0`도 포함해 그 값이 최종
  결정이다. legacy `LUMINA_KPKT_FB_MULTI`는 새 변수가 **unset일 때만** fallback
  alias다. 두 GPU free-bound 방출점은 선택 continuum의 level을 모두 전달하고,
  하나의 `sigma_bf(nu)*nu^2` photon-number Milne sampler를 공유한다.
- `NEUTRAL`과 `ETA_SPINGATE` 구현은 변경하지 않았다.
- CPU frozen oracle의 unset, 신규 게이트 전부 `=0`, 그리고 적대 조합
  `FIX_BF_MULTI_EDGE=0 + KPKT_FB_MULTI=1`은 s0/s8/s43에서 모두 byte-identical이며
  §4의 SHA-256을 재현했다. Git 및 GPU 명령은 실행하지 않았다.

## 1. 결론

Wave 1 물리 수정은 모두 독립적인 `LUMINA_FIX_*` 기본-OFF 게이트 아래에
구현했다. 새 게이트를 전부 unset한 작업 직전/직후 frozen-cell oracle의 셀별
CSV는 byte-identical이었다. `make cuda`도 성공했다.

`CODEX_CLAMP_PROVENANCE.md`가 가리키는 B형 16건 중 bf/free-bound 직접 항목은
다음 두 건으로 판정했다.

- **C59:** k-packet free-bound 방출을 단일 대표 edge로 치환
- **C61:** `REC_SPINGATE`가 재결합률을 0으로 치환; 이 중 Wave 1 명시 범위는
  B18-ⓐ `eta_bf` 소비지점의 술어 불일치

나머지 14건은 §8에 Wave 1 범위 밖으로 보존했다. 선 방출/선 불투명도
falsifier인 C45/C68-C71은 bf가 아니므로 수정하지 않았다.

## 2. 항목별 diff 골자와 물리 근거

### 2.1 D-3 — `chi_bf` stimulated-recombination corrfactor

`compute_bf_opacity()`의 각 `(level, shell, frequency)` 항에 다음 ARTIS 식을
적용한다.

```text
r_mod = (n_upper / n_lower) (n_e clump_factor) SAHACONST
        (g_lower / g_upper) T_e^(-3/2)
stimfactor = r_mod exp[-h(nu - nu_edge)/(k T_e)]
corrfactor = max(0, 1 - stimfactor)
chi_level = n_lower sigma_bf target_probability corrfactor
```

정본 출처는 `docs/V3_FUNCTION_AUDIT_VERIFICATION_2026-07-31.md` §2와
ARTIS `rpkt.cc:733-765`이다. 상수 `SAHACONST=2.0706659e-16`은 ARTIS
`constants.h`와 동일하다.

upper target은 로드된 `ma_rr_target[level]`을 우선 사용하고, 없으면 현재
Lumina 원자 표현의 `(Z, stage+1)` ground를 사용한다. 현재 CMFGEN target
manifest의 resolved target은 모두 upper-ion ground이므로 이 fallback은 해당
자료에서 항등이다. `LUMINA_BF_NLTE_POPS=1`이고 upper target의 해 인구가
있으면 그 값을 사용하며, 아니면 기존 dilute population 표현을 쓴다.

순흡수 `chi_bf`와 자발 Milne 방출을 분리했다.

- `chi_bf`: `corrfactor`를 곱한 net absorption
- departure-form `eta_bf`: 보정 전 `n_lower sigma_bf`로 spontaneous emissivity
- thermal fallback `eta_bf`: `chi_net B_nu(T_e)`로 Kirchhoff 관계 유지

LTE에서

```text
r_mod = exp[-h nu_edge/(kT)]
corrfactor = 1 - exp[-h nu/(kT)]
```

가 된다. 독립 산술 KA에서 `T=50000 K`, `nu_edge=2e15 Hz`,
`nu=3e15 Hz`일 때 구현식과 해석식은 모두
`0.943839737437635495`, 절대차 `0`이었다.

기존 TF32 bf-GEMM은 이 level×frequency corrfactor를 분리 행렬곱으로 표현할
수 없다. 따라서 D-3 게이트 ON에서 coarse opacity는 정확 CPU 합산을 쓰고,
fine-GEMM은 corrected coarse-grid interpolation을 덮어쓰지 않도록 `-1`
fallback한다. 게이트 OFF의 GEMM 선택 조건은 기존과 같다.

### 2.2 neutral bf — stage 0 복원

기존 `if (stage < 1) continue;`는 이온 전하 0을 photoionization 부재로
오해했다. 전하 0은 free-free 합에서 제외할 근거이지,

```text
X I + h nu -> X II + e
```

를 제외할 근거가 아니다. `LUMINA_FIX_BF_NEUTRAL=1`이면 stage 0도 기존
ionization-energy lookup, per-level edge, CMFGEN/Kramers sigma 경로를 그대로
통과한다. O I → O II가 대표 복원 항이다.

CPU loop와 `lumina_bf_gemm.cu`의 coarse/fine population·threshold 양쪽에 같은
게이트를 연결했다. free-free의 `ion_stage < 1` skip은 물리적으로 맞으므로
변경하지 않았다. Neutral에 CMFGEN sigma가 없을 때의 Kramers residual charge는
`Z_eff=stage+1=1`로 잡아, neutral 복원 arm이 기존 `Z-stage` 오류를 다시
사용하지 않게 했다. 다른 stage의 legacy fallback은 이 게이트가 바꾸지 않는다.

### 2.3 B18-ⓐ / C61 — `eta_bf` REC_SPINGATE 술어 일관성

`LUMINA_FIX_BF_ETA_SPINGATE=1`과 `LUMINA_REC_SPINGATE=1`이 함께 켜졌을 때
level별 `eta_bf` 합산에 공유 함수 `spingate_level_forbidden()`을 적용한다.
즉 알려진 multiplicity에 대해 daughter가 `M_core±1`이 아니면 그 level의
Milne emissivity를 합산하지 않는다. multiplicity 미상은 기존 공유 술어처럼
허용한다.

이 수정은 `docs/VERIFICATION_REGISTERS.md` B18-ⓐ의 “개체수/캐스케이드에서
제외한 재결합이 판정 연속체 `eta_bf`에 남는” 내부 불일치만 닫는다.
`LUMINA_ALPHA_SPINGATE=1` 없이 multiplicity table이 NULL이면 기존 fail-loud
경고와 함께 inert하다.

중요: C61의 더 근본적인 “ground-core spin 선택과 실제 excited target term의
동일시” 문제는 해결하지 않았다. 따라서 `REC_SPINGATE` 자체는 계속
기본 OFF이며 production 정물리로 승격하지 않는다. B18-ⓑ FB-MILNE exact
미게이트와 B18-ⓒ TOPSTAGE_IV 사문/배너 과대도 Wave 1 범위 밖이다.

### 2.4 C59 — free-bound multi-edge 물리 경로

저장소에는 이미 `LUMINA_KPKT_FB_MULTI` 아래에 실제 per-continuum
Milne-weighted CDF와 `C_fb_real=sum(weight_all_continua)`가 구현돼 있었다.
Wave 1은 이를 복제하지 않고 표준 게이트
`LUMINA_FIX_BF_MULTI_EDGE=1`로 연결했다.

free-bound emissivity는 여러 재결합 continuum의 합이므로 한 dominant ion의
대표 Kramers edge로 치환할 수 없다. 새 이름은 기존 per-ion/per-level
Milne cooling weight, top-edge CDF, 전체 `C_fb_real` 합산을 그대로 사용한다.
인증 계보 보존을 위해 `LUMINA_KPKT_FB_MULTI`는 fallback legacy alias로 남겼다.
단, 새 이름이 명시되면 ON/OFF 모두 새 이름이 우선하며 old alias는 읽지 않는다.
old alias만 켠 경우 기존 배너 문자열은 유지된다. 두 GPU 방출점은 같은
level-resolved `sigma_bf*nu^2` Milne sampler를 호출한다.

## 3. 게이트 표

| 게이트 | 기본 | 작동 조건/의존성 | ON 효과 | OFF 불변 근거 |
|---|---:|---|---|---|
| `LUMINA_FIX_BF_STIM_RECOMB` | 0 | `T_e,n_e,n_upper,n_lower`; target map은 게이트가 독립 로드 | ARTIS `sigma*p*corr`, clumped `n_e`, ARTIS H/KB를 `chi_bf`에 적용 | 기존 `n_level*sigma` 대입·합산 그대로; target/clump 자료도 미할당; GEMM 조건도 `!gate`일 때 기존과 동일 |
| `LUMINA_FIX_BF_NEUTRAL` | 0 | stage 0 ionization energy/level 존재 | neutral continuum을 CPU/GEMM에 포함 | 기존 `stage<1` 즉시 skip 그대로 |
| `LUMINA_FIX_BF_ETA_SPINGATE` | 0 | `REC_SPINGATE=1`; 실효에는 multiplicity data 필요 | forbidden level의 `eta_bf` 합산 제외 | 술어 호출이 short-circuit되어 기존 eta 산술 그대로 |
| `LUMINA_FIX_BF_MULTI_EDGE` | 0 | k-packet/free-bound 경로 | 기존 Milne-weighted multi-edge CDF와 공용 GPU Milne sampler 사용 | 명시 `0`이면 alias와 무관하게 single-edge; unset일 때만 legacy alias 검사 |

기존 `LUMINA_KPKT_FB_MULTI=1`은 새 변수가 unset인 launcher에서만 새 C59
게이트 ON과 동등하다. `LUMINA_FIX_BF_MULTI_EDGE=0`은 언제나 명시적 OFF다.

## 4. OFF byte-불변 검증

작업 직전 `lumina_plasma.c` 스냅샷과 현재 파일을 같은
`bench_frozen_oracle.c`로 별도 컴파일했다. 환경은 `env -i`, 새 게이트는 모두
unset, frozen input은 `logs/coevolve_consume_parity50`, model은
`data/tardis_reference_toy06_19p48d_sivcaiv`, 셀은 s0/s8/s43이었다.

셀별 CSV SHA-256은 pre/post가 각각 동일했다.

```text
s0  8cbccb2cac2fb7b860eac45edd8479f36f5f5b010e0dd3708d463eff389332b6
s8  dad29ce6b39a00609f6b63aa06cb85c8fb323212921081d434d8ca5510115767
s43 432952ec471323a7d164a31792c21d117cbc3221af3ac63d753696f57f182112
```

stderr는 양쪽 모두 빈 파일이었다. stdout의 유일한 원시 차이는 서로 다른
임시 output-directory 경로 3줄이었고, 그 경로를 같은 토큰으로 정규화한
stdout은 byte-identical이었다.

이 검증은 bf `chi/eta`, ff, rate matrix, thermal ledger를 포함한 frozen-cell
observer 출력 전체를 덮는다. CUDA cross-binary transport 결정론은 GPU 실행
금지 때문에 주장하지 않는다.

## 5. clamp 처리

Wave 1에서 제거한 clamp는 **0건**이다.

- D-3에 직접 귀속된 A형 downstream clamp가 provenance 문서에 없다.
- C59/C61은 B형 물리 치환/가정이며 upstream 수리 후 제거할 A형 clamp가 아니다.
- ARTIS 식의 `max(0, 1-stimfactor)`는 임의 튜닝 clamp가 아니라 순흡수 계수의
  공식 일부이므로 그대로 구현했다.

따라서 “upstream 수리가 같은 diff에 있는 항목만 clamp 제거” 규율을
위반하지 않았다.

## 6. 빌드

최종 `make cuda` 성공. GPU 실행은 하지 않았다.

첫 시도는 작업 전부터 있던 존재하지 않는 API `cudaMemsetToSymbol` 두 호출에서
중단됐다. 전체 빌드 확인을 위해 `cuda_ma_line_destruct_reset()`의 두 호출을
동등한 all-zero host array의 `cudaMemcpyToSymbol`로 교체했다. 이는 observer
counter reset 배선만 고친 것이며 물리/RNG/게이트 의미는 바꾸지 않는다.

최종 빌드의 유일한 메시지는 기존
`src/lumina_nlte_gemm.cu:75 g_fgemm_nulo set but unused` 경고였다.

## 7. Codex B 검증 지침

### 7.1 공통 OFF 및 oracle

1. `make cuda`를 먼저 통과시킨다. 바이너리는 실행하지 않는다.
2. `make bench_frozen_oracle`로 CPU observer를 만든다.
3. 같은 frozen input에 대해 새 게이트 unset arm과 모두 `=0` arm을 각각
   실행하고 셀별 CSV를 `cmp`한다.
4. 별도 pre-Wave1 소스가 있으면 같은 하니스/컴파일러/플래그로 빌드해 §4의
   세 SHA-256을 재현한다. stdout은 output path만 정규화한다.

### 7.2 D-3 KA 및 oracle

- 위 §2.1 LTE 구성으로 `corrfactor == 1-exp(-h nu/kT)`를 상대오차
  `1e-13` 이내에서 확인한다.
- `r_mod exp[-h(nu-nu_edge)/kT] >= 1` 구성에서 `chi_level==0`이고 음수가
  없음을 확인한다.
- 같은 frozen cell의 OFF/D-3 ON을 비교해 모든 기록 항에서
  `0 <= chi_on <= chi_off`인지 검사한다.
- `CMF_BF_MILNE=2`의 departure-form level에서는 D-3 단독 ON/OFF의
  spontaneous `eta_bf`가 동일해야 한다. thermal fallback은
  `eta=chi_net B` 관계를 검사한다.
- ARTIS oracle은 `rpkt.cc:733-765`의 `r_mod`, exponential, `max` 세 항을
  각각 dump하여 Lumina level/bin과 대조한다.

### 7.3 neutral bf KA

- O I가 실제로 존재하는 frozen cell을 고르고 O I threshold보다 높은
  주파수에서 ion별 기여를 계측한다.
- OFF에서는 stage-0 bf 기여가 정확히 0, ON에서는
  `sum_l n_l sigma_l(nu)>0`이어야 한다.
- O II 이상 이온의 per-level 기여와 ff 기여는 ON/OFF에서 byte-identical이어야
  한다.
- CPU 합산과 bf-GEMM을 각각 실행해 neutral 추가분을 비교한다. GEMM은 TF32이므로
  byte 동일이 아니라 기존 GEMM 오차 예산을 적용한다.

### 7.4 B18-ⓐ spin 일관성 KA

다음 4-arm을 쓴다.

```text
A: REC=0, ETA_FIX=0
B: REC=1, ETA_FIX=0
C: REC=1, ETA_FIX=1, ALPHA_SPINGATE=1
D: REC=1, ETA_FIX=1, ALPHA_SPINGATE=0
```

- C에서 forbidden level별 `eta`는 0, allowed/unknown level은 B와 동일해야 한다.
- C와 B의 `chi_bf`는 byte-identical이어야 한다.
- `eta_B-eta_C`는 forbidden level의 B-arm 합과 일치해야 한다.
- D는 inertness 경고를 내고 수치 출력은 B와 동일해야 한다.
- Si II의 기존 parity47 양성 지표와 FUV 악화 방향을 다시 확인하되,
  이 결과로 C61을 production 정물리로 승격하면 안 된다.

### 7.5 C59 multi-edge KA

- legacy alias만 ON한 arm과 새 `LUMINA_FIX_BF_MULTI_EDGE=1`만 ON한 arm의
  host `kpacket_fb_edge_{nu,cdf,zstage,lev,count}`, `p_kpacket_fb`,
  `p_kpacket_ff`를 byte 비교한다.
- 각 shell에서 CDF는 단조, 마지막 유효 값은 1이어야 한다.
- `C_fb_real`은 top-N CDF에 보존된 weight가 아니라 **모든** 유효 continuum
  weight의 합과 일치해야 한다.
- single-edge OFF arm과 비교할 때 선택 edge 분포는 달라져야 하지만 전체
  `p_ff+p_fb+p_collexc=1` 정규화는 유지돼야 한다.

### 7.6 CMFGEN/dual-oracle 해석

현재 `scripts/oracle_compare_cmfgen.py`와 선택 CMFGEN snapshot은
monochromatic `chi_bf_at_*`/`eta_bf_at_*`의 동일 계수를 노출하지 않아 해당
행을 `unavailable`로 판정한다. 이를 합격으로 오인하지 말아야 한다.

- D-3 직접 oracle: ARTIS 식 + LTE KA
- CMFGEN 대조: 가능한 `Gamma_photoion_total`, `alpha_recomb_total`,
  `cooling_bf_net`만 동일 정의로 비교
- neutral/B18/C59: Lumina A/B와 per-level/continuum KA를 1차 판정으로 사용

## 8. 비-bf B형 잔여 14건 — Wave 1 범위 밖

| ID | 영역 | 잔여 물리 오해 |
|---|---|---|
| C21 | bound-bound | inversion/maser absorption을 0으로 치환 |
| C28 | radiation field | J를 인공 factor·WB 상하한으로 제한 |
| C29 | radiation field | UV Jν를 `W_cap Bν`로 제한 |
| C32 | collision | 실제 Υ 위에 인공 하한 적용 |
| C45 | line response | upper population≤LTE, lagged emissivity≥0 |
| C47 | radiation field | 고정 T_rad/TEPIN/W cap을 물리 해로 강제 |
| C68 | line formal | thick-line `S_l`을 `B(T_e)`로 치환 |
| C69 | line opacity | IGE forest opacity 제거 falsifier |
| C70 | line source | Fe 창 source에 비물리 배율 적용 |
| C71 | line transport | line re-emission을 `B(T_e)`로 치환 |
| SC05 | composition | 외곽 Fe 질량분율 floor |
| SC12 | atomic data | GPU memory level cap |
| SC15 | prototype | ETLA upper population≤LTE |
| SC16 | prototype | placeholder `gbar=0.2`, `f>=1e-6` |

이 중 C68-C71은 “방출”이라는 단어는 포함하지만 bound-free/free-bound 경로가
아닌 line falsifier이므로 이번 bf Wave에서 건드리지 않았다.
