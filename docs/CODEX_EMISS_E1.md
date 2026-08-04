# 방출률 캠페인 E1 — 인구-스왑 판별

작성일: 2026-08-01 (KST)  
상태: **UNRESOLVED (조립식 운반자 쪽의 강한 보조 증거)**

## 0. 결론

사전등록 표를 기계적으로 읽으면 B-lane도 UV 과잉을 그대로 유지했다. 600–3000 Å에서
`J_det/CMFGEN`은 권위 A의 11.7038에서 B의 11.4055로만 변했고, 과잉
`(R-1)`의 2.79%만 줄었다. 개별 대역도 7.28–33.99배로 남았다. 따라서 관측된 방향은
**인구보다 조립식(`S_l`, 열화/산란 분할, 주파수 재분배)이 주 운반자**라는 쪽이다.

그러나 이것을 사전등록의 확정 판결로 채택하지 않는다.

1. 동결 A 상태를 현재 소스의 동일 `cmfgen_assemble`에 재입력한 대조가 권위 캡처와
   `chi` L1 8.23%, `eta_fixed` L1 35.17% 달랐다. 같은 함수명이어도 캡처 실행파일과
   현재 소스/숨은 등록 상태의 완전한 왕복 항등이 아니다.
2. CMFGEN 인구는 Lumina 26,592 레벨 중 14,395개(54.13%)만 엄밀히 매핑됐다.
   이 중 Lumina 이온의 모든 레벨이 덮인 것은 9개 이온뿐이다.
3. RVTJ 속도 범위 밖인 바깥 6개 셸(s44–s49)은 외삽하지 않고 A를 유지했다.
4. `*PRRR`의 명목 Ion Density는 `*OUT` 헤더의 Saha 기준밀도와 일부 깊이에서
   최대 214배 불일치한다. B의 수치 입력은 직접 `POP*`와 왕복 검증된 `*OUT` 헤더를
   사용했지만, PRRR 불일치 자체의 원인은 미해결이다.

따라서 정식 판정은 **UNRESOLVED**다. 다만 ① 검증된 인구 교체가 61.51%의 전체
line-shell cell(600–3000 Å에서는 23,119,184 cell)에 실제 적용됐고, ② 그 결과가
과잉을 거의 움직이지 않았으며, ③ CMFGEN 원소스와 Lumina 조립식 사이에 직접적인
구조 차이가 확인되므로, 다음 노선은 **조립식 대조 감사 우선**이다. element-wide/장
교체는 인구의 완전 왕복 대조가 마련될 때까지 병행 후보로만 남긴다.

## 1. 사전등록 판독과 실제 상태

| 사전등록 분기 | 필요한 관측 | 이번 관측 | 상태 |
|---|---|---|---|
| 인구 운반자 | B가 CMFGEN 장으로 수렴 | BALL 11.4055배, 수렴하지 않음 | 지지 안 함 |
| 조립식 운반자 | B도 과잉 유지 | 5개 세부 대역 모두 6.90–33.99배 | 강하게 지지 |
| 확정 판결 요건 | A 왕복 항등 + 충분한 B 커버 | 둘 다 불충족 | **UNRESOLVED** |

이 보고서에서 “B”는 완전 커버 이온의 이온밀도·BF와 양 끝 레벨이 모두 검증된 선의
Sobolev `tau`를 CMFGEN 값으로 바꾸고, 나머지는 A를 유지한 **부분 인구 스왑**이다.
이를 “CMFGEN 전체 인구”로 부르지 않는다.

## 2. 입력과 고정 계약

### A-lane

- 권위 캡처:
  `/gpfs/kjhan/lumina_runner2/scratch/chieta_capture_parity59_188605/chieta_iter10`
- payload SHA-256:
  `94d75988034454f55fb6b130f04521f01c56f875cb22ef3a711850d7382ffa2f`
- 50 shell × 1000 frequency, iteration/generation 10/10, post-damping.
- 권위 J_det는 `docs/s31_results/stage31_jdet_s8_round7d.tsv`를 재사용했다.

### B-lane CMFGEN 앵커

- `/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/`
- 온도, 전자밀도, 속도: `RVTJ` (ND=90).
- departure coefficient: `*OUT`.
- Saha 기준밀도: 같은 `*OUT`의 depth header 두 번째 열.
- 레벨 `g`, 에너지, edge: `*_F_OSCDAT`.
- 독립 왕복 기준: `POPIRON`, `POPCOB`, `POPNICK`, `POPSIL`, `POPSUL`, `POPCAL`.
- `*PRRR`은 스키마 감사에는 포함했으나 불일치 때문에 수치 입력으로 쓰지 않았다.

신규 모델/GPU 실행은 없었다. CMFGEN 파일 읽기, CPU 조립 1회(A/B), CPU J_det만
수행했다. 조립+J_det 벽시간은 약 58초였고 조립기 최대 RSS는 약 6.9 GB였다.

## 3. CMFGEN 인구 임포트의 단위·스키마 항등

### 3.1 `*OUT`은 `b-1`이 아니라 `b`

CMFGEN 원소스
`/gpfs/kjhan/cmfgen_src/cur_cmf/subs/writedc_v3.f:20,68-110`은 “departure
coefficients, not b-1”이라고 명시하고 실제로

```text
b_i = exp(log(n_i) - log(n_i,LTE))
```

를 기록한다. `*OUT`의 네 번째 줄에서 `(NLEV,ND)`를 검사하고, 매 depth의 8개 header
값과 정확히 NLEV개의 `b`를 소비했다. 폭이 남거나 부족하면 즉시 실패한다.

### 3.2 Saha–Boltzmann 식

`LTEPOP_WLD_V2` (`subs/ltepop_wld_v2.f:64-77`)의 식을 그대로 사용했다.

```text
n_i,LTE = W_occ,i g_i × 2.07078e-22 × n_e × DIC2
          × T4^(-3/2) / g_next × exp(edge_i × HDKT / T4)
n_i = b_i n_i,LTE
HDKT = 1e11 h/k
```

현재 매핑된 레벨에서는 `W_occ=1`인 출력 정의를 사용한다. `DIC2`는 해당 이온의
총밀도가 아니라 Saha 관계의 **다음 이온 기준밀도**다. 원소스 주석과 인수 정의는
`ltepop_wld_v2.f:4-9,25-35`에 있고, `HDKT` 정의는
`new_main/cmfgen.f:119`에 있다. `CNVT_FR_DC_V2`는 `b`와 `log LTE`를 다시 더해
population으로 바꾸는 역변환을 `subs/cnvt_fr_dc_v2.f:31-35`에서 수행한다.

### 3.3 직접 POP 왕복

CMFGEN은 최종 반복에서 `POP<SPECIES>`를 쓰며
`new_main/cmfgen_sub.f:4524-4538`이 full-level population을 `RITE_ASC`에 넘긴다.
전 이온의 `OUT→Saha→b×LTE` 복원과 이 직접 POP를 비교한 결과:

- 이온별 median ratio: 0.99999656–1.00000163
- 이온별 p01: 0.99997583–0.99999760
- 이온별 p99: 1.00000242–1.00002438
- 이온별 최대 절대 오차: 1.51e-6–1.18e-5 dex

따라서 `*OUT` 기반 왕복과 단위는 확정했다. 반대로 PRRR/OUT 기준밀도의 최대 상대
차이는 이온별 0.140–213.789였다. 이는 출력 정밀도만으로 설명되지 않으므로
**PRRR 의미/동시성은 UNRESOLVED**다.

## 4. 매핑과 커버리지

레벨은 `(Z, ion, zero-based level_number)`가 같고, 정수 `g`가 같으며, CMFGEN의
cm⁻¹ 텍스트 정밀도만 포괄하는 `|Delta E| <= 2e-6 eV`일 때만 매핑했다. 이 허용폭은
population floor나 물리 clamp가 아니다.

| 이온 | OUT NLEV | Lumina 매핑/전체 | 상태 |
|---|---:|---:|---|
| Si II / III / IV | 125 / 147 / 61 | 125/157, 147/147, 61/66 | III만 FULL |
| S II / III / IV | 322 / 256 / 176 | 322/324, 256/380, 176/194 | PARTIAL |
| Ca II / III / IV | 62 / 232 / 375 | 62/77, 200/200, 375/378 | III만 FULL |
| Fe II / III / IV / V | 2599 / 1500 / 1000 / 1000 | 2599/2698, 1500/1500, 200/200, 200/200 | III–V FULL |
| Co II / III / IV | 2558 / 3214 / 1000 | 2558/2747, 3214/3917, 200/200 | IV만 FULL |
| Ni II / III / IV | 1000 / 1000 / 1000 | 1000/1000, 1000/1000, 200/200 | 모두 FULL |
| Si V, S V, Ca V | — | 0 | next-ion ground `g` 없음 |
| Co V, Ni V | 1000 | Lumina 해당 레벨 0 | UNRESOLVED/비적용 |

완전 커버 9개 이온은 Si III, Ca III, Fe III–V, Co IV, Ni II–IV다. 이들만 이온밀도와
BF projection 전체를 교체했다. 선은 양 끝 레벨 population이 모두 유한할 때만
`tau ∝ f lambda t [n_l-(g_l/g_u)n_u]`를 다시 계산했다.

- 전체 level: 26,592; 검증 매핑: 14,395 (54.1328%).
- RVTJ 안 shell: 44/50; s44–s49는 NaN으로 기록하고 A 유지.
- 교체 level-shell cell: 204,468.
- 교체 ion-shell cell: 396.
- 교체 line-shell cell: 79,466,068 / 129,206,600 (61.51%).
- 이 중 600–3000 Å: 23,119,184 cell.

외삽, endpoint hold, population floor를 쓰지 않았다.

## 5. 같은 조립기의 A/B 구성

`scripts/emiss_population_swap_e1_driver.c`는 캡처의 환경, plasma state, ion/level
population, deposition, J를 읽고 다음 생산 경로를 호출한다.

1. `compute_tau_sobolev`와 `nlte_update_tau_sobolev`의 frozen-oracle wrapper.
2. `compute_bf_opacity`.
3. `cmfgen_assemble`.
4. `cmfgen_dump_frozen_chieta`.

wrapper는 `LUMINA_FROZEN_ORACLE` 빌드에만 존재하며 정상 CPU/GPU 바이너리에는 없다.
A와 B의 `eta_decomposition_bitwise=true`, 최대 분해 오차 0을 writer가 확인했다.

### 5.1 A 왕복 감사

| 항목 | 권위 캡처 대비 상대 L1 |
|---|---:|
| chi_total | 0.0822606 |
| eta_fixed | 0.351702 |

이는 허용 가능한 roundoff가 아니다. 가능한 원인은 캡처 바이너리와 현재 소스 트리의
revision 차이, 캡처 시점에만 존재한 정적 registration/lag state, 최종 dump와 CSV
dump의 시점 차이다. 어느 것인지 분해하지 않았으므로 **A assembler replay identity는
UNRESOLVED**다.

그럼에도 J_det 수준에서 A 재생은 권위 A와 가깝다. BALL은 11.6578 대 11.7038
(0.39% 차이)다. 따라서 B-A의 작은 변화는 보고하되 확정 판결에는 사용하지 않는다.

### 5.2 clamp 규율

캠페인 코드에는 새로운 clamp/floor를 넣지 않았다. stage31 driver의 기록은 B에서
`clamp=0`, `solution_negative_excess=0`, `sign_uncertain=0`, `nonfinite=0`이다.

단, “동일 생산 조립기” 자체에는 기존 동작이 있다. `cmfgen_assemble`의 물리 eps
floor/cap(`src/lumina_cmfgen.c:485-509,545-550`), 음의 BF/FF 제거
(`:682-693`), production Sobolev population-inversion 무흡수 처리와 `1e-100` tau가
그것이다. 이번 작업은 이를 추가하거나 바꾸지 않았다. “어떤 기존 clamp도 거친 수치를
허용하지 않는다”는 더 강한 의미라면 같은 생산 경로와 동시에 만족할 수 없으며, 그
해석에서도 본 판정은 UNRESOLVED다.

## 6. 600–3000 Å J_det 결과

CMFGEN 비교장은 stage31 7D와 같은
`/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/`의 s8 속도 보간/적분보존 bin 평균이다.
modern run은 population 앵커로만 사용했다.

| band [Å] | A 권위 / CMFGEN | A 재생 / CMFGEN | B / CMFGEN | B/A 권위 | 제거된 과잉 `(A-B)/(A-1)` |
|---|---:|---:|---:|---:|---:|
| 600–1000 | 33.7413 | 33.7321 | 33.9945 | 1.00751 | -0.77% |
| 1000–1500 | 32.1914 | 32.1613 | 31.4396 | 0.97665 | 2.41% |
| 1500–2000 | 7.38976 | 7.39200 | 7.28328 | 0.98559 | 1.67% |
| 2000–2500 | 6.86443 | 6.86641 | 6.89917 | 1.00506 | -0.59% |
| 2500–3000 | 15.5806 | 15.4522 | 14.9004 | 0.95634 | 4.67% |
| **600–3000** | **11.7038** | **11.6578** | **11.4055** | **0.97451** | **2.79%** |

B의 stage31 수치 guard:

- transport residual: 6.689e-7.
- clamp / excess-negative / sign-uncertain / nonfinite: 0 / 0 / 0 / 0.
- 권위 A의 `J_det/J_MC=0.977181`; B의 `J_det/J_MC=0.952276`.

B는 CMFGEN으로 수렴하지 않는다. 완전한 B-lane이었다면 사전등록상 “조립식 운반자”
판결이다. 현재는 부분 B와 A 왕복 실패 때문에 **그 방향의 보조 증거**로 제한한다.

## 7. 조립식 정식화 대조 — Lumina 대 CMFGEN 원소스

### 7.1 선 opacity와 emissivity

CMFGEN의 현재 `SET_LINE_OPAC`은
`new_main/mod_subs/set_line_opac.f:294-318`에서 각 선을 직접 구성한다.

```text
chi_l ∝ f_lu [L_ratio n_l - (g_l/g_u) U_ratio n_u]
eta_l ∝ nu A_ul U_ratio n_u
```

즉 유도방출은 `chi_l`의 음의 `n_u` 항으로 들어가고, 자발 방출은 `eta_l`의 `A_ul n_u`
항이다. 이후 선 profile을 곱해 주파수별 opacity에 더한다
(`new_main/cmfgen_sub.f:1498-1517`). super-level/full-level ratio도 두 항에 별도로
반영한다.

Lumina는 `src/lumina_cmfgen.c:523-566`에서 먼저 Sobolev
`(1-exp(-tau)) nu/(ct Delta-nu)`를 한 coarse bin에 더하고, 캡처 설정에서는
`LUMINA_CMFGEN_SRC_NLTE`가 꺼져 있어 `S_l=B_nu(T_e)`로 대체한다. 따라서 CMFGEN의
직접 `A_ul n_u` emissivity가 아니라 `w eps_l B_nu(T_e)`가 thermal line eta가 된다.

### 7.2 열화/산란 분할

Lumina는 각 선에
`eps_l=C_ul/(C_ul+A_ul beta_esc)`를 적용하고 eps 몫만 thermal emissivity로 보내며,
나머지를 coherent `chi_es`로 합친다 (`lumina_cmfgen.c:545-565,715-733`). 캡처에서는
`LUMINA_CMFGEN_LINE_EPS_PHYS=1`이다.

CMFGEN `SET_LINE_OPAC`에는 이와 같은 사후 `eps` 분할이 없다. population으로 계산한
`chi_l,eta_l`가 line profile을 통해 full CMF transfer에 직접 들어간다. 산란성은
statistical equilibrium과 `eta_l/chi_l`의 결과이지, 조립 단계의 별도 열화분율로
강제되지 않는다.

### 7.3 재분배와 선 중첩

CMFGEN은 한 선의 resonance zone과 깊이 의존 profile을 보존하고 겹치는 선들을 같은
주파수 solve에 더한다. `SET_LINE_OPAC`의 resonance storage/profile 경로와
`cmfgen_sub.f:1507-1517`이 그 소비점이다. `SOBEW`는 각도 의존 Sobolev tau와
continuum transfer를 별도로 적분한다 (`subs/sobew.f:48-66,115-131`).

Lumina coarse assembler는 한 선을 중심주파수의 단일 1000-bin에 넣는다. 선 profile,
한 선 내부의 주파수 재분배, 같은 bin 안 선별 source contrast가 사라진다. coherent
몫은 같은 bin의 `J`에만 결합된다. 이 차이는 B가 인구를 바꿔도 UV 과잉을 유지한
결과와 방향이 맞는다.

### 7.4 continuum와 에너지 재형상

CMFGEN `GENOPAETA_V10`은 free-free를 `chi_ff`와 `eta_ff`로 직접 구성
(`newsubs/genopaeta_v10.f:127-171`)하고, bound-free도 실제 lower population과
target-ion LTE/Milne 항으로 `chi_bf,eta_bf`를 함께 구성한다 (`:175-195,252-316`).
level dissolution도 같은 주파수별 합에 포함된다.

Lumina는 BF/FF와 line thermal eta를 만든 뒤 `LUMINA_CMF_EPAY=2`에서 셸별 흡수
예산으로 전체 방출을 재규격화하고, hot regime에서는 Milne+`chi_line_th B` 모양으로
다시 배분한다 (`lumina_cmfgen.c:584-613,739-825`). 이 EPAY 사후 변환에 대응하는
동일한 per-shell scalar/re-shape 단계는 위 CMFGEN `SET_LINE_OPAC/GENOPAETA` 경로에
없다. CMFGEN의 에너지 폐쇄는 결합된 transfer/statistical/radiative-equilibrium
방정식에서 이뤄진다.

### 7.5 구조 차이 요약

| 항목 | Lumina 캡처 경로 | CMFGEN 원소스 |
|---|---|---|
| line chi | binned expansion `(1-e^-tau)` | profile-resolved `n_l-(g_l/g_u)n_u` |
| line eta | `eps_l w B(T_e)` (`SRC_NLTE=0`) | 직접 `nu A_ul n_u` |
| 유도방출 | tau 계산 후 inversion 무흡수 | chi의 명시적 upper-pop 음항 |
| 열화 | 명시적 `eps_l`, 잔여를 coherent J로 | 별도 사후 eps 분할 없음 |
| 중첩/재분배 | 중심주파수 단일 coarse bin | resonance-zone/profile CMF solve |
| super-level | Lumina level mapping/production tau | `L_STAR_RATIO/U_STAR_RATIO`를 chi/eta에 별도 적용 |
| continuum | BF Milne + 별도 FF 근사 | level/target별 BF와 FF를 주파수별 직접 합 |
| 에너지 폐쇄 | EPAY 셸별 scale/re-shape | 결합 방정식의 RE/SE 폐쇄 |

이 목록에서 가장 직접적인 다음 falsifier는 **CMFGEN 식의 `chi_l,eta_l`를 같은
population으로 Lumina grid에 profile-integrated 투영**한 뒤, (a) current eps/B/EPAY
조립과 (b) 직접 `A_ul n_u` 조립을 나란히 J_det에 넣는 것이다.

## 8. 재현 명령

workspace root에서 실행한다.

```bash
python3 -m py_compile scripts/emiss_population_swap_e1.py
python3 scripts/emiss_population_swap_e1.py

gcc -O2 -w -std=gnu11 -D_GNU_SOURCE -DLUMINA_FROZEN_ORACLE \
  -ffunction-sections -fdata-sections -Isrc -Wl,--gc-sections \
  -o /tmp/emiss_e1_driver \
  scripts/emiss_population_swap_e1_driver.c \
  src/lumina_plasma.c src/lumina_element_wide.c \
  src/lumina_atomic.c src/lumina_cmfgen.c -lm

/usr/bin/time -f 'elapsed=%E maxrss=%MKB' /tmp/emiss_e1_driver

gcc -O2 -std=c11 -D_POSIX_C_SOURCE=200809L -Isrc \
  scripts/stage31_cmf_field_driver.c src/lumina_cmf_field.c -lm \
  -o /tmp/stage31_cmf_field_driver

/tmp/stage31_cmf_field_driver \
  validation/emiss_e1/chieta_A_replay \
  validation/emiss_e1/chieta_A_replay.manifest.json \
  8 16 10020 1 validation/emiss_e1/jdet_A_replay.tsv

/tmp/stage31_cmf_field_driver \
  validation/emiss_e1/chieta_B \
  validation/emiss_e1/chieta_B.manifest.json \
  8 16 10020 1 validation/emiss_e1/jdet_B.tsv

python3 scripts/emiss_population_swap_e1_bands.py
```

band table 생성기는 `scripts/stage31_cmf_field_bench.py`의 `canonical_grid`,
`load_gamma_context`, `parse_driver_table`, `make_band_rows`를 재사용했고 결과 전체를
`validation/emiss_e1/band_ratios.json`에 기록했다.

## 9. 산출물과 checksum

| 산출물 | SHA-256 |
|---|---|
| `cmfgen_b_populations.bin` | `9c07cc1afb0ded7b03a190f929f0e34f74a5c160655387150798f214294cfb5c` |
| `chieta_A_replay` | `fa13326d50fff3d84d89bed0041a7b85828bd91952fcddb134d9942f38f625e1` |
| `chieta_B` | `ef2c9f41d4bb9669ff73dd1df6d89fd5265abc4697fda42cbb8427583203831c` |
| `jdet_A_replay.tsv` | `47baac8568423e2751438e2f10d8636badfbe5a0c1d22d01aead1b0c5849a3fb` |
| `jdet_B.tsv` | `0f9d8563e8c4536cd4af30c0fc56bdaa53142442c6c89bf1c58565ecd54b4fa2` |
| `band_ratios.json` | `055aa2e4525a82e7bfe9cf39a77216783a573734a756a04d93294e6d053cd844` |

부가 감사 파일:

- `validation/emiss_e1/population_import_summary.json`
- `validation/emiss_e1/population_import_audit.csv`
- `validation/emiss_e1/assembly_audit.json`
- 각 `chieta_*`의 writer manifest

## 10. 다음 결정점

1. **우선:** 캡처 당시 exact source revision/registration snapshot으로 A replay의
   `chi,eta_fixed` 항등을 먼저 회복한다.
2. CMFGEN `SET_LINE_OPAC` 식의 profile-integrated `chi_l,eta_l` 직접 조립 대조를 만든다.
   `eps_l B(T_e)`와 EPAY를 각각 독립 축으로 제거하는 2×2 감사면 충분하다.
3. population 쪽 확정에는 modern RVTJ 외곽을 덮는 앵커와 partial 이온의 full-level
   OUT/POP가 필요하다. 확보 전에는 외삽하거나 이온합을 추정하지 않는다.
4. 위 1이 성립한 뒤에도 B가 약 11배를 유지하면 사전등록에 따라 조립식 운반자를
   확정하고, element-wide+장 교체 노선보다 조립식 수정을 우선한다.

커밋은 만들지 않았다.
