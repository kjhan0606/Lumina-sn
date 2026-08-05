# classic 부채 전수 census

기준일: 2026-08-05

상태: 정적 판독 census. 코드·덱·`/gpfs`를 변경하거나 실행하지 않았다.

범위: `src/*.c`, `src/*.cu`, `src/*.h` 22파일 50,372행 전부와, 생산 덱/원자자료 생성·재생성 경로 14개 및 이름과 호출부로 생산 경로임을 확인한 수송 wrapper 6개. 테스트·bench 전용 리터럴은 판독했으나 물리 산출 경로가 아니므로 등재하지 않았다.

## 1. 요약 통계

| 부류 | 정의 | 열린 항목 수 |
|---|---|---:|
| 1 | 하드코딩 | 19 |
| 2 | 간략화 | 18 |
| 3 | 미리짐작 | 11 |
| 4 | Phase 주석 고고학 | 4 |
| **합계** | 기전·소유권 단위 열린 항목 | **52** |

계수법: 아래 네 표의 ID 행을 한 번씩 셌다. 한 기전이 CPU/GPU 또는 로더/소비자에 복제된 경우 한 행에 위치를 병기하여 중복 계수하지 않았고, 주된 부류 하나에만 귀속했다. 따라서 합계는 파일별 hit 수가 아니라 **서로 다른 열린 물리 기전/계약 52건**이다. 폐합 seed 2건은 이 합계에서 제외했다. 모든 `처분` 칸은 의도적으로 비어 있다.

클램프·floor·rail·양수화·범위 제한은 이 문서에서 세지 않았다. 경계가 맞닿는 경우에도 정본인 [`CODEX_CLAMP_CENSUS_2026-07-31.md`](CODEX_CLAMP_CENSUS_2026-07-31.md)의 88항목을 참조하며, 아래에는 대체식·고정 물리모형처럼 클램프 자체와 독립인 기전만 남겼다.

### 1.1 요청 seed의 상태

| seed | 상태 | census 연결/근거 |
|---|---|---|
| `src/lumina_main.c:92` `T_e_T_rad_ratio = 0.9` | **열림** | H01 |
| `src/lumina_atomic.c:823-826` `t_electrons = T_rad /* for now */` | **폐합됨 (TE-DEAD)** | seed로만 기록. 열린 표와 합계에서 제외 |
| `scripts/build_toy06_epoch.py:236` `ne = n_atom * 1.0  # <Z_ion>~1` | **폐합됨 (NE-NAMING)** | 현행 위치는 236행이며 승인 없는 생산을 차단한다. 깨끗한 재측정은 광구 위치 `+71.79%` ([`OUTSIDE_LOOP_POOL.md:1273`](OUTSIDE_LOOP_POOL.md)); 열린 표와 합계에서 제외 |
| `src/lumina_plasma.c:9875,11644` `radeq_damp = 0.5` | **열림** | H03 |
| van Regemorter·`OMEGA_SET` 차용부 | **열림** | S06 |
| Kramers 차용부 | **열림** | S07, S08 |

## 2. 부류 1 — 하드코딩 (19)

| ID | 파일:행 | 부류 | 현 코드가 하는 것 | 정확 물리가 요구하는 것 | 영향 추정 | A-2 단계 매핑 | 처분 |
|---|---|---:|---|---|---|---|---|
| H01 | `src/lumina_main.c:92`<br>`src/lumina_atomic.c:734`<br>`src/lumina_plasma.c:2999,3042` | 1 | 기본 전자온도를 모든 shell에서 `0.9 T_rad`로 결박한다. | 항별 가열·냉각과 입자수 보존을 함께 푼 국소 `T_e`가 필요하다. | UNMEASURED | A2-07, A2-10, A2-17 | |
| H02 | `src/lumina_main.c:107-108`<br>`src/lumina_cuda.cu:6905-6906` | 1 | 복사장/전이확률 갱신에 감쇠 `0.5`, hold 3회를 CPU/GPU 기본값으로 둔다. | 잔차·스펙트럼 반경에 근거한 수렴 제어와 seed 독립성이 필요하다. | UNMEASURED | A2-04, A2-18 | |
| H03 | `src/lumina_plasma.c:9875-9876,11644-11647` | 1 | 복사평형 `T_e` 갱신을 두 경로 모두 고정 `0.5`로 감쇠한다. | 완전 결합 에너지 잔차의 Jacobian/line search에 의해 갱신량이 정해져야 한다. | UNMEASURED | A2-10 | |
| H04 | `src/lumina_plasma.c:2601-2604,2630,2725-2730` | 1 | 전자밀도 고정점 반복을 `0.5`, 5%, 최대 100회로 끝낸다. | 전하중성 잔차에 대한 명시적 허용오차와 수렴 증명이 필요하다. | UNMEASURED | A2-07 | |
| H05 | `src/lumina_plasma.c:17506-17508`<br>`src/lumina_cuda.cu:1027-1029` | 1 | charge-exchange 외부 반복을 5회, 1%, 감쇠 0.5로 제한한다. | CE를 포함한 단일 SE 계 또는 잔차 기반 완전수렴이 필요하다. | UNMEASURED | A2-07, A2-13, A2-18 | |
| H06 | `src/lumina_transport.c:351,523`<br>`src/lumina_cuda.cu:2906,2934,2942,7138,7163,7173` | 1 | packet 100,000회, 전체 step 2,000,000회, macro-atom hop 5,000회에서 물리 이력을 끊는다. | 확률과정은 물리적 흡수·탈출·deactivation으로 종료되고, 수치 중단은 물리 결과로 받아들이지 않아야 한다. | UNMEASURED | A2-12, A2-15, A2-18 | |
| H07 | `src/lumina.h:512-514` | 1 | `J_nu`/BF 격자를 1,000 bin, `1.5e14–3.0e16 Hz`로 고정하며 끝점도 반올림했다. | 모든 소비 주파수의 합집합과 해상도 수렴으로 좌표·격자를 정해야 한다. | UNMEASURED | A2-02, A2-05, A2-06, A2-13, A2-14 | |
| H08 | `src/lumina_main.c:262-263,794-800,819-823` | 1 | 출력 범위 500–20,000 Å/2,000 bin, formal ray 100개, CMF `nz=2000`, ray 50개, `v_turb=0`을 기본값으로 둔다. | observer-frame profile과 flux가 실제 line broadening 및 주파수·impact parameter·공간 해상도 사다리에서 수렴해야 한다. | UNMEASURED | A2-11, A2-18 | |
| H09 | `src/lumina.h:20,29,33`<br>`src/lumina_cmfgen.c:20,2017`<br>`src/lumina_cuda.cu:3613-3614` | 1 | 서로 다른 절단값의 Thomson/Sobolev/FF 계수와 역사적 ARTIS `h,k_B`를 경로별로 쓴다. | 단일 단위계·동일 CODATA에서 유도된 상수와 불확도/버전 계약이 필요하다. | UNMEASURED | A2-08, A2-11, A2-14 | |
| H10 | `src/lumina.h:35`<br>`src/lumina_cuda.cu:2356-2359,5581` | 1 | shell당 free-bound 재결합 연속체를 최대 16개만 저장·표본한다. | 유의한 모든 level-resolved continuum를 누락 없이 정규화해 표본해야 한다. | UNMEASURED | A2-09, A2-15 | |
| H11 | `src/lumina.h:531-532,594,616` | 1 | NLTE ion/pair, CE 반응, DR 항을 각각 38/23, 20, 10 슬롯으로 제한한다. | 입력 원자모형의 실제 ion·반응·fit term 수를 보존하는 동적 topology가 필요하다. | UNMEASURED | A2-07, A2-13 | |
| H12 | `src/lumina_plasma.c:9694,9698,9805-9817`<br>`src/lumina_radeq_col_pairs.h:40,44,139-140` | 1 | 복사평형 충돌합을 ion당 최저에너지 512준위로 자르고, 미표 전이에 `Omega=0.1`과 고정 Gaunt 수를 쓴다. | 전 수준쌍의 실제 `Upsilon(T)`/충돌자료를 동일 population 위에서 적분해야 한다. | UNMEASURED | A2-10 | |
| H13 | `src/lumina_plasma.c:1028-1040` | 1 | dilute-Planck fit을 1,500–50,000 K에서 80점+60점 탐색으로 제한한다. | 일반 `J_nu`를 보존하거나 적어도 데이터가 정한 범위에서 검증된 연속 최적화가 필요하다. | UNMEASURED | A2-03, A2-04 | |
| H14 | `src/lumina_plasma.c:3022-3033` | 1 | “self-consistent” 온도 경로의 충돌 결합을 Compton rate의 12배로 둔다. | 실제 bound-bound, bound-free, free-free, Compton 항을 개별 rate로 합산해야 한다. | UNMEASURED | A2-10 | |
| H15 | `src/lumina_plasma.c:17859-17861,17884,17961` | 1 | gamma 수송을 회색 `kappa=0.025`, 비열적 몫 0.05, ion pair당 35 eV로 닫는다. | 에너지 의존 gamma 수송과 전자분율 의존 Spencer–Fano 에너지 분배가 필요하다. | UNMEASURED | A2-07, A2-10 | |
| H16 | `src/lumina_cmfgen.c:3980-3984,4205-4208` | 1 | fine CMF/Jbar 창을 1,000–4,000 Å, Doppler 폭 10 km/s, 폭당 12점, ALI 24회와 선폭 ±4로 둔다. | 실제 열·미세난류 profile과 주파수/ALI 잔차 수렴으로 해상도를 정해야 한다. | UNMEASURED | A2-02, A2-06, A2-11 | |
| H17 | `src/lumina_cmfgen.c:5131-5151` | 1 | cross-line overlap의 탈출광 기여를 `f_out=0.5`로 곱한다. | 방향·주파수 의존 line transport가 방출·재흡수 확률을 직접 산출해야 한다. | UNMEASURED | A2-06, A2-09, A2-11 | |
| H18 | `src/lumina_atomic.c:1094-1156`<br>`scripts/slurm_ddc15_FI_prod.sh:181-206`<br>`scripts/slurm_nlte_o_recal_prod.sh:193-228` | 1 | 생산 wrapper가 선택 원소/이온/파장대의 `A_ul,B,f`를 0.05–0.3 등 경험 계수로 일괄 변조한다. | 검증된 원자자료의 전이별 계수와 불확도를 그대로 소비해야 한다. | UNMEASURED | A2-06, A2-09, A2-13, A2-15 | |
| H19 | `src/lumina_plasma.c:3822-3846`<br>`scripts/slurm_prod_dr7ion_ce17_ionlock.sh:47` | 1 | 생산 wrapper가 UV→optical macro-atom 방출 branch를 1.7배 증폭한다. | 완전한 전이망과 물리 rate의 정규화가 branch 확률을 결정해야 한다. | UNMEASURED | A2-09, A2-15 | |

## 3. 부류 2 — 간략화 (18)

| ID | 파일:행 | 부류 | 현 코드가 하는 것 | 정확 물리가 요구하는 것 | 영향 추정 | A-2 단계 매핑 | 처분 |
|---|---|---:|---|---|---|---|---|
| S01 | `src/lumina_plasma.c:847-963`<br>`src/lumina_main.c:530-535` | 2 | MC의 `J_nu`를 두 모멘트의 `(W,T_rad)` dilute blackbody로 압축해 되쓴다. | generation·frame·unit이 명시된 주파수별 `J_nu`를 생산자에서 소비자까지 보존해야 한다. | UNMEASURED | A2-03, A2-04, A2-17 | |
| S02 | `src/lumina_plasma.c:1072-1096` | 2 | 1,000-bin field를 ARTIS식 24개 `(W_b,T_b)` Planck 조각으로 재구성한다. | BF/BB rate는 선택된 정본 `J_nu`와 line-profile 평균을 직접 적분해야 한다. | UNMEASURED | A2-03, A2-05, A2-06 | |
| S03 | `src/lumina_plasma.c:2051-2080,2835-2874` | 2 | partition·level population·Sobolev tau를 `T_rad` dilute-Boltzmann으로 만든다. | partition/excitation은 `T_e`와 실제 SE population을 사용해야 한다. | UNMEASURED | A2-07, A2-08 | |
| S04 | `src/lumina_plasma.c:2215-2225,2397-2528,2601-2730` | 2 | ionization과 `n_e`를 Mazzali–Lucy nebular-Saha 폐합으로 반복한다. | 동일 `J_nu`, 충돌, 재결합, 전하중성을 묶은 ion rate-SE가 필요하다. | UNMEASURED | A2-05, A2-07 | |
| S05 | `src/lumina_transport.c:65-83,174`<br>`src/lumina_plasma.c:2874,2955-2958`<br>`src/lumina_cmfgen.c:3030-3118` | 2 | 선을 무한히 얇은 Sobolev resonance와 escape probability/jump로 처리한다. | 유한 profile, velocity gradient, overlap을 포함한 comoving/observer transfer가 필요하다. | UNMEASURED | A2-08, A2-11, A2-12, A2-14 | |
| S06 | `src/lumina_plasma.c:374-379,836-837,15444-15461`<br>`src/lumina_nlte_assemble.cu:192-202`<br>`src/lumina_radeq_col_pairs.h:138-140` | 2 | 표 없는 충돌강도를 van Regemorter 또는 `OMEGA_SET`/Axelrod 대표값으로 대체한다. | 전이별 온도의존 `Upsilon(T)` 또는 검증된 충돌 cross section이 필요하다. | 실소비 census: tabulated 29,840, vR 1,742,025(67%), `0.1` 812,267(31%) ([`CODEX_INPUT_ATOMIC_SUMMARY.md:35-36`](CODEX_INPUT_ATOMIC_SUMMARY.md)); 기존 정적 감사의 저-ΔE rate 오차 1–2 orders ([`ARTIS_PARITY_GAP_AUDIT.md:18`](ARTIS_PARITY_GAP_AUDIT.md)); 최종 산출 영향 UNMEASURED | A2-06, A2-10, A2-13 | |
| S07 | `src/lumina_atomic.c:1891-1897`<br>`src/lumina_plasma.c:7008,7365-7468`<br>`src/lumina_nlte_gemm.cu:186-190,379`<br>`src/lumina_bf_gemm.cu:31`<br>`src/lumina_element_wide.c:518-526` | 2 | 미등록 photoionization 단면을 `sigma_0(nu_0/nu)^3` Kramers 곡선으로 채운다. | level별 resonance 구조와 upper-ion target을 가진 tabulated/evaluated `sigma_bf(nu)`가 필요하다. | coverage 실측 26,087/26,592, fallback 505준위 ([`CODEX_INPUT_ATOMIC_SUMMARY.md:129-131`](CODEX_INPUT_ATOMIC_SUMMARY.md)); Fe II 122준위 시험 `Gamma +0.0158125051 dex` ([`CODEX_WAVE32_B_TEST_SUMMARY.md:8`](CODEX_WAVE32_B_TEST_SUMMARY.md)); 그 밖은 UNMEASURED | A2-05, A2-08, A2-13, A2-14 | |
| S08 | `src/lumina_plasma.c:5188-5221`<br>`src/lumina_cuda.cu:2236,2350-2360,5581-5649` | 2 | K-packet free-bound 냉각을 Kramers `alpha`와 한 dominant edge 또는 최대 16 edge로 방출한다. | level별 Milne 재결합 emissivity와 모든 target route를 에너지 보존적으로 표본해야 한다. | 기존 설계 판독은 현 방식이 detailed balance를 깬다고 확인 ([`EUV_MILNE_SOURCE_DESIGN.md:18-22`](EUV_MILNE_SOURCE_DESIGN.md)); 산출 영향 UNMEASURED | A2-09, A2-15 | |
| S09 | `src/lumina_plasma.c:3089-3104,6031-6069` | 2 | 모든 재결합 상부 ion을 ground core 하나로 대표하고 excited-core channel을 버린다. | upper-ion core별 target과 spin/상세평형 짝을 모두 추적해야 한다. | UNMEASURED | A2-05, A2-07, A2-09 | |
| S10 | `src/lumina_atomic.c:1984-2003` | 2 | v1 target map은 level당 `p=1` 단일 target이고, 파일이 없으면 ground-only/no-continuum으로 간다. | branch-resolved multi-target photoionization/recombination CSR과 정규화가 필요하다. | UNMEASURED | A2-09, A2-15 | |
| S11 | `src/lumina_cmfgen.c:1897-1910` | 2 | binned line forest를 `S=(1-epsilon)J+epsilon B`인 2준위 원자로 환원한다. | multilevel SE/macro-atom redistribution에서 line별 source를 산출해야 한다. | UNMEASURED | A2-06, A2-09, A2-11 | |
| S12 | `src/lumina_cmfgen.c:2011-2018` | 2 | free-free에서 Gaunt=1, `n_i≈n_e`, `Z^2≈1`로 둔다. | 실제 ion charge distribution과 주파수·온도 의존 Gaunt factor의 합이 필요하다. | UNMEASURED | A2-08, A2-09, A2-11 | |
| S13 | `src/lumina_transport.c:470-479`<br>`src/lumina_cuda.cu:5397-5406` | 2 | gate가 켜지면 Fe II/전체 Fe를 macro-atom 대신 2준위 공명산란으로 바꾼다. | 실제 Fe multilevel branching·fluorescence·collisional destruction이 필요하다. | UNMEASURED | A2-06, A2-09, A2-13, A2-15 | |
| S14 | `src/lumina_atomic.c:1189-1208` | 2 | cutoff 위 모든 준위를 한 Boltzmann super-level로 뭉친다. | 관측량 민감도에 수렴한 super-level partition 또는 full-level SE가 필요하다. | UNMEASURED | A2-07, A2-13 | |
| S15 | `src/lumina_plasma.c:16246-16330,16625-16667` | 2 | top ion을 고정 Saha-IV reservoir로 두고 고립 행을 dilute-Boltzmann anchor로 메운다. | 위 ion stage까지 포함한 연속체 rate network와 rank-complete SE가 필요하다. | UNMEASURED | A2-05, A2-07, A2-13 | |
| S16 | `src/lumina_plasma.c:16597-16618` | 2 | 시간의존 ionization을 폭발시각 한 번의 backward-Euler step으로 축약한다. | 실제 초기상태에서 시간격자·방사장 이력을 따라 적분한 stiff rate system이 필요하다. | UNMEASURED | A2-07 | |
| S17 | `scripts/build_ddc15_epoch.py:65-81`<br>`scripts/build_ddc15_initial_epoch.py:118-140`<br>`scripts/build_ddc15_real_composition.py:126-145` | 2 | 알려진 decay chain을 고정 `dt=0.01/0.002 d` forward Euler로 전개한다. | Bateman matrix exponential 또는 오차제어형 decay 적분이 필요하다. | UNMEASURED | A2-16 | |
| S18 | `scripts/build_ddc15_epoch.py:153`<br>`scripts/build_ddc15_initial_epoch.py:241`<br>`scripts/build_ddc15_real_composition.py:218` | 2 | 초기 dilution을 소각근사 `0.5(r_in/r)^2`로 쓴다. | 구형 광구의 정확한 `0.5[1-sqrt(1-(r_in/r)^2)]` 또는 정본 `J_nu` seed가 필요하다. | UNMEASURED | A2-16 | |

## 4. 부류 3 — 미리짐작 (11)

| ID | 파일:행 | 부류 | 현 코드가 하는 것 | 정확 물리가 요구하는 것 | 영향 추정 | A-2 단계 매핑 | 처분 |
|---|---|---:|---|---|---|---|---|
| G01 | `src/lumina_atomic.c:1755`<br>`src/lumina_plasma.c:1681-1687` | 3 | ionization 자료가 없으면 neutral ladder로 시작하고, 누락 energy는 `1e10 eV`로 만들어 ionization을 막는다. | 누락을 명시적으로 실패시키거나 검증된 ion ladder/energy 자료를 공급해야 한다. | UNMEASURED | A2-07 | |
| G02 | `src/lumina_plasma.c:2020-2032` | 3 | zeta 자료가 없는 ion은 `zeta=1`, 즉 LTE로 간주한다. | ion별 재결합 ground fraction 또는 level-resolved recombination rate가 필요하다. | UNMEASURED | A2-07 | |
| G03 | `src/lumina_plasma.c:17167-17190`<br>`src/lumina_cuda.cu:1457-1480` | 3 | singular/non-finite NLTE solve를 `Boltzmann@T_rad` population으로 대체한다. | 잘 조건화된 보존 SE solve가 성공해야 하며 실패값이 물리 population이 되어서는 안 된다. | UNMEASURED | A2-07, A2-13, A2-18 | |
| G04 | `src/lumina_atomic.c:908-910`<br>`src/lumina_cmfgen.c:4203-4220`<br>`src/lumina_cuda.cu:4027` | 3 | 미작성 `line_source_S=0`을 `B_nu(T_e)` thermal source로 읽는다. | 모든 소비 line에 generation-valid SE source 또는 명시적 산란 source가 필요하다. | UNMEASURED | A2-09, A2-11, A2-15 | |
| G05 | `src/lumina_atomic.c:2396-2456` | 3 | gate 시 O IV ground 한 준위(`g=6`)를 합성하고 Kramers 단면을 붙인다. | 실제 O IV level/continuum/target 원자모형이 필요하다. | UNMEASURED | A2-07, A2-13, A2-14 | |
| G06 | `src/lumina_plasma.c:17240-17268`<br>`src/lumina_cuda.cu:1852-1862`<br>`scripts/slurm_prod_dr7ion_ce17_ionlock.sh:46`<br>`scripts/slurm_skipsi_physical_champion.sh:115`<br>`scripts/slurm_ddc15_FI_prod.sh:180` | 3 | 생산 wrapper가 Si(`Z=14`)를 NLTE tau 갱신에서 제외하고 nebular 값을 유지한다. | 모든 선택 원소에 동일한 rate/population/opacity 소유권을 적용해야 한다. | UNMEASURED | A2-07, A2-08, A2-13, A2-14 | |
| G07 | `scripts/build_ddc15_epoch.py:117,130` | 3 | gold luminosity가 없으면 gas `T`의 Stefan–Boltzmann luminosity를 쓰고, `T_rad` seed를 `t^-1`로 외삽한다. | 관측/모델 luminosity와 수송된 radiation field에서 일관된 seed를 생성해야 한다. | UNMEASURED | A2-16 | |
| G08 | `scripts/build_ddc15_real_composition.py:20-21,173-176` | 3 | 미추적 원소를 버린 뒤 추적 원소만 합 1로 재정규화한다. | 전 조성의 질량·전자·opacity 기여를 보존하거나 누락종의 폐합 오차를 명시해야 한다. | UNMEASURED | A2-16 | |
| G09 | `src/lumina_cmfgen.c:5120,5148` | 3 | deterministic `Jbar`에 실제 표본수가 아닌 `jbar_count=1000`을 써서 소비 guard를 통과시킨다. | estimator 종류와 유효성 계약을 별도 표지하고 실제 통계량만 count로 전달해야 한다. | UNMEASURED | A2-03, A2-06 | |
| G10 | `src/lumina_main.c:171-174`<br>`src/lumina_cuda.cu:7118-7121`<br>`scripts/slurm_prod_dr7ion_ce17_ionlock.sh:44`<br>`scripts/slurm_skipsi_physical_champion.sh:114`<br>`scripts/slurm_ddc15_FI_prod.sh:179` | 3 | 생산 경로가 첫 5회를 non-NLTE로 돌린 뒤 NLTE를 켜는 history-dependent seed를 사용한다. | 동일 물리 고정점에 도달함을 seed 독립성으로 입증하거나 처음부터 같은 방정식을 풀어야 한다. | UNMEASURED | A2-07, A2-13, A2-18 | |
| G11 | `scripts/deck_quarantine_driver.py:156-165,168-184,462-464` | 3 | `_active` 생산 덱을 만들 때 model-spec에 없는 원소의 mass/abundance 행과 비활성 ion의 zeta 행을 필터링한다. | 전체 조성의 질량·전자·opacity 기여를 유지하고, 원자자료 미연결은 물리적 0으로 바꾸지 않는 명시적 계약이 필요하다. | UNMEASURED | A2-16 | |

## 5. 부류 4 — Phase 주석 고고학 (4)

`Phase N - Step M` 문자열은 1,703행, 6파일에서 발견됐다. 대부분은 이미 완성된 provenance 주석이므로 등재하지 않았고, 주석과 현재 본문을 함께 읽어 아직 임시 계약인 네 기전만 남겼다.

| ID | 파일:행 | 부류 | 현 코드가 하는 것 | 정확 물리가 요구하는 것 | 영향 추정 | A-2 단계 매핑 | 처분 |
|---|---|---:|---|---|---|---|---|
| P01 | `src/lumina_transport.c:12-35,496-500` | 4 | Phase 3 frame transform가 명시적으로 “partial relativity only”이며 1차 Doppler 식을 쓴다. | Lorentz factor를 포함한 일관된 frequency·direction·energy 변환이 필요하다. | UNMEASURED | A2-11, A2-12 | |
| P02 | `src/lumina_main.c:419-420,478` | 4 | Phase 5 병렬 수송은 local estimator를 `n_lines=0`으로 만들고 `j_blue/Edotlu`를 수집하지 않는다. | line별 resonance estimator를 병렬 안전하게 누적해 `Jbar`/rate 생산자에 commit해야 한다. | UNMEASURED | A2-04, A2-06 | |
| P03 | `src/lumina_transport.c:341-409,426-432` | 4 | Phase 3 macro-atom은 빈/비정규 block을 BB 방출로 강제하고 BF/FF deactivation은 “would go here”로 남겼다. | 완전한 macro-atom transition graph가 모든 radiative·thermal continuum channel을 보존적으로 닫아야 한다. | UNMEASURED | A2-09, A2-15, A2-18 | |
| P04 | `src/lumina_cmfgen.c:3656-3665` | 4 | “STAGE-1 PROOF” fine emergent 경로가 static no-Doppler formal integral로 색만 시험한다. | observer-frame frequency coupling과 velocity field를 포함한 formal spectrum이 필요하다. | UNMEASURED | A2-11 | |

## 6. A-2 단계 매핑 역인덱스

단계 의미는 [`ORDER_L0_JNU_OWNER_BY_CODEX.md:650-673`](ORDER_L0_JNU_OWNER_BY_CODEX.md)의 A2-00∼18 계약을 따랐다. 한 항목이 여러 소비층을 지나면 모든 직접 단계에 나타난다. `NONE`인 열린 항목은 없다.

| A-2 단계 | 이 단계가 밟는 항목 |
|---|---|
| A2-00 | 없음 |
| A2-01 | 없음 |
| A2-02 | H07, H16 |
| A2-03 | H13, S01, S02, G09 |
| A2-04 | H02, H13, S01, P02 |
| A2-05 | H07, S02, S04, S07, S09, S15 |
| A2-06 | H07, H16, H17, H18, S02, S06, S11, S13, G09, P02 |
| A2-07 | H01, H04, H05, H11, H15, S03, S04, S09, S14, S15, S16, G01, G02, G03, G05, G06, G10 |
| A2-08 | H09, S03, S05, S07, S12, G06 |
| A2-09 | H10, H17, H18, H19, S08, S09, S10, S11, S12, S13, G04, P03 |
| A2-10 | H01, H03, H12, H14, H15, S06 |
| A2-11 | H08, H09, H16, H17, S05, S11, S12, G04, P01, P04 |
| A2-12 | H06, S05, P01 |
| A2-13 | H05, H07, H11, H18, S06, S07, S13, S14, S15, G03, G05, G06, G10 |
| A2-14 | H07, H09, S05, S07, G05, G06 |
| A2-15 | H06, H10, H18, H19, S08, S10, S13, G04, P03 |
| A2-16 | S17, S18, G07, G08, G11 |
| A2-17 | H01, S01 |
| A2-18 | H02, H05, H06, H08, G03, G10, P03 |

## 7. 탐지 방법과 커버리지 한계

### 7.1 기계 후보 수집

- `src`의 22개 `*.c/*.cu/*.h`를 `wc -l`로 합산해 50,372행을 확정하고 전 파일을 대상에 넣었다.
- 부동소수·지수형 리터럴 정규식으로 **5,691 후보 행**을 얻었다. 이 수는 `rg -n --pcre2`의 출력 행 수이며, 같은 행의 복수 리터럴은 한 후보 행으로 센다. CODATA, 기하계수, 배열 인덱스, 단위변환, selftest/bench 기대값을 판독해 제외했다.
- 대소문자 무시 패턴 `for now`, `placeholder`, `assume/assumption`, `approximate`, `TODO`, `XXX`, `roughly`, `simple`, `crude`로 **36 후보 행**을 얻어 앞뒤 함수와 caller를 읽었다. 문자열만 찾는 것으로 끝내지 않고 누락자료 fallback, 대표 edge, 고정 seed, 첫/일부 species 처리도 별도로 추적했다.
- `Phase [0-9]+ - Step`은 **1,703행/6파일**이었다. 주석 provenance와 현재 함수 본문이 어긋나는지 판독해 P01–P04만 잔류 임시 구현으로 판정했다.
- `Kramers`, `Regemorter`, `Axelrod`, `Boltzmann`, `Saha`, `grey`, `two-level`, `dominant`, `representative`, `ground-only`, `fallback`, `damp`, `max_iter`, `MAX`, `N_FREQ`의 정의와 모든 주요 CPU/GPU 소비자를 교차 추적했다.
- scripts는 덱/원자자료 생산·재생성 경로 14개(`build_toy06*`, `build_ddc15*`, `export_tardis_reference.py`, `expand_atomic_data*.py`, `deck_quarantine_driver.py`, `deck_regen_*_driver.py`)와 명시적 생산 수송 wrapper 6개(`slurm_*prod*`, `slurm_skipsi_physical_champion.sh`, `slurm_plain_ddc15_sn2002bo.sh`, 기준 paper wrapper)를 읽었다. sweep/probe/selftest 전용 wrapper는 생산 기본값으로 세지 않았다.

### 7.2 판독 규율

- 동일한 상수가 여러 backend에 복제되면 한 ID에 묶었다. 반대로 같은 위치라도 물리적으로 독립인 모델(예: 회색 gamma opacity와 비열적 에너지 분배)은 요구 정확 물리가 같을 때만 한 행에 묶었다.
- “정확 물리”는 현재 근사의 반대말을 임의로 만든 것이 아니라, 코드 주석이 이미 밝힌 상세평형·Milne·SE·Lorentz·observer-transfer 계약과 A-2 발주 계약을 기준으로 적었다.
- 영향값은 기존 대장에 수치가 있는 NE, vR/Kramers만 인용했다. 그 외는 모두 `UNMEASURED`로 두었다. 코드 주석의 감상적 수치나 서로 다른 gate의 결과를 해당 항목 영향으로 전용하지 않았다.
- clamp/floor 자체는 0건 등재했다. 예컨대 `W` cap, population/tau/source floor, 온도·단면·확률 clamp, rail retry, `max/min` 양수화는 이 census의 51건에 포함되지 않는다.

### 7.3 이 census가 못 찾는 것

- 정적 판독이므로 환경변수 조합별 실제 hit 수, 분기 빈도, 출력 민감도는 알 수 없다. 그래서 활성 wrapper가 확인되지 않은 gate도 “reachable 물리 경로”임을 명시해 등재하되 영향은 측정하지 않았다.
- 외부 바이너리 원자자료 내부의 잘못된 값, 생성 후 `/gpfs`에서 교체된 파일, 런타임 생성 코드, 링크되지 않은 별도 repository는 이 작업의 읽기 범위 밖이다.
- 숫자나 표식 없이 수학적으로 틀린 구현은 패턴 검색만으로 완전성을 보장할 수 없다. 이를 줄이기 위해 radiation field, population, opacity, emissivity, transfer의 producer→consumer를 함수 단위로 읽었지만, 50,372행 전체에 대한 형식검증 증명은 아니다.
- 생산 wrapper 범위는 파일명·주석·실행 명령으로 판정했다. 이름 없는 개인 wrapper나 문서 밖 수동 환경변수 주입은 발견할 수 없다.
- 행 번호는 기준일 working tree의 현재 내용 기준이다. 이후 코드 이동 시 ID와 기전 설명을 정본 키로 사용해야 한다.
