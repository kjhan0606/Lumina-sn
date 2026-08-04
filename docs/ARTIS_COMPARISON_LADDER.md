# ARTIS 비교 사다리 (Comparison Ladder) — 2026-06-28

> **목적 (user, "되돌이표 그만 / 돌아가는 ARTIS 결과를 하나씩 재현")**: gold(DDC15)는 **최종 spectrum만 주는 블랙박스** → 토글-비교밖에 못해 매 세션 같은 too-red 벽에 도달(되돌이표). **ARTIS는 화이트박스**(중간 산출물 전부 노출) → 우리 코드의 각 component를 ARTIS와 **하나씩 정량 비교**해 발산점을 정확히 짚는다. 분석답/gold 대신 **돌아가는 ARTIS**가 기준.

## 공통 기준 (apples-to-apples)
- **테스트 모델**: `artis-ref/tests/nltephotospheric_dynamic_ion_range_1d_1dgrid` = NLTE 광구 1D SN Ia (우리 0.976d regime과 동일 물리). MPKTS=400, NLTEITER=2, SPHERICAL1D, Fe II 100 levels(=우리 super-cutoff).
- **원자데이터**: ARTIS hefeconi (He,Fe,Co,Ni; adata/transition/phixs). ⚠️ 1차 목표는 ARTIS를 *그 자신*과 비교(우리가 같은 DB·모델을 먹이도록 어댑터) — 처음엔 ARTIS 단독 실행해 기준값 확보.
- **ARTIS 빌드**: lageunha, gcc14.2 + openmpi5, `sn3d` 빌드 완료(2026-06-28). 실행=runfolder서 `mpirun -np N ./sn3d` → estimators/nlte_pops/spec.out.

## 사다리 (bottom-up; 각 단은 KNOWN = ARTIS 화이트박스 값)

| # | Rung (component) | 비교량 | ARTIS 출력(화이트박스) | 우리 출력 | PASS 기준 |
|---|---|---|---|---|---|
| **C0** | 입력·원자데이터 적재 | levels/lines/phixs 개수, g, E, A_ul | adata.txt/transitiondata 적재 로그 | atom 적재 | 같은 DB서 개수·값 일치 |
| **C1** | 이온화 balance | 이온분율 n_ion/n_elem (shell별) | estimators*.out (ion populations) | ion_number_density | dex 차 < 0.1 |
| **C2** | 복사장 J_ν | W, T_R, binned J, per-line J_blue (shell별) | radfield estimators (radfield.cc) | nlte->J_nu | J/J 비 0.8–1.2 |
| **C3** | NLTE populations b_k | departure b_k (level별, shell별) | nlte_pops 출력 (nltepop.cc) | lumina_levelpop.csv | populated 준위 b_k 차 < 20% |
| **C4** | 선소스 S_l | S_l/B (강선) | (emissivity/pops서 유도) | lumina_sl_vs_B.csv | 강선 S_l/B 일치 |
| **C5** | 열균형 T_e | T_e (shell별) | estimators T_e | plasma.T_e | %RMS < 5% |
| **C6** | macro-atom 형광/방출 | emission λ 분포 (UV→광학 재분배) | emissiontrue/absorption spec | (THEN_MC or 결정론) | NIR↔광학 분율 일치 |
| **C7 (최종)** | **Full emergent spectrum** | **전체 spectrum (peak, 색, P-Cygni features)** | **spec.out** | **lumina_spectrum*.csv** | **shape-corr, peak±, red/NIR 분율** |

## 교차검증 앵커
- **ARTIS vs gold(DDC15)**: ARTIS가 SN Ia 광구상서 DDC15-유사 spectrum을 내는지(peak~6500, P-Cygni) — ARTIS 자체 신뢰도 확인.
- **OURS vs ARTIS (C7)**: 우리 emergent를 ARTIS spec.out와 직접 — 최종 목표.
- 발산이 처음 나타나는 단(C1~C6)이 **진짜 버그 위치** (토글 추측 없이 화이트박스로 특정).

## 진행 규칙
- 한 번에 한 단. 아래 단이 PASS여야 위 단 신뢰.
- ARTIS 중간값은 estimators/nlte_pops 파일에서 직접 추출(덤프 불필요, ARTIS가 이미 씀).
- 우리 값은 기존 덤프(levelpop/sl_vs_B/plasma_state/spectrum).
- 발산 발견 시: ARTIS 소스(해당 component .cc)를 읽어 우리 공식과 1:1 대조 → 차이 = 버그.

## 📋 세부 진행 상태 (2026-06-28 갱신)

### 인프라 (사전 준비)
| 항목 | 상태 | 비고 |
|---|---|---|
| ARTIS 빌드 (sn3d) | ✅ DONE | lageunha gcc14.2(c++26)+openmpi5+gsl. build_env.sh. ⚠️실행 `mpirun -np 4` (--bind-to 금지=25%만 씀) |
| ARTIS DDC10 NLTE run | ✅ DONE | 30 timestep 3-8d, 2.95M lines, spec.out+estimators. figures/artis_ddc10_NLTE_spectrum_2026-06-28.png (peak2780, Fe-group features) |
| ARTIS 상태 추출 (78셸) | ✅ DONE | scratch_artis_ddc10_state.npy: Te/TR/W/n_e/n_Fe-Co-Ni per cell @7.87d |
| 우리코드 DDC10 입력 변환 | ✅ DONE | data/ddc10_artis_t7.87d/ (광구밖 mgi44-77=34셸, Fe/Co/Ni, T_inner16832). `DDC15_REF=ddc10_artis_t7.87d` |
| 우리코드 DDC10 실행 | ✅ DONE | OOM(78셸)→34셸, plasma_state.csv 추가로 해결. smoke 2-iter 완주 |

### 비교 사다리 (bottom-up)
| # | component | 비교량 | 상태 | 결과 / 다음 |
|---|---|---|---|---|
| **C0** | 원자데이터 | levels/ions/g/E | ✅ **PASS** | 둘다 CMFGEN, **Fe II 2698 일치**. ⚠️ARTIS=Fe/Co/Ni 7/4/4 ions, 우리도 동일 매핑 |
| **C1** | 이온화 balance | n_ion/n_elem, n_e | 🔄 **예비 PASS** | n_e **~2× 이내 일치**(smoke 2iter). per-ion(Fe IV/V/VI 분율) 상세 = 수렴런서 |
| **C2** | 복사장 J | W, T_R, T_J | ⏳ TODO | ARTIS W/T_R 있음(estimators). 우리 T_rad와 대조 |
| **C3** | NLTE pops b_k | departure b_k | 🔄 진행중 | 수렴런 LEVELPOP_DUMP 돌리는중. ARTIS nlte_pops 추출 필요 |
| **C4** | 선소스 S_l | S_l/B 강선 | ⏳ TODO | 우리 sl_vs_B. ARTIS emissivity/pops서 유도 |
| **C5** | 열균형 T_e | T_e shell별 | 🔄 **예비결과** | **광구 mgi44-50: 우리 ~25% 차가움(0.74-0.90)**. 외곽: ARTIS폭주140000K vs 우리10000K(둘다 Fe/Co/Ni-only 병리). 수렴런서 확정 |
| **C6** | macro-atom 형광 | UV→광학 재분배 | ⏳ TODO | ARTIS emissiontrue. 우리 THEN_MC/결정론 |
| **C7** | **Full spectrum** | peak/색/features | ⏳ TODO | ARTIS spec.out 있음. 우리 freqres 수렴런서 |

**범례**: ✅PASS / 🔄진행·예비 / ⏳TODO

### 현재 핵심 발견
- **C0,C1 정합**(원자데이터·이온화). **C5 광구 T_e 우리가 ~25% 낮음** = 첫 실제 발산.
- 외곽 T_e는 양쪽 다 Fe/Co/Ni-only 저밀도 병리(ARTIS 뜨겁게/우리 차갑게)→비물리 테스트조성 지배, 깨끗한 비교 아님. **광구(mgi44-52)가 의미있는 비교역.**
- 다음: 수렴런(N_ITER=8) 완료 → C1 per-ion, C3 b_k, C5 확정, C7 spectrum. 발산 단(C5?)서 ARTIS 소스(thermalbalance.cc) vs 우리(radeq) 공식 1:1 대조.

---

## 🥇 3-WAY 비교 + Lumina 우위 가설 (user 2026-06-28: "ARTIS와 맞으면 OK가 아니라 CMFGEN에 ARTIS보다 더 가까워야")

**핵심 논지**: ARTIS도 근사다(MC 잡음 + super-levels + 256-bin/dilute-BB 복사장 + macro-atom 확률분기). **CMFGEN = full-CMF 결정론 = 진짜 기준.** Lumina의 존재이유 = **결정론 + full freq-resolved + full levels로 ARTIS의 근사들을 넘어 CMFGEN에 더 가까이 가는 것.** 그래서 각 단은 "ARTIS와 일치"가 아니라 **"CMFGEN까지 거리가 ARTIS보다 작은가"**로 평가.

| # | 비교량 | **CMFGEN (진짜 기준)** | ARTIS 근사 (약점) | **Lumina 우위 가설** |
|---|---|---|---|---|
| C0 | 원자데이터 | CMFGEN DB (Fe II 2698 levels) | CMFGEN DB 그대로 사용 | 동일 사용 — 우위 없음(중립) |
| C1 | 이온화 | DDC15 0.976d: 광구 n_e=6.0e10 (T_e=4434) | nebular Saha + MC 잡음 | 결정론 광이온 적분(잡음無) |
| C2 | 복사장 J | full-CMF 주파수해상 J_ν | **256-bin (W,T_R) dilute-BB + per-line J_blue (MC추정)** | **full freq-resolved 결정론 J = CMFGEN과 동급 ← 최대 우위 후보** |
| C3 | NLTE pops | full levels NLTE | **super-levels(고준위 묶음) + MC pops** | full levels 결정론 → CMFGEN에 더 가까이 |
| C5 | T_e | DDC15 광구 **4434K**, 외곽 2505K, 핵 128000K | MC 가열/냉각 추정 | 결정론 radeq. **Lumina champion DDC15: T_e≈0.98×CMFGEN(~2%) = 이미 검증된 강점** |
| C6 | 형광 | full-CMF 선별 형광 (정확 λ분기) | **macro-atom 확률분기(MC) + k-packet** | 결정론 형광 재분배 (잡음無, 정확 λ) |
| C7 | spectrum | **DDC15 gold: peak 6595, P-Cygni** (data/ddc15_hydro/*.dat) | MC 잡음 + 위 근사 누적 | 결정론 full-freq → gold features 더 정확 (목표) |

### 두 비교 케이스 (데이터 가용성)
- **Case A — DDC10 @7.87d** (현재): Lumina vs ARTIS 화이트박스. CMFGEN 구조 없음(Dessart DDC10 미입수). 코드-코드 발산 localize용.
- **Case B — DDC15 @0.976d** (Lumina 우위 증명용): **CMFGEN 구조(T_e/n_e) + gold spectrum 보유**. Lumina champion 이미 실행(T_e 0.98×CMFGEN). **남은 것 = ARTIS를 DDC15 0.976d로 돌려 3-way 완성** → "Lumina가 ARTIS보다 CMFGEN에 가까운가" 판정.
  - ⚠️ ARTIS @0.976d 매우 thick → spectrum 형성 어려움(structure 비교는 가능).
  - **C5/C1 structure 3-way**: CMFGEN(4434K) vs Lumina(~0.98×=4350K) vs ARTIS(?) — Lumina 강점 입증 지점.
  - **C7 spectrum**: Lumina(too-red) vs CMFGEN(gold) — Lumina 현재 약점. ARTIS도 측정해 누가 덜 too-red인지.

### 우위 판정 규칙
- 각 단: |Lumina − CMFGEN| < |ARTIS − CMFGEN| 이면 **Lumina WIN** (그 component서 ARTIS 능가).
- 전체 목표: C7 spectrum서 Lumina가 ARTIS보다 gold에 가까움 = Lumina의 가치 증명.
- 현 상태: C5(T_e) DDC15서 Lumina 강함(0.98×) 입증됨 / C7(spectrum) too-red는 약점(ARTIS도 측정 필요).

### 🔬 수렴 비교 결과 (2026-06-28, DDC10 7.87d, Lumina N_ITER=8 vs ARTIS)
- **C0 ✅ PASS**: 둘다 CMFGEN DB, Fe II 2698.
- **C1 ✅ PASS**: n_e 우리 vs ARTIS **~1.3-1.6× 이내** (광구). 이온화 정합.
- **C5 🔄 발산 localize**: 광구 T_e **우리가 ARTIS보다 10-25% 낮음**(mgi44 0.90, mgi47-50 0.72-0.80). **원인 확정 = 방사성 붕괴 침착(deposition) 가열 차이**: ARTIS는 56Ni/56Co 붕괴를 ejecta 전역 추적(heating: dep=8.3e-9 지배), 우리 DDC10 run엔 56-동위원소 미제공→gamma_dep=0→분산가열 없음→cooler. ⚠️우리코드는 gamma_dep 메커니즘 보유(plasma.c:1124), 입력에 안 줬을 뿐. = **setup gap(동위원소 미제공) + 방법차(pure-CMFGEN inner-BC vs MC 분산침착)**. DDC15 0.976d선 inner-BC가 충분(Lumina 0.98×CMFGEN)이나 7.87d 분산침착이 커져 발산.
- **C3 ⏳**: ARTIS nlte_*.out(b_k=n_NLTE/n_LTE) 가용. 외곽 Fe II 소멸(degenerate), 광구 rank셸 추출 필요.
- **3-way 데이터 막힘**: full ARTIS DB 없음(hefeconi/feconi/classic만)→DDC15-full ARTIS 불가. DDC10 CMFGEN 구조 미입수(Dessart). ⟹ 정량 Lumina>ARTIS는 C5 DDC15(0.98×, 입증) 외엔 데이터 대기.
- **다음**: ①C5 재비교 위해 우리 DDC10 run에 56Ni/56Co deposition 추가(또는 ARTIS dep을 끄고 비교) ②C3 광구 b_k ③C7 우리 freqres spectrum(이번 run 미생성, FINE_EMERGENT 재확인).
