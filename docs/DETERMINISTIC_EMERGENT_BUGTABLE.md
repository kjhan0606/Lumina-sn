# Deterministic Emergent — Bug/Problem Resolution Table (2026-06-25)

Goal: one deterministic integrator producing **P-Cygni profiles + correct color +
fluorescence**, comparable to DDC15 CMFGEN gold (peak 6790Å, grn/nir 0.58).
Current best pieces: plasma ✅ (T_e 0.98, n_e dex 0.18); static-freqres color ✅
(peak 6782); P-Cygni ✅ via scatter source on *binned* field (contrast 2.0-2.4).
The unification has the bugs below.

## Problem inventory (observed today)

| ID | Problem | Evidence | Root-cause hypothesis |
|----|---------|----------|----------------------|
| **P1** | obs Doppler march **too-red** | 169874 scatter peak 11826 vs static-freqres 6782 on the *same fine field* | Doppler obs march (`cmf_obs_march_sob`) reddens the continuum: either D³ weighting, the `S_c=(χ_ab·B+χ_es·J)/χ_c` form, or `nu_cmf` interpolation. **The #1 blocker.** |
| **P2** | red-edge upturn | 169874 flux rises at 11000-12000 | `nu_cmf=q·nu_obs` leaves fine window `[nu_lo,nu_hi]` → `cmf_interp_nu_asc` clamps to edge value → spurious flux |
| **P3** | fluorescence pump **FAIL** | 169838 green Fe II S_l/B≈1.000 with CONSUME=1 | confounded: **P3a** TAUMIN=0.1 may have skipped the UV forest pump (count unknown — FINE_DIAG wasn't plumbed); **P3b** orthodox floor thermalizes pumped upper levels |
| **P4** | Fix-B/Fix-C not built | design only | genuine super-thermal fluorescence S_l would be killed by clamp (Fix-B) + orthodox floor (Fix-C) even if pump works |
| **P5** | cost: obs emergent every iter | ~28 min/iter × 8 | producer+obs emergent rerun each NLTE iteration; only the converged one is needed |
| **P6** | two-window (pump vs emission) | UV forest deposit blew up (169831) | pump needs UV 1000-3000 (line-resolved J̄), emission needs optical 3000-12000; one fine window can't cheaply cover both |
| **P7** | harness plumbing gaps | FINE_DIAG never reached binary (169873 wasted) | harness passes an explicit env allowlist; new gates must be added |

## Resolution plan (ordered by dependency)

| # | Step | Fix / action | Predicted result (PASS gate) | Contingency (if prediction FAILS) |
|---|------|--------------|------------------------------|-----------------------------------|
| **1** | **Isolate P1** ✅ DONE | Ran obs `OBS_CONTONLY=1` (169882) on fine field | **RESULT: CONTONLY-obs peak=6513 ≈ gold 6610 → Doppler CONTINUUM march is COLOR-CORRECT.** P1 is NOT a continuum bug. The too-red (full-scatter 8576) is the LINE path → line blanketing without fluorescence = **P1 ≡ P3**. | — |
| **2** | ~~Fix P1 continuum~~ **NOT NEEDED** | — | continuum march sound (step 1 verdict) | — |
| **3** | **P-Cygni + color milestone** | With P1 fixed, full scatter obs (lines on) on fine field | peak ≈ 6800 (gold 6790) AND P-Cygni contrast: blue >2, CaII-NIR >2 (like standalone 2.0-2.4) → **MILESTONE: P-Cygni + correct color**. Draw figure vs gold | P-Cygni weak (contrast <1.5) → output grid too coarse (raise NObs) or DVRES too large (lower `OBS_DVRES` to 20 km/s) |
| **4** | **Fix P5 cost** | Gate obs emergent to the final converged iter only (pass iter index into `cmfgen_fine_jbar`, or run a one-shot post-convergence emergent call) | run time ≈ baseline (~18 min/iter) + one obs pass (~10 min); total ~2.5 h not 4+ h | If iter index unavailable cheaply → run N_ITER=8 plasma-only first, then a separate single emergent pass reading the converged state |
| **5** | **Diagnose P3 pump** ✅ DONE | Ran `FINE_DIAG=1 TAUMIN=0.1 UV-window CONSUME=1 N_ITER=3` (169884) | **RESULT: skipped weak=586763/599100 = 97.9% → P3a CONFIRMED (TAUMIN=0.1 starves the UV forest).** BUT mechanism is **LIVE**: UV pump lifts Fe II 5031 to S_l/B=**1.34** in thin mid-vel shells 11-12; chain producer→consumer→NLTE→S_l works end-to-end. STARVED: only 8 super-thermal optical lines total; main carrier 5170 stays thermal. → re-test with TAUMIN=0.01 (169911 launched) to restore forest-overlap pump. | (was the contingency, now executed: lower TAUMIN 0.1→0.01; if green carriers still thermal → full forest / wide coupled solve) |
| **5b** | TAUMIN 0.01 재시도 | ✅ DONE (169911) | **green 신호 0 변화** (8 super-thermal 라인, max 1.45, 5170 sub-thermal — 169884와 동일). 단 0.01은 forest의 절반(16k/27k)만 잡음 → forest-overlap 미완 테스트 |
| **5c** | full forest TAUMIN=1e-3 | ✅ **DONE (169914)** | 30,547 라인(전부) deposit → super-thermal optical **여전히 정확히 8개, max 1.45, byte-identical**. 5170 thin shell11-17 **sub-thermal 0.65→0.29**(<1.0, floored 아님). → **forest-overlap 사망 + floor cap 아님 확정** |
| **5d** | **ROOT CAUSE 확정** ✅ **삼중검증** | 코드(`lumina_cmfgen.c:1736` producer가 forest를 `S_l→B(T_e)` thermal emitter로 deposit) + 데이터(UV 펌프 2300-2600Å Jline/B=**1.001** in green shell11-15, super-thermal 0%) + 물리 에이전트 | **펌프장이 green-방출 shell서 THERMAL → fluoresce할 super-thermal UV 광자 없음.** 병목=forest수도/floor/cascade/consumer 아님 = **producer가 thermal 펌프장 생성**. 라인 3.4× 늘려도 byte-identical 이유(thermal emitter 추가일뿐). | — |
| **6** | **Fix = producer 라인 산란 (ε<1)** ✅ 빌드 / ⚠️ **Test A 미작동(branch B)** | `lumina_cmfgen.c` `LUMINA_CMF_FINE_LINE_EPS`: (1-ε)·χ_line→χ_es(ALI), ε·χ_line→B emit. ε=1 byte-identical. 빌드(.preLineEps). | **Test A (169955)**: green shell11-15 UV Jline/B **thermal과 EXACTLY 동일 1.000/1.001/1.003** (안오름). super-thermal 0개. → fix 미작동 OR 두 confound 검정중 | **검정중**: (B①) **169958 ALI=96**(16→96, albedo~0.9 미수렴이면 J̄가 warm-start B에 갇혀 1.000) (B②) green shell UV 본질적 thermal(hot UV 미도달) → 물리에이전트 재질의(blue-pump 3000-4500 가설). EXACTLY 1.000 = warm-start 의심 강함 |
| **6b** | **J̄/B(λ,shell) 맵 진단** ✅ **DONE (169972)** | read-only J̄/B(λ) 2000-5500Å (`LUMINA_CMF_FINE_JMAP`). shell0 검증=optical 0.54 정상(artifact 아님). | **RESULT: super-thermal 펌프장 = far-UV 2000-2500Å에만**(green sh13 J/B 2050Å=133, 2450Å=1.85, 2550부터 <1). **blue 3000-4500 NOT super-thermal(branch-1 REFUTED)**. **NEW: optical(4000-5500) fine field COLLAPSE**(sh13 J/B=1e-4 vs sh0 0.54) — fine producer 광역 transport flag. | 펌프=far-UV(이미 UV창 커버)지만 line-J̄ consumer가 못 harvest(강선이 core 열평형, super-thermal은 inter-line 연속체) → **cheap window pump 불가 확정 → orthodox full-coupled 필수** |
| **6c** | **개념 교정: 형광=branching** | (물리에이전트) 진짜 형광 = 흡수광자 branching(UV→z6P°→optical, J̄=B여도 발생). 내 아키텍처 불가=optical 라인이 자기 binned-J 산란→분기 green 재흡수. 형광=optical 라인이 UV-pumped pop서 emit(scatter 아님) | — | coupled multi-level + 광역 line-resolved 필요(현 UV-only pump window로 불충분) |
| **7** | **Build Fix-B (b-ceiling)** | Replace `FINE_SL_CLAMP` scalar with per-level `b_j ≤ B(T_phot)/B(T_e)` cap gated on pump provenance (design §Fix-B) | super-thermal green S_l (1<S_l/B<10²) survives to emergent; cold-shell garbage stays clamped; no NaN/blowup | b-ceiling over-clamps (kills green) → T_phot mis-set (MC blue-tilt 168221) → validate T_phot first; or under-relax ω=0.25 |
| **8** | **Fluorescence emergent** | Two-window producer (P6): UV pump 1000-3000 (TAUMIN, coarse PPD) ∪ optical 3000-12000 (fine) → jbar_line_det for both; NLTE source obs emergent | **MULTI-BAND falsifier (169874 obs decomp, user read 2026-06-25): ①blue 3000-5000 model/gold 6.5→1× (absorb) ②green 5600-7300 0.57→1× (emit) — ①②must move TOGETHER (energy conserv) ③7700 dip deepen (0.44→0.24) ④NIR 0.94 hold.** green Fe II emission appears; P-Cygni preserved → **FULL deterministic emergent** | green up but blue stays high → not redistribution, continuum re-color artifact; both flat despite S_l>1 → 2nd root (green Fe II opacity/ionization too low) — pivot to opacity/ionization audit |

## Predicted end state
After step 3: **P-Cygni + correct color** figure vs gold (fluorescence green still
missing — the honest near-term milestone). After step 8: **full deterministic
emergent** with fluorescence. Steps 1-4 are the immediate critical path
(unblock + milestone + make iteration affordable); 5-8 are the fluorescence
science. Each step has a falsifiable gate so we never proceed on a wrong assumption.

## 🎉 MILESTONE (2026-06-26): P-Cygni + 정확한 색 달성
- scatter-source obs(170032): peak 6408(gold6595), grn/nir 0.508(gold0.583), P-Cygni(CaII trough 0.57). 적색화 원인=OBS extractor의 NLTE thermal 라인소스(cold 리셋)→scatter(W·B) fix. Step3 달성. 형광은 green에 직교(gold green=연속체색). fig 2026-06-26_obs_scatter_vs_nlte.png.

## Orthodox build progress (2026-06-26 자율)
- **P0 (optical 수송 붕괴) ✅ PASS**: cmf_solve_J advection=HB upwind, Courant β·ds~80 → e^-80가 radial 소멸 → optical J/B 1e-4. fix=implicit-upwind operator-split (LUMINA_CMF_ADV_SPLIT). 169993: optical 1e-4→0.89, far-UV 보존. fig 2026-06-26_phase0_fix_verify.png.
- **P1 (full-range producer 1000-12000) 🔄 169997**.

## Cross-cutting rules
- One change per run; verify config tag + flux scale + producer ran (FINE_DIAG) before trusting a result ([[feedback_verify_before_concluding]]).
- Launch emergent runs via `scripts/run_emergent.sh` (10-knob guardrail, [[feedback_emergent_launch_wrapper]]).
- Plasma gate (T_e≥0.95, n_e dex≤0.3) on every run — any regression = stop.
- Design reference: docs/FLUORESCENCE_DESIGN.md; status: [[project_autonomous_stage2_2026-06-25]].
