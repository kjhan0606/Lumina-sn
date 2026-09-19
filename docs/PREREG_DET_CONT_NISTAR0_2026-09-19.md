# 사전등록 — DET-CONT-NISTAR-0: 「연속체 방출률 n_i* 재정의」는 항등(이미 그 형) — CONT 초열비 S_bf/B 10^79 의 인자별 귀속 (2026-09-19)

기안 = **Fable fresh**(`claude-fable-5-1`, 개정25-b). 상태 = **기안 / 경량 사전감사 대기(개정25-d)**. 실행 승인이 아니다.
근거 = `docs/BRIEF_CONT_NISTAR_2026-09-19.md`(운전자 브리프, 141ca96) · `docs/READOUT_CONT_NLTE_SUPERTHERMAL_2026-09-18.md` · `docs/PREREG_DET_CONT_NLTE_BALANCE_2026-09-18.md`(형식 계승) · `docs/PREREG_DET_DR_ON_2026-09-19.md`(base 후보). 소스 정본 = 동결 트리 `/gpfs/kjhan/lumina/cmfgen_map/lteunmap_eps_20260917/source`(CONT run1 379330 이 돈 트리, 행 번호는 이 트리 기준; repo 대응 행 병기).

## 0. ★기안자 차단 통지 — 브리프 §근인은 성립하지 않는다 (런 0회, 저자 실행 grep/sed)

브리프: 「Lumina `S_bf = 2hν³/c²·e^{-x}/Cinv_l`, `Cinv_l = saha_m·n_level/g` ⟹ 분모 n_level 고갈 시 S_bf→∞; CMFGEN η∝n_i*(고정) 유계」.
실측(동결 트리 `src/lumina_plasma.c`):
- 8548 `chi_raw = n_level·σ`; 8562 `chi_spont_base = chi_raw·Σp`; 8598–8613 `S_l = (2hν³/c²)·e^{-x}/Cinv_l`, `eta_opacity = chi_spont_base`; 8621 `eta_bf += eta_opacity·S_l`.
- 8336–8339 `saha_m = 2U_next/(n_next·n_e·(h²/2πm_e kT_e)^{3/2})`; 8416 `Cinv_l = saha_m·n_level/g`.
- ⟹ **η_bf(level) = n_level σ Σp · (2hν³/c²) e^{-x} · g/(saha_m n_level) = (2hν³/c²)·σ·[n_next n_e (g/2U_next)(h²/2πm_e kT_e)^{3/2}]·e^{-h(ν−ν_edge)/kT_e}.** n_level 은 **소거**된다. 이것이 정확히 CMFGEN `genopaeta_v10.f:266-268` `ETA += (2hν³/c²)·α·n_i*·e^{-hν/kT}`(`n_i* = HNST·COR_FAC`, :192 · `ltepop.f:36`) 이다. 코드 주석 8599–8606 이 이를 명시한다: 「= σ(2hν³/c²)(n_e n_+ g_l saha/2U_+)e^{-x} — **no denominator, always bounded/positive**. (The b e^x−1 source-function form lases … → the epay4 J=1e42 runaway.)」 ⟹ 브리프가 제안한 교체는 **과거에 이미 시행된 수리**이고, 지금 트리의 η_bf 는 이미 n_i* 형이다(repo `src/lumina_plasma.c` 8506–8517 동일).
- 브리프가 S_bf 라 부른 `S_l` 은 방출률이 아니라 **η/χ_raw** 이다. census 의 비(동결 `src/lumina_cmfgen.c` 2436–2445) = `bf_get_eta/(bf_get_chi·B)` = η/(n_l σ B) ≈ **1/b_l**, b_l = n_l / n_l*(n_next, U_next, n_e, T_e) — 발산하는 것은 η 가 아니라 **χ_bf ∝ n_l 의 붕괴 또는 n_l* 의 폭증**이다. CMFGEN 도 같은 형(χ = α(n_l − n_i*e^{-hν/kT}) ≤ α n_l)이라 같은 비를 낸다.
- CMFGEN 과의 유일한 형식 차이 = χ 의 유도방출 항 `−n_i*e^{-hν/kT}`. CONT run1 은 `LUMINA_FIX_BF_STIM_RECOMB` 미설정(exports 부재 → 7480 기본 0) ⟹ χ_bf = n_l σ. 그 항을 넣으면 χ 는 **더 작아지거나 음수**가 되고(조립기 2420 이 `chi_bf<0→0` 클램프), S 는 더 커진다 — 유계 기전이 될 수 없다.
- **결론**: 「n_i* 재정의」 변경집합은 η 에 대해 **항등(no-op)** 이다. 이 위에 Opus 4.8 구현·판정런(≈20 GPU-h)을 발주하면 물리 측정 0 이 예정된다. 메모리 `reference_cmfgen_bf_emissivity_lte_referenced.md` 의 「핵심 기전 = 방출률 정의」는 정정 대상이며(운전자 09-19 정정 완료), 그 노트가 「보조」로 둔 CMFGEN 의 동시 선형화·보정 상한·점유확률이 실제 방어층이다(이 계약 밖, §8).
- ⟹ 이 사전등록은 요청된 계약을 기안하지 않고, **런 발주 3요건의 첫째(기전 오프라인 특정)** 를 채우는 rung 으로 축소한다. 표적 교체 여부는 user 결정(Q-GOAL «폐합 차단» 규칙과 같은 처분) — **user 2026-09-19 「값싼 것을 먼저」 로 이 census 승인**.

## 1. 계약 — 물리 계약 1개

**「CONT run1 census 의 초열비(셸별 max 10^39–10^79, mean 10–30)는 η_bf 정의의 결함이 아니라(η_Lumina ≡ η_CMFGEN, 항등 시연), 비 1/b_l = n_l*(n_next, U_next, n_e, T_e)/n_l 의 인자 중 하나의 붕괴이며, 그 인자·이온쌍·준위·인구 권위(NLTE 커밋 대 LTE 폴백 대 `ion_number_density`)는 인자별 census 1회로 특정된다.」**
산출 = (a) 항등 시연(런 0회, selftest) (b) 셸별 최상위 기여 빈의 인자 분해표 50 행(계측 런 1회 또는 CPU 리플레이 — §6 결정 항목). 판정런 아님·스펙트럼 아님.
⚠**말하지 않는 것**: 수리안(권위 통일·IV/V NLTE 승격·솔버 보정 상한·음-불투명도 처리 중 무엇인가) — 귀속 뒤 후속 계약. 황 이동 기대치는 이 단에 없다(잣대를 만들 물리 변경이 없으므로 「구성상 참」 예측 금지 규칙에 따라 등록하지 않는다).

### Q-GOAL (기안자 자기 답 — 경량 사전감사가 독립 재판정)
① 줄이는 칸: 직접 줄이는 칸은 없다. 이 단은 중간 셸 9–24 S II/III 칸(CONT 가 악화시킨 그 칸)의 **차단 원인의 자리**를 특정한다. ② 산출은 물리 측정이 아니라 **기전 귀속**(인자 분해)이다 — 단, 대안(브리프 원안)은 항등 변경 위의 GPU 20 h 라 물리 측정 0 이 확정적이므로, 이 단이 그것을 대체한다. ③ 더 가까운 대안: readout §6-4 「S/Si/Ca IV·V NLTE 승격」이 열려 있으나, 10^79 의 자리가 IV·V 쌍인지 II·III 준위인지 U_next 인지 모르는 채 고르면 귀속 불가 — 이 단이 그 선택의 근거를 만든다. ④ 건너뛰면 브리프 원안(무효 변경)이 발주되거나, 후속 표적을 추정으로 고른다.

### Q-LEAN (기안자 자기 답)
L1 물리량 = 셸별 최상위 빈의 인자 분해(n_l·n_next·U_next·n_e·T_e·e^{(χ−E_l)/kT}·권위 플래그) 50 행 + 항등 selftest 1. 신규 = env 1 · census 1 종(bf 빌더, OFF byte-동일) · selftest 1. 소비처: 분해표 → §5 E-1 분기 → 후속 계약 표적. L2 게이트 4(G-ID·G-RUN·G-ALG·G-ATTR) — 각각 실재 가능한 실패(키 혼입 / 미완주 / 항등 불성립 / census 가 max 를 재현 못 함). L3 = 1 걸음(계측 1 + 런 ≤1). L4 삭제 후보: 없음 — 추가하지 않은 것: 판정런·황 기대치·새 판독기·CMFGEN χ 유도항 구현·음-불투명도 처리.

## 2. 기전 — 런 0회 확정분

- **항등**: §0. η_bf 는 분기 셋(Milne `use_milne_l`·열 폴백 `η=χ_contrib·B`·비활성) 모두 유계. 따라서 절대 방출률은 유계이고, census 비 10^79 는 **비**(η/χ_bf B)의 발산이다. 조립기 S_fixed = (η_bf+χ_ff B+η_ln)/χ_t 도 분자 유계.
- **비의 대수**: 준위 l(이온 X, 다음 이온 X+1): 1/b_l = n_l*/n_l, n_l* = n_next n_e (g_l/2U_next)(h²/2πm_e kT_e)^{3/2} e^{(χ_X−E_l)/kT_e}. **미추적(LTE 폴백) 준위**(8364–8371 `n_l = n_ion·f_LTE(l;T_e)`): 1/b_l = (n_next/n_ion)_실제 / (n_next/n_ion)_Saha(T_e) — **준위 무관**, 이온쌍의 Saha 이탈비. **추적(NLTE 커밋) 준위**(8357–8363): 1/b_l 은 준위별.
- **mean 10–30 은 과이온화 비 그 자체다** [저자 실행]: CONT run1 `physics_DET_iter0001.ion.csv` gen 4 셸 20: S q2/q1 = 5.12e6/2.21e4 = **231**(stage 권위) · 5.13e6/1.38e4 = **371**(nlte 권위); base 376251 r = 8.79(Saha(T_e)) ⟹ 1/b ≈ 26–42 대 census 셸 20 mean 12.8(pass 3)·9.6(pass 4) — 0.5 dex 안(전 엣지 가중 평균이라 자릿수 일치만 주장). ⟹ mean 은 정직한 NLTE 재결합 연속체(CMFGEN 동형)이며 결함이 아니다.
- **max 10^79 는 이온쌍 Saha 이탈로는 불가능하다**: 셸 20 T_e 8849 K(kT 0.763 eV)에서 미추적 쌍의 1/b = r_실제/r_Saha ≤ n_element/(n_ion r_Saha) — r_Saha(III→IV, χ 35 eV) ~10^{-6}–10^{-12} 라 최대 ~10^18. 10^79 에 닿으려면 (i) **추적 준위의 n_l 이 n_l* 대비 10^{-79}**(단 8373 `n_level<1e-30 → continue` 라 n_l* ≥ 10^49 필요 — 물리적 n_l* 상한 ~10^10 과 모순) (ii) **U_next_m 소멸**(8331 `U_next_m = partition_functions[ip_next]`; 검사는 `>0 && isfinite` 뿐 — 주석 8323 「U floor 1 — tiny-positive U exploded the v1 Milne」이 말하는 floor 는 이 블록에 없다; η ∝ 1/U_next 라 U=10^{-70} 이면 그대로 10^70) (iii) **깊은 엣지** e^{(χ_X−E_l)/kT} ~10^85(χ−E_l ≈ 150 eV — `chi_erg` 자리표시자·다음 이온 오배정 포함) 중 하나이거나 곱이다. 셋 다 **계측 없이 갈 수 없다** — 이것이 이 단의 계측 정당성이다.
- **가용 산출 한계** [저자 실행 find]: CONT run1 에 `nlte_levels_iter*.csv`(`LUMINA_NLTE_LEVEL_DUMP=1` 설정됐으나 미기록)·`lumina_ion_pops.csv`(취소로 미커밋) 없음; ion.csv 는 q1·q2 만. ⟹ 준위·U_next 인자는 **기존 산출로 복원 불가** — census 가 필요하다.
- **부차(기재)**: 조립기 census 는 `bf_get_chi`·`bf_get_eta` 가 ff 를 포함한 격자를 별도 보간하므로 엣지 근방 비는 보간 잡음을 가진다(10^79 규모는 설명 못 함). CONT 실측 max 는 pass 2 에 이미 내핵 10^39·셸 4–11 10^68, pass 3 셸 17–24 10^76–10^79(셸 20 1.60e79), 외곽 47–49 10^17–10^21 ⟹ **첫 외부반복 안에서 발현**하므로 계측 런은 ITER=1 로 족하다.

## 3. 수리안 — 변경집합 (구현 Opus 4.8; ★src 편집 1개 규칙: DET-DR-ON 385315 폐합 후 착수)

원칙: 물리 경로 무변경. env `LUMINA_BF_MILNE_CENSUS`(기본 0 = byte-동일·출력 없음). ON 이면 bf 빌더가 셸마다 **비 η_l/(χ_raw,l·B) 최대인 (준위, 빈) 상위 3** 을 stderr 1 행/셸 `[BF-MILNE][CENSUS] shell=<s> rank=<k> Z=<Z> stage=<st> level=<l> E_eV=<> nu_edge=<> nu=<> n_l=<> n_next=<> U_next=<> n_e=<> T_e=<> exp_edge=<e^{(χ−E_l)/kT}> ratio=<η/(χ_raw B)> auth=<nlte|lte_fb> milne=<0|1>` 로 낸다. 항등 `ratio = (2hν³/c²)e^{-x}/(Cinv_l·B)` 는 **구성상 참** — 배선 음성대조로만 쓴다(예측 아님). 열 값은 빌더 지역 변수 그대로(재계산 금지). 조립기·GPU 경로·EPAY 무접촉.

### 변경집합 — BF_MILNE_CENSUS (표)

| 경로 | 담당 | 내용 |
|---|---|---|
| `src/lumina_plasma.c` | Opus 4.8 | bf 빌더 준위 루프에 상위 3 추적·셸 종료 시 census 행(게이트 ON 만); OFF byte-동일 |
| `src/env_universe.h` | Opus 4.8 | `LUMINA_BF_MILNE_CENSUS` 등록(COUNT 갱신) |
| `tests/bf_milne_identity_selftest.c` | Opus 4.8 | (a) **항등**: 격자 {T_e 4e3–2e4, n_e 1e6–1e10, n_next 1e-20–1e8, U_next 1–50, ν/ν_edge 1–5} × n_level 1e-30–1e10 에서 빌더 식 η(8598–8621 연쇄)과 CMFGEN 식 (2hν³/c²)σ n_i*e^{-hν/kT}(n_i* = HNST·COR_FAC 재구성)의 상대차 ≤ 1e-12, **η 의 n_level 무관** ≤ 1e-12; 음성대조 = 구형 `(2hν³/c²)/(b e^x−1)` 로 바꾼 사본은 n_level→1e-30 에서 발산(FAIL 시연) (b) census: OFF 출력 0 행·ON 셸당 3 행·`ratio` 항등 ≤ 1e-9; 음성대조 = 픽스처 U_next=1e-80 주입 ⟹ rank 1 이 그 준위·`U_next=1e-80`·ratio ≈ 1e80 을 보고, n_level=1e-29 주입 ⟹ auth·n_l 로 보고 |

### 변경집합 끝

## 4. 게이트 명부 (표)

| ID | 조건 | 음성대조 |
|---|---|---|
| G-ID | base = §6 ① 선택 런의 `input/resolved_lumina.exports` 대비 차이가 정확히 `LUMINA_BF_MILNE_CENSUS`(=1)·`LUMINA_CMF_CONT_NLTE_SOURCE`(=1, CONT 소스 재현 — 비 10^79 는 이 게이트 ON 의 조립기 census 로만 대조 가능)·`LUMINA_PURE_CMFGEN_ITER`(=1) [DR-on base 면 DR 5 키는 그대로 상속]; `LUMINA_ION_NT`·`LUMINA_NT_FULL`·`LUMINA_FIX_BF_STIM_RECOMB` 부재; 소스 차이 = §3 표 3 파일(기준 = 동결 `source.sha256`); 러너 편집 = ITER 검사·`outer_iterations`·job-name·`--time`(직접 편집, sed `\n` 금지), 해시 기재 | 다른 키 주입·소스 4 파일째 변조 ⟹ FAIL |
| G-RUN | 외부반복 1 완주(`RUN_FOOTER.txt` `terminal_state=SUCCESS`, `outer_iterations=1`) 또는 CPU 리플레이 완주; stderr 에 `[cmfgen_assemble][CONT-SRC]` 셸 0–49 × 패스 전건, `[BF-MILNE][CENSUS]` 셸 0–49 × 3 행 × 패스 전건 | 행 결손 ⟹ FAIL |
| G-ALG | selftest §3 (a)(b) 전건 PASS **그리고** 음성대조 2 건이 FAIL 을 시연(로그 첨부) | 음성대조가 PASS 로 나오면 게이트 FAIL |
| G-ATTR | 셸 17–24 의 마지막 패스에서 census rank-1 `ratio` 가 같은 패스·같은 셸의 `S_bf_over_B_max` 와 \|Δlog10\| ≤ 1(조립기 보간·ff 희석 여유); rank-1 행의 인자 곱 항등 ≤ 1e-6 | 조립기 max 를 1e79 로 위조하고 census 를 건드리지 않으면 FAIL; 인자 열 바꿔치기 ⟹ 항등 FAIL |

### 게이트 명부 끝

## 5. 기대치 — 런에서만 얻는 수 (예측은 §2 의 후보 셋; 잣대를 base 에 먼저 적용: base 376251 은 CONT 게이트 OFF 라 census 행이 없고 G-ATTR 은 정의상 미시연 ⟹ NIS0-X)

| ID | 양 | 예측 | **기대(분기 조건)** |
|---|---|---|---|
| E-1 귀속 | 셸 17–24 rank-1 행의 **지배 인자**: `ratio` 를 (n_l*/n_l) 로 쓰고 log10 몫이 가장 큰 항 ∈ {n_l 붕괴(auth=nlte), 이온쌍 Saha 이탈(auth=lte_fb), U_next < 1e-3, exp_edge > 1e40} | §2: (i) 은 n_l* ≥ 1e49 필요라 배제 예상, (ii)/(iii) 중 하나 | **8 셸 중 ≥ 6 이 같은 (Z, stage) 쌍·같은 지배 인자** |
| E-2 mean 재현 | 셸 12·16·20 조립기 mean 대 ion.csv 의 (n_next/n_ion)_실제/r_Saha(base 376251) | §2 실측 26–42 대 12.8 | **0.1 ≤ mean/(r_실제/r_Saha) ≤ 10** (기재용 — 「mean = 과이온화 비」 독해의 확인) |
| E-3 내핵 대조 | 셸 0–3 rank-1 `ratio`(CONT 실측 max 2.4e39–3.7e39) | 같은 쌍이면 exp_edge 가 T_e 로 줄어 낮아야 함 | 셸 17–24 와 **같은 지배 인자면** 기재, 다르면 E-1 의 「6/8」 로만 판정 |

### 분기 (표)

| 분기 | 조건 | 의미 |
|---|---|---|
| NIS0-X | G-ID/G-RUN/G-ALG/G-ATTR 중 미충족 | 미시연 |
| NIS0-A | 게이트 전건 **and** E-1 | **귀속 완료** — 지배 인자·이온쌍 확정. 후속 계약 표적 = 그 인자(권위 통일 / IV·V NLTE 승격 / U_next 계산 경로 / 엣지 데이터) — user 결정 |
| NIS0-B | 게이트 전건 **and not** E-1 | 귀속 분산(셸마다 다른 쌍·인자) — 판정자가 분해표로 명명; 후속은 다중 표적 |
| NIS0-C | G-ID·G-RUN·G-ATTR 충족 **and** G-ALG (a) FAIL | 항등 불성립 — §0 이 틀렸다는 뜻. 브리프 원안(n_i* 재정의)을 **다시 기안**(Fable) |

### 분기 끝

재분기 금지. 판정기 = 운전자 도구(census/ion.csv 파서, `READOUT_TOOLS.sha256` 기재; 소스 아님). 음성대조: base 376251 ⟹ census 없음 ⟹ NIS0-X; CONT run1 stderr(census 부재) ⟹ NIS0-X.

## 6. 자원 · 실행 · 결정 항목(운전자·user)

새 RUN 루트 `/gpfs/kjhan/lumina/cmfgen_map/cont_nistar0_20260919/run1`. 봉인 base 읽기만. 순서: 경량 사전감사(Opus 4.8 fresh) → 게이트 4종 → **DR-ON 폐합 대기** → Opus 4.8 구현+selftest → Fable 코드 간단 감사(Q0) → 빌드 → 스테이징 → 실행 → 판독 → 판정(운전자) → 종료감사(Opus 4.8 fresh; 표적 교체가 걸린 단이라 생략하지 않음).
① **base**: (a) DR-on 385315(`dr_on_20260919/run_h100a100`, 착지 후 폐합 조건부 — 후속 수리가 이 위에서 돌 것이므로 권장) (b) 376251(CONT run1 과 동일 계보라 census max 대조가 가장 직접적). 기안자는 못 박지 않는다 — (a) 면 G-ATTR 의 ±1 dex 여유가 DR 차이를 흡수하는지 판정문에 기재.
② **실행 티어**: GPU ITER=1(376251 실측 5 h/2 반복 ⟹ ≈2.5 h, `--time=05:00:00`, 헤지 여부) 대 CPU 리플레이(grammar-debug/lageunha — 조립기 CPU 경로 소요 미실측이면 GPU).
③ census 상위 K=3 고정(늘리지 않는다).

## 7. 분장 장부 (개정25 · 규약 / 실제 / 위반)

| 단계 | 규약 | 실제 | 위반 |
|---|---|---|---|
| 기안 | Fable fresh(25-b) | **Fable fresh**(`claude-fable-5-1`, 이 문서) — 브리프 근인 반증 포함 | — |
| 경량 사전감사 | Opus 4.8 fresh(25-d) | 미집행 | |
| 구현+selftest | Opus 4.8 | 미집행(DR-ON 폐합 후) | |
| 코드 간단 감사 | Fable fresh(25-a) | 미집행 — ⚠기안자와 같은 모델·다른 컨텍스트(개정25-b 기재 편향) | |
| 빌드·스테이징·제출·판독·판정 | 운전자 | 미집행 | |
| 종료감사 | Opus 4.8 fresh | 미집행 | |

## 8. 주장 상한

말할 수 있는 것: 「Lumina η_bf 는 CMFGEN n_i* 형과 항등이다」(런 0회, selftest 로 박제)와 「10^79 비의 지배 인자·이온쌍·권위」(census 1회). 말하지 않는 것: 수리안의 옳음, 황의 CMFGEN 방향 이동, CMFGEN 의 실제 유계 기전(동시 선형화 `do_var_cont.f`·보정 상한 `solve_for_pops.f`·점유확률 `occupation_prob.f` — 후보로 기재, 미검증), CMFGEN χ 유도항·음-불투명도 처리(이번 밖), 스펙트럼·자유-T_e. twin-T_e 핀 조건부. §2 의 「(i) 배제」는 8373 floor 를 전제로 한 대수 추정이며 census 가 최종.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 — BF_MILNE_CENSUS (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "(?:src|tests)/[A-Za-z0-9_/.]+\\.(?:c|h)",
    "symbol": "BF_MILNE_CENSUS",
    "roots": ["src", "tests"],
    "expected_extra": ["src/lumina_plasma.c", "src/env_universe.h", "tests/bf_milne_identity_selftest.c"]
  },
  "branches": {
    "regimes": [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
    "metrics": {"count": "len(v)", "gate_fail": "sum(1 for x in v if x < 0.5)", "attributed": "sum(1 for x in v if 0.5 <= x < 1.5)", "dispersed": "sum(1 for x in v if 1.5 <= x < 2.5)", "identity_fail": "sum(1 for x in v if x >= 2.5)"},
    "rules": [
      {"name": "NIS0-X", "predicate": "count == 0 or gate_fail > 0"},
      {"name": "NIS0-C", "predicate": "count > 0 and gate_fail == 0 and identity_fail > 0"},
      {"name": "NIS0-A", "predicate": "count > 0 and gate_fail == 0 and identity_fail == 0 and dispersed == 0"},
      {"name": "NIS0-B", "predicate": "count > 0 and gate_fail == 0 and identity_fail == 0 and dispersed > 0"}
    ],
    "adversarial_fixtures": [
      {"name": "empty", "mix": []},
      {"name": "gate-fail", "mix": [[0.0, 1], [1.0, 2]]},
      {"name": "attributed", "mix": [[1.0, 3]]},
      {"name": "dispersed", "mix": [[1.0, 2], [2.0, 1]]},
      {"name": "identity-fail", "mix": [[1.0, 2], [3.0, 1]]}
    ],
    "residual": "NIS0-X"
  },
  "references": [
    {"path": "docs/BRIEF_CONT_NISTAR_2026-09-19.md"},
    {"path": "docs/READOUT_CONT_NLTE_SUPERTHERMAL_2026-09-18.md"},
    {"path": "docs/PREREG_DET_CONT_NLTE_BALANCE_2026-09-18.md"},
    {"path": "docs/PREREG_DET_DR_ON_2026-09-19.md"},
    {"path": "CLAUDE.md"},
    {"path": "src/lumina_plasma.c", "flags_existing": ["bf_get_eta", "Cinv_l", "saha_m", "chi_spont_base", "LUMINA_CMF_BF_MILNE", "LUMINA_FIX_BF_STIM_RECOMB", "ion_number_density"]},
    {"path": "src/lumina_cmfgen.c", "flags_existing": ["cmfgen_assemble_impl", "bf_get_chi", "bf_get_eta", "S_fixed"]},
    {"path": "src/env_universe.h", "flags_existing": ["LUMINA_CMF_BF_MILNE", "LUMINA_FIX_BF_STIM_RECOMB"]}
  ]
}
```

## 9. 집행 기록 (CLOSED-OFFLINE)

1. **게이트 4종**: prereg-preflight rc=0(files=3·refs=8)·gate_referents rc=0·doc_canonical rc=0. 커밋 17d3cd1.
2. **경량 사전감사(Opus 4.8 fresh, `claude-opus-4-8`)**: Q_GOAL 부분부합·**Q_LEAN 비대·통과 불가·BLOCKING 1**. §0 항등 코드 재현 확증(census 선회 정당). **L4 «폐합 차단»: census 런이 오프라인 정적 조회 대비 과provision** — U_next(T_e)·엣지는 정적 원자데이터+기존 T_e 로 런 없이 귀속 가능. 발주 정지·오프라인 우선 권고.
3. **운전자 오프라인 귀속(런 0회, 정본 `docs/READOUT_NISTAR0_OFFLINE_2026-09-19.md`)**: (i) n_l 붕괴 floor 배제 (ii) 텍스트북 U(T_e) I·II·III 정상 (iii) **Si V(χ 166.8 eV) 깊은 엣지 Saha e^{χ/kT}=9.5e94 ⟹ n_l*/n_next=2.42e79 = census 셸20 max 1.60e79 일치**. ⟹ 1e79 = 과이온화가 밀어올린 고이온의 정확한 Saha 물리(버그 아님), 근인 = decoupled solve(완전 선형화 부재).
   ⟹ **DET-CONT-NISTAR-0 종결: 구현·census 런 없이 오프라인 귀속.** 게이트 G-ID/G-RUN/G-ALG/G-ATTR·selftest·판정런 전부 미집행(불요). 상태 CLOSED-OFFLINE. 다음 표적(완전 선형화 대 국소 처방 대 pivot)은 user 결정.
