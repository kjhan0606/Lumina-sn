# 사전등록 — DET-DJDN-MVP: 라이브 이온화 balance 에 대각 ∂J/∂N 선형화 항을 넣으면 CONT 초열 폭주가 유한 고정점에 착지하는가 (2026-09-19)

기안 = **Fable fresh**(`claude-fable-5-1`, 개정25-b). 상태 = **기안 / 경량 사전감사 대기(개정25-d, Q-GOAL·Q-LEAN 전용)**. 실행 승인이 아니다.
근거 = `docs/BRIEF_DJDN_MVP_2026-09-19.md`(운전자 브리프, f349597) · `docs/DESIGN_PATHB_LINEARIZATION_2026-09-19.md`(설계 정찰) · `docs/READOUT_DAMPING_DISCRIMINANT_2026-09-19.md`(A 불가·B 필요) · `docs/READOUT_CONT_NLTE_SUPERTHERMAL_2026-09-18.md`(폭주 실측) · 직전 계약 `docs/PREREG_DET_DR_ON_2026-09-19.md`(형식 계승). base = **DR-on 판정런 385315 폐합본**(`/gpfs/kjhan/lumina/cmfgen_map/dr_on_20260919/run_h100a100`, 기안 시점 진행 중 — **폐합 전 이 계약은 발주 불가**). 소스 기준 = 동결 트리 `/gpfs/kjhan/lumina/cmfgen_map/lteunmap_eps_20260917/source`(행 번호 기준; CONT 변경집합 포함 상태·repo 미이식).

## 0. 기안자 대수·배선 검사 — 브리프 대비 정정 2건 (운전자·사전감사 필독)

**(a) 위치 — 브리프의 구현 자리는 이 구성에서 죽은 경로다.** 브리프는 `radeq_simul_all:12223`·`simul_ladder:11646/11706`·`simul_r1:11767`·`SIMUL_ION_DAMP:13362` 를 지목했다. 동결 트리에서 `radeq_simul_all` 의 **호출은 0건**(정의 1·주석 2뿐) [저자 실행: `grep -nE "radeq_simul_all\s*\(" src/*.c src/*.cu`]; `simul_ladder`·`simul_r1` 호출은 전부 그 안(`:11769`·`:13073`·`:13213–13351`); 게이트 `LUMINA_RADEQ_SIMUL`(`g_simul_on:11566`)은 base 376251·CONT run1·DR-on 385315 exports 에 **부재** [저자 실행 grep]. **라이브 이온화 경로** = `compute_electron_density:3120`(n_e 고정점, 최대 100회, `:3172` 0.5 혼합) → `compute_ion_populations_shell:2808`(원소별 비 사다리, `:3053` 이후 `atom->ion_number_density` 제자리 덮어쓰기) → `parity_rate_se_ratio_checked:2720`(r = num/den) → `parity_gamma_phot_checked:2605` → `nlte_bf_gamma_canonical:472`(커밋된 `radfield_view.J_nu`). 증거 = CONT run1 채널 덤프(gen 71)가 `parity_rate_se_ratio_checked` 안(`:2773–2803`)에서 쓰였다. Γ 동결 기전은 여기서도 그대로다(J 는 세대당 1회 커밋, 인구 되먹임 없음). ⟹ **변경집합은 라이브 경로에 둔다**(§3). 인구 커밋의 라이브 자리 = `nlte_population_candidate_commit_bundle:22211`(`:22236` memcpy).

**(b) 부호 — 「이온화↑ ⟹ G↓(복원)」은 대각 S-채널 항의 성질이 아니다.** CONT 소스(`lumina_cmfgen.c:2373` 게이트·`:2436` `bf_get_eta`)에서 bf-불투명 빈의 S_bf,b = η_bf,b/χ_t,b 는 Milne 로 x/x_Saha 에 비례한다(x = N_{i+1}/N_i): η ∝ N_{i+1}, χ ∝ N_i ⟹ ∂S_b/∂x > 0. λ*_{s,b} ∈ [0,1](`cmfgen_solve_J:2972`, 기록 `:3079`·`:3116`) · ∂Γ/∂J_b = 4πσ_b/(hν_b) > 0 ⟹ **a ≡ dΓ/dx > 0 구성상**. 대각 항은 Γ 를 **올린다** — 이것이 곧 판독이 말한 양성 되먹임이다. 「복원」은 항의 부호가 아니라 **뉴턴이 근을 찾는 것**이다: 쌍별 선형 모델 x·n_eα = Γ_lag + a(x − x_lag) 의 근은 x* = (x_map − g·x_lag)/(1 − g), g = a/(n_eα), x_map = Γ_lag/(n_eα)(lagged 맵값). g < 1 이면 x* 유한(x_map 이 x_lag 위면 x* 는 x_map 보다 **더 위** — 가속); g ≥ 1 이면 선형 모델에 양의 근이 없다 — 이 경우는 **보고**하지 클램프하지 않는다. 「이온화↑ ⟹ J↓」복원 채널은 ∂J/∂χ(Λ*(χ) 의존)이며 저장된 λ* 로는 얻을 수 없다 — 이 MVP 밖(§8). ⟹ G-CPL 은 「G_eff < G_lag」를 걸지 않고 **항등 + a > 0** 을 건다(§4). 이 정정은 계약 질문(§1)을 바꾸지 않는다 — 대각 뉴턴이 바깥 반복의 궤적을 잡는가는 여전히 런이 정한다.

**(c) 첫 소비자.** λ* 는 매 바깥 반복 `lumina_cuda.cu:8563`(`radeq_set_line_re_source`) 로 plasma 에 전달돼 `g_lre_lambda_star:13690` 에 저장되지만 동결 트리에 **소비자가 0** 이다 [저자 실행 grep `g_lre_lambda_star\[`]. 이 MVP 가 첫 소비자다 — 빈 격자 정합(`cs.n_bins` = `nlte->n_freq_bins`, `lumina_cuda.cu` J 커밋이 같은 인덱스)은 selftest 항목(§3).

## 1. 계약 — 물리 계약 1개

**「라이브 이온화 balance(`parity_rate_se_ratio_checked`)에 대각 ∂Γ/∂N 선형화 항(∂J/∂S = 저장 λ*, ∂S/∂N = CONT 소스 η_bf/χ_t 의 Milne 인구 의존)을 넣어 쌍별 뉴턴 근을 취하면, CONT 소스가 유발하는 중간 셸 12–24 황 II→III 비 r 의 반복별 단조 상승이 유한 고정점에 착지하는가(대각 충분, MVP-A), 여전히 상승하는가(셸간 결합 필요, MVP-B)」.** 확인 런 1회(twin-T_e, binned 챔피언 경로, CONT 소스 ON, DR-on base 위). 대조 = CONT run1(DJDN off·DR off·2 반복 커밋)과 DR-on 385315(CONT off). 차이가 둘(CONT 소스·DJDN)이므로 **구동의 실재**는 런 안의 census(x_map/x_lag, G-DRV)로 확인한다.
⚠**말하지 않는 것**: CMFGEN 도달 · 완전 Path B(tri_*·셸간·`coupled_newton_solve_all:16863`·rate 행렬 bf 결합·이중 권위) · ∂J/∂χ 채널 · 정확 multi-GPU 솔버 Λ* · Milne 소스 자체의 옳음(인구 역전 빈) · Si/Ca 판정(기재만, §5) · 스펙트럼 · 자유-T_e.

### Q-GOAL (기안자 자기 답 — 경량 사전감사가 독립 재판정)
① 칸: 어긋남 지도 **중간 셸 9–24 S II/III**(CMFGEN f_II .85/.80/.71). 직접 줄이지 않는다 — 이 단은 그 칸을 CONT 소스로 채우려던 시도가 **왜 폭주했는가**의 첫 수리 가설(∂J/∂N 결손)을 잰다. CONT 소스는 어긋남 지도의 램프·연속체 채움 계보(09-17 판독 A ①)의 본선이고, 폭주가 잡히지 않으면 그 본선이 닫힌다. ② 산출 = 물리 측정(r 궤적·국소 이득 g·λ*-가중 σ 적분). 계측 = census 파일 1(§3)·게이트 4·selftest 1. ③ 대안: 완전 Path B 로 직행(3–4 파일·야코비안 조립 커널·아키텍처) — §0(b) 대수는 대각만으론 부족할 가능성을 가리키므로 이 대안이 실재한다. 그러나 완전 Path B 의 크기를 정하는 수(국소 이득 g·λ* 크기·비국소 몫)를 아무 런도 준 적이 없다 — 이 MVP 가 그 수를 준다. ④ 건너뛰면 Path B 를 크기 미지로 짓거나, 대각으로 충분한 경우를 놓친다.

### Q-LEAN (기안자 자기 답)
L1 물리량 = 셸별 S II/III 비 r 의 반복 궤적(`physics_DET_iter000k.ion.csv`, 기존)·국소 이득 g 와 a(census `lumina_pop_djdn.csv`, 신규 1 파일 13 열 — G-CPL·G-DRV·E-1 이 전 열을 소비한다; 소비되지 않는 열 0). L2 게이트 4: G-ID(다른 키·소스 혼입) · G-RUN(미완주) · G-CPL(항이 실제로 적용됐는가 — 그렇지 않으면 MVP-A 가 공허) · G-DRV(구동이 실재하는가 — 그렇지 않으면 MVP-A 가 공허). 중복 없음. L3 = 2 걸음(구현·selftest → 확인 런 1). 더 적은 걸음: 오프라인 프로브(설계 §4 ①)는 λ*·S_fixed·J 의 셸×빈 덤프(`LUMINA_CMFGEN_JDUMP`)가 있어야 하는데 CONT run1·DR-on 어느 런도 켜지 않았다 [저자 실행: run1 `work/` 에 `lumina_cmfgen_jnu.csv` 없음] ⟹ 런 0회 경로 없음. L4 삭제 후보: MAXCH 는 판정런 기본 off(§6 ②) — 켜면 「유계 궤적」이 스텝 상한 탓인지 항 탓인지 섞인다(감쇠 판별 정리). Si/Ca 는 판정 조건에서 제외(데이터 근거 §2).

## 2. 기전 — 런 0회 확정분

- **항의 대수(쌍 i→i+1, 셸 s, x = N_{i+1}/N_i, N_tot = N_i + N_{i+1} 고정)**: Γ_i = Σ_l f_l Σ_b 4πσ_{l,b} J_b/(hν_b) Δν_b(`parity_gamma_phot_checked:2605` 의 그 빈·그 σ_row·그 f_l). 대각 ALI: δJ_b = λ*_{s,b} δS_b. CONT 소스 S_b = (η_bf,b + χ_ff B + η_ln)/χ_t,b 에서 이 쌍의 몫: η_{i,b} = Σ_l n*_l σ_{l,b}(2hν³/c²)e^{−hν/kT}, n*_l = N_{i+1} n_e Φ_l(T_e) ⟹ ∂η/∂N_{i+1} = η_{i,b}/N_{i+1}; χ_{i,b} = Σ_l N_i f_l σ_{l,b}(1 − 유도항) ⟹ ∂χ/∂N_i = χ_{i,b}/N_i. ∂N_{i+1}/∂x = N_tot/(1+x)² = −∂N_i/∂x. ⟹ **∂S_b/∂x = [η_{i,b}/N_{i+1} + S_b·χ_{i,b}/N_i]·N_tot/((1+x)² χ_t,b) > 0**, **a = Σ_l f_l Σ_b 4πσ_{l,b}/(hν_b)·λ*_{s,b}·∂S_b/∂x·Δν_b**. 피연산자 전부 기존 배열(σ_row·λ*·B(T_e)·Φ_l·χ_t 는 `cs.chi_tot` 전달분 `:13688`) — **새 커널 0**, 비용 = Γ 적분 1회분/쌍.
- **뉴턴**: 기존 num/den(충돌항 포함) 위에 Γ_lag → Γ_lag + a(x − x_lag). 스칼라 근 x*(§0(b)); ΔN = 0 이면 Γ_eff = Γ_lag ⟹ off 와 같은 r(고정점 불변). x_lag 는 **세대 커밋값**(`atom->ion_number_density` 를 `:3053` 덮어쓰기 **전**에 읽어야 하며, n_e 루프 `:3120` 안의 직전 반복값이 아니다) — 구현 제약. g ≥ 1(양의 근 없음): 그 쌍은 off 값 x_map 을 쓰고 census `root=0` 으로 **표기**(무음 폴백·클램프 금지).
- **구동 재현 예측(오프라인)**: CONT run1 sh20 γ_phot/den = 231(DR 0) [판독 §2]; DR-on 은 분모를 ≤ 2.5× 올린다 [DR-on 계약 §2] ⟹ ≥ ~90× — CONT 소스 위에서 x_map/x_lag ≫ 1 이 유지될 것으로 예측. G-DRV 가 검사한다(예측이 틀리면 MVP-A 가 아니라 **미시연**이다).
- **잣대를 base 에 먼저(저자 실행, `physics_DET_iter0000/0001.ion.csv`)**: r = q2/q1. CONT run1 S: sh12 42.4→56.6(+0.13 dex)·sh16 60.0→84.4(+0.15)·sh20 128→231(+0.26)·sh24 414→842(+0.31) — 4 셸 전부 상승. base run5(LTE 소스) S: sh12 44.8→57.0·sh16 63.1→17.7·sh20 126→8.79·sh24 405→76.3 — 하강. **Si 는 두 런 다 비단조**(CONT sh16 1.04e4→8.03e3, sh20 1.52e4→6.96e3; base 도 하강) — CONT 폭주 잣대가 아니다. **Ca 는 r 10⁶–10⁷**(f_II ≤ 4e-7) — 비 잣대 무의미. ⟹ 판정량 = S 4 셸; Si·Ca 기재.
- **비용**: CONT run1 은 2 반복 커밋에 4h59m(h200×1) ⟹ ≈2.5 h/반복.
- **미확정(계약 밖)**: 스칼라 선형화와 N×N 준위 해(이중 권위, 설계 §2)의 불일치 크기; ∂J/∂χ 채널의 크기; 셸간 항의 크기.

## 3. 수리안 — 변경집합

env `LUMINA_POP_DJDN`(기본 0 = 항 미적용·census 파일 없음·**byte-동일**). ON: §2 의 a 와 쌍별 뉴턴 근을 `parity_rate_se_ratio_checked` 에서 취한다(라이브 경로, §0(a)); census `lumina_pop_djdn.csv` 를 작업 디렉토리에 append — 열 `gen,shell,Z,stage,x_lag,x_map,a,g,x_star,root,gamma_lag,gamma_eff,maxch_factor`, 행 = (Z 16·14·20, stage 1)×셸 0–49 = 세대당 150 행, 채널 덤프와 같은 arming(`g_ion_ch_armed`, 커밋 스윕만). env `LUMINA_POP_MAXCH`(기본 미설정 = 미적용·byte-동일): 세대 간 인구 스텝 축척 ‖ΔN/N‖_∞ ≤ MAXCH(CMFGEN `solve_for_pops.f:132` 계열), 적용 자리는 라이브 커밋(`nlte_population_candidate_commit_bundle:22211`) 또는 후보 조립 — 구현 결정; 고정점 불변·**상태 클램프 아님**(`:13378` 식 상한 모방 금지)·적용 인자를 census `maxch_factor` 에 기록. 두 노브 모두 SIMUL 경로(`:11646`·`:13362`)는 건드리지 않는다(죽은 경로).

### 변경집합 — POP-DJDN (표)

| 경로 | 담당 | 내용 |
|---|---|---|
| `src/env_universe.h` | Opus 4.8 | `LUMINA_POP_DJDN`·`LUMINA_POP_MAXCH` 등록 |
| `src/lumina_plasma.c` | Opus 4.8 | `parity_rate_se_ratio_checked`: a(§2 대수, 저장 λ*·σ_row·Φ_l·χ_t 만)·쌍별 뉴턴 근·root 표기·census 행; `compute_ion_populations_shell`: x_lag = 세대 커밋값 확보(덮어쓰기 전); 커밋/후보 조립: MAXCH 스텝 축척(env 미설정 시 미적용); OFF 두 노브 byte-동일 |
| `tests/pop_djdn_selftest.c` | Opus 4.8 | CPU selftest(§4 G-CPL 오프라인 부): OFF r bit-동일(게이트 미진입 ⟹ 원 r 그대로) · ON·ΔN=0(x_lag = x_map)·a>0 ⟹ x* = x_map, rel ≤ 1e-12(고정점 불변 — **bit-동일 아님**: x* = (num − a·x_lag)/(den − a) 는 num/den 과 다른 FP 연산열이라 반올림 차가 남는다; 픽스처 g ≤ 0.9 — (den − a) 상쇄가 반올림을 1/(1−g) 배 키우므로) · ON·a≡0(λ* 미전달) ⟹ x* = num/den 이 FP 정확 ⟹ r 이 OFF 와 bit-동일(음성대조 1 이 이 assert 를 담당, 이중 기재 아님) · a = 독립 재계산(1e-12 rel) · 빈 격자 정합(λ* 인덱스 = J 인덱스) · **음성대조 1**: λ* 전 빈 0 ⟹ ON == OFF bit-동일(항이 λ* 밖 무언가를 읽으면 FAIL) · **음성대조 2**: a 를 1e-6 교란 ⟹ census 항등(G-CPL) FAIL 시연 · g ≥ 1 픽스처 ⟹ `root=0` 표기·r = OFF 값(클램프·무음 폴백 검출) · MAXCH: 축척 인자 = min(1, MAXCH/‖ΔN/N‖_∞)·미설정 시 인자 1 bit-동일 |

### 변경집합 끝

★**src 편집 1개 규칙**: DR-on 385315 폐합·커밋 후 발주(DR-on 은 코드 0줄이라 동결 트리는 CONT 변경집합 상태 그대로). ⚠프리플라이트: 표의 3 파일은 구현 전 repo 에 심볼이 없으므로 `expected_extra` 로 선언(직전 CONT 계약과 동일 처리).

## 4. 게이트 명부 (표)

| ID | 조건 | 음성대조 |
|---|---|---|
| G-ID | 새 런 exports 를 385315 계보(`/gpfs/kjhan/lumina/cmfgen_map/dr_on_20260919/run_h100a100/input/resolved_lumina.exports`)와 diff — 추가·변경 키가 정확히 `LUMINA_CMF_CONT_NLTE_SOURCE`(=1)·`LUMINA_POP_DJDN`(=1)·`LUMINA_PURE_CMFGEN_ITER`(2→§6 ①) [§6 ② 선택 시 `LUMINA_POP_MAXCH` 추가; 프로필 종속 `LUMINA_CMF_FINE_MGPU_DEVICES` 허용]; `LUMINA_ION_CHANNEL_DUMP`(=1)·DR 5 키(=1) 유지; `LUMINA_ION_NT`·`LUMINA_NT_FULL`·`LUMINA_RADEQ_SIMUL`·`LUMINA_SIMUL_ION_DAMP` **부재**; 동결 트리 `…/lteunmap_eps_20260917/source.sha256` 의 `sha256sum -c` non-OK 가 정확히 §3 표의 파일(신설 `tests/pop_djdn_selftest.c` 는 manifest 밖이라 별도 해시 기재); 새 바이너리 해시 `input/binary.sha256` 기재; 러너 편집 = job-name·`--time`·ITER 검사·`outer_iterations` 뿐(해시 STAGE_IDENTITY 기재) | NT/SIMUL 키 혼입·§3 밖 소스 파일 변조 ⟹ FAIL |
| G-RUN | `RUN_FOOTER.txt` `terminal_state=SUCCESS`·`model_rc=0`·`outer_iterations`=선택 ITER; `work/physics_comparison/physics_DET_iter<ITER−1>.ion.csv`·`work/lumina_ion_channels.csv`·`work/lumina_pop_djdn.csv` 실재 | 미완주·census 결손 ⟹ FAIL |
| G-CPL | (오프라인) `tests/pop_djdn_selftest.c` rc=0, §3 음성대조 둘 다 FAIL 시연 — **봉인 접촉 전 Fable 코드 감사 선행**(개정16 ②); (in-vivo) `work/lumina_pop_djdn.csv` 최대 `gen` 150 행·13 열 헤더 일치, 전 행 항등 \|gamma_eff − gamma_lag − a·(x_star − x_lag)\| ≤ 1e-9·max(둘)(root=1 행), Z=16 셸 12·16·20·24 행 a > 0(구성상 참 — 배선 음성대조: 음이면 결함) | census 열 바꿔치기 ⟹ 항등 FAIL; λ* 미전달 빌드 ⟹ a ≡ 0 ⟹ FAIL |
| G-DRV | census 에서 gen ≤ 3 인 어느 세대에 Z=16 셸 20·24 의 x_map/x_lag ≥ 1.3(CONT 구동 실재; CONT run1 iter0→1 실측 ×1.8·×2.0) | DR-on 385315(CONT off) 계보 궤적은 하강 ⟹ 조건 불성립 ⟹ 미시연으로 분류 |

### 게이트 명부 끝

## 5. 기대치 — 런에서만 얻는 수

| ID | 양 | 대조 | **기대·역할** |
|---|---|---|---|
| E-1 국소 이득 | g = a/(n_eα), Z=16 셸 12·16·20·24, 첫 DJDN 세대와 최대 세대(census) | 없음(첫 측정) | 판정 조건 아님 — 분기 명명·Path B 크기 산정에 기재. g ≥ 1 셸 수(root=0) 기재 |
| E-2 S 궤적 | Δ_k = log10 r_k − log10 r_{k−1}, r = n_III/n_II, Z=16 셸 12·16·20·24(`physics_DET_iter000k.ion.csv`, k = 1..ITER−1) | CONT run1 Δ_1 = +0.13/+0.15/+0.26/+0.31 | **판정량**(분기 표) |
| (기재) | Si 셸 12·20 r 궤적 · Ca f_II · 외곽 S f_III 셸 28·32·36 · Fe f_IV 셸 0–3 · MAXCH 인자(켠 경우) | base 376251·DR-on 385315 | 판정 조건 아님, 판정문 기재 |

문턱 근거: 수렴 창 0.05 dex = CONT 1 반복 이동(≥ 0.13)의 절반 이하이고 base LTE 소스의 반복 이동(sh16 −0.55)과도 갈린다; MVP-B 하한 0.10 = CONT 최소 셸(sh12 +0.13) 아래·수렴 창 위. 기안자 예상(§0(b) 선형 모델, 판정 조건 아님·확률 미부여): 국소 대각만으로는 MVP-B 또는 MVP-C(근 부재)가 대수적으로 유리하고, MVP-A 는 g < 1 이면서 비국소 구동이 없을 때만 열린다 — 그 둘 다 런이 정한다.

### 분기 (표)

| 분기 | 조건 | 의미 |
|---|---|---|
| MVP-X | G-ID/G-RUN/G-CPL/G-DRV 중 하나라도 미충족 | 미시연 |
| MVP-A | 게이트 전건 **and** 4 셸 전부 마지막 2 반복 \|Δ_k\| ≤ 0.05 **and** 그 2 세대에 그 셸 root=1 **and** 그 반복에 maxch_factor = 1 | **대각 커플링이 유한 고정점에 착지** — 대각 충분(관측 등급; 수렴값의 CMFGEN 근접은 기재) |
| MVP-B | 게이트 전건 **and** 4 셸 전부 모든 k 에 Δ_k > 0 **and** 마지막 Δ ≥ 0.10 | **단조 상승 지속** — 대각 불충분, 셸간 결합(완전 Path B) 필요 |
| MVP-C | 게이트 전건 **and** A 도 B 도 아님(진동·하강·일부 셸만 상승·root=0 세대·MAXCH 포화 반복 포함) | 판정자가 census(g·root)와 궤적으로 명명 |

### 분기 끝

재분기 금지. 판정기 = `/gpfs/kjhan/lumina/cmfgen_map/repro_claude_20260917/judge_contnlte.py` 의 궤적 판독부 재사용 + census 항등·E-1·E-2(운전자 도구, 해시 `READOUT_TOOLS.sha256` 기재; 소스 아님). 음성대조: CONT run1 자기검사 ⟹ census 없음 ⟹ G-CPL FAIL ⟹ MVP-X; DR-on 385315 ⟹ G-DRV 불성립 ⟹ MVP-X.

## 6. 자원 · 실행 · **결정 항목(운전자·user)**

새 RUN 루트 `/gpfs/kjhan/lumina/cmfgen_map/djdn_mvp_20260919/run1`(형제 루트), 입력 = 385315 입력 복사 + 새 빌드 + exports 편집(G-ID). GPU 1 런. 봉인 base 읽기만.

① **ITER** — 4 또는 6(브리프 4–6). 4 ≈ 10 h(`--time=14:00:00`) / 6 ≈ 15 h(`--time=20:00:00`; CONT run1 실측 2.5 h/반복). 수렴 창이 「마지막 2 반복」이므로 ITER 4 는 판정 세대 2 개(k=2,3), 6 은 4 개. 기안자는 못 박지 않는다.
② **MAXCH** — 판정런 기본 **off**(Q-LEAN L4; 감쇠 판별 정리상 상한은 늦출 뿐이라 유계 궤적을 항에 귀속할 수 없게 만든다). 켜면 값(0.25–0.5)과 함께 기재하고, 포화 반복(인자 < 1)은 MVP-A 창에서 제외한다.
③ **뉴턴 자리** — 쌍별 스칼라 근(권장: 라이브 경로에 고정점 루프가 없고 사다리가 비의 곱이라 폐형 근이 있다) 대 원소 사다리 전체 뉴턴. 구현자 결정; 제약 = x_lag 세대 커밋값·a 는 저장 λ*·σ_row 만·off byte-동일.
④ 헤지 프로필 — 먼저 시작하는 쪽만(직전 계약 §6 ② 동일).
⑤ NT 키 — DR-on base 그대로 부재(CONT run1 은 NT on 이었으나 γ_phot 몫 98.7–99.3% 라 구동에 무관 [판독 §2]).

순서: 경량 사전감사(Opus 4.8 fresh, Q-GOAL·Q-LEAN) → 게이트 4종(prereg-preflight `files=3`·gate_referents·doc_canonical·dispatch-preflight) → **DR-on 폐합 확인** → Opus 4.8 구현(동결 트리 복사본) → selftest → Fable 코드 감사(첫 항목 = 봉인 접촉 가부) → 빌드 → 스테이징 → 제출 → 판독 → 판정(운전자) → 종료감사(Opus 4.8 fresh; 물리 주장·GPU 지출이 걸린 단이라 생략하지 않음).

## 7. 분장 장부 (개정25 · 규약 / 실제 / 위반)

| 단계 | 규약 | 실제 | 위반 |
|---|---|---|---|
| 기안 | Fable fresh(25-b) | **Fable fresh**(`claude-fable-5-1`, 이 문서) | — |
| 경량 사전감사(Q-GOAL·Q-LEAN) | Opus 4.8 fresh(25-d) | 미집행 | |
| 구현 + selftest | Opus 4.8 | 미집행 | |
| 코드 간단 감사 | Fable fresh(25-a) | 미집행 | ⚠기안자 = 감사자(자기 계약 편향, 개정25-b 기재) |
| 스테이징·제출·판독·판정 | 운전자 | 미집행 | |
| 종료감사 | Opus 4.8 fresh | 미집행 | |

## 8. 주장 상한

말할 수 있는 것: 「라이브 경로의 대각(S-채널) ∂J/∂N 선형화가 CONT 폭주를 유한 고정점에 착지시키는가」와 그 국소 이득 g·a 의 in-vivo 크기(Path B 크기 산정 입력). 착지해도 그 값이 CMFGEN 근처인지는 기재 사항이지 성공 조건이 아니다. 말하지 않는 것: ∂J/∂χ 채널·셸간 항·N×N 준위 해와의 정합·정확 솔버 Λ*·Milne 소스의 인구 역전 빈 처리·Si/Ca·스펙트럼·자유-T_e. 「DJDN 토글 단독」 주장은 off 런이 없으므로 selftest byte-동일 + Fable 코드 감사의 **오프라인 주장** 조건부(직전 계약과 같은 계급). twin-T_e 핀 조건부.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 — POP-DJDN (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "(?:src|tests)/[A-Za-z0-9_/.]+\\.(?:c|h)",
    "symbol": "LUMINA_POP_DJDN",
    "roots": ["src", "tests"],
    "expected_extra": ["src/env_universe.h", "src/lumina_plasma.c", "tests/pop_djdn_selftest.c"]
  },
  "branches": {
    "regimes": [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
    "metrics": {"count": "len(v)", "gate_fail": "sum(1 for x in v if x < 0.5)", "landed": "sum(1 for x in v if 0.5 <= x < 1.5)", "rising": "sum(1 for x in v if 1.5 <= x < 2.5)", "other": "sum(1 for x in v if x >= 2.5)"},
    "rules": [
      {"name": "MVP-X", "predicate": "count == 0 or gate_fail > 0"},
      {"name": "MVP-A", "predicate": "count > 0 and gate_fail == 0 and rising == 0 and other == 0"},
      {"name": "MVP-B", "predicate": "count > 0 and gate_fail == 0 and rising > 0 and other == 0"},
      {"name": "MVP-C", "predicate": "count > 0 and gate_fail == 0 and other > 0"}
    ],
    "adversarial_fixtures": [
      {"name": "empty", "mix": []},
      {"name": "gate-fail", "mix": [[0.0, 1], [1.0, 2]]},
      {"name": "landed", "mix": [[1.0, 4]]},
      {"name": "rising", "mix": [[2.0, 4]]},
      {"name": "mixed-rise", "mix": [[1.0, 2], [2.0, 2]]},
      {"name": "other", "mix": [[1.0, 3], [3.0, 1]]}
    ],
    "residual": "MVP-X"
  },
  "references": [
    {"path": "docs/BRIEF_DJDN_MVP_2026-09-19.md"},
    {"path": "docs/DESIGN_PATHB_LINEARIZATION_2026-09-19.md"},
    {"path": "docs/READOUT_DAMPING_DISCRIMINANT_2026-09-19.md"},
    {"path": "docs/READOUT_CONT_NLTE_SUPERTHERMAL_2026-09-18.md"},
    {"path": "docs/PREREG_DET_DR_ON_2026-09-19.md"},
    {"path": "CLAUDE.md"},
    {"path": "src/lumina_plasma.c", "flags_existing": ["parity_rate_se_ratio_checked", "compute_ion_populations_shell", "parity_gamma_phot_checked", "nlte_bf_gamma_canonical", "radeq_set_line_re_source", "g_lre_lambda_star", "nlte_population_candidate_commit_bundle", "frozenin_alpha_rr"], "flags_planned": ["LUMINA_POP_DJDN", "LUMINA_POP_MAXCH"]},
    {"path": "src/env_universe.h", "flags_existing": ["LUMINA_SIMUL_ION_DAMP", "LUMINA_CMFGEN_JDUMP"], "flags_planned": ["LUMINA_POP_DJDN", "LUMINA_POP_MAXCH"]},
    {"path": "src/lumina_cmfgen.c", "flags_existing": ["cmfgen_solve_J", "lambda_star", "LUMINA_CMFGEN_JDUMP"]}
  ]
}
```

## 9. 집행 기록

1. **게이트 4종(발주 전)**: prereg-preflight rc=0(`changeset-consistent files=3`·branch-partition trials=200000 rules=4 fixtures=6 residual=MVP-X·references-resolved refs=9)·gate_referents rc=0(changeset_exempt=3)·doc_canonical rc=0(계약 + 인용 문서 5). 커밋 38e7601. (dispatch-preflight 는 구현 발주서 작성 시.)
2. **경량 사전감사(Opus 4.8 fresh, `claude-opus-4-8`, 개정25-d, Q-GOAL·Q-LEAN 전용)**: **PREAUDIT 통과 · BLOCKING 0.** 감사자가 저장소 grep 으로 §0 정정을 독립 재도출 — `radeq_simul_all` 호출 0(§0a 확증)·`g_lre_lambda_star` 소비자 0(§0c 확증·G-CPL 의 「a≡0 무음 no-op」이 실재 실패 모드임을 뒷받침)·`parity_rate_se_ratio_checked` 라이브 호출 실재. **Q-GOAL 부합**(진단·크기측정 단 — 직접 칸 축소 아님, 판정문 명시 요구)·**Q-LEAN 통과**(census 13열 전건 소비·군살 0·게이트 4 무중복·2 걸음). 기재 3: (a) Q-GOAL 진단 성격 명시 (b) L3 JDUMP-on 짧은 덤프로 오프라인 단일셀 프로브 가능하나 라이브 다중셸 궤적은 못 재현이라 라이브 런 정당 (c) L4 `LUMINA_POP_MAXCH` 축척 구현 «기재로 족함»(판정런 off·미소비) — 구현 범위 결정을 운전자에게 위임(발주 정지 아님).
3. **DR-on 폐합 확인 · §6 결정(운전자)**: DR-on 385315 폐합 확인(커밋 4442dfd, 종료감사 통과) ⟹ src-편집 가동 태스크 0, 구현 발주 가능. **①ITER=4**(판정 세대 k=2,3 2개·~10h·`--time=14:00:00`; 진단 단이라 MVP-A/B 판별에 충분). **②MAXCH 판정런 off**(L4 + 감쇠 판별 정리; 축척 구현은 off-기본 **유지** = user 승인 b737446 「감쇠는 B 뉴턴-스텝 보정 상한으로 흡수」= MAX_CHNG 강건층, 판정런 inert ⟹ 사전감사 L4 «기재로 족함» 옵션② 채택). **③뉴턴 자리 = 쌍별 스칼라 폐형 근 권장**(라이브 경로 고정점 루프 없음), 구현자 결정, 제약 = x_lag 세대 커밋값·a 는 저장 λ*·σ_row 만·off byte-동일. **④헤지 yes**(먼저 시작하는 쪽). **⑤NT 키 부재**(DR-on base 그대로). ①④는 스테이징 시 확정.
4. **구현 1회차(Opus 4.8, `claude-opus-4-8`) · Fable 코드 감사 = 폐합 차단 2 → 재발주**: 구현자가 동결 트리 3파일(env_universe.h·lumina_plasma.c·tests/pop_djdn_selftest.c, 매니페스트 대조로 잡편집 0 확인)·selftest 16/16 PASS + 결함 주입 FAIL 시연. **Fable 코드 간단 감사(`claude-fable-5-1`, 개정25-a): CODE_AUDIT 폐합 차단·SEAL_CONTACT 가·BLOCKING 2.** ① **x_lag 참조점 결함** — 구현이 `atom->ion_number_density` 제자리 읽기, 그러나 `compute_electron_density` 가 셸당 ≤100회 하위반복하며 덮어씀 ⟹ census 가 직전 하위반복값을 x_lag 로 읽어 **항 자기 소거**(x*→x_map=OFF; δ_{k+1}=−[g'/(1−g')]δ_k) ⟹ x_map/x_lag≈1·G-DRV 불성립·MVP-X 10h 소진 또는 거짓 MVP-B. 수리 = x_lag 참조점을 J·S_fixed·λ\* 조립 시점 공개 커밋 인구(`radeq_set_line_re_source`)로, 전 하위반복·armed 스윕 공통(기존 `ion_before` :21551 스냅샷 활용; 단 후보 세대 커밋 cadence 확인). ② **selftest §3 축소** — `a` 독립 재계산(1e-12 rel) 부재 ⟹ G-CPL 오프라인부가 a 대수 무검증(픽스처 손계산 1개). 대각 항 대수·부호·야코비안·뉴턴 근·Φ_l 레시피는 **통과**(실제 빌더 대조)·클램프/floor 없음. «기재» 8건 중 §3 문면 「ON ΔN=0 ⟹ r bit-동일」 과강(기안자 오류) → 계약 개정 대상. ⟹ **운전자 빌드 보류 판단 정당**(감사 전 봉인 no); 구현 재발주.
5. **계약 §3 개정(Fable OLD/NEW, d5c5d23)**: 「ON ΔN=0 ⟹ r bit-동일」 과강 정정(OFF bit-동일 / a≡0 bit-동일 / ΔN=0·a>0 rel≤1e-12). 행 diff 1줄·게이트 재실행 rc=0.
6. **구현 2회차(Opus 4.8, `claude-opus-4-8`)**: 두 «폐합 차단» 수리 — ① x_lag = `ion_before` 스냅샷(정적 포인터, prepare_ionization set·compute_ion_populations 후 clear, 전 하위반복+armed 공통; cadence=pass당 1회 확증) ② selftest a 독립 재계산 1e-12 + 개정 §3 불변식 3종 + x_lag 회귀. selftest 21/21 PASS + 주입 FAIL 2종 시연. OFF byte-동일 유지·레거시 root=0 가시화. 매니페스트 대조 잡편집 0(3파일).
7. **Fable 코드 재감사(`claude-fable-5-1`, 개정25-a): CODE_AUDIT 통과·SEAL_CONTACT 가·BLOCKING 0.** cadence 주장 소스 독립 검증(begin fresh 복사·closure 재시작·공개 인구 commit 전 불변·R7 pass당 커밋 1회)·자기 소거 소멸·OMP 무경합·a 재계산 유효·새 결함 0. 기재 5: ①a 손계산=코드경로 독립(대수유도 아님) ②주입 FAIL 로그 미영속(정적 추론으로 검출력 확인) ③**census arming=`LUMINA_ION_CHANNEL_DUMP=1` ⟹ 발주 env 에 `LUMINA_POP_DJDN=1`+`LUMINA_ION_CHANNEL_DUMP=1` 병기 필수** ④maxch_factor 한 세대 지연(판정런 off 상수 1) ⑤빈 격자 정합 selftest 부재(1회차 영역). ⟹ 빌드·스테이징 진행 가능.
7-b. **grok 독립 교차감사(`grok-4.6`, user 09-20 지시 「Fable 결과에 동의하기 힘들면 grok 독립감사」)**: Fable=계약 기안자+코드 감사자 자기감사 편향 + GPU 10h 지출 게이트 + cadence 미묘 주장 ⟹ 다른 계열로 교차. **GROK_AUDIT 통과·SEAL_CONTACT 가·BLOCKING 0·FABLE 결론 확인·반박 0.** grok 이 Fable 결론 미열람으로 소스 독립 재도출(begin fresh 복사·closure 기본 OFF·공개 인구 commit 전 불변·R7 pass당 commit `:16873` 1회·자기 소거 소멸·OMP 직렬 무경합·OFF byte-동일·a 리듀서 독립). 기재 동일 계급 재도출(차단 승격 0). ⟹ 두 독립 계열 일치, 높은 확신.
8. **재봉인·빌드·스테이징·제출(운전자)**: 재봉인 = source.sha256 184→185(신규 tests/pop_djdn_selftest.c 추가), 델타 정확히 3파일(env_universe.h·lumina_plasma.c 변경 + selftest 신규, 잡편집 0). 빌드 = **job 387077 build.rc=0, binary `73d8d3390e8112...`, source_after fails=0**(빌드 중 소스 불변). ENV_UNIVERSE.json 재생성 **불요**(동결 selftest 미배선·빌드 사슬 밖). 스테이징(`stage_djdn.sh`, driver-hygiene rc=0): 기준 385315(run_h100a100, DR on·ION_CHANNEL_DUMP=1)·**exports 편집 = ITER 2→4 · `LUMINA_CMF_CONT_NLTE_SOURCE=1` 추가(CONT 소스 ON — DJDN 이 잡을 폭주 생성) · `LUMINA_POP_DJDN=1` 추가**, DR 5키·ION_CHANNEL_DUMP=1 유지, MAXCH 미설정(off), NT/SIMUL 부재. G-ID exports 델타 = h100-a100x2 **4줄**·h200x1 **6줄**(+MGPU 2→1). 러너 ITER 검사 2→4·outer_iterations 4·--time 14:00:00. deck 해시 OK. **헤지 제출(§6④)**: **387121**(h100-a100x2, run_h100a100)·**387122**(h200x1, run1) — 먼저 시작하는 쪽 유지·나머지 소유 확증 후 취소.
8-b. **멀티파티션 전환(user 2026-09-20 「h100/h200/a100 멀티파티션으로 · h200 이상해」)**: 헤지 387121/387122 취소(소유 확증 WorkDir=djdn_mvp). 원인 = h200(syn104)은 GPU 8장 중 여유(mix)인데 387122(h200x1)가 START_TIME 없이 PENDING — 스케줄 미착. 러너 h200x1(1 GPU·140GB)과 h100-a100x2(2 GPU·80GB)는 GPU 수·메모리·kind 가 파티션에 묶여 한 잡이 못 걸침. ⟹ **2-GPU 프로파일을 h200 까지 확장**(러너 line 4·51 편집: `--partition=h100,h200,a100`·`expected_partition=h100,a100,h200`·`expected_kinds=H100,A100,H200`; MGPU=2·min 80GB 불변 — h200 노드가 2×H200 제공·140GB≥80GB). ⚠**G-ID 러너 편집 확장(기재)**: 계약 §4 허용 편집(job-name·time·ITER·outer·프로필 partition/gres/GPU_PROFILE)에 **multipart_h200 추가** — 물리 무관(같은 binary·exports·MGPU=2, kind 만 넓힘), STAGE_IDENTITY runner_edits 기재. driver-hygiene rc=0. **재제출 단일 잡 387154**(h100,h200,a100, 2-GPU). 판정 대상 = 387154.
8-c. **조기 착지(user 2026-09-20)**: 387154(cpus=32)가 h200/h100(syn08) sjjlee 30일 예약(kjhan 불허)·클러스터 포화로 syn09 12:01(≈11h) 대기. syn101(a100)에 A100 2장 유휴이나 유휴 CPU 22<32 로 막힘. user 승인(질문 「CPU 16 축소 조기착지」) ⟹ 387154 취소(WorkDir 확증)·**cpus-per-task 32→16**(러너 line 8·57 ALLOCATION_SHAPE 검사 ==16). ⚠G-ID 러너 편집 확장 기재 `cpus32to16` — GPU-bound 라 런타임 무영향·판정 분기(거친 비) 무영향; 결정론 FP 하위자리 미세변동 가능하나 이 진단 런은 byte-배터리 없음. driver-hygiene rc=0. **재제출 387175 → syn101(a100) 01:13:27 RUNNING**(즉시 착지, ~11h 절약). 판정 대상 = **387175**.
9. **착지·판독·판정(운전자)**: 387175 완주(a100 syn101, 9.8h, model_rc=0·outer_iterations=4·census 18450행·physics iter 0-3). **형식 분기 MVP-X**(판정문 `docs/VERDICT_DET_DJDN_MVP_2026-09-20.md`), **단 「대각 불충분」이 아니라 「대각 미시험」**. G-CPL 통과(항등 위반 0·a>0·대각 대수 in-vivo 정확). G-RUN 미충족 = 운전자 postcheck `-ne 2` 스테이징 버그(런 무결성 확인, `-ne 4` 였으면 SUCCESS). G-DRV 미충족 = census gen 이 하위패스 카운터라는 계약 referent 결함. **운전자 재현**: census=5 epoch×~25 하위패스, **대각 항 root=1 은 씨앗 epoch 뿐**(sh16/20/24 epoch2-5 전부 root=0), sh24 폭주(x_map 60→338)에 항 OFF, g≥1 12중 11 양의근(「g≥1⟹근없음」 대수오류). ⟹ 구동∧적용 교집합 0 = 미시험. **독립 방향 평가 2건**(Fable+grok, `FORK_EVIDENCE.md`) 수렴: 권고 (1)·(3) 금지. **운전자 원 프레임 「g≥1→대각불충분→Path B」 철회**. 운전자 이탈 4(postcheck·multipart_h200·cpus32to16·오독 프레임). 계약 결함 3(Fable §0b·§2·§4 자기지목). 다음 = 새 계약(Fable 기안, g≥1 폴백 없는 감쇠 뉴턴·구동 국면·s−g 측정).
10. **종료감사(Opus 4.8 fresh, `claude-opus-4-8`): CLOSEOUT 통과·BLOCKING 0.** 판정문 주장을 런 산출로 전건 재현 일치 — census 5 epoch×~25 하위패스·대각 root=1 씨앗 epoch뿐(구동 epoch2-5 root=0)·sh24 x_map 60.8→118.6→338.3 폭주·g≥1 12중 11 양의근(sh24 epoch5 하나만 음)·G-CPL 항등 위반 0·postcheck TE 4≠2. 「대각 미시험(구동∧적용 교집합 0)」 재프레임·운전자 원 프레임 철회 정당(데이터 지지)·운전자 이탈 4건 정직 기재(postcheck 「-ne 4 면 SUCCESS」 골대이동 아님)·주장 상한 준수·다음 표적 타당. 지적 1건 «기재로 족함»: G-CPL 잔차 정규화 부기(판정문 §1 반영). **DET-DJDN-MVP 폐합.**

---

**Return notes for the driver (not part of the prereg):**

- §0(a) and §0(b) are the load-bearing deviations from the brief. Both are grep-verifiable on the frozen tree in seconds; I recommend the Q-GOAL pre-audit be asked to re-derive them independently rather than take my word. If the driver believes `radeq_simul_all` is reachable in the DR-on configuration by some route I missed, that must be shown before dispatch — otherwise the brief's `LUMINA_POP_DJDN` in `simul_ladder` would be inert and the run would be MVP-X by construction.
- I registered a linear-model expectation (MVP-B/C favoured) as 기안자 예상, explicitly not a judgment condition and without a probability; the branches do not depend on it. If the driver's "예측: 미정" ruling is meant to forbid even that, delete the one sentence in §5 — nothing else references it.
- Repo `src/env_universe.h` does not carry the CONT keys (`LUMINA_CMF_CONT_NLTE_SOURCE`, `LUMINA_ION_CHANNEL_DUMP`) because the CONT change set lives only in the frozen tree; the `references` block therefore cites only symbols present in the repo. `check_prereg_preflight --root .` should report `changeset-consistent files=3` via `expected_extra`, as for the CONT contract.
- Data-driven narrowing: judged quantity is S only (Si non-monotone in both CONT and base; Ca at r ~10⁶–10⁷). Si/Ca stay as recorded quantities.
