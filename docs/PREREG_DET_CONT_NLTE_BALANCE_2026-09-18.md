# 사전등록 — DET-CONT-NLTE-BALANCE: 갇힌 연속체의 이온화 균형 (연속체 소스 = Milne η_bf · DR · 비열적 · 외부반복 10) (2026-09-18)

저자 = 운전자(Claude, 개정25). 상태 = **기안 / AUDIT_PENDING**. 실행 승인이 아니다. user 결정(2026-09-18 「권고안대로 진행해」): ITER 10 레인 · `NT_FULL` · 한 계약.
근거 판독 = `docs/READOUT_DEGENERATE_LEDGER_2026-09-18.md`(판독 C) + `docs/READOUT_LAMP_PREDICTION_2026-09-17.md` §1 ①. 상위 = `docs/VERDICT_DET_LTEUNMAP_EPS_2026-09-18.md`(§7 후보 ①). base = 판정런 **376251**(`…/lteunmap_eps_20260917/run5_h200`, 램프 소등 상태).

## 1. 계약 — 물리 계약 1개

**「bin 조립기(커밋 J 의 생산자)의 연속체 방출은 Kirchhoff-LTE(χ_abs·B(T_e)) 가 아니라 실제 인구의 Milne 방출 η_bf(`bf_get_eta`, ff 는 열적) 이고, 이온화 균형은 광이온화 + 충돌 + 비열적(heating/W) 대 RR + DR 로 완결된다.」**
판정런 1회(twin-T_e, 외부반복 **10**). 참이면 중간 셸(9–24)의 S 가 LTE(T_e)의 III 지배에서 벗어나 II 로 옮겨 CMFGEN(.71–.85) 쪽으로 간다(판독 C 의 퇴화 극한 r_deg(NT 1.0) = 0.34–1.7 ⟹ f_II .55–.75).
⚠**이 계약이 말하지 않는 것**: Si·Ca 의 정확한 착지(FUV 광이온화 지배 — mapped 선 S=B 열숲이 FUV 를 열화시키면 과재결합으로 튈 수 있음, §5 상한으로 기재)·스펙트럼·자유-T_e.

### Q-GOAL (기안자 자기 답 — 사전감사가 독립 재판정)
① 줄이는 칸: 어긋남 지도의 **중간 셸 9–24 S II/III**(현 .02–.10 대 CMFGEN .71–.85; 질량 최대 영역) — 폐합된 DET-LTEUNMAP-EPS 가 남긴 최대 어긋남. ② 물리 측정(이온 분율·반복별 수렴). 신규 계측 = 조립기 census 1 행/셸/패스(§4 G-SRC 소비). ③ 대안: NLTE 승격 IV·V(외곽), mapped 선 ε 채널 — 둘 다 중간 셸 S 를 건드리지 않음. ④ 건너뛰면 중간 셸의 LTE 고정이 모든 후속 CMFGEN 대조를 오염(판독 B 의 「올바른 T 의 LTE」 상태에 머묾).

### Q-LEAN (기안자 자기 답)
L1 물리량 = 셸별 이온 분율(반복별 `physics_DET_iter000k.ion.csv`·최종 `lumina_ion_pops.csv`), 창/문턱 빈 J/B. 신규 = 게이트 env 1 + 조립기 census 1 행/셸/패스 + selftest 1; 설정 키 3(DR·NT·NT_FULL)과 반복수 1 은 **기존 노브**. L2 게이트 3(G-ID·G-RUN·G-SRC) — 각각 실재 가능한 실패. L3 = 1 걸음(수리 1 + 판정런 1). L4 삭제 후보 없음 — 추가하지 않은 것: 새 판독기·새 상태 필드·둘째 런.

## 2. 기전 — 런 0회 확정분

- **채움 ①**: `cmfgen_assemble_impl` 의 `S_fixed = (chi_a·B + eta_ln)/chi_t`(동결 트리 `lumina_cmfgen.c` ~2425 행; `chi_a = chi_bf + chi_ff`) — bf-불투명 빈(100–912 Å 의 65–100 %)에서 J → B(T_e)(`planck_census`), 그래서 중간 셸 이온비 = Φ_LTE(판독 B·A; 376251 에서 E-5 = 0.81–0.96).
- **η_bf 는 이미 빌드된다**: `LUMINA_CMF_BF_MILNE=2`(런 env) → `bf->eta_bf`(Menzel 이탈 b_l 로 준안정 준위는 NLTE 형, 그 밖은 열적; LTE 극한 S_bf→B; `lumina_plasma.c` ~7949 행) → `bf_get_eta(bf,s,ν)`(~7618 행, ν 보간). 조립기는 EPAY 예산(미설정)에서만 참조하고 **S_fixed 에는 쓰지 않는다**.
- **퇴화**: J → S_bf = B·(r/Φ) 이면 광이온화-RR 이 항등 상쇄되어 r 은 **남는 채널**(충돌·비열적 대 RR+DR)로 정해진다. 장부(판독 C §2): S II→III sh12–24 r_deg(NT 1.0) 0.34–1.7 대 CMFGEN 0.18–0.72; NT 0.05 면 0.017–0.084. Si·Ca 는 FUV 광이온화 지배(CMFGEN r 5–68 / 10⁵).
- **채널 실물**: DR = `LUMINA_FROZENIN_DR=1` → `frozenin_alpha_rr` 안 `dr_lookup(Z, stage+1)`(ADAS/Badnell `DR_TABLE`, S·Si·Ca II→III 재결합 커버) · 비열적 = `LUMINA_ION_NT=1` → `parity_rate_se_ratio_checked` 의 `Gamma += gamma_nt_atom`(`gamma_deposition_compute_nonthermal`, `heating/W`, `LUMINA_NT_FULL=1` 이면 pref 1.0; 퇴적 = `deposition_cmfgen.csv`, `LUMINA_GAMMA_DEP=1` 런 env) · 충돌 = 코드 내장. 전부 strict env 등록됨.
- **반복수**: LTE 시드에서 소스만 바꾸면 반복당 r 이 α_RR/(α_RR+α_DR) = 0.40(S)·0.27(Si) 배 이동 ⟹ 2회는 구성상 미결(판독 C §4). 러너의 `STRICT_TWO_ITERATION_CONTRACT`(동결 러너 116 행)·`outer_iterations=2`(132 행)·`ION_POP_DUMP_ITER==1`(118 행) 검사는 스테이징 복사본에서 ITER=10 으로 편집한다(러너 해시 G-ID 에 기재).
- **미확정(이 계약 밖)**: mapped 선 S=B 열숲의 FUV 열화(Si·Ca 착지) · x_e 의존 η_ion(비열적 강도의 중간값) · CMFGEN toy06 의 비열적 처리 포함 여부(문헌 미확인 — 기재).

## 3. 수리안 — 변경집합

원칙: 게이트 env `LUMINA_CMF_CONT_NLTE_SOURCE`(기본 0 = byte-동일). ON 이면 `cmfgen_assemble_impl` 의 연속체 소스를 `S_fixed = (η_bf(s,ν) + chi_ff·B + eta_ln)/chi_t` 로 바꾼다(`η_bf = bf_get_eta(bf,s,ν)`; ff 는 열적 유지; `chi_a`·`chi_t`·`eta_ln`·EPAY·`kappa_dep` 항은 불변). **fail-closed**: 게이트 ON 인데 `bf==NULL`·`bf->eta_bf==NULL`·`LUMINA_CMF_BF_MILNE<1` 이거나 **Milne(bf) 부분** η_bf 가 전 빈 0 인 셸(bf 빌더가 셸별로 기록하는 `bf->eta_milne_ok[s]`: bf 불투명 빈이 있는데 ff 를 더하기 **전** η_bf,milne>0 인 빈이 하나도 없으면 0)이 있으면 조립 실패(무음 B·순-ff 열적 폴백 금지). ⚠`bf_get_eta`/`bf_get_chi` 는 빌더 ff(`chi_ff_c`·`chi_ff_c·B`)를 포함하므로 「η_bf 전 0」 검사는 ff 만으로도 항상 통과한다 — 그래서 Milne 부분을 따로 본다(사전감사 B-2). census: 패스·셸당 stderr 1 행 `[cmfgen_assemble][CONT-SRC] pass=<p> shell=<s> n_bf_bins=<n> S_bf_over_B_mean=<Σ η_bf dν / Σ chi_bf B dν> S_bf_over_B_min=<..> max=<..>`(bf 불투명 빈 기준). fine 경로는 건드리지 않는다(연속체 보간은 `chi_abs/chi_es` 만 받으므로 무영향).
**채널 분해 덤프**(사전감사 B-1·Q-LEAN L1 — 착지를 귀속할 계측): env `LUMINA_ION_CHANNEL_DUMP`(기본 0 = 파일 없음·byte-동일). ON 이면 `parity_rate_se_ratio_checked` 가 (Z∈{16,14,20}, stage 1→2) 쌍에 대해 작업 디렉토리 `lumina_ion_channels.csv` 에 행을 append 한다 — 열 `gen,shell,Z,stage,T_e,n_e,x_e,gamma_phot,gamma_nt,c_ion,ne_alpha_rr,ne_alpha_dr,c_rec,r`(`gen` = 셸 0 호출마다 +1 하는 파일-static 세대 번호; `x_e = n_e/n_atom`; `ne_alpha_rr`·`ne_alpha_dr` 는 `frozenin_alpha_rr` 안의 RR 과 DR(`dr_lookup`) 몫을 갈라 n_e 를 곱한 값; `r` 은 함수가 돌려주는 비 그대로). 판정은 최대 `gen` 의 150 행만 쓴다. 이 덤프의 항등 `r = (gamma_phot+gamma_nt+c_ion)/(ne_alpha_rr+ne_alpha_dr+c_rec)` 는 **구성상 참**이라 예측이 아니고 열 배선의 음성대조다. 소스 기준 = **동결 트리 복사본**(`/gpfs/kjhan/lumina/cmfgen_map/lteunmap_eps_20260917/source`, DET-LTEUNMAP-EPS 변경집합 포함 상태) — 그 위에 아래 파일만.
설정(코드 변경 아님, exports 키): `LUMINA_FROZENIN_DR=1`, `LUMINA_ION_NT=1`, `LUMINA_NT_FULL=1`, `LUMINA_PURE_CMFGEN_ITER=10`, 게이트 `LUMINA_CMF_CONT_NLTE_SOURCE=1`, `LUMINA_ION_CHANNEL_DUMP=1`; 기존 `LUMINA_CMF_FINE_UNMAPPED_EPS=1` 유지.

### 변경집합 — CMF_CONT_NLTE_SOURCE (표)

| 경로 | 담당 | 내용 |
|---|---|---|
| `src/lumina_cmfgen.c` | Opus 4.8 | `cmfgen_assemble_impl` 연속체 소스 교체(게이트 ON 만)·fail-closed·census 행; 게이트 OFF byte-동일 |
| `src/env_universe.h` | Opus 4.8 | `LUMINA_CMF_CONT_NLTE_SOURCE`·`LUMINA_ION_CHANNEL_DUMP` 등록 |
| `src/lumina_plasma.c` | Opus 4.8 | bf 빌더: ff 합산 **전** 셸별 `eta_milne_ok` 기록; `parity_rate_se_ratio_checked`: 채널 분해 덤프(게이트 ON 만, RR/DR 분리) |
| `src/lumina.h` | Opus 4.8 | `BFOpacity` 에 `eta_milne_ok`(셸별 int) 추가 |
| `tests/cmf_cont_nlte_source_selftest.c` | Opus 4.8 | CPU selftest: 게이트 OFF S_fixed byte-동일 · ON 이면 S_fixed = (η_bf+χ_ff B+η_ln)/χ_t(1e-12) · LTE 극한(η_bf = χ_bf B) 에서 OFF 와 동일 · η_bf 부재/Milne 부분 전 0(`eta_milne_ok=0`) → rc≠0 · ff 만 있는 셸은 fault(순-ff 폴백 검출) · census 형식 · 채널 덤프 행 항등(열 바꿔치기 ⟹ FAIL) |

### 변경집합 끝

## 4. 게이트 명부 (표)

| ID | 조건 | 음성대조 |
|---|---|---|
| G-ID | 376251 계보(`/gpfs/kjhan/lumina/cmfgen_map/lteunmap_eps_20260917/run5_h200/{RUN_FOOTER.txt,input/resolved_lumina.exports,input/binary.sha256}`) 대비 exports 차이가 정확히 6 키(`LUMINA_CMF_CONT_NLTE_SOURCE`·`LUMINA_ION_CHANNEL_DUMP`·`LUMINA_FROZENIN_DR`·`LUMINA_ION_NT`·`LUMINA_NT_FULL`·`LUMINA_PURE_CMFGEN_ITER`; 프로필 종속 `LUMINA_CMF_FINE_MGPU_DEVICES` 허용), 소스 차이 = §3 변경집합 표의 5 파일(`src/lumina_cmfgen.c`·`src/env_universe.h`·`src/lumina_plasma.c`·`src/lumina.h`·`tests/cmf_cont_nlte_source_selftest.c`; 기준 = 376251 의 `input/run_build_identity.json` 소스 해시 — 편집된 동결 트리가 아니라), 러너 편집 = ITER/덤프 검사·`--time`·`outer_iterations` 4 곳뿐(해시 기재) | 다른 키 주입 ⟹ FAIL |
| G-RUN | 외부반복 10 완주(`[R7][PHASE] event=R7_MATERIAL_PHASE_COMMITTED … iter=9`), `work/physics_comparison/physics_DET_iter0009.ion.csv`·`work/lumina_ion_pops.csv`·`work/cmfgen_final_publication.json` 실재, `[cmf_fine][EXACT-MULTIGPU-EPOCH] status=OK` ≥ 11 | 반복수 변조·덤프 결손 |
| G-SRC | stderr `[cmfgen_assemble][CONT-SRC]` 행이 매 패스마다 셸 0–49 전부 있고, 마지막 패스 셸 12·16·20 의 `S_bf_over_B_mean` 이 **1 이 아니고**(\|log10\| > 0.05 — 소스가 실제로 NLTE 로 바뀌었음) **물리 밴드 안**(0.1 ≤ mean ≤ 10, `S_bf_over_B_min` > 0 — 순-ff 붕괴 소스(mean ≪ 0.1)나 재점화 램프를 「바뀜」으로 오통과시키지 않음; base 376251 은 이 행이 없고 S_bf/B 는 판독 `opaque_source_check_run5` 의 0.8–1.7) | census 를 1.0 으로 위조 ⟹ FAIL; 0.01 로 위조(ff 붕괴) ⟹ FAIL; 행 결손 ⟹ FAIL |
| G-CHAN | `work/lumina_ion_channels.csv` 실재, 최대 `gen` 에 (Z 16·14·20)×셸 0–49 = 150 행, 14 열 헤더 일치, 전 행 항등 \|r·(ne_alpha_rr+ne_alpha_dr+c_rec) − (gamma_phot+gamma_nt+c_ion)\| ≤ 1e-9·max(둘) | 열 두 개 바꿔치기 ⟹ FAIL; 행 결손(쌍이 rate 가지에서 안 풀림) ⟹ FAIL |

### 게이트 명부 끝

## 5. 기대치 — 런에서만 얻는 수 (자릿수 예측 = 판독 C §2 퇴화 극한; 잣대를 base 에 먼저 적용: base 376251 은 E-1·E-2 를 넘지 못한다)

| ID | 양 | base(376251) | 예측 | **기대** |
|---|---|---|---|---|
| E-1a S 중간 이탈 | S II→III 비 r 셸 12·16·20 (`lumina_ion_pops.csv`; f_II 환산 병기) | r 70 / 21 / 9.1 (f_II .017/.053/.102) | deg(1.0, C1 정정) r .57/.85/1.37 (f .64/.54/.42); CMFGEN r .18/.25/.40 (f .85/.80/.71) | **셋 다 r ≤ base/10^0.5**(≤ 22 / 6.6 / 2.9 — LTE 에서 반 자릿수 이상 이탈) |
| E-1b 갇힘 측정 | D = gamma_phot / (r · ne_alpha_rr) 셸 12·16·20 (`lumina_ion_channels.csv` 최대 gen) | (base 는 덤프 없음; LTE 극한 J=B 라 D ≫ 1 상당) | 갇힌 연속체 ⟹ 광이온화≈RR ⟹ D ≈ 불투명 분율 0.65–0.94 | **셋 다 0.3 ≤ D ≤ 3** |
| E-1c 과재결합 상한 | S f_II 셸 12·16·20 | .017/.053/.102 | pref 0.05 물리값이면 deg .97/.96/.94(CMFGEN 을 넘음) | **셋 다 ≤ 0.95** |
| E-2 Si 중간 | Si f_II 셸 12·20 | 0.000 / 0.000 | CMFGEN .014/.13; deg(1.0, C1 정정) .78/.58(과재결합 상한 — 758 Å 빈이 갇히면 여기로) | **둘 다 ≥ 0.01 그리고 ≤ 0.50**(하한 = 광이온화 회복, 상한 = FUV 열화 없음) |
| E-3 수렴 | S II→III 비 r(sh12) 의 반복별 `physics_DET_iter000k.ion.csv` 궤적: \|log10 r₉ − log10 r₈\| | (2회 런: 미측) | 기하 수렴비 0.4 ⟹ 9 회에 잔차 < 0.1 dex | **< 0.10 dex** |
| E-4 외곽 유지 | S f_III 셸 28·32·36 | .998/.994/.902 | 램프 소등 유지 | **셋 다 ≥ 0.70** |
| E-5 대조 | 셸 0–8 IGE IV 분율 \|Δ\| | 0.98 | LTE 극한 S_bf→B 라 소폭 | **\|Δ\| < 0.05** (기재용) |

### 분기 (표)

| 분기 | 조건 | 의미 |
|---|---|---|
| CB-X | G-ID/G-RUN/G-SRC 미충족 | 미시연 |
| CB-A | 게이트 전건 **and** E-1a·b·c **and** E-2 **and** E-3 **and** E-4 | **갇힌 연속체 균형 — 관측 등급**(pref 1.0 단일 런; 정격 승격은 pref 브래킷 동반런 후) — 중간 셸 S 가 LTE 를 벗어나 채널 균형에 착지, Si 광이온화 회복 |
| CB-B | 게이트 전건 **and** E-1a·b·c **and** E-3 **and not** E-2 | S 는 옮겼으나 Si 가 과재결합(FUV 열숲) 또는 미회복 ⟹ 다음 rung = mapped 선 ε/NLTE 소스 |
| CB-C | 게이트 전건 **and** E-3 **and not** E-1a | 수렴했는데 S 가 안 옮김 ⟹ 채널 부족(NT 강도·창 탈출) — 판정자가 명명 |
| CB-C′ | 게이트 전건 **and** E-3 **and** E-1a **and not**(E-1b **and** E-1c) | 옮겼으나 갇힘 아님(D 밖) 또는 과재결합(f_II > .95) — 판정자가 채널 덤프로 귀속·명명 |
| CB-M | 그 외(특히 E-3 미수렴) | 판정자가 명명, 재분기 금지 |

### 분기 끝

문턱 근거: E-1a 0.5 dex 는 C1 정정 후 deg(1.0) 예측(sh20 이 base 아래 0.8 dex 뿐)에 0.3 dex 여유를 둔 값(구 「f_II ≥ 0.50」은 α_RR 을 분모에 남긴 틀린 대수 위의 문턱이라 폐기); E-1b [0.3, 3] 은 불투명 분율 0.65–0.94 의 ±0.5 dex; E-1c 0.95 는 CMFGEN 최대 .85 위이며 pref 0.05 퇴화 예측(.94–.97)을 실패로 읽는 선; E-2 상한 0.50 은 퇴화 극한 과재결합(.65–.83)을 실패로 읽는 선; E-3 0.1 dex 는 수렴비 0.4 의 9 회 잔차(0.4⁹ ≈ 3e-4 dex·초기 2 dex ≈ 0.03 dex)의 3 배 여유.

## 6. 자원 · 실행

GPU **1 런**, 프로필 h200x1 우선(대기 길면 h100-a100x2 헤지, 먼저 시작하는 쪽만) — 376251 실측 5 h/2 반복 ⟹ ITER 10 ≈ **18–20 h**, `--time=30:00:00`, 상한 **40 GPU-hour**. 새 RUN 루트(`…/lteunmap_eps_20260917/` 형제 루트 `cont_nlte_20260918/`), 입력 = 376251 입력 복사 + 새 바이너리 + exports 5 키. 오프라인 ≤ 1 CPU-hour. 봉인 base 는 읽기만.
순서(개정16 ②): Opus 4.8 구현 → selftest → 운전자 코드 간단 감사(Q0) → 빌드(CPU) → 스테이징(러너 4 곳 편집·해시 기재) → 제출.

## 7. 분장 장부 (개정25)

| 단계 | 규약 | 실제 | 비고 |
|---|---|---|---|
| 기안 | 운전자 → **Fable fresh**(개정25-b, user 09-18) | 원안·개정 1 = 운전자(개정25-b 전) · 이후 개정 = Fable fresh | |
| ~~사전감사~~ (개정25-c 로 폐지, user 09-18) | ~~Opus 4.8 fresh~~ | 폐지 **전** 이 계약은 이미 통과: 1회차 차단 2(집행 기록 5) → 2회차 통과·차단 0(집행 기록 9) | 남은 개정은 사전감사 없이 진행 |
| 구현 + selftest | Opus 4.8 fresh | Opus 4.8 fresh(`claude-opus-4-8` 자기보고) — 집행 기록 2 | `model` 생략 |
| 코드 간단 감사 | 운전자 → **Fable fresh**(개정25-a) | 1회차 = 운전자(집행 기록 3) · 개정 1 = **Fable fresh 통과·차단 0**(집행 기록 11) | |
| 빌드·스테이징·제출·판독·판정 | 운전자 | 빌드 제출(집행 기록 4) · 이후 미집행 | |
| 종료감사 | Opus 4.8 fresh | 미집행 | 생략하지 않음 |

## 8. 주장 상한
「갇힌 연속체 소스를 NLTE 로 두고 채널을 완결하면 중간 셸 S 가 CMFGEN 쪽으로 옮기는가」와 「Si 가 광이온화 지배로 회복되는가(상하한)」까지. 비열적 강도의 정확값(0.05–1.0 사이)·x_e 의존 η_ion·mapped 선 열숲·자유-T_e·스펙트럼은 말하지 않는다. twin 핀 조건부.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 — CMF_CONT_NLTE_SOURCE (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "(?:src|tests)/[A-Za-z0-9_/.]+\\.(?:c|h)",
    "symbol": "CMF_CONT_NLTE_SOURCE",
    "roots": ["src", "tests"],
    "expected_extra": ["src/lumina_cmfgen.c", "src/env_universe.h", "src/lumina_plasma.c", "src/lumina.h", "tests/cmf_cont_nlte_source_selftest.c"]
  },
  "branches": {
    "regimes": [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
    "metrics": {"count": "len(v)", "gate_fail": "sum(1 for x in v if x < 0.5)", "balance": "sum(1 for x in v if 0.5 <= x < 1.5)", "si_fail": "sum(1 for x in v if 1.5 <= x < 2.5)", "s_stuck": "sum(1 for x in v if 2.5 <= x < 3.5)", "other": "sum(1 for x in v if x >= 3.5)"},
    "rules": [
      {"name": "CB-X", "predicate": "count == 0 or gate_fail > 0"},
      {"name": "CB-A", "predicate": "count > 0 and gate_fail == 0 and si_fail == 0 and s_stuck == 0 and other == 0"},
      {"name": "CB-B", "predicate": "count > 0 and gate_fail == 0 and si_fail > 0 and s_stuck == 0 and other == 0"},
      {"name": "CB-C", "predicate": "count > 0 and gate_fail == 0 and s_stuck > 0 and other == 0"},
      {"name": "CB-C'", "predicate": "count > 0 and gate_fail == 0 and s_stuck > 0 and other == 0 and False"},
      {"name": "CB-M", "predicate": "count > 0 and gate_fail == 0 and other > 0"}
    ],
    "adversarial_fixtures": [
      {"name": "empty", "mix": []},
      {"name": "gate-fail", "mix": [[0.0, 1], [1.0, 2]]},
      {"name": "balance", "mix": [[1.0, 3]]},
      {"name": "si-fail", "mix": [[1.0, 1], [2.0, 1]]},
      {"name": "s-stuck", "mix": [[3.0, 1]]},
      {"name": "other", "mix": [[4.0, 1]]}
    ],
    "residual": "CB-X"
  },
  "references": [
    {"path": "docs/READOUT_DEGENERATE_LEDGER_2026-09-18.md"},
    {"path": "docs/READOUT_LAMP_PREDICTION_2026-09-17.md"},
    {"path": "docs/VERDICT_DET_LTEUNMAP_EPS_2026-09-18.md"},
    {"path": "CLAUDE.md"},
    {"path": "src/lumina_cmfgen.c", "flags_existing": ["cmfgen_assemble_impl", "bf_get_eta", "S_fixed", "LUMINA_CMF_EPAY"]},
    {"path": "src/lumina_plasma.c", "flags_existing": ["bf_get_eta", "LUMINA_CMF_BF_MILNE", "LUMINA_FROZENIN_DR", "LUMINA_ION_NT", "LUMINA_NT_FULL", "dr_lookup", "gamma_deposition_compute_nonthermal", "parity_rate_se_ratio_checked", "frozenin_alpha_rr", "chi_ff_c"]},
    {"path": "src/lumina.h", "flags_existing": ["BFOpacity", "bf_get_eta"]},
    {"path": "src/env_universe.h", "flags_existing": ["LUMINA_FROZENIN_DR", "LUMINA_ION_NT", "LUMINA_NT_FULL"]}
  ]
}
```

## 개정 1 (2026-09-18, 사전감사 1회차 Opus 4.8 fresh 「통과 불가·차단 2」 반영) — OLD/NEW 전건은 §9 집행 기록 5 에 지목한 대로 본문에 적용

| # | 출처 | 요지 |
|---|---|---|
| R1-1 | B-2 | fail-closed 를 **Milne 부분**으로(`bf->eta_milne_ok[s]`, ff 합산 전 기록) — `bf_get_eta` 가 ff 를 포함해 「전 0」 검사는 무력했다 |
| R1-2 | B-2 | G-SRC 에 **물리 밴드** 0.1 ≤ mean ≤ 10·min > 0 추가(순-ff 붕괴·재점화 램프 오통과 차단), 음성대조 「0.01 위조 ⟹ FAIL」 추가 |
| R1-3 | B-1·Q-LEAN L1 | **채널 분해 덤프** `lumina_ion_channels.csv`(env `LUMINA_ION_CHANNEL_DUMP`, Z 16·14·20 stage 1→2, 14 열, RR/DR 분리) + 게이트 G-CHAN(항등 = 배선 음성대조) — 착지가 「기전 재현」인지 「NT 노브 착지」인지 귀속 가능 |
| R1-4 | C1 | 판독 C 대수 정정: 광이온화–RR 이 항등 상쇄되면 α_RR 은 **소거**되어 r_deg = (C_ion+Γ_nt)/(n_e α_DR + C_rec). S 는 판독표 ×1.67, Si ×1.37, Ca(α_DR≈0) 는 r→C/C_rec ≫ 1(f_II→0, CMFGEN 0 과 정합). deg(1.0) S f_II .64/.54/.42(sh12/16/20) — 구 E-1 「≥ 0.50」 은 틀린 대수 위의 문턱이라 폐기 |
| R1-5 | B-1 | E-1 을 셋으로: **E-1a** LTE 이탈(r ≤ base/10^0.5) · **E-1b** 갇힘 측정 D = Γ_phot/(r n_e α_RR) ∈ [0.3, 3] · **E-1c** 과재결합 상한 f_II ≤ 0.95. CB-A 는 **관측 등급**(pref 1.0 단일 런)으로 강등, CB-C′ 신설 |
| R1-6 | B-1 pref | **pref 는 1.0 유지**(user 09-18 「권고안대로」 결정) — 사전감사 지적 「x_e≈2 에서 Xu&McCray η_ion→0 이라 물리 pref ≤ 0.05, 1.0 은 물리 상한 초과」 는 **기재**하고, 채널 덤프의 Γ_nt 몫으로 착지 귀속을 남긴다. pref 0.05 동반런(브래킷) 은 user 결정 항목(+20 GPU-h). 방향: pref 1.0 은 E-1a 를 **더 어렵게** 한다(f_II 를 .97→.64 로 끌어내림) — 통과용 과대구동 아님 |
| R1-7 | C3 | DR ON 시 RR α 는 U_II-정정본(`lumina_plasma.c` ~6572 주석: 「구 2.8× α 가 DR 대행이었음을 U_II 정정이 드러냄」) ⟹ DR 이중 계상 없음 [저자 실행 grep]. DR 역과정(자동이온화) 미구현은 비대칭으로 기재 |
| R1-8 | 절차 | 사전감사보다 구현이 선행했음을 기재(개정16 ② 순서 위반 아님 — 봉인 접촉 게이트는 아직 안 돌렸다); G-ID 소스 기준을 376251 `run_build_identity.json` 으로 명시 |

E-3(수렴 10 회 충분성은 추정 — 축퇴가 클수록 수축비가 1 에 접근, C4)·C7(4 노브 동시 변경, 대조 런 없음 — user 결정 「한 계약」 유지; 채널 덤프가 부분 보전)·E-5(음성대조로서 정당)은 «기재로 족함» 으로 여기 남긴다.

## 9. 집행 기록

1. **게이트(발주 전)**: `check_prereg_preflight` rc=0 · `check_gate_referents` rc=0(「셸 50 개」·행 범위 표기 정정 후) · `check_doc_canonical` rc=0 · `check_dispatch_preflight` rc=0(발주서의 재서술 경로·건수를 절 지목으로 교체 후). 커밋 4d4b86c.
2. **구현(Opus 4.8 fresh, `claude-opus-4-8` 자기보고)**: 동결 트리 `…/lteunmap_eps_20260917/source` 위 3 파일 — `src/lumina_cmfgen.c`(+145: 분자·전제·셸 fail-closed 술어·census 서식 헬퍼 4, 게이트 판독, `bf_get_eta` 소비, `S_fixed` 교체, 드라이버 차단 `g_asm_cont_src_fault`), `src/env_universe.h`(+1 키, COUNT 522), `tests/cmf_cont_nlte_source_selftest.c`(신규). 보고: 게이트 OFF byte-동일(분자 헬퍼가 OFF 에서 원식과 같은 연산 순서), EPAY·`kappa_dep`·fine 경로 불변.
3. **코드 간단 감사(운전자, Q0 = 봉인 접촉 가부)**: `diff -rq` 대 343204 base = 5 파일(두 계약 합) ✔. 분자 OFF 식 `(chi_bf+chi_ff)*B+eta_ln` 은 base `chi_a*B+eta_ln`(`chi_a=chi_bf+chi_ff`)과 동일 연산 ✔. ON 식 = 계약 §3 ✔. 전제 fail-closed(bf/eta_bf 부재·`BF_MILNE<1`)·셸 fail-closed(bf 빈 있고 η_bf 전 0) → `[cmfgen_assemble][BLOCKED]` + 드라이버 `cmfgen_run` rc −1 ✔. census 1 행/셸/패스, mean = Σ η dν / Σ χ_bf B dν ✔. `bf_get_eta` 는 EPAY≥2 에서 이미 `chi*B` 와 같은 단위로 쌍을 이룸 ✔. selftest2 12 PASS(LTE 극한 등식·OFF byte 등식·fail-closed 5·census 서식) · 기존 selftest 26 PASS 유지(grammar-debug 재실행). **봉인 접촉 가 — 진행.**
   ⚠**조용한 기재(base 부채, 이 계약 밖)**: (a) `bf_get_chi` 가 돌려주는 `bf->chi_bf` 격자는 plasma 빌더에서 **ff(Z²·n_ion) 를 이미 합산**하고(`lumina_plasma.c` ~8623), 조립기는 그 위에 **Kramers n_e² ff 를 다시 더한다**(`lumina_cmfgen.c` ~2426) — base 부터 있던 ff 불투명도 이중 계상(량은 bf 대비 10⁻⁹ 급으로 무시 가능하나 부채). 본 계약의 ON 식은 `bf_get_eta` 가 같은 ff(η=χ_ff·B, ~8624)를 담고 있어 **OFF 와 항별 대응이 유지**되며 새 이중 계상은 없다. (b) 같은 이유로 census 의 「bf 불투명 빈(`chi_bf>0`)」은 ff 때문에 사실상 전 빈이 잡힌다 — ff-only 빈의 비는 정확히 1 이라 mean 을 1 쪽으로 희석하지만 가중치(χ·B)가 bf 대비 무시 가능. `n_bf_bins` 는 「bf+ff>0 빈 수」로 읽을 것. CLASSIC_DEBT 대장 기재 대상.
4. **동결·빌드**: `source.sha256` 184 파일(base 182 + selftest 2) 재동결, `source_before.json` 184 항목 재생성, `sha256sum -c` 전건 OK. CPU 빌드 `build4_cpu.slurm` 제출 **job 379038**(`LUMINA_contnlte_build4`, → `cuda_build4/`). 사전감사(Opus 4.8 fresh) 결과 대기 중 — 스테이징·제출은 그 뒤.
5. **사전감사 1회차(Opus 4.8 fresh, `claude-opus-4-8` 자기보고, 216k 토큰·27 tool)**: Q-GOAL 부분부합 · Q-LEAN 비대(결손 방향, 추가 후보 1) · **통과 불가 · «폐합 차단» 2**(B-1 채널 귀속 계측 부재+E-1 상한 부재+pref 물리 상한 / B-2 fail-closed·G-SRC 의 ff-붕괴 무감) · «기재로 족함» C1(판독 C 대수)·C3·C4·C7·절차·E-5. 정지 조건(Q-GOAL «폐합 차단»·L4 삭제 차단) 아님 ⟹ **개정 1** 로 반영(위 표 R1-1~8; 본문 치환은 python 으로 OLD/NEW 쌍 15 건, 행 단위 diff 로 의도 외 소실 0 확인). 빌드 379038 은 트리가 바뀌므로 취소(소유 확인 후). 다음: 게이트 재실행 → 개정 1 구현 발주(Opus 4.8) ∥ 사전감사 2회차(Opus 4.8 fresh, 개정 1 한정) → 코드 간단 감사 → 빌드 5.
6. **판정 도구·스테이징 개정 1 반영(운전자)**: `repro_claude_20260917/judge_contnlte.py` — G-SRC 밴드(0.1–10·min>0)·G-CHAN(최대 gen 150 행·14 열·항등)·E-1a/b/c·분기 CB-C′ (해시 `READOUT_TOOLS.sha256` 추가행). 음성대조 합성 픽스처: 건강 ⟹ G-SRC/G-CHAN PASS · census 0.01 위조 ⟹ G-SRC FAIL · 1.0 위조 ⟹ FAIL · 채널 열 바꿔치기 ⟹ G-CHAN FAIL(항등) · 행 1 결손 ⟹ FAIL(149). base 376251 자기검사: G_SRC/G_CHAN/E1a/E1b/E2/E3 FAIL · E1c/E4/E5 PASS ⟹ CB-X(잣대를 base 에 먼저 적용 — base 는 E-1a 를 넘지 못한다; base r_S sh12/16/20 = 57.0/17.7/8.79). `cont_nlte_20260918/stage_run1.sh` 에 `LUMINA_ION_CHANNEL_DUMP` 키 추가(exports delta 기대 8 행). ff 이중 계상은 `docs/CLASSIC_DEBT_CENSUS.md` §9 에 기재. 구현(Opus 4.8)·사전감사 2회차(Opus 4.8 fresh) 진행 중.
7. **분장 변경(user 09-18 「코드 간단 감사는 fable에게 맡겨」)**: 개정25-a 로 코드 간단 감사 = Fable fresh. 1회차 구현 감사(집행 기록 3)는 운전자 저작으로 남기고, 개정 1 구현 착지분부터 Fable 이 맡는다.
8. **분장 변경(user 09-18 「그리고 단계 계획도 fable이」)**: 개정25-b 로 단계 계획·사전등록 기안(개정 포함) = Fable fresh. 원안·개정 1 은 운전자 기안으로 남기고, 사전감사 2회차를 반영하는 개정부터 Fable 이 기안한다.
9. **사전감사 2회차(Opus 4.8 fresh, `claude-opus-4-8` 자기보고, 146k 토큰)**: **PRE_AUDIT 통과 · BLOCKING 0.** B-2 해소(ff 합산 전 `eta_milne_ok` 가 순-ff 폴백을 잡음, G-SRC 밴드가 미변화·ff 붕괴·램프를 3방향으로 가름) · B-1 해소(D = [Γ_phot 이온화 분율]×[총재결합/RR] 이 항등 1 이 아닌 실제 판별식, G-CHAN 항등은 배선 음성대조로 올바르게 표기) · C1 정정 대수 독립 재계산 전건 일치(S ×1.676, Si ×1.35–1.37, Ca f_II→0) · 「구성상 참」 예측 등록 0 · R1-6 pref 는 «기재로 족함»(pref 1.0 은 E-1a 를 어렵게 함; 코드 주석상 0.05 는 고-x_e 내부 값이라 표적 셸에 부적절 쪽). «기재로 족함» 처분(운전자 사실 확인분):
   - (a) **요율 정정 게이트(`LUMINA_RATES_FIX`)가 lineage exports 에 없다**(376251 OFF). 감사자는 「DR 이중 계상 부재는 U_II 경로 ON 전제」라 했으나 [저자 실행 grep] 결과 U_II 분배함수 정정은 `frozenin_alpha_rr` 에서 **무조건**(분배합 `u≥1` 이면 `U_ion=u`; `lumina_plasma.c` ~6428–6443) 적용되고, 게이트가 바꾸는 것은 F2(상위 이온 준위 0 개일 때의 대체값)·F3(x>700, T ≲ 수백 K 에서만 차이) 뿐 ⟹ S/Si/Ca II→III 의 표적 조건(T_e ~10⁴ K, III 준위 존재)에서 **DR 이중 계상 없음**. 게이트 OFF 유지(exports 변경 없음).
   - (b) 5-파일 트리 기준 게이트 재실행: 개정 1 적용 후 `check_prereg_preflight` rc=0(changeset files=5)·`check_gate_referents` rc=0·`check_doc_canonical` rc=0·`check_dispatch_preflight` rc=0 — 집행 기록 5 직후 실행분(커밋 f2a5e9b 이전 게이트 로그).
   - (c) CB-A 대 CB-C′ 경계: preflight 1-D 분할은 이 경계를 못 본다 — 판정기 `judge_contnlte.py` 가 셸별 D·f_II 를 직접 계산해 E-1b∧E-1c 로 분기하므로 regime 수동 인코딩에 의존하지 않는다(집행 기록 6).
   - (d) E-1b 「불투명 분율」 라벨은 표현 부정확(양 자체는 유효) — 문면 개정은 기안자(개정25-b Fable) 몫으로 남기고 판정문에서 D 의 대수 정의를 병기한다. E-1c 는 pref 1.0 에서 여유 가드 — 기재.
   다음: 개정 1 구현 착지 → 코드 간단 감사(Fable fresh, 개정25-a) → 빌드 5 → 스테이징 → 제출.
10. **분장 변경(user 09-18 「사전 감사는 없앰.」)**: 개정25-c 로 계약 사전감사 폐지. 이 계약은 폐지 전 사전감사 2회를 이미 통과했으므로 소급 없음. ⚠Q-GOAL·Q-LEAN 은 사전감사 발주서 전용 고정질문이라 폐지로 고아가 됐다(재배치는 user 결정) — 이 계약은 사전감사 1·2회차에서 이미 물었다.
11. **개정 1 구현(Opus 4.8, `claude-opus-4-8`)·집행 점검(운전자)·코드 간단 감사(Fable fresh, `claude-fable-5-1`)**:
   - 구현: `lumina.h`(+`eta_milne_ok`)·`env_universe.h`(+`LUMINA_ION_CHANNEL_DUMP`, COUNT 523)·`lumina_plasma.c`(ff 합산 전 `eta_milne_ok` 스캔·`frozenin_alpha_dr` DR-only·채널 덤프·arm 은 커밋 패스만)·`lumina_cmfgen.c`(shell fail-closed 를 `csrc_any_eta`→`eta_milne_ok`)·selftest. selftest2 20/20·직전 30/30 PASS.
   - 운전자 집행 점검: `diff -rq` 6 파일(두 계약 합집합)·env COUNT 523=배열 523(중복 0)·재동결 184 파일 불일치 0.
   - **Fable 코드 감사: Q0 봉인접촉 가 · CODE_AUDIT 통과 · BLOCKING 0.** 게이트 OFF byte-동일(부동소수 경로 불변·`eta_milne_ok` 스캔은 출력 무접촉·배열 수명 정합·candidate publish 이중 free 없음)·fail-closed 정합·DR 분리 정직(`frozenin_alpha_dr` = `frozenin_alpha_rr` 의 DR 항 동일 경로)·채널 항등 구성상 참·arm 커밋 패스만·COUNT 정합 확인.
   - «기재로 족함»(폐합 차단 아님): **F-1** `eta_milne_ok` 가 「Milne 부분」이 아니라 「ff 전 bf 부분(Milne 또는 열폴백 S_l=B)」 — BF_MILNE=2 에서 폴백은 `Cinv_l` 무효 시에만이라 드물고 G-SRC 밴드가 이중 방어. 문면 정정은 기안자(Fable) 몫으로 다음 개정에 이관. **F-2** `gen` 이 커밋 표시 없는 사설 카운터 — run5 로그·제어흐름상 최대 gen=커밋 replay 확인(G-RUN iter=9·G-CHAN 150행이 교차 검증). **F-3** 채널 덤프 `%.9e` 인쇄가 판독기 항등 `1e-9` 재검산에 헛 FAIL(최대 수십 %) — **판독기 허용오차 1e-9→1e-6 으로 수정**(음성대조 재확인: 정상 PASS·열바꿔치기 FAIL·7e-10 섭동 PASS; 도구 해시 갱신). **F-4** `candidate_bf_free` 가 `eta_milne_ok` 미해제(200 B/후보, 무시 가능) → CLASSIC_DEBT §10. **F-5** selftest 「ff-only」 항목이 앞 항목과 같은 호출·CSV 왕복 없음 — 빌드 명령줄 기재(아래). **F-6** 셸 0 rate 미포함 gen 경계는 판독기 >150 FAIL 로 무음 아님.
   - 빌드(F-5 명령줄): CPU 빌드 slurm **379210 COMPLETED 1m11s**(cuda_build4, binary sha 8ceafa36, `source_after` 불일치 0, `make -j1 cuda NVCC=/opt/ohpc/pub/cuda/13.0.2/bin/nvcc`). ⚠379038 은 개정 1 **전** 트리를 빌드해 COMPLETED(내 scancel 이 완료와 경합) — stale 바이너리(6d71be16) 제거 후 재빌드.
   다음: 스테이징(러너 ITER=10 4곳 편집·해시) → 제출.
12. **경량 사전감사 부활(개정25-d)·스테이징·제출(운전자)**:
   - user 가 개정25-c 뒤 Q-GOAL·Q-LEAN 고아를 지적 → **개정25-d**(경량 사전감사 = Opus 4.8 fresh, 이 둘만, 빌드 전, 독립). 이 계약은 폐지 전 사전감사 1·2회차(집행 기록 5·9)에서 이미 물었고, 개정 1 의 채널 덤프 추가가 1회차 Q-LEAN 지적(「비대=측정 결손 방향」)을 메운 것이라 25-d 충족 — 소급 재발주 불요.
   - 스테이징 `stage_run1.sh h200x1`: seal-build(184 파일, binary 8ceafa36), exports delta 8 키(ITER 2→10·FROZENIN_DR 0→1·CONT_NLTE/CHANNEL_DUMP/ION_NT/NT_FULL 추가), 러너 ITER 검사 `== 10`·`--time=30:00:00`·job-name. deck 해시 OK.
   - ⚠**스테이징 위생 결함 2건(수리·기재)**: (a) 러너 sed `outer_iterations=2\n→10\n` 이 sed 의 `\n`=개행 해석으로 조용히 실패 → 푸터 printf 가 10회 런을 `outer_iterations=2` 로 거짓 기재. **러너 사본 132행 직접 수정**(2→10)·runner_sha256 재계산·STAGE_IDENTITY 갱신. 비소비 메타지만 이력 거짓. (b) STAGE_IDENTITY 가 원본(run5)에서 LTEUNMAP 계약·physical_run lineage 줄을 물려받아 두 계약을 가리킴 → stale 2행 제거(CONT 계약·run5 lineage 만 유지). 둘 다 런 제어 무영향(judge·G-RUN 불소비)이나 provenance 정정.
   - **제출: job 379330**(LUMINA_contnlte_r1, h200×1, syntax 로그인 노드 제출, WorkDir OWNER.txt=DO_NOT_CANCEL). ITER 10 ≈ 18–20 h, `--time` 30 h. 헤지 없음(h200 단독).
   다음: 착지 → `judge_contnlte.py` → 판정문 → 종료감사(Opus 4.8 fresh) → 폐합 커밋 → 인수인계·메모리.
