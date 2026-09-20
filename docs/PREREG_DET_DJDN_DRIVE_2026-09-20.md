# 사전등록 — DET-DJDN-DRIVE: 폴백 없는 감쇠 log-뉴턴으로 대각 ∂J/∂N 항을 구동 국면에 실제 적용하면 유한 고정점에 착지하는가, 누락 채널의 크기 s−g 는 얼마인가 (2026-09-20)

기안 = **Fable fresh**(`claude-fable-5-1`, 개정25-b). 상태 = **기안 / 경량 사전감사 대기(개정25-d, Q-GOAL·Q-LEAN 전용)**. 실행 승인이 아니다.
근거 = `docs/VERDICT_DET_DJDN_MVP_2026-09-20.md`(직전 판정: 대각 항 미시험) · `docs/PREREG_DET_DJDN_MVP_2026-09-19.md`(직전 계약, 구조 계승 + §0 결함 정정) · `/gpfs/kjhan/lumina/cmfgen_map/djdn_mvp_20260919/FORK_EVIDENCE.md`(독립 방향 평가 증거) · `docs/READOUT_DAMPING_DISCRIMINANT_2026-09-19.md` · `docs/BRIEF_POPDAMP_A_2026-09-19.md`(user 09-19 「감쇠는 B 뉴턴-스텝 보정 상한으로 흡수」, b737446) · `docs/PREREG_DET_DR_ON_2026-09-19.md`(base 계보). **base = DR-on 판정런 385315 폐합본** 입력 + CONT 소스 ON. **대조군 = DET-DJDN-MVP 런 387175**(`/gpfs/kjhan/lumina/cmfgen_map/djdn_mvp_20260919/run_h100a100`, 같은 base·같은 exports·구 뉴턴 폴백 바이너리 73d8d339). 소스 기준 = 동결 트리 `/gpfs/kjhan/lumina/cmfgen_map/lteunmap_eps_20260917/source`(DJDN-MVP 변경집합 포함 상태, manifest 185 파일·repo 미이식).

## 0. 기안자 검사 — 직전 계약 결함 정정 (운전자·사전감사 필독, 개정25-b 자기 지목)

**(a) §0(b) 대수 정정.** 선형 모델 x·den = num + a(x − x_lag) 의 근 x* = x_lag·(ρ − g)/(1 − g), ρ ≡ x_map/x_lag, g ≡ a/den. 양의 근은 (ρ−g)/(1−g) > 0 일 때 존재한다 — g≥1 이라도 ρ<g 면 있고(대조군 g≥1 12 (epoch,셸) 중 11), g<1 이라도 ρ<g 면 없다. 직전 코드의 `den−a≤0 ⟹ root=0·x_map 커밋`은 이 조건의 반쪽만 본 규칙이었고, 구동 epoch 에서 항을 전부 껐다. **새 설계는 근 존재에 분기하지 않는다**(§2 log-뉴턴 한 스텝 + 보정 상한). 폴백 삭제.
**(b) §2 범주 정정.** 「x_map/x_lag ≥ 90×」의 231 은 CONT run1 sh20 의 r(=x_map) 자체였다. 구동 잣대는 epoch 별 ρ 다. 대조군 실측(§2 표): 구동은 sh24·sh28 의 epoch 4–5 에만 있고(ρ 1.95→2.85·2.82→11.1), sh16·sh20 은 전 epoch ρ<1(단조 재결합) — **CONT 폭주는 외곽 두 셸의 국면**이다.
**(c) §4 G-DRV referent 정정.** census `gen` 은 `compute_ion_populations` 호출(n_e 폐합 하위패스) 카운터다(동결 트리 `lumina_plasma.c:2930`, s==0 랩마다 ++). epoch = `radeq_set_line_re_source`(`lumina_cuda.cu:8563`, 바깥 반복당 1회) 호출 세대. 새 census 는 `epoch` 열을 갖고, G-CPL 이 「epoch 안 x_lag 불변·epoch 경계에서 x_lag^{(k+1)} = 마지막 x_star^{(k)}」를 기계로 검사한다. 정렬 사실 [저자 실행, grammar-debug: 대조군 census 를 x_lag 변화점으로 분할해 각 블록 마지막 행을 취함 · `physics_DET_iter000{0..3}.ion.csv` `Z16_stage_q2_cm3/Z16_stage_q1_cm3`]: epoch 1 = 씨앗 완화(x_lag = Saha 씨앗), **epoch k+2 커밋값 = iter k 의 stage r**(sh12 12.11/17.72/12.79/22.62 · sh16 13.29/10.89/8.525/7.58 · sh20 23.46/18.5/13.09/10.22 · sh24 89.19/60.78/118.6/338.3 · sh28 398.7/283.1/799.6/1.297e4, 전건 일치) ⟹ epoch 수 = ITER+1, 판정 epoch = ITER·ITER+1.
**(d) 러너 postcheck.** 대조군은 `job.slurm:186` `-ne 2` 하드코딩으로 G-RUN 을 잃었다. §4 G-ID·§6 에 「러너 ITER 상수 셋 = 선택 ITER」를 스테이징 게이트로 둔다.
**(e) 대조군의 성격(기재).** 387175 는 순수 OFF 가 아니다 — sh12 는 epoch 3–5 에 구 뉴턴 근이 적용됐고(root=1), sh16/20/24/28 은 epoch 2–5 전부 root=0 이라 Picard 커밋이다. 판정 셸 다섯 중 넷이 Picard 대조군이고 sh12 만 「구 뉴턴」 대조군이다.

## 1. 계약 — 물리 계약 1개

**「라이브 이온화 balance 의 대각 ∂J/∂N 항(DJDN-MVP 대수 그대로: a = Σ_l f_l Σ_b Gw_b λ*_{s,b} ∂S_b/∂x)을, 근 존재 폴백 없이 log 변수 뉴턴 한 스텝 + 보정 상한으로 **모든 epoch·모든 쌍에 실제 적용**하면, CONT 소스 위 황 II→III 비 x 의 epoch 궤적이 판정 셸(Z=16, 셸 12·16·20·24·28)에서 유한 고정점에 착지하는가(DRV-A), 아니면 어떤 꼴로 착지하지 않는가(DRV-B 진동 / DRV-C 단조 상승 / DRV-D 혼재); 그리고 그 국면에서 경험 총미분 s 와 대각 이득 g 의 차 s−g(정규화 κ=(s−g)/(1−g))는 얼마인가.」** 판정런 1회(twin-T_e, binned 챔피언 경로, CONT 소스 ON, DR-on base). 대조 = 387175(같은 exports·구 폴백 바이너리, §0(e)).
⚠**말하지 않는 것**: CMFGEN 도달 · s−g 의 국소(∂J/∂χ)/비국소(셸간) 분할 · 완전 Path B(tri_*·`coupled_newton_solve_all`·tensor core) · N×N 준위 해와의 정합(이중 권위) · Milne 소스 자체의 옳음 · Si/Ca 판정(기재만) · 스펙트럼 · 자유-T_e.

### Q-GOAL (기안자 자기 답 — 경량 사전감사가 독립 재판정)
① 칸: 어긋남 지도 **중간 셸 9–24 S II/III**(CMFGEN f_II .85/.80/.71 대 base .04/.12/.22). 직접 줄이지 않는다 — 이 단은 그 칸을 채우는 본선(CONT 연속체 소스)이 폭주하는 기전의 첫 수리 가설(대각 ∂J/∂N)을 **처음으로 실제 시험**한다. 직전 단은 대수만 세우고(G-CPL) 구동 국면에서는 항이 꺼져 있었다(구동∧적용 = ∅). ② 산출 = 물리 측정(착지 여부 · in-vivo s−g · 셸별 κ). 신규 계측 = census 열 셋(`epoch`·`c_newton`·`damp`)·env 1. ③ 대안: (i) 완전 Path B 직행 — 독립 방향 평가 2건이 「근거 오독」으로 금지했고, 크기 입력(s−g)이 없다. (ii) 런 없이 대조군 s 로 κ 를 계산해 답하기 — §2 표가 그것이며 예상은 「혼재」다. 그러나 대조군 s 는 항이 꺼진 Picard 과도의 시컨트라 부호까지 요동한다(sh16: −0.69, +0.99, +0.40) — **항을 전 셸에 건 상태의 s 는 런만 준다**. 이 점이 이 단의 유일한 물리 산출이자 Path B 설계의 크기 입력이다. ④ 건너뛰면 Path B 를 크기 미지·형태 미지로 짓거나, 대각만으로 되는 경우(폭주 셸 sh24 의 κ=−0.59)를 놓친다.

### Q-LEAN (기안자 자기 답)
L1 물리량 = (i) 판정 셸 커밋 x 의 epoch 궤적 Δ_k(착지) (ii) s_k·g_k·κ_k. 소비: `epoch` → Δ_k·s_k 의 키(G-CPL·G-DRV·E-1·E-2) / `c_newton` → G-CPL 항등·E-3 / `damp` → 착지 정의·E-3. 소비되지 않는 열: 기존 `gamma_eff`·`maxch_factor`(직전 단 계측, 코드 무접촉으로 남김 — L4 후보). L2 게이트 넷: G-ID(계보·소스·러너 상수) · G-RUN(완주, postcheck ITER 정합) · G-CPL(항이 실제로·옳게 적용됐는가 — 아니면 A 가 공허) · G-DRV(구동 epoch 에 적용됐고 대조군과 갈렸는가 — 아니면 미시험 재발). 중복 없음. L3 = 2 걸음(구현·selftest → 런 1). L4 삭제·통합 후보: `LUMINA_POP_MAXCH` 노브 + `maxch_factor` 열 — 감쇠가 뉴턴 보정 상한으로 흡수됐으므로(user b737446) 죽은 노브다. **삭제하지 않는다**(삭제도 src 편집이며 이 단의 변경집합을 넓힌다) — 판정런 미설정·열 상수 1·판정 미소비로 기재. 판정 셸을 넷→다섯(sh28 추가)으로 늘린 것은 대조군에서 폭주가 가장 큰 셸(epoch 5 ρ=11.1)을 판정 밖에 두면 「폭주 제어」를 재지 못하기 때문이다(계측 증가 0 — 같은 파일의 행).

## 2. 기전 — 런 0회 확정분

- **대조군 epoch 구조 [저자 실행]**: census 123 gen = 5 epoch × (26/27/25/23/22) 하위패스. 아래 표는 각 epoch 마지막 행(Z=16). ρ=x_map/x_lag, s_k=(x_map^{(k)}−x_map^{(k−1)})/(x_lag^{(k)}−x_lag^{(k−1)}), κ_k=(s_k−g_k)/(1−g_k).

| 셸 | epoch | x_lag | x_map | ρ | g | root | s | κ |
|---|---|---|---|---|---|---|---|---|
| 12 | 2 / 3 / 4 / 5 | 5.20 / 12.11 / 17.72 / 12.79 | 12.11 / 14.16 / 14.80 / 15.02 | 2.33 / 1.17 / 0.84 / 1.18 | 3.19 / 0.64 / 0.41 / 0.77 | 0 / 1 / 1 / 1 | — / +0.30 / +0.12 / −0.04 | — / −0.93 / −0.50 / −3.60 |
| 16 | 2 / 3 / 4 / 5 | 9.81 / 13.29 / 10.89 / 8.53 | 13.29 / 10.89 / 8.53 / 7.58 | 1.35 / 0.82 / 0.78 / 0.89 | 3.68 / 2.05 / 1.45 / 1.42 | 0 / 0 / 0 / 0 | — / −0.69 / +0.99 / +0.40 | — / +2.61 / +1.03 / +2.42 |
| 20 | 2 / 3 / 4 / 5 | 29.1 / 23.5 / 18.5 / 13.1 | 23.5 / 18.5 / 13.1 / 10.2 | 0.81 / 0.79 / 0.71 / 0.78 | 2.08 / 3.62 / 2.68 / 2.81 | 0 / 0 / 0 / 0 | — / +0.88 / +1.09 / +0.53 | — / +1.05 / +0.95 / +1.26 |
| 24 | 2 / 3 / 4 / 5 | 122.8 / 89.2 / 60.8 / 118.6 | 89.2 / 60.8 / 118.6 / 338.3 | 0.73 / 0.68 / **1.95** / **2.85** | 0.77 / 1.82 / 12.2 / 2.76 | 0 / 0 / 0 / 0 | — / +0.84 / −2.04 / +3.80 | — / +1.19 / +1.27 / **−0.59** |
| 28 | 2 / 3 / 4 / 5 | 327 / 399 / 283 / 800 | 370 / 325 / 800 / 8893 | 1.13 / 0.81 / **2.82** / **11.1** | 0.41 / 0.36 / 4.16 / 0.34 | 1 / 1 / 0 / 1 | — / −0.63 / −4.11 / +15.7 | — / −1.55 / +2.62 / +23 |

  ⟹ 구동(ρ≥1.3)은 **sh24·sh28 의 epoch 4·5**에 실재하고 그 행은 전부 root=0(항 OFF) — 직전 판정의 「구동∧적용 = ∅」의 수치. sh12(항 ON, g<1)의 κ<0·|κ|>1 이 판정문의 「뉴턴이 Picard 보다 진동↑」의 대수다.
- **대각 뉴턴의 증폭률(새 대수, 이 계약의 잣대)**: 진짜 맵 x_map = M(x_lag) 의 고정점 x° 근방에서 M ≈ x° + s(x−x°)(s = 총미분). 선형-x 뉴턴 갱신 x_{k+1} = x_k + (x_map−x_k)/(1−g) 의 편차 δ=x−x° 는 **δ_{k+1} = κ δ_k, κ = (s−g)/(1−g)**. g=s 면 한 스텝 수렴(진짜 뉴턴); g=0 은 Picard(κ=s). **s<1<g 이면 κ=(g−s)/(g−1)>1 — 항이 Picard-수렴 셸을 단조 발산시킨다**; g>1, 1<s<2g−1 이면 |κ|<1 수렴; g<1, s<g 이면 κ<0 — 부호 교대, |s−g|>1−g 면 발산. log 변수 스텝도 ρ→1 에서 같은 1차식으로 환원되므로 κ 는 변수 선택과 무관하다. ⟹ **착지 여부 = |κ|<1 이고, κ 는 (s−g) 의 함수** — 이 단의 두 물음(착지·s−g)은 같은 양의 두 얼굴이다. 기안자 예상(판정 조건 아님·확률 미부여): 대조군 s 를 대입하면 sh16/20 은 κ>1(발산), sh24 epoch 5 는 −0.59(수축), sh12 는 진동 — **D(혼재)가 대수적으로 유리**. s 가 과도 시컨트라(sh16 부호 요동) 이 예상은 강건하지 않다 — 런이 정한다.
- **변수·스텝(수리안의 대수)**: u = ln x. 잔차 F(u) = u − ln[(num + a(e^u − x_lag))/den]. F(u_lag) = −ln ρ, F'(u_lag) = 1 − g/ρ. 뉴턴 한 스텝 Δu = ρ ln ρ/(ρ−g) = ln ρ + c, **보정 c = ln ρ · g/(ρ−g)**(a=0 ⟹ g=0 ⟹ c=0 정확). 감쇠: c_d = sign(c)·min(|c|, C_max), damp = c_d/c(c=0 이면 1). 커밋 **x_star = x_map · exp(c_d)** — 어떤 유한 입력에서도 양·유한, 근 존재 분기 없음, Picard 몫(ln ρ)은 감쇠하지 않음. 1차: Δu ≈ (ρ−1)/(1−g)(선형-x 뉴턴과 일치). 특이점 ρ=g(F'=0): c→±∞ ⟹ damp→0, x_star = x_map·e^{±C_max}; ρ−g 가 정확히 0 이면 c_d=0·damp=0 기록(측도 0, 판정 행에서 발생 시 기재). x_lag=0 또는 x_map=0 이면 ln 불능 ⟹ root=0(입력 부재, OFF 값·기록) — 판정 셸 S 는 x ≥ 5 라 해당 없음.
  ⚠f_II 변수를 택하지 않는 이유: f∈(0,1) 은 유계라 뉴턴 스텝이 범위를 나가면 클립(상태 클램프)이 필요하다; logit f = ln x 이므로 log 변수가 그 의도의 클램프 없는 형태다.
  ⚠log-뉴턴 한 스텝이 선형-x 근과 다른 점(기재): 근이 존재할 때 두 값은 O(Δu²)로 다르고, 근이 없을 때(ρ>g>1 등) 잔차-log 형은 Picard 방향으로 움직인다(sh24 epoch 5: c=+32 → 상한). 어느 쪽도 「옳은 근」이 아니다 — 선형 모델에 근이 없는 국면이며, 그 크기가 E-2 에 남는다.
- **x_lag 참조점·cadence**: DJDN-MVP 2회차 수리 그대로(`ion_before` 스냅샷, pass당 커밋 1회, Fable+grok 이중 확인). 변경 없음.
- **비용**: 387175 실측 ITER 4 = 9.8 h(a100×2, cpus 16) ⟹ ≈2.2 h/반복 + 씨앗.
- **미확정(계약 밖)**: s−g 의 국소/비국소 분할 · 판정 셸 밖(내핵 IGE·Si·Ca)의 응답 · 스칼라 선형화와 N×N 해의 불일치.

## 3. 수리안 — 변경집합

env `LUMINA_POP_DJDN`(기존, 기본 0 = 항 미적용·census 없음·**byte-동일**) 의 ON 동작을 §2 log-뉴턴으로 **교체**한다(모드 노브 신설 없음 — 구 폴백 동작은 대조군 바이너리에 동결). env `LUMINA_POP_DJDN_CMAX`(신설, 보정 상한 C_max; 미설정 = 1.0). census `lumina_pop_djdn.csv` 열 = 기존 13 + `epoch`(J/λ* 조립 세대: `radeq_set_line_re_source` 호출 카운터) + `c_newton`(감쇠 전 c) + `damp`; `g` 열 정의를 a/den 으로(기존 a/(n_eα) 와 c_rec 차이뿐, 항등 정확성 위해). `root` 의미 = 항 입력 실재·적용(1) / 입력 부재(0) — 폴백 상태는 없다. `gamma_eff`·`maxch_factor`·MAXCH 노브는 무접촉.

### 변경집합 — DJDN-DRIVE (표)

| 경로 | 담당 | 내용 |
|---|---|---|
| `src/env_universe.h` | Opus 4.8 | `LUMINA_POP_DJDN_CMAX` 등록 |
| `src/lumina_plasma.c` | Opus 4.8 | `pop_djdn_newton_root`(den−a≤0 폴백) 를 `pop_djdn_log_newton`(§2: ρ·g→c·c_d·damp·x_star=x_map·exp(c_d)) 으로 교체; g=a/den; census 헤더·행 포맷에 `epoch,c_newton,damp` 추가; `radeq_set_line_re_source` 에 epoch 카운터 ++; 판정런 OFF byte-동일 유지; 상태 클램프·floor 금지 |
| `tests/pop_djdn_selftest.c` | Opus 4.8 | 기존 21 케이스 유지(a 독립 재계산·x_lag 회귀·OFF bit-동일) + **c 독립 재계산**(1e-12 rel) · **직전 폴백 픽스처 셋(g>1∧ρ>g / g<1∧ρ<g / ρ=g 특이)** ⟹ x_star 유한·양, 특이 픽스처 damp<1 · **λ*≡0(a=0) ⟹ x_star == x_map bit-동일**(음성대조 1, 존속) · **ρ=1 ⟹ c=0 ⟹ x_star == x_map == x_lag bit-동일**(고정점 불변) · \|c\|≤C_max ⟹ damp==1 이고 x_star 가 무감쇠와 bit-동일 · \|c\|>C_max ⟹ x_star = x_map·exp(±C_max) 1e-12 rel · **음성대조 2**: c 를 1e-6 교란 ⟹ G-CPL 항등(§4) FAIL 시연 · 행 포맷 16 열 헤더 일치 |

### 변경집합 끝

★**src 편집 1개 규칙**: 가동 중 src-편집 태스크 0 확인 후 발주(DJDN-MVP 폐합 c38df3e). ⚠프리플라이트: 표의 세 파일은 repo 에 심볼이 없으므로(변경집합은 동결 트리 복사본에만) `expected_extra` 로 선언 — 직전 두 계약과 동일 처리, 기대 출력 `changeset-consistent files=3`.

## 4. 게이트 명부 (표)

| ID | 조건 | 음성대조 |
|---|---|---|
| G-ID | 새 런 exports 를 대조군 387175 계보(`/gpfs/kjhan/lumina/cmfgen_map/djdn_mvp_20260919/run_h100a100/input/resolved_lumina.exports`)와 diff — 추가·변경 키가 정확히 `LUMINA_POP_DJDN_CMAX`(=§6 ① 값) [§6 ② ITER 변경 시 `LUMINA_PURE_CMFGEN_ITER`; 프로필 종속 `LUMINA_CMF_FINE_MGPU_DEVICES` 허용]; `LUMINA_CMF_CONT_NLTE_SOURCE`(=1)·`LUMINA_POP_DJDN`(=1)·`LUMINA_ION_CHANNEL_DUMP`(=1)·`LUMINA_FROZENIN_DR`(=1)·`LUMINA_DR_BOOST_AUTOSTRUCT`·`LUMINA_DR_BOOST_BADNELL`·`LUMINA_DR_BOOST_MAZZOTTA`·`LUMINA_DR_BOOST_NORAD`(각 =1) 유지; `LUMINA_POP_MAXCH`·`LUMINA_ION_NT`·`LUMINA_NT_FULL`·`LUMINA_RADEQ_SIMUL`·`LUMINA_SIMUL_ION_DAMP` **부재**; 동결 트리 `…/lteunmap_eps_20260917/source.sha256` 의 `sha256sum -c` non-OK 가 정확히 §3 표의 파일; 새 바이너리 해시 `input/binary.sha256` 기재(대조군 73d8d339 와 달라야 함); 러너의 **ITER 상수 셋** — guard `"${LUMINA_PURE_CMFGEN_ITER:-}" == N` · `outer_iterations=N` · postcheck `-ne N` — 이 전부 선택 ITER 와 같음을 `grep -nE 'PURE_CMFGEN_ITER:-\}" == \|outer_iterations=\|-ne [0-9]+ \]\]; then' job.slurm` 출력으로 STAGE_IDENTITY 에 기재; 러너 편집 = job-name·`--time`·ITER 상수 셋·프로필(partition/gres/GPU_PROFILE/cpus, §6 ③)뿐, `runner_edits` 에 `postcheck_te_count` 포함 | 다른 키 혼입·§3 밖 소스 변조·postcheck `-ne` 가 ITER 와 불일치 ⟹ FAIL(대조군 러너 그대로 = FAIL) |
| G-RUN | `RUN_FOOTER.txt` `terminal_state=SUCCESS`·`model_rc=0`·`outer_iterations`=선택 ITER; `work/lumina_pop_djdn.csv`·`work/lumina_ion_channels.csv`·`work/physics_comparison/physics_DET_iter<ITER−1>.ion.csv` 실재 | 미완주·census 결손 ⟹ FAIL; 대조군 = `POSTCHECK_FAILED_FIXED_TE_EVIDENCE` ⟹ FAIL |
| G-CPL | (오프라인) `tests/pop_djdn_selftest.c` rc=0, §3 음성대조 둘 다 FAIL 시연 — **봉인 접촉 전 Fable 코드 감사 선행**(개정16 ②); (in-vivo) census 헤더 = 기존 13 열 + `epoch,c_newton,damp` 순서 일치; 항등 검사 행 집합 = root=1 ∧ x_lag>0 ∧ x_map>0 ∧ **damp>0** 인 전 행: ρ=x_map/x_lag, c' = ln ρ·g/(ρ−g) 로 재계산해 \|c_newton − c'\| ≤ 1e-9·max(1,\|c'\|) **and** \|x_star − x_map·exp(damp·c_newton)\| ≤ 1e-9·x_star **and** damp = min(1, C_max/\|c_newton\|)(1e-9); **damp==0 행(§2 특이 국면: ρ−g 정확히 0 ⟹ c_d=0, 또는 c 계산 overflow ⟹ c_d=±C_max)은 c' 가 정의되지 않으므로 항등 검사에서 제외**하고, 그 행 수와 (Z,shell,epoch) 를 판정문에 기재(E-3 의 damp<1 집계에 포함)·판정 셸 마지막 두 epoch 에 있으면 E-1 정의로 착지 불가 — 예외는 항등만이며 root=1·a>0·epoch 구조 검사는 그 행에도 적용; epoch 구조: 같은 (Z,shell,epoch) 안 x_lag 불변, epoch 단조 증가, 각 판정 셸에서 epoch k+1 첫 행의 x_lag = epoch k 마지막 행의 x_star(1e-9 rel), 최대 epoch = ITER+1; 행 구조: census 행은 gen 단위로 기록되고 epoch 은 gen 의 묶음이다(§0(c)) — gen 당 행수가 전 gen 에 걸쳐 동일하고(대조군 150 = 3 Z × 50 셸, 기재값이지 요구값 아님), 최대 epoch 안에 gen ≥ 1, 최대 epoch 의 각 gen 에 판정 셸 다섯(Z=16, 셸 12·16·20·24·28) 행이 각각 정확히 1개 실재, 최대 epoch 의 행수 = (그 epoch 의 gen 수) × (gen 당 행수); 판정 셸(Z=16, 셸 12·16·20·24·28) epoch ≥ 2 전 행 root=1·a>0 | census 열 바꿔치기·c 교란 ⟹ 항등 FAIL; λ* 미전달 빌드 ⟹ a≡0 ⟹ 판정 셸 a>0 FAIL; 대조군 census(13 열·`epoch` 없음) ⟹ FAIL |
| G-DRV | **(i) 구동 실재 — 대조군 사실 [저자 실행]**: 387175 census 를 epoch 으로 분할하면 Z=16 sh24 epoch 4·5 와 sh28 epoch 4·5 의 ρ 가 1.95·2.85·2.82·11.1(≥1.3), 전부 root=0 — 구동 국면은 이 base·CONT 소스에서 실재하며 이 단에서 다시 재지 않는다; **(ii) 구동 epoch 적용**: 새 런 census 판정 셸의 epoch 2 … ITER+1 전 행 root=1 이고 `damp` 기록(G-CPL 과 같은 행 집합, 여기서는 「구동 epoch」 키로 재확인); **(iii) 대조군과의 분리**: 새 런 커밋 x^{(k)}(epoch k 마지막 x_star)와 대조군 커밋 x_ctrl^{(k)} 가 sh24 또는 sh28 의 epoch 4 또는 5 중 어느 하나에서 \|log10 x^{(k)} − log10 x_ctrl^{(k)}\| > 0.05 | 대조군 자기검사 ⟹ (ii) root=0·(iii) 차 0 ⟹ FAIL(미시험 재발 검출); 항이 배선만 되고 c≡0 인 빌드 ⟹ (iii) FAIL |

### 게이트 명부 끝

## 5. 기대치 — 런에서만 얻는 수 (잣대를 대조군에 먼저 적용: 대조군은 E-1 창을 넘지 못한다 — sh24 Δ_5 = +0.455, sh28 +1.21)

x^{(k)} = 판정 셸 epoch k 마지막 행 x_star(= epoch k+1 x_lag), k = 1 … K, K = ITER+1. Δ_k = log10 x^{(k)} − log10 x^{(k−1)}, k = 3 … K(epoch 1→2 는 씨앗 완화라 제외).

| ID | 양 | 대조(387175) | **기대·역할** |
|---|---|---|---|
| E-1 착지 | 판정 셸 다섯의 Δ_{K−1}·Δ_K 와 그 두 epoch 의 damp·root | Δ_4/Δ_5: sh12 −0.14/+0.25 · sh16 −0.11/−0.05 · sh20 −0.15/−0.11 · sh24 +0.29/+0.46 · sh28 +0.45/+1.21 | **판정량**(분기 표). 착지 = \|Δ_{K−1}\|≤0.05 ∧ \|Δ_K\|≤0.05 ∧ 그 두 epoch damp=1 ∧ root=1 |
| E-2 s−g | 판정 셸의 s_k·g_k·κ_k(§2 정의, k = 3 … K; x_lag 불변이면 s 미정의·\|1−g\|<1e-3 이면 κ 미정의 → NaN 기재) | §2 표 | 판정 조건 아님 — **기재 + 다음 계약 입력**. 정합 기재: 착지 셸 \|κ_K\|<1 / 진동 셸 κ_K<−1(s<g, 누락 채널 복원성) / 단조 상승 셸 κ_K>1(s>g, 누락 채널 증폭성). 정합 위반은 «기재»(선형 모델의 사정거리 밖) |
| E-3 감쇠 사용 | census epoch ≥ 2 전 행(Z 14·16·20) 중 damp<1 행 수와 그 (Z,shell,epoch); 판정 셸 K−1·K 의 damp | 해당 없음(구 바이너리) | 기재. 판정 셸 마지막 두 epoch 에 damp<1 이면 그 셸은 착지 불가(정의) |
| (기재) | `Z16_nlte_q2/q1`(NLTE 권위 r, 판정 셸·전 반복) · Si 셸 12·20 · Ca f_II · 외곽 S f_III 셸 28·32·36 · Fe f_IV 셸 0–3 · 대조군 대비 같은 epoch 의 log10 x 차 | 대조군·385315 | 판정 조건 아님 |

문턱 근거: 수렴 창 0.05 dex = 직전 계약과 동일(CONT 1 반복 이동 ≥0.13 의 절반 이하, 대조군 sh16 Δ_5 −0.05 가 경계 — 대조군은 sh16 만 창 안이고 다른 넷은 밖이라 A 가 아니다); 상승 하한 0.10 = 대조군 sh24/28 의 최소 상승(0.29) 아래·창 위; 대조군 분리 0.05 dex = 같은 창(FP 미세변동 — cpus 수 등 — 은 1e-3 dex 급).

### 분기 (표)

셸별 코드: 게이트 미충족 ⟹ 전 셸 0. 게이트 충족 시 판정 셸마다 — **1 착지**(E-1 정의) / **2 진동**(착지 아님 ∧ Δ_{K−1}·Δ_K < 0 ∧ min(\|Δ_{K−1}\|,\|Δ_K\|) > 0.05 ∧ \|Δ_K\| ≥ \|Δ_{K−1}\|) / **3 단조 상승**(착지 아님 ∧ Δ_{K−1} > 0.10 ∧ Δ_K > 0.10) / **4 기타**(감쇠 상한 걸린 마지막 epoch·감쇠 진동·단조 하강·root=0 등).

| 분기 | 조건 | 의미 |
|---|---|---|
| DRV-X | G-ID/G-RUN/G-CPL/G-DRV 중 하나라도 미충족 | 미시연 |
| DRV-A | 게이트 전건 **and** 판정 셸 다섯 전부 코드 1 | **대각 항이 구동 국면에서 유한 고정점에 착지** — 대각 충분(관측 등급). \|κ_K\|<1 정합·수렴값의 CMFGEN 근접은 기재 |
| DRV-B | 게이트 전건 **and** 코드 2 셸 ≥ 1 **and** 코드 3·4 셸 0 | **진동 발산** — s<g: 대각 이득이 총미분을 과대, 누락 채널은 **복원성**·크기 ≈ g−s(다음 계약 입력) |
| DRV-C | 게이트 전건 **and** 코드 3 셸 ≥ 1 **and** 코드 2·4 셸 0 | **단조 상승 지속** — s>g: 누락 채널은 **증폭성**, 대각으로는 잡히지 않는 구동(다음 계약 입력) |
| DRV-D | 게이트 전건 **and** (코드 4 셸 ≥ 1 **or** 코드 2·3 셸이 함께 있음) | 혼재 — 판정자가 셸별 κ·damp 로 명명(예: 폭주 셸 착지·수축 셸 발산) |

### 분기 끝

재분기 금지. 판정기 = `judge_djdn.py`(대조군 디렉토리, 운전자 도구)를 **epoch 키·16 열·§4 항등·§5 코드**로 개정한 것(해시 `READOUT_TOOLS.sha256` 기재; 소스 아님). 음성대조: 대조군 387175 자기검사 ⟹ G-RUN·G-CPL(열 부재)·G-DRV(ii)(iii) FAIL ⟹ DRV-X; DR-on 385315 ⟹ census 없음 ⟹ DRV-X.

## 6. 자원 · 실행 · **결정 항목(운전자·user)**

새 RUN 루트 `/gpfs/kjhan/lumina/cmfgen_map/djdn_drive_20260920/run1`(형제 루트), 입력 = 387175 입력 복사 + 새 빌드 + exports 편집(G-ID). GPU 1 런. 봉인 base·대조군 읽기만.

① **C_max** — 기본 **1.0**(보정 인자 e). 근거: 대조군 (epoch,판정 셸) 에서 무감쇠 \|c\| 는 ρ≈g 특이 근방(sh24 epoch 5: +32)을 빼면 ≤0.8 이라 1.0 은 특이 국면만 건드린다. 0.5 를 택하면 sh24 epoch 4(−0.80) 도 걸린다(그 epoch 착지 불가로 분류) — 값은 판정문에 기재. 기안자는 못 박지 않는다.
② **ITER** — 4(≈10 h, `--time=14:00:00`, 판정 epoch 4·5) 또는 **6**(≈15 h, `--time=20:00:00`, 판정 epoch 6·7). 대조군에서 구동이 epoch 4 에야 나타났으므로(씨앗 과도 epoch 1–3) ITER 4 는 판정 창이 구동 개시와 겹친다 — 6 이 착지/미착지 판별에 여유를 준다. 기안자는 못 박지 않는다.
③ 프로필·헤지 — 직전 단과 동일(먼저 시작하는 쪽만; 멀티파티션·cpus 16 편집은 G-ID 에 기재). GPU-bound·byte-배터리 없음.
④ NT 키 부재·MAXCH 미설정(base 그대로).

순서: 경량 사전감사(Opus 4.8 fresh, Q-GOAL·Q-LEAN — 특히 §0 「남는 위험 ①」을 물을 것) → 게이트 4종(prereg-preflight `files=3`·gate_referents·doc_canonical·dispatch-preflight) → Opus 4.8 구현(동결 트리 복사본) → selftest → Fable 코드 감사(첫 항목 = 봉인 접촉 가부) → 재봉인·빌드 → 스테이징(**러너 ITER 상수 셋 grep 확인**) → 제출 → 판독 → 판정(운전자) → 종료감사(Opus 4.8 fresh; GPU 지출·물리 주장이 걸린 단이라 생략하지 않음).

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

말할 수 있는 것: 「폴백 없는 감쇠 log-뉴턴으로 대각 S-채널 ∂J/∂N 항을 구동 국면에 실제 적용했을 때 황 II/III 비가 착지하는가와 그 꼴」, 「그 국면의 in-vivo s−g·κ(누락 채널의 크기·부호)」. 착지해도 CMFGEN 근접은 기재 사항이지 성공 조건이 아니다. **s−g 의 국소/비국소 분할은 말하지 않는다** — s 는 셸간 과도가 섞인 총미분 시컨트이며, 분할은 다음 계약(단일 셸 교란 재생 등)의 몫이다. 「DJDN 토글 단독」은 OFF 런이 없으므로 selftest byte-동일 + Fable 코드 감사의 오프라인 주장 조건부; 대조군은 sh12 에 구 뉴턴이 걸린 준-OFF(§0(e)). 말하지 않는 것: ∂J/∂χ 채널·셸간 항·N×N 해와의 정합·정확 솔버 Λ*·Milne 소스의 인구 역전 빈·Si/Ca·스펙트럼·자유-T_e. twin-T_e 핀 조건부.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 — DJDN-DRIVE (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "(?:src|tests)/[A-Za-z0-9_/.]+\\.(?:c|h)",
    "symbol": "LUMINA_POP_DJDN_CMAX",
    "roots": ["src", "tests"],
    "expected_extra": ["src/env_universe.h", "src/lumina_plasma.c", "tests/pop_djdn_selftest.c"]
  },
  "branches": {
    "regimes": [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
    "metrics": {"count": "len(v)", "gate_fail": "sum(1 for x in v if x < 0.5)", "landed": "sum(1 for x in v if 0.5 <= x < 1.5)", "osc": "sum(1 for x in v if 1.5 <= x < 2.5)", "runaway": "sum(1 for x in v if 2.5 <= x < 3.5)", "other": "sum(1 for x in v if x >= 3.5)"},
    "rules": [
      {"name": "DRV-X", "predicate": "count == 0 or gate_fail > 0"},
      {"name": "DRV-A", "predicate": "count > 0 and gate_fail == 0 and osc == 0 and runaway == 0 and other == 0"},
      {"name": "DRV-B", "predicate": "count > 0 and gate_fail == 0 and osc > 0 and runaway == 0 and other == 0"},
      {"name": "DRV-C", "predicate": "count > 0 and gate_fail == 0 and runaway > 0 and osc == 0 and other == 0"},
      {"name": "DRV-D", "predicate": "count > 0 and gate_fail == 0 and (other > 0 or (osc > 0 and runaway > 0))"}
    ],
    "adversarial_fixtures": [
      {"name": "empty", "mix": []},
      {"name": "gate-fail", "mix": [[0.0, 1], [1.0, 2]]},
      {"name": "landed", "mix": [[1.0, 5]]},
      {"name": "osc", "mix": [[1.0, 2], [2.0, 3]]},
      {"name": "runaway", "mix": [[1.0, 3], [3.0, 2]]},
      {"name": "osc-and-runaway", "mix": [[2.0, 2], [3.0, 2]]},
      {"name": "other", "mix": [[1.0, 4], [4.0, 1]]}
    ],
    "residual": "DRV-X"
  },
  "references": [
    {"path": "docs/VERDICT_DET_DJDN_MVP_2026-09-20.md"},
    {"path": "docs/PREREG_DET_DJDN_MVP_2026-09-19.md"},
    {"path": "docs/PREREG_DET_DR_ON_2026-09-19.md"},
    {"path": "docs/READOUT_DAMPING_DISCRIMINANT_2026-09-19.md"},
    {"path": "docs/BRIEF_POPDAMP_A_2026-09-19.md"},
    {"path": "CLAUDE.md"},
    {"path": "src/lumina_plasma.c", "flags_existing": ["parity_rate_se_ratio_checked", "compute_ion_populations_shell", "radeq_set_line_re_source", "g_lre_lambda_star", "nlte_population_candidate_commit_bundle"], "flags_planned": ["LUMINA_POP_DJDN_CMAX", "pop_djdn_log_newton"]},
    {"path": "src/env_universe.h", "flags_existing": ["LUMINA_SIMUL_ION_DAMP", "LUMINA_CMFGEN_JDUMP"], "flags_planned": ["LUMINA_POP_DJDN_CMAX"]}
  ]
}
```

## 9. 집행 기록

1. **게이트 4종(발주 전)**: prereg-preflight rc=0(`changeset-consistent files=3`·branch-partition rules=5 fixtures=7 residual=DRV-X·references-resolved refs=8)·gate_referents rc=0(changeset_exempt=3)·doc_canonical rc=0(계약+인용5). 커밋 94a11e0. (dispatch-preflight 는 구현 발주 시.)
2. **경량 사전감사(Opus 4.8 fresh, `claude-opus-4-8`, 개정25-d): PREAUDIT 통과·BLOCKING 0.** 감사자가 대조군 census 로 §2 κ 표 재현. **Q-GOAL ③ 핵심 해소**: 런은 오프라인 표로 대체 불가(대조군 구동 epoch 항 OFF·오프라인 리플레이는 M(x_lag) 새점 재평가 불가·κ 외삽 비강건) ⟹ 런이 실측 착지+적용국면 s−g 를 더함(계측 위한 계측 아님). Q-GOAL 부합·Q-LEAN 통과(신규 계측 전건 소비·게이트 신규 종 0·G-DRV(ii) 경미중복은 (i)(iii)가 지탱·죽은 MAXCH 기재처분 정당·sh28/5셸 정당). L4 차단 후보 0. 경미 관측(폐합 조건 아님): G-DRV(ii)↔G-CPL root=1 행집합 중복(축약 가능·비용 무관).
3. **§6 결정(운전자·user)**: ①C_max=**1.0**(기본, 특이 국면만) ②**ITER=6**(user 2026-09-20 선택 — 판정 epoch 6·7=iter 4·5, 구동 발달 후 착지 판별 여유; `--time=20:00:00`) ③프로필·헤지=직전 단 동일(먼저 시작하는 쪽·멀티파티션·cpus 16, G-ID 기재) ④NT 부재·MAXCH 미설정. DR-on 385315·DJDN-MVP 387175 폐합 확인 ⟹ src-편집 가동 0, 구현 발주 가능.
4. **구현(Opus 4.8, `claude-opus-4-8`) + selftest**: `pop_djdn_newton_root`(den−a≤0 폴백)→`pop_djdn_log_newton`(§2: c=lnρ·g/(ρ−g)·c_d=sign min(|c|,C_max)·x_star=x_map·exp(c_d))·g=a/den·census 16열(epoch/c_newton/damp)·epoch=radeq_set_line_re_source ++·env LUMINA_POP_DJDN_CMAX(COUNT 526)·폴백 삭제·OFF byte-동일·클램프 없음. selftest 30/30 PASS + 주입 FAIL 2종. 3파일·잡편집 0.
5. **Fable 코드 간단 감사(`claude-fable-5-1`, 개정25-a): CODE_AUDIT 통과·SEAL_CONTACT 가·BLOCKING 1(계약 문면, 코드 0).** log-뉴턴 대수 FD 독립 검증(F'=1−g/ρ·Δu 6자리 일치)·폴백 삭제 확인(구 함수 참조 0)·OFF byte-동일 구조·클램프 부재(보정 c만 상한)·epoch 카운터 pass당 1회(cmfgen_run 이 레거시 루프 우회, 호출 2곳 중 생산은 7999)·g=a/den 정합·selftest §3 전건 커버 미좁힘. **«폐합 차단» 1 = §4 G-CPL 「최대 epoch 행수 150」 오류**(실제 gen×150·마지막 epoch ~22gen→3300행; 문면대로면 G-CPL FAIL→DRV-X) — 코드 무관, **제출 전 사전등록 OLD/NEW 개정 필수**(①행수 ②damp==0 측도-0 행 c'=∞ 예외). «기재»: c 독립재계산=전사검사(대수 독립은 §2 FD 가 세움, 판정문 표기)·MVP g=a/(n_eα) 와 인자차(미소)·overflow c_newton 덮음(미도달). ⟹ 빌드·재봉인 진행 가·제출 전 계약 개정.
5-b. **§4 G-CPL 개정 적용(Fable OLD/NEW 반환분, 운전자 파일화)**: ①「행수 150」→ 실제 구조(행=gen×(Z,셸)·최대 epoch 행수=gen수×gen당행수·판정 셸 5행 실재) ②항등 검사를 damp>0 행에 한정·damp==0 측도-0 행(denom==0/overflow) 예외·E-3 집계·E-1 착지불가로 처분. 행 diff 1줄·게이트 재통과(prereg-preflight 4 PASS·referents·canonical rc=0). ⚠**Fable 토큰 소진(user 2026-09-20) — 이후 계약 기안·개정·코드 간단 감사·갈림길 평가는 grok 이 대행.**
6. 재봉인·빌드·스테이징(러너 ITER 상수 셋 grep 출력 기재)·제출: (미집행)
7. 착지·판독·판정(운전자): (미집행)
8. 종료감사(Opus 4.8 fresh): (미집행)
