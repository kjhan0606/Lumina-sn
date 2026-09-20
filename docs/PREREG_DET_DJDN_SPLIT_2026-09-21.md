# 사전등록 — DET-DJDN-SPLIT: 이웃 고정 재평가로 s−g 의 국소 몫과 비국소 몫을 가르면, 다음 수리는 국소 ∂J/∂χ 인가 셸간 tri_* 인가 (2026-09-21)

기안 = **Fable 대행**(`grok-4.6`, user 2026-09-20 Fable 토큰 소진 이후 계약 기안 대행). 상태 = **기안 / 경량 사전감사 대기(개정25-d, Q-GOAL·Q-LEAN 전용)**. 실행 승인이 아니다.
근거 = `docs/VERDICT_DET_DJDN_DRIVE_2026-09-21.md`(직전 판정 정본: DRV-D, 대각 단독 불충분 측정·국소/비국소 분할 미해결) · `docs/PREREG_DET_DJDN_DRIVE_2026-09-20.md`(직전 계약, 구조 계승 + §0 결함 정정) · `docs/DESIGN_PATHB_LINEARIZATION_2026-09-19.md`(Path B 아키텍처 정찰: λ*·tri_*·`coupled_newton_solve_all` 스텁·근인 배선) · `docs/PREREG_DET_DJDN_MVP_2026-09-19.md`(§0(b) 부호: 「이온화↑ ⟹ J↓」는 ∂J/∂χ 이며 저장 λ* 밖). **base = DRIVE 판정런 388393 폐합본**(`/gpfs/kjhan/lumina/cmfgen_map/djdn_drive_20260920/run_h100a100`, log-뉴턴 대각 항 ON, ITER=6, binary `7d42780346bf5d2c...`). **대조군 = 같은 388393**(물리 동일·hold 맵 재평가 없음) — 음성대조 보조 = DJDN-MVP 387175(`/gpfs/kjhan/lumina/cmfgen_map/djdn_mvp_20260919/run_h100a100`, 13 열·`epoch` 없음·hold 없음). 소스 기준 = 동결 트리 `/gpfs/kjhan/lumina/cmfgen_map/lteunmap_eps_20260917/source`(DRIVE 변경집합 포함 상태, manifest 185 파일·repo 미이식).

## 0. 기안자 검사 — 직전 두 계약 결함 정정 (운전자·사전감사 필독)

**(a) 다음 표적은 「완전 Path B」가 아니다.** DRIVE 판정문 §5 는 「다음 = 완전 Path B(tri_*·`coupled_newton_solve_all`·tensor core)」를 **방향**으로 적었고, 같은 문단이 s−g 의 국소(∂J/∂χ)/비국소(셸간) 분할을 **말하지 않는다**고 상한을 걸었다. 완전 N×N 결합 뉴턴을 이 계약에 넣지 않는다 — 크기·형태 미지로 짓는 것(Q-GOAL ④, 직전 독립 방향 평가가 금지한 형태).
**(b) 게이트 허용오차·referent 는 대조군 census 실측으로 먼저 고정한다.** DJDN-MVP §4 의 in-vivo 항등 1e-9 와 「행수 150」, DJDN-DRIVE 원안 1e-9 는 CSV 그레인(~2e-8 rel)보다 빡빡한 오기재였다. 388393 종료감사 실측: root=1∧damp>0 전 행 항등 위반 0, 최대 잔차 **2.16e-8 rel**, 개정 허용오차 **1e-6**. 이 계약은 (i) 생산 census 의 CSV 재계산 항등 = 1e-6(같은 그레인·같은 연산) (ii) 맵 재평가(assemble+`cmfgen_solve_J`+ion) 잔차는 그 연산의 잣대 = 기본 ALI_TOL 1e-3 의 5배(`lumina_cmfgen.c:2621`, 388393 은 `LUMINA_CMFGEN_ALI_TOL` 미설정) (iii) restore 는 스냅샷 memcpy 이므로 생산 `x_star` 의 `%.9e` 문자열 일치. **다른 연산에 1e-6 을 복사하지 않는다.**
**(c) 「행수 150」은 요구값이 아니다.** 388393 실측: 헤더 1 + 25051 데이터행 = gen 1..167 × **gen당 150행**(Z 14·16·20 × 셸 0–49), 최대 epoch=7=ITER+1. 기재값. 게이트는 「최대 epoch 안에 판정 셸 다섯 행이 각각 실재」만 요구한다(DRIVE §4 개정과 동일).
**(d) 판정량은 「구성상 참」이 아니다.** a>0·g=a/den·root=1 은 DRIVE 가 이미 닫은 배선 사실이다. 이 단의 판정량은 **런 없이 못 얻는 수**: 이웃 고정 재평가의 x_map_hold 와 그것으로 나눈 s_hold·s_nl. 388393 산출물에는 J/χ/tri 덤프가 없고(`LUMINA_CMFGEN_JDUMP` 미설정, `work/` 에 `lumina_cmfgen_jnu.csv` 없음) 재시작 체크포인트도 없다 ⟹ 오프라인으로 s_hold 를 못 만든다.
**(e) s_hold + s_nl = s 를 항등 게이트로 쓰지 않는다.** s_nl ≡ s − s_hold 로 정의하면 항등은 항진이다. 비항진 검사는 null-hold(교란 0 재평가가 생산 x_map 을 재현)와  sanity 셸(|s−g| 작은 셸에서 방법이 국소 채널을 날조하지 않음)이다.
**(f) 스텝 크기 ε 를 신설하지 않는다.** 직전 단의 C_max 가 발주자 결정 항목이 된 전례를 반복하지 않는다. 재평가는 같은 런의 **직전 epoch x_lag**(이미 census 에 있는 값)로 황 비를 되돌리는 반사실 한 점이다.

## 1. 계약 — 물리 계약 1개

**「DRIVE 착지 국면(판정 셸 Z=16 셸 12·16·20·24·28, 최대 epoch K=ITER+1)에서, 이웃 셸 인구를 현재 lag 에 고정한 채 대상 셸의 황 II→III 비만 직전 epoch 의 x_lag 로 되돌려 맵 M 을 한 번 재평가하면, 총미분 시컨트 s 와 대각 이득 g 의 차 s−g 가 국소 몫 Δ_loc = s_hold − g 인가 비국소 몫 Δ_nl = s − s_hold 인가를 가르는가.」** 판정런 1회(twin-T_e, binned 챔피언 경로, CONT 소스 ON, DR-on base, DJDN log-뉴턴 ON — 388393 과 같은 물리 + 마지막 epoch 의 hold 재평가). 대조 = 388393(같은 exports 계보·hold 없음).
⚠**말하지 않는 것**: CMFGEN 도달 · 착지(DRV-A/B/C/D 재질문) · 국소 묶음 안의 χ vs n_e 재분할 · 완전 Path B(N×N·tensor core·`coupled_newton_solve_all` 활성) · tri_* 블록 활성(후보 C) · 저장 λ* 밖 ∂J/∂χ 항을 솔버에 넣는 것(후보 B) · 이중 권위 · Milne 소스의 옳음 · Si/Ca 판정 · 스펙트럼 · 자유-T_e.

### Q-GOAL (기안자 자기 답 — 경량 사전감사가 독립 재판정)
① 칸: 어긋남 지도 **중간 셸 9–24 S II/III**(CMFGEN f_II .85/.80/.71 대 base .04/.12/.22). 이 단은 그 칸을 직접 줄이지 않는다 — CONT 연속체 소스가 폭주하는 기전의 **다음 수리 형태**(국소 ∂J/∂χ vs 셸간 tri_*)를 고르는 입력이다. 그 칸의 본선(CONT)은 DRIVE 가 대각만으로 닫히지 않음을 측정으로 확정했다. ② 산출 = 물리 측정(s_hold·s_nl·Δ_loc/Δ_nl 의 부호·크기, 판정 셸). 신규 계측 = hold census 파일 1(`lumina_pop_djdn_hold.csv`)·env 1. 생산 census 16 열은 무접촉. ③ 대안: (i) 후보 B 직행(국소 ∂J/∂χ 항 추가) — 형태(Λ*(χ) 없는 ∂J/∂χ 의 식)와 크기가 미지이고, sh24 s<0 은 국소 가설과 양립하나 측정이 아니다. (ii) 후보 C 직행(tri_* 활성) — 같은 이유로 형태 미지. (iii) 완전 Path B — Q-GOAL ④ 위반, 독립 방향 평가가 금지. (iv) 런 0회: 388393 에 J/χ/tri 덤프 없음·재시작 없음·저장 λ* 로 ALI 를 다시 나누는 것은 **모델의 분할**이지 맵 M 의 분할이 아님. ④ 건너뛰면 다음 단이 B·C·N×N 중 무엇을 지을지 형태 미지로 짓거나, sh24 에서 국소만으로 충분한 경우(또는 그 반대)를 놓친다.

### Q-LEAN (기안자 자기 답)
L1 물리량 = (i) 판정 셸의 s_hold·s_nl (ii) sh24 의 Δ_loc·Δ_nl 비(분기). 소비: hold census 의 `x_map_hold`·`x_map_null`·`x_lag_prev` → s_hold·null-hold·G-CPL / 생산 census 의 기존 `x_lag`·`x_map`·`g`·`epoch` → s·g·K(신규 열 0). 소비되지 않는 것: `LUMINA_CMFGEN_JDUMP`(388393 과 같이 미설정) · `gamma_eff`·`maxch_factor`·MAXCH 노브(직전 단과 동일, 코드 무접촉) · `tri_*` 활성 · `coupled_newton_solve_all`. L2 게이트 넷: G-ID(계보·소스·러너 상수) · G-RUN(완주·hold 파일 실재) · G-CPL(재평가 경로가 생산 맵을 재현하고 되돌리는가 — 아니면 분할이 공허) · G-SPLT(sh24 에 가를 |s−g| 가 실재하는가 — 아니면 국면 미재획득). DRIVE 의 G-DRV(착지·구동 적용)는 이 단이 묻지 않으므로 신설하지 않는다. L3 = 2 걸음(구현·selftest → 런 1). 더 적은 걸음: 388393 산출만의 오프라인 분할은 L1 의 맵 재평가가 없어 불가(§0(d)). L4 삭제·통합 후보: JDUMP 를 켜 오프라인 formal_solve 를 돌리는 경로 — 총 χ 1% 교란은 황 비 교란이 아니고, σ 적분을 밖에서 재현하면 이중 권위가 생긴다. **채택하지 않는다.** MAXCH 는 이번에도 삭제하지 않는다(삭제도 src 편집). 판정 셸 다섯을 유지하는 이유: sh24 가 분기 투표자이고 나머지 넷(|s−g|_7 ≤ 0.36)은 방법이 국소 채널을 날조하지 않는지의 sanity(G-CPL)다. 계측 증가 = 마지막 epoch 의 assemble+J+ion 6회(null 1 + 셸 5), Te 이분·준위 NLTE 해 재진입 없음.

## 2. 기전 — 런 0회 확정분

- **대조군 s−g [388393 judge_output.json · 판정문 §2, 저자 조회 census 원행]**. x^{(k)} = epoch k 마지막 행 x_star, s_k=(x_map^{(k)}−x_map^{(k−1)})/(x_lag^{(k)}−x_lag^{(k−1)}), g=a/den, κ=(s−g)/(1−g). K=7.

| 셸 | g_7 | s_7 | s−g | κ_7 | damp_7 | 비고 |
|---|---|---|---|---|---|---|
| 12 | 2.780568659 | 3.1439719425250967 | +0.36 | −0.20 | 0.03170800121 | |s−g| 소, sanity |
| 16 | 0.2096581554 | −0.020633488432262568 | −0.23 | −0.29 | 0.68930287 | sanity |
| 20 | 0.4081369659 | 0.3555552391667972 | −0.05 | −0.09 | 0.2066156222 | |s−g| 최소, sanity |
| 24 | **8.189966386** | **−4.584583035960915** | **−12.77** | **1.78** | 0.4140611145 | 부호 반대, 분할 표적 |
| 28 | 0.06706843776 | −0.20663429835432645 | −0.28 | −0.29 | 1.000 | 인접 폭주 셸, |s−g| 소 |

  sh24 epoch 7 생산 행(gen 166·167 동일, census `%.9e`): x_lag=1.092563223e+02, x_map=4.072876518e+02, a=7.779456053e-04, g=8.189966386e+00, x_star=1.498327537e+02, root=1, c_newton=−2.415102421e+00, damp=4.140611145e-01. 387175 는 같은 셸 epoch 5 에서 g=2.76·s=+3.80·root=0(항 OFF). ⟹ **가를 차 |s−g| 는 sh24 에 집중**되고 다른 넷은 ≤ 0.36. 이 집중은 국소 가설에 유리한 **시사**이지 분할이 아니다(s 는 이웃 과도를 포함).
- **맵 M 과 세 수.** 생산의 x_map_s^{(k)} = M(x_lag_s^{(k)}; N_{−s}^{(k)}) — 이웃 인구가 같이 움직인 점. 대각 g 는 저장 λ* 의 S-채널(∂J/∂S·∂S/∂x, `pop_djdn_a_reduce`: dSdx = (n_e Ec + S Xc)·geom/χ_t; χ 는 S=η/χ 의 미분에만 들어가고 **고정 S 에서 수송 연산자의 ∂J/∂χ 는 없음**). 정의:

  denom = x_lag_s^{(K)} − x_lag_s^{(K−1)}
  s = (x_map_s^{(K)} − x_map_s^{(K−1)}) / denom     (생산 census, DRIVE 와 동일)
  x_map_hold = M(x_lag_s^{(K−1)}; N_{−s}^{(K)})     (이 단의 한 점: 이웃은 현재, 대상 셸 비만 직전 lag)
  s_hold = (x_map_s^{(K)} − x_map_hold) / denom      (이웃 고정 몫)
  s_nl   = (x_map_hold − x_map_s^{(K−1)}) / denom    (이웃 변동 몫)
  Δ_loc = s_hold − g
  Δ_nl  = s − s_hold   (= s_nl, 정의)

  x_lag 불변(|denom|/max(|x_lag^{(K)}|,|x_lag^{(K−1)}|,1) < 1e-3) 이면 그 셸 s·s_hold 미정의 → NaN 기재, 그 셸은 분기 투표·sanity 에서 제외. |1−g|<1e-3 이면 κ 미정의(DRIVE 와 동일, 이 단은 κ 를 판정량으로 쓰지 않음).
- **재평가 절차 (생산 궤적을 바꾸지 않음).** 최대 epoch K 의 **생산** assemble + `cmfgen_solve_J` + `compute_ion_populations`(x_map·x_star 기록) **직후, 커밋 전**:
  1. 스냅샷: 전 셸 ion 밀도·n_e·χ·η·S·J·λ*·tri_* (T_e 는 twin 핀, 재이분 없음).
  2. **null-hold (sh24 만)**: 교란 0 으로 assemble+`cmfgen_solve_J`+ion 을 한 번 더 돌려 x_map_null 을 기록하고 스냅샷 memcpy 복귀.
  3. 판정 셸 다섯 각각: 스냅샷 복귀 → 그 셸 황 II+III 총량 N_tot 고정, 비 x 만 x_lag^{(K−1)} 로 재분배(N_III=N_tot·x/(1+x), N_II=N_tot/(1+x); 그 셸의 다른 원소·다른 셸 전부 스냅샷 유지) → assemble+`cmfgen_solve_J`+`compute_ion_populations`(n_e 폐합 포함, **log-뉴턴 커밋은 읽지 않음** — 기록량은 Picard x_map) → x_map_hold 기록 → 스냅샷 memcpy 복귀.
  4. 생산 x_star 커밋은 스냅샷의 생산 결과 그대로. 궤적에 FD 가 섞이면 G-CPL restore FAIL.
- **왜 이 한 점이 가르는가.** s_hold ≈ g (Δ_loc ≈ 0) 이면 저장 λ* 의 국소 S-채널이 같은 셸 감도를 이미 재현하고, 빠진 차는 이웃 변동 ⟹ 다음 = 후보 C. s_hold ≈ s (Δ_nl ≈ 0) 이면 이웃을 멈춰도 부호 반대의 큰 차가 남고, 빠진 차는 같은 셸 묶음(∂J/∂χ|_S · 국소 n_e · 이 셸 χ 가 전역 형식해에 미치는 대각 dCHI) ⟹ 다음 = 후보 B. 둘 다 |s−g| 의 1/3 을 넘으면 BOTH. sh24 의 s<0 은 이 측정이 닫기 전에는 B 의 근거가 되지 못한다.
- **비용.** 388393 실측 ITER 6 ≈ 13.6 h(a100×2, cpus 16) ⟹ ≈ 2.3 h/반복. 이 단의 추가분은 마지막 epoch 의 assemble+J+ion **6회**(null+5 셸), Te 이분·준위 해 없음. 상한은 1 반복분보다 작고, 하한은 런 없이 0.
- **미확정(계약 밖).** s_hold 국소 묶음 안의 χ vs n_e 분할 · 형식해 한 패스의 비선형 · 판정 셸 밖 · 스칼라 사다리 vs N×N 준위 해.

## 3. 수리안 — 변경집합

env `LUMINA_POP_DJDN_FDHOLD`(신설, 기본 미설정/0 = hold 없음·hold 파일 없음·**생산 경로 byte-동일**). ON(=1): §2 절차를 최대 epoch 에 한 번 수행하고 `work/lumina_pop_djdn_hold.csv` 를 쓴다. 생산 census `lumina_pop_djdn.csv` 열·대수·log-뉴턴·C_max **무접촉**. `LUMINA_POP_DJDN`·`LUMINA_POP_DJDN_CMAX` 유지. `LUMINA_CMFGEN_JDUMP`·`LUMINA_CMFGEN_LAMBDA_TRI`·`LUMINA_COUPLED_NEWTON` 을 이 단에서 켜지 않는다(J 솔버·물리 변경). 상태 클램프·floor 금지.

hold 파일 헤더(순서 고정, 16 열):

`shell,Z,stage,epoch,x_lag,x_lag_prev,x_map,x_map_hold,x_map_null,g,s,s_hold,s_nl,denom,restore_ok,null_rel`

- `x_map_null` 은 sh24 행만 채우고 다른 셸은 빈 칸이 아니라 NaN 표기 `nan`(null-hold 는 sh24 만).
- `restore_ok` = 1(memcpy 후 생산 x_star 의 `%.9e` 가 스냅샷과 일치) / 0.
- `null_rel` = |x_map_null/x_map − 1| (sh24), 다른 셸 `nan`.
- 행 = 판정 셸 다섯 × 최대 epoch 1개 = **5행**.

### 변경집합 — DJDN-SPLIT (표)

| 경로 | 담당 | 내용 |
|---|---|---|
| `src/env_universe.h` | Opus 4.8 | `LUMINA_POP_DJDN_FDHOLD` 등록 |
| `src/lumina_plasma.c` | Opus 4.8 | 황 비 → ion 밀도 재분배(N_tot 고정); epoch 별 x_lag 스태시; hold census 기록; 생산 log-뉴턴·16 열 census 무접촉; OFF byte-동일 |
| `src/lumina_cuda.cu` | Opus 4.8 | 최대 epoch 의 생산 assemble+`cmfgen_solve_J`+ion **직후·커밋 전** §2 절차 오케스트레이션(기존 `cmfgen_assemble`·`cmfgen_solve_J`·`compute_ion_populations` 재진입, 새 커널 0); 스냅샷 memcpy; Te 이분·`coupled_newton_solve_all` 호출 추가 금지 |
| `tests/pop_djdn_selftest.c` | Opus 4.8 | 기존 30 케이스 유지 + **합성 맵** M = g·Δx_s + b·Δx_{s±1} 픽스처에서 s_hold → g, s_nl → b (1e-12 rel) · **null 픽스처** Δx=0 ⟹ x_map_hold == x_map bit-동일 · **OFF** FDHOLD 미설정 ⟹ hold 파일 없음·생산 경로 byte-동일 · **음성대조 1**: 이웃 Δx 를 대상 셸에 섞는 구현 ⟹ s_hold 가 g 를 못 맞춤 FAIL 시연 · **음성대조 2**: restore 생략 ⟹ restore_ok=0 FAIL 시연 · 헤더 16 열 일치 |

### 변경집합 끝

★**src 편집 1개 규칙**: 가동 중 src-편집 태스크 0 확인 후 발주(DRIVE 폐합). ⚠프리플라이트: 표의 네 파일 중 `tests/pop_djdn_selftest.c` 와 DJDN 심볼은 repo 에 없고 동결 트리 복사본에만 있으므로 `expected_extra` 로 선언 — 직전 세 계약과 동일 처리, 기대 출력 `changeset-consistent files=4`. `path_pattern` 은 `.cu` 를 포함한다.

## 4. 게이트 명부 (표)

| ID | 조건 | 음성대조 |
|---|---|---|
| G-ID | 새 런 exports 를 대조군 388393 계보(`/gpfs/kjhan/lumina/cmfgen_map/djdn_drive_20260920/run_h100a100/input/resolved_lumina.exports`)와 diff — 추가·변경 키가 정확히 `LUMINA_POP_DJDN_FDHOLD`(=1) [§6 ① ITER 변경 시 `LUMINA_PURE_CMFGEN_ITER`; 프로필 종속 `LUMINA_CMF_FINE_MGPU_DEVICES` 허용]; `LUMINA_CMF_CONT_NLTE_SOURCE`(=1)·`LUMINA_POP_DJDN`(=1)·`LUMINA_POP_DJDN_CMAX`(=1.0)·`LUMINA_ION_CHANNEL_DUMP`(=1)·`LUMINA_FROZENIN_DR`(=1)·`LUMINA_DR_BOOST_AUTOSTRUCT`·`LUMINA_DR_BOOST_BADNELL`·`LUMINA_DR_BOOST_MAZZOTTA`·`LUMINA_DR_BOOST_NORAD`(각 =1) 유지; `LUMINA_POP_MAXCH`·`LUMINA_ION_NT`·`LUMINA_NT_FULL`·`LUMINA_RADEQ_SIMUL`·`LUMINA_SIMUL_ION_DAMP`·`LUMINA_CMFGEN_JDUMP`·`LUMINA_CMFGEN_LAMBDA_TRI`·`LUMINA_COUPLED_NEWTON` **부재**; 동결 트리 `/gpfs/kjhan/lumina/cmfgen_map/lteunmap_eps_20260917/source.sha256` 의 `sha256sum -c` non-OK 가 정확히 §3 표의 파일; 새 바이너리 해시 `input/binary.sha256` 기재(대조군 388393 의 `7d42780346bf5d2c...` 와 달라야 함); 러너의 **ITER 상수 셋** — guard `"${LUMINA_PURE_CMFGEN_ITER:-}" == N` · `outer_iterations=N` · postcheck `-ne N` — 이 전부 선택 ITER 와 같음을 `grep -nE 'PURE_CMFGEN_ITER:-\}" == \|outer_iterations=\|-ne [0-9]+ \]\]; then' job.slurm` 출력으로 STAGE_IDENTITY 에 기재; 러너 편집 = job-name·`--time`·ITER 상수 셋·프로필(partition/gres/GPU_PROFILE/cpus, §6 ②)뿐, `runner_edits` 에 `postcheck_te_count` 포함 | 다른 키 혼입·§3 밖 소스 변조·postcheck `-ne` 가 ITER 와 불일치·JDUMP/LAMBDA_TRI/COUPLED_NEWTON 혼입 ⟹ FAIL(388393 러너 그대로 = FDHOLD 키 부재로 FAIL) |
| G-RUN | `RUN_FOOTER.txt` `terminal_state=SUCCESS`·`model_rc=0`·`outer_iterations`=선택 ITER; `work/lumina_pop_djdn.csv`·`work/lumina_pop_djdn_hold.csv`·`work/lumina_ion_channels.csv`·`work/physics_comparison/physics_DET_iter<ITER−1>.ion.csv` 실재 | 미완주·hold 결손 ⟹ FAIL; 388393 = hold 파일 없음 ⟹ FAIL; 387175 = hold 없음·생산 census 13 열 ⟹ FAIL |
| G-CPL | (오프라인) `tests/pop_djdn_selftest.c` rc=0, §3 음성대조 둘 다 FAIL 시연 — **봉인 접촉 전 Fable 코드 감사 선행**(개정16 ②); (in-vivo 생산 census) 헤더 16 열 = DRIVE 와 동일(`epoch,c_newton,damp` 포함), 항등 검사 행 집합 = root=1 ∧ x_lag>0 ∧ x_map>0 ∧ damp>0 인 전 행: ρ=x_map/x_lag, c' = ln ρ·g/(ρ−g) 로 재계산해 \|c_newton − c'\| ≤ 1e-6·max(1,\|c'\|) **and** \|x_star − x_map·exp(damp·c_newton)\| ≤ 1e-6·x_star **and** damp = min(1, C_max/\|c_newton\|)(1e-6) [388393 실측 그레인 최대 2.16e-8 rel, 허용오차 1e-6 — 맵 재평가에 이 숫자를 복사하지 않음]; damp==0 행은 항등에서 제외·행 수와 (Z,shell,epoch) 기재; epoch 구조: 같은 (Z,shell,epoch) 안 x_lag 불변, epoch 단조 증가, 각 판정 셸에서 epoch k+1 첫 행의 x_lag = epoch k 마지막 행의 x_star(`%.9e` 일치), 최대 epoch = ITER+1; 최대 epoch 의 각 gen 에 판정 셸 다섯(Z=16, 셸 12·16·20·24·28) 행이 각각 정확히 1개 실재; 판정 셸 epoch ≥ 2 전 행 root=1·a>0; (in-vivo hold) 헤더 16 열 순서 일치·데이터 5행·각 판정 셸 1행·epoch=ITER+1·Z=16·stage=1; 전 행 restore_ok=1; sh24 의 \|x_map_null/x_map − 1\| ≤ 5e-3 [5× 기본 ALI_TOL 1e-3, `lumina_cmfgen.c:2621`, 388393 미설정]; hold 의 x_map·g·x_lag 가 생산 census 같은 (shell,epoch 마지막 gen) 행과 `%.9e` 일치; denom = x_lag − x_lag_prev 가 생산 census 의 그 두 epoch x_lag 차와 1e-6 rel; sanity: 판정 셸 중 |s−g|≤ 0.5 이고 s 가 정의된 셸(388393 실측 sh12/16/20/28 의 |s−g|_7 = 0.36/0.23/0.05/0.28, 전부 ≤ 0.5)에서 \|s_hold − g\| ≤ 1.0 [1.0 = 그 넷의 최대 |s−g| 0.36 위·sh24 의 12.77 아래] | 생산 census 열 바꿔치기·c 교란 ⟹ 항등 FAIL; hold 헤더 바꿔치기 ⟹ FAIL; restore_ok=0 빌드 ⟹ FAIL; null-hold 가 x_map 을 0.05 dex 옮기는 빌드 ⟹ 5e-3 FAIL; 388393 자기검사 ⟹ hold 부재 FAIL |
| G-SPLT | 새 런 생산 census 의 Z=16 sh24 최대 epoch 에서 s 가 정의되고 \|s − g\| > 1.0 [388393 실측 12.77; 다른 넷의 최대 0.36 위] | 388393 은 hold 가 없어 이 게이트의 분할 산출은 FAIL(파일 결손, G-RUN 에서 이미); 항만 있고 재평가가 생산 x_map 을 그대로 복사하는 빌드 ⟹ s_hold 가 s 또는 g 로 붕괴해 sanity/분기에서 잡힘; \|s−g\|≤1 인 새 런 ⟹ 국면 미재획득 FAIL |

### 게이트 명부 끝

## 5. 기대치 — 런에서만 얻는 수 (잣대를 대조군에 먼저 적용: 388393 은 hold 파일이 없어 E-1 을 못 만든다)

K = ITER+1. 판정 셸 다섯의 s·g 는 생산 census 의 epoch K 와 K−1 마지막 행(DRIVE 정의). s_hold·s_nl 은 hold 파일. 투표자는 **sh24 하나**. 나머지 넷은 G-CPL sanity 이지 분기가 아니다.

| ID | 양 | 대조(388393) | **기대·역할** |
|---|---|---|---|
| E-1 분할 | sh24 의 s_hold, s_nl, Δ_loc=s_hold−g, Δ_nl=s−s_hold, |s−g| | 없음(hold 부재). 생산 쪽 s−g=−12.77, g=8.189966386, s=−4.584583035960915 | **판정량**(분기 표). 런 없이 못 얻음 |
| E-2 sanity 기재 | sh12/16/20/28 의 s_hold−g (정의된 셸) | 없음 | 기재. G-CPL 이 \|s_hold−g\|≤1.0 을 이미 강제 |
| E-3 null/restore | sh24 null_rel, 전 셸 restore_ok | 해당 없음 | 기재. 첫 실측 그레인. 게이트 문턱(5e-3 / restore_ok=1)과 별개로 판정문에 수치를 남긴다 |
| (기재) | 생산 궤적 Δ_k · damp · `Z16_nlte_q2/q1` · 388393 대비 같은 epoch 의 log10 x 차 | 388393·387175 | 판정 조건 아님. 이 단은 착지를 다시 묻지 않음 |

문턱 근거(전부 388393 census·judge_output 실측에서 읽음): |s−g|>1.0 은 sh24 의 12.77 과 다른 넷의 최대 0.36 사이; sanity 1.0 은 같은 간격; null-hold 5e-3 rel 은 기본 ALI_TOL 1e-3 의 5배(맵 재평가 연산, CSV 항등 1e-6 과 다른 연산); 생산 CSV 항등 1e-6 은 388393 종료감사 최대 2.16e-8 rel 의 여유(직전 단과 동일 연산). 2:1  Dominances 는 분류 규칙이다 — |s−g|~12 에서 작은 쪽이 1/3 미만이면 다음 수리의 구동원이 아니라고 본다. 기안자 예상(판정 조건 아님·확률 미부여): Path B §1 의 χ 복원 부호와 sh24 s<0 은 SPLIT-LOC 에 유리하고, sh28 |s−g| 소는 비국소 약함을 시사한다. s 가 여전히 과도 시컨트라 이 예상은 강건하지 않다 — 런이 정한다.

### 분기 (표)

sh24 코드: 게이트 미충족 ⟹ 0. 게이트 충족 시 |s−g|≤1 이면 이 단은 G-SPLT 에서 이미 FAIL 이므로 여기에 도달하지 않는다. 도달 시 Δ_loc=s_hold−g, Δ_nl=s−s_hold.

| 분기 | 조건 | 의미 |
|---|---|---|
| SPLIT-X | G-ID/G-RUN/G-CPL/G-SPLT 중 하나라도 미충족 | 미시연 |
| SPLIT-LOC | 게이트 전건 **and** \|Δ_loc\| ≥ 2·\|Δ_nl\| | **국소 몫이 벌크** — 다음 최소 수리는 후보 B(저장 λ* 밖 같은 셸 ∂J/∂χ 복원 채널, 대각 여전). tri_* 활성·N×N 은 이 분기에서 열리지 않음 |
| SPLIT-NL | 게이트 전건 **and** \|Δ_nl\| ≥ 2·\|Δ_loc\| | **비국소 몫이 벌크** — 다음 최소 수리는 후보 C(인접 셸 tri_* 블록만, 전 N×N 아님) |
| SPLIT-BOTH | 게이트 전건 **and** LOC 도 NL 도 아님 | 두 몫이 비교 가능 — 다음 단은 B 와 C 중 **더 작은 구현**을 한 번에 하나만 기안(이 계약이 둘을 짓지 않음) |
| SPLIT-D | (예비; 기계 분기 표의 잔여 코드) 게이트는 통과했으나 s_hold 비유한·denom 부호 붕괴 등 G-CPL 이 놓친 기재 이상 — 판정자가 0 으로 재분류하면 X | 도구 결함. 재분기 금지, 판정문에 명명 |

### 분기 끝

재분기 금지. 판정기 = `judge_drive.py` 계보의 운전자 도구를 **hold 5행·§4 null/restore/sanity·§5 sh24 코드**로 개정한 것(해시 `READOUT_TOOLS.sha256` 기재; 소스 아님). 음성대조: 388393 자기검사 ⟹ G-RUN(hold 부재) FAIL ⟹ SPLIT-X; 387175 ⟹ 생산 13 열·hold 없음 ⟹ SPLIT-X.

## 6. 자원 · 실행 · **결정 항목(운전자·user)**

새 RUN 루트 `/gpfs/kjhan/lumina/cmfgen_map/djdn_split_20260921/run1`(형제 루트), 입력 = 388393 입력 복사 + 새 빌드 + exports 편집(G-ID). GPU 1 런. 봉인 base·대조군 읽기만.

① **ITER** — **6**(≈14 h + 마지막 epoch 재평가, `--time=20:00:00`, K=7 = 388393 과 같은 창) 또는 **4**(≈10 h, `--time=14:00:00`, K=5). 388393 에서 |s−g| 인용값은 epoch 7 이다. ITER 4 는 그 창이 아니다 — G-SPLT 의 |s−g|>1 가 그 창에서 성립하는지는 런이 정한다. 기안자는 못 박지 않는다.
② 프로필·헤지 — 직전 단과 동일(먼저 시작하는 쪽만; 멀티파티션·cpus 16 편집은 G-ID 에 기재). GPU-bound·byte-배터리 없음.
③ NT 키 부재·MAXCH 미설정·JDUMP 미설정·LAMBDA_TRI 미설정·COUPLED_NEWTON 미설정(388393 그대로).
④ C_max=1.0 유지(새 결정 아님, G-ID 키 유지).

순서: 경량 사전감사(Opus 4.8 fresh, Q-GOAL·Q-LEAN — 특히 「B/C 직행이 지금 열려 있는가」를 물을 것) → 게이트 4종(prereg-preflight `files=4`·gate_referents·doc_canonical·dispatch-preflight) → Opus 4.8 구현(동결 트리 복사본) → selftest → Fable 코드 감사(첫 항목 = 봉인 접촉 가부) → 재봉인·빌드 → 스테이징(**러너 ITER 상수 셋 grep 확인**) → 제출 → 판독 → 판정(운전자) → 종료감사(Opus 4.8 fresh; GPU 지출·물리 주장·다음 수리 형태가 걸린 단이라 생략하지 않음).

## 7. 분장 장부 (개정25 · 규약 / 실제 / 위반)

| 단계 | 규약 | 실제 | 위반 |
|---|---|---|---|
| 기안 | Fable fresh(25-b) | **Fable 대행**(`grok-4.6`, user 2026-09-20 지시) | 규약 담당 ≠ 실제 모델(Fable 토큰 소진, 기재) |
| 경량 사전감사(Q-GOAL·Q-LEAN) | Opus 4.8 fresh(25-d) | 미집행 | |
| 구현 + selftest | Opus 4.8 | 미집행 | |
| 코드 간단 감사 | Fable fresh(25-a) | 미집행 | ⚠기안 대행자 = 감사 대행 후보(자기 계약 편향, 개정25-b 기재) |
| 스테이징·제출·판독·판정 | 운전자 | 미집행 | |
| 종료감사 | Opus 4.8 fresh | 미집행 | |

## 8. 주장 상한

말할 수 있는 것: 「DRIVE 착지 국면에서 이웃을 고정한 맵 재평가가 s−g 의 국소 몫과 비국소 몫을 가르면, 다음 최소 수리가 같은 셸 ∂J/∂χ 묶음인가 셸간 tri_* 인가(또는 둘 다인가)」. 착지·CMFGEN 근접은 기재 사항이지 이 단의 성공 조건이 아니다. **국소 묶음을 χ 와 n_e 로 다시 가르지 않는다.** s_hold 는 한 epoch 의 반사실 한 점이며 전 야코비안이 아니다. 「FDHOLD 토글 단독」은 OFF 런이 없으므로 selftest byte-동일 + 코드 감사의 오프라인 주장 조건부; 생산 궤적 불변은 G-CPL restore 가 in-vivo 로 본다. 말하지 않는 것: 후보 B/C 의 구현 · N×N 해와의 정합 · 정확 솔버 Λ* · Milne 소스 · Si/Ca · 스펙트럼 · 자유-T_e · `coupled_newton_solve_all` 활성. twin-T_e 핀 조건부.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 — DJDN-SPLIT (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "(?:src|tests)/[A-Za-z0-9_/.]+\\.(?:c|h|cu)",
    "symbol": "LUMINA_POP_DJDN_FDHOLD",
    "roots": ["src", "tests"],
    "expected_extra": ["src/env_universe.h", "src/lumina_plasma.c", "src/lumina_cuda.cu", "tests/pop_djdn_selftest.c"]
  },
  "branches": {
    "regimes": [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
    "metrics": {"count": "len(v)", "gate_fail": "sum(1 for x in v if x < 0.5)", "loc": "sum(1 for x in v if 0.5 <= x < 1.5)", "nl": "sum(1 for x in v if 1.5 <= x < 2.5)", "both": "sum(1 for x in v if 2.5 <= x < 3.5)", "d": "sum(1 for x in v if x >= 3.5)"},
    "rules": [
      {"name": "SPLIT-X", "predicate": "count == 0 or gate_fail > 0"},
      {"name": "SPLIT-LOC", "predicate": "count > 0 and gate_fail == 0 and loc > 0 and nl == 0 and both == 0 and d == 0"},
      {"name": "SPLIT-NL", "predicate": "count > 0 and gate_fail == 0 and nl > 0 and loc == 0 and both == 0 and d == 0"},
      {"name": "SPLIT-BOTH", "predicate": "count > 0 and gate_fail == 0 and both > 0 and loc == 0 and nl == 0 and d == 0"},
      {"name": "SPLIT-D", "predicate": "count > 0 and gate_fail == 0 and d > 0 and loc == 0 and nl == 0 and both == 0"}
    ],
    "adversarial_fixtures": [
      {"name": "empty", "mix": []},
      {"name": "gate-fail", "mix": [[0.0, 1]]},
      {"name": "loc", "mix": [[1.0, 1]]},
      {"name": "nl", "mix": [[2.0, 1]]},
      {"name": "both", "mix": [[3.0, 1]]},
      {"name": "d", "mix": [[4.0, 1]]},
      {"name": "loc-and-nl", "mix": [[1.0, 1], [2.0, 1]]}
    ],
    "residual": "SPLIT-X"
  },
  "references": [
    {"path": "docs/VERDICT_DET_DJDN_DRIVE_2026-09-21.md"},
    {"path": "docs/PREREG_DET_DJDN_DRIVE_2026-09-20.md"},
    {"path": "docs/DESIGN_PATHB_LINEARIZATION_2026-09-19.md"},
    {"path": "docs/PREREG_DET_DJDN_MVP_2026-09-19.md"},
    {"path": "CLAUDE.md"},
    {"path": "src/lumina_plasma.c", "flags_existing": ["parity_rate_se_ratio_checked", "compute_ion_populations_shell", "radeq_set_line_re_source", "g_lre_lambda_star", "radeq_set_tri_response", "coupled_newton_solve_all", "nlte_population_candidate_commit_bundle"], "flags_planned": ["LUMINA_POP_DJDN_FDHOLD"]},
    {"path": "src/lumina_cuda.cu", "flags_existing": ["cmfgen_assemble", "cmfgen_solve_J", "radeq_set_line_re_source", "coupled_newton_solve_all"], "flags_planned": ["LUMINA_POP_DJDN_FDHOLD"]},
    {"path": "src/env_universe.h", "flags_existing": ["LUMINA_SIMUL_ION_DAMP", "LUMINA_CMFGEN_JDUMP", "LUMINA_CMFGEN_LAMBDA_TRI"], "flags_planned": ["LUMINA_POP_DJDN_FDHOLD"]},
    {"path": "src/lumina_cmfgen.c", "flags_existing": ["cmfgen_solve_J", "cmfgen_assemble", "lambda_star", "tri_lo", "LUMINA_CMFGEN_JDUMP"]}
  ]
}
```

## 9. 집행 기록

1. **게이트 4종(발주 전)**: prereg-preflight rc=0(`changeset-consistent files=4`·branch-partition rules=5 fixtures=7 residual=SPLIT-X·references-resolved refs=9)·gate_referents rc=0(exempt=4)·doc_canonical rc=0. 커밋 26f979f.
2. **경량 사전감사(Opus 4.8 fresh, `claude-opus-4-8`, 개정25-d): PREAUDIT 통과·BLOCKING 0.** Q-GOAL 부합(measure-before-build 옳음 — B/C 직행은 형태 미지 Q-GOAL④ 위반; grok A 선택 타당)·**오프라인 불가 실측 확증**(대조군 388393 work/ 에 J장·JDUMP·λ*·tri_* 덤프·체크포인트 전무·census 스칼라 궤적만 → 반사실 맵 재평가 재런 필수). Q-LEAN 통과(hold census 16열 전건 소비·생산 census 무접촉·4게이트 실재 비중복[G-DRV 드롭]·2걸음 최소·4파일 삭제/통합 후보 0). 비차단 기재 3: ①**칸 미이동·3단째 진단(MVP→DRIVE→SPLIT)임을 판정문·user 보고에 명시** ②**ITER=6 권고**(ITER=4 는 G-SPLT |s−g|>1 창 위험) ③s_hold 단일점 분할 잔여 위험(상한 문제).
3. **§6 결정(운전자·user)**: 미집행 — ①ITER 4 또는 6.
4. **구현 + selftest / 코드 간단 감사 / 재봉인·빌드·스테이징·제출 / 판독·판정 / 종료감사**: 미집행.
