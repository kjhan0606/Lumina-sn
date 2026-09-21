# 사전등록 — DET-PATHB-FULL: CMFGEN 식 완전 ∂J/∂N 선형화(결합 뉴턴)를 라이브 생산 루프에 적용하면 CONT 폭주 셸이 유한 고정점에 착지하는가 (2026-09-21)

기안 = **Fable 대행**(`grok-4.6`, user 2026-09-20 Fable 토큰 소진 이후 계약 기안 대행). 상태 = **★폐기(2026-09-21, user 결정 B — 접지된 단일 채널 단으로 대체)**. 근거 = grok 물리·대수 사전감사(개정25-e) «폐합 차단» 7건(§9 집행 기록 2), 특히 ⑩ 「형태-확실」반증 ⟹ user 가 완전 Path B 직행을 접고 **접지된 작은 단 먼저**(단일 ∂J/∂χ|_S 채널·뉴턴 A2-10 슬롯·tri/N×N/이중권위 이 단 밖)를 택함. 후속 = DET-PATHB-LOCAL(grok 재기안). 이 문서는 **완전 직행이 왜 안 되는가의 기록**으로 보존(판별런 미발주). 실행 승인이 아니었다.
근거 = `docs/VERDICT_DET_DJDN_DRIVE_2026-09-21.md`(직전 판정 정본: DRV-D, 대각 단독 불충분 **측정**·s−g 가 Path B 크기) · `docs/PREREG_DET_DJDN_DRIVE_2026-09-20.md`(직전 폐합 계약, 구조 계승) · `docs/PREREG_DET_DJDN_SPLIT_2026-09-21.md`(**폐기** — 국소/비국소 판별; 재평가가 mini-Path-B 를 요구해 「값싼 판별」전제 붕괴) · `docs/DESIGN_PATHB_LINEARIZATION_2026-09-19.md`(Path B 아키텍처 정찰: `do_var_cont.f` 발자국·tri_*·`coupled_newton_solve_all` 스텁·tensor core·이중 권위) · user 2026-09-21 「완전 선형화(Path B full) 직행」(CMFGEN 식은 형태-확실 ⟹ Q-GOAL④ 「크기·형태 미지로 짓는 것」에 해당하지 않음). **base = DRIVE 판정런 388393 폐합본**(`/gpfs/kjhan/lumina/cmfgen_map/djdn_drive_20260920/run_h100a100`, log-뉴턴 대각 항 ON, ITER=6, binary `7d42780346bf5d2c...`). **대조군 = 같은 388393**(대각 스칼라 선형화·착지 0/5). 소스 기준 = 동결 트리 `/gpfs/kjhan/lumina/cmfgen_map/lteunmap_eps_20260917/source`(DRIVE 변경집합 + SPLIT-substrate OFF-게이트 포함, manifest 185 파일·repo 미이식).

## 0. 기안자 검사 — 직전 결함 정정 (운전자·사전감사 필독)

**(a) 이 계약의 표적은 완전 Path B 이다.** SPLIT 계약은 user 2026-09-21 로 **폐기**됐다(판별런 미발주). 직전 프롬프트 「완전 Path B 를 한 계약에 짓지 마라」는 그 결정으로 무효. DRIVE 판정문 §5 「다음 = 완전 Path B」가 방향이었고, user 가 판별 단을 건너뛰고 직행을 택했다. 근거(user 원문): CMFGEN 식 완전 선형화는 **형태-확실**(추측 아닌 정답 수리)이라 「크기·형태 미지로 짓는 것」(Q-GOAL④)에 해당하지 않는다. 크기 입력은 DRIVE s−g 가 이미 준다(아래 §2). Q-LEAN(계측 최소·게이트 비양산)은 여전히 적용한다.

**(b) 반드시 피할 3 배선 오류 (직전 3연속 재발 — 기계 게이트 미포착).**
1. **오케스트레이션 파일 = live `cmfgen_run`(`lumina_cmfgen.c`).** `lumina_cuda.cu` 의 per-epoch 루프는 `LUMINA_DET_TRANSACTIONAL=1` 에서 `cmfgen_run` 후 `return EXIT_SUCCESS`(동결 트리 `lumina_cuda.cu:8073-8100`) — **죽은 코드**. 변경집합에 `lumina_cuda.cu` 를 넣지 마라(DJDN-MVP §0a·SPLIT §3). `coupled_newton_solve_all` 시그니처를 바꾸지 마라(cuda.cu 호출부가 남아 컴파일만 됨 — 시그니처 변경은 cuda.cu 접촉을 강제).
2. **장 되먹임 = `cmfgen_commit_jnu`.** 인구 solve 가 갱신된 J 를 소비하려면 반드시 commit 을 거친다(bf-rate 추정기 `radfield_view` 재구축, 동결 트리 정의 `:3873`·생산 호출 `:8095`). commit 을 건너뛰면 ∂J 되먹임이 낡은 추정기를 읽어 감도가 무의미(SPLIT §2).
3. **referent·허용오차·기대치는 대조군 census 실측으로 먼저 고정.** 388393 census 16 열: `gen,shell,Z,stage,x_lag,x_map,a,g,x_star,root,gamma_lag,gamma_eff,maxch_factor,epoch,c_newton,damp`. CSV 재계산 항등 그레인 최대 **2.16e-8 rel** ⟹ 항등 게이트 허용오차 **1e-6**(1e-9 금지, DRIVE §4 오기재). 맵 재평가·FD 잔차는 그 연산의 잣대(기본 ALI_TOL 1e-3; 이 단은 hold 재평가를 하지 않음). 판정량은 「구성상 참」이 아닌 「런 없이 못 얻는 수」로 등록. **재등록 금지**: `g=a/den`·`root=1` 은 DRIVE 가 이미 닫음. 「행수 150」은 기재값(388393: gen당 150행 = Z 14·16·20 × 셸 0–49)이지 요구값이 아님.

**(c) 대각 log-뉴턴과 Path B 를 동시에 적용하지 않는다.** DRIVE 의 `a = Σ_l f_l Σ_b Gw_b λ*_s,b ∂S_b/∂x` 는 완전 야코비안의 **∂J/∂S 대각 조각**이다. Path B ON 에서 그 조각을 스칼라 log-뉴턴으로 한 번 더 적용하면 λ* 채널이 이중 계산된다. Path B ON ⇒ 이온화 뉴턴 커밋은 `coupled_newton_solve_all` 뿐. 생산 census 의 `g`(대각 a/den)는 **기재·E-2 비교용**으로 남기고, 커밋 `x_star` 는 결합 뉴턴의 값이다.

**(d) `LUMINA_CMFGEN_LAMBDA_TRI` 로 ALI J 솔버를 Thomas 삼중대각으로 바꾸지 않는다.** 388393 은 대각 ALI(미설정). `LAMBDA_TRI=1` 은 J 장 자체를 바꿔 착지 비교를 교란한다. Path B 가 쓰는 것은 `formal_solve_bin` 이 이미 계산하는 Tlo/Tup(이웃 응답)을 `cs.tri_lo`/`cs.tri_up` 에 **저장**하는 것(변분 야코비안). Thomas ALI 폐합은 이 계약 밖. NT·JDUMP·MAXCH 신설/삭제 금지. SPLIT `LUMINA_POP_DJDN_FDHOLD` 는 미설정(OFF-게이트 잔존, 삭제하지 않음).

**(e) 러너 ITER 상수 셋.** DJDN-MVP 는 postcheck `-ne 2` 하드코딩으로 G-RUN 을 잃었다. 이 계약은 선택 ITER = N 을 §6 에서 고르고, 러너의 guard `"${LUMINA_PURE_CMFGEN_ITER:-}" == N` · `outer_iterations=N` · postcheck `-ne N` 을 G-ID 가 강제한다. 퇴화 장부: 고정점 도달에 ITER 8–10(≈20–25 h/런)이 필요할 수 있음 — 기안자는 값을 못 박지 않되 **N 을 계약·러너·postcheck 에 같은 숫자로 적으라**.

## 1. 계약 — 물리 계약 1개

**「CMFGEN `do_var_cont.f` 발자국(dCHI·dETA 를 BA 야코비안에 접어 dJ/dN)의 완전 Path B — ① 저장 λ* 밖의 국소 ∂J/∂χ|_S ② 셸간 삼중대각 `tri_lo`/`tri_up`/`tri_r` ③ 셸별 N×N rate 행렬(STEQ/BA, `nlte_solve_ion_shell`)에 bf 소스 결합 ④ 이중 인구 권위 해소(단계 총량=스칼라 사다리 `parity_rate_se_ratio_checked` 대 준위 분포=N×N; η_bf 는 둘 다 의존) — 를 라이브 `cmfgen_run` 의 `coupled_newton_solve_all` 로 적용하면, CONT 소스 위 황 II→III 비 x 의 epoch 궤적이 판정 셸(Z=16, 셸 12·16·20·24·28)에서 유한 고정점에 착지하는가(PATHB-A), 아니면 어떤 꼴로 착지하지 않는가(PATHB-B 진동 / PATHB-C 단조 상승 / PATHB-D 혼재); 그리고 그 국면에서 경험 총미분 s 와 완전 야코비안 이득 G_full 의 차 s−G_full(정규화 κ_B=(s−G_full)/(1−G_full))는 대각-only κ_diag=(s−g)/(1−g) 대비 얼마인가.」** 판정런 1회(twin-T_e, binned 챔피언 경로 `formal_solve_bin`/`cmfgen_solve_J`, CONT 소스 ON, DR-on base, DJDN census ON·스칼라 log-뉴턴 적용 OFF). 대조 = 388393(같은 물리 base·대각 log-뉴턴 커밋).

⚠**말하지 않는 것**: CMFGEN f_II 도달 · s−g 의 국소/비국소 **분할 측정**(이 단은 둘을 형태로서 모두 넣지, SPLIT 을 되풀이하지 않음) · 정확 multi-GPU 솔버 Λ* · Milne 소스 자체의 옳음 · Si/Ca 판정(기재만) · 스펙트럼 · 자유-T_e · NT 활성 · `LUMINA_CMFGEN_LAMBDA_TRI` Thomas ALI · JDUMP · MAXCH 노브 신설/삭제 · `lumina_cuda.cu` 레거시 루프.

### Q-GOAL (기안자 자기 답 — 경량 사전감사가 독립 재판정)
① 칸: 어긋남 지도 **중간 셸 9–24 S II/III**(CMFGEN f_II .85/.80/.71 대 base .04/.12/.22). 이 단은 그 칸을 직접 「CMFGEN 숫자로」 줄이는 것을 성공 조건으로 두지 않는다 — CONT 연속체 소스(그 칸의 본선)가 폭주하는 기전을, CMFGEN 이 쓰는 수리(완전 dJ/dN)로 **맵 자체를** 바꿔 유한 고정점에 착지시키는가를 잰다. 착지 없이 그 칸은 닫히지 않는다. ② 산출 = 물리 측정(착지 여부 · in-vivo s, g, G_full, κ_diag, κ_B). 신규 계측 = jac census 파일 1(`lumina_pathb_jac.csv`, 판정 셸×최대 epoch 5행)·신규 env 0(`LUMINA_COUPLED_NEWTON` 기존). 생산 census 16 열 포맷 유지. ③ 대안: (i) SPLIT 판별 — 폐기(mini-Path-B 비용으로 「값싼 판별」붕괴; user 직행). (ii) 대각 유지 — DRIVE 착지 0/5, sh24 κ_diag=1.78>1 측정. (iii) NT ON — CMFGEN toy06 VADAT `F [TRT_NON_TE]`(비열 OFF); 발자국 규약상 Lumina `LUMINA_ION_NT` 부재 유지. (iv) EUV 펌프 — SPLIT §9 오프라인 기각(중간 셸 +0.01–0.02). 지금 열린 더 가까운 표적 없음. ④ 건너뛰면 CONT 본선이 폭주 맵으로 남고, 최종목적(MC 수송으로 CMFGEN 수준 SN Ia 스펙트럼)의 그 칸이 닫히지 않는다. 형태는 CMFGEN 식이 주고 크기는 DRIVE s−g 가 줬으므로, 이 단을 「미지 형태를 추측으로 짓는 단」으로 세지 않는다.

### Q-LEAN (기안자 자기 답)
L1 물리량 = (i) 판정 셸 커밋 x 의 epoch 궤적 Δ_k(착지) (ii) s, g, G_full, κ_diag, κ_B. 소비: 생산 census `x_lag`/`x_map`/`x_star`/`epoch`/`root`/`damp`/`c_newton`/`g` → Δ_k·s·착지 정의·리미터 항등 / jac census `G_chi`/`G_tri`/`G_full`/`tri_persisted`/`dchi_assembled`/`commit_gen`/`n_newton`/`jac_res` → G-JAC·E-2. 소비되지 않는 것: `LUMINA_CMFGEN_JDUMP`(388393 과 같이 미설정) · `gamma_eff`·`maxch_factor`·MAXCH 노브(코드 무접촉) · FDHOLD · NT. L2 게이트 넷만: G-ID(계보·소스·러너 상수) · G-RUN(완주) · G-JAC(야코비안이 조립·소비됐는가 — 아니면 A 가 공허) · G-LND(착지 **측정**이 가능한가: 적용·분리·창 실재). G-LND 는 착지 **성공**이 아니다(성공은 E-1/분기). 중복 없음. L3 = 2 걸음(구현·selftest → 런 1). 더 적은 걸음: 388393 에 J/χ/tri 덤프·체크포인트 없음 ⟹ 오프라인으로 결합 뉴턴 착지를 못 잰다. L4 삭제·통합 후보: 새 env 신설 — **채택하지 않음**(`LUMINA_COUPLED_NEWTON` 기존, 388393 은 부재=OFF). JDUMP 로 오프라인 formal_solve 재현 — 이중 권위·SPLIT 과 같은 mini-Path-B. **채택하지 않음.** MAXCH·FDHOLD 삭제하지 않음(삭제도 src 편집). 판정 셸 다섯 유지: DRIVE 와 같은 착지 정의의 같은 창.

## 2. 기전 — 런 0회 확정분

- **대조군 s−g [388393 judge_output.json · 판정문 §2 · SPLIT §2 원행]**. x^{(k)} = epoch k 마지막 행 x_star, s_k=(x_map^{(k)}−x_map^{(k−1)})/(x_lag^{(k)}−x_lag^{(k−1)}), g=a/den, κ_diag=(s−g)/(1−g). K=7, ITER=6.

| 셸 | g_7 | s_7 | s−g | κ_diag | damp_7 | 비고 |
|---|---|---|---|---|---|---|
| 12 | 2.780568659 | 3.1439719425250967 | +0.36 | −0.20 | 0.03170800121 | \|s−g\| 소 |
| 16 | 0.2096581554 | −0.020633488432262568 | −0.23 | −0.29 | 0.68930287 | 진동 |
| 20 | 0.4081369659 | 0.3555552391667972 | −0.05 | −0.09 | 0.2066156222 | \|s−g\| 최소 |
| 24 | **8.189966386** | **−4.584583035960915** | **−12.77** | **1.78** | 0.4140611145 | 부호 반대, 폭주 셸 |
| 28 | 0.06706843776 | −0.20663429835432645 | −0.28 | −0.29 | 1.000 | 인접 폭주 |

  sh24 epoch 7 생산 행(census `%.9e`): x_lag=1.092563223e+02, x_map=4.072876518e+02, a=7.779456053e-04, g=8.189966386e+00, x_star=1.498327537e+02, root=1, c_newton=−2.415102421e+00, damp=4.140611145e-01. ⟹ 누락 채널은 **a 이득 이상 크기**이고 대각 스칼라(저장 λ* 의 S-채널)가 맵 감도를 재현하지 못함(부호 반대). 이 수가 Path B 가 넣어야 할 항의 크기 하한이다.

- **CMFGEN 형태 (발자국, 추측 아님).** `do_var_cont.f` 가 dCHI·dETA 를 BA 야코비안에 접어 dJ/dN. Lumina 대응:

  dχ_bf = σ_bf · dN                          (GEMM, ∂χ_bf/∂N=σ_bf 공짜)
  dη_bf = (n_i, n_i*, Φ) 경유               (이중 권위: 준위·단계 총량)
  dS = (dη − S dχ)/χ
  dJ = Λ_diag(λ*) · dS + Λ_off(tri_lo, tri_up) · dS + (∂J/∂χ)\|_S · dχ
  dΓ_phot = ∫ 4π σ_bf /(hν) · dJ dν
  dΓ 를 STEQ/BA(N×N, `nlte_solve_ion_shell`)와 스칼라 사다리에 **같은 N** 으로 접음
  단계 총량 = Σ 준위 (한 권위). 스칼라만 미분하고 행렬-해 준위를 그대로 두면 η_bf 가 불일치.

  Λ_diag 만 넣은 것이 DRIVE(g). 나머지 세 조각(∂J/∂χ|_S, tri_*, 준위-정합 dETA)이 s−g 의 후보. 이 단은 셋을 **형태로서 모두** 넣는다(분할 측정 없음).

- **라이브 루프 (동결 트리 grep 2026-09-21).** `cmfgen_run`(`lumina_cmfgen.c`): pass 루프 `cmfgen_assemble` → `cmfgen_solve_J`(binned `formal_solve_bin`; `LUMINA_CMF_LINERES` 미설정=챔피언) → `radeq_set_tri_response(cs.tri_lo, cs.tri_up, cs.tri_r)` → `radeq_set_line_re_source`(`cs.lambda_star`) → `cmfgen_fine_jbar` → **`cmfgen_commit_jnu`** → init 아니면 `lumina_r7_publish_and_solve_te`. `CMFGenState cs`(chi/eta/J/λ*/tri 버퍼)는 `cmfgen_run` 스코프에만. `coupled_newton_solve_all` 은 `lumina_plasma.c` 스텁(인자 전부 `(void)`). 호출은 cuda.cu 죽은 경로에만(`LUMINA_COUPLED_NEWTON` 게이트). 시그니처는 `lumina.h` 원형 그대로 — 뉴턴은 `g_tri_*`(`radeq_set_tri_response`)·`g_lre_lambda_star`(line_re_source)·**commit 된** `nlte->radfield_view` 로 소비. Γ 와 dΓ 는 같은 pass 의 commit 뷰를 읽는다(cs.J 를 추정기 우회로 쓰지 않음).

- **tri_* 저장 (ALI 변경 아님).** `formal_solve_bin` 은 Tlo/Tup 포인터가 있으면 이웃 응답을 쌓는다. 현재는 `LUMINA_CMFGEN_LAMBDA_TRI=1` 일 때만 포인터를 넘기고 Thomas ALI 로 J 를 푼다. Path B 는 COUPLED_NEWTON ON 에서 Tlo/Tup 을 넘겨 `cs.tri_*` 에 **기록**하되, J 갱신은 388393 과 같은 대각 ALI 식을 유지한다. `radeq_set_tri_response` 는 이미 매 pass 호출됨 — 버퍼가 0 이면 비국소 채널이 침묵(G-JAC `tri_persisted`).

- **tensor core.** 야코비안 **조립**만 FP32 저장·TF32 tensor-core(`lumina_pathb_jac.cu`, `bf_gemm`/`nlte_gemm` 계열: ∂χ_bf/∂N=σ_bf). **N×N 해는 FP64**(기존 STEQ 경로; 뻣뻣·TF32 위험). 정확 multi-GPU 솔버는 Λ* 미노출 ⟹ 조립은 binned `formal_solve_bin` 챔피언 경로에서만.

- **뉴턴 스텝·리미터.** 결합 뉴턴이 Δu(log x 또는 준위 log n)를 낸다. 보정 상한은 **기존** `LUMINA_POP_DJDN_CMAX`(388393 = 1.0, 신설 아님). c_newton = 감쇠 전 Δu(스칼라 판정 비의 log), damp = min(1, C_max/|c|), 커밋 x_star = x_map · exp(damp · c_newton) — 리미터 항등(DRIVE 와 같은 꼴, 1e-6 CSV 그레인). 상태 클램프·floor 금지. T_e 는 twin 핀(뉴턴 미지수 아님).

- **x_map 의 뜻.** x_map = commit 된 J 에서 평가한 Picard/SE 비(뉴턴 보정 전). x_lag = J 조립 세대의 커밋 비(기존 `ion_before` 스냅샷). s 의 정의는 DRIVE 와 동일. G_full = 완전 야코비안이 예측하는 dx_map/dx_lag(판정 셸, 최대 epoch). g = 대각 a/den(기존, 적용하지 않음).

- **비용.** 388393 실측 ITER 6 ≈ 13.6 h(a100×2, cpus 16) ⟹ ≈ 2.3 h/바깥반복. ITER 8 ≈ 18 h(`--time=24:00:00`) · ITER 10 ≈ 23 h(`--time=30:00:00`). 결합 뉴턴이 반복당 비용을 키울 수 있음 — 상한은 런이 정하고, `--time` 은 선택 ITER 의 1.3× 이상.

- **미확정(계약 밖).** 착지값의 CMFGEN 근접 · 준위별 점유와 CMFGEN BA 원소 단위 정합 · 정확 솔버 Λ* · 판정 셸 밖(Si·Ca·내핵 IGE).

## 3. 수리안 — 변경집합

env `LUMINA_COUPLED_NEWTON`(기존, 기본 미설정/0 = 스텁 no-op·jac 파일 없음·**생산 경로 byte-동일**, 388393 부재). ON(=1): §2 완전 야코비안 결합 뉴턴을 `cmfgen_run` 에서 `cmfgen_commit_jnu` **이후** 인구 갱신으로 적용. `LUMINA_POP_DJDN=1` 유지(생산 16 열 census). Path B ON 에서 스칼라 `pop_djdn_log_newton` **적용 금지**(g·a·x_map 는 기록). `LUMINA_POP_DJDN_CMAX=1.0` 유지(리미터). `LUMINA_CMFGEN_LAMBDA_TRI`·`LUMINA_CMFGEN_JDUMP`·`LUMINA_ION_NT`·`LUMINA_NT_FULL`·`LUMINA_POP_MAXCH`·`LUMINA_POP_DJDN_FDHOLD` **부재**. 상태 클램프·floor 금지. `coupled_newton_solve_all` 시그니처 불변.

jac 파일 `work/lumina_pathb_jac.csv` 헤더(순서 고정, 12 열), 행 = 판정 셸 다섯 × 최대 epoch 1개 = **5행**:

`shell,Z,epoch,g_diag,G_chi,G_tri,G_full,tri_persisted,dchi_assembled,commit_gen,n_newton,jac_res`

- `g_diag` = 생산 census 같은 (shell, 최대 epoch 마지막 gen) 의 `g` 와 `%.9e` 일치.
- `G_chi` = ∂J/∂χ|_S 채널이 기여하는 dx_map/dx_lag.
- `G_tri` = tri_* 비국소 채널 기여.
- `G_full` = 완전 야코비안 이득(대각 λ* 조각 포함).
- `tri_persisted` = 1(그 pass `cs.tri_lo`/`cs.tri_up` 가 형식해에서 기록됨) / 0.
- `dchi_assembled` = 1(dCHI 조립이 공짜 σ_bf GEMM 경로로 수행) / 0.
- `commit_gen` = 그 pass `cmfgen_commit_jnu` 의 `rad_generation`(뉴턴이 같은 세대 뷰를 읽음).
- `n_newton` = 결합 뉴턴 반복 수(≥1 이어야 적용).
- `jac_res` = 커밋점에서 선형화 잔차(유한; 잣대는 selftest, in-vivo 는 유한·비-NaN).

### 변경집합 — PATHB-FULL (표)

| 경로 | 담당 | 내용 |
|---|---|---|
| `src/lumina_cmfgen.c` | Opus 4.8 | (i) `formal_solve_bin`/`cmfgen_solve_J`: `LUMINA_COUPLED_NEWTON=1` 이면 Tlo/Tup 를 넘겨 `cs.tri_lo`/`cs.tri_up`/`cs.tri_r` 에 기록. J 갱신 식은 대각 ALI 유지(`LAMBDA_TRI` 켜지 않음). (ii) `cmfgen_run` 라이브 루프: `cmfgen_commit_jnu` **직후**(init_pass 아님) `coupled_newton_solve_all` 호출. 호출 전 commit 실패면 뉴턴 진입 금지. `lumina_cuda.cu` 무접촉 |
| `src/lumina_plasma.c` | Opus 4.8 | `coupled_newton_solve_all` 스텁 채움(시그니처 불변): g_lre_*·g_tri_*·commit 뷰로 Γ·dΓ 구성, `pathb_jac_assemble` 로 야코비안, N×N STEQ(`nlte_solve_ion_shell`)에 bf 소스 결합, 단계 총량=Σ준위, 스칼라 사다리는 같은 N 의 총량만 기록. Path B ON 에서 `pop_djdn_log_newton` 적용 스킵(census 기록은 유지, x_star=결합 뉴턴, c_newton/damp=리미터). OFF byte-동일. 클램프·floor 금지. T_e 미지수 아님 |
| `src/lumina_pathb_jac.cu` | Opus 4.8 | 야코비안 조립 커널 `pathb_jac_assemble`: FP32 저장·TF32 tensor-core GEMM(∂χ_bf/∂N=σ_bf, dETA 조립). N×N 해 호출 없음(해는 FP64 기존 경로). binned 격자만 |
| `tests/pop_djdn_selftest.c` | Opus 4.8 | 기존 케이스 유지(DRIVE 30 + SPLIT 추가분, FDHOLD OFF 경로) + **Path B**: (1) 합성 3셸 픽스처에서 ∂χ_bf/∂N == σ_bf bit-동일(1e-12) (2) 대각-only 조립 ⟹ G_full == g (1e-12) · tri/dCHI 켜면 G_full ≠ g (3) 이중 권위: 스칼라만 미분한 G 와 준위-정합 G 가 픽스처에서 갈림 FAIL 시연 (4) OFF(`LUMINA_COUPLED_NEWTON` 미설정) ⟹ jac 파일 없음·생산 경로 byte-동일 (5) **음성대조 1**: commit_gen 을 한 세대 어긋나게 주입 ⟹ G-JAC FAIL 시연 (6) **음성대조 2**: tri_persisted=0 또는 dchi_assembled=0 주입 ⟹ G-JAC FAIL 시연 (7) 리미터 항등 x_star = x_map·exp(damp·c) 1e-12(selftest) / 헤더 12 열 일치 |

### 변경집합 끝

★**src 편집 1개 규칙**: 가동 중 src-편집 태스크 0 확인 후 발주(DRIVE 폐합·SPLIT 폐기). ⚠프리플라이트: 표의 네 파일은 기안 시점 repo 에 `pathb_jac_assemble` 이 없으므로 `expected_extra` 로 선언(DRIVE/SPLIT 과 동일 처리). `path_pattern` 은 `.cu` 를 포함한다. **`src/lumina_cuda.cu` 는 표에 없다.** 기대 출력 `changeset-consistent files=4`.

## 4. 게이트 명부 (표)

| ID | 조건 | 음성대조 |
|---|---|---|
| G-ID | 새 런 exports 를 대조군 388393 계보(`/gpfs/kjhan/lumina/cmfgen_map/djdn_drive_20260920/run_h100a100/input/resolved_lumina.exports`)와 diff — 추가·변경 키가 정확히 `LUMINA_COUPLED_NEWTON`(=1) [§6 ① ITER 변경 시 `LUMINA_PURE_CMFGEN_ITER`; 프로필 종속 `LUMINA_CMF_FINE_MGPU_DEVICES` 허용]; `LUMINA_CMF_CONT_NLTE_SOURCE`(=1)·`LUMINA_POP_DJDN`(=1)·`LUMINA_POP_DJDN_CMAX`(=1.0)·`LUMINA_ION_CHANNEL_DUMP`(=1)·`LUMINA_FROZENIN_DR`(=1)·`LUMINA_DR_BOOST_AUTOSTRUCT`·`LUMINA_DR_BOOST_BADNELL`·`LUMINA_DR_BOOST_MAZZOTTA`·`LUMINA_DR_BOOST_NORAD`(각 =1)·`LUMINA_DET_TRANSACTIONAL`(=1) 유지; `LUMINA_POP_MAXCH`·`LUMINA_ION_NT`·`LUMINA_NT_FULL`·`LUMINA_RADEQ_SIMUL`·`LUMINA_SIMUL_ION_DAMP`·`LUMINA_CMFGEN_JDUMP`·`LUMINA_CMFGEN_LAMBDA_TRI`·`LUMINA_POP_DJDN_FDHOLD`·`LUMINA_CMF_LINERES` **부재**; 동결 트리 `/gpfs/kjhan/lumina/cmfgen_map/lteunmap_eps_20260917/source.sha256` 의 `sha256sum -c` non-OK 가 정확히 §3 표의 파일(`lumina_cuda.cu` 가 non-OK 이면 FAIL); 새 바이너리 해시 `input/binary.sha256` 기재(대조군 `7d42780346bf5d2c...` 와 달라야 함); 러너의 **ITER 상수 셋** — guard `"${LUMINA_PURE_CMFGEN_ITER:-}" == N` · `outer_iterations=N` · postcheck `-ne N` — 이 전부 선택 ITER 와 같음을 `grep -nE 'PURE_CMFGEN_ITER:-\}" == \|outer_iterations=\|-ne [0-9]+ \]\]; then' job.slurm` 출력으로 STAGE_IDENTITY 에 기재; 러너 편집 = job-name·`--time`·ITER 상수 셋·프로필(partition/gres/GPU_PROFILE/cpus, §6 ②)뿐, `runner_edits` 에 `postcheck_te_count` 포함 | 다른 키 혼입·§3 밖 소스 변조(특히 `lumina_cuda.cu`)·postcheck `-ne` 가 ITER 와 불일치·LAMBDA_TRI/JDUMP/NT/FDHOLD 혼입 ⟹ FAIL(388393 러너 그대로 = COUPLED_NEWTON 키 부재로 FAIL) |
| G-RUN | `RUN_FOOTER.txt` `terminal_state=SUCCESS`·`model_rc=0`·`outer_iterations`=선택 ITER; `work/lumina_pop_djdn.csv`·`work/lumina_pathb_jac.csv`·`work/lumina_ion_channels.csv`·`work/physics_comparison/physics_DET_iter<ITER−1>.ion.csv` 실재 | 미완주·jac 결손 ⟹ FAIL; 388393 = jac 파일 없음 ⟹ FAIL; 387175 = 생산 census 13 열·jac 없음 ⟹ FAIL |
| G-JAC | (오프라인) `tests/pop_djdn_selftest.c` rc=0, §3 음성대조 둘 다 FAIL 시연 — **봉인 접촉 전 grok 코드 간단 감사 선행**(개정16 ②; 개정25-a Fable 자리는 user 2026-09-21 로 grok 대행); 합성 픽스처 ∂χ_bf/∂N==σ_bf 1e-12, 대각-only G_full==g 1e-12. (in-vivo jac) 헤더 12 열 순서 일치·데이터 5행·각 판정 셸 1행·epoch=ITER+1·Z=16; 전 행 `tri_persisted=1` ∧ `dchi_assembled=1` ∧ `n_newton`≥1 ∧ `jac_res` 유한; `commit_gen` 이 그 런 최대 epoch 의 생산 `rad_generation` 과 같음(같은 정수); `g_diag` 가 생산 census 같은 (shell, 최대 epoch 마지막 gen) `g` 와 `%.9e` 일치. **G_chi·G_tri·G_full 의 값 자체는 이 게이트의 문턱이 아님**(E-2). 1e-9 금지. FD/맵 재평가 잣대를 CSV 1e-6 에 복사하지 않음 | jac 헤더 바꿔치기 ⟹ FAIL; tri 미저장 빌드 ⟹ tri_persisted=0 FAIL; dCHI 미조립 ⟹ dchi_assembled=0 FAIL; commit 건너뛴 뉴턴 ⟹ commit_gen 불일치 FAIL; 388393 자기검사 ⟹ jac 부재 FAIL |
| G-LND | (생산 census) 헤더 16 열 = DRIVE 와 동일(`epoch,c_newton,damp` 포함). epoch 구조: 같은 (Z,shell,epoch) 안 x_lag 불변, epoch 단조 증가, 각 판정 셸에서 epoch k+1 첫 행의 x_lag = epoch k 마지막 행의 x_star(`%.9e` 일치), 최대 epoch = ITER+1; 최대 epoch 의 각 gen 에 판정 셸 다섯(Z=16, 셸 12·16·20·24·28) 행이 각각 정확히 1개 실재; 판정 셸 epoch ≥ 2 전 행 root=1(결합 뉴턴 적용) ∧ x_lag>0 ∧ x_map>0 ∧ x_star 유한; 리미터 항등 행 집합 = root=1 ∧ damp>0 인 전 행: \|x_star − x_map·exp(damp·c_newton)\| ≤ 1e-6·x_star **and** damp = min(1, C_max/\|c_newton\|)(1e-6) [388393 CSV 그레인; **c' = lnρ·g/(ρ−g) 대각 log-뉴턴 재계산은 적용하지 않음** — Path B 의 c 는 완전 야코비안 Δu]. damp==0 행은 항등에서 제외·행 수와 (Z,shell,epoch) 기재·판정 셸 마지막 두 epoch 에 있으면 E-1 정의로 착지 불가. (분리) 새 런 커밋 x^{(k)} 와 대조군 388393 커밋 x_ctrl^{(k)} 가 sh24 또는 sh28 의 **양쪽 런에 존재하는** epoch 4 또는 5 또는 7 중 어느 하나에서 \|log10 x^{(k)} − log10 x_ctrl^{(k)}\| > 0.05. ⚠G-LND 통과 ≠ 착지 성공 | 생산 census 열 바꿔치기 ⟹ FAIL; root=0 빌드(스텁 잔존) ⟹ 적용 실패 FAIL; 대각 log-뉴턴을 그대로 커밋하는 빌드 ⟹ 분리 0 이거나 리미터 항등은 통과해도 G-JAC 채널 비트로 갈림; 388393 자기검사 ⟹ COUPLED_NEWTON 부재·jac 부재 FAIL |

### 게이트 명부 끝

## 5. 기대치 — 런에서만 얻는 수 (잣대를 대조군에 먼저 적용: 388393 은 jac 파일이 없고 착지 창을 넘지 못한다 — sh24 Δ_7=+0.137, damp_7=0.414)

x^{(k)} = 판정 셸 epoch k 마지막 행 x_star, k = 1 … K, K = ITER+1. Δ_k = log10 x^{(k)} − log10 x^{(k−1)}, k = 3 … K(epoch 1→2 는 씨앗 완화라 제외). s,g 정의는 §2(DRIVE 와 동일). G_full 은 jac 파일. x_lag 불변(\|denom\|/max(\|x_lag^{(K)}\|,\|x_lag^{(K−1)}\|,1) < 1e-3) 이면 그 셸 s 미정의 → NaN 기재, 착지 투표에서 제외. \|1−g\|<1e-3 이면 κ_diag 미정의, \|1−G_full\|<1e-3 이면 κ_B 미정의 → NaN 기재.

| ID | 양 | 대조(388393) | **기대·역할** |
|---|---|---|---|
| E-1 착지 | 판정 셸 다섯의 Δ_{K−1}·Δ_K 와 그 두 epoch 의 damp·root | Δ_6/Δ_7: sh12 −0.003/−0.004 · sh16 +0.280/−0.880 · sh20 −0.066/−0.076 · sh24 −0.086/+0.137 · sh28 −0.700/+0.282; 착지 0/5 | **판정량**(분기 표). 착지 = \|Δ_{K−1}\|≤0.05 ∧ \|Δ_K\|≤0.05 ∧ 그 두 epoch damp=1 ∧ root=1 |
| E-2 s−G | 판정 셸의 s_K, g_K, G_full, G_chi, G_tri, κ_diag, κ_B | §2 표(G_full 없음). sh24 κ_diag=1.78 | 판정 조건 아님 — **기재**. 정합 기재: 착지 셸 \|κ_B\|<1 / 진동 셸 κ_B<−1 / 단조 상승 셸 κ_B>1. 정합 위반은 «기재». sh24 에서 \|G_full−g\| 가 \|s−g\|(=12.77) 과 같은 자릿수인지는 기재(게이트 아님) |
| E-3 감쇠·뉴턴 | census epoch ≥ 2 전 행(Z 14·16·20) 중 damp<1 행 수와 (Z,shell,epoch); 판정 셸 K−1·K 의 damp; jac `n_newton` | 해당 없음(구 바이너리 jac 없음). 생산 damp 는 §2 | 기재. 판정 셸 마지막 두 epoch 에 damp<1 이면 그 셸은 착지 불가(정의) |
| (기재) | `Z16_nlte_q2/q1`(NLTE 권위 r) · Si 셸 12·20 · Ca f_II · 외곽 S f_III 셸 28·32·36 · Fe f_IV 셸 0–3 · 388393 대비 같은 epoch 의 log10 x 차 | 388393·387175 | 판정 조건 아님 |

문턱 근거: 수렴 창 0.05 dex = DRIVE 와 동일(CONT 1 반복 이동 ≥0.13 의 절반 이하). 상승 하한 0.10 = DRIVE 와 동일. 대조군 분리 0.05 dex = 같은 창(FP 미세변동은 1e-3 dex 급). 기안자 예상(판정 조건 아님·확률 미부여): 형태가 맞으면 sh24 의 \|κ_B\| < \|κ_diag\| 이고 G_chi·G_tri 가 0 이 아니다. 착지 자체는 비선형 맵·ITER 창의 문제라 예상을 강건하다고 하지 않는다 — 런이 정한다.

### 분기 (표)

셸별 코드: 게이트 미충족 ⟹ 전 셸 0. 게이트 충족 시 판정 셸마다 — **1 착지**(E-1 정의) / **2 진동**(착지 아님 ∧ Δ_{K−1}·Δ_K < 0 ∧ min(\|Δ_{K−1}\|,\|Δ_K\|) > 0.05 ∧ \|Δ_K\| ≥ \|Δ_{K−1}\|) / **3 단조 상승**(착지 아님 ∧ Δ_{K−1} > 0.10 ∧ Δ_K > 0.10) / **4 기타**(감쇠 상한 걸린 마지막 epoch·감쇠 진동·단조 하강·root=0 등).

| 분기 | 조건 | 의미 |
|---|---|---|
| PATHB-X | G-ID/G-RUN/G-JAC/G-LND 중 하나라도 미충족 | 미시연 |
| PATHB-A | 게이트 전건 **and** 판정 셸 다섯 전부 코드 1 | **완전 Path B 결합 뉴턴이 CONT 폭주 셸을 유한 고정점에 착지**. 수렴값의 CMFGEN 근접·\|κ_B\|<1 정합은 기재 |
| PATHB-B | 게이트 전건 **and** 코드 2 셸 ≥ 1 **and** 코드 3·4 셸 0 | **진동 발산** — G_full 이 s 를 과대(또는 부호). 다음 최소 수리는 이 계약이 열지 않음(재분기 금지, 판정문이 명명) |
| PATHB-C | 게이트 전건 **and** 코드 3 셸 ≥ 1 **and** 코드 2·4 셸 0 | **단조 상승 지속** — 결합 뉴턴이 맵을 아직 수축시키지 못함(ITER 부족은 E-3·장부에 기재, 분기 재해석 금지) |
| PATHB-D | 게이트 전건 **and** (코드 4 셸 ≥ 1 **or** 코드 2·3 셸이 함께 있음) | 혼재 — 판정자가 셸별 κ_B·damp 로 명명 |

### 분기 끝

재분기 금지. 판정기 = `judge_drive.py` 계보의 운전자 도구를 **jac 5행·§4 G-JAC/G-LND·§5 코드**로 개정한 것(해시 `READOUT_TOOLS.sha256` 기재; 소스 아님). 음성대조: 388393 자기검사 ⟹ G-RUN(jac 부재)·G-JAC·G-LND FAIL ⟹ PATHB-X; 387175 ⟹ 생산 13 열·jac 없음 ⟹ PATHB-X.

## 6. 자원 · 실행 · **결정 항목(운전자·user)**

새 RUN 루트 `/gpfs/kjhan/lumina/cmfgen_map/pathb_full_20260921/run1`(형제 루트), 입력 = 388393 입력 복사 + 새 빌드 + exports 편집(G-ID). GPU 1 런. 봉인 base·대조군 읽기만.

① **ITER** — **8**(≈18 h, `--time=24:00:00`, K=9, 판정 epoch 8·9) 또는 **10**(≈23 h, `--time=30:00:00`, K=11, 판정 epoch 10·11). DRIVE ITER=6 은 착지 창에서 0/5 였고, 퇴화 장부는 고정점 도달에 8–10 을 시사한다. ITER 6 재사용은 같은 창에서 「뉴턴만 바꾸고 반복은 그대로」가 되어 PATHB-C/D 와 ITER 부족을 가르지 못할 수 있다. 기안자는 못 박지 않는다. **고른 N 을 러너 상수 셋 세 곳과 `LUMINA_PURE_CMFGEN_ITER` 에 같은 숫자로 적을 것**(§0(e)·G-ID).
② 프로필·헤지 — 직전 단과 동일(먼저 시작하는 쪽만; 멀티파티션·cpus 16 편집은 G-ID 에 기재). GPU-bound·byte-배터리 없음.
③ NT 키 부재·MAXCH 미설정·JDUMP 미설정·LAMBDA_TRI 미설정·FDHOLD 미설정(388393 그대로 + COUPLED_NEWTON=1 만 추가).
④ C_max=1.0 유지(새 결정 아님, G-ID 키 유지).

순서: 운전자 기계 게이트 4종(prereg-preflight `files=4`·gate_referents·doc_canonical·dispatch-preflight, 전건 rc=0) → **grok 물리·대수 사전감사(짓기 전, fresh)** → Opus 4.8 경량 사전감사(Q-GOAL·Q-LEAN) → Opus 4.8 구현(동결 트리 복사본) → selftest → grok 코드 간단 감사(첫 항목 = 봉인 접촉 가부; cuda.cu 무접촉·commit_jnu 이후 호출·시그니처 불변) → 재봉인·빌드 → 스테이징(**러너 ITER 상수 셋 grep 확인**) → 제출 → 판독 → 판정(운전자) → 종료감사(Opus 4.8 fresh; GPU 지출·물리 주장이 걸린 단이라 생략하지 않음).

## 7. 분장 장부 (개정25-e · 규약 / 실제 / 위반)

| 단계 | 규약 | 실제 | 위반 |
|---|---|---|---|
| 기안 | Fable fresh(25-b) | **Fable 대행**(`grok-4.6`, user 2026-09-20 지시) | 규약 담당 ≠ 실제 모델(Fable 토큰 소진, 기재) |
| 물리·대수 사전감사(짓기 전) | grok fresh(user 2026-09-21, 개정25-e) | 미집행 | |
| 경량 사전감사(Q-GOAL·Q-LEAN) | Opus 4.8 fresh(25-d) | 미집행 | |
| 구현 + selftest | Opus 4.8 | 미집행 | |
| 코드 간단 감사 | Fable fresh(25-a) → grok 대행(user 2026-09-20/21) | 미집행 | ⚠기안 대행자 = 감사 대행 후보(자기 계약 편향, 개정25-b 기재) |
| 스테이징·제출·판독·판정 | 운전자 | 미집행 | |
| 종료감사 | Opus 4.8 fresh | 미집행 | |

## 8. 주장 상한

말할 수 있는 것: 「CMFGEN 식 완전 ∂J/∂N 선형화를 라이브 `cmfgen_run` 결합 뉴턴으로 적용했을 때 황 II/III 비가 판정 셸에서 유한 고정점에 착지하는가와 그 꼴」, 「그 국면의 in-vivo s, g, G_full, κ_diag, κ_B(대각 대비 개선의 크기)». 착지해도 CMFGEN 근접은 기재 사항이지 성공 조건이 아니다. **s−g 를 국소/비국소로 다시 가르지 않는다** — 이 단은 둘을 형태로서 넣었다. 「COUPLED_NEWTON 토글 단독」은 OFF 런이 없으므로 selftest byte-동일 + 코드 감사의 오프라인 주장 조건부. 말하지 않는 것: 정확 솔버 Λ* · Milne 소스 · Si/Ca · 스펙트럼 · 자유-T_e · NT · Thomas ALI. twin-T_e 핀 조건부.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 — PATHB-FULL (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "(?:src|tests)/[A-Za-z0-9_/.]+\\.(?:c|h|cu)",
    "symbol": "pathb_jac_assemble",
    "roots": ["src", "tests"],
    "expected_extra": ["src/lumina_cmfgen.c", "src/lumina_plasma.c", "src/lumina_pathb_jac.cu", "tests/pop_djdn_selftest.c"]
  },
  "branches": {
    "regimes": [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
    "metrics": {"count": "len(v)", "gate_fail": "sum(1 for x in v if x < 0.5)", "landed": "sum(1 for x in v if 0.5 <= x < 1.5)", "osc": "sum(1 for x in v if 1.5 <= x < 2.5)", "runaway": "sum(1 for x in v if 2.5 <= x < 3.5)", "other": "sum(1 for x in v if x >= 3.5)"},
    "rules": [
      {"name": "PATHB-X", "predicate": "count == 0 or gate_fail > 0"},
      {"name": "PATHB-A", "predicate": "count > 0 and gate_fail == 0 and osc == 0 and runaway == 0 and other == 0"},
      {"name": "PATHB-B", "predicate": "count > 0 and gate_fail == 0 and osc > 0 and runaway == 0 and other == 0"},
      {"name": "PATHB-C", "predicate": "count > 0 and gate_fail == 0 and runaway > 0 and osc == 0 and other == 0"},
      {"name": "PATHB-D", "predicate": "count > 0 and gate_fail == 0 and (other > 0 or (osc > 0 and runaway > 0))"}
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
    "residual": "PATHB-X"
  },
  "references": [
    {"path": "docs/VERDICT_DET_DJDN_DRIVE_2026-09-21.md"},
    {"path": "docs/PREREG_DET_DJDN_DRIVE_2026-09-20.md"},
    {"path": "docs/PREREG_DET_DJDN_SPLIT_2026-09-21.md"},
    {"path": "docs/DESIGN_PATHB_LINEARIZATION_2026-09-19.md"},
    {"path": "CLAUDE.md"},
    {"path": "src/lumina_plasma.c", "flags_existing": ["parity_rate_se_ratio_checked", "compute_ion_populations_shell", "radeq_set_tri_response", "coupled_newton_solve_all", "nlte_solve_ion_shell", "g_lre_lambda_star", "g_tri_lo"], "flags_planned": ["pathb_jac_assemble"]},
    {"path": "src/lumina.h", "flags_existing": ["coupled_newton_solve_all", "radeq_set_tri_response"], "flags_planned": []},
    {"path": "src/lumina_cmfgen.c", "flags_existing": ["cmfgen_commit_jnu", "cmfgen_run", "formal_solve_bin", "lambda_star", "tri_lo", "tri_up"], "flags_planned": []},
    {"path": "src/env_universe.h", "flags_existing": ["LUMINA_COUPLED_NEWTON", "LUMINA_CMFGEN_LAMBDA_TRI", "LUMINA_ION_NT", "LUMINA_CMFGEN_JDUMP"], "flags_planned": []}
  ]
}
```

## 9. 집행 기록

1. **기계 게이트 3종(2026-09-21, 운전자 grammar-debug, 발주 전)**: **전건 rc=0**.
   - `check_prereg_preflight`: `changeset-consistent files=4` · `branch-partition trials=200000 rules=5 fixtures=7 residual=PATHB-X` · `references-resolved refs=9` · `prereg-preflight PASS`.
   - `check_gate_referents`: PASS `changeset_exempt=4`.
   - `check_doc_canonical`(인용 4문서): 전건 `single` PASS.
   - dispatch-preflight 는 발주서 작성 후(스테이징 단계) 집행.
2. **grok 물리·대수 사전감사(짓기 전, fresh — 개정25-e 첫 적용, 2026-09-21)**: **«폐합 차단» 7건 ⟹ 구현 발주 정지·재기안**. (grok 이 기안자 = 자기 계약 자기 감사, 한계 기재; 그럼에도 기계 게이트 4종이 통과시킨 물리·배선 결함을 잡음 — 개정25-e 근거 실증.)
   - ①**«차단» ∂J/∂χ|_S 생산자 부재**(§2 dJ 체인): `pathb_jac_assemble` 은 dχ 만 조립, ∂J/∂χ 아님 — 이름만 있고 이산화 없음(09-05 E-1 동어반복 계급). `Λ_off(tri)` 도 최근접 이웃뿐(`lumina_cmfgen.c:2945-2955`), 전각 BA 변분 아님. 빠진 항·중복 항(dS 안의 dχ 와 겹침)·오전제 동시.
   - ②**«차단» 이중 권위 미해소**: `compute_ion_populations_shell` 이 여전히 단계 소유(`:3378-3379`), 계약은 쓰기 순서(`ion_number_density` vs 준위 vs η_bf) 미지정 — 슬로건.
   - ④**«차단» tri 저장이 J 를 바꿈**(③ 재발): `formal_solve_bin` 포인터가 살면 접선항이 **대각 L\*** 에 더해져(`:2945-2951`) J 가 388393 과 달라짐 — 「대각 ALI 유지」는 코드로 거짓. §0(d) 금지가 이 우회로 재발.
   - ⑥**«차단» 뉴턴 호출 슬롯이 죽음**(③ 인구판): `cmfgen_run` 에 `coupled_newton_solve_all` 호출 0건. commit_jnu 직후 = r7/A2-10 인구 커밋 **앞** ⟹ A2-10 사다리가 덮어씀(DJDN-MVP 죽은 `radeq_simul_all` 배정과 같은 계급). 뉴턴은 A2-10 인구 커밋 슬롯에 들어가야.
   - ⑦**«차단» pathb_jac.cu 미링크**(① 계급): Makefile `lumina_cuda` 링크 목록에 없음(`:65-79`)·Makefile 이 변경집합에도 없음 ⟹ 라이브 빌드가 안 타는 파일에 조립 커널.
   - ⑧**«차단» 음성대조 구멍**: 「대각 log-뉴턴 그대로 커밋」이 G-LND 를 통과할 수 있음(분리·jac 비트 둘 다 위조 가능) — 직전 3계약 배선오류 구멍과 동일.
   - ⑩**«차단» 「형태-확실」 반증**: binned Λ·최근접 tri 절단·미지정 ∂J/∂χ·TF32·LTE/NLTE 혼재 ⟹ 결과 형태·크기 미지. SPLIT 이 폐기한 분할 자유도가 이 단에 **숨은 채** 남음. Q-GOAL④ 실재. ★**user 「형태-확실」 전제와 감사 결론이 충돌 — 방향 결정 사안(user 보고).**
   - «기재»: ③ 대각 스킵 대수 정확·⑤ 시그니처 불변 OK·⑨ 리미터 항등 등 구성상 참(G-JAC 판정치 문면 과대)·⑪ 묶음.
   - grok 최소 재기안 표적 6: (i) ∂J/∂χ 생산자·이산화 또는 항 삭제 (ii) 이중권위 쓰기 순서 (iii) tri 저장을 L\*/J 와 분리 (iv) 뉴턴을 A2-10 인구 커밋 슬롯에 (v) pathb_jac.cu 링크 정합(Makefile 변경집합) (vi) 대각 잔존·cs.J 우회가 실제 FAIL 하는 음성대조.
   - **처분: 정지. ①②④⑥⑦⑧ 은 재기안 수리 가능(계약-공작·배선). ⑩ 은 방향 결정(user).**
3. **경량 사전감사(Opus 4.8 fresh, Q-GOAL·Q-LEAN)**: 미집행.
4. **§6 결정(운전자·user)**: 미집행 — ①ITER ∈ {8,10} ②프로필·헤지 ③NT/JDUMP/LAMBDA_TRI/MAXCH/FDHOLD 부재 ④C_max=1.0 유지.
5. **구현(Opus 4.8) + selftest**: 미집행.
6. **grok 코드 간단 감사**: 미집행.
7. **재봉인·빌드·스테이징·제출**: 미집행.
8. **착지·판독·판정**: 미집행.
9. **종료감사(Opus 4.8 fresh)**: 미집행.