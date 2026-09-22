# 사전등록 — DET-PATHB-LOCAL: DRIVE 이득에 고정-S `∂J_fs/∂χ` 한 항을 더해 A2-10 뉴턴에 넣으면, 그 기여와 CONT 폭주 셸 착지가 이 창에서 어떻게 바뀌는가 (2026-09-21)

기안 = **Fable 대행**(`grok-4.6`, user 2026-09-20 Fable 토큰 소진 이후 계약 기안 대행). 상태 = **코드 감사 통과·봉인 접촉 가(astra 재감사 2026-09-21) — 재봉인·빌드·스테이징·판정런 대기(ITER=6)**. 두 사전감사·코드 감사(astra 대행) 전건 통과. 실행 승인이 아니다. 개정 = **astra 교차감리 5건 반영**(user 2026-09-21 「수정된 LOCAL 진행·도함수=고정-S ∂J_fs/∂χ」).
근거 = `docs/PREREG_DET_PATHB_FULL_2026-09-21.md`(**폐기** — grok 물리·대수 사전감사 «폐합 차단» 7건, 특히 ⑩ 「형태-확실」반증; user 2026-09-21 결정 B 「접지된 작은 단 먼저」) · astra 교차감리(`gpt-6-astra`, Codex 계열, 재현 렌즈, 2026-09-21; FULL 정지 지지·「국소 실패⇒비국소」기각·도함수를 고정-S `∂J_fs/∂χ` 로 못박음) · `docs/VERDICT_DET_DJDN_DRIVE_2026-09-21.md`(직전 판정 정본: DRV-D, 대각 단독 불충분 **측정**·s−g; §3/§30 국소/비국소 분할 미주장) · `docs/PREREG_DET_DJDN_DRIVE_2026-09-20.md`(형식·이득·log-뉴턴 구조) · `docs/DESIGN_PATHB_LINEARIZATION_2026-09-19.md`(근인 배선·`do_var_cont.f` 발자국). **base = 대조군 = DRIVE 판정런 388393**(`/gpfs/kjhan/lumina/cmfgen_map/djdn_drive_20260920/run_h100a100`, log-뉴턴 대각 항 ON, ITER=6, 16 열 census, binary `7d42780346bf5d2c...`). 채널 OFF = 그 런의 대각만. 소스 기준 = 동결 트리 `/gpfs/kjhan/lumina/cmfgen_map/lteunmap_eps_20260917/source`(DRIVE 변경집합 + SPLIT-substrate OFF-게이트 포함, manifest 185 파일·repo 미이식). 최종목적 = MC 수송으로 CMFGEN 수준 SN Ia 스펙트럼 재현.

## 0. 기안자 검사 — 직전 결함 정정 (자기 «폐합 차단» 7건을 이 단이 없애는 방식)

폐기된 FULL 계약(`docs/PREREG_DET_PATHB_FULL_2026-09-21.md` §9 집행 기록 2)의 7건. 번호는 감사 원번호를 유지한다(3·5·9 는 «기재»라 여기 없음).

**(① ∂J/∂χ|_S 생산자 부재 → 이 단의 핵심 물리 내용.)** FULL 은 `pathb_jac_assemble` 이 dχ 만 조립하고 `(∂J/∂χ)|_S` 이산화가 없었다(이름만, 09-05 E-1 계급). 이 단의 **유일한 신설 물리** = binned `formal_solve_bin`(동결 `lumina_cmfgen.c:2848-2969`) 과 **같은 광선·같은 세그먼트·같은 작은-τ 분기·같은 핵 경계 재설정·같은 비핵 접선 귀환**을, **고정 S=S_fixed** 에서 χ 에 대해 미분한 재귀 `∂J_fs/∂χ`(§2). 별도 버퍼에만 누적한다(생산 I/Jacc/Lacc 무접촉). 0 으로 두거나 `-S (dχ)/χ` 를 a_ds 위에 더하거나, 초안의 L* 붕괴식 `-(τ*/χ)(J−S)` 로 바꾸면 FAIL(§4 음성대조). CMFGEN `do_var_cont.f:171-172` 의 분리(F2DA=`dJ/dCHI`, FC=`dJ/dETA`)를 **형태로서** 따른다 — Lumina 에서 FC 대각은 이미 DRIVE(`λ* ∂S/∂x`), 이 단은 F2DA 의 **같은-깊이 대각**만. PERTJFEAU Thomas 전각·NUM_BNDS 이웃은 이식하지 않는다(다른 J 솔버; tri 이 단 밖). ⚠이것은 커밋 J 의 완전 국소 도함수가 아니다(§2).

**(② 이중 권위 → 이 단은 열지 않음.)** N×N rate 행렬·준위 분포·`nlte_solve_ion_shell`·단계 총량=Σ준위 쓰기를 **건드리지 않는다**. 항은 스칼라 사다리 단계-총량 이득 a 에만 더해진다. 주장 상한(§8) = 「단계-총량 스칼라 이득에 정의된 국소 χ 응답 한 항을 더했을 때의 기여·착지 변화만 측정, 준위-정합 아님」. selftest: 준위 벡터 무접촉·CHI OFF 생산 경로 byte-동일. 범위 밖 선언이 준위·n_e 효과를 궤적에서 지우지는 않는다(착지 실패를 비국소로 읽지 않는 이유, astra).

**(④ tri 저장이 J 를 바꿈 → tri 를 이 단에서 켜지 않음. 형식해 본체 무변경.)** `formal_solve_bin` 에 Tlo/Tup 포인터를 넘기지 않는다. `LUMINA_CMFGEN_LAMBDA_TRI` 미설정(388393 과 같음). J 갱신식은 `:3093-3098` 대각 ALI 그대로 — 같은 입력 상태에서 같은 J 연산(388393 과 **동일**; 새 이득으로 인구가 달라지면 다음 epoch 의 χ·S·J 가 달라지는 것은 정상). `(∂J_fs/∂χ)|_S` 는 **별도 버퍼 재귀에서만** 얻고 생산 I/Jacc/Lacc 에 쓰지 않는다. `formal_solve_bin` 본체·ALI 루프는 수정하지 않는다. 신설 `formal_djdchi_fs_S` 는 그 광선 구조를 **복사**해 δI/δJ 버퍼에만 쓴다. `lumina_cmfgen.c` 는 이 신설 함수·등록 호출 때문에 변경집합에 **있다**(④ 함정은 형식해·L* 변경이지 파일 접촉 자체가 아님). `lumina_cuda.cu`·Makefile 은 없다.

**(⑥ 뉴턴 호출 슬롯이 죽음 → A2-10 라이브 슬롯.)** `coupled_newton_solve_all` 을 **쓰지 않는다**(cuda.cu 죽은 경로 `:8580-8582`·`:10956-10958` 에만 호출; `cmfgen_run` 호출 0). 국소 항은 DRIVE 의 `pop_djdn_log_newton`(`lumina_plasma.c:2718-2762`)에 **a 를 a_local 로 바꿔** 적용한다. 호출점은 `parity_rate_se_ratio_checked` 안 `:3317-3323`(A2-10 인구 커밋 슬롯). astra 확인: P:3317 뉴턴 → P:3323 r=x_star → P:3360 ratio_out → P:3548 사다리 소비 → P:3611-3621 후보 단계 인구 → P:17161 commit_bundle → P:22829 공개 인구. `pop_djdn_log_newton` 은 최종 커밋 함수가 아니라 최종 커밋으로 이어지는 비율 생성 슬롯이다. 「commit_jnu 직후」의 죽은 슬롯이 아니다. `lumina_cuda.cu` 무접촉. 시그니처 불변.

**(⑦ 새 .cu 미링크 → 새 .cu 를 만들지 않음.)** `(∂J_fs/∂χ)|_S` 는 기존 링크 파일(`lumina_cmfgen.c` 신설 정적 함수 + `lumina_plasma.c` 이득 조립)에서 계산. Makefile 무접촉. GEMM 불요(이 단의 제약).

**(⑧ 음성대조 구멍 → 대각-only 가 실제 FAIL. 뉴턴 소비를 수치로 묶음.)** 「대각 이득 그대로(`∂J_fs/∂χ` 항=0)」빌드가 G-ASM 을 **실제로 FAIL** 하도록 게이트·selftest 를 설계한다. 388393 재현 = 채널 OFF = FAIL. 채널 ON 인데 항 0(`a_dchi≡0`) = FAIL. commit_gen·비트 자기보고만으로 통과하지 않는다. **추가(astra 최소 검증):** a_dchi 가 실제 뉴턴 입력에 소비됐는지를 수치 사슬로 묶는다(§4 G-ASM: 생산 `a`/`g`/`c_newton`/`x_star` 가 a_local 대수와 맞을 것 — a_ds 대수와 맞으면 소비 실패).

**(⑩ 형태-확실 → 단일 채널이라 형태가 좁고 접지됨. 판별이 아님.)** 이 단은 완전판이 아니다. 항 하나(고정-S `∂J_fs/∂χ` 국소 대각). tri·N×N·이중 권위·ALI/산란 반응은 이 단 밖. Q-GOAL④ 는 「이 채널의 기여를 쟀다」로 닫힌다. 착지 못 해도 「비국소가 원인/필요」가 **측정으로 열리지 않는다**(잔차에 준위 분율·n_e·산란·ALI·타 셸·비선형·감쇠가 남음, astra 기각).

**(astra 교차감리 정정 — 초안 framing 5건, 이 개정에서 없앰.)**
1. sh24 s−g=−12.77 의 s 는 두 epoch 시컨트(타 셸·준위·n_e 동시 이동의 총미분)이다. 고정상태 편미분과 동일시 불가·채널 크기 하한 아님. DRIVE 판정문 §3/§30 도 국소/비국소 분할을 주장하지 않는다. s−g 표는 대각이 맵 감도를 재현하지 못한 **대조 맥락**으로만 남긴다. 그것으로 a_dchi 의 필요·크기를 정당화하지 않는다.
2. 「국소 χ 단독 실패 ⇒ 비국소 필요」는 기각. 이 단은 판별기가 아니다. 주장 상한 = DRIVE 에 정의된 국소 χ 응답 한 항을 더했을 때의 기여·착지 변화. 착지 실패 상한 = 이 국소 χ 항 추가로 이 창에서 착지 못 함.
3. 도함수 = 고정 S=S_fixed 에서 원시 형식해 J_fs 의 χ 도함수. 생산 J 의 ALI·산란 반응은 넣지 않는다(넣으면 형식해·L* 를 건드려 ④).
4. 기존 `pop_djdn_a_reduce`(`:2701`)는 양수 항만 합산한다. a_dchi 를 그 필터에 넣으면 음의 기여가 사라진다. 양수-필터 밖으로 라우팅한다. 뉴턴 함수는 음수 a 를 배제하지 않는다(`:2732-2744`).
5. χ 응답은 `x_lag` = J 조립 세대 스냅샷(`:21895-21902`)과 같은 상태·격자·소스 정의에 대응한다.

**(그 밖, 직전 3연속 재발 방어 — FULL §0(b) 계승.)** 오케스트레이션 = 라이브 `cmfgen_run` 의 A2-10 슬롯(`lumina_plasma.c`). 장 되먹임 경계 `cmfgen_commit_jnu` 는 이 단이 호출·우회하지 않는다(Γ 는 기존처럼 커밋 뷰를 읽고, 이득은 같은 pass 의 저장 `g_lre_*` 및 같은 세대 `dJ_fs/dχ`). referent·허용오차 = 388393 census 실측: CSV 재계산 항등 그레인 최대 **2.16e-8 rel** ⟹ 항등 허용오차 **1e-6**(1e-9 금지). 리미터 항등 `x_star = x_map·exp(damp·c)` 는 DRIVE 와 같은 꼴. 「행수 150」은 기재값(gen당 150행 = Z 14·16·20 × 셸 0–49)이지 요구값이 아님. **재등록 금지**: `g=a/den`·`root=1` 은 DRIVE 가 이미 닫음. 단 a 정의가 a_local 로 바뀌므로 **새 a 의 조립 항등** `a = a_ds + a_dchi` 와 **뉴턴 소비 항등**(c_newton·x_star 가 a_local 대수)은 이 단의 정당한 판정량(런 없이 못 얻는 in-vivo s vs g_local 은 E-2).

## 1. 계약 — 물리 계약 1개

**「DRIVE 의 대각 S-채널 이득 a_ds = Σ Gw λ* ∂S/∂x 위에, CMFGEN `do_var_cont.f` 가 dCHI 로 접는 국소 `(∂J/∂χ)|_S` 를 Lumina 에서 **고정 S=S_fixed 의 원시 형식해 J_fs 의 χ 도함수** `∂J_fs/∂χ`(별도 버퍼 재귀, 생산 형식해·ALI 무변경)로 이산화한 한 항 a_dchi 를 더해 a_local = a_ds + a_dchi 를 A2-10 라이브 슬롯 `pop_djdn_log_newton` 에 넣으면, 그 항의 기여(부호·크기·뉴턴 소비)와 CONT 소스 위 황 II→III 비 x 의 epoch 궤적이 판정 셸(Z=16, 셸 12·16·20·24·28)에서 유한 고정점에 착지하는가(LOCAL-A), 아니면 어떤 꼴로 착지하지 않는가(LOCAL-B 진동 / LOCAL-C 단조 상승 / LOCAL-D 혼재); 그리고 그 국면에서 경험 총미분 s 와 국소 이득 g_local = a_local/den 의 차 s−g_local(정규화 κ_loc=(s−g_local)/(1−g_local))는 대각-only κ_diag=(s−g_ds)/(1−g_ds) 대비 얼마인가.」** 판정런 1회(twin-T_e, binned 챔피언 경로 `formal_solve_bin`/`cmfgen_solve_J`, CONT 소스 ON, DR-on base, DJDN census ON·log-뉴턴 적용 ON, **CHI 채널 ON**). 대조 = 388393(같은 물리 base·대각 log-뉴턴만·CHI 부재).

⚠**말하지 않는 것**: CMFGEN f_II 도달 · s−g 의 국소/비국소 **분할** · 「국소 실패 ⇒ 비국소 원인/필요」 · 셸간 tri_* · N×N rate 결합 · 이중 권위 해소 · 커밋 J 의 완전 국소 도함수(산란·ALI 반응 포함) · 정확 multi-GPU 솔버 Λ* · Milne 소스 자체의 옳음 · Si/Ca 판정(기재만) · 스펙트럼 · 자유-T_e · NT 활성 · `LUMINA_CMFGEN_LAMBDA_TRI` Thomas ALI · JDUMP · MAXCH 노브 신설/삭제 · `coupled_newton_solve_all` 활성 · `lumina_cuda.cu` 레거시 루프 · 새 .cu · Makefile 접촉 · J 형식해 변경(같은 입력 상태에서 같은 J 연산이 아님).

### Q-GOAL (기안자 자기 답 — 경량 사전감사가 독립 재판정)
① 칸: 어긋남 지도 **중간 셸 9–24 S II/III**(CMFGEN f_II .85/.80/.71 대 base .04/.12/.22). 이 단은 그 칸을 직접 「CMFGEN 숫자로」 줄이는 것을 성공 조건으로 두지 않는다 — CONT 연속체 소스(그 칸의 본선)가 폭주하는 기전에서 DRIVE 가 넣지 않은 **국소 `∂J_fs/∂χ` 한 항**을 더해, 그 기여와 이 창의 착지 변화를 잰다. 착지 없이 그 칸은 닫히지 않는다. 이 항이 「지배 채널」인지는 이 단의 질문이 아니다. ② 산출 = 물리 측정(착지 여부 · in-vivo s, g_ds, g_local, κ_diag, κ_loc · a_ds/a_dchi 조립·뉴턴 소비). 신규 계측 = chi census 파일 1(`lumina_pop_djdn_chi.csv`)·신규 env 1(`LUMINA_POP_DJDN_CHI`). 생산 census 16 열 포맷 유지(`a` 칸 = a_local). ③ 대안: (i) 완전 Path B 직행 — 폐기(⑩ 형태 미지·7건 차단). (ii) SPLIT 판별 — 폐기(mini-Path-B 비용; 이 단도 판별기가 아님). (iii) 대각 유지 — DRIVE 착지 0/5, sh24 κ_diag=1.78>1 측정. (iv) NT ON — CMFGEN toy06 VADAT `F [TRT_NON_TE]`; `LUMINA_ION_NT` 부재 유지. 지금 열린 더 가까운 표적 없음. ④ 건너뛰면 이 채널의 기여를 재지 못한 채 다음 형태를 짓는다. 이 단의 형태는 고정-S `∂J_fs/∂χ` 한 항으로 좁고 §2 이산화가 접지한다. Q-GOAL④ 는 「이 채널의 기여를 쟀다」로 닫힌다 — 「비국소 필요를 쟀다」로 닫지 않는다.

### Q-LEAN (기안자 자기 답)
L1 물리량 = (i) 판정 셸 커밋 x 의 epoch 궤적 Δ_k(착지) (ii) s, g_local, a_ds, a_dchi, κ_loc. 소비: 생산 census `x_lag`/`x_map`/`x_star`/`epoch`/`root`/`damp`/`c_newton`/`a`/`g` → Δ_k·s·착지·리미터 항등·뉴턴 소비 / chi census `a_ds`/`a_dchi`/`a_local` → G-ASM. 소비되지 않는 것: `LUMINA_CMFGEN_JDUMP`(388393 과 같이 미설정) · `gamma_eff`·`maxch_factor`·MAXCH 노브(코드 무접촉) · FDHOLD · NT · tri_* · `coupled_newton_solve_all`. L2 게이트 넷만: G-ID(계보·소스·러너 상수) · G-RUN(완주) · G-ASM(a_local 이 **조립·뉴턴에 소비**됐는가 — 아니면 A 가 공허; 대각-only 가 여기 FAIL) · G-LND(착지 **측정**이 가능한가: 적용·분리·창 실재). G-LND 는 착지 **성공**이 아니다(성공은 E-1/분기). 중복 없음. L3 = 2 걸음(구현·selftest → 런 1). 더 적은 걸음: 388393 에 J/χ 덤프·체크포인트 없음 ⟹ 오프라인으로 국소 항 착지를 못 잰다. L4 삭제·통합 후보: JDUMP 로 오프라인 formal_solve 재현 — 이중 권위·SPLIT 과 같은 mini-Path-B. **채택하지 않음.** 새 .cu·Makefile·`lumina_cuda.cu` 채택하지 않음. `lumina_cmfgen.c` 는 별도-버퍼 재귀의 자리로만 채택(형식해 본체·ALI 무변경). MAXCH·FDHOLD 삭제하지 않음. 판정 셸 다섯 유지: DRIVE 와 같은 착지 정의의 같은 창.

## 2. 기전 — 런 0회 확정분

- **대조군 s−g [388393 judge_output.json · 판정문 §2]. DRIVE 대조 맥락만.** x^{(k)} = epoch k 마지막 행 x_star, s_k=(x_map^{(k)}−x_map^{(k−1)})/(x_lag^{(k)}−x_lag^{(k−1)}), g_ds=a_ds/den, κ_diag=(s−g_ds)/(1−g_ds). K=7, ITER=6. s 는 두 epoch 시컨트이다 — 그 사이 타 셸·준위·n_e 가 함께 움직인 총미분. **고정상태 편미분과 동일시하지 않는다. a_dchi 의 필요·크기 하한이 아니다.** 국소/비국소 분할을 이 표로 주장하지 않는다(DRIVE 판정문 §3/§30 과 같음).

| 셸 | g_7 | s_7 | s−g | κ_diag | damp_7 | 비고 |
|---|---|---|---|---|---|---|
| 12 | 2.780568659 | 3.1439719425250967 | +0.36 | −0.20 | 0.03170800121 | \|s−g\| 소 |
| 16 | 0.2096581554 | −0.020633488432262568 | −0.23 | −0.29 | 0.68930287 | 진동 |
| 20 | 0.4081369659 | 0.3555552391667972 | −0.05 | −0.09 | 0.2066156222 | \|s−g\| 최소 |
| 24 | **8.189966386** | **−4.584583035960915** | **−12.77** | **1.78** | 0.4140611145 | 부호 반대, 폭주 셸 |
| 28 | 0.06706843776 | −0.20663429835432645 | −0.28 | −0.29 | 1.000 | 인접 폭주 |

  sh24 epoch 7 생산 행(census `%.9e`): x_lag=1.092563223e+02, x_map=4.072876518e+02, a=7.779456053e-04, g=8.189966386e+00, x_star=1.498327537e+02, root=1, c_newton=−2.415102421e+00, damp=4.140611145e-01. ⟹ 대각 스칼라(저장 λ* 의 S-채널)가 맵 감도를 재현하지 못함(부호 반대). **이 수가 이 단이 넣는 항의 크기 하한이 아니다.**

- **왜 이 항을 더하는가 (판별이 아님).** CONT 폭주의 근인(`READOUT_CONT_NLTE_SUPERTHERMAL`·설계 §1): 이온화가 올라가도 J 가 고정이라 복원 되먹임이 없다 — 「중성 불투명도↓·EUV 얇아져 J↓」가 되어야 재결합이 복원하는데, 그 되먹임이 끊겼다. DRIVE 의 대각 이득 `a=Σ Gw λ* ∂S/∂x`(`lumina_plasma.c:2686-2704`, `dSdx=(n_e Ec+S Xc)·geom/χ` `:2699`)는 **`∂J/∂S` 만** 재현하고 **고정-S 불투명도→J 감도는 빠졌다**. MVP §0(b): 「이온화↑ ⟹ J↓」는 저장 λ* 밖의 `∂J/∂χ` 이다. 이 단은 그 **한 항**을 DRIVE 이득에 더하고 기여·착지 변화를 잰다. 이 항이 폭주의 지배 채널인지, 착지 실패가 비국소를 가리키는지는 이 단의 질문이 아니다.

- **CMFGEN 형태 (발자국, 추측 아님).** `do_var_cont.f:171-172`: 「perturbations to J in terms of the perturbations to CHI and ETA. F2DA is the dCHI matrix; FC the dETA matrix」. VJ 조립 `:791-793`: `VJ += VCHI*dJ_LOC(CHI) + VETA*dJ_LOC(ETA)`. Lumina 대응(이 단이 **넣는 것 / 넣지 않는 것**):

  dχ_bf = Xc · dN_i = −Xc · geom · dx     (σ_bf·∂N/∂x, DRIVE 가 이미 쓰는 Xc)
  dS   = (dη − S dχ)/χ                    (DRIVE a_ds 가 λ* 로 접음 = FC 대각)
  dJ_fs|_S = (∂J_fs/∂χ)|_S · dχ           (F2DA 같은-깊이 대각의 **고정-S 형식해 근사** = 이 단)
  dJ|_ALI, dJ|_scatter, dJ|_tri, dJ|_N×N, dETA 준위-정합   (**이 단 밖**)

  DRIVE 의 dS 안 `+S Xc geom/χ` 는 S=η/χ 의 ∂S/∂χ 이지 `(∂J_fs/∂χ)|_S` 가 아니다. 후자를 `-S (dχ)/χ` 로 치환해 a_ds 에 더하면 중복 가산.

- **S 정의 (못박음).** DRIVE a_ds 는 `S_fixed` 를 쓴다(`lumina_plasma.c:3313,2699`). 생산 형식해가 받는 S 는 `S_fixed + r·J_old` 이다(`lumina_cmfgen.c:3035`, r=χ_es/χ_tot). **이 단의 a_dchi 재귀는 S 를 S_fixed 로 고정한다.** 생산 형식해의 산란 소스와 다르다. 근사임을 기재한다. `S_fixed+r·J_old` 로 바꾸려면 r 조립이 필요하고 χ_es 는 g_lre 에 없다 — 이 단에서 열지 않는다.

- **`(∂J_fs/∂χ)|_S` 이산화 (런 0회, 코드에서 유도). 초안의 L* 붕괴식을 폐기한다.**

  생산 한 세그먼트(`formal_solve_bin` `:2912-2922` inbound, `:2937-2944` outbound; Tlo=Tup=NULL 이므로 접선항 `:2951` 은 L* 에만 더해지고 **J_fs 는 이 포인터로 바뀌지 않음**, astra):

  ```
  tau = χ_tot * ds
  if (tau < 0) tau = 0
  ex  = exp(-tau)
  psi = (tau > 1e-4) ? (1 - ex) : (tau - 0.5*tau*tau)
  I_out = I_in * ex + S * psi
  ```

  고정 S, 같은 구조의 **별도** 재귀(생산 I/Jacc/Lacc 에 쓰지 않음·Tlo/Tup 를 넘기지 않음):

  ```
  psi' = (tau > 1e-4) ? ex : (1 - tau)
  dI_out = ex * dI_in + [ -I_in * ex + S * psi' ] * ds * dchi
  ```

  I_in 은 세그먼트 갱신 **전** I. 비핵 광선: 내측 경계에서 I 와 dI 를 재설정하지 않고 접선 귀환(생산 `:2927-2930` 과 같음). 핵 광선: I = Bnu_inner, **dI = 0 으로 재설정**(생산 `:2929`; B 는 χ 의 함수가 아님). 작은 τ 다항 분기의 psi' 도 위와 같이 같은 분기. 각도 가중 ray_w·wacc 정규화는 J_fs 와 같다.

  **국소 대각만 저장한다.** 광선마다 셸별 dI[t] = ∂I/∂χ_t 를 유지한다. 셸 s 세그먼트:

  ```
  for all t: dI[t] *= ex
  dI[s] += [ -I_in * ex + S * psi' ] * ds     # unit dchi at s
  ```

  그 샘플에서 dJacc[s] += ray_w * dI[s] 만 더한다(∂J_s/∂χ_{t≠s} 는 저장하지 않음 = tri 밖). 다른 셸을 지날 때 dI[s] 의 감쇠는 **같은 셸의 inbound→outbound 재방문**을 위해 유지한다(국소 2-pass; 이웃 χ 응답이 아님). 핵 경계에서 dI[*]=0.

  빈 b, 셸 s:

  ```
  (∂J_fs/∂χ)|_S [s,b] = dJacc[s] / wacc[s]     # wacc==0 이면 0
  ```

  Bnu_inner 는 그 세대 생산과 같은 식(`inner_bb_scale * cm_planck(nu, T_inner)`). χ = 그 세대 `cs->chi_tot`. S = 그 세대 `cs->S_fixed`. 격자 = 그 세대 p_ray / r_inner / r_outer. **호출 자리:** `cmfgen_run` 에서 `cmfgen_solve_J` **이후**, `radeq_set_line_re_source` 와 같은 pass(CHI ON 만). ALI 루프 안에 넣지 않는다. 함수 이름 `formal_djdchi_fs_S`. 등록 `radeq_set_djdchi_fs`. CHI OFF 는 호출·할당·등록 없음.

  **폐기하는 식 (초안, FAIL):**

  ```
  tau_star = -ln(1 - L*)
  dJdchi_S = -(tau_star / χ) * (J - S)     # L* 붕괴. 이 단의 도함수가 아님
  dI/dχ = -ds * (I - S)                    # δI_in=0 국소 세그먼트만. 핵 재설정·접선 귀환·재방문 없음
  ```

  **1/(1−L* r) 을 이 단에 곱하지 않는다.** (i) χ_es 가 g_lre 에 없어 r 을 조립하지 못함 (ii) DRIVE a_ds 는 명시 Λ*(λ*) 이지 λ*/(1−L* r) 가 아님 (iii) ALI 반응을 넣으면 형식해·L* 를 건드려 ④. r 및 L* 의 ALI 가속기 미분은 이 단 밖.

  생산 J 는 `S=S_fixed+r·J_old`(`:3035`) 후 `J_new=(J_fs − q·J_old)/(1−q)`(`:3093-3098`, q=L* r) 를 거친다. 그 미분

  ```
  dJ_new = (dJ_fs − q * dJ_old + (J_new − J_old) * dq) / (1 − q)
  ```

  는 **이 단에 넣지 않는다.** ⟹ **a_dchi = 고정-S 형식해 근사 이득이지, 산란·ALI 반응을 포함한 커밋 J 의 완전 국소 도함수가 아니다.**

- **상태 일치.** `x_lag` 는 J 조립 세대의 커밋 인구 스냅샷이다(`lumina_plasma.c:21895-21902`, `ion_before`; `pop_djdn_set_xlag_snapshot`). χ 응답 `∂J_fs/∂χ` 는 **그 세대** 의 χ_tot·S_fixed·격자·Bnu_inner 로 계산된 것이어야 한다. n_e 하위반복이 덮어쓴 live 인구로 χ 를 다시 풀지 않는다. 이득 조립은 같은 pass 의 저장 `g_lre_*` 와 `radeq_set_djdchi_fs` 버퍼를 읽는다.

- **이득 조립 (signed, 음수 항을 버리지 않음). a_dchi 는 양수-필터 밖.**

  기존 `pop_djdn_a_reduce`(`:2701`)는 `term>0` 만 합산한다. a_dchi 를 그 함수에 넣으면 음의 χ 기여가 사라진다. **금지.** 뉴턴 함수 자체는 음수 a 를 배제하지 않는다(`:2732-2744`).

  ```
  geom     = N_tot / (1 + x_lag)^2          # 기존
  dSdx_b   = (n_e*Ec_b + S_fixed_b*Xc_b) * geom / χ_b   # 기존 :2699, >=0
  a_ds     = Σ_b Gw_b * λ*_b * dSdx_b       # 기존 pop_djdn_a_reduce, 양수 항만(구성상, 이 단이 바꾸지 않음)
  dchi_dx_b= -Xc_b * geom                   # ∂χ_bf/∂x < 0 (이온화↑ → 하단계 밀도↓)
  a_dchi   = Σ_b Gw_b * (∂J_fs/∂χ)|_S,b * dchi_dx_b
             # pop_djdn_a_chi_reduce. 부호 있는 합. term>0 필터 금지
  a_local  = a_ds + a_dchi
  g_local  = a_local / den
  ```

  a_dchi 의 부호는 빈 합이 정한다. 기안자가 부호를 못 박지 않음. 음수 a_dchi 는 복원 후보일 수 있으나 이 단의 성공 조건이 아니다.

  **중복 가산 금지:** a_ds 의 `+S Xc geom/χ` 는 ∂S/∂χ 경유. a_dchi 는 |_S 라 ∂S/∂χ 를 포함하지 않음. `-S/χ` 또는 `-(τ*/χ)(J−S)` 를 (∂J_fs/∂χ)|_S 자리에 넣으면 생산 재귀와 일반적으로 불일치. selftest 가 둘의 불일치를 시연한다.

- **뉴턴.** `pop_djdn_log_newton(num, den, a_local, ...)` — 함수 대수 불변(u=ln x, c=lnρ·g/(ρ−g), g=a_local/den, damp, x_star=x_map·exp(damp·c)). a_local 이 음수여도 클램프 금지(복원 항의 부호). a_local=0 ⇒ c=0 ⇒ x_star=x_map(기존). 폴백 없음. C_max=1.0 유지.

- **라이브 루프 (동결 트리 grep 2026-09-21).** `cmfgen_run`(`lumina_cmfgen.c`): pass `cmfgen_assemble` → `cmfgen_solve_J`(binned `formal_solve_bin`; `LUMINA_CMF_LINERES` 미설정=챔피언; Tlo/Tup = LAMBDA_TRI 미설정이라 NULL) → **CHI ON 이면 `formal_djdchi_fs_S`(S=S_fixed, Tlo/Tup 없음, 별도 버퍼) → `radeq_set_djdchi_fs`** → `radeq_set_tri_response`(버퍼 0, 이 단 비소비) → `radeq_set_line_re_source`(`cs.lambda_star`·`cs.S_fixed`·`cs.chi_tot`·`cs.J`) → `cmfgen_fine_jbar` → `cmfgen_commit_jnu` → init 아니면 `lumina_r7_publish_and_solve_te` → A2-10 `compute_ion_populations_shell` → `parity_rate_se_ratio_checked` 에서 DJDN 블록. Γ 와 dΓ 는 같은 pass 의 커밋 뷰·같은 세대 g_lre_* / dJ_fs/dχ 를 읽는다. `cmfgen_solve_J` 의 J 갱신 호출은 **추가하지 않는다**. `formal_solve_bin` 시그니처·본문 불변.

- **x_map 의 뜻.** x_map = commit 된 J 에서 평가한 Picard/SE 비(뉴턴 보정 전). x_lag = J 조립 세대의 커밋 비(기존 `ion_before` 스냅샷). s 의 정의는 DRIVE 와 동일. g_local = a_local/den. g_ds = a_ds/den(chi census 기재, 커밋에 쓰지 않음).

- **비용.** 388393 실측 ITER 6 ≈ 13.6 h(a100×2, cpus 16) ⟹ ≈ 2.3 h/바깥반복. CHI ON 추가분 = J 조립 세대당 빈도별 `formal_solve_bin` 과 같은 계급의 고정-S 광선 순회 1회(ALI 루프 안이 아님, 새 커널 0). a_ds 빈 환산보다 크다. `--time` 은 선택 ITER 의 1.3× 이상(헤지; 못 박지 않음).

- **미확정(계약 밖).** 착지값의 CMFGEN 근접 · 준위별 점유 정합 · 정확 솔버 Λ* · 판정 셸 밖(Si·Ca·내핵 IGE) · a_dchi 가 s−g 를 메우는지(이 단이 그 크기를 하한으로 쓰지 않음·런이 정함) · 비국소 잔여(이 단이 분할하지 않음) · 산란·ALI 반응을 포함한 커밋 J 의 완전 국소 도함수 · S_fixed 대 S_fixed+r J_old 불일치의 크기.

## 3. 수리안 — 변경집합

env `LUMINA_POP_DJDN_CHI`(신설, 기본 미설정/0 = a_dchi 미적용·chi 파일 없음·`formal_djdchi_fs_S` 미호출·**생산 경로 byte-동일**, 388393 부재). ON(=1) 은 `LUMINA_POP_DJDN=1` 위에서만 유효(G-ID 가 둘 다 요구). ON: §2 a_dchi 를 a_ds 에 더해 `pop_djdn_log_newton` 에 a_local 을 넘긴다. `LUMINA_POP_DJDN=1` 유지(생산 16 열 census, `a` 칸 = a_local). `LUMINA_POP_DJDN_CMAX=1.0` 유지. `LUMINA_CMFGEN_LAMBDA_TRI`·`LUMINA_CMFGEN_JDUMP`·`LUMINA_ION_NT`·`LUMINA_NT_FULL`·`LUMINA_POP_MAXCH`·`LUMINA_POP_DJDN_FDHOLD`·`LUMINA_COUPLED_NEWTON` **부재**. 상태 클램프·floor 금지. `coupled_newton_solve_all` 시그니처·스텁 불변. `formal_solve_bin` 본체·J 갱신 불변.

chi 파일 `work/lumina_pop_djdn_chi.csv`(CHI ON 만 존재). 헤더(순서 고정, 8 열), 행 = 생산 census 와 같은 arming(Z 14·16·20, stage 1, 커밋 스윕; 388393 기재값 gen당 150행):

`gen,shell,Z,stage,epoch,a_ds,a_dchi,a_local`

- `a_ds` = §2 기존 환산(λ* ∂S/∂x, 양수-필터 유지).
- `a_dchi` = §2 국소 χ 항(부호 있음, 양수-필터 없음).
- `a_local` = a_ds + a_dchi = 같은 (gen,shell,Z,stage) 생산 census `a` 와 `%.9e` 일치.

CHI OFF 에서 이 파일은 **없어야** 한다(388393).

### 변경집합 — PATHB-LOCAL (표)

| 경로 | 담당 | 내용 |
|---|---|---|
| `src/env_universe.h` | Opus 4.8 | `LUMINA_POP_DJDN_CHI` 등록(COUNT +1). 다른 키 신설/삭제 금지 |
| `src/lumina.h` | Opus 4.8 | `radeq_set_djdchi_fs` 선언만. `radeq_set_line_re_source`·`coupled_newton_solve_all` 시그니처 불변 |
| `src/lumina_cmfgen.c` | Opus 4.8 | (i) `formal_djdchi_fs_S`: §2 재귀, S=S_fixed, Tlo/Tup 없음, 별도 δI/δJacc/wacc 버퍼, 생산 I/Jacc/Lacc 무접촉. 핵 경계 dI=0·비핵 접선 귀환·작은-τ psi' 분기 포함. 국소 대각만 저장. (ii) `cmfgen_run` 에서 `cmfgen_solve_J` 이후 CHI ON 만 호출하고 `radeq_set_djdchi_fs` 로 등록. CHI OFF 는 호출·할당 없음. (iii) `formal_solve_bin` 본체·`:3093-3098` ALI J 갱신·Tlo/Tup 전달 **무변경**. `LUMINA_CMFGEN_LAMBDA_TRI` 미설정 유지 |
| `src/lumina_plasma.c` | Opus 4.8 | (i) `pop_djdn_chi_gate`(미설정/0=OFF). (ii) `radeq_set_djdchi_fs`(포인터 등록; OFF 이면 NULL). (iii) `pop_djdn_a_chi_reduce`(Gw, Xc, dJ_fs_dchi, N_tot, x_lag) — 부호 있는 합, term>0 필터 **없음**. `pop_djdn_a_reduce` 에 a_dchi 를 넣지 않음. (iv) `parity_rate_se_ratio_checked` DJDN 블록 `:3315-3323`: CHI ON 이면 a_ds 계산 후 a_dchi 를 **필터 밖에서** 더해 a_local 을 `pop_djdn_log_newton` 에 전달; census `a`/`g` 는 a_local·a_local/den; chi 파일 append(생산과 같은 arming, hold_mode==0). CHI OFF 는 dJ 버퍼를 읽지 않고 a_dchi 경로 미진입 — 388393 byte-동일. (v) `nlte_solve_ion_shell`·`coupled_newton_solve_all`·`formal_solve_bin` 무호출·무변경. 클램프·floor 금지. T_e 미지수 아님. x_lag 스냅샷(`:21895-21902`) 무변경 |
| `tests/pop_djdn_selftest.c` | Opus 4.8 | 기존 케이스 유지(DRIVE 30 + SPLIT 추가분, FDHOLD OFF 경로) + **LOCAL**: (1) 한 세그먼트 δI 독립 재계산, 큰-τ·작은-τ 분기 각각 1e-12 (2) 핵 경계 뒤 dI==0 bit-동일 (3) 비핵 접선: inbound dI 가 outbound 시작으로 이어짐 (4) 1-광선 픽스처에서 고정-S 유한차분 (J_fs(χ+ε)−J_fs(χ))/ε 와 재귀가 그레인 안에서 일치 (5) 재귀가 `-S/χ` 및 `-(τ*/χ)(J−S)` 와 픽스처에서 갈림 (6) a_local == a_ds + a_dchi 1e-12; 음수 a_dchi 픽스처가 합에 남음 (7) CHI unset ⇒ a_dchi==0 · a_local==a_ds bit-동일 · chi 헤더 미사용 · `formal_djdchi_fs_S` 미호출 (8) **음성대조 1**: 대각-only(a_dchi=0)를 ON 조립으로 제출 ⇒ G-ASM 항등 FAIL 시연 (9) **음성대조 2**: dJdchi 를 `-S/χ` 또는 `-(τ*/χ)(J−S)` 로 바꿔 넣은 a_dchi 는 생산 재귀와 불일치 FAIL 시연 (10) **음성대조 3**: 음의 a_dchi 픽스처를 `pop_djdn_a_reduce` 양수-필터에 통과시키면 합에서 사라져 FAIL 시연; `pop_djdn_a_chi_reduce` 경로에선 남음 (11) **음성대조 4**: a_dchi≠0 픽스처에서 log-newton(a_ds) 의 (c,x_star) 가 log-newton(a_local) 과 같으면 소비 실패 FAIL 시연 (12) CHI ON 픽스처에서 생산 J·L* 가 CHI OFF 와 bit-동일(도함수 계측이 형식해를 안 바꿈) (13) 리미터 항등 x_star=x_map·exp(damp·c) 가 a=a_local 에서도 1e-12(selftest) (14) chi 헤더 8 열 일치 (15) 준위 벡터 인자 없음(훅 시그니처가 level/N×N 을 받지 않음) |

### 변경집합 끝

★**src 편집 1개 규칙**: 가동 중 src-편집 태스크 0 확인 후 발주(DRIVE 폐합·FULL/SPLIT 폐기). ⚠프리플라이트: 표의 다섯 파일은 기안 시점 repo 에 `LUMINA_POP_DJDN_CHI`·`formal_djdchi_fs_S`·`radeq_set_djdchi_fs` 가 없으므로 `expected_extra` 로 선언. `path_pattern` 은 `.c|.h` 만(`.cu` 없음). **`src/lumina_cuda.cu`·Makefile 은 표에 없다.** 기대 출력 `changeset-consistent files=5`. 오케스트레이션 = A2-10 슬롯 in `lumina_plasma.c` · J 형식해 본체 무변경 · cuda.cu 무접촉 · 재귀는 `lumina_cmfgen.c` 별도 버퍼.

## 4. 게이트 명부 (표)

| ID | 조건 | 음성대조 |
|---|---|---|
| G-ID | 새 런 exports 를 대조군 388393 계보(`/gpfs/kjhan/lumina/cmfgen_map/djdn_drive_20260920/run_h100a100/input/resolved_lumina.exports`)와 diff — 추가·변경 키가 정확히 `LUMINA_POP_DJDN_CHI`(=1) [§6 ① ITER 변경 시 `LUMINA_PURE_CMFGEN_ITER`; 프로필 종속 `LUMINA_CMF_FINE_MGPU_DEVICES` 허용]; `LUMINA_CMF_CONT_NLTE_SOURCE`(=1)·`LUMINA_POP_DJDN`(=1)·`LUMINA_POP_DJDN_CMAX`(=1.0)·`LUMINA_ION_CHANNEL_DUMP`(=1)·`LUMINA_FROZENIN_DR`(=1)·`LUMINA_DR_BOOST_AUTOSTRUCT`·`LUMINA_DR_BOOST_BADNELL`·`LUMINA_DR_BOOST_MAZZOTTA`·`LUMINA_DR_BOOST_NORAD`(각 =1)·`LUMINA_DET_TRANSACTIONAL`(=1) 유지; `LUMINA_POP_MAXCH`·`LUMINA_ION_NT`·`LUMINA_NT_FULL`·`LUMINA_RADEQ_SIMUL`·`LUMINA_SIMUL_ION_DAMP`·`LUMINA_CMFGEN_JDUMP`·`LUMINA_CMFGEN_LAMBDA_TRI`·`LUMINA_POP_DJDN_FDHOLD`·`LUMINA_COUPLED_NEWTON`·`LUMINA_CMF_LINERES` **부재**; 동결 트리 `/gpfs/kjhan/lumina/cmfgen_map/lteunmap_eps_20260917/source.sha256` 의 `sha256sum -c` non-OK 가 정확히 §3 표의 파일(`lumina_cuda.cu`·Makefile 이 non-OK 이면 FAIL; `lumina_cmfgen.c` 는 표에 있으므로 non-OK 허용 — 단 코드 감사: `formal_solve_bin` 본체·ALI J 갱신 무변경); 새 바이너리 해시 `input/binary.sha256` 기재(대조군 `7d42780346bf5d2c...` 와 달라야 함); 러너의 **ITER 상수 셋** — guard `"${LUMINA_PURE_CMFGEN_ITER:-}" == N` · `outer_iterations=N` · postcheck `-ne N` — 이 전부 선택 ITER 와 같음을 `grep -nE 'PURE_CMFGEN_ITER:-\}" == \|outer_iterations=\|-ne [0-9]+ \]\]; then' job.slurm` 출력으로 STAGE_IDENTITY 에 기재; 러너 편집 = job-name·`--time`·ITER 상수 셋·프로필(partition/gres/GPU_PROFILE/cpus, §6 ②)뿐, `runner_edits` 에 `postcheck_te_count` 포함 | 다른 키 혼입·§3 밖 소스 변조(특히 `lumina_cuda.cu`·Makefile)·`formal_solve_bin`/ALI 변경·postcheck `-ne` 가 ITER 와 불일치·LAMBDA_TRI/JDUMP/NT/FDHOLD/COUPLED_NEWTON 혼입 ⟹ FAIL(388393 러너 그대로 = CHI 키 부재로 FAIL) |
| G-RUN | `RUN_FOOTER.txt` `terminal_state=SUCCESS`·`model_rc=0`·`outer_iterations`=선택 ITER; `work/lumina_pop_djdn.csv`·`work/lumina_pop_djdn_chi.csv`·`work/lumina_ion_channels.csv`·`work/physics_comparison/physics_DET_iter<ITER−1>.ion.csv` 실재 | 미완주·chi 결손 ⟹ FAIL; 388393 = chi 파일 없음 ⟹ FAIL; 387175 = 생산 census 13 열·chi 없음 ⟹ FAIL |
| G-ASM | (오프라인) `tests/pop_djdn_selftest.c` rc=0, §3 음성대조 1–4 FAIL 시연 — **봉인 접촉 전 grok 코드 간단 감사 선행**(개정16 ②; 개정25-a Fable 자리는 user 2026-09-21 로 grok 대행); 세그먼트 δI 독립 재계산 1e-12, 핵 dI=0, 재귀 ≠ `-S/chi` ≠ `-(τ*/χ)(J−S)`, 음수 항이 양수-필터에서 사라지면 FAIL 시연, CHI ON 에서 생산 J·L* bit-동일. (in-vivo chi) 헤더 8 열 순서 일치; 생산 census 의 root=1 ∧ damp>0 인 각 (gen,shell,Z,stage) 행에 chi 행이 정확히 1개; 그 쌍에서 `a_local` 이 생산 `a` 와 `%.9e` 일치 **and** \|a_ds + a_dchi − a_local\| ≤ 1e-6·max(1,\|a_local\|); 생산 `g` 가 a_local/den 과 1e-6(뉴턴이 a_ds 가 아니라 a_local 을 씀 — ⑧). 판정 셸(Z=16, 셸 12·16·20·24·28) epoch ≥ 2 전 행 a_dchi 유한·비-NaN 이고, **sh24 의 그 행 집합에서 max\|a_dchi\| > 0**(항이 실제로 모임). **a_dchi·g_local 의 값 자체는 이 게이트의 문턱이 아님**(E-2). 1e-9 금지. commit_gen 비트·파일 존재·CHI=1 자기보고만으로 통과 금지 | chi 헤더 바꿔치기 ⟹ FAIL; CHI ON 인데 a_dchi≡0(대각 이득 그대로) ⟹ sh24 max\|a_dchi\|=0 FAIL; a_dchi 를 `pop_djdn_a_reduce` 로 합산한 빌드 ⟹ 음성대조 3 미시연; 뉴턴이 a_ds 를 소비(생산 g 가 a_ds/den 과 일치하고 a_local/den 과 불일치) ⟹ FAIL; `-S/chi` 또는 L* 붕괴식 빌드 ⟹ 음성대조 2 미시연; 388393 자기검사 ⟹ chi 부재 FAIL |
| G-LND | (생산 census) 헤더 16 열 = DRIVE 와 동일(`epoch,c_newton,damp` 포함; a 칸 = a_local). epoch 구조: 같은 (Z,shell,epoch) 안 x_lag 불변, epoch 단조 증가, 각 판정 셸에서 epoch k+1 첫 행의 x_lag = epoch k 마지막 행의 x_star(`%.9e` 일치), 최대 epoch = ITER+1; 최대 epoch 의 각 gen 에 판정 셸 다섯(Z=16, 셸 12·16·20·24·28) 행이 각각 정확히 1개 실재; 판정 셸 epoch ≥ 2 전 행 root=1(뉴턴 적용) ∧ x_lag>0 ∧ x_map>0 ∧ x_star 유한; 리미터 항등 행 집합 = root=1 ∧ damp>0 인 전 행: \|x_star − x_map·exp(damp·c_newton)\| ≤ 1e-6·x_star **and** damp = min(1, C_max/\|c_newton\|)(1e-6) **and** c' = lnρ·g/(ρ−g) 로 재계산해 \|c_newton − c'\| ≤ 1e-6·max(1,\|c'\|) — 여기서 g = 생산 census `g` = a_local/den, ρ=x_map/x_lag [388393 CSV 그레인]. damp==0 행은 항등에서 제외·행 수와 (Z,shell,epoch) 기재·판정 셸 마지막 두 epoch 에 있으면 E-1 정의로 착지 불가. (분리) 새 런 커밋 x^{(k)} 와 대조군 388393 커밋 x_ctrl^{(k)} 가 sh24 또는 sh28 의 **양쪽 런에 존재하는** epoch 4 또는 5 또는 7 중 어느 하나에서 \|log10 x^{(k)} − log10 x_ctrl^{(k)}\| > 0.05. ⚠G-LND 통과 ≠ 착지 성공 | 생산 census 열 바꿔치기 ⟹ FAIL; root=0 빌드 ⟹ 적용 실패 FAIL; CHI OFF 바이너리로 ON 런을 가장 ⟹ G-ASM chi 부재 또는 분리 0; 388393 자기검사 ⟹ CHI 부재·chi 부재 FAIL |

### 게이트 명부 끝

## 5. 기대치 — 런에서만 얻는 수 (잣대를 대조군에 먼저 적용: 388393 은 chi 파일이 없고 착지 창을 넘지 못한다 — sh24 Δ_7=+0.137, damp_7=0.414)

x^{(k)} = 판정 셸 epoch k 마지막 행 x_star, k = 1 … K, K = ITER+1. Δ_k = log10 x^{(k)} − log10 x^{(k−1)}, k = 3 … K(epoch 1→2 는 씨앗 완화라 제외). s 정의는 §2(DRIVE 와 동일; 총미분 시컨트, 편미분이 아님). g_local 은 생산 census `g`. a_ds·a_dchi 는 chi 파일. x_lag 불변(\|denom\|/max(\|x_lag^{(K)}\|,\|x_lag^{(K−1)}\|,1) < 1e-3) 이면 그 셸 s 미정의 → NaN 기재, 착지 투표에서 제외. \|1−g_local\|<1e-3 이면 κ_loc 미정의, \|1−g_ds\|<1e-3 이면 κ_diag 미정의 → NaN 기재. g_ds = a_ds/den 은 chi 파일에서 계산(기재, 커밋 아님).

| ID | 양 | 대조(388393) | **기대·역할** |
|---|---|---|---|
| E-1 착지 | 판정 셸 다섯의 Δ_{K−1}·Δ_K 와 그 두 epoch 의 damp·root | Δ_6/Δ_7: sh12 −0.003/−0.004 · sh16 +0.280/−0.880 · sh20 −0.066/−0.076 · sh24 −0.086/+0.137 · sh28 −0.700/+0.282; 착지 0/5 | **판정량**(분기 표). 착지 = \|Δ_{K−1}\|≤0.05 ∧ \|Δ_K\|≤0.05 ∧ 그 두 epoch damp=1 ∧ root=1 |
| E-2 s−g_local | 판정 셸의 s_K, g_ds, g_local, a_ds, a_dchi, κ_diag, κ_loc | §2 표(a_dchi 없음). sh24 κ_diag=1.78 | 판정 조건 아님 — **기재**. 정합 기재: 착지 셸 \|κ_loc\|<1 / 진동 셸 κ_loc<−1 / 단조 상승 셸 κ_loc>1. 정합 위반은 «기재». s−g 로 a_dchi 크기를 재지 않음(자릿수 비교 없음). in-vivo s vs g_local 은 런 없이 못 얻음 |
| E-3 감쇠·뉴턴 | census epoch ≥ 2 전 행(Z 14·16·20) 중 damp<1 행 수와 (Z,shell,epoch); 판정 셸 K−1·K 의 damp | 해당 없음(구 바이너리 chi 없음). 생산 damp 는 §2 | 기재. 판정 셸 마지막 두 epoch 에 damp<1 이면 그 셸은 착지 불가(정의) |
| (기재) | `Z16_nlte_q2/q1`(NLTE 권위 r) · Si 셸 12·20 · Ca f_II · 외곽 S f_III 셸 28·32·36 · Fe f_IV 셸 0–3 · 388393 대비 같은 epoch 의 log10 x 차 | 388393·387175 | 판정 조건 아님 |

문턱 근거: 수렴 창 0.05 dex = DRIVE 와 동일(CONT 1 반복 이동 ≥0.13 의 절반 이하). 상승 하한 0.10 = DRIVE 와 동일. 대조군 분리 0.05 dex = 같은 창(FP 미세변동은 1e-3 dex 급). 기안자 예상(판정 조건 아님·확률 미부여): 형태가 맞으면 sh24 의 a_dchi 가 0 이 아니고 \|κ_loc\| 가 \|κ_diag\| 보다 작을 수 있다. 착지 자체는 비선형 맵·ITER 창의 문제라 예상을 강건하다고 하지 않는다 — 런이 정한다. 착지 실패를 비국소 필요로 읽지 않는다.

### 분기 (표)

셸별 코드: 게이트 미충족 ⟹ 전 셸 0. 게이트 충족 시 판정 셸마다 — **1 착지**(E-1 정의) / **2 진동**(착지 아님 ∧ Δ_{K−1}·Δ_K < 0 ∧ min(\|Δ_{K−1}\|,\|Δ_K\|) > 0.05 ∧ \|Δ_K\| ≥ \|Δ_{K−1}\|) / **3 단조 상승**(착지 아님 ∧ Δ_{K−1} > 0.10 ∧ Δ_K > 0.10) / **4 기타**(감쇠 상한 걸린 마지막 epoch·감쇠 진동·단조 하강·root=0 등).

| 분기 | 조건 | 의미 |
|---|---|---|
| LOCAL-X | G-ID/G-RUN/G-ASM/G-LND 중 하나라도 미충족 | 미시연 |
| LOCAL-A | 게이트 전건 **and** 판정 셸 다섯 전부 코드 1 | **정의된 국소 `∂J_fs/∂χ` 한 항을 더했을 때 이 창에서 CONT 폭주 셸이 유한 고정점에 착지**. 수렴값의 CMFGEN 근접·\|κ_loc\|<1 정합은 기재. 이 분기가 비국소를 닫거나 열지 않음 |
| LOCAL-B | 게이트 전건 **and** 코드 2 셸 ≥ 1 **and** 코드 3·4 셸 0 | **진동 발산** — 이 국소 χ 항 추가로 이 창에서 착지 못 함. g_local 과 s 의 관계는 E-2 기재. 다음 최소 수리는 이 계약이 열지 않음(재분기 금지, 판정문이 명명). 「비국소가 원인/필요」로 확장 금지 |
| LOCAL-C | 게이트 전건 **and** 코드 3 셸 ≥ 1 **and** 코드 2·4 셸 0 | **단조 상승 지속** — 이 국소 χ 항 추가로 이 창에서 착지 못 함(ITER 부족은 E-3·장부에 기재, 분기 재해석 금지). 「비국소가 원인/필요」로 확장 금지 |
| LOCAL-D | 게이트 전건 **and** (코드 4 셸 ≥ 1 **or** 코드 2·3 셸이 함께 있음) | 혼재 — 판정자가 셸별 κ_loc·damp·a_dchi 로 명명. 착지 못 한 셸이 있으면 「이 국소 χ 항 추가로 그 셸이 이 창에서 착지 못 함」을 기재. 「비국소가 원인/필요」로 확장 금지 |

### 분기 끝

재분기 금지. 판정기 = `judge_drive.py` 계보의 운전자 도구를 **chi 8 열·§4 G-ASM/G-LND·§5 코드**로 개정한 것(해시 `READOUT_TOOLS.sha256` 기재; 소스 아님). 음성대조: 388393 자기검사 ⟹ G-RUN(chi 부재)·G-ASM·G-LND FAIL ⟹ LOCAL-X; 387175 ⟹ 생산 13 열·chi 없음 ⟹ LOCAL-X. **대각-only(CHI 부재 또는 a_dchi≡0) = FAIL** — 이것이 ⑧ 을 닫는 시연.

## 6. 자원 · 실행 · **결정 항목(운전자·user)**

새 RUN 루트 `/gpfs/kjhan/lumina/cmfgen_map/pathb_local_20260921/run1`(형제 루트), 입력 = 388393 입력 복사 + 새 빌드 + exports 편집(G-ID). GPU 1 런. 봉인 base·대조군 읽기만.

① **ITER** — **못 박지 않음.** 트레이드오프만 적는다: DRIVE ITER=6(388393)이 채널 OFF 의 착지 0/5 대조이므로, LOCAL 도 **ITER=6 로 맞추면 「항 효과」를 같은 반복 예산에서 직접 비교**(term-on vs term-off)할 수 있다. 8–10 은 착지 여유를 주나 「항 대 반복수」를 혼동시킨다. 고른 N 을 러너 상수 셋 세 곳과 `LUMINA_PURE_CMFGEN_ITER` 에 **같은 숫자**로 적을 것(§0·G-ID).
② 프로필·헤지 — 직전 단과 동일(먼저 시작하는 쪽만; 멀티파티션·cpus 16 편집은 G-ID 에 기재). GPU-bound·byte-배터리 없음.
③ NT 키 부재·MAXCH 미설정·JDUMP 미설정·LAMBDA_TRI 미설정·FDHOLD 미설정·COUPLED_NEWTON 미설정(388393 그대로 + `LUMINA_POP_DJDN_CHI=1` 만 추가).
④ C_max=1.0 유지(새 결정 아님, G-ID 키 유지).

순서: 운전자 기계 게이트 4종(prereg-preflight `files=5`·gate_referents·doc_canonical·dispatch-preflight, 전건 rc=0) → **grok 물리·대수 사전감사(짓기 전, fresh)** → Opus 4.8 경량 사전감사(Q-GOAL·Q-LEAN) → 구현(Opus 4.8, 동결 트리 복사본) → selftest → grok 코드 간단 감사(첫 항목 = 봉인 접촉 가부; cuda.cu 무접촉·Makefile 무접촉·`formal_solve_bin` 본체·ALI 무변경·뉴턴 = A2-10 `pop_djdn_log_newton`·J 형식해 불변(같은 입력·같은 연산)·도함수 = §2 고정-S 재귀이지 L* 붕괴/`-S/χ` 가 아닌지·a_dchi 가 `pop_djdn_a_reduce` 를 안 타는지·I/Jacc/Lacc 무접촉) → 재봉인·빌드 → 스테이징(**러너 ITER 상수 셋 grep 확인**) → 제출 → 판독 → 판정(운전자) → 종료감사(Opus 4.8 fresh; GPU 지출·물리 주장이 걸린 단이라 생략하지 않음).

## 7. 분장 장부 (개정25-e · 규약 / 실제 / 위반)

| 단계 | 규약 | 실제 | 위반 |
|---|---|---|---|
| 기안 | Fable fresh(25-b) | **Fable 대행**(`grok-4.6`, user 2026-09-20 지시). 개정 = astra 교차감리 5건 반영 | 규약 담당 ≠ 실제 모델(Fable 토큰 소진, 기재) |
| 교차감리(재현 렌즈) | astra / Codex 계열 | **집행**(`gpt-6-astra`, 2026-09-21). 초안 framing 5건 | |
| 물리·대수 사전감사(짓기 전) | grok fresh(user 2026-09-21, 개정25-e) | 미집행 | |
| 경량 사전감사(Q-GOAL·Q-LEAN) | Opus 4.8 fresh(25-d) | 미집행 | |
| 구현 + selftest | Opus 4.8 | 미집행 | |
| 코드 간단 감사 | Fable fresh(25-a) → grok 대행(user 2026-09-20/21) | 미집행 | ⚠기안 대행자 = 감사 대행 후보(자기 계약 편향, 개정25-b 기재) |
| 스테이징·제출·판독·판정 | 운전자 | 미집행 | |
| 종료감사 | Opus 4.8 fresh | 미집행 | |

## 8. 주장 상한

말할 수 있는 것: 「단계-총량 스칼라 이득에 **고정-S `∂J_fs/∂χ` 한 항**(DRIVE 에 정의된 국소 χ 응답)을 더해 A2-10 log-뉴턴으로 적용했을 때, 그 항의 기여와 황 II/III 비가 판정 셸에서 이 창의 유한 고정점에 착지하는가와 그 꼴」, 「그 국면의 in-vivo s, g_ds, g_local, κ_diag, κ_loc」. 착지해도 CMFGEN 근접은 기재 사항이지 성공 조건이 아니다. **준위-정합이 아니다.** N×N·tri·이중 권위는 이 단이 열지 않았다. **a_dchi 는 고정-S 형식해 근사 이득이지 산란·ALI 반응 포함 완전 도함수가 아니다.** 착지 못 하면 「이 국소 χ 항 추가로 이 창에서 착지 못 함」까지 — 「비국소가 원인/필요」로 확장하지 않는다. 다음 최소 수리의 이름은 판정문이 붙인다(재분기 금지). 「CHI 토글 단독」은 OFF 런이 없으므로 selftest byte-동일 + 코드 감사의 오프라인 주장 조건부. 말하지 않는 것: 정확 솔버 Λ* · Milne 소스 · Si/Ca · 스펙트럼 · 자유-T_e · NT · Thomas ALI · 완전 Path B · s−g 의 국소/비국소 분할. twin-T_e 핀 조건부.

## 9. 집행 기록

1. **기계 게이트 3종(2026-09-21, 운전자 grammar-debug, 발주 전)**: **전건 rc=0**.
   - `check_prereg_preflight`: `changeset-consistent files=5` · `branch-partition trials=200000 rules=5 fixtures=7 residual=LOCAL-X` · `references-resolved refs=9` · `prereg-preflight PASS`.
   - `check_gate_referents`: PASS `changeset_exempt=5`.
   - `check_doc_canonical`(인용 4문서): 전건 `single` PASS.
   - dispatch-preflight 는 발주서 작성 후(스테이징) 집행.
2. **grok 물리·대수 사전감사(짓기 전, fresh — 개정25-e, 2026-09-21)**: **«폐합 차단» 0건 — 통과**. (grok 이 기안자 = 자기 계약 자기 감사, 한계 기재; 앞선 astra 교차감리가 독립 렌즈로 보완.) 검증: ① δI 재귀 = 실제 `formal_solve_bin` 세그먼트(`lumina_cmfgen.c:2912-2944`)의 고정-S χ 편미분과 일치(ψ'=ex/1−τ·핵 dI=0·접선 귀환·Tlo 무관) — **a_ds+a_dchi=(∂J/∂η)dη+(∂J/∂χ|_η)dχ 라 이중가산 아님**(astra 「둘 다 필요」확인, `-S/χ` 치환 금지가 지킴). ③ a_dchi 차원·부호(dχ/dx=−Xc·geom<0, `:2690-2700` 형제 미분)·den 정합·a_local→`pop_djdn_log_newton` 대수 정확. ④ 별도 버퍼로 I/Jacc/Lacc·ALI·`formal_solve_bin` 본체 무변경 가능. ⑤ 뉴턴이 a_local 소비·CHI OFF byte-동일. ⑥ 부호 필터 우회 확인. ⑩⑪ 판별 프레임 무유출·단일 물리 계약.
   - **★구현 지침 (grok «기재» 2건 — 구현·코드감사에 반영, 블로커 아님)**:
     (a) **NC1 오라벨 정정**: §3 selftest (8)/§4 G-ASM 음성대조 1 「대각-only(a_dchi=0) ⟹ G-ASM **항등** FAIL」은 틀림 — a_dchi=0 이면 합 항등 `a_ds+0=a_local` 은 **참**이다. **NC1 의 실제 FAIL 신호 = sh24 max|a_dchi|>0**(항이 실제로 모임). 구현자는 합 항등을 깨는 테스트를 짜지 말 것. 이 정정을 selftest·판정기에 적용.
     (b) **⑧-폐쇄는 G-ASM 단독이 아니라 G-ASM+G-LND 5중 사슬**: (i) G-LND `c'=lnρ·g/(ρ−g)`↔`c_newton` 1e-6(census `g`=a_local/den 이면 newton(a_ds) 는 여기서 깨짐) (ii) epoch k+1 `x_lag`=epoch k `x_star`(%.9e; census 만 a_local 위조·커밋 r 은 a_ds 면 사슬 깨짐) (iii) 대조군 분리 0.05 dex (iv) sh24 max|a_dchi|>0 (v) selftest NC4. §0⑧ 이 이 사슬을 G-ASM 에 몰아 적은 것은 과대 귀속(구멍 재개방 아님) — 판정기·코드감사는 다섯을 함께 본다.
     (c) 잔여 구현 감사 항목(grok): `formal_djdchi_fs_S` 필드를 `CMFGENState` 에 달면 표 밖 파일 ⟹ 파일정적/plasma 복사로 `files=5` 유지 · CHI 게이트는 챔피언(LINERES 부재) 경로 안 · selftest 훅이 생산 `formal_djdchi_fs_S` 를 실제로 링크(build.sh:16 이미 `lumina_cmfgen.c` 링크) · 복제 공식만 검사하면 생산 함수 표류 ⟹ 코드감사 첫 항목.
3. **경량 사전감사(Opus 4.8 fresh, Q-GOAL·Q-LEAN — 개정25-d, 2026-09-21)**: **«폐합 차단» 0건 — 통과**(모델 `claude-opus-4-8` 자기보고 확인). 독립(grok 결론 미승계). Q-GOAL ①②③④ 전건 부합: 기전이 지목한 빠진 국소 ∂J/∂χ 복원 채널을 접지 이산화로 하나 넣는 **물리 측정·사다리의 다음 칸(우회 아님)**. Q-LEAN L1–L4 부합: 프로젝트 기준 여윔(게이트 4종 비중복·chi 8열 전건 소비·음성대조는 규약 강제이지 양산 아님). **L4 삭제 후보 1건 = selftest NC1(case 8)** — 죽은 음성대조(a_dchi=0 이면 합 항등 `a_ds+0=a_local` 참이라 FAIL 미시연) ⟹ **sh24 max|a_dchi|>0 검사로 재작성**(물리 범위 불변, «기재»). ★**이 NC1 지적은 grok 물리·대수 사전감사(집행 기록 2(a))와 독립 수렴** — 구현·판정기에서 반드시 재작성. 발주 정지 신호 없음.
4. **§6 결정(운전자·user, 2026-09-21)**: ①**ITER=6**(user 결정 — 388393 이 동일 예산 term-off 대조라 「항 효과」직접 귀속·주장 상한이 기여 측정이라 착지 여유보다 term-attribution 우선). ⟹ 러너 상수 셋(guard `PURE_CMFGEN_ITER:-}" == 6`·`outer_iterations=6`·postcheck `-ne 6`)·`LUMINA_PURE_CMFGEN_ITER=6` 같은 숫자. ②프로필·헤지 = 직전 동일(멀티파티션 h100/h200/a100·cpus 16, 먼저 시작하는 쪽). ③NT/JDUMP/LAMBDA_TRI/MAXCH/FDHOLD/COUPLED_NEWTON 부재(388393 + `LUMINA_POP_DJDN_CHI=1` 만). ④C_max=1.0 유지.
5. **구현(Opus 4.8) + selftest(2026-09-21)**: **완료 — selftest 67/67 PASS(0 real FAIL), teeth 검증**(틀린 스케일 주입 시 case 1·4 FAIL·복원). 모델 `claude-opus-4-8`. 동결 트리 5파일 편집. δI 재귀 = `fs_seg_eval`(생산·테스트 공유, 복제 표류 없음, ψ'=ex/1−τ·핵 dI=0·접선 귀환·S=S_fixed). `pop_djdn_a_chi_reduce` 양수-필터 없음(음의 χ 보존). a_local→`pop_djdn_log_newton`, census g=a_local/den. CHI OFF: `formal_djdchi_fs_S`·`radeq_set_djdchi_fs` 미호출·g_lre_djdchi NULL·chi 파일 미개방 ⟹ 388393 byte-동일. NC1 재작성(a_dchi≡0 ⟹ max|a_dchi|>0 위반 FAIL, 합 항등 아님). NC4 크기 확보(a_dchi=0.55 on a_ds=0.30, c 0.112→0.374). 버퍼 = `lumina_cmfgen.c` **file-static**(`CMFGENState` 필드 아님·헤더 무접촉) ⟹ files=5. **운전자 독립 mtime 확인**: 변경집합 5파일만 최신·`lumina_cmfgen.h`(09-05)·`lumina_cuda.cu`(09-09)·Makefile(09-10) 무접촉. 빌드(전체)·제출 미실행(운전자 몫).
6. **코드 간단 감사 (2026-09-21)**: **★grok 대행 = astra**(user 2026-09-21 승인 — grok CLI 가 코드 감사에서 4회 연속 간헐 중단, grok 자체는 생존[프로브 PASS]; 일회 일탈, [[feedback_allocation_rev25_claude_driver_opus48]] 25-a). astra(`gpt-6-astra`, read-only, 계약 저자도 코드 저자도 아님 → 독립). **판정 = «봉인 접촉 불가»(진행 불가)** — 10항목 중 8 «기재»(형식해·ALI 무변경·버퍼 격리·재귀 정확[비음수 τ]·부호 보존·a_local 소비·OFF 구조 일치·금지파일 무접촉[3파일 SHA 일치]), **2건 수리 요**:
   - **#10 «수리 요»(생산 함수 미검증)**: selftest `T:522,540,557,705` 가 **test-replica `formal_djdchi_ray_test`(C:8563-8589)** 를 부르고 **생산 `formal_djdchi_fs_S` 는 실행 안 함**(공유는 세그먼트 primitive 뿐). ⟹ 생산 핵 재설정·광선 가중·인덱싱이 표류해도 통과. 구현자 「생산 함수 실제 호출」주장 반증. grok 사전감사 record 2(c) 「복제 표류」경고가 실현. **수리**: selftest 를 생산 `formal_djdchi_fs_S` 직접 호출 fixture 로 연결 + 생산 버퍼(cs.J/Jacc/Lacc/λ*) 불변 실측 테스트 추가.
   - **#7 (도함수-클램프 정합)**: `C:2991 if(dtau<0)dtau=0` 은 **생산 형식해의 기하 τ-가드 충실 복제**(금지 클램프 아님 — 감사 조건 「무클램프」가 과광). 그러나 τ 클램프 영역에서 세그먼트 소스-도함수 `dsrc=(S−I_in)ds·δχ` 가 **비영**인데 생산 I_out 은 그 영역에서 χ-무관(τ=0)이라 ∂I_out/∂χ 소스항은 **0** 이어야 함 ⟹ τ<0 코너(χ<0/부호 불투명도 발화 시)에서 도함수 불일치. **수리**: 클램프 유지, 단 dtau 클램프 시 dsrc=0(τ 가 χ-무관); τ<0 fixture 로 도함수=0 검증.
   - ⟹ **구현 재발주(Opus 4.8) → 재감사.** 봉인 접촉 게이트 미개방.
   - **구현 재발주(Opus 4.8, 2026-09-21)**: 수리 완료 — #10 = 생산 `formal_djdchi_fs_S` 직접 호출 fixture(P1 FD 대 생산 formal_solve_bin·P2 memcmp cs.J·λ* 불변, replica `formal_djdchi_ray_test` 대체)·#7 = `dsrc=clamped?0.0:...`(ex=1 유지·τ<0 도함수=0). **selftest 73/73 PASS**(두 수리 teeth: 생산 core-reset 파괴→P1 FAIL·클램프 zeroing 제거→1b·P3 FAIL, 복원). 운전자 확인: fix 는 lumina_cmfgen.c(18:02)·selftest(17:59)만 접촉·헤더 09-05 무접촉·files=5 유지.
   - **astra 재감사(2026-09-21, `gpt-6-astra`)**: **#7·#10 닫힘·회귀 없음 ⟹ 봉인 접촉 가부 = «가».** #7(C:2999-3005 dsrc=0·ex=1·P3 생산 도함수=0∧생산 forward FD=0)·#10(C:8573-8584 wrapper 생산 함수 호출·T:669-707 P1 생산 walker FD·P2 실제 cs.J·λ* memcmp). fs_seg_eval 은 도함수 walker+hook 만(forward/ALI/Tlo-Tup 무전파). ⚠astra 캐비앳: 73/73 재실행·byte-diff 미수행(운전자 mtime·심볼 부재로 files=5 독립 확인). ⟹ **봉인 접촉 게이트 개방.**
7. **재봉인·빌드·스테이징·제출(운전자, 2026-09-21, syntax)**: **완료·제출 job 390839.**
   - **재봉인**: `source.sha256` 185파일 재생성·**델타 정확히 5**(변경집합)·`sha256sum -c` rc=0·backup=`source.sha256.pre_local`. ⚠DRIVE reseal pipefail 버그 회피(비-OK 가드 + delta `|| true` + mv 항상 실행).
   - **빌드**: `cuda_build7`(syntax nvcc CUDA 13.0.2, build6 명령 그대로) rc=0·**binary `7fcf0e632e3b4629...`**(DRIVE `7d42780...`와 다름 ✔ G-ID).
   - **스테이징**: 새 루트 `/gpfs/kjhan/lumina/cmfgen_map/pathb_local_20260921/run1`·388393 계보(cp -a input)·경로참조 7 재작성(0 old/7 new)·**exports 델타 정규화 후 정확히 +LUMINA_POP_DJDN_CHI="1"**(COUPLED_NEWTON=0 등 388393 상속)·바이너리 교체+binary.sha256·job-name=LUMINA_pathblocal·**ITER triple 전부 6**(388393 상속, 편집 불요)·partition h100,h200,a100·gres gpu:2·cpus16·GPU_PROFILE h100-a100x2·`--time=20:00:00`·STAGE_IDENTITY/OWNER(DO_NOT_CANCEL)·deck OK·`check_driver_hygiene` PASS.
   - **dispatch-preflight**: N/A — 구현이 Agent(Opus 4.8)라 Codex 발주서 없음(개정25). 제출 방어선 = 러너 fail-closed 게이트(partition·GPU kind/mem·binary sha·ITER==6·deck·catalog·exports source)가 job 시작 시 실행.
   - **제출**: `sbatch job.slurm <run_root>` → job 390839 PENDING(18:20). ⚠**user 2026-09-21 「h100/a100 으로·h200 당분간 사용금지」** ⟹ 390839 취소(소유 확증 WorkDir=run1·PENDING 무낭비)·partition `h100,h200,a100`→**`h100,a100`**·STAGE_IDENTITY runner_sha·runner_edits 갱신·driver_hygiene PASS·**재제출 job 390846** PENDING(18:31). (러너 expected_partition 은 h100,a100,h200 유지=상위집합, SBATCH 에서 h200 제외라 착지 불가.)
   - ⚠**390846 FAILED(00:00:01, syn102) = `RUN_BUILD_IDENTITY_MISMATCH`**: 스테이징 시 바이너리 교체하며 `input/binary.sha256`·STAGE_IDENTITY `run_binary_sha256` 는 갱신했으나 **`input/run_build_identity.json` 의 `binary.sha256` 미갱신**(388393 DRIVE 값 `7d42780...` 잔존). 러너 line 94 게이트가 record.binary.sha256==actual==STAGE 셋 일치를 요구. **운전자 스테이징 교훈**: 바이너리 교체 = 3곳 동기(`binary.sha256`·STAGE `run_binary_sha256`·`run_build_identity.json` binary.{sha256,path,bytes}). **수리**: run_build_identity.json 을 build7 + 재봉인 source.sha256(185파일)로 재생성.
   - **재제출 run2 = job 391837**(21:55 KST, h100,a100). run1 은 `RUN_ATTEMPTED`(390846) 로 소진 ⟹ 새 루트 run2. 게이트 8종 사전확인 전건 OK(run_build_identity 셋 일치·binary==actual·exports 델타=+CHI·ITER triple 3·partition h100,a100·RUN_ATTEMPTED 무·driver_hygiene PASS·deck OK). ⟹ 착지 감시.
   - ⚠**G-ID 계약 문면 «기재»**: LOCAL G-ID 부재 목록에 `LUMINA_COUPLED_NEWTON` 이 있으나 388393 은 `COUPLED_NEWTON="0"`(off·상속). 「부재」는 「미enable(=0/off)」로 읽는다 — 델타는 오직 +CHI. 판정기 G-ID 는 델타(=+CHI)로 집행(리터럴 key-absent 아님).
8. **착지·판독·판정(운전자, 2026-09-22)**: **완주 SUCCESS(391837, ~11.5h, a100 syn102, model_rc=0) → 분기 LOCAL-D.** 판정문 정본 = `docs/VERDICT_DET_PATHB_LOCAL_2026-09-22.md`. 판독기 `judge_local.py`(judge_drive.py 계보 + chi 8열·G-ASM·LOCAL 분기·κ_diag/κ_loc·NC1 실신호)·산출 `judge_output.json`. **게이트 전건 PASS**(G-RUN·G-ASM[sh24 max|a_dchi|=2.7e-3>0·합항등 0·census↔chi 0]·G-LND[분리 sh28 ep5=0.554·limiter 0위반]). **착지 0/5**. ★핵심 측정: 국소 ∂J/∂χ 채널 폭주 셸 sh24 집중(a_dchi 1.05e-3>a_ds 7.8e-4·이득 8.18→19.21)·**κ 1.775→1.305 감소(발산 완화)이나 >1** ⟹ 국소 채널 옳은 방향 기여하나 단독 불충분. 주장 상한 준수(「비국소 필요」로 확장 안 함·astra 정정).
9. **종료감사(Opus 4.8 fresh)**: 미집행.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 — PATHB-LOCAL (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "(?:src|tests)/[A-Za-z0-9_/.]+\\.(?:c|h)",
    "symbol": "LUMINA_POP_DJDN_CHI",
    "roots": ["src", "tests"],
    "expected_extra": ["src/env_universe.h", "src/lumina.h", "src/lumina_cmfgen.c", "src/lumina_plasma.c", "tests/pop_djdn_selftest.c"]
  },
  "branches": {
    "regimes": [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
    "metrics": {"count": "len(v)", "gate_fail": "sum(1 for x in v if x < 0.5)", "landed": "sum(1 for x in v if 0.5 <= x < 1.5)", "osc": "sum(1 for x in v if 1.5 <= x < 2.5)", "runaway": "sum(1 for x in v if 2.5 <= x < 3.5)", "other": "sum(1 for x in v if x >= 3.5)"},
    "rules": [
      {"name": "LOCAL-X", "predicate": "count == 0 or gate_fail > 0"},
      {"name": "LOCAL-A", "predicate": "count > 0 and gate_fail == 0 and osc == 0 and runaway == 0 and other == 0"},
      {"name": "LOCAL-B", "predicate": "count > 0 and gate_fail == 0 and osc > 0 and runaway == 0 and other == 0"},
      {"name": "LOCAL-C", "predicate": "count > 0 and gate_fail == 0 and runaway > 0 and osc == 0 and other == 0"},
      {"name": "LOCAL-D", "predicate": "count > 0 and gate_fail == 0 and (other > 0 or (osc > 0 and runaway > 0))"}
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
    "residual": "LOCAL-X"
  },
  "references": [
    {"path": "docs/PREREG_DET_PATHB_FULL_2026-09-21.md"},
    {"path": "docs/VERDICT_DET_DJDN_DRIVE_2026-09-21.md"},
    {"path": "docs/PREREG_DET_DJDN_DRIVE_2026-09-20.md"},
    {"path": "docs/DESIGN_PATHB_LINEARIZATION_2026-09-19.md"},
    {"path": "CLAUDE.md"},
    {"path": "src/lumina_plasma.c", "flags_existing": ["parity_rate_se_ratio_checked", "compute_ion_populations_shell", "radeq_set_line_re_source", "g_lre_lambda_star", "g_lre_J", "g_lre_S_fixed", "g_lre_chi_tot", "nlte_solve_ion_shell", "coupled_newton_solve_all"], "flags_planned": ["LUMINA_POP_DJDN_CHI", "pop_djdn_chi_gate", "pop_djdn_a_chi_reduce", "radeq_set_djdchi_fs"]},
    {"path": "src/lumina.h", "flags_existing": ["coupled_newton_solve_all", "radeq_set_line_re_source"], "flags_planned": ["radeq_set_djdchi_fs"]},
    {"path": "src/lumina_cmfgen.c", "flags_existing": ["cmfgen_commit_jnu", "cmfgen_run", "formal_solve_bin", "lambda_star"], "flags_planned": ["formal_djdchi_fs_S"]},
    {"path": "src/env_universe.h", "flags_existing": ["LUMINA_COUPLED_NEWTON", "LUMINA_CMFGEN_LAMBDA_TRI", "LUMINA_ION_NT", "LUMINA_CMFGEN_JDUMP"], "flags_planned": ["LUMINA_POP_DJDN_CHI"]}
  ]
}
```

읽기 전용 — 파일화하지 않았다. 변경집합이 초안 3파일에서 5파일로 늘었다: 고정-S 재귀는 `formal_solve_bin` 과 같은 광선 구조를 별도 버퍼에 복사해야 하고, 그 자리는 동결 `lumina_cmfgen.c` 이다. 형식해 본체·ALI·Makefile·`.cu` 는 그대로 밖이다.