# 사전등록 — DET-COLDSTART-NT: 자연해 + 이온별 재규격 pin 의 차가운 출발 + 비열 이온화 상한 (쌍안정 판별, 1런)

**표적:** Lumina 6원소 이온 인구를 공개 StaNdaRT CMFGEN toy06 19.480 d 이온분율(차가운 가지)로 초기화하되 고정 창의 준위 해를 **base 와 같은 자연해(단일 보존 행 + GTH) 뒤 이온별 재규격**으로 두고(직전 계약의 두 보존 행 = PIN-FAIL 원인 제거), 코드가 가진 비열 이온화 채널을 **상한(pref = 1.0)** 으로 켠 상태에서 생산 반복을 돌렸을 때 sh10–30 의 S III/S II 가 차가운 값 곁에 머무는가(COLD-STAY 또는 DRIFT/COLD-ABOVE), 뜨거운 가지로 돌아가는가(HOT-RETURN). 상태 = **기안**(실행 승인·구현·판정 아님). 기안자 = Fable fresh(`claude-fable-5-1`, 개정25-k). user 결정(2026-09-24) = 갈림길 권고 (iii)+NT ON 채택. 최종목적 = MC 수송으로 CMFGEN 수준 SN Ia 스펙트럼 재현. 잣대 = **공개 파일만**(자체 CMFGEN 런 인용 금지; 공개 CMFGEN 은 NT ON — A&A 668 A163, `docs/RECHECK_SELFRUN_CMFGEN_DEPENDENT_VERDICTS_2026-09-24.md`).

## §0. 선행 사실 (저자 실행 = 로그인 노드 비연산 조회 + grammar-debug 소형 스크립트 1개; 값은 referent, 예측 아님)

1. **직전 계약의 승계**: `docs/PREREG_DET_COLDSTART_2026-09-24.md`(개정 8) 의 §0-1·§0-2(공개 파일·단계 배열)·§0-3(초기화 경로·bf 소스 계수·ρ 정의)·§0-4(DJDN 증인)·§0-5(base 398939)·§0-7(뜨거운 가지 미수렴), §2 의 속도 보간·단계 사상·armed 판정·주입 지점 ①·n_e 폐합·applied.csv writer·준위 증인 writer, §3 산출물, §4 G-ID/G-INJ/G-COLD/G-RUN 본문과 NC-INJ/NC-HOT/NC-COLD/NC-B, §5 판정 통계·J·referent·경계 근거·소스 증인 정의는 **이 계약이 그대로 상속**한다(아래에서 바꾸는 항목만 적는다; 바꾸지 않은 절은 직전 계약 문면이 정본이며 판독기 동결본 `coldstart_20260924/coldstart_judge_frozen.py`(33b3635) 가 그 구현이다). 판정 `docs/VERDICT_DET_COLDSTART_2026-09-24.md` §2·§6: 두 보존 행 강제 정방계의 정확해가 음수(캡처 3계 mp 판별) — 설계 근원 = 유지된 200 방정식에 이온 간 bf/재결합 열이 남은 채 총량 두 개를 건 것. 자연해(단일 보존 행 + GTH)는 세 셸 모두 음수 0(`replay_s24.json` sh24 min 4.0e-10).
2. **현 구현(동결 트리 `/gpfs/kjhan/lumina/cmfgen_map/lteunmap_eps_20260917/source`, 직전 변경집합 반영본; 변경 전 사본 `coldstart_20260924/impl/pre/`)**: armed 는 `nlte_population_candidate_build_bundle` 래퍼(`lumina_plasma.c`:23102–23119)가 `g_coldstart_armed = current_iter < n_pin` 으로 세우고 본문 반환 후 0 으로 내린다(`[COLDSTART] current_iter= armed= n_pin=` 태그 :23113). `coldstart_pin_armed()`(:3595) 가 OR 된 곳은 **셋**: (α) `:20382–20383 int ion_lock_mode = nlte_ion_lock_active(...) || coldstart_pin_armed();` → 두 보존 행 조립(:20594–20612) — **음수의 원인**; (β) `:20813–20814 int capture_generator = !(nlte_ion_lock_active(...) || coldstart_pin_armed()) && ...` → GTH 경로 차단; (γ) `:21031–21033 int lock = nlte_ion_lock_active(...) || coldstart_pin_armed() || nlte_per_ion_rescale_active() || pair_shares_slot;` → 해 뒤 이온별 재규격(:21128–21146: lo 블록 합 → `ion_number_density[ip_lo]`, hi 블록 합 → `[ip_hi]` = 주입 ① 값). (γ) 의 폴백 :21134–21137: `scale_lo = (sum_lo > 0 && n_lo_total > 0) ? n_lo_total/sum_lo : (n_lo_total == 0 ? 0.0 : 1.0)` — `sum_lo == 0 ∧ n_lo_total > 0` 이면 **조용히 1.0**(준위 합 0 ≠ 주입 총량), `n_lo_total == 0` 이면 0(주입 총량 0 과 정합). (γ) 는 `LUMINA_NLTE_PER_ION_RESCALE=1` 과 같은 블록이나 그 노브는 env 정적 캐시(:11807)라 창으로 못 자른다(직전 §0-3 「재사용 불가」 유지).
3. **NT 채널(코드)**: (a) 침적 배열: DET 레인은 `LUMINA_DEPOSITION_FILE`(398939 exports 행 28, `input/model/deposition_cmfgen.csv`, 열 `shell_id,heating_rate`, 50 행, sha256 `847e80d1…4791` = repo `data/tardis_reference_toy06_19p48d/deposition_cmfgen.csv` 와 byte-동일) 을 `lumina_cuda.cu`:7929–7955 가 읽어 `gamma_deposition_publish(…, GAMMA_PROVENANCE_EXTERNAL_FILE, …)` 로 넘기고, 그 경로는 `lumina_plasma.c`:23975–23978 에서 `gamma_deposition_compute_nonthermal`(:23777–23799) 을 호출한다 ⟹ `nonthermal_ioniz_rate[s] = pref · heating[s] / (W_ION_EV · EV_TO_ERG)`, `W_ION_EV = 35.0`(:23771), `EV_TO_ERG = 1.602176634e-12`(`lumina.h`:37), `pref = LUMINA_NT_FULL ? 1.0 : ETA_NONTHERMAL(0.05)`(:23770·:23792–23794), 그리고 `frozenin_set_nt_rate`(:7939) 로 등록. **398939 stderr 행 8 `[GAMMA][PUBLISHED] generation=1 … provenance=EXTERNAL_FILE heating_manifest_sha256=1761455f… nonthermal_manifest_sha256=f9c08170… shells=50`** ⟹ base 에서도 배열은 pref 0.05 로 **채워져 있었다**(「미확인」 항목 해소). (b) 소비자 셋: ① 단계 균형 `compute_ion_populations_shell`(:3939–3948): `static ion_nt = getenv("LUMINA_ION_NT")` 1회 평가, `ion_nt && g_nt_ioniz_rate` 이면 `gamma_nt_atom = R_nt[s]/n_atom`(n_atom = Σ_e X_e ρ/(A_e amu), 전 원소) → `parity_rate_se_ratio_checked`(:3315–3330, 호출 :4100–4103) 에서 `Gamma += gamma_nt_atom`(:3327, 전 원소·전 인접단계 쌍에 원자당 균등) — **398939 는 ION_NT 미설정이라 이 소비자만 OFF**; ② 준위 행렬 `:20304–20330`: `if (gamma_dep != NULL && nonthermal_ioniz_rate[shell] > 0)` — **env 무관**, DET 레인은 `build_bundle` 이 `gamma_dep` 를 요구(:22623)하므로 base 398939 의 준위 행렬에도 pref 0.05 NT 가 바닥→바닥(`ACM(ground_hi,0) += R_nt/n_atom`, :20325–20327; `ew_capture` 경로는 element-wide 게이트 `nlte_element_wide_commit_enabled`, 398939 exports 에 관련 키 없음) 로 이미 들어 있었다; ③ 동결(frozen-in) 판정 :8045 와 `[FROZENIN][WARN]`(:7960) 은 `frozenin_active()`(:7593, `LUMINA_FROZENIN`) 안 — 398939 exports 에 `LUMINA_FROZENIN` 없음(`LUMINA_FROZENIN_DR=1` 만) ⟹ 도달 불가. :14160 `sh.gnt` 는 RADEQ_SIMUL 사다리(`g_simul_on`) — exports 에 `LUMINA_RADEQ_SIMUL` 없음(운전자 확인 항목 (b)-3). ⟹ **이 런의 NT ON = ① 을 켜고(`LUMINA_ION_NT=1`) ①②의 pref 를 0.05→1.0 으로(`LUMINA_NT_FULL=1`)**. T_e 는 고정 프로필(398939 `fixed_te_loader_consumed=1`)이라 NT 가열은 T_e 에 닿지 않는다 — NT 는 이온화 채널로만 들어간다.
4. **★차단 사실 — XSHELL 런타임 계약**: `lumina_cmfgen.c`:8090–8108 `xshell_runtime_contract_ok` 의 `forbidden[]` 에 `"LUMINA_ION_NT"` 가 있다(`xshell_env_is_off` = 미설정 또는 "0", :8077). `LUMINA_XSHELL_RESPONSE=1` 이면 :8839–8848 에서 이 계약이 평가되고 실패 시 `[XSHELL][BLOCKED] reason=RUNTIME_CONTRACT_OR_ALLOCATION` 후 `cmfgen_run` 이 **반복 0 전에 −1 반환**(MODEL_FAILED). ⟹ 갈림길 권고의 「XSHELL_RESPONSE 1 유지 + ION_NT=1」 은 **실행 불가**. 선택지: (A) `LUMINA_XSHELL_RESPONSE=0`; (B) `lumina_cmfgen.c` forbidden 목록 편집; (C) `NT_FULL` 만(준위 행렬 ×20, 단계 균형 OFF). **이 계약 = (A).** 근거: XSHELL 캡처는 생산 경로 밖의 관측이다 — XSHELL 판정문(`docs/VERDICT_DET_PATHB_XSHELL_2026-09-24.md` G-RUN `production_state_untouched`, G-NULL `J_null ≡ J_prod6` bit-동일)과 코드(`xshell_gate` 소비처 :8850·8897·8911·9002·9172·9200 = 리셋·epoch 6/7 캡처·종료 후 측정) 가 생산 solve 불변을 실측했다. (B) 는 이 단이 소비하지 않는 계측의 C 코드를 건드리고, (C) 는 퇴화 장부의 기전(단계 균형)을 켜지 않는다. ⚠직전 계약 §6 은 「XSHELL=0 은 base 교체 = user 결정」이라 했다 — **이 계약은 새 계약이므로 그 조항을 승계하지 않고 여기서 결정하되, user 에게 (b)-1 로 명시 보고**한다. x_hot referent(398939, XSHELL=1) 계보는 생산 상태 불변 실측 위에서 유지된다.
5. **NT 크기 referent(저자 실행, grammar-debug `coldstart_20260924/fable_nt/nt_referent.py` → `ref_nt.txt`, 2026-09-24; 입력 = 398939 `input/model/deposition_cmfgen.csv`·`work/lumina_ion_channels.csv`·`stderr.log`)**: (a) 침적 manifest 알고리즘 재현 = SHA-256(domain 문자열 ∥ BE-u64(n) ∥ Π_i [BE-u64(i) ∥ BE-u64(bits(f64 v_i))]) (:23737–23750·hash_u64/f64) — 398939 heating `1761455f…` **MATCH**, nonthermal(pref 0.05) `f9c08170…` **MATCH** ⟹ pref 1.0 배열의 referent manifest = **`fe6aced5bb909553f3deb21d6cd6a8efbe2a8b2ccce4db4b63521d325758c0a8`**(기재용; 게이트 아님). (b) R_nt(1.0)[s] = heating[s]/(35.0·1.602176634e-12) [/s/cm³]: sh10 2.4798e5 · sh15 2.8653e4 · sh19 6.3574e3 · sh24 **1.1832e3** · sh30 1.4309e2. 원자당 Γ_nt(1.0) = R_nt/n_atom(n_atom = n_e/x_e, 398939 채널 덤프 마지막 sweep gen 167): sh10 9.95e-4 · sh15 4.00e-4 · sh19 2.42e-4 · sh24 **1.587e-4** · sh30 8.68e-5 s⁻¹(퇴화 장부의 sh24 Γ_nt(1.0)/n_e·n_e = 1.06e-11 × 1.49e7 ≈ 1.6e-4 와 정합). (c) 398939 DET 단계 균형의 뜨거운 Γ_phot(gen 167): sh24 3.875e-2 s⁻¹ ⟹ Γ_nt/Γ_phot = 0.41 %(갈림길의 0.6 % 는 XSHELL Γ_out 2.82e-2 기준). **뜨거운 가지 이동 추정** Δlog10 x_hot ≈ log10(1 + Γ_nt/(Γ_phot + C_ion)) — 국소·수송 없음·J 되먹임 제외: J 전 셸 **≤ 0.049 dex**(최대 sh19; sh10–15 0.008–0.031; sh20–30 ≤ 0.044; sh24 0.0018) — 모두 위쪽(NT 는 이온화를 더한다). r 띠 해상도 gap/3 ≥ 0.34 dex 의 1/7 이하 ⟹ **x_hot referent 는 398939(NT OFF) 값을 그대로 쓰고**, NT ON 이 뜨거운 가지를 올리는 방향이므로 r ≥ 2/3 판정은 보수적이다(r > 1 허용·기재). (d) **퇴화 극한 착지 추정**(장부 §2 형 r_A = (C_ion+Γ_nt)/(n_e α_RR + n_e α_DR + C_rec)·§7 정정형 r_B = (C_ion+Γ_nt)/(n_e α_DR + C_rec); 398939 gen-167 분모·**뜨거운 n_e**): sh24 r_A **1.67**·r_B **2.79** = 장부 값 재현; d_A = log10(r_A/x_pub) = +0.14(sh10)…+0.48(sh30), sh24 +0.35; d_B = +0.37…+0.70, sh24 +0.57. 차가운 n_e(공개 sh24 7.73e6 = 뜨거운 1.47e7 의 0.53)면 분모가 줄어 최대 ≈ +0.3 dex 더 위. 배선 항등 r·den = num 은 1e-10 안(구성상 참). ⟹ **NT 상한의 차가운 끌개가 있다면 그 착지는 x_pub 의 +0.1 … +1.0 dex 위**로 추정된다 — COLD 띠(d ≤ 0.5)의 안팎에 걸친다(§5 가 이 사실을 사전등록한다).
6. **base = 398939**(직전 §0-5·집행 2a: 현 동결 트리 = 398939 빌드 신원 186 파일 일치 + 직전 변경집합 4 C 파일). 봉인 입력·deck·sigma·fixed_te 동일. 398939 채널 덤프 Z16 stage 1 sh24 전 sweep `gamma_nt = 0.000000000e+00`(NT OFF 실측).

## §1. 물리 계약 1개

**질문:** 「398939 계보(같은 봉인 입력·솔버; exports 델타 = 차가운 출발 2키 + NT 상한 2키 + XSHELL 캡처 OFF)에서, SEED 와 논리 반복 0–1 동안 6원소 이온 단계 총량(단계 배열·준위 합)을 공개 CMFGEN 19.480 d 분율에 고정하되 창 안 준위 분포는 base 와 같은 자연해로 두고(이온별 재규격만 추가), 비열 이온화를 코드 상한(pref 1.0, 단계 균형 + 준위 행렬)으로 켠 채 반복 2 부터 풀어 놓으면, 판정 셸 J 의 S III/S II 가 반복 3–5 동안 (i) 공개값 반 자릿수 안에 머무는가(COLD-STAY), (ii) gap 의 아래 1/3 에 머물되 공개값 위 한 자릿수 안에 착지하는가(DRIFT/COLD-ABOVE = 차가운 끌개 존재·NT 상한 과잉), (iii) 뜨거운 기준으로 돌아가는가(HOT-RETURN)?」

**말하지 않는 것:** 직전 §1 전부 승계(공개 재현·어느 가지가 옳은가·쌍안정 일반론·수렴·물리적 안정성·시간진화·Si/Ca/철족 판정·자유 T_e) + **NT 의 물리 강도**(pref 1.0 은 η_ion = 1 상한이지 물리값이 아니다; 코드 주석 :23785–23790 이 x_e 의존 η_ion(Xu–McCray) 미구현을 명시) + NT 준위 행렬 항의 분배(원자당 균등·바닥→바닥 = 코드 그대로, CMFGEN Spencer–Fano 아님) + 뜨거운 출발 NT ON 의 거동(이 런은 차가운 출발뿐). **창 안 준위 분포는 자연해**(base 연산자)이므로 직전 계약의 「고정 창 방정식 상이」 캐비앗은 소멸하고, 대신 「창 안 준위 인구 = 이온별 재규격된 상태(율 방정식의 해가 아님)」 가 남는다 — 해제 반복 2 에서 솔버가 회수한다(초기조건과 같은 지위).

## §2. 초기화·주입 설계 — (iii) 자연해 + 이온별 재규격, N_pin = 2

- **env 2키(승계)**: `LUMINA_INIT_IONFRAC_DIR`(진입 조건 하나)·`LUMINA_INIT_IONFRAC_PIN_ITERS=2`. 속도 보간·단계 사상(II←p₀+p₁ … 최상단 두 몫 직접 분할)·0 셀 검사·armed 판정(래퍼 :23102–23119)·주입 ① 단계 배열(`compute_ion_populations_shell` 끝, 전 6원소)·n_e 폐합(혼합 전하합)·applied.csv writer·준위 증인 writer·`[COLDSTART]`·`[COLDSTART][LEVELS]` 태그 = **직전 §2 그대로**.
- **주입 지점 ② 의 교체(이 계약의 핵심 변경)**: (α) :20382–20383 와 (β) :20813–20814 에서 `|| coldstart_pin_armed()` 를 **삭제**(2행 편집), (γ) :21031–21033 는 **유지**. 귀결: armed 창의 준위 해 = base 와 동일한 단일 보존 행(`nlte_pair_total_density` = 주입된 II+III 합) + GTH 캡처 + 같은 solve ⟹ J 가 나눈 자연 분할 → (γ) 가 lo 블록 합을 `ion_number_density[ip_lo]`(주입 n_II), hi 블록 합을 `[ip_hi]`(주입 n_III) 에 맞춘다(양의 스케일 ⟹ 비음수 보존; 준위합비 = 단계비 = x_pub 정확). Milne 방출의 n_III 도 주입값. **주입 범위 = 전 6원소 유지**: 차가운 상태는 공개 파일의 전 원소 상태이고 n_e 폐합·불투명도·J 가 전 원소를 섞으므로 S 만 차갑고 나머지가 뜨거운 상태는 어느 참조에도 없는 혼합이다; (γ) 는 armed 이면 전 27 이온 쌍에 같은 연산(자연해 × 주입 총량)을 하므로 비-S 원소에도 뜻이 같다(직전 판정문 §4 「A 는 주입 범위 명시」 요구의 답). `pair_shares_slot`·`PER_ION_RESCALE`·`ION_LOCK`·`LOCK_START_ITER` 는 398939 값(0) 그대로.
- **폴백 처리**: (γ) 의 `scale = 1.0` 폴백(:21134–21137, `sum == 0 ∧ n_total > 0`)은 조용한 기본값이다. 자연해가 블록 전체를 정확 0 으로 내는 경우에만 도달하며(양의 재결합 유입이 있는 한 실질 불가) S 에서는 G-COLD `nlte_q1 = stage_q1`(1e-9) 가 잡지만 비-S 는 조용하다 ⟹ **armed 이고 폴백이 발동하면 stderr 1행 `[COLDSTART][RESCALE-FALLBACK] Z=<Z> ion_lo=<i> shell=<s> sum_lo=<v> n_lo=<v> sum_hi=<v> n_hi=<v>`** 를 남긴다(:21137 뒤, `coldstart_pin_armed()` 조건 — armed 아니면 어떤 문장도 없다; 값 변경 없음·종료 없음: 이 블록은 ION_LOCK 경로와 공유되는 base 코드라 부작용을 넣지 않는다). 판독기는 이 행이 하나라도 있으면 G-RUN FAIL(X). 근거 = 「조용한 기본값 없음」 원칙; 비용 = 코드 ≤ 6행.
- **NT ON(env 2키, 승계 아님)**: `LUMINA_ION_NT=1`·`LUMINA_NT_FULL=1`. 코드 변경 0(§0-3). **XSHELL 캡처 OFF(env 1키 변경)**: `LUMINA_XSHELL_RESPONSE` 1→0(§0-4). `LUMINA_ION_CHANNEL_DUMP=1`·`LUMINA_POP_DJDN=1`·`LUMINA_POP_DJDN_CHI=1` 은 398939 값 유지(NT 증인·G-INJ(b) 가 소비). **상한 선택의 문면(오버피팅 금지)**: pref 1.0 은 코드가 가진 **유일한 상한 값**(η_ion = 1)이고 CMFGEN 값이 0.05 와 1.0 사이라는 장부 추정에 맞춰 중간값을 고르지 않는다 — 이 런은 **「NT 상한에서도 차가운 끌개가 없는가」 존재 시험**이다: HOT-RETURN 이면 어떤 pref 로도 없음(결정적); 차가운 착지면 끌개 존재 확정 + 착지의 정밀도(η_ion(x_e))는 다음 단.
- **고정 창의 DJDN·n_e 폐합·준위 증인**: 직전 §2 그대로(epoch 2 x_lag = x_pub; 준위 증인 gen 2–8; 창 안 x_map·x_star 는 NT 포함 율 해가 내지만 ① 이 덮어쓴다).
- clamp·floor·부호 교정 없음. env 4키 모두 `env_universe.h` 등록 확인(:292·293·300·436; `LUMINA_XSHELL_RESPONSE` :545).

## §3. 변경집합과 산출물

### 변경집합 — DET-COLDSTART-NT (표)
| 경로 | 분류 | 허용 변경 |
|---|---|---|
| `src/lumina_plasma.c` | pin 교체 | :20382–20383 와 :20813–20814 의 `\|\| coldstart_pin_armed()` **삭제**(2곳; `ion_lock_mode`·`capture_generator` 는 base 식으로 복귀), :21031–21033 유지, :21137 뒤 `[COLDSTART][RESCALE-FALLBACK]` stderr 1행(armed 조건, 값·흐름 무변경). 로더·사상·armed 래퍼·주입 ①·writer 2종·태그 2종·율 해·n_e 폐합·커밋·DJDN·NT 코드 **무변경** |
| `scripts/coldstart_judge.py` | 판독·판정 | 33b3635 계보 개정: 계약 경로·rev, `EXPORT_DELTA` 4키 + `LUMINA_XSHELL_RESPONSE` 변경 허용(want "0")·`LUMINA_ION_NT`/`LUMINA_NT_FULL` = "1", G-RUN 에 NT 증인(§4)·RESCALE-FALLBACK 부재, DRIFT 하위 판독 `COLD-ABOVE` 필드(§5), `--nt-only`(NC-NT 픽스처), PIN-FAIL 뜻 문면. 분기 규칙·G-INJ·G-COLD·r·분류·소스 증인 코드 무변경 |
### 변경집합 끝
`expected_extra` = 위 2파일. 변경 금지: `src/lumina_cmfgen.c`(forbidden 목록 포함 — (B) 기각)·`src/lumina_cuda.cu`·`src/lumina.h`·`src/env_universe.h`·`tests/`·`Makefile`·봉인 입력·398939 산출. 산출(승계) + 이 계약 추가 없음: `coldstart_referents.csv`(직전 것 재사용 가 — sha 동일 확인) · `coldstart_result.json`(+ `nt_witness`·`cold_above` 필드) · applied.csv · `coldstart_levels_gen002–008.csv` · 표준 dump · **`work/lumina_ion_channels.csv`**(기존 산출, NT 증인으로 소비). 런 루트 `/gpfs/kjhan/lumina/cmfgen_map/coldstart_20260924/run2/`(run1 = PIN-FAIL 보존). 빌드 = 새 cuda_build(직전 `build_coldstart.sh` 계보), 스테이징 = `stage_run1.sh` 개정(exports 델타 5키·`changeset_files=` 갱신·RUN 경로).

## §4. 게이트와 음성대조 (허용오차: 산술 항등 상대 1e-6; 물리 판정은 §5 의 r 띠)

- **G-ID**(개정): 봉인 입력 sha = 398939 · exports 델타 = **추가 4키**(`+LUMINA_INIT_IONFRAC_DIR`·`+LUMINA_INIT_IONFRAC_PIN_ITERS=2`·`+LUMINA_ION_NT=1`·`+LUMINA_NT_FULL=1`) **+ 변경 1키**(`LUMINA_XSHELL_RESPONSE` 1→0), 그 밖 추가·삭제·변경 0(ION_LOCK/LOCK_START_ITER/PER_ION_RESCALE = 0, ION_CHANNEL_DUMP/POP_DJDN/POP_DJDN_CHI = 1, DEPOSITION_FILE 경로 = 런 사본, `LUMINA_FROZENIN`·`LUMINA_RADEQ_SIMUL` 부재) · ITER 6·ALI 8 · 소스 = 동결 트리 + 변경집합(398939 빌드 신원 대비 diff = 직전 4 C 파일, 그 sha = 트리 현행) · 6 공개 사본 sha256 = repo · 침적 파일 sha256 = `847e80d1…4791`(repo `data/tardis_reference_toy06_19p48d/deposition_cmfgen.csv`) · 바이너리 sha 기록 · h200 아님.
- **G-INJ**(a)(b)·**G-COLD**: 승계(무변경). G-COLD 의 세 등식은 (iii) 에서 (γ) 재규격의 귀결이라 **구성상 참**(직전과 같이 배선 시험; 직전 §5 「구성상 참」 문면 유지).
- **G-RUN**(개정 = 항목 2 추가, 새 게이트 아님): 승계 항목(SUCCESS·rc 0·outer 6·ion.csv·gen 3–8·q 유한 양수·준위 증인 7파일·`[COLDSTART][LEVELS]` 태그 등식·G 확정 규칙·PIN-FAIL 식별) + **NT 증인**: `work/lumina_ion_channels.csv` 의 Z 16·stage 1·s∈J 행 **전부**(전 sweep gen)에서 `gamma_nt > 0` ∧ `gamma_nt · n_e / x_e = R_nt(1.0)[s]`(상대 1e-6; R_nt 는 판독기가 봉인 침적 파일에서 `heating[s]/(35.0·1.602176634e-12)` 로 재계산 — 코드 :23796 식·상수 :23771·`lumina.h`:37; n_e/x_e = 덤프 행의 n_atom 정의 :3376–3379) ∧ s∈J 마다 행 ≥ 1 — 이것이 「NT 가 단계 균형에 실제로 소비됐고 pref = 1.0 이었다」 의 런 내 증인이다(코드 0행: 기존 R1-3 덤프 `ion_channel_dump_gate() && g_ion_ch_armed`, :3362–3392). **RESCALE-FALLBACK 부재**: stderr 에 `[COLDSTART][RESCALE-FALLBACK]` 0행. 부재·불일치 = G-RUN FAIL ⟹ X. 기재(게이트 아님): stderr `[GAMMA][PUBLISHED]` 의 `nonthermal_manifest_sha256` 대 referent `fe6aced5…c0a8`(§0-5; 준위 행렬 소비자 ② 의 입력 배열 증인 — 소비 자체는 미증인, §6 주장 상한). ⚠준위 행렬 NT 항 자체는 덤프되지 않는다(기재).
- **PIN-FAIL 식별**: 승계(마지막 `[COLDSTART]` armed=1 ∧ current_iter = k < N_pin · `solve_stage ∈ {PAIR_SOLVE, CE_NONCONVERGED}` · FATAL iter < N_pin · G 등식). **뜻은 §5 에서 재정의**.

| NC | 주입 | 반드시 FAIL / 결과 | 시점 |
|---|---|---|---|
| NC-INJ · NC-HOT · NC-COLD · NC-B | 승계(직전 §4 표 그대로; NC-B referent sh24 ρ 33.66696·κ 0.9549447·b_g 2.836444e-2·창 22.20462, ×3 주입 11.22232/19.63796) | 승계 | 런 전 |
| **NC-NT** | `--nt-only <channels.csv> <deposition.csv> <stderr>`: (0) 양성: 398939 `stderr.log` 의 heating/nonthermal manifest 를 판독기가 재계산해 **일치**(알고리즘 재현; 저자 실행 MATCH) (1) 398939 `work/lumina_ion_channels.csv` 투입 → **FAIL**(gamma_nt = 0, 실제 부재) (2) 같은 파일 사본의 Z16 stage1 gamma_nt 열을 R_nt(1.0)·x_e/n_e 로 치환 → PASS(배선 양성) (3) (2) 를 pref 0.05(×1/20) 로 → **FAIL**(실제 어긋남 ln 20) (4) 398939 stderr 에 `[COLDSTART][RESCALE-FALLBACK]` 합성 1행 삽입 사본 → G-RUN FAIL | (0)(2) 일치 · (1)(3)(4) FAIL 시연 | 런 전 |

OFF byte-동일: 진입 조건 = `LUMINA_INIT_IONFRAC_DIR`(armed·writer·FALLBACK 태그 모두 armed 또는 DIR 조건 아래) — 삭제 2행은 armed 가 0 이면 이미 base 식과 동치였으므로 OFF 경로 불변; NT 는 기존 env 채널(코드 변경 0). OFF GPU 재런 없음(오프라인 주장).

## §5. 기대치·분기

**판정 통계·J·referent·경계 근거·소스 증인 정의 = 직전 §5 승계**(r_s(k)·d_s(k)·COLD_s/HOT_s/MID_s·방향·J = {10–15, 19–30}·referent 표·x_hot = 398939 반복 3–5 기하평균·ρ_s(g) 코드-충실 정의·NC-B). **x_hot referent 캐비앗(개정)**: NT ON 은 자유 창 연산자도 바꾸나 국소 추정 ≤ 0.049 dex 위쪽(§0-5(c)) — 띠 해상도 안이며 방향이 뜨거운 쪽이라 HOT 판정에 보수적; r > 1 은 허용·기재. 판정문은 「x_hot 는 NT OFF 계보」 를 명시한다.

**구성상 참 / 예측 (사전등록)**:
- 구성상 참: G-COLD 값(r = 0, gen 3·4) · epoch-2 x_lag = x_pub · NC-HOT 합성 결과 · 채널 덤프 항등 r·den = num · **ρ_s(4) < 1(차가운 소스 출발)** — (iii) 에서 b_g = f_g,nat/(x_pub·n_e·Φ_g), f_g,nat 은 자연해의 S II 바닥 분율이므로 b_g(창)/b_g(뜨거운) ≈ x_nat/x_pub ≫ 1(sh24 SEED 자연 분할 x_star 81.4 대 x_pub 0.744 = 110×; 갈림길 §2: b_g ≈ 15–35, ρ ≈ 0.03–0.06) — **구성 귀결로 등록, 물리 결과로 읽지 않는다**(「구성상 참≠예측」). 따라서 직전 §5 ② 의 HOT-RETURN (A)「뜨거운 소스 출발」 갈래는 설계로 배제되며, 실측이 (A) 로 나오면 **구성 이상**(재규격·ρ 계산·준위 파일 어딘가의 결함) 으로 기재하고 판정을 보류한다; HOT-RETURN 의 기본 읽기 = (B) 되먹임/비국소 복귀 **후보**(인과 증명 아님).
- 예측(등록, 자릿수 수준; §0-5(d) 국소 대수·수송 없음·T_e 고정): **차가운 끌개가 있다면 착지 x_land ≈ r_deg,s ∈ [r_A, r_B]×(1 … 1.9)** = x_pub 의 **+0.1 … +1.0 dex 위**, 방향은 위(NT 는 이온화를 더한다) — sh24 1.7–5.3(d +0.35 … +0.85). ⟹ **COLD-STAY(d ≤ 0.5) 는 끌개가 있어도 가장 가능성 높은 결과가 아니다**; 가장 가능성 높은 「차가운」 결과는 아래 DRIFT/COLD-ABOVE 다. 이것을 미리 적는 이유 = 사후에 DRIFT 를 「성공」으로 다시 읽는 골대 이동을 막기 위해서다(읽기 규칙을 지금 고정). HOT-RETURN 의 확률은 등록하지 않는다(갈림길 시험). x 가 x_pub **아래**로 0.5 dex 넘게 가면(d > 0.5 ∧ x < x_pub) 장부 그림(NT 상한 ⟹ 위쪽)과 **모순** — 기재·해석 보류.
- 소스 증인 궤적(측정): ρ_s(g), g = 5–8(자유 반복) — COLD 계열이면 ρ < 1 유지 여부, HOT-RETURN 이면 ρ 가 1 을 건넌 g* 와 r ≥ 2/3 최초 반복의 선후(승계 규칙 ②③).

| 분기 | 판정식(승계, 규칙 무변경) | 뜻(개정) | 다음 |
|---|---|---|---|
| **COLD-STAY** | 게이트 전부 PASS ∧ 전 s∈J COLD_s(r ≤ 1/3 ∧ d ≤ 0.5, k=3–5) | 이 초기화·솔버·NT 상한·4 자유 반복 창에서 차가운 상태가 공개값 반 자릿수 안에 유지 — NT 상한이 공개값을 넘지 않았다(장부 추정과 어긋남·기재) | 스펙트럼·EUV 창 J/B 대조(런 0회) 후 user 결정 |
| **HOT-RETURN** | 게이트 전부 PASS ∧ 전 s∈J HOT_s(r ≥ 2/3) | **NT 상한에서도** 해제 후 뜨거운 기준으로 복귀 ⟹ 어떤 pref 로도 차가운 끌개 없음(이 창·이 솔버) · 「불안정」 동일시 금지(승계) · ρ 궤적으로 (B) 후보 기재, (A) 면 구성 이상 | r_s(2)·ρ 궤적·iter0002 S II 엣지 J/B·Γ 판독 → 원천(EUV 채움·비국소, 09-17 후보 B) 또는 bf 소스 조립 수사 |
| **DRIFT** | 게이트 전부 PASS ∧ 그 밖 | 지도 판독(승계 표) + **사전등록 하위 판독 1행(새 분기 아님·규칙 무변경·판독기 필드)**: **COLD-ABOVE** ⟺ 전 s∈J·전 k: r ≤ 1/3 ∧ 0 ≤ log10(x/x_pub) ≤ 1.0 = 차가운 끌개 존재, NT 상한이 공개값 위로 밀었다(§0-5(d) 기대 위치) — 착지 d 표·r_deg 예측 대조 기재. 그 밖 = 승계 해석 규칙(내/외측·UP/DOWN/FLAT) | COLD-ABOVE 면 η_ion(x_e) 물리(pref 0.05–1.0 사이, 코드 주석의 Xu–McCray/KF) 를 다음 표적 후보로 user 보고 — EUV 원천 수사가 아님; 그 밖은 지도 후 user 결정 |
| **PIN-FAIL** | G-ID PASS ∧ G-RUN 고정 창 실패 식별 | **재정의**: 창 안 준위 해가 base 연산자 자체이므로 이 분기는 「base 솔버가 주입된 차가운 상태(차가운 n_e·총량·NT 상한) 위에서 수용 가능한 해를 내지 못했다」 = 설계 결함이 아니라 **솔버–상태 불일치의 관측**(NEGATIVE 는 자연해의 음수 = 캡처 3계에선 0, 도달 가능성 낮음; CE_NONCONVERGED·NONFINITE 가 더 있음직) — 「차가운 상태 불가」 물리 주장은 여전히 아님 | 실패 진단(셸·쌍·solve_stage·n_e 폐합 궤적) 판독 후 user 결정 |
| **X** | G-ID 등 독립 신원 게이트 FAIL · 고정 창 밖 실패 · NT 증인 부재/불일치 · RESCALE-FALLBACK · 준위 해 아닌 실패 | 미시연 | 계측만 수리, 같은 계약·같은 델타로 1회 재실행 |

## §6. 자원·분장·주장 상한·집행 기록

- **GPU 1런**: partition h100,a100(h200 금지), 프로필 h100-a100x2, `--time 16:00:00`, 예상 ≤ 13.5 h(398939 47,833 s 에서 XSHELL 종료 측정분 빠짐; run1 은 49 분에 실패). 재발주 규칙: X = 같은 델타로 1회. PIN-FAIL = 재발주 아님. 런 전 CPU(grammar-debug): 판독기 NC 5종(NC-NT 는 398939 산출·침적 파일 읽기만) · `selftest_a2_10_radeq`(승계 케이스, 변경 없음 — 회귀 확인용).
- **프리플라이트 합성 root**: docs·CLAUDE.md·data·scripts → repo, src·tests → 동결 트리.
- **분장(개정25 + user 2026-09-24 「코딩구현은 운전자야」)**: 기안 Fable fresh **완료**(이 본문) · 기계 게이트 운전자 · 경량 사전감사 Q-GOAL/Q-LEAN grok · 물리·대수 사전감사 astra · **구현 + 직접 selftest = 운전자(Claude, Opus 5.5; Opus 5.5 중첩 CLI 발주 없음)** · 코드 감사 Fable fresh(⚠자기 계약 편향 기재) · 발주·집행·판독·**판정 운전자** · 종료감사 Fable + astra. ⚠**구조적 귀결(기재, 폐합 조건 아님)**: 운전자가 **구현과 판정을 겸한다** — 개정18 ①(판정 저자 ≠ 이탈 당사자)·개정23 의 「구현 ≠ 판정」 분리가 user 결정으로 축소됐다. 남는 독립 검사 = ① 코드 감사(Fable, 구현자와 다른 모델·컨텍스트) — 「구현자 ≠ 감사자」 분리는 이 축이 유지한다 ② 종료감사 2겹(Fable + astra, 판정문 대상) ③ 기계 게이트·NC ④ user. 실제 열은 집행 시 기입(`단계 / 규약상 / 실제 / 위반`).
- **Q-GOAL 자기답**: ① 지도 칸 = sh10–30 S 이온화(공개 대비 2–4 dex) — 이 단은 그 칸의 **원인 분류**(Saha bootstrap 이력·쌍안정 대 방정식의 유일해)를 NT 상한 조건에서 잰다 ② 산출 = 물리 측정(r·d·셸 지도·ρ 궤적·NT 상한 착지 대 장부 예측) ③ 대안 = 뜨거운 출발 NT ON 단독(예상 null 0.4 %·기각) · 율 반쪽 오프라인(공개 J 없어 잣대 불가) · EUV 원천(HOT-RETURN 뒤 걸음) ④ 건너뛰면 09-21 NT OFF 결정(자체 런 근거, 09-24 무효)을 안은 채 원천을 판다.
- **Q-LEAN 자기답**: 물리량 = r_s(k)·d_s(k)(18셸×3)+지도+ρ 궤적. 소비: applied.csv·djdn epoch 2(G-INJ) · ion.csv 0·1(G-COLD) · ion.csv 2–5(r) · referents · `[COLDSTART]`(PIN-FAIL 식별) · 준위 증인(ρ, 해석 층) · `[COLDSTART][LEVELS]`(G-RUN) · **채널 덤프 gamma_nt(G-RUN NT 증인 — 기존 산출, 코드 0행)** · `[COLDSTART][RESCALE-FALLBACK]`(조용한 기본값 폭로, ≤ 6행). 직전 대비 **코드 −2행 +≤6행, 게이트 4 불변, NC +1(NT), env +2(+XSHELL 1키 변경)**. 뺀 것: 두 보존 행 경로·generator 캡처 차단(그 캐비앗 문면 전부). 넣지 않은 것: NT 준위 행렬 항 덤프·manifest 게이트(기재로 족함)·NT 태그 신설(기존 덤프가 대신)·XSHELL forbidden 편집·OFF 재런·뜨거운 출발 대조런. 걸음 수 = 1런.
- **주장 상한**: 「398939 계보 + NT 상한(pref 1.0, 단계 균형·준위 행렬) + XSHELL 캡처 OFF 에서, SEED·반복 0–1 의 6원소 총량을 공개 분율에 고정(창 안 준위 = 자연해 × 이온별 재규격)하고 반복 2 에 풀면, 반복 3–5 의 J 셸 S III/S II 는 r = …, d = … dex 에 있었고 ρ 궤적은 …였다」까지. 어느 분기도 「공개 CMFGEN 재현」·「쌍안정 확정/부정」·「차가운 가지 안정/불안정」·「NT 물리 강도」 가 아니다. NT 준위 행렬 소비는 미증인(배열 manifest 만 기재). x_hot 는 NT OFF 계보(국소 이동 ≤ 0.05 dex 추정). twin-T_e 핀·DJDN·ALI 8·미수렴 x_hot 조건부.
- **집행 기록 뼈대**: 1 기안(Fable, 이 본문) · 2 base 확정(직전 집행 2a 승계)·프리플라이트 rc(합성 root: prereg·referents·canonical·dispatch) · 3 grok Q-GOAL/Q-LEAN · astra 물리·대수 사전감사(회차·차단 수·경로) · 4 구현(운전자)·selftest·NC 5종(경로·rc) · 5 코드 감사 Fable(첫 항목 = 봉인 접촉 가부; 삭제 2행·태그 1곳·판독기 diff) · 6 빌드·스테이징(`stage_run1.sh` 개정본·`STAGE_IDENTITY` 델타 5키)·referents 봉인·제출(job id·노드·partition·`OWNER.txt`) · 7 판독(G-ID/INJ/COLD/RUN·NT 증인·r 표·지도·COLD-ABOVE·분기) · 8 판정(운전자)·종료감사(Fable + astra)·폐합 계급 반영 회차.
- **집행 1** 기안: Fable `claude-fable-5-1`, 2026-09-24. 저자 실행: `coldstart_20260924/fable_nt/nt_referent.py`(grammar-debug) → `ref_nt.txt`.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 — DET-COLDSTART-NT (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "(?:(?:src|tests)/[A-Za-z0-9_/.]+\\.(?:c|h)|scripts/[A-Za-z0-9_/.]+\\.py)",
    "symbol": "coldstart_pin_armed",
    "roots": ["src", "tests"],
    "expected_extra": ["src/lumina_plasma.c", "scripts/coldstart_judge.py"]
  },
  "branches": {
    "regimes": [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
    "metrics": {"count": "len(v)", "gate_fail": "sum(1 for x in v if x < 0.5)", "cold": "sum(1 for x in v if 0.5 <= x < 1.5)", "hot": "sum(1 for x in v if 1.5 <= x < 2.5)", "pin_fail": "sum(1 for x in v if 3.5 <= x < 4.5)"},
    "rules": [
      {"name": "PIN-FAIL", "predicate": "count > 0 and pin_fail > 0 and gate_fail == 0"},
      {"name": "X", "predicate": "count == 0 or gate_fail > 0"},
      {"name": "COLD-STAY", "predicate": "count > 0 and pin_fail == 0 and gate_fail == 0 and cold == count"},
      {"name": "HOT-RETURN", "predicate": "count > 0 and pin_fail == 0 and gate_fail == 0 and hot == count"},
      {"name": "DRIFT", "predicate": "count > 0 and pin_fail == 0 and gate_fail == 0 and cold < count and hot < count"}
    ],
    "adversarial_fixtures": [
      {"name": "empty", "mix": []}, {"name": "gate-fail", "mix": [[0.0, 1], [1.0, 1]]},
      {"name": "all-cold", "mix": [[1.0, 1]]}, {"name": "all-hot", "mix": [[2.0, 1]]},
      {"name": "split", "mix": [[1.0, 1], [2.0, 1]]}, {"name": "band", "mix": [[3.0, 1]]},
      {"name": "pin-fail", "mix": [[4.0, 1]]}, {"name": "pin-fail-id-fail", "mix": [[4.0, 1], [0.0, 1]]}
    ],
    "residual": "X"
  },
  "references": [
    {"path": "docs/PREREG_DET_COLDSTART_2026-09-24.md"},
    {"path": "docs/VERDICT_DET_COLDSTART_2026-09-24.md"},
    {"path": "docs/VERDICT_DET_PATHB_XSHELL_2026-09-24.md"},
    {"path": "docs/READOUT_DEGENERATE_LEDGER_2026-09-18.md"},
    {"path": "docs/RECHECK_SELFRUN_CMFGEN_DEPENDENT_VERDICTS_2026-09-24.md"},
    {"path": "CLAUDE.md"},
    {"path": "data/standart_data1/toy06/ionfrac_s_toy06_cmfgen.txt"},
    {"path": "data/tardis_reference_toy06_19p48d/deposition_cmfgen.csv"},
    {"path": "src/lumina_plasma.c", "flags_existing": ["coldstart_pin_armed", "nlte_ion_lock_active", "nlte_per_ion_rescale_active", "nlte_pair_total_density", "nlte_population_candidate_build_bundle", "nlte_population_candidate_commit_bundle", "nlte_population_candidate_commit_seed_material", "compute_ion_populations_shell", "parity_rate_se_ratio_checked", "gamma_deposition_compute_nonthermal", "gamma_deposition_publish", "frozenin_set_nt_rate", "frozenin_active", "ion_channel_dump_gate", "gamma_nonthermal_ioniz_rate_manifest_sha256", "ETA_NONTHERMAL", "W_ION_EV"]},
    {"path": "src/lumina_cmfgen.c", "flags_existing": ["cmfgen_run", "xshell_runtime_contract_ok", "xshell_env_is_off"]},
    {"path": "src/lumina_cuda.cu", "flags_existing": ["LUMINA_DEPOSITION_FILE", "GAMMA_PROVENANCE_EXTERNAL_FILE"]},
    {"path": "src/lumina_element_wide.c", "flags_existing": ["nlte_element_wide_commit_enabled"]},
    {"path": "src/env_universe.h", "flags_existing": ["LUMINA_INIT_IONFRAC_DIR", "LUMINA_INIT_IONFRAC_PIN_ITERS", "LUMINA_ION_NT", "LUMINA_NT_FULL", "LUMINA_XSHELL_RESPONSE", "LUMINA_ION_CHANNEL_DUMP", "LUMINA_DEPOSITION_FILE", "LUMINA_FROZENIN"]},
    {"path": "src/nlte_population_candidate.c", "flags_existing": ["solve_effects_allowed", "PAIR_SOLVE"]},
    {"path": "scripts/coldstart_judge.py", "flags_existing": ["EXPORT_DELTA", "J_CANDIDATES", "g_run", "g_id", "classify"]}
  ]
}
```
