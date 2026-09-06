# 사전등록 — D-C-1: NLTE 커밋 뒤 전자 밀도 폐합 (DET-DC1-NECLOSE, 2026-09-06)

저자 = Fable(운전석). 상태 = **발주 전**. 상위 = `docs/DESIGN_DC_DD_CMFGEN_CONTRAST_2026-09-05.md` §3 첫 단. 근거 = `docs/AUDIT_DUAL_ION_DENSITY_WIRING_2026-09-05.md`(이중 권위), `docs/VERIFICATION_REGISTERS.md` 09-06(DET-DUAL-AUTH-2 폐합: Saha/NLTE Fe II 0.884·Co 1.0231·Ni 0.9549, shell 0). user 지시(09-06): 「D-C-1 사전등록 착수」.

## 1. 계약 (물리 계약 1개)

**현행**: n_e 는 Saha 층 `compute_electron_density`(`src/lumina_plasma.c:3048`) 가 `Σ_ip charge·ion_number_density`(Saha 단계 밀도)로 정하고, NLTE 준위해가 II/III 를 재분배해도 n_e 는 **되돌아오지 않는다**(NLTE 트랜잭션은 `work_ne` 를 들고만 있고 쓰지 않음, `:20206`). 전하 보존이 NLTE 분배를 못 본다(감사 §2 표).
**계약**: 공개 population commit 직전에 **폐합 n_e** 를 계산해 같은 commit 으로 발행한다. DET 레인의 공개 발행은 후보 bundle(`nlte_population_candidate_commit_bundle`, worktree `:21773`; seed 는 `…_commit_seed_material` `:21886`)이 `candidate->electron_density` 를 `public_plasma->n_electron` 으로 memcpy 하는 것이므로(개정 1, astra 사전감사 Q2), 폐합은 **후보 파이프라인에서 사설 core commit 이 `candidate->level_population`·`candidate->ion_population` 을 채운 뒤, bundle/seed commit 호출(`:16368`·`:16559`) 직전**에 `candidate->electron_density[s]` 를 덮어쓴다. 말단 `population_transaction_commit`(`:20928`)은 이 레인에서 미도달이라 삽입 지점이 아니다.
`n_e^closed(s) = Σ_{(Z,q)} q · n_{Z,q}(s)`, `n_{Z,q} = Σ_{준위∈(Z,q)} nlte_level_populations`(NLTE 집합 (II,III)×{Si,S,Ca,Fe,Co,Ni}) / 그 밖 (Z,q) 는 `atom->ion_number_density`(Saha). `q = ion_pop_stage`(현행 `:3091` 와 같은 전하 규약).
**게이트 env**: `LUMINA_NLTE_NE_CLOSURE=1` 일 때만. 미설정·0 ⟹ 현행과 **byte-동일**(코드 경로 진입 0).
**발행 기록**: 공개 commit 마다 shell 별 1행 `[A2-07][NE-CLOSURE] commit=SEED|R7 generation=… shell=… n_e_saha=%.17g q_saha=%.17g q_closed=%.17g dlog_total=%.17g dlog_saha_residual=%.17g dlog_resplit=%.17g residual=%.3e` — `n_e_saha` = commit 직전 저장된 n_e, `q_saha = Σ q·ion_population(Saha)`, `q_closed = Σ q·n`(NLTE 집합 Σ준위 + 그 밖 Saha), `dlog_total = ln(q_closed/n_e_saha) = dlog_saha_residual + dlog_resplit`, `dlog_saha_residual = ln(q_saha/n_e_saha)`(Saha 층 5% 수렴 문턱의 잔차), `dlog_resplit = ln(q_closed/q_saha)`(NLTE 재분배 효과), `residual = |Σ q n − q_closed|/q_closed`(같은 합 두 번, ≤1e-12 by construction). **0·비유한 처분**: `q_closed` 가 양의 유한값이 아니면 발행하지 않고 `[A2-07][NE-CLOSURE][FATAL] shell=… reason=NONPOSITIVE|NONFINITE` 로 commit 을 거부한다(floor·대체값 금지).
**같은 반복 안의 소비(개정 2 기재, astra 코드 검수 Q5)**: bundle commit 은 폐합 n_e 를 `plasma->n_electron` 과 `opacity->electron_density` 에 동시에 복사하고, CMF 는 χ_es·χ_ff 를 `plasma->n_electron` 으로 계산하므로(`lumina_cmfgen.c:2077–2104`) 폐합 n_e 는 **같은 반복의 physics snapshot 과 다음 pass 의 CMF 재조립**에도 들어간다. 반면 그 commit 의 CPU χ/η·bf·열원장은 폐합 **전** 값으로 생성돼 폐합 n_e 와 함께 발행된다(`:21522–21529`, `:21869–21885`) — 한 commit 안에 두 n_e 시대가 공존한다. 이 단은 이를 고치지 않고 측정한다(E-4·E-5 가 그 크기).
**바꾸지 않는 것**: Saha 층의 n_e 반복(TARDIS식 감쇠 `n_e = ½(new+old)`, **수렴 문턱 5% 상대**, `:3105`) — 이것은 별도 계약 후보로 대장에 조용히 기재한다(전하 보존이 5% 안에서만 닫히는 현행 상태). 다음 반복의 Saha 층은 `plasma->n_electron[s]` 를 시작값으로 읽으므로(`:3069`) 폐합 n_e 가 자연히 다음 반복의 이온화 입력이 된다 — 그 되먹임 자체가 계약의 목적(전하 보존 권위 하나).

## 2. 소스 상태 (단일 변수 유지)
패치 base = **331050 의 소스 상태** = `364ee72` + 보존 패치 `a642a2dd…`(worktree `/gpfs/kjhan/lumina/dproxy_audits/worktree_364ee72_dualauth`, identity 4/4 base 동일). 작업 트리의 Codex 기간 미커밋 변경(quarantine 수정 등)은 **넣지 않는다** — OFF 팔의 base 대비 bit-동일이 게이트라서. 새 바이너리 = 그 상태 + 이 계약 패치 뿐(sm_80/86/90 fatbin, sha 등록).

## 3. 런 설계 — **2런** (OFF 대조 1 + ON 팔 1), 각 A100×2·32 CPU·64 GiB·`--time=08:00:00`, base 331050 과 같은 물리 exports
| 런 | env | 목적 |
|---|---|---|
| **OFF** | 새 바이너리, `LUMINA_NLTE_NE_CLOSURE` 미설정 | 증분 byte-배터리: 새 바이너리가 base 와 **bit-동일**한 물리를 내는가(G-A). 실패 시 ON 발주 금지 |
| **ON** | 새 바이너리, `LUMINA_NLTE_NE_CLOSURE=1` | 계약 본체 |
OFF 가 G-A 를 통과하면 ON 대 OFF 가 단일 변수 대조다(base 331050 대 ON 도 결과적으로 단일 변수).

## 4. 기대치 (「구성상 도출」과 「측정」을 구분)
- **E-1 (구성상 — 작동 확인)**: ON 의 모든 `[A2-07][NE-CLOSURE]` 행에서 `residual ≤ 1e-12`. 코드 귀결이며 예측이 아니다.
- **E-2a (구성상 상한 — Saha 잔차 항)**: `|dlog_saha_residual| ≤ 0.05` 전 셸·전 commit — Saha 층 수렴 문턱 5%(`:3105`)의 귀결. 실제 크기는 **측정**(현재 어느 로그에도 없음: `[A2-07][n_e]` 잔차는 `emit_diagnostics` 시 stdout 뿐).
- **E-2b (도출값 대조, shell 0 — NLTE 재분배 항)**: `dlog_resplit(shell 0)` 를 09-06 단계 비로 도출: 쌍 총량 보존 ⟹ 쌍마다 ΔQ = −Δn_II; Fe Δn_II ≈ +4.9e3 cm⁻³, Ni +7e3, Co −1.4e4; shell 0 은 Si/S/Ca 질량분율 0(`abundances.csv`) 이라 기여 0 ⟹ ΔQ ≈ +1.5e3 / 3.2e9 ⟹ **|dlog_resplit(shell 0)| ≲ 1e-5**. 대조이지 예측이 아니다. |dlog_resplit(shell 0)| > 1e-4 이면 도출 전제(쌍 총량 보존·II/III 만 NLTE·shell 0 세 원소) 붕괴 ⟹ NC-N. 바깥 셸은 Si/S/Ca 가 있어 도출하지 않는다(E-3).
- **E-1b (측정, 기대치 없음 — 개정 2)**: seed commit 의 `dlog_resplit` 셸별 값. seed 후보도 NLTE core 를 거치므로(`:16551`, `:21516–21518`) LTE 동일성은 보장되지 않는다(astra 코드 검수 Q6). 「≤1e-9」 기대치는 철회.
- **E-3 (측정, 기대치 없음)**: 셸별 `dlog_total`·`dlog_saha_residual`·`dlog_resplit` 프로파일(50 셸 × R7 commit 2). 바깥 셸에서 II/III 몫이 크면 클 수 있다 — 어디서 얼마나가 이 런의 정보.
- **E-4 (측정)**: ON 대 OFF 의 반복 1 단계 비 변화 — shell 0 S-역산 바닥 `n(ON)/n(OFF)` 이온별(`scripts/dproxy_readout/cross_run_ratio.py`), 폐합 n_e 가 다음 반복 Saha·행렬에 되먹인 크기.
- **E-5 (측정)**: formal 스펙트럼 ON/OFF 5대역 `Δlog F`(`scripts/dproxy_readout/spectrum_contrast.py`, 격자 bit-동일).

### 게이트 명부 (표)
| 게이트 | 판정 | 참조 |
|---|---|---|
| G-A OFF bit-동일 | `grep -o 'EXACT-MULTIGPU-EPOCH\] status=[A-Z]* devices=[0-9/]* iterations=[0-9]* cap=[0-9]* residual=[0-9.e+-]*' <OFF_ROOT>/stderr.log` 3행이 `/gpfs/kjhan/lumina/superlevel_b_h200x2_k36/slb_20260903T080407Z_dproxy_provenance_k42_ion1_h200/stderr.log` 의 3행과 문자열 동일 **그리고** `scripts/dproxy_readout/readout_dproxy_sl_331050.py --run-root <OFF_ROOT> --out-dir <OUT>` 의 `pairs.csv` 가 base `out_331050/pairs.csv` 와 `n_upper_0,n_upper_1,Jbar_0,Jbar_1` 열에서 338/338 문자열 동일(base 자기대조 참 — astra 사전감사 재현) | OFF 대 base |
| G-B env 단일 차이 | `diff <(sort <OFF_ROOT>/input/resolved_lumina.exports) <(sort <ON_ROOT>/input/resolved_lumina.exports)` 에서 런루트 경로 치환 행 제외 = `> declare -x LUMINA_NLTE_NE_CLOSURE="1"` 1행 (OFF 에는 키 부재) | OFF 대 ON |
| G-C 봉인 입력 | OFF·ON 의 `SUPERLEVEL_B_STAGE_SEAL.txt` 에서 `registered_levels_sha256 = b3645052d00dbd535de4f9b3687eda3da02f3dd0584ccef4079c8f9e700daf67`, `registered_line_list_sha256 = cbd81c4f9b3bd85d723b3b1eee6367735e8f52145920412aaf68388b465554b5`, `candidate_registry_sha256 = c96f74f0a78088bbc7cb3f7ad851060da6ee492d2d09476c26a57bc247ef1923`, `registered_binary_sha256` = 새 바이너리 사전 등록값(두 런 동일); 스테이징 stdout 의 `SUPERLEVEL_B_RUN_SEAL_PASS` 2행(census entries=136·R-II entries=135, non_ok=0). (`resolved_lumina.sha256` 은 스테이징 전 exports 해시라 봉인 검증이 아님 — astra 사전감사 Q4) | stage seal |
| G-D 완주 | 두 런 model rc=0, publication 2, provenance OK 전건, iter-0 pub ⊇ registry 338, 두 pub 공통 ≥330 | readout |
| G-E 폐합 기록 | ON 의 `[A2-07][NE-CLOSURE]` 행 수 = 50 셸 × 공개 commit 3(seed 1→2 + R7 iter 0 + iter 1) = **150**, `[A2-07][NE-CLOSURE][FATAL]` 0행(FATAL 이 있는 commit 의 선행 셸 행은 거부된 commit 의 흔적이므로 계수·판독에서 제외), OFF 에는 0행, 전 행 ` residual=` ≤ 1e-12(`dlog_saha_residual=` 와 구별해 파싱), `dlog_total = dlog_saha_residual + dlog_resplit` 을 1e-12 안에서 만족 | ON `stderr.log` |
| G-F 역산 정합 | 두 런 `probe_lower_any.py` 산포 ≤ 1e-8 | S-역산 |
### 게이트 명부 끝

### 분기 (표)
| 분기 | 조건 | 뜻 |
|---|---|---|
| **NC-A** | G-A~G-F 통과, E-2a·E-2b 일치, 전 셸·전 R7 commit \|dlog_total\| ≤ 1e-3 | n_e 폐합이 이 시대(K42·2반복·T_e 핀)에서 **수치적으로 불활성**. 스펙트럼(E-5)은 별도 측정이며 이 라벨에서 도출되지 않는다 |
| **NC-B** | G 통과, E-2a·E-2b 일치, 어느 셸·commit 이든 \|dlog_total\| > 1e-3 | 폐합이 **어딘가에서 물리량** — 셸·commit·두 성분(Saha 잔차 대 재분배) 보고, E-4·E-5 가 되먹임 크기 |
| **NC-N** | G 통과, E-2b 불일치(shell 0 \|dlog_resplit\| > 1e-4) | 도출 전제 붕괴 — REWORK, 수치 해석 금지 |
| **NC-X** | 게이트 실패(G-A 실패면 ON 미발주; ON 의 A2-07 n_e 반복 미수렴·FATAL 은 G-D 실패 ⟹ 여기) | 무효, 사유 이름 |
### 분기 끝

### 변경집합 — DC1_NE_CLOSURE (표)
| 경로 | 변경 |
|---|---|
| `src/nlte_population_candidate.h` | `nlte_ne_closure_candidate` 프로토타입 1개(개정 2) |
| `src/lumina_plasma.c` | `nlte_ne_closure_candidate(...)` 신설: 후보 배열(`candidate->level_population`·`candidate->ion_population`)로 `q_saha`·`q_closed` 계산, `candidate->electron_density[s]` 를 `q_closed` 로 덮어쓰기, `[A2-07][NE-CLOSURE]` 행 발행(0·비유한이면 FATAL 로 commit 거부); 호출 = bundle commit(`:16368`)·seed commit(`:16559`) **직전** 각 1회; env 미설정 시 즉시 return(코드 경로 진입 0, 부동소수 연산 0). 말단 `population_transaction_commit`(`:20928`)에는 손대지 않는다 |
| `src/env_universe.h` | 전집 도출 스크립트(references 의 derive 스크립트) 재생성으로 `LUMINA_NLTE_NE_CLOSURE` 등록(수작업 편집 금지) |
| `validation/instrumentation_debt/ENV_UNIVERSE.json` | 위와 같은 재생성 산출 |
| `tests/dc1_ne_closure_selftest.c` | 합성 원자(원소 2, 단계 3, NLTE 쌍 1): ON 에서 residual ≤1e-12·Saha 대 NLTE 단계 차이가 n_e 에 정확히 반영, OFF 에서 `n_electron` 배열 byte-동일; 음성대조 ≥4(NLTE 준위 하나 누락·전하 규약 뒤집기·비 NLTE 단계 누락·env=0 인데 발행) |
| `scripts/stage_superlevel_b.sh` | 모드 `DC1-OFF`·`DC1-ON` 신설: `DPROXY-PROV-K42` 복제 + 새 바이너리 sha 등록 + (ON 만) env 1개 추가(키 신규라 「추가」가 맞음, 중복 검사 유지); 자기 기술 7곳 모드명; selftest 양성 2·음성 ≥4. **개정 2**: 목적지 검사에 상대 성분(`.`/`..`) 거부와 realpath 가 봉인 루트(R-II·census) 아래이면 거부를 추가(astra Q0 차단), selftest 음성 3 추가; source identity 목록에 `src/nlte_population_candidate.h` 추가 |
### 변경집합 끝

## 5. 주장 상한
NC-A/B 어느 쪽도 「전하 보존이 옳게 닫혔다」를 뜻하지 않는다 — Saha 층의 5% 수렴 문턱과 Saha 단계(I, IV+·quarantine 원소)의 LTE 가중 Γ_phot 이 그대로다. CMFGEN 대조는 이 단의 산출로 하지 않는다(쌍둥이 T_e 레인 필요). 반복 2회·T_e 핀·K42 시대에 한정.

## 6. 분장 장부 (집행 기록 — 정본)
| 단계 | 규약상 | **실제** | 위반 |
|---|---|---|---|
| 사전등록 | Fable | Fable(운전석) | — |
| 코딩 | **운전자**(개정19) | 발주 6(gpt-5.6-sol, 개정19 전 발주분)의 착지분을 운전자가 검토·인수해 직접 고침 | |
| 사전감사 1(계약) | **astra** | `gpt-6-astra` read-only: **차단 4·기재 4** — `docs/ASTRA_PREAUDIT_DC1_R1_2026-09-06.md` → 개정 1 | 상류 원인 = 저자(삽입 지점·E-2 대수·G-C 피연산자·G-E 계수) |
| 사전감사 2(코드 검수) | **astra** | 1회차: **차단 3**(Q0 stager 봉인 내부 목적지 허용·Q6 selftest `residual=` 파서·E-1b seed 전제) 기재 5 — `docs/ASTRA_CODE_REVIEW_DC1_R1_2026-09-06.md` → 운전자 수정 + 개정 2 → 2회차 **통과, SAFE_TO_STAGE YES**, 기재 5 — `docs/ASTRA_CODE_REVIEW_DC1_R2_2026-09-06.md` | Q0 결함은 선재(stager), 나머지 둘은 운전자 |
| 발주·빌드·스테이징·제출 | 운전석 | 운전석 | |
| 판정 | Fable fresh | (기입 예정) | 운전석 금지 |
| 마감감사 | **astra** | (기입 예정) | |
집행 기록:
- 빌드(09-06 21:24 KST, 로그인 노드, worktree): `lumina_cuda` sha256 `ca721b07cb4fef632b1a0a0b2d3c7b11cdaac0e95f74e6cc56b131a7c5449059`, fatbin sm_80/86/90, 폐합 심볼 4건 실재(base `a4990a59` 0건). C selftest 양성 3·음성 7, strict-env·stager selftest PASS.
- OFF 스테이징(21:56 KST, grammar-debug): `/gpfs/kjhan/lumina/superlevel_b_a100x2_k36/slb_20260906T125629Z_dc1_ne_closure_off`, `SUPERLEVEL_B_STAGE_OK mode=DC1-OFF env_delta=11 shared_hardlinks=0`, seal census 136·R-II 135 PASS, identity 5파일(git_head 364ee72), G-C digest 4종 ✔, exports 에 closure 키 0·base 대비 경로 외 차이 0.
- OFF 제출: Slurm **333303** `LUMINA_DC1OFF` a100. 착지 09-07 08:38 KST: `FAILED 70`(예상된 post-model seal 오판), model.rc=0, 02:58:54 @ syn102.
- **G-A PASS** [운전석 실행, grammar-debug]: exact epoch 3행 base 와 문자열 동일(`45/9.6662782724980344e-09 · 53/7.8976358675842993e-09 · 53/8.5618986192583131e-09`), readout 338/338 짝 `n_upper_0,n_upper_1,Jbar_0,Jbar_1` 문자열 동일(`out_dc1off/`), ROW 685, NE-CLOSURE 행 0, 가드 0 ⟹ 새 바이너리 `ca721b07` 의 OFF 물리 = base.
- ON 스테이징(08:40 KST): `/gpfs/kjhan/lumina/superlevel_b_a100x2_k36/slb_20260906T234024Z_dc1_ne_closure_on`, `SUPERLEVEL_B_STAGE_OK mode=DC1-ON env_delta=12`, G-B 사전 = `> LUMINA_NLTE_NE_CLOSURE="1"` 1행만, G-C digest 동일. ON 제출: Slurm **333531** `LUMINA_DC1ON` a100.
- 프리플라이트 rc: `check_prereg_preflight` 0(1차 FAIL `changeset-extra-file` — 표 안 스크립트 경로 리터럴 제거 후 0) · `check_gate_referents` 0 · `check_doc_canonical` 0 · `check_dispatch_preflight` 0.
- 발주 6 = `/gpfs/kjhan/lumina/dproxy_audits/dc1/ORDER_DC1_CHANGESET.md`, 계약 HEAD `34a113e`, 작업 디렉토리 = worktree(§2). worktree 에 Makefile strict-env 타깃·`tests/env_universe_strict_selftest.py`·`scripts/derive_env_universe.py` 를 메인 트리에서 복사(identity 밖). `[미실행]` 없음. 발주 중단 없음.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 — DC1_NE_CLOSURE (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "(?:src|tests|scripts|validation)/[A-Za-z0-9_/.]+\\.(?:c|h|py|sh|json)",
    "symbol": "DC1_NE_CLOSURE",
    "roots": ["src", "tests", "scripts", "validation"],
    "expected_extra": ["src/lumina_plasma.c",
                       "src/env_universe.h",
                       "validation/instrumentation_debt/ENV_UNIVERSE.json",
                       "src/nlte_population_candidate.h",
                       "tests/dc1_ne_closure_selftest.c",
                       "scripts/stage_superlevel_b.sh"]
  },
  "branches": {
    "regimes": [[0.0, 0.45], [0.9, 1.1], [2.0, 3.0], [10.5, 60.0]],
    "metrics": {
      "g_fail": "sum(1 for x in v if x < 0.5)",
      "material": "sum(1 for x in v if 2.0 <= x <= 3.0)",
      "broken": "sum(1 for x in v if x > 10.0)"
    },
    "rules": [
      {"name": "NC-X", "predicate": "g_fail > 0"},
      {"name": "NC-N", "predicate": "g_fail == 0 and broken > 0"},
      {"name": "NC-B", "predicate": "g_fail == 0 and broken == 0 and material > 0"},
      {"name": "NC-A", "predicate": "g_fail == 0 and broken == 0 and material == 0"}
    ],
    "adversarial_fixtures": [
      {"name": "gates-pass-inert", "mix": [[1.0, 9]]},
      {"name": "gates-pass-material-shell", "mix": [[1.0, 8], [2.5, 1]]},
      {"name": "gates-pass-premise-broken", "mix": [[1.0, 8], [42.0, 1]]},
      {"name": "gate-fail", "mix": [[1.0, 7], [0.3, 1], [2.5, 1]]}
    ],
    "residual": "NC-X"
  },
  "references": [
    {"path": "src/lumina_plasma.c",
     "flags_existing": ["compute_electron_density", "population_transaction_commit", "work_ne", "ion_pop_stage", "nlte_level_populations"]},
    {"path": "src/env_universe.h",
     "flags_existing": ["LUMINA_NLTE_PER_ION_RESCALE"]},
    {"path": "scripts/derive_env_universe.py",
     "flags_existing": ["getenv"]},
    {"path": "scripts/stage_superlevel_b.sh",
     "flags_existing": ["DPROXY-PROV-K42-RESCALE"]},
    {"path": "scripts/dproxy_readout/cross_run_ratio.py",
     "flags_existing": ["source_function"]},
    {"path": "scripts/dproxy_readout/spectrum_contrast.py",
     "flags_existing": ["wavelength_angstrom"]}
  ]
}
```


## 개정 1 (2026-09-06, astra 사전감사 1회차 반영 — OLD/NEW 8쌍, 전건)
- 삽입 지점: 말단 commit → 후보 파이프라인(bundle·seed commit 직전, `candidate->electron_density`). G-E 150행(commit 3).
- 발행 기록: `q_saha`·`q_closed`·`dlog_saha_residual`·`dlog_resplit` 분해, 0·비유한 처분(FATAL, floor 금지).
- E-2 → E-2a(Saha 잔차 상한 0.05, 구성상)·E-2b(재분배 항 shell 0 도출 ≲1e-5, Si/S/Ca 질량분율 0)·E-1b(seed 대조).
- G-C: `resolved_lumina.sha256` cmp 삭제(스테이징 전 exports 해시) → stage seal digest 4종 + RUN_SEAL_PASS 2행. G-A `--out-dir`.
- 분기: NC-A 「스펙트럼 귀결 없음」 삭제, NC-N 을 E-2b/E-1b 전제로 한정, 미수렴은 NC-X.
- 분장 장부: 개정19(astra 감사·운전자 구현).
행 단위 diff 로 의도 외 소실 0 확인 [저자 실행].


## 개정 2 (2026-09-06, astra 코드 검수 1회차 반영 — OLD/NEW, 전건)
- E-1b: 구성상 대조 → 측정(seed 도 NLTE core 경유). NC 분기에서 E-1b 삭제.
- G-E: 거부 commit 흔적 행 제외, `residual=` 파싱 규약.
- §1 같은 반복 안 소비 기재(χ_es·χ_ff 즉시 반영, CPU χ/η·bf 는 폐합 전 값 — 두 n_e 시대 공존, 측정 대상).
- 변경집합: 헤더 프로토타입 행, stager 봉인 내부 목적지 거부 + identity 목록. 선언 블록 expected_extra 에 헤더.
행 단위 diff 로 의도 외 소실 0 확인 [저자 실행].
