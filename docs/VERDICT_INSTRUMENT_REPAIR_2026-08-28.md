# 판정문 — DET-INSTRUMENT-REPAIR: 판독 증거의 판별력·귀속 수리 — 분기 **IR-A (수리 성립)** (2026-08-28 **재개정 2판 — 감리 A·B 반영**; 초판 IR-X 는
`.rev1`, 재판정 1판은 `.rev2` 로 byte-보존)

- **단**: DET-INSTRUMENT-REPAIR. 계약 정본 = `docs/RUNG_INSTRUMENT_REPAIR_2026-08-28.md`
  (커밋 `ddf6701`; 판정 시점 HEAD 와 diff 0 [V2] — 이하 «계약»).
- **구현**: 커밋 `d597258` — 변경집합 정확 2파일 `scripts/analyze_a210_z2a_probe.py`(+199/−16)
  · `scripts/analyze_l6c_redispatch.py`(+151/−21), 합계 +350/−37 [V2, `--numstat`].
  GPU 런 0회(계약 §9).
- **판정자**: 초판·재판정 1판 = Fable fresh (분담 개정14 ③) — 사전등록·구현·검수·감리 무관여
  컨텍스트. ★**본 재개정 2판(감리 반영 정정) = 운전석** — user 지시(2026-08-28)로 9월 2일
  08:00 까지 Fable 몫을 Codex 가 임시 승계하되 **판정과 코드 검수는 운전자가 맡는다**.
  ⚠**자기 채점 위험 명시**: 정정 대상 4위치는 전부 **운전석 자신의 규약 위반 기재**이고
  정정 방향은 운전석에게 유리하다(입증되지 않은 위반 1건의 철회). 그래서 정정 문면은
  **감리-A 가 지정한 문면을 그대로** 옮겼고 무르게 만들지 않았다 — 감리-A 가 남긴
  「기재 지연 위반은 존속」·「발주 전 집행은 미확증」 두 문장이 §2 G-0 행·§8-3(a) 에 그대로
  살아 있다. 재감리는 이 diff 를 그 기준으로 검사해야 한다.
- **장부 = 계약 §12 (집행 기록)** — 본 문서는 장부를 복제하지 않고 지목한다(개정16 ①).
  판정자는 §12 분장 장부의 **③ 판정 행만** 기입했다.
- ★**본 문서는 폐합 선언이 아니다.** 폐합 여부·감리 상태의 정본은 **장부(계약 §12)** 다 —
  판정문 문면에서 찾지 마라. (근거: 판정문 문면 리터럴 매칭이 fail-open 벡터로 등재됨 —
  `docs/VERIFICATION_REGISTERS.md:365-367`.)
- **표기**: `[V#]` = 판정자 실행 명령(부록 A, 전건 2026-08-28 실행 — V1~V12 초판·V13~V19 재판정) ·
  `[판독]` = 운전석 판독 실측(`/tmp/claude-10396/INSTR_READOUT.md`, 105행 — 추가 집행 절 포함,
  ⚠노드-로컬 /tmp, syntax 로그인) ·
  `[계약 C#]` = 계약 §14 저자 명령 원장. 소스 행 번호는 전건 판정자 직독 확인 [V12];
  본 문서는 C 소스를 인용하지 않으므로 전처리 死블록 규정(`lumina_plasma.c:1324-1693` ·
  `lumina_cuda.cu:1083-2063`)은 해당 없음(계약 §2 동일 판정).
- **판본**: 초판(2026-08-28, 분기 IR-X — 게이트 3행 미집행 지목·정지·보고)은
  `VERDICT_INSTRUMENT_REPAIR_2026-08-28.md.rev1` 로 byte-보존 [V18]. 본 판본 = **재판정** —
  초판이 지목한 잔여 집행(G-A4·G-B2·G-B3 + G-0 후반부 rc·장부 기입)이 운전석에 의해 완료된
  전제 위에서, 계약 게이트 9행 전부를 계약 문면으로 재검증했다. 초판의 판정 중 본 판본이
  갱신하지 않는 절(§5·§7 처분)은 문면 그대로 유효하다.
  ★**재개정 2판(본 판본)** = 감리-A(차단 1건: 「소급 집행」 무증거 단정)와 감리-B(차단 4건:
  자기증명 주장 불일치 + 환경 재현 불능 3건)를 반영한 **관측층 정정**이다. **판정식·분기는
  손대지 않았다** — 감리-A 가 「분기 영향 없음, 정정 범위가 지정 네 위치를 벗어나지 않는 한
  재감리는 그 diff 확인으로 족하다」고 판정했고, 본 판본의 변경은 그 네 위치 + 감리-B #4·#16
  + §8 신규 관측 2건(8·9)에 국한된다. 재개정 시 본 판본을 `.rev3` 으로 보존.

---

## 0. 세 줄 요약

> ① 초판(IR-X)이 지목한 잔여 — G-A4·G-B2·G-B3 집행 + G-0 후반부 rc·장부 §12 기입 — 가
> 완료됐고, 판정자가 **계약 게이트 9행 전부**를 계약 문면으로 재검증했다: 추가 산출물 **5본
> 전건을 같은 입력으로 재실행해 byte-동일 재현**(G-A4 3본 [V14]·G-B2 [V16]·G-B3 [V15]),
> rc 는 판정자 **직접 재실측**(운전석 러너의 rc 포집은 무효 — §8-4). ② **9행 전건이 등록
> 기대치와 일치** ⟹ 분기 = **IR-A (수리 성립)**; IR-F1~F4 전건 미발화 — 특히 G-B3 은 PASS
> 가 아니라 1차 기대 그대로의 이름 있는 FAIL(`RII_NOT_TERMINAL` record_count=0)이다(IR-F2
> 미발화; 수리 전 같은 자리는 침묵 PASS 였다 [계약 C7]). ③ 시연의 경계는 넘지 않는다:
> 결함 A 는 계약 §1(i) 전건 시연(G-A4 실물 사본 음성대조 포함 — 단 **1차 시도는 시험 무효**,
> §4-1), 결함 B 는 기계 시연이 **라이브 양·음 쌍까지** 확장됐으나 **«정본 증거의 취득은
> 규율» 한계는 그대로**다(§6). Z-2A 의 ZI-A 판정 불변(§5). ★**재개정 2판이 철회한 것 2건**: 「G-0 후반부를 판정 후
> **소급 집행**했다」(무증거 — 확정 가능한 이탈은 **§12 기재 지연**뿐; §8-3(a)) 와
> 「PR-0 산출물이 **집행 클러스터를 자기증명**한다」(두 JSON 에 실행-호스트 필드가 없고
> `pr0_grammar.json` 이 다른 클러스터에서 byte-동일 재생산된다; §6-4(b)). 둘 다 관측층이라
> **분기 IR-A 는 불변**이다. 본 문서는 폐합 선언이 아니다.

---

## 1. 입력 식별 (전건 판정자 실측)

| 항목 | 실측 |
|---|---|
| 계약 | `docs/RUNG_INSTRUMENT_REPAIR_2026-08-28.md` — `ddf6701` 수록, `git diff ddf6701..HEAD` 공(변경 0) [V2]. ★재판정 시점 워킹트리 = **M(+11/−11)**: 전건 §12 집행 기록 기입(필수 4항목·분장 장부 — 운전석) — 판정자 행 단위 실사: 계약 본문 §0~§11·§13·§14 무접촉; 단 감리-A 행 «규약상 담당» 칸 문면 1건 변경 관측(§8-3(c)) [V18] |
| 구현 | `d597258` — `git show --stat`: 정확 2파일, +350/−37 (계약 §4 변경집합과 1:1) [V2] |
| byte-사슬 | 검수 패치 `INSTR_PATCH.diff` ≡ `git show d597258` patch (cmp 무차) · `/gpfs/kjhan/lumina/zo_artifacts/probe_repaired.py` ≡ `probe_committed.py` ≡ HEAD `scripts/analyze_a210_z2a_probe.py` (cmp 무차 ×2) · worktree clean [V2·V3] ⟹ 검수·집행·커밋이 전부 같은 바이트 |
| 봉인 R-II | `/gpfs/kjhan/lumina/a210_z2a_diag_isolation_a100x2_k36/rii_20260826T134337Z_ion1` (`<RII>`) — 등록 경로 = Z-2A 판정문 `:417` |
| 봉인 census | `/gpfs/kjhan/lumina/a210_zero_opacity_z1r_a100x2_k36/zo_20260826T050021Z_census_ion1` (`<ZO>`) — 등록 경로 = Z-2A 판정문 `:418` |
| 봉인 무결 | 운전석 전수(전 게이트 후): `RII_RUN_SEAL.sha` 135/135 OK · `ZO_RUN_SEAL.sha` 136/136 OK [판독:60,105]. 판정자 spot(재판정 시점 — 전 재현 이후): .sha 행수 135·136 · 양 봉인 `stderr.log` sha256 재계산 = 지문 일치 · 지문 파일 mtime RII 08-27 17:32 / ZO 08-26 18:16(전건 전 게이트 이전)·모드 440 [V17] · 앵커 4행 자면 = 계약 §2-A 십진값 그대로 [V8] · G-A4 무변조 사본 ≡ 봉인 census `stderr.log` **byte-동일**(cmp 전장) [V14] |
| 게이트 산출물 | `/gpfs/kjhan/lumina/zo_artifacts/` — 초판분 `{za_selfcheck,selfinput_after,ga5_retro}.json` (11:57/11:56/14:52) [V9] + ★추가 집행분 `{ga4,ga4_clean,ga4b}.json`(15:27-29) · `pr0_syntax.json`(15:26) · `pr0_grammar.json`(15:27) · 사본 3본 `ga4_scratch/{clean,tampered,tampered2}.log` · 변조 스크립트 `ga4_scratch/mk.sh`·`mk_ga4.py` · 러너 `run_pr0.sh`·`run_gb2.sh` [V19] — ⚠계약 등록 거처 `validation/instrument_repair/` 는 여전히 미신설(§8-1) |
| 판독 실측 | `/tmp/claude-10396/INSTR_READOUT.md` (운전석, 최종 15:30 — 105행, 추가 집행 절 포함) — 인용 수치 전건 판정자 대조 |
| 판정자 재현 거처 | `/gpfs/kjhan/lumina/instr_verdict_repro_rev1/` — 재현 산출 7본 + 발주서 사본(`codex_instr_order.md`, mtime 08:39 보존) + 발주시점 재구성 트리 `tree_ddf6701`(git worktree, detached·clean) [V14~V16] |

주: `za_selfcheck`·`selfinput_after`(11:56-57)는 커밋(14:46:48)보다 앞서나, 그 시점 실행본
`probe_repaired.py` 가 커밋본과 byte-동일이므로 [V3] 세 산출물 전부 같은 판독기 바이트에
귀속된다. 입력 귀속(어느 stderr 를 넣었는가)은 산출 JSON 에 경로가 없어 문면으로는 열려
있으나, 판정자가 등록 입력으로 재실행한 산출과 **3본 전건 byte-동일**이라 잠긴다 [V5·V6].

---

## 2. 게이트별 판정 — 계약 §5 의 **9행 전부** (판독기·게이트 라벨 무의존, 판정식 재적용)

| 게이트 | 계약 판정식 (요지) | 집행 (주체·노드) | 실측 (판정자 재검증) | 판정 |
|---|---|---|---|---|
| **G-0** | 사전등록 프리플라이트 rc=0(§12 기재) **+** 발주 시 `check_dispatch_preflight` rc=0 | 전반부: 저자, 발주 전, grammar-debug [계약 C9]. 후반부: 운전석, grammar-debug — ★**집행 시각 직접 증거 부재** — 기록 rc=0 은 구현 착지(14:46:48) 이전 상태에서만 산출 가능 [V15]; 확정 이탈은 **§12 기재 시점**(초판 후 기입), §12 항목 1 기재 | 전반부 rc=0 [계약 C9]. 후반부 rc=0 — DP-1 `no-restatement` tokens_checked=5 · DP-2 `frontpage-sealed` · DP-3 `contract-preflight rc=0` · DP-4 [§12 항목 1]. ★판정자 재현 [V15]: **발주시점 재구성**(발주서 사본 mtime 08:39 + `ddf6701` clean 트리)에서 **rc=0, 4행 전건 동일**; **라이브 상태 재실행은 FAIL** `contract-worktree-drift` rc=1 — rc=0 은 발주시점 상태에서만 재현되고, 그것이 §11 시점-술어 규정의 옳은 거동이다 | **충족 — 단 후반부는 기재 지연 동반** 충족; **집행-시점 위반은 미입증**(발주 전 집행이 rc=0 과 가장 정합)이고 확정 이탈은 §12 기재 시점이며 §8-3(a) 로 기재. 게이트 FAIL 은 아니다 — 인증 내용 자체는 발주시점 입력으로 참임이 이중 확립 |
| **G-A1** z2a selftest | NC-I0~I8 전건 — 성한 반복-구별 픽스처 PASS ∧ 주입 이름 있는 FAIL ∧ 복원 PASS | 운전석(1차) + 판정자 | `PASS controls=NC-I0..I8 failed=0` rc=0 [판독:12 + 판정자 재현 V4 + **재판정 시점 재확인 V13**]. NC-I7: `inject=ITER1_WITH_ITER0_ANCHOR → FAIL ANCHOR_FIELD_MISMATCH` / 복원 PASS [V4] | **PASS** |
| **G-A2** census 자기입력 양성대조 (PR-Z1) | `--census-selfcheck <ZO>/stderr.log` → rc=0 — «봉인 census 자신을 넣으면 PASS 해야 한다» | 운전석(Q0 후) + 판정자 | rc=0 · `status=PASS anchor_line=669992 input_access=READ_ONLY streaming=1` [판독:29 + 판정자 재현 V5, 산출 byte-동일] | **PASS — 자격 완성**: 초판이 유보한 이 게이트의 등록 음성대조(G-A4, 실물 사본 팔)가 집행됐다(아래 행) |
| **G-A3** 결함 명령 쌍대 (PR-Z2) | 원 결함 시연 명령 그대로(`--mode R-II --stderr <ZO> --census-stderr <ZO>`) → blocking **정확** `['SPROBE_COUNTER_MISSING']`·앵커 계열 0건·f_bit=1.0 | 운전석 + 판정자 | `status=FAIL branch=ZI-B blocking=['SPROBE_COUNTER_MISSING'] f_bit=1.0` rc=4 — 앵커 계열 0건 [`selfinput_after.json` + 판정자 재현 V6, byte-동일]. 대조군(수리 전 동일명령 `['SPROBE_COUNTER_MISSING','ANCHOR_FIELD_MISMATCH']`) = [계약 C3]·Z-2A 판정문 §3-(b) 실재. 이 FAIL 은 계약 `:149` 가 등록한 기대 형식 그 자체다(IR-F3 의 «게이트 실패» 아님) | **PASS** |
| **G-A4** 변조 사본 음성대조 | census 스크래치 사본(`/gpfs`, /tmp 금지)의 iter1 앵커 `n_upper` 1자리 변조 → `--census-selfcheck` 이름 있는 FAIL(`ANCHOR_FIELD_MISMATCH`) · 사본 생성 시 봉인 read-only | 운전석(추가 집행, 2차 시도 유효) + 판정자 | 무변조 사본 ≡ 봉인 census **byte-동일**(cmp 전장 [V14]) · 변조 = 정확 **1 byte**(`'2'→'7'`, **6번째** 유효자리(소수 9자리째) — 십진 `…4235457…→…4735457…`, hex `0x1.daa141c1395e6p-12→0x1.daa2995a27817p-12` 판정자 왕복 재계산 [V14]) → **FAIL `ANCHOR_FIELD_MISMATCH detail=iter=1`** rc=4 [`ga4b.json`] · 대조군(무변조) **PASS** rc=0 [`ga4_clean.json`]. 판정자 재현 **3본 전건 byte-동일**(변조·대조군·무효변조 [V14]) | **PASS**. ★운전석 **1차 시도는 시험 무효** — 게이트 표본 불산입, 처분은 §4-1 |
| **G-A5** 소급 재판독 (PR-Z3) | `--stderr <RII> --census-stderr <ZO>` → status=PASS · branch=ZI-A · f_bit=1.0 · blocking=[] (4항) | 운전석 + 판정자 | 4항 전건 일치 [`ga5_retro.json` + 판정자 재현 V6, byte-동일]. 대조군(수리 전 기준선 `FAIL ZI-B ['ANCHOR_FIELD_MISMATCH'] f_bit=1.0`) = [계약 C4] + 봉인 M14 판정자 직독 [V8] | **PASS — 지위는 §5** («계측 검증 관측, 판정 불변») |
| **G-B1** l6c selftest | NC-R0~R18 전건 (기존 R0~R16 존속 + 신설 R17·R18) | 운전석(1차) + 판정자 | `PASS controls=NC-R0..R18 failed=0 synthetic_fixtures=1 sealed_access=0` rc=0 [판독:13 + 판정자 재현 V4 + **재판정 시점 재확인 V13**]. NC-R17 2팔·NC-R18 4팔 전건 발화 [V4] | **PASS** |
| **G-B2** 라이브 양성대조 (syntax) | 수리된 `check_pr0_unsealed` 를 syntax 에서 → PASS ∧ 레코드 정확 `324557\|LUMINA_Z2A_II\|FAILED\|70:0\|syntax` | 운전석(추가 집행) — ⚠등록 형식(unsealed 반쪽·봉인 무접촉)이 아니라 **전체 판독기 상위집합**으로 집행(봉인 read-only 접촉 — §8-5) | PR-0 절: `status=PASS` ∧ `sacct_records=['324557\|LUMINA_Z2A_II\|FAILED\|70:0\|syntax']` — 등록 문자열과 **일자일자 일치** [`pr0_syntax.json`]. 등록 unsealed 검사는 상위집합 안에 **원문 그대로** 들어 있다(`check_pr0` 가 `check_pr0_unsealed` 를 내부 호출, `:1250` [V16]). 판정자 재현: 전체 형식 **rc=0·보고서 byte-동일** [V16] + **등록 형식 그대로**(unsealed 단독) 별도 실행 PASS·레코드 동일 [V16]. ★집행 클러스터의 산출물 자기증명 주장은 **철회**됐다(§6-4(b)) — 귀속은 집행 기록에 의존 | **충족** — 형식 이탈은 §8-5 기재(판정식 두 절은 두 형식 모두에서 성립) |
| **G-B3** ★라이브 교차-클러스터 음성대조 (grammar-debug) | 같은 호출을 grammar-debug 에서 → 이름 있는 FAIL 의무(1차 기대 `RII_NOT_TERMINAL` record_count=0; 허용 집합 §6) · **PASS 는 즉시 IR-F2** | 운전석(추가 집행, grammar-debug — 계약 §10 의 «의도된 오노드») + 판정자(grammar-debug 재현) | **FAIL `RII_NOT_TERMINAL detail=record_count=0` branch=X-P** [`pr0_grammar.json`] — 허용 집합의 **1차 기대 그대로**·PASS 아님. 판정자 재현: grammar-debug 에서 rc=4·**보고서 byte-동일** [V15]. 원시 대조 [V15]: grammar 로컬 `sacct` → `324557\|UPP_QE68_SCF\|COMPLETED\|0:0\|grammar`(실해 레코드 존속) · `-M syntax` → **빈 출력 rc=0**. 수리 전 같은 자리 = `PASS ['324557\|COMPLETED\|0:0']` [계약 C7] — 침묵 오귀속이 이름 있는 FAIL 로 바뀌었다. fail-closed 로 봉인 무접촉(`PR-0.sealed_inputs_opened=false`) | **충족** (IR-F2 미발화) |

**부기 (게이트 표 밖 실측)**: PF-1 표지 `A210_INSTRUMENT_REPAIR` — `grep -rln … scripts src`
정확 2파일 [판독:15 + 판정자 재현 V4]. `py_compile` 2파일 OK [판독:14]. **사전등록**
프리플라이트의 판정 시점 재실행은 초판과 같은 이유로 하지 않았다(§11 시점 술어 — 구현 착지
후 재실행은 정의상 FAIL; 판정 대상은 발주 전 rc=0 기록 [계약 C9]). **발주** 프리플라이트는
같은 술어를 **발주시점 상태 재구성**으로 평가했다 [V15] — 라이브 재실행 FAIL 이 재구성의
필요성을 실측으로 증명한다(G-0 행). 두 판독기 스크립트의 워킹트리 = HEAD `d597258` 와 무차
(git status 공) [V13].

---

## 3. 기대치·분기 재판정 (계약 §6 — 판정식 재적용으로 분기 확정)

**세부 기대치 대조** (재판정에서 전건 닫힘):

- G-B2 기대 레코드(등급 1): 등록 문자열 `324557|LUMINA_Z2A_II|FAILED|70:0|syntax` — 게이트
  산출물 `pr0_syntax.json` 의 `sacct_records` 와 **일자일자 일치**, 판정자 syntax 직접
  조회와도 일치 [V1·V13·V16].
- G-B3 허용 FAIL 집합(등급 2) = {`RII_NOT_TERMINAL`(1차 기대 — record_count=0),
  `SACCT_PROVENANCE_MISMATCH`, `SACCT_UNAVAILABLE`}: 실측 = `RII_NOT_TERMINAL`
  `record_count=0` — **집합 원소이자 1차 기대 그 자체**. 집합 밖 사유 0 ⟹ IR-X 유입 없음.
- G-A2·G-A3·G-A5 기대(등급 1)의 등록 근거(«수리 전 실측에서 앵커 사유를 제외한 모든 검사가
  이미 통과» — 계약 §6)는 수리 전 실측 [계약 C3·C4]·봉인 M14 [V8] 로 성립 재확인(초판 그대로).
- G-A4 기대(이름 있는 FAIL `ANCHOR_FIELD_MISMATCH`): 실측 일치 + detail 이 반복 인덱스
  (`iter=1`)까지 특정 — 계약 §3-A 의 반복-구별 의도와 정합 [V14].

**상위 분할 적용** (판독기 라벨이 아니라 계약 판정식으로):

| 라벨 | 판정식 | 대조 | 발화 |
|---|---|---|---|
| **IR-A** | G-0~G-B3 **전건** 충족 | §2 표: **9행 전건 충족** — 초판의 미집행 3행(G-A4·G-B2·G-B3) 집행·기대 일치, G-0 후반부 rc=0 기재·판정자 재현, G-A2 자격 완성 | ★**성립** |
| IR-F1 | G-A5 산출이 PASS·ZI-A·f_bit=1.0·blocking=[] 과 다름 | 4항 전건 일치 [V6] | 미발화 |
| IR-F2 | G-B3 이 PASS | G-B3 = 이름 있는 FAIL(1차 기대) — PASS 아님 [V15] | 미발화 |
| IR-F3 | 그 밖의 G-* FAIL | 게이트 수준 기대 이탈 0 — G-A3·G-A4·G-B3 의 FAIL 라벨은 각 판정식이 **요구한 기대 형식**이다(판독기 exit 라벨 ≠ 게이트 실패) | 미발화 |
| IR-F4 | 환경 차단 (sacct 불능·봉인 접근 불능·grammar-debug 불통) | syntax sacct 가용 [V1·V13]·봉인 접근 가용 [V5-V8·V14]·grammar-debug 왕복 가용 [V15] — 차단 관측 0 | 미발화 |

⟹ **분기 = IR-A (수리 성립)** — 계약 §6 상위 분할 «게이트 전건 충족» 이 9행 전부의 실측으로
성립하고, 어떤 IR-F 판정식도 발화하지 않는다. 초판의 IR-X 는 «집행 미완» 관측으로서
옳았고(그 시점의 산출), 그 관측이 지목한 잔여가 전건 집행됨으로써 해소됐다 — 골대는
움직이지 않았다(판정식·기대치·허용 집합 전건 계약 `ddf6701` 문면 그대로 [V2·V18]).

**판정의 한정 두 건** (IR-A 에 얹는 명시 조건이 아니라 기재 — 게이트 FAIL 아님):
① G-0 후반부는 **기재 지연 동반** 충족이다(§2 G-0 행·§8-3(a)) — **집행-시점 위반은
미입증**(발주 전 집행이 rc=0 과 가장 정합). ② G-B2 는 집행 형식 이탈 동반
충족이다(§2 G-B2 행·§8-5). 두 이탈 모두 판정식의 어느 절도 거짓으로 만들지 않음을 판정자가
재현으로 확인했다 [V15·V16].

**처분** (계약 §6 IR-A 행 문면): 폐합 절차의 잔여 단계로 넘어간다 — 그 절차의 상태·결과의
정본은 **장부(계약 §12)** 이며, 본 문서는 그것을 선취하지 않는다(§0 머리 — 본 문서는 폐합
선언이 아니다).

---

## 4. 결함 A (앵커 반복 스코프 — 판별력 0) — 무엇이 확정됐고 무엇이 아닌가

**확정 (전건 실측)**:

1. **원 결함 문면의 부정이 봉인 실물에서 성립한다.** 결함의 등재 문면 = «봉인 census 자신을
   넣어도 FAIL»(대장 `VERIFICATION_REGISTERS.md:360-362`). 수리 후: census 자기입력 rc=0 PASS
   (G-A2 [V5]) · R-II 모드 자기입력에서 `ANCHOR_FIELD_MISMATCH` **소멸**, 잔여 = 계약 `:149`
   가 예견한 구성적 잔여 정확 1건(`SPROBE_COUNTER_MISSING` — census 는 토큰 이전 산출물)
   (G-A3 [V6]).
2. **수리 실체가 계약 §3-A 문면과 1:1 이다** [V12]: `ANCHOR_BITS_BY_ITERATION`(`:31` — iter0
   값 불변·iter1 신설, 등록 hex 4값 = 봉인 십진값의 변환과 전건 일치 [V8]) ·
   `anchor_matches(raw_rows, iteration)`(`:198`, 범위 밖 `ANCHOR_ITERATION_INVALID` `:207`) ·
   호출부 `:303` 이 루프 반복 인덱스를 전달 · R-III/R-IV 경로 무접촉(diff 에 해당 함수 hunk 0
   [V12]).
3. **재발 차단이 상설 게이트가 됐다.** 원 selftest 맹점(픽스처가 양 반복에 동일 행 재사용 —
   대장 `:364`)이 `synthetic_zero_row(iteration)` 로 제거됐고, NC-I7/I8 이 원 결함 계급
   (반복 무구별 등록값)을 잡는다. **판정자 역주입 재현**: iter1 등록 hex 2값을 iter0 값으로
   회귀시킨 스크래치 사본에서 selftest 가 `FAIL failed=12` rc=1 [V11] — 검수 실측
   [판독:65-66]과 동수. 역주입 상태에서 NC-I8 inject 팔이 `status=PASS`(변조 census 통과)로
   반전됨을 관측 [V11] — 이 게이트가 정확히 원 결함 계급을 밟고 서 있다.
4. ★**실물 사본 음성대조가 집행돼 판별력의 쌍이 실물에서 닫혔다** (G-A4 — 초판의 마지막
   공백). 봉인과 byte-동일한 사본은 통과하고(대조군 PASS rc=0), 유효숫자 안쪽 1자리(1 byte)
   변조는 반복 인덱스까지 특정된 이름 있는 FAIL(`ANCHOR_FIELD_MISMATCH detail=iter=1` rc=4)
   로 거부된다 — 판정자 3본 byte-동일 재현 [V14]. 이로써 계약 §1(i) 의 두 절(«성한 봉인
   실물을 통과» ∧ «주입 결함을 이름 있는 사유로 거부»)이 **합성 픽스처(NC-I7/I8)와 봉인
   실물 사본 양쪽에서** 시연됐고, G-A2 의 PASS 자격(음성대조 의무)이 완성됐다(§2).
5. G-A5 소급 재판독이 등록 기대치 4항과 일치 [V6] — 지위는 §5.

### 4-1. G-A4 운전석 1차 시도의 처분 — **시험 무효 (게이트 표본 불산입)**

**사실 (전건 판정자 재검증 [V14])**: 1차 시도는 iter1 앵커 `n_upper` 십진 자면의 **마지막
자릿수**를 바꿨다(`…62295→…62294`, 1 byte — `mk.sh` 판정자 직독). 그 십진값은 double 왕복에서
**같은 bit** 다 — 판정자 독립 재계산: 마지막 자릿수를 3·4·6·7 로 바꾼 네 값 전부
`0x1.daa141c1395e6p-12` 로 왕복(그 자리 ±1 = 1e-20 < 반 ulp ≈ 2.7e-20). 게이트는 이 사본을
PASS 시켰고(`ga4.json`), 판정자 재실행도 rc=0 PASS·byte-동일 [V14].

**판정 세 항**:

1. ★**이 1차 PASS 는 «G-A4 가 한 번 통과했다(주입을 못 잡았다)»로도, «G-A4 음성대조가
   집행됐다»로도 세어지지 않는다.** 변조가 bit 항등이므로 **주입된 결함이 0 이다** — 음성대조
   시험으로서 무효이고, PASS 는 앵커-bit 무변경 입력에 대한 게이트의 **옳은 결정론적
   거동**이다(무변조 대조군과 동치). 게이트 결함의 증거도, 게이트 통과 실적도 아니다 —
   표본에서 제외하고 기록한다. G-A4 의 집행 실적은 2차(유효 변조) 단독이다.
2. **운전석의 처리 자체는 규약이 명하는 그대로였다** — 게이트 결함으로 오판하지 않고 자기
   시험부터 검사했다(«잣대부터 감사»). 2차 변조 스크립트(`mk_ga4.py`)는 변조 유효성 검증
   (`float(cand).hex()!=float(v).hex()`)을 내장하고 **6번째** 유효자리를 변조했다 — 판정자 직독
   + hex 왕복 재계산 일치 [V14].
3. ★**이 사건이 G-A4 의 음성대조 설계에 대해 말하는 것 — 계약의 한계다.** 계약 §5 G-A4 행의
   변조 사양은 «`n_upper` 1자리 변조» 뿐이고, **주입 유효성(변조가 비교 표현인 bit 를 실제로
   바꿨는가)의 검증 요구가 등록돼 있지 않다**. 십진 자면은 double 의 분해능보다 자릿수가
   많으므로(이 값에서 마지막 ~2자리는 잉여), 계약 문면을 충족하는 변조가 **널 주입**일 수
   있다 — 1차 시도가 그 실증이다. 이 단에서는 집행이 스스로 해소했으나(위 2), 해소는 등록되지
   않은 집행 스크립트에 있다 — **계급으로서의 한계는 존속**한다. 처분 = 조용한 대장 기재
   (§10 표 4행; 수리·도구 설계는 이 단 밖 — 계약 §8-7).

**확정하지 않는 것**:

- 이 수리는 **판독기(계측)의 판별력 복원**이다. **Z-2A 의 어떤 판정도 움직이지 않는다**
  (ZI-A 불변 — §5). **E-I1 문면 FAIL·R-III/R-IV 제출 불가**(Z-2A 판정문 §5·§7)의 해소가
  아니다 — 그 해소는 GPU 런이 필요한 별도 단이고 이 수리는 그 전제를 회복할 뿐이다(계약 §7
  말미·§13). 관측: 별건 판정런 `325913 LUMINA_Z2A_III RUNNING` · `325914 LUMINA_Z2A_IV
  PENDING` [V1] — 본 단 밖, 본 판정문은 판정하지 않는다.
- z2a 의 «어떤 오류든 ZI-B» 확장은 계약 §8-5 대로 **미수리 존속** — 앵커 오탐 제거로 그
  확장의 실해 경로(오탐→ZI-B)만 소멸했다.

---

## 5. G-A5 의 지위 — «계측 검증 관측, 판정 불변» (계약 `:283`·`:291` 등록 그대로)

- G-A5 가 확인한 것은 **«등록 기대치 4항과 일치했다» 까지**다. 본 판정문은 이를 «수리가
  검증됐다» 로 확대하지 않는다 — 수리의 시연 범위는 §4 확정 목록의 각 항이 개별적으로
  말하는 것까지다.
- **Z-2A 의 ZI-A 판정은 불변이다.** 그 판정은 판정자가 계약 §8-2 판정식 문면을 오탐 무관
  1차 실측으로 채워 내린 것이고 폐합됐다(감리 최종 상태 = 그 단 장부·판정문 스탬프 — 재판정
  금지). G-A5 는 그 판정과의 **일치 확인**이다 — 계측이 판정을 따라잡은 것이지 판정이 움직인
  것이 아니다.
- 불일치였다면 IR-F1(수리안 결함 — Z-2A 재판정 아님)이었다 — 발화하지 않았다(§3).

---

## 6. 결함 B (sacct 교차-클러스터 provenance) — 기계로 막힌 것 / 규율로만 되는 것

계약 §3-B 의 경계 판정을 실측에 대조해 **문장 단위로** 가른다. **이 문서 어디에도 «게이트가
전부를 덮는다»는 진술은 없다** — 게이트가 있다는 착각은 원 결함보다 위험하다.

**[기계 — 시연됨 (합성 채널·selftest 상설)]**

1. **오귀속 레코드의 수용 차단**: 삼중 검증(JobID·JobName·Cluster,
   `analyze_l6c_redispatch.py:1160-1162`) 불일치 → `SACCT_PROVENANCE_MISMATCH`(원문 레코드
   보존). 시연: NC-R17 2팔(텍스트 주입 — 검사 로직) + NC-R18(a)(**실물 subprocess 채널**의
   PATH-shim 이 역사적 실해 레코드 원문 `324557|UPP_QE68_SCF|COMPLETED|0:0|grammar` 를 방출)
   → 전건 이름 있는 FAIL [V4]. 실해 레코드(`UPP_QE68_SCF`)가 이제 기계로 거부된다.
2. **명령 표류 차단**: NC-R18(d)가 shim 이 수신한 argv 를 `expected_argv`(`:2014`) 와 **완전
   목록 등치**로 대조 — `-M syntax`·5열 `-o` 의 실재가 selftest 마다 기계 검증된다 [V4].
3. **오노드 집행의 «침묵 통과 → 이름 있는 FAIL» 전환 장치**: `-M syntax` 핀(`:1139`) +
   레코드 0건 → `RII_NOT_TERMINAL`(`:1158`, fail-closed 기존 유지 — `SACCT_UNAVAILABLE`
   `:1144`·비종단 `:1164` 도 존속). 낙하 경로는 NC-R18(c) `EMPTY_RC0 → RII_NOT_TERMINAL` 로
   shim 채널에서 시연 [V4]. 근거 실측(교차 조회 **양방향** 빈 출력·rc=0)은 [계약 C6, 저자
   실행].

**[기계 — 라이브 시연 완료 (재판정에서 갱신된 유일 블록)]**

4. **오노드 집행의 «침묵 통과 → 이름 있는 FAIL» 전환이 역사적 실해 채널 그 자체에서
   실기로 시연됐다** (G-B3, grammar-debug — §2). 문장 단위로 가른다:
   - **(a) G-B3 이 덮은 것**: selftest 가 원리적으로 못 덮는 잔여였던 «실행 노드의 실제
     slurmdbd 귀속»(계약 §3-B 원리 판정 — shim 은 진짜 sacct 이 아니다)이, 실해가 났던 바로
     그 노드의 **진짜 sacct** 로 시연됐다. 수리 전 같은 자리 = 침묵 오귀속 PASS
     `['324557|COMPLETED|0:0']` [계약 C7] → 수리 후 = `RII_NOT_TERMINAL record_count=0`
     이름 있는 FAIL [V15]. 실해 레코드(`UPP_QE68_SCF`)는 grammar 측에 **지금도 존속**하며
     [V15] 그 채널의 침묵 수용 경로는 죽었다.
   - **(b) ★산출물의 집행 클러스터 자기증명은 성립하지 않는다** — 재판정 1판의 이 주장을
     본 판본이 **철회**한다(감리-B 차단 사유 #4; 판정자 재확인 [V20]). 두 JSON 어디에도
     `hostname`·실행 호스트·실행 클러스터·`argv`/`command` 필드가 **없다**;
     `pr0_grammar.json` 의 PR-0 절은 `{"sealed_inputs_opened": false, "status": "FAIL"}` 와
     `detail="record_count=0"` 이 전부다 [V20]. ★**결정적 반증**: 감리-B 가 `syntax` 에서
     Slurm 소켓이 차단된 상태로 빈 sacct(rc=0)를 주입해 `pr0_grammar.json` 을 **byte-동일**로
     재생산했다 — 같은 산출물이 「틀린」 클러스터에서 만들어진다. ⟹ `record_count=0` 은
     «비-syn 집행» 이 아니라 «레코드를 얻지 못한 **모든** 경우»와 양립한다. 반대쪽도 대칭이
     아니다: 정본 5열의 다섯째 필드 `syntax` 는 **작업이 돈 클러스터**이지 판독기를 부른
     호스트가 아니며, 레코드를 얻었다는 사실은 syn slurmdbd 도달 가능성을 시사하는 **정황**일
     뿐 산출물에 박힌 증명이 아니다. ⟹ **집행 클러스터의 귀속은 여전히 운전석 진술과 집행
     기록에 의존한다** — §6-4 의 «정본 증거의 취득은 규율» 한계가 여기에도 그대로 걸린다.
     라이브 양·음 쌍이 실제로 덮은 것은 클러스터 귀속이 아니라 «수리된 판독기가 두 입력
     상태에서 각각 등록 기대대로 거동한다» 는 것뿐이다.
   - **(c) G-B3 이 덮지 않은 것 — 정밀 경계**: 라이브에서 발화한 팔은 «`-M` 핀 → 빈 출력 →
     fail-closed(`RII_NOT_TERMINAL`)» 팔이다. **삼중 검증의 오귀속 레코드 거부 팔
     (`SACCT_PROVENANCE_MISMATCH`)은 라이브에서 발화하지 않았고, 현행 코드로는 원리상
     발화할 수 없다** — `-M syntax` 핀이 타 클러스터 레코드의 도달 자체를 끊기 때문이다.
     그 팔의 시연은 shim 채널(NC-R18(a), 실해 레코드 원문)과 텍스트 주입(NC-R17)에 있다
     [V4]. G-B3 이 그 팔까지 라이브로 덮었다고 읽으면 과대주장이다.
   - **(d)** G-B3 관측이 허용 FAIL 집합(등급 2 사전등록) 밖으로 새지 않았다 — 1차 기대
     그대로(§3). 집합 등록이 IR-X 유입을 막은 실례다.
5. **G-B2**(syntax 라이브 양성)도 집행됐다 — 수리된 판독기가 정본 채널에서 PASS 에 실제로
   도달하고 5열 원문 레코드를 verdict JSON 에 자기 기술함(§3-B 4)이 시연됐다(§2 G-B2 행;
   집행 형식 이탈은 §8-5). 판정자는 전체 형식 byte-동일 재현(rc=0)과 등록 형식(unsealed
   단독) 실행 양쪽으로 재검증했다 [V16].

**[규율로만 — 게이트가 못 덮는다 (계약 명시 한계, 본 판정문도 넘지 않는다)]**

6. **정본 증거의 취득(PASS 도달)은 기계가 보장하지 못한다.** sacct 는 클러스터 귀속 상태이고
   교차 조회는 양방향 불능이므로 [계약 C6], PASS 는 syn slurmdbd 귀속 노드(실측 성립: syntax
   로그인 [V1])에서 집행해야만 가능하다. 기계는 오집행을 **가시화**(이름 있는 FAIL)할 뿐
   취득을 **대체하지 못한다**. 이 절반은 계약 §10 집행 조건(규율)이다 — 수리 후 달라진 것은
   «규율 위반이 침묵 오귀속이 아니라 이름 있는 FAIL 로 나타난다»는 것까지다.
7. G-B3 자체도 **상설 selftest 가 아니라 환경 의존 집행 게이트**다(계약 §3-B 한계 기재
   그대로) — 집행돼도 이 성격은 바뀌지 않는다.

---

## 7. 검수 지적 3건의 처분 (계약 §8-7·§8-8 — 이 단의 수리 권한은 §3 의 두 결함뿐)

계약이 이 단에 준 수리 범위 = §3-A 1~4 · §3-B 1~7, 변경집합 = 2파일의 그 항목들(§4), 그리고
§8-7 «발견의 처분 = 조용한 대장 기재 — **이 단의 범위를 늘리지 않는다**» · §8-8 «두 파일 밖
수리 금지·등록 상수 밖 값 변경 금지». ⟹ **세 건 모두 지금 수리는 계약 밖**이다. 처분은
전건 **조용한 대장 기재**(기재 행위는 운전석 몫 — 아래는 기재 후보 문면)이며, 수리는 별도 단
후보로 존속한다.

| # | 지적 (검수 실측 [판독:70-73]) | 판정자 확인 | 처분 |
|---|---|---|---|
| ① | `report_outside_census` 가 `realpath(census.parent)` 를 써서 **심링크 이중 오조작** 시 실패 report 가 봉인 안에 쓰인다(검수 합성 실증) | 코드 실재 `analyze_a210_z2a_probe.py:433-443`(realpath `:434-435`) [V12]. **자매 판독기 계승 잔여**(같은 계급이 `analyze_l6c_redispatch.py` 에도 있음 — 검수 판정)이고, 이 단의 등록 게이트 명령은 전건 명시 `--report` 봉인 밖이라 그 경로를 지나지 않는다 [V5·V6 — 판정자 재현도 동일 형식] | 대장 기재 + **차기 판독기 수리 단 후보**(자매와 함께 한 계약으로) — 지금 수리 금지(§8-7·8) |
| ② | z2a R-II 모드 **기본 report 경로 무가드** — `--report` 생략 시 `run_root` 안에 기본 기록(`:675`) | **기존치** 확인: 봉인 안 `a210_z2a_probe_report.json`(M14)이 원 런처 시절 그 경로의 산출 [V8]. Codex 보고 §4 가 «봉인 재판독 시 보고서 외부 지정 규율이 계속 필요»를 명시 재진술 [V10] — Codex 자진 개시 | 대장 기재 + 규율 명기(봉인 재판독은 **항상** 명시 `--report` 봉인 밖 — 이 단 게이트는 전건 준수) — 지금 수리 금지 |
| ③ | CENSUS-SELFCHECK 실패 report 가 **`ZI-B` 라벨 재사용**(`:656`) — `mode` 필드로 구별되나 하류 오독 소지 | 실재 확인 [V12]. PASS 쪽도 `ZI-A` 재사용임을 판정자 추가 관측(`za_selfcheck.json` `branch=ZI-A`·`mode=CENSUS-SELFCHECK` [V5]) — Z-2A 분기 라벨의 타 모드 차용은 양방향이다. 하류는 `branch` 전에 `mode` 를 읽어야 한다 | 대장 기재 — 라벨·분기 의미론 정리는 §8-5 의 «ZI-B 확장»과 같은 계열의 **별도 단 후보** |

**재판정에서의 처분 재검토 — 세 건 전부 초판 처분 유지.** 추가 집행이 각 처분의 전제를
바꾸지 않았고 오히려 재확인했다: ① 추가 집행 게이트도 전건 명시 `--report` 봉인 밖
(`ga4*.json`·`pr0_*.json` 전부 `/gpfs/…/zo_artifacts/` [V19]) — 그 경로를 지나지 않았다.
② 규율(«봉인 재판독은 항상 명시 `--report` 봉인 밖») 준수가 추가 집행에서도 지속됐다.
③ 라벨 재사용의 새 실례 2건 추가 관측(`ga4b.json` FAIL 이 `branch=ZI-B`·`mode=CENSUS-SELFCHECK`
[V14]; `ga4.json`/`ga4_clean.json` PASS 가 `branch=ZI-A` [V14]) — 처분(대장 기재·별도 단
후보) 불변, 실례 수만 는다.

---

## 8. 집행 이탈 관측 (조용한 기재 — 게이트 판정 산입은 §2·§3 에서 이미 했다)

1. **산출물 거처 이탈 — 지속**: 계약 §4·§5·§10 이 등록한 `validation/instrument_repair/` 는
   재판정 시점에도 **미신설** [V19] — 초판분·추가 집행분 전부의 실거처는
   `/gpfs/kjhan/lumina/zo_artifacts/` [V9·V19]. 실질 요건(봉인 밖·/gpfs·비-/tmp)은 충족하나
   등록 경로와 다르다. Codex 합성 검증 로그의 노드-로컬 /tmp worktree 거처 문제도 지속 [V10]
   — 영속 거처 이관 필요(운전석 몫).
2. **G-A1·G-B1 산출 미보존 — 지속**: selftest 산출이 파일로 남지 않았다 — 판정자 재현·재확인
   [V4·V13] 으로 보강.
3. **장부 §12 — 기입 완료, 단 신규 관측 3건** [V18]: 필수 4항목·분장 장부가 채워졌고 운전석
   집행 누락 2건(G-A4 / G-B2·G-B3)이 위반 칸에 자기 기재됐다(초판 관측 3 해소). 신규:
   **(a)** G-0 후반부의 **기재 지연**(§12 항목 1 을 초판 판정 후 기입)이 해당 행(발주
   프리플라이트 집행)의 위반 칸에 «없음» 으로 남아 있다 — 위반 칸에 «기재 지연» 으로 기입
   권고. ★**집행-시점 위반(개정15 «발주 전 rc=0»)은 미입증이다**(재판정 1판의 «판정 후 소급
   집행» 단정은 오류였고 본 판본이 **철회**한다 — 감리-A 차단 사유 1건): 집행 시각의 직접
   증거는 없고(프리플라이트는 파일을 남기지 않으며 발주 자료 3건 grep 0 [V10]), 기록 rc=0 은
   구현 착지(14:46:48) 이전 상태에서만 산출 가능하므로 «판정 후 소급» 은 **배제**된다
   [V15 + 감리-A 독립 재현]. 발주서 mtime 08:39:04 → worktree 생성 **08:39:28** 의 24초 공백이
   발주 전 집행 1회와 정합하는 **정황**은 있으나 증명이 아니고, 발주(08:39:28~08:47:31) 후
   14:46:48 이전 집행이었다면 「발주 전」 의무는 여전히 이탈이다 — **증거 불충분으로 판정한다.** **(b)** 항목 1 기재가 **재현 경로를
   남기지 않았다**(발주서 사본의 소재·`--root` 대상 트리 — 라이브 상태에서는 FAIL 이 나므로
   [V15] 이 정보 없이는 rc=0 이 재현 불능; 판정자가 발주시점 재구성으로 보충했다 [V15]).
   **(c)** 분장 장부 감리-A 발주 행의 «규약상 담당» 칸 문면이 변경됐다(«CLAUDE.md 상시 승인
   ④» 삭제 — user 매건-승인 지시 2026-08-28 의 반영; 지시 실재는 §12 문면 자체가 기록하나,
   규약상-담당 칸은 집행 기록이 아니라 등록 문면이라 개정 형식(OLD/NEW) 밖의 변경이다).
4. **러너 rc 포집 결함**: `run_pr0.sh`·`run_gb2.sh` 의 `RC=$?` 는 파이프 말단 `tail` 의 rc 를
   포집한다(pipefail 부재) [V19 직독] — G-B2 «rc=0» 주장은 **러너 문면으로는 무근거**다.
   주장 자체는 참임을 판정자 직접 재실측으로 확립했다(G-B2 rc=0 [V16]·G-B3 rc=4 [V15]).
   «로그 숫자 자체 검증» 계급의 잣대 함정 — 대장 기재 후보.
5. **G-B2 집행 형식 이탈**: 등록 = `check_pr0_unsealed` 반쪽·봉인 무접촉(§5 행). 실집행 =
   전체 판독기 상위집합(`--run-root <RII>` — 봉인 read-only 접촉, `PR-0.sealed_marker_inputs_
   opened=true` [V19]). 등록 판정식 두 절은 상위집합 안에서 성립하고(`:1250` 내부 호출 [V16])
   판정자가 등록 형식 단독으로도 재확인했다 [V16] — 게이트 실질 무영향, 형식 기록만. 부수
   산출(전체 판독의 `branch=C-M-S ff=0.10734463276836158` 등)은 L6C-R 단의 재판독이며 본 단
   판정에 불산입(타 단 재판정 아님 — §9).
6. **스트레이 러너**: `run_gb2.sh`(l6c `--census-selfcheck` 형식, report 목적지 `gb2.json`)가
   산출물 없이 잔존한다(`gb2.json` 부재 [V19]) — 집행 여부 문면 불명. 관측만.
7. **판독 문서 거처 — 지속**: `INSTR_READOUT.md`(105행) 노드-로컬 /tmp — 영속 거처 이관 필요
   (운전석 몫; 본 판정문이 핵심 수치를 전건 인용·재검증했으므로 증거 사슬은 본 문서로도
   보존된다).
8. ★**운전석 이의 증거 표의 잣대 오류 — 확정**(감리-A 비차단 지적 ③): 운전석이 「소급 집행」
   기재에 이의를 내며 제출한 시각 표에서 「worktree `wt_instr` 생성 = **08:48:51**」은 **오측**
   이다. 08:48:51 은 그 디렉토리의 **최종 변경 mtime**(Codex 가동 중 최상위 항목이 바뀐 값)
   이고, 실제 생성은 `.git/worktrees/wt_instr/{gitdir,HEAD,commondir}` mtime 과 디렉토리 birth
   가 일치하는 **08:39:28** 이다 [감리-A 실측 + V21 판정자 재확인]. **이의를 내면서 낸 증거
   표 자체에 잣대 오류가 있었다** — 대장 「잣대부터 감사」 계급의 재발이다. 정정된 시각은
   운전석 주장에 오히려 유리하지만(공백이 9분 47초에서 24초로 줄어 발주 전 집행과 더 정합),
   ★**그것이 증명이 되지는 않는다**(§8-3(a)). 이 항목은 판정문이 아니라 **운전석 기록** 대상
   이므로 장부(계약 §12)에도 기입한다.
9. ★**감리-B 요청 대리 집행 3건 — 완료**(감리-B 차단 사유 #2·#3·#6, 「운전석 대리 실행이
   필요한 항목」): 감리-B 가 이번 회차에 `syntax` 에서 돌았고 Slurm 소켓 차단·읽기 전용
   샌드박스(임시 디렉토리 생성 불가)로 라이브 G-B2·G-B3 와 selftest 2본을 직접 재현하지 못
   했다. 운전석이 대리 집행했고 **3건 전건이 계약 등록 기대와 일치**한다 [V22·V23·V24]:
   **(a) G-B2 라이브(syntax)** — `PASS branch=C-M-S`, `sacct_records=['324557|LUMINA_Z2A_II|`
   `FAILED|70:0|syntax']` 등록 문자열과 일자일자 일치, 기존 `pr0_syntax.json` 과 **byte-동일**
   [V22]. rc 는 **`pipefail`·무파이프 러너**(`run_gb2c.sh`)로 다시 받아 **rc=0** — §8-4 의
   결함 러너(`run_pr0.sh`, 파이프 말단 `tail`)에 얹지 않았고, 그 재집행 산출도 원본과
   byte-동일이다 [V22]. **(b) G-B3 라이브(grammar-debug)** — `FAIL branch=X-P RII_NOT_TERMINAL`
   `record_count=0` **rc=4**, 계약 §6 의 **1차 기대 그대로**이고 IR-F2(PASS) 미발화 [V23].
   ★이 rc 는 `pipefail` 을 켠 러너에서 포집했다 — §8-4 의 결함 러너 계급을 반복하지 않았다.
   **(c) selftest 2본** — z2a `rc=0`(NC-I0~I8) · l6c `rc=0`(NC-R0~R18) [V24].
   ⟹ 감리-B 의 세 「재현 불능(환경)」은 **판정 하중 변경 없이** 해소된다 — 대리 집행이 감리
   자신의 재현을 대체하지는 못하므로, 재감리 시 감리-B 는 이 대리 산출을 **자기 실측으로
   재확인**하거나 환경 제약을 명시해야 한다.

---

## 9. 하지 않은 것 (계약 §8 전수 준수)

- **타 단 재판정 0**: Z-2A ZI-A·E-I1 문면 FAIL·L6C-R C-M-S·F1b·L6 분기 B·A207·Z-O·BITLOCK
  전부 불변 — 본 문서는 인용만 했다. G-B2 상위집합 집행이 부수적으로 L6C-R 문면(C-M-S)을
  재산출했으나 본 단 판정에 불산입(§8-5).
- `src/` 접촉 0(별건 판정런 `325913 RUNNING`·`325914 PENDING` 재판정 시점 재확인 [V13] —
  이중 금지 준수) · **봉인 쓰기 0**(초판·재판정 전 접근 read-only [V5-V8·V14-V17]; 판정자의
  변조 생성 0 — G-A4 재현은 운전석이 만든 사본 3본의 **판독**이고, 사본 무결은 cmp 로
  판정자가 독립 확증했다 [V14]) · 물리량·판정식·분기 의미론 접촉 0 · 오라클(CMFGEN 자체런)
  정량 인용 0 · clamp/floor/cap/jitter/repair 0 · **커밋 0**(운전석 몫).
- **판정자가 새로 만든 것(전건 봉인·src·계약 본문 밖 — 공개 기재)**:
  `/gpfs/kjhan/lumina/instr_verdict_repro_rev1/`(재현 산출·발주서 사본·`remote_batch.sh`) ·
  git worktree `tree_ddf6701`(발주시점 재구성 — **관리 메타데이터만**, 추적 파일·커밋 0;
  제거는 폐합 정리 때 운전석 몫: `git worktree remove`) · 본 판정문 개정(초판 `.rev1` 보존
  [V18]) · 계약 §12 분장 장부 **③ 판정 행의 재판정 갱신**(개정16 ① — 자기 단계 칸만).
- 수리·튜닝 제안 0 — §4-1·§7·§10 은 처분(대장 기재)과 별도 단 후보 지목까지다.
- 감리 대행·감리 상태 진술 0 — 감리 상태는 장부(계약 §12)에서 읽어라(§0 머리).

---

## 10. 남은 미결과 다음 표적

★**운전석 누락의 계급 판정** (재판정 의뢰 항목): **반복 가능한 계급이다 — 이 단 하나에서
집행 누락 2건 + 기재 누락 1건**(★계급이 다르다): **집행 누락** ① G-A4 누락 ② G-B2·G-B3
누락(«게이트 전부 통과» 1차 보고 시점에 계약 9행 중 3행 미집행 — FAIL 0 관측을 전건 충족으로
센 fail-open 독법) / **기재 누락** ③ G-0 후반부의 §12 항목 1 **기재 지연**(집행 자체의 누락은
미입증 — §8-3(a)). 아래 기계화 논지는 **집행 누락 2건으로도 그대로 존속**한다 — 장부-대-게이트
대조 게이트는 기재 누락 계급도 함께 잡는다. **막는 기계는 현재 0 이다**: 사전등록 프리플라이트는
계약을, 발주 프리플라이트는 발주서를, 봉인 지문은 런 산출물을 검사하지만, **«계약 §5 게이트
표의 전 행 ↔ 집행 산출물» 대조를 검사하는 게이트는 없다** — 이번 검출은 판정 단계(사람
컨텍스트)의 전수 대조가 수행했고, 초판·재판정 두 번 다 그랬다. 처분 = 후속 단 후보로 기재
(아래 표 1 — 도구 설계·수리 제안은 이 문서의 몫이 아니다, 계약 §8-7).

| # | 항목 | 주인 |
|---|---|---|
| 1 | ★**게이트 전수 집행 대조의 기계화** — «계약 게이트 표 전 행 대 집행 산출» 대조 게이트의 부재(위 계급 판정). 설계·범위는 그 단 몫 | 별도 단 후보 (user 결정) |
| 2 | 감리 A·B — 상태·착수는 장부(계약 §12)가 정본 (⚠user 지시 2026-08-28: Fable 의뢰 매건 승인 — §12 문면) | user 승인 → 운전석 발주 |
| 3 | E-I1 해소·R-III/R-IV 수용 실물 — 별건 진행 관측(`325913 RUNNING`/`325914 PENDING` [V13]) | 그 단 (GPU — 이 단 밖) |
| 4 | 대장 기재 후보 — 신규 5건: §7 표 3건 + **G-A4 음성대조 계급 한계**(주입 유효성 등록 부재 — 십진 자면 변조가 bit 항등 널 주입일 수 있다, §4-1-3) + **러너 rc 포집 함정**(파이프 말단 rc — §8-4). 기존 존속: F-2(σ 스코프)·F-3(proxy≢capture)·Z-2A 대장 2번(fail-open 벡터)·잣대 함정(`sacct FAILED 만 보면 오독`, 대장 `:371`) | 운전석 (대장 기재) |
| 5 | z2a «어떤 오류든 ZI-B» 확장 + 라벨 의미론(§7-③) 정리 | 별도 단 후보 (계약 §8-5) |
| 6 | 반상태 처분 (DET-SUPERLEVEL-B 사전등록 `c518c18` — 별건) | user 결정 대기 |
| 7 | 집행 위생 잔여(§8): 산출물 등록 거처 이관·`INSTR_READOUT` 영속화·§12 (a)(b) 보완·재현 인프라(`instr_verdict_repro_rev1`·`tree_ddf6701`) 정리 | 운전석 (폐합 정리) |

---

## 한 줄 결론

**분기 = IR-A (수리 성립 — 재판정; 초판 IR-X 는 `.rev1`)** · **근거** = 계약 §5 게이트 9행
**전건 충족**을 계약 문면 재적용으로 확정(§2·§3) — 추가 집행 산출물 5본 판정자 byte-동일
재현·rc 직접 재실측, G-B3 은 PASS 가 아니라 1차 기대 그대로의 이름 있는
FAIL(`RII_NOT_TERMINAL record_count=0`; 수리 전 같은 자리는 침묵 PASS) · **결함 A** = 계약
§1(i) 전건 시연(봉인 자기입력 PASS·실물 사본 변조 FAIL/대조군 PASS·역주입 시 selftest 사망;
G-A4 1차 시도는 **시험 무효** — bit-항등 널 주입, 표본 불산입 §4-1; Z-2A ZI-A 판정 불변) ·
**결함 B** = 삼중 provenance·argv 검증의 합성 채널 + **라이브 양·음 쌍**까지 시연 — 단
**«정본 증거의 취득은 규율» 한계와 G-B3 의 비상설성은 그대로**이고
`SACCT_PROVENANCE_MISMATCH` 팔의 라이브 발화는 원리상 없으며(§6-4(c)), ★재판정 1판의
«산출물이 집행 클러스터를 자기증명» 주장은 **철회**한다(§6-4(b) — 실행-호스트 필드 부재 +
`pr0_grammar.json` 의 타 클러스터 byte-동일 재생산) · **잔여**
= 집행 이탈 관측 7건(§8 — 전건 게이트 FAIL 아님)·대장 기재 후보 5건·집행 누락 계급의 기계
부재(§10), 그리고 감리 A·B 의 상태는 **장부(계약 §12)가 정본**이다 — 본 판정문은 폐합을
선언하지 않는다.

---

## 부록 A — 판정자 재현 명령 원장 (전건 2026-08-28 실행 — syntax 로그인, 판독·조회 비연산 계급)

경로 축약: `REPO=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn` ·
`ZO=/gpfs/kjhan/lumina/a210_zero_opacity_z1r_a100x2_k36/zo_20260826T050021Z_census_ion1` ·
`RII=/gpfs/kjhan/lumina/a210_z2a_diag_isolation_a100x2_k36/rii_20260826T134337Z_ion1` ·
`ART=/gpfs/kjhan/lumina/zo_artifacts` · `SCR=<봉인 밖 스크래치>` (판정자는 세션 스크래치 사용;
감리-B 는 임의의 봉인 밖 쓰기 가능 경로로 대체 — ⚠`/tmp` 는 노드-로컬) ·
`RP=/gpfs/kjhan/lumina/instr_verdict_repro_rev1` (재판정 V13~V19 의 산출 거처 — /gpfs 라
노드 공유; 재현 시 자기 경로로 대체).

⚠**sacct 클러스터 귀속 경고 (감리-B 필독)**: V1 은 **syntax 에서만 정본**이다. grammar 계열
노드에서 같은 명령은 무관 레코드(`UPP_QE68_SCF`) 또는 `-M syntax` 시 빈 출력 rc=0 을 낸다 —
**그것이 결함 B 그 자체다.** 재현 전 `scontrol show config | grep ClusterName` 으로 노드
귀속을 확인하라. Codex 샌드박스는 Slurm DB 소켓 접근 불가 전례(Z-2A 판정문 §12-1 NOTE) —
그 경우 V1 은 재현 불능으로 보고하는 것이 옳다.

```bash
# V1 — 클러스터 귀속·정본 레코드·별건 판정런 (syntax)
hostname                                            # -> syntax
scontrol show config | grep ClusterName             # -> ClusterName = syntax
sacct -j 324557 -X -n -P -o JobIDRaw,JobName,State,ExitCode,Cluster
#  -> 324557|LUMINA_Z2A_II|FAILED|70:0|syntax   (rc=0)
sacct -j 325913,325914 -X -n -P -o JobIDRaw,JobName,State,ExitCode
#  -> 325913|LUMINA_Z2A_III|RUNNING|0:0 · 325914|LUMINA_Z2A_IV|PENDING|0:0 (관측 시점)

# V2 — 커밋·변경집합·문서 불변·worktree clean
cd $REPO
git show --stat --oneline ddf6701 | head -8         # -> RUNG_INSTRUMENT_REPAIR ... 465행 수록
git show --stat --oneline d597258 | head -6         # -> 정확 2파일, 350 insertions(+), 37 deletions(-)
git diff ddf6701..HEAD -- docs/RUNG_INSTRUMENT_REPAIR_2026-08-28.md   # -> 공출력 (diff 0)
git status --porcelain scripts/analyze_a210_z2a_probe.py scripts/analyze_l6c_redispatch.py \
  docs/RUNG_INSTRUMENT_REPAIR_2026-08-28.md         # -> 공출력 (clean)

# V3 — byte-사슬 (검수 패치 ≡ 커밋 ≡ 집행본 ≡ HEAD)
git show d597258 --format= --patch > $SCR/commit_patch.diff
cmp /tmp/claude-10396/INSTR_PATCH.diff $SCR/commit_patch.diff          # -> 무차 (byte-same)
cmp $ART/probe_committed.py $ART/probe_repaired.py                     # -> 무차
cmp $ART/probe_committed.py scripts/analyze_a210_z2a_probe.py          # -> 무차

# V4 — G-A1·G-B1 재현 + PF-1
python3 -B scripts/analyze_a210_z2a_probe.py --selftest   # 말미: ... PASS controls=NC-I0,...,NC-I8 failed=0 ; rc=0
python3 -B scripts/analyze_l6c_redispatch.py --selftest   # 말미: ... PASS controls=NC-R0,...,NC-R18 failed=0 synthetic_fixtures=1 sealed_access=0 ; rc=0
#  NC-R17/R18 발화 행 (실측 문면):
#   NC-R17 inject=JOB_NAME_MISMATCH status=FAIL reason=SACCT_PROVENANCE_MISMATCH branch=X-P sealed_access=0
#   NC-R17 inject=CLUSTER_MISMATCH  status=FAIL reason=SACCT_PROVENANCE_MISMATCH branch=X-P sealed_access=0
#   NC-R18 channel=PATH_SHIM inject=HISTORICAL_WRONG_RECORD status=FAIL reason=SACCT_PROVENANCE_MISMATCH branch=X-P sealed_access=0
#   NC-R18 channel=PATH_SHIM measure=ARGV status=PASS ... cluster_arg=-M,syntax fields=JobIDRaw,JobName,State,ExitCode,Cluster
#   NC-R18 channel=PATH_SHIM remove=RESTORE_CANONICAL_RECORD status=PASS ... record=324557|LUMINA_Z2A_II|FAILED|70:0|syntax
#   NC-R18 channel=PATH_SHIM inject=EMPTY_RC0 status=FAIL reason=RII_NOT_TERMINAL branch=X-P sealed_access=0
grep -rln A210_INSTRUMENT_REPAIR scripts src        # -> 정확 2파일 (scripts/의 두 판독기)

# V5 — G-A2 재현 (봉인 read-only, report 봉인 밖)
python3 -B scripts/analyze_a210_z2a_probe.py --census-selfcheck $ZO/stderr.log \
  --report $SCR/za_selfcheck_repro.json
#  -> A210_Z2A_PROBE_CENSUS_SELFCHECK_RESULT status=PASS anchor_line=669992 input_access=READ_ONLY streaming=1 ; rc=0
diff $SCR/za_selfcheck_repro.json $ART/za_selfcheck.json               # -> 무차 (byte-동일)

# V6 — G-A3·G-A5 재현 (봉인 read-only)
python3 -B scripts/analyze_a210_z2a_probe.py --mode R-II \
  --stderr $ZO/stderr.log --census-stderr $ZO/stderr.log --report $SCR/selfinput_repro.json
#  -> status=FAIL branch=ZI-B ; rc=4 ; JSON: blocking=['SPROBE_COUNTER_MISSING'] f_bit=1.0
diff $SCR/selfinput_repro.json $ART/selfinput_after.json               # -> 무차
python3 -B scripts/analyze_a210_z2a_probe.py --mode R-II \
  --stderr $RII/stderr.log --census-stderr $ZO/stderr.log --report $SCR/ga5_repro.json
#  -> status=PASS branch=ZI-A ; rc=0 ; JSON: f_bit=1.0 blocking=[]
diff $SCR/ga5_repro.json $ART/ga5_retro.json                           # -> 무차

# V7 — 봉인 지문 spot
wc -l $RII/RII_RUN_SEAL.sha $ZO/ZO_RUN_SEAL.sha     # -> 135 · 136
grep stderr.log $RII/RII_RUN_SEAL.sha ; (cd $RII && sha256sum stderr.log)   # -> 31b4a23a... 양쪽 일치
grep stderr.log $ZO/ZO_RUN_SEAL.sha  ; (cd $ZO  && sha256sum stderr.log)    # -> 5b3fbb20... 양쪽 일치
ls -la $RII/RII_RUN_SEAL.*                          # -> mtime 2026-08-27 17:32 · r--r-----

# V8 — 앵커 4행·hex·M14 (전건 read-only)
sed -n '1242p;226118p' $ZO/stderr.log ; sed -n '1244p;226122p' $RII/stderr.log
#  -> iter0: n_upper=0.0004437688873013011 emission_per_sr=2.7310010625545125e-16
#     iter1: n_upper=0.00045264235457262295 emission_per_sr=2.7856093265405119e-16
#     (census·R-II 쌍 bit-동일 · A_ul=0.52225560000000004 · tau_validity=2)
python3 -c "print(float.hex(0.0004437688873013011), float.hex(2.7310010625545125e-16),
float.hex(0.00045264235457262295), float.hex(2.7856093265405119e-16), float.hex(0.52225560000000004))"
#  -> 0x1.d1534dba20c26p-12 0x1.3adcec9caadc0p-52 0x1.daa141c1395e6p-12 0x1.4128acfd24836p-52 0x1.0b6516044e436p-1
python3 -c "import json; d=json.load(open('$RII/a210_z2a_probe_report.json'));
print(d['status'], d['branch'], d['blocking_reasons'], d.get('f_bit'))"
#  -> FAIL ZI-B ['ANCHOR_FIELD_MISMATCH'] 1.0   (M14 — 수리 전 봉인 기준선)

# V9 — 게이트 산출물 소재·미집행 게이트 산출 부재
ls -la $ART/            # -> za_selfcheck.json(11:57) selfinput_after.json(11:56) ga5_retro.json(14:52)
                        #    probe_repaired.py(11:56) probe_committed.py(14:52) — G-A4/G-B2/G-B3 산출 0건
ls $REPO/validation/instrument_repair/ 2>&1        # -> No such file or directory

# V10 — 발주·구현 보고 문면
sed -n '86,102p' /tmp/claude-10396/CODEX_INSTR_REPORT.md
#  -> "live G-B2/G-B3 및 봉인 G-A2~A5를 실행하지 않은 현재 결과로는 IR-A를 주장하지 않습니다" 등
grep -c "check_dispatch_preflight" /tmp/claude-10396/codex_instr_order.md \
  /tmp/claude-10396/CODEX_INSTR_REPORT.md /tmp/claude-10396/INSTR_READOUT.md   # -> 0 · 0 · 1
#  ⚠ 셋째 기대치 정정(감리-B #16): 초판 관측 시점에는 0 이었으나, 이후 판독문(105행)에 추가
#     G-0 집행 기록이 §93 으로 들어가 현재는 **1** 이다. 초판의 «발주 자료에 rc 기록 부재»
#     논지는 앞 둘(발주서·Codex 보고 = 0)이 지탱하며 변하지 않는다.

# V11 — 역주입 (판정자 음성대조 — 스크래치의 판독기 사본만 변조)
cp scripts/analyze_a210_z2a_probe.py $SCR/probe_reverted.py
sed -i 's/0x1.daa141c1395e6p-12/0x1.d1534dba20c26p-12/; s/0x1.4128acfd24836p-52/0x1.3adcec9caadc0p-52/' \
  $SCR/probe_reverted.py
python3 -B $SCR/probe_reverted.py --selftest        # -> ... FAIL ... failed=12 ; rc=1
#  (관측: 역주입 상태에서 NC-I8 inject 팔이 status=PASS 로 반전 — 변조 census 가 통과한다)

# V12 — 소스 인용 행 실확인 (HEAD = d597258)
grep -n "ANCHOR_BITS_BY_ITERATION = (\|def anchor_matches\|ANCHOR_ITERATION_INVALID\|def census_selfcheck\|def report_outside_census\|A210_INSTRUMENT_REPAIR = True" \
  scripts/analyze_a210_z2a_probe.py                 # -> :23 :31 :198 :207 :392 :433 :446
grep -n "anchor_matches(raw_rows, iteration)" scripts/analyze_a210_z2a_probe.py   # -> :303 :475
grep -n "RII_JOB_NAME\|RII_CLUSTER = \|SACCT_PROVENANCE_MISMATCH\|sacct.*-M" scripts/analyze_l6c_redispatch.py | head
#  -> :40 :41 :1139 :1160 :1162 (등)

# ───────────────── 재판정 (V13~V19, 2026-08-28 오후 — syntax 로그인 + grammar-debug 원격) ─────────────────

# V13 — 재판정 시점 상태: selftest 2본·sacct·워킹트리 (syntax)
python3 -B scripts/analyze_a210_z2a_probe.py --selftest    # -> PASS controls=NC-I0..I8 failed=0 ; rc=0
python3 -B scripts/analyze_l6c_redispatch.py --selftest    # -> PASS controls=NC-R0..R18 failed=0 synthetic_fixtures=1 sealed_access=0 ; rc=0
sacct -j 324557 -X -n -P -o JobIDRaw,JobName,State,ExitCode,Cluster   # -> 324557|LUMINA_Z2A_II|FAILED|70:0|syntax ; rc=0
sacct -j 325913,325914 -X -n -P -o JobIDRaw,JobName,State,ExitCode    # -> RUNNING · PENDING (별건 무접촉 확인)
git status --porcelain scripts/analyze_a210_z2a_probe.py scripts/analyze_l6c_redispatch.py   # -> 공출력 (HEAD d597258 그대로)

# V14 — G-A4 재검증: 사본 무결·변조 실체·널 주입 재계산·게이트 3본 재현 (syntax)
cmp $ART/ga4_scratch/clean.log $ZO/stderr.log              # -> 무차 (무변조 사본 ≡ 봉인 byte-동일)
cmp -l $ART/ga4_scratch/tampered.log  $ART/ga4_scratch/clean.log   # -> 정확 1건: offset 76246971, 0x34('4')↔0x35('5')
cmp -l $ART/ga4_scratch/tampered2.log $ART/ga4_scratch/clean.log   # -> 정확 1건: offset 76246960, 0x37('7')↔0x32('2')
for f in $ZO/stderr.log clean.log tampered.log tampered2.log ; do sed -n '226118p' ... | grep -o "n_upper=[0-9.eE+-]*" ; done
#  -> …62295(봉인=clean) · …62294(tampered, 널) · …4735457…(tampered2, 유효)
python3 -c "v='0.00045264235457262295'
for last in '0123456789':
    cand=v[:-1]+last
    print(last, float(cand).hex()==float(v).hex())"
#  -> 마지막 자릿수 3·4·5·6·7 전부 True (같은 bit 0x1.daa141c1395e6p-12) — 1차 변조(5→4)는 널 주입
python3 -c "print(float('0.00045264735457262295').hex())"   # -> 0x1.daa2995a27817p-12 (2차 변조 hex — 판독 기재와 일치)
cat $ART/ga4_scratch/mk.sh $ART/mk_ga4.py   # 직독: 1차=마지막 자릿수, 2차=6번째 유효자리(스크립트 주석의 「8번째」는 오기) + hex 상이 검증 내장
python3 -B scripts/analyze_a210_z2a_probe.py --census-selfcheck $ART/ga4_scratch/tampered2.log --report $RP/ga4_repro_tampered2.json
#  -> FAIL reason=ANCHOR_FIELD_MISMATCH detail=iter=1 ; rc=4
python3 -B scripts/analyze_a210_z2a_probe.py --census-selfcheck $ART/ga4_scratch/clean.log     --report $RP/ga4_repro_clean.json      # -> PASS ; rc=0
python3 -B scripts/analyze_a210_z2a_probe.py --census-selfcheck $ART/ga4_scratch/tampered.log  --report $RP/ga4_repro_tampered.json   # -> PASS ; rc=0 (널 주입 사본)
cmp $RP/ga4_repro_tampered2.json $ART/ga4b.json ; cmp $RP/ga4_repro_clean.json $ART/ga4_clean.json ; cmp $RP/ga4_repro_tampered.json $ART/ga4.json
#  -> 3본 전건 무차 (byte-동일)

# V15 — grammar-debug 원격 배치 (nested ssh: ssh grammar "ssh grammar-debug 'bash $RP/remote_batch.sh'")
#   스크립트 원문 = $RP/remote_batch.sh (보존). 실측:
#   hostname -> grammar-debug · ClusterName = grammar
#   sacct -j 324557 -X -n -P -o JobIDRaw,JobName,State,ExitCode,Cluster -> 324557|UPP_QE68_SCF|COMPLETED|0:0|grammar ; rc=0
#   sacct -M syntax  (같은 열)                                          -> 빈 출력 ; rc=0   (교차 조회 불능 = 결함 B 그 자체)
#   G-B3 재현: python3 -B scripts/analyze_l6c_redispatch.py --run-root $RII --census-root $ZO --deck $ZO/input/model \
#     --report $RP/pr0_grammar_repro.json --expected-head d5972584f2e7cbe94276b76d7f4859f8f0eae6cb --repo-root $REPO
#     -> A210_L6C_REDISPATCH_FAIL branch=X-P reason=RII_NOT_TERMINAL detail=record_count=0 ; rc=4
#   cmp $RP/pr0_grammar_repro.json $ART/pr0_grammar.json -> 무차 (byte-동일)
#   발주 프리플라이트 — 발주시점 재구성: python3 $REPO/scripts/check_dispatch_preflight.py $RP/codex_instr_order.md \
#     $RP/tree_ddf6701/docs/RUNG_INSTRUMENT_REPAIR_2026-08-28.md --root $RP/tree_ddf6701
#     -> PASS no-restatement tokens_checked=5 · PASS frontpage-sealed · PASS contract-preflight rc=0 · PASS dispatch-preflight ; rc=0
#   같은 명령을 라이브 repo 로: --root $REPO -> PASS no-restatement · FAIL contract-worktree-drift ; rc=1
#   (재구성 재료: cp -p /tmp/claude-10396/codex_instr_order.md $RP/  — mtime 08:39 보존(발주 시각);
#    git worktree add --detach $RP/tree_ddf6701 ddf6701 — clean 확인)

# V16 — G-B2 재현 (syntax): 전체 형식 byte-재현 + 등록 형식 단독
python3 -B scripts/analyze_l6c_redispatch.py --run-root $RII --census-root $ZO --deck $ZO/input/model \
  --report $RP/pr0_syntax_repro.json --expected-head d5972584f2e7cbe94276b76d7f4859f8f0eae6cb --repo-root $REPO
#  -> A210_L6C_REDISPATCH_RESULT status=PASS branch=C-M-S ff=0.10734463276836158 sigma=SIGNAL f_mapped=1.0 ; rc=0 (18.9s)
cmp $RP/pr0_syntax_repro.json $ART/pr0_syntax.json         # -> 무차 (byte-동일)
python3 - <<'EOF'   # 등록 형식: check_pr0_unsealed 단독 (docs+sacct 만 — 봉인 무접촉)
import sys; from pathlib import Path; import importlib.util as iu
spec=iu.spec_from_file_location('l6cmod','scripts/analyze_l6c_redispatch.py'); m=iu.module_from_spec(spec)
sys.modules['l6cmod']=m; spec.loader.exec_module(m)
r=m.check_pr0_unsealed(Path('docs')); print(r['status'], r['sacct_records'])
EOF
#  -> PASS ['324557|LUMINA_Z2A_II|FAILED|70:0|syntax'] ; rc=0
grep -n "check_pr0_unsealed(" scripts/analyze_l6c_redispatch.py | head -2   # -> :1123(정의) :1250(check_pr0 내부 호출 — 상위집합 포함 증명)

# V17 — 봉인 spot 재판 (전 재현 이후)
wc -l $RII/RII_RUN_SEAL.sha $ZO/ZO_RUN_SEAL.sha            # -> 135 · 136
grep -m1 " stderr.log$" $RII/RII_RUN_SEAL.sha ; (cd $RII && sha256sum stderr.log)   # -> 31b4a23a… 양쪽 일치
grep -m1 " stderr.log$" $ZO/ZO_RUN_SEAL.sha  ; (cd $ZO  && sha256sum stderr.log)    # -> 5b3fbb20… 양쪽 일치
stat -c "%y %a" $RII/RII_RUN_SEAL.sha $ZO/ZO_RUN_SEAL.sha  # -> 08-27 17:32 / 08-26 18:16 · 모드 440 (전 게이트 이전 그대로)

# V18 — 판본 보존·계약 §12 diff 실사
cp -p docs/VERDICT_INSTRUMENT_REPAIR_2026-08-28.md docs/VERDICT_INSTRUMENT_REPAIR_2026-08-28.md.rev1   # (개정 전 실행)
cmp docs/VERDICT_INSTRUMENT_REPAIR_2026-08-28.md docs/VERDICT_INSTRUMENT_REPAIR_2026-08-28.md.rev1     # -> 무차 (보존 시점)
git diff --stat docs/RUNG_INSTRUMENT_REPAIR_2026-08-28.md   # -> 1 file, +11/−11 (전건 §12 — 행 단위 실사, §1 표 계약 행)
diff docs/VERDICT_INSTRUMENT_REPAIR_2026-08-28.md.rev1 docs/VERDICT_INSTRUMENT_REPAIR_2026-08-28.md    # (개정 후 행 단위 대조 — 의도 외 소실 0)

# V19 — 산출물 소재·러너 직독
ls -la $ART/            # -> ga4.json(15:27) ga4_clean.json(15:28) ga4b.json(15:29) pr0_syntax.json(15:26)
                        #    pr0_grammar.json(15:27) ga4_scratch/{clean,tampered,tampered2}.log mk_ga4.py run_pr0.sh run_gb2.sh …
                        #    gb2.json 부재 (run_gb2.sh 스트레이 — §8-6)
ls $REPO/validation/instrument_repair/ 2>&1                 # -> No such file or directory (§8-1 지속)
cat $ART/run_pr0.sh $ART/run_gb2.sh    # 직독: `… 2>&1 | tail -N ; echo "RC=$?"` — rc 포집이 tail 의 rc (§8-4)
python3 -c "import json; d=json.load(open('$ART/pr0_syntax.json'));
p=d['PR-0']; print(p['status'], p['sacct_records'], p['sealed_marker_inputs_opened'])"
#  -> PASS ['324557|LUMINA_Z2A_II|FAILED|70:0|syntax'] True  (G-B2 실집행의 봉인 read-only 접촉 — §8-5)
python3 -c "import json; d=json.load(open('$ART/pr0_grammar.json')); print(d['PR-0'], d['blocking_reasons'], d['detail'])"
#  -> {'sealed_inputs_opened': False, 'status': 'FAIL'} ['RII_NOT_TERMINAL'] record_count=0  (fail-closed 봉인 무접촉)
```

# ── 재개정 2판 (감리 A·B 반영) ─────────────────────────────────────────────

# V20 — 감리-B #4: PR-0 산출물의 실행-환경 필드 전수 (자기증명 주장의 반증)
python3 -c "
import json
for n in ('pr0_syntax','pr0_grammar'):
    r=json.load(open('$ART/%s.json'%n))
    print(n, json.dumps(r['PR-0'],ensure_ascii=False)[:160])
    print('  실행환경 필드:', [k for k in r['PR-0'] if k.lower() in
          ('hostname','host','cluster','argv','command','node','execution_host')] or '없음')
    print('  sacct 계열:', {k:v for k,v in r['PR-0'].items() if 'sacct' in k})"
#  -> pr0_syntax  : sacct_cluster=syntax · sacct_records=['324557|LUMINA_Z2A_II|FAILED|70:0|syntax']
#                   실행환경 필드: 없음
#  -> pr0_grammar : PR-0 = {"sealed_inputs_opened": false, "status": "FAIL"} · detail=record_count=0
#                   실행환경 필드: 없음
#  ⟹ 두 JSON 어디에도 집행 호스트·클러스터가 없다. 정본 5열의 다섯째 필드 `syntax` 는 **작업이
#     돈 클러스터**이지 판독기를 부른 호스트가 아니다. 감리-B 는 `syntax` 에서 빈 sacct 를 주입해
#     `pr0_grammar.json` 을 byte-동일 재생산했다 — 산출물은 집행 클러스터를 증명하지 못한다.

# V21 — 감리-A 지적 ③: worktree 생성 시각 (운전석 이의 표의 오측 확인)
stat -c '%n mtime=%y' /tmp/claude-10396/wt_instr
stat -c '%n birth=%w' /tmp/claude-10396/wt_instr
stat -c '%n mtime=%y' .git/worktrees/wt_instr/{gitdir,HEAD,commondir}
#  -> mtime = 2026-08-28 08:48:51.575333596  (★운전석이 「생성」으로 오독한 값)
#  -> birth = 2026-08-28 08:39:28.112786772  (실제 생성 — .git/worktrees 3파일 mtime 08:39:28 과 일치)
#  ⟹ 감리-A 실측과 일치. 공백은 9분 47초가 아니라 **24초**다.

# V22 — 감리-B #3 대리: G-B2 라이브 (syntax) ★pipefail·무파이프 러너
sed -e '2i set -o pipefail' -e 's# 2>&1 | tail -1##' $ART/run_pr0.sh > $ART/run_gb2c.sh
bash $ART/run_gb2c.sh syntax3
#  -> A210_L6C_REDISPATCH_RESULT status=PASS branch=C-M-S ff=0.10734463276836158
#     sigma=SIGNAL f_mapped=1.0 ; RC=0
cmp $ART/pr0_syntax.json $ART/pr0_syntax3.json    # -> rc=0 (원본과 byte-동일)
sacct -M syntax -j 324557 -X -n -P -o JobIDRaw,JobName,State,ExitCode,Cluster
#  -> 324557|LUMINA_Z2A_II|FAILED|70:0|syntax  (등록 문자열과 일자일자 일치)

# V23 — 감리-B #2 대리: G-B3 라이브 (grammar-debug) ★pipefail 러너
ssh grammar "ssh grammar-debug 'bash $ART/run_gb3.sh'"
#  -> A210_L6C_REDISPATCH_FAIL branch=X-P reason=RII_NOT_TERMINAL detail=record_count=0
#     RC=4
#  ⟹ 계약 §6 의 **1차 기대 그대로**. IR-F2(PASS) 미발화. 수리 전 같은 자리는 침묵
#     `PASS ['324557|COMPLETED|0:0']` 였다 [계약 C7].

# V24 — 감리-B #6 대리: selftest 2본
python3 -B scripts/analyze_a210_z2a_probe.py   --selftest ; echo rc=$?   # -> rc=0 (NC-I0..I8)
python3 -B scripts/analyze_l6c_redispatch.py   --selftest ; echo rc=$?   # -> rc=0 (NC-R0..R18)

*(V5·V6 의 판독기 실행은 판독(비연산) — 로그인 노드 허용 범위, Z-2A 판정자·계약 C3/C4 선례와
동일 명령. 봉인 접근은 전건 read-only 였고 V7 지문 spot 이 사후 무결을 확인한다. 재판정
V14·V16 의 판독기 실행·V17 재계산도 같은 계급이며 V17 이 전 재현 이후의 무결을 재확인한다.
V15 는 grammar-debug(허용 티어) 원격 집행이다. ⚠V15 의 발주 프리플라이트 재현은 **발주시점
재구성 트리에서만 rc=0** 이다 — 라이브 트리에서는 `contract-worktree-drift`(§12 기입 때문),
구현 착지 트리에서는 §11 시점 술어로 FAIL 이 정상이다. 감리-B 는 `$RP/tree_ddf6701` 이
정리됐다면 `git worktree add --detach <경로> ddf6701` + 발주서 사본으로 재구성하라.)*
