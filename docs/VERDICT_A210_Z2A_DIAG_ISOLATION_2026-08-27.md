# 판정문 — A210-Z2A-DIAG-ISOLATION · R-II (job 324557) — 분기 **ZI-A** · 판독기 라벨 ZI-B 기각(계측 오탐 실증) · 폐합

**요약** — ① R-II 는 봉인 L6C 를 죽인 환경(env delta 0 실측)에서 **모델이 완주**했고
(`model.rc=0` · SUMMARY 2줄 `complete=1` · R7 커밋 · RUN FOOTER), L6C 를 죽인 차단
`INDEPENDENT_SPROBE_UNDEFINED` 은 **0회** 발화했다 — hazard 조건 자체는 **442,764회**
지나갔다(반복당 221,382행 저장·계수). ② 판독기가 낸 `branch=ZI-B` 는 **기각**한다:
계약 §8-2 의 ZI-B 판정식(`f_bit<1.0 ∨ E-I4 토큰 불일치`)은 실측으로 **불성립**이다
(f_bit=1.0 판정자 독립 재확증·E-I4 토큰 전건 일치). FAIL 의 유일 사유
`ANCHOR_FIELD_MISMATCH` 는 **계약이 등록하지 않은 검사**(iter0 앵커 상수의 iter1 적용)의
오탐이며, **봉인 census 자기입력에서도 같은 사유로 발화함을 판정자가 실증**했다 —
판별력 0. ③ 계약 §8-2 판정식 문면 적용 결과 = **ZI-A**. 단 **E-I1 은 문면 FAIL**
(sacct FAILED 70:0 — 소유는 수용 하니스)이고, **런처 수용 PASS 실물이 없으므로
R-III·R-IV 는 현 상태에서 제출 불가**다. ④ **이 판정은 폐합이다** — 규약(개정17)의
폐합 조건이 충족됐다(감리 두 겹의 최종 상태 = §13 스탬프 구조화 표·정본은 장부). ZI-A 처분
잔여(G5·G6)는 §7 실측(수용 실물 부재)으로 이 단이 자기 힘으로 착지시킬 수 없는 항목이라
**폐합 후 큐**(§11)로 명기해 넘긴다 — 확정/미결 분류는 불변이다.

| 항목 | 값 |
|---|---|
| 단 | A210-Z2A-DIAG-ISOLATION (계약 `docs/RUNG_A210_Z2A_DIAG_ISOLATION_2026-08-26.md`, 커밋 `8a48647` + 개정 1 `0abed3b`) |
| 구현 / 집행 기록 | `406712d`(변경집합 4파일 — `git show --stat 406712d` 재확인) / `7fb9bca` |
| job | `324557` **FAILED ExitCode=70:0** · Elapsed 04:34:36 · batch MaxRSS 37165420K(35.4 GiB) · a100 · syn101 · WorkDir=본 저장소 [`sacct -j 324557`] |
| 런루트(봉인, 읽기 전용 열람만) | `/gpfs/kjhan/lumina/a210_z2a_diag_isolation_a100x2_k36/rii_20260826T134337Z_ion1` (이하 `<RII>`) |
| 대조 봉인 | census `<ZO>=/gpfs/kjhan/lumina/a210_zero_opacity_z1r_a100x2_k36/zo_20260826T050021Z_census_ion1` · L6C `<L6C>=/gpfs/kjhan/lumina/det_stage12_fixed_te_a100x2_k36/l6c_20260822T154507Z_cover` (`<RII>/Z2A_STAGE_SEAL.txt` 의 `sealed_base`) |
| 판정자 | Fable fresh 컨텍스트 (분담 개정14 ③). 인용 수치 전건 판정자 직접 실행(부록 §12) — 전부 로그인 노드 허용 계급(grep·diff·cmp·판독기 실행). 봉인 3곳 쓰기 0 (산출은 스크래치에만) |
| **장부** | **장부 = 사전등록 §10** (개정16 ① — 복제하지 않는다). 이 판정문은 판정 칸의 실제만 공급한다: 판정 = Fable fresh, 2026-08-27 · **개정 1**(감리 두 겹 반영 — 감리-B 차단 #14 해소 §6 · NOTE 2 · 감리-A 지적 3; 분기·게이트 판정 불변) = 저자, 2026-08-27 · **개정 2**(최종 — 두 겹 감리의 재판정 통과 후 폐합: §6 논증 승격(실측 교체 — M17·§12-14) · §12-8 명령 정정(감리-B 지적) · §9 대장 2건 추가 · §10/§11/§13 폐합 상태화; 분기·E-I 판정·확정/미결 분류 불변 — 개정 전 판본 보존 `.md.rev1`, §9-7) = 저자, 2026-08-27 |

---

## 1. 실측 원장 (판정자 독립 재현 — 명령 전문은 부록 §12)

| # | 실측 | 값 | 출처 |
|---|---|---|---|
| M1 | job 상태 | FAILED 70:0 · 04:34:36 · MaxRSS 37165420K | `sacct -j 324557` [§12-1] |
| M2 | model rc | **0** | `<RII>/model.rc` |
| M3 | `LINE-SATURATION-ISOLATED` / `-BLOCKED` / `INDEPENDENT_SPROBE_UNDEFINED` / `DET_FLIGHT_FATAL`(stderr) | **0 / 0 / 0 / 0** | `grep -c` `<RII>/stderr.log` [§12-2] |
| M4 | SUMMARY | 정확 **2줄**, 양쪽 `complete=1` (`<RII>/stderr.log:225083`, `:449543`) | [§12-3] |
| M5 | 완주 꼬리 | `R7_MATERIAL_PHASE_COMMITTED lane=DET iter=1 … te_generation=2->3` (`:449756`) · `[PHYSICS-COMPARISON] lane=DET iter=1 status=COMMITTED` (`:449757`) · `RUN_FOOTER.txt` 실존 — **111행 전건 `key=value`**: 환경변수 95(`LUMINA_` 89 · `OMP*` 5 · `CUDA*` 1) + provenance 16키(git_head·binary_sha256·argv 등) [초판 "90 vars" 는 어느 집계와도 불일치(감리-B NOTE#4) — 판정자 재계수, §12-2] | [§12-2] |
| M6 | 런처 die 문면 | `slurm-324557.out` **정확 1행**: `DET_FLIGHT_FATAL Z2A R-II analyzer failed` — die 사유 = **판독기** | [§12-1] |
| M7 | ROW 스트림 | R-II·census 각 **442,764행**(반복당 221,382), 추출 스트림 **byte-동일**(`cmp` 무출력) ⟹ **f_bit=1.0 판정자 독립 재확증** | [§12-4] |
| M8 | SUMMARY 토큰 | 공유 토큰 **전건 자면 동일**(candidate_rows=749644 · selected_rows=1201/992 · total_scaled_emission=0.544556800593765592605 / 0.633250617162941479828 포함), R-II 만 말미 `sprobe_undefined_rows=221382` 추가(양 반복) — §2-1 사양 그대로 | [§12-3] |
| M9 | 선정열 | (rank,line) **2,193쌍**(1201+992) 시퀀스 R-II==census 동일(`cmp` 무출력) | [§12-5] |
| M10 | 앵커 행 line=669992 | iter0(`<RII>/stderr.log:1244`)=**등록 4필드 전건 bit-일치** · iter1(`:226122`)=n_upper `0x1.daa141c1395e6p-12`·emission_per_sr `0x1.4128acfd24836p-52` 로 등록 상수와 상이하되 **census(`:1242`/`:226118`)와 양 반복 bit-동일** | [§12-6] |
| M11 | env delta | `<RII>` ↔ `<L6C>` `input/resolved_lumina.exports` 경로 정규화 후 **diff 공백** — 캡처 2종=1 · DIAG=2 · TARGET_ION=1 · TE_K=10020 동일 | [§12-7] |
| M12 | 바이너리 delta | sha256 상이(신 `39d452e4…` / 구 `b9a30a8…`) · git_head 신=`406712d` / 구=`0e899cb`(각 `input/git_head.txt`) ⟹ **소스 delta 는 커밋 둘**: `e500e29`(Z-O Z-1r 계측) + `406712d`(Z-2A 격리); strings 신판: ROW 표지 1(`e500e29` 산)·ISOLATED 1·`sprobe_undefined_rows` 3·구 센티널 **0** / 구판: 0·0·0·구 센티널 **1** | [§12-8] |
| M13 | L6C 사망 사슬(봉인) | `<L6C>/stderr.log` 총 **1,251행**: `:1245` `BLOCKED reason=INDEPENDENT_SPROBE_UNDEFINED … candidate_rows=66110` → `:1250` `[R7][FATAL] lane=DET iter=0 rc=4` · `model.rc=1` · SUMMARY **0줄** | [§12-9] |
| M14 | 판독기 report | `<RII>/a210_z2a_probe_report.json`: status=FAIL · branch=ZI-B · blocking=`['ANCHOR_FIELD_MISMATCH']`(단 1건) · **f_bit=1.0** · isolated=0 · blocked=0 · clamp/floor/cap/jitter/repair=0 · physical_values_modified=false | [§12-10] |
| M15 | 판독기 재현·자기입력 시연 | §3 참조 | [§12-11] |
| M16 | R1/R2 잔차 (관측) | R-II `stderr.log:12`(iterations=45 residual=9.6662782724980344e-09)·`:423`(52 / 8.1222406993212508e-09)·`:225301`(iter1 R1, 9.2357138113829785e-09) — census `:12`/`:423`/`:225299` 와 **자릿수까지 동일** | [§12-12] |
| M17 | (개정 2) `e500e29` in-vivo 실적 · capture 단일 링크 · 전건 source_defined=0 | census `git_head`=`e500e2995814…` **정확히 그 커밋 단독** · 노드 실측 L6C=**syn102**(job 322449)·census=**syn102**(job 324388)·R-II=syn101(job 324557) · census↔L6C exports diff=**캡처 2종 0↔1 두 행뿐** · 사망 가드 술어 자구 동일(0e899cb `:14111` ↔ e500e29 `:14125`, diff 공백) · hazard 행 **442,764/442,764 전건 `source_defined=0`**(R-II·census 각각 — 반례 0) · L6C 사망 행 = WITNESS `:1244` line=669992 → `:1245` BLOCKED | [§12-14] |

---

## 2. 판독기 라벨 `ZI-B` — **기각** (계약 §8-2 판정식 문면 대조)

계약 §8-2 의 ZI-B 판정식 문면:

> **ZI-B** ⟺ R-II 완주 ∧ ISOLATED 0 ∧ (**f_bit < 1.0 ∨ E-I4 토큰 불일치**)

실측: f_bit = **1.0** (M7 — 판정자 독립 byte-대조), E-I4 토큰 **전건 일치** (M8·M9).
⟹ 괄호 안 선언(disjunction)의 양 항이 모두 거짓이다. **ZI-B 판정식은 이 실측에서
성립하지 않는다.** ZI-B 의 처분("섭동 조사 — 변경집합 또는 결정론의 결함")이 조사할
섭동은 측정상 존재하지 않는다(442,764행 전건 bit-동일).

판독기가 ZI-B 를 낸 것은 계약 판정식의 적용이 아니라 구현의 자체 확장이다:
`scripts/analyze_a210_z2a_probe.py:301` `branch = "ZI-A" if not errors and f_bit >= 1.0
else "ZI-B"` — **어떤 오류든** ZI-B 로 보낸다. 계약 §8-2 는 앵커를 ZI-B 판정식에 넣은 적이
없다(앵커는 E-I3·G4 항목이다). 오류의 실체는 §3 대로 **등록되지 않은 검사의 오탐**이다.

**«오탐이니 통과» 가 아닌 이유** (쟁점 — 계약이 이 처분을 허용하는가): 이 판정은 오탐을
무시하고 통과시키는 것이 아니라, **계약 §8-2 가 등록한 판정식의 각 항을 오탐과 무관한
1차 실측으로 채워서** 분기를 결정한다(§4). 오탐이 실제로 위반시킨 유일한 등록 문면은
E-I1 의 sacct 항이고, 그것은 §5 에서 **FAIL 로 정직 기재**한다. 반면 오탐 자체는 §8-2 의
어느 판정식 항도 위반시키지 못했다 — E-I3 의 앵커 등록값은 iter0 유래이고(§3-c), iter1
에 그 상수를 요구하는 절은 계약 어디에도 없다.

---

## 3. 오탐의 기전 — 운전석 진단의 독립 재확인 + 추가 실증 3건

**(a) 코드 지점 (판정자 직독)** — `scripts/analyze_a210_z2a_probe.py`(커밋 `406712d`,
이 파일의 유일 커밋 [`git log`]): `ANCHOR_LINE=669992`(`:28`) · `ANCHOR_BITS`(`:29-34`) =
**iter0 값 고정** 4필드. `anchor_matches()`(`:188-210`, hex 비교 `:206`)가 반복 루프
(`:255` `for iteration, (actual, reference) in enumerate(…)`) 안의 `:290`
`for raw_rows in (actual_zero, reference_zero):` 에서 **양 반복 × run·census 양쪽**에
적용된다. 운전석 판독문(`/tmp/claude-10396/Z2A_RII_READOUT.md`)의 진단과 일치.

**(b) ★자기입력 실증 (판정자 실행)** — 봉인 census 를 run·census **양쪽에** 넣어도
발화한다 [§12-11]:

```
analyze_a210_z2a_probe.py --mode R-II --stderr <ZO>/stderr.log --census-stderr <ZO>/stderr.log
→ status=FAIL branch=ZI-B blocking=['SPROBE_COUNTER_MISSING','ANCHOR_FIELD_MISMATCH'] f_bit=1.0
```

(`SPROBE_COUNTER_MISSING` 은 정의상 기대되는 부산물 — 봉인 census 는 신설 토큰 이전
산출물이다. 실제 R-II 판독에서는 발생하지 않았다, M14.) **성한 봉인 기준 자신이 FAIL
하는 검사는 성한 런과 섭동 런을 가르지 못한다** — 규약의 음성 대조 의무("게이트는 주입
결함으로 FAIL 을 시연해야 PASS 자격")의 쌍대로, **성한 입력에서 FAIL 하는 검사는 FAIL
의 증거 자격이 없다.** 아울러 실제 호출의 재현(R-II vs census)도 봉인 report 와 동일
결과를 냈다(FAIL·ZI-B·ANCHOR_FIELD_MISMATCH 단독·f_bit=1.0) [§12-11].

**(c) iter1 값은 정당하다 — 등록 상수는 iter0 유래다**: 앵커 행의 iter0→iter1 변화
(n_upper `0x1.d1534dba20c26p-12`→`0x1.daa141c1395e6p-12`, 상대변화 판정자 계산
**+1.999569e-02** [§12-6])는 **선행 폐합 문서가 이미 봉인 census 실측으로 등재한 그
값**이다: `docs/RUNG_BIT_LOCKSTEP_PRODUCER_2026-08-26.md:184`(Z=26 행 — 정확히 이 hex
쌍·+1.999569e-02) · `docs/VERDICT_BIT_LOCKSTEP_PRODUCER_2026-08-27.md:73`. 또
`docs/VERIFICATION_REGISTERS.md:247` — 반복 간 n_upper **bit-동일 0/221,382**(상대변화
중앙값 1.442e-2). ⟹ **어떤 hazard 행을 앵커로 골랐어도 iter0 상수의 iter1 적용은
필패하는 구조**다. E-I3 의 등록 근거 자체가 "앵커 자체의 교차빌드 재현(**선행 E-Z3**)"
인데, E-Z3 는 iter0 검사였다(아래 d).

**(d) 운전석 미확인 항목의 해소 — Z-O 판독기는 앵커를 iter0 에만 적용했다**:
`scripts/analyze_a210_zero_opacity_census.py:216-218` — `iter0 = public_blocks[0]` 의
`valid0` 에서만 앵커를 찾는다. Z-O 판정 E-Z3 문면도 "iter0 ROW 에 line=669992 …
4필드 hex 동일"(`docs/VERDICT_A210_ZERO_OPACITY_2026-08-26.md:45`). ⟹ 선행 단의 같은
계급 검사가 PASS 한 이유는 **적용 범위가 달랐기 때문**이다. Z2A 판독기는 검사 범위를
iter1·census 측으로 넓혔고, 그 확장이 오탐의 전부다.

**(e) selftest 가 구조적으로 못 잡은 이유**: `--selftest` 의 성한 픽스처(NC-I0)는
`synthetic_zero_row()`(`:372`) **하나를 양 반복에 재사용**한다(`:402-405`
`for _ in range(2)`) — 반복 간 진화가 없는 픽스처라 iter1-적용 결함이 원리적으로 발화할
수 없었다. G2 PASS(NC-I0~I6 7종)는 참이되, 픽스처 현실성의 구멍이 이 결함의 통과 경로다.

---

## 4. 분기 판정 — **ZI-A** (계약 §8-2 판정식 문면 적용)

평가 순서대로:

1. **ZI-X 아님**: `LINE-SATURATION-ISOLATED` 0건 (M3).
2. **ZI-D 아님**: ZI-D 판정식 = "어느 런이든 **미완주**(어떤 차단이든)", 처분 형식은
   "L6C D2 처분 형식 승계" — 즉 **런을 죽인 차단**의 계급이다(L6C D2 = iter0 물리 레인
   사망, `docs/VERDICT_DET_L6C_COVER_2026-08-26.md:68-`). R-II 에 그런 차단은 없다:
   차단 마커 전건 0(M3), 모델은 SUMMARY 2줄·R7 커밋·PHYSICS-COMPARISON·RUN FOOTER 까지
   완주(M2·M4·M5). 런처 die(M6)는 **모델 정상 종료 후 수용 블록**에서, 그 문면대로
   "analyzer failed" — §3 이 실증한 오탐이다.
3. **분할 적용**: f_bit = **1.0** ≥ 1.0 (M7) ∧ E-I4 토큰 **전건 일치** (M8·M9) ∧
   ISOLATED 0 ∧ 완주(위) ⟹ **ZI-A**. §9 기계 선언(PF-2)의 분할 규칙도 동일하게 낸다
   (`"ZI-A": f_bit >= 1.0`).

**"완주" 의 해석 근거** (E-I1 의 sacct 항과의 관계는 §5): §8-2 의 "R-II 완주" 를 모델
트랜잭션의 완주로 읽는 근거 셋 — ① G4(격리 본판정)의 등록 문면은 "완주 ∧ ISOLATED 0 ∧
… SUMMARY 2줄 complete=1 ∧ …" 로 sacct 를 포함하지 않으며 완주를 SUMMARY 와 별도
항목으로 둔다(계약 §4 표). ② §5.5 의 제출 유보 사유문은 "R-II 가 **죽으면** …
두 런은 같은 사유로 **죽거나** 무의미" — 죽음의 언어다. 모델은 죽지 않았다. ③ ZI-D 가
승계하는 L6C D2 형식이 정확히 런-사망 계급이다(위 2). 이 해석 하에서 ZI-A 의 네 항은
전건 오탐과 무관한 1차 실측으로 채워졌다 — 골대 이동이 아니다.

⟹ **분기 = ZI-A. 판독기 라벨 ZI-B 는 기각**하고, 그 기각의 근거(오탐 기전·자기입력
실증)는 §3 에 실측으로 남긴다.

---

## 5. 기대치 판정 (계약 §8-1 — 게이트/관측 구분은 등록 그대로)

| # | 등급 | 판정 | 근거 |
|---|---|---|---|
| E-I1 | 게이트 | **FAIL (문면)** — 3항 중 `sacct COMPLETED` 위반(State=FAILED 70:0, M1). `model rc=0` ✓(M2) · `SUMMARY 정확 2줄 complete=1` ✓(M4). **위반 항의 소유 = 수용 하니스**: die 는 모델 사후·사유 문면 "analyzer failed"(M6), 그 판독기 FAIL 은 §3 실증 오탐 ⟹ sacct 항의 위반은 런의 성질이 아니라 **실격 계측의 출력이 job 상태로 전파된 것**이며, 격리 계약에 대한 반증 정보를 담지 않는다. 그래도 등록 문면은 문면이다 — FAIL 로 기재한다 |
| E-I2 | 게이트 | **PASS** | ISOLATED 0 ∧ BLOCKED 0 (M3) |
| E-I3 | 게이트 | **PASS** (**E-I3 PASS**) | 양 반복 각 221,382행·총 442,764행 **전건 bit-동일**(M7, f_bit=1.0) ∧ 앵커 4필드: 등록 상수(iter0 유래 — §3-c·d)와 iter0 행 전건 bit-일치, iter1 행은 census 와 bit-동일(M10). 판독기의 iter1-상수 적용은 계약 밖 검사(§3) — E-I3 위반이 아니다 |
| E-I4 | 게이트 | **PASS** (**E-I4 PASS**) | 토큰 전건: candidate_rows=749644 · zero=221382 · selected 1201/992 · total_scaled_emission 양 반복 bit-동일 · sprobe_undefined_rows=221382(양 반복) · 선정열 2,193쌍 동일 (M8·M9). [추정 1건(§8-1)도 in-vivo 해소 — hazard 행 전건 저장 도달·ZI-X 미발화, M3·M7] |
| E-I5 | 게이트 | **미착지** | R-III 미제출 (§7 참조) |
| E-I6 | 관측 | 미착지 | 〃 |
| E-I7 | 관측·조건부 | 미착지 | 〃 |
| E-I8 | 게이트 | **미착지** | R-IV 미제출 |
| E-I9 | 관측 | **부합** | Elapsed 04:34:36 ≈ 04:34 등급 · MaxRSS 35.4 GiB ∈ 31-38 GiB (M1) |

**E-I1 에 대한 대장 기재 (수리 제안 없음)**: E-I1 의 sacct 항은 "모델 완주" 와 "수용
성공" 을 한 마커에 묶었다 — 런처가 판독기 rc 로 die 하는 설계(계약 §5 수용 블록)에서
sacct 는 수용의 함수이지 완주만의 함수가 아니다. 이번이 그 둘이 갈라진 첫 실측이다.
같은 구성은 선행 E-Z1 에서 승계됐으나(Z-O 판정 §2) 거기서는 갈라지지 않았다.

---

## 6. 계약 본체 판정 — 격리가 작동했는가 (G4)

**계약 (하나)**: "진단 트랜잭션의 실패는 진단만 무효화하고, 물리 트랜잭션을 중단시키지
않는다." 이 단의 판별 설계는 **봉인 사망↔완주 페어**(G4 in-vivo 음성 대조)다.

**페어의 성립 요건 — 전건 실측 확인**:

| 요건 | 실측 |
|---|---|
| 같은 env | exports 경로 정규화 후 **diff 공백** — 캡처 2종=1·DIAG=2·ION=1·TE_K=10020 동일 (M11). 스테이지 봉인 `resolved_env_delta_count=0`·`shared_hardlinks=0` (`<RII>/Z2A_STAGE_SEAL.txt`) 과 합치 |
| 같은 base | `Z2A_STAGE_SEAL.txt` `sealed_base=<L6C>` — L6C 봉인 base 자체에서 스테이징 |
| 다른 것 = 바이너리 하나 — 단 그 **소스 delta 는 커밋 둘**(`e500e29`+`406712d`) | sha256·git_head 상이(M12); 심볼 delta = 두 커밋 변경집합의 합 — ROW 표지=`e500e29` 산 · ISOLATED/sprobe 심볼·구 센티널 소멸=`406712d` 산 (M12). **귀속에 필요한 추가 단계는 아래 「페어 전제의 정정」** |
| 구판의 사망 | `<L6C>/stderr.log:1245` `INDEPENDENT_SPROBE_UNDEFINED`(candidate_rows=66110 시점) → `:1250` R7 FATAL → `model.rc=1`, SUMMARY 0줄, 총 1,251행 (M13) — job 322449 (L6C 판정문 D2) |
| 신판의 완주 | 발화 **0회**(M3) · SUMMARY 2줄 complete=1 · R7 iter=1 커밋 · FOOTER (M4·M5) — stderr 449,757+행 |

**hazard 조건은 실제로 발생했는가** — 발생했다. 이것이 판정의 핵심 항이다: L6C 를 죽인
조건(capture ON ∧ τ==0 행)은 R-II 에서 **반복당 221,382회, 총 442,764회** 지나갔고
(M7; SUMMARY `sprobe_undefined_rows=221382` 양 반복, M8 — 신설 계수기가 그 발생을
명시 계수), 물리는 계속됐다. L6C 는 스캔 순서상 첫 hazard(line=669992, census `:1242`)
에서 죽었다 — R-II 는 정확히 그 행을 기록하고(`<RII>/stderr.log:1244`) 지나갔다.

**★페어 전제의 정정 — 감리-B 차단 #14 의 해소 (개정 1)**: 초판 §6 은 「두 런의 차이는
Z-2A 변경집합뿐」을 전제로 썼다 — **그 전제는 거짓이었다.** 실측(M12): 두 바이너리
사이의 소스 delta 는 `git log 0e899cb..406712d -- src/lumina_plasma.c` 로 **커밋 둘** —
`e500e29`(Z-O Z-1r 행-계측) + `406712d`(Z-2A 격리)다. 신판 심볼의 ROW 표지
(`LINE-SATURATION-ZERO-OPACITY-ROW` 신판 1·구판 0 — 감리-B 실측, 판정자 재확인
[§12-8])도 `e500e29` 산이다. 따라서 «완주는 격리 덕» 에는 단계가 하나 더 필요하다 —
**`e500e29` 가 사망↔완주를 가를 수 없음의 논증**:

1. **`e500e29` 의 모델 소스 변경은 hunk 정확 1개다** — `git diff 0e899cb e500e29 --
   src/lumina_plasma.c` 의 hunk 는 `@@ -14072,7 +14072,21 @@` **단독**(판정자 직독
   [§12-8]): census 분기(`if(tau==0.0&&n_upper>0.0)`) 안에 ROW `fprintf(stderr,…)` 를
   추가할 뿐이다. return 없음 · 가드 술어 입력(`independent_capture`·`source_defined`·
   `source_function`) 무변경 · 상태 변경은 `e500e29` 이전부터 있던 카운터 증가뿐.
2. **L6C 를 죽인 가드는 `e500e29` 트리에 원문 그대로 존속한다** —
   `INDEPENDENT_SPROBE_UNDEFINED` 차단(`a210_line_saturation_blocked(…)` → `return -1`)은
   0e899cb `:14135` ↔ e500e29 `:14149`(유일 hunk 가 가드를 지나지 않으므로 텍스트 불변,
   행번호 +14 이동뿐 — 판정자 양 트리 직독 [§12-8]). **가드를 제거한 유일 커밋은
   `406712d` 다**(그 diff 의 hunk `@@ -14117,39 +14125,8 @@` 가 가드 블록을 삭제; HEAD
   심볼 0 — §12-13).
3. ★**`e500e29` 는 가설의 커밋이 아니다 — in-vivo 실적이 있다** (감리-A(재) 실측 →
   판정자 독립 재현, M17): 봉인 ZO census 런 자체가 정확히 그 커밋이다
   (`<ZO>/input/git_head.txt` = `e500e2995814…` 단독). 그 런은 **L6C 가 죽은 바로 그
   노드(syn102)** 에서 capture=0 으로 **완주**했다(model.rc=0·SUMMARY 2줄) — 노드 실측:
   L6C=syn102(job 322449)·census=syn102(job 324388)·R-II=syn101(job 324557), 각 런루트
   `slurm_environment_before_sanitize.txt` 의 `SLURMD_NODENAME` [§12-14]. 그리고 그
   hazard 행 442,764행 전건이 R-II 와 byte-동일하다(M7) ⟹ **e500e29 바이너리의
   가드-상류 연산은 전건 in-vivo 실측이 끝나 있다.**
4. **미실측 링크는 env 플래그 capture 0→1 단 하나다**: census↔L6C exports diff 는
   캡처 2종(`LUMINA_A210_INDEPENDENT_CAPTURE`·`_SPRODUCER_CAPTURE`)의 0↔1 **정확 두 행**
   뿐이다 [§12-14]. 그 플래그가 사망 경로에 진입하는 자리는 가드 술어 **제1연언지**
   (`if(independent_capture && (!source_defined || !isfinite(source_function)))`,
   e500e29 `:14125`)이고, 그 술어는 0e899cb `:14111` 과 **자구 동일**하다(diff 공백
   [§12-14]). 같은 함수의 유일한 다른 제어 흐름 진입(INVALID_ROW_PROVENANCE 가드의
   j_cont 절, e500e29 `:14054`)은 R-II(capture=1·byte-동일 스트림)에서 발화 0(M3
   BLOCKED 0) — 전 행 거짓이 in-vivo 실측됐다.
5. **링크의 양쪽 끝이 실측이다**: (0쪽) capture=0 ⟹ 완주 — census 실측(위 3). (1쪽)
   `0e899cb + capture=1` ⟹ **같은 노드(syn102)·같은 첫 hazard 행에서 사망** — L6C
   WITNESS `:1244` 가 line=669992 를 지목하고 `:1245` BLOCKED(candidate_rows=66110),
   census 첫 ROW `:1242` 가 정확히 그 행이다(M13·[§12-14]). ★아울러 hazard 행
   **442,764/442,764 전건 `source_defined=0`**(판정자 계수 — R-II·census 각각, 반례 0
   [§12-14]) ⟹ 가드 제2연언지가 전 행 성립 ⟹ **가드 보유 바이너리 + capture=1 은 단
   한 행도 통과할 수 없다** — 첫 hazard 행 사망은 우연이 아니라 전건 필연이고, stderr
   인쇄(fprintf)가 하류 가드를 침묵시킬 기전은 소스에 없다.
6. **계보 — 선행 폐합이 같은 성격을 확정했다**: Z-O 계약 §5 변경집합 표가 이 커밋의
   성격을 사전등록했고(「판정 분기·반환값·차단 시점 불변. 물리식 무접촉.」 —
   `docs/RUNG_A210_ZERO_OPACITY_2026-08-19.md` §5 표 `src/lumina_plasma.c` 행), Z-O 판정
   Z1 이 hazard 분기 직독으로 보강 확인했으며(「카운터 증가와 fprintf 뿐, return·상태
   변경 없음」 — `docs/VERDICT_A210_ZERO_OPACITY_2026-08-26.md:82`), 그 판정은 감리 두 겹
   (A: 차단 0 · B: 차단 0)을 통과해 폐합됐다(장부 = Z-O 사전등록 §10).

**논증의 계급 (개정 2 — 실측으로 승격. 낮춰 쓰는 것도 부정확이다)**: 개정 1 은 이 논증을
「소스 직독 + 결정론 실측 + 선행 폐합 계보」로 한정하고 「`e500e29`-단독 + 캡처 ON 의 실런은
존재하지 않는다」로 낮춰 썼다 — **그 한정은 과도하게 약했다**(감리-A(재) 실측). 정확한
계급은 **소스 직독 + `e500e29` 의 in-vivo 실적(같은 노드 완주·가드-상류 442,764행
byte-동일, 위 3) + capture 플래그 단일 링크(위 4) + 전건 `source_defined=0`(위 5)** 다.
형식상 부재인 실런은 `e500e29`-단독+capture=1 하나뿐이나, 그 런이 지나야 할 가드는 술어가
0e899cb 와 자구 동일하고 제1연언지(capture=1)·제2연언지(`source_defined=0`, 전 행)가 모두
실측 참이다 — **단 한 행도 통과 불능.** 이 계급이 기각하는 대안 "`e500e29` 가 완주를
만들었다"는 상류(무접촉 소스·in-vivo byte-동일)와 하류(전건 실측 술어) 양쪽에서 막혀 있다.

**대안 설명의 소거**: (a) "hazard 미발생으로 인한 완주" — 기각(위). (b) "env·base 차이"
— 기각(M11·스테이지 봉인). (c) "결정론 파탄으로 다른 경로 통과" — 기각: 진단 스트림
전행 bit-동일(M7)·SUMMARY 총계 bit-동일(M8)·R1/R2/R1' 잔차 3에포크 자릿수 동일(M16).
(d) 노드 상이(L6C=syn102, R-II=syn101; 동일 A100-SXM4-80GB×2): 차단은 결정론적 코드
경로(구판 소스의 가드)이지 하드웨어 사건이 아니며, 442,764행 byte-동일이 **교차-노드**
(census=syn102 ↔ R-II=syn101) 실측이고(M7·M17), e500e29 완주와 0e899cb 사망은 **같은
노드(syn102)** 실측이다 — 노드는 사망↔완주를 가르는 변수가 아니다. (e) "동승 커밋
`e500e29`(Z-1r 계측)가 만든 완주" — 기각: 위 「페어 전제의
정정」 1-5 (가드 무접촉·제어 흐름 무변경·in-vivo 실적·링크 양끝 실측·가드 제거는
`406712d` 단독). ⟹ **완주의
소유자는 Z-2A 변경집합(`406712d`)이다 — 바이너리 delta 의 소스는 커밋 둘이지만, 사망
기전(가드)에 접촉한 것은 `406712d` 하나다. 봉인 사망↔완주 페어는 음성 대조로
성립한다.**

**실증 범위의 정직한 한정** (과대주장 금지):

1. **Class 1(정당 미정의 상태의 탈-실패화)** — **in-vivo 실증 완료**. 가드 삭제 경로가
   442,764회 실통과했고 행·계수·보고가 계약 §2-1 사양 그대로다(증가 지점
   `src/lumina_plasma.c:14156-14157`, SUMMARY 2변형 `:14413`·`:14524` — 전건 R-II 에서
   실발화 확인).
2. **Class 2(진단 실제 실패의 격리)** — **발동 실적 0**(ISOLATED 0 — E-I2 가 요구한
   그대로). in-vivo 실증이 아니라 **G2 selftest 계급 검증**(NC-I1)까지다. 격리 인쇄
   지점(`src/lumina_plasma.c:15328-15341`)과 `UPSTREAM_LINE_SCAN_INCOMPLETE` 보존
   (`:15323-15327`)은 소스·바이너리 실존만 확인(M12).
3. **물리 무접촉(G4)의 실증 범위 = 계약이 등록한 bit 대조 범위**: Z-1r ROW 4필드
   ×442,764행 + SUMMARY 토큰(총계 21자리 포함) + 선정열 (rank,line). 이 범위 안에서
   **완전 bit-동일**이다 — 진단이 표본하는 물리량(**hazard(Z-1r) 행**의 n_upper·A_ul·
   emission_per_sr, 양 반복)이 capture=0 봉인 census 와 bit 로 같다. bit-대조된 값
   필드는 Z-1r ROW 의 것이다 — Stage-4 **선정행**에서는 (rank,line) 두 토큰만 소비했다
   (§7). **이 범위 밖**
   (값 파일 전역 — spectrum/plasma CSV·physics_DET)은 이 단에서 **G6/R-IV 의 등록
   몫**이고 R-IV 는 미착지다 ⟹ 전역 물리 무접촉은 **주장하지 않는다**. R1/R2 잔차
   bit-재현(M16)은 등록 범위 밖 보강 관측으로만 기재한다.

---

## 7. 경계 준수 (계약 §7) 와 R-III·R-IV 처분

**§7 준수 기재**: R-II 가 부수 산출한 ion=1 Stage-4 행(capture 필드 보유 선정행)은
**봉인 상태 그대로 두고 어떤 물리 판정도 내리지 않았다.** 판정자는 선정행에서
(rank,line) 두 토큰만 읽었고(M9 — 선정열 대조는 계약 E-I4 의 등록 항목) S_probe·J̄
값 필드는 열람하지 않았다. 본선 물리 계약(A2-07 solve 의 J̄ 응답)의 미결은 이 단 뒤에도
미결이다 — 바뀐 것은 측정이 가능해졌다는 사실뿐(Z-O 판정 §9 #4 문언 승계). 수리 제안
없음 — 발견의 처분은 §9 의 조용한 대장 기재뿐이다.

**R-III·R-IV — 현 상태 제출 불가.** 근거:

1. 계약 §5.5 표의 제출 시점 문면 = "**R-II 런처 수용 PASS 후**". 수용 PASS 실물이
   없다 — 런처는 die 했다(M6). 이 판정문의 ZI-A 는 §8-2 판정식의 적용이지 런처 수용
   PASS 의 대체물이 아니다.
2. 같은 계약 안에서 §8-2 ZI-A 처분행은 "R-III·R-IV 진행" 이라 쓴다 — **§5.5 의 기계
   전제와 §8-2 의 처분이 이 결과(모델 완주 ∧ 수용 거부)에서 갈라진다.** 계약이 이
   경우를 예정하지 않았다. 갈림의 해소(판독기 결함의 처분과 수용 실물의 회복 경로 —
   재발주든 개정이든)는 **사전등록 저자(개정15 형식)와 운전석의 몫**이다. 판정문은
   제출을 개시하지 않는다.
3. 부기: R-III/R-IV 판독 경로(`analyze_a210_z2a_probe.py:309-370`)는 census 대조·앵커
   검사를 지나지 않으므로 이 오탐이 그 두 런의 수용을 직접 막지는 않는다. 그러나 제출
   전제가 R-II 수용인 이상 이 사실은 처분을 바꾸지 못한다.

**후속 단 함의 (기재 의무)**: DET-L6C-COVER-R(`docs/RUNG_L6C_REDISPATCH_2026-08-27.md`,
커밋 `776efeb`)의 PR-0 은 ① R-II 착지(**sacct COMPLETED**) ② Z-2A 판정 ZI-A + 감리
두 겹 통과(**문면 리터럴 매칭** — 원문 리터럴은 `:318` 참조)를 요구한다. ② 의 분기·
게이트 앵커(ZI-A·E-I3 PASS·E-I4 PASS)는 이 판정문이 공급하고 감리의 최종 상태(충족)는
§13 스탬프·장부가 공급하나, ② 가 요구하는 **감리 문구 리터럴은 이 판정문이 의도적으로
본문에 싣지 않는다**(§10 fail-open 경고) ⟹ ② 도 현 문면 그대로면 리터럴 부재로
오차단된다(fail-closed 방향 — 봉인 위험 없음). **① 은 job 324557 로는 영구
불충족**이다 — sacct 상태는 불변이다. ⟹ 현 문면
그대로면 COVER-R 은 X-P 로 간다. 해소는 PR-0 개정(폐합 후 큐 §11 #2 — ① 의 개정 또는
R-II 재착지, ② 의 감리 앵커를 장부로)이며 그 단 저자·user 의 결정 사항이다 — 판정문은
사실과 함의만 기재한다.

---

## 8. 검수 인계 사항의 처분 (개정16 ② — 판정 시 확인 요청분)

| 항 | 처분 |
|---|---|
| Q6-N1 (R-III/IV 도 `branch=ZI-A` 라벨) | **코드로 확인** — `analyze_a210_z2a_probe.py:363` `branch = "ZI-A" if not errors else "ZI-P"`. R-III/IV report 는 `mode` 필드와 함께 읽어야 한다. 이번 판정 하중 0(두 런 미착지) — 후속 판독 시 주의로 존속 |
| Q2-N1 (init 실패·스캔 전 물리 조기 return 시 꼬리 ISOLATED 생략) | in-vivo 미발화(ISOLATED 0·init 실패 0). **한계로 존속** — §9 대장 기재 |
| 불확신 (a) 빌드·심볼 미재현 | **해소** — 스테이지드 바이너리 양쪽 strings 실측(M12): 신판 심볼 실존·구 센티널 소멸, 구판 정반대. in-vivo 로는 신설 토큰(`sprobe_undefined_rows`)이 **실발화**(M8)했고, ISOLATED 코드는 심볼 **실존**까지다(M12 — 실발화 아님: 발화 0 이 E-I2 의 요구다, §6-2) |
| 불확신 (b) in-vivo bit 동일 미검증 | **해소** — M7 (442,764행 byte-동일, 판정자 독립 cmp) |
| 불확신 (c) E-I4 추정 1건 (capture=1 hazard 셀의 producer decode 계약) | **해소** — hazard 행 전건 저장 도달(M7)·ZI-X/BLOCKED 미발화(M3): 추정이 in-vivo 로 참이었다 |
| 불확신 (d) 판독기 메모리 | **해소(이 규모 한정)** — 152MB 스트림을 in-job(러너)·판정자 재실행 양쪽에서 완주. 더 큰 로그로의 일반화는 주장하지 않는다 |
| 불확신 (e) `verify_export_delta` sed 패턴 | **이 런에 한해 우회 해소** — 판정자가 sed 패턴과 무관하게 exports 전문 diff 로 delta 0 을 독립 확인(M11). 패턴 자체의 검증은 미실시 — 존속 |

---

## 9. 발견의 처분 — 조용한 대장 기재 (수리 제안 없음, 국면 규칙별 별도 결정)

1. **판독기 앵커 검사 결함** — iter0 상수(`:29-34`)의 양 반복·양 스트림 적용(`:290`).
   성한 봉인 기준 자기입력에서 FAIL(§3-b) ⟹ 판별력 0. 등록 계보: 계약 E-I3 은 iter0
   유래 상수만 등록(§3-c·d), iter1 적용은 계약 밖.
2. **selftest 픽스처 비현실성** — NC-I0 픽스처가 반복 간 진화 없음(`:372`·`:405`) ⟹
   이 계급의 결함을 원리적으로 검출 불능. (G2 의 "성한 픽스처" 요건이 '성함'의 현실성을
   보증하지 않는다 — 계측 부채 계급.)
3. **E-I1 sacct 항의 조작화 얽힘** — 완주 마커에 수용 성공이 묶임(§5 말미). 이번이 첫
   분기 실측.
4. **계약 내 처분 갈림** — §5.5 "런처 수용 PASS 후" ↔ §8-2 ZI-A "R-III·R-IV 진행"
   이 (완주 ∧ 수용 거부) 에서 상충(§7-2).
5. **COVER-R PR-0 ① 의 job-영구성 함정** — "sacct COMPLETED" 문면은 실패 job 재판독
   으로는 영구 불충족(§7 후속 단 함의).
6. (승계 확인) 검수 Q5-N1 — NC-I1/I5 픽스처의 f-string 누락으로 `{clean}` 이 리터럴
   텍스트로 들어감(`:427`·`:438` — 이어붙인 둘째 문자열 리터럴이 f-string 이 아님):
   기대 분기(ZI-X/ZI-D)는 ISOLATED/BLOCKED 검출의 조기 return 경로라 판정 불변 —
   비차단 유지, 픽스처 의도와 실물의 불일치로만 기재.
7. **판정문 개정 전 판본 미보존** (감리-A(재) 지적 — 절차·대장 계급): 이 판정문은 미추적
   파일이라 개정 1 적용 시 개정 전 판본이 보존되지 않았고, 「분기·판정 불변」 선언을 행
   단위 diff 로 기계 검증할 수 없었다(감리-B 도 같은 제약). 판정문 개정 형식은 등록 규칙이
   없어 위반은 아니나, 이 캠페인에는 **사전등록 개정이 소리 없이 치환한 전례**가 있다
   (개정15 형식의 성립 사유). 개정 2 부터는 저자가 편집 전 판본을 보존했다
   (`docs/VERDICT_A210_Z2A_DIAG_ISOLATION_2026-08-27.md.rev1`, sha256 `cd73bea8…` —
   COVER-R 판독기류의 `*.md` glob 오매칭을 피하는 확장자) ⟹ 개정 2 의 불변 선언은 행
   단위 diff 로 검증 가능하다. 「판정문 개정 시 개정 전 판본 보존」은 **차기 규약 후보로
   인계**(폐합 후 큐 §11 #4).
8. **운전석 감리 발주문의 노드 오기** (감리-A(재) 지적 — 기재만, 재발 방지): 발주문이
   L6C 를 「syn104 계열」로 썼으나 실측은 **syn102** 다(M17·[§12-14]). 이 판정문 본문
   (§6 대안 소거 d)은 초판부터 syn102 로 옳게 썼다 — 판정 하중 0.

---

## 10. 폐합 (조건 충족) 과 후속 문면 앵커 (COVER-R PR-0 매칭용)

이 판정문의 분기·게이트 문면: **ZI-A** · **E-I3 PASS** · **E-I4 PASS**.

**이 판정은 폐합이다 — 규약(개정17)의 폐합 조건이 충족됐다.** 최종 상태는 §13 스탬프의
구조화 표가 기술하고, 감리 기재의 정본은 이 문서가 아니라 장부다 — 장부 = 사전등록 §10.
초판·개정 1 이 폐합 조건에 병기했던 「ZI-A 처분 잔여(G5·G6)의 착지」는 §7 실측(수용 실물
부재 — 그 회복이 계약 개정·재발주라는 이 단 밖의 행위를 요구)으로 이 단이 자기 힘으로
착지시킬 수 없는 항목이므로, 개정 2 에서 폐합 조건이 아니라 **폐합 후 큐**(§11)로 옮겨
명기한다 — G5·G6 의 **미결 분류 자체는 불변**이다(⚠골대 이동 아님: 옮긴 것은 처분의
거처이지 판정·분류가 아니다).

**★후속 단 계측 경고 (감리-A 지적 ③ — COVER-R 로 인계, 기재 의무)**: COVER-R PR-0 ②
는 문면 리터럴 매칭이고 그 판독기는 아직 미구현(발주 산물)이다. 판독기가 **판정문
텍스트를 스캔**하는 설계가 되면, 판정문 조건문 속의 감리-통과 리터럴이 **감리 미통과
상태에서도 ② 를 오충족시키는 fail-open 벡터**가 된다 — 이 문서 초판이 실제로 그
리터럴(COVER-R 사전등록 `:318` 의 문면 — 같은 이유로 이 개정에는 재수록하지 않는다)을
조건문으로 보유했다. 처분은 COVER-R 단 몫이다: 판독기는 그
앵커를 판정문이 아니라 **장부(사전등록 §10 감리 칸)**에서 찾아야 한다. 이 개정은 본문
조건문의 해당 리터럴 문면을 제거했으나, 인계 의무는 그와 무관하게 존속한다.

---

## 11. 폐합 후 큐 (미결 명기 — 확정/미결 분류 불변)

| # | 항목 | 성격 / 조건 |
|---|---|---|
| 1 | **계약 §5.5↔§8-2 갈림 해소 + 「완주」 정의 고정** — 모델 완주(SUMMARY·R7·FOOTER)인가 `sacct COMPLETED` 인가 (§7-2·§9-3·§9-4) | 사전등록 저자 개정(개정15 OLD/NEW) 또는 재발주 — R-II 수용 실물 회복(판독기 결함 §9-1·2 의 처분 포함)의 전제. 저자·운전석·user 몫 |
| 2 | **COVER-R PR-0 개정** — ① `sacct COMPLETED` 가 job 324557 로 **영구 불충족**(§7·§9-5) ② 감리 앵커를 판정문 텍스트가 아니라 **장부(사전등록 §10 감리 칸)** 에서 찾도록(fail-open 수리 — §10 경고) | COVER-R 단 저자·user 몫 |
| 3 | **Z2A 판독기 앵커 검사 수리** — iter0 상수의 양 반복·양 스트림 적용 결함(§3·§9-1); 봉인 census 자기입력에서도 FAIL(판별력 0, §3-b) + selftest 픽스처 비현실성(§9-2) | 국면 규칙별 별도 결정 — 대장 §9-1·2 지목(수리 제안 아님) |
| 4 | **판정문 개정 형식 규약화** — 개정 전 판본 보존(§9-7) | 차기 규약 후보 — user 결정 |
| 5 | **R-III (G5=Z3+Z2 제3원) · R-IV (G6=Z4)** — ZI-A 처분 잔여 | #1(·필요시 #2) 해소 후 제출(§5.5) — 미결 분류 불변 |
| 6 | **본선 물리 계약 — A2-07 solve 의 J̄ 응답** = DET-L6C-COVER-R (사전등록 `776efeb` 기완성·잠김) | PR-0 개정(#2) 후 |
| 7 | Class 2 격리의 in-vivo 미실증 (§6-2) | 실제 진단 실패 발생 시에만 — 유발하지 않는다 |
| 8 | ion_owner 같은 계급 이웃 (계약 §2-6) · 스캔 범위 밖 coverage (Z-O §9 #6) | 존속 — 의무 없는 미결 |

---

## 12. 부록 — 재현 명령 (전건 판정자 실행 확인, 2026-08-27, syntax 로그인 노드 비연산 계급)

`RII=/gpfs/kjhan/lumina/a210_z2a_diag_isolation_a100x2_k36/rii_20260826T134337Z_ion1`
`ZO=/gpfs/kjhan/lumina/a210_zero_opacity_z1r_a100x2_k36/zo_20260826T050021Z_census_ion1`
`L6C=/gpfs/kjhan/lumina/det_stage12_fixed_te_a100x2_k36/l6c_20260822T154507Z_cover`
`S=<스크래치>` (봉인 3곳 쓰기 0 — 산출 전부 스크래치)

```bash
# 12-1 job 계정·die 문면
#  (주: sacct 재현에는 Slurm DB 접근 권한 필요 — Codex 샌드박스에서 소켓 Operation not
#   permitted 로 재현 불능 [감리-B NOTE#1; Z-O NOTE#2 전례]. 운전석 대리 실행 일치:
#   324557 FAILED 70:0 · Elapsed 04:34:36 · MaxRSS 37165420K · syn101 · a100)
sacct -j 324557 -o JobID,JobName%20,State,ExitCode,Elapsed,MaxRSS,Partition,NodeList
#  -> FAILED 70:0 04:34:36 a100 syn101 · batch MaxRSS=37165420K ; WorkDir=본 저장소
cat /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/slurm-324557.out   # 정확 1행:
#  -> DET_FLIGHT_FATAL Z2A R-II analyzer failed
cat $RII/model.rc                                                    # -> 0

# 12-2 차단·완주 마커
for p in LINE-SATURATION-ISOLATED LINE-SATURATION-BLOCKED INDEPENDENT_SPROBE_UNDEFINED DET_FLIGHT_FATAL; do
  grep -c "$p" $RII/stderr.log; done                                 # -> 0 0 0 0
grep -n 'R7_MATERIAL_PHASE_COMMITTED lane=DET iter=1' $RII/stderr.log  # -> :449756 te_generation=2->3
grep -n 'PHYSICS-COMPARISON.*iter=1 status=COMMITTED' $RII/stderr.log  # -> :449757
wc -l $RII/RUN_FOOTER.txt                                          # -> 111 (전건 key=value)
grep -c '^LUMINA_' $RII/RUN_FOOTER.txt                             # -> 89
grep -c '^OMP' $RII/RUN_FOOTER.txt ; grep -c '^CUDA' $RII/RUN_FOOTER.txt   # -> 5 · 1
grep -vcE '^(LUMINA_|OMP|CUDA)' $RII/RUN_FOOTER.txt                # -> 16 (provenance 키)

# 12-3 SUMMARY 2줄·토큰 자면 대조
grep -n 'LINE-SATURATION-SUMMARY\]' $RII/stderr.log   # -> :225083 / :449543 (전문은 본문 M8)
grep    'LINE-SATURATION-SUMMARY\]' $ZO/stderr.log    # 공유 토큰 자면 동일, R-II 만 sprobe 토큰 추가

# 12-4 ROW 스트림 byte-대조 (f_bit 독립 재확증)
grep -F '[A2-10][LINE-SATURATION-ZERO-OPACITY-ROW]' $RII/stderr.log > $S/rii_zero.txt
grep -F '[A2-10][LINE-SATURATION-ZERO-OPACITY-ROW]' $ZO/stderr.log  > $S/census_zero.txt
wc -l $S/*.txt          # -> 각 442764
cmp $S/rii_zero.txt $S/census_zero.txt && echo BYTE-IDENTICAL       # -> BYTE-IDENTICAL
awk '/ZERO-OPACITY-ROW/{n++} /LINE-SATURATION-SUMMARY\]/{print n; n=0}' $RII/stderr.log  # -> 221382 / 221382

# 12-5 선정열 (rank,line)
grep -F '[A2-10][LINE-SATURATION-ROW]' $RII/stderr.log | grep -o 'rank=[0-9]* line=[0-9]*' > $S/rii_sel.txt
grep -F '[A2-10][LINE-SATURATION-ROW]' $ZO/stderr.log  | grep -o 'rank=[0-9]* line=[0-9]*' > $S/census_sel.txt
wc -l $S/rii_sel.txt    # -> 2193 (=1201+992)
cmp $S/rii_sel.txt $S/census_sel.txt && echo IDENTICAL              # -> IDENTICAL

# 12-6 앵커 행 (양 런·양 반복) + hex
grep -n 'ZERO-OPACITY-ROW.* line=669992 ' $RII/stderr.log   # -> :1244(iter0) :226122(iter1)
grep -n 'ZERO-OPACITY-ROW.* line=669992 ' $ZO/stderr.log    # -> :1242 :226118 (내용 R-II 와 자면 동일)
python3 -c "print(float('0.0004437688873013011').hex(), float('0.00045264235457262295').hex(),
float('2.7310010625545125e-16').hex(), float('2.7856093265405119e-16').hex(),
float('0.52225560000000004').hex(), 0.00045264235457262295/0.0004437688873013011-1)"
#  -> 0x1.d1534dba20c26p-12 0x1.daa141c1395e6p-12 0x1.3adcec9caadc0p-52 0x1.4128acfd24836p-52
#     0x1.0b6516044e436p-1  0.019995694...   (iter0 3필드=등록 상수·iter1 2필드 상이·A_ul 불변)

# 12-7 env delta 0
diff <(sed "s|$RII|RUNROOT|g" $RII/input/resolved_lumina.exports) \
     <(sed "s|$L6C|RUNROOT|g" $L6C/input/resolved_lumina.exports) && echo IDENTICAL   # -> IDENTICAL
head -30 $RII/Z2A_STAGE_SEAL.txt   # sealed_base=$L6C · resolved_env_delta_count=0 · shared_hardlinks=0

# 12-8 바이너리 페어 — sha256·git_head·심볼 + ★소스 delta 두 커밋의 분해 (개정 1)
cat $RII/input/binary.sha256 $L6C/input/binary.sha256               # -> 39d452e4… / b9a30a81…
cat $RII/input/git_head.txt $L6C/input/git_head.txt                 # -> 406712df… / 0e899cb3…
for b in $RII/input/lumina_cuda $L6C/input/lumina_cuda; do
  for s in LINE-SATURATION-ZERO-OPACITY-ROW LINE-SATURATION-ISOLATED \
           sprobe_undefined_rows INDEPENDENT_SPROBE_UNDEFINED; do
    strings -a $b | grep -c "$s"; done; done                        # -> 1 1 3 0 / 0 0 0 1
git log --oneline 0e899cb..406712d -- src/lumina_plasma.c           # -> 406712d(Z-2A) · e500e29(Z-1r) 둘
git diff 0e899cb e500e29 -- src/lumina_plasma.c | grep -c '^@@'     # -> 1 (@@ -14072,7 +14072,21 @@ 단독 = ROW fprintf)
git show 0e899cb:src/lumina_plasma.c | grep -n INDEPENDENT_SPROBE_UNDEFINED   # -> :14135
git show e500e29:src/lumina_plasma.c | grep -n INDEPENDENT_SPROBE_UNDEFINED   # -> :14149 (가드 원문 존속 — §6 정정 블록 2)
git show 406712d -- src/lumina_plasma.c | grep -n '^-.*INDEPENDENT_SPROBE_UNDEFINED'
#  -> 187:-        a210_line_saturation_blocked("INDEPENDENT_SPROBE_UNDEFINED",   (삭제 행 정확 1건)
git show 406712d -- src/lumina_plasma.c | grep -c '^+.*INDEPENDENT_SPROBE_UNDEFINED'  # -> 0 (추가 행 0 = 가드 제거는 406712d 단독)
#  (개정 2 정정: '^-' 필터 없는 원 명령은 커밋 메시지의 동일 문자열(:10)도 출력하므로 초판
#   주석 「삭제 행(-)만」은 그 형태에선 부정확했다 — 감리-B 지적. 정정 명령 판정자 실행 확인.)

# 12-9 L6C 사망 사슬 (봉인 실측)
grep -n 'INDEPENDENT_SPROBE_UNDEFINED\|R7\]\[FATAL' $L6C/stderr.log  # -> :1245 BLOCKED / :1250 FATAL
wc -l $L6C/stderr.log; cat $L6C/model.rc                             # -> 1251 · 1
grep -c 'LINE-SATURATION-SUMMARY\]' $L6C/stderr.log                  # -> 0

# 12-10 봉인 report
cat $RII/a210_z2a_probe_report.json    # blocking=['ANCHOR_FIELD_MISMATCH'] branch=ZI-B f_bit=1.0

# 12-11 판독기 재현 + ★자기입력 시연 (report 는 스크래치로 — 봉인 무접촉)
python3 scripts/analyze_a210_z2a_probe.py --mode R-II --stderr $RII/stderr.log \
  --census-stderr $ZO/stderr.log --report $S/repro.json
#  -> status=FAIL branch=ZI-B rc=4 · blocking=['ANCHOR_FIELD_MISMATCH'] f_bit=1.0  (봉인 report 재현)
python3 scripts/analyze_a210_z2a_probe.py --mode R-II --stderr $ZO/stderr.log \
  --census-stderr $ZO/stderr.log --report $S/selfpair.json
#  -> status=FAIL branch=ZI-B · blocking=['SPROBE_COUNTER_MISSING','ANCHOR_FIELD_MISMATCH'] f_bit=1.0
#     (봉인 census 자신이 FAIL — 판별력 0 실증)

# 12-12 R1/R2 잔차 (관측)
grep -n 'EXACT-MULTIGPU-EPOCH' $RII/stderr.log | head -3   # :12 :423 :225301
grep -n 'EXACT-MULTIGPU-EPOCH' $ZO/stderr.log  | head -3   # :12 :423 :225299 — residual 자릿수 동일

# 12-13 소스·커밋 앵커 (HEAD 776efeb 에서 확인)
grep -c INDEPENDENT_SPROBE_UNDEFINED src/lumina_plasma.c            # -> 0
grep -n sprobe_undefined_rows src/lumina_plasma.c | head -6         # :13914 :14157 :14413 :14524 …
grep -n 'if(saturation_diag.invalidated){' src/lumina_plasma.c      # :15328
sed -n '28,34p;188,210p;255p;290p;301p;363p;372p;402,405p' scripts/analyze_a210_z2a_probe.py
sed -n '216,218p' scripts/analyze_a210_zero_opacity_census.py       # Z-O 앵커=iter0 한정
git show --stat 406712d | tail -6                                   # 변경집합 4파일

# 12-14 (개정 2) e500e29 in-vivo 실적 · capture 단일 링크 · 전건 source_defined=0  (M17·§6-3~5)
cat $ZO/input/git_head.txt          # -> e500e2995814f5f4f4eaf82984e48063a4aaf571 (정확히 그 커밋 단독)
cat $ZO/model.rc; grep -c 'LINE-SATURATION-SUMMARY\]' $ZO/stderr.log   # -> 0 · 2 (census 완주)
for r in $L6C $ZO $RII; do
  grep '^SLURMD_NODENAME\|^SLURM_JOB_ID=' $r/slurm_environment_before_sanitize.txt; done
#  -> L6C: syn102 (job 322449) · census: syn102 (job 324388) · R-II: syn101 (job 324557)
diff <(sed "s|$ZO|RUNROOT|g" $ZO/input/resolved_lumina.exports) \
     <(sed "s|$L6C|RUNROOT|g" $L6C/input/resolved_lumina.exports)
#  -> 정확 2행 쌍: LUMINA_A210_INDEPENDENT_CAPTURE "0"↔"1" · LUMINA_A210_SPRODUCER_CAPTURE "0"↔"1"
git show 0e899cb:src/lumina_plasma.c | grep -n 'if(independent_capture &&'   # -> 14111
git show e500e29:src/lumina_plasma.c | grep -n 'if(independent_capture &&'   # -> 14125
diff <(git show 0e899cb:src/lumina_plasma.c | sed -n '14111,14112p') \
     <(git show e500e29:src/lumina_plasma.c | sed -n '14125,14126p') && echo GUARD-TEXT-IDENTICAL
#  -> GUARD-TEXT-IDENTICAL : if(independent_capture && (!source_defined || !isfinite(source_function)))
git show e500e29:src/lumina_plasma.c | grep -n 'independent_capture'
#  -> 제어 흐름 진입은 :14054(INVALID_ROW_PROVENANCE 가드의 j_cont 절)·:14125(사망 가드)뿐
#     (:14038 파라미터·:14202 데이터 필드 대입·:15158-:15207 env 판독·호출부)
sed -n '1244,1245p' $L6C/stderr.log
#  -> :1244 WITNESS line=669992 (census 첫 ROW :1242 와 동일 행·동일 값) → :1245 BLOCKED candidate_rows=66110
grep -F '[A2-10][LINE-SATURATION-ZERO-OPACITY-ROW]' $RII/stderr.log | grep -vc 'source_defined=0'  # -> 0
grep -F '[A2-10][LINE-SATURATION-ZERO-OPACITY-ROW]' $ZO/stderr.log  | grep -vc 'source_defined=0'  # -> 0
#  (각 총 442,764행 — 전건 source_defined=0, 반례 0 ⟹ 가드 제2연언지 전 행 성립)
```

(소스 인용 행은 전부 HEAD `776efeb` 작업 트리에서 판정자가 직독했다 — 단 §12-8·§12-14 의
0e899cb/e500e29 대조는 `git show` 로 해당 커밋 객체에서 직독했다. `src/lumina_plasma.c`
인용 지점(:13914-15341 대역)은 기지 死블록(:1324-1693) 밖이며, 해당 코드의 활성은 in-vivo
발화(M8 의 sprobe 토큰 실인쇄)와 바이너리 심볼(M12)로 이중 확인했다. `lumina_cuda.cu` 는
인용하지 않았다.)

---

## 13. 부록 — 감리 스탬프 (개정17 두 겹 · 최종 상태)

| 겹 | 담당 | 1차 | 2차 | 최종 |
|---|---|---|---|---|
| 감리-A | Fable fresh (판정·검수와 별개 컨텍스트, 개정14 ④) | 통과 — 차단 0 · 지적 3(전건 비차단, 개정 1 반영) | **범위 한정 재감리** — 차단 0 · 지적 3(전건 비차단, 개정 2 반영) | **통과** |
| 감리-B | Codex read-only (계열 분리, 개정17) | **차단 1**(#14 — §6 핵심 논증의 **전제**를 침) · NOTE 2 · 재현 21항목 중 재현 18·불일치 2·재현불능 1(환경 제약 — Slurm DB 소켓, 운전석 대리 실행 일치 §12-1) | **재판정** — 차단 #14 해소 확인 · 새 차단 0 | **통과** |

**감리-A 1차 요지**: 판정식 양 항 **독립 실측** — ROW 442,764행 `cmp` · 선정열 2,193쌍
`cmp` · SUMMARY 자면 대조. 판독기 라벨(ZI-B) 기각의 정당성 3요건 확인 — ① 등록 판정식
우선(계약 §8-2 문면) ② 판별력 0 실증(봉인 census 자기입력 FAIL, §3-b) ③ 비자기이익
(기각이 판정자 자신의 어떤 선행 주장도 구제하지 않음). 지적 반영 위치: 지적①→§8
불확신(a) 병기 해소 · 지적②→§6 「실증 범위의 정직한 한정」 3 라벨 정정 · 지적③→§10
fail-open 경고 기재 + 본문 감리 문면의 사실화.

**감리-A 2차(범위 한정 재감리) 요지**: ⓐ 전제 정정 후 새 논증(§6)의 반례 수색 —
fprintf 반환값 미소비·`#define` 재정의 0·errno 리셋·`setvbuf` 0·stderr 상태 판독 코드 0
⟹ 「fprintf 가 하류 가드를 침묵시킬 기전 없음」 확인. ⓑ 노드 상이에도 성립 —
442,764행 byte-동일이 **교차-노드**(syn101↔syn102) 실측(M7·M17). ⓒ ★1차 통과분의 전제
의존성 분석: ZI-B 기각·분기 ZI-A·E-I 전건·대안 소거 (a)-(d)·Class 한정 — **전부 거짓
전제(「delta=커밋 하나」)에 무의존**. 거짓 전제의 유일한 역할은 소거 목록을 (e) 하나
모자라게 한 것이고 개정 1 이 정확히 그 결손만 메웠다 ⟹ 연쇄 재감리 불요. ⓓ 재감리의
실측 기여: `e500e29` 의 **in-vivo 실적**(봉인 census 런 = 정확히 그 커밋·syn102 완주)을
밝혀 개정 1 §6 의 논증 저평가를 시정할 근거를 공급 — 개정 2 §6·M17·§12-14 로 반영.
나머지 지적 반영: 개정 전 판본 미보존→§9-7 · 운전석 감리 발주문 노드 오기→§9-8.

**감리-B 요지**: ★1차 차단 #14 는 **결론이 아니라 전제**를 쳤다 — 초판의 「두 런의
차이는 Z-2A 변경집합뿐」이 거짓, 실측 소스 delta 는 `e500e29`+`406712d` **커밋 둘**(M12).
저자가 전제를 사실로 교체하고(§6 「페어 전제의 정정」) 논증을 보강해 해소했다 — 귀속
«격리(`406712d`) 덕» 유지. 재판정이 그 해소를 확인했다(새 차단 0). NOTE 반영: #1→§12-1
sacct 대리 실행 주석 · #4→M5 RUN_FOOTER 계수 정정(111행·환경변수 95).

**★폐합 조건(개정17: 두 겹 모두 통과) — 충족** (위 표의 최종 열). 이 스탬프는 장부
(사전등록 §10)의 지목 대상이지 대체가 아니다. 후속 단(COVER-R PR-0)의 감리 앵커는 이
문서의 본문이 아니라 **장부에서** 찾아야 한다(§10 경고 — 본문은 해당 리터럴을 의도적으로
싣지 않는다).

---

## 한 줄 결론

**분기 ZI-A**(판독기의 ZI-B 는 계약 §8-2 판정식 불성립으로 기각 — 오탐이 봉인 census
자기입력에서도 발화함을 실증) · **확정**: 격리 Class-1 이 in-vivo 로 작동했다 — L6C 를
죽인 hazard 조건이 442,764회 지나가는 동안 물리가 완주했다(사망↔완주 페어 성립 —
페어의 소스 delta 커밋 둘 중 사망 기전(가드)에 접촉한 것은 `406712d` 하나, §6 「페어
전제의 정정」: `e500e29` 의 in-vivo 실적·capture 단일 링크·전건 source_defined=0 으로
링크 양끝이 실측이다), 등록 bit-대조 범위 전건 무섭동(f_bit=1.0)·E-I2/
**E-I3 PASS**/**E-I4 PASS** · **미결**(폐합 후 큐 §11): E-I1 문면 FAIL(sacct — 소유는
수용 하니스)로 **런처 수용 PASS 실물이 없어 R-III/R-IV 제출 불가**, Class-2 격리
in-vivo 미실증 · **폐합**: 규약(개정17) 폐합 조건 충족 — 최종 상태는 §13 스탬프,
정본은 장부(사전등록 §10).
