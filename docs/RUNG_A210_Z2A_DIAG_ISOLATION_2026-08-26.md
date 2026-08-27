# 단 사전등록 — A210-Z2A-DIAG-ISOLATION (2026-08-26)

저자 = Fable fresh 컨텍스트 (분담 개정14 ① — 기획·계획·사전등록). 선행 단
A210-ZERO-OPACITY 폐합(분기 **ZC-A**, 커밋 `4b61915`,
`docs/VERDICT_A210_ZERO_OPACITY_2026-08-26.md`)의 §8 처분 — *"Z-2A 발주 준비(§5 재등록·
프리플라이트 재실행) + B 병행 검토"* — 를 집행하는 문서다. **Z-2A 의 선택(A=진단 격리)은
census 무의존으로 기확정**이다(선행 사전등록 `docs/RUNG_A210_ZERO_OPACITY_2026-08-19.md`
§3 개정 1: "A(진단 격리)는 지금 선택한다"). 이 문서가 정하는 것은 선택이 아니라
**설계·계약·게이트·기대치**다.

소스 인용 행 번호는 전부 **현 HEAD `4b61915`** 에서 저자가 직접 확인했다
[저자 실행: `git rev-parse HEAD` → `4b61915911aaf8f2defc21356c4c9afa747bce2a`;
`git diff --stat e500e29 HEAD -- src/` → 공백 = census 런 바이너리 커밋 이후 src 불변].
봉인 런루트에는 아무것도 쓰지 않았다(읽기·grep 만).

## 계약 (하나)

> **A2-10 line-saturation 진단 트랜잭션의 실패는 진단만 무효화하고,
> 물리 트랜잭션을 중단시키지 않는다.**

경계를 명확히 한다:

- **"실패"로 간주되는 것** = 진단 트랜잭션 내부(`a210_line_saturation_init/add/
  log_complete/log_per_ion_union` 와 그 행 빌더)에서 발원한 모든 비정상 — env 파싱 불량·
  스키마/프로버넌스 불량·비유한 파생 진단값·메모리 실패·stderr 쓰기 실패·선정 대수 실패.
  처분 = **진단 무효화 + 사유·건수 보고 + 물리 계속** (§2 Class 2).
- **"실패"가 아닌 것** = capture 가 켜진 채 χ_eff==0(τ==0) 행에서 S_probe 가 미정의인 상태.
  census 가 전건 (a) 정당 exact-zero 임을 실측했다(fa=1.0, 221,382/221,382 — 선행 판정
  §3). 이것은 **정당한 이름 있는 미정의 상태**이며, 행 스키마가 이미 그 표현을 갖고 있다
  (`source_function=UNDEFINED_CHI_ZERO`·`S_probe=UNAVAILABLE`·
  `independent_fields_defined=0`). 처분 = 행을 **저장·계수·보고**한다 (§2 Class 1).
- **그대로 차단이어야 하는 것** = 물리 레인 자신의 가드 전부. 이 단은 그 어느 것도
  제거·완화하지 않는다: validity 상류 필터(`src/lumina_plasma.c:14944-14948`
  `RADEQ_STALE_OPACITY`), 물리 합산 비유한 검출(`RADEQ_NONFINITE`), cancellation census
  fail-closed(`:15078-15079` 계급 — 선행 판정 §5-2 ② 검증분, src 불변 [저자 실행]),
  `RADEQ_SIGN_MISMATCH` 계열, R7 세대 보존 차단. **물리값·물리 가드 무접촉이 이 계약의
  전제 조건이다.**

exact-zero 행의 처분은 이 계약의 구현 귀결이다(선행 계약 문언 승계 — 별도 계약이 아니다).

## 1. 기전 특정 (오프라인 — 전 행 번호 HEAD 직독 [저자 실행])

`INDEPENDENT_SPROBE_UNDEFINED` 가 물리를 죽이는 전파 사슬:

1. **발원**: `src/lumina_plasma.c:14125-14126` —
   `if(independent_capture && (!source_defined || !isfinite(source_function))){` .
   WITNESS 인쇄 `:14127-14148` → `a210_line_saturation_blocked("INDEPENDENT_SPROBE_UNDEFINED",…)`
   `:14149-14151` → **`return-1;` `:14152`**. 함수는 `a210_line_saturation_add`(`:14032`).
   `source_defined = material->effective_integrated_opacity != 0.0` 이고 χ_eff==0 ⟺ tau==0
   정확(선행 §1-1)이므로 발화 조건 = **capture ON ∧ hazard 행**.
2. **소비자가 status 를 갈아치운다**: `:15203-15211` 호출 →
   `:15212-15215` `if(saturation_add!=0){ status=saturation_add==-2?RADEQ_NONFINITE:
   RADEQ_TERM_SCHEMA; first_bad_line=line; first_bad_shell=s; break; }` —
   진단의 반환값 하나가 물리 스캔 루프를 끊는다. 함수는
   `a210_private_line_energy_build`(`:14728`, 반환형 `RadeqStatus`).
3. **꼬리에서 이름이 바뀌어 출판이 죽는다**: `:15382-15384`(log_complete 경로 우회) →
   `:15385-15389` `UPSTREAM_LINE_SCAN_INCOMPLETE` 인쇄 → `:15394-15401`
   `[A2-10][LINE-NET-BLOCKED] status=RADEQ_TERM_SCHEMA …` → publication 미기입 →
   `:15421` `return status;`.
4. **번들이 죽고 레인이 죽는다**: 호출자 `a210_production_bundle_ledger`(`:15554`)가
   `:15597-15599` 에서 그대로 반환 → 고정-T 레인 `:15996-16000`
   `if(replay_status!=RADEQ_OK…){ a210_fixed_te_reject(ct,"RADEQ_FIXED_T_BUNDLE_BUILD_FAILED");
   goto cleanup; }`(`:15999`) → R7 물질 갱신 차단·`action=TERMINATE` → `[R7][FATAL]` rc=4 →
   model rc=1. 하류 사슬의 in-vivo 실측은 봉인 L6C stderr 그대로다(L6C 판정문 §3 —
   job 322449 사망 사슬 축자 인쇄).
5. **init 실패도 같은 계급이다**: `:14808-14811` — `a210_line_saturation_init` 가 -1 을
   반환하면 소비자가 `return RADEQ_TERM_SCHEMA;` 로 **빌드 자체를 중단**한다(발화 실적은
   없으나 같은 전파 형태).

in-vivo 확증(승계): 차단의 소유자가 진단 트랜잭션임은 **네 실측**으로 폐합돼 있다 —
S4-III(ion=2 사망) · L6(ion=3 침묵 완주) · L6C(ion=1 사망) · **census 토글 페어**(같은
봉인 base·같은 이온에서 capture 만 0 으로 → 완주, 선행 판정 §4 Z5). 이 단은 기전 특정에
런을 쓰지 않는다 — 위 사슬은 전부 소스 직독 + 봉인 실측이다.

## 2. 수리안 — A(진단 격리)의 구현 형식

### 2-1. Class 1 — 정당한 미정의 상태의 탈-실패화 (행을 잃지 않는다)

- **지점**: `:14125-14153` 의 guard + WITNESS 블록 **전체 삭제**. hazard 행은 나머지 행과
  동일하게 저장 경로로 흐른다 — 행 스키마의 기존 이름 있는 미정의 표현이 그대로 쓰인다
  (`independent_fields_defined` 계산 `:14202-14204` 불변; 선정 시 인쇄는
  `source_function=UNDEFINED_CHI_ZERO`·`S_probe=UNAVAILABLE`). **물리값·진단값 어느 쪽도
  만들어 넣지 않는다.**
- WITNESS 인쇄의 증거 역할은 Z-1r ROW(`:14075-14089`, capture·선정 무관, 행당 1줄)가 이미
  전건 대체한다 — WITNESS 를 남기면 capture=1 런에서 hazard 행당 중복 1줄(±22만 줄/반복)이
  난다. 삭제가 옳다. 두 접두 모두 `[A2-10][LINE-SATURATION-` 네임스페이스 안이라 phase
  스트림 비교기는 영향 없다 [저자 실행:
  `scripts/compare_a210_phase_baseline_streams.py` `SATURATION_PREFIX` skip 직독].
- **건수 보고 (사유를 잃지 않는다)**: 진단 구조체에 `size_t sprobe_undefined_rows` 를
  신설하고, 행 저장 시 `independent_capture && !source_defined` 이면 증가. 두 SUMMARY
  변형(레거시·per-ion-union) 말미에 ` sprobe_undefined_rows=%zu` 토큰을 덧붙인다
  (기존 토큰 뒤 추가 — census 봉인 로그와의 토큰 단위 비교 가능성 유지). capture=0 런에서는
  정의상 0 이다.
- **Z-1r ROW 의 필드 세트·서식은 동결한다** — census 기준선(봉인 442,764행)과의 bit-비교
  가능성이 이 단의 최강 무섭동 검출기다(§4 G4).

### 2-2. Class 2 — 진단 실패의 격리 (무효화하되, 조용히 통과하지 않는다)

- **구조체**: `int invalidated; const char *invalidated_reason;` 신설(정적 문자열만).
- **helper `a210_line_saturation_fail(diag, reason, phase, rc)`** 신설: 기존
  `a210_line_saturation_blocked` 인쇄(필드 세트 불변 — 사유·행수·hazard 계수 보존)를
  수행하고 `diag->invalidated=1; diag->invalidated_reason=reason; diag->active=0;` 후 rc 반환.
  init/add/log_complete/per_ion_union 의 **모든 실패 지점**(INVALID_ENV_VALUE ·
  INVALID_TARGET_ION_ENV_VALUE · MISSING_ATOMIC_IDENTITY · INVALID_ROW_PROVENANCE ·
  NONFINITE_SCALED_COMPONENT · EXPONX_FAILED · NONFINITE_DERIVED_DIAGNOSTIC · sproducer
  decode 사유들 · ROW_CAPACITY_OVERFLOW · ALLOCATION_FAILED · NONFINITE_TOTAL_EMISSION ·
  NO_POSITIVE_TARGET_EMISSION · SELECTION_INCOMPLETE · UNEXPECTED_TARGET_IDENTITY ·
  NONFINITE_ION_TOTAL_EMISSION · PER_ION_SELECTION_INCOMPLETE · UNION_SELECTION_INCOMPLETE ·
  STDERR_WRITE_FAILED)이 이 helper 를 경유한다. 사유 문자열은 전부 불변.
- **소비자 4지점의 격리**:
  1. init 소비자 `:14809-14811`: `return RADEQ_TERM_SCHEMA` **삭제** — 실패 시 비활성
     진단으로 스캔을 계속한다.
  2. 루프 소비자 `:15212-15215`: status 쓰기 + break **삭제** — add 의 반환값이 물리로
     들어가는 유일한 배선을 끊는다(호출은 유지, 반환 무소비).
  3. 꼬리 `:15382-15384`: `if(status==RADEQ_OK && !saturation_diag.invalidated)
     (void)a210_line_saturation_log_complete(…);` — log_complete 실패는 fail() 로 자기
     무효화하며 status 로 승격되지 않는다.
  4. `UPSTREAM_LINE_SCAN_INCOMPLETE` 분기(`:15385-15389`)는 **불변** — 물리가 자기 사유로
     실패했을 때 진단이 미완주를 정직 보고하는 자리다(진단 실패가 아니다).
- **격리의 가시화 (조용한 통과 금지)**: 꼬리에서 `saturation_diag.invalidated` 이면 정확
  1회 인쇄한다:
  ```
  [A2-10][LINE-SATURATION-ISOLATED] reason=<invalidated_reason> phase=<phase>
      rows_at_invalidation=<count> zero_opacity_emitting_rows=<계수>
      sprobe_undefined_rows=<계수> physics_transaction=CONTINUED complete=0
      interpretation=DIAGNOSTIC_ONLY physical_values_modified=0
      clamp=0 floor=0 cap=0 jitter=0 repair=0
  ```
  (한 줄로. 접두는 `[A2-10][LINE-SATURATION-` 네임스페이스 안 — phase 스트림 비교기 중립.)
  무효화 이후의 계수들은 **절단 부분합**이며 그렇게 읽어야 한다(선행 §3 "부분합 합산 금지"
  문구 승계 — 무효화가 계측도 함께 멈추는 것이 이 설계의 의도다: 유효성이 죽은 트랜잭션의
  계수를 계속 키우지 않는다).
- **이름 충돌 0 확인** [저자 실행: `grep -rn "LINE-SATURATION-ISOLATED" src scripts` → 0건 ·
  `grep -rn "sprobe_undefined_rows" src scripts` → 0건 · `grep -rn "A210_Z2A" src scripts`
  → 0건].

### 2-3. ★clamp/floor 금지 판별식 적용

- 이 수리는 **어떤 물리량도 읽어-고쳐 쓰지 않는다.** 제거되는 것은 진단의 status 쓰기 배선
  하나(위 소비자 4지점)와 진단 내부의 guard 블록 하나다. **신설되는 가드가 없으므로
  "정확해가 위반 가능한 가드"를 만들 여지 자체가 없다.**
- **실격 조항 명문**: 진단을 살리기 위해 S_probe 의 정의역을 넓히는 어떤 형태(χ_eff 에
  ε-floor·max(χ,ε)·조건부 대체값)도 이 단에서 실격이다. τ==0 행의 S_probe 는
  **UNDEFINED 로 남는 것이 정확한 물리**다(0/0 을 이름으로 부르는 것이지 값으로 메우는
  것이 아니다).
- 검수 word-diff 의무(§4 G7): 변경이 §2-1·§2-2 열거 지점 밖의 물리 코드에 무접촉임을
  확인한다. `physical_values_modified=0` 표지 불변.

### 2-4. 판정 — B(행 건너뛰기) 병행: **기각**

선행 판정 §5-3 이 준 입력(skip 대상 전부 (a)·수 221,382·목록 확보) 위에서 판정한다:

1. **B 가 지키려던 것은 Class 1 이 더 강하게 지킨다.** B 의 내용 = "그 행만 skip + 건너뛴
   수 보고"(NC3 정신). Class 1 은 행을 **버리지 않고** 저장·계수·보고한다 — 건수 보고
   (`sprobe_undefined_rows`)는 그대로 담기고, 행의 정보(준위쌍·방출)는 잃지 않는다.
2. **B 는 census 기준선을 파괴한다.** capture=0 경로는 이미 hazard 행을 저장·계수한다 —
   봉인 census 의 `candidate_rows=749644`·`total_scaled_emission`(bit 봉인값)이 그 증명이다.
   B 로 capture=1 팔에서만 행을 빼면 두 팔의 행 집합·선정 총계가 갈라져, 캡처 토글
   무섭동성(§4 G4 의 기반이자 선행 Z6 의 계보)이 원리적으로 깨진다.
3. **B 는 물리적으로 정당한 행을 진단 장부에서 지운다.** fa=1.0 — 이 행들은 inversion
   경계라는, 정확히 이 캠페인이 추적 중인 대상이다.

⟹ **병행하지 않는다.** C(차단 유지)는 선행 사전등록이 전면 정책으로 기각했고, (b) 행이
스캔 범위 내 0 이므로(fa=1.0) 이름 있는 거부 사유로 재등장할 대상도 현재 없다.

### 2-5. 판정 — coverage 별도 단 개설: **개설하지 않는다**

의무 조건 ZC-B 는 미발동이다(fa=1.0 — 선행 판정 §3). 그 단이 소비할 입력((b) 행 목록)이
0건이라 **열어도 측정할 것이 없다.** 스캔 범위 밖 coverage(shell≠0·타 이온·창 밖·타 위상)는
선행 판정 §9 #6 문언 그대로 **의무 없는 미결로 존속**한다 — 이 단의 범위 밖.

### 2-6. 범위 밖 — 같은 계급의 이웃 (조용한 대장 기재)

`a210_line_ion_owner_init/add`(`:13705-13777` 계열, `:14812-14820`·`:15153-15157` 소비자)도
실패 시 물리 status 를 갈아치운다 — **같은 원리 위반의 네 번째 자리 후보**다(NC3 ·
OUT_OF_GRID · SPROBE 에 이어). 이 단은 손대지 않는다: 발화 실적 0 · opt-in env 이중 게이트
(`LUMINA_RADEQ_DIAG` + `LUMINA_A210_LINE_ION_OWNER_SHELLS`) · 봉인 기준선 부재 · 변경집합
최소 원칙. **대장 기재만 하고 후속 큐에 남긴다** — 수리·확장 제안 없음.

## 3. 오프라인 검증 계획 (런 전)

런 발주 3요건의 앞 둘: ① 기전은 §1 로 오프라인 특정 완료. ② 수리안의 오프라인 검증 =
§4 의 봉인 무접촉 게이트(G1 빌드·심볼, G2 판독기 selftest, G3 스테이저 selftest)와 검수
word-diff(G7). ③ 기대치는 §8 에 사전등록. 그 뒤에만 §5.5 판정런.

## 4. 게이트 (★게이트별 음성대조·봉인 접촉 표기)

| # | 내용 | 음성대조 (주입 결함 FAIL 시연 의무) | 봉인 접촉 |
|---|---|---|---|
| **G1** 빌드·심볼 | CPU `lumina`·CUDA `lumina_cuda` 재빌드, `LINE-SATURATION-ISOLATED` 와 `sprobe_undefined_rows` 심볼이 두 바이너리에 실존(무효 재빌드 배제) | **NC-G1**: 봉인 census input 바이너리(구판)에 같은 grep → 0건이어야 함 — 검출기가 구판·신판을 실제로 가른다는 시연 | 본체 무 / NC 는 **읽기 전용 유** ⟹ 검수 답 후 실행 |
| **G2** 판독기 selftest | 신설 판독기 `--selftest` **NC-I0~I6** rc=0 (grammar-debug) | NC-I1 ISOLATED 주입→ZI-X FAIL · NC-I2 ROW 1필드 변조→ZI-B FAIL · NC-I3 SUMMARY 토큰 변조→FAIL · NC-I4 중도절단 오독→ZI-D FAIL · NC-I5 BLOCKED 주입→FAIL · NC-I6 앵커 부재→FAIL (NC-I0 = 성한 픽스처 통과) | 무 |
| **G3** 스테이저 selftest | `--selftest`(스크래치 전용): 봉인 base 하드링크 0·exports delta 정확성 | NC-S1 하드링크 주입 검출 · NC-S2 delta 초과 주입 검출 | 무 |
| **G4** 격리 본판정 (R-II) | 완주 ∧ ISOLATED 0 ∧ BLOCKED 0 ∧ SUMMARY 2줄 complete=1 ∧ **ROW 스트림 전행 bit-동일**(봉인 census 대비, 양 반복 각 221,382행) ∧ SUMMARY 토큰 일치 ∧ 선정열 (rank,line) 시퀀스 일치 ∧ 앵커 4필드 bit ∧ sprobe_undefined_rows=zero_opacity_emitting_rows=221,382 | **(in-vivo, 주입 없음)**: 봉인 L6C = 같은 env·같은 base·구판 바이너리 → 사망(322449 실측) ↔ R-II = 동일 조건·신판 바이너리 → 완주 ⟹ 완주의 소유자가 이 변경집합임이 페어로 시연됨 · **(계측)**: G2 의 NC 전종 | **유** (봉인 L6C·census 읽기 + 스테이징) |
| **G5** III Stage-4 (R-III) | 완주 ∧ SUMMARY 2줄(target_ion=2) ∧ Stage-4 row(LINE-SATURATION-ROW, producer 필드 보유) ≥ 1 ∧ **S2 판정**(기존 DET-SPROD S2 판정기 — Jbar == continuum_term + local_emission_term, tol 4e-16) ∧ ion=2 hazard census 보고 — 선행 게이트 Z3·Z2 제3원의 집행 | **NC-G5**: `producer_terms_defined=0` 행을 섞은 스크래치 로그 → S2 판정기가 UNDECIDABLE 로 fail-closed 함을 시연 | **유** |
| **G6** IV byte 불변 (R-IV) | 값 파일 Tier1 byte-동일 (봉인 L6 대비: `work/lumina_spectrum.csv`·`lumina_spectrum_comoving.csv`·`lumina_plasma_state.csv`·`physics_DET_iter0000/0001` 각 manifest.json/shell.csv/spectral.csv) ∧ SUMMARY `zero_opacity_emitting_rows=0` — 선행 게이트 Z4 의 집행. ⚠stderr 전체 byte 대조 금지(SUMMARY 토큰 추가로 자동 실패 — 선행 Z4 경고 승계·이번 단이 그 사유를 하나 더 만든다) | **NC-G6**: 스크래치 사본 1바이트 변조 → byte 비교기 FAIL 시연 (비교기에 selftest 부재 실측 [저자 실행: `--help`·소스 grep] ⟹ 주입 시연이 유일한 음성대조) | **유** (읽기 전용) |
| **G7** 격리 word-diff 검수 | 변경이 §2 열거 지점 밖 물리 코드 무접촉·물리 가드 제거 0·사유 문자열 불변 — 코드 검수(Fable)의 word-diff 전수 | 기계 NC 아님 — 검수 절차임을 정직 기재. 검수 고정질문은 §6 | 무 |

개정16 ② 준수: **검수 질문 0("이 산출물을 봉인·실물에 돌려도 되는가")의 답 전에 봉인 접촉
게이트(G1-NC·G4·G5·G6)를 돌리지 않는다.** G1 본체·G2·G3 은 병행 가능.

## 5. 기대 변경집합 (PF-1 대상 — 표 밖 파일이 표지 심볼을 얻으면 거부)

표지 심볼 = `A210_Z2A_PROBE` (런처 모드명). `src/lumina_plasma.c` 는 심볼 무보유가
정당하다(스크립트 모드명을 C 물리 소스에 넣지 않는다 — 선행 단과 같은 처분,
expected_extra 행). src-편집 태스크는 이 하나뿐이다(발주 전 가동 중 변조 태스크 0 확인은
운전석 몫).

### 변경집합 — Z-2A (표)

| 파일 | 내용 |
|---|---|
| `src/lumina_plasma.c` | §2 격리 본체: Class 1 guard+WITNESS 블록(:14125-14153) 삭제 + `sprobe_undefined_rows` 계수·SUMMARY 2변형 토큰 추가 + Class 2 `invalidated`/`fail()` 도입·소비자 4지점(:14809-14811·:15212-15215·:15382-15384) 배선 절단 + ISOLATED 1회 인쇄. **물리식·물리 가드 무접촉. Z-1r ROW 서식 동결. 사유 문자열 불변.** |
| `scripts/run_det_convergence_2026-08-08.slurm` | `A210_Z2A_PROBE` 모드 신설: iterations==2 강제 · A100 하드웨어 강제 목록 편입 · `partition==a100` 가드 · env 가드(SPRODUCER_CAPTURE==1 · INDEPENDENT_CAPTURE==1 · DIAG==2 · DIAG_TE_K==10020 · TARGET_ION==`z2a_target_ion.txt` 값 ∈ {1,2,3}) · 수용 블록(model rc=0 + SUMMARY 정확 2줄 complete=1 + `LINE-SATURATION-BLOCKED` 0건 + `LINE-SATURATION-ISOLATED` 0건 + 판독기 rc=0). **기존 모드 무변경.** |
| `scripts/stage_a210_z2a_probe.sh` | 신설 — 봉인 base 읽기 전용 복사 스테이징(하드링크 금지 — 개정16 ② 전례): ion=1·2 는 봉인 L6C base, ion=3 은 봉인 L6 base. exports 값 delta = ion 1: **정확 0** / ion 2: **정확 1**(TARGET_ION 1→2) / ion 3: **정확 0** (+경로 재배치만). `diagnostic_mode.txt`=표지 심볼 · `z2a_target_ion.txt` 기록. `--selftest` = NC-S1·S2 |
| `scripts/analyze_a210_z2a_probe.py` | 신설 — R-II: ROW 스트림 bit-대조(봉인 census stderr 인자, 읽기 전용)·SUMMARY 토큰 대조·선정열 (rank,line) 대조·앵커 4필드·ISOLATED/BLOCKED 계수·sprobe 계수 검증. R-III: SUMMARY·Stage-4 row 계수·hazard census 보고·line=262210 존부 관측. §8 분기 라벨을 **분할 밖 포함 전부 명시 부여**(미완주=ZI-D, ISOLATED=ZI-X — census 검수 Q2-1 의 UNRESOLVED 결함 재발 방지). `--selftest` = NC-I0~I6 |

### 변경집합 끝

## 5.5 런 발주 경로 — ★런 횟수: 정확 3회

선행 §5.5 를 승계하고 저자가 재확인했다:

- **a100 한정 (h200/h100 폴백 금지)** [저자 실행: 런처 `:179-189` 직독 — TARGETED/L6/L6C/
  ZO 4모드에 `*A100*) ;; *) die "A2-10 diagnostic requires A100 hardware"` (`:186-188`);
  ZO 는 추가로 `:190-192` `partition==a100` 제출 가드]. 8-19 시도가 h200 배정으로 죽은
  실사고(선행 §5.5)를 승계한다. **Z2A 모드는 §5 대로 하드웨어 강제 목록과 partition 가드에
  편입된다.** 제출은 `--partition=a100` 명시.
- `--gres=gpu:2` (two-device exact owner — 런처 `:183-184` 가드) · `--mem=64G`
  (실측 등급: census 30.7 GiB · L6 35.4 GiB · L6C 35.4 GiB) · `--time=08:00:00`
  (실측: L6 완주 04:34 · census 02:43 — 백필 자격).
- job-name = `LUMINA_Z2A_II` / `LUMINA_Z2A_III` / `LUMINA_Z2A_IV` (프로젝트 접두 의무) ·
  런루트 `/gpfs/kjhan/lumina/a210_z2a_diag_isolation_a100x2_k36/…` · `OWNER.txt`
  (project·repo·목적·`DO_NOT_CANCEL`) 기재.
- 봉인 root 를 `job.slurm` 인자로 주지 않는다(제출 스크립트가 자기 위치를 run_root 로
  앵커 — census 검수 지적 승계).
- syn101 수동 제출 금지 존속(선행 §5.5 user 지시). S4-III 의 manual 형식은 재현하지
  않는다 — §8 E-I7 의 스코프 주의 참조.

**런 3회 (전부 고정-T 레인 · iter=2 · capture 2종=1 · DIAG=2 · DIAG_TE_K=10020)**:

| 런 | TARGET_ION | base | 게이트 | 제출 시점 |
|---|---|---|---|---|
| **R-II** | 1 | 봉인 L6C (env delta 0) | G4 (격리 본판정) | 첫 제출 |
| **R-III** | 2 | 봉인 L6C (delta 1: TARGET_ION 1→2) | G5 (Z3 + Z2 제3원) | R-II 수용 확정(§5.5-1) 후 |
| **R-IV** | 3 | 봉인 L6 (env delta 0) | G6 (Z4) | R-II 수용 확정(§5.5-1) 후 (R-III 과 병행 가) |

R-II 가 죽으면 R-III/R-IV 를 제출하지 않는다(GPU 낭비 방지 — 격리 자체가 실패한 상태에서
두 런은 같은 사유로 죽거나 무의미).

### 5.5-1. 「R-II 수용 확정」의 조작화 (개정 2 — 「완주」 정의 고정, 판정 §11 #1 이관분)

구 문면 「런처 수용 PASS 후」는 수용의 최종 판정자를 런처 수용 블록의 판독기 rc
(`scripts/run_det_convergence_2026-08-08.slurm:497-502`)에 두었다. 그 판독기의 앵커 검사는
계약이 등록하지 않은 검사이며 **봉인 census 자기입력에서도 FAIL 하는 판별력 0 의 계측**으로
실증됐다(판정문 §3 — `ANCHOR_FIELD_MISMATCH`). 결함이 실증된 계측의 rc 를 제출 자격의
필요조건으로 유지하는 것은 잣대 결함을 계약 위에 두는 것이므로, 「완주」·수용을 다음으로
**고정**한다.

**정의 — R-II 수용 확정 ⟺ (i) ∧ (ii) ∧ (iii)**:

- **(i) sacct 종단 상태**: 제출 클러스터(syn)에서 `sacct -j 324557 -X -n -P -o
  JobIDRaw,State,ExitCode` 레코드 정확 1건 ∧ State ∉ {PENDING, RUNNING, REQUEUED,
  RESIZING, SUSPENDED}. **COMPLETED 를 요구하지 않는다** — State·ExitCode 원문을 기재한다.
  ⚠sacct 는 클러스터 귀속: grammar 조회는 동번호 무관 작업의 레코드를 낸다(COVER-R 판정
  F-1 실측, `docs/VERDICT_L6C_REDISPATCH_2026-08-28.md:182`).
- **(ii) 모델 완주 마커 5건 전건·정확 계수** (COVER-R §7 PR-0 ①-(ii) 와 동일 목록 —
  `docs/RUNG_L6C_REDISPATCH_2026-08-27.md:321`): `model.rc` 실재 ∧ 내용 정확 `0` ·
  `[A2-10][LINE-SATURATION-SUMMARY]` 정확 2건 ∧ 각 행 `complete=1` ·
  `event=R7_MATERIAL_PHASE_COMMITTED lane=DET iter=1 … te_generation=2->3` 정확 1건 ·
  `[PHYSICS-COMPARISON] lane=DET iter=1 status=COMMITTED` 정확 1건 · `RUN_FOOTER.txt` 실재
  ∧ 첫 행 `schema=LUMINA_DET_FLIGHT_V1`.
- **(iii) 분기 ZI-A 확정·폐합**: §8-2 판정식 문면의 ZI-A 판정 + 감리 두 겹 통과(개정17)의
  실물 — 앵커는 자유 산문 스캔이 아니라 **등록물**로 확인한다(판정문 §10 fail-open 경고
  승계): 판정문 §10 등록 라인(`docs/VERDICT_A210_Z2A_DIAG_ISOLATION_2026-08-27.md:379`)
  실재 ∧ 장부(이 문서 §10) 감리-A·감리-B 행의 최종 상태 «통과» ∧ 폐합 커밋 실재.

**fail-closed 술어 (정의가 죽은 런을 열지 못함의 보증)**: 모델 사망 런은 (ii) 의
conjunction(전건·정확 계수)을 필연 위반한다 — SUMMARY·R7 커밋·FOOTER 는 완주의 산물이다.
비종단 런(쓰기 주체 잔존 가능)은 (i) 이 차단한다. 이 계급의 기계 음성대조는 실물로
존재한다 — COVER-R 판독기 `scripts/analyze_l6c_redispatch.py` 의 **NC-R14**(비종단 주입 →
`RII_NOT_TERMINAL` FAIL; raise `:1148-1154`·selftest `:1857-1865`)·**NC-R15**(마커 결손
주입 → `RII_MODEL_INCOMPLETE` FAIL; raise `:1218-1231`·selftest `:1884-1892`) [저자 실행:
행 직독] — COVER-R 판정이 그 selftest rc=0 을 폐합 실측했다. 운전석은 R-III/IV 제출 전
같은 `--selftest` rc=0 을 §10 에 기재한다(주입 시연의 재확인 — grammar-debug).

**런처 수용 블록과의 관계**: `A210_Z2A_PROBE_ACCEPT status=PASS`(런처 `:514-516`)는 이
확정의 **충분조건이되 필요조건이 아니다**. R-II 실물에서 그 블록의 판독기 전(前) 검사
3건(SUMMARY 정확 2줄 `complete=1` `:486-490` · BLOCKED 0 `:491-493` · ISOLATED 0
`:494-496`)은 전건 통과했고 die 는 판독기 rc 단독이었다(`slurm-324557.out` 정확 1행
`DET_FLIGHT_FATAL Z2A R-II analyzer failed` [저자 실행]) — 그 FAIL 사유가 판별력 0
실증분이다. **R-III/IV 자신의 수용 블록은 그대로 둔다**: 해당 판독 경로
(`scripts/analyze_a210_z2a_probe.py:309-370` `analyze_r3_or_r4`)는 census 대조·앵커
검사를 포함하지 않는다 [저자 실행: 함수 전문 직독 — `anchor_matches` 호출은
`analyze_r2` `:291` 에만 있다] ⟹ 기지 결함이 그 두 런의 수용을 죽일 수 없다.

**후보 처분 기록**: 후보 B(구 문면 유지 — 런처 수용 블록 PASS 실물 요구) **기각** —
실물 회복 경로가 둘뿐인데 (a) R-II 재런은 판독기 수리 전이면 같은 앵커 검사(iter0 상수의
iter1 적용 — 어떤 hazard 행을 앵커로 골라도 필패, 판정문 §3-c)가 스트림 결정론(E-I3)
위에서 재발화해 **같은 die 를 재현**하며 **GPU 04:34:36 × a100 2장(≈9.2 GPU-시간)** 을
정보 0 에 소모하고, (b) 봉인 런루트에 대한 런처 재실행은 봉인 접촉이다. 후보 A(마커
단독) **기각** — 비종단 런을 차단하지 못한다(COVER-R 개정 3 이 같은 사유로 기각,
`docs/RUNG_L6C_REDISPATCH_2026-08-27.md:704-726`). 채택 = **후보 C: A 의 강화형(종단성 ∧
마커 전건) ∧ 분기 확정(iii)** — COVER-R 의 «판독 개시 자격» 조작화 (i)(ii) 위에 **제출
자격 고유의 (iii)** 을 얹는다. **선례 정합**: COVER-R 판정 §9 #3(:244-246)의 «모순 없음»
(그 단의 정의 = 판독 개시 자격 / 이 절 = R-III/IV 제출 자격) 판정과 정합하고, «sacct
COMPLETED 만이 완주» 방향이 아니므로 COVER-R §15 #9(:642-648)가 등록한 그 단의 재개정
의무는 **불발동**이다.

**적용 (개정 2 시점 실측 [저자 실행, 2026-08-28])** — 세 항 전건 성립.
**⟹ R-III·R-IV 는 이 R-II 로 열린다. 재런 불요(GPU 추가 비용 0).**

| 항 | 실측 |
|---|---|
| (i) | syntax 에서 `324557\|LUMINA_Z2A_II\|FAILED\|70:0\|04:34:36\|a100\|syn101` — 정확 1건·종단 (WorkDir=본 저장소) |
| (ii) | `<RII>` read-only: `model.rc`=`0` · SUMMARY 정확 2건 전건 `complete=1` · R7 iter=1 마커 정확 1건 · PHYSICS-COMPARISON iter=1 COMMITTED 정확 1건 · `RUN_FOOTER.txt` 첫 행 `schema=LUMINA_DET_FLIGHT_V1` |
| (iii) | 판정문 등록 라인 `:379` 실재 · 장부 §10 감리-A·B 최종 «통과» · 폐합 커밋 `4fc84fc` 실재 |

제출 행위 자체는 운전석 몫이다(분담 개정14). **판정문의 «현 상태 제출 불가»(§7)는 구
문면 하에서 옳았고 불변이다** — 이 개정이 그 «현 상태»(제출 자격의 술어)를 바꾼다.
E-I1 의 문면 FAIL 판정도 불변이다 — 이 정의는 장래 술어이지 폐합 판정의 소급 개서가
아니다.

**계측 수리 단(DET-INSTRUMENT-REPAIR)과의 관계 — 별개, 자동 해소 아님**: ① 그 수리(Z2A
판독기 앵커 결함 + sacct 클러스터 결함)가 착지해도 이 갈림은 자동 해소되지 않는다 — 구
문면의 「수용 PASS 실물」은 job 324557 에 소급 생성되지 않고(sacct 불변·봉인 런루트에
런처 재실행 불가), 갈림의 뿌리는 계측이 아니라 계약 문면의 정의 미고정이다(해소 주체 =
이 개정 — 개정 권한 전속, 개정15). ② 역 의존도 없다 — (i)(ii)(iii) 은 그 단의 산출물을
소비하지 않으며, **R-III/IV 제출은 수리 착지를 기다리지 않는다**(기지 결함이 그 수용
경로에 없음은 위 실측). ③ **조건부 접점 1건(분기 기재)**: 그 단이
`analyze_a210_z2a_probe.py` 의 R-III/IV 경로를 변경·착지한 **후에** 제출되는 R-III/IV 는
변경판 판독기로 수용된다 — 이 단의 G5·G6 판정식(§4)은 불변이므로 접점은 수용 계측의
판본뿐이고, 그 경우 운전석은 제출 전 그 단의 검수·게이트 완료를 확인한다(변조 태스크
동시 가동 금지의 상설 규약 그대로). 그 단의 계약은 여기 쓰지 않는다. 관계 판정의 현재
상태 실측 [저자 실행, 08-28 — 그 단 사전등록 초안(`docs/RUNG_INSTRUMENT_REPAIR_2026-08-28.md`,
미커밋·발주 전) 직독]: 초안 §3-A 가 R-III/R-IV 판독 경로(`analyze_r3_or_r4`) **무접촉**을
명문화했고, «E-I1 문면 FAIL·R-III/R-IV 제출 불가는 이 단이 건드리지 않는다»로 이 갈림의
해소를 자기 범위에서 명시 배제했다 — 위 ①·②·③ 판정과 양방향 합치. 그 초안이 개정돼
R-III/IV 경로에 접촉하게 되면 ③ 의 조건부 접점이 다시 산다.

## 6. 판정 절차 (분담 개정14·16·17)

- 사전등록(이 문서)=Fable / 코딩=Codex / 검수=Fable(별개 컨텍스트) / 발주·빌드·실행·제출·
  계측·대장·커밋=운전석 / 판정=Fable fresh / 감리-A=Fable fresh(판정과 다른 컨텍스트·
  고정질문 4) + 감리-B=Codex read-only(판정문 실측 주장 재현 한정). 폐합 = A·B 둘 다 통과.
- **장부 단일화(개정16 ①)**: 분장 장부 정본 = **이 문서 §10**. 판정문·감리문은 복제하지
  않고 지목한다.
- **검수 선행(개정16 ②)**: 코드 검수 고정질문 —
  0. **이 산출물을 봉인·실물에 돌려도 되는가** (이 답 전 봉인 접촉 게이트 금지)
  1. 발주서가 사전등록의 범위를 좁혔는가
  2. word-diff: 변경이 §2 열거 지점 밖 물리 코드·물리 가드에 접촉하는가 (G7)
  3. 격리가 조용한 통과를 만들 수 있는가 — 모든 실패 경로가 BLOCKED(사유)+ISOLATED(1회)로
     가시화되는가, `complete=1` 이 무효화된 진단에서 인쇄될 수 있는가
  4. Class 1 에서 물리값·진단값 대체가 하나라도 생겼는가 (§2-3 실격 조항)
- **발주 프리플라이트(개정15)**: 발주 전 `scripts/check_prereg_preflight.py`(§9 블록)·
  `scripts/check_dispatch_preflight.py` rc=0 (grammar-debug). 발주서 뒷겹은 절 번호로만
  지목한다.

## 7. 처분 원칙 / ★하지 않는 것

이 단은 **차단 제거 단**이다. 다음을 명시적으로 하지 않는다:

1. **★본선 물리 계약(A2-07 solve 의 J̄ 응답) 측정 — 아니다.** 이 단은 그 측정을 막던
   차단을 걷는 단이며, 측정 자체는 L6C-형 재발주 단의 몫이다(그 단은 별도 사전등록 +
   L6C 판정문 §8-3 의 시드 전제 교체 의무). R-II 가 부수 산출하는 ion=1 Stage-4 행들
   (S_probe·J̄ 정의 행)은 **봉인만 하고 이 단에서 어떤 물리 판정도 내리지 않는다.**
   선행 판정 §5-2 ③ 문언 그대로: 그 미결은 이 단 뒤에도 미결이다 — 바뀌는 것은 측정이
   "가능해진다"는 사실뿐이다.
2. bit-동일 구조적 생산자 귀속 — 별도 단(선행 §9 #5). §11 은 그 단으로의 승계 원장이다.
3. 스캔 스코프 확장(shell≠0·타 이온·창 밖) — §2-5.
4. ion_owner 격리 — §2-6 대장 기재만.
5. census 판독기·Z-1r 계측의 개변 — ROW 서식 동결(§2-1).
6. 발견의 처분 = 조용한 대장 기재. clamp·대체값으로 증상을 덮지 않는다(§2-3).

## 8. 기대치·분기 사전등록

### 8-1. 기대치

| # | 기대 | 등급·근거 |
|---|---|---|
| E-I1 | R-II 완주: model rc=0 ∧ SUMMARY 정확 2줄 complete=1 ∧ sacct COMPLETED | 게이트. 근거 = census 가 같은 스캔을 capture=0 으로 완주(봉인 실측) + §2 가 capture 경로의 유일 차단 지점을 제거 [실행 미검증 — GPU] |
| E-I2 | R-II `LINE-SATURATION-ISOLATED` 0건 ∧ `LINE-SATURATION-BLOCKED` 0건 | 게이트. 성한 런에서 격리는 **발동하지 않아야 한다** — 격리 발동은 그 자체로 조사 대상(ZI-X) |
| E-I3 | R-II ROW 스트림 bit-동일: 양 반복 각 221,382행, 총 442,764행 전건 (f_bit=1.0) ∧ 앵커 line=669992 4필드 bit(`tau_validity=2`·`n_upper` hex `0x1.d1534dba20c26p-12`·`A_ul` hex `0x1.0b6516044e436p-1`·`emission_per_sr` hex `0x1.3adcec9caadc0p-52`) | 게이트. 교차빌드 결정론 실적: R1/R2 exact-solve 잔차가 세 바이너리(3fc2cbb·b9a30a8·e500e29)에서 자릿수까지 동일 [저자 실행: census stderr `:12` residual=9.6662782724980344e-09 · `:423` iterations=52 residual=8.1222406993212508e-09 — S4-III 판정문 §1 표와 bit 일치] + 앵커 자체의 교차빌드 재현(선행 E-Z3) |
| E-I4 | R-II SUMMARY 토큰: `candidate_rows=749644` · `zero_opacity_emitting_rows=221382` · `selected_rows` 1201(iter0)/992(iter1) · `total_scaled_emission` bit-동일(0.544556800593765592605 / 0.633250617162941479828) · **`sprobe_undefined_rows=221382`**(양 반복) · 선정열 (rank,line) 시퀀스 동일 | 게이트. sprobe==hazard 계수의 도출: hazard 행 전건 A_ul>0 [저자 실행: 봉인 census ROW 442,764행 awk — `Aul_nonpos=0 Aul_min=1.093279e-4`] ⟹ η>0 ⟹ 전건 저장 도달. [추정 1건 명기: producer 출판이 hazard 셀에서도 decode 계약(provenance≤3·eta≥0·tau_eff=0)을 지킨다 — capture=1 hazard 셀은 in-vivo 미검증이며 위반 시 ZI-X 가 이름 있게 받는다] |
| E-I5 | R-III 완주 ∧ SUMMARY 2줄(target_ion=2) ∧ Stage-4 row ≥1 ∧ S2 PASS | 게이트 (S2 UNDECIDABLE/FAIL → ZI-P). Z3 + Z2 제3원(행 수 보고)의 집행 |
| E-I6 | R-III ion=2 hazard 총수·준위쌍 목록 | 관측(비게이트). 사전값 없음 — 10020 K 물질의 ion=2 hazard 는 미측정. ≥1 [추정: ion=1·2 가 같은 시드-물질 성질을 공유 — 빗나가도 판정 무영향, 기재만] |
| E-I7 | R-III ROW 에 `line=262210` | **관측·조건부**(비게이트). 있으면 = Z2 제3원의 전 필드 기록이 **10020 K 스코프에서** 충족 — 그렇게 한정 기재한다. 없으면 = hazard 집합의 시행-T 의존 실측이며 «262210 의 S4-III 조건(매칭-T probe) 필드 미기록» 은 **존속 미결**로 남긴다. ⚠S4-III 의 발화는 매칭-T probe 물질(`candidate_rows=51807` 시점)이었다 — 이 단은 그 조건을 재현하지 않으므로(§5.5) 두 스코프를 혼동하지 않는다 |
| E-I8 | R-IV 값 파일 Tier1 byte-동일(봉인 L6 대비, §4 G6 목록) ∧ `zero_opacity_emitting_rows=0`(양 반복) ∧ `sprobe_undefined_rows=0` | 게이트. hazard 0 인 구성에서 변경집합의 관측 가능 효과는 SUMMARY 신설 토큰 하나여야 한다 |
| E-I9 | 경과 ≈04:34 등급 · MaxRSS 31-38 GiB 등급 (각 런) | 관측(비게이트) — census/L6/L6C 실측 유추 |

### 8-2. 분기 (판별량 = f_bit ≡ R-II ROW 스트림의 census-대비 bit-일치 행 비율)

| 분기 | 판정식 | 처분 |
|---|---|---|
| **ZI-A** | R-II 완주 ∧ ISOLATED 0 ∧ f_bit ≥ 1.0 ∧ E-I4 토큰 전건 일치 | **격리 착지.** R-III·R-IV 진행, G5·G6 판정 후 폐합. 폐합 효과 = L6C-형 재발주(본선 계약 측정 단)의 전제 충족 선언 + 선행 게이트 Z2(3원 완결)·Z3·Z4 의 집행 완료 |
| **ZI-B** | R-II 완주 ∧ ISOLATED 0 ∧ (f_bit < 1.0 ∨ E-I4 토큰 불일치) | **섭동 조사** — 변경집합 또는 결정론의 결함. 물리 결론 금지·수리 발주 금지·조사 단 개설. 폐합 보류 |
| ZI-X (분할 밖) | 어느 런이든 `LINE-SATURATION-ISOLATED` ≥ 1 | 격리 기전은 작동했으나 진단이 실제로 실패 — ISOLATED 의 reason 을 이름 있게 귀속(사유·행·계수 기재) 후 재발주. 물리 완주 여부는 별도 기재. 물리 결론 금지 |
| ZI-D (분할 밖) | 어느 런이든 미완주(어떤 차단이든) | fail-closed — 차단 사유·행 기재, 이름 있는 귀속 후 재발주 (L6C D2 처분 형식 승계) |
| ZI-P (분할 밖) | R-III S2 FAIL/UNDECIDABLE ∨ R-IV 값 파일 byte 불일치 | 해당 게이트만 미착지로 폐합 보류 — R-II 의 격리 판정과 분리해 조사(교차빌드 결정론 또는 producer 계약의 별건) |

판독기는 분할 밖 분기에도 **라벨을 명시 부여**한다(§5 표 — UNRESOLVED 금지).

### 8-3. 계측 음성대조 (판독기 `--selftest`, 발주 전 grammar-debug rc=0 의무)

NC-I0 성한 픽스처 통과 / NC-I1 ISOLATED 주입→ZI-X / NC-I2 ROW 필드 변조→ZI-B /
NC-I3 SUMMARY 토큰 변조→FAIL / NC-I4 중도절단→ZI-D / NC-I5 BLOCKED 주입→FAIL /
NC-I6 앵커 부재→FAIL. (§4 G2 와 동일 목록 — 이중 기재는 의도: 게이트 표가 정본이다.)

## 9. 기계 프리플라이트 선언 (형식 = 선행 §9)

`scripts/check_prereg_preflight.py <이 문서> --root .` 이 발주 전에 검사한다(fail-closed).
PF-1 은 §5 표와 표지 심볼의 양방향 1:1 을(구현 전은 expected_extra 가 계획 경로를 공급,
커밋 후에는 grep 실물이 같은 집합 — 5번째 파일이 심볼을 얻으면 거부), PF-2 는 f_bit 1차원
분할(ZI-X·ZI-D·ZI-P 는 분할 밖 이름 있는 계급 — A207·L6C·ZO 전례 형식)을, PF-3 은 인용
앵커의 실존을 강제한다. **정직한 한계**: E-I3/I4 의 bit-재현 의미 검사와 §2 의 "물리 경로
무접촉"은 정적 검사 밖이다 — 각각 판독기 게이트(G2·G4)와 검수 word-diff(G7)가 잡는다.
신설 심볼의 사전 부재는 저자가 grep 으로 확인했다(§2-2 말미 — flags_planned 를 쓰지 않는
이유: 커밋 후 재실행 시 이 게이트가 자기 문서에 FAIL 을 내는 형식이라 배제했다).
**같은 원리의 거울상이 이 선언의 실사고가 됐다(개정 1 — §13)**: §2 가 **삭제를 요구**하는
심볼을 `flags_existing` 에 올리면, 올바른 구현이 착지한 뒤의 재실행이 정확히 같은 형식으로
FAIL 한다. 게이트는 계약 산문(§2)과 기계 선언(§9)을 교차 검사하지 않으므로 이 모순을
구조적으로 못 잡는다 — **한계 기재만 하며, 게이트 개선은 이 단의 범위 밖이다.** 삭제 대상
심볼의 실존 앵커는 상태 불변인 봉인 판정문 항목
(`docs/VERDICT_DET_L6C_COVER_2026-08-26.md` 행)이 계속 담당한다.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 — Z-2A (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "(?:scripts|src)/[a-z0-9_\\-]+\\.(?:py|sh|slurm|c)",
    "symbol": "A210_Z2A_PROBE",
    "roots": ["scripts", "src"],
    "expected_extra": ["src/lumina_plasma.c",
                       "scripts/run_det_convergence_2026-08-08.slurm",
                       "scripts/stage_a210_z2a_probe.sh",
                       "scripts/analyze_a210_z2a_probe.py"]
  },
  "branches": {
    "regimes": [[0.0, 0.1], [0.9, 1.1]],
    "metrics": {
      "f_bit": "sum(1 for x in v if x > 0.5)/len(v)"
    },
    "rules": [
      {"name": "ZI-A", "predicate": "f_bit >= 1.0"},
      {"name": "ZI-B", "predicate": "f_bit < 1.0"}
    ],
    "adversarial_fixtures": [
      {"name": "all-match", "mix": [[1.0, 221382]]},
      {"name": "single-mismatch", "mix": [[1.0, 221381], [0.0, 1]]},
      {"name": "all-mismatch", "mix": [[0.0, 442764]]},
      {"name": "anchor-only", "mix": [[1.0, 1]]}
    ]
  },
  "references": [
    {"path": "src/lumina_plasma.c",
     "flags_existing": ["LINE-SATURATION-ZERO-OPACITY-ROW",
                        "UPSTREAM_LINE_SCAN_INCOMPLETE",
                        "zero_opacity_emitting_rows"]},
    {"path": "src/lumina.h", "flags_existing": ["a210_sproducer_raw_decode"]},
    {"path": "src/opacity_publication.h", "flags_existing": ["A208_EXACT_ZERO"]},
    {"path": "scripts/run_det_convergence_2026-08-08.slurm",
     "flags_existing": ["A210_L6C_PROBE", "A210_ZO_CENSUS",
                        "requires A100 hardware", "l6c_target_ion.txt"]},
    {"path": "scripts/compare_a210_phase_baseline_streams.py",
     "flags_existing": ["SATURATION_PREFIX", "[A2-10][LINE-SATURATION-"]},
    {"path": "scripts/analyze_a210_zero_opacity_census.py",
     "flags_existing": ["NC-Z0"]},
    {"path": "scripts/a210_sproducer_ledger_check.py",
     "flags_existing": ["producer_terms_defined", "S2b"]},
    {"path": "scripts/byte_parity_compare.py",
     "flags_existing": ["--value-file"]},
    {"path": "docs/VERDICT_A210_ZERO_OPACITY_2026-08-26.md",
     "flags_existing": ["candidate_rows=749644",
                        "zero_opacity_emitting_rows=221382", "ZC-A"]},
    {"path": "docs/VERDICT_DET_L6C_COVER_2026-08-26.md",
     "flags_existing": ["INDEPENDENT_SPROBE_UNDEFINED", "candidate_rows=66110"]},
    {"path": "docs/VERDICT_DET_SPROD_S4_III_2026-08-19.md",
     "flags_existing": ["262210", "candidate_rows=51807"]},
    {"path": "docs/RUNG_A210_ZERO_OPACITY_2026-08-19.md",
     "flags_existing": ["A(진단 격리)는 지금 선택한다", "ZC-A"]}
  ]
}
```

### 9-1. 프리플라이트 자체검증·본검사 (저자 실행)

- `python3 scripts/check_prereg_preflight.py --selftest` →
  `SELFTEST_NC total=9 failed=0` rc=0 (NC-P0 포함 9종) [저자 실행, 2026-08-26].
- `python3 scripts/check_prereg_preflight.py docs/RUNG_A210_Z2A_DIAG_ISOLATION_2026-08-26.md --root .` →
  ```
  [PREREG-PREFLIGHT][PASS] reason=changeset-consistent detail=files=4
  [PREREG-PREFLIGHT][PASS] reason=branch-partition detail=trials=200000 rules=2 fixtures=4 residual=-
  [PREREG-PREFLIGHT][PASS] reason=references-resolved detail=refs=12
  [PREREG-PREFLIGHT][PASS] reason=prereg-preflight detail=RUNG_A210_Z2A_DIAG_ISOLATION_2026-08-26.md
  ```
  rc=0 [저자 실행, 2026-08-26 — 아래 §10 첫 행에도 기재. ⚠이 두 기록은 **개정 1 전 선언**
  기준(발주 전·HEAD 상태)이다 — 개정 1 후 재실행 매트릭스(HEAD·구현 양 상태)는 §13].

## 10. 집행 기록 (정본 장부 — 개정16 ①)

이 절이 분장 장부의 **유일 정본**이다. 판정문·감리문은 "장부 = 사전등록 §10" 으로 지목만
한다. 개정15 필수 4항목(발주 프리플라이트 rc · `[미실행]` 주장 실행 결과 · 분장 장부(검수
완료 후 기입) · 발주 중단 사유와 주인)을 운전석이 기입한다.

| 단계 | 규약상 담당 | 실제 | 위반 |
|---|---|---|---|
| 사전등록 | Fable | Fable fresh 컨텍스트 (2026-08-26) — 소스 행 번호·봉인 실측·이름 충돌·프리플라이트 rc=0 전건 [저자 실행], `[미실행]` 은 GPU 런 계급뿐. **결함 1건**: §9 선언이 §2-1 삭제 대상 센티널을 `flags_existing` 으로 요구(자기모순, 운전석이 구현 worktree 프리플라이트 FAIL 로 검출) → **개정 1** 로 수정(08-26, §13 — 계약 무변경·기계 선언만) · **개정 2**(08-28, §13 — 판정 §11 #1 이관분: §5.5↔§8-2 갈림 해소·「완주」 정의 고정 §5.5-1. §8-2 판정식·처분·게이트 문면 불변) | — |
| 구현 | Codex | Codex `gpt-5.6-terra` xhigh, clean worktree `8a48647` detached. 변경집합 4행 1:1, 초과·미달 0. ★**§9 자기모순을 발견하고 멈춰 보고** — 삭제한 센티널을 주석·더미로 되살려 게이트를 통과시킬 수 있었으나 거부했다(발주서 «출력을 꾸미지 마라» 발화) | — |
| 코드 검수 | Fable (별개 컨텍스트) | Fable fresh (2026-08-26). **차단 0 · 지적 4 전부 비차단**(Q0-N1 스테이저 dest 접두 가드 · Q2-N1 조기 return 시 ISOLATED 생략 · Q5-N1 NC-I1/I5 f-string 픽스처 · Q6-N1 R-III/IV 의 ZI-A 라벨). 질문 0 = **YES**. ★물리 가드가 격리로 오분류된 곳 **0**, 삭제 142행에 물리 계산·물리 가드 **0행** | — |
| 발주·집행·런 제출 | 운전석 | 발주 프리플라이트 **rc=1 → 커밋 후 rc=0**(개정15 «앞겹은 봉인된 문서» 가 미커밋 발주를 차단) → Codex 발주 → **검수 선행**(개정16 ②) → 봉인 무접촉 게이트 5건 → 개정 커밋 `0abed3b` → 구현 커밋 `406712d` → 봉인 접촉 스테이징 → 봉인 3곳 불변 실측 → R-II 제출 `324557` | — |
| 판독·계측·봉인 | 운전석 | R-II 착지(04:34:36) → 판독기 산출 판독 → **오탐 기전 실측 특정**(앵커 iter0 상수를 양 반복 적용) → 런루트 봉인(135파일 지문) | — |
| 판정 | Fable (fresh) | Fable fresh. `docs/VERDICT_A210_Z2A_DIAG_ISOLATION_2026-08-27.md`. ★**판독기 라벨 ZI-B 를 기각하고 계약 §8-2 문면으로 ZI-A 재판정** · E-I1 을 **문면 FAIL 로 정직 기재** · R-III/IV **제출 불가** 판정 · **계약 §5.5↔§8-2 갈림**과 **COVER-R PR-0 영구 불충족**을 자발 발굴 | — |
| 감리-A | Fable (fresh, 판정과 별개) | 1차 **통과**(차단 0·지적 3) — 판정식 양 항 독립 실측(ROW 442,764행·선정열 2,193쌍 `cmp`). **범위 한정 재감리 통과** — 새 논증 반례 수색 + ★**저자가 자기 논증을 저평가했음을 발견**(`e500e29` 는 실런 실적 보유) | — |
| 감리-B | Codex (read-only) | 1차 **차단 1**(#14 — 「바이너리 delta = Z2A 변경집합뿐」이 provenance 와 불일치. **핵심 논증의 전제**를 침) → **재판정 통과**. 재현 21항목 중 18 재현·2 불일치·1 재현불능(환경 제약) | — |
| 감리 반영·대장·커밋 | 운전석 | 차단을 저자에게 반송 → 논증 재구성 → 두 겹 재판정 → 반영·폐합 커밋 | — |

★**감리 두 겹의 검출이 이번에도 겹치지 않았다** — 감리-A 는 판정식을 독립 실측하고도 provenance
전제를 못 봤고, **감리-B 가 커밋 계보를 파서 잡았다**. 세 단 연속으로 계열 분리가 값을 냈다.
★그리고 **차단이 결론이 아니라 전제를 쳤다** — 저자가 전제를 사실로 교체하고 논증을 보강해
해소했고, 결론(ZI-A)은 불변이었다. 「결론을 지키려 전제를 꾸미지 마라」가 지켜졌는지는
감리-A(재)가 골대 이동 여부로 별도 판정했다(이상없음).

★**운전석 오기 1건**: 감리 발주문에 「L6C=syn104 계열」로 썼으나 실측은 **syn102** 다
(L6C=syn102 · census=syn102 · R-II=syn101). 판정문은 옳게 썼고 감리-A 가 잡았다.

## 11. ★승계 기재 (선행 판정 부록 C — 감리-A 후속 권고의 원장)

선행 감리-A 는 **생산자 귀속 단**(선행 판정 §9 #5 — bit-동일 준위쌍의 구조적 생산자)의
사전등록에 다음 둘을 승계 기재하라고 권고했다. 그 단은 아직 열리지 않았으므로 이 절이
승계 원장이다 — **이 단(Z-2A)의 판정에는 하중 0** 이며, 귀속 단 개설 시 그 사전등록으로
이관한다. 행 번호는 저자가 HEAD 에서 재확인했다 [저자 실행 — 아래 각 행].

1. **global→nlte 단사성 (별칭 배제)**: 슬롯 부여는
   `src/lumina_plasma.c:16431` `int g = nlte->nlte_ion_level_offset[i] + cursor[i]++;` —
   같은 ion 안에서 커서 단조 증가 ⟹ 서로 다른 global 준위는 서로 다른 nlte 슬롯
   (`:16434` `global_to_nlte_level[l] = g`). 소비 측은 `:8925-8926` 이 그 맵으로
   `nlte_lo/up` 을 얻고 `:8927` 이 음수(미매핑)를 fail-closed 로 걸러낸다.
   in-vivo 보강: 봉인 census 전 442,764 ROW 행에서 `lower_global==upper_global` 0건
   (선행 판정 부록 A-10b awk — 저자 재확인 생략, 판정문 실측 승계).
2. **K-FRESH 사슬 (τ 재출판 세대 보증)**: 정의 `:6893-6901`
   `tau_sobolev_require_refresh`(세대 overflow 시 abort) · `:6903-6911`
   `tau_sobolev_mark_computed`(불량 생산자 상태 abort) — NLTE 재출판이 진입에서 요구 세대를
   올리고(`:19519`) 완료에서 기록하며(`:19655`), 커밋 preflight 가 `computed==required` 를
   fail-closed 로 강제(`:20840-20847` — 불일치 시 `POP_STALE_DERIVED_TEMPERATURE` +
   `NLTE_CANDIDATE_OPACITY_FAILED` + `active=0`), 번들 preflight 동일 검사(`:21224-21226`
   return 0).

귀속 단의 입력(선행 판정 §6-4 완비분)도 여기 지목해 둔다: 준위쌍 221,382 목록
(`/gpfs/kjhan/lumina/zo_artifacts/zo_hazard_pairs_iter0.tsv`, 221,382행 [저자 실행:
`wc -l`]) · "한 세대 보존·인구는 ~1.4% 이동" 실측 · 전건 등g 관측.

## 12. 저자 실행 원장 (개정15 — 사실 주장의 명령줄 병기)

| 주장 | 명령 [저자 실행, 2026-08-26] | 결과 |
|---|---|---|
| HEAD·src 불변 | `git rev-parse HEAD` · `git diff --stat e500e29 HEAD -- src/` | `4b61915…` · 공백 |
| 차단 지점 행 번호 | `grep -n "INDEPENDENT_SPROBE_UNDEFINED" src/lumina_plasma.c` + `sed -n '14125,14153p'` | 가드 `:14125-14126`·blocked `:14149-14151`·return `:14152` |
| 소비자·꼬리·번들 행 번호 | `sed -n '15203,15215p;15380,15402p;15996,16000p'` + `grep -n "RADEQ_FIXED_T_BUNDLE_BUILD_FAILED"` | `:15212-15215`·`:15382-15401`·`:15999` |
| init 소비자 | `sed -n '14806,14812p'` | `:14809-14811` `return RADEQ_TERM_SCHEMA` |
| Z-1r 계측 위치 | `grep -n "if(tau==0.0&&n_upper>0.0){"` | `:14075` |
| 선정 가드 | `grep -n "if(!selected\[i\])continue;"` | `:14351` |
| ion_owner 소비자 | `grep -n "a210_line_ion_owner_add(\|a210_line_ion_owner_init"` + 직독 | `:14813-14820`·`:15153-15157` |
| 런처 A100·모드 구조 | `grep -n "requires A100\|partition" scripts/run_det_convergence_2026-08-08.slurm` + `sed -n '179,232p'` | `:186-188` die·`:190-192` ZO partition 가드·L6C env 가드 `:216-229`(INDEPENDENT `:221-222`) |
| phase 비교기 중립 | `sed -n '20,30p;75,82p' scripts/compare_a210_phase_baseline_streams.py` | `SATURATION_PREFIX` skip + `SATURATION_BLOCKED` 는 오류로 승격(격리 후에도 BLOCKED 는 fail-visible) |
| 봉인 census SUMMARY | `grep -n "LINE-SATURATION-SUMMARY\]" <census>/stderr.log` | `:225081`·`:449539` — candidate 749644·rows 221382·selected 1201/992·총계 §8 E-I4 값 |
| 봉인 census ROW 수·A_ul | `grep -c ROW` + awk | 442,764행·`Aul_nonpos=0 Aul_min=1.093279e-4` |
| 교차빌드 잔차 재현 | `grep -n "EXACT-MULTIGPU-EPOCH.*residual=9.6662782724980344e-09\|EXACT-MULTIGPU-EPOCH.*residual=8.1222406993212508e-09" <census>/stderr.log` | 정확히 2행: `:12`(R1 `iterations=45 residual=9.6662782724980344e-09`)·`:423`(R2 `iterations=52 residual=8.1222406993212508e-09`) — S4-III 판정문 §1 값과 bit 일치 (R6 echo `:16`·`:428` 은 EXACT-MULTIGPU-EPOCH 앵커가 배제) |
| census env | `grep -E "CAPTURE\|SATURATION\|DIAG_TE_K" <census>/input/resolved_lumina.exports` | 캡처 2종=0·DIAG=2·ION=1·TE_K=10020 |
| L6C env·base | 동일 grep + `ZO_CENSUS_STAGE_SEAL.txt` | 캡처 2종=1·ION=1; `sealed_l6c_base=…l6c_20260822T154507Z_cover` |
| L6 값 파일 실존 | `ls <L6>/work` | spectrum/comoving/plasma_state csv + physics_comparison |
| hazard 쌍 목록 | `wc -l /gpfs/kjhan/lumina/zo_artifacts/zo_hazard_pairs_iter0.tsv` | 221,382행 |
| sprod-III 봉인 실존 | `ls …/manual_20260818T122358Z_sprod_iii` | manual 형식 런루트 실재(재현 안 함 — §5.5) |
| S2 판정기·byte 비교기 | `head scripts/a210_sproducer_ledger_check.py` · `--help scripts/byte_parity_compare.py` | S2 정의 실재·비교기 selftest 부재(NC-G6 로 대체) |
| 이름 충돌 0 | `grep -rn "LINE-SATURATION-ISOLATED\|sprobe_undefined_rows\|A210_Z2A" src scripts \| wc -l` | 0·0·0 |
| 프리플라이트 | §9-1 | selftest 9종 failed=0 · 본검사 rc=0 |

`[미실행]` 잔여 = **GPU 런 3회의 결과 전부**(E-I1~I9) — 저자가 실행할 수 없는 계급.
그 외 사실 주장에 `[미실행]` 은 없다.

## 13. 개정 이력

### 개정 1 (2026-08-26, 사전등록 저자) — §9 기계 선언의 자기모순 수정 · **계약 무변경**

**결함**: §9 선언의 `references` 첫 항목이 `src/lumina_plasma.c` 에
`INDEPENDENT_SPROBE_UNDEFINED` 의 실존(`flags_existing`)을 요구했다. 그런데 계약 §2-1 은
바로 그 센티널을 담은 guard+WITNESS 블록(`:14125-14153`)의 **전체 삭제**를 요구한다 ⟹
**올바른 구현이 프리플라이트를 반드시 FAIL 시킨다.** 운전석이 구현 worktree 에서
`reference-unsupported-flag detail=src/lumina_plasma.c:INDEPENDENT_SPROBE_UNDEFINED` rc=1 로
검출했다. 실측: 구현 후 src 출현 0회(HEAD `8a48647` 에서는 1회). Codex 는 삭제한 센티널을
주석·더미 문자열로 되살려 게이트를 통과시키는 것을 **거부하고 보고**했다 — 그 판단이 옳다.

**원인 판정 = 기계 선언만의 결함 (계약 무결)**. 근거 셋: ① §2-1 의 삭제 요구는 census 토글
페어·fa=1.0·Z-1r ROW 전건 대체 실측 위에 선 판정이고 이 결함은 그 어느 근거도 건드리지
않는다. ② §9 산문 자신이 "커밋 후 재실행 시 자기 문서에 FAIL" 형식을 배제한다는 설계
원칙을 명문화했고(flags_planned 배제 사유), 이 항목은 그 원칙의 **거울상 위반**이다 — 선언이
자기 원칙을 어긴 것이지 계약이 틀린 것이 아니다. ③ 해당 문자열의 실존 앵커 역할은 상태
불변인 봉인 판정문 항목(`docs/VERDICT_DET_L6C_COVER_2026-08-26.md`, 3회 출현 [저자 실행:
`grep -c`])이 이미 담당하며 그 항목은 삭제 대상이 아니라 존치한다 — **구분 삭제**다.

**처분 (개정15 형식 — OLD/NEW 쌍 5건, 절 통째 교체 없음)**:
1-a §9 선언 블록: `flags_existing` 4항 → 3항 (`"INDEPENDENT_SPROBE_UNDEFINED",` 1행 제거;
잔존 3항 `LINE-SATURATION-ZERO-OPACITY-ROW`·`UPSTREAM_LINE_SCAN_INCOMPLETE`·
`zero_opacity_emitting_rows` 는 구현 후에도 실존 [저자 실행: worktree grep 1·1·15회] —
상태 불변) · 1-b §9 산문에 거울상 한계 기재 · 1-c §9-1 기록의 시점 한정 주석 ·
1-d §10 사전등록 행에 결함·개정 기재 · 1-e 본 절 신설. 적용 후 개정 전 사본 대비 행 단위
diff 로 의도 외 소실 0 확인 [저자 실행 — 계약·§1~§8·§11·§12 무접촉].

**개정 후 재발주 의무(개정15) = 불발동**. 근거: ① 이 개정의 delta 는 §9 기계 선언과
기재(한계·주석·장부·이력)뿐 — 계약 본문·§2 수리안·§4 게이트·§5 변경집합·§8 기대치·분기는
한 글자도 바뀌지 않았다(행 단위 diff 가 증명). ② 착지한 구현은 개정 전·후 계약에 동시
적합하다(§2-1 동일) — 아래 매트릭스 [D] 가 실측 증명. ③ §9 선언의 소비자는 구현자가 아니라
운전석의 게이트 실행이고, 발주서 뒷겹은 절 번호로만 지목하므로(개정15) 발주서 문면에도
delta 가 없다. 재발주 의무는 "발주서가 첨부한 계약이 낡는 것"의 방지 장치인데 계약 delta 가
0 이다. 단, 이 단에서 **향후 새 발주가 생기면** 그 발주서는 개정 후 문서 기준으로 처음부터
새로 쓴다 — 개정15 의 일반 의무다.

**프리플라이트 매트릭스 [저자 실행, 2026-08-26 — 게이트 = 저장소
`scripts/check_prereg_preflight.py`(worktree 사본과 diff 동일 확인), worktree 는 `--root
/tmp/claude-10396/wt_z2a` 로만 지목(무오염·무수정)]**:

| | HEAD 상태(센티널 실존) | 구현 상태(센티널 삭제) |
|---|---|---|
| 개정 전 선언 | PASS rc=0 (§9-1 기록 재현) | **FAIL rc=1** `reference-unsupported-flag` (운전석 재현과 동일) |
| 개정 후 선언 | **PASS rc=0** (refs=12) | **PASS rc=0** (refs=12) |

개정 후 선언이 양 상태에서 PASS = 선언이 상태 불변이 됐다는 실측 — 이것이 이 개정의 정당성
근거다(계약을 무르게 해서 얻은 통과가 아니라, 상태 의존 항목을 상태 불변 앵커로 옮겨 얻은
통과다).

**게이트 한계 진단 (기재만 — 개선은 이 단 범위 밖)**: 이 결함이 프리플라이트 자신에게 안
잡힌 이유 — PF-3 은 선언 각 항목을 **현재 트리 상태에 대한 독립 사실**로만 검사하며, 검사
시점(발주 전 = HEAD)에는 센티널이 실존했으므로 참이었다. 모순은 "계약 §2 가 삭제를 요구하는
심볼 ↔ §9 가 실존을 요구하는 심볼"의 **교집합**에 있는데, §2 의 삭제 요구는 한국어 산문이라
게이트가 파싱하지 않는다 — 계약과 기계 선언의 교차 검사는 구조적으로 스코프 밖이다. 이는
신설 게이트의 기지 한계("의미 검사는 못 잡는다", `project_prereg_preflight_gate.md`)의 한
사례이며, 시간 축 변형이라는 점이 새롭다: 선언이 **참인 시점과 검사되는 시점이 다르다**
(발주 전 참 → 올바른 구현 후 거짓). §9 의 flags_planned 배제 사유가 이 함정의 절반만 봤다.

### 개정 2 (2026-08-28, 사전등록 저자) — §5.5↔§8-2 갈림 해소: 「완주」·수용의 정의 고정 · **§8-2 무변경**

**경위**: 판정문(`docs/VERDICT_A210_Z2A_DIAG_ISOLATION_2026-08-27.md`, 폐합 `4fc84fc`)이
계약 내 갈림 1건을 발견으로 남겼다(§7-2·§9-4·§11 #1) — §8-2 ZI-A 처분행은 「R-III·R-IV
진행」, §5.5 는 「R-II 런처 수용 PASS 후」인데, 실물은 (모델 완주 ∧ 수용 거부)로 갈라졌다
(sacct FAILED 70:0 — die 사유는 판독기 오탐 `ANCHOR_FIELD_MISMATCH`, 봉인 census
자기입력에서도 FAIL 하는 판별력 0 실증 — 판정문 §3). 판정자는 개정 권한 전속(개정15)에
따라 해소를 이 저자에게 이관했다. **폐합 판정(분기 ZI-A·E-I 전건·확정/미결 분류)은 이
개정이 한 글자도 건드리지 않는다.**

**처분 (개정15 형식 — OLD/NEW 쌍 5건, 절 통째 교체 없음)**:

| 쌍 | 자리 | OLD → NEW (요지) |
|---|---|---|
| 2-a | §5.5 표 R-III 행 제출 시점 | «R-II 런처 수용 PASS 후» → «R-II 수용 확정(§5.5-1) 후» |
| 2-b | §5.5 표 R-IV 행 제출 시점 | 〃 (병행 가 문구 불변) |
| 2-c | §5.5 각주 직후 | §5.5-1 신설 — 「R-II 수용 확정」 조작화: (i) sacct 종단(제출 클러스터·COMPLETED 불요) ∧ (ii) 모델 완주 마커 5건 전건·정확 계수 ∧ (iii) 분기 ZI-A 확정·폐합(등록물 앵커). fail-closed 술어(NC-R14/R15 실물)·후보 A/B 기각 기록·적용 실측·계측 수리 단 관계 포함 |
| 2-d | §10 사전등록 행 | 개정 2 기재 추가 |
| 2-e | §13 | 본 절 신설 |

**정의 선택의 근거 (요지 — 전문은 §5.5-1)**: 자매 단 COVER-R 개정 3
(`docs/RUNG_L6C_REDISPATCH_2026-08-27.md:704-726`)이 같은 문제를 «sacct 종단 ∧ 모델 완주
마커 5건»으로 조작화했고 그 정의로 F1b 가 실제로 착지했다(`e11de26` — 폐합). 그 단의
정의는 «판독 개시 자격», 이 §5.5 는 «R-III/IV 제출 자격»이라 모순 없음이 그 단 판정
(`docs/VERDICT_L6C_REDISPATCH_2026-08-28.md:244-246`)으로 확정돼 있다 — 이 개정은 같은
(i)(ii) 위에 제출 자격 고유의 (iii)(분기 ZI-A 확정·폐합)을 얹어 그 판정과 정합한다.
구 문면(후보 B)을 유지하지 않는 근거: 그 문면의 최종 판정자가 판별력 0 실증 계측이고,
실물 회복은 재런(판독기 수리 전 같은 die 결정론 재현 — GPU ≈9.2 GPU-시간을 정보 0 에
소모) 또는 봉인 접촉뿐이다.

**판정 ① — R-III·R-IV 제출 가부**: §5.5-1 세 항 전건 성립 실측(저자 실행, 08-28) ⟹
**이 R-II 로 열린다. 재런 불요.** 제출은 운전석 몫이며, 제출 전 §5.5-1 이 요구하는
selftest rc 기재(NC-R14/R15 재확인) 의무가 §10 에 추가된다.

**판정 ② — DET-INSTRUMENT-REPAIR 와의 관계**: **별개 — 자동 해소 아님, 양방향 의존
없음, 조건부 접점 1건**(그 단이 Z2A 판독기 R-III/IV 경로를 변경·착지한 후의 제출은
변경판으로 수용 — G5·G6 판정식 불변). 전문은 §5.5-1 말미. 그 단의 계약은 이 문서가
쓰지 않는다.

**개정 후 재발주 의무(개정15) = 불발동**. 근거: ① 이 개정의 delta 는 제출 시점 술어와
기재뿐 — 계약 본문·§2 수리안·§4 게이트·§5 변경집합·§8-1 기대치·§8-2 분기 판정식·처분은
한 글자도 바뀌지 않았다(행 단위 diff 가 증명 — 아래). ② 코드 발주가 생기지 않는다
(변경집합 무변경·구현 0). ③ 이 단의 유일 구현 발주는 완료·폐합됐다. 향후 새 발주가
생기면 개정 후 문서 기준으로 처음부터 새로 쓴다(개정15 일반 의무).

**적용 확인 [저자 실행, 2026-08-28]**: 개정 전 판본 보존
`docs/RUNG_A210_Z2A_DIAG_ISOLATION_2026-08-26.md.rev1`(sha256 `f292cde8…` — COVER-R
판독기류의 `*.md` glob 오매칭을 피하는 확장자, 판정문 §9-7 전례). 행 단위 diff 실측:
삭제 행 **정확 3**(치환 대상 그 자체 — §5.5 표 2행 + §10 사전등록 행 1행, 각각 확장판으로
재기입) · hunk 4곳 = `244,245c`(표) · `249a`(§5.5-1 삽입) · `417c`(장부 행) · `547a`(본 절)
— **계약·§1~§4·§5 변경집합·§8·§9 선언 블록·§11·§12 무접촉, 의도 외 소실 0.**
프리플라이트 재실행: `python3 scripts/check_prereg_preflight.py
docs/RUNG_A210_Z2A_DIAG_ISOLATION_2026-08-26.md --root .` → 4항 전건 PASS
(changeset-consistent files=4 · branch-partition trials=200000 · references-resolved
refs=12 · prereg-preflight) **rc=0**.

### 10.1 집행 기록 본문 (운전석 기입, 2026-08-26)

**① 발주 프리플라이트 rc** — 1차 **rc=1** `contract-not-committed`: 사전등록이 미커밋인 채로
발주하려 했다(운전석 실수). 개정15 «앞겹 = 봉인된 문서» 규칙이 막았다. 커밋(`8a48647`) 후
재실행 **rc=0** (DP-1 `no-restatement` tokens_checked=5 · DP-2 `frontpage-sealed` ·
DP-3 `contract-preflight rc=0`). ★**이번 캠페인에서 발주 게이트가 운전석 실수를 잡은 다섯 번째.**

**② `[미실행]` 주장 실행 결과** — 저자가 `[미실행]` 을 **GPU 런 결과 계급에만** 남기고
나머지를 [저자 실행] 으로 표기했다(개정16 ③). 운전석이 §12 저자 실행 원장 **20건을 축자 재현**:
**19건 재현 · 1건 명령 결함.** 결함 = 「교차빌드 잔차 재현」의 grep 패턴이 주장한 `:12` 를 내지
못함(`:12` 의 residual 값이 다르고, `iterations=52` 대안이 후반 반복 블록까지 오염). **본문
§8 E-I3 의 주장 자체는 정확**했고, 저자가 `EXACT-MULTIGPU-EPOCH` 앵커로 정정해
**20/20 재현**(운전석 축자 재실행: 정확히 `:12`·`:423` 2행).

**③ 발주 중단** — **0회.** (Codex 의 §9 모순 보고는 중단이 아니라 **구현 완료 후 보고**다 —
발주서 §1 의 "어긋나면 멈추고 보고" 지시대로 가짜 통과를 거부한 것이다.)

**④ ★사전등록 결함 1건과 그 처분** — §9 선언이 §2-1 삭제 대상 센티널을 `flags_existing` 으로
요구 ⟹ **올바른 구현일수록 게이트가 반드시 실패**. 운전석 재현: 구현 후 src 센티널 grep **0건**
(HEAD 1건), 프리플라이트 `reference-unsupported-flag` rc=1.
★**저자와 검수가 독립 컨텍스트에서 같은 결론에 같은 근거로** 도달했다 — 원인=§9 선언만(계약
무결) · 수리 범위=선언 1항목 · **재발주 의무 불발동**(구현자가 소비한 의무 무변경 · 발주서가
절 지목뿐이라 표류할 뒷겹 없음 · 구현이 개정 전·후 문서에 동시 적합). 한쪽 판단만으로 면제하지
않았다. 개정 `0abed3b`.

프리플라이트 매트릭스 (운전석 4칸 전건 재현 — 로그인 노드. ⚠1차 시도는 문서·worktree 를
`/tmp` 에 두고 grammar-debug 로 보내 **4칸 전부 `FileNotFoundError`** 였다. `/tmp` 는 노드-로컬):

| 선언 | 본체(센티널 실존) | 구현 worktree(센티널 삭제) |
|---|---|---|
| 개정 전 | rc=0 | **rc=1** `reference-unsupported-flag` |
| 개정 후 | rc=0 | **rc=0** |

⟹ 개정 후 선언이 **양 상태 PASS = 상태 불변**. 통과는 계약을 굽혀 얻은 것이 아니라 상태 의존
항목을 상태 불변 앵커(봉인 판정문 `VERDICT_DET_L6C_COVER`, 센티널 3회 실존)로 옮겨 얻었다.
행 단위 diff: 삭제 정확 4행, 변경 hunk 전부 `:328` 이후 ⟹ **계약 §1~§8·§11·§12 무접촉**.

**⑤ 게이트 실행 결과** — 개정16 ② 에 따라 **검수 답(질문 0) 전 봉인 접촉 0회**:

| 게이트 | 봉인 접촉 | 결과 |
|---|---|---|
| CPU·CUDA 빌드 | 무 | PASS — 바이너리 > 소스 **이고 새 심볼 실존·구 센티널 소멸**(무효 재빌드 배제) |
| 사전등록 프리플라이트(구현 상태) | 무 | **rc=0** — 개정 1 의 실증 |
| `selftest_det_stage12` | 무 | PASS — NC-C1~C3, 기존 판정 불변 |
| 판독기 `--selftest` | 무 | PASS — NC-I0~I6 7종 `failed=0` |
| 스테이저 `--selftest` | 무 (스크래치) | PASS — NC-S1·S2 |
| ★**Z-1r ROW 서식 동결** | 무 | **PASS — 운전석 독립 byte 비교, diff 공백 12행.** 이 단의 게이트가 봉인 census 442,764행과 bit 대조를 하므로 서식이 깨지면 판정 자체가 불가능해진다 |
| **봉인 접촉 스테이징** | **유** | 검수 통과 후. `A210_Z2A_PROBE_STAGE_OK env_delta=0 shared_hardlinks=0` |

**⑥ ★봉인 불변 실측 — 3곳 전부** (직전 단은 1곳이었다):
L6C(119파일) `sha256 0행 · 메타(inode 포함) 0행` · L6(133파일) `0행 · 0행` ·
census(136파일) `sha256sum -c` **전건 통과**. 지문은 봉인 파일 자신을 제외해 자기참조를 배제했다
(직전 단에서 자기참조 결함을 검출·수정한 형식 승계).

**⑦ 스테이지 산출 독립 검증**: 경로 정규화 후 exports delta **0**(R-II 는 L6C 를 한 글자도 바꾸지
않는다) · 공유 inode 0 · 캡처 2종=**1**(ON) · `DIAG=2` · `TARGET_ION=1` · `TE_K=10020` ·
staged `binary.sha256` = repo `lumina_cuda`(`39d452e4…`) · staged `git_head` = repo HEAD(`406712d`).
검수 지적 Q0-N1 준수: dest 가 **봉인 3곳 모두의 밖**임을 제출 전 기계 확인.

**⑧ 런 제출** — §5.5 사양 그대로 **R-II 1회**:
`sbatch --partition=a100 --gres=gpu:2 --mem=64G --time=08:00:00 --job-name=LUMINA_Z2A_II`
→ **job 324557**. 런루트 `/gpfs/kjhan/lumina/a210_z2a_diag_isolation_a100x2_k36/rii_20260826T134337Z_ion1`, `OWNER.txt`(`DO_NOT_CANCEL=1`).
**R-III·R-IV 미제출** — 계약이 R-II 수용 PASS 후로 못박았다(격리 실패 상태에서 GPU 낭비 방지).

★**판정의 깨끗함**: R-II 는 봉인 L6C 환경을 `env_delta=0` 으로 재현하며 **달라진 것은 바이너리
하나뿐**이다. L6C 는 정확히 이 환경에서 `INDEPENDENT_SPROBE_UNDEFINED` 로 죽었다 ⟹
**완주=격리 작동, 재사망=격리 실패**. 봉인 사망↔완주 페어가 그대로 음성 대조가 된다.
