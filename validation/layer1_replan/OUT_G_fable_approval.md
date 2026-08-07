# OUT_G — Fable 최종 판정: 배선도(OUT_F) 검증 + 승인

판정자 주: OUT_F 가 근거로 삼은 인용 행 중 본 판정이 의존하는 것은 전부 발췌본
(`lumina/`, `artis/`) 원문을 직접 재열람해 대조했다 [실측] — phi_neb 폐기(2643-2650)·
Z=1(92-105)·T_inner 복제(1058-1061)·T_e_generation=0(645-655)·MC commit 순서(545→633→685-702)·
pure lane a209 부재(`lumina_cmfgen.c` 전 파일 grep 무매치)·seed 권한 폐기(radiation_field.c:604-612)·
top-ion 앵커(2676-2728, env-gated)·COEVOLVE-COLOR(lumina_cuda.cu:9392-9445)·
begin_mc의 iter+1 하드코딩(lumina_main.c:429)·생존 함수 정의행 전수(453/1933/2446/2843/7319/
8036/8127/11917/14873/15126/2404/2528/2967/3212/113-138/211-217/170-207/893) 일치 확인.
D 와 E 의 상충 지점은 E 를 기준으로 판정했다(발주 규칙).

---

## 0. 최종 판정

# **APPROVED WITH CONDITIONS** — 조건 8건 (아래 §2). 조건 충족 없이 구현 금지.

한 줄 요지: 아키텍처는 옳다 — 계약 10건 무손상, A2-10 위상 충족 불능 해소, ABSENT 7건·
끊긴 간선 6건 전부 공급자 지정, 노브 순증 0. 남은 것은 **명세 공백 1건**(부트스트랩→2-arm
세대 장부 초기화 규칙 부재 — 문면 그대로는 그림과 장부가 동시 성립 불능)과 **OUT_E 인증
사실의 미반영 2건**([COEVOLVE-COLOR] 계승 미명기·비교 provenance 필드 부재), 그리고
처분 미명기 잔재들이다. 전부 조건으로 폐합 가능하며 배선 구조 자체의 결함은 아니다.

---

## 1. 검증 항목 6개

### 1.1 완전성 — **PASS**

OUT_A §3 의 `ABSENT` 7건과 §4 의 끊긴 간선 6건 전부에 공급자가 지정되었다 [실측].

| OUT_A ABSENT | OUT_F 공급자 | 확인 |
|---|---|---|
| transported canonical J_nu view | 반복 0: S10 DET formal solve → S11 commit(r:0→1); 반복 r: 두 팔 각자 commit ([OUT_F:76-84](OUT_F_functions_and_wiring.md:76)) | [실측] `cmfgen_commit_jnu` 실재(lumina_cmfgen.c:5185-5190), `parity_field_built` provenance 무검사(lumina_plasma.c:2372-2377) |
| committed Z(T_e) | R2 bootstrap + R3 top-ion partition 수리 ([OUT_F:33-34](OUT_F_functions_and_wiring.md:33)) | [실측] 절단 지점 2643-2650 원문 확인 |
| solver-owned n_e·ion | R2(전하중성 Saha) + R4(잔차 장부화) ([OUT_F:33,35](OUT_F_functions_and_wiring.md:35)) | [실측] 5%/감쇠 지점 2745-2800 원문 확인 |
| fresh LTE level population | R2 committed ions → S8 내부 계산 ([OUT_F:66-70](OUT_F_functions_and_wiring.md:66)) | [실측] 2931-2945 경로 OUT_A §3 대조 |
| explicit NLTE level population | R6 결정론 line-J̄ + S14 배선 ([OUT_F:37](OUT_F_functions_and_wiring.md:37)) | [실측] POP_BB_STALE 지점 15168-15177 원문 확인 |
| fresh solver-owned tau | R2 사슬 → S8 tau mark(k:0→1) ([OUT_F:70-71](OUT_F_functions_and_wiring.md:70)) | [실측] require/assert 6488-6528 확인 |
| current-gen opacity/emissivity | R7 위상 수리(commit 직후·T_e 전; pure lane a209 신설) ([OUT_F:38](OUT_F_functions_and_wiring.md:38)) | [실측] 현 순서 545→633-655→685-702 원문 확인 |

끊긴 간선 6건: ①transported J→pre-loop ion = R2 가 반복 0 의 field 요구 자체를 제거,
②ion→n_e→commit = R2/R4, ③committed→fresh tau = R2 사슬, ④fresh tau→첫 수송 =
S8+E04/E06(`TRANSITION_STALE`·`MATERIAL_MANIFEST_STALE` 신설), ⑤동세대 발행→T_e = R7,
⑥qualified T_e_generation→Z = R8. 전부 처리 [실측].

추가로 OUT_C 1.2 의 잔여 비물리 2건(덱 NPY 전이확률 상시 기본=R9 로 폐합, 무연산 스텁
호출부=제거 선언 [OUT_F:43](OUT_F_functions_and_wiring.md:43))도 포섭됐다. 누락에 준하는
잔재 1건: **덱 seed n_e(`electron_densities.csv` 복사, lumina_main.c:146-149)의 처분 미명기**
— OUT_C E′ 가 지적한 T_e/n_e 격리 비일관이 배선도에서 침묵 [실측]. ABSENT/끊긴 간선
목록 밖이므로 완전성 FAIL 사유는 아니나 조건 7 로 강제한다.

### 1.2 무모순 — **UNDECIDED** (해소 조건 1)

- **A2-10 위상 문제(OUT_C 지점 G) 해소 확인 [실측]**: 배선도가 두 lane 모두
  `commit → barrier → a208+a209 → A2-10 → 물질 갱신` 순서로 재배열했고
  ([OUT_F:81-91, 110-127](OUT_F_functions_and_wiring.md:110)), a208/a209 가 읽는
  {r,m,t,k} 는 수송이 실제 소비한 M_m 과 그것이 생산한 field r 의 짝이므로 동세대
  삼중항(lumina_plasma.c:11983-11992 의 요구)이 **원리적으로 존재 가능**해진다.
  계약을 깎지 않고 위상을 맞췄다 — OUT_C 수리 순서 2번과 정합.
- 순서 사슬(수송→추정자→[barrier→발행→T_e]→Z→n_e→이온→준위→tau→전이확률)은 세대
  관계(t·r·m·k·o/e·p 장부, [OUT_F:177-186](OUT_F_functions_and_wiring.md:177))와 동시
  성립 가능 [실측: 제어흐름 도출].
- **그러나 문면 모순 1건 [실측]**: 그림은 부트스트랩에서 DET 가 r:0→1 을 commit 한 뒤
  ([OUT_F:81](OUT_F_functions_and_wiring.md:81)) 반복 r 에서 `commit[MC,r]`/`commit[DET,r]`
  를 **같은 r** 로 그리는데([OUT_F:110](OUT_F_functions_and_wiring.md:110)), 장부는 "각
  arm owner 의 requested=computed+1"([OUT_F:182](OUT_F_functions_and_wiring.md:182))
  이다. 부트스트랩 뒤 DET owner computed=1, MC owner computed=0 이므로 첫 공진화
  반복에서 DET 는 2, MC 는 1 을 요청한다 — barrier 의 "같은 논리 r" 확인과 동시 성립
  불능. 실제 begin 계약은 이 연속성을 강제한다(radiation_field.c:127-129 [실측];
  현 코드의 iter+1 하드코딩(lumina_main.c:429,545, lumina_cmfgen.c:5185)은 OUT_C
  조건 3이 이미 금지). **실현 가능한 해석은 존재한다** — `twoarm_field_epoch_begin` 이
  epoch 개시 때 MC 슬롯 baseline 을 DET committed 세대에 동기화하면 두 팔 모두
  computed+1=같은 r 이 된다 [추정] — 그러나 그 규칙이 문서에 없다. OUT_C 가 경고한
  바로 그 지점("아니면 결정론 부트스트랩 뒤 MC 첫 begin 이 즉사한다", OUT_C 2.2 조건 3)
  이므로 조건 1 로 사전등록을 강제한다.

### 1.3 계약 보존 — **PASS** (완화 0건)

| 계약 | 판정 | 근거 |
|---|---|---|
| C2-EXEC | 보존 | [실측] E01/E06; S1 생존 범위에서 `T_e=T_inner` 생성·덱 전이확률 소비를 **제외**하고 수리 대상으로 분리([OUT_F:9](OUT_F_functions_and_wiring.md:9)) — 생존 명분으로 계약을 넓히지 않았다 |
| H-TRANSFORM | 보존 | [실측] E05/E08/E19; frame 변환 단일 소유(S15, lumina_cmfgen.c:2967/3212 실재 확인) |
| GEN-GUARD | 보존+강화 | [실측] 세대 장부 6종 유지 + p(전이확률) 신설([OUT_F:186](OUT_F_functions_and_wiring.md:186)); 신설 코드 전부 fail-closed(`BOOTSTRAP_REENTRY`·`TRANSITION_STALE`·`TWOARM_*`) |
| K-SHAPE | 보존 | [실측] E01; loader 형상검사 생존(lumina_atomic.c:1152-1165 의 K-SHAPE FATAL 확인) |
| K-FRESH | 보존+강화 | [실측] tau 계약 유지(E04/E06) + 같은 인식론을 전이확률로 확장("덱 NPY 는 generation 0 seed 로도 소비 불가", [OUT_F:186](OUT_F_functions_and_wiring.md:186)) — OUT_C 1.2 비일관의 정방향 해소 |
| Z-INERT | 보존 | [실측] S7 생존(lumina_plasma.c:1933/1956); E14/E17/E18 에 명시 |
| D-BUILD | 보존 | [실측] E01/E06; compute_bf_opacity 가 실공급자(7319) |
| TE-DEAD | 보존 | [실측] seed 1회 발행+첫 commit 시 권한 폐기 유지(radiation_field.c:604-612); R1 은 기하·BC 만 입력, 복사 유래 스칼라 온도 없음 |
| A2-00 | 보존 | [실측] A2-04…10 을 하위 간선 계약으로 전부 표기([OUT_F:5](OUT_F_functions_and_wiring.md:5)); A2-10 삼중 일치 계약 내용 무손·위상만 이동 |
| CONFIG-PREC | 보존 | [실측] R1 이 CMFGEN 답·plasma_state.csv 주입을 명시 금지([OUT_F:32](OUT_F_functions_and_wiring.md:32)); 단 덱 n_e 잔재는 침묵(조건 7) |

특기: **R8 은 완화가 아니라 수리다** [실측] — 현 코드는 비적격 시 `T_e_generation=0` 으로
committed 세대 자체를 무효화(lumina_main.c:645-655, 주석 "Preserve" 와 자기모순 — OUT_C
지점 H NON-PHYSICAL). R8 은 committed (T_e,t) 를 보존하고 **다음 물질 갱신을 차단**하며
"이전 온도로 계속 진행" fallback 을 명시 금지([OUT_F:39](OUT_F_functions_and_wiring.md:39))
— 소비자는 여전히 t+1 부재로 fail-closed 다. 단 "차단"의 실행 의미론이 미정(조건 5).
반복 중 Saha fallback 금지·rate-SE 전용 계약도 그대로다([OUT_F:15](OUT_F_functions_and_wiring.md:15)).

### 1.4 클램프·노브 금지 — **PASS** (관찰 조건 4·6)

- [실측] "새 환경 노브는 하나도 추가하지 않는다" 명시([OUT_F:43](OUT_F_functions_and_wiring.md:43)); 신설 상태코드는 전부 거부 코드이지 값 제조가 아니다.
- [실측] R0 은 현 env-gated 앵커(`LUMINA_TOPSTAGE_ANCHOR`, lumina_atomic.c:2676-2683 기본 off)를 **스위치 없는 atomic-model 불변식**으로 바꾼다 — 노브 감소.
- [실측] 부트스트랩은 exactly-once + `BOOTSTRAP_REENTRY`(E03) — 선언적 1회 공급자이지 스위치가 아니다. 현 seed 발행부의 2회 호출 거부(lumina_plasma.c:6563-6569)와 동형.
- [실측] R4 "clamp/floor 금지" 명문([OUT_F:35](OUT_F_functions_and_wiring.md:35)); R6 "보간 floor 나 MC 값 대입 금지"([OUT_F:37](OUT_F_functions_and_wiring.md:37)).
- 위험 관찰 2건: ① R1 구체식 미정("확정 불가 [추정]") — 미정 식이 파라미터화되는 순간 노브가 된다(조건 4). ② 기존 노브(`LUMINA_CMF_LINERES_JBAR`(lumina_cmfgen.c:5200-5203)·`LUMINA_PURE_CMFGEN`(lumina_main.c:346)·`LUMINA_TOPSTAGE_ANCHOR`)의 처분이 배선도에 없다 — 구조화되면 제거해야지 죽은 노브로 남기면 노브 표면 동결 감사의 재발이다(조건 6).

### 1.5 2-arm 정합 — **PASS** (누락 2건, 조건 2·3 강제)

- [실측] OUT_E 핵심 인증의 반영 확인: 서두에서 "두 팔의 일치는 차분 진단일 뿐 최종 인증이
  아니다"를 명시하고 OUT_E:12-14·128-146 을 직접 인용([OUT_F:1](OUT_F_functions_and_wiring.md:1));
  R10 은 **3열 지도(J^MC·J^DET·J^CMFGEN)** 이며 "두 팔 일치를 pass/fail 물리판정으로
  승격하지 않음"을 명문화([OUT_F:41](OUT_F_functions_and_wiring.md:41)); E19 에서
  `CERT_CMFGEN_MISMATCH` 만 물리판정 실패([OUT_F:175](OUT_F_functions_and_wiring.md:175)).
  초판(1층 문서)의 "독립 잣대" 과대평가를 되풀이하지 않는다.
- [실측] 소유권 구조의 반영: DET 팔을 상태 소유자로 **선언적으로 고정**하고 MC 를 동세대
  shadow 로 — OUT_E §2 가 인증한 "state 소유권 선택형 별도 아키텍처"의 한 극한을 스위치
  없이 채택한 것으로 무모순. `Field[MC]`/`Field[DET]` 슬롯 분리로 덮어쓰기 없는 동세대
  차분([OUT_F:151](OUT_F_functions_and_wiring.md:151))은 GPU shadow 설계(lumina_cuda.cu:9140-9146)와 동형.
- **누락 1 [실측]**: R5/R10 이 전부 `[신설]` 로 표기되고 GPU coevolve lane 의 기존 동일-런
  두-장 비교(`[COEVOLVE-COLOR]` 3-셸 색/진폭 + 전 셸×빈 `lumina_coevolve_field.csv` 덤프,
  lumina_cuda.cu:9392-9445 원문 확인)가 배선도 어디에도 없다. OUT_E 1.5-#6 이 명시한
  "계승 대상"의 미반영 — R5 의 근거 인용도 "CPU 에는 한 owner/한 팔"(lumina_main.c:429-557)
  로 **CPU lane 에 한정된 사실**을 전체처럼 쓰고 있다. 조건 2.
- **누락 2 [실측]**: OUT_E 1.5-#4 의 비교 provenance(frozen-replay=수송 커널만 시험 vs
  독립 조립=조립까지 시험; paper_main.tex:872-876 교훈)가 R10/E19 스키마에 없다. 또한
  DET 소유 고정이 지불하는 대가(MC-state 인과 되먹임 상실 — OUT_E §2 CERTIFIED)와
  source 축 침묵 조건(공유 B_ν fallback 하에서는 두-장 비교가 형광 축에 맹목 —
  OUT_E 1.5-#5, lumina_cmfgen.c:1755-1765)이 범위 경계로 기재되지 않았다. 조건 3.

### 1.6 구현 가능성 — **PASS**

의존 검사 [실측]: R0(무의존)→R1(기하/BC)→R2(R0+기존 transaction)→R3(R0)→R4(R2 또는
기존 rate-SE)→R5(기존 begin/commit)→R6(기존 S10·q-set)→R7(R5·R6+기존 a208/a209/A2-10)→
R8(기존 a210_solve_transaction)→R9(기존 계산 핵심+신설 stamp)→R10(R5·R6+외부 CMFGEN).
**어느 항목도 뒤 항목에 의존하지 않는다** — 순서 적격.

관찰 2건(순서 오류 아님): ① R9 는 런타임 부트스트랩 사슬(E04, p:0→1 이 첫 수송 전)에
필요하지만 구현 순서상 7·8 뒤다 — R9 착륙 전까지는 덱 NPY 전이확률이 등록 부채로
잔존하며, R2~R8 각 단의 게이트는 pre-R9 의미론으로 정의해야 한다(전이확률 K-FRESH
폐합 주장은 R9 에서만). ② R5 barrier 의 완결 검증은 R6(DET commit 수리) 착륙 후에만
가능 — R5 단독 단은 MC-단독 barrier 로 부분 검증. [추정: 사다리 분해 도출]

---

## 2. 조건 목록 (충족 없이 구현 금지)

1. **세대 장부 초기화 규칙 사전등록** — 부트스트랩 DET commit(r=1) 뒤 두 owner 슬롯의
   세대 대응(예: `twoarm_field_epoch_begin` 이 MC 슬롯 baseline 을 DET committed 세대로
   동기화)을 §1.2 의 문면 모순이 사라지도록 명문화하고, `TWOARM_GENERATION_MISMATCH`
   음성 대조(고의 off-by-one 주입 → barrier FAIL 시연)를 함께 등록할 것. 이것 없이
   R5 구현 착수 금지. 근거: [OUT_F:81,110,182](OUT_F_functions_and_wiring.md:182),
   radiation_field.c:127-129, OUT_C 2.2 조건 3.
2. **[COEVOLVE-COLOR] 계승 명기** — R5/R10 을 신설이 아니라 GPU coevolve lane 기존
   비교 진단(`[COEVOLVE-COLOR]` + `lumina_coevolve_field.csv`, lumina_cuda.cu:9392-9445)의
   **CPU lane 이식·계승**으로 재표기하고, 셸×빈 두-장 덤프 스키마 호환을 유지할 것.
   근거: OUT_E 1.5-#6.
3. **R10 지도 스키마 보강** — 비교 provenance 필드(frozen-replay vs 독립 조립) 필수화,
   각 팔의 line-source closure(공유 B_ν fallback 여부, lumina_cmfgen.c:1755-1765) 병기,
   그리고 "DET 소유 고정 = MC-state 인과 되먹임 상실"을 범위 경계로 대장 기재.
   근거: OUT_E §2·1.5-#4·#5, paper_main.tex:872-876.
4. **R1 구체식의 선(先)등록** — 프로파일은 기하·내부 BC 만으로 닫힌 **단일 선언식**으로
   구현 전 사전등록(조정 가능 파라미터·env 스위치 금지). 운전석의 L1-1 이 현행 T_inner
   복제 seed 로 선행하는 것(O6 "seed 발행 불변")은 사다리 분해로 **허용**한다 — 근거:
   R2 사슬의 성립은 seed 의 물리성이 아니라 양·유한성에만 걸리고(OUT_C 지점 E [추정],
   publisher 가 이미 fail-closed, lumina_plasma.c:6570-6576), A′ 부채도 같은 논리로 격리
   승인됨. **단** 그 경우 R0·R1 미착륙 상태에서 CMFGEN 대조 물리 판정을 선언하지 말
   것(부트스트랩 seed 부채 잔존 상태의 스펙트럼은 판정 자격 없음), 그리고 RUNG 문서에
   배선도 순서(0→1→2)와 단 분해의 차이를 명시 개정할 것.
5. **R8 "차단" 의미론 확정** — 비적격 A2-10 시: committed (T_e,t) 보존 + 다음 물질
   갱신 차단 = **표면화된 종료/보류**(오류코드·이벤트 사전등록, 원장 보존)이며, 낡은
   M_m 으로 수송을 **조용히 속행하는 경로가 아님**을 명문화. 음성 대조: no-bracket 주입
   → 세대 0 화 없이 보존+차단이 관측되어야 함. 근거: [OUT_F:39](OUT_F_functions_and_wiring.md:39), OUT_C 지점 H.
6. **노브 회계** — R0 구조화 시 `LUMINA_TOPSTAGE_ANCHOR`, R6 정본화 시
   `LUMINA_CMF_LINERES_JBAR`, 2-arm 구조화 시 `LUMINA_PURE_CMFGEN` 상호배타 스위치의
   처분(제거 또는 명시 사유의 잔존)을 각 단의 기대 변경집합에 포함할 것. 죽은 노브
   1,384건 대장의 재발 방지. 근거: lumina_atomic.c:2676-2683, lumina_cmfgen.c:5200-5203, lumina_main.c:340-348.
7. **덱 seed n_e 처분 명기** — R2 가 전하중성 n_e 를 자가 공급하므로
   `electron_densities.csv` 복사(lumina_main.c:146-149)는 CONFIG-PREC 의 T_e 격리와
   동일 규칙으로 격리(부트스트랩 이후 소비 금지; root bracket 초기값 용도조차 쓰려면
   명시 등록). 근거: OUT_C 지점 E′ 비일관.
8. **n_e 수렴 정책 사전등록** — R4 착수 전에 TARDIS 5% 상수의 대체 기준(전하보존 잔차
   허용치)과 잔차 장부 관측 가능화(이벤트/카운터)를 등록할 것. 음성 대조: 조성 전하합
   섭동 주입 → 잔차 게이트 FAIL 시연. 근거: OUT_C 지점 C, lumina_plasma.c:2745-2800.

---

## 3. L1-1 사전등록(RUNG_L1_1_BOOTSTRAP_SUPPLIER.md) 게이트 6종 판정

**총평: 골격은 적격 — G3 은 올바른 안전대다. 보강 3건 요.**

- **G3 판정: 올바르다 [실측+판정].** "반복 ≥1 에서 장이 깨지면 여전히 fail-closed —
  LTE 로 미끄러지지 않는다"는 OUT_C 지점 B 의 판정 경계(루프 중 fail-closed 는 PHYSICAL
  계약, 비물리는 그것을 반복 0 에 과잉 적용한 것) 를 정확히 게이트화한 것이다. 이 단의
  최대 위험은 정말로 "부트스트랩 fallback 의 전역화"이며 G3 이 그 병을 직접 겨눈다.
  보강: 주입 방법을 사전등록하라 — 반복 ≥1 에서 view 를 고의 무효화(`r1_use=0` 강제)
  했을 때 기대 관측은 `POP_BF_STALE` 표면화 + transaction abort + **반복 ≥1 에 [BOOTSTRAP]
  로그 0줄**(G5 의 `BOOTSTRAP_REENTRY` 와 결합해 재진입 부재까지 이중 확인).
- G1: 적격이되 **범위를 명시하라** — "반복 1회 완주"는 iter=0 에 한한다. iter≥1 은 지점
  G·H(R7·R8 소관)로 전 구성 사망이 **기대 결과**다(OUT_C 요약 3). 이를 안 적으면 G1
  실패로 오독되거나, 반대로 L1-1 안에서 위상 수리까지 손대는 범위 초과(계약 1개=단 1개
  위반)를 유혹한다.
- G2: 적격이되 **끄는 방법이 노브여선 안 된다** — "공급자를 끄면"은 테스트 하니스 전용
  주입(빌드 변형·강제 실패 패치)이어야 하며 출하 경로에 on/off 스위치를 남기면 §1.4
  원칙 위반이다.
- G4·G5·G6: 적격 [실측] — G4/G5 는 기존 publisher 의 fail-closed·1회성(lumina_plasma.c:6563-6576)과 동형이고 G6 은 덱 고유성 배제로 옳다.
- **부족 1**: 부트스트랩 n_e 의 전하보존 잔차 관측이 없다 — O2 "n_e 수렴"만으로는 잔차
  무장부 커밋(OUT_C 지점 C)을 그대로 물려받는다. L1-1 자체에 "잔차 기록 + 사전등록
  허용치" 관측 항목을 추가하라(정책 확정은 R4 단이되, **기록**은 지금부터).
- **부족 2**: 커밋 산물의 provenance 스탬프 단언이 약하다 — O1 의 [BOOTSTRAP] 1줄 외에,
  population generation m=1 의 stamp 에 부트스트랩 provenance 가 결박되어 후속 감사가
  "이 상태는 부트스트랩 산"임을 기계 판독할 수 있어야 한다(E03 의 계약을 관측 가능하게).
- 부족 3(경미): O5 "CPU·GPU 양쪽"의 GPU 쪽은 판정런 1회 규약과 겹친다 — GPU 확인은
  §4 판정런에서만, 오프라인 단계에선 CPU 재현으로 한정함을 명시(이미 §4 가 그렇게 읽히나
  O5 문면과 어긋남).

---

## 4. 운전석이 구현 착수 전에 반드시 사전등록해야 할 기대치 (+음성 대조)

각 항목 = 단 하나의 계약. 음성 대조는 "무엇을 주입하면 FAIL 이 시연되는가".

1. **L1-1 반복 0 물질 공급자** (기등록 — §3 보강 반영해 개정)
   - 기대: 반복 0 에 [BOOTSTRAP] 1줄·m=1 commit·POP_BF_STALE 소멸(반복 0 한정)·첫 수송 도달.
   - 음성 대조: (a) 공급자 강제실패 주입 → 원래 `POP_BF_STALE` 복귀(G2), (b) 반복 ≥1
     view 무효화 → fail-closed 유지+[BOOTSTRAP] 0줄(G3), (c) seed 비유한 주입 → publisher
     거부(G4), (d) 2회 호출 → `BOOTSTRAP_REENTRY`(G5), (e) 전하합 섭동 → 잔차 기록치 초과 검출.
2. **R0 top-ion 준위 주입** — 기대: 15/74 zero-level top 이 정본 (E0,g0) 1준위를 얻고
   Z_top=g0(≠1); offset/hash/NLTE 매핑 재구축이 주입 **뒤**에 일어나며 atomic-model
   sha256 변경이 전 소비자 stamp 에 전파(변경집합에 해시 변경을 **기대치로** 등록 —
   안 하면 GEN-GUARD 가 정당한 변경을 결함으로 오인).
   - 음성 대조: 앵커 g 1건 고의 누락 → `POP_ATOMIC_MISSING` fail-closed(Z=1 로 복귀하지 않음을 확인).
3. **R1 물리 seed 프로파일** — 기대: 사전등록된 단일 선언식·provenance 스탬프·전 셸 양·유한.
   - 음성 대조: (a) 비양수 셸 주입 → publisher 거부, (b) `plasma_state.csv` t_electrons
     열에 독약값 기입 → 결과 무영향(아무도 안 읽음의 실증, CONFIG-PREC 재확인).
4. **R7 발행 위상** — 기대: 이벤트로그 순서가 반복마다 `commit(r)→view(r)→a208(o=r)→a209(e=r)→A2-10(t→t+1)→물질 갱신` 으로 실측되고 pure lane 에도 a209 가 찍힘.
   - 음성 대조: 테스트 빌드에서 a209 를 T_e 뒤로 되돌림 → `blocked_stale`/RADEQ_STALE_EMISSIVITY 발화(계약이 위상을 실제로 감시함의 시연 — 현 배선의 원리적 충족 불능이 재현되는지로 확인).
5. **R8 세대 보존** — 기대: 비적격 시 T_e_generation 이 t 로 **보존**되고(0 화 없음) 다음 물질 갱신 차단이 표면화된 오류로 기록.
   - 음성 대조: no-bracket 조건 주입(RADEQ_NO_BRACKET) → 보존+차단 관측; 0 화나 조용한 속행이 보이면 FAIL.
6. **R6 결정론 line-J̄** — 기대: DET lane 이 MC 와 동일 q_set_hash·profile_id 로 canonical line view 를 commit 하고 `nlte_solve_all` 이 DET 장으로 통과.
   - 음성 대조: q-set hash 1비트 섭동 → `LINE_JBAR_VIEW_QHASH`/`POP_QUERY_HASH_MISMATCH`; commit 생략 → 현행 `POP_BB_STALE` 재현(기준선 자체가 음성 대조).
7. **R5 barrier** — 기대: 동일 M_m 해시·같은 논리 r·같은 q/profile 에서만 통과(조건 1 의 초기화 규칙 포함).
   - 음성 대조: (a) 세대 off-by-one 주입 → `TWOARM_GENERATION_MISMATCH`, (b) 두 팔 사이 물질 1셸 변조 → `TWOARM_MATERIAL_MISMATCH`.
8. **R9 전이확률 세대화** — 기대: p 장부 신설·덱 NPY 는 p=0 으로 소비 불가·수송 전 p 최신 단언.
   - 음성 대조: publish 생략 후 수송 시도 → `TRANSITION_STALE`; 정규화 고의 파괴 → `TRANSITION_NORMALIZATION_FAILED`. 아울러 frozen-NPY vs 재계산 差 지도(셸×전이 블록)를 기대 크기 대역과 함께 등록(차이가 0 이면 그것대로 이상 — 재계산이 실제로 도는지의 효과 카운터).
9. **R10 3열 지도(잣대 자신의 음성 대조 — OUT_E 1.5-#4 그대로)** — 기대: J^MC·J^DET·J^CMFGEN 셸×대역 지도 + provenance/closure 필드.
   - 음성 대조: (a) 팔-특이 결함 주입(패킷 수 1/100) → MC-DET 差 에서 **잡힘**, (b) 공통모드 결함 주입(테스트 덱의 A_ul 1개 다중항 ×2 섭동) → MC-DET 差 는 **침묵**하고 두 팔 공동으로 CMFGEN 열에서만 이탈 — 잣대의 맹목 경계가 게이트로 확정되어야 지도 PASS.
10. **n_e 잔차 장부(R4)** — 기대: 셸별 전하보존 잔차가 이벤트로 기록되고 사전등록 허용치 이내.
    - 음성 대조: 조성 전하합 섭동 → 잔차 게이트 FAIL; 5% 상수로의 회귀(잔차 무기록 통과)가 보이면 FAIL.

---

*판정 종료. 본 문서의 승인은 OUT_F 문면 + 위 조건 8건의 합에 대한 것이다 — 조건 이행이
OUT_F 를 실질 변경하면(특히 조건 1 의 세대 규칙) 그 변경분은 배선도 개정으로 기록하고
재승인 없이 구현을 확장하지 말 것.*
