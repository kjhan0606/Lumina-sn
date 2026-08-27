# 단 사전등록 — DET-L6C-COVER-R (L6C 재발주, F1b): 교체된 인구-모델 전제 위에서의 커버 이온 J̄ 응답 판독 (GPU 런 0회) (2026-08-27)

저자 = Fable (분담 개정14 ①: 사전등록). 노선 = **user 선택 F1b**
(`docs/DESIGN_SEED_PREMISE_REPLACEMENT_2026-08-26.md` §7 주 권고 — 교체된 전제의 사전등록을
R-II 열람 전에 세우고, Z-2A ZI-A 폐합 후 봉인 R-II 산출을 등록 판독기로 판정. **GPU 런 0회**).
HEAD `d02146c` [저자 실행: `git rev-parse HEAD`], branch `thenmc-macroatom-fluorescence`.
소스 인용은 전부 HEAD 직독으로 재확인했고(§14), 인용 구간이 전처리 死블록에 들지 않음을
확인했다 [저자 실행: `lumina_plasma.c` 의 `#if 0` 블록은 1324–1693 뿐 —
`lumina_atomic.c`·`population_contract.c` 0건. 직전 단 cuda 死코드 인용 사고의 재발 방지 검사].

★**F1b 시간 규율 (이 문서의 결정적 전제)**:

- R-II(job `324557`)는 이 문서 저작 시점에 **PENDING — 산출이 존재하지 않는다**
  [저자 실행: `sacct -j 324557 -X` → `LUMINA_Z2A_II PENDING 00:00:00`].
- 저자는 R-II 런루트(`/gpfs/kjhan/lumina/a210_z2a_diag_isolation_a100x2_k36/rii_*`)를
  **일절 열지 않았다**(입력 포함). R-II 의 형상 주장은 전부 Z-2A 사전등록 집행 기록
  (`docs/RUNG_A210_Z2A_DIAG_ISOLATION_2026-08-26.md` §10.1 ⑦: env delta 0·캡처 2종=1·
  TARGET_ION=1·바이너리 `39d452e4…`)의 **인용**이다.
- 이 문서가 **커밋된 뒤에만** R-II 산출을 열 수 있다. R-II 가 먼저 착지해도 순서는 같다 —
  기대치를 데이터에 맞추는 것이 이 규율이 막는 것이다.
- 봉인 접근은 전부 read-only(census·L6·L6C — grep·스트리밍 파스만, 무거운 파스는
  grammar-debug nested ssh). 오라클(CMFGEN 자체런) 정량 인용 0 — B_ν 는 물리상수
  함수(CODATA)다(전례 절단 유지). `/tmp` 는 노드-로컬(파스는 stdin 파이프).
- 표기: `[저자 실행]` = 저자가 직접 실행(명령 §14) / `[프록시]` = 봉인 Z-1r census 의 ROW
  빌더 필드로 잰 값(등록 잣대인 SPRODUCER 캡처 필드가 아님 — §5-4 계급 명시) /
  `[추정]` = 추측 / 폐합 인용 = 폐합 판정문의 실측.

## 0. 표적 수령 — 무엇의 재발주인가

L6C 판정문 `:232-233` 이 재발주 조건을 등록했다: *"재발주 시 사전등록 §3-5 의 시드 전제를
§5 실측으로 교체할 것."* 설계 문서가 갈림길의 올바른 이름을 정정했다 — 시드(실물 생산 경로)를
바꾸는 것이 아니라 **계약의 인구-모델 전제를 실측으로 다시 쓰는 것**이다(설계 §4; F2 계열은
증거가 전멸시켰다 — §3 대조표). user 가 F1b 를 선택했다.

이 문서 저작과 설계 사이에 **신규 폐합 둘**이 끼어들었고, 이 문서는 둘을 정면으로 반영한다:

1. **Z-O 폐합**(`4b61915`): hazard 총수 221,382 · 전건 (a) `A208_EXACT_ZERO` · 전건 등g ·
   bit-lockstep 1세대 유지.
2. **★BIT-LOCKSTEP 폐합**(`4bf4a5b`, 분기 PB-A): lockstep 의 생산자 =
   `LUMINA_SUPER_CUTOFF=100` **단독 설정의 반(半)상태** — 무조건 K-덮어쓰기 lump 레이아웃
   (`lumina_atomic.c:1468-1478`) × `super_mode=0` 의 `f≡1.0`(`population_contract.c:199-200`)
   × SL-차원 solve ⟹ **NLTE solve 의 행렬 차원은 13,015 FL 이 아니라 1,173 SL** 이고, 한
   lump 안의 모든 준위가 이온·셸당 한 값이다. 같은 반상태가 봉인 L6·L6C·census exports 에
   실려 있다 [저자 실행: 세 exports 전부 `LUMINA_SUPER_CUTOFF="100"` — §14]. 설계 F1b 는 이
   폐합 **전에** 쓰였다 — §3 이 그 유효성을 재판정한다.

설계의 병행 권고 F6(생산자 귀속)은 그 사이 BITLOCK 단으로 **착지 완료**됐다 — 권고는
소진됐고, 그 산출(H-FLAT)이 이 단의 행-계급 설계(§5)의 입력이 됐다.

## 1. 계약 (하나)

> **Z-2A 가 R-II(봉인 L6C 와 env delta 0: ion=1·캡처 2종=1·iter=2)를 ZI-A 로 폐합한 뒤,
> 그 봉인 산출의 사전등록 판독 1회(GPU 런 0회)로, 반복 1 의 커버 이온(ion=1·Z∈{26,27,28})
> 생산자 선 소스가 — 행-계급 분할 아래 — 동결 클래스(d ≤ 1e-13)와 구별되게 **이 구성의 실물
> solve**(반상태 1,173-SL 차원) 산출에 응답하는지를 기계 분기로 판정한다.**

- 판별량은 L6C 등록 그대로 **SPRODUCER 캡처의 producer_eta/tau_eff**(캡처 2종=1)다 —
  §5 [프록시] 수치는 설계 입력이지 판정이 아니다.
- 수리 아님 · src/tests 접촉 0 · 물리값 무접촉 · 봉인 무변조(read-only) · 스테이징 0 ·
  빌드 0 ⟹ **V5 원상 유지, 권한 요청 없음.** 판독기는 집행 인프라이지 src 가 아니다.
- 이 단은 Z-2A 의 게이트·분기를 재판정하지 않는다 — Z-2A 판정의 몫이다. 이 단의 판독은
  Z-2A 의 E-I3/E-I4(결정론·무섭동)가 신선성 앵커를 대신 서 준 **뒤에만** 유효하다(§7 PR-0).

## 2. 교체된 전제 — OLD(축자; ★§2-5 만 요약 의역) ↔ NEW(실측 근거)

L6C 사전등록의 전제 다섯을 교체하고 하나를 신설한다. 어느 것도 판별 기계의 문턱을 건드리지
않는다(§5-3).

### 2-1. 인구 공급자 (구 §3-5 본문)

**OLD (축자)**: *"iter0 의 물질은 LTE-at-10020 [추정: 시드=LTE, FINDING 사슬]이므로 iter0
위험은 오프라인 재현 가능 — … [추정] 기전 = 에너지 축퇴 준위쌍(LTE 에서 n_l g_u = n_u g_l
정확 상쇄 — T 무관). iter1(NLTE 산출) 위험은 오프라인 예측 불능 — 분기 D2."*

**NEW**: iter0 의 물질은 **시드-물질 예측자의 NLTE solve 커밋분**(P2)이다 — 발행 사슬:
부트스트랩 LTE/Saha(P1) → init 패스 fine R1 → `lumina_init_seed_material_predictor`
(호출 `lumina_cmfgen.c:7525`·정의 `lumina_plasma.c:16060` — 무조건 1회·실패 시 종료·폴백
없음) → 커밋 gen 1→2 [저자 실행: 행 재확인; in-vivo 근거 = L6C 판정 §5-1 ⑥ stderr:419
폐합 인용]. iter1 은 iter0 solve 커밋분(P3). **그리고 그 solve 는 반상태 solve 다**:
`LUMINA_SUPER_CUTOFF="100"` 단독 설정(세 봉인 exports 실측)이 이온당 «명시 ≤100 + lump 1»
레이아웃을 만들고, `super_mode=0`(`lumina_plasma.c:16350-16351`·`:16456-16457`) ⟹
`within_sl_frac ≡ 1.0`(`population_contract.c:199-200`), solve 차원 = SL(`:19042-19043`),
재분배 `xfl[i]=b[sl]·1`(`:19397-19398`, 기록 `:19418/:19422/:19433`)이 lumped 준위 전원에
한 값을 준다 — BITLOCK PB-A 폐합 인용 + 전 행 저자 재확인 [저자 실행]. P2·P3 는 같은
재분배를 지난다(BITLOCK §4-4).

시드=LTE 였다면 hazard=0 이어야 한다(PC-2 대수, 3자 일치) — 실측 221,382(Z-O)가 그 반증을
소진했다. 커버 이온 실물 시드의 S/B 는 1 근방이 아니다: χ>0 선정 행 1,053 중
|S/B−1|≤1e-3 는 **6행(0.57%)**, 중앙 **26.14** [저자 실행 · 프록시].

### 2-2. R4/E3 잣대 앵커 (구 §5 R4 · §6 E3)

**OLD (축자)**: *"iter0 매치 행 S_prod/B(10020 K) 중앙 ∈ [0.999, 1.001] [추정 — 시드=LTE]"* ·
*"iter0 앵커 중앙 1±1e-3 [추정 — 시드 LTE]"*.

**폐기 근거**: 실측 중앙 26.14 — 4자릿수 밖 [저자 실행 · 프록시, 설계 §1-4 와 독립 파스
일치]. 문면 유지 시 재발주는 ANCHOR_FAIL→D3 로 확정 소모된다.

**NEW (대체 앵커 = 결정론 bit-앵커)**: ① Z-2A E-I3/E-I4 의 PASS(앵커 행 `line=669992`
4필드 bit · SUMMARY 토큰 · ROW 스트림 bit-동일 · 선정열 (rank,line) 시퀀스)를 판독 전제로
소비하고(§7 PR-0), ② 이 단의 판독기가 같은 앵커(4필드·SUMMARY 토큰·행 재고·계급 census)를
**거울 재검증**한다(§7 RD-1·RD-2·RD-6). iter0 S/B 분포는 **관측으로 강등**(§9 E-5) —
사전 그림은 §5-5 의 [프록시] 분위수다.

### 2-3. PC-2 발주 전제 (구 §5 PC-2)

**OLD (축자)**: *"ion=1 = 0 이 발주 전제"* (덱 × LTE(10020) 오프라인 census).

**NEW**: 오프라인 LTE census 는 커버 이온의 발주 전제가 될 수 없다 — 시드 solve 산출은 GPU
런 없이 재현 불능(L6C 판정 §5-3, 성립 증명된 한계). 대체 발주(판독) 전제 = **봉인 census
실측**(hazard 221,382 · 전건 (a) · 앵커 행 bit — Z-O 폐합) + **Z-2A ZI-A 착지**(§7 PR-0).
PC-2 자체는 비커버 이온(LTE-공급 클래스)용으로 존속한다(L6C 판정 §8-4 — ion=3 무-hazard
재현이 그 증거).

### 2-4. R3 행 전제 (구 §5 R3)

**OLD (축자)**: *"전 행 … `producer_terms/raw/independent_fields_defined` 100%,
UNAVAILABLE=0"*.

**폐기 근거**: 선정 집합에 hazard 가 실재한다 — iter0 **93** · iter1 **61** [저자 실행 ·
프록시; 설계 일치]. 문면 유지 시 기계적 FAIL.

**NEW (행-계급 분할)**: §5-1 의 4계급(+χ-부호 축) 분할. hazard(UNDEFINED_CHI_ZERO)와
NEGATIVE_CHI 는 **이름 있는 계급 — 전량 보고, 조용한 탈락 금지**("정당한 0"≠"무효" 계승).
정의성 요구는 계급별로: 비-hazard 선정 행의 producer 필드 100% defined(§7 RD-3), hazard
행은 `S_probe=UNAVAILABLE`·`independent_fields_defined=0` 이 **정당한 표현**이다(Z-2A
Class 1 계약).

### 2-5. D2 위험 모델 (구 §3-5 말미·§6 D2)

**OLD (요약 의역 — 축자 아님; 원문 = L6C 사전등록 §3-5 말미·§6 D2 행, 취지 왜곡 없음은
검수 확인 — 지적 C)**: iter0 위험은 오프라인 소거 가능, 잔여 위험은 iter1 에만 — D2 는
예정 분기.

**NEW**: 차단의 소유자 = **opt-in 진단 트랜잭션**(세 런 + 캡처 토글 페어로 폐합 — Z-O §4
Z5)이고, Z-2A 가 그 차단을 격리한다. **ZI-A 착지 후의 차단 재발은 예정 분기가 아니라 격리
실패의 반증**이다 — 이 단에는 런이 없으므로 이 전제는 판독 게이트의 형태로 나타난다:
R-II 산출에서 BLOCKED/ISOLATED 가 보이면 그것은 Z-2A 게이트가 이미 잡았어야 할 상태이고,
이 단은 X-D 로 **Z-2A 에 반송**한다(§9).

### 2-6. ★신설 전제 — 측정 대상의 차원 (BITLOCK 폐합의 직접 반영)

이 구성에서 «solve 산출»은 **1,173-SL 해**다. lumped 준위(per-ion level ≥100, 커버 3이온
기준 21,919준위)는 이온·셸당 한 값 — **준위별(FL) 응답은 lumped 구간에서 원리적으로
부재**하며, 이것은 측정 보류가 아니라 측정 대상의 부재다. 따라서 «응답»의 유일하게 정의
가능한 단위는 **SL 단위**이고, 선정 행의 응답 가능성은 행-계급(§5-1)의 순수 함수다.
이 전제의 함의 — 무엇을 재는가 — 는 §3 이 판정한다.

## 3. ★갈림길 판정 — 「무엇을 재는가」 + F1b 유효성 (BITLOCK 반영)

### 3-1. 갈림길: SL-단위 응답을 지금 재는가, 먼저 반상태를 걷는가

**판정: SL-단위 응답 측정을 선행한다. 반상태 수리는 이 단이 선점하지 않는다.** 근거 다섯:

1. **계약의 대상은 실물 생산 경로다.** 살아 있는 A2-07 solve — 봉인 사슬의 모든 런이 실제로
   실행한 그것 — 이 반상태 solve 다. 설계가 F2 계열을 전멸시킨 원리(실물 경로에서 멀어지는
   측정은 본선이 아니다)가 여기에도 적용된다: «solve 가 J̄ 에 응답하는가»의 solve 는 지금
   돌아가는 그 solve 다. 반상태 수리는 질문을 선명하게 하는 것이 아니라 **질문의 대상을
   바꾼다.**
2. **응답 가능한 자유도가 분모의 대부분이다 [저자 실행 · 프록시].** 매치 FPC 885행 중
   BOTH_EXPLICIT 284 + MIXED 506 = **790행(89.3%)** 이 반상태 아래서도 서로 다른 SL 성분을
   갖고, 실제로 **전건 d ≥ 1.823e-4** 로 움직였다(§5-5). 측정은 저출력이 아니다.
3. **구조 동결은 이름 있는 계급으로 완전 분리된다 [저자 실행 · 프록시].** 동결 행(d≤1e-13)은
   FPC 885 중 95행 — **정확히 BOTH_LUMPED 계급 그 자체**(95/95)이고, 응답 가능 계급의 동결은
   **0/790** 이다. 계급은 (line→덱 준위번호) 순수 함수라 사전 예측 가능하고, BITLOCK §6-2 의
   «잣대 오염 소지(구조로서 존재)»는 이 분할로 계량 가능한 형태가 된다 — 오염이 판별을
   흐리는 것이 아니라, 오염 자체가 이름을 갖고 따로 계수된다.
4. **수리 선행은 이 단이 결정할 수 없는 것을 전제한다.** 반상태 수리 단 개설은 user 결정
   대기(BITLOCK §8 #4)이고, src/env 접촉 + **bit-앵커 계보 전멸**(hazard 221,382·앵커 행·
   SUMMARY 토큰 전부 반상태 위의 실측 — 수리하면 Z-O→Z-2A 사슬의 검증 기준선이 사라진다)을
   수반한다. 그 비용·권한 문제는 «지금 이 봉인 데이터로 잴 수 있는 것»을 재지 않을 이유가
   못 된다.
5. **지금 재 두어야 수리 후 비교의 前항이 생긴다.** 이 단의 계급 census 와 SL-단위 응답
   실측은, 수리 단이 착지한 뒤의 재측정(FL-단위)에 대해 «수리 전후 응답 구조의 차동»이라는
   더 강한 비교를 준다. 지금 재지 않으면 그 비교가 성립하지 않는다.

**수리 선행이 옳았을 조건 (반증 가능 형태)**: 응답 가능 행이 분모의 소수였거나, 계급 분할이
불가능(순수 함수가 아님)했거나, 동결이 계급 경계를 넘나들었을 경우. 실측은 셋 다 반대다
(89.3% · 순수 함수 · 동결≡BL, EXP/MIX 동결 0) [프록시 — 등록 잣대의 확증은 이 단의 RD-2·
E-5 몫].

**주장 상한 (미리 긋는다)**: 이 단이 어느 분기로 착지해도 판정문이 주장할 수 있는 것은
«반상태 solve 의 SL-단위 응답»까지다. «NLTE 준위 구조의 응답» · lumped 구간의 준위별 물리 ·
반상태 자체의 물리적 정당성 · 응답의 정량적 옳음(오라클 INELIGIBLE)은 **주장 불가**를
사전 등록한다.

### 3-2. F1b 는 BITLOCK 폐합 후에도 유효한가

**판정: 유효 — 조건부(ZI-A) 그대로, 오히려 강화됐다.** 근거:

- **측정 매체가 불변이다.** R-II 는 봉인 L6C 와 env delta 0 — 같은 반상태를 싣고 있다
  (세 exports 실측 + Z-2A 스테이저 delta 0 기계 검증 인용). BITLOCK 은 «측정하려는 실물
  경로가 무엇인가»를 정정했지, R-II 가 그 실물 경로임을 바꾸지 않았다.
- **BITLOCK 이 죽인 것은 전제의 서술과 해석의 순진성이지 판별 기계가 아니다.** 죽은 서술
  («시드 NLTE solve» → «반상태 1,173-SL solve»)은 §2-6 이 교체했고, 죽은 해석(동결=무응답)은
  §5-1 행-계급이 교체했다. ff 문턱·판별량·매체는 불변이다.
- **F1b 의 결정적 조건(Z-2A ZI-A)이 불변이고, 시간 규율이 지금도 성립한다** — R-II PENDING
  [저자 실행]. 오히려 BITLOCK 덕에 이 사전등록은 설계 시점보다 강한 기대치(계급별 bit-급
  예측, §9)를 등록할 수 있다 — 강화의 실질이다.
- 설계 F1b 서술 중 낡은 것 하나: 병행 권고 F6 은 착지 완료(BITLOCK) — §0 에 기재했다.

## 4. 기전·표적 특정 (오프라인 — 전부 HEAD 직독 또는 폐합 인용)

1. **판정량 공급 사슬 (L6C §3-4 계승)**: iter k 의 producer_eta/tau_eff = R6 fine 이 읽은
   gen-(k+1) pre-commit 물질(iter0=시드 P2, iter1=P3).
   `d := |S₁/S₀−1| = |(η₁/η₀)/(τ₁/τ₀)−1|`(같은 ν·t — 정확 항등)가 공급 경로 판별량.
2. **표적 필터·선정층 (불변 계승)**: `a210_line_saturation_target_ion`(`lumina_plasma.c:13956`)·
   표적 검사(`:14022`) · PER_ION_UNION 90% 선정. R-II 는 TARGET_ION=1 (Z-2A 스테이징 인용).
3. **행 인쇄 실물 (HEAD 좌표)**: hazard ROW `:14090`(Z-1r, 서식 동결 — Z-2A 게이트) · 선정
   ROW `:14333`·`:14476`(두 변형) [저자 실행]. R-II 는 캡처 2종=1 이므로 선정 ROW 에 캡처
   suffix(producer 항)가 실릴 것 [추정 강 — L6(capture=1) 전례; 부재 시 §9 X-D 가 이름
   있게 받는다].
4. **Z-2A 격리의 실물 (HEAD)**: 구 센티널 `INDEPENDENT_SPROBE_UNDEFINED` 소멸(src 0건),
   `sprobe_undefined_rows` 계수(`:13914`·`:14157`·`:14413`) · `LINE-SATURATION-ISOLATED`
   (`:15330`) [저자 실행]. 성한 R-II 에서 ISOLATED=0·BLOCKED=0 이 Z-2A E-I2 게이트다.
5. **반상태 사슬 (BITLOCK 폐합 + 저자 전 행 재확인)**: §2-1 NEW 의 사슬 그대로.
   커버 3이온의 lumped 경계 = per-ion level 100 (K=100, exports 실측).
6. **비교 기준선**: 봉인 Z-1r census(`zo_20260826T050021Z_census_ion1`) — 선정 ROW 스트림
   2반복(1,201/992행)·SUMMARY 토큰·앵커 행. ZI-A 가 R-II 의 (rank,line) 시퀀스·ROW bit-동일을
   census 에 못박으므로(Z-2A E-I3/E-I4), census 위에서 계산한 계급·행 재고는 R-II 에 대한
   **bit-급 예측**이 된다(§9 E-1·E-2).

## 5. 판별 설계 — 행-계급 분할 + 판별량

### 5-1. 행-계급 (선정 행의 분할 — 순수 함수)

행의 계급은 (line_id → 덱 per-ion 준위번호, K=100) 와 τ_validity 의 순수 함수다:

| 계급 | 정의 | 반상태 아래의 성질 |
|---|---|---|
| **HAZ** | `tau_validity=2`(`A208_EXACT_ZERO`, χ=0) | BOTH_LUMPED ∩ 등g 의 표본(H-FLAT) — d **미정의**. 이름 있는 계급, 전량 보고 |
| **BL** (BOTH_LUMPED) | lower_level≥100 ∧ upper_level≥100, 비-hazard | n_l ≡ n_u(lump 단일값) ⟹ S = (2hν³/c²)/(g_u/g_l−1) — **solve 산출에 구조적으로 무감**. d 는 정의되나 ulp 급 동결이 대수 필연 |
| **MIX** | 한쪽만 ≥100 | 명시 SL × lump SL — **SL-단위 응답 가능** |
| **EXP** (BOTH_EXPLICIT) | 양쪽 <100 | 명시 준위끼리 — SL=FL, **준위별 응답 가능** |

χ-부호 축(직교): **FPC**(FINITE_POSITIVE_CHI — 양 반복 χ_eff>0) / **NEG**(χ_eff<0 —
inversion 내부, 이름 있는 계급·전량 보고·d 는 관측만). 판별 가능 분모 = 매치 ∧ FPC.

### 5-2. 판별량·분기 판별식 (불변 계승)

`d_i := |(η₁ᵢ/η₀ᵢ)/(τ₁ᵢ/τ₀ᵢ) − 1|` — **SPRODUCER 캡처 필드**(producer_eta/tau_eff)로 계산.
`ff := frac(d ≤ 1e-13)` — 분모 = **매치 FPC 행 전체**(BL 포함).
`σ := (q10(J̄₁/B) < 0.95 ? SIGNAL : WEAK-SIGNAL)` — L6C 등록 그대로.

### 5-3. ★골대 불변 선언

- **ff 문턱(C-R ≤0.10 · C-F ≥0.99)은 L6C 등록값 그대로 계승한다.** [프록시] ff=0.1073 이
  경계 바로 위임을 **안 상태에서** 등록한다 — 문턱을 움직이지 않는다(설계 §7 경고 축자 준수).
- **분모도 설계 F1 이 등록한 형태 그대로다**: 판별 가능 행 = FPC. BL 은 χ>0 이므로 분모에
  **남는다** — BL 을 빼면 [프록시] ff=0/790 으로 C-R 이 되는 것을 저자는 알고 있다. 빼지
  않는다. 그것이 골대 이동이기 때문이다. 반상태의 함의는 분모 조작이 아니라 **C-M 의 하위
  라벨**(§9 C-M-S)과 등록 관측(E-5)으로 받는다.
- 등록 잣대(캡처 필드)는 아직 존재하지 않는 데이터다(R-II PENDING) — [프록시] 를 보고 문턱
  아닌 **해석층**을 등록하는 것은 사전등록의 정의 그 자체다.

### 5-4. [프록시] 잣대의 계급 (정직 기재)

§5-5 수치는 봉인 census 의 ROW 빌더 필드(eta_per_sr·tau_effective·chi_effective·Jbar)로
계산했다 — capture=0 런이라 등록 잣대(캡처 producer 필드)가 없다. ROW 필드와 캡처 필드의
동일성은 [미확인]이며 **R-II 가 처음으로 교차검증한다**(§9 E-5 ① — 다르면 그 자체가 발견).
매치 행 = 두 반복 모두 가중 상위 90% 선정 — 생존 편향 있음. 단일 런·단일 경로.

### 5-5. ★저자 신규 실측 — 행-계급 × d 교차 census [저자 실행, 봉인 census read-only]

전 수치 [프록시]. 명령·산법 §14. 설계 §1-4 와의 관계: **885 분모 재현 완결** — 설계의
"교집합 비-hazard 885행" 은 FPC 였고, 제외된 46행은 전건 BL∩NEG 임을 이 실측이 특정했다.

| 측정 | 실측값 |
|---|---|
| 선정 행 (iter0/iter1) | **1,201 / 992** · line-id 교집합 **992**(iter1 ⊆ iter0) |
| 계급 census (iter0) | EXP **323** · MIX **621** · BL **164** · HAZ **93** |
| 계급 census (iter1) | EXP **284** · MIX **506** · BL **141** · HAZ **61** |
| hazard 인쇄 전건(154행) | **전건 (lumped,lumped) · 전건 등g** — H-FLAT 표본 문언 그대로 |
| 매치 비-hazard | **931** = FPC **885** + NEG **46**(NEG 전건 BL — χ<0 은 g_l>g_u 인 BL 의 대수 필연) |
| 매치 FPC 계급 | BL **95** · EXP **284** · MIX **506** |
| **d 교차(핵심)** | **BL: 95/95 전건 d ≤ 6.661e-16**(med 2.220e-16·exact-0 27) — **동결(d≤1e-13) ≡ BL 계급, 정확 일치.** EXP: 동결 **0**, d_min **1.823e-4**·med 2.006e-2·max 1.141e-1. MIX: 동결 **0**, d_min 1.797e-2·med 7.136e-2·max 1.528e-1 |
| ff [프록시] | **95/885 = 0.1073**(설계 0.107 재현) · 응답 가능 계급 ff_resp = **0/790** |
| 분리도 | 동결 계급 max(6.66e-16) ↔ 응답 계급 min(1.82e-4): **11.4 자릿수** — 문턱 1e-13 은 양쪽에서 150×·10⁹× 여유 |
| 부호 일치 sign(ΔS)=sign(ΔJ̄) (FPC) | **MIX 506/506 = 1.000** · EXP 260/284 = 0.915 · BL 47/70(무의미 — ulp 잡음) |
| iter0 S/B (χ>0 1,053행) | min 0.0248 · q10 0.789 · **med 26.14** · q90 1,135 · max 2.284e4 · \|S/B−1\|≤1e-3 **6행(0.57%)** |
| iter0 J̄/B | min 0.168 · **q10 0.7275** · med 1.038 · max 254.7 ⟹ σ [추정] = **SIGNAL** |
| BL∩FPC 의 τ | med 3.10e-5 · max 1.16e-4(박형) — 설계 "동결 잔여 95행" 서술과 일치 |

**함의 (프록시 한정)**: 설계 §8 #4 의 미결("동결 잔여 95행의 기전 [미확인]")은 이 census 가
**구조 귀속 후보를 특정**했다 — 동결 잔여 = BL(비등g) = 반상태의 비등g 표본추출(등g 표본은
hazard 로 갔다). hazard·동결 잔여·NEG 셋 다 같은 생산자(H-FLAT)의 세 단면이라는 그림이
계급 분할과 정확히 포개진다. **등록 잣대에서의 확증은 이 단의 몫이다**(§9 C-M-S).

## 6. 기대 변경집합 (이 목록 밖 변경 = 실패) + V5

**src/ 접촉 0 · tests/ 접촉 0 · env 접촉 0 · 스테이징 0 · 빌드 0 · 봉인 무변조 · 덱/`/gpfs`
정본 불변.** V5 원상 유지. 기존 판독기(`analyze_det_stage12_l6c.py` 등)·런처·스테이저
**무변경**.

### 변경집합 — L6C-R (표)

| 파일 | 변경 |
|---|---|
| `scripts/analyze_l6c_redispatch.py` | 신설 판독기(자립형, 표지 심볼 `A210_L6C_REDISPATCH`). 입력: 봉인 R-II 런루트·봉인 census 런루트·덱(read-only)·`--report`(봉인 밖 의무)·`--expected-head`. 기능: §7 RD-1~RD-7 전량(행 재고·계급 census(덱 조인 재계산)·캡처 정의성·재구성 항등·권한 사상·앵커 거울·capture-d 분포·ff·σ·분기 라벨)+E-5 관측 전량 보고·verdict JSON. §9 분기 라벨을 **분할 밖 포함 전부 명시 부여**(UNRESOLVED 금지 — Z-2A 판독기 전례 계승). `--selftest` = NC-R0~R12(전부 합성/스크래치 픽스처 — 봉인 불접촉; R10 = PR-0 판정문-부재 · R11·R12 = 실패-보고 봉인-내부 쓰기 가드 — §16 개정 2) |

### 변경집합 끝

- 패턴 밖 산출: 본 문서, `validation/det_stage12/l6c_redispatch/`(판독 보고·verdict JSON),
  판정문. 커밋 규율: 사전등록 커밋(본 문서, **R-II 열람 전**) → 도구 커밋 1개 → (Z-2A ZI-A
  폐합 확인) → 판독 → 판정문 커밋.
- PF-1 충족 방법: 심볼 보유 파일 = 신설 1파일뿐(구현 전엔 expected_extra 가 계획 경로 공급,
  구현 후엔 grep 실물이 같은 집합 — 모든 커밋 경계에서 green). 코드 트리(PF-1 roots 계열)
  심볼 선재 0 [저자 실행: `grep -rn A210_L6C_REDISPATCH src scripts` → 0. 범위에 `docs` 를
  넣지 않는다 — 본 문서가 `docs/` 에 놓이는 순간 자기 자신을 잡아 축자 재현이 깨진다
  (검수 지적 A, §16 개정 1)].

## 7. 게이트 (각 행: 요구 / ★음성대조 / 봉인 접촉)

### 전제·오프라인 (판독 전)

| # | 요구 (기계 판정식) | ★음성대조 | 봉인 접촉 |
|---|---|---|---|
| **PR-0 판독 전제** | ① R-II 착지(sacct COMPLETED) ② **Z-2A 판정이 R-II 를 ZI-A 로 선언하고 그 판정이 감리 A·B 를 통과**(G4 = E-I3/E-I4 PASS 문면 실재 — 판독기가 판정문 텍스트에서 `ZI-A`·`E-I3`·`E-I4` PASS 앵커를 확인, 부재 시 rc≠0) | **NC-R10**(판정문-부재 합성 픽스처): Z-2A 판정문이 없는 스크래치 문서 집합 위에서 게이트가 이름 있는 FAIL(`Z2A_VERDICT_ABSENT`) — **판독 전** 시연 의무. 게이트는 신설 판독기 안에 살므로(발주 산물) "발주 전" 시연은 성립 불능이었고, 합성 픽스처는 Z-2A 판정문이 판독기 완성 전에 착지해도 시연 창을 보존한다(검수 지적 B — 두 처방 모두 채택) | 무 (repo 문서·sacct 만) |
| **PR-1 판독기 selftest** | `--selftest` rc=0, NC-R0~R12 전종 발화·복원 | NC 목록 자체(아래) | 무 (스크래치) |
| **PR-2 프리플라이트** | `check_prereg_preflight.py <본 문서> --root .` rc=0 · 발주 시 `check_dispatch_preflight.py` rc=0 | 게이트 자신의 `--selftest`(NC 9종/8종) | 무 |

★**PR-0 문면 매칭 — 판정 라인에 대한 형식 요구 (개정 2 명시; 게이트 판정식 무변경)**: PR-0 의
앵커 확인은 **문면(리터럴) 매칭**이다(1차 검수 잔여 지적 — 수리 델타 밖 무변경 구간). 따라서
**Z-2A 판정문(과 그 감리 기재)은 `ZI-A`·`E-I3 PASS`·`E-I4 PASS`·`감리 A·B 통과` 앵커를 문면
그대로 실어야 한다** — 싣지 않으면 **성한 판정문이 X-P 로 오차단**된다(fail-closed 방향의
오류라 봉인 위험은 없으나 F1b 를 헛되이 소멸시킨다). 이것은 이 단 게이트의 변경이 아니라
Z-2A 판정 라인(Fable fresh 저작·운전석 기재)에 대한 요구다.

### 판독 (봉인 read-only — ★검수 Q0 답 + PR-0 통과 후에만)

| # | 요구 (기계 판정식) | ★음성대조 | 봉인 접촉 |
|---|---|---|---|
| **RD-1 행 재고** | R-II: SUMMARY 2줄 complete=1 · `R7_MATERIAL_PHASE_COMMITTED lane=DET iter=1 … te_generation=2->3` 정확 1건 · BLOCKED 0 · ISOLATED 0 · 선정 행 1,201/992 · (rank,line) 시퀀스 = census [E-1] | **NC-R4**: 픽스처에서 R7 마커 삭제 → FAIL(침묵 금지) / BLOCKED 1줄 주입 → X-D 라벨 | 유 (읽기) |
| **RD-2 계급 census** | 덱 조인 재계산으로 계급 계수 = §5-5 예측(E-2) — 순수 함수의 재현 | **NC-R9**: 덱 사본의 준위번호 1건 변조 → 계급 계수 불일치 FAIL | 유 (읽기) |
| **RD-3 캡처 정의성** | 비-hazard 선정 행: producer 필드 100% defined·finite. hazard 행: 이름 있는 계급으로 전량 계수(탈락 0) | **NC-R4b**: hazard 행 1건을 픽스처에서 무단 삭제 → 재고 불일치 FAIL | 유 |
| **RD-4 재구성 항등** (R4b 계승) | 전 FPC 행: β_rec=exponx(τ_eff^prod) 재구성으로 캡처 두 항·Jbar 상대편차 ≤1e-12 · 뺄셈·역산 0회 | **NC-R5**: η 1e-9 섭동 → FAIL | 유 |
| **RD-5 권한 사상** (R5 계승) | f_mapped = 1.0 정확(양 반복 전 행, 발주 HEAD blob 테이블) | **NC-R6**: 테이블 사본에서 (26,1) 제거 → f_mapped<1 검출 | 유 |
| **RD-6 결정론 앵커 거울** | 앵커 행 `line=669992` 4필드 bit(Z-2A E-I3 값) ∧ SUMMARY 토큰(E-I4 값: candidate_rows=749644·hazard 221382·sprobe_undefined_rows=221382) | **NC-R7**: 앵커 1필드 변조 → FAIL | 유 |
| **RD-7 판정 (계약 본체)** | 매치 FPC 행 capture-d 분포 → ff → §9 분기 (판정식이 RD-1~RD-6 PASS 를 명시 conjunct 로 포함 — L6C 감리 권고 (a) 계승) | **NC-R1** 동결 픽스처 → C-F / **NC-R2** 응답 픽스처 → C-R / **NC-R3** 프록시-착지 픽스처(§5-5 조성) → C-M + **C-M-S 라벨** / **NC-R8** 비표적 행 주입 → W-V (NC-R0 = 성한 픽스처 통과) | 유 |

clamp/floor/cap 0. 계급 분류는 물리값을 바꾸지 않는 분석 분류 — 전량 보고. 음성대조는 전부
스크래치 사본, 봉인 쓰기 접촉 0. 판독기 `--report` 기본값은 봉인 밖(검수 조건 전례).

## 8. 런 발주 경로 — ★GPU 런 0회

- **GPU 런 0회. 스테이징 0회. 빌드 0회.** 측정 매체 = Z-2A R-II 의 봉인 산출(F1b). 근거:
  R-II 는 봉인 L6C 와 env delta 0 이라 판정런과 형상이 같고, Z-2A 사전등록 §7-1 이 그
  데이터의 물리 판정을 이 단 몫으로 예약했다. 새 런이 필요해지는 유일한 경우 = X-P(§9)이며
  그때는 **F1 로 회귀하되 별도 사전등록·별도 발주**다(이 문서는 런을 발주하지 않는다).
- Codex 발주 **1회**(판독기 1파일 코딩). 발주 전: 본 문서 커밋 + 프리플라이트 2종 rc=0
  (grammar-debug). 발주서 앞겹 = 본 문서 원문 그대로, 뒷겹은 절 번호 지목만(개정15).
- 오프라인 실행 전부 grammar-debug(nested ssh, stdin 파이프 — `/tmp` 노드-로컬). 로그인
  노드 연산 금지. 판독 실행 = 운전석, 판정 = Fable fresh 재측정.

## 9. 기대치·분기 사전등록

### 9-1. 기대치 (등급 명시; bit-급은 전부 **ZI-A 조건부**)

| # | 등급 | 기대 |
|---|---|---|
| **E-1** | 게이트 | 행 재고 = §5-5: 선정 1,201/992 · 교집합 992 · hazard 93/61 · (rank,line) = census 시퀀스 · R7 커밋 마커 1건(gen 2→3) |
| **E-2** | 게이트 | 계급 census = §5-5: iter0 {EXP 323·MIX 621·BL 164·HAZ 93} · iter1 {EXP 284·MIX 506·BL 141·HAZ 61} · 매치 FPC {BL 95·EXP 284·MIX 506} · NEG 46(전건 BL) |
| **E-3** | 게이트 | RD-4 ≤1e-12(전 FPC 행) · RD-5 f_mapped=1.0 · RD-6 앵커 bit |
| **E-4** | **계약 본체** | 1순위 [추정 강] **C-M, ff ≈ 0.107 등급** — 구조 근거: BL∩FPC 95행의 d 는 반상태 대수로 ulp 급 강제(§5-1), 응답 계급 프록시 동결 0. **하위 라벨 [추정] C-M-S**(동결 ≡ BL). C-R(캡처가 프록시와 다르면)·C-F 도 등록 분기로 존속 — 사전 확률 수치는 부여하지 않는다(드라마화 금지). ★C-M 은 이 계약에서 실패가 아니라 등록된 착지다 |
| **E-5** | 관측(하중 0) | ① **proxy≡capture 동일성**(§5-4 [미확인] 의 해소 — ROW 빌더 η·τ vs 캡처 producer 항; 불일치는 그 자체가 발견·전량 보고) ② 계급별 d 분포(프록시: EXP med 2.0e-2·MIX 7.1e-2·BL 2.2e-16) · ff_resp(EXP∪MIX) [프록시 0/790] ③ 부호 일치율(프록시: MIX 1.000·EXP 0.915) ④ σ [추정 SIGNAL — q10(J̄/B) 프록시 0.73] · S/B 분포(med 26 등급) · 초열 행(S/B≫1) census ⑤ NEG 46행 전량 보고 ⑥ hazard 행 캡처 decode 계약 관측(위반 시 Z-2A ZI-X 몫으로 기재 — 이 단이 판정하지 않음) |
| **E-6** | 관측 | 판독 비용: grammar-debug 수 분·수백 MB 파스 등급 |

### 9-2. 분기 (판독기는 분할 밖 포함 **전부 라벨 명시 부여** — UNRESOLVED 금지)

ff 분할(문턱 불변): 평가 순서 X-P → X-D → W-V → C-F → C-R → C-M(잔여).

| 분기 | 판정식 | 함의·처분 |
|---|---|---|
| **C-R** | RD-1~6 PASS ∧ ff ≤ 0.10 | 응답 실증 — **단 «반상태 solve 의 SL-단위 응답»으로 절단**(§3-1 주장 상한). 프록시와의 괴리(BL 동결 소멸)를 E-5① 발견으로 함께 보고 |
| **C-F** | RD-1~6 PASS ∧ ff ≥ 0.99 | 전면 동결 재현 — 후보 라벨(실증 아님): A2-07 커밋→생산자 소비 사슬 감사 + 설계 F4(섭동) 예약 도구 활성 후보 |
| **C-M** | RD-1~6 PASS ∧ 0.10 < ff < 0.99 | 부분 동결 — 행별·계급별 census 전량 보고 = 다음 단 입력(L6C 등록 처분 계승) |
| **└ C-M-S** (C-M 하위) | C-M ∧ 동결 집합 F ≡ BL∩FPC(집합 동일) ∧ EXP∪MIX 동결 0 | **부분 동결의 전량 구조 귀속**: 동결 = 반상태 구조 계급뿐, 응답 가능 계급은 전건 동결-클래스 이탈 ⟹ «생산자 선 소스는 응답 가능 행-계급에서 solve 산출에 응답한다(SL-단위)» 를 **관측 등급으로 병기**(C-R 급 «실증» 라벨은 부여하지 않는다 — ff 골대 불변의 대가를 정직하게 짐). 반상태 수리 단(user 결정 대기)의 정량 입력 완성 |
| **└ C-M-U** (C-M 하위) | C-M ∧ 위 불성립(BL 밖 동결 행 존재 **또는** BL∩FPC 일부 비동결) | 집합-동일 파탄 — 전자면 미귀속 동결 잔존, 후자면 미귀속 동결 없이 BL 내 이탈(그 경우 미귀속 동결은 없다). 어느 쪽이든 순수 미결 + 이탈 행 전량 census(신규 표적) |
| **X-P** (분할 밖) | Z-2A 가 ZI-A 밖 착지(ZI-B/X/D) ∨ R-II 미착지 지속 ∨ PR-0 FAIL | **F1b 소멸 — F1 회귀**(설계 등록 경로). 판독 0회. 이 문서의 §2 교체 전제·§5 판별 설계·계급 census 는 F1 재등록의 입력으로 이관하되 **발주서·사전등록은 처음부터 새로 쓴다**(개정15). ZI-P(R-III/IV 만 미착지)는 X-P 가 아니다 — R-II 의 G4 판정과 분리(Z-2A §8-2), PR-0 문면대로 진행 |
| **X-D** (분할 밖) | RD-1~RD-6 어느 것이든 이름 있는 FAIL | 미결·판정 금지. RD-1/RD-6 급(결정론·차단 재발) → **Z-2A 반송**(§2-5 NEW — 격리 실패의 반증) · RD-2 급 → 덱/census 재검 · RD-3/RD-4/RD-5 급 → 캡처·사상 조사. 산출물 보존 |
| **W-V** (분할 밖) | 비표적 행 >0 ∨ f_mapped < 1.0 | 권한 전제 위반 = 발견(PC-1 구성 증명과 모순). 계측 보존·미결 |

분기는 라벨이지 필터가 아니다 — verdict JSON 은 ff·d 전 분포·계급 census·σ·E-5 전량을 어느
분기에서든 보고한다.

## 10. 처분 원칙 / ★하지 않는 것

- **src/tests/env 접촉 0**(V5 원상 — 권한 요청 0) · GPU 런 0 · 스테이징 0 · 빌드 0.
- **Z-2A 게이트·분기의 재판정 없음**(PR-0 은 그 판정문의 소비다) · R-III/R-IV 산출 무접촉
  (G5/G6 은 Z-2A 몫) · R-II 산출은 **본 문서 커밋 + PR-0 통과 전 무열람**.
- **반상태의 수리·수리 제안 없음**(SUPER 노브·로더·컷오프 — 전부 BITLOCK §8 #4, user 결정
  대기). lumped 준위 구조에 대한 물리 주장 없음(§3-1 주장 상한).
- **ff 문턱·분모·판별량·매체 변경 없음**(§5-3 골대 불변 선언).
- 자유-T 재개 없음 · shell 0 밖/타 이온 확장 없음 · 수렴 주장 없음 · 오라클 정량 인용 0 ·
  W5 오프셋 불접촉 · 봉인 쓰기 0 · clamp/floor/cap 0.
- 발견의 처분 = 조용한 대장 기재.

## 11. 판정 절차 (개정14·16·17)

- 사전등록(본 문서)=Fable / 코딩=Codex / 검수=Fable 별개 컨텍스트 / 집행·판독 실행·대장·
  커밋=운전석 / 판정=Fable fresh / 감리-A=Fable fresh(판정과 다른 컨텍스트·고정질문 4) +
  감리-B=Codex read-only(판정문 실측 주장 재현 한정). 폐합 = A·B 둘 다 통과.
- **검수 고정질문**: 0. 이 산출물을 봉인·실물에 돌려도 되는가(★이 답 전 봉인 접촉 게이트
  RD-1~7 금지 — PR-1·PR-2 는 병행 가) 1. 발주서가 사전등록의 범위를 좁혔는가 2. 판독기가
  행을 조용히 탈락시킬 수 있는가(hazard·NEG 전량 보고 검증) 3. 계급 분류가 물리값·판별량을
  변형하는가(§5-3 위반 검사) 4. 분할 밖 상태에 UNRESOLVED 가 남을 수 있는가.
- 판정 하중 항목(계급 census·d 분포·ff·앵커·f_mapped)은 판정자가 **도구 import 없이 자체
  경로로 재측정**(L6C·BITLOCK 전례). 봉인 무변조 전후 지문(생성 시점 정렬 — L6C 감리-B
  지적 계승)을 판정문이 기재.

## 12. 기계 프리플라이트 선언

PF-1 은 §6 표와 표지 심볼의 양방향 1:1 을, PF-2 는 ff 1차원 분할(C-M 잔여; X-P/X-D/W-V 는
분할 밖, C-M-S/U 는 **C-M 하위**(§9-2) — 어느 쪽도 ff 분할의 셀이 아닌 이름 있는 계급,
전례 형식)을, PF-3 은 인용 앵커 실존을 강제한다.
**정직한 한계**: C-M-S 의 집합-동일 판정식·RD 게이트의 의미 검사·PR-0 의 문서 판독은 정적
검사 밖 — 각각 판독기 NC(NC-R3)·검수 word-diff·PR-0 음성대조가 잡는다. `flags_existing` 은
전부 **상태 불변 앵커**만 올렸다(Z-2A 개정 1 의 시간축 함정 회피 — 이 단은 삭제 요구가
없으므로 그 함정 자체가 없다).

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 — L6C-R (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "scripts/[a-z0-9_\\-]+\\.py",
    "symbol": "A210_L6C_REDISPATCH",
    "roots": ["scripts"],
    "expected_extra": ["scripts/analyze_l6c_redispatch.py"]
  },
  "branches": {
    "regimes": [[1e-16, 1e-14], [1e-10, 1e-4], [0.001, 2.0]],
    "metrics": {
      "ff": "sum(1 for x in v if x <= 1e-13)/len(v)"
    },
    "rules": [
      {"name": "C-R", "predicate": "ff <= 0.10"},
      {"name": "C-F", "predicate": "ff >= 0.99"}
    ],
    "adversarial_fixtures": [
      {"name": "sealed-l6-frozen-class", "mix": [[6.7e-16, 297]]},
      {"name": "solver-response-class", "mix": [[1e-9, 297]]},
      {"name": "partial-freeze", "mix": [[6.7e-16, 150], [1e-9, 147]]},
      {"name": "exact-ten-percent-boundary", "mix": [[6.7e-16, 30], [1e-9, 270]]},
      {"name": "proxy-landing-2026-08-27", "mix": [[2.2e-16, 95], [2.0e-2, 284], [7.1e-2, 506]]}
    ],
    "residual": "C-M"
  },
  "references": [
    {"path": "src/lumina_plasma.c",
     "flags_existing": ["LINE-SATURATION-ROW]",
                        "sprobe_undefined_rows",
                        "a210_line_saturation_target"]},
    {"path": "src/lumina_atomic.c", "flags_existing": ["LUMINA_SUPER_CUTOFF"]},
    {"path": "src/population_contract.c", "flags_existing": ["super_mode"]},
    {"path": "src/lumina.h", "flags_existing": ["byte-identical to baseline"]},
    {"path": "scripts/analyze_a210_z2a_probe.py", "flags_existing": ["NC-I0"]},
    {"path": "docs/VERDICT_BIT_LOCKSTEP_PRODUCER_2026-08-27.md",
     "flags_existing": ["H-FLAT", "PB-A"]},
    {"path": "docs/VERDICT_A210_ZERO_OPACITY_2026-08-26.md",
     "flags_existing": ["ZC-A", "zero_opacity_emitting_rows=221382"]},
    {"path": "docs/VERDICT_DET_L6C_COVER_2026-08-26.md",
     "flags_existing": ["INDEPENDENT_SPROBE_UNDEFINED"]},
    {"path": "docs/RUNG_A210_Z2A_DIAG_ISOLATION_2026-08-26.md",
     "flags_existing": ["ZI-A", "E-I3"]},
    {"path": "docs/RUNG_DET_L6C_COVER_2026-08-22.md", "flags_existing": ["C-M"]},
    {"path": "docs/DESIGN_SEED_PREMISE_REPLACEMENT_2026-08-26.md",
     "flags_existing": ["F1b"]}
  ]
}
```

프리플라이트 본검사: `python3 scripts/check_prereg_preflight.py
docs/RUNG_L6C_REDISPATCH_2026-08-27.md --root .` → **PASS 4건·rc=0** [저자 실행,
2026-08-27 — changeset-consistent files=1 · branch-partition trials=200000 rules=2
fixtures=5 residual=C-M · references-resolved refs=11 · prereg-preflight].
`--selftest` → `SELFTEST_NC total=9 failed=0` rc=0 [저자 실행].
★**개정 1(§16, 검수 반영) 후 본검사 재실행(grammar-debug) → 동일 PASS 4건·rc=0**
[저자 실행, 2026-08-27 — changeset-consistent files=1 · branch-partition trials=200000
rules=2 fixtures=5 residual=C-M · references-resolved refs=11].

★**개정 2(§16, 재검수 지적 반영) 후 본검사 재실행(grammar-debug) → 동일 PASS 4건·rc=0**
[저자 실행, 2026-08-27 — changeset-consistent files=1 · branch-partition trials=200000
rules=2 fixtures=5 residual=C-M · references-resolved refs=11].

## 13. 집행 기록 (★정본 장부 — 개정16 ①; 판정문·감리문은 "장부 = 사전등록 §13" 으로 지목만)

개정15 필수 4항목(발주 프리플라이트 rc · `[미실행]` 실행 결과 · 분장 장부(검수 후 기입) ·
발주 중단 사유·주인)을 운전석이 기입한다. ★BITLOCK §11 의 공백(A-감사 집행 주체 부재)을
반영해 **A-감사 행에 집행 주체를 명시**로 둔다.

| 단계 | 규약상 담당 (개정14·16) | **실제** | 위반 |
|---|---|---|---|
| 노선 결정 (F1b) | user | user — F1b 선택 (2026-08-27) | — |
| 사전등록 (본 문서) | Fable | Fable (2026-08-27) — R-II 무열람·PENDING 실측 하 저작, 프리플라이트 rc=0 [저자 실행] | |
| **★A-감사 ①: 행-계급 × d 교차 census** (봉인 Z-1r read-only) | **저자 (개정16 ③ — 저자 실행 가능 계급)** | **Fable — §5-5, 명령 §14 [저자 실행]. 판정 시 판정자 전건 재계산 의무(§11)** | |
| **★A-감사 ②: 반상태 실측 재확인** (세 봉인 exports·소스 사슬 전 행·死블록 검사) | **저자** | **Fable — §0·§2-1·§4, 명령 §14 [저자 실행]** | |
| 코딩 (판독기 1파일) | Codex | | |
| 코드 검수 (Q0 선행 — 개정16 ②) | Fable (별개 컨텍스트) | | |
| 발주·프리플라이트·커밋 | 운전석 | | |
| PR-0 확인 (R-II 착지·Z-2A ZI-A 폐합) | 운전석 | | |
| 판독 실행·계측 패킷·봉인 지문 | 운전석 | | |
| 판정 (판정문 저작) | Fable (fresh) | | |
| 감리-A | Fable (fresh, 판정과 별개) | | |
| 감리-B | Codex (read-only) | | |
| 감리 반영·대장·커밋 | 운전석 | | |

`[미실행]` 잔여 = **R-II 봉인 산출에 대한 전 기대치(E-1~E-5)** — 저자가 실행할 수 없는
계급(F1b 규율상 무열람 + 산출 미존재). 그 외 사실 주장에 `[미실행]` 없음.

## 14. 저자 실행 원장 (전부 read-only; 무거운 파스는 grammar-debug nested ssh·stdin 파이프)

| 주장 | 명령 [저자 실행, 2026-08-27] | 결과 |
|---|---|---|
| HEAD | `git rev-parse HEAD` | `d02146c…` |
| src 불변 (BITLOCK 좌표 유효) | `git diff --stat a849306 HEAD -- src/` · `git diff --stat 45d931d HEAD -- src/` | 양쪽 공백 |
| R-II PENDING (F1b 시간 규율) | `sacct -j 324557 -o JobID,JobName%24,State,Elapsed,ExitCode -X` | `LUMINA_Z2A_II PENDING 00:00:00` |
| 반상태 exports 3본 | `grep -h SUPER {L6C,L6,census}/input/resolved_lumina.exports` | `LUMINA_SUPER_CUTOFF="100"` 정확 3줄 |
| 시드 예측자 | `grep -n lumina_init_seed_material_predictor src/lumina_cmfgen.c src/lumina_plasma.c src/lumina.h` | `:7525` 호출·`:16060` 정의·`lumina.h:1450` |
| K-덮어쓰기 | `sed -n '1466,1479p' src/lumina_atomic.c` | getenv `:1468`·`if(K>0)` `:1470`·덮어쓰기 `:1473-1474` |
| super_mode·f≡1 | `sed -n '16350,16351p;16456,16457p' src/lumina_plasma.c` · `sed -n '197,201p' src/population_contract.c` | 문면 §2-1 그대로 |
| 재분배·기록·SL 차원 | `sed -n '19386,19388p;19397,19398p;19042,19043p' src/lumina_plasma.c` + `grep -n "nlte_level_populations\[" \| awk 19400..19440` | 주석·`xfl[i]=b[sl]*f`·기록 3행(19418/19422/19433)·`N=super_offset 차` |
| 계약 문면 | `sed -n '723,725p' src/lumina.h` | "byte-identical to baseline" |
| ROW 프린터(HEAD 좌표) | `grep -n 'LINE-SATURATION-ROW]\|ZERO-OPACITY-ROW]' src/lumina_plasma.c` | hazard `:14090`·선정 `:14333`/`:14476` |
| Z-2A 격리 실물 | `grep -n sprobe_undefined_rows src/lumina_plasma.c` · `grep -c INDEPENDENT_SPROBE_UNDEFINED src/lumina_plasma.c` | `:13914/:14157/:14413`·ISOLATED `:15330` · 센티널 **0건** |
| 死블록 무관 | `awk '#if 0' src/lumina_plasma.c` + 깊이 추적 | 유일 블록 1324–1693 — 인용 행 전부 밖. `population_contract.c`·`lumina_atomic.c` 0건 |
| 심볼 선재 0 (코드 트리) | `grep -rn A210_L6C_REDISPATCH src scripts` | 0건 (rc=1) [저자 재실행 2026-08-27 — 검수 지적 A 반영: `docs` 제외, 본 문서 자기참조 소거] |
| ROW 서식 (census) | `grep -m1 "\[LINE-SATURATION-ROW\]" census/stderr.log` | `lower_level=`·`upper_level=`·`tau_validity=`·`chi_effective=` 필드 실재 (§5-1 의 전제) |
| ★계급 census (§5-5 전량) | grammar-debug 스트리밍 파스 2회 — 선정 ROW 전행 파스(SUMMARY 경계 분할)·덱 `levels.csv` (Z,ion,level)→g 조인·계급 분류(K=100)·d=\|(η₁/η₀)/(τ₁/τ₀)−1\|·FPC/NEG 분리·B_ν(10020K, CODATA cgs) | §5-5 표 전 수치. 핵심 재현: FPC=885(설계 885 일치)·ff=0.1073(설계 0.107 일치)·동결 95 ≡ BL∩FPC·NEG 46 전건 BL·hazard 인쇄 154 전건 (lumped,lumped)·등g |
| 프리플라이트 | §12 명령 2건 | rc=0 · selftest 9종 failed=0 |

Z-2A 집행 기록 인용분(재실행 불요·문서 계급): R-II env delta 0 vs 봉인 L6C · 캡처 2종=1 ·
TARGET_ION=1 · 바이너리 `39d452e4…` = `RUNG_A210_Z2A_DIAG_ISOLATION_2026-08-26.md` §10.1 ⑦.

## 15. 승계 기재 (이 단이 열거나 닫지 않고 존속시키는 것)

1. **본선 후속**: 이 단 착지 후 — C-M-S 면 반상태 수리 단(user 결정, BITLOCK §8 #4)의 정량
   입력 완성; 수리 착지 시 FL-단위 재측정 단이 이 단의 census 를 前항으로 쓴다(§3-1 근거 5).
   수리 없이도, 계급-인지 판별식의 **전향적** 재등록(새 데이터 열람 전)은 정당한 경로로
   남는다 — 이 단은 그것을 하지 않았다(§5-3).
2. **BITLOCK §8 #3** (반상태 전제 재감사 — 사정거리 11이온): 존속. 이 단의 계급 census 는
   커버 3이온 몫의 입력이다.
3. **BITLOCK §8 #5** (n_lower 비인쇄 — «n_l≡n_u» 추론 의존): hazard·BL 계급 해석에 승계
   한정 그대로.
4. **설계 §8 #2/#4**: proxy≡capture 동일성 — 이 단 E-5① 이 해소 예정. 동결 잔여 기전 —
   §5-5 가 구조 귀속 후보(BL 비등g)를 특정, 확증은 C-M-S 판정.
5. **Z-O §9 #6**(스캔 밖 coverage) · **W5 공통 오프셋**(+7.265e-6) · **S4-III 행 262210**
   (Z-2A E-I7 몫) · **초열 행의 물리 정당성**(오라클 없이는 판정 불능): 전부 존속.
6. **F6 = BITLOCK**: 폐합 완료(`4bf4a5b`) — 설계 병행 권고의 소진 기재.
7. **★검수(2026-08-27 통과 — 차단 0·지적 4 전부 문면 계급)가 남긴 불확신 4건 (미결/한계)**:
   ① §5-5 의 S/B 중앙 26.14·부호 일치율·J̄/B 분위수는 **저자-계보 [프록시] 실측**이고 운전석
   독립 재현은 d-census 에 한정됐다 — 하중이 관측(E-5)이라 계약 위험은 없으나, **판정자 자체
   재측정 의무(§11)가 이를 덮는지 판정 단계에서 확인할 것**. ② **proxy≡capture 동일성
   [미확인]** — 다르면 §5-5 의 사전 그림이 해석층에서 무너진다(분기 C-R/C-M-U 가 받게 설계돼
   있음은 검수 확인). ③ PR-0 은 **타 단 산출(Z-2A 판정문 + 감리 A·B 통과)에 의존** — 이 단이
   통제할 수 없는 전제이며, 부재 시 fail-closed 임만 확인됐다. ④ C-M-S 집합-동일은 **885행 중
   1행 이탈로도 C-M-U** 가 된다 — 착지 민감도가 높다.
8. **★재검수(2026-08-27 통과)가 남긴 한계 — 가드의 암묵 순서 불변식**: 실패-보고 봉인-내부
   쓰기 가드(§16 개정 2)의 안전성은 *"`report_outside_inputs` 가 `main()` try 블록의 첫
   문장"* 이라는 **암묵 순서 불변식**에 걸려 있다. 장래에 그 앞에 raise 가능한 문장이
   삽입되면 타 실패 사유가 봉인-내부 report 와 공존할 수 있다. NC-R11/R12 가 회귀망이 되지만
   **불변식 자체는 코드 주석으로 명시돼 있지 않다** — 이 단은 판독기 재수정을 발주하지
   않으므로(최소 개정) 한계로 기재하고, 장래 판독기 수정 발주 시 검수 고정질문이 이 불변식
   유지를 확인할 것을 승계한다.

## 16. 개정 이력

**개정 1 (2026-08-27, 저자 = Fable)** — 검수 통과(차단 0·지적 4, 전부 문면 계급) 반영.
검수가 **골대 이동 없음**으로 판정(ff 문턱·판별량·1e-13 동결 문턱 전부 불변 · §5-3 경계 위
0.1073 인지 하 불변 선언 · 분모 선택 결과 중립(세 변형 전부 C-M) · C-M-S = 데이터 존재 전
등록된 반증 가능 술어 + 실패 가지 C-M-U 동반 · 소스 인용 전건 라이브)했고, 계약·전제 교체·
판별식·문턱·분기·경계는 **일절 건드리지 않았다**. 전 개정은 OLD/NEW 쌍으로 적용, 행 단위
diff 로 의도 외 소실 0 확인.

| 지적 | 자리 | OLD → NEW (요지) |
|---|---|---|
| A (심볼 grep 자기참조) | §6 PF-1 · §14 | `grep … src scripts docs` → `grep … src scripts`(본 문서가 docs/ 에서 자기 자신을 잡아 축자 재현 불능). **고친 명령 재실행: 0건 (rc=1)** — 실질 주장(코드 트리 선재 0) 불변 |
| B (PR-0 음성대조 시점 성립 불능) | §7 PR-0·PR-1 · §6 표 | 검수 처방 **둘 다** 채택: "발주 전 시연"→"**판독 전** 시연"(게이트는 발주 산물인 판독기 안에 산다) + **NC-R10**(판정문-부재 합성 픽스처) 신설 ⟹ NC 로스터 R0~R9 → R0~R10. 판정식("부재 시 rc≠0" fail-closed)은 무변경 |
| C (§2-5 OLD 비축자) | §2 제목 · §2-5 | 해당 쌍에 «요약 의역 — 축자 아님» 명시 + 절 제목에 예외 표기(내용 왜곡 없음은 검수 확인). B-4 «토큰/블록 축자 구별» 교훈 |
| D (문구 2건) | §9-2 C-M-U · §12 | ① C-M-U 함의를 집합-동일 파탄의 **두 방향**(BL 밖 동결 / BL 내 비동결 — 후자엔 미귀속 동결 없음)으로 정확화(처분은 양쪽 동일) ② §12 의 "C-M-S/U = 분할 밖" 을 "**C-M 하위**(§9-2)" 로 정정(기계 선언과는 원래 무모순) |

검수 불확신 4건은 §15 #7 로 기재. 개정 후 프리플라이트 재실행 rc=0 (§12 말미 갱신).

**개정 2 (2026-08-27, 저자 = Fable)** — 재검수 **통과**(차단 0 · 폐합 전 지적 1 = 장부-실물
불일치, 비차단) 반영. 계약·전제 교체·판별식·문턱·분모·분기·경계는 **일절 건드리지 않았다** —
변경은 ① NC 로스터 기재의 실물 정합(R0~R10 → R0~R12) ② PR-0 문면 매칭의 형식 요구 명시(§7)
③ 가드 취약성 한계 기재(§15 #8) 뿐이다. 전 개정 OLD/NEW 쌍 적용, 행 단위 diff 로 의도 외
소실 0 확인.

**개정 사유 — 장부-실물 불일치와 그 근원(검수 차단 1건의 수리가 막은 결함)**: 1차 코드
검수가 차단 1건을 잡았다 — 판독기 `main()` 첫 except 블록이 `REPORT_INSIDE_SEALED_INPUT` 을
검출해 놓고 **그 실패 보고를 바로 그 금지된 경로에 썼다**: `mkdir(parents=True)` 로 봉인 안에
디렉토리를 만들고, 기존 파일이면 `os.replace` 로 **침묵 교체**했다. 운전석이 합성 디렉토리에서
실측 재현했다 — 봉인 내부 `model.rc` 가 실패 JSON 으로 덮어써졌고 신규 하위경로가 생성됐다.
★**방어 게이트가 자기 실패 처리에서 방어 대상 행위를 수행**한 결함으로, 개정16 ② 를 만든
사고(스테이저의 봉인 base 침묵 파괴)와 **같은 계급**이며 발화 조건도 같다(운전석이 실물에
돌리는 도중). 수리 발주서가 명시 지시한 신설 음성대조 둘이 로스터에 추가됐고(구현 결함 아님 —
장부가 실물에 뒤처져 있었다), 수리 후 재검수가 **통과**했다:

- **NC-R11**: `inject=REPORT_EXISTING_SEALED_FILE` → `FAIL reason=REPORT_INSIDE_SEALED_INPUT
  branch=X-P rc=4 byte_unchanged=1` / `remove=REPORT_OUTSIDE_INPUTS` → `PASS branch=PR-0`
- **NC-R12**: `inject=REPORT_NEW_SEALED_SUBPATH` → `FAIL reason=REPORT_INSIDE_SEALED_INPUT
  branch=X-P rc=4 path_created=0 parent_created=0` / `remove=REPORT_OUTSIDE_INPUTS` →
  `PASS branch=PR-0`

| 지적 | 자리 | OLD → NEW (요지) |
|---|---|---|
| 재검수 지적(장부-실물 불일치) | §6 표 · §7 PR-1 | 로스터 기재 `NC-R0~R10` → `NC-R0~R12`. `--selftest` 언급 **전수 확인**: 로스터 범위는 §6 표·§7 PR-1 **두 곳뿐** — §7 PR-2·§12 말미·§14 프리플라이트 행의 `--selftest` 는 프리플라이트 게이트 자신의 NC-P/NC-D 로스터(9종/8종)라 무관, §7 PR-0 의 NC-R10 은 개별 음성대조 지목이라 무관, §16 개정 1 표의 "R0~R9 → R0~R10" 은 이력 기록이라 불변 |
| PR-0 문면 매칭 취약(1차 검수 잔여 — 후속 실해 있음) | §7 전제 표 직후 | Z-2A 판정 라인에 대한 형식 요구 명시(`ZI-A`·`E-I3 PASS`·`E-I4 PASS`·`감리 A·B 통과` 문면 필수 — 성한 판정문의 X-P 오차단 방지). 게이트 판정식 무변경 |
| 가드 취약성(재검수 잔여) | §15 #8 신설 | 암묵 순서 불변식(`report_outside_inputs` = try 첫 문장) 한계 기재 — NC-R11/R12 는 회귀망이나 불변식 자체는 코드 주석 부재 |

**1차 검수 잔여 지적 4건 무영향 확인**: ① σ 의 iteration ② X-D↔W-V 파생 사유 면제 ③ PR-0
문면 매칭 취약 ④ 등록 밖 추가 conjunct — 네 건 모두 **수리 델타 밖 무변경 구간**에 있어 이번
수리(실패-보고 경로 가드)에 영향받지 않았다. 그중 ③ 은 후속에 실해가 있어(성한 Z-2A 판정문의
X-P 오차단) 위와 같이 §7 에 형식 요구로 명시했고, 나머지 셋은 잔여 지적으로 존속한다.

개정 후 프리플라이트 재실행 rc=0 (§12 말미 갱신).
