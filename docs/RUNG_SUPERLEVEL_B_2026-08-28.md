# 사전등록 — DET-SUPERLEVEL-B: 선언된 SL 형상의 f-일관화(승격) (2026-08-28)

- **단 이름**: DET-SUPERLEVEL-B
- **저자**: Fable (사전등록, 분담 개정14 ① — CLAUDE.md 상시 승인)
- **상태**: 발주 전 (커밋 전 — 운전석 지시)
- **성격**: **env-계약 수리 단** — `src/` 무접촉. GPU 판정런 **2회 + 조건부 예비 1회(상한 3)** (§10).
- **근거 설계**: `docs/DESIGN_SUPERLEVEL_DISPOSITION_2026-08-28.md` (커밋 `ddf6701` = 작성 시점
  HEAD [저자 실행: `git log -1`]) — user 가 **B안(SL 승격)** 을 승인했다.
- **표기**: `[저자 실행]` = 저자가 직접 실행·실측(명령 원장 §16) · `[설계 실측 인용]` = 설계
  문서의 [저자 실행]을 지목만(본 저자 미재확인) · `[추정]` = 실측 아님 · `[미실행]` = 저자가
  실행할 수 없는 주장 — **본 문서에 0건**.
- 봉인 4곳(census·R-II·L6C·L6) 접근 전부 read-only. 오라클(CMFGEN 자체런) 정량 인용 **0**.
  전처리 死블록 규정 준수: 인용한 소스 행 전건이 `lumina_plasma.c:1324-1693` ·
  `lumina_cuda.cu:1083-2063` **밖**임을 저자가 깊이 오토마톤으로 재확인했다(§2-5 — 유일한
  예외는 §2-5 의 死블록 **내부** 소비행 지적 그 자체다).

---

## §0. 경위 (개정15 면제 절 — **이 절의 값은 요구가 아니다**)

BITLOCK(`docs/VERDICT_BIT_LOCKSTEP_PRODUCER_2026-08-27.md`)이 반상태(super-level collapse)를
귀속했고, 미결 #4(«반상태 수리 단 개설 여부»)를 user 결정 대기로 남겼다. 설계 연구
(`ddf6701`)가 갈림길 5개(A~E)를 증거·비용·falsifier 와 함께 열거했고, user 가 **B안 승격**을
승인했다. 이 단이 그 승인을 집행한다.

반상태의 요지: `LUMINA_SUPER_CUTOFF=100` 이 `LUMINA_SUPER_LEVELS` 없이 봉인 계보 전체에
적용되어, 레이아웃은 SL(1,173)로 접혔는데 `within_sl_frac ≡ 1.0` 이라 ① SL 방정식이 무가중
오조립되고 ② 한 lump 안 전 준위 인구가 bit-동일이 되어 τ 정확-0(hazard 221,382/반복)·위장
inversion 이 만들어지며 ③ 배너는 `off (identity)` 라고 거짓말한다.

---

## §1. 계약 (하나)

> **선언 형상-가중 일관성 계약**: NLTE solve 가 **선언된** super-level 형상(K=100 · 1,173 SL)
> 위에서 돌 때, SL 방정식 조립과 SL→FL 재분배는 **같은 within-SL Boltzmann 분율 f 를 일관되게**
> 소비해야 하며(어떤 lump 도 f≡1 무가중이 아니어야 하며), 런의 구성 선언(env)·배너·헤더 계약
> (`lumina.h:720-726`)이 실태와 일치해야 한다.

수리 형식은 §3(env 1행 승격). 이 계약은 **형상(axis 1)을 바꾸지 않는다** — K=100 형상은
런처가 fail-closed 로 선언해 온 topology 그대로다(설계 §3·§7).

---

## §2. 기전 특정 (전 행 번호 [저자 실행] — HEAD `ddf6701`, `git status -- src/` 공백,
`git diff af7441d HEAD -- src/` 공백 ⟹ 설계·BITLOCK 의 행 번호와 동일 트리)

### 2-1. 반상태 사슬 (설계 §0 의 재확인 — 전건 저자 직독)

1. `src/lumina_atomic.c:1468` `getenv("LUMINA_SUPER_CUTOFF")` · `:1470` `if (K > 0)` —
   **무조건** 덮어쓰기(`:1473` `level_super[l] = K`), 배너 `:1476-1477` 는 "set
   LUMINA_SUPER_LEVELS=1 to activate" **권고 문구만**.
2. `src/lumina_plasma.c:16350-16351` `super_mode = super_requested && n_super_total <
   n_nlte_levels_total` · `:16456-16457` `super_requested = getenv("LUMINA_SUPER_LEVELS")…
   || element_wide`. 레이아웃(`fl_to_super`, `:16384-16387`)은 super_mode 와 **무관하게**
   `level_super[]` 에서 온다.
3. `src/population_contract.c:199-200` `if (!super_mode) work[i]=1.0` ⟹ `within_sl_frac ≡ 1.0`.
   super_mode=1 분기(`:241` 부근~— g·exp(−ΔE_anchor/kT_e)·SL 분배함수 정규화)가 Boltzmann 팔.
4. **행렬 조립**: `src/lumina_plasma.c:16993-16994` `SOLVE_OF`/`FRAC_OF` — 주석(`:16987-16991`)
   문면 *"FL contributions aggregate into their SL row/col, weighted by the within-SL
   Boltzmann fraction"*. f≡1 이면 무가중 합.
5. **재분배**: `:19396-19398` `xfl[i] = b[sl] * f` → **재정규화** `:19401-19437`(lock 분할 또는
   pair-total 스케일 — 총량은 `ion_number_density` 에 보존). ⟹ 반상태에서 lump 전 멤버 인구
   동등, 승격 시 상대 분포만 Boltzmann 으로 바뀌고 총량 보존은 불변.
6. **배너**: `:16483-16485` `[NLTE] Super-levels: %s (%d FL -> %d SL across ions)` — 봉인 실측
   `off (identity) (13015 FL -> 1173 SL)` (`<RII>/stdout.log:236` [저자 실행]).

### 2-2. f 의 생산 — 살아있는 경로는 CPU 단일 사슬

`src/lumina_plasma.c:19831` (매 solve 전 무조건 호출) → `:19633`
`nlte_precompute_within_sl_frac_checked` → `population_within_superlevel_build`
(`population_contract.c`). 함수 머리 주석(`:19622-19632`) 문면: *"MUST be called before every
solve … using the SAME current T_e … The f-weighting, when current, preserves detailed balance
exactly for both bb and bf."* — f 는 **반복마다 현재 T_e 로 재계산**된다(스탬프 강제).
`LUMINA_SUPER_LEVELS` 의 소스 독자는 `lumina_plasma.c:16456` **단일**(`grep -rn` 전수
[저자 실행]).

### 2-3. env 수용 사슬 [저자 실행]

- `src/env_universe.h:490` 에 `LUMINA_SUPER_LEVELS` 등재 ⟹ 봉인 계보의
  `LUMINA_ENV_STRICT=1`(exports 실측)이 **거부하지 않는다**.
- 봉인 계보 job 스크립트(`<RII>/input/job.slurm:268-282` 직독)의 수리-노브 0 강제 목록에
  `LUMINA_SUPER_LEVELS` 는 **없다**. `LUMINA_SUPER_CUTOFF == 100` fail-closed 는 그대로
  충족된다. ⟹ **env 1행 추가는 기존 fail-closed 어느 것과도 충돌하지 않는다.**

### 2-4. 봉인 실물의 구성 [저자 실행 — read-only]

| 항목 | census `<ZO>` | R-II `<RII>` |
|---|---|---|
| 런루트 | `/gpfs/kjhan/lumina/a210_zero_opacity_z1r_a100x2_k36/zo_20260826T050021Z_census_ion1` | `/gpfs/kjhan/lumina/a210_z2a_diag_isolation_a100x2_k36/rii_20260826T134337Z_ion1` |
| job | 324388 `LUMINA_ZO_CENSUS_II` 02:42:56 COMPLETED syn102 | 324557 `LUMINA_Z2A_II` 04:34:36 FAILED(70:0 — 판독기 오탐, 모델 `model.rc=0`) syn101 |
| 자원 (sacct [저자 실행, syntax]) | gpu=2·cpu=32·node=1 | 동일 + `ReqMem=64G`·`Timelimit=08:00:00` |
| exports | 89행, `LUMINA_SUPER_CUTOFF="100"` 단독(`LUMINA_SUPER_LEVELS` 부재) | 동일 |
| exports 상호 diff | **비경로 실질 차 = capture 2노브뿐**: `LUMINA_A210_INDEPENDENT_CAPTURE` 0↔1 · `LUMINA_A210_SPRODUCER_CAPTURE` 0↔1 (+ 런루트 경로 6키: `LUMINA_CMFGEN_SIGMA_BF`·`LUMINA_DEPOSITION_FILE`·`LUMINA_IONIZATION_REFERENCE_FILE`·`LUMINA_MODEL_DIR`·`LUMINA_PHYSICS_COMPARISON_DIR`·`LUMINA_TOPION_LEVELS_FILE`) | ← |
| binary sha256 | `0e2fc7b4…e7df` | `39d452e4…aa42` (상이 — Z-2A 계측 사이 빌드) |
| 덱 sha256 (RUN_SEAL 등재) | levels.csv `b3645052…daf67` · line_list.csv `cbd81c4f…54b5` | **동일 값** |
| argv | `<run>/input/lumina_cuda <run>/input/model 1 2 spectrum nlte` | 동일 형 |
| SUMMARY 수치 | R-II 와 **문자열 동일**(diff 는 R-II 의 신설 필드 `sprobe_undefined_rows` 유무 1건뿐 — 바이너리 세대 차) [저자 실행: 양 stderr SUMMARY 추출 diff] | ← |

### 2-5. ★死블록 신규 실측 — GPU 쪽 super 소비는 전부 死코드

`lumina_cuda.cu` 의 `within_sl_frac`/`fl_to_super` 소비 행(`:1088-1090`·`:1213`·`:1793-1794`)은
**전부 死블록 1083–2063 내부**다 [저자 실행 — `#if 0` 깊이 오토마톤: plasma open 1324/close
1693 · cuda open 1083/close 2063 재확인]. ⟹ 이 수리의 살아있는 소비자는 §2-1·§2-2 의 CPU
사슬(조립·재분배·f-생산)이 전부다. 봉인 계보는 `LUMINA_NLTE_ASSEMBLE_GPU=0` 강제(job.slurm
zero-노브 목록)와도 정합한다. 설계 문서가 cuda 행을 인용하지 않은 것은 옳았다.

---

## §3. 수리의 정확한 형식 — 판정 (과제 ①)

### 3-1. 「env 1행」의 특정

**수리 = 판정런 env 에 `LUMINA_SUPER_LEVELS=1` 1행 추가. 그것이 전부다.** 소스의
`super_requested` 조건(`:16456-16457`)은 **바꾸지 않는다** — 조건은 성하다(env 를 읽어 그대로
행동한다). 결함은 조건이 아니라 **계보의 env 선언이 반쪽**(CUTOFF 만 있고 LEVELS 가 없음)인
것이고, 수리는 선언을 완성하는 것이다. 근거 사슬(전건 §2 [저자 실행]):

1. `LUMINA_SUPER_LEVELS=1` ⟹ `super_requested=1` ⟹ `super_mode = 1 && (1173 < 13015) = 1`.
2. `super_mode=1` ⟹ `population_contract.c` Boltzmann 팔 ⟹ f 가 조립(`FRAC_OF`)과 재분배
   (`:19397`)에 **동시에** 일관 적용 — 두 소비처가 같은 배열을 읽으므로 by-construction 일관.
3. env 전집 등재(§2-3) ⟹ strict env 통과. 기존 fail-closed 무충돌(§2-3).

### 3-2. V5 판정 — **불요. 단 용어를 갈라 둔다**

- **판정 규약의 V5**(`docs/VERDICT_PROTOCOL.md`)는 「대장 갱신」 단계다 — src 와 무관하다.
- 설계·BITLOCK·L6C-R 판정문의 「V5 보존/원상 유지」는 캠페인 관용으로 **«src/tests/Makefile
  무변조»**를 뜻한다(L6C-R 판정문 머리 불릿 「V5 원상 유지: `git diff --stat … -- src tests
  Makefile` 공백」이 그 조작화다 [저자 실행 직독]).
- **본 단은 양쪽 의미 모두에서 V5 를 충족한다**: ① src/tests/Makefile 접촉 **0** — 수리가
  env 1행 + 신규 스크립트 2본(§8)뿐이므로 「src 접촉 권한」이 필요 없다. **재빌드도 불요** —
  판정런은 봉인 R-II 의 바이너리 사본(sha `39d452e4…aa42`)을 그대로 쓴다(§7-1). ② 판정 규약
  V5(대장 갱신)는 폐합 절차에서 통상대로 수행한다.
- 후속 문서는 「V5」 단독 표기를 피하고 **「src 무접촉」/「대장 갱신(V5)」**으로 갈라 쓸 것을
  기재한다(용어 충돌의 조용한 대장 기재 — 수리 아님).

### 3-3. ★노브 거짓말 문제의 처분 — 이 계보에서는 소멸, 소스 결함으로는 잔존

- **소멸하는 것**: 승격 런에서 배너는 `ACTIVE` 를 인쇄하고(§11 E-2) env 선언(양 노브)·헤더
  계약·실태가 일치한다 — 이 계보의 자기모순은 없어진다.
- **잔존하는 것**: `lumina_atomic.c:1470` 의 무조건 덮어쓰기는 그대로이므로, **CUTOFF 단독
  설정은 여전히 반상태를 만들 수 있다**(상태 공간에 함정 잔존). 본 단은 src 를 고치지 않으므로
  이것을 수리하지 않는다. **처분 = 조용한 대장 기재**: 후보 수리(로더의 super_requested 종속화
  또는 CUTOFF-단독 fail-closed 거부, 그리고 설계 D안(배너·헤더 정직화))는 **별도 단**(개설은
  user 몫). 신규 job 스크립트의 `LUMINA_SUPER_LEVELS == 1` fail-closed(§8)가 이 계보 안에서의
  재발은 기계로 막는다.

### 3-4. 하지 않는 수리 (경계)

- 형상 변경 없음(A안 full-FL·E안 f_to_s 복귀 — 설계 §7 대로 국면 밖 이월).
- `SUPER_LEVELS=1`(접두 없는 死노브) 77본 스크립트 정리 없음 — 관측 기록만.
- occupation probability(W)·INT_SEQ 등 CMFGEN 관례의 나머지 부분 구현 없음(설계 §4-B 한계 ⓒ
  — 개량 금지 규약).

---

## §4. 수리안의 오프라인 검증 — 덱 전수 산술의 저자 재실측 (과제 ②의 falsifier 재확인)

설계 §4-B 의 falsifier(「등g ∧ 등E-bit = 0/1,537,729」)를 **저자가 독립 스크립트로 재실측**했다
[저자 실행 — grammar-debug(nested ssh·stdin 파이프), 봉인 `<RII>/input/model` read-only,
스크립트 전문 §16 C6]:

| 량 | 설계 등록값 | **저자 재실측** | 일치 |
|---|---|---|---|
| 덱 전체 행수 | 2,588,798 | 2,588,798 | ✓ |
| 인벤토리 실재 슬롯 | 12 (11절단+Ca II) | 12 — 절단 11: Si II 125·Si III 147·S II 322·S III 256·Ca III 232·Fe II 2,599·Fe III 1,500·Co II 2,558·Co III 3,214·Ni II 1,000·Ni III 1,000 / Ca II 62준위(무럼프) | ✓ |
| FL 합 / SL 합 (K=100 산법) | 13,015 / (배너 1,173) | **13,015 / 1,173** — 봉인 배너와 정확 일치 | ✓ |
| 양준위 lumped 선 | 1,537,729 (덱의 59.4%) | **1,537,729 (59.4%)** | ✓ |
| 그 중 등g | 539,149 | **539,149** | ✓ |
| ★등g ∧ 등E-bit | **0** | **0** | ✓ |
| 커버 ion=1 등g (Z 26/27/28) | (BITLOCK D5: 325,489) | **152,821 / 159,064 / 13,604 · 합 325,489** — BITLOCK D5 와 정확 일치 | ✓ |

**저자 신규 강화 3건** (설계에 없던 실측):

1. **등E-bit 는 g 조건 없이도 0/1,537,729** — 어떤 lumped 쌍도 에너지가 bit-동일하지 않다.
2. **E 강단조**: 전 1,537,729행에서 `E_lower < E_upper` 강부등(위반 0) ⟹ Boltzmann f 에서
   `exp(−ΔE_l/kT) > exp(−ΔE_u/kT)` (exp 단조) ⟹ 럼프-내 쌍의 `n_l/g_l − n_u/g_u > 0` —
   **hazard(τ 정확-0)와 위장 inversion(χ<0)이 같은 대수로 동시에 구조 소멸**한다.
   반상태 대수와의 대비: f≡1 에서는 이 차가 `b·s·(1/g_l − 1/g_u)` 라 등g ⟹ 정확-0 필연이었다.
3. **exp 언더플로 경로 부재**: 럼프-내 최대 ΔE = 18.864 eV(Co III; 이온별 표 §16 C6) ⟹
   `exp(−ΔE/kT_e)` 가 0 으로 언더플로하려면 T_e < ~294 K 가 필요 — 이 모델 계급에서 비현실
   [추정 — T_e 는 런 산출이나, `population_contract.c` 가 T_e≤0 을 거부하고 봉인 계보 T_e 는
   수천 K 대다]. ⟹ 「양쪽 f 가 0 으로 깔려 새 정확-0 계급이 생기는」 경로는 닫혀 있다.

**잔여 [추정] 1건**: 승격 후 정확-0 잔존은 `exp` 왕복의 **부동소수 우연 충돌**(인수 상이 →
값 bit-동일)에서만 가능하다. 크기는 사전 계산 불가 — §11 의 SLB-P 분기가 이를 이름 있는
계급으로 흡수하고, 행 단위 귀속 절차(§9 G-5)를 사전등록한다.

---

## §5. 사정거리 (과제 ⑤ — 재확인·기재)

수리는 **커버 3이온에 한정되지 않는다**:

- **직접**: 절단 11이온의 lumped 준위 전부(§4 표) — f 가 1.0→Boltzmann 으로 바뀌는 준위.
  양준위 lumped 선 **1,537,729행(덱 59.4%)** 의 τ·S 산법이 g-비 아티팩트에서 Boltzmann 재분배로
  바뀐다. Ca II(62준위)는 무럼프라 f=1 그대로(싱글턴 SL 의 f 는 양 모드에서 정확 1.0 —
  `population_contract.c` 산법상 w/w).
- **간접**: SL 방정식 조립 가중이 바뀌므로 **해 `b[sl]` 자체가 바뀌고**(§2-1 ④), 재정규화
  (`:19401-19437`)를 거쳐 **explicit 준위(랭크<100) 인구에도 파급**된다. 즉 12슬롯 인벤토리의
  전 13,015 FL 인구·그 위의 전 선 τ/S·후속 장·스펙트럼이 잠재 변경 집합이다 — 이것이 §6 의
  bit-앵커 전멸이 필연인 이유다.
- **비커버 잔여**: NLTE 인벤토리 밖 이온(예: ion IV 594행 — A207 W-A)은 LTE-at-핀-T 경로
  그대로 — 이 수리의 직접 사정거리 밖(간접으로는 장 변화 경유 [추정]).

---

## §6. bit-앵커 전멸의 처리 (과제 ③)

### 6-1. 원칙 — 앵커가 죽는 것과 결론이 틀린 것은 다르다

설계 §6-1 의 사망 목록(Z-O hazard census 221,382·zero-스트림 sha `84f977bb…`/`64ccbb87…`·
결정론 앵커 행 `line=669992` 4필드×2반복·BITLOCK hex 6종·Z-2A `f_bit=1.0`·F1b 판별 전량·
K36 R1/R2 residual·DET-SPROD S2·L6/A207 수치 앵커·parity 마이크로픽스처)과 생존 목록
(덱-계급 사실·PC-2·소스 감사·합성 픽스처·판정문/봉인 자체)을 **그대로 인수**한다.

★**폐합 4단(Z-O·BITLOCK·Z-2A·F1b)의 판정은 뒤집히지 않는다.** 각 판정문의 자기 구성 한정을
저자가 문면으로 재확인했다 [저자 실행]:

| 단 | 자기 한정 문면 (확인 지점) | 승격 후 지위 |
|---|---|---|
| Z-O (5회차) | census 는 「이 구성의 관측」 자기 명기 — 물리 일반화 불가 명시 (설계 §2-3 재확인) | 반상태 구성의 관측으로 영구 유효 |
| BITLOCK (6회차) | 귀속이 곧 내용 — 반상태의 서술 그 자체 | 본 단의 §2 가 그 사슬의 재확인 |
| Z-2A (7회차) | 격리 계약 검증은 수리와 독립 존속; in-vivo 스트레서가 반상태 산물이라 「같은 스트레스 재현 불가」 주석만 | 계약 결론 불변 |
| F1b (8회차) | 판정문 스스로 «SL-단위»·«이 구성의 실물 solve(반상태 1,173-SL)» 한정 — 관측 등급 병기, C-R 급 실증 라벨 거부 [저자 실행: 판정문 요약 직독] | **수리-후 재측정의 前항으로 설계된 그대로 소비**(§7) |

⟹ 전멸하는 것은 **결론이 아니라 앵커 값**(미래 측정의 대조 기준선)이다. 이 단이 새 구성의
기준선을 재-bake 한다(§6-2).

### 6-2. 재구축 계획 — **런 2회(+예비 1회)로 재봉인·차동·재-bake 를 겸한다**

- **B-1** (census-형, capture 2노브=0): 새 구성의 hazard census 후항 + 미러 반쪽.
- **B-2** (R-II-형, capture 2노브=1): F1b-형 계급 census·d-분포의 후항 + 미러 반쪽 + 새 앵커
  bake 의 본체.
- **B-3** (예비 — **SLB-F-D 분기 발화 시에만**, B-2 와 동일 구성 반복): 미러 실패가
  비결정론인지 스테이징 사고인지 가르는 재현런. 그 외 분기에서 추가 런 금지 — 상한 3회.
- 판독·판정·감리 1사이클(설계 §6-3 정량과 일치 — 봉인 실측 02:42:56/04:34:36, sacct 재확인
  [저자 실행]).
- **새 앵커 재-bake 절차** (값은 사전 예측 불가 — 관측 등록, pass/fail 아님): ① B-2 capture
  스트림에서 **양 반복에 모두 존재하고 양쪽 `chi_effective>0` 인 FPC 교집합의 수치상 최저
  `line` 정확 1개**를 고른 뒤, 같은 행의 iter0/1 네 필드 `tau_validity`·`n_upper`·`A_ul`·
  `emission_per_sr`를 등록한다. 표현 기준은 `tau_validity`의 정규 십진 정수 문자열 1종 + 나머지
  세 필드를 십진 입력에서 파싱한 binary64의 Python `float.hex()` 3종이다 ② SUMMARY 전 필드
  문자열(양 런×양 반복) ③ §7-2 ③의 B-1/B-2 등록 산출물 9파일별 sha256 목록 ④ 새
  zero-스트림 행수(예측 0 — E-4). 등재처: 본 단 판정문 + 대장. 기존 앵커 행 `line=669992` 는
  zero-스트림 소속이라 **승격 구성에서 스트림째 소멸이 예측**되므로(§11 E-4) 같은 행 재사용을
  요구하지 않는다.

### 6-3. ★봉인은 그대로 둔다

기존 봉인 4곳은 read-only 로만 연다 — **다시 만들 수 없는 반상태 시대의 증거**다. 새 런은
새 런루트(`/gpfs/kjhan/lumina/superlevel_b_a100x2_k36/`)에 봉인한다. 하드링크 금지·복사 후
sha 대조(§8·§9 G-1).

G-7의 **런 전체 checksum 검증 사정거리**는 다음 두 봉인으로 한정한다. 정본성 판별은 manifest
파일만 존재하면 된다는 뜻이 아니라 표의 조건을 전건 만족해야 한다는 뜻이다.

| 봉인 | 봉인 root · checksum manifest 상대경로 | manifest 정본성 판별 기준 |
|---|---|---|
| census | `/gpfs/kjhan/lumina/a210_zero_opacity_z1r_a100x2_k36/zo_20260826T050021Z_census_ion1` · `ZO_RUN_SEAL.sha` | root의 정규파일·정확 136행 ∧ 동반 `ZO_RUN_SEAL.txt`의 `schema=A210_ZO_CENSUS_RUN_SEAL_V1`·`run_root=<왼쪽 절대경로>`·`DO_NOT_MUTATE=1` 정확 일치 ∧ root에서 `sha256sum -c ZO_RUN_SEAL.sha` OK 정확 136·non-OK 0 |
| R-II | `/gpfs/kjhan/lumina/a210_z2a_diag_isolation_a100x2_k36/rii_20260826T134337Z_ion1` · `RII_RUN_SEAL.sha` | root의 정규파일·정확 135행 ∧ 동반 `RII_RUN_SEAL.txt`의 `schema=A210_Z2A_RII_RUN_SEAL_V1`·`run_root=<왼쪽 절대경로>`·`DO_NOT_MUTATE=1` 정확 일치 ∧ root에서 `sha256sum -c RII_RUN_SEAL.sha` OK 정확 135·non-OK 0 |

**사정거리 밖의 명시적 한계**: L6 root는
`/gpfs/kjhan/lumina/det_stage12_fixed_te_a100x2_k36/sprim_l6_20260821T054111Z_probe`, L6C root는
`/gpfs/kjhan/lumina/det_stage12_fixed_te_a100x2_k36/l6c_20260822T154507Z_cover`로 확정한다. 두
root에는 런 전체 `.sha` manifest가 없고 각각 `L6_STAGE_SEAL.txt`만 있다. 그 문서는 스테이징 당시
base-input 불변·구성 delta를 기록하지만 런 전체 내용 checksum을 열거하지 않으므로 G-7의
`sha256sum -c` 피연산자로 승격하지 않는다. 따라서 G-7은 L6·L6C의 역사적 내용 무결을 주장하지
않으며, 이 둘의 read-only 의무만 유지한다. 봉인 안 manifest 신규 생성·교체로 이 한계를 덮는
것은 금지한다.

---

## §7. 전후 차동 설계 (과제 ④)

### 7-1. 통제 — 바뀌는 것은 정확히 env 1행

| 통제 축 | 값 | 강제 장치 |
|---|---|---|
| 덱 | 봉인과 동일 — levels.csv `b3645052…daf67`·line_list.csv `cbd81c4f…54b5` (RII/ZO RUN_SEAL 등재값 동일 [저자 실행]) | G-1 sha 대조 |
| 바이너리 | **봉인 R-II 사본 그대로** `39d452e4…aa42` — **재빌드 0**(현 트리 상태와 무관해짐) | G-1 sha 대조 + RUN_FOOTER |
| argv (시드 포함) | `<run>/input/lumina_cuda <run>/input/model 1 2 spectrum nlte` (봉인 두 런과 동형) | RUN_FOOTER 대조 |
| 반복수 | 2 (outer_iterations=2) | RUN_FOOTER |
| 노드 계열 | a100×2 (파티션 a100 한정 — §10; 봉인 쌍이 syn101↔syn102 교차에서 zero-스트림 bit-동일을 실증했으므로 노드 핀은 불요, 계열 핀만) | sbatch |
| env — **물리 유효 델타** | **`LUMINA_SUPER_LEVELS=1` 정확 1키** (B-2 vs 봉인 R-II) | **V1 `verdict_gate.py --intended-diff`** + G-2 (exports 실측 diff 의 키 집합 정확 일치 — 초과·누락 0) |
| env — 자명 델타 | 런루트 경로 6키(§2-4 목록, 값 = 새 런루트 치환) · B-1 은 추가로 capture 2노브=0(봉인 ZO 값) | G-2 등록 키 목록 |

### 7-2. 차동 3축

1. **主차동 — B-2 vs 봉인 R-II (F1b 前항)**: 같은 판독 산법(신규 판독기가 먼저 봉인 R-II 에서
   F1b 등록값을 bit-재현해 자격을 얻는다 — G-RC)으로 B-2 를 재측정. **95행 행-수준 이동표의
   모집단**은 前항 정본 `/gpfs/kjhan/lumina/zo_artifacts/f1b_result.json`의 `matched_rows` 중
   `chi_axis=="FPC" && class=="BL"` 인 **서로 다른 line-id 정확 95개**다. 각 행에 前항
   axis/class/d/frozen과 B-2 후항의 존재 여부·axis/class/d/frozen을 기록한다. `chi_axis=="NEG"`
   인 46행(전항에서는 전부 class=BL)은 이 95행과 **서로소인 별도 모집단**이며 95행 표에 넣지
   않고 별도 line-id 추적표와 census로 기록한다. census는 곱집합 8셀이 아니라 독립 축
   `matched_FPC`의 class별 3셀(BL 95·EXP 284·MIX 506)과 `matched_NEG`의 class별 셀(전항 BL
   46)이고, `d_distribution_by_class`의 5수치(min·q10·median·q90·max)는 FPC 축만을 분모로
   한다. G-6·E-5의 **BL 잔여**는 B-2의 matched_FPC 중 `class=="BL" && d<=1e-13` 인 행 수,
   **NEG 잔여**는 B-2의 matched_NEG 전 행 수로 정의한다. ff는 B-2 matched_FPC의 유한 d 행 중
   `d<=1e-13`인 비율이다 [저자 실행 직독: 전항 BL-frozen 95·EXP 284·MIX 506·NEG 46·ff
   0.10734463276836158·EXP 최소 d 2.928690524872657e-6].
2. **hazard 차동 — B-1·B-2 vs 봉인 census**: SUMMARY `zero_opacity_emitting_rows`·
   `sprobe_undefined_rows`(전항 221,382·양 반복 [저자 실행: f1b RD-1 직독 + 봉인 SUMMARY
   추출]) 대 후항(예측 0 — E-4). 주석: 봉인 census 바이너리는 `0e2fc7b4…` 로 R-II 와 다르나
   두 봉인의 SUMMARY 수치·zero-스트림이 동일함이 실증되어(§2-4·F1b A-5) 차동 유효.
3. **결정론 미러 — B-1 ≡ B-2**: 미러 잣대는 **SUMMARY 전 필드 문자열**(같은 바이너리이므로
   필드 집합까지 동일해야 한다 — 봉인 쌍에서 값 동일이 실증된 잣대 [저자 실행 §2-4]) + 다음
   **등록 산출물 9파일**이다: `work/lumina_plasma_state.csv`·`work/lumina_spectrum.csv`·
   `work/lumina_spectrum_comoving.csv`·`work/physics_comparison/physics_DET_iter0000.manifest.json`·
   `work/physics_comparison/physics_DET_iter0000.shell.csv`·
   `work/physics_comparison/physics_DET_iter0000.spectral.csv`·
   `work/physics_comparison/physics_DET_iter0001.manifest.json`·
   `work/physics_comparison/physics_DET_iter0001.shell.csv`·
   `work/physics_comparison/physics_DET_iter0001.spectral.csv`. sha 기준은 각 상대경로의 정규파일
   **전체 원바이트를 무정규화로 읽은 SHA-256**(소문자 64자리 hex)이며, B-1/B-2 양쪽에 9파일이
   모두 정확히 존재하고 같은 상대경로끼리 digest가 일치해야 한다(누락·비정규파일·digest 불일치
   어느 하나도 G-7 FAIL). ★전항의 미러 잣대였던 zero-스트림은 승격 구성에서 **비어버리는 것이
   본 단의 1차 예측**이므로, 미러 기질을 SUMMARY·이 9파일로 **지금 재선언**한다(잣대 공백의
   사전 봉쇄).

---

## §8. 기대 변경집합 — PF-1 대상

### 변경집합 — SLB (표)

| 파일 | 변경 성격 |
|---|---|
| `scripts/stage_superlevel_b.sh` | 신설 — B-1/B-2(/B-3) 스테이징+제출: 봉인 R-II input 실복사(`cp`·reflink 허용·**하드링크 불변량 검사 계승**)·복사 후 sha256 대조(§7-1 표)·env 델타 fail-closed(`LUMINA_SUPER_LEVELS == 1` 요구 추가 + 기존 zero-노브·`CUTOFF==100` 계승)·`OWNER.txt`(프로젝트·목적·DO_NOT_CANCEL)·sbatch(§10 파라미터)·기대 export 델타 각인. ★**job 스크립트에 판독기 인라인 호출 금지**(324557 의 `FAILED 70:0` 을 만든 수용-블록 오탐 전례 — 판독은 착지 후 오프라인). 표지 심볼 `SUPERLEVEL_B_PROMOTION` |
| `scripts/analyze_superlevel_b.py` | 신설 — 도구 import 0 자체 파서(F1b 판정자 전례): G-2~G-7 판독·G-RC 전항 보정(봉인 R-II 에서 F1b 등록값 bit-재현)·행-수준 이동표·새 앵커 bake 산출·`--selftest`(합성 픽스처 NC 군, §9). 표지 심볼 `SUPERLEVEL_B_PROMOTION` |

### 변경집합 끝

**이 밖의 변경 0**: `src/` 0 · `tests/` 0 · `Makefile` 0 · 기존 스크립트 수정 0 (계측 수리
단의 두 판독기 무접촉 — §15) · 봉인 0 (read-only) · 본 문서 외 docs 0. 산출물 거처:
`validation/superlevel_b/` 신설(소형 JSON·로그 — PF-1 밖) + 새 런루트(/gpfs). 표지 심볼은
발주 전 저장소 전체에서 **0파일** [저자 실행 grep], 구현 후 정확히 이 2파일.

---

## §9. 게이트 (각각 음성대조·봉인 접촉 표기)

★집행 순서(개정16 ②): **코드 검수 Q0**(«이 산출물을 봉인·실물에 돌려도 되는가») 답 전에는
봉인 무접촉 게이트만(G-0·G-S) 집행 가능. 봉인 접촉 게이트(G-RC)와 **런 제출**(G-1 이후 전부)
은 Q0 답 후. 판독 실행 티어: 대형 stderr 파스는 grammar-debug(F1b 전례) — 산출은
`validation/superlevel_b/`(노드-로컬 `/tmp` 금지). **sacct 를 소비하는 게이트는 두지 않는다**
(완주는 RUN_FOOTER·model.rc 로 판정 — 계측 수리 단이 실증한 sacct 교차-클러스터 함정의 원천
회피; 운전석의 job 시간 기록은 집행 기록 몫, 실행 노드 = syntax 로그인).

| 게이트 | 판정식 | 음성대조 | 봉인 접촉 |
|---|---|---|---|
| **G-0 프리플라이트** | 본 문서 `check_prereg_preflight` rc=0(§14 기재) + 발주 시 `check_dispatch_preflight` rc=0 | 게이트 자신의 `--selftest`(NC-P0·NC-D0 포함, 기존) | 무 |
| **G-S 판독기 selftest** | `--selftest` 전건 — 성한 합성 픽스처 PASS ∧ 주입 전건 이름 있는 FAIL ∧ 복원 PASS. NC 최소 집합: **NC-S1** 배너 위조(`off (identity)` 주입→FAIL) · **NC-S2** SUMMARY 절단/`complete=1` 결손 · **NC-S3** zero-행 1행 주입(승격 픽스처에 hazard 행 삽입→E-4 FAIL 검출) · **NC-S4** env 델타 초과 키 주입(G-2 FAIL) · **NC-S5** 계급 이동표 오귀속 주입 · **NC-S0** 성한 픽스처 PASS | 이 게이트 자신 | 무 (합성) |
| **G-RC ★전항 보정(판독기 자격)** | 봉인 R-II·census 를 넣어 F1b 등록값 **bit-재현**: ff=95/885 · BL 95·EXP 284·MIX 506·NEG 46 · EXP 최소 d `2.928690524872657e-6` · zero-행 221,382×2 · 앵커 4필드×2(`tau_validity` 정수 문자열 + `n_upper`·`A_ul`·`emission_per_sr`의 `float.hex()`) — 전건 일치해야 판독기가 후항 측정 자격 획득 (잣대부터 감사) | 변조 사본(스크래치, /gpfs) 1바이트 훼손 → 이름 있는 FAIL | **유** — R-II·census read-only |
| **G-1 스테이징 무결성** | 실복사·복사본 sha256 = §7-1 등록값 · 하드링크 불변량(공유 inode 0) · §6-3의 census·R-II RUN_SEAL manifest 사전 검사 전건 PASS | 스테이저 selftest 의 하드링크 주입 팔(전례 계승: `stage_a210_z2a_probe.sh` 의 주입-검출-복원 3단) | **유** — 봉인 read-only 복사원 |
| **G-2 구성 계약** | 새 런 exports 실측 diff(대 봉인 R-II) = 등록 키 집합 **정확**(B-2: `LUMINA_SUPER_LEVELS` + 경로 6키 / B-1: + capture 2키. 초과·누락 0) + `verdict_gate.py --intended-diff` rc=0 | NC-S4 + verdict_gate 자체의 초과-키 FAIL 경로 | 무 (exports 파일만) |
| **G-3 배너·모드 계약** | 각 런 stdout: `[NLTE] Super-levels: ACTIVE (13015 FL -> 1173 SL across ions)` 정확 1행 ∧ `off (identity)` 0건 ∧ `[ARTIS super-cutoff] K=100: 21919 levels lumped` 존재 ∧ `[A2-07] within-super-level blocked` 0건 | **실물 음성대조 공짜**: 봉인 R-II stdout 을 넣으면 FAIL(`off (identity)` 실재 [저자 실행 :236]) | 유(대조군만) — read-only |
| **G-4 완주** | 각 런: `model.rc=0` · RUN_FOOTER `schema=LUMINA_DET_FLIGHT_V1` · SUMMARY 정확 2건 각 `complete=1` · R7 `te_generation` iter0/1 각 1건 · `[PHYSICS-COMPARISON] … COMMITTED` (봉인 R-II 마커 전례 거울) | NC-S2 | 무 |
| **G-5 hazard 후항 판독** | B-1·B-2 zero-스트림(`[A2-10][LINE-SATURATION-ZERO-OPACITY-ROW]`)·SUMMARY 카운터 계수 → §11 E-4 분기표 적용. 잔여 >0 이면 **행 단위 귀속**: 해당 (Z,ion,l,u) 의 덱 g·E 재검(등E-bit 이면 §4 모순 = SLB-R)·비등E-bit 이면 셸 T_e 에서 exp 쌍 재계산으로 bit-충돌 확증(확증 실패 = SLB-R) | NC-S3 | 무 (새 런만) |
| **G-6 F1b-형 후항 + 이동표** | B-2 를 G-RC 자격 판독기로 재측정 — 계급 census·d-분포·ff·§7-2 ①이 정의한 **BL∩FPC 95행 이동표 + 별도 NEG 46행 추적표**와 BL·NEG 잔여를 산출 → §11 E-5 분기표 | NC-S5 | 무 (전항은 G-RC 에서 이미) |
| **G-7 결정론 미러 + 봉인 사후** | B-1 ≡ B-2: SUMMARY 전 필드 문자열 동일(양 반복) ∧ §7-2 ③의 등록 산출물 9파일별 SHA-256 동일(원바이트·무정규화·같은 상대경로끼리) · 종료 시 §6-3의 checksum 검증 가능 양 봉인(census·R-II) RUN_SEAL manifest 사후 검사 전건 PASS | 미러: B-1/B-2 가 서로의 음성대조(불일치 = SLB-F-D) · 봉인: G-RC 의 변조-사본 팔이 검사기 유효성을 담보 · L6/L6C는 §6-3의 명시적 미검증 한계 | **유** — census·R-II 봉인 지문 재검(read-only) |

---

## §10. 런 발주 경로

- **제출 선행조건** (전건 충족 후에만 sbatch):
  1. ⚠별건 판정런 **R-III(325913)·R-IV(325914) 종단 후** — 양 작업 PENDING 실측 [저자 실행:
     `squeue`/`sacct`, syntax]. 이 단의 런은 그 뒤다(발주 정책).
  2. ⚠**계측 수리 단(DET-INSTRUMENT-REPAIR, Codex 가동)의 트리 변조 종료(폐합 커밋) 후** —
     「src-편집/트리-변조 태스크 동시 1개」 규약. 의존의 성격은 §15-3(순서 의존이지 산출 의존
     아님).
  3. 발주 시점 「가동 중 변조 태스크 0」 재확인 + 검수 Q0 답 (개정16 ②).
- **sbatch 파라미터** (봉인 실측 계승 [저자 실행 sacct]): **`--partition=a100` 한정**
  (★h100/h200 교차 제출 금지 — bit-앵커 계보가 a100×2 위라 파티션 순서 규약(h100·h200 선호)의
  **등록 예외**다; 근거 = 전항과의 계열 통제 §7-1) · `--gres=gpu:2` · `--nodes=1` ·
  `--mem=64G` · `--time=08:00:00`(봉인 상한 04:34:36 의 ~1.75×) · job-name **`LUMINA_SLB_I`**
  (B-1)·**`LUMINA_SLB_II`**(B-2)·(발화 시 `LUMINA_SLB_III`). ⚠grammar 불량노드 `--exclude` 는
  **붙이지 않는다**(LUMINA=syn 클러스터 — grammar 노드명 exclude 는 제출 거부를 만든다).
- **런 횟수 — 명시**: **기본 2회(B-1·B-2, 상호 독립이라 동시 제출 가능) + SLB-F-D 분기
  전용 예비 1회(B-3) = 상한 3회.** 그 밖의 분기에서 추가 런 금지 — 재발주는 재사전등록을
  요구한다.
- 런루트: `/gpfs/kjhan/lumina/superlevel_b_a100x2_k36/slb_<UTC>Z_{census|probe}_ion1` —
  **새 런루트**(봉인 불가침, §6-3). `OWNER.txt` 각 런루트(스테이저 각인 — §8).
- 발주 3요건 충족: 기전 오프라인 특정 = §2 [저자 실행] · 수리안 오프라인 검증 = §4 덱 전수
  재실측 + §2-3 수용 사슬 · 기대치 사전등록 = §11. 판정런은 위 2(+1)회가 전부다.

---

## §11. 기대치·분기 사전등록 (라벨 명시 부여 — UNRESOLVED 금지)

**등급**: 1 = 대수·덱 산술·봉인 실측 필연 / 2 = [추정] 요소 포함(이탈은 분기표로 흡수).

| # | 기대치 | 등급 |
|---|---|---|
| E-1 | 스테이징·구성: G-1·G-2 전건 (§7-1 통제표가 열거한 sha 3종(levels.csv·line_list.csv·바이너리) 일치 · env 델타 키 집합 정확) | 1 |
| E-2 | 배너: `ACTIVE (13015 FL -> 1173 SL across ions)` 정확 1행/런 · `off (identity)` 0건 · `[ARTIS super-cutoff] K=100: 21919 levels lumped` 불변 · `[A2-07] within-super-level blocked` 0건 · FL/SL 차원 **불변**(레이아웃은 super_mode 무관 — §2-1 ②·§4 산술 재현) | 1 |
| E-3 | 완주: G-4 전건 (R7 iter0/1 위상 커밋 포함 — 봉인 R-II 전례) | 1 |
| E-4 | ★**hazard 소멸(본체)**: `zero_opacity_emitting_rows` 221,382 → **0** · `sprobe_undefined_rows` 221,382 → **0** (양 런·양 반복) · zero-스트림 행수 0 | 구조 계급 소멸 = 1 (§4 대수+덱 전수) · **정확-0 착지 = 2** (우연 충돌 잔여 [추정] — SLB-P) |
| E-5 | F1b 후항: **BL-frozen 95 → 0**(BL 잔여의 정확한 뜻은 §7-2 ①; 동결 계급 소멸 — T_e 세대 갱신(봉인 실증 2→3)이 f 를 반복 간 상이하게 만들고 S 의 f-의존이 조립·재분배 양쪽에 실재) · **별도 NEG 46 → 0**(§4 강화 2 의 단조 대수) · ff 0.10734463276836158 → **0.0** · EXP/MIX 재분포는 관측 등록(값 예측 불가 [추정]) | BL·ff = 2 (셸별 T_e bit-동결 잔여 가능 [추정]) · NEG 구조 = 1·정확-0 = 2 |
| E-6 | 결정론 미러: B-1 ≡ B-2 (SUMMARY 전 필드 문자열·산출물 sha — §7-2 ③) | 2 (새 구성의 결정론은 재확립 대상 그 자체) |
| E-7 | 새 앵커 bake 산출 4종(§6-2) 등록 — 관측, pass/fail 아님 | — |
| E-8 | checksum 검증 가능 양 봉인(census·R-II) 사후 무변조 (G-7·§6-3). L6·L6C의 런 전체 내용 무결은 manifest 부재로 미검증이며 주장하지 않음 | 1 |

**상위 분할** (PF-2 기계 스윕 대상 — 하위 라벨은 산문 분기):

| 라벨 | 판정식 | 처분 |
|---|---|---|
| **SLB-A** (승격 성립·예측 적중) | G 전건 PASS ∧ E-4 정확-0 ∧ E-5 의 BL·NEG 0 | 폐합 절차(판정 → 감리-A·B) · 새 기준선 등재 · §15 후속 회복 개방 |
| **SLB-PR** (성립·잔여 있음) | G 전건 PASS ∧ (E-4 또는 E-5 에 잔여 >0) — 산문 하위: **SLB-P** = 잔여 전 행이 G-5/G-6 귀속 절차로 이름 있는 기전(exp bit-충돌·셸 T_e bit-동결)에 전수 귀속 → 관측 병기 폐합 / **SLB-R** = 1행이라도 귀속 불능 → **기전 이해 반증**(설계 §4-B falsifier 문면) — BLOCKED·user 에스컬레이션·재귀속 단 | P: 폐합 / R: BLOCKED |
| **SLB-F** (게이트 실패) | 어느 G-* 이름 있는 FAIL — 산문 하위: **SLB-F-D** = G-7 미러 실패 → **B-3 예비런 1회 발주 자격**(그 결과로 비결정론 vs 스테이징 사고 판별) / 그 외 = 사유 특정 후 재작업(런 소모 없는 것 우선) | 사유별 재작업 또는 BLOCKED |
| **SLB-X** (분할 밖) | 위 어느 판정식에도 들지 않는 산출 | 라벨 명시 부여 후 정지·보고 (침묵 금지) |

**분기별 함의 등록** (★예측이 빗나가면 무엇을 뜻하는가):

- E-4 잔여 > 0 이고 해당 행이 **등E-bit**로 판명되면 → §4 의 덱 전수(0/1,537,729)와 정면
  모순 = 저자 산술 또는 덱-로더 대응의 오류 — SLB-R 강제(덱↔로더 왕복 재감사 단 개설 사유).
- E-5 의 BL 잔존이 **전 행 셸 T_e bit-동결**로 귀속되면 → 계약 위반 아님(f 는 T_e 의 함수 —
  응답할 입력이 안 변한 것) — SLB-P 로 관측 병기, «자유-T 레인 재측정» 을 후속 큐에 기재.
- E-2 의 배너가 `ACTIVE` 인데 차원이 1,173 이 아니면 → 레이아웃 경로에 미지의 분기 실재 —
  SLB-X(즉시 정지, src 재감사 사유).
- E-6 미러 실패(B-3 로도 재현 불일치)면 → 승격이 비결정론 경로를 노출 — BLOCKED·별도 단
  (K36 결정론 계약의 승격-구성 재검증) 개설 사유.

---

## §12. 처분 원칙 / 하지 않는 것

1. **src/tests/Makefile 접촉 0 · 재빌드 0**(봉인 바이너리 사본 재사용 — §7-1).
2. **타 단 재판정 0**: Z-O·BITLOCK·Z-2A·F1b·L6·L6C·A207·K36 전부 불변(§6-1). 이 단의 산출은
   그 판정들의 「구성 한정」 밖에 새 구성을 **추가**할 뿐이다.
3. **형상 무변**: A안·E안 계급의 어떤 조치도 없음(§3-4). CMFGEN 관례 나머지(W·INT_SEQ) 구현
   없음.
4. **스펙트럼 판정 없음**: 주된 경로는 배선·물리 검사 — 스펙트럼 산출물은 sha·보존만(관측).
5. clamp/floor/cap/jitter/repair **0** — 이 수리는 값을 만들지 않고 선언을 완성한다. 판정식에
   허용오차 신설 없음(bit 비교 유지).
6. 봉인 **읽기 전용** 절대(하드링크 금지·지문 검증 전후) · `git add -A` 금지 · 커밋은 폐합 시
   운전석(계약 하나=커밋 하나) — **본 문서도 발주 전 커밋 금지**(운전석 지시).
7. 발견의 처분 = 조용한 대장 기재(§3-3 의 소스 함정 잔존 포함 — 이 단의 범위를 늘리지 않는다).
8. 등록 상수 밖 값 변경 금지 · 신규 2파일 밖 수리 금지 · GPU 런 상한 3회 초과 금지(재발주 =
   재사전등록).
9. 오라클(CMFGEN 자체런) 정량 인용 금지 — 본 단의 잣대는 전항 봉인·덱 산술·자기 일관성뿐.

---

## §13. 프리플라이트 선언 블록

⚠PF-1 은 발주 전 시점 술어를 겸한다: 표지 심볼은 지금 저장소에 **0파일**이어야 하고(그래서
`expected_extra` 두 계획 파일 = 표), 구현 착지 후에는 정확히 그 2파일이어야 한다.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 — SLB (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "scripts/[a-z0-9_]+\\.(?:py|sh)",
    "symbol": "SUPERLEVEL_B_PROMOTION",
    "roots": ["scripts"],
    "expected_extra": ["scripts/stage_superlevel_b.sh",
                       "scripts/analyze_superlevel_b.py"]
  },
  "branches": {
    "regimes": [[0.0, 0.45], [0.9, 1.1], [10.5, 60.0]],
    "metrics": {
      "g_fail": "sum(1 for x in v if x < 0.5)",
      "hz": "sum(1 for x in v if x > 10.0)"
    },
    "rules": [
      {"name": "SLB-F", "predicate": "g_fail > 0"},
      {"name": "SLB-A", "predicate": "g_fail == 0 and hz == 0"},
      {"name": "SLB-PR", "predicate": "g_fail == 0 and hz > 0"}
    ],
    "adversarial_fixtures": [
      {"name": "all-gates-pass-hazard-zero", "mix": [[1.0, 10]]},
      {"name": "hazard-residual-nonzero", "mix": [[1.0, 9], [42.0, 1]]},
      {"name": "gate-fail-with-hazard", "mix": [[1.0, 8], [0.3, 1], [42.0, 1]]}
    ],
    "residual": "SLB-X"
  },
  "references": [
    {"path": "scripts/run_det_convergence_2026-08-08.slurm",
     "flags_existing": ["LUMINA_SUPER_CUTOFF", "declared super-level topology"]},
    {"path": "src/lumina_atomic.c",
     "flags_existing": ["set LUMINA_SUPER_LEVELS=1 to activate"]},
    {"path": "src/lumina_plasma.c",
     "flags_existing": ["LUMINA_SUPER_LEVELS", "within_sl_frac", "SOLVE_OF"]},
    {"path": "src/population_contract.c",
     "flags_existing": ["super_mode"]},
    {"path": "src/env_universe.h",
     "flags_existing": ["LUMINA_SUPER_LEVELS"]},
    {"path": "scripts/verdict_gate.py",
     "flags_existing": ["--intended-diff"]},
    {"path": "docs/DESIGN_SUPERLEVEL_DISPOSITION_2026-08-28.md",
     "flags_existing": ["LUMINA_SUPER_LEVELS=1", "1,537,729"]},
    {"path": "docs/VERDICT_L6C_REDISPATCH_2026-08-28.md",
     "flags_existing": ["C-M-S", "95/885"]},
    {"path": "docs/VERDICT_A210_ZERO_OPACITY_2026-08-26.md",
     "flags_existing": ["A208_EXACT_ZERO"]},
    {"path": "docs/RUNG_INSTRUMENT_REPAIR_2026-08-28.md",
     "flags_existing": ["DET-INSTRUMENT-REPAIR"]}
  ]
}
```

### 개정 1 — 정의 공백 4건 보완 (2026-08-29, Codex 한시 승계)

발주 1차 중단 뒤 **GPU 런 0회·구현 산출 0·결과 미관측** 상태에서 §14의 네 계약 결함만
보완했다. ① §6-2 새 앵커의 선정 규칙·4필드 이름·필드별 표현 기준을 선행 판독기 두 본의
역사적 계약에서 명시했다 ② §7-2 ③/G-7의 미러 대상을 값 산출물 9파일로 열거하고 SHA-256을
원바이트·무정규화·동일 상대경로 비교로 고정했다 ③ E-1의 `sha 4종`을 §7-1의 덱 2 + 바이너리
1에 맞춰 `sha 3종`으로 정정했다 ④ F1b 정본 JSON의 독립 축을 따라 95행 모집단을
`BL∩FPC`로, NEG 46행을 서로소 별도 모집단으로 갈라 G-6의 표·잔여 피연산자를 정의했다.
§11의 수치 기대·상위/하위 분기 판정식과 §13 preflight 분기 규칙은 바꾸지 않았다. 이 개정은
관측 후 통과를 위한 기준 조정이 아니라 런 전 정의 폐합이다.

### 개정 2 — G-7 봉인 사정거리·E-1 지목 폐합 (2026-08-29, Codex 한시 승계)

발주 2차 중단 뒤 **GPU 런 0회·결과 미관측** 상태에서 남은 정의 공백 2건만 보완했다. ① G-7의
런 전체 checksum 검증은 실존 RUN_SEAL manifest와 정본성 판별 기준을 갖춘 census·R-II 2곳으로
좁혔다. L6·L6C의 절대 root는 확정했으나 두 곳의 `L6_STAGE_SEAL.txt`를 런 전체 checksum
manifest로 승격하지 않았고, 그 역사적 내용 무결은 미검증 한계로 명시했다. 봉인 내부 파일은
생성·교체하지 않았다. ② E-1의 sha 3종을 §7-1 통제표와 그 열거값
(levels.csv·line_list.csv·바이너리)에 직접 지목했다. §11의 SLB-* 판정식과 §13 preflight 분기
규칙은 바꾸지 않았고, 구현 초안 2파일·`src/`·봉인·GPU 런에는 접촉하지 않았다. 관측값을 본 뒤
통과하도록 기준을 맞출 자리도 사용하지 않았다.

---

## §14. 집행 기록 (★장부 정본 — 판정문·감리문은 이 절을 지목만 한다)

### 정정 1 — §7-1 등록 digest 오기 (2026-08-29, ★**운전석 정정** — user 지시)

**경위**: 검수 통과·G-RC 통과 후 **B-1 스테이징(=G-1)** 이 fail-closed 로 죽었다:
`SUPERLEVEL_B_STAGE_FATAL reason=REGISTERED_LEVELS_DIGEST_MISMATCH`.

**실측 대조** [운전석 실행]:

| 파일 | 계약 등록 (구) | 봉인 실측 | 판정 |
|---|---|---|---|
| `input/lumina_cuda` | `39d452e4*aa42` | `39d452e4dec7…c5aa42` | OK |
| **`input/model/levels.csv`** | `b3645052*9f67` | `b3645052d00d…**0daf67**` | ★**불일치** |
| `input/model/line_list.csv` | `cbd81c4f*54b5` | `cbd81c4f9b3b…5554b5` | OK |

★**봉인 전체에서 `9f67` 로 끝나는 digest 는 0건** — 타 파일 혼동이 아니라 **접미 전사 오류**다.
봉인은 무죄(`RII_RUN_SEAL` 135행 `non_ok=0`, 실물 = 등재값). 구현도 무죄 — 계약 축약을
glob `b3645052*9f67` 로 충실히 옮겼고 **fail-closed 로 옳게 죽었다.**

**정정** (축자): `OLD: b3645052…9f67` → `NEW: b3645052…daf67` (§7-1·§7-2 2곳).
검증: diff **+2/−2 · 2헝크 · 순삭 0** · §11 분기 표 **byte-동일** ·
`check_gate_referents` rc=0 · `check_prereg_preflight` rc=0. 런 0회·결과 미관측 ⟹ 골대 이동 아님.

★**역할 이탈 기재**: 계약(사전등록) 저자는 개정14 ① 상 운전석이 아니다. **user 가
「운전석에서 정정으로 하자」고 지시**해 운전석이 직접 고쳤다. 규범 창설이 아니라 **사실
오류의 실측 정정**이며 판정식 무접촉이지만, **계약의 사실 주장을 운전석이 바꾼 것**은 사실이다.

★**신설 게이트가 못 잡는 계급**: `check_gate_referents` 는 참조 대상의 **실존·계수**를 보지
계약에 적힌 **해시값의 진위**를 보지 않는다. 이 계급을 막으려면 「계약이 인용한 digest 를
봉인 manifest 와 대조」하는 검사가 필요하다 — 대장 후보.

---

### 발주 2차 — **중단** (2026-08-29) · ★**트리거 발동 (2회 연속)**

| 항목 | 값 |
|---|---|
| 발주 프리플라이트 rc | **0** · 사전등록 프리플라이트 rc **0** · 위생/정본 게이트 rc **0** |
| 진척 | 구현 초안 **2파일 착지**(`stage_superlevel_b.sh`·`analyze_superlevel_b.py`), 음성대조 다수 시연(G-0 prereg NC-P 9·dispatch NC-D 10, G-1 `SHARED_INODE` 주입 등) |
| **중단 사유** | **계약 결함 1건 — G-7 봉인 정의 공백** |
| **중단 주인** | **사전등록** |

**결함**: G-7 이 「봉인 **4곳** `sha256sum -c` non-OK 0」을 요구하나, 정본은 **R-II·census 의
manifest 만 식별**한다. **L6·L6C** 는 (봉인 root · checksum manifest 상대경로 · manifest 정본성
판별 기준) 셋 다 미정이다. 운전석 독립 확인 [실행]:

| 봉인 | manifest 실존 |
|---|---|
| census | `ZO_RUN_SEAL.sha` ✔ |
| R-II | `RII_RUN_SEAL.sha` ✔ |
| **L6** | **없음** — `L6_STAGE_SEAL.txt` 뿐 |
| **L6C** | root 절대경로가 계약에 미지목 |

⟹ G-7 은 **문면 그대로는 충족 불가능**하다. 선행 L6 스테이저가 `sha256sum -c` 용 manifest 를
만들지 않는다. (Z-2A 감리-B(6)도 같은 것을 관측했다 — 「L6 루트에 `*RUN_SEAL*.sha` 부재로
run-wide 재현 불능」.) 구현자는 임의 manifest 선택을 계약 충족으로 간주할 수 없다고 판단해
멈췄다 — 옳다.

★**두 중단의 공통 계급**: 계약이 게이트 판정식에서 **참조하는 대상이 실물로 확정되지 않음**
(1차: 4필드 미열거·sha 대상 부재·계수 불일치·축 혼동 / 2차: 봉인 manifest 부재).
**사전등록 프리플라이트는 두 번 다 rc=0** — 정적 참조는 성하고 **의미가 비어 있는** 계급이라
현 게이트가 원리적으로 못 잡는다(대장 등재 한계 그대로).

---

### 발주 1차 — **중단** (2026-08-29)

| 항목 | 값 |
|---|---|
| 발주 프리플라이트 rc | **0** (`no-restatement` tokens=5 · `frontpage-sealed` · `contract-preflight rc=0`) |
| 사전등록 프리플라이트 rc | **0** (4항 전건) |
| 운전석 게이트 | `check_driver_hygiene` rc=0 · `check_doc_canonical` rc=0 |
| 가동 중 변조 태스크 | 0 (★`pgrep -f` 자기매칭·타 사용자 프로세스를 걸러 확인) |
| **중단 사유** | **계약 결함 4건 — 정의 부재/축 혼동** |
| **중단 주인** | **사전등록**(운전석 발주서 아님 — 뒷겹은 계약 값을 재서술하지 않았고 프리플라이트가 확인) |

**Codex 가 지목하고 운전석이 독립 확인한 4건**:
1. §6-2 「최저 line-id FPC 행 iter0/1 **4필드** hex」 — 그 4필드가 문서에 **열거되지 않음**.
   신규 판독기는 §8 이 「도구 import 0」으로 못박아 승계도 불가.
2. G-7 「**등록 산출물 sha** 동일」 — 비교 대상 산출물 목록·sha 기준 **부재**.
3. E-1 「**sha 4종** 일치」 대 §7-1 통제표 — 표의 sha 통제는 덱 2 + 바이너리 1 = **3종**.
4. §7-2 「**BL∩FPC 95행(NEG 46 포함)**」 — 축 혼동. `f1b_result.json` 실측상
   `d_distribution_by_class` 는 **BL/EXP/MIX**(BL=95)이고 **FPC/NEG 는 별개 축**이다
   [운전석 실행: JSON 직독].

★**프리플라이트가 못 잡은 계급**: 대장 등재 한계 「정적 참조가 아닌 **의미 검사**는 못 잡는다」
그대로다. 4건 모두 참조는 성하고 **뜻이 비어 있다**.
★**구현 산출 0 · `src/` 0 · 봉인 0 · GPU 런 0.** worktree clean 복귀 확인.
★**이 시점 개정은 골대 이동이 아니다** — 런 0회, 결과 미관측 상태의 정의 보완이다
(Z-2A 사고는 결과를 본 뒤의 개정이었다).

**필수 4항목**:

1. 발주 프리플라이트(`check_dispatch_preflight`) rc: __ (운전석 기입)
2. `[미실행]` 주장 실행 결과: **본 사전등록의 [미실행] 주장 0건** — 전 사실 주장 저자 실행
   (§16). 운전석 추가 실행분: __
3. 분장 장부: 아래 표 (검수 완료 후 기입)
4. 발주 중단 유무·사유·주인: __

**사전등록 프리플라이트 (G-0 전반부)**: `python3 scripts/check_prereg_preflight.py
docs/RUNG_SUPERLEVEL_B_2026-08-28.md --root .` (집행은 절대 경로 형 — §16 C12) → **rc=0**
(`changeset-consistent files=2` · `branch-partition trials=200000 rules=3 fixtures=3
residual=SLB-X` · `references-resolved refs=10`) [저자 실행, grammar-debug, 2026-08-28]

**분장 장부** (개정14·16·17 — 운전석 칸 비움):

| 단계 | 규약상 담당 | 실제 | 위반 |
|---|---|---|---|
| ① 사전등록 | Fable (개정14 ①) | Fable — 초안 (2026-08-28) · Codex — 개정 1 정의 보완(2026-08-29, 한시 승계) | 없음 |
| 발주서 작성·발주 행위 | 운전석 (앞겹=본 문서 그대로 첨부·뒷겹 재서술 금지 — 개정15) | | |
| 발주 프리플라이트 집행 | 운전석 (grammar-debug) | | |
| 구현 (신규 2파일) | Codex | | |
| ② 코드 검수 (Q0 선행 — 개정16 ②) | Fable fresh | | |
| 게이트 집행 — 봉인 무접촉 (G-0·G-S) | 운전석 (Q0 답 전 병행 가능) | | |
| 게이트 집행 — 봉인 접촉·런 제출 (G-RC·G-1~G-7·sbatch) | 운전석 (★Q0 답 후·§10 선행조건 충족 후) | | |
| ③ 판정 | Fable fresh | | |
| ④ **감리-A (물리·주장 — 개정17)** | Fable fresh (①③ 과 다른 컨텍스트) | | |
| ★**감리-A 발주 집행 주체** | **운전석** (Agent 위임 — CLAUDE.md 상시 승인 ④) | | |
| 감리-B (재현 — 개정17) | Codex read-only (판정문 실측 주장 재현 한정) | | |
| 폐합·커밋 | 운전석 (감리 A·B 둘 다 통과 후 — 개정17 폐합 조건) | | |

---

## §15. 승계 기재

1. **개설 근거**: BITLOCK 미결 #4 → 설계 `ddf6701` 갈림길 → **user B안 승인**. 설계 §8 권고의
   「11이온 재감사 단(#3)과 한 묶음」 중 **본 단은 수리·차동 절반만** 집행한다 — 재감사 단
   (§6-1 생존/사망 목록의 대장 반영·클램프 무죄 8건 재확인·[TAU-BY-ION] 류 절대 관측치 재확인)
   은 본 단 폐합 후 별도 개설(개설은 user·운전석 몫).
2. **F1b 와의 접속**: F1b 판정문이 «수리 후 재측정의 前항» 으로 설계한 그 소비가 §7-2 ①이다.
   F1b 의 C-M-S 판정·주장 상한은 불변(§6-1).
3. **계측 수리 단(DET-INSTRUMENT-REPAIR)과의 의존 — 판정**: **산출 의존 없음·순서 의존만.**
   근거: ① 본 단의 판독기는 신규 파일 + 도구 import 0 자체 파서 — 그 단의 두 판독기
   산출물을 소비하지 않는다 ② 본 단의 런은 봉인 바이너리 사본이라 **트리 재빌드 자체가 없다**
   ③ 파일 교집합 0. 남는 것은 「트리-변조 태스크 동시 1개」 규약뿐 — §10 선행조건 2 로 집행.
   그 단이 실패해도 본 단 게이트는 성립한다(sacct 무소비 설계 — §9 머리).
4. **별건 판정런 R-III(325913)·R-IV(325914)**: PENDING 실측 [저자 실행] — 본 단의 sbatch 는
   그 종단 후(§10). 물리·파일 상호작용 없음(그 두 런은 Z-2A 계보 후속).
5. **후속 회복 (SLB-A/SLB-P 시)**: ① **L6C-COVER 분기 D2 의 차단 전제 해소** —
   `INDEPENDENT_SPROBE_UNDEFINED`(=`sprobe_undefined_rows`) 가 반상태 hazard 와 동일 모집단
   임이 봉인 실측(221,382=221,382 [저자 실행])이므로, E-4 적중 시 A210-ZERO-OPACITY 가 요구한
   «Z-O 수리 선행 조건» 이 충족되어 **L6C 계약 본체(solve 의 J̄ 응답)** 재개 가능 — 그 발주는
   별도 단(user·운전석 몫) ② 본선 다음 단(수렴·자유-T)이 새 기준선 위에서 발주 가능(§6-4
   앵커 부채 복리의 정지) ③ D안(배너·헤더 정직화)·로더 CUTOFF-단독 함정 수리 — 별도 단 후보로
   대장 존속(§3-3).
6. **죽는 앵커의 대장 처리**: §6-1 사망 목록의 대장 행들은 **삭제하지 않는다** — 「반상태
   구성 한정 기준선(승격 후 비교 불가)」 를 병기하는 것은 재감사 단의 몫으로 이월.

---

## §16. 저자 실행 명령 원장 (전건 2026-08-28)

| # | 명령 (요지) | 결과 |
|---|---|---|
| C1 | `git log -1` · `git diff af7441d HEAD -- src/` · `git status --short -- src/` | HEAD `ddf6701`·src diff 공백·작업트리 src clean — 설계·BITLOCK 행 번호 유효 |
| C2 | `sed -n`/`grep -n` 소스 직독: `lumina_atomic.c:1438-1482` · `lumina_plasma.c:16344-16487·16985-17000·19376-19446·19622-19660·19820-19845` · `population_contract.c:190-262` · `lumina.h:718-730` · `env_universe.h:489-490` · `getenv` 전수 grep | §2 전 행 번호 확정 (반상태 사슬·f 생산 사슬·재정규화·env 등재) |
| C3 | awk `#if 0` 깊이 오토마톤 (plasma 1200-1800·cuda 1000-2100) + cuda 의 super 소비행 grep | 死블록 정확 1324-1693·1083-2063 · ★cuda 소비행 4곳 전부 死블록 내부(§2-5) |
| C4 | 봉인 read-only: `<RII>`·`<ZO>` exports 전수·상호 diff · RUN_FOOTER · stdout 배너(`:129`·`:236`) · `<RII>/input/job.slurm:265-282` · RUN_SEAL 덱 해시 grep | §2-4 표 전건 (env 실질 차=capture 2노브·경로 6키 / 배너 off (identity) / CUTOFF==100 단독 / 덱 해시 양 봉인 동일) |
| C5 | `sacct -j 324388,324557,321636 -X`(syntax 로그인) · `squeue`/`sacct -j 325913,325914` | 봉인 job 실측(02:42:56·04:34:36·ReqMem 64G·Timelimit 8h·a100 계열 syn101/102) · R-III/R-IV PENDING |
| C6 | ★덱 전수 census 재실측 — grammar-debug(nested ssh, stdin 파이프): levels.csv 를 (Z,ion,level_num)→(g,E)로 적재, line_list.csv 2,588,798행 스트리밍, 인벤토리 12슬롯·양준위 rank≥100 판정, 등g·등E(bit)·E 단조·럼프-내 최대 ΔE 계수 (스크립트 사본: 세션 스크래치 `deck_census_slb.py`) | §4 표 전건: 1,537,729 / 539,149 / **0** / 12슬롯(11절단+Ca II 62준위) / FL 13,015·SL 1,173 / 커버 등g 325,489 / 등E-bit(g 무관) 0 / E 비단조 0 / max ΔE 18.864 eV |
| C7 | `<RII>`·`<ZO>` stderr 에서 `[LINE-SATURATION-SUMMARY]` 전 행 추출·diff | 값 문자열 동일(차이=`sprobe_undefined_rows` 필드 유무 1건) — §7-2 ③ 미러 기질 실증·hazard 221,382 양 반복 재확인 |
| C8 | `/gpfs/kjhan/lumina/zo_artifacts/f1b_result.json` 직독(전 필드 워크) | F1b 前항 등록값 §7-2 ① (BL 95·EXP 284·MIX 506·NEG 46·ff 0.10734463276836158·min d 2.928690524872657e-6·anchor hex×2·RD-1 SUMMARY 양 반복) |
| C9 | `analyze_a210_zero_opacity_census.py:17-18` 직독 · src `A208_EXACT_ZERO` grep | zero-스트림 행 prefix·SUMMARY prefix 확정 · A208Validity enum 실재 |
| C10 | `docs/VERDICT_PROTOCOL.md` 전문 · L6C-R 판정문 「V5 원상 유지」 불릿 · BITLOCK 「V5 권한」 행 grep | §3-2 용어 판정 (V5 = 대장 갱신 vs 캠페인 관용 = src 무변조) |
| C11 | `grep -rl SUPERLEVEL_B_PROMOTION scripts/ src/ docs/` · 선언 블록 flags 전건 존재 grep · `stage_a210_z2a_probe.sh` 하드링크 불변량·OWNER 각인 행 grep | 표지 심볼 0파일 · PF-3 참조 전건 실재 · 스테이저 전례 확인(§8) |
| C12 | grammar-debug(nested ssh): `python3 /home/kjhan/BACKUP/…/Lumina-sn/scripts/check_prereg_preflight.py /home/kjhan/BACKUP/…/Lumina-sn/docs/RUNG_SUPERLEVEL_B_2026-08-28.md --root /home/kjhan/BACKUP/…/Lumina-sn` | **rc=0** — PF-1 files=2 · PF-2 trials=200000 rules=3 fixtures=3 residual=SLB-X · PF-3 refs=10 (§14 기재) |

*(C4·C7·C8 의 봉인 열람은 read-only(grep·sed·파이썬 직독) — 로그인 노드 비연산 범위이거나
grammar-debug 집행. `<ZO>`/`<RII>` = §2-4 표의 런루트.)*
