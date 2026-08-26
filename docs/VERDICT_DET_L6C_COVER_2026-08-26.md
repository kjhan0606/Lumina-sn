# 판정문 — DET-L6C-COVER 판정런 (job 322449) — 분기 **D2** 발화, 부분 착지

**요약** — ① 판정런은 iter=0 의 REQUESTED_TE 국면에서 `INDEPENDENT_SPROBE_UNDEFINED` 로
차단됐고, 이는 사전등록 §6 이 처분까지 등록해 둔 분기 **D2 그대로**다 — 계약 본체(R6:
A2-07 solve 의 J̄ 응답)는 **판정되지 않았고 미결로 남는다**. ② 이 단의 실질 산출 = PC-2
오프라인 예측(`ion=1 hazard=0`)의 in-vivo 반증 + ZERO-OPACITY WITNESS 1행이며, 판정자
재측정으로 기전을 규명했다: **PC-2 의 검출 대수는 옳았고(판정자 독립 census 가 후보 수까지
정확 재현), 빗나간 것은 "iter0 물질 = LTE-at-10020" 이라는 계약 자신의 [추정] 전제다** —
실제 물질은 시드-예측자의 NLTE solve 커밋분(stderr:419, `population_generation 1→2`)이고,
증인 행은 그 ladder 에서 `n_lower ≡ n_upper` **bit-동일**(g_l=g_u=8)이 만든 χ=0 정확
상쇄다. ③ 운전석 집행 기록(§13-5)의 해석 2건을 정정한다: "총 1행" 은 **중도절단 하한(≥1)**
이고, "inversion 경계가 아니다" 는 **역이다** — 이 증인은 S4-III 의 [소스 추론]을 **확증**한다.

## 0. 판정자·원칙

- 판정자 = Fable **fresh 컨텍스트**(분담 개정14 ③). 운전석·검수자 서술은 신뢰하지 않았고
  **판정 하중 항목 전건을 자체 경로로 재측정**했다(부록 A). 도구 import 0 — PC-2 앵커는
  자작 census, 판독기 산출은 별도 실행으로 byte-대조.
- read-only 준수: 저장소·봉인·판정런 산출물 쓰기 접촉 0.
- **오라클 인용 0**(CMFGEN 런 `INELIGIBLE`). **clamp/floor/cap 0** — 이 판정문의 모든 수치는
  관측·대수이며 어떤 값도 수리하지 않았다.
- **V5 원상 유지**: `git diff 5d711d06..0e899cb -- src tests Makefile` 공집합 ∧
  `git diff 0e899cb..9394762 -- src tests Makefile` 공집합 [판정자 실행].

## 1. 봉인 (전건 판정자 재측정)

| 항목 | 실측값 |
|---|---|
| job | `322449` **FAILED exit=70:0** elapsed **02:37:20** node=syn102 partition=a100 gpu 2 |
| `model.rc` | **1** (`DET_FLIGHT_FATAL model exited rc=1`; stderr 말미 `[R7][FATAL] lane=DET iter=0 rc=4` → `[DET-TRANSACTIONAL][FATAL] cmfgen_run rc=4`) |
| `git_head` | `0e899cb3a82ef40ae4ba439ce359feb97b546a83` = 발주 HEAD |
| binary sha256 | `b9a30a81ebea…af99` — **런 input 과 봉인 L6 원본 양쪽 재해시, 동일** |
| stderr sha256 | 런 `433b9777…8e3d` (302,491 B·1,251행) · 봉인 L6 `19af9184…880c3e` — 재해시 일치 |
| R1 스크립트 신선성 | RUN_FOOTER 의 5 sha = `git show 0e899cb:scripts/<파일> \| sha256sum` **전건 일치** |
| **봉인 무변조** | 운전석 지문 5본 상호 동일 + **판정자가 판정 시점에 재지문 → 256/256 동일** |

⚠**지문의 계급(정직 기재)**: 봉인 지문은 **경로+크기+mtime** 이지 내용 해시가 아니다.
판정자는 내용 계급 앵커 3종(봉인 binary sha·봉인 stderr sha·두 exports 문면)을 별도
재측정했다. **내용 계급 전수 검증은 누구도 수행하지 않았다.** 후속 봉인에 내용 해시
manifest 를 권고한다(비차단).

## 2. 게이트 표

| 게이트 | 판정 | 근거 (판정자 재측정) |
|---|---|---|
| **PC-1** | **PASS** | 발주 HEAD blob 직독으로 표적쌍 (26,1)(27,1)(28,1) 실재·ion=3 부재 ⟹ 봉인 594행(전수 ion=3, 재계수 594/594) × base 0/594·×ION4 594/594 는 산술 강제. SKIP 교집합 자체 파스: 양 루트 `SKIP_Z="14"` 각 1건·OPACITY 0건 ⟹ {14}∩{26,27,28}=∅. 금지 env 0건 재계수 |
| **PC-2** | **PASS(발주 시점·등록 처분 준수)** — 단 본실행이 ion=1 예측을 반증(§5) | 판정자 **독립 census(도구 미사용)** 가 candidate {1:1,048,375 · 2:708,767 · 3:215,038}·hazard {0,0,0} 을 **정확 재현** ⟹ 게이트 산술 무결. 앵커 ① 빗나감은 등록 처분대로 이름 있는 발견 기재 후 진행 |
| **PC-3** | **PASS** | 판정자 diff: vs IDSEAL **값 delta 정확 5** · vs 봉인 L6 **정확 1**(TARGET_ION 3→1). 나머지는 런루트 경로 재배치 6건 |
| **PC-4** | **PASS** | `--selftest` rc=0, NC-B1~B5·NC-R1·R4·R7 전부 발화·복원 |
| **PC-5** | **PASS** | binary sha 재해시 일치 ∧ src diff 공집합. 사전 게이트 selftest 도 rc=0(NC-P1~P4·NC-PC5 발화) |
| **R1** | **PASS** | §1 표 — 5 sha·git_head·binary 전건 재측정 일치 |
| **R2** | **미도달** | `R7_MATERIAL_PHASE_COMMITTED` 0건 · `PHYSICS-COMPARISON` 0건 · `iter=1` 0건 [재grep] |
| **R3** | **미도달** | ROW 줄 0건; 판독기 `observed_row_lines=0` |
| **R4 / R4b** | **미도달** | iter0 ROW 자체가 없다 |
| **R5** | **미도달** | `f_mapped=None` — 판독기가 값을 **지어내지 않음**을 재실행으로 확인 |
| **R6 (계약 본체)** | **미도달 — 미결** | d 분포·ff 부재. *"A2-07 solve 가 커버 이온에서 J̄ 에 응답하는가"* 는 **여전히 열려 있다** |
| **R7** | **미도달**(게이트) · **관측(하중 0)** | `LINE-COEFFICIENT-IDENTITY` 0/100줄 ⟹ 게이트 불성립. 단 대조: **[A2-09] 접두 1,203줄 봉인 L6 과 byte-동일**, `[cmf_fine]` 17줄은 타이밍 필드 제외 시 동일, R2 exact-solve 잔차 `8.1222406993212508e-09`·반복 52·셀 109,014,300 **자릿수까지 동일**. TARGET_ION 이 사망 시점까지 물리를 안 건드렸다는 **관측**이지 PASS 가 아니다 |

판독기 산출 무결: 판정자가 `--run-root`(판정런 — 봉인 아님)·`--report`(스크래치)·
`--expected-head 0e899cb` 로 재실행 → 운전석 저장본과 **byte-동일**, rc=4
`status=PARTIAL verdict=D2 ff=None f_mapped=None`.

⚠**재현 절차 함정(감리용)**: 판독기의 기대 HEAD 기본값은 **현재 repo HEAD** 다. 판정런 뒤
커밋이 얹힌 지금 무인자 재실행하면 `R1_GIT_HEAD_MISMATCH`→D3 가 나온다[판정자 실측].
재현은 반드시 `--expected-head 0e899cb…` 명시. 운전석 원 실행은 커밋 전이라 무관 —
**결함 아님, 함정 기재.**

## 3. ★분기 판정 — §6 **D2** 발화 (축자)

```
[A2-10][LINE-SATURATION-ZERO-OPACITY-WITNESS] …
[A2-10][LINE-SATURATION-BLOCKED] reason=INDEPENDENT_SPROBE_UNDEFINED phase=REQUESTED_TE
    shell=0 candidate_rows=66110 target_ion=1 complete=0 zero_opacity_emitting_rows=1
[A2-10][LINE-SATURATION-BLOCKED] reason=UPSTREAM_LINE_SCAN_INCOMPLETE …
[A2-10][LINE-NET-BLOCKED] status=RADEQ_TERM_SCHEMA line_status=INVALID_INPUT line=669992
[A2-10][BLOCKED] reason=RADEQ_FIXED_T_BUNDLE_BUILD_FAILED rc=3
[A2-10][BLOCKED] event=R7_MATERIAL_UPDATE_BLOCKED lane=DET iter=0
    te_generation_before=1 te_generation_after=1 generation_preserved=1 action=TERMINATE
[R7][FATAL] lane=DET iter=0 rc=4
[DET-TRANSACTIONAL][FATAL] cmfgen_run rc=4
```

발화 분기 = 사전등록 §6 의:

> | **D2** | `INDEPENDENT_SPROBE_UNDEFINED` 차단 | PC-2 census 의 반증 + WITNESS 행 = Z-O 우선순위 증거. 부분 착지(계측 성립·판정 미결), 재시도는 Z-O 처분 후 |

- 차단 사유 문자열이 D2 판정식과 **축자 일치**하고 평가 순서상 유일 발화다.
- **계약이 예정한 착지다.** 단 §2-1·§8-3 은 위험을 **반복 1** 에 등재했는데 실제 발화는
  **iter=0 의 시드 물질**이었다 — D2 판정식은 위치 무관이라 분기 체계 안이되, **위치의
  빗나감 자체를 §5 에 발견으로 기재**한다.
- `RADEQ_TERM_SCHEMA` 는 `saturation_add==-1` 의 직접 전파다(`lumina_plasma.c:15199-15201`
  직독) — **진단 트랜잭션이 물리 번들을 죽인다**는 S4-III §1-2 의 기전이 **두 번째 이온에서
  재현**된 것이다.

## 4. 기대치 대조 (§6 E1~E7)

| # | 기대 | 실측 | 판정 |
|---|---|---|---|
| E1 | 각 반복 ≥30 매치 행 | **0 행** | 미달성 — D2 경로 |
| E2 | f_mapped=1.0 | None | 미도달 |
| E3 | iter0 앵커 1±1e-3 | 미측정 | 미도달 — 단 §5 증인이 "시드=정확 LTE" 가정에 **반례**를 준다(전역 판정 아님) |
| E4 | 1순위 [추정] C-R | **D2 발화** | **빗나감** — 등록 분기 내 착지. C-R/C-F/C-M 어느 것도 평가 불능 |
| E5 | 관측 5종 | ①~④ 미산출. ⑤ zero-χ census 만 부분: 증인 1행(**중도절단**)+`candidate_rows=66110` | 부분 |
| E6 | 경과 ≈04:34·MaxRSS ≈37.1 GiB 등급 | **02:37:20**(조기 종료)·MaxRSS 37,153,392K ≈ **35.4 GiB** | 경과 빗나감(사유 명백)·메모리 동일 등급 |
| E7 | R7 byte 동일 | 게이트 미도달. 관측: 사망 전 접두 무편차 | 미도달 |

## 5. ★PC-2 예측의 반증 — 기전의 실측 규명 (이 단의 핵심 산출)

### 5-1. 실측 사슬

증인: `Z=26 ion=1 (Fe II) line=669992, lower_global=2771, upper_global=5213, tau_validity=2,
tau_raw=0, n_upper=4.437688873013011e-4, A_ul=0.5222556, ν=2.234785e15,
emission_per_sr=2.7310010625545125e-16, srce_chk=0, exact_zero_provenance=0,
clamp/floor/cap/jitter/repair=0`.

1. **`tau_validity=2 = A208_EXACT_ZERO`**(`opacity_publication.h:8-9` 직독) — 불투명도 출판이
   이 τ 를 **정당한 정확 0 으로 등록**한 것이지 미기록·무효가 아니다.
2. **덱 실물**: `line_list.csv:669994` `f_lu=1.409e-10`·`f_ul=−1.409e-10` ⟹ **g_l=g_u**,
   λ=1341.48Å. `levels.csv`: E_l=6.5194 eV(**g=8**)·E_u=15.7618 eV(**g=8**) — ΔE=9.2423 eV ↔
   λ1341.5Å **정합**(덱 내부 불일치 아님).
3. f_lu>0 강제: 행이 per-shell 루프에 닿으려면 `line_net_einstein_opacity_ratio` 가드
   (`line_net_rate.c:227` — `f_lu<=0 → 거부`)를 통과해야 한다. 통과했다.
4. τ = `coeff·f_lu·λ·t·(n_l − (g_l/g_u)·n_u)`(`opacity_publication.c:26-29`). 계수부 ≈8.4e-11
   이라 `value==0.0` 정확은 언더플로 불능 ⟹ **difference==0.0 bit-정확 ⟹ n_l ≡ n_u
   bit-동일**(g 비 1).
5. 이 τ 의 공급자는 **NLTE ladder 기록자**(`lumina_plasma.c:19615`)다. LTE(10020)이었다면
   `n_l/n_u = e^{ΔE/kT} ≈ 4.4e4` 라 **상쇄 불능**이고, n_ion=0 류의 전면 0 이었다면 스캔
   순서상 앞선 **Fe II 33,133행**이 먼저 발화했어야 하는데 안 했다 ⟹ 배제.
6. ladder 의 정체 = **시드-물질 예측자의 NLTE solve 커밋분**: stderr:419
   `[A2-INIT][SEED-MATERIAL] … population_generation=1->2` [직독]. A2-INIT 주석
   (`:16081-16094`)대로 시드는 LTE 가 아니라 **공표 seed Te 에서의 1회 NLTE 물질 solve** 다.
7. **교차검증(강)**: `candidate_rows=66110` = 덱에서 증인 이전의 ion=1·Z∈{26,27,28} 행 수
   **정확 66,110**[awk] ⟹ 스캔은 증인까지 전수였고 증인이 **스캔 순서상 첫 hazard** 다.

### 5-2. PC-2 산술의 독립 재계산 — **게이트는 무죄**

판정자가 도구 import 없이 자작 census(덱 × LTE(10020) Boltzmann·단위 이온밀도·
`difference==0 ∧ η>0`)를 실행: **candidate {1:1,048,375 · 2:708,767 · 3:215,038}·
hazard {0,0,0}·f_lu==0 변종 {0,0,0}** — PC verdict JSON 과 **후보 수까지 정확 일치**.
검출 대수의 FAIL 가지도 살아 있다(NC-P3 축퇴쌍 주입 → `ZERO_CHI_HAZARD` 발화, 재실행).

### 5-3. 그러면 무엇이 반증됐나 — 게이트 결함인가, 예고된 한계인가

**둘 다 아니고, 정확히는 "계약의 [추정] 전제의 반증" 이다.**

- **게이트 결함 아님**: PC-2 는 등록된 모델을 정확히 계산했다(§5-2).
- 반증된 것은 §3-5 의 **"iter0 의 물질은 LTE-at-10020 [추정: 시드=LTE]이므로 iter0 위험은
  오프라인 재현 가능"**. 실물 시드는 NLTE solve 산출(§5-1 ⑥)이고, 그 안의 χ==0 은 — 계약이
  **iter1 에 대해서만** 인정했던 — **"오프라인 예측 불능"** 계급이었다. 계약은 이 무지를
  §8-4·§8-8 에 등재하고 D2 안전망을 걸어 뒀다 — **그 안전망이 설계대로 작동했다.**
- [추정] 시드-예측자의 존재는 소스에 있었으므로 원리상 알 수 있었다 — 맹점의 형태는
  **표적을 NLTE 권한 안으로 옮기면서 인구 모델은 LTE-공급 시대의 것을 계승**한 것이다.
  다만 **시드 solve 산출 자체는 GPU 런 없이 재현 불능**이므로 *"오프라인 census 로 이
  hazard 를 잡을 수 있었다"* 는 주장은 **성립하지 않는다.**

### 5-4. 운전석 집행 기록(§13-5)의 정정 — 해석 2건 + 확인 1건

§13 수치는 전건 재측정 결과 **수치로는 전부 일치**했다. 해석 2건을 정정한다:

1. **"총 1행" → 중도절단 하한 ≥1.** 카운터는 증가 직후 같은 호출에서 차단이 발화해 죽는다
   (`:14076`→`:14136` 직독) ⟹ **구조상 1 을 넘길 수 없고 hazard 총수는 미지**다.
   Z-O §Z-1 의 *"부분합 합산 금지"* 경고가 정확히 이 자리다.
2. **"기전이 다르다·inversion 경계가 아니다" → 역.** 실측 기전은 `n_l·g_u = n_u·g_l`
   **정확 상쇄 그 자체**이고(§5-1 ④) PC-2 가 모델링한 **대수 계급과 같다** — 다른 것은
   **인구 공급자**(LTE Boltzmann ↔ 시드 NLTE ladder)다. 그리고 χ ∝ (n_l/g_l − n_u/g_u) 의
   정확한 영점 = S4-III 가 정의한 *"inversion 경계(순 불투명도 정확 0·순수 방출)"* **그
   자체**다. 증인은 그 추론을 **반박이 아니라 확증**한다.
3. **확인**: *"표적 이온을 바꿔도 같은 결함(ion=2·ion=1 독립) — 회피 불능"* 은 지지된다.
   덧붙여 봉인 L6(ion=3) stderr 에 line 669992 흔적 **0건**[grep]: hazard 는 **표적 필터를
   통과해 A2-10 수집에 들어올 때만** 문다 — 물리 번들의 net-rate 자체(η−χJ̄, χ=0)는 계산
   가능하기 때문이다. **차단의 소유자가 진단 트랜잭션임이 세 런(S4-III·L6 침묵·L6C)의
   대조로 폐합**된다.

**신규 미결**: 시드 NLTE solve 산출에서 서로 다른 super-level(덱 SL 26 vs 128)에 속한 두
준위의 인구가 **bit-동일**이 되는 구조적 생산자. 연속 산술 solve 가 자연히 bit-동일을 낼
확률은 사실상 0 이므로 구조적 원인이 있을 것이다 [추정]. 판정런은 super_mode OFF(identity
투영) 실측. 귀속은 Z-1 census/ladder 덤프 계측의 몫.

## 6. ★WITNESS 행의 처분

요구의 계보: *"수리 전 필수: 그 행이 정말 inversion 경계인지 **실측 확인**(현재는 소스 추론)"*
의 원 출처는 **S4-III 판정문 :82** 이고, Z-O §2 는 *"왜 tau==0 인가 — (a) A2-08 이 정당한
exact-zero 로 등록했는가 / (b) 미기록(coverage 결손)이 0 으로 남았는가. **둘은 처분이
다르다**"* 로 정식화했다.

**이 증인은 그 요구를 — 표본 1행에 한해 — 충족한다.** 말하는 바:

1. **(a) 다**: `tau_validity=A208_EXACT_ZERO` — 출판이 **정당한 정확 0 으로 등록**한 행이다.
   coverage 결손(그 경우 `A208_UNSAMPLED/MISS` 로 앞서 거부)이 **아니다**.
2. **inversion 경계 맞다**: n_l/g_l = n_u/g_u 정확, 순수 방출(η>0·absorption=0). S4-III 의
   소스 추론이 **첫 in-vivo 전 필드 실측으로 확증**됐다 — 단 S4-III 자신의 ion=2 차단 행
   (line 262210)은 **여전히 미실측**이고 이번 확증을 그 행으로 **이월할 수 없다** [추정만 가능].
3. **Z-2 후보 처분에의 함의**: Z-O 표의 *"Z-1 이 '물리적으로 정당한 행' 을 보이면 **A(진단
   격리)** 유력"* 의 **전건이 표본 1행에서 성립**했다. **B(행 건너뛰기)** 병행 가능성도 열려
   있다. 그러나 **1행은 census 가 아니다** — Z-1 전수 계측 없이 Z-2 를 고르지 말라는 Z-O
   자신의 규율이 그대로 유효하다.
4. **★Z-1 설계에의 신규 입력**: hazard 는 LTE 인구가 아니라 **시드 NLTE 물질의 성질**이다
   ⟹ Z-1 census 는 **오프라인 덱 산법으로 불가능하고 라이브 커밋 물질 위에서** 돌아야 한다.
   Z-O 가 처방한 census 런 형상(`DIAG=2` + `INDEPENDENT_CAPTURE=0`)은 차단 가드가
   `independent_capture &&` 조건부이므로(`:14111` 직독) **완주할 수 있을 것**이다
   [소스 추론 — 실행 미검증].

## 7. 이 판정이 주장할 수 있는 것 / 없는 것

**있는 것 (실측)**
- D2 가 발화했고 그 처분은 사전등록이 예정한 그대로다.
- **계측 기계는 성립했다**: PC 4종·NC 전 종(재실행 rc=0 ×2)·판독기(값 무날조·byte-동일
  재현)·봉인 무변조·R1 전건.
- PC-2 반증의 기전: **검출 대수 무죄 · "시드=LTE" [추정] 전제의 반증 · 증인 = 시드 NLTE
  ladder 의 bit-동일 준위쌍이 만든 χ=0 정확 상쇄**.
- 진단 트랜잭션이 물리 번들을 죽인다는 S4-III 기전의 **ion=1 재현** + **이온 회피 불능**.
- TARGET_ION 은 사망 시점까지 물리 무섭동(A2-09 접두 1,203줄 byte-동일 — **관측**).

**없는 것**
- ★**계약 본체 R6**: A2-07 solve 가 커버 이온에서 J̄ 에 응답하는지 — **미결. 이 단은 그
  질문에 어떤 방향의 증거도 보태지 못했다.**
- hazard 총수(중도절단 — ≥1 만) · 시드 ladder 의 전역 LTE 근접 여부(반례는 1쌍뿐) ·
  bit-동일의 구조적 생산자 · S4-III ion=2 행의 기전 · R4 앵커의 성부 · σ 라벨·J̄/B 이탈 크기 ·
  C-F/C-R 어느 쪽으로도의 경향.
- *"오프라인 census 를 더 잘 설계했으면 잡았을 것"* — **성립하지 않는다**(§5-3).

## 8. 미결·다음 단 후보 (D2 처분 문언과 Z-O 규율에 종속)

1. **Z-O Z-1 계측 런** — L6C 형상(ion=1) ± ion=2/3 에서 `INDEPENDENT_CAPTURE=0`·`DIAG=2` 로
   hazard **전수 census**. 이 단의 신규 입력(§6-4): census 는 **라이브 물질 전용**, 카운터
   중도절단 규칙 유의. 시드 ladder bit-동일 쌍의 생산자 귀속(§5-4)을 겸측 후보로.
2. **Z-2 처분**(user 보류 중) — 표본 1행은 **A(진단 격리) 유력 정황**, 결정은 Z-1 census 후.
3. **DET-L6C-COVER 재발주** — D2 문언 그대로 *"재시도는 Z-O 처분 후"*. 재발주 시 사전등록
   §3-5 의 **시드 전제를 §5 실측으로 교체**할 것.
4. (관측) **PC-2 census 는 폐기하지 않는다** — LTE-공급 클래스(비커버 이온)에 대한 그 모델은
   여전히 유효하고 ion=3 무-hazard 재현이 그 증거다.

## 9. 분장 장부

**장부 = 사전등록 §13-6**(`docs/RUNG_DET_L6C_COVER_2026-08-22.md`) — 개정16 ① 에 따라
복제하지 않고 **지목**한다. 판정 단계 칸 기입:

> | 판정 | Fable (fresh) | **Fable fresh — 본 판정문. 판정 하중 전건 자체 재측정(부록 A), §13 해석 2건 정정·1건 확인** | — |

## 10. 규약 준수 명시

**V5 원상 유지**(§0). 오라클 인용 0. clamp/floor/cap 0. 봉인 무변조 실측(§1). 판독기
`--report` 는 봉인·런루트 밖으로 명시. 폐합 전 **감리 필수** — 본 판정문은 감리 대기 상태다.

**감리 권고 검사점**: ① §5-4 정정 2건의 대수 재검(특히 "inversion 경계" 처분의 **역전**)
② §2 R7 행의 "관측/게이트" 절단 유지 여부 ③ §5-3 의 "게이트 무죄·전제 반증" 구획이 과대/
과소 주장이 아닌지 ④ 판독기 재현 함정(`--expected-head`)의 재확인.

---

### 부록 A. 판정자 재측정 명령 (전부 read-only; 무거운 것은 grammar-debug nested ssh)

- 봉인·job: `sacct -j 322449 -X` · `sha256sum {런,봉인L6}/input/lumina_cuda·stderr.log` ·
  `find <봉인2루트> -type f -printf '%p %s %T@\n' | sort | diff - seal_before` → 256/256 동일 ·
  `diff seal_before seal_{after,after2,after3,final}` → 전부 동일
- R1: `for f in …; do git show 0e899cb:scripts/$f | sha256sum; done` ↔ RUN_FOOTER 5건
- 차단 사슬·계수: `grep -n 'LINE-SATURATION-BLOCKED\|R7_MATERIAL\|FATAL' stderr.log` ·
  ROW/COMMIT/`iter=1`/PHYSICS-COMPARISON 각 `grep -c` = 0 · `sed -n '1244p'`(증인 전문)
- 기전: `opacity_publication.{h,c}`(:8-9·:16-35) · `lumina_plasma.c`(:3219-3355·:8953-9010·
  :14027-14176·:14900-15040·:15150-15201·:16081-16250·:16380-16460·:19540-19640) ·
  `line_net_rate.c`(:219-246) 직독 · 덱 `line_list.csv:669994`·`levels.csv`(26,1,100)/(26,1,2542) ·
  Fe II 선행 33,133 · 표적 선행 **66,110 = candidate_rows**[awk] · 봉인 L6 `grep -c 669992` → 0
- PC 재검: **자작 census**(도구 미사용) → PC-2 정확 재현 · 두 도구 `--selftest` rc=0 ·
  판독기 재실행(`--expected-head 0e899cb`) → 운전석 JSON 과 **byte-동일** · exports 3자 diff
- 부분-R7 관측: `[A2-09]` 접두 1,203줄 byte-diff 무차 · `[cmf_fine]` 타이밍 정규화 후 무차
