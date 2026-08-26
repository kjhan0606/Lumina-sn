# 단 사전등록 — A210-ZERO-OPACITY (2026-08-19 · 개정 1: 2026-08-26)

user 지시 "A210-ZERO-OPACITY 단 세워". 발단은 S4(III 음성대조) BLOCKED
(`docs/VERDICT_DET_SPROD_S4_III_2026-08-19.md`).
조사 중 **표적이 넓어졌다** — 아래 §1 이 그 경위이며, 계약을 그에 맞춰 정의한다.

## 개정 이력

- **개정 1 (2026-08-26, user 진행 지시)** — DET-L6C-COVER 폐합 산출
  (`docs/VERDICT_DET_L6C_COVER_2026-08-26.md`) 반영. ① §2 미확정 3건 재기재(②는 표본 1행에
  한해 (a) 확정·③은 봉인 L6 실측으로 해소). ② Z-1 형상 모순(capture=0 완주 요구 ↔ 차단행
  필드 인쇄 요구의 양립 불능 — L6C 감리-A 발견·운전석 소스 확증) 해소: **선정·capture 무관
  hazard-행 계측(Z-1r, src 접촉=V5)** + capture=0 census 런 1회. ③ 표적 = **ion=1 단독**
  (IV 는 L6 실측 0·III 행 262210 은 Z-2A 후 Z3 겸측). ④ Z-2 처분: **A(진단 격리)는 지금
  선택한다** — 선택 근거가 census 결과에 무의존임을 §3 원문이 자증한다. **발주만 Z-1 착지
  후다.** ⑤ §5.5 정정: h200 우선 순서가 런 319967 을 죽였다 ⟹ **a100 한정**. ⑥ §6 을
  개정14·16·17 로 갱신, §8 기대치·분기, §9 프리플라이트 선언, §10 집행 기록(정본 장부) 신설.
  계약 본문(§계약·§1 기전·§7 처분 원칙 골자)은 불변이다.

## 계약 (하나)

> **진단 경로의 실패는 진단만 무효화하고, 물리 트랜잭션을 중단시키지 않는다.**

exact-zero 불투명도 행의 처분은 이 계약의 **구현 귀결**이다(별도 계약이 아니다).

## 1. 왜 표적이 넓어졌나 — 오프라인 기전 특정 (실측)

### 1-1. χ_eff 가 정확히 0 이 되는 경로 (`src/line_net_rate.c`)

```c
material->raw_integrated_opacity      = tau * nu / (c * t_exp);
material->effective_integrated_opacity = material->raw_integrated_opacity;
...
if (policy == CMFGEN_SRCE_CHK && tau < -0.5) {   /* 음수 큰 것만 대체 */
    material->effective_integrated_opacity = CMFGEN_INTERNAL_OPACITY_TO_CGS / n;
}
```
ν>0·t>0 이므로 **χ_eff == 0 ⟺ tau == 0 정확**. SRCE_CHK 는 `tau < −0.5` 에서만 발화하므로
tau==0 을 구제하지 않는다.

그리고 상류 계약은 그런 행을 **유효로 받아들인다**:
`a210_line_saturation_add` 입구가 `tau_validity==A208_VALID || tau_validity==A208_EXACT_ZERO`.
⟹ **"tau 는 정당하게 0" 이라고 등록해 놓고, 그 행에서 S 가 미정의라며 막는다.**

또한 `exact_zero_provenance = (n_upper==0.0 && tau==0.0)` 이므로
**tau==0 이면서 n_upper>0** 인 행은 그 표지를 얻지 못한다 — 방출은 있고 순 불투명도만 0인 상태다.

### 1-2. ★진단이 물리 트랜잭션을 중단시킨다 (이것이 더 무겁다)

```c
if(independent_capture && (!source_defined || !isfinite(source_function))){
    a210_line_saturation_blocked("INDEPENDENT_SPROBE_UNDEFINED", ...); return -1;
}
```
그리고 소비자:
```c
int saturation_add = a210_line_saturation_add(...);
if(saturation_add != 0){
    status = saturation_add==-2 ? RADEQ_NONFINITE : RADEQ_TERM_SCHEMA;
    first_bad_line=line; first_bad_shell=s; break;
}
```
⟹ **opt-in read-only 진단(`LUMINA_A210_INDEPENDENT_CAPTURE`)의 행 빌더 실패가
A2-10 물리 트랜잭션의 status 를 갈아치우고 루프를 끊는다.**

이는 진단 자신의 선언과 모순된다 — 그 경로가 찍는 문자열이
`interpretation=DIAGNOSTIC_ONLY physical_values_modified=0 publication_authority=NONE` 이다.
값을 **바꾸지는** 않지만 **결과를 없앤다**. 실측 확증:
III 런에서 `phase=REQUESTED_TE status=RADEQ_TERM_SCHEMA valid=0`,
같은 런의 다른 3국면은 정상 완주(`RADEQ_NO_BRACKET`).
노브를 끄면 이 차단은 발생하지 않는다(코드상 `independent_capture &&` 가드).

### 1-3. 계급 — 같은 실수의 세 번째 자리

"정당하게 0" 과 "무효" 를 혼동한다. SH-GAMMA **NC3**, MC-EVT **OUT_OF_GRID** 에 이어 세 번째.

## 2. 미확정 3건 — 개정 1 재기재 (원문 2026-08-19 → L6C 폐합 2026-08-26 반영)

- **어느 행인가 (목록·총수)** — **부분 해소·목록은 미확보.**
  해소분: ion=1 첫 hazard 행의 전 필드가 in-vivo 실측됐다(L6C WITNESS: `line=669992` Fe II ·
  `tau_validity=2` · `n_upper=4.437688873013011e-4` · `A_ul=0.5222556` ·
  `emission_per_sr=2.7310010625545125e-16`; `candidate_rows=66110` 교차검증으로 **스캔 순서상
  첫 hazard** 확증 — L6C 판정문 §5-1 ⑦). 미해소분: **총수는 중도절단 하한 ≥1**(카운터 증가
  직후 같은 호출에서 차단 — 구조상 1 을 넘길 수 없다, L6C §5-4-1·감리-A 대수 폐합).
  **목록은 현 계측 형상으로는 원리적으로 안 나온다**(§3 의 형상 모순 — WITNESS 는 capture
  분기 안·ROW 는 선정 행만·카운터는 수만). S4-III 의 ion=2 행 `262210` 은 여전히 필드
  미기록이고 L6C 확증을 그 행으로 **이월할 수 없다**(L6C §6-2).
- **왜 tau==0 인가 — (a) 정당한 exact-zero vs (b) coverage 결손** — **표본 1행에 한해 (a)
  확정.** `tau_validity=2 = A208_EXACT_ZERO`: 출판이 정당한 정확 0 으로 등록한 행이다
  (coverage 결손이면 `A208_UNSAMPLED/MISS` 로 앞서 거부됐다). 기전: 시드 NLTE ladder 의
  준위쌍 인구 **bit-동일**(g_l=g_u=8 ⟹ χ ∝ n_l/g_l − n_u/g_u 의 정확한 영점) = S4-III `:57`
  이 정의한 **inversion 경계 그 자체** — 소스 추론이 전 필드 실측으로 확증됐다.
  ★귀결(L6C §6-4): hazard 는 LTE 인구가 아니라 **시드-물질 예측자의 NLTE solve 커밋분의
  성질**이다(LTE(10020)면 n_l/n_u ≈ 4.4e4 라 상쇄 불능 — PC-2 오프라인 예측 `hazard=0` 의
  in-vivo 반증; 검출 대수 자체는 3자 census 일치로 무죄) ⟹ **census 는 오프라인 덱 산법으로
  불가능하고 라이브 커밋 물질 위에서 돌아야 한다.** 잔여: 1행 밖 전 행의 (a)/(b) 판별 +
  승계 미결 «bit-동일의 구조적 생산자»(L6C §5 말미) — 후자의 귀속 입력이 곧 행 목록
  (준위쌍 식별자)이다.
- **IV 에는 왜 없었나** — **해소.** 후보 211,887행 전수 스캔 범위(shell 0)에서
  `zero_opacity_emitting_rows=0`·양 반복·완주 [저자 실행: `grep -n zero_opacity_emitting_rows
  /gpfs/kjhan/lumina/det_stage12_fixed_te_a100x2_k36/sprim_l6_20260821T054111Z_probe/stderr.log`
  → `:1893`(iter0)·`:3581`(iter1) 둘 다 `complete=1 … zero_opacity_emitting_rows=0`].
  그런 선이 **없었다** — 순서 문제가 아니다.

⚠ 남은 미확정 = **①의 목록·총수 + ②의 1행 밖 (a)/(b) 판별**. 이것이 Z-1 census 가 측정할
전부이며, Z-2 **후보 간 선택**에는 더 이상 필요하지 않다(§3 Z-2 개정 1 — A 는 census 무의존).
census 가 정하는 것은 A 의 검증 기준선과 **B 병행·coverage 별도 단 개설 여부**다.

## 3. 단계

### Z-1 계측 (이 단의 첫 실행) — 기존 봉인 로그 재사용 불가 ⟹ 계측 추가 필요

★**Z-1 형상 모순과 해소 (개정 1 — 이 개정의 발단)**
원문 §3 은 (i) *"census 런은 DIAG=2 를 켠 채 INDEPENDENT_CAPTURE 만 끈다"*(차단 회피 완주)와
(ii) *"차단 시점에 그 행의 전 필드를 한 줄로 찍는다"* 를 동시에 요구했다. **양립하지 않는다**
[저자 실행 — HEAD `e1c63cc` 소스 직독]: 카운터 증가(`src/lumina_plasma.c:14075`)는 capture
무관이지만 **WITNESS 인쇄(`:14113-14114`)는 `independent_capture &&` 분기 안**이고, ROW 인쇄는
**선정 행만**(`:14337` `if(!selected[i])continue;`) ⟹ capture=0 이면 총수만 나오고 행 필드는
하나도 안 나오며, 방출 가중이 작은 hazard 행은 어디에도 안 찍힌다. (L6C 감리-A 발견 = L6C
판정문 B-3 ③; 본 문서 §6.5 "수 1개 + 증인 행 1개" 한계가 예고한 자리.)

**해소 = Z-1r 계측 확장 (src 접촉 1점 ⟹ V5 요구)**: 카운터 증가 지점(`:14075` 직후·판정
분기보다 앞)에 **선정·capture 무관** 행 인쇄 1개를 추가한다 —
`[A2-10][LINE-SATURATION-ZERO-OPACITY-ROW]`, 필드 = WITNESS 와 동일 세트(phase·shell·line·Z·
ion·lower_global·upper_global·tau_validity·tau_raw·effective_tau·n_upper·A_ul·nu·
emission_per_sr·srce_chk·exact_zero_provenance·source_defined). **판정 분기·반환값·차단 시점
불변**(순수 fprintf — `a8fd187` Z-1 계측과 같은 계급). 네임스페이스는 `LINE-SATURATION-` 안 —
phase 스트림 비교기가 그 접두를 스트림에서 제외한다 [저자 실행:
`scripts/compare_a210_phase_baseline_streams.py:24-25,78-79` `SATURATION_PREFIX` skip 직독].
★**V5 가 이 단에서 부여돼야 하는 이유**: §2-① 의 목록과 승계 미결(bit-동일 생산자 귀속 —
준위쌍 식별자 필요)은 위 세 실측(:14113 capture 분기·:14337 선정 가드·:14075 수만)으로
**현 형상에서 원리적으로 산출 불능**이다. src 밖 우회로가 없다.
★**행-계측이 census 의 정본**이고 카운터는 대조용이다(ROW 줄 수 == 카운터, §4 Z1).
인쇄량에 상한을 두지 않는다(cap 금지) — stderr 은 /gpfs 런루트에 쓴다. 관측 규모(후보
66,110·hazard ≥1)에서 위험은 낮고, 전 행 hazard 인 극단에서도 후보 수 × 1줄이다.

★**런 노브 (감사 지적 1 — 틀리면 런 하나를 버린다)**
카운터는 `diag->active` 안에서만 증가하고 `diag->active` 는
`LUMINA_A210_LINE_SATURATION_DIAG` 가 켜야 선다(`lumina_plasma.c:13943,13970`).
⟹ **census 런은 `LUMINA_A210_LINE_SATURATION_DIAG=2` 를 켠 채
`LUMINA_A210_INDEPENDENT_CAPTURE=0`·`LUMINA_A210_SPRODUCER_CAPTURE=0` 으로 돈다.**
(개정 1: producer 캡처도 끈다 — `src/lumina.h:963` `a210_sproducer_raw_decode` 는
`capture_active=0` 이면 무조건 NULL 반환이라, hazard 행 전용 차단 지점이 **capture 분기뿐**임이
소스로 닫힌다 [저자 실행 직독]. 일반 차단(ROW_CAPACITY 류)은 §8 ZC-D 분기가 받는다.)
DIAG 까지 끄면 카운터가 0회 증가하고 SUMMARY 도 안 찍힌다.
`scripts/stage_a210_line_saturation_diagnostic.sh:112` 는 캡처를 기본 ON 으로 두므로 그
스테이징을 그대로 쓰면 죽는다 — **§5 의 census 전용 스테이저·런처 모드로만 발주한다**
(기존 L6C 모드는 capture=1 을 강제한다 [저자 실행: `scripts/run_det_convergence_2026-08-08.slurm:198`
"L6C independent continuum capture is not armed" 가드 직독] ⟹ 재사용 불가).

★**카운터의 정확한 스코프 (감사 지적 3 — 대장 기재 필수 문구)**
이 수는 "스캔 전체" 가 **아니다**. **shell 0 · target ion 후보행 중** `tau==0 && n_upper>0` 인 수다.
빠지는 것: shell≠0 전부 · target ion 밖 전부 · 주파수 창 밖 · 비활성 Z ·
`UNRESOLVED_CANCELLATION`/`INVALID_INPUT` 셀 · **첫 차단 이후 전부**(노브 ON 런은 접두 부분합).
SUMMARY 가 여러 줄이면 각 줄은 독립 부분합이므로 **합산 금지**.
⟹ "III 전체에 tau==0 행이 N개" 로 읽으면 틀린다.
(개정 1 정리) 차단-시점 WITNESS 인쇄와 카운터는 **`a8fd187` 로 구현 완료·L6C 실전 1회**
(첫 hazard 행 전 필드 기록 — 단 capture=1 런에서만, 그리고 첫 행뿐이다). 완주 census 의 행
기록은 위 **Z-1r 행-계측**이 담당한다 — capture·선정 무관, 전 행. **판정 로직 불변** 조건은
Z-1r 에도 그대로다. 완주 스코프 주의도 그대로다: 스캔 범위 = shell 0·REQUESTED_TE·활성 Z·
주파수 창 [저자 실행: 봉인 L6 SUMMARY 두 줄 모두 `phase=REQUESTED_TE shell=0`]. 스코프
확장은 이 단의 범위 밖이다.

### Z-2 계약 수리 — ★개정 1: **A(진단 격리)는 지금 선택한다. 발주만 Z-1 착지 후다**
후보:
| 안 | 내용 | 조건 |
|---|---|---|
| **A** 진단 격리 | 진단 실패는 `diag` 만 무효화(`diag->active=0` + 사유 기록), 물리 루프 계속 | Z-1 이 "물리적으로 정당한 행" 을 보이면 **유력** |
| **B** 행 건너뛰기 | 그 행만 skip + **건너뛴 수 보고**(NC3 정신). 기존 `scaled_emission==0 → return 0` 과 대칭 | A 와 병행 가능 |
| **C** 차단 유지 | Z-1 이 "미기록 coverage 결손" 을 보이면 차단이 옳다 — 대신 **사유 이름을 바꾼다** | (b) 인 경우 |

★A 와 C 는 배타가 아니다: **진단이 물리를 죽이지 않는다(A)** 는 (b) 인 경우에도 유지되어야 하고,
coverage 결손은 **별도 단**으로 다뤄야 한다.

★**개정 1 — A 선택의 확정과 그 논리 (census 결과 무의존)**. A 를 지금 선택할 수 있는 근거는
census 데이터가 아니라 대수다: ① 바로 윗 문단(원문 2026-08-19)이 이미 **A 는 (b) 인 경우에도
유지된다** 고 명시했다 — census 의 어느 결과 가지에서도 A 는 뒤집히지 않으므로 선택을 미루는
것의 정보 가치가 0 이다. ② 차단의 소유자가 진단 트랜잭션임은 세 런의 대조로 폐합됐다
(S4-III ion=2 사망 · L6 ion=3 침묵 완주 · L6C ion=1 사망 — L6C 판정문 §5-4-3, "이온 회피
불능"). ③ A 의 유력 전건("물리적으로 정당한 행")이 표본 1행에서 성립했다(§2-②). 그리고
A 는 이 단의 계약("진단 경로의 실패는 진단만 무효화하고, 물리 트랜잭션을 중단시키지 않는다")
의 구현 그 자체다. **발주는 Z-1 착지 후다** — A 구현의 검증 게이트(Z3·Z4)가 census 기준선을
소비하고, A 의 세부(실패 시 diag 처분 형식)와 **B 병행 여부·coverage 별도 단 개설 여부는
census 종속**으로 남는다. Z-2A 발주 시 §5 기대 변경집합을 재등록하고 프리플라이트를 다시
돈다. **C(차단 유지)는 전면 정책으로는 기각**됐다(표본이 (a) 라서) — (b) 행이 census 에
나타나면 그 행들에 한한 이름 있는 거부 사유로만 재등장할 수 있다.

## 4. 게이트

| # | 조건 |
|---|---|
| **Z1** | (개정 1) Z-1r 계측이 census 런에서 **모든 hazard 행**의 전 필드 ROW 줄을 찍고, ROW 줄 수 == SUMMARY `zero_opacity_emitting_rows` 정합. **판정 로직 불변**(word-diff 로 차단 시점·사유·반환값 불변 확인) |
| **Z2** | (개정 1 재기재) 행 수 보고 3원: **IV = 소급 충족**(봉인 L6 실측 0·완주 — §2-③ [저자 실행]) · **II = census 런**(이 단) · **III = Z-2A 후 Z3 재현런의 겸측**(Z-1r 이 capture 무관이므로 그 런이 행 `262210` 의 전 필드를 자동 기록한다) |
| **Z3** | (수리 후) **III 가 Stage-4 row 를 낸다** — S2 덧셈 항등이 III 에서도 bit 로 성립하는가 |
| **Z4** | (수리 후) **IV byte 불변 — 값 파일 한정**. `scripts/byte_parity_compare.py` Tier1 로 IV 재현런의 값 산출물(`lumina_spectrum.csv` 등)을 봉인 IV 와 대조. ⚠**stderr 로그 전체에 걸면 자동 실패**한다 — Z-1 이 SUMMARY 줄에 `zero_opacity_emitting_rows=` 를 더했고 그것은 IV 에서도 바뀌기 때문(감사 지적 2). 로그를 대조하려면 그 토큰의 정규화 규칙을 **선언**하고 Tier2 census 에 남길 것 |
| **Z5** | (개정 1 재기재) 음성 대조 2겹: **(in-vivo) capture 토글 페어** — 봉인 L6C(ion=1·capture=1·사망 실측 322449) ↔ census 런(ion=1·capture=0·완주 기대), 같은 레인·같은 이온·같은 시드 ⟹ 차단이 캡처 진단 탓임을 주입 없이 시연 · **(계측 NC)** census 판독기 `--selftest` NC-Z0~Z4(§8) — 주입 결함으로 FAIL 시연 의무 |
| **Z6** | (개정 1 신설) 결정론·무섭동 앵커: census ROW 목록에 `line=669992` 존재 + §8 E-Z3 의 4개 필드 bit-재현. 캡처 토글(진단)이 물리를 건드리지 않았음의 실측 검출기를 겸한다 |

★**Z5 가 이 단의 NC3 다** — "진단이 원인" 이라는 주장을 주입 없이 시연한다.
(개정 1: III OFF 런은 불필요해졌다 — 봉인 L6C 사망이 ON 팔을 공짜로 준다. 런 예산 1 이 줄었다.)
★**Z4 는 오늘 만든 byte 비교기의 첫 실전**이다(08-08 의 R6-4 실패 모드를 되풀이하지 않는다).

## 5. 기대 변경집합 (개정 1 — Z-1r 계측 + census 집행. Z-2 는 census 후 재등록)

원문 Z-1 몫(WITNESS 출력 + 카운터)은 `a8fd187` 로 **집행 완료**됐다. 아래가 개정 1 의
증분이며 이 단의 전 변경집합이다 — 다른 파일이 표지 심볼을 얻으면 §9 PF-1 이 거부한다.
src-편집 태스크는 이 하나뿐이다(발주 전 가동 중 변조 태스크 0 확인은 운전석 몫).

### 변경집합 — 개정 1 (표)

| 파일 | 내용 |
|---|---|
| `src/lumina_plasma.c` | Z-1r 행-계측 1개: 카운터 증가(:14075) 직후 `[A2-10][LINE-SATURATION-ZERO-OPACITY-ROW]` 인쇄, capture·선정 무관. **판정 분기·반환값·차단 시점 불변. 물리식 무접촉.** (표지 심볼 무보유가 정당 — expected_extra 행) |
| `scripts/run_det_convergence_2026-08-08.slurm` | `A210_ZO_CENSUS` 모드 신설: iterations==2 강제(L6C 와 동형)·A100 강제 case 편입·env 가드(`INDEPENDENT_CAPTURE==0`·`SPRODUCER_CAPTURE==0`·`DIAG==2`·`TARGET_ION==1`)·완주 수용 블록(model rc=0 + 양 반복 SUMMARY `complete=1` + 판독기 rc=0) |
| `scripts/stage_a210_zero_opacity_census.sh` | 신설 — 봉인 L6C 런루트 기반 스테이징. exports 값 delta **정확 2**(두 캡처 1→0) + 경로 재배치만, `diagnostic_mode.txt` = 표지 심볼. 봉인 접촉은 읽기 전용 복사(하드링크 금지 — 개정16 ② 의 봉인 파괴 전례) |
| `scripts/analyze_a210_zero_opacity_census.py` | 신설 — ROW 파서·카운터 정합·E-Z3 앵커 대조·§8 분기 판정·`--selftest` = NC-Z0~Z4 |

### 변경집합 끝

## 5.5 ★런 발주 경로 (2026-08-19 변경 · ★개정 1 정정: a100 한정)

user 지시 **"syn101 수동 제출은 금지. 해당 노드는 정상 운영중."**
⟹ census 런은 **slurm 으로 제출**한다. tripwire 수동런을 쓰지 않는다.
가드: `scripts/run_manual_det_with_tripwire.sh` 가 syn101 이면 무조건 거부하도록 박았다.

★**개정 1 정정 — 원문의 "파티션 순서 h200→h100→a100" 이 런 하나를 죽였다.** A2-10 진단은
런처가 A100 을 강제한다(`scripts/run_det_convergence_2026-08-08.slurm:177-178`). 8-19 시도가
정확히 그렇게 죽었다 [저자 실행: `tail …/z1_20260818T230745Z_iii_capture_on/slurm-319967.err`
→ `DET_FLIGHT_FATAL A2-10 targeted gate requires A100 hardware: NVIDIA H200 NVL`;
`sacct -j 319967,319968 -X` → 둘 다 partition=h200/syn104, 319967 FAILED 70:0(40분 소모)·
319968 CANCELLED]. ⟹ **`-p a100` 한정**(h200/h100 폴백 금지) · `--gres=gpu:2`(two-device
exact owner) · `--mem` 명시(`--mem=64G` — L6C 실측 MaxRSS 35.4 GiB 등급) · `--time` 명시
(백필 자격 — L6 완주 04:34/2iter 실측 유추로 `08:00:00`) · job-name **`LUMINA_ZO_CENSUS_II`**
(프로젝트 접두 의무) · 런루트에 `OWNER.txt`(project·repo·목적·`DO_NOT_CANCEL`).
★8-19 시도 산출 4 하위 디렉토리(`/gpfs/kjhan/lumina/a210_zero_opacity_z1_a100x2_k36/z1_*`)는
**봉인이 아니다** — 스테이징만 있고 착지가 없다 [저자 실행: 디렉토리 실사 — slurm err/out 과
input 뿐, 판정문 0]. 재사용 금지, 신규 런루트로 발주한다.

## 6. 판정 절차 (개정 1 — 분담 개정14·16·17 반영)

기획·사전등록·검수·판정=**Fable**(각각 별개 컨텍스트 — 위임 상시 승인 4종) / 코딩=**Codex** /
발주 행위·빌드·실행·제출·계측·대장·커밋=**운전석**.
- **감리 두 겹(개정17)**: 판정 후 **감리-A=Fable fresh**(판정과 다른 컨텍스트·고정질문 4) +
  **감리-B=Codex read-only**. 폐합 조건 = A·B 둘 다 통과(L6C 가 첫 적용 — 두 겹이 겹치지 않는
  결함 종류를 각각 잡았다).
- **장부 단일화(개정16 ①)**: 분장 장부 정본 = **이 문서 §10 집행 기록 한 곳**. 판정문·감리문은
  복제하지 않고 "장부 = 사전등록 §10" 으로 지목하고 자기 칸만 기입한다.
- **검수 선행(개정16 ②)**: 코드 검수 고정질문 첫 항목 = *"이 산출물을 봉인·실물에 돌려도
  되는가"* — 그 답이 오기 전 운전석은 봉인 접촉 게이트를 돌리지 않는다(봉인 무접촉
  게이트 — 빌드·selftest — 는 병행 가능).
- **발주 프리플라이트(개정15)**: 모든 Codex 발주 전 `scripts/check_prereg_preflight.py`(§9
  블록)·`scripts/check_dispatch_preflight.py` rc=0(grammar-debug). 발주서 뒷겹은 계약 값을
  재서술하지 않고 절 번호로만 지목한다. 검수 고정질문에 "발주서가 사전등록의 범위를
  좁혔는가" 포함.

## 6.5 감사 (2026-08-19, Fable 독립 컨텍스트)

Z-1 적용 diff 를 독립 감사에 넘겨 **승인(조건부)** 을 받았다.
- 코드 수정 불필요: word-diff 전수로 판정 로직·반환값·차단 시점 불변 확인,
  호출부 26곳 전수 갱신, 포맷/인자 22:22·7:7·7:7 손 계수 + `-Wformat=2` 일치.
- 손 이관본이 Codex 원안보다 두 곳 정확(실물 `%.21Lg` 보존, `isfinite(...)?1:0`).
- 조건 3건은 전부 코드 밖이며 위 §3·§4 에 반영했다(런 노브·Z4 문언·대장 문구).
- **권고(Z-2 착수 시)**: `[A2-10][ZERO-OPACITY-WITNESS]` 접두사를
  `LINE-SATURATION-` 네임스페이스 안으로 옮길 것. 차단을 걷어내는 순간
  `scripts/compare_a210_phase_baseline_streams.py:78-79` 가 이 줄을 phase 스트림에 담아
  조용히 레코드 수 불일치로 깨진다.
  (개정 1 주기: **이행 완료** — 실물 접두는 이미 `[A2-10][LINE-SATURATION-ZERO-OPACITY-WITNESS]`
  다 [저자 실행: `src/lumina_plasma.c:14114` 직독]. Z-1r 신규 인쇄도 같은 네임스페이스를 쓴다.)
- **한계 기재**: Z-1 산출은 **수 1개 + 증인 행 1개**뿐이다. §2 가 요구한 "목록" 과
  "(a) 정당한 exact-zero 인가 (b) coverage 결손인가" 는 **그 한 행에 한해서만** 판별된다.
  Z-2 안 선택 근거로 삼기 전에 이 한계를 명시할 것.

## 7. 처분 원칙

이 단은 **측정 단**으로 시작한다. Z-1 결과 없이 Z-2 를 발주하지 않는다.
(개정 1: Z-2A 의 **선택**은 §3 의 census-무의존 논리로 지금 확정했다 — 이 문장이 금지하는
것은 **발주**이며 그 금지는 그대로 유효하다.)
발견은 조용한 대장 기재이며, 클램프·대체값으로 증상을 덮지 않는다.

## 8. census 기대치·분기 사전등록 (개정 1 신설)

**런 1회**: `A210_ZO_CENSUS`(ion=1 · 캡처 2종=0 · DIAG=2 · iter=2 · a100×2). iter=2 인 이유:
봉인 L6C 와 동형(비교 가능성) + iter1 SUMMARY 가 **무료 관측**을 준다 — bit-동일 준위쌍이
한 세대 solve 뒤에도 유지되는지(생산자 귀속의 입력). 오프라인으로 닫히지 않는 이유: hazard 는
시드-물질 예측자의 **NLTE solve 커밋분**의 성질이고 그 산출은 GPU 런 없이 재현 불능이다
(L6C 판정문 §5-3 — *"오프라인 census 로 잡을 수 있었다" 는 주장은 성립하지 않는다*).

| # | 기대 | 등급·근거 |
|---|---|---|
| E-Z1 | 완주: model rc=0 ∧ 양 반복 SUMMARY `complete=1` | 소스 직독 근거(§3 — hazard 행 전용 차단 지점은 capture 분기뿐), 단 [실행 미검증 — 저자가 실행할 수 없는 GPU 런] |
| E-Z2 | iter0 `zero_opacity_emitting_rows ≥ 1` ∧ **== iter0 ROW 줄 수** | 하한 = L6C 실측 승계(중도절단 ≥1) |
| E-Z3 | iter0 ROW 목록에 `line=669992` · 필드 bit-재현: `tau_validity=2` · `n_upper=4.437688873013011e-4` · `A_ul=0.5222556` · `emission_per_sr=2.7310010625545125e-16` | 결정론 앵커(K36 baseline·같은 시드) — 빗나가면 ZC-Z 조사 |
| E-Z4 | iter0 `candidate_rows` > 66110(중도절단 값 초과 = 완주의 표지). 절대값·iter1 값·ion=1 hazard 의 준위쌍 목록은 관측으로 기재 | 관측(비게이트) — 덱 ion=1 후보 1,048,375 의 창-절사 값 [추정: L6 의 IV 비율 211,887/215,038 유추 ≈1.03M] |
| E-Z5 | 경과 ≈04:34 등급 · MaxRSS ≈35.4 GiB 등급 | L6·L6C 실측 유추(비게이트) |

**분기(판별량 = fa ≡ iter0 hazard 행 중 `tau_validity==A208_EXACT_ZERO` 비율)**:

| 분기 | 판정식 | 처분 |
|---|---|---|
| **ZC-A** | 완주 ∧ hazard>0 ∧ fa ≥ 1.0 (전건 (a)) | Z-2A 발주 준비(§5 재등록·프리플라이트 재실행) + B 병행 검토. 준위쌍 목록을 bit-동일 생산자 귀속에 인계 |
| **ZC-B** | 완주 ∧ hazard>0 ∧ fa < 1.0 ((b) 혼재) | Z-2A 불변 + **coverage 결손 별도 단 개설**(원문 §3 명문) — (b) 행 목록이 그 단의 입력 |
| ZC-Z (분할 밖) | 완주 ∧ hazard=0 | E-Z3 와 모순 ⟹ 결정론 또는 계측 결함 조사 — 물리 결론 금지 |
| ZC-D (분할 밖) | 미완주(어떤 차단이든) | 차단 사유·행 기재, 사유의 이름 있는 귀속 후 재발주. fail-closed — L6C D2 전례의 처분 형식 승계 |

**계측 음성대조(NC — 판독기 `--selftest`, 발주 전 grammar-debug rc=0 의무)**:
NC-Z0 성한 census 로그는 통과해야 한다 / NC-Z1 카운터>0 ∧ ROW 0줄 주입 → FAIL /
NC-Z2 ROW 줄 수 ≠ 카운터 주입 → FAIL / NC-Z3 앵커 행 필드 1자릿수 변조 주입 → FAIL /
NC-Z4 `complete=0`(중도절단) 로그를 완주로 오독 → FAIL.

## 9. 기계 프리플라이트 선언 (개정 1 신설 — 형식 = DET-L6C-COVER §12)

`scripts/check_prereg_preflight.py <이 문서> --root .` 이 발주 전에 검사한다(fail-closed).
PF-1 은 §5 표와 표지 심볼의 양방향 1:1 을(발주 시점(구현 전)은 expected_extra 가 계획 경로를
공급하고, 커밋 후에는 grep 실물이 같은 집합을 낸다 — 5번째 파일이 심볼을 얻으면 거부),
PF-2 는 fa 1차원 분할(ZC-Z·ZC-D 는 A207·L6C 전례와 같은 분할 밖 이름 있는 계급)을, PF-3 은
인용 앵커의 실존을 강제한다. **정직한 한계**: E-Z3 bit-재현의 의미 검사와 Z-1r 의 "판정 로직
불변" 은 정적 검사 밖이다 — 각각 판독기 게이트와 검수 word-diff 가 잡는다.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 — 개정 1 (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "(?:scripts|src)/[a-z0-9_\\-]+\\.(?:py|sh|slurm|c)",
    "symbol": "A210_ZO_CENSUS",
    "roots": ["scripts", "src"],
    "expected_extra": ["src/lumina_plasma.c",
                       "scripts/run_det_convergence_2026-08-08.slurm",
                       "scripts/stage_a210_zero_opacity_census.sh",
                       "scripts/analyze_a210_zero_opacity_census.py"]
  },
  "branches": {
    "regimes": [[0.0, 0.5], [1.9, 2.1]],
    "metrics": {
      "fa": "sum(1 for x in v if x > 1.5)/len(v)"
    },
    "rules": [
      {"name": "ZC-A", "predicate": "fa >= 1.0"},
      {"name": "ZC-B", "predicate": "fa < 1.0"}
    ],
    "adversarial_fixtures": [
      {"name": "witness-only", "mix": [[2.0, 1]]},
      {"name": "all-exact-zero", "mix": [[2.0, 66110]]},
      {"name": "single-coverage-defect", "mix": [[2.0, 66109], [0.0, 1]]},
      {"name": "all-coverage-defect", "mix": [[0.0, 51807]]}
    ]
  },
  "references": [
    {"path": "src/lumina_plasma.c",
     "flags_existing": ["zero_opacity_emitting_rows",
                        "LINE-SATURATION-ZERO-OPACITY-WITNESS",
                        "INDEPENDENT_SPROBE_UNDEFINED"]},
    {"path": "src/lumina.h", "flags_existing": ["a210_sproducer_raw_decode"]},
    {"path": "src/opacity_publication.h", "flags_existing": ["A208_EXACT_ZERO"]},
    {"path": "scripts/run_det_convergence_2026-08-08.slurm",
     "flags_existing": ["A210_L6C_PROBE", "requires A100 hardware",
                        "A210_CANCELLATION_CENSUS"]},
    {"path": "scripts/compare_a210_phase_baseline_streams.py",
     "flags_existing": ["[A2-10][LINE-SATURATION-"]},
    {"path": "docs/VERDICT_DET_L6C_COVER_2026-08-26.md",
     "flags_existing": ["candidate_rows=66110", "A208_EXACT_ZERO",
                        "population_generation"]},
    {"path": "docs/VERDICT_DET_SPRIM_L6_2026-08-22.md",
     "flags_existing": ["zero_opacity_emitting_rows=0"]},
    {"path": "docs/VERDICT_DET_SPROD_S4_III_2026-08-19.md",
     "flags_existing": ["262210", "INDEPENDENT_SPROBE_UNDEFINED"]}
  ]
}
```

## 10. 집행 기록 (정본 장부 — 개정16 ①)

이 절이 분장 장부의 **유일 정본**이다. 판정문·감리문은 "장부 = 사전등록 §10" 으로 지목만
한다. 개정15 필수 4항목(발주 프리플라이트 rc · `[미실행]` 주장 실행 결과 · 분장 장부(검수
완료 후 기입) · 발주 중단 사유와 주인)을 운전석이 기입한다.

| 단계 | 규약상 담당 | 실제 | 위반 |
|---|---|---|---|
| 사전등록 개정 1 | Fable | Fable fresh 컨텍스트 (2026-08-26) — 봉인·소스·런 흔적 주장 전건 [저자 실행], 프리플라이트 자체검증 rc=0 | — |
| 구현 | Codex | Codex `gpt-5.6-terra` xhigh, clean worktree `d949dd7` detached, workspace-write·네트워크 차단. 변경집합 4행 1:1, 초과·미달 0 | — |
| 코드 검수 | Fable (별개 컨텍스트) | Fable fresh (2026-08-26). **차단 0건 · 지적 6건 전부 비차단.** 질문 0(봉인 접촉 가부)=**YES** — 스테이저 쓰기 프리미티브 전수 조사로 봉인 파괴 경로 0 확인 | — |
| 발주·집행·런 제출 | 운전석 | 발주 프리플라이트 rc=0 → Codex 발주 → **검수 선행**(개정16 ②) → 봉인 무접촉 게이트 5건 → 커밋 `e500e29` → 봉인 접촉 스테이징 → 봉인 불변 실측 → 제출 `324388` | — |
| 판정 | Fable (fresh) | (기입 대기) | |
| 감리-A | Fable (fresh) | (기입 대기) | |
| 감리-B | Codex (read-only) | (기입 대기) | |

### 10.1 집행 기록 본문 (운전석 기입, 2026-08-26)

**① 발주 프리플라이트 rc** — `scripts/check_dispatch_preflight.py <발주서> docs/RUNG_A210_ZERO_OPACITY_2026-08-19.md --root .` (grammar-debug) → **rc=0**.
DP-1 `no-restatement` (뒷겹 재서술 0, tokens_checked=5) · DP-2 `frontpage-sealed` · DP-3 `contract-preflight rc=0`.
사전등록 프리플라이트도 개정 1 적용 시 rc=0 (`changeset-consistent files=4` · `branch-partition trials=200000 rules=2 fixtures=4` · `references-resolved refs=8`).

**② `[미실행]` 주장 실행 결과** — 개정 1 은 봉인·소스·런 흔적 주장을 전건 `[저자 실행]` 으로
표기했다(개정16 ③). 운전석이 발주 전 재실행한 것: L6 봉인 stderr `:1893`/`:3581` 의
`zero_opacity_emitting_rows=0` (양 반복 완주) · `sacct -j 319967` FAILED 70:0 @syn104(h200) ·
런처 `:198` 의 L6C `INDEPENDENT_CAPTURE==1` 강제 · `a210_sproducer_raw_decode` 의 `capture_active` 게이트.
**전건 재현 — 미실행 잔여 0.**

**③ 발주 중단** — **0회.** (직전 캠페인의 발주 중단 9건 중 4건이 운전석 발주서였다. 개정15
뒷겹 재서술 금지 적용 후 이번 발주는 중단 없이 통과했다.)

**④ 게이트 실행 결과** — 개정16 ② 에 따라 **검수 답(질문 0)이 오기 전 봉인 접촉 게이트를
돌리지 않았다.** 봉인 무접촉만 병행:

| 게이트 | 봉인 접촉 | 결과 |
|---|---|---|
| CPU 빌드 `lumina` | 무 | PASS — 바이너리 mtime > 소스 mtime **이고 새 심볼이 바이너리에 실존**(무효 재빌드 배제) |
| CUDA 빌드 `lumina_cuda` | 무 | PASS — 동상 |
| `selftest_det_stage12` | 무 | PASS — 자체 NC-C1~C3 음성대조 정상, 기존 판정 무변 |
| 판독기 `--selftest` | 무 | PASS — NC-Z0~Z4 5종 `failed=0` |
| 스테이저 `--selftest` | 무 (스크래치 전용) | PASS — NC-ZS0·ZS1 |
| bash 문법 (스테이저·런처) | 무 | OK |
| **봉인 접촉 스테이징** | **유** | **검수 통과 후 실행.** `A210_ZERO_OPACITY_CENSUS_STAGE_OK env_delta=2 mode=A210_ZO_CENSUS` |

**⑤ ★봉인 불변 실측** — 스테이징 **전·후** 봉인 L6C 런루트(`l6c_20260822T154507Z_cover`,
119 파일 · 5.1 GiB) 전수 지문을 정렬 비교했다:
`sha256 차이 0행` · `메타(크기·mtime·**inode**) 차이 0행`.
inode 까지 동일하므로 하드링크 오염도 0 이다 — 직전 단에서 스테이저가 `cp -al` + 제자리
truncate 로 봉인 base 를 침묵 파괴할 뻔한 그 계급이 실측으로 배제됐다.
(지문 정렬은 의도적이다 — 직전 단에서 `seal_before.txt` 를 정렬하지 않아 판정문의 재현
명령이 실패한 전례가 있다.)

**⑥ 스테이지 산출 독립 검증** (스테이저 자기주장을 믿지 않고 운전석이 재계산):
경로 정규화 후 exports delta = **정확 2쌍**(`SPRODUCER_CAPTURE 1→0`·`INDEPENDENT_CAPTURE 1→0`) ·
봉인·사본 **공유 inode 0** · `diagnostic_mode=A210_ZO_CENSUS` · `outer_iterations=2` ·
`TARGET_ION=1` · `LINE_SATURATION_DIAG=2` · `RADEQ_DIAG_TE_K=10020` ·
staged `binary.sha256` = repo `lumina_cuda` sha256 · staged `git_head` = repo HEAD = `e500e29`.

**⑦ 런 제출** — §5.5 사양 그대로 **slurm 1회**:
`sbatch --partition=a100 --gres=gpu:2 --mem=64G --time=08:00:00 --job-name=LUMINA_ZO_CENSUS_II`
→ **job 324388**. 런루트 `/gpfs/kjhan/lumina/a210_zero_opacity_z1r_a100x2_k36/zo_20260826T050021Z_census_ion1`, `OWNER.txt`(`DO_NOT_CANCEL=1`) 기재.
제출 스크립트는 자기 위치를 run_root 로 앵커하므로 봉인 root 가 `job.slurm` 인자로 가지 않는다
(검수 지적 2 준수). h200/h100 폴백 없음 — 개정 1 이 정정한 그대로다(8-19 시도가 h200 배정으로 죽었다).

**⑧ 검수 지적 6건의 처분** (전부 비차단):
| # | 지적 | 처분 |
|---|---|---|
| Q0-1 | dest 가드가 봉인 root **내부의 미존재 하위경로**를 막지 못함 | 운전석이 dest 를 봉인 root 밖 신규 트리로 지정하고 실행 전 대조했다. 기존 봉인 바이트의 덮어쓰기·삭제 경로는 0 으로 전수 확인됨 |
| Q0-2 | 런처를 봉인 런루트에 직접 재실행하면 3파일을 덮어쓴 뒤 죽음 (기존 결함, 이 패치 소산 아님) | 봉인 root 를 `job.slurm` 인자로 주지 않음. **별건 큐로 이월** — 런처의 "work dir not empty" 검사가 덮어쓰기보다 **뒤**에 있다 |
| Q1 | 발주서의 "플랑크 자체 계산" 조항이 이 변경집합에서 공회전 | 좁힘 아님. 무처분 |
| Q2-1 | 판독기가 §8 의 **ZC-D 라벨을 부여하지 않음** (미완주는 이름 있는 FAIL 로 fail-closed 되나 branch=`UNRESOLVED`) | **판정자에게 인계** — FAIL→ZC-D 매핑을 판정문에서 수동 수행할 것 |
| Q2-2 | 런처 가드 계약 4종 + 2종(`partition==a100`·`DIAG_TE_K==10020`) 초과 | 무해. 전자는 §5.5 강제, 후자는 봉인 exports 실측값 |
| Q2-3 | 워크트리에 미추적 빌드 산출물 `selftest_det_stage12` | 커밋에서 제외됨 (`git add` 4경로 명시, `git add -A` 미사용) |

**⑨ 검수자가 남긴 불확신 지점 4건** — 판정 시 확인할 것:
(1) E-Z3 앵커 bit-재현은 GPU 런에서만 판명 — 빗나가면 `ANCHOR_FIELD_MISMATCH`→ZC-Z 조사.
(2) **★해소됨 (운전석 코드 전수추적, 2026-08-26, 런 대기 중 실시).** 검수자는 봉인 실측에만
의존했으나, 추적 결과 **두 SUMMARY 는 상호배타**다:
호출부는 `:15383` **단 하나**(`a210_line_saturation_log_complete`) → 그 함수 `:14484` 가
`if(diag->mode==2) return a210_line_saturation_log_per_ion_union(diag);` 로 **조기 반환**하므로
DIAG=2 에서 `:14572` 는 **도달 불능**이다. 실행되는 `:14452` 는 per-ion 루프가 닫힌 뒤
함수 스코프에 있어 호출당 정확히 1회. 문면은 `phase=REQUESTED_TE … target_ion=%d …
complete=1` 순서로 런처 grep 과 일치한다. 같은 함수의 다른 인쇄는 태그가 달라
(`LINE-SATURATION-UNION-META`·`LINE-SATURATION-UNION-ION-SUMMARY`) 런처의
`\[A2-10\]\[LINE-SATURATION-SUMMARY\]` 앵커에 걸리지 않는다 — **오탐 0**.
⟹ 반복 2회 × 1줄 = 2, 런처 `grep -c == 2` 가 옳다. 이 확인이 없었으면 8시간 GPU 를 태우고
수용 블록에서 죽을 수 있었다.
(3) 판독기 `log_blocks` 의 비-census SUMMARY 처리 — census 런에선 발생 불능.
(4) 검수자는 CUDA 빌드·GPU 완주를 실행하지 않았다 — **운전석이 CUDA 빌드를 실행해 심볼 실존까지 확인**(위 ④). GPU 완주는 런 324388 이 판정한다.

### 10.2 ★두 층 「정확-영 τ」 계량은 비교 불가 — 판독 전 확정 (운전석, 2026-08-26, 코드 실측)

런 `324388` 초반 stderr 에 `cmf_fine` 센서스가 `exact_zero_tau=81,265,890` 을 냈다. 이 단이
겨누는 `zero_opacity_emitting_rows` 와 **이름도 현상도 닮았으나 같은 것이 아니다.** 판독·판정에서
섞어 읽는 것을 막기 위해 소스 실측으로 차이를 확정한다. **어느 쪽도 다른 쪽의 분자·분모가 아니다.**

| 축 | `cmf_fine` `exact_zero_tau` | A2-10 `zero_opacity_emitting_rows` |
|---|---|---|
| 소스 | `lumina_cmfgen.c:5418` | `lumina_plasma.c:14075` |
| 판별식 | `tau_validity[cell]==A208_EXACT_ZERO` — **플래그** | `tau==0.0 && n_upper>0.0` — **값 + 발광 요구** |
| 셸 범위 | **전 셸** (`s∈[0,NS)`) | ★**셸 0 단독** (`shell!=0` 이면 즉시 return 0, `:14045`) |
| 원소·이온 | **전 원소·전 이온** | ★**Z∈{26,27,28} ∧ ion==target_ion(=1)** (`a210_line_saturation_target`, `:14027-14030`) |
| 선 범위 | fine 창 `[line_nu_lo,line_nu_hi]` ∧ 약선 미절단(`line_use[l]`) | 진단 후보 행 (provenance 가드 통과) |
| 위상 | 물질 구성 시점 (`INIT_SHARED_GAUSSIAN`) | `REQUESTED_TE` radeq 위상 |
| 단위 | 셀 (line×shell) | 진단 행 |

⟹ 스코프가 **셸·이온 두 축에서 동시에** 다르다. A2-10 쪽은 셸 0 의 Fe/Co/Ni ion 1 뿐이므로
`cmf_fine` 총수의 **극히 일부 부분집합조차 아니다**(판별식이 달라 포함관계도 성립하지 않는다).

★**추가 함정 — `line_shells=` 라벨은 총수가 아니다.** `:5494` 가 그 자리에 인쇄하는 것은
`signed_cells`(= `A208_VALID ∧ isfinite(tau)`) 이고, 이것은 `exact_zero_tau` 와 **서로소**다.
총 셀 = `signed_cells + exact_zero_tau` 이며 이 항등식은 코드 자신이 검사한다
(`lumina_cmfgen.c:6442` — `sobolev_jbar_cells != signed_cells + exact_zero_tau_cells` 이면 실패).

실측 산술 (런 `324388` 초반):
`signed 27,748,410` + `exact_zero 81,265,890` = **총 109,014,300 셀**, 정확-영 비율 **74.55%**.
`upper_population_cells=129,439,900` 은 창·약선 필터 **전**이라 차 `20,425,600` 이 필터 제외분이다.
⟹ 분모를 `line_shells` 로 잡으면 **293%** 라는 헛수가 나온다. 「잣대부터 감사」 계열의 전형이다.

**처분**: 판정문은 두 계량을 **한 문장 안에서 비율로 엮지 않는다.** `cmf_fine` 수치는 배경 관찰로만
인용하고, §8 분기 판정은 **A2-10 계량 단독**으로 한다. `cmf_fine` 정확-영이 74.55% 라는 사실은
Z-2 설계 입력으로 **조용히 기재**한다(이 단의 표적이 아니다).
