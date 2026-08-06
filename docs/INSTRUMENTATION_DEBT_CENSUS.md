# 계측·배선 부채 census (C1–C11)

> ## ⚠ L3 적대 검수 판정 (Fable, 2026-08-07) = **B(검증 대상), C(확정) 아님**
> *"하한 목록으로는 신뢰하되, **폐합 문서로는 신뢰 불가**. \"수리됨\" 표기를
> \"수리 주장(음성대조 미기록)\"으로 강등하고, 집계 grep 패턴을 커밋된 스크립트로
> 승격하기 전에는 이 문서를 폐합 근거로 인용하면 안 된다."*
> 검수가 **부류 4개를 새로 찾았고(C8–C11)**, 기존 부류의 누락 인스턴스와
> census 자체의 잣대 결함 6건을 지적했다. 아래에 전부 반영했다.

2026-08-06/07 (운전석 기계 census). 착수 계기 = user *"아예, 계측 및 배선을 전체적으로 한번 볼까?"*

부류는 발명한 것이 아니라 **2026-08-06 하루의 실제 발견에서 유도**했다. 그날 물리 결함보다
계측·배선 결함이 압도적으로 많이 나왔고, 전부 *"목록을 좁혀서 놓쳤다가 전수로 다시 세니
나왔다"* 는 같은 패턴이었다.

## 방법 기록 — Codex 위임 실패와 전환

Codex 발주가 **4회 연속** 탐색 중 종료했다(exit 0, 최종 메시지 없음).
범위 8문항→3문항→7부류→**1부류**, 예산 리셋, `-o /tmp` 전부 무효.
`reasoning effort: xhigh` 로 도는데 저장소가 커서 탐색 단계에서 소진된다.

⟹ **이 저장소에서 Codex 는 개방 탐색을 완주하지 못한다.**
전환: **증거는 운전석이 기계적으로 모으고, 판정만 위임한다.**

---

## ★C8 fail-open 소프트 가드 — C2 의 여집합 (Fable Q1-1, HIGH)

C2 는 **하드 거부**만 셌다. 그 여집합 — **발화하되 멈추지 않는 검사** — 이 부류로 없었다.
하드 거부는 런처가 죽어서 스스로 드러나지만 **fail-open 은 영원히 조용하다.**

| 실물 | 근거 |
|---|---|
| 0-K 형상 fail-open **805회 발화** | 50셸 모델을 30열 npy 로 805번 경고하며 그냥 돌았다 |
| `src/lumina_element_wide.c:1814-1830` **16종 env 가드** | 실측 확인. `legacy_guard_configured_count` 로 **세기만 하고 아무것도 막지 않는다**(:2021 집계, :2327 덤프) |

## ★C9 계약 목록 사본 비동기 (Fable Q1-2, HIGH)

같은 "금지 노브" 계약이 **서로 다른 3개 사본**으로 존재하며 원소가 다르다 — 실측:

| 사본 | 원소 수 |
|---|---|
| `src/lumina_plasma.c:14954-14964` (하드 FATAL) | **10** |
| `src/lumina_element_wide.c:1814-1830` (진단 전용) | **16** |
| `validation/a2_16/FATAL_CONTRACT_SWEEP.json` (내 산출물) | **7** ← 화석 |

C1 의 "핀이 두 곳에 이중"과 **동일 기전이 계약 리스트 층에서 이미 3중**으로 재발해 있었다.
census 는 그것을 사건 디테일로만 적고 부류로 승격하지 않았다.

## ★C10 설정되지만 소비되지 않는 env — silent no-op (Fable Q1-3, MED-HIGH)

C2 의 **역방향**: "env 설정 → 런 사망"이 아니라 "env 설정 → 아무 일도 없음".
실측: plasma 금지 검사는 `nlte_solve_all` 안에 있고 `src/lumina_main.c:656-663` 이
**`enable_nlte && iter >= nlte_start_iter` 일 때만** 호출한다.
⟹ 비-NLTE 런처가 `LUMINA_FROZENIN=1` 을 설정하면 **죽지도 않고 효과도 없다.**
작성자는 frozen-in 이 켜졌다고 믿는다.

**이것은 C2 처분을 바꾼다** — 157 중 비-NLTE 런처에 "의도적 실행 불가" 표시는 **오처분**이다.

## ★C11 비-덱 소비 입력의 계보 부재 (Fable Q1-4, MED)

C5 는 덱만 다룬다. 그러나 GPH_JTABLE/TE_TABLE 게이트가 로드하는 CMFGEN J-table·T_e table
은 캠페인 교리("고리가 소비하되 생산하지 않는 것")의 정중앙인데 계보 행이 없고,
덱과 달리 `deck_regen.py`/`DECK_PROVENANCE.json` 같은 **수리 인프라 자체가 없다.**

---

## C1 기대값 고정

특정 덱/epoch 에 묶인 하드코딩 기준선이 그 사실을 표시하지 않은 채 검증에 쓰이는 곳.

| 실물 | 상태 |
|---|---|
| `tests/zinert_canonical_tau_fixture.c` + `scripts/run_zinert_selftest.py` | **수리 주장(음성대조 미기록)** — 덱별 표 + 미등록 fail-closed. 핀이 **두 곳에 이중**이라 하나만 고치면 증상이 동일해 오진을 유발했다 |
| ⚠**검수 지적(Q2, HIGH)**: 기계 기준이 "64자리 sha256"뿐이라 **C1 을 낳은 사건 자체(FNV64+정수 카운트 핀)를 자기 grep 이 못 찾는다.** 누락 형태 = 16-hex FNV64 · 정수 카운트 핀 · float 기대값 · `tests/*.c` 전수 | 미수리 |
| 64자리 sha256 리터럴을 든 python 검증 스크립트 **4개** | `a2_06_l1bb_gate.py` · `emiss_t5_rank1.py` · `uv_t2n9_offline.py` · `verify_trad_fix.py` — 기대값인지 기록값인지 **미판정** |

## C2 계약 강화 시 호출자 미이관 — **최대 항목**

정본 `validation/a2_16/C2_CALLER_MIGRATION_CENSUS.json`.
하드 거부 env **12종**, **영향 런처 157개**(앞선 집계 98은 과소였다).

| env | 설정 / 발동 |
|---|---|
| `LUMINA_NLTE_FLOOR_REG` | 146 / **71** |
| `LUMINA_FROZENIN` | 92 / **37** |
| `LUMINA_TE_TRAD_RATIO` | 78 / **78** |
| `LUMINA_NLTE_FLOOR_MODE` | 18 / 9 |
| `LUMINA_TRAD_COLOR_FIX` | 12 / 12 |
| 나머지 7종 | 각 0–2 |

처분 2갈래: **의미 중립 제거**(앞 2종, 죽은 노브) 대 **의도적 실행 불가 표시**(나머지 10종 —
실제 기능이었고 A2-07 이 물리적 판단으로 금지했다. 줄을 지우면 그 런의 의미가 바뀐다).

## C3 음성 픽스처가 실제 결함물에 의존

음성대조가 "진짜 결함을 가진 산출물"을 픽스처로 쓰면, 그 산출물이 고쳐지는 순간
대조가 **조용히 무력해진다**.

| 게이트 | data/ 참조 | 상태 |
|---|---|---|
| `run_gate_battery.py` | 5 | **수리 주장 — 만기 내장(Q3-2)**. `--negative-deck` 을 **살아있는** bare 생산 덱에 고정했는데, 전환 계획이 바로 그 덱을 퇴역시키고 새 덱은 30열 결함을 고친다 ⟹ **결함이 고쳐지는 순간 대조가 무력해진다 = C3 의 정의 그 자체**. 동결 사본에 앵커해야 한다 |
| `run_k_gate.py` | 2 | 미점검 |
| `run_composition_c_gate.py` | 4 | 미점검 |

## C4 생성 파이프라인 불완전

| 실물 | 상태 |
|---|---|
| `deck_regen_*_driver.py` **5개**가 `finalize` 이전에서 멈춤 → 39파일 누락(충돌자료 34 포함) | **수리 주장 — 불완전(Q3-3)**. `deck_regen.py` 로 통합했으나 **구 드라이버 5개가 전부 그대로 존재·호출 가능**하다. 불완전 덱을 재생산하는 진입점이 살아있다. 완전성 게이트의 음성 시연(파일을 빼고 FAIL 확인) 기록도 없다 |

## C5 계보 미기록

| 덱 | vintage manifest | provenance stamp |
|---|---|---|
| **`toy06_19p48d` (생산)** | **없음** | **없음** |
| `_sivcaiv` | 없음 | 없음 |
| `_sivcaiv_fullcov` | 없음 | 없음 |
| `_ftos` · `_links` · `_vac` | 있음 | 없음 |
| `_jnu4` · `_ophys` · `_ophys_exacthyd` | 있음 | 있음 |

**생산 덱에 원자자료 계보가 아예 없다.** user 08-03 동일성 교리 아래에서 대조군 자격의
전제를 충족하지 못한다.

부수: **회귀 대장에 덱 축이 없다** — 138행이 `binary_identifier.argv` 와 `model_geometry`
로만 덱을 간접 노출하며 스키마 필드가 아니다.

## C6 검증기 미실행

`validation/a2_*` 는 단계마다 산출물 2–9개가 있어 **기계적 고아는 없다.**
확인된 미실행은 **L-1bb · L-4 · L-3 · L-5 4종**이며 사유는 오라클 미인증이다
(user 결정으로 BLOCKED 유지). 게이트 **코드 경로 자체가 한 번도 실행된 적 없다**는
위험은 `A2_18_CAMPAIGN_CLOSURE.md` §2 에 기재돼 있다.

## C7 재현 스크립트 불일치

`schema` 를 만드는 스크립트가 저장소에 없는 산출물 **5건**:

| 산출물 | schema |
|---|---|
| `A2_18_GATE_ROLLUP.json` | `lumina-a2-18-campaign-closure-v1` |
| `A2_18_OPHYS_ORACLE_REFUSAL.json` | `lumina-a2-18-ophys-oracle-refusal-v1` |
| `L1_REMEASURE_02A_INTERNAL_AUL.json` | `lumina-layer1-remeasure-02a-v1` |
| `L1_REMEASURE_02B_SOURCE_ANCHOR.json` | `lumina-layer1-remeasure-02b-v1` |
| `L1_VINTAGE_CROSSWALK.json` | `lumina-layer1-vintage-crosswalk-v2` |

**★자기 기재**: 이 중 `A2_18_OPHYS_ORACLE_REFUSAL.json` 은 **내가 2026-08-06 에 인라인
`python3 -c` 로 만든 것**이다. R-8(재현 스크립트 격차)을 비판해 놓고 같은 결함을
재생산했다. 인라인 생성은 편하지만 재현 경로를 남기지 않는다.

---

## ★Q3-5 수리 — 재현 가능한 집계기 (2026-08-07)

`scripts/instrumentation_debt_census.py` → `validation/instrumentation_debt/CENSUS.json`.
세 가지를 강제한다: **패턴을 산출물에 박고 · 패턴 민감도(spread)를 함께 보고하고 ·
트리 상태(git HEAD·미추적 수)를 앵커한다.**

### ★검수의 예측이 실증됐다 — 수치는 패턴 아티팩트였다

C2 영향 런처를 세 패턴으로 세면:

| 패턴 | 수 |
|---|---|
| `export` 접두 + 비-0 | **83** |
| 대입 비-0 (내가 보고했던 것) | **157** |
| 대입 전부 | **166** |
| **spread_ratio** | **2.0** |

⟹ **정직한 진술은 "157"이 아니라 "83–166"이다.** Fable Q3-5 의 우려가 방향까지 맞았다.

### 모집단을 넓히자 모든 수치가 움직였다 (Fable Q2)

| 항목 | 좁은 집계 | **전수** |
|---|---|---|
| C5 덱 계열 | 9 | **107** — vintage manifest **6** · provenance stamp **3** |
| C7 재현 고아 | 5 | **29** (validation JSON 152 중, schema 무 20 별도) |
| C9 계약 사본 합집합 | — | **23종** (plasma 10 · element_wide 16 · scalar 2) |
| C1 고정 기대값 | 4 | 2–4 (spread 2.0) |

**C5 가 가장 크게 변했다: 107 덱 중 계보를 가진 것이 6개뿐이다.**
그리고 `element_wide` 만 아는 노브 8종(`LTE_FLOOR`·`BK_CEIL`·`COLL_FLOOR`·`ION_LOCK`·
`METASTABLE_COLL`·`PER_ION_RESCALE`·`STAGE4`·`DR_FLOOR_CMS`)은 plasma 가 **전혀 막지 않는다** —
C8(fail-open)과 C9(사본 비동기)가 겹치는 지점이다.

## ★census 자체의 잣대 결함 (Fable Q3) — 전부 인정

| # | 결함 | 심각도 |
|---|---|---|
| Q3-1 | **내 증거 사슬 안에 화석**: `FATAL_CONTRACT_SWEEP.json` 이 plasma env 를 7종으로 적었으나 실코드는 **10종**. "나머지 6개는 런처에서 안 켠다"도 틀렸다(4종이 각 1런처에서 발동). **두 정본이 모순인 채 둘 다 validation/ 에 있었다** — 잣대 사고 사례17(화석 CSV) 재생산. → 화석 표시함 | HIGH |
| Q3-2 | C3 "수리"가 C3 를 재생산 (위 표) | HIGH |
| Q3-3 | C4 "수리"의 구 진입점 생존 (위 표) | MED |
| Q3-4 | "수리됨" 전반에 **음성대조 스탬프 없음**. 프로젝트 규약이 "주입 결함으로 FAIL 시연해야 PASS 자격"인데 fail-closed 신경로 시연 기록이 없다 | MED |
| Q3-5 | ~~**census 수치에 재현 경로가 없다**~~ → **수리됨(위 절)**. 원 지적: — 157·12·5·5 어느 것도 커밋된 집계 스크립트가 없다. C7 자기위반의 일반형. 98→157 교정 자체가 "패턴이 수치를 60% 흔든다"는 실증인데 제3 패턴이 190 을 안 낸다는 보장이 없다. **집계한 트리 상태(커밋 해시)도 미기재** — 미추적 ~1,500파일의 더러운 트리 | HIGH |
| Q3-6 | C6 "고아 없음"의 근거가 **산출물 존재**이지 실행 증거가 아니다 | LOW |

**공격에서 살아남은 것**: 하드 거부 env 총수 **12종**(plasma 10 + scalar 2)은 적대 실측과 정확히 일치. C2 의 핵심 수치는 건재하다.

## 총평

**수리 완료 0부류.** C1 일부·C3 일부·C4 는 **"수리 주장"으로 강등**됐다(음성대조 미기록,
만기 내장, 구 진입점 생존). **최대 미결은 C2(157 런처)**, 다음이 C5·C8·C9.

**C6 를 제외한 모든 부류에서 "첫 집계가 좁았다"** — 오늘 네 번 반복된 실패 방식이다
(생산 덱 오인 · 6파일만 비교 · `--deck` 배선 · C2 집계 98→157).
전수로 세기 전에는 어떤 수치도 하한으로 취급한다.

## ★노브 표면 동결 1–4단계 완료 (2026-08-07)

user 지시로 4단계(거부)까지 도달했다. 정본 = `scripts/derive_env_universe.py` ·
`src/env_universe.h`(생성물) · `src/lumina_atomic.c` 의 `lumina_env_surface_report()`.

| 단계 | 내용 | 증거 |
|---|---|---|
| 1 보고만 | 환경의 `LUMINA_*` 중 전집 밖인 것을 센다. 동작 불변 | 음성대조: 가짜 노브 2개 주입 → 정확히 2개 탐지 |
| 2 집계 | 전 런처 정적 집계 | 런처 404 · env 설정 11,101 · **죽은 노브 1,384(12.5%)** · 종류 50 |
| 3 이관 | T3 런처가 must-unset 22종을 **소스에서 유도** | 손으로 목록을 적지 않는다 |
| 4 거부 | `LUMINA_ENV_STRICT=1` 인 런만 미등록 env 거부 | 양성(통과)·음성(거부) 대조 통과 |

배터리: 새 덱·구 덱(통제) 양쪽 `verdict=PASS rc=0` — 동작을 바꾸지 않았음을 확인.

### env 전집 도출 — 손으로 세지 않는다

```
getenv 리터럴 419 · env 배열 38 · snprintf 조립 51 · ★래퍼 호출부 7  ⟹ 합집합 483
```

**(4) 래퍼가 함정이다.** 리터럴이 `getenv` 옆이 아니라 호출부에 있어서
`grep 'getenv("'` 로는 **원리적으로 안 보인다**. 래퍼 스캔이 없었으면
`LUMINA_CONFIG_PREC` 등 3종을 놓쳤을 것이다.
(앞서 보고한 "418종"은 64종 짧았다 — 열 번째)

### ★1단계가 곧바로 드러낸 것 — 죽은 노브 1,384건

런처가 설정하지만 **src 가 읽지 않는** 노브. C10(silent no-op)의 첫 실측이다.

| 죽은 노브 | 설정 런처 |
|---|---|
| `LUMINA_RADEQ_COOL_NLTE_ONLY` | 102 |
| `LUMINA_RADEQ_COOL_ESCAPE` | 99 |
| `LUMINA_RADEQ_COOL_NONNEG` | 97 |
| `LUMINA_COUPLED_TDEP` | 88 |
| `LUMINA_COUPLED_JNU_PHOTOION` | 84 |
| `LUMINA_COUPLED_JNU_LSTAR` · `COUPLED_LAMBDA_STAR` | 81 |
| `LUMINA_RADEQ_LINE_RE` · `RADEQ_LINE_RESPOND` | 78 · 77 |

**복사평형·coupled 솔버 노브**다 — 캠페인이 몇 달째 디버깅해 온 물리다.
검증: `COUPLED_JNU_PHOTOION`·`RADEQ_LINE_RESPOND`·`CN_DAMP` 는 src 언급 **0회**.
`NLTE_FALLBACK_TE` 는 `lumina_cuda.cu:1482` **주석에만** 있다(읽던 코드가 제거되고 주석만 남음).

⟹ **회귀 대장의 `gate_set`(111 항목)에 아무 일도 하지 않은 설정이 들어 있다.**
이 노브를 변화시킨 과거 A/B 는 동일 설정끼리 비교했을 수 있다. 별도 추적 필요.

### ★부수 발견 — 챔피언 설정이 재현 불가다

정본 `validation/instrumentation_debt/CHAMPION_UNRUNNABLE.json`.
참조(챔피언) 런처가 `LUMINA_CMF_EPAY=2` 와 `LUMINA_CMF_EPAY_HOTF=0` 을 켜는데,
A2-17 이 *"retired scalar hot/cold classifier"* 를 제거해 현 코드가 **로드 단계에서 거부**한다.
전수 **재현 불가 런처 36개**(KPEMISS_BSRC_TAU 25 · FIXED_TRAD_PROFILE 4 · CMF_EPAY 계열 3 …).

그 런처들의 과거 결과는 현 코드로 재현할 수 없다. unset 하고 돌리면 같은 런이 아니라
**다른 물리**다. T3 는 두 arm 이 동일 설정이므로 덱 A/B 는 성립하나
**"챔피언 재현"이라 불러서는 안 된다.**

## 미결 (판정 필요)

1. **C2 처분** — 2갈래 중 어느 쪽을, 어느 런처 집합에
2. **C5** — 생산 덱 계보 소급 기록 여부, 대장 스키마 v2(덱 축)
3. **C1 잔여** — sha256 리터럴 4개가 기대값인지 기록값인지
4. **C3 잔여** — `run_k_gate.py` · `run_composition_c_gate.py` 미점검 + **C3 수리 재설계**(동결 사본 앵커)
5. **C7** — 5건에 생성 스크립트 부여
6. **★Q3-5 최우선** — census 집계를 **커밋된 스크립트**로 승격하고 커밋 해시를 앵커할 것.
   그 전에는 이 문서를 폐합 근거로 인용 금지
7. **C8–C11** — 신규 4부류 전수 미실시
8. **C2 처분 재설계** — C10 때문에 "실행 불가 표시"가 비-NLTE 런처엔 오처분이다
