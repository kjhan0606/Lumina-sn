# 게이트 전수조사 — 정적 스윕 + 발화 이력 대조

작성일: 2026-08-01  
범위: `src/`의 C/CUDA/header `getenv` 소비점 전수, `scripts/` 런처 설정점, 지정 로그/`.env`, 기존 `docs/` 감사 흔적  
규율: 읽기 전용 소스 감사. 이 조사에서는 `src/`를 수정하거나 Lumina 실행·신규 판정런을 만들지 않았다.

## 1. 결론

현재 스냅샷에는 **논리 게이트 500개**가 있다. 문자열 리터럴 `getenv` 이름은 447개이고, 동적 이름 조립을 펼치면 AUL scale 45개, DR provenance multiplier 6개, element-wide 보조 게이트 2개가 추가된다. 비-`LUMINA_*` C/CUDA `getenv` 이름은 0개다.

핵심 수치는 다음과 같다.

| 항목 | 수 |
|---|---:|
| 논리 게이트 / CSV 행 | 500 |
| 소스의 literal `getenv` 호출 표현 | 689 |
| 실제 dynamic-name `getenv` 호출점 | 9 |
| 런처에서 설정점이 발견된 게이트 | 384 |
| 런처 설정점 `file:line` 기록 | 11,419 |
| 지정 증거원에서 ON/비중립 값 관측 | 101 |
| 지정 증거원에서 중립값만 관측 | 18 |
| 지정 증거원 기준 휴면 | 381 |
| 기존 문서에 정확한 게이트명 흔적 있음 | 148 |
| 정확한 게이트명 흔적 없음 | 352 |
| OFF-중립성 실측 흔적 있음 | 7 |
| OFF-중립성 `UNTESTED` | 493 |
| 전역 부작용 위험 HIGH / MEDIUM / LOW | 103 / 332 / 65 |

가장 큰 체계 결함은 **OFF-중립성 증거가 500개 중 7개뿐**이라는 점이다. 문서에서 다룬 148개조차 대부분 기능 설계·효과 판정 흔적일 뿐, `unset ↔ explicit OFF ↔ ON(shadow)`의 바이트 계약을 닫지 않았다.

또한 지정된 `/gpfs/kjhan/lumina_runner2/logs/*/RESOLVED_CONFIG*`는 **0개**였다. 같은 6개 런 디렉터리의 `stdout.log`에는 바이너리의 `RUN FOOTER (env/arg as actually used)`가 있어 보조 실측으로 사용했다. 따라서 이 보고서의 `DORMANT`는 “프로젝트 역사 전체에서 한 번도 실행되지 않음”이 아니라 **지정 `RESOLVED_CONFIG` + `scripts/*.env` + 현재 마운트의 6개 RUN FOOTER에서 무장값을 확인하지 못함**이라는 엄격한 뜻이다. 예를 들어 `LUMINA_NLTE_ELEMENT_WIDE`는 Wave 3 문서에 실행 이력이 있으나, 지정 발화 증거원에는 남아 있지 않아 CSV의 이력 상태는 `DORMANT`다.

## 2. 전수성 및 판정 규칙

### 2.1 소비점

- `src/**/*.{c,cc,cpp,cu,h,hpp}`의 모든 `getenv(...)`를 조사했다.
- 같은 이름을 여러 파일/라인에서 읽으면 한 행으로 합치고 모든 소비 라인을 `consumer_sites`에 `|`로 기록했다.
- 동적 호출도 실제 가능한 이름으로 펼쳤다.
  - `src/lumina_atomic.c:679,687,689,691,693`: `LUMINA_AUL_SCALE[2..9]_{FACTOR,ZMASK,IONMASK,LAMBDA_MIN,LAMBDA_MAX}` 45개
  - `src/lumina_plasma.c:7981`: `LUMINA_DR_BOOST_{BADNELL,NORAD,MAZZOTTA,AUTOSTRUCT,EST_ISOEL,CMFGEN}` 6개
  - `src/lumina_element_wide.c:43,106-107`: `LUMINA_NLTE_ELEMENT_WIDE_{COMMIT,DUMP}` 2개
  - `getenv(blockers[i])`, `banner_gate_off(..., var, ...)`, element-wide guard-array 소비는 각 실제 게이트 행에 동적 getter와 binding line을 함께 기록했다.
- `scripts/`는 실행 증거와 혼동하지 않도록 `launcher_set_sites`에 별도 기록했다. 런처의 존재는 “설정 가능한 배선”의 증거이지 완료 런의 발화 증거가 아니다.

### 2.2 발화 이력

- 1차 증거: `/gpfs/kjhan/lumina_runner2/logs/*/RESOLVED_CONFIG*` — 0개.
- 2차 증거: `scripts/*.env` — `scripts/parity_baseline.env` 1개.
- 보조 증거: 6개 `/gpfs/.../logs/*/stdout.log`의 바이너리 출력 환경/footer.
- `ACTIVE_OBSERVED`: 비어 있지 않고 일반적인 중립값(`0`, empty, false/off)이 아닌 값이 관측됨. 명백한 default-ON 예외인 `LUMINA_NLTE_RATES_GEMM=0`은 기능을 끄는 발화로 취급한다.
- `SEEN_NEUTRAL_ONLY`: 설정 흔적은 있으나 지정 증거원에서 중립값만 관측됨.
- `DORMANT`: 위 증거원에서 값이 전혀 없거나 무장값이 없음.
- 최근 런 ID는 실제 footer에 무장값이 있는 가장 최근 런을 사용한다. `.env`만 있는 경우 `NO_RUN_ID`, 없으면 `NONE`이다.

### 2.3 감사·OFF 중립성·위험

- 감사 커버는 `docs/**/*.md`에서 정확한 게이트명이 등장하는 경우만 `COVERED`로 셌다. 별칭이나 문맥상 유사 기능을 임의로 join하지 않았다.
- OFF-중립성은 설계 요구문이 아니라 완료된 비교가 문서에 있는 경우만 증거로 인정했다.
- HIGH는 무장 시 레이아웃/테이블 크기, 공용 원자 배열, 전 모델 profile, backend, 전역 mode 또는 여러 하위 게이트를 강제하는 경우다.
- MEDIUM은 물리 산술·마스크·여러 셸/레코드를 바꾸지만 레이아웃 재구성까지 직접 확인되지 않은 경우다.
- LOW는 진단/I/O/self-test 중심이며 생산 배열·레이아웃 변경을 소비점에서 확인하지 못한 경우다.

정확한 unset 초기값/극성을 함수 수준에서 단정할 수 없었던 43개는 `UNRESOLVED`로 유지했다. 나머지는 명시 ternary, guarded branch, 인접 초기값·주석 또는 동적 family 정의에서 unset 거동을 판독했다.

## 3. 군집 요약

군집은 서로 배타적이지 않다. 예를 들어 휴면·미감사·HIGH 게이트는 ①과 ②에 동시에 속한다.

| 군집 | 조건 | 수 |
|---|---|---:|
| ① 휴면 + 미감사 | `DORMANT && UNAUDITED` | 297 |
| ② 휴면 + 전역 부작용 HIGH | `DORMANT && HIGH` | 80 |
| ③ 활성 + OFF-중립성 UNTESTED | `ACTIVE_OBSERVED && UNTESTED` | 99 |
| ④ 폐기 후보 | `DORMANT && proven dead consumer` | 0 |

### 3.1 군집 ① 상위 10개

상위 순서는 전역 mode/layout/table 위험을 먼저 두고, 같은 위험에서는 master/실효 factor를 selector보다 먼저 두었다.

| 게이트 | 기본/극성 요약 | 감사 | 위험 근거 |
|---|---|---|---|
| `LUMINA_TRANSPORT` | unset은 현 상위 수송 mode 유지; 상세 극성 미결 | 없음 | 전역 수송 backend/mode |
| `LUMINA_FIXED_NE_PROFILE` | unset은 profile override 없음 | 없음 | 전 셸 `n_e` profile 강제 |
| `LUMINA_OPACITY_SKIP_Z` | unset 거동 일부 미결 | 없음 | 원소 마스크가 전 모델 opacity를 제외 |
| `LUMINA_NLTE_RATES_GEMM` | unset=GPU rate table ON; `0`=OFF | 없음 | 전 rate backend/table 변경 |
| `LUMINA_SPEC_RANGE` | unset은 컴파일된 spectrum grid | 없음 | bin 수/범위와 출력 layout 변경 |
| `LUMINA_BF_GEMM` | unset은 GEMM 분기 미진입 | 없음 | BF opacity backend 전환 |
| `LUMINA_CMF_NZ` | CMF mode에서 unset fallback 사용 | 없음 | CMF 공간 grid 크기 |
| `LUMINA_CMF_FINE_LAMHI` | unset=4000 Å | 없음 | fine-grid 범위/크기 |
| `LUMINA_AUL_SCALE_FACTOR` | unset factor=1 | 없음 | 공용 A/B/f line 배열 직접 변경 |
| `LUMINA_AUL_SCALE2_FACTOR` | unset factor=1 | 없음 | 두 번째 동적 scale family, 공용 배열 변경 |

### 3.2 군집 ② 상위 10개 — 최우선 감사 대상

| 게이트 | 감사 흔적 | OFF 증거 | HIGH 판정의 직접 이유 |
|---|---|---|---|
| `LUMINA_NLTE_ELEMENT_WIDE` | `OPUS5_WAVE3_CODE_AUDIT.md`, Wave 3 문서 | 있음 | D1/D8: 무장만으로 전역 NLTE layout/33-slot/원자 target 준비 변경, s0·s43 off-target oracle 변경 |
| `LUMINA_EMERGENT_MODE` | `FLUORESCENCE_DESIGN_AB.md` | `UNTESTED` | `setenv(...,1)`로 여러 하위 게이트를 강제하는 mode fan-out |
| `LUMINA_TRANSPORT` | 없음 | `UNTESTED` | 전역 수송 backend/mode |
| `LUMINA_NLTE_STAGE4` | Wave 3/Stage 4 문서 | `UNTESTED` | NLTE 산술·경로를 전역 stage mode로 전환 |
| `LUMINA_CMFGEN_THEN_MC` | 기능 설계 문서 | `UNTESTED` | 결정론/MC 전역 phase 전환과 공유 상태 재배선 |
| `LUMINA_FIXED_NE_PROFILE` | 없음 | `UNTESTED` | 전 셸 전자밀도 profile 강제 |
| `LUMINA_FIXED_TE_PROFILE` | `COEVOLVE_REWIRING_PLAN.md` | `UNTESTED` | 전 셸 온도 solve short-circuit/freeze |
| `LUMINA_FIXED_TRAD_PROFILE` | `COEVOLVE_REWIRING_PLAN.md` | `UNTESTED` | 전 셸 radiation field fit short-circuit/freeze |
| `LUMINA_OPACITY_SKIP_Z` | 없음 | `UNTESTED` | 원소 단위 전역 opacity 제거 |
| `LUMINA_TOPSTAGE_IV` | Opus/Wave 문서 | `UNTESTED` | 상단 이온 reservoir/RHS/closure mode 변경 |

`LUMINA_NLTE_ELEMENT_WIDE`는 OFF↔unset 자체는 byte-identical 실측이 있지만, ON-shadow가 off-target 출력을 바꾼 실물 결함이 이미 있으므로 군집 ②의 첫 항목을 유지한다. 이는 OFF 테스트만으로 D1형 무장 부작용을 막을 수 없고 반드시 세 번째 ON-shadow arm이 필요하다는 선례다.

### 3.3 군집 ③ 상위 10개

모두 현재 마운트의 실제 footer에서 무장값이 확인됐고, 가장 최근 관측 런은 표의 전 항목에서 `coevolve_consume_parity60`이다.

| 게이트 | 감사 | 위험 | OFF 증거 |
|---|---|---|---|
| `LUMINA_SUPER_LEVELS` | Opus/D8 | HIGH — 전역 super-level layout | `UNTESTED` |
| `LUMINA_SUPER_CUTOFF` | Opus/D8 | HIGH — level 수/투영 경계 | `UNTESTED` |
| `LUMINA_PURE_CMFGEN` | parity 진단 문서 | HIGH — 전역 실행 mode | `UNTESTED` |
| `LUMINA_MC_COEVOLVE` | `COEVOLVE_REWIRING_PLAN.md` | HIGH — MC field/state feedback 재배선 | `UNTESTED` |
| `LUMINA_CMF_SOLVE_GPU` | 없음 | HIGH — CPU/GPU solve backend | `UNTESTED` |
| `LUMINA_ARTIS_PARITY` | 기능 감사 문서 | HIGH — 다수 rate/assembler 분기 | `UNTESTED` |
| `LUMINA_BF_OPACITY` | Wave 2 문서 | MEDIUM — 전 모델 BF opacity 산술 | `UNTESTED` |
| `LUMINA_BF_RATE_POPS` | parity 진단 문서 | MEDIUM — BF rate population source | `UNTESTED` |
| `LUMINA_CMF_FINE_ALI` | 없음 | HIGH — fine-grid solve iteration/backend 비용·상태 | `UNTESTED` |
| `LUMINA_CMFGEN_SIGMA_BF` | Gamma/Wave 문서 | HIGH — 공용 BF sigma dataset load/교체 | `UNTESTED` |

### 3.4 군집 ④

폐기 후보는 0개다. `#if 0`/명시적 never guard가 없었고, 저장소의 `scripts/dead_code_audit.py --quiet-passing` 정적 검사에서도 소비 함수를 포함한 unreferenced non-static function이 나오지 않았다. 다만 이 결과는 static local 함수의 전 호출그래프 생존을 증명하지 않으므로, “휴면+런처 setter 없음”만으로 폐기 후보를 만들지 않았다.

## 4. OFF-중립성 배터리 요구 목록

CSV는 군집 ②·③의 **179개 전 항목**에 `off_neutral_battery_priority`와 요구 계약을 기록한다.

| 우선순위 | 수 | 선정 규칙 |
|---|---:|---|
| P0 | 20 | 두 군집의 master/layout/backend/data-source 최상위 10개씩 |
| P1 | 77 | 나머지 HIGH |
| P2 | 67 | 나머지 MEDIUM |
| P3 | 15 | 나머지 LOW 진단/계기 |

P0 전수는 다음과 같다.

| 군집 ② P0 | 군집 ③ P0 |
|---|---|
| `LUMINA_NLTE_ELEMENT_WIDE` | `LUMINA_SUPER_LEVELS` |
| `LUMINA_EMERGENT_MODE` | `LUMINA_SUPER_CUTOFF` |
| `LUMINA_TRANSPORT` | `LUMINA_PURE_CMFGEN` |
| `LUMINA_NLTE_STAGE4` | `LUMINA_MC_COEVOLVE` |
| `LUMINA_CMFGEN_THEN_MC` | `LUMINA_CMF_SOLVE_GPU` |
| `LUMINA_FIXED_NE_PROFILE` | `LUMINA_ARTIS_PARITY` |
| `LUMINA_FIXED_TE_PROFILE` | `LUMINA_BF_OPACITY` |
| `LUMINA_FIXED_TRAD_PROFILE` | `LUMINA_BF_RATE_POPS` |
| `LUMINA_OPACITY_SKIP_Z` | `LUMINA_CMF_FINE_ALI` |
| `LUMINA_TOPSTAGE_IV` | `LUMINA_CMFGEN_SIGMA_BF` |

### 4.1 공통 3-arm 계약

같은 binary, 입력, seed, thread/GPU 설정에서 다음 세 arm을 요구한다.

1. **A — UNSET:** 게이트 및 그 selector/companion을 환경에서 제거한다.
2. **B — explicit OFF/neutral:** boolean은 `0`, factor는 source default identity 값, selector는 비활성 master 아래 canonical neutral로 둔다.
3. **C — ON(shadow):** 게이트 산술·layout 후보를 계산하되 생산 state에는 commit하지 않는다. native shadow가 없는 게이트는 향후 테스트 구현 시 shadow hook이 선행 요건이다.

바이트 판정면은 다음과 같다.

- A↔B: 등록 production state, layout/table manifest, spectrum/oracle, RNG/event counter, stdout 정규화본이 모두 byte-identical이어야 한다.
- A↔C: gate 전용 diagnostic 후보 파일만 차이를 허용한다. 생산 commit state, 대상 밖 원소/셸/레인, 공유 layout/table manifest, RNG/event counter는 byte-identical이어야 한다.
- backend gate는 C에서 양 backend를 모두 계산하되 A backend만 commit하고, candidate 결과는 별도 파일에 기록한다.
- profile/mask/factor family는 최소 범위와 광범위 mask 두 경우를 모두 포함해 off-target 누출을 검사한다.
- self-test/early-exit gate는 일반 production 시작 전후 전역 상태와 파일 side effect를 별도로 비교한다.

이는 테스트 구현이 아니라 요구 명세다. 신규 런은 이번 조사에서 만들지 않았다.

## 5. 기존 OFF 증거 7건

| 게이트 | 증거 |
|---|---|
| `LUMINA_NLTE_ELEMENT_WIDE` | `CODEX_WAVE3_A_IMPLEMENTATION_2026-07-31.md:69-76` — unset vs `0`, s0/s8/s43 `cmp=0` |
| `LUMINA_FIX_BF_MULTI_EDGE` | `CODEX_WAVE1_1_FIXUP.md:17` — unset/`0`/alias 충돌 arm byte-identical |
| `LUMINA_FIX_BF_CONTINUUM_EVENT` | `CODEX_WAVE2_D1_REPAIR.md:18-28` — three-cell OFF oracle |
| `LUMINA_FIX_MA_J_UNCLAMP` | 같은 Wave 2 batch OFF oracle |
| `LUMINA_FIX_MA_NO_LINE_THERM` | 같은 Wave 2 batch OFF oracle |
| `LUMINA_FORMAL_CONS_WINDOW` | `VERIFICATION_REGISTERS.md:24` — OFF 동일성 18/18 및 ride-along byte identity |
| `LUMINA_JBAR_DAMP_UNIFY` | `VERIFICATION_REGISTERS.md:34` — parity50 OFF identity; 단 전체 출력 strict byte identity가 아니라 1-ulp 잔여를 명시한 제한 증거 |

## 6. 한계와 UNRESOLVED

- 지정 `RESOLVED_CONFIG` 파일이 0개라 footer fallback을 사용했다. CSV의 모든 행에 이 증거 공백을 명시했다.
- 43개 게이트는 local getter만으로 exact unset initializer/극성을 닫지 못했다. 해당 행은 값을 만들어내지 않고 `UNRESOLVED`로 남겼다.
- 문서 커버는 exact-name 방식이므로 별칭만 사용한 감사는 `UNAUDITED`로 남을 수 있다. 반대로 이름이 등장했다고 완전한 물리 감사가 끝났다는 뜻도 아니다.
- dead path는 보수적으로 판정했다. 호출그래프/링커 수준의 완전한 reachability 증명이 없는 항목은 폐기 후보로 승격하지 않았다.
- 감사 중 외부에서 소스 라인이 이동한 흔적이 있어, 상세 CSV는 최종 시점의 현재 소스에서 소비 라인을 다시 추출했다. 고정한 소스 집합 digest는 아래와 같다.

```text
src C/CUDA/header sorted sha256 manifest digest:
40f27d564dcdae151c9a31fd086d8cb63db77e14f62210582a77f93f94f483b5
```

상세 원장: `docs/gate_census_2026-08-01.csv`
