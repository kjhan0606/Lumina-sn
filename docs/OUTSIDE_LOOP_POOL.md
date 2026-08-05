# 「고리 밖 감사」 증거 풀 대장

캠페인 정본: memory `project_outside_the_loop_audit.md`. 개시 2026-08-03.

**이 대장은 용의자 목록이 아니라 증거 목록이다.** 항목의 성과는 "유죄 확정"이 아니라
**풀에서 제거됨**이다. 소거는 단조롭다 — 완전검증으로 빠진 항목은 다시 들어오지 않는다.

**우선순위 = 중요도가 아니라 독립도.** 미검증 의존이 0인 것부터 닫는다(위상정렬).

| 층 | 뜻 |
|---|---|
| **1** | 완전검증 가능. 고리 무관. 파일·코드 대조로 참/거짓 결판 |
| **2** | 불일치의 **존재**는 완전검증되나 **원인**은 고리 얽힘 |
| **3** | 고리 얽힘. 1·2층이 닫히기 전 결론은 잠정 |

**제거는 근거 없이 기재하지 않는다.** 과거 소거도 epoch 확인 후에만 승계한다.

---

## ★상태 어휘 (2026-08-03 정정 — user 지적)

초판이 `잔류` 한 단어에 **성격이 다른 둘**을 섞었다. "전수조사를 했는데 증거인지
아닌지 미정"으로 읽히는 오류였다. 실제로는 **다르다는 것이 검증된 것**과
**아직 비교를 못 한 것**이다.

| 상태 | 뜻 |
|---|---|
| **제거** | 일치 확인. 풀에서 나감. **되돌아오지 않는다** |
| **확정 불일치** | **다르다는 것이 검증됨. 크기도 알려짐. 증거로 확정.** 미정인 것은 증거성이 아니라 **파급 크기**다 |
| **UNRESOLVED** | 대조 불가·정의 미확립. **아직 증거가 아니다** |

## ★분류와 exit 경로

확정 불일치는 성격이 셋이고 **풀에서 나가는 방식이 다르다.** 섞으면 "고칠 수 없는
것"을 고치려다 코드를 다른 코드로 바꾸게 된다.

| 부류 | 뜻 | exit 경로 |
|---|---|---|
| **A. 버그** | 데이터·임포트가 틀림 | **수리 → CMFGEN과 일치 확인 → 제거.** 풀이 실제로 줄어든다 |
| **B. 설계 차이** | 둘 다 정당한 다른 선택 | **수리로 제거 불가.** 정량화 후 수용, 또는 무시 가능함 입증. 기록으로 남되 풀에서 안 나감 |
| **C. 정의 문제** | 같은 이름의 다른 양 | **정의 확립으로 결판.** 코드 수리 아님 |

**애매 사례**: `I9`의 ε clamp는 CMFGEN 대응물 없는 Lumina 고유 구조물이다. 클램프
규약상 제거 대상이나 그것은 **버그 수리가 아니라 솔버 수리**다(클램프를 빼면 솔버가
답을 내야 한다). 별도 부류로 취급한다.

### ★수리 규율 (A 부류에만 적용)

1. **한 번에 하나.** 레버 가산성이 기각됐다(D2) — **여러 변경의 효과는 사후에 쪼갤 수
   없다.** 둘을 같이 고치면 어느 쪽이 무엇을 했는지 영원히 모른다.
2. **매 수리마다 회귀 대장 1행.** 수리는 epoch를 바꾼다 — 13세대 지나는 동안 심부
   T_e가 부호까지 뒤집혔고 아무도 어느 수리가 그랬는지 몰랐다(0j). 대장을 안 통과하면
   4개월 문제를 재생산한다.
3. ⟹ **회귀 대장의 진짜 용도는 과거 발굴이 아니라 앞으로의 수리를 안전하게 만드는
   것**이다. 있으면 수리가 축적되고, 없으면 표류한다.

### 확정 불일치 8건의 분류

| 항목 | 부류 | 근거 |
|---|---|---|
| **I1 Co IV Υ = Fe III** | **A** | 최대 절대차 **0**, 4,455전이 전부. CMFGEN 자신의 Co IV tabulated는 0개 ⟹ 임포트 매핑 오류 또는 미문서화 폴백 |
| **I2b Ni IV `A_ul` 90%** | **A** | 준위결합·단위변환 계열 의심 |
| **I3a Co IV σ 91%** | **A** | 동상 |
| **I2/I3 상위 이온 꼬리**(Ni III·Ni II·Co III·S III) | **A** | 상위 5이온이 85% ⟹ 전역 아닌 특정 임포트 경로 |
| **I4 슈퍼레벨 분할** | **B** | `min(level,100)` 대 `F_TO_S` — 둘 다 정당 |
| **I5 재결합·DR** | A/B 미정 | Co IV→III DR 잔존 대 `[DIE_CoIV]=F,F`. 게이트 문제면 A, 처방 차이면 B |
| **I6 모델 덱** | A/B 미정 | 배열 상이의 성격 미확인 |
| **I7 ν 격자 1,000 대 196,185** | **B** | Lumina는 binned 코드. 196,185빈은 **다른 코드**가 된다 |
| **I9 외곽 반복 12 대 4 / damping** | **B** | 수렴 정책 |
| **I9 ε clamp(대응물 없음)** | **솔버 수리** | 클램프 제거는 버그 수리가 아니다 |
| **I8 광도 31.07배** | **C** | 두 L의 좌표·정의 미확립. 발주 중 |

---

## ★★★층 0 — 조성 (2026-08-03 늦게 발견. 원자데이터보다 근본)

### 정본 = StaNdaRT `snia_toy06_*.dat` (CMFGEN이 아니다)

CMFGEN도 이것을 **입력으로 받았을 뿐**이다. StaNdaRT 8개 코드 중 **7개가 6원소**를
보고한다(cmfgen·tardis·artis·sedona·sumo·urilight·crab; supernu만 C·O 추가).
정본 = `data/standart_data1/input_models/snia_toy06_1h_lowres.dat` (202행·21컬럼).

**실측(운전석)**: `X_Ti`·`X_O`·`X_C`가 **전 202셸 정확히 0**. Ni/Co/Fe는 62셸에서만,
Ca/S/Si는 169셸에서만 양수. v 100–40,300 km/s.
t=0 심부 `Ni 0.995 · Co 0.005`, 외곽 `Ca 0.10 · S 0.35 · Si 0.55`.
19.48d(붕괴 후, CMFGEN `SN_HYDRO_DATA`) 심부 `Fe 0.098 · Co 0.794 · Ni 0.108`.
헤더 47–49행: `X_Ni`는 `X_56Ni` 포함, `X_Fe`는 ⁵⁶Co 붕괴분 포함.

### ★덱 계보 실측 — 오늘 밤 재생성이 조성을 후퇴시켰다

| 덱 | `abundances.csv` 열 | 행(원소) | 상태 |
|---|---|---|---|
| `_sivcaiv` (**캡처 188932·rung1이 쓴 것**) | **51 (=50셸)** | 9 (8원소) | **정상** |
| `_fullcov` · `_links` · `_ftos` (**오늘 생성**) | **31 (=30셸)** | 16 (15원소) | **결함** |

로더는 CSV 폭과 무관하게 `strtod`를 50회 호출한다(`lumina_atomic.c:837`).
libc 실측: `SHORT_ROW nonzero=30 implicit_zero=20 pointer_stalls=20` /
`FULL_ROW nonzero=50 implicit_zero=0`. ⟹ **결함 덱의 셸 30–49는 전 원소 0.**

**⟹ 오늘 만든 덱 셋은 원자데이터는 정렬됐으나 조성이 망가져 그대로 쓸 수 없다.**
원자데이터 게이트가 선·준위만 보고 **조성 파일을 아예 안 봤다.**

### ⚠운전석 오보 철회 (2026-08-03)

- ~~"벤치마크 CMFGEN 런이 NaN·EXIT=2로 끝났다"~~ → **철회.** 그것은
  `toy06_19.48d`(별개 런). 벤치마크 `toy06_19.48d_jnu4`는 **`CMFGEN_EXIT=0`, NaN 0건**
- ~~"캡처의 외곽 20셸에 원소가 없다 / far-outer 결과가 빈 껍데기"~~ → **철회.**
  캡처가 쓴 `_sivcaiv`는 **50셸 정상**. 결함은 오늘 만든 덱 셋에만 있다
- ~~"30셸→50셸 확장 시 입력파일이 안 따라갔다"~~ → **반대.** 원본이 이미 50셸이고
  **오늘 재생성이 30셸로 되돌렸다**
- ~~"CMFGEN을 진리로 놓고 잰 지난 수치는 전부 안 살아남는다"~~ → **철회.**
  재기 전에 오염을 단정했다. 실측 결과 조성차는 지배원소 ≤3.33%·밀도 0.07%로
  배수급 불일치를 설명 못 한다(아래 0-C4). 규약 `feedback_audit_the_yardstick_first`
  ("X=N배는 가설, 분모 실측 증명 먼저") 위반이었다
- ~~"조성 출처 통일(J)은 강제다 / J와 격자범위(F)는 한 건이다"~~ → **정정.**
  J는 3% 정밀화이지 강제가 아니다. F는 **광구를 어디 두는가**의 모델링 결정이라
  데이터 출처 문제와 별건이다(3,900 vs 1,000 km/s 내부경계)

### 0-C 근인 폐합 — 조성 파괴는 어디서 왔나 (2026-08-03, 코드에서 닫힘)

| # | 위치 | 내용 |
|---|---|---|
| 1 | `expand_atomic_data_cmfgen.py:300` | `N_SHELLS = 30` 하드코딩 |
| 2 | `:301-305` | `DEFAULT_ABUNDANCES` 15원소 **균질 자리표시자**(합 0.94993) |
| 3 | `:918-925` | `write_abundances()` 전 셸 동일값 기록 |
| 4 | `:1717` | `main()`에서 무조건 호출 |
| 5 | `deck_regen_fullcov_driver.py:24` | `abundances.csv` ∈ `REBUILT` |
| 6 | `:58` | `copy_companions()`가 `REBUILT` 건너뜀 ⟹ **정본 미복사** |

5·6이 정본 복사를 막고 1–4가 자리표시자를 써 넣었다. `DEFAULT_ABUNDANCES`의 15원소는
`atom_masses.csv`의 15원소와 정확히 일치 — 자리표시자가 원자데이터 원소 목록을
그대로 따라 쓴 것. **A부류(버그) 확정.** 수리 발주 = `docs/ORDER_CD_COMPOSITION_IDENTITY.md`.

### 0-C2 판독기 fail-open — 결함이 런타임에 무증상이었다

`src/lumina_atomic.c:817-842`. 무증상 실패 **16종**(Codex 3회 감사로 열거 완결):
파일 부재 / 헤더 열수 불일치 / 행 필드수 불일치 / 미지 Z 폐기 / NaN·Inf·음수 /
후행 쓰레기·NUL / 중복 Z(양쪽) / 버퍼 절단 / **헤더 순서 역전** / **데이터 행 0개** /
`atom_masses` 무결성 / **셸별 ΣX ≤ 0** / **X > 1** / **`strtod`·`strtol` ERANGE**.

특히: `strtol("4294967302")` → int cast **6** ⟹ **Z=6으로 위장**.
`strtod("1e-9999")` → 0.0 (errno 34) ⟹ 유한값으로 통과.
**헤더만 있고 본문 0행**이면 전 조성 0인데 종료코드 0.

⟹ 수리 발주 D(FATAL 16 / WARN 2). **결함 덱으로 런을 던졌다면 외곽 20셸이 빈 채로
경고 없이 스펙트럼이 나왔다.**

### 0-C3 계보 — Lumina와 CMFGEN은 **다른 StaNdaRT 파일**을 쓴다 (확정)

| | Lumina 정본 | CMFGEN |
|---|---|---|
| 생성기 | `scripts/build_toy06_epoch.py` (byte-equal 재현 확증) | `mk_sn_hydro.py` (우리가 작성) |
| StaNdaRT 원본 | **1시간판** 202 zone | **19.48일판** 807 zone |
| 19.48d 획득 | **자체 Bateman 붕괴** | StaNdaRT가 이미 적용 |
| 재격자 | 중심점 선형보간 → 50셸 | v∈[1000,36000] 절단 → 700점 |
| 추가 변환 | 없음 | **IGE floor 1e-10 + 6원소 Σ=1 재규격화** |

⚠ `build_toy06_epoch.py:226`의 **기본 출력이 현 정본**이다. 인자 없이 실행하면
정본을 덮어쓴다. 이 스크립트를 쓸 때는 출력경로 절대지정 + 정본충돌 가드 필수.

### ★0-C4 차이의 크기 — **측정 완료. 조성은 원인 풀에서 제거** (2026-08-03)

방법: 정본 `geometry.csv`의 고정 50셸 중심에 `SN_HYDRO_DATA` 6원소를 선형보간해
정본 `abundances.csv`와 직접 대조. **격자를 건드리지 않는다**(단일변수).

```
피복        완전 44셸(0–43) / 부분 1셸(44, 5.91%) / 무피복 5셸(45–49)
지배원소    max rel 3.33%  (Si, 셸 4, v=7176)   ← 셸 4에서 IME 3원소 함께 +3.33%,
            max |ΔX| 1.0379e-3 (Co, X=0.76)        IGE 3원소 함께 −0.14%
미량원소    max rel 63.21% (셸 11, v=12272; Co 5.19e-4 vs 3.18e-4) ← IME 영역의 IGE 꼬리
floor 아티팩트  108쌍 (X_lum==0 且 X_cmf≤2e-10). 실차 아님. 실차 있는 쌍 = 60
★밀도       max rel 0.070%, median 0.0456%   ← 재격자 기계 건전성의 독립 증거
```

### ★★0-C4 정정 — **"제거" 철회, 미결로 복귀** (Codex 독립 재계산, 2026-08-03)

**운전석의 3.33%는 표본 규칙의 산물이었다.** 같은 두 배열을 다른 규칙으로 재면:

| 표본 규칙 | major max rel | 비고 |
|---|---|---|
| 셸중심 선형-v (운전석) | **3.334%** | 다섯 수치 전부 소수점까지 재현됨 |
| 셸중심 선형-log v | 3.333% | |
| 유한체적 `ρdV` 평균 | **60.457%** | Co, s11 |
| 순수 체적평균 | 56.950% | |
| **CMFGEN 실제 90-depth 격자** | **67.131%** | `SN_HYDRO_FOR_NEXT_MODEL`, max\|ΔX\| 2.3915e-3 |
| 위를 질량가중 | **78.077%** | |

CMFGEN은 700점을 계산격자로 쓰지 않는다 — `rd_sn_data.f:260-303`이 `log R`에서
내부 90-depth로 다시 보간한다(`LIN_INTERP`, 조성은 선형 X vs log R).
**⟹ 운전석이 고른 규칙이 가장 작은 답을 냈다. 3.33%를 강건한 상한처럼 쓸 수 없다.**

**확인된 것**: 블록 파싱·outer→inner 순서·단위(geometry cm/s, 두 밀도 g/cm³)·
원소 매핑 전부 정확. **`IRON`은 총 Fe**이고 A56 블록은 그 안의 동위원소 분해라
이중계산 없음(`max_abs(element−A56)=0.000e+00`, 90-depth 출력에서도 동일).
이 toy06은 `M(stable IGE)=0`이라 총 Fe가 전량 Fe56일 뿐이다.
다섯 수치는 **"raw 700점 중심보간, 완전피복 44셸 한정"** 조건에서만 유효하다
(전 50셸로 외삽하면 floor 126쌍·밀도 max 74.5% — `np.interp`의 끝값 반복 탓).

**⟹ 조성은 원인 풀에 잔류. 미결.** 필요한 것 = 동일 geometry·density·T/ne seed에서
**조성만 바꾼 통제 A/B**, 또는 s11 IGE의 band별 선불투명도 기여 측정.
**→ 아래 0-C8에서 후자로 종결됨(GPU 런 불요).**

### ★★★0-C8 조성 **제거 확정** — 오프라인 종결 (2026-08-03, GPU 런 불요)

user가 조성-only A/B GPU 런을 승인했으나, 규약(런 발주 3요건 offline-first)에 따라
기전을 오프라인에서 특정하는 과정에서 **런보다 강한 답이 나와 발주하지 않았다.**

#### (a) A/B가 애초에 잘 정의되지 않았다 — 격자 모호성 실측

v=12272 km/s(셸 11)에서 "CMFGEN 조성"이라 부를 수 있는 값들:
```
700점 선형보간      4.01e-4
90점 선형보간       1.99e-3     ← 상한
90점 로그선형보간   3.25e-5     ← 하한
Lumina arm A        6.54e-4     ← 이 대역 안에 있다
```
IGE가 **560 km/s에 3자릿수** 급감한다(v=11815에서 1.07e-2 → v=12376에서 8.69e-6).
이런 기울기 위에서는 **보간 규칙이 값을 61배 흔든다.** "Lumina 조성 대 CMFGEN 조성"을
재면 물리가 아니라 **표본 규칙**을 재게 된다.
⟹ 90-depth 정본은 `SN_DATA_INPUT_CHK`(`rd_sn_data.f:671`, 판독 직후 검사덤프).
   `SN_HYDRO_FOR_NEXT_MODEL`(런 종료 산출)과 조성은 **바이트 동일**(6종 max\|Δ\|=0).

#### (b) 원소 질량은 보존된다 — 정확한 항등식 (보간 무관)

`M_Z = Σ_s X_Z(s) ρ(s) V(s)`, 균질팽창이라 `V ∝ v_out³−v_in³`. 공통구간 3900–35932 km/s:
```
밀도 표현 차를 나눠 제거한 조성만의 오차
   CMFGEN 700점 / StaNdaRT 807존 :  1.00000  (전 6원소, 완벽 보존)
   Lumina 덱     / StaNdaRT 807존 :  Si/S/Ca 0.99942 · Fe/Co/Ni 1.00052
```
**s11에서 값이 61배 흔들려도 총 원소 질량은 0.06% 안에서 맞는다.** 급기울기 구간이
품은 질량이 그만큼 적다.

#### (c) 상한 감도 시험 — A/B런이 잴 값을 오프라인으로 계산

arm B = CMFGEN 90-depth를 **arm A와 같은 방법**(셸중심 선형보간)으로 재표본.
이는 전이층 IGE를 최대화하는 **defensible 상한**이다(s11에서 3.04× arm A).
대역 선불투명도 `χ_line ∝ Σ_lines n_low f_lu λ`, `n_Z ∝ X_Z/A_Z` 근사.
**비(比)를 내므로 준위분포 가정이 분자·분모에서 상쇄된다.**

```
전 44 교체셸의 최대 χ_B/χ_A       설명 대상 격차
   EUV  1.00117  (셸 10)  = 0.117%     u(EUV) 5.13×
   FUV  1.01312  (셸 10)  = 1.312%     u(FUV) 2.53×
   UV   1.01719  (셸 10)  = 1.719%     s12+ FUV 기근 13-20×
효과 국소성: 셸 0–2 정확히 1.0(순수 IGE 코어) · 셸 3–11만 변화 · 셸 12+ 양쪽 IGE=0
```

**EUV 세 자릿수·FUV 두 자릿수 부족.** 근사의 강건성: 셸 11 IGE의 FUV 몫이 실제로
proxy의 **10배**(0.39%→3.9%)여도 비는 1.08(8%), **100배**여도 1.80 — 여전히 2.53× 미달.
(proxy 내부정합 교차확인: s8에서 단위 X당 Co의 FUV 불투명도는 Si의 4.25× — s11의
Co 몫 0.4%와 정합.)

| 항목 | 처분 |
|---|---|
| **조성 (질량분율)** | **✅ 원인 풀에서 제거.** EUV/FUV 격차 기여 ≤1.7%(상한) |
| 지난 코퍼스 | 조성 때문에 무효화되지 않는다 |
| 동일성 | **여전히 미성립**하나 그 실체는 "급기울기 전이층의 격자 표현 모호성"이며 물리적 귀결이 ≤1.7%다. 별건 J는 **정밀화 항목으로 강등** |
| GPU 런 | **미발주.** 승인은 받았으나 오프라인 결론이 더 강하다(정확 항등식 + 상한). user가 직접 확인을 원하면 arm B 덱 생성기는 `~/.lumina_scratch/build_armB.py`에 준비돼 있다 |

**부수 발견**: 정본 덱의 `tau_sobolev.npy`가 **(2565342, 30) — 30셸이고 전 원소 0**.
죽은 자리표시자이자 30셸 화석. 소비처가 50셸을 기대하면 또 하나의 fail-open. **별건 K.**

⚠ **운전석 자기정정 2건 (같은 날)**: ① "지난 코퍼스 전부 무효" → 재기 전 단정,
철회. ② "조성 = 제거, 코퍼스 유효" → 표본 규칙 미검증 상태의 단정, 철회.
**두 번 다 `feedback_audit_the_yardstick_first` 위반**이다. 세 번째로 적을 문장은
"미결"뿐이다.

### ★★★0-C5 `n_e` 잣대 붕괴 — 1.92×는 Lumina 대 CMFGEN 값이 아니었다

Codex가 "운전석이 대조하지 않은 입력"으로 지적해 추적한 결과.

```
                                  n_e(s0)              median 비
Lumina 덱 seed                  1.61246e+09              —
Lumina 수렴                     4.36158e+09       seed 대비 1.91676×  (전 셸 1.680–2.772)
CMFGEN seed (SN_HYDRO_DATA)     5.07439e+09       Lumina seed 대비 1.93479×
CMFGEN 수렴 (RVTJ)              4.85287e+09
```
**전부 4.4–5.1e9에 모이고, 혼자 튀는 것은 Lumina 덱 seed뿐이다.**

실제 수렴 대조 산출물 `validation/gate_b_dual_oracle/phase1/oracle_vs_cmfgen.md`:
```
s0  n_e  Lumina 4.621814e+09  CMFGEN 4.852872e+09  ratio 0.952387   (RVTJ)
s8  n_e  Lumina 7.496125e+08  CMFGEN 7.341043e+08  ratio 1.021125
s45 n_e  Lumina 1.017327e+05  CMFGEN 2.004388e+05  ratio 0.507550   ← s45는 CMFGEN 격자 밖
```

**⟹ 수렴 `n_e`는 광구에서 5%·s8에서 2% 안에 맞는다.**

**★추적 종결 — 이것은 재발견이었다.** 메모리 정본
`project_artis_parity_campaign.md`에 이미 3중 사살 기록이 있다:
- **07-23 Dig A**: 1.92× **기각** — 기준 `electron_densities.csv`가 CMFGEN이 아니라
  `build_toy06_epoch.py:117`의 **⟨Z⟩=1 placeholder**(`ne = n_atom * 1.0`, 주석
  "singly ionized — photosphere search"가 자기선언). 산출물 `logs/coevolve_consume_parity8/analysis/dig_A_ne/`
- **dig_F9**: s8 정합 1.04× ("구 n_e 1.92× 잣대 최종 정리")
- **parity38**: n_e/CMFGEN 중앙 0.975 ("과거 1.92× 류 해당 없음")

**⟹ 결함의 실체는 "인덱스 좀비"다.** 정본 파일에서 07-23에 죽은 값이
`MEMORY.md` 인덱스 행("실미결=n_e 1.92×")에 살아남아 매 세션 로드됐고, 운전석이
08-03 대장·발주서에 그대로 옮겨 적었다. **오늘 측정은 Dig A의 독립 재확인**
(다른 경로: Dig A=⟨Z⟩ 분석 / 오늘=seed·수렴·CMFGEN 4점 대조 — 같은 결론).

| 처분 | |
|---|---|
| `n_e 1.92×` | **기각 확정** (07-23 원판정 + 08-03 독립 재확인). 인용 금지. `MEMORY.md` 인덱스 수리 완료 |
| 인덱스 좀비 | **계보축 교훈**: 정본에서 죽은 값이 요약·인덱스에서 부활한다. 잣대 사고 원장 사례 추가. 소거의 단조성은 **원장뿐 아니라 그 요약에도** 강제해야 한다 |
| 실물 결함 | 덱 `electron_densities.csv` = 광구탐색 seed(⟨Z⟩=1), 광구에서 ~3× 낮음. 솔버가 교정하므로 무해하나 **파일명이 "electron_densities"라 진리로 오인되는 함정** — Dig A 때도 오늘도 같은 함정이 작동했다 |
| s45 불일치(0.51×) | CMFGEN 계산격자 밖(45–49 무피복)이므로 **비교 자체가 무효**. 별건 F |

### ★0-C7 수리 결과 — **검증 완료 / 생산 배포 분리 기재** (2026-08-03)

발주서 `docs/ORDER_CD_COMPOSITION_IDENTITY.md` v4. Codex 3회 검수(반박 26건) 후 구현.

| 발주 | 내용 | **검증 완료** | **생산 배포** |
|---|---|---|---|
| **D** | `lumina_atomic.c` 조성 판독 fail-closed (FATAL 16 / WARN 2 / 문서 1) | ✅ **게이트 19/19** | ❌ **미커밋·미빌드** |
| **C1** | 확장기에서 조성 생산 제거 | ✅ G6 | ❌ 미커밋 |
| **C2** | 덱 드라이버 5기 형상 게이트 | ✅ 정적 | ❌ 미커밋·**미실행**(드라이버 재생성 안 돌림) |
| **C3** | 결함 덱 3기 조성 복원 | ✅ G1–G5, G7 | ✅ **디스크 반영됨**(데이터는 직접 소비) |

**게이트 실적** (grammar-debug 실행, 운전석):
```
D  FATAL 16/16 비영종료 · WARN 2/2 · 정본 대조 1/1        PASS=19 FAIL=0
   음성대조: 수리 전 코드로 빌드 → PASS=0 FAIL=19, 종료 1
   정본 판독 실물: [D5][WARN] missing Z: 12,13,21,22,23,24,25 / 종료 0 / D6 미발화
                   = G10 사전등록 기대치와 정확히 일치
C  G1 바이트동일 · G2 50열=50행 · G3 원소집합+O/C=0 · G4 max dev 2.2204e-16
   G5 균질원소 0 · G6 정적 · G7 사전해시 9/9 OK        9/9 PASS
   음성대조: NEG-G1·NEG-G2 (30열 결함 사본 주입 → FAIL 확인)
   ★G6 음성대조는 러너에 없어 운전석이 추가 — 수리 전 판(git HEAD)에 같은 술어를
     걸어 FAIL 확인(`write_abundances 정의=1, main 호출=True`). 그 전까지 G6은
     "아무것도 안 재는 것과 구별 불가"였다
```

**무결성 확인** (운전석):
- `src/lumina_atomic.c`의 D 변경 6건 생존, mtime 22:18 — C가 안 건드림
- G7 `sha256sum -c` 9/9 OK — 조성만 바꿨고 원자데이터는 안 움직임(단일변수 성립)
- 드라이버 5기 신규 코드에 clamp/정규화 패턴 0건
- 결함본 3기 `abundances.csv.defective_20260803`(2717 B·30열·15원소) 보존, 삭제 없음

**⚠ 배포 잔여 (bakefix5 전례 방지용 명시)**:
1. D·C1·C2 **미커밋**. 커밋은 user 요청 시에만
2. **생산 GPU 바이너리 미빌드** — D는 전 런이 타는 판독 경로다. 재빌드 없이는
   생산 런에 반영되지 않는다
3. C2 드라이버는 **수정만 됐고 재생성을 돌리지 않았다** — 다음 덱 생성 때 검증됨
4. **이 항목은 "결착"이 아니다.** 위 3건이 남아 있는 한 대장에 폐합으로 적지 않는다

### ★★★0-K `.npy` 형상 fail-open — **가설이 아니라 805회 발화한 이력** (2026-08-04)

발주서 `docs/ORDER_L0_CLOSURE_BY_CODEX.md` (Codex 저작) 검수 중 운전석 실측.

#### 코드 (판독기 2곳, D가 안 고친 것)

```c
src/lumina_atomic.c:474   WARNING: tau_sobolev [%d x %d] != expected [%d x %d], reinitializing
src/lumina_atomic.c:477   opacity->tau_sobolev = calloc(n_lines * n_shells, ...)
src/lumina_atomic.c:490   WARNING: transition_probabilities cols %d != n_shells %d, reinitializing
src/lumina_atomic.c:493   opacity->transition_probabilities = calloc(tr * n_shells, ...)
```
`lumina_atomic.c` 전체 WARNING 3건 중 **2건이 이 부류**(나머지 1건은 `:1355` level_offset).
D는 조성 판독만 fail-closed로 만들었고 **이 둘은 그대로 남았다.**

#### 발화 이력 실측 (`logs/` 전수)

```
재초기화 경고가 찍힌 로그 파일 813개
  812  tau_sobolev [RxC] != expected, reinitializing
  463  transition_probabilities cols 30 != n_shells 49
  267  transition_probabilities cols 30 != n_shells 50
   40  cols 30 != n_shells 1
   16  cols 30 != n_shells 34
   15  cols 30 != n_shells 59
    4  cols 30 != n_shells {78, 66, 62, 58}
              transition_probabilities 합계 805
```
**셸 수 49·50이 730건** — 옛 DDC15 시절이 아니라 현재 격자다.

#### 귀결 (코드로 폐합)

`transition_probabilities`는 **게이트가 꺼져 있으면 런 전체를 지배**한다:
```c
lumina_main.c:178   int enable_transprob_update = 0;              /* 기본 OFF */
lumina_main.c:179   if (getenv("LUMINA_DYNAMIC_TRANSPROB") && atoi(...) > 0) enable=1;
lumina_main.c:602   if (enable_transprob_update && iter >= config.hold_iterations)   /* hold=3 */
```

배열이 전부 0이면 macro-atom 전이 선택이 이렇게 된다:
```c
lumina_transport.c:372   double tp = transition_probabilities[tid*n_shells + shell];
                         probability += tp;                        /* 0 누적 */
                         if (probability > probability_event) ...   /* 절대 참 안 됨 */
lumina_transport.c:387   if (!found) {
                             if (block_start >= block_end) { MA_BB_EMISSION; break; }
lumina_transport.c:394       int tid = block_end - 1;              /* ★항상 마지막 전이 */
```
⟹ **확률 추첨이 아니라 매 블록에서 결정론적으로 마지막 전이를 고른다.**
미묘한 편향이 아니라 macro-atom 물리의 붕괴다.

#### 캡처 188932는 무사 (대조군)

```
Transition probs: DYNAMIC
tau_sobolev: [2584132 x 50] (expect [2584132 x 50])
transition_probabilities: [7752396 x 50]
```
재초기화 미발화. 30열 화석은 맨 모델 덱 `tardis_reference_toy06_19p48d`에만 있고
(`(7696026, 30)` 비영 데이터·`tau_sobolev (2565342, 30)` 전 원소 0),
`_sivcaiv`(7752396×50)·`_ftos`(6662859×50)는 정상.

#### ★게이트 상태 집계 (측정 완료)

```
경고가 있는 런 디렉터리 804개
  DYNAMIC     760   ← iter 0..hold_iterations-1 만 0배열, 이후 재계산
  FROZEN       36   ← ★0배열이 런 전체 지배. macro-atom 전면 오염
  NO_BANNER     8   ← 미확정
셸 수 분포: {49: 464, 50: 264, 1: 40, 34: 16, 59: 15, 78: 1, 58: 1, 66: 1}
```

| 부류 | 수 | 귀결 |
|---|---|---|
| FROZEN | **36** | 런 전체에서 macro-atom이 매 블록 **마지막 전이 결정론 선택**. 물리 붕괴 |
| DYNAMIC | 760 | `hold_iterations`(기본 3) 동안 0배열. 이후 `compute_transition_probabilities`가 채움 — **초기 iteration만 오염** |
| NO_BANNER | 8 | 배너 없어 미확정 |

#### ★FROZEN 36건의 정체 — **현 캠페인 판정런은 없다** (측정 완료)

```
35건  n_shells = 1     단일셸 toy/selftest (06-22~23: mc · cmf · s4v3-8 · jdump ·
                       therm · mc_scat · cmf_scat · R0beta0 · R0cmfobs · noemit · kA …)
 1건  n_shells = 50    parity_smoke_180688 (07-21) — 이름 그대로 smoke test
```
**단일셸에서 macro-atom 수송은 의미가 없고**, 50셸 1건은 smoke다.
⟹ **캠페인 판정 사슬에 전면오염 런은 없다.**

NO_BANNER 8건은 전부 `n_shells=49`·DDC15 시대(163xxx–167xxx:
`ddc15_frozenin_*`·`ddc15_radeqcool_*`·`ddc15_a3lstar_*`·`ddc15_pc_phase3_*`) —
현 CMFGEN toy06 캠페인이 아니다. **게이트 미확정으로 남긴다.**

#### 잔여 우려 3건 (측정된 것만)

1. **DYNAMIC 760건의 초기 구간** — `iter < hold_iterations`(기본 3) 동안 0배열로
   돌았다. 수렴 후 영향은 **미측정**. 판정에 쓰인 런이 다수 포함될 수 있다
2. **NO_BANNER 8건** — DDC15 시대, 게이트 미확정
3. **fail-open 자체가 열려 있다** — `lumina_atomic.c:474`·`:490`. D가 조성만 닫았다

⟹ 처분: **K-SHAPE**(형상 fail-closed) + **K-FRESH**(첫 소비 전 신선도).
1번은 K-FRESH가 닫히면 자동 해소된다(첫 소비 전 재계산이 강제되므로).

⟹ 처분: 발주서 계약 **K-SHAPE**(형상 fail-closed) + **K-FRESH**(첫 소비 전 신선도).
D와 같은 처방을 이 둘에 적용해야 한다.

### ★★0-H H-TRANSFORM **폐합** — 벤치마크 입력단 clamp 정량 완료 (2026-08-04)

발주서 `ORDER_L0_CLOSURE_BY_CODEX.md` §3.3. 검증기 `scripts/verify_h_transform.py`.
운전석이 grammar-debug에서 실행. **GPU 런 불요, 오프라인 종결.**

```
raw_sum_range = [0.999992, 1.000012]      elemental_floor_injection_sum = 1.416e-07
PASS raw_to_final_max_abs_delta_x            5.14565825216e-06  ≤ 6e-06
PASS sum_x_over_a_max_relative_change        1.19998560018e-05  ≤ 1.3e-05
PASS floor_only_proxy_max_relative_change    1.67931779593e-09  ≤ 2e-09
PASS floor_renorm_proxy_max_relative_change  1.19998560020e-05  ≤ 1.3e-05
DISPOSITION floor=REMOVE  exact-zero=KEEP_ZERO  renormalization=SEPARATE_EXPLICIT_PROJECTION
```

**`mk_sn_hydro.py`의 IGE floor 1e-10 + 6원소 Σ=1 재규격화가 대역 선불투명도에
미치는 영향 = 1.2e-5 (0.0012%).** 조성 감도(0-C8, ≤1.7%)보다 세 자릿수 아래,
캠페인 격차(u EUV 5.13× · FUV 2.53×)보다 여섯 자릿수 아래.

⚠ **처분이 "작으니 허용"이 아니다**: `floor=REMOVE`. 규약상 정확해가 0인 자리를
비영으로 덮는 것은 크기와 무관하게 위반이다. 다만 `mk_sn_hydro.py`는 `/gpfs`의
CMFGEN 런 입력이므로 **제거하려면 CMFGEN 재실행이 필요**하다 ⟹ 별건 결정.

음성 대조 3/3: exact-zero 입력에서 raw는 0 유지·floor 위반 검출(IGE 3항목) /
off-sum 행에서 자동보정 아닌 명시 실패 / 별도 투영 산출.

⚠ 한계: 검증기 산출값이 Codex 예비실측과 1e-13~1e-23 수준으로 일치한다
(`PRELIM` 행). **독립 물리 검증이 아니라 같은 계산의 재현**이다. 값어치는
"수치가 맞다"가 아니라 **"이제 재현 가능한 스크립트가 있다"**(계측 부채 해소).

| 축 | 값 |
|---|---|
| posedness | WELL |
| outcome | DIFFER (효과 실재, 크기 1.2e-5) |
| kind | NUMERIC, DEFINITION |
| disposition | REPAIR(floor 제거) + ACCEPT(효과는 무시 가능) |

### ★★0-G GEN-GUARD **폐합** — 정본 덱 덮어쓰기 차단 (2026-08-04)

발주서 §3.8. `scripts/build_toy06_epoch.py` 수정. 운전석 grammar-debug 실행.

```
거부(음성 대조) 5/5, 전부 rc=2
  출력 생략            error: the following arguments are required: keeper, out
  정본 직접            error: refusing canonical output tree or alias
  정본 별칭 (data/../data/…)  error: refusing canonical output tree or alias  ← realpath 해소 확인
  입력 == 출력         error: refusing input==output (including aliases)
  기존 출력 디렉터리    error: output must be a new directory
양성 대조 1/1, rc=0
  keeper=정본 · out=신규 → 50셸 × 8원소, abundances.csv 2808 B (정본과 동일 크기)
정본 트리 해시 실행 전후 동일 PASS
```
**양성 대조가 통과했으므로 "전부 거부하는 무용 게이트"가 아니다.**

⟹ `build_toy06_epoch.py:226`의 "기본 출력이 정본" 위험(0-C3 ⚠) **제거**.

| 축 | posedness=WELL · outcome=RESOLVED · kind=BUG · disposition=CLOSE |
|---|---|

### ★★0-K K-SHAPE **폐합** / K-FRESH **구조 폐합** (2026-08-04)

발주서 §3.4·§3.5. 구현 Codex, 검증 운전석(grammar-debug).

#### K-SHAPE — 양성 1/1 · 음성 6/6

계약 사이드카 `kshape_contract.txt` (schema·line_list/tau/transprob SHA-256·
n_lines·n_macro_transitions·n_shells·dtype `<f8`·byte_order little·array_order C).
검증기 `lumina_atomic.c:207 validate_kshape_contract`, 호출 `:726`
(**`load_tardis_reference_data` 안**).

```
양성  정상 _sivcaiv → rc=0
      K-SHAPE contract: line epoch 2a0b5f9f...; dtype=<f8 byte_order=little order=C
음성 6/6 전부 rc=1
  30열 정본 덱        [FATAL] invalid contract (lines=2565342/2565342 shells=30/50)
  계약 파일 없음      [FATAL] missing contract
  line_list 해시 불일치 [FATAL] line_list hash/line-epoch mismatch
  계약 n_shells=30    [FATAL] invalid contract (shells=30/50)
  tau_sobolev 절단    [FATAL] tau_sobolev hash/line-epoch mismatch
  stale sentinel      [FATAL] tau_sobolev hash/line-epoch mismatch  ← 형상 정상·값 1개 오염도 검출
```
⟹ `lumina_atomic.c:474`·`:490`의 **"경고 후 0 재할당" fail-open 제거**.
804회 발화 이력(위 참조)의 재발 경로가 닫혔다.

계약 발급 실측 (`scripts/kshape_contract.py write`):
```
_sivcaiv  n_lines=2584132  n_macro=7752396  n_shells=50
_ftos     n_lines=2220953  n_macro=6662859  n_shells=50
정본(30열) n_lines=2565342  n_macro=7696026  n_shells=30   ← 50셸 런에서 정확히 거부됨
```

#### K-FRESH — 구조 폐합 (초크포인트 지배 by-construction)

```c
lumina_atomic.c:748-749   적재 직후 required=1, computed=0            ← 구성상 stale
lumina_plasma.c:6340      tau_sobolev_require_refresh()   required++
lumina_plasma.c:6350      tau_sobolev_mark_computed()     computed=required
lumina_plasma.c:6360      tau_sobolev_assert_fresh()      computed==0 || computed<required → -1
lumina_main.c:256         prepare_solver_owned_tau(...) != 0 → EXIT_FAILURE   ← 반복루프 이전
lumina_cuda.cu:7457       동일 (GPU 진입점)
```
**tau 소유자 = solver 확정.** 덱 NPY는 epoch 검증된 seed일 뿐이다.

⚠ **구조 검증의 성격**: `assert_fresh` 호출은 **1곳**이고 `tau_sobolev[...]` 읽기는
**48곳**이다(운전석 실측). 그러나 두 진입점(`main.c:256`·`cuda.cu:7457`)이 수송 루프
**앞**에 있고 적재 직후가 구성상 stale이므로, **생산자를 건너뛰면 반드시 FATAL**한다.
48 소비처가 전부 그 뒤에 있어 초크포인트가 지배한다 — spot test보다 강한 논거다.

**잔여**: end-to-end 런으로 `[K-FRESH] first consumer=...` 배너를 확인하지 않았다.
D-BUILD 후 실런에서 확인한다.

#### 회귀 (K 변경이 기존 게이트를 깨지 않았는가)

```
CPU 하니스 빌드 rc=0 · D 조성 게이트 19/19 PASS · C 조성 게이트 9/9 PASS
Z-INERT 미접촉 확인: lumina_plasma.c 의 `= 1e-100` 3곳 그대로
D 마커 보존 6건
```

#### ⚠ 배포 순서 제약 (신규)

**새 바이너리는 계약 없는 덱을 FATAL시킨다.** 생산 덱 2기에 계약을 발급했으므로
그 둘은 사용 가능하나, **다른 덱을 쓰려면 먼저 `kshape_contract.py write`가 필요**하다.
30열 정본 덱은 계약이 `n_shells=30`이라 50셸 런에서 의도대로 거부된다.

#### ⚠ 운전석 계측 오류 2건 (기록)

1. **첫 배터리 6/6 MISS는 코드 결함이 아니라 계측 실패였다.**
   `composition_d_harness.c`는 `load_atomic_data()`만 부르는데 K-SHAPE 검증기는
   `load_tardis_reference_data()`(`:519`~)에 있다. **배터리 전에 계측기가 코드에
   닿는지 확인하지 않았다** — 잣대 원장 사례 11(엉뚱한 로그 감시)과 같은 부류.
   전용 하니스 `scripts/kshape_harness.c`를 만들어 재측정.
2. **`gcc … | tail` 뒤 `$?`를 빌드 종료코드로 읽어 실패를 rc=0으로 3회 오독.**
   파이프 뒤 `$?`는 tail의 것이다.

첫 결과를 그대로 보고했으면 **"K-SHAPE가 전혀 작동 안 한다"는 오보**가 대장에 올라갔다.

### 0-Z Z-INERT — 정확한 0이 하류에서 0으로 남는가 (신규, Codex 발견)

운전석 10건 목록에 **없던** 항목. 주 tau 계산 경로에 `1e-100` 대입 3곳:
```c
src/lumina_plasma.c:2597-2600   opacity_skip_z[Z] 마스킹 원소  → tau = 1e-100
src/lumina_plasma.c:2605-2608   이온 population 인덱스 없음    → tau = 1e-100
src/lumina_plasma.c:2623-2626   준위 못 찾음                   → tau = 1e-100
src/lumina_cuda.cu:10692        !isfinite(t)                   → tau = 1e-100
```
셋 다 **데이터 부재**로 물리적 정답이 **정확히 0**(흡수체 없음)인 자리다.
규약 판별식("정확해가 위반 가능한 가드")에 걸린다.

**단 물리 영향은 미확정**: `e^-τ`·`β=(1-e^-τ)/τ`에서 1e-100과 0은 double로 구분
불가. 소비처 grep에서 `tau_sobolev > 0` 시험이나 τ 나눗셈은 **발견되지 않았다**
(매치는 전부 포인터 null 검사). ⟹ **수치적으로 무해할 가능성이 크나, 0원소 선이
수송 후보에 드는지는 별도 계측 필요**(발주서 §3.7 Z-INERT 계약).

⚠ **운전석 자기정정**: 앞서 `src/lumina_cmfgen.c:838`의 `tau_pop = 1e-100`을
Z-INERT 실물로 지목했으나, 그것은 **진단용 round-trip 검사** 안이라 물리 경로가
아니다. 실물은 위 `lumina_plasma.c` 3곳이다.

### ★★0-Z Z-INERT **폐합** + K-FRESH 잔여 해소 (2026-08-04 07:45)

발주서 §3.7. 구현 Codex(`scripts/verify_zinert.py`·`run_zinert_selftest.sh` + src),
검증 운전석(grammar-debug, `bash scripts/run_zinert_selftest.sh` rc=0).

```
[Z-INERT-NEGATIVE]  phantom population rejected rc=1 PASS
[Z-INERT-TAU]       inactive_valid=0 missing_ion=0 missing_level=0
                    active_tau_bits=IDENTICAL PASS
[Z-INERT-POP]       inactive_ground=0 inactive_upper=0
                    active_ground_bits=IDENTICAL active_upper_floor_bits=IDENTICAL PASS
[Z-INERT-CANONICAL-TAU]
    inactive_lines=353770  inactive_nonzero=0
    active_lines=2211572   active_tau_bit_differences=0
    active_tau_fnv64=e093d193a78c0af5  audit_rc=0  PASS
```

비활성 9원소 `Z = 6,8,12,13,21,22,23,24,25` 전부에서:
`ion_nonzero=0 · population_nonzero=0 · line_opacity_nonzero=0 · line_source_nonzero=0 ·
continuum_candidates=0 · emissivity_candidates=0 · heating_cooling_candidates=0 ·
transport_candidates=0`

**⟹ 0원소 선이 수송 후보에 들지 않는다** (발주서가 요구한 카운터 증거).
활성 원소 `Z = 14,16,20,26,27,28` 은 **비트 동일** — 단일변수 성립.

#### ★K-FRESH end-to-end 확인 (잔여 해소)

같은 로그에 배너가 찍혔다:
```
[K-FRESH] first consumer=Z-INERT population fixture
          computed_generation=2 required_generation=2 owner=solver
```
운전석이 "구조 폐합, end-to-end 미확인"으로 남긴 잔여가 **실런으로 닫혔다.**
초크포인트가 실제로 첫 소비 전에 통과되고 `computed == required` 다.

#### 잔여 (범위 밖으로 명시)

**활성 원소의 상위 이온 단계 `1e-300` 은 유지됐다**(`active_upper=1e-300`,
`active_upper_floor_bits=IDENTICAL`). 이 계약의 범위는 **"입력 조성이 정확히 0인
원소"** 였고, 활성 원소의 상위 단계는 population 이 계산된 결과이지 구조적 0 이
아니다. **같은 부류인지는 별도 판정 필요** — 층 1 I9(수치 상수)로 이관.

**CUDA production packet 런 미수행** — NVCC 전체 빌드는 통과. 다음 운전 런에서
`LUMINA_ZINERT_AUDIT=1` 로 `post-nlte-gpu` 카운터 확인 가능(Codex 보고).

| 축 | posedness=WELL · outcome=RESOLVED · kind=BUG · disposition=CLOSE |
|---|---|

### ★★★층 0 계약 추가 5건 (user 2026-08-04 지시: *"전부 0층에 추가. 1번은 선행 계약으로 묶어"*)

T-SEED 준비 중 운전석 실측에서 나왔고 기존 14개 계약 어디에도 없던 것들.
**층 0 계약 14 → 19** (검증 8→13 · 범위판정 4 · 배포 2).

#### ★★★TRAD-FIX — 덱의 T_rad 프로파일이 통째로 대체된다 (**T-SEED 선행 계약**)

```c
src/lumina_atomic.c:622
  if (LUMINA_TRAD_COLOR_FIX)
      for (i2 = 1; i2 < n; i2++) plasma->T_rad[i2] = plasma->T_rad[0];
캡처 실물: [TRAD-COLOR-FIX] T_rad[s>=1] := T_rad[0]=10470 K (W unchanged)
```

| | |
|---|---|
| 덱의 원 프로파일 | 10470 → 3134 K (s10 5868 · s25 4184 · s40 3428 · s49 3134) |
| FIX 적용 후 | **전 50셸 10470 K** — s49 에서 **3.34배 상향** |
| 부류 | H(`mk_sn_hydro.py` floor, 영향 1.2e-5)와 **동일 부류인데 크기가 다섯 자릿수 크다** |

⚠ **모순**: 메모리 정본 `project_gph_alllevel_ab_verdict.md` 가 **"T_rad 전셸 10470핀 =
잣대 결함"**으로 이미 판정했다. **그런데 캡처 188932 가 그 게이트를 켜고 돌았다.**
이 모순의 처분이 없다.

⚠ **T-SEED 선행 사유**: T-SEED 의 seed 가 `T_e = ratio × T_rad`(`lumina_plasma.c:11631`)
인데 그 `T_rad` 가 이미 대체된 값이다. **TRAD-FIX 를 정리하지 않고 T-SEED 를 던지면
seed 민감도가 아니라 "대체된 T_rad 위에서의 seed 민감도"를 재게 된다.**

#### ★★CONFIG-PREC — 설정 우선순위 미감사

덱 `config.json` / argv / env 셋이 같은 값을 다투는데 정본 계약이 없다. 실측 사례:
```
T_e_T_rad_ratio :  덱 config.json 에 없음 → 코드 기본 0.9 → env LUMINA_TE_TRAD_RATIO=1.0 이 덮음
```
**덱이 선언한 상태와 실제로 돈 상태가 다를 수 있고 어느 쪽이 정본인지 규정이 없다.**

#### ★★DECK-FOSSIL — 정본 덱이 여전히 틀린 형상을 담고 있다

```
tardis_reference_toy06_19p48d/tau_sobolev.npy              (2565342, 30)  전 원소 0
tardis_reference_toy06_19p48d/transition_probabilities.npy (7696026, 30)  비영 데이터
```
K-SHAPE 가 fail-closed 로 만들어 **쓰면 죽는** 상태지만 덱 자체는 안 고쳐졌다.
재생성 또는 격리 필요.

#### ★NE-NAMING — 진리처럼 명명된 seed 파일

`electron_densities.csv` 는 `build_toy06_epoch.py:117` 의 ⟨Z⟩=1 placeholder
(`ne = n_atom * 1.0`)인데 이름이 진리를 시사한다.
**함정이 두 번 작동했다** — 07-23 Dig A, 08-03 운전석(0-C5). 개명 또는 파일 내 경고.

#### TE-DEAD — 죽은 배열 (사소, 등재만)

`opacity->t_electrons` 참조 3개뿐: alloc(`:668`) · init(`:670`, `= T_rad`) · free(`:832`).
**읽히는 곳 0.** 입력 상태를 하나 더 들고 있으나 소비되지 않는다.

---

### ★★★0-N/0-F/0-P 실측 (2026-08-04 23:30, 운전석) — 세 계약이 하나로 묶인다

A2-00 발주 대기 중 운전석이 NE-NAMING·DECK-FOSSIL·CONFIG-PREC 을 실측했다.
**셋은 별개 계약이 아니라 같은 사슬의 세 지점이었다.**

#### TE-DEAD **폐합** (등재로 종료)

```
$ grep -rn 't_electrons' src/ | wc -l     ->  4
  src/lumina.h:217          선언
  src/lumina_atomic.c:668   alloc
  src/lumina_atomic.c:670   init  ( = plasma->T_rad[i] )
  src/lumina_atomic.c:832   free
```
`.cu` 포함 전 소스에서 **읽는 곳 0**. 대장이 적은 3개는 `.c` 기준이었고 헤더 선언을
더하면 4개. 값이 소비되지 않으므로 물리 영향 없음. **등재로 종료.**

#### ★★★NE-NAMING — "이름 문제"가 아니라 **경계조건 계약**이다 (등급 상향)

대장의 줄번호 `build_toy06_epoch.py:117` 은 **틀렸다**(그 줄은 `bateman()` 안).
실제 위치는 `:162`. 그리고 사슬이 등재된 것보다 훨씬 길다:

```
:162  ne = n_atom * 1.0            <- <Z>=1 placeholder
:167  tau_es 적분
:168  above = np.where(tau >= tau_phot)
:169  i_phot                        <- 광구 zone 선택
:170  v_inner = v[i_phot]
:171  r_inner = r[i_phot]
:185  T_inner = (L/(4 pi r_inner^2 sigma))^(1/4)
:191  v_edge = linspace(v_inner, v_max, n_shells+1)   <- 전 50셸 격자
:231  electron_densities.csv
```

**즉 placeholder 가 광구 위치·내부 경계 속도·`T_inner`·전 셸 격자를 정한다.**

권위 원본 대조 — CMFGEN `RVTJ` 의 `Electron density / Atom Density` 로 `<Z>(v)` 실측
(계측기 `~/.lumina_scratch/ne_zbar.py`, grammar-debug):

```
CMFGEN RVTJ  ND=90,  v 1025 .. 35975 km/s
  <Z> = n_e/n_atom :  min 0.0346   median 2.2463   max 3.5856
현행 덱             :  <Z> = 1  (전 zone)
모델 격자 202 zone 중 CMFGEN 커버 175 (v<1025 5개, v>35975 22개)
```

| case | i_phot | v_inner [km/s] | r_inner [cm] | tau_total |
|---|---|---|---|---|
| A 현행 `<Z>=1` | 19 | **3900.00** | 6.56398e14 | 2.2990 |
| B CMFGEN `<Z>(v)`, 격자밖=최근접 | 30 | 6100.00 | 1.02667e15 | 7.1389 |
| C CMFGEN `<Z>(v)`, 격자밖=1 유지 | 30 | 6100.00 | 1.02667e15 | 5.6478 |

**Δi_phot = +11 zone · Δv_inner = +56.41% · ΔT_inner = −20.04% · tau_total 3.1배.**
B 와 C 가 동일하므로 **격자 밖 처리 선택에 무관**하다 — 외삽 규약의 문제가 아니다.

★ **양성 대조 성립**: case A 의 `v_inner = 3900.00 km/s` 가 실제 덱
`config.json: v_inner_min_cm_s = 3.9e8` 와 **정확히 일치**. 사슬은 가설이 아니라 재현이다.

`electron_densities.csv` 값 자체: `n_e(CMFGEN)/n_e(placeholder) = <Z>(v)` →
min 0.0639 · median 1.8521 · max 3.5854 (커버 구간 median **2.0000**).

⟹ **처분: 개명이 아니라 계약.** 파일명 변경으로 닫히지 않는다.

#### ★★★DECK-FOSSIL — 정본 덱이 **자기 생성기로 재현되지 않는다** (범위 확대)

등재된 것은 `.npy` 형상 화석이었다. 실측한 것은 그보다 상위다.

```
파일 mtime (data/tardis_reference_toy06_19p48d/)
  electron_densities.csv   Jun 29 14:54
  plasma_state.csv         Jun 29 14:54
  config.json              Jun 29 19:19      <- 4시간 25분 뒤
```

계측기 `~/.lumina_scratch/deck_fossil.py` (builder 를 import 해 상수·열규약 전사오류 제거):

```
덱 config.json 선언
  target_epoch_d 19.48 · v_inner 3.9e8 cm/s · t_exp 1683072 s
  T_inner_K 10020.0 · luminosity_inner_erg_s 3.092725510802548e+42
  -> r_inner = 6.563981e+14 cm
  -> T(L_cfg, r_inner) = 10018.3523 K   (선언 10020.0 과 차 -1.65 K)   내부 정합 OK

현재 생성기 + 현재 입력, 같은 epoch
  최근접 시각 index 26 = 19.48 d  (epoch 오선택 아님)
  L_gen = 1.238648e+43        L_cfg = 3.092726e+42
  L_gen / L_cfg = 4.005038
  T(L_gen, r_inner) = 14172.5490 K   (선언 대비 1.414426배)

L_cfg 에 가장 가까운 시각은 49.61 d (1.55% 차) — 덱 선언 epoch 이 아니다
```

⟹ **덱의 (L, T) 쌍은 자기 자신 안에서는 정합하나, 자기 생성기로 재현되지 않는다.**

`4.005038` 배의 출처 — **가설 4개 전부 기각** (계측기 `~/.lumina_scratch/l_cut.py`):

| 가설 | 검정 | 결과 |
|---|---|---|
| epoch 오선택 | 56개 시각 전부의 L 대조 | ✗ `L/L_cfg ∈ [0.9,1.15]` 는 5.14d(1.083)·45.10d(1.149)·49.61d(0.985) 뿐. 어느 것도 1.000 아님 |
| 파장 절단 적분 | 누적 L(λ) 이 `L_cfg` 에 닿는 λ | ✗ **3557 Å** — 둥근 수도 아니고 광학역을 통째로 버리는 값 |
| 상수 1/4 | `1/4.005038 = 0.249686` | ✗ 0.25 와 0.126% 차 — 정확히 4 가 아님 |
| git 계보 | `git log -- .../config.json` | ✗ **덱 전체가 UNTRACKED** — 버전 계보 없음 |

⟹ **`T_inner_K=10020` / `luminosity_inner_erg_s=3.0927e42` 의 출처는 확립되지 않았다.**
선언된 입력·생성기·epoch 로부터 단순 변환으로 도달할 수 없다. `10020` 이 둥근 수이고
`config.json` 만 4시간 25분 뒤에 쓰였다는 점에서 수기 설정 가능성이 있으나 **증거 없음**.
이 값이 내부 경계 흑체와 광도 재조정 루프를 직접 구동한다(아래 CONFIG-PREC).

#### ★★★CONFIG-PREC — 한 덱이 내부 경계온도를 **두 개** 선언하고 **둘 다 돈다**

```
plasma_state.csv (50행, 열 = shell_id,W,T_rad)
  T_rad/W^0.25 = 14172.549003 K   전 50셸 일정 (max-min = 3.6e-12)
  T_rad[0] = 10470.093240   W[0] = 0.29785873
  T_rad[49]= 3133.594393    W[49]= 0.00238990

config.json
  T_inner_K = 10020.0
```

**14172.549 는 우연이 아니다.** 위 DECK-FOSSIL 측정의 `T(L_gen, r_inner) = 14172.5490 K`
와 **7자리 일치**. 즉 `plasma_state.csv` 는 생성기의 `T_inner` 를 담고 있고,
`config.json` 은 4시간 뒤 다른 값으로 덮였다.

소비 실측 (`grep`, 전 소스):

| 선언 | 코드 진입 | 소비 |
|---|---|---|
| `config.json T_inner_K` = 10020 | `lumina_atomic.c:577` → `config->T_inner` | `cmfgen_solve_J`/`cmf_solve_J` **내부 경계 흑체** `cm_planck(nu, T_inner)` (`lumina_cmfgen.c:2570,2877,3332,3471,3531,3714,3860,3904`), `main.c:695`·`cuda.cu:10827` **T_inner 재조정 루프** |
| `config.json luminosity_inner_erg_s` = 3.0927e42 | `:578` → `config->luminosity_requested` | `plasma.c:1650` 광도비, `cuda.cu:10423` |
| `plasma_state.csv T_rad` (내재 color 14172.549) | `:613` → `plasma->T_rad` | 복사장 seed 전체 |

⟹ **둘 다 살아 있다. 41.4% 어긋난 두 경계온도가 같은 런에서 동시에 작동한다.**

#### ★TRAD-FIX 「3중 모순」 해소 — 셋이 아니라 **둘이고, 하나는 파생값이다**

`docs/ORDER_L0_TRADFIX_TSEED_BY_CODEX.md:386` 이 `10020 / 10470.093 / 14172.549` 를
"셋 중 어느 것도 아니다"로 등재했다. 위 실측으로 각각의 출처가 특정됐다:

| 값 | 정체 | 독립 주장인가 |
|---|---|---|
| **14172.549 K** | 생성기의 `T_inner` (⟨Z⟩=1 광구 위, `L_gen`) — `plasma_state.csv` 에 기록 | ○ |
| **10020 K** | `config.json` 이 4시간 뒤 선언한 값 (`L_cfg = L_gen/4.005`, 출처 미확정) | ○ |
| **10470.093 K** | `= 14172.549003 × W[0]^0.25`. s0 의 **희석값**이지 별개 온도가 아님 | ✗ **파생** |

⟹ **모순은 3중이 아니라 2중이다.** 그리고 `LUMINA_TRAD_COLOR_FIX` 게이트가 전 셸에
박는 `10470.093` 은 **두 후보 중 어느 쪽도 아닌 세 번째 값** — s0 의 희석값을 color 로
오인해 전 셸에 복사한다. 게이트의 자기 주석("keeps the photospheric COLOR")과
실제 동작이 어긋난다.

#### 처분

| 계약 | 상태 | 처분 |
|---|---|---|
| TE-DEAD | **폐합** | 등재로 종료. 읽는 곳 0 확인 |
| NE-NAMING | **등급 상향 → 경계조건 계약** | 개명 불가. `<Z>` 를 조성·이온화에서 산출하거나, placeholder 임을 fail-closed 로 선언 |
| DECK-FOSSIL | **범위 확대** | `4.005038` 배 출처 규명이 선행. 규명 전 재생성 금지(현 덱이 유일한 계보) |
| CONFIG-PREC | **실측 완료, 계약 미작성** | 두 선언의 우선순위 정본 규정 + 불일치 시 FATAL |

★ **이 4건은 A-2 의 선행 조건이다.** A-2 의 L-0 게이트는 "덱의 `W B_nu(T_rad)` 를 넣으면
s0 다섯 대역이 전부 FAIL 해야 한다"를 음성 대조로 쓰는데, 그 `T_rad` 가 어느 경계온도에서
왔는지가 위와 같이 미확정이면 **음성 대조의 의미가 정의되지 않는다.**

---

### ★★★★A2-00 원장 자격 **PASS** — 그리고 **원장 자체가 세대 혼합**임이 증명됐다 (2026-08-04 23:56)

A-2 18단계 중 첫 단계. 저작·구현 Codex(`scripts/cmfgen_oracle_contract.py` 1,271행 +
`scripts/a2_00_oracle_negative_controls.py` + `docs/A2_00_OPHYS_PROFILE.json`),
검수·실행 운전석(grammar-debug, `~/.lumina_scratch/run_a2_00.sh`).

```
8.1 주입대조 7종   rc=0   controls_passed=7/7 failures=0
    delete_one_file 11 · truncate_1024 14 · replace_from_other_run 13
    mtime_only_must_pass 0 · info_declared_nd_plus_one 14 · add_unclassified 15
    POSITIVE current_like_snapshot_ophys_failure 16
8.2 manifest 생성  rc=0   entries=362 unclassified=0
    role_counts={oracle-data:72, oracle-metadata:145, run-log:28, scratch:117}
    manifest sha256=ede416159ec12969... 크기 268,661 B
8.3 unchanged 대조 rc=0   PASS ... mtime_used=false
8.4 O-PHYS 양성    rc=16  MISSING 7 + MISSING_ATTESTATION
판정  A2_00_ORACLE_ELIGIBILITY = PASS
```

★ 음성대조 4번(**mtime 만 변경 → rc=0 이어야 함**)이 통과했다. 검증기가 mtime 에
의존하지 않음을 역방향으로 시연한 것이다.

#### ★★★핵심 발견 — `generation = MIXED_GENERATION_PROVEN`

발주서 §13 무증상 실패경로 **20번**("EDDFACTOR, RVTJ, POP, OBSFLUX 가 서로 다른 CMFGEN
iteration 이다")이 **가설이 아니라 실물**로 확인됐다.

| 링크 | 판정 | 근거 (내용 기반, mtime 불사용) |
|---|---|---|
| `EDDFACTOR ↔ JH_AT_CURRENT_TIME` | **MATCH** | R·V 격자 비트 동일, 주파수 196,185개 비트 불일치 0, `JH.RSQ_J` 대 `EDD.J×R²` **17,656,650개 max 상대오차 0.0** |
| `OBSFLUX ↔ OBS_FREQ` | MATCH | 166,151개, max 5.0e-7 (선언 정밀도 1.1e-6 이내) |
| `RVTJ ↔ *PRRR` 27개 + `GENCOOL` | **MISMATCH** | **n_e 가 90 depth 중 68개에서 불일치** |

**운전석 독립 재계산** (`~/.lumina_scratch/prrr_ne.py`, 자작 파서 — PRRR 은 depth 10개
블록 반복 포맷이므로 블록 전부를 이어붙여야 한다):

```
FeIIIPRRR / CoIVPRRR / S2PRRR  대 RVTJ  (셋 다 동일 결과)
  Radius       max|rel| = 3.247e-05    선언정밀도 6e-5 초과 depth 0/90
  Temperature  max|rel| = 4.797e-05    초과 depth 0/90
  n_e          max|rel| = 1.999186     초과 depth 68/90
  n_e 비 PRRR/RVTJ :  min 0.9190   median 1.0032   max 2.9992
  depth1-5  PRRR = 46052 51298 53749 52326 53079
  depth1-5  RVTJ = 25324 29177 32291 31447 32453
```
(Codex manifest 의 `max_relative_error=0.66658` 은 PRRR 값을 분모로 쓴 것.
`1.999186/2.9992 = 0.666578` — 같은 사실의 다른 정규화. 재현 일치.)

★ **R·T 일치는 세대 증거가 아니다.** `run_jnu4.info:7` 이 `FIX_T=T` 이므로 T 는
**원리적으로 전 iteration 불변**이고, R 은 고정 격자다. 세 양 중 **진화하는 것은 n_e
하나뿐이고, 그것이 어긋난다.** 따라서 MISMATCH 판정은 견고하다.

⟹ **원장은 Lumina 와 비교하기 **전에** 자기 안에서 외곽 depth 최대 3배 어긋나 있다.**
`*PRRR` 의 재결합률(L-1bf 가 쓸 것)과 `RVTJ` 의 `n_e`(§6.1 이 공간좌표·밀도로 지정한
것)를 짝지으면 **그 자체가 세대 혼합**이다.

미결(A2-01 로 인계): `*PRRR` 의 `n_e` 가 **(a)** 다른 great iteration 산출인지,
**(b)** 같은 iteration 안에서 rate 계산 시점(갱신 전)의 값인지. 둘의 진단은 다르나
**운영 결론은 동일** — 짝지을 수 없다. `EDDFACTOR`/`JH` 쌍이 어느 쪽에 속하는지도 미측정.

#### 자격 4필드 (Codex 재측정, 발주서 값 복사 아님)

| 필드 | 값 | 근거 |
|---|---|---|
| `CMFGEN_FILE_INTEGRITY` | **PASS** | 크기식 정확 일치 + 전 hash + `FINISH_REC` 확인 (8.2 rc=0) |
| `CMFGEN_SNAPSHOT_REPLAY` | **ELIGIBLE** | 단일 파일 replay 한정. cross-file 물리 자격 아님 |
| `CMFGEN_NONLINEAR_CONVERGENCE` | **FAIL** | `OUTGEN:197,239,288` 마지막 세 최대 population 증가 `7.97e5% · 3.69e5% · 3.52e5%` (기준 1%) |
| `CMFGEN_PHYSICAL_ORACLE` | **INELIGIBLE** | `FIX_T=T` + 비선형 FAIL + **세대 혼합** + O-PHYS 필수 7개 부재 |

#### 운전석이 정정당한 것 (발주서 오류)

`_INFO` 가 record 수·단위·frame·iteration ID 를 선언한다고 발주서에 썼으나 **틀렸다.**
실제 필드는 `ND RECL WORD_SIZE UNIT_SIZE INT_SIZE LIT_END` 여섯 개뿐. Codex 가 없는
선언을 만들어내지 않고 `NOT_DECLARED_BY_INFO` 로 기록한 뒤 record 수를 유도했다:

```
EDDFACTOR           14 + NCF(196185) = 196199 record × 728 B = 142,832,872 B  (실제와 일치)
JH_AT_CURRENT_TIME  6 + 1 + 196185  = 196192 record × 1456 B = 285,655,552 B  (일치)
```
운전석이 두 곱을 독립 검산해 정확 일치 확인.

#### 파급 — A-2 층별 게이트 전부에 선행조건이 하나 생겼다

§6.1 은 "CMFGEN depth 를 `RVTJ` 의 속도로 매핑한다"고 규정한다. 그 `RVTJ` 가 rate
파일들과 다른 세대이므로, **어느 세대를 공간좌표 정본으로 삼을지 정하지 않으면
L-1bf 이하 전 게이트의 분모가 정의되지 않는다.**

---

### ★★0-P CONFIG-PREC **폐합** (2026-08-05 01:02)

계약 저작·구현 Codex(`docs/CODEX_L0_NFP_CONFIG_PREC.md` 618행 + src 3파일 +
`scripts/run_config_prec_negative_controls.py`), 검수·실행 운전석
(`~/.lumina_scratch/run_config_prec.sh`).

```
8.1 음성대조 4종  rc=0   passed=4/4
8.2 gate OFF      rc=0   [CONFIG-PREC][WARN] 후 계속
8.3 gate ON       rc=1   [CONFIG-PREC][FATAL] boundary-temperature declarations disagree
8.4 CUDA 전체빌드 rc=0   ★단 syntax 에서. grammar-debug 에는 nvcc 가 없다(Error 127)
8.5 회귀          rc=0   D 19/19 · K 7/7 · Z-INERT 전항 PASS
8.6 diff/금지     rc=0   정본 덱 변경 없음
판정  CONFIG_PREC_ACCEPTANCE = PASS (7/7)
```

계약 요지: 우선순위 `argv > env > config.json > compiled default`.
`plasma_state.csv` 는 **override 가 아니라 consistency witness**.
게이트 `LUMINA_CONFIG_PREC`, 기본 OFF(WARN) — 켜면 현 덱이 반드시 FATAL:
`Δ_T = 14172.549003 − 10020.0 = 4152.549003 K > τ_decl ≈ 5.000014 K` (41.4426%).
τ 의 5 K 는 builder 의 10 K 반올림 반폭(운전석 실측 10018.3523 대 선언 10020.0 과 정합).
★설계 요점: `LUMINA_T_INNER_FIX`(env)를 **무결성 검사 뒤에** 대입한다 — env 를
14172.549 로 맞춰도 화석 불일치를 숨길 수 없다.

★운전석 계측 교훈 추가: **`nvcc` 는 grammar-debug 에 없다. CUDA 빌드는 syntax(로그인
노드) 또는 syn.** 규약 "빌드=로그인 노드 가능"이 여기 해당한다.

`4.005038` 은 **UNRESOLVED 유지**. Codex 조사에서 새 사실 2건:
정본·`_sivcaiv`·`_ftos`·`_fullcov`·`_links` 의 config SHA-256 이 전부 동일
(`cf61ab7c880243ff...`)이고 하나는 심링크·나머지는 `copy2` ⟹ **출처가 하나이고
저장소 안에 없다**. StaNdaRT 자체 볼로메트릭 표 19.48d 는 `1.27878e43` 로 역시 아니다
(`data/standart_data1/toy06/lbol_edep_toy06_cmfgen.txt:29`).

---

### ★★★★F3 — **오라클이 CMFGEN 런 두 개이고, 둘째는 원장 밖이다** (2026-08-05 01:05)

fable 소급 감사(1% 예산 단발)가 찾고 운전석이 실측 검증.

```
$ ls -d /gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern   ->  존재, 363 entry
$ grep -c modern docs/OUTSIDE_LOOP_POOL.md            ->  0
$ grep -c modern docs/ORDER_L0_JNU_OWNER_BY_CODEX.md  ->  0
$ grep -c modern docs/CODEX_A2_00_ORACLE_ELIGIBILITY.md -> 0
docs/GATE_B_DUAL_ORACLE_SPEC.md:42   CMFGEN 측 = toy06_19p48d_modern/{RVTJ,*PRRR,GENCOOL}
scripts/oracle_compare_cmfgen.py:334 default=Path(".../toy06_19p48d_modern")
```

**Gate-B 는 `_modern` 에서 rate·냉각을, `jnu4` 에서 J 를 가져온다.** 그런데 `_modern` 은
캠페인 대장에도, A-2 발주서에도, A2-00 자격심사에도 **한 번도 등장하지 않는다.**

두 런은 같은 모델이 아니다 (`run_jnu4.info` 대 `run_modern.info`):

| | jnu4 | `_modern` |
|---|---|---|
| `NUM_ITS` | 4 (record 62 재시작, 순수 LAMBDA) | 40 (LTE fresh) |
| `MAX_LIN/MAX_LAM` | 3.0 / 3.0 | 10.0 / 10.0 |
| 동결 이온 | **9개** (Si VI·S VI·Ca VI·Fe VI 전준위·Fe VII·Ni VI 전준위·Ni VII·Co VI 전준위·Co VII) | **없음** |

⟹ **A2-00 은 오라클의 절반만 심사했다.** `_modern` 은 `FINISH_REC`·완료토큰·세대정합을
검사받은 적이 없다. 다행히 **도구는 이미 있다** —
`python3 scripts/cmfgen_oracle_contract.py write /gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern`.

#### ★재구성 — "세대 혼합"과 "비선형 미수렴"은 한 결함의 두 관측이다

| depth | v [km/s] | jnu4 RVTJ n_e | `_modern` RVTJ n_e | 차 |
|---|---|---|---|---|
| 67 (≈s0) | 4,394 | 4.8478544E+09 | 4.8528721E+09 | **0.10%** |
| 54 (≈s8) | 10,164 | 7.1561685E+08 | 7.3410434E+08 | **2.5%** |
| 52 | ~12,000 | 2.9625396E+08 | 5.4230246E+08 | **−45%** |
| 1 | 35,975 | 2.5324408E+04 | 2.0043885E+05 | **−87% (7.9배)** |

그리고 jnu4 외곽 n_e 는 **비단조**다: d51 `4.47e8` → d52 `2.96e8` → d53 `7.06e8`.
`_modern` 은 같은 구간이 매끈한 단조 증가.
같은 depth 1 의 RVTJ↔PRRR 균열이 jnu4 는 **82%**, `_modern` 은 **0.54%**
(`docs/CODEX_GATEB_C_REVIEW.md:49`).

⟹ 잠정 재구성: *"균열 크기 = 반복당 상태 변화량"*, *"`NONLINEAR_CONVERGENCE=FAIL` 과
`MIXED_GENERATION_PROVEN` 은 한 결함의 두 얼굴"*.

#### ★★운전석 정정 (01:50) — 위 재구성은 **성립하지 않는다**. 런이 셋이고 축이 셋이다

`_modern` 의 fail-closed 거부(rc=15, 미분류 11항목)가 아무도 안 읽은
`PROVENANCE.txt`(13,221 B, 07-29 22:32)를 드러냈다. 전문 판독 결과:

**CMFGEN 런은 둘이 아니라 셋이다.**

| 런 | 정체 | 감사 |
|---|---|---|
| `toy06_19.48d` (**base**) | snapshot route 자체런. **stint-1(FIX_T=T) RVTJ 가 공개 StaNdaRT n_e 를 `v ≤ 10,000 km/s` 에서 ~1% 로 재현** | 미심사 |
| `toy06_19.48d_jnu4` | base 의 restart clone. `NUM_ITS=4`·순수 LAMBDA·동결 9이온·**구 원자데이터** | **A2-00 심사함** |
| `toy06_19p48d_modern` | base + **20이온을 19apr23 로 재지정**(LUMINA 빈티지 정합, 80 심링크)·`NUM_ITS=40`·동결 없음 | **미심사** (rc=15) |

`PROVENANCE.txt` 원문: 형제 clone `{armF, freeze, jnu4, deepdamp, conv, cmfflux}` 은
**같은 `MODEL_SPEC`(md5 `e8f37ade`)과 같은 126 심링크**를 공유하고 VADAT damping /
FIX_T / FIX_ion / IN_ITS 만 다르다. `_modern` 의 "THE ONE CHANGE" 는 그 심링크 중
80개(20이온 × 4파일)를 `5dec96/10apr99/3oct00/18oct00` → `19apr23` 로 바꾼 것이다.

⟹ **`jnu4` 대 `_modern` 의 n_e 차는 세 축이 겹친 것이다:**
```
① 원자데이터 빈티지 20이온 (구 대 19apr23)   ← 물리 차이. 수렴과 무관
② 반복수 4 대 40
③ 동결 이온 9 대 0
```
게다가 3이온은 **초준위 구조 자체가 바뀌었다**: `SV` NS 39→47, `Co2` NS 55→134
(NF 1000→2558), **`CoIII` NS 52→120 (NF 1000→3214)**.

**따라서 "차이 = 반복당 변화량"은 근거 없다.** 위 재구성은 ①을 ②로 오귀속했다.
`_modern` 이 실제로 돌았음도 확인했다(산출 mtime 07-30 09:37, base 와 해시 상이).

**생존하는 것:**
- jnu4 **런 내부** RVTJ↔PRRR 불일치 68/90 — 그대로(단일 런 안이므로 ①③ 무관)
- 안전대 `s0–s8` — 그대로. 원인이 무엇이든 그 구간 일치가 ≤2.5% 인 것은 실측
- ★독립 확증 추가: base 런 README 가 **`v ≤ 10,000 km/s` 에서 공개 StaNdaRT 와 ~1%**
  를 이미 기록 — 내 10,706 km/s 경계와 정합. **같은 경계의 네 번째 독립 관측**

**무너지는 것:** *"외곽에서는 어느 세대도 답이 아니다"* 는 **미판정**으로 되돌린다.

#### ★★★심판을 세워 결판 (02:10) — 공개 StaNdaRT 대 세 런 전량

계측기 `~/.lumina_scratch/three_runs.py`. 진리 =
`data/standart_data1/toy06/phys_toy06_cmfgen.txt` 의 `#TIME: 19.480` 블록
(NVEL=100, `vel_mid / temp / rho / ne / natom`). 값 = 런 `n_e` / 진리 `n_e`.

| v [km/s] | 진리 n_e | base | **jnu4** | `_modern` |
|---:|---|---|---|---|
| 4,000 | 5.5851e+09 | 0.9999 | 0.9980 | 1.0009 |
| 8,000 | 1.1870e+09 | 0.9797 | 0.9685 | 0.9849 |
| 10,000 | 7.6138e+08 | 1.0028 | 0.9802 | 1.0033 |
| 11,000 | 5.3298e+08 | 1.1000 | 0.9141 | 1.0993 |
| 12,000 | 3.6910e+08 | 1.1661 | 0.8345 | 1.1635 |
| 14,000 | 1.8621e+08 | 1.1683 | **0.0481** | 1.1583 |
| 18,000 | 4.6549e+07 | 1.1759 | 0.6851 | 1.1821 |
| 24,000 | 6.3302e+06 | 1.2657 | 1.3099 | 1.2018 |
| 30,000 | 9.0357e+05 | 1.3886 | **0.2899** | 1.3872 |
| 35,000 | 1.8125e+05 | 1.3758 | **0.2516** | 1.3953 |

```
안쪽부터 연속 5% 이내:  base v<=10,485 · jnu4 v<=10,310 · _modern v<=10,485 km/s
```

**세 판정:**

1. **안전대 `s0–s8` 이 공개 진리로 확정됐다.** 세 런 모두 `v ≲ 10,400 km/s` 까지만
   5% 이내. 상호대조가 아니라 **외부 심판에 대한 값**이라 이것이 정본이다.
   (같은 경계의 다섯 번째 독립 관측.)

2. ★**`jnu4` 외곽은 파손이다.** `v=14,000` 에서 진리의 **0.048배(20배 낮음)**,
   30,000에서 0.29, 35,000에서 0.25. 게다가 비단조(18,000에서 0.69 → 24,000에서 1.31).
   **A2-00 이 심사한 런이자 캠페인 J 진리(`EDDFACTOR`)의 출처가 바로 이것이다.**

3. ★**원자데이터 빈티지 축은 n_e 에 거의 영향이 없다.** base 대 `_modern` 이
   전 구간에서 소수 셋째 자리까지 같다(11,000: 1.1000 대 1.0993 · 30,000: 1.3886 대
   1.3872). ⟹ `jnu4` 대 `_modern` 차이는 **사실상 ②반복수 + ③동결이온**이다.

#### ★운전석 자기정정 2 — 01:50 의 정정은 방법은 옳았으나 결론이 과했다

01:50 에 나는 fable 의 *"차이 = 반복당 변화량"* 을 "세 축 오귀속"으로 되돌렸다.
**방법론은 옳다** — 세 축이 실제로 겹쳐 있었고 통제 없이 ②로 귀속할 근거가 없었다.
그러나 base 런을 대조군으로 세워 실측하니 **①의 기여가 n_e 에서 무시할 만하고,
fable 의 귀속이 실질적으로 맞았다.**

⟹ 남길 교훈은 "fable 이 맞았다"가 아니라 **"대조군 없는 귀속은 옳든 그르든 근거가
없다"** 이다. base 런이라는 통제를 세우기 전에는 어느 쪽도 주장할 수 없었다.
`CoIII` NS 52→120 같은 구조 변경이 **이온화·rate 에 무엇을 하는지는 여전히 미측정**
이다 — n_e 에 안 나타난다고 rate 에 안 나타난다는 근거는 없다.

#### ★★★결판 (02:45) — `MIXED_GENERATION_PROVEN` 은 **세대 결함이 아니라 미수렴 계량기**다

`_modern` 을 분류기 확장 후 심사(rc=0)하니 **두 번째 관측점**이 생겼다. 두 런의
`generation_consistency` 를 나란히 놓으면 답이 나온다 — CMFGEN Fortran 을 읽지 않고.

| | `jnu4` (NUM_ITS=4) | `_modern` (NUM_ITS=40) |
|---|---|---|
| `EDDFACTOR ↔ JH` | MATCH | MATCH |
| `OBSFLUX ↔ OBS_FREQ` | MATCH | MATCH |
| `RVTJ ↔ *PRRR` **n_e** | MISMATCH **27/27** | MISMATCH **27/27** |
| `RVTJ ↔ *PRRR` **radius** | MATCH 27/27 | MATCH 27/27 |
| `RVTJ ↔ *PRRR` **temperature** | MATCH 27/27 | MATCH 27/27 |
| `RVTJ ↔ POP*` 완료토큰 | 동일 `18-Jul 21:38:00` | 동일 `30-Jul 09:37:58` |
| **max 상대오차 / 불일치 depth** | **0.6666 / 68 of 90** | **0.0451 / 49 of 90** |

**구조는 완전히 동일하고 크기만 15배 다르다.**

⟹ **진단: 같은 iteration 안의 쓰기 시점 차(b)이지, 서로 다른 great iteration(a)이 아니다.**

근거 셋:
1. **패턴이 두 런에서 불변** — 언제나 `n_e` 만, 언제나 27개 `*PRRR` 전부, 언제나 R·T 는
   일치. 무작위 세대 혼합이면 양상이 달라야 한다. 고정된 write-order 오프셋의 서명이다.
2. **크기가 수렴도를 따라간다** — 반복 4회 67% → 40회 4.5%. 수렴하면 갱신 전/후가
   같아지므로 오프셋이 0으로 간다. 세대 혼합이라면 반복수와 무관해야 한다.
3. **`RVTJ ↔ POP*` 는 두 런 모두 동일 완료토큰** — `RVTJ` 는 POP 계열과 한 묶음이고
   `*PRRR` 만 떨어져 나온다. 이는 계보 사고가 아니라 **설계된 인쇄 시점**이다.

물리적으로: `*PRRR` 은 **rate 계산에 투입한(갱신 전) `n_e`** 를 인쇄하고 `RVTJ` 는
**갱신 후**를 인쇄한다. 수렴한 런에서는 둘이 일치한다.

#### 귀결 — 도구가 재는 것의 이름이 틀렸다

`CMFGEN_NONLINEAR_CONVERGENCE=FAIL` 과 `generation=MIXED_GENERATION_PROVEN` 은
A2-00 이 별개 필드로 적었으나 **독립 정보가 아니다.** 후자는 전자를
write-order 오프셋이라는 창으로 본 것이다. `MIXED_GENERATION_PROVEN` 이 실제로 재는
것은 provenance 결함이 아니라 **미수렴도**다.

⟹ **A2-01 로 넘기는 처분**: 필드명을 실체에 맞게 바꾸고(예:
`WRITE_ORDER_OFFSET_<magnitude>`), 판정을 이진(`MIXED/SAME`)이 아니라 **연속량**으로
기록한다. 그리고 대장이 적었던 *"어느 세대를 공간좌표 정본으로 삼을까"* 는
**질문 자체가 소멸한다** — 고를 세대가 애초에 둘이 아니었다.

#### 자기정정 3 — 01:50 정정의 최종 처분

01:50 에 나는 fable 의 *"한 결함의 두 얼굴"* 재구성을 "세 축 오귀속"이라며 되돌렸다.
**방법론은 옳았다**(통제 없이 ②로 귀속할 근거가 없었다). 이제 **통제를 갖췄다** —
같은 write-order 구조를 가지면서 수렴도만 다른 두 런. 그 통제 위에서 재보니
**fable 의 귀속이 맞았다.**

남길 교훈은 두 겹이다:
- **대조군 없는 귀속은 옳든 그르든 근거가 없다.** 되돌린 것은 옳은 절차였다.
- **되돌린 뒤에는 통제를 만들어 결판내야 한다.** `UNDECIDABLE` 로 두고 떠나면
  절차만 지키고 답은 못 얻는다. 여기서는 `_modern` 심사 하나가 통제가 됐다.

#### 남은 것

- **base 와 `_modern` 이 `v ≳ 10,500` 에서 진리보다 계통적으로 10–40% 높다.**
  잡음이 아니라 편향이다. 자체런 인프라 전체의 미해결 사항이며 별건으로 등재한다.
- `EDDFACTOR`(= jnu4) 를 외곽에서 쓴 판정은 **n_e 가 20배 틀린 런 위에서 잰 것**이다.
  내부(s0–s8)는 0.98–1.00 이므로 무해. **외곽 전선은 이 사실 위에서 재검토해야 한다.**

#### ★오염 경계선 — **두 개다**. 운전석이 전량 실측해 확정 (2026-08-05 01:20)

계측기 `~/.lumina_scratch/modern_ne.py` 로 두 런 `RVTJ` 90 depth **전량** 대조
(fable 은 s0·s8·s43 세 점만 봤다). 속도격자는 두 런 동일(max|rel| 확인).

```
jnu4 대 _modern  n_e :  >5% 차 depth = 48/90,  범위 d1..d53, v 10706..35975 km/s
내부에서 외곽으로 가며 첫 5% 초과 = depth 53, v = 10,706 km/s
d54(v=10164) 이하 내부는 ≤2.5%,  d67(≈s0) 0.10%
```

**경계가 두 개인 이유: 비교가 두 종류다.**

| 비교 | 경계 | Lumina 셸 |
|---|---|---|
| **런 내부** jnu4 `RVTJ` ↔ jnu4 `*PRRR` | v ≳ 12,376 km/s (d≤50) | s12+ |
| **런 교차** jnu4 `RVTJ` ↔ `_modern` `RVTJ` | **v ≳ 10,706 km/s (d≤53)** | **s10+ (s9 걸침)** |

Gate-B 가 두 런을 가로질러 짝짓기 때문에 **작동 경계는 엄격한 쪽**이다.
덱 `geometry.csv` 실측 매핑(50셸):

```
s8   v 9724 - 10452 km/s   안전
s9   v 10452 - 11180       ★경계 10,706 이 셸 안을 지난다 — 걸침
s10  v 11180 - 11908       오염
```

⟹ **안전 = s0–s8. s9 = 걸침. s10+ = 오염.**
(fable 의 "s0–s10 안전"은 런 내부 경계만 본 것으로, 교차 비교에서는 성립하지 않는다.)

★ 대장 M1 의 `s8 Fe 4.249× / Co 17.255×` 는 **안전대 안**이다 — 그리고 fable 감사가
그 값이 자체 CMFGEN 런 파일을 아예 쓰지 않음(공개 StaNdaRT `ionfrac` 분자 / Lumina
자체 population 분모)을 별도로 증명했다. **두 축 모두에서 생존.**

같은 경계를 이미 세 계측기가 각자 잡았다:
`validation/cmfgen_toy06_19p48d/analysis/rates_certification/run_log.txt:25-27`
(*"33/90 depths deviate >5%: d1-d50 (v=12376-35975)"* + **`gated shells inside the
inconsistent band: NONE`**),
`validation/gate_b_dual_oracle/parity59/cmfgen_snapshot_consistency.csv`,
메모리 `reference_cmfgen_published_toy06_19p48d.md:32`(07-18).
같은 경계를 이미 세 계측기가 각자 잡았다:
`validation/cmfgen_toy06_19p48d/analysis/rates_certification/run_log.txt:25-27`
(*"33/90 depths deviate >5%: d1-d50 (v=12376-35975) … Gamma_PRRR is not a truth there"*
+ **`gated shells inside the inconsistent band: NONE`**),
`validation/gate_b_dual_oracle/parity59/cmfgen_snapshot_consistency.csv`,
메모리 `reference_cmfgen_published_toy06_19p48d.md:32`(07-18).

#### ★운전석 자기정정 — NE-NAMING 수치가 오염대에 착지했다

08-04 23:30 에 기재한 `Δv_inner +56.41% · ΔT_inner −20.04% · τ_total 3.1배` 는
**jnu4 RVTJ 단독**에서 나온 `<Z>(v)` 로 계산했다(`~/.lumina_scratch/ne_zbar.py:11`).
`τ_es` 는 `:167` 에서 **바깥에서 안으로** 적분하므로 링잉 구간(d≤52)이 지배한다.
실측된 `<Z> min 0.0346` 은 jnu4 외곽 n_e 가 `_modern` 대비 7.9배 낮은 바로 그 구간이다.

⟹ **사슬은 생존, 수치는 폐기.** case A 가 덱의 `v_inner=3900 km/s` 를 정확히 재현한
사실은 그대로이므로 *placeholder 가 광구를 정한다*는 것은 확정이다. 그러나
*얼마나 틀렸는가*는 오염되지 않은 `<Z>` 로 다시 재야 한다.

#### ★★재측정 완료 (08-05 04:05) — 깨끗한 `<Z>` 로. **효과가 오염판보다 크다**

`UNQUANTIFIED_PENDING_CLEAN_ZBAR` 해소. 계측기 `~/.lumina_scratch/zbar_clean.py`.
입력 = **공개 StaNdaRT `phys_toy06_cmfgen.txt` 19.48d 블록의 `ne/natom`** (외부 심판,
자체런 무관). 모델 202 zone 중 197 커버.

```
<Z> 진리 :  min 1.7075   median 2.1211   max 3.5809     ← 어디서도 1 미만이 아니다
(오염판 jnu4 는 min 0.0346 — 그 값 자체가 외곽 링잉의 산물이었다)
```

| case | i_phot | v_inner [km/s] | T_inner [K] | tau_total |
|---|---|---|---|---|
| A 현행 `<Z>=1` | 19 | 3900.0 | 14172.5 | 2.299 |
| B/C 공개진리 `<Z>(v)` | **33** | **6700.0** | **10812.9** | 7.27 / 5.76 |

**Δi_phot +14 · Δv_inner +71.79% · ΔT_inner −23.71% · tau 3.16×.** B=C 동일
(커버리지 밖 정책 무관 — 오염판과 같은 성질).

⟹ 오염판(+56.41%/−20.04%)은 방향이 맞았고 **크기를 과소평가**하고 있었다.
진리 `<Z>` 가 전 구간 ≥1.71 이므로 placeholder 는 **어디서나** n_e 를 낮잡고,
광구는 확정적으로 더 바깥이다. NE-NAMING 계약의 근거 수치는 이것으로 교체한다
(checker 의 판정 근거는 여전히 provenance 부재 — 크기가 아니다).

#### ★계보축 신규 — 소거 단조성이 한 방향만 강제된다

같은 균열을 이미 세 계측기가 잡았고 **전부 스스로 옳게 게이트를 걸었다**
(`rates_certification/certify_rate_machine.py:470-492` 는 오염 depth 를 열거한 뒤
*"gated shells inside the inconsistent band: NONE"* 까지 확인하고 진행;
`scripts/emiss_population_swap_e1.py:249-250` 는 *"never use PRRR to alter the
validated"*; `scripts/oracle_compare_cmfgen.py:515-526` 은 매 실행 CSV 기록).
**계측기는 옳았는데 원장이 그 사실을 승계하지 못했다.**
`n_e 1.92×` 좀비의 정반대다 — 그때는 원장에서 죽은 값이 요약에서 부활했고,
이번엔 아티팩트에서 살아 있는 사실이 원장에 도달하지 못했다.
⟹ **소거의 단조성이 원장→요약 방향만 강제되고 아티팩트→원장 방향은 강제되지 않는다.**

---

### ★★0-D D-BUILD **폐합** (2026-08-04 23:30)

봉인된 미커밋 소스로 D·K 배터리 재통과. 계측기 `~/.lumina_scratch/run_dbuild_gates.sh`,
실행 grammar-debug.

```
D 게이트   cases=18 controls=1 PASS=19 FAIL=0   rc=0
           FATAL 16 (D1-D4,D7a-c,D8-D10,D12-D17) 전부 rc=1
           WARN  2 (D5,D6) rc=0 · canonical control rc=0
K 배터리   PASS=7 MISS=0  (양성 1 + 음성 6)
판정       DBUILD_GATE_REPASS = PASS
```

게이트 결과의 귀속 대상(같은 로그에 봉인):
`HEAD 47bfa20` · 미커밋 `src` diff 행수·sha · `src` 트리 해시 · 개별 파일 해시.

**⟹ 인수인계에 남았던 "새 소스로 D 게이트 19/19 재통과" 잔여가 닫혔다.**

### 0-C6 입력 seed 전수 — 대조 안 된 블록의 처분 (★ Codex)

| 블록 | 처분 |
|---|---|
| `Temperature` | CMFGEN이 소비(`SN_T_OPT=USE_HYDRO`). CMF seed / Lumina `0.9 T_rad` = min 1.833 · median 2.264 · max 5.302. **seed와 수렴을 분리 대조해야 함** |
| `Electron density` | 위 0-C5 |
| `Kappa` | **파일에 있으나 `rd_sn_data.f`에 판독 분기 없음 — CMFGEN이 안 먹는다.** `σ_T n_e/ρ`와 5.75e-9로 일치(생성기 내부 정합성만) |
| `Sigma (dlnV/dlnr−1)` | 입력부터 전부 0이고 `PURE_HUB=T`가 다시 0으로 설정. 무쟁점 |

### 독립 감사 2건의 판정 (Codex·fable, 상호 열람 금지)

| 주장 | 판정 |
|---|---|
| 6원소만(Si·S·Ca·Fe·Co·Ni) | **양쪽 지지** (fable: `NT=1645=ΣNS 1637+닫힘6+n_e+T` 산술 포함 4중 교차) |
| 활성 이온 27개 | **양쪽 지지**. 단 원소마다 닫힘 이온 1개 자동추가(이온화 방정식 33단계), 이 런은 `VADAT FIX`로 Fe VI·Co VI·Ni VI 동결 ⟹ **실제 풀린 full 이온 24개** |
| `[X_ISF]` = `NS,NS,NF` | **양쪽 반증** — 실제 **`NV, NS, NF`**(`cmfgen.f:374` `RD_STORE_3INT`). 이 런이 우연히 `NV=NS` |
| VADAT 원소비는 비정본 | **양쪽 지지**(`OUTGEN` `NUM_SPECIES=28 6`). **`MOD_SUM` 종족 abundance 표도 비정본**(템플릿 잔재 오염, fable) |
| 덱이 10/27 이온에서 준위 초과 | **양쪽 지지**. Fe II 2698/135 대 **2599/131**, S III 380/127 대 **256/79** |
| 잉여 32이온 | **양쪽 지지**. 9원소 26 + 중성 6(Si I·S I·Ca I·Fe I·Co I·Ni I). **9원소는 C·O·Mg·Al·Sc·Ti·V·Cr·Mn**(운전석 목록에 **Al 누락**이었음) |

**★정본 정정**: `atomic_links.txt`는 **정본이 아니다** — CMFGEN이 읽지 않는다.
**`MODEL_SPEC`이 이온을 만든다**(`cmfgen.f:368-446`). 활성 FL/SL 정본도 `MODEL_SPEC`이며
osc/`f_to_s` 파일은 **상한일 뿐**이고 `f_to_s` **파일명 숫자도 비정본**(예 `f_to_s_79`인데
활성 SL 69). R1 전체를 `atomic_links.txt` 기준으로 세웠으나 심링크 108개 전건 일치라
결과만 같았다.

### 격자 범위 문제 (첫 조성 발주가 `COMPOSITION_INVALID`로 정직하게 실패)

CMFGEN `SN_HYDRO_DATA` 셀 경계 **1,000–36,000 km/s** vs Lumina 50셸 **3,900–40,300**.
미피복 내측 1,000–3,900(⁵⁶Ni 코어 상당부분이 **Lumina 격자 밖**), shell 44 부분피복,
**shells 45–49 완전 미피복**. ⟹ 외삽 없이 50셸 충전과 총질량 `0.99393 M⊙` 동시 만족 불가.
**StaNdaRT 원본(100–40,300)을 쓰면 범위 문제가 달라진다** — 미확인.

## 층 1 — 입력축: 고리가 소비하지만 생산하지 않는 것

**감사 완료 (2026-08-03): I1–I9 전부 대조 실시. 판정 = 9/9 잔류.** 기준 런 = capture
188932, 덱은 **`data/tardis_reference_toy06_19p48d_sivcaiv`**(운전석 발주서가 `_sivcaiv`
없는 경로로 잘못 적었고 Codex가 argv·`LUMINA_MODEL_DIR`로 정정).
산출=`docs/CODEX_INPUT_ATOMIC_SUMMARY.md`, `docs/CODEX_INPUT_CONFIG_SUMMARY.md`,
`docs/CODEX_INPUT_ATOMIC_LOCALIZE_SUMMARY.md`.

| # | 항목 | 층 | 상태 | 근거 / 비고 |
|---|---|---|---|---|
| I1 | **충돌강도 Υ** | 1 | **잔류 — 확정 불일치** | **Co IV 표 4,455전이 전부가 Fe III 표의 정확한 부분집합**(최대 절대차 **0**, 4,357개는 레벨명까지 동일, 98개는 이름만 상이). 출처 `COB/IV/19apr23/col_data` 대 `FE/III/19apr23/col_data`. **CMFGEN 런의 Co IV tabulated 전이 = 0개**. 부수: 전 선 census **tabulated 29,840 / van Regemorter 1,742,025(67%) / `OMEGA_SET=0.1` 812,267(31%)** — 표 있는 선은 1.2%뿐 |
| I2 | `A_ul` | 1 | **잔류 — 이온별 세분** | 엄격결합 880,406선 중 **75,075 불일치**(임계 `r>1e-6`, 사전 고정). 상위: **Ni III 28.4% · Ni II 21.9% · Co III 17.4% · Ca IV 11.3% · S III 7.4%**(상위5=86%). UV창(600–3000Å) 안 **46.1%**. `r≥1`은 **76선**뿐. 부호 중립(중앙비 0.9999869, 중앙 log비 −5.7e-6 dex) ⟹ **체계 편향 아닌 꼬리 문제** |
| I2a | └ **Fe IV** | 1 | **확정 불일치 (제거 실패)** | ⚠**분모 함정**: `4,336`은 전체가 아니라 **양쪽에 다 있는 선**이다. **CMFGEN Fe IV 72,223선 중 Lumina에 67,887선이 없다**(→ I17). 남은 2선도 실제 **10배 차이**. 임계 `1e-6/1e-9/1e-12` 전부 2건 |
| I2b | └ **Ni IV** | 1 | 잔류 | **3,658/4,085 (90%)** 불일치. σ는 완전 일치 ⟹ 별개 경로 결함 |
| I2c | └ Co IV | 1 | 잔류 | 1,223/3,557 (34%) |
| I2d | └ **Fe III** | 1 | **★제거 (확정)** | **1,500준위·136,263선 전수 엄격결합**, 값 전부 동일, 임계 `1e-6/1e-9/1e-12` **전부 0**, 결합 실패 **0**. 4개 요건 완전 충족 — 오늘 유일한 정식 제거 |
| I3 | 광이온 단면적 σ(ν) | 1 | **잔류 — 이온별 세분** | 3,953,894점 중 **1,233,529 불일치**(임계 `r>1e-6`). 상위: **Ni II 33.3% · Co III 27.1% · S III 13.1% · S IV 5.8% · S V 5.7%**(상위5=85%). **EUV(450–918Å) 18.3% + FUV(918–1290Å) 7.0% = 25.3%** — 오늘 밤 `u`가 CMFGEN 대비 **5.13×·2.53× 역전**된 두 대역과 일치. `r≥1`은 **189,972점** |
| I3a | └ **Co IV** | 1 | **잔류 — 최악** | **46,827/51,411 (91%)**. I1(Υ 대용)·I2c와 합쳐 **Υ·A_ul·σ 세 축 전부** 불일치 |
| I3b | └ Fe III | 1 | 잔류 | 62,782/687,690 (9%) |
| I3c | └ Fe IV·Ni IV σ | 1 | **확정 불일치 (제거 실패)** | ⚠**"0건"은 "동일"이 아니라 "임계 아래"였다.** 임계별 Fe IV `0 / 1 / 22,904`, Ni IV `0 / 0 / 36,746`(1e-6 / 1e-9 / 1e-12). 게다가 **200/1,000준위만 비교 가능**(800준위는 grid slot 결합 불가), 양쪽 0이라 비율 제외 **≈15만 점** |
| **I17** | **선·준위 커버리지 결손** (신규 부류) | 1 | **확정 불일치** | 값 불일치가 아니라 **아예 없는 것**. Fe IV: CMFGEN 72,223선 중 **67,887선 부재**. σ: Fe IV·Ni IV 모두 **800/1,000준위 미결합**. exit 경로가 값 불일치와 다르다 — 임포트 커버리지 문제 |
| I4 | 슈퍼레벨 분할 | 1 | **잔류** | 공통 **21이온 전부** SL 수 상이. Lumina `min(level,100)` 대 CMFGEN `F_TO_S` |
| I5 | 재결합·DR | 1 | **잔류** | Lumina 생산설정에 Co IV→III DR 잔존, CMFGEN 런은 `[DIE_CoIV]=F,F`. RR 전수 계수 대조는 결판 요건 잔존 |
| I6 | 모델 덱 | 1 | **잔류** | `t_exp`만 Δ=0(1,683,072 s = 19.48 d 양쪽 일치). 공간범위·속도·밀도·조성 배열 상이 |
| I7 | 격자 | 1 | **잔류** | **ν 1,000 대 196,185**, 공간 50셸 대 90 depth, 각도 58 대 105 rays |
| I8 | 경계조건 | 1 | **잔류 — 정의 확립 필요** | Lumina `Bν(T_inner=10020K)`+`INNER_BB_SCALE=1`, `L_inj=3.0948e42`(요청 대비 +0.066%), 외부 입사 `I=0`. CMFGEN `DIF=T`/`IB_METH=DIFFUSION`, `LSTAR=2.60e7 Lsun ≈ 9.9528e40`. **비 31.07**. ⚠**두 L이 같은 반지름·같은 정의인지 미확립** — SN은 γ침착으로 `L(r)`이 바깥으로 증가하므로 정의 차이일 수 있다. **결판 요건=같은 속도좌표에서의 L 대조.** CMFGEN 외부 BC 코드값 `UNRESOLVED` |
| I9 | 수치 상수 | 1 | **잔류** | Lumina ε clamp는 **CMFGEN 대응물 없음**(대응물 부재 자체가 잔류 사유). 외곽 반복 **12 대 4**. damping·임계 계약 상이. `eps_floor=1e-5`/`eps_cap=1.0` 발화 **83.4%**(대장 0l) |
| I10 | γ-deposition | 1 | **제거(잠정, epoch 확인 필요)** | 과거 감사 *"Same γ-deposition enters both codes verbatim"*(`radeq_ledger_audit/VERDICT.md:12`). 1층 정적 사실이라 epoch 강건하나 **코드 무변경 확인 후 정식 승계** |

### ★★★층 1 잣대 감사 1차 (2026-08-04, Codex 검수 — 발주서 L1 반려와 함께 산출)

발주서 `docs/ORDER_L1_YARDSTICK_AUDIT.md` 검수 중 나온 실측. **위 표의 여러 행이
이미 무효이거나 epoch-stale이다. 아래를 읽지 않고 위 표를 인용하지 말 것.**

#### ★가장 큰 것 — **epoch 혼합**: 층 1 수치 대부분이 구 덱의 것이다

층 1 판정은 전부 `_sivcaiv`(커버리지 51.74%)에서 측정됐다. `_ftos`에서는:
```
levels        26,592 → 31,792
Fe IV lines    4,336 → 72,223   (= CMFGEN 원본 72,223)
Ni IV lines    4,199 → 72,898   (= CMFGEN 원본 72,898)
σ addressable 26,592 → 31,792   present 26,087 → 31,237  (R1 gate PASS)
σ SHA 변경
```
**⟹ I2·I2a–I2d·I3·I3a–I3c·I17의 분모가 통째로 바뀌었다. 재측정 없이는 어느 것도
확정도 제거도 불가.** 이 사실 하나가 층 1의 다음 행동을 정한다.

#### 행별 정정

| # | 정정 |
|---|---|
| **I7** | ⚠**"ν 1,000 대 196,185"는 다른 것을 비교한 것.** 운전석 실측: `OUTGEN:91,93` = "continuum will be evaluated at **15662** frequencies" / "Number of frequencies is: **196185**". Lumina의 1,000은 `src/lumina.h:507 NLTE_N_FREQ_BINS` = **σ_bf 프리베이크·NLTE Jν 로그빈**. 역할 대응은 **1,000 ↔ 15,662**(둘 다 continuum 격자), 196,185는 선 삽입 후 총 수송 격자. **해당 행 폐기, 15.7× 로 재기술** |
| **I2** | ⚠**임계 `r>1e-6`이 데이터의 유효숫자보다 엄격하다.** Codex 실측: 연결 osc의 A는 대부분 **5자리**(Si IV 4자리), Lumina `line_list.csv` 2,584,132개 중 2,584,131개가 ≤5자리. 운전석 교차확인: 실값 `96411.0 · 247110.0 · 287.21 · 15443.0` = 5자리. ⟹ 양자화 ~1e-5, 임계 1e-6은 그보다 10× 엄격. **75,075 불일치 판정은 반올림을 세고 있을 수 있다.** 동일원본 변환 무결성은 exact/ULP로, 서로 다른 원본의 물리 비교는 양자화 반영 임계로 **분리** |
| **I17** | **`_ftos`에서 명목상 해소.** R1 gate: Fe IV `72223/72223 PASS`, Ni IV `72898/72898 PASS`, 27이온 rank identity 전부 PASS, σ flag Fe IV·Ni IV 각 `1000/1000`. ⚠단 저장된 `verification.log`가 `ERROR: R4 verifier contract failure: 'NoneType'` 1행 — **stale 실패 로그 처분 필요.** G7은 "조성 수리가 파일을 안 바꿈"만 증명하므로 불충분 |
| **I3c** | "200/1,000준위만 결합 가능"은 **구 epoch**. `_ftos`는 Fe IV·Ni IV 모두 `levels=1000, sigma_flag=1000` |
| **I2b** | "σ 동일, A만 90% 불일치"는 **구 epoch**(Ni IV 전체 4,199선). `_ftos`에서 line·level·σ가 다 바뀌므로 재실측 대상 |
| **I3** | ⚠**"실소비 σ가 bin-averaged"는 운전석 전제였고 미확증.** 구 덱 σ 바이너리 mtime 07-28, 현재 생성기는 08-03 수정(1,092행 미커밋 diff). 바이너리 헤더는 `version/nlevels/nbin/ν범위`만 담고 **평균법·생성 commit이 없다.** Codex 추론: point-sampled 가능성. **bake semantics 확정이 선행** |
| **I4** | 측정 가능(`LUMINA_SUPER_CUTOFF=100`, `K=100: 21581 levels lumped`, `super=min(level_num,K)`가 로드된 `f_to_s`를 덮어씀). 단 **잘 정의된 설계 차이**이므로 "수리 대상"으로 보내면 안 됨 — `ACCEPTED-DESIGN` 처분 필요 |
| **I8** | `31.07` 비를 판정 입력에서 제외. 운전석 stdout `L_inj=3.094761e42`·`r_in=6.5640e14`·침착 포함 `L_total_in=1.088240e43` 대 CMFGEN `LSTAR=2.60e7 Lsun`. **같은 반지름에서 CMFGEN `L(r)`을 뽑는 명령·파일이 아직 없다** |
| **I9** | "ε 대응물 없음"은 바이너리 전체 의미론을 닫지 않고는 확증 불가. CMFGEN `NUM_ITS=4·NUM_LAM=2·ACC_F=1e-4·EPS_TERM=0.1%`는 **적용 대상이 다른 묶음**. `NO-COUNTERPART`를 별도 판정값으로 |
| **I1** | "CMFGEN Co IV tabulated 0개"는 확인(`col_guess.dat` = `0 !Number of transitions`, `0.1 !Value for OMEGA if f=0`). 폴백은 `omega_gen_v3.f:158-203`(`EIN_A<=1e-5`면 `OMEGA_SET`, 아니면 oscillator/A 근사). **단 그 런이 현 소스의 폴백을 썼다는 확증 없음** — 실행 당시 바이너리 SHA 미기록 |
| **I5** | `I5a DR 설정` / `I5b level-resolved RR·Milne`로 분리 필요. 현 Lumina NLTE 행렬은 `dr_lookup()` 후 `R_dr`을 조건 없이 더하고 `LUMINA_FROZENIN_DR=0`은 별도 경로만 끈다 |

#### 방법론 정정 — 판정 어휘가 부족했다

운전석은 `WELL-POSED / ILL-POSED / ARTIFACT` 3값을 제안했으나 실측 항목이 안 들어간다
(I4=잘 정의된 설계차 · I9=대응물 없음 · I17=새 epoch에서 해소 · I5=부분판정).
**4축으로 분리한다**:
```
posedness   : WELL / ILL / UNVERIFIABLE
outcome     : MATCH / DIFFER / NO-COUNTERPART / RESOLVED / PARTIAL
kind        : BUG / DESIGN / DEFINITION / COVERAGE / PROVENANCE / NUMERIC
disposition : REPAIR / ACCEPT / DEFINE / REMEASURE
```

#### 함정 목록 확장 (오늘 세 함정으로는 부족)

운전석은 분모·좌표/단위·표본규칙·임계4종을 걸었으나 **`n_e 1.92×` 사고는 잘못된
oracle/epoch였고 그 넷 어디에도 안 걸린다.** 추가 5종:
① 권위원본·실소비경로·symlink target ② binary/source SHA와 epoch
③ 결합키와 중복 처리 ④ `missing/zero/unsupported` 네 상태 구분
⑤ 영분모·절대/상대 오차 규칙

#### 재현기 부재

`rg --glob '*.py' '880406|75075|3953894|1233529'` → **출력 없음.**
층 1의 핵심 수치를 낸 재현 스크립트가 저장소에 없다. 감사자가 매번 처음부터
재구현해야 하며 이는 **사례 23(계기 세대)과 같은 부류의 계측 부채**다.

### ★층 1 잣대 감사 2차 (2026-08-04, L1-A 검수) — 새 덱은 무조건 낫지 않다

#### ★`_ftos`가 충돌표 7종을 잃었다 — 다만 identity 방향일 수 있다

```
collision manifest 대조 (구 덱 → _ftos)
   (14,3) Si IV · (20,3) Ca IV · (27,1) Co II · (27,3) Co IV
   (28,1) Ni II · (28,2) Ni III · (28,3) Ni IV      OLD=OK → FTOS=SKIP
   (26,5) Fe VI                                      OLD=—  → FTOS=OK  (신규)
```
`_ftos`에 이온은 있으나 **CMFGEN 현 vintage가 "0 tabulated transitions"라 바이너리가
없다.** ⟹ 이것은 **퇴행이 아니라 CMFGEN 쪽으로의 정렬**일 수 있다. I1에서 확인된
"Lumina의 Co IV 표 = Fe III 값 / CMFGEN Co IV tabulated 0개"와 정확히 같은 구조다.
7종이 폴백(van Regemorter / `OMEGA_SET=0.1`)으로 내려가면 CMFGEN과 같은 경로가 된다.

**그러나 물리는 바뀐다.** 처분 필요: `identity 개선` 대 `물리 퇴행`을 이온별로 판정.
**⟹ 신규 항목 I19.**

#### 규모 실측 (재현기 자원 산정용)

```
line_list.csv   구 390,625,345 B / 2,584,132행   _ftos 334,306,916 B / 2,220,953행
                합 724,932,261 B / 4,805,085행
cmfgen_sigma_bf 구 212,762,624 B                 _ftos 254,367,824 B
                합 467,130,448 B  = raw double 58,384,000점 (≈445.5 MiB)
collision bin   구  18,662,868 B                 _ftos  21,568,164 B
선 수 변화      Fe IV 4,336→72,223 · Ni IV 4,199→72,898 · Co IV 4,041→69,425
                Fe III 136,263→136,263 (무변화)
σ 헤더          구 (magic,1,26592,1000) → 현 (magic,1,31792,1000)
이온 수         53 → 59 (구 덱 전용 이온 0, 신규 6: 14,4 · 26,5 · 27,4 · 27,5 · 28,4 · 28,5)
```
awk 1패스 4,805,085행 = wall 3.15 s / max RSS 19 MB (캐시 상태, I/O 하한).
⟹ **σ 전량 적재 금지**(445 MiB raw + 중간배열 1 GiB 초과 가능). memmap/chunk 필수.

#### 계보 충전 가능성 — 엄격 기준 **0/18**

```
deck_atomic_sha   ✅ 구 01492367…/2a0b5f9f…/7135be60…  _ftos 17865d5f…/b20069e5…/e38e80a7…
binary_sha        ✅ withParityAH = bcb1292707d33d…dc44  (mtime 08-02 12:33)
source_snapshot   ❌ 그 바이너리를 만든 소스 스냅샷 미확립 (src/ 3파일이 그보다 나중 수정)
cmfgen_snapshot   ❌ 측정시점 CMFGEN 런 SHA manifest 없음 (find 결과 0건)
```
⟹ **I15(바이너리·실행환경)는 A1의 7열로 해소되지 않는다.** 운전석이 발주서에
"A1이 I15·I16 해소를 겸한다"고 쓴 것은 **왜곡**이다(Codex 지적, 채택).
I16(symlink)은 부분 해소, I15는 **build attestation이 별도로 필요**하다.

부수: 구 덱과 `_ftos`의 companion 파일(`config.json`·`geometry.csv`·`density.csv`·
`abundances.csv`·`deposition_cmfgen.csv`·`electron_densities.csv`·`plasma_state.csv`)은
`cmp` 전부 **SAME** ⟹ I6은 companion 기준 `CURRENT`.

#### PHOT evaluator 미지원 범위

`_ftos` σ present 31,237/31,792이나 **CMFGEN type 2/3/8의 2,084레벨 평가기가 없다.**
I3 계열 비교의 분모에서 이를 분리해야 한다(`unsupported` 상태).

### ★★I17 **제거** — `_ftos`에서 실제 검증기로 확인 (2026-08-04)

운전석이 grammar-debug에서 직접 실행(Codex 결과와 독립 일치):
```
$ python3 scripts/verify_deck_r1_vintage.py \
    --new data/tardis_reference_toy06_19p48d_sivcaiv_ftos \
    --cmf-run /gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4

GATE 1/2  Fe IV 72223/72223 PASS · Ni IV 72898/72898 PASS
          27이온 rank identity 전부 PASS · mapping nonidentity ions = 0 PASS
GATE 3    same-vintage ions=7, mismatches=0 PASS
          (Fe II 531,662선 · Fe III 136,263선 f/A/lambda 비트 동일 포함)
GATE 4    deck levels 26592→31792 · sigma addressable 26592→31792
          sigma present 26087→31237 · Upsilon addressable 26592→31792
VERDICT: all four R1 vintage gates PASS      종료코드 0
```
**층 1 최대 항목(Fe IV 67,887선 결손)이 닫혔다.** 커버리지 결손은 `_ftos`에 없다.

| 축 | 값 |
|---|---|
| posedness | WELL |
| outcome | **RESOLVED** (구 epoch `_sivcaiv`에서는 DIFFER, 현 epoch `_ftos`에서 MATCH) |
| kind | COVERAGE |
| disposition | **CLOSE** |
| evidence_status | VALID (검증기 실행, 종료 0) |

⚠ **잔여 2건**: ① `_ftos/verification.log`의 stale 실패 1행 — 원인 특정됨:
`module_from_spec()` 후 `sys.modules` 미등록 → `dataclasses.py:757`에서
`AttributeError: 'NoneType' object has no attribute '__dict__'`. **현
`verify_deck_r4_ftos.py:28-32`에 `sys.modules[name]=module` 수정이 이미 들어 있으나
파일이 untracked라 실패 당시 소스와의 계보는 추론.** R4 전체 재실행으로 갱신 필요.
② `scripts/gate_ftos.py`는 **존재하지 않는다**(운전석 발주서 오기). `gate_ftos`는
`verify_deck_r4_ftos.py:82`의 내부 함수이며 CLI는 `--off-control` 필수.

### ★I19 충돌표 상실 — census 확정 (판정은 미결)

```
7이온 (14,3 Si IV · 20,3 Ca IV · 27,1 Co II · 27,3 Co IV · 28,1 Ni II · 28,2 Ni III · 28,3 Ni IV)
  구 덱  mapped 합 11,329   (378 · 3 · 105 · 4455 · 2775 · 2485 · 1128)  전부 19apr23 col_data, OK
  _ftos  mapped 합      0   (5dec96 / 10apr99 / 18oct00 col_guess.dat)   전부 SKIP
전체 순변화: Upsilon mapped rows 114,952 → 106,091  (순손실 8,861 = 상실 11,329 − 신규 2,468)
```
⚠ **4,455 = I1의 Co IV 표**(= Fe III 값을 그대로 쓰던 그것). 그것이 사라진 것은
**identity 개선**이다. 나머지 6이온의 처분은 미결.

**census만으로는 개선/퇴행을 판정할 수 없다**(Codex 지적, 채택). 필요한 두 축:
- **identity metric**: CMFGEN 실제 선택 branch(`TABULATED`/`VR`/`OMEGA_SET`)와의 일치율,
  `identity_distance_to_current`, `authoritative_tabulated_retention`
- **physics-change metric**: 공통 transition×온도에서 old→new `Υ_eff(T)`·`q_ij(T)` 변화

⟹ **판정 보류. 계측기(`collision` 엔진)의 두 metric 산출 후 결정.**

### ★열거 완전성 — `UNRESOLVED` (소거법 수렴의 전제)

파일 63개·환경 131개는 닫혔으나 **캡처 바이너리에 정확히 대응하는 소스 스냅샷이 없어**
하드코딩 상수의 전수성이 증명되지 않았다(캡처 bin `withParityAH` sha `bcb12927…`,
mtime 08-02 12:33; 현재 `src/` 3파일은 그보다 나중 수정).

**I1–I9 밖에서 발견된 누락 항목** — 풀에 추가해야 한다:

| # | 누락 항목 |
|---|---|
| I11 | 초기 `n_e`·`W`·`T_rad`·`τ`·transition-probability |
| I12 | 원자 구조 · **macro-atom topology** |
| I13 | 패킷 수 · RNG |
| I14 | 연산자 게이트 |
| I15 | 바이너리 · 실행환경 |
| I16 | symlink 해석 |

## 정적 코드축 — 다중주인(같은 물리량, 여러 주인)

| # | 항목 | 층 | 상태 | 근거 / 비고 |
|---|---|---|---|---|
| S1 | **β 다중주인** | 1 | **제거** | `radeq_beta_esc`의 `τ≤1e-6→1.0` 절단 대 페이로드의 정확 `-expm1(-τ)/τ`. 위반 5,826,525행(15.5%), **최대 상대오차 5.0e-07** ⟹ 물리적 무시 가능. 대장 0l |
| S2 | ε 다중주인(`ε′` vs `ε₀`) | 1 | **제거(정의 확정)** | 유도로 판정: `ε′=C/(C+Aβ)`가 Sobolev 사다리 합산의 닫힌 형태로 **옳음**. rung1이 `ε₀`를 쓴 것이 틀린 짝이었고 v4에서 4열 전부 기록으로 수리. `docs/AUDIT_STAGE32_RUNG1_EPSILON_DISCREPANCY.md` |
| S3 | Λ 대각(Sobolev `1−β` vs `cs->lambda_star`) | 1 | **제거(정의 확정)** | 반복 연산자는 `S=S_fixed+(χ_es/χ_tot)Λ_formal[S]`이므로 `lambda_star`가 옳은 짝. v4에서 1차 측정량 교체 |
| S4 | EPAY 처분(branch-site vs writer 재구성) | 1 | **제거(수리 완료)** | writer가 `acc_w>0` 누락 → v4에서 branch-site 실측 + evidence bit. 대장 0l |
| S5 | `chi_es`의 `n_e` 신선도 | 1 | **잔류(미결)** | `opacity->electron_density` 갱신 지점 **1곳**(`plasma.c:6388`) 대 `plasma->n_electron` 갱신 **6곳**(`:2545,6327,6362,10757,13124,14009`). "전혀 복사 안 함"은 반증됐으나 **모든 갱신 경로가 동기화되는가**는 미확인. CUDA 업로드=`cuda.cu:2092` |
| S6 | 다중주인 census의 나머지 `DIFFERENT`/`CONDITIONAL` 행 | 1 | 잔류 | `docs/CODEX_MULTIOWNER_CENSUS_SUMMARY.md` 정독 후 이 표로 전개 |

## 계보축 — 버전을 가로지르는 축

| # | 항목 | 층 | 상태 | 근거 / 비고 |
|---|---|---|---|---|
| L1 | epoch 시계열(회귀 대장) | 1 | **구축 중** | `scripts/regression_ledger.py` + `validation/regression_ledger/ledger.jsonl`(append-only). 백필 대상 69런. 자기검사 통과(음성대조·이중가중·append-only·결손행 유지) |
| L2 | 07-15 → parity59 기근 역전 지점 | 1 | 잔류 | L1 백필 완료 시 **조회로 전환**. 런 이름이 게이트를 담고 있어 게이트 축 × 지표 축 교차 가능 |

## 2층 — 사실은 완전검증, 원인은 고리 얽힘

| # | 항목 | 상태 | 근거 |
|---|---|---|---|
| M1 | 셸별·원소별 이온화 대 CMFGEN 진리 | 잔류 | 진리=`data/standart_data1/toy06/ionfrac_*.txt`. **s8 Fe 4.25× / Co 17.26×**(대장 0g) |
| M2 | ε 클램프 발화율 | **사실 확정** | 83.4%(31,353,733/37,586,850). 의미는 소스 함수 의존 |
| M3 | s0 T_e 대 진리 | **사실 확정** | 21,228 K, 진리 대비 **+2,468 K**. `root-found` 600/600 (대장 0j) |

## 3층 — 고리 얽힘 (1·2층 이후)

| # | 항목 | 상태 |
|---|---|---|
| D1 | T_e 초과의 기전(폴백 양성 되먹임 가설) | 잔류 — 잠정 |
| D2 | 레버 가산성 시험 | **제거(측정 완료) — 가산성 기각** |
| D3 | ALI/MALI가 UV를 고치는가 (rung 2~5) | 잔류 |

---

## ★레버 가산성 판정 (D2, 08-03 — 방법론 확정)

baseline 게이트 PASS(`18,277 K`와 레버 `+3,497/+1,660/+483` 재현). parity59 s0:
committed 21,227.6 / own `cs.J` root 22,801.4 / CMFGEN-J root 18,385.8 K,
순차 레버 `+1,573.8 / −4,415.6 / +374.2`.

- **독립 단독합 − 전체 = +4,415.6 K**
- **6개 순서의 R/J 귀속 폭 = 4,415.6 K** — 설명 대상 불일치(2,468 K)**보다 크다**
- 누적 최종−전체는 모든 순서에서 0 K이나 **망원경 폐합일 뿐 가산성 증거 아님**(Codex 자인)

⟹ **가산성 기각. 실체 = `R(root 재해결) × J(장 교체)`의 비선형 고정점 결합.**
**고리 안 단일 원인 귀속은 이 계에서 정의되지 않는다** — 순서가 답을 만든다.
user 진단(*"자기되먹임이 진짜 용의자를 숨긴다"*)이 수치로 측정됐고, 은폐 규모가
설명 대상보다 크다. 1층 우선 원칙의 근거가 하나 더 생겼다: 3층은 값이 순서 의존이라
확정 자체가 불가능하고, 1층은 파일·코드 대조라 순서가 없다.
산출=`docs/CODEX_LEVER_PARITY59.md`, `validation/chain_replay_parity59/radeq_ledger_audit/`.

## 잔량 (2026-08-03)

- **입력축**: **9 잔류**(I1–I9 전수 대조 완료, 9/9 불일치) / 1 제거(잠정 I10)
  / **제거 후보 4**(I2a Fe IV·I2d Fe III·I3c Fe IV·Ni IV) / **누락 6 신규**(I11–I16)
- **정적 코드축**: 2 잔류(S5 n_e 신선도·S6 census 잔여) / 4 제거
- **계보축**: L1 **구축 완료**(대장 69행) / L2 **부분 해소** — s0 T_e 부호 역전 지점
  **`kpr` 런(07-19)에서 −4,236 K → +4,164 K**로 특정, 이후 미복귀. 단
  `u`·FUV·EUV 축은 하니스 **EDDFACTOR 판독 결함**으로 미특정
- **2층**: 1 잔류 / 2 사실확정
- **3층**: 2 잔류 / **1 제거**(D2 가산성)

**열거 완전성이 `UNRESOLVED`이므로 현 잔량은 하한이다.**

## 다음 표적 후보

1. **I11–I16 열거 완결** — 소거 수렴의 전제. 풀이 안 닫히면 다 지워도 답이 안 남는다
2. **하니스 EDDFACTOR 판독 수리** — 계보축 `u`·FUV·EUV 지점 특정이 막혀 있음.
   작동 참조 구현 = `validation/chain_replay_parity59/common.py`
3. **I8 정의 확립** — 같은 속도좌표 L 대조로 31.07배가 실물인지 정의차인지 결판
4. **제거 후보 4건 정식 제거** — Fe IV·Fe III의 무결을 근거와 함께 확정

---

## ★★★현 시점 인수인계 (2026-08-04, 컴팩 직전)

### 층 0 계약 19개 상태

```
✅ 폐합 8   C2-EXEC · H-TRANSFORM · GEN-GUARD · K-SHAPE · K-FRESH · Z-INERT
            + D-BUILD (08-04 23:30, D 19/19 · K 7/7, 봉인 소스 귀속)
            + TE-DEAD  (08-04 23:30, 읽는 곳 0 확인, 등재로 종료)
◐  실측완료·계약미작성 3
            NE-NAMING   <Z>=1 -> 광구·v_inner·T_inner·전셸격자.  Δv_inner +56.41%
            DECK-FOSSIL config.json 이 자기 생성기로 재현 불가 (4.005038배, 가설4 전부 기각)
            CONFIG-PREC 한 덱이 T_inner 를 2개 선언, 둘 다 런타임에서 살아있음 (41.4% 차)
⬜ 잔여 1   A-2(J_ν 단일 소유권) — 18단계. **A2-00 PASS(1/18)**, A2-01~18 미착수
            TRAD-FIX·T-SEED 를 흡수. 「3중 모순」은 위 실측으로 2중으로 축소·출처 특정
            ★A2-00 이 원장 세대 혼합(RVTJ 대 *PRRR n_e, 외곽 3배)을 증명 —
              A2-01 이전에 공간좌표 정본 세대를 정해야 L-1bf 이하 분모가 정의된다
범위판정 4  F(층1) · G(층1) · I(토폴로지=층1, 0원소=Z-INERT 로 폐합) · J(정밀화 강등)
배포 2      커밋·푸시(user 조건부 승인: "0층이 완료되면") · 생산 적용
```

### 다음 행동 (순서)

1. **A2-00 수령·검수** — Codex 산출 `docs/CODEX_IMPL_A2_00.md`. 운전석이 grammar-debug 에서
   실행하고 §11 회귀 대장 A2-00 행에 서명. 발주 프롬프트
   `<scratch>/dispatch_a2_00.txt` (음성대조 6종 중 4번 = mtime 변경 시 **PASS** 요구가 핵심)
2. **NE-NAMING·DECK-FOSSIL·CONFIG-PREC 계약 발주** — 실측은 끝났고 계약 문안이 없다.
   ★ A-2 의 L-0 음성 대조(`W B_ν(T_rad)` 주입)가 이 셋에 의존하므로 **A2-04 이전에** 닫아야 한다
3. `L0_VALIDATION_CLOSED` 판정 → 커밋·푸시

### ⚠실행 자원 제약 (2026-08-04 현재)

**lageunha 사용 불가** — user 작업 점유(`ramses_final3d` 4개 × 98% + `cf4_lg_pea` 161%,
load 32.65). 규약상 즉시 양보. A-2 의 해상도 사다리(1000/2000/4000/8000/16000빈)는
대형 CPU 작업이므로 **grammar slurm 으로 우회하거나 대기**.
grammar 제출 시 `--exclude=grammar072,grammar078,grammar080` 필수.
**`/usr/bin/time` 은 grammar-debug 에 없다.**

### 운전석 계측 자산 (재사용)

```
scripts/kshape_harness.c              K-SHAPE/K-FRESH 전용 하니스(운전석 작성)
scripts/verify_h_transform.py         H 정량
scripts/verify_trad_fix.py            TRAD-FIX 4상태 + CMFGEN 대조 (--final-contract 필수)
scripts/verify_zinert.py              Z-INERT
scripts/kshape_contract.py            {write,check} <deck>
~/.lumina_scratch/*.sh                운전석 러너 (detach+폴링 패턴; ssh 가 장시간 끊김)
~/.lumina_scratch/wnu_range.py        W(ν) 동적범위 측정
```

### 이 세션에서 확정된 상설 규약

- **분담 개정 4**: 발주서 저작=Codex / 검수·실행·원장=운전석 (memory 상설 기재)
- **운전석 3규율**: 개수는 세는 명령과 함께 · 파일명은 그 턴에 확인 · **측정 전 결론 금지**
- 검증 폐합과 생산 배포를 **두 필드로 분리 기재**(bakefix5 전례)

---

### ★★0-N NE-NAMING **폐합** (2026-08-05 11:38, 처분 A)

계약 저작·구현 Codex(`docs/CODEX_L0_NE_DECK_CHECKERS.md` §1), 검수·실행 운전석.
분할 실행 첫 적용: [A]=grammar-debug / [B]=lageunha (user 08-05 "2개 노드 동시 활용").

```
음성대조 5종      NE_NAMING_CONTROL_SUMMARY passed=5 total=5   rc=0
legacy 양성       [NE-NAMING][WARN] mode=PLACEHOLDER_ZBAR_ONE claim=legacy-read-only
                  disposition=A read_only=yes                   rc=0
production 차단   [NE-NAMING][FATAL] unapproved placeholder     rc=1
회귀              D 19/19 · K 7/7 · Z-INERT · CONFIG-PREC(8.4 포함) · 분류기   전부 PASS
덱 불변           git status -- data/ 공백 · 4 companion SHA/size/mtime 동일
```

- **처분 A 이행**: `build_toy06_epoch.py` 에 `authorize_ne_boundary()`(`:251`) — `i_phot`
  (`:260`) **직전** fail-closed. GEN-GUARD 불변식 유지(guard 가 여전히 첫 연산).
- placeholder 는 `PLACEHOLDER_ZBAR_ONE` mode + 승인 토큰 없이는 production 경계를 만들
  수 없다. legacy 봉인 = `docs/manifests/ne_naming_toy06_19p48d_legacy.json`.
- 처분 B(참값 경로)는 **명세만**(`true_path_specification()`, `SPECIFICATION_ONLY`) —
  `4.005038` 미해결 동안 계약이 이식을 금지(§3.2 금지 4).
- 근거 수치는 오염판 +56.41% 가 아니라 **깨끗한 재측정 +71.79% / −23.71%**
  (04:05절, 공개 StaNdaRT `<Z>` ≥1.71 전 구간). checker 판정 근거는 크기가 아니라
  **provenance 부재**이므로 수치 교체가 판정을 바꾸지 않는다.

### ★★0-F DECK-FOSSIL **폐합** (2026-08-05 11:38, fossil quarantine 경로)

계약 저작·구현 Codex(같은 문서 §2), 검수·실행 운전석.

```
음성대조 5종      DECK_FOSSIL_CONTROL_SUMMARY passed=5 total=5   rc=0
fossil 양성       [DECK-FOSSIL][WARN] producer=UNRESOLVED mode=legacy-read-only
                  epsilon_L=3.005038 Delta_SB=1.65K canonical_production_eligible=no   rc=0
canonical 주장    [DECK-FOSSIL][FATAL] missing manifest          rc=1
```

- **quarantine 레코드** = `docs/manifests/deck_fossil_toy06_19p48d_quarantine.json`:
  4 companion exact SHA-256 · `producer=UNRESOLVED` · 기각 가설 4(REJECTED) ·
  `R_L=4.005038 / epsilon_L=3.005038 / Delta_SB=1.65 K` · "내부 정합은 generation
  재현성을 대신하지 못한다". 수치는 증거이지 writer 상수가 아니다.
- **단계적 이행**: 현 덱은 legacy-read-only 로 계속 쓰되 매 load WARN. 무조건
  canonical 주장은 FATAL. **현 fossil 은 영구히 canonical production seed 자격 없음.**
- atomic writer(`scripts/deck_generation_atomic.py`) = 임시 디렉터리 → 검증 → 한
  generation commit. 새 canonical 생성 시 별도 발주·검증 필요(미가동).
- 합격선 사전등록: `epsilon_L ≤ 1e-6` · `Delta_SB ≤ 5 K`. 현 builder 를 producer 로
  가정하면 `epsilon_L=3.005` ⟹ FATAL — 즉 계약이 현 상태를 정확히 거부한다.

⟹ **A2-04 HARD BLOCK 3계약(NE·DECK·CONFIG) 전부 폐합.** A-2 는 A2-01 부터 자유 진행.

★lageunha 정정(user): **nvcc 있음** — `module load cuda` 로 13.0.2. GPU=RTX 5000 Ada
32GB(sm_89, sm_80/86 cubin 상위호환 실행 가능). 빌드·소형 GPU 검증 가능, full-NLTE
생산런(80GB)은 여전히 h100/h200. 8.4 CUDA 빌드 lageunha 실측 rc=0.

---

## ★인수인계 (2026-08-05 03:20, user 로그아웃)

### 지금 도는 것

**Codex: NE-NAMING + DECK-FOSSIL checker 구현** (`pid=1658332`, 03:15 발주).
발주 프롬프트 = `<scratch>/dispatch_ne_deck.txt`, 산출 예정 =
`docs/CODEX_IMPL_NE_DECK_CHECKERS.md`.

처분은 계약이 강제한 것이며 재논의 대상 아님:
- **NE = 처분 A** (§3.2 금지 4 가 `4.005038` 미해결 상태에서 B 를 금지)
- **DECK = fossil quarantine** (§4.6 이 OR; producer 미발견 ⟹ 후자.
  단 §4.4 의 legacy read-only mode = **WARN rc=0** 이므로 덱은 계속 쓴다)

★발주서에 박은 것: **`Δv_inner +56.41%` 를 근거로 인용 금지.** 그 값은 `jnu4` 단독
`RVTJ` 산출인데 그 런 외곽이 공개 진리 대비 `v=14,000` 에서 0.048배로 파손이다.
계약 §3.3 근거란은 `UNQUANTIFIED_PENDING_CLEAN_ZBAR` 로 둔다.
**사슬의 존재는 확정**(case A `v_inner=3900.00` 이 덱과 정확 일치)이므로 checker 는
크기가 아니라 **provenance 부재**로 판정한다.

### 운전석 인수 절차 (Codex 복귀 시)

1. 보고서 §별 검수 — 두 checker 가 **독립 실행**되는가, 음성대조 marker/rc 가 계약
   §3.4·§4.4 표와 **정확히** 일치하는가, `+56.41%` 를 인용하지 않았는가,
   `git status --short -- data/` 가 비었는가(덱 바이트·mtime 불변).
2. grammar-debug 실행 — checker 2종 + 회귀 4종
   (`run_dbuild_gates.sh` · `run_zinert_selftest.sh` · `run_config_prec.sh` ·
   `run_cls_verify.sh`).
3. **계약별로 따로 커밋** (user 08-05 규약 [[feedback-one-contract-one-commit]]).
   좁게 스테이징할 것 — `git add -A <디렉터리>` 는 지난번 입도 실패의 원인이었다.

### 그 다음: A-2 17단계

```
A2-01 소유권 census (157행 disposition)   ← 선행 없음. 지금 착수 가능
A2-02 좌표·격자
A2-03 RadiationField shadow
──── A2-04 HARD BLOCK: NE · DECK · CONFIG 3계약 폐합 ────
A2-04 생산자 commit → A2-05~11 CPU → A2-12~15 GPU → A2-16~18
```
CONFIG-PREC 은 08-05 01:02 폐합. **지금 도는 둘이 A2-04 의 마지막 자물쇠다.**

★A2-01 착수 시 반영할 것: A2-00 이 남긴 `MIXED_GENERATION_PROVEN` 필드는 실체가
**write-order 오프셋으로 본 미수렴도**임이 밝혀졌다(02:45절). 필드명을 실체에 맞추고
이진이 아닌 **연속량**으로 기록하는 것이 A2-01 처분이다.

### ⚠환경 제약 (실측)

- **`nvcc` 는 grammar-debug 에 없다**(Error 127). CUDA 빌드는 **syntax(로그인 노드)**
  또는 syn. 규약 "빌드=로그인 노드 가능" 적용.
- `lageunha` 는 user 작업 점유 — 확인 후 사용.
- `/usr/bin/time` 은 실행 노드에 없다.
- fable 은 **비용 때문에 사용 중단**(user 08-05). 세대 진단류는 Codex 또는 운전석이 한다.

### 계측 자산 (재사용)

```
~/.lumina_scratch/run_dbuild_gates.sh    D 19/19 + K 7/7
~/.lumina_scratch/run_config_prec.sh     CONFIG-PREC 8.1-8.6
~/.lumina_scratch/run_cls_verify.sh      분류기 jnu4/modern/음성
~/.lumina_scratch/run_a2_00.sh           A2-00 원장 자격
~/.lumina_scratch/three_runs.py          ★공개 StaNdaRT 대 자체런 3개 (안전대 잣대)
~/.lumina_scratch/modern_ne.py           두 런 RVTJ 90 depth 전량
~/.lumina_scratch/ne_zbar.py             <Z> 사슬 (단 jnu4 오염 — 입력 교체 필요)
~/.lumina_scratch/deck_fossil.py         덱 대 생성기 정합
scripts/cmfgen_oracle_contract.py        원장 manifest {write,check}
```

### git

```
fa6f283  분류기 확장 + MIXED_GENERATION 규명
a97d0e1  층 0 계약 10건 폐합 + 안전대 확정
브랜치 thenmc-macroatom-fluorescence · push 완료 · 미커밋 추적변경 0
```
