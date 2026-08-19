# DET-STAGE12 판정런 (L4) — 2026-08-20 [감리 반영 확정본]

감리(Fable 독립 컨텍스트) 결론 **수정**. 지적 R1–R5 를 전부 반영했고,
수치·소스 주장은 운전석이 독립 재확인 후 채택했다. 파생 지적 R6 은
`docs/RUNG_DET_PHYSCMP_2026-08-20.md` 에 반영했다.

## 0. 봉인
- job **320193**, syn101, a100×2, elapsed **01:51:26**, State **FAILED 70:0**
- run root: `/gpfs/kjhan/lumina/det_stage12_fixed_te_a100x2_k36/l4_20260819T121216Z_pin_seed`
- binary SHA `48017efcf2a64de56217351c5630d082bcc2ab69ff5f6c2956dc3fc882695019`
- stderr SHA `f32aaffa90c4a3d4a49fd3e5f1f00fb807faf0e68f8aa730e121ef04a7f71e5d`
- 핀 프로파일: `/gpfs/kjhan/lumina/te_profiles/seed_uniform_10020.txt` (전 셸 10020.0 K 균일)
- 신호 발췌: `validation/det_stage12/L4_320193_signals.txt`

## 1. 게이트

| 게이트 | 결과 | 실측 |
|---|---|---|
| **L3** 공시 | **부분 PASS — stderr 절반만** | `stderr.log:34-35` **2줄 8토큰**: `te_lane=FIXED_T te_profile_sha256=25b03b07… te_source=… pinned_shells=50 re_root_required=0` + `te_min_K=10020 te_max_K=10020 publication_authority=NONE`. ⚠**manifest 절반은 미실행** — `work/` 하위 정규파일 **0개**. manifest 작성 코드가 도달조차 못 했다(감리 R2) |
| **L4** 물질 갱신·반복 1 도달 | **부분 PASS — 전반부만** | `[R7][PHASE] event=R7_MATERIAL_PHASE_COMMITTED … **te_generation=1->2**`. ⚠사전등록 문언의 후반부 **"= 반복 1 에 도달"은 미충족** — 로그에 `iter=1` 이 **0건**이다(감리 R1). L6 행과의 자기모순을 해소한다 |
| **L5** 핀 온도 잔차 | **PASS(계량)** | 셸 0–49 전수 기록. shell 0: heating 2.2906e-03 / cooling 3.4671 / residual **−3.4649**. shell 49: 3.107e-11 / 1.6248e-05 / **−1.6248e-05** |
| **L6** population 의 LTE 이탈 | **미판정** | 커밋 직후 런이 죽어 반복 1 population 산출 없음 |
| **L2** 자유레인 byte 불변 | **미실행** | 별도 자유레인 런 필요 |

## 2. ★L4 의 의미 — 결정론 팔이 처음으로 T_e 세대를 커밋했다

`te_generation=1->2`. 캠페인 내내 결정론 팔은 `RADEQ_NO_BRACKET` →
`R7_MATERIAL_UPDATE_BLOCKED` 로 **세대 1 에 갇혀 있었다**. 고정-T 레인이 그 벽을 열었다.
이 단이 열겠다고 사전등록한 바로 그것이다.

## 3. 교차 정합 — **상류 무섭동**의 크로스-빌드 확증 (제목 정정, 감리 R3)

1. **잔차 17자리 일치**: 고정-T 레인의 shell 0 잔차 `−3.4648553111533551` 이
   IV 자유레인 런의 `PUBLIC_SEED` 스캔 `res_mid(T=10020)` 와 **완전히 같다**.
   ★**감리 R3 정정**: 이것은 "자유레인 무접촉의 런타임 확증" 이 **아니다** — 이 런에서 `FREE_T`
   가지는 **한 줄도 실행되지 않았다**(고정레인 런이다). 레인 분기는 A2-10 **안**에서 일어나므로
   동일한 것은 **분기 이전** 값들이다. 분기 이후에는 대조군이 없다. **L2 를 대체하지 못한다.**
   ★그러나 감리가 더 강한 증거를 찾았다: `[A2-10][LINE-COEFFICIENT-IDENTITY] phase=INTERIOR`
   **50줄 전수가 두 런에서 바이트 동일**이며, 두 런은 binary sha(`3fc2cbbc` vs `48017efc`)·
   git HEAD·진단 노브·스레드 수가 **모두 다르다** ⟹ **결정론 확증 + 이번 단의 편집이 A2-10
   상류를 섭동하지 않았다는 크로스-빌드 증거.** 그것이 이 관측의 정확한 함의다.
2. **핀 해시 일치**: 공시의 `te_profile_sha256=25b03b07…` 가 IV 런의
   `[A2-10][SEED] bootstrap T_e published … manifest=25b03b07ae22` 와 접두 일치.
   ⟹ 핀이 정확히 부트스트랩 seed T_e 임이 해시로 확증됐다(기계 시험이라는 선언과 부합).

## 4. L5 실측 — 계량 (감리 R4 로 **전면 재서술**)

**살아남는 진술은 여기까지다**:
> 핀 온도(= 부트스트랩 seed T_e, 10020 K)에서 **50셸 전수**의 복사평형 잔차가 음수이고,
> cooling/heating 이 **1.5e3 – 5.2e5 배**다(shell 0: 1,514배 · shell 49: 5.2e5배).
> 셸 0–3 은 08-18 IV 런과 **비트 동일**하다.

### ★철회한 것과 그 이유

초안은 "핀 = 시행이라 **상태 불일치가 제거**됐는데도 순수 냉각 ⟹ 08-19 발견을 상태 불일치와
무관하게 확인" 이라 적었다. **철회한다.**

자유레인은 **이미 10020 K 를 `PUBLIC_SEED` 로 샘플했고**(IV 런 `res_mid(T=10020)`),
그 수가 `finding_nobracket` 사슬이 인용한 바로 그 값이다.
⟹ 이것은 **새 조건에서의 확인이 아니라 08-18 수의 재발행**이다.
그리고 §3 의 "17자리 일치" 가 그 사실을 스스로 증명하므로 **두 절이 양립 불가**였다.
"이번 런의 가장 큰 과학적 소득" 도 근거 없음 — 삭제.

### 새로운 것 (한정)

셸 **4–49** 의 잔차가 새롭다(IV 는 `VECTOR-NOBRACKET count=4` 로 셸 0–3 만).
⚠단 그 46셸에는 **다른 온도에서의 잔차가 없다** — 고정레인은 근탐색을 건너뛰므로
`phase=LOWER/UPPER/REQUESTED_TE/GEOMETRIC_MID` 가 **0건**이다.
⟹ **"근이 원리적으로 없다" 에 대한 브래킷 증거는 여전히 셸 0–3 뿐이다.**
50셸 전수를 그 결론의 확인으로 쓰는 것은 초과다.

## 5. 새 차단 — 발화 지점 **미특정** (감리 R5 로 정정)

`[PHYSICS-COMPARISON][FATAL] lane=DET iter=0 status=PHYSICS_COMPARISON_INVALID_ARGUMENT`

★**초안의 "우리 코드가 아니다" 는 철회한다.** 근거로 든 것이 성립하지 않았다.

- [실측] `PHYSICS_COMPARISON_INVALID_ARGUMENT` 는 `src/physics_comparison.c` 의
  **여섯 자리**에서 반환된다: **99 · 112 · 133**(`comparison_validate` 내부) ·
  **255 · 258**(`snapshot_write` 진입) · **448**(`dump_if_requested` 인자 가드).
  로그는 상태명만 찍으므로 **여섯 중 어느 것인지 구별 불가.**
- [실측] 초안이 근거로 든 "`BLOCKED reason=` 줄 부재" 는 **동어반복**이다 —
  그 줄을 찍는 블록(:353-364)은 **`IO_ERROR` 를 반환**한다.
  `INVALID_ARGUMENT` 가 관측된 이상 그 줄의 부재는 **논리적으로 강제**되며 추가 정보가 없다.

**구별해야 할 두 진술**:
| | 진술 | 판정 |
|---|---|---|
| (가) 가드 **코드**가 이번 단 이전 것이다 | dd9f7c1 이 추가한 `INVALID_ARGUMENT` 반환은 **0개**; 호출부 무접촉 | **참** |
| (나) 발화 **원인**이 이번 단과 무관하다 | 후보 99/112/133 은 전부 `temperature_publication` 을 읽고, 그 구조체는 **이번 단이 고정레인에서 새로 채운다**(`te_manifest_sha256` 자리에 프로파일 해시를 넣는 것 포함). 이 경로는 고정레인이 **처음 도달**시켰다 | **미입증** |

⟹ 초안은 (가)를 논증하고 (나)를 결론으로 적었다. **비약이었다.**
**발화 지점은 오프라인으로 특정할 수 없다 — 모른다.** 그것이 DET-PHYSCMP 의 존재 이유다.

## 6. 판정

- **L3 부분 PASS(stderr 절반; manifest 미실행)** · **L4 부분 PASS(전반부; 반복 1 미도달)** ·
  **L5 PASS(계량)** · L6 미판정 · L2 미실행 · L1·NL 은 `dd9f7c1` 에서 별도 확인
  ⟹ **부분 폐합**. "폐합" 으로 적지 않는다.
- 신규 단 후보 **DET-PHYSCMP**: 기존 인자 가드가 **어느 필드에서** 걸리는지 찍고,
  DET 레인의 physics_comparison 계약을 폐합한다. (Γ단·A210-ZERO-OPACITY 와 같은 계급 —
  "가드가 사유를 안 찍는다".)
- 08-20 회수(부적격 오라클)의 영향 **없음** — 이 판정은 전부 Lumina 내부 실측이다.

## 7. 미결
1. L6 — 반복 1 population 의 LTE 이탈 여부(DET-PHYSCMP 폐합 후 재시도)
2. L2 — 자유레인 byte 불변(별도 런)
3. `physics_comparison` 가드의 위반 필드 특정
