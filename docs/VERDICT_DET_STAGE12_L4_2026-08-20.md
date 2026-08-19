# DET-STAGE12 판정런 (L4) — 2026-08-20 [감리 전]

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
| **L3** 공시 | **PASS(stderr)** | `te_lane=FIXED_T te_profile_sha256=25b03b07… te_source=… pinned_shells=50 re_root_required=0` — **6필드 전부**. `[A2-10][TE_PUBLICATION]` 도 동반 |
| **L4** 물질 갱신·반복 1 도달 | **★PASS** | `[R7][PHASE] event=R7_MATERIAL_PHASE_COMMITTED lane=DET iter=0 phase=A2-10 r=2 o=3 e=3 **te_generation=1->2**` |
| **L5** 핀 온도 잔차 | **PASS(계량)** | 셸 0–49 전수 기록. shell 0: heating 2.2906e-03 / cooling 3.4671 / residual **−3.4649**. shell 49: 3.107e-11 / 1.6248e-05 / **−1.6248e-05** |
| **L6** population 의 LTE 이탈 | **미판정** | 커밋 직후 런이 죽어 반복 1 population 산출 없음 |
| **L2** 자유레인 byte 불변 | **미실행** | 별도 자유레인 런 필요 |

## 2. ★L4 의 의미 — 결정론 팔이 처음으로 T_e 세대를 커밋했다

`te_generation=1->2`. 캠페인 내내 결정론 팔은 `RADEQ_NO_BRACKET` →
`R7_MATERIAL_UPDATE_BLOCKED` 로 **세대 1 에 갇혀 있었다**. 고정-T 레인이 그 벽을 열었다.
이 단이 열겠다고 사전등록한 바로 그것이다.

## 3. 교차 정합 — 두 레인이 같은 물리를 계산한다 (독립 확인 2)

1. **잔차 17자리 일치**: 고정-T 레인의 shell 0 잔차 `−3.4648553111533551` 이
   IV 자유레인 런의 `PUBLIC_SEED` 스캔 `res_mid(T=10020)` 와 **완전히 같다**.
   ⟹ 레인 분기가 물리를 바꾸지 않았다(감사가 verbatim 으로 본 "자유레인 무접촉" 의 런타임 확증).
2. **핀 해시 일치**: 공시의 `te_profile_sha256=25b03b07…` 가 IV 런의
   `[A2-10][SEED] bootstrap T_e published … manifest=25b03b07ae22` 와 접두 일치.
   ⟹ 핀이 정확히 부트스트랩 seed T_e 임이 해시로 확증됐다(기계 시험이라는 선언과 부합).

## 4. L5 가 말하는 것 — 08-19 발견의 정량 확인

전 셸에서 **cooling ≫ heating**, 잔차가 전부 음수:
- shell 0: cooling/heating = **1,514배**
- shell 49: **5.2e5배**
⟹ 핀 온도가 시행 상태와 **일치하는데도**(상태 불일치 제거) 여전히 순수 냉각이다.
이는 `FINDING_NOBRACKET_LTE_SEED_2026-08-19.md` 의 사슬 — **LTE 시드에서는 근이 원리적으로
없다** — 를 상태 불일치와 무관하게 확인한다. 이번 런의 가장 큰 과학적 소득이다.

## 5. 새 차단 — 그리고 그것은 **우리 코드가 아니다**

`[PHYSICS-COMPARISON][FATAL] lane=DET iter=0 status=PHYSICS_COMPARISON_INVALID_ARGUMENT`

- `[PHYSICS_COMPARISON][BLOCKED] reason=…` 줄이 **없다** ⟹ 우리가 넣은 공시 검증 경로가 아니다.
- 실체는 `physics_comparison_dump_if_requested` 의 **기존 인자 가드**:
  `!geometry || !atom || !plasma || !opacity || !nlte || geometry->n_shells < 2 ||
  plasma->n_shells != geometry->n_shells`.
- 이 가드는 **DET 레인에서 한 번도 실행된 적이 없다** — 결정론 팔이 여기까지 온 적이 없기 때문이다.

⟹ **L4 가 문을 열자마자 그다음 계약이 걸렸고, 그 계약은 기존 것이다.**
어느 필드가 위반인지는 로그가 말하지 않는다(가드가 사유를 찍지 않는다) — **계측 필요.**

## 6. 판정

- **L3·L4·L5 PASS**, L6 미판정, L2 미실행 ⟹ **부분 폐합**. "폐합" 으로 적지 않는다.
- 신규 단 후보 **DET-PHYSCMP**: 기존 인자 가드가 **어느 필드에서** 걸리는지 찍고,
  DET 레인의 physics_comparison 계약을 폐합한다. (Γ단·A210-ZERO-OPACITY 와 같은 계급 —
  "가드가 사유를 안 찍는다".)
- 08-20 회수(부적격 오라클)의 영향 **없음** — 이 판정은 전부 Lumina 내부 실측이다.

## 7. 미결
1. L6 — 반복 1 population 의 LTE 이탈 여부(DET-PHYSCMP 폐합 후 재시도)
2. L2 — 자유레인 byte 불변(별도 런)
3. `physics_comparison` 가드의 위반 필드 특정
