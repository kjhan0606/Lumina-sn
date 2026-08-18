# 미실행 게이트 전수 정리 — 2026-08-18

user 지시 "미실행 게이트 회수해" / "나머지도 정리해".
08-08 감사는 **세 단**(SH-GAMMA·DET-R6·SH-GRID)만 강등했으나, 전 단 문서를 훑은 결과
**MC-EVT(E단)에도 미실행 게이트가 있고 그 판정은 `BLOCKED`** 였다. 아래가 전수 목록이다.

## A. 오늘 회수 완료 — 5건

| 게이트 | 단 | 증거 |
|---|---|---|
| **B-6** 값 이동 장부 | SH-GRID | `validation/sh_grid_b6/` — T1 이동은 FUV 집중·최대 2.6e-4·생산 T_e 에서 4.5e-5 ⟹ 재빈닝 오차 상한 ≈3e-4. T2(sigma bake 전후)는 dlog·상단 **bit-identical**, 이동 ≤1.3e-15 |
| **NC2** 구세대 epoch 거부 | SH-GAMMA | `GAMMA_PUBLICATION_STALE_EPOCH` |
| **NC4** 이중발행 차단 | SH-GAMMA | rc=4 `GAMMA_DOUBLE_PUBLISH` + 배열 byte 보존 + 양성대조(다른 epoch=정당 재발행) |
| **N6-2** q-set 해시 한 글자 | DET-R6 | `QHASH_MISMATCH` — 근접 오탐도 거부 |
| **N6-3** 창 밖 선 VALID 위장 | DET-R6 | `NEGATIVE_OR_NONFINITE_JBAR cell=1 validity=1 value=-1`, `rejected_at=commit` |

증거: `validation/gate_recovery/`, `validation/sh_grid_b6/`.

## B. ★오늘 해소된 차단 — MC-EVT NE3 의 전제

08-08 에 Fable 이 MC-EVT 를 `BLOCKED` 로 판정한 사유 셋 중 하나가
"저주파 BF edge census 미완" 이었다. 그 census 는 **707개 활성 edge** 를 찾아
`action=REOPEN_SH_GRID` 를 냈고(최저 witness = Ca II level 61, 5.84852771e13 Hz),
그것이 SH-GRID 재개방 → sigma fresh bake 로 이어졌다.

**2026-08-18 재측정 결과 (오프라인, `validation/evt/BF_EDGE_CENSUS_2026-08-18.log`)**:

```
sigma_grid=[5.84127859e+13, 4.03625815e+16]  levels=24542
positive=24542  above=24542  below_or_at_all=0
[PASS] active_below_or_at_nu_min=0  oog_bf_exact_zero_contract=ELIGIBLE  rc=0
```

**707 → 0.** 자가검사(음성 대조)도 PASS. 새 하한 5.84127859e13 이 최저 활성 edge
5.84852771e13 보다 아래이므로 707개 전부 격자 내부가 됐다.
⟹ `docs/SH_GRID_REOPEN_CONTRACT_2026-08-08.md` §4 가 요구한 "재-census" 조건 충족,
격자 밖 BF 를 exact zero 로 분류할 **자격이 생겼다**(ELIGIBLE).
⟹ **다른 목적의 작업이 열흘 묵은 차단을 풀었다.**

단 재개방 계약은 "CMFGEN 동종 처리 대조" 도 요구한다
(`[5.84127859196e13,1.5e14)` 대역의 level 별 BF rate·emissivity 가 CMFGEN 과 닫힐 것).
`docs/CURRENT_PLAN.md` 의 "저주파 707 level×90 depth 재검증"(global rate 6.962e-5 ·
ion rate 1.573e-4 · weighted-L1 2.455e-4)이 이에 해당하는 것으로 보이나,
**두 문서가 서로를 인용하지 않아 계약 충족 선언은 보류**한다(연결 확인 필요).

## C. ★신규 발견 — 재개방 계약의 사전등록 범위 이탈

`SH_GRID_REOPEN_CONTRACT` 사전등록 vs 실제 구현:

| 항목 | 사전등록 | 실제 | 판정 |
|---|---|---|---|
| BF `dlog` | 0.00529831736655 보존 | 0.0052983173665480362 | **일치**(상대차 −3.7e-13 = 사전등록값의 소수 절단) |
| 하한 | 5.84127859196e13 | 5.8412785919616062e13 | **일치** |
| 상한 | **3.0e16 보존** | **4.0362581455823112e16** | **이탈** |
| bin 수 | **1178** | **1234** | **이탈**(+56 빈, 청색 끝) |

사유는 Si V 바닥 threshold 를 격자 안에 넣기 위한 것으로 보인다
(`docs/HANDOVER_2026-08-18.md` §3.1 "Si V ground threshold 가 새 grid 안에 들어오고
실제 CMFGEN row 가 등록된다"). 물리적으로 정당해 보이나 **재개방 계약의 기대 변경집합에
개정으로 등재되지 않았다.** 단 규약("이 목록 밖의 변경은 실패로 본다") 적용 대상이다.
⟹ 처분 = 조용한 대장 기재(본 문서) + 계약 개정 또는 이탈 승인은 user 판단.

## D. 아직 미실행 — 런 의존 5건

| 게이트 | 단 | 필요한 것 | 비고 |
|---|---|---|---|
| **Γ4** M1/M2 | SH-GAMMA | MC 팔이 A2-10 **항 조립**에 도달 + **셸별** 가열항 장부(감마 포함) | 08-08 기록은 "A2-10 성공 필요" 라 했으나, 항 장부는 NO_BRACKET 에서도 찍힌다 ⟹ **근을 찾을 필요는 없다**. 다만 현행 출력은 shell-0 진단뿐이라 **셸별 계측 추가 필요**(코딩) |
| **R6-4** MC 바이트-parity | DET-R6 | MC 팔 2회 런(결정론 발행 전/후) + **진짜 byte 비교기** | 08-08 실패 사유가 "필터된 로그 줄 비교" 였으므로 비교기부터 만들어야 한다 |
| **NE2** GPU 전송 음수 주입 | MC-EVT | GPU transport kernel 통합 주입 + 런 | Fable: "실제 GPU transport 통합 음수 주입 미실행" |
| **ME2** GPU ON/OFF 스펙트럼 | MC-EVT | GPU 런 2회(event ON/OFF) end-to-end | Fable: 배열 차이 0 은 **필요조건일 뿐**, 수송 분기·RNG 소비 순서가 달라질 수 있어 스펙트럼 대조로만 판정 가능 |
| **E4** 회귀 바이트-parity | MC-EVT | event ON 구성에서 기존 GPU 런과 byte 대조 | R6-4 와 같은 비교기 재사용 가능 |

**공통 선행물 1개**: 진짜 byte 비교기(필터·다중집합 비교 금지).
R6-4·E4 가 같이 쓰고, ME2 의 스펙트럼 대조에도 쓰인다. **이것부터 만드는 것이 효율적이다.**

**자원**: 현재 syn101 GPU 2쌍을 DET-SPROD 판정런(IV·III)이 점유 중.
GPU 2,3,4,5 는 비어 있으나 CPU 32코어가 이미 두 런에 할당돼 있다.
⟹ 판정런 착지 후 순차 발주가 맞다.

## E. 회수 원칙 (오늘 확인된 것)

N6-3 회수에서 검수가 잡은 결함이 교훈이다. Codex v1 의 판정식이 OR 라서
**commit 이 무관한 이유로 실패해도 PASS** 로 보고되는 구조였다 — 주입 경로를 한 번도
밟지 않고 통과한다. 08-08 에 세 단을 강등시킨 결함과 **같은 계급**이다.

⟹ **회수는 "게이트를 돌렸다" 가 아니라 "주입이 시도됐고, 의도한 가드가, 이름 있는
사유로 막았다" 를 셋 다 보여야 한다.** v2 가 `rejected_at` 단계 분류와 항상 출력되는
trace 를 넣은 덕에 이번 실행은 그 셋을 모두 보인다.
