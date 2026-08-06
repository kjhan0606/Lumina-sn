# 층 1 진입 준비 — 「고리 밖 감사」 캠페인

작성 2026-08-06 (운전석). A-2 수리 캠페인 종료 시점에 맞춘 진입 문서.
**이 문서는 층 1을 시작하기 전에 무엇이 준비됐고 무엇이 아직 아닌지를 정직하게 적는다.**

---

## 1. 왜 층 1인가 — 캠페인 원리 재확인

역추적은 닫힌 고리에서 원리적으로 순환한다(자기되먹임이 감도를 은폐). 그래서
**고리가 소비하되 생산하지 않는 것**을 감사한다. 우선순위는 중요도가 아니라
**독립도**다(위상정렬): 1층 완전검증 → 2층 사실만 → 3층 고리얽힘.

근거는 2026-08-03 하룻밤의 실측이다: **살아남은 판정은 전부 1층, 뒤집힌 것은 전부 3층.**

## 2. 진입 조건 — A-2 완료 상태

층 0은 완결됐다(`L0_VALIDATION_CLOSED=yes`). 층 1 진입 조건은 A-2(정본 J_ν 이관)
완료다. A-2 진행 상황은 `validation/a2_*/`의 각 CLOSURE 문서와 원장이 정본이며,
이 문서 §6에 종결 시점 상태를 기재한다.

**A-2가 층 1에 남기는 것**(이것이 진입의 실질적 의미):

| A-2가 만든 것 | 층 1에서 왜 필요한가 |
|---|---|
| 정본 `RadiationField.J_nu` (4000빈) + checked view | 층 1의 I7(격자) 판정이 "1,000 대 196,185"라는 **잘못된 대조**였음이 밝혀졌다. 이제 실제 격자가 무엇인지 코드가 아니라 계약으로 말할 수 있다 |
| `LineJbarCache` (선별 φ-가중 estimator) | I1(충돌강도 Υ)·I2(A_ul)의 영향이 선률로 어떻게 전파되는지 **분리 측정** 가능 |
| bf Γ / bb J̄ 의 checked 소비 + validity | I3(σ(ν)) 불일치가 rate에 미치는 영향을 **미표본과 구분해서** 측정 가능 |
| population/partition 단일 `Z(T_e)` 정본 | I4(슈퍼레벨 분할)·I9(수치 상수) 비교의 분모가 고정됨 |
| signed χ 게시 + `BLOCKED_NEGATIVE_OPACITY_SEMANTICS` | I9의 "Lumina ε clamp는 CMFGEN 대응물 없음" 항목이 **실제로 몇 곳에서 발화하는지** 카운터로 드러남 |
| 잣대 복구(census 앵커 + 회귀 편입) | 층 1 판정을 기록할 원장이 살아 있음을 보장 |

## 3. ★층 1의 첫 행동은 재측정이다 (가장 중요)

`docs/OUTSIDE_LOOP_POOL.md` 층 1 절의 **1차 잣대 감사(2026-08-04)** 결론:

> **층 1 수치 대부분이 구 덱(`_sivcaiv`)의 것이다.** `_ftos`에서는 분모가 통째로 바뀐다:
> ```
> levels        26,592 → 31,792
> Fe IV lines    4,336 → 72,223   (= CMFGEN 원본)
> Ni IV lines    4,199 → 72,898   (= CMFGEN 원본)
> σ addressable 26,592 → 31,792
> ```
> ⟹ **I2·I2a–I2d·I3·I3a–I3c·I17의 분모가 통째로 바뀌었다. 재측정 없이는 어느 것도
> 확정도 제거도 불가.**

**따라서 층 1의 1번 작업은 새 표적 감사가 아니라 `_ftos` 위에서의 전면 재측정이다.**
기존 표를 인용해 결론을 내리는 것은 금지(그 표 자체가 "읽지 말 것" 경고를 달고 있다).

부수적으로 이미 확인된 정정 3건도 승계한다:
- **I7 폐기·재기술**: 역할 대응은 1,000 ↔ 15,662(둘 다 continuum 격자)이지 196,185가 아니다.
  → **A-2 이후에는 4,000 ↔ 15,662로 다시 기술해야 한다**(A2-02가 격자를 바꿨으므로).
- **I2 임계 재설정**: 원자료 유효숫자가 5자리인데 임계 `r>1e-6`은 10× 더 엄격 —
  75,075 불일치가 **반올림을 세고 있을 수 있다**. 동일원본 변환 무결성(exact/ULP)과
  서로 다른 원본의 물리 비교(양자화 반영 임계)를 **분리**해야 한다.
- **I17 명목 해소**: `_ftos`에서 Fe IV 72,223/72,223, Ni IV 72,898/72,898 PASS.
  단 stale `verification.log` 1행 처분 필요.

## 4. 층 1 표적 우선순위 (독립도 순)

| 순위 | 표적 | 근거 |
|---|---|---|
| **0** | **`_ftos` 전면 재측정** | §3. 이것 없이는 아래 전부가 무의미 |
| 1 | **I1 충돌강도 Υ** | ε가 `C/(C+Aβ)`이므로 Υ가 틀리면 **모든 선의 열화·산란 분기가 통째로 편향**되면서 역추적으로는 영원히 안 잡힌다. 실측: Co IV 표 4,455전이 전부가 Fe III 표의 정확한 부분집합(최대 절대차 0) — **Co IV 자료가 Fe III 대용** |
| 2 | I3a Co IV σ(ν) | 46,827/51,411 (91%) 불일치 — I1·I2c와 합쳐 **Υ·A_ul·σ 세 축 전부** 불일치하는 유일 이온 |
| 3 | I8 경계조건 | L 비 31.07이 정의 차이인지 실물인지 미확립. 결판 요건 = **같은 속도좌표에서의 L 대조** |
| 4 | I4 슈퍼레벨 분할 | Lumina `min(level,100)` 대 CMFGEN `F_TO_S` — A-2가 population 정본을 만들었으므로 이제 영향 분리 가능 |
| 5 | I9 수치 상수 | A-2의 클램프 census·카운터가 발화 빈도를 실측해 줌 |

## 5. 층 1이 쓸 수 있게 된 도구 (A-2 산출물)

- `scripts/a2_01_census_contract.py check` — 원장 앵커 무결성(배터리 preflight로 자동)
- `validation/a2_05/L1BF_GATE_LEDGER.json` — bf Γ 대 PRRR, truth-측 f_cov 규약 확립
- `validation/a2_06/A2_06_AUL_CROSSWALK.json` — **A_ul 2,220,953선 전량 match(1.7e-16)**
  → I2 재측정의 기준선. 단 이는 Lumina 내부 일관성이고, CMFGEN 원본 대조는 별개
- `validation/a2_07/` — population/partition 정본과 L-2 게이트 인프라
- 음성대조 패턴(poison별 marker·기대 FAIL·rc)과 truth-측 f_cov 규약 — 층 1 게이트에 재사용

## 6. A-2 상태 (2026-08-06 재판정)

**18/18 폐합.** 정본 = `validation/a2_18/A2_18_CAMPAIGN_CLOSURE.md`(재판정본).

- **폐합**: A2-00~A2-17 전 단계. 초판의 미완 2단계(A2-14/15 → A2-16/17)가 닫혔다.
  원장 157행·미분류 0, `src/` production 스칼라 역참조 0(잔여 1건은 완료 tombstone 주석).
- **게이트**: L-1bf PASS · L-2 self PASS · A2-12 GPU PASS · A2-13~15 마이크로 오라클 일치 /
  L-1bb·L-4·L-3·L-5·L-6 **BLOCKED**.
  ⚠**차단 사유가 바뀌었다**: `BLOCKED_MISSING_*`(파일 부재) →
  **`BLOCKED_ORACLE_NOT_CERTIFIED`**. O-PHYS formal 이 정상 종료해 진리 파일은 전부
  확보됐으나(CHI/ETA 각 346MB, 475,154 레코드 검증), 기계 계약이 오라클 인증을 거부했다:
  `REFUSE: expected exactly F [FIX_T], found ['T']` + population 보정 9e4–1e7%.
  사전 합의했던 `PASS_UNCONVERGED_ORACLE` 명명은 **철회**했다.
  남은 것은 파일 생산이 아니라 **수렴한 free-T 런 하나**다.
- **PASS 세탁 0건.**

층 1 진입에 필요한 A-2 산출물은 전부 확보됐고, 실제로 층 1은 이미 진입해
**I20(공기파장 규약) 확정·수리·인수**까지 마쳤다 — 게이트가 하나도 안 풀린 상태에서.
이것이 "층 1은 고리 안쪽 BLOCKED 와 독립"이라는 §7 판단의 실측 확인이다.

## 7. 층 1 진입을 막지 *않는* 것 — 정직한 미결 목록

다음은 **BLOCKED로 기재된 채 층 1을 시작해도 되는** 항목이다. 층 1은 입력축
감사이고, 아래는 고리 안쪽 물리 판정이라 독립도가 다르기 때문이다.

| 미결 | 왜 층 1을 막지 않는가 | 언제 풀리는가 |
|---|---|---|
| L-1bb `BLOCKED_MISSING_RATE_EXPORT` | 층 1은 **입력 자료**(Υ·A_ul·σ)를 CMFGEN 원본과 직접 대조한다. rate export는 고리 안쪽 검증용 | O-PHYS STAGE-1이 NETRATE/TOTRATE를 내면 |
| L-4 `BLOCKED_MISSING_CHI_DATA` | 동상 | O-PHYS STAGE-1 formal 단계 |
| L-6 `BLOCKED_FIXED_T_AND_MISSING_LINEHEAT` | 복사평형은 3층(고리 얽힘)에 가깝다 | O-PHYS STAGE-2(free-T) 성공 시 |
| `BLOCKED_NEGATIVE_OPACITY_SEMANTICS` (A2-08) | maser 수송 해법은 별도 arm. 층 1 입력 감사와 독립 | 별도 arm |

**반대로 층 1을 실제로 막는 것은 §3의 재측정 하나뿐이다.**

## 8. O-PHYS 상태 (층 1 판정의 진리 공급원)

권위 노트 = `/gpfs/kjhan/cmfgen_runs/OPHYS_RESUME_NOTE.txt` (job ID는 메모리 금지 규약).
STAGE-1(T 고정)로 populations·NETRATE/TOTRATE·CHI/ETA를 먼저 확보하고, free-T는
STAGE-2로 분리했다. **STAGE-1 산출물은 기계 게이트의 ORACLE_INPUT으로 유효**하고
(공시 의무: `run.fix_t=true`, `run.temperature_solved=false`, heat_residual 미달),
복사평형이 필요한 L-6만 STAGE-2 대기다.

층 1 관점에서 STAGE-1의 값어치: **비동결 이온 런**이므로 L-2ion을 괴롭히던
동결 9이온(Si VI·S VI·Ca VI·Fe VI·Fe VII·Ni VI·Ni VII·Co VI·Co VII) 문제가 해소된다.
