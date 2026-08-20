# 판정 — DET-PHYSCMP **P-1**(계측): 발화점은 133 행, 원인은 `grid_manifest_sha256` 미기재

날짜 2026-08-20 · 사전등록 `docs/RUNG_DET_PHYSCMP_2026-08-20.md`(부록 A·B) ·
판정런 slurm **320568**(a100 / syn101) · 계측 커밋 `ccfaab1`
분담(개정13): 코딩=Codex · 검수·판정=Fable · 빌드·실행·대장=운전석

---

## 1. 판정: **PASS**

사전등록한 기대치 **B1·B2·B3 이 전부 적중**했다. 단 **B3 는 증거가 아니다**(§1-1 자기 정정) — 판정은 **B1·B2** 위에 선다. 세 항목 모두 런 제출 **전에**
커밋돼 있었다(부록 B, 커밋 시각 < 제출 시각).

| # | 사전등록한 기대 | 실측 | |
|---|---|---|---|
| **B1** | 133 자리의 사유 **정확히 한 줄** | `reason=COMPARISON_HASH_INVALID site=133` **1줄** | ✅ |
| **B2** | 네 해시 중 **`grid_manifest_sha256` 만** 불량 | `atomic_model=1 geometry=1 te_manifest=1` / **`grid_manifest=0`** | ✅ |
| **B3** | 나머지 다섯 자리 **0줄** | 신규 계측 BLOCKED **총 1줄**(전부 133) | ⚠**적중이나 증거 아님** — §1-1 |

### ⚠B3 는 적중했으나 **독립 증거가 아니다** (자기 정정, 감리 이전)

[실측] 여섯 자리는 **호출 순서가 하나로 정해져 있다**:
`dump_if_requested`(448) → `snapshot_write`(255 → 258) → `comparison_validate`(99 → 112 → 133).
각 가드는 위반 즉시 **반환**한다. 따라서 **133 이 발화했다는 사실만으로 상류 다섯 자리가
0줄인 것은 논리적으로 강제**된다 — 그들은 통과했기 때문에 조용한 것이지, 관측이
그것을 독립적으로 확인한 것이 아니다. 133 하류에 `INVALID_ARGUMENT` 반환 자리는 없다.

⟹ **B3 를 "세 기대치 중 하나가 맞았다" 로 세는 것은 부풀리기다.** 실질 증거는 **B1·B2 둘**이다.
B3 가 남기는 유일한 정보는 "발화가 반복되지 않았다" 인데, 런이 그 자리에서 죽으므로
이것도 강제된다.

이 오류는 L4 판정에서 감리가 잡은 것과 **같은 형태**다(그때: `BLOCKED 줄 부재`를 근거로
썼으나 그 경로가 `IO_ERROR` 를 반환하므로 부재가 강제였다). 같은 실수를 반복했고,
감리 이전에 스스로 발견해 여기 적는다. 사전등록 부록 B 의 B3 문언 자체가 이미
증거력 없는 항목이었다 — **기대치를 세울 때 "이 관측이 다른 가설 아래서도 같은 값을
내는가" 를 묻지 않았다.**

### 실측 원문 (`/gpfs/kjhan/lumina/det_stage12_fixed_te_a100x2_k36/p1_20260819T231437Z_physcmp_named_reason/stderr.log`)

```
[R7][PHASE] event=R7_MATERIAL_PHASE_COMMITTED lane=DET iter=0 phase=A2-10 r=2 o=3 e=3 te_generation=1->2
[PHYSICS_COMPARISON][BLOCKED] reason=COMPARISON_HASH_INVALID site=133 lane=DET iteration=0
  atomic_model_sha256_valid=1 geometry_sha256_valid=1 te_manifest_sha256_valid=1
  grid_manifest_sha256_valid=0
  atomic_model_sha256=b779a8111617fc5be4993e03066d607fd2fccf0fec3848fed30579f174eaf85e
  geometry_sha256=5f4c3d5bf49338d41d55307030de45f86c34dbb06808a495d8a83aa36d4df15f
  te_manifest_sha256=25b03b07ae2292cfafde72992a6091b860e01307ae5b6c27ed538c0527d1cb4f
  grid_manifest_sha256=
  physical_values_modified=0 clamp=0 floor=0 cap=0 jitter=0 repair=0
[PHYSICS-COMPARISON][FATAL] lane=DET iter=0 status=PHYSICS_COMPARISON_INVALID_ARGUMENT
[DET-TRANSACTIONAL][FATAL] cmfgen_run rc=-1
```

`grid_manifest_sha256=` 뒤가 **비어 있다** — 부록 B 가 예측한 대로 NUL 64개다.

## 2. 이 판정이 무엇을 확정하고 무엇을 확정하지 않는가

**확정한 것**
- ★L4 를 죽인 `INVALID_ARGUMENT` 의 발화점은 **`physics_comparison.c:133`** 이며,
  걸린 필드는 **`em->grid_manifest_sha256`** 하나다. 20일간 상태명뿐이던 자리에
  이름과 실측값이 붙었다.
- ★**이 단(고정-T 레인)이 원인이 아니다.** 온도 발행체의 세 해시가 모두
  `valid=1` 로 나왔다 — 판정문 R5 가 미결로 남긴 (나) 항이 실측으로 닫혔다.
- ★부수: `te_manifest_sha256=25b03b07…` 는 L4 공시의 `te_profile_sha256=25b03b07…`
  와 같다. R5 는 이를 "프로필 해시를 매니페스트 슬롯에 넣은 것 아니냐" 는 의심으로
  열어 두었으나, [실측] `a210_fixed_te_profile_load` 는 **정본 헬퍼
  `population_te_manifest_sha256(profile,n,hash)` 을 그대로 쓴다**
  (소스 주석: *"Existing canonical T_e manifest helper; do not add another hash."*).
  고정레인에서는 프로필과 발행 T_e 가 **같은 배열**이므로 두 값이 같은 것이
  당연하다. **의심 해소 — 결함 아님.**

**확정하지 않은 것**
- `grid_manifest_sha256` 을 **어떻게 채워야 하는가**. 이 단은 측정에서 끝난다.
- 나머지 다섯 자리의 가드가 옳은가. 이번 런에서 **발화하지 않았을 뿐**이며,
  옳음이 증명된 것이 아니다(P-4 단위시험은 12조건 전부를 주입했으나 그것은
  **계측의 정확성** 증명이지 **가드 논리의 정당성** 증명이 아니다).
- 이 게이트를 통과시킨 뒤 비교가 실제로 성립하는지. 아직 아무도 그 너머를 못 봤다.

## 3. 게이트 대조표

| 게이트 | 사전등록 요구 | 실측 | |
|---|---|---|---|
| P1 | 여섯 자리 전부에 이름 있는 사유 | 12개 사유, 여섯 자리 전부 | ✅ |
| P2 | 재현런에서 BLOCKED **정확히 한 줄** + 위반 필드 특정 | 1줄, `grid_manifest_sha256_valid=0` 로 특정 | ✅ |
| P3 | 반환값·차단 시점 불변 | `model_rc=1`, `cmfgen_run rc=-1`, L4 와 동일 지점 | ✅ |
| P4 | 음성 대조: 조건별 **서로 다른** 사유 | 12조건 주입, 사유 문자열 정확 대조(OR 없음) | ✅ |
| — | 음성 대조의 **시연** | 133 사유를 `COMPARISON_HASH_BROKEN` 으로 주입 ⟹ `p4-site-133-reason` **FAIL**, 복구 ⟹ PASS | ✅ |
| — | 오프라인 회귀 | `physics_comparison` / `_regrid` / `det_stage12` / `a2_10_radeq` **4/4 PASS** | ✅ |

## 4. ★방법론 소득 — 오프라인 특정이 계측을 **선행**했고 적중했다

[실측] 부록 B(발화점 특정)는 판정런 **제출 전**에 커밋됐고, 계측 결과가 그것을
글자 그대로 확인했다. 근거는 런이 아니라 `grep` 이었다 —
"이 필드에 값을 쓰는 코드가 저장소에 있는가" 라는 한 질문.

동시에 **부록 A 의 순위는 빗나갔다**(1순위 258, 실제 133). 순위를 세운 근거
("커밋 직후라 불투명도 발행이 아직일 수 있다")는 `[R7][PHASE] o=3 e=3` 과
`physics_comparison.c:472` 가 구조체 **멤버 주소**를 넘긴다는 사실로 즉시
기각됐어야 했다. **순위는 지우지 않고 남긴다** — 부록 A 는 그대로 두었다.

교훈: **"어느 조건일까" 를 추측하기 전에 "그 조건이 읽는 값을 누가 쓰는가" 를 물어라.**
전자는 순위를 낳고 후자는 답을 낳았다.

## 5. 남는 것

1. **A2-09 신원 필드 수리** — 별도 계약. 정본
   `docs/FINDING_A209_IDENTITY_FIELDS_UNPOPULATED_2026-08-20.md`.
   SPEC 요구 3필드(`atomic_model`·`grid_manifest`·`source_manifest`)가 미기재다.
   ⚠수리는 hex64 를 통과시키는 값이 아니라 `nu_edge`·원자모형·소스항에서 **유도한**
   해시여야 하고, 음성 대조는 **생산 경로에서** 격자를 바꿔 해시가 바뀌는 것을 보여야
   한다(픽스처 주입이 이 결함을 20일 숨긴 방법이다).
2. 그 수리 뒤에야 `physics_comparison` 이 처음으로 산출물을 낸다 ⟹ L2·L6 이 열린다.

## 6. 사고 기록
판정런 1차(**320567**)는 다른 프로젝트 세션(lagRamses)이 `squeue -u kjhan` 을
계정 전체가 아닌 자기 것으로 오독해 `scancel` 했다. 손실 = 큐 대기 19분.
재제출 **320568** 로 판정. 상세·재발방지는 단 문서의 「사고 기록」 절.

---
**판정: P-1 PASS.** 단 **DET-PHYSCMP 는 미폐합** — 이 단은 P-1 하나로 끝나지 않으며,
무엇보다 **감리를 아직 받지 않았다**(폐합의 전제, 개정13).
