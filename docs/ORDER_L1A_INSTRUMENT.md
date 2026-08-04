# 발주서 L1-A(v3) — 층 1 **계측기 구축** · 구현 발주

캠페인 「고리 밖 감사」 층 1, 1/4단계. 정본 대장 `docs/OUTSIDE_LOOP_POOL.md`.
**v3 (2026-08-04)** — Codex 검수 7회, 반박 57건 전부 채택. **이번은 구현 발주다.**

---

## 0. 왜 구현으로 넘어가는가

오늘 발주서 7판, Codex 반박 57건. 전부 유효했다. 그런데 **검수가 돌 때마다
발주서가 줄지 않고 늘었다.** 명세 완결성은 원리적으로 끝나지 않는다.

반면 v2 검수는 **핵심이 실현 가능함을 실제로 돌려서 보였다**:
```
두 σ 바이너리 58,384,000점 chunk 대조 → wall 0.57 s, peak RSS 94,720 KB
```
그리고 **I17이 이미 닫혀 있음**을 실제 검증기로 확인했다(운전석 독립 재현,
`VERDICT: all four R1 vintage gates PASS`).

⟹ 남은 반박은 전부 **스키마 필드 추가**이고 Codex가 필드명까지 지정했다.
반영하고 구현한다. 결함이 남으면 **산출물을 감사**한다 — 그 고리는 수렴한다.

---

## 1. 계약

**층 1의 원자데이터 비교를 재현 가능하게 만드는 read-only 계측기를 만든다.**
물리 판정 금지. 기존 수치 수정 금지. 다르면 **보고**.

---

## 2. 엔진과 피복

| 엔진 | ID | 비교량 |
|---|---|---|
| `collision` | I1, **I19** | Υ 표, tabulated/폴백 census, `Υ_eff(T)`·`q_ij(T)` |
| `lines` | I2, I2a–I2d, I4, I12(부분), I17(선) | 선 결합·`A_ul`·`f_lu`·λ, 준위 결합, 커버리지, 슈퍼레벨 |
| `sigma` | I3, I3a–I3c, **I17(σ 커버리지)** | σ(ν), PHOT type별 지원/미지원 |

### 이관 (명시)

`I5a`(DR 설정)/`I5b`(RR·Milne) → L1-C · `I6`(시간/공간범위/속도/밀도 분해) ·
`I7`(역할 대응 **1,000 ↔ 15,662**, 196,185 아님) · `I8`(CMFGEN `L(r)` 추출 계약) ·
`I9`(ε clamp/외부반복/Λ반복/damping/종료조건/온도탐색 분해) · `I10` → **L1-D**
`I11`·`I13`·`I14` → 별건
**`I12`의 macro-atom topology**(`macro_atom_data.csv`·`macro_atom_references.csv`·
`line2macro_level_upper.npy`·`transition_probabilities.npy`) → **별건. 이 계측기는
level/line identity만 부분 피복**
**`I15` build attestation** → 별건. **단 L1-B 발주 전 선행조건**
`I16` → `resolved_path` 기록으로 부분 해소

---

## 3. 레코드 스키마 (비교 metric 1건당 1 object)

```jsonc
{
  "id":     "I1|I2|I2a|I2b|I2c|I2d|I3|I3a|I3b|I3c|I4|I5|I6|I7|I8|I9|I10|I12|I17|I19",
  "metric": "<ID 안의 비교량 이름>",

  "left":  { "role":"", "stage":"", "quantity_definition":"",
             "authority":"", "consumed_path":"", "resolved_path":"",
             "sha256":"", "epoch":"" },
  "right": { /* 동일 */ },

  "build_attestation": {            // 값이 없으면 UNKNOWN. 억지로 채우지 말 것
    "binary_sha":"", "source_tree_sha":"", "dirty_diff_sha":"",
    "build_command":"", "toolchain":"", "env_manifest":"" },

  "universe": { "selection":"", "denominator":0, "cardinality":0,
                "member_manifest_sha":"" },

  "coordinate": { "frame":"", "unit":"", "range":[], "interpolation":"",
                  "extrapolation":"" },

  "sampling": { "rule":"", "sensitive":true, "alternatives":[],
                "weighting":"", "measure":"", "bin_edges":"" },

  "precision": { "digits_left":0, "digits_right":0,
                 "threshold":0.0, "threshold_mode":"exact|ulp|abs|rel",
                 "dtype":"", "endianness":"", "ulp_distance_rule":"" },

  "join": { "keys":[], "normalization":"", "multiplicity":"",
            "duplicate_policy":"", "duplicate_count":0 },

  "states": { "present":0, "missing":0, "zero":0, "unsupported":0 },
  "entity_flags": { "ion_present":true, "quantity_present":true,
                    "counterpart_present":true, "evaluator_supported":true },

  "error": { "absolute":0.0, "relative":0.0, "ulp":0,
             "zero_denominator_rule":"NA|skip|absolute_only" },

  "evidence": { "producer_sha":"", "command":"", "exit_code":0,
                "validator":"", "created_at":"", "negative_control":"",
                "input_shas":[], "record_count":0,
                "evidence_status":"VALID|STALE|FAILED|MISSING|MIXED" },

  "resources": { "peak_rss_bytes":0, "wall_seconds":0.0,
                 "chunk_points":0, "processed":0, "unsupported":0 },

  "verdict": { "posedness":"WELL|ILL|UNVERIFIABLE|NOT_APPLICABLE|UNKNOWN",
               "outcome":"MATCH|DIFFER|NO-COUNTERPART|RESOLVED|PARTIAL|NOT-ASSESSED|INCOMPARABLE",
               "kind":[], "disposition":[] }   // kind·disposition 은 배열(복합 허용)
}
```
`disposition` 값역: `REPAIR|ACCEPT|DEFINE|REMEASURE|CLOSE|NONE`
`kind` 값역: `BUG|DESIGN|DEFINITION|COVERAGE|PROVENANCE|NUMERIC`

### 교차 제약 — 계측기가 자기 출력을 검사, 위반 시 **비영 종료**

| # | 제약 | 차단하는 함정 |
|---|---|---|
| 1 | `states` 4개 합 == `universe.denominator` | I3c "0건=임계 아래" |
| 2 | `left.role == right.role` 且 `left.stage == right.stage` | I7 `1,000 ↔ 196,185` |
| 3 | `sampling.sensitive` 이면 `alternatives.length >= 2` | 조성 61배 |
| 4 | `threshold` < `10^-min(digits_left,digits_right)` 이면 **경고 + 레코드 표시** | I2 `1e-6` 대 5자리 |
| 5 | `threshold_mode == ulp` 이면 `dtype`·`endianness`·`ulp_distance_rule` 필수 | — |
| 6 | 분모 0인 항목에서 `relative` 산출 금지 (`zero_denominator_rule` 적용) | 조성 상대차 미정의 |
| 7 | `left.epoch != right.epoch` 이면 `EPOCH_MIXED` 표시 | epoch 혼합 |
| 8 | `join.keys` 비공허 · `duplicate_count` 기록 · `duplicate_policy` 적용결과 명시 | 결합키 |
| 9 | `id` 가 위 enum 밖이면 스키마 오류 | universe 열거 |
| 10 | `evidence.record_count > 0` 且 `input_shas` 비공허 (비공허성) | 공허한 PASS |
| 11 | `resources.peak_rss_bytes <= 2^30` — **초과 시 FAIL** | 자원 |
| 12 | `build_attestation` 미충전 시 `UNKNOWN` 명시 (빈 문자열 금지) | I15 |

---

## 4. 구현 요구

### 4a. 구조
- CLI 진입점 1개, 엔진 3개 **분리 구현**. 한 엔진 실패가 다른 출력을 오염시키지 않음
- 입력 `--deck --cmfgen-tree --cmfgen-run --engine --chunk-points --threshold-mode` (**하드코딩 금지**)
- **같은 명령이 구 덱(`_sivcaiv`)과 `_ftos` 양쪽에서** 돌아야 함
- 출력 JSONL(레코드/행) + 사람이 읽는 요약표

### 4b. 자원 (v2 검수 실측 기준)
```
line_list 합 724,932,261 B / 4,805,085행     σ 합 467,130,448 B = 58,384,000점
Codex 실증: chunk 1,048,576점 → wall 0.57 s, peak RSS 94,720 KB
```
- σ 전량 적재 금지. memmap/chunk. 88만 결합키를 Python dict로 만들지 말 것
- **1% fixture 벤치마크와 full-run 벤치마크를 둘 다 산출**
- `peak_rss_bytes <= 2^30` 은 **경고가 아니라 실패 조건**

### 4c. golden manifest
`docs/L1_GOLDEN_MANIFEST.json` 에 `command`·`version`·`expected_denominator`·`checksum`.
기존 수치(`880,406`·`75,075`·`3,953,894`·`1,233,529`)를 **코드에 박지 말고** 예상값으로
두고 대조. **다르면 양쪽 다 고치지 말고 보고.**

### 4d. 음성 대조 (엔진별 필수, 픽스처가 결함을 표현하는지 먼저 확인)
- `lines`: 결합키 훼손 → 제약 8 위반 검출
- `sigma`: `states` 합을 분모와 어긋나게 → 제약 1 비영 종료
- `collision`: 없는 이온 요구 → `unsupported`로 잡히고 `missing`과 **구분**

### 4e. I17 실행 계약 — **검증기 이름 정정**
```
scripts/gate_ftos.py 는 존재하지 않는다.  gate_ftos 는
scripts/verify_deck_r4_ftos.py:82 의 내부 함수이며 CLI 는 --off-control 필수.
```
- `scripts/verify_deck_r1_vintage.py --new <deck> --cmf-run <run>` — **운전석 실행 확인:
  4게이트 전부 PASS, 종료 0.** 레코드 `evidence`에 그대로 기록
- `scripts/verify_deck_r4_ftos.py` — 필수 인자 `--new --links --cmf-run --links-deck
  --off-control` 을 확정하고, **이것이 내부에서 R1 검증기를 재실행하는 중복관계**를 명시
- `_ftos/verification.log`(83 B, 오류 1행)는 **`STALE` 표시 + 원인 기록**:
  "dynamic import의 `sys.modules` 미등록 → `dataclasses.py:757`". 현
  `verify_deck_r4_ftos.py:28-32` 에 수정이 있으나 파일이 untracked라 계보는 추론

### 4f. ID별 특수 계약
- **I1**: `구 tabulated 수치`와 `현 NO-COUNTERPART/fallback census`를 **별도 metric**.
  추가로 **공통 온도격자에서 CMFGEN 실제 branch(`TABULATED`/`VR`/`OMEGA_SET`)를 판별해
  `Υ_eff(T)`·`q_ij(T)` 대조**
- **I19**: 두 축 분리 —
  ① `identity`: CMFGEN branch 일치율, `identity_distance_to_current`,
     `authoritative_tabulated_retention`
  ② `physics_change`: 공통 transition×온도에서 old→new `Υ_eff(T)`·`q_ij(T)` abs/rel
  (⚠ 실측: 7이온 mapped 11,329 → 0. 그중 **4,455 = I1의 Co IV 표(= Fe III 값)** 이므로
   그 항목의 상실은 identity 개선이다. 전체 순변화 114,952 → 106,091)
- **I3 계열**: **PHOT type 2/3/8 미지원 2,084레벨**을 `unsupported` 분모로 명시
- **I12**: R1 검증기의 level/rank identity + line bit identity PASS 를 근거로 **부분** 기록.
  macro-atom topology 는 범위 밖임을 레코드에 명시

---

## 5. 규율

1. 물리 판정 금지. 기존 수치 수정 금지 — 다르면 보고
2. **`src/` 금지** — 발주 D의 미커밋 변경(`lumina_atomic.c`)을 보존하라
3. 쓰기 대상은 `scripts/` 아래 자립 파일 + `docs/L1_GOLDEN_MANIFEST.json` 뿐
4. **실행은 운전석**(grammar-debug). 구현·픽스처·문법검사·1% fixture까지가 범위.
   full-run 벤치마크 명령은 보고서에 적고 **돌리지 말 것**
5. 커밋·푸시·PR 금지

## 6. 보고

변경 파일 전수 / 엔진별 구현 위치 / 교차제약 12개가 코드 어디에 있는지 /
음성대조 픽스처 목록과 표현가능성 확인 결과 / **운전석이 실행할 정확한 명령** /
1% fixture 결과(peak RSS·wall) / 스키마에서 구현 못 한 필드와 사유 / 남은 위험.

**교차제약을 세어 12개인지 확인하고 보고서에 그 개수를 적어라** —
운전석이 오늘 개수 산술을 세 번 틀렸다.
