# 단 사전등록 — DET-PHYSCMP: DET 레인의 physics_comparison 계약 (2026-08-20)

발단: L4 판정런(`docs/VERDICT_DET_STAGE12_L4_2026-08-20.md`).
결정론 팔이 **처음으로** T_e 세대를 커밋했고(`te_generation=1->2`), 그 직후
`PHYSICS_COMPARISON_INVALID_ARGUMENT` 로 런이 죽었다.

## 계약 (하나)

> **인자 가드가 거부할 때 어느 필드가 왜 거부됐는지 말한다.**

수리가 아니다. 지금은 **사유를 모른다** — 그것이 문제다.

## 1. 실측 — 왜 지금까지 안 보였나

`src/physics_comparison.c` `physics_comparison_dump_if_requested`:
```c
const char *directory = getenv("LUMINA_PHYSICS_COMPARISON_DIR");
if (!directory || !*directory) return PHYSICS_COMPARISON_NOT_REQUESTED;
if (!geometry || !atom || !plasma || !opacity || !nlte ||
    geometry->n_shells < 2 || plasma->n_shells != geometry->n_shells)
    return PHYSICS_COMPARISON_INVALID_ARGUMENT;      /* ← 사유를 찍지 않는다 */
```
- 조건이 **7개** 인데 반환값은 **하나**다. 로그에 남는 것은 상태명뿐.
- 호출부(`lumina_cmfgen.c:7519`)는 DET 레인이고, **결정론 팔이 여기까지 온 적이 없다**
  (그 전에 `RADEQ_NO_BRACKET` 으로 죽었다).
- ★**근거 교체 (감리 R6)**: 초안은 "MC 레인 참조 런에서도 0건(실측: IV 런 stderr)" 이라 적었으나
  **IV 런은 `LUMINA_PURE_CMFGEN=1` 의 DET 전용 런**이다(`lane=MC` 0건) — **DET 런의 침묵을
  MC 레인의 증거로 인용한 범주 오류**였다. 올바른 근거로 교체한다:
  [실측] `/gpfs/kjhan/lumina` 전 `stderr.log` 중 `PHYSICS-COMPARISON` 문자열을 가진 파일은
  **이 L4 런 하나뿐**이다 ⟹ 이 경로는 사실상 전 코퍼스에서 미실행이었다.
- env 는 설정돼 있었고(`LUMINA_PHYSICS_COMPARISON_DIR=…/work/physics_comparison`)
  디렉터리도 생성됐다 ⟹ `NOT_REQUESTED` 가 아니라 진짜 인자 거부다.

## 2. 계급 — 같은 실수의 **네 번째**

"거부는 하는데 사유를 안 남긴다" 계열:
① SH-GAMMA **NC3**(정당한 0 을 무조건 차단) ② MC-EVT **OUT_OF_GRID**
③ A210-ZERO-OPACITY(`INDEPENDENT_SPROBE_UNDEFINED` 가 어느 행인지 안 찍음)
④ **여기**.
③은 아직 열려 있다 — 같은 처방이 두 곳에 필요하다.

## 3. 단계

### P-1 계측 (이 단의 첫 실행) — 판정 로직 불변

★**범위 정정 (감리 R6 — 이 정정이 없으면 단이 답을 못 찾는다)**:
초안은 `dump_if_requested` 의 **7조건만** 계측하려 했다. 그러나 [실측]
`PHYSICS_COMPARISON_INVALID_ARGUMENT` 는 `src/physics_comparison.c` 의 **여섯 자리**에서
반환된다 — **99 · 112 · 133**(`comparison_validate`) · **255 · 258**(`snapshot_write` 진입) ·
**448**(`dump_if_requested` 인자 가드).
진짜 발화점이 448 이 아니면 **계측 후에도 BLOCKED 줄이 0줄**이고, 게이트 P2("정확히 한 줄")가
실패하거나 공허한 근거로 오독된다.
⟹ **여섯 자리 전부**에 사유를 붙인다.

가드를 **조건별로 분리**해 각각 이름 있는 사유를 내고, 위반 필드의 실측값을 찍는다:
```
[PHYSICS_COMPARISON][BLOCKED] reason=<NAME> lane=<..> iteration=<..>
  geometry=<0|1> atom=<0|1> plasma=<0|1> opacity=<0|1> nlte=<0|1>
  geometry_n_shells=<%d> plasma_n_shells=<%d>
```
사유 이름은 **자리마다 구별**되어야 한다. 최소:
- 448: `_GEOMETRY_MISSING` / `_ATOM_MISSING` / `_PLASMA_MISSING` / `_OPACITY_MISSING` /
  `_NLTE_MISSING` / `_SHELL_COUNT_TOO_SMALL` / `_SHELL_COUNT_MISMATCH`
- 255 · 258: `_SNAPSHOT_INPUT_MISSING` / `_SNAPSHOT_BIN_OR_SHELL_INVALID`
- 99 · 112 · 133: `comparison_validate` 의 각 조건에 대응하는 이름(그 함수를 읽고 정하라)

⚠**어느 자리가 발화하든 정확히 한 줄이 나와야 한다.** 여섯 자리 중 다섯이 조용하면
계측이 실패한 것이다.

⚠**반환값·차단 시점 불변.** 여전히 `PHYSICS_COMPARISON_INVALID_ARGUMENT` 를 돌려주고
같은 자리에서 거부한다. **이 단은 측정 단이다.**

### P-2 수리 (P-1 결과를 보고 결정 — 지금 고르지 않는다)
후보: 호출부가 누락 필드를 채운다 / 가드가 과잉이면 완화(단 **거부→통과 전환은 별도 승인**) /
DET 레인에서는 이 덤프를 요구하지 않는다.

## 4. 게이트

| # | 조건 |
|---|---|
| **P1** | 빌드 CPU+GPU 두 타깃, 에러 0 |
| **P2** | 재현런에서 `[PHYSICS_COMPARISON][BLOCKED] reason=…` 이 **정확히 한 줄** 나오고 위반 필드가 특정된다. ⚠**여섯 반환 자리 전부**가 계측돼 있어야 이 게이트가 유효하다(감리 R6) |
| **P3** | **판정 불변** — 반환 상태와 종료 시점이 L4 런과 동일(`INVALID_ARGUMENT`, 커밋 직후) |
| **P4** | 음성 대조: **여섯 자리의 모든 조건**을 단위 시험에서 주입해 **서로 다른 사유**를 낸다 |
| **P5** | MC 레인 무접촉 — `lumina_main.c:739` 경로의 거동 불변 |

★**P3 가 이 단의 자기 규율이다** — 계측이 판정을 바꾸면 측정 단이 아니다.
★**P4 가 NC3 다** — 사유가 갈리는지를 주입으로 시연한다.

## 5. 기대 변경집합

- `src/physics_comparison.c` — 가드 분리 + 진단 출력. **반환값 불변.**
- `tests/physics_comparison_selftest.c` — P4 음성대조 7건 추가.
- 그 외 파일 변경 없음. 물리식 무접촉. 새 env 노브 없음.

## 6. 이 단이 열면

L4 런이 커밋 직후 죽지 않고 진행 ⟹ **L6(반복 1 population 의 LTE 이탈)** 판정이 가능해진다.
그것이 DET-STAGE12 의 마지막 미판정 게이트이며, STAGE-1 → STAGE-2 전이의 전제다.

## 7. 판정 절차 (개정13)

사전등록·검수·판정·감리=Fable(감리는 독립 컨텍스트) / 코딩=Codex(clean worktree) /
빌드·실행·대장·커밋=운전석. 런은 slurm, **a100 전용 · `--gres=gpu:2` · `--mem` 명시**.

---

## 부록 A — 발화점 후보의 **사전등록** (P-1 산출 전에 적는다)

P-1 이 답을 가져오면 사후에 "그럴 줄 알았다" 로 각색하기 쉽다. 그래서 계측이 돌기 **전에**
후보와 근거를 확정해 둔다. **맞히는 것이 목적이 아니라, 빗나감을 기록할 수 있게 하는 것이 목적이다.**

[실측] 여섯 자리를 소스에서 읽어 조건을 분해했다(`src/physics_comparison.c`):

| 자리 | 조건 요지 | 순위 | 근거 |
|---|---|---|---|
| **258** | `nb = in->opacity ? in->opacity->n_bins : 0;` → `!nb \|\| !in->n_shells \|\| ...` | **1** | FATAL 은 `R7_MATERIAL_PHASE_COMMITTED ... te_generation=1->2` **직후**다. T_e 세대는 막 커밋됐지만 그 세대의 **불투명도 발행은 아직**일 수 있다 ⟹ `n_bins==0` |
| **133** | `comparison_hex64()` 를 `atomic_model_sha256` · `geometry_sha256` · **`te_manifest_sha256`** · `grid_manifest_sha256` 에 요구 | **2** | ★이 단이 **`te_manifest_sha256` 슬롯에 프로필 sha256 을 새로 써 넣는다.** 고정레인이 나머지 셋 중 하나를 비워 두면(예: 매니페스트 미실행 — L3 가 부분 PASS 인 바로 그 이유) 여기서 걸린다 |
| **112** | `te->n_shells != ns` · `te->ledger/shell_status/residual_status` 비어 있음 등 | 3 | 발행체 배열이 미할당이면 걸림 |
| **99** | `temperature_publication` 자체를 포함한 15개 포인터·범위 검사 | 4 | 발행 자체는 일어났으므로(공시 2줄 실측) 낮음 |
| **255** | `!directory \|\| !*directory \|\| !in` | 5 | 448 에서 이미 통과한 조건과 겹침 |
| **448** | geometry/atom/plasma/opacity/nlte 널 + 셸수 정합 7조건 | 6 | 50셸 정합은 상류에서 여러 번 확인됨 |

**[중요] 이 표는 순위일 뿐 판정이 아니다.** 여섯 자리 전부를 계측하는 이유가 이것이다 —
1순위만 계측했으면 2·3순위였을 때 **0줄**이 나왔을 것이다(감리 R6 의 요지).

**빗나감의 처분**: 실제 발화점이 1·2순위가 아니면 판정문에 그대로 적는다. 순위를 지운다거나
"사실 그것도 예상 범위였다" 로 쓰지 않는다.

### 부수 관찰 — 이 자리는 `STALE_GENERATION` 이 아니다
[실측] `comparison_validate` 의 **세대 정합 블록**(`tg/pg/rg/og/eg` 16개 등식)은 실패 시
`PHYSICS_COMPARISON_STALE_GENERATION` 을 돌려준다. 관측된 상태는 `INVALID_ARGUMENT` 이므로
**세대 불일치는 원인이 아니다** — 이 배제는 계측 없이 지금 확정할 수 있다.
