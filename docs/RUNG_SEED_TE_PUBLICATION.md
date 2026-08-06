# 단(rung): 첫 수송 전에 solver-owned tau 가 존재하게 한다 — seed T_e 1세대 발행

**계약 1개.** user 판정 = 안 **B**(2026-08-07). 선행 진단 = `docs/BLOCKER_KFRESH_TE_GENERATION.md`.
이 문서는 **패치 이전에** 기대 변경집합을 등록한다(물리 복원 사다리 규약).

---

## 1. 왜 B 인가

반복 안 순서는 **수송 → T_e → 플라즈마/tau** 다. 따라서 반복 0 의 수송에는
솔버가 만든 tau 가 원리적으로 존재할 수 없다:

```
tau ← population ← 발행된 T_e ← 복사장 ← 수송 ← tau
```

이 고리를 끊는 정직한 지점은 **"첫 상태는 seed 온도의 LTE"** 하나뿐이며,
CMFGEN·ARTIS 가 실제로 하는 방식이다. A2-07 의 `generation-zero material seed` 는
seed 를 **소비 금지**로 두었는데, B 는 그것을 **명시적 1세대 발행으로 승격**한다 —
즉 계약 개정이며, 그래서 user 판정 사항이었다.

## 2. 구현 중 발견한 선행 결함 (별도 단)

`PlasmaState plasma;` 는 **영초기화되지 않는 자동 변수**다
(`lumina_main.c:87` · `lumina_cuda.cu:6886`; main 은 각각 75 · 6796 행에서 시작).
같은 블록에서 `config` 만 `memset` 된다.

⟹ `plasma.te_publication` 이 쓰레기값이다. 반복 0 의 첫 radeq 는
`ElectronTemperaturePublication old=*pub; *pub=c; a210_publication_free(&old);`
를 하므로 **쓰레기 포인터를 해제**한다. K-FRESH 를 통과시키면 즉시 이걸 밟는다.

## 3. 발행 불변식 (실측으로 확립)

radeq 의 규약:

```
gen = plasma->T_e_generation + 1
a210_solve_transaction(...) 성공 → te_publication.committed_te_generation = gen
호출부 → plasma->T_e_generation++          (= gen)
compute_plasma_state → partition_stamp.te_generation = plasma->T_e_generation
A2-10 대조 → stamp.te_generation == publication.committed_te_generation ✔
```

seed 발행은 **같은 불변식**을 지킨다: `gen=1`, `committed=1`, `T_e_generation=1`,
manifest = `population_te_manifest_sha256(seed T_e)`.
반복 0 의 radeq 는 `gen=2` 로 풀고 commit 2 → 스탬프 2. 일관.

## 4. 기대 변경집합 (사전등록)

**코드**
1. `lumina_main.c` · `lumina_cuda.cu`: `memset(&plasma, 0, sizeof(plasma))` 추가
2. `lumina_plasma.c`: `lumina_publish_seed_te()` 신설 · `lumina.h` 선언
3. 두 main: K-FRESH 호출 **직전**에 seed 발행 호출

**런타임 관측 (이것 말고는 바뀌지 않아야 한다)**
| # | 기대 |
|---|---|
| O1 | `[A2-10][SEED] bootstrap T_e published generation=1 …` 이 런당 **1회**, K-FRESH 줄보다 먼저 |
| O2 | `[K-FRESH] first consumer=… computed_generation=1 required_generation=1 owner=solver` |
| O3 | 런이 반복 0 에 진입한다 (기존: `POP_INVALID_TE` 로 exit 1) |
| O4 | A2-10 계수 덤프에 `seed_generation_attempts=1` |
| O5 | 그 이전 출력(로더·LOAD-STAGE·ENV-SURFACE)은 **불변** |

## 5. 게이트 (음성대조 의무)

| 게이트 | 내용 | 자격 |
|---|---|---|
| G1 양성 | CPU 재현이 K-FRESH 를 통과하고 반복 ≥1 완료 | — |
| **G2 음성** | T_e 에 NaN 주입 → 발행이 **fail-closed**. 클램프·통과 금지 | 주입으로 FAIL 시연 |
| **G3 음성** | seed 발행을 끄면 원래의 `POP_INVALID_TE` 가 **돌아온다** | 수리가 원인임을 증명 |
| **G4 음성** | 두 번 발행 시도 → 거부(부트스트랩은 1회) | |
| G5 | `T_e <= 0` · `n_shells<=0` · NULL → 거부 | |

G2·G4·G5 는 덱을 건드리지 않는다 — `tests/seed_te_publish_selftest.c` 가
함수를 직접 부른다(덱 정본 불변 규약).

## 6. 클램프 금지 준수

seed T_e 가 비유한·비양수면 **고치지 않고 거부**한다.
"솔버가 답을 내게 만든다" — 잘못된 seed 를 눌러 담는 순간 그것이 클램프다.
