# BLOCKER — 현 HEAD 는 런을 **시작할 수 없다** (K-FRESH ↔ A2-07 계약 충돌)

판정: **BLOCKED_UNRUNNABLE_HEAD**. 수리안은 물리 계약의 순서 문제이므로 **user 판정 대기**.
발견: 2026-08-07, T3 잡 226529/228594/228595 진단 중.

---

## 1. 증상

CPU(`lumina`)·GPU(`lumina_cuda`) **양쪽 main 경로**가 반복 루프에 진입하기 전에
`exit 1` 로 죽는다. 어제까지 **메시지가 없었다** — 이것이 T3 를 여섯 번 죽이고
내가 그 위에 가설을 다섯 번 쌓게 만든 원인이다.

진단을 붙이자 즉시 말했다:

```
[K-FRESH][FATAL] compute_plasma_state failed: POP_INVALID_TE
    (consumer=CPU transport/CMFGEN n_shells=50 T_e_gen=0 err_count=1)
```

## 2. 기전 (오프라인 특정 완료)

두 계약이 **순서에서 충돌**한다.

| 계약 | 출처 | 요구 |
|---|---|---|
| A2-07 population 이관 | `3ddd95c` | `compute_plasma_state` 는 **발행된 T_e**(generation ≥ 1)에서만 돈다. generation 0 = `POP_INVALID_TE` |
| K-FRESH solver-owned tau | `a97d0e1` | 덱 NPY tau 는 seed 일 뿐이므로 **transport 이전에** 덮어써야 한다 |

K-FRESH 는 tau 를 다시 계산하려고 `compute_plasma_state` 를 부른다. 그런데 그 호출은
**반복 루프 앞**(`lumina_main.c:250` · `lumina_cuda.cu:7488`)에 있고, 그 시점의
`T_e_generation` 은 초기값 **0** 이다(`lumina_main.c:152` · `lumina_cuda.cu:7104`).

세대를 올리는 곳은 **루프 안뿐**이며(`lumina_main.c:640`), 그것도
`compute_radiative_equilibrium_te()` 가 T_e 를 자격부여한 뒤에만 오른다 —
그리고 그 함수는 **그 반복의 복사장**을 필요로 한다.

⟹ 루프 앞에서는 발행된 T_e 가 원리적으로 존재할 수 없다. 이 호출은 **성공할 수 없다**.

### 발행자 부재 — 구성적 확인

`lumina_main.c` 152→250 구간의 함수 호출 전수:
`rescale_epoch` · `printf` · `getenv` · `atoi` · `atof` · `strcmp`
그리고 실패하는 `lumina_prepare_solver_owned_tau` 자신.
호출 그래프 1단계에서 `T_e_generation` 을 건드리는 것은 **그 함수 자신뿐**이다(읽기).

### 계보

```
a97d0e1^ : compute_plasma_state 는 루프 안에만 있었다 (lumina_main.c:576, 634)
a97d0e1  : 루프 앞 lumina_prepare_solver_owned_tau 호출을 **추가**
```

`a97d0e1` 은 「고리 밖 감사 층 0: 계약 10건 폐합」 커밋이다 —
**계약을 닫으면서 런을 닫았다.** 이 커밋은 3주·1,369파일이라 이등분 탐색이 안 된다
(기존 지적: `feedback_one_contract_one_commit`).

## 3. 재현 (전부 오프라인, GPU 불요)

`scripts/t3_cpu_repro.sh` — CPU 바이너리로 동일 지점 재현. env 는 T3 런처와 같은 방식으로 만든다.

```
ssh grammar "ssh grammar-debug 'cd <repo> && T3_DECK=data/<덱> PKTS=1000 NITER=1 bash scripts/t3_cpu_repro.sh'"
```

**덱 고유가 아니다** — 세 덱 전부 동일:

| 덱 | 결과 |
|---|---|
| `_ophys` | POP_INVALID_TE, T_e_gen=0 |
| `_jnu4` | POP_INVALID_TE, T_e_gen=0 |
| `_sivcaiv` | POP_INVALID_TE, T_e_gen=0 |

GPU 3잡(226529·228594·228595)도 같은 지점. 표지 `LOAD-STAGE` 7개는 전부 통과했다
(로더는 무죄).

## 4. 반증 시도 — 실패(즉 주장이 살아남음), 단 범위는 좁다

"a97d0e1 이후 성공한 런이 하나라도 있으면 내가 틀렸다" 로 찾았다.
`/gpfs/kjhan` 08-05 이후 런 산출물은 **전부 T3 시도 9건**이고 전부 같은 지점에서 죽었다.

⚠ 정직한 범위: **그 이후 시도된 것이 T3 뿐이다.** "온갖 런이 실패했다" 가 아니라
"시도된 전부(9/9)가 실패했다" 이다. 다른 종류의 런은 아예 발주되지 않았다.

⚠ parent 빌드로 날짜를 못박으려 했으나 **툴체인 비호환으로 막혔다**
(`a97d0e1^` 은 `-std=c11` 에서 `M_PI` 미선언; 플래그를 덮으면 CPU 타깃이 CUDA 심볼을 요구).
따라서 "parent 는 돌았다" 는 **실측하지 못했다** — 계보는 diff 로만 확인했다.

## 5. 왜 계측이 못 봤나 (★부채)

**`main()` 의 기동 경로를 실행하는 게이트가 하나도 없다.**
배터리·selftest·픽스처는 전부 하위 함수를 직접 부르므로 이 충돌을 통과한다.
계약을 10건 닫고도 "런이 시작되는가" 를 아무도 묻지 않았다.

부채 대장 `docs/INSTRUMENTATION_DEBT_CENSUS.md` 에 편입할 것. 후보 수리:
`scripts/t3_cpu_repro.sh` 를 **기동 연막 게이트**로 승격(1 iter·1000 pkt·CPU·수 분).

## 6. 수리 선택지 (★user 판정 — 나는 고르지 않는다)

물리 계약의 순서 문제이므로 운전석이 임의로 정하지 않는다.

| 안 | 내용 | 위험 |
|---|---|---|
| **A** | 루프 앞 K-FRESH 호출을 **제거**하고, 루프 안 첫 `compute_plasma_state` 직후 tau 를 solver-owned 로 확정 | K-FRESH 의도(첫 pure-CMFGEN 조립이 seed tau 를 보면 안 된다)가 깨지는지 확인 필요 |
| **B** | 루프 앞에 **T_e 발행 단계**를 신설 — 덱 seed T_e 를 A2-07 규약으로 1세대 발행 | "seed 를 발행으로 승격" 이 A2-07 취지에 맞는지가 쟁점. `generation-zero material seed` 문구는 그 반대를 시사 |
| **C** | K-FRESH 를 tau **형상/신선도 검사만** 하고 populations 을 요구하지 않게 분리 | 가장 작지만 tau 를 실제로 덮어쓰지는 못한다 |

**클램프식 해법(생성 0 을 허용하도록 검사 완화)은 후보에 넣지 않았다** —
정확해가 위반 가능한 가드를 무르는 것이므로 `feedback_clamps_are_not_physics_fix_the_solver`
판별식에 걸린다.

## 7. 걸려 있는 것

- **T3 덱 A/B 전체** — 판정런이 불가능하다. 회귀 대장 v3 첫 점도 대기.
- 그 이후의 모든 생산 런.
