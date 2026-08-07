# R6 판정 — 결정론 정본 line-J̄ (2026-08-08 05:5x)

사전등록 `docs/RUNG_R6_DETERMINISTIC_LINE_JBAR.md`.

## 판정: **PASS** — 결정론 팔이 처음으로 A2-10 에 도달했다

```
[R6][LINE-IDENTITY] lane=DET generation=1
    q_set_hash=08db84862c9332ee... profile_id=1 profile_hash=2fe22e7be8f7c80e...
    statistic_kind=DETERMINISTIC
    provenance=A2-06:line-Jbar:deterministic-profile-integral:v1
[R7][PHASE] lane=DET iter=0 phase=view  r=1  line_status=0 line_r=1
[R7][PHASE] lane=DET iter=0 phase=a208  r=1 o=1
[R7][PHASE] lane=DET iter=0 phase=a209  r=1 o=1 e=1
[A2-10][PRE] lane=DET iter=0 te_gen=1 rad=1 line=1 opacity=1 emissivity=1 population=1
```

수리 전 `line_status=-1 line_r=0` 이 `line_status=0 line_r=1` 로 바뀌었고,
a209 가 통과해 **동세대 사중항이 성립**한다.

| 게이트 | 판정 | 근거 |
|---|---|---|
| **R6-1** DET 가 a209 통과 | **PASS** | `phase=a209 r=1 o=1 e=1` |
| 동세대 | **PASS** | rad·line·opacity·emissivity = 전부 1 |
| **R6-4** MC 바이트-parity | **PASS** | 물리 줄 157/157 동일 (아래 §R6-4) |
| **R6-5** 적용범위 | **관측** | 아래 |
| **N6-4** 부분 적용범위가 전체 차단이 아니다 | **PASS** | 70% UNSAMPLED 인데 a209 통과 |
| 정적 | PASS | 새 getenv 0 · clamp 0 · **침묵 실패경로 0** · 노브 4줄 제거 |

## R6-5 적용범위 — 정직한 부분 정보

```
all_lines=2588798  q_lines=1777859  valid_lines=533172  partial_lines=0
unsampled_lines=1244687   valid_pct_qset=29.99%  valid_pct_all=20.60%
valid_cells=26658600  exact_zero_cells=0
```

q-set 178만 선 중 **30% 만** 결정론 J̄ 를 갖는다(생산자가 UV 펌프 창만 적분).
나머지 70% 는 `UNSAMPLED` 로 **정직하게** 올라간다.

★**N6-4 가 주입 없이 성립했다** — 자연 상태가 곧 대조다.
부분 적용범위인데 a209 가 통과했고, 창 밖 선을 실제로 조회하는 SE 만
`POP_BB_UNSAMPLED` 로 막히게 되어 있다.  "부분 = 전체 차단" 이면 게이트 과잉인데
그렇지 않다.

⚠**이 30% 가 물리적으로 충분한지는 이 단이 판정하지 않는다.**  적용범위 수치를
대장에 올리는 것이 이 단의 몫이고, 창을 넓힐지는 물리 결정이다.

## R6-4 — 검사기 잡음을 걷어낸 뒤

체인의 1차 판정은 FAIL 이었으나 **검사기 결함**이었다.  두 로그 전체 diff = **14줄**:

```
2줄  하니스 헤더 (bin=./lumina.preR6 sha=... vs postR6)
12줄 진행 배너와 stderr 의 끼어듦
     pre :   [A2-08][BLOCKED] consumer=T03 ...        (배너 없음)
     post:   Packets: ~10/100[A2-08][BLOCKED] ...     (배너가 앞에 붙음)
```
T03 블록 수는 양쪽 **100 = 패킷 수**로 동일하고, 물리 줄은 **157/157 동일**하다.
진행 배너는 스레드 타이밍 의존이라 물리가 아니다.

★교훈: **바이트-parity 검사기는 stdout/stderr 병합 로그에 그대로 쓸 수 없다.**
같은 바이너리를 두 번 돌려도 끼어듦 위치가 달라진다.  물리 줄만 비교해야 한다.

## 미실행 (정직하게 기재)

| | 상태 |
|---|---|
| **N6-2** q-hash 한 글자 변조 → `QHASH_MISMATCH` | 시험 빌드 필요 — 미실행 |
| **N6-3** 센티널 `-1` 을 VALID 로 위장 → 소비자 거부 | 시험 빌드 필요 — 미실행 |
| **R6-2** 두 팔 해시 동일 | **구조적 보장** — `lumina_main.c` 가 유일 소유자이고 두 팔이 같은 `line_qset` 객체를 쓴다. view 가 이제 profile-hash 까지 검사한다(전에는 profile-id 만) |

---

# ★다음 전선 — 두 팔이 **같은 지점**에서 막힌다

```
DET: [A2-10][BLOCKED] reason=RADEQ_TERM_MISSING missing_term_delta=1
MC : [A2-10][BLOCKED] reason=RADEQ_TERM_MISSING missing_term_delta=1
```

MC 는 원인이 밝혀져 있다 — 사건 측도 부재로 전송이 죽어 복사장이 전부 UNSAMPLED
(`docs/RUNG_EVENT_MEASURE_LANE_AGREEMENT.md`).

**DET 는 다르다.**  결정론 팔은 패킷이 없고 장이 직접 계산되므로 "표본 부족" 이 아니다.
`a210_rebin_checked_J`(`lumina_plasma.c:12335-12344`)의 실패 조건은 둘뿐이다.

```c
if(rf->validity[qi]!=RADIATION_FIELD_VALID && rf->validity[qi]!=RADIATION_FIELD_EXACT_ZERO) return-1;
if(fabs(covered-(hi-lo))>1e-10*(hi-lo)) return-1;   /* 방출률 빈 경계를 복사장 빈이 못 덮는다 */
```

⟹ 다음 단의 표적: **결정론 복사장의 빈 격자와 방출률 발행의 빈 격자가 일치하는가.**
이것은 표본 문제가 아니라 **격자 계약** 문제다.  오프라인으로 특정 가능하다.
