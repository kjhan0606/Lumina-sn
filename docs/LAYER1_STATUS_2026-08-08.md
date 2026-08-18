# 1층 현황 — 2026-08-08

★단 이름은 `docs/RUNG_NAMING.md` 규약(DET-/MC-/SH- 접두)을 따른다.

기준 목록 `docs/LAYER1_REPLAN_2026-08-07.md`("물리가 한 조각씩 서는 순서").
이 문서는 **그 목록에 대한 진척 보고**다.  각 단은 게이트 통과 시점에 커밋됐다.

## 2026-08-08 후속 구현 갱신

- Fable의 MC-EVT 판정 조건을 실제 flight 덱에 적용한 BF edge census에서
  `NLTE_NU_MIN` 이하 기본 활성 edge **707개**를 확인했다. 따라서 OOG BF를 0으로
  선언하는 안은 기각됐고 **SH-GRID는 재개방**됐다.
- SH-RADEQ 선 방출 생산자를 `chi*line_source_S`에서
  `n_u A_ul h nu beta_esc/(4 pi dnu)` 직접식으로 교체했다. tau=0 cancellation은
  `beta=1`의 유한 방출로 자가검사한다.
- A2-10 schema는 `RE_INTEGRAL`을 유일한 온도 producer로, `EHB_THERMAL`을 독립
  diagnostic으로 분리했다.
- 현 단열항은 `ELECTRON_TRANSLATIONAL_ONLY`이므로 fixed/free-T 모두
  `RADEQ_INCOMPLETE_ADIABATIC`으로 fail-closed한다. 아래 `RADEQ_NO_BRACKET`은
  이 수리 전 flight의 역사적 실측이며, 새 코드의 다음 예상 정지는 불완전 단열항이다.
- Fable 중요 통합 재심은 raw τ writer census/양끝 bracket, 공유 NLTE authority,
  공유 LTE/NLTE population, signed τ 등록, SH-GRID 소비계약을 모두 확인해
  `IMPLEMENTATION_CLOSURE=ACCEPT`로 판정했다. 동시에
  `FLIGHT_STATE=BLOCKED_INCOMPLETE_ADIABATIC`을 유지했다.
- CPU/OpenMP/full CUDA `sm_80` 링크와 A2-07/08/09/10 및 정적·음성대조 gate가
  통과했다. 로그인 노드에서는 모델 flight를 실행하지 않았다.

근거: `validation/evt/CODEX_BF_EDGE_CENSUS_2026-08-08.md`,
`docs/FABLE_VERDICT_SH_RADEQ_2026-08-08.md`,
`docs/FABLE_PREFLIGHT_CLOSURE_SH_RADEQ_2026-08-08.md`,
`validation/a2_09/SH_RADEQ_FABLE_REVISE_CLOSURE_2026-08-08.md`,
`docs/ORDER_SH_RADEQ_CMFGEN_TERMS_2026-08-08.md`.

---

## 요약

**하루 전 상태**: HEAD 가 **런을 시작조차 못 했다**(K-FRESH 가 루프 앞의 T_e 를 요구하나
발행자는 루프 안뿐 — 원리적 불성립).

**현재**: 결정론 팔이 `seed T_e → 물질 → tau → 장 → a208 → a209 → A2-10` 까지 도달하고
**A2-10 이 실제로 솔브를 돈다**.  막히는 곳이 **배선에서 물리로 넘어갔다**.

```
08-07 아침  : 런 시작 불가
08-08 새벽  : [A2-09][BLOCKED] blocked_stale_line   (발행 위상은 섰으나 line-J̄ 부재)
08-08 오전  : [A2-10][BLOCKED] RADEQ_TERM_MISSING   (입력을 모을 수 없다 — 격자)
08-08 현재  : [A2-10][BLOCKED] RADEQ_NO_BRACKET     (입력을 모았고 풀었으나 근이 없다)
```

---

## 1. 단계별 현황

| 단계 | 계약 | 상태 | 근거 |
|---|---|---|---|
| **L1-1** 반복 0 물질 공급자 | 복사장 없는 반복 0 에 물질을 공급하는 **선언적 1회 공급자** | **폐합** | G4·G5·G6 PASS(덱 3종), 부트스트랩 창=런당 1회 래치. `validation/l1_1_gates/` |
| **L1-2** 세대 장부 위상 정합 (=**SH-PUB**) | commit/view 직후 a208·a209 발행 → 그 다음 A2-10.  실패 시 `(T_e,t)` 보존 | **폐합** `9012995` | `[A2-10][PRE] lane=MC te=1 r=1 line=1 o=1 e=1 m=1`, 검사기 `violations=0` |
| **L1-3** 결정론 부트스트랩 + 두 팔 합류 | 최소조건 ①세대 단일장부 ②pure a209 ③결정론 line-J̄ | **부분 폐합** | ②=SH-PUB ✓ · ③=DET-R6 ✓ · ①=SH-R5 **미착수**.  두 팔 합류는 **MC-EVT 대기** |
| **L1-4** 계측 배선도 재작성 | 3열 어긋남 지도(J^MC · J^det · J^CMFGEN) | **미착수** | L1-3 완주 전제 |
| **L1-5** 물질 입력 물리화 | 시작 상태를 물리로 | **1/5 폐합** | 아래 |
| **L1-5b** 비열적 퇴적 물리화 | 비열 excitation 부재 · 상수효율 처방 | **미착수** | SH-GAMMA 가 **소유권**만 폐합 |
| **L1-6** 입력축 감사 재개 | 원자데이터 대 CMFGEN 직접대조 | **미착수** | 작동하는 런 전제 |

### 목록에 없던 단 (진행 중 발견 — 전부 폐합)

| 단 | 무엇이었나 | 상태 |
|---|---|---|
| **SH-GAMMA** 감마 침착 소유권 | 결정론 팔의 방사성 가열이 **항등 0** 이었다 | **부분 폐합** `5450518` — Γ2-b **무효**(다중집합 비교), NC2·NC4·Γ4 미실행 |
| **DET-R6** 결정론 정본 line-J̄ | 생산자는 있으나 commit 에 안 실려 a209 가 막힘 | **부분 폐합** `3ca077d` — R6-4 가 byte-parity 보다 약함, N6-2·N6-3 미실행 |
| **SH-GRID** 격자 포함 계약(안 B) | 생산자·소비자가 **같은 격자**인데 중개 격자가 양 끝을 파괴 | **부분 폐합** `2e26c2f` — B-6 **미측정** |
| **MC-EVT** 사건 측도 | 같은 부재에 세 지점이 서로 다르게 행동 — CPU MC 전송 전멸 | **발주 중** |

---

## 2. 폐합된 단의 실질 (게이트 근거)

### SH-PUB — 발행 위상
- 위상 `view → a208 → a209 → A2-10`, 동세대 사중항 성립
- **R8 보존이 실물 실패에서 시연**됐다(주입 불필요) — A2-10 이 실제로 실패했고
  T_e 매니페스트·세대가 **둘 다 보존**된 채 표면화 종료

### SH-GAMMA — 감마 침착 소유권
- ★게이트가 **양방향**으로 갈린다: `NC1`(발행자 제거)→`RADEQ_GAMMA_UNPUBLISHED` ·
  `NC3`(Ni·Co=0, 전 셸 침착 0)→**발행 성공·런 진행**
- Γ2-b 바이트-parity: 수치 다중집합 pre=post=514, 차집합 공집합
- 폐합한 결함 3건: 외부파일 비열률 누락(기왕수리 확인) · 결정론 팔 생산자 부재 ·
  CUDA 에서 `DEPOSITION_FILE` 주입을 반복 1부터 내부 Bateman 이 무가드로 덮어씀

### DET-R6 — 결정론 정본 line-J̄
- `line_status=-1 gen=0` → `line_status=0 gen=1`, a209 통과
- 적용범위 정직 발행: q-set 1,777,859 선 중 **valid 533,172(29.99%)**, 나머지 UNSAMPLED
- ★N6-4 가 **주입 없이** 성립 — 70% 가 UNSAMPLED 인데 a209 통과("부분=전체차단" 아님)

### SH-GRID — 격자 포함 계약(안 B)
- canonical 을 BF 격자의 **정수 세분(K=2)으로 파생** — 주파수 리터럴 제거,
  `j=0 ≡ NLTE_NU_MIN` 이라 정렬이 구성상 성립, `_Static_assert` 로 동결
- ★**왕복 항등식 `max_abs=0`** — 기계정밀도가 아니라 비트 단위 정확 일치.
  이 검사는 **구 격자에서는 원리적으로 세울 수 없었다**(dln 비율 2.069684)
- 덱 정본 불변 유지(`NLTE_*` 미변경 ⟹ `cmfgen_sigma_bf.bin` 무영향)

---

## 3. L1-5 물질 입력 — 항목별

| 항목 | 08-07 판정 | 현재 |
|---|---|---|
| top-ion `Z=1` 임시 대입 | NON-PHYSICAL | **폐합** — catalog 기반 `Z(T)` (15이온 7,242준위, 단일 vintage).  런에서 `Z=4.96·2·4.21·18.46·24.05·21.04` 관측 |
| 초기 `T_e` = `T_inner` 전 셸 복제 | NON-PHYSICAL(프로파일로서) | **미해결** — `lumina_atomic.c:1058-1061` 그대로.  회색 moment 기반 대안은 발주서만 있음 |
| 덱 `transition_probabilities.npy` 상시 기본 | NON-PHYSICAL(기본값으로서) | **미해결** |
| n_e 5% 문턱 | 5% 선언이 NON-PHYSICAL | **미해결** — 전하보존 잔차 `1.586e-02`(덱 3종 동일 ⟹ **구조적**) 장부화만 됨 |
| `solve_radiation_field`·`coupled_*` 무연산 스텁 | NON-PHYSICAL(화석) | **미해결** |

---

## 4. 지금 막고 있는 것 — 둘

### (a) `RADEQ_NO_BRACKET` — **SH-RADEQ**
★감사 정정: "물리 실패" 가 아니다.  잔차 정의 자체가 CMFGEN 과 다르다.
A2-10 이 입력을 모두 모으고 솔브를 돈다.  그러나 `[10 K, 1e7 K]` **양 끝에서 잔차 부호가
같다** — ⚠**"어디에도 근이 없다" 는 수학적으로 틀린 추론이다**(중간에 두 근 가능).
증명되는 것은 **현재 이분법이 이 구간으로 bracket 을 못 만든다**뿐이다.
가열·냉각 항 중 하나가 부호나 규모에서 크게 틀렸다는 뜻이고, **오프라인 특정 가능**하다
(셸 하나에서 항별 장부를 찍으면 된다).

### (b) 사건 측도 부재 — **MC-EVT**, 발주 중
`lumina_transport.c:571` 이 `bf->event_enabled=0` 이면 패킷을 재흡수하고 끝낸다.
그 값은 `LUMINA_FIX_BF_CONTINUUM_EVENT`(기본 0)이고 **저장소의 어떤 런처도 켜지 않는다**
(`BF_OPACITY=1` 런처 299개 / `EVENT=1` 런처 0개).
실측 800 패킷 → T03 블록 정확히 800건 → 복사장 전 빈 UNSAMPLED.

★같은 부재에 **세 지점이 다르게** 행동한다: CPU 전송=패킷 살해 / CPU a208=`bfnet` 대체 ·
음수만 차단 / GPU 전송=`chi_bf` **무검사 대체**.  두 양은 같은 양이 아니다
(`event_chi_bf`=자발흡수만 ≥0, `bfnet`=순 불투명도로 **음수 가능**).
⟹ GPU 는 음수를 사건 확률로 쓸 수 있다.  **정직한 쪽이 멈추고 조용한 쪽이 생산을 계속했다.**

---

## 5. 미결 (판단 대기)

| # | 항목 | 상태 |
|---|---|---|
| 1 | SH-R1 물리적 seed T_e 단일 선언식 | 평면 Eddington **실측 기각**(37.4% vs 22.9%).  CMFGEN 은 회색 moment 를 푼다.  발주서 있음 |
| 2 | n_e 수렴 문턱 | 잔차 `1.586e-02` 가 덱 3종 동일 ⟹ 조성이 아니라 **구조적** |
| 3 | catalog 가 `population_atomic_model_sha256` 에 미포함 | catalog 를 바꿔도 stamp 동일 → GEN-GUARD 침묵 |
| 4 | MC-EVT OFF 경로(legacy argmax) 존폐 | **ME2**(두 생산자 측도 차이) 측정 후 결정 |
| 5 | DET-R6 적용범위 30% 가 물리적으로 충분한가 | 창 확장은 물리 결정 |

---

## 6. 방법론에서 확립된 것 (재사용)

- **게이트는 양방향으로 갈려야 자격이 있다** — Γ단이 NC1(부재→차단)과 NC3(정당한 0→통과)를
  둘 다 시연해야 했다.  한쪽만 봤으면 과잉 게이트를 "잘 막는다"로 읽었을 것이다
- **주입한 결함이 실재하는지 독립 검사로 먼저 확인** — 안 하면 음성대조가 무효다
- **판정 기준은 단 경계와 정확히 같아야 한다** — 오늘 R7·격자B 에서 두 번 어겼다
  (남의 단의 실패를 내 단의 실패로 적었다)
- **"env 를 넘겼다" ≠ "그 env 로 돌았다" ≠ "그 팔의 온전한 구성으로 돌았다"** —
  세 번 데였다.  하니스에 `T3_LANE`·`T3_BIN` 을 넣어 관측으로 확인한다
- **바이트-parity 검사기는 stdout/stderr 병합 로그에 쓸 수 없다** — 같은 바이너리를
  두 번 돌려도 끼어듦 위치가 달라진다.  물리 줄만 비교한다
- **빌려온 계약은 원래 양의 불변식을 데려온다** — Γ단이 침착률을 온도 매니페스트로
  해싱해 모든 덱에서 발행 불가가 됐다
