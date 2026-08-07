# L1-1 게이트 판정 — 2026-08-07

사전등록 `docs/RUNG_L1_1_BOOTSTRAP_SUPPLIER.md`.
자율 체인 `scripts/l1_1_gate_chain.sh`, 원 로그 `validation/l1_1_gates/`.

**계약**: 복사장이 없는 반복 0 에서 물질 상태를 공급하는 **선언적 1회 공급자**가 존재한다.

---

## 판정표

| 게이트 | 내용 | 판정 | 근거 |
|---|---|---|---|
| **G1** | `iter=0` 이 첫 수송 도달 | **미달(기대된 것)** | 결정론 팔이 R7 에서 막힘. 사전등록대로 **이 단의 실패가 아니다** |
| **G4** | seed T_e 비유한 주입 → fail-closed | **PASS** | 9항목 `fails=0`(NaN·0·음수·inf·중복·NULL·n_shells=0) |
| **G5** | 재진입 거부 | **PASS** | 8항목 `fails=0`. 닫힌 뒤 **재개방 불가**(래치) 포함 |
| **G6** | 덱 3종 동일 통과 | **PASS** | `_sivcaiv_active`·`_ophys`·`_jnu4` 세 덱 모두 사슬이 동일하게 섬 |
| **G7** | 전하합 섭동 → 잔차 검출 | **미달(설계 결함)** | 게이트가 **기록만** 하고 문턱이 없어 FAIL 이 시연되지 않는다 |
| **R0 음성** | catalog `{E=0,g=0}` → `POP_INVALID_PARTITION` | **무효 → 결함 적발** | 주입은 걸렸으나(`catalog=1`) 통과했다. 추적 결과 **로더 결함**(아래) |
| **O7** | population m=1 provenance | 관측됨 | `[A2-07] partition Z(T_e) committed generation=1` |
| **O8** | 전하보존 잔차 기록 | 관측됨 | `max=1.586e-02 at shell 3` |

---

## 통과한 것의 실질

### G6 — 사슬이 덱과 무관하게 선다

```
_sivcaiv_active : level-less ion-pop 6개  (Z=14/16/20 stage5, Z=26/27/28 stage6) 전부 catalog 해소
_ophys · _jnu4  : level-less ion-pop 15개, 미매칭 0
공통            : partition committed gen=1 → n_e 수렴 → 이온 → tau → K-FRESH 통과
```

★**전하보존 잔차가 세 덱에서 정확히 동일**하다(`1.586e-02 at shell 3`).
원소가 15 → 6 으로 줄어도 값이 안 바뀐다 ⟹ 이 잔차는 **조성이 아니라 구조적**이다.
R4(n_e 수렴 정책) 착수 시 이것이 출발점이다.

### R7 재현 (덱 3종)

```
[A2-10][PRE] te_gen=1 | radfield: status=0 gen=1 | line: status=-1 gen=0
             | opacity: req=1 com=1 rad=0 pop=1 | **emissivity: com=0**
[CMFGEN][FATAL] radiative-equilibrium T_e not qualified
```
pure lane 에 a209 발행이 없다.  덱과 무관하게 같은 지점.  **다음 단은 R7 이다.**

---

## ★음성대조가 잡은 것 — 내 로더의 미정의 동작

R0 음성대조가 "통과" 한 것이 오히려 결함을 드러냈다.  뷰 진단을 넣자:

```
[A2-07][VIEW] n_ions=33 n_levels=24542 level-less=6 topion_n=750
  ion  4 -> POP_OK Z=2.89658e-65
  ion  9 -> POP_OK Z=4.64478e-66
  ion 14 -> POP_OK Z=4.85097e-66
  ion 20 -> POP_OK Z=4.64478e-66    <- 결함 주입한 이온인데 POP_OK
  ion 26 -> POP_OK Z=3.14639e-65
```

세 가지가 동시에 틀렸다:
1. **Z 가 1e-65 규모** — 분배함수는 최소 g₀(≥1) 여야 한다. 물리적으로 불가능.
2. 결함 주입 이온이 `POP_OK`.
3. **ion 20 과 ion 9 의 값이 완전히 동일** — 다른 이온인데 같을 수 없다.

단위 시험(`population_partition_ion` 직접 호출)에서는 같은 코드가 **옳게** 동작했다
(`g=0 → POP_INVALID_PARTITION`, `g=5 → Z=5`, catalog 없음 → `POP_ATOMIC_MISSING`).
⟹ 로직이 아니라 **자료가 깨졌다**.

원인: `lumina_atomic.c` catalog 로더의
```c
sscanf(ln, "%d,%d,%63[^,],%d,%lf,%lf,%63s", &z,&stage,lbl,&li,&e,&g)
```
**변환 지정자 7개에 인자 6개** — 마지막 `%63s` 에 대응 포인터가 없어 **미정의 동작**이다.
수리: `prov` 를 인자로 추가.  ⚠이것이 3증상을 전부 설명하는지는 **Codex 정합성 검토 중**
(특히 3번: 두 이온이 같은 값).

---

## 다시 짜야 하는 것

| 항목 | 왜 무효였나 | 어떻게 고칠 것인가 |
|---|---|---|
| **R0 음성대조** | 주입은 걸렸으나 로더 결함이 결과를 덮었다 | 로더 수리 확인 후 재실행.  **주입이 실제로 결함을 만들었는지**를 먼저 확인(`catalog=1` 같은 중간 관측)하고 판정 |
| **G7** | 게이트에 **문턱이 없다** — 기록만 한다 | 문턱은 물리 결정(Fable G 조건 8, user 판정 대기).  그 전까지는 **잔차의 민감도**만 시연: 섭동 크기 대 잔차 변화의 단조성 |

★공통 교훈: **주입한 결함이 실제로 존재하는지 확인하지 않으면 음성대조가 무효다.**
오늘 두 건 다 "주입했다고 믿었으나 결과가 통과" 였고, 한 건은 주입 자체가 no-op(잘못된
stage 지목), 한 건은 주입은 됐으나 별도 결함이 결과를 덮었다.

---

## 계약 관점 미결 (Codex 검토 중)

`population_atomic_model_sha256` 이 **catalog 를 해시에 넣지 않는다**.
그래서 catalog 를 바꿔 돌려도 stamp 가 같고 GEN-GUARD 가 침묵한다.
오늘 음성대조가 조용히 지나간 이유이기도 하다.  A2-07/GEN-GUARD 관점 판정 필요.

---

## 결론

**L1-1 계약 자체는 성립한다** — 선언적 1회 공급자가 존재하고, 덱 3종에서 물질 사슬이
서며, 창의 음성대조(재진입·래치·fail-closed)가 전항 통과한다.

**남은 것은 두 가지 성격이 다르다**:
- G1 은 **R7 소관**(사전등록된 기대 결과)
- R0 음성대조·G7 은 **내 시험 설계 결함**(계약의 결함이 아니다)
