# 단 L1-1 사전등록 — 반복 0 물질 공급자 복원

**아직 착수하지 않는다.** Fable G 승인 후 구현한다.
offline-first 규약상 **기대 변경집합을 착수 전에** 등록하는 문서다.

계약 1개: *"복사장이 없는 반복 0 에서 물질 상태(Z · n_e · 이온 · 준위 population)를
공급하는 선언적 1회 공급자가 존재한다."*

---

## 1. 기전 (오프라인 특정 완료)

반복 안 순서는 **수송 → T_e → 플라즈마/tau** 다. 따라서 반복 0 의 수송에 solver-owned
tau 를 주려면 그 앞에 플라즈마 풀이가 있어야 하는데, 그 풀이가 요구하는 것이 없다:

```
tau ← population ← 발행된 T_e ← 복사장 ← 수송 ← tau
```

`OUT_C`(Fable) 판정: 이것은 설계 공백이 아니라 **절단**이다.
코드 자신의 주석이 fallback 을 선언해 놓고(`lumina_plasma.c:2347`
*"Fail-closed to the B2 LTE-Saha pin otherwise (mirrors ARTIS's LTE start)"*),
실물은 그 값을 버린다(`2643-2650`: `(void)phi_neb; if(!r1_use) return POP_BF_STALE;`).

`OUT_E`(Fable 인증): 운전석 가설("LTE start 는 MC 전용이라 필요")은 **전제 기각**.
어떤 수송이든 opacity/source 를 요구하고 그것은 population 을 요구한다 — 인과 순서 문제다.
성립하는 사슬은 고리를 하나 늘린 것:

```
seed T_e → LTE(Saha) 물질 상태 → solver-owned tau/χ → 결정론 formal solve
        → 무잡음 J_nu (generation 1) → rate-SE/NLTE → (MC 팔 합류)
```

## 2. 기대 변경집합 (착수 전 등록)

**코드**
1. 반복 0 전용 물질 공급자 — seed-T_e LTE(Saha). `(void)` 폐기 대신 **provenance
   스탬프를 단 선언적 공급자**로 실행.
2. 범위 못박기: **부트스트랩 1회만.** 루프 중 fail-closed 계약은 그대로 둔다.
3. `(void)phi_neb` 뒤의 사문 사슬(zeta 보간 · ML 보정 · twocomp lock ·
   `OUTER_ION_BOOST`) 정리 — 생산 경로에서 결과 무영향인 shadow.
4. 주석 화석 제거: `2347` (코드와 반대를 선언).

**런타임 관측 (이것 말고는 바뀌지 않아야 한다)**

| # | 기대 |
|---|---|
| O1 | 반복 0 에 `[BOOTSTRAP]` 계열 1줄 — provenance 와 셸 수, **런당 1회** |
| O2 | `[A2-07] partition Z(T_e) committed generation=1` 이후 **n_e 수렴** |
| O3 | `POP_BF_STALE` 가 반복 0 에서 사라진다 (반복 ≥1 에서는 **여전히 발생해야 한다**) |
| O4 | `[K-FRESH] first consumer=… computed_generation≥1 owner=solver` |
| O5 | 런이 **첫 수송에 도달**한다 — CPU·GPU 양쪽 |
| O6 | 로더 출력(LOAD-STAGE · ENV-SURFACE · seed 발행)은 **불변** |

## 3. 게이트 (음성대조 의무)

| 게이트 | 내용 | 자격 조건 |
|---|---|---|
| G1 양성 | CPU 재현이 첫 수송 도달, 반복 1회 완주 | — |
| **G2 음성** | 공급자를 끄면 **원래의 `POP_BF_STALE` 가 돌아온다** | 수리가 원인임을 증명 |
| **G3 음성** | 반복 ≥1 에서 장이 깨지면 **여전히 fail-closed** — LTE 로 미끄러지지 않는다 | 범위가 부트스트랩 1회임을 증명 |
| **G4 음성** | seed T_e 에 결함 주입 → 공급자가 **fail-closed** (클램프·보정 금지) | |
| G5 | 두 번째 호출 거부 (부트스트랩은 1회) | |
| G6 | 덱 3종(`_ophys` · `_jnu4` · `_sivcaiv`)에서 동일하게 통과 | 덱 고유가 아님 |

★**G3 가 이 단의 핵심 위험**이다. 부트스트랩 fallback 을 전역으로 열면
반복 ≥1 에서 장이 깨졌을 때 LTE 로 조용히 미끄러져 **결함을 은폐**한다.
그것은 이 캠페인이 고치려는 바로 그 병이다.

## 4. 판정런

CPU 오프라인(`scripts/t3_cpu_repro.sh`, grammar-debug)으로 G1~G6 전부 확인한 뒤
**GPU 판정런 1회**. 그 전에는 GPU 를 쓰지 않는다.

## 5. 하지 않는 것

- 노브를 만들지 않는다. 공급자는 **선언적 1회**이지 스위치가 아니다.
- 클램프·floor 로 수렴시키지 않는다.
- 계약을 완화하지 않는다 — `OUT_C` 판정대로 틀린 것은 **위상과 절단**이지 계약이 아니다.
- 승인된 배선도(Fable G)를 벗어나지 않는다. 벗어나야 하면 재승인.
