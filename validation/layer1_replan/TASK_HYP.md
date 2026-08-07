# 과제 — **가설 수립** 2건 (운전석은 가설을 세우지 않는다)

운전석은 사실 수집과 계측만 한다. **원인 지목은 너의 몫이다.**
증거는 아래에 인라인으로 있고, 원 소스는 `./lumina/` 에 **현재 상태**로 있다(오늘 수리 반영됨).
탐색을 넓히지 마라 — 아래 두 질문에만 답한다.

---

## 배경 (오늘 여기까지 왔다)

`docs`/배선도(`OUT_F_functions_and_wiring.md`)가 정한 구현 목록 R0–R10 중 **R2 하나만** 착지했다.

R2 = **반복 0 물질 공급자**. 반복 안 순서가 수송 → T_e → 플라즈마/tau 이므로 반복 0 의 수송에
solver-owned tau 를 주려면 그 앞에 플라즈마 풀이가 있어야 하는데, 그 풀이가 요구하는
복사장이 없다(인과 순서). 그 지점이 `(void)phi_neb; return POP_BF_STALE` 로 **절단**돼 있었다.

복원 방식: **부트스트랩 창**(런당 1회 래치). `main` 이 K-FRESH 호출을 감싸 열고 **즉시 닫는다**.
창이 열려 있는 동안에만 seed-T_e LTE(Saha)가 이온비를 공급한다(`ratio = phi_neb / n_e`).
노브가 아니다 — env 로 열 수 없고 재진입은 `BOOTSTRAP_REENTRY`.
`lumina_bootstrap_window_{open,close}` · `_active` · `_note_supply` (`lumina_plasma.c`).

---

## 질문 1 — 결정론 팔이 `iter=0` 에서 T_e 자격부여에 실패한다

### [실측] 런 로그 (전문 `run_g1c_full.log`)

```
[A2-10][SEED] bootstrap T_e published generation=1 n_shells=50 manifest=25b03b07ae22
[BOOTSTRAP] iteration-0 material supplier OPEN (provenance=BOOTSTRAP_LTE_SAHA)
  [A2-07] partition Z(T_e) committed generation=1 te_generation=1
    n_e[0]=3.1739e+09, n_e[49]=3.4760e+04
  [A2-07][n_e] charge-conservation residual: max=1.586e-02 at shell 3 (bootstrap)
  [A2-07] level-less reservoir: max fraction=3.980e-28 (x30 bound, limit 1e-02)
[K-FRESH] first consumer=CPU transport/CMFGEN computed_generation=2 required_generation=2 owner=solver
[BOOTSTRAP] CLOSED — supplies=9900; 이후 반복은 fail-closed
=== PURE-CMFGEN deterministic radiation path (MC transport bypassed) ===
[CMFGEN] pure deterministic radiation driver: 50 shells, 1000 bins, 58 rays, 1 outer iters, 8 ALI/iter
  [BF+FF] Shell 0 (optical/UV) ...
  [BF] Macro-atom activation: 49734/50000 bins have valid levels
[CMFGEN][FATAL] radiative-equilibrium T_e not qualified iter=0 (te_generation=0)
[CMFGEN][FATAL] deterministic path failed
```

### [실측] 관측 사실

- seed 발행은 `T_e_generation = 1` 로 성공했다.
- K-FRESH 소비자는 `computed_generation=2 / required_generation=2` 로 통과했다.
- 그런데 **그 뒤 `cmfgen_run` 안에서 `plasma->T_e_generation` 이 0 이다.**
- `a208_publish_cpu_opacity` 실패 진단은 **찍히지 않았다**(그 지점은 통과).
- 사망은 `cmfgen_run` 의 `if (!te_qualified) { ...; return -1; }` 이다.

### 답할 것

1. **`te_qualified` 가 0 인 물리적/배선적 원인은 무엇인가.**
   `compute_radiative_equilibrium_te` 가 요구하는 입력 중 이 시점에 없는 것을
   `파일:행` 으로 지목하라.
2. **`T_e_generation` 이 1 에서 0 으로 바뀐 지점**을 `파일:행` 으로 지목하라.
   그것이 원인인가 결과인가.
3. 이것이 배선도의 **R7(발행 위상)** 소관인가 **R8(세대 보존)** 소관인가, 아니면 제3의 것인가.
4. **다음에 무엇을 계측하면** 이 가설이 확정/기각되는가(운전석이 붙일 진단을 지정하라).

---

## 질문 2 — 최상단 이온 3건의 바닥 g 를 어디서 얻는가 (R0)

### [실측]

로더는 전리에너지 n 개 → population n+1 개를 만들어, 원소마다 최상단 population 의
**속박준위가 0 개**다(실측 15/74, 전부 최상단). 현재는 임시로 `Z_top = 1` 을 대입하고
상한(×30) 게이트로 감싸 두었다(이번 런에서 참 분율 상한 3.98e-28 로 통과).

정본 수리는 ARTIS `SINGLE_LEVEL_TOP_ION` 처럼 **바닥준위 1 개(E₀=0, g₀)** 를 주는 것이다.
필요한 15 이온: C IV · O IV · Mg IV · Al V · Si VI · S VI · Ca VI · Sc IV · Ti V · V II ·
Cr V · Mn IV · Fe VII · Co VII · Ni VII.

CMFGEN 원본 데이터베이스(`/gpfs/kjhan/cmfgen_21jun23/atomic`) 조사 결과 **12/15 존재**:

| 있음 | 없음 |
|---|---|
| CARB/IV · OXY/IV · MG/IV · AL/V · SIL/VI · SUL/VI · CA/VI · CHRO/V · MAN/IV · FE/VII · COB/VII · NICK/VII | **SCAN(Sc) = I,II,III 뿐 → Sc IV 없음**<br>**TIT(Ti) = II,III,IV 뿐 → Ti V 없음**<br>**VAN(V) = I 뿐 → V II 없음** |

### 답할 것

1. 그 3 이온의 바닥 g 를 **어디서** 가져와야 하는가. 이 저장소·디스크 안에 다른 출처가 있는가?
   (없다면 그렇게 답하라 — 지어내지 마라.)
2. 자료가 끝내 없을 때 **fail-closed 처분은 무엇이어야 하는가.**
   현재 `Z_top=1` 임시 대입 + 상한 게이트를 유지하는 것과, 그 원소를 명시적으로
   `POP_ATOMIC_MISSING` 으로 거부하는 것 중 어느 쪽이 규약에 맞는가.
   (규약: 클램프·floor 금지 · 검증불가 고아 금지(외부 앵커로 해소, 삭제·방치 금지) ·
    "틀린 값은 조용히 대장 기재")
3. 12 이온의 osc 파일에서 바닥 (E₀, g₀) 를 읽는 **정확한 규약**은 무엇인가.
   CMFGEN osc 파일의 첫 준위가 바닥이라고 가정해도 되는가? 파일 형식 근거를 대라.

---

## 형식

한국어. `[실측]`(근거 행 있음) / `[가설]` 을 문장마다 구분. `파일:행` 필수.
**수리 코드를 쓰지 마라** — 가설과 계측 지정까지.
확신이 없으면 "무엇을 더 재야 하는가" 를 적어라. 그것도 답이다.
