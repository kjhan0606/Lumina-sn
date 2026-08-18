# DET-SPROD S4 (III 음성대조) 판정 — 2026-08-19

사전등록 `docs/RUNG_SPRODUCER_CAPTURE_2026-08-18.md` 게이트 **S4**.
결과: **BLOCKED — 음성대조가 기존 계약의 구멍을 찾았다.** 이것이 S4 의 성과다.

## 0. 봉인

- run root: `/gpfs/kjhan/lumina/a210_sproducer_capture_a100x2_nonoverlap_sobolev_k36/manual_20260818T122358Z_sprod_iii`
- binary SHA `3fc2cbbcf6f8788357e6242dcb061d2c55746cf5cd47a557bbccbe1eeacf40e5` (IV 와 동일)
- 최종 stderr SHA `87ac4e1c0447119dc877b686c366ba56094f20b36da88cc9747b3b4087c7591e` (864,632 B)
- 종료: 자연 `RADEQ_NO_BRACKET` → `R7_MATERIAL_UPDATE_BLOCKED`(세대 보존) → rc=4, child_rc=70

## 1. IV 와 동일했던 것 (진단 노브가 물리를 안 건드린다는 실측)

| 마일스톤 | IV(ion=3) | III(ion=2) |
|---|---|---|
| R1 exact solve | 45회, 9.6662782724980344e-09 | **동일** |
| R2 exact solve | 52회, 8.1222406993212508e-09 | **동일** |
| INDEPENDENT-JCONT | PASS, cells 109,014,300 | **동일** |
| SPRODUCER-CAPTURE | CAPTURED, cells 109,014,300, n_shells 50 | **동일** |
| 브래킷 4국면 | 전부 NO_BRACKET | **동일** |

⟹ `LUMINA_A210_LINE_SATURATION_TARGET_ION` 은 진단 선택자일 뿐 물리 경로에 영향이 없다.

## 2. 차단

```
[A2-10][LINE-SATURATION-BLOCKED] reason=INDEPENDENT_SPROBE_UNDEFINED
    phase=REQUESTED_TE shell=0 candidate_rows=51807 target_ion=2 complete=0
[A2-10][LINE-SATURATION-BLOCKED] reason=UPSTREAM_LINE_SCAN_INCOMPLETE
[A2-10][LINE-NET-BLOCKED] status=RADEQ_TERM_SCHEMA line=262210 shell=0
```
Stage-4 row **0행**. ⟹ S2·S2b·S3 를 III 에서 판정할 수 없다.

## 3. ★귀속 — 내 변경이 아니고, 계약 자체의 모순이다

**내 변경이 아님(확증 2)**
1. `git log -S 'INDEPENDENT_SPROBE_UNDEFINED'` → `f6c2eb6`(08-17 독립 캡처 작업).
   오늘 DET-SPROD 가 추가한 것은 `producer_*` 필드뿐이다.
2. **동일 바이너리의 IV 런은 이 차단 0회.** target_ion 만 3→2 로 바꿨을 때만 발화한다.

**계약의 모순(소스 실측, `a210_line_saturation_add`)**
```c
(tau_validity!=A208_VALID && tau_validity!=A208_EXACT_ZERO) || ...   // ← EXACT_ZERO 를 받아들인다
...
if(scaled_emission==0.0L) return 0;                                  // ← 방출 0 은 조용히 건너뛴다
...
int source_defined = material->effective_integrated_opacity != 0.0;
if(independent_capture && (!source_defined || !isfinite(source_function))){
    blocked("INDEPENDENT_SPROBE_UNDEFINED"); return -1;              // ← χ_eff==0 은 전체 차단
}
```
계약은 **입구에서 `A208_EXACT_ZERO` 를 유효로 받아들이고, 방출 0 은 행 단위로 건너뛰면서,
"방출>0 이고 유효 불투명도=0" 인 행에서는 A2-10 항 조립 전체를 멈춘다.**

그 상태는 물리적으로 정당하다 — Sobolev 물질에서 χ_eff ∝ (n_l g_u/g_l − n_u) 이므로
**population inversion 경계**(순 불투명도가 정확히 0, 순수 방출)에서 발생한다.
III 후보 51,807행에는 그런 선이 있고 IV 1,282행에는 없다.

## 4. 계급 — 같은 실수의 **세 번째** 자리

"정당하게 0" 과 "무효" 를 구별하지 못하는 결함이다.
- SH-GAMMA **NC3** 가 바로 이것을 겨냥해 만든 음성대조였다
  ("Ni·Co 존비 0 인 덱은 **통과**해야 한다 … 이 대조가 없으면 게이트가 0 을 무조건 막는
  잘못된 게이트가 된다")
- MC-EVT 의 OUT_OF_GRID exact-zero 논쟁도 같은 뿌리
- 이번이 **세 번째**다.

## 5. 부수 관찰 (경증)

`[A2-10][LINE-NET-BLOCKED] ... line_status=INVALID_INPUT line=262210` 에서
차단을 낸 것은 `a210_line_saturation_add` 인데 보고된 `line_status` 는 line_net 계열 값이다.
**보고가 원인을 오도할 수 있다** — 별도 경로의 실패를 line_net 상태로 표기한다. 기재만 한다.

## 6. 판정

- **S4 [BLOCKED]** — III 에서 S2·S2b·S3 판정 불가. `docs/VERDICT_DET_SPROD_IV_2026-08-18.md` 의
  `target_ion=3` 한정이 **사후에 정당화됐다**(감리 지적 E 반영이 옳았다).
- **신규 단 후보 A210-ZERO-OPACITY**: exact-zero 순 불투명도 행의 처분을 계약에 명문화한다.
  후보 처분 = 행 단위 건너뛰기 + **건너뛴 수를 보고**(NC3 정신), 또는 S_probe 를
  `UNAVAILABLE` 로 두고 그 행만 무효화. **전체 차단은 과잉이다.**
  ⚠수리 전 필수: 그 행이 정말 inversion 경계인지 **실측 확인**(현재는 소스 추론).
- **처분 원칙 유지**: 이 발견은 조용한 대장 기재이며, 오늘 고치지 않는다.
