# Codex 운전 기간(2026-09-10~17) 정리 — 개정25 운전자 인수 (2026-09-17)

저자 = 운전자(Claude). **판정이 아니라 기재·재현이다.** user 지시 「A 부터 진행해」.

## 1. 미기재 사실의 기재

| 작업 | 실물 | 기재 여부(Codex) |
|---|---|---|
| **337602** `cmfgen_twin_map` h100/syn08, 09-09 23:06→10 00:50 | **FAILED** exit 70 — `[A2-10][BLOCKED] RADEQ_FIXED_T_BUNDLE_BUILD_FAILED te_lane=FIXED_T rc=3` → `[R7][FATAL] iter=0 rc=4` → `CMFGEN_TWIN_MAP_RUN_FAIL MODEL_EXIT_NONZERO rc=1`. WorkDir 가 저장소 루트(GPFS 규약 위반) | 09-17 인수인계서 **없음**; `BUNDLE_LINE_NET_SHELL_2026-09-10.md` 에만 간접 언급 |
| **338745** 재런 a100/syn101, 09-11 00:25, 3h18m | **COMPLETED** rc 0 → 지도 산출 `evid_writer_empty_gpu_20260910/result/` | 묶음 문서에 기재됨 |

## 2. 미커밋 5,491행의 처분 = **가지 스냅샷**

`codex-agents-era-20260910-17` 가지에 커밋 `2b67e57` 로 **전건 보존**(21 파일 + 09-10~17 묶음 문서 50여 건).
메인 `thenmc-macroatom-fluorescence` 추적 변경 0. 어느 것도 개정25 검수·판정을 받지 않았으며,
필요한 부분(rev3/rev4 판독기 등)은 **계약 폐합과 함께** 메인으로 되가져온다.
⚠메인의 `docs/PREREG_CMFGEN_TWIN_MAP_2026-09-08.md` 는 마지막 커밋 상태이며 Codex 의 미커밋 수정분은 가지에 있다.

## 3. 운전자 독립 재현 — 생산 코드 미사용(csv+Decimal 직접 산술), grammar-debug

도구 = `/gpfs/kjhan/lumina/cmfgen_map/repro_claude_20260917/{repro_ez.py,repro_ee2.py}` (해시 `TOOLS.sha256`), 출력 `*.stdout`.
입력 = `run_writer_empty_gpu_20260910/work/lumina_plasma_state.csv`(twin) · 봉인 base `work/lumina_plasma_state.csv` ·
`evid_…/inputs/anchors.csv`(중점 최근접 표본) · 덱 `density/abundances/atom_masses/geometry.csv` · AMU=1.660539066e-24(소스) ·
비교 질량 Fe/Co/Ni = **56** · 질량 가중 = ρ·(4/3)π(r_out³−r_in³).

### 3-1. 평균전하 지도 `e_z_operands` — **전건 자릿수 일치**

| | base(전 셸 10,020 K) | **twin(쌍둥이 T_e)** |
|---|---|---|
| 질량가중 `M_Z` = ⟨\|Δlog(n_e/n_atom)\|⟩ | **0.0970760679 dex** | **0.0309230148 dex** |
| 부호 있는 `S_Z` | −0.0570883109 | +0.0151862685 |
| 부호 전환 | 2 | 3 |

질량 규약 대조 max|dex| = 0.053991(일치) · 내곽 5셸 질량분율 0.3893(Fable 09-09 의 38.9% 와 일치).
⟹ **T 교란을 걷어내자 질량가중 평균전하 차이가 3.1배 줄었다.** 관측 등급; 정확도 인증 아님.

### 3-2. 원소별 `E_e`(09-14, NLTE 권위, BASE 질량가중 mean|Δf| over q=1,2) — **6/6 자릿수 일치**

| Fe | Ca | Ni | Co | Si | **S** |
|---|---|---|---|---|---|
| 0.01847 | 0.02095 | 0.02330 | 0.03385 | 0.05241 | **0.39481** |

★**황만 한 자릿수 크다**(분율 단위 — S II/III 의 39.5 %p 가 잘못된 단계에 있다). Codex 가 09-11~17 을 S/Si 원인에 쓴 근거가 이것이다.
⚠summary 자신이 `metric_classification = E_CRITERION_NOT_SUPPORTED`, `scope = metric-only comparison; never a standalone physics PASS` 로 표기.
Saha 권위로 계산하면 각 0.1 % 안에서 다르다(S 0.39537).

### 3-3. 재현이 잡은 결함 — «기재로 족함»

지도의 `CONDITION_MISMATCH/T_NOT_TWIN` **14행(셸 14–17,19,20,22–27,29,31)은 거짓 불일치**다:
등록 `9696.1` 대 공개 `9696.1000000000004` — **마지막 ulp 차이**를 조건 불일치로 표기했다.
실제로는 **50셸 전부 프로파일을 소비**했다. `cell_count=50` 이라 M_Z·S_Z 에는 영향 없음. 이것은
09-08 코드 검수 2회차 R7(표기 차이 거짓 거부)과 같은 계급이 rev3/rev4 판독기에서 재발한 것이다.

## 4. 승계하지 않는 것

Codex 의 「운전자 자체 종료감사 PASS」·Opus5 자문·도구 없는 Fable CLI 감사. 위 수치는 **운전자 재현 사실**로만 승계한다.
rate ledger 묶음(09-17)은 마지막 런이 **ledger CSV 파일 자체가 없이**(RC=124) 끝났고, 그 코드는 가지에만 있다.

## 5. 남은 미결(대기 큐)

`LUMINA_RADEQ_SIMUL` 배선 보완(가지) · `/tmp/lumina_rate_ledger.*` 10개 임시 루트 처분 · 09-10 line-net §5 대체 규칙 · T_NOT_TWIN ulp 결함 수리(rev3 판독기 되가져올 때).
