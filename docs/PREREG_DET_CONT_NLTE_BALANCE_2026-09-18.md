# 사전등록 — DET-CONT-NLTE-BALANCE: 갇힌 연속체의 이온화 균형 (연속체 소스 = Milne η_bf · DR · 비열적 · 외부반복 10) (2026-09-18)

저자 = 운전자(Claude, 개정25). 상태 = **기안 / AUDIT_PENDING**. 실행 승인이 아니다. user 결정(2026-09-18 「권고안대로 진행해」): ITER 10 레인 · `NT_FULL` · 한 계약.
근거 판독 = `docs/READOUT_DEGENERATE_LEDGER_2026-09-18.md`(판독 C) + `docs/READOUT_LAMP_PREDICTION_2026-09-17.md` §1 ①. 상위 = `docs/VERDICT_DET_LTEUNMAP_EPS_2026-09-18.md`(§7 후보 ①). base = 판정런 **376251**(`…/lteunmap_eps_20260917/run5_h200`, 램프 소등 상태).

## 1. 계약 — 물리 계약 1개

**「bin 조립기(커밋 J 의 생산자)의 연속체 방출은 Kirchhoff-LTE(χ_abs·B(T_e)) 가 아니라 실제 인구의 Milne 방출 η_bf(`bf_get_eta`, ff 는 열적) 이고, 이온화 균형은 광이온화 + 충돌 + 비열적(heating/W) 대 RR + DR 로 완결된다.」**
판정런 1회(twin-T_e, 외부반복 **10**). 참이면 중간 셸(9–24)의 S 가 LTE(T_e)의 III 지배에서 벗어나 II 로 옮겨 CMFGEN(.71–.85) 쪽으로 간다(판독 C 의 퇴화 극한 r_deg(NT 1.0) = 0.34–1.7 ⟹ f_II .55–.75).
⚠**이 계약이 말하지 않는 것**: Si·Ca 의 정확한 착지(FUV 광이온화 지배 — mapped 선 S=B 열숲이 FUV 를 열화시키면 과재결합으로 튈 수 있음, §5 상한으로 기재)·스펙트럼·자유-T_e.

### Q-GOAL (기안자 자기 답 — 사전감사가 독립 재판정)
① 줄이는 칸: 어긋남 지도의 **중간 셸 9–24 S II/III**(현 .02–.10 대 CMFGEN .71–.85; 질량 최대 영역) — 폐합된 DET-LTEUNMAP-EPS 가 남긴 최대 어긋남. ② 물리 측정(이온 분율·반복별 수렴). 신규 계측 = 조립기 census 1 행/셸/패스(§4 G-SRC 소비). ③ 대안: NLTE 승격 IV·V(외곽), mapped 선 ε 채널 — 둘 다 중간 셸 S 를 건드리지 않음. ④ 건너뛰면 중간 셸의 LTE 고정이 모든 후속 CMFGEN 대조를 오염(판독 B 의 「올바른 T 의 LTE」 상태에 머묾).

### Q-LEAN (기안자 자기 답)
L1 물리량 = 셸별 이온 분율(반복별 `physics_DET_iter000k.ion.csv`·최종 `lumina_ion_pops.csv`), 창/문턱 빈 J/B. 신규 = 게이트 env 1 + 조립기 census 1 행/셸/패스 + selftest 1; 설정 키 3(DR·NT·NT_FULL)과 반복수 1 은 **기존 노브**. L2 게이트 3(G-ID·G-RUN·G-SRC) — 각각 실재 가능한 실패. L3 = 1 걸음(수리 1 + 판정런 1). L4 삭제 후보 없음 — 추가하지 않은 것: 새 판독기·새 상태 필드·둘째 런.

## 2. 기전 — 런 0회 확정분

- **채움 ①**: `cmfgen_assemble_impl` 의 `S_fixed = (chi_a·B + eta_ln)/chi_t`(동결 트리 `lumina_cmfgen.c` ~2425 행; `chi_a = chi_bf + chi_ff`) — bf-불투명 빈(100–912 Å 의 65–100 %)에서 J → B(T_e)(`planck_census`), 그래서 중간 셸 이온비 = Φ_LTE(판독 B·A; 376251 에서 E-5 = 0.81–0.96).
- **η_bf 는 이미 빌드된다**: `LUMINA_CMF_BF_MILNE=2`(런 env) → `bf->eta_bf`(Menzel 이탈 b_l 로 준안정 준위는 NLTE 형, 그 밖은 열적; LTE 극한 S_bf→B; `lumina_plasma.c` ~7949 행) → `bf_get_eta(bf,s,ν)`(~7618 행, ν 보간). 조립기는 EPAY 예산(미설정)에서만 참조하고 **S_fixed 에는 쓰지 않는다**.
- **퇴화**: J → S_bf = B·(r/Φ) 이면 광이온화-RR 이 항등 상쇄되어 r 은 **남는 채널**(충돌·비열적 대 RR+DR)로 정해진다. 장부(판독 C §2): S II→III sh12–24 r_deg(NT 1.0) 0.34–1.7 대 CMFGEN 0.18–0.72; NT 0.05 면 0.017–0.084. Si·Ca 는 FUV 광이온화 지배(CMFGEN r 5–68 / 10⁵).
- **채널 실물**: DR = `LUMINA_FROZENIN_DR=1` → `frozenin_alpha_rr` 안 `dr_lookup(Z, stage+1)`(ADAS/Badnell `DR_TABLE`, S·Si·Ca II→III 재결합 커버) · 비열적 = `LUMINA_ION_NT=1` → `parity_rate_se_ratio_checked` 의 `Gamma += gamma_nt_atom`(`gamma_deposition_compute_nonthermal`, `heating/W`, `LUMINA_NT_FULL=1` 이면 pref 1.0; 퇴적 = `deposition_cmfgen.csv`, `LUMINA_GAMMA_DEP=1` 런 env) · 충돌 = 코드 내장. 전부 strict env 등록됨.
- **반복수**: LTE 시드에서 소스만 바꾸면 반복당 r 이 α_RR/(α_RR+α_DR) = 0.40(S)·0.27(Si) 배 이동 ⟹ 2회는 구성상 미결(판독 C §4). 러너의 `STRICT_TWO_ITERATION_CONTRACT`(동결 러너 116 행)·`outer_iterations=2`(132 행)·`ION_POP_DUMP_ITER==1`(118 행) 검사는 스테이징 복사본에서 ITER=10 으로 편집한다(러너 해시 G-ID 에 기재).
- **미확정(이 계약 밖)**: mapped 선 S=B 열숲의 FUV 열화(Si·Ca 착지) · x_e 의존 η_ion(비열적 강도의 중간값) · CMFGEN toy06 의 비열적 처리 포함 여부(문헌 미확인 — 기재).

## 3. 수리안 — 변경집합

원칙: 게이트 env `LUMINA_CMF_CONT_NLTE_SOURCE`(기본 0 = byte-동일). ON 이면 `cmfgen_assemble_impl` 의 연속체 소스를 `S_fixed = (η_bf(s,ν) + chi_ff·B + eta_ln)/chi_t` 로 바꾼다(`η_bf = bf_get_eta(bf,s,ν)`; ff 는 열적 유지; `chi_a`·`chi_t`·`eta_ln`·EPAY·`kappa_dep` 항은 불변). **fail-closed**: 게이트 ON 인데 `bf==NULL`·`bf->eta_bf==NULL`·`LUMINA_CMF_BF_MILNE<1` 이거나 η_bf 가 전 빈 0 인 셸이 있으면 조립 실패(무음 B 폴백 금지). census: 패스·셸당 stderr 1 행 `[cmfgen_assemble][CONT-SRC] pass=<p> shell=<s> n_bf_bins=<n> S_bf_over_B_mean=<Σ η_bf dν / Σ chi_bf B dν> S_bf_over_B_min=<..> max=<..>`(bf 불투명 빈 기준). fine 경로는 건드리지 않는다(연속체 보간은 `chi_abs/chi_es` 만 받으므로 무영향). 소스 기준 = **동결 트리 복사본**(`/gpfs/kjhan/lumina/cmfgen_map/lteunmap_eps_20260917/source`, DET-LTEUNMAP-EPS 변경집합 포함 상태) — 그 위에 아래 파일만.
설정(코드 변경 아님, exports 키): `LUMINA_FROZENIN_DR=1`, `LUMINA_ION_NT=1`, `LUMINA_NT_FULL=1`, `LUMINA_PURE_CMFGEN_ITER=10`, 게이트 `LUMINA_CMF_CONT_NLTE_SOURCE=1`; 기존 `LUMINA_CMF_FINE_UNMAPPED_EPS=1` 유지.

### 변경집합 — CMF_CONT_NLTE_SOURCE (표)

| 경로 | 담당 | 내용 |
|---|---|---|
| `src/lumina_cmfgen.c` | Opus 4.8 | `cmfgen_assemble_impl` 연속체 소스 교체(게이트 ON 만)·fail-closed·census 행; 게이트 OFF byte-동일 |
| `src/env_universe.h` | Opus 4.8 | `LUMINA_CMF_CONT_NLTE_SOURCE` 등록 |
| `tests/cmf_cont_nlte_source_selftest.c` | Opus 4.8 | CPU selftest: 게이트 OFF S_fixed byte-동일 · ON 이면 S_fixed = (η_bf+χ_ff B+η_ln)/χ_t(1e-12) · LTE 극한(η_bf = χ_bf B) 에서 OFF 와 동일 · η_bf 부재/전 0 → rc≠0 · census 형식 |

### 변경집합 끝

## 4. 게이트 명부 (표)

| ID | 조건 | 음성대조 |
|---|---|---|
| G-ID | 376251 계보(`/gpfs/kjhan/lumina/cmfgen_map/lteunmap_eps_20260917/run5_h200/{RUN_FOOTER.txt,input/resolved_lumina.exports,input/binary.sha256}`) 대비 exports 차이가 정확히 5 키(`LUMINA_CMF_CONT_NLTE_SOURCE`·`LUMINA_FROZENIN_DR`·`LUMINA_ION_NT`·`LUMINA_NT_FULL`·`LUMINA_PURE_CMFGEN_ITER`; 프로필 종속 `LUMINA_CMF_FINE_MGPU_DEVICES` 허용), 소스 차이 = §3 변경집합 표의 3 파일(`src/lumina_cmfgen.c`·`src/env_universe.h`·`tests/cmf_cont_nlte_source_selftest.c`), 러너 편집 = ITER/덤프 검사·`--time`·`outer_iterations` 4 곳뿐(해시 기재) | 다른 키 주입 ⟹ FAIL |
| G-RUN | 외부반복 10 완주(`[R7][PHASE] event=R7_MATERIAL_PHASE_COMMITTED … iter=9`), `work/physics_comparison/physics_DET_iter0009.ion.csv`·`work/lumina_ion_pops.csv`·`work/cmfgen_final_publication.json` 실재, `[cmf_fine][EXACT-MULTIGPU-EPOCH] status=OK` ≥ 11 | 반복수 변조·덤프 결손 |
| G-SRC | stderr `[cmfgen_assemble][CONT-SRC]` 행이 매 패스마다 셸 0–49 전부 있고, 마지막 패스 셸 12·16·20 의 `S_bf_over_B_mean` 이 **1 이 아님**(\|log10\| > 0.05 — 소스가 실제로 NLTE 로 바뀌었음; base 376251 은 이 행이 없고 S_bf/B 는 판독 `opaque_source_check_run5` 의 0.8–1.7) | census 를 1.0 으로 위조 ⟹ FAIL; 행 결손 ⟹ FAIL |

### 게이트 명부 끝

## 5. 기대치 — 런에서만 얻는 수 (자릿수 예측 = 판독 C §2 퇴화 극한; 잣대를 base 에 먼저 적용: base 376251 은 E-1·E-2 를 넘지 못한다)

| ID | 양 | base(376251) | 예측 | **기대** |
|---|---|---|---|---|
| E-1 S 중간 | S f_II 셸 12·16·20 (`lumina_ion_pops.csv`) | .017 / .053 / .102 | deg(1.0) .75/.66/.55; CMFGEN .85/.80/.71 | **셋 다 ≥ 0.50** |
| E-2 Si 중간 | Si f_II 셸 12·20 | 0.000 / 0.000 | CMFGEN .014/.13; deg(1.0) .83/.65(과재결합 상한) | **둘 다 ≥ 0.01 그리고 ≤ 0.50**(하한 = 광이온화 회복, 상한 = FUV 열화 없음) |
| E-3 수렴 | S II→III 비 r(sh12) 의 반복별 `physics_DET_iter000k.ion.csv` 궤적: \|log10 r₉ − log10 r₈\| | (2회 런: 미측) | 기하 수렴비 0.4 ⟹ 9 회에 잔차 < 0.1 dex | **< 0.10 dex** |
| E-4 외곽 유지 | S f_III 셸 28·32·36 | .998/.994/.902 | 램프 소등 유지 | **셋 다 ≥ 0.70** |
| E-5 대조 | 셸 0–8 IGE IV 분율 \|Δ\| | 0.98 | LTE 극한 S_bf→B 라 소폭 | **\|Δ\| < 0.05** (기재용) |

### 분기 (표)

| 분기 | 조건 | 의미 |
|---|---|---|
| CB-X | G-ID/G-RUN/G-SRC 미충족 | 미시연 |
| CB-A | 게이트 전건 **and** E-1 **and** E-2 **and** E-3 **and** E-4 | **갇힌 연속체 균형 확정** — 중간 셸 S 가 CMFGEN 쪽으로, Si 광이온화 회복 |
| CB-B | 게이트 전건 **and** E-1 **and** E-3 **and not** E-2 | S 는 옮겼으나 Si 가 과재결합(FUV 열숲) 또는 미회복 ⟹ 다음 rung = mapped 선 ε/NLTE 소스 |
| CB-C | 게이트 전건 **and** E-3 **and not** E-1 | 수렴했는데 S 가 안 옮김 ⟹ 채널 부족(NT 강도·창 탈출) — 판정자가 명명 |
| CB-M | 그 외(특히 E-3 미수렴) | 판정자가 명명, 재분기 금지 |

### 분기 끝

문턱 근거: E-1 0.50 은 base(.02–.10)와 CMFGEN(.71–.85) 사이 **두 자릿수 위**의 중간; E-2 상한 0.50 은 퇴화 극한 과재결합(.65–.83)을 실패로 읽는 선; E-3 0.1 dex 는 수렴비 0.4 의 9 회 잔차(0.4⁹ ≈ 3e-4 dex·초기 2 dex ≈ 0.03 dex)의 3 배 여유.

## 6. 자원 · 실행

GPU **1 런**, 프로필 h200x1 우선(대기 길면 h100-a100x2 헤지, 먼저 시작하는 쪽만) — 376251 실측 5 h/2 반복 ⟹ ITER 10 ≈ **18–20 h**, `--time=30:00:00`, 상한 **40 GPU-hour**. 새 RUN 루트(`…/lteunmap_eps_20260917/` 형제 루트 `cont_nlte_20260918/`), 입력 = 376251 입력 복사 + 새 바이너리 + exports 5 키. 오프라인 ≤ 1 CPU-hour. 봉인 base 는 읽기만.
순서(개정16 ②): Opus 4.8 구현 → selftest → 운전자 코드 간단 감사(Q0) → 빌드(CPU) → 스테이징(러너 4 곳 편집·해시 기재) → 제출.

## 7. 분장 장부 (개정25)

| 단계 | 규약 | 실제 | 비고 |
|---|---|---|---|
| 기안 | 운전자 | 운전자(본 문서) | 자기 기안 |
| 사전감사(Q-GOAL·Q-LEAN·물리) | (미배정) → 운전자 발주 | 미집행 → Opus 4.8 fresh 발주 | GPU 20 h 단 |
| 구현 + selftest | Opus 4.8 fresh | 미집행 | `model` 생략 |
| 코드 간단 감사 | 운전자 | 미집행 | |
| 빌드·스테이징·제출·판독·판정 | 운전자 | 미집행 | |
| 종료감사 | Opus 4.8 fresh | 미집행 | 생략하지 않음 |

## 8. 주장 상한
「갇힌 연속체 소스를 NLTE 로 두고 채널을 완결하면 중간 셸 S 가 CMFGEN 쪽으로 옮기는가」와 「Si 가 광이온화 지배로 회복되는가(상하한)」까지. 비열적 강도의 정확값(0.05–1.0 사이)·x_e 의존 η_ion·mapped 선 열숲·자유-T_e·스펙트럼은 말하지 않는다. twin 핀 조건부.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 — CMF_CONT_NLTE_SOURCE (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "(?:src|tests)/[A-Za-z0-9_/.]+\\.(?:c|h)",
    "symbol": "CMF_CONT_NLTE_SOURCE",
    "roots": ["src", "tests"],
    "expected_extra": ["src/lumina_cmfgen.c", "src/env_universe.h", "tests/cmf_cont_nlte_source_selftest.c"]
  },
  "branches": {
    "regimes": [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
    "metrics": {"count": "len(v)", "gate_fail": "sum(1 for x in v if x < 0.5)", "balance": "sum(1 for x in v if 0.5 <= x < 1.5)", "si_fail": "sum(1 for x in v if 1.5 <= x < 2.5)", "s_stuck": "sum(1 for x in v if 2.5 <= x < 3.5)", "other": "sum(1 for x in v if x >= 3.5)"},
    "rules": [
      {"name": "CB-X", "predicate": "count == 0 or gate_fail > 0"},
      {"name": "CB-A", "predicate": "count > 0 and gate_fail == 0 and si_fail == 0 and s_stuck == 0 and other == 0"},
      {"name": "CB-B", "predicate": "count > 0 and gate_fail == 0 and si_fail > 0 and s_stuck == 0 and other == 0"},
      {"name": "CB-C", "predicate": "count > 0 and gate_fail == 0 and s_stuck > 0 and other == 0"},
      {"name": "CB-M", "predicate": "count > 0 and gate_fail == 0 and other > 0"}
    ],
    "adversarial_fixtures": [
      {"name": "empty", "mix": []},
      {"name": "gate-fail", "mix": [[0.0, 1], [1.0, 2]]},
      {"name": "balance", "mix": [[1.0, 3]]},
      {"name": "si-fail", "mix": [[1.0, 1], [2.0, 1]]},
      {"name": "s-stuck", "mix": [[3.0, 1]]},
      {"name": "other", "mix": [[4.0, 1]]}
    ],
    "residual": "CB-X"
  },
  "references": [
    {"path": "docs/READOUT_DEGENERATE_LEDGER_2026-09-18.md"},
    {"path": "docs/READOUT_LAMP_PREDICTION_2026-09-17.md"},
    {"path": "docs/VERDICT_DET_LTEUNMAP_EPS_2026-09-18.md"},
    {"path": "CLAUDE.md"},
    {"path": "src/lumina_cmfgen.c", "flags_existing": ["cmfgen_assemble_impl", "bf_get_eta", "S_fixed", "LUMINA_CMF_EPAY"]},
    {"path": "src/lumina_plasma.c", "flags_existing": ["bf_get_eta", "LUMINA_CMF_BF_MILNE", "LUMINA_FROZENIN_DR", "LUMINA_ION_NT", "LUMINA_NT_FULL", "dr_lookup", "gamma_deposition_compute_nonthermal"]},
    {"path": "src/env_universe.h", "flags_existing": ["LUMINA_FROZENIN_DR", "LUMINA_ION_NT", "LUMINA_NT_FULL"]}
  ]
}
```
