# 사전등록 — DET-EUV-JSUBST: 중간 셸 이온화의 장 귀속 반증 시험 (2026-09-17)

저자 = 운전자(Claude, 개정25). 상태 = **기안 / AUDIT_PENDING**. 실행 승인이 아니다.
근거 판독 = `docs/READOUT_S_IONIZATION_EUV_LTE_2026-09-17.md`(+부록 A). 상위 = `docs/CLOSEOUT_CODEX_PERIOD_2026-09-17.md`.

## 1. 계약 — 물리 계약 1개

**「DET/FIXED_T 레인의 중간 셸(9–36) 이온화 상태는 canonical view 의 EUV J 가 정한다.」**
반증 시험: EUV J 를 **CMFGEN J 표**(`data/cmfgen_jtable_toy06_19p48d.bin`, 7월 jnu4)로 치환한 **단일 변수 판정런 1회**.
참이면 S II/III(및 Si·Ca 의 해당 단계)가 **CMFGEN 쪽으로** 움직인다. 거짓이면 이온화는 장이 아니라 율·자료·경로가 정한다.
⚠이 계약은 **CMFGEN 이 옳다**고도, **치환이 수리**라고도 말하지 않는다 — **귀속 시험**이다.

### Q-GOAL (기안자 자기 답 — 사전감사가 독립 재판정)
① 줄이는 칸: CMFGEN 어긋남 지도의 **S II/III(sh9–24) 39.5 %p**, Si III/IV·Ca III/IV(외곽)의 **원인 귀속**. 크기를 줄이는 것이 아니라 **어느 축(장/율/자료)인지를 확정**한다. ② 물리 측정(이온 분율 응답)이다 — 계측 신설은 overlay 1개뿐. ③ 대안: D-C-2/D-D 는 이중 권위 수리인데 판독이 「Saha=NLTE 동일 답」임을 보였으므로 지금 표적이 아니다. 「장 자체 수리」는 귀속 전에는 수리안이 없다. ④ 건너뛰면 이후 모든 이온화 수리가 **원인 미확정 위에** 서게 된다.

### Q-LEAN (기안자 자기 답)
L1 물리량 = 셸별 이온 분율·`n_e`(기존 지도 판독기 재사용). 신규 계측 = **commit 덤프 1개**(기존 `LUMINA_RADFIELD_COMMIT_DUMP`) — 치환된 장을 view 가 실제로 봤는지에 소비. L2 게이트 3(§4) — 각각 실재 가능한 실패(치환 미적용·identity 이탈·미완주)만. L3 = 귀속 1걸음. L4 삭제 후보: 없음 — 대신 **추가하지 않은 것**을 명시: 새 판독기·새 상태 필드·새 판정기 0.

## 2. 기전 — 런 0회 확정분과 미확정분

**확정**(판독 §3·부록 A): iter1 의 EUV J(150–532 Å, sh12–36) = **B_ν(T_e) × bin-center 상수**, iter0 seed(B(T_inner)) 와 **같은 상수** ⟹ Planck 채움. Γ_phot = `nlte_bf_gamma_canonical` → `bf_rate_gamma_legacy_grid(&nlte->radfield_view)`. 세부균형 ⟹ 중간 셸 LTE(T_e). 실측 S II/III = LTE(T_e) ± 1 %.
**확정**: `LUMINA_GPH_JTABLE`(7월)은 `radeq_simul_all` 의 **레거시 루프(:12672)에만** 배선 — view 경로 참조 0 ⟹ 이 레인에서 **죽은 노브**. `LUMINA_RADEQ_SIMUL` 배선 보완은 Codex 가지에만 있다.
**미확정**(이 계약이 다루지 않음): 채움의 소스 행(`cmfgen_commit_jnu` 이전 `cs->J` 의 EUV 구성; 후보 `CMF_LP_F_SL_FALLBACK`·`η_dep=κB`) · 외곽 sh28–36 III→IV 의 LTE 3,600배 과이온화(공개 장으로 설명 불가 — **별개 계약**) · `physics_DET` 덤프의 χ/η 세대 동일성.

## 3. 수리안(시험 장치) — 변경집합

`radiation_field_commit` 은 임의 격자의 `source_J_nu`·`source_validity` 를 받으며 provenance `CMFGEN_REPLAY` 가 이미 있다.
⟹ **`cmfgen_commit_jnu`(`src/lumina_cmfgen.c:3476`) 의 commit 직전에 overlay**: 새 env `LUMINA_RADFIELD_EUV_JTABLE=<path>` 가 있으면
표를 읽어 **표값 > 0 인 (셸,빈)만** `source_J_nu` 를 치환(7월 GPH_JTABLE 과 같은 skip-0 의미론), validity 는 유지, `producer` 문자열에 `+EUV_JTABLE` 표기.
표 격자(1.5e14–3e16 Hz, 1000 log 빈) → commit 격자 사상은 **빈 중심 최근접**(보간 금지 — 계약 원칙). env 부재 ⟹ **byte-동일**(음성대조 자격).
**금지**: 이온화 solve·율·원자자료·NLTE·T_e 어느 것도 건드리지 않는다. 채움 소스 수리도 하지 않는다(귀속 전 수리 금지).

### 변경집합 — RADFIELD_EUV_JTABLE (표)

| 경로 | 담당 | 내용 |
|---|---|---|
| `src/lumina_cmfgen.c` | Opus 4.8 | `cmfgen_commit_jnu` 직전 overlay(env 게이트, skip-0, 최근접 빈, producer 표기) |
| `src/env_universe.h` | Opus 4.8 | `LUMINA_RADFIELD_EUV_JTABLE` 등록(strict env 수용) |
| `tests/radfield_euv_jtable_selftest.c` | Opus 4.8 | CPU selftest: env 부재 byte-동일 · 표 적용 시 commit 덤프에 표값 · 0 빈 skip · 격자 사상 경계 |

### 변경집합 끝

## 4. 게이트 명부 (표)

| ID | 조건 | 음성대조 |
|---|---|---|
| G-OVL | `LUMINA_RADFIELD_COMMIT_DUMP` 덤프의 iter1 J(셸 12–36, 150–532 Å 빈) = `data/cmfgen_jtable_toy06_19p48d.bin` 의 §3 최근접-빈 사상값, 상대차 ≤ 1e-12 | 전값 0 인 표 파일 ⟹ env 부재 런과 덤프 byte-동일; 존재하지 않는 표 경로 ⟹ 시작 전 fail-closed |
| G-ID | 338745 계보 대비 차이가 exports 에서는 `LUMINA_RADFIELD_EUV_JTABLE` 키 하나, source 에서는 §3 변경집합 표의 행 전부(`src/lumina_cmfgen.c`·`src/env_universe.h`·`tests/radfield_euv_jtable_selftest.c`)뿐임을 `identity.json` 으로 확인 | 이전 계보의 exports 해시를 identity 에 주입 ⟹ FAIL |
| G-RUN | 2 외부반복 완주, `R7_MATERIAL_PHASE_COMMITTED iter=1`, `lumina_ion_pops.csv`·`physics_DET_iter0001.*` 실재 | 반복수 변조·덤프 결손 주입 |

### 게이트 명부 끝

## 5. 기대치 — 런에서만 얻는 수

**E-1(구성상 확인, 기대치 아님)**: G-OVL 통과 = 치환이 view 에 도달. **E-2(물리, 런 전 미지)**: 셸 12·20 의 **S II 분율** `f_II`.
현 base = **0.004 / 0.000**, CMFGEN = **0.845 / 0.713**, LTE(T_e) = 0.010 / 0.077.
잣대를 base 에 먼저 적용: base 의 `f_II(sh12)=0.004` 는 아래 어느 분기 문턱도 넘지 않는다(E-B 영역) — 잣대 자기값 위.
보조 관측(분기 미참여, 기재): sh30 S IV(현 0.805) — 외곽 기전이 별개라면 **거의 불변**이어야 한다(부록 A-3 의 예측; 변하면 A-3 정정).

### 분기 (표)

| 분기 | 조건 | 의미 |
|---|---|---|
| JS-X | G-OVL/G-ID/G-RUN 중 미충족 | 미시연 — 원인 명명 |
| JS-A | 게이트 전건 충족 **and** `f_II(sh12) ≥ 0.10` **and** `f_II(sh20) ≥ 0.10` | **장 귀속 확정**: EUV J 가 중간 셸 이온화를 정한다 ⟹ 다음 = 채움 소스 수리 계약 |
| JS-B | 게이트 전건 충족 **and** `f_II(sh12) < 0.02` **and** `f_II(sh20) < 0.02` | **장 아님**: 율/자료/경로 ⟹ 다음 = Γ·α 장부 계약(Codex ledger 표적의 정당화) |
| JS-M | 그 외 | 혼합/부분 — 판정자가 명명, 재분기 금지 |

### 분기 끝

문턱 0.10/0.02 근거: base 0.004 와 CMFGEN 0.85 사이 **한 자릿수 이상** 벌어진 곳에 둔다(잡음·표본 위치 인공물 ≪ 0.02).

## 6. 자원 · 실행

GPU **1 런**, 파티션 h200→h100→a100(338745 실적 a100 3h18m) — `--time=05:00:00`, 상한 **10 GPU-hour**. 새 RUN/EVID 루트(봉인 밖). runner = 338745 의 `job.slurm` 계보 + env 1개(스테이저 기존 R-ACQ/twin 모드 재사용, 상한 집행 유지). 오프라인 ≤ 1 CPU-hour(selftest·판독).

## 7. 분장 장부 (개정25)

| 단계 | 규약 | 실제 | 비고 |
|---|---|---|---|
| 기안 | 운전자 | 운전자(본 문서) | **자기 기안** |
| 사전감사(Q-GOAL·Q-LEAN) | 운전자 | 미집행 | ⚠자기 감사 — user 지시 시 Opus 4.8 fresh 로 대체 가능 |
| 구현 + selftest | Opus 4.8 fresh | 미집행 | `model` 생략 발주, 첫 줄 모델 ID 자기보고 |
| 코드 간단 감사 | 운전자 | 미집행 | 구현자 ≠ 검수자 |
| 스테이징·제출·판독·대장·커밋 | 운전자 | 미집행 | |
| 판정 | 운전자 | 미집행 | |
| **종료감사** | **Opus 4.8 fresh** | 미집행 | GPU 지출 단 — **생략하지 않음**(CLAUDE.md 개정25 구조적 귀결) |

## 8. 주장 상한
「중간 셸 이온화의 **장 귀속** 여부」까지. 채움의 소스 행·외곽 기전·CMFGEN 정확도·스펙트럼 영향은 말하지 않는다.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 — RADFIELD_EUV_JTABLE (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "(?:src|tests)/[A-Za-z0-9_/.]+\\.(?:c|h)",
    "symbol": "RADFIELD_EUV_JTABLE",
    "roots": ["src", "tests"],
    "expected_extra": ["src/lumina_cmfgen.c", "src/env_universe.h", "tests/radfield_euv_jtable_selftest.c"]
  },
  "branches": {
    "regimes": [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
    "metrics": {"count": "len(v)", "gate_fail": "sum(1 for x in v if x < 0.5)", "field": "sum(1 for x in v if 0.5 <= x < 1.5)", "notfield": "sum(1 for x in v if 1.5 <= x < 2.5)", "mixed": "sum(1 for x in v if x >= 2.5)"},
    "rules": [
      {"name": "JS-X", "predicate": "count == 0 or gate_fail > 0"},
      {"name": "JS-A", "predicate": "count > 0 and gate_fail == 0 and notfield == 0 and mixed == 0"},
      {"name": "JS-B", "predicate": "count > 0 and gate_fail == 0 and field == 0 and mixed == 0"},
      {"name": "JS-M", "predicate": "count > 0 and gate_fail == 0 and not (notfield == 0 and mixed == 0) and not (field == 0 and mixed == 0)"}
    ],
    "adversarial_fixtures": [
      {"name": "empty", "mix": []},
      {"name": "gate-fail", "mix": [[0.0, 1], [1.0, 2]]},
      {"name": "field", "mix": [[1.0, 2]]},
      {"name": "not-field", "mix": [[2.0, 2]]},
      {"name": "mixed", "mix": [[1.0, 1], [2.0, 1]]},
      {"name": "mixed-explicit", "mix": [[3.0, 1]]}
    ],
    "residual": "JS-X"
  },
  "references": [
    {"path": "docs/READOUT_S_IONIZATION_EUV_LTE_2026-09-17.md"},
    {"path": "docs/CLOSEOUT_CODEX_PERIOD_2026-09-17.md"},
    {"path": "CLAUDE.md"},
    {"path": "data/cmfgen_jtable_toy06_19p48d.bin"},
    {"path": "scripts/build_cmfgen_jtable.py"},
    {"path": "src/lumina_cmfgen.c", "flags_existing": ["cmfgen_commit_jnu", "radiation_field_commit"]},
    {"path": "src/radiation_field.h", "flags_existing": ["RADIATION_FIELD_PROVENANCE_CMFGEN_REPLAY", "source_J_nu"]},
    {"path": "src/lumina_plasma.c", "flags_existing": ["nlte_bf_gamma_canonical", "LUMINA_GPH_JTABLE", "radeq_simul_all"]},
    {"path": "src/radiation_field.c", "flags_existing": ["LUMINA_RADFIELD_COMMIT_DUMP"]}
  ]
}
```

---

## 집행 기록 1 — ★런 전 오프라인 반증: 계약 **BLOCKED_BY_OFFLINE_PREDICTION** (2026-09-17, 운전자)

「수리안 오프라인 검증」(3요건 ②)으로 치환 결과를 런 없이 예측했다(`repro_claude_20260917/predict_subst.py`, σ∝ν⁻³, 세부균형):

| sh | T_e | Γ_fill/Γ_B(T_e) | Γ_CMF/Γ_B | f_II 현재 | f_II LTE | **f_II 치환 예측** | f_II CMFGEN |
|---|---|---|---|---|---|---|---|
| 9 | 10114 | **1.00** | 2.0 | 0.033 | 0.022 | 0.011 | 0.505 |
| 12 | 10138 | **1.00** | 3.9 | 0.004 | 0.010 | **0.0027** | 0.845 |
| 20 | 8849 | **1.00** | 160 | 0.000 | 0.077 | **0.0005** | 0.713 |

① `Γ_fill/Γ_B = 1.00` — **채움 가설(J = B(T_e)) 확정**, 3자리. ② CMFGEN 장은 B(T_e) 보다 **강하다** ⟹ 치환하면 **더** 이온화.
⟹ **JS-A(f_II ≥ 0.10) 는 구성상 도달 불가**이며 런은 JS-B 를 자동으로 낸다. 「구성상 참을 예측으로 등록하지 마라」 계급 — **GPU 발주 금지**.
③ 물리적 함의: 세부균형 아래 「장 ≥ B(T_e) 인데 LTE 보다 덜 이온화」는 **불가능**하다. CMFGEN 의 S II 0.85 는 Lumina 가
쓰는 **α(T_e)·σ_bf 로는 어떤 장으로도 재현되지 않는다** ⟹ 차이는 장이 아니라 **율/자료**(α_RR+α_DR, σ_bf(S II) 문턱 모양·크기,
또는 J 표 자체의 부적격 — 7월 자체런 J, 대장 「이온화 오라클 부적격」).
**처분**: 이 계약은 실행하지 않는다(사전감사 결과와 무관). 후속 = **율 장부 계약**(오프라인): S II→III 의 Γ(실제 σ_bf 표)·
α_RR·α_DR 을 Lumina 가 실제로 쓰는 값으로 재구성하고, CMFGEN 의 f_II 를 내려면 α/Γ 가 몇 배여야 하는지 산출.

## 집행 기록 2 — 사전감사 (Opus 4.8 fresh, `model` 생략 → 자기보고 `Opus 4.8 / claude-opus-4-8`) (2026-09-17)

**결과**: `Q_GOAL: 부분부합` · `Q_LEAN: 적정` · **`PRE_AUDIT: 통과 불가` · `BLOCKING: 2`**. 기안자(운전자)와 **분리된 fresh 컨텍스트**.

| | 내용 |
|---|---|
| **BLOCKING-1** | 분기 의미가 세부균형과 어긋남 — CMFGEN 표 J 가 B(T_e) 보다 크므로 치환은 f_II 를 **base 이하로** 내린다 ⟹ **JS-B 가 구조적으로 강제**되고 JS-A 는 도달 불가. 「잣대를 base 에 먼저 적용」이 **크기만 보고 부호를 안 봤다.** 장이 진짜 원인이어도 정합성 없는 더 강한 장을 넣으면 거짓 음성. 올바른 조작화 = 「committed field 에 이온화가 **반응**하는가(양방향 유의 변화) 대 무반응(bypass)」 |
| **BLOCKING-2** | 계약 문면 「**EUV** J」 대 장치 「표값>0 인 빈 **전 대역**(2 µm–100 Å) 치환」 불일치 — EUV/FUV/광학 bf 의 분리 귀속 불가, 다른 이온 bf 변화로 `n_e` 되먹임 |

**독립 확증(소스 직독)**: overlay → `radiation_field_commit` → `:3797 radiation_field_read_view`("the only replay-lane view refresh point") → `radfield_view` → `nlte_bf_gamma_canonical` — **치환은 소비 경로에 도달한다**; `LUMINA_GPH_JTABLE` 은 `radeq_simul_all` 의 static local — **죽은 노브 확인**; 신규 env 실재 부재·입력 파일 실재·`INPUT_PENDING` 부정직 없음·봉인/자원 적정.
**«기재로 족함»**: `cs` 가 `const CMFGENState*` 이므로 overlay 는 가변 복사본으로(캐스팅 변조 금지) · `cmfgen_write_jnu` 의 `nlte->J_nu` memcpy 는 미치환으로 남음(DET 무해, 단일 소비자 명시) · 주입 표는 대장이 「이온화 오라클 부적격」으로 기록한 자체런 J — 「CMFGEN 쪽으로」 문면은 과대 · 음성대조는 장치만 검증하고 분기 오조작화는 못 잡음.
**Q-GOAL ③ 대안**: 판독 §7·A-4 의 **채움 소스(`CMF_LP_F_SL_FALLBACK`·`η_dep=κB`) 비-LTE 시험**이 최종목적에 더 직접적.
**감사 자기 한계**: 「로드된 CLAUDE.md 에 개정25 본문이 없다」 — 운전자 확인 결과 파일에는 실재(하네스가 서브에이전트에 주입하는 CLAUDE.md 가 **세션 시작 시점 판**이라 개정25 이전 본). 담당표 재현 불가는 그 때문이며 판정 원칙은 로드된 규약으로 충족.

**운전자 처분**: 집행 기록 1(런 전 반증)과 **독립 수렴** — 계약 유지 불가. 실행하지 않는다. BLOCKING-1 의 「응답 대 무응답」 재조작화는 **런이 필요**하고 정보량이 낮아(판독 §5 가 이미 열적 장을 확립) 채택하지 않는다. 후속 = 오프라인 「함의 J 장부」 또는 채움 소스 비-LTE 시험 — **user 결정**.

