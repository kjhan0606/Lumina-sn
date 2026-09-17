# 사전등록 — DET-LTEUNMAP-EPS: NLTE 밖 이온 선의 LTE 열적 방출 폐기(두준위 ε_phys 소스) (2026-09-17)

저자 = 운전자(Claude, 개정25). 상태 = **기안 / AUDIT_PENDING**. 실행 승인이 아니다.
근거 판독 = `docs/READOUT_IMPLIED_J_LEDGER_2026-09-17.md` + `docs/READOUT_LAMP_PREDICTION_2026-09-17.md`. 상위 = `docs/CLOSEOUT_CODEX_PERIOD_2026-09-17.md`.

## 1. 계약 — 물리 계약 1개

**「`cmf_fine` 조립기는 NLTE 밖 이온(LTE-unmapped)의 선을 LTE 상준위 채움으로 열적 방출하지 않는다 — 두준위 원자: 열적 분율 ε_phys(선,셸) 만 B(T_e) 로 방출하고, (1−ε_phys) 는 산란 불투명도로 접는다.」**
판정런 1회(twin-T_e, 343204 계보와 env 1 키 차이). 참이면 외곽 창 빈 스파이크(J/B 10³–10⁶)가 꺼지고 셸 25–40 의 S·Si 가 IV 에서 III 로 복귀한다.
⚠이 계약은 **중간 셸(9–24)의 S II 가 CMFGEN 으로 복귀한다고 말하지 않는다**(그것은 연속체 Kirchhoff-LTE 채움 ①, 별개 rung — 판독 A §3). **NLTE 밖 이온을 NLTE 로 푸는 것(CMFGEN 발자국)이 정도(正道)**이나 램프 이온이 S V·Ca V 라 기존 STAGE4 표로 닿지 않는다; 이 계약은 「우리가 풀지 않는 이온에 대해 코드가 이미 가진 ε 기전을 parity 분기에도 적용」하는 것으로, 새 물리를 넣지 않는다.

### Q-GOAL (기안자 자기 답 — 사전감사가 독립 재판정)
① 줄이는 칸: 어긋남 지도의 **외곽 25–40 S/Si III→IV 과이온화**(S IV 0.5–0.99 → CMFGEN 0.03–0.15)와 중간 셸 σ-가중 Γ 초과(3–55×)의 램프 성분. 크기 = 판독 A §2 표. ② 물리 측정(이온 분율·창 빈 J 응답). 신규 계측 = stderr census 1 행/셸(§4 G-EPS 가 소비). ③ 대안 = ① 연속체 채움 수리 — 질량이 큰 중간 셸을 겨냥하나 퇴화(판독 A §3)로 단독 예측이 불가해 **뒤**에 둔다; NLTE 승격(S/Ca IV·V) — 정도이나 표·슬롯·GPU 메모리 변경이 커 다음 단. ④ 건너뛰면 외곽 과이온화의 귀속(램프 대 다른 구동)이 미확정으로 남고, ① 수리 시 램프가 결과를 오염한다.

### Q-LEAN (기안자 자기 답)
L1 물리량 = 창 빈 J/B·σ-가중 Γ/Γ(B)·셸별 단계 분율(기존 판독 도구 재사용: `window_leak.py`·`gamma_locate.py`·`ion_share.py`·지도 판독기). 신규 = env 1·census 1 행/셸·selftest 1. L2 게이트 3(§4) — 각각 실재 가능한 실패(정체성 이탈·미완주·ε 미적용/무음 열화)만. L3 = 1 걸음(수리 1 + 판정런 1). L4 삭제 후보: 없음 — 추가하지 않은 것: 새 판독기·새 상태 필드·새 판정기·두 번째 GPU 런(음성대조는 CPU selftest).

## 2. 기전 — 런 0회 확정분

- 이온비 = σ-가중 J(재적분 0.5–3 % 재현; `Γ(B)/α=Φ` 0.1 %). 초과의 45–85 % 는 창 빈(557.7 Ca V·696.7/922.5 S V …), 방출력 96–99 % 가 셸 41–49(T_e 12–24 kK, twin 핀).
- 방출 코드: `cmf_fine_line_material` parity 분기(`src/lumina_cmfgen.c`, `cmf_fine_line_material`, 4498–4540 행) η₀ = n_upper·A_ul·hν/4π(`line_net_sobolev_material`, `src/line_net_rate.c` 106 행); n_upper = `upper_population_cache` ← `lumina_line_upper_population_fill_for_tau`(`src/lumina_plasma.c` 9019 행) ← `a209_upper_population_for_tau` 의 `lte_level_density`(**`used_nlte==0`** 인 선) ⟹ S_l = B(T_e) 정확히(실측 η/χ = 1.02 B; χ_bb 는 Saha+Boltzmann+Sobolev 로 3자리 재현).
- parity 분기는 `line_eps`/`radeq_line_eps_phys` 를 **보지 않는다**(legacy 분기만 `emit = line_eps<1 ? line_eps·B : source`). 런의 `LUMINA_CMFGEN_LINE_EPS_PHYS=1` 은 bin 조립기용이다.
- NLTE 표적 31 슬롯 = 15 원소 × ion II·III(런 배너 `Z=.. ion=1/2`); 램프 이온 S V·Ca V 는 어떤 승격 표에도 없다.
- **미확정(이 계약 밖)**: ① 연속체 `S_fixed=(χ_abs·B+η)/χ_tot`(`:5686`) 의 LTE 채움 · 퇴화-파괴 채널(DR·비열적) · CMF 선숲 흡수의 정량(정적 예측기의 sh≤24 과대).

## 3. 수리안 — 변경집합

원칙: **parity 분기에서 `used_nlte==0` 인 선에 한해** ε_phys 적용. NLTE 사상 선·연속체·이온화 solve·율·원자자료·T_e 는 건드리지 않는다. env 부재 ⟹ **byte-동일**(음성대조 자격). ε 를 얻지 못한 unmapped 선(τ>1e-12) 이 하나라도 있으면 **fail-closed**(무음 열화 금지).

- 분류: 채움 루틴이 이미 계산하는 `used_nlte`(`a209_upper_population_for_tau` 출력)를 (선,셸) 바이트 배열로 **함께 내보내** `cmf_fine` 이 그것을 읽는다 — 재유도 금지(같은 술어).
- 적용: unmapped 선에서 `eps = radeq_line_eps_phys(line, n_e[s], T_e[s], τ)`; `η₀ ← η₀·eps`; 불투명도는 총량 보존 — 열적 `eps·χ₀` 는 `chi_line`, 산란 `(1−eps)·χ₀` 는 `chi_es`(기존 ALI 산란 알베도 경로) 로 적립. 프로파일 가중 p 는 둘 다 같은 식.
- census: 셸당 stderr 1 행 `[cmf_fine][UNMAPPED-EPS] shell=<s> n_unmapped=<n> n_mapped=<m> eps_min=<..> eps_median=<..> eps_max=<..> thermal_emit_fraction=<Σε η₀/Σ η₀>` — G-EPS 가 소비.

### 변경집합 — CMF_FINE_UNMAPPED_EPS (표)

| 경로 | 담당 | 내용 |
|---|---|---|
| `src/lumina_cmfgen.c` | Opus 4.8 | parity 분기 ε 적용(unmapped 만)·chi_es 적립·fail-closed·census 행; env `LUMINA_CMF_FINE_UNMAPPED_EPS`(기본 0 = byte-동일) |
| `src/lumina_plasma.c` | Opus 4.8 | `lumina_line_upper_population_fill_for_tau` 가 `used_nlte` (선,셸) 바이트 배열을 함께 발행 |
| `src/env_universe.h` | Opus 4.8 | `LUMINA_CMF_FINE_UNMAPPED_EPS` 등록(strict env 수용) |
| `tests/cmf_fine_unmapped_eps_selftest.c` | Opus 4.8 | CPU selftest: env 부재 (χ₀,η₀,chi_es) byte-동일 · unmapped 선 η₀ = ε·χ₀·B(1e-12) 및 chi_es += (1−ε)χ₀ · mapped 선 불변 · ε 부재 → rc≠0 · census 행 형식 |

### 변경집합 끝

## 4. 게이트 명부 (표)

| ID | 조건 | 음성대조 |
|---|---|---|
| G-ID | 343204 계보(`/gpfs/kjhan/lumina/cmfgen_map/consistent_ionization_state_20260911/retry_20260912/physical_run/{RUN_FOOTER.txt,input_manifest.sha256}`) 대비 차이가 exports 에서는 `LUMINA_CMF_FINE_UNMAPPED_EPS` 키 하나, source 에서는 §3 변경집합 표의 행 전부(`src/lumina_cmfgen.c`·`src/lumina_plasma.c`·`src/env_universe.h`·`tests/cmf_fine_unmapped_eps_selftest.c`)뿐임을 `identity.json` 으로 확인 | 이전 계보의 exports 해시를 identity 에 주입 ⟹ FAIL |
| G-RUN | 2 외부반복 완주, stderr `[cmf_fine][EXACT-MULTIGPU-EPOCH] status=OK` ≥ 2, `work/physics_comparison/physics_DET_iter0001.spectral.csv`·`work/lumina_ion_pops.csv`·`work/nlte_levels_final.csv` 실재 | 반복수 변조·덤프 결손 주입 |
| G-EPS | stderr `[cmf_fine][UNMAPPED-EPS]` 행이 셸 50개 전부 있고, 셸 41–49 에서 `thermal_emit_fraction ≤ 1e-2` 이며 `n_unmapped > 0` | census 를 ε=1 로 위조한 로그 ⟹ FAIL(thermal_emit_fraction=1); 행 결손 ⟹ FAIL |

### 게이트 명부 끝

## 5. 기대치 — 런에서만 얻는 수 (판독 A §2 정적 예측; 자릿수 기대)

잣대를 base(343204 산출)에 먼저 적용: base 는 E-1·E-2·E-3·E-4 어느 문턱도 넘지 못한다(각 「지금」 값이 문턱의 10–10⁴ 배 밖).

| ID | 양 (기존 도구) | base(343204) | **기대(런 후)** |
|---|---|---|---|
| E-1 램프 | 창 빈 853/898/758/811 의 J/B(T_e), 셸 28·32·36 (12 값, `window_leak.py`) | 최소 1.4e3, 최대 6.8e6 | **12 값 전부 < 100** (정적 예측 0.03–15) |
| E-2 Γ | S II σ-가중 Γ(J)/Γ(B), 셸 28·32·36 (`rebalance_check.py`) | 2.2e3 / 5.1e3 / 3.1e3 | **셋 다 < 30** (예측 1.9/1.2/0.6) |
| E-3 S 외곽 | S f_IV 셸 32·36 (`lumina_ion_pops.csv`) | 0.933 / 0.992 | **둘 다 < 0.20**, f_III > 0.70 (예측 .990/.918) |
| E-4 Si 외곽 | Si f_IV 셸 28·32 | 0.963 / 0.997 | **둘 다 < 0.30** (예측 .016/.114) |
| E-5 중간 | S II→III Γ/Γ(B) 셸 12·16·20 | 3.2 / 12.7 / 55 | < 2 / < 4 / < 20 (예측 0.8/1.3/8.4) |
| E-6 대조(불변) | 셸 0–8 Fe·Co·Ni IV 분율; S f_II 셸 12 | IV 0.98; 0.004 | \|Δ\| < 0.02; **f_II(sh12) < 0.05**(CMFGEN 0.845 에 **가지 않는다** — ① 의 몫) |

### 분기 (표)

| 분기 | 조건 | 의미 |
|---|---|---|
| LU-X | G-ID/G-RUN/G-EPS 중 미충족 | 미시연 — 원인 명명 |
| LU-A | 게이트 전건 충족 **and** E-1 **and** E-2 **and** E-3 **and** E-4 | **램프 = 외곽 LTE-unmapped 선 방출 확정**, 외곽 이온화 III 복귀 ⟹ 다음 = ① 연속체 채움 rung(퇴화-파괴 채널 동반) |
| LU-B | 게이트 전건 충족 **and** E-1 **and not**(E-3 **and** E-4) | 램프는 꺼졌으나 외곽 과이온화 잔존 ⟹ 다른 구동(① 또는 창 밖 빈) — 판정자가 명명 |
| LU-C | 게이트 전건 충족 **and not** E-1 | 방출원이 parity 분기 LTE 채움이 아니다(INIT 패스·연속체·다른 경로) ⟹ 귀속 정정 |
| LU-M | 그 외 | 판정자가 명명, 재분기 금지 |

### 분기 끝

E-5·E-6 은 분기에 참여하지 않는다(기재·정정용). 문턱 근거: base 와 예측 사이 **두 자릿수 이상** 벌어진 곳(정적 예측기의 자릿수 불확실성 ≤ 1 자릿수 — 판독 A §2 검증 표).

## 6. 자원 · 실행

GPU **1 런**, 파티션 h200→h100→a100(343204 실적 a100, 73.5 GB, 약 3.3 h) — `--time=05:00:00`, 상한 **10 GPU-hour**. 새 RUN/EVID 루트(봉인 밖). runner = 343204 의 `job.slurm` 계보 + env 1개(`LUMINA_CMF_FINE_UNMAPPED_EPS=1`). 오프라인 ≤ 1 CPU-hour(selftest·판독). 봉인 base(343204 산출)는 읽기만.
순서(개정16 ②): Opus 4.8 구현 → selftest(봉인 무접촉) → **운전자 코드 간단 감사(Q0 봉인 접촉 가부)** → 빌드 → 스테이징 → 제출.

## 7. 분장 장부 (개정25)

| 단계 | 규약 | 실제 | 비고 |
|---|---|---|---|
| 기안 | 운전자 | 운전자(본 문서) | **자기 기안** |
| 사전감사(Q-GOAL·Q-LEAN) | (개정25 미배정) | 미집행 → **Opus 4.8 fresh 발주 예정**(JSUBST 전례; GPU 지출 단) | 기안자 ≠ 감사자 |
| 구현 + selftest | Opus 4.8 fresh | 미집행 | `model` 생략 발주, 첫 줄 모델 ID 자기보고 |
| 코드 간단 감사 | 운전자 | 미집행 | 구현자 ≠ 검수자 |
| 스테이징·제출·판독·대장·커밋 | 운전자 | 미집행 | |
| 판정 | 운전자 | 미집행 | ⚠판정문에 운전자 이탈이 기재될 수 있음 — 종료감사로 보완 |
| **종료감사** | **Opus 4.8 fresh** | 미집행 | GPU 지출 단·물리 주장 등급 상승 단 — **생략하지 않음** |

## 8. 주장 상한
「외곽 창 빈 램프의 방출원이 LTE-unmapped 선 채움인가」와 「그것을 걷으면 외곽 이온화가 III 로 복귀하는가」까지. 중간 셸의 CMFGEN 복귀·① 의 수리·스펙트럼 영향·NLTE 승격의 필요는 말하지 않는다. twin 핀(외곽 20 kK)은 유지되므로 자유-T_e 에서의 크기는 범위 밖.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 — CMF_FINE_UNMAPPED_EPS (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "(?:src|tests)/[A-Za-z0-9_/.]+\\.(?:c|h)",
    "symbol": "CMF_FINE_UNMAPPED_EPS",
    "roots": ["src", "tests"],
    "expected_extra": ["src/lumina_cmfgen.c", "src/lumina_plasma.c", "src/env_universe.h", "tests/cmf_fine_unmapped_eps_selftest.c"]
  },
  "branches": {
    "regimes": [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
    "metrics": {"count": "len(v)", "gate_fail": "sum(1 for x in v if x < 0.5)", "lamp_off_outer_iii": "sum(1 for x in v if 0.5 <= x < 1.5)", "lamp_off_outer_iv": "sum(1 for x in v if 1.5 <= x < 2.5)", "lamp_on": "sum(1 for x in v if 2.5 <= x < 3.5)", "other": "sum(1 for x in v if x >= 3.5)"},
    "rules": [
      {"name": "LU-X", "predicate": "count == 0 or gate_fail > 0"},
      {"name": "LU-A", "predicate": "count > 0 and gate_fail == 0 and lamp_off_outer_iv == 0 and lamp_on == 0 and other == 0"},
      {"name": "LU-B", "predicate": "count > 0 and gate_fail == 0 and lamp_off_outer_iv > 0 and lamp_on == 0 and other == 0"},
      {"name": "LU-C", "predicate": "count > 0 and gate_fail == 0 and lamp_on > 0 and other == 0"},
      {"name": "LU-M", "predicate": "count > 0 and gate_fail == 0 and other > 0"}
    ],
    "adversarial_fixtures": [
      {"name": "empty", "mix": []},
      {"name": "gate-fail", "mix": [[0.0, 1], [1.0, 2]]},
      {"name": "lamp-off-iii", "mix": [[1.0, 3]]},
      {"name": "lamp-off-iv", "mix": [[1.0, 1], [2.0, 1]]},
      {"name": "lamp-on", "mix": [[3.0, 1]]},
      {"name": "other", "mix": [[4.0, 1]]}
    ],
    "residual": "LU-X"
  },
  "references": [
    {"path": "docs/READOUT_IMPLIED_J_LEDGER_2026-09-17.md"},
    {"path": "docs/READOUT_LAMP_PREDICTION_2026-09-17.md"},
    {"path": "docs/CLOSEOUT_CODEX_PERIOD_2026-09-17.md"},
    {"path": "CLAUDE.md"},
    {"path": "src/lumina_cmfgen.c", "flags_existing": ["cmf_fine_line_material", "upper_population_cache", "LUMINA_CMF_FINE_LINE_EPS", "chi_es"]},
    {"path": "src/lumina_plasma.c", "flags_existing": ["lumina_line_upper_population_fill_for_tau", "a209_upper_population_for_tau", "radeq_line_eps_phys", "lte_unmapped_emit"]},
    {"path": "src/line_net_rate.c", "flags_existing": ["line_net_sobolev_material", "emission_per_sr"]}
  ]
}
```
