# 사전등록 — D-C-1: NLTE 커밋 뒤 전자 밀도 폐합 (DET-DC1-NECLOSE, 2026-09-06)

저자 = Fable(운전석). 상태 = **발주 전**. 상위 = `docs/DESIGN_DC_DD_CMFGEN_CONTRAST_2026-09-05.md` §3 첫 단. 근거 = `docs/AUDIT_DUAL_ION_DENSITY_WIRING_2026-09-05.md`(이중 권위), `docs/VERIFICATION_REGISTERS.md` 09-06(DET-DUAL-AUTH-2 폐합: Saha/NLTE Fe II 0.884·Co 1.0231·Ni 0.9549, shell 0). user 지시(09-06): 「D-C-1 사전등록 착수」.

## 1. 계약 (물리 계약 1개)

**현행**: n_e 는 Saha 층 `compute_electron_density`(`src/lumina_plasma.c:3048`) 가 `Σ_ip charge·ion_number_density`(Saha 단계 밀도)로 정하고, NLTE 준위해가 II/III 를 재분배해도 n_e 는 **되돌아오지 않는다**(NLTE 트랜잭션은 `work_ne` 를 들고만 있고 쓰지 않음, `:20206`). 전하 보존이 NLTE 분배를 못 본다(감사 §2 표).
**계약**: NLTE population commit 직전에 **폐합 n_e** 를 계산해 같은 트랜잭션으로 발행한다.
`n_e^closed(s) = Σ_{(Z,q)} q · n_{Z,q}(s)`, `n_{Z,q} = Σ_{준위∈(Z,q)} nlte_level_populations`(NLTE 집합 (II,III)×{Si,S,Ca,Fe,Co,Ni}) / 그 밖 (Z,q) 는 `atom->ion_number_density`(Saha). `q = ion_pop_stage`(현행 `:3091` 와 같은 전하 규약).
**게이트 env**: `LUMINA_NLTE_NE_CLOSURE=1` 일 때만. 미설정·0 ⟹ 현행과 **byte-동일**(코드 경로 진입 0).
**발행 기록**: 커밋 시점에 shell 별 1행 `[A2-07][NE-CLOSURE] generation=… shell=… n_e_saha=%.17g n_e_closed=%.17g dlog=%.17g residual=%.3e nlte_charge=%.17g saha_charge=%.17g` — `residual = |Σ q n − n_e_closed| / n_e_closed` 는 같은 합을 두 번 계산한 자기 검사(≤ 1e-12 by construction).
**바꾸지 않는 것**: Saha 층의 n_e 반복(TARDIS식 감쇠 `n_e = ½(new+old)`, **수렴 문턱 5% 상대**, `:3105`) — 이것은 별도 계약 후보로 대장에 조용히 기재한다(전하 보존이 5% 안에서만 닫히는 현행 상태). 다음 반복의 Saha 층은 `plasma->n_electron[s]` 를 시작값으로 읽으므로(`:3069`) 폐합 n_e 가 자연히 다음 반복의 이온화 입력이 된다 — 그 되먹임 자체가 계약의 목적(전하 보존 권위 하나).

## 2. 소스 상태 (단일 변수 유지)
패치 base = **331050 의 소스 상태** = `364ee72` + 보존 패치 `a642a2dd…`(worktree `/gpfs/kjhan/lumina/dproxy_audits/worktree_364ee72_dualauth`, identity 4/4 base 동일). 작업 트리의 Codex 기간 미커밋 변경(quarantine 수정 등)은 **넣지 않는다** — OFF 팔의 base 대비 bit-동일이 게이트라서. 새 바이너리 = 그 상태 + 이 계약 패치 뿐(sm_80/86/90 fatbin, sha 등록).

## 3. 런 설계 — **2런** (OFF 대조 1 + ON 팔 1), 각 A100×2·32 CPU·64 GiB·`--time=08:00:00`, base 331050 과 같은 물리 exports
| 런 | env | 목적 |
|---|---|---|
| **OFF** | 새 바이너리, `LUMINA_NLTE_NE_CLOSURE` 미설정 | 증분 byte-배터리: 새 바이너리가 base 와 **bit-동일**한 물리를 내는가(G-A). 실패 시 ON 발주 금지 |
| **ON** | 새 바이너리, `LUMINA_NLTE_NE_CLOSURE=1` | 계약 본체 |
OFF 가 G-A 를 통과하면 ON 대 OFF 가 단일 변수 대조다(base 331050 대 ON 도 결과적으로 단일 변수).

## 4. 기대치 (「구성상 도출」과 「측정」을 구분)
- **E-1 (구성상 — 작동 확인)**: ON 의 모든 `[A2-07][NE-CLOSURE]` 행에서 `residual ≤ 1e-12`. 코드 귀결이며 예측이 아니다.
- **E-2 (도출값 대조, shell 0)**: 09-06 실측 단계 비로 도출한 shell-0 폐합 변화 `Δlog(n_e^closed/n_e^saha)` 의 크기: Fe II Saha→NLTE +13% (Δn_II ≈ +4.9e3 cm⁻³, 쌍 총량 보존이라 III 이 같은 양 감소 ⟹ Δn_e = −Δn_II), Ni +7e3, Co −1.4e4 ⟹ 합 ≈ +1e3 / 3.2e9 ⟹ **|Δlog n_e(shell 0)| ≲ 1e-5**. 이는 09-06 측정에서 도출되므로 대조(일치 여부)이지 새 정보가 아니다. 어긋나면(>1e-4) 도출의 전제(쌍 총량 보존·II/III 만 NLTE)가 깨진 것 ⟹ NC-N.
- **E-3 (측정, 기대치 없음)**: 셸별 `dlog` 프로파일(50 셸). 바깥 셸에서 II/III 몫이 크면 클 수 있다 — 어디서 얼마나가 이 런의 정보.
- **E-4 (측정)**: ON 대 OFF 의 반복 1 단계 비 변화 — shell 0 S-역산 바닥 `n(ON)/n(OFF)` 이온별(`scripts/dproxy_readout/cross_run_ratio.py`), 폐합 n_e 가 다음 반복 Saha·행렬에 되먹인 크기.
- **E-5 (측정)**: formal 스펙트럼 ON/OFF 5대역 `Δlog F`(`scripts/dproxy_readout/spectrum_contrast.py`, 격자 bit-동일).

### 게이트 명부 (표)
| 게이트 | 판정 | 참조 |
|---|---|---|
| G-A OFF bit-동일 | `grep -o 'EXACT-MULTIGPU-EPOCH\] status=[A-Z]* devices=[0-9/]* iterations=[0-9]* cap=[0-9]* residual=[0-9.e+-]*' <OFF_ROOT>/stderr.log` 3행이 `/gpfs/kjhan/lumina/superlevel_b_h200x2_k36/slb_20260903T080407Z_dproxy_provenance_k42_ion1_h200/stderr.log` 의 3행과 문자열 동일 **그리고** `scripts/dproxy_readout/readout_dproxy_sl_331050.py --run-root <OFF_ROOT>` 의 `pairs.csv` 가 base `out_331050/pairs.csv` 와 `n_upper_0,n_upper_1,Jbar_0,Jbar_1` 열에서 338/338 문자열 동일 | OFF 대 base |
| G-B env 단일 차이 | `diff <(sort <OFF_ROOT>/input/resolved_lumina.exports) <(sort <ON_ROOT>/input/resolved_lumina.exports)` 에서 런루트 경로 치환 행 제외 = `> declare -x LUMINA_NLTE_NE_CLOSURE="1"` 1행 (OFF 에는 키 부재) | OFF 대 ON |
| G-C 봉인 입력 | OFF·ON 의 `input/resolved_lumina.sha256` 가 base 와 `cmp` 동일, registry `c96f74f0a78088bbc7cb3f7ad851060da6ee492d2d09476c26a57bc247ef1923`, 두 런의 `RUN_FOOTER.txt binary_sha256` 동일(새 바이너리, 사전 등록값) | stage seal |
| G-D 완주 | 두 런 model rc=0, publication 2, provenance OK 전건, iter-0 pub ⊇ registry 338, 두 pub 공통 ≥330 | readout |
| G-E 폐합 기록 | ON 의 `[A2-07][NE-CLOSURE]` 행 수 = 50 셸 × commit 세대 수(2), OFF 에는 0행, 전 행 `residual ≤ 1e-12` | ON `stderr.log` |
| G-F 역산 정합 | 두 런 `probe_lower_any.py` 산포 ≤ 1e-8 | S-역산 |
### 게이트 명부 끝

### 분기 (표)
| 분기 | 조건 | 뜻 |
|---|---|---|
| **NC-A** | G-A~G-F 통과, E-2 일치(shell 0 ≤1e-4), 전 셸 \|dlog\| ≤ 1e-3 | 폐합은 이 시대(K42·2반복·T_e 핀)에서 **수치적으로 불활성** — 발자국 한 걸음이되 스펙트럼 귀결 없음. E-5 는 기술통계 |
| **NC-B** | G 통과, E-2 일치, 어느 셸이든 \|dlog\| > 1e-3 | 폐합이 **어딘가에서 물리량** — 셸·크기 보고, E-4·E-5 가 되먹임 크기 |
| **NC-N** | G 통과, E-2 불일치(shell 0 \|dlog\| > 1e-4) 또는 ON 의 A2-07 n_e 반복 미수렴 | 도출 전제 붕괴 또는 되먹임 발산 — REWORK, 수치 해석 금지 |
| **NC-X** | 게이트 실패(G-A 실패면 ON 미발주) | 무효, 사유 이름 |
### 분기 끝

### 변경집합 — DC1_NE_CLOSURE (표)
| 경로 | 변경 |
|---|---|
| `src/lumina_plasma.c` | `nlte_ne_closure_publish(...)` 신설: NLTE 집합 Σ준위 + 비 NLTE Saha 로 `Σ q n` 계산, `pop_tx.work_ne` 에 기록, `[A2-07][NE-CLOSURE]` 행 발행; `nlte_solve_all_impl` 의 `population_transaction_commit`(`:20806`) **직전** 1회 호출; env 미설정 시 즉시 return(코드 경로 진입 0, 부동소수 연산 0) |
| `src/env_universe.h` | 전집 도출 스크립트(references 의 derive 스크립트) 재생성으로 `LUMINA_NLTE_NE_CLOSURE` 등록(수작업 편집 금지) |
| `validation/instrumentation_debt/ENV_UNIVERSE.json` | 위와 같은 재생성 산출 |
| `tests/dc1_ne_closure_selftest.c` | 합성 원자(원소 2, 단계 3, NLTE 쌍 1): ON 에서 residual ≤1e-12·Saha 대 NLTE 단계 차이가 n_e 에 정확히 반영, OFF 에서 `n_electron` 배열 byte-동일; 음성대조 ≥4(NLTE 준위 하나 누락·전하 규약 뒤집기·비 NLTE 단계 누락·env=0 인데 발행) |
| `scripts/stage_superlevel_b.sh` | 모드 `DC1-OFF`·`DC1-ON` 신설: `DPROXY-PROV-K42` 복제 + 새 바이너리 sha 등록 + (ON 만) env 1개 추가(키 신규라 「추가」가 맞음, 중복 검사 유지); 자기 기술 7곳 모드명; selftest 양성 2·음성 ≥4 |
### 변경집합 끝

## 5. 주장 상한
NC-A/B 어느 쪽도 「전하 보존이 옳게 닫혔다」를 뜻하지 않는다 — Saha 층의 5% 수렴 문턱과 Saha 단계(I, IV+·quarantine 원소)의 LTE 가중 Γ_phot 이 그대로다. CMFGEN 대조는 이 단의 산출로 하지 않는다(쌍둥이 T_e 레인 필요). 반복 2회·T_e 핀·K42 시대에 한정.

## 6. 분장 장부 (집행 기록 — 정본)
| 단계 | 규약상 | **실제** | 위반 |
|---|---|---|---|
| 사전등록 | Fable | Fable(운전석) | — |
| 코딩 | Codex | gpt-5.6-sol (발주 예정, worktree 위) | |
| 코드 검수 | Fable fresh | (기입 예정) — Q0 「봉인 접촉 가부」 첫 항목 | |
| 발주·빌드·스테이징·제출 | 운전석 | 운전석 | |
| 판정 | Fable fresh | (기입 예정) | 운전석 금지 |
| 감리 A/B | Fable fresh / Codex read-only | (기입 예정) | |
집행 기록:
- 프리플라이트 rc: `check_prereg_preflight` 0(1차 FAIL `changeset-extra-file` — 표 안 스크립트 경로 리터럴 제거 후 0) · `check_gate_referents` 0 · `check_doc_canonical` 0 · `check_dispatch_preflight` 0.
- 발주 6 = `/gpfs/kjhan/lumina/dproxy_audits/dc1/ORDER_DC1_CHANGESET.md`, 계약 HEAD `34a113e`, 작업 디렉토리 = worktree(§2). worktree 에 Makefile strict-env 타깃·`tests/env_universe_strict_selftest.py`·`scripts/derive_env_universe.py` 를 메인 트리에서 복사(identity 밖). `[미실행]` 없음. 발주 중단 없음.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 — DC1_NE_CLOSURE (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "(?:src|tests|scripts|validation)/[A-Za-z0-9_/.]+\\.(?:c|h|py|sh|json)",
    "symbol": "DC1_NE_CLOSURE",
    "roots": ["src", "tests", "scripts", "validation"],
    "expected_extra": ["src/lumina_plasma.c",
                       "src/env_universe.h",
                       "validation/instrumentation_debt/ENV_UNIVERSE.json",
                       "tests/dc1_ne_closure_selftest.c",
                       "scripts/stage_superlevel_b.sh"]
  },
  "branches": {
    "regimes": [[0.0, 0.45], [0.9, 1.1], [2.0, 3.0], [10.5, 60.0]],
    "metrics": {
      "g_fail": "sum(1 for x in v if x < 0.5)",
      "material": "sum(1 for x in v if 2.0 <= x <= 3.0)",
      "broken": "sum(1 for x in v if x > 10.0)"
    },
    "rules": [
      {"name": "NC-X", "predicate": "g_fail > 0"},
      {"name": "NC-N", "predicate": "g_fail == 0 and broken > 0"},
      {"name": "NC-B", "predicate": "g_fail == 0 and broken == 0 and material > 0"},
      {"name": "NC-A", "predicate": "g_fail == 0 and broken == 0 and material == 0"}
    ],
    "adversarial_fixtures": [
      {"name": "gates-pass-inert", "mix": [[1.0, 9]]},
      {"name": "gates-pass-material-shell", "mix": [[1.0, 8], [2.5, 1]]},
      {"name": "gates-pass-premise-broken", "mix": [[1.0, 8], [42.0, 1]]},
      {"name": "gate-fail", "mix": [[1.0, 7], [0.3, 1], [2.5, 1]]}
    ],
    "residual": "NC-X"
  },
  "references": [
    {"path": "src/lumina_plasma.c",
     "flags_existing": ["compute_electron_density", "population_transaction_commit", "work_ne", "ion_pop_stage", "nlte_level_populations"]},
    {"path": "src/env_universe.h",
     "flags_existing": ["LUMINA_NLTE_PER_ION_RESCALE"]},
    {"path": "scripts/derive_env_universe.py",
     "flags_existing": ["getenv"]},
    {"path": "scripts/stage_superlevel_b.sh",
     "flags_existing": ["DPROXY-PROV-K42-RESCALE"]},
    {"path": "scripts/dproxy_readout/cross_run_ratio.py",
     "flags_existing": ["source_function"]},
    {"path": "scripts/dproxy_readout/spectrum_contrast.py",
     "flags_existing": ["wavelength_angstrom"]}
  ]
}
```
