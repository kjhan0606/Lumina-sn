# 사전등록 — 이온 단계 밀도 이중 권위: CMFGEN 발자국 대조 + 단일 변수 대조 1런 (DET-DUAL-AUTH, 2026-09-05)

저자 = Fable(운전석). 상태 = **발주 전**. 근거 문서 = `docs/AUDIT_DUAL_ION_DENSITY_WIRING_2026-09-05.md`, `docs/READOUT_DPROXY_SL_COVARIATION_2026-09-05.md` §9.
user 지시(09-05): 「1번(수리 방향 사전등록, CMFGEN 소스 대조부터) 후에 2번(정량 영향, 단일 변수 대조 1런)을 수행한다.」

## 1. CMFGEN 발자국 (소스 실측, `/gpfs/kjhan/cmfgen_src/cur_cmf`) [저자 실행]

| 요소 | CMFGEN | 근거 |
|---|---|---|
| 미지수 벡터 | 한 깊이점에서 전 종·전 단계·전 (super)준위 population + **ED(전자 밀도) = 변수 NT−1** + **T = 변수 NT** | `new_main/cmfgen_sub.f:1557 DO I=1,NT-1`, `:1887 BA_T(NT-1,…)/ED(K)` |
| 통계 평형 | 종별 전 단계를 한 계에 (`DI_S` 다음 단계 밀도 포함), 「ionization equilibrium equations 도 증분」 | `new_main/subs/steq_multi_v10.f` 헤더·`EQGS` 행 배치 |
| 종 수 보존 | 종별 abundance equation 1행 (`SPEC_EQ`) | `subs/vsebyj_multi_v4.f:27`, `evalse_qwvj_v7.f:84` |
| 전하 보존 | BA 행렬의 1행, ED 에 대한 변분 포함 | `new_main/subs/steqne_v4.f` 헤더(“charge conservation equation, and the variation charge equation”), `cmfgen_sub.f:1680 CALL STEQNE_V4` |
| 갱신 | 위 전부를 **한 Newton 선형화(BA)로 동시** 갱신 | `cmfgen_sub.f:1548 IF(.NOT. LAMBDA_ITERATION .AND. COMPUTE_BA)` |
| `EVAL_ED` | 초기값·LTE 추정 전용(Newton for Ne) — 주 루프 권위 아님 | `subs/eval_ed.f` 헤더 |

⟹ **발자국 = 단일 권위**: 이온 단계 밀도는 준위 population 의 합이며 따로 없고, n_e 는 전하 보존 방정식이 정하는 미지수다.

## 2. Lumina 현행 (감사 문서 §2 요약) [저자 실행]

두 권위: Saha `atom->ion_number_density`(rate-SE 네뷸라 Saha, `lumina_plasma.c:2742`; n_e·쌍 총량·η_bf 재결합·비사상 선·Γ_phot(LTE 가중)) 대 NLTE 준위해 Σ(쌍 총량은 Saha 합, II/III 분배는 행렬; 사상 선 τ/S/η·χ_bf·ROW·CMF). CPU 경로 `nlte_solve_ion_shell:19823` 와 **GPU 경로 `lumina_cuda.cu:1555–1605` 가 같은 논리**(`gpu_fb_lock_mode = nlte_ion_lock_active || nlte_per_ion_rescale_active`; 기본 = 「combined Σ x_i = n_pair_total」 `:1619`). 이 런들의 population solve 는 GPU(`[NLTE-GPU] cuBLAS solver`, 331050 stdout:243). 실측 어긋남 = 바닥 준위 S-역산 Fe II +3.62%/반복(승격)·+12.5%(반상태).

## 3. 수리 방향 분류 (결정 아님 — user 결정 자료)

| 방향 | 스위치/작업 | 발자국 대비 | 통합되는 것 | 남는 것 |
|---|---|---|---|---|
| D-A | `LUMINA_NLTE_OPACITY_IONSTAGE=1` (되쓰기, `:3381`; GPU 끝 `cuda.cu:1931`) | **향함(부분)** | 같은 반복 안 η_bf 재결합 항·bulk τ 가 NLTE 분배를 봄 | n_e 는 Saha(`compute_electron_density`), 다음 반복 `compute_plasma_state` 가 Saha 로 다시 덮음 |
| D-B | `LUMINA_NLTE_PER_ION_RESCALE=1` (또는 `ION_LOCK=1`) | **반대** — NLTE Σ 를 단계별 Saha 총량에 핀 | 권위 하나(Saha) | 이온화가 LTE 가중 Saha 로 고정(CMFGEN 과 반대) |
| D-C | n_e 폐합을 NLTE Σ(+비 NLTE 단계 Saha)로, Saha 층의 Γ_phot 을 NLTE pop 으로 | **향함** | 전하 보존이 NLTE 를 봄 | 코드 작업(계약 2개), 쌍 총량은 여전히 Saha |
| D-D | 종 수 보존 + 전하 보존 행을 준위 행렬에 넣어 단일 계 | **정확** | 전부 | 큰 작업, GPU 행렬 확장 |

「지금은 CMFGEN 발자국 그대로」 원칙의 종착은 D-D 이고, 증분 사다리는 D-A → D-C → D-D 다. **D-B 는 수리가 아니라 계측 팔이다**(아래 §4).

## 4. 2번: 단일 변수 대조 1런 (DET-DUAL-AUTH-2)

**질문**: 이중 권위(NLTE 재분배 대 Saha 분배)의 차이가 population 반복 이동과 formal 스펙트럼에 얼마나 들어가는가.
**설계**: base = 331050(H200)≡328645(A100, 685/685 bit-일치). arm = base 와 **정확히 하나** 다름: `LUMINA_NLTE_PER_ION_RESCALE=1`. 반발자국 방향이지만 「NLTE 분배 대 Saha 분배」의 차이를 그대로 재는 유일한 기존 스위치다(D-A 는 η_bf 항만 바꿔 감도가 낮다).
**자원**: A100×2 또는 H200×2 (등가), 32 CPU, 64 GiB, `--time=08:00:00`, 예상 2.5–3.2 h. GPU **1런**.
**환경**: `DPROXY-PROV-K42` 모드의 exports 에서 **기존 행 `declare -x LUMINA_NLTE_PER_ION_RESCALE="0"` 의 값을 `"1"` 로 치환**(추가 아님). 전집 등록 [저자 실행] `grep -c LUMINA_NLTE_PER_ION_RESCALE src/env_universe.h` = 1 ✔; base 키 실존 [저자 실행, 개정 1] `grep -c '^declare -x LUMINA_NLTE_PER_ION_RESCALE="0"' <331050>/input/resolved_lumina.exports` = 1 ✔ (검수 F-1).

### 4-1. 기대치 (arm 대 base)
- **E-1 (분기 결정)**: 바닥 준위 S-역산 `Δlog n(iter0→1)` — base Fe II +3.62%, Co II +0.49%, Ni II +4.09%. arm 에서는 NLTE Σ 가 Saha 에 핀되므로 **|Δlog n_ground| ≤ 0.3%** (Saha 상한 0.2% + 여유). 세 이온 전부.
- **E-2 (작동 대조, 음성대조 내장)**: generation 1 exact epoch(LTE 시드, population commit 전) 는 base 와 **bit-동일**(`iterations=45 residual=9.6662782724980344e-09`); generation 2 epoch 는 base(`53, 7.8976358675842993e-09`)와 **달라야** 한다. 둘 다 같으면 스위치가 안 걸린 것 ⟹ 무효.
- **E-3 (계측)**: formal 스펙트럼 `work/lumina_spectrum.csv`(1234 bin, 열 `wavelength_angstrom,flux`) arm/base `Δlog F = ln(F_arm/F_base)` 의 대역별(광자 에너지 `[0,2) [2,4) [4,6) [6,12) [12,∞)` eV, 5대역) 중앙·최대·최소, 비양수 flux 는 제외 계수, 파장 격자는 bit-동일이어야 함 — **기대치 없음, 측정 보고**. `Δlog n_upper`(338 짝, iter 1) arm 대 base 도 보고.

### 4-2. 게이트 (fail-closed) — `### 게이트 명부 (표)`
| 게이트 | 판정 | 참조 |
|---|---|---|
| G-0 env 단일 차이 | `diff <(sort /gpfs/kjhan/lumina/superlevel_b_h200x2_k36/slb_20260903T080407Z_dproxy_provenance_k42_ion1_h200/input/resolved_lumina.exports) <(sort <NEW_RUN_ROOT>/input/resolved_lumina.exports)` 의 결과에서 런루트 경로 치환 행을 제외하면 정확히 2행: `< declare -x LUMINA_NLTE_PER_ION_RESCALE="0"` 와 `> declare -x LUMINA_NLTE_PER_ION_RESCALE="1"` (같은 키의 값 변경, 그 밖 0행) | base `input/resolved_lumina.exports` 대 arm `input/resolved_lumina.exports` |
| G-1 봉인 입력 | `cmp /gpfs/kjhan/lumina/superlevel_b_h200x2_k36/slb_20260903T080407Z_dproxy_provenance_k42_ion1_h200/input/resolved_lumina.sha256 <NEW_RUN_ROOT>/input/resolved_lumina.sha256` 동일, `sha256sum <NEW_RUN_ROOT>/input/dproxy_candidate_registry.json` = `c96f74f0a78088bbc7cb3f7ad851060da6ee492d2d09476c26a57bc247ef1923`, `<NEW_RUN_ROOT>/RUN_FOOTER.txt` 의 `binary_sha256` = `a4990a592e422d737d2236ba111fdd60141c110c0c23204fc22c7054691dc27a` | base `input/` 대 arm `input/`·`RUN_FOOTER.txt` |
| G-2 gen-1 bit-동일 | `grep -m1 'EXACT-MULTIGPU-EPOCH' /gpfs/kjhan/lumina/superlevel_b_h200x2_k36/slb_20260903T080407Z_dproxy_provenance_k42_ion1_h200/stderr.log` 와 `grep -m1 'EXACT-MULTIGPU-EPOCH' <NEW_RUN_ROOT>/stderr.log` 의 `iterations`·`residual` 토큰 문자열 동일 | base 대 arm `stderr.log` |
| G-3 gen-2 상이 | 위 두 파일의 2회차 `EXACT-MULTIGPU-EPOCH` 행 `residual` 토큰이 서로 다름 | base 대 arm `stderr.log` |
| G-4 완주 | model rc=0, publication 2, 338 registry 짝 전건, provenance OK 685/685 계급 | `readout_dproxy_sl_331050.py` |
| G-5 역산 정합 | 같은 하준위 공유 선 간 `Δlog n_l` 산포 ≤ 1e-9 | `probe_lower_any.py` |
### 게이트 명부 끝

### 4-3. 분기 — `### 분기 (표)`
| 분기 | 조건 | 뜻 |
|---|---|---|
| **DA-A** | G-0~G-5 전건 통과, E-1 세 이온 만족 | 공통모드 = NLTE 재분배 **확증**(경로 2: S-역산 + 스위치 응답 ⟹ 등급 A 후보). E-3 이 이중 권위의 스펙트럼 발자국 크기 |
| **DA-N** | 게이트 통과, E-1 불만족(어느 이온이든 \|Δlog n_ground\| > 1%) | §9 귀속 **틀림** — 공통모드는 NLTE 재분배가 아니다. REWORK, 스펙트럼 수치 해석 금지 |
| **DA-X** | 게이트 하나라도 실패 | 무효, 사유 이름으로 정지 |
### 분기 끝

### 변경집합 — DUAL_AUTHORITY_CONTRAST (표)
| 경로 | 변경 |
|---|---|
| `scripts/stage_superlevel_b.sh` | 모드 `DPROXY-PROV-K42-RESCALE`(A100)·`DPROXY-PROV-K42-RESCALE-H200` 신설: 각각 `DPROXY-PROV-K42`·`DPROXY-PROV-K42-H200` 의 구현을 복제하되 (a) exports 의 기존 `LUMINA_NLTE_PER_ION_RESCALE` 행 **값을 0→1 치환**(추가 금지, 중복 키 금지), (b) base 모드의 exports 에 그 키가 `"0"` 으로 있는 것을 **정상**으로 취급(기존 base 모드 스테이징 회귀 0), (c) arm 의 자기 기술을 base 와 구별: run-root suffix `dproxy_provenance_k42_ion1_rescale`(H200: `…_rescale_h200`), job name `LUMINA_DPXRESC`(H200: `LUMINA_DPXRESCH200`), `input/superlevel_b_mode.txt`·`SUPERLEVEL_B_EXPECTED_EXPORT_DELTA.txt`·`OWNER.txt`·`SUPERLEVEL_B_STAGE_SEAL.txt`·post-model seal 의 mode 필드 전부 RESCALE 모드명; partition·GPU·CPU·mem·time·진단 플래그·landing/postflight 는 불변. selftest 양성 ≥2(base 키 `"0"` 실존 픽스처 포함)·음성 ≥4(값 미치환·중복 키·다른 env 추가·mode 자기 기술이 base), 기존 `--selftest` 전건 유지 + 봉인 331050 루트에 대한 `verify_staged_superlevel_contract`(base 모드) 재적용 rc=0 |
| `scripts/dproxy_readout/spectrum_contrast.py` | 신설: E-3 정의 그대로(격자 bit-동일 검증·5대역·중앙/최대/최소·제외 계수) JSON 산출, `python3 -B`·표준 라이브러리만. selftest 양성 1·음성 2(격자 불일치·비양수 flux) |
| `scripts/dproxy_readout/probe_lower_any.py` | 기존, 무변경 |
### 변경집합 끝

## 5. 주장 상한
DA-A 라도 「Saha 분배가 옳다」·「스펙트럼이 CMFGEN 에 가까워졌다」를 뜻하지 않는다. 수리 방향(D-A/C/D)은 별도 결정. 반복 2회(0→1)의 차분·shell 0·Fe/Co/Ni II 에 한정.

## 6. 분장 장부 (집행 기록 — 정본)
| 단계 | 규약상 | **실제** | 위반 |
|---|---|---|---|
| 사전등록 | Fable | Fable(운전석) | 개정18 ① 은 판정에 한함 |
| 코딩(모드·스크립트) | Codex | gpt-5.6-sol (발주 예정) | |
| 코드 검수 | Fable | Fable fresh 1회차: SAFE_TO_STAGE NO, 차단 3(F-1 중복 키·F-2 base 회귀·F-3 자기 기술) — `docs/REVIEW_DUAL_AUTH_ORDER4_2026-09-05.md` | 상류 원인 = 사전등록 저자(운전석)의 「1행 더함」 오전제 |
| 코드 검수 2회차 | Fable | Fable fresh(다른 컨텍스트): SAFE_TO_STAGE **YES**, 차단 0, 기재 4 — `docs/REVIEW_DUAL_AUTH_ORDER5_2026-09-05.md` | — |
| 발주·제출·계측 | 운전석 | 운전석 | |
| 판정 | Fable fresh | (기입 예정) | 운전석 금지 |
| 감리 A/B | Fable fresh / Codex | (기입 예정) | |

집행 기록:
- 프리플라이트 rc: `check_prereg_preflight` 0 · `check_gate_referents` 0 (1차 FAIL `OPERAND_VAGUE` G-0/G-1 → 피연산자 경로 명시 후 0) · `check_doc_canonical` 0 · `check_dispatch_preflight` 0 (1차 FAIL: 사전등록 원문을 발주서에 붙여 넣어 `DISPATCH-RESTATES-CONTRACT` → 앞겹 = 경로 지목으로 재작성 후 0).
- `[미실행]` 없음(전건 [저자 실행]). 발주 중단 없음. 발주 4 = `/gpfs/kjhan/lumina/dproxy_audits/readout_331050_fable/order4/ORDER4_DUAL_AUTHORITY_CHANGESET.md`, 계약 HEAD `d1e39d7`.
- GPU 가용(제출 시점 사전 확인): a100 2노드 16 GPU 전점유, h200 8 GPU 전점유 ⟹ 대기열 제출.
- 스테이징(09-05 18:07 KST, grammar-debug, worktree `/gpfs/kjhan/lumina/dproxy_audits/worktree_364ee72_dualauth` = `364ee72` + 331050 보존 패치 `a642a2dd…` ⟹ 소스 identity 4/4 base 동일): `SUPERLEVEL_B_STAGE_OK mode=DPROXY-PROV-K42-RESCALE env_delta=12 shared_hardlinks=0`, 봉인 census 136/135 PASS. run root `/gpfs/kjhan/lumina/superlevel_b_a100x2_k36/slb_20260905T090712Z_dproxy_provenance_k42_ion1_rescale`. G-0 사전 확인 = RESCALE 0→1 한 쌍뿐 ✔, G-1 sha 목록 `cmp` 동일·registry·binary ✔.
- user 제안(lageunha RTX 5000 Ada 1장·32 GB·sm_89): fatbin 은 sm_80/86/90 SASS 라 구동은 가능하나 GPU 2장·G-2 bit-동일 계약과 80 GB 요건에 부적격 ⟹ 미사용(2런 설계는 별건).
- 제출(18:09 KST): Slurm **332463** `LUMINA_DPXRESC`, partition a100, GPU 2·CPU 32·64 GiB·08:00:00, 제출 직후 `PENDING(Priority)`. OWNER.txt `DO_NOT_CANCEL=1`.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 — DUAL_AUTHORITY_CONTRAST (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "scripts/[a-z0-9_/]+\\.(?:py|sh)",
    "symbol": "DUAL_AUTHORITY_CONTRAST",
    "roots": ["scripts"],
    "expected_extra": ["scripts/stage_superlevel_b.sh",
                       "scripts/dproxy_readout/spectrum_contrast.py",
                       "scripts/dproxy_readout/probe_lower_any.py"]
  },
  "branches": {
    "regimes": [[0.0, 0.45], [0.9, 1.1], [10.5, 60.0]],
    "metrics": {
      "g_fail": "sum(1 for x in v if x < 0.5)",
      "moved": "sum(1 for x in v if x > 10.0)"
    },
    "rules": [
      {"name": "DA-X", "predicate": "g_fail > 0"},
      {"name": "DA-A", "predicate": "g_fail == 0 and moved == 0"},
      {"name": "DA-N", "predicate": "g_fail == 0 and moved > 0"}
    ],
    "adversarial_fixtures": [
      {"name": "gates-pass-ground-pinned", "mix": [[1.0, 9]]},
      {"name": "gates-pass-ground-moves", "mix": [[1.0, 8], [42.0, 1]]},
      {"name": "gate-fail", "mix": [[1.0, 7], [0.3, 1], [42.0, 1]]}
    ],
    "residual": "DA-X"
  },
  "references": [
    {"path": "src/lumina_cuda.cu",
     "flags_existing": ["nlte_per_ion_rescale_active", "gpu_fb_lock_mode", "nlte_writeback_ion_stage"]},
    {"path": "src/lumina_plasma.c",
     "flags_existing": ["LUMINA_NLTE_PER_ION_RESCALE", "nlte_per_ion_rescale_active", "nlte_writeback_ion_stage", "compute_electron_density"]},
    {"path": "src/env_universe.h",
     "flags_existing": ["LUMINA_NLTE_PER_ION_RESCALE", "LUMINA_NLTE_OPACITY_IONSTAGE"]},
    {"path": "scripts/stage_superlevel_b.sh",
     "flags_existing": ["DPROXY-PROV-K42"]},
    {"path": "scripts/dproxy_readout/probe_lower_any.py",
     "flags_existing": ["source_function"]}
  ]
}
```


## 개정 1 (2026-09-05, 검수 1회차 반영 — OLD/NEW 3쌍)
- §4 환경: OLD 「+ … =1 만」 → NEW 「기존 행 값 0→1 치환」.
- G-0: OLD 「정확히 1행」 → NEW 「같은 키 값 변경 2행」.
- 변경집합 stager 행: OLD 「더함」 → NEW 「치환·base 키 정상·arm 자기 기술 구별·픽스처 보강」.
행 단위 diff 로 의도 외 소실 0 확인 [저자 실행].
