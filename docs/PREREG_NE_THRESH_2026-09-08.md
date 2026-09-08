<!-- 저작: astra(gpt-6-astra) 기안 (개정21 계획 담당). 운전석은 파일화·프리플라이트·커밋만 수행하고 본문을 고치지 않았다.
     발주 = /gpfs/kjhan/lumina/dproxy_audits/nethresh/ORDER_ASTRA_PLAN_NETHRESH.md · 원문 = /gpfs/kjhan/lumina/dproxy_audits/nethresh/ASTRA_PLAN_NETHRESH.md -->
# 사전등록 기안 — Saha 층 n_e 수렴 문턱 (DET-NE-THRESH, 2026-09-08)

**런 0회로 답할 수 있는 것 / 런이 필요한 것:** 봉인 소스의 정지 기전·대수, 보존된 상태가 충분한 경우의 전하 사상 F와 F′, 문턱별 재생·독립 근 대조·OFF 함수 배터리는 오프라인으로 확인할 수 있다. 실제 수송과 결합한 새 상태의 수렴, 공개 population·스펙트럼 변화, 전체 OFF 증분 동일성은 런이 필요하다. 기존 로그의 n_e·전하합만으로 F′를 복원할 수는 없다. 필요한 고정 상태가 없으면 `INPUT_PENDING`이며, 자료획득런을 오프라인으로 세거나 판정런으로 기전 탐색을 대신하지 않는다.

저자 = **astra(본 응답)**. 상태 = **기안·미집행**. 이번 작업은 조회·해시 확인만 수행했다. 파일 생성·코드 변경·계산런·프리플라이트 실행 없음. 문턱의 실제 선택과 기대치 봉인은 아래 오프라인 절차 뒤 완료한다.

## 1. 계약 — 물리 계약 1개

**계약:** 현재 원자모형·복사장·열역학 상태가 정하는 Saha/rate-SE 전하중성 방정식 `n_e = Fθ(n_e) = Σ q·n_q(n_e;θ)`의 **반복 정지 문턱만** 수리한다. 「지금은 CMFGEN 발자국 그대로」를 적용하여 원자자료·율·이온 사다리·전하 규약·NLTE 재분배·온도·수송 격자를 유지한다. CMFGEN 일치 자체는 이 계약의 판정 대상이 아니다.

현행 `x → m=(x+Fθ(x))/2`, 채택 전 `abs(Fθ(x)−x)/x < 0.05`, 최대 100회를 유지하되, ON에서는 `0.05`만 사전 봉인한 `τ*`로 교체한다. 새 키 `LUMINA_SAHA_NE_THRESH` 미설정·`0`은 기존 경로이며, ON은 양의 유한 `τ*<0.05`; 그 밖 설정은 명시적 실패다. `LUMINA_NLTE_NE_CLOSURE=1`은 양팔 공통이다.

100회 미수렴은 기존 `NE_FAIL` 및 후보 거부로 전파한다. 실패값 발행·마지막 값 성공 처리·자동 한도 증가·다른 해법으로의 fallback은 허용하지 않는다. 정확해를 배제하는 clamp/floor를 추가하지 않는다. 문턱과 반복 한도는 계산의 성공·실패를 정할 뿐 물리해의 허용 범위를 정하지 않는다.

**잔차 구분:** `δ=(Fθ(x)−x)/x`, `r_old=abs(Fθ(x)−m)/m`, `r_m=ln(Fθ(m)/m)`를 별도 기록한다. `m≠x`이면 구간 할선 `s=(Fθ(m)−Fθ(x))/(m−x)`에 대해 **정확히** `r_m=log1p[δ(1+s)/(2+δ)]`; `δ=0`이면 같은 결정론 사상에서 `r_m=0`이다. `δ(1+F′θ(x))/2`는 선형근사이며, 측정한 점 미분도 구간 전체의 엄밀 상한은 아니다.

## 2. 소스 상태 — 봉인 identity 기준 확인

기준 런 `<BASE>` = `/gpfs/kjhan/lumina/superlevel_b_a100x2_k36/slb_20260906T234024Z_dc1_ne_closure_on`(**333531**). 기준 소스 `<SRC>` = `/gpfs/kjhan/lumina/dproxy_audits/worktree_364ee72_dualauth`.
`<BASE>/input/current_source_identity.json`의 source_files **5/5**와 `<SRC>` 파일 해시 일치를 본 저자가 확인했다. git head는 `364ee72ba17dbd944bacc4a7c3420693500b9c88`; HEAD만을 소스 identity로 쓰지 않는다.
봉인 패치 `<BASE>/input/current_source.patch` SHA256 = `916eb7bb0a5ef41a876b13f59f71245a2cc49490d0f40fc94f1070a72e0ab9a9`.
`<SRC>/src/lumina_plasma.c` SHA256 = `90fd73cc51cae34649533fc8ed87f84a39e08521b4fdd6fb8a57b107b959210d`.
`<BASE>/input/lumina_cuda`와 `<SRC>/lumina_cuda` SHA256 = `ca721b07cb4fef632b1a0a0b2d3c7b11cdaac0e95f74e6cc56b131a7c5449059`.
이 identity의 `:3048–3144`에서 감쇠·문턱·100회·실패 경로를 재확인했다. 이온 분율은 `compute_ion_populations_shell(:2742)`, rate-SE 비는 `parity_rate_se_ratio_checked(:2689)`이며 bootstrap 분기는 별도다. 후보는 `:21052`의 n_e 반복 뒤 `:21063`에서 이온 분율을 재평가한다.
새 작업본은 이 봉인 상태의 별도 사본에서 만든다. 현재 로컬 미커밋 변경은 섞지 않는다. `<EVID>`는 운전석이 새로 지정할 절대 증거 루트이며, `<OFF>`·`<ON>`도 새 절대 런루트로 실행 전에 고정한다.

## 3. 런 설계 — 오프라인 선행, OFF 대조 1회 + ON 판정 1회

**O-0 입력 충분성:** `<EVID>/offline/state_manifest.json`에 bootstrap 및 SEED/R7 관련 전 셸의 상태 출처·시점·해시를 등록한다. θ는 T_e·밀도·조성·원자자료·분배함수·BF/복사장 generation·NT 입력·활성 env·분기 상태 등 F의 모든 의존성을 포함한다. 기존 입력으로 정확히 재구성 가능한 상태와 보존되지 않은 상태를 구분하며, 누락을 LTE 대체나 로그 맞춤 계수로 메우지 않는다.

**O-1 기전 특정·F′ 측정:** 생산 함수를 사용하는 사설 재생기에서 θ를 고정하고 n_e만 바꾼다. 매 평가 전 배열·캐시·카운터·bootstrap 상태를 복원하거나 격리 프로세스를 사용한다. 평가 순서를 뒤집어 F가 동일한지도 확인한다.
각 상태의 기존 마지막 x·채택점 m·독립 근 n*에서 `n±=n·exp(±h)`, `D_h=[F(n+)−F(n−)]/(n+−n−)`를 계산한다. `h={1e-2,1e-3,1e-4,1e-5,1e-6}` 전부와 좌·우 차분을 `<EVID>/offline/gain.csv`에 보존한다.
인접 h 두 쌍 이상에서 차분 간 차이가 `1e-4·max(1,abs(D_h))` 이내인 구간을 요구한다. 불안정·분기 전환은 `GAIN_UNRESOLVED`; F′의 부호나 `−1<F′≤0`을 가정하지 않는다. `(1+F′)/2`는 국소 감쇠 이득 측정치로만 보고하고, 실제 반복 궤적과 위 할선 항등식으로 유한 이동을 확인한다.

**O-2 수리안 검증:** 같은 고정 상태에서 `τ∈{5e-3,5e-4,5e-5,5e-6,5e-7,5e-8}`를 전부 재생한다. 별도 고정밀 구현이 동일 율·정규화 사다리로 F를 계산하고, 부호가 확인된 구간에서 전하중성 근 n*를 구한다. 구간은 수치 근 찾기용이며 물리적 cap이 아니다. 근 미확보·비유일성 미해결은 실패다.
수치 분해능 잣대 `ε=1e-6`은 기존 분류 문턱 `1e-3`보다 세 자릿수 작게 택한 **설계 기준이며 예측·물리 정확도 보증이 아니다**. 전 재생 상태에서 `abs(r_m)≤ε`, `abs(ln(m/n*))≤ε`, 100회 이내 성공을 만족하는 **가장 느슨한 τ를 τ***로 택한다. 후보가 없으면 해법·한도를 바꾸지 않고 `REWORK`다.

**O-3 기대치 봉인:** `<EVID>/offline/expectations.json`에 τ*·상태별 F′ 안정 구간·잔차·근 차이·반복 수·최대값의 셸/시점·입력/도구 해시를 기록하고 계약 개정으로 커밋한다. F′와 실제 궤적을 측정하기 전에는 τ*나 잔차 감소 배수를 확정하지 않는다. 자료 부족이면 `INPUT_PENDING`; 기대치 봉인·사전감사 PASS 전 GPU 런은 발주하지 않는다.

**R-0 OFF:** 새 바이너리·새 키 미설정, 기준은 **333531의 폐합 ON 상태**다. G-A 통과 후에만 R-1을 연다. 새 키 `0`의 동등성은 별도 프로세스 오프라인 배터리로 검증한다.
**R-1 ON:** OFF와 같은 바이너리·물리 exports에서 새 키 `τ*`만 추가한다. K42·ion1·T_e 핀·2반복·50셸을 계승한다. ON의 모든 n_e solve에 호출 ID·셸·θ identity·x·F(x)·m·F(m)·반복 수·상태를 기록하고, 공개 commit 장부와 연결한다. 탐침은 사설 상태에서 평가하며 production 상태를 바꾸지 않는다.
**비용·보존:** 오프라인은 grammar-debug 단일 CPU, 총 2 CPU-hour·16 GiB·신규 증거 10 GiB 상한; 초과 예상 시 중단 보고한다. 각 GPU 런은 A100×2·32 CPU·64 GiB·8시간, 합계 최대 32 GPU-hour이며 자동 재런은 없다. 신규 저장량은 기준 런 실측 크기와 계측 예상량을 합쳐 staging 전에 등록한다. 기존 봉인·산출물은 보존하며 rollback은 새 패치 비채택이다.

## 4. 기대치 — 구성상 도출과 측정의 구분

- **E-1 / 구성상 도출:** 성공한 ON solve의 채택 전 잔차 `<τ*`, 감쇠식 및 할선 항등식 성립. 작동 검사이며 물리 예측으로 세지 않는다. 기존 NLTE 폐합의 같은 합 재계산 residual도 이 부류다.
- **E-2 / 측정 기반 검증가설:** O-3에서 봉인한 수리안이 실제 ON의 전 solve에서도 채택점 잔차와 독립 근 차이를 각각 `≤ε`로 유지하는가. 코드는 이 두 조건으로 정지하지 않으므로 자동 보장이 아니다. 실패는 기대 불일치로 남기며 사후 τ 조정은 금지한다.
- **E-3 / 측정, 크기·부호 기대 없음:** 공개 commit의 Saha 잔차·NLTE 재분배·합계, 각 항의 우세 셸 수. solve와 commit 사이 θ가 달라졌으면 별도 상태로 판독한다.
- **E-4 / 측정, 크기·부호 기대 없음:** ON/OFF population 비와 formal 스펙트럼 대역별 중앙·최소·최대, 전역 최대 절댓값. 직전 +0.783%·−2.325%·0.4274%를 다음 응답의 상한으로 사용하지 않는다.

### 게이트 명부 (표)

| 게이트 | 판정 및 피연산자 | 실패 처분 |
|---|---|---|
| G-0 사전 준비 | `<EVID>/identity.json`이 §2 원본과 일치; `<EVID>/offline/state_manifest.json` 의존성 결손 0; `<EVID>/preflight.log`·`<EVID>/design_audit.md`에 PASS | 미집행/PENDING |
| G-1 오프라인 | `<EVID>/offline/gain.csv` 미해결 0; `<EVID>/offline/sweep.csv`와 `<EVID>/offline/oracle.csv`로 O-2 충족; `<EVID>/offline/expectations.json`이 선택 규칙·해시와 일치 | REWORK 또는 INPUT_PENDING |
| G-2 음성대조·격리 | `<EVID>/selftest/results.json`에서 성한 fixture PASS; 낡은 F(x)를 F(m)로 오기·전하종 누락·θ 변경·탐침 상태 누출·NaN·100회 실패 은폐·OFF 오진입·장부 결손 주입 각각 지정 실패; `<EVID>/selftest/off_bytes.json`에서 기존/미설정/0 출력 배열 byte-동일 | 코드 REWORK |
| G-A OFF 증분 | `<EVID>/baseline/epochs.txt` 대 `<EVID>/off/epochs.txt`의 epoch 3행 문자열 동일; 두 디렉터리 `pairs.csv`의 338짝 키와 n_upper_0·n_upper_1·Jbar_0·Jbar_1 문자열 동일; `closure.txt`의 150행 문자열 동일; `spectrum.csv`의 파장·flux 문자열 동일. 각각 `<BASE>`·`<OFF>`에서 동일 봉인 판독기로 추출; `<OFF>/stderr.log` 새 문턱 계측 0행 | ON 발주 금지 |
| G-B 단일 변수·봉인 | `<OFF>/input/resolved_lumina.exports` 대 `<ON>/input/resolved_lumina.exports`는 `<EVID>/path_map.json`에 열거한 경로 치환 후 새 키 τ* 추가만 허용; 양팔 `input/binary.sha256` 동일; 양팔 `SUPERLEVEL_B_STAGE_SEAL.txt`의 입력 digest는 `<BASE>/SUPERLEVEL_B_STAGE_SEAL.txt`와 동일 | 무효 |
| G-C staging 증거 | `<EVID>/staging/OFF.stdout.log`·`ON.stdout.log` 원문과 종료코드 보존; §2 기준 seal의 sealed_base 아래 `RII_RUN_SEAL.sha`, sealed_census 아래 `ZO_RUN_SEAL.sha`를 해당 루트에서 `sha256sum -c`하여 staging 전·완료 후 전건 OK; 양팔 seal shared_hardlinks=0 | 무효 |
| G-D 완주·커버리지 | 양팔 `model.rc`=0; `<EVID>/off/readout.json`·`on/readout.json` 오류 0·publication 2·registry 338 포함·provenance 전건 OK; `<EVID>/on/solve_index.csv`와 `solve.csv` 호출/셸 키 완전 일치; `<EVID>/on/closure.csv`는 SEED·R7 두 commit 각각 shell 0–49 완비 | 무효 |
| G-E 계측 진실성 | `<EVID>/on/solve.csv`의 원수치로 E-1 재계산; `<EVID>/on/replay.csv`가 같은 θ에서 F(m)를 재현; `<EVID>/on/oracle.csv`로 E-2 독립 계산; 누락·비유한·epoch 혼합 0. E-2 수치 초과 자체는 이 게이트 실패로 숨기지 않고 NT-R로 판정 | 증거 결함이면 무효 |

### 게이트 명부 끝

### 분기 (표)

| 분기 | 조건 | 뜻 |
|---|---|---|
| NT-X | G-0~G-E 중 실패·미충족 존재 | 무효 또는 미집행; INPUT_PENDING·AUDIT_PENDING·구현/집행/증거 실패를 명명 |
| NT-R | G 전건 PASS, `<EVID>/on/metrics.json`의 E-2 잔차 또는 근 차이 최대값 `>ε` | 오프라인 기대가 실제 상태로 전이되지 않음; 원인 분류 후 후속 결정 |
| NT-A | G 전건 PASS, 같은 경로의 두 최대값 모두 `≤ε` | 봉인 범위에서 문턱 수리의 수치 목적 충족; E-3·E-4는 별도 측정 보고 |

### 분기 끝

기계 분기 입력 `<EVID>/branch_tokens.json`은 각 게이트 FAIL/미충족→0, PASS→1, E-2 개별 측정 충족→1·초과→2로 변환한다. 결손은 0이며 빈 입력은 NT-X다. 변환과 실제 임계값 경계의 음성대조는 판독기 selftest가 담당하고, 아래 JSON은 분기의 배타성·완전성을 검사한다.

### 변경집합 — SAHA_NE_THRESH (표)

| 경로 | 변경 |
|---|---|
| `src/lumina_plasma.c` | 새 키 정지 문턱 선택·ON 전용 사설 계측/재생 접점; OFF 기존 연산·로그 유지; 감쇠·100회·물리 사상 불변 |
| `src/env_universe.h` | 전집 생성기로 새 키 등록 |
| `validation/instrumentation_debt/ENV_UNIVERSE.json` | 같은 전집 재생성 산출 |
| `tests/saha_ne_thresh_selftest.c` | 신설: OFF 별도 프로세스 byte 배터리·합성 F의 알려진 미분/근·실패 전파·탐침 격리 검사 |
| `scripts/saha_ne_thresh_replay.py` | 신설: 상태 검증·F′ 차분 분석·독립 고정밀 사다리/근·τ 선택·런 판독·분기 변환 및 음성대조 |
| `scripts/stage_superlevel_b.sh` | 새 OFF/ON 모드·바이너리/계약 identity·새 키 단일 차이·staging stdout 원문 보존; 봉인 내부 목적지 거부 유지 |

### 변경집합 끝

## 5. 주장 상한

NT-A는 **이 원자모형과 봉인된 실행 범위에서 정지 문턱의 수치 잔차를 제어했다**는 뜻이다. 전하 방정식의 물리적 완전성·CMFGEN 일치·수송/열평형 수렴·스펙트럼 오차 상한을 뜻하지 않는다. F′ 표본은 전역 수렴 증명이 아니며, 독립 근도 유일성이 확인된 범위에 한정한다. 기존 CPU χ/η·bf·열원장과 폐합 n_e의 시대 차이는 계승 미결이다.

## 6. 분장 장부

| 단계 | 규약상 담당 — 본 발주·개정21 | 실제 | 위반·독립성 |
|---|---|---|---|
| 사전등록 기안 | astra | **astra, 본 응답 저자** | 조회·identity 확인만 완료 |
| 코딩·직접 selftest·코드 검수 | astra | 미집행; 이후 본 저작의 구현·검수도 astra 담당 | 자기 코드 검수임을 명시 |
| 판정·마감감사 | astra | 미집행; 이후 본 계약의 판정·마감감사도 astra 담당 | 자기 계획 판정·감사이며 독립 모델 감사로 표기 금지 |
| 파일화·프리플라이트·빌드·staging·제출·판독·커밋 | 운전석 | 미집행 | 원문 로그와 재현 명령 보존 |
| 실측 주장 전건 재현·불일치 장부 | 운전석 | 미집행 | 동일 source/binary/input/evidence 해시와 결정적 출력으로 reproduced/not-reproduced/blocked 기록 |

재현 명령 인터페이스는 신설 판독기의 `--manifest <EVID>/offline/state_manifest.json --offline`, `--manifest <EVID>/identity.json --compare <BASE> <OFF> <ON>`, `--selftest`로 고정한다. 운전석은 실제 절대경로·실행 위치·도구 해시·명령·종료코드를 `<EVID>/reproduce.log`에 보존한다. 아직 실행하지 않은 검수·판정·감사·재현의 PASS를 이 기안에 기재하지 않는다.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 — SAHA_NE_THRESH (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "(?:src|tests|scripts|validation)/[A-Za-z0-9_/.]+\\.(?:c|h|py|sh|json)",
    "symbol": "SAHA_NE_THRESH",
    "roots": ["src", "tests", "scripts", "validation"],
    "expected_extra": ["src/lumina_plasma.c", "src/env_universe.h", "validation/instrumentation_debt/ENV_UNIVERSE.json", "tests/saha_ne_thresh_selftest.c", "scripts/saha_ne_thresh_replay.py", "scripts/stage_superlevel_b.sh"]
  },
  "branches": {
    "regimes": [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
    "metrics": {"g_fail": "sum(1 for x in v if x < 0.5)", "miss": "sum(1 for x in v if x > 1.5)", "count": "len(v)"},
    "rules": [
      {"name": "NT-X", "predicate": "count == 0 or g_fail > 0"},
      {"name": "NT-R", "predicate": "count > 0 and g_fail == 0 and miss > 0"},
      {"name": "NT-A", "predicate": "count > 0 and g_fail == 0 and miss == 0"}
    ],
    "adversarial_fixtures": [
      {"name": "all-pass", "mix": [[1.0, 10]]},
      {"name": "expectation-miss", "mix": [[1.0, 9], [2.0, 1]]},
      {"name": "invalid-before-miss", "mix": [[0.0, 1], [2.0, 1]]},
      {"name": "missing-input", "mix": [[0.0, 1]]},
      {"name": "empty", "mix": []}
    ],
    "residual": "NT-X"
  },
  "references": [
    {"path": "src/lumina_plasma.c", "flags_existing": ["compute_electron_density", "compute_ion_populations_shell", "parity_rate_se_ratio_checked", "NE_FAIL"]},
    {"path": "scripts/derive_env_universe.py", "flags_existing": ["getenv"]},
    {"path": "scripts/check_gate_referents.py", "flags_existing": ["OPERAND_VAGUE"]},
    {"path": "docs/PREREG_DC1_NE_CLOSURE_2026-09-06.md"},
    {"path": "docs/VERDICT_DC1_NECLOSE_R2_2026-09-08.md"},
    {"path": "docs/VERIFICATION_REGISTERS.md"},
    {"path": "docs/CLASSIC_DEBT_CENSUS.md"},
    {"path": "CLAUDE.md"}
  ]
}
```