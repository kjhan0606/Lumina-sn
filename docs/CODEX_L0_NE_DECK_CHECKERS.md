# NE-NAMING · DECK-FOSSIL checker 구현·음성대조 인수 보고

- 작성일: 2026-08-05 KST
- 처분: NE-NAMING **A**, DECK-FOSSIL **approved fossil quarantine + read-only legacy**
- 실행 상태: **PENDING_DRIVER_EXECUTION**. 이 세션은 로그인 노드에서 Python/checker/회귀를
  실행하지 않았다. `rg`, `stat`, `sha256sum`, diff 판독만 했으며 아래 실제 rc는
  grammar-debug 운전석이 확정한다.
- 공통 rc: PASS/WARN=0, FATAL=1. 두 checker는 서로 import하지 않으며 각자 단독 실행된다.
- 불변 범위: `src/`, 모든 `data/tardis_reference_*`, `/gpfs/kjhan/cmfgen_runs/**`는 수정하지
  않았고 덱 재생성·commit·push를 하지 않았다.

---

## 1. NE-NAMING checker — 처분 A

### 1.1 신규·수정 파일과 의존성

| 파일 | 종류 | 구현 |
|---|---|---|
| `scripts/build_toy06_epoch.py:90` | 수정 | 참값 경로 schema만 명세 |
| `scripts/build_toy06_epoch.py:114` | 수정 | placeholder provenance/승인을 `tau_i/i_phot`보다 먼저 검사 |
| `scripts/build_toy06_epoch.py:345` | 수정 | mode·수식·zone·builder/input hash·`tau_phot`·처분과 경계 재현량 manifest 기록 |
| `scripts/check_ne_naming.py:1` | 신규 | read-only 독립 checker |
| `scripts/run_ne_naming_controls.py:1` | 신규 | §3.4 FATAL 4 + WARN 1 독립 배터리 |
| `docs/manifests/ne_naming_toy06_19p48d_legacy.json:1` | 신규 | 현 legacy 덱 exact-hash 승인/처분 A seal |
| `docs/CODEX_L0_NFP_CONFIG_PREC.md:123` | 수정 | 오염 수치 근거 제거 및 구현 상태 갱신 |

checker와 runner는 Python 표준 라이브러리에만 의존한다. runner는 checker를 import하지 않고
별도 child process로 호출한다. builder만 기존과 같이 NumPy/Pandas에 의존하며 DECK checker나
atomic writer를 import하지 않는다.

### 1.2 manifest schema와 판정 순서

`lumina.ne-naming/v1`의 필수 항목은 다음과 같다.

| schema 묶음 | 필드 |
|---|---|
| mode/근사 | `electron_density_mode`, `formula`, `applicable_zones` |
| provenance | `builder.path`, `builder.sha256`, `builder.producer_status`, 각 `inputs.*.sha256` |
| 물리 좌표 | `source.epoch_days`, composition/frame/units, `tau_phot`, `sigma_T_cm2` |
| 승인 | `approved_disposition`, `approval.token`, `approval.scope`, legacy exact hash |
| generation | `generation_id`, 4 companion의 SHA-256와 generation ID |
| 재현 | zone별 `Zbar_s=n_e/n_atom`, `tau_i`, `i_phot`, `v_inner`, `r_inner`, `tau_total` |
| 미래 mode | `CMFGEN_CHARGE_BALANCE`와 RVTJ/epoch/unit/ND/interpolation/duplicate/non-monotonic/coverage/outside-grid 명세 |

builder의 기본 `production` 호출은 승인 토큰이 없으므로
`scripts/build_toy06_epoch.py:251`에서 멈춘다. 이는 `tau_i`가 처음 계산되는 `:255`보다
앞이다. 토큰이 있어도 처분 A에서는 production/canonical placeholder 출력이 차단되고,
오직 scratch `diagnostic`만 WARN으로 열린다. 참값 mode는 값 공간과 manifest 명세만 있으며
CMFGEN 읽기·보간·덱 생성 코드는 구현하지 않았다.

checker도 같은 순서다. mode/승인/provenance/epoch/generation을 먼저 판정한 후에만
`scripts/check_ne_naming.py:263`의 재현 함수가 호출된다. 이 함수는 덱에서 zone별 `Zbar_s`와
계약 그대로의 trapezoid `tau_i`, `i_phot=max{i:tau_i>=tau_phot}`를 로그에 남긴다. 영향
크기는 `UNQUANTIFIED_PENDING_CLEAN_ZBAR`; 확정 증거는 case A의 `3900 km/s`가
`config.json:v_inner_min_cm_s=3.9e8`과 정확히 같은 경계 사슬의 존재뿐이다.

### 1.3 운전석 실행 명령과 기대 rc

음성대조 5종 전체:

```bash
ssh grammar "ssh grammar-debug 'set -euo pipefail; cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn; env PYTHONPYCACHEPREFIX=/tmp/lumina_ne_naming_pycache python3 -m py_compile scripts/check_ne_naming.py scripts/run_ne_naming_controls.py scripts/build_toy06_epoch.py; python3 scripts/run_ne_naming_controls.py | tee /tmp/lumina_ne_naming_controls.txt'"
```

- 기대 최종 rc: **0**.
- 기대 summary: `NE_NAMING_CONTROL_SUMMARY passed=5 total=5`.
- child rc: FATAL 네 건은 1, 승인 legacy 한 건은 0.

승인된 현 legacy 덱 단독 양성 대조:

```bash
ssh grammar "ssh grammar-debug 'set -euo pipefail; cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn; python3 scripts/check_ne_naming.py --deck data/tardis_reference_toy06_19p48d --manifest docs/manifests/ne_naming_toy06_19p48d_legacy.json --claim legacy-read-only --approval-token NE-NAMING-A-LEGACY-READONLY-2026-08-05 | tee /tmp/lumina_ne_naming_legacy.txt'"
```

- 기대 rc: **0**.
- 기대 marker: `[NE-NAMING][WARN]`와
  `impact=UNQUANTIFIED_PENDING_CLEAN_ZBAR`.

기본 production fail-closed 직접 대조(명령 자체의 기대 rc는 1):

```bash
ssh grammar "ssh grammar-debug 'cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn; python3 scripts/check_ne_naming.py --deck data/tardis_reference_toy06_19p48d --manifest docs/manifests/ne_naming_toy06_19p48d_legacy.json > /tmp/lumina_ne_naming_production_fatal.txt 2>&1'"
```

- 기대 rc: **1**.
- 기대 marker: `[NE-NAMING][FATAL] unapproved placeholder`.

### 1.4 §3.4 음성대조 표

| 주입 | 기대 marker | 기대 child rc |
|---|---|---:|
| scratch manifest에서 `electron_density_mode` 삭제 | `NE-NAMING][FATAL] missing mode` | 1 |
| placeholder + production, 승인 토큰 미제공 | `NE-NAMING][FATAL] unapproved placeholder` | 1 |
| CMFGEN 참값 manifest epoch를 18.0 d로 변경 | `NE-NAMING][FATAL] epoch mismatch` | 1 |
| 새 `electron_densities.csv` generation과 구 companion 혼합 | `NE-NAMING][FATAL] generation mismatch` | 1 |
| exact legacy hash + placeholder + 승인 토큰 | `NE-NAMING][WARN]` | 0 |

fixture root는 `scripts/run_ne_naming_controls.py:127`에서 반드시
`/tmp/lumina_ne_naming_controls_*`로 만들어지며 원본은 `copy2`로만 읽는다.

### 1.5 양성 대조·폐합조건·남은 위험

- 양성 대조: exact 4-hash seal, `PLACEHOLDER_ZBAR_ONE`, 처분 A, 명시적 read-only token이
  모두 맞을 때만 WARN rc=0이다. PASS나 canonical 승격이 아니다.
- 현재 상태: 처분 A의 mode/provenance, builder production 차단, hash seal, checker/fixture는
  구현 완료. §3.4 실제 5/5와 회귀는 `PENDING_DRIVER_EXECUTION`이다.
- DECK/CONFIG가 새 generation 하나를 가리키는 B 경로는 의도적으로 열지 않았다.
- 남은 위험: clean `<Z>`가 없어 경계 영향 크기는 미정량이며, legacy WARN을 수집하지 않는
  launcher는 결함을 놓칠 수 있다.

---

## 2. DECK-FOSSIL checker — fossil quarantine 경로

### 2.1 신규 파일과 의존성

| 파일 | 종류 | 구현 |
|---|---|---|
| `scripts/check_deck_fossil.py:1` | 신규 | fossil read-only + canonical scratch replay 독립 checker |
| `scripts/deck_generation_atomic.py:1` | 신규 | sibling 임시 디렉터리 생성·검증·단일 rename commit |
| `scripts/run_deck_fossil_controls.py:1` | 신규 | §4.4 FATAL 4 + WARN 1 독립 배터리 |
| `docs/manifests/deck_fossil_toy06_19p48d_quarantine.json:1` | 신규 | approved exact-hash fossil quarantine |

세 파일은 Python 표준 라이브러리만 사용한다. DECK checker는 NE checker/builder를 import하지
않는다. fossil mode는 writer를 전혀 호출하지 않고, generation manifest를 심사할 때만 등록
writer를 `/tmp/lumina_deck_fossil_controls_replay_*`에 재실행한다. §4.4 runner의 다섯 case는
모두 replay 전 gate에서 판정되므로 atomic writer나 물리 builder를 실행하지 않는다.

### 2.2 generation manifest schema와 atomic writer

`lumina.deck-generation/v1` schema:

| schema 묶음 | 필드 |
|---|---|
| writer | `producer.writer.path/sha256`, 실제 전체 `argv`, replay argv template, 정확한 전체 declared `environment`, working directory, registration SHA-256 |
| 입력·물리 | 모든 `inputs.*.sha256`, epoch value/unit, `constants`, `units` |
| config | `time_explosion_s`, `T_inner_K`, `luminosity_inner_erg_s`, `n_shells`, `v_inner_min_cm_s`, `v_outer_max_cm_s` |
| companion | config/geometry/electron/plasma 각각 SHA-256 + 동일 generation ID |
| transaction | generation ID, `started_at`, `committed_at`, `atomic_commit=true` |
| replay | `L`, `r_inner`, `T_inner`, W/T_rad/n_e profile hash, `R_L`, `epsilon_L`, `Delta_SB_K`, 합격선 |

atomic writer는 `scripts/deck_generation_atomic.py:305`에서 target과 같은 filesystem의 sibling
임시 디렉터리를 만들고, writer가 그 아래 전체 generation을 쓰게 한다. 4 companion,
shell 수, 여섯 config key, epoch, 단위/상수, `epsilon_L<=1e-6`, `Delta_SB<=5 K`를 검증하고
manifest까지 fsync한 뒤 `:331`의 directory rename 한 번으로 commit한다. 현 canonical 경로와
그 하위는 `:150`에서 영구 거부한다. 이번 작업에서는 이 writer로 덱을 생성하거나 commit하지
않았다. 이후 canonical checker는 `scripts/check_deck_fossil.py:245`에서 attested argv/env/cwd를
scratch에 재실행하고 실제 replay의 `L`, `r_inner`, `T_inner`, W/T_rad/n_e를 target과 대조한다.

### 2.3 fossil quarantine 레코드

현 덱은 `producer=UNRESOLVED`, `canonical_production_eligible=false`, 허용 mode는
`legacy-read-only` 하나다. 봉인 hash는 다음과 같다.

| companion | exact SHA-256 |
|---|---|
| `config.json` | `cf61ab7c880243ffa94bba95b55c3bb4c88e526bcdf1d9b76bd81f44ff81293b` |
| `geometry.csv` | `21bb9349c11bceb7c815ca6fd5b21a647bddf1ba3f3da4311b97da93dc3ce3d6` |
| `electron_densities.csv` | `a8288513ad7b11a0a71b897849e9e93c900f211db481a12f6f79e984e2afe457` |
| `plasma_state.csv` | `45ccb86be1296e644f8491eb9b1346b2b5746f96055003d7f7009082a336a6f4` |

레코드는 epoch 오선택, 3557 Å 파장절단, `1/4=0.249686` 맞춤 상수, git `UNTRACKED`
가설을 모두 `REJECTED`로 남긴다. `R_L=4.005038`, `epsilon_L=3.005038`, 내부
`Delta_SB=1.65 K`를 함께 기록하며, 내부 정합은 generation 재현성을 대신하지 못한다고
명시한다. 이 수치들은 quarantine 증거이지 writer 코드의 상수나 보정값이 아니다.

### 2.4 운전석 실행 명령과 기대 rc

음성대조 5종 전체:

```bash
ssh grammar "ssh grammar-debug 'set -euo pipefail; cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn; env PYTHONPYCACHEPREFIX=/tmp/lumina_deck_fossil_pycache python3 -m py_compile scripts/check_deck_fossil.py scripts/run_deck_fossil_controls.py scripts/deck_generation_atomic.py; python3 scripts/run_deck_fossil_controls.py | tee /tmp/lumina_deck_fossil_controls.txt'"
```

- 기대 최종 rc: **0**.
- 기대 summary: `DECK_FOSSIL_CONTROL_SUMMARY passed=5 total=5`.
- child rc: FATAL 네 건은 1, fossil legacy 한 건은 0.

승인 fossil read-only 양성 대조:

```bash
ssh grammar "ssh grammar-debug 'set -euo pipefail; cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn; python3 scripts/check_deck_fossil.py --deck data/tardis_reference_toy06_19p48d --mode legacy-read-only --quarantine docs/manifests/deck_fossil_toy06_19p48d_quarantine.json | tee /tmp/lumina_deck_fossil_legacy.txt'"
```

- 기대 rc: **0**.
- 기대 marker: `[DECK-FOSSIL][WARN] producer=UNRESOLVED mode=legacy-read-only`.

같은 fossil의 무조건 canonical 주장 대조(명령 자체의 기대 rc는 1):

```bash
ssh grammar "ssh grammar-debug 'cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn; python3 scripts/check_deck_fossil.py --deck data/tardis_reference_toy06_19p48d > /tmp/lumina_deck_fossil_canonical_fatal.txt 2>&1'"
```

- 기대 rc: **1**.
- 기대 marker: `[DECK-FOSSIL][FATAL] missing manifest`.

### 2.5 §4.4 음성대조 표

| 주입 | 기대 marker | 기대 child rc |
|---|---|---:|
| scratch generation manifest 삭제 | `DECK-FOSSIL][FATAL] missing manifest` | 1 |
| config만 다른 generation으로 교체 | `DECK-FOSSIL][FATAL] generation mismatch` | 1 |
| `plasma_state.csv` 끝에 한 바이트 추가 | `DECK-FOSSIL][FATAL] companion hash mismatch` | 1 |
| registered manifest argv 하나 추가, attestation 유지 | `DECK-FOSSIL][FATAL] writer replay mismatch` | 1 |
| exact fossil hash + read-only legacy mode | `DECK-FOSSIL][WARN]` | 0 |

fixture root는 `scripts/run_deck_fossil_controls.py:222`에서 반드시
`/tmp/lumina_deck_fossil_controls_*`로 만들어진다.

### 2.6 양성 대조·폐합조건·남은 위험

- 양성 대조: 네 exact hash와 approved quarantine가 맞는 read-only legacy 소비만 WARN rc=0.
  캠페인을 멈추지 않되 매번 경고하고 무조건 canonical 주장은 FATAL이다.
- 현재 상태: producer 회복 불능/`UNRESOLVED` 처분과 quarantine 경로, generation schema,
  atomic writer, checker/fixture 구현 완료. 실제 §4.4 5/5는 `PENDING_DRIVER_EXECUTION`이다.
- 남은 위험: quarantine는 과거 producer를 복구하지 않는다. 새 canonical generation을 만들
  때는 atomic writer를 실제 등록 writer와 함께 별도 발주·검증해야 한다. 현 fossil은 영구히
  canonical production seed 자격이 없다.

---

## 3. 독립성 자기검수

| 항목 | NE-NAMING | DECK-FOSSIL |
|---|---|---|
| checker entry point | `python3 scripts/check_ne_naming.py ...` | `python3 scripts/check_deck_fossil.py ...` |
| battery entry point | `python3 scripts/run_ne_naming_controls.py` | `python3 scripts/run_deck_fossil_controls.py` |
| 다른 checker import | 없음 | 없음 |
| fixture prefix | `/tmp/lumina_ne_naming_controls_*` | `/tmp/lumina_deck_fossil_controls_*` |
| 양성 판정 | WARN rc=0 | WARN rc=0 |
| FATAL 판정 | rc=1 | rc=1 |

marker 문자열과 기대 rc는 각 runner의 case table(`run_ne_naming_controls.py:131-167`,
`run_deck_fossil_controls.py:226-262`)에서 계약 표와 문자 단위로 대조했다. Python 실행 전
정적 검수 상태이며 실제 child stdout/rc는 운전석 로그가 확정한다.

---

## 4. 회귀 인수조건과 불변 확인

checker 음성대조 뒤 아래 네 회귀가 모두 통과해야 인수한다.

```bash
ssh grammar "ssh grammar-debug 'set -euo pipefail; cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn; ~/.lumina_scratch/run_dbuild_gates.sh | tee /tmp/lumina_ne_deck_dk.txt; bash scripts/run_zinert_selftest.sh | tee /tmp/lumina_ne_deck_zinert.txt; ~/.lumina_scratch/run_config_prec.sh | tee /tmp/lumina_ne_deck_config_prec.txt; ~/.lumina_scratch/run_cls_verify.sh | tee /tmp/lumina_ne_deck_cls.txt'"
```

기대 최종 rc=0이며 인수 문자열은 D **19/19**, K **7/7**, **Z-INERT**,
CONFIG-PREC **7/7**, 분류기 jnu4/modern/음성 **7/7**이다.

금지 대상 최종 확인 명령:

```bash
git status --short -- data/
```

이 세션의 실제 출력:

```text
(empty)
```

4개 정본 companion의 size/mtime은 작업 전후 각각 동일해야 하며 기준은 다음과 같다.

| 파일 | bytes | mtime KST |
|---|---:|---|
| `config.json` | 396 | `2026-06-29 19:19:58.456639984 +0900` |
| `geometry.csv` | 3359 | `2026-06-29 14:54:10.077580202 +0900` |
| `electron_densities.csv` | 1073 | `2026-06-29 14:54:10.079136800 +0900` |
| `plasma_state.csv` | 2113 | `2026-06-29 14:54:10.081294706 +0900` |

최종 정적 검수에서 `git diff --check`는 출력 없이 끝났고 위 size/mtime과 SHA-256도 작업 전
기준과 같았다. checker/회귀 PASS는 grammar-debug 운전석 결과가 들어오기 전에는 주장하지
않는다.
