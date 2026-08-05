# A2-04 구현 보고 — 단일 producer commit과 L-0 replay

- 구현일: 2026-08-06
- 기준 커밋: `01630604c0a643774ff32add952d20108d1e4303`
- 이 보고서의 source bundle SHA-256: `37a4522ed6aab0703655e31ff47c3c2b179542c39e6ae6432cdbfc49214cf6c0`
- 범위: 기존 트리에 실재하는 A2-04 구현·selftest·음성 대조를 문서화했다. 이
  세션에서는 `docs/CODEX_IMPL_A2_04.md` 외의 파일을 수정하지 않았다.
- 판정: 공통 commit API, MC/pure-CMFGEN 두 CPU 호출자, 원자적 후보 검증,
  synthetic L-0 wiring replay, dilute-Planck 5대역 음성 대조, classic-debt sweep은
  PASS다. 실자료 `EDDFACTOR` 운전석 산출물은 트리에 없으므로
  `PENDING_DRIVER_EXECUTION`이며 최종 L-0 production PASS나 운전석 서명을 주장하지
  않는다.

## 1. commit API 시그니처와 호출자

정본 요청 구조체는 `src/radiation_field.h:155-178`, 공개 API 선언은
`src/radiation_field.h:198-199`, 구현은 `src/radiation_field.c:402-403`이다.

```c
int radiation_field_commit(
    RadiationFieldOwner *owner,
    const RadiationFieldCommitRequest *request);
```

요청은 producer/provenance, generation/epoch/shell geometry와 다음 두 입력 형식 중
정확히 하나를 전달한다.

- MC: raw path length, count, volume, simulation time, canonical 4000-bin shape
- deterministic: source edge, source bin-average `J_nu`, validity, source bin 수

현재 production CPU caller는 compiler callgraph로 확인한 아래 둘뿐이다.

| caller | 호출 위치 | 역할 |
|---|---|---|
| `main` | `src/lumina_main.c:485` | MC raw/count를 `4*pi*V*dt*Delta_nu`로 한 번만 정규화해 commit |
| `cmfgen_commit_jnu` | `src/lumina_cmfgen.c:3434` | pure-CMFGEN source grid와 validity를 보존 재빈 commit |

pure-CMFGEN wrapper의 선언은 `src/lumina_cmfgen.h:263-264`, 구현은
`src/lumina_cmfgen.c:3390-3442`, 실제 solver-loop 호출은
`src/lumina_cmfgen.c:5122`다. MC 작업 버퍼의 시작·thread-local 생성·reduce는 각각
`src/lumina_main.c:396`, `:427`, `:477`; 같은 공이동 path-length 항의 생산은
`src/lumina_transport.c:117-120`이다.

`scripts/a2_04_commit_callgraph.py:70-145`가 8개 production translation unit을 GCC
`-fdump-ipa-cgraph`로 확인했다. 관측값은 commit caller
`[cmfgen_commit_jnu, main]`, canonical J/generation/validity/count의 owner module 밖
writer 각 0, canonical physics consumer 0, Planck-fit→canonical commit edge 0이다.

## 2. commit 동작과 generation 원자성

owner는 gate 없이 항상 할당·활성화된다(`src/radiation_field.c:26-85`,
`src/lumina_main.c:281-287`). commit은 다음 순서다.

1. shell 수, 다음 generation, geometry, producer, 허용 provenance를 먼저 거부한다
   (`src/radiation_field.c:405-414`).
2. 공개 field와 분리된 후보 value/validity/count를 할당한다(`:416-424`).
3. MC와 deterministic 입력 형식을 서로 배타적으로 검사하고 후보를 만든다
   (`:254-369`).
4. 후보 전체의 finite/nonnegative/value-validity-count 불변식을 검사한다
   (`:372-400`).
5. 검사 완료 뒤에만 J, validity, count, provenance, required/computed generation을
   한 choke point에서 공개한다(`:442-468`). 새 field generation이 전진할 때 아직
   비어 있는 `LineJbarCache`는 required만 같은 값으로 전진하고 computed=0으로 남는다
   (`:464-468`). 실제 line-cache 생산은 A2-06이다.

관측 selftest marker와 정상 rc는 다음과 같다.

```text
A2_04_COMMIT_SELFTEST PASS common_callers=MC,CMFGEN negative_1=PASS negative_4=PASS negative_7=PASS unsampled_floor=0 out_of_grid=EXPLICIT generation_atomic=PASS
```

failure injection은 Planck provenance overwrite, generation gap, raw와 normalized 입력을
동시에 준 double-normalization 형식을 각각 거부하고 공개 J/validity/count/generation의
byte snapshot이 유지됨을 검사한다(`tests/a2_04_commit_selftest.c:82-97,149-157`).

## 3. 제거·차단 경로 목록

| 경로 | A2-04 처분 | 근거 |
|---|---|---|
| production shadow ON/OFF gate | **제거**. owner는 항상 enabled | `src/radiation_field.c:26-85`; production에서 `radiation_field_shadow_gate_enabled`와 `LUMINA_RADFIELD_SHADOW` hit 0 |
| 별도 shadow init/begin/commit/free API | **제거·통합**. owner lifecycle와 공통 commit으로 교체 | `src/radiation_field.h:184-201`; `src/lumina_main.c:286,350,396,485,862` |
| pure-CMFGEN loop의 직접 `cmfgen_write_jnu()` 공개 | **교체**. loop는 wrapper를 호출하고 wrapper가 먼저 canonical commit | `src/lumina_cmfgen.c:5122-5127` |
| legacy compatibility `cs.J -> nlte->J_nu` | **잔류**. canonical commit 성공 뒤 pre-A2-05 소비자용 복사 | `src/lumina_cmfgen.c:3438-3440` |
| dilute-Planck `(W,T_R)`의 canonical overwrite | **차단**. 허용 provenance는 MC와 CMFGEN replay뿐 | `src/radiation_field.c:412-414`; 동적 거부 `tests/a2_04_commit_selftest.c:149-157`; call edge 0 `scripts/a2_04_commit_callgraph.py:96-103` |
| generation skip/split publication | **차단** | `src/radiation_field.c:407-414`; 음성 대조 `tests/a2_04_commit_selftest.c:82-88` |
| raw estimator double normalization | **차단**. MC 요청에 source J/edge/validity가 섞이면 거부 | `src/radiation_field.c:259-266`; 음성 대조 `tests/a2_04_commit_selftest.c:90-97` |
| unsampled 또는 out-of-grid에 작은 양수 삽입 | **canonical에서 차단**. 값 0과 별도 validity를 요구 | `src/radiation_field.c:216-227,277-287,317-323,355-365` |
| BF의 `bf_rate_estimator` 우회, stale `jbar_line/j_blue` | **A2-04에서 구조적으로 격리, 미제거**. canonical read API가 아직 없음 | `scripts/a2_04_commit_callgraph.py:105-143`; A2-05/A2-06 인계 |
| canonical GPU mirror/upload/reset | **미도입**. 따라서 §13 경로 5·6은 A2-12 전에는 도달 불가 | `src/radiation_field.h:147-154`; `scripts/a2_04_commit_callgraph.py:139-142` |

`RadiationFieldShadow` typedef와 `LUMINA_RADFIELD_SHADOW` 문자열은 A2-03 fixture의
source compatibility 때문에 테스트·스크립트에 남아 있다
(`src/radiation_field.h:147-149`). production gate는 아니다. dump의 과거 환경변수
`LUMINA_RADFIELD_SHADOW_DUMP`도 진단 호환 fallback으로만 남았다
(`src/radiation_field.c:475-480`).

## 4. `1e-30` floor의 정확한 처분

결론은 **전역 즉시 제거가 아니라, A2-03 shadow에서 승격된 canonical owner 쪽만 즉시
제거**다.

- canonical MC의 count=0은 `J_nu=0`과 `UNSAMPLED`다
  (`src/radiation_field.c:272-282`). sampled raw exact-zero는 값 0과 `EXACT_ZERO`다
  (`:280-287`).
- deterministic source의 out-of-grid는 값 0과 `OUT_OF_GRID`, unavailable은 값 0과
  `UNSAMPLED`다(`src/radiation_field.c:312-365`).
- owner validator는 unsampled/out-of-grid의 nonzero 값, 즉 `1e-30` 삽입을 거부한다
  (`src/radiation_field.c:210-234`). A2-03 음성 대조 11도 계속 PASS다.
- 그러나 기존 1000-bin `nlte_normalize_j_nu()`는 raw가 없으면 여전히
  `nlte->J_nu[idx] = 1e-30`을 쓴다(`src/lumina_plasma.c:14672-14688`).
  `nlte_build_perbin_dilute_field()`도 legacy 배열을 직접 갱신한다
  (`src/lumina_plasma.c:1436-1445`).

정적 census에는 legacy `J_nu` 직접 소비 위치가 40개 남아 있다
(`scripts/a2_04_commit_callgraph.py:105-133`). canonical 소비 API가 없는 A2-04에서 이
floor만 제거하면 기존 rate/opacity/emissivity 결과를 바꾸므로 이전 세션은
`DEFER_LEGACY_REMOVAL_TO_A2_05_BECAUSE_ACTIVE_CONSUMERS_EXIST`를 선택했다. 따라서
A2-03 인계의 “A2-04에서 legacy floor overwrite 제거”를 전역 제거로 해석하면 이
요구는 **미완 결함**이다. 이 보고서 세션에서는 고치지 않았다.

## 5. L-0 replay와 완료된 음성 대조

`scripts/a2_04_l0_replay.py`는 CMFGEN `EDDFACTOR`의 196,185 frequency records를
RVTJ velocity에 매핑하고 fine grid에서 canonical edge로 piecewise-linear 보존 적분한다
(`:140-176`). s0-s43만 wiring replay 범위이며 현재 snapshot에서 물리적으로 안전하게
등록된 범위는 s0-s8이다(`:331-381`). fixture는 canonical input을 실제 C commit API에
통과시켜 generation/J/validity를 다시 읽는다(`:260-287`,
`tests/a2_04_replay_commit.c:55-92`).

완료된 synthetic wiring replay는 9개 셸, 4000 bins에서 다음을 관측했다.

```text
max E_1             = 1.0670693551008627e-17
max band E_B        = 7.678525915832037e-18
max P95 log10 dex   = 9.6432746655328700e-17
commit marker       = A2_04_REPLAY_COMMIT PASS shells=9 bins=4000 generation=1
guard/fallback hits = 0/0
```

같은 실행의 dilute-Planck 음성 대조는 deck s0의
`W=0.2978587261676735`, `T_color=14172.549003071521 K`를 주입했다
(`scripts/a2_04_l0_replay.py:290-320`). 다섯 대역 모두 사전 기준 10%를 실패했다.

| 대역 | 관측 `E_B` | 기대 |
|---|---:|---|
| EUV 450-918 A | 0.9973461023 | FAIL > 0.10 |
| FUV 918-1290 A | 0.9196778266 | FAIL > 0.10 |
| UV 1290-2000 A | 0.4955145707 | FAIL > 0.10 |
| OPT 2000-10000 A | 0.3087160539 | FAIL > 0.10 |
| IR 10000-25000 A | 0.5837358479 | FAIL > 0.10 |

음성 대조의 기대 실패를 관측했기 때문에 전체 verifier rc는 0이다. 이 수치는 synthetic
positive field에 대한 음성 대조이며 실자료 EDDFACTOR 수치를 가장하지 않는다.

CMFGEN snapshot manifest의 입력 결박은 다음과 같다.

| 파일 | SHA-256 | 크기/자격 |
|---|---|---|
| `EDDFACTOR` | `83acc14a35999aaf39cf728ce783308be31fa52676b1c5410b5e84f4cc009705` | 142,832,872 B; FINISH_REC=1 |
| `EDDFACTOR_INFO` | `2c032445a9483d5154c15cdac5c0f14dfbb3f45dbb628e2d4936c29b3efabd42` | 284 B |
| `RVTJ` | `a042fd49c726dc1c2b710c997fa3d27780189e98edebc28f9d77c06ffe034f78` | 604,183 B |

근거 manifest는 `validation/a2_00_oracle/toy06_19.48d_jnu4.manifest.json`이며 그
SHA-256은 `ede416159ec129699424f5771d78a532ab04d4a399741599dbb60402f49b0e37`다.
`FINISH_REC=1`은 replay 파일 완결만 뜻한다. 이 snapshot은
`CMFGEN_SNAPSHOT_REPLAY=ELIGIBLE`, `CMFGEN_PHYSICAL_ORACLE=INELIGIBLE`이다.

## 6. classic sweep 실측표

정책은 `MEASURE_AND_RECORD_ONLY_NO_REPAIR`다. sweep은
`logs/ddc15_radeqesc_163604/stdout.log`와 현 source/deck를 읽었고 census 문서를
수정하지 않았다(`scripts/a2_04_classic_debt_sweep.py:27-115`).

| ID | reachability·현재 위치 | 실측 | 영향과 처분 |
|---|---|---|---|
| H02 | FIRED; `src/lumina_main.c:107`, `src/lumina_cuda.cu:6905` | 기본 damping 0.5, hold 3; archived damping-on 6회/off 1회 | armed update는 old value 50% 유지. A2-04 무수리, A2-18로 OPEN |
| H13 | FIRED; `src/lumina_plasma.c:1030` | 1500-50000 K, coarse 80/refine 60; binned-J banner 1, outer iter 10 | bounded fit은 10회 시도됐으나 canonical commit edge 0. 진단 영향만 남기고 OPEN |
| S01 | FIRED_LEGACY_CANONICAL_BLOCKED; `src/lumina_plasma.c:847`, `src/lumina_main.c:534` | archived radiation-field solve 10회; canonical Planck commit edge 0 | legacy W/T_rad와 pre-A2-17 소비자는 변하지만 canonical bytes overwrite는 차단. A2-17로 잔류 |
| P02 | FIRED_BY_CONSTRUCTION; `src/lumina_main.c:420,478` | worker당 제외 line 2,565,342; local n_lines=0; local j_blue/Edotlu reduction 0 | CPU line-resonance estimator가 thread-local reduction에서 빠짐. A2-06으로 OPEN |

classic sweep verdict는 `PASS_MEASURED_NO_REPAIR`다. `docs/CLASSIC_DEBT_CENSUS.md`의
기존 행번호는 A2-04 source 이동 전 값일 수 있으므로 위 표는 이 보고서 작성 직전
`nl -ba`와 sweep이 재확인한 현재 줄번호를 사용했다.

## 7. lageunha 운전석 명령 — 실자료 L-0와 음성 대조

아래 블록은 그대로 복사할 수 있다. 실제 `EDDFACTOR/INFO/RVTJ` hash/schema를 먼저
검사하고, 공통 commit selftest/callgraph 뒤 실자료 44-shell L-0 replay와 그 실자료에
대한 dilute-Planck 5대역 음성 대조를 한 실행에서 수행한다. 전체 블록과 각 하위 명령의
기대 rc는 **0**이다.

```bash
ssh lageunha 'bash -s' <<'EOF' 2>&1 | tee /tmp/a2_04_lageunha.log
set -euo pipefail
REPO=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
CMF=/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4
OUT="$REPO/validation/a2_04"
MANIFEST="$REPO/validation/a2_00_oracle/toy06_19.48d_jnu4.manifest.json"
cd "$REPO"
mkdir -p "$OUT"

PYTHONDONTWRITEBYTECODE=1 python3 scripts/cmfgen_oracle_contract.py check \
  "$CMF" --manifest "$MANIFEST" --profile snapshot
make selftest_a2_04_commit selftest_a2_04_replay_commit
./selftest_a2_04_commit
PYTHONDONTWRITEBYTECODE=1 python3 scripts/a2_04_commit_callgraph.py \
  | tee "$OUT/a2_04_commit_callgraph.json"
PYTHONDONTWRITEBYTECODE=1 python3 scripts/a2_04_l0_replay.py \
  --fixture ./selftest_a2_04_replay_commit \
  --deck data/tardis_reference_toy06_19p48d \
  --cmf-dir "$CMF" --scratch-root /tmp \
  --output "$OUT/a2_04_l0_replay_eddfactor.json" \
  | tee "$OUT/a2_04_l0_replay_eddfactor.stdout.json"
PYTHONDONTWRITEBYTECODE=1 python3 - \
  "$OUT/a2_04_l0_replay_eddfactor.json" <<'PY'
import json, pathlib, sys
p = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert p["source"]["mode"] == "DIRECT_EDDFACTOR"
assert p["source"]["FINISH_REC"] == 1.0
assert p["source"]["valid_frequency_records"] == 196185
assert p["wiring_replay_shells"] == list(range(44))
assert p["positive_summary"]["verdict"] == "PASS"
assert p["negative_control"]["verdict"] == "EXPECTED_FAIL_OBSERVED_ALL_5"
assert p["negative_control"]["failed_bands"] == ["EUV", "FUV", "UV", "OPT", "IR"]
assert p["guard_hits"] == p["fallback_hits"] == 0
print("A2_04_EDDFACTOR_AND_NEGATIVE_CONTROL PASS")
PY
PYTHONDONTWRITEBYTECODE=1 python3 scripts/a2_04_classic_debt_sweep.py \
  | tee "$OUT/a2_04_classic_debt_sweep.json"
EOF
```

기대 marker는 다음이다.

```text
PASS CMFGEN_ORACLE_CONTRACT ... profile=snapshot
A2_04_COMMIT_SELFTEST PASS ...
"commit_callers": ["cmfgen_commit_jnu", "main"]
A2_04_REPLAY_COMMIT PASS shells=44 bins=4000 generation=1
"positive_summary": {... "verdict": "PASS"}
"negative_control": {... "verdict": "EXPECTED_FAIL_OBSERVED_ALL_5"}
A2_04_EDDFACTOR_AND_NEGATIVE_CONTROL PASS
"verdict": "PASS_MEASURED_NO_REPAIR"
```

positive L-0 기준은 모든 replay shell에서 `E_1<=0.10`, 각 5대역
`E_B<=0.10`, `P95<=0.15 dex`다. 음성 대조는 다섯 대역이 각각 `E_B>0.10`이어야
전체 rc=0이다. s44-s49를 hold·외삽·복사하지 않는다.

## 8. §11 단계 회귀 대장 — 정확히 한 행

source bundle hash는 이 보고서에 열거한 13개 A2-04 구현/검증 파일의 `sha256sum`
출력을 다시 SHA-256한 값이다. input manifest hash는 oracle manifest, A2-02C union과
resolution result, deck config와 geometry의 `sha256sum` 출력을 다시 SHA-256한 값이다.

```json
{
  "stage_id": "A2-04",
  "contract": "MC estimator와 pure-CMFGEN CPU producer가 같은 radiation_field_commit API를 사용하고 Planck canonical overwrite를 차단하며 L-0 replay를 통과한다",
  "source_tree_hash": "37a4522ed6aab0703655e31ff47c3c2b179542c39e6ae6432cdbfc49214cf6c0",
  "input_manifest_hash": "d6b789c717084b9681cd58a57a799e449a5fdae32b639ddd994d8166f5c8b240",
  "oracle_id": "/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4; snapshot replay eligible; physical oracle ineligible",
  "node": "local sandbox completed; lageunha DIRECT_EDDFACTOR pending",
  "command": "local: make selftest_a2_04_commit selftest_a2_04_replay_commit && selftest/callgraph/synthetic-replay/classic-sweep; driver: docs/CODEX_IMPL_A2_04.md section 7",
  "exit_status": "local gates 0; lageunha DIRECT_EDDFACTOR PENDING_DRIVER_EXECUTION",
  "new_layer_status": "common commit PASS; synthetic L-0 wiring PASS; DIRECT_EDDFACTOR L-0 PENDING_DRIVER_EXECUTION",
  "all_previous_layer_statuses": "A2-03 selftest/callgraph/micro parity rerun PASS; A2-00 through A2-02 not rerun in this session",
  "negative_control_status": "PASS: dilute-Planck expected failure observed in EUV,FUV,UV,OPT,IR; DIRECT_EDDFACTOR negative rerun pending with driver block",
  "coverage": "8 CPU translation units; commit callers=main,cmfgen_commit_jnu; canonical external writers=0; synthetic shells=9; actual replay target shells=44",
  "metric_values": "synthetic max_E1=1.0670693551008627e-17,max_band_EB=7.678525915832037e-18,max_P95=9.64327466553287e-17; negative EB=[0.9973461023,0.9196778266,0.4955145707,0.3087160539,0.5837358479]",
  "changed_output_allowlist": ["canonical RadiationField state", "optional radiation-field commit dump", "offline replay/callgraph/classic JSON"],
  "guard_hits": 0,
  "fallback_hits": 0,
  "rng_seed": null,
  "mc_confidence": "N/A deterministic commit/replay; no MC physical comparison claimed",
  "artifact_paths": ["docs/CODEX_IMPL_A2_04.md", "scripts/a2_04_commit_callgraph.py", "scripts/a2_04_l0_replay.py", "scripts/a2_04_classic_debt_sweep.py", "tests/a2_04_commit_selftest.c", "tests/a2_04_replay_commit.c", "/tmp/a2_04_commit_selftest.log", "/tmp/a2_04_callgraph.log", "/tmp/a2_04_l0_selftest.json", "/tmp/a2_04_classic.log"],
  "driver_signoff": "PENDING_DRIVER_EXECUTION"
}
```

## 9. 확인된 결함·coverage gap

코드를 수정하지 않고 다음을 인계한다.

1. **legacy floor 전역 제거 미완**: §4와 같이 canonical 쪽만 제거됐고
   `src/lumina_plasma.c:14687`의 `1e-30`은 남았다. A2-03 인계를 전역 요구로 읽으면
   A2-04 미완이다.
2. **실자료 운전석 artifact 부재**: synthetic selftest와 음성 대조 로그는 있으나
   `validation/a2_04/a2_04_l0_replay_eddfactor.json`은 없다. §7 실행 전에는 L-0
   production PASS와 driver signoff를 부여할 수 없다.
3. **live pure-CMFGEN wrapper coverage gap**: offline replay는 EDDFACTOR를 Python에서
   canonical 4000-bin average로 만든 뒤 C fixture의 `radiation_field_commit()`을
   검사한다. live `cmfgen_run -> cmfgen_commit_jnu`가 만든 1000-bin `cs.J`의 의미와
   재빈 결과를 EDDFACTOR와 직접 대조하지 않는다. 따라서 wrapper 호출선의 존재는
   compiler gate로 증명되지만 그 production 물리값의 L-0 PASS는 증명되지 않았다.
4. **post-publication 진단 실패의 API 의미**: commit은 public field를 갱신한 뒤
   owner validation과 선택적 dump를 호출한다(`src/radiation_field.c:442-472`). dump open
   또는 close가 실패하면 API는 `-1`을 반환하지만 공개 generation은 이미 전진했다
   (`:475-513`). selftest의 “failed commit is atomic”은 publish 전 거부만 검사한다.
   완전한 실패 원자성을 요구한다면 결함이다.
5. **legacy dual view 잔류**: `cmfgen_write_jnu()` compatibility copy와 40개 legacy
   consumer가 남아 있어 A2-04는 producer 정본만 단일화했다. 소비 정본 단일화는 아직
   완료되지 않았다.

## 10. A2-05 인계

A2-05는 CPU bound-free rate 하나만 이행한다.

- generation/frame/unit/validity를 검사하는 canonical read-only view를 먼저 만들고,
  BF caller가 legacy `nlte->J_nu`, `bf_rate_estimator`, `(W,T_rad)`를 우회하지 못하게 한다.
- `VALID`과 `EXACT_ZERO`만 적분하고 `UNSAMPLED/OUT_OF_GRID/STALE`는 작은 수로 계속하지
  말고 명시적으로 실패 또는 계약된 unavailable 상태로 전파한다.
- 광이온율은 canonical bin-average `J_nu`와 단면적을 직접 적분하고 source edge
  overlap을 보존한다. legacy 1000-bin center sample이나 `1e-30`을 정본 입력으로 쓰지
  않는다.
- BF가 더 이상 legacy 배열을 읽지 않음을 compiler/runtime trace로 증명한 뒤에만 BF
  경로의 `1e-30` 의존을 제거한다. 다른 40개 소비자의 배열 자체를 A2-05에서 성급히
  삭제하지 않는다.
- L-1bf positive replay와 `bf_rate_estimator` 우회, stale generation, out-of-grid
  silent floor, double normalization 음성 대조를 추가한다.
- A2-04의 §7 실자료 L-0 replay와 앞 단계 gate를 먼저 재실행한다. L-0
  `PENDING_DRIVER_EXECUTION`을 하류 L-1bf PASS로 가리지 않는다.
- P02 line-estimator 누락과 `jbar_line/j_blue`는 A2-06, GPU mirror는 A2-12,
  `(W,T_rad)` 구조체·환경변수 전면 제거는 A2-17에 남긴다.

## 11. 최종 줄번호 검증

이 보고서의 코드 줄번호는 작성 직전 현재 working tree에 대해 `nl -ba`와 다음 symbol
검색을 교차 확인했다.

```text
commit declaration                 src/radiation_field.h:198-199
commit implementation              src/radiation_field.c:402-403
MC commit caller                    src/lumina_main.c:485
pure-CMFGEN commit caller           src/lumina_cmfgen.c:3434
pure-CMFGEN wrapper runtime caller  src/lumina_cmfgen.c:5122
canonical MC unsampled/zero         src/radiation_field.c:272-287
legacy 1e-30 floor                  src/lumina_plasma.c:14687
H02                                 src/lumina_main.c:107; src/lumina_cuda.cu:6905
H13                                 src/lumina_plasma.c:1030
S01                                 src/lumina_plasma.c:847; src/lumina_main.c:534
P02                                 src/lumina_main.c:420,478
```

보고서 추가는 source 파일의 줄번호를 바꾸지 않는다. 이후 source가 이동하면 심볼과
기전 설명을 정본 키로 사용하고 이 표를 다시 생성해야 한다.
