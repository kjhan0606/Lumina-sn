# A2-02C 개정 발주 집행 보고

- 집행 기준: `docs/ORDER_A2_GRID_AMENDMENT_BY_CODEX.md`
- 근거·보존 commit: `43ffe3186f926e887139228d465a8c63fa5c42a8`
- 새 단계명: **A2-02C**
- 현재 판정: **IMPLEMENTED / PART1_DRIVER_READY / estimator ladder = PENDING_CAPTURE_RUN**
- A2-02 최종 판정: **BLOCKED_MISSING_SEGMENT_CAPTURE 유지**
- A2-03: 계속 발주 보류

종전 `validation/a2_02/*`, `docs/A2_02_FREQUENCY_UNION.json`과 v1 schema는
수정·삭제·덮어쓰기하지 않았다. 새 산출물은 모두 `A2_02C` 또는 `a2_02c` 이름을 쓴다.
덱과 `/gpfs`는 읽기 전용이며 commit/push하지 않았다.

## Part 1 — §4.2 단계 1~5

### 1. line census와 100–20000 Å hash 결박

`scripts/a2_02c_frequency_union.py`가 `line_list.csv`를 streaming 한 번 읽어 다음을
동시에 수행한다.

- 원 행 수, 주파수 유한·양수, 내림차순, `line_id == source_row`를 검사한다.
- `levels.csv`의 `(Z, ion, level)` 집합으로 lower/upper 연결을 전수 검사한다.
- 닫힌 line-center 창 `100 Å <= lambda <= 20000 Å`를 canonical JSON으로 만들고
  SHA-256을 계산한다.
- 현재 등록 profile을 `exp(-x^2)`, `v_D=1.0e6 cm/s`, `+-4 dnu_D`, 잘린 support에서
  적분 1로 정규화한 계약으로 결박한다. provenance는
  `src/lumina_cmfgen.c:3980-3982,4205-4208`이다.
- `A_ul`, `gf`, population, 예상 rate는 분할 판정에 쓰지 않는다.

production 실행 시 다음 새 파일이 생긴다.

```text
validation/a2_02c/A2_02C_LINE_CENSUS.json
validation/a2_02c/A2_02C_BB_IN_DOMAIN.csv
validation/a2_02c/A2_02C_BB_EXCLUDED_OUTSIDE_DOMAIN.csv
```

census manifest는 원 line-list/levels hash, domain/profile hash, 두 CSV hash와
`in_domain + excluded == finite_positive_input_census`를 보존한다. 운전석 실측 음성대조인
`nu<1e10:382`, `<1e13:34945`, `<1e14:245123` 및 원 행 `2,220,953`과 다르면 rc=2다.

### 2. BB 원장·전량 제외·개정 1 음성대조

두 CSV의 공통 필드는 다음이다.

```text
line_id,element,atomic_number,ion,lower,upper,nu_lu_hz,lambda_lu_A,
A_ul_s-1,reason,source_row,source_hash,domain_contract_hash,profile_id,profile_hash
```

Ni I `l461->u462` 강제 재진입, 제외 line 조회, census 삭제/중복/hash 변조, 양 경계
안/경계/밖, 약한 in-domain line strength floor, continuum에 의한 line-ID 재활성화의
6개 독립 child가 각각 명시적 `A2_02C_UNION_NEGATIVE_FAIL`과 rc=4를 내도록
사전등록했다. 정상 wrapper는 6/6을 확인하고 rc=0이다.

full Q는 다음 새 manifest로 고정한다.

```text
validation/a2_02c/A2_02C_Q_SET.json
```

현재 loader/rate-graph 계약에 따라 `BB_IN_DOMAIN 전량 x s0..s49`이며, in-domain CSV
hash, shell census, profile hash로 `q_set_hash`를 만든다. strength/rate pruning은 없다.
capture generation은 실행 때 raw header generation으로 결박한다.

### 3. 개정 7소비자 합집합

새 schema는 `lumina-a2-02c-frequency-union-v2`이고 manifest는 다음에 생성된다.

```text
validation/a2_02c/A2_02C_FREQUENCY_UNION.json
```

종전 여섯 소비자는 보존하고 bound-bound 소비자만 `BB_IN_DOMAIN` line center와 등록된
전체 profile support로 교체한다. `amends_after`는 정확히 `43ffe31`이며 종전 union의
경로/hash는 `preserves_blocked_artifact`에 남는다. 일곱 소비자의 최소·최대에서 union을
다시 계산하므로 종전 edge/result hash를 재사용할 수 없다. 20000–25000 Å formal,
observer, validation coverage는 그대로다.

### 4. fine dump builder와 cohort

`scripts/a2_02_prepare_fine_dump.py`를 v2 입력으로 갱신했다.

- 기본 입력: amended union, Q/cohort, 새 v2 template.
- 기본 출력: `validation/a2_02c/a2_02c_fine_bin_averages.npz`와
  `validation/a2_02c/A2_02C_RESOLUTION_INPUT.json`.
- 종전 delta top-hat 대신 등록된 잘린 Gaussian profile의 보존 bin average를 만든다.
- profile support마다 Doppler 폭당 12점의 edge를 fine grid에 실행 전에 넣는다.
- EDDFACTOR/CHIETA/CMFD/BF의 기존 보존 재빈과 네 validity는 유지한다.

`A2_02C_ESTIMATOR_COHORT.json`은 종전 마지막 8000→16000 결과에서 실제 유효했던
42 record를 hash로 읽어 **전부 이관**한다. 개정 창 밖이 된 record도 삭제하거나 결과를
본 뒤 대체하지 않고 `CARRIED_EXCLUDED_OUTSIDE_DOMAIN`으로 남긴다. s8 Fe II
`l61->u1308`은 별도 mandatory 검사로 고정된다.

보충 표본은 결과를 보기 전 다음 규칙으로 결정한다.

```text
6 wavelength strata x ion strata {0,1,2,3+}
각 occupied stratum에서
min SHA256(domain_hash|wavelength_stratum|ion_stratum|line_id|source_row)
선택 후 s0, s8에 배치; 기존 이관 record와 중복 제거
```

따라서 실행 사이 cohort membership과 `q_set_hash`를 바꿀 수 없다.

### 5. 전역 1000→16000 사다리

`scripts/a2_02_resolution_ladder.py`의 새 출력 schema는
`lumina-a2-02c-global-resolution-result-v2`이다. 전역 `Jbar` 계산 loop와 결과 key를
제거했고 다음 네 지표만 남겼다.

1. 등록 대역 `integral J_nu dnu`
2. matched `Gamma`
3. 대역 적분 `chi_nu`
4. 대역 적분 `eta_nu`

각 지표의 최대 `<=1%`, 중앙값 `<=0.2%`, invalid-eligible 0을 그대로 요구한다. 가장
작은 PASS pair의 coarse N을 고르고 8000→16000도 실패하면 rc=3/BLOCKED다. 출력은
`validation/a2_02c/a2_02c_global_resolution_result.json`이다.

### Part 1 운전석 명령

새 출력 경로가 아직 없는 첫 실행용이다. `/usr/bin/time`을 쓰지 않으며 `/gpfs`와 덱에
쓰지 않는다.

```bash
ssh lageunha 'bash -s' <<'EOF'
set -euo pipefail
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
bash scripts/a2_02c_run_part1.sh
EOF
```

기대 rc는 census/음성대조/builder가 모두 정상이면서 사다리가 선택되면 **0**이다.
새 edge의 마지막 pair가 과학적으로 실패하면 wrapper도 **3**을 그대로 반환하며 이것은
입력 오류가 아니라 규범적 `BLOCKED`다. 입력/schema/hash/validity 오류는 **2**다. 주요
marker는 다음이다.

```text
A2_02C_UNION_NEGATIVE_SUMMARY passed=6 total=6
A2_02C_UNION PASS rows=2220953 ... consumers=7 amends_after=43ffe31
A2_02C_PREPARE PASS rc=0 ...
A2_02C_LADDER SELECTED ...                 # rc=0
# 또는 A2_02C_LADDER BLOCKED ...           # rc=3
A2_02C_PART1_DONE ladder_rc=0_or_3 ...
```

동일 이름이 이미 있으면 도구는 덮어쓰지 않고 rc=2다. 재실행이 필요하면 운전석이 새
`A2_02C_OUT` 경로를 지정해야 하며 종전 v1 또는 앞선 A2-02C 증거를 삭제하지 않는다.

## Part 2 — §4.4 estimator 기반 계측

### 6. 기본 OFF raw segment capture

구현 파일은 다음이다.

```text
src/a2_02c_segment_capture.c
src/a2_02c_segment_capture.h
docs/A2_02C_SEGMENT_CAPTURE_SCHEMA.json
docs/A2_02C_ESTIMATOR_SCHEMA.json
```

CPU transport의 실제 `move_r_packet` 직전에서 packet을 읽기만 한다. 각 88-byte record는
packet/segment ID, generation, shell, 공이동 `nu_start/nu_end`, 공이동
`energy_start/energy_end`, path length, `V_s`, `delta_t`를 가진다. homologous segment에서
공이동 주파수와 energy의 선형 endpoint trajectory를 저장한다. header는 production
packet count, generation, `t_exp`, `delta_t`, shell-volume table, segment count와 complete
flag를 가진다. 불완전 파일, byte-count 불일치, generation/V/dt/frame 불일치,
packet별 segment-ID 누락·중복은 replay가 거부한다.

gate는 세 변수가 모두 필요하다.

```text
LUMINA_A2_02C_SEGMENT_CAPTURE=1
LUMINA_A2_02C_CAPTURE_GENERATION=<positive generation>
LUMINA_A2_02C_CAPTURE_PATH=<반드시 존재하지 않는 새 파일>
```

기본 OFF에서는 capture 파일과 메시지가 없다. ON도 named file만 추가하며 packet,
estimator, RNG, plasma를 쓰지 않고 stdout/stderr를 추가하지 않는다. `wbx`로 열어 기존
파일을 교체하지 않는다. Makefile CPU source와 A2-01 instrumented CPU tree에도 새 모듈을
연결했다.

빌드·게이트 fixture의 lageunha 명령과 기대 rc는 다음과 같다.

```bash
ssh lageunha 'bash -s' <<'EOF'
set -euo pipefail
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
PYTHONPYCACHEPREFIX=/tmp/a2_02c_pycache python3 -m py_compile \
  scripts/a2_02c_segment_replay.py scripts/a2_02c_capture_gate_selftest.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/a2_02c_capture_gate_selftest.py
gcc -O2 -Wall -Wextra -std=c11 -fopenmp \
  -o /tmp/lumina_a2_02c_capture \
  src/lumina_main.c src/lumina_transport.c src/a2_02c_segment_capture.c \
  src/lumina_plasma.c src/lumina_element_wide.c src/lumina_atomic.c \
  src/lumina_cmfgen.c -lm -fopenmp
EOF
```

기대 최종 rc는 **0**, gate marker는
`A2_02C_CAPTURE_GATE_SELFTEST PASS off_file=0 on_file=1 output_parity=byte packet_mutation=0`이다.
기존 소스 경고는 남지만 새 모듈은 `-Wall -Wextra` build/link를 통과한다.

### offline replay와 §3.3 게이트

`scripts/a2_02c_segment_replay.py`는 같은 capture를 두 view에 사용한다.

- 전역 `J_nu`: segment의 선형 주파수 궤적을 canonical edge에서 정확히 분할하고 선형
  energy path integral을 `4*pi*V_s*delta_t*dnu`로 정규화한다.
- line `Jbar`: 같은 segment에서 등록 Gaussian `phi`를 16-point Gauss-Legendre로
  적분한다. point-at-center와 coarse-grid fallback은 없다.
- P는 한 2P capture 안의 `packet_id<P`, 2P는 `packet_id<2P`다. 저장 energy가 2P
  normalization이므로 replay가 `N_capture/effort`로 재가중한다. 따라서 P ledger는 2P
  ledger의 진부분집합이다.
- 각 line record는 generation, shell/line/profile, value/units/frame/validity,
  count/variance/standard error, Q hash, raw-ledger hash와 provenance를 가진다.

§3.3은 독립 결과 key 세 개로 판정한다.

1. `same_measure_commit_gate`: raw hash, generation, frame, V, dt, packet normalization,
   Q hash, canonical edge hash 일치.
2. `canonical_projection_closure`: 실제 canonical bin에서 상수이고 적분 1인 control
   profile의 direct 값과 global-bin projection을 비교.
3. `fine_diagnostic_closure`: 같은 segment로 만든 profile support 12점/폭과 24점/폭
   histogram의 선행 수렴을 확인한 뒤 direct line estimator와 24점/폭 진단값을 비교.

모두 최대 1%, 중앙값 0.2%, invalid-eligible 0을 요구한다. 독립 RNG capture가 없으면
의도적으로 rc=3, `decision=PENDING_CAPTURE_RUN`이다.

개정 §4.6의 7개 음성대조도 각각 child rc=4/FAIL marker로 사전등록했다.

```text
legacy delta-top-hat schema
cohort/Q hash swap
mandatory Fe II record removal
label-only fake P->2P
line/profile identity swap
median-only false PASS
old union/edge result reuse
```

정상 wrapper marker는
`A2_02C_REPLAY_NEGATIVE_SUMMARY passed=7 total=7`, rc=0이다.

### capture/replay 운전석 명령 골격

크기는 코드나 문서에 고정하지 않았다. 운전석이 아래 `A2_02C_P`를 먼저 정한 뒤 쓴다.
예시는 2-iteration run의 generation 2만 capture한다. canonical 덱은 변경하지 않는다.

```bash
ssh lageunha 'bash -s' <<'EOF'
set -euo pipefail
: "${A2_02C_P:?운전석이 production effort P를 먼저 지정해야 함}"
REPO=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
OUT=$REPO/validation/a2_02c
TWO_P=$((2 * A2_02C_P))
CAPTURE=$OUT/a2_02c_segments_g2_2P${TWO_P}.bin
cd "$REPO"
test ! -e "$CAPTURE"
export OMP_NUM_THREADS=60
export LUMINA_A2_02C_SEGMENT_CAPTURE=1
export LUMINA_A2_02C_CAPTURE_GENERATION=2
export LUMINA_A2_02C_CAPTURE_PATH="$CAPTURE"
/tmp/lumina_a2_02c_capture \
  data/tardis_reference_toy06_19p48d_sivcaiv_ftos \
  "$TWO_P" 2 spectrum nlte
EOF
```

capture run 정상 rc는 **0**이다. 이어 Part 1 결과의 `selected_bins`를 운전석이 읽어
다음 replay에 넣는다. 독립 stream 파일도 같은 schema·generation·2P 이상이어야 한다.

```bash
ssh lageunha 'bash -s' <<'EOF'
set -euo pipefail
: "${A2_02C_P:?}" "${A2_02C_CAPTURE:?}" "${A2_02C_CAPTURE_INDEPENDENT:?}" \
  "${A2_02C_SELECTED_BINS:?}"
REPO=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
OUT=$REPO/validation/a2_02c
cd "$REPO"
PYTHONDONTWRITEBYTECODE=1 python3 scripts/a2_02c_segment_replay.py self-test
PYTHONDONTWRITEBYTECODE=1 python3 scripts/a2_02c_segment_replay.py negative-controls
PYTHONDONTWRITEBYTECODE=1 python3 scripts/a2_02c_segment_replay.py run \
  --capture "$A2_02C_CAPTURE" \
  --independent-capture "$A2_02C_CAPTURE_INDEPENDENT" \
  --cohort "$OUT/A2_02C_ESTIMATOR_COHORT.json" \
  --union "$OUT/A2_02C_FREQUENCY_UNION.json" \
  --global-bins "$A2_02C_SELECTED_BINS" \
  --effort "$A2_02C_P" --double-effort "$((2 * A2_02C_P))" \
  --output "$OUT/a2_02c_estimator_effort_result.json"
EOF
```

모든 수렴·closure·독립 stream이 PASS하면 기대 rc **0**이다. 유효 측정이지만 어느
1%/0.2% gate든 실패하면 **3/BLOCKED**, schema/hash/generation/Q/identity 오류는 **2**다.

## 7. capture 크기 제안 — 운전석 선택 사항

기존 실측 `2000 packet x 2 iter x OMP1 = 편도 약 35분`에서 compute-only 이상적
OMP60 식은 다음이다.

```text
T_ideal(2P,k iter) = 35 min * (2P/2000) * (k/2) / 60
raw bytes = 128 + 16*N_shell + 88*N_segment
```

실제 capture는 segment 수, serialized file write, memory bandwidth와 OMP scaling의
영향을 받으므로 이상식만으로 예약하지 않는다. 아래 범위는 P=2000, 즉 2P=4000을
**크기 고정값이 아닌 산정 예시**로 대입한 lageunha 계획 범위다.

| P 정의 후보 | 2P 자료를 얻는 방식 | compute-only ideal | OMP 유효 speedup 15–35 + capture I/O 계획 범위 | 장단점 | 제안 |
|---|---|---:|---:|---|---|
| 전체 production iterations | 4000 packet x 2 iter 모두를 generation별 같은 schema 파일로 capture/합산 | 1.17분 | 약 3–19분/stream | 실제 production effort 정의에 가장 충실하지만 generation별 Q/field가 달라 단일-generation cache 수렴과 섞인다 | 보조 sensitivity |
| 수렴장 단일-iteration replay | 수렴 snapshot에서 4000 packet x 1 iter; 또는 2-iter run에서 마지막 generation만 capture | 0.58분(진짜 1 iter), warm-up 포함 시 1.17분 | 약 2–10분/stream, warm-up 포함 3–19분 | 같은 generation/Q를 유지해 §3.3과 cache 계약이 명확하다 | **권고** |

독립 RNG 재현까지는 두 stream이므로 wall allocation 또는 순차 시간은 대략 2배다.
P가 한 단계 배가되면 compute, raw bytes와 replay work를 거의 2배로 계획한다. 먼저
작은 pilot에서 `segment_count`, capture bytes, 실제 elapsed와 OMP 효율을 기록한 뒤
resource ceiling과 마지막 `(P,2P)` pair를 manifest에 고정해야 한다. 이번 구현은 P,
ceiling, 마지막 pair를 정하지 않았다.

고정 prefix 운전 규칙은 별도 P run과 2P run을 비교하는 것이 아니라 **한 2P capture를
한 번 만들고 packet ID prefix로 두 표본을 재생**하는 것이다. 현재 CPU RNG가 thread별
stream+dynamic scheduling이므로 이 규칙만이 계측 overhead나 OMP scheduling 차이로
두 effort의 prefix가 깨지는 일을 막는다. 독립 재현은 별도 seed의 2P capture 전체를
추가 입력으로 사용한다.

## 8. 이 세션에서 완료한 검증

대용량 deck scan, `/gpfs` fine build, model/capture run은 로그인 노드에서 하지 않았다.
build와 작은 synthetic fixture만 수행했다.

```text
Python py_compile: PASS
A2_02C_UNION_SELFTEST: PASS
개정 1 음성대조: 6/6 PASS wrapper (각 child rc=4)
A2_02C_PREPARE_SELFTEST: PASS
A2_02C_LADDER_SELFTEST: PASS, global_metrics=4, Jbar_removed=1
A2_02C_REPLAY_SELFTEST: PASS
개정 §4.6 음성대조: 7/7 PASS wrapper (각 child rc=4)
A2_02C_CAPTURE_GATE_SELFTEST: PASS, OFF/ON output byte parity, packet mutation 0
CPU OMP full link to /tmp: PASS
A2-01 classifier: rows=157, unclassified=0 PASS
A2-01 trace selftest: PASS; instrumented CPU tree including capture module link PASS
git diff --check: PASS
```

### lageunha 회귀 묶음

src가 바뀌었으므로 다음은 운전석에서 전부 rc=0이어야 한다. `/usr/bin/time`은 쓰지
않는다.

```bash
ssh lageunha 'bash -s' <<'EOF'
set -euo pipefail
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
python3 scripts/a2_01_census_contract.py check
python3 scripts/a2_01_read_trace.py selftest
python3 scripts/a2_01_oracle_compat_selftest.py
python3 scripts/a2_00_oracle_negative_controls.py --scratch-root /tmp
python3 scripts/run_ne_naming_controls.py
python3 scripts/run_deck_fossil_controls.py
python3 scripts/run_config_prec_negative_controls.py --binary /tmp/lumina_a2_02c_capture
bash scripts/run_zinert_selftest.sh
/home/kjhan/.lumina_scratch/run_dbuild_gates.sh
/home/kjhan/.lumina_scratch/run_cls_verify.sh
EOF
```

기대 marker는 A2-00 음성대조 전항, A2-01 157/157·trace/compat, NE 5/5,
DECK 5/5, CONFIG-PREC 7/7, Z-INERT 전항, D 19/19, K 7/7, classifier 7/7이다.
production A2-01 OFF parity/read count는 운전석의 기존 2000x2 trace 절차로 별도
재실행한다. 이 보고 시점의 production 회귀 상태는 `PENDING_DRIVER_EXECUTION`이며
위 selftest를 production PASS로 과장하지 않는다.

## 9. 변경 파일과 비변경선

Part 1/오프라인:

```text
scripts/a2_02c_frequency_union.py
scripts/a2_02_prepare_fine_dump.py
scripts/a2_02_resolution_ladder.py
scripts/a2_02c_run_part1.sh
docs/A2_02C_RESOLUTION_INPUT_TEMPLATE.json
```

Part 2 계측/재생:

```text
src/a2_02c_segment_capture.c
src/a2_02c_segment_capture.h
src/lumina_main.c
src/lumina_transport.c
src/lumina.h
Makefile
scripts/a2_02c_segment_replay.py
scripts/a2_02c_capture_gate_selftest.py
scripts/a2_01_read_trace.py
docs/A2_02C_SEGMENT_CAPTURE_SCHEMA.json
docs/A2_02C_ESTIMATOR_SCHEMA.json
```

`src`의 물리·rate·population·opacity·emissivity 경로는 바꾸지 않았다. 바뀐 호출은
capture begin/end와 segment read-only hook뿐이다. 덱, `/gpfs`, 기존 A2-02 BLOCKED
artifact는 변경하지 않았다.

## 10. 남은 위험

1. **production 미실행:** 2,220,953행 census 수, amended union edge, 새 fine NPZ와 네
   지표 사다리는 lageunha 결과 전에는 PASS가 아니다.
2. **capture 부재:** estimator effort, 독립 RNG, canonical/fine closure는 실제 raw
   ledger가 없어 `PENDING_CAPTURE_RUN`이다.
3. **CPU lane 한정:** 이번 raw hook은 CPU `move_r_packet` 경로다. CUDA transport
   capture는 구현하지 않았다. A2-12/13 production GPU 이식 근거로 바로 승격할 수 없다.
4. **Q 가정:** 현재 runtime loader가 모든 in-domain line을 모든 50 shell의 enabled
   graph로 본다는 기존 계약으로 Q를 만든다. A2-01 실측 graph가 subset임을 보이면 capture
   전에 Q manifest와 cohort를 재생성해야 하며 strength/rate로 사후 삭제할 수 없다.
5. **OMP와 I/O:** ON-path record write는 lock으로 파일 원자성을 지킨다. packet/RNG를
   쓰지는 않지만 OMP dynamic scheduling과 thread별 RNG 때문에 별도 P/2P run 사이
   packet-ID trajectory는 prefix가 아니다. 그래서 한 2P ledger 내부 prefix만 허용했다.
6. **독립 seed 운전:** CPU 실행에는 seed env override가 없다. 독립 stream은 canonical
   덱을 고치지 않고 별도 read-only scratch deck generation/승인된 seed fixture를 써야
   한다. 그 운전 절차와 seed 값은 운전석 결정 사항이다.
7. **CHIETA support:** 기존 builder의 chi/eta capture 밖 0 padding에는 validity 배열이
   없다. 이는 두 grid 사이 보존 비교이지 capture 밖 물리 coverage 증명은 아니다.
8. **대용량 replay 비용:** Python reader는 memmap이지만 ID 정렬과 line/fine closure는
   segment 수에 비례한다. pilot의 segment count/bytes/elapsed 전에는 최종 ceiling을
   확정할 수 없다.

## 11. 개정 회귀 대장 행

| stage | source/input hash | node | Part 1 | estimator ladder | closure | 음성대조 | 종전 결과 | driver signoff | 다음 단계 |
|---|---|---|---|---|---|---|---|---|---|
| A2-02C | `amends_after=43ffe31`; line/domain/Q/raw hashes는 각 새 manifest에서 결박 | `PENDING_DRIVER_EXECUTION` | `IMPLEMENTED_DRIVER_READY`; census/fine/global ladder 실측 대기 | **`PENDING_CAPTURE_RUN`** | canonical/fine/independent RNG=`PENDING_CAPTURE_RUN` | synthetic union 6/6·estimator 7/7 PASS; production rerun pending | v1 `BLOCKED` 보존, overwrite 없음 | `PENDING_DRIVER_SIGNOFF` | A2-03 HOLD |

A2-02C PASS 서명은 Part 1 rc=0, 실제 P→2P와 독립 stream 수렴, 세 §3.3 gate,
production 음성대조와 전체 회귀가 모두 PASS한 뒤에만 가능하다.
