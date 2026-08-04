# Codex A3 — Wave-3.2 최종 라운드 구현 보고서

작성일: 2026-08-01 (Asia/Seoul)  
입력: `docs/CODEX_WAVE32_C2_SUMMARY.md`, `docs/CODEX_WAVE32_B2_TEST.md`  
M_V 정본: `docs/CODEX_WAVE32_C_REVIEW.md` §6

## 0. 납품 판정

고정 순서의 물리 복원 사다리 7개를 구현했고, 각 rung 종료 상태를 독립 patch로
고정했다. 일곱 patch는 A3 시작 스냅샷에 1→7 순서로 모두 적용되며, 적용 후 소유
파일은 현재 작업트리와 byte 일치한다.

| rung | 물리/운영 계약 | 판정 | 핵심 관측 |
|---:|---|---|---|
| 1 | 투영 빌더 단일화 | PASS | 12/12 armed/unarmed byte matrix 유지 |
| 2 | harness 실패 전파 | PASS | 정상 RC=0, 잘못된 commit env RC=1 |
| 3 | top-stage bf helper + 생산 우회 계측 | PASS | parity frozen byte 불변, default/C2/JEQB 생산 우회 0/1/1 |
| 4 | runtime 카운터 실소유 경로 | PASS | 실제 `nlte_solve_all`: 1/15/14, 음성 1/0/0 |
| 5 | R7 writer 정직화 | PASS (offline) | 실측 η 감사, quarantine, η/debit seeded 결함 검출 |
| 6 | within-SL OOM 폐합 | PASS | OOM RC=-1, 정상 frozen byte 불변 |
| 7 | Fe M_V 경계질량 stage | PASS, 기대 크기 편차 기록 | s0/s8 `EW_PASS`; s0 전 stage 방향 PASS, D 57.84% 개선 유지 |

규율 준수:

- 신규 모델 실행 0, GPU 실행 0. parity59 frozen CPU replay와 빌드만 수행했다.
- 신규 clamp/floor/cap 0. q_t는 기록 후 strict validation하며 보정하지 않는다.
- 커밋 0. 기존 dirty 작업트리의 비소유 변경은 복원하거나 정리하지 않았다.
- 실제 CUDA-loop의 parity59 iter=10 capture는 실행 금지 때문에
  **UNRESOLVED-until-capture**다. writer offline 계약은 검증했다.
- 성공 `COMMIT=1`의 pilot/off-target 격리 검증은 이 구현으로 가능해졌지만,
  발주문대로 후속 B3 판정 범위로 남겼다.

## 1. 순차 patch 원장

| 순서 | patch | SHA-256 |
|---:|---|---|
| 1 | `patches/w32a3_rung1_projection_builder.patch` | `8a32a405a64c52e375af71a8b74cb01ff310375af986261e6d123ad85946d4aa` |
| 2 | `patches/w32a3_rung2_harness_failure_status.patch` | `f376230f37f6f40745a3de49e8b1ceaea170268c162610e2f0264e53b9cc8dc6` |
| 3 | `patches/w32a3_rung3_topstage_bf_gpu_telemetry.patch` | `1023ce4ee9c503863193bfc809b418eac909fe2d078a7e4bf137179dcb026e44` |
| 4 | `patches/w32a3_rung4_runtime_counter_owners.patch` | `63a1d77513a16f471e235ec2b15d950b903dedd949027981be0d7d02bd9b55b5` |
| 5 | `patches/w32a3_rung5_r7_honest_writer.patch` | `d4a285331520e3002cbe815b50f321e3b25ec5526d4929216cfa1ef8a291d7fd` |
| 6 | `patches/w32a3_rung6_within_sl_oom.patch` | `6ecdbb124c744179db61d22ef62041a3ebece0d7936d72fcdb0eb880655468d6` |
| 7 | `patches/w32a3_rung7_fe_v_boundary_mass.patch` | `bd9258939c22683907904cea6a073e136728b6a78bd4337a4728f74fbcb74d04` |

순차 적용 자기검증:

```bash
for p in patches/w32a3_rung1_projection_builder.patch \
         patches/w32a3_rung2_harness_failure_status.patch \
         patches/w32a3_rung3_topstage_bf_gpu_telemetry.patch \
         patches/w32a3_rung4_runtime_counter_owners.patch \
         patches/w32a3_rung5_r7_honest_writer.patch \
         patches/w32a3_rung6_within_sl_oom.patch \
         patches/w32a3_rung7_fe_v_boundary_mass.patch; do
  git -C "$A3_START_COPY" apply --check "$PWD/$p"
  git -C "$A3_START_COPY" apply "$PWD/$p"
done
```

결과: `SEQUENTIAL_APPLY_PASS`; 최종 Makefile, harness, 5개 `src` 소유 파일과
신규 fixture를 `cmp`해 `FINAL_OWNED_FILES_MATCH`를 얻었다.

## 2. rung 1 — 투영 빌더 단일화

### 복원 물리 계약

production/commit 레인과 COMMIT=0 private 레인이 서로 다른 layout·global map·SL
projection을 만들지 않게 `nlte_build_projection()` 한 함수로 통합했다. 두 레인은
동일한 level/ion/super offsets, global-to-NLTE map, line map, within-SL partition과
population buffer를 만든다. `atom->n_lines != opacity->n_lines`는 runtime 오류와
assert로 fail closed한다.

### 사전등록 기대 변경 집합

순수 구조 통합이므로 정상 산출은 하나도 바뀌지 않아야 한다. 허용 변화는 layout
불일치 입력의 조기 실패뿐이다.

### patch

`patches/w32a3_rung1_projection_builder.patch`

### 자기검증

```bash
make -B bench_frozen_oracle
python3 scripts/wave32_r1_byte_invariant.py --no-build \
  --out /tmp/w32a3_r1b.W0FlkN
```

결과: `SUPER_LEVELS={0,1} × shell={0,8} × {oracle,pair-ion,pair-level}`
12/12 byte-identical, RC=0.

## 3. rung 2 — harness 실패 전파

### 복원 운영 계약

`bench_frozen_oracle.c`가 EW observer 반환값을 `(void)`로 폐기하지 않는다. 어느
원소의 EW 호출이라도 음수면 할당 상태와 임시 σ를 복구한 뒤 process RC가
non-zero가 된다.

### 사전등록 기대 변경 집합

성공 경로 파일은 불변이다. EW 실패 주입 시에만 exit status가 0에서 non-zero로
바뀐다.

### patch

`patches/w32a3_rung2_harness_failure_status.patch`

### 자기검증

```bash
env -i PATH="$PATH" LUMINA_FROZEN_ORACLE_ONLY_SHELL=0 \
  ./bench_frozen_oracle "$FROZEN" "$MODEL" "$OK_DIR"
cmp "$PRE_RUNG2/lumina_oracle_cell_s0.csv" \
    "$OK_DIR/lumina_oracle_cell_s0.csv"

env -i PATH="$PATH" LUMINA_FROZEN_ORACLE_ONLY_SHELL=0 \
  LUMINA_NLTE_ELEMENT_WIDE=1 LUMINA_NLTE_ELEMENT_WIDE_Z=26 \
  LUMINA_NLTE_ELEMENT_WIDE_SHELL=0 LUMINA_EW_FROZEN_COMMIT=bad \
  ./bench_frozen_oracle "$FROZEN" "$MODEL" "$BAD_DIR"
```

결과: 정상 RC=0 및 `cmp` RC=0. 실패 주입은
`[EW][CONFIG-FAIL] ... must be exactly 1`, observer failure를 출력하고 RC=1.

## 4. rung 3 — top-stage bf helper와 생산 GPU 우회 계측

### 복원 물리 계약

Fe top-stage III↔IV 조립도 estimator/C2/JEQB와 J 선택을 재구현하지 않고
`nlte_bf_field_source()`를 호출한다. 선택한 장과 GPU lookup의 장이 다르면 GPU
lookup을 쓰지 않으며, 이 우회를 frozen observer 전용이 아닌 process-wide 생산
카운터 `nlte_bf_gpu_field_bypass_count()`에 누적한다. EW manifest에도 생산 누계를
기록한다.

### 사전등록 기대 변경 집합

같은 장을 선택하는 parity 환경의 물리 산출은 byte 불변이다. 새로 바뀌는 관측량은
생산 우회 카운터/manifest 항목뿐이다.

### patch

`patches/w32a3_rung3_topstage_bf_gpu_telemetry.patch`

### 자기검증

```bash
# tests/wave32_bf_policy_selftest.c를 plasma/element/atomic과 CPU link
env -i PATH="$PATH" "$BF_SELFTEST"
env -i PATH="$PATH" LUMINA_C2_MATRIX_BF=1 "$BF_SELFTEST"
env -i PATH="$PATH" LUMINA_NLTE_BF_JEQB=1 "$BF_SELFTEST"
```

결과:

```text
default source=0 use_gpu=1 gpu_bypass=0 production_bypass=0
C2      source=1 use_gpu=0 gpu_bypass=1 production_bypass=1
JEQB    source=2 use_gpu=0 gpu_bypass=1 production_bypass=1
```

unarmed parity59 s0 frozen output은 rung 2와 byte-identical이었다.

## 5. rung 4 — runtime 카운터 3개 실소유 경로 배선

### 복원 운영 계약

`save_restore`, `per_ion_pin`, `topstage_IV`는 hook 호출 가능성이 아니라 실제
`nlte_solve_all()` 소유 경로에서 증가한다. 최종 production 누계는 manifest에
`runtime_final_*` 세 항목으로 게시한다. fixture도 가짜 hook 직접 호출 대신 frozen
state의 실제 full solve에 진입하며 필요한 `line_source_S`를 소유한다.

### 사전등록 기대 변경 집합

물리 배열과 oracle은 바뀌지 않고 카운터 값과 manifest telemetry만 바뀐다.

### patch

`patches/w32a3_rung4_runtime_counter_owners.patch`

### 자기검증

```bash
# bench 호출 심볼만 tests/wave32_runtime_counter_wrap.c로 연결
env -i PATH="$PATH" W32_RUNTIME_COUNTER_ONLY=1 \
  LUMINA_FROZEN_ORACLE_ONLY_SHELL=8 \
  LUMINA_SUPER_LEVELS=0 LUMINA_NLTE_ELEMENT_WIDE=1 \
  LUMINA_NLTE_ELEMENT_WIDE_Z=26 LUMINA_NLTE_ELEMENT_WIDE_SHELL=8 \
  LUMINA_NLTE_ELEMENT_WIDE_COMMIT=0 \
  LUMINA_NLTE_ION_LOCK=1 LUMINA_TOPSTAGE_IV=1 \
  "$RUNTIME_WRAPPED_BENCH" "$FROZEN" "$MODEL" "$POS_DIR"

env -i PATH="$PATH" W32_RUNTIME_COUNTER_ONLY=1 \
  W32_RUNTIME_DISABLE_PIN_TOPSTAGE=1 \
  LUMINA_FROZEN_ORACLE_ONLY_SHELL=8 \
  LUMINA_SUPER_LEVELS=0 LUMINA_NLTE_ELEMENT_WIDE=1 \
  LUMINA_NLTE_ELEMENT_WIDE_Z=26 LUMINA_NLTE_ELEMENT_WIDE_SHELL=8 \
  LUMINA_NLTE_ELEMENT_WIDE_COMMIT=0 \
  "$RUNTIME_WRAPPED_BENCH" "$FROZEN" "$MODEL" "$NEG_DIR"
```

결과: 실제 solve 양성은 `save_restore=1 per_ion_pin=15 topstage_IV=14`, 의도적
pin/topstage 음성은 `1/0/0`, 두 실행 모두 RC=0.

## 6. rung 5 — R7 writer 정직화

### 복원 물리/아티팩트 계약

CUDA producer가 dump 직전의 실제 `eta_total = chi_total*S_fixed + chi_es*J`를
별도 snapshot에 기록하고, writer가 payload의 `eta_total`과
`fixed+coherent`를 독립 재계산해 `max|eta_total-(fixed+coherent)|` 실값을
sidecar에 쓴다. payload 또는 sidecar write가 실패하면 payload를
`.quarantine`으로 이동한다.

선택 iteration은 `LUMINA_CMF_FROZEN_CHIETA_EXPECTED_ITER`가 반드시 존재하고
producer iteration과 정확히 같아야 한다. parity59 정본은 producer iter 10으로
`scripts/parity59_chieta.env`에 고정했다. selector도 sidecar 문자열을 신뢰하지 않고
payload를 다시 계산한다.

신규 음성대조는 잘못된 η를 selector가 거부하고, matrix diagonal debit 손상을
독립 D6 ledger가 거부하는지 확인한다.

### 사전등록 기대 변경 집합

payload 물리는 변하지 않는다. sidecar의 감사값, quarantine 동작, iter 계약과
seeded fixture만 바뀐다.

### patch

`patches/w32a3_rung5_r7_honest_writer.patch`

### 자기검증

```bash
make -B selftest_cmf_chieta_dump selftest_wave32_matrix_debit
python3 scripts/cmf_chieta_roundtrip_selftest.py --input "$VALID"
python3 tests/test_wave32_seeded_defects.py
```

결과: 정상 payload 424 bytes, SHA-256
`3981641ed3fa6f9bfac8425b248f546012501fac78094522816a22ab950c6d52`로
왕복 PASS. 잘못된 η selector RC=1, sidecar failure writer RC=1,
payload 부재 + `.quarantine` 존재. D6 baseline residual=0, seeded residual=0.25,
gate_pass=0. 실제 CUDA OFF 중립성과 iter=10 capture는 GPU 실행 금지로
**UNRESOLVED-until-capture**다.

## 7. rung 6 — within-SL 중첩 OOM 폐합

### 복원 운영 계약

`nlte_precompute_within_sl_frac()` 내부 임시 allocation 실패를 무시하지 않는다.
ABI를 유지하는 void wrapper와 checked 함수
`nlte_precompute_within_sl_frac_checked()`를 분리하고 private EW, CPU solve, GPU
host solve, frozen harness의 실제 소유자들이 checked 반환을 전파한다.

### 사전등록 기대 변경 집합

정상 allocation 경로는 byte 불변이다. 해당 allocation OOM에서만 조용한
dereference 대신 오류 반환이 생긴다.

### patch

`patches/w32a3_rung6_within_sl_oom.patch`

### 자기검증

```bash
make -B selftest_wave32_within_sl_oom bench_frozen_oracle
./selftest_wave32_within_sl_oom
# parity59 unarmed s0 replay 후 rung3 정상 파일과 cmp
```

결과: `[NLTE][OOM] ... allocation failed`, inner RC=-1, fixture RC=0. 정상 s0
frozen replay RC=0이고 oracle은 pre-rung6 파일과 byte-identical.

## 8. rung 7 — Fe M_V 경계질량 stage

### 복원 물리 계약

Fe II/III/IV super-level 뒤에 scalar `M_V`를 한 개 추가했다. Fe V 200개 target
identity의 `(E_t,g_t,T_e)`로 Boltzmann `q_t`를 만들고 합/min/max/checksum을
artifact에 기록한다. q_t 음수·비유한·합 불일치는 보정 없이 fail closed한다.

Fe IV 200/200 CMFGEN σ와 MA route만 IV↔V interface 생산자로 인정한다.
각 `(l,t)`에 대해 기존 EW와 같은 target threshold, 공유 bf field helper와 Milne
inverse를 사용한다.

- IV→V: `M_V` row / Fe IV source column에 `p_lt R_ion f_l`;
- V→IV: Fe IV row / `M_V` column에 `q_t p_lt R_rec`;
- collisional bf도 실제 생산 gate가 켜졌을 때 같은 규칙과 별도 plane;
- V→VI, Fe V bb, 신규 DR/autoionization producer는 만들지 않고 manifest에
  명시한다.

보존행은 `II+III+IV+M_V = n_Fe,total-M_outside`로 교체했다. `M_outside`는 입력
Fe I와 Fe VI+ 질량이며 0으로 가정하지 않는다. 해의 음수/비유한, 200/200 data,
route, projection, matrix/ledger, boundary flux, 보존 residual을 모두 commit gate에
연결했다. commit이 허용되면 Fe V total density도 scalar 해로 writeback한다.

신규 `boundary_mass` artifact는 질량, q_t, rad/coll gross forward/reverse/net flux,
matrix 대 독립 event-ledger residual, 외부 질량과 명시적 V→VI/DR 부재를 기록한다.

### 사전등록 기대 변경 집합

Fe 이외 원소는 불변이다. Fe에서는 새 scalar와 IV↔V rate 때문에 Fe stage 분율과
신규 boundary artifact만 바뀐다. s0 Fe IV/anchor는 1.0111에서 약 1로 이동하고
II/III/IV의 `d_k(elem)<d_k(pair)` 전항, D 약 58% 개선 유지, boundary gate 해소를
기대했다. s8은 실질 무변화를 기대했다.

### patch

`patches/w32a3_rung7_fe_v_boundary_mass.patch`

### 자기검증 명령

```bash
make -B bench_frozen_oracle selftest_wave32_boundary_q
./selftest_wave32_boundary_q

env -i PATH="$PATH" LUMINA_FROZEN_ORACLE_ONLY_SHELL=0 \
  LUMINA_SUPER_LEVELS=0 LUMINA_NLTE_ELEMENT_WIDE=1 \
  LUMINA_NLTE_ELEMENT_WIDE_Z=26 LUMINA_NLTE_ELEMENT_WIDE_SHELL=0 \
  LUMINA_NLTE_ELEMENT_WIDE_COMMIT=0 LUMINA_NLTE_ELEMENT_WIDE_DUMP=1 \
  LUMINA_NLTE_ELEMENT_WIDE_DUMP_DIR="$S0" \
  ./bench_frozen_oracle "$FROZEN" "$MODEL" "$S0"

# 같은 명령에서 ONLY_SHELL/EW_SHELL/DUMP_DIR만 8로 바꿔 s8 replay
```

q_t seeded fixture 결과:

```text
good=1 sum=1 negative=0 nonfinite=0 bad_sum=0
matrix_good=0 bad_debit=0.25 bad_target=0.25
```

### s0 실측

실행 디렉터리: `/tmp/w32a3_r7s0.M6rnwA` (임시 증거)  
결과: RC=0, `EW_PASS`, rank `304/304`, κ₂ `2.5544e4`, scaled SE residual
`1.5255e-15`, conservation `3.5966e-16`, 음수/비유한 0, 세 gate 모두 1.

| 양 | 값 |
|---|---:|
| Fe total | `1.657257776693131e8 cm^-3` |
| M_V before / Fe | `0.01374908015` |
| M_V after / Fe | `0.01709051580` |
| M_outside / Fe | `6.95593e-14` |
| Fe II/III/IV | `2.70401e-13 / 3.23485e-5 / 0.982877136` |
| Fe IV / 조건부 anchor(0.989) | `0.993809` |
| q count / sum | `200 / 0.99999999999999967` |
| matrix-event flux residual | `2.13867e-16` |
| forward / reverse flux | `68041.773126977947 / 68041.773126977889` |

조건부 정본의 반올림 anchor를 사용한 stage별 오차는
`1.56494 / 0.97445 / 0.002697`로, pair의
`4.8875 / 1.1385 / 0.00306`보다 전 항목 작다. D(세 d의 산술평균)는
`2.0097→0.84736`, 즉 57.84% 개선으로 기존 58% 개선을 유지한다.

방향과 gate는 사전등록을 만족했지만 M_V의 크기는 예상 약 1.1%가 아니라
**1.709%**였다. Fe IV/anchor도 정확한 1.000이 아니라 0.9938이다. 이 차이를
재튜닝하거나 q_t를 조정하지 않았으며, 사전등록 크기 예측의 편차로 남긴다.

### s8 실측

실행 디렉터리: `/tmp/w32a3_r7s8.5UFu61` (임시 증거)  
결과: RC=0, `EW_PASS`, rank `304/304`, scaled SE residual `1.1646e-15`,
conservation `1.0084e-15`, boundary gate=1.

기존 post-repair Fe II/III/IV 분율 대비 최대 절대 변화는
`2.4261e-11`; 새 M_V/Fe는 `4.0084e-11`이다. 따라서 s8 실질 무변화 기대를
만족한다.

## 9. 최종 회귀와 남은 항목

최종 fixture 묶음:

```bash
make -B selftest_wave32_matrix_debit selftest_wave32_within_sl_oom \
  selftest_wave32_boundary_q selftest_cmf_chieta_dump
./selftest_wave32_matrix_debit
./selftest_wave32_within_sl_oom
./selftest_wave32_boundary_q
python3 tests/test_wave32_seeded_defects.py
make -B
```

결과: 모든 fixture RC=0. 전체 build RC=0, compiler error 0, 기존 계열 warning
60. build 중 GPU binary 실행은 없었다.

남은 항목은 구현 실패로 숨기지 않는다.

1. 실제 CUDA producer OFF 중립성 및 parity59 producer iter=10 capture:
   **UNRESOLVED-until-capture** (GPU 실행 금지).
2. 성공 `COMMIT=1` pilot만 변화/off-target byte 불변: gate 해소까지는 구현·shadow
   입증됐고, 양성 격리 판정은 발주문대로 후속 B3 범위다.
3. s0 M_V 크기 1.709%는 사전 예상 1.1%보다 크다. stage별 acceptance와 58%
   개선은 통과했지만 이 수치 편차 자체는 후속 물리 해석 대상이다.
