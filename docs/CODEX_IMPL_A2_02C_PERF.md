# A2-02C segment replay 성능 수리 보고

## 범위와 불변 조건

변경 범위는 `scripts/a2_02c_segment_replay.py`의 offline replay 성능 경로와 구현
대조 gate뿐이다. segment 선형 energy 적분, 16-point Gauss-Legendre line profile
적분, `4*pi*V_s*delta_t` 정규화, P-prefix 규칙, 1%/0.2% 합격선, production output
schema와 7종 음성대조는 바꾸지 않았다. `src`, 덱, `/gpfs` 파일은 변경하지 않았다.

## 변경 함수

- `capture_layout`, `worker_capture`: header만 읽고 capture를 memmap으로 여는 경량
  경로다. spawn worker마다 경로를 받아 자체 memmap을 열고 process-local cache에 둔다.
  capture 배열은 Pool 인자로 보내지 않는다.
- `read_capture`: 물리 필드와 packet/segment identity 검사를 1,000,000-record 청크로
  수행하며 청크마다 진행률을 출력한다. capture 크기의 정렬 배열은 만들지 않는다.
- `raw_hist_legacy`, `line_raw_legacy`, `jbar_from_fine_legacy`: 수리 전 행 단위
  Python 구현을 보존한 reference 경로다. `run --legacy-slow`와 implementation gate만
  사용한다.
- `_accumulate_hist_chunk`, `raw_hist`, `raw_hist_efforts`: edge 교차 bin을
  `searchsorted`로 구하고 `(record, sub-interval)`을 전개한다. 각 구간 값은
  `F(x)=length*(e0*x+0.5*(e1-e0)*x^2)`의 차이며 `np.add.at`으로 누산한다.
  기본 record 청크는 1,000,000이고 전개 배열은 4,000,000 interval에서 재분할한다.
  production global histogram은 한 capture scan에서 active shell과 P/2P를 함께 만든다.
- `packet_prefix_counts`: independent capture는 global/diagnostic histogram을 다시 만들지
  않고 한 청크 scan으로 2P prefix completeness와 segment count만 검증한다.
- `_line_chunk_values`, `line_raw`: 선택 segment 전체의 16점 Gauss 적분을 배열 연산으로
  바꾸고, 임시 `(record,16)` 배열은 65,536행 단위로 제한한다.
- `_cohort_row_worker`, `compute_fast_components`: 74 cohort 행을 독립 Pool 작업으로
  실행한다. worker 수는 정확히 `min(60, os.cpu_count())`이며 `spawn` context를 쓴다.
  한 worker scan에서 P/2P direct estimator, packet variance, canonical-bin histogram,
  12/24 bins-per-Doppler histogram을 함께 계산한다.
- `compute_legacy_components`, `effort_view`, `assemble_result`, `calculate`: 구/신 계산
  component를 동일한 기존 판정·schema 조립 함수에 넣어 산식 분기를 없앴다.
- `implementation_gate`: pilot을 legacy와 vector+Pool 양쪽으로 재생하고 hist,
  estimator, delta를 상대 `1e-12` gate로 각각 비교한다.

진행 로그는 validation/global record 청크 및 cohort 작업 완료마다 한 줄이다. 예:

```text
[replay] validate capture=a2_02c_segments.bin chunk 12/105 records=12000000/105000000
[replay] global capture=a2_02c_segments.bin chunk 12/105 records=12000000/105000000
[replay] cohort 12/74 shell=8 line=1332571 capture=a2_02c_segments.bin workers=60
```

## 구현 대조 gate 결과 형식

gate output schema는 `lumina-a2-02c-replay-implementation-gate-v1`이다. 핵심 형식은
다음과 같다.

```json
{
  "comparisons": {
    "hist": {"values_compared": 0, "maximum_relative_error": 0.0,
             "tolerance": 1e-12, "passed": true},
    "estimator": {"values_compared": 0, "maximum_relative_error": 0.0,
                  "tolerance": 1e-12, "passed": true},
    "delta": {"values_compared": 0, "maximum_relative_error": 0.0,
              "tolerance": 1e-12, "passed": true}
  },
  "elapsed_seconds": {
    "legacy_slow": 0.0,
    "vector_mp": 0.0,
    "speedup": 0.0
  },
  "pilot_extrapolation": {
    "target_records_per_capture": 105000000,
    "capture_count": 2,
    "conservative_linear_seconds": 0.0,
    "model": "2 * target_records/pilot_records * vector_mp_seconds"
  },
  "passed": true
}
```

`hist`는 P/2P global raw histogram과 각 cohort의 canonical/12/24-bin raw histogram
전체, `estimator`는 raw direct value, packet variance 및 production line-record의
`jbar_value`/variance/standard error, `delta`는 packet convergence와 세 closure의
record별 상대 변화 전체를 포함한다. count, validity와 null identity는 별도로 exact
일치를 요구한다. 세 numeric group 모두 최대 상대오차 `<=1e-12`여야 rc=0이다.

## lageunha 운전석 명령과 기대 rc

로그인 노드에서는 아래 pilot/full replay를 실행하지 않는다.

```bash
ssh lageunha 'bash -s' <<'EOF'
set -euo pipefail
REPO=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
OUT=$REPO/validation/a2_02c
cd "$REPO"

PYTHONPYCACHEPREFIX=/tmp/a2_02c_perf_pycache python3 -m py_compile \
  scripts/a2_02c_segment_replay.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/a2_02c_segment_replay.py self-test
PYTHONDONTWRITEBYTECODE=1 python3 scripts/a2_02c_segment_replay.py negative-controls

PYTHONDONTWRITEBYTECODE=1 python3 scripts/a2_02c_segment_replay.py implementation-gate \
  --capture "$OUT/a2_02c_segments_pilot_g2_2P4000.bin" \
  --cohort "$OUT/A2_02C_ESTIMATOR_COHORT.json" \
  --union "$OUT/A2_02C_FREQUENCY_UNION.json" \
  --global-bins 4000 \
  --effort 2000 --double-effort 4000 \
  --chunk-records 1000000 \
  --output "$OUT/a2_02c_replay_implementation_gate.json"
EOF
```

각 명령의 기대 rc는 모두 **0**이다. 마지막 marker는
`A2_02C_REPLAY_IMPLEMENTATION_GATE PASS tolerance=1e-12 ...`다. legacy reference를
별도로 재생할 때는 기존 `run` 명령에 `--legacy-slow`만 추가한다.

production 두 capture replay는 다음과 같다.

```bash
ssh lageunha 'bash -s' <<'EOF'
set -euo pipefail
: "${A2_02C_P:?}" "${A2_02C_CAPTURE:?}" "${A2_02C_CAPTURE_INDEPENDENT:?}" \
  "${A2_02C_SELECTED_BINS:?}"
REPO=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
OUT=$REPO/validation/a2_02c
cd "$REPO"
PYTHONDONTWRITEBYTECODE=1 python3 scripts/a2_02c_segment_replay.py run \
  --capture "$A2_02C_CAPTURE" \
  --independent-capture "$A2_02C_CAPTURE_INDEPENDENT" \
  --cohort "$OUT/A2_02C_ESTIMATOR_COHORT.json" \
  --union "$OUT/A2_02C_FREQUENCY_UNION.json" \
  --global-bins "$A2_02C_SELECTED_BINS" \
  --effort "$A2_02C_P" --double-effort "$((2 * A2_02C_P))" \
  --chunk-records 1000000 \
  --output "$OUT/a2_02c_estimator_effort_result.json"
EOF
```

모든 기존 물리 gate까지 통과하면 기대 rc는 **0**이다. 유효 계산이지만 convergence나
closure/independent gate가 실패하면 기존대로 rc=3, 입력/schema 오류는 rc=2다.

## 파일럿 실측 기반 성능 외삽

pilot header의 실제 크기는 1,068,485 records다. 요청된 production 크기
105,000,000 records/capture의 비는 `98.270`이고 두 capture 보수 외삽 계수는
`196.540`이다. 따라서 운전석 gate가 기록한 `elapsed_seconds.vector_mp = T_pilot`에
대해 보수 예상치는 다음과 같다.

```text
T_full_two_capture_seconds = 196.540 * T_pilot_seconds
T_full_two_capture_minutes = 3.2757 * T_pilot_seconds
```

이 값은 두 번째 independent capture도 P/2P diagnostic을 모두 수행한다고 보는 상한
모델이다. 실제 fast run의 independent 쪽은 2P direct estimator만 필요하므로 보통 이
선형값보다 짧다. gate JSON의 `pilot_extrapolation.conservative_linear_seconds`가 같은
식을 pilot 실측 직후 자동으로 채운다. pilot gate 자체는 운전석 전용이므로 이 로그인
노드 세션에서는 실행하거나 실측값을 임의로 기입하지 않았다.

## 이 세션의 경량 검증

대형 pilot/production capture는 실행하지 않았다.

```text
python -m py_compile: PASS
A2_02C_CAPTURE_GATE_SELFTEST: PASS
A2_02C_REPLAY_SELFTEST: PASS (same_measure, segment_split, profile_integral)
A2_02C_REPLAY_NEGATIVE_SUMMARY: 7/7 PASS
20,000-record randomized legacy/vector histogram max relative error: 1.63e-16
20,000-record randomized legacy/vector line max relative error: 7.67e-16
4-record synthetic end-to-end implementation gate: PASS
  hist=0, estimator=0, delta=0 maximum relative error
```
