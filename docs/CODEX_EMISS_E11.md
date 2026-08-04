# E11 — 형광 재분배 행렬의 정식 추정기 구현

작성일: 2026-08-02 (Asia/Seoul)  
범위: 생산 수송 계측 구현, CUDA build, 오프라인 binary/왜곡/E10 소비자 fixture.
신규 모델·GPU transport run, clamp/floor/fallback, 정규화 보정, commit 없음. 실제
모델 capture 및 pre-patch 대비 OFF byte 대조 제출은 운전석 범위다.

## 1. 판정

**구현·build·오프라인 fixture는 PASS다. 실제 모델 수치와 production OFF byte 대조는
CAPTURE-PENDING이다.** E9의 400M-record capped event prefix 복원 대신, 모든 실제 선
상호작용에서 입력/출력 comoving frequency와 packet energy를 반환 직후 직접 누적한다.
별도 event buffer, packet pairing, 저장 cap 또는 prefix가 없다. 기존 transport 자체의
`max_interactions`는 바꾸지 않았다.

unset gate는 host/device 메모리를 할당하지 않고 파일을 쓰지 않는다. device 기록 함수는
`d_fluor_matrix_on==0`에서 즉시 반환하며 atomics, RNG draw, packet write가 없다. 추가된
`d_line_scatter_event` 출력은 thread-local 관측 flag뿐이다. 오프라인 seeded packet oracle은
unset/empty gate 2,560 bytes가 SHA-256
`d9936c055cd15e7aa49e70a9bf0b94d6b30aa1e74ba5e919bfda739a86ae7e5d`로 byte-identical임을
확인했다. 다만 차터가 금지한 GPU/model run 없이 pre-patch production binary와 patched
binary의 실제 spectrum artifact를 비교할 수 없으므로, 그 최종 실증을 완료했다고
과장하지 않는다. `scripts/emiss_e11_off_byte_check.py`가 운전석 capture를 fail-closed로
판정한다.

## 2. 추정기

### 2.1 기록 시점과 정의

`transport_kernel`의 유일한 LINE interaction 호출점에서 다음을 기록한다.

```text
input_bin  = bin(nu_comov immediately before d_line_scatter_event)
output_bin = bin(nu_comov immediately after  d_line_scatter_event)
R_raw[input_bin,output_bin] += emitted_comoving_packet_energy
```

입력/출력 에너지는 각각 old/new Doppler factor로 lab packet energy를 comoving energy로
변환한 값이다. matrix에는 출력 에너지를 넣고, 독립 `input_energy[input_bin]`과
`terminal_energy[input_bin]` 장부를 함께 쓴다. 소비자는
`R[j,i]=edge_output_energy/terminal_energy[i]`를 사용한다. 값을 맞추기 위한 rescale,
column renormalization 또는 미관측 edge fill은 없다.

다음은 matrix와 별도로 보존한다.

- 전체 event/classified event
- input-bin 밖, output-bin 밖, invalid energy, unresolved MA route 카운트
- 전체 absorbed/reemitted energy와 `(reemitted/absorbed)-1`
- k-packet 경유 event와 absorbed/reemitted energy
- 입력 빈별 event/input/terminal/outside-grid energy
- 셸별 event/k-packet event/input/output energy

`transition_type=-2/-3/-4` 또는 line-activated cascade의 k-packet re-excitation router를
지난 경우를 k-packet 경유로 분류한다. MA internal cap에서 terminal emission이 확정되지
않은 route는 대각 edge로 숨기지 않고 `unclassified_route`로 빠진다.

### 2.2 셸 분해와 메모리 상한

device에서는 sparse hash overflow/충돌/용량 절단을 피하기 위해 bounded dense accumulator를
쓰고 파일에서만 sparse edge로 직렬화한다.

| 항목 | 배열 수 | 1000-bin device bytes |
|---|---:|---:|
| 전역 matrix | 1 | 8,000,000 (7.63 MiB) |
| deep shells | 0--4 | 8,000,000 |
| photospheric shells | 5--12 | 8,000,000 |
| envelope shells | 13--끝 | 8,000,000 |
| 합계 matrix | 4 | **32,000,000 (30.52 MiB)** |

50개 셸 full decomposition은 400,000,000 bytes(381.47 MiB)이므로 이번 rung에서 쓰지
않았다. 대신 위 세 대표 셸군 matrix와 50개 전 셸의 1차 장부를 동시에 둔다. 1000-bin
전역 matrix의 entry 상한은 정확히 1,000,000이고 sparse dump 한 matrix의 최악 edge
payload는 16,000,000 bytes다. 따라서 device와 dump 모두 event 수와 무관한 유한 상한을
가진다.

각 MC pass 시작에 reset하고 완료 직후 같은 지정 path를 정규화 보정 없이 다시 쓴다.
따라서 최종 파일은 최종 완료 iteration의 동시대 matrix이며 여러 iteration을 섞지 않는다.

## 3. gate와 LFMAT001 v1

```bash
LUMINA_FLUOR_MATRIX_DUMP=/absolute/path/formal_matrix.bin
```

환경 변수가 없거나 empty이면 OFF다. 순수 CMFGEN이며 THEN_MC/MC_COEVOLVE가 없는 상태에서
path를 지정하면 빈 파일을 만드는 대신 fail-loud한다.

모든 정수/실수 필드는 little-endian이다.

```text
header = 8s magic "LFMAT001"
         7*u32 endian=0x01020304, version=1, flags=15,
               n_bins, n_shells, iteration, n_shell_groups
         3*f64 nu_min, nu_max, d_log_nu
         7*u64 events, classified, unclassified_input,
               unclassified_output, unclassified_energy,
               unclassified_route, kpacket_events
         4*f64 absorbed, reemitted, kpacket_absorbed, kpacket_reemitted
         u64 global_nnz
body   = u64 input_count[n_bins]
         f64 input_energy[n_bins], terminal_energy[n_bins], outside_energy[n_bins]
         u64 shell_count[n_shells], shell_kpacket_count[n_shells]
         f64 shell_absorbed[n_shells], shell_reemitted[n_shells]
         global_nnz * {u32 input_bin, u32 output_bin, f64 output_energy}
         for each group:
             {u32 first_shell, u32 last_shell, u64 nnz}
             nnz * {u32 input_bin, u32 output_bin, f64 output_energy}
```

`<path>.sha256`는 외부 executable에 의존하지 않고 생산 binary 내부 SHA-256 구현으로
생성한다. 판독기는 checksum, magic/endian/version/flags, exact file length, grid/range,
duplicate/negative/nonfinite edge, input-column energy closure, 대표군 합=전역 matrix를
검증한다.

## 4. 음성 대조와 소비자

seeded fixture는 `(input=2, output=0)` edge만 정확히 7배로 바꾸고 terminal ledger를
건드리지 않았다. 정상본 column closure는 0, 왜곡본은 input 2에서
`matrix+outside=22`, `terminal=10`, relative mismatch **1.2**로 거부됐다. 측정된 edge
배율은 정확히 7.0이다. clamp, normalization repair, fallback은 모두 0이다.

`scripts/emiss_e10_apply_redistribution.py`에는
`--matrix-format auto|prefix|formal`을 추가했다. `auto`는 magic `LFMAT001`을 판독한다.

- 기존 E9 prefix CSV/normalization/summary 경로는 그대로 남았다.
- prefix regression output SHA-256은 기존과 새 소비자 모두
  `e64a59c4b2a6b760443b53ca7fa5f4d2208ce40b222322db8e0122ece56e79fd`로
  byte-identical이다.
- 1000-bin identity formal fixture 적용은 removed/injected
  `0.006765195798996928`, application relative error 0, full-source relative error 0,
  negative/nonfinite/clamp/fallback 모두 0이다. fixture identity는 소비자 wiring test일 뿐
  production missing-edge fallback이 아니다.
- formal matrix는 전역+대표 셸군 operator다. 현재 E10 frozen probe는 선택한 shell 8에
  전역 operator를 적용하며 다른 49개 shell source는 기존처럼 건드리지 않는다.

## 5. rung별 기대 변경집합

정확한 목록은 `patches/e11_expected_changes.txt`에 있다.

1. Rung 1: `src/lumina_cuda.cu` — 직접 누적, 정직 장부, k-packet/미분류 계측.
2. Rung 2: `src/lumina_cuda.cu`, `src/lumina_cmfgen.[ch]` — LFMAT001 + SHA sidecar.
3. Rung 3: `scripts/emiss_e11_*.py`, `Makefile` — reader, 왜곡/OFF fixture, capture checker.
4. Rung 4: `scripts/emiss_e10_apply_redistribution.py` — formal reader, prefix 보존.
5. Rung 5: 이 보고서와 expected-change ledger.

신규 clamp/floor/fallback/normalization repair는 0이다. launcher/model deck 수정도 0이다.

## 6. 실행한 검증과 결과

```bash
# CUDA compile only; exit 0. 기존 nlte_gemm unused-variable warning만 존재.
make lumina_cuda

# Python syntax + canonical/negative/OFF/1000-bin fixtures
make selftest_emiss_e11_fluor_matrix

# 정상/왜곡 판독
python3 scripts/emiss_e11_fluor_matrix.py \
  /tmp/emiss_e11_fixture/formal_matrix_base.bin
python3 scripts/emiss_e11_fluor_matrix.py \
  /tmp/emiss_e11_fixture/formal_matrix_distorted.bin  # expected exit 2

# E10 formal consumer integration
python3 scripts/emiss_e10_apply_redistribution.py \
  --matrix /tmp/emiss_e11_fixture/formal_matrix_identity_1000.bin \
  --matrix-format formal --out-dir /tmp/emiss_e11_apply
python3 scripts/cmf_chieta_check.py \
  /tmp/emiss_e11_apply/emiss_e10_redistributed_iter10

# 기존 prefix regression (cmp exit 0)
python3 scripts/emiss_e10_apply_redistribution.py \
  --out-dir /tmp/emiss_e11_prefix_apply
cmp validation/emiss_e10/emiss_e10_redistributed_iter10 \
    /tmp/emiss_e11_prefix_apply/emiss_e10_redistributed_iter10

# 공유 SHA helper가 기존 LCMFCE01 writer를 바꾸지 않았는지 확인
make selftest_cmf_chieta_dump
./selftest_cmf_chieta_dump /tmp/e11_cmf_chieta_fixture
python3 scripts/cmf_chieta_roundtrip_selftest.py \
  --input /tmp/e11_cmf_chieta_fixture --no-build

git diff --check -- Makefile src/lumina_cuda.cu src/lumina_cmfgen.c \
  src/lumina_cmfgen.h scripts/emiss_e10_apply_redistribution.py \
  scripts/emiss_e11_fluor_matrix.py scripts/emiss_e11_seeded_fixture.py \
  scripts/emiss_e11_off_byte_check.py patches/e11_expected_changes.txt \
  docs/CODEX_EMISS_E11.md
```

## 7. 운전석 capture 명령

아래는 이번 작업에서 실행하지 않았다. 동일 seed/deck의 pre-patch와 patched binary를
각각 gate unset으로 실행한 뒤 spectrum/state artifact를 byte 비교해야 OFF 실증이
완료된다. run command와 artifact 목록은 운전석의 승인된 deck에 맞춰 명시한다.

```bash
# A: 승인된 pre-patch reference binary, gate unset
env -u LUMINA_FLUOR_MATRIX_DUMP "$E11_PRE_BIN" ...

# B: patched binary, 같은 seed/deck, gate unset
env -u LUMINA_FLUOR_MATRIX_DUMP "$E11_NEW_BIN" ...

python3 scripts/emiss_e11_off_byte_check.py "$E11_RUN_PRE" "$E11_RUN_OFF" \
  --artifact lumina_spectrum.csv \
  --artifact lumina_spectrum_formal.csv \
  --artifact lumina_plasma_state.csv \
  --out "$E11_RUN_OFF/e11_off_byte_check.json"

# C: 정식 matrix capture (event log 불필요)
LUMINA_FLUOR_MATRIX_DUMP="$E11_RUN_ON/formal_matrix.bin" \
  "$E11_NEW_BIN" ...
python3 scripts/emiss_e11_fluor_matrix.py \
  "$E11_RUN_ON/formal_matrix.bin" \
  > "$E11_RUN_ON/formal_matrix_summary.json"
sha256sum "$E11_RUN_ON/formal_matrix.bin" \
  "$E11_RUN_ON/formal_matrix.bin.sha256"
```

capture가 도착하면 `events_total`, 네 미분류 카운트, 전체/k-packet energy closure,
global 및 세 shell-group B2→B0 edge를 같은 iteration source와 함께 판독해야 한다. 그
전에는 E10의 B0 20.91 악화가 prefix bias였는지에 대한 물리 판정을 갱신하지 않는다.
