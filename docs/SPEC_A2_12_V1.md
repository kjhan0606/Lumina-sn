# A2-12 구현 명세 V1 — GPU 정본 mirror lifecycle과 generation 결박

상태: **구현 전 규범 명세**

저작 역할: A2-12 구현 명세 저작자 (개정 11)

대상 단계: A-2 캠페인의 **A2-12 하나**

저작 기준 HEAD: `068fb36`

이 문서의 `MUST`, `MUST NOT`, `SHALL`, `FAIL`, `BLOCKED`는 시험 가능한 계약어다.
구현 편의를 이유로 완화할 수 없다.

## 0. 결론

A2-12의 단일 계약은 다음과 같다.

> CPU의 유일한 정본 `RadiationField`와 그 안의 파생 `LineJbarCache`를 하나의
> generation-bound GPU mirror transaction으로 할당·업로드·reset·동기화한다. GPU
> launch 직전 CPU 정본 generation, GPU generation, line-cache generation, line ID
> mapping, profile/Q-set identity, shape, validity가 모두 일치해야 한다. 하나라도 다르거나
> 업로드가 부분 성공했으면 수치값을 공개하지 않고 명시적으로 실패한다.

정본은 계속 CPU `RadiationFieldOwner` 하나다. GPU 객체는 소유자가 아니라 폐기 가능한
read-only mirror다. GPU에서 별도 generation을 생산하거나 CPU로 역게시하지 않는다.

A2-12의 합격은 GPU rate 수치가 맞다는 뜻이 아니다. 이 단계는 **lifecycle만** 닫는다.
현재 scalar 또는 legacy accumulator를 읽는 GPU 물리 경로는 해당 후속 단계가 이관할 때까지
명시적으로 실행 불가여야 한다.

## 1. 상위 규범과 현재 사실

### 1.1 규범

- `docs/ORDER_L0_JNU_OWNER_BY_CODEX.md:57-112`의 단일 정본, generation, frame, unit,
  validity 계약.
- `docs/ORDER_L0_JNU_OWNER_BY_CODEX.md:667-670`의 단계 경계. A2-12는 GPU
  소유권·lifecycle, A2-13은 GPU rate, A2-14는 GPU opacity, A2-15는 GPU emissivity다.
- `docs/ORDER_L0_JNU_OWNER_BY_CODEX.md:679-681`의 GPU 실행 위치와
  `/usr/bin/time` 금지.
- `docs/ORDER_L0_JNU_OWNER_BY_CODEX.md:689-710`의 단계 회귀 대장 필드.
- `docs/ORDER_L0_JNU_OWNER_BY_CODEX.md:720-773`의 최종 점검과 무증상 실패, 특히 CPU
  갱신 뒤 GPU upload 누락 및 한 iteration 어긋난 reset.
- `docs/ORDER_A2_GRID_AMENDMENT_BY_CODEX.md:138-195`의 `LineJbarCache` 비독립
  lifecycle과 원자적 dual-view commit.
- `docs/ORDER_A2_GRID_AMENDMENT_BY_CODEX.md:363-372`의 A2-12 변경표: line-cache
  ID mapping·generation·validity·값의 upload/reset/sync, stale·CPU/GPU generation
  차이·partial upload의 명시적 실패, cache upload bytes 기록.
- `docs/ORDER_A2_GRID_AMENDMENT_BY_CODEX.md:390-413`의 단계 보류와 음성대조 5.
  cache generation을 하나 늦추거나 line ID mapping을 shuffle하면 GPU rate 실행 전에
  FAIL해야 한다.

### 1.2 저작 HEAD의 사실

- CPU schema는 `src/radiation_field.h:89-152`에 있다. `LineJbarCache`는
  `RadiationFieldOwner` 내부 파생 객체다.
- checked CPU view는 `src/radiation_field.h:235-303`의 `RadiationFieldView`,
  `LineJbarView`, `line_jbar_lookup`이다.
- dual commit은 `src/radiation_field.c:515-646`, checked view는
  `src/radiation_field.c:696-818`에 있다.
- 현행 CUDA main에는 `radiation_field_owner_init`, `radiation_field_commit`,
  `radiation_field_read_view`, `radiation_field_line_jbar_view` 호출이 **0개**다.
- `src/lumina_nlte_assemble.cu:88-91,353-357,401-414`의 `d_J_nu/d_W`는 canonical
  view upload가 아니라 legacy `nlte->J_nu/plasma->W` 복사다.
- `src/lumina_cuda.cu:131-132,2977-2990,7994-8001,8793-8796,9206-9211,
  10165-10195,10444-10449`의 `d_jbar_line/d_jbar_count`는 GPU에서 생산해 CPU로
  내려받는 legacy MC accumulator다. CPU 정본 `LineJbarCache`의 GPU mirror가 아니다.
- 따라서 기존 심볼 이름을 재사용해 "이미 upload 중"이라고 판정하면 FAIL이다.

## 2. 단계 경계: A2-13~15 침범 금지

### 2.1 A2-12가 하는 일

1. 정본 두 checked view로부터 GPU mirror의 크기 계산, 할당, upload, 검증, 게시, reset,
   free를 담당한다.
2. CPU required/computed generation과 GPU committed generation을 launch 직전에 비교하는
   단일 readiness gate를 제공한다.
3. line ID mapping, Q-set/profile identity, validity 및 값의 동일성을 검증한다.
4. upload byte와 lifecycle failure counter를 기록한다.
5. 아직 이관되지 않은 GPU rate/opacity/emissivity launch를 명시적 상태코드로 차단한다.

### 2.2 A2-12가 하지 않는 일

- `R_lu`, `R_ul`, BF `Gamma`, SE matrix의 물리식 또는 kernel arithmetic 변경: **A2-13**.
- `chi_nu`, Sobolev opacity, BF population/coefficient kernel 변경: **A2-14**.
- `eta_nu`, Planck 재방출, macro-atom redistribution, packet sampling 변경: **A2-15**.
- CPU rate·opacity·emissivity·transfer 식 변경.
- GPU에서 coarse/fine `J_nu`를 재적분해 `Jbar`를 새로 생산하는 일.
- 192,922-bin fine dump를 런타임 GPU 정본으로 도입하는 일.

후속 물리 kernel은 A2-12 mirror descriptor를 받아야 하지만, A2-12 diff에서 kernel 수식을
새 정본 값으로 바꾸지 않는다. 현행 scalar 물리 입력을 요구하는 launch는 조용한 CPU
fallback 없이 각각 `GPU_RATE_NOT_MIGRATED`, `GPU_OPACITY_NOT_MIGRATED`,
`GPU_EMISSIVITY_NOT_MIGRATED`로 끝낸다. 이 상태는 A2-12 lifecycle gate의 PASS를 막지
않지만 생산 GPU 물리 PASS로 보고할 수 없다.

## 3. GPU mirror 자료형과 불변식

구현 이름은 달라도 되지만 의미와 단일성은 다음과 같아야 한다.

```text
GpuRadiationFieldMirror {
    state = EMPTY | DIRTY | UPLOADING | READY | FAILED

    d_frequency_bin_edges       double[n_bins+1]
    d_J_nu                      double[n_shells*n_bins]
    d_radfield_validity         fixed-width enum[n_shells*n_bins]

    d_line_id                   uint64[n_lines]          # ascending Q_g mapping
    d_line_jbar                 double[n_lines*n_shells]
    d_line_validity             fixed-width enum[n_lines*n_shells]
    d_line_count                uint64[n_lines*n_shells]
    d_line_se                   double[n_lines*n_shells]

    cpu_required_generation
    cpu_computed_generation
    line_required_generation
    line_computed_generation
    gpu_committed_generation
    n_shells, n_bins, n_lines
    edge_sha256, q_set_sha256, profile_id, profile_sha256
    units, frame, epoch
    upload_serial
    ready_event
    counters
}
```

`d_line_count`와 `d_line_se`는 A2-13에서 직접 쓰지 않더라도 validity 증명과 진단 재현을
위해 같은 transaction으로 올린다. 이를 빼서 byte 수를 낮춰 보고하면 FAIL이다.

필수 불변식은 다음과 같다.

```text
READY iff
  owner.enabled
  AND CPU field.required == CPU field.computed == expected_generation
  AND line.required == line.computed == expected_generation
  AND gpu_committed_generation == expected_generation
  AND units/frame/epoch/n_shells/edge hash/shape are canonical and equal
  AND line ID is strictly ascending and unique
  AND q_set/profile identity and n_lines are equal
  AND every required validity code is representable without remapping
  AND the whole upload event completed successfully
  AND uploaded-byte ledger equals the checked size equation
```

GPU mirror에 독립 setter, 독립 generation increment, raw public pointer getter를 만들지 않는다.
후속 kernel은 opaque checked descriptor만 받는다. descriptor 안의 pointer 하나만 골라
직접 launch하는 API도 금지한다.

## 4. lifecycle state machine

### 4.1 CPU generation이 먼저다

CPU producer가 새 `required_generation=g`를 연 순간 기존 GPU mirror는 논리적으로
`DIRTY`다. 물리 메모리에 이전 값이 남아 있어도 `gpu_require_ready(owner,g)`는 반드시
실패한다. CPU required generation을 매 launch 때 읽으므로 invalidate 호출 하나가
누락돼도 stale mirror가 실행될 수 없어야 한다.

CPU dual commit이 실패하면 GPU에 새 generation을 게시하지 않는다. 이전 GPU allocation을
진단을 위해 보존할 수는 있지만 CPU required와 다르므로 소비 불가다.

### 4.2 reset

reset은 pointer를 NULL로 보이게 하거나 READY stamp를 먼저 지워 **소비 자격을 먼저
박탈**한 뒤 수행한다.

- `gpu_committed_generation=0`, `state=DIRTY`를 host/device launch descriptor 모두에 반영.
- line ID, values, validity 중 한 배열만 reset한 상태를 READY로 공개하지 않음.
- `cudaMemset(...,0)`은 보안성/결정론적 cleanup일 뿐 validity를 `EXACT_ZERO`로 만드는
  연산이 아니다.
- reset generation이 CPU보다 하나 빠르거나 늦으면 `GPU_RESET_GENERATION_MISMATCH`.
- stream 비동기 reset은 event 완료 전 다음 upload/launch와 겹치지 않게 한다.

### 4.3 transactional upload

1. CPU owner에서 같은 `g`의 `RadiationFieldView`와 `LineJbarView`를 연속 취득한다.
2. 두 view 취득 뒤 owner의 required/computed generation을 다시 읽어 TOCTOU 변경이 없음을
   확인한다. 바뀌었으면 재시도하지 말고 현재 attempt를 `GPU_CPU_CHANGED_DURING_UPLOAD`로
   실패시킨다.
3. size overflow와 allocation byte 식을 checked arithmetic으로 검증한다.
4. public mirror와 분리된 candidate buffer 전부를 할당한다.
5. edge, J, field validity, line ID, Jbar, line validity, count, SE와 metadata를 한 stream에
   복사한다.
6. device-side 또는 D2H attestation fixture로 generation·mapping·sentinel·byte 수를
   검사하고 event를 동기화한다.
7. 전부 성공한 경우에만 candidate descriptor를 한 번에 READY로 publish한다.
8. 실패하면 candidate 전부 free하고 public descriptor/generation을 갱신하지 않는다.
   CPU required가 전진한 상태라 이전 descriptor도 readiness에서 실패해야 한다.

in-place overwrite, "먼저 J 공개 후 cache 복사", 배열별 generation stamp, 실패한
`cudaMemcpy` 뒤 나머지 배열만 계속 쓰기는 모두 금지한다. double buffer의 일시 peak
allocation도 보고한다.

### 4.4 validity와 fallback 금지

- `VALID`와 `EXACT_ZERO`만 후속 수치 소비 자격이 있다. `EXACT_ZERO`는 명시적 0이다.
- `UNSAMPLED`, `OUT_OF_GRID/OUT_OF_BB_DOMAIN`, `STALE`, MISS, unknown enum은 값이 0이어도
  launch를 차단한다.
- cache miss에 coarse/fine `J_nu` 적분, 이전 generation, `W B_nu(T_rad)`, 이웃 line,
  배열 0, `1e-30`을 대입하지 않는다.
- field/cache 어느 한쪽만 upload된 partial state는 소비하지 않는다.
- CUDA allocation/copy/event 오류를 CPU path 실행으로 바꾸지 않는다. 명시적 실패다.

## 5. 단일 readiness gate와 상태코드

모든 A2-13~15 물리 launch는 같은 함수의 성공 뒤에만 호출할 수 있어야 한다.

```text
gpu_radiation_field_sync(owner, expected_generation, mirror, report)
gpu_radiation_field_require_ready(owner, expected_generation, mirror, report)
gpu_radiation_field_reset(owner, required_generation, mirror, report)
gpu_radiation_field_free(mirror, report)
```

`require_ready`는 kernel enqueue와 같은 host call path의 바로 앞에 있고 그 사이에 CPU
commit/reset 호출이 들어갈 수 없다. check를 초기화 시 한 번만 하거나 debug build에서만
하는 것은 FAIL이다.

최소 실패 코드는 서로 구분되어야 한다.

| 상태 | 의미 |
|---|---|
| `GPU_RF_DISABLED` | CPU owner/view disabled |
| `GPU_RF_STALE_CPU` | CPU required/computed 불일치 |
| `GPU_RF_STALE_LINE` | line cache generation 불일치 |
| `GPU_RF_CPU_GPU_GENERATION_MISMATCH` | CPU와 READY GPU generation 불일치 |
| `GPU_RF_SHAPE_OR_HASH_MISMATCH` | edge/shape/unit/frame/epoch 불일치 |
| `GPU_RF_LINE_ID_MISMATCH` | line ID 순서·중복·mapping 불일치 |
| `GPU_RF_PROFILE_OR_QSET_MISMATCH` | profile/Q-set identity 불일치 |
| `GPU_RF_INVALID_CELL` | 필수 cell validity가 소비 불가 |
| `GPU_RF_PARTIAL_UPLOAD` | 일부 component만 copy/attest 완료 |
| `GPU_RF_CUDA_FAILURE` | allocation/copy/event/sync 실패 |
| `GPU_RF_NOT_READY` | EMPTY/DIRTY/UPLOADING/FAILED descriptor |

오류 함수는 값을 반환하지 않고 top-level nonzero rc로 전파한다. caller가 오류를 무시한
횟수도 0이어야 한다.

## 6. 카운터, byte 원장과 보고

### 6.1 필수 카운터

각 run과 누적 summary에 다음을 남긴다.

```text
sync_attempts, sync_commits, sync_failed_attempts, reset_count, free_count
ready_checks, ready_passes, ready_failures
launch_attempts, blocked_launches, physical_launches
stale_cpu_failures, stale_line_failures, cpu_gpu_generation_failures
shape_hash_failures, line_id_failures, profile_qset_failures
invalid_field_cells, invalid_line_cells
allocation_failures, copy_failures, event_failures, partial_upload_failures
fallback_attempts, zero_substitution_attempts
```

카운터는 다음 보존식을 매 run과 누적 summary에서 모두 만족해야 한다.

```text
sync_attempts = sync_commits + sync_failed_attempts
sync_failed_attempts = sum(sync_root_cause[status])
ready_checks = ready_passes + ready_failures
launch_attempts = physical_launches + blocked_launches
```

`sync_root_cause[status]`는 한 sync attempt당 정확히 하나만 증가하는 배타적 histogram이다.
우선순위는 검사 순서 그대로 (1) `GPU_CPU_CHANGED_DURING_UPLOAD`,
(2) `GPU_RF_STALE_CPU`, (3) `GPU_RF_STALE_LINE`,
(4) `GPU_RF_CPU_GPU_GENERATION_MISMATCH`, (5) `GPU_RF_SHAPE_OR_HASH_MISMATCH`,
(6) `GPU_RF_LINE_ID_MISMATCH`, (7) `GPU_RF_PROFILE_OR_QSET_MISMATCH`,
(8) `GPU_RF_INVALID_CELL`, (9) allocation 실패,
(10) 하나 이상의 component copy가 성공한 후 나머지 copy/attestation이 실패한
`GPU_RF_PARTIAL_UPLOAD`, (11) 성공 component가 0개인 첫 copy 실패,
(12) 모든 copy 성공 후 event/sync 실패, (13) `GPU_RF_NOT_READY` 순이다.
한 attempt에서 둘 이상이 동시 관측되면 위 순서의 첫 항목만 root cause다.
후순위 사실은 JSON의
`secondary_diagnostics[]`에만 기록하고 카운터를 추가 증가시키지 않는다.
따라서 N5는 `partial_upload_failures=1`, `copy_failures=0`,
`sync_failed_attempts=1`로 귀속한다. 성공 component가 하나도 없는 첫 copy 실패만
`copy_failures=1`이다. 위 보존식과 root-cause histogram 합이 다르거나 한 attempt에
root cause가 둘 이상이면 `FAIL_COUNTER_NONCONSERVATION`이다.

정상 A2-12 positive lane에서 `fallback_attempts=0`, `zero_substitution_attempts=0`,
`partial_upload_failures=0`, `blocked_launches=0`이다. 음성대조에서는 해당 failure counter가
정확히 증가하고 `physical_launches=0`이어야 한다.

### 6.2 upload byte 식

매 sync attempt마다 component별 **요청 bytes와 성공 bytes**를 둘 다 기록한다.

```text
field_edge_bytes       = (n_bins + 1) * sizeof(double)
field_value_bytes      = n_shells * n_bins * sizeof(double)
field_validity_bytes   = n_shells * n_bins * sizeof(device_validity_type)
line_id_bytes          = n_lines * sizeof(uint64_t)
line_value_bytes       = n_lines * n_shells * sizeof(double)
line_validity_bytes    = n_lines * n_shells * sizeof(device_line_validity_type)
line_count_bytes       = n_lines * n_shells * sizeof(uint64_t)
line_se_bytes          = n_lines * n_shells * sizeof(double)
metadata_bytes         = 실제 전송한 fixed-width metadata의 합
cache_upload_bytes     = line_id + line_value + line_validity + line_count
                         + line_se + line-cache metadata
total_upload_bytes     = field components + cache_upload_bytes
```

`sizeof` 값, `n_shells/n_bins/n_lines`, requested/succeeded, H2D direction, stream,
generation, upload serial을 JSON에 기록한다. profiler 추정치만 쓰거나 `cudaMemcpy` 호출 수만
보고하는 것은 불충분하다. `cache_upload_bytes`는 amendment가 요구한 독립 필드다.

추가로 committed bytes, discarded candidate bytes, peak live device bytes, elapsed upload
event time을 기록한다. bandwidth는 `successful_H2D_bytes/event_elapsed_seconds`로 계산하되
A2-12 합격선으로 쓰지 않는다.

## 7. A2-01 원장 19행의 현행 1:1 처분

고정 ID는 `docs/A2_01_DISPOSITION_LEDGER.md:60-70,89-96`의 원장 행이다. 현행 위치는
저작 HEAD에서 다시 측정했다. 구현 보고서에는 아래 19개 ID를 누락·병합·중복 없이 각각
종결해야 한다.

### 7.1 `GPU_lifecycle` 8행

| ID | 원장 고정 witness | 현행 witness | A2-12 처분 |
|---|---|---|---|
| `GL01` | `lumina_bf_gemm.cu:140 d_T_rad malloc` | `src/lumina_bf_gemm.cu:140` | scalar owner 제거. A2-14 전 opacity launch는 `GPU_OPACITY_NOT_MIGRATED`; canonical mirror allocation으로 대체 |
| `GL02` | `lumina_bf_gemm.cu:141 d_W malloc` | `src/lumina_bf_gemm.cu:141` | GL01과 같은 단일 mirror lifecycle; 별도 W owner 금지 |
| `GL03` | `lumina_bf_gemm.cu:390 d_T_rad free` | `src/lumina_bf_gemm.cu:390` | scalar free 제거; 공통 mirror transactional free로 종결 |
| `GL04` | `lumina_bf_gemm.cu:391 d_W free` | `src/lumina_bf_gemm.cu:391` | GL03과 같음 |
| `GL05` | `lumina_cuda.cu:273 d_T_rad malloc` | `src/lumina_cuda.cu:273` | transport scalar owner 제거; 공통 mirror candidate allocation만 허용 |
| `GL06` | `lumina_cuda.cu:341 d_T_rad test` | `src/lumina_cuda.cu:341` | lazy scalar allocation 분기 제거; mirror state/shape check로 대체 |
| `GL07` | `lumina_cuda.cu:342 d_T_rad malloc` | `src/lumina_cuda.cu:342` | partial/lazy allocation 금지; 전 component transaction으로 대체 |
| `GL08` | `lumina_cuda.cu:3286 d_T_rad free` | `src/lumina_cuda.cu:3286` | 공통 mirror free와 READY stamp 선무효화로 종결 |

### 7.2 `GPU_transport` 11행

이 표는 lifecycle 소관과 후속 물리 소관을 분리한다. `DEFER`는 현행 경로를 계속 실행해도
된다는 뜻이 아니다. A2-12에서 readiness/blocking을 설치한 뒤 후속 단계까지 실행 불가다.

| ID | 원장 고정 witness | 현행 witness | A2-12 처분과 최종 소유 단계 |
|---|---|---|---|
| `GT01` | `lumina_cuda.cu:3760 d_T_rad[shell]` | `src/lumina_cuda.cu:3760` | Planck sampling 수식은 무수정, launch 차단; **A2-15 DEFER** |
| `GT02` | `lumina_cuda.cu:3793 d_T_rad[shell]` | `src/lumina_cuda.cu:3793` | band Planck sampling 수식은 무수정, launch 차단; **A2-15 DEFER** |
| `GT03` | `lumina_cuda.cu:5978 d_T_rad argument` | `src/lumina_cuda.cu:5978` | raw scalar pointer plumbing은 qualified descriptor가 아님을 static gate로 차단; 물리 인자 제거는 **A2-15 DEFER** |
| `GT04` | `lumina_cuda.cu:6242 d_T_rad call` | `src/lumina_cuda.cu:6242` | launch 전 lifecycle readiness 검사 설치; interaction 식은 **A2-15 DEFER** |
| `GT05` | `lumina_cuda.cu:6552 d_T_rad call` | `src/lumina_cuda.cu:6552` | legacy BF re-emission launch 차단; **A2-15 DEFER** |
| `GT06` | `lumina_cuda.cu:8842 dev.d_T_rad launch` | `src/lumina_cuda.cu:8842` | 이 launch site에 공통 readiness gate 필수; scalar launch는 **A2-15 DEFER** |
| `GT07` | `lumina_cuda.cu:10256 dev.d_T_rad launch` | `src/lumina_cuda.cu:10256` | final launch에도 같은 gate 필수; scalar launch는 **A2-15 DEFER** |
| `GT08` | `lumina_cuda.cu:8557 plasma.W[s]` | `src/lumina_cuda.cu:8557` | host packet-source tier는 lifecycle upload가 아님; 무수정·GPU qualified lane 차단, **A2-15 DEFER** |
| `GT09` | `lumina_cuda.cu:8558 plasma.W` | `src/lumina_cuda.cu:8558` | GT08의 pointer validity fallback을 lifecycle로 세탁 금지; **A2-15 DEFER** |
| `GT10` | `lumina_cuda.cu:10814 plasma.T_rad[i]` | `src/lumina_cuda.cu:10814` 첫 occurrence | output-only ratio 진단. A2-12 diff 0, 허용 잔류로 계수; CPU/GPU mirror 입력 아님 |
| `GT11` | 같은 줄 두 번째 `plasma.T_rad[i]` | `src/lumina_cuda.cu:10814` 두 번째 occurrence | GT10과 별개 고정 ID로 동일 처분 |

`GT01`~`GT09`를 A2-12가 새 수식으로 고치면 단계 침범이다. 반대로 readiness gate 없이
현행 scalar 값으로 실행시키면 A2-12 FAIL이다. `GT10`과 `GT11`은 같은 물리 표현의 두
occurrence지만 원장 ID를 합치지 않는다.

## 8. 원장 밖 `.cu` 소비자 전수검사와 처분

### 8.1 build-authoritative 다섯 파일

`Makefile:18-46`이 가리키는 tracked `.cu` 다섯 파일을 모두 검사한다.

| 파일 | 저작 시 관련 grep 행 수 | 원장 밖 핵심 군 | 처분 |
|---|---:|---|---|
| `src/lumina_cuda.cu` | 86 | scalar 선언/upload/reset/free, legacy J/Jbar/Jblue accumulator, 두 transport launch | 아래 8.2; lifecycle 이관 또는 A2-15 차단 |
| `src/lumina_bf_gemm.cu` | 13 | scalar upload와 BF population kernel | lifecycle scalar owner 제거, 물리식 A2-14 차단 |
| `src/lumina_nlte_assemble.cu` | 14 | legacy `d_J_nu/d_W` allocation/upload/kernel | canonical mirror로 가장 금지; A2-13 전 launch 차단 |
| `src/lumina_cmf_solve.cu` | 10 | `d_J/d_Jnew` 할당·H2D·kernel·D2H·free | 8.2의 CMF J lifecycle로 명시 처분; GPU 실패의 CPU fallback 금지 |
| `src/lumina_nlte_gemm.cu` | 7 | `d_J_nu` 할당·H2D·rate GEMM·free | raw `nlte->J_nu` upload는 qualified mirror가 아님; A2-13 전 rate GEMM 차단 |

행 수는 14절의 고정 regex에 일치한 **행 수**이며 symbol occurrence 수가 아니다.
대소문자를 정규화하지 않고 `d_j_nu_estimator`와 `d_J_nu`, `d_jbar_line`과
`d_jblue_line`, `d_J/d_Jnew`를 서로 다른 심볼로 검사한 결과다. 다섯 파일의
불변식은 `86 + 13 + 14 + 10 + 7 = 130`이다.

### 8.2 목록 밖 발견 군

| 군 | 현행 위치 | A2-12 처분 |
|---|---|---|
| transport scalar 선언 | `src/lumina_cuda.cu:138-140` | old scalar owner 제거; canonical mirror descriptor는 별도 이름·타입 사용 |
| transport scalar upload helper | `src/lumina_cuda.cu:511-542` | `plasma->T_rad` raw upload를 정본 sync로 인정 금지; A2-15 전 해당 branch 차단 |
| upload 호출 3군 | `src/lumina_cuda.cu:7837-7840,8673-8682,10694-10700` | 모든 iteration에서 CPU generation 기반 sync/readiness 순서 강제; scalar helper는 qualified path 아님 |
| legacy global J estimator lifecycle | `src/lumina_cuda.cu:126-132,303-328,2180-2200,3278-3283` | producer accumulator와 read-only canonical mirror를 별도 타입·이름·counter로 분리 |
| legacy line accumulator lifecycle | `src/lumina_cuda.cu:2977-2990,7994-8001,8793-8796,9206-9211,10165-10195,10444-10449` | `LineJbarCache` mirror로 재명명/재사용 금지. raw producer라면 producer로만 남기고 CPU dual commit 뒤 새 mirror를 별도 upload |
| legacy blue-wing accumulator lifecycle | `src/lumina_cuda.cu:133,310,3284,8006-8008,8798-8800,8993,9300-9303` | `d_jblue_line`은 선언·NULL 할당·device 할당·reset·D2H·free를 갖는 별도 producer다. canonical line mirror로 재사용 금지; CPU dual commit 전 rate 소비 0 |
| Planck event propagation/calls | `src/lumina_cuda.cu:3753-3796,5347,5435-5472,5723-5740` | 수식 변경 금지; A2-15 전 physical launch 0 |
| BF-GEMM scalar reads/uploads | `src/lumina_bf_gemm.cu:44-45,60-100,208-230,295-308` | lifecycle owner 제거 후 A2-14 전 physical launch 0; kernel arithmetic 무수정 |
| NLTE assembly legacy field | `src/lumina_nlte_assemble.cu:88-94,120-128,140-171,353-357,394-429,462-484` | A2-13 전 physical launch 0. 특히 OOG `1e-30`과 dilute Planck fallback을 A2-12에서 고쳐 새 rate 식을 만들지 않음 |
| NLTE rate-GEMM field lifecycle | `src/lumina_nlte_gemm.cu:52,265-266,325,352,414-425,439,475` | `d_J_nu`는 `nlte->J_nu`를 FP32 staging해 rate GEMM이 소비하는 legacy field다. A2-13 이관 전 H2D/GEMM launch 0; A2-12 mirror로 가장 금지 |
| CMF J GPU-transfer lifecycle | `src/lumina_cmf_solve.cu:253,262-263,286,314-319,330,333` | `d_J/d_Jnew`의 allocation·H2D·kernel·D2H·free를 lifecycle census에 이관. `src/lumina_cmf_solve.cu:247-250`의 OOM `return -1` 및 `src/lumina_cmfgen.c:3577-3588`의 CPU 대체 경로는 **`BLOCKED_GPU_FALLBACK_FORBIDDEN`**. GPU 실패는 nonzero로 전파하고 같은 attempt에서 CPU solver를 실행·게시하지 않으며 `fallback_attempts` 증가 및 `physical_launches=0`으로 종결 |

GPU producer accumulator의 reset generation과 CPU canonical mirror reset generation은 같은
개념이 아니다. 이름·counter·state를 분리하고, producer 결과는 CPU dual commit을 거치기
전에는 GPU rate 입력이 될 수 없다.

### 8.3 작업트리의 archival `.cu`

저작 시 `rg --files | rg '\.cu$'`는 build-authoritative 다섯 파일 외에 다음 다섯 파일도
찾았다.

- `backup_groupA_1422/lumina_cuda.cu` — 확장 regex 86행
- `backup_groupA_1422/lumina_nlte_assemble.cu` — 확장 regex 14행
- `impl_withParityAA/orig/lumina_cuda.cu` — 확장 regex 86행
- `impl_withParityW/orig/lumina_cuda.cu` — 확장 regex 86행
- `impl_withParityY/orig/lumina_cuda.cu` — 확장 regex 86행

이들은 `git ls-files '*.cu'`와 `Makefile:18-46`에 없으므로 구현·판정 입력에서 제외한다.
복사본을 고쳐 PASS를 만들거나 이들 결과를 build-authoritative 소비자 수에 합치지 않는다.
단, 전수 grep에서 발견했다는 사실과 제외 근거는 구현 보고서에 보존한다.

## 9. 게이트 사전등록

### 9.1 static gate

구현 전에 checker manifest에 다음을 동결한다.

1. 원장 고정 ID `GL01`~`GL08`, `GT01`~`GT11` 정확히 19개, disposition 누락/중복 0.
2. tracked `.cu` 다섯 파일 전수 목록과 8.2 발견 군.
3. GPU mirror owner/setter가 정확히 하나이며 CPU 정본 외 upload source 0.
4. raw `plasma.W`, `plasma.T_rad`, `nlte->J_nu`, legacy `d_jbar_line/d_jblue_line`,
   `d_J_nu`, `d_J/d_Jnew`에서 canonical mirror로 복사하는 경로 0.
5. 모든 A2-13~15 launch site가 공통 readiness gate에 지배됨. call graph 우회 0.
6. coarse/fine Jbar 재적분, cache-miss fallback, zero/floor 대입 0.
7. kernel rate/opacity/emissivity 산식 diff 0. 허용 diff는 descriptor plumbing, allocation,
   upload, reset, readiness, counters, tests뿐.

### 9.2 positive lifecycle fixture

작은 deterministic fixture와 실제 amended shape fixture를 둘 다 쓴다.

- CPU dual commit `g=1` -> sync -> READY -> prelaunch check PASS.
- CPU `required=2, computed=1` -> 즉시 DIRTY, prelaunch FAIL, physical launch 0.
- CPU dual commit `g=2` -> reset/sync -> READY; generation 1 pointer를 재사용하지 않음.
- field/cache 값, validity, ID mapping, hashes를 D2H attestation으로 byte/element 비교.
- exact byte 식과 measured requested/succeeded bytes 일치.
- free를 두 번 호출해도 leak/double-free가 없고 READY가 남지 않음.
- non-default CUDA stream에서도 event 전 READY 공개 0.

### 9.3 음성대조

각 poison은 독립 child process다. 결함 검출이 정상인 경우 child는 아래 **기대 nonzero
rc**와 marker를 내고, 상위 verifier는 이를 확인한 뒤 rc=0으로 끝난다. 모든 poison에서
`physical_launches=0`을 함께 검증한다.

| poison | 주입 | 기대 marker | child rc |
|---|---|---|---:|
| `N1` | CPU field required만 `g+1` | `A2_12_NEG_CPU_STALE_FAIL` | 41 |
| `N2` | amendment §5.4-5: cache generation을 field보다 하나 늦춤 | `A2_12_NEG_CACHE_GENERATION_FAIL` | 42 |
| `N3` | amendment §5.4-5: upload candidate의 line ID mapping 두 항 shuffle | `A2_12_NEG_LINE_ID_MAPPING_FAIL` | 43 |
| `N4` | CPU g와 GPU committed g를 다르게 stamp | `A2_12_NEG_CPU_GPU_GENERATION_FAIL` | 44 |
| `N5` | line values 복사 뒤 validity 복사 실패 주입 | `A2_12_NEG_PARTIAL_UPLOAD_FAIL` | 45 |
| `N6` | active field 또는 line validity를 `UNSAMPLED/STALE`로 poison | `A2_12_NEG_INVALID_VALIDITY_FAIL` | 46 |
| `N7` | cache miss에 0/coarse-J fallback hook을 요청 | `A2_12_NEG_FALLBACK_FAIL` | 47 |
| `N8` | reported cache bytes를 한 component만큼 축소 | `A2_12_NEG_UPLOAD_BYTES_FAIL` | 48 |
| `N9` | reset generation을 CPU보다 ±1 이동 | `A2_12_NEG_RESET_GENERATION_FAIL` | 49 |

특히 N2와 N3는 **GPU rate kernel enqueue 전에** 실패해야 한다. kernel 내부 assert,
실행 뒤 수치 mismatch, CUDA illegal access로 검출하면 FAIL이다. poison마다 독립 marker와 rc가
없거나 여러 poison을 한 generic FAIL로 합치면 FAIL이다.

### 9.4 합격 판정

A2-12 PASS는 다음의 논리곱이다.

```text
static_census_PASS
AND lifecycle_positive_PASS
AND generation_prelaunch_PASS
AND line_mapping_profile_qset_PASS
AND validity_no_fallback_PASS
AND upload_byte_ledger_PASS
AND all_9_negative_controls_PASS
AND cpu_regression_invariant_PASS
AND gpu_node_evidence_PASS
```

GPU node evidence가 없으면 PASS가 아니라 11.3의 `BLOCKED_GPU_UNAVAILABLE`이다.

## 10. 회귀 전판과 CPU 경로 불변 증명

### 10.1 battery census preflight

비싼 build/GPU 실행 전에 반드시 다음을 독립 실행한다.

```bash
python3 scripts/a2_01_census_contract.py check
python3 scripts/run_gate_battery.py --verify-equivalence
```

`scripts/run_gate_battery.py:25-40,369-373`의 A2-01 census preflight가 먼저 PASS해야 한다.
저작 시 battery 정본은 `scripts/run_gate_battery.py:20-22`의 D19/K7/Z6/CP4, 총 36행이다.
구현 시 동시 작업으로 row 수가 바뀌었으면 source의 그 시점 수를 재측정해 보고하며 36으로
억지 고정하지 않는다. serial/parallel result table은 동일해야 한다.

### 10.2 현재 존재하는 selftest 전종

회귀 범위는 A2 이름만 붙은 target으로 제한하지 않는다. 구현 직전 아래 두 inventory를
artifact로 동결하고, 그때 존재하는 **모든 Makefile `selftest*` target과 모든 독립
selftest runner**를 실행한다.

```bash
awk -F: '/^selftest[^[:space:]]*:/ {print $1}' Makefile
rg --files scripts tests \
  | rg '(^|/)(selftest[^/]*|.*_selftest\.(py|sh|c))$' \
  | sort
```

Make target 안에서 fixture 실행까지 하는 recipe와 binary만 만드는 recipe를 구분한다.
후자는 recipe가 만든 실제 binary를 추가 실행한다. C selftest source는 대응 Make target이나
동결된 build command로 build/run하고, Python/Bash runner는 source에 적힌 실제 CLI로
실행한다. 외부 입력 또는 node 자격이 필요한 항목은 조용히 skip하지 않고 입력 hash와
`BLOCKED_MISSING_FIXTURE` 또는 `BLOCKED_NODE_INELIGIBLE`를 남긴다. A2-12 자체의 GPU
selftest만 11절 SLURM으로 보낸다. 구현 중 동시 작업으로 inventory가 늘면 새 항목도 포함하며,
이 명세 저작 시 개수를 합격선으로 고정하지 않는다.

그 전종 안에서 최소 다음 A2 target/script는 반드시 포함한다.

구현으로 신규 TU가 생기면 "모든 selftest"라는 문구만으로 충족하지 못한다.
`scripts/run_gate_battery.py`의 Z 독립 링크 네 곳(`:139`, `:149`, `:157`, `:167`;
`Z-validator`, `Z-tau`, `Z-population`, `Z-canonical`)에 그 TU를 모두 추가하고,
Z runner 전달부(`scripts/run_gate_battery.py:248`, 구현 HEAD에서 `Run("Z", ...)`로
재측정)에 신규 binary/fixture 인자를 전달해야 한다. 동일 인자를
`scripts/run_zinert_selftest.py:59`의 required CLI와 `:82`의 실행 `definitions`에
동시 추가하고, 파일 존재성 검사·rc·stdout 판정에도 포함한다. 네 링크 중
하나라도 누락하거나 runner/CLI/definition 중 하나라도 누락하면
`FAIL_Z_TU_WIRING_INCOMPLETE`이다.

다음 target/script를 build 후 각각 실행하고 명령, rc, stdout artifact hash를 남긴다.

```text
selftest_a2_03_radiation_field
selftest_a2_03_producer_parity_fixture
selftest_a2_04_commit
selftest_a2_04_replay_commit
selftest_a2_05_bf_rate
selftest_a2_06_line_jbar
selftest_a2_06_dual_commit
selftest_a2_07_population
scripts/a2_03_byte_parity.py
scripts/a2_03_callgraph_audit.py
scripts/a2_04_commit_callgraph.py
scripts/a2_04_l0_replay.py
scripts/a2_05_l1bf_gate.py
scripts/a2_06_l1bb_gate.py
scripts/a2_07_population_census.py
scripts/a2_07_population_gate.py
scripts/a2_07_classic_sweep.py
```

각 Python script의 실제 CLI는 실행 직전 `--help`와 source로 확정한다. 존재하지 않는 옵션을
추측해 호출하지 않는다. upstream의 기존 `BLOCKED`는 그대로 보존하며 A2-12가 PASS로
세탁하지 않는다.

### 10.3 CUDA selftest 전종

`Makefile:44-58`의 `lumina_cuda`, `selftest_nlte_assemble`과 구현 시 추가되는
`selftest_a2_12_gpu_lifecycle`을 같은 source hash로 build한다. 기존 CUDA test/bench는
입력 덱이 필요 없는 selftest는 전부 실행하고, bench는 smoke가 아니라 diagnostic으로
표시한다. A2-12 negative fixture는 반드시 실제 CUDA allocation/copy/event를 사용한다.

### 10.4 CPU 불변 증명

A2-12 changed-output allowlist는 lifecycle status/counter/JSON/log와 GPU mirror memory뿐이다.
CPU `Gamma`, `Jbar`, population, `chi`, `eta`, spectrum, packet RNG/output은 allowlist가 아니다.

1. 구현 전후 CPU build의 source manifest와 compiler/env를 고정한다.
2. A2-03 producer parity fixture, A2-04 replay, A2-05/06 deterministic fixture,
   A2-07 population fixture의 stdout과 binary artifact를 SHA-256/byte compare한다.
3. 36행(또는 구현 시 재측정된 정본 행 수) battery serial/parallel result table을 전후
   byte compare한다.
4. A2-12 관련 환경변수를 unset한 CPU production lane의 정본 output을 byte compare한다.
5. 차이가 하나라도 있으면 허용 오차를 발명하지 않고 `FAIL_CPU_PATH_CHANGED`다.

CPU `.c` 물리 파일을 바꿔 lifecycle을 구현하지 않는다. CUDA linkage 때문에 헤더 선언을
추가해야 하면 CPU object code diff가 없음을 disassembly 또는 binary section hash로 함께
증명한다.

## 11. SLURM GPU 운전 계약

운전석만 제출한다. 구현자는 제출 가능한 driver와 정확한 resource request를 제공하고 직접
제출했다고 가장하지 않는다. 모든 driver는 `set -euo pipefail`, source/binary/input hash,
`nvidia-smi` 이름·UUID·driver·memory, `SLURM_JOB_ID`, rc, marker, artifact hash를 남긴다.
`/usr/bin/time`은 쓰지 않는다.

### 11.1 lifecycle/selftest job

제출 전에 `validation/a2_12`가 존재해야 한다. 운전석 명령은 다음과 같다.

```bash
mkdir -p validation/a2_12
sbatch --parsable --job-name=a2-12-life \
  --partition=h200,h100 --nodes=1 --ntasks=1 --cpus-per-task=8 \
  --mem=32G --gres=gpu:1 --time=01:00:00 \
  --output=validation/a2_12/a2_12_lifecycle-%j.out \
  --error=validation/a2_12/a2_12_lifecycle-%j.err \
  scripts/run_a2_12_gpu_lifecycle.slurm
```

파티션 선호는 h200 다음 h100이다. job 안에서 실제 GPU memory와 compute capability를
기록한다. 이 job은 small fixture, amended-shape upload, N1~N9를 실행한다.

### 11.2 full-NLTE integration이 필요한 경우

full-NLTE는 GPU memory 80 GB가 필요하므로 a40을 제외하고 h200/h100만 요청한다.

```bash
mkdir -p validation/a2_12
sbatch --parsable --job-name=a2-12-full \
  --partition=h200,h100 --nodes=1 --ntasks=1 --cpus-per-task=16 \
  --mem=64G --gres=gpu:1 --time=06:00:00 \
  --output=validation/a2_12/a2_12_full_nlte-%j.out \
  --error=validation/a2_12/a2_12_full_nlte-%j.err \
  scripts/run_a2_12_full_nlte.slurm
```

driver는 시작 즉시 `nvidia-smi --query-gpu=name,memory.total,uuid --format=csv,noheader`를
검사하고 total memory가 80000 MiB 미만이거나 A40이면 물리 실행 없이
`A2_12_GPU_MEMORY_INELIGIBLE`과 nonzero rc로 끝낸다. full job은 CPU dual commit과 GPU
sync/readiness까지만 lifecycle 판정 대상으로 삼는다. A2-13~15 kernel 수치 PASS를 요구하거나
현행 scalar kernel을 실행하지 않는다.

### 11.3 GPU를 못 잡았을 때

h200와 h100 모두에서 allocation을 받지 못했거나 scheduler가 job을
`CANCELLED`, `TIMEOUT`, `NODE_FAIL`, `OUT_OF_MEMORY`로 끝낸 경우 다음을 남긴다.

```text
stage_status=BLOCKED_GPU_UNAVAILABLE
marker=A2_12_BLOCKED_GPU_UNAVAILABLE
submitted_job_ids=[...]
partitions_tried=[h200,h100]
scheduler_states=[...]
sinfo_snapshot_sha256=...
squeue_or_sacct_snapshot_sha256=...
last_runner_rc=75
```

이는 PASS도 lifecycle FAIL도 아니다. CPU/static/compile 결과는 보존하되 단계 최종 상태는
`BLOCKED_GPU_UNAVAILABLE`이다. 로그인 노드에서 GPU가 있는 척 mock PASS하거나 a40 full
run으로 대체하지 않는다. 단순 PENDING은 곧바로 BLOCKED로 판정하지 않고 운전석이 job의
종결 상태를 확인한다.

## 12. 구현 순서와 변경 제약

순서를 바꾸지 않는다.

1. HEAD/source hash와 tracked `.cu` 다섯 파일, 원장 19 ID, 8.2 발견 군을 manifest로 동결.
2. CPU/battery baseline과 selftest 전종 artifact를 먼저 확보.
3. mirror 상태·status·counter·checked-size/byte ledger를 구현. 물리 kernel diff는 아직 0.
4. candidate allocation/reset/free와 failure injection을 구현.
5. 두 checked CPU view에서 transactional upload, event sync, atomic READY publish를 구현.
6. 모든 미래/현행 GPU physics launch 앞에 공통 readiness gate를 설치하고 미이관 단계
   상태코드로 차단.
7. static gate와 small positive/N1~N9를 통과.
8. SLURM h200→h100 lifecycle job, 필요시 full-NLTE sync job을 운전석이 제출.
9. CPU 회귀 전판과 byte-invariance를 재실행.
10. 회귀 대장에 정확히 한 A2-12 행과 구현 보고서를 남김.

구현 diff의 허용면은 CUDA lifecycle 코드/헤더, build wiring, A2-12 전용 fixture/driver,
`validation/a2_12/`, A2-12 구현 보고서와 원장 addendum뿐이다. 다음은 금지한다.

- CMFGEN 원본, deck, `/gpfs` 입력 수정.
- A2-13~15 kernel formula 변경.
- CPU 물리식이나 CPU 결과 변경.
- 독립 GPU radiation-field owner/setter/generation 생성.
- partial upload를 성능 최적화로 허용.
- stale/invalid를 0, floor, Planck, previous generation, CPU fallback으로 대체.
- GPU 미확보를 PASS 또는 NOT_RUN으로 축약.
- commit/push 및 `/usr/bin/time` 사용.

구현 중 새 `.cu` 소비자가 나오면 멈추지 말고 frozen manifest의
`DISCOVERED_OUTSIDE_CENSUS`에 추가하고 현행 줄번호·call graph·처분을 기록한다. 발견 0도
명령과 0 결과를 남긴다.

## 13. 산출물과 회귀 대장

구현 단계의 최소 산출물은 다음이다.

```text
validation/a2_12/source_manifest.json
validation/a2_12/gpu_lifecycle_report.json
validation/a2_12/upload_bytes.jsonl
validation/a2_12/negative_controls.json
validation/a2_12/cuda_consumer_census.json
validation/a2_12/cpu_invariance.json
validation/a2_12/A2_12_REGRESSION_LEDGER.jsonl
docs/CODEX_IMPL_A2_12.md
```

대장에는 `docs/ORDER_L0_JNU_OWNER_BY_CODEX.md:689-710`의 필드를 전부 넣고 다음도
포함한다.

```text
cpu_required_generation
cpu_computed_generation
line_required_generation
line_computed_generation
gpu_committed_generation
upload_serial
cache_upload_bytes
total_upload_bytes
peak_live_device_bytes
physical_launches
blocked_launches
fallback_attempts
negative_marker_rc_map
slurm_job_ids
gpu_name_uuid_memory
```

`new_layer_status=PASS`는 9.4 논리곱이 모두 참일 때만 쓴다. A2-13~15는 각각 별도 대장
행이며 A2-12 행에 rate/opacity/emissivity parity를 미리 PASS로 적지 않는다.

## 14. 저작 시 실측

저작 시각 기준 source HEAD는 `068fb36`다. 줄번호는 이 HEAD의 작업트리에서 `nl -ba`와
`rg -n`으로 재측정했다. 동시 편집 대상인 `docs/SPEC_A2_08_V2.md`는 읽기만 했고 수정하지
않았으며 `docs/SPEC_A2_09_10_V1.md`도 수정하지 않았다.

실측 명령은 다음과 같다.

```bash
git rev-parse HEAD
git ls-files '*.cu' | sort
rg --files | rg '\.cu$' | sort
rg -n '\bd_T_rad\b|\bd_W\b|\bd_jbar_line\b|\bd_jbar_count\b|\bd_jblue_line\b|\bd_j_nu_estimator\b|\bd_j_nu_count\b|\bd_J_nu\b|\bd_J\b|\bd_Jnew\b|RadiationField|LineJbar|line_jbar|radiation_field' src/lumina_cuda.cu src/lumina_bf_gemm.cu src/lumina_nlte_assemble.cu src/lumina_cmf_solve.cu src/lumina_nlte_gemm.cu
rg -n 'radiation_field_owner_init|radiation_field_commit|radiation_field_read_view|radiation_field_line_jbar_view' src/lumina_cuda.cu
nl -ba src/lumina_cuda.cu
nl -ba src/lumina_bf_gemm.cu
nl -ba src/lumina_nlte_assemble.cu
nl -ba src/lumina_cmf_solve.cu
nl -ba src/lumina_nlte_gemm.cu
```

결과 요약:

- tracked/build-authoritative `.cu`: **5개**.
- 작업트리 전체 존재 `.cu`: **10개**(tracked 5 + archival/untracked 5).
- 고정 원장 처분: `GPU_lifecycle` **8/8**, `GPU_transport` **11/11**, 합계 **19/19**.
- 대소문자·별도 심볼을 포함한 관련 regex 일치 행: tracked 기준
  `lumina_cuda.cu` **86**, `lumina_bf_gemm.cu` **13**,
  `lumina_nlte_assemble.cu` **14**, `lumina_cmf_solve.cu` **10**,
  `lumina_nlte_gemm.cu` **7**, 합계 **130**. 기존 regex 96행에서 **34 site 추가**다.
- CUDA main의 CPU canonical owner/commit/checked-view 호출: **0개**.
- 목록 밖 핵심 군: 8.2의 **11군**. 특히 legacy line/blue-wing accumulator와 canonical
  `LineJbarCache` mirror는 별개다.
- 신규 34행은 `d_jblue_line` 10행, 기존 NLTE assembly의 대문자
  `d_J_nu` 재포착 7행, NLTE rate-GEMM `d_J_nu` 7행, CMF `d_J/d_Jnew`
  10행이다. 이 중 새 처분군은 blue-wing, NLTE rate-GEMM, CMF J의 3군이다.
- CMF GPU OOM/실패를 CPU로 대체하는 경로의 처분은
  **`BLOCKED_GPU_FALLBACK_FORBIDDEN`**이다.
- `GPU_lifecycle` 8행과 `GPU_transport` 11행의 원장 줄번호는 저작 HEAD에서도 각각
  `140,141,390,391,273,341,342,3286` 및
  `3760,3793,5978,6242,6552,8842,10256,8557,8558,10814(2 occurrences)`로 유지됐다.

구현 직전 HEAD가 달라지면 이 절의 숫자를 복사하지 말고 같은 명령으로 전부 재실측한다.
