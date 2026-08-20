# 감사: DET 판정런 GPU 경로 병목 — 원인 지목·수리 설계·측정 계획 (2026-08-20)

- 감사자: Fable (분담 개정14 — 기획·판단·평가·검수). **read-only 감사 — 코드 0줄.**
- 대상: `src/cmf_exact_multigpu.cu` (branch `thenmc-macroatom-fluorescence`; 감사 시점
  HEAD `dd23d60`, 발주서 기준 커밋 `76f5982` 는 그 2커밋 전 — 두 커밋 모두 대장 기재라
  솔버 소스는 동일).
- 입력: 운전석 증거 패킷 `/tmp/claude-10396/gpuaudit/EVIDENCE.md`,
  `docs/CLASSIC_DEBT_CENSUS.md` 8절+정정, `docs/CMF_EXACT_MULTIGPU_PROTOTYPE_2026-08-10.md`,
  소스 전문(`cmf_exact_multigpu.cu` 3,475줄 통독), 호출부 `src/lumina_cmfgen.c`,
  `src/cmf_error_envelope.c`, `tests/cmf_exact_multigpu_selftest.cu`,
  `tests/cmf_exact_multigpu_reduced_bench.cu`.
- 표기 규율: 모든 주장에 `[실측]`(로그·소스·문서 직접 확인) / `[모델]`(코드에서 유도한
  산술 재구성) / `[추정]`(정합적 가설) / `[모름]`. **순간값·요약값은 잣대로 쓰지 않았다**
  (패킷 §0 의 두 함정 반복 금지).

---

## 0. 요지

1. 판정런의 GPU 시간 대부분은 fine-grid exact 솔버의 **sweep 반복 횟수**가 만든다.
   콜 하나가 sweep 을 `(고정점 반복수) + 3 + (seed 시도수 + 1 + 2×36)` 번 돈다 [실측 —
   §2.3]. 이 중 **오차-봉투(supersolution certificate) 기계가 콜당 ~73+ sweep** 을
   차지하고, 그 안에 **입력이 완전히 동일한 K·v 적용을 두 번 계산하는 중복이 반복당
   1회씩 실존**한다 [실측 — §5.1]. 이 중복 제거는 **바이트 불변**이면서 콜당 sweep 을
   ~37회 줄이는, 이 감사가 찾은 가장 크고 가장 안전한 수리안이다.
2. 두 장치의 불균형(0/100 꼬리 6초)은 **후보 기전이 둘**이다: (H1) 파티션 가중(세그먼트
   수)이 epoch-경로의 실제 비용과 다르다, (H2) 단일 호스트 스레드의 런치 공급이 큐
   배압으로 장치 간 결합된다. **소스에는 sweep 중 장치 간 동기 지점이 없음을 확정**했다
   [실측 — §3.1]. 둘 중 어느 것이 지배인지는 **아직 모른다** — §7 의 M2(스택×GPU상태
   타임스탬프 결합)가 무편집으로 판별한다.
3. 양쪽 0% 구간의 정체는 후보 4개(반복 사이 호스트 위상 / 콜 초입·말미의 이중
   할당·해제 / 봉투 호스트 검사 / 솔버 밖 caller 작업)로 좁혔고, **이미 코드에 박혀
   있는 `[EXACT-MULTIGPU-TIMING]` 위상 타이머 로그가 콜 단위 누적치로 전부 분해해
   준다** [실측 — §4]. 새 계측 없이 지난 런 stderr 수확(M0)이 최우선 측정이다.
4. 결정론 판정: 수리안별 개별 판정을 §6 표에 모았다. **스케줄 상수 변경은 계약·게이트
   양쪽으로 bit-neutral 이 보증**되고 [실측], **파티션 경계 이동은 장치-합 결합 순서가
   바뀌어 바이트가 변한다** [실측 — 프로토타입 문서의 2/3/4-GPU 상호차 1e-16 이 그
   증거]. 커널 내부 체인의 스캔화(병렬 prefix)는 합산 순서 변경이므로 DET 잣대
   아래에서는 금지 목록에 올린다.
5. **측정 없이 착수 금지**: 아래 수리안 중 어떤 것도 M0–M3 이전에 발주하지 않는다.
   유일한 예외 없음. M0 은 소스 0줄·런 0회(로그 판독)다.

---

## 1. 감사가 재구성한 실행 구조 [실측]

### 1.1 콜 토폴로지

```
NLTE 생산자 루프 (lumina_cmfgen.c:7382, pass = 0..n_iter)
 └─ cmfgen_fine_jbar(...)                       :7444, pass당 1회
     └─ cmf_fine_exact_owner_solve(...)          :5876 (본 솔브)
         └─ cmf_exact_multigpu_positive_solve_envelope_epoch_partitioned
             ├─ solve_impl(...)                  고정점: iteration_cap=64, tol=1e-8
             │    └─ iteration마다 launch_sweep  :1949
             ├─ PersistentBoundContext.initialize (두 번째 전체 할당+업로드)
             ├─ apply_bounds: rounding -1/0/+1   → sweep 3회
             ├─ seed 루프: 시도당 verify(K·v 1회) → sweep×시도수
             └─ cmf_error_envelope_refine(36회)  → 반복당 K·v 2회 = sweep 72회 + 초입 1회
     └─ (LUMINA_A210_INDEPENDENT_CAPTURE=1 이면) continuum 솔브 :6503 — 한 콜 더
```

- 판정런 스케줄은 **하드코딩** `{block=128, batch=64, replay_window=32}`
  (`lumina_cmfgen.c:4646`) [실측].
- sweep 1회 = 세그먼트 50개 × 2방향 = 장치당 100 세그먼트-패스 + partial_j 커널 +
  실패-플래그 회수 2회 (`launch_sweep`, `cmf_exact_multigpu.cu:1413-1533`) [실측].
- 세그먼트-패스 1회 = replay 커널 1런치 + epoch 배치 루프
  `ceil(max_epochs_seg/64)` 런치 (`:1361-1410`) [실측].

### 1.2 문제 규모 — 패킷 §4 의 "1234 bin" 은 fine 격자가 아니다 [실측/모델]

- fine 격자 `dlognu = (vdop/c)/ppd = (1e6/2.998e10)/12 = 2.780e-6`
  (`lumina_cmfgen.c:5066`) [실측].
- `NF = ceil(log(nu_hi/nu_lo)/dlognu)` (`:5127`). λ 100–20,000 Å 이면 ln200/2.78e-6
  ≈ **2.0×10⁶ 빈** [모델]. reduced bench 가 production 계약으로 못박은 상한이
  `2,013,113` (`tests/cmf_exact_multigpu_reduced_bench.cu:209 등`) [실측].
- 같은 bench 계약: 총 drift `47,649.x` bins, **max_window = 9,108 bins**,
  ray-segment 총수 **2,025** (66 rays = 16 core + 50 tangent) [실측].
- cells = 50 × ~2.01e6 ≈ **1.0×10⁸**. 장치 메모리 재구성 [모델]:
  `in/out` = (계산 세그먼트 ~1,061)×NF×8B ≈ 17.1 GB ×2 = 34.2 GB, `dt1/t1/source/
  source_cell/partial` 5×0.81 GB, epoch workspace(장치1) ≈ 2.9 GB → **합계 ≈ 39 GB**.
  **GPU1 실측 38,919 MiB 와 정합** [실측과 대조]. GPU0 실측 54,215 MiB 의 초과
  ~15 GB 는 이 솔버 소속이 아니다 — 앱의 다른 CUDA 모듈(NLTE GEMM 등)이 device 0 에
  상주하는 것으로 본다 [추정 — 확인 항목 §7 V3].
- ⟹ 패킷 §4 의 "1234 bin" 은 fine 솔버의 NF 가 아니라 다른 격자(코스 격자로 추정
  [추정])다. 운전석은 stderr 의 `[cmf_fine] ... NF=%d (%.1fM cells)` 라인
  (`lumina_cmfgen.c:5157`)으로 확정할 것.

### 1.3 sweep 횟수 산술 [실측 코드 + 모델]

콜당 upper-operator 적용 수는 selftest 가 공식으로 못박고 있다
(`tests/cmf_exact_multigpu_selftest.cu:286-293`):

```
persistent_upper_operator_applications = seed_attempts + 1 + 2×refinements
persistent_bound_applications          = 3 + 위 값
```

여기에 고정점 반복(`iterations_used` ≤ 64)이 더해진다. refinements=36 이므로
**콜당 sweep ≈ iterations_used + 3 + seeds + 73**. iterations_used·seeds 는 로그의
`[EXACT-MULTIGPU-EPOCH]` 라인에 이미 찍힌다 [실측] — 즉 **sweep당 벽시계는 지난
런 로그만으로 산출 가능**하다 (M0).

---

## 2. 질문 1 — 불균형의 원인 (코드 특정)

### 2.1 확정 사실 [실측]

1. **sweep 내부에 장치 간 동기 지점은 없다.** 파일 전체에
   `cudaDeviceSynchronize`/`cudaStreamSynchronize`/이벤트/peer access 가 0회.
   모든 커널은 장치별 default stream 에 비동기 enqueue 되고, 세그먼트 간 의존은
   같은 장치의 스트림 순서로만 이행된다 (`launch_sweep :1437-1467`).
   장치 간 결합은 오직 (a) **단일 호스트 스레드의 enqueue 대역**과 (b) sweep 말미의
   **차단형 회수 순서**(d0→d1, `:1470-1490`, `:1508-1530`) 뿐이다.
2. epoch 배치 경계는 장치 간 직렬화가 **아니다** — 배치 루프(`:1393-1408`)는 같은
   장치 스트림에 연속 enqueue 할 뿐 호스트 동기가 없다.
3. 파티션은 `CMF_MGPU_PARTITION_WEIGHTED_SEGMENTS`: **활성 (ray,segment) 쌍 수**의
   누적 분위수로 경계를 자른다 (`build_ray_partition :202-280`). 2,025 쌍을 2 장치에
   ~1,012/1,013 으로 나눈다 [모델 — bench 의 4장치 상수 490..539 와 같은 규칙].
   장치0 = 안쪽 연속 광선(코어 16 포함), 장치1 = 바깥 광선.
4. 커널 구조: `positive_epoch_large_segment_kernel` (`:1018`) 의 1단계는
   **블록당 스레드 0·1·2 셋만** window 길이의 순차 compose 체인을 돈다(나머지
   125 스레드는 `__syncthreads()` 대기, `:1063-1105`). 2단계는 전 스레드가 window
   출력을 분담(`:1124-1143`). window 는 33..9,108 [실측 bench 계약]. 즉 이 커널은
   의도적으로 **결합 순서를 보존하는 대신 SM 효율을 버린** 설계다 — 프로토타입
   문서도 "production performance ... not yet sealed" 로 명시 [실측
   `docs/CMF_EXACT_MULTIGPU_PROTOTYPE_2026-08-10.md`].

### 2.2 후보 기전 — 어느 쪽이 지배인지는 모른다 [추정]

- **H1 부하 불균형 (가중 잣대 불일치)**: 파티션이 균등화하는 것은 세그먼트 **수**다.
  epoch-경로의 실제 비용은 (a) window>32 인 쌍의 분포, (b) 쌍당 블록 수
  `ceil(NF/window)`, (c) 블록당 직렬 체인 길이(window), (d) 세그먼트-패스별 활성
  광선 수(점유율)로 정해진다. 이들은 전부 **기하+dlognu+NF 만의 결정론적 함수**라
  오프라인에서 정확히 계산 가능하다(M3). 장치1(바깥 광선)은 접선 인접 세그먼트의
  ds=√(2rΔr) 가 커서 긴 체인이 몰리고, 깊은 세그먼트-패스에서 활성 광선이 적어
  저점유가 된다 [추정 — M3 로 정량].
- **H2 공급 직렬화 (단일 스레드 enqueue + 큐 배압)**: 세그먼트당 런치 수가
  `1 + ceil(max_epochs_seg/64)` 인데, max_epochs_seg = NF/window 는 **window 33 인
  쌍이 있으면 61,004 epochs → 954 런치**다 [모델]. 드라이버의 미결 런치 큐는
  유한하므로(정확한 깊이는 [모름]) `cuLaunchKernel` 이 스핀 대기한다 — 스택 표본
  5/5 가 그 상태였다는 패킷 실측과 정합 [실측]. 호스트가 장치 d 의 큐에 막혀 있는
  동안 반대편 장치의 큐가 마르면 그 장치가 논다.
- **H3 (꼬리 회수 순서)**: sweep 말미 실패-플래그 회수가 d0→d1 순 차단형이지만,
  이는 이미 끝난 일의 대기라 불균형의 **표출**이지 원인이 아니다 [실측 코드].

### 2.3 관측과의 대조 [실측+추정]

관측된 주기(8초 양쪽 100 → 6초 GPU1 단독 → 양쪽 0)는 두 기전 모두와 정합한다:
- 꼬리 국면에서 호스트가 `cuMemcpyDtoH_v2`(d0 실패-플래그 회수에서 d1 완료 대기)면
  → enqueue 는 일찍 끝났고 꼬리는 **순수 부하 불균형(H1)**.
- 꼬리 국면에서 호스트가 여전히 `cuLaunchKernel` 스핀이면 → **공급 직렬화(H2)** 가
  GPU0 기아를 만들고 있는 것.
패킷의 스택 표본에는 두 상태가 모두 등장하나 **GPU 위상과 시각 정렬이 안 되어
판별 불가** [실측 한계]. → M2 가 이 판별을 무편집으로 수행한다.

**판정: 불균형의 원인은 H1/H2 복합 후보로 좁혔고 코드 좌표는 특정했으나, 지배
기전의 확정은 측정(M2·M3) 몫이다. "그럴듯함"으로 확정하지 않는다.**

---

## 3. 질문 2 — 양쪽 0% 구간의 정체

코드상 GPU 가 양쪽 다 놀 수 있는 구간은 다음 넷뿐이다 [실측 코드]:

| 후보 | 코드 좌표 | 내용 | 규모 추산 [모델] | 실측 잣대 |
|---|---|---|---|---|
| (a) 반복 사이 호스트 위상 | solve_impl `:1908-2016`, apply_round `:3040-3117` | source 조립(1e8 셀)·HtoD 1.6 GB×2·partial DtoH 0.8 GB×2(pageable)·host 환원·수렴검사 | 반복당 ~2-4 s | `source_assembly_s`+`h2d_s`+`d2h_s`+`host_reduction_s`+`convergence_check_s` (TIMING 로그) |
| (b) 콜 초입·말미 | solve_impl 할당 `:1813-1902` + `PersistentBoundContext.initialize` `:2746-3020` + cleanup | **같은 ~39 GB 를 콜마다 두 번 할당·두 번 업로드·두 번 해제** (solve → persistent) + dt1/t1 exp() 1e8 셀 ×2 | 콜당 수-수십 s | `initialization_s`+`envelope_context_setup_s`+`cleanup_s` |
| (c) 봉투 호스트 검사 | `cmf_error_envelope.c:76-215`, residual `:3289-3310` | K·v 적용 사이의 1e8 셀 비교·margin 루프 | 적용당 ~0.5-1 s ×~75회 | `envelope_residual_s`+`envelope_verify_s`+`envelope_refine_s` 에서 sweep 시간 차감 |
| (d) 솔버 밖 caller | `lumina_cmfgen.c` 라인 deposit·per-line Jbar 추출(:6328 일대)·코스 솔브·NLTE | 솔브 콜 사이의 상위 루프 작업 | [모름] | `caller_total_s − reported_total_s` + stdout 표지 간격 |

`cuMemcpyDtoH_v2` 표본은 (a)의 partial 회수 또는 sweep 말미 실패-플래그 회수(그
경우 반대편 GPU 는 100%)와 정합 [실측+추정]. **네 후보의 상대 지분은 전부
`[EXACT-MULTIGPU-TIMING]` 로그가 콜 단위 누적으로 이미 찍고 있다** — 판정은 M0.

---

## 4. 커널·구조에서 확인된 비효율 (원인 후보 대장)

1. **중복 K·v 적용** [실측 — 최대 발견]: `cmf_error_envelope_refine` 는 반복마다
   `apply_upper(candidate)` (`cmf_error_envelope.c:163`) 후 `verify(next)` (`:194`,
   내부에서 `apply_upper(next)` `:95`)를 부른다. verify 통과 시
   `candidate ← next` (`:200`) 이므로 **다음 반복의 `apply_upper(candidate)` 는 직전
   verify 가 방금 계산한 K·(같은 벡터) 를 그대로 재계산**한다. 같은 이유로 refine
   초입 verify 는 seed 루프의 마지막 성공 verify 와 중복이다. 콜당 순수 낭비
   ≈ **37 sweep** (73 → 36) [모델 — refinements=36 기준].
2. **콜당 이중 컨텍스트** [실측]: solve_impl 이 ~39 GB/장치를 할당·업로드하고
   해제한 뒤, PersistentBoundContext 가 **같은 정적 자료를 다시** 할당·업로드한다.
   프로토타입 문서가 "K·u 적용마다의 재할당"은 고쳤으나 solve↔envelope 사이의
   이중화는 남아 있다.
3. **1단계 3-스레드 직렬 체인** [실측]: 블록당 유효 병렬 3/128, 체인 길이 최대
   9,108 compose. SM 효율의 구조적 하한. 단 이것은 결합 순서 보존의 대가로 **의도된
   설계**이고, 병렬화(스캔화)는 합산 순서를 바꾼다(§6 금지 목록).
4. **pageable 호스트 버퍼** [실측]: 모든 HtoD/DtoH 가 `std::vector`/`malloc` 대상
   동기 `cudaMemcpy` — pinned 대비 대역 절반 이하가 보통이고, 회수도 d0→d1 직렬.
5. **파티션 가중의 대표성** [실측 코드 + 추정]: 가중치가 `rn`(세그먼트 수)뿐
   (`build_ray_partition :213-218`). epoch-경로 비용·점유 구조는 반영 안 됨.

---

## 5. 수리안 — 우선순위·예상 이득·위험·결정론 판정

착수 전제: **각 안은 해당 지분의 분모 실측(M0-M3) 후에만 발주.** 코딩은 Codex,
발주·게이트·커밋은 운전석. 소스 편집 동결(A2-10 귀속 전) 해제와 단(rung) 발의는
user 판정 사항이다.

### 5.1 [1순위] C1 — 봉투 K·v 중복 제거 (캐시 재사용)

- 내용: refine 의 verify 가 계산한 `K·next` 를 다음 반복의 apply 입력으로 재사용
  (verify 가 ku 를 반환하도록 `cmf_error_envelope.c` 시그니처 확장, 또는 refine
  내부 재구성). seed 루프 마지막 verify ↔ refine 초입 verify 중복도 함께 제거.
- 예상 이득 [모델]: 콜당 sweep 73→36 (봉투分). iterations_used≈20·seeds≈2 로 놓으면
  콜 전체 sweep 98→61 — **GPU-sweep 시간 ~35-40% 절감**. 실제 분모는 M0 로 확정.
- 위험: `persistent-upper-application-count` selftest 의 계수 공식이 바뀐다(출력
  아닌 **테스트 계약** 갱신 — 사전등록에 명시). 봉투 모듈은 CPU exact 오너와 공유
  — 양쪽 다 이득, 양쪽 다 회귀 필요.
- ★결정론: **바이트 불변 — by construction.** 같은 결정론 연산자(고정 구성의 GPU
  실행은 재실행 byte-identical [실측 — 프로토타입 문서 "two ordinary executions
  were byte-identical"; 합산에 원자적 연산 없음, 출력 셀당 단일 스레드 기록])에
  같은 입력 벡터를 재적용하는 것을 생략할 뿐이다. J·error_upper 모두 불변.
  합산 순서 변경 없음.

### 5.2 [2순위] C2 — solve/persistent 이중 컨텍스트 통합 (+pinned 버퍼)

- 내용: solve_impl 의 shard 를 PersistentBoundContext 로 승격해 콜당 할당·업로드를
  1회로; 호스트 staging 버퍼 pinned 화; 가능하면 콜 간(패스 간) 컨텍스트 보존.
- 예상 이득: `initialization_s + envelope_context_setup_s + cleanup_s` 지분 전액과
  h2d/d2h 의 ~½ [분모는 M0]. 콜당 수십 GB 의 cudaMalloc/Free ×2 제거.
- 위험: 수명·실패 경로(transactional 계약 — 비-OK 시 J 바이트 불변)를 깨지 않는
  구조 변경이라 리뷰 부담 중간. 앱의 다른 CUDA 모듈과의 메모리 공존(장치0 +15 GB)
  확인 필요.
- ★결정론: **바이트 불변** — 같은 값을 같은 순서로 업로드·연산; 할당 방식과 전송
  방식(pinned)은 산술에 관여하지 않는다. 단 구현에서 source 조립 산술의 위치·순서를
  옮기지 않을 것(조립 루프는 그대로 호스트, 셀별 독립).

### 5.3 [3순위] A1 — epoch_batch_cardinality 64 → 상향 (예: 1024·4096)

- 내용: `lumina_cmfgen.c:4646` 의 상수 1개. 세그먼트당 런치 수가 최대 954→60·15 로
  감소 → `cuLaunchKernel` 스핀·큐 배압 완화 (H2 성분 회수).
- 예상 이득: H2 지배가 M2 로 확인될 때만 유의미. GPU 가 일로 포화라면 런치
  오버헤드는 대부분 숨어 이득 미미 [추정].
- 위험: no-op 블록(조기 exit) 증가 — 스케줄링 오버헤드 미미 [모델]; workspace
  불변(스케줄과 무관, `:415-427`); grid 한계 여유.
- ★결정론: **바이트 불변 — 계약+게이트 이중 보증.** 헤더가 "changing it must not
  change any result bit" 를 명시하고(`cmf_exact_multigpu.h` epoch 항), selftest 의
  스케줄 매트릭스 `{32,1,1},{128,7,4},{256,1000,64}` 가 bitwise 동일성을 실검증한다
  (`epoch-schedule-matrix-bitwise-serial`) [실측]. 단 production 규모(NF≈2e6)
  마이크로픽스처에서 byte-parity 1회 재확인 후 적용(게이트는 소격자였다).

### 5.4 [4순위] A2 — 장치별 공급 스레드 (launch_sweep 재구성)

- 내용: 장치당 호스트 스레드 1개가 자기 장치의 세그먼트 루프·회수를 전담. 장치 간
  결합(H2·H3) 제거, 꼬리 회수도 병렬화.
- 예상 이득: H2 성분 전액 + 회수 직렬화 소거. H1 은 못 고친다.
- 위험: 구현 복잡도 최고(스레드별 cudaSetDevice, 실패 경로 합류, 에러 전파).
  A1 로 H2 가 이미 소거된다면 불필요할 수 있음 — A1 측정 후 결정.
- ★결정론: **바이트 불변** — 각 장치 스트림에 들어가는 커널 열과 그 순서가 변하지
  않는 한(그렇게 설계할 것) 산술은 동일. 호스트 스레드 경합은 enqueue 시점만
  바꾼다. 검증은 byte-parity 게이트.

### 5.5 [5순위] D — 호스트 위상 병렬화 (OMP; 유휴 31코어의 옳은 용처)

- 내용: source 조립(`:1909-1930`)·host 환원(`:1978-1992`)·수렴검사(`:1994-2016`)·
  봉투 검사 루프의 셀-병렬 OMP. 셀별로 완전 독립이라 배정만 나누면 된다.
- 예상 이득: 0/0 구간 중 (a)·(c) 지분 [분모는 M0]. 참고: 대장 8절의 "OMP 는 틀린
  처방" 판정은 **sweep 구간**에 대한 것이고 유효하다 — 이 안은 sweep 이 아니라
  로그가 호스트 위상으로 실측한 지분에만 적용한다.
- ★결정론: **바이트 불변 — 조건부.** 셀 루프의 병렬화는 셀당 산술(장치-합 d 역순
  포함)을 건드리지 않는다. 수렴검사의 max 환원은 결합 순서 무관(max 는 결합적).
  단 **셀 내부 합의 분할·리덕션 절대 금지** — 그 순간 합산 순서 변경이 된다.
  구현 리뷰에서 이 경계를 명시 조항으로.

### 5.6 [6순위·별도 단] B — 파티션 재설계 (epoch-비용 가중)

- 내용: `build_ray_partition` 의 가중을 세그먼트 수에서 M3 비용 모델(블록 수·체인
  길이·점유 구조)로 교체, 또는 경계 위치만 이동.
- 예상 이득 상한: 편측-busy 꼬리 전액 = 사이클당 ~6/14 ≈ **43%p 중 절반**(균형화는
  꼬리를 절반씩 나눔) ≈ sweep 시간 ~21% [모델 — 관측 1주기 기준, M1 로 분포 확정].
- ★결정론: **바이트 변경 — 반드시 명시.** 근거 [실측]: (i) 셀당 J 는
  `sum += partial[d]` 를 d 내림차순으로 더하는데(`:1979-1982`, `:3102-3115`) 장치
  경계가 움직이면 셀-합의 **괄호 묶임이 이동**한다(부동소수 덧셈 비결합). (ii)
  프로토타입 문서 실측: 같은 문제의 1/2/3/4-GPU 파티션 상호차 3-6×10⁻¹⁶ — 즉
  경계가 다르면 이미 비트가 다르다. **DET 레인의 bit-identity 잣대 아래에서는
  재베이스라인(새 기준 바이트 확정 + 봉투 교차검증)을 동반한 별도 단으로만 발의.**
  대안(경계 불변 유지한 채 가중만 검증용 A/B)은 이득이 없다. 참고: 파티션과 무관하게
  광선별 결과 자체는 halo 재계산으로 파티션 불변 — 오염되는 것은 각 셀의 합 결합
  순서뿐이며 크기는 ~1-2 ulp [실측 문서].

### 5.7 금지 목록 (결정론 위반이 구조적인 안)

- **커널 1단계 체인의 병렬 스캔화**: compose 는 부동소수 곱·합의 사슬 — 결합 순서
  변경 = 바이트 변경. 프로토타입이 "CPU two-stack composition order 보존"을 계약으로
  삼았다 [실측]. 바이트-변경 단으로 별도 발의하지 않는 한 금지.
- **장치-합을 GPU 에서 수행(NCCL/peer reduce)**: 같은 이유.
- **seed 초기값 개선**: J 는 불변이나 error_upper 후보 궤적이 변해 **error_upper
  바이트가 변한다**(그리고 error_upper 는 `jbar_line_det_error_upper` 로 하류 공표됨
  `lumina_cmfgen.c:6328` 일대) — 바이트-변경 항목으로만 취급.
- **refinements=36 축소**: 같은 이유로 error_upper 바이트 변경 + 이 판정런의
  사전등록 구성 훼손. 이 감사는 권고하지 않는다.

---

## 6. 결정론 판정 총괄표

| 안 | 바이트 영향 | 근거 | 검증 게이트 |
|---|---|---|---|
| C1 중복 K·v 제거 | **불변** | 동일 입력·동일 결정론 연산자 재계산의 생략 | production-규모 byte-parity + selftest 계수식 갱신 |
| C2 컨텍스트 통합·pinned | **불변** | 산술 비관여(할당·전송 방식) | byte-parity + transactional 실패경로 회귀 |
| A1 batch 상향 | **불변** | 헤더 계약 명문 + 스케줄 매트릭스 bitwise 게이트 [실측] | 기존 게이트 + production-규모 1회 |
| A2 장치별 스레드 | **불변** | 장치 스트림 내 커널 열·순서 불변 설계 | byte-parity |
| D 호스트 OMP | **불변(조건부)** | 셀 독립; 셀 내부 합 분할 금지 조항 | byte-parity + 1/32스레드 교차 |
| B 파티션 이동 | **변경** | 장치-합 괄호 이동; 프로토타입 실측 1e-16 상호차 | 재베이스라인 + 봉투 교차(AB 하니스) |
| 스캔화·GPU 환원·seed·refinements | **변경** | 합산 순서/공표값 | (금지 또는 별도 바이트-변경 단) |

공통 byte-parity 픽스처: `tests/cmf_exact_multigpu_reduced_bench.cu` — production
기하 CSV(`shell,r_inner,r_outer,v_inner,v_outer`) + `CMF_MGPU_REDUCED_BINS`(상한
2,013,113)·`CMF_MGPU_EPOCH_*`·파티션·장치수 env 전부 노출, J/error_upper 를
`LUMINA_MGPU_S2` 매직의 바이너리로 덤프 [실측] — Makefile 표적
`bench_cmf_exact_multigpu_reduced` (`Makefile:104`). **수리 전후 비교는 전부 이
픽스처의 파일 byte-diff 로 한다** (3층 인수 프로토콜의 마이크로픽스처).

---

## 7. 측정 계획 — 잣대 설계 (순간값 금지)

순서대로. M0-M2 는 소스 0줄. 모든 수리안의 착수 조건이 여기 걸린다.

- **M0 (지금·로그 수확·최우선)**: 직전 완주 런과 이번 런의 stderr 에서
  `[cmf_fine][EXACT-MULTIGPU-TIMING]`(위상별 **콜-누적 초** 15종),
  `[EXACT-MULTIGPU-DEVICE]`(장치별 rays·owned/computed work·bytes),
  `[EXACT-MULTIGPU-EPOCH]`(iterations_used·seed_attempts·refinements·max_drift),
  `[cmf_fine] ... NF=` 를 전부 수확.
  산출 잣대: ① 위상별 지분표(분모=caller_total_s), ② sweep당 초 =
  device_sweep_s ÷ (iterations+3+seeds+1+2×36), ③ §3 후보 (a)-(d) 지분 확정,
  ④ NF 확정(패킷 "1234 bin" 정정). **이 로그는 누적치라 순간값 함정이 없다.**
  이번 런에서 아직 콜이 안 끝나 TIMING 이 없으면 직전 런 로그로.
- **M1 (살아있는 런·비침습)**: `nvidia-smi --query-gpu=timestamp,index,
  utilization.gpu,power.draw,memory.used --format=csv -lms 500` 을 **≥5분(전체
  주기 ≥3개)** 연속 기록. 주기 경계(0/0→양쪽 100)로 접어 per-cycle
  {양쪽busy, 편측busy, 양쪽idle} 길이의 **중앙값·IQR** 산출. 20표본 창 재사용 금지.
- **M2 (판별 실험·비침습·★H1 vs H2 확정)**: M1 과 동시에 1 Hz `eu-stack` 60회를
  **타임스탬프와 함께** 수집, GPU 위상별로 조인:
  0/100 구간에서 호스트가 `cuMemcpyDtoH`(회수 대기)면 H1, `cuLaunchKernel` 스핀이면
  H2. 30표본 이상으로 위상별 상태 점유율 표를 만든다.
- **M3 (오프라인 비용 모델·결정론적)**: 기하(run 의 geometry.csv)+t_exp+dlognu+NF
  에서 장치별로 (i) owned/computed segment work(→DEVICE 로그와 대조해 모델 검증),
  (ii) window>32 쌍 수, (iii) Σ ceil(NF/window)(블록 수), (iv) Σ(3×window×epochs)
  (체인 연산량), (v) 세그먼트-패스별 max window·활성 광선 수. 장치1/장치0 비를
  산출해 관측 busy-비 (~14/8) 와 대조 — H1 의 정량 상한. 스크립트는 Codex 발주,
  실행은 grammar-debug (로그인 노드 연산 금지).
- **M4 (마이크로픽스처 A/B·사전등록 후 slurm 1회씩)**: reduced bench 로
  {batch 64↔2048} × {devices 1↔2} × {NF 축소판·전판}. 각 구성 (i) J/error_upper
  byte-diff(결정론 게이트), (ii) 벽시계 **5회 반복 중앙값**(워밍업 1회 제외),
  (iii) M1 방식 시계열. 기대치(예: "batch 상향으로 편측busy 비율 감소")를 **런 전에
  수치로 등록**하고 1회 판정.
- **M5 (선택)**: syn 노드에 `nsys` 존재 여부 확인(`which nsys`) — 있으면
  마이크로픽스처 30초 캡처 1회로 장치별 커널 타임라인 직접 판독. 판정런에는 붙이지
  않는다.
- **V3 (확인 항목)**: ① RUN_FOOTER 에서 `LUMINA_CMF_FINE_MGPU_AB`·
  `LUMINA_A210_INDEPENDENT_CAPTURE` 값(켜져 있으면 콜 수·벽시계 해석이 달라짐 —
  AB=1 이면 콜마다 CPU 전판 솔브가 추가로 돈다 `lumina_cmfgen.c:5854-5876`),
  ② GPU0 의 +15 GB 가 어느 모듈 소속인지, ③ 두 A100 이 동일 SKU·클럭인지
  (`nvidia-smi -q` — 이기종·스로틀링이면 H1 해석이 오염된다).

---

## 8. 요약 판정

- **지목한 원인**: (확정 [실측]) sweep 중 장치 간 동기 없음·유일 결합은 단일 스레드
  enqueue 와 말미 회수; 봉투 기계의 K·v 중복(반복당 1 sweep 낭비)·콜당 이중
  컨텍스트는 코드로 확정. (미확정 [추정]) 0/100 꼬리의 지배 기전은 H1(가중 잣대
  불일치) vs H2(공급 직렬화) — M2·M3 판별 대기.
- **권고 순서**: M0→M1/M2→M3 측정 후, C1(바이트 불변·최대 이득 후보) → C2 → A1
  → (필요시) A2 → D. B(재파티션)는 바이트-변경 단으로만.
- **결정론 위험이 있는 안**: B(파티션 이동 — 장치-합 괄호 이동), seed/refinements
  변경(error_upper 공표값 변경), 커널 체인 스캔화·GPU 환원(합산 순서) — 전부 명시
  완료(§6).
- **운전석이 추가로 잴 것**: M0 로그 수확(즉시·무편집), M1+M2 동시 시계열(즉시),
  V3 세 가지 확인, M3 스크립트 발주, M4 사전등록.

— 이상. 이 문서는 지목과 설계다. 코드 변경 0줄.

---

## 부록 M0 — 운전석 계측 (감사가 지목한 "즉시·무편집" 항목, 2026-08-20 실측)

감사 §M0 지시대로 완주한 판정런(job 321104)의 로그에서 **이미 코드에 있던 15위상 타이머**를
수확했다. 편집 0줄.

### M0-1. 위상별 벽시계 — 분모가 확정됐다

한 콜(`reported_total_s=2945.56`, `caller_total_s=2945.59` — 오차 0.03 s):

| 위상 | 초 | 총계 대비 |
|---|---:|---:|
| **`device_sweep_s`** | **2362.60** | **80.2 %** |
| **`envelope_refine_s`** | **1945.53** | **66.1 %** |
| `fixed_point_s` | 894.38 | 30.4 % |
| `source_assembly_s` | 226.33 | 7.7 % |
| `host_reduction_s` | 182.07 | 6.2 % |
| `bounds_s` | 74.51 | 2.5 % |
| `h2d_s` / `d2h_s` | 43.37 / 23.65 | 1.5 / 0.8 % |
| `envelope_verify_s` | 26.46 | 0.9 % |
| `convergence_check_s` | 7.44 | 0.3 % |
| `initialization_s` | 4.78 | 0.2 % |
| `publication_s` · `cleanup_s` | 0.24 · 0.09 | ~0 |

⟹ **오차-봉투 refine 이 콜 전체의 66%다.** 감사가 지목한 「refine 루프가 동일 입력의 K·v 를
반복당 두 번 계산」이 사실이면, C1(캐시)의 상한은 **총 벽시계의 ~33%** 다.
감사의 "~35-40% GPU-sweep 절감 후보" 와 정합한다.

⟹ 전송은 병목이 아니다(`h2d+d2h` = 2.3%). **계산이 병목이고, 그 안에서 refine 이 지배한다.**

### M0-2. ★장치 분할 실측 — 불균형의 형태가 특정됐다

```
[cmf_fine][EXACT-MULTIGPU-DEVICE] index=0 rays=[0,20)  owned_segment_work=990  computed_segment_work=1035  allocated=38.74 GB
[cmf_fine][EXACT-MULTIGPU-DEVICE] index=1 rays=[20,66) owned_segment_work=1035 computed_segment_work=1035  allocated=40.36 GB
```

| 잣대 | GPU0 | GPU1 | 비 |
|---|---:|---:|---:|
| **rays** | **20** | **46** | **1 : 2.3** |
| `owned_segment_work` | 990 | 1035 | 1 : 1.045 |
| `computed_segment_work` | 1035 | 1035 | 1 : 1 |

⟹ **파티션은 `segment_work` 를 균등화한다(4.5% 차).** 그런데 커널 그리드는
`dim3 grid(count, shard.local_rays, 1)`(`cmf_exact_multigpu.cu:1398`)로 **rays 를 y 차원에
쓴다** — GPU0 는 블록이 **2.3배 적다.**

⟹ 두 장치가 같은 수의 커널을 띄우되 **GPU0 의 커널이 훨씬 작아 먼저 끝나고 논다.**
관측된 `100/100 → 0/100` 패턴과 정합한다.

★**감사의 H1 이 지지된다**(파티션이 세그먼트 *수* 만 균등화, 블록 구조 미반영).
H2(런치 큐 배압)를 배제하지는 못했다 — 감사가 지시한 **M1+M2 타임스탬프 조인**이 여전히 판별자다.

★**단정하지 않는다**: `computed(1035) > owned(990)` 이 GPU0 에 45단위의 halo 계산을 준다는
사실도 있어, 실제 커널 시간이 rays 비에 선형인지는 **재지 않았다.**

### M0-3. 감사가 정정한 운전석 오류

[감사 실측] fine 격자는 **1,234 bin 이 아니라 NF ≈ 2.01M bin** 이다. 운전석 패킷 §4 가
manifest 의 `n_bins=1234`(= **비교 격자**)를 fine 솔버 격자로 잘못 적었다.
GPU1 의 38.9 GB 메모리와 정합하는 것은 2.01M 쪽이다. **정정한다.**

### M0-4. 다음 계측 (감사 지시 순서 그대로)

- **M1+M2**: `nvidia-smi` 500 ms × ≥5분(주기 3개 이상) + 1 Hz `eu-stack` 60회를
  **타임스탬프 조인** → 0/100 구간에서 호스트가 `cuMemcpyDtoH` 면 H1, `cuLaunchKernel`
  스핀이면 H2. ⚠**다음 판정런이 있어야 한다**(살아 있는 런 필요).
- **M4**: `bench_cmf_exact_multigpu_reduced`(`Makefile:104`, byte-dump 내장) 마이크로픽스처로
  C1·A1 의 **바이트 불변 A/B** — 측정 없이 착수 금지.
