# CMF exact within-ray scan specification — 2026-08-10

## 판정

일반적인 associative tree scan은 **기각**한다. 현재 positive affine 합성은
실수 대수에서는 결합적이지만, sealed binary64 operator는 각 multiply/add node에서
nearest 또는 directed rounding을 수행한다. 따라서 괄호를 바꾸는 Blelloch,
Hillis–Steele, CUB inclusive scan은 현재 결과를 모든 입력에서 bitwise 재현할 수 없다.

대신 구현 후보는 **canonical two-stack transfer epoch replay**다. 이 방법은 각 serial
fold의 괄호와 각 node 내부 `multiply → multiply → add` 순서를 그대로 유지한다.
서로 독립인 epoch와 fold chain, 그리고 aggregate 이후의 bin별 pointwise transport만
병렬 실행한다. 이는 logarithmic-depth associative scan이 아니라, 기존 expression DAG를
재배치 없이 병렬 스케줄하는 exact decomposition이다.

이 단계에서는 production dispatch나 GPU transport kernel을 변경하지 않는다. 명세와
구조 증명 하니스만 추가한다. floor, cap, clamp, jitter, `abs` 대체, 음수 무시 또는
물리값 수리는 허용하지 않는다.

## Sealed primitive

한 transform을 `A=(T_A,E_A)`, 즉 `A(x)=T_A x+E_A`로 둔다. 저장 순서에서 A 다음 B를
통과하는 기존 reverse composition을 `R_d(A,B)`라 쓰면, mode
`d ∈ {lower, nearest, upper}`마다 정확히 다음 세 node를 순서대로 평가한다.

```text
T           = multiply_d(T_B, T_A)
attenuated  = multiply_d(T_B, E_A)
E           = add_d(E_B, attenuated)
R_d(A,B)    = (T,E)
```

Directed multiply는 nonzero product를 지정 방향의 인접 binary64 값으로 한 번 넓힌다.
Directed add는 기존 TwoSum residual의 부호에 따라서만 한 번 넓힌다. 이 node와 순서는
`src/cmf_exact_sliding.c:249-317` 및 `src/cmf_exact_multigpu.cu:394-466`이 정본이다.
새 경로는 동일 helper 또는 bitwise 동등성이 별도로 증명된 helper만 호출할 수 있다.

여기서 “directed-rounding 순서 보존”은 각 logical chain의 parent-child 관계와 각
`R_d` 내부 세 primitive의 순서를 뜻한다. 서로 operand 의존성이 없는 두 logical node의
wall-clock 실행 순서는 결과 의미가 없으므로 병렬일 수 있다. 전역 호출을 하나의 total
order로 강제하면 어떤 병렬 실행도 정의상 불가능하다.

## 표준 O(log n) scan이 exact일 수 없는 이유

Sequential fold 상태는 `S_k=R_d(S_{k-1},V_k)`이며 `S_k`는 이미 반올림된 두 binary64
값이다. 다음 node는 그 반올림 결과를 직접 operand로 사용한다. 같은 `(T,E)` 표현만
허용하면 `S_k` 이전에 `S_{k+1}`을 평가할 정보가 없다. Balanced tree가 이를 우회하려면
`R_d(R_d(A,B),C)=R_d(A,R_d(B,C))`가 모든 입력에서 성립해야 하지만 실제로 성립하지
않는다.

구조 증명 하니스 `scripts/verify_cmf_exact_epoch_formula.py`는 아래 hex binary64 witness를
현재 primitive 그대로 평가해 lower/nearest/upper 모두에서 두 괄호가 다름을 확인한다.

| mode | A `(T,E)` | B `(T,E)` | C `(T,E)` | 달라지는 결과 |
|---|---|---|---|---|
| nearest | `0x1.7ef1e3b8709dcp-7`, `0x1.14368367ce85ap-30` | `0x1.f47d6ab739746p-3`, `0x1.2916788bc7ab5p-12` | `0x1.f858297efd535p-2`, `0x1.94c196e83936ep+1` | left/right `T`와 `E`가 각각 1 ulp |
| lower | `0x1.fda66fd3b058fp-4`, `0x1.42f617f05525ap-3` | `0x1.4f9fee1fe330ep-7`, `0x1.fb0e76b3ec976p-14` | `0x1.b9d3be38204abp-4`, `0x1.7e82febc2e09cp-24` | left/right `E`가 1 ulp |
| upper | `0x1.c0a2e8eeb73d8p-4`, `0x1.742d6735617fep+4` | `0x1.05e72455d5868p-4`, `0x1.f7875c0da4e2ap-1` | `0x1.64b1e4f1221f8p-7`, `0x1.05e51e0c95836p+7` | left/right `T`가 1 ulp |

더 큰 transition-function 표현 또는 모든 binary64 상태에 대한 표를 합성하면 이론적으로
chain을 다른 형태로 들 수 있지만, 이는 현재 두-double 상태 표현을 바꾸며 크기도
비현실적이다. exact accumulator나 superaccumulator도 현재 node별 rounding operator를
다른 operator로 바꾸므로 이 단계의 해답이 아니다.

## Exact transfer-epoch decomposition

### 정의

`B=n_bins-1`, `W=max(qtop-1,0)`이고 다음 raw transform을 둔다.

```text
V_m = (t1[bounded_bin_index(m)], source_cell[bounded_bin_index(m)])
```

`bounded_bin_index`는 기존 frequency-array index 경계 규칙이며 물리값에 대한 floor/cap이
아니다. `W=0`이면 aggregate는 모든 bin에서 identity `(1,0)`이고 각 bin은 즉시 독립이다.

`W>0`일 때 epoch `e=0,1,...`의 boundary output bin은

```text
b_e = B - e W,      b_e >= 0
L_e = min(W, b_e+1)
```

이다. 현재 queue는 정확히 W회 pop마다 front가 비고 W개의 back raw value를 transfer한다.
따라서 각 epoch boundary에서 back에 들어 있는 raw push 순서는 항상

```text
V_(b_e+W), V_(b_e+W-1), ..., V_(b_e+1)
```

이다. 이는 이전 epoch의 rounded aggregate가 아니라 raw `value` node들만으로 다시 만들
수 있다. 기존 transfer도 back aggregate를 사용하지 않고 raw value를 역순으로 꺼내므로
epoch끼리 독립 재생해도 canonical node operand가 같다.

### 세 canonical chain

각 epoch는 다음 세 chain을 각각 정해진 방향의 serial fold로 만든다. 이 chain 내부에는
tree scan을 적용하지 않는다.

1. Boundary back chain `Q_e`

```text
Q[0] = V_(b_e+W)
Q[r] = R_d(Q[r-1], V_(b_e+W-r)),       r=1..W-1
G_(e,0) = Q[W-1]
```

2. Transferred front chain `F_e`

```text
F[0] = V_(b_e+1)
F[r] = R_d(V_(b_e+1+r), F[r-1]),       r=1..W-1
```

3. New back chain `P_e`

```text
P[0] = V_(b_e)
P[r] = R_d(P[r-1], V_(b_e-r)),         r=1..L_e-2
```

Epoch 내부 offset `j=1..L_e-1`의 window aggregate는 기존
`positive_window_aggregate(front,back)`와 같은 orientation으로

```text
G_(e,j) = R_d(F[W-j-1], P[j-1])
```

이다. `j=0`은 boundary back-only aggregate `Q[W-1]`다. 마지막 partial epoch도 F와 Q는
W개 raw value를 사용하고 P만 `L_e-1`개를 사용하므로 기존 lower-frequency tail과 같다.

`Q`, `F`, `P`는 서로 raw input만 공유하고 dependency를 공유하지 않는다. 세 lane이
동시에 각자의 serial chain을 계산할 수 있다. 모든 epoch도 서로 독립이다. 현재 queue가
이전 epoch의 P aggregate를 다음 boundary Q로 재사용하는 것과 달리, 병렬 후보는 Q를
동일 raw 순서로 재계산한다. work는 늘지만 결과 bit pattern은 같다.

### Pointwise transport phase

`G_(e,j)`가 준비된 후 output bin `b=b_e-j`는 기존 순서 그대로 다음을 실행한다.

1. upstream의 `b+q`, `b+q+1` interpolation;
2. top partial-cell transmission/source update;
3. `G_(e,j)` transmission/emission 적용;
4. local half-cell update;
5. finite/nonnegative 검사와 transactional output staging.

이 다섯 단계는 다른 output bin의 queue state를 읽거나 쓰지 않으므로 bin threads로
병렬화할 수 있다. 첫 구현에서는 outward/inward ray-segment 순서, angular reconstruction,
device partial, host reduction 순서를 모두 그대로 둔다. 즉 최적화 범위는 한 segment의
frequency 계산 안으로 한정한다.

`beta<=0.5` fast path와 `W=0` path는 원래부터 bin 간 state가 없으므로 기존 per-bin
primitive 순서를 한 thread가 맡는 직접 병렬 kernel을 사용한다.

## CUDA scheduling and memory contract

첫 prototype의 logical mapping은 `(device shard, ray, segment, rounding, epoch)` block이다.
한 block에서:

- chain lane 0은 Q, lane 1은 F, lane 2는 P의 canonical recurrence만 수행한다;
- block synchronization 뒤 remaining threads가 `j=0..L_e-1` pointwise output을 맡는다;
- 동일 output address는 정확히 한 epoch/thread만 소유한다;
- failure는 기존 atomic first-failure record에 logical epoch/chain/node를 추가한다;
- 실패 시 caller `J`, lower/nearest/upper, error envelope는 byte-unchanged다.

한 live epoch의 aggregate workspace는 Q scalar 외에 F와 P의 `(T,E)` pair를 저장하므로
최대 약 `32 W` bytes/ray다. 기존 두 `PositiveWindowNode` stack은 약 `64 W` bytes/ray다.
다만 모든 epoch를 동시에 resident하게 하면 `ceil(n_bins/W)`배 workspace가 필요하고
대략 `32 n_bins` bytes/ray가 된다. 따라서 epoch batch cardinality는 launch/resource
scheduling 값으로만 조절할 수 있다. 이 값은 어떤 물리량이나 계산 결과도 자르거나
대체하지 않으며, batch 수가 달라도 결과가 byte-identical해야 한다.

Production shape 기준:

- reduced 8k에서 최대 `W=9108`인 segment는 epoch가 하나뿐이다. 이 경우 기대 가능한
  이득은 세 chain 동시 실행과 pointwise-bin 병렬화뿐이며 큰 속도 향상을 미리 주장하지
  않는다;
- full `n_bins=2013113`, 보수적 `W=47649`이면 최대 43 epochs다. epoch 병렬성이 생기지만
  global workspace와 occupancy를 반드시 실측해야 한다;
- chain 하나의 dependency depth는 여전히 `O(W)`다. `O(log W)`라고 보고하면 실패다.

CUDA build는 `--use_fast_math`를 금지한다. Nearest pointwise expression의 FMA 여부까지
기존 kernel과 달라질 수 있으므로, 동일 device helper를 재사용하고 SASS/bitwise gate로
확인한다. 성능을 이유로 primitive를 합치거나 `exp`, `sqrt`, multiply/add 순서를 바꾸면
exact candidate 자격을 잃는다.

## Structural evidence

`scripts/verify_cmf_exact_epoch_formula.py`는 다음을 수행한다.

- 위 세 nonassociative witness가 lower/nearest/upper에서 실제로 갈라짐을 확인;
- 기존 serial push/transfer/pop/aggregate를 독립 구현;
- epoch 공식으로 모든 output-bin aggregate를 독립 재구성;
- `n_bins=1..17,31/32/33,63/64/65,96`과
  `W=0,1,2,3,n-1,n,n+1,2n+3` 경계;
- identity, zero emission, minimum subnormal emission, near-one transmission,
  fixed-seed broad-exponent random transforms;
- lower/nearest/upper 전 mode의 `(T,E)` pair bit equality.

현재 결과는 `6588/6588` cases bit-identical, numerical repair 0이다. 이 Python 하니스는
구조 공식 증거이며 CUDA 구현 인증을 대신하지 않는다.

## Small-grid proof gates

다음 구현 단계는 아래 gate를 순서대로 통과해야 한다.

1. **G0 nonassociation guard** — 위 세 witness가 계속 다른 결과를 내야 한다. 같아지면
   compiler arithmetic 또는 helper가 바뀐 것이므로 fail한다.
2. **G1 aggregate identity** — serial CPU two-stack과 epoch CPU/CUDA가 모든
   `(ray,segment,bin,rounding)`의 aggregate T/E를 bitwise 비교한다.
3. **G2 logical trace identity** — 각 replay node를 대응하는 serial push/transfer
   fold node에 mapping하고 `(primitive,input-bits,output-bits)`가 같아야 한다. 독립
   epoch를 위해 재계산하는 `Q_e`와 이전 epoch의 `P_(e-1)`는 같은 serial back-fold
   node에 각각 mapping해 둘 다 검사한다. 재계산 때문에 실행 node 수가 늘어나는 것은
   허용하지만 operand, parenthesization, primitive 내부 순서는 바꿀 수 없다.
4. **G3 segment identity** — upstream zero/nonzero, high-frequency index repetition,
   `phi`의 0.5 양쪽, W의 0/1/2 및 epoch/warp/block 경계를 포함해 segment output을
   lower/nearest/upper 각각 bitwise 비교한다.
5. **G4 full small sweep** — direct oracle, serial positive, epoch positive의 finite J를
   비교하고 `lower <= nearest <= upper`; 기존 componentwise envelope가 direct oracle
   차이를 전 셀에서 덮어야 한다.
6. **G5 scheduling invariance** — epoch batch cardinality, block size, 1/2/4 GPU,
   repeated run을 바꿔도 mode별 result가 byte-identical이어야 한다.
7. **G6 fail-closed** — invalid mode/index/workspace, injected nonfinite, allocation/CUDA
   failure가 정확한 logical node를 보고하고 모든 public output을 보존해야 한다.
8. **G7 hygiene** — compute-sanitizer 0 errors, repair/floor/cap/clamp/jitter 0,
   production dispatch와 H200 staged input unchanged.

G0–G7 전부를 통과하기 전에는 reduced 8k timing이나 full-grid 실행으로 넘어가지 않는다.
Bitwise identity가 실패하면 tolerance나 envelope만으로 “exact”라고 승격하지 않는다.

## Alternatives verdict

요구 조건 아래 우선순위는 다음과 같다.

1. **Epoch replay + pointwise parallelism** — exact 후보. 현재 명세의 구현 대상이다.
2. **Serial frontier + pointwise parallelism** — epoch 병렬 구현이 실패할 때의 더 단순한
   exact fallback. 속도 상한은 낮지만 canonical queue를 그대로 실행한다.
3. **Full canonical-state checkpoints** — exact할 수 있으나 checkpoint마다 O(W) state,
   직렬 생성 비용과 큰 복제 메모리가 필요해 기본 후보가 아니다.
4. **One thread per output bin canonical replay** — exact하지만 O(n_bins W) work이므로
   small-grid 독립 oracle로만 허용한다.
5. **Balanced tree scan** — 빠를 수 있고 그 tree 자체의 새 lower/upper enclosure는 만들
   수 있으나, 현재 discrete operator와 byte identity가 아니다. 별도 operator 변경 연구로
   분리하며 이번 exact 경로에는 들어오지 않는다.

## Fable review status

이 비결합/scan 가능성 한 쟁점만 Fable에 제한 질의했으나 Claude CLI가 응답 전에
`Exceeded USD budget (0.5)`로 종료했다. Fable 판정은 수신되지 않았고 재질의하지 않았다.
따라서 위 판정은 정본 코드, explicit binary64 witness, 6,588-case epoch identity 하니스에
근거한다.

## G0/G1 C/CUDA 구현 증거 — 완료

독립 하니스 `tests/cmf_exact_epoch_scan_selftest.cu`가 정본 CPU two-stack reference와
CUDA transfer-epoch kernel을 같은 sealed primitive로 구현했다. production source와
dispatch는 연결하거나 변경하지 않았다. CUDA block은 Q/F/P 세 canonical chain을 각각
serial lane에서 계산하고 동기화 뒤 output offset을 분배한다. 이는 chain 내부를 tree로
재결합하지 않는다.

fatbin은 sm_80/sm_86/sm_90을 포함하며 `--use_fast_math`를 사용하지 않았다. binary SHA는
`7024d38fae8e51acb1c418e06c29b9e2cebbae04f7878bd0008b82dc8aa74be3`, source SHA는
`af083156015d2568adb069218d485ab3b933b6d9d8b5e654197952a1b7b134e1`다.

A40 job `252405`는 `syn07`, UUID
`GPU-3106be74-877e-dbb7-363e-3e229868c84e`에서 `COMPLETED 0:0`했다.

- G0: lower/nearest/upper 세 비결합 witness 모두 CPU left/right가 다르고, CUDA가 CPU의
  여섯 결과 pair를 bitwise 재현했다.
- G1: 2,196 base cases × 3 modes = 6,588 cases에서 CPU serial two-stack과 CUDA epoch의
  153,972 aggregate `(T,E)` pair가 전부 bit-identical이다.
- 각 output bin의 `(epoch,offset,boundary,Q/F/P fold index)` 153,972개도 기대 mapping과
  일치했다. 이는 output ownership/mapping 증거이며, 모든 primitive node의 operand와
  result trace를 대조하는 G2를 대신하지 않는다.
- lower≤nearest≤upper는 모든 output pair에서 성립했다. 두 full run의 stdout/stderr는
  각각 byte-identical이다.
- sanitizer-smoke 666 mode cases/29,007 aggregate pair에서 compute-sanitizer는
  `ERROR SUMMARY: 0 errors`, leak 0 bytes다.
- numerical repair/floor/cap/clamp/jitter는 모두 0이다.

전체 ledger는
`validation/sh_grid_low_band_exacthyd_canonical_2026-08-09/cmf_exact_epoch_scan_g01_2026-08-10.log`,
SHA `12d0be3de592a48e90a900e5205591a6d99b05e4963663bdad0bc15e5a9812cc`다.
이 증거는 G0/G1만 닫으며 segment 출력, multi-GPU scheduling, performance 또는 finite
CMFGEN 물리량 재현을 주장하지 않는다.

## G2/G3 CUDA 증거 — 완료

G2는 정본 serial queue의 합성 사건을 `Q/F/P/G`로 분류하고 각 reverse-compose를 다시
`mul_T/mul_E/add_E` 세 primitive record로 분해했다. record에는 mode, epoch, chain,
node, 두 operand bits와 result bits가 들어간다. CUDA epoch replay의 338,472 records가
serial record와 모두 bitwise 일치했다. `e>0`의 재계산 Q는 직전 epoch가 실제로 만든
마지막 P fold를 alias한 정본 record와 대조했다.

G3는 현재 production CUDA serial two-stack segment와 epoch segment를 같은 A40에서
직접 비교했다. upstream zero/nonzero, `phi`의 0.5 아래/정확히/위, W의
0/1/2/3/31/32/33/63/64/65/127/128/129 및 n 주변과 2n+3, high-index 반복을 포함한
3,726 mode cases의 139,788 output이 전부 bit-identical이고 lower≤nearest≤upper다.

첫 실행은 `W=0`, `beta=nextafter(0.5,+∞)`, lower mode에서 1 ulp 차이로 실패했다.
원인은 새 direct-bin 일반 경로가 수학적 identity aggregate를 생략한 것이었다. directed
operator에서는 `1×intensity`도 한 번 인접값으로 넓히는 실제 sealed node이므로, 생략은
허용되지 않는다. 누락 node를 복원한 뒤 job `252408`이 PASS했다. 이는 floor/cap이나
사후 수리가 아니다.

두 full run은 byte-identical이고 sanitizer는 0 errors/leak 0 bytes다. binary SHA는
`f89d6a7b1784201487001900c8831c9286e259b9f01b3b999db040981db80f86`다. ledger는
`validation/sh_grid_low_band_exacthyd_canonical_2026-08-09/cmf_exact_epoch_scan_g23_2026-08-10.log`,
SHA `89d753da92736bad6f59c04333caa5614673a482f1bc5bd90e4892f656e125bb`다.

## 다음 단계

1. G4 full small sweep에서 direct oracle, serial positive, epoch positive의 finite J,
   directed ordering과 componentwise envelope coverage를 확인한다.
2. G5에서 block size와 epoch batch cardinality를 바꿔도 result가 byte-identical인지
   확인한다.
3. G5의 1/2/4 GPU shard/reduction 결과와 반복 실행 결정성을 확인한다.
4. G6에서 invalid/nonfinite/
   allocation failure의 transactional fail-closed를 검증한다.
5. G7 sanitizer와 memory model을 닫은 뒤에만 reduced 8k A40×4 timing으로 간다. 이후
   H200 full-state memory, full-grid smoke, same-identity finite CMFGEN 순서다.

## G4–G7 및 production prototype 구현 증거 — 완료

G4 full small sweep는 3 sweeps/9 rounding sweeps에서 3,456 serial/epoch J values를
bitwise 일치시켰고 direct oracle 1,152/1,152를 lower/upper가 덮었다(job `252411`). G5는
block `32/64/128/256`, batch `1/2/7/all`의 1,206 scheduling runs/347,328 values를
bitwise 봉인했다(job `252425`). A40 1/2/4-device 독립 실행도 canonical ray-order reduction
digest `1667cc4c2584f887`을 반복 재현했다(job `252426`).

G6 transactional probe는 invalid mode/index/workspace, injected allocation/CUDA failure,
실제 NaN을 포함한 7 cases를 실행했다. 실패 6건은 caller output을 byte 보존했고 성공
1건만 staging을 publish했다. NaN은 `epoch=0,Q,node=0,source_index=3`으로 보고됐다.
job `252427`은 반복 byte-identical이고 sanitizer 0 errors/leak 0 bytes다.

production prototype은 기존 serial public API를 그대로 두고 별도의 explicit epoch API를
추가했다. `CMFMultiGPUEpochSchedule`의 block size, epoch batch cardinality,
direct-replay maximum W가 실행 grouping만 선택한다. W가 threshold 이하이면 output별
canonical replay를, 그보다 크면 shard-local global front/back workspace와 epoch blocks를
사용한다. Q/F/P chain 내부는 한 lane의 정본 순서를 유지하며 output offset만 block thread에
분배한다. W=0 identity aggregate node도 생략하지 않는다.

production selftest job `252432`는 solve, directed bounds, persistent componentwise envelope를
serial과 bitwise 일치시켰다. block `32/64/128/256`, batch `1/2/7/1000`, replay threshold
`1/4/64`도 결과를 바꾸지 않았다. invalid schedule은 caller J를 byte 보존한다. 반복 output은
byte-identical이고 compute-sanitizer는 0 errors다.

production-shaped 1,024-bin job `252431`은 51,200 cells, lower/nearest/upper ordering failure
0, compute-sanitizer 0 errors다. 8,192-bin A40×4 job `252433`은 36.502375973 s에 끝났고
serial 371.813001576 s와 result file이 byte-identical이다. 공통 SHA-256은
`aa43bb667c8602691ce89f1169ed014a90474d759a48c0f68b364e2eb7e57b9b`, speedup은
`10.185994518577x`다. finite J는
`[8.8332793258264307e-08,2.4965020783600907e-05]`다.

peak VRAM은 epoch `373/373/397/517 MiB`, serial `373/373/399/515 MiB`이므로 memory-saving
claim은 하지 않는다. 모든 단계에서 repair/floor/cap/clamp/jitter는 0이다. 이 결과는
finite synthetic-coefficient transport의 exact 재현이며 same-identity finite CMFGEN 물리량
비교는 아니다. 통합 ledger SHA는
`337e90efe7dafe516e905d08d3fe672422ca756be2024039d3013eebfbcb96f9`다.

## Compact full-grid 및 production CMF 계수 검증 — 완료

active segment만 저장하는 compact epoch layout으로 full production shape의 메모리 게이트를
닫았다. A40×4 jobs `252438/252443`은 50 shells, 2,013,113 bins,
100,655,650 cells에서 finite synthetic 결과를 cross-binary byte-identical로 재현했다.
공통 result SHA는
`dcda52e5a97cbc92e95522ba92406ad54706354bcbee8fd9511acf70bf0e028c`다.

production owner는 `LUMINA_CMF_FINE_MGPU_DEVICES`로 explicit 선택된다. positive request
실패는 terminal이고 CPU fallback은 없다. `LUMINA_CMF_FINE_MGPU_AB=1`은 같은 조립
상태의 CPU positive owner와 GPU owner를 별도 buffer에서 실행한 뒤 전 셀 finite,
nonnegative, relative J, combined directed envelope를 검사한다. 비교는 결과를 수정하지
않으며 floor/cap/clamp/jitter를 추가하지 않는다.

sealed CMFGEN-derived production deck job `252448`에서 CPU와 A40×4는 각각 45회 수렴했다.
finite J 범위는 CPU
`[8.4086208255147163e-82,1.9072381379446642e-4]`, GPU
`[8.408620825514714e-82,1.9072381379446645e-4]`; max relative 차이
`3.1710829615213259e-15`, combined-envelope ratio max
`0.25924739579810846`이다. R6 E-line 2,180,286개 전부 valid다.

이로써 G0–G7 구조/스케줄/fail-closed 증거, full-grid 메모리·반복 결정성, production
동일-input finite-value 검증까지 닫혔다. 단, 외부 CMFGEN executable의 출력과 직접 비교한
것은 아니며 그 독립 물리 비교는 다음 단계다. 통합 ledger SHA는
`50d1811ac41dec475816f7bdf567c3276096d628e4db5916ba21a58085bac0c4`다.
