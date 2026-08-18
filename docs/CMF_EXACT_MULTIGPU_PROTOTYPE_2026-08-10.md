# CMF exact multi-GPU direct/positive ray-sharding prototype — 2026-08-10

## Verdict

The isolated direct oracle, subtraction-free positive affine monoid, directed
lower/upper application, and componentwise supersolution all pass on four
physical A40 GPUs. Contiguous ray ownership plus one recomputed halo ray per
device boundary needs no pooled device memory, peer access, NCCL, Unified
Memory, floor, cap, clamp, or jitter.

This is still **not a production replacement** for
`CMF_EXACT_MODE_POSITIVE_SLIDING`. The current positive kernel preserves the
CPU two-stack composition order but assigns one sequential frequency walk to
each ray/segment thread. The componentwise prototype also creates a fresh
multi-device context for each `K*u` application. Correctness is sealed on the
small grid; production performance, persistent allocation, full-grid evidence,
and coexistence with the rest of Lumina's CUDA state are not yet sealed.

## Implemented ownership

- `src/cmf_exact_multigpu.cu/.h`: transactional multi-device direct and
  positive solves.
- The 16 core rays plus one tangent ray per shell are partitioned into
  contiguous global ray ranges.
- Each shard recomputes its next global ray as a one-ray halo. This supplies
  the adjacent angular-quadrature sample without peer traffic.
- Each device writes one partial `J(shell,nu)` field. The host reduction order
  is fixed from the largest-impact-parameter block to the smallest.
- The caller's `J` is updated only after convergence. Invalid opacity,
  insufficient devices, CUDA failure, allocation failure, nonfinite output,
  and iteration-cap exhaustion preserve the input bytes.
- The production positive-sliding source and its publication path are not
  called or modified by this API.
- `cmf_exact_multigpu_positive_solve()` uses the same positive transform and
  two-stack reverse-composition order as the CPU owner. Work per segment is
  `O(n_bins)`, independent of beta.
- `cmf_exact_multigpu_apply_positive_bounds()` evaluates lower, nearest, and
  upper together. Segment arithmetic, angular reconstruction, shard-local
  sums, and the final fixed-order host sum are directed outward.
- `cmf_exact_multigpu_positive_solve_envelope()` publishes `J` and the local
  error field together only after verifying `u >= |r| + K*u` componentwise.
  The `K` application has zero fixed source and zero inner boundary.

## Runtime evidence

Final sealed flight:

- Slurm job `252098`, node `syn06`, partition `a40`, four A40 46,068-MiB cards.
- run root:
  `/gpfs/kjhan/lumina/cmf_multigpu_prototype/mgpu_20260809T150124Z_57b252862954`
- staged binary SHA-256:
  `57b2528629545a7761ef096e7dabc181ab83faae42d8a83dbe864930b73c5ea0`
- state `COMPLETED 0:0`, elapsed 5 s.
- CPU direct vs four-GPU max relative difference: `6.805e-16`.
- one-GPU vs four-GPU max relative difference: `5.856e-16`.
- all 2/3/4-GPU partitions max relative difference: `3.075e-16`.
- CPU/four-GPU/one-GPU convergence iterations: `20/20/20`.
- 19 owned rays exactly once; 22 computed rays = 19 + 3 boundary halos.
- two ordinary executions were byte-identical.
- CUDA compute-sanitizer: `ERROR SUMMARY: 0 errors`.
- repair/floor/cap/clamp/jitter: all zero.
- output-manifest SHA-256:
  `d67e118efe172121f0ba449358b53bec1fed532b7f278bb281f0940f0d75a6f1`.
- footer SHA-256:
  `780edbab54bb33550c54db15a184a96a6f82bf8cdd832a57c7431456c12682e4`.

Final positive/envelope flight:

- Slurm job `252103`, node `syn06`, partition `a40`, four A40 46,068-MiB cards.
- run root:
  `/gpfs/kjhan/lumina/cmf_multigpu_prototype/mgpu_20260809T153049Z_fadc17e7377f`
- staged binary SHA-256:
  `fadc17e7377f6e0ed4f7a7188466e5da141fd9ba1f4f51aab34a5b1edb4a9d30`
- state `COMPLETED 0:0`, elapsed 15 s.
- CPU positive vs four-GPU maximum relative difference: `6.199e-16`.
- one-GPU vs four-GPU positive difference: `4.349e-16`; all 2/3/4-device
  partitions: `3.661e-16`.
- CPU directed nearest vs four GPU: `5.905e-16`; one GPU: `4.449e-16`;
  all partitions: `4.348e-16`.
- every cell satisfied `lower <= nearest <= upper`; maximum relative directed
  width was `6.954e-15`.
- componentwise residual upper maximum: `9.487e-20`; refined local envelope
  range: `[2.604e-20,1.021e-19]`; maximum observed direct-error/envelope ratio:
  `0.3546`.
- two ordinary executions were byte-identical; CUDA compute-sanitizer reported
  `ERROR SUMMARY: 0 errors`; repair/floor/cap/clamp/jitter were all zero.
- output-manifest SHA-256:
  `06391eeb5bf46de5303e0fc3d7a85adc64067d4ee53d78f37e83140a906e179a`.
- footer SHA-256:
  `ed7a79c60887fedc2167ad98e0c3a2253b396a6c39464d416461f54cdaafe749`.

Repository-side closure after the flight:

- CPU exact selftest passed with one and four OpenMP threads; maximum
  positive/direct difference was `1.557e-12` and no repair path fired.
- production CPU link and `sm_86` CUDA link passed. The final local
  `sm_80/sm_86/sm_90` fatbin SHA-256 is
  `b6a89050d93e366304ee8f024eaba28cd2722bd26d9035fa531205e36250d700`.
  The production dispatch and source were not changed by this prototype;
  the queued H200 diagnostic remains sealed to its earlier staged SHA.
- header closure is `31/31`; `git diff --check` passed.
- the final full D19/K7/Z12/CP4 gate battery passed. Its log is
  `validation/sh_grid_low_band_exacthyd_canonical_2026-08-09/gate_battery_cmf_multigpu_positive_envelope_2026-08-10.log`,
  SHA-256
  `f6f5f474a0331f6dc47ba2f8b34bb03043585070913002d9220aa5f4a55d7491`.

## Persistent envelope context flight

The componentwise envelope no longer allocates and destroys every device
shard for each `K*u` application.  `PersistentBoundContext` owns the geometry,
static coefficients, physical and zero-boundary buffers, shard partials, and
per-device work buffers for one complete envelope solve.  It uploads the
static state once, then reuses exactly the same shards for the initial
lower/nearest/upper evaluation and all subsequent upper-operator applications.
Failure still publishes neither `J` nor the error field, and the fixed host
reduction order is unchanged.

The sealed persistent-context flight is:

- Slurm job `252105`, node `syn06`, partition `a40`, four A40 cards;
  `COMPLETED 0:0` in 13 s.
- run root:
  `/gpfs/kjhan/lumina/cmf_multigpu_prototype/mgpu_20260809T154516Z_175bd7c3845a`.
- staged binary SHA-256:
  `175bd7c3845ab625426e746af63638ae4cda528293bd9381fc8d59f23fa18c72`.
- one persistent context initialized, 15 directed-bound applications and 12
  upper-operator applications: `persistent_contexts/bounds/upper=1/15/12`.
- after removing only that newly added counter token, every printed numerical
  metric is byte-identical to job `252103`.  In particular, CPU positive vs
  four GPU is `6.199e-16`, one GPU vs four GPU is `4.349e-16`, the maximum
  positive partition difference is `3.661e-16`, and the maximum observed
  direct-error/envelope ratio remains `0.3546`.
- the two ordinary output logs are byte-identical, both SHA-256
  `1eaffc08889585d9f8f5a3288260ee1297f7013d89c92f0f2daf028a35c5ca46`.
  CUDA compute-sanitizer reported `ERROR SUMMARY: 0 errors`; its log SHA-256 is
  `ecc0d61b1b86ea7e3bb6a32d849c18c8269e8be87fe2da9248f791b1e9116957`.
- output-manifest SHA-256:
  `7a4c88640a78e78c06c2b5880925ddb5b741603d0cc95dc604b55b5099c00ca0`;
  footer SHA-256:
  `834c1f04171eaeb9b067136e1b1b2a14574497c750f25f9797879099397611d0`.
- repair/floor/cap/clamp/jitter counters are all zero.  No numerical
  positivity repair was introduced.

Repository closure was repeated after the persistent flight.  CPU exact tests
passed with one and four OpenMP threads, strict CUDA-host and pedantic C
compiles passed, the production CPU and `sm_86` links passed, header closure is
`31/31`, and `git diff --check` passed.  The rebuilt production fatbin contains
`sm_80`, `sm_86`, and `sm_90` cubins and has local SHA-256
`0bf6cd95140be6c0fd9e87885a031482716b14371c3632d97523b5912ad180fe`.
The production source and dispatch are still not connected to this prototype.

The repeated full D19/K7/Z12/CP4 battery passed.  Its log is
`validation/sh_grid_low_band_exacthyd_canonical_2026-08-09/gate_battery_cmf_multigpu_persistent_envelope_2026-08-10.log`,
SHA-256
`2587726c364a2fdf4083a94aa699164244b0cb36b645b6d4274e05fc3b780272`.
H200 diagnostic `251978` remains pending by priority and full job `251976`
remains user-held; neither staged input was changed.

Failure provenance was retained rather than hidden:

- job `252095` exposed a real angular-boundary ownership error: the first
  tangent sample had incorrectly been paired with zero intensity at `mu=0`.
  The CPU owner holds that first sampled intensity constant to `mu=0`; the
  prototype now does the same. This changed no physical input and added no
  numerical repair.
- job `252097` passed every numerical and memcheck test, but the driver looked
  for CUDA 13's zero-error marker only on stderr while this version writes it
  to stdout. The final driver accepts the marker from either captured stream.

## Allocation model

`scripts/cmf_exact_multigpu_memory_model.py` reproduces the exact `cudaMalloc`
sizes in the prototype. For 50 shells, 2,013,113 bins, and the conservative
47,649-bin maximum positive window:

| devices | direct max/GPU GiB | positive max/GPU GiB | positive total GiB | A10 | A40 |
|---:|---:|---:|---:|:---:|:---:|
| 1 | 103.237 | 104.925 | 104.925 | no | no |
| 2 | 54.281 | 55.877 | 110.222 | no | no |
| 4 | 29.803 | 31.354 | 120.817 | no | yes |
| 8 | 17.564 | 19.092 | 142.007 | yes | yes |

The positive columns include `t1`, `source_cell`, both monoid stacks, the
directed operator, and the componentwise envelope's device application. They
still exclude existing NLTE/transport allocations and production persistent
integration. Aggregate card memory is never treated as one address space.

## Production blockers

1. Measure and improve full-frequency parallelism without changing the sealed
   two-stack grouping. The current one-thread-per-ray sequential walk is a
   correctness implementation, not a production performance claim.
2. Measure the existing full Lumina CUDA peak. If it does not fit alongside the
   shard on one A40, separate its lifetime or independently shard its owner;
   silent oversubscription and Unified-Memory paging are not allowed.
3. Seal reduced-grid and full-grid positive/envelope agreement before adding
   any production dispatch edge or finite CMFGEN comparison.

## Production-shaped reduced-grid split flight

The first production-shaped benchmark keeps the actual toy06 50-shell
geometry, 66 rays, production `dlognu=2.7797007933179339e-6`, total
characteristic drift `47649.254516728804` bins, and maximum single positive
window of 9,108 bins.  Only the frequency slice length is reduced to 8,192
bins.  Its finite opacity, scattering, and fixed-source arrays are synthetic
performance/numerical-contract coefficients.  They are **not** a same-identity
CMFGEN physics fixture and this flight is not a finite CMFGEN comparison.

The original driver ran the independent one-device and four-device solves
serially.  That is useful for an uncontended same-node timing comparison but
wastes three allocated cards throughout the one-device solve.  The benchmark
now also accepts `one` and `four` split modes.  Separate Slurm jobs write
self-describing `J` and componentwise error-upper arrays; an offline checker
then compares every cell against the sum of the two certified bounds.  This
changes neither the CMF operator nor any physical/numerical value.

Sealed split evidence:

- common run root:
  `/gpfs/kjhan/lumina/cmf_multigpu_reduced_split/split_20260810T001144Z_7fa25bf4e3e1`;
  staged binary SHA-256
  `7fa25bf4e3e1763f18c324f1244377c43f6d3b0676732055ec2d2bdf238dc8b7`.
- four-device job `252352`, node `syn07`, state `COMPLETED 0:0`, elapsed
  28:34 and measured solve time `1711.016270583` s.  It used 13 fixed-point
  iterations, one persistent context, seven bound applications, and four
  upper-operator applications.  Per-device peak VRAM deltas were
  `415/421/415/415 MiB`.
- one-device job `252351`, node `syn05`, measured solve time
  `3047.908084386` s with the same `13/1/7/4` iteration/context/operator
  counts and a 755-MiB peak VRAM delta.  The computation and result write
  succeeded, then the Slurm driver exited `70:0` because its inherited VRAM
  summarizer required four traces.  The one-device trace contains 15,234
  valid samples; after correcting the summarizer to accept an explicit
  expected-device count, it reports the 755-MiB peak.  The driver failure is
  retained as provenance rather than relabelled as a successful job.
- the observed cross-node operational speedup is `1.781343717642`.  This is
  not claimed as an uncontended same-node microbenchmark.  Four-device
  sharding reduced the maximum per-card peak by a factor of `1.793349168646`,
  but its aggregate allocation was larger; the result is a per-card capacity
  result, not a total-memory reduction claim.
- the finite four-device `J` range is
  `[8.8332793258264307e-08,2.4965020783600907e-05]`; its error-upper range is
  `[2.9241455256969676e-17,1.5167651242540617e-16]`.  The one-device ranges
  agree to rounding-envelope precision.
- all 409,600 cells passed the independent split comparison.  Maximum
  one/four relative difference is `1.6222759467960672e-15`, maximum consumed
  fraction of the combined componentwise envelope is
  `5.4656493245645533e-05`, and coverage is `409600/409600`.
- one/four result SHA-256 values are respectively
  `37c709e591972631260efe83f710856c39c2c3506d81c6c5663050765b5a49d9`
  and
  `8adde94b741fda31a1febf79b515ddd82f70eb89c3fef993b326e0f00ceb4fa8`.
  The comparison record SHA-256 is
  `50b0810d9ba59432310c9d567e82843102990f660f78acad7feaf7ea35c96d09`.
- every result is finite and nonnegative; numerical repair, physical floor,
  cap, clamp, and jitter counts are all zero.

Performance/failure provenance is also retained.  Job `252345` failed before
the benchmark because `/usr/bin/time` was absent on the compute node.  Job
`252346` showed that the 32,768-bin first one-device solve remained GPU-bound
after 33:07 and was deliberately cancelled before a no-result timeout; its
trace SHA-256 is
`3826712e8818009d21167c64d4638b15408de8bf3a4dc381dcd15e1e7bae4d8f`.
The superseded 8,192-bin serial job `252350` was cancelled at 30:17 after the
split jobs were stable, releasing three idle allocated cards; its trace
SHA-256 is
`4742c30e44d86fa8b45d6d991dd03dd853613ebcf72311ca2209a2c8763d68c5`.
The two-hour one-device insurance job `252360` was cancelled at 6:04 after the
original numerical result passed comparison.

The current kernel maps one complete sequential frequency recurrence to each
ray thread.  Sixty-six rays occupy only three warps on one GPU, so splitting
them across four devices cannot approach fourfold acceleration.  Contiguous
equal-ray-count ranges also have unequal path work: the core-ray shard stays
busy while the short tangent-ray shard becomes idle.  Weighted contiguous
ray boundaries can improve load balance, but production performance still
requires within-ray frequency parallelism that preserves the sealed
subtraction-free two-stack grouping and directed rounding order.

Repository closure after the split flight passed bash/Python syntax, all
three `one/four/both` geometry-only contracts, CUDA compilation,
`git diff --check`, and the complete D19/K7/Z12/CP4 battery.  Serial and
parallel battery result tables were identical.  The log is
`validation/sh_grid_low_band_exacthyd_canonical_2026-08-09/gate_battery_cmf_multigpu_reduced_split_2026-08-10.log`,
SHA-256
`79453c756f21d4c5bc5571176a966db147d5840361103c7fd84b129badc7aa28`.
Production dispatch remains untouched.  H200 diagnostic `251978` remains
pending by priority and full job `251976` remains user-held.

## Weighted contiguous partition and hardware-qualified 8k closure

Equal ray counts were not equal work.  With production geometry the old
boundaries `[0,16,33,49,66]` owned `800/697/392/136` active ray/segment pairs
and computed `849/729/408/136` after the right-halo ray.  The deterministic
cumulative-work partition now uses `[0,10,20,34,66]`; it owns
`500/490/539/496` and computes `550/535/570/496`.  The same boundaries are
used by the direct solve, positive solve, one-shot directed bounds, and the
persistent envelope context.  Reports and the reduced benchmark independently
check the `490..539` owned and `496..570` computed-work contracts.

The implementation also gained fail-closed diagnostics rather than a
numerical repair:

- every positive sweep records the failing device, shard, phase, segment,
  direction, shell/bin, and directed values;
- each device partial must independently satisfy lower <= nearest <= upper
  before the fixed host reduction;
- nearest angular reconstruction uses the same checked positive kernel as the
  directed rounds, with `rounding=0`;
- the non-atomic cross-thread early-exit read was removed, so a failing thread
  cannot leave unrelated partial cells unwritten;
- the former `sqrt(fmax(0,1-p^2/r^2))` numerical clamp was removed.  A negative
  or nonfinite analytic mu-squared now fails explicitly.  No floor, cap,
  clamp, absolute-value replacement, jitter, or positivity repair was added.

The first weighted 8k flight, job `252364` on `syn07`, completed in
`1339.255235898` s.  It had work ranges `490..539/496..570`, peak VRAM
`373/373/399/515 MiB`, and compared with the existing independent one-GPU
result at maximum relative difference `1.4194914534465592e-15`, with all
409,600 cells inside the combined componentwise envelope.  Relative to the
earlier equal-count four-GPU flight (`1711.016270583` s), this is an
operational 1.2776x speedup and 21.73% wall-time reduction.  Its maximum-card
VRAM is 22.33% higher than the old 421-MiB four-GPU maximum because dense
padded storage follows ray count, not active path work; this tradeoff is not
hidden.

A production-size diagnostic then exposed a hardware-dependent missing
partial rather than a round-off error.  At shell 43/bin 768 the affected
device partial was zero although all 25 device-resident ray intensities were
finite and positive and the same buffers recomputed on the host to
`9.8902735445828712e-08`.  Compute-sanitizer job `252382` reproduced the
failure with `ERROR SUMMARY: 0 errors`.  Same-binary jobs `252384`/`252385`
failed on `syn06` and passed on `syn07`.  CUDA device reordering made the
failure follow UUID `GPU-906578dd-9007-fdbd-3c6a-a0c5821e24d6` from logical
device 3 to 0 (`252386`) and then to 2 (`252388`).  A five-card allocation
that excluded that UUID and used replacement UUID
`GPU-53c15d39-350a-5498-2988-a100418300b1` passed (`252390`).  Volatile ECC,
uncorrected aggregate ECC, and pending row-remap counters were zero, so the
UUID is quarantined for this workload rather than explained away by a
software floor or retry.

Final clamp-free evidence is sealed on the known-good `syn07` set:

- final staged benchmark SHA-256:
  `1a0480fe321b89c4036ded02b9809a2363024cd67c030012795b6e3fcd9a7a31`;
- 1024-bin directed diagnostic job `252395`: `COMPLETED 0:0`, maximum
  one/four relative difference `8.5689990180891535e-16`, ordering failures
  zero, and repair/floor/cap/clamp/jitter zero;
- selftest/memcheck job `252394`: `COMPLETED 0:0`, two ordinary runs
  byte-identical, compute-sanitizer zero errors, footer SHA-256
  `7a59ae76d21d4f4fad04ab4ef3d1f99f6816d8efaf415af83bf3535711e021b5`;
- final 8k jobs `252396` and `252397`: both `COMPLETED 0:0`, solve times
  `371.818018868` and `371.813001576` s, identical 13 iterations and `1/7/4`
  context/bound/upper counts, and identical peak VRAM
  `373/373/399/515 MiB`;
- the two 6,553,640-byte result files are byte-identical, SHA-256
  `aa43bb667c8602691ce89f1169ed014a90474d759a48c0f68b364e2eb7e57b9b`;
- the independent one/four comparison passes all 409,600 cells, maximum
  relative difference `1.5503706747130466e-15`, maximum combined-envelope
  consumption `1.0930758301523256e-04`, and numerical repairs zero;
- finite final `J` is
  `[8.8332793258264307e-08,2.4965020783600907e-05]`; the error-upper range is
  `[2.9241481245511831e-17,1.5167598257518997e-16]`.

The final repeat run root is
`/gpfs/kjhan/lumina/cmf_multigpu_reduced_split/split_20260810T024336129806915Z_p1612015_1a0480fe321b`.
The complete D19/K7/Z12/CP4 battery and serial/parallel equivalence passed;
log SHA-256 is
`cb31e365d95a4a09dd7b9b4871116b53e2a37365854e8311ce52f07fc2fd2c5d`.
The final 371.8-s timing is repeatable for this binary/node pair, but the
additional difference from earlier binaries is not attributed solely to the
partition without a controlled same-binary equal/weighted A/B run.

These are finite synthetic-coefficient transport values, not a same-identity
finite CMFGEN comparison.  Production dispatch and the queued H200 inputs
remain unchanged.

Next stages:

1. add a controlled runtime equal/weighted A/B mode to separate partition
   speedup from binary/node effects, while retaining weighted as the current
   correctness candidate;
2. design an exact within-ray two-stack parallel scan that preserves the
   existing binary64 grouping and directed rounding order;
3. prove the scan on the small direct oracle before any production dispatch;
4. collect the pending H200 full-state memory flight and test coexistence with
   the A40 shard state;
5. run the full-grid smoke, then begin same-identity finite CMFGEN comparison.

## Controlled same-binary equal/weighted A/B closure

The partition-only A/B is complete.  The ordinary public envelope API remains
weighted by default; a separate explicit enum entry point selects the
historical equal-ray partition only for controlled experiments.  No hidden
environment dependency was added to the solver core.  The reduced benchmark
maps `CMF_MGPU_REDUCED_PARTITION=equal|weighted` to that explicit API and
checks each mode's exact work contract.

Four full 8k jobs used binary SHA-256
`f9f84912ee5dd84c5cb449d9cca186835a41b57f68e5e4cb215f1ad4759a34eb`
on `syn07`.  Each job allocated the same five UUIDs and computed on the same
first four in the same order.  Jobs `252398/252399` and `252400/252401` were
two sequential equal/weighted pairs:

- equal: `458.240888931/458.235787507 s`, byte-identical result SHA-256
  `5db7dd2f801190e45826f9548abe86c6a5b78d796440f8790f6e766439e439ec`;
- weighted: `371.797192502/371.819500103 s`, byte-identical result SHA-256
  `aa43bb667c8602691ce89f1169ed014a90474d759a48c0f68b364e2eb7e57b9b`;
- mean time is `458.238338219/371.8083463025 s`, so weighted is
  `1.232458450102x` faster and reduces wall time by `18.861362026674%`;
- both A/B comparisons are identical: maximum relative difference
  `8.7441847530776422e-16`, maximum absolute difference
  `1.3552527156068805e-20`, and `409600/409600` cells inside the combined
  error envelope with maximum consumption `1.0755378417557555e-04`;
- equal peak VRAM is `415/421/415/415 MiB`; weighted is
  `373/373/399/515 MiB`.  Weighted wins time but has the larger single-card
  allocation, so no memory-saving claim is made.

All jobs used 13 iterations and `1/7/4` context/bound/upper applications.
Finite `J` remained
`[8.8332793258264307e-08,2.4965020783600907e-05]`; repair, floor, cap, clamp,
and jitter counts are zero.  The complete D19/K7/Z12/CP4 battery passed with
serial/parallel tables identical; log SHA-256 is
`e0c4f0e596db822862d89319873aec84ff773dadc616073c0fb2a24cecc0d319`.
The consolidated ledger is
`validation/sh_grid_low_band_exacthyd_canonical_2026-08-09/cmf_multigpu_partition_ab_final_2026-08-10.log`.

Verdict: retain weighted contiguous segment-work partitioning as the default
candidate.  This remains finite synthetic-coefficient transport evidence,
not a same-identity finite CMFGEN reproduction.  Production dispatch,
coevolution ownership, and the staged H200 jobs were not changed.

Next stages:

1. specify an exact within-ray parallel scan for the two-stack affine monoid,
   preserving the current binary64 grouping and directed rounding order;
2. prove lower/nearest/upper enclosure and repeat byte determinism on a small
   direct-oracle grid before any full-grid optimization;
3. collect H200 diagnostic `251978` CPU MaxRSS/GPU VRAM when scheduled and
   decide whether the A40 shard state can coexist with the full device state;
4. integrate only the proved scan into the prototype and rerun reduced
   CPU/1-GPU/4-GPU gates;
5. run the A40 full-grid smoke, then start a same-identity finite CMFGEN
   comparison with explicitly matched physical quantities.

## Exact within-ray scan design verdict

The design stage is complete in
`docs/CMF_EXACT_WITHIN_RAY_SCAN_SPEC_2026-08-10.md`.  A conventional balanced
prefix scan is rejected: the binary64 reverse-compose primitive is not
associative in nearest, directed-lower, or directed-upper mode, and explicit
hex witnesses differ by one ulp when reparenthesized.  No tolerance can make a
different tree the same sealed discrete operator.

The accepted implementation candidate is canonical two-stack transfer-epoch
replay.  A window of W transforms resets its front stack every W pops.  Each
epoch can independently reconstruct the exact boundary-back, transferred-front,
and new-back serial folds from raw transform values.  Those three chains and
different epochs may run concurrently; no chain itself is tree-reassociated.
After their aggregate pairs are available, the existing per-bin interpolation,
top-cell, aggregate, and half-cell sequence is parallel across bins.

`scripts/verify_cmf_exact_epoch_formula.py` checks the nonassociative witnesses
and compares every aggregate pair from the serial queue and epoch equations.
All 6,588 lower/nearest/upper cases are bit-identical, including W=0/1/2,
frequency/warp boundaries, W around and above n_bins, identity, minimum
subnormal emission, and broad-exponent deterministic inputs.  Script SHA-256
is `3f5601b99cbc7f9a5013e4c4867fa1d0fe8bab58186440f13bef7915d7fa82b2`.

This is structural evidence, not a CUDA implementation or performance result.
The worst reduced-8k segment has only one epoch, while the full production
shape can have 43; reduced speedup therefore cannot be assumed.  Small-grid
C/CUDA aggregate, logical-node, segment, full-sweep, scheduling, and fail-closed
gates must pass before reduced timing.  Production dispatch remains unchanged.
The one narrowly scoped Fable request returned no verdict because the CLI
stopped at its USD budget, and it was not retried.

## Exact transfer-epoch production-prototype closure

The explicit frequency-parallel path is now implemented in the prototype.
The ordinary direct, positive, directed-bound, and envelope APIs retain their
serial behavior.  New explicit entry points accept a schedule containing CUDA
block size, epoch batch cardinality, and the largest window handled by direct
canonical replay.  Large windows use per-shard global front/back workspaces;
no Unified Memory, paging fallback, numerical floor, cap, clamp, or jitter was
introduced.

The isolated G4–G6 gates passed on A40 `syn07`: 3,456 full-sweep J values and
1,152 direct-oracle enclosures (`252411`), 347,328 schedule-invariance values
(`252425`), byte-identical 1/2/4-device canonical-ray reductions (`252426`),
and six transactional failure publications preserved out of six failures
(`252427`).  All sanitizer runs reported zero errors.

Production API job `252432` proved serial/epoch bit identity for the fixed-point
solve, all three directed bound fields, and the persistent componentwise error
envelope.  It also covered blocks 32/64/128/256, multiple batch sizes, both
replay and workspace paths, and fail-closed invalid schedules.  Its binary
SHA-256 is
`122c5a04d42efeaccd991c090c11b8d7e485b51593c415eff9ca71fa653862e1`.

The production-shaped 1,024-bin sanitizer flight `252431` had zero ordering
failures and zero CUDA errors.  The 8,192-bin, 50-shell, four-A40 flight
`252433` completed in `36.502375973 s`, versus `371.813001576 s` for the
weighted serial reference: `10.185994518577x` speedup and
`90.182598290464%` lower wall time.  Both result files are byte-identical with
SHA-256
`aa43bb667c8602691ce89f1169ed014a90474d759a48c0f68b364e2eb7e57b9b`.
Finite J is
`[8.8332793258264307e-08,2.4965020783600907e-05]`.

Peak VRAM remained essentially unchanged: epoch `373/373/397/517 MiB` versus
serial `373/373/399/515 MiB`; this is a speed result, not a memory-saving
claim.  The coefficient fixture is finite but synthetic, so this does not
claim finite CMFGEN physical reproduction.  Production dispatch and the
coevolution owner remain unchanged pending full-state memory, full-grid, and
same-identity CMFGEN gates.  Consolidated ledger SHA-256 is
`337e90efe7dafe516e905d08d3fe672422ca756be2024039d3013eebfbcb96f9`.

## Compact full-grid and production-owner closure

The prototype is now a production-selectable owner.  An unset or zero
`LUMINA_CMF_FINE_MGPU_DEVICES` retains the portable CPU positive-sliding
owner; a positive integer requires exactly that many visible devices.  A
failed requested GPU attempt is terminal.  The production schedule is block
128, epoch batch 64, direct-replay threshold 32, with weighted contiguous
segment-work partitioning.

Full-grid jobs `252438/252443` each solved 50 shells × 2,013,113 bins on four
A40s.  Their result files are byte-identical across staged binaries, SHA-256
`dcda52e5a97cbc92e95522ba92406ad54706354bcbee8fd9511acf70bf0e028c`.
The repeat took `895.507151513 s`; finite synthetic J was
`[5.9347823429553942e-08,9.7311679856602645e-05]`.

The final production binary SHA-256 is
`9549e375aeaf439aace587eb4b02b42051b2cac1c3d8910c46d6e767aea08f8b`.
Owner selftest `252447` passed repeated byte determinism, transactional
negative-input rejection, invalid configuration rejection, and compute-
sanitizer with zero errors.

`LUMINA_CMF_FINE_MGPU_AB=1` copies one assembled initial state into private CPU
buffers, runs both owners, then requires every J/error value to be finite and
nonnegative, maximum relative J disagreement at most `1e-12`, and every point
distance within the sum of the independently directed error radii.  It does
not modify either result and adds no floor, cap, clamp, or jitter.

On the sealed CMFGEN-derived production deck, job `252448` solved 100,655,650
cells in 45 CPU and 45 GPU iterations.  CPU finite J was
`[8.4086208255147163e-82,1.9072381379446642e-4]`; A40×4 finite J was
`[8.408620825514714e-82,1.9072381379446645e-4]`.  Maximum relative difference
was `3.1710829615213259e-15`; maximum distance divided by the combined envelope
was `0.25924739579810846`.  R6 published all 2,180,286 E-lines with zero
partial or unsampled lines.  Host MaxRSS was 84,688,916 KiB and peak VRAM was
`38545/21245/22569/21341 MiB`, including the existing 17-GiB NLTE/BF state on
device 0.

The scoped checker passed before unrelated downstream A2-10 completed.  The
job was cancelled deliberately to release four idle A40s; its Slurm CANCELLED
state is not an exact-owner failure.  This is a same-assembled-state CPU/GPU
result on finite production CMF physical coefficients, not an independent
execution of the external CMFGEN code.  Consolidated ledger SHA-256 is
`50d1811ac41dec475816f7bdf567c3276096d628e4db5916ba21a58085bac0c4`.
