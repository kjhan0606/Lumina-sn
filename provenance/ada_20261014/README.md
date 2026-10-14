# R2 on lageunha RTX 5000 Ada — 2026-10-14

The user directed GPU work to lageunha's Ada GPU. This branch starts at
9996b4a (the preserved coupled-Newton production snapshot). The sole source
change is an explicit compile-time allocation variant in the exact CMF
transport: `LUMINA_CMF_EXACT_MANAGED_MEMORY=1` replaces its cudaMalloc calls
with cudaMallocManaged, requiring managedMemory and concurrentManagedAccess.
The default remains cudaMalloc. Kernels, equations, grids, tolerances, and
failure behavior are unchanged. No prefetch, memory hints, or runtime env knob
was added. This is a one-GPU portability/cost experiment, not physics approval.

The original two-GPU run allocated 79,094,633,748 bytes in exact transport
(40,356,205,424 bytes on the larger shard). The Ada card has 32,760 MiB; its
host has 1 TiB RAM. Managed-memory oversubscription is needed. Physical
VRAM residency is not the allocation counter printed by the solver.

Evidence root:
`/gpfs/kjhan/lumina/cmfgen_map/sprint_20261012/ada_20261014`

- `managed_probe.log`: a 35,941,842,944-byte allocation exceeding the reported
  33,794,359,296 device bytes was written/read by kernels, zero mismatches.
- `build1/bench_{default,managed}.log`: existing reduced production-geometry
  test, 50 shells, 8192 bins, mode=one, persistent epoch=1; both PASS.
  J and error arrays in `result_{default,managed}.bin` compare byte-for-byte.
  Both FNV64 values are 62893f78bb1068ff; numerical repairs=0.
- `build1/parity.txt`: ALLOCATION_VARIANT_BYTE_PARITY_PASS.
- `build1/build.rc`: 0. NVCC13.0.2, GNU13.2.0, -O2 -arch=sm_89.
- `build1/source.sha256` and `source_verify_after.log`: all189 manifest
  entries unchanged during build. The recipe explicitly includes
  a210_coupled_newton.c, unlike the generic Makefile target.
- `build1/binary.sha256`: lumina_cuda SHA256
  6bff25d9336ac4232826e4867439ab2d1724a69483b809674717f8566ba02fba.

The build ran under the finite remote user unit
`lumina-ada-build-20261014-b.service` (117s, success). The first setup attempt
failed during module initialization before creating the build directory;
the runner now explicitly sets MODULEPATH and disables nounset around Lmod.
This was not a production-model attempt.

Run root: `ada_20261014/run1`; effective requested settings: `exports.env`.
Relative to staged R2, only the output path and GPU count (2 to1) change.
FREE_T, six outer iterations, deposition source0, and all grid/tolerance
settings are retained. Automatic retries are forbidden. H100 job422361 and
its dependent readout422362 were canceled with elapsed0 before this move.

`launch_r2_ada.sh` uses a live SSH session and bounded systemd user service,
without enabling linger or creating a resident launcher. Runtime86280s plus
120s termination allowance is at most24h; MemoryMax192G, CPUQuota1600%, one
GPU. The runner refuses a busy GPU, takes a cooperative flock, verifies the
build/source/deck, refuses reused output, and stops if first-iteration
manifest completion is later than21600s after model start. Runtime limits
were verified in a live resource-check unit before production.

Ada FP64 throughput and page migration can make this much slower than H100.
Reduced-byte parity does not establish full-grid speed, Newton convergence,
or bitwise equivalence between one- and two-GPU summation order. A successful
run requires process rc0, the single whole-run COMPLETE marker, six R7
material commits and iter0005 manifest; scientific acceptance additionally
requires reviewing residuals, replay/generation consistency and observables.
PURE-CMFGEN bypasses Monte-Carlo transport, so this does not demonstrate the
final Monte-Carlo spectrum goal.

Provisional readout can be run on lageunha using
`tools/sprint_readout_remote.py --run <run1> --label R2_Ada --out <new-path>`.
Its only behavioral change from the prior reader permits the authorized
compute host; absence of a spectrum is still not a successful physics result.
No automatic readout service is attached to this attempt.
