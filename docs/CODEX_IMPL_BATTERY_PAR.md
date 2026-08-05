# Gate battery case parallelization

## Scope and result

This change only alters the script-side D, K, Z-INERT, and CONFIG-PREC gate
orchestration. It does not change `src/`, canonical decks, `/gpfs` data, or any
gate predicate. There are 36 independently scheduled result rows:

| battery | cases | retained verdict contract |
|---|---:|---|
| D | 18 defects/warnings + 1 canonical control | existing marker counts, stdout/stderr placement, child rc tests, summary rc |
| K | 1 positive + 6 negatives | the established driver rule: rc 0 is `OK`, nonzero is `FATAL`, 7/7 required |
| Z-INERT | 6 invocations | positive cases rc 0; injected phantom must be nonzero |
| CONFIG-PREC | 4 negatives | child rc exactly 1 and the case-specific marker required |

Every runner emits one `PROGRESS ...` line when each case completes, then emits
its human table in canonical case order. A path-free `RESULT ...` row accompanies
each table row and is the serial/parallel identity authority. A runner returns 0
only under its pre-existing all-pass condition; setup errors and failed builds
remain nonzero.

## Changed runners

- `scripts/run_composition_d_gate.py`: D's 19 subprocesses use the shared case
  pool; the existing checks were moved unchanged into a worker. `--serial` and
  per-case `cwd`/`TMPDIR` isolation were added.
- `scripts/run_config_prec_negative_controls.py`: the existing four mutations,
  exact rc=1 test, and marker tests now run in isolated case directories through
  the pool. `--serial` is retained.
- `scripts/run_k_gate.py`: repository-owned form of the former
  `~/.lumina_scratch/run_dbuild_gates.sh` K section. It preserves the one
  positive and six negative fixtures and expected `OK`/`FATAL` classification.
- `scripts/run_zinert_selftest.py`: schedules the six independent Z-INERT
  invocations and retains the phantom-negative nonzero contract.
- `scripts/run_zinert_selftest.sh`: still provides the legacy one-command entry;
  its four builds are concurrent and it delegates case execution to the Python
  runner. `bash scripts/run_zinert_selftest.sh --serial` is the fallback.
- `scripts/run_gate_battery.py`: integrated D+K+Z+CP build/run command, built-in
  tee, global `--serial`, and `--verify-equivalence` acceptance mode.
- `scripts/gate_parallel.py`: common `ProcessPoolExecutor` adapter. Parallel
  `max_workers` is exactly `os.cpu_count()` (with only the Python-required
  fallback of 1 if it returns `None`); serial mode executes directly.
- `scripts/generate_composition_d_fixtures.py`: content-addressed fixture cache.

All subprocesses receive a case-unique working directory and `TMPDIR`. The K and
CONFIG-PREC mutations are materialized only below that directory. No canonical
deck file is opened for writing.

## D fixture cache binding

The cache key is SHA-256 over a canonical JSON binding containing:

1. the resolved base-deck identity;
2. SHA-256 of the four deck NPY inputs
   `line2macro_level_upper.npy`, `tau_sobolev.npy`,
   `transition_probabilities.npy`, and `zeta_data.npy`;
3. SHA-256 of `generate_composition_d_fixtures.py` itself.

The four large files are streamed in 8 MiB blocks and hashed concurrently. The
cache target is `/tmp/lumina-composition-d-fixture-cache/<binding-sha256>`.
Materialization occurs in a temporary sibling and is atomically renamed. A hit
is accepted only if the manifest has the same binding key, case count, and all
18 case directories. A bad entry is replaced; a different input or generator
hash naturally selects a new directory and regenerates once.

`/tmp` was selected instead of `/gpfs` scratch because these fixture trees are
node-local, short-lived metadata/symlink overlays, and the requested reuse is on
the same execution node/session. This avoids repeating GPFS directory creation
and large-file copies. It intentionally does not provide cross-node or
post-reboot reuse. The content hashes still read the four authoritative files;
the cache never writes them.

The legacy explicit-output interface remains available:

```bash
python3 scripts/generate_composition_d_fixtures.py \
  --base data/tardis_reference_toy06_19p48d \
  --output /tmp/one_off_d_fixtures
```

The integrated runner always uses the hash-bound cache interface.

## Integrated scheduling and tee contract

The integrated critical path is structurally:

```text
max(7 independent harness builds, D cache hash/materialization)
    + max(all 36 independent case executions)
```

After the build/cache barrier, D, K, Z, and CP runners start together. Each
runner uses an `os.cpu_count()` pool, but Python creates at most one live worker
per submitted case; the whole battery has only 36 cases on the 128-logical-core
lageunha node.

Preferred logging uses the built-in tee, which cannot mask the runner rc:

```bash
python3 scripts/run_gate_battery.py --log /tmp/lumina_gate_battery.log
```

If an external shell tee is required, `pipefail` plus `PIPESTATUS[0]` is the
contract:

```bash
set -o pipefail
python3 scripts/run_gate_battery.py 2>&1 | tee /tmp/lumina_gate_battery.log
gate_rc=${PIPESTATUS[0]}
```

Reading plain `$?` after `tee` is prohibited because it is tee's status.

## Serial/parallel identity gate

`--verify-equivalence` builds once, obtains one bound fixture tree, then runs:

1. a globally serial pass (the four batteries in D/K/Z/CP order and every case
   with `--serial`);
2. a parallel pass against the same source tree, binaries, and fixture cache;
3. a byte-for-byte comparison of all 36 canonical `RESULT` rows, including
   verdict and child rc.

Acceptance requires both aggregate runner rc values to be 0 and the tables to
be identical. Success ends with:

```text
EQUIVALENCE serial_rc=0 parallel_rc=0 table=IDENTICAL
```

`--serial` without the verification flag remains the operational fallback.

## Lageunha commands and expected results

Run only from the lageunha driver/compute seat, not a login node:

```bash
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
export OMP_NUM_THREADS=128
python3 scripts/run_gate_battery.py \
  --verify-equivalence \
  --log /tmp/lumina_gate_battery_equivalence.log
echo "$?"
```

Expected rc is 0 and the last identity line is the one shown above. Based on the
reported 25--30 minute serial battery, the acceptance command is expected to
take roughly 30--40 minutes because it deliberately includes that serial pass.
This must be measured on lageunha.

After acceptance, the ordinary A-2 reusable command is:

```bash
python3 scripts/run_gate_battery.py \
  --log /tmp/lumina_gate_battery_parallel.log
echo "$?"
```

Expected rc is 0. Its scheduling target is the longest build/cache-hash phase
plus the longest single case, rather than the sum of 36 cases. A provisional
operational expectation is 3--8 minutes cold and less with a warm `/tmp` fixture
cache; only the lageunha measurement should be recorded as the final wall time.

Fallback:

```bash
python3 scripts/run_gate_battery.py --serial \
  --log /tmp/lumina_gate_battery_serial.log
```

## Verification performed here

Only login-safe checks were performed in this workspace:

- Python byte compilation for all changed/new Python runners;
- `bash -n scripts/run_zinert_selftest.sh`;
- CLI parser/help smoke tests;
- a small-file cache smoke test proving atomic miss then hit with all 18 case
  directories (the four production NPY reads were substituted only in this
  smoke test);
- a four-process CONFIG-PREC executor/scratch/progress smoke test with
  `/bin/false` (all four verdicts intentionally failed because the real markers
  were absent; this was not counted as a physics-gate run);
- `git diff --check` on the scoped files.

The CPU builds, large deck hashes/loads, full tables, wall clock, and serial vs
parallel demonstration were intentionally not executed on the login node. They
are the lageunha acceptance command above.

## Remaining risks

- Concurrent D/K full-deck reads trade elapsed time for memory and GPFS read
  pressure. The case count (36) is below the 128 logical CPUs, but peak memory
  and storage bandwidth must be observed on the first lageunha run.
- The K stale-sentinel case writes one tau-shaped file below `/tmp`; free space
  must cover it for both passes of equivalence mode.
- The cache binding deliberately follows the requested four NPY inputs plus the
  generator. Other deck entries are symlinked live. A change in base-directory
  membership that is not accompanied by one of those five hash changes requires
  manual removal of the node-local cache directory.
- `/tmp` reuse is node-local. A scheduler migration, reboot, or cleanup causes a
  safe cache miss, not a verdict change.
- No commit or push was made. The user-approved commit remains deferred until
  the lageunha equivalence gate and wall-clock review are complete.
