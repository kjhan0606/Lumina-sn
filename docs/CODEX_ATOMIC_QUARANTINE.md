# 32-ion atomic quarantine and bidirectional gate

This delivery prepares, but does not run, the CPU deck bake for
`data/tardis_reference_toy06_19p48d_sivcaiv_active/`.  The four predecessor
decks remain immutable.

## Layout and manifest

The generated deck has explicit, non-recursive active inputs at its root:

```text
tardis_reference_toy06_19p48d_sivcaiv_active/
  active_ions.csv
  levels.csv, line_list.csv, ...
  quarantine/
    DO_NOT_LOAD                       mode 000 sentinel
    manifest.json                    canonical machine manifest
    manifest.csv                     human-readable 32-ion index
    source_deck_snapshot/            byte-preserving source-deck snapshot
```

`manifest.json` records, for every quarantined ion, `(Z, ion0)`, spectroscopic
name, exclusive `(a)/(b)/(c)` classification, reason code, composition/link/data
facts, source evidence hash, FL/SL count, abundance range and nonzero-shell
count, prior physical activity, restore requirements, and archive location.
The seal adds SHA-256, byte counts, and CSV row counts for every active root
file and SHA-256/byte counts for every archived source file.

The pre-registration is part of the canonical JSON: class `(c)` was physical
contamination, so removing positive-abundance C/O/Mg/Al/Sc/Ti/V/Cr/Mn opacity
must make the ejecta more transparent by that contribution.  An opposite
direction falsifies this mechanism.  The immutable regression ledger must be
compared before and after the operator's model run; these scripts never write
`validation/regression_ledger/` or `scripts/regression_ledger.py`.

## Loader contract

`scripts/atomic_quarantine_contract.py` is fail-closed:

1. active files are named explicitly and must be immediate children of the
   deck root;
2. recursive globbing is not an input mechanism;
3. every attempted path containing `quarantine` raises
   `[ATOMIC-ACTIVE-SET-LEAK]` before the operating-system open;
4. the verifier compares the inventory reconstructed from active root files,
   not merely `active_ions.csv`;
5. the mode-000 sentinel makes accidental traversal fail visibly even for a
   consumer that does not import the Python contract.

The C runtime already opens fixed root paths such as `line_list.csv`,
`levels.csv`, and `ionization_energies.csv` rather than recursively walking the
deck.  `src/` is intentionally unchanged in this delivery.  The sealed hashes,
row counts, active-set checks, HDF5 group inventory, derived-array cardinality,
and sentinel detect a partial or leaked view before a model run is accepted.

## Bidirectional identity

`scripts/verify_atomic_quarantine_identity.py` enforces:

```text
MODEL_SPEC ions = atomic_links F_OSCDAT ions = loaded active ions
loaded active ions intersect quarantined ions = empty
loaded active ions union quarantined ions = archived original 59 ions
```

Levels are compared in both directions over each `MODEL_SPEC` NF, including
rank, configuration, `g`, energy representation, metastable flag, and the
linked `f_to_s` prefix.  All 27 prefix N_SL counts must equal `MODEL_SPEC`.
Lines use a multiplicity-preserving `(Z, ion0, lower, upper, occurrence)`
identity and compare `f_lu`, `A_ul`, and wavelength with exact float equality.
Every mismatch is written to CSV outside the deck; there is no tolerance or
sampling.  `config.json:n_lines` must equal the actual `line_list.csv` row
count.  Ion-bearing CSV rows and HDF5 groups must all be active.

The batch first runs the unchanged R4 verifier (which itself retains R1) on the
immutable `_ftos` source and an ephemeral gate-OFF control.  It then applies
the new NF-scoped R4 membership and bidirectional gates to the active view.

## Operator command

When node-local scratch is available:

```bash
sbatch --export=ALL,REPO_ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn \
  scripts/sbatch_deck_atomic_quarantine.sh
```

Without `SLURM_TMPDIR`, provide the explicit escape hatch:

```bash
sbatch --export=ALL,REPO_ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn,R4_OFF_DIR=/gpfs/kjhan/lumina_runner2/work/r4_off_atomic_quarantine \
  scripts/sbatch_deck_atomic_quarantine.sh
```

Slurm stdout/stderr goes to `/gpfs/kjhan/lumina_runner2/slurm/`.  The identity
verifier is the last pipeline and `set -o pipefail` makes its exit status the
job result.

## Restore

Restoration is promotion into another new deck, never mutation of this deck:

1. require a new fixed CMFGEN run whose `MODEL_SPEC` and `atomic_links.txt`
   include the requested `(Z, ion0)` and whose composition includes its element;
2. verify every `quarantine/source_deck_snapshot` SHA-256 against
   `manifest.json`;
3. copy the snapshot to a new, absent staging deck and promote all associated
   ion material together (levels, lines, sigma, collision data, ionization
   boundary, macro-atom/reference mappings, offsets, and NPY arrays);
4. regenerate global identifiers and every derived sidecar from that new CMFGEN
   target set; regenerate `active_ions.csv` from its MODEL_SPEC/links;
5. append a `restored` event to the new manifest (do not delete quarantine
   history), seal it, and run the full R1/R4, bidirectional, and leak gates;
6. activate only a passing new deck.

With the current CMFGEN target set, restoring any of the 32 ions intentionally
fails `FAIL_EXTRA_ION`; that failure is the restoration safety interlock.

## UNRESOLVED

- The full run was not submitted by this delivery, so no new deck, full gate
  verdict, GPU result, or before/after regression-ledger comparison exists yet.
- Exact source-float round-trip for line `f_lu`, `A_ul`, and wavelength may
  expose the existing decimal serialization/finalizer re-derivation issue.  The
  gate reports it without tolerance or repair.
- The inherited abundance table has 30 shell columns while `config.json` says
  50 shells.  This remains explicitly unresolved; this delivery does not invent
  the missing 20 values or change the existing loader behavior.
- Sigma every point and collision-strength every item are sealed and restricted
  to active ions here, but a semantic CMFGEN-value comparator for all sigma and
  Upsilon values remains the separate R5 scope in `docs/ATOMIC_EQUIV_PLAN.md`.
- GPU allocation remains based on full-level offsets in `src/`; logical active
  N becomes 240, but allocation reduction is explicitly outside this delivery.
