# parity59 chain replay

Read-only replay copies for capture `instr_capture_188932`.  Every executable
accepts input paths; none writes into the capture, the historical analysis
directories, `logs/`, or `src/`.

Executed lightweight copies:

- `trapping_audit/replay_trapping.py`: historical Audit U and electron-scattering Audit T definitions.
- `reddening_localization/replay_reddening.py`: historical Task A band/local-concentration definitions.
- `radeq_ledger_audit/replay_radeq_observables.py`: direct capture observables and solver-root history.

The exact invocations and source/field definitions are recorded in
`docs/CODEX_CHAIN_REPLAY.md`.  `baseline_0715_results/` was produced by the
same parameterized code against the historical run, while `results/` uses the
capture.  This is the side-by-side definition gate.

Intentionally not executed:

- expansion/FUV line opacity: `lumina_line.csv` is absent, so the requested
  branch is `UNRESOLVED`; no line-list/level-pop substitute is used.
- event forensics: the parameterized compute-node copy is added separately and
  requires a full pass over the 8 GB `lumina_events.bin`:
  `reddening_localization/taskB_event_forensics_compute_node.py`.
- exact parity59 production-solver counterfactual roots remain unavailable
  because the July estimator is not the current solver (`DB_FB=1`,
  `BF_RATE_POPS=1`) and its trial tables were not dumped.  The requested July
  estimator itself is now baseline-gated and applied to the parity59 state in
  `radeq_ledger_audit/radeq_coupledroot.py`; its own/CMFGEN roots and R/J/O
  additivity cube are persistent in `baseline_0715_results/` and `results/`.
