# Fable audit: A2-INIT seed-material predictor

Date: 2026-08-16  
Mode: read-only architecture and implementation audit

## Verdict

`APPROVE`

## Architecture

Fable confirmed the intended causal order:

1. The initialization pass publishes R1 at radiation generation 1.
2. It publishes gamma once and invokes the fixed-seed-temperature material
   predictor exactly once.
3. The predictor commits P1 to P2 without publishing an A2-10 temperature or
   ledger and the initialization pass skips R7/A2-10.
4. The next pass recomputes radiation from P2 and publishes R2 at generation 2.
5. Only then does the ordinary R7/A2-10 T2/P3 transaction run.

The generation boundary `R=1, T_e=1, atom P=1, NLTE P=0` and the independent
seed-commit preflight make re-entry fail closed after P1 to P2.

## Code findings

No correctness defects were found. Fable specifically verified:

- The rejected `LUMINA_A210_PRECORE_TAU_REFRESH` diagnostic cannot enter the
  production initialization path.
- The material-only commit does not write the public T_e array, opacity T_e
  mirror, T_e generation, or T_e publication, and does not fabricate A2-10
  ledger/counter ownership.
- Candidate material and temperature arrays are private copies and pointer
  detachment is rechecked before commit.
- The targeted gate requires exactly two R6 publications, generations 1 and 2,
  exactly one predictor strictly between them, then R7 and the physics snapshot,
  with all numerical-repair fields zero.
- The seed commit selftest and its fail-closed controls are wired into the
  Makefile battery.

Two non-blocking observations were recorded: a private build-bundle failure does
not repeat the post-build public T_e manifest check (the build owns private
copies only), and enabling the rejected pre-core environment variable now
terminates production initialization by design.

## Required changes

`NONE`

## GPU gate

`READY`

