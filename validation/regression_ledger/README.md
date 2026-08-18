# Regression ledger

`ledger.jsonl` is append-only. Run `scripts/regression_ledger.py`; do not sort,
format, compact, deduplicate, or rewrite prior rows. A repeated run path produces
a new row with `recomputed_at` and `prior_measurement_count`.

Metric semantics are frozen by `ledger_schema_version`. A semantic change
requires a new version and applies only to newly appended rows.
