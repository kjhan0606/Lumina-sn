# Gate-B Phase-1.6 Lane C comparison (REPORT-ONLY)

No threshold or gate verdict is implemented here.

## Speed mapping

| shell | Lumina km/s | CMFGEN depth | CMFGEN km/s | delta | in range |
|---:|---:|---:|---:|---:|:---:|
| s0 | 4264.000 | 67 | 4394.182 | +130.182 | True |
| s8 | 10088.000 | 54 | 10163.506 | +75.506 | True |
| s43 | 35568.000 | 10 | 35497.710 | -70.290 | True |

The outer cell is s43, the outermost recorded Lumina cell whose centre remains inside the CMFGEN RVTJ velocity range.

## Census

- `compared`: 99
- `context_only_nonidentical`: 9
- `lumina_unavailable`: 161
- `unavailable`: 313

Strict identical coverage: **99/582 = 17.01%**. Including explicitly nonidentical numeric context: **108/582 = 18.56%**.

`coverage_disposition.csv` accounts for every non-compared row. An unknown or blank unavailability reason aborts generation.

## Parser and unit evidence

- RVTJ n_e is identity-transcribed. `cmfgen_source_evidence.csv` records the RVTJ header writer, the following `WRITE ... ED`, and the `ED(:) !Electron density (#/cm^3)` declaration.
- PRRR requires exactly MODEL_SPEC N_SL rows in every 10-depth chunk and all ND chunks. Alpha is source-certified by `TOTRR=TOTRR/ED/DHYD`.
- Ion OUT requires exactly source-declared NLEV coefficients for all ND depth blocks. GENCOOL requires all depths and reads its explicit volumetric-rate headers.
- `cmfgen_snapshot_consistency.csv` records RVTJ-versus-PRRR n_e; a mismatch is disclosed and never silently treated as one snapshot.
- Every unavailable comparison remains in `oracle_vs_cmfgen.csv` with its reason.
