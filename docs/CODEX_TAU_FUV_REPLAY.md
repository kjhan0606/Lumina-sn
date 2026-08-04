# `tau_FUV` chain replay — capture 188932

## Outcome

The baseline reproduction gate **passed**.  Only after that pass, the same
parameterized audit was run against capture `instr_capture_188932`.

At s0, `tau_FUV` changed from **69.787 to 177.624** (2.545x), while the already
known bolometric energy-density ratio changed from `u/CMFGEN=0.576` to 2.518.
Thus `tau_FUV` moved **with**, not against, the energy-density increase.  The
capture is not optically under-trapped: its FUV outward depth is even larger than
the 07-15 value.  The trapping audit's conclusion that “insufficient optical
trapping” is refuted therefore **still holds, more strongly**, for this capture.

No model was run and no GPU was used.  This is an offline replay of the two
existing `lumina_plasma_state.csv` and `lumina_levelpop.csv` captures with the
two existing static inputs.  No source under `src/` and neither the historical
`audit_t_expop.py` nor `VERDICT.md` was modified.

## Baseline reproduction gate

Parameterized replay:
`validation/chain_replay_parity59/trapping_audit/audit_t_expop.py`.
Persistent baseline output:
`validation/chain_replay_parity59/trapping_audit/baseline_0715_results/tau_lumina_line.csv`.

| shell | quantity | existing `audit_T_optical_depth.csv` | replay | gate |
|---|---|---:|---:|---|
| s0 | `tau_FUV` | 69.79 | 69.79 | PASS |
| s0 | `tau_es` | 1.800 | 1.800 | PASS |
| s0 | `tau_Ross` | 5.828 | 5.828 | PASS |
| s1 | `tau_FUV` | 59.64 | 59.64 | PASS |
| s1 | `tau_es` | 1.439 | 1.439 | PASS |
| s1 | `tau_Ross` | 4.978 | 4.978 | PASS |

The stronger machine check compared all six numeric output fields
(`tau_Ross_out`, `tau_FUV_out`, `tau_es_out`, `kap_Ross`, `kap_FUV`, `kap_es`)
for all seven s0-s6 rows against the preserved historical
`trapping_audit/tau_lumina_line.csv`: **42/42 values matched exactly; maximum
absolute difference = 0.0**.

## 07-15 versus capture 188932

Values are outward optical depths.  `x` is capture divided by 07-15.

| shell | FUV 07-15 | FUV 188932 | x | Ross 07-15 | Ross 188932 | x | es 07-15 | es 188932 | x |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| s0 | 69.787 | 177.624 | 2.545 | 5.828 | 33.182 | 5.693 | 1.800 | 1.554 | 0.863 |
| s1 | 59.640 | 146.138 | 2.450 | 4.978 | 19.114 | 3.840 | 1.439 | 1.177 | 0.818 |
| s2 | 51.248 | 110.244 | 2.151 | 4.352 | 8.283 | 1.903 | 1.144 | 0.939 | 0.820 |
| s3 | 44.004 | 84.269 | 1.915 | 3.875 | 5.110 | 1.319 | 0.911 | 0.767 | 0.842 |
| s4 | 37.518 | 61.450 | 1.638 | 3.483 | 3.351 | 0.962 | 0.728 | 0.635 | 0.873 |
| s5 | 30.566 | 39.281 | 1.285 | 3.092 | 2.137 | 0.691 | 0.582 | 0.530 | 0.912 |
| s6 | 17.678 | 18.649 | 1.055 | 1.998 | 1.163 | 0.582 | 0.464 | 0.439 | 0.948 |

Classification:

- **`tau_FUV`: 변화(증가), trapping 판정 유지, 역전 없음.**  It increased at
  every audited shell.  At s0 the change is +154.5%, and both values are far
  above unity.  There is no CMFGEN FUV optical-depth column, so no FUV L/CMFGEN
  ratio or crossing is claimed.
- **`tau_Ross`: 변화(프로파일 재편), CMFGEN 대비 판정 유지, 역전 없음.**  It
  increased in s0-s3 and decreased in s4-s6.  At s0 it increased by 469.3%.
  Relative to the unchanged CMFGEN Ross depth at s0, the ratio is
  1.427 -> **8.122**; capture remains on the more-opaque side throughout s0-s6.
- **`tau_es`: 변화(감소), CMFGEN 대비 판정 유지, 역전 없음.**  It decreased
  throughout s0-s6; at s0 the change is -13.7%.  The s0 L/CMFGEN ratio is the
  already replayed 1.182 -> **1.020**, so electron scattering moved close to
  agreement without crossing below CMFGEN.

The s0 behavior is consequently not “more stored radiation because FUV photons
can now escape more easily.”  `u/CMFGEN` reversed from 0.576 to 2.518 while
`tau_FUV` rose by 2.545x and `tau_Ross` rose by 5.693x.  Electron scattering alone
became better matched, but line expansion opacity dominates the FUV result.  This
two-snapshot replay establishes the direction and rules out an optical-depth
shortage; it does not by itself assign causality for the energy-density
overcharge.

## Definitions and provenance

Every result row in both `tau_lumina_line.csv` files contains its source paths,
source fields, level-hit fractions, zero-population line counts, and the literal
calculation definitions.

- Geometry: `data/tardis_reference_toy06_19p48d/geometry.csv`, fields
  `shell_id`, `r_inner`, `r_outer`; `dr=r_outer-r_inner`.
- 07-15 plasma and populations:
  `logs/coevolve_consume_a10_kx_gphall/lumina_plasma_state.csv` field `n_e`, and
  `lumina_levelpop.csv` fields `shell,Z,ion,level_num,g,n_k`.
- Capture plasma and populations:
  `/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/` with the same two
  filenames and fields.
- Static line data:
  `data/tardis_reference_cmfgen_superlev_ionfix_ddc15strat/line_list.csv`, fields
  `atomic_number,ion_number,level_number_lower,level_number_upper,f_lu,wavelength,nu`.
- `tau_es_out`: sum of `n_e sigma_T dr` from the current shell to the surface.
- Sobolev depth: `2.6540281e-2 f_lu lambda_cm t_exp n_lower stim`, with
  `t_exp=19.48 d`.  Matching Lumina's defined zero-population branch, `stim=1`
  unless both populations are positive; when both are positive,
  `stim=max(1-g_l n_u/(g_u n_l),0)`.  The replay does **not** apply Lumina's
  production `tau>=1e-100` floor.
- `kap_FUV`: one 918-1290 Angstrom rest-wavelength expansion-opacity bin,
  `sum[(nu/dnu_FUV)(1-exp(-tau_S))]/(c t_exp) + n_e sigma_T`.
- `kap_Ross`: 2000 log-frequency bins over 1.5e14-3.0e16 Hz, harmonic
  `dB_nu/dT` mean at 13120 K of line expansion opacity plus `n_e sigma_T`.
- `tau_FUV_out` and `tau_Ross_out`: line-inclusive opacity summed through s6,
  plus electron scattering only from s7 to the surface.  This is the historical
  audit definition, not a claim that outer-shell line opacity is zero.
- Contextual `u` values come from the already existing chain-replay files
  `baseline_0715_results/audit_U_energy_density.csv` and
  `results/audit_U_energy_density.csv`, fields `u_cmfgen_full`, `u_lumina_mc`,
  and `mc_over_cmfgen`; they were not recomputed here.
- CMFGEN Ross/es comparison values come from the preserved
  `validation/cmfgen_toy06_19p48d/analysis/trapping_audit/audit_T_optical_depth.csv`,
  fields `CMFGEN_TauRoss` and `CMFGEN_Tau_es`.

The captured population dump contains exact zeros (838,446 of 1,051,900 rows;
the 07-15 dump has none).  The historical vector expression initially exposes
these as an artificial `0/0`.  The replay copy uses Lumina's actual conditional
definition above, under which a zero lower population gives zero line depth.
No population floor, result cap, or capture-specific fallback was introduced.
Re-running the baseline after this source-faithful zero branch still gave the
42/42 exact match reported by the gate.

## Remaining UNRESOLVED

1. The level-population dump matches 93.946% of lower and upper line-list keys.
   The parameterized replay preserves the historical audit's documented
   missing-level convention (`n=1e-30`, `g=1`) solely to keep the audited metric
   identical.  A full-population value for the unmatched 6.054% cannot be
   reconstructed from these captures.  As recorded in the historical verdict,
   the omitted population-bearing opacity could add opacity; it does not supply
   evidence for an opacity shortage.
2. No CMFGEN `tau_FUV` value exists in the preserved audit, so only the absolute
   optically-thick statement (`tau_FUV >> 1`) and the 07-15/capture change are
   resolved.
3. Line opacity beyond s6 is not available under the historical definition.
   Ross/FUV depths therefore retain its “line s0-s6 plus es to surface” scope.
4. These two captures establish correlation, not a causal decomposition of why
   `u` became overcharged.  The narrower trapping-shortage falsifier is resolved.
