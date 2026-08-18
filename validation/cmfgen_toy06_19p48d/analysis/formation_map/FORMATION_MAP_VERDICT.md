# Per-shell SPECTRUM-FORMATION map: CMFGEN vs Lumina (toy06 19.48d)

CMFGEN-divergence framing. Source: self-run `cmf_flux.exe` on the converged jnu4 snapshot
(`/gpfs/kjhan/cmfgen_runs/toy06_19.48d_cmfflux/`, ETA_DATA/CHI_DATA) vs the Lumina B-run
corpse event log (`logs/coevolve_consume_a10_kx_gphall/lumina_events.bin`).

## Method (both sides CMF / comoving-frame band edges)
- **CMFGEN**: radial contribution function CF(band,depth) = Sigma_{nu in band}
  eta_nu(r) exp(-tau_nu(r->out)) r^2 dr, normalized per band. eta_nu,chi_nu = total CMF
  emissivity/opacity (depth x 460668 CMF freqs); tau_nu = radial optical depth to the
  outer boundary (trapezoidal, chi per 10^10 cm). ND=90, v 1025..35975 km/s.
- **Lumina**: for every escaping packet, the shell of its LAST photon-creating emission
  (etype 2/4/5) recovered by vectorized grouped forward-fill over pkt_id; escape energy
  binned by (band via comoving escape lambda, emission shell). 50 shells, v_mid 4264..39935.
- **Caveats**: (1) CMFGEN CF is a RADIAL (p=0) escape proxy, not the full (p,z) observer
  integral -- it locates the FORMATION REGION, not the exact emergent flux (its band
  budget is redder than the true OBSFLUX SED, see below). (2) Lumina event log = one
  converged iteration (iter 11), cap-truncated to 134673 of ~200000 escapes (unbiased in
  time -> fractional CF robust). (3) eta is TOTAL emissivity (incl. e-scattering source) =
  where a photon last acquired its frequency, the analogue of Lumina 'last emission'.

## Band energy budgets (fraction of that side's total)
| band A | CMFGEN radial-CF | CMFGEN OBSFLUX (true emergent, observer) | Lumina event-log (comoving) | Lumina kromer (observer) |
|---|---|---|---|---|
| 300-450   | 0.0% | 0.0%  | 0.2% | 0.1% |
| 450-918   | 0.3% | 1.2%  | 1.8% | 2.7% |
| 918-1290  | 1.4% | 2.1%  | 5.6% | 7.3% |
| 1290-2000 | 21.6%| **72.0%** | 9.6% | 7.4% |
| 2000-4500 | 55.1%| 17.8% | 36.1%| 38.9%|
| 4500-7000 | 15.0%| 5.8%  | 22.7%| 30.3%|
| 7000+     | 3.7% | 1.0%  | 24.1%| 12.9%|

**The emergent-SED divergence itself:** CMFGEN's true emergent flux is **72% UV (1290-2000)**;
Lumina's is **7.4%**. Lumina is drastically redder -- this is the "too-red" defect, and the
formation map below shows WHY.

## Median formation velocity per band -- CMFGEN vs Lumina (km/s)
| band | CMFGEN | Lumina | ratio L/C | reading |
|---|---|---|---|---|
| 300-450   | 32997 | 33384 | 1.01 | both far-outer thin-layer escape (negligible flux) |
| 450-918   | 35975 | 31928 | 0.89 | both outer |
| **918-1290 (FUV)** | **10706** | **31928** | **2.98** | CMFGEN=photosphere, Lumina=far outer |
| **1490-1650 complex** | **11256** | **31928** | **2.84** | CMFGEN=photosphere, Lumina=far outer |
| 1290-2000 | 11815 | 22464 | 1.90 | CMFGEN photosphere, Lumina outer |
| 2000-4500 | 8572  | 11544 | 1.35 | Lumina ~1.35x further out |
| 4500-7000 | 7435  | 10816 | 1.45 | Lumina ~1.45x further out |
| 7000+     | 7983  | 8632  | 1.08 | converge (NIR) |

## Divergence matrix (band x velocity zone, CMFGEN% vs Lumina%, C-L), key rows
```
band        vzone      CMFGEN%  Lumina%   C-L
918-1290    6-8k        20.7     0.0     +20.7   <- CMFGEN forms FUV at/below photosphere
918-1290    8-10k       12.9     0.0     +12.9
918-1290    10-12k      25.1     0.0     +25.1
918-1290    >30k        14.4    82.9     -68.5   <- Lumina forms FUV in far-outer ejecta
1490-1650   10-12k      63.5     4.5     +58.9
1490-1650   >30k         0.0    68.1     -68.1
1290-2000   10-12k      61.0     3.9     +57.1
1290-2000   12-30k+      21.5    64.3    (Lumina spread to fast layers)
2000-4500   6-8k        31.4     0.0     +31.4
2000-4500   8-10k       24.9     0.6     +24.3
2000-4500   10-12k      30.9    60.3     -29.4   <- Lumina optical piles up further out
4500-7000   6-8k        57.0     1.1     +55.8
4500-7000   10-12k       7.8    74.8     -67.0
7000+       6-8k        34.0    16.7     +17.3
7000+       <3.9k(BC)    3.4    15.1     -11.6   <- Lumina NIR leans on inner-BC photosphere
```
Full table: `formation_divergence_matrix.csv`.

## Highlights (the answers requested)
1. **Where CMFGEN forms FUV vs where Lumina forms it.** CMFGEN forms 918-1290 A at the
   PHOTOSPHERE (median v=10706 km/s; 59% within 6-12k km/s) and lets it ESCAPE (UV = 72%
   of its emergent SED). Lumina forms 918-1290 A in the FAR-OUTER fast ejecta (median
   v=31928 km/s; 83% at v>30k) -- i.e. its photospheric UV is trapped/reprocessed and only
   the thin outer P-Cygni line emission escapes as UV (UV = 7.4% of Lumina's SED). The two
   sides form the FUV in **disjoint** velocity regions (>68 pts of the >30k bin swing).
2. **1490-1650 complex.** CMFGEN: photosphere, median 11256 km/s (63% at 10-12k). Lumina:
   far outer, median 31928 km/s (68% at >30k). Same disjoint pattern as the FUV.
3. **Red/NIR split.** CMFGEN forms the optical DEEP (2000-4500 at v~8572; 4500-7000 at
   v~7435 -- 57% at 6-8k). Lumina forms the SAME optical ~1.35-1.45x further out (2000-4500
   at 11544; 4500-7000 at 10816 -- 75% at 10-12k). NIR (7000+) converges (v~8000-8600,
   ratio 1.08), with Lumina leaning ~15% on the inner-BC/photosphere vs CMFGEN's 3%.

## Physical reading (CMFGEN-divergence)
CMFGEN's UV forms at, and escapes from, the photosphere. Lumina cannot release
photospheric UV: those photons are reprocessed and only re-emerge as line emission in the
fast outer ejecta (and as redder photons), so Lumina forms UV ~3x too far out AND forms
its optical/red ~1.4x too far out. This is the per-shell fingerprint of the
fluorescence/reddening defect -- a formation-region shift outward + UV->red reprocessing,
not merely a normalization error.

## Driver placeholder numbers (docs/KP_EMISS_REPAIR_DESIGN.md)
- **#1 -- DEEP (v=4394 km/s, s0-equiv; T=18536 K, n_e=4.85e9) continuum share over 300-2000 A:**
  emissivity is **~99.7% LINE, ~0.3% continuum (ff+bf)** (robust 0.1-1.5% across
  percentile/window; e-scattering negligible; floor elevated by line blanketing so 0.3% is
  an UPPER bound). k-packet continuum branch calibration: **continuum(ff+bf)/(cont+line) ~ 0.3%**
  in the deep UV. By sub-band: 300-450 0.5%, 918-1290 0.9%, 1290-1490 2.5%, 1490-1650 0.45%.
  File: `cmfgen_deep_continuum_fraction.csv`.
- **#2 -- FUV (918-1290 A) formation depth (CMFGEN):** mean v=14857, **median v=10706 km/s**
  (68.6% within the forming shells 3900-11908 km/s; 30.9% outer). Lumina median = 31928 km/s.

## Artifacts (this dir)
`cmfgen_CF_band_depth.csv`, `cmfgen_formation_velocity.csv`, `cmfgen_deep_continuum_fraction.csv`,
`lumina_CF_band_shell.csv`, `lumina_formation_velocity.csv`, `lumina_kromer_band_budget_observer.csv`,
`formation_divergence_matrix.csv`, `formation_velocity_compare.csv` + the 4 generating scripts.
CMF_FLUX run + README: `/gpfs/kjhan/cmfgen_runs/toy06_19.48d_cmfflux/`.
