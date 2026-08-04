# B1 — Element-wide NLTE statistical-equilibrium matrix owns the ionization balance

**Date:** 2026-07-23 · **Branch:** thenmc-macroatom-fluorescence · **Status:** DESIGN (no source edited)
**Principle:** exact-physics — give the ion split to the self-consistent SE matrix (CMFGEN's architecture), not a two-bucket nebular closure or a tuned cap.
**Read first:** logs/coevolve_consume_parity8/analysis/{dig_S3_gph_repair,dig_B2_cmfgen_levelpop_judge,dig_B3_floorm_arithmetic}/ verdicts.

## 0. What this repairs
Committed ion owner = radeq_simul_all's nebular ladder (lumina_plasma.c:6549+; compute_ion_populations early-return :1400-1401 under RADEQ_SIMUL=1). It writes all ion stages and derives n_e (:7487-7502). The NLTE matrix runs AFTER (cuda.cu:6489->6493->6504) and its writeback RESCALES each ion back to the ladder's totals (cuda.cu:1480-1507; CPU mirror plasma.c:13225-13257; armed via LUMINA_NLTE_PER_ION_RESCALE=1). The matrix's internal split (sum_lo/sum_hi, cuda.cu:1488-1490) is computed then thrown away. dig_S3/B3 proved the ladder closure r=Gph/(alpha*n_e) cannot reach CMFGEN's S II/III from any (field x channel x population) dial: ground-Kramers x4.8 under, all-level nebular cell x8.7-13 over; target in an unbracketed gap. B1 stops discarding sum_hi/sum_lo for gated elements and closes n_e around it.

## 1. Mechanism — the inversion
Current: ladder -> ion_number_density -> n_e; NLTE solve -> writeback rescale per ion (split discarded).
B1 (gated Z only): ladder PASSES THROUGH persisted ion densities for masked Z (still folded into n_e); NLTE writeback takes the COMBINED-conservation branch (preserves sum_hi/sum_lo) and commits scaled sums to ion_number_density with damp + per-iter |dlog| cap; n_e updates next outer iter. Matrix unchanged; off-mask byte-identical.

## 2. CRITICAL pre-registration risk — two CMFGEN "truths" disagree on S
| source | S s8 charge | s10 | s12 | trend |
|---|---|---|---|---|
| reference_ionization_compare.csv (established parity target) | 1.839 (r~5.2) | 1.198 | 1.156 | drops with depth |
| dig_B2 POP snapshot (FIX_T it40) | 1.975 (r~39) | 1.986 | — | rises |

Not reconcilable by interpolation — qualitatively opposite. Design uses 1.839 as ion-balance judge; dig_B2 POP valid ONLY for within-ion b_k shapes (normalized inside each ion). OPEN QUESTION (blocking judgment): confirm provenance of 1.839 and whether it40 FIX_T POP is pre-ionization-convergence. This discrepancy is falsifier #2.

## 3. Scope ladder
### Stage 0 — instrument only (zero risk, rides any run)
sum_lo/sum_hi already accumulated at cuda.cu:1488-1490; compute for every pair above the gpu_lock_mode branch and dump per (iter, shell, pair): lumina_nlte_ionsplit.csv: iter,shell,Z,ion_lo,ion_hi,sum_lo_preresc,sum_hi_preresc,r_solve,committed_lo,committed_hi,r_committed. CPU mirror plasma.c:13234-13236. Gate LUMINA_NLTE_IONSPLIT_DUMP=1; banner "[IONSPLIT] dump armed". This is the missing observable + Stage-1 pre-flight.

### Stage 1 — S-first pilot (LUMINA_NLTE_ION_OWNER=16)
Gate: Z-list/bitmask parsed like init_twocomp_lock (:763-788); helper nlte_ion_owner_masked(Z). Banner "[ION-OWNER] zmask=... NLTE SE owns II/III split, ladder pass-through".
- Edit A — writeback (cuda.cu:1480-1507; CPU :13225-13257): masked pair -> combined-conservation scaling (preserves split), then commit n_lo=scale*sum_lo, n_hi=scale*sum_hi via owner_commit() to ion_number_density. n_total from nlte_pair_total_density (Edit D).
- Edit B — ladder pass-through (plasma.c:7487-7502): masked ip -> mixed=prev (persist NLTE-committed; drop nebular sh.nion); else unchanged blend. n_e still sums S.
- Edit C — owner_commit(atom,ip,s,n_new): damp LUMINA_NLTE_ION_OWNER_DAMP (default 0.5) + hard cap LUMINA_NLTE_ION_OWNER_DLOGCAP (default 0.3 dex/iter).
- Edit D — pair total (plasma.c:821-844): auto-arm NO_ML_LOCK mass-conservation path for masked Z. Limitation: folds S I + S IV+ mass into pair (s8 f(SI)+f(SIV)~4e-5, negligible).
Banner per iter: "[ION-OWNER s8 Z=16] r_solve=... -> committed=... n_e d=+X%".

### Stage 2 — extension order by comb-cleanliness (dig_B2 comb_statistics, s8 b_k max): Si II 1.0 < Si III 15.8 < S II 9.96 < S III 360 << Fe III 5.2e5.
- 2a Si (Z=14): lowest risk (Si II comb-free; CMFGEN s8 ~2.014).
- 2b IGE: GATED on comb fix. Analysis: the garbage lives in the SOLVED VECTOR, not the rate matrix — R_bf/R_rec/C_ion/C_rec entries are population-independent rate coefficients (nlte_gemm.cu:432-443; plasma.c:12198-12296, :12342-12348). The comb is the LU solution of an ill-conditioned system at low n_e. Committing IGE split from a comb-poisoned solve is unsafe; blanket caps rejected. 2b OFF until ill-conditioning resolved at source.

## 4. Consistency loop + runaway guard
Inner ce_iter holds n_e/T_e/J fixed (plasma.c:13515-13527); feedback period = one outer co-evolve iter. ion_damp no longer damps masked S (pass-through) -> owner_commit supplies analog damp 0.5 + 0.3 dex/iter cap. Dominant n_e loop is self-limiting (more S III -> higher n_e -> more recomb -> pushes back). Guard banner "[ION-OWNER] iter N: max|dlog n_S|=... damped=...".

## 5. Why the matrix can land near 5.2 when the offline all-level cell gave 49
dig_S3's 49 = all-level ionization / GROUND-only frozen-in alpha (two-bucket, comb populations as inputs). The SE matrix balances all-level photoionization against ALL-LEVEL Milne recombination (R_rec into every S II level) with self-consistently solved populations. The recombination the ladder under-counts is what pulls r down. HONEST LIMIT: split cannot be predicted offline without solving the matrix -> the offline pair replica (5.1) is the pre-registration instrument.
S IV limitation (verified plasma.c:5021-5044): S has only ion 1,2 in the NLTE set; no III->IV drain. s8 f(SIV)=4.3e-5 negligible -> judge at s8; bias grows hotter/deeper. S IV addition is a Stage-2+ prerequisite for hot shells.

### 5.1 Offline replica spec (pre-registration instrument; FIRST GATE before any live edit)
Extend analysis/replica_core.py into a full (S II, S III) pair SE assembler. Inputs: parity17 state CSVs; field variants (a) parity11diag jnu_fine J_pub (railed), (b) DEGEN_FALLBACK-corrected raw, (c) cmfgen_jtable truth; sigma_bf bin; level_multiplicity. Assemble mirroring plasma.c:11292-12360 (bb radiative/collisional real-Omega, R_bf = sum sigma*(4pi/hnu)*J per level, R_rec = n_star_ratio*I_rec Milne, C_ion/C_rec Seaton+3-body, continuum coupling to ground_hi; parity17 LTE_FLOOR cuda.cu:1411-1435). LU-solve; output dig_B1_replica.csv: shell {8,10,12} x field {a,b,c}: r, charge, per-channel current decomposition, +-band from T_e +-500K / n_e +-10% jitter. These solved r ARE the Stage-1 pre-registration.

## 6. Field dependence
raw/truth in S band (dig_B3): 1.29x (s8), 0.14x (s10), 0.00 (s12). (a) railed fit -> ~52x hot -> over-ionize; (b) DEGEN_FALLBACK -> good at s8, cold at s10, dead at s12; (c) truth. Judge PRIMARILY at s8; s10/s12 = observations. Pre-register field-(b) column as judge, (c) as physics ceiling.

## 7. Judgment design
Pre-registration = replica solved r (s8, field b, +-band). Judge: committed S charge vs 1.839, pass charge in [1.80,1.85] (r in [4,6]). Secondary: dig_B2 levelpop judge for S II/III within-ion b_k shapes. Non-regression: Fe f(IV) in [0.018,0.026], Ni in [2.02,2.03], T_e roots within parity17 +-100K.
n_e coupling COMPUTED: S = 21.9% of n_e at s8 (1.493e8/6.815e8); moving S to r=5.2 -> +7.2% n_e -> Fe/Ni nudge less ionized (both stay in/toward band) + T_e perturbation. Guards are live falsifiers; n_e is the single coupling channel.

## 8. Rejected alternatives
FLOORM/BKMAX cap (dig_B3: no S leverage, per-element-incompatible knob, 78-87% of IGE Gph = capped plateau). Bounded intermediate ladder channel (re-parameterizes the two-bucket closure; user no-patch). Stage 2b hazards: comb-poisoned IGE solve; T2 lev17 trap biases bb pumping (though ~0% of Fe Gph).

## 9. Falsifiers
1. Replica at s8 field-(c) returns r far from [4,6] -> B1 premise wrong. RUN REPLICA FIRST.
2. CMFGEN S-truth discrepancy (Sec.2) unresolved -> judge ill-posed.
3. n_e feedback pushes Fe/Ni/T_e out of guard bands -> locality claim fails.
4. Live deep-EUV field dead at s8 (like s10/s12) -> field problem, not owner problem.
5. S II solve ill-conditioned at s8 (wild/oscillatory r_solve in Stage-0 dump) -> committed split is artifact.

## 10. Rollback
All edits behind LUMINA_NLTE_ION_OWNER (unset = empty mask = byte-identical to parity17). Stage-0 dump behind LUMINA_NLTE_IONSPLIT_DUMP. DAMP/DLOGCAP read only inside owner_commit. Rollback = drop env vars.
