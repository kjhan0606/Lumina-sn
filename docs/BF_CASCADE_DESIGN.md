# bf recombination-cascade channel for the macro-atom (ARTIS-faithful, 2026-06-29)

Goal: fill DDC15 gold's smooth plateaus ③ 6500-7200 / ⑤ 8800-10200 that our bb-only
macro-atom misses (sharp peaks at 7055/9290 instead). Recombination cascade = high ion
recombines → lower-ion excited level → cascades down emitting a spread of lines.

## Key architectural finding (makes increment 1 free)
- LUMINA macro-atom carries a **global level index** `activation_level_id`; the walker
  (`lumina_cuda.cu:2388`) does NO ion check. So a cross-ion INTERNALDOWNLOWER is just a
  transition whose `destination_level_id` is in the lower ion → **no GPU state-machine
  change needed for increment 1**. Only RADRECOMB (continuum emit, increment 2) needs a
  new opcode (-4).
- bf-absorbed packets activate at the **upper-ion ground** (`ionized_ground`), which has no
  downward line → today a dead-end (resonance scatter). Recomb is the only physical
  downward exit → why bb-only can't make the plateaus.

## ARTIS formulas (macroatom.cc:139-163, 240-279, 467-494, 608-627; ratecoeff.cc:51-58,145-163,496-542)
For upper-ion level i, summed over lower-ion target levels j (with bf edge):
- `R = n_e · α_sp(i→j, T_e)`   (Milne spontaneous recomb coeff)
- **INTERNALDOWNLOWER** weight = `R · ε_target(j)`     (neutral-ground energy of j)  → cascade, no photon
- **RADRECOMB**        weight = `R · (ε_i − ε_j)`       (= hν of bf photon)            → continuum emit
- α_sp = 4π·SAHACONST·(g_j/g_i)·T_e^−3/2·∫(2/c²)σ_bf,j(ν)ν²exp(−h(ν−ν_edge)/kT_e)dν ; ν_edge=(ε_i−ε_j)/h
- RADRECOMB photon: ν sampled ∝ σ_bf(ν)ν³exp(−hν/kT_e), ν≥ν_edge (reuse our fb sampler).
- n_upper cancels in the CDF (only relative R matters). g_i = g(ionized_ground), NOT 2·U_ion.

## LUMINA mapping (all data present)
- neutral-ground ε: `level_energy_eV[l] + accum_ip_eV[Z*8+stage]` (plasma.c:1376-1396)
- Milne α_sp: `frozenin_alpha_rr` kernel (plasma.c:2116-2138), evaluate per-target-level
- n_e: `plasma->n_electron[s]`; upper level i = `ionized_ground[ip]`; targets j = lower-ion
  levels with `cmfgen_has_sigma[j]`; edges as in compute_bf_opacity (plasma.c:2783-2786)
- global index: `level_offset[ip]` (lumina.h:247)

## Increments (gate LUMINA_MACROATOM_BF; 0=baseline byte-identical)
- **Inc 0 plumbing**: parallel recomb arrays in Opacity (recomb_block_refs[n_lev+1],
  recomb_dest_level[], recomb_nu_edge[], recomb_prob[n_recomb*n_shells], recomb_is_emit[]).
  Build topology once at load; device setter `cuda_set_recomb(...)` (pattern cuda_set_kpacket).
  Gate off → NULL → skip.
- **Inc 1 (BF=1) INTERNALDOWNLOWER cascade (no photon)**: host adds w_down=R·ε_j into the SAME
  sum_rates_total; GPU walker continues partial-sum into recomb block, on hit sets
  activation_level_id=recomb_dest_level[k]; current_type=0 (loop continues in lower ion).
- **Inc 2 (BF=2) RADRECOMB continuum**: host adds w_emit=R·hν_ij; GPU opcode -4 → fb sampler
  ν=nu_edge−(kTe/h)lnξ, isotropic, next_line by binary search.

## Caveats (no guess-patching)
1. Radiative-only recomb (k-packet off) — leave COLRECOMB off unless LUMINA_KPACKET=1
   (detailed balance: radiative recomb pairs with radiative photoion).
2. Neutral-ground energy reference mandatory (energy conservation; ε_i−ε_j=true bf photon E).
3. g_upper = g(ionized_ground); n_e once; n_upper cancels.
4. Start with i=ionized_ground only (ARTIS maxrecombininglevel subset); extend later if under-filled.
5. recomb weights enter the SAME normalization as the loaded block.

## Validate: DDC15 emergent vs gold at ③ 6500-7200 / ⑤ 8800-10200; MA-fate cross-ion deactivations.
A/B: BF=0 → BF=1 (cascade) → BF=2 (+continuum). Inc1 alone should move the needle.
