# ARTIS parity gap — complete difference audit (6-subsystem whitebox)

**Date:** 2026-07-21. Purpose: the complete list of Lumina-vs-ARTIS differences in the NLTE ionization/emission physics, to implement ALL AT ONCE (per user directive: stop incremental channel-matching), then run + debug against ARTIS.

**Config being matched:** CMFGEN/ARTIS toy06 @19.48d. **ARTIS comparison baseline:** `artis-ref/tests/toy06_nlte_bk/` (**[사례 18 정정 2026-07-31] 19.48 d를 포함하는 timestep = 27**(start 19.42/mid 20.2549 d); 구 표기 "timestep 20 = 19.4945 d"는 **adata.txt:1479의 C II 준위 20 에너지 19.4945 eV를 timestep-일수로 이식한 오염**이었음(ts20 실제 mid=11.2353 d) — 정본 감사 validation/.../artis_ts_contamination_audit/REPORT.md; `nlte_*.out` = per-level n_LTE/n_NLTE → ARTIS b_k; `estimators_*.out` = per-cell J/Te/TR/W/ionfrac). ⚠️경고 상환 완료: ts27 재기저 결과 매칭-epoch ARTIS는 **sub-thermal**(Si II L1-4 평균 0.27·S II 0.80) — "super-thermal Si II ~18/S II ~48"은 ts11(~5.27d) 초기상 전용 인용만 허용.

**KEY META-FINDING:** a large fraction of these gaps already exist as **gated Lumina code paths, OFF by default** (METACOLL, COLL_FIX, IUP_JBLUE, IDOWN_BETA, IDOWN_COLL, EWEIGHT, NEUTRAL_E, KPACKET, KPACKET_EXIT, FEIII_COLDATA, BINNED_J, MACROATOM_BF…). We only ever tested them one at a time. "Implement all at once" ≈ **(1) enable the full ARTIS-consistent gate-set together + make it default**, plus **(2) add the ~5 genuinely-MISSING pieces** (collisional ionization/recombination, element-wide NLTE coupling, per-bin field fit, bf MC estimator, ion-changing macro-atom).

Classification: **[MISSING]** = new physics; **[APPROX]** = replace formula/data; **[GATE]** = exists but off by default; **[EXTRA]** = Lumina-only patch to disable for ARTIS-faithful; **[DONE]** = already matched; **[DEFER]** = upgrade later.

---

## A. COLLISIONAL DRAIN NETWORK — the b_k-trap root (why Fe III excited/metastable levels over-populate)

**THE CRUX** (NLTE-solve audit §D): ARTIS force-connects the first `NLEVELS_REQUIRETRANSITIONS` levels to *every* higher level with inserted `A=0, coll_str=-2, forbidden` transitions (input.cc:481-486), so every metastable has a collisional de-excitation channel (Axelrod floor `nne·8.629e-6·0.01·g_lo/√Te`, macroatom.cc:721-723) + real CMFGEN Ω where data exists. **Lumina builds bb collisions ONLY where a radiative line exists** (plasma.c:10372) → a drainless metastable (level 17, 25…) gets ZERO radiative AND ZERO collisional drain → pins at the b_k cap.

- **A1 [MISSING/GATE]** bb collisional connectivity for drainless levels. Lumina `LUMINA_NLTE_METASTABLE_COLL` mode 2 (plasma.c:10750-10797) replicates it but is OFF; make it the default network (couple every metastable to all lower levels).
- **A2 [APPROX, default-on, severe]** van Regemorter **Bethe (H_ionpot/ΔE)² factor DROPPED** in Lumina default (plasma.c:10583) → collisional excitation **1–2 orders too LOW** for low-ΔE lines. ARTIS: full vR + energy-dependent Gaunt Γ=max(0.2, 0.276·e^u(−γ−ln u)) (macroatom.cc:757). `LUMINA_NLTE_COLL_FIX` restores the (Ry/ΔE)² but is OFF and still fixed Γ=0.2. → make ARTIS-form default.
- **A3 [MISSING]** real close-coupling Ω for **Co III, Ni III, Fe II, S III, …** — Lumina proxy-only; real Ω exists only for Fe III (Zhang, `LUMINA_FEIII_COLDATA` OFF). ARTIS reads per-transition `coll_str` for all. Add a `coll_str` data channel; **dispatch on the data flag, not on f_lu** (a forbidden line with f~1e-8 is mis-routed to vR→0). (Co being proxy-only ⇔ the campaign's "Co ~10× rate-deficient".)
- **A4 [MISSING — entire channels]** thermal **collisional ionization** (Seaton, macroatom.cc:662-682) + **3-body recombination** (macroatom.cc:630-658) — ABSENT everywhere in Lumina. ARTIS has both in the NLTE matrix (nltepop.cc:590,605).
- **A5 [APPROX]** forbidden floor: Lumina flat Ω=1 (plasma.c:4605) vs ARTIS g-scaled 0.01·g_lo·g_up (macroatom.cc:723).
- **A6 [APPROX]** Lumina has **3 inconsistent** vR/Axelrod implementations (matrix, RADEQ cooling, k-packet) with different constants (gbar 0.2 vs 1.0, (Ry/ΔE)¹ vs ²). ARTIS uses ONE pair everywhere. → unify.

## B. NLTE SOLVE STRUCTURE

- **B1 [APPROX]** coupling topology: ARTIS = ONE SE matrix per **element** (all ions simultaneously, ionization is the solved output; nltepop.cc:1225). Lumina = independent **adjacent-ion-pair** matrices + 5-iter 50%-damped outer loop (plasma.c:11687) → Fe III split across (II,III)&(III,IV); cross-stage cascade broken.
- **B2 [APPROX]** closure: ARTIS pure rate-SE, `FORCE_SAHA_ION_BALANCE=false` (ion fractions solved). Lumina pins pair-total to **nebular (Lucy-Mazzali) Saha** `ion_number_density` at T_rad,W (plasma.c:11657, 631-790) → absolute ionization is Saha-supplied; matrix only reshuffles the split. **Primary structural ionization difference.**
- **B3 [APPROX]** excitation/partition temperature: ARTIS T_e/T_J (USE_TJ=false→T_e). Lumina partition fns + non-metastable Boltzmann at **T_rad,W dilute** (plasma.c:494-525). Known TARDIS gap, still default.
- **B4 [APPROX]** bf routing: ARTIS photoion→specific phixs target level, recomb from that level (level-resolved cascade). Lumina all photoion→single ground_hi, recomb←ground_hi (plasma.c:11024) → excited-level cascade collapsed.

## C. RADIATION FIELD (the photoionizing field)

- **C1 [APPROX — biggest field gap]** ARTIS fits a **per-bin (W, T_R)** dilute-BB (radfield.cc:735-815); rates evaluate `radfield(nu)=W_bin·planck(nu,T_R_bin)`. Lumina fits **ONE whole-shell dilute-BB** (Wien T_rad, one W; plasma.c:110-138) — no per-bin nu_bar accumulated (cuda.cu:2837), so it structurally cannot carry super-thermal EUV + sub-thermal optical at once. → add per-bin nu_bar + per-bin W/T_R fit.
- **C2 [MISSING]** detailed **bf MC rate estimator** (path-integrated σ_bf·dist·e/nu; radfield.cc:194-222,855) — the actual sampled photoion rate. Lumina has NO analog; photoion is nebular-Saha or Milne-at-T_e proxy. → add a transport-accumulated bf-rate estimator.
- **C3 [MISSING/config]** the binned MC field IS accumulated (cuda.cu:2845) but **discarded** in champion — `cmfgen_write_jnu` overwrites nlte.J_nu with deterministic cs_J (cuda.cu:5885); MC field only via gated co-evolve alpha. → wire the sampled MC field into photoion+excitation rates.
- **C4 [APPROX/GATE]** per-line Jb_lu MC estimator exists (cuda.cu:3132) but gated (IUP_JBLUE) + ≥10-count fallback to dilute-BB.

## D. MACRO-ATOM / CASCADE / LINE EMISSION

- **D1 [MISSING — structural]** Lumina macro-atom is **bound-bound ONLY** (transition_type ∈ {-1,0,1}); ARTIS macro-atom **changes ion stage** (INTERNALUPHIGHER photoion/collion, INTERNALDOWNLOWER/RADRECOMB/COLRECOMB) — it IS the ionization engine. Lumina does ionization in a *separate* offline solver → cascade decoupled from ionization; no recomb-cascade fluorescence. (`LUMINA_MACROATOM_BF` is a partial gated bolt-on.) + **radiative-recombination continuum emission MISSING** (macroatom.cc:240-278).
- **D2 [APPROX/GATE]** internal-up = thermal-anchored **binned-J** `B_lu·J_line` (b_k→1, kills UV fluorescence); internal-down = `A_ul·(1−β)` (wrong form, trapping double-count); **no energy (ε) weighting**. ARTIS: `(B_lu−B_ul n_u/n_l)·β·J` up, `(R+C)·ε` down, intrinsic neutral-ground ε weighting. Gates: IUP_JBLUE/IUP_BETA, IDOWN_BETA/IDOWN_COLL, EWEIGHT/NEUTRAL_E — all OFF.
- **D3 [MISSING]** macro-atom activation: Lumina base activates via **bound-bound line absorption only**; ARTIS also activates via **bf absorption** (nu_edge/nu split) and **k-packet collexc/collion**. (bf→MA only under MACROATOM_BF gate.)
- **D4 [EXTRA→disable]** Lumina thermal line-source overrides with NO ARTIS analog: BSRC/BSRC_PHOT B(Te), EPS_UV/EPS_IR Planck(T_rad) redistribution, LTHERM, deterministic S_l for the cs_J solver, MA_CAP_EMIT. ARTIS line source is macro-atom-emergent only. → disable for ARTIS-faithful (the "S_line ≁ B fluorescence funnel" degradations).
- **D5 [MISSING]** continuum **free-free heating** r→k channel (rpkt.cc:402,886) — absent from Lumina chi_continuum (cuda.cu:4451; ff only as a k-packet exit).
- **D6 [GATED REPAIR]** `LUMINA_FIX_BF_CONTINUUM_EVENT=1` restores the
  per-continuum opacity draw and `nu_edge/nu` ionization/kinetic split;
  `LUMINA_FIX_BF_STIM_RECOMB=1` independently restores the corrfactor.
  Full MA-vs-k behavior requires `LUMINA_KPACKET=1`; with the event gate OFF
  Lumina retains legacy argmax level-map activation.

## E. K-PACKET (overlaps A)

- **E1 [MISSING]** collisional-ionization cooling channel in k-packet (ARTIS activates MA at upper ion; kpkt.cc:527). Lumina denominator = ff+fb+collexc only → collion weight redistributed. + collisional-recombination→k-packet entry missing.
- **E2 [APPROX]** same col-rate approximations as A2 in the k-packet weights + p_kpacket.
- **E3 [EXTRA→disable]** B(Te)(-4), FB_OTS, one-shot/single-exit, COLLISION-LIMIT — no ARTIS analog.
- **E4 [DONE]** ff formula + ν sampling, fb Milne cooling+emission, collexc upper-level target distribution — MATCHED.

## F. DEFERRED (user: upgrade after Lumina works)

- Non-thermal Spencer-Fano ionization+excitation, MC gamma, positron deposition. NEGLIGIBLE at the toy06 @19.48d photosphere (radiation-dominated). Gamma *heating* present; ARTIS-comparison injects ARTIS deposition directly.

---

## Debug instrumentation to build WITH the implementation (user requirement)
Event-log subsystem tagging (which channel produced each emission/absorption: k-packet ff/fb/collexc/collion, macro-atom RAD_DEEXC/COL_DEEXC/RAD_RECOMB, bf-abs, …) + per-subsystem diagnostic dumps (channel census, macro-atom fate histogram, NLTE rate-contribution decomposition, per-level b_k, per-bin mc_J/B_ν). Compare each Lumina subsystem census to ARTIS's `toy06_nlte_bk` estimators/nlte/packets → **event-query battery that names the wrong subsystem.**

## Implementation grouping (for the all-at-once build)
1. **Collisional network (A1-A6)** — default ARTIS-form vR+Bethe+Γ, coll_str data channel (all IGE) dispatched on data-flag, collisional ionization+3-body recomb, forbidden g-scaled floor, unify the 3 impls, metastable full-connect.
2. **NLTE structure (B1-B4)** — element-wide coupling, rate-SE closure, T_e partition, phixs-resolved bf.
3. **Field (C1-C4)** — per-bin (W,T_R) fit, bf MC estimator, wire sampled MC field into rates.
4. **Macro-atom/emission (D1-D6, E1-E3)** — ion-changing macro-atom + recomb emission, ARTIS internal-up/down + ε weighting, bf/collisional activation, disable thermal knobs, ff-heat, k-packet collion.
5. **Diagnostics** — event-log subsystem tags + per-subsystem census.
