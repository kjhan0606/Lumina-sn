# Photospheric EUV source — kpr5 forensics

Run: `logs/coevolve_consume_a10_kx_kpr5/` (single iter=11, CAP128M=128,000,000 events).
Shells: LUMINA 50-grid; phot=[6,7,8,9] (v=8632–10816 km/s), deep=[0,1,2].
Scripts: `euv_source.py`, `field_partition.py`, `gph_kernel_probe.py` (+ `*_ledger.csv`, `field_partition.csv`).
Coverage: etype 8 (bf-reemit) UNLOGGED → fb *re*-emission via etype-3 path invisible, but the
kpkt-fb (etype 5) recomb-continuum channel IS logged (fb shares are lower bounds). etype 7 logged.

## 0. Which field drives Gph? (field_partition.py)
Gph loop (`lumina_plasma.c:5845–5868, 5901–5924`) reads per bin `J = mc_J if count>0 else cs_J`
with alpha=1 (full mc override). **In the ionizing EUV 300–912 Å every bin is mc-sampled**
(300–450: 77/77 bins mc; 450–912: 133/133 bins mc). cs_J bins exist only <300 Å.
→ **Gph reads mc_J** — the MC packet field. H-A/H-B (about the MC field) is the correct frame.
(The large cs_J EUV values are present but OVERRIDDEN by mc_J and never reach Gph.)

## 1. H-B (leaked deep field) = 0% — FALSIFIED (euv_source.py Part 3)
EUV interactions at the photosphere (line-abs+bf-abs+e-scat, n=1,076,965), classified by the
shell of the packet's most-recent preceding emission:
- **H-A local (src 6–9): 99.5% by E** · H-B deep-leaked (src 0–5): **0.0%** · outer-in: 0.5%.
- band 300–450: local 100.0% / deep 0.0% · band 450–912: local 99.2% / deep 0.0%.
Deep EUV creation IS bright (10.95 vs phot 2.30) but is absorbed en route — it does NOT reach s6–9.
The photospheric EUV field is regenerated **locally**.

## 2. Local creation composition (euv_source.py Part 1, energy-weighted)
EUV<912 creation = 2.296:  **line-emit 59.5%** · **kpkt-fb (recomb continuum) 40.5%** · ff/B(Te) **0.0%**.
Band split (the ionizing edges):
| band | line-emit | kpkt-fb | dominant line ions |
|---|---|---|---|
| 450–912 (excited-lvl edges) | **95.0%** | 5.0% | **S III 81%**, Co IV 14% |
| 300–450 (Fe III GROUND edge 404 Å) | 1.5% | **98.5%** | Fe V 80%, Co III 17%, S III 1% |

## 3. Same-ion test → the kp_emiss global CDF (euv_source.py Part 2)
EUV line-emit paired with governing line-abs:
- **Photosphere: same-ion 17.1% ⇒ CROSS-ion 82.9%** (= k-packet global thermal CDF, kp_emiss).
  Cross-ion emitters: **S III 82.9%**, Co IV 15.4% → the A2 **S III/IGE attractor is ACTIVE**.
- Deep: same-ion **100.0%** (genuine cascade; deep k-packets exit via B(Te), not the line CDF).
k-packet exit mix (Part 4): phot **91% line-CDF** / 8.9% continuum; deep 99.8% continuum (B(Te)+ff).
kpr counters it11: bteq_exits=90.7M (deep W>0.13) vs **cdf_exits=20.8M** (mid+phot W<0.13).
W(s6–9)=0.054–0.034 << 0.13 ⇒ photosphere disqualified from B3 → runs the kp_emiss line CDF.

## 4. Attribution split of the photospheric EUV field
| source | share of EUV creation | drives | repair |
|---|---|---|---|
| **H-A: kp_emiss cross-ion LINE CDF (S III)** | ~59% (95% of 450–912) | excited-lvl Fe III edges | **B3 extension** |
| **fb RECOMBINATION continuum** (3rd local channel) | ~41% (98.5% of 300–450) | GROUND Fe III edge 404 Å | opacity / detailed-balance |
| H-B leaked-deep | **~0%** | — | (outward-opacity fix wasted here) |
| legitimate same-ion cascade | ~17% of lines ≈ 10% | — | — |

Ground Fe III Gph kernel (gph_kernel_probe.py, edge-normalized Kramers): **100% from 300–450**
(the 404 Å edge). Field there is fb-recomb, NOT lines. At 404 Å (s8): LUMINA mc_J=2.8e-8; at 300 Å
LUMINA 3.7e-12 vs **CMFGEN J300=9.5e-15 (≈400× at 300 Å; edge is worse)**.

## 5. CMFGEN contrast
CMFGEN J300 declines **1.42e6** s0→s8 (τ_eff~14): EUV is optically THICK **and** faint at the phot.
LUMINA mc_J(450–912) declines only ~53×; cs_J(450–912) RISES outward. → CMFGEN's photosphere is
faint because it (a) emits far less EUV there AND (b) re-absorbs it en route; LUMINA over-emits
locally AND under-absorbs. (Limitation: CMFGEN jnu extract stops at 918 Å; <912 only at 300 Å.)

## 6. Repair verdict — BOTH, targeting DIFFERENT channels
The -4 B(T_e) exit (`lumina_cuda.cu:3282–3325`) intercepts **only the resonant CDF (line) path**;
-2(ff)/-3(fb) fire BEFORE it. Therefore:
- **Extend B3 to the photosphere** → replaces the **S III cross-ion line CDF** (≈59% of EUV creation,
  the 450–912 excited-level driver) with B(T_e≈12 kK) continuum (EUV Wien-dead ~1e-13). Does NOT
  touch fb. Caveat: T_e(s8)=12.2 kK is +1.9 kK over CMFGEN 10.3 kK (T_rad pinned 10470 K) → residual.
- **fb-recomb / EUV-opacity fix** (NOT B3, NOT H-B) → removes the **fb recombination continuum**
  (≈41%, the 300–450 GROUND-edge driver at 404 Å). This is the dominant GROUND Fe III Gph driver.
  The outward-opacity idea is right physics for the WRONG reason: not leaked-deep (0%), but the
  LOCALLY-made recomb continuum is under-absorbed (contra CMFGEN's optically-thick EUV).

**Neither repair alone suffices.** B3 kills the line channel (~59% of creation, excited-level Gph);
the fb/opacity fix kills the recomb channel (~41%, ground-level Gph). Exact Gph-integral split needs
per-level σ_bf + populations (not reconstructed) — creation-energy shares + the ground-Kramers probe
are the proxies. Evidence points to the **fb-recombination continuum as the larger GROUND-state
driver**, so B3 is necessary but the higher-leverage lever for the 404 Å over-ionization is the
fb detailed-balance / EUV-opacity repair.
