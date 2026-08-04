# Codex A Wave 2.1 C-review repair

Date: 2026-07-31

Scope: direct repair of the FAIL coordinates in
`docs/CODEX_WAVE2_C_REVIEW.md`. No git command or GPU kernel run was used.

## Repairs

- CPU and GPU continuum selectors now construct both the total bound-free
  opacity and the route CDF at the packet's actual event `nu_cmf`.
  Kramers cross sections and the stimulated-recombination exponential use that
  frequency directly. CMFGEN's baked per-bin-average cross sections are
  interpolated in log frequency instead of selecting a bin-centre value.
- A route is eligible only when `nu_cmf >= nu_edge`. Therefore an edge above
  the event frequency cannot be selected, and the `p_ion > 1` concealment
  clamps were removed from both CPU and GPU split sites in the same change.
- CPU transport recomputes the event Doppler factor after packet movement,
  matching the GPU endpoint coordinate. The CPU kinetic-pool branch also checks
  the actual `LUMINA_KPACKET` pool state rather than treating a bookkeeping CDF
  allocation as enablement.
- Once the bound-free channel is selected, the CDF draw and energy-split draw
  are consumed in ARTIS order even when inconsistent input makes the route fail
  closed.
- Each route records whether its target came from the phixs map or from the
  represented upper-ion ground fallback. CPU and GPU count realized direct
  macro-atom activations through the fallback and print
  `[BF-PHIXS-FALLBACK] ... fallback activations=N`.

The existing CDF threshold form (`U*total`, first cumulative `> threshold`,
last-valid roundoff fallback), mapped target, energy-branch sign, and normal
channel/CDF/split random order are unchanged.

## Verification

- `make -B bench_frozen_oracle`: exit 0.
- CPU source syntax check: exit 0.
- `make -B cuda`: exit 0; only the pre-existing
  `g_fgemm_nulo set but never used` warning. The CUDA binary was not run.
- CPU CDF probe:

```text
PASS edge_above=10000 edge_below=10000
actual_nu_p0=0.296625000 expected=0.295737452 z=1.230
failed_route_cdf_draws=1
```

The two edge cases respectively require a route that the old bin centre would
exclude and reject a route that the old bin centre would admit.

### Three-cell OFF oracle

Unset and explicit-zero arms are byte-identical (`cmp=0`) for all three cells.

| cell | eligible SHA-256 | full CSV SHA-256 |
|---|---|---|
| s0 | `beaac19b21bd5b9c0d8c7c81903a1c8c13c8f139ba05cf2e01c414f193678cfa` | `4789f13c89a3bb613e89cb23e836242285aae31bee6065b2631d61324eee1952` |
| s8 | `54f9fafad8da44602a419562a2ef37c9f0c726fdad6780c72e99df436e87d05f` | `a4f1a146a313501a3eaf56232d2d7d3cd4f798425ebd8f426067292edb1538e2` |
| s43 | `b971a0381d4d6c8246979c3bb8d013290d65deac6985898795bee94894380804` | `c48d2619f160191d4a91e37334cf165d2fc312d2263635a281112523e70b72aa` |

Both stderr files were empty.
