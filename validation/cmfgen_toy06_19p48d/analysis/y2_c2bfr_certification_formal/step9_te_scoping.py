"""Y2 step 9 (NEW in the formal cert; no prelim counterpart).

Mechanises the two report items that the prelim did by hand:

 (A) T_e crosscheck.  LUMINA_C1_SUPERBIN_TEPIN=1 sets T_R := T_e(shell) for the
     pinned coarse bins (lumina_plasma.c:952-956), so TR[iter, shell, cbin=14]
     recovered from lumina_c1_bins.csv IS T_e.  Compare against every
     '[CMFGEN] iter N: T_e[0]=..K T_e[25]=..K T_e[49]=..K' line in stdout.log.

 (B) P7 scoping facts, re-verified on the clean run's stdout.log:
       - LUMINA_TOPSTAGE_IV / LUMINA_NLTE_STAGE4 absent (0 hits)  ->  no top-ion
         drain, so nlte_get_pairs returns the 16 BASE pairs
         (lumina_plasma.c:7268-7273)
       - the '[NLTE-GEMM] init: 16 pairs, 10340 phot levels' banner
       - the 16 lower-ion level counts read out of the '[NLTE] Total NLTE levels'
         slot table must sum to exactly 10340
"""
import os
import re
import numpy as np
import pandas as pd
import y2_common as Y

TR = np.load(os.path.join(Y.OUT, "_cache_TR.npy"))
ni = TR.shape[0]
log = open(Y.STDOUT_LOG, "r", errors="replace").read()
lines = log.splitlines()

# ---------------------------------------------------------------- (A) T_e
pat = re.compile(r"^\[CMFGEN\] iter\s+(\d+): T_e\[0\]=(\d+)K T_e\[25\]=(\d+)K "
                 r"T_e\[49\]=(\d+)K")
rows = []
for ln_no, ln in enumerate(lines, 1):
    m = pat.match(ln)
    if not m:
        continue
    it = int(m.group(1))
    if it >= ni:
        continue
    for sh, gi in ((0, 2), (25, 3), (49, 4)):
        rows.append(dict(iter=it, shell=sh, line_no=ln_no,
                         T_e_stdout=float(m.group(gi)),
                         T_e_C1bin14=float(TR[it, sh, 14])))
te = pd.DataFrame(rows)
# the pin only exists where coarse bin 14 was non-empty; where the MC put no
# energy in 728.8-905.6 A the C1 bin is 'empty' and T_R stays 0 (no T_e to read)
mode = np.load(os.path.join(Y.OUT, "_cache_mode.npy"), allow_pickle=True).astype(str)
te["c1_mode_bin14"] = [mode[r.iter, r.shell, 14] for r in te.itertuples()]
te["pinned"] = te.c1_mode_bin14 == "pin"
te["abs_diff_K"] = (te.T_e_stdout - te.T_e_C1bin14).abs()
te["match_to_1K"] = te.abs_diff_K < 1.0
te.to_csv(os.path.join(Y.OUT, "y2_te_crosscheck.csv"), index=False)
print("=== (A) T_e crosscheck: C1 pinned bin 14 vs stdout.log [CMFGEN] iter lines ===")
print(te.to_string(index=False))
p = te[te.pinned]
print(f"\n  rows compared                       : {len(te)}  (iters "
      f"{[int(v) for v in sorted(te['iter'].unique())]})")
print(f"  rows where C1 bin 14 is actually pin: {len(p)}")
print(f"  max |diff| [K] over pinned rows     : {p.abs_diff_K.max():.6f}  "
      f"(stdout prints integer K -> +-0.5 K is exact)")
print(f"  mismatches (>= 1 K) among pinned    : {int((~p.match_to_1K).sum())}")
print(f"  non-pinned rows (no T_e recoverable): {int((~te.pinned).sum())}  "
      f"{[(int(r.iter), int(r.shell), r.c1_mode_bin14) for r in te[~te.pinned].itertuples()]}")
nonpin = [(i, s) for i in range(mode.shape[0]) for s in range(mode.shape[1])
          if mode[i, s, 14] != "pin"]
print(f"  (whole grid) iter x shell cells with bin14 != pin: {len(nonpin)} / "
      f"{mode.shape[0]*mode.shape[1]}  -> {nonpin}")
print(f"  FINAL ITER {Y.IT_FINAL}: all 50 shells pinned = "
      f"{bool((mode[-1, :, 14] == 'pin').all())}")

# ---------------------------------------------------------------- (B) scoping
print("\n=== (B) P7 scoping re-verification on the clean stdout.log ===")
for tok in ("TOPSTAGE_IV", "STAGE4", "C2_MATRIX_BF", "C2_BFR_DUMP",
            "C1_SUPERBIN_TEPIN", "C1_DEGEN_FALLBACK"):
    hits = [i + 1 for i, ln in enumerate(lines) if tok in ln]
    print(f"  {tok:20s}: {len(hits):3d} hits" +
          (f"  lines {hits[:6]}" if hits else "   (ABSENT)"))

ban = [(i + 1, ln.strip()) for i, ln in enumerate(lines)
       if "[NLTE-GEMM] init:" in ln]
print("\n  GEMM init banner:")
for n, ln in ban:
    print(f"    stdout.log:{n}: {ln}")
mb = re.search(r"init: (\d+) pairs, (\d+) phot levels", ban[0][1]) if ban else None
n_pairs, n_phot = (int(mb.group(1)), int(mb.group(2))) if mb else (-1, -1)

# slot table -> per-ion level counts
slot = {}
for ln in lines:
    m = re.match(r"\s+Z=(\d+) ion=(\d+): (\d+) levels\s*$", ln)
    if m:
        key = (int(m.group(1)), int(m.group(2)))
        if key not in slot:
            slot[key] = int(m.group(3))
LOWER = [(14, 1, "Si II"), (20, 1, "Ca II"), (26, 1, "Fe II"), (16, 1, "S II"),
         (27, 1, "Co II"), (28, 1, "Ni II"), (6, 1, "C II"), (12, 1, "Mg II"),
         (22, 1, "Ti II"), (24, 1, "Cr II"), (13, 1, "Al II"), (21, 1, "Sc II"),
         (23, 1, "V II"), (25, 1, "Mn II"), (8, 0, "O I"), (8, 1, "O II")]
tot = 0
print("\n  base-pair LOWER ions (lumina_plasma.c:7268-7273; only the lower member\n"
      "  enters the R_bf loop, lumina_plasma.c:14564) and their slot-table counts:")
for Z, io, nm in LOWER:
    n = slot.get((Z, io), None)
    tot += n or 0
    print(f"    {nm:7s} Z={Z:2d} ion={io}: {n}")
print(f"  sum of the 16 lower-ion level counts = {tot}")
print(f"  banner phot levels                   = {n_phot}   "
      f"MATCH={tot == n_phot}")
print(f"  banner pairs                         = {n_pairs}  MATCH={n_pairs == 16}")

REF = [(14, 2, "Si III"), (16, 2, "S III"), (26, 2, "Fe III"), (26, 3, "Fe IV"),
       (27, 2, "Co III")]
print("\n  reference-only ions (NOT lower members -> receive NO matrix R_bf):")
for Z, io, nm in REF:
    print(f"    {nm:7s} Z={Z:2d} ion={io}: in-lower-set="
          f"{(Z, io) in [(a, b) for a, b, _ in LOWER]}  "
          f"slot levels={slot.get((Z, io))}")

pd.DataFrame([dict(n_pairs_banner=n_pairs, n_phot_banner=n_phot,
                   sum_lower_levels=tot, match=bool(tot == n_phot),
                   n_TOPSTAGE_IV_hits=sum("TOPSTAGE_IV" in l for l in lines),
                   n_STAGE4_hits=sum("STAGE4" in l for l in lines),
                   n_C2_MATRIX_BF_hits=sum("C2_MATRIX_BF" in l for l in lines))
              ]).to_csv(os.path.join(Y.OUT, "y2_scoping.csv"), index=False)
