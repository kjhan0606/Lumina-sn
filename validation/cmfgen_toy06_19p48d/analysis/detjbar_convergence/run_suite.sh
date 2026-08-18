#!/bin/bash
# Reproduction driver.  CPU only, no GPU, no production run.
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-12}
B=./detjbar_conv
# --- similarity / size scan at the production-equivalent n_ali (=20000*r) ----
$B --nb   800 --depositall 1 --lagconv 4000 --out n800_all   > log_n800_all.txt   2>&1
$B --nb  3200 --depositall 1 --lagconv 4000 --out n3200_all  > log_n3200_all.txt  2>&1
$B --nb 12800 --depositall 1                --out n12800_all > log_n12800_all.txt 2>&1
# --- n_ali scan at NB=3200 (production-equivalent = 128) --------------------
for A in 32 64 128 256 512 1024; do
  $B --nb 3200 --depositall 1 --nali $A --out n3200_a$A > log_n3200_a$A.txt 2>&1
done
# --- opacity-composition bracket (r-similar subsample; blanket too thin) ----
$B --nb 3200 --depositall 0 --chimul 1  --out n3200_sub   > log_n3200_sub.txt  2>&1
# --- continuum-only (no forest): isolates the pure advection lag ------------
$B --nb 3200 --depositall 1 --chimul 0  --out n3200_cont  > log_n3200_cont.txt 2>&1
# --- chi_abs sensitivity ---------------------------------------------------
$B --nb 3200 --depositall 1 --fabs 0.0  --out n3200_fa0   > log_n3200_fa0.txt  2>&1
$B --nb 3200 --depositall 1 --fabs 3.0  --out n3200_fa3   > log_n3200_fa3.txt  2>&1
echo SUITE1_DONE

# ===========================================================================
# DISCRETISATION suite (2026-07-29; see DISCRETIZATION.md).  Adds the EXACT
# drifting-characteristic path and the window-truncation A/B.  Run with
#   OMP_NUM_THREADS=28 ./run_suite.sh disc
# Needs ~45 GB peak (the NB=498721 runs) and ~2.5 h wall on 28 cores.
# ===========================================================================
[ "$1" = "disc" ] || exit 0
python3 make_lines_bin.py --blue          # -> lines4.bin (adds 900-1000 A)
python3 make_lines_bin.py --blue2         # -> lines5.bin (adds 800-1000 A)

# (V1) REF/LAG must be untouched by the extension: rerun a legacy invocation and
#      diff against the archived detjbar_n800_all.csv.gz of the SAME binary.
$B --nb 800 --depositall 1 --lagconv 4000 --out REGF800 > log_REGF800.txt 2>&1
# (V2) the two EXACT implementations must agree (bit-identical at low beta)
for M in 1 2; do
  $B --nb 3200 --depositall 1 --lag 0 --exact $M --out EXV3200m$M > log_EXV3200m$M.txt 2>&1
  $B --nb 51200 --depositall 1 --ref 0 --lag 0 --exact $M --out EXV51200m$M > log_EXV51200m$M.txt 2>&1
done
python3 -c "
import pandas as pd,numpy as np
for n in ['3200','51200']:
    a=pd.read_csv('detjbar_EXV%sm1.csv'%n);b=pd.read_csv('detjbar_EXV%sm2.csv'%n)
    print('NB=%-6s EXACT mode1-vs-mode2 max rel: %.3e'%(n,np.abs(a.jbar_exact/b.jbar_exact-1).max()))"

# (a) frequency-Courant / operator-split error: similarity ladder + production
for N in 800 3200 12800 51200; do
  $B --nb $N --depositall 1 --lag 0 --exact 1 --transprobe 1 --out LAD$N > log_LAD$N.txt 2>&1
done
$B --nb 498721 --depositall 1 --lag 0 --exact 1 --transprobe 1 --out PROD > log_PROD.txt 2>&1
# continuum-only arm: separates the smooth-gradient part from the line-comb part
for N in 3200 12800 51200; do
  $B --nb $N --depositall 1 --lag 0 --exact 1 --chimul 0 --out CONT$N > log_CONT$N.txt 2>&1
done

# (b) 1000 A window truncation: narrow (production) vs wide (900 A) reference.
#     Same lines file both times; --outsel 1 keeps line_idx run-invariant.
for NT in "12800 WIN12800" "51200 WIN51200"; do
  set -- $NT; N=$1; T=$2
  for L in 1000 900; do
    $B --nb $N --lines lines4.bin --lamlo $L --depositall 1 --lag 0 --exact 0 \
       --outsel 1 --outstride 40 --out ${T}_$L > log_${T}_$L.txt 2>&1
  done
  python3 stats.py --pair detjbar_${T}_1000.csv detjbar_${T}_900.csv
done
# (b) at PRODUCTION resolution.  --lamhi 1400 drops the reddest bins, which is
# LOSSLESS for everything kept (the solve is strictly one-way blue->red) and
# makes this 4x cheaper.  Losslessness re-checked by the TR12800 pair below.
$B --nb 12800 --lines lines4.bin --lamhi 1400 --lamlo 1000 --depositall 1 --lag 0 \
   --exact 0 --outsel 1 --outstride 40 --out TR12800 > log_TR12800.txt 2>&1
python3 -c "
import pandas as pd,numpy as np
a=pd.read_csv('detjbar_TR12800.csv');b=pd.read_csv('detjbar_WIN12800_1000.csv')
m=a.merge(b,on=['line_idx','shell'],suffixes=('_t','_f')); m=m[m.lambda_A_t<1390]
print('red-truncation lossless check: median %.1e  p90 %.1e'%tuple(
  np.percentile(np.abs(m.jbar_ref_t/m.jbar_ref_f-1),[50,90])))"
for L in 1000 900; do
  $B --nb 498721 --lines lines4.bin --lamhi 1400 --lamlo $L --depositall 1 --lag 0 \
     --exact 0 --outsel 1 --outstride 4 --out WINPROD_$L > log_WINPROD_$L.txt 2>&1
done
python3 stats.py --pair detjbar_WINPROD_1000.csv detjbar_WINPROD_900.csv
# is the 900 A reference converged?  3-point ladder 1000/900/800 (lines5.bin)
for L in 1000 900 800; do
  $B --nb 12800 --lines lines5.bin --lamlo $L --depositall 1 --lag 0 --exact 0 \
     --outsel 1 --outstride 40 --out W3_$L > log_W3_$L.txt 2>&1
done
python3 stats.py --pair detjbar_W3_900.csv detjbar_W3_800.csv   # must be ~0
python3 stats.py detjbar_PROD.csv
echo DISC_DONE
