#!/bin/bash
# dispgen_ionfrac.sh -- drive dispgen.exe to dump per-depth ionization fractions
# (log10 N_ion/N_species) for each modelled element, to ASCII WXY files.
#
#   usage: dispgen_ionfrac.sh <run_dir> <out_prefix> [workdir]
#
# NEEDS A CONVERGED MODEL: dispgen reads RVTJ + the per-species POP<SPECIES> files
# (written only in the final LST_ITERATION), plus the full atomic-data links.  It
# therefore runs in a <workdir> that MIRRORS <run_dir> (every entry symlinked, so
# atomic links + RVTJ + POP files resolve) and never writes into <run_dir>.
#
# Verified live (2026-07-15): dispgen loads the atomic env and reaches the
# "Structure file [RVTJ]:" prompt; the IF_ dump path itself requires the
# converged RVTJ/POP files, so confirm the prompt count at the first converged
# model.  GRAMON WXY output format is identical to plt_jh's (verified).
#
# Per element the driven sequence is:
#   RVTJ            (structure file)
#   1000            (photoionization smoothing km/s, default)
#   XN              (abscissa = depth index 1..ND)
#   IF_<SPECIES>    (one curve per ionization stage: log10 N_ion/N_species)
#   T               ('Species fraction?' -> yes)
#   GR / WXY <f> 0 0 -1 / E / EX
set -u
RUN="${1:?run dir}"; PREFIX="${2:?out prefix}"
WORK="${3:-$(dirname "$PREFIX")/.disp_work_$$}"
CMF=/gpfs/kjhan/cmfgen_src/cur_cmf
DISP="$CMF/exe/dispgen.exe"
RUN="$(cd "$RUN" && pwd)"
mkdir -p "$(dirname "$PREFIX")"
PREFIX="$(cd "$(dirname "$PREFIX")" && pwd)/$(basename "$PREFIX")"
mkdir -p "$WORK"; WORK="$(cd "$WORK" && pwd)"

# mirror the run dir (symlinks) so the atomic environment + outputs are present
for e in "$RUN"/*; do ln -sf "$e" "$WORK/$(basename "$e")"; done
ln -sf "$CMF/txt_files/maingen_options.txt"  "$WORK/MAINGEN_OPTIONS"  2>/dev/null
ln -sf "$CMF/txt_files/maingen_opt_desc.txt" "$WORK/MAINGEN_OPT_DESC" 2>/dev/null

# element species-token  Z  (lowest stage is III for all toy06 species)
ELEMS=("SIL 14" "SUL 16" "CAL 20" "IRON 26" "COB 27" "NICK 28")
MAN="${PREFIX}.manifest"; : > "$MAN"
cd "$WORK"
echo "[dispgen_ionfrac] $RUN -> ${PREFIX}.<elem>.ifrac  work=$WORK"
RC=0
for pair in "${ELEMS[@]}"; do
  set -- $pair; SP="$1"; Z="$2"
  of="${SP}.ifrac"
  {
    echo RVTJ
    echo 1000
    echo XN
    echo "IF_${SP}"
    echo T
    echo GR; echo WXY; echo "$of"; echo 0; echo 0; echo -1; echo E; echo EX
  } | "$DISP" > "disp_${SP}.log" 2>&1
  if [ -s "$WORK/$of" ]; then
    dest="${PREFIX}.${SP}.ifrac"
    mv "$WORK/$of" "$dest"
    echo "$dest $SP $Z" >> "$MAN"
    echo "[dispgen_ionfrac]   $SP (Z=$Z): $(head -1 "$dest") stages -> $(basename "$dest")"
  else
    echo "[dispgen_ionfrac]   WARN no ion-frac dump for $SP (see $WORK/disp_${SP}.log)"
    RC=3
  fi
done
cp "$WORK"/disp_*.log "$(dirname "$PREFIX")/" 2>/dev/null
cd - >/dev/null
[ "${DISP_KEEP_WORK:-0}" = 1 ] || rm -rf "$WORK"
echo "[dispgen_ionfrac] manifest: $MAN"
exit $RC
