#!/bin/bash
# jnu_dump.sh -- drive plt_jh.exe non-interactively to dump J(lambda) at EVERY
# model depth from a CMFGEN run's EDDFACTOR file.
#
#   usage: jnu_dump.sh <run_dir> <out_prefix> [workdir]
#
# GRAMON stores at most MAX_PLTS=50 curves (pgplt/mod_curve_data.f:10), so J is
# dumped in batches of <=45 depths.  Produces:
#     <out_prefix>.bNNN.dat   one WXY table per depth batch
#     <out_prefix>.manifest   lines "<batchfile> <first_depth>" for parse_jnu
#
# Runs inside <workdir> with read-only SHORT-name symlinks to <run_dir> inputs,
# so <run_dir> (e.g. the untouchable live toy06_2d) is never modified.
# plt_jh reads stdin (gen_in.f); PGPLOT is a no-op stub (no X display).
# Sequence per batch: EDDFACTOR / {RVTJ|NULL} / XU / Ang / (JD d)* / GR / WXY
#   file 0 0 -1 / E / EX.  Verified live against toy06_2d EDDFACTOR (2026-07-15).
set -u
RUN="${1:?run dir}"; PREFIX="${2:?out prefix}"
WORK="${3:-$(dirname "$PREFIX")/.jnu_work_$$}"
BATCH="${JNU_BATCH:-45}"
CMF=/gpfs/kjhan/cmfgen_src/cur_cmf
PLT="$CMF/exe/plt_jh.exe"
RUN="$(cd "$RUN" && pwd)"
mkdir -p "$(dirname "$PREFIX")"
PREFIX="$(cd "$(dirname "$PREFIX")" && pwd)/$(basename "$PREFIX")"
mkdir -p "$WORK"; WORK="$(cd "$WORK" && pwd)"

ND=$(awk 'NR==3{print $1; exit}' "$RUN/EDDFACTOR_INFO" 2>/dev/null)
[ -z "$ND" ] && ND=$(awk '/\[ND\]/{print $1; exit}' "$RUN/MODEL_SPEC" 2>/dev/null)
[ -z "$ND" ] && { echo "cannot determine ND"; exit 1; }

cd "$WORK"
ln -sf "$RUN/EDDFACTOR"      EDDFACTOR
ln -sf "$RUN/EDDFACTOR_INFO" EDDFACTOR_INFO
ln -sf "$CMF/txt_files/plt_jh_options.txt"  PLT_JH_OPTIONS
ln -sf "$CMF/txt_files/plt_jh_opt_desc.txt" PLT_JH_OPT_DESC
STRUCT=NULL
[ -f "$RUN/RVTJ" ] && { ln -sf "$RUN/RVTJ" RVTJ; STRUCT=RVTJ; }

MAN="${PREFIX}.manifest"; : > "$MAN"
echo "[jnu_dump] $RUN  ND=$ND  batch<=$BATCH  struct=$STRUCT  -> ${PREFIX}.bNNN.dat"
k=0; start=1; RC=0
while [ "$start" -le "$ND" ]; do
  end=$(( start + BATCH - 1 )); [ "$end" -gt "$ND" ] && end="$ND"
  k=$(( k + 1 )); bf=$(printf "jnu.b%03d.dat" "$k")
  {
    echo EDDFACTOR; echo "$STRUCT"; echo XU; echo Ang
    for d in $(seq "$start" "$end"); do echo JD; echo "$d"; done
    echo GR; echo WXY; echo "$bf"; echo 0; echo 0; echo -1; echo E; echo EX
  } | "$PLT" > "pltjh.b${k}.log" 2>&1
  if [ -s "$WORK/$bf" ]; then
    dest=$(printf "%s.b%03d.dat" "$PREFIX" "$k")
    mv "$WORK/$bf" "$dest"
    echo "$dest $start" >> "$MAN"
    echo "[jnu_dump]   batch $k depths $start-$end -> $(basename "$dest")"
  else
    echo "[jnu_dump]   ERROR batch $k (depths $start-$end); see $WORK/pltjh.b${k}.log"
    RC=2
  fi
  start=$(( end + 1 ))
done
cp "$WORK"/pltjh.b*.log "$(dirname "$PREFIX")/" 2>/dev/null
cd - >/dev/null
[ "${JNU_KEEP_WORK:-0}" = 1 ] || rm -rf "$WORK"
echo "[jnu_dump] manifest: $MAN"
exit $RC
