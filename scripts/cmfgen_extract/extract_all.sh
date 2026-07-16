#!/bin/bash
# extract_all.sh -- one-call extraction battery for a CMFGEN toy06 epoch.
#
#   usage: extract_all.sh <run_dir> [out_dir] [age_label]
#
# Produces  <out_dir>/cmfgen_toy06_<age>d/  containing:
#     meanopac.csv   per-depth mean opacities + tau scales + v_kms  (ALWAYS; the
#                    file exists every iteration, so works on partial models too)
#     rvtj.csv       r/v/T_e/n_e/opacities/moments + per-element densities
#                    (needs converged model -> RVTJ)
#     jnu.csv        J(lambda) at every depth (depth_index,v_kms,lambda_A,J_nu),
#                    band-limited by default (JNU_LAM_MIN..MAX, default 900-1500 A;
#                    set JNU_ALL=1 for the full CMF grid)  (needs EDDFACTOR; ALWAYS)
#     jnu_full.dat.* raw plt_jh WXY batches + manifest (kept for re-parsing)
#     ionfrac.csv    per-depth ionization fractions (Z,element,ion_stage,depth,
#                    v_kms,log_frac,frac)  (needs converged model -> RVTJ + POP*)
#     EXTRACT_INFO.txt  provenance + which outputs are present/skipped.
#
# Never writes inside <run_dir> (all tools use mirror/symlink workdirs).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
RUN="${1:?run dir}"; RUN="$(cd "$RUN" && pwd)"
OUTBASE="${2:-$HERE}"
# age label from arg or VADAT [SN_AGE]
AGE="${3:-}"
if [ -z "$AGE" ]; then
  AGE=$(awk '/\[SN_AGE\]/{print $1; exit}' "$RUN/VADAT" 2>/dev/null)
fi
[ -z "$AGE" ] && AGE=unknown
OUT="$OUTBASE/cmfgen_toy06_${AGE}d"
mkdir -p "$OUT"
INFO="$OUT/EXTRACT_INFO.txt"
: > "$INFO"
log(){ echo "$@"; echo "$@" >> "$INFO"; }

log "# CMFGEN extraction  $(date '+%Y-%m-%d %H:%M:%S')"
log "run_dir = $RUN"
log "out_dir = $OUT"
log "age     = $AGE d"
log ""

# -- 1. MEANOPAC (always available) -----------------------------------------
VMAP="$OUT/.vmap.txt"
if [ -f "$RUN/MEANOPAC" ]; then
  python3 "$HERE/parse_meanopac.py" "$RUN/MEANOPAC" "$OUT/meanopac.csv" --vmap "$VMAP" \
      && log "[ok] meanopac.csv"
else
  log "[skip] MEANOPAC absent"
fi

# -- 2. RVTJ (converged) ----------------------------------------------------
RVTJ_CSV=""
if [ -f "$RUN/RVTJ" ]; then
  python3 "$HERE/parse_rvtj.py" "$RUN/RVTJ" "$OUT/rvtj.csv" && { RVTJ_CSV="$OUT/rvtj.csv"; log "[ok] rvtj.csv"; }
else
  log "[skip] rvtj.csv -- RVTJ not present (written only at convergence)"
fi

# depth->v source for the J join: prefer rvtj.csv, else meanopac vmap
VOPT=()
if [ -n "$RVTJ_CSV" ]; then VOPT=(--rvtj-csv "$RVTJ_CSV")
elif [ -f "$VMAP" ]; then VOPT=(--vmap "$VMAP"); fi

# -- 3. J_nu (needs EDDFACTOR; available while running) ----------------------
if [ -f "$RUN/EDDFACTOR" ] && [ -f "$RUN/EDDFACTOR_INFO" ]; then
  bash "$HERE/jnu_dump.sh" "$RUN" "$OUT/jnu_full.dat"
  if [ -f "$OUT/jnu_full.dat.manifest" ]; then
    BANDOPT=(--lam-min "${JNU_LAM_MIN:-900}" --lam-max "${JNU_LAM_MAX:-1500}")
    [ "${JNU_ALL:-0}" = 1 ] && BANDOPT=(--all)
    python3 "$HERE/parse_jnu.py" "$OUT/jnu.csv" --manifest "$OUT/jnu_full.dat.manifest" \
        "${VOPT[@]}" "${BANDOPT[@]}" && log "[ok] jnu.csv (${JNU_ALL:+full}${JNU_ALL:-band ${JNU_LAM_MIN:-900}-${JNU_LAM_MAX:-1500} A})"
    # raw WXY batches are large (~250 MB each) and regenerable via jnu_dump.sh;
    # delete by default, keep with EXTRACT_KEEP_RAW=1.
    if [ "${EXTRACT_KEEP_RAW:-0}" != 1 ]; then
      rm -f "$OUT"/jnu_full.dat.b*.dat
      log "[info] raw jnu WXY batches removed (EXTRACT_KEEP_RAW=1 to keep)"
    fi
  else
    log "[warn] jnu dump produced no manifest"
  fi
else
  log "[skip] jnu.csv -- EDDFACTOR absent"
fi

# -- 4. Ionization fractions (converged: RVTJ + POP*) -----------------------
if [ -f "$RUN/RVTJ" ] && ls "$RUN"/POP* >/dev/null 2>&1; then
  bash "$HERE/dispgen_ionfrac.sh" "$RUN" "$OUT/ionfrac"
  if [ -f "$OUT/ionfrac.manifest" ]; then
    python3 "$HERE/parse_ionfrac.py" "$OUT/ionfrac.manifest" "$OUT/ionfrac.csv" \
        ${RVTJ_CSV:+--rvtj-csv "$RVTJ_CSV"} && log "[ok] ionfrac.csv"
  else
    log "[warn] ionfrac produced no manifest"
  fi
else
  log "[skip] ionfrac.csv -- needs converged model (RVTJ + POP<species> files)"
fi

# -- 5. level populations -> Gamma trigger (converged) -----------------------
# Feeds scripts/gamma_photoion_cmp.py (trigger (1) of the integrated-arm decision).
# NB the earlier note here claimed Gamma is readable via "dispgen NETR_<ion>".
# That was WRONG: NETR is a *line* option (maingen.f:825 groups it with
# MOMR/SOBR/EW/LAM/CHIL/TAUL/BETA; its handler maingen.f:1618 computes
# ZNET=1-JBAR*CHIL/ETAL, a bound-bound net rate).  dispgen exposes NO
# photoionization-rate option -- PHOT_*/PLTPHOT_* give cross-sections only, RR_*
# gives recombination.  And DC_*/POP_* cap at 10 levels (maingen_opt_desc.txt:213-216),
# useless for Co III's ~3900.  So we read the POP<SPECIES> files directly and do
# the Gamma integral offline with the same estimator LUMINA uses.
if ls "$RUN"/POP* >/dev/null 2>&1; then
  POPF=$(ls "$RUN"/POPCOB "$RUN"/POPIRON "$RUN"/POPNICK "$RUN"/POPSIL "$RUN"/POPSUL "$RUN"/POPCAL 2>/dev/null)
  if [ -n "$POPF" ]; then
    python3 "$HERE/parse_pops.py" $POPF -o "$OUT/pops.csv" && log "[ok] pops.csv (per-level n_k; feeds gamma_photoion_cmp.py)"
  else
    log "[warn] POP* present but no known species file (POPCOB/POPIRON/...)"
  fi
else
  log "[skip] pops.csv -- POP<SPECIES> written only at LST_ITERATION (convergence)"
fi
log ""
log "# Gamma(Co III) trigger -- run AFTER this battery, from the repo root:"
log "#   JNU_ALL=1 $0 $RUN $OUTBASE        # re-dump J on the FULL grid (EUV thresholds!)"
log "#   python3 scripts/gamma_photoion_cmp.py --Z 27 --ion 2 --shell 0 \\"
log "#       --jnu-csv $OUT/jnu.csv --pop-csv $OUT/pops.csv \\"
log "#       --depth <d(v~4200km/s)> --r-cm <R_outer_cm> --lum-lsun <LSTAR>"
log "#   Decision rule (pre-registered): Gamma_C/G_L >=5 -> gap is real, G-correction"
log "#   is the arm's third component; <=2 -> the '~10x' inference is refuted, re-audit alpha."

log ""
log "# done."
echo "[extract_all] outputs in $OUT"
ls -la "$OUT" | sed 's/^/  /'
