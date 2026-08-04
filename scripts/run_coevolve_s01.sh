#!/bin/bash
# COEVOLVE Stage 0+1 gate tests (docs/COEVOLVE_REWIRING_PLAN.md).
# Base plasma = epay27 champion config (commit ff58168) from run_bfdark.sh.
#
#   byteid  R1: run the EXACT champion (THEN_MC, gate OFF) on the NEW binary and
#              on ./lumina_cuda.preCoevolve, diff outputs -> must be byte-identical.
#   coev    Stage-1: same deterministic plasma, THEN_MC replaced by LUMINA_MC_COEVOLVE=1
#              (+ INJECT_SHELL=5). Grep [COEVOLVE-COLOR]: MC field must be BLUER than cs.J.
#
# Env: PKTS / NITER override packet count / iterations (SMOKE: PKTS=2000 NITER=2).
set -e
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
module load cuda/13.0.2 2>/dev/null || true

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-60}
export OMP_PLACES=${OMP_PLACES:-cores} OMP_PROC_BIND=${OMP_PROC_BIND:-close}
export LUMINA_CMF_SOLVE_GPU=1
# Reference dir. Overridable so an atomic-data A/B can change ONLY the data set
# (same ${VAR-default} idiom as LUMINA_NLTE_SKIP_Z on line 28 and
# LUMINA_NLTE_LTE_FLOOR on line 45: substitutes only when UNSET, so every
# existing launcher stays byte-identical).
MODEL=${LUMINA_MODEL_DIR-data/tardis_reference_toy06_19p48d}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export LUMINA_NLTE=1
export LUMINA_NLTE_GREY_ITERS=2 SUPER_LEVELS=1 LUMINA_NLTE_GREY_TAU=2
export LUMINA_MACROATOM_NEUTRAL_E=1 LUMINA_SUPER_CUTOFF=100
export LUMINA_GAMMA_DEP=1 LUMINA_NLTE_ASSEMBLE_GPU=1
export LUMINA_DEPOSITION_FILE=$MODEL/deposition_cmfgen.csv
export LUMINA_PURE_CMFGEN=1 LUMINA_PURE_CMFGEN_ITER=${NITER:-12} LUMINA_CMFGEN_ALI_ITER=8
export LUMINA_BF_OPACITY=1 LUMINA_CMFGEN_SIGMA_BF=$MODEL/cmfgen_sigma_bf.bin
export LUMINA_DYNAMIC_TRANSPROB=1 LUMINA_NLTE_SKIP_Z=${LUMINA_NLTE_SKIP_Z-14} LUMINA_NLTE_START_ITER=2
export LUMINA_NLTE_FLOOR_REG=0 LUMINA_NLTE_INV_CEIL=1e4
export LUMINA_RADEQ_TE=1 LUMINA_RADEQ_COOL_ESCAPE=0
export LUMINA_RADEQ_COOL_NONNEG=0 LUMINA_RADEQ_COOL_NLTE_ONLY=0
export LUMINA_RADEQ_LINE_RESPOND=1 LUMINA_RADEQ_DAMP=0.5
export LUMINA_COUPLED_NEWTON=0 LUMINA_DIP_TRACE=1 LUMINA_CN_DAMP=0.5 LUMINA_COUPLED_NEWTON_SMIN=20 LUMINA_COUPLED_JNU_PHOTOION=${LUMINA_COUPLED_JNU_PHOTOION:-1} LUMINA_FROZENIN=${LUMINA_FROZENIN:-0}
export LUMINA_NLTE_PER_ION_RESCALE=1 LUMINA_COUPLED_JNU_LSTAR=0
export LUMINA_COUPLED_LAMBDA_STAR=1 LUMINA_COUPLED_TDEP=1 LUMINA_RADEQ_LINE_RE=0
export LUMINA_TE_TRAD_RATIO=1.0 LUMINA_LINE_INTERACTION=macroatom
export LUMINA_TAU_BY_ION=1 LUMINA_DIFFUSE_INNER_BC=1 LUMINA_ENERGY_BUDGET=1
# LTE_FLOOR: honour a value set by the caller (same ${VAR-default} idiom as
# LUMINA_NLTE_SKIP_Z on line 28 — substitutes only when the var is UNSET, so every
# existing launcher is byte-identical). It was an unconditional =1, which silently
# voided parity38's floor-off arm: the launcher exported 0, the binary reported 1,
# and the run would have been an exact duplicate of parity37. Caught by the
# post-start envcheck against the binary's own RESOLVED CONFIG block; same failure
# mode as the unconditional unset on line 115 that cost parity34 83 GPU-minutes.
export LUMINA_NLTE_LTE_FLOOR=${LUMINA_NLTE_LTE_FLOOR-1} LUMINA_NLTE_COLL_FIX=1 LUMINA_NLTE_ION_LOCK=1
export LUMINA_NLTE_LOCK_START_ITER=0 LUMINA_NLTE_FALLBACK_TE=1
export LUMINA_CMFGEN_FROZEN_CONT=1 LUMINA_CMFGEN_FROZEN_ALI=60
export LUMINA_MAX_INTERACTIONS=50000 LUMINA_MACROATOM_EWEIGHT=1
export LUMINA_CMFGEN_LINE_EPS_PHYS=1
export LUMINA_RADEQ_FB_RATE=1 LUMINA_HRESP_CLAMP=1.0 LUMINA_TE_STEP_CLAMP=1
export LUMINA_BF_RATE_POPS=1 LUMINA_ETLA_ALLOW_HEAT=1 LUMINA_RADEQ_SIMUL=1
export LUMINA_TRAD_COLOR_FIX=1 LUMINA_J_DAMP=0.5 LUMINA_RADEQ_VR_STD=1
export LUMINA_INNER_BB_SCALE=1.0 LUMINA_CMF_DEP_SOURCE=1
# K6 verdict (parity44/45 twin, V0-V5 2026-07-29): reference VADAT has DIE off for
# every element in this composition, and blanket frozen-in DR double-counts alpha
# (S III->II 2.43x, Fe III->II 3.82x @10kK). Production baseline = all-DR-off; the
# DR_TABLE data stays in the binary behind these masks (override per-run to re-arm).
export LUMINA_TDEP_EQIC=1 LUMINA_FROZENIN_DR=${LUMINA_FROZENIN_DR-0}
export LUMINA_DR_BOOST_BADNELL=${LUMINA_DR_BOOST_BADNELL-0}
export LUMINA_DR_BOOST_NORAD=${LUMINA_DR_BOOST_NORAD-0}
export LUMINA_DR_BOOST_MAZZOTTA=${LUMINA_DR_BOOST_MAZZOTTA-0}
export LUMINA_DR_BOOST_AUTOSTRUCT=${LUMINA_DR_BOOST_AUTOSTRUCT-0}
# epay27 plasma-shaping flags (the champion's deterministic loop A)
export LUMINA_CMF_EPAY=2 LUMINA_CMF_EPAY_SMIN=5 LUMINA_CMF_BF_MILNE=2
export LUMINA_CMF_EPAY_HOTF=0 LUMINA_CMF_EPAY_TAUBIN=10
export LUMINA_CMF_LINERES_JBAR=2 LUMINA_MACROATOM_IDOWN_BETA=1
export LUMINA_SIMUL_NESTED=6 LUMINA_KPACKET=1

PK=${PKTS:-100000}; NI=${NITER:-8}

case "${1:-coev}" in
  byteid)
    # R1: exact champion (THEN_MC on, coevolve gate unset). Run NEW + champion, diff.
    export LUMINA_CMFGEN_THEN_MC=1 LUMINA_MC_INJECT_SHELL=5
    unset LUMINA_MC_COEVOLVE
    for BIN in ./lumina_cuda ./lumina_cuda.preCoevolve; do
      TAG=$(basename "$BIN")
      OUT=logs/coevolve_byteid_$TAG; mkdir -p "$OUT"
      echo "[byteid] $BIN  PK=$PK NI=$NI -> $OUT"
      "$BIN" "$MODEL" "$PK" "$NI" spectrum nlte > "$OUT/stdout.log" 2> "$OUT/stderr.log" || echo "  rc=$?"
      for csv in lumina_spectrum_formal.csv lumina_plasma_state.csv; do
        [ -f "$csv" ] && cp -f "$csv" "$OUT/$csv"
      done
    done
    echo "=== DIFF plasma_state (new vs champion) ==="
    diff logs/coevolve_byteid_lumina_cuda/lumina_plasma_state.csv \
         logs/coevolve_byteid_lumina_cuda.preCoevolve/lumina_plasma_state.csv \
         && echo "BYTE-IDENTICAL plasma_state ✓" || echo "DIFFERS ✗"
    echo "=== DIFF spectrum_formal ==="
    diff logs/coevolve_byteid_lumina_cuda/lumina_spectrum_formal.csv \
         logs/coevolve_byteid_lumina_cuda.preCoevolve/lumina_spectrum_formal.csv \
         && echo "BYTE-IDENTICAL spectrum ✓" || echo "DIFFERS ✗"
    ;;
  coev)
    # Stage-1: co-evolve shadow transport ON (THEN_MC replaced). Color diagnostic.
    unset LUMINA_CMFGEN_THEN_MC
    export LUMINA_MC_COEVOLVE=1 LUMINA_MC_INJECT_SHELL=5 LUMINA_JPROBE=1
    # tag output by injection mode so Stage-1 (inj0) and Stage-2 (inj2) don't clash
    if [ "${LUMINA_MC_COEVOLVE_INJECT:-0}" != "0" ]; then
      OUT=logs/coevolve_s01_inj${LUMINA_MC_COEVOLVE_INJECT}
    else
      OUT=logs/coevolve_s01
    fi
    mkdir -p "$OUT"
    echo "[coev] ./lumina_cuda  PK=$PK NI=$NI -> $OUT"
    ./lumina_cuda "$MODEL" "$PK" "$NI" spectrum nlte > "$OUT/stdout.log" 2> "$OUT/stderr.log" || echo "  rc=$?"
    for csv in lumina_spectrum_formal.csv lumina_plasma_state.csv lumina_coevolve_field.csv; do
      [ -f "$csv" ] && cp -f "$csv" "$OUT/$csv"
    done
    echo "=== [COEVOLVE] setup + [COEVOLVE-COLOR] per-iter blue-tilt + amplitude ==="
    grep -E 'COEVOLVE' "$OUT/stdout.log" || echo "(no COEVOLVE lines — check stderr)"
    ;;
  consume)
    # Consumer: feed the co-evolve MC blue-wing jbar_line into the b_k solve via the
    # beta*J_blue up-rate (mode-3). Target: Lumina b_k lifts 1->(ARTIS-like >>1) = fluorescence.
    # THEN_MC off by default; preset =1 to append the final MC macro-atom spectrum
    # pass (fair MC-emergent observable for ARTIS comparison).
    if [ "${LUMINA_CMFGEN_THEN_MC:-0}" != "1" ]; then unset LUMINA_CMFGEN_THEN_MC; fi
    export LUMINA_MC_COEVOLVE=1 LUMINA_MC_INJECT_SHELL=5 LUMINA_JPROBE=1
    export LUMINA_MC_COEVOLVE_INJECT=2        # HOT volumetric deposition birth (Planck(T_e) SED)
                                              # -> transported blue excess (field-color prereq); w/o
                                              # this the jbar_line is cool and b_k cannot lift.
    export LUMINA_MC_COEVOLVE_CONSUME=1        # download+arm MC jbar_line each iter
    export LUMINA_NLTE_JBAR_POPS=3             # beta*J_blue faithful-Sobolev up-rate (plasma.c:8615)
    export LUMINA_NLTE_ASSEMBLE_GPU=0          # CPU assembly (GPU lacks mode-3)
    export LUMINA_KPACKET=${LUMINA_KPACKET:-0} # default 0: avoid the Div-2 k-packet coherent-UV artifact
                                               # (override to 1 + LUMINA_KPACKET_EXIT=1 for ARTIS single-exit)
    # det-override OFF by default so mode-3 fires; caller may opt in with =1
    # (same idiom as THEN_MC above; the unconditional unset silently voided parity34).
    if [ "${LUMINA_CMF_LINERES_CONSUME:-0}" != "1" ]; then unset LUMINA_CMF_LINERES_CONSUME; fi
    export LUMINA_JBAR_MIN=${LUMINA_JBAR_MIN:-3}          # sparse optical carriers (S II/Ca II/Si II) pump too
    export LUMINA_COEVOLVE_JBAR_DAMP=${LUMINA_COEVOLVE_JBAR_DAMP:-0.5}  # tame far-outer hot-band runaway
    export LUMINA_LEVELPOP_DUMP=1              # dump b_k per shell/ion/level -> compare to ARTIS
    export LUMINA_ION_POP_DUMP=1               # live per-stage ion state (the dump site in the
                                               # pure-CMFGEN end block is on this path; without this
                                               # env the copy below laundered a stale repo-root CSV
                                               # as the run's ionization — parity1-9 fossil incident)
    OUT=logs/coevolve_consume${P0TAG:+_$P0TAG}; mkdir -p "$OUT"
    CBIN="${LUMINA_BIN:-./lumina_cuda}"    # [PARALLEL] own-binary path avoids ./lumina_cuda "Text file busy" contention
    case "$CBIN" in */*) ;; *) CBIN="./$CBIN" ;; esac   # bare filename -> ./name (not in PATH)
    echo "[consume] $CBIN  PK=$PK NI=$NI -> $OUT  (JBAR_POPS=3 CONSUME=1 KPACKET=${LUMINA_KPACKET} CPU-asm)"
    touch "$OUT/.run_start"                    # freshness marker: only products newer than this are copied
    "$CBIN" "$MODEL" "$PK" "$NI" "${LUMINA_SPEC_ARG:-spectrum}" nlte > "$OUT/stdout.log" 2> "$OUT/stderr.log" || echo "  rc=$?"
    for csv in lumina_spectrum_formal.csv lumina_plasma_state.csv lumina_levelpop.csv lumina_ion_pops.csv lumina_jbar_line.csv; do
      if [ -f "$csv" ]; then
        if [ "$csv" -nt "$OUT/.run_start" ]; then cp -f "$csv" "$OUT/$csv"
        else echo "[consume] SKIP stale $csv (older than run start — not this run's product)"; fi
      fi
    done
    echo "=== consumer signals: CONSUME arm + b_k tripwire + T_e stability ==="
    grep -E 'COEVOLVE-CONSUME|S_l/B|tripwire|CMFGEN\] iter' "$OUT/stdout.log" | tail -20 || echo "(none)"
    ;;
esac
echo "DONE"
